"""Planner agent: natural-language task -> validated structured plan.

The planner asks the model for JSON, then validates it hard with Pydantic
and graph checks.  LLM JSON is never trusted:

* wrong / missing fields          -> validation error -> one bounded retry
* unknown agent name              -> coerced to ``custom`` (logged)
* dangling / self / cyclic deps   -> rejected
* duplicate step ids              -> rejected

If validation still fails after the retry budget, a deterministic
fallback plan is returned so the pipeline can always run.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from ..core.exceptions import ProviderConnectionError, ProviderTimeoutError
from ..core.logging import get_logger
from ..llm.base import BaseLLM

_log = get_logger("planner")

_KNOWN_AGENTS = {"researcher", "analyst", "executor", "summarizer", "verifier", "custom"}

_SYSTEM = (
    "You are the planning module of a local, sovereign agentic inspection "
    "workbench. Decompose the user's task into a minimal ordered set of steps. "
    "Each step is handled by one agent role: researcher (gather facts), analyst "
    "(reason/compute), executor (perform an action/tool), summarizer (final "
    "synthesis). Respond with ONLY this JSON:\n"
    "{\n"
    '  "goal": "<one sentence>",\n'
    '  "steps": [\n'
    '    {"id": "step_1", "agent": "researcher", "description": "...", '
    '"depends_on": [], "criticality": "required"}\n'
    "  ],\n"
    '  "requirements": [\n'
    '    {"description": "<one short sentence: a concrete thing the final '
    'answer must contain or have done>", "step_ids": ["step_1"], '
    '"needs_evidence": true}\n'
    "  ]\n"
    "}\n"
    "Rules: 2-6 steps; ids are unique; depends_on references earlier ids only; "
    "end with a summarizer step. List 1-6 requirements; they are the checklist "
    "used to judge whether the task was actually completed -- so be concrete "
    "(e.g. one requirement per fact that must be gathered, calculation that "
    "must be performed, or document that must be consulted), and for every "
    'requirement, "step_ids" MUST list the id(s) of the step(s) above that are '
    "responsible for satisfying it (verification checks that step's outcome "
    "and evidence directly -- a requirement with no step_ids can only be "
    "checked by a much weaker heuristic, so leave step_ids empty only for a "
    "requirement that is genuinely not the responsibility of any single "
    'step). Set "needs_evidence" to false only for a requirement about '
    "reasoning, formatting or producing the final answer itself (e.g. "
    '"state a recommendation") rather than about a specific retrieved or '
    "computed fact -- it defaults to true. Set each step's "
    '"criticality" to "optional" if the task can still be answered without '
    'it succeeding, "critical" if the whole task is meaningless without it, '
    'or "required" (the default) otherwise -- do not mark every step '
    "required; steps that only gather background context are usually "
    "optional."
)


_KNOWN_CRITICALITIES = {"optional", "required", "critical"}


class RequirementSpec(BaseModel):
    """One planner-authored completion-checklist entry.

    Carries the requirement -> step link that makes verification able to
    check execution outcomes and evidence directly, instead of falling back
    to comparing this description's wording against the final answer's
    wording (see ``verification/verifier.py``).
    """

    description: str
    # Step id(s) responsible for satisfying this requirement. Validated
    # against the plan's actual step ids by ``Plan._validate_requirements``
    # below -- an unknown id is treated the same as an unknown ``depends_on``
    # id (rejected, triggering the planner's bounded retry).
    step_ids: list[str] = Field(default_factory=list)
    needs_evidence: bool = True

    @model_validator(mode="before")
    @classmethod
    def _coerce_plain_string(cls, data: Any) -> Any:
        """Accept a bare string requirement (the pre-traceability shape,
        still the easiest fallback for a weak/uncooperative local model)
        and normalise it into ``{"description": ...}``."""
        if isinstance(data, str):
            return {"description": data}
        return data


class PlanStep(BaseModel):
    id: str
    agent: str = "custom"
    description: str = ""
    depends_on: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    # How much this step's failure should matter -- see
    # ``src.pipeline.models.Criticality``. Defaults to "required": most
    # steps in a plan exist because the task needs them.
    criticality: str = "required"

    @field_validator("agent")
    @classmethod
    def _known_agent(cls, v: str) -> str:
        v = (v or "custom").strip().lower()
        if v not in _KNOWN_AGENTS:
            _log.info("planner_unknown_agent", agent=v)
            return "custom"
        return v

    @field_validator("criticality")
    @classmethod
    def _known_criticality(cls, v: str) -> str:
        v = (v or "required").strip().lower()
        return v if v in _KNOWN_CRITICALITIES else "required"

    @field_validator("id")
    @classmethod
    def _clean_id(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("step id must be non-empty")
        return v


class Plan(BaseModel):
    goal: str
    steps: list[PlanStep]
    # Requirements the planner LLM was asked to list, each linked to the
    # step(s) responsible for satisfying it (see ``RequirementSpec``);
    # consumed by ``src/pipeline/requirements.extract_requirements`` to
    # build the completion checklist. Left empty for the deterministic
    # fallback plan (which has no model to ask), in which case that
    # function derives a baseline checklist from the steps instead.
    requirements: list[RequirementSpec] = Field(default_factory=list)
    # Set when this plan is the deterministic fallback used because the
    # model could not produce a valid plan after all retries -- surfaced to
    # the orchestrator/audit trail as a degraded-planning signal rather than
    # being indistinguishable from a normal model-generated plan.
    degraded: bool = False
    degraded_reason: str | None = None

    @field_validator("steps")
    @classmethod
    def _validate_steps(cls, steps: list[PlanStep]) -> list[PlanStep]:
        if not steps:
            raise ValueError("plan has no steps")
        if not 2 <= len(steps) <= 6:
            # The prompt asks for 2-6 steps as a target, not a hard limit --
            # a model that ignores it still produces a usable plan, so this
            # is logged (visibly, unlike before) rather than rejected, which
            # would only burn a retry attempt for no correctness benefit.
            _log.info("plan_step_count_outside_target_range", count=len(steps))
        ids = [s.id for s in steps]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate step ids")
        seen: set[str] = set()
        for step in steps:
            for dep in step.depends_on:
                if dep == step.id:
                    raise ValueError(f"step {step.id!r} depends on itself")
                if dep not in ids:
                    raise ValueError(f"step {step.id!r} depends on unknown step {dep!r}")
                if dep not in seen:
                    raise ValueError(
                        f"step {step.id!r} depends on {dep!r} which is not defined earlier"
                    )
            seen.add(step.id)
        return steps

    @model_validator(mode="after")
    def _validate_requirement_step_ids(self) -> Plan:
        """A requirement's ``step_ids`` must reference real steps.

        Mirrors the ``depends_on`` check above: an LLM-invented step id here
        would otherwise silently produce a requirement verification can
        never resolve via the primary (execution/evidence) path, quietly
        degrading it to the secondary text-overlap heuristic with no signal
        that anything was wrong. Rejecting it instead routes through the
        planner's existing bounded-retry-then-fallback machinery.
        """
        known_ids = {s.id for s in self.steps}
        for spec in self.requirements:
            for step_id in spec.step_ids:
                if step_id not in known_ids:
                    raise ValueError(
                        f"requirement {spec.description!r} references unknown step {step_id!r}"
                    )
        return self


class Planner:
    def __init__(
        self, llm: BaseLLM, *, max_attempts: int = 2, backoff_seconds: float = 0.5
    ) -> None:
        self._llm = llm
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds

    async def plan(self, task: str, *, context: dict[str, Any] | None = None) -> Plan:
        prompt = task if not context else f"{task}\n\nContext:\n{json.dumps(context, default=str)}"
        last_error: str | None = None

        for attempt in range(1, self._max_attempts + 1):
            user = prompt
            if last_error:
                user += (
                    f"\n\nYour previous response was invalid: {last_error}\n"
                    "Return corrected JSON only."
                )
            try:
                raw = await self._llm.generate_json(
                    [BaseLLM.system(_SYSTEM), BaseLLM.user(user)],
                    metadata={"aionsite_kind": "plan"},
                    temperature=0.0,
                )
                plan = Plan.model_validate(raw)
                _log.info("plan_created", attempt=attempt, steps=len(plan.steps))
                return plan
            except (ProviderConnectionError, ProviderTimeoutError) as exc:
                # Infrastructure failure, not a bad model response: retrying
                # the exact same request instantly is pointless (and, for a
                # connection refusal, indistinguishable from a busy-loop), so
                # back off, and don't prepend "your previous response was
                # invalid" -- there was no response to critique.
                last_error = str(exc)
                _log.warning(
                    "plan_attempt_failed_infra",
                    attempt=attempt, error=last_error, kind=type(exc).__name__,
                )
                if attempt < self._max_attempts:
                    await asyncio.sleep(self._backoff_seconds * attempt)
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                _log.warning("plan_attempt_failed", attempt=attempt, error=last_error)

        _log.error("plan_fallback_used", error=last_error)
        fallback = self.fallback_plan(task)
        fallback.degraded = True
        fallback.degraded_reason = last_error
        return fallback

    @staticmethod
    def lookup_plan(task: str, equipment_id: str) -> Plan:
        """Deterministic 2-step plan for a classified ``SIMPLE_LOOKUP`` task.

        Used by the orchestrator instead of a full ``plan()`` LLM call when
        ``agents.classifier.classify_task`` is confident the task is a
        direct equipment-identity question (see
        ``orchestrator.py::Orchestrator._run``). Skips only the planning
        LLM call -- the resulting ``Plan`` still goes through the same
        ``build_pipeline``/``PipelineExecutor``/``ResultVerifier`` path as
        any other plan, so DAG validation, failure propagation and
        verification are unchanged.

        Deliberately requests only ``query_type="info"`` in the step
        description (not "all"): a plain identity question does not need
        limits/inspection/maintenance pulled in by default, which is the
        planner-scope-creep pattern that produced the false verification
        failure this fast path is designed to avoid.
        """
        goal = task.strip().splitlines()[0][:200] if task.strip() else f"Look up {equipment_id}"
        return Plan(
            goal=goal,
            steps=[
                PlanStep(
                    id="step_1", agent="researcher",
                    description=(
                        f"Retrieve equipment identity information for {equipment_id} "
                        f"(equipment_lookup, query_type='info') needed to answer: {task}"
                    ),
                    depends_on=[], criticality="required",
                    tools=["equipment_lookup"],
                ),
                PlanStep(
                    id="step_2", agent="summarizer",
                    description=(
                        f"Answer the user's question about {equipment_id} using only the "
                        "retrieved equipment evidence. Do not introduce operating limits, "
                        "inspection or maintenance detail unless the question asked for it."
                    ),
                    depends_on=["step_1"], criticality="required",
                ),
            ],
            requirements=[
                RequirementSpec(
                    description=f"Report {equipment_id}'s equipment identity/information",
                    step_ids=["step_1"], needs_evidence=True,
                ),
            ],
        )

    @staticmethod
    def fallback_plan(task: str) -> Plan:
        """Deterministic 3-step plan used when the model cannot produce one."""
        goal = task.strip().splitlines()[0][:200] if task.strip() else "Complete the task"
        return Plan(
            goal=goal,
            steps=[
                # Background gathering is useful but not load-bearing: the
                # task text itself usually already carries the values the
                # analyst needs, so a failure here should degrade the node,
                # not block the whole run (see Criticality in
                # ``src/pipeline/models.py``).
                PlanStep(id="step_1", agent="researcher",
                         description=f"Gather information relevant to: {goal}", depends_on=[],
                         criticality="optional"),
                PlanStep(id="step_2", agent="analyst",
                         description="Analyse the gathered information and compute results.",
                         depends_on=["step_1"], criticality="required"),
                PlanStep(id="step_3", agent="summarizer",
                         description="Produce the final verified answer.", depends_on=["step_2"],
                         criticality="required"),
            ],
        )
