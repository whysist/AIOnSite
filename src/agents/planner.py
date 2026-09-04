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

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator

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
    '    {"id": "step_1", "agent": "researcher", "description": "...", "depends_on": []}\n'
    "  ]\n"
    "}\n"
    "Rules: 2-6 steps; ids are unique; depends_on references earlier ids only; "
    "end with a summarizer step."
)


class PlanStep(BaseModel):
    id: str
    agent: str = "custom"
    description: str = ""
    depends_on: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)

    @field_validator("agent")
    @classmethod
    def _known_agent(cls, v: str) -> str:
        v = (v or "custom").strip().lower()
        if v not in _KNOWN_AGENTS:
            _log.info("planner_unknown_agent", agent=v)
            return "custom"
        return v

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

    @field_validator("steps")
    @classmethod
    def _validate_steps(cls, steps: list[PlanStep]) -> list[PlanStep]:
        if not steps:
            raise ValueError("plan has no steps")
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


class Planner:
    def __init__(self, llm: BaseLLM, *, max_attempts: int = 2) -> None:
        self._llm = llm
        self._max_attempts = max_attempts

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
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                _log.warning("plan_attempt_failed", attempt=attempt, error=last_error)

        _log.error("plan_fallback_used", error=last_error)
        return self.fallback_plan(task)

    @staticmethod
    def fallback_plan(task: str) -> Plan:
        """Deterministic 3-step plan used when the model cannot produce one."""
        goal = task.strip().splitlines()[0][:200] if task.strip() else "Complete the task"
        return Plan(
            goal=goal,
            steps=[
                PlanStep(id="step_1", agent="researcher",
                         description=f"Gather information relevant to: {goal}", depends_on=[]),
                PlanStep(id="step_2", agent="analyst",
                         description="Analyse the gathered information and compute results.",
                         depends_on=["step_1"]),
                PlanStep(id="step_3", agent="summarizer",
                         description="Produce the final verified answer.", depends_on=["step_2"]),
            ],
        )
