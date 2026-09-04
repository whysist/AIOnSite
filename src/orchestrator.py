"""Top-level orchestration service.

Wires the pieces together for one task:

    task
      -> policy check (sovereign / confidential)
      -> Planner            (LLM -> validated Plan)
      -> build_pipeline     (Plan -> DAG)
      -> PipelineExecutor   (ModelRouter picks provider/model per node,
                             agents run tools, results accumulate)
      -> Verifier           (terminal node -> VerificationResult)
      -> final answer + full audit trail

Keeps orchestration separate from the API and from the domain services.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .agents.planner import Planner
from .agents.roles import build_agent_library
from .agents.router import ModelRouter
from .audit.events import AuditEventType
from .audit.trail import AuditTrail
from .core.config import Settings, get_settings
from .core.exceptions import AIOnSiteError
from .core.logging import bind_execution, clear_execution, get_logger
from .pipeline.builder import build_pipeline
from .pipeline.executor import PipelineExecutor
from .pipeline.models import ExecutionStatus, NodeStatus, PipelineNode
from .pipeline.requirements import RequirementStatus, extract_requirements
from .pipeline.state import ExecutionContext
from .pipeline.task_status import TaskStatus, compute_task_status
from .tools.registry import ToolRegistry, default_registry
from .verification.verifier import ResultVerifier

_log = get_logger("orchestrator")


class Orchestrator:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        audit: AuditTrail | None = None,
        registry: ToolRegistry | None = None,
        use_llm_verifier: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.audit = audit or AuditTrail()
        self.registry = registry or default_registry(allow_network=False)
        self._model_config = _load_model_config(self.settings.configs_dir)
        self._use_llm_verifier = use_llm_verifier
        self._routers: list[ModelRouter] = []

    # ------------------------------------------------------------------
    async def run_task(self, task: str, *, confidential: bool = False) -> ExecutionContext:
        ctx = ExecutionContext(
            task=task,
            sovereign_mode=self.settings.sovereign_mode,
            confidential=confidential,
        )
        bind_execution(execution_id=ctx.execution_id)
        ctx.status = ExecutionStatus.RUNNING
        self.audit.record(
            ctx.execution_id, AuditEventType.TASK_RECEIVED, component="orchestrator",
            message=task[:200],
            metadata={"confidential": confidential, "sovereign_mode": ctx.sovereign_mode},
        )
        if self.settings.sovereign_mode or confidential:
            self.audit.record(
                ctx.execution_id, AuditEventType.POLICY_ENFORCED, component="orchestrator",
                message="local-only routing enforced for this execution",
                metadata={
                    "sovereign_mode": self.settings.sovereign_mode,
                    "confidential": confidential,
                },
            )

        router = ModelRouter(
            self.settings, model_config=self._model_config, confidential=confidential
        )
        self._routers.append(router)

        try:
            await self._run(ctx, router, confidential)
        except AIOnSiteError as exc:
            ctx.error = exc.message
            ctx.mark_finished(ExecutionStatus.FAILED)
            ctx.task_status = TaskStatus.FAILED
            self.audit.record(
                ctx.execution_id, AuditEventType.EXECUTION_FAILED, component="orchestrator",
                status="failed", message=exc.message, metadata=exc.details,
            )
        except Exception as exc:  # noqa: BLE001
            ctx.error = f"{type(exc).__name__}: {exc}"
            ctx.mark_finished(ExecutionStatus.FAILED)
            ctx.task_status = TaskStatus.FAILED
            self.audit.record(
                ctx.execution_id, AuditEventType.EXECUTION_FAILED, component="orchestrator",
                status="failed", message=ctx.error,
            )
        finally:
            clear_execution()
        return ctx

    # ------------------------------------------------------------------
    async def _run(
        self, ctx: ExecutionContext, router: ModelRouter, confidential: bool
    ) -> None:
        plan_node = PipelineNode(
            id="plan", name="plan", agent="planner",
            require_local=confidential or self.settings.sovereign_mode,
        )
        planner_llm, _ = router.get_llm(plan_node)
        planner = Planner(
            planner_llm,
            max_attempts=self.settings.max_replans + 1,
            backoff_seconds=self.settings.plan_retry_backoff_seconds,
        )

        plan = await planner.plan(ctx.task)
        ctx.plan = plan.model_dump()
        if plan.degraded:
            ctx.plan_degraded = True
            ctx.plan_degraded_reason = plan.degraded_reason
            self.audit.record(
                ctx.execution_id, AuditEventType.PLAN_FAILED, component="planner",
                status="degraded",
                message=plan.degraded_reason or "planner fell back to the deterministic plan",
                metadata={"goal": plan.goal},
            )
        self.audit.record(
            ctx.execution_id, AuditEventType.PLAN_CREATED, component="planner",
            message=plan.goal,
            metadata={"steps": [s.model_dump() for s in plan.steps], "degraded": plan.degraded},
        )
        ctx.requirements = extract_requirements(plan)

        pipeline = build_pipeline(
            plan, settings=self.settings,
            require_local=confidential or self.settings.sovereign_mode,
        )
        ctx.pipeline = pipeline

        default_llm, _ = router.get_llm(plan_node)
        agent_library = build_agent_library(default_llm)
        verifier = ResultVerifier(default_llm if self._use_llm_verifier else None)

        executor = PipelineExecutor(
            agent_library=agent_library,
            registry=self.registry,
            router=router,
            audit=self.audit,
            verifier=verifier,
            settings=self.settings,
        )
        await executor.run(ctx, pipeline)

        failed = [n.id for n in pipeline.nodes if n.status is NodeStatus.FAILED]
        if ctx.final_answer is None:
            status = ExecutionStatus.FAILED
            ctx.error = ctx.error or "no final answer produced"
        elif ctx.final_verification and not ctx.final_verification.passed:
            status = ExecutionStatus.NEEDS_REVIEW
        elif failed:
            status = ExecutionStatus.NEEDS_REVIEW
        else:
            status = ExecutionStatus.COMPLETED
        ctx.mark_finished(status)

        # Execution status only says the DAG finished; task status says
        # whether the task was actually accomplished -- the two are tracked
        # separately on purpose (execution COMPLETED + task INCOMPLETE is a
        # valid, expected combination when required evidence went missing).
        ctx.task_status = compute_task_status(ctx.requirements, ctx.final_verification, status)

        missing_descriptions: list[str] = [
            r.description for r in ctx.requirements
            if r.required and r.status is not RequirementStatus.SATISFIED
        ]
        if ctx.final_verification:
            missing_descriptions += [
                m.description for m in ctx.final_verification.missing_requirements
            ]
        missing_descriptions = list(dict.fromkeys(missing_descriptions))  # de-dup, keep order

        if ctx.task_status is not TaskStatus.COMPLETED and ctx.final_answer is not None:
            # Deterministic (not another LLM call, so it cannot itself fail
            # or hallucinate) annotation making the gap explicit rather than
            # letting a confident-sounding partial answer stand unqualified.
            prefix = f"[INSPECTION STATUS: {ctx.task_status.value.upper()}]\n"
            if missing_descriptions:
                prefix += "Missing or unsatisfied: " + "; ".join(missing_descriptions) + "\n"
            prefix += (
                "The following is the best available partial result; treat any "
                "conclusion below as provisional, not a verified recommendation.\n\n"
            )
            ctx.final_answer = prefix + ctx.final_answer
            ctx.final_answer_degraded = True

        if ctx.final_answer is not None:
            self.audit.record(
                ctx.execution_id, AuditEventType.FINAL_ANSWER_GENERATED,
                component="orchestrator", message=ctx.final_answer[:200],
            )

        self.audit.record(
            ctx.execution_id, AuditEventType.TASK_STATUS_DETERMINED, component="orchestrator",
            status=ctx.task_status.value, message=f"task_status={ctx.task_status.value}",
            metadata={
                "missing_requirements": missing_descriptions,
                "plan_degraded": ctx.plan_degraded,
            },
        )
        self.audit.record(
            ctx.execution_id, AuditEventType.EXECUTION_COMPLETED, component="orchestrator",
            status=status.value, message=f"duration_ms={ctx.duration_ms}",
            metadata={"failed_nodes": failed},
        )

    # ------------------------------------------------------------------
    async def aclose(self) -> None:
        for router in self._routers:
            await router.aclose()
        self._routers.clear()


def _load_model_config(configs_dir: Path) -> dict[str, Any]:
    path = Path(configs_dir) / "models.yaml"
    if not path.is_file():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover
        _log.warning("models_config_unreadable", error=str(exc))
        return {}
