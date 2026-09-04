"""FastAPI application.

Thin HTTP layer over :class:`src.orchestrator.Orchestrator`.  All
orchestration logic lives in the orchestrator / pipeline packages; this
module only translates HTTP <-> those calls and stores run results in an
in-process registry keyed by ``execution_id``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from ..audit.trail import AuditTrail
from ..core.config import get_settings
from ..core.exceptions import AIOnSiteError
from ..core.logging import configure_logging, get_logger
from ..orchestrator import Orchestrator
from ..pipeline.state import ExecutionContext
from ..tools.registry import default_registry
from .schemas import (
    AuditResponse,
    ErrorResponse,
    ExecutionResponse,
    HealthResponse,
    PipelineResponse,
    TaskCreatedResponse,
    TaskRequest,
)

_log = get_logger("api")


class _AppState:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.audit = AuditTrail()
        self.registry = default_registry(allow_network=False)
        self.orchestrator = Orchestrator(
            self.settings, audit=self.audit, registry=self.registry
        )
        self.executions: dict[str, ExecutionContext] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(get_settings().log_level)
    app.state.ctx = _AppState()
    _log.info("api_startup", provider=app.state.ctx.settings.llm_provider.value,
              sovereign=app.state.ctx.settings.sovereign_mode)
    try:
        yield
    finally:
        await app.state.ctx.orchestrator.aclose()


app = FastAPI(title="AIOnSite", version="0.2.0", lifespan=lifespan)


def _state() -> _AppState:
    return app.state.ctx


@app.exception_handler(AIOnSiteError)
async def _aionsite_error_handler(_request, exc: AIOnSiteError):
    settings = get_settings()
    payload = ErrorResponse(
        error=exc.code,
        message=exc.message,
        details={} if settings.is_production else exc.details,
    )
    return JSONResponse(status_code=400, content=payload.model_dump())


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    st = _state()
    return HealthResponse(
        status="ok",
        environment=st.settings.environment.value,
        llm_provider=st.settings.llm_provider.value,
        sovereign_mode=st.settings.sovereign_mode,
        tools=st.registry.names(),
    )


@app.post("/tasks", response_model=TaskCreatedResponse)
async def create_task(request: TaskRequest) -> TaskCreatedResponse:
    st = _state()
    ctx = await st.orchestrator.run_task(request.task, confidential=request.confidential)
    st.executions[ctx.execution_id] = ctx
    return TaskCreatedResponse(
        execution_id=ctx.execution_id,
        status=ctx.status.value,
        final_answer=ctx.final_answer,
        verification=(
            ctx.final_verification.model_dump() if ctx.final_verification else None
        ),
        error=ctx.error,
    )


@app.get("/tasks", response_model=list[str])
async def list_tasks() -> list[str]:
    return list(_state().executions)


def _get_ctx(execution_id: str) -> ExecutionContext:
    ctx = _state().executions.get(execution_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"unknown execution {execution_id!r}")
    return ctx


@app.get("/tasks/{execution_id}", response_model=ExecutionResponse)
async def get_task(execution_id: str) -> ExecutionResponse:
    ctx = _get_ctx(execution_id)
    return ExecutionResponse(
        execution_id=ctx.execution_id,
        status=ctx.status.value,
        task=ctx.task,
        sovereign_mode=ctx.sovereign_mode,
        confidential=ctx.confidential,
        duration_ms=ctx.duration_ms,
        final_answer=ctx.final_answer,
        final_verification=(
            ctx.final_verification.model_dump() if ctx.final_verification else None
        ),
        error=ctx.error,
        node_results={k: v.model_dump() for k, v in ctx.state.node_results.items()},
    )


@app.get("/tasks/{execution_id}/pipeline", response_model=PipelineResponse)
async def get_pipeline(execution_id: str) -> PipelineResponse:
    ctx = _get_ctx(execution_id)
    return PipelineResponse(
        execution_id=ctx.execution_id,
        status=ctx.status.value,
        pipeline=ctx.pipeline.as_graph_dict() if ctx.pipeline else None,
    )


@app.get("/tasks/{execution_id}/audit", response_model=AuditResponse)
async def get_audit(execution_id: str) -> AuditResponse:
    _get_ctx(execution_id)
    return AuditResponse(
        execution_id=execution_id,
        events=_state().audit.as_dicts(execution_id),
    )
