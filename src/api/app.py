"""FastAPI application.

Thin HTTP layer over :class:`src.orchestrator.Orchestrator`.  All
orchestration logic lives in the orchestrator / pipeline packages; this
module only translates HTTP <-> those calls and stores run results in an
in-process registry keyed by ``execution_id``.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..audit.trail import AuditTrail
from ..core.config import get_settings
from ..core.exceptions import AIOnSiteError
from ..core.logging import configure_logging, get_logger
from ..orchestrator import Orchestrator
from ..pipeline.state import ExecutionContext
from ..tools.builtin.docx_writer import DEFAULT_OUTPUT_ROOT
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
from .streaming import ExecutionEventBus

_log = get_logger("api")


class _AppState:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.bus = ExecutionEventBus()
        # Every audit event recorded anywhere in an execution (task
        # classified, plan created, node started/completed, tool
        # called/cache-hit/failed, verification completed, ...) is
        # published live to the matching execution's subscribers -- this is
        # what makes the pipeline's steps visible to a client in real time,
        # not just after the fact.
        self.audit = AuditTrail(
            on_event=lambda ev: self.bus.publish(ev.execution_id, ev.model_dump(mode="json"))
        )
        self.registry = default_registry(
            allow_network=False,
            allow_sandbox=self.settings.sandbox_enabled,
            allow_write_filesystem=self.settings.output_writing_enabled,
        )
        self.orchestrator = Orchestrator(
            self.settings, audit=self.audit, registry=self.registry
        )
        self.executions: dict[str, ExecutionContext] = {}
        # Keeps fire-and-forget background task objects referenced so
        # asyncio cannot garbage-collect them mid-run (a well-known footgun
        # with ``asyncio.create_task`` otherwise).
        self.background_tasks: set[asyncio.Task] = set()


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

# The frontend (frontend/index.html) is a plain static file opened directly
# from disk or served by an unrelated dev server, so its origin never
# matches this API's -- without CORS every fetch() call would be silently
# blocked by the browser. Wide open ("*") is fine here: this is a local,
# read-mostly demo API with no cookies/auth to leak; tighten this before any
# non-local deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Read-only static serving of whatever output-writing tools generate (e.g.
# GenerateWordDocumentTool) -- the exact same directory those tools write
# to, imported from one place (DEFAULT_OUTPUT_ROOT) so this can never drift
# out of sync with what a tool result's `path` field actually points at.
# Mounted unconditionally: harmless (and empty) when output-writing tools
# are disabled, since nothing is ever written into it in that case.
DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(DEFAULT_OUTPUT_ROOT)), name="generated_files")


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
        task_status=ctx.task_status.value,
        final_answer=ctx.final_answer,
        final_answer_degraded=ctx.final_answer_degraded,
        verification=(
            ctx.final_verification.model_dump() if ctx.final_verification else None
        ),
        error=ctx.error,
    )


@app.post("/tasks/stream", response_model=TaskCreatedResponse)
async def create_task_streaming(request: TaskRequest) -> TaskCreatedResponse:
    """Start a task in the background and return immediately.

    Unlike ``POST /tasks`` (which blocks until the task finishes -- kept
    unchanged for existing callers/tests), this returns as soon as an
    ``execution_id`` exists, so the caller can immediately open
    ``GET /tasks/{execution_id}/events`` and watch the pipeline's steps
    (classification, plan, each node starting/completing, each tool call)
    live, ending with the final answer.
    """
    st = _state()
    execution_id = f"exec_{uuid.uuid4().hex[:16]}"
    # Registered before the background task starts, so no event from the
    # very first moment of execution can be published before there is
    # somewhere for it to land.
    st.bus.create(execution_id)

    async def _run() -> None:
        try:
            ctx = await st.orchestrator.run_task(
                request.task, confidential=request.confidential, execution_id=execution_id,
            )
            st.executions[ctx.execution_id] = ctx
            st.bus.publish(execution_id, {
                "event_type": "stream_end", "execution_id": execution_id,
                "status": ctx.status.value, "task_status": ctx.task_status.value,
                "final_answer": ctx.final_answer, "final_answer_degraded": ctx.final_answer_degraded,
                "error": ctx.error,
            })
        except Exception as exc:  # noqa: BLE001 -- the stream must still terminate
            _log.error("streamed_task_failed", execution_id=execution_id, error=str(exc))
            st.bus.publish(execution_id, {
                "event_type": "stream_end", "execution_id": execution_id,
                "status": "failed", "task_status": "failed",
                "final_answer": None, "final_answer_degraded": False,
                "error": f"{type(exc).__name__}: {exc}",
            })

    task = asyncio.create_task(_run())
    st.background_tasks.add(task)
    task.add_done_callback(st.background_tasks.discard)

    return TaskCreatedResponse(
        execution_id=execution_id, status="running", task_status="pending",
    )


_SSE_KEEPALIVE_SECONDS = 15.0


async def _sse_event_stream(queue: asyncio.Queue, *, keepalive_seconds: float = _SSE_KEEPALIVE_SECONDS):
    """Yield SSE-formatted lines from *queue* until a ``stream_end`` payload.

    Every real event uses the default SSE "message" type (the event's own
    kind is inside the JSON payload as ``event_type``) rather than a custom
    ``event:`` line -- ``EventSource.onmessage`` then handles every event
    with one listener, including any new audit event type added later,
    with nothing to keep in sync client-side.

    A slow local model can leave genuinely long gaps (a minute or more)
    between real events -- long enough that a client or intermediary with
    its own idle-read timeout (a strict HTTP client, a reverse proxy)
    could tear the connection down thinking it stalled. A ``:`` line is a
    valid SSE comment (ignored by ``EventSource``, but still bytes on the
    wire) sent whenever the gap since the last real event exceeds
    *keepalive_seconds*, so the connection never looks idle for that long.
    """
    while True:
        try:
            payload = await asyncio.wait_for(queue.get(), timeout=keepalive_seconds)
        except asyncio.TimeoutError:
            yield ": keep-alive\n\n"
            continue
        yield f"data: {json.dumps(payload, default=str)}\n\n"
        if payload.get("event_type") == "stream_end":
            break


@app.get("/tasks/{execution_id}/events")
async def stream_task_events(execution_id: str):
    """Server-Sent Events feed of one execution's live steps.

    Each event is one audit event as it is recorded (task_classified,
    plan_created, node_started, tool_called, tool_cache_hit, node_completed,
    verification_completed, ...), followed by a terminal ``stream_end``
    event carrying the final answer. Safe to open before, during, or after
    the task runs: ``ExecutionEventBus`` replays everything buffered so far
    on subscribe, so a late connection never misses earlier steps.
    """
    st = _state()
    queue = st.bus.subscribe(execution_id)

    async def _events():
        try:
            async for line in _sse_event_stream(queue):
                yield line
        finally:
            st.bus.unsubscribe(execution_id, queue)

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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
        task_status=ctx.task_status.value,
        task=ctx.task,
        sovereign_mode=ctx.sovereign_mode,
        confidential=ctx.confidential,
        duration_ms=ctx.duration_ms,
        final_answer=ctx.final_answer,
        final_answer_degraded=ctx.final_answer_degraded,
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
