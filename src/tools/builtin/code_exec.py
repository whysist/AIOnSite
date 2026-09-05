"""Sandboxed Python execution tool -- the only tool in this project that
runs LLM-authored code rather than a fixed, audited implementation.

Disabled by default: the registry only grants ``ToolPermission.SANDBOXED_EXEC``
when a deployment explicitly opts in (``default_registry(allow_sandbox=True)``,
gated by ``Settings.sandbox_enabled``) -- the same opt-in pattern already
used for network tools.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from ...core.config import get_settings
from ...core.exceptions import ToolExecutionError
from ...sandbox.executor import SandboxExecutor
from ..base_tool import BaseTool, ToolCategory, ToolPermission


class _CodeIn(BaseModel):
    code: str = Field(
        ..., max_length=20_000,
        description=(
            "Self-contained Python source to run in an isolated sandbox. "
            "No file or network access, no imports beyond the standard "
            "numeric/string stdlib. Print the result -- only stdout is "
            "returned."
        ),
    )


class _CodeOut(BaseModel):
    stdout: str
    stderr: str
    return_code: int | None
    timed_out: bool
    used_docker: bool
    duration_ms: float


class SandboxedPythonTool(BaseTool[_CodeIn]):
    name = "execute_python"
    description = (
        "Run a short Python snippet in an isolated sandbox for calculations "
        "the calculator/deviation tools can't express (multi-step numeric "
        "logic, unit conversions, small data transforms). No file or "
        "network access; hard timeout; print() the result."
    )
    permissions: ClassVar[tuple[ToolPermission, ...]] = (ToolPermission.SANDBOXED_EXEC,)
    category: ClassVar[ToolCategory] = ToolCategory.ACTION
    # Executing the same source text twice is not guaranteed to be
    # side-effect-free or produce the same output (wall-clock timing,
    # non-deterministic code the model might write) -- never cache.
    cacheable: ClassVar[bool] = False
    InputModel = _CodeIn
    OutputModel = _CodeOut

    def __init__(self, executor: SandboxExecutor | None = None) -> None:
        if executor is not None:
            self._executor = executor
        else:
            settings = get_settings()
            self._executor = SandboxExecutor(
                use_docker=settings.sandbox_use_docker,
                docker_image=settings.sandbox_docker_image,
                timeout_seconds=settings.sandbox_timeout_seconds,
                max_output_chars=settings.sandbox_max_output_chars,
                memory_limit_mb=settings.sandbox_memory_limit_mb,
                cpu_limit=settings.sandbox_cpu_limit,
            )

    async def _run(self, args: _CodeIn) -> _CodeOut:
        result = await self._executor.run(args.code)
        if result.stage == "rejected":
            raise ToolExecutionError(f"code rejected by sandbox: {result.error}")
        return _CodeOut(
            stdout=result.stdout, stderr=result.stderr, return_code=result.return_code,
            timed_out=result.timed_out, used_docker=result.used_docker,
            duration_ms=result.duration_ms,
        )
