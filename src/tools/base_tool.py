"""Tool abstraction.

A tool is a typed, permissioned, individually-loggable capability an agent
can call.  Subclasses declare Pydantic input/output schemas; the base
class validates arguments, enforces permissions, times execution and
returns a normalised :class:`ToolResult` -- agents never call
:meth:`_run` directly.
"""

from __future__ import annotations

import enum
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar, cast

from pydantic import BaseModel, ValidationError

from ..core.exceptions import ToolExecutionError
from ..core.logging import get_logger
from ..pipeline.models import ToolResult

_log = get_logger("tools")

#: The tool's validated-input model.  A concrete tool binds this by
#: subclassing ``BaseTool[MyInputModel]`` so that :meth:`_run` receives the
#: precise model type instead of a bare :class:`~pydantic.BaseModel`.
TInput = TypeVar("TInput", bound=BaseModel)


class ToolPermission(str, enum.Enum):
    PURE = "pure"                 # no side effects, no I/O
    READ_FILESYSTEM = "read_fs"
    NETWORK = "network"
    WRITE_FILESYSTEM = "write_fs"
    # Arbitrary code execution, even when isolated (subprocess/container) --
    # kept distinct from the other permissions because its risk profile is
    # categorically different (untrusted, LLM-authored code, not a fixed,
    # audited implementation), so a deployment must opt in explicitly (see
    # ``default_registry(allow_sandbox=...)``) the same way network access
    # already requires an explicit opt-in.
    SANDBOXED_EXEC = "sandboxed_exec"


class ToolCategory(str, enum.Enum):
    """Coarse functional classification, orthogonal to :class:`ToolPermission`.

    ``ToolPermission`` gates *what a tool is allowed to touch*;
    ``ToolCategory`` describes *what kind of work it does*, so planning/
    routing/caching logic can reason about a tool without hardcoding its
    name. A tool with an unclassified category (``UNSPECIFIED``) is treated
    conservatively everywhere a category matters (e.g. never auto-cached).
    """

    RETRIEVAL = "retrieval"
    COMPUTATION = "computation"
    TRANSFORMATION = "transformation"
    DOCUMENT = "document"
    ACTION = "action"
    VERIFICATION = "verification"
    PERCEPTION = "perception"
    UNSPECIFIED = "unspecified"


class BaseTool(ABC, Generic[TInput]):
    #: unique registry key
    name: ClassVar[str] = ""
    #: human/LLM-facing description
    description: ClassVar[str] = ""
    #: permissions this tool needs -- the registry can refuse to grant some
    permissions: ClassVar[tuple[ToolPermission, ...]] = (ToolPermission.PURE,)
    #: functional classification -- see :class:`ToolCategory`.
    category: ClassVar[ToolCategory] = ToolCategory.UNSPECIFIED
    #: Whether identical calls (same tool + same normalised arguments) may
    #: be served from an execution-scoped cache instead of re-running.
    #: Defaults to ``False`` (never cache) so a new tool is only cached
    #: after someone deliberately asserts it is safe to: pure/deterministic,
    #: no side effects, and not sensitive to rapidly-changing external
    #: state. See ``src/pipeline/state.py::ToolCallCache``.
    cacheable: ClassVar[bool] = False
    #: Pydantic models describing the contract.  ``InputModel`` stays a
    #: ``ClassVar`` (it must be readable off the class in :meth:`spec`), so
    #: it is typed as ``type[BaseModel]`` here and re-narrowed to ``TInput``
    #: at the one place it is instantiated (see :meth:`run`).
    InputModel: ClassVar[type[BaseModel]]
    OutputModel: ClassVar[type[BaseModel] | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls._run, "__isabstractmethod__", False):
            return  # still an abstract intermediate class
        if not cls.name:
            raise TypeError(f"{cls.__name__} must set a non-empty `name`")
        if not hasattr(cls, "InputModel"):
            raise TypeError(f"{cls.__name__} must define an `InputModel`")

    # ------------------------------------------------------------------
    async def run(self, **kwargs: Any) -> ToolResult:
        started = time.perf_counter()
        call = ToolResult(tool=self.name, input=_sanitise(kwargs))
        try:
            args = self.InputModel(**kwargs)
        except ValidationError as exc:
            call.ok = False
            call.error_kind = "validation"
            call.expected_schema = self.InputModel.model_json_schema()
            call.error = "invalid arguments: " + "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or '(value)'}: {e['msg']}"
                for e in exc.errors()
            )
            call.duration_ms = (time.perf_counter() - started) * 1000
            _log.warning("tool_bad_args", tool=self.name, error=call.error)
            return call

        try:
            # ``args`` is an instance of ``self.InputModel`` which, by the
            # ``BaseTool[TInput]`` contract, is ``type[TInput]``.  A ``ClassVar``
            # cannot carry the type variable, so bridge it here once.
            output = await self._run(cast(TInput, args))
            if self.OutputModel is not None and not isinstance(output, self.OutputModel):
                output = self.OutputModel(**output) if isinstance(output, dict) else output
            call.output = (
                output.model_dump() if isinstance(output, BaseModel) else output
            )
            call.ok = True
        except ToolExecutionError as exc:
            call.ok = False
            call.error_kind = "execution"
            call.error = exc.message
        except Exception as exc:  # noqa: BLE001
            call.ok = False
            call.error_kind = "execution"
            call.error = f"{type(exc).__name__}: {exc}"
        finally:
            call.duration_ms = (time.perf_counter() - started) * 1000

        _log.info(
            "tool_call",
            tool=self.name,
            ok=call.ok,
            duration_ms=round(call.duration_ms or 0, 2),
            error=call.error,
        )
        return call

    # ------------------------------------------------------------------
    @abstractmethod
    async def _run(self, args: TInput) -> Any:
        """Perform the work.  Raise :class:`ToolExecutionError` on failure.

        ``args`` is an instance of this tool's own ``InputModel``.
        """
        raise NotImplementedError

    @classmethod
    def spec(cls) -> dict[str, Any]:
        """JSON-serialisable description for prompts / the frontend."""
        return {
            "name": cls.name,
            "description": cls.description,
            "permissions": [p.value for p in cls.permissions],
            "category": cls.category.value,
            "cacheable": cls.cacheable,
            "input_schema": cls.InputModel.model_json_schema(),
            "output_schema": (
                cls.OutputModel.model_json_schema() if cls.OutputModel else None
            ),
        }


_SECRET_KEYS = ("key", "token", "secret", "password", "authorization")


def _sanitise(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in data.items():
        if any(s in k.lower() for s in _SECRET_KEYS):
            out[k] = "***redacted***"
        elif isinstance(v, str) and len(v) > 500:
            out[k] = v[:500] + "...(truncated)"
        else:
            out[k] = v
    return out
