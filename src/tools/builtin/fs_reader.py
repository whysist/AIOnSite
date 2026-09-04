"""Read a text file from within a fixed, sandboxed root directory.

Guards: path is resolved and must stay inside ``root`` (no ``..`` escape,
no symlink escape), size is capped, and only UTF-8 text is returned.
The default root is ``<cwd>/data``.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from ...core.exceptions import ToolExecutionError
from ..base_tool import BaseTool, ToolPermission

_MAX_BYTES = 1_000_000


class _In(BaseModel):
    relative_path: str = Field(..., max_length=400, description="path relative to the data root")
    max_bytes: int = Field(default=_MAX_BYTES, gt=0, le=_MAX_BYTES)


class _Out(BaseModel):
    path: str
    bytes: int
    truncated: bool
    content: str


class FileReadTool(BaseTool[_In]):
    name = "read_file"
    description = "Read a UTF-8 text file located under the project data directory."
    permissions: ClassVar = (ToolPermission.READ_FILESYSTEM,)
    InputModel = _In
    OutputModel = _Out

    def __init__(self, root: str | Path | None = None) -> None:
        self._root = Path(root or Path.cwd() / "data").resolve()

    async def _run(self, args: _In) -> _Out:
        candidate = (self._root / args.relative_path).resolve()
        try:
            candidate.relative_to(self._root)
        except ValueError:
            raise ToolExecutionError("path escapes the allowed data root") from None
        if not candidate.is_file():
            raise ToolExecutionError(f"not a file: {args.relative_path}")

        raw = candidate.read_bytes()
        truncated = len(raw) > args.max_bytes
        raw = raw[: args.max_bytes]
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise ToolExecutionError("file is not valid UTF-8 text") from None
        return _Out(
            path=str(candidate.relative_to(self._root)),
            bytes=len(raw),
            truncated=truncated,
            content=text,
        )
