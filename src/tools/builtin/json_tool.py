"""Parse / query JSON with a restricted dotted path -- pure tool."""

from __future__ import annotations

import json
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from ...core.exceptions import ToolExecutionError
from ..base_tool import BaseTool, ToolPermission


class _In(BaseModel):
    data: str = Field(..., max_length=200_000, description="a JSON document")
    path: str | None = Field(
        default=None,
        description="optional dotted path, e.g. 'items.0.value'",
        max_length=200,
    )


class _Out(BaseModel):
    value: Any
    type: str


class JsonParseTool(BaseTool):
    name = "json_parse"
    description = "Parse a JSON string and optionally extract a value by dotted path."
    permissions: ClassVar = (ToolPermission.PURE,)
    InputModel = _In
    OutputModel = _Out

    async def _run(self, args: _In) -> _Out:
        try:
            obj = json.loads(args.data)
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(f"invalid JSON: {exc}") from exc

        value: Any = obj
        if args.path:
            for part in args.path.split("."):
                if isinstance(value, list):
                    try:
                        value = value[int(part)]
                    except (ValueError, IndexError) as exc:
                        raise ToolExecutionError(
                            f"cannot index list with {part!r}"
                        ) from exc
                elif isinstance(value, dict):
                    if part not in value:
                        raise ToolExecutionError(f"key {part!r} not found")
                    value = value[part]
                else:
                    raise ToolExecutionError(
                        f"cannot descend into {type(value).__name__} at {part!r}"
                    )
        return _Out(value=value, type=type(value).__name__)
