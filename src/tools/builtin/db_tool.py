"""Structured equipment/maintenance lookups against the local SQLite database.

Source: PR #10 ("ocr, rag, db_tool, doc", reuben-it), reconciled onto the
``BaseTool[TInput]`` contract established in this project. The original PR
also included a second tool, ``CalculateDeviationTool`` -- dropped here
because it duplicated the existing ``calculate_deviation`` tool
(``src/tools/builtin/calculator.py``) under the *same* registry name, and
represented limit=0 as ``float('inf')`` (renders as the non-standard
``Infinity`` JSON token via plain ``json.dumps``) where the existing tool
returns ``None`` -- the existing tool is kept as the single source of truth
for that calculation.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, Field

from ...database.models import (
    get_equipment,
    get_inspection_records,
    get_maintenance_history,
    get_operating_limits,
)
from ..base_tool import BaseTool, ToolPermission


class _EquipmentIn(BaseModel):
    equipment_id: str = Field(..., description="Target equipment ID (e.g. V-101 or P-101)")
    query_type: str = Field(
        default="all",
        description="Query type: info / limits / inspection / maintenance / all",
    )


class _EquipmentOut(BaseModel):
    equipment_id: str
    info: dict[str, Any] | None = None
    limits: list[dict[str, Any]] = Field(default_factory=list)
    inspection: list[dict[str, Any]] = Field(default_factory=list)
    maintenance: list[dict[str, Any]] = Field(default_factory=list)


class EquipmentLookupTool(BaseTool[_EquipmentIn]):
    name = "equipment_lookup"
    description = (
        "Look up equipment info, operating limits, inspection records, or "
        "maintenance history from the local equipment database."
    )
    permissions: ClassVar[tuple[ToolPermission, ...]] = (ToolPermission.PURE,)
    InputModel = _EquipmentIn
    OutputModel = _EquipmentOut

    async def _run(self, args: _EquipmentIn) -> _EquipmentOut:
        result = _EquipmentOut(equipment_id=args.equipment_id)
        qtype = args.query_type.lower()
        eid = args.equipment_id

        if qtype in ("info", "all"):
            eq = get_equipment(eid)
            result.info = eq.model_dump() if eq else None

        if qtype in ("limits", "all"):
            result.limits = [x.model_dump() for x in get_operating_limits(eid)]

        if qtype in ("inspection", "all"):
            result.inspection = [x.model_dump() for x in get_inspection_records(eid)]

        if qtype in ("maintenance", "all"):
            result.maintenance = [x.model_dump() for x in get_maintenance_history(eid)]

        return result
