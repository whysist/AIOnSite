from typing import Any, Dict, List, Optional, ClassVar
from pydantic import BaseModel, Field
from src.tools.base_tool import BaseTool, ToolPermission
from src.database.models import (
    get_equipment,
    get_operating_limits,
    get_inspection_records,
    get_maintenance_history
)


# --- 1. EquipmentLookupTool ---

class _EquipmentIn(BaseModel):
    equipment_id: str = Field(..., description="Target equipment ID (e.g. V-101 or P-101)")
    query_type: str = Field(default="all", description="Query type: info/limits/inspection/maintenance/all")


class _EquipmentOut(BaseModel):
    equipment_id: str
    info: Optional[Dict[str, Any]] = None
    limits: List[Dict[str, Any]] = Field(default_factory=list)
    inspection: List[Dict[str, Any]] = Field(default_factory=list)
    maintenance: List[Dict[str, Any]] = Field(default_factory=list)


class EquipmentLookupTool(BaseTool):
    name: ClassVar[str] = "equipment_lookup"
    description: ClassVar[str] = (
        "Look up equipment info, operating limits, inspection records, or maintenance history from the local equipment database."
    )
    permissions: ClassVar[tuple[ToolPermission, ...]] = (ToolPermission.PURE,)
    InputModel: ClassVar[type[BaseModel]] = _EquipmentIn
    OutputModel: ClassVar[type[BaseModel]] = _EquipmentOut

    async def _run(self, args: _EquipmentIn) -> _EquipmentOut:
        result = _EquipmentOut(equipment_id=args.equipment_id)
        qtype = args.query_type.lower()
        eid = args.equipment_id

        if qtype in ("info", "all"):
            eq = get_equipment(eid)
            result.info = eq.model_dump() if eq else None

        if qtype in ("limits", "all"):
            limits = get_operating_limits(eid)
            result.limits = [x.model_dump() for x in limits]

        if qtype in ("inspection", "all"):
            inspections = get_inspection_records(eid)
            result.inspection = [x.model_dump() for x in inspections]

        if qtype in ("maintenance", "all"):
            maintenance = get_maintenance_history(eid)
            result.maintenance = [x.model_dump() for x in maintenance]

        return result


# --- 2. CalculateDeviationTool ---

class _DeviationIn(BaseModel):
    actual: float = Field(..., description="Observed numeric value")
    limit: float = Field(..., description="Reference operating limit")
    label: Optional[str] = Field(default=None, description="Optional parameter name or label")


class _DeviationOut(BaseModel):
    actual: float
    limit: float
    absolute_deviation: float
    percent_deviation: float
    within_limit: bool
    formula: str
    label: Optional[str] = None


class CalculateDeviationTool(BaseTool):
    name: ClassVar[str] = "calculate_deviation"
    description: ClassVar[str] = (
        "Calculate percentage deviation between an observed value and a reference limit. Input: actual (float), limit (float), label (optional string)."
    )
    permissions: ClassVar[tuple[ToolPermission, ...]] = (ToolPermission.PURE,)
    InputModel: ClassVar[type[BaseModel]] = _DeviationIn
    OutputModel: ClassVar[type[BaseModel]] = _DeviationOut

    async def _run(self, args: _DeviationIn) -> _DeviationOut:
        actual_val = float(args.actual)
        limit_val = float(args.limit)

        abs_dev = actual_val - limit_val
        if limit_val != 0:
            pct_dev = (abs_dev / limit_val) * 100
        else:
            pct_dev = float('inf') if actual_val > 0 else float('-inf') if actual_val < 0 else 0.0

        formula = f"(({actual_val} - {limit_val}) / {limit_val}) * 100"

        return _DeviationOut(
            label=args.label,
            actual=actual_val,
            limit=limit_val,
            absolute_deviation=round(abs_dev, 4),
            percent_deviation=round(pct_dev, 2),
            within_limit=actual_val <= limit_val,
            formula=formula
        )
