"""Deterministic industrial engineering tools."""

from __future__ import annotations

from typing import Any, ClassVar
from pydantic import BaseModel, Field

from src.core.exceptions import ToolExecutionError
from src.tools.base_tool import BaseTool, ToolPermission
from src.database.repository import IndustrialRepository

# ---------------------------------------------------------------------------
# Parameter Comparison Tool
# ---------------------------------------------------------------------------

class ParameterComparisonInput(BaseModel):
    equipment_id: str = Field(..., description="Canonical ID (e.g. V-101 or V101)")
    parameter: str = Field(..., description="Parameter name (e.g. pressure, shell_thickness)")
    observed_value: float = Field(..., description="Observed numeric reading")

class ParameterComparisonOutput(BaseModel):
    equipment_id: str
    parameter: str
    observed_value: float
    recommended_limit: float
    absolute_limit: float
    unit: str
    deviation_from_recommended_pct: float
    headroom_to_absolute_limit: float
    status: str

class ParameterComparisonTool(BaseTool[ParameterComparisonInput]):
    name: ClassVar[str] = "compare_parameter"
    description: ClassVar[str] = "Compare observed reading against limits, computing headroom and deviation."
    permissions: ClassVar[tuple[ToolPermission, ...]] = (ToolPermission.READ_FILESYSTEM,)
    InputModel: ClassVar[type[BaseModel]] = ParameterComparisonInput
    OutputModel: ClassVar[type[BaseModel] | None] = ParameterComparisonOutput

    def __init__(self, db_path: str = "data/SIH26117_Synthetic_Dataset_v2/database/equipment.db") -> None:
        super().__init__()
        self.db_path = db_path

    async def _run(self, args: ParameterComparisonInput) -> dict[str, Any]:
        repo = IndustrialRepository(self.db_path)
        limits = repo.get_operating_limit(args.equipment_id, args.parameter)
        if not limits:
            raise ToolExecutionError(
                f"No limits configured for parameter '{args.parameter}' on equipment '{args.equipment_id}'"
            )

        obs = args.observed_value
        rec = limits.recommended_limit
        abs_lim = limits.absolute_limit

        is_lower_bound = rec > abs_lim

        if is_lower_bound:
            dev_pct = round(((rec - obs) / rec) * 100.0, 2) if obs < rec else 0.0
            headroom = round(obs - abs_lim, 2)
            if obs <= abs_lim:
                status = "ABSOLUTE_EXCEEDED"
            elif obs <= rec:
                status = "RECOMMENDED_EXCEEDED"
            else:
                status = "NORMAL"
        else:
            dev_pct = round(((obs - rec) / rec) * 100.0, 2) if obs > rec else 0.0
            headroom = round(abs_lim - obs, 2)
            if obs >= abs_lim:
                status = "ABSOLUTE_EXCEEDED"
            elif obs >= rec:
                status = "RECOMMENDED_EXCEEDED"
            else:
                status = "NORMAL"

        return {
            "equipment_id": args.equipment_id,
            "parameter": args.parameter,
            "observed_value": obs,
            "recommended_limit": rec,
            "absolute_limit": abs_lim,
            "unit": limits.unit,
            "deviation_from_recommended_pct": dev_pct,
            "headroom_to_absolute_limit": headroom,
            "status": status,
        }

# ---------------------------------------------------------------------------
# Risk Score Tool
# ---------------------------------------------------------------------------

class RiskScoreInput(BaseModel):
    observed_deviation_pct: float
    headroom: float
    criticality_factor: float = 1.0

class RiskScoreOutput(BaseModel):
    risk_level: str
    risk_score: float
    formula: str
    version: str

class RiskScoreTool(BaseTool[RiskScoreInput]):
    name: ClassVar[str] = "calculate_risk_score"
    description: ClassVar[str] = "Deterministic calculation of engineering risk level"
    permissions: ClassVar[tuple[ToolPermission, ...]] = (ToolPermission.PURE,)
    InputModel: ClassVar[type[BaseModel]] = RiskScoreInput
    OutputModel: ClassVar[type[BaseModel] | None] = RiskScoreOutput

    async def _run(self, args: RiskScoreInput) -> dict[str, Any]:
        score = round(args.observed_deviation_pct * args.criticality_factor, 2)
        if args.observed_deviation_pct > 15.0 or args.headroom <= 0.5:
            level = "HIGH"
        elif args.observed_deviation_pct > 5.0 or args.headroom < 2.0:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "risk_level": level,
            "risk_score": score,
            "formula": "observed_deviation_pct * criticality_factor",
            "version": "2.0.0",
        }