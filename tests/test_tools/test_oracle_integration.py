# tests/test_tools/test_oracle_integration.py
import json
import pytest
from src.database.repository import IndustrialRepository
from src.tools.builtin.industrial import ParameterComparisonTool, RiskScoreTool

DB_PATH = "data/SIH26117_Synthetic_Dataset_v2/database/equipment.db"
GT_PATH = "data/SIH26117_Synthetic_Dataset_v2/ground_truth/ground_truth.json"

async def test_ground_truth_v101_oracle():
    with open(GT_PATH) as f:
        gt = json.load(f)

    # 1. Repository lookup
    repo = IndustrialRepository(DB_PATH)
    equip = repo.get_equipment("V-101")
    assert equip is not None
    assert equip.equipment_id == "V101"

    # 2. Compare parameter via tool
    obs_pressure = gt["observations"]["V-101"]["pressure_bar"]
    expected_calcs = gt["expected_calculations"]["V-101"]

    tool = ParameterComparisonTool(DB_PATH)
    res = await tool.run(
        equipment_id="V-101",
        parameter="pressure",
        observed_value=obs_pressure,
    )
    assert res.ok is True
    assert res.output["deviation_from_recommended_pct"] == expected_calcs["pressure_deviation_from_recommended_upper_pct"]
    assert res.output["headroom_to_absolute_limit"] == expected_calcs["pressure_headroom_to_absolute_limit_bar"]
    assert res.output["status"] == "RECOMMENDED_EXCEEDED"

    # 3. Risk calculation
    risk_tool = RiskScoreTool()
    risk_res = await risk_tool.run(
        observed_deviation_pct=res.output["deviation_from_recommended_pct"],
        headroom=res.output["headroom_to_absolute_limit"],
    )
    assert risk_res.ok is True
    assert risk_res.output["risk_level"] == gt["expected_risk"]