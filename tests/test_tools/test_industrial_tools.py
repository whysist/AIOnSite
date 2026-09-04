import pytest
from src.database.bootstrap import initialize_database
from src.database.seed import seed_synthetic_database
from src.tools.builtin.industrial import ParameterComparisonTool, RiskScoreTool


@pytest.fixture
def seeded_db(tmp_path):
    db_file = str(tmp_path / "test_tools.db")
    initialize_database(db_file)
    seed_synthetic_database(db_file)
    return db_file


async def test_compare_parameter_safety_thresholds(seeded_db):
    tool = ParameterComparisonTool(seeded_db)

    # 1. Normal thickness (> 10.5mm)
    res_normal = await tool.run(
        equipment_id="V-101",
        parameter="shell_thickness",
        observed_value=11.0,
    )
    assert res_normal.ok is True
    assert res_normal.output["status"] == "NORMAL"

    # 2. Exceeding recommended limit (warning zone: <= 10.5mm and > 8.0mm)
    res_rec = await tool.run(
        equipment_id="V-101",
        parameter="shell_thickness",
        observed_value=9.5,
    )
    assert res_rec.ok is True
    assert res_rec.output["status"] == "RECOMMENDED_EXCEEDED"

    # 3. Exceeding absolute limit (critical trip zone: <= 8.0mm)
    res_abs = await tool.run(
        equipment_id="V-101",
        parameter="shell_thickness",
        observed_value=7.5,
    )
    assert res_abs.ok is True
    assert res_abs.output["status"] == "ABSOLUTE_EXCEEDED"


async def test_risk_score_reproducibility():
    tool = RiskScoreTool()
    res = await tool.run(
        observed_deviation_pct=20.83,
        headroom=0.5,
        criticality_factor=1.0,
    )
    assert res.ok is True
    assert res.output["risk_score"] == 20.83
    assert res.output["risk_level"] == "HIGH"
    assert res.output["version"] == "2.0.0"