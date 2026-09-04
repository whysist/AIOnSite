# tests/test_database/test_repository.py
import pytest
from src.database.bootstrap import initialize_database
from src.database.seed import seed_synthetic_database
from src.database.repository import IndustrialRepository

@pytest.fixture
def test_db(tmp_path):
    db_file = str(tmp_path / "test_industrial.db")
    initialize_database(db_file)
    seed_synthetic_database(db_file)
    return db_file

def test_equipment_lookup(test_db):
    repo = IndustrialRepository(test_db)
    equip = repo.get_equipment("V-101")
    assert equip is not None
    assert equip.name == "Crude Separator"
    assert equip.area == "Area-12"

def test_operating_limits(test_db):
    repo = IndustrialRepository(test_db)
    limits = repo.get_operating_limit("V-101", "shell_thickness")
    assert limits is not None
    assert limits.recommended_limit == 10.5
    assert limits.absolute_limit == 8.0