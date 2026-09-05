
from pydantic import BaseModel

from .connection import get_connection


class Equipment(BaseModel):
    equipment_id: str
    name: str
    type: str
    service: str
    status: str
    area: str

class MaintenanceRecord(BaseModel):
    record_id: int
    equipment_id: str
    date: str
    issue: str
    action: str
    status: str
    notes: str | None = None

class OperatingParameter(BaseModel):
    equipment_id: str
    parameter: str
    normal_min: float | None = None
    normal_max: float | None = None
    recommended_limit: float | None = None
    absolute_limit: float | None = None
    unit: str | None = None
    revision: str | None = None

class InspectionRecord(BaseModel):
    inspection_id: int
    equipment_id: str
    date: str
    parameter: str
    observed_value: float | None = None
    unit: str | None = None
    observation: str | None = None

def normalize_equipment_id(eid: str) -> str:
    """Strips hyphens and uppercases equipment ID."""
    return eid.replace("-", "").upper()

def get_equipment(equipment_id: str, db_path: str | None = None) -> Equipment | None:
    norm_id = normalize_equipment_id(equipment_id)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM equipment WHERE equipment_id = ?", (norm_id,))
        row = cursor.fetchone()
        if row:
            return Equipment(**dict(row))
        return None

def get_all_equipment(db_path: str | None = None) -> list[Equipment]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM equipment")
        return [Equipment(**dict(row)) for row in cursor.fetchall()]

def get_operating_limits(equipment_id: str, db_path: str | None = None) -> list[OperatingParameter]:
    norm_id = normalize_equipment_id(equipment_id)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM operating_parameters WHERE equipment_id = ?", (norm_id,))
        return [OperatingParameter(**dict(row)) for row in cursor.fetchall()]

def get_inspection_records(equipment_id: str, db_path: str | None = None) -> list[InspectionRecord]:
    norm_id = normalize_equipment_id(equipment_id)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM inspection_records WHERE equipment_id = ?", (norm_id,))
        return [InspectionRecord(**dict(row)) for row in cursor.fetchall()]

def get_maintenance_history(equipment_id: str, db_path: str | None = None) -> list[MaintenanceRecord]:
    norm_id = normalize_equipment_id(equipment_id)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM maintenance_records WHERE equipment_id = ?", (norm_id,))
        return [MaintenanceRecord(**dict(row)) for row in cursor.fetchall()]

def search_equipment(query: str, db_path: str | None = None) -> list[Equipment]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        search_pattern = f"%{query}%"
        cursor.execute(
            "SELECT * FROM equipment WHERE name LIKE ? OR type LIKE ?",
            (search_pattern, search_pattern)
        )
        return [Equipment(**dict(row)) for row in cursor.fetchall()]
