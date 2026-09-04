# src/database/repository.py
from typing import Optional, List
from src.database.connection import DatabaseConnection
from src.database.schemas import Equipment, MaintenanceRecord, OperatingParameter, InspectionRecord

def normalize_equipment_id(equipment_id: str) -> str:
    return equipment_id.replace("-", "").strip().upper()

class IndustrialRepository:
    def __init__(self, db_path: str = "industrial_data.db"):
        self.db = DatabaseConnection(db_path)

    def get_equipment(self, equipment_id: str) -> Optional[Equipment]:
        norm_id = normalize_equipment_id(equipment_id)
        with self.db.get_cursor() as cur:
            cur.execute("SELECT * FROM equipment WHERE equipment_id = ? OR equipment_id = ?", (norm_id, equipment_id))
            row = cur.fetchone()
            return Equipment(**dict(row)) if row else None

    def get_maintenance_records(self, equipment_id: str, status: Optional[str] = None) -> List[MaintenanceRecord]:
        norm_id = normalize_equipment_id(equipment_id)
        query = "SELECT * FROM maintenance_records WHERE (equipment_id = ? OR equipment_id = ?)"
        params = [norm_id, equipment_id]
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY date ASC"
        with self.db.get_cursor() as cur:
            cur.execute(query, tuple(params))
            return [MaintenanceRecord(**dict(row)) for row in cur.fetchall()]

    def get_equipment_history(self, equipment_id: str) -> List[MaintenanceRecord]:
        return self.get_maintenance_records(equipment_id)

    def get_operating_limit(self, equipment_id: str, parameter: str) -> Optional[OperatingParameter]:
        norm_id = normalize_equipment_id(equipment_id)
        with self.db.get_cursor() as cur:
            cur.execute(
                "SELECT * FROM operating_parameters WHERE (equipment_id = ? OR equipment_id = ?) AND parameter = ?",
                (norm_id, equipment_id, parameter)
            )
            row = cur.fetchone()
            return OperatingParameter(**dict(row)) if row else None

    def get_inspection_records(self, equipment_id: str) -> List[InspectionRecord]:
        norm_id = normalize_equipment_id(equipment_id)
        query = "SELECT * FROM inspection_records WHERE equipment_id = ? OR equipment_id = ? ORDER BY date DESC"
        with self.db.get_cursor() as cur:
            cur.execute(query, (norm_id, equipment_id))
            return [InspectionRecord(**dict(row)) for row in cur.fetchall()]