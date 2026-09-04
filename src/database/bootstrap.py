# src/database/bootstrap.py
from src.database.connection import DatabaseConnection

def initialize_database(db_path: str = "industrial_data.db") -> None:
    db = DatabaseConnection(db_path)
    with db.get_cursor() as cur:
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS equipment (
                equipment_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                service TEXT NOT NULL,
                status TEXT NOT NULL,
                area TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS maintenance_records (
                record_id TEXT PRIMARY KEY,
                equipment_id TEXT NOT NULL,
                date TEXT NOT NULL,
                issue TEXT NOT NULL,
                action TEXT NOT NULL,
                status TEXT NOT NULL,
                notes TEXT,
                FOREIGN KEY(equipment_id) REFERENCES equipment(equipment_id)
            );
            CREATE TABLE IF NOT EXISTS operating_parameters (
                equipment_id TEXT NOT NULL,
                parameter TEXT NOT NULL,
                normal_min REAL,
                normal_max REAL,
                recommended_limit REAL NOT NULL,
                absolute_limit REAL NOT NULL,
                unit TEXT NOT NULL,
                revision TEXT DEFAULT 'v1',
                PRIMARY KEY(equipment_id, parameter)
            );
            CREATE TABLE IF NOT EXISTS inspection_records (
                inspection_id TEXT PRIMARY KEY,
                equipment_id TEXT NOT NULL,
                date TEXT NOT NULL,
                parameter TEXT NOT NULL,
                observed_value REAL NOT NULL,
                unit TEXT NOT NULL,
                observation TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_maint_equip ON maintenance_records(equipment_id);
            CREATE INDEX IF NOT EXISTS idx_param_equip ON operating_parameters(equipment_id);
        """)