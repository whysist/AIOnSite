# src/database/seed.py
from src.database.connection import DatabaseConnection

def seed_synthetic_database(db_path: str = "industrial_data.db") -> None:
    db = DatabaseConnection(db_path)
    with db.get_cursor() as cur:
        # 1. Canonical Equipment Identity
        cur.execute("""
            INSERT OR REPLACE INTO equipment 
            (equipment_id, name, type, service, status, area)
            VALUES ('V-101', 'Crude Separator', 'Pressure Vessel', 'Hydrocarbon', 'ACTIVE', 'Area-12')
        """)

        # 2. Operating Limits & Thresholds
        cur.execute("""
            INSERT OR REPLACE INTO operating_parameters 
            (equipment_id, parameter, normal_min, normal_max, recommended_limit, absolute_limit, unit, revision)
            VALUES ('V-101', 'shell_thickness', 12.0, 15.0, 10.5, 8.0, 'mm', 'v1')
        """)
        cur.execute("""
            INSERT OR REPLACE INTO operating_parameters 
            (equipment_id, parameter, normal_min, normal_max, recommended_limit, absolute_limit, unit, revision)
            VALUES ('V-101', 'operating_pressure', 2.0, 5.0, 6.0, 7.5, 'bar', 'v1')
        """)

        # 3. Past Maintenance Records
        cur.execute("""
            INSERT OR REPLACE INTO maintenance_records 
            (record_id, equipment_id, date, issue, action, status, notes)
            VALUES ('M-2024-01', 'V-101', '2024-01-15', 'Flange Leak', 'Replaced gasket', 'RESOLVED', 'Turnaround inspect')
        """)

        # 4. Frozen Inspection Records (Observed Readings)
        cur.execute("""
            INSERT OR REPLACE INTO inspection_records 
            (inspection_id, equipment_id, date, parameter, observed_value, unit, observation)
            VALUES ('INSP-2026-001', 'V-101', '2026-03-01', 'shell_thickness', 9.2, 'mm', 'Localized wall thinning detected near bottom head')
        """)
        cur.execute("""
            INSERT OR REPLACE INTO inspection_records 
            (inspection_id, equipment_id, date, parameter, observed_value, unit, observation)
            VALUES ('INSP-2026-002', 'V-101', '2026-03-01', 'operating_pressure', 5.8, 'bar', 'Pressure within normal peak operating range')
        """)