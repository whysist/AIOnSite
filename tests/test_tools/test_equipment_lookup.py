"""EquipmentLookupTool against a self-contained temp SQLite DB (same schema
as the real synthetic dataset) -- no dependency on the dataset being
extracted.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager

import pytest

from src.tools.builtin.db_tool import EquipmentLookupTool


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "equipment.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE equipment (
            equipment_id TEXT, name TEXT, type TEXT, service TEXT, status TEXT, area TEXT
        );
        CREATE TABLE operating_parameters (
            equipment_id TEXT, parameter TEXT, normal_min REAL, normal_max REAL,
            recommended_limit REAL, absolute_limit REAL, unit TEXT, revision TEXT
        );
        CREATE TABLE inspection_records (
            inspection_id INTEGER, equipment_id TEXT, date TEXT, parameter TEXT,
            observed_value REAL, unit TEXT, observation TEXT
        );
        CREATE TABLE maintenance_records (
            record_id INTEGER, equipment_id TEXT, date TEXT, issue TEXT,
            action TEXT, status TEXT, notes TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO equipment VALUES (?,?,?,?,?,?)",
        ("V101", "V-101 Separator Vessel", "Pressure Vessel", "Separation", "Active", "Area 1"),
    )
    conn.execute(
        "INSERT INTO operating_parameters VALUES (?,?,?,?,?,?,?,?)",
        ("V101", "pressure", 10.0, 12.0, 12.0, 15.0, "bar", "B"),
    )
    conn.execute(
        "INSERT INTO inspection_records VALUES (?,?,?,?,?,?,?)",
        (1, "V101", "2026-08-28", "pressure", 14.5, "bar", "Above recommended limit"),
    )
    conn.commit()
    conn.close()

    # get_connection() defaults to get_db_path() when db_path is None; the
    # tool never passes a db_path itself, so point the module-level default
    # at our temp file instead of the real dataset location.
    @contextmanager
    def _fake_connection(db_path=None):
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    monkeypatch.setattr("src.database.models.get_connection", _fake_connection)
    return path


async def test_equipment_lookup_returns_real_row(db_path):
    tool = EquipmentLookupTool()
    result = await tool.run(equipment_id="V-101", query_type="all")
    assert result.ok is True
    assert result.output["info"]["name"] == "V-101 Separator Vessel"
    assert result.output["limits"][0]["recommended_limit"] == 12.0
    assert result.output["inspection"][0]["observed_value"] == 14.5


async def test_equipment_lookup_unknown_equipment_returns_none_info(db_path):
    tool = EquipmentLookupTool()
    result = await tool.run(equipment_id="Z-999", query_type="info")
    assert result.ok is True
    assert result.output["info"] is None


async def test_equipment_lookup_query_type_filters_sections(db_path):
    tool = EquipmentLookupTool()
    result = await tool.run(equipment_id="V-101", query_type="limits")
    assert result.output["info"] is None
    assert len(result.output["limits"]) == 1
    assert result.output["inspection"] == []
