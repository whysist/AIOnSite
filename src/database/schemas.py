# src/database/schemas.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class Equipment:
    equipment_id: str
    name: str
    type: str
    service: str
    status: str
    area: str

@dataclass
class MaintenanceRecord:
    record_id: str
    equipment_id: str
    date: str
    issue: str
    action: str
    status: str
    notes: Optional[str] = None

@dataclass
class OperatingParameter:
    equipment_id: str
    parameter: str
    normal_min: float
    normal_max: float
    recommended_limit: float
    absolute_limit: float
    unit: str
    revision: Optional[str] = "v1"

@dataclass
class InspectionRecord:
    inspection_id: str
    equipment_id: str
    date: str
    parameter: str
    observed_value: float
    unit: str
    observation: str