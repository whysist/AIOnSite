# src/tools/builtin/equipment.py
from pydantic import BaseModel, Field
from typing import Optional, List
from src.tools.base import BaseTool
from src.database.repository import IndustrialRepository

class EquipmentLookupInput(BaseModel):
    equipment_id: str = Field(..., description="Canonical equipment identifier, e.g., V-101")

class EquipmentLookupOutput(BaseModel):
    found: bool
    equipment_id: str
    name: Optional[str] = None
    type: Optional[str] = None
    service: Optional[str] = None
    status: Optional[str] = None
    area: Optional[str] = None

class EquipmentInfoTool(BaseTool):
    name = "get_equipment_info"
    description = "Fetch canonical structured equipment metadata from SQLite"
    input_model = EquipmentLookupInput
    output_model = EquipmentLookupOutput

    def __init__(self, db_path: str = "industrial_data.db"):
        self.db_path = db_path

    def run(self, input_data: EquipmentLookupInput) -> EquipmentLookupOutput:
        repo = IndustrialRepository(self.db_path)
        item = repo.get_equipment(input_data.equipment_id)
        if not item:
            return EquipmentLookupOutput(found=False, equipment_id=input_data.equipment_id)
        return EquipmentLookupOutput(
            found=True,
            equipment_id=item.equipment_id,
            name=item.name,
            type=item.type,
            service=item.service,
            status=item.status,
            area=item.area
        )