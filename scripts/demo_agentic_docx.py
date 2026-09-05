"""Demo: scanned report -> VLM reading -> Word approval note, proved sovereign.

Exercises three of the SIH26117 demo requirements in one run:
  * multimodal image/scanned-document understanding (the local VLM reads a
    real scanned inspection note from the dataset -- no OCR involved),
  * an agentic task carried through to a file artifact (a .docx approval
    note, not a chat reply),
  * live proof of zero external calls (NetworkGuard watches the socket
    layer for the whole run and prints exactly what was contacted).

Usage:
    python scripts/demo_agentic_docx.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.network_guard import NetworkGuard  # noqa: E402
from src.tools.builtin.docx_export import DocxExportTool  # noqa: E402
from src.tools.builtin.vision_tool import AnalyzeImageTool  # noqa: E402

_DATA_ROOT = Path.cwd() / "data"
_SCANNED_NOTE = "synthetic/SIH26117_Synthetic_Dataset_v2/auxiliary/03_scanned_inspection_note.png"
_SCHEMATIC = "synthetic/SIH26117_Synthetic_Dataset_v2/auxiliary/06_V101_Synthetic_Schematic.png"


async def main() -> None:
    guard = NetworkGuard()
    with guard:
        vision = AnalyzeImageTool(root=_DATA_ROOT)

        print("== Step 1: reading the scanned inspection note with the local VLM ==")
        note_result = await vision.run(
            file_path=_SCANNED_NOTE,
            prompt=(
                "This is a scanned equipment inspection note. Transcribe the key "
                "findings, readings, and any equipment tags you can read."
            ),
        )
        print(f"ok={note_result.ok}  ({note_result.duration_ms:.0f}ms)")
        if not note_result.ok:
            print(f"error: {note_result.error}")
            return
        summary = note_result.output["summary"]
        entities = note_result.output["detected_entities"]
        print(f"summary: {summary}")
        print(f"detected entities: {entities}")

        print("\n== Step 2: reading the equipment schematic with the local VLM ==")
        schematic_result = await vision.run(
            file_path=_SCHEMATIC,
            prompt="Identify every equipment tag visible in this schematic.",
        )
        print(f"ok={schematic_result.ok}  ({schematic_result.duration_ms:.0f}ms)")
        schematic_entities = schematic_result.output["detected_entities"] if schematic_result.ok else []
        print(f"detected entities: {schematic_entities}")

        print("\n== Step 3: drafting the Word approval note ==")
        equipment_id = (entities + schematic_entities + ["V-101"])[0]
        docx = DocxExportTool(root=_DATA_ROOT / "output")
        export_result = await docx.run(
            title=f"{equipment_id} Inspection Approval Note",
            equipment_id=equipment_id,
            summary=summary,
            findings=[
                f"Scanned inspection note (VLM read): {summary}",
                f"Equipment tags identified: {', '.join(entities + schematic_entities) or 'none detected'}",
            ],
            recommendation=(
                "Route to the responsible engineer for review before the next "
                "scheduled inspection window."
            ),
        )
        print(f"ok={export_result.ok}  ({export_result.duration_ms:.0f}ms)")
        if export_result.ok:
            print(f"written to: {export_result.output['path']}")
            print(f"size: {export_result.output['bytes']} bytes")

    print("\n== Sovereignty proof ==")
    proof = guard.summary()
    print(f"hosts contacted: {proof['hosts_contacted']}")
    print(f"external hosts:  {proof['external_hosts']}")
    print(f"sovereign:       {proof['sovereign']}")


if __name__ == "__main__":
    asyncio.run(main())
