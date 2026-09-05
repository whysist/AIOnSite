"""Draft a real Word (.docx) approval note from structured findings.

The agentic-task demo scenario this exists for: read a (scanned) inspection
report, pull out key findings, and hand back a file a human could actually
route for sign-off -- not a chat reply. Writes into a sandboxed output
directory using the same ``Path.relative_to()`` containment pattern as
``FileReadTool``/``ProcessDocumentTool`` (never ``str.startswith()`` --
see PROJECT_CONTEXT.md sec 6).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from ...core.exceptions import ToolExecutionError
from ..base_tool import BaseTool, ToolPermission

_APPROVAL_ROLES = ("Prepared by", "Reviewed by", "Approved by")


def _safe_filename(requested: str | None, equipment_id: str | None) -> str:
    """Resolve a user-suppliable filename to something safe to write.

    ``Path(...).name`` drops any directory components (so
    ``../../evil``, ``C:\\x``, etc. all collapse to a bare filename); the
    caller still re-checks containment with ``relative_to()`` afterwards as
    defence in depth, matching the rest of this project's file tools.
    """
    name = Path(requested).name if requested else ""
    if not name or name in {".", ".."}:
        tag = (equipment_id or "note").replace(" ", "_")
        name = f"approval_note_{tag}_{datetime.now():%Y%m%d_%H%M%S}"
    if not name.lower().endswith(".docx"):
        name += ".docx"
    return name


class _ExportIn(BaseModel):
    title: str = Field(..., max_length=200, description="Document title, e.g. 'Inspection Approval Note'")
    equipment_id: str | None = Field(default=None, max_length=50, description="e.g. V-101")
    summary: str = Field(default="", max_length=4000)
    findings: list[str] = Field(default_factory=list, max_length=50, description="Key findings, one per bullet")
    recommendation: str = Field(default="", max_length=2000)
    output_filename: str | None = Field(
        default=None, max_length=150,
        description="Optional filename; .docx is added automatically if missing",
    )


class _ExportOut(BaseModel):
    path: str
    filename: str
    bytes: int


class DocxExportTool(BaseTool[_ExportIn]):
    name = "export_approval_note"
    description = (
        "Draft a formatted Word (.docx) approval note from a title, summary, "
        "findings and a recommendation, with a sign-off table. Use this as "
        "the final step of a document-review task when the deliverable is a "
        "file, not chat text."
    )
    permissions: ClassVar[tuple[ToolPermission, ...]] = (ToolPermission.WRITE_FILESYSTEM,)
    InputModel = _ExportIn
    OutputModel = _ExportOut

    def __init__(self, root: str | Path | None = None) -> None:
        self._root = Path(root or Path.cwd() / "data" / "output").resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    async def _run(self, args: _ExportIn) -> _ExportOut:
        try:
            from docx import Document
        except ImportError as exc:  # pragma: no cover - exercised only if the dep is missing
            raise ToolExecutionError(
                "python-docx is not installed. Run: pip install python-docx"
            ) from exc

        filename = _safe_filename(args.output_filename, args.equipment_id)
        candidate = (self._root / filename).resolve()
        try:
            candidate.relative_to(self._root)
        except ValueError:
            raise ToolExecutionError("output filename escapes the allowed output directory") from None

        doc = Document()
        doc.add_heading(args.title, level=1)

        meta = doc.add_paragraph()
        meta.add_run(f"Equipment: {args.equipment_id or 'N/A'}    ").bold = True
        meta.add_run(f"Date: {datetime.now():%Y-%m-%d}").bold = True

        if args.summary:
            doc.add_heading("Summary", level=2)
            doc.add_paragraph(args.summary)

        if args.findings:
            doc.add_heading("Key Findings", level=2)
            for finding in args.findings:
                doc.add_paragraph(finding, style="List Bullet")

        if args.recommendation:
            doc.add_heading("Recommendation", level=2)
            doc.add_paragraph(args.recommendation)

        doc.add_heading("Approval", level=2)
        table = doc.add_table(rows=len(_APPROVAL_ROLES), cols=2)
        table.style = "Table Grid"
        for row, role in zip(table.rows, _APPROVAL_ROLES, strict=True):
            row.cells[0].text = role
            row.cells[1].text = ""

        doc.save(str(candidate))
        return _ExportOut(
            path=str(candidate),
            filename=filename,
            bytes=candidate.stat().st_size,
        )
