"""Word (.docx) document generation.

The only *write*-capable built-in tool in this project (every other tool is
read-only against a fixed source or purely computational). Requires
``ToolPermission.WRITE_FILESYSTEM``, which -- like ``NETWORK`` and
``SANDBOXED_EXEC`` -- ``default_registry`` only grants when a deployment
explicitly opts in (``allow_write_filesystem=True``, gated by
``Settings.output_writing_enabled``); a plain, unconfigured registry never
lets an agent write anything to disk.

Output is confined to a single fixed directory (``DEFAULT_OUTPUT_ROOT``) with
a sanitised, extension-forced filename -- the same containment pattern
``FileReadTool``/``ProcessDocumentTool`` already use for reads, mirrored here
for writes. ``src/api/app.py`` serves that same directory read-only at
``/files`` so a generated document is actually retrievable, not just a path
string in a tool result.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import ClassVar

from docx import Document
from pydantic import BaseModel, Field

from ...core.exceptions import ToolExecutionError
from ..base_tool import BaseTool, ToolCategory, ToolPermission

#: Shared with ``src/api/app.py`` (mounted at ``/files``) so the path a tool
#: call writes to and the path the API serves from can never drift apart.
DEFAULT_OUTPUT_ROOT = Path.cwd() / "data" / "generated"

_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitise_filename(name: str | None) -> str:
    """Strip any path component and disallowed characters; force ``.docx``.

    Never trusts the model's requested filename as a path -- only the base
    name survives, and only from a narrow safe-character set -- so a
    filename can never be used to escape ``DEFAULT_OUTPUT_ROOT`` or collide
    with something unexpected.
    """
    if not name:
        return f"document_{uuid.uuid4().hex[:12]}.docx"
    base = Path(name).name  # drops any directory components
    base = _SAFE_CHARS.sub("_", base).strip("._")
    if not base:
        base = f"document_{uuid.uuid4().hex[:12]}"
    if not base.lower().endswith(".docx"):
        base += ".docx"
    return base


class _Section(BaseModel):
    heading: str | None = Field(default=None, description="Optional section heading")
    body: str = Field(default="", description="Section paragraph text")


class _TableSpec(BaseModel):
    heading: str | None = Field(default=None, description="Optional heading shown above the table")
    headers: list[str] = Field(default_factory=list, description="Column headers")
    rows: list[list[str]] = Field(default_factory=list, description="Table rows, each a list of cell strings")


class _DocIn(BaseModel):
    title: str = Field(..., max_length=300, description="Document title, rendered as the top heading")
    sections: list[_Section] = Field(default_factory=list, description="Ordered body sections")
    tables: list[_TableSpec] = Field(default_factory=list, description="Ordered tables, appended after the sections")
    filename: str | None = Field(
        default=None, max_length=150,
        description="Optional base filename (no path; .docx is enforced). A generated name is used if omitted.",
    )


class _DocOut(BaseModel):
    filename: str
    path: str
    bytes: int
    sections: int
    tables: int


class GenerateWordDocumentTool(BaseTool[_DocIn]):
    name = "generate_word_document"
    description = (
        "Generate a Word (.docx) document from structured content (a title, "
        "ordered sections of heading+body text, and optional tables). Use "
        "this to produce a final deliverable document (e.g. an approval "
        "note or findings summary) -- not for arbitrary file writes. "
        "Returns the generated file's name and path; it is served read-only "
        "at /files/<filename>."
    )
    permissions: ClassVar[tuple[ToolPermission, ...]] = (ToolPermission.WRITE_FILESYSTEM,)
    category: ClassVar[ToolCategory] = ToolCategory.ACTION
    # Writing a document is a side effect with a fresh identity each time
    # (a new file, even for identical content) -- never memoize it.
    cacheable: ClassVar[bool] = False
    InputModel = _DocIn
    OutputModel = _DocOut

    def __init__(self, root: str | Path | None = None) -> None:
        self._root = Path(root or DEFAULT_OUTPUT_ROOT).resolve()

    async def _run(self, args: _DocIn) -> _DocOut:
        self._root.mkdir(parents=True, exist_ok=True)
        filename = _sanitise_filename(args.filename)
        candidate = (self._root / filename).resolve()
        try:
            candidate.relative_to(self._root)
        except ValueError:
            raise ToolExecutionError("generated filename escapes the allowed output directory") from None

        doc = Document()
        doc.add_heading(args.title, level=0)
        for section in args.sections:
            if section.heading:
                doc.add_heading(section.heading, level=1)
            if section.body:
                doc.add_paragraph(section.body)
        for table_spec in args.tables:
            if table_spec.heading:
                doc.add_heading(table_spec.heading, level=1)
            if table_spec.headers:
                table = doc.add_table(rows=1, cols=len(table_spec.headers))
                try:
                    table.style = "Light Grid Accent 1"
                except KeyError:  # pragma: no cover -- style availability depends on the template
                    pass
                for cell, header in zip(table.rows[0].cells, table_spec.headers, strict=False):
                    cell.text = str(header)
                for row in table_spec.rows:
                    cells = table.add_row().cells
                    for cell, value in zip(cells, row, strict=False):
                        cell.text = str(value)

        try:
            doc.save(str(candidate))
        except OSError as exc:
            raise ToolExecutionError(f"could not write document: {exc}") from exc

        return _DocOut(
            filename=filename,
            path=str(candidate.relative_to(self._root)),
            bytes=candidate.stat().st_size,
            sections=len(args.sections),
            tables=len(args.tables),
        )
