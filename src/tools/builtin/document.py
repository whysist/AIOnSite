"""Document processing tool -- OCR/PDF parsing via the P3 vision package.

Source: PR #10 ("ocr, rag, db_tool, doc", reuben-it), reconciled onto the
``BaseTool[TInput]`` contract. The original PR's path check
(``candidate.startswith(data_dir)``) is a string-prefix comparison, which a
sibling path like ``data_evil/...`` also satisfies -- replaced here with the
same ``Path.relative_to()`` containment check ``FileReadTool``
(``src/tools/builtin/fs_reader.py``) already uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from ...core.exceptions import ToolExecutionError
from ...vision.document_processor import process_document
from ...vision.schemas import DocumentResult, PageResult, TableResult
from ..base_tool import BaseTool, ToolCategory, ToolPermission


class _ProcessDocIn(BaseModel):
    file_path: str = Field(..., description="Path to the document, relative to the data directory")
    ocr_engine: str = Field(
        default="auto", description="OCR backend: auto / paddleocr / tesseract / fallback"
    )


class _ProcessDocOut(BaseModel):
    document_id: str
    source: str
    total_pages: int
    full_text: str
    pages: list[PageResult] = Field(default_factory=list)
    tables: list[TableResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ProcessDocumentTool(BaseTool[_ProcessDocIn]):
    name = "process_document"
    description = "Parse a document (native text, OCR, tables) into pages with provenance."
    permissions: ClassVar[tuple[ToolPermission, ...]] = (ToolPermission.READ_FILESYSTEM,)
    category: ClassVar[ToolCategory] = ToolCategory.DOCUMENT
    InputModel = _ProcessDocIn
    OutputModel = _ProcessDocOut

    def __init__(self, root: str | Path | None = None) -> None:
        self._root = Path(root or Path.cwd() / "data").resolve()

    async def _run(self, args: _ProcessDocIn) -> _ProcessDocOut:
        candidate = (self._root / args.file_path).resolve()
        try:
            candidate.relative_to(self._root)
        except ValueError:
            raise ToolExecutionError("path escapes the allowed data root") from None
        if not candidate.is_file():
            raise ToolExecutionError(f"not a file: {args.file_path}")

        doc_res: DocumentResult = process_document(str(candidate), ocr_engine=args.ocr_engine)
        return _ProcessDocOut(
            document_id=doc_res.document_id,
            source=doc_res.source,
            total_pages=doc_res.total_pages,
            full_text=doc_res.full_text,
            pages=doc_res.pages,
            tables=doc_res.all_tables,
            metadata=doc_res.metadata,
            warnings=doc_res.processing_warnings,
        )
