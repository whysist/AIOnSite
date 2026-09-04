import os
from typing import Dict, Any, List, Optional, ClassVar
from pydantic import BaseModel, Field
from src.tools.base_tool import BaseTool, ToolPermission
from src.vision.document_processor import process_document
from src.vision.schemas import DocumentResult, PageResult, TableResult


class _ProcessDocIn(BaseModel):
    file_path: str = Field(..., description="Relative or absolute path to document file")
    ocr_engine: str = Field(default="auto", description="OCR backend engine: auto/paddleocr/tesseract/fallback")


class _ProcessDocOut(BaseModel):
    document_id: str
    source: str
    total_pages: int
    full_text: str
    pages: List[PageResult] = Field(default_factory=list)
    tables: List[TableResult] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class ProcessDocumentTool(BaseTool):
    name: ClassVar[str] = "process_document"
    description: ClassVar[str] = "Parse a document (text/OCR/tables) into pages with provenance."
    permissions: ClassVar[tuple[ToolPermission, ...]] = (ToolPermission.READ_FILESYSTEM,)
    InputModel: ClassVar[type[BaseModel]] = _ProcessDocIn
    OutputModel: ClassVar[type[BaseModel]] = _ProcessDocOut

    async def _run(self, args: _ProcessDocIn) -> _ProcessDocOut:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        data_dir = os.path.join(project_root, "data")
        
        candidates = [
            os.path.normpath(args.file_path),
            os.path.normpath(os.path.join(data_dir, args.file_path)),
            os.path.normpath(os.path.join(project_root, args.file_path))
        ]
        
        target_path = None
        for c in candidates:
            if os.path.exists(c) and (c.startswith(data_dir) or c.startswith(project_root)):
                target_path = c
                break
                
        if not target_path or not os.path.exists(target_path):
            raise FileNotFoundError(f"File not found or access denied outside data directory: {args.file_path}")

        doc_res: DocumentResult = process_document(target_path, ocr_engine=args.ocr_engine)
        
        return _ProcessDocOut(
            document_id=doc_res.document_id,
            source=doc_res.source,
            total_pages=doc_res.total_pages,
            full_text=doc_res.full_text,
            pages=doc_res.pages,
            tables=doc_res.all_tables,
            metadata=doc_res.metadata,
            warnings=doc_res.processing_warnings
        )
