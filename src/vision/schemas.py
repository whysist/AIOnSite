from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

ProcessingMode = Literal["native", "ocr", "mixed"]


class OCRBlock(BaseModel):
    text: str
    confidence: float = 1.0
    bbox: Optional[List[float]] = None  # [x1, y1, x2, y2]


class OCRResult(BaseModel):
    text: str
    confidence: float = 1.0
    blocks: List[OCRBlock] = Field(default_factory=list)
    engine: str = "auto"
    warnings: List[str] = Field(default_factory=list)


class TableResult(BaseModel):
    table_id: str
    page: int
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    bbox: Optional[List[float]] = None


class ImageResult(BaseModel):
    image_id: str
    page: int
    format: str = "png"
    width: int = 0
    height: int = 0
    bbox: Optional[List[float]] = None
    image_bytes: Optional[bytes] = None


class PageResult(BaseModel):
    page_number: int
    text: str
    extraction_method: ProcessingMode = "native"
    tables: List[TableResult] = Field(default_factory=list)
    images: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class DocumentResult(BaseModel):
    document_id: str
    source: str
    pages: List[PageResult] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_warnings: List[str] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Concatenated full text across all pages."""
        return "\n\n".join([p.text for p in self.pages if p.text])

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def all_tables(self) -> List[TableResult]:
        tables = []
        for p in self.pages:
            tables.extend(p.tables)
        return tables


class VisionResult(BaseModel):
    """Optional VLM image analysis result."""
    summary: str
    detected_entities: List[str] = Field(default_factory=list)
    confidence: float = 1.0
    raw_response: Dict[str, Any] = Field(default_factory=dict)
