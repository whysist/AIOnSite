from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    text: str
    source: str
    page: Optional[int] = None
    score: float
    document_id: Optional[str] = None
    section: Optional[str] = None
    chunk_id: Optional[str] = None
    parent_id: Optional[str] = None
    parent_text: Optional[str] = None
    revision: Optional[str] = None
    authority: Optional[str] = None  # e.g., "authoritative_limit", "sop", "observation", "history"
    equipment_ids: List[str] = Field(default_factory=list)


class Fact(BaseModel):
    parameter: str
    value: float | str
    unit: Optional[str] = None
    equipment_id: Optional[str] = None
    source: str
    page: Optional[int] = None
    document_id: Optional[str] = None
    revision: Optional[str] = None
    is_conflict: bool = False
    conflict_notes: Optional[str] = None
    confidence: float = 1.0


class Chunk(BaseModel):
    chunk_id: str
    text: str
    document_id: str
    source: str
    page_number: int
    chunk_index: int
    section: Optional[str] = None
    revision: Optional[str] = None
    authority: Optional[str] = None
    equipment_ids: List[str] = Field(default_factory=list)
    document_type: str = ""
    # Hierarchical Parent-Child Fields
    parent_id: Optional[str] = None
    parent_text: Optional[str] = None
    is_parent: bool = False
    chunk_type: str = "child"  # 'parent', 'child', 'table_row'
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalQuery(BaseModel):
    query: str
    equipment_id: Optional[str] = None
    document_type: Optional[str] = None
    revision: Optional[str] = None
    top_k: int = 5
    return_parent_context: bool = False  # When true, substitutes parent_text in evidence


class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float


class IngestionResult(BaseModel):
    document_id: str
    chunks_created: int
    parent_chunks: int = 0
    child_chunks: int = 0
    equipment_ids: List[str] = Field(default_factory=list)
    status: str = "success"
    warnings: List[str] = Field(default_factory=list)


class EvaluationCaseResult(BaseModel):
    case_id: str
    description: str
    passed: bool
    details: str


class EvaluationReport(BaseModel):
    total_cases: int
    passed_cases: int
    accuracy_score: float
    results: List[EvaluationCaseResult] = Field(default_factory=list)


class KnowledgeBaseStats(BaseModel):
    total_documents: int
    total_chunks: int
    equipment_ids: List[str] = Field(default_factory=list)
    document_types: List[str] = Field(default_factory=list)
