from typing import Any

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    text: str
    source: str
    page: int | None = None
    score: float
    document_id: str | None = None
    section: str | None = None
    chunk_id: str | None = None
    parent_id: str | None = None
    parent_text: str | None = None
    revision: str | None = None
    authority: str | None = None  # e.g., "authoritative_limit", "sop", "observation", "history"
    equipment_ids: list[str] = Field(default_factory=list)


class Fact(BaseModel):
    parameter: str
    value: float | str
    unit: str | None = None
    equipment_id: str | None = None
    source: str
    page: int | None = None
    document_id: str | None = None
    revision: str | None = None
    is_conflict: bool = False
    conflict_notes: str | None = None
    confidence: float = 1.0


class Chunk(BaseModel):
    chunk_id: str
    text: str
    document_id: str
    source: str
    page_number: int
    chunk_index: int
    section: str | None = None
    revision: str | None = None
    authority: str | None = None
    equipment_ids: list[str] = Field(default_factory=list)
    document_type: str = ""
    # Hierarchical Parent-Child Fields
    parent_id: str | None = None
    parent_text: str | None = None
    is_parent: bool = False
    chunk_type: str = "child"  # 'parent', 'child', 'table_row'
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalQuery(BaseModel):
    query: str
    equipment_id: str | None = None
    document_type: str | None = None
    revision: str | None = None
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
    equipment_ids: list[str] = Field(default_factory=list)
    status: str = "success"
    warnings: list[str] = Field(default_factory=list)


class EvaluationCaseResult(BaseModel):
    case_id: str
    description: str
    passed: bool
    details: str


class EvaluationReport(BaseModel):
    total_cases: int
    passed_cases: int
    accuracy_score: float
    results: list[EvaluationCaseResult] = Field(default_factory=list)


class KnowledgeBaseStats(BaseModel):
    total_documents: int
    total_chunks: int
    equipment_ids: list[str] = Field(default_factory=list)
    document_types: list[str] = Field(default_factory=list)
