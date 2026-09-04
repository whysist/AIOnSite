from typing import List, Optional, Any, ClassVar
from pydantic import BaseModel, Field
from src.tools.base_tool import BaseTool, ToolPermission
from src.retrieval.service import KnowledgeBase
from src.retrieval.schemas import Evidence, Fact


# --- 1. KnowledgeBaseSearchTool Schemas & Implementation ---

class _SearchIn(BaseModel):
    query: str = Field(..., max_length=2000, description="Search query string")
    equipment_id: Optional[str] = Field(default=None, description="Optional equipment ID filter e.g. V-101")
    k: int = Field(default=5, ge=1, le=20, description="Number of results to retrieve")


class _SearchOut(BaseModel):
    query: str
    equipment_id: Optional[str] = None
    total_results: int
    evidence: List[Evidence]


class KnowledgeBaseSearchTool(BaseTool):
    name: ClassVar[str] = "search_knowledge_base"
    description: ClassVar[str] = (
        "Retrieve ranked evidence passages for a query. Each item has text, source, page, score, document_id."
    )
    permissions: ClassVar[tuple[ToolPermission, ...]] = (ToolPermission.PURE,)
    InputModel: ClassVar[type[BaseModel]] = _SearchIn
    OutputModel: ClassVar[type[BaseModel]] = _SearchOut

    def __init__(self, kb: Optional[KnowledgeBase] = None):
        self.kb = kb or KnowledgeBase()

    async def _run(self, args: _SearchIn) -> _SearchOut:
        results: List[Evidence] = self.kb.search_knowledge_base(
            query=args.query,
            equipment_id=args.equipment_id,
            top_k=args.k
        )
        return _SearchOut(
            query=args.query,
            equipment_id=args.equipment_id,
            total_results=len(results),
            evidence=results
        )


# --- 2. ExtractStructuredEvidenceTool Schemas & Implementation ---

class _FactIn(BaseModel):
    query: str = Field(..., description="Query specifying the parameter or condition to extract")
    equipment_id: Optional[str] = Field(default=None, description="Target equipment ID e.g. V-101")


class _FactOut(BaseModel):
    query: str
    equipment_id: Optional[str] = None
    total_facts: int
    facts: List[Fact]


class ExtractStructuredEvidenceTool(BaseTool):
    name: ClassVar[str] = "extract_structured_evidence"
    description: ClassVar[str] = (
        "Return structured facts (parameter/value/unit/equipment_id/source/page) — replaces fragile number scraping."
    )
    permissions: ClassVar[tuple[ToolPermission, ...]] = (ToolPermission.PURE,)
    InputModel: ClassVar[type[BaseModel]] = _FactIn
    OutputModel: ClassVar[type[BaseModel]] = _FactOut

    def __init__(self, kb: Optional[KnowledgeBase] = None):
        self.kb = kb or KnowledgeBase()

    async def _run(self, args: _FactIn) -> _FactOut:
        facts: List[Fact] = self.kb.extract_facts(
            query=args.query,
            equipment_id=args.equipment_id
        )
        return _FactOut(
            query=args.query,
            equipment_id=args.equipment_id,
            total_facts=len(facts),
            facts=facts
        )
