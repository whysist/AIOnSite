"""Knowledge-base search and structured-fact-extraction tools (P2 -> P1 contract).

Source: PR #10 ("ocr, rag, db_tool, doc", reuben-it), reconciled onto the
``BaseTool[TInput]`` contract. Both tools default to a single shared
``KnowledgeBase`` instance (rather than each constructing its own) so a
document ingested through one is visible to the other without relying on
disk persistence alone.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from ...retrieval.schemas import Evidence, Fact
from ...retrieval.service import KnowledgeBase
from ..base_tool import BaseTool, ToolCategory, ToolPermission

_default_kb: KnowledgeBase | None = None


def get_default_knowledge_base() -> KnowledgeBase:
    """Process-wide default ``KnowledgeBase``, built lazily on first use."""
    global _default_kb
    if _default_kb is None:
        _default_kb = KnowledgeBase()
    return _default_kb


# ---------------------------------------------------------------------
class _SearchIn(BaseModel):
    query: str = Field(..., max_length=2000, description="Search query string")
    equipment_id: str | None = Field(default=None, description="Optional equipment ID filter, e.g. V-101")
    k: int = Field(default=5, ge=1, le=20, description="Number of results to retrieve")


class _SearchOut(BaseModel):
    query: str
    equipment_id: str | None = None
    total_results: int
    evidence: list[Evidence]


class KnowledgeBaseSearchTool(BaseTool[_SearchIn]):
    name = "search_knowledge_base"
    description = (
        "Retrieve ranked evidence passages for a query. Each item has text, "
        "source, page, score, document_id."
    )
    permissions: ClassVar[tuple[ToolPermission, ...]] = (ToolPermission.PURE,)
    category: ClassVar[ToolCategory] = ToolCategory.RETRIEVAL
    # Read-only search over an already-ingested, execution-local knowledge
    # base: identical (query, equipment_id, k) yields the same ranked
    # evidence within one execution, so it is safe to reuse.
    cacheable: ClassVar[bool] = True
    InputModel = _SearchIn
    OutputModel = _SearchOut

    def __init__(self, kb: KnowledgeBase | None = None) -> None:
        self.kb = kb or get_default_knowledge_base()

    async def _run(self, args: _SearchIn) -> _SearchOut:
        results = self.kb.search_knowledge_base(
            query=args.query, equipment_id=args.equipment_id, top_k=args.k
        )
        return _SearchOut(
            query=args.query, equipment_id=args.equipment_id,
            total_results=len(results), evidence=results,
        )


# ---------------------------------------------------------------------
class _FactIn(BaseModel):
    query: str = Field(..., description="Query specifying the parameter or condition to extract")
    equipment_id: str | None = Field(default=None, description="Target equipment ID, e.g. V-101")


class _FactOut(BaseModel):
    query: str
    equipment_id: str | None = None
    total_facts: int
    facts: list[Fact]


class ExtractStructuredEvidenceTool(BaseTool[_FactIn]):
    name = "extract_structured_evidence"
    description = (
        "Return structured facts (parameter/value/unit/equipment_id/source/page) "
        "-- replaces fragile number scraping."
    )
    permissions: ClassVar[tuple[ToolPermission, ...]] = (ToolPermission.PURE,)
    category: ClassVar[ToolCategory] = ToolCategory.RETRIEVAL
    cacheable: ClassVar[bool] = True
    InputModel = _FactIn
    OutputModel = _FactOut

    def __init__(self, kb: KnowledgeBase | None = None) -> None:
        self.kb = kb or get_default_knowledge_base()

    async def _run(self, args: _FactIn) -> _FactOut:
        facts = self.kb.extract_facts(query=args.query, equipment_id=args.equipment_id)
        return _FactOut(
            query=args.query, equipment_id=args.equipment_id,
            total_facts=len(facts), facts=facts,
        )
