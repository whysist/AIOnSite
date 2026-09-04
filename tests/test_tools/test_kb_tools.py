"""KnowledgeBaseSearchTool / ExtractStructuredEvidenceTool against a
self-contained, temp-directory KnowledgeBase -- no dependency on the
synthetic dataset being extracted, and no real embedding/vector-store
libraries required (both degrade to deterministic fallbacks).
"""

from __future__ import annotations

from src.retrieval.schemas import Evidence
from src.retrieval.service import KnowledgeBase
from src.tools.builtin.kb import ExtractStructuredEvidenceTool, KnowledgeBaseSearchTool
from src.vision.schemas import DocumentResult, PageResult


def _kb(tmp_path) -> KnowledgeBase:
    kb = KnowledgeBase(persist_dir=str(tmp_path / "kb"))
    kb.ingest_document(
        DocumentResult(
            document_id="V101_ops_limits",
            source="V101_Operating_Limits.pdf",
            pages=[
                PageResult(
                    page_number=1,
                    text=(
                        "Equipment V-101 operating limits.\n"
                        "Row: V-101 | Pressure | 100 | bar\n"
                        "Recommended limit: 100 bar for V-101 pressure."
                    ),
                    extraction_method="native",
                )
            ],
            metadata={"equipment_ids": ["V-101"], "document_type": "operating_limits"},
        )
    )
    return kb


async def test_search_tool_returns_real_evidence(tmp_path):
    tool = KnowledgeBaseSearchTool(kb=_kb(tmp_path))
    result = await tool.run(query="V-101 pressure limit", equipment_id="V-101", k=3)
    assert result.ok is True
    assert result.output["total_results"] > 0
    first = result.output["evidence"][0]
    assert isinstance(Evidence.model_validate(first), Evidence)
    assert "V-101" in first["text"] or "Pressure" in first["text"]


async def test_extract_evidence_tool_returns_structured_facts(tmp_path):
    tool = ExtractStructuredEvidenceTool(kb=_kb(tmp_path))
    result = await tool.run(query="V-101 pressure limit", equipment_id="V-101")
    assert result.ok is True
    assert result.output["total_facts"] > 0
    params = {f["parameter"] for f in result.output["facts"]}
    assert "pressure" in params


async def test_search_tool_validates_arguments(tmp_path):
    tool = KnowledgeBaseSearchTool(kb=_kb(tmp_path))
    result = await tool.run(query="x", k=999)  # k is capped at 20
    assert result.ok is False
    assert "invalid arguments" in result.error
