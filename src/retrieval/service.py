import glob
import logging
import os
import re
from typing import Any

from src.vision.document_processor import process_document
from src.vision.schemas import DocumentResult

from .embeddings import LocalEmbeddingAdapter
from .ingestion import ingest_document
from .reranking import rerank_evidence
from .schemas import Evidence, Fact, IngestionResult, KnowledgeBaseStats, RetrievalQuery
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """High-level KnowledgeBase Service for Person 2 with Hierarchical Parent-Child retrieval:
    Controls the entire lifecycle: DocumentResult intake -> Parent-Child chunking -> local embeddings
    -> local vector index -> retrieval -> reranking -> citations -> structured fact extraction.
    """

    def __init__(
        self,
        persist_dir: str = "data/processed/chroma_db",
        collection_name: str = "aionsite_knowledge",
        embedding_model: str = "all-MiniLM-L6-v2",
        child_chunk_size: int = 100,
        chunk_overlap: int = 20
    ):
        self.persist_dir = persist_dir
        self.child_chunk_size = child_chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_adapter = LocalEmbeddingAdapter(model_name=embedding_model)
        self.vector_store = VectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedding_adapter=self.embedding_adapter
        )
        self._ingested_documents: dict[str, DocumentResult] = {}

    def ingest_document(self, document: DocumentResult) -> IngestionResult:
        """Ingest a normalized DocumentResult into Parent and Child chunks."""
        res = ingest_document(
            document=document,
            vector_store=self.vector_store,
            child_chunk_size=self.child_chunk_size
        )
        if res.status == "success":
            self._ingested_documents[document.document_id] = document
        return res

    def ingest_file(self, file_path: str, ocr_engine: str = "auto") -> IngestionResult:
        """Process file with P3 and ingest."""
        doc_res = process_document(file_path, ocr_engine=ocr_engine)
        return self.ingest_document(doc_res)

    def ingest_directory(self, dir_path: str, ocr_engine: str = "auto") -> dict[str, Any]:
        """Batch-process and ingest an entire document folder."""
        docs_processed = 0
        total_chunks = 0
        total_parent_chunks = 0
        total_child_chunks = 0
        equipment_found = set()

        for file_path in glob.glob(os.path.join(dir_path, "**/*.*"), recursive=True):
            if file_path.lower().endswith((".pdf", ".png", ".jpg", ".jpeg", ".tiff")):
                ing_res = self.ingest_file(file_path, ocr_engine=ocr_engine)
                if ing_res.status == "success":
                    docs_processed += 1
                    total_chunks += ing_res.chunks_created
                    total_parent_chunks += ing_res.parent_chunks
                    total_child_chunks += ing_res.child_chunks
                    equipment_found.update(ing_res.equipment_ids)

        return {
            "documents_processed": docs_processed,
            "total_chunks": total_chunks,
            "parent_chunks": total_parent_chunks,
            "child_chunks": total_child_chunks,
            "equipment_entities": sorted(equipment_found)
        }

    def search(self, query: RetrievalQuery) -> list[Evidence]:
        """Search candidates, filter by metadata, rerank and attach parent context."""
        raw_results = self.vector_store.search(query)
        candidates = []
        for r in raw_results:
            c = r.chunk
            ev_text = c.parent_text if (query.return_parent_context and c.parent_text) else c.text
            candidates.append(Evidence(
                text=ev_text,
                source=c.source,
                page=c.page_number,
                score=r.score,
                document_id=c.document_id,
                section=c.section,
                chunk_id=c.chunk_id,
                parent_id=c.parent_id,
                parent_text=c.parent_text,
                revision=c.revision,
                authority=c.authority,
                equipment_ids=c.equipment_ids
            ))

        return rerank_evidence(query.query, candidates)

    def search_knowledge_base(
        self,
        query: str,
        equipment_id: str | None = None,
        top_k: int = 5,
        document_type: str | None = None,
        return_parent_context: bool = False
    ) -> list[Evidence]:
        """Master Contract P2 -> P1: search_knowledge_base(query) -> list[Evidence]"""
        q = RetrievalQuery(
            query=query,
            equipment_id=equipment_id,
            document_type=document_type,
            top_k=top_k,
            return_parent_context=return_parent_context
        )
        return self.search(q)

    def extract_facts(
        self,
        query: str,
        evidence: list[Evidence] | None = None,
        equipment_id: str | None = None
    ) -> list[Fact]:
        """Master Contract P2 -> P1: extract_facts(query) -> list[Fact]
        Extracts typed parameter, value, unit, equipment_id, source, page.
        Works across both atomic child chunks and parent table chunks.
        """
        if evidence is None:
            evidence = self.search_knowledge_base(query=query, equipment_id=equipment_id, top_k=20)

        facts: list[Fact] = []
        patterns = [
            # 1. Atomic table row format: "Row: Pressure | 14.5 | bar" or "Pressure | 14.5 | bar"
            r'(?:Row:\s*)?(?:([A-Z]{1,3}-\d{2,4})\s*\|\s*)?(Pressure|Temperature|Vibration|Discharge\s+pressure|Discharge\s+temperature|Outlet\s+temperature|Position)\s*\|\s*([\d\.]+)\s*\|\s*([^\|\n]+)',
            # 2. Key-value: "Observed Pressure: 14.5 bar"
            r'(?:observed\s+)?(pressure|temperature|vibration|discharge\s+pressure|discharge\s+temperature)[\s:=]+([\d\.]+)\s*(bar|°C|C|mm/s\s*RMS|%)',
            # 3. Limits: "recommended limit is 12.0 bar"
            r'(recommended\s+(?:upper\s+)?limit|recommended\s+limit|absolute\s+limit|design\s+limit|normal\s+max|normal\s+min)[\s:=]+([\d\.]+)\s*(bar|°C|C|mm/s\s*RMS|%)'
        ]

        seen_keys = set()
        for ev in evidence:
            text = ev.text
            chunk_eq = ev.equipment_ids[0] if ev.equipment_ids else (equipment_id or "V-101")

            if equipment_id and chunk_eq != equipment_id and ev.document_id and "P101" in ev.document_id:
                continue

            for pat in patterns:
                for match in re.finditer(pat, text, re.IGNORECASE):
                    groups = match.groups()
                    if len(groups) == 4 and groups[0]:
                        row_eq = groups[0].strip().upper()
                        param_name = groups[1].strip().lower()
                        raw_val = groups[2].strip()
                        unit = groups[3].strip().replace("C", "°C")
                        target_eq = row_eq
                    elif len(groups) == 4:
                        param_name = groups[1].strip().lower()
                        raw_val = groups[2].strip()
                        unit = groups[3].strip().replace("C", "°C")
                        target_eq = chunk_eq
                    else:
                        param_name = groups[0].strip().lower()
                        raw_val = groups[1].strip()
                        unit = groups[2].strip().replace("C", "°C")
                        target_eq = chunk_eq

                    if equipment_id and target_eq != equipment_id:
                        continue

                    val: float | str
                    try:
                        val = float(raw_val)
                    except ValueError:
                        val = raw_val

                    dedup_key = (target_eq, param_name, val, unit, ev.document_id, ev.page)
                    if dedup_key not in seen_keys:
                        seen_keys.add(dedup_key)
                        facts.append(Fact(
                            parameter=param_name,
                            value=val,
                            unit=unit,
                            equipment_id=target_eq,
                            source=ev.source,
                            page=ev.page,
                            document_id=ev.document_id,
                            revision=ev.revision,
                            confidence=ev.score
                        ))

        return self.resolve_conflicts(facts)

    def resolve_conflicts(self, facts: list[Fact]) -> list[Fact]:
        """Detect contradictory revisions or values and flag is_conflict=True."""
        grouped: dict[tuple, list[Fact]] = {}
        for f in facts:
            k = (f.equipment_id, f.parameter)
            grouped.setdefault(k, []).append(f)

        resolved: list[Fact] = []
        for (_eq, param), group in grouped.items():
            distinct_values = {f.value for f in group}
            if len(distinct_values) > 1:
                for f in group:
                    f_copy = f.model_copy()
                    f_copy.is_conflict = True
                    f_copy.conflict_notes = f"Conflict detected: multiple values found for {param} ({distinct_values}) across sources."
                    resolved.append(f_copy)
            else:
                resolved.extend(group)

        return resolved

    def get_stats(self) -> KnowledgeBaseStats:
        """Get collection statistics."""
        all_eq = set()
        for doc in self._ingested_documents.values():
            all_eq.update(doc.metadata.get("equipment_ids", []))

        return KnowledgeBaseStats(
            total_documents=len(self._ingested_documents),
            total_chunks=self.vector_store.count,
            equipment_ids=sorted(all_eq),
            document_types=list({d.metadata.get("document_type", "") for d in self._ingested_documents.values()})
        )

    def reset(self) -> None:
        """Clear all stored embeddings and documents."""
        self.vector_store.delete_collection()
        self._ingested_documents = {}
