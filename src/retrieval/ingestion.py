import logging
from src.vision.schemas import DocumentResult
from .schemas import IngestionResult
from .chunking import chunk_document
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


def ingest_document(
    document: DocumentResult,
    vector_store: VectorStore,
    child_chunk_size: int = 100,
    create_parent_chunks: bool = True
) -> IngestionResult:
    """Master Contract: Ingests normalized DocumentResult into VectorStore using Hierarchical Parent-Child chunking."""
    warnings = list(document.processing_warnings)
    if not document.pages:
        warnings.append("Document has no readable pages.")
        return IngestionResult(
            document_id=document.document_id,
            chunks_created=0,
            parent_chunks=0,
            child_chunks=0,
            equipment_ids=document.metadata.get("equipment_ids", []),
            status="empty",
            warnings=warnings
        )

    try:
        chunks = chunk_document(
            document=document,
            child_chunk_size=child_chunk_size,
            create_parent_chunks=create_parent_chunks
        )
        count = vector_store.add_chunks(chunks)

        parent_count = sum(1 for c in chunks if c.is_parent)
        child_count = sum(1 for c in chunks if not c.is_parent)

        all_equipment = document.metadata.get("equipment_ids", [])
        for c in chunks:
            all_equipment = sorted(list(set(all_equipment + c.equipment_ids)))

        return IngestionResult(
            document_id=document.document_id,
            chunks_created=count,
            parent_chunks=parent_count,
            child_chunks=child_count,
            equipment_ids=all_equipment,
            status="success",
            warnings=warnings
        )
    except Exception as e:
        logger.error(f"Failed to ingest document {document.document_id}: {e}")
        warnings.append(f"Ingestion exception: {str(e)}")
        return IngestionResult(
            document_id=document.document_id,
            chunks_created=0,
            parent_chunks=0,
            child_chunks=0,
            equipment_ids=[],
            status="failed",
            warnings=warnings
        )
