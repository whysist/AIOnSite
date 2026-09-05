import logging
import math
import os
import pickle
from typing import Any

from .embeddings import LocalEmbeddingAdapter, get_default_embedding_adapter
from .schemas import Chunk, RetrievalQuery, RetrievalResult

logger = logging.getLogger(__name__)


def _meta_str(value: Any) -> str:
    """Coerce one ChromaDB metadata value (``str | int | float | bool | None``,
    per its stub types) to a plain string. ``None`` becomes ``""`` rather
    than the literal ``"None"`` that a bare ``str(value)`` would produce.
    """
    if value is None:
        return ""
    return str(value)


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(x * y for x, y in zip(v1, v2, strict=False))
    norm1 = math.sqrt(sum(x * x for x in v1))
    norm2 = math.sqrt(sum(x * x for x in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


class VectorStore:
    """Persistent local index abstraction with disk persistence fallback."""

    def __init__(
        self,
        persist_dir: str = "data/processed/chroma_db",
        collection_name: str = "aionsite_knowledge",
        embedding_adapter: LocalEmbeddingAdapter | None = None
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_adapter = embedding_adapter or get_default_embedding_adapter()
        self.disk_fallback_file = os.path.join(self.persist_dir, f"{collection_name}_store.pkl")

        self.client = None
        self.collection = None
        self._fallback_records: list[dict[str, Any]] = []

        try:
            import chromadb
            os.makedirs(self.persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            logger.info(f"Initialized ChromaDB vector store at {self.persist_dir}")
        except Exception as e:
            logger.debug(f"ChromaDB not active ({e}). Using local persistent vector store.")
            self._load_fallback_from_disk()

    def _load_fallback_from_disk(self):
        """Loads cached fallback vector records from disk if available."""
        if os.path.exists(self.disk_fallback_file):
            try:
                with open(self.disk_fallback_file, "rb") as f:
                    self._fallback_records = pickle.load(f)
            except Exception as e:
                logger.debug(f"Failed to load fallback store: {e}")
                self._fallback_records = []

    def _save_fallback_to_disk(self):
        """Saves fallback records to disk for persistence across runs."""
        os.makedirs(os.path.dirname(self.disk_fallback_file), exist_ok=True)
        try:
            with open(self.disk_fallback_file, "wb") as f:
                pickle.dump(self._fallback_records, f)
        except Exception as e:
            logger.debug(f"Failed to save fallback store to disk: {e}")

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Add chunks to index with full metadata."""
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self.embedding_adapter.embed_texts(texts)
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "document_id": c.document_id,
                "source": c.source,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "section": c.section or "",
                "revision": c.revision or "",
                "authority": c.authority or "general",
                "equipment_ids": ",".join(c.equipment_ids),
                "document_type": c.document_type
            }
            for c in chunks
        ]

        if self.collection is not None:
            try:
                # chromadb's stub types are narrower unions than the plain
                # dict/float data we pass (and than its own runtime accepts);
                # verified working against a real ChromaDB collection.
                self.collection.upsert(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas,  # type: ignore[arg-type]
                    embeddings=embeddings,  # type: ignore[arg-type]
                )
                return len(chunks)
            except Exception as e:
                logger.warning(f"Chroma upsert error: {e}. Storing in memory fallback.")

        # In-Memory / File fallback store
        for i, chunk in enumerate(chunks):
            self._fallback_records = [r for r in self._fallback_records if r["chunk"].chunk_id != chunk.chunk_id]
            self._fallback_records.append({
                "chunk": chunk,
                "embedding": embeddings[i]
            })
        self._save_fallback_to_disk()
        return len(chunks)

    def search(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """Search candidates, filter by equipment/revision/source, rank and return."""
        query_vec = self.embedding_adapter.embed_query(query.query)
        top_k = query.top_k or 5

        # Chroma search
        if self.collection is not None:
            try:
                where_clause = {}
                if query.document_type:
                    where_clause["document_type"] = query.document_type
                if query.revision:
                    where_clause["revision"] = query.revision

                n_fetch = top_k * 5 if query.equipment_id else top_k
                results = self.collection.query(
                    query_embeddings=[query_vec],  # type: ignore[arg-type]
                    n_results=min(n_fetch, max(1, self.count)),
                    where=where_clause if where_clause else None,  # type: ignore[arg-type]
                )

                candidates: list[RetrievalResult] = []
                result_ids = results.get("ids")
                result_metadatas = results.get("metadatas")
                result_documents = results.get("documents")
                result_distances = results.get("distances")
                if result_ids and result_ids[0] and result_metadatas and result_documents:
                    for idx in range(len(result_ids[0])):
                        meta = result_metadatas[0][idx]
                        eq_str = _meta_str(meta.get("equipment_ids"))
                        eq_list = [e.strip() for e in eq_str.split(",") if e.strip()]

                        if query.equipment_id and query.equipment_id not in eq_list:
                            continue

                        chunk = Chunk(
                            chunk_id=result_ids[0][idx],
                            text=result_documents[0][idx],
                            document_id=_meta_str(meta.get("document_id")),
                            source=_meta_str(meta.get("source")),
                            page_number=int(_meta_str(meta.get("page_number")) or 1),
                            chunk_index=int(_meta_str(meta.get("chunk_index")) or 0),
                            section=_meta_str(meta.get("section")) or None,
                            revision=_meta_str(meta.get("revision")) or None,
                            authority=_meta_str(meta.get("authority")) or "general",
                            equipment_ids=eq_list,
                            document_type=_meta_str(meta.get("document_type")),
                        )
                        dist = (
                            float(result_distances[0][idx])
                            if result_distances and result_distances[0]
                            else 0.0
                        )
                        score = 1.0 / (1.0 + dist)
                        candidates.append(RetrievalResult(chunk=chunk, score=score))
                        if len(candidates) >= top_k:
                            break
                    return candidates
            except Exception as e:
                logger.debug(f"Chroma query failed ({e}), searching fallback store.")

        # In-Memory Cosine Similarity fallback
        scored = []
        for r in self._fallback_records:
            c: Chunk = r["chunk"]
            if query.equipment_id and query.equipment_id not in c.equipment_ids:
                continue
            if query.document_type and query.document_type != c.document_type:
                continue
            if query.revision and query.revision != c.revision:
                continue

            sim = _cosine_similarity(query_vec, r["embedding"])
            scored.append(RetrievalResult(chunk=c, score=sim))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def delete_collection(self) -> None:
        """Clear the collection."""
        if self.client is not None and self.collection is not None:
            try:
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.create_collection(self.collection_name)
            except Exception:
                pass
        self._fallback_records = []
        if os.path.exists(self.disk_fallback_file):
            try:
                os.remove(self.disk_fallback_file)
            except Exception:
                pass

    @property
    def count(self) -> int:
        if self.collection is not None:
            try:
                return self.collection.count()
            except Exception:
                pass
        return len(self._fallback_records)
