import hashlib
import logging
from typing import List

logger = logging.getLogger(__name__)


class LocalEmbeddingAdapter:
    """Provider-neutral local embedding interface."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._is_available = False
        self._dimension = 384
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self._is_available = True
            logger.info(f"Loaded local SentenceTransformer model: {model_name}")
        except Exception as e:
            logger.debug(f"SentenceTransformer not loaded ({e}). Using deterministic local embedding.")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings locally through a replaceable adapter."""
        if not texts:
            return []
        if self._is_available and self.model is not None:
            try:
                embeddings = self.model.encode(texts, show_progress_bar=False)
                return embeddings.tolist()
            except Exception as e:
                logger.warning(f"SentenceTransformer encode failed: {e}. Using fallback.")

        return [self._fallback_embed(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        """Embed a single search query."""
        if self._is_available and self.model is not None:
            try:
                return self.model.encode([query], show_progress_bar=False)[0].tolist()
            except Exception:
                pass
        return self._fallback_embed(query)

    def _fallback_embed(self, text: str) -> List[float]:
        """Deterministic, zero-network hash-based vector generator."""
        tokens = text.lower().split()
        vec = [0.0] * self._dimension
        for idx, token in enumerate(tokens):
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            pos = h % self._dimension
            weight = 1.0 / (1.0 + (idx * 0.1))
            vec[pos] += weight

        # Normalize
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def is_available(self) -> bool:
        return self._is_available


# Backward-compatible alias
EmbeddingEngine = LocalEmbeddingAdapter

# Module-level convenience functions
_default_adapter = None

def get_default_embedding_adapter() -> LocalEmbeddingAdapter:
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = LocalEmbeddingAdapter()
    return _default_adapter

def embed_texts(texts: List[str]) -> List[List[float]]:
    return get_default_embedding_adapter().embed_texts(texts)

def embed_query(query: str) -> List[float]:
    return get_default_embedding_adapter().embed_query(query)
