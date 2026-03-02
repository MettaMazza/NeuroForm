"""
Tier 2: Vector Store — Semantic Memory
=========================================

In-memory vector store with Ollama embeddings for semantic similarity search.
Ported from ErnOS V3 vector.py.

Stores conversation snippets as embeddings and retrieves the most relevant
ones via cosine similarity. This gives the bot "associative memory" — it can
recall related conversations even if they don't share exact keywords.
"""
import logging
import math
import time
from typing import List, Dict, Any, Optional

try:
    import ollama
except ImportError:
    ollama = None  # type: ignore

logger = logging.getLogger(__name__)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorEntry:
    """A single vector memory entry."""
    __slots__ = ("text", "embedding", "metadata", "timestamp", "valid")

    def __init__(self, text: str, embedding: List[float],
                 metadata: Optional[Dict[str, Any]] = None):
        self.text = text
        self.embedding = embedding
        self.metadata = metadata or {}
        self.timestamp = time.time()
        self.valid = True


class VectorStore:
    """
    In-memory vector store with semantic search.

    - Stores text + embeddings with scope/user metadata
    - Cosine similarity retrieval
    - Invalidation by content (for reconciliation)
    """

    # Default embedding model — must support /api/embed endpoint
    DEFAULT_EMBED_MODEL = "nomic-embed-text"

    def __init__(self, model: str = "gemma3:4b", max_entries: int = 5000,
                 embed_model: str = ""):
        self.model = model
        self.embed_model = embed_model or self.DEFAULT_EMBED_MODEL
        self.max_entries = max_entries
        self.entries: List[VectorEntry] = []
        logger.info(f"VectorStore initialized (embed_model={self.embed_model}, max={max_entries})")

    def embed(self, text: str) -> List[float]:
        """Generate embedding via Ollama."""
        try:
            if ollama is None:
                logger.warning("ollama not installed")
                return []
            result = ollama.embed(model=self.embed_model, input=text)
            # Handle both old and new ollama API
            if isinstance(result, dict):
                embeddings = result.get("embeddings", result.get("embedding", []))
                if embeddings and isinstance(embeddings[0], list):
                    return embeddings[0]
                return embeddings
            return []
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return []

    def store(self, text: str, user_id: str = "", scope: str = "PUBLIC",
              metadata: Optional[Dict[str, Any]] = None):
        """Embed and store a text snippet."""
        if not text or len(text.strip()) < 10:
            return  # Skip trivial content

        embedding = self.embed(text)
        if not embedding:
            return

        entry_meta = {"user_id": user_id, "scope": scope}
        if metadata:
            entry_meta.update(metadata)

        self.entries.append(VectorEntry(text, embedding, entry_meta))

        # Evict oldest if over capacity
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

        logger.debug(f"Stored vector: {text[:50]}... ({len(self.entries)} total)")

    def retrieve(self, query: str, scope: str = "PUBLIC",
                 user_id: Optional[str] = None,
                 max_results: int = 5,
                 min_score: float = 0.3) -> List[Dict[str, Any]]:
        """
        Find the most relevant memories for a query.

        Returns list of {text, score, metadata} dicts.
        """
        if not self.entries:
            return []

        query_vec = self.embed(query)
        if not query_vec:
            return []

        scored = []
        for entry in self.entries:
            if not entry.valid:
                continue
            # Scope filtering
            entry_scope = entry.metadata.get("scope", "PUBLIC")
            if scope == "PUBLIC" and entry_scope != "PUBLIC":
                continue
            # User filtering (optional)
            if user_id and entry.metadata.get("user_id"):
                entry_user = entry.metadata.get("user_id", "")
                if entry_user and entry_user != str(user_id) and entry_scope != "PUBLIC":
                    continue

            score = cosine_similarity(query_vec, entry.embedding)
            if score >= min_score:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, entry in scored[:max_results]:
            results.append({
                "text": entry.text,
                "score": score,
                "metadata": entry.metadata,
            })

        return results

    def invalidate_by_content(self, keywords: List[str],
                               reason: str = ""):
        """Invalidate entries containing any of the keywords."""
        count = 0
        for entry in self.entries:
            for kw in keywords:
                if kw.lower() in entry.text.lower():
                    entry.valid = False
                    count += 1
                    break
        if count:
            logger.info(f"Invalidated {count} vector entries ({reason})")

    def snapshot(self) -> Dict[str, Any]:
        """Diagnostic snapshot."""
        valid = sum(1 for e in self.entries if e.valid)
        return {
            "total_entries": len(self.entries),
            "valid_entries": valid,
            "model": self.model,
        }
