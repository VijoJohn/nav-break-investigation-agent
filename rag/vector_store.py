"""Knowledge index over rule-tagged chunks with a graceful fallback.

Primary path: embed each knowledge chunk with sentence-transformers and index
it in an in-memory Qdrant collection (one point per chunk, payload carries the
rule id and text). Retrieval embeds the query and returns the nearest chunks.

Fallback path: if embeddings or Qdrant are unavailable (no model download, no
package, offline CI), the index degrades to transparent lexical scoring over
the same chunks so the agent and the eval harness still run end to end.
"""

import re

from rag.knowledge_loader import load_chunks

VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output dimension
COLLECTION = "nav_knowledge"

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text):
    return set(_WORD.findall(text.lower()))


class KnowledgeIndex:
    """Retrieve rule-tagged knowledge chunks relevant to a NAV break query."""

    def __init__(self):
        self.chunks = load_chunks()
        self.backend = "lexical"
        self._embed = None
        self._client = None
        self._try_build_vector_backend()

    # ------------------------------------------------------------------
    # Backend setup
    # ------------------------------------------------------------------
    def _try_build_vector_backend(self):
        try:
            from rag.embedding_model import embed_text
            from qdrant_client import QdrantClient
            from qdrant_client.models import (
                VectorParams,
                Distance,
                PointStruct,
            )

            client = QdrantClient(":memory:")
            client.recreate_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

            points = []
            for i, chunk in enumerate(self.chunks):
                vector = embed_text(chunk["text"])
                points.append(
                    PointStruct(id=i, vector=vector, payload=chunk)
                )
            client.upsert(collection_name=COLLECTION, points=points)

            self._embed = embed_text
            self._client = client
            self.backend = "vector"
        except Exception as exc:  # pragma: no cover - depends on environment
            # Stay on the lexical fallback; surface the reason for debugging.
            self.backend = "lexical"
            self._fallback_reason = str(exc)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query, k=3):
        """Return up to k chunk payloads most relevant to the query."""
        if self.backend == "vector":
            response = self._client.query_points(
                collection_name=COLLECTION,
                query=self._embed(query),
                limit=k,
                with_payload=True,
            )
            return [p.payload for p in response.points]
        return self._lexical_retrieve(query, k)

    def _lexical_retrieve(self, query, k):
        q = _tokens(query)
        scored = []
        for chunk in self.chunks:
            overlap = len(q & _tokens(chunk["text"]))
            if overlap:
                scored.append((overlap, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:k]]


# Module-level singleton so the index is built once per process.
_INDEX = None


def get_index():
    global _INDEX
    if _INDEX is None:
        _INDEX = KnowledgeIndex()
    return _INDEX
