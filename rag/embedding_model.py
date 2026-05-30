"""Sentence-transformers embedding wrapper (lazy-loaded).

The model is loaded on first use rather than at import time so that modules
which only need the lexical fallback (or run in an environment without the
model cached) can still import the package without triggering a download.
"""

_model = None

MODEL_NAME = "all-MiniLM-L6-v2"


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_text(text):
    return _get_model().encode(text).tolist()
