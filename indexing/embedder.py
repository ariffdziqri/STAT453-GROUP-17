"""
embedder.py
===========
Generates dense vector embeddings for text chunks.

Supports two backends, switchable via the `backend` parameter:

  "openai"  — OpenAI text-embedding-3-small (1536-dim, requires OPENAI_API_KEY)
  "local"   — sentence-transformers/all-MiniLM-L6-v2 (384-dim, free, runs on CPU)

The local backend is the default so the pipeline works out-of-the-box
without any API key.  Switch to "openai" when you have a key and want
higher-quality embeddings for the final evaluation.

NOTE: You cannot mix vectors from different backends in the same ChromaDB
collection — the dimensions differ (384 vs 1536).  Use --reset when
switching backends.

Usage:
    from indexing.embedder import embed_chunks

    # free, local
    vectors = embed_chunks(chunks, backend="local")

    # OpenAI (needs OPENAI_API_KEY)
    vectors = embed_chunks(chunks, backend="openai")
"""

import logging
import os
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_MODEL, EMBEDDING_MODEL_OS, EMBED_BATCH_SIZE

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  OpenAI backend
# --------------------------------------------------------------------------- #

_openai_client = None

def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set.\n"
                "Either export it:  export OPENAI_API_KEY=sk-...\n"
                "Or use the free local backend:  --backend local"
            )
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _embed_openai(texts: list[str], batch_size: int) -> list[list[float]]:
    client  = _get_openai_client()
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        logger.debug("OpenAI embedding batch %d–%d / %d", start, start + len(batch), len(texts))
        for attempt in range(5):
            try:
                response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
                batch_vecs = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
                vectors.extend(batch_vecs)
                break
            except Exception as exc:
                wait = 2 ** attempt
                logger.warning("OpenAI API error (attempt %d/5): %s — retrying in %ds", attempt + 1, exc, wait)
                time.sleep(wait)
        else:
            raise RuntimeError(f"OpenAI embedding failed for batch at index {start}")
    return vectors


# --------------------------------------------------------------------------- #
#  Local backend (sentence-transformers)
# --------------------------------------------------------------------------- #

_local_model = None

def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading local embedding model '%s' (first run downloads ~90MB)…", EMBEDDING_MODEL_OS)
        _local_model = SentenceTransformer(EMBEDDING_MODEL_OS)
        logger.info("Local model loaded.")
    return _local_model


def _embed_local(texts: list[str], batch_size: int) -> list[list[float]]:
    model = _get_local_model()
    logger.debug("Local embedding %d texts (batch_size=%d)…", len(texts), batch_size)
    # encode() handles batching internally and returns a numpy array
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 200,
        convert_to_numpy=True,
    )
    return vectors.tolist()


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #

def embed_texts(
    texts:      list[str],
    backend:    str = "local",
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[list[float]]:
    """
    Embed a list of strings and return a list of float vectors.

    Parameters
    ----------
    texts      : strings to embed
    backend    : "local" (free, no key) or "openai" (better quality, needs key)
    batch_size : chunks per batch

    Returns
    -------
    list[list[float]] — one vector per input string, same order.
    """
    if backend == "openai":
        logger.info("Backend: OpenAI %s (%d texts)", EMBEDDING_MODEL, len(texts))
        return _embed_openai(texts, batch_size)
    elif backend == "local":
        logger.info("Backend: local %s (%d texts)", EMBEDDING_MODEL_OS, len(texts))
        return _embed_local(texts, batch_size)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'local' or 'openai'.")


def embed_chunks(
    chunks:     list[dict],
    backend:    str = "local",
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[list[float]]:
    """
    Convenience wrapper: embed the 'text' field of each chunk dict.
    """
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts, backend=backend, batch_size=batch_size)
    logger.info("Done — %d vectors, dim=%d", len(vectors), len(vectors[0]) if vectors else 0)
    return vectors
