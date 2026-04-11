"""
retriever.py
============
Embeds a user query and retrieves the top-k most relevant chunks
from ChromaDB.

Supports optional metadata filters so users can narrow results
to a specific company, year, or section before doing similarity search.

Usage:
    from generation.retriever import retrieve

    results = retrieve("What are Tesla's risk factors?", top_k=5)
    results = retrieve("Apple revenue 2023", filters={"ticker": "AAPL", "year": 2023})
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TOP_K
from indexing.embedder     import embed_texts
from indexing.vector_store import get_collection, query as chroma_query

logger = logging.getLogger(__name__)


def retrieve(
    query_text:  str,
    top_k:       int  = TOP_K,
    filters:     dict | None = None,
    backend:     str  = "local",
) -> list[dict]:
    """
    Embed `query_text` and return the top-k most similar chunks.

    Parameters
    ----------
    query_text : the user's natural language question
    top_k      : number of chunks to return
    filters    : optional ChromaDB `where` filter, e.g.:
                   {"ticker": "TSLA"}
                   {"year": {"$gte": 2023}}
                   {"$and": [{"ticker": "AAPL"}, {"section_key": "item_1a"}]}
    backend    : embedding backend — "local" or "openai" (must match what
                 was used during indexing)

    Returns
    -------
    list[dict] — ranked by similarity, each dict has:
        text, ticker, year, section_label, chunk_index, distance, …
    """
    logger.info("Retrieving top-%d chunks for: %r", top_k, query_text[:80])

    query_vector = embed_texts([query_text], backend=backend)[0]
    collection   = get_collection()

    if collection.count() == 0:
        raise RuntimeError(
            "ChromaDB collection is empty — run the indexing pipeline first:\n"
            "  python -m indexing.index_pipeline --backend local"
        )

    results = chroma_query(collection, query_vector, top_k=top_k, filters=filters)

    logger.info("Retrieved %d chunks (best distance: %.4f)", len(results),
                results[0]["distance"] if results else float("nan"))
    return results


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block for the LLM prompt.

    Each chunk is labelled with its source metadata so the LLM can cite it.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = (
            f"[{i}] {chunk['ticker']} {chunk['year']} | "
            f"{chunk['section_label']} (chunk {chunk['chunk_index']})"
        )
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)
