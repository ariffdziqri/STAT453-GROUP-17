"""
vector_store.py
===============
Thin wrapper around ChromaDB that handles:
  - creating / loading a persistent collection
  - upserting chunks + embeddings + metadata
  - similarity search with optional metadata filters

ChromaDB stores everything locally under data/chromadb/ so there is
no external service to run.

Usage:
    from indexing.vector_store import get_collection, upsert_chunks, query

    col = get_collection()
    upsert_chunks(col, chunks, vectors)
    results = query(col, query_vector, top_k=5, filters={"ticker": "TSLA"})
"""

import logging
import os
import sys

import chromadb
from chromadb import Collection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHROMA_DIR, CHROMA_COLLECTION, CHROMA_DISTANCE, TOP_K

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Fields from chunk dicts that we store as ChromaDB metadata
#  (ChromaDB metadata values must be str / int / float / bool)
# --------------------------------------------------------------------------- #
_META_FIELDS = ["ticker", "cik", "year", "section_key", "section_label",
                "chunk_index", "total_chunks", "token_count"]


# --------------------------------------------------------------------------- #
#  Collection management
# --------------------------------------------------------------------------- #

def get_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client backed by data/chromadb/."""
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_collection(reset: bool = False) -> Collection:
    """
    Get (or create) the SEC filings collection.

    Parameters
    ----------
    reset : bool
        If True, delete and recreate the collection — useful for a full
        re-index from scratch.  Default False (safe for incremental updates).
    """
    client = get_client()

    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION)
            logger.info("Deleted existing collection '%s'", CHROMA_COLLECTION)
        except Exception:
            pass  # didn't exist yet

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": CHROMA_DISTANCE},
    )
    logger.info(
        "Collection '%s' ready — %d documents currently indexed",
        CHROMA_COLLECTION, collection.count(),
    )
    return collection


# --------------------------------------------------------------------------- #
#  Upsert
# --------------------------------------------------------------------------- #

def _make_id(chunk: dict) -> str:
    """
    Build a stable, unique string ID for a chunk.
    Format:  {TICKER}_{YEAR}_{section_key}_{chunk_index}
    """
    return f"{chunk['ticker']}_{chunk['year']}_{chunk['section_key']}_{chunk['chunk_index']:04d}"


def upsert_chunks(
    collection: Collection,
    chunks: list[dict],
    vectors: list[list[float]],
) -> None:
    """
    Insert or update chunks in ChromaDB.

    Upsert (not insert) means re-running the pipeline is safe — existing
    documents with the same ID are overwritten rather than duplicated.

    Parameters
    ----------
    collection : ChromaDB Collection object
    chunks     : list of chunk dicts (from chunker.py)
    vectors    : list of embedding vectors — must be same length as chunks
    """
    if len(chunks) != len(vectors):
        raise ValueError(f"chunks ({len(chunks)}) and vectors ({len(vectors)}) must have the same length")

    ids        = [_make_id(c) for c in chunks]
    documents  = [c["text"] for c in chunks]
    metadatas  = [{k: c[k] for k in _META_FIELDS if k in c} for c in chunks]

    # ChromaDB upsert in batches of 500 to avoid memory spikes on large corpora
    batch_size = 500
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.upsert(
            ids        = ids[start:end],
            embeddings = vectors[start:end],
            documents  = documents[start:end],
            metadatas  = metadatas[start:end],
        )
        logger.debug("Upserted %d–%d", start, min(end, len(chunks)))

    logger.info("Upserted %d chunks → collection now has %d total", len(chunks), collection.count())


# --------------------------------------------------------------------------- #
#  Query
# --------------------------------------------------------------------------- #

def query(
    collection: Collection,
    query_vector: list[float],
    top_k: int = TOP_K,
    filters: dict | None = None,
) -> list[dict]:
    """
    Retrieve the top-k most similar chunks for a query embedding.

    Parameters
    ----------
    collection   : ChromaDB Collection
    query_vector : embedding of the user query
    top_k        : number of results to return
    filters      : optional ChromaDB `where` clause to restrict results,
                   e.g. {"ticker": "TSLA"} or {"year": {"$gte": 2023}}

    Returns
    -------
    list[dict] — each dict has keys:
        id, text, distance, ticker, year, section_label, chunk_index, …
    """
    kwargs: dict = {"query_embeddings": [query_vector], "n_results": top_k,
                    "include": ["documents", "metadatas", "distances"]}
    if filters:
        kwargs["where"] = filters

    results = collection.query(**kwargs)

    # Unpack the nested lists ChromaDB returns (one list per query)
    ids        = results["ids"][0]
    documents  = results["documents"][0]
    metadatas  = results["metadatas"][0]
    distances  = results["distances"][0]

    output = []
    for doc_id, text, meta, dist in zip(ids, documents, metadatas, distances):
        output.append({"id": doc_id, "text": text, "distance": dist, **meta})

    return output


def collection_stats(collection: Collection) -> dict:
    """Return a quick summary dict for logging / EDA."""
    count = collection.count()
    if count == 0:
        return {"total_chunks": 0}

    # Sample up to 5000 items to get metadata distributions
    sample = collection.get(limit=min(count, 5000), include=["metadatas"])
    metas  = sample["metadatas"]

    tickers  = sorted(set(m["ticker"]  for m in metas))
    years    = sorted(set(m["year"]    for m in metas))
    sections = sorted(set(m["section_label"] for m in metas))

    return {
        "total_chunks": count,
        "tickers":      tickers,
        "years":        years,
        "sections":     sections,
    }
