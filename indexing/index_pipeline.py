"""
index_pipeline.py
=================
Orchestrates the full indexing flow:

  data/chunks/*.jsonl
      → embedder.embed_chunks()       # call OpenAI text-embedding-3-small
      → vector_store.upsert_chunks()  # store in ChromaDB
      → print summary stats

Run from the project root:
    python -m indexing.index_pipeline                        # index everything
    python -m indexing.index_pipeline --tickers TSLA AAPL   # subset
    python -m indexing.index_pipeline --years 2023          # subset
    python -m indexing.index_pipeline --reset               # wipe + re-index
    python -m indexing.index_pipeline --dry-run             # count only, no API calls

Prerequisites:
    export OPENAI_API_KEY=sk-...
"""

import argparse
import glob
import json
import logging
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHUNKS_DIR, FILING_YEARS, COMPANIES
from indexing.embedder     import embed_chunks
from indexing.vector_store import get_collection, upsert_chunks, collection_stats


# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #

def _setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Load chunk files
# --------------------------------------------------------------------------- #

def load_chunks(tickers: list[str], years: list[int]) -> list[dict]:
    """
    Load all JSONL chunk files matching the given tickers and years.
    Returns a flat list of chunk dicts.
    """
    all_chunks = []
    pattern    = os.path.join(CHUNKS_DIR, "*.jsonl")
    files      = sorted(glob.glob(pattern))

    if not files:
        logger.error("No chunk files found in %s — run the ingestion pipeline first.", CHUNKS_DIR)
        return []

    for filepath in files:
        fname  = os.path.basename(filepath)             # e.g. TSLA_2023.jsonl
        parts  = fname.replace(".jsonl", "").split("_") # ["TSLA", "2023"]
        if len(parts) < 2:
            continue
        ticker, year_str = parts[0], parts[1]
        year = int(year_str)

        if ticker not in tickers or year not in years:
            continue

        with open(filepath, encoding="utf-8") as fh:
            file_chunks = [json.loads(line) for line in fh if line.strip()]
        logger.info("Loaded %d chunks from %s", len(file_chunks), fname)
        all_chunks.extend(file_chunks)

    return all_chunks


# --------------------------------------------------------------------------- #
#  Already-indexed check (skip chunks that are already in ChromaDB)
# --------------------------------------------------------------------------- #

def _already_indexed_ids(collection) -> set[str]:
    """Return the set of IDs already in the collection (up to 100k)."""
    count = collection.count()
    if count == 0:
        return set()
    result = collection.get(limit=min(count, 100_000), include=[])
    return set(result["ids"])


def _make_id(chunk: dict) -> str:
    return f"{chunk['ticker']}_{chunk['year']}_{chunk['section_key']}_{chunk['chunk_index']:04d}"


# --------------------------------------------------------------------------- #
#  Main indexing run
# --------------------------------------------------------------------------- #

def run(
    tickers: list[str],
    years:   list[int],
    reset:   bool = False,
    dry_run: bool = False,
    backend: str  = "local",
    debug:   bool = False,
):
    _setup_logging(debug)

    # --- Load chunks ---
    logger.info("Loading chunks for tickers=%s years=%s", tickers, years)
    chunks = load_chunks(tickers, years)

    if not chunks:
        logger.error("No chunks to index. Aborting.")
        return

    logger.info("Total chunks loaded: %d", len(chunks))

    if dry_run:
        print(f"\nDry run — would embed and index {len(chunks)} chunks. Exiting.")
        return

    # --- ChromaDB collection ---
    collection = get_collection(reset=reset)

    # Skip chunks already indexed (makes re-runs cheap)
    existing_ids = _already_indexed_ids(collection)
    new_chunks   = [c for c in chunks if _make_id(c) not in existing_ids]
    skipped      = len(chunks) - len(new_chunks)

    if skipped:
        logger.info("Skipping %d already-indexed chunks (%d new to embed)", skipped, len(new_chunks))

    if not new_chunks:
        logger.info("All chunks already indexed. Nothing to do.")
        _print_stats(collection)
        return

    # --- Embed in batches with a progress bar ---
    logger.info("Embedding %d chunks…", len(new_chunks))
    vectors = embed_chunks(new_chunks, backend=backend)

    # --- Upsert into ChromaDB ---
    logger.info("Upserting into ChromaDB…")
    upsert_chunks(collection, new_chunks, vectors)

    # --- Summary ---
    _print_stats(collection)


def _print_stats(collection):
    stats = collection_stats(collection)
    print("\n" + "=" * 56)
    print(f"  ChromaDB collection: '{collection.name}'")
    print(f"  Total vectors indexed: {stats['total_chunks']:,}")
    print(f"  Tickers:  {', '.join(stats.get('tickers', []))}")
    print(f"  Years:    {', '.join(str(y) for y in stats.get('years', []))}")
    print(f"  Sections: {len(stats.get('sections', []))}")
    print("=" * 56 + "\n")


# --------------------------------------------------------------------------- #
#  Quick query test (sanity check after indexing)
# --------------------------------------------------------------------------- #

def test_query(query_text: str = "What are Tesla's main risk factors?", top_k: int = 3):
    """
    Quick end-to-end sanity check: embed a query and retrieve top-k chunks.
    """
    from indexing.embedder     import embed_texts
    from indexing.vector_store import query as vstore_query

    print(f"\nTest query: '{query_text}'")
    print("-" * 56)

    collection   = get_collection()
    query_vector = embed_texts([query_text])[0]
    results      = vstore_query(collection, query_vector, top_k=top_k)

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r['ticker']} {r['year']} | {r['section_label']} | chunk {r['chunk_index']}")
        print(f"    distance: {r['distance']:.4f}")
        print(f"    {r['text'][:200]}…")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def _parse_args():
    parser = argparse.ArgumentParser(
        description="SEC 10-K indexing pipeline — embed chunks and store in ChromaDB"
    )
    parser.add_argument(
        "--tickers", nargs="+", default=list(COMPANIES.keys()), metavar="TICKER",
        help="Tickers to index (default: all in config.COMPANIES)",
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=FILING_YEARS, metavar="YEAR",
        help="Filing years to index (default: config.FILING_YEARS)",
    )
    parser.add_argument(
        "--backend", choices=["local", "openai"], default="local",
        help="Embedding backend: 'local' (free, no key) or 'openai' (needs OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Wipe the ChromaDB collection and re-index from scratch",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Count chunks that would be indexed without making API calls",
    )
    parser.add_argument(
        "--test-query", action="store_true",
        help="After indexing, run a test query to verify end-to-end retrieval",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        tickers = args.tickers,
        years   = args.years,
        reset   = args.reset,
        dry_run = args.dry_run,
        backend = args.backend,
        debug   = args.debug,
    )
    if args.test_query and not args.dry_run:
        test_query()
