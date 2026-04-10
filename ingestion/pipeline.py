"""
pipeline.py
===========
Orchestrates the full ingestion flow:

  Ticker list
      → edgar_fetcher.fetch_ticker()     # download raw HTML to data/raw/
      → section_parser.parse_file()      # extract Item 1, 1A, 7, 7A, 8
      → chunker.chunk_filing()           # split into ~500-token chunks
      → write JSONL to data/chunks/      # one file per (ticker, year)

Run from the project root:
    python -m ingestion.pipeline                         # all tickers, all years
    python -m ingestion.pipeline --tickers TSLA AAPL    # subset of tickers
    python -m ingestion.pipeline --years 2023 2024      # subset of years
    python -m ingestion.pipeline --tickers TSLA --years 2023 --debug
"""

import argparse
import json
import logging
import os
import sys

from tqdm import tqdm

# Allow running as script or module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COMPANIES, FILING_YEARS, CHUNKS_DIR
from ingestion.edgar_fetcher  import fetch_ticker
from ingestion.section_parser import parse_file
from ingestion.chunker        import chunk_filing


# --------------------------------------------------------------------------- #
#  Logging setup
# --------------------------------------------------------------------------- #

def _setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )
    # Suppress noisy HTTP logs unless in debug mode
    if not debug:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Per-filing ingestion
# --------------------------------------------------------------------------- #

def ingest_filing(record: dict) -> list[dict]:
    """
    Parse and chunk a single downloaded filing.

    Parameters
    ----------
    record : dict
        A record returned by edgar_fetcher.fetch_ticker(), expected to have:
        local_path, ticker, cik, year.

    Returns
    -------
    list[dict]  — chunk dicts ready for writing / embedding
    """
    local_path = record.get("local_path")
    if not local_path or not os.path.exists(local_path):
        logger.warning("Skipping missing file: %s", local_path)
        return []

    ticker = record["ticker"]
    cik    = record["cik"]
    year   = record["year"]

    logger.info("Parsing %s %d…", ticker, year)
    sections = parse_file(local_path)

    if not sections:
        logger.warning("No sections extracted from %s %d", ticker, year)
        return []

    chunks = chunk_filing(sections, ticker=ticker, cik=cik, year=year)
    logger.info("  → %d chunks from %d sections", len(chunks), len(sections))
    return chunks


# --------------------------------------------------------------------------- #
#  Output writer
# --------------------------------------------------------------------------- #

def save_chunks(chunks: list[dict], ticker: str, year: int) -> str:
    """
    Write chunks to a JSONL file at data/chunks/{TICKER}_{YEAR}.jsonl.
    Overwrites if the file already exists (idempotent re-runs).

    Returns the output file path.
    """
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    outpath = os.path.join(CHUNKS_DIR, f"{ticker}_{year}.jsonl")
    with open(outpath, "w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    logger.info("Wrote %d chunks → %s", len(chunks), outpath)
    return outpath


# --------------------------------------------------------------------------- #
#  Summary stats
# --------------------------------------------------------------------------- #

def _print_summary(all_chunks: list[dict]):
    if not all_chunks:
        print("\nNo chunks produced.")
        return

    total  = len(all_chunks)
    tokens = sum(c["token_count"] for c in all_chunks)
    by_ticker: dict[str, int] = {}
    by_section: dict[str, int] = {}

    for c in all_chunks:
        by_ticker[c["ticker"]] = by_ticker.get(c["ticker"], 0) + 1
        label = c["section_label"]
        by_section[label] = by_section.get(label, 0) + 1

    print("\n" + "=" * 56)
    print(f"  Ingestion complete: {total:,} chunks  |  {tokens:,} tokens")
    print("=" * 56)
    print("\nChunks per ticker:")
    for t, n in sorted(by_ticker.items()):
        print(f"  {t:<8}  {n:>5}")
    print("\nChunks per section:")
    for s, n in sorted(by_section.items(), key=lambda x: -x[1]):
        print(f"  {s:<45}  {n:>5}")
    print()


# --------------------------------------------------------------------------- #
#  Main pipeline
# --------------------------------------------------------------------------- #

def run(tickers: list[str], years: list[int], debug: bool = False):
    _setup_logging(debug)

    all_chunks: list[dict] = []
    ticker_year_pairs = [(t, y) for t in tickers for y in years]

    for ticker, year in tqdm(ticker_year_pairs, desc="Tickers × Years", unit="filing"):
        logger.info("─── %s  %d ────────────────────────────────", ticker, year)

        # 1. Fetch (downloads if not already cached)
        records = fetch_ticker(ticker, years=[year])

        # 2. Parse + chunk
        for record in records:
            chunks = ingest_filing(record)
            if chunks:
                save_chunks(chunks, ticker=ticker, year=year)
                all_chunks.extend(chunks)

    _print_summary(all_chunks)
    return all_chunks


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def _parse_args():
    parser = argparse.ArgumentParser(
        description="SEC 10-K ingestion pipeline — fetch, parse, chunk"
    )
    parser.add_argument(
        "--tickers", nargs="+", default=list(COMPANIES.keys()),
        metavar="TICKER",
        help="Tickers to ingest (default: all in config.COMPANIES)",
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=FILING_YEARS,
        metavar="YEAR",
        help="Filing years to ingest (default: config.FILING_YEARS)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(tickers=args.tickers, years=args.years, debug=args.debug)
