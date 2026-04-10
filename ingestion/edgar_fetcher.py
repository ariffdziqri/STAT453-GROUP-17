"""
edgar_fetcher.py
================
Handles all communication with the SEC EDGAR API.

Flow per company:
  1. get_cik(ticker)          — resolve ticker -> 10-digit CIK
  2. get_10k_filings(cik)     — list all 10-K filings with metadata
  3. filter_by_year(filings)  — keep only filings in FILING_YEARS
  4. download_filing(...)     — fetch + save the primary HTML document
"""

import os
import time
import logging
import requests

# resolve import whether run as module or from project root
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HEADERS, REQUEST_DELAY, RAW_DIR, FILING_YEARS

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Internal helpers
# --------------------------------------------------------------------------- #

def _get(url: str) -> requests.Response:
    """GET with required User-Agent header and polite rate limiting."""
    time.sleep(REQUEST_DELAY)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp


def _accession_nodash(accession: str) -> str:
    """'0001193125-22-039986'  ->  '000119312522039986'"""
    return accession.replace("-", "")


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #

def get_cik(ticker: str) -> str | None:
    """
    Resolve a stock ticker to a zero-padded 10-digit CIK string.
    Uses the official company_tickers.json mapping from EDGAR.

    Returns None if the ticker is not found.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    data = _get(url).json()
    for entry in data.values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    logger.warning("Ticker %s not found in company_tickers.json", ticker)
    return None


def get_10k_filings(cik: str) -> list[dict]:
    """
    Return a list of 10-K filing records for the given CIK.

    Each record contains:
      - accession_number  (original dashed format)
      - filing_date       (YYYY-MM-DD)
      - year              (int, fiscal year inferred from filing date)
      - primary_document  (filename of the main filing document)
      - document_url      (full download URL)

    EDGAR's submissions endpoint only returns the ~40 most recent filings in
    'recent'.  For older filings it paginates via 'files'; we handle that too.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = _get(url).json()

    filings = []
    filings.extend(_extract_filings(cik, data["filings"]["recent"]))

    # Handle pagination for companies with many historical filings
    for extra in data["filings"].get("files", []):
        page_url = f"https://data.sec.gov/submissions/{extra['name']}"
        page_data = _get(page_url).json()
        filings.extend(_extract_filings(cik, page_data))

    return filings


def _extract_filings(cik: str, recent: dict) -> list[dict]:
    """
    Parse a 'recent' block from the submissions JSON into a list of filing dicts.
    Skips any entries that are not form type '10-K'.
    """
    results = []
    forms      = recent.get("form", [])
    dates      = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primaries  = recent.get("primaryDocument", [])

    for form, date, acc, primary in zip(forms, dates, accessions, primaries):
        if form != "10-K":
            continue
        year = int(date[:4])
        nodash = _accession_nodash(acc)
        cik_int = int(cik)  # drop leading zeros for the archive path
        doc_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_int}/{nodash}/{primary}"
        )
        results.append({
            "accession_number": acc,
            "filing_date":      date,
            "year":             year,
            "primary_document": primary,
            "document_url":     doc_url,
        })
    return results


def filter_by_year(filings: list[dict], years: list[int] = None) -> list[dict]:
    """Keep only filings whose filing_date falls in the target years."""
    if years is None:
        years = FILING_YEARS
    return [f for f in filings if f["year"] in years]


def download_filing(ticker: str, cik: str, filing: dict) -> str | None:
    """
    Download the primary HTML document for a filing and save it locally.

    Saved path:  data/raw/{TICKER}_{YEAR}_{accession_nodash}.htm

    Returns the local file path, or None if the download fails.
    Skips download if the file already exists (idempotent).
    """
    os.makedirs(RAW_DIR, exist_ok=True)

    nodash   = _accession_nodash(filing["accession_number"])
    filename = f"{ticker}_{filing['year']}_{nodash}.htm"
    filepath = os.path.join(RAW_DIR, filename)

    if os.path.exists(filepath):
        logger.info("Already downloaded: %s", filename)
        return filepath

    logger.info("Downloading %s %s (%s)…", ticker, filing["year"], filing["primary_document"])
    try:
        resp = _get(filing["document_url"])
    except requests.HTTPError as exc:
        logger.error("HTTP error for %s: %s", filing["document_url"], exc)
        return None

    with open(filepath, "w", encoding="utf-8", errors="replace") as fh:
        fh.write(resp.text)

    logger.info("Saved → %s", filepath)
    return filepath


# --------------------------------------------------------------------------- #
#  Convenience: fetch everything for a single ticker
# --------------------------------------------------------------------------- #

def fetch_ticker(ticker: str, years: list[int] = None) -> list[dict]:
    """
    High-level helper: resolve ticker, find 10-K filings in `years`,
    download each one, and return a list of records augmented with
    `local_path`.

    Example:
        records = fetch_ticker("TSLA", years=[2023])
        # records[0]["local_path"] -> "data/raw/TSLA_2023_000095017024017585.htm"
    """
    if years is None:
        years = FILING_YEARS

    cik = get_cik(ticker)
    if cik is None:
        return []

    all_filings  = get_10k_filings(cik)
    target       = filter_by_year(all_filings, years)

    if not target:
        logger.warning("No 10-K filings found for %s in years %s", ticker, years)

    results = []
    for filing in target:
        path = download_filing(ticker, cik, filing)
        results.append({**filing, "ticker": ticker, "cik": cik, "local_path": path})

    return results
