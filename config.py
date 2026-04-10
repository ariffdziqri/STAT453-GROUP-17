# config.py — central settings for the SEC RAG pipeline

# Companies to ingest: ticker -> CIK (10-digit zero-padded)
COMPANIES = {
    "AAPL":  "0000320193",
    "MSFT":  "0000789019",
    "GOOGL": "0001652044",
    "NVDA":  "0001045810",
    "TSLA":  "0001318605",
    "AMZN":  "0001018724",
    "META":  "0001326801",
    "JNJ":   "0000200406",
    "XOM":   "0000034088",
    "PG":    "0000080424",
}

# Filing years to fetch
FILING_YEARS = [2022, 2023, 2024]

# 10-K sections to extract  (item_key -> human label)
TARGET_SECTIONS = {
    "item_1":   "Business",
    "item_1a":  "Risk Factors",
    "item_7":   "Management Discussion and Analysis",
    "item_7a":  "Quantitative and Qualitative Disclosures About Market Risk",
    "item_8":   "Financial Statements",
}

# Chunking
CHUNK_SIZE    = 500   # target tokens per chunk
CHUNK_OVERLAP = 50    # overlap between consecutive chunks

# Retrieval
TOP_K = 5

# HTTP headers required by SEC EDGAR (https://www.sec.gov/developer)
HEADERS = {"User-Agent": "Group17 aismadi@wisc.edu"}

# Rate limit: SEC asks for ≤10 req/s; we use 0.15 s between calls to be safe
REQUEST_DELAY = 0.15  # seconds

# Directories
import os
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")     # downloaded HTML filings
CHUNKS_DIR = os.path.join(BASE_DIR, "data", "chunks")  # JSONL chunk files
