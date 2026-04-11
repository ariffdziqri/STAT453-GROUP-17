# SEC Filing Analyzer — RAG-Based Chatbot for Financial Document Understanding
**STAT 453 · Spring 2026 · Group 17**

> A Retrieval-Augmented Generation (RAG) system that answers natural language questions about SEC 10-K filings, grounded in retrieved document evidence to reduce hallucination.

---

## Team

| Name | Role | Contact |
|---|---|---|
| Ariff Ismadi | Data ingestion pipeline | aismadi@wisc.edu |
| Jake Genewich | Retrieval system & ChromaDB | genewich@wisc.edu |
| Dinsabqi Nor Azmi | LLM integration & evaluation | binnorazmi@wisc.edu |
| Dylan Chapman | Frontend & final report | dmchapman2@wisc.edu |

---

## How It Works

```
SEC EDGAR API → Raw 10-K HTML → Section Parser → Chunker (~500 tokens)
                                                        ↓
                                               Embedding Model
                                          (all-MiniLM-L6-v2, local)
                                                        ↓
User Query → Embed Query → ChromaDB Similarity Search → Top-k Chunks
                                                        ↓
                                              Ollama LLM (llama3.2:3b)
                                                        ↓
                                          Answer with Citations
```

---

## Prerequisites

- Python 3.10+  (Anaconda recommended)
- [Ollama](https://ollama.com) — runs the LLM locally on your machine
- Mac, Linux, or Windows (Mac M-series runs fastest)

---

## Quick Start

### 1. Clone the repo

```bash
git clone <repo-url>
cd project
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install and start Ollama

```bash
# macOS
brew install ollama
brew services start ollama
ollama pull llama3.2:3b      # ~2GB download, one time only
```

> **Windows/Linux:** Download the installer from [ollama.com](https://ollama.com), then run `ollama pull llama3.2:3b` in a terminal.

### 4. Run the full pipeline

```bash
# Step 1 — fetch + parse + chunk 10-K filings (saves to data/)
python -m ingestion.pipeline

# Step 2 — embed chunks and store in ChromaDB (no API key needed)
python -m indexing.index_pipeline --backend local

# Step 3 — ask a question
python -m generation.rag_pipeline --query "What are Tesla's top risk factors?"
```

That's it. The `data/` folder is not in the repo — each teammate generates it locally by running the pipeline.

---

## Jupyter Notebook

Open `sec_rag_pipeline.ipynb` to run the full pipeline interactively with explanations at each step.

```bash
jupyter lab sec_rag_pipeline.ipynb
```

The notebook covers: ingestion → EDA → indexing → retrieval → generation.

---

## Usage Examples

### Command line

```bash
# Single question
python -m generation.rag_pipeline --query "What are Tesla's top risk factors?"

# Filter to a specific company and year
python -m generation.rag_pipeline --query "How did Apple explain revenue growth?" --ticker AAPL --year 2023

# Interactive chat loop
python -m generation.rag_pipeline
```

Inside the chat loop you can use `@` prefixes to filter:
```
Question: @TSLA @2023 What supply chain risks did Tesla mention?
Question: @AAPL How did Apple describe competition?
```

### Python API

```python
from generation.rag_pipeline import ask

response = ask(
    question = "What are NVIDIA's main risk factors?",
    ticker   = "NVDA",
    year     = 2023,
    top_k    = 5,
)
print(response["answer"])
print(response["sources"])   # list of retrieved chunks with metadata
```

---

## Project Structure

```
project/
├── config.py                  # companies, years, chunk settings, paths
├── requirements.txt
├── sec_rag_pipeline.ipynb     # full pipeline notebook (start here)
│
├── ingestion/                 # Ariff — SEC EDGAR fetching & parsing
│   ├── edgar_fetcher.py       # CIK lookup, filing download
│   ├── section_parser.py      # HTML → Item 1/1A/7/7A/8 text
│   ├── chunker.py             # token-aware chunking with metadata
│   └── pipeline.py            # CLI orchestrator
│
├── indexing/                  # Jake — embedding + ChromaDB
│   ├── embedder.py            # local (MiniLM) or OpenAI embeddings
│   ├── vector_store.py        # ChromaDB upsert + similarity search
│   └── index_pipeline.py     # CLI orchestrator
│
├── generation/                # Dinsabqi — LLM + RAG
│   ├── retriever.py           # embed query → ChromaDB search
│   ├── generator.py           # Ollama LLM with RAG prompt + citations
│   └── rag_pipeline.py        # end-to-end CLI + chat loop
│
└── data/                      # generated locally — NOT in repo
    ├── raw/                   # downloaded .htm filings
    ├── chunks/                # .jsonl chunk files
    └── chromadb/              # vector database
```

---

## Configuration

Edit `config.py` to change companies, years, or chunk settings:

```python
# Add or remove companies
COMPANIES = {
    "AAPL": "0000320193",
    "TSLA": "0001318605",
    # add more tickers + CIKs here
}

FILING_YEARS = [2022, 2023, 2024]  # years to fetch
CHUNK_SIZE   = 500                  # target tokens per chunk
TOP_K        = 5                    # chunks retrieved per query
```

To find a company's CIK, search at [www.sec.gov/cgi-bin/browse-edgar](https://www.sec.gov/cgi-bin/browse-edgar).

---

## Optional: OpenAI Embeddings

The default backend uses a free local model. For higher-quality embeddings (used in our final evaluation), you can switch to OpenAI:

```bash
export OPENAI_API_KEY=sk-...

# Re-index with OpenAI embeddings (--reset wipes the local collection first)
python -m indexing.index_pipeline --backend openai --reset
```

Cost: under $1 for the full corpus (~300k chunks).

> **Note:** You cannot mix local and OpenAI vectors in the same ChromaDB collection. Always use `--reset` when switching backends.

---

## Pipeline CLI Reference

```bash
# Ingestion
python -m ingestion.pipeline --tickers TSLA AAPL --years 2023
python -m ingestion.pipeline                        # all companies, all years

# Indexing
python -m indexing.index_pipeline --backend local   # free, no key
python -m indexing.index_pipeline --backend openai  # needs OPENAI_API_KEY
python -m indexing.index_pipeline --dry-run         # count chunks, no API calls
python -m indexing.index_pipeline --reset           # wipe + re-index

# Generation
python -m generation.rag_pipeline --query "..."
python -m generation.rag_pipeline --query "..." --ticker TSLA --year 2023 --top-k 5
python -m generation.rag_pipeline                   # interactive chat loop
```

---

## Data Sources

- **SEC EDGAR API** — free, public: [sec.gov/developer](https://www.sec.gov/developer)
- Filings: 10-K annual reports, 2022–2024
- Companies: AAPL, MSFT, GOOGL, NVDA, TSLA, AMZN, META, JNJ, XOM, PG
- Sections extracted: Business (Item 1), Risk Factors (Item 1A), MD&A (Item 7), Market Risk (Item 7A), Financial Statements (Item 8)
