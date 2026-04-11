"""
rag_pipeline.py
===============
End-to-end RAG: user question → retrieve → generate → answer with citations.

Can be used as a module or run directly as a CLI chatbot.

    python -m generation.rag_pipeline
    python -m generation.rag_pipeline --query "What are Tesla's top risk factors?"
    python -m generation.rag_pipeline --query "..." --ticker TSLA --year 2023
    python -m generation.rag_pipeline --query "..." --model llama3.2:3b --top-k 5
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TOP_K
from generation.retriever import retrieve, format_context
from generation.generator import generate, list_available_models, DEFAULT_MODEL

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Core RAG function
# --------------------------------------------------------------------------- #

def ask(
    question:   str,
    top_k:      int  = TOP_K,
    ticker:     str | None = None,
    year:       int | None = None,
    section:    str | None = None,
    model:      str  = DEFAULT_MODEL,
    backend:    str  = "local",
    stream:     bool = True,
    verbose:    bool = False,
) -> dict:
    """
    Full RAG pipeline: retrieve relevant chunks then generate a grounded answer.

    Parameters
    ----------
    question  : natural language question
    top_k     : number of chunks to retrieve
    ticker    : optional — restrict retrieval to one company (e.g. "TSLA")
    year      : optional — restrict retrieval to one year (e.g. 2023)
    section   : optional — restrict to a section key (e.g. "item_1a")
    model     : Ollama model name
    backend   : embedding backend ("local" or "openai")
    stream    : stream answer tokens to stdout as they arrive
    verbose   : if True, print retrieved chunks before the answer

    Returns
    -------
    dict with keys:
        question   : str
        answer     : str
        sources    : list[dict]  — the retrieved chunks used as context
    """
    # --- Build optional metadata filter ---
    where_clauses = []
    if ticker:
        where_clauses.append({"ticker": ticker})
    if year:
        where_clauses.append({"year": year})
    if section:
        where_clauses.append({"section_key": section})

    filters = None
    if len(where_clauses) == 1:
        filters = where_clauses[0]
    elif len(where_clauses) > 1:
        filters = {"$and": where_clauses}

    # --- Retrieve ---
    chunks  = retrieve(question, top_k=top_k, filters=filters, backend=backend)
    context = format_context(chunks)

    if verbose:
        print("\n" + "=" * 60)
        print("RETRIEVED CONTEXT")
        print("=" * 60)
        print(context)
        print("=" * 60 + "\n")

    # --- Generate ---
    if stream:
        print("\nAnswer:\n")

    answer = generate(question, context, model=model, stream=stream)

    if not stream:
        print(f"\nAnswer:\n{answer}")

    # --- Print sources ---
    print("\nSources:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}] {chunk['ticker']} {chunk['year']} | "
              f"{chunk['section_label']} | chunk {chunk['chunk_index']} "
              f"(distance: {chunk['distance']:.4f})")

    return {"question": question, "answer": answer, "sources": chunks}


# --------------------------------------------------------------------------- #
#  Interactive chatbot loop
# --------------------------------------------------------------------------- #

def chat_loop(model: str = DEFAULT_MODEL, backend: str = "local", top_k: int = TOP_K):
    """Simple REPL for interactive Q&A."""
    print(f"\nSEC 10-K RAG Chatbot  |  model: {model}  |  embeddings: {backend}")
    print("Type your question and press Enter. Type 'quit' to exit.")
    print("Optional prefixes:  @TSLA  @2023  @risk  (to filter by ticker/year/section)")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input or user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        # Parse optional @ prefixes:  @TSLA @2023 @item_1a
        ticker, year, section = None, None, None
        tokens = user_input.split()
        remaining = []
        for token in tokens:
            if token.startswith("@"):
                tag = token[1:].upper()
                if tag.isdigit() and len(tag) == 4:
                    year = int(tag)
                elif tag.startswith("ITEM"):
                    section = tag.lower()
                else:
                    ticker = tag
            else:
                remaining.append(token)
        question = " ".join(remaining) if remaining else user_input

        try:
            ask(
                question,
                top_k=top_k,
                ticker=ticker,
                year=year,
                section=section,
                model=model,
                backend=backend,
                stream=True,
            )
        except Exception as exc:
            print(f"\nError: {exc}")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def _parse_args():
    parser = argparse.ArgumentParser(
        description="SEC RAG chatbot — retrieve from ChromaDB + generate with Ollama"
    )
    parser.add_argument("--query",   type=str, default=None,
                        help="Single question (omit for interactive chat loop)")
    parser.add_argument("--ticker",  type=str, default=None,
                        help="Filter retrieval to a specific ticker, e.g. TSLA")
    parser.add_argument("--year",    type=int, default=None,
                        help="Filter retrieval to a specific year, e.g. 2023")
    parser.add_argument("--section", type=str, default=None,
                        help="Filter retrieval to a section key, e.g. item_1a")
    parser.add_argument("--top-k",   type=int, default=TOP_K,
                        help=f"Number of chunks to retrieve (default: {TOP_K})")
    parser.add_argument("--model",   type=str, default=DEFAULT_MODEL,
                        help=f"Ollama model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--backend", choices=["local", "openai"], default="local",
                        help="Embedding backend (default: local)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print retrieved context before the answer")
    parser.add_argument("--debug",   action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(format="%(levelname)s  %(message)s", level=level)

    # Show available models
    models = list_available_models()
    if models:
        logger.debug("Available Ollama models: %s", models)

    if args.query:
        ask(
            args.query,
            top_k   = args.top_k,
            ticker  = args.ticker,
            year    = args.year,
            section = args.section,
            model   = args.model,
            backend = args.backend,
            stream  = True,
            verbose = args.verbose,
        )
    else:
        chat_loop(model=args.model, backend=args.backend, top_k=args.top_k)
