"""
generator.py
============
Takes retrieved context chunks and a user question, then calls a local
Ollama LLM to generate a grounded, cited answer.

The prompt instructs the model to:
  - answer ONLY from the provided context (reduces hallucination)
  - cite sources using [1], [2], … notation matching the context numbering
  - explicitly say "I don't know" if the context doesn't contain the answer

Supports streaming output so answers appear word-by-word in the terminal.

Usage:
    from generation.generator import generate

    answer = generate(question, context_str)           # full string
    generate(question, context_str, stream=True)       # prints as it streams
"""

import logging
import os
import sys

import ollama

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama3.2:3b"

# --------------------------------------------------------------------------- #
#  System prompt
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = """You are a financial analyst assistant that answers questions
about SEC 10-K filings. You will be given numbered context passages extracted
directly from official filings.

Rules:
1. Answer ONLY using information from the provided context.
2. Cite your sources inline using [1], [2], etc. matching the passage numbers.
3. If the context does not contain enough information to answer, say:
   "The provided filings do not contain enough information to answer this question."
4. Be concise and factual. Do not speculate beyond what is stated.
5. When quoting numbers or statistics, always include the citation."""


def _build_prompt(question: str, context: str) -> str:
    return f"""Here are the relevant passages from SEC 10-K filings:

{context}

---

Question: {question}

Answer (cite sources using [1], [2], etc.):"""


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #

def generate(
    question:  str,
    context:   str,
    model:     str  = DEFAULT_MODEL,
    stream:    bool = False,
    temperature: float = 0.1,   # low temp = more factual, less creative
) -> str:
    """
    Generate an answer grounded in the retrieved context.

    Parameters
    ----------
    question    : user's natural language question
    context     : formatted context string from retriever.format_context()
    model       : Ollama model name (default: llama3.2:3b)
    stream      : if True, print tokens as they arrive and return full string
    temperature : generation temperature (0.0–1.0; lower = more deterministic)

    Returns
    -------
    str — the full generated answer
    """
    prompt = _build_prompt(question, context)

    logger.info("Generating answer with %s (stream=%s)…", model, stream)

    try:
        if stream:
            return _generate_streaming(prompt, model, temperature)
        else:
            return _generate_blocking(prompt, model, temperature)
    except ollama.ResponseError as exc:
        if "model" in str(exc).lower() and "not found" in str(exc).lower():
            raise RuntimeError(
                f"Ollama model '{model}' is not installed.\n"
                f"Run:  ollama pull {model}"
            ) from exc
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Ollama call failed: {exc}\n"
            "Make sure Ollama is running:  brew services start ollama"
        ) from exc


def _generate_blocking(prompt: str, model: str, temperature: float) -> str:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system",  "content": _SYSTEM_PROMPT},
            {"role": "user",    "content": prompt},
        ],
        options={"temperature": temperature},
    )
    return response["message"]["content"].strip()


def _generate_streaming(prompt: str, model: str, temperature: float) -> str:
    full_text = []
    stream = ollama.chat(
        model=model,
        messages=[
            {"role": "system",  "content": _SYSTEM_PROMPT},
            {"role": "user",    "content": prompt},
        ],
        options={"temperature": temperature},
        stream=True,
    )
    for chunk in stream:
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        full_text.append(token)
    print()  # newline after streaming completes
    return "".join(full_text).strip()


def list_available_models() -> list[str]:
    """Return names of locally available Ollama models."""
    try:
        return [m["name"] for m in ollama.list()["models"]]
    except Exception:
        return []
