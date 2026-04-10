"""
chunker.py
==========
Splits extracted section text into overlapping token-aware chunks and attaches
structured metadata for downstream embedding + retrieval.

Each chunk is a dict:
  {
    "text":          str,    # the chunk text
    "ticker":        str,    # e.g. "TSLA"
    "cik":           str,    # 10-digit zero-padded CIK
    "year":          int,    # filing year (from filingDate)
    "section_key":   str,    # e.g. "item_1a"
    "section_label": str,    # e.g. "Risk Factors"
    "chunk_index":   int,    # 0-based position within the section
    "total_chunks":  int,    # how many chunks this section was split into
    "token_count":   int,    # actual token count of this chunk
  }

Token counting uses tiktoken with the cl100k_base encoding (same as
text-embedding-3-small and GPT-4o), so CHUNK_SIZE is an accurate budget.
"""

import re
import logging
from typing import Iterator

import tiktoken

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHUNK_SIZE, CHUNK_OVERLAP, TARGET_SECTIONS

logger = logging.getLogger(__name__)

# Use cl100k_base — matches OpenAI text-embedding-3-small and GPT-4o
_ENC = tiktoken.get_encoding("cl100k_base")


# --------------------------------------------------------------------------- #
#  Token helpers
# --------------------------------------------------------------------------- #

def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentence-level units.
    We use a simple regex that handles common abbreviations well enough
    for financial prose.
    """
    # Split on '. ', '! ', '? ', '\n\n' but keep the delimiter with the sentence
    units = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])|(?<=\n)\n+", text)
    return [u.strip() for u in units if u.strip()]


# --------------------------------------------------------------------------- #
#  Core chunker
# --------------------------------------------------------------------------- #

def chunk_section(
    text: str,
    ticker: str,
    cik: str,
    year: int,
    section_key: str,
    chunk_size: int   = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split `text` into overlapping chunks of ≤ `chunk_size` tokens.

    Algorithm
    ---------
    1. Split text into sentence-level units.
    2. Greedily pack sentences into a buffer until adding the next sentence
       would exceed `chunk_size` tokens.
    3. Emit the buffer as a chunk.
    4. Backtrack by `chunk_overlap` tokens (keep the tail of the current
       buffer as the start of the next buffer).

    This ensures chunks respect sentence boundaries rather than cutting
    mid-sentence, which improves embedding quality.
    """
    section_label = TARGET_SECTIONS.get(section_key, section_key)
    sentences     = _split_sentences(text)

    if not sentences:
        return []

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    def flush_buffer():
        if buffer:
            chunks.append(" ".join(buffer))

    for sentence in sentences:
        sent_tokens = count_tokens(sentence)

        # Single sentence longer than chunk_size — force-split on whitespace
        if sent_tokens > chunk_size:
            if buffer:
                flush_buffer()
                buffer, buffer_tokens = [], 0
            words = sentence.split()
            sub_buf, sub_tok = [], 0
            for word in words:
                wt = count_tokens(word + " ")
                if sub_tok + wt > chunk_size and sub_buf:
                    chunks.append(" ".join(sub_buf))
                    # overlap: keep last few words
                    overlap_words = _tail_by_tokens(sub_buf, chunk_overlap)
                    sub_buf   = overlap_words + [word]
                    sub_tok   = count_tokens(" ".join(sub_buf))
                else:
                    sub_buf.append(word)
                    sub_tok += wt
            if sub_buf:
                chunks.append(" ".join(sub_buf))
            continue

        if buffer_tokens + sent_tokens > chunk_size and buffer:
            flush_buffer()
            # Keep an overlap tail from the current buffer
            overlap = _tail_by_tokens(buffer, chunk_overlap)
            buffer       = overlap + [sentence]
            buffer_tokens = count_tokens(" ".join(buffer))
        else:
            buffer.append(sentence)
            buffer_tokens += sent_tokens

    flush_buffer()

    if not chunks:
        return []

    # Attach metadata
    total = len(chunks)
    result = []
    for i, chunk_text in enumerate(chunks):
        result.append({
            "text":          chunk_text,
            "ticker":        ticker,
            "cik":           cik,
            "year":          year,
            "section_key":   section_key,
            "section_label": section_label,
            "chunk_index":   i,
            "total_chunks":  total,
            "token_count":   count_tokens(chunk_text),
        })

    logger.info(
        "%s %d | %s → %d chunks (avg %.0f tokens)",
        ticker, year, section_key, total,
        sum(c["token_count"] for c in result) / total,
    )
    return result


def _tail_by_tokens(sentences: list[str], max_tokens: int) -> list[str]:
    """Return the suffix of `sentences` whose total token count ≤ max_tokens."""
    tail, tok = [], 0
    for s in reversed(sentences):
        t = count_tokens(s)
        if tok + t > max_tokens:
            break
        tail.insert(0, s)
        tok += t
    return tail


# --------------------------------------------------------------------------- #
#  Convenience: chunk all sections for one filing
# --------------------------------------------------------------------------- #

def chunk_filing(
    sections:  dict[str, str],
    ticker:    str,
    cik:       str,
    year:      int,
    chunk_size: int    = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Chunk every section in `sections` and return a flat list of chunk dicts.

    Parameters
    ----------
    sections : dict[str, str]
        Output of section_parser.parse_sections() — {item_key: text}.
    ticker, cik, year :
        Filing metadata passed through to each chunk.
    """
    all_chunks = []
    for section_key, text in sections.items():
        chunks = chunk_section(
            text, ticker, cik, year, section_key,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(chunks)
    return all_chunks
