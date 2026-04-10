"""
section_parser.py
=================
Parses a raw 10-K HTML filing and extracts the text for each target section
(Item 1, Item 1A, Item 7, Item 7A, Item 8).

Strategy
--------
10-K filings are inconsistently formatted, but almost every company uses the
Item X headings as navigational anchors.  We use a two-pass approach:

  Pass 1 — find "boundary nodes": elements whose text matches one of the
           standard 10-K item header patterns (e.g. "ITEM 1A. RISK FACTORS").
           We look at <p>, <div>, <span>, <td>, and <h1>-<h6> tags, filtering
           by bold/uppercase styling to skip body text that happens to contain
           the same words.

  Pass 2 — for each target item, collect all text nodes that appear *after*
           that item's boundary and *before* the next item's boundary.

This avoids relying on a fixed document structure and handles both iXBRL/HTML
and older plain-HTML filings.
"""

import re
import logging
import warnings
from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TARGET_SECTIONS

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Section header patterns
#  Each entry: (item_key, compiled_regex)
#  Order matters — use the canonical 10-K ordering so we can detect "next item"
# --------------------------------------------------------------------------- #

_SECTION_ORDER = [
    "item_1",
    "item_1a",
    "item_1b",
    "item_2",
    "item_3",
    "item_4",
    "item_5",
    "item_6",
    "item_7",
    "item_7a",
    "item_8",
    "item_9",
    "item_9a",
    "item_9b",
    "item_10",
    "item_11",
    "item_12",
    "item_13",
    "item_14",
    "item_15",
]

_SEP = r"[\s\xa0.]*"   # separator between "Item X" and the item name
                       # handles plain spaces, non-breaking spaces (\xa0), and dots

_PATTERNS: dict[str, re.Pattern] = {
    "item_1":   re.compile(r"(?i)^\s*item[\s\xa0]+1\b(?!a|b)" + _SEP + r"business",          re.S),
    "item_1a":  re.compile(r"(?i)^\s*item[\s\xa0]+1a\b"        + _SEP + r"risk[\s\xa0]+factor", re.S),
    "item_1b":  re.compile(r"(?i)^\s*item[\s\xa0]+1b\b",                                        re.S),
    "item_2":   re.compile(r"(?i)^\s*item[\s\xa0]+2\b"         + _SEP + r"propert",             re.S),
    "item_3":   re.compile(r"(?i)^\s*item[\s\xa0]+3\b"         + _SEP + r"legal",               re.S),
    "item_4":   re.compile(r"(?i)^\s*item[\s\xa0]+4\b",                                         re.S),
    "item_5":   re.compile(r"(?i)^\s*item[\s\xa0]+5\b",                                         re.S),
    "item_6":   re.compile(r"(?i)^\s*item[\s\xa0]+6\b",                                         re.S),
    "item_7":   re.compile(r"(?i)^\s*item[\s\xa0]+7\b(?!a)"    + _SEP + r"management",          re.S),
    "item_7a":  re.compile(r"(?i)^\s*item[\s\xa0]+7a\b",                                        re.S),
    "item_8":   re.compile(r"(?i)^\s*item[\s\xa0]+8\b"         + _SEP + r"financial[\s\xa0]+statement", re.S),
    "item_9":   re.compile(r"(?i)^\s*item[\s\xa0]+9\b(?!a|b)",                                  re.S),
    "item_9a":  re.compile(r"(?i)^\s*item[\s\xa0]+9a\b",                                        re.S),
    "item_9b":  re.compile(r"(?i)^\s*item[\s\xa0]+9b\b",                                        re.S),
    "item_10":  re.compile(r"(?i)^\s*item[\s\xa0]+10\b",                                        re.S),
    "item_11":  re.compile(r"(?i)^\s*item[\s\xa0]+11\b",                                        re.S),
    "item_12":  re.compile(r"(?i)^\s*item[\s\xa0]+12\b",                                        re.S),
    "item_13":  re.compile(r"(?i)^\s*item[\s\xa0]+13\b",                                        re.S),
    "item_14":  re.compile(r"(?i)^\s*item[\s\xa0]+14\b",                                        re.S),
    "item_15":  re.compile(r"(?i)^\s*item[\s\xa0]+15\b",                                        re.S),
}

# Tags we consider as potential section headers
_HEADING_TAGS = {"p", "div", "span", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6"}

# Minimum text length for a heading candidate (filters empty/whitespace nodes)
_MIN_HEADING_LEN = 3
_MAX_HEADING_LEN = 300  # real headings are short; long text is body content


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _is_bold_or_upper(tag: Tag) -> bool:
    """
    Returns True if the element is visually a heading:
      - rendered with font-weight:bold / <b> / <strong> wrapping
      - OR the text is ALL-CAPS
    """
    # Check inline style
    style = tag.get("style", "")
    if "font-weight" in style and "bold" in style:
        return True
    # Check for bold/strong ancestor within the same block
    if tag.find(["b", "strong"]):
        return True
    # All-caps heuristic (ignore punctuation/spaces)
    text = tag.get_text(" ", strip=True)
    letters = re.sub(r"[^a-zA-Z]", "", text)
    if letters and letters == letters.upper():
        return True
    return False


def _clean_text(text: str) -> str:
    """Collapse whitespace, strip leading/trailing space."""
    text = re.sub(r"\xa0", " ", text)       # non-breaking spaces
    text = re.sub(r"[ \t]+", " ", text)     # horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)  # excessive blank lines
    return text.strip()


# --------------------------------------------------------------------------- #
#  Core parser
# --------------------------------------------------------------------------- #

def parse_sections(html: str) -> dict[str, str]:
    """
    Parse a 10-K HTML document and return a dict mapping item_key -> section text.
    Only keys in TARGET_SECTIONS are returned; keys with no content are omitted.

    Parameters
    ----------
    html : str
        Raw HTML content of the filing.

    Returns
    -------
    dict[str, str]
        e.g. {"item_1": "Apple Inc. designs, manufactures...", "item_1a": "..."}
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove script / style / footer boilerplate
    for tag in soup(["script", "style", "head", "nav", "footer"]):
        tag.decompose()

    # ------------------------------------------------------------------ #
    #  Pass 1: locate boundary nodes for every item
    # ------------------------------------------------------------------ #
    # boundaries maps item_key -> list of Tag objects that look like headers
    boundaries: dict[str, list[Tag]] = {k: [] for k in _SECTION_ORDER}

    for tag in soup.find_all(_HEADING_TAGS):
        text = tag.get_text(" ", strip=True)
        if not (_MIN_HEADING_LEN <= len(text) <= _MAX_HEADING_LEN):
            continue
        for key, pattern in _PATTERNS.items():
            if pattern.search(text):
                # Prefer bold/uppercase tags; store both and pick the best later
                boundaries[key].append(tag)
                break  # a tag can only match one item

    # For each item key, take the LAST match that has substantial content
    # after it (to skip the Table of Contents entry, which appears early).
    # Heuristic: prefer the match that is deepest in the document tree
    # (i.e. highest tag index in the linearised DOM).
    all_tags = list(soup.find_all(True))  # all tags in document order
    tag_index = {id(t): i for i, t in enumerate(all_tags)}

    def best_boundary(candidates: list[Tag]) -> Tag | None:
        if not candidates:
            return None
        # Prefer bold/uppercase candidates (more likely to be actual headings).
        # Among ties, pick the LAST occurrence — Table of Contents entries
        # always appear before the real section.
        bold = [t for t in candidates if _is_bold_or_upper(t)]
        pool = bold if bold else candidates
        return max(pool, key=lambda t: tag_index.get(id(t), -1))

    chosen: dict[str, Tag | None] = {
        key: best_boundary(boundaries[key]) for key in _SECTION_ORDER
    }

    # ------------------------------------------------------------------ #
    #  Pass 2: extract text between consecutive chosen boundaries
    # ------------------------------------------------------------------ #
    # Build an ordered list of (item_key, tag_position) for chosen items only
    ordered = [
        (key, tag_index[id(chosen[key])])
        for key in _SECTION_ORDER
        if chosen[key] is not None
    ]
    ordered.sort(key=lambda x: x[1])

    sections: dict[str, str] = {}

    for i, (key, start_pos) in enumerate(ordered):
        if key not in TARGET_SECTIONS:
            continue

        end_pos = ordered[i + 1][1] if i + 1 < len(ordered) else len(all_tags)

        # Collect text from all_tags[start_pos+1 : end_pos]
        text_parts = []
        for tag in all_tags[start_pos + 1 : end_pos]:
            # Only grab leaf-level text to avoid duplicating nested content
            for child in tag.children:
                if isinstance(child, NavigableString):
                    s = str(child).strip()
                    if s:
                        text_parts.append(s)

        raw_text = " ".join(text_parts)
        cleaned  = _clean_text(raw_text)

        if len(cleaned) < 100:
            # Too short — likely landed in the ToC; skip
            logger.debug("Skipping %s — extracted text too short (%d chars)", key, len(cleaned))
            continue

        sections[key] = cleaned
        logger.info("Extracted %s: %d chars", key, len(cleaned))

    if not sections:
        logger.warning("No sections extracted — filing may use non-standard formatting")

    return sections


def parse_file(filepath: str) -> dict[str, str]:
    """Convenience wrapper: read a local .htm file and parse its sections."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        html = fh.read()
    return parse_sections(html)
