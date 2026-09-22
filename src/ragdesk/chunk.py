"""Paragraph-aware chunking with character overlap and line tracking.

Symbol-aware chunking was measured three ways and rejected on this corpus:
per-symbol segmentation (fragmentation + BM25 term density), symbol headers in
the chunk text (test files out-rank the implementation) and a symbol lane over
definition sites (loose LIKE matching dilutes RRF). See AGENTS.md for the
numbers; symbol intelligence belongs in a call-graph index, not in chunking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Lines that open a named block in the languages we index most.
SYMBOL_RE = re.compile(
    r"^\s*(?:async\s+)?(?:def|class|func|function|impl|struct|interface|enum|"
    r"fn|pub\s+fn)\s+([A-Za-z_][\w]*)"
)

DEFAULT_MAX_CHARS = 1000
DEFAULT_OVERLAP = 150
# Bump when the chunker's behaviour changes: every document stamps this, so an
# upgrade re-chunks existing files exactly once (see index.index_document).
CHUNKER_VERSION = 2


def chunk_config(max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP) -> str:
    return f"v{CHUNKER_VERSION}:{max_chars}/{overlap}"


# A hard split prefers a sentence end, then a word boundary: a chunk that
# starts mid-word ("phí bả…") reads as broken and forces another search.
SENTENCE_END_RE = re.compile(r"[.!?…][\"')\]]?\s")


def _cut_point(text: str, start: int, max_chars: int) -> int:
    """End of the next piece: the last sentence end in the window, else a space."""
    limit = min(start + max_chars, len(text))
    window = text[start:limit]
    floor = max_chars // 2
    cut = -1
    for match in SENTENCE_END_RE.finditer(window):
        if match.end() >= floor:
            cut = match.end()
    if cut > 0:
        return start + cut
    space = window.rfind(" ")
    if space >= floor:
        return start + space + 1
    return limit


def _resume(text: str, end: int, overlap: int) -> int:
    """Where the next piece starts: overlap back, snapped forward to a word."""
    start = max(0, end - overlap)
    space = text.find(" ", start, end)
    if space != -1:
        start = space + 1
    return start if start < end else end


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    text: str
    line_start: int = 1


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def chunk_text(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split text into overlapping, paragraph-aware chunks.

    Paragraphs are packed until ``max_chars``; oversized paragraphs are
    hard-split. The overlap is taken from the tail of the previous chunk so
    context survives boundaries. Each chunk remembers the source line it starts
    on, which is what a citation's ``:line`` points at.
    """
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    paragraphs: list[tuple[int, str]] = []
    cursor = 0
    for raw in text.split("\n\n"):
        stripped = raw.strip()
        if stripped:
            offset = cursor + raw.index(stripped)
            paragraphs.append((offset, stripped))
        cursor += len(raw) + 2

    pieces: list[tuple[int, str]] = []
    for offset, para in paragraphs:
        if len(para) <= max_chars:
            pieces.append((offset, para))
        else:
            start = 0
            while start < len(para):
                end = _cut_point(para, start, max_chars)
                pieces.append((offset + start, para[start:end].strip()))
                if end >= len(para):
                    break
                start = _resume(para, end, overlap)

    chunks: list[tuple[int, str]] = []
    buf = ""
    buf_offset = 0
    for offset, piece in pieces:
        if not buf:
            buf = piece
            buf_offset = offset
            continue
        candidate = f"{buf}\n\n{piece}"
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        chunks.append((buf_offset, buf))
        tail = buf[-overlap:]
        space = tail.find(" ")
        if space != -1:
            # Resume at a word start — but only when the text has word breaks
            # at all: minified JSON/base64 has none, and dropping the tail there
            # would silently kill the overlap.
            tail = tail[space + 1 :]
        joined = f"{tail}\n\n{piece}" if tail else piece
        if len(joined) <= max_chars:
            buf = joined
        else:
            buf = piece
            buf_offset = offset
    if buf:
        chunks.append((buf_offset, buf))
    return [
        Chunk(ordinal=i, text=chunk, line_start=_line_number(text, offset))
        for i, (offset, chunk) in enumerate(chunks)
    ]
