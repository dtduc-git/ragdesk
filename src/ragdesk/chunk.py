"""Paragraph-aware chunking with character overlap and line tracking."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_CHARS = 1000
DEFAULT_OVERLAP = 150

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
            step = max_chars - overlap
            pieces.extend(
                (offset + i, para[i : i + max_chars]) for i in range(0, len(para), step)
            )

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
        joined = f"{tail}\n\n{piece}"
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
