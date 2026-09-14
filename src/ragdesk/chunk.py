"""Paragraph-aware chunking with character overlap."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_CHARS = 1000
DEFAULT_OVERLAP = 150


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    text: str


def chunk_text(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split text into overlapping, paragraph-aware chunks.

    Paragraphs are packed until ``max_chars``; oversized paragraphs are
    hard-split. The overlap is taken from the tail of the previous chunk so
    context survives boundaries.
    """
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            pieces.append(para)
        else:
            step = max_chars - overlap
            pieces.extend(para[i : i + max_chars] for i in range(0, len(para), step))

    chunks: list[str] = []
    buf = ""
    for piece in pieces:
        if not buf:
            buf = piece
            continue
        candidate = f"{buf}\n\n{piece}"
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        chunks.append(buf)
        tail = buf[-overlap:]
        joined = f"{tail}\n\n{piece}"
        buf = joined if len(joined) <= max_chars else piece
    if buf:
        chunks.append(buf)
    return [Chunk(ordinal=i, text=c) for i, c in enumerate(chunks)]
