"""Index local files into the store (incremental by content hash)."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ragdesk.chunk import chunk_text
from ragdesk.embed import Embedder
from ragdesk.store import Store

TEXT_EXTENSIONS = {
    ".md", ".markdown", ".rst", ".txt",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb",
    ".sh", ".bash", ".zsh",
    ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg",
    ".tf", ".hcl", ".sql", ".html", ".css", ".xml", ".csv", ".tmpl", ".tpl",
}
TEXT_FILENAMES = {"Dockerfile", "Makefile", "README", "LICENSE", "CHANGELOG", "CONTRIBUTING"}
SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
    ".ragdesk", "dist", "build", "target", ".terraform",
    ".mypy_cache", ".ruff_cache", ".pytest_cache",
}
MAX_FILE_BYTES = 1_000_000
EMBED_BATCH = 32


@dataclass
class IndexStats:
    files_scanned: int = 0
    indexed: int = 0
    unchanged: int = 0
    skipped: int = 0
    chunks: int = 0


def iter_files(paths: list[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_file():
            yield path
            continue
        for candidate in sorted(path.rglob("*")):
            if not candidate.is_file():
                continue
            if any(part in SKIP_DIRS for part in candidate.parts):
                continue
            yield candidate


def is_text_file(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    if path.name in TEXT_FILENAMES:
        return True
    return path.suffix == ""


def is_indexable(path: Path, size: int) -> bool:
    """Admission rules shared by local files and connector payloads."""
    if size > MAX_FILE_BYTES:
        return False
    if any(part in SKIP_DIRS for part in path.parts):
        return False
    return is_text_file(path)


def read_text(path: Path) -> str | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:1024]:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def index_document(
    store: Store,
    embedder: Embedder,
    *,
    source: str,
    path: str,
    content: str,
    mtime: float = 0.0,
) -> int:
    """Embed + upsert one document. Returns the chunk count, or 0 if unchanged."""
    digest = hashlib.sha256(content.encode()).hexdigest()
    if store.doc_hash(path) == digest:
        return 0
    chunks = chunk_text(content)
    embeddings: list[list[float]] = []
    for start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[start : start + EMBED_BATCH]
        embeddings.extend(embedder.embed([chunk.text for chunk in batch]))
    store.upsert_document(
        source=source,
        path=path,
        content_hash=digest,
        mtime=mtime,
        texts=[chunk.text for chunk in chunks],
        embeddings=embeddings,
    )
    return len(chunks)


def index_paths(store: Store, embedder: Embedder, paths: list[Path]) -> IndexStats:
    store.ensure_embedder(embedder.name, embedder.dim)
    stats = IndexStats()
    for file in iter_files(paths):
        stats.files_scanned += 1
        try:
            size = file.stat().st_size
        except OSError:
            stats.skipped += 1
            continue
        if not is_indexable(file, size):
            stats.skipped += 1
            continue
        content = read_text(file)
        if content is None or not content.strip():
            stats.skipped += 1
            continue

        chunks = index_document(
            store,
            embedder,
            source="local",
            path=str(file),
            content=content,
            mtime=file.stat().st_mtime,
        )
        if chunks:
            stats.indexed += 1
            stats.chunks += chunks
        else:
            stats.unchanged += 1
    return stats
