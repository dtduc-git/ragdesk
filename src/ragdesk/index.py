"""Index local files into the store (incremental by content hash)."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ragdesk.chunk import chunk_text
from ragdesk.embed import Embedder
from ragdesk.office import DOCUMENT_EXTENSIONS, extract_document
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
MAX_DOCUMENT_BYTES = 25_000_000  # PDFs and office files are legitimately large
EMBED_BATCH = 16  # 32 peaked ~270MB higher in the ONNX workspace for no speed gain


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
        # os.walk (not rglob) so skipped trees are pruned, never traversed:
        # a repo's target/ or node_modules/ can hold tens of thousands of files.
        for root, dirs, files in os.walk(path):
            dirs[:] = sorted(name for name in dirs if name not in SKIP_DIRS)
            for name in sorted(files):
                yield Path(root) / name


def is_text_file(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    if path.name in TEXT_FILENAMES:
        return True
    return path.suffix == ""


def is_document_file(path: Path) -> bool:
    return path.suffix.lower() in DOCUMENT_EXTENSIONS


def is_indexable(path: Path, size: int) -> bool:
    """Admission rules shared by local files and connector payloads."""
    if any(part in SKIP_DIRS for part in path.parts):
        return False
    if is_document_file(path):
        return size <= MAX_DOCUMENT_BYTES
    if size > MAX_FILE_BYTES:
        return False
    return is_text_file(path)


def read_text(path: Path) -> str | None:
    if is_document_file(path):
        return extract_document(path)
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
    store.set_local_roots([str(path) for path in paths if path.exists()])
    return stats
