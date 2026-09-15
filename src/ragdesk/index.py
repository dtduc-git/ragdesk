"""Index local files into the store (incremental by content hash)."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ragdesk.chunk import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP, chunk_text
from ragdesk.embed import Embedder
from ragdesk.office import DOCUMENT_EXTENSIONS
from ragdesk.store import Store
from ragdesk.vision import IMAGE_EXTENSIONS

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
    skipped_samples: list = field(default_factory=list)

    def skip(self, path: Path, reason: str) -> None:
        self.skipped += 1
        if len(self.skipped_samples) < 12:
            self.skipped_samples.append({"path": str(path), "reason": reason})


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


FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_front_matter(content: str) -> tuple[dict[str, str], str]:
    """A simple ``--- key: value ---`` header: metadata plus the remaining text.

    Only flat ``key: value`` pairs are read — enough for service/type/env tags
    without pulling in a YAML dependency.
    """
    match = FRONT_MATTER_RE.match(content)
    if not match:
        return {}, content
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip().strip('"').strip("'")
        if key and value and re.fullmatch(r"[a-z0-9_-]+", key):
            meta[key] = value[:100]
    return meta, content[match.end() :]


def is_text_file(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    if path.name in TEXT_FILENAMES:
        return True
    return path.suffix == ""


def is_document_file(path: Path) -> bool:
    return path.suffix.lower() in DOCUMENT_EXTENSIONS


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_indexable(path: Path, size: int) -> bool:
    """Admission rules shared by local files and connector payloads."""
    if any(part in SKIP_DIRS for part in path.parts):
        return False
    if is_document_file(path) or is_image_file(path):
        return size <= MAX_DOCUMENT_BYTES
    if size > MAX_FILE_BYTES:
        return False
    return is_text_file(path)


def extract_bytes(data: bytes, name: str) -> str | None:
    """Text for a connector payload, by file name: documents, images, or plain text."""
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        from ragdesk.office import MAX_DOCUMENT_CHARS, extract_pdf_text  # noqa: PLC0415

        text = extract_pdf_text(data)
        return text[:MAX_DOCUMENT_CHARS] if text else None
    if suffix in DOCUMENT_EXTENSIONS:
        from ragdesk.office import MAX_DOCUMENT_CHARS, extract_office_text  # noqa: PLC0415

        text = extract_office_text(data, suffix)
        return text[:MAX_DOCUMENT_CHARS] if text else None
    if suffix in IMAGE_EXTENSIONS:
        from ragdesk.vision import extract_image_bytes  # noqa: PLC0415

        return extract_image_bytes(data, name)
    if b"\x00" in data[:1024]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return extract_bytes(data, path.name)


def group_parents(texts: list[str], max_chars: int = 4000) -> tuple[list[str], list[int]]:
    """Group child chunks into parent sections; returns (parent texts, child→parent)."""
    parents: list[str] = []
    assignment: list[int] = []
    current: list[str] = []
    size = 0
    for text in texts:
        current.append(text)
        size += len(text)
        if size >= max_chars:
            parents.append("\n\n".join(current))
            assignment.extend([len(parents) - 1] * len(current))
            current = []
            size = 0
    if current:
        parents.append("\n\n".join(current))
        assignment.extend([len(parents) - 1] * len(current))
    return parents, assignment


def index_document(
    store: Store,
    embedder: Embedder,
    *,
    source: str,
    path: str,
    content: str,
    mtime: float = 0.0,
    chunk_chars: int = 0,
    chunk_overlap: int = 0,
) -> int:
    """Embed + upsert one document. Returns the chunk count, or 0 if unchanged."""
    metadata: dict[str, str] = {}
    if Path(path).suffix.lower() in {".md", ".markdown", ".txt", ""}:
        metadata, content = parse_front_matter(content)
    digest = hashlib.sha256(content.encode()).hexdigest()
    if store.doc_hash(path) == digest:
        if mtime:
            store.touch_document(path, mtime)
        return 0
    chunks = chunk_text(
        content,
        max_chars=chunk_chars or DEFAULT_MAX_CHARS,
        overlap=chunk_overlap or DEFAULT_OVERLAP,
    )
    embeddings: list[list[float]] = []
    for start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[start : start + EMBED_BATCH]
        embeddings.extend(embedder.embed([chunk.text for chunk in batch]))
    texts = [chunk.text for chunk in chunks]
    parents, assignment = group_parents(texts)
    store.upsert_document(
        source=source,
        path=path,
        content_hash=digest,
        mtime=mtime,
        metadata=metadata,
        texts=texts,
        embeddings=embeddings,
        parents=parents,
        parent_index=assignment,
        line_starts=[chunk.line_start for chunk in chunks],
    )
    return len(chunks)


def index_paths(
    store: Store,
    embedder: Embedder,
    paths: list[Path],
    progress: Callable[[str, int, int], None] | None = None,
    chunk_chars: int = 0,
    chunk_overlap: int = 0,
) -> IndexStats:
    store.ensure_embedder(embedder.name, embedder.dim)
    if not chunk_chars or not chunk_overlap:
        from ragdesk import settings as app_settings  # noqa: PLC0415 - optional knob

        values = app_settings.load()
        chunk_chars = chunk_chars or int(values.get("chunk_chars") or DEFAULT_MAX_CHARS)
        chunk_overlap = chunk_overlap or int(
            values.get("chunk_overlap") or DEFAULT_OVERLAP
        )
    stats = IndexStats()
    for file in iter_files(paths):
        stats.files_scanned += 1
        if progress is not None:
            progress(f"indexing {file.name}", stats.files_scanned, 0)
        try:
            info = file.stat()
        except OSError:
            stats.skip(file, "unreadable")
            continue
        # Fast path: an unchanged mtime means we did not touch the file at all,
        # so a 60s watcher pass costs a stat per file and nothing else.
        stored = store.doc_mtime(str(file))
        if stored is not None and abs(stored - info.st_mtime) < 1e-6:
            stats.unchanged += 1
            continue
        if not is_indexable(file, info.st_size):
            stats.skip(file, "unsupported or too large")
            continue
        content = read_text(file)
        if content is None or not content.strip():
            stats.skip(file, "no extractable text")
            continue

        chunks = index_document(
            store,
            embedder,
            source="local",
            path=str(file),
            content=content,
            mtime=info.st_mtime,
            chunk_chars=chunk_chars,
            chunk_overlap=chunk_overlap,
        )
        if chunks:
            stats.indexed += 1
            stats.chunks += chunks
        else:
            stats.unchanged += 1
    store.set_local_roots([str(path) for path in paths if path.exists()])
    return stats
