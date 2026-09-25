"""Index local files into the store (incremental by content hash)."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ragdesk.chunk import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP, chunk_config, chunk_text
from ragdesk.embed import Embedder
from ragdesk.office import DOCUMENT_EXTENSIONS
from ragdesk.store import Store, matches_any
from ragdesk.vision import IMAGE_EXTENSIONS

TEXT_EXTENSIONS = {
    ".md",
    ".markdown",
    ".rst",
    ".txt",
    ".adoc",
    ".org",
    ".tex",
    ".py",
    ".js",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".jsx",
    ".vue",
    ".svelte",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".kts",
    ".scala",
    ".clj",
    ".cs",
    ".swift",
    ".m",
    ".mm",
    ".c",
    ".h",
    ".cc",
    ".cpp",
    ".cxx",
    ".hpp",
    ".hxx",
    ".php",
    ".pl",
    ".lua",
    ".dart",
    ".groovy",
    ".gradle",
    ".r",
    ".rb",
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
    ".bat",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".properties",
    ".tf",
    ".tfvars",
    ".hcl",
    ".cmake",
    ".mk",
    ".proto",
    ".sql",
    ".html",
    ".css",
    ".scss",
    ".xml",
    ".csv",
    ".tmpl",
    ".tpl",
}
TEXT_FILENAMES = {"Dockerfile", "Makefile", "README", "LICENSE", "CHANGELOG", "CONTRIBUTING"}
SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".ragdesk",
    "dist",
    "build",
    "target",
    "sidecar",
    ".terraform",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".obsidian",
    ".trash",  # an Obsidian vault's config and deleted notes
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
    attachments: int = 0  # email attachments indexed as their own documents
    removed: int = 0  # local documents dropped because their file is gone
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
    # Hidden files are not prose, and `.env` is where secrets live: indexing it
    # would embed API keys into the searchable index.
    if path.name.startswith("."):
        return False
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


def vault_roots() -> list[str]:
    """Folders the user added as Obsidian vaults (Settings / `ragdesk obsidian`)."""
    from ragdesk import settings as app_settings  # noqa: PLC0415 - optional knob

    raw = app_settings.load().get("vaults") or []
    return [str(item).strip() for item in raw if str(item).strip()]


def vault_for(path: Path, roots: list[str]) -> str:
    """The vault name this file belongs to, or ""."""
    from ragdesk.obsidian import vault_name  # noqa: PLC0415 - avoids an import cycle

    text = str(path)
    for root in roots:
        if text == root or text.startswith(root.rstrip("/") + "/"):
            return vault_name(root)
    return ""


def never_index_patterns() -> list[str]:
    """Globs the user never wants indexed (secrets, dumps, private folders)."""
    from ragdesk import settings as app_settings  # noqa: PLC0415 - optional knob

    raw = app_settings.load().get("never_index") or []
    return [str(item).strip() for item in raw if str(item).strip()]


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


def _embed_with_cache(store: Store, embedder: Embedder, texts: list[str]) -> list[list[float]]:
    """Embed only what changed: identical chunk text reuses its stored vector.

    Chunk text + embedder identity is the cache key, so a one-line edit in a
    long document costs one embedding instead of hundreds, and a duplicated
    file indexes with zero model calls. Vectors come from the same model, so
    retrieval results are unchanged — the cache is never allowed to alter an
    answer, only the work it takes to produce one.
    """
    keys = [store.embed_key(embedder.name, embedder.dim, text) for text in texts]
    cached = store.embed_cache_get(keys)
    vectors: list[list[float] | None] = [None] * len(texts)
    pending: list[tuple[int, str, str]] = []
    for position, (key, text) in enumerate(zip(keys, texts, strict=True)):
        hit = cached.get(key)
        if hit is None:
            pending.append((position, key, text))
        else:
            vectors[position] = hit
    for start in range(0, len(pending), EMBED_BATCH):
        batch = pending[start : start + EMBED_BATCH]
        fresh = embedder.embed([text for _position, _key, text in batch])
        store.embed_cache_put(
            [(key, vector) for (_position, key, _text), vector in zip(batch, fresh, strict=True)]
        )
        for (position, _key, _text), vector in zip(batch, fresh, strict=True):
            vectors[position] = vector
    return [vector or [] for vector in vectors]


def resolve_chunk_settings(chunk_chars: int | None, chunk_overlap: int | None) -> tuple[int, int]:
    """Fill unset chunk knobs from settings; 0 stays 0 for the overlap.

    ``None`` means unset; the previous ``0 or default`` pattern made a zero
    overlap impossible and left every connector stuck on the built-in sizes.
    """
    if chunk_chars is None or chunk_overlap is None:
        from ragdesk import settings as app_settings  # noqa: PLC0415 - optional knob

        values = app_settings.load()
        if chunk_chars is None:
            chunk_chars = int(values.get("chunk_chars") or DEFAULT_MAX_CHARS)
        if chunk_overlap is None:
            stored = values.get("chunk_overlap")
            # an explicit 0 in settings.json is a value, not "unset"
            chunk_overlap = DEFAULT_OVERLAP if stored in (None, "") else int(stored)
    return chunk_chars, chunk_overlap


def index_document(
    store: Store,
    embedder: Embedder,
    *,
    source: str,
    path: str,
    content: str,
    mtime: float = 0.0,
    chunk_chars: int | None = None,
    chunk_overlap: int | None = None,
    metadata: dict[str, str] | None = None,
) -> int:
    """Embed + upsert one document. Returns the chunk count, or 0 if unchanged."""
    parsed: dict[str, str] = {}
    if Path(path).suffix.lower() in {".md", ".markdown", ".txt", ""}:
        parsed, content = parse_front_matter(content)
    metadata = {**(metadata or {}), **parsed}
    max_chars, overlap = resolve_chunk_settings(chunk_chars, chunk_overlap)
    config = chunk_config(max_chars, overlap)
    digest = hashlib.sha256(content.encode()).hexdigest()
    # A different chunker means the stored chunks are stale even when the text
    # is identical: re-chunk. The stamp is per document, so a partial run (one
    # vault, one folder) never marks the rest of the index as up to date.
    if store.doc_hash(path) == digest and store.doc_chunk_config(path) == config:
        if mtime:
            store.touch_document(path, mtime)
        return 0
    chunks = chunk_text(content, max_chars=max_chars, overlap=overlap)
    texts = [chunk.text for chunk in chunks]
    embeddings = _embed_with_cache(store, embedder, texts)
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
        chunk_config=config,
    )
    return len(chunks)


def index_paths(
    store: Store,
    embedder: Embedder,
    paths: list[Path],
    progress: Callable[[str, int, int], None] | None = None,
    chunk_chars: int | None = None,
    chunk_overlap: int | None = None,
) -> IndexStats:
    store.ensure_embedder(embedder.name, embedder.dim)
    chunk_chars, chunk_overlap = resolve_chunk_settings(chunk_chars, chunk_overlap)
    stats = IndexStats()
    patterns = never_index_patterns()
    vaults = vault_roots()
    config = chunk_config(chunk_chars, chunk_overlap)
    for file in iter_files(paths):
        stats.files_scanned += 1
        if progress is not None:
            progress(f"indexing {file.name}", stats.files_scanned, 0)
        if patterns and matches_any(str(file), patterns):
            stats.skip(file, "never-index pattern")
            continue
        try:
            info = file.stat()
        except OSError:
            stats.skip(file, "unreadable")
            continue
        # Fast path: an unchanged mtime means we did not touch the file at all,
        # so a 60s watcher pass costs a stat per file and nothing else. A stale
        # chunk stamp (upgrade, changed chunk size) must fall through instead.
        stored = store.doc_mtime(str(file))
        if (
            stored is not None
            and abs(stored - info.st_mtime) < 1e-6
            and store.doc_chunk_config(str(file)) == config
        ):
            stats.unchanged += 1
            continue
        if not is_indexable(file, info.st_size):
            stats.skip(file, "unsupported or too large")
            continue
        # One bad file (corrupt archive, unreadable bytes, a wedged DB write)
        # must never take down the whole pass — the watcher runs unattended.
        try:
            content = read_text(file)
            if content is None or not content.strip():
                stats.skip(file, "no extractable text")
                continue

            metadata = None
            vault = vault_for(file, vaults) if vaults else ""
            if vault:
                from ragdesk.obsidian import vault_metadata  # noqa: PLC0415

                metadata, content = vault_metadata(vault, content)
                if not content.strip():
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
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001 - report and keep indexing
            stats.skip(file, f"error: {type(exc).__name__}: {exc}")
            continue
        if chunks:
            stats.indexed += 1
            stats.chunks += chunks
        else:
            stats.unchanged += 1
    store.set_local_roots([str(path) for path in paths if path.exists()])
    # Unfiltered: a chosen file that is gone must still be pruned (see
    # delete_missing_local); a relative path is ignored there on purpose.
    stats.removed = len(store.delete_missing_local([str(path) for path in paths]))
    store.record_index_report(
        {
            "at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
            "roots": [str(path) for path in paths],
            "files_scanned": stats.files_scanned,
            "indexed": stats.indexed,
            "unchanged": stats.unchanged,
            "removed": stats.removed,
            "skipped": stats.skipped,
            "chunks": stats.chunks,
            "skipped_samples": list(stats.skipped_samples),
        }
    )
    return stats
