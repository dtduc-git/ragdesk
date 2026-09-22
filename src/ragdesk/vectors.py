"""Dense vector search backends for the chunk index.

Embeddings are fp32 blobs in ``chunks.embedding``. A plain Python scan is
always available; the numpy matrix covers personal corpora (up to ~1M chunks
in RAM); usearch adds an approximate HNSW index as an explicit opt-in
(``docs/vector-scale.md`` has the measurements). Every backend returns
``(chunk_id, score)`` candidates sorted by score, so the store keeps one
payload-fetch and one ranking pipeline.

Indexes are cached per database path at module level because the desktop
server opens a fresh ``Store`` per request. The guard connection watches the
``chunks_revision`` meta counter that every chunk write bumps, so an index is
rebuilt only when the vectors actually changed — not on every chat message
commit (``PRAGMA data_version`` would invalidate on any write, including the
answer cache, making every question pay a full matrix rebuild).
"""

from __future__ import annotations

import array
import math
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Any

BACKENDS = ("python", "numpy", "usearch")
# Keep only the most recently built indexes: a couple of databases (say the app
# plus a CLI run) must not pin several multi-GB matrices for the process life.
MAX_CACHE = 4
_cache: dict[str, tuple[tuple[Any, ...], Any]] = {}
_guards: dict[str, sqlite3.Connection] = {}
_lock = threading.Lock()
_warned: set[str] = set()
_override: str | None = None


def set_backend_override(name: str | None) -> None:
    """Process-wide preference for stores opened without an explicit one.

    ``serve``/``mcp``/``tui`` create their own ``Store`` per request, so the
    CLI flag reaches them here — same pattern as ``embed.set_thread_override``.
    """
    global _override
    _override = name or None


def backend_override() -> str:
    return _override or ""


def _warn(message: str) -> None:
    """Print once per message: the same bad setting hits every lane of every query."""
    with _lock:
        if message in _warned:
            return
        _warned.add(message)
    print(f"ragdesk: {message}", file=sys.stderr)


def _numpy():
    try:
        import numpy  # noqa: PLC0415 - optional dependency, imported lazily
    except ImportError:  # pragma: no cover - core installs without the onnx extra
        return None
    return numpy


def _usearch():
    try:
        from usearch.index import Index  # noqa: PLC0415 - optional dependency
    except ImportError:  # pragma: no cover - only installed with the vec extra
        return None
    return Index


def available(name: str) -> bool:
    """Whether this install can run the backend (extras differ per install)."""
    if name == "usearch":
        return _usearch() is not None
    if name == "numpy":
        return _numpy() is not None
    return name == "python"


def _dim(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT length(embedding) FROM chunks LIMIT 1").fetchone()
    return (int(row[0]) // 4) if row and row[0] else 0


def _revision(conn: sqlite3.Connection) -> str:
    """Chunk-write counter: bumped inside every transaction that changes chunks."""
    row = conn.execute("SELECT value FROM meta WHERE key = 'chunks_revision'").fetchone()
    return str(row[0]) if row else "0"


def _load_matrix(conn: sqlite3.Connection) -> tuple[Any, Any]:
    """Every ``(id, embedding)`` under one read snapshot.

    The count and the row stream used to be separate statements: a watcher pass
    inserting between them overflowed the preallocated matrix (IndexError). The
    deferred transaction pins one snapshot, so a concurrent writer waits (WAL:
    proceeds) and the build stays consistent.
    """
    np = _numpy()
    if np is None:
        raise RuntimeError("numpy is not installed")
    owns_txn = not conn.in_transaction
    if owns_txn:
        conn.execute("BEGIN")
    try:
        total = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        ids = np.empty(total, dtype=np.int64)
        matrix = np.empty((total, _dim(conn)), dtype=np.float32)
        # Stream rows: fetchall() would hold every blob in Python memory next
        # to the matrix, doubling the peak at large corpora.
        index = 0
        for row in conn.execute("SELECT id, embedding FROM chunks ORDER BY id"):
            ids[index] = row[0]
            matrix[index] = np.frombuffer(row[1], dtype=np.float32)
            index += 1
    except BaseException:
        if owns_txn:
            conn.execute("ROLLBACK")
        raise
    if owns_txn:
        conn.execute("COMMIT")
    return ids[:index], matrix[:index]


def _scope_ids(conn: sqlite3.Connection, scope_sql: str, scope_params: Any) -> list[int]:
    rows = conn.execute(
        "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE 1=1" + scope_sql,
        scope_params,
    ).fetchall()
    return [int(row[0]) for row in rows]


class PythonScan:
    """The fallback: exact cosine in Python over every chunk (never approximate)."""

    name = "python"

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.dim = _dim(conn)

    def search(
        self,
        conn: sqlite3.Connection,
        query_vec: list[float],
        limit: int,
        scope_sql: str = "",
        scope_params: Any = (),
    ) -> list[tuple[int, float]]:
        rows = conn.execute(
            "SELECT c.id, c.embedding FROM chunks c "
            "JOIN documents d ON d.id = c.doc_id WHERE 1=1" + scope_sql,
            scope_params,
        ).fetchall()
        q_norm = math.sqrt(sum(v * v for v in query_vec)) or 1.0
        scored: list[tuple[float, int]] = []
        for row in rows:
            vec = array.array("f")
            vec.frombytes(row[1])
            dot = sum(a * b for a, b in zip(query_vec, vec, strict=True))
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            scored.append((dot / (q_norm * norm), int(row[0])))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [(chunk_id, float(score)) for score, chunk_id in scored[:limit]]


class NumpyMatrix:
    """Exact cosine over an in-RAM fp32 matrix with L2-normalized rows."""

    name = "numpy"

    def __init__(self, conn: sqlite3.Connection) -> None:
        np = _numpy()
        if np is None:
            raise RuntimeError("numpy is not installed")
        self.np = np
        self.ids, self.matrix = _load_matrix(conn)
        if self.ids.size:
            norms = np.linalg.norm(self.matrix, axis=1)
            norms[norms == 0] = 1.0
            self.matrix /= norms[:, None]
        self.dim = int(self.matrix.shape[1])

    def search(
        self,
        conn: sqlite3.Connection,
        query_vec: list[float],
        limit: int,
        scope_sql: str = "",
        scope_params: Any = (),
    ) -> list[tuple[int, float]]:
        np = self.np
        count = min(limit, int(self.ids.size))
        if count <= 0:
            return []
        query = np.asarray(query_vec, dtype=np.float32)
        norm = float(np.linalg.norm(query)) or 1.0
        scores = self.matrix @ (query / norm)
        if scope_sql:
            allowed = np.asarray(_scope_ids(conn, scope_sql, scope_params), dtype=np.int64)
            if allowed.size == 0:
                return []
            scores = np.where(np.isin(self.ids, allowed), scores, -np.inf)
        if count < self.ids.size:
            top = np.argpartition(scores, -count)[-count:]
        else:
            top = np.arange(self.ids.size)
        # lexsort matches the Python scan: score desc, then chunk id asc.
        order = top[np.lexsort((self.ids[top], -scores[top]))]
        return [
            (int(self.ids[i]), float(scores[i])) for i in order if math.isfinite(float(scores[i]))
        ][:limit]


class UsearchIndex:
    """Approximate HNSW search (usearch). Recall is measured, never assumed."""

    name = "usearch"

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        dtype: str = "f16",
        connectivity: int = 16,
        expansion_add: int = 128,
        expansion_search: int = 256,
    ) -> None:
        Index = _usearch()
        if Index is None:
            raise RuntimeError("usearch is not installed")
        np = _numpy()
        if np is None:
            raise RuntimeError("numpy is required for the usearch backend")
        self.np = np
        ids, matrix = _load_matrix(conn)
        self.dim = int(matrix.shape[1]) if matrix.size else 0
        self.ids = ids.astype(np.uint64)
        self.index = None
        if matrix.size:
            self.index = Index(
                ndim=self.dim,
                metric="cos",
                dtype=dtype,
                connectivity=connectivity,
                expansion_add=expansion_add,
                expansion_search=expansion_search,
            )
            self.index.add(self.ids, matrix)

    def search(
        self,
        conn: sqlite3.Connection,
        query_vec: list[float],
        limit: int,
        scope_sql: str = "",
        scope_params: Any = (),
    ) -> list[tuple[int, float]]:
        if self.index is None:
            return []
        count = min(limit, int(self.ids.size))
        matches = self.index.search(self.np.asarray(query_vec, dtype=self.np.float32), count)
        return [
            (int(key), float(1.0 - distance))
            for key, distance in zip(matches.keys, matches.distances, strict=True)
        ]


def pick_backend(conn: sqlite3.Connection, prefer: str = "") -> str:
    """Backend name for this corpus: explicit preference, else numpy when available.

    The preference comes from user-editable settings, so an unknown or
    unavailable value falls back to auto with a warning instead of bricking
    retrieval. The benchmark (docs/vector-scale.md) showed the numpy matrix is
    exact, faster than sqlite-vec and smaller than usearch at every measured
    size, so it is the automatic choice; usearch stays opt-in for special cases
    and ``python`` is the dependency-free fallback.
    """
    if prefer in ("", "auto"):
        return "numpy" if available("numpy") else "python"
    if prefer not in BACKENDS:
        _warn(f"unknown vector backend {prefer!r}; using auto")
        return pick_backend(conn)
    if not available(prefer):
        _warn(f"vector backend {prefer!r} is not installed; using auto")
        return pick_backend(conn)
    return prefer


def build_backend(conn: sqlite3.Connection, name: str, *, dtype: str = "f16"):
    if name == "python":
        return PythonScan(conn)
    if name == "numpy":
        return NumpyMatrix(conn)
    if name == "usearch":
        return UsearchIndex(conn, dtype=dtype)
    raise ValueError(f"unknown vector backend {name!r}")


def cached_backend(
    path: str | Path,
    conn: sqlite3.Connection,
    *,
    prefer: str = "",
):
    """Shared backend for this database, rebuilt whenever the file changes."""
    if str(path) == ":memory:":
        # An in-memory database cannot be watched from a second connection.
        return build_backend(conn, pick_backend(conn, prefer))
    key = str(path)
    guard = _guards.get(key)
    if guard is None:
        with _lock:
            guard = _guards.get(key)
            if guard is None:
                # timeout= is the busy timeout: without it a search during a
                # watcher write raises "database is locked" instead of waiting.
                guard = sqlite3.connect(key, check_same_thread=False, timeout=5)
                _guards[key] = guard
    backend_name = pick_backend(conn, prefer)
    token = (_revision(guard), _dim(conn), backend_name)
    entry = _cache.get(key)
    if entry is not None and entry[0] == token:
        # A hit is a use: keep hot databases away from the eviction end.
        with _lock:
            if key in _cache:
                _cache[key] = _cache.pop(key)  # reinsert moves it to the end
        return entry[1]
    with _lock:
        entry = _cache.get(key)
        if entry is not None and entry[0] == token:
            return entry[1]
        backend = build_backend(conn, backend_name)
        # Re-inserting a key must move it to the end, or the first database seen
        # is always the one evicted, however hot it is.
        _cache.pop(key, None)
        _cache[key] = (token, backend)
        _evict_locked()
        return backend


def _evict_locked() -> None:
    """Drop the oldest entries; guards are dropped, never closed.

    Another thread may be reading a revision through one of these connections
    at this moment, and closing it under them raises "Cannot operate on a
    closed database". The last reference finalizes the connection.
    """
    while len(_cache) > MAX_CACHE:
        oldest = next(iter(_cache))
        _cache.pop(oldest, None)
        _guards.pop(oldest, None)


def clear_cache() -> None:
    """Drop every cached index (tests only — production lets the guard expire them)."""
    with _lock:
        _cache.clear()
        for guard in _guards.values():
            guard.close()
        _guards.clear()
