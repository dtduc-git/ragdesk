"""Dense vector search backends for the chunk index.

Embeddings are fp32 blobs in ``chunks.embedding``. A plain Python scan is
always available; the numpy matrix covers personal corpora (up to ~1M chunks
in RAM); usearch adds an approximate HNSW index as an explicit opt-in
(``docs/vector-scale.md`` has the measurements). Every backend returns
``(chunk_id, score)`` candidates sorted by score, so the store keeps one
payload-fetch and one ranking pipeline.

Indexes are cached per database path at module level because the desktop
server opens a fresh ``Store`` per request. A long-lived guard connection
watches ``PRAGMA data_version``, so any commit from any connection (watcher
thread, CLI, a second app instance) invalidates the cached index.
"""

from __future__ import annotations

import array
import math
import sqlite3
import threading
from pathlib import Path
from typing import Any

BACKENDS = ("python", "numpy", "usearch")
_cache: dict[str, tuple[tuple[Any, ...], Any]] = {}
_guards: dict[str, sqlite3.Connection] = {}
_lock = threading.Lock()


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


def _dim(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT length(embedding) FROM chunks LIMIT 1").fetchone()
    return (int(row[0]) // 4) if row and row[0] else 0


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
        total = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        self.ids = np.empty(total, dtype=np.int64)
        self.matrix = np.empty((total, _dim(conn)), dtype=np.float32)
        # Stream rows: fetchall() would hold every blob in Python memory next
        # to the matrix, doubling the peak at large corpora.
        index = 0
        for row in conn.execute("SELECT id, embedding FROM chunks ORDER BY id"):
            self.ids[index] = row[0]
            self.matrix[index] = np.frombuffer(row[1], dtype=np.float32)
            index += 1
        self.ids = self.ids[:index]
        self.matrix = self.matrix[:index]
        if index:
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
        rows = conn.execute("SELECT id, embedding FROM chunks ORDER BY id").fetchall()
        self.dim = len(rows[0][1]) // 4 if rows else 0
        self.ids = np.array([int(row[0]) for row in rows], dtype=np.uint64)
        self.index = None
        if rows:
            vectors = np.empty((len(rows), self.dim), dtype=np.float32)
            for index, row in enumerate(rows):
                vectors[index] = np.frombuffer(row[1], dtype=np.float32)
            self.index = Index(
                ndim=self.dim,
                metric="cos",
                dtype=dtype,
                connectivity=connectivity,
                expansion_add=expansion_add,
                expansion_search=expansion_search,
            )
            self.index.add(self.ids, vectors)

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

    The benchmark (docs/vector-scale.md) showed the numpy matrix is exact,
    faster than sqlite-vec and smaller than usearch at every measured size, so
    it is the automatic choice; usearch stays opt-in for special cases and
    ``python`` is the dependency-free fallback.
    """
    if prefer in BACKENDS:
        return prefer
    if prefer not in ("", "auto"):
        raise ValueError(
            f"unknown vector backend {prefer!r} (expected one of {BACKENDS} or 'auto')"
        )
    return "numpy" if _numpy() is not None else "python"


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
                guard = sqlite3.connect(key, check_same_thread=False)
                _guards[key] = guard
    token = (guard.execute("PRAGMA data_version").fetchone()[0], _dim(conn), prefer)
    entry = _cache.get(key)
    if entry is not None and entry[0] == token:
        return entry[1]
    with _lock:
        entry = _cache.get(key)
        if entry is not None and entry[0] == token:
            return entry[1]
        backend = build_backend(conn, pick_backend(conn, prefer))
        _cache[key] = (token, backend)
        return backend


def clear_cache() -> None:
    """Drop every cached index (tests only — production lets the guard expire them)."""
    with _lock:
        _cache.clear()
        for guard in _guards.values():
            guard.close()
        _guards.clear()
