"""SQLite-backed chunk store: FTS5 (BM25) + float32 vectors."""

from __future__ import annotations

import array
import math
import re
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL,
    mtime REAL NOT NULL,
    indexed_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id'
);
CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks(doc_id);
"""

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class EmbedderMismatch(RuntimeError):
    """Raised when the configured embedder does not match the existing index."""


class Store:
    """One SQLite file holds documents, chunks, the BM25 index and the vectors."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA busy_timeout = 5000")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- meta -----------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def ensure_embedder(self, name: str, dim: int) -> None:
        """Fail-closed guard: vectors from different models must not mix."""
        known_name = self.get_meta("embedder.name")
        known_dim = self.get_meta("embedder.dim")
        if known_name is None and known_dim is None:
            self.set_meta("embedder.name", name)
            self.set_meta("embedder.dim", str(dim))
            return
        if known_name != name or known_dim != str(dim):
            raise EmbedderMismatch(
                f"index was built with {known_name!r} (dim {known_dim}); current "
                f"embedder is {name!r} (dim {dim}). Use the matching embedder or "
                f"rebuild the index."
            )

    # --- documents/chunks -----------------------------------------------------

    def doc_hash(self, path: str) -> str | None:
        row = self.conn.execute(
            "SELECT content_hash FROM documents WHERE path = ?", (path,)
        ).fetchone()
        return row["content_hash"] if row else None

    def _delete_doc(self, doc_id: int) -> None:
        rows = self.conn.execute(
            "SELECT id, text FROM chunks WHERE doc_id = ?", (doc_id,)
        ).fetchall()
        for row in rows:
            self.conn.execute(
                "INSERT INTO chunks_fts (chunks_fts, rowid, text) VALUES ('delete', ?, ?)",
                (row["id"], row["text"]),
            )
        self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    def upsert_document(
        self,
        *,
        source: str,
        path: str,
        content_hash: str,
        mtime: float,
        texts: list[str],
        embeddings: list[list[float]],
    ) -> None:
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have the same length")
        with self.conn:
            row = self.conn.execute(
                "SELECT id FROM documents WHERE path = ?", (path,)
            ).fetchone()
            if row:
                self._delete_doc(row["id"])
            cursor = self.conn.execute(
                "INSERT INTO documents (source, path, content_hash, mtime) VALUES (?, ?, ?, ?)",
                (source, path, content_hash, mtime),
            )
            doc_id = cursor.lastrowid
            for ordinal, (text, vec) in enumerate(zip(texts, embeddings, strict=True)):
                chunk = self.conn.execute(
                    "INSERT INTO chunks (doc_id, ordinal, text, embedding) VALUES (?, ?, ?, ?)",
                    (doc_id, ordinal, text, array.array("f", vec).tobytes()),
                )
                self.conn.execute(
                    "INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)",
                    (chunk.lastrowid, text),
                )

    def documents(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id, source, path, content_hash, mtime, indexed_at FROM documents "
            "ORDER BY path"
        ).fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> dict[str, int]:
        docs = self.conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        chunks = self.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        return {"documents": docs, "chunks": chunks}

    # --- search lanes -----------------------------------------------------------

    def bm25_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        tokens = TOKEN_RE.findall(query.lower())
        if not tokens:
            return []
        match = " OR ".join(f'"{token}"' for token in tokens)
        rows = self.conn.execute(
            """
            SELECT c.id, c.doc_id, c.ordinal, c.text, d.path, d.source,
                   bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN documents d ON d.id = c.doc_id
            WHERE chunks_fts MATCH ?
            ORDER BY bm25(chunks_fts)
            LIMIT ?
            """,
            (match, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def dense_search(self, query_vec: list[float], limit: int) -> list[dict[str, Any]]:
        # ponytail: brute-force cosine over all chunks. Fine to ~100k chunks for a
        # personal index; swap in sqlite-vec / a cached numpy matrix when it grows.
        rows = self.conn.execute(
            """
            SELECT c.id, c.doc_id, c.ordinal, c.text, d.path, d.source, c.embedding
            FROM chunks c JOIN documents d ON d.id = c.doc_id
            """
        ).fetchall()
        q_norm = math.sqrt(sum(v * v for v in query_vec)) or 1.0
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            vec = array.array("f")
            vec.frombytes(row["embedding"])
            dot = sum(a * b for a, b in zip(query_vec, vec, strict=True))
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            score = dot / (q_norm * norm)
            scored.append(
                (
                    score,
                    {
                        "id": row["id"],
                        "doc_id": row["doc_id"],
                        "ordinal": row["ordinal"],
                        "text": row["text"],
                        "path": row["path"],
                        "source": row["source"],
                        "score": score,
                    },
                )
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        return [payload for _, payload in scored[:limit]]
