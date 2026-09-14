"""SQLite-backed chunk store: FTS5 (BM25) + float32 vectors."""

from __future__ import annotations

import array
import json
import math
import re
import sqlite3
from datetime import UTC, datetime
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
CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    citations TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS messages_chat_idx ON messages(chat_id);
CREATE TABLE IF NOT EXISTS answer_cache (
    key TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    citations TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
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

    def document_text(self, path: str) -> str | None:
        rows = self.conn.execute(
            "SELECT c.text FROM chunks c JOIN documents d ON d.id = c.doc_id "
            "WHERE d.path = ? ORDER BY c.ordinal",
            (path,),
        ).fetchall()
        if not rows:
            return None
        return "\n\n".join(row["text"] for row in rows)

    def corpus_revision(self) -> str:
        """Latest document write time — changes whenever the index content does."""
        row = self.conn.execute(
            "SELECT COALESCE(MAX(indexed_at), '') AS rev FROM documents"
        ).fetchone()
        return str(row["rev"])

    def sources(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT d.source AS source, COUNT(DISTINCT d.id) AS documents, "
            "COUNT(c.id) AS chunks, MAX(d.indexed_at) AS indexed_at "
            "FROM documents d LEFT JOIN chunks c ON c.doc_id = d.id "
            "GROUP BY d.source ORDER BY d.source"
        ).fetchall()
        return [dict(row) for row in rows]

    # --- local roots ------------------------------------------------------------

    def set_local_roots(self, roots: list[str]) -> None:
        """Remember the chosen paths so the Indexed tab can break them down."""
        stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        entries = {entry["path"]: entry for entry in self._local_roots()}
        for root in roots:
            entries[root] = {"path": root, "indexed_at": stamp}
        self.set_meta("local_roots", json.dumps(list(entries.values())))

    def _local_roots(self) -> list[dict[str, str]]:
        raw = self.get_meta("local_roots")
        if not raw:
            return []
        try:
            entries = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return [
            entry
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("path"), str)
        ]

    def local_paths(self) -> list[dict[str, Any]]:
        """Per chosen path: documents, chunks and the last index run."""
        rows = self.conn.execute(
            "SELECT d.path AS path, COUNT(c.id) AS chunks FROM documents d "
            "LEFT JOIN chunks c ON c.doc_id = d.id WHERE d.source = 'local' "
            "GROUP BY d.id"
        ).fetchall()
        out: list[dict[str, Any]] = []
        for entry in self._local_roots():
            root = str(entry["path"])
            prefix = root.rstrip("/") + "/"
            under = [
                row for row in rows if row["path"] == root or row["path"].startswith(prefix)
            ]
            out.append(
                {
                    "path": root,
                    "documents": len(under),
                    "chunks": sum(int(row["chunks"]) for row in under),
                    "indexed_at": str(entry.get("indexed_at", "")),
                }
            )
        return out

    # --- chat history -----------------------------------------------------------

    def create_chat(self, title: str) -> int:
        with self.conn:
            cursor = self.conn.execute("INSERT INTO chats (title) VALUES (?)", (title,))
        return int(cursor.lastrowid)

    def add_message(
        self, chat_id: int, role: str, text: str, citations: list | None = None
    ) -> int:
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO messages (chat_id, role, text, citations) VALUES (?, ?, ?, ?)",
                (chat_id, role, text, json.dumps(citations or [])),
            )
            self.conn.execute(
                "UPDATE chats SET updated_at = datetime('now') WHERE id = ?", (chat_id,)
            )
        return int(cursor.lastrowid)

    def chats(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT c.id, c.title, c.updated_at, COUNT(m.id) AS messages "
            "FROM chats c LEFT JOIN messages m ON m.chat_id = c.id "
            "GROUP BY c.id ORDER BY c.updated_at DESC, c.id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def chat(self, chat_id: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT id, title, created_at, updated_at FROM chats WHERE id = ?", (chat_id,)
        ).fetchone()
        if row is None:
            return None
        messages = self.conn.execute(
            "SELECT id, role, text, citations, created_at FROM messages "
            "WHERE chat_id = ? ORDER BY id",
            (chat_id,),
        ).fetchall()
        return {
            "chat": dict(row),
            "messages": [
                {**dict(message), "citations": json.loads(message["citations"] or "[]")}
                for message in messages
            ],
        }

    def delete_chat(self, chat_id: int) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))

    def recent_turns(self, chat_id: int, turns: int = 3) -> list[tuple[str, str]]:
        """Last ``turns`` exchanges as (role, text), oldest first."""
        rows = self.conn.execute(
            "SELECT role, text FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
            (chat_id, turns * 2),
        ).fetchall()
        return [(str(row["role"]), str(row["text"])) for row in reversed(rows)]

    # --- answer cache -----------------------------------------------------------

    def cache_get(self, key: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT question, answer, citations FROM answer_cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return {
            "question": row["question"],
            "answer": row["answer"],
            "citations": json.loads(row["citations"] or "[]"),
        }

    def cache_put(self, key: str, question: str, answer: str, citations: list) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO answer_cache (key, question, answer, citations) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET "
                "answer = excluded.answer, citations = excluded.citations, "
                "created_at = datetime('now')",
                (key, question, answer, json.dumps(citations)),
            )
            # Answer keys change with the corpus, so old rows are dead weight.
            self.conn.execute(
                "DELETE FROM answer_cache WHERE key NOT IN "
                "(SELECT key FROM answer_cache ORDER BY created_at DESC LIMIT 500)"
            )

    def cache_clear(self) -> int:
        with self.conn:
            cursor = self.conn.execute("DELETE FROM answer_cache")
        return int(cursor.rowcount)

    # --- memory -----------------------------------------------------------------

    def add_memory(self, text: str, embedding: list[float]) -> int:
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO memories (text, embedding) VALUES (?, ?)",
                (text, array.array("f", embedding).tobytes()),
            )
        return int(cursor.lastrowid)

    def memories(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id, text, created_at FROM memories ORDER BY id DESC"
        ).fetchall()
        return [dict(row) for row in rows]

    def memory_vectors(self) -> list[tuple[int, str, list[float]]]:
        rows = self.conn.execute("SELECT id, text, embedding FROM memories").fetchall()
        out: list[tuple[int, str, list[float]]] = []
        for row in rows:
            vector = array.array("f")
            vector.frombytes(row["embedding"])
            out.append((int(row["id"]), str(row["text"]), list(vector)))
        return out

    def delete_memory(self, memory_id: int) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

    def has_memory(self, text: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM memories WHERE lower(text) = lower(?)", (text.strip(),)
        ).fetchone()
        return row is not None

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
