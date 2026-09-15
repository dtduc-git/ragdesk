"""SQLite-backed chunk store: FTS5 (BM25) + float32 vectors."""

from __future__ import annotations

import array
import json
import math
import re
import sqlite3
import unicodedata
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
    metadata TEXT NOT NULL DEFAULT '{}',
    indexed_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    parent_ordinal INTEGER NOT NULL DEFAULT 0,
    line_start INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS parents (
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (doc_id, ordinal)
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text);
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
    feedback INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS messages_chat_idx ON messages(chat_id);
CREATE TABLE IF NOT EXISTS answer_cache (
    key TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    citations TEXT NOT NULL DEFAULT '[]',
    fingerprint TEXT NOT NULL DEFAULT '',
    embedding BLOB,
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
_COMBINING = "\u0300-\u036f"


def fold_text(text: str) -> str:
    """Lowercase and strip diacritics so "thue" matches "thuế" (VN search)."""
    # "đ" is a letter of its own, not a decomposed accent, so map it explicitly.
    lowered = text.lower().replace("đ", "d")
    return re.sub(f"[{_COMBINING}]", "", unicodedata.normalize("NFD", lowered))


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
        self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after the first release."""
        migrations = {
            "chunks": {
                "parent_ordinal": "INTEGER NOT NULL DEFAULT 0",
                "line_start": "INTEGER NOT NULL DEFAULT 1",
            },
            "messages": {"feedback": "INTEGER NOT NULL DEFAULT 0"},
            "documents": {"metadata": "TEXT NOT NULL DEFAULT '{}'"},
            "answer_cache": {
                "fingerprint": "TEXT NOT NULL DEFAULT ''",
                "embedding": "BLOB",
            },
        }
        for table, columns in migrations.items():
            existing = {
                row["name"] for row in self.conn.execute(f"PRAGMA table_info({table})")
            }
            for name, spec in columns.items():
                if name not in existing:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {spec}")
        self._rebuild_fts_if_external()
        self._backfill_parents()
        self.conn.commit()

    def _rebuild_fts_if_external(self) -> None:
        """Legacy DBs index raw text through an external-content FTS table;
        replace it with the folded standalone copy and rebuild once."""
        row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'chunks_fts'"
        ).fetchone()
        if row is None or "content='chunks'" not in str(row["sql"]):
            return
        self.conn.execute("DROP TABLE chunks_fts")
        self.conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(text)")
        rows = self.conn.execute("SELECT id, text FROM chunks").fetchall()
        for chunk in rows:
            self.conn.execute(
                "INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)",
                (chunk["id"], fold_text(str(chunk["text"]))),
            )

    def _backfill_parents(self, max_chars: int = 4000) -> int:
        """Group existing chunks into parents — no re-embedding needed."""
        from ragdesk.index import group_parents  # noqa: PLC0415 - shared grouping

        rows = self.conn.execute(
            "SELECT id FROM documents d WHERE NOT EXISTS "
            "(SELECT 1 FROM parents p WHERE p.doc_id = d.id)"
        ).fetchall()
        created = 0
        for row in rows:
            chunks = self.conn.execute(
                "SELECT id, ordinal, text FROM chunks WHERE doc_id = ? ORDER BY ordinal",
                (row["id"],),
            ).fetchall()
            if not chunks:
                continue
            parents, assignment = group_parents(
                [str(chunk["text"]) for chunk in chunks], max_chars
            )
            for ordinal, text in enumerate(parents):
                self.conn.execute(
                    "INSERT INTO parents (doc_id, ordinal, text) VALUES (?, ?, ?)",
                    (row["id"], ordinal, text),
                )
            for chunk, parent_ordinal in zip(chunks, assignment, strict=True):
                self.conn.execute(
                    "UPDATE chunks SET parent_ordinal = ? WHERE id = ?",
                    (parent_ordinal, chunk["id"]),
                )
            created += 1
        return created

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
                "DELETE FROM chunks_fts WHERE rowid = ?",
                (row["id"],),
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
        metadata: dict | None = None,
        texts: list[str],
        embeddings: list[list[float]],
        parents: list[str] | None = None,
        parent_index: list[int] | None = None,
        line_starts: list[int] | None = None,
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
                "INSERT INTO documents (source, path, content_hash, mtime, metadata) "
                "VALUES (?, ?, ?, ?, ?)",
                (source, path, content_hash, mtime, json.dumps(metadata or {})),
            )
            doc_id = cursor.lastrowid
            for ordinal, text in enumerate(parents or []):
                self.conn.execute(
                    "INSERT INTO parents (doc_id, ordinal, text) VALUES (?, ?, ?)",
                    (doc_id, ordinal, text),
                )
            for ordinal, (text, vec) in enumerate(zip(texts, embeddings, strict=True)):
                parent_ordinal = (
                    parent_index[ordinal]
                    if parent_index is not None and ordinal < len(parent_index)
                    else 0
                )
                line_start = (
                    int(line_starts[ordinal])
                    if line_starts is not None and ordinal < len(line_starts)
                    else 1
                )
                chunk = self.conn.execute(
                    "INSERT INTO chunks "
                    "(doc_id, ordinal, text, embedding, parent_ordinal, line_start) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        doc_id,
                        ordinal,
                        text,
                        array.array("f", vec).tobytes(),
                        parent_ordinal,
                        line_start,
                    ),
                )
                self.conn.execute(
                    "INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)",
                    (chunk.lastrowid, fold_text(text)),
                )

    def documents(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id, source, path, content_hash, mtime, indexed_at, metadata FROM documents "
            "ORDER BY path"
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            entry = dict(row)
            entry["metadata"] = json.loads(entry.get("metadata") or "{}")
            out.append(entry)
        return out

    def touch_document(self, path: str, mtime: float) -> None:
        """Record a new mtime for unchanged content so the watcher fast-path holds."""
        with self.conn:
            self.conn.execute(
                "UPDATE documents SET mtime = ? WHERE path = ?", (mtime, path)
            )

    def doc_mtime(self, path: str) -> float | None:
        row = self.conn.execute(
            "SELECT mtime FROM documents WHERE path = ?", (path,)
        ).fetchone()
        return float(row["mtime"]) if row else None

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
            "SELECT id, role, text, citations, feedback, created_at FROM messages "
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

    def set_feedback(self, message_id: int, value: int) -> bool:
        with self.conn:
            cursor = self.conn.execute(
                "UPDATE messages SET feedback = ? WHERE id = ?", (value, message_id)
            )
        return bool(cursor.rowcount)

    def feedback_rows(self) -> list[dict[str, Any]]:
        """Assistant messages that were rated, with the question that produced them."""
        rows = self.conn.execute(
            """
            SELECT m.id, m.chat_id, m.text, m.citations, m.feedback,
                   (SELECT u.text FROM messages u
                    WHERE u.chat_id = m.chat_id AND u.id < m.id AND u.role = 'user'
                    ORDER BY u.id DESC LIMIT 1) AS question
            FROM messages m
            WHERE m.role = 'assistant' AND m.feedback != 0
            ORDER BY m.id
            """
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            citations = json.loads(row["citations"] or "[]")
            out.append(
                {
                    "id": int(row["id"]),
                    "chat_id": int(row["chat_id"]),
                    "question": str(row["question"] or ""),
                    "feedback": int(row["feedback"]),
                    "relevant": [
                        str(item.get("path", ""))
                        for item in citations
                        if item.get("path")
                    ],
                }
            )
        return out

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

    def cache_put(
        self,
        key: str,
        question: str,
        answer: str,
        citations: list,
        embedding: list[float] | None = None,
        fingerprint: str = "",
    ) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO answer_cache "
                "(key, question, answer, citations, embedding, fingerprint) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET "
                "answer = excluded.answer, citations = excluded.citations, "
                "embedding = excluded.embedding, fingerprint = excluded.fingerprint, "
                "created_at = datetime('now')",
                (
                    key,
                    question,
                    answer,
                    json.dumps(citations),
                    array.array("f", embedding).tobytes() if embedding else None,
                    fingerprint,
                ),
            )
            # Answer keys change with the corpus, so old rows are dead weight.
            self.conn.execute(
                "DELETE FROM answer_cache WHERE key NOT IN "
                "(SELECT key FROM answer_cache ORDER BY created_at DESC LIMIT 500)"
            )

    def cache_nearest(
        self, embedding: list[float], fingerprint: str, min_cosine: float
    ) -> dict[str, Any] | None:
        """Closest cached answer for the same corpus generation, if close enough."""
        rows = self.conn.execute(
            "SELECT question, answer, citations, embedding FROM answer_cache "
            "WHERE fingerprint = ? AND embedding IS NOT NULL",
            (fingerprint,),
        ).fetchall()
        q_norm = math.sqrt(sum(v * v for v in embedding)) or 1.0
        best: tuple[float, dict[str, Any]] | None = None
        for row in rows:
            vec = array.array("f")
            vec.frombytes(row["embedding"])
            dot = sum(a * b for a, b in zip(embedding, vec, strict=False))
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            score = dot / (q_norm * norm)
            if score >= min_cosine and (best is None or score > best[0]):
                best = (
                    score,
                    {
                        "question": row["question"],
                        "answer": row["answer"],
                        "citations": json.loads(row["citations"] or "[]"),
                        "cosine": score,
                    },
                )
        return best[1] if best else None

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

    def _filter_sql(self, filters: Any) -> tuple[str, list[Any]]:
        """WHERE fragments for scoping (``folder:`` / ``source:`` prefixes)."""
        clauses: list[str] = []
        params: list[Any] = []
        if filters is not None:
            if getattr(filters, "path_like", ""):
                clauses.append("lower(d.path) LIKE ?")
                params.append(f"%{str(filters.path_like).lower()}%")
            if getattr(filters, "source", ""):
                clauses.append("d.source = ?")
                params.append(str(filters.source))
            for key, value in (getattr(filters, "meta", None) or {}).items():
                safe = re.sub(r"[^a-z0-9_-]", "", key.lower())
                if not safe:
                    continue
                clauses.append(f"lower(json_extract(d.metadata, '$.{safe}')) = ?")
                params.append(str(value).lower())
        return (" AND " + " AND ".join(clauses)) if clauses else "", params

    def bm25_search(
        self, query: str, limit: int, filters: Any = None
    ) -> list[dict[str, Any]]:
        tokens = TOKEN_RE.findall(fold_text(query))
        if not tokens:
            return []
        match = " OR ".join(f'"{token}"' for token in tokens)
        scope, params = self._filter_sql(filters)
        rows = self.conn.execute(
            f"""
            SELECT c.id, c.doc_id, c.ordinal, c.text, d.path, d.source,
                   COALESCE(p.text, '') AS parent_text, c.line_start, d.mtime, d.metadata,
                   bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN documents d ON d.id = c.doc_id
            LEFT JOIN parents p ON p.doc_id = d.id AND p.ordinal = c.parent_ordinal
            WHERE chunks_fts MATCH ?{scope}
            ORDER BY bm25(chunks_fts)
            LIMIT ?
            """,
            (match, *params, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def path_search(
        self, query: str, limit: int, filters: Any = None
    ) -> list[dict[str, Any]]:
        """Match query words against file paths, best (most words) first."""
        tokens = [token for token in TOKEN_RE.findall(query.lower()) if len(token) >= 3]
        if not tokens:
            return []
        tokens = tokens[:5]
        patterns = " OR ".join("lower(d.path) LIKE ?" for _ in tokens)
        scope, scope_params = self._filter_sql(filters)
        rows = self.conn.execute(
            f"""
            SELECT c.id, c.doc_id, c.ordinal, c.text, d.path, d.source,
                   COALESCE(p.text, '') AS parent_text, c.line_start, d.mtime,
                   d.metadata
            FROM documents d
            JOIN chunks c ON c.doc_id = d.id
            LEFT JOIN parents p ON p.doc_id = d.id AND p.ordinal = c.parent_ordinal
            WHERE ({patterns}){scope}
            LIMIT 2000
            """,
            (*[f"%{token}%" for token in tokens], *scope_params),
        ).fetchall()
        scored: list[tuple[int, Any]] = []
        for row in rows:
            path = str(row["path"]).lower()
            matches = sum(1 for token in tokens if token in path)
            scored.append((matches, row))
        scored.sort(key=lambda item: (-item[0], int(item[1]["ordinal"])))
        return [dict(row) for _matches, row in scored[:limit]]

    def dense_search(
        self, query_vec: list[float], limit: int, filters: Any = None
    ) -> list[dict[str, Any]]:
        # ponytail: brute-force cosine over all chunks. Fine to ~100k chunks for a
        # personal index; swap in sqlite-vec / a cached numpy matrix when it grows.
        scope, params = self._filter_sql(filters)
        rows = self.conn.execute(
            f"""
            SELECT c.id, c.doc_id, c.ordinal, c.text, d.path, d.source, c.embedding,
                   COALESCE(p.text, '') AS parent_text, c.line_start, d.mtime,
                   d.metadata
            FROM chunks c JOIN documents d ON d.id = c.doc_id
            LEFT JOIN parents p ON p.doc_id = d.id AND p.ordinal = c.parent_ordinal
            WHERE 1=1{scope}
            """,
            params,
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
                        "parent_text": row["parent_text"],
                        "line_start": row["line_start"],
                        "mtime": row["mtime"],
                        "metadata": json.loads(row["metadata"] or "{}"),
                        "score": score,
                    },
                )
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        return [payload for _, payload in scored[:limit]]
