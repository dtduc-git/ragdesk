"""SQLite-backed chunk store: FTS5 (BM25) + float32 vectors."""

from __future__ import annotations

import array
import hashlib
import json
import math
import re
import sqlite3
import unicodedata
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from ragdesk import settings as app_settings
from ragdesk import vectors

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
CREATE TABLE IF NOT EXISTS embed_cache (
    key TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
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


def matches_any(path: str, patterns: list[str]) -> bool:
    """True when the full path or the file name matches any user glob."""
    name = Path(path).name
    return any(fnmatch(path, pattern) or fnmatch(name, pattern) for pattern in patterns if pattern)


class EmbedderMismatch(RuntimeError):
    """Raised when the configured embedder does not match the existing index."""


class Store:
    """One SQLite file holds documents, chunks, the BM25 index and the vectors."""

    def __init__(self, path: str | Path, *, vector_backend: str = "") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_backend = vector_backend
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA busy_timeout = 5000")
        try:
            # WAL: readers (search) never block on the writer (watcher, another
            # process). A locked file from an older instance just keeps its mode.
            self.conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.OperationalError:
            pass
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
            existing = {row["name"] for row in self.conn.execute(f"PRAGMA table_info({table})")}
            for name, spec in columns.items():
                if name not in existing:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {spec}")
        self._rebuild_fts_if_external()
        self._prune_orphan_fts_rows()
        self._backfill_parents()
        self.conn.commit()

    def _prune_orphan_fts_rows(self) -> int:
        """Delete FTS rows whose chunk is gone.

        A surviving FTS row wedges indexing: the next chunk that reuses its id
        fails the ``chunks_fts`` insert ("constraint failed") and the file can
        never be indexed again. Seen in the wild (a doc's chunks replaced while
        its FTS rows survived), so the repair runs on every open, cheaply.
        """
        row = self.conn.execute(
            "SELECT (SELECT COUNT(*) FROM chunks_fts) AS fts, "
            "(SELECT COUNT(*) FROM chunks) AS chunks"
        ).fetchone()
        if row is None or int(row["fts"]) <= int(row["chunks"]):
            return 0
        with self.conn:
            cursor = self.conn.execute(
                "DELETE FROM chunks_fts WHERE rowid NOT IN (SELECT id FROM chunks)"
            )
        return int(cursor.rowcount)

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
            parents, assignment = group_parents([str(chunk["text"]) for chunk in chunks], max_chars)
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

    def _bump_chunks_revision(self) -> None:
        """Signal cached dense indexes that the vectors changed (see vectors.py)."""
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES ('chunks_revision', '1') "
            "ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)"
        )

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
        self._bump_chunks_revision()

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
            row = self.conn.execute("SELECT id FROM documents WHERE path = ?", (path,)).fetchone()
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
            self._bump_chunks_revision()

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

    def web_pages(self, limit: int = 200) -> list[dict[str, Any]]:
        """Pages saved from the web — the bookmark list behind the Sources tab."""
        rows = self.conn.execute(
            "SELECT path, source, indexed_at, metadata FROM documents "
            "WHERE source LIKE 'web:%' ORDER BY indexed_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            metadata = json.loads(row["metadata"] or "{}")
            url = str(metadata.get("url") or "").strip()
            if not url:
                # pages saved before the url metadata existed: rebuild a best guess
                host_path = str(row["path"])[len("web://") :]
                url = f"https://{host_path}" if host_path else str(row["path"])
            out.append(
                {
                    "url": url,
                    "path": str(row["path"]),
                    "source": str(row["source"]),
                    "indexed_at": str(row["indexed_at"]),
                }
            )
        return out

    def record_index_report(self, report: dict[str, Any]) -> None:
        """Keep the last local index run for the health card."""
        self.set_meta("last_index_report", json.dumps(report))

    def last_index_report(self) -> dict[str, Any]:
        raw = self.get_meta("last_index_report")
        if not raw:
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def oldest_documents(self, limit: int = 5) -> list[dict[str, Any]]:
        """Local docs untouched the longest — the ones worth reviewing."""
        rows = self.conn.execute(
            "SELECT path, mtime, indexed_at FROM documents "
            "WHERE mtime > 0 ORDER BY mtime ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def delete_documents_matching(self, patterns: list[str]) -> list[str]:
        """Drop documents whose path matches a never-index glob; returns paths.

        This is the redaction half of never-index: adding a pattern while the
        matching documents are already indexed would otherwise leave them
        searchable forever.
        """
        removed: list[str] = []
        with self.conn:
            for row in self.conn.execute("SELECT id, path FROM documents").fetchall():
                path = str(row["path"])
                if matches_any(path, patterns):
                    self._delete_doc(int(row["id"]))
                    removed.append(path)
        return removed

    def touch_document(self, path: str, mtime: float) -> None:
        """Record a new mtime for unchanged content so the watcher fast-path holds."""
        with self.conn:
            self.conn.execute("UPDATE documents SET mtime = ? WHERE path = ?", (mtime, path))

    def doc_mtime(self, path: str) -> float | None:
        row = self.conn.execute("SELECT mtime FROM documents WHERE path = ?", (path,)).fetchone()
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
            under = [row for row in rows if row["path"] == root or row["path"].startswith(prefix)]
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

    def add_message(self, chat_id: int, role: str, text: str, citations: list | None = None) -> int:
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
                        str(item.get("path", "")) for item in citations if item.get("path")
                    ],
                }
            )
        return out

    def refused_questions(self, limit: int = 25) -> list[dict[str, Any]]:
        """Questions that got a refusal, newest first, grouped by wording.

        A refusal is stored verbatim, so this is one ordered scan of assistant
        messages; ``resolved`` marks a wording whose *latest* reply did answer
        (the user added a source or rephrased), so the report does not keep
        listing gaps that are closed.
        """
        from ragdesk.answer import REFUSAL  # noqa: PLC0415 - avoids an import cycle

        latest: dict[str, bool] = {}
        refused: dict[str, dict[str, Any]] = {}
        for row in self.conn.execute(
            """
            SELECT m.id, m.created_at, m.text,
                   (SELECT u.text FROM messages u
                    WHERE u.chat_id = m.chat_id AND u.id < m.id AND u.role = 'user'
                    ORDER BY u.id DESC LIMIT 1) AS question
            FROM messages m
            WHERE m.role = 'assistant'
            ORDER BY m.id
            """
        ):
            question = str(row["question"] or "").strip()
            if not question:
                continue
            was_refusal = str(row["text"]) == REFUSAL
            latest[question] = was_refusal
            if not was_refusal:
                continue
            entry = refused.setdefault(
                question, {"question": question, "count": 0, "last_seen": ""}
            )
            entry["count"] += 1
            entry["last_seen"] = str(row["created_at"])
        rows = [
            {**entry, "resolved": not latest.get(entry["question"], True)}
            for entry in refused.values()
        ]
        rows.sort(key=lambda entry: (entry["count"], entry["last_seen"]), reverse=True)
        return rows[:limit]

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

    # --- embedding cache ----------------------------------------------------------

    EMBED_CACHE_MAX = 50_000

    @staticmethod
    def embed_key(embedder_name: str, dim: int, text: str) -> str:
        """Identity of one embedding: model + dimensions + exact chunk text."""
        return hashlib.sha1(f"{embedder_name}|{dim}|{text}".encode()).hexdigest()

    def embed_cache_get(self, keys: list[str]) -> dict[str, list[float]]:
        """Vectors for the keys that are already cached (misses are absent)."""
        found: dict[str, list[float]] = {}
        for start in range(0, len(keys), 500):
            batch = keys[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = self.conn.execute(
                f"SELECT key, embedding FROM embed_cache WHERE key IN ({placeholders})",  # noqa: S608 - placeholders only
                batch,
            ).fetchall()
            for row in rows:
                vec = array.array("f")
                vec.frombytes(row["embedding"])
                found[str(row["key"])] = list(vec)
        return found

    def embed_cache_put(self, pairs: list[tuple[str, list[float]]]) -> None:
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO embed_cache (key, embedding) VALUES (?, ?)",
                [(key, array.array("f", vector).tobytes()) for key, vector in pairs],
            )
            self.conn.execute(
                "DELETE FROM embed_cache WHERE key NOT IN "
                "(SELECT key FROM embed_cache ORDER BY created_at DESC LIMIT ?)",
                (self.EMBED_CACHE_MAX,),
            )

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

    # --- corrections --------------------------------------------------------------

    def add_correction(self, question: str, answer: str, embedding: list[float]) -> int:
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO corrections (question, answer, embedding) VALUES (?, ?, ?)",
                (question, answer, array.array("f", embedding).tobytes()),
            )
        return int(cursor.lastrowid)

    def corrections(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id, question, answer, created_at FROM corrections ORDER BY id DESC"
        ).fetchall()
        return [dict(row) for row in rows]

    def delete_correction(self, correction_id: int) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM corrections WHERE id = ?", (correction_id,))

    def corrections_revision(self) -> str:
        """Changes whenever a correction is added or removed (cache invalidation)."""
        row = self.conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(MAX(id), 0) AS last FROM corrections"
        ).fetchone()
        return f"{row['n']}:{row['last']}"

    def nearest_correction(
        self, embedding: list[float], min_cosine: float
    ) -> dict[str, Any] | None:
        """Closest correction for this question, if one is close enough."""
        rows = self.conn.execute(
            "SELECT id, question, answer, embedding FROM corrections ORDER BY id DESC"
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
                        "id": int(row["id"]),
                        "question": str(row["question"]),
                        "answer": str(row["answer"]),
                        "cosine": score,
                    },
                )
        return best[1] if best else None

    # --- navigation -------------------------------------------------------------

    def _doc_vectors(self, exclude_doc: int = 0) -> dict[int, tuple[str, list[float]]]:
        rows = self.conn.execute(
            "SELECT d.id, d.path, c.embedding FROM documents d "
            "JOIN chunks c ON c.doc_id = d.id "
            "WHERE d.id != ? "
            "ORDER BY d.id, c.ordinal LIMIT 40000",
            (exclude_doc,),
        ).fetchall()
        buckets: dict[int, tuple[str, list[float], int]] = {}
        for row in rows:
            vec = array.array("f")
            vec.frombytes(row["embedding"])
            entry = buckets.get(row["id"])
            values = list(vec)
            if entry is None:
                buckets[row["id"]] = (str(row["path"]), values, 1)
            else:
                summed = [a + b for a, b in zip(entry[1], values, strict=False)]
                buckets[row["id"]] = (entry[0], summed, entry[2] + 1)
        return {
            doc_id: (path, [value / count for value in summed])
            for doc_id, (path, summed, count) in buckets.items()
        }

    def related_documents(self, path: str, limit: int = 5) -> list[dict[str, Any]]:
        """Other documents closest to this one's average chunk vector."""
        row = self.conn.execute("SELECT id FROM documents WHERE path = ?", (path,)).fetchone()
        if row is None:
            return []
        doc_id = int(row["id"])
        vectors = self._doc_vectors()
        mine = vectors.get(doc_id)
        if mine is None:
            return []
        norm = math.sqrt(sum(v * v for v in mine[1])) or 1.0
        scored: list[tuple[float, str]] = []
        for other_id, (other_path, vector) in vectors.items():
            if other_id == doc_id:
                continue
            other_norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            dot = sum(a * b for a, b in zip(mine[1], vector, strict=False))
            scored.append((dot / (norm * other_norm), other_path))
        scored.sort(key=lambda item: -item[0])
        return [
            {"path": other_path, "score": round(score, 3)} for score, other_path in scored[:limit]
        ]

    def duplicate_clusters(
        self, min_ratio: float = 0.5, min_shared: int = 3
    ) -> list[dict[str, Any]]:
        """Documents that share a large fraction of their exact chunks.

        Chunking is deterministic, so a copy of a file (or a slice export of
        it) hashes to the same chunk texts while topical neighbours do not.
        Containment (shared / smaller side) catches "the whole book plus its
        part-2 slice", which document-level cosine cannot separate from a
        similar book on the same topic.
        """
        paths = {
            int(row["id"]): str(row["path"])
            for row in self.conn.execute("SELECT id, path FROM documents")
        }
        digests: dict[int, set[str]] = {}
        for row in self.conn.execute("SELECT doc_id, text FROM chunks"):
            digests.setdefault(int(row["doc_id"]), set()).add(
                hashlib.sha1(str(row["text"]).encode()).hexdigest()
            )
        parent = {doc_id: doc_id for doc_id in digests}

        def find(doc_id: int) -> int:
            while parent[doc_id] != doc_id:
                parent[doc_id] = parent[parent[doc_id]]
                doc_id = parent[doc_id]
            return doc_id

        ids = [doc_id for doc_id, hashes in digests.items() if len(hashes) >= min_shared]
        matches: list[tuple[int, int, int, float]] = []
        for index, left in enumerate(ids):
            for right in ids[index + 1 :]:
                shared = len(digests[left] & digests[right])
                if shared < min_shared:
                    continue
                ratio = shared / min(len(digests[left]), len(digests[right]))
                if ratio < min_ratio:
                    continue
                matches.append((left, right, shared, ratio))
                root_left, root_right = find(left), find(right)
                if root_left != root_right:
                    parent[root_right] = root_left

        clusters: dict[int, dict[str, Any]] = {}
        for left, right, shared, ratio in matches:
            entry = clusters.setdefault(
                find(left), {"paths": set(), "shared_chunks": 0, "ratio": 0.0}
            )
            entry["paths"].update((left, right))
            entry["shared_chunks"] = max(entry["shared_chunks"], shared)
            entry["ratio"] = max(entry["ratio"], ratio)
        return sorted(
            (
                {
                    "paths": sorted(paths[doc_id] for doc_id in entry["paths"]),
                    "shared_chunks": entry["shared_chunks"],
                    "ratio": round(entry["ratio"], 3),
                }
                for entry in clusters.values()
            ),
            key=lambda entry: (-entry["ratio"], -entry["shared_chunks"]),
        )

    def backlinks(self, path: str, limit: int = 8) -> list[dict[str, Any]]:
        """Documents that mention this file's name, via the folded FTS index."""
        stem = fold_text(Path(path).stem.replace("_", " "))
        tokens = [token for token in TOKEN_RE.findall(stem) if len(token) >= 3]
        if not tokens:
            return []
        match = " OR ".join(f'"{token}"' for token in tokens[:4])
        rows = self.conn.execute(
            """
            SELECT d.path, COUNT(*) AS hits FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN documents d ON d.id = c.doc_id
            WHERE chunks_fts MATCH ? AND d.path != ?
            GROUP BY d.path ORDER BY hits DESC LIMIT ?
            """,
            (match, path, limit),
        ).fetchall()
        return [{"path": str(row["path"]), "hits": int(row["hits"])} for row in rows]

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

    def bm25_search(self, query: str, limit: int, filters: Any = None) -> list[dict[str, Any]]:
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

    def path_search(self, query: str, limit: int, filters: Any = None) -> list[dict[str, Any]]:
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
        if limit <= 0:
            return []
        scope, params = self._filter_sql(filters)
        prefer = (
            self.vector_backend
            or vectors.backend_override()
            or str(app_settings.load().get("vector_backend") or "")
        )
        backend = vectors.cached_backend(self.path, self.conn, prefer=prefer)
        if backend.name == "usearch" and scope:
            # HNSW cannot filter; a scoped query scans only the matching chunks,
            # so the exact Python scan stays proportional to the scope subset.
            backend = vectors.PythonScan(self.conn)
        candidates = backend.search(self.conn, query_vec, limit, scope, params)
        return self._payloads(candidates)

    def _payloads(self, candidates: list[tuple[int, float]]) -> list[dict[str, Any]]:
        """Fetch display fields for the winning chunk ids, best score first."""
        if not candidates:
            return []
        placeholders = ",".join("?" * len(candidates))
        rows = self.conn.execute(
            f"""
            SELECT c.id, c.doc_id, c.ordinal, c.text, d.path, d.source,
                   COALESCE(p.text, '') AS parent_text, c.line_start, d.mtime, d.metadata
            FROM chunks c JOIN documents d ON d.id = c.doc_id
            LEFT JOIN parents p ON p.doc_id = d.id AND p.ordinal = c.parent_ordinal
            WHERE c.id IN ({placeholders})
            """,
            [chunk_id for chunk_id, _score in candidates],
        ).fetchall()
        by_id = {int(row["id"]): dict(row) for row in rows}
        out: list[dict[str, Any]] = []
        for chunk_id, score in candidates:
            row = by_id.get(chunk_id)
            if row is None:
                continue
            row["metadata"] = json.loads(row["metadata"] or "{}")
            row["score"] = score
            out.append(row)
        return out
