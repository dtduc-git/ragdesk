"""Minimal MCP server over stdio (dependency-free).

Exposes the local index to MCP clients (Claude Code, Cursor, …) as tools:

- ``ragdesk_search`` — hybrid retrieval over the local index
- ``ragdesk_document`` — full text of one indexed document
- ``ragdesk_sources`` — indexed sources with document/chunk counts
- ``ragdesk_symbol`` — definitions and call sites for a code symbol
- ``ragdesk_topics`` — the corpus grouped into labelled topic clusters
- ``ragdesk_save`` — save one web page into the index (writes to the local
  index only; sources are never written to)

Run with ``ragdesk mcp`` (optional global flags: ``--db``, ``--embedder``,
``--rerank``). Messages are newline-delimited JSON-RPC 2.0 on stdin/stdout.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from ragdesk import __version__
from ragdesk.embed import Embedder
from ragdesk.search import retrieve
from ragdesk.store import Store

PROTOCOL_VERSION = "2024-11-05"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "ragdesk_search",
        "description": (
            "Search the local ragdesk index (hybrid BM25 + embeddings) and return "
            "ranked passages with their source paths."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "natural-language query"},
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "number of passages (default 8)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "ragdesk_document",
        "description": (
            "Fetch the full indexed text of one document by its path "
            "(paths are shown in ragdesk_search results)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "ragdesk_symbol",
        "description": (
            "Where a code symbol is defined and who calls it, scanned from the "
            "indexed code files. Use for 'who calls X' / 'where is X defined'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "symbol name"}},
            "required": ["name"],
        },
    },
    {
        "name": "ragdesk_topics",
        "description": "Group the indexed documents into labelled topic clusters.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "ragdesk_save",
        "description": (
            "Fetch one web page and save it into the local index so future "
            "searches can cite it. Writes to the index, never to your sources."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "ragdesk_sources",
        "description": "List indexed sources with document and chunk counts.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


MCP_BLOCK_CHARS = 4000  # the parent-section limit: clipping is for top_k=50, not the norm


def _clip(text: str, limit: int = MCP_BLOCK_CHARS) -> str:
    """Cut a section at a sentence (or word) boundary, never mid-sentence."""
    if len(text) <= limit:
        return text
    window = text[:limit]
    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "), window.rfind("\n"))
    if cut < limit // 2:
        cut = window.rfind(" ")
    if cut <= 0:
        return window
    return window[: cut + 1].rstrip()


class McpServer:
    def __init__(self, *, db: str, embedder: Embedder, reranker: Any = None) -> None:
        self.db = db
        self.embedder = embedder
        self.reranker = reranker

    # --- protocol -------------------------------------------------------------

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """One JSON-RPC message in, one response out (None for notifications)."""
        method = str(message.get("method", ""))
        msg_id = message.get("id")
        if method == "initialize":
            return self._ok(
                msg_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "ragdesk", "version": __version__},
                },
            )
        if method.startswith("notifications/"):
            return None
        if method == "tools/list":
            return self._ok(msg_id, {"tools": TOOLS})
        if method == "tools/call":
            return self._ok(msg_id, self._call_tool(message.get("params") or {}))
        if method == "ping":
            return self._ok(msg_id, {})
        return self._error(msg_id, -32601, f"method not found: {method}")

    def serve(self) -> int:
        # stderr is the MCP log channel: "silence is the enemy" applies here too.
        label = getattr(self.reranker, "name", None) or "none"
        print(
            f"ragdesk mcp: embedder={self.embedder.name} rerank={label}",
            file=sys.stderr,
            flush=True,
        )
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            response = self.handle(message)
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
        return 0

    # --- tools ----------------------------------------------------------------

    def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = str(params.get("name", ""))
        arguments = params.get("arguments") or {}
        if name == "ragdesk_search":
            query = str(arguments.get("query", "")).strip()
            if not query:
                return self._tool_error("query is required")
            top_k = int(arguments.get("top_k") or 8)
            with Store(self.db) as store:
                hits = retrieve(store, self.embedder, query, top_k=top_k, reranker=self.reranker)
            if not hits:
                return self._tool_text("no matches in the local index")
            blocks: list[str] = []
            seen: set[tuple[str, str]] = set()
            for hit in hits:
                # The parent section when one was stored: a 1000-char window cut
                # at 600 loses the sentence that answers the question. Overlapping
                # chunks of the same section are one block, not a duplicate.
                text = hit.context
                if (hit.path, text) in seen:
                    continue
                seen.add((hit.path, text))
                blocks.append(
                    f"[{len(blocks) + 1}] {hit.path} (score {hit.score:.4f}, lanes {hit.lanes})\n"
                    f"{_clip(text)}"
                )
            return self._tool_text("\n\n".join(blocks))

        if name == "ragdesk_document":
            path = str(arguments.get("path", "")).strip()
            if not path:
                return self._tool_error("path is required")
            with Store(self.db) as store:
                text = store.document_text(path)
            if text is None:
                return self._tool_error(f"not indexed: {path}")
            return self._tool_text(text)

        if name == "ragdesk_symbol":
            symbol = str(arguments.get("name", "")).strip()
            if not symbol:
                return self._tool_error("name is required")
            from ragdesk.symbols import find_symbol, symbol_answer  # noqa: PLC0415

            with Store(self.db) as store:
                result = find_symbol(store, symbol)
            text, _hits = symbol_answer(symbol, result)
            return self._tool_text(text or f"no definitions or call sites for {symbol}")

        if name == "ragdesk_topics":
            from ragdesk.topics import topic_map  # noqa: PLC0415

            with Store(self.db) as store:
                topics = topic_map(store)
            if not topics:
                return self._tool_text("nothing indexed yet")
            lines = [
                f"{topic['label'] or '(mixed)'} — {topic['documents']} documents "
                f"(e.g. {topic['paths'][0]})"
                for topic in topics
            ]
            return self._tool_text("\n".join(lines))

        if name == "ragdesk_save":
            url = str(arguments.get("url", "")).strip()
            if not url:
                return self._tool_error("url is required")
            from ragdesk.web import WebError, save_page  # noqa: PLC0415

            with Store(self.db) as store:
                store.ensure_embedder(self.embedder.name, self.embedder.dim)
                try:
                    stats = save_page(store, self.embedder, url)
                except WebError as exc:
                    return self._tool_error(str(exc))
            if stats.indexed:
                return self._tool_text(f"saved {url} ({stats.chunks} chunks)")
            if stats.unchanged:
                return self._tool_text(f"already saved and unchanged: {url}")
            return self._tool_error(f"fetched but no readable text: {url}")

        if name == "ragdesk_sources":
            with Store(self.db) as store:
                sources = store.sources()
            if not sources:
                return self._tool_text("nothing indexed yet")
            lines = [
                f"{entry['source']}: {entry['documents']} documents, {entry['chunks']} chunks"
                for entry in sources
            ]
            return self._tool_text("\n".join(lines))

        return self._tool_error(f"unknown tool: {name}")

    # --- helpers --------------------------------------------------------------

    @staticmethod
    def _ok(msg_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    @staticmethod
    def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}

    @staticmethod
    def _tool_text(text: str) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": text}]}

    @classmethod
    def _tool_error(cls, text: str) -> dict[str, Any]:
        return {**cls._tool_text(text), "isError": True}
