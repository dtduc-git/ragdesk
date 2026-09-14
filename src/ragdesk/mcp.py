"""Minimal MCP server over stdio (dependency-free).

Exposes the local index to MCP clients (Claude Code, Cursor, …) as tools:

- ``ragdesk_search`` — hybrid retrieval over the local index
- ``ragdesk_document`` — full text of one indexed document
- ``ragdesk_sources`` — indexed sources with document/chunk counts

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
        "name": "ragdesk_sources",
        "description": "List indexed sources with document and chunk counts.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


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
                hits = retrieve(
                    store, self.embedder, query, top_k=top_k, reranker=self.reranker
                )
            if not hits:
                return self._tool_text("no matches in the local index")
            blocks = [
                f"[{index}] {hit.path} (score {hit.score:.4f}, lanes {hit.lanes})\n"
                f"{hit.text[:600]}"
                for index, hit in enumerate(hits, start=1)
            ]
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
