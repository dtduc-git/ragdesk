"""Local HTTP API for the desktop app. stdlib only; binds 127.0.0.1.

The Tauri shell spawns ``ragdesk serve`` and talks to these endpoints. No
auth: the server is loopback-only by design.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ragdesk import __version__
from ragdesk.answer import REFUSAL, answer
from ragdesk.confluence import ConfluenceError, sync_confluence
from ragdesk.embed import Embedder
from ragdesk.gdrive import GdriveError, sync_gdrive
from ragdesk.github import GitHubError, sync_github
from ragdesk.index import index_paths
from ragdesk.ollama import OllamaUnavailable
from ragdesk.rerank import get_reranker
from ragdesk.search import Hit, retrieve
from ragdesk.store import Store

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
}


def hit_to_dict(hit: Hit) -> dict[str, Any]:
    return {
        "path": hit.path,
        "source": hit.source,
        "ordinal": hit.ordinal,
        "text": hit.text,
        "score": hit.score,
        "cosine": hit.cosine,
        "lanes": hit.lanes,
    }


class AppState:
    """Shared server state. One lock serializes model + DB work: this is a
    single-user desktop app, deterministic behavior beats throughput."""

    def __init__(
        self,
        *,
        db: str,
        embedder: Embedder,
        rerank: str = "none",
        llm_model: str = "",
        llm_host: str = "",
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.rerank = rerank
        self.llm_model = llm_model
        self.llm_host = llm_host
        self.lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    state: AppState

    server_version = f"ragdesk/{__version__}"

    def log_message(self, *args: Any) -> None:  # keep the console quiet
        pass

    # --- plumbing -----------------------------------------------------------

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for key, value in CORS_HEADERS.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length) or b"{}")

    def do_OPTIONS(self) -> None:
        self._send(200, {})

    def do_GET(self) -> None:
        if self.path == "/api/status":
            with Store(self.state.db) as store:
                stats = store.stats()
                self._send(
                    200,
                    {
                        "version": __version__,
                        "db": self.state.db,
                        "embedder": {
                            "name": store.get_meta("embedder.name"),
                            "dim": store.get_meta("embedder.dim"),
                        },
                        "rerank": self.state.rerank,
                        "llm_model": self.state.llm_model,
                        "documents": stats["documents"],
                        "chunks": stats["chunks"],
                        "sources": store.sources(),
                    },
                )
            return
        self._send(404, {"error": f"not found: {self.path}"})

    def do_POST(self) -> None:
        try:
            body = self._read_json()
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON"})
            return
        try:
            if self.path == "/api/index":
                self._handle_index(body)
            elif self.path == "/api/sync/github":
                self._handle_sync_github(body)
            elif self.path == "/api/sync/confluence":
                self._handle_sync_confluence(body)
            elif self.path == "/api/sync/gdrive":
                self._handle_sync_gdrive(body)
            elif self.path == "/api/search":
                self._handle_search(body)
            elif self.path == "/api/ask":
                self._handle_ask(body)
            else:
                self._send(404, {"error": f"not found: {self.path}"})
        except OllamaUnavailable as exc:
            self._send(503, {"error": str(exc)})
        except (GitHubError, ConfluenceError, GdriveError) as exc:
            self._send(502, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - surface errors to the UI
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    # --- endpoints ------------------------------------------------------------

    def _reranker_for(self, override: str | None):
        return get_reranker(override or self.state.rerank)

    def _handle_index(self, body: dict[str, Any]) -> None:
        paths = [Path(p) for p in body.get("paths", [])]
        if not paths:
            self._send(400, {"error": "paths required"})
            return
        with self.state.lock, Store(self.state.db) as store:
            stats = index_paths(store, self.state.embedder, paths)
        self._send(
            200,
            {
                "scanned": stats.files_scanned,
                "indexed": stats.indexed,
                "unchanged": stats.unchanged,
                "skipped": stats.skipped,
                "chunks": stats.chunks,
            },
        )

    def _handle_sync_github(self, body: dict[str, Any]) -> None:
        repo = str(body.get("repo", "")).strip()
        if not repo:
            self._send(400, {"error": "repo required (owner/name)"})
            return
        with self.state.lock, Store(self.state.db) as store:
            stats = sync_github(
                store,
                self.state.embedder,
                repo=repo,
                ref=str(body.get("ref", "")),
                subdir=str(body.get("subdir", "")),
            )
        self._send(
            200,
            {
                "repo": repo,
                "scanned": stats.files_scanned,
                "indexed": stats.indexed,
                "unchanged": stats.unchanged,
                "skipped": stats.skipped,
                "chunks": stats.chunks,
            },
        )

    def _handle_sync_confluence(self, body: dict[str, Any]) -> None:
        base_url = str(body.get("base_url", "")).strip()
        space = str(body.get("space", "")).strip()
        if not base_url or not space:
            self._send(400, {"error": "base_url and space required"})
            return
        with self.state.lock, Store(self.state.db) as store:
            stats = sync_confluence(
                store,
                self.state.embedder,
                base_url=base_url,
                space=space,
                email=body.get("email"),
                token=body.get("token"),
            )
        self._send(
            200,
            {
                "space": space,
                "scanned": stats.files_scanned,
                "indexed": stats.indexed,
                "unchanged": stats.unchanged,
                "skipped": stats.skipped,
                "chunks": stats.chunks,
            },
        )

    def _handle_sync_gdrive(self, body: dict[str, Any]) -> None:
        with self.state.lock, Store(self.state.db) as store:
            stats = sync_gdrive(
                store,
                self.state.embedder,
                client_id=str(body.get("client_id", "")),
                client_secret=str(body.get("client_secret", "")),
                folder_id=str(body.get("folder_id", "")),
            )
        self._send(
            200,
            {
                "scanned": stats.files_scanned,
                "indexed": stats.indexed,
                "unchanged": stats.unchanged,
                "skipped": stats.skipped,
                "chunks": stats.chunks,
            },
        )

    def _handle_search(self, body: dict[str, Any]) -> None:
        query = str(body.get("query", "")).strip()
        if not query:
            self._send(400, {"error": "query required"})
            return
        top_k = int(body.get("top_k", 8))
        with self.state.lock, Store(self.state.db) as store:
            hits = retrieve(
                store,
                self.state.embedder,
                query,
                top_k=top_k,
                reranker=self._reranker_for(body.get("rerank")),
            )
        self._send(200, {"query": query, "hits": [hit_to_dict(hit) for hit in hits]})

    def _handle_ask(self, body: dict[str, Any]) -> None:
        query = str(body.get("query", "")).strip()
        if not query:
            self._send(400, {"error": "query required"})
            return
        top_k = int(body.get("top_k", 6))
        min_cosine = float(body.get("min_cosine", 0.0))
        with self.state.lock, Store(self.state.db) as store:
            hits = retrieve(
                store,
                self.state.embedder,
                query,
                top_k=top_k,
                reranker=self._reranker_for(body.get("rerank")),
            )
        text = answer(
            query,
            hits,
            model=self.state.llm_model,
            host=self.state.llm_host,
            min_cosine=min_cosine,
        )
        self._send(
            200,
            {
                "query": query,
                "answer": text,
                "refused": text == REFUSAL,
                "hits": [hit_to_dict(hit) for hit in hits],
            },
        )


def make_server(
    state: AppState, host: str = "127.0.0.1", port: int = 8765
) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"state": state})
    return ThreadingHTTPServer((host, port), handler)
