"""Local HTTP API for the desktop app. stdlib only; binds 127.0.0.1.

The Tauri shell spawns ``ragdesk serve`` and talks to these endpoints. No
auth: the server is loopback-only by design.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ragdesk import __version__, credentials
from ragdesk.answer import REFUSAL, answer, answer_stream
from ragdesk.confluence import ConfluenceError, sync_confluence
from ragdesk.confluence import whoami as confluence_whoami
from ragdesk.embed import Embedder
from ragdesk.gdrive import TOKEN_FILE as GDRIVE_TOKEN_FILE
from ragdesk.gdrive import GdriveError, load_token_file, run_loopback_flow, sync_gdrive
from ragdesk.gdrive import whoami as gdrive_whoami
from ragdesk.github import GitHubError, _token_from_gh, sync_github, token_source
from ragdesk.github import whoami as github_whoami
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

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".json": "application/json",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
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
        preset: str = "",
        ui_dir: str = "",
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.rerank = rerank
        self.llm_model = llm_model
        self.llm_host = llm_host
        self.preset = preset
        self.ui_dir = Path(ui_dir) if ui_dir else None
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
                        "preset": self.state.preset,
                        "documents": stats["documents"],
                        "chunks": stats["chunks"],
                        "sources": store.sources(),
                    },
                )
            return
        if self.path == "/api/health":
            self._send(200, {"ok": True})
            return
        if self.path == "/api/connections":
            self._send(200, self._connections_payload())
            return
        if self._serve_static():
            return
        self._send(404, {"error": f"not found: {self.path}"})

    def _serve_static(self) -> bool:
        """Serve the built UI when ``--ui`` was given (browser dev + e2e)."""
        ui_dir = self.state.ui_dir
        if ui_dir is None:
            return False
        request_path = urllib.parse.urlparse(self.path).path
        relative = "index.html" if request_path in ("", "/") else request_path.lstrip("/")
        candidate = (ui_dir / relative).resolve()
        if not str(candidate).startswith(str(ui_dir.resolve())) or not candidate.is_file():
            # SPA fallback: unknown non-asset paths get the app shell
            if "." in Path(relative).name:
                return False
            candidate = ui_dir / "index.html"
            if not candidate.is_file():
                return False
        body = candidate.read_bytes()
        content_type = MIME_TYPES.get(candidate.suffix.lower(), "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in CORS_HEADERS.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)
        return True

    def do_POST(self) -> None:
        try:
            body = self._read_json()
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON"})
            return
        try:
            if self.path == "/api/index":
                self._handle_index(body)
            elif self.path == "/api/connections/github":
                self._handle_connect_github(body)
            elif self.path == "/api/connections/github/gh":
                self._handle_connect_github_gh()
            elif self.path == "/api/connections/confluence":
                self._handle_connect_confluence(body)
            elif self.path == "/api/connections/gdrive":
                self._handle_connect_gdrive(body)
            elif self.path.endswith("/disconnect") and self.path.startswith(
                "/api/connections/"
            ):
                self._handle_disconnect(self.path.split("/")[3])
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
            elif self.path == "/api/ask/stream":
                self._handle_ask_stream(body)
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

    def _connections_payload(self) -> dict[str, Any]:
        github_entry = credentials.get("github")
        gh_source = token_source()
        confluence_entry = credentials.get("confluence")
        confluence_store = bool(
            confluence_entry.get("base_url")
            and confluence_entry.get("email")
            and confluence_entry.get("token")
        )
        confluence_env = bool(
            os.environ.get("CONFLUENCE_EMAIL") and os.environ.get("CONFLUENCE_TOKEN")
        )
        gdrive_payload = load_token_file()
        return {
            "github": {
                "connected": gh_source is not None or bool(github_entry.get("token")),
                "source": gh_source[0] if gh_source else None,
                "login": github_entry.get("login", ""),
            },
            "confluence": {
                "connected": confluence_store or confluence_env,
                "source": (
                    "credentials" if confluence_store else ("env" if confluence_env else None)
                ),
                "base_url": confluence_entry.get("base_url", ""),
                "email": confluence_entry.get("email", ""),
                "display_name": confluence_entry.get("display_name", ""),
            },
            "gdrive": {
                "connected": bool(gdrive_payload.get("refresh_token")),
                "email": gdrive_payload.get("email", ""),
            },
        }

    def _handle_connect_github(self, body: dict[str, Any]) -> None:
        token = str(body.get("token", "")).strip()
        if token:
            login = github_whoami(token)
            credentials.set_provider("github", {"token": token, "login": login})
            self._send(200, {"connected": True, "source": "credentials", "login": login})
            return
        found = token_source()
        if not found:
            self._send(400, {"error": "no GitHub token: paste one or log in with gh"})
            return
        source, value = found
        login = github_whoami(value)
        credentials.set_provider("github", {"login": login})
        self._send(200, {"connected": True, "source": source, "login": login})

    def _handle_connect_github_gh(self) -> None:
        token = _token_from_gh()
        if not token:
            self._send(400, {"error": "gh CLI not found or not logged in (gh auth login)"})
            return
        login = github_whoami(token)
        credentials.set_provider("github", {"token": token, "login": login})
        self._send(200, {"connected": True, "source": "gh", "login": login})

    def _handle_connect_confluence(self, body: dict[str, Any]) -> None:
        base_url = str(body.get("base_url", "")).strip()
        email = str(body.get("email", "")).strip()
        token = str(body.get("token", "")).strip()
        if not (base_url and email and token):
            self._send(400, {"error": "base_url, email and token are all required"})
            return
        display_name = confluence_whoami(base_url, email, token)
        credentials.set_provider(
            "confluence",
            {
                "base_url": base_url,
                "email": email,
                "token": token,
                "display_name": display_name,
            },
        )
        self._send(200, {"connected": True, "display_name": display_name})

    def _handle_connect_gdrive(self, body: dict[str, Any]) -> None:
        stored = credentials.get("gdrive")
        client_id = str(body.get("client_id", "")).strip() or str(
            stored.get("client_id", "")
        )
        client_secret = str(body.get("client_secret", "")).strip() or str(
            stored.get("client_secret", "")
        )
        if not client_id:
            self._send(400, {"error": "an OAuth client ID is required"})
            return
        payload = run_loopback_flow(client_id, client_secret)
        email = ""
        access_token = str(payload.get("access_token", ""))
        if access_token:
            email = gdrive_whoami(access_token)
            payload["email"] = email
            from ragdesk.gdrive import save_token_file

            save_token_file(payload)
        saved = {"client_id": client_id}
        if client_secret:
            saved["client_secret"] = client_secret
        credentials.set_provider("gdrive", saved)
        self._send(200, {"connected": True, "email": email})

    def _handle_disconnect(self, provider: str) -> None:
        if provider not in ("github", "confluence", "gdrive"):
            self._send(404, {"error": f"unknown provider: {provider}"})
            return
        credentials.clear(provider)
        if provider == "gdrive":
            GDRIVE_TOKEN_FILE.unlink(missing_ok=True)
        self._send(200, {"connected": False})

    def _handle_sync_confluence(self, body: dict[str, Any]) -> None:
        base_url = str(body.get("base_url", "")).strip() or str(
            credentials.get("confluence").get("base_url", "")
        )
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

    def _handle_ask_stream(self, body: dict[str, Any]) -> None:
        """Stream the answer as newline-delimited JSON (headers are committed
        before the LLM call, so failures arrive as an ``error`` line)."""
        query = str(body.get("query", "")).strip()
        if not query:
            self._send(400, {"error": "query required"})
            return
        top_k = int(body.get("top_k", 6))
        min_cosine = float(body.get("min_cosine", 0.0))
        try:
            with self.state.lock, Store(self.state.db) as store:
                hits = retrieve(
                    store,
                    self.state.embedder,
                    query,
                    top_k=top_k,
                    reranker=self._reranker_for(body.get("rerank")),
                )
        except Exception as exc:  # noqa: BLE001 - headers not sent yet
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Connection", "close")
        for key, value in CORS_HEADERS.items():
            self.send_header(key, value)
        self.end_headers()
        self.close_connection = True
        try:
            for piece in answer_stream(
                query,
                hits,
                model=self.state.llm_model,
                host=self.state.llm_host,
                min_cosine=min_cosine,
            ):
                self.wfile.write((json.dumps({"delta": piece}) + "\n").encode())
                self.wfile.flush()
            done = {"done": True, "hits": [hit_to_dict(hit) for hit in hits]}
            self.wfile.write((json.dumps(done) + "\n").encode())
            self.wfile.flush()
        except Exception as exc:  # noqa: BLE001 - headers already sent
            try:
                self.wfile.write((json.dumps({"error": str(exc)}) + "\n").encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

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
