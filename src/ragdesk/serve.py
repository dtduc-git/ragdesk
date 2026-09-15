"""Local HTTP API for the desktop app. stdlib only; binds 127.0.0.1.

The Tauri shell spawns ``ragdesk serve`` and talks to these endpoints. No
auth: the server is loopback-only by design.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import threading
import time
import urllib.parse
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ragdesk import __version__, credentials, settings
from ragdesk.answer import REFUSAL, answer, answer_stream, wants_diagram
from ragdesk.confluence import (
    ConfluenceError,
    connect_oauth,
    oauth_ready,
    resolve_oauth_credentials,
    sync_confluence,
)
from ragdesk.confluence import whoami as confluence_whoami
from ragdesk.embed import Embedder
from ragdesk.evaluate import evaluate, ground_answer_detail, load_golden
from ragdesk.gdrive import TOKEN_FILE as GDRIVE_TOKEN_FILE
from ragdesk.gdrive import GdriveError, load_token_file, run_loopback_flow, sync_gdrive
from ragdesk.gdrive import resolve_client_credentials as gdrive_client_credentials
from ragdesk.gdrive import whoami as gdrive_whoami
from ragdesk.github import (
    GH_HOST,
    GitHubError,
    _token_from_gh,
    device_flow_poll_once,
    device_flow_start,
    resolve_client_id,
    sync_github,
    token_source,
)
from ragdesk.github import whoami as github_whoami
from ragdesk.gitlab import DEFAULT_BASE_URL as GITLAB_DEFAULT_BASE
from ragdesk.gitlab import GitLabError, sync_gitlab
from ragdesk.gitlab import whoami as gitlab_whoami
from ragdesk.index import index_paths
from ragdesk.llm import (
    LLMUnavailable,
    llm_status,
    mlx_available,
    mlx_model_cached,
    ollama_models,
    resolve_llm,
)
from ragdesk.msgraph import MsGraphError, sync_onedrive
from ragdesk.msgraph import device_flow_poll_once as ms_poll_once
from ragdesk.msgraph import device_flow_start as ms_device_start
from ragdesk.msgraph import resolve_client_id as resolve_ms_client_id
from ragdesk.msgraph import whoami as ms_whoami
from ragdesk.notion import NotionError, sync_notion
from ragdesk.notion import resolve_token as notion_resolve_token
from ragdesk.notion import whoami as notion_whoami
from ragdesk.ollama import DEFAULT_HOST, OllamaUnavailable, post_stream
from ragdesk.presets import PRESETS
from ragdesk.rerank import get_reranker
from ragdesk.search import Hit, parse_filters, retrieve
from ragdesk.store import Store
from ragdesk.web import WebError, crawl_site

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
}

MEMORY_MIN_COSINE = 0.35
MEMORY_LIMIT = 3
# Calibrated with the real embedder: same-intent paraphrases score 0.91-0.93,
# different intents 0.14-0.35, so 0.88 replays only true paraphrases.
SEMANTIC_CACHE_MIN_COSINE = 0.88
SMART_RETRIEVAL_PROMPT = """You prepare a search over the reader's own notes and code.
Given the conversation so far and the new question, reply with ONLY a JSON object:
{"standalone": "<the question rewritten to stand alone, same language>",
 "sub_queries": ["<at most 2 alternative phrasings or sub-questions>"],
 "hypothetical": "<a short factual paragraph that would answer it, as if from the notes>"}
Keep every field short. Use the same language as the question.

Conversation:
{history}
Question: {question}
JSON:"""
HYDE_PROMPT = (
    "Write a short factual paragraph that would answer the question below, as if it "
    "were an excerpt from the reader's own notes. Do not mention being hypothetical.\n\n"
    "Question: {question}\nAnswer:"
)
MEMORY_PROMPT = (
    "From the conversation below, extract durable facts about the user: "
    "preferences, projects, constraints, decisions. Reply with a JSON array of "
    "short strings and nothing else; use [] when there is nothing durable.\n\n"
    "{conversation}"
)

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
        "line": hit.line,
        "metadata": (hit.metadata or {}) if hasattr(hit, "metadata") else {},
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
        llm_spec: str = "",
        preset: str = "",
        ui_dir: str = "",
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.rerank = rerank
        self.llm_model = llm_model
        self.llm_host = llm_host
        self.llm_spec = llm_spec
        self.llm: Any = None
        self.llm_setup: dict[str, Any] = {
            "running": False,
            "kind": "",
            "model": "",
            "progress": 0.0,
            "detail": "",
            "error": "",
        }
        self.preset = preset
        self.ui_dir = Path(ui_dir) if ui_dir else None
        self.github_device: dict[str, Any] | None = None
        self.msgraph_device: dict[str, Any] | None = None
        self.last_used = time.monotonic()
        self.activity: dict[str, Any] = {
            "running": False,
            "kind": "",
            "detail": "",
            "done": 0,
            "total": 0,
            "started": 0.0,
            "owner": "",
        }
        self.lock = threading.Lock()

    def touch(self) -> None:
        self.last_used = time.monotonic()


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
                # The flag wins at launch; otherwise the saved setting, else light.
                preset = self.state.preset or settings.load()["preset"] or "light"
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
                        "llm": llm_status(
                            self.state.llm_spec or None,
                            preset=preset,
                            host=self.state.llm_host,
                            preference=str(settings.load().get("llm_preference") or ""),
                        ),
                        "llm_setup": {
                            **llm_setup_options(
                                preset,
                                self.state.llm_host,
                                self.state.llm_spec,
                            ),
                            "job": dict(self.state.llm_setup),
                        },
                        "preset": preset,
                        "documents": stats["documents"],
                        "chunks": stats["chunks"],
                        "sources": store.sources(),
                        "local_paths": store.local_paths(),
                        "auto_index": {
                            "hours": settings.load()["auto_index_hours"],
                            "last_run": settings.load()["auto_index_last"],
                        },
                        "hyde": bool(settings.load()["hyde"]),
                        "llm_setting": {
                            "preference": str(settings.load().get("llm_preference") or ""),
                            "openai_host": str(settings.load().get("openai_host") or ""),
                            "openai_model": str(settings.load().get("openai_model") or ""),
                            "openai_key_set": bool(
                                credentials.get("openai").get("api_key")
                            ),
                        },
                        "onboarded": bool(settings.load()["onboarded"]),
                        "system": system_info(),
                        "activity": dict(self.state.activity),
                        "memory": {
                            "models_loaded": self.state.llm is not None
                            or getattr(self.state.embedder, "loaded", False),
                            "idle_unload_minutes": settings.load()["idle_unload_minutes"],
                        },
                        "presets": [
                            {
                                "name": name,
                                "note": str(entry["note"]),
                                "rerank": str(entry["rerank"]),
                                "llm": str(entry["llm"]),
                            }
                            for name, entry in PRESETS.items()
                        ],
                    },
                )
            return
        if self.path == "/api/health":
            self._send(200, {"ok": True})
            return
        if self.path == "/api/feedback/golden":
            with Store(self.state.db) as store:
                rows = store.feedback_rows()
            jsonl = "\n".join(
                json.dumps(
                    {
                        "query": row["question"],
                        "relevant": row["relevant"],
                        "category": "feedback",
                    }
                )
                for row in rows
                if row["question"] and row["relevant"]
            )
            self._send(200, {"rows": len(rows), "jsonl": jsonl})
            return
        if self.path == "/api/memories":
            with Store(self.state.db) as store:
                self._send(200, {"memories": store.memories()})
            return
        if self.path == "/api/chats":
            with Store(self.state.db) as store:
                self._send(200, {"chats": store.chats()})
            return
        if self.path.startswith("/api/chats/"):
            try:
                chat_id = int(self.path.rsplit("/", 1)[-1])
            except ValueError:
                self._send(400, {"error": "chat id required"})
                return
            with Store(self.state.db) as store:
                detail = store.chat(chat_id)
            if detail is None:
                self._send(404, {"error": "chat not found"})
                return
            self._send(200, detail)
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
        self.state.touch()
        try:
            body = self._read_json()
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON"})
            return
        try:
            if self.path == "/api/index":
                self._handle_index(body)
            elif self.path == "/api/llm/setup":
                self._handle_llm_setup(body)
            elif self.path == "/api/settings":
                self._handle_settings(body)
            elif self.path == "/api/connections/github":
                self._handle_connect_github(body)
            elif self.path == "/api/connections/github/gh":
                self._handle_connect_github_gh()
            elif self.path == "/api/connections/github/client-id":
                self._handle_github_client_id(body)
            elif self.path == "/api/connections/github/device/start":
                self._handle_github_device_start()
            elif self.path == "/api/connections/github/device/poll":
                self._handle_github_device_poll()
            elif self.path == "/api/connections/confluence":
                self._handle_connect_confluence(body)
            elif self.path == "/api/connections/confluence/oauth":
                self._handle_connect_confluence_oauth(body)
            elif self.path == "/api/connections/gdrive":
                self._handle_connect_gdrive(body)
            elif self.path == "/api/connections/notion":
                self._handle_connect_notion(body)
            elif self.path == "/api/sync/notion":
                self._handle_sync_notion()
            elif self.path == "/api/connections/gitlab":
                self._handle_connect_gitlab(body)
            elif self.path == "/api/sync/gitlab":
                self._handle_sync_gitlab(body)
            elif self.path == "/api/connections/msgraph":
                self._handle_connect_msgraph(body)
            elif self.path == "/api/connections/msgraph/device/start":
                self._handle_msgraph_device_start()
            elif self.path == "/api/connections/msgraph/device/poll":
                self._handle_msgraph_device_poll()
            elif self.path == "/api/sync/msgraph":
                self._handle_sync_msgraph(body)
            elif self.path.endswith("/disconnect") and self.path.startswith(
                "/api/connections/"
            ):
                self._handle_disconnect(self.path.split("/")[3])
            elif self.path == "/api/feedback":
                self._handle_feedback(body)
            elif self.path == "/api/verify":
                self._handle_verify(body)
            elif self.path == "/api/memories":
                self._handle_memory_add(body)
            elif self.path == "/api/memories/delete":
                self._handle_memory_delete(body)
            elif self.path == "/api/memories/extract":
                self._handle_memory_extract(body)
            elif self.path == "/api/chats/delete":
                self._handle_chat_delete(body)
            elif self.path == "/api/sync/github":
                self._handle_sync_github(body)
            elif self.path == "/api/sync/confluence":
                self._handle_sync_confluence(body)
            elif self.path == "/api/sync/gdrive":
                self._handle_sync_gdrive(body)
            elif self.path == "/api/sync/web":
                self._handle_sync_web(body)
            elif self.path == "/api/search":
                self._handle_search(body)
            elif self.path == "/api/ask":
                self._handle_ask(body)
            elif self.path == "/api/ask/stream":
                self._handle_ask_stream(body)
            elif self.path == "/api/eval":
                self._handle_eval(body)
            else:
                self._send(404, {"error": f"not found: {self.path}"})
        except OllamaUnavailable as exc:
            self._send(503, {"error": str(exc)})
        except LLMUnavailable as exc:
            self._send(503, {"error": str(exc)})
        except (
            GitHubError,
            ConfluenceError,
            GdriveError,
            WebError,
            NotionError,
            GitLabError,
            MsGraphError,
        ) as exc:
            self._send(502, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - surface errors to the UI
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    # --- endpoints ------------------------------------------------------------

    def _reranker_for(self, override: str | None):
        return get_reranker(override or self.state.rerank)

    def _handle_llm_setup(self, body: dict[str, Any]) -> None:
        options = llm_setup_options(
            self.state.preset or "light", self.state.llm_host, self.state.llm_spec
        )
        if self.state.llm_setup.get("running"):
            self._send(409, {"error": "a model download is already running"})
            return
        kind = str(body.get("kind", ""))
        if kind == "ollama":
            if not options["ollama_reachable"]:
                self._send(400, {"error": "Ollama is not reachable — start it first"})
                return
            model = str(options["ollama_model"])
        elif kind == "mlx":
            if not options["mlx_available"]:
                self._send(
                    400,
                    {"error": 'MLX is not installed — run: uv tool install "ragdesk[mlx]"'},
                )
                return
            model = str(options["mlx_repo"])
        else:
            self._send(400, {"error": "kind must be 'ollama' or 'mlx'"})
            return
        self.state.llm_setup = {
            "running": True,
            "kind": kind,
            "model": model,
            "progress": 0.0,
            "detail": "starting",
            "error": "",
        }
        threading.Thread(
            target=run_llm_setup, args=(self.state, kind), daemon=True
        ).start()
        self._send(202, {"started": True, **self.state.llm_setup})

    def _handle_settings(self, body: dict[str, Any]) -> None:
        updates: dict[str, Any] = {}
        if "auto_index_hours" in body:
            try:
                hours = int(body["auto_index_hours"])
            except (TypeError, ValueError):
                self._send(400, {"error": "auto_index_hours must be a number"})
                return
            updates["auto_index_hours"] = max(0, min(hours, 168))
        if "hyde" in body:
            updates["hyde"] = bool(body["hyde"])
        if "onboarded" in body:
            updates["onboarded"] = bool(body["onboarded"])
        if "llm_preference" in body:
            preference = str(body["llm_preference"])
            if preference not in ("", "mlx", "ollama", "openai"):
                self._send(400, {"error": "llm_preference must be '', 'mlx', 'ollama' or 'openai'"})
                return
            updates["llm_preference"] = preference
        if "openai_host" in body:
            host = str(body["openai_host"]).strip()
            if host and not host.startswith(("http://", "https://")):
                self._send(400, {"error": "openai_host must start with http:// or https://"})
                return
            updates["openai_host"] = host.rstrip("/")
        if "openai_model" in body:
            updates["openai_model"] = str(body["openai_model"]).strip()
        if "openai_api_key" in body:
            credentials.set_provider("openai", {"api_key": str(body["openai_api_key"]).strip()})
        if "idle_unload_minutes" in body:
            try:
                minutes = int(body["idle_unload_minutes"])
            except (TypeError, ValueError):
                self._send(400, {"error": "idle_unload_minutes must be a number"})
                return
            updates["idle_unload_minutes"] = max(0, min(minutes, 1440))
        if "preset" in body:
            preset = str(body["preset"])
            if preset not in PRESETS:
                self._send(400, {"error": f"unknown preset: {preset!r}"})
                return
            updates["preset"] = preset
        if not updates:
            self._send(400, {"error": "nothing to update"})
            return
        settings.save(updates)
        if "preset" in updates:
            # Apply without a restart: reranking is query-time, the LLM re-resolves,
            # and every preset shares the same embedder so no re-index is needed.
            selected = PRESETS[updates["preset"]]
            self.state.preset = updates["preset"]
            self.state.rerank = selected["rerank"]
            self.state.llm_model = selected["llm"]
            self.state.llm = None
        self._send(200, updates)

    def _handle_index(self, body: dict[str, Any]) -> None:
        paths = [Path(p) for p in body.get("paths", [])]
        if not paths:
            self._send(400, {"error": "paths required"})
            return
        token = self._begin_activity("index")
        try:
            with self.state.lock, Store(self.state.db) as store:
                stats = index_paths(
                    store,
                    self.state.embedder,
                    paths,
                    progress=lambda detail, done, total: self._update_activity(
                        token, detail, done, total
                    ),
                )
        finally:
            self.state.activity["running"] = False
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
        token = self._begin_activity(f"github:{repo}")
        try:
            with self.state.lock, Store(self.state.db) as store:
                stats = sync_github(
                    store,
                    self.state.embedder,
                    repo=repo,
                    ref=str(body.get("ref", "")),
                    subdir=str(body.get("subdir", "")),
                    progress=lambda detail, done, total: self._update_activity(
                        token, detail, done, total
                    ),
                )
        finally:
            self.state.activity["running"] = False
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
        confluence_oauth = bool(
            confluence_entry.get("cloud_id") and confluence_entry.get("refresh_token")
        )
        if confluence_oauth:
            confluence_source = "oauth"
        elif confluence_store:
            confluence_source = "credentials"
        elif confluence_env:
            confluence_source = "env"
        else:
            confluence_source = None
        gdrive_payload = load_token_file()
        notion_entry = credentials.get("notion")
        gitlab_entry = credentials.get("gitlab")
        msgraph_entry = credentials.get("msgraph")
        return {
            "github": {
                "connected": gh_source is not None or bool(github_entry.get("token")),
                "source": gh_source[0] if gh_source else None,
                "login": github_entry.get("login", ""),
                "gh_available": _token_from_gh() is not None,
                "device_flow_ready": resolve_client_id() is not None,
            },
            "confluence": {
                "connected": confluence_oauth or confluence_store or confluence_env,
                "source": confluence_source,
                "base_url": confluence_entry.get("base_url", ""),
                "email": confluence_entry.get("email", ""),
                "display_name": confluence_entry.get("display_name", ""),
                "oauth_ready": oauth_ready(),
                "oauth_connected": confluence_oauth,
                "site_name": confluence_entry.get("site_name", ""),
                "site_url": confluence_entry.get("site_url", ""),
            },
            "gdrive": {
                "connected": bool(gdrive_payload.get("refresh_token")),
                "email": gdrive_payload.get("email", ""),
                "oauth_ready": bool(gdrive_client_credentials()[0]),
            },
            "notion": {
                "connected": notion_resolve_token() is not None,
                "name": notion_entry.get("name", ""),
            },
            "gitlab": {
                "connected": bool(gitlab_entry.get("token"))
                or bool(os.environ.get("GITLAB_TOKEN")),
                "name": gitlab_entry.get("name", ""),
                "base_url": gitlab_entry.get("base_url", GITLAB_DEFAULT_BASE),
            },
            "msgraph": {
                "connected": bool(msgraph_entry.get("refresh_token")),
                "account": msgraph_entry.get("account", ""),
                "client_id_set": resolve_ms_client_id() is not None,
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
        settings.save({"github_ignore_gh": False})
        credentials.set_provider("github", {"token": token, "login": login})
        self._send(200, {"connected": True, "source": "gh", "login": login})

    def _handle_github_client_id(self, body: dict[str, Any]) -> None:
        client_id = str(body.get("client_id", "")).strip()
        if not client_id:
            self._send(400, {"error": "client_id required"})
            return
        credentials.set_provider("github", {"client_id": client_id})
        self._send(200, {"saved": True, "verified": False})

    def _handle_github_device_start(self) -> None:
        client_id = resolve_client_id()
        if not client_id:
            self._send(
                400,
                {
                    "error": "no OAuth client ID: create a GitHub OAuth app with device flow "
                    "enabled and save its client ID, or set RAGDESK_GITHUB_CLIENT_ID"
                },
            )
            return
        data = device_flow_start(client_id)
        interval = int(data.get("interval", 5) or 5)
        self.state.github_device = {
            "device_code": str(data.get("device_code", "")),
            "client_id": client_id,
            "interval": interval,
        }
        self._send(
            200,
            {
                "user_code": data.get("user_code", ""),
                "verification_uri": data.get("verification_uri", f"{GH_HOST}/login/device"),
                "interval": interval,
                "expires_in": data.get("expires_in", 900),
            },
        )

    def _handle_github_device_poll(self) -> None:
        flow = self.state.github_device
        if not flow:
            self._send(400, {"error": "no device flow in progress"})
            return
        status, value = device_flow_poll_once(flow["client_id"], flow["device_code"])
        if status == "pending":
            self._send(200, {"connected": False, "pending": True})
            return
        if status == "slow_down":
            flow["interval"] = int(flow.get("interval", 5)) + 5
            self._send(200, {"connected": False, "pending": True, "interval": flow["interval"]})
            return
        token = value or ""
        login = github_whoami(token)
        credentials.set_provider("github", {"token": token, "login": login})
        self.state.github_device = None
        self._send(200, {"connected": True, "login": login})

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

    def _handle_connect_confluence_oauth(self, body: dict[str, Any]) -> None:
        entry = credentials.get("confluence")
        client_id = (
            str(body.get("client_id", "")).strip()
            or str(entry.get("client_id", ""))
            or os.environ.get("RAGDESK_ATLASSIAN_CLIENT_ID", "")
        )
        client_secret = (
            str(body.get("client_secret", "")).strip()
            or str(entry.get("client_secret", ""))
            or os.environ.get("RAGDESK_ATLASSIAN_CLIENT_SECRET", "")
        )
        if not (client_id and client_secret):
            self._send(
                400,
                {
                    "error": "an Atlassian OAuth app client ID and secret are required "
                    "(save them here once, or set RAGDESK_ATLASSIAN_CLIENT_ID/SECRET)"
                },
            )
            return
        timeout = float(body.get("timeout", 300) or 300)
        session = connect_oauth(client_id, client_secret, timeout=timeout)
        credentials.set_provider(
            "confluence",
            {
                "client_id": client_id,
                "client_secret": client_secret,
                "cloud_id": session["cloud_id"],
                "site_url": session["site_url"],
                "site_name": session["site_name"],
                "access_token": session["access_token"],
                "refresh_token": session["refresh_token"],
                "expires_at": time.time() + int(session["expires_in"]),
            },
        )
        self._send(
            200,
            {
                "connected": True,
                "display_name": session["site_name"] or session["site_url"],
                "base_url": session["site_url"],
            },
        )

    def _handle_connect_gdrive(self, body: dict[str, Any]) -> None:
        client_id, client_secret = gdrive_client_credentials(
            str(body.get("client_id", "")).strip(),
            str(body.get("client_secret", "")).strip(),
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
        if provider not in ("github", "confluence", "gdrive", "notion", "gitlab", "msgraph"):
            self._send(404, {"error": f"unknown provider: {provider}"})
            return
        credentials.clear(provider)
        if provider == "gdrive":
            GDRIVE_TOKEN_FILE.unlink(missing_ok=True)
        if provider == "github":
            # The gh CLI login on this machine is ambient: an explicit disconnect
            # must switch it off too, or a refresh would silently reconnect.
            found = token_source()
            if found and found[0] == "gh":
                settings.save({"github_ignore_gh": True})
                self._send(200, {"connected": False, "ignored_ambient": True})
                return
            keep = bool(found)
            self._send(
                200,
                {"connected": keep, "ignored_ambient": False, "source": found[0] if found else ""},
            )
            return
        self._send(200, {"connected": False})

    def _handle_sync_confluence(self, body: dict[str, Any]) -> None:
        space = str(body.get("space", "")).strip()
        if not space:
            self._send(400, {"error": "space required"})
            return
        oauth = resolve_oauth_credentials()
        with self.state.lock, Store(self.state.db) as store:
            if oauth:
                api_base, bearer = oauth
                entry = credentials.get("confluence")
                label_host = urllib.parse.urlparse(str(entry.get("site_url", ""))).netloc
                stats = sync_confluence(
                    store,
                    self.state.embedder,
                    space=space,
                    api_base=api_base,
                    bearer=bearer,
                    label_host=label_host,
                )
            else:
                base_url = str(body.get("base_url", "")).strip() or str(
                    credentials.get("confluence").get("base_url", "")
                )
                if not base_url:
                    self._send(
                        400,
                        {
                            "error": "no Confluence connection: connect with Atlassian "
                            "OAuth, or provide a site URL + API token"
                        },
                    )
                    return
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

    def _handle_connect_gitlab(self, body: dict[str, Any]) -> None:
        token = str(body.get("token", "")).strip()
        base_url = str(body.get("base_url", "")).strip() or GITLAB_DEFAULT_BASE
        if not token:
            self._send(400, {"error": "token required"})
            return
        name = gitlab_whoami(token, base_url=base_url)
        credentials.set_provider(
            "gitlab", {"token": token, "base_url": base_url, "name": name}
        )
        self._send(200, {"connected": True, "display_name": name})

    def _handle_sync_gitlab(self, body: dict[str, Any]) -> None:
        project = str(body.get("project", "")).strip()
        if not project:
            self._send(400, {"error": "project required (group/name)"})
            return
        base_url = str(body.get("base_url", "")).strip() or str(
            credentials.get("gitlab").get("base_url", GITLAB_DEFAULT_BASE)
        )
        with self.state.lock, Store(self.state.db) as store:
            stats = sync_gitlab(
                store,
                self.state.embedder,
                project=project,
                ref=str(body.get("ref", "")),
                subdir=str(body.get("subdir", "")),
                base_url=base_url,
            )
        self._send(
            200,
            {
                "project": project,
                "scanned": stats.files_scanned,
                "indexed": stats.indexed,
                "unchanged": stats.unchanged,
                "skipped": stats.skipped,
                "chunks": stats.chunks,
            },
        )

    def _handle_connect_msgraph(self, body: dict[str, Any]) -> None:
        client_id = str(body.get("client_id", "")).strip()
        if not client_id:
            self._send(400, {"error": "client_id required (Azure app registration)"})
            return
        credentials.set_provider("msgraph", {"client_id": client_id})
        self._send(200, {"saved": True, "client_id_set": True})

    def _handle_msgraph_device_start(self) -> None:
        client_id = resolve_ms_client_id()
        if client_id is None:
            self._send(
                400,
                {
                    "error": "no Microsoft client id: create an Azure app registration "
                    "(public client) and save its Application ID, or set "
                    "RAGDESK_MS_CLIENT_ID"
                },
            )
            return
        data = ms_device_start(client_id)
        interval = int(data.get("interval", 5) or 5)
        self.state.msgraph_device = {
            "device_code": str(data.get("device_code", "")),
            "client_id": client_id,
            "interval": interval,
        }
        self._send(
            200,
            {
                "user_code": data.get("user_code", ""),
                "verification_uri": data.get("verification_uri", "https://microsoft.com/devicelogin"),
                "interval": interval,
                "expires_in": data.get("expires_in", 900),
            },
        )

    def _handle_msgraph_device_poll(self) -> None:
        flow = self.state.msgraph_device
        if not flow:
            self._send(400, {"error": "no device flow in progress"})
            return
        status, payload = ms_poll_once(flow["client_id"], flow["device_code"])
        if status == "pending":
            self._send(200, {"connected": False, "pending": True})
            return
        if status == "slow_down":
            flow["interval"] = int(flow.get("interval", 5)) + 5
            self._send(200, {"connected": False, "pending": True, "interval": flow["interval"]})
            return
        token = str(payload.get("access_token", ""))
        account = ms_whoami(token)
        updates: dict[str, Any] = {"client_id": flow["client_id"], "account": account}
        if payload.get("refresh_token"):
            updates["refresh_token"] = str(payload["refresh_token"])
        credentials.set_provider("msgraph", updates)
        self.state.msgraph_device = None
        self._send(200, {"connected": True, "login": account})

    def _handle_sync_msgraph(self, body: dict[str, Any]) -> None:
        site = str(body.get("site", "")).strip()
        folder_id = str(body.get("folder_id", "")).strip()
        with self.state.lock, Store(self.state.db) as store:
            stats = sync_onedrive(
                store, self.state.embedder, site=site, folder_id=folder_id
            )
        self._send(
            200,
            {
                "site": site or "onedrive",
                "scanned": stats.files_scanned,
                "indexed": stats.indexed,
                "unchanged": stats.unchanged,
                "skipped": stats.skipped,
                "chunks": stats.chunks,
            },
        )

    def _handle_connect_notion(self, body: dict[str, Any]) -> None:
        token = str(body.get("token", "")).strip()
        if not token:
            self._send(400, {"error": "token required"})
            return
        name = notion_whoami(token)
        credentials.set_provider("notion", {"token": token, "name": name})
        self._send(200, {"connected": True, "display_name": name})

    def _handle_sync_notion(self) -> None:
        with self.state.lock, Store(self.state.db) as store:
            stats = sync_notion(store, self.state.embedder)
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

    def _handle_sync_web(self, body: dict[str, Any]) -> None:
        url = str(body.get("url", "")).strip()
        if not url:
            self._send(400, {"error": "url required"})
            return
        try:
            max_pages = int(body.get("max_pages") or 50)
            max_depth = int(body.get("max_depth") or 2)
        except (TypeError, ValueError):
            self._send(400, {"error": "max_pages and max_depth must be numbers"})
            return
        with self.state.lock, Store(self.state.db) as store:
            stats = crawl_site(
                store,
                self.state.embedder,
                start_url=url,
                max_pages=max_pages,
                max_depth=max_depth,
            )
        self._send(
            200,
            {
                "url": url,
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
        """Stream phased status lines, then the answer as NDJSON.

        Headers are committed before any work, so the UI can show live progress
        ("searching…", "loading the model…", "writing…") instead of a dead caret.
        """
        query, filters = parse_filters(str(body.get("query", "")))
        if not query:
            self._send(400, {"error": "query required (folder:/source: alone is not a query)"})
            return
        top_k = int(body.get("top_k", 6))
        min_cosine = float(body.get("min_cosine", 0.0))
        chat_id = int(body.get("chat_id") or 0)

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Connection", "close")
        for key, value in CORS_HEADERS.items():
            self.send_header(key, value)
        self.end_headers()
        self.close_connection = True

        def emit(payload: dict[str, Any]) -> None:
            self.wfile.write((json.dumps(payload) + "\n").encode())
            self.wfile.flush()

        try:
            emit({"status": "searching your sources…"})
            with self.state.lock, Store(self.state.db) as store:
                history = store.recent_turns(chat_id) if chat_id else []
            smart = self._smart_retrieval(query, history)
            if smart:
                emit({"status": "preparing the search (rewrite + draft)…"})
            search_query = str(smart.get("standalone") or query)
            with self.state.lock, Store(self.state.db) as store:
                query_vec = self.state.embedder.embed_query(search_query)
                hits = retrieve(
                    store,
                    self.state.embedder,
                    search_query,
                    top_k=top_k,
                    reranker=self._reranker_for(body.get("rerank")),
                    query_vec=query_vec,
                    hyde_text=str(smart.get("hypothetical") or ""),
                    extra_queries=[str(item) for item in (smart.get("sub_queries") or [])],
                    filters=filters,
                )
                fingerprint = self._fingerprint(store)
                cache_key = self._cache_key(fingerprint, query)
                cached = store.cache_get(cache_key)
                if cached is None:
                    cached = store.cache_nearest(
                        query_vec, fingerprint, SEMANTIC_CACHE_MIN_COSINE
                    )
                history = store.recent_turns(chat_id) if chat_id else []
                memory = None if cached is not None else self._relevant_memories(store, query)
            if cached is not None:
                cached_answer = str(cached["answer"])
                emit({"delta": cached_answer})
                chat_id = self._record_exchange(
                    chat_id, query, cached_answer, list(cached["citations"])
                )
                emit(
                    {
                        "done": True,
                        "hits": cached["citations"],
                        "cached": True,
                        "cached_question": str(cached["question"]),
                        "chat_id": chat_id,
                    }
                )
                return
            if self.state.llm is None:
                emit({"status": "loading the answer model…"})
            emit({"status": "thinking…"})
            pieces: list[str] = []
            for piece in answer_stream(
                query,
                hits,
                LazyLLM(self.state),
                min_cosine=min_cosine,
                history=history,
                memory=memory,
                diagram=wants_diagram(query),
            ):
                pieces.append(str(piece))
                emit({"delta": piece})
            text = "".join(pieces).strip() or REFUSAL
            citations = [hit_to_dict(hit) for hit in hits]
            chat_id = self._record_exchange(
                chat_id,
                query,
                text,
                citations,
                cache_key=None if text == REFUSAL else cache_key,
                cache_embedding=None if text == REFUSAL else query_vec,
                fingerprint=fingerprint,
            )
            emit(
                {
                    "done": True,
                    "hits": citations,
                    "cached": False,
                    "chat_id": chat_id,
                    "answer_id": self.last_answer_id,
                }
            )
        except Exception as exc:  # noqa: BLE001 - headers already sent
            try:
                emit({"error": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _handle_eval(self, body: dict[str, Any]) -> None:
        golden_path = str(body.get("golden", "")).strip()
        if not golden_path:
            self._send(400, {"error": "golden path required (JSONL of queries)"})
            return
        try:
            golden = load_golden(golden_path)
        except (OSError, ValueError) as exc:
            self._send(400, {"error": f"cannot read golden set: {exc}"})
            return
        top_k = int(body.get("top_k") or 10)
        with self.state.lock, Store(self.state.db) as store:
            metrics, per_query = evaluate(
                store,
                self.state.embedder,
                golden,
                top_k=top_k,
                reranker=self._reranker_for(body.get("rerank")),
            )
        self._send(200, {"golden": golden_path, "metrics": metrics, "queries": per_query})

    def _handle_search(self, body: dict[str, Any]) -> None:
        query, filters = parse_filters(str(body.get("query", "")))
        if not query:
            self._send(400, {"error": "query required (folder:/source: alone is not a query)"})
            return
        top_k = int(body.get("top_k", 8))
        hyde_text = self._hyde_text(query)
        with self.state.lock, Store(self.state.db) as store:
            hits = retrieve(
                store,
                self.state.embedder,
                query,
                top_k=top_k,
                reranker=self._reranker_for(body.get("rerank")),
                hyde_text=hyde_text,
                filters=filters,
            )
        self._send(200, {"query": query, "hits": [hit_to_dict(hit) for hit in hits]})

    def _begin_activity(self, kind: str) -> str:
        token = f"{kind}:{time.time()}"
        self.state.activity = {
            "running": True,
            "kind": kind,
            "detail": "starting",
            "done": 0,
            "total": 0,
            "started": time.time(),
            "owner": token,
        }
        return token

    def _update_activity(self, token: str, detail: str, done: int = 0, total: int = 0) -> None:
        """Progress from one run only: a stale callback must not clobber the
        activity a newer (or the auto-index) run owns."""
        if not self.state.activity.get("running"):
            return
        if token and self.state.activity.get("owner") != token:
            return
        self.state.activity.update({"detail": detail, "done": done, "total": total})

    def _fingerprint(self, store: Store) -> str:
        """Everything an answer depends on: corpus, embedder, model, spec."""
        return "|".join(
            [
                str(store.get_meta("embedder.name") or ""),
                str(self.state.llm_model or ""),
                str(self.state.llm_spec or ""),
                str(store.stats()["documents"]),
                str(store.stats()["chunks"]),
                store.corpus_revision(),
            ]
        )

    def _cache_key(self, fingerprint: str, query: str) -> str:
        """Reuse an answer only while the corpus, embedder and model are identical."""
        return hashlib.sha256(f"{fingerprint}|{query.strip().lower()}".encode()).hexdigest()

    def _smart_retrieval(self, question: str, history: list[tuple[str, str]]) -> dict[str, Any]:
        """One LLM call that rewrites, decomposes and drafts (HyDE) the query.

        Only runs when it can pay for itself: a follow-up (history present),
        a multi-part question, or HyDE switched on. Any failure falls back to
        plain retrieval — this is an enhancement, never a requirement.
        """
        values = settings.load()
        hyde_wanted = bool(values.get("hyde"))
        multi_part = any(
            marker in question.lower()
            for marker in (" and ", " vs ", " both ", " so sánh", " và ", " với ")
        )
        if not (hyde_wanted or (history and len(question.split()) <= 12) or multi_part):
            return {}
        if self.state.llm_setup.get("running"):
            return {}
        convo = "\n".join(
            f"{'User' if role == 'user' else 'ragdesk'}: {text[:300]}"
            for role, text in history[-4:]
        )
        try:
            raw = LazyLLM(self.state).generate(
                SMART_RETRIEVAL_PROMPT.format(history=convo or "(none)", question=question),
                {"num_predict": 300, "temperature": 0.2},
            )
        except Exception:  # noqa: BLE001 - retrieval never depends on this
            return {}
        return parse_smart_retrieval(str(raw))

    def _hyde_text(self, question: str) -> str:
        """A hypothetical answer used as an extra retrieval lane (optional)."""
        if not settings.load().get("hyde"):
            return ""
        try:
            if self.state.llm_setup.get("running"):
                return ""
            text = LazyLLM(self.state).generate(
                HYDE_PROMPT.format(question=question),
                {"num_predict": 120, "temperature": 0.3},
            )
        except Exception:  # noqa: BLE001 - HyDE is an enhancement, never a requirement
            return ""
        return str(text).strip()[:1200]

    def _record_exchange(
        self,
        chat_id: int,
        query: str,
        text: str,
        citations: list,
        cache_key: str | None = None,
        cache_embedding: list[float] | None = None,
        fingerprint: str = "",
    ) -> int:
        with self.state.lock, Store(self.state.db) as store:
            if not chat_id:
                title = query if len(query) <= 60 else f"{query[:57]}…"
                chat_id = store.create_chat(title)
            store.add_message(chat_id, "user", query)
            answer_id = store.add_message(chat_id, "assistant", text, citations)
            if cache_key:
                store.cache_put(
                    cache_key,
                    query,
                    text,
                    citations,
                    embedding=cache_embedding,
                    fingerprint=fingerprint,
                )
        self.last_answer_id = answer_id
        return chat_id

    def _handle_ask(self, body: dict[str, Any]) -> None:
        query, filters = parse_filters(str(body.get("query", "")))
        if not query:
            self._send(400, {"error": "query required (folder:/source: alone is not a query)"})
            return
        chat_id = int(body.get("chat_id") or 0)
        top_k = int(body.get("top_k", 6))
        min_cosine = float(body.get("min_cosine", 0.0))
        with self.state.lock, Store(self.state.db) as store:
            history = store.recent_turns(chat_id) if chat_id else []
        smart = self._smart_retrieval(query, history)
        search_query = str(smart.get("standalone") or query)
        with self.state.lock, Store(self.state.db) as store:
            query_vec = self.state.embedder.embed_query(search_query)
            hits = retrieve(
                store,
                self.state.embedder,
                search_query,
                top_k=top_k,
                reranker=self._reranker_for(body.get("rerank")),
                query_vec=query_vec,
                hyde_text=str(smart.get("hypothetical") or ""),
                extra_queries=[str(item) for item in (smart.get("sub_queries") or [])],
                filters=filters,
            )
            fingerprint = self._fingerprint(store)
            cache_key = self._cache_key(fingerprint, query)
            cached = store.cache_get(cache_key)
            if cached is None:
                cached = store.cache_nearest(
                    query_vec, fingerprint, SEMANTIC_CACHE_MIN_COSINE
                )
            history = store.recent_turns(chat_id) if chat_id else []
            memory = None if cached is not None else self._relevant_memories(store, query)
        if cached is not None:
            text = str(cached["answer"])
            citations = list(cached["citations"])
            from_cache = True
        else:
            citations = [hit_to_dict(hit) for hit in hits]
            text = answer(
                query,
                hits,
                LazyLLM(self.state),
                min_cosine=min_cosine,
                history=history,
                memory=memory,
                diagram=wants_diagram(query),
            )
            from_cache = False
        chat_id = self._record_exchange(
            chat_id,
            query,
            text,
            citations,
            cache_key=None if from_cache or text == REFUSAL else cache_key,
            cache_embedding=None if from_cache else query_vec,
            fingerprint=fingerprint,
        )
        self._send(
            200,
            {
                "query": query,
                "answer": text,
                "refused": text == REFUSAL,
                "hits": citations,
                "cached": from_cache,
                "cached_question": str(cached["question"]) if from_cache else "",
                "chat_id": chat_id,
            },
        )

    def _relevant_memories(self, store: Store, query: str) -> list[str]:
        """Top durable notes for this question; empty when nothing is close."""
        vectors = store.memory_vectors()
        if not vectors:
            return []
        query_vec = self.state.embedder.embed_query(query)
        q_norm = math.sqrt(sum(v * v for v in query_vec)) or 1.0
        scored: list[tuple[float, str]] = []
        for _memory_id, text, vector in vectors:
            dot = sum(a * b for a, b in zip(query_vec, vector, strict=False))
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            score = dot / (q_norm * norm)
            if score >= MEMORY_MIN_COSINE:
                scored.append((score, text))
        scored.sort(key=lambda item: -item[0])
        return [text for _score, text in scored[:MEMORY_LIMIT]]

    def _handle_feedback(self, body: dict[str, Any]) -> None:
        message_id = int(body.get("message_id") or 0)
        value = int(body.get("value") or 0)
        if not message_id or value not in (-1, 0, 1):
            self._send(400, {"error": "message_id and value (-1, 0, 1) required"})
            return
        with Store(self.state.db) as store:
            updated = store.set_feedback(message_id, value)
        self._send(200, {"updated": updated, "value": value})

    def _handle_verify(self, body: dict[str, Any]) -> None:
        """Ground every sentence of a stored answer against its own citations."""
        message_id = int(body.get("message_id") or 0)
        if not message_id:
            self._send(400, {"error": "message_id required"})
            return
        with Store(self.state.db) as store:
            row = store.conn.execute(
                "SELECT text, citations FROM messages WHERE id = ?", (message_id,)
            ).fetchone()
        if row is None:
            self._send(404, {"error": "message not found"})
            return
        citations = json.loads(row["citations"] or "[]")
        texts = [str(item.get("text", "")) for item in citations]
        verdict = ground_answer_detail(str(row["text"]), texts)
        self._send(200, verdict)

    def _handle_memory_add(self, body: dict[str, Any]) -> None:
        text = str(body.get("text", "")).strip()[:400]
        if not text:
            self._send(400, {"error": "text required"})
            return
        with self.state.lock, Store(self.state.db) as store:
            if store.has_memory(text):
                self._send(200, {"added": False, "duplicate": True})
                return
            memory_id = store.add_memory(text, self.state.embedder.embed_query(text))
        self._send(200, {"added": True, "id": memory_id})

    def _handle_memory_delete(self, body: dict[str, Any]) -> None:
        memory_id = int(body.get("id") or 0)
        if not memory_id:
            self._send(400, {"error": "id required"})
            return
        with Store(self.state.db) as store:
            store.delete_memory(memory_id)
        self._send(200, {"deleted": memory_id})

    def _handle_memory_extract(self, body: dict[str, Any]) -> None:
        """One LLM pass over the latest conversation, then store what is durable."""
        chat_id = int(body.get("chat_id") or 0)
        with self.state.lock, Store(self.state.db) as store:
            if not chat_id:
                chats = store.chats(limit=1)
                chat_id = int(chats[0]["id"]) if chats else 0
            detail = store.chat(chat_id) if chat_id else None
        if not detail or not detail["messages"]:
            self._send(400, {"error": "no conversation to read yet"})
            return
        conversation = "\n".join(
            f"{'User' if message['role'] == 'user' else 'ragdesk'}: {message['text'][:600]}"
            for message in detail["messages"][-12:]
        )
        raw = LazyLLM(self.state).generate(
            MEMORY_PROMPT.format(conversation=conversation),
            {"num_predict": 300, "temperature": 0.0},
        )
        added: list[str] = []
        with self.state.lock, Store(self.state.db) as store:
            for text in parse_memory_list(raw)[:5]:
                if len(text) < 4 or store.has_memory(text):
                    continue
                store.add_memory(text, self.state.embedder.embed_query(text))
                added.append(text)
        self._send(200, {"added": added, "raw": "" if added else str(raw)[:300]})

    def _handle_chat_delete(self, body: dict[str, Any]) -> None:
        chat_id = int(body.get("chat_id") or 0)
        if not chat_id:
            self._send(400, {"error": "chat_id required"})
            return
        with Store(self.state.db) as store:
            store.delete_chat(chat_id)
        self._send(200, {"deleted": chat_id})


def make_server(
    state: AppState, host: str = "127.0.0.1", port: int = 8765
) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"state": state})
    return ThreadingHTTPServer((host, port), handler)


def parse_smart_retrieval(raw: str) -> dict[str, Any]:
    """Pull the JSON object out of an LLM reply; tolerant of prose around it."""
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = {}
    standalone = str(data.get("standalone") or "").strip()
    if standalone and standalone.lower() != "none":
        out["standalone"] = standalone[:300]
    subs = data.get("sub_queries")
    if isinstance(subs, list):
        cleaned = [str(item).strip()[:200] for item in subs if str(item).strip()]
        if cleaned:
            out["sub_queries"] = cleaned[:2]
    hypothetical = str(data.get("hypothetical") or "").strip()
    if hypothetical and hypothetical.lower() != "none":
        out["hypothetical"] = hypothetical[:1200]
    return out


def system_info() -> dict[str, Any]:
    """Machine facts the wizard uses to suggest a preset (never leaves the box)."""
    try:
        ram_gb = int(
            (os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) / (1024**3)
        )
    except (ValueError, OSError, AttributeError):
        ram_gb = 0
    if ram_gb and ram_gb < 12:
        suggested = "light"
    elif ram_gb and ram_gb < 24:
        suggested = "balanced"
    else:
        suggested = "quality"
    return {"ram_gb": ram_gb, "platform": sys.platform, "suggested_preset": suggested}


def parse_memory_list(raw: str) -> list[str]:
    """Pull the JSON array out of an LLM reply; tolerant of prose around it."""
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [
        str(item).strip()[:200]
        for item in data
        if isinstance(item, str | int | float) and str(item).strip()
    ]


def ensure_llm(state: AppState) -> Any:
    """Resolve once per process; the ladder never downloads behind your back."""
    if state.llm is None:
        if state.llm_setup.get("running"):
            raise LLMUnavailable("the answer model is still downloading — try again shortly")
        state.llm = resolve_llm(
            state.llm_spec or None,
            preset=state.preset or "light",
            host=state.llm_host,
            preference=str(settings.load().get("llm_preference") or ""),
        )
    return state.llm


def llm_setup_options(preset: str, host: str, spec: str = "") -> dict[str, Any]:
    """What the onboarding wizard may offer on this machine."""
    selected = PRESETS.get(preset, PRESETS["light"])
    tag = str(selected["llm"])
    repo = str(selected.get("llm_mlx", ""))
    if spec.startswith("mlx:"):
        repo = spec.partition(":")[2] or repo
    models = ollama_models(host or DEFAULT_HOST)
    return {
        "ollama_model": tag,
        "ollama_reachable": bool(models),
        "ollama_has_model": tag in models or f"{tag}:latest" in models,
        "mlx_available": mlx_available(),
        "mlx_repo": repo,
        "mlx_cached": mlx_model_cached(repo),
    }


def _pull_ollama(job: dict[str, Any], host: str) -> None:
    for event in post_stream(
        host or DEFAULT_HOST,
        "/api/pull",
        {"model": job["model"], "stream": True},
        timeout=3600.0,
    ):
        if event.get("error"):
            raise LLMUnavailable(str(event["error"]))
        job["detail"] = str(event.get("status") or "pulling")
        total = int(event.get("total") or 0)
        completed = int(event.get("completed") or 0)
        if total and completed:
            job["progress"] = min(0.99, completed / total)


def _byte_bar(job: dict[str, Any], name: str, base: int, total: int, size: int) -> Any:
    """tqdm subclass mirroring download bytes into the job; None without tqdm."""
    try:
        from tqdm.auto import tqdm  # noqa: PLC0415 - ships with huggingface_hub
    except ImportError:
        return None

    class _Bar(tqdm):  # type: ignore[misc, valid-type]
        def update(self, n: int = 1) -> Any:
            result = super().update(n)
            if size:
                loaded = base + self.n
                job["progress"] = min(0.99, loaded / total)
                job["detail"] = (
                    f"downloading {name} — {loaded / 1e9:.2f}/{total / 1e9:.2f} GB"
                )
            return result

    return _Bar


def _download_mlx(job: dict[str, Any]) -> None:
    from huggingface_hub import HfApi, hf_hub_download  # noqa: PLC0415 - comes with mlx-lm

    repo = str(job["model"])
    job["detail"] = "listing model files"
    info = HfApi().model_info(repo, files_metadata=True)
    files = [
        (str(sibling.rfilename), int(sibling.size or 0))
        for sibling in info.siblings or []
        if not str(sibling.rfilename).startswith(".")
    ]
    total = sum(size for _, size in files) or 1
    done = 0
    for name, size in files:
        job["detail"] = f"downloading {name} — {done / 1e9:.2f}/{total / 1e9:.2f} GB"
        bar = _byte_bar(job, name, done, total, size)
        try:
            if bar is not None:
                hf_hub_download(repo, name, tqdm_class=bar)
            else:
                hf_hub_download(repo, name)
        except TypeError:  # hub too old for a custom progress bar
            hf_hub_download(repo, name)
        done += size
        job["progress"] = min(0.99, done / total)


def run_llm_setup(state: AppState, kind: str) -> None:
    """Download the answer model; progress is reported through ``state.llm_setup``."""
    job = state.llm_setup
    try:
        if kind == "ollama":
            _pull_ollama(job, state.llm_host)
        elif kind == "mlx":
            _download_mlx(job)
        else:
            raise LLMUnavailable(f"unknown setup kind: {kind!r}")
        job["detail"] = "ready"
        job["progress"] = 1.0
        state.llm = None  # re-resolve on the next ask
    except Exception as exc:  # noqa: BLE001 - whatever failed is the message
        job["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        job["running"] = False


class LazyLLM:
    """Resolves the backend on first real use — a gated refusal never needs one."""

    def __init__(self, state: AppState) -> None:
        self._state = state

    @property
    def kind(self) -> str:
        return str(getattr(ensure_llm(self._state), "kind", ""))

    @property
    def model(self) -> str:
        return str(getattr(ensure_llm(self._state), "model", ""))

    def generate(self, prompt: str, options: dict[str, Any]) -> str:
        return str(ensure_llm(self._state).generate(prompt, options))

    def generate_stream(self, prompt: str, options: dict[str, Any]) -> Any:
        return ensure_llm(self._state).generate_stream(prompt, options)


def release_idle_models(state: AppState, minutes: float) -> dict[str, Any] | None:
    """Drop loaded models after an idle stretch; the next use reloads lazily."""
    if minutes <= 0:
        return None
    if state.llm is None and not getattr(state.embedder, "loaded", False):
        return None
    idle_seconds = time.monotonic() - state.last_used
    if idle_seconds < minutes * 60:
        return None
    state.embedder.unload()
    state.llm = None
    return {"released": True, "idle_seconds": int(idle_seconds)}


def run_auto_index(state: AppState) -> dict[str, Any]:
    """Re-index the recorded local paths; called by the serve auto-index timer."""
    token = f"auto-index:{time.time()}"
    state.activity = {
        "running": True,
        "kind": "auto-index",
        "detail": "starting",
        "done": 0,
        "total": 0,
        "started": time.time(),
        "owner": token,
    }

    def progress(detail: str, done: int, total: int) -> None:
        if state.activity.get("owner") == token:
            state.activity.update({"detail": detail, "done": done, "total": total})

    with state.lock, Store(state.db) as store:
        roots = [
            Path(entry["path"]) for entry in store.local_paths() if Path(entry["path"]).exists()
        ]
        stats = (
            index_paths(store, state.embedder, roots, progress=progress) if roots else None
        )
    state.activity["running"] = False
    settings.save(
        {"auto_index_last": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")}
    )
    return {
        "roots": [str(root) for root in roots],
        "indexed": stats.indexed if stats else 0,
        "unchanged": stats.unchanged if stats else 0,
        "chunks": stats.chunks if stats else 0,
    }


def auto_index_due(values: dict) -> bool:
    """True when the stored interval has elapsed since the last auto run."""
    hours = float(values.get("auto_index_hours") or 0)
    if hours <= 0:
        return False
    last = str(values.get("auto_index_last") or "")
    if not last:
        return True
    try:
        ran_at = datetime.strptime(last, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    except ValueError:
        return True
    return (datetime.now(UTC) - ran_at).total_seconds() >= hours * 3600
