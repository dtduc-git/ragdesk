"""Local HTTP API for the desktop app. stdlib only; binds 127.0.0.1.

The Tauri shell spawns ``ragdesk serve`` and talks to these endpoints. No
auth: the server is loopback-only by design.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import urllib.parse
import zipfile
from collections.abc import Callable
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
from ragdesk.email_source import DEFAULT_FOLDER as EMAIL_DEFAULT_FOLDER
from ragdesk.email_source import DEFAULT_IMAP_PORT as EMAIL_DEFAULT_PORT
from ragdesk.email_source import DEFAULT_LIMIT as EMAIL_DEFAULT_LIMIT
from ragdesk.email_source import EmailError
from ragdesk.email_source import index_mbox as email_index_mbox
from ragdesk.email_source import sync_imap as email_sync_imap
from ragdesk.email_source import whoami as email_whoami
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
from ragdesk.index import IndexStats, index_paths, never_index_patterns
from ragdesk.llm import (
    LLMUnavailable,
    is_local_host,
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
from ragdesk.notes import NotesError, notes_available, sync_notes
from ragdesk.notion import NotionError, sync_notion
from ragdesk.notion import resolve_token as notion_resolve_token
from ragdesk.notion import whoami as notion_whoami
from ragdesk.ollama import DEFAULT_HOST, OllamaUnavailable, post_stream
from ragdesk.presets import PRESETS
from ragdesk.rerank import get_reranker, rerank_label
from ragdesk.s3 import DEFAULT_REGION as S3_DEFAULT_REGION
from ragdesk.s3 import S3Error, probe, resolve_credentials, sync_s3
from ragdesk.search import Hit, hit_to_dict, parse_filters, retrieve
from ragdesk.store import Store, matches_any
from ragdesk.symbols import find_symbol, parse_symbol_question, symbol_answer
from ragdesk.topics import topic_map
from ragdesk.web import WebError, crawl_site, save_page

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
}

MEMORY_MIN_COSINE = 0.35
MEMORY_LIMIT = 3
# Same calibration as the semantic cache: paraphrases score 0.91-0.93, so a
# correction only rides along when the question clearly repeats its own.
CORRECTION_MIN_COSINE = 0.88
CORRECTION_CHARS = 2000
# Calibrated with the real embedder: same-intent paraphrases score 0.91-0.93,
# different intents 0.14-0.35, so 0.88 replays only true paraphrases.
SEMANTIC_CACHE_MIN_COSINE = 0.88
# Bump when the answer prompt/format changes: cached answers from an older
# prompt must never be replayed (they would look like the change did nothing).
# v4: the corrections lookup keys on the standalone rewrite, not the raw query.
ANSWER_PROMPT_VERSION = "answer-v4-corrections"
SMART_RETRIEVAL_PROMPT = """You prepare a search over the reader's own notes and code.
Given the conversation so far and the new question, reply with ONLY a JSON object:
{{"standalone": "<the question rewritten to stand alone, same language>",
 "sub_queries": ["<at most 2 alternative phrasings or sub-questions>"],
 "hypothetical": "<a short factual paragraph that would answer it, as if from the notes>"}}
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
        self.reranker: Any = None
        self.reranker_spec: str = ""
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
                        "watch_seconds": int(settings.load().get("watch_seconds") or 0),
                        "sync_jobs": sync_jobs(),
                        "vaults": [str(entry) for entry in (settings.load().get("vaults") or [])],
                        "hyde": bool(settings.load()["hyde"]),
                        "answer_length": str(settings.load()["answer_length"]),
                        "embed_threads": int(settings.load().get("embed_threads") or 0),
                        "vector_backend": str(settings.load().get("vector_backend") or ""),
                        "llm_setting": {
                            "preference": str(settings.load().get("llm_preference") or ""),
                            "openai_host": str(settings.load().get("openai_host") or ""),
                            "openai_model": str(settings.load().get("openai_model") or ""),
                            "openai_key_set": bool(credentials.get("openai").get("api_key")),
                            # The UI warns when a remote endpoint would receive the
                            # question and the retrieved passages.
                            "openai_host_remote": not is_local_host(
                                str(settings.load().get("openai_host") or "")
                            ),
                        },
                        "onboarded": bool(settings.load()["onboarded"]),
                        "notes_available": notes_available(),
                        "s3": {
                            "configured": bool(credentials.get("s3").get("access_key")),
                            "bucket": str(credentials.get("s3").get("bucket", "")),
                            "region": str(credentials.get("s3").get("region") or S3_DEFAULT_REGION),
                            "endpoint": str(credentials.get("s3").get("endpoint", "")),
                        },
                        "system": system_info(),
                        "activity": dict(self.state.activity),
                        "memory": {
                            "models_loaded": self.state.llm is not None
                            or getattr(self.state.embedder, "loaded", False)
                            or getattr(self.state.reranker, "loaded", False),
                            "idle_unload_minutes": settings.load()["idle_unload_minutes"],
                        },
                        "presets": [
                            {
                                "name": name,
                                "note": str(entry["note"]),
                                "rerank": str(entry["rerank"]),
                                "rerank_label": rerank_label(str(entry["rerank"])),
                                "llm": str(entry["llm"]),
                            }
                            for name, entry in PRESETS.items()
                        ],
                    },
                )
            return
        if self.path == "/api/health":
            with Store(self.state.db) as store:
                self._send(200, self._health_payload(store))
            return
        if self.path == "/api/sync-jobs":
            self._send(200, {"jobs": sync_jobs()})
            return
        if self.path == "/api/bundles":
            self._send(
                200,
                {"dir": str(bundles_dir(self.state.db)), "bundles": list_bundles(self.state.db)},
            )
            return
        if self.path == "/api/backups":
            self._send(
                200,
                {
                    "dir": str(backups_dir(self.state.db)),
                    "backups": list_backups(self.state.db),
                },
            )
            return
        if self.path.startswith("/api/related"):
            target = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get(
                "path", [""]
            )[0]
            if not target:
                self._send(400, {"error": "path required"})
                return
            with Store(self.state.db) as store:
                self._send(200, {"related": store.related_documents(target)})
            return
        if self.path.startswith("/api/backlinks"):
            target = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get(
                "path", [""]
            )[0]
            if not target:
                self._send(400, {"error": "path required"})
                return
            with Store(self.state.db) as store:
                self._send(200, {"backlinks": store.backlinks(target)})
            return
        if self.path == "/api/mcp":
            self._send(200, mcp_setup_info(self.state))
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
        if self.path == "/api/corrections":
            with Store(self.state.db) as store:
                self._send(200, {"corrections": store.corrections()})
            return
        if self.path == "/api/bookmarks":
            with Store(self.state.db) as store:
                self._send(200, {"pages": store.web_pages()})
            return
        if self.path == "/api/duplicates":
            with Store(self.state.db) as store:
                clusters = store.duplicate_clusters()
            self._send(200, {"clusters": clusters})
            return
        if self.path.startswith("/api/refusals"):
            probe = "probe=1" in self.path
            with Store(self.state.db) as store:
                rows = store.refused_questions()
                if probe:
                    for row in rows[:10]:
                        hits = retrieve(
                            store,
                            self.state.embedder,
                            str(row["question"]),
                            top_k=1,
                        )
                        if hits:
                            row["best_hit"] = hits[0].path
                            row["best_cosine"] = round(hits[0].cosine, 3)
            self._send(
                200,
                {
                    "rows": rows,
                    "total": sum(int(row["count"]) for row in rows),
                    "resolved": sum(1 for row in rows if row["resolved"]),
                    "probed": probe,
                },
            )
            return
        if self.path == "/api/topics":
            with Store(self.state.db) as store:
                clusters = topic_map(store)
            self._send(200, {"clusters": clusters})
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
            elif self.path == "/api/connections/email":
                self._handle_connect_email(body)
            elif self.path == "/api/connections/msgraph":
                self._handle_connect_msgraph(body)
            elif self.path == "/api/connections/msgraph/device/start":
                self._handle_msgraph_device_start()
            elif self.path == "/api/connections/msgraph/device/poll":
                self._handle_msgraph_device_poll()
            elif self.path == "/api/sync/notes":
                self._handle_sync_notes(body)
            elif self.path == "/api/sync/msgraph":
                self._handle_sync_msgraph(body)
            elif self.path.endswith("/disconnect") and self.path.startswith("/api/connections/"):
                self._handle_disconnect(self.path.split("/")[3])
            elif self.path == "/api/mcp/install":
                self._handle_mcp_install()
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
            elif self.path == "/api/corrections":
                self._handle_correction_add(body)
            elif self.path == "/api/corrections/delete":
                self._handle_correction_delete(body)
            elif self.path == "/api/chats/delete":
                self._handle_chat_delete(body)
            elif self.path == "/api/sync/github":
                self._handle_sync_github(body)
            elif self.path == "/api/sync/confluence":
                self._handle_sync_confluence(body)
            elif self.path == "/api/sync/gdrive":
                self._handle_sync_gdrive(body)
            elif self.path == "/api/connections/s3":
                self._handle_connect_s3(body)
            elif self.path == "/api/sync/s3":
                self._handle_sync_s3(body)
            elif self.path == "/api/sync/web":
                self._handle_sync_web(body)
            elif self.path == "/api/obsidian":
                self._handle_add_vault(body)
            elif self.path == "/api/sync/email":
                self._handle_sync_email(body)
            elif self.path == "/api/sync/email-mbox":
                self._handle_sync_email_mbox(body)
            elif self.path == "/api/save":
                self._handle_save_page(body)
            elif self.path == "/api/never-index":
                self._handle_never_index(body)
            elif self.path == "/api/sync-jobs":
                self._handle_add_sync_job(body)
            elif self.path == "/api/sync-jobs/delete":
                self._handle_delete_sync_job(body)
            elif self.path == "/api/export":
                self._send(200, run_export(self.state.db))
            elif self.path == "/api/import":
                self._handle_import(body)
            elif self.path == "/api/backup":
                self._send(200, run_backup(self.state.db))
            elif self.path == "/api/restore":
                self._handle_restore(body)
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
            S3Error,
            GitHubError,
            ConfluenceError,
            GdriveError,
            WebError,
            NotionError,
            GitLabError,
            MsGraphError,
            NotesError,
            EmailError,
        ) as exc:
            self._send(502, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - surface errors to the UI
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    # --- endpoints ------------------------------------------------------------

    def _reranker_for(self, override: str | None):
        """One cached instance per spec: a fresh session per question costs
        seconds of model load, and the idle unload only frees what is cached."""
        spec = (override or self.state.rerank or "none").strip()
        if self.state.reranker_spec != spec:
            self.state.reranker = get_reranker(spec)
            self.state.reranker_spec = spec
        return self.state.reranker

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
        threading.Thread(target=run_llm_setup, args=(self.state, kind), daemon=True).start()
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
        if "watch_seconds" in body:
            try:
                seconds = int(body["watch_seconds"])
            except (TypeError, ValueError):
                self._send(400, {"error": "watch_seconds must be a number (0 = off)"})
                return
            updates["watch_seconds"] = max(0, min(seconds, 3600))
        if "hyde" in body:
            updates["hyde"] = bool(body["hyde"])
        if "embed_threads" in body:
            try:
                threads = int(body["embed_threads"])
            except (TypeError, ValueError):
                self._send(400, {"error": "embed_threads must be a number (0 = all cores)"})
                return
            updates["embed_threads"] = threads
        if "vector_backend" in body:
            backend = str(body["vector_backend"])
            if backend not in ("", "auto", "python", "numpy", "usearch"):
                self._send(
                    400,
                    {"error": "vector_backend must be '', 'auto', 'python', 'numpy' or 'usearch'"},
                )
                return
            updates["vector_backend"] = backend
            # the pool size is fixed when a session is built: drop the models so
            # the next question or index run picks the new value up
            for model in (self.state.embedder, self.state.reranker):
                unload = getattr(model, "unload", None)
                if callable(unload):
                    unload()
        if "answer_length" in body:
            length = str(body["answer_length"])
            if length not in ("short", "medium", "long"):
                self._send(400, {"error": "answer_length must be short, medium or long"})
                return
            updates["answer_length"] = length
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
                "skipped_samples": list(stats.skipped_samples),
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
        email_entry = credentials.get("email")
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
            "email": {
                "connected": bool(email_entry.get("host") and email_entry.get("user")),
                "host": email_entry.get("host", ""),
                "user": email_entry.get("user", ""),
                "folder": email_entry.get("folder", EMAIL_DEFAULT_FOLDER),
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
        if provider not in (
            "github",
            "confluence",
            "gdrive",
            "notion",
            "gitlab",
            "msgraph",
            "email",
            "s3",
        ):
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
        credentials.set_provider("gitlab", {"token": token, "base_url": base_url, "name": name})
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

    def _handle_connect_email(self, body: dict[str, Any]) -> None:
        host = str(body.get("host", "")).strip()
        user = str(body.get("user", "")).strip()
        password = str(body.get("password", ""))
        port = int(body.get("port") or EMAIL_DEFAULT_PORT)
        if not host or not user or not password:
            self._send(400, {"error": "host, user and password are required"})
            return
        with self.state.lock:
            name = email_whoami(host=host, user=user, password=password, port=port)
        credentials.set_provider(
            "email",
            {
                "host": host,
                "user": user,
                "password": password,
                "port": port,
                "folder": str(body.get("folder", "")).strip() or EMAIL_DEFAULT_FOLDER,
                "name": name,
            },
        )
        self._send(200, {"connected": True, "display_name": name})

    def _handle_sync_email(self, body: dict[str, Any]) -> None:
        entry = credentials.get("email")
        host = str(body.get("host", "")).strip() or str(entry.get("host", ""))
        user = str(body.get("user", "")).strip() or str(entry.get("user", ""))
        password = str(body.get("password", "")) or str(entry.get("password", ""))
        folder = (
            str(body.get("folder", "")).strip()
            or str(entry.get("folder", ""))
            or EMAIL_DEFAULT_FOLDER
        )
        if not host or not user or not password:
            self._send(
                400,
                {"error": "no email connection: connect the account in the app first"},
            )
            return
        limit = int(body.get("limit") or EMAIL_DEFAULT_LIMIT)
        with self.state.lock, Store(self.state.db) as store:
            stats = email_sync_imap(
                store,
                self.state.embedder,
                host=host,
                user=user,
                password=password,
                port=int(entry.get("port") or EMAIL_DEFAULT_PORT),
                folder=folder,
                limit=limit,
            )
        self._send(
            200,
            {
                "host": host,
                "folder": folder,
                "scanned": stats.files_scanned,
                "indexed": stats.indexed,
                "unchanged": stats.unchanged,
                "skipped": stats.skipped,
                "chunks": stats.chunks,
                "attachments": stats.attachments,
            },
        )

    def _handle_sync_email_mbox(self, body: dict[str, Any]) -> None:
        path = str(body.get("path", "")).strip()
        if not path:
            self._send(400, {"error": "path required (an .mbox file)"})
            return
        limit = int(body.get("limit") or 0)
        with self.state.lock, Store(self.state.db) as store:
            stats = email_index_mbox(store, self.state.embedder, path, limit=limit)
        self._send(
            200,
            {
                "path": path,
                "scanned": stats.files_scanned,
                "indexed": stats.indexed,
                "unchanged": stats.unchanged,
                "skipped": stats.skipped,
                "chunks": stats.chunks,
                "attachments": stats.attachments,
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
                "verification_uri": data.get(
                    "verification_uri", "https://microsoft.com/devicelogin"
                ),
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

    def _handle_sync_notes(self, body: dict[str, Any]) -> None:
        token = self._begin_activity("notes")
        try:
            with self.state.lock, Store(self.state.db) as store:
                stats = sync_notes(
                    store,
                    self.state.embedder,
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
                "chunks": stats.chunks,
            },
        )

    def _handle_sync_msgraph(self, body: dict[str, Any]) -> None:
        site = str(body.get("site", "")).strip()
        folder_id = str(body.get("folder_id", "")).strip()
        with self.state.lock, Store(self.state.db) as store:
            stats = sync_onedrive(store, self.state.embedder, site=site, folder_id=folder_id)
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

    def _handle_add_sync_job(self, body: dict[str, Any]) -> None:
        provider = str(body.get("provider", "")).strip()
        if provider not in SYNC_HANDLERS:
            supported = ", ".join(sorted(SYNC_HANDLERS))
            self._send(
                400,
                {"error": f"cannot auto-sync {provider!r} (supported: {supported})"},
            )
            return
        params = body.get("params")
        if not isinstance(params, dict) or not params:
            self._send(400, {"error": "params required (the same fields the sync used)"})
            return
        job = add_sync_job(provider, params)
        self._send(200, {"added": True, "job": job, "jobs": sync_jobs()})

    def _handle_delete_sync_job(self, body: dict[str, Any]) -> None:
        job_id = str(body.get("id", "")).strip()
        if not job_id:
            self._send(400, {"error": "id required"})
            return
        self._send(200, {"deleted": delete_sync_job(job_id), "jobs": sync_jobs()})

    def _handle_connect_s3(self, body: dict[str, Any]) -> None:
        """Validate the key against the bucket, then remember it (0600)."""
        bucket = str(body.get("bucket", "")).strip()
        access_key = str(body.get("access_key", "")).strip()
        secret_key = str(body.get("secret_key", "")).strip()
        if not bucket or not access_key or not secret_key:
            self._send(400, {"error": "bucket, access_key and secret_key are required"})
            return
        values = {
            "access_key": access_key,
            "secret_key": secret_key,
            "session_token": str(body.get("session_token", "")).strip(),
            "region": str(body.get("region", "")).strip() or S3_DEFAULT_REGION,
            "endpoint": str(body.get("endpoint", "")).strip().rstrip("/"),
            "bucket": bucket,
        }
        with self.state.lock:
            probe(resolve_credentials(values), bucket)
        credentials.set_provider("s3", values)
        self._send(200, {"connected": True, "bucket": bucket, "region": values["region"]})

    def _handle_sync_s3(self, body: dict[str, Any]) -> None:
        bucket = str(body.get("bucket", "")).strip() or str(credentials.get("s3").get("bucket", ""))
        if not bucket:
            self._send(400, {"error": "bucket required (e.g. my-data-bucket)"})
            return
        with self.state.lock, Store(self.state.db) as store:
            stats = sync_s3(
                store,
                self.state.embedder,
                bucket=bucket,
                prefix=str(body.get("prefix", "")),
                limit=int(body.get("limit") or 500),
            )
        self._send(
            200,
            {
                "bucket": bucket,
                "scanned": stats.files_scanned,
                "indexed": stats.indexed,
                "unchanged": stats.unchanged,
                "skipped": stats.skipped,
                "chunks": stats.chunks,
                "skipped_samples": list(stats.skipped_samples),
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

    def _handle_add_vault(self, body: dict[str, Any]) -> None:
        """Add an Obsidian vault: remember it, then index it vault-aware."""
        path = str(body.get("path", "")).strip()
        if not path:
            self._send(400, {"error": "path required (the folder holding .obsidian/)"})
            return
        vault = Path(path).expanduser()
        if not vault.is_dir():
            self._send(400, {"error": f"not a folder: {vault}"})
            return
        vaults = [str(entry) for entry in (settings.load().get("vaults") or [])]
        if str(vault) not in vaults:
            settings.save({"vaults": [*vaults, str(vault)]})
        with self.state.lock, Store(self.state.db) as store:
            stats = index_paths(store, self.state.embedder, [vault])
        self._send(
            200,
            {
                "path": str(vault),
                "is_vault": (vault / ".obsidian").is_dir(),
                "scanned": stats.files_scanned,
                "indexed": stats.indexed,
                "unchanged": stats.unchanged,
                "skipped": stats.skipped,
                "chunks": stats.chunks,
            },
        )

    def _handle_save_page(self, body: dict[str, Any]) -> None:
        """Save one page from the web (the Sources tab's Save button)."""
        url = str(body.get("url", "")).strip()
        if not url:
            self._send(400, {"error": "url required"})
            return
        with self.state.lock, Store(self.state.db) as store:
            stats = save_page(store, self.state.embedder, url)
        self._send(
            200,
            {
                "url": url,
                "indexed": stats.indexed,
                "unchanged": stats.unchanged,
                "skipped": stats.skipped,
                "chunks": stats.chunks,
            },
        )

    def _health_payload(self, store: Store) -> dict[str, Any]:
        """What the Indexed tab's health card needs, all from the store itself."""
        stats = store.stats()
        indexed_embedder = str(store.get_meta("embedder.name") or "")
        current_embedder = str(self.state.embedder.name)
        patterns = never_index_patterns()
        matches = (
            [
                str(row["path"])
                for row in store.conn.execute("SELECT path FROM documents")
                if matches_any(str(row["path"]), patterns)
            ]
            if patterns
            else []
        )
        try:
            db_bytes = Path(self.state.db).stat().st_size
        except OSError:
            db_bytes = 0
        return {
            "ok": True,
            "documents": stats["documents"],
            "chunks": stats["chunks"],
            "db_bytes": db_bytes,
            "embedder": {
                "index": indexed_embedder,
                "current": current_embedder,
                "matches": bool(indexed_embedder) and indexed_embedder == current_embedder,
            },
            "last_index": store.last_index_report(),
            "oldest": store.oldest_documents(),
            "never_index": {"patterns": patterns, "indexed_matches": matches},
        }

    def _handle_never_index(self, body: dict[str, Any]) -> None:
        raw = body.get("patterns")
        if not isinstance(raw, list):
            self._send(400, {"error": "patterns must be a list"})
            return
        patterns = [str(item).strip() for item in raw if str(item).strip()][:100]
        settings.save({"never_index": patterns})
        with self.state.lock, Store(self.state.db) as store:
            removed = store.delete_documents_matching(patterns) if patterns else []
        self._send(200, {"patterns": patterns, "removed": removed})

    def _handle_import(self, body: dict[str, Any]) -> None:
        path = str(body.get("path", "")).strip()
        if not path:
            self._send(400, {"error": "path required (a ragdesk bundle zip)"})
            return
        with self.state.lock:
            try:
                result = run_import(
                    self.state.db, path, embedder_name=str(self.state.embedder.name)
                )
            except (OSError, ValueError, sqlite3.Error, json.JSONDecodeError) as exc:
                self._send(400, {"error": str(exc)})
                return
        self._send(200, result)

    def _handle_restore(self, body: dict[str, Any]) -> None:
        path = str(body.get("path", "")).strip()
        if not path:
            self._send(400, {"error": "path required"})
            return
        with self.state.lock:
            try:
                result = restore_backup(self.state.db, path)
            except (OSError, ValueError, sqlite3.Error) as exc:
                self._send(400, {"error": str(exc)})
                return
        self._send(200, result)

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

    def _symbol_lookup(self, query: str) -> tuple[str, list[Hit], str]:
        """Deterministic "who calls X" answer: (text, hits, matched name).

        Navigation, not retrieval: nothing here touches the RRF lanes, and an
        empty text means "not a symbol question, or nothing found" — the caller
        falls back to the normal grounded answer.
        """
        candidate, certain = parse_symbol_question(query)
        if not candidate:
            return "", [], ""
        name = candidate
        with Store(self.state.db) as store:
            result = find_symbol(store, name)
        text, hits = symbol_answer(name, result)
        if not text:
            if not certain:
                return "", [], ""
            return (
                f"No definitions or call sites found for `{name}` in the indexed code files.",
                [],
                name,
            )
        return text, hits, name

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
            symbol_text, symbol_hits, symbol_name = self._symbol_lookup(query)
            if symbol_text:
                emit({"delta": symbol_text})
                citations = [hit_to_dict(hit) for hit in symbol_hits]
                chat_id = self._record_exchange(chat_id, query, symbol_text, citations)
                emit(
                    {
                        "done": True,
                        "hits": citations,
                        "cached": False,
                        "chat_id": chat_id,
                        "answer_id": self.last_answer_id,
                        "symbol": symbol_name,
                    }
                )
                return
            # Reads never take the writer lock: a long index must not block a
            # question (SQLite busy_timeout covers the rare write collision).
            with Store(self.state.db) as store:
                history = store.recent_turns(chat_id) if chat_id else []
            smart = self._smart_retrieval(query, history)
            if smart:
                emit({"status": "preparing the search (rewrite + draft)…"})
            search_query = str(smart.get("standalone") or query)
            with Store(self.state.db) as store:
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
                    cached = store.cache_nearest(query_vec, fingerprint, SEMANTIC_CACHE_MIN_COSINE)
                history = store.recent_turns(chat_id) if chat_id else []
                if cached is None:
                    # The rewrite variants are the right key for the lookups: a
                    # follow-up ("and how long do they last?") matches nothing alone.
                    memory = self._relevant_memories(store, search_query, query_vec)
                    corrections = self._corrections_for(
                        store,
                        [search_query, *(str(item) for item in (smart.get("sub_queries") or []))],
                    )
                else:
                    memory = None
                    corrections = []
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
                corrections=corrections,
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
                    "correction": str(corrections[0]["question"]) if corrections else "",
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
        with Store(self.state.db) as store:
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
                ANSWER_PROMPT_VERSION,
                str(store.get_meta("embedder.name") or ""),
                str(self.state.llm_model or ""),
                str(self.state.llm_spec or ""),
                str(store.stats()["documents"]),
                str(store.stats()["chunks"]),
                store.corpus_revision(),
                store.corrections_revision(),
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
        except Exception as exc:  # noqa: BLE001 - retrieval never depends on this
            # Loud on purpose: a broken prompt hid here once (JSON braces) and
            # returned plain retrieval for weeks without a trace.
            print(f"smart retrieval skipped: {exc}", file=sys.stderr)
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
        except Exception as exc:  # noqa: BLE001 - HyDE is an enhancement, never a requirement
            print(f"hyde skipped: {exc}", file=sys.stderr)
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
        # A short write must never queue behind a long index: SQLite's
        # busy_timeout serialises writers on its own.
        with Store(self.state.db) as store:
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
        symbol_text, symbol_hits, symbol_name = self._symbol_lookup(query)
        if symbol_text:
            citations = [hit_to_dict(hit) for hit in symbol_hits]
            chat_id = self._record_exchange(chat_id, query, symbol_text, citations)
            self._send(
                200,
                {
                    "query": query,
                    "answer": symbol_text,
                    "refused": False,
                    "hits": citations,
                    "cached": False,
                    "cached_question": "",
                    "correction": "",
                    "symbol": symbol_name,
                    "chat_id": chat_id,
                },
            )
            return
        with Store(self.state.db) as store:
            history = store.recent_turns(chat_id) if chat_id else []
        smart = self._smart_retrieval(query, history)
        search_query = str(smart.get("standalone") or query)
        with Store(self.state.db) as store:
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
                cached = store.cache_nearest(query_vec, fingerprint, SEMANTIC_CACHE_MIN_COSINE)
            history = store.recent_turns(chat_id) if chat_id else []
            if cached is None:
                # Same as the streaming path: the rewrite variants are the better
                # key when the question is a follow-up.
                memory = self._relevant_memories(store, search_query, query_vec)
                corrections = self._corrections_for(
                    store,
                    [search_query, *(str(item) for item in (smart.get("sub_queries") or []))],
                )
            else:
                memory = None
                corrections = []
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
                corrections=corrections,
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
                "correction": str(corrections[0]["question"]) if corrections else "",
                "chat_id": chat_id,
            },
        )

    def _relevant_memories(
        self, store: Store, query: str, query_vec: list[float] | None = None
    ) -> list[str]:
        """Top durable notes for this question; empty when nothing is close."""
        vectors = store.memory_vectors()
        if not vectors:
            return []
        query_vec = query_vec or self.state.embedder.embed_query(query)
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

    def _corrections_for(self, store: Store, texts: list[str]) -> list[dict[str, Any]]:
        """The closest correction for any phrasing of the question, or nothing.

        The rewrites matter: a follow-up's standalone can be wordier than the
        correction ("in the described flow") while a sub-query lands closer.
        """
        best: dict[str, Any] | None = None
        for text in texts:
            if not text:
                continue
            found = store.nearest_correction(
                self.state.embedder.embed_query(text), CORRECTION_MIN_COSINE
            )
            if found and (best is None or found["cosine"] > best["cosine"]):
                best = found
        return [best] if best else []

    def _handle_mcp_install(self) -> None:
        self._send(200, install_cli_shim())

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

    def _handle_correction_add(self, body: dict[str, Any]) -> None:
        """Save a user-edited answer so matching questions reuse it."""
        question = str(body.get("question", "")).strip()[:500]
        corrected = str(body.get("answer", "")).strip()[:CORRECTION_CHARS]
        if not question or not corrected:
            self._send(400, {"error": "question and answer required"})
            return
        with self.state.lock, Store(self.state.db) as store:
            correction_id = store.add_correction(
                question, corrected, self.state.embedder.embed_query(question)
            )
        self._send(200, {"added": True, "id": correction_id})

    def _handle_correction_delete(self, body: dict[str, Any]) -> None:
        correction_id = int(body.get("id") or 0)
        if not correction_id:
            self._send(400, {"error": "id required"})
            return
        with Store(self.state.db) as store:
            store.delete_correction(correction_id)
        self._send(200, {"deleted": correction_id})

    def _handle_chat_delete(self, body: dict[str, Any]) -> None:
        chat_id = int(body.get("chat_id") or 0)
        if not chat_id:
            self._send(400, {"error": "chat_id required"})
            return
        with Store(self.state.db) as store:
            store.delete_chat(chat_id)
        self._send(200, {"deleted": chat_id})


BACKUP_KEEP = 5


def backups_dir(db: str) -> Path:
    return Path(db).expanduser().resolve().parent / "backups"


def run_backup(db: str) -> dict[str, Any]:
    """Snapshot the live index (safe while it is in use) and prune old copies."""
    target_dir = backups_dir(db)
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = target_dir / f"index-{stamp}.db"
    suffix = 1
    while target.exists():
        # same-second calls must not overwrite each other — the safety snapshot
        # taken during a restore would clobber the backup being restored
        suffix += 1
        target = target_dir / f"index-{stamp}-{suffix}.db"
    with Store(db) as store:
        dest = sqlite3.connect(target)
        try:
            store.conn.backup(dest)
        finally:
            dest.close()
    kept = sorted(target_dir.glob("index-*.db"), reverse=True)
    for stale in kept[BACKUP_KEEP:]:
        stale.unlink(missing_ok=True)
    return {"path": str(target), "bytes": target.stat().st_size, "kept": len(kept[:BACKUP_KEEP])}


def list_backups(db: str) -> list[dict[str, Any]]:
    target_dir = backups_dir(db)
    if not target_dir.exists():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(target_dir.glob("index-*.db"), reverse=True):
        stat = path.stat()
        out.append(
            {
                "path": str(path),
                "name": path.name,
                "bytes": stat.st_size,
                "at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            }
        )
    return out


def restore_backup(db: str, source: str) -> dict[str, Any]:
    """Copy a backup's content over the live index (SQLite backup API, in place).

    A safety snapshot of the current index is taken first, so a wrong pick is
    one more restore away from being undone.
    """
    path = Path(source).expanduser()
    if not path.is_file():
        raise ValueError(f"backup not found: {source}")
    safety = run_backup(db)
    source_conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        with Store(db) as store:
            source_conn.backup(store.conn)
    finally:
        source_conn.close()
    return {"restored": str(path), "safety_backup": safety["path"]}


# --- portable bundles -----------------------------------------------------------

BUNDLE_MANIFEST = "manifest.json"
BUNDLE_DB = "index.db"


def bundles_dir(db: str) -> Path:
    return Path(db).expanduser().resolve().parent / "bundles"


def run_export(db: str) -> dict[str, Any]:
    """Zip a consistent snapshot of the index plus a manifest, for another machine."""
    target_dir = bundles_dir(db)
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = target_dir / f"ragdesk-{stamp}.zip"
    suffix = 1
    while target.exists():
        suffix += 1
        target = target_dir / f"ragdesk-{stamp}-{suffix}.zip"
    manifest: dict[str, Any] = {"ragdesk": __version__}
    with Store(db) as store:
        stats = store.stats()
        manifest.update(
            {
                "created_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
                "documents": stats["documents"],
                "chunks": stats["chunks"],
                "embedder": {
                    "name": store.get_meta("embedder.name"),
                    "dim": store.get_meta("embedder.dim"),
                },
            }
        )
        with tempfile.TemporaryDirectory(prefix="ragdesk-export-") as tmp:
            snapshot = Path(tmp) / BUNDLE_DB
            dest = sqlite3.connect(snapshot)
            try:
                store.conn.backup(dest)
            finally:
                dest.close()
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.write(snapshot, BUNDLE_DB)
                archive.writestr(BUNDLE_MANIFEST, json.dumps(manifest, indent=2))
    return {
        "path": str(target),
        "bytes": target.stat().st_size,
        **{key: manifest[key] for key in ("documents", "chunks", "created_at")},
    }


def list_bundles(db: str) -> list[dict[str, Any]]:
    target_dir = bundles_dir(db)
    if not target_dir.exists():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(target_dir.glob("ragdesk-*.zip"), reverse=True):
        stat = path.stat()
        out.append(
            {
                "path": str(path),
                "name": path.name,
                "bytes": stat.st_size,
                "at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            }
        )
    return out


def run_import(db: str, bundle: str, *, embedder_name: str = "") -> dict[str, Any]:
    """Replace the live index with a bundle's database (safety snapshot first).

    Fail-closed on a major-version gap or an embedder that does not match the
    one this install is configured with: a silent mismatch would poison every
    future query.
    """
    path = Path(bundle).expanduser()
    if not path.is_file():
        raise ValueError(f"bundle not found: {bundle}")
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"not a zip file: {bundle}") from exc
    with archive:
        names = set(archive.namelist())
        if BUNDLE_MANIFEST not in names or BUNDLE_DB not in names:
            raise ValueError("not a ragdesk bundle (no manifest/index)")
        manifest = json.loads(archive.read(BUNDLE_MANIFEST))
        build = str(manifest.get("ragdesk") or "")
        if build.split(".")[0] != __version__.split(".")[0]:
            raise ValueError(
                f"bundle was written by ragdesk {build}; this install is {__version__}"
            )
        bundled = str((manifest.get("embedder") or {}).get("name") or "")
        if embedder_name and bundled and embedder_name != bundled:
            raise ValueError(
                f"bundle was indexed with {bundled}; this install uses {embedder_name} — "
                "switch the embedder first"
            )
        safety = run_backup(db)
        with tempfile.TemporaryDirectory(prefix="ragdesk-import-") as tmp:
            archive.extract(BUNDLE_DB, tmp)
            source = sqlite3.connect(Path(tmp) / BUNDLE_DB)
            try:
                with Store(db) as store:
                    source.backup(store.conn)
            finally:
                source.close()
    return {
        "restored": str(path),
        "safety_backup": safety["path"],
        "documents": manifest.get("documents"),
        "chunks": manifest.get("chunks"),
    }


def make_server(state: AppState, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
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


def mcp_setup_info(state: AppState) -> dict[str, Any]:
    """Everything the Settings card needs to wire ragdesk into Claude/Codex."""
    on_path = shutil.which("ragdesk")
    return {
        "cli_on_path": bool(on_path),
        "cli_path": on_path or "",
        "db": state.db,
        "snippets": {
            "claude_code": "claude mcp add ragdesk -- ragdesk mcp",
            "claude_desktop": json.dumps(
                {"mcpServers": {"ragdesk": {"command": "ragdesk", "args": ["mcp"]}}},
                indent=2,
            ),
            "codex": '[mcp_servers.ragdesk]\ncommand = "ragdesk"\nargs = ["mcp"]',
        },
    }


def install_cli_shim() -> dict[str, Any]:
    """Make `ragdesk` reachable on PATH for MCP clients (macOS/Linux)."""
    if shutil.which("ragdesk"):
        return {"installed": False, "reason": "already on PATH", "path": shutil.which("ragdesk")}
    target = Path.home() / ".local" / "bin" / "ragdesk"
    source = Path(sys.argv[0]).resolve()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source)
    except OSError as exc:
        return {"installed": False, "reason": str(exc), "path": ""}
    return {"installed": True, "reason": "linked", "path": str(target)}


def system_info() -> dict[str, Any]:
    """Machine facts the wizard uses to suggest a preset (never leaves the box)."""
    try:
        ram_gb = int((os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) / (1024**3))
    except (ValueError, OSError, AttributeError):
        ram_gb = 0
    if ram_gb and ram_gb < 12:
        suggested = "light"
    elif ram_gb and ram_gb < 24:
        suggested = "balanced"
    else:
        suggested = "quality"
    screenshots = [
        str(candidate)
        for candidate in (
            Path.home() / "Desktop",
            Path.home() / "Pictures" / "Screenshots",
        )
        if candidate.is_dir()
    ]
    executable = Path(sys.executable)
    bundled = any(part.endswith(".app") or part == "Resources" for part in executable.parts)
    return {
        "ram_gb": ram_gb,
        "platform": sys.platform,
        "suggested_preset": suggested,
        "screenshot_dirs": screenshots,
        "runtime": "bundled" if bundled else "system",
        "runtime_path": str(executable),
    }


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
                job["detail"] = f"downloading {name} — {loaded / 1e9:.2f}/{total / 1e9:.2f} GB"
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
    if (
        state.llm is None
        and not getattr(state.embedder, "loaded", False)
        and not getattr(state.reranker, "loaded", False)
    ):
        return None
    idle_seconds = time.monotonic() - state.last_used
    if idle_seconds < minutes * 60:
        return None
    state.embedder.unload()
    unload = getattr(state.reranker, "unload", None)
    if callable(unload):
        unload()
    state.llm = None
    return {"released": True, "idle_seconds": int(idle_seconds)}


def sync_jobs() -> list[dict[str, Any]]:
    """Saved connector syncs, run by the same timer that refreshes local roots."""
    raw = settings.load().get("sync_jobs") or []
    return [
        entry
        for entry in raw
        if isinstance(entry, dict)
        and entry.get("provider")
        and isinstance(entry.get("params"), dict)
    ]


def sync_job_id(provider: str, params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(f"{provider}|{payload}".encode()).hexdigest()[:12]


def add_sync_job(provider: str, params: dict[str, Any]) -> dict[str, Any]:
    """Idempotent: the same provider+params is one job."""
    clean = {key: value for key, value in params.items() if value not in ("", None)}
    clean.pop("keep", None)
    job = {"id": sync_job_id(provider, clean), "provider": provider, "params": clean}
    jobs = [entry for entry in sync_jobs() if entry.get("id") != job["id"]]
    settings.save({"sync_jobs": [*jobs, job]})
    return job


def delete_sync_job(job_id: str) -> bool:
    jobs = sync_jobs()
    kept = [entry for entry in jobs if entry.get("id") != job_id]
    settings.save({"sync_jobs": kept})
    return len(kept) != len(jobs)


def _email_password() -> str:
    return str(credentials.get("email").get("password") or "")


SYNC_HANDLERS: dict[str, Callable[[Store, Embedder, dict], IndexStats]] = {
    "github": lambda store, embedder, p: sync_github(
        store, embedder, repo=p["repo"], ref=p.get("ref", ""), subdir=p.get("subdir", "")
    ),
    "gitlab": lambda store, embedder, p: sync_gitlab(
        store,
        embedder,
        project=p["project"],
        ref=p.get("ref", ""),
        subdir=p.get("subdir", ""),
        base_url=p.get("base_url", "") or GITLAB_DEFAULT_BASE,
    ),
    "confluence": lambda store, embedder, p: sync_confluence(
        store,
        embedder,
        space=p["space"],
        base_url=p.get("base_url", ""),
        email=p.get("email") or None,
        token=p.get("token") or None,
        limit=int(p.get("limit") or 100),
    ),
    "gdrive": lambda store, embedder, p: sync_gdrive(
        store,
        embedder,
        client_id=p.get("client_id", ""),
        client_secret=p.get("client_secret", ""),
        folder_id=p.get("folder_id", ""),
        interactive=False,
    ),
    "msgraph": lambda store, embedder, p: sync_onedrive(
        store,
        embedder,
        site=p.get("site", ""),
        folder_id=p.get("folder_id", ""),
        client_id=resolve_ms_client_id(),
    ),
    "notion": lambda store, embedder, p: sync_notion(store, embedder),
    "email": lambda store, embedder, p: email_sync_imap(
        store,
        embedder,
        host=p["host"],
        user=p["user"],
        password=p.get("password") or _email_password(),
        port=int(p.get("port") or EMAIL_DEFAULT_PORT),
        folder=p.get("folder") or EMAIL_DEFAULT_FOLDER,
        limit=int(p.get("limit") or EMAIL_DEFAULT_LIMIT),
    ),
    "web": lambda store, embedder, p: crawl_site(
        store,
        embedder,
        start_url=p["url"],
        max_pages=int(p.get("max_pages") or 50),
        max_depth=int(p.get("max_depth") or 2),
    ),
    "s3": lambda store, embedder, p: sync_s3(
        store,
        embedder,
        bucket=p["bucket"],
        prefix=p.get("prefix", ""),
        profile=p.get("profile", ""),
        limit=int(p.get("limit") or 500),
    ),
}


def run_sync_job(state: AppState, job: dict[str, Any]) -> dict[str, Any]:
    """Run one saved connector sync; a failure is reported, never fatal."""
    provider = str(job.get("provider") or "")
    handler = SYNC_HANDLERS.get(provider)
    if handler is None:
        return {"provider": provider, "error": f"cannot auto-sync {provider!r}"}
    params = dict(job.get("params") or {})
    try:
        with state.lock, Store(state.db) as store:
            stats = handler(store, state.embedder, params)
    except Exception as exc:  # noqa: BLE001 - one connector must not stop the pass
        return {"provider": provider, "error": f"{type(exc).__name__}: {exc}"[:200]}
    return {
        "provider": provider,
        "indexed": stats.indexed,
        "unchanged": stats.unchanged,
        "chunks": stats.chunks,
        **({"attachments": stats.attachments} if stats.attachments else {}),
    }


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
        stats = index_paths(store, state.embedder, roots, progress=progress) if roots else None
    connectors: list[dict[str, Any]] = []
    for job in sync_jobs():
        progress(f"syncing {job.get('provider')}", 0, 0)
        connectors.append(run_sync_job(state, job))
    state.activity["running"] = False
    settings.save({"auto_index_last": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")})
    return {
        "roots": [str(root) for root in roots],
        "indexed": stats.indexed if stats else 0,
        "unchanged": stats.unchanged if stats else 0,
        "chunks": stats.chunks if stats else 0,
        **({"connectors": connectors} if connectors else {}),
    }


def watch_pass(state: AppState) -> dict[str, Any] | None:
    """Cheap freshness pass: index_paths skips unchanged mtimes without reading."""
    with state.lock, Store(state.db) as store:
        roots = [
            Path(entry["path"]) for entry in store.local_paths() if Path(entry["path"]).exists()
        ]
        if not roots:
            return None
        stats = index_paths(store, state.embedder, roots)
    if not (stats.indexed or stats.chunks):
        return None
    return {
        "indexed": stats.indexed,
        "unchanged": stats.unchanged,
        "skipped": stats.skipped,
        "chunks": stats.chunks,
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
