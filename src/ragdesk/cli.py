"""ragdesk command line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from ragdesk import __version__
from ragdesk import settings as app_settings
from ragdesk.answer import REFUSAL, answer, answer_stream
from ragdesk.complete import bash_script, fish_script, man_page, zsh_script
from ragdesk.confluence import ConfluenceError, sync_confluence
from ragdesk.email_source import (
    DEFAULT_FOLDER,
    DEFAULT_IMAP_PORT,
    DEFAULT_LIMIT,
    EmailError,
    index_mbox,
    sync_imap,
)
from ragdesk.email_source import resolve_imap_credentials as email_credentials
from ragdesk.embed import get_embedder
from ragdesk.envfile import load_env_file
from ragdesk.evaluate import category_metrics, evaluate, format_report, ground_answer, load_golden
from ragdesk.gdrive import GdriveError, sync_gdrive
from ragdesk.github import GitHubError, sync_github
from ragdesk.gitlab import GitLabError, sync_gitlab
from ragdesk.index import index_paths
from ragdesk.llm import LLMUnavailable, resolve_llm
from ragdesk.mcp import McpServer
from ragdesk.msgraph import MsGraphError, device_flow_connect, resolve_client_id
from ragdesk.msgraph import sync_onedrive as sync_msgraph
from ragdesk.notion import NotionError, sync_notion
from ragdesk.ollama import DEFAULT_HOST, OllamaUnavailable
from ragdesk.presets import DEFAULT_PRESET, PRESETS
from ragdesk.presets import resolve as resolve_preset
from ragdesk.rerank import get_reranker
from ragdesk.search import parse_filters, retrieve
from ragdesk.serve import (
    HYDE_PROMPT,
    SMART_RETRIEVAL_PROMPT,
    AppState,
    auto_index_due,
    make_server,
    parse_smart_retrieval,
    release_idle_models,
    run_auto_index,
    watch_pass,
)
from ragdesk.store import Store
from ragdesk.web import WebError, crawl_site, save_page

# Same database the desktop app uses, so CLI/TUI/MCP all see one index.
DEFAULT_DB = str(Path.home() / ".ragdesk" / "index.db")


def emit_json(payload: dict) -> None:
    """Stable, script-friendly output for `--json` commands."""
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragdesk",
        description="Personal, local-first RAG over your own sources.",
    )
    parser.add_argument("--version", action="version", version=f"ragdesk {__version__}")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"index database (default: {DEFAULT_DB})")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default=None,
        help="hardware preset (default: saved setting, else light)",
    )
    parser.add_argument(
        "--embedder",
        default=None,
        help="embedder spec: onnx[:repo], ollama[:model] or hash[:dim] (default: from preset)",
    )
    parser.add_argument(
        "--rerank",
        default=None,
        help="reranker spec: none | lexical | fastembed[:model] (default: from preset)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="index local files/directories")
    p_index.add_argument("paths", nargs="+", type=Path)
    p_index.add_argument("--json", action="store_true", help="machine-readable output")

    p_search = sub.add_parser("search", help="hybrid retrieval (no LLM)")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=8)
    p_search.add_argument("--json", action="store_true", help="machine-readable output")

    p_ask = sub.add_parser("ask", help="cited answer via a local LLM")
    p_ask.add_argument("query")
    p_ask.add_argument("--model", default=None, help="Ollama model tag (default: from preset)")
    p_ask.add_argument(
        "--llm",
        default=None,
        help="backend spec: auto (reuse Ollama with the model, else MLX), "
        "ollama[:model] or mlx[:hf-repo]",
    )
    p_ask.add_argument(
        "--min-cosine",
        type=float,
        default=0.0,
        help="grounding gate: refuse below this dense cosine (calibrate per embedder)",
    )
    p_ask.add_argument("--top-k", type=int, default=6)
    p_ask.add_argument("--stream", action="store_true", help="print tokens as they arrive")
    p_ask.add_argument("--llm-host", default=DEFAULT_HOST, help="Ollama host override")
    p_ask.add_argument("--json", action="store_true", help="machine-readable output")

    p_chat = sub.add_parser("chat", help="terminal chat over the same index (TUI)")
    p_chat.add_argument("--chat-id", type=int, default=0, help="resume a conversation")
    p_chat.add_argument("--top-k", type=int, default=6)

    p_eval = sub.add_parser("eval", help="retrieval eval on a golden set")
    p_eval.add_argument("--golden", required=True, type=Path)
    p_eval.add_argument("--top-k", type=int, default=10)
    p_eval.add_argument(
        "--min-recall",
        type=float,
        default=None,
        help="exit non-zero if recall@5 is below this (CI gate)",
    )
    p_eval.add_argument("--json", action="store_true", help="machine-readable output")
    p_eval.add_argument(
        "--min-recall-category",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="per-category CI gate, e.g. --min-recall-category tax=0.8 (repeatable)",
    )
    p_eval.add_argument(
        "--answers",
        action="store_true",
        help="also answer each query with the local LLM and score grounding",
    )
    p_eval.add_argument(
        "--rewrite",
        action="store_true",
        help="score follow-up rows (a history field) both raw and rewritten by the local model",
    )

    p_stats = sub.add_parser("stats", help="index statistics")
    p_stats.add_argument("--json", action="store_true", help="machine-readable output")

    p_gh = sub.add_parser("github", help="index a GitHub repository (read-only tarball sync)")
    p_gh.add_argument("repo", help="owner/name")
    p_gh.add_argument("--ref", default="", help="branch/tag/sha (default: repo default branch)")
    p_gh.add_argument("--subdir", default="", help="index only a subtree")
    p_gh.add_argument(
        "--token", default=None, help="GitHub token (default: GITHUB_TOKEN env or gh auth token)"
    )
    p_gh.add_argument("--json", action="store_true", help="machine-readable output")

    p_cf = sub.add_parser("confluence", help="index a Confluence space (read-only)")
    p_cf.add_argument("space", help="space key, e.g. DOCS")
    p_cf.add_argument("--base-url", required=True, help="e.g. https://team.atlassian.net")
    p_cf.add_argument("--email", default=None, help="default: CONFLUENCE_EMAIL env")
    p_cf.add_argument("--token", default=None, help="default: CONFLUENCE_TOKEN env")
    p_cf.add_argument(
        "--api-path", default="/wiki/rest/api/content/search", help="REST path (Server/DC differs)"
    )
    p_cf.add_argument("--json", action="store_true", help="machine-readable output")

    p_gd = sub.add_parser("gdrive", help="index Google Drive (read-only, BYO OAuth client)")
    p_gd.add_argument("--folder-id", default="", help="index one folder (default: all files)")
    p_gd.add_argument("--client-id", default="", help="default: GDRIVE_CLIENT_ID env")
    p_gd.add_argument("--client-secret", default="", help="default: GDRIVE_CLIENT_SECRET env")
    p_gd.add_argument(
        "--no-browser", action="store_true", help="do not run the interactive OAuth flow"
    )
    p_gd.add_argument("--json", action="store_true", help="machine-readable output")

    p_gl = sub.add_parser("gitlab", help="index a GitLab repository (read-only archive sync)")
    p_gl.add_argument("project", help="group/name")
    p_gl.add_argument("--ref", default="", help="branch/tag/sha (default: repo default branch)")
    p_gl.add_argument("--subdir", default="", help="index only a subtree")
    p_gl.add_argument(
        "--token", default=None, help="default: GITLAB_TOKEN env or saved connection"
    )
    p_gl.add_argument("--base-url", default="https://gitlab.com", help="self-hosted GitLab URL")
    p_gl.add_argument("--json", action="store_true", help="machine-readable output")

    p_ms = sub.add_parser("msgraph", help="index OneDrive/SharePoint files (read-only)")
    p_ms.add_argument("--folder-id", default="", help="OneDrive folder id (default: all files)")
    p_ms.add_argument("--site", default="", help="SharePoint site as hostname:/sites/name")
    p_ms.add_argument("--client-id", default=None, help="Azure application (client) id")
    p_ms.add_argument("--json", action="store_true", help="machine-readable output")

    p_notion = sub.add_parser(
        "notion", help="index Notion pages shared with an integration (read-only)"
    )
    p_notion.add_argument(
        "--token", default=None, help="default: NOTION_TOKEN env or saved connection"
    )
    p_notion.add_argument("--json", action="store_true", help="machine-readable output")

    p_web = sub.add_parser("web", help="crawl a docs site and index it (read-only)")
    p_web.add_argument("url", help="start URL, e.g. https://docs.example.com/")
    p_web.add_argument("--max-pages", type=int, default=50)
    p_web.add_argument("--depth", type=int, default=2)
    p_web.add_argument("--json", action="store_true", help="machine-readable output")

    p_save = sub.add_parser("save", help="save one web page (bookmark) into the index")
    p_save.add_argument("url", help="page URL, e.g. https://example.com/article")
    p_save.add_argument("--json", action="store_true", help="machine-readable output")

    p_email = sub.add_parser("email", help="index email (read-only mbox or IMAP)")
    email_mode = p_email.add_mutually_exclusive_group(required=True)
    email_mode.add_argument("--mbox", type=Path, help="mbox file to index")
    email_mode.add_argument("--imap", metavar="HOST", help="IMAP host, e.g. imap.gmail.com")
    p_email.add_argument("--user", default="", help="IMAP username")
    p_email.add_argument(
        "--password",
        default="",
        help="IMAP password (prompted or read from the saved connection when omitted)",
    )
    p_email.add_argument("--port", type=int, default=DEFAULT_IMAP_PORT)
    p_email.add_argument("--folder", default=DEFAULT_FOLDER)
    p_email.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="newest N messages")
    p_email.add_argument("--json", action="store_true", help="machine-readable output")

    p_completions = sub.add_parser(
        "completions", help="print a shell completion script (bash/zsh/fish)"
    )
    p_completions.add_argument("shell", choices=["bash", "zsh", "fish"])

    sub.add_parser("man", help="print the man page (roff) for ragdesk")

    sub.add_parser("mcp", help="run the MCP server over stdio (for Claude Code / Cursor)")

    p_serve = sub.add_parser("serve", help="local HTTP API for the desktop app")
    p_serve.add_argument("--port", type=int, default=8765)
    p_serve.add_argument("--llm-model", default=None, help="LLM model (default: from preset)")
    p_serve.add_argument(
        "--llm",
        default="",
        help="backend spec: auto (default), ollama[:model] or mlx[:hf-repo]",
    )
    p_serve.add_argument("--llm-host", default=DEFAULT_HOST)
    p_serve.add_argument("--ui", default="", help="serve a built UI directory (browser mode)")
    p_serve.add_argument(
        "--watch-parent",
        action="store_true",
        help="exit when the parent process dies (the desktop app passes this)",
    )
    return parser


def emit_stats(stats, as_json: bool, **extra) -> None:
    """One shape for every indexing command, human or machine."""
    payload = {
        "scanned": stats.files_scanned,
        "indexed": stats.indexed,
        "unchanged": stats.unchanged,
        "skipped": stats.skipped,
        "chunks": stats.chunks,
        **extra,
    }
    if as_json:
        emit_json(payload)
        return
    print(
        f"scanned={stats.files_scanned} indexed={stats.indexed} "
        f"unchanged={stats.unchanged} skipped={stats.skipped} chunks={stats.chunks}"
    )


def emit_hits(hits, as_json: bool) -> None:
    if not as_json:
        _print_hits(hits)
        return
    emit_json(
        {
            "hits": [
                {
                    "path": hit.path,
                    "line": hit.line,
                    "ordinal": hit.ordinal,
                    "score": round(hit.score, 4),
                    "cosine": round(hit.cosine, 3),
                    "lanes": hit.lanes,
                    "text": hit.text,
                }
                for hit in hits
            ]
        }
    )


def _print_hits(hits) -> None:
    for rank, hit in enumerate(hits, start=1):
        snippet = " ".join(hit.text.split())[:160]
        print(f"{rank}. score={hit.score:.4f} cos={hit.cosine:.3f} [{hit.lanes}] {hit.path}")
        print(f"   {snippet}")


def _imap_password(user: str) -> str:
    """Saved IMAP password, else an interactive prompt (never a CLI argument)."""
    stored = email_credentials().get("password")
    if stored:
        return str(stored)
    if sys.stdin.isatty():
        import getpass  # noqa: PLC0415 - only needed interactively

        return getpass.getpass(f"IMAP password for {user or 'the account'}: ")
    raise EmailError(
        "no IMAP password: pass --password, connect the account in the app, "
        "or run interactively to be prompted"
    )


def _standalone_rewrite(llm, question: str, history: list[tuple[str, str]]) -> str:
    """The same rewrite ``serve`` does before searching; ``eval --rewrite`` scores it."""
    convo = "\n".join(
        f"{'User' if role == 'user' else 'ragdesk'}: {text[:300]}"
        for role, text in history[-4:]
    )
    raw = llm.generate(
        SMART_RETRIEVAL_PROMPT.format(history=convo or "(none)", question=question),
        {"num_predict": 300, "temperature": 0.2},
    )
    return str(parse_smart_retrieval(str(raw)).get("standalone") or "").strip()


def main(argv: list[str] | None = None) -> int:
    load_env_file()  # local .env (gitignored) for development
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "completions":
        emitters = {"bash": bash_script, "zsh": zsh_script, "fish": fish_script}
        print(emitters[args.shell](parser), end="")
        return 0
    if args.command == "man":
        print(man_page(parser), end="")
        return 0
    try:
        # The saved preset wins over the built-in default; an explicit flag wins over both.
        saved_preset = app_settings.load().get("preset") or DEFAULT_PRESET
        settings = resolve_preset(
            args.preset or saved_preset,
            embedder=args.embedder,
            rerank=args.rerank,
            llm=getattr(args, "llm_model", None) or getattr(args, "model", None),
        )
        embedder = get_embedder(settings["embedder"])
        reranker = get_reranker(settings["rerank"])
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.command == "mcp":
        return McpServer(db=args.db, embedder=embedder, reranker=reranker).serve()

    if args.command == "serve":
        state = AppState(
            db=args.db,
            embedder=embedder,
            rerank=settings["rerank"],
            llm_model=args.llm_model or settings["llm"],
            llm_host=args.llm_host,
            llm_spec=args.llm,
            preset=settings["preset"],
            ui_dir=args.ui,
        )
        server = make_server(state, port=args.port)
        host, port = server.server_address[:2]
        print(f"ragdesk serving on http://{host}:{port} (Ctrl-C to stop)")
        if args.watch_parent:
            parent = os.getppid()

            def watch_parent() -> None:
                while os.getppid() == parent:
                    time.sleep(3)
                os._exit(0)

            threading.Thread(target=watch_parent, daemon=True).start()

        def watch_loop() -> None:
            """Re-index changed files every watch_seconds (0 disables it)."""
            while True:
                values = app_settings.load()
                seconds = int(values.get("watch_seconds") or 0)
                if seconds <= 0:
                    time.sleep(30)
                    continue
                time.sleep(max(10, seconds))
                if state.activity.get("running"):
                    continue

                def progress(detail: str, done: int, total: int) -> None:
                    state.activity.update(
                        {"detail": detail, "done": done, "total": total}
                    )

                state.activity = {
                    "running": True,
                    "kind": "watch",
                    "detail": "checking for changes",
                    "done": 0,
                    "total": 0,
                    "started": time.time(),
                    "owner": f"watch:{time.time()}",
                }
                try:
                    summary = watch_pass(state)
                    if summary:
                        print(f"watch: {summary}", flush=True)
                except Exception as exc:  # noqa: BLE001 - the loop must outlive a bad pass
                    print(f"watch pass failed: {type(exc).__name__}: {exc}", flush=True)
                finally:
                    state.activity["running"] = False

        def auto_index_loop() -> None:
            while True:
                time.sleep(60)
                try:
                    values = app_settings.load()
                    if auto_index_due(values):
                        summary = run_auto_index(state)
                        print(f"auto-index: {summary}", flush=True)
                    released = release_idle_models(
                        state, float(values.get("idle_unload_minutes") or 0)
                    )
                    if released:
                        print(f"idle-release: {released}", flush=True)
                except Exception as exc:  # noqa: BLE001 - never let the timer die
                    print(
                        f"auto-index failed: {type(exc).__name__}: {exc}", flush=True
                    )

        threading.Thread(target=auto_index_loop, daemon=True).start()
        threading.Thread(target=watch_loop, daemon=True).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0

    with Store(args.db) as store:
        if args.command == "stats":
            stats = store.stats()
            embedder_name = store.get_meta("embedder.name")
            embedder_dim = store.get_meta("embedder.dim")
            if getattr(args, "json", False):
                emit_json(
                    {
                        "documents": stats["documents"],
                        "chunks": stats["chunks"],
                        "embedder": {"name": embedder_name, "dim": embedder_dim},
                        "sources": store.sources(),
                    }
                )
                return 0
            print(f"documents: {stats['documents']}")
            print(f"chunks   : {stats['chunks']}")
            print(f"embedder : {embedder_name} (dim {embedder_dim})")
            return 0

        if args.command == "index":
            stats = index_paths(store, embedder, args.paths)
            emit_stats(stats, getattr(args, 'json', False))
            return 0

        if args.command == "github":
            try:
                stats = sync_github(
                    store,
                    embedder,
                    repo=args.repo,
                    token=args.token,
                    ref=args.ref,
                    subdir=args.subdir,
                )
            except GitHubError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            print(
                f"scanned={stats.files_scanned} indexed={stats.indexed} "
                f"unchanged={stats.unchanged} chunks={stats.chunks}"
            )
            return 0

        if args.command == "confluence":
            try:
                stats = sync_confluence(
                    store,
                    embedder,
                    base_url=args.base_url,
                    space=args.space,
                    email=args.email,
                    token=args.token,
                    api_path=args.api_path,
                )
            except ConfluenceError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            emit_stats(stats, getattr(args, 'json', False))
            return 0

        if args.command == "gdrive":
            try:
                stats = sync_gdrive(
                    store,
                    embedder,
                    client_id=args.client_id,
                    client_secret=args.client_secret,
                    folder_id=args.folder_id,
                    interactive=not args.no_browser,
                )
            except GdriveError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            emit_stats(stats, getattr(args, 'json', False))
            return 0

        if args.command == "gitlab":
            try:
                stats = sync_gitlab(
                    store,
                    embedder,
                    project=args.project,
                    token=args.token,
                    ref=args.ref,
                    subdir=args.subdir,
                    base_url=args.base_url,
                )
            except GitLabError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            emit_stats(stats, getattr(args, 'json', False))
            return 0

        if args.command == "msgraph":
            from ragdesk import credentials as credentials_module

            try:
                client_id = resolve_client_id(args.client_id)
                if not credentials_module.get("msgraph").get("refresh_token"):
                    if client_id is None:
                        raise MsGraphError(
                            "no Microsoft client id: set RAGDESK_MS_CLIENT_ID "
                            "or pass --client-id (see docs/sources.md)"
                        )

                    def show_code(start: dict) -> None:
                        print(
                            f"Open {start.get('verification_uri')} and enter code: "
                            f"{start.get('user_code')}"
                        )

                    device_flow_connect(client_id, on_code=show_code)
                    print("connected.")
                stats = sync_msgraph(
                    store,
                    embedder,
                    site=args.site,
                    folder_id=args.folder_id,
                )
            except MsGraphError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            emit_stats(stats, getattr(args, 'json', False))
            return 0

        if args.command == "notion":
            try:
                stats = sync_notion(store, embedder, token=args.token)
            except NotionError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            emit_stats(stats, getattr(args, 'json', False))
            return 0

        if args.command == "web":
            try:
                stats = crawl_site(
                    store,
                    embedder,
                    start_url=args.url,
                    max_pages=args.max_pages,
                    max_depth=args.depth,
                )
            except WebError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            emit_stats(stats, getattr(args, 'json', False))
            return 0

        if args.command == "save":
            try:
                stats = save_page(store, embedder, args.url)
            except WebError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            emit_stats(stats, getattr(args, "json", False), url=args.url)
            return 0

        if args.command == "email":
            try:
                if args.mbox:
                    stats = index_mbox(
                        store, embedder, args.mbox, limit=args.limit
                    )
                    extra = {"mbox": str(args.mbox)}
                else:
                    password = args.password or _imap_password(args.user)
                    stats = sync_imap(
                        store,
                        embedder,
                        host=args.imap,
                        user=args.user,
                        password=password,
                        port=args.port,
                        folder=args.folder,
                        limit=args.limit,
                    )
                    extra = {"host": args.imap, "folder": args.folder}
            except EmailError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            emit_stats(stats, getattr(args, "json", False), **extra)
            return 0

        if args.command == "search":
            query, filters = parse_filters(args.query)
            hits = retrieve(
                store,
                embedder,
                query or args.query,
                top_k=args.top_k,
                reranker=reranker,
                filters=filters,
            )
            emit_hits(hits, getattr(args, "json", False))
            return 0

        if args.command == "ask":
            query, filters = parse_filters(args.query)
            args.query = query or args.query
            hits = retrieve(
                store,
                embedder,
                args.query,
                top_k=args.top_k,
                reranker=reranker,
                filters=filters,
            )
            spec = args.llm or (f"ollama:{args.model}" if args.model else None)
            try:
                llm = resolve_llm(spec, preset=settings["preset"], host=args.llm_host)
                if getattr(args, "json", False):
                    text = answer(args.query, hits, llm, min_cosine=args.min_cosine)
                    emit_json(
                        {
                            "answer": text,
                            "refused": text == REFUSAL,
                            "hits": [
                                {"path": hit.path, "line": hit.line, "lanes": hit.lanes}
                                for hit in hits
                            ],
                        }
                    )
                    return 0
                if args.stream:
                    for piece in answer_stream(
                        args.query,
                        hits,
                        llm,
                        min_cosine=args.min_cosine,
                    ):
                        print(piece, end="", flush=True)
                    print()
                else:
                    print(
                        answer(
                            args.query,
                            hits,
                            llm,
                            min_cosine=args.min_cosine,
                        )
                    )
            except (OllamaUnavailable, LLMUnavailable) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            if hits:
                print("\nSources:")
                for rank, hit in enumerate(hits, start=1):
                    print(f"  [{rank}] {hit.path}")
            return 0

        if args.command == "chat":
            from ragdesk.tui import run_chat

            try:
                chat_llm = resolve_llm(None, preset=settings["preset"])
            except (LLMUnavailable, OllamaUnavailable) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            return run_chat(
                store,
                embedder,
                chat_llm,
                chat_id=args.chat_id,
                reranker=reranker,
                top_k=args.top_k,
            )

        if args.command == "eval":
            golden = load_golden(args.golden)
            use_hyde = bool(app_settings.load().get("hyde"))
            hyde_for = None
            if use_hyde:
                try:
                    hyde_llm = resolve_llm(None, preset=settings["preset"])
                except LLMUnavailable:
                    hyde_llm = None
                if hyde_llm is not None:
                    hyde_for = lambda text: str(  # noqa: E731 - tiny adapter
                        hyde_llm.generate(
                            HYDE_PROMPT.format(question=text),
                            {"num_predict": 120, "temperature": 0.3},
                        )
                    ).strip()[:1200]
            rewrite_for = None
            if args.rewrite:
                try:
                    rewrite_llm = resolve_llm(None, preset=settings["preset"])
                except LLMUnavailable as exc:
                    print(f"--rewrite needs a local model: {exc}", file=sys.stderr)
                    return 2
                rewrite_for = lambda question, history: _standalone_rewrite(  # noqa: E731
                    rewrite_llm, question, history
                )
            baseline = None
            if rewrite_for is not None:
                baseline, _baseline_rows = evaluate(
                    store, embedder, golden, top_k=args.top_k, reranker=reranker
                )
            metrics, per_query = evaluate(
                store,
                embedder,
                golden,
                top_k=args.top_k,
                reranker=reranker,
                hyde_for=hyde_for,
                rewrite_for=rewrite_for,
            )
            report: dict[str, Any] = {"metrics": metrics, "queries": per_query}
            if baseline is not None:
                report["baseline"] = baseline
            if args.answers:
                answer_llm = resolve_llm(None, preset=settings["preset"])
                grounded: list[dict[str, Any]] = []
                for row in per_query:
                    hits = retrieve(
                        store, embedder, row["query"], top_k=args.top_k, reranker=reranker
                    )
                    text = answer(row["query"], hits, answer_llm)
                    verdict = ground_answer(
                        text, [hit.context for hit in hits]
                    )
                    row["grounded_ratio"] = verdict["grounded_ratio"]
                    row["citation_valid"] = verdict["citation_valid"]
                    grounded.append(verdict)
                metrics["grounded_ratio"] = (
                    sum(item["grounded_ratio"] for item in grounded) / (len(grounded) or 1)
                )
                metrics["citation_valid"] = (
                    sum(1.0 for item in grounded if item["citation_valid"])
                    / (len(grounded) or 1)
                )
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                if baseline is not None:
                    print("baseline (no rewrite):")
                    print(format_report(baseline, []))
                    print("with rewrite:")
                print(format_report(metrics, per_query))
                if args.answers:
                    print(f"grounded: {metrics['grounded_ratio']:.3f}")
                    print(f"citations valid: {metrics['citation_valid']:.3f}")
            if args.min_recall is not None and metrics["recall@5"] < args.min_recall:
                print(
                    f"eval gate failed: recall@5 {metrics['recall@5']:.3f} < {args.min_recall:.3f}",
                    file=sys.stderr,
                )
                return 1
            for gate in args.min_recall_category:
                name, _, raw = gate.partition("=")
                try:
                    floor = float(raw)
                except ValueError:
                    print(f"bad gate {gate!r}; expected NAME=VALUE", file=sys.stderr)
                    return 2
                grouped = category_metrics(per_query)
                value = grouped.get(name, {}).get("recall@5")
                if value is None:
                    print(f"eval gate: no queries in category {name!r}", file=sys.stderr)
                    return 1
                if value < floor:
                    print(
                        f"eval gate failed: [{name}] recall@5 {value:.3f} < {floor:.3f}",
                        file=sys.stderr,
                    )
                    return 1
            return 0

    return 0
