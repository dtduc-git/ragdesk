"""ragdesk command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ragdesk import __version__
from ragdesk.answer import answer, answer_stream
from ragdesk.confluence import ConfluenceError, sync_confluence
from ragdesk.embed import get_embedder
from ragdesk.evaluate import evaluate, format_report, load_golden
from ragdesk.gdrive import GdriveError, sync_gdrive
from ragdesk.github import GitHubError, sync_github
from ragdesk.index import index_paths
from ragdesk.ollama import DEFAULT_HOST, OllamaUnavailable
from ragdesk.presets import DEFAULT_PRESET, PRESETS
from ragdesk.presets import resolve as resolve_preset
from ragdesk.rerank import get_reranker
from ragdesk.search import retrieve
from ragdesk.serve import AppState, make_server
from ragdesk.store import Store
from ragdesk.web import WebError, crawl_site

DEFAULT_DB = ".ragdesk/index.db"


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
        default=DEFAULT_PRESET,
        help="hardware preset for embedder/reranker/LLM defaults (default: light)",
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

    p_search = sub.add_parser("search", help="hybrid retrieval (no LLM)")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=8)

    p_ask = sub.add_parser("ask", help="cited answer via a local Ollama LLM")
    p_ask.add_argument("query")
    p_ask.add_argument("--model", default=None, help="LLM model (default: from preset)")
    p_ask.add_argument(
        "--min-cosine",
        type=float,
        default=0.0,
        help="grounding gate: refuse below this dense cosine (calibrate per embedder)",
    )
    p_ask.add_argument("--top-k", type=int, default=6)
    p_ask.add_argument("--stream", action="store_true", help="print tokens as they arrive")

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

    sub.add_parser("stats", help="index statistics")

    p_gh = sub.add_parser("github", help="index a GitHub repository (read-only tarball sync)")
    p_gh.add_argument("repo", help="owner/name")
    p_gh.add_argument("--ref", default="", help="branch/tag/sha (default: repo default branch)")
    p_gh.add_argument("--subdir", default="", help="index only a subtree")
    p_gh.add_argument(
        "--token", default=None, help="GitHub token (default: GITHUB_TOKEN env or gh auth token)"
    )

    p_cf = sub.add_parser("confluence", help="index a Confluence space (read-only)")
    p_cf.add_argument("space", help="space key, e.g. DOCS")
    p_cf.add_argument("--base-url", required=True, help="e.g. https://team.atlassian.net")
    p_cf.add_argument("--email", default=None, help="default: CONFLUENCE_EMAIL env")
    p_cf.add_argument("--token", default=None, help="default: CONFLUENCE_TOKEN env")
    p_cf.add_argument(
        "--api-path", default="/wiki/rest/api/content/search", help="REST path (Server/DC differs)"
    )

    p_gd = sub.add_parser("gdrive", help="index Google Drive (read-only, BYO OAuth client)")
    p_gd.add_argument("--folder-id", default="", help="index one folder (default: all files)")
    p_gd.add_argument("--client-id", default="", help="default: GDRIVE_CLIENT_ID env")
    p_gd.add_argument("--client-secret", default="", help="default: GDRIVE_CLIENT_SECRET env")
    p_gd.add_argument(
        "--no-browser", action="store_true", help="do not run the interactive OAuth flow"
    )

    p_web = sub.add_parser("web", help="crawl a docs site and index it (read-only)")
    p_web.add_argument("url", help="start URL, e.g. https://docs.example.com/")
    p_web.add_argument("--max-pages", type=int, default=50)
    p_web.add_argument("--depth", type=int, default=2)

    p_serve = sub.add_parser("serve", help="local HTTP API for the desktop app")
    p_serve.add_argument("--port", type=int, default=8765)
    p_serve.add_argument("--llm-model", default=None, help="LLM model (default: from preset)")
    p_serve.add_argument("--llm-host", default=DEFAULT_HOST)
    p_serve.add_argument("--ui", default="", help="serve a built UI directory (browser mode)")
    return parser


def _print_hits(hits) -> None:
    for rank, hit in enumerate(hits, start=1):
        snippet = " ".join(hit.text.split())[:160]
        print(f"{rank}. score={hit.score:.4f} cos={hit.cosine:.3f} [{hit.lanes}] {hit.path}")
        print(f"   {snippet}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        settings = resolve_preset(
            args.preset,
            embedder=args.embedder,
            rerank=args.rerank,
            llm=getattr(args, "llm_model", None) or getattr(args, "model", None),
        )
        embedder = get_embedder(settings["embedder"])
        reranker = get_reranker(settings["rerank"])
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.command == "serve":
        state = AppState(
            db=args.db,
            embedder=embedder,
            rerank=settings["rerank"],
            llm_model=settings["llm"],
            llm_host=args.llm_host,
            preset=settings["preset"],
            ui_dir=args.ui,
        )
        server = make_server(state, port=args.port)
        host, port = server.server_address[:2]
        print(f"ragdesk serving on http://{host}:{port} (Ctrl-C to stop)")
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
            print(f"documents: {stats['documents']}")
            print(f"chunks   : {stats['chunks']}")
            print(f"embedder : {embedder_name} (dim {embedder_dim})")
            return 0

        if args.command == "index":
            stats = index_paths(store, embedder, args.paths)
            print(
                f"scanned={stats.files_scanned} indexed={stats.indexed} "
                f"unchanged={stats.unchanged} skipped={stats.skipped} chunks={stats.chunks}"
            )
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
            print(
                f"scanned={stats.files_scanned} indexed={stats.indexed} "
                f"unchanged={stats.unchanged} skipped={stats.skipped} chunks={stats.chunks}"
            )
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
            print(
                f"scanned={stats.files_scanned} indexed={stats.indexed} "
                f"unchanged={stats.unchanged} skipped={stats.skipped} chunks={stats.chunks}"
            )
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
            print(
                f"scanned={stats.files_scanned} indexed={stats.indexed} "
                f"unchanged={stats.unchanged} skipped={stats.skipped} chunks={stats.chunks}"
            )
            return 0

        if args.command == "search":
            hits = retrieve(
                store, embedder, args.query, top_k=args.top_k, reranker=reranker
            )
            _print_hits(hits)
            return 0

        if args.command == "ask":
            hits = retrieve(
                store, embedder, args.query, top_k=args.top_k, reranker=reranker
            )
            try:
                if args.stream:
                    for piece in answer_stream(
                        args.query,
                        hits,
                        model=settings["llm"],
                        min_cosine=args.min_cosine,
                    ):
                        print(piece, end="", flush=True)
                    print()
                else:
                    print(
                        answer(
                            args.query,
                            hits,
                            model=settings["llm"],
                            min_cosine=args.min_cosine,
                        )
                    )
            except OllamaUnavailable as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            if hits:
                print("\nSources:")
                for rank, hit in enumerate(hits, start=1):
                    print(f"  [{rank}] {hit.path}")
            return 0

        if args.command == "eval":
            golden = load_golden(args.golden)
            metrics, per_query = evaluate(
                store, embedder, golden, top_k=args.top_k, reranker=reranker
            )
            if args.json:
                print(json.dumps({"metrics": metrics, "queries": per_query}, indent=2))
            else:
                print(format_report(metrics, per_query))
            if args.min_recall is not None and metrics["recall@5"] < args.min_recall:
                print(
                    f"eval gate failed: recall@5 {metrics['recall@5']:.3f} < {args.min_recall:.3f}",
                    file=sys.stderr,
                )
                return 1
            return 0

    return 0
