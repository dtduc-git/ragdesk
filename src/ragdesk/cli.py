"""ragdesk command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ragdesk import __version__
from ragdesk.answer import DEFAULT_LLM_MODEL, answer
from ragdesk.embed import DEFAULT_OLLAMA_MODEL, get_embedder
from ragdesk.evaluate import evaluate, format_report, load_golden
from ragdesk.index import index_paths
from ragdesk.ollama import DEFAULT_HOST, OllamaUnavailable
from ragdesk.rerank import get_reranker
from ragdesk.search import retrieve
from ragdesk.serve import AppState, make_server
from ragdesk.store import Store

DEFAULT_DB = ".ragdesk/index.db"
DEFAULT_EMBEDDER = f"ollama:{DEFAULT_OLLAMA_MODEL}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragdesk",
        description="Personal, local-first RAG over your own sources.",
    )
    parser.add_argument("--version", action="version", version=f"ragdesk {__version__}")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"index database (default: {DEFAULT_DB})")
    parser.add_argument(
        "--embedder",
        default=DEFAULT_EMBEDDER,
        help=f"embedder spec: ollama[:model] or hash[:dim] (default: {DEFAULT_EMBEDDER})",
    )
    parser.add_argument(
        "--rerank",
        default="none",
        help="reranker spec: none | lexical | fastembed[:model] (default: none)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="index local files/directories")
    p_index.add_argument("paths", nargs="+", type=Path)

    p_search = sub.add_parser("search", help="hybrid retrieval (no LLM)")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=8)

    p_ask = sub.add_parser("ask", help="cited answer via a local Ollama LLM")
    p_ask.add_argument("query")
    p_ask.add_argument("--model", default=DEFAULT_LLM_MODEL)
    p_ask.add_argument(
        "--min-cosine",
        type=float,
        default=0.0,
        help="grounding gate: refuse below this dense cosine (calibrate per embedder)",
    )
    p_ask.add_argument("--top-k", type=int, default=6)

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

    p_serve = sub.add_parser("serve", help="local HTTP API for the desktop app")
    p_serve.add_argument("--port", type=int, default=8765)
    p_serve.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    p_serve.add_argument("--llm-host", default=DEFAULT_HOST)
    return parser


def _print_hits(hits) -> None:
    for rank, hit in enumerate(hits, start=1):
        snippet = " ".join(hit.text.split())[:160]
        print(f"{rank}. score={hit.score:.4f} cos={hit.cosine:.3f} [{hit.lanes}] {hit.path}")
        print(f"   {snippet}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        embedder = get_embedder(args.embedder)
        reranker = get_reranker(args.rerank)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.command == "serve":
        state = AppState(
            db=args.db,
            embedder=embedder,
            rerank=args.rerank,
            llm_model=args.llm_model,
            llm_host=args.llm_host,
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
                text = answer(args.query, hits, model=args.model, min_cosine=args.min_cosine)
            except OllamaUnavailable as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            print(text)
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
