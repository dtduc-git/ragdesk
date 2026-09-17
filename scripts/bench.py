"""Retrieval benchmark: index once per chunking config, then sweep rerankers
and lane weights over the same index. Every number in the README comes from
here; run it before and after retrieval changes.

    uv run --extra onnx python scripts/bench.py --root . --golden fixtures/golden_repo.jsonl
    uv run --extra onnx python scripts/bench.py --sizes 600,1000,1500 \
        --reranks none,fastembed:BAAI/bge-reranker-base
    uv run --extra onnx python scripts/bench.py --weights "dense=1.0,bm25=0.9,path=0.6"
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ragdesk.embed import get_embedder  # noqa: E402
from ragdesk.evaluate import category_metrics, evaluate, load_golden  # noqa: E402
from ragdesk.index import index_paths  # noqa: E402
from ragdesk.rerank import get_reranker  # noqa: E402
from ragdesk.store import Store  # noqa: E402


def parse_weights(raw: str) -> dict[str, float] | None:
    if not raw:
        return None
    weights: dict[str, float] = {}
    for part in raw.split(","):
        key, _, value = part.partition("=")
        if key.strip() and value.strip():
            weights[key.strip()] = float(value)
    return weights or None


def build_index(root: Path, db: Path, embedder, chars: int, overlap: int) -> None:
    with Store(db) as store:
        index_paths(store, embedder, [root], chunk_chars=chars, chunk_overlap=overlap)


def score(
    db: Path,
    golden: list[dict],
    embedder,
    reranker,
    weights: dict[str, float] | None,
    top_k: int,
) -> tuple[dict, list[dict]]:
    if weights is None:
        with Store(db) as store:
            return evaluate(store, embedder, golden, top_k=top_k, reranker=reranker)

    # Manual pass so the lane weights reach hybrid_search.
    import ragdesk.evaluate as evaluate_module
    from ragdesk.search import parse_filters

    per_query: list[dict] = []
    with Store(db) as store:
        for row in golden:
            query, filters = parse_filters(row["query"])
            from ragdesk.search import hybrid_search

            hits = hybrid_search(
                store, embedder, query, top_k=top_k, filters=filters, weights=weights
            )
            ranking = evaluate_module._rank_docs(hits)
            relevant = row["relevant"]

            def is_relevant(path: str, targets: list[str] = relevant) -> bool:
                return any(evaluate_module._matches(path, target) for target in targets)

            top5 = [p for p in ranking[:5] if is_relevant(p)]
            per_query.append(
                {
                    "query": row["query"],
                    "category": str(row.get("category") or "uncategorised"),
                    "recall@5": len(top5) / len(relevant) if relevant else 0.0,
                    "ndcg@10": 0.0,  # weights mode ranks recall only
                    "mrr@10": 0.0,
                    "top_docs": ranking[:5],
                }
            )
    metrics = {
        "queries": float(len(per_query)),
        "recall@5": sum(r["recall@5"] for r in per_query) / (len(per_query) or 1),
        "ndcg@10": 0.0,
        "mrr@10": 0.0,
    }
    return metrics, per_query


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="folder to index (default: repo)")
    parser.add_argument("--golden", default="fixtures/golden_repo.jsonl")
    parser.add_argument("--sizes", default="1000", help="chunk chars, comma separated")
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--embedder", default="onnx")
    parser.add_argument("--reranks", default="none", help="rerank specs, comma separated")
    parser.add_argument("--weights", default="", help='l"dense=1.0,bm25=0.9"')
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", default="", help="write the full results here")
    parser.add_argument(
        "--db", default="", help="reuse this index instead of building one per size"
    )
    args = parser.parse_args()

    golden = load_golden(args.golden)
    embedder = get_embedder(args.embedder)
    sizes = [int(value) for value in args.sizes.split(",") if value.strip()]
    reranks = [value for value in args.reranks.split(",") if value.strip()] or ["none"]
    weights = parse_weights(args.weights)

    results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="ragdesk-bench-") as tmp:
        for chars in sizes:
            if args.db:
                db = Path(args.db)
                build_seconds = 0.0
            else:
                db = Path(tmp) / f"index-{chars}.db"
                started = time.perf_counter()
                build_index(Path(args.root), db, embedder, chars, args.overlap)
                build_seconds = time.perf_counter() - started
            with Store(db) as store:
                stats = store.stats()
            for spec in reranks:
                reranker = None if spec == "none" else get_reranker(spec)
                metrics, per_query = score(db, golden, embedder, reranker, weights, args.top_k)
                row = {
                    "chunk_chars": chars,
                    "overlap": args.overlap,
                    "rerank": spec,
                    "weights": weights or {},
                    "chunks": stats["chunks"],
                    "index_seconds": round(build_seconds, 1),
                    **{key: round(value, 3) for key, value in metrics.items()},
                    "categories": {
                        name: round(entry["recall@5"], 3)
                        for name, entry in category_metrics(per_query).items()
                    },
                }
                results.append(row)
                print(
                    f"chars={chars:<5} rerank={spec:<28} chunks={stats['chunks']:<6} "
                    f"recall@5={row['recall@5']:.3f} nDCG@10={row['ndcg@10']:.3f} "
                    f"MRR@10={row['mrr@10']:.3f}  ({row['index_seconds']}s index)"
                )

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json}")
    best = max(results, key=lambda row: (row["ndcg@10"], row["mrr@10"]))
    print(
        f"best: chars={best['chunk_chars']} rerank={best['rerank']} "
        f"nDCG={best['ndcg@10']:.3f} MRR={best['mrr@10']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
