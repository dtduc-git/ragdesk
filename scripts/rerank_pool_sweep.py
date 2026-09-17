#!/usr/bin/env python3
"""Size the reranker's candidate pool: quality and latency per pool size.

The pool is how many chunks the cross-encoder re-scores per question, so both
cost and quality scale with it. Reranking quality only changes when a relevant
chunk was in the pool and got pushed out of the top-k, which is what this
measures. Point it at an index of this repo built with the shipped embedder:

    uv run ragdesk --db /tmp/repo.db index "$PWD"
    scripts/rerank_pool_sweep.py --db /tmp/repo.db --pools 50 30 20 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ragdesk.embed import get_embedder
from ragdesk.evaluate import evaluate
from ragdesk.rerank import get_reranker
from ragdesk.store import Store

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="index of this repo (see docstring)")
    parser.add_argument("--golden", default=str(ROOT / "fixtures/golden_repo.jsonl"))
    parser.add_argument("--pools", type=int, nargs="*", default=[50, 30, 20, 10])
    parser.add_argument("--embedder", default="onnx")
    args = parser.parse_args()

    golden = [json.loads(line) for line in open(args.golden, encoding="utf-8")]
    store = Store(args.db)
    embedder = get_embedder(args.embedder)
    reranker = get_reranker("onnx")

    print("\n| pool | recall@5 | ndcg@10 | mrr@10 | per question |")
    print("|---|---|---|---|---|")
    for pool in args.pools:
        mark = time.monotonic()
        metrics, _rows = evaluate(store, embedder, golden, top_k=10, reranker=reranker, pool=pool)
        seconds = (time.monotonic() - mark) / len(golden)
        print(
            f"| {pool} | {metrics['recall@5']:.3f} | {metrics['ndcg@10']:.3f} "
            f"| {metrics['mrr@10']:.3f} | {seconds:.2f}s |",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
