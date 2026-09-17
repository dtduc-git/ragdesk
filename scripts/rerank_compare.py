#!/usr/bin/env python3
"""Compare reranker models on the golden set: quality, latency, RSS and size.

Same idea as scripts/embedder_compare.py, one stage later in the pipeline.
Runs at the shipped pool (RERANK_POOL) so the numbers answer "what would a
question cost if this model were the reranker".

    uv run ragdesk --db /tmp/repo.db index "$PWD"
    scripts/rerank_compare.py --db /tmp/repo.db
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from ragdesk.embed import get_embedder
from ragdesk.evaluate import evaluate
from ragdesk.rerank import get_reranker
from ragdesk.search import RERANK_POOL
from ragdesk.store import Store

ROOT = Path(__file__).resolve().parent.parent

CANDIDATES = [
    "onnx:onnx-community/gte-multilingual-reranker-base",  # shipped
    "onnx:mixedbread-ai/mxbai-rerank-base-v1",
    "onnx:mixedbread-ai/mxbai-rerank-xsmall-v1",
    "onnx:cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
]


def rss_mb() -> int:
    """Current RSS of this process — ru_maxrss is a high-water mark and would
    make every later candidate look identical."""
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True
    ).stdout.strip()
    return int(out) // 1024 if out else 0


def model_size(repo: str) -> str:
    cache = Path.home() / ".cache/huggingface/hub" / f"models--{repo.replace('/', '--')}"
    blobs = cache / "blobs"
    if not blobs.is_dir():
        return "?"
    total = sum(f.stat().st_size for f in blobs.rglob("*") if f.is_file())
    return f"{total / (1024**2):.0f}M"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--golden", default=str(ROOT / "fixtures/golden_repo.jsonl"))
    parser.add_argument("--candidates", nargs="*", default=CANDIDATES)
    parser.add_argument("--pool", type=int, default=RERANK_POOL)
    args = parser.parse_args()

    golden = [json.loads(line) for line in open(args.golden, encoding="utf-8")]
    store = Store(args.db)
    embedder = get_embedder("onnx")
    # Warm the embedder first: it is the same for every candidate, and paying
    # for it inside the first row would make that row look heavier. The same
    # pass is the no-reranker baseline the candidates have to beat.
    started = time.monotonic()
    base_metrics, _ = evaluate(store, embedder, golden, top_k=10, reranker=None)
    base_seconds = (time.monotonic() - started) / len(golden)
    baseline = rss_mb()
    print(
        f"| *(no reranker)* | - | {base_metrics['recall@5']:.3f} | {base_metrics['ndcg@10']:.3f} "
        f"| {base_metrics['mrr@10']:.3f} | {base_seconds:.2f}s | - |",
        flush=True,
    )
    print(f"# pool {args.pool} · {len(golden)} queries · process baseline {baseline} MB\n")
    print("| reranker | weights | recall@5 | ndcg@10 | mrr@10 | per question | loaded RAM |")
    print("|---|---|---|---|---|---|---|")

    for spec in args.candidates:
        repo = spec.split(":", 1)[1]
        reranker = get_reranker(spec)
        before = rss_mb()
        started = time.monotonic()
        metrics, rows = evaluate(
            store, embedder, golden, top_k=10, reranker=reranker, pool=args.pool
        )
        seconds = (time.monotonic() - started) / len(golden)
        # Sample after the run, not after the load: onnxruntime mmaps the weights
        # and only faults pages in as inference touches them.
        resident = rss_mb() - before
        if hasattr(reranker, "unload"):
            reranker.unload()
        print(
            f"| `{repo}` | {model_size(repo)} | {metrics['recall@5']:.3f} "
            f"| {metrics['ndcg@10']:.3f} | {metrics['mrr@10']:.3f} "
            f"| {seconds:.2f}s | {resident} MB |",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
