"""Retrieval eval harness — the numbers ragdesk publishes.

Golden set format (JSONL, one object per line):

    {"query": "how do access tokens expire", "relevant": ["docs/auth.md"]}

``relevant`` entries match by exact path or path suffix, so fixtures stay
portable between machines.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ragdesk.embed import Embedder
from ragdesk.search import Hit, retrieve
from ragdesk.store import Store


def load_golden(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _matches(path: str, relevant: str) -> bool:
    return path == relevant or path.endswith(relevant) or relevant.endswith(path)


def _rank_docs(hits: list[Hit]) -> list[str]:
    seen: set[str] = set()
    ranking: list[str] = []
    for hit in hits:
        if hit.path not in seen:
            seen.add(hit.path)
            ranking.append(hit.path)
    return ranking


def evaluate(
    store: Store,
    embedder: Embedder,
    golden: list[dict[str, Any]],
    top_k: int = 10,
    reranker: Any = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    per_query: list[dict[str, Any]] = []
    for row in golden:
        hits = retrieve(
            store, embedder, row["query"], top_k=top_k, reranker=reranker
        )
        ranking = _rank_docs(hits)
        relevant = row["relevant"]

        def is_relevant(path: str, targets: list[str] = relevant) -> bool:
            return any(_matches(path, target) for target in targets)

        top5 = [p for p in ranking[:5] if is_relevant(p)]
        recall = len(top5) / len(relevant) if relevant else 0.0

        dcg = sum(
            1.0 / math.log2(rank + 2)
            for rank, path in enumerate(ranking[:10])
            if is_relevant(path)
        )
        ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(relevant), 10)))
        ndcg = dcg / ideal if ideal else 0.0
        reciprocal = next(
            (1.0 / (rank + 1) for rank, path in enumerate(ranking[:10]) if is_relevant(path)),
            0.0,
        )

        per_query.append(
            {
                "query": row["query"],
                "recall@5": recall,
                "ndcg@10": ndcg,
                "mrr@10": reciprocal,
                "top_docs": ranking[:5],
            }
        )

    count = len(per_query) or 1
    metrics = {
        "queries": float(len(per_query)),
        "recall@5": sum(r["recall@5"] for r in per_query) / count,
        "ndcg@10": sum(r["ndcg@10"] for r in per_query) / count,
        "mrr@10": sum(r["mrr@10"] for r in per_query) / count,
    }
    return metrics, per_query


def format_report(metrics: dict[str, float], per_query: list[dict[str, Any]]) -> str:
    lines = [
        f"queries : {int(metrics['queries'])}",
        f"recall@5: {metrics['recall@5']:.3f}",
        f"ndcg@10 : {metrics['ndcg@10']:.3f}",
        f"mrr@10  : {metrics['mrr@10']:.3f}",
    ]
    misses = [row for row in per_query if row["recall@5"] < 1.0]
    for row in misses:
        lines.append(f"  miss: {row['query']!r} -> {row['top_docs']}")
    return "\n".join(lines)
