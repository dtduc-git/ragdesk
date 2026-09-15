"""Retrieval eval harness — the numbers ragdesk publishes.

Golden set format (JSONL, one object per line):

    {"query": "how do access tokens expire", "relevant": ["docs/auth.md"],
     "category": "auth"}

``relevant`` entries match by exact path or path suffix, so fixtures stay
portable between machines. ``category`` is optional and groups the report, so
one weak area cannot hide behind a strong overall number.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ragdesk.embed import Embedder
from ragdesk.search import Hit, parse_filters, retrieve
from ragdesk.store import Store

Sentence = Callable[[str], list[str]]
_CITATION_RE = re.compile(r"\[(\d+)\]")
HydeFor = Callable[[str], str]


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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate(
    store: Store,
    embedder: Embedder,
    golden: list[dict[str, Any]],
    top_k: int = 10,
    reranker: Any = None,
    hyde_for: HydeFor | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    per_query: list[dict[str, Any]] = []
    for row in golden:
        query, filters = parse_filters(row["query"])
        hyde_text = hyde_for(query) if hyde_for is not None else ""
        hits = retrieve(
            store,
            embedder,
            query,
            top_k=top_k,
            reranker=reranker,
            hyde_text=hyde_text,
            filters=filters,
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
                "category": str(row.get("category") or "uncategorised"),
                "recall@5": recall,
                "ndcg@10": ndcg,
                "mrr@10": reciprocal,
                "top_docs": ranking[:5],
            }
        )

    count = len(per_query) or 1
    metrics: dict[str, float] = {
        "queries": float(len(per_query)),
        "recall@5": sum(r["recall@5"] for r in per_query) / count,
        "ndcg@10": sum(r["ndcg@10"] for r in per_query) / count,
        "mrr@10": sum(r["mrr@10"] for r in per_query) / count,
    }
    return metrics, per_query


def category_metrics(per_query: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Group the per-query rows so a weak area stays visible."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in per_query:
        grouped.setdefault(str(row["category"]), []).append(row)
    return {
        name: {
            "queries": float(len(rows)),
            "recall@5": _mean([row["recall@5"] for row in rows]),
            "ndcg@10": _mean([row["ndcg@10"] for row in rows]),
            "mrr@10": _mean([row["mrr@10"] for row in rows]),
        }
        for name, rows in sorted(grouped.items())
    }


def ground_answer(answer: str, citations: list[str]) -> dict[str, Any]:
    """Deterministic faithfulness proxy: does every answer sentence live in the
    cited chunks, and is every ``[n]`` citation in range?

    Cheaper and reproducible next to an LLM judge, and it catches the failure
    this repo actually fears: an answer that cites sources it does not use.
    """
    if not answer.strip():
        return {"citation_valid": False, "grounded_ratio": 0.0, "sentences": 0}

    max_ref = len(citations)
    refs = [int(match) for match in _CITATION_RE.findall(answer)]
    citation_valid = bool(refs) and all(1 <= ref <= max_ref for ref in refs)

    corpus_tokens = set(re.findall(r"\w+", " ".join(citations).lower(), flags=re.UNICODE))
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?…])\s+", answer)
        if part.strip()
    ]
    grounded = 0
    for sentence in sentences:
        tokens = set(re.findall(r"\w+", sentence.lower(), flags=re.UNICODE))
        if not tokens:
            continue
        overlap = len(tokens & corpus_tokens) / len(tokens)
        if overlap >= 0.5:
            grounded += 1
    return {
        "citation_valid": citation_valid,
        "grounded_ratio": grounded / len(sentences) if sentences else 0.0,
        "sentences": len(sentences),
    }


def format_report(metrics: dict[str, float], per_query: list[dict[str, Any]]) -> str:
    lines = [
        f"queries : {int(metrics['queries'])}",
        f"recall@5: {metrics['recall@5']:.3f}",
        f"ndcg@10 : {metrics['ndcg@10']:.3f}",
        f"mrr@10  : {metrics['mrr@10']:.3f}",
    ]
    for name, row in category_metrics(per_query).items():
        lines.append(
            f"  [{name}] n={int(row['queries'])} recall@5={row['recall@5']:.3f} "
            f"mrr@10={row['mrr@10']:.3f}"
        )
    misses = [row for row in per_query if row["recall@5"] < 1.0]
    for row in misses:
        lines.append(f"  miss: {row['query']!r} -> {row['top_docs']}")
    return "\n".join(lines)
