"""Topic map: the shape of the corpus, as labelled clusters of documents.

Greedy leader clustering over the document average vectors that
``store._doc_vectors`` already computes (pure Python, no numpy), labelled with
the terms that are most distinctive to the cluster. Display only: this is a
view of the corpus, not a retrieval lane — nothing here changes ranking.
"""

from __future__ import annotations

import math
from collections import Counter

from ragdesk.embed import tokenize
from ragdesk.store import Store

CLUSTER_THRESHOLD = 0.75  # same-topic documents scored 0.8+ when measured
MAX_DOCS = 2000  # ponytail: O(docs × clusters); needs numpy beyond this
MAX_CLUSTERS = 40
LABEL_TERMS = 5
MIN_TERM_LENGTH = 3


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    return dot / (left_norm * right_norm)


def _merge(centroid: list[float], count: int, vector: list[float]) -> list[float]:
    total = count + 1
    return [(a * count + b) / total for a, b in zip(centroid, vector, strict=False)]


def cluster_documents(store: Store, threshold: float = CLUSTER_THRESHOLD) -> list[dict]:
    """Greedy leader clustering of documents by their average chunk vector."""
    vectors = store._doc_vectors()  # noqa: SLF001 - the shared average-vector helper
    if not vectors:
        return []
    items = sorted(vectors.items())[:MAX_DOCS]
    clusters: list[dict] = []
    for doc_id, (path, vector) in items:
        best: dict | None = None
        best_score = 0.0
        for cluster in clusters:
            score = _cosine(cluster["centroid"], vector)
            if score >= threshold and score > best_score:
                best, best_score = cluster, score
        if best is None:
            clusters.append(
                {
                    "centroid": list(vector),
                    "doc_ids": [doc_id],
                    "paths": [path],
                    "cohesion": 1.0,
                }
            )
            continue
        best["doc_ids"].append(doc_id)
        best["paths"].append(path)
        best["centroid"] = _merge(best["centroid"], len(best["doc_ids"]) - 1, vector)
    clusters.sort(key=lambda cluster: -len(cluster["doc_ids"]))
    return clusters[:MAX_CLUSTERS]


def cluster_labels(store: Store, clusters: list[dict], limit: int = LABEL_TERMS) -> None:
    """Attach the most distinctive terms of each cluster, in place.

    Distinctiveness = term frequency inside the cluster minus the corpus-wide
    rate, so "the most common words" do not win the label.
    """
    corpus_counts: Counter[str] = Counter()
    corpus_tokens = 0
    # ponytail: the corpus-wide baseline is sampled past 20k chunks — labels
    # stay useful, and the pass stays fast on big corpora.
    for row in store.conn.execute("SELECT text FROM chunks LIMIT 20000"):
        tokens = [token for token in tokenize(str(row["text"])) if len(token) >= MIN_TERM_LENGTH]
        corpus_counts.update(tokens)
        corpus_tokens += len(tokens)

    for cluster in clusters:
        placeholders = ",".join("?" for _ in cluster["doc_ids"])
        counts: Counter[str] = Counter()
        total = 0
        rows = store.conn.execute(
            f"SELECT text FROM chunks WHERE doc_id IN ({placeholders})",  # noqa: S608 - ids are ints
            cluster["doc_ids"],
        ).fetchall()
        for row in rows:
            tokens = [
                token for token in tokenize(str(row["text"])) if len(token) >= MIN_TERM_LENGTH
            ]
            counts.update(tokens)
            total += len(tokens)
        if not total:
            cluster["label"] = "(no text)"
            continue
        scored = [
            (
                (count / total) - (corpus_counts[term] / max(1, corpus_tokens)),
                term,
            )
            for term, count in counts.items()
            if count >= 2
        ]
        scored.sort(reverse=True)
        cluster["label"] = ", ".join(term for _score, term in scored[:limit]) or "(mixed)"


def topic_map(store: Store, threshold: float = CLUSTER_THRESHOLD) -> list[dict]:
    """Display-ready clusters: label, size and a few sample paths."""
    clusters = cluster_documents(store, threshold)
    if not clusters:
        return []
    cluster_labels(store, clusters)
    return [
        {
            "label": cluster.get("label", ""),
            "documents": len(cluster["doc_ids"]),
            "paths": cluster["paths"],
        }
        for cluster in clusters
    ]
