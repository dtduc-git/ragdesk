"""Hybrid search: BM25 + dense, fused with Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ragdesk.embed import Embedder
from ragdesk.store import Store

if TYPE_CHECKING:
    from ragdesk.rerank import Reranker

RRF_K = 60
RERANK_POOL = 30


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    doc_id: int
    path: str
    source: str
    ordinal: int
    text: str
    score: float
    cosine: float
    lanes: str


def fuse(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion over chunk-id rankings."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def hybrid_search(
    store: Store,
    embedder: Embedder,
    query: str,
    top_k: int = 8,
    pool: int = 50,
) -> list[Hit]:
    """BM25 + dense lanes, RRF-fused, chunk-level."""
    query_vec = embedder.embed_query(query)
    dense_rows = store.dense_search(query_vec, pool)
    bm25_rows = store.bm25_search(query, pool)

    payloads: dict[int, dict] = {}
    for row in dense_rows:
        payloads[row["id"]] = row
    for row in bm25_rows:
        payloads.setdefault(row["id"], row)
    cosine = {row["id"]: row["score"] for row in dense_rows}
    in_bm25 = {row["id"] for row in bm25_rows}
    in_dense = {row["id"] for row in dense_rows}

    fused = fuse([[row["id"] for row in bm25_rows], [row["id"] for row in dense_rows]])
    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]

    hits: list[Hit] = []
    for chunk_id, score in ranked:
        row = payloads[chunk_id]
        lanes = "+".join(
            lane for lane, member in (("bm25", in_bm25), ("dense", in_dense)) if chunk_id in member
        )
        hits.append(
            Hit(
                chunk_id=chunk_id,
                doc_id=row["doc_id"],
                path=row["path"],
                source=row["source"],
                ordinal=row["ordinal"],
                text=row["text"],
                score=score,
                cosine=cosine.get(chunk_id, 0.0),
                lanes=lanes,
            )
        )
    return hits


def retrieve(
    store: Store,
    embedder: Embedder,
    query: str,
    top_k: int = 8,
    reranker: Reranker | None = None,
    pool: int = RERANK_POOL,
) -> list[Hit]:
    """Two-stage retrieval: fuse a larger candidate pool, optionally rerank,
    then cut to ``top_k``."""
    if reranker is not None:
        candidates = hybrid_search(store, embedder, query, top_k=max(top_k, pool))
        return reranker.rerank(query, candidates)[:top_k]
    return hybrid_search(store, embedder, query, top_k=top_k)
