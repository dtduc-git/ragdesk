"""Hybrid search: BM25 + dense + path lanes, fused with Reciprocal Rank Fusion.

Optional extra lanes: a HyDE vector (an LLM-written hypothetical answer) and
query scoping via ``folder:`` / ``source:`` prefixes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ragdesk.embed import Embedder
from ragdesk.store import Store

if TYPE_CHECKING:
    from ragdesk.rerank import Reranker

RRF_K = 60
RERANK_POOL = 30

_FILTER_RE = re.compile(r"(?<![\w-])(folder|source):(\S+)")


@dataclass(frozen=True)
class Filters:
    path_like: str = ""
    source: str = ""

    def __bool__(self) -> bool:
        return bool(self.path_like or self.source)


def parse_filters(query: str) -> tuple[str, Filters]:
    """Pull ``folder:`` / ``source:`` out of a query; the rest is the real query."""
    folder = ""
    source = ""

    def take(match: re.Match[str]) -> str:
        nonlocal folder, source
        key, value = match.group(1), match.group(2)
        if key == "folder":
            folder = value
        else:
            source = value
        return ""

    cleaned = _FILTER_RE.sub(take, query)
    return re.sub(r"\s+", " ", cleaned).strip(), Filters(path_like=folder, source=source)


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
    parent_text: str = ""

    @property
    def context(self) -> str:
        """What the LLM should read: the parent section when one was stored."""
        return self.parent_text or self.text


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
    *,
    query_vec: list[float] | None = None,
    hyde_vec: list[float] | None = None,
    filters: Filters | None = None,
) -> list[Hit]:
    """BM25 + dense (+ HyDE, + path) lanes, RRF-fused, chunk-level."""
    query_vec = query_vec if query_vec is not None else embedder.embed_query(query)
    lane_rows: list[tuple[str, list[dict]]] = [
        ("bm25", store.bm25_search(query, pool, filters=filters)),
        ("dense", store.dense_search(query_vec, pool, filters=filters)),
        ("path", store.path_search(query, pool, filters=filters)),
    ]
    if hyde_vec is not None:
        lane_rows.append(("hyde", store.dense_search(hyde_vec, pool, filters=filters)))

    payloads: dict[int, dict] = {}
    for _lane, rows in lane_rows:
        for row in rows:
            payloads.setdefault(row["id"], row)
    dense_rows = next(rows for lane, rows in lane_rows if lane == "dense")
    cosine = {row["id"]: row["score"] for row in dense_rows}
    members = {lane: {row["id"] for row in rows} for lane, rows in lane_rows}
    fused = fuse([[row["id"] for row in rows] for _lane, rows in lane_rows])
    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]

    hits: list[Hit] = []
    for chunk_id, score in ranked:
        row = payloads[chunk_id]
        lanes = "+".join(lane for lane, ids in members.items() if chunk_id in ids)
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
                parent_text=str(row.get("parent_text") or ""),
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
    *,
    query_vec: list[float] | None = None,
    hyde_text: str = "",
    filters: Filters | None = None,
) -> list[Hit]:
    """Two-stage retrieval: fuse a larger candidate pool, optionally rerank,
    then cut to ``top_k``."""
    query_vec = query_vec if query_vec is not None else embedder.embed_query(query)
    hyde_vec = embedder.embed_query(hyde_text) if hyde_text else None
    if reranker is not None:
        candidates = hybrid_search(
            store,
            embedder,
            query,
            top_k=max(top_k, pool),
            query_vec=query_vec,
            hyde_vec=hyde_vec,
            filters=filters,
        )
        return reranker.rerank(query, candidates)[:top_k]
    return hybrid_search(
        store,
        embedder,
        query,
        top_k=top_k,
        query_vec=query_vec,
        hyde_vec=hyde_vec,
        filters=filters,
    )
