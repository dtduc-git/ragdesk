"""Hybrid search: BM25 + dense + path lanes, fused with Reciprocal Rank Fusion.

Optional extra lanes: a HyDE vector (an LLM-written hypothetical answer) and
query scoping via ``folder:`` / ``source:`` prefixes.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ragdesk.embed import Embedder
from ragdesk.store import Store

if TYPE_CHECKING:
    from ragdesk.rerank import Reranker

RRF_K = 60
RERANK_POOL = 30
MAX_CHUNKS_PER_DOC = 2  # diversity: one long document must not fill every slot
RECENCY_WEIGHT = 0.10  # a mild nudge for fresh documents, not a re-rank
RECENCY_TAU_DAYS = 45.0
# Front-matter tags that move a document a little, never a lot: same intent as
# the recency nudge, and the same rule applies — measure before changing them.
META_BOOST = 1.05
META_PENALTY = 0.90
CANONICAL_VALUES = {"canonical", "authoritative", "official", "high", "true", "yes", "1"}
DRAFT_VALUES = {"draft", "deprecated", "archived", "superseded", "obsolete", "wip"}

# `folder:Technical` and, for paths with spaces, `folder:"/Users/me/My Docs"`
_FILTER_RE = re.compile(r'(?<![\w-])([a-z][a-z0-9_-]*):(?:"([^"]+)"|(\S+))')
_RESERVED_KEYS = {"http", "https", "file", "ragdesk", "ollama", "github", "web"}


@dataclass(frozen=True)
class Filters:
    path_like: str = ""
    source: str = ""
    meta: dict[str, str] | None = None

    def __bool__(self) -> bool:
        return bool(self.path_like or self.source or self.meta)


def parse_filters(query: str) -> tuple[str, Filters]:
    """Pull ``folder:`` / ``source:`` out of a query; the rest is the real query."""
    folder = ""
    source = ""
    meta: dict[str, str] = {}

    def take(match: re.Match[str]) -> str:
        nonlocal folder, source
        key = match.group(1)
        value = match.group(2) or match.group(3)
        if key in _RESERVED_KEYS:
            return match.group(0)  # a URL or a word with a colon, not a filter
        if key == "folder":
            folder = value
        elif key == "source":
            source = value
        else:
            meta[key] = value
        return ""

    cleaned = _FILTER_RE.sub(take, query)
    return (
        re.sub(r"\s+", " ", cleaned).strip(),
        Filters(path_like=folder, source=source, meta=meta or None),
    )


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
    line: int = 1
    metadata: dict | None = None

    @property
    def context(self) -> str:
        """What the LLM should read: the parent section when one was stored."""
        return self.parent_text or self.text


def hit_to_dict(hit: Hit) -> dict:
    """The one citation shape every writer stores (GUI and terminal share a db)."""
    return {
        "path": hit.path,
        "source": hit.source,
        "ordinal": hit.ordinal,
        "text": hit.text,
        "score": hit.score,
        "cosine": hit.cosine,
        "lanes": hit.lanes,
        "line": hit.line,
        "metadata": hit.metadata or {},
    }


def recency_factor(mtime: float, *, now: float | None = None) -> float:
    """Fresh documents get a small bonus; older ones settle back to 1.0."""
    if mtime <= 0:
        return 1.0
    age_days = max(0.0, ((now or time.time()) - mtime) / 86_400)
    return 1.0 + RECENCY_WEIGHT * math.exp(-age_days / RECENCY_TAU_DAYS)


def metadata_factor(metadata: dict | None) -> float:
    """Small rank nudge from front-matter tags: canonical docs up, drafts down.

    Opt-in by tag: a document without ``authority``/``status`` ranks exactly as
    it did before, so the boost can never quietly re-rank an untagged corpus.
    """
    if not metadata:
        return 1.0
    authority = str(metadata.get("authority", "")).strip().lower()
    status = str(metadata.get("status", "")).strip().lower()
    factor = 1.0
    if authority in CANONICAL_VALUES:
        factor *= META_BOOST
    if status in DRAFT_VALUES:
        factor *= META_PENALTY
    return factor


def _decode_metadata(raw: object) -> dict:
    """Rows reach the fusion from lanes that may or may not have decoded the JSON."""
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(str(raw or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def diversify(hits: list[Hit], top_k: int, max_per_doc: int = MAX_CHUNKS_PER_DOC) -> list[Hit]:
    """Cap how many chunks one document may contribute, then backfill if short."""
    picked: list[Hit] = []
    overflow: list[Hit] = []
    counts: dict[int, int] = {}
    for hit in hits:
        if counts.get(hit.doc_id, 0) < max_per_doc:
            picked.append(hit)
            counts[hit.doc_id] = counts.get(hit.doc_id, 0) + 1
        else:
            overflow.append(hit)
        if len(picked) >= top_k:
            break
    if len(picked) < top_k:
        picked.extend(overflow[: top_k - len(picked)])
    return picked[:top_k]


def fuse(
    rankings: list[list[int]],
    k: int = RRF_K,
    weights: list[float] | None = None,
) -> dict[int, float]:
    """Reciprocal Rank Fusion over chunk-id rankings, with optional lane weights."""
    scores: dict[int, float] = {}
    for lane, ranking in enumerate(rankings):
        weight = weights[lane] if weights and lane < len(weights) else 1.0
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight * (1.0 / (k + rank + 1))
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
    extra_queries: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> list[Hit]:
    """BM25 + dense (+ HyDE, + path, + sub-query) lanes, RRF-fused, chunk-level."""
    query_vec = query_vec if query_vec is not None else embedder.embed_query(query)
    lane_rows: list[tuple[str, list[dict]]] = [
        ("bm25", store.bm25_search(query, pool, filters=filters)),
        ("dense", store.dense_search(query_vec, pool, filters=filters)),
        ("path", store.path_search(query, pool, filters=filters)),
    ]
    if hyde_vec is not None:
        lane_rows.append(("hyde", store.dense_search(hyde_vec, pool, filters=filters)))
    for index, sub_query in enumerate(extra_queries or []):
        sub_vec = embedder.embed_query(sub_query)
        lane_rows.append((f"sub{index + 1}", store.dense_search(sub_vec, pool, filters=filters)))

    payloads: dict[int, dict] = {}
    for _lane, rows in lane_rows:
        for row in rows:
            row["metadata"] = _decode_metadata(row.get("metadata"))
            payloads.setdefault(row["id"], row)
    dense_rows = next(rows for lane, rows in lane_rows if lane == "dense")
    cosine = {row["id"]: row["score"] for row in dense_rows}
    members = {lane: {row["id"] for row in rows} for lane, rows in lane_rows}
    lane_weights = (
        [float(weights.get(lane, 1.0)) for lane, _rows in lane_rows]
        if weights
        else None
    )
    fused = fuse(
        [[row["id"] for row in rows] for _lane, rows in lane_rows],
        weights=lane_weights,
    )
    for chunk_id in list(fused):
        payload = payloads.get(chunk_id, {})
        fused[chunk_id] *= recency_factor(float(payload.get("mtime") or 0.0))
        fused[chunk_id] *= metadata_factor(payload.get("metadata"))
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
                line=int(row.get("line_start") or 1),
                metadata=row.get("metadata") or {},
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
    extra_queries: list[str] | None = None,
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
            extra_queries=extra_queries,
        )
        return diversify(reranker.rerank(query, candidates), top_k)
    hits = hybrid_search(
        store,
        embedder,
        query,
        top_k=top_k,
        query_vec=query_vec,
        hyde_vec=hyde_vec,
        filters=filters,
        extra_queries=extra_queries,
    )
    return diversify(hits, top_k)
