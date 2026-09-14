"""Reranking stage: pluggable rerankers over fused retrieval hits.

``LexicalReranker`` is a dependency-free baseline (shared-token overlap) used
by tests and as a fallback. ``FastEmbedReranker`` is the real cross-encoder
backend, behind the ``rerank`` optional extra (ONNX Runtime via fastembed).

Note: fastembed does not ship ``BAAI/bge-reranker-v2-m3`` yet (upstream issue
qdrant/fastembed#494); the default model here is ``BAAI/bge-reranker-base``
(MIT). Do not default to ``jinaai/jina-reranker-v2-base-multilingual`` — it is
CC-BY-NC-4.0.
"""

from __future__ import annotations

from typing import Protocol

from ragdesk.embed import tokenize
from ragdesk.search import Hit

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]: ...


class LexicalReranker:
    """Deterministic shared-token-overlap reranker (no dependencies).

    A weak but honest baseline: it can only reorder what retrieval already
    found. Real quality comes from the cross-encoder backend below.
    """

    name = "lexical"

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return list(hits)
        scored = [
            (len(query_tokens & set(tokenize(hit.text))), rank, hit)
            for rank, hit in enumerate(hits)
        ]
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        return [hit for _, _, hit in scored]


class FastEmbedReranker:
    """Cross-encoder reranker via fastembed (ONNX Runtime).

    The model downloads on first use (~0.08-1.1 GB depending on model).
    Requires the ``rerank`` extra: ``pip install 'ragdesk[rerank]'``.
    """

    def __init__(self, model: str = DEFAULT_RERANK_MODEL) -> None:
        self.name = f"fastembed:{model}"
        self.model = model
        self._encoder = None

    def _load(self):
        if self._encoder is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "fastembed is not installed. Install the rerank extra: "
                    "pip install 'ragdesk[rerank]'"
                ) from exc
            self._encoder = TextCrossEncoder(model_name=self.model)
        return self._encoder

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return []
        scores = list(self._load().rerank(query, [hit.text for hit in hits]))
        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
        return [hits[i] for i in order]


def get_reranker(spec: str):
    """Build a reranker from a spec: ``none | lexical | fastembed[:model]``."""
    spec = spec.strip()
    if spec in ("", "none"):
        return None
    if spec == "lexical":
        return LexicalReranker()
    if spec == "fastembed":
        return FastEmbedReranker()
    if spec.startswith("fastembed:"):
        return FastEmbedReranker(model=spec.split(":", 1)[1])
    raise ValueError(
        f"unknown reranker spec: {spec!r} (use none, lexical or fastembed[:model])"
    )
