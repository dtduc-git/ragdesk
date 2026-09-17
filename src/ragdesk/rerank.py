"""Reranking stage: pluggable rerankers over fused retrieval hits.

``LexicalReranker`` is a dependency-free baseline (shared-token overlap) used
by tests and as a fallback. ``FastEmbedReranker`` is the fast English-first
cross-encoder backend, and ``OnnxReranker`` is the multilingual one
(``gte-multilingual-reranker-base``, Apache-2.0, 70+ languages) — both behind
the ``onnx`` extra.

Note: fastembed does not ship ``BAAI/bge-reranker-v2-m3`` yet (upstream issue
qdrant/fastembed#494); the default FastEmbed model here is
``BAAI/bge-reranker-base`` (MIT). Do not default to
``jinaai/jina-reranker-v2-base-multilingual`` — it is CC-BY-NC-4.0.
"""

from __future__ import annotations

import platform
from typing import Protocol

from ragdesk.embed import configured_threads, session_options, tokenize
from ragdesk.search import Hit

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"
DEFAULT_ONNX_RERANK_REPO = "onnx-community/gte-multilingual-reranker-base"
# Pairs per forward pass. All 50 at once padded to 512 tokens balloons the ONNX
# arena: measured peak RSS 6.2GB vs 3.4GB at 8 — and 13% slower, all padding.
RERANK_BATCH = 8


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
    Requires the ``onnx`` extra: ``pip install 'ragdesk[onnx]'``.
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
                    "fastembed is not installed. Install the onnx extra: "
                    "pip install 'ragdesk[onnx]'"
                ) from exc
            self._encoder = TextCrossEncoder(model_name=self.model)
        return self._encoder

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return []
        scores = list(self._load().rerank(query, [hit.text for hit in hits]))
        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
        return [hits[i] for i in order]


class OnnxReranker:
    """Multilingual cross-encoder reranker via ONNX Runtime.

    Default: ``gte-multilingual-reranker-base`` (Apache-2.0, 306M, 70+
    languages, 8K context). Downloads on first use (~0.3 GB int8). Requires
    the ``onnx`` extra.
    """

    def __init__(self, repo: str = DEFAULT_ONNX_RERANK_REPO, *, max_length: int = 512) -> None:
        self.name = f"onnx:{repo}"
        self.repo = repo
        self.max_length = max_length
        self._tokenizer = None
        self._session = None
        self._input_names: list[str] = []

    def _load(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "onnx reranker dependencies missing. Install the onnx extra: "
                "pip install 'ragdesk[onnx]'"
            ) from exc

        tokenizer = Tokenizer.from_file(hf_hub_download(self.repo, "tokenizer.json"))
        tokenizer.enable_truncation(max_length=self.max_length)
        tokenizer.enable_padding()

        variants = ["onnx/model_quantized.onnx"]
        if platform.machine() in ("arm64", "aarch64"):
            # some exports ship a platform-specific int8 build instead
            variants.insert(0, "onnx/model_qint8_arm64.onnx")
        variants.append("onnx/model.onnx")
        model_path = None
        for candidate in variants:
            try:
                model_path = hf_hub_download(self.repo, candidate)
                break
            except Exception:  # noqa: BLE001 - try the next variant
                continue
        if model_path is None:
            raise RuntimeError(f"no ONNX weights found in {self.repo}")

        self._tokenizer = tokenizer
        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=session_options(configured_threads()),
        )
        self._input_names = [item.name for item in self._session.get_inputs()]

    def _score(self, query: str, documents: list[str]) -> list[float]:
        """Batched so the arena stays flat; order is preserved batch by batch."""
        scores: list[float] = []
        for start in range(0, len(documents), RERANK_BATCH):
            scores.extend(self._score_batch(query, documents[start : start + RERANK_BATCH]))
        return scores

    def _score_batch(self, query: str, documents: list[str]) -> list[float]:
        import numpy as np

        self._load()
        encodings = self._tokenizer.encode_batch([(query, doc) for doc in documents])
        feed = {
            "input_ids": np.array([enc.ids for enc in encodings], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask for enc in encodings], dtype=np.int64),
        }
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.array([enc.type_ids for enc in encodings], dtype=np.int64)
        outputs = self._session.run(
            None, {key: value for key, value in feed.items() if key in self._input_names}
        )
        names = [output.name for output in self._session.get_outputs()]
        index = names.index("logits") if "logits" in names else 0
        return [float(row[0]) for row in outputs[index].tolist()]

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def unload(self) -> None:
        """Drop the session so idle RAM goes back to ~nothing; reloads lazily."""
        self._session = None
        self._tokenizer = None

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return []
        scores = self._score(query, [hit.text for hit in hits])
        order = sorted(range(len(hits)), key=lambda index: scores[index], reverse=True)
        return [hits[index] for index in order]


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
    if spec == "onnx":
        return OnnxReranker()
    if spec.startswith("onnx:"):
        return OnnxReranker(repo=spec.split(":", 1)[1])
    raise ValueError(
        f"unknown reranker spec: {spec!r} (use none, lexical, fastembed[:model] or onnx[:repo])"
    )
