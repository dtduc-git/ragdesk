"""Embedding backends.

The core is stdlib-only: real embedding models are reached through a local
Ollama server (default ``embeddinggemma:300m``). ``HashingEmbedder`` is a
deterministic offline stand-in used by tests and CI — never for quality
claims.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

from ragdesk.ollama import post_json

DEFAULT_OLLAMA_MODEL = "embeddinggemma:300m"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_ONNX_REPO = "onnx-community/embeddinggemma-300m-ONNX"
ONNX_QUERY_PROMPT = "task: search result | query: "
ONNX_DOC_PROMPT = "title: none | text: "

# Toy-embedder stopwords only: without this, stopword overlap between unrelated
# documents dominates the hashing vector (and signed hashing can cancel to 0).
_STOPWORDS = frozenset(
    """a an and are as at be but by do does for from how i in is it of on or
    that the this to was were what when where which who will with you your""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, strip edge punctuation, drop stopwords.

    Shared by the hashing embedder and the lexical reranker (toy/fallback
    paths only — real models do their own tokenization).
    """
    tokens: list[str] = []
    for token in text.lower().split():
        token = token.strip(".,;:!?()[]{}`'\"#*_-")
        if token and token not in _STOPWORDS:
            tokens.append(token)
    return tokens


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Deterministic bag-of-tokens hashing embedder (tests/CI only).

    Cheap, dependency-free, reproducible — useful for exercising the whole
    pipeline offline. Dim is large enough that hash collisions stay rare
    (collisions are what corrupt a bag-of-words cosine). Retrieval quality
    numbers must come from a real model.
    """

    def __init__(self, dim: int = 4096) -> None:
        self.name = f"hash-{dim}"
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in tokenize(text):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                vec[index] += 1.0 if digest[4] % 2 == 0 else -1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class OllamaEmbedder:
    """Real embedder via a local Ollama server."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        dim: int = 768,
    ) -> None:
        self.name = f"ollama:{model}"
        self.model = model
        self.host = host
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = post_json(self.host, "/api/embed", {"model": self.model, "input": texts})
        embeddings = data["embeddings"]
        if embeddings:
            self.dim = len(embeddings[0])
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class OnnxEmbedder:
    """EmbeddingGemma-300M via ONNX Runtime (int8 by default), CPU-only.

    Downloads the tokenizer + model from Hugging Face on first use
    (~0.3 GB). Requires the ``onnx`` extra. Uses the model's recommended
    query/document prompts, which matters for retrieval quality.
    """

    def __init__(self, repo: str = DEFAULT_ONNX_REPO, variant: str = "quantized") -> None:
        self.name = f"onnx:{repo}:{variant}"
        self.repo = repo
        self.variant = variant
        self.dim = 768
        self._tokenizer = None
        self._session = None
        self._output_index = 0

    def _load(self) -> None:
        if self._session is not None:
            return
        try:
            import numpy as np  # noqa: F401
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "onnx embedder dependencies missing. Install the onnx extra: "
                "pip install 'ragdesk[onnx]'"
            ) from exc

        tokenizer = Tokenizer.from_file(hf_hub_download(self.repo, "tokenizer.json"))
        tokenizer.enable_truncation(max_length=2048)
        tokenizer.enable_padding()

        model_path = hf_hub_download(
            self.repo, f"model_{self.variant}.onnx", subfolder="onnx"
        )
        try:
            hf_hub_download(
                self.repo, f"model_{self.variant}.onnx_data", subfolder="onnx"
            )
        except Exception:  # noqa: BLE001 - external data file exists for some variants only
            pass

        self._tokenizer = tokenizer
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._output_index = self._find_sentence_embedding()

    def _find_sentence_embedding(self) -> int:
        outputs = self._session.get_outputs()
        names = [output.name for output in outputs]
        if "sentence_embedding" in names:
            return names.index("sentence_embedding")
        for index, output in enumerate(outputs):
            if len(output.shape) == 2:
                return index
        raise RuntimeError(
            f"ONNX model exposes no sentence embedding output (outputs: {names})"
        )

    def _embed_prefixed(self, texts: list[str], prefix: str) -> list[list[float]]:
        self._load()
        import numpy as np

        encodings = self._tokenizer.encode_batch([prefix + text for text in texts])
        input_ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask for enc in encodings], dtype=np.int64)
        outputs = self._session.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        vectors: list[list[float]] = []
        for vec in outputs[self._output_index].tolist():
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embed_prefixed(texts, ONNX_DOC_PROMPT)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_prefixed([text], ONNX_QUERY_PROMPT)[0]


def get_embedder(spec: str) -> Embedder:
    """Build an embedder from a spec: ``ollama[:model]``, ``onnx[:repo]`` or ``hash[:dim]``."""
    if spec == "hash" or spec.startswith("hash:"):
        dim = int(spec.split(":", 1)[1]) if ":" in spec else 4096
        return HashingEmbedder(dim)
    if spec == "ollama":
        return OllamaEmbedder()
    if spec.startswith("ollama:"):
        return OllamaEmbedder(model=spec.split(":", 1)[1])
    if spec == "onnx":
        return OnnxEmbedder()
    if spec.startswith("onnx:"):
        return OnnxEmbedder(repo=spec.split(":", 1)[1])
    raise ValueError(
        f"unknown embedder spec: {spec!r} (use ollama[:model], onnx[:repo] or hash[:dim])"
    )
