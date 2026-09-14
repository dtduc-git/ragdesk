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


def get_embedder(spec: str) -> Embedder:
    """Build an embedder from a spec: ``ollama[:model]`` or ``hash[:dim]``."""
    if spec == "hash" or spec.startswith("hash:"):
        dim = int(spec.split(":", 1)[1]) if ":" in spec else 4096
        return HashingEmbedder(dim)
    if spec == "ollama":
        return OllamaEmbedder()
    if spec.startswith("ollama:"):
        return OllamaEmbedder(model=spec.split(":", 1)[1])
    raise ValueError(f"unknown embedder spec: {spec!r} (use ollama[:model] or hash[:dim])")
