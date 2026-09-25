"""Embedding backends.

The core is stdlib-only: real embedding models are reached through a local
Ollama server (default ``embeddinggemma:300m``). ``HashingEmbedder`` is a
deterministic offline stand-in used by tests and CI — never for quality
claims.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Protocol

from ragdesk.ollama import DEFAULT_HOST, post_json

DEFAULT_OLLAMA_MODEL = "embeddinggemma:300m"
DEFAULT_ONNX_REPO = "onnx-community/embeddinggemma-300m-ONNX"
ONNX_QUERY_PROMPT = "task: search result | query: "
ONNX_DOC_PROMPT = "title: none | text: "
ONNX_MAX_TOKENS = 2048  # what a chunk is truncated to when a model allows more


def models_dir() -> Path:
    """Where materialized ONNX files live (real files, not cache symlinks)."""
    return Path.home() / ".ragdesk" / "models"


def _materialize(source: Path, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    real = source.resolve()
    try:
        os.link(real, dest)  # hardlink: same inode, no extra disk
        return
    except OSError:  # different filesystem (or no hardlink support): copy
        pass
    temp = dest.with_name(dest.name + ".tmp")
    shutil.copy2(real, temp)  # a crash must not leave a truncated model in place
    os.replace(temp, dest)


def local_model_file(repo: str, filename: str, subfolder: str = "") -> Path:
    """A real ONNX file sitting next to its external data.

    onnxruntime validates that a model's external-data file lives inside the
    model's own directory. The HuggingFace cache symlinks the model and its
    ``.onnx_data`` into *different* blob folders (sharded caches put each blob
    in its own subdirectory), which onnxruntime rejects with "External data
    path escapes model directory" — a fresh install cannot index anything.
    Hardlink the pair into one real directory under ``~/.ragdesk/models``,
    keyed by the model blob's hash so a re-published model is picked up.
    """
    from huggingface_hub import hf_hub_download  # noqa: PLC0415 - optional extra

    source = Path(hf_hub_download(repo, filename, subfolder=subfolder or None))
    if not source.is_symlink():
        return source
    revision = source.resolve().name  # HF blob name: the file's content hash
    dest = models_dir() / repo.replace("/", "--") / revision / subfolder / filename
    _materialize(source, dest)
    try:
        data = Path(hf_hub_download(repo, filename + "_data", subfolder=subfolder or None))
    except Exception:  # noqa: BLE001 - most models keep everything in one file
        return dest
    _materialize(data, dest.with_name(dest.name + "_data"))
    return dest


# Each family expects its own prefixes, and the wrong ones quietly cost recall.
_ONNX_PROMPTS: tuple[tuple[str, str, str], ...] = (
    ("embeddinggemma", ONNX_QUERY_PROMPT, ONNX_DOC_PROMPT),
    ("multilingual-e5", "query: ", "passage: "),
    ("multilingual-gte", "", ""),
    ("bge-m3", "", ""),
)


def position_limit(config: dict) -> int:
    """Tokenizer truncation cap: what the model's position table can hold.

    BERT-family exports learned positions for 512 tokens; feeding them a longer
    sequence is a hard failure, not a quiet truncation.
    """
    try:
        raw = int(config.get("max_position_embeddings") or ONNX_MAX_TOKENS)
    except (TypeError, ValueError):
        return ONNX_MAX_TOKENS
    return max(64, min(ONNX_MAX_TOKENS, raw))


def onnx_prompts(repo: str) -> tuple[str, str]:
    """The (query, document) prefixes a model was trained with — "" when none.

    An unrecognised repo gets no prefixes: guessing another family's prompts
    silently costs recall, and a wrong prefix is worse than none.
    """
    lowered = repo.lower()
    for marker, query, doc in _ONNX_PROMPTS:
        if marker in lowered:
            return query, doc
    return "", ""


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


_THREAD_OVERRIDE: int | None = None


def set_thread_override(threads: int | None) -> None:
    """Process-wide override (``--embed-threads``), beats the saved setting."""
    global _THREAD_OVERRIDE  # noqa: PLW0603 - a one-shot CLI knob
    _THREAD_OVERRIDE = threads


def configured_threads() -> int:
    """ONNX thread cap: 0 = onnxruntime's default (every core).

    Quiet indexing caps the intra-op pool so a long index run leaves the
    machine usable; the setting is read when a session is built, so the app
    unloads the embedder when the toggle changes.
    """
    if _THREAD_OVERRIDE is not None:
        return _THREAD_OVERRIDE
    from ragdesk import settings as app_settings  # noqa: PLC0415 - optional knob

    try:
        return max(0, int(app_settings.load().get("embed_threads") or 0))
    except (TypeError, ValueError):
        return 0


def session_options(threads: int = 0):
    """ONNX Runtime session options; only touch the pools when capping."""
    import onnxruntime as ort  # noqa: PLC0415 - optional extra

    options = ort.SessionOptions()
    if threads > 0:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
    return options


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

    # Model-free backends have nothing to release; the protocol stays uniform.
    loaded = False

    def unload(self) -> None:
        return None


class OllamaEmbedder:
    """Real embedder via a local Ollama server."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        host: str = DEFAULT_HOST,
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

    loaded = False  # the model lives in the Ollama daemon, not here

    def unload(self) -> None:
        return None


def require_onnx() -> None:
    """Fail fast with the fix when the ``onnx`` extra is absent.

    Called from the constructors: otherwise a plain `pip install ragdesk` gets
    a traceback deep inside the first search (or a per-file skip during index),
    instead of one line telling the user which extra to install.
    """
    try:
        import huggingface_hub  # noqa: F401
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "onnx dependencies missing. Install the onnx extra: pip install 'ragdesk[onnx]'"
        ) from exc


class OnnxEmbedder:
    """EmbeddingGemma-300M via ONNX Runtime (int8 by default), CPU-only.

    Downloads the tokenizer + model from Hugging Face on first use
    (~0.3 GB). Requires the ``onnx`` extra. Uses the model's recommended
    query/document prompts, which matters for retrieval quality.
    """

    def __init__(self, repo: str = DEFAULT_ONNX_REPO, variant: str = "quantized") -> None:
        require_onnx()
        self.name = f"onnx:{repo}:{variant}"
        self.repo = repo
        self.variant = variant
        self._query_prompt, self._doc_prompt = onnx_prompts(repo)
        self._tokenizer = None
        self._session = None
        self._output_index = 0
        self._pool = False
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        """Read off the model itself — exports differ (384 / 768 / 1024)."""
        self._load()
        if self._dim is None:  # symbolic axis in the graph: ask the model
            self._dim = len(self._embed_prefixed(["dimension probe"], "")[0])
        return self._dim

    def _load(self) -> None:
        if self._session is not None:
            return
        require_onnx()
        import numpy as np  # noqa: F401
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(hf_hub_download(self.repo, "tokenizer.json"))
        tokenizer.enable_truncation(max_length=self._max_tokens())
        tokenizer.enable_padding()

        model_path = local_model_file(self.repo, f"model_{self.variant}.onnx", "onnx")

        self._tokenizer = tokenizer
        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=session_options(configured_threads()),
        )
        names = [output.name for output in self._session.get_outputs()]
        if "sentence_embedding" in names:
            self._output_index = names.index("sentence_embedding")
        else:
            self._output_index = self._find_token_output()
            self._pool = True
        hidden = self._session.get_outputs()[self._output_index].shape[-1]
        self._dim = int(hidden) if isinstance(hidden, int) and hidden > 0 else None

    def _max_tokens(self) -> int:
        try:
            from huggingface_hub import hf_hub_download  # noqa: PLC0415 - optional extra

            config = json.loads(Path(hf_hub_download(self.repo, "config.json")).read_text())
        except Exception:  # noqa: BLE001 - a missing config just means the default
            return ONNX_MAX_TOKENS
        return position_limit(config)

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def unload(self) -> None:
        """Drop the session so idle RAM goes back to ~nothing; reloads lazily.

        Measured: the session's workspace floor (~1.1GB on the int8 300M model)
        does not respond to arena/batch/sequence tuning, so releasing the whole
        session is the only honest way to give memory back.
        """
        self._session = None
        self._tokenizer = None

    def _find_token_output(self) -> int:
        """Fall back to a per-token output (batch, tokens, dim) for mean pooling."""
        outputs = self._session.get_outputs()
        for index, output in enumerate(outputs):
            if len(output.shape) == 3:
                return index
        names = [output.name for output in outputs]
        raise RuntimeError(f"ONNX model exposes no usable embedding output (outputs: {names})")

    def _embed_prefixed(self, texts: list[str], prefix: str) -> list[list[float]]:
        self._load()
        import numpy as np

        encodings = self._tokenizer.encode_batch([prefix + text for text in texts])
        input_ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask for enc in encodings], dtype=np.int64)
        # Feed exactly what the export declares: BERT-family conversions also
        # want token_type_ids, some text-embedding exports want input_ids only.
        feed: dict[str, np.ndarray] = {}
        for model_input in self._session.get_inputs():
            if model_input.name == "input_ids":
                feed["input_ids"] = input_ids
            elif model_input.name == "attention_mask":
                feed["attention_mask"] = attention_mask
            elif model_input.name == "token_type_ids":
                feed["token_type_ids"] = np.zeros_like(input_ids)
        outputs = self._session.run(None, feed)
        array = outputs[self._output_index]
        if self._pool:
            # Mean-pool over real tokens only, then normalise below — the
            # sentence-transformers recipe for exports without a pooled output.
            mask = attention_mask[:, :, None].astype("float32")
            array = (array * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1.0, None)
        vectors: list[list[float]] = []
        for vec in array.tolist():
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embed_prefixed(texts, self._doc_prompt)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_prefixed([text], self._query_prompt)[0]


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
