"""LLM backends: reuse a running Ollama when the model is already there,
otherwise run MLX in-process (Apple Silicon; weights come from the shared
Hugging Face cache).

Spec syntax (``--llm`` / ``RAGDESK_LLM``):
``auto`` (default), ``ollama[:model]`` or ``mlx[:hf-repo]``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

from ragdesk.ollama import DEFAULT_HOST, post_json, post_stream
from ragdesk.presets import PRESETS

DEFAULT_SPEC = "auto"


class LLMUnavailable(RuntimeError):
    """No usable LLM: Ollama lacks the model and the MLX extra is absent."""


def _get_json(host: str, path: str, timeout: float = 1.5) -> dict:
    request = urllib.request.Request(f"{host.rstrip('/')}{path}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read())
    return data if isinstance(data, dict) else {}


def ollama_models(host: str = DEFAULT_HOST) -> set[str]:
    """Model tags on a running Ollama; empty when it is not reachable."""
    try:
        data = _get_json(host, "/api/tags")
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return set()
    entries = data.get("models", [])
    names = set()
    for entry in entries if isinstance(entries, list) else []:
        if isinstance(entry, dict) and entry.get("name"):
            names.add(str(entry["name"]))
    return names


def ollama_has_model(tag: str, host: str = DEFAULT_HOST) -> bool:
    names = ollama_models(host)
    return tag in names or (":" not in tag and f"{tag}:latest" in names)


def mlx_available() -> bool:
    return importlib.util.find_spec("mlx_lm") is not None


class OllamaLLM:
    """Talks to a running Ollama daemon — an existing install is reused as-is."""

    kind = "ollama"

    def __init__(self, model: str, host: str = DEFAULT_HOST) -> None:
        self.model = model
        self.host = host

    def generate(self, prompt: str, options: dict[str, Any]) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            # Grounded QA wants the fast path: thinking models otherwise burn
            # a hidden chain-of-thought before the (short) cited answer.
            "think": False,
            "options": dict(options),
        }
        data = post_json(self.host, "/api/generate", payload)
        return str(data.get("response", ""))

    def generate_stream(self, prompt: str, options: dict[str, Any]) -> Iterator[str]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "think": False,
            "options": dict(options),
        }
        for event in post_stream(self.host, "/api/generate", payload):
            piece = str(event.get("response", ""))
            if piece:
                yield piece


class MlxLLM:
    """Runs an MLX model in-process; nothing else to install or keep running."""

    kind = "mlx"

    def __init__(self, repo: str) -> None:
        self.model = repo
        self.host = "in-process"
        self._loaded: tuple[Any, Any] | None = None

    def _ensure(self) -> tuple[Any, Any]:
        if self._loaded is None:
            from mlx_lm import load  # noqa: PLC0415 - optional extra, Apple-only

            self._loaded = load(self.model)
        return self._loaded

    def _prompt(self, text: str) -> str:
        _, tokenizer = self._ensure()
        messages = [{"role": "user", "content": text}]
        # Thinking models burn the answer budget on hidden reasoning; templates
        # that do not know the switch simply ignore or reject it.
        for kwargs in ({"enable_thinking": False}, {}):
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **kwargs
                )
            except Exception:  # template kwargs vary per model
                continue
        return text

    def generate(self, prompt: str, options: dict[str, Any]) -> str:
        from mlx_lm import generate  # noqa: PLC0415

        model, tokenizer = self._ensure()
        base = {
            "model": model,
            "tokenizer": tokenizer,
            "prompt": self._prompt(prompt),
            "max_tokens": int(options.get("num_predict", 400)),
            "verbose": False,
        }
        temp = float(options.get("temperature", 0.2))
        try:
            from mlx_lm.sample_utils import make_sampler  # noqa: PLC0415
        except ImportError:
            return str(generate(**base, temp=temp))
        return str(generate(**base, sampler=make_sampler(temp=temp)))

    def generate_stream(self, prompt: str, options: dict[str, Any]) -> Iterator[str]:
        from mlx_lm import stream_generate  # noqa: PLC0415

        model, tokenizer = self._ensure()
        for response in stream_generate(
            model,
            tokenizer,
            prompt=self._prompt(prompt),
            max_tokens=int(options.get("num_predict", 400)),
        ):
            piece = getattr(response, "text", None)
            if piece is None:
                piece = str(response)
            if piece:
                yield piece


def resolve_llm(spec: str | None = None, *, preset: str = "light", host: str = "") -> Any:
    """Pick a backend without downloading anything twice.

    ``auto`` ladder: a running Ollama that already has the preset model wins
    (zero download — never re-fetch 2.5GB the user already has); otherwise MLX
    in-process, whose weights share the Hugging Face cache with the embedder.
    """
    selected = PRESETS.get(preset, PRESETS["light"])
    tag = selected["llm"]
    repo = selected.get("llm_mlx", "")
    ollama_host = host or DEFAULT_HOST
    spec = (spec or os.environ.get("RAGDESK_LLM") or DEFAULT_SPEC).strip() or DEFAULT_SPEC

    if spec == DEFAULT_SPEC:
        if ollama_has_model(tag, ollama_host):
            return OllamaLLM(tag, ollama_host)
        if repo and mlx_available():
            return MlxLLM(repo)
        raise LLMUnavailable(
            "no LLM backend: start Ollama and pull the model "
            f"(ollama serve; ollama pull {tag}), or install the MLX extra on "
            "Apple Silicon (uv tool install 'ragdesk[mlx]')"
        )

    kind, _, rest = spec.partition(":")
    if kind == "ollama":
        return OllamaLLM(rest or tag, ollama_host)
    if kind == "mlx":
        chosen = rest or repo
        if not chosen:
            raise LLMUnavailable(f"no MLX model for preset {preset!r}; use mlx:<hf-repo>")
        return MlxLLM(chosen)
    raise LLMUnavailable(
        f"unknown llm spec: {spec!r} (use auto, ollama[:model] or mlx[:hf-repo])"
    )


def llm_status(spec: str | None = None, *, preset: str = "light", host: str = "") -> dict:
    """Cheap snapshot for ``/api/status`` — never loads weights."""
    try:
        llm = resolve_llm(spec, preset=preset, host=host)
    except LLMUnavailable as exc:
        return {"kind": "none", "model": "", "note": str(exc)}
    if llm.kind == "mlx":
        note = "in-process; weights from the Hugging Face cache"
    else:
        note = f"reused running Ollama at {llm.host}"
    return {"kind": llm.kind, "model": llm.model, "note": note}
