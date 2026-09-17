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
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ragdesk.ollama import DEFAULT_HOST, post_json, post_stream
from ragdesk.presets import PRESETS

DEFAULT_SPEC = "auto"


class LLMUnavailable(RuntimeError):
    """No usable LLM: Ollama lacks the model and the MLX extra is absent."""


# Hosts that mean "this machine": anything else gets a privacy warning in the UI,
# because the prompt contains the retrieved passages.
_LOCAL_HOSTNAMES = {"localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"}


def is_local_host(host: str) -> bool:
    """True when a configured endpoint stays on this machine ("" = not configured)."""
    raw = host.strip()
    if not raw:
        return True
    try:
        name = urllib.parse.urlsplit(raw).hostname
    except ValueError:
        return False
    if name is None:  # bare host:port without a scheme
        name = raw.split("/")[0].split(":")[0]
    name = name.lower()
    return name in _LOCAL_HOSTNAMES or name.startswith("127.")


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


def _hf_cache_dir() -> str:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE  # noqa: PLC0415

        return str(HF_HUB_CACHE)
    except Exception:  # hub not installed (or moved the constant)
        return str(Path.home() / ".cache" / "huggingface" / "hub")


def mlx_model_cached(repo: str, cache_dir: str | None = None) -> bool:
    """True when a snapshot with weights is already in the shared HF cache."""
    if not repo:
        return False
    folder = Path(cache_dir or _hf_cache_dir()) / f"models--{repo.replace('/', '--')}"
    snapshots = folder / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(snapshots.glob("*/*.safetensors"))


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


class OpenAICompatLLM:
    """Any OpenAI-compatible /chat/completions endpoint: LM Studio, llama.cpp,
    vLLM, Ollama's own /v1 shim, or the real OpenAI."""

    kind = "openai"

    def __init__(self, model: str, host: str, api_key: str = "") -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.api_key = api_key

    def _request(self, payload: dict) -> urllib.request.Request:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return urllib.request.Request(
            f"{self.host}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
        )

    def generate(self, prompt: str, options: dict[str, Any]) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(options.get("temperature", 0.2)),
            "max_tokens": int(options.get("num_predict", 400)),
            "stream": False,
        }
        try:
            with urllib.request.urlopen(self._request(payload), timeout=300) as response:
                data = json.loads(response.read())
        except (urllib.error.URLError, OSError) as exc:
            raise LLMUnavailable(
                f"cannot reach the OpenAI-compatible endpoint at {self.host} ({exc}); "
                "check the host in Settings"
            ) from exc
        try:
            return str(data["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailable(f"unexpected reply from {self.host}: {data}") from exc

    def generate_stream(self, prompt: str, options: dict[str, Any]) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(options.get("temperature", 0.2)),
            "max_tokens": int(options.get("num_predict", 400)),
            "stream": True,
        }
        try:
            with urllib.request.urlopen(self._request(payload), timeout=300) as response:
                for raw in response:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        event = json.loads(chunk)
                        piece = event["choices"][0]["delta"].get("content") or ""
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                        continue
                    if piece:
                        yield str(piece)
        except (urllib.error.URLError, OSError) as exc:
            raise LLMUnavailable(
                f"cannot reach the OpenAI-compatible endpoint at {self.host} ({exc})"
            ) from exc


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


def openai_config() -> dict[str, str]:
    """Endpoint settings the wizard or Settings tab saved (key stays in credentials)."""
    from ragdesk import credentials, settings  # noqa: PLC0415 - avoid a cycle

    values = settings.load()
    return {
        "host": str(values.get("openai_host") or ""),
        "model": str(values.get("openai_model") or ""),
        "api_key": str(credentials.get("openai").get("api_key") or ""),
    }


def resolve_llm(
    spec: str | None = None,
    *,
    preset: str = "light",
    host: str = "",
    preference: str = "",
) -> Any:
    """Pick a backend without downloading anything twice.

    ``auto`` ladder: a running Ollama that already has the preset model wins
    (zero download — never re-fetch 2.5GB the user already has); then a
    configured OpenAI-compatible endpoint; otherwise MLX in-process, whose
    weights share the Hugging Face cache with the embedder.
    """
    selected = PRESETS.get(preset, PRESETS["light"])
    tag = selected["llm"]
    repo = selected.get("llm_mlx", "")
    ollama_host = host or DEFAULT_HOST
    spec = (spec or os.environ.get("RAGDESK_LLM") or DEFAULT_SPEC).strip() or DEFAULT_SPEC
    if not preference:
        from ragdesk import settings  # noqa: PLC0415 - avoid a cycle

        preference = str(settings.load().get("llm_preference") or "")
    kind, _, rest = spec.partition(":")

    if kind == "ollama":
        return OllamaLLM(rest or tag, ollama_host)
    if kind == "mlx":
        chosen = rest or repo
        if not chosen:
            raise LLMUnavailable(f"no MLX model for preset {preset!r}; use mlx:<hf-repo>")
        return MlxLLM(chosen)
    if kind == "openai":
        config = openai_config()
        model = rest or config["model"]
        if not model or not config["host"]:
            raise LLMUnavailable(
                "no OpenAI-compatible endpoint configured: set host + model in Settings"
            )
        return OpenAICompatLLM(model, config["host"], config["api_key"])
    if spec != DEFAULT_SPEC:
        raise LLMUnavailable(
            f"unknown llm spec: {spec!r} "
            "(use auto, ollama[:model], mlx[:hf-repo] or openai[:model])"
        )

    # spec == auto: an explicit preference from the wizard/Settings wins first.
    config = openai_config()
    if preference == "openai" and config["host"] and config["model"]:
        return OpenAICompatLLM(config["model"], config["host"], config["api_key"])
    if preference == "mlx" and repo and mlx_available():
        return MlxLLM(repo)
    if preference == "ollama" and ollama_has_model(tag, ollama_host):
        return OllamaLLM(tag, ollama_host)

    if ollama_has_model(tag, ollama_host):
        return OllamaLLM(tag, ollama_host)
    if config["host"] and config["model"]:
        return OpenAICompatLLM(config["model"], config["host"], config["api_key"])
    if repo and mlx_available():
        return MlxLLM(repo)
    raise LLMUnavailable(
        "no LLM backend: start Ollama and pull the model "
        f"(ollama serve; ollama pull {tag}), install the MLX extra on Apple Silicon "
        "(uv tool install 'ragdesk[mlx]'), or point Settings at an OpenAI-compatible "
        "endpoint (LM Studio, llama.cpp)"
    )


def llm_status(
    spec: str | None = None, *, preset: str = "light", host: str = "", preference: str = ""
) -> dict:
    """Cheap snapshot for ``/api/status`` — never loads weights."""
    try:
        llm = resolve_llm(spec, preset=preset, host=host, preference=preference)
    except LLMUnavailable as exc:
        return {"kind": "none", "model": "", "note": str(exc)}
    if llm.kind == "mlx":
        note = "in-process; weights from the Hugging Face cache"
    elif llm.kind == "openai":
        note = f"OpenAI-compatible endpoint at {llm.host}"
    else:
        note = f"reused running Ollama at {llm.host}"
    return {"kind": llm.kind, "model": llm.model, "note": note}
