"""Minimal Ollama HTTP client (stdlib only)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

DEFAULT_HOST = "http://localhost:11434"


class OllamaUnavailable(RuntimeError):
    """Ollama is not reachable (not running, or model missing)."""


def post_json(host: str, path: str, payload: dict, *, timeout: float = 300.0) -> dict:
    url = f"{host.rstrip('/')}{path}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise OllamaUnavailable(
            f"cannot reach Ollama at {host}: {exc}. Is it running? "
            f"(ollama serve; ollama pull <model>)"
        ) from exc
