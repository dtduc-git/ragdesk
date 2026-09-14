"""Minimal Ollama HTTP client (stdlib only)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator

DEFAULT_HOST = "http://localhost:11434"


class OllamaUnavailable(RuntimeError):
    """Ollama is not reachable (not running, or model missing)."""


def _raise_unavailable(host: str, exc: BaseException) -> None:
    raise OllamaUnavailable(
        f"cannot reach Ollama at {host}: {exc}. Is it running? "
        f"(ollama serve; ollama pull <model>)"
    ) from exc


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
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:200]
        _raise_unavailable(host, f"HTTP {exc.code}: {detail!r}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        _raise_unavailable(host, exc)
    return {}  # unreachable; _raise_unavailable always raises


def post_stream(
    host: str, path: str, payload: dict, *, timeout: float = 300.0
) -> Iterator[dict]:
    """Stream newline-delimited JSON objects from Ollama (``stream=true``)."""
    url = f"{host.rstrip('/')}{path}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:200]
        _raise_unavailable(host, f"HTTP {exc.code}: {detail!r}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        _raise_unavailable(host, exc)
    with response:
        for raw_line in response:
            line = raw_line.strip()
            if line:
                yield json.loads(line)

