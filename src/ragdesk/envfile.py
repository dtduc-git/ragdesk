"""Tiny ``.env`` reader (no dependency).

Loads ``KEY=VALUE`` lines into ``os.environ`` without overriding values that
are already set. Used by the CLI/serve entrypoint for local development;
release artifacts get their values baked at build time (see
``ragdesk.buildenv``).
"""

from __future__ import annotations

import os
from pathlib import Path


def parse_env_file(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _candidates(path: Path | None) -> list[Path]:
    if path is not None:
        return [path]
    explicit = os.environ.get("RAGDESK_ENV_FILE")
    if explicit:
        return [Path(explicit)]
    # repo checkout (src/ragdesk/envfile.py -> repo root), then cwd
    repo_root = Path(__file__).resolve().parents[2]
    return [
        Path.cwd() / ".env",
        repo_root / ".env",
        Path.home() / ".config" / "ragdesk" / ".env",
    ]


def load_env_file(path: Path | None = None) -> int:
    """Load the first existing candidate ``.env``; existing env vars win.

    Returns the number of variables that were set.
    """
    for candidate in _candidates(path):
        if not candidate.is_file():
            continue
        values = parse_env_file(candidate.read_text())
        loaded = 0
        for key, value in values.items():
            if value and key not in os.environ:
                os.environ[key] = value
                loaded += 1
        return loaded
    return 0
