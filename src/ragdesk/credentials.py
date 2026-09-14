"""Local credential store: ``~/.config/ragdesk/credentials.json`` (0600).

Only used to remember connections the user explicitly set up (GitHub token,
Confluence site credentials, Google OAuth client). Never leaves the machine.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def credentials_file() -> Path:
    base = os.environ.get("RAGDESK_CONFIG_DIR")
    if base:
        return Path(base) / "credentials.json"
    return Path.home() / ".config" / "ragdesk" / "credentials.json"


def load(path: Path | None = None) -> dict:
    target = path or credentials_file()
    if not target.exists():
        return {}
    try:
        data = json.loads(target.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save(data: dict, path: Path | None = None) -> None:
    target = path or credentials_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2))
    target.chmod(0o600)


def get(provider: str, path: Path | None = None) -> dict:
    entry = load(path).get(provider, {})
    return entry if isinstance(entry, dict) else {}


def set_provider(provider: str, values: dict, path: Path | None = None) -> None:
    data = load(path)
    merged = dict(data.get(provider, {}))
    merged.update(values)
    data[provider] = merged
    save(data, path)


def clear(provider: str, path: Path | None = None) -> None:
    data = load(path)
    if provider in data:
        del data[provider]
        save(data, path)
