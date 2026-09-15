"""App settings: ``~/.config/ragdesk/settings.json`` (0600). Not credentials."""

from __future__ import annotations

import json
from pathlib import Path

from ragdesk.credentials import credentials_file

DEFAULTS: dict = {
    "auto_index_hours": 1,
    "auto_index_last": "",
    "idle_unload_minutes": 15,
    "preset": "",
    "hyde": False,
}


def settings_file() -> Path:
    return credentials_file().parent / "settings.json"


def load(path: Path | None = None) -> dict:
    target = path or settings_file()
    values = dict(DEFAULTS)
    if target.exists():
        try:
            data = json.loads(target.read_text())
        except (OSError, json.JSONDecodeError):
            return values
        if isinstance(data, dict):
            values.update(data)
    return values


def save(values: dict, path: Path | None = None) -> None:
    target = path or settings_file()
    merged = load(target)
    merged.update(values)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(merged, indent=2))
    target.chmod(0o600)
