from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragdesk import credentials


def test_roundtrip_and_permissions(tmp_path: Path):
    path = tmp_path / "credentials.json"
    credentials.set_provider("github", {"token": "t1", "login": "duke"}, path)
    assert credentials.get("github", path)["login"] == "duke"
    assert (path.stat().st_mode & 0o777) == 0o600
    # merging keeps existing keys
    credentials.set_provider("github", {"token": "t2"}, path)
    entry = credentials.get("github", path)
    assert entry["token"] == "t2" and entry["login"] == "duke"
    # other providers untouched
    assert credentials.get("confluence", path) == {}


def test_clear_and_missing_file(tmp_path: Path):
    path = tmp_path / "credentials.json"
    assert credentials.load(path) == {}
    credentials.set_provider("gdrive", {"client_id": "c1"}, path)
    credentials.clear("gdrive", path)
    assert credentials.get("gdrive", path) == {}


def test_corrupt_file_is_ignored(tmp_path: Path):
    path = tmp_path / "credentials.json"
    path.write_text("{not json")
    assert credentials.load(path) == {}


def test_config_dir_env_override(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RAGDESK_CONFIG_DIR", str(tmp_path))
    assert credentials.credentials_file() == tmp_path / "credentials.json"
    credentials.set_provider("confluence", {"email": "a@b.c"})
    saved = json.loads((tmp_path / "credentials.json").read_text())
    assert saved["confluence"]["email"] == "a@b.c"


def test_failed_save_keeps_the_previous_file_and_no_temp_litter(tmp_path, monkeypatch):
    path = tmp_path / "credentials.json"
    credentials.set_provider("github", {"token": "first"}, path)

    def broken_replace(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(credentials.os, "replace", broken_replace)
    with pytest.raises(OSError):
        credentials.set_provider("github", {"token": "second"}, path)
    assert credentials.get("github", path)["token"] == "first"
    assert list(tmp_path.glob("*.tmp")) == []


def test_concurrent_writes_never_publish_a_mixed_file(tmp_path):
    """Two writers sharing one temp path could interleave their bytes."""
    import threading

    path = tmp_path / "credentials.json"
    bodies = [json.dumps({"writer": name, "payload": name * 200_000}) for name in "ab"]
    failures: list[BaseException] = []

    def writer(text: str) -> None:
        try:
            for _ in range(40):
                credentials.write_private(path, text)
        except BaseException as exc:  # noqa: BLE001 - reported below
            failures.append(exc)

    threads = [threading.Thread(target=writer, args=(text,)) for text in bodies]
    for thread in threads:
        thread.start()
    saw_partial = False
    while any(thread.is_alive() for thread in threads):
        try:
            json.loads(path.read_text())
        except FileNotFoundError:
            pass  # before the first rename there is nothing to read
        except (OSError, json.JSONDecodeError):
            saw_partial = True
    for thread in threads:
        thread.join()
    assert not failures
    assert not saw_partial, "a reader saw a partially written file"
    final = json.loads(path.read_text())
    assert any(final == json.loads(text) for text in bodies)
