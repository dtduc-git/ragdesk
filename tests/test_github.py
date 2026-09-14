from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from ragdesk.embed import HashingEmbedder
from ragdesk.github import (
    GitHubError,
    _tar_members,
    device_flow_poll,
    resolve_token,
    sync_github,
)
from ragdesk.store import Store


def make_tarball(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(f"owner-repo-deadbeef/{name}")
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


@pytest.fixture()
def store(tmp_path: Path):
    with Store(tmp_path / "index.db") as store:
        yield store


def test_tar_members_filters( ):
    data = make_tarball(
        {
            "README.md": b"# hello",
            "src/app.py": b"print('hi')",
            "node_modules/dep/index.js": b"module.exports = 1",
            "assets/logo.bin": b"\x00\x01\x02binary",
            "docs/notes.txt": b"notes",
        }
    )
    files = dict(_tar_members(data, ""))
    assert set(files) == {"README.md", "src/app.py", "docs/notes.txt"}
    subset = dict(_tar_members(data, "docs"))
    assert set(subset) == {"docs/notes.txt"}


def test_sync_github_indexes_and_dedupes(store: Store, monkeypatch):
    payload = make_tarball(
        {
            "README.md": b"# ragdesk test repo\n\nRollback steps live here.",
            "src/app.py": b"def main():\n    return 42\n",
        }
    )
    monkeypatch.setattr("ragdesk.github._download_tarball", lambda *a, **k: payload)

    stats = sync_github(store, HashingEmbedder(), repo="owner/repo", token="fake")
    assert stats.indexed == 2
    assert stats.chunks >= 2
    paths = [doc["path"] for doc in store.documents()]
    assert all(path.startswith("github://owner/repo/") for path in paths)
    assert {doc["source"] for doc in store.documents()} == {"github:owner/repo"}

    stats = sync_github(store, HashingEmbedder(), repo="owner/repo", token="fake")
    assert stats.indexed == 0
    assert stats.files_scanned == 0  # whole-repo sha skip


def test_sync_github_requires_token(store: Store, monkeypatch):
    monkeypatch.setattr("ragdesk.github.resolve_token", lambda explicit=None: None)
    with pytest.raises(GitHubError):
        sync_github(store, HashingEmbedder(), repo="owner/repo")


def test_resolve_token_precedence(monkeypatch):
    assert resolve_token("explicit") == "explicit"
    monkeypatch.setenv("GITHUB_TOKEN", "env-token")
    assert resolve_token() == "env-token"
    monkeypatch.delenv("GITHUB_TOKEN")
    monkeypatch.setenv("GH_TOKEN", "gh-token")
    assert resolve_token() == "gh-token"
    monkeypatch.delenv("GH_TOKEN")
    monkeypatch.setattr("ragdesk.github._token_from_gh", lambda: "cli-token")
    assert resolve_token() == "cli-token"


def test_device_flow_poll(monkeypatch):
    calls = {"n": 0}

    def fake_post(url: str, data: dict, timeout: float = 30.0) -> dict:
        calls["n"] += 1
        if calls["n"] == 1:
            return {"error": "authorization_pending"}
        return {"access_token": "tok-123"}

    monkeypatch.setattr("ragdesk.github._post_form", fake_post)
    token = device_flow_poll("client", "device-code", interval=0.0, timeout=5.0)
    assert token == "tok-123"
    assert calls["n"] == 2


def test_device_flow_timeout(monkeypatch):
    monkeypatch.setattr(
        "ragdesk.github._post_form", lambda *a, **k: {"error": "authorization_pending"}
    )
    with pytest.raises(GitHubError):
        device_flow_poll("client", "device-code", interval=0.0, timeout=0.0)
