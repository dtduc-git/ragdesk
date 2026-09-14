from __future__ import annotations

import io
import tarfile
import urllib.error
from pathlib import Path

import pytest

from ragdesk.embed import HashingEmbedder
from ragdesk.github import (
    GitHubError,
    _post_form,
    _tar_members,
    device_flow_poll,
    device_flow_poll_once,
    resolve_client_id,
    resolve_token,
    sync_github,
    token_source,
    whoami,
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
    monkeypatch.setattr("ragdesk.github.credentials.get", lambda provider: {})
    assert resolve_token("explicit") == "explicit"
    monkeypatch.setenv("GITHUB_TOKEN", "env-token")
    assert resolve_token() == "env-token"
    monkeypatch.delenv("GITHUB_TOKEN")
    monkeypatch.setenv("GH_TOKEN", "gh-token")
    assert resolve_token() == "gh-token"
    monkeypatch.delenv("GH_TOKEN")
    monkeypatch.setattr("ragdesk.github._token_from_gh", lambda: "cli-token")
    assert resolve_token() == "cli-token"
    # stored credentials beat the gh CLI
    monkeypatch.setattr(
        "ragdesk.github.credentials.get", lambda provider: {"token": "stored"}
    )
    assert resolve_token() == "stored"


def test_token_source_reports_origin(monkeypatch):
    monkeypatch.setattr("ragdesk.github.credentials.get", lambda provider: {})
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr("ragdesk.github._token_from_gh", lambda: None)
    assert token_source() is None
    monkeypatch.setattr("ragdesk.github._token_from_gh", lambda: "cli")
    assert token_source() == ("gh", "cli")


def test_token_source_respects_disconnect_flag(monkeypatch):
    monkeypatch.setattr("ragdesk.github.credentials.get", lambda provider: {})
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr("ragdesk.github._token_from_gh", lambda: "cli")
    monkeypatch.setattr("ragdesk.settings.load", lambda path=None: {"github_ignore_gh": True})
    assert token_source() is None
    # a stored token still wins over the flag
    monkeypatch.setattr(
        "ragdesk.github.credentials.get", lambda provider: {"token": "stored"}
    )
    assert token_source() == ("credentials", "stored")


def test_whoami_rejects_bad_token(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", {}, None)

    monkeypatch.setattr("ragdesk.github.urllib.request.urlopen", fake_urlopen)
    with pytest.raises(GitHubError):
        whoami("bad-token")


def test_post_form_wraps_http_errors(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr("ragdesk.github.urllib.request.urlopen", fake_urlopen)
    with pytest.raises(GitHubError, match="OAuth client ID"):
        _post_form("https://github.com/login/device/code", {})


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


def test_device_flow_poll_once_states(monkeypatch):
    responses = iter(
        [
            {"error": "authorization_pending"},
            {"error": "slow_down"},
            {"access_token": "tok-9"},
        ]
    )
    monkeypatch.setattr("ragdesk.github._post_form", lambda *a, **k: next(responses))
    assert device_flow_poll_once("cid", "dc") == ("pending", None)
    assert device_flow_poll_once("cid", "dc") == ("slow_down", None)
    assert device_flow_poll_once("cid", "dc") == ("token", "tok-9")


def test_resolve_client_id_precedence(monkeypatch):
    monkeypatch.setattr("ragdesk.github.credentials.get", lambda provider: {})
    monkeypatch.setattr("ragdesk.defaults.GITHUB_CLIENT_ID", "")
    monkeypatch.delenv("RAGDESK_GITHUB_CLIENT_ID", raising=False)
    assert resolve_client_id() is None
    monkeypatch.setenv("RAGDESK_GITHUB_CLIENT_ID", "env-cid")
    assert resolve_client_id() == "env-cid"
    monkeypatch.setattr(
        "ragdesk.github.credentials.get", lambda provider: {"client_id": "stored-cid"}
    )
    assert resolve_client_id() == "env-cid"  # env wins
    monkeypatch.delenv("RAGDESK_GITHUB_CLIENT_ID")
    assert resolve_client_id() == "stored-cid"
    assert resolve_client_id("explicit") == "explicit"


def test_device_flow_timeout(monkeypatch):
    monkeypatch.setattr(
        "ragdesk.github._post_form", lambda *a, **k: {"error": "authorization_pending"}
    )
    with pytest.raises(GitHubError):
        device_flow_poll("client", "device-code", interval=0.0, timeout=0.0)
