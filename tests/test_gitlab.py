from __future__ import annotations

import io
import tarfile
import urllib.error
from pathlib import Path

import pytest

from ragdesk.embed import HashingEmbedder
from ragdesk.gitlab import GitLabError, resolve_token, sync_gitlab, whoami
from ragdesk.store import Store


def make_tarball(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(f"group-repo-deadbeef/{name}")
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


@pytest.fixture()
def store(tmp_path: Path):
    with Store(tmp_path / "index.db") as store:
        yield store


def test_sync_gitlab_indexes_and_dedupes(store: Store, monkeypatch):
    payload = make_tarball(
        {
            "README.md": b"# lab repo\n\nDeploy and rollback notes.",
            "src/app.py": b"def main():\n    return 1\n",
        }
    )
    monkeypatch.setattr(
        "ragdesk.gitlab._get_json",
        lambda url, token, timeout=30.0: {
            "id": 42,
            "default_branch": "main",
            "path_with_namespace": "group/repo",
        },
    )
    monkeypatch.setattr("ragdesk.gitlab._download_archive", lambda *a, **k: payload)

    stats = sync_gitlab(store, HashingEmbedder(), project="group/repo", token="fake")
    assert stats.indexed == 2
    paths = {doc["path"] for doc in store.documents()}
    assert paths == {
        "gitlab://gitlab.com/group/repo/README.md",
        "gitlab://gitlab.com/group/repo/src/app.py",
    }
    assert {doc["source"] for doc in store.documents()} == {"gitlab:gitlab.com/group/repo"}

    stats = sync_gitlab(store, HashingEmbedder(), project="group/repo", token="fake")
    assert stats.indexed == 0
    assert stats.files_scanned == 0  # whole-archive digest skip


def test_sync_gitlab_requires_token(store: Store, monkeypatch):
    monkeypatch.setattr("ragdesk.gitlab.resolve_token", lambda explicit=None: None)
    with pytest.raises(GitLabError):
        sync_gitlab(store, HashingEmbedder(), project="group/repo")


def test_resolve_token_precedence(monkeypatch):
    monkeypatch.setattr("ragdesk.gitlab.credentials.get", lambda provider: {})
    assert resolve_token("explicit") == "explicit"
    monkeypatch.setenv("GITLAB_TOKEN", "env-token")
    assert resolve_token() == "env-token"
    monkeypatch.delenv("GITLAB_TOKEN")
    monkeypatch.setattr("ragdesk.gitlab.credentials.get", lambda provider: {"token": "stored"})
    assert resolve_token() == "stored"


def test_whoami_maps_auth_errors(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", {}, None)

    monkeypatch.setattr("ragdesk.gitlab.urllib.request.urlopen", fake_urlopen)
    with pytest.raises(GitLabError, match="read_api"):
        whoami("bad-token")
