from __future__ import annotations

from pathlib import Path

import pytest

from ragdesk.embed import HashingEmbedder
from ragdesk.gdrive import (
    GdriveError,
    _callback_code,
    auth_url,
    load_token_file,
    pkce_pair,
    save_token_file,
    sync_gdrive,
)
from ragdesk.store import Store


@pytest.fixture()
def store(tmp_path: Path):
    with Store(tmp_path / "index.db") as store:
        yield store


FILES_PAGE = {
    "files": [
        {"id": "d1", "name": "Runbook", "mimeType": "application/vnd.google-apps.document"},
        {"id": "d2", "name": "Notes", "mimeType": "text/markdown"},
        {"id": "d3", "name": "Photo", "mimeType": "image/png"},
        {"id": "d4", "name": "Folder", "mimeType": "application/vnd.google-apps.folder"},
    ],
    "nextPageToken": "",
}


def test_pkce_pair_and_auth_url():
    verifier, challenge = pkce_pair()
    assert 43 <= len(verifier) <= 128
    assert "=" not in challenge and "=" not in verifier
    url = auth_url("client-123", "http://127.0.0.1:9999/oauth/callback", challenge, "st8")
    assert url.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
    assert "code_challenge_method=S256" in url
    assert "client-123" in url
    assert "state=st8" in url


def test_token_file_roundtrip(tmp_path: Path):
    path = tmp_path / "gdrive.json"
    save_token_file({"refresh_token": "r1", "access_token": "a1"}, path)
    assert load_token_file(path)["refresh_token"] == "r1"
    assert (path.stat().st_mode & 0o777) == 0o600


def test_sync_gdrive_indexes_and_dedupes(store: Store, monkeypatch):
    monkeypatch.setattr(
        "ragdesk.gdrive._list_files", lambda token, folder: iter(FILES_PAGE["files"])
    )

    def fake_fetch(file: dict, token: str) -> str | None:
        if file["mimeType"] == "application/vnd.google-apps.document":
            return "deploy rollback canary terraform"
        if file["mimeType"] == "text/markdown":
            return "notes about oauth pkce"
        return None

    monkeypatch.setattr("ragdesk.gdrive._fetch_text", fake_fetch)
    stats = sync_gdrive(store, HashingEmbedder(), access_token="tok")
    assert stats.indexed == 2
    assert stats.skipped == 1  # the png; folders are not counted at all
    paths = {doc["path"] for doc in store.documents()}
    assert "gdrive://d1/Runbook" in paths
    assert {doc["source"] for doc in store.documents()} == {"gdrive"}

    stats = sync_gdrive(store, HashingEmbedder(), access_token="tok")
    assert stats.indexed == 0
    assert stats.unchanged == 2


def test_sync_gdrive_requires_credentials(store: Store, monkeypatch, tmp_path: Path):
    monkeypatch.delenv("GDRIVE_CLIENT_ID", raising=False)
    monkeypatch.delenv("GDRIVE_CLIENT_SECRET", raising=False)
    with pytest.raises(GdriveError):
        sync_gdrive(
            store,
            HashingEmbedder(),
            token_file=tmp_path / "missing.json",
            interactive=False,
        )


def test_callback_code_validation():
    with pytest.raises(GdriveError):
        _callback_code({}, "state-1")
    with pytest.raises(GdriveError):
        _callback_code({"code": "c", "state": "WRONG"}, "state-1")
    with pytest.raises(GdriveError):
        _callback_code({"state": "state-1", "error": "access_denied"}, "state-1")
    assert _callback_code({"code": "c", "state": "state-1"}, "state-1") == "c"
