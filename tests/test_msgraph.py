from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from ragdesk.embed import HashingEmbedder
from ragdesk.msgraph import (
    device_flow_connect,
    device_flow_poll_once,
    resolve_client_id,
    sync_onedrive,
)
from ragdesk.office import extract_office_text
from ragdesk.store import Store


def make_docx(text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "word/document.xml",
            f"<w:document><w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>",
        )
    return buffer.getvalue()


@pytest.fixture()
def store(tmp_path: Path):
    with Store(tmp_path / "index.db") as store:
        yield store


def test_extract_docx_text():
    text = extract_office_text(make_docx("Rollback runbook"), ".docx")
    assert text is not None and "Rollback runbook" in text


def test_extract_office_bad_zip():
    assert extract_office_text(b"not a zip", ".docx") is None


def test_device_flow_poll_pending_then_token(monkeypatch):
    responses = iter(
        [{"error": "authorization_pending"}, {"access_token": "at", "refresh_token": "rt"}]
    )
    monkeypatch.setattr("ragdesk.msgraph._post_form", lambda *a, **k: next(responses))
    status, payload = device_flow_poll_once("cid", "dc")
    assert status == "pending" and payload == {}
    status, payload = device_flow_poll_once("cid", "dc")
    assert status == "token"
    assert payload["access_token"] == "at"
    assert payload["refresh_token"] == "rt"


def test_resolve_client_id_precedence(monkeypatch):
    monkeypatch.setattr("ragdesk.msgraph.credentials.get", lambda provider: {})
    monkeypatch.setattr("ragdesk.defaults.MS_CLIENT_ID", "")
    monkeypatch.delenv("RAGDESK_MS_CLIENT_ID", raising=False)
    assert resolve_client_id() is None
    monkeypatch.setenv("RAGDESK_MS_CLIENT_ID", "env-ms")
    assert resolve_client_id() == "env-ms"
    monkeypatch.delenv("RAGDESK_MS_CLIENT_ID")
    monkeypatch.setattr("ragdesk.defaults.MS_CLIENT_ID", "shipped-ms")
    assert resolve_client_id() == "shipped-ms"
    monkeypatch.setattr(
        "ragdesk.msgraph.credentials.get", lambda provider: {"client_id": "stored-ms"}
    )
    assert resolve_client_id() == "stored-ms"


def test_device_flow_connect_saves_tokens(monkeypatch):
    saved: dict = {}
    monkeypatch.setattr(
        "ragdesk.msgraph.device_flow_start",
        lambda client_id: {
            "device_code": "dc",
            "user_code": "CODE-1",
            "interval": 0,
        },
    )
    responses = iter([("pending", {}), ("token", {"access_token": "at", "refresh_token": "rt"})])
    monkeypatch.setattr("ragdesk.msgraph.device_flow_poll_once", lambda cid, dc: next(responses))
    monkeypatch.setattr("ragdesk.msgraph.whoami", lambda token: "duke@outlook.com")
    monkeypatch.setattr(
        "ragdesk.msgraph.credentials.set_provider",
        lambda provider, values: saved.update(values),
    )
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    seen: dict = {}
    payload = device_flow_connect("cid", on_code=lambda start: seen.update(start))
    assert seen["user_code"] == "CODE-1"
    assert payload["access_token"] == "at"
    assert saved["refresh_token"] == "rt"
    assert saved["account"] == "duke@outlook.com"


def test_sync_onedrive_indexes_and_dedupes(store: Store, monkeypatch):
    items = [
        {"id": "f1", "name": "notes.md", "size": 20, "file": {}},
        {"id": "f2", "name": "runbook.docx", "size": 100, "file": {}},
        {"id": "f3", "name": "photo.png", "size": 500, "file": {}},
    ]
    contents = {
        "f1": b"# notes\n\nalpha rollback",
        "f2": make_docx("docx canary rollback"),
        "f3": b"\x89PNG\r\n\x1a\n not a real image",
    }
    monkeypatch.setattr("ragdesk.msgraph.resolve_access_token", lambda *a, **k: "tok")
    monkeypatch.setattr(
        "ragdesk.msgraph._drive_base",
        lambda *a, **k: "https://graph.microsoft.com/v1.0/me/drive",
    )
    monkeypatch.setattr("ragdesk.msgraph.walk_drive", lambda *a, **k: items)
    monkeypatch.setattr(
        "ragdesk.msgraph._get_bytes",
        lambda url, token, timeout=180.0: contents[url.rsplit("/items/", 1)[1].split("/")[0]],
    )

    stats = sync_onedrive(store, HashingEmbedder(), access_token="tok")
    assert stats.indexed == 2
    assert stats.skipped == 1  # the png
    paths = {doc["path"] for doc in store.documents()}
    assert paths == {"onedrive://f1/notes.md", "onedrive://f2/runbook.docx"}
    assert {doc["source"] for doc in store.documents()} == {"msgraph:onedrive"}

    stats = sync_onedrive(store, HashingEmbedder(), access_token="tok")
    assert stats.indexed == 0
    assert stats.unchanged == 2
