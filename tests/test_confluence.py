from __future__ import annotations

from pathlib import Path

import pytest

from ragdesk.confluence import (
    ConfluenceError,
    html_to_text,
    sync_confluence,
)
from ragdesk.embed import HashingEmbedder
from ragdesk.store import Store


@pytest.fixture()
def store(tmp_path: Path):
    with Store(tmp_path / "index.db") as store:
        yield store


def page(pid: str, title: str, body: str) -> dict:
    return {
        "id": pid,
        "title": title,
        "body": {"storage": {"value": body}},
        "version": {"number": 1},
    }


PAGE_ONE = {
    "results": [
        page(
            "1",
            "Deploy runbook",
            "<p>Use <strong>terraform</strong> for infra.</p>"
            "<ul><li>rollback</li><li>canary</li></ul>",
        ),
        page("2", "Empty page", "<p>   </p>"),
    ],
    "_links": {"next": "/wiki/rest/api/content/search?cql=more&start=2"},
}

PAGE_TWO = {
    "results": [page("3", "Auth notes", "<p>OAuth PKCE for desktop</p>")],
    "_links": {},
}


def test_html_to_text():
    text = html_to_text("<h1>Title</h1><p>Hello <b>world</b></p><ul><li>one</li><li>two</li></ul>")
    assert "Title" in text
    assert "Hello world" in text
    assert "one" in text and "two" in text
    assert "<" not in text and ">" not in text


def test_sync_confluence_paginates_and_indexes(store: Store, monkeypatch):
    seen_urls: list[str] = []

    def fake_get_json(url: str, auth: str, timeout: float = 60.0) -> dict:
        seen_urls.append(url)
        return PAGE_ONE if "start=2" not in url else PAGE_TWO

    monkeypatch.setattr("ragdesk.confluence._get_json", fake_get_json)
    stats = sync_confluence(
        store,
        HashingEmbedder(),
        base_url="https://team.atlassian.net",
        space="DOCS",
        email="a@b.c",
        token="tok",
    )
    assert stats.indexed == 2  # empty page skipped
    assert stats.skipped == 1
    assert len(seen_urls) == 2
    assert "start=2" in seen_urls[1]

    docs = store.documents()
    paths = {doc["path"] for doc in docs}
    assert "confluence://team.atlassian.net/DOCS/1" in paths
    assert {doc["source"] for doc in docs} == {"confluence:team.atlassian.net/DOCS"}

    # unchanged on re-sync
    stats = sync_confluence(
        store,
        HashingEmbedder(),
        base_url="https://team.atlassian.net",
        space="DOCS",
        email="a@b.c",
        token="tok",
    )
    assert stats.indexed == 0
    assert stats.unchanged == 2


def test_sync_confluence_requires_credentials(store: Store, monkeypatch):
    monkeypatch.delenv("CONFLUENCE_EMAIL", raising=False)
    monkeypatch.delenv("CONFLUENCE_TOKEN", raising=False)
    monkeypatch.setattr("ragdesk.confluence.credentials.get", lambda provider: {})
    with pytest.raises(ConfluenceError):
        sync_confluence(
            store,
            HashingEmbedder(),
            base_url="https://team.atlassian.net",
            space="DOCS",
        )


def test_sync_confluence_uses_stored_credentials(store: Store, monkeypatch):
    monkeypatch.delenv("CONFLUENCE_EMAIL", raising=False)
    monkeypatch.delenv("CONFLUENCE_TOKEN", raising=False)
    monkeypatch.setattr(
        "ragdesk.confluence.credentials.get",
        lambda provider: {"email": "a@b.c", "token": "tok"},
    )
    monkeypatch.setattr(
        "ragdesk.confluence._get_json", lambda *a, **k: {"results": [], "_links": {}}
    )
    stats = sync_confluence(
        store,
        HashingEmbedder(),
        base_url="https://team.atlassian.net",
        space="DOCS",
    )
    assert stats.files_scanned == 0


def test_sync_confluence_rejects_bad_space(store: Store):
    with pytest.raises(ConfluenceError):
        sync_confluence(
            store,
            HashingEmbedder(),
            base_url="https://team.atlassian.net",
            space='DOCS" or 1=1',
            email="a@b.c",
            token="tok",
        )
