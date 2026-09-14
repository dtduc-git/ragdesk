from __future__ import annotations

from pathlib import Path

import pytest

from ragdesk.embed import HashingEmbedder
from ragdesk.notion import NotionError, block_text, page_text, sync_notion, whoami
from ragdesk.store import Store

PAGES = {
    "results": [
        {
            "id": "page-1",
            "url": "https://notion.so/page-1",
            "last_edited_time": "2026-09-14T10:00:00.000Z",
            "properties": {
                "Name": {
                    "type": "title",
                    "title": [{"plain_text": "Deploy runbook"}],
                }
            },
        },
        {
            "id": "page-2",
            "url": "https://notion.so/page-2",
            "last_edited_time": "2026-09-13T09:00:00.000Z",
            "properties": {
                "Name": {
                    "type": "title",
                    "title": [{"plain_text": "Auth notes"}],
                }
            },
        },
    ],
    "has_more": False,
    "next_cursor": None,
}

BLOCKS = {
    "page-1": {
        "results": [
            {
                "id": "b1",
                "type": "heading_1",
                "heading_1": {"rich_text": [{"plain_text": "Rollback"}]},
            },
            {
                "id": "b2",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"plain_text": "Use kubectl rollout undo."}]},
                "has_children": True,
            },
            {
                "id": "b3",
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": [{"plain_text": "canary first"}]},
            },
            {"id": "b4", "type": "divider", "divider": {}},
        ],
        "has_more": False,
    },
    "b2": {
        "results": [
            {
                "id": "b2a",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"plain_text": "Nested detail."}]},
            }
        ],
        "has_more": False,
    },
    "page-2": {"results": [], "has_more": False},
}


def fake_request(method: str, path: str, token: str, *, body=None, timeout=60.0) -> dict:
    if path == "/users/me":
        return {"name": "Duke's Bot"}
    if path == "/search":
        return PAGES
    if path.startswith("/blocks/"):
        block_id = path.split("/blocks/")[1].split("/")[0]
        return BLOCKS.get(block_id, {"results": [], "has_more": False})
    raise AssertionError(f"unexpected path {path}")


@pytest.fixture()
def store(tmp_path: Path):
    with Store(tmp_path / "index.db") as store:
        yield store


def test_block_text_variants():
    heading = {"type": "heading_2", "heading_2": {"rich_text": [{"plain_text": "Hi"}]}}
    bullet = {
        "type": "bulleted_list_item",
        "bulleted_list_item": {"rich_text": [{"plain_text": "x"}]},
    }
    assert block_text(heading).strip() == "## Hi"
    assert block_text(bullet) == "- x"
    assert block_text({"type": "divider", "divider": {}}) == ""


def test_whoami(monkeypatch):
    monkeypatch.setattr("ragdesk.notion._request", fake_request)
    assert whoami("tok") == "Duke's Bot"


def test_page_text_walks_children(monkeypatch):
    monkeypatch.setattr("ragdesk.notion._request", fake_request)
    text = page_text("page-1", "tok")
    assert "Rollback" in text
    assert "kubectl rollout undo" in text
    assert "Nested detail." in text  # child block followed
    assert "canary first" in text


def test_sync_notion_indexes_and_skips_unchanged(store: Store, monkeypatch):
    monkeypatch.setattr("ragdesk.notion._request", fake_request)
    monkeypatch.setattr("ragdesk.notion.resolve_token", lambda explicit=None: "tok")

    stats = sync_notion(store, HashingEmbedder())
    assert stats.indexed == 1  # page-2 has no content
    assert stats.skipped == 1
    paths = {doc["path"] for doc in store.documents()}
    assert paths == {"notion://page-1/deploy-runbook"}

    stats = sync_notion(store, HashingEmbedder())
    assert stats.indexed == 0
    assert stats.unchanged == 2  # last_edited_time unchanged → no re-fetch


def test_sync_notion_requires_token(store: Store, monkeypatch):
    monkeypatch.setattr("ragdesk.notion.resolve_token", lambda explicit=None: None)
    with pytest.raises(NotionError):
        sync_notion(store, HashingEmbedder())
