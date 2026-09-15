"""Full-screen TUI tests — skipped unless the ``tui`` extra is installed."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("textual")

from textual.widgets import Input  # noqa: E402

from ragdesk.embed import HashingEmbedder  # noqa: E402
from ragdesk.index import index_paths  # noqa: E402
from ragdesk.screen import ChatScreen  # noqa: E402
from ragdesk.store import Store  # noqa: E402


class FakeLLM:
    def generate(self, prompt: str, options: dict) -> str:
        return "the answer [1]"

    def generate_stream(self, prompt: str, options: dict):
        yield "the "
        yield "answer [1]"


def test_tui_answers_and_records_the_exchange(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "auth.md").write_text("access tokens expire after 60 minutes")
    db = str(tmp_path / "index.db")
    embedder = HashingEmbedder(dim=512)
    with Store(db) as store:
        index_paths(store, embedder, [docs])

    async def scenario() -> None:
        app = ChatScreen(db=db, embedder=embedder, llm=FakeLLM(), top_k=3)
        async with app.run_test() as pilot:
            app.query_one("#prompt", Input).value = "when do access tokens expire"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.busy is False

    asyncio.run(scenario())

    with Store(db) as store:
        chats = store.chats()
        assert chats and chats[0]["messages"] == 2
        detail = store.chat(int(chats[0]["id"]))
    roles = [message["role"] for message in detail["messages"]]
    assert roles == ["user", "assistant"]
    assert detail["messages"][1]["text"] == "the answer [1]"
    assert detail["messages"][1]["citations"][0]["path"].endswith("auth.md")
