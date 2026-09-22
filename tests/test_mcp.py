from __future__ import annotations

from pathlib import Path

import pytest

from ragdesk.embed import HashingEmbedder
from ragdesk.index import index_paths
from ragdesk.mcp import McpServer
from ragdesk.store import Store

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture()
def server(tmp_path: Path) -> McpServer:
    db = tmp_path / "index.db"
    with Store(db) as store:
        index_paths(store, HashingEmbedder(), [FIXTURES / "docs"])
    return McpServer(db=str(db), embedder=HashingEmbedder())


def test_initialize(server: McpServer):
    response = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert response is not None
    assert response["result"]["serverInfo"]["name"] == "ragdesk"
    assert response["result"]["protocolVersion"]


def test_notifications_are_silent(server: McpServer):
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_tools_list(server: McpServer):
    response = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = [tool["name"] for tool in response["result"]["tools"]]
    assert set(names) == {
        "ragdesk_search",
        "ragdesk_document",
        "ragdesk_sources",
        "ragdesk_symbol",
        "ragdesk_topics",
        "ragdesk_save",
    }


def test_search_tool(server: McpServer):
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "ragdesk_search", "arguments": {"query": "access tokens refresh"}},
        }
    )
    text = response["result"]["content"][0]["text"]
    assert "auth.md" in text


def test_search_tool_returns_one_block_per_section(tmp_path: Path):
    """Two overlapping chunks of one section are one result, not a duplicate."""
    from ragdesk.index import index_document

    db = tmp_path / "index.db"
    embedder = HashingEmbedder()
    body = "Access tokens expire after 60 minutes and refresh tokens renew them. " * 40
    with Store(db) as store:
        index_document(store, embedder, source="local", path="long.md", content=body)
    server = McpServer(db=str(db), embedder=embedder)
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "ragdesk_search", "arguments": {"query": "access tokens expire"}},
        }
    )
    text = response["result"]["content"][0]["text"]
    assert text.count("] long.md") == 1


def test_search_tool_requires_query(server: McpServer):
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "ragdesk_search", "arguments": {}},
        }
    )
    assert response["result"]["isError"] is True


def test_document_tool(server: McpServer):
    with Store(server.db) as store:
        path = store.documents()[0]["path"]
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "ragdesk_document", "arguments": {"path": path}},
        }
    )
    assert len(response["result"]["content"][0]["text"]) > 20

    missing = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "ragdesk_document", "arguments": {"path": "nope.md"}},
        }
    )
    assert missing["result"]["isError"] is True


def test_sources_tool(server: McpServer):
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": "ragdesk_sources", "arguments": {}},
        }
    )
    assert "local:" in response["result"]["content"][0]["text"]


def test_unknown_method_and_tool(server: McpServer):
    response = server.handle({"jsonrpc": "2.0", "id": 8, "method": "nope"})
    assert response["error"]["code"] == -32601

    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {"name": "does-not-exist", "arguments": {}},
        }
    )
    assert response["result"]["isError"] is True


def test_symbol_and_topics_tools(server: McpServer):
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {"name": "ragdesk_symbol", "arguments": {"name": "hybrid_search"}},
        }
    )
    text = response["result"]["content"][0]["text"]
    assert "no definitions or call sites" in text  # the fixture corpus has no code

    response = server.handle(
        {"jsonrpc": "2.0", "id": 10, "method": "tools/call", "params": {"name": "ragdesk_topics"}}
    )
    assert not response["result"].get("isError")


def test_save_tool_is_the_one_writer(server: McpServer, monkeypatch):
    from ragdesk.index import IndexStats

    page = "<html><title>Saved</title><body><p>a saved page</p></body></html>"
    monkeypatch.setattr("ragdesk.web._fetch", lambda url, timeout=30.0: page)
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {"name": "ragdesk_save", "arguments": {"url": "https://example.com/post"}},
        }
    )
    assert "saved https://example.com/post" in response["result"]["content"][0]["text"]

    missing = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/call",
            "params": {"name": "ragdesk_save", "arguments": {}},
        }
    )
    assert missing["result"]["isError"] is True
    _ = IndexStats
