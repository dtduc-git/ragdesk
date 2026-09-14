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
    assert names == ["ragdesk_search", "ragdesk_document", "ragdesk_sources"]


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
