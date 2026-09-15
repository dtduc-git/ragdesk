from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from ragdesk.embed import HashingEmbedder
from ragdesk.index import index_paths
from ragdesk.store import Store
from ragdesk.tui import ask_remote, chat_once, run_remote_chat


class FakeLLM:
    def generate(self, prompt: str, options: dict) -> str:
        return "answer"

    def generate_stream(self, prompt: str, options: dict):
        yield "the "
        yield "answer [1]"


def test_chat_once_records_and_cites(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "auth.md").write_text("access tokens expire after 60 minutes")
    out: list[str] = []
    with Store(tmp_path / "index.db") as store:
        index_paths(store, HashingEmbedder(dim=512), [docs])
        chat_id, text, hits = chat_once(
            store,
            HashingEmbedder(dim=512),
            FakeLLM(),
            "when do access tokens expire",
            out=out.append,
            stream=False,
        )
        assert chat_id > 0
        assert text == "the answer [1]"
        assert hits and hits[0].path.endswith("auth.md")
        detail = store.chat(chat_id)
    assert [message["role"] for message in detail["messages"]] == ["user", "assistant"]
    assert any("auth.md" in line for line in out)  # the citation footer


class _NdjsonHandler(BaseHTTPRequestHandler):
    events: list[dict] = []
    fail_with_error = False

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        if _NdjsonHandler.fail_with_error:
            self.wfile.write(json.dumps({"error": "model not loaded"}).encode() + b"\n")
            return
        for event in _NdjsonHandler.events:
            self.wfile.write(json.dumps(event).encode() + b"\n")

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture()
def fake_server():
    _NdjsonHandler.events = [
        {"status": "searching your sources…"},
        {"delta": "hello "},
        {"delta": "world", "done": True, "chat_id": 7, "hits": [{"path": "a.md", "line": 3}]},
    ]
    _NdjsonHandler.fail_with_error = False
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _NdjsonHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_ask_remote_reads_the_ndjson_stream(fake_server: str):
    chat_id, text, hits = ask_remote(
        fake_server, "hi", out=lambda _line: None, stream=False
    )
    assert chat_id == 7
    assert text == "hello world"
    assert hits == [{"path": "a.md", "line": 3}]


def test_ask_remote_surfaces_a_server_error(fake_server: str):
    _NdjsonHandler.fail_with_error = True
    with pytest.raises(RuntimeError, match="model not loaded"):
        ask_remote(fake_server, "hi", out=lambda _line: None, stream=False)


def test_run_remote_chat_repl_prints_deltas(fake_server: str):
    out: list[str] = []
    code = run_remote_chat(
        fake_server,
        input_stream=io.StringIO("/help\nhi\n/quit\n"),
        out=out.append,
        interactive=False,
    )
    assert code == 0
    assert any("hello world" in line for line in out)
    assert any("/new starts a conversation" in line for line in out)
    assert any("[1] a.md:3" in line for line in out)
