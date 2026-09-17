from __future__ import annotations

import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from ragdesk.embed import HashingEmbedder
from ragdesk.store import Store
from ragdesk.web import WebError, crawl_site

PAGES: dict[str, object] = {
    "/": (
        "<html><head><title>Home</title></head><body>"
        "<a href='/docs/a'>A</a> <a href='/docs/b'>B</a>"
        "<a href='https://elsewhere.example/x'>external</a>"
        "<a href='/logo.png'>image</a>"
        "</body></html>"
    ),
    "/docs/a": ("<html><head><title>Doc A</title></head><body>alpha rollback canary</body></html>"),
    "/docs/b": ("<html><head><title>Doc B</title></head><body>bravo oauth pkce</body></html>"),
    "/logo.png": b"\x89PNG\x00binary",
}


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = PAGES.get(self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        if isinstance(body, bytes):
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        payload = str(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:
        pass


@pytest.fixture()
def site():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture()
def store(tmp_path: Path):
    with Store(tmp_path / "index.db") as store:
        yield store


def test_crawl_indexes_same_host_html_pages(store: Store, site: str):
    stats = crawl_site(store, HashingEmbedder(), start_url=f"{site}/", max_depth=1)
    assert stats.indexed == 3  # home + a + b
    assert stats.skipped == 1  # the png link is not HTML
    host = urllib.parse.urlparse(site).netloc
    paths = {doc["path"] for doc in store.documents()}
    assert paths == {f"web://{host}/", f"web://{host}/docs/a", f"web://{host}/docs/b"}
    assert {doc["source"] for doc in store.documents()} == {f"web:{host}"}


def test_crawl_depth_limit(store: Store, site: str):
    stats = crawl_site(store, HashingEmbedder(), start_url=f"{site}/", max_depth=0)
    assert stats.indexed == 1  # only the start page


def test_crawl_invalid_url(store: Store):
    with pytest.raises(WebError):
        crawl_site(store, HashingEmbedder(), start_url="not-a-url")


def test_crawl_is_incremental(store: Store, site: str):
    embedder = HashingEmbedder()
    crawl_site(store, embedder, start_url=f"{site}/", max_depth=1)
    stats = crawl_site(store, embedder, start_url=f"{site}/", max_depth=1)
    assert stats.indexed == 0
    assert stats.unchanged == 3
