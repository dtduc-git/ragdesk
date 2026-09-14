from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from ragdesk.embed import HashingEmbedder
from ragdesk.index import index_paths
from ragdesk.serve import AppState, make_server
from ragdesk.store import Store

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture()
def base_url(tmp_path: Path):
    db = tmp_path / "index.db"
    embedder = HashingEmbedder()
    with Store(db) as store:
        index_paths(store, embedder, [FIXTURES / "docs"])
    state = AppState(
        db=str(db),
        embedder=embedder,
        rerank="none",
        llm_model="test-model",
        llm_host="http://127.0.0.1:9",  # unreachable: LLM calls must fail fast
    )
    httpd = make_server(state, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def request(url: str, payload: dict | None = None) -> tuple[int, dict]:
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(req) as response:
        return response.status, json.loads(response.read())


def test_status(base_url: str):
    status, payload = request(f"{base_url}/api/status")
    assert status == 200
    assert payload["documents"] == 3
    assert payload["chunks"] == 3
    assert payload["embedder"]["name"] == "hash-4096"
    assert payload["llm_model"] == "test-model"


def test_search(base_url: str):
    status, payload = request(
        f"{base_url}/api/search", {"query": "access tokens refresh", "top_k": 3}
    )
    assert status == 200
    assert payload["hits"][0]["path"].endswith("auth.md")
    assert payload["hits"][0]["lanes"]


def test_index_endpoint(base_url: str, tmp_path: Path):
    docs = tmp_path / "more"
    docs.mkdir()
    (docs / "note.md").write_text("a brand new note about widgets")
    status, payload = request(f"{base_url}/api/index", {"paths": [str(docs)]})
    assert status == 200
    assert payload["indexed"] == 1
    status, payload = request(f"{base_url}/api/status")
    assert payload["documents"] == 4


def test_index_requires_paths(base_url: str):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        request(f"{base_url}/api/index", {"paths": []})
    assert excinfo.value.code == 400


def test_ask_grounding_gate_skips_llm(base_url: str):
    status, payload = request(
        f"{base_url}/api/ask", {"query": "access tokens", "min_cosine": 0.99}
    )
    assert status == 200
    assert payload["refused"] is True
    assert payload["hits"]


def test_ask_llm_unavailable_surfaces_503(base_url: str):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        request(f"{base_url}/api/ask", {"query": "access tokens"})
    assert excinfo.value.code == 503
    body = json.loads(excinfo.value.read())
    assert "cannot reach Ollama" in body["error"]


def test_sync_github_requires_repo(base_url: str):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        request(f"{base_url}/api/sync/github", {})
    assert excinfo.value.code == 400


def test_sync_github_success(base_url: str, monkeypatch):
    from ragdesk.index import IndexStats

    monkeypatch.setattr(
        "ragdesk.serve.sync_github",
        lambda *args, **kwargs: IndexStats(files_scanned=2, indexed=1, chunks=3),
    )
    status, payload = request(f"{base_url}/api/sync/github", {"repo": "owner/repo"})
    assert status == 200
    assert payload["indexed"] == 1
    assert payload["repo"] == "owner/repo"
