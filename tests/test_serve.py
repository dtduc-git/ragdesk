from __future__ import annotations

import http.client
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from ragdesk.embed import HashingEmbedder
from ragdesk.index import index_paths
from ragdesk.serve import AppState, make_server
from ragdesk.store import Store

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture()
def base_url(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RAGDESK_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("RAGDESK_GITHUB_CLIENT_ID", raising=False)
    monkeypatch.setattr("ragdesk.defaults.GITHUB_CLIENT_ID", "")
    monkeypatch.setattr("ragdesk.defaults.ATLASSIAN_CLIENT_ID", "")
    monkeypatch.setattr("ragdesk.defaults.ATLASSIAN_CLIENT_SECRET", "")
    monkeypatch.setattr("ragdesk.defaults.GOOGLE_CLIENT_ID", "")
    monkeypatch.setattr("ragdesk.defaults.GOOGLE_CLIENT_SECRET", "")
    monkeypatch.setattr("ragdesk.serve.token_source", lambda: None)
    monkeypatch.setattr("ragdesk.serve.load_token_file", lambda: {})
    monkeypatch.delenv("NOTION_TOKEN", raising=False)
    monkeypatch.delenv("GITLAB_TOKEN", raising=False)
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
    assert len(payload["sources"]) == 1
    local = payload["sources"][0]
    assert local["source"] == "local"
    assert local["documents"] == 3
    assert local["chunks"] == 3
    assert local["indexed_at"]


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


def test_connections_empty(base_url: str):
    status, payload = request(f"{base_url}/api/connections")
    assert status == 200
    assert payload["github"]["connected"] is False
    assert payload["confluence"]["connected"] is False
    assert payload["gdrive"]["connected"] is False


def test_connect_github_with_token(base_url: str, monkeypatch):
    monkeypatch.setattr("ragdesk.serve.github_whoami", lambda token: "duke")
    status, payload = request(f"{base_url}/api/connections/github", {"token": "tok"})
    assert status == 200
    assert payload["login"] == "duke"
    _, connections = request(f"{base_url}/api/connections")
    assert connections["github"]["connected"] is True
    assert connections["github"]["login"] == "duke"


def test_connect_github_gh_missing(base_url: str, monkeypatch):
    monkeypatch.setattr("ragdesk.serve._token_from_gh", lambda: None)
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        request(f"{base_url}/api/connections/github/gh", {})
    assert excinfo.value.code == 400


def test_github_device_flow_connect(base_url: str, monkeypatch):
    monkeypatch.setattr("ragdesk.serve._token_from_gh", lambda: None)
    # no client ID yet -> start refuses
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        request(f"{base_url}/api/connections/github/device/start", {})
    assert excinfo.value.code == 400

    # save a client ID, then start succeeds
    status, payload = request(
        f"{base_url}/api/connections/github/client-id", {"client_id": "cid-1"}
    )
    assert payload["saved"] is True
    monkeypatch.setattr(
        "ragdesk.serve.device_flow_start",
        lambda client_id: {
            "device_code": "dc-1",
            "user_code": "ABCD-1234",
            "verification_uri": "https://github.com/login/device",
            "interval": 1,
        },
    )
    status, payload = request(f"{base_url}/api/connections/github/device/start", {})
    assert status == 200
    assert payload["user_code"] == "ABCD-1234"

    # pending poll keeps waiting
    monkeypatch.setattr(
        "ragdesk.serve.device_flow_poll_once", lambda cid, dc: ("pending", None)
    )
    status, payload = request(f"{base_url}/api/connections/github/device/poll", {})
    assert payload["connected"] is False and payload["pending"] is True

    # approved poll stores the token
    monkeypatch.setattr(
        "ragdesk.serve.device_flow_poll_once", lambda cid, dc: ("token", "tok-1")
    )
    monkeypatch.setattr("ragdesk.serve.github_whoami", lambda token: "duke")
    status, payload = request(f"{base_url}/api/connections/github/device/poll", {})
    assert payload["connected"] is True and payload["login"] == "duke"
    _, connections = request(f"{base_url}/api/connections")
    assert connections["github"]["connected"] is True
    assert connections["github"]["device_flow_ready"] is True


def test_connect_confluence_and_disconnect(base_url: str, monkeypatch):
    monkeypatch.setattr("ragdesk.serve.confluence_whoami", lambda *a, **k: "Duke Dinh")
    status, payload = request(
        f"{base_url}/api/connections/confluence",
        {"base_url": "https://x.atlassian.net", "email": "a@b.c", "token": "t"},
    )
    assert status == 200
    assert payload["display_name"] == "Duke Dinh"
    _, connections = request(f"{base_url}/api/connections")
    assert connections["confluence"]["connected"] is True
    assert connections["confluence"]["base_url"] == "https://x.atlassian.net"

    status, payload = request(f"{base_url}/api/connections/confluence/disconnect", {})
    assert payload["connected"] is False
    _, connections = request(f"{base_url}/api/connections")
    assert connections["confluence"]["connected"] is False


def test_connect_confluence_requires_all_fields(base_url: str):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        request(f"{base_url}/api/connections/confluence", {"base_url": "https://x"})
    assert excinfo.value.code == 400


def test_connect_confluence_oauth_and_sync(base_url: str, monkeypatch):
    monkeypatch.setattr(
        "ragdesk.serve.connect_oauth",
        lambda client_id, client_secret, timeout=300.0: {
            "cloud_id": "cloud-1",
            "site_url": "https://team.atlassian.net",
            "site_name": "Team",
            "access_token": "at",
            "refresh_token": "rt",
            "expires_in": 3600,
        },
    )
    status, payload = request(
        f"{base_url}/api/connections/confluence/oauth",
        {"client_id": "cid", "client_secret": "sec"},
    )
    assert status == 200
    assert payload["display_name"] == "Team"
    _, connections = request(f"{base_url}/api/connections")
    assert connections["confluence"]["connected"] is True
    assert connections["confluence"]["source"] == "oauth"
    assert connections["confluence"]["oauth_connected"] is True

    captured: dict = {}

    def fake_sync(store, embedder, **kwargs):
        from ragdesk.index import IndexStats

        captured.update(kwargs)
        return IndexStats()

    monkeypatch.setattr("ragdesk.serve.sync_confluence", fake_sync)
    monkeypatch.setattr(
        "ragdesk.serve.resolve_oauth_credentials",
        lambda: ("https://api.atlassian.com/ex/confluence/cloud-1", "fresh"),
    )
    status, payload = request(f"{base_url}/api/sync/confluence", {"space": "DOCS"})
    assert status == 200
    assert captured["api_base"].endswith("cloud-1")
    assert captured["bearer"] == "fresh"


def test_connect_confluence_oauth_requires_client(base_url: str):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        request(f"{base_url}/api/connections/confluence/oauth", {})
    assert excinfo.value.code == 400


def test_connect_gdrive_requires_client_id(base_url: str):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        request(f"{base_url}/api/connections/gdrive", {})
    assert excinfo.value.code == 400


def test_sync_web_endpoint(base_url: str, monkeypatch):
    from ragdesk.index import IndexStats

    captured: dict = {}

    def fake_crawl(store, embedder, **kwargs):
        captured.update(kwargs)
        return IndexStats(files_scanned=2, indexed=2, chunks=5)

    monkeypatch.setattr("ragdesk.serve.crawl_site", fake_crawl)
    status, payload = request(
        f"{base_url}/api/sync/web",
        {"url": "https://docs.example.com/", "max_pages": 10, "max_depth": 1},
    )
    assert status == 200
    assert payload["indexed"] == 2
    assert captured["max_pages"] == 10
    assert captured["max_depth"] == 1


def test_connect_gitlab_and_sync(base_url: str, monkeypatch):
    from ragdesk.index import IndexStats

    monkeypatch.setattr("ragdesk.serve.gitlab_whoami", lambda token, base_url="": "duke")
    status, payload = request(
        f"{base_url}/api/connections/gitlab", {"token": "glpat-test"}
    )
    assert status == 200
    assert payload["display_name"] == "duke"
    _, connections = request(f"{base_url}/api/connections")
    assert connections["gitlab"]["connected"] is True
    assert connections["gitlab"]["base_url"] == "https://gitlab.com"

    captured: dict = {}

    def fake_sync(store, embedder, **kwargs):
        captured.update(kwargs)
        return IndexStats(files_scanned=1, indexed=1, chunks=3)

    monkeypatch.setattr("ragdesk.serve.sync_gitlab", fake_sync)
    status, payload = request(
        f"{base_url}/api/sync/gitlab", {"project": "group/repo", "ref": "main"}
    )
    assert status == 200
    assert payload["indexed"] == 1
    assert captured["project"] == "group/repo"
    assert captured["ref"] == "main"
    assert captured["base_url"] == "https://gitlab.com"

    status, payload = request(f"{base_url}/api/connections/gitlab/disconnect", {})
    assert payload["connected"] is False


def test_sync_gitlab_requires_project(base_url: str):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        request(f"{base_url}/api/sync/gitlab", {})
    assert excinfo.value.code == 400


def test_msgraph_device_flow(base_url: str, monkeypatch):
    from ragdesk.index import IndexStats

    monkeypatch.delenv("RAGDESK_MS_CLIENT_ID", raising=False)
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        request(f"{base_url}/api/connections/msgraph/device/start", {})
    assert excinfo.value.code == 400  # no client id yet

    status, payload = request(
        f"{base_url}/api/connections/msgraph", {"client_id": "ms-client-1"}
    )
    assert payload["saved"] is True

    monkeypatch.setattr(
        "ragdesk.serve.ms_device_start",
        lambda client_id: {
            "device_code": "dc-ms",
            "user_code": "WXYZ-1234",
            "verification_uri": "https://microsoft.com/devicelogin",
            "interval": 1,
        },
    )
    status, payload = request(f"{base_url}/api/connections/msgraph/device/start", {})
    assert payload["user_code"] == "WXYZ-1234"

    monkeypatch.setattr(
        "ragdesk.serve.ms_poll_once", lambda cid, dc: ("pending", {})
    )
    status, payload = request(f"{base_url}/api/connections/msgraph/device/poll", {})
    assert payload["pending"] is True

    monkeypatch.setattr(
        "ragdesk.serve.ms_poll_once",
        lambda cid, dc: ("token", {"access_token": "at", "refresh_token": "rt"}),
    )
    monkeypatch.setattr("ragdesk.serve.ms_whoami", lambda token: "duke@outlook.com")
    status, payload = request(f"{base_url}/api/connections/msgraph/device/poll", {})
    assert payload["connected"] is True
    assert payload["login"] == "duke@outlook.com"
    _, connections = request(f"{base_url}/api/connections")
    assert connections["msgraph"]["connected"] is True
    assert connections["msgraph"]["account"] == "duke@outlook.com"

    monkeypatch.setattr(
        "ragdesk.serve.sync_onedrive",
        lambda store, embedder, **kwargs: IndexStats(files_scanned=1, indexed=1, chunks=2),
    )
    status, payload = request(f"{base_url}/api/sync/msgraph", {})
    assert status == 200
    assert payload["indexed"] == 1

    status, payload = request(f"{base_url}/api/connections/msgraph/disconnect", {})
    assert payload["connected"] is False


def test_eval_endpoint(base_url: str):
    status, payload = request(
        f"{base_url}/api/eval", {"golden": str(FIXTURES / "golden.jsonl")}
    )
    assert status == 200
    assert payload["metrics"]["recall@5"] >= 0.8
    assert payload["queries"]
    assert payload["golden"].endswith("golden.jsonl")


def test_eval_requires_golden(base_url: str):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        request(f"{base_url}/api/eval", {})
    assert excinfo.value.code == 400


def test_eval_rejects_missing_file(base_url: str):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        request(f"{base_url}/api/eval", {"golden": "/nope/golden.jsonl"})
    assert excinfo.value.code == 400


def test_sync_web_requires_url(base_url: str):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        request(f"{base_url}/api/sync/web", {})
    assert excinfo.value.code == 400


def test_connect_notion_and_sync(base_url: str, monkeypatch):
    from ragdesk.index import IndexStats

    monkeypatch.setattr("ragdesk.serve.notion_whoami", lambda token: "Duke's bot")
    status, payload = request(
        f"{base_url}/api/connections/notion", {"token": "ntn_test"}
    )
    assert status == 200
    assert payload["display_name"] == "Duke's bot"
    _, connections = request(f"{base_url}/api/connections")
    assert connections["notion"]["connected"] is True
    assert connections["notion"]["name"] == "Duke's bot"

    monkeypatch.setattr(
        "ragdesk.serve.sync_notion",
        lambda store, embedder, **kwargs: IndexStats(files_scanned=2, indexed=2, chunks=6),
    )
    status, payload = request(f"{base_url}/api/sync/notion", {})
    assert status == 200
    assert payload["indexed"] == 2

    status, payload = request(f"{base_url}/api/connections/notion/disconnect", {})
    assert payload["connected"] is False
    _, connections = request(f"{base_url}/api/connections")
    assert connections["notion"]["connected"] is False


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


def test_ask_stream(base_url: str, monkeypatch):
    def fake_stream(question, hits, model="", host="", min_cosine=0.0):
        yield "Hel"
        yield "lo"

    monkeypatch.setattr("ragdesk.serve.answer_stream", fake_stream)
    req = urllib.request.Request(
        f"{base_url}/api/ask/stream",
        data=json.dumps({"query": "access tokens"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as response:
        lines = [
            json.loads(line) for line in response.read().decode().splitlines() if line
        ]
    text = "".join(line["delta"] for line in lines if "delta" in line)
    assert text == "Hello"
    assert lines[-1]["done"] is True
    assert lines[-1]["hits"]


def test_ask_stream_requires_query(base_url: str):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        request(f"{base_url}/api/ask/stream", {})
    assert excinfo.value.code == 400


def test_static_ui_serving_and_traversal_guard(tmp_path: Path):
    ui = tmp_path / "ui"
    (ui / "assets").mkdir(parents=True)
    (ui / "index.html").write_text("<html><body>ragdesk ui shell</body></html>")
    (ui / "assets" / "app.js").write_text("console.log('ui')")
    (tmp_path / "secret.txt").write_text("do not serve me")

    db = tmp_path / "index.db"
    with Store(db) as store:
        index_paths(store, HashingEmbedder(), [FIXTURES / "docs"])
    state = AppState(db=str(db), embedder=HashingEmbedder(), ui_dir=str(ui))
    httpd = make_server(state, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        def raw_get(path: str) -> tuple[int, bytes, str]:
            conn = http.client.HTTPConnection(host, port, timeout=5)
            conn.request("GET", path)
            response = conn.getresponse()
            body = response.read()
            content_type = response.getheader("Content-Type") or ""
            conn.close()
            return response.status, body, content_type

        status, body, content_type = raw_get("/")
        assert status == 200
        assert b"ragdesk ui shell" in body
        assert content_type.startswith("text/html")

        status, body, _ = raw_get("/assets/app.js")
        assert status == 200 and b"console.log" in body

        status, body, _ = raw_get("/../secret.txt")
        assert status == 404, "path traversal must not escape the ui directory"
        assert b"do not serve me" not in body

        status, body, content_type = raw_get("/some/deep/route")
        assert status == 200 and b"ragdesk ui shell" in body
    finally:
        httpd.shutdown()
