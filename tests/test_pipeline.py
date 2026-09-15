from __future__ import annotations

from pathlib import Path

import pytest

from ragdesk.answer import REFUSAL, answer, answer_stream, build_prompt
from ragdesk.chunk import chunk_text
from ragdesk.embed import HashingEmbedder, get_embedder
from ragdesk.evaluate import evaluate, load_golden
from ragdesk.index import index_document, index_paths
from ragdesk.llm import OllamaLLM
from ragdesk.rerank import LexicalReranker, get_reranker
from ragdesk.search import Hit, hybrid_search
from ragdesk.store import EmbedderMismatch, Store
from ragdesk.web import WebError, save_page

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def make_store(tmp_path: Path) -> Store:
    return Store(tmp_path / "index.db")


def make_hit(path: str, text: str, chunk_id: int = 1) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        doc_id=1,
        path=path,
        source="local",
        ordinal=0,
        text=text,
        score=1.0,
        cosine=0.0,
        lanes="dense",
    )


def test_chunking_covers_text_and_overlaps():
    text = "\n\n".join(f"paragraph {i} " + "x" * 200 for i in range(10))
    chunks = chunk_text(text, max_chars=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 300 for chunk in chunks)
    joined = " ".join(chunk.text for chunk in chunks)
    for i in range(10):
        assert f"paragraph {i}" in joined
    assert chunks[0].text[-50:] in chunks[1].text


def test_chunking_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello", max_chars=100, overlap=100)


def test_index_skips_heavy_dirs(tmp_path: Path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "note.md").write_text("kept")
    for heavy in ("node_modules/pkg", "target/debug", ".git/objects"):
        (tmp_path / heavy).mkdir(parents=True)
        (tmp_path / heavy / "junk.md").write_text("dropped")
    with Store(tmp_path / "index.db") as store:
        stats = index_paths(store, HashingEmbedder(), [tmp_path])
        assert stats.indexed == 1
        assert [d["path"] for d in store.documents()] == [str(tmp_path / "docs" / "note.md")]
        assert store.local_paths()[0]["path"] == str(tmp_path)


def test_index_and_search_end_to_end(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "auth.md").write_text(
        "Access tokens expire after 60 minutes. Use refresh tokens to renew. PKCE is required."
    )
    (docs / "deploy.md").write_text(
        "Deploys use Terraform and Helm. Rollback with kubectl rollout undo if canary fails."
    )
    embedder = HashingEmbedder()
    with make_store(tmp_path) as store:
        stats = index_paths(store, embedder, [docs])
        assert stats.indexed == 2
        hits = hybrid_search(store, embedder, "how do access tokens refresh", top_k=3)
        assert hits
        assert hits[0].path.endswith("auth.md")


def test_reindex_is_incremental(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    first = docs / "a.md"
    second = docs / "b.md"
    first.write_text("alpha bravo charlie")
    second.write_text("delta echo foxtrot")
    embedder = HashingEmbedder()
    with make_store(tmp_path) as store:
        index_paths(store, embedder, [docs])
        stats = index_paths(store, embedder, [docs])
        assert stats.indexed == 0
        assert stats.unchanged == 2
        first.write_text("alpha bravo charlie delta")
        stats = index_paths(store, embedder, [docs])
        assert stats.indexed == 1
        assert stats.unchanged == 1


def test_embedder_mismatch_is_fail_closed(tmp_path: Path):
    with make_store(tmp_path) as store:
        index_paths(store, HashingEmbedder(dim=256), [FIXTURES / "docs"])
        with pytest.raises(EmbedderMismatch):
            index_paths(store, HashingEmbedder(dim=64), [FIXTURES / "docs"])


def test_orphan_fts_rows_are_pruned_on_open(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    note = docs / "note.md"
    note.write_text("alpha bravo charlie")
    db = tmp_path / "index.db"
    embedder = HashingEmbedder()
    with Store(db) as store:
        index_paths(store, embedder, [docs])
        orphan = int(store.conn.execute("SELECT MAX(id) FROM chunks").fetchone()[0]) + 1
        store.conn.execute(
            "INSERT INTO chunks_fts (rowid, text) VALUES (?, 'ghost')", (orphan,)
        )
        store.conn.commit()

    with Store(db) as store:  # opening the store prunes the ghost row
        assert store.conn.execute(
            "SELECT 1 FROM chunks_fts WHERE rowid = ?", (orphan,)
        ).fetchone() is None
        note.write_text("alpha bravo charlie delta")
        stats = index_paths(store, embedder, [docs])
        # would raise sqlite3.IntegrityError (the live wedge) without the prune
        assert stats.indexed == 1


def test_save_page_indexes_one_page_and_keeps_its_url(tmp_path: Path, monkeypatch):
    page = (
        "<html><head><title>Rollback notes</title></head>"
        "<body><p>Roll back with kubectl rollout undo when the canary fails.</p></body></html>"
    )
    monkeypatch.setattr("ragdesk.web._fetch", lambda url, timeout=30.0: page)
    embedder = HashingEmbedder()
    with make_store(tmp_path) as store:
        stats = save_page(
            store, embedder, "https://docs.example.com/runbooks/rollback"
        )
        assert stats.indexed == 1 and stats.skipped == 0
        again = save_page(store, embedder, "https://docs.example.com/runbooks/rollback")
        assert again.unchanged == 1
        pages = store.web_pages()
        assert [page["url"] for page in pages] == [
            "https://docs.example.com/runbooks/rollback"
        ]
        assert pages[0]["path"] == "web://docs.example.com/runbooks/rollback"
        assert store.document_text(pages[0]["path"])


def test_save_page_rejects_a_non_http_url(tmp_path: Path):
    with make_store(tmp_path) as store:
        with pytest.raises(WebError):
            save_page(store, HashingEmbedder(), "file:///etc/passwd")


def test_never_index_patterns_skip_and_report(tmp_path: Path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("alpha bravo")
    (docs / "secrets.md").write_text("an api key lives here")
    monkeypatch.setattr(
        "ragdesk.settings.load",
        lambda path=None: {
            "chunk_chars": 1000,
            "chunk_overlap": 150,
            "never_index": ["*secret*"],
        },
    )
    with make_store(tmp_path) as store:
        stats = index_paths(store, HashingEmbedder(), [docs])
        assert stats.indexed == 1
        assert stats.skipped == 1
        assert stats.skipped_samples[0]["reason"] == "never-index pattern"
        report = store.last_index_report()
        assert report["roots"] == [str(docs)]
        assert report["indexed"] == 1 and report["skipped"] == 1
        assert report["at"]
        assert [d["path"] for d in store.documents()] == [str(docs / "notes.md")]


def test_oldest_documents_and_redaction(tmp_path: Path):
    embedder = HashingEmbedder()
    with make_store(tmp_path) as store:
        for path, text, mtime in (
            ("/docs/old.md", "old", 100.0),
            ("/docs/new.md", "new", 200.0),
            ("/docs/secret.md", "shh", 300.0),
        ):
            index_document(
                store, embedder, source="local", path=path, content=text, mtime=mtime
            )
        assert [row["path"] for row in store.oldest_documents(limit=2)] == [
            "/docs/old.md",
            "/docs/new.md",
        ]
        removed = store.delete_documents_matching(["*secret*"])
        assert removed == ["/docs/secret.md"]
        assert [row["path"] for row in store.documents()] == ["/docs/new.md", "/docs/old.md"]


def test_index_survives_one_bad_file(tmp_path: Path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "good.md").write_text("alpha bravo")
    (docs / "bad.md").write_text("delta echo")
    from ragdesk import index as index_module

    original = index_module.index_document

    def boom(store, embedder, *, path: str, **kwargs):
        if path.endswith("bad.md"):
            raise ValueError("wedged write")
        return original(store, embedder, path=path, **kwargs)

    monkeypatch.setattr("ragdesk.index.index_document", boom)
    with make_store(tmp_path) as store:
        stats = index_paths(store, HashingEmbedder(), [docs])
    assert stats.indexed == 1
    assert stats.skipped == 1
    assert any("wedged write" in row["reason"] for row in stats.skipped_samples)


def test_eval_on_fixtures_is_good(tmp_path: Path):
    embedder = HashingEmbedder()
    with make_store(tmp_path) as store:
        index_paths(store, embedder, [FIXTURES / "docs"])
        golden = load_golden(FIXTURES / "golden.jsonl")
        metrics, per_query = evaluate(store, embedder, golden, top_k=10)
        assert metrics["recall@5"] >= 0.8, per_query
        assert metrics["mrr@10"] >= 0.8, per_query


def test_eval_multiturn_rewrites_followups_with_history(tmp_path: Path):
    embedder = HashingEmbedder()
    with make_store(tmp_path) as store:
        index_paths(store, embedder, [FIXTURES / "docs"])
        golden = load_golden(FIXTURES / "golden_multiturn.jsonl")
        assert golden and all(row.get("history") for row in golden)

        raw_metrics, raw_rows = evaluate(store, embedder, golden, top_k=10)
        assert all(row["rewritten"] is False for row in raw_rows)
        assert all(row["search_query"] == row["query"] for row in raw_rows)

        seen: list[tuple[str, int]] = []

        def rewrite(question: str, history: list[tuple[str, str]]) -> str:
            seen.append((question, len(history)))
            return history[0][1]  # the opening turn names the topic

        metrics, rows = evaluate(store, embedder, golden, top_k=10, rewrite_for=rewrite)
        assert len(seen) == len(golden)
        assert all(row["rewritten"] for row in rows)
        assert metrics["recall@5"] >= 0.8, rows
        assert metrics["recall@5"] >= raw_metrics["recall@5"], (raw_metrics, metrics)


class FakeLLM:
    """Test double for a ragdesk.llm backend: never touches the network."""

    kind = "fake"
    model = "fake"
    host = "fake"

    def __init__(self, replies: list[str] | None = None, pieces: list[str] | None = None):
        self.replies = list(replies) if replies is not None else ["ok"]
        self.pieces = list(pieces or [])
        self.calls = 0
        self.options: list[dict] = []

    def generate(self, prompt: str, options: dict) -> str:
        self.calls += 1
        self.options.append(dict(options))
        return self.replies.pop(0) if self.replies else ""

    def generate_stream(self, prompt: str, options: dict):
        self.calls += 1
        self.options.append(dict(options))
        yield from self.pieces


def test_grounding_gate_skips_llm_when_weak():
    hit = Hit(
        chunk_id=1,
        doc_id=1,
        path="docs/a.md",
        source="local",
        ordinal=0,
        text="some context",
        score=1.0,
        cosine=0.1,
        lanes="dense",
    )
    assert answer("q", [hit], FakeLLM(), min_cosine=0.5) == REFUSAL
    assert answer("q", [], FakeLLM(), min_cosine=0.0) == REFUSAL
    prompt = build_prompt("what is the plan?", [hit])
    assert "what is the plan?" in prompt
    assert "docs/a.md" in prompt
    assert "some context" in prompt


def test_lexical_reranker_reorders_by_overlap():
    hits = [
        make_hit("a.md", "unrelated words here", chunk_id=1),
        make_hit("b.md", "rollback canary deploy", chunk_id=2),
        make_hit("c.md", "deploy notes", chunk_id=3),
    ]
    reranked = LexicalReranker().rerank("how to rollback a bad deploy", hits)
    assert reranked[0].path == "b.md"
    assert reranked[-1].path == "a.md"


def test_get_reranker_specs():
    assert get_reranker("none") is None
    assert get_reranker("lexical").name == "lexical"
    assert get_reranker("fastembed").name == "fastembed:BAAI/bge-reranker-base"
    assert get_reranker("fastembed:custom/model").model == "custom/model"
    assert get_reranker("onnx").name.startswith("onnx:onnx-community/gte-multilingual")
    assert get_reranker("onnx:custom/repo").repo == "custom/repo"
    with pytest.raises(ValueError):
        get_reranker("nope")


def test_eval_with_lexical_reranker(tmp_path: Path):
    embedder = HashingEmbedder()
    with make_store(tmp_path) as store:
        index_paths(store, embedder, [FIXTURES / "docs"])
        golden = load_golden(FIXTURES / "golden.jsonl")
        metrics, per_query = evaluate(
            store, embedder, golden, reranker=LexicalReranker()
        )
        assert metrics["recall@5"] >= 0.8, per_query


def test_get_embedder_specs():
    assert get_embedder("hash:128").dim == 128
    assert get_embedder("hash").dim == 4096
    assert get_embedder("ollama:embeddinggemma:300m").name == "ollama:embeddinggemma:300m"
    assert get_embedder("onnx").name.startswith("onnx:onnx-community/embeddinggemma")
    assert get_embedder("onnx:some/repo").repo == "some/repo"
    with pytest.raises(ValueError):
        get_embedder("not-a-spec")


def test_embed_query_matches_embed_for_single_text():
    embedder = HashingEmbedder(dim=64)
    assert embedder.embed_query("hello world") == embedder.embed(["hello world"])[0]


def test_answer_stream_gate_skips_llm():
    hit = make_hit("a.md", "some context", chunk_id=1)
    llm = FakeLLM(pieces=["never called"])
    assert list(answer_stream("q", [hit], llm, min_cosine=0.99)) == [REFUSAL]
    assert list(answer_stream("q", [], llm)) == [REFUSAL]
    assert llm.calls == 0


def test_ollama_backend_sends_grounded_payload(monkeypatch):
    captured: dict = {}

    def fake_post_json(host: str, path: str, payload: dict, timeout: float = 300.0) -> dict:
        captured.update(payload)
        return {"response": "ok"}

    monkeypatch.setattr("ragdesk.llm.post_json", fake_post_json)
    hit = make_hit("a.md", "some context", chunk_id=1)
    assert answer("q", [hit], OllamaLLM("test-tag", "http://127.0.0.1:9")) == "ok"
    assert captured["model"] == "test-tag"
    assert captured["think"] is False
    assert captured["stream"] is False
    assert captured["options"]["num_predict"] == 400
    assert captured["options"]["num_ctx"] == 8192
    assert "bullets" in captured["prompt"]  # the answer stays scannable


def test_answer_retries_empty_response_then_refuses():
    hit = make_hit("a.md", "some context", chunk_id=1)
    llm = FakeLLM(replies=["   ", "   "])
    assert answer("q", [hit], llm) == REFUSAL
    assert llm.calls == 2


def test_answer_stream_maps_pieces():
    hit = make_hit("a.md", "some context", chunk_id=1)
    llm = FakeLLM(pieces=["Hel", "", "lo"])
    assert list(answer_stream("q", [hit], llm)) == ["Hel", "lo"]


def test_answer_stream_refuses_when_model_emits_nothing():
    hit = make_hit("a.md", "some context", chunk_id=1)
    assert list(answer_stream("q", [hit], FakeLLM(pieces=[]))) == [REFUSAL]


def test_corrections_roundtrip_and_nearest(tmp_path: Path):
    embedder = HashingEmbedder()
    with make_store(tmp_path) as store:
        key = embedder.embed_query("when do access tokens expire")
        far = embedder.embed_query("what is the on-call escalation path")
        assert store.corrections_revision() == "0:0"
        correction_id = store.add_correction("when do access tokens expire", "90 minutes [1]", key)
        rows = store.corrections()
        assert [row["id"] for row in rows] == [correction_id]
        assert rows[0]["answer"] == "90 minutes [1]"
        assert store.nearest_correction(key, 0.88)["answer"] == "90 minutes [1]"
        assert store.nearest_correction(far, 0.88) is None
        assert store.corrections_revision() != "0:0"
        store.delete_correction(correction_id)
        assert store.corrections() == []
        assert store.nearest_correction(key, 0.88) is None


def test_build_prompt_injects_corrections():
    hit = make_hit("docs/auth.md", "tokens expire after 60 minutes", chunk_id=1)
    corrections = [{"question": "when do tokens expire", "answer": "after 90 minutes [1]"}]
    prompt = build_prompt("when do tokens expire", [hit], corrections=corrections)
    assert "the user fixed an earlier answer" in prompt
    assert "after 90 minutes [1]" in prompt
    # the correction rides right before the question: small models follow that
    assert prompt.index("the user fixed an earlier answer") > prompt.index("Sources:")
    assert prompt.index("after 90 minutes [1]") < prompt.index("Question: when do tokens expire")
    plain = build_prompt("when do tokens expire", [hit])
    assert "the user fixed an earlier answer" not in plain
