from __future__ import annotations

from pathlib import Path

import pytest

from ragdesk.answer import REFUSAL, answer, answer_stream, build_prompt
from ragdesk.chunk import chunk_text
from ragdesk.embed import HashingEmbedder, get_embedder
from ragdesk.evaluate import evaluate, load_golden
from ragdesk.index import index_paths
from ragdesk.llm import OllamaLLM
from ragdesk.rerank import LexicalReranker, get_reranker
from ragdesk.search import Hit, hybrid_search
from ragdesk.store import EmbedderMismatch, Store

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
