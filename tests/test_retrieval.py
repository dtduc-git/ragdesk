from __future__ import annotations

from pathlib import Path

from ragdesk.embed import HashingEmbedder
from ragdesk.index import index_document, index_paths
from ragdesk.search import Filters, Hit, hybrid_search, parse_filters, retrieve
from ragdesk.store import Store


def make_store(tmp_path: Path, docs: dict[str, str]) -> Store:
    store = Store(tmp_path / "index.db")
    embedder = HashingEmbedder(dim=512)
    for name, text in docs.items():
        index_document(store, embedder, source="local", path=name, content=text)
    return store


def hit(path: str, ids: int = 1, parent: str = "") -> Hit:
    return Hit(
        chunk_id=ids,
        doc_id=ids,
        path=path,
        source="local",
        ordinal=0,
        text="child text",
        score=1.0,
        cosine=0.5,
        lanes="dense",
        parent_text=parent,
    )


def test_parse_filters_takes_folder_and_source():
    query, filters = parse_filters("folder:Financial mã số thuế")
    assert query == "mã số thuế"
    assert filters.path_like == "Financial"
    assert filters.source == ""

    query, filters = parse_filters("source:github deploy rollback")
    assert query == "deploy rollback"
    assert filters.source == "github"

    query, filters = parse_filters("what is rrf fusion?")
    assert query == "what is rrf fusion?"
    assert not filters


def test_hit_context_prefers_parent():
    assert hit("a.md", parent="big parent section").context == "big parent section"
    assert hit("a.md").context == "child text"


def test_path_lane_finds_files_by_name(tmp_path: Path):
    with make_store(
        tmp_path,
        {
            "/docs/RRF-fusion-notes.md": "some unrelated body text",
            "/docs/other.md": "totally different words here",
        },
    ) as store:
        rows = store.path_search("rrf fusion notes", 10)
        assert rows
        assert rows[0]["path"].endswith("RRF-fusion-notes.md")

        hits = hybrid_search(
            store, HashingEmbedder(dim=512), "rrf fusion notes", top_k=5
        )
        assert hits
        assert any("RRF" in h.path for h in hits)


def test_folder_and_source_filters_scope_every_lane(tmp_path: Path):
    with make_store(
        tmp_path,
        {
            "/notes/finance/tax.md": "thuế thu nhập cá nhân khấu trừ",
            "/notes/tech/k8s.md": "thuế thu nhập cá nhân khấu trừ",
        },
    ) as store:
        unfiltered = store.bm25_search("thuế thu nhập", 10)
        assert len({row["path"] for row in unfiltered}) == 2

        scoped = store.bm25_search("thuế thu nhập", 10, filters=Filters(path_like="finance"))
        assert [row["path"] for row in scoped] == ["/notes/finance/tax.md"]

        vec = HashingEmbedder(dim=512).embed_query("thuế thu nhập")
        dense = store.dense_search(vec, 10, filters=Filters(path_like="finance"))
        assert [row["path"] for row in dense] == ["/notes/finance/tax.md"]

        hits = retrieve(
            store,
            HashingEmbedder(dim=512),
            "thuế thu nhập",
            top_k=5,
            filters=Filters(path_like="finance"),
        )
        assert {h.path for h in hits} == {"/notes/finance/tax.md"}


def test_parents_are_stored_and_returned(tmp_path: Path):
    long_text = "\n\n".join(f"paragraph {i} " + "word " * 60 for i in range(12))
    with make_store(tmp_path, {"/notes/big.md": long_text}) as store:
        rows = store.conn.execute("SELECT COUNT(*) AS n FROM parents").fetchone()
        assert rows["n"] >= 2  # grouped into parent sections
        first = store.bm25_search("paragraph", 1)
        assert first and first[0]["parent_text"]
        assert len(first[0]["parent_text"]) > len(first[0]["text"])


def test_semantic_cache_roundtrip(tmp_path: Path):
    with Store(tmp_path / "cache.db") as store:
        store.cache_put(
            "k1",
            "what is the grounding gate?",
            "an answer",
            [{"path": "a.md"}],
            embedding=[1.0, 0.0, 0.0],
            fingerprint="fp1",
        )
        hit = store.cache_nearest([0.99, 0.01, 0.0], "fp1", 0.88)
        assert hit is not None
        assert hit["answer"] == "an answer"
        assert hit["cosine"] > 0.99

        # a different corpus generation never reuses the answer
        assert store.cache_nearest([1.0, 0.0, 0.0], "fp2", 0.88) is None
        # a different question stays below the threshold
        assert store.cache_nearest([0.0, 1.0, 0.0], "fp1", 0.88) is None
        # rows without an embedding are ignored by the semantic path
        store.cache_put("k2", "no vector", "x", [], fingerprint="fp1")
        assert store.cache_nearest([0.0, 1.0, 0.0], "fp1", 0.0)["answer"] == "an answer"


def test_hyde_lane_can_change_the_ranking(tmp_path: Path):
    with make_store(
        tmp_path,
        {
            "/notes/a.md": "alpha beta gamma",
            "/notes/b.md": "delta epsilon zeta",
        },
    ) as store:
        query = "gamma"
        plain = hybrid_search(store, HashingEmbedder(dim=512), query, top_k=2)
        with_hyde = hybrid_search(
            store,
            HashingEmbedder(dim=512),
            query,
            top_k=2,
            hyde_vec=HashingEmbedder(dim=512).embed_query("delta epsilon zeta"),
        )
        assert plain and with_hyde
        assert any(h.lanes == "dense+hyde" or "hyde" in h.lanes for h in with_hyde)


def test_category_metrics_group_rows():
    from ragdesk.evaluate import category_metrics

    per_query = [
        {"category": "tax", "recall@5": 1.0, "ndcg@10": 1.0, "mrr@10": 1.0},
        {"category": "tax", "recall@5": 0.5, "ndcg@10": 0.5, "mrr@10": 0.5},
        {"category": "k8s", "recall@5": 0.0, "ndcg@10": 0.0, "mrr@10": 0.0},
    ]
    grouped = category_metrics(per_query)
    assert grouped["tax"]["recall@5"] == 0.75
    assert grouped["tax"]["queries"] == 2.0
    assert grouped["k8s"]["recall@5"] == 0.0


def test_ground_answer_scores_sentences_and_citations():
    from ragdesk.evaluate import ground_answer

    citations = ["Access tokens expire after 60 minutes and clients refresh them."]
    good = ground_answer("Access tokens expire after 60 minutes [1].", citations)
    assert good["citation_valid"] is True
    assert good["grounded_ratio"] == 1.0

    invented = ground_answer("Tokens never expire and the sky is green [1].", citations)
    assert invented["citation_valid"] is True  # the ref is in range
    assert invented["grounded_ratio"] == 0.0

    out_of_range = ground_answer("Something true [7].", citations)
    assert out_of_range["citation_valid"] is False

    no_citations = ground_answer("An answer without any reference.", citations)
    assert no_citations["citation_valid"] is False

    empty = ground_answer("", citations)
    assert empty == {"citation_valid": False, "grounded_ratio": 0.0, "sentences": 0}


def test_evaluate_groups_categories_and_accepts_hyde(tmp_path: Path):
    from ragdesk.evaluate import category_metrics, evaluate

    with make_store(
        tmp_path,
        {
            "/notes/finance/tax.md": "thuế thu nhập cá nhân khấu trừ tại nguồn",
            "/notes/tech/rrf.md": "rrf fusion ranks by reciprocal rank",
        },
    ) as store:
        golden = [
            {"query": "thuế thu nhập", "relevant": ["tax.md"], "category": "tax"},
            {"query": "rrf fusion", "relevant": ["rrf.md"], "category": "retrieval"},
        ]
        calls: list[str] = []

        def hyde_for(text: str) -> str:
            calls.append(text)
            return "rrf fusion ranks by reciprocal rank"

        metrics, per_query = evaluate(
            store, HashingEmbedder(dim=512), golden, hyde_for=hyde_for
        )
        assert calls == ["thuế thu nhập", "rrf fusion"]
        assert metrics["recall@5"] == 1.0
        grouped = category_metrics(per_query)
        assert set(grouped) == {"tax", "retrieval"}

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.md").write_text("\n\n".join("line " * 50 for _ in range(10)))
    with Store(tmp_path / "db.sqlite") as store:
        index_paths(store, HashingEmbedder(dim=256), [docs])
        parents = store.conn.execute("SELECT COUNT(*) AS n FROM parents").fetchone()
        assert parents["n"] >= 1
        rows = store.conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE parent_ordinal IS NULL"
        ).fetchone()
        assert rows["n"] == 0


def test_index_paths_creates_parents_for_new_docs(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.md").write_text("\n\n".join("line " * 50 for _ in range(10)))
    with Store(tmp_path / "db.sqlite") as store:
        index_paths(store, HashingEmbedder(dim=256), [docs])
        parents = store.conn.execute("SELECT COUNT(*) AS n FROM parents").fetchone()
        assert parents["n"] >= 1
        rows = store.conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE parent_ordinal IS NULL"
        ).fetchone()
        assert rows["n"] == 0


def test_wants_diagram_detection():
    from ragdesk.answer import wants_diagram

    assert wants_diagram("draw a diagram of the pipeline")
    assert wants_diagram("vẽ sơ đồ kiến trúc ragdesk")
    assert wants_diagram("show me a flowchart of the auth flow")
    assert not wants_diagram("what does the grounding gate do?")


def test_prompt_includes_diagram_rules_only_when_asked(tmp_path: Path):
    from ragdesk.answer import DIAGRAM_NOTE, build_prompt

    plain = build_prompt("what is rrf?", [hit("a.md")])
    asked = build_prompt("vẽ sơ đồ rrf", [hit("a.md")], diagram=True)
    assert DIAGRAM_NOTE.splitlines()[0] not in plain
    assert "```mermaid" in asked
    assert "accent" in asked


def test_chunk_text_tracks_line_starts():
    from ragdesk.chunk import chunk_text

    text = "line one\n\nline three\n\n" + "x" * 50
    chunks = chunk_text(text, max_chars=200, overlap=20)
    assert chunks[0].line_start == 1
    assert all(chunk.line_start >= 1 for chunk in chunks)

    # a hard-split paragraph keeps pointing at its own line
    big = ("y" * 120) + "\n\n"
    parts = chunk_text("head\n\n" + big * 2, max_chars=100, overlap=10)
    assert any(part.line_start == 3 for part in parts)


def test_diversify_caps_chunks_per_document():
    from ragdesk.search import diversify

    # three chunks of one doc followed by two of another: the cap keeps the
    # second document in the result instead of letting the first fill it.
    def chunk(path: str, doc_id: int, chunk_id: int) -> Hit:
        return Hit(
            chunk_id=chunk_id,
            doc_id=doc_id,
            path=path,
            source="local",
            ordinal=chunk_id,
            text="body",
            score=1.0,
            cosine=0.5,
            lanes="dense",
        )

    hits = [chunk("big.md", 1, index) for index in range(1, 4)]
    hits += [chunk("other.md", 2, index) for index in range(4, 6)]
    picked = diversify(hits, top_k=4)
    assert sum(1 for item in picked if item.path == "big.md") == 2
    assert sum(1 for item in picked if item.path == "other.md") == 2
    # with nothing else to fill from, the cap yields rather than returning fewer hits
    only_one = [chunk("same.md", 3, index) for index in range(1, 6)]
    assert len(diversify(only_one, top_k=3)) == 3


def test_recency_factor_decays():
    import time

    from ragdesk.search import recency_factor

    now = time.time()
    assert recency_factor(now, now=now) > 1.09
    assert recency_factor(now - 45 * 86_400, now=now) < 1.05
    assert recency_factor(now - 365 * 86_400, now=now) < 1.01
    assert recency_factor(0.0, now=now) == 1.0


def test_parse_smart_retrieval_tolerates_noise():
    from ragdesk.serve import parse_smart_retrieval

    parsed = parse_smart_retrieval(
        'Sure! {"standalone": "how does rrf rank?", "sub_queries": ["rrf fusion", "rank"], '
        '"hypothetical": "RRF sums 1/(k+rank)."} done'
    )
    assert parsed["standalone"] == "how does rrf rank?"
    assert parsed["sub_queries"] == ["rrf fusion", "rank"]
    assert "RRF sums" in parsed["hypothetical"]
    assert parse_smart_retrieval("no json") == {}
    assert parse_smart_retrieval('{"standalone": "none", "sub_queries": []}') == {}


def test_ground_answer_detail_marks_sentences():
    from ragdesk.evaluate import ground_answer_detail

    citations = ["Access tokens expire after 60 minutes and refresh tokens renew them."]
    detail = ground_answer_detail(
        "Access tokens expire after 60 minutes [1]. The moon is made of cheese.",
        citations,
    )
    assert detail["citation_valid"] is True
    assert detail["sentences_detail"][0]["grounded"] is True
    assert detail["sentences_detail"][1]["grounded"] is False
    assert detail["sentences_detail"][0]["citations"] == [1]


def test_fold_text_strips_vietnamese_diacritics():
    from ragdesk.store import fold_text

    assert fold_text("Thuế Thu Nhập Cá Nhân") == "thue thu nhap ca nhan"
    assert fold_text("ĐÀ NẴNG") == "da nang"
    assert fold_text("Kubernetes") == "kubernetes"


def test_accent_insensitive_bm25(tmp_path: Path):
    with make_store(
        tmp_path,
        {
            "/notes/thue.md": "Thuế thu nhập cá nhân khấu trừ tại nguồn cho năm 2025",
            "/notes/other.md": "something entirely different about kubernetes",
        },
    ) as store:
        folded = store.bm25_search("thue thu nhap", 5)
        assert folded and folded[0]["path"].endswith("thue.md")
        accented = store.bm25_search("thuế thu nhập", 5)
        assert accented and accented[0]["path"].endswith("thue.md")
        # the stored text stays unfolded for citations
        assert "Thuế" in str(folded[0]["text"])


def test_front_matter_becomes_metadata_and_filters(tmp_path: Path):
    from ragdesk.index import parse_front_matter
    from ragdesk.search import parse_filters, retrieve

    meta, body = parse_front_matter(
        "---\nservice: payments\ntype: runbook\nweird key: dropped\n---\n# Rollback\nsteps"
    )
    assert meta == {"service": "payments", "type": "runbook"}
    assert body.startswith("# Rollback")
    assert parse_front_matter("no header here") == ({}, "no header here")

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "runbook.md").write_text("---\nservice: payments\ntype: runbook\n---\nrollback steps")
    (docs / "chat.md").write_text("---\nservice: payments\ntype: chatlog\n---\nrollback steps")
    (docs / "other.md").write_text("---\nservice: search\n---\nrollback steps")
    with Store(tmp_path / "meta.db") as store:
        index_paths(store, HashingEmbedder(dim=256), [docs])
        rows = {row["path"]: row["metadata"] for row in store.documents()}
        assert rows[str(docs / "runbook.md")]["type"] == "runbook"

        query, filters = parse_filters("type:runbook rollback")
        assert query == "rollback"
        assert filters.meta == {"type": "runbook"}
        hits = retrieve(store, HashingEmbedder(dim=256), query, top_k=5, filters=filters)
        assert {hit.path for hit in hits} == {str(docs / "runbook.md")}

        # a word with a colon (a URL) is not a filter
        query, filters = parse_filters("see https://example.com/x for details")
        assert filters.meta is None and "https://example.com/x" in query


def test_watcher_fast_path_and_touch(tmp_path: Path):
    from ragdesk.index import index_paths

    docs = tmp_path / "docs"
    docs.mkdir()
    note = docs / "note.md"
    note.write_text("alpha bravo")
    embedder = HashingEmbedder(dim=128)
    with Store(tmp_path / "watch.db") as store:
        first = index_paths(store, embedder, [docs])
        assert first.indexed == 1

        second = index_paths(store, embedder, [docs])
        assert second.unchanged == 1 and second.indexed == 0

        # a touch with identical content refreshes the stored mtime
        note.touch()
        third = index_paths(store, embedder, [docs])
        assert third.indexed == 0 and third.unchanged == 1
        assert store.doc_mtime(str(note)) is not None

        # skipped files carry a reason now
        (docs / "photo.raw").write_bytes(b"\x00binary")
        fourth = index_paths(store, embedder, [docs])
        assert fourth.skipped >= 1
        assert any(row["reason"] for row in fourth.skipped_samples)
