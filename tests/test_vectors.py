from __future__ import annotations

from pathlib import Path

import pytest

from ragdesk import vectors
from ragdesk.embed import HashingEmbedder
from ragdesk.index import index_document
from ragdesk.search import Filters
from ragdesk.store import Store

DOCS = {
    "notes/alpha.md": "alpha beta gamma",
    "notes/delta.md": "delta epsilon zeta",
    "other/eta.md": "eta theta iota",
    "other/kappa.md": "kappa lambda mu",
}


@pytest.fixture(autouse=True)
def fresh_cache():
    vectors.clear_cache()
    yield
    vectors.clear_cache()


def make_store(tmp_path: Path, backend: str = "") -> tuple[Store, HashingEmbedder]:
    store = Store(tmp_path / "index.db", vector_backend=backend)
    embedder = HashingEmbedder(dim=64)
    for path, text in DOCS.items():
        index_document(store, embedder, source="local", path=path, content=text)
    return store, embedder


def ranked_paths(store: Store, vec: list[float], limit: int = 5, filters=None) -> list[str]:
    return [row["path"] for row in store.dense_search(vec, limit, filters=filters)]


def test_numpy_matches_python_scan(tmp_path):
    python_store, embedder = make_store(tmp_path, backend="python")
    numpy_store, _ = make_store(tmp_path, backend="numpy")
    for query in ["alpha beta", "delta", "kappa lambda", "theta"]:
        vec = embedder.embed_query(query)
        assert ranked_paths(python_store, vec) == ranked_paths(numpy_store, vec)


def test_numpy_matches_python_scan_with_filters(tmp_path):
    python_store, embedder = make_store(tmp_path, backend="python")
    numpy_store, _ = make_store(tmp_path, backend="numpy")
    filters = Filters(path_like="notes/")
    vec = embedder.embed_query("alpha delta epsilon")
    expected = ranked_paths(python_store, vec, filters=filters)
    assert expected
    assert all(path.startswith("notes/") for path in expected)
    assert ranked_paths(numpy_store, vec, filters=filters) == expected


def test_usearch_approximation_keeps_the_top_hit(tmp_path):
    if vectors._usearch() is None:
        pytest.skip("usearch is not installed")
    python_store, embedder = make_store(tmp_path, backend="python")
    usearch_store, _ = make_store(tmp_path, backend="usearch")
    for query in ["alpha beta", "eta theta", "lambda"]:
        vec = embedder.embed_query(query)
        exact = ranked_paths(python_store, vec, 3)
        approx = ranked_paths(usearch_store, vec, 3)
        assert approx[0] == exact[0]
        assert len(set(approx) & set(exact)) >= 2


def test_backend_cache_follows_writes_from_another_connection(tmp_path):
    db = tmp_path / "index.db"
    store = Store(db, vector_backend="numpy")
    embedder = HashingEmbedder(dim=64)
    index_document(store, embedder, source="local", path="a.md", content="alpha beta")
    vec = embedder.embed_query("zebra")
    assert ranked_paths(store, vec, 3) == ["a.md"]

    writer = Store(db)
    index_document(writer, embedder, source="local", path="zebra.md", content="zebra stripes")
    writer.close()
    assert ranked_paths(store, vec, 3)[0] == "zebra.md"

    remover = Store(db)
    remover.delete_documents_matching(["zebra.md"])
    remover.close()
    assert "zebra.md" not in ranked_paths(store, vec, 3)


def test_policy_prefers_explicit_then_numpy(tmp_path):
    store, _ = make_store(tmp_path)
    assert vectors.pick_backend(store.conn, "python") == "python"
    assert vectors.pick_backend(store.conn, "numpy") == "numpy"
    assert vectors.pick_backend(store.conn, "usearch") == "usearch"
    default = "numpy" if vectors._numpy() is not None else "python"
    assert vectors.pick_backend(store.conn) == default
    assert vectors.pick_backend(store.conn, "auto") == default
    with pytest.raises(ValueError):
        vectors.pick_backend(store.conn, "bogus")


def test_dense_search_keeps_the_payload_contract(tmp_path):
    store, embedder = make_store(tmp_path, backend="numpy")
    rows = store.dense_search(embedder.embed_query("alpha"), 2)
    assert rows
    first = rows[0]
    assert isinstance(first["metadata"], dict)
    assert set(first) == {
        "id",
        "doc_id",
        "ordinal",
        "text",
        "path",
        "source",
        "parent_text",
        "line_start",
        "mtime",
        "metadata",
        "score",
    }
    assert store.dense_search(embedder.embed_query("alpha"), 0) == []
    no_match = Filters(source="nope")
    assert store.dense_search(embedder.embed_query("alpha"), 2, filters=no_match) == []
