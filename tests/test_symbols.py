from __future__ import annotations

from pathlib import Path

from ragdesk.embed import HashingEmbedder
from ragdesk.index import index_paths
from ragdesk.store import Store
from ragdesk.symbols import find_symbol, parse_symbol_question, symbol_answer


def test_parse_symbol_question_forms():
    assert parse_symbol_question("who calls hybrid_search?") == ("hybrid_search", True)
    assert parse_symbol_question("Who calls `hybrid_search`") == ("hybrid_search", True)
    assert parse_symbol_question("callers of Store") == ("Store", True)
    assert parse_symbol_question("where is find_symbol defined?") == ("find_symbol", True)
    assert parse_symbol_question("where does Store.get get used") == ("get", True)
    assert parse_symbol_question("ai gọi retrieve") == ("retrieve", False)
    # a plain word without a definition is not treated as a symbol at all
    assert parse_symbol_question("who calls the shots here") == ("", False)
    assert parse_symbol_question("what is the deploy process") == ("", False)


def make_corpus(tmp_path: Path) -> Path:
    docs = tmp_path / "src"
    docs.mkdir()
    # single blank lines between blocks: chunk line numbers stay exact
    (docs / "core.py").write_text(
        "def hybrid_search(query):\n"
        "    return query\n"
        "\n"
        "def helper():\n"
        "    return hybrid_search('x')\n"
    )
    (docs / "app.py").write_text(
        "from core import hybrid_search\n"
        "\n"
        "def main():\n"
        "    hits = hybrid_search('q')\n"
        "    return hits\n"
    )
    (docs / "notes.md").write_text("hybrid_search is documented here")
    return docs


def test_find_symbol_defs_calls_and_mentions(tmp_path: Path):
    docs = make_corpus(tmp_path)
    with Store(tmp_path / "index.db") as store:
        index_paths(store, HashingEmbedder(), [docs])
        result = find_symbol(store, "hybrid_search")

    assert [row["path"] for row in result["defs"]] == [str(docs / "core.py")]
    assert result["defs"][0]["line"] == 1
    calls = {(row["path"].split("/")[-1], row["line"]) for row in result["calls"]}
    assert calls == {("core.py", 5), ("app.py", 4)}
    # the import line is a mention; the markdown file is not code at all
    assert [row["path"].split("/")[-1] for row in result["mentions"]] == ["app.py"]

    text, hits = symbol_answer("hybrid_search", result)
    assert "1 definition(s), 2 call site(s)" in text
    assert "core.py:1" in text and "app.py:4" in text
    assert [hit.path.split("/")[-1] for hit in hits] == ["core.py", "app.py"]
    assert all(hit.lanes == "symbol" for hit in hits)


def test_symbol_lookup_is_empty_for_unknown_names(tmp_path: Path):
    docs = make_corpus(tmp_path)
    with Store(tmp_path / "index.db") as store:
        index_paths(store, HashingEmbedder(), [docs])
        result = find_symbol(store, "never_defined_anywhere")
    assert result["defs"] == [] and result["calls"] == []
    assert symbol_answer("never_defined_anywhere", result) == ("", [])


def test_symbol_lookup_ignores_prose_mentions(tmp_path: Path):
    with Store(tmp_path / "index.db") as store:
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "guide.md").write_text("call hybrid_search to find hits")
        index_paths(store, HashingEmbedder(), [docs])
        assert find_symbol(store, "hybrid_search")["calls"] == []
