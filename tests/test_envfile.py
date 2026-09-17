from __future__ import annotations

import os
from pathlib import Path

from ragdesk.envfile import load_env_file, parse_env_file


def test_parse_env_file_handles_comments_quotes_export():
    text = "# comment\n\n export FOO=bar\nQUOTED=\"a b\"\nSINGLE='c'\nNO_VALUE\n"
    assert parse_env_file(text) == {"FOO": "bar", "QUOTED": "a b", "SINGLE": "c"}


def test_load_env_file_sets_missing_only(tmp_path: Path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("RAGDESK_TEST_A=one\nRAGDESK_TEST_B=two\n")
    monkeypatch.setenv("RAGDESK_TEST_B", "preexisting")
    monkeypatch.delenv("RAGDESK_TEST_A", raising=False)
    try:
        loaded = load_env_file(env)
        assert loaded == 1
        assert os.environ["RAGDESK_TEST_A"] == "one"
        assert os.environ["RAGDESK_TEST_B"] == "preexisting"
    finally:
        os.environ.pop("RAGDESK_TEST_A", None)


def test_load_env_file_missing_returns_zero(tmp_path: Path):
    assert load_env_file(tmp_path / "nope.env") == 0
