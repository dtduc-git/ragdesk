from __future__ import annotations

from pathlib import Path

import pytest

from ragdesk.embed import HashingEmbedder
from ragdesk.notes import NotesError, parse_notes, sync_notes
from ragdesk.store import Store

SAMPLE = """
===RAGDESK NOTE===
Groceries
<html><body><div>milk<br>eggs</div></body></html>
===RAGDESK NOTE===
Prod incident 2026-01-19
<html><body><h1>Summary</h1><p>Kong returned 401 after the migration.</p></body></html>
"""


def test_parse_notes_splits_titles_and_html_bodies():
    notes = parse_notes(SAMPLE)
    assert [title for title, _body in notes] == ["Groceries", "Prod incident 2026-01-19"]
    assert "milk" in notes[0][1] and "eggs" in notes[0][1]
    assert "<html>" not in notes[1][1]
    assert "Kong returned 401" in notes[1][1]


def test_parse_notes_ignores_empty_blocks():
    assert parse_notes("===RAGDESK NOTE===\n\n<html></html>\n") == []


def test_sync_notes_indexes_each_note(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("ragdesk.notes.fetch_notes", lambda runner=None: parse_notes(SAMPLE))
    with Store(tmp_path / "notes.db") as store:
        stats = sync_notes(store, HashingEmbedder(dim=256))
        assert stats.indexed == 2
        paths = {row["path"] for row in store.documents()}
        assert paths == {"notes://Groceries", "notes://Prod incident 2026-01-19"}
        # re-sync is incremental
        stats2 = sync_notes(store, HashingEmbedder(dim=256))
        assert stats2.indexed == 0 and stats2.unchanged == 2


def test_notes_errors_are_actionable(monkeypatch):
    def denied(script: str) -> str:
        raise NotesError(
            "Apple Notes refused the request — allow ragdesk to control Notes in "
            "System Settings → Privacy & Security → Automation"
        )

    # the platform gate would short-circuit before the runner on Linux CI
    monkeypatch.setattr("ragdesk.notes.notes_available", lambda: True)
    monkeypatch.setattr("ragdesk.notes._run_osascript", denied)
    with pytest.raises(NotesError) as excinfo:
        from ragdesk.notes import fetch_notes

        fetch_notes()
    assert "Automation" in str(excinfo.value)


def test_notes_are_refused_off_macos(monkeypatch):
    monkeypatch.setattr("ragdesk.notes.notes_available", lambda: False)
    from ragdesk.notes import fetch_notes

    with pytest.raises(NotesError) as excinfo:
        fetch_notes()
    assert "only available on macOS" in str(excinfo.value)
