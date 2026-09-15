"""Apple Notes connector (macOS): export through AppleScript, index as notes://.

The first run triggers the macOS Automation prompt ("ragdesk wants to control
Notes"); denying it is a normal outcome and surfaces as a clear error. Nothing
leaves the machine — the script reads the local Notes database through Apple's
own automation layer.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Iterator
from typing import Any

from ragdesk.embed import Embedder
from ragdesk.htmlutil import html_to_text
from ragdesk.index import IndexStats, index_document
from ragdesk.store import Store

NOTES_SCRIPT = """
tell application "Notes"
    set output to ""
    repeat with noteItem in notes
        set output to output & "===RAGDESK NOTE===" & linefeed
        set output to output & (name of noteItem) & linefeed
        set output to output & (body of noteItem) & linefeed
    end repeat
    return output
end tell
"""

SEPARATOR = "===RAGDESK NOTE==="


class NotesError(RuntimeError):
    """Apple Notes could not be read (missing, denied, or not macOS)."""


def notes_available() -> bool:
    return sys.platform == "darwin"


def _run_osascript(script: str) -> str:
    try:
        completed = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise NotesError(f"cannot run AppleScript: {exc}") from exc
    if completed.returncode != 0:
        raise NotesError(
            "Apple Notes refused the request — allow ragdesk to control Notes in "
            f"System Settings → Privacy & Security → Automation ({completed.stderr.strip()})"
        )
    return completed.stdout


def parse_notes(raw: str) -> list[tuple[str, str]]:
    """``(title, body text)`` per exported note; the body arrives as HTML."""
    notes: list[tuple[str, str]] = []
    for block in raw.split(SEPARATOR)[1:]:
        lines = block.strip().splitlines()
        if not lines:
            continue
        title = lines[0].strip() or "untitled note"
        body = html_to_text("\n".join(lines[1:])).strip()
        if body:
            notes.append((title[:200], body))
    return notes


def fetch_notes(runner: Callable[[str], str] | None = None) -> list[tuple[str, str]]:
    if not notes_available():
        raise NotesError("Apple Notes is only available on macOS")
    return parse_notes((runner or _run_osascript)(NOTES_SCRIPT))


def sync_notes(
    store: Store,
    embedder: Embedder,
    progress: Callable[[str, int, int], None] | None = None,
) -> IndexStats:
    notes = fetch_notes()
    stats = IndexStats()
    total = len(notes)
    for index, (title, body) in enumerate(notes, start=1):
        stats.files_scanned += 1
        if progress is not None:
            progress(f"note: {title[:60]}", index, total)
        chunks = index_document(
            store,
            embedder,
            source="notes",
            path=f"notes://{title}",
            content=body,
        )
        if chunks:
            stats.indexed += 1
            stats.chunks += chunks
        else:
            stats.unchanged += 1
    return stats


def iter_note_paths() -> Iterator[str]:
    """Kept for the CLI listing: yields ``notes://title`` without indexing."""
    for title, _body in fetch_notes():
        yield f"notes://{title}"


def status() -> dict[str, Any]:
    return {"available": notes_available()}
