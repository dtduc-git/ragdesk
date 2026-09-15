"""Symbol navigation: where something is defined and who calls it.

Deliberately regex-based, on-demand and *outside* retrieval: the graph is for
navigation ("who calls X"), never a ranking lane — the symbol-lane experiment
diluted RRF (see AGENTS.md) and graph/backlink data stays at the display layer.
No AST, no extra dependency, no index-time tables: the scan reads the chunk
text that is already stored, so it works on old indexes without a re-index.

Line numbers are derived from the stored chunk (``line_start`` + offset inside
the chunk). The chunker rejoins paragraphs with a single blank line, so a file
with *runs* of blank lines can report a line one or two lower than the editor
shows; single-blank-line code (the common style) is exact.
"""

from __future__ import annotations

import re
from pathlib import Path

from ragdesk.chunk import SYMBOL_RE
from ragdesk.search import Hit
from ragdesk.store import Store

CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".go", ".rs", ".java", ".kt", ".kts", ".swift", ".cs", ".rb", ".php",
    ".sh", ".bash", ".zsh", ".fish", ".pl", ".lua", ".scala", ".ex", ".exs",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".m", ".mm", ".sql", ".tf",
}
MAX_SITES = 25
MAX_TEXT = 160
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

QUESTION_PATTERNS = (
    r"(?:who|what)\s+calls\s+(.+?)[\s?!.]*$",
    r"callers?\s+of\s+(.+?)[\s?!.]*$",
    r"who\s+uses\s+(.+?)[\s?!.]*$",
    r"where\s+is\s+(.+?)\s+defined",
    r"where\s+does\s+(.+?)\s+(?:get\s+)?(?:defined|called|used)",
    r"(?:ai|đứa nào)\s+(?:gọi|dùng|gọi tới)\s+(.+?)[\s?!.]*$",
    r"(?:hàm|biến|class|module)\s+(.+?)\s+(?:được\s+)?(?:định nghĩa|gọi)",
)
_QUESTION_RES = tuple(re.compile(pattern, re.IGNORECASE) for pattern in QUESTION_PATTERNS)


def parse_symbol_question(question: str) -> tuple[str, bool]:
    """``(name, certain)`` for a "who calls X" style question, else ``("", False)``.

    ``certain=False`` means the name is a plain lowercase word: the caller only
    trusts it when the scan finds an actual definition, so "who calls the shots"
    cannot turn into a lookup for "shots".
    """
    for pattern in _QUESTION_RES:
        match = pattern.search(question.strip())
        if not match:
            continue
        candidate = match.group(1).strip().strip("`'\"()[]{}:;,.")
        name = candidate.rsplit(".", 1)[-1]
        if not IDENTIFIER_RE.fullmatch(name):
            continue
        certain = "_" in name or any(char.isupper() for char in name) or "." in candidate
        return name, certain
    return "", False


def _snippet(line: str) -> str:
    flat = " ".join(line.split())
    return flat[: MAX_TEXT - 1] + "…" if len(flat) > MAX_TEXT else flat


def _true_line(
    path: str, estimate: int, snippet: str, cache: dict[str, list[str]]
) -> int:
    """Map the chunk-relative estimate onto the real file line (local docs only).

    The chunk text rejoins paragraphs, so the estimate can be a few lines low in
    files with runs of blank lines; the file itself is the source of truth.
    """
    lines = cache.get(path)
    if lines is None:
        try:
            lines = Path(path).read_text(errors="replace").split("\n")
        except OSError:
            lines = []
        cache[path] = lines
    if not lines:
        return estimate
    target = snippet.rstrip("…")
    index = estimate - 1
    window = 40
    candidates = sorted(
        range(max(0, index - window), min(len(lines), index + window)),
        key=lambda position: abs(position - index),
    )
    for position in candidates:
        if " ".join(lines[position].split()).startswith(target):
            return position + 1
    return estimate


def find_symbol(store: Store, name: str, limit: int = MAX_SITES) -> dict:
    """Definitions, call sites and mentioning files for ``name`` in code files."""
    result: dict = {
        "name": name,
        "defs": [],
        "calls": [],
        "mentions": [],
        "scanned": 0,
    }
    if not IDENTIFIER_RE.fullmatch(name):
        return result
    call_re = re.compile(rf"\b{re.escape(name)}\s*\(")
    seen: set[tuple[str, int]] = set()
    mention_files: dict[str, int] = {}
    file_lines: dict[str, list[str]] = {}

    rows = store.conn.execute(
        "SELECT c.id, c.doc_id, c.ordinal, c.text, c.line_start, d.path, d.source "
        "FROM chunks c JOIN documents d ON d.id = c.doc_id"
    ).fetchall()
    for row in rows:
        path = str(row["path"])
        if Path(path).suffix.lower() not in CODE_EXTENSIONS:
            continue
        result["scanned"] += 1
        text = str(row["text"])
        source = str(row["source"])
        for offset, line in enumerate(text.split("\n")):
            if name not in line:
                continue
            line_number = int(row["line_start"]) + offset
            if (path, line_number) in seen:
                continue
            match = SYMBOL_RE.match(line)
            if match and match.group(1) == name:
                kind = "defs"
            elif call_re.search(line):
                kind = "calls"
            else:
                mention_files[path] = mention_files.get(path, 0) + 1
                continue
            snippet = _snippet(line)
            if source == "local":
                line_number = _true_line(path, line_number, snippet, file_lines)
            if (path, line_number) in seen:
                continue
            seen.add((path, line_number))
            if len(result[kind]) >= limit:
                continue
            result[kind].append(
                {
                    "path": path,
                    "line": line_number,
                    "text": snippet,
                    "hit": Hit(
                        chunk_id=int(row["id"]),
                        doc_id=int(row["doc_id"]),
                        path=path,
                        source=source,
                        ordinal=int(row["ordinal"]),
                        text=text,
                        score=1.0,
                        cosine=1.0,
                        lanes="symbol",
                        line=line_number,
                    ),
                }
            )
        if len(result["defs"]) >= limit and len(result["calls"]) >= limit:
            break
    result["mentions"] = [
        {"path": path, "count": count}
        for path, count in sorted(mention_files.items(), key=lambda item: -item[1])[:8]
    ]
    return result


def symbol_answer(name: str, result: dict) -> tuple[str, list[Hit]]:
    """A deterministic answer plus citation hits for a symbol lookup."""
    defs, calls = result["defs"], result["calls"]
    if not defs and not calls:
        return "", []
    hits: list[Hit] = []
    index_of: dict[int, int] = {}

    def mark(site: dict) -> int:
        hit = site["hit"]
        if hit.chunk_id not in index_of:
            hits.append(hit)
            index_of[hit.chunk_id] = len(hits)
        return index_of[hit.chunk_id]

    lines = [f"`{name}` — {len(defs)} definition(s), {len(calls)} call site(s).", ""]
    if defs:
        lines.append("Defined in:")
        lines += [
            f"- {site['path']}:{site['line']} — `{site['text']}` [{mark(site)}]"
            for site in defs
        ]
        lines.append("")
    if calls:
        lines.append("Called from:")
        lines += [
            f"- {site['path']}:{site['line']} — `{site['text']}` [{mark(site)}]"
            for site in calls
        ]
        lines.append("")
    if result["mentions"]:
        mentions = ", ".join(
            f"{item['path']} ({item['count']}×)" for item in result["mentions"][:5]
        )
        lines.append(f"Also mentioned in: {mentions}.")
    return "\n".join(lines).strip(), hits
