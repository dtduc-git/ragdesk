"""Generate a golden set from the indexed corpus with the local model.

Samples chunks per source group, asks the model for ONE short question each
chunk answers, and rejects anything that leaks the file name (the path lane
would otherwise make the test trivial). The output is a JSONL file in the same
shape as fixtures/golden_repo.jsonl, so `ragdesk eval` and scripts/bench.py
consume it directly.

    uv run --extra onnx --extra mlx python scripts/make_golden.py \\
        --db ~/.ragdesk/index.db --out fixtures/golden_corpus.jsonl --per-group 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ragdesk.llm import LLMUnavailable, resolve_llm  # noqa: E402
from ragdesk.store import Store  # noqa: E402

QUESTION_PROMPT = """Below is a passage from a personal knowledge base. Write ONE short
question that this passage answers, in the same language as the passage itself.

Rules: do not mention file names, paths or "the passage"; make the question
specific enough that only this passage answers it; reply with the question only.

Passage:
{text}
Question:"""

# group name -> (WHERE fragment, number of queries, needs folder scoping)
GROUPS: dict[str, tuple[str, str, bool]] = {
    "docs-vn": (
        "d.path LIKE '%/Documents/documents/%' AND d.path NOT LIKE '%.pdf'",
        "VN notes",
        False,
    ),
    "docs-pdf": ("d.path LIKE '%/Documents/documents/%.pdf'", "PDF documents", False),
    "images": ("d.path LIKE '%/Documents/images/%'", "screenshots", False),
    "code-ragdesk": (
        "d.path LIKE '%/dtduc-git/ragdesk/%' AND d.path LIKE '%.py'",
        "ragdesk code",
        True,
    ),
    "code-opsrag": ("d.path LIKE '%/opsrag/%.py'", "opsrag code", True),
}


def fold_name_tokens(path: str) -> set[str]:
    stem = Path(path).stem.lower()
    tokens = {part for part in stem.replace("_", "-").split("-") if len(part) > 3}
    return tokens | {stem}


def repos_groups(repos_dir: Path) -> dict[str, tuple[str, str, bool]]:
    """One group per cloned repo, scoped to ``<dir>/<repo>`` (see public_corpus.py)."""
    anchor = repos_dir.name
    groups: dict[str, tuple[str, str, bool]] = {}
    for sub in sorted(path for path in repos_dir.iterdir() if path.is_dir()):
        groups[sub.name] = (
            f"d.path LIKE '%/{anchor}/{sub.name}/%'",
            f"{anchor}/{sub.name}",
            True,
        )
    return groups


def scope_prefix(path: str, anchor: str = "") -> str:
    """Keep repo-mixed queries honest: scope them to the repo the answer lives in."""
    parts = path.split("/")
    if anchor and anchor in parts:
        index = parts.index(anchor)
        if index + 1 < len(parts):
            return f"folder:{anchor}/{parts[index + 1]} "
    if "dtduc-git" in parts:
        index = parts.index("dtduc-git")
        if index + 1 < len(parts):
            return f"folder:dtduc-git/{parts[index + 1]} "
    if "opsrag" in parts:
        return "folder:opsrag "
    return ""


def sample_chunks(store: Store, where: str, limit: int) -> list[dict]:
    rows = store.conn.execute(
        f"""
        SELECT c.text, d.path FROM chunks c JOIN documents d ON d.id = c.doc_id
        WHERE {where} AND length(c.text) >= 400
        ORDER BY c.id
        """
    ).fetchall()
    if not rows:
        return []
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{row['path']}:{row['text'][:80]}".encode()).hexdigest(),
    )
    picked: list[dict] = []
    seen_docs: set[str] = set()
    for row in ranked:
        if row["path"] in seen_docs:
            continue
        seen_docs.add(row["path"])
        picked.append({"path": str(row["path"]), "text": str(row["text"])})
        if len(picked) >= limit:
            break
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(Path.home() / ".ragdesk/index.db"))
    parser.add_argument("--out", default="fixtures/golden_corpus.jsonl")
    parser.add_argument("--per-group", type=int, default=10)
    parser.add_argument("--only", default="", help="comma separated group names")
    parser.add_argument(
        "--repos-dir",
        type=Path,
        default=None,
        help="treat each subdirectory as one group (public corpus; see public_corpus.py)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print without writing")
    args = parser.parse_args()

    try:
        llm = resolve_llm(None)
    except LLMUnavailable as exc:  # pragma: no cover - needs a local model
        print(f"no local model available: {exc}", file=sys.stderr)
        return 2

    groups = repos_groups(args.repos_dir) if args.repos_dir else GROUPS
    anchor = args.repos_dir.name if args.repos_dir else ""
    wanted = {name for name in args.only.split(",") if name.strip()} or set(groups)
    out_rows: list[dict] = []
    with Store(args.db) as store:
        for group, (where, _label, _scope) in groups.items():
            if group not in wanted:
                continue
            for sample in sample_chunks(store, where, args.per_group):
                raw = str(
                    llm.generate(
                        QUESTION_PROMPT.format(text=sample["text"][:1200]),
                        {"num_predict": 80, "temperature": 0.3},
                    )
                ).strip()
                question = raw.splitlines()[0].strip().strip('"').strip()
                if not (12 <= len(question) <= 200) or "/" in question:
                    continue
                # Any name fragment as a substring is a leak: the path lane would
                # otherwise answer the question for free.
                if any(token in question.lower() for token in fold_name_tokens(sample["path"])):
                    continue
                if any(existing["query"].endswith(question) for existing in out_rows):
                    continue
                out_rows.append(
                    {
                        "query": question,
                        "relevant": [sample["path"]],
                        "category": group,
                    }
                )
                print(f"[{group}] {question}")

    rows = [
        {
            "query": scope_prefix(row["relevant"][0], anchor) + row["query"],
            "relevant": row["relevant"],
            "category": row["category"],
        }
        for row in out_rows
    ]
    if args.dry_run:
        print(f"--- dry run: {len(rows)} queries")
        return 0
    target = Path(args.out)
    target.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    print(f"wrote {len(rows)} queries to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
