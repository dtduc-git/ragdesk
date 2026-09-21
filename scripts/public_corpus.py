"""Build a public multi-repo corpus for retrieval evaluation.

Clones famous public repositories (shallow), indexes a bounded, deterministic
subset of each into one scratch database, and prints what the corpus looks
like. ``scripts/make_golden.py --repos-dir`` then writes questions from that
index; ``ragdesk eval`` scores them.

    uv run --extra onnx python scripts/public_corpus.py clone
    uv run --extra onnx python scripts/public_corpus.py index
    uv run --extra onnx --extra mlx python scripts/make_golden.py \
        --db /tmp/ragdesk-public/public.db --repos-dir /tmp/ragdesk-public/repos \
        --per-group 12 --out fixtures/golden_public.jsonl

Per-repo scores never leave the machine; ``docs/public-corpus.md`` publishes
aggregate numbers and the method. Raw clones and the scratch index are
throwaway (delete /tmp/ragdesk-public when done).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ragdesk.embed import get_embedder  # noqa: E402
from ragdesk.index import index_paths  # noqa: E402
from ragdesk.store import Store  # noqa: E402

# Big, famous, and different from each other: a systems language, a web
# framework, a browser-scale TypeScript app, a data stack, infra tools.
REPOS = [
    "kubernetes/kubernetes",
    "django/django",
    "facebook/react",
    "fastapi/fastapi",
    "golang/go",
    "microsoft/vscode",
    "ggml-org/llama.cpp",
    "langchain-ai/langchain",
    "prometheus/prometheus",
    "helm/helm",
    "hashicorp/terraform",
    "python/cpython",
]

SKIP_DIRS = {
    ".git",
    ".github",  # workflows are boilerplate for questions; docs and code carry the signal
    ".venv",
    "__pycache__",
    "node_modules",
    "vendor",
    "third_party",
    "testdata",
    "dist",
    "build",
    "target",
    "site-packages",
    "out",
    "bin",
}
TEXT_SUFFIXES = {
    ".md",
    ".mdx",
    ".rst",
    ".txt",
    ".py",
    ".go",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".rs",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".java",
    ".rb",
    ".yaml",
    ".yml",
    ".toml",
    ".sh",
}
MIN_BYTES = 400
MAX_BYTES = 300_000


def repo_name(slug: str) -> str:
    return slug.split("/", 1)[1]


def pick_files(root: Path, limit: int) -> list[Path]:
    """Deterministic subset: hash order, bounded size, no vendor trees.

    Deterministic matters more than "best" files: the corpus must rebuild
    identically on another machine so the numbers are comparable.
    """
    candidates: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if not (MIN_BYTES <= size <= MAX_BYTES):
            continue
        candidates.append(path)
    candidates.sort(key=lambda path: hashlib.sha256(str(path).encode()).hexdigest())
    return candidates[:limit]


def clone(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for slug in REPOS:
        target = dest / repo_name(slug)
        if target.exists():
            print(f"  {slug}: already cloned")
            continue
        print(f"  cloning {slug} ...", flush=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--quiet",
                f"https://github.com/{slug}.git",
                str(target),
            ],
            check=True,
        )


def index(repos_dir: Path, db: Path, files_per_repo: int) -> None:
    db.parent.mkdir(parents=True, exist_ok=True)
    embedder = get_embedder("onnx")
    report: list[dict] = []
    with Store(db) as store:
        for slug in REPOS:
            root = repos_dir / repo_name(slug)
            if not root.is_dir():
                print(f"  {slug}: not cloned, skipping", file=sys.stderr)
                continue
            files = pick_files(root, files_per_repo)
            stats = index_paths(store, embedder, files)
            chunks = int(
                store.conn.execute(
                    "SELECT COUNT(*) FROM chunks c JOIN documents d ON d.id = c.doc_id "
                    "WHERE d.path LIKE ?",
                    (f"%/{root.name}/%",),
                ).fetchone()[0]
            )
            docs = len(
                store.conn.execute(
                    "SELECT id FROM documents WHERE path LIKE ?", (f"%/{root.name}/%",)
                ).fetchall()
            )
            row = {
                "repo": slug,
                "files": len(files),
                "indexed": stats.indexed,
                "skipped": stats.skipped,
                "documents": docs,
                "chunks": chunks,
                "mb": round(sum(f.stat().st_size for f in files) / 1e6, 1),
            }
            report.append(row)
            print("  " + json.dumps(row), flush=True)
    print(json.dumps({"repos": len(report), "chunks": sum(r["chunks"] for r in report)}))
    target = db.with_suffix(".corpus.json")
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("clone", "index"))
    parser.add_argument("--dir", type=Path, default=Path("/tmp/ragdesk-public/repos"))
    parser.add_argument("--db", type=Path, default=Path("/tmp/ragdesk-public/public.db"))
    parser.add_argument("--files-per-repo", type=int, default=250)
    args = parser.parse_args()

    if args.step == "clone":
        clone(args.dir)
        return 0
    index(args.dir, args.db, args.files_per_repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
