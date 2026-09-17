#!/usr/bin/env python3
"""Compare ONNX embedders on the golden sets: quality, speed, RAM and size.

Reproduces the table in docs/embedder-comparison.md. Reranking stays off so the
embedder is the only variable; every run gets its own database and a scratch
config dir, so your real settings are never touched.

    scripts/embedder_compare.py                    # all candidates, both corpora
    scripts/embedder_compare.py --repo-only        # skip the small fixtures set
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CANDIDATES = [
    "onnx:onnx-community/embeddinggemma-300m-ONNX",  # the shipped default
    "onnx:Xenova/multilingual-e5-small",
    "onnx:Xenova/multilingual-e5-base",
    "onnx:Xenova/multilingual-e5-large",
    "onnx:onnx-community/gte-multilingual-base",
]

CORPORA = {
    "fixtures": (ROOT / "fixtures/docs", ROOT / "fixtures/golden.jsonl"),
    "repo": (ROOT, ROOT / "fixtures/golden_repo.jsonl"),
}


def run(
    cmd: list[str], env: dict[str, str], log: Path, *, timed: bool = False
) -> tuple[float, int]:
    """Run a command; return (wall seconds, peak RSS MiB). timed uses time -l."""
    real = ["/usr/bin/time", "-l", *cmd] if timed else cmd
    started = time.monotonic()
    with log.open("w") as handle:
        done = subprocess.run(real, stdout=handle, stderr=subprocess.STDOUT, env=env)
    elapsed = time.monotonic() - started
    text = log.read_text()
    if done.returncode != 0:
        print(f"!! failed: {' '.join(cmd)}\n{text[-1500:]}", file=sys.stderr)
        raise SystemExit(1)
    match = re.search(r"(\d+)\s+maximum resident set size", text)
    return elapsed, int(match.group(1)) // (1024 * 1024) if match else 0


def metrics(eval_stdout: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in ("recall@5", "ndcg@10", "mrr@10"):
        match = re.search(rf"{name}\s*:\s*([0-9.]+)", eval_stdout)
        if match:
            values[name] = float(match.group(1))
    return values


def model_size(repo: str) -> str:
    cache = Path.home() / ".cache/huggingface/hub" / f"models--{repo.replace('/', '--')}"
    if not cache.is_dir():
        return "?"
    total = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file())
    return f"{total / (1024**3):.2f}G" if total > 1024**3 else f"{total / (1024**2):.0f}M"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-only", action="store_true")
    parser.add_argument("--candidates", nargs="*", default=CANDIDATES)
    args = parser.parse_args()

    corpora = {"repo": CORPORA["repo"]} if args.repo_only else CORPORA
    work = Path(tempfile.mkdtemp(prefix="ragdesk-embedder-compare-"))
    env = {**os.environ, "RAGDESK_CONFIG_DIR": str(work / "config")}
    print(f"# workdir {work}", flush=True)
    print(
        "\n| embedder | dim | weights | corpus | recall@5 | nDCG@10 | MRR@10 | index | peak RSS |",
        "\n|---|---|---|---|---|---|---|---|---|",
        flush=True,
    )

    for spec in args.candidates:
        repo = spec.split(":", 1)[1]
        tag = repo.replace("/", "--")
        probe = f"from ragdesk.embed import get_embedder;print(get_embedder({spec!r}).dim)"
        dim = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            env=env,
            cwd=ROOT,
            check=True,
        ).stdout.strip()
        size = model_size(repo)
        for corpus, (target, golden) in corpora.items():
            db = work / f"{tag}-{corpus}.db"
            index_log = work / f"{tag}-{corpus}.index.log"
            seconds, peak = run(
                [
                    "uv",
                    "run",
                    "ragdesk",
                    "--embedder",
                    spec,
                    "--rerank",
                    "none",
                    "--db",
                    str(db),
                    "index",
                    str(target),
                ],
                env,
                index_log,
                timed=True,
            )
            chunks = re.search(r"chunks=(\d+)", index_log.read_text())
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "ragdesk",
                    "--embedder",
                    spec,
                    "--rerank",
                    "none",
                    "--db",
                    str(db),
                    "eval",
                    "--golden",
                    str(golden),
                ],
                capture_output=True,
                text=True,
                env=env,
                cwd=ROOT,
            )
            m = metrics(result.stdout)
            count = chunks.group(1) if chunks else "?"
            print(
                f"| `{repo}` | {dim} | {size} | {corpus} ({count} chunks) "
                f"| {m.get('recall@5', float('nan')):.3f} | {m.get('ndcg@10', float('nan')):.3f} "
                f"| {m.get('mrr@10', float('nan')):.3f} | {seconds:.0f}s | {peak} MB |",
                flush=True,
            )
    print(f"\nlogs kept in {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
