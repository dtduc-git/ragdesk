"""Vector search at scale: every candidate backend over synthetic corpora built
from a real index, so latency/RAM/recall numbers describe the shipped code.

    uv run --extra onnx python scripts/vector_scale_bench.py --source /tmp/chk.db \
        --sizes 10000,100000,500000 --queries 30

    # sqlite-vec needs an interpreter with loadable SQLite extensions (the
    # current uv-managed 3.12 and Homebrew 3.13 both have them; verify with
    # hasattr(sqlite3.connect(':memory:'), 'enable_load_extension')):
    uv run --extra onnx --with sqlite-vec --with numpy --with usearch \
        python scripts/vector_scale_bench.py --source /tmp/chk.db \
        --sizes 10000,100000 --queries 30

Each backend runs in its own child process so RSS numbers are per backend.
The numpy matrix is exact, so it is the ground truth for recall@5; it is
measured first and its top-5 ids are handed to the approximate candidates.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path


def _numpy():
    import numpy

    return numpy


# One pool drawn once: query i is the same vector no matter how many the caller asks for.
QUERY_POOL = 64


class TextPool:
    """Real vectors and texts from a source index — the synthetic corpus base."""

    def __init__(self, source: Path, limit: int = 20000) -> None:
        conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT embedding, text FROM chunks WHERE embedding IS NOT NULL LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        if not rows:
            raise SystemExit(f"source index has no chunks: {source}")
        np = _numpy()
        self.dim = len(rows[0][0]) // 4
        self.vectors = np.frombuffer(b"".join(row[0] for row in rows), dtype=np.float32).reshape(
            -1, self.dim
        )
        self.texts = [(row[1] or "")[:1200] for row in rows]

    def queries(self, count: int, seed: int):
        np = _numpy()
        # Draw the whole pool before slicing: the RNG state must not depend on
        # ``count``, or the slow backends (3 queries) would be scored against a
        # ground truth built from different query vectors.
        rng = np.random.default_rng(seed + 1000)
        picked = rng.integers(0, len(self.vectors), QUERY_POOL)
        noise = rng.normal(0, 0.08, (QUERY_POOL, self.dim)).astype(np.float32)
        vecs = self.vectors[picked] + noise
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        if count > QUERY_POOL:
            raise SystemExit(f"--queries must be <= {QUERY_POOL}")
        return vecs[:count]


def build_synthetic(dest: Path, pool: TextPool, size: int, seed: int) -> None:
    np = _numpy()
    if dest.exists():
        dest.unlink()
    conn = sqlite3.connect(dest)
    conn.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY, source TEXT NOT NULL, path TEXT NOT NULL UNIQUE,
            content_hash TEXT NOT NULL, mtime REAL NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}',
            indexed_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL, ordinal INTEGER NOT NULL,
            text TEXT NOT NULL, embedding BLOB NOT NULL,
            parent_ordinal INTEGER NOT NULL DEFAULT 0, line_start INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE parents (
            doc_id INTEGER NOT NULL, ordinal INTEGER NOT NULL, text TEXT NOT NULL
        );
        """
    )
    docs = 32
    conn.executemany(
        "INSERT INTO documents (id, source, path, content_hash, mtime) VALUES (?, ?, ?, ?, ?)",
        [(i, "synthetic", f"synthetic://doc-{i}.md", "0", 0.0) for i in range(docs)],
    )
    rng = np.random.default_rng(seed)
    batch = 20000
    for start in range(0, size, batch):
        count = min(batch, size - start)
        picked = rng.integers(0, len(pool.vectors), count)
        noise = rng.normal(0, 0.08, (count, pool.dim)).astype(np.float32)
        vectors = pool.vectors[picked] + noise
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        conn.executemany(
            "INSERT INTO chunks (id, doc_id, ordinal, text, embedding, parent_ordinal) "
            "VALUES (?, ?, ?, ?, ?, 0)",
            [
                (
                    start + offset,
                    (start + offset) % docs,
                    (start + offset) // docs,
                    pool.texts[picked[offset]],
                    vectors[offset].tobytes(),
                )
                for offset in range(count)
            ],
        )
        conn.commit()
    conn.close()


def fetch_payloads(conn: sqlite3.Connection, ids: list[int]) -> int:
    if not ids:
        return 0
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"""
        SELECT c.id, c.text, d.path, COALESCE(p.text, '') AS parent_text
        FROM chunks c JOIN documents d ON d.id = c.doc_id
        LEFT JOIN parents p ON p.doc_id = d.id AND p.ordinal = c.parent_ordinal
        WHERE c.id IN ({placeholders})
        """,
        ids,
    ).fetchall()
    return len(rows)


def old_scan(conn: sqlite3.Connection, query_vec, limit: int) -> list[int]:
    """The pre-2026-09 dense lane: payload join, then Python cosine over everything."""
    import array
    import math

    rows = conn.execute(
        """
        SELECT c.id, c.embedding, c.text, d.path, COALESCE(p.text, '') AS parent_text
        FROM chunks c JOIN documents d ON d.id = c.doc_id
        LEFT JOIN parents p ON p.doc_id = d.id AND p.ordinal = c.parent_ordinal
        """
    ).fetchall()
    q_norm = math.sqrt(sum(v * v for v in query_vec)) or 1.0
    scored = []
    for row in rows:
        vec = array.array("f")
        vec.frombytes(row[1])
        dot = sum(a * b for a, b in zip(query_vec, vec, strict=True))
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        scored.append((dot / (q_norm * norm), int(row[0])))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk_id for _score, chunk_id in scored[:limit]]


def rss_mb() -> int:
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True
    ).stdout.strip()
    return int(out) // 1024 if out else 0


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index] * 1000.0


def run_one(
    db: Path,
    pool: TextPool,
    size: int,
    backend: str,
    query_count: int,
    seed: int,
    gt: dict[str, float] | None,
    expansion_search: int = 256,
) -> dict:
    from ragdesk import vectors

    np = _numpy()
    # ef=64 looked like a 0.55-0.65 "recall" on this data — it is a search-effort
    # knob, not a backend property. 256 was the smallest value that recovered it.
    queries = pool.queries(query_count, seed)
    conn = sqlite3.connect(db)
    baseline = rss_mb()
    started = time.perf_counter()
    engine = None
    if backend == "old_scan":
        pass
    elif backend in ("python", "numpy", "usearch_f16", "usearch_f32"):
        if backend.startswith("usearch"):
            engine = vectors.UsearchIndex(
                conn, dtype=backend[-3:], expansion_search=expansion_search
            )
        elif backend == "python":
            engine = vectors.PythonScan(conn)
        else:
            engine = vectors.NumpyMatrix(conn)
    elif backend == "sqlitevec":
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("DROP TABLE IF EXISTS vec0")
        conn.execute(
            f"CREATE VIRTUAL TABLE vec0 USING vec0(chunk_id integer primary key, "
            f"embedding float[{pool.dim}])"
        )
        normalized = []
        for row in conn.execute("SELECT id, embedding FROM chunks").fetchall():
            vec = np.frombuffer(row[1], dtype=np.float32)
            vec = vec / (float(np.linalg.norm(vec)) or 1.0)
            normalized.append((row[0], vec.tobytes()))
        conn.executemany("INSERT INTO vec0(chunk_id, embedding) VALUES (?, ?)", normalized)
        conn.commit()
    else:
        raise SystemExit(f"unknown backend {backend}")
    build_s = time.perf_counter() - started

    search_ids: list[list[int]] = []
    times: list[float] = []
    warmup = min(2, query_count)
    for index, query in enumerate(queries):
        begin = time.perf_counter()
        if backend == "old_scan":
            ids = old_scan(conn, query, 50)
            search_ids.append(ids)
        elif backend == "sqlitevec":
            rows = conn.execute(
                "SELECT chunk_id FROM vec0 WHERE embedding MATCH ? AND k = 50",
                (query.tobytes(),),
            ).fetchall()
            ids = [int(row[0]) for row in rows]
            fetch_payloads(conn, ids)
            search_ids.append(ids)
        else:
            candidates = engine.search(conn, query.tolist(), 50)
            fetch_payloads(conn, [chunk_id for chunk_id, _score in candidates])
            search_ids.append([chunk_id for chunk_id, _score in candidates])
        elapsed = time.perf_counter() - begin
        if index >= warmup:
            times.append(elapsed)
    floors = None
    recall = None
    if gt is None and backend == "numpy":
        # The exact index is the ground truth: publish the 5th score per query
        # so approximate backends are judged on result quality, not tie order.
        floors = {
            str(index): _fifth_score(conn, np, queries[index], ids)
            for index, ids in enumerate(search_ids)
        }
        recall = 1.0
    if gt is not None:
        hits = 0
        for index, ids in enumerate(search_ids):
            floor = float(gt.get(str(index), -2.0))
            for chunk_id in ids[:5]:
                score = _true_score(conn, np, queries[index], chunk_id)
                if score is not None and score >= floor - 1e-4:
                    hits += 1
        recall = hits / (5 * len(search_ids)) if search_ids else 0.0
    conn.close()
    return {
        "size": size,
        "backend": backend,
        "build_s": round(build_s, 2),
        "p50_ms": round(percentile(times, 0.50), 1),
        "p95_ms": round(percentile(times, 0.95), 1),
        "rss_baseline_mb": baseline,
        "rss_mb": rss_mb(),
        "rss_delta_mb": rss_mb() - baseline,
        "recall@5": None if recall is None else round(recall, 3),
        "queries": len(times),
        "floors": floors,
    }


def _vector(conn: sqlite3.Connection, np, chunk_id: int):
    row = conn.execute("SELECT embedding FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
    return None if row is None else np.frombuffer(row[0], dtype=np.float32)


def _true_score(conn: sqlite3.Connection, np, query, chunk_id: int) -> float | None:
    vec = _vector(conn, np, chunk_id)
    if vec is None:
        return None
    norm = float(np.linalg.norm(query)) * float(np.linalg.norm(vec))
    return float(np.dot(query, vec)) / (norm or 1.0)


def _fifth_score(conn: sqlite3.Connection, np, query, ids: list[int]) -> float:
    for chunk_id in ids[:5][::-1]:
        score = _true_score(conn, np, query, chunk_id)
        if score is not None:
            return score
    return -2.0


def child_main(args) -> int:
    pool = TextPool(args.source)
    gt = None
    if args.gt:
        gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    result = run_one(
        Path(args.db),
        pool,
        args.internal_run[0],
        args.internal_run[1],
        args.queries,
        args.seed,
        gt,
        expansion_search=args.usearch_ef,
    )
    print(json.dumps(result))
    return 0


def spawn(
    db: Path,
    source: Path,
    size: int,
    backend: str,
    queries: int,
    seed: int,
    gt: Path | None,
    usearch_ef: int,
):
    command = [
        sys.executable,
        __file__,
        "--internal-run",
        str(size),
        backend,
        "--db",
        str(db),
        "--source",
        str(source),
        "--queries",
        str(queries),
        "--seed",
        str(seed),
        "--usearch-ef",
        str(usearch_ef),
    ]
    if gt is not None:
        command += ["--gt", str(gt)]
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        error = (proc.stderr or proc.stdout).strip().splitlines()
        return {"size": size, "backend": backend, "error": error[-1] if error else "failed"}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/tmp/chk.db"))
    parser.add_argument("--sizes", default="10000,100000,500000")
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--usearch-ef", type=int, default=256)
    parser.add_argument("--keep", action="store_true", help="keep the synthetic databases")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--internal-run", nargs=2, metavar=("SIZE", "BACKEND"))
    parser.add_argument("--db", type=Path)
    parser.add_argument("--gt", type=Path)
    args = parser.parse_args()

    if args.internal_run:
        return child_main(args)

    pool = TextPool(args.source)
    print(f"source: {args.source} | vectors: {len(pool.vectors)} | dim {pool.dim}")
    backends = ["numpy", "python", "old_scan", "usearch_f32", "usearch_f16", "sqlitevec"]
    rows: list[dict] = []
    for size in [int(part) for part in args.sizes.split(",")]:
        db = Path(f"/tmp/vector-scale-{size}.db")
        print(f"building synthetic corpus: {size} chunks ...", flush=True)
        started = time.perf_counter()
        build_synthetic(db, pool, size, args.seed)
        print(f"  built in {time.perf_counter() - started:.1f}s", flush=True)

        slow_queries = 3 if size >= 30000 else args.queries
        gt_path = Path(f"/tmp/vector-scale-gt-{size}.json")
        result = spawn(
            db, args.source, size, "numpy", args.queries, args.seed, None, args.usearch_ef
        )
        if result.get("error"):
            # The ground truth is missing: report the row instead of a KeyError.
            rows.append(result)
            print(" ", json.dumps(result), flush=True)
            if not args.keep:
                db.unlink(missing_ok=True)
            continue
        ground_truth = result.pop("floors", None)
        gt_path.write_text(json.dumps(ground_truth), encoding="utf-8")
        rows.append(result)
        print(" ", json.dumps(result), flush=True)

        for backend in backends[1:]:
            queries = slow_queries if backend in ("python", "old_scan") else args.queries
            if backend == "old_scan" and size > 200000:
                continue
            result = spawn(
                db, args.source, size, backend, queries, args.seed, gt_path, args.usearch_ef
            )
            rows.append(result)
            print(" ", json.dumps(result), flush=True)
        gt_path.unlink(missing_ok=True)
        if not args.keep:
            db.unlink(missing_ok=True)

    header = "| size | backend | build s | p50 ms | p95 ms | RSS +MB | recall@5 |"
    print("\n" + header)
    print("|" + "---|" * 7)
    for row in rows:
        if row.get("error"):
            print(f"| {row['size']} | {row['backend']} | - | - | - | - | {row['error'][:60]} |")
            continue
        recall = "-" if row["recall@5"] is None else f"{row['recall@5']:.3f}"
        print(
            f"| {row['size']} | {row['backend']} | {row['build_s']} | {row['p50_ms']} | "
            f"{row['p95_ms']} | {row['rss_delta_mb']} | {recall} |"
        )
    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
