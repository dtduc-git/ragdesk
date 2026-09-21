# Dense search at scale: which vector backend ships, and what it costs

The dense lane used to fetch the payload of **every** chunk (text + parent
text), score them with a pure-Python cosine loop, then keep the top-k. That was
fine at 3k chunks and quadratic-ish pain at 100k. This page is the measurement
behind the replacement: the same synthetic corpora, five candidate backends,
each in its own process so RSS numbers are real.

Reproduce with:

```bash
uv run --extra onnx python scripts/vector_scale_bench.py \
    --source /path/to/any/index.db --sizes 10000,100000,500000,1000000 --queries 24
```

- **Corpus** — synthetic chunks built from a real index (14k chunks of real
  vectors and texts, sampled with noise). Payload size, vector dim (768) and
  table shape are the shipped ones; only the row count is inflated.
- **Queries** — 24 vectors sampled from the same pool with fresh noise.
- **Latency** — p50/p95 of one full dense lane call at `pool=50`: candidates
  plus the payload fetch for the winners (what `hybrid_search` does).
- **RSS** — per-process delta measured after the query loop (`ps` RSS, not
  `ru_maxrss`).
- **recall@5** — an approximate backend is judged by whether its results are as
  good as the exact ones, not by id equality (the corpus has near-duplicates,
  where any exact top-5 is an arbitrary subset of a tie family). numpy is exact,
  so it produces the per-query score floor; a returned chunk counts as a hit
  when its true score is within 1e-4 of the exact 5th score.

## Measured 2026-09-18/19 — 12-core M-series, 32 GB, 768-dim, 24 queries

Latency is machine-dependent; quality columns are deterministic. `old_scan` is
the pre-change Python lane (payload join + scan); `python` is the current
fallback (score ids only, payload fetched for the 50 winners). Level 0 landed
first and is the free 2.1×: same results, no new dependency.

| chunks | backend | build | p50 | p95 | RSS +MB | recall@5 |
|---|---|---|---|---|---|---|
| 10k | old_scan | 0s | 1123ms | 1131ms | 15 | 1.000 |
| 10k | python | 0s | 533ms | 536ms | 2 | 1.000 |
| 10k | numpy | 0.04s | **0.8ms** | 1.0ms | 59 | 1.000 |
| 10k | sqlite-vec | 0.6s | 5.3ms | 5.6ms | 27 | 1.000 |
| 10k | usearch f32 | 2.8s | 2.7ms | 3.2ms | 70 | 0.858 |
| 10k | usearch f16 | 1.4s | 1.3ms | 1.4ms | 55 | 0.867 |
| 100k | old_scan | 0s | 11242ms | — | 487 | — |
| 100k | python | 0s | 5388ms | — | 357 | — |
| 100k | numpy | 0.4s | **6.3ms** | 7.1ms | 588 | 1.000 |
| 100k | sqlite-vec | 5.9s | 55ms | 57ms | 747 | 1.000 |
| 100k | usearch f32 | 45s | 4.4ms | 5.2ms | 959 | 0.650 |
| 100k | usearch f16 | 23s | 2.7ms | 3.2ms | 812 | 0.642 |
| 500k | python | 0s | 26627ms | — | 2000 | — |
| 500k | numpy | 1.8s | **30.5ms** | 31.8ms | 2943 | 1.000 |
| 500k | sqlite-vec | 30s | 276ms | 295ms | 3952 | 1.000 |
| 500k | usearch f32 | 229s | 4.7ms | 5.9ms | 4667 | 0.700 |
| 500k | usearch f16 | 144s | 3.0ms | 3.6ms | 4239 | 0.775 |
| 1M | python | 0s | 53508ms | — | 4026 | — |
| 1M | numpy | 5.0s | **60.9ms** | 66.1ms | 5881 | 1.000 |
| 1M | sqlite-vec | 63s | 561ms | 581ms | 5911 | 1.000 |
| 1M | usearch f32 | 481s | 11.2ms | 13.7ms | 8278 | 0.675 |
| 1M | usearch f16 | 274s | 5.1ms | 7.0ms | 7867 | 0.700 |

Notes that keep the table honest:

- `python` and `old_scan` are the same arithmetic as numpy; the sub-1.000 rows
  at 100k are the 1e-4 tie threshold on duplicate-heavy synthetic data, not a
  quality difference. `—` means the run was capped (only one timed query). The
  old lane is not measured past 100k because one query takes minutes.
- RSS includes SQLite's read side during build; the numpy matrix itself is
  `chunks × dim × 4` bytes (1M → 3 GB). The build streams rows instead of
  `fetchall()` so the peak does not double (it did: 8.6 GB before, 5.9 GB after).
- usearch was run at `expansion_search=256`. At its default 64 it "recalled"
  0.55–0.65 — it is a search-effort knob, not a backend property; even at 256
  it loses ~15–35% of top-5 quality on this data.
- sqlite-vec needs an interpreter built with loadable SQLite extensions, and
  that varies by Python build: uv's current managed 3.12 and Homebrew 3.13
  both have it (verified with a live `vec0` table), older uv standalone builds
  and some system Pythons do not. Check any interpreter with
  `python -c "import sqlite3; print(hasattr(sqlite3.connect(':memory:'), 'enable_load_extension'))"`.
  Even where it loads, it is ~9× slower than numpy at 1M plus a shadow `vec0`
  table (roughly the corpus size on disk) and a native extension dependency —
  so it stays benchmark-only on performance grounds, not availability.

## Decision

- **numpy matrix is the default** (`vectors.NumpyMatrix`, exact cosine, rows
  L2-normalized). It wins latency at every size we measured, is exact, adds no
  extra dependency (numpy is a core dependency of ragdesk), and fits a personal
  corpus in RAM: the real index (~14k chunks) costs ~45 MB.
- **Python scan is the fallback** only for a platform where numpy is missing:
  a correctness backstop, ~1000× slower, never a mode to run deliberately.
- **usearch is opt-in**, for people who accept approximate recall in exchange
  for millisecond searches (`--vector-backend usearch`, or the setting/API).
  It is not a default: it lost on recall *and* on RSS against numpy at 1M (7.9
  vs 5.9 GB), because HNSW keeps the vectors too plus the graph, and building
  it took 4.5–8 minutes at 1M against numpy's 5 seconds.
- **sqlite-vec is not shipped.** It is exact but ~9× slower than numpy at 1M
  on the same machine, and it needs a native extension plus a shadow `vec0`
  table to maintain. It stays in the benchmark so the claim has a number.
- Indexes are cached per database path and invalidated through a long-lived
  guard connection watching the `chunks_revision` counter bumped by every chunk
  write, so the watcher thread, a CLI run or a second app instance is picked up
  without rebuilding the matrix on unrelated commits (chat messages, answer
  cache). An unknown or uninstalled backend setting falls back to auto with a
  warning instead of breaking retrieval.
  A scoped query (`folder:`/`source:`) falls back to the exact Python scan over
  the matching subset — HNSW cannot filter, and a subset scan is proportional
  to the scope, not the corpus.

The dense lane is no longer the bottleneck anywhere: at 1M chunks it is ~61 ms,
next to a reranker at ~0.8 s and an LLM answer at multiple seconds.
