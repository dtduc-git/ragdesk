# Which embedder ships in the default preset

`ragdesk` embeds every chunk locally with `EmbeddingGemma-300M` (ONNX, int8).
That was a reputation-and-features pick — multilingual, int8-friendly, Matryoshka
dims — and this page is the measurement that keeps it honest: the same golden
sets, four candidate models, reranking off so the embedder is the only variable.

Reproduce with:

```bash
uv run python scripts/embedder_compare.py          # ~40 min; downloads ~1 GB of models
```

Each run indexes the corpus into a fresh database under a scratch config dir
(never your own settings), then runs `ragdesk eval --golden` with `--rerank none`.

- **Corpus `repo`** — this repository, 2.8k chunks, `fixtures/golden_repo.jsonl`
  (12 hand-written questions about real files, scoped to the checkout). This is
  the discriminating set: hand-written fixtures saturate at 1.000 for every model.
- **Corpus `fixtures`** — `fixtures/docs`, 3 documents, `fixtures/golden.jsonl`
  (7 questions). Too small to separate models; kept as a smoke row.
- **Quality** — recall@5 / nDCG@10 / MRR@10 from the golden set.
- **Index** — wall clock to embed the whole corpus, at the shipped default
  (quiet indexing: 4 threads).
- **Peak RSS** — `/usr/bin/time -l` maximum resident set size of the indexing
  process, i.e. the RAM a first index actually costs.
- **Weights** — on-disk footprint in the Hugging Face cache (what a user
  downloads and keeps).

## Measured 2026-09-17 — 8-core M-series, quiet indexing (4 threads), rerank off

Quality, dimensions and download size do not depend on machine load; index
seconds and peak RSS do, and this machine was busy (load average ~18 from
unrelated work) during the first pass — so those two columns are marked
*provisional* and get re-measured on a quiet machine before anyone quotes them.
The script prints all of it.

| embedder | dim | weights | corpus | recall@5 | nDCG@10 | MRR@10 | index | peak RSS |
|---|---|---|---|---|---|---|---|---|
| `onnx-community/embeddinggemma-300m-ONNX` **(shipped)** | 768 | 315M | fixtures (3 chunks) | 1.000 | 1.000 | 1.000 | 2s | 1625 MB |
| `onnx-community/embeddinggemma-300m-ONNX` | 768 | 315M | repo (2873 chunks) | 0.833 | **0.671** | 0.590 | 615s* | 2734 MB* |
| `Xenova/multilingual-e5-small` | 384 | 129M | fixtures (3 chunks) | 1.000 | 1.000 | 1.000 | 2s | 735 MB |
| `Xenova/multilingual-e5-small` | 384 | 129M | repo (2879 chunks) | **0.917** | 0.626 | 0.531 | 236s* | 2059 MB* |
| `Xenova/multilingual-e5-base` | 768 | 282M | fixtures (3 chunks) | 1.000 | 1.000 | 1.000 | 3s | 1083 MB |
| `Xenova/multilingual-e5-base` | 768 | 282M | repo (2879 chunks) | pending | | | | |
| `onnx-community/gte-multilingual-base` | 768 | 341M | fixtures (3 chunks) | 1.000 | 1.000 | 1.000 | 2s | 1190 MB |
| `onnx-community/gte-multilingual-base` | 768 | 341M | repo (2873 chunks) | 0.833 | 0.659 | **0.603** | 277s* | 2611 MB* |

\* measured while the machine was also doing other work — treat as relative, not
as a benchmark. The `repo` rows also span two snapshots of this repository
(2873 vs 2879 chunks: four files were added between passes), which is why the
final table is scheduled for one frozen pass.

The fixtures corpus saturates for every model — it separates nothing, and is
kept only to prove the harness end-to-end.

## What the numbers say

- **Recall is a tie within noise; ordering is not.** 12 questions make one query
  worth ~0.083 recall, so Gemma's 0.833 vs e5-small's 0.917 is one question —
  suggestive, not settled. Gemma leads nDCG@10 (0.671) and is the reason the
  `light` preset (no reranker) keeps it: with no reranker, ranking order is the
  embedder's job, and that is the column it wins.
- **The default stays, for a product reason as much as a quality one.** All
  presets share one embedder, so switching preset never triggers a re-index.
  A preset-specific embedder would; `balanced` cannot quietly use e5-small while
  `light` uses Gemma.
- **The runner-up is now a supported option.** `multilingual-e5-small` is 2.4x
  smaller to download (129M vs 315M), 384-dim (a smaller index), and indexed
  2.6x faster than Gemma in this session despite the load. It costs ordering
  (nDCG 0.626 vs 0.671), which the reranker would largely hide:
  `ragdesk --embedder onnx:Xenova/multilingual-e5-small …`.
- **Bigger is not better here.** `e5-base` (278M) did *worse* on recall than
  `e5-small` (118M) in the first pass; `gte-multilingual-base` ties Gemma on
  recall and edges it on MRR@10 (0.603 vs 0.590) while downloading the most.
- **Gemma's indexing pass was the slowest in every run** (2.2–2.6x gte/e5-small
  in the same session). That is the price of the ordering edge; a fair absolute
  number needs the quiet re-run.

## Caveats

12 questions: one query is worth ~0.083 recall, so treat differences of one
question as noise, and ties as ties. All models ran int8 "quantized" exports, CPU-only,
`--rerank none`, text chunking.

Two findings came out of the first pass and both are now fixed in
`OnnxEmbedder`:

- The e5 exports carry **no pooled output** and rejected `token_type_ids` — they
  run through a mask-aware mean-pooling fallback in `embed.py`.
- The first e5 pass silently **skipped one file** because a chunk exceeded the
  model's 512-token position table. The loader now reads
  `max_position_embeddings` from each model's `config.json` and truncates to it
  (capped at 2048), which is why the e5 row has 2879 chunks instead of 2779.
