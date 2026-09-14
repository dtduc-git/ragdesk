# ragdesk — agent notes

Personal, local-first RAG over your own sources. Core is **stdlib-only Python**
(retrieval + eval); models are reached through a local Ollama server.

## Layout

- `src/ragdesk/` — package. `chunk` (paragraph-aware + overlap), `embed`
  (HashingEmbedder for CI, OllamaEmbedder for real), `store` (SQLite: FTS5 +
  float32 vectors + fail-closed embedder guard), `search` (BM25 + dense + RRF),
  `index` (incremental local files), `evaluate` (recall@5 / nDCG@10 / MRR),
  `answer` (Ollama LLM + grounding gate), `cli`.
- `fixtures/` — tiny corpus + golden set for the offline CI eval.
- `tests/` — pytest; always uses `HashingEmbedder` (never requires Ollama).

## Commands

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run ragdesk --embedder hash --db /tmp/eval.db index fixtures/docs
uv run ragdesk --embedder hash --db /tmp/eval.db eval --golden fixtures/golden.jsonl
```

## Conventions

- Core stays dependency-free (argparse/sqlite3/urllib). New runtime deps need a
  good reason; ONNX/reranker deps belong in an optional extra.
- `HashingEmbedder` is for tests/CI only — never quote its numbers as quality.
- Embedder/dimension mismatch must stay fail-closed (`Store.ensure_embedder`).
- Sources are read-only. No telemetry, ever.
- Eval numbers published in README must be reproducible from the repo.

## Roadmap order

reranker → ONNX embedder → GitHub connector → Tauri shell + RAM presets →
Confluence/GDrive → eval badge per release.

## Notes

- If code is ever reused from OpsRAG (Apache-2.0), add the required NOTICE
  attribution before release.
- PyPI name `ragdesk` verified free (2026-09-14); re-check the similarity
  checker at publish time.
