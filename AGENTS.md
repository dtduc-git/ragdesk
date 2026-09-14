# ragdesk — agent notes

Personal, local-first RAG over your own sources. Core is **stdlib-only Python**
(retrieval + eval); models are reached through a local Ollama server.

## Layout

- `src/ragdesk/` — package. `chunk` (paragraph-aware + overlap), `embed`
  (HashingEmbedder for CI, OllamaEmbedder, OnnxEmbedder = EmbeddingGemma int8
  with query/doc prompts; shared `tokenize`), `store` (SQLite: FTS5 + float32
  vectors + fail-closed embedder guard), `search` (BM25 + dense + RRF;
  `retrieve` adds the optional rerank stage), `rerank` (LexicalReranker
  baseline; FastEmbedReranker behind the `onnx` extra), `index` (incremental
  local files), `github` / `confluence` / `gdrive` connectors, `web` (same-host
  HTML crawl, capped pages/depth), `htmlutil` (shared HTML→text),
  `evaluate` (recall@5 / nDCG@10 / MRR), `answer` (Ollama LLM: non-stream +
  stream, grounding gate), `credentials` (0600 store under `~/.config/ragdesk/`),
  `envfile` (.env loader for dev), `buildenv` (bakes .env into a gitignored
  `_build_env.py` for release builds), `serve` (loopback JSON API:
  status/search/ask/index/sync + connections connect/disconnect + optional
  static UI), `presets` (RAM tiers), `cli`.
- `desktop/` — Tauri 2 shell (card-catalog UI). Rust spawns `ragdesk serve`
  with `--db $HOME/.ragdesk/index.db`; env overrides: `RAGDESK_BIN`,
  `RAGDESK_DB`, `RAGDESK_LLM_MODEL`, `RAGDESK_PROJECT`. Browser mode:
  `ragdesk serve --ui desktop/dist`.
- Connections: GitHub has three paths (device code with `RAGDESK_GITHUB_CLIENT_ID`
  or a saved client ID, `gh` login, or a pasted token); Confluence and GDrive
  connect flows validate before saving to `~/.config/ragdesk/credentials.json`
  (`0600`; override the dir with `RAGDESK_CONFIG_DIR` for tests).
- Credentials policy: `.env` (gitignored) → `python -m ragdesk.buildenv` bakes
  non-confidential values into `src/ragdesk/_build_env.py` (gitignored) for
  release artifacts. The Atlassian 3LO secret is refused by the baker — it stays
  per-user. User-facing setup guides: `docs/sources.md`.
- `fixtures/` — tiny corpus + two golden sets (fixtures, repo) for offline CI.
- `tests/` — pytest; always uses `HashingEmbedder` (never requires Ollama or
  network). Connector tests monkeypatch HTTP.

## Commands

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run ragdesk --embedder hash --db /tmp/eval.db index fixtures/docs
uv run ragdesk --embedder hash --db /tmp/eval.db eval --golden fixtures/golden.jsonl
# repo golden (real numbers; --embedder onnx downloads ~0.3GB on first run)
uv run ragdesk --embedder hash --db /tmp/eval-repo.db index README.md AGENTS.md SECURITY.md src .github fixtures/docs
uv run ragdesk --embedder hash --db /tmp/eval-repo.db eval --golden fixtures/golden_repo.jsonl
# desktop shell
cd desktop && npm install && npm run tauri build
# release build with baked credentials
cp .env.example .env   # fill non-confidential values (see SECURITY.md)
uv run python -m ragdesk.buildenv && uv build
```

## Conventions

- Core stays dependency-free (argparse/sqlite3/urllib). New runtime deps need a
  good reason; ONNX/reranker deps belong in an optional extra.
- `HashingEmbedder` is for tests/CI only — never quote its numbers as quality.
- Never default the reranker to a CC-BY-NC model (jina-reranker-v2 is
  non-commercial); `BAAI/bge-reranker-base` (MIT) is the fastembed default.
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
