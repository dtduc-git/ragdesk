# ragdesk — agent notes

Personal, local-first RAG over your own sources. Core is **stdlib-only Python**
(retrieval + eval); models are reached through a local Ollama server.

## Layout

- `src/ragdesk/` — package. `chunk` (paragraph-aware + overlap), `embed`
  (HashingEmbedder for CI, OllamaEmbedder, OnnxEmbedder = EmbeddingGemma int8
  with query/doc prompts; shared `tokenize`), `store` (SQLite: FTS5 + float32
  vectors + fail-closed embedder guard; keeps the chosen local roots in `meta`
  for the per-path Indexed breakdown via `local_paths()`), `search` (BM25 +
  dense + RRF; `retrieve` adds the optional rerank stage), `rerank`
  (LexicalReranker baseline; FastEmbedReranker + OnnxReranker = multilingual
  gte behind the `onnx` extra), `index` (incremental local files; `iter_files`
  prunes `SKIP_DIRS` during the walk so `target/`/`node_modules/` are never
  traversed), `github` / `gitlab` / `confluence` / `gdrive` / `notion` /
  `msgraph` (OneDrive + SharePoint, device flow) connectors, `web` (same-host
  HTML crawl, capped pages/depth), `archive`
  (shared repo-tarball extraction), `htmlutil` (shared HTML→text),
  `evaluate` (recall@5 / nDCG@10 / MRR), `answer` (Ollama LLM: non-stream +
  stream, grounding gate), `credentials` (0600 store under `~/.config/ragdesk/`),
  `envfile` (.env loader for dev), `buildenv` (bakes .env into a gitignored
  `_build_env.py` for release builds), `mcp` (stdio MCP server exposing
  search/document/sources to Claude Code & co), `serve` (loopback JSON API:
  status/search/ask/index/sync + connections connect/disconnect + optional
  static UI; hosts the auto-index timer), `settings` (app config in
  `~/.config/ragdesk/settings.json`, 0600: `auto_index_hours`, default 1,
  0 = off), `presets` (RAM tiers), `cli`.
- `desktop/` — Tauri 2 shell (card-catalog UI). Rust spawns `ragdesk serve`
  with `--db $HOME/.ragdesk/index.db`; a watchdog thread respawns it if it
  dies (skipping the respawn when another instance owns the port) and the
  child's output goes to `~/.ragdesk/serve.log`; env overrides: `RAGDESK_BIN`,
  `RAGDESK_DB`, `RAGDESK_LLM_MODEL`, `RAGDESK_PROJECT`. Browser mode:
  `ragdesk serve --ui desktop/dist`.
- Auto-index: `serve` runs a 60s timer; when `auto_index_hours` (Settings tab,
  default 1, 0 = off) has elapsed since `auto_index_last`, it re-indexes the
  recorded local roots via `run_auto_index` (connectors stay manual until
  their sync params are persisted).
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
  non-commercial). Defaults: `BAAI/bge-reranker-base` (MIT, fastembed) and
  `onnx-community/gte-multilingual-reranker-base` (Apache-2.0, `--rerank onnx`).
- Embedder/dimension mismatch must stay fail-closed (`Store.ensure_embedder`).
- Sources are read-only. No telemetry, ever.
- Eval numbers published in README must be reproducible from the repo.

## Roadmap order

sources (Notion → GitLab → OneDrive/SharePoint → S3) → multilingual reranker →
MCP server → release v0.1.0 (sidecar bundling + DMG release workflow + PyPI
trusted publisher) → eval badge automation → Windows/Linux builds.

## Notes

- If code is ever reused from OpsRAG (Apache-2.0), add the required NOTICE
  attribution before release.
- PyPI name `ragdesk` verified free (2026-09-14); re-check the similarity
  checker at publish time.
