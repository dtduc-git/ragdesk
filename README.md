# ragdesk

Personal, local-first RAG over your own sources. Index your notes, repos and
docs — ask in natural language, get cited answers. Everything runs on your
machine, and **every release publishes its retrieval numbers**.

> **Status: pre-alpha (0.1.0).** Retrieval core, eval harness, connectors
> (local, GitHub, Confluence, Google Drive), streaming answers and a Tauri
> desktop app are in. Not on PyPI yet — run from source.

## Why another personal RAG?

Because "runs locally" is not a quality claim. ragdesk ships a retrieval eval
harness from day one and publishes recall@5 / nDCG@10 / MRR, so quality is a
number you can check — not a vibe. The core is stdlib-only Python: hybrid
search (SQLite FTS5 BM25 + embeddings + RRF fusion) in a single SQLite file.

## Quickstart

```bash
# 1. local models via Ollama (embedding ~330MB, LLM ~3GB)
ollama pull embeddinggemma:300m
ollama pull qwen3.5:4b

# 2. install (pre-release: from source; [onnx] extra for local ONNX models)
uv tool install 'ragdesk[onnx] @ git+https://github.com/dtduc-git/ragdesk'

# 3. index your stuff (incremental, read-only)
ragdesk index ~/notes ~/repos/myrepo

# no Ollama? EmbeddingGemma runs on CPU via ONNX (downloads ~0.3GB once):
ragdesk --embedder onnx index ~/notes

# 4. ask (grounded + cited, refuses when context is weak)
ragdesk ask "how does the deploy rollback work?"
ragdesk ask "..." --min-cosine 0.35   # grounding gate (calibrate per embedder)

# retrieval only
ragdesk search "oauth pkce desktop"

# reranking (optional)
ragdesk --rerank lexical search "..."    # dependency-free baseline
ragdesk --rerank fastembed search "..."  # ONNX cross-encoder (model downloads on first use)

# numbers
ragdesk eval --golden fixtures/golden.jsonl
ragdesk stats
```

## Desktop app (Tauri 2)

```bash
# one-time: make the CLI visible to the packaged app
uv tool install ".[onnx]"

cd desktop
npm install
npm run tauri dev     # dev window; spawns the local server automatically
npm run tauri build   # .app + .dmg on macOS
```

The shell spawns `ragdesk serve` (loopback only) and renders the card-catalog
UI: streaming chat with citations, hybrid search, source connectors, and the
status/preset panel.

- **Local sources** use a native folder/file picker (multi-select), so you
  don't paste paths.
- **Cloud sources have a one-time Connect flow**: GitHub (device code — a
  client ID ships with the app, so no CLI is needed; `gh` login and tokens
  also work), Confluence (API token — no app registration — or your own
  Atlassian OAuth app for one-click consent), Google Drive (BYO OAuth client,
  browser consent). Credentials are stored `0600` under `~/.config/ragdesk/`
  and can be disconnected from the same card. Atlassian client secrets are
  never shipped in the repository (see `SECURITY.md`).

The same UI also runs in a browser for development:
`uv run ragdesk serve --ui desktop/dist`.

## What works today (honest list)

| Works | Not yet |
|---|---|
| Desktop app (Tauri 2): chat, search, sources, settings, streaming answers | PyPI release (v0.1.0 pending) |
| Indexing: local files (native picker), GitHub repos (device code / gh / token + tarball sync), Confluence spaces (connect + CQL), Google Drive (connect + doc export), website crawl (same-host, HTML) | GitLab connector |
| Hybrid retrieval: FTS5 BM25 + EmbeddingGemma int8 (ONNX) + RRF | Multilingual reranker (`bge-reranker-v2-m3` not in fastembed yet) |
| Reranking: `lexical` baseline + `fastembed` cross-encoder (`[onnx]` extra) | Windows / Linux builds |
| Grounded cited answers via local Ollama, grounding gate, token streaming | Eval badge automation per release |
| RAM presets (`light` / `balanced` / `quality`) with per-flag overrides | |
| Eval harness + CI gates on the fixtures and repo golden sets | |

## Eval

Every number below is reproducible from the repo. `fixtures/` is a tiny
7-query smoke corpus; `fixtures/golden_repo.jsonl` is a 9-query golden set over
this repository's own docs and source.

| corpus / preset | recall@5 | nDCG@10 | MRR@10 |
|---|---|---|---|
| fixtures (7 queries) / hash-4096 — CI gate | 1.000 | 1.000 | 1.000 |
| repo (9 queries) / hash-4096 | 0.889 | 0.832 | 0.778 |
| repo (9 queries) / EmbeddingGemma-300M int8 (ONNX) | 1.000 | 0.918 | 0.889 |
| repo (9 queries) / EmbeddingGemma + `lexical` rerank | 1.000 | 0.862 | 0.815 |

Measured 2026-09-14 on the repo tree; doc edits shift these by ~1 query, and
they are re-measured on every release.

Honest notes: the tiny fixtures corpus saturates, so the repo golden set is the
one that says something. The dependency-free `lexical` reranker **lowers**
nDCG/MRR here — it is a test baseline, not a quality feature; a real
cross-encoder reranker is the next milestone. CI gates `recall@5 >= 0.8` on
both harnesses, so retrieval regressions fail the build.

Reproduce (first run downloads the ~0.3 GB int8 model):

```bash
uv run ragdesk --embedder onnx --db /tmp/eval.db index README.md AGENTS.md SECURITY.md src .github fixtures/docs
uv run ragdesk --embedder onnx --db /tmp/eval.db eval --golden fixtures/golden_repo.jsonl
uv run ragdesk --embedder onnx --db /tmp/eval.db --rerank lexical eval --golden fixtures/golden_repo.jsonl
```

## Roadmap

1. Release v0.1.0: PyPI (trusted publishing) + desktop DMG on GitHub Releases
2. Eval badge automation per release
3. Multilingual reranker backend (`bge-reranker-v2-m3` / `gte-multilingual-reranker-base`)
4. More sources, following the Bedrock Knowledge Base connector set as a reference:
   S3 (bucket/prefix), SharePoint / OneDrive (Microsoft Graph), Notion
5. Windows / Linux builds
6. MCP-server expansion path for connectors

## Non-goals

- No server, no multi-user, no account.
- No telemetry — nothing leaves your machine unless you point it at a service.
- Sources are read-only; ragdesk never writes to what it indexes.
- No graph database, no agent loop in v0.1.

## Development

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
```

## Security & privacy

Local-first by default: the index is one SQLite file under `.ragdesk/`.
Network calls happen only to your local Ollama server; adding cloud providers
is opt-in and bring-your-own-key. Connection credentials live in
`~/.config/ragdesk/credentials.json` (mode `0600`). See [SECURITY.md](SECURITY.md).

## License

Apache-2.0.
