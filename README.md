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
# 1. install ([onnx] for CPU embeddings; [mlx] on Apple Silicon to run the
#    answer model in-process; [vision] for image OCR — all optional)
uv tool install 'ragdesk[onnx,mlx,vision] @ git+https://github.com/dtduc-git/ragdesk'

# 2. index your stuff (incremental, read-only)
ragdesk index ~/notes ~/repos/myrepo

# 3. ask — the backend ladder never re-downloads what you already have:
#    a running Ollama with qwen3.5:4b (or another model via RAGDESK_LLM) wins;
#    otherwise MLX runs mlx-community/Qwen3.5-4B-MLX-4bit in-process.
ragdesk ask "how does the deploy rollback work?"
ragdesk ask "..." --llm ollama:qwen3.5:9b        # force a specific backend
ragdesk ask "..." --llm mlx:some/hf-repo         # or a specific MLX repo

# no Ollama and no MLX? Indexing and search still work — only chat needs a model:
ragdesk --embedder onnx index ~/notes
# EmbeddingGemma runs on CPU via ONNX (downloads ~0.3GB once)

# 4. grounding gate (calibrate the cosine threshold per embedder)
ragdesk ask "..." --min-cosine 0.35

# retrieval only
ragdesk search "oauth pkce desktop"

# reranking (optional)
ragdesk --rerank lexical search "..."    # dependency-free baseline
ragdesk --rerank fastembed search "..."  # ONNX cross-encoder (model downloads on first use)

# numbers
ragdesk eval --golden fixtures/golden.jsonl
ragdesk stats
```

## Terminal chat (same index as the app)

```bash
ragdesk chat                    # REPL: streaming answers + citations
ragdesk chat --chat-id 12       # resume a conversation (shared with the GUI)
echo "what is RRF fusion?" | ragdesk chat   # script-friendly one-shot
```

`/new`, `/history`, `/sources` and the `folder:` / `key:value` filters work in
the REPL; answers stream, citations print as `path:line`.

## Use ragdesk from Claude Code, Claude Desktop or Codex (MCP)

`ragdesk mcp` speaks MCP over stdio, read-only, and reads the same index the
desktop app uses — the app does not need to be running.

```bash
# Claude Code
claude mcp add ragdesk -- ragdesk mcp

# Codex (~/.codex/config.toml)
[mcp_servers.ragdesk]
command = "ragdesk"
args = ["mcp"]

# Claude Desktop (claude_desktop_config.json)
{ "mcpServers": { "ragdesk": { "command": "ragdesk", "args": ["mcp"] } } }
```

Tools exposed: `ragdesk_search` (hybrid retrieval), `ragdesk_document` (full
text of one indexed file), `ragdesk_sources` (document/chunk counts per source).

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

Per-source connection guides — API tokens, browser consent, bring-your-own
OAuth apps and their caveats — live in **[docs/sources.md](docs/sources.md)**.

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

## Use the index from Claude Code / Cursor (MCP)

`ragdesk mcp` runs a dependency-free MCP server over stdio, so any MCP client
can search your local index:

```json
{
  "mcpServers": {
    "ragdesk": {
      "command": "ragdesk",
      "args": ["--embedder", "onnx", "--db", "/Users/you/.ragdesk/index.db", "mcp"]
    }
  }
}
```

Tools: `ragdesk_search` (hybrid retrieval with paths and scores),
`ragdesk_document` (full text of one indexed document), `ragdesk_sources`
(per-source document/chunk counts). Pass the same `--embedder` you indexed
with — the index refuses mismatched embeddings.

## What works today (honest list)

| Works | Not yet |
|---|---|
| Desktop app (Tauri 2): chat with history + streaming status + stop, sources, indexed stats (per source and per chosen path), settings, dark theme — **built from source** (no DMG release yet) | Packaged DMG + notarization (Apple Developer ID) when the project ships binaries |
| Auto re-index of the chosen local paths every N hours (Settings, default 1h, Off switch) | Connector auto-sync (local paths only for now) |
| Idle unload: models leave RAM after a quiet stretch (Settings, default 15 min) | |
| Metadata: a `--- key: value ---` front-matter header is parsed, stored per document and filterable — no YAML dependency | |
| Indexing: local files (native picker), **PDF / DOCX / PPTX / XLSX text extraction** (sheets keep row refs; legacy `.xls` and scanned PDFs need converting), **image OCR** (screenshots, scans, photos with text — Apple Vision, on-device, no model download), GitHub repos (device code / gh / token), GitLab repos (token), Confluence spaces (connect + CQL), Google Drive (connect + doc export), Microsoft OneDrive/SharePoint (device flow), Notion (shared pages), website crawl (same-host, HTML) | Legacy `.xls`, audio; OCR for scanned PDFs; VLM captions for text-free images; sidecar bundling in the DMG |
| Hybrid retrieval: FTS5 BM25 + EmbeddingGemma int8 (ONNX) + RRF | Windows / Linux builds |
| Reranking: `lexical` baseline, `fastembed` (English-first), `onnx` multilingual gte (70+ languages) | Eval badge automation per release |
| Grounded cited answers with a backend ladder: reuses Ollama when the model is there, else MLX in-process; one-click model download in Settings (live progress) | |
| Chat history: conversations in SQLite, multi-turn context, a history popover (open/delete), resume or start fresh | |
| Live progress: phased status while answering (searching → thinking, elapsed seconds) with a Stop button; sync/index activity in the rail | |
| Answer cache: an identical question on an unchanged corpus replays instantly | |
| Memory: durable notes you add (or extract from a chat) ride along with every answer | |
| Semantic answer cache: a paraphrase of an answered question replays instantly (cosine ≥ 0.88, calibrated) | |
| Answer engines: Ollama, local MLX, or any OpenAI-compatible endpoint (LM Studio, llama.cpp, vLLM, OpenAI) — switchable in Settings, no terminal | |
| First-run wizard: folders → answer engine → RAM-sized preset → index, all in the UI (re-runnable from Settings) | |
| Diagrams on request: "vẽ sơ đồ …" returns a themed Mermaid figure inline, with SVG download | Charts beyond Mermaid's set |
| HyDE lane (Settings, off by default): drafts an answer with the local model, then searches with it too | |
| Scoping: `folder:` / `source:` and any front-matter key (`type:runbook service:payments`) as query filters; metadata shows as chips on citations | |
| Parent-child context: children are embedded, parents (~4k chars) go to the LLM | |
| Eval: per-category metrics + category gates, plus `--answers` faithfulness scoring | |
| MCP server for Claude Code / Cursor (`ragdesk mcp`) | |
| RAM presets (`light` / `balanced` / `quality`) — switchable in Settings, applied live; per-flag overrides still work | |
| Eval harness + CI gates on the fixtures and repo golden sets | |

## Eval

Every number below is reproducible from the repo. `fixtures/` is a tiny
7-query smoke corpus; `fixtures/golden_repo.jsonl` is a 9-query golden set over
this repository's own docs and source.

| corpus / preset | recall@5 | nDCG@10 | MRR@10 |
|---|---|---|---|
| fixtures (7 queries) / hash-4096 — CI gate | 1.000 | 1.000 | 1.000 |
| corpus (24 queries: VN notes, PDFs, two codebases) | 1.000 | 0.964 | 0.951 |
| repo (12 queries, `folder:`-scoped) / EmbeddingGemma int8, text chunking | **0.917** | **0.874** | **0.833** |
| repo (12 queries) / + smart retrieval (rewrite + HyDE + sub-queries, one call) | 0.917 | 0.832 | 0.778 |

Per-category breakdown (the `[eval]` row is the honest weak spot):

```
[answers] n=1 recall@5=1.000   [code] n=3 recall@5=1.000
[eval]    n=2 recall@5=0.500   [general] n=3 recall@5=1.000
```

Measured 2026-09-15 on the repo tree (824-document corpus including a synced
repo, hence the `folder:` scoping in the golden set).

Tuning measured with `scripts/bench.py` (fresh index per chunking config, the
same 12 scoped queries):

| lever | recall@5 | nDCG@10 | MRR@10 | verdict |
|---|---|---|---|---|
| chunk 600 chars (4158 chunks) | 0.833 | **0.792** | **0.778** | ranking up, recall down |
| chunk 1000 chars (2288) — default | 0.917 | 0.746 | 0.688 | kept |
| chunk 1500 chars (1464) | 0.917 | 0.724 | 0.660 | no gain |
| rerank `fastembed` bge-reranker-base (EN) | 0.750 | 0.708 | 0.694 | hurts a VN+code corpus |
| rerank `onnx` gte-multilingual | **0.917** | 0.782 | 0.739 | **kept for the quality preset** |
| rerank `lexical` (baseline) | 0.750 | 0.538 | 0.465 | test baseline only |
| lane weights (path 0.6 / bm25 0.9 / dense 1.2) | 0.833 | — | — | no effect; equal weights stay |

- **Symbol-aware code chunking was tried three ways and rejected**, each
  measured on fresh indexes of the same corpus (plain baseline: recall 1.000 /
  nDCG 0.819 / MRR 0.757): per-symbol segmentation (0.743 / 0.656 — fragments
  the file and raises BM25 term density), symbol headers inside chunks
  (0.788 / 0.715 — test files mention symbols in their names and out-rank the
  implementation), and a symbol lane over definition sites (0.523 / 0.419 —
  loose LIKE matching dilutes RRF). The path lane already answers "where is X
  defined"; symbol intelligence belongs in a call-graph index. Chunks still
  carry `line_start`, so citations show `file:line`.
- **Smart retrieval is a toggle, off by default.** One local-model call that
  rewrites a follow-up, drafts a hypothetical answer and proposes sub-queries.
  It helped on a noisier corpus (+0.031 nDCG) and cost a little on the current
  one (−0.042), which is why the reader decides.
- The 24-query `fixtures/golden_corpus.jsonl` set is generated by
  `scripts/make_golden.py` from corpus chunks (questions a reader could ask
  about a passage, file names filtered out). It is a **regression set** — every
  document must stay findable — not a challenge set; the hand-written
  12-query repo golden is the harder one.
- The dependency-free `lexical` reranker **lowers** nDCG/MRR here — it is a
  test baseline, not a quality feature.
- CI gates `recall@5 >= 0.8` on both harnesses, so retrieval regressions fail
  the build.

Reproduce (first run downloads the ~0.3 GB int8 model):

```bash
uv run ragdesk --embedder onnx --db /tmp/eval.db index README.md AGENTS.md SECURITY.md src .github fixtures/docs
uv run ragdesk --embedder onnx --db /tmp/eval.db eval --golden fixtures/golden_repo.jsonl
uv run ragdesk --embedder onnx --db /tmp/eval.db --rerank lexical eval --golden fixtures/golden_repo.jsonl
```

## Roadmap

1. Release v0.1.0: PyPI (trusted publishing) + desktop DMG on GitHub Releases
2. Eval badge automation per release
3. More sources, following the Bedrock Knowledge Base connector set as a reference:
   S3 (bucket/prefix)
4. Windows / Linux builds
5. MCP-server expansion path for connectors

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
