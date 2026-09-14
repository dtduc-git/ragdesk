# ragdesk

Personal, local-first RAG over your own sources. Index your notes, repos and
docs — ask in natural language, get cited answers. Everything runs on your
machine, and **every release publishes its retrieval numbers**.

> **Status: pre-alpha (0.1.0).** Retrieval core + eval harness landed. The
> desktop shell and cloud connectors are on the roadmap.

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

# 2. install (pre-release: from source)
uv tool install git+https://github.com/dtduc-git/ragdesk

# 3. index your stuff (incremental, read-only)
ragdesk index ~/notes ~/repos/myrepo

# 4. ask (grounded + cited, refuses when context is weak)
ragdesk ask "how does the deploy rollback work?"
ragdesk ask "..." --min-cosine 0.35   # grounding gate (calibrate per embedder)

# retrieval only
ragdesk search "oauth pkce desktop"

# numbers
ragdesk eval --golden fixtures/golden.jsonl
ragdesk stats
```

## What works today (honest list)

| Works | Not yet |
|---|---|
| Incremental local file indexing (hash-based) | Desktop UI (Tauri) |
| Hybrid retrieval: FTS5 BM25 + dense + RRF | GitHub / Confluence / Google Drive connectors |
| Cited answers via local Ollama LLM + grounding gate | ONNX embedder backend (EmbeddingGemma int8 direct) |
| Eval harness + CI gate (`--min-recall`) | Reranker (bge-reranker-v2-m3 planned) |
| Fail-closed embedder/dimension guard | RAM presets + model manager |

## Eval

Measured on the in-repo fixtures (`fixtures/golden.jsonl`, 7 queries) with the
deterministic offline `hash-4096` embedder — this is the CI baseline, not a
quality claim:

| preset | recall@5 | nDCG@10 | MRR@10 |
|---|---|---|---|
| fixtures / hash-4096 | 1.000 | 1.000 | 1.000 |

CI gates `recall@5 >= 0.8` on this harness, so retrieval regressions fail the
build.

Reproduce:

```bash
uv run ragdesk --embedder hash --db /tmp/eval.db index fixtures/docs
uv run ragdesk --embedder hash --db /tmp/eval.db eval --golden fixtures/golden.jsonl
```

Real-corpus numbers (EmbeddingGemma + Ollama) land with the eval badge in an
upcoming release.

## Roadmap

1. Reranker stage (bge-reranker-v2-m3, ONNX)
2. ONNX embedder backend (EmbeddingGemma-300M int8)
3. GitHub connector (device flow, RFC 8628)
4. Tauri desktop shell + RAM presets (8GB floor, model manager)
5. Confluence connector (API token), Google Drive (BYO OAuth client + PKCE)
6. Published eval badge per release

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
is opt-in and bring-your-own-key. See [SECURITY.md](SECURITY.md).

## License

Apache-2.0.
