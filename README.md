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

## What works today (honest list)

| Works | Not yet |
|---|---|
| Incremental local file indexing (hash-based) | Desktop UI (Tauri) |
| Hybrid retrieval: FTS5 BM25 + dense + RRF | GitHub / Confluence / Google Drive connectors |
| Reranking: `lexical` baseline + `fastembed` cross-encoder (`[onnx]` extra) | Multilingual reranker (`bge-reranker-v2-m3` is not in fastembed yet, upstream qdrant/fastembed#494) |
| ONNX embedder: EmbeddingGemma-300M int8, CPU (`[onnx]` extra) | RAM presets + model manager |
| Cited answers via local Ollama LLM + grounding gate | Confluence / GDrive connectors |
| Eval harness + CI gate (`--min-recall`) | Published eval badge automation |
| Fail-closed embedder/dimension guard | |

## Eval

Every number below is reproducible from the repo. `fixtures/` is a tiny
7-query smoke corpus; `fixtures/golden_repo.jsonl` is a 9-query golden set over
this repository's own docs and source.

| corpus / preset | recall@5 | nDCG@10 | MRR@10 |
|---|---|---|---|
| fixtures (7 queries) / hash-4096 — CI gate | 1.000 | 1.000 | 1.000 |
| repo (9 queries) / hash-4096 | 0.889 | 0.846 | 0.796 |
| repo (9 queries) / EmbeddingGemma-300M int8 (ONNX) | 1.000 | 0.918 | 0.889 |
| repo (9 queries) / EmbeddingGemma + `lexical` rerank | 1.000 | 0.862 | 0.815 |

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

1. Multilingual reranker backend (`bge-reranker-v2-m3` / `gte-multilingual-reranker-base`
   via ONNX or llama.cpp — not in fastembed yet)
2. GitHub connector (device flow, RFC 8628)
3. Tauri desktop shell + RAM presets (8GB floor, model manager)
4. Confluence connector (API token), Google Drive (BYO OAuth client + PKCE)
5. Published eval badge per release

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
