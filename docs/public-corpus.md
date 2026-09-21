# Public corpus: 12 famous repositories, 101 questions

The repo golden is 12 questions about ragdesk itself. This is the independent
one: does retrieval hold up on code and documentation written by other people,
at a personal-corpus scale?

## Method

- **12 public repositories**, shallow clones: kubernetes, django, react,
  fastapi, go, vscode, llama.cpp, langchain, prometheus, helm, terraform,
  cpython.
- **200 files per repository**, chosen deterministically: sha256 of the path
  over the files ragdesk would itself admit at that size, vendor/testdata/
  build trees skipped. 2,378 documents, 34,800 chunks.
- **Questions generated locally** (MLX Qwen3.5-4B) from one passage each, one
  question per source document, in the passage's language. Rejected: anything
  that leaks a file name or path, near-duplicates, and too-short/too-long
  output. Every query is scoped to its repository (`folder:`), so a
  multi-repo index cannot inflate the score. **101 questions** survived.
- **Scoring** is `ragdesk eval` — the same code path the CI gate runs
  (`recall@5`, `nDCG@10`, `MRR@10`).

## Corpus

| repository | documents | chunks |
|---|---|---|
| kubernetes/kubernetes | 200 | 2,884 |
| django/django | 197 | 2,549 |
| facebook/react | 199 | 968 |
| fastapi/fastapi | 189 | 1,465 |
| golang/go | 200 | 2,572 |
| microsoft/vscode | 199 | 5,101 |
| ggml-org/llama.cpp | 197 | 4,112 |
| langchain-ai/langchain | 200 | 2,258 |
| prometheus/prometheus | 200 | 3,789 |
| helm/helm | 200 | 1,867 |
| hashicorp/terraform | 198 | 2,258 |
| python/cpython | 199 | 4,977 |
| **total** | **2,378** | **34,800** |

## Results

EmbeddingGemma int8, text chunking (1000/150), the shipped pipeline — only the
reranker changes:

| rerank | recall@5 | nDCG@10 | MRR@10 |
|---|---|---|---|
| none | 0.921 | 0.849 | 0.815 |
| mmarco-mMiniLMv2 (light / balanced default) | **0.960** | 0.895 | 0.869 |
| gte-multilingual (quality preset) | 0.950 | **0.930** | **0.919** |

What it says:

- The reranker earns its keep on a corpus it has never seen: +0.039 recall@5
  for the small multilingual model that ships in the two light presets.
- gte ranks slightly better (nDCG/MRR) but recovers fewer documents in the top
  five — the same trade the repo golden showed, now on foreign code.
- 0.921 with no reranker at all means the hybrid lane (BM25 + dense + path,
  RRF-fused) is doing the heavy lifting, not the reranker.

Per-repository scores are deliberately not published — only the aggregate
above. The golden set itself (`fixtures/golden_public.jsonl`) is in the repo so
the run is reproducible.

## Reproduce

```bash
# 12 shallow clones (~1.7 GB), one scratch index
uv run --extra onnx python scripts/public_corpus.py clone
uv run --extra onnx python scripts/public_corpus.py index

# 101 questions from that index (needs a local model: MLX or Ollama)
uv run --extra onnx --extra mlx python scripts/make_golden.py \
    --db /tmp/ragdesk-public/public.db --repos-dir /tmp/ragdesk-public/repos \
    --per-group 12 --out fixtures/golden_public.jsonl

# score all three rerank configurations
uv run --extra onnx ragdesk --db /tmp/ragdesk-public/public.db eval \
    --golden fixtures/golden_public.jsonl                              # no reranker
uv run --extra onnx ragdesk --db /tmp/ragdesk-public/public.db \
    --rerank onnx:cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 eval \
    --golden fixtures/golden_public.jsonl                              # light/balanced
uv run --extra onnx ragdesk --db /tmp/ragdesk-public/public.db \
    --rerank onnx:onnx-community/gte-multilingual-reranker-base eval \
    --golden fixtures/golden_public.jsonl                              # quality
```

The clones and the scratch index are throwaway: delete `/tmp/ragdesk-public`
when done. Numbers above measured 2026-09-22 on an 8-core Apple Silicon
machine.
