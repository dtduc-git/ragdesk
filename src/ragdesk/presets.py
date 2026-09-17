"""RAM-tier presets: embedder / reranker / LLM defaults by machine budget."""

from __future__ import annotations

PRESETS: dict[str, dict[str, str]] = {
    "light": {
        "embedder": "onnx",
        # mmarco-mMiniLMv2 (mMARCO includes Vietnamese): measured 2026-09-17 on
        # 15 real-document questions, recall@5 0.933 -> 1.000, nDCG@10
        # 0.871 -> 0.937, for +34MB resident and ~0.8s per question. Cheap
        # enough that even the 8GB preset reranks; the session unloads when idle.
        "rerank": "onnx:cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "llm": "qwen3.5:4b",
        "llm_mlx": "mlx-community/Qwen3.5-4B-MLX-4bit",
        "note": "8GB machines: int8 embedder, the small multilingual reranker, 4B LLM",
    },
    "balanced": {
        "embedder": "onnx",
        "rerank": "onnx:cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "llm": "qwen3.5:4b",
        "llm_mlx": "mlx-community/Qwen3.5-4B-MLX-4bit",
        "note": "16GB machines: same reranker, room for the bigger model later",
    },
    "quality": {
        "embedder": "onnx",
        # The heavy option: gte-multilingual scored a perfect 1.000/1.000/1.000
        # on the real-document golden (2026-09-17), at 341MB on disk and ~1.5GB
        # resident — the reason it is not the default anywhere else.
        "rerank": "onnx:onnx-community/gte-multilingual-reranker-base",
        "llm": "qwen3.5:9b",
        "llm_mlx": "mlx-community/Qwen3.5-9B-4bit",
        "note": "32GB / GPU: the strongest reranker (gte) plus a 9B LLM",
    },
}
DEFAULT_PRESET = "light"


def resolve(
    preset: str,
    *,
    embedder: str | None = None,
    rerank: str | None = None,
    llm: str | None = None,
) -> dict[str, str]:
    """Explicit flags win over the preset; the preset wins over universal defaults."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset: {preset!r} (use {', '.join(PRESETS)})")
    selected = PRESETS[preset]
    return {
        "preset": preset,
        "embedder": embedder or selected["embedder"],
        "rerank": rerank or selected["rerank"],
        "llm": llm or selected["llm"],
    }
