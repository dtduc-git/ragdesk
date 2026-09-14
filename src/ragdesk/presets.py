"""RAM-tier presets: embedder / reranker / LLM defaults by machine budget."""

from __future__ import annotations

PRESETS: dict[str, dict[str, str]] = {
    "light": {
        "embedder": "onnx",
        "rerank": "none",
        "llm": "qwen3.5:4b",
        "llm_mlx": "mlx-community/Qwen3.5-4B-MLX-4bit",
        "note": "8GB machines: EmbeddingGemma int8 on CPU, no reranker, 4B LLM",
    },
    "balanced": {
        "embedder": "onnx",
        "rerank": "fastembed:BAAI/bge-reranker-base",
        "llm": "qwen3.5:4b",
        "llm_mlx": "mlx-community/Qwen3.5-4B-MLX-4bit",
        "note": "16GB machines: adds a cross-encoder reranker",
    },
    "quality": {
        "embedder": "onnx",
        "rerank": "onnx",
        "llm": "qwen3.5:9b",
        "llm_mlx": "mlx-community/Qwen3.5-9B-4bit",
        "note": "32GB / GPU: larger LLM plus the multilingual ONNX reranker (gte)",
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
