"""Grounded answer generation via a local Ollama LLM (optional).

The grounding gate refuses to call the LLM when retrieval is weak — the
cosine threshold is embedder-specific, so calibrate it per embedder with the
eval harness instead of trusting the default.
"""

from __future__ import annotations

from collections.abc import Iterator

from ragdesk.ollama import DEFAULT_HOST, post_json, post_stream
from ragdesk.search import Hit

DEFAULT_LLM_MODEL = "qwen3.5:4b"

REFUSAL = "I could not find this in your indexed sources."

PROMPT_TEMPLATE = """You are ragdesk, a retrieval assistant. Answer ONLY from the sources below.
Cite sources as [1], [2] and so on. If the sources do not contain the answer, say
exactly: "{refusal}". Never use outside knowledge.

Sources:
{context}

Question: {question}
Answer:"""


def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = [f"[{i}] {hit.path}\n{hit.text}" for i, hit in enumerate(hits, start=1)]
    return PROMPT_TEMPLATE.format(
        refusal=REFUSAL,
        context="\n\n".join(blocks),
        question=question,
    )


def answer(
    question: str,
    hits: list[Hit],
    model: str = DEFAULT_LLM_MODEL,
    host: str = DEFAULT_HOST,
    min_cosine: float = 0.0,
) -> str:
    best_cosine = max((hit.cosine for hit in hits), default=0.0)
    if not hits or best_cosine < min_cosine:
        return REFUSAL
    data = post_json(
        host,
        "/api/generate",
        {"model": model, "prompt": build_prompt(question, hits), "stream": False},
    )
    return data["response"].strip()


def answer_stream(
    question: str,
    hits: list[Hit],
    model: str = DEFAULT_LLM_MODEL,
    host: str = DEFAULT_HOST,
    min_cosine: float = 0.0,
) -> Iterator[str]:
    """Same contract as :func:`answer`, but yields text pieces as they arrive."""
    best_cosine = max((hit.cosine for hit in hits), default=0.0)
    if not hits or best_cosine < min_cosine:
        yield REFUSAL
        return
    payload = {"model": model, "prompt": build_prompt(question, hits), "stream": True}
    for chunk in post_stream(host, "/api/generate", payload):
        piece = chunk.get("response", "")
        if piece:
            yield str(piece)
