"""Grounded answer generation over any :mod:`ragdesk.llm` backend.

The grounding gate refuses to call the LLM when retrieval is weak — the
cosine threshold is embedder-specific, so calibrate it per embedder with the
eval harness instead of trusting the default.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ragdesk.search import Hit

DEFAULT_LLM_MODEL = "qwen3.5:4b"
# Grounded answers are short; the cap also stops small models from looping
# (a 1k-token ramble holds the model for minutes on laptop hardware).
# ``num_ctx`` is Ollama-specific and ignored by other backends.
ANSWER_OPTIONS = {"num_predict": 400, "temperature": 0.2, "num_ctx": 8192}

REFUSAL = "I could not find this in your indexed sources."

PROMPT_TEMPLATE = """You are ragdesk, a retrieval assistant. Answer ONLY from the sources below.
Write a complete answer in plain prose (no LaTeX, no markdown headings or tables).
Cite the sources you used inline as [1], [2] and so on — but never reply with
citations alone. If the sources do not contain the answer, say exactly:
"{refusal}". Never use outside knowledge.

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
    llm: Any,
    *,
    min_cosine: float = 0.0,
) -> str:
    best_cosine = max((hit.cosine for hit in hits), default=0.0)
    if not hits or best_cosine < min_cosine:
        return REFUSAL
    prompt = build_prompt(question, hits)
    # Small models occasionally return an empty completion; one retry.
    for _ in range(2):
        text = str(llm.generate(prompt, ANSWER_OPTIONS)).strip()
        if text:
            return text
    return REFUSAL


def answer_stream(
    question: str,
    hits: list[Hit],
    llm: Any,
    *,
    min_cosine: float = 0.0,
) -> Iterator[str]:
    """Same contract as :func:`answer`, but yields text pieces as they arrive."""
    best_cosine = max((hit.cosine for hit in hits), default=0.0)
    if not hits or best_cosine < min_cosine:
        yield REFUSAL
        return
    emitted = False
    for piece in llm.generate_stream(build_prompt(question, hits), ANSWER_OPTIONS):
        if piece:
            emitted = True
            yield str(piece)
    if not emitted:
        yield REFUSAL
