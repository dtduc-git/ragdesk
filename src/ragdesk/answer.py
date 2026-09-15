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
Cite the sources you used inline as [1], [2] and so on — end each factual
sentence with its citation marker, but never reply with citations alone.
If the sources do not contain the answer, say exactly:
"{refusal}". Never use outside knowledge.
{history}{memory}{diagram}
Sources:
{context}

Question: {question}
Answer:"""

HISTORY_HEADER = (
    "Recent conversation (context only — the sources are the only source of truth):\n"
)
HISTORY_TURNS = 3
HISTORY_CHARS = 400
MEMORY_HEADER = (
    "Durable notes the user asked you to remember (context, still answer from the sources):\n"
)
DIAGRAM_NOTE = """The user asked for a diagram. Reply with ONE fenced ```mermaid block and at
most two sentences of context. Use ONLY this Mermaid vocabulary: `flowchart TD` or
`flowchart LR`, nodes `id[Label]`, edges `A --> B` or `A -->|label| B`, optional grouping
`subgraph Name ... end`, plus `classDef accent fill:#2e6b58,color:#f6f7f2,stroke:#2e6b58;`
and `class NodeId accent;` for the node to notice first. Never invent keywords such as
`subregion`, `flowchart graph`, `linkStyle`, or HTML labels. At most 12 nodes, short labels
in the same language as the question.
"""
# Words that mean "draw me something" rather than "tell me something".
DIAGRAM_WORDS = (
    "diagram", "flowchart", "flow chart", "sequence diagram", "architecture diagram",
    "draw", "chart", "graph", "sơ đồ", "biểu đồ", "vẽ",
)


def wants_diagram(question: str) -> bool:
    lowered = question.lower()
    return any(word in lowered for word in DIAGRAM_WORDS)


def build_prompt(
    question: str,
    hits: list[Hit],
    history: list[tuple[str, str]] | None = None,
    memory: list[str] | None = None,
    diagram: bool = False,
) -> str:
    blocks = [
        f"[{i}] {hit.path}\n{hit.context}" for i, hit in enumerate(hits, start=1)
    ]
    lines: list[str] = []
    for role, text in (history or [])[-HISTORY_TURNS * 2 :]:
        speaker = "User" if role == "user" else "ragdesk"
        lines.append(f"{speaker}: {text[:HISTORY_CHARS]}")
    notes = [f"- {note}" for note in (memory or [])]
    return PROMPT_TEMPLATE.format(
        refusal=REFUSAL,
        history=(HISTORY_HEADER + "\n".join(lines) + "\n\n") if lines else "",
        memory=(MEMORY_HEADER + "\n".join(notes) + "\n\n") if notes else "",
        diagram=DIAGRAM_NOTE if diagram else "",
        context="\n\n".join(blocks),
        question=question,
    )


def answer(
    question: str,
    hits: list[Hit],
    llm: Any,
    *,
    min_cosine: float = 0.0,
    history: list[tuple[str, str]] | None = None,
    memory: list[str] | None = None,
    diagram: bool = False,
) -> str:
    best_cosine = max((hit.cosine for hit in hits), default=0.0)
    if not hits or best_cosine < min_cosine:
        return REFUSAL
    prompt = build_prompt(question, hits, history, memory, diagram)
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
    history: list[tuple[str, str]] | None = None,
    memory: list[str] | None = None,
    diagram: bool = False,
) -> Iterator[str]:
    """Same contract as :func:`answer`, but yields text pieces as they arrive."""
    best_cosine = max((hit.cosine for hit in hits), default=0.0)
    if not hits or best_cosine < min_cosine:
        yield REFUSAL
        return
    emitted = False
    prompt = build_prompt(question, hits, history, memory, diagram)
    for piece in llm.generate_stream(prompt, ANSWER_OPTIONS):
        if piece:
            emitted = True
            yield str(piece)
    if not emitted:
        yield REFUSAL
