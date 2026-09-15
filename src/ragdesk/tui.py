"""Terminal chat: the dev-facing front-end of the same core the GUI drives.

``chat_once`` does one exchange (retrieve → stream → record) and is unit
testable; ``run_chat`` is the REPL around it. Conversations live in the same
SQLite file as the desktop app, so history is shared between both.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, TextIO

from ragdesk.answer import REFUSAL, answer_stream, wants_diagram
from ragdesk.embed import Embedder
from ragdesk.search import parse_filters, retrieve
from ragdesk.store import Store

BANNER = "ragdesk chat — /new, /history, /sources, folder:… filters, /quit to leave"
HELP = (
    "/new starts a conversation · /history lists them · "
    "/sources shows the per-source counts · /quit exits"
)


def chat_once(
    store: Store,
    embedder: Embedder,
    llm: Any,
    question: str,
    chat_id: int = 0,
    *,
    reranker: Any = None,
    top_k: int = 6,
    out: Callable[[str], None] = print,
    stream: bool = True,
    style: dict[str, str] | None = None,
) -> tuple[int, str, list]:
    """One question → answer exchange. Returns ``(chat_id, text, hits)``."""
    paint = style or {}
    query, filters = parse_filters(question)
    if not query:
        out("please ask something after the filter, e.g. 'folder:Financial thuế'")
        return chat_id, "", []

    hits = retrieve(store, embedder, query, top_k=top_k, reranker=reranker, filters=filters)
    pieces: list[str] = []
    for piece in answer_stream(
        query,
        hits,
        llm,
        history=store.recent_turns(chat_id) if chat_id else None,
        diagram=wants_diagram(query),
    ):
        pieces.append(str(piece))
        if stream:
            sys.stdout.write(str(piece))
            sys.stdout.flush()
    text = "".join(pieces).strip() or REFUSAL
    if not stream:
        out(text)

    if not chat_id:
        chat_id = store.create_chat(query[:60])
    store.add_message(chat_id, "user", question)
    store.add_message(
        chat_id,
        "assistant",
        text,
        [
            {"path": hit.path, "line": hit.line, "text": hit.text, "lanes": hit.lanes}
            for hit in hits
        ],
    )
    if stream:
        out("")
    for rank, hit in enumerate(hits, start=1):
        where = f"{hit.path}:{hit.line}" if hit.line > 1 else hit.path
        out(paint.get("cite", "") + f"  [{rank}] {where}" + paint.get("reset", ""))
    return chat_id, text, hits


def run_chat(
    store: Store,
    embedder: Embedder,
    llm: Any,
    *,
    chat_id: int = 0,
    reranker: Any = None,
    top_k: int = 6,
    input_stream: TextIO | None = None,
    out: Callable[[str], None] = print,
    interactive: bool | None = None,
) -> int:
    """REPL loop; with a pipe on stdin it answers the piped questions once each."""
    source = input_stream or sys.stdin
    tty = source.isatty() if interactive is None else interactive
    style = {"cite": "\033[2m", "reset": "\033[0m"} if tty else {}
    if tty:
        out(BANNER)
    for raw in source:
        question = raw.strip()
        if not question:
            continue
        if question in ("/quit", "/exit"):
            break
        if question == "/help":
            out(HELP)
            continue
        if question == "/new":
            chat_id = 0
            out("new conversation")
            continue
        if question == "/history":
            for chat in store.chats(limit=10):
                marker = "*" if chat["id"] == chat_id else " "
                out(f"{marker} [{chat['id']}] {chat['title']} ({chat['messages']} messages)")
            continue
        if question == "/sources":
            for row in store.sources():
                out(f"  {row['source']:<14} {row['documents']:>5} docs  {row['chunks']:>6} chunks")
            continue
        chat_id, _text, _hits = chat_once(
            store,
            embedder,
            llm,
            question,
            chat_id,
            reranker=reranker,
            top_k=top_k,
            out=out,
            stream=tty,
            style=style,
        )
    return 0
