"""Terminal chat: the dev-facing front-end of the same core the GUI drives.

``chat_once`` does one exchange (retrieve → stream → record) and is unit
testable; ``run_chat`` is the REPL around it. Conversations live in the same
SQLite file as the desktop app, so history is shared between both.

With ``--server URL`` the REPL talks to a *running* ragdesk server instead of
opening the database itself: the models are already loaded there, so a second
process costs no extra RAM and both front-ends share one index.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, TextIO

from ragdesk.answer import REFUSAL, answer_stream, wants_diagram
from ragdesk.embed import Embedder
from ragdesk.search import hit_to_dict, parse_filters, retrieve
from ragdesk.store import Store

BANNER = "ragdesk chat — /new, /history, /sources, folder:… filters, /quit to leave"
HELP = (
    "/new starts a conversation · /history lists them · "
    "/sources shows the per-source counts · /quit exits"
)
REMOTE_BANNER = "ragdesk chat (attached) — /new, /quit to leave"


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
        [hit_to_dict(hit) for hit in hits],
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


def ask_remote(
    base_url: str,
    question: str,
    *,
    chat_id: int = 0,
    out: Callable[[str], None] = print,
    stream: bool = True,
    timeout: float = 600.0,
) -> tuple[int, str, list]:
    """One exchange against a running server; returns (chat_id, text, hits)."""
    payload = json.dumps({"query": question, "chat_id": chat_id}).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/ask/stream",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    text = ""
    hits: list = []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                if event.get("delta"):
                    text += str(event["delta"])
                    if stream:
                        sys.stdout.write(str(event["delta"]))
                        sys.stdout.flush()
                if event.get("done"):
                    chat_id = int(event.get("chat_id") or chat_id)
                    hits = list(event.get("hits") or [])
    except urllib.error.URLError as exc:
        raise RuntimeError(f"cannot reach the server at {base_url}: {exc}") from exc
    if not stream:
        out(text)
    return chat_id, text, hits


def run_remote_chat(
    base_url: str,
    *,
    chat_id: int = 0,
    input_stream: TextIO | None = None,
    out: Callable[[str], None] = print,
    interactive: bool | None = None,
) -> int:
    """REPL that talks to a running server instead of opening the database."""
    source = input_stream or sys.stdin
    tty = source.isatty() if interactive is None else interactive
    style = {"cite": "\033[2m", "reset": "\033[0m"} if tty else {}
    if tty:
        out(REMOTE_BANNER)
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
        try:
            chat_id, text, hits = ask_remote(
                base_url, question, chat_id=chat_id, out=out, stream=tty
            )
        except (RuntimeError, urllib.error.URLError) as exc:
            out(f"error: {exc}")
            continue
        if tty:
            out("")
        for rank, hit in enumerate(hits, start=1):
            where = hit.get("path", "")
            if hit.get("line") and int(hit["line"]) > 1:
                where = f"{where}:{hit['line']}"
            out(style.get("cite", "") + f"  [{rank}] {where}" + style.get("reset", ""))
    return 0
