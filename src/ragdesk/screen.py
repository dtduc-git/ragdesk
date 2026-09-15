"""Full-screen terminal chat (Textual) — the ``tui`` extra.

A single screen: transcript, prompt, status line. Answers stream in as they
are written, citations land under each answer, and the same SQLite file as the
desktop app is used, so conversations are shared between both front-ends.

This module is only imported by ``ragdesk tui``; without the extra installed
the CLI prints how to install it instead of a traceback.
"""

from __future__ import annotations

from typing import Any

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input, Static

from ragdesk import __version__
from ragdesk.answer import REFUSAL, answer_stream, wants_diagram
from ragdesk.embed import Embedder
from ragdesk.search import hit_to_dict, parse_filters, retrieve
from ragdesk.store import Store

HELP = "/new starts a conversation · /sources shows the index · /help · /quit exits"
PROMPT = "Ask about your sources… (folder: filters · /help)"


class ChatScreen(App):
    """ragdesk in the terminal: the same core, a full-screen shell around it."""

    CSS = """
    Screen { layout: vertical; }
    #transcript { height: 1fr; padding: 1 2; }
    #status { height: 1; padding: 0 2; color: $text-muted; }
    #prompt { dock: bottom; }
    .question { color: $accent; text-style: bold; }
    .answer { margin: 0 0 1 0; }
    .cite { color: $text-muted; }
    .error { color: $error; }
    """

    BINDINGS = [
        ("ctrl+n", "new_chat", "New chat"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        *,
        db: str,
        embedder: Embedder,
        llm: Any,
        reranker: Any = None,
        top_k: int = 6,
        chat_id: int = 0,
    ) -> None:
        super().__init__()
        self.db = db
        self.embedder = embedder
        self.llm = llm
        self.reranker = reranker
        self.top_k = top_k
        self.chat_id = chat_id
        self.busy = False
        self._answer_widget: Static | None = None
        self._answer_text = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with VerticalScroll(id="transcript"):
            yield Static("", id="intro")
        yield Static("", id="status")
        yield Input(placeholder=PROMPT, id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "ragdesk"
        with Store(self.db) as store:
            stats = store.stats()
        self.query_one("#intro", Static).update(
            f"ragdesk {__version__} — {stats['documents']} documents · "
            f"{stats['chunks']} chunks\n{HELP}"
        )
        self.query_one("#prompt", Input).focus()

    # --- rendering helpers ----------------------------------------------------

    def write_line(self, text: str, css_class: str = "") -> None:
        line = Static(text, classes=css_class, markup=False)
        self.query_one("#transcript", VerticalScroll).mount(line)
        self.call_after_refresh(line.scroll_visible)

    def set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    # --- input -----------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        question = event.value.strip()
        event.input.value = ""
        if not question:
            return
        if question.startswith("/"):
            self.command(question)
            return
        if self.busy:
            self.set_status("still answering — Ctrl+Q to quit")
            return
        self.busy = True
        self._answer_widget = None
        self._answer_text = ""
        self.write_line(f"> {question}", "question")
        self.set_status("thinking…")
        self.run_worker(
            lambda: self.exchange(question), thread=True, exclusive=True
        )

    def command(self, raw: str) -> None:
        name = raw.split()[0]
        if name in ("/quit", "/exit"):
            self.exit()
            return
        if name == "/new":
            self.action_new_chat()
            return
        if name == "/help":
            self.write_line(HELP)
            return
        if name == "/sources":
            with Store(self.db) as store:
                rows = store.sources()
            for row in rows:
                self.write_line(
                    f"  {row['source']:<16} {row['documents']:>5} docs  "
                    f"{row['chunks']:>6} chunks"
                )
            return
        self.write_line(f"unknown command: {name} — {HELP}", "error")

    def action_new_chat(self) -> None:
        self.chat_id = 0
        self.write_line("new conversation")

    # --- the exchange (worker thread) -----------------------------------------

    def exchange(self, question: str) -> None:
        query, filters = parse_filters(question)
        if not query:
            self.call_from_thread(self.write_line, "ask something after the filter", "error")
            self.call_from_thread(self.busy_done, "")
            return
        try:
            # A worker thread gets its own SQLite connection: never share one
            # across threads, and never block the UI on retrieval.
            with Store(self.db) as store:
                history = store.recent_turns(self.chat_id) if self.chat_id else []
                hits = retrieve(
                    store,
                    self.embedder,
                    query,
                    top_k=self.top_k,
                    reranker=self.reranker,
                    filters=filters,
                )
                pieces: list[str] = []
                for piece in answer_stream(
                    query,
                    hits,
                    self.llm,
                    history=history,
                    diagram=wants_diagram(query),
                ):
                    pieces.append(str(piece))
                    self.call_from_thread(self.stream_piece, str(piece))
                text = "".join(pieces).strip() or REFUSAL
                if not self.chat_id:
                    self.chat_id = store.create_chat(query[:60])
                store.add_message(self.chat_id, "user", question)
                store.add_message(
                    self.chat_id,
                    "assistant",
                    text,
                    [hit_to_dict(hit) for hit in hits],
                )
        except Exception as exc:  # noqa: BLE001 - a failed answer must show up
            self.call_from_thread(self.write_line, f"error: {exc}", "error")
            self.call_from_thread(self.busy_done, "")
            return
        self.call_from_thread(self.finish, hits, len(text))

    def stream_piece(self, piece: str) -> None:
        """One growing Static per answer, so deltas do not become 400 lines."""
        self._answer_text += piece
        if self._answer_widget is None:
            self._answer_widget = Static("", classes="answer", markup=False)
            self.query_one("#transcript", VerticalScroll).mount(self._answer_widget)
        self._answer_widget.update(self._answer_text)

    def finish(self, hits: list, length: int) -> None:
        for rank, hit in enumerate(hits, start=1):
            where = f"{hit.path}:{hit.line}" if hit.line > 1 else hit.path
            self.write_line(f"  [{rank}] {where}", "cite")
        self.busy_done(f"{len(hits)} source(s) · {length} characters")

    def busy_done(self, status: str) -> None:
        self.busy = False
        self.set_status(status)
        self.query_one("#prompt", Input).focus()


def run_screen(
    *,
    db: str,
    embedder: Embedder,
    llm: Any,
    reranker: Any = None,
    top_k: int = 6,
    chat_id: int = 0,
) -> int:
    ChatScreen(
        db=db,
        embedder=embedder,
        llm=llm,
        reranker=reranker,
        top_k=top_k,
        chat_id=chat_id,
    ).run()
    return 0
