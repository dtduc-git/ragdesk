"""Email connector: index mbox files and IMAP folders (read-only).

One document per message — headers (subject, from, to, date) plus the text
body — so "the email from the bank about the loan" retrieves by either field.
IMAP is opened with ``EXAMINE``/``readonly=True`` and fetched with
``BODY.PEEK``: the mailbox is never modified and nothing is marked as read.

Attachments are not indexed (their names are listed in the message text);
point ragdesk at the folder where you keep those files instead.
"""

from __future__ import annotations

import contextlib
import email
import email.policy
import hashlib
import imaplib
import mailbox
from collections.abc import Callable, Iterator
from email.message import Message
from email.utils import parsedate_to_datetime
from pathlib import Path

from ragdesk import credentials
from ragdesk.embed import Embedder
from ragdesk.htmlutil import html_to_text
from ragdesk.index import IndexStats, index_document
from ragdesk.store import Store

DEFAULT_IMAP_PORT = 993
DEFAULT_FOLDER = "INBOX"
DEFAULT_LIMIT = 200
MAX_MESSAGE_CHARS = 60_000


class EmailError(RuntimeError):
    """mbox parsing or IMAP failure."""


def resolve_imap_credentials(explicit: dict | None = None) -> dict:
    """Explicit values win over the saved connection."""
    stored = credentials.get("email")
    return {**stored, **(explicit or {})}


def _decode_part(part: Message) -> str:
    raw = part.get_payload(decode=True)
    if raw is None:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        text = raw.decode(charset, errors="replace")
    except LookupError:
        text = raw.decode("utf-8", errors="replace")
    if part.get_content_type() == "text/html":
        return html_to_text(text)
    return text


def message_text(message: Message) -> str:
    """Flatten one message into indexable text: headers first, then the body."""
    header_lines = []
    for label, field in (
        ("Subject", "Subject"),
        ("From", "From"),
        ("To", "To"),
        ("Date", "Date"),
    ):
        value = " ".join(str(message.get(field) or "").split())
        if value:
            header_lines.append(f"{label}: {value}")

    bodies: list[str] = []
    parts: Iterator[Message] = (
        (part for part in message.walk() if not part.is_multipart())
        if message.is_multipart()
        else iter([message])
    )
    for part in parts:
        filename = part.get_filename()
        if filename:
            bodies.append(f"[attachment: {filename}]")
            continue
        if part.get_content_maintype() != "text":
            continue
        text = _decode_part(part).strip()
        if text:
            bodies.append(text)
    return "\n".join([*header_lines, "", *bodies]).strip()[:MAX_MESSAGE_CHARS]


def _message_key(message: Message) -> str:
    message_id = str(message.get("Message-ID") or "").strip().strip("<>")
    if message_id:
        return message_id[:120]
    return hashlib.sha1(message.as_bytes()).hexdigest()[:16]


def _message_date(message: Message) -> str:
    try:
        parsed = parsedate_to_datetime(str(message.get("Date") or ""))
    except (TypeError, ValueError):
        return ""
    return parsed.strftime("%Y-%m-%d") if parsed else ""


def _metadata(message: Message, extra: dict[str, str] | None = None) -> dict[str, str]:
    sender = " ".join(str(message.get("From") or "").split())[:120]
    return {
        **(extra or {}),
        "kind": "email",
        "date": _message_date(message),
        **({"from": sender} if sender else {}),
    }


def _index_message(
    store: Store,
    embedder: Embedder,
    *,
    source: str,
    base_path: str,
    message: Message,
) -> int:
    text = message_text(message)
    if not text:
        return -1
    return index_document(
        store,
        embedder,
        source=source,
        path=f"{base_path}::{_message_key(message)}",
        content=text,
        metadata=_metadata(message),
    )


def index_mbox(
    store: Store,
    embedder: Embedder,
    path: str | Path,
    *,
    progress: Callable[[str, int, int], None] | None = None,
    limit: int = 0,
) -> IndexStats:
    """Index every message of an mbox file (``limit`` 0 = all)."""
    box_path = Path(path).expanduser()
    if not box_path.is_file():
        raise EmailError(f"mbox not found: {box_path}")
    store.ensure_embedder(embedder.name, embedder.dim)

    stats = IndexStats()
    try:
        box = mailbox.mbox(str(box_path), create=False)
    except (OSError, mailbox.Error) as exc:
        raise EmailError(f"cannot open mbox: {exc}") from exc

    source = f"mbox:{box_path.stem}"
    base_path = f"mbox://{box_path}"
    try:
        for position, message in enumerate(box):
            if limit and position >= limit:
                break
            stats.files_scanned += 1
            if progress is not None:
                progress(f"message {position + 1}", stats.files_scanned, 0)
            chunks = _index_message(
                store, embedder, source=source, base_path=base_path, message=message
            )
            if chunks < 0:
                stats.skipped += 1
            elif chunks:
                stats.indexed += 1
                stats.chunks += chunks
            else:
                stats.unchanged += 1
    except (OSError, mailbox.Error) as exc:
        raise EmailError(f"cannot read mbox: {exc}") from exc
    finally:
        box.close()
    return stats


@contextlib.contextmanager
def _imap_session(
    host: str, user: str, password: str, port: int, timeout: float
) -> Iterator[imaplib.IMAP4_SSL]:
    try:
        connection = imaplib.IMAP4_SSL(host, port, timeout=timeout)
    except (OSError, imaplib.IMAP4.error) as exc:
        raise EmailError(f"cannot reach {host}:{port}: {exc}") from exc
    try:
        try:
            status, _ = connection.login(user, password)
        except imaplib.IMAP4.error as exc:
            raise EmailError(f"IMAP login failed for {user!r}: {exc}") from exc
        if status != "OK":
            raise EmailError(f"IMAP login rejected for {user!r}")
        yield connection
    finally:
        with contextlib.suppress(imaplib.IMAP4.error, OSError):
            connection.logout()


def whoami(*, host: str, user: str, password: str, port: int = DEFAULT_IMAP_PORT) -> str:
    """Log in and open the inbox read-only; the account is the username."""
    with _imap_session(host, user, password, port, 30.0) as connection:
        status, data = connection.select(DEFAULT_FOLDER, readonly=True)
        if status != "OK":
            raise EmailError(f"cannot open {DEFAULT_FOLDER}: {_clean(data)}")
    return user


def _clean(data: object) -> str:
    return " ".join(str(data).split())[:200]


def _exists_count(data: list) -> int:
    raw = data[0] if data else None
    if isinstance(raw, bytes):
        raw = raw.decode(errors="ignore")
    try:
        return int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def sync_imap(
    store: Store,
    embedder: Embedder,
    *,
    host: str,
    user: str,
    password: str,
    port: int = DEFAULT_IMAP_PORT,
    folder: str = DEFAULT_FOLDER,
    limit: int = DEFAULT_LIMIT,
    progress: Callable[[str, int, int], None] | None = None,
) -> IndexStats:
    """Index the newest ``limit`` messages of an IMAP folder, read-only."""
    if not host or not user or not password:
        raise EmailError("host, user and password are required")
    store.ensure_embedder(embedder.name, embedder.dim)

    stats = IndexStats()
    with _imap_session(host, user, password, port, 60.0) as connection:
        # readonly: EXAMINE, so nothing gets marked \Seen by opening the folder
        status, data = connection.select(folder, readonly=True)
        if status != "OK":
            raise EmailError(f"cannot open folder {folder!r}: {_clean(data)}")
        exists = _exists_count(list(data or []))
        if exists == 0:
            return stats
        start = max(1, exists - max(1, limit) + 1)
        folder_name = str(folder)
        status, rows = connection.fetch(f"{start}:{exists}", "(BODY.PEEK[])")
        if status != "OK":
            raise EmailError(f"IMAP fetch failed: {_clean(rows)}")

        source = f"imap:{user}@{host}"
        base_path = f"imap://{user}@{host}/{folder_name}"
        for item in rows or []:
            if not isinstance(item, tuple) or len(item) < 2:
                continue
            raw = item[1]
            if not isinstance(raw, bytes):
                continue
            stats.files_scanned += 1
            if progress is not None:
                progress(f"message {stats.files_scanned}", stats.files_scanned, 0)
            message = email.message_from_bytes(raw, policy=email.policy.default)
            chunks = _index_message(
                store, embedder, source=source, base_path=base_path, message=message
            )
            if chunks < 0:
                stats.skipped += 1
            elif chunks:
                stats.indexed += 1
                stats.chunks += chunks
            else:
                stats.unchanged += 1
    return stats
