from __future__ import annotations

import email
from pathlib import Path

import pytest

from ragdesk.email_source import (
    EmailError,
    index_mbox,
    message_text,
    sync_imap,
)
from ragdesk.embed import HashingEmbedder
from ragdesk.search import hybrid_search
from ragdesk.store import Store

MESSAGE_ONE = """From bank@example.com Mon Sep 15 10:00:00 2026
Subject: Your loan statement
From: bank@example.com
To: me@example.com
Message-ID: <loan-1@example.com>
Date: Mon, 15 Sep 2026 10:00:00 +0700

Your loan statement is ready. The outstanding balance is 250 million VND.
"""

MESSAGE_TWO = """From friend@example.com Mon Sep 15 11:30:00 2026
Subject: Lunch on Friday
From: friend@example.com
To: me@example.com
Message-ID: <lunch-1@example.com>
Date: Mon, 15 Sep 2026 11:30:00 +0700

Are you free for lunch on Friday? I will book the usual place.
"""


def test_message_text_keeps_headers_body_and_attachment_names():
    raw = (
        "Subject: Invoice\n"
        "From: vendor@example.com\n"
        "To: me@example.com\n"
        "Date: Tue, 16 Sep 2026 09:00:00 +0700\n"
        "MIME-Version: 1.0\n"
        'Content-Type: multipart/mixed; boundary="b"\n'
        "\n"
        "--b\n"
        "Content-Type: text/plain; charset=utf-8\n"
        "\n"
        "The invoice is attached.\n"
        "--b\n"
        'Content-Type: application/pdf; name="Invoice-2026.pdf"\n'
        "Content-Transfer-Encoding: base64\n"
        'Content-Disposition: attachment; filename="Invoice-2026.pdf"\n'
        "\n"
        "JVBERi0xLjQK\n"
        "--b--\n"
    )
    text = message_text(email.message_from_string(raw))
    assert "Subject: Invoice" in text
    assert "From: vendor@example.com" in text
    assert "The invoice is attached." in text
    assert "[attachment: Invoice-2026.pdf]" in text


def make_mbox(path: Path) -> None:
    path.write_text(f"{MESSAGE_ONE}\n{MESSAGE_TWO}")


def test_index_mbox_indexes_each_message(tmp_path: Path):
    box = tmp_path / "inbox.mbox"
    make_mbox(box)
    embedder = HashingEmbedder()
    with Store(tmp_path / "index.db") as store:
        stats = index_mbox(store, embedder, box)
        assert stats.indexed == 2 and stats.chunks == 2
        paths = sorted(row["path"] for row in store.documents())
        assert paths == [
            f"mbox://{box}::loan-1@example.com",
            f"mbox://{box}::lunch-1@example.com",
        ]
        # the body is searchable and the sender/date ride along as metadata
        hits = hybrid_search(store, embedder, "outstanding loan balance", top_k=3)
        assert hits and hits[0].path.endswith("loan-1@example.com")
        assert hits[0].metadata["kind"] == "email"
        assert hits[0].metadata["from"] == "bank@example.com"
        assert hits[0].metadata["date"] == "2026-09-15"

        again = index_mbox(store, embedder, box)
        assert again.indexed == 0 and again.unchanged == 2


def write_mbox_with_attachment(path: Path) -> None:
    import email.message

    message = email.message.EmailMessage()
    message["Subject"] = "Invoice for September"
    message["From"] = "vendor@example.com"
    message["To"] = "me@example.com"
    message["Message-ID"] = "<with-attachment@example.com>"
    message["Date"] = "Mon, 15 Sep 2026 09:00:00 +0700"
    message.set_content("Please find the invoice attached.")
    message.add_attachment(
        b"Invoice total is 250 million VND, payable in 30 days.",
        maintype="text",
        subtype="plain",
        filename="invoice.txt",
    )
    message.add_attachment(
        b"\xff\xfe\x00binary", maintype="application", subtype="octet-stream", filename="blob.bin"
    )
    path.write_bytes(
        b"From vendor@example.com Mon Sep 15 09:00:00 2026\n" + message.as_bytes()
    )


def test_index_mbox_indexes_attachment_contents(tmp_path: Path):
    box = tmp_path / "inbox.mbox"
    write_mbox_with_attachment(box)
    embedder = HashingEmbedder()
    with Store(tmp_path / "index.db") as store:
        stats = index_mbox(store, embedder, box)
        assert stats.indexed == 1 and stats.attachments == 1  # blob.bin is not indexable
        paths = sorted(row["path"] for row in store.documents())
        assert len(paths) == 2
        attachment = next(path for path in paths if path.endswith("00-invoice.txt"))
        assert attachment.startswith(f"mbox://{box}::with-attachment@example.com::")
        row = store.conn.execute(
            "SELECT metadata FROM documents WHERE path = ?", (attachment,)
        ).fetchone()
        assert '"kind": "attachment"' in row["metadata"]
        assert "invoice.txt" in row["metadata"]

        # the attachment body is searchable on its own
        hits = hybrid_search(store, embedder, "payable in 30 days", top_k=3)
        assert hits and hits[0].path.endswith("00-invoice.txt")

        # and its name stays listed in the message text
        message = next(path for path in paths if path.endswith("with-attachment@example.com"))
        assert "[attachment: invoice.txt]" in (store.document_text(message) or "")


def test_index_mbox_missing_file(tmp_path: Path):
    with Store(tmp_path / "index.db") as store:
        with pytest.raises(EmailError):
            index_mbox(store, HashingEmbedder(), tmp_path / "nope.mbox")


class FakeIMAP:
    """Records every call so the test can prove the session stays read-only."""

    calls: list[tuple] = []
    messages: list[bytes] = []

    def __init__(self, host: str, port: int, timeout: float | None = None) -> None:
        FakeIMAP.calls.append(("connect", host, port))

    def login(self, user: str, password: str) -> tuple[str, list[bytes]]:
        FakeIMAP.calls.append(("login", user))
        return ("OK", [b"logged in"])

    def select(self, folder: str, readonly: bool = False) -> tuple[str, list]:
        FakeIMAP.calls.append(("select", folder, readonly))
        return ("OK", [str(len(FakeIMAP.messages)).encode()])

    def fetch(self, spec: str, query: str) -> tuple[str, list]:
        FakeIMAP.calls.append(("fetch", spec, query))
        rows: list = []
        for raw in FakeIMAP.messages:
            rows.append((b"meta", raw))
            rows.append(b")")
        return ("OK", rows)

    def logout(self) -> tuple[str, list[bytes]]:
        FakeIMAP.calls.append(("logout",))
        return ("BYE", [b"bye"])


@pytest.fixture()
def fake_imap(monkeypatch):
    FakeIMAP.calls = []
    FakeIMAP.messages = [
        MESSAGE_ONE.encode(),
        MESSAGE_TWO.encode(),
    ]
    monkeypatch.setattr("ragdesk.email_source.imaplib.IMAP4_SSL", FakeIMAP)
    return FakeIMAP


def test_sync_imap_is_read_only_and_indexes(tmp_path: Path, fake_imap):
    with Store(tmp_path / "index.db") as store:
        stats = sync_imap(
            store,
            HashingEmbedder(),
            host="imap.example.com",
            user="me@example.com",
            password="secret",
            limit=10,
        )
        paths = sorted(row["path"] for row in store.documents())
    assert stats.indexed == 2
    assert ("select", "INBOX", True) in fake_imap.calls
    fetch = next(call for call in fake_imap.calls if call[0] == "fetch")
    assert fetch[1] == "1:2"
    assert "BODY.PEEK" in fetch[2]
    assert paths[0].startswith("imap://me@example.com@imap.example.com/INBOX::")


def test_sync_imap_login_failure_is_loud(tmp_path: Path, monkeypatch):
    class Rejecting(FakeIMAP):
        def login(self, user: str, password: str) -> tuple[str, list[bytes]]:
            raise __import__("imaplib").IMAP4.error("AUTHENTICATIONFAILED")

    monkeypatch.setattr("ragdesk.email_source.imaplib.IMAP4_SSL", Rejecting)
    with Store(tmp_path / "index.db") as store:
        with pytest.raises(EmailError) as excinfo:
            sync_imap(
                store,
                HashingEmbedder(),
                host="imap.example.com",
                user="me@example.com",
                password="wrong",
            )
    assert "login failed" in str(excinfo.value)
    assert "wrong" not in str(excinfo.value)  # never echo the secret
