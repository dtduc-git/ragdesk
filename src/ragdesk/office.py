"""Text extraction for office documents and PDFs, stdlib + pypdf only.

``.docx`` / ``.pptx`` are zip+XML, so they need no dependency at all. PDFs go
through pypdf (BSD, pure Python); scanned PDFs without a text layer come back
empty and are skipped upstream — no OCR in scope.
"""

from __future__ import annotations

import html
import logging
import re
import zipfile
from io import BytesIO
from pathlib import Path

DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".pptx"}
MAX_DOCUMENT_CHARS = 200_000  # a whole book would otherwise stall the indexer

# pypdf warns per font when fontTools is absent, yet its fallback decoding is
# correct for the PDFs we have seen (Vietnamese included). Keep logs quiet.
logging.getLogger("pypdf").setLevel(logging.ERROR)


def extract_office_text(data: bytes, suffix: str) -> str | None:
    """Extract text from .docx / .pptx (both are zip+XML) without dependencies."""
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            if suffix == ".docx":
                xml = archive.read("word/document.xml").decode("utf-8", "replace")
            elif suffix == ".pptx":
                names = sorted(
                    name
                    for name in archive.namelist()
                    if name.startswith("ppt/slides/slide") and name.endswith(".xml")
                )
                xml = "\n".join(
                    archive.read(name).decode("utf-8", "replace") for name in names
                )
            else:
                return None
    except (KeyError, zipfile.BadZipFile, OSError):
        return None
    text = re.sub(r"</(?:w:p|a:p)>", "\n", xml)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip() or None


def extract_pdf_text(data: bytes) -> str | None:
    """Text layer of a PDF; None when there is none (scanned) or it is broken."""
    try:
        from pypdf import PdfReader  # noqa: PLC0415 - optional heavy-ish import
    except ImportError:  # pragma: no cover - pypdf ships with the package now
        return None
    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:  # noqa: BLE001 - any failure means "locked"
                return None
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception:  # noqa: BLE001 - malformed PDFs must never kill indexing
        return None
    text = "\n\n".join(page.strip() for page in pages if page.strip())
    return text or None


def extract_document(path: Path) -> str | None:
    """Read a .pdf / .docx / .pptx from disk and return its text, or None."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = extract_pdf_text(data)
    else:
        text = extract_office_text(data, suffix)
    if not text:
        return None
    return text[:MAX_DOCUMENT_CHARS]
