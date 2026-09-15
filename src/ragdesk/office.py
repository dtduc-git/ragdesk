"""Text extraction for office documents and PDFs, stdlib + pypdf only.

``.docx`` / ``.pptx`` / ``.xlsx`` are zip+XML, so they need no dependency at
all. PDFs go through pypdf (BSD, pure Python); scanned PDFs without a text
layer come back empty and are skipped upstream — no OCR in scope.
"""

from __future__ import annotations

import html
import logging
import re
import zipfile
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree

DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".xlsm"}
MAX_DOCUMENT_CHARS = 200_000  # a whole book would otherwise stall the indexer

# pypdf warns per font when fontTools is absent, yet its fallback decoding is
# correct for the PDFs we have seen (Vietnamese included). Keep logs quiet.
logging.getLogger("pypdf").setLevel(logging.ERROR)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _safe_parse(xml_bytes: bytes) -> ElementTree.Element | None:
    """Parse Office XML with DTDs refused.

    Real docx/xlsx/pptx parts never carry a DTD, so rejecting one closes the
    entity-expansion hole that comes with the stdlib parser.
    """
    head = xml_bytes[:4096].lower()
    if b"<!doctype" in head or b"<!entity" in head:
        return None
    try:
        return ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError:
        return None


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    """The workbook's shared string table; rich-text runs are concatenated."""
    try:
        xml = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = _safe_parse(xml)
    if root is None:
        return []
    strings: list[str] = []
    for item in root:
        if _local(item.tag) != "si":
            continue
        parts = [
            (node.text or "")
            for node in item.iter()
            if _local(node.tag) == "t"
        ]
        strings.append("".join(parts))
    return strings


def _sheet_titles(archive: zipfile.ZipFile) -> list[str]:
    try:
        xml = archive.read("xl/workbook.xml")
    except KeyError:
        return []
    root = _safe_parse(xml)
    if root is None:
        return []
    return [
        str(node.get("name") or "")
        for node in root.iter()
        if _local(node.tag) == "sheet"
    ]


DATE_NUMFMT_IDS = set(range(14, 23)) | {27, 30, 36, 45, 46, 47, 50, 57}
_DATE_TOKENS = re.compile(r"[yYdDhHsS]|[mM]{2,}")


def _date_styles(archive: zipfile.ZipFile) -> set[int]:
    """Style indices whose number format is a date (builtin id or custom code)."""
    try:
        styles_xml = archive.read("xl/styles.xml")
    except KeyError:
        return set()
    root = _safe_parse(styles_xml)
    if root is None:
        return set()
    custom: dict[int, str] = {}
    for node in root.iter():
        if _local(node.tag) == "numFmt":
            try:
                custom[int(node.get("numFmtId", "0"))] = str(node.get("formatCode", ""))
            except (TypeError, ValueError):
                continue
    styles: set[int] = set()
    index = 0
    for node in root.iter():
        if _local(node.tag) != "xf":
            continue
        try:
            numfmt = int(node.get("numFmtId", "0"))
        except (TypeError, ValueError):
            numfmt = 0
        code = custom.get(numfmt, "")
        is_date = numfmt in DATE_NUMFMT_IDS or bool(code and _DATE_TOKENS.search(code))
        if is_date:
            styles.add(index)
        index += 1
    return styles


def _serial_to_date(value: str, with_time: bool) -> str:
    from datetime import datetime, timedelta  # noqa: PLC0415 - only needed here

    try:
        serial = float(value)
    except ValueError:
        return value
    # Excel serials count from 1899-12-30 (the 1900 leap-year quirk included).
    stamp = datetime(1899, 12, 30) + timedelta(days=serial)
    return stamp.strftime("%Y-%m-%d %H:%M" if with_time else "%Y-%m-%d")


def _cell_text(
    cell: ElementTree.Element,
    shared: list[str],
    date_styles: set[int] | None = None,
) -> str:
    kind = str(cell.get("t") or "")
    value = ""
    for node in cell.iter():
        if _local(node.tag) == "v":
            value = node.text or ""
        elif _local(node.tag) == "t" and kind == "inlineStr":
            value += node.text or ""
    if kind == "s":
        try:
            return shared[int(value)]
        except (ValueError, IndexError):
            return ""
    if date_styles and str(cell.get("s") or "") in {str(index) for index in date_styles}:
        return _serial_to_date(value, with_time=":" in value)
    return value.strip()


def _sheet_rows(
    xml_bytes: bytes, shared: list[str], date_styles: set[int] | None = None
) -> list[str]:
    root = _safe_parse(xml_bytes)
    if root is None:
        return []
    rows: list[str] = []
    for row in root.iter():
        if _local(row.tag) != "row":
            continue
        values = [
            _cell_text(cell, shared, date_styles)
            for cell in row
            if _local(cell.tag) == "c"
        ]
        values = [value for value in values if value]
        if not values:
            continue
        ref = str(row.get("r") or "")
        prefix = f"r{ref}: " if ref else ""
        rows.append(prefix + " | ".join(values))
    return rows


def extract_xlsx_text(data: bytes) -> str | None:
    """Text from a workbook: every sheet with its rows (shared + inline strings).

    Cell values keep their row number so a citation can point at ``r12``.
    Date serials stay numeric — formatting them needs the styles table, which
    is not worth the code until someone actually misses it.
    """
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            names = archive.namelist()
            sheets = sorted(
                name
                for name in names
                if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
            )
            if not sheets:
                return None
            shared = _shared_strings(archive)
            titles = _sheet_titles(archive)
            date_styles = _date_styles(archive)
            blocks: list[str] = []
            for index, sheet in enumerate(sheets):
                rows = _sheet_rows(archive.read(sheet), shared, date_styles)
                if not rows:
                    continue
                title = titles[index] if index < len(titles) else Path(sheet).stem
                # The first row is the header: repeat it as column context so a
                # chunk holding only later rows still knows what the columns are.
                header = ""
                if rows and "|" in rows[0]:
                    columns = rows[0].split(": ", 1)[-1]
                    header = f"columns: {columns}\n"
                blocks.append(f"[sheet] {title}\n{header}" + "\n".join(rows))
    except (KeyError, zipfile.BadZipFile, OSError):
        return None
    text = "\n\n".join(blocks).strip()
    return text or None


def extract_office_text(data: bytes, suffix: str) -> str | None:
    """Extract text from .docx / .pptx / .xlsx, all zip+XML, no dependencies."""
    if suffix in (".xlsx", ".xlsm"):
        return extract_xlsx_text(data)
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
    if text:
        return text
    # No text layer: a scan. Hand it to the on-device OCR when available.
    from ragdesk.vision import ocr_pdf  # noqa: PLC0415 - optional extra

    return ocr_pdf(data)


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
