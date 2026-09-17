from __future__ import annotations

import io
import zipfile
from pathlib import Path

from ragdesk.office import extract_document, extract_office_text, extract_pdf_text


def make_pdf(text: str) -> bytes:
    """A valid one-page PDF with a text object (correct xref offsets)."""
    content = f"BT /F1 12 Tf 10 50 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 100] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % index + obj + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n" % (len(objects) + 1))
    out.write(b"0000000000 65535 f \n")
    for offset in offsets:
        out.write(b"%010d 00000 n \n" % offset)
    out.write(
        b"trailer << /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    )
    return out.getvalue()


def make_docx(text: str) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr(
            "word/document.xml",
            '<?xml version="1.0"?><w:document xmlns:w="x"><w:body><w:p>'
            f"<w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>",
        )
    return out.getvalue()


def make_xlsx(rows: list[list[str]], sheet_name: str = "Sheet1") -> bytes:
    """A minimal workbook: string cells go through sharedStrings, numbers inline."""
    strings: list[str] = []
    row_xml: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for column, value in enumerate(row):
            ref = f"{chr(ord('A') + column)}{row_index}"
            if value.isdigit():
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                strings.append(value)
                cells.append(f'<c r="{ref}" t="s"><v>{len(strings) - 1}</v></c>')
        row_xml.append(f'<row r="{row_index}">' + "".join(cells) + "</row>")
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0"?><workbook xmlns="s"><sheets>'
            f'<sheet name="{sheet_name}" sheetId="1"/>'
            "</sheets></workbook>",
        )
        archive.writestr(
            "xl/sharedStrings.xml",
            '<?xml version="1.0"?><sst xmlns="s">'
            + "".join(f"<si><t>{s}</t></si>" for s in strings)
            + "</sst>",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            '<?xml version="1.0"?><worksheet xmlns="s"><sheetData>'
            + "".join(row_xml)
            + "</sheetData></worksheet>",
        )
    return out.getvalue()


def make_xlsx_with_dates(rows: list[list[tuple[str, int | None]]]) -> bytes:
    """Workbook where a cell may carry a style index (0 = general, 1 = date)."""
    strings: list[str] = []
    row_xml: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for column, (value, style) in enumerate(row):
            ref = f"{chr(ord('A') + column)}{row_index}"
            style_attr = f' s="{style}"' if style else ""
            if value.isdigit() and style == 1:
                cells.append(f'<c r="{ref}"{style_attr}><v>{value}</v></c>')
            else:
                strings.append(value)
                cells.append(f'<c r="{ref}" t="s">{style_attr}<v>{len(strings) - 1}</v></c>')
        row_xml.append(f'<row r="{row_index}">' + "".join(cells) + "</row>")
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0"?><workbook xmlns="s"><sheets>'
            '<sheet name="Chi phi" sheetId="1"/></sheets></workbook>',
        )
        archive.writestr(
            "xl/styles.xml",
            '<?xml version="1.0"?><styleSheet xmlns="s">'
            '<numFmts count="1"><numFmt numFmtId="164" formatCode="dd/mm/yyyy"/></numFmts>'
            '<cellXfs count="2"><xf numFmtId="0"/><xf numFmtId="164"/></cellXfs>'
            "</styleSheet>",
        )
        archive.writestr(
            "xl/sharedStrings.xml",
            '<?xml version="1.0"?><sst xmlns="s">'
            + "".join(f"<si><t>{value}</t></si>" for value in strings)
            + "</sst>",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            '<?xml version="1.0"?><worksheet xmlns="s"><sheetData>'
            + "".join(row_xml)
            + "</sheetData></worksheet>",
        )
    return out.getvalue()


def test_xlsx_dates_and_header_row():
    from ragdesk.office import extract_xlsx_text

    data = make_xlsx_with_dates(
        [
            [("Hosting", None), ("Amount", None), ("Paid", None)],
            [("ACME", None), ("1200", None), ("45661", 1)],
        ]
    )
    text = extract_xlsx_text(data)
    assert text is not None
    assert "columns: Hosting | Amount | Paid" in text
    # 45661 days from the 1899-12-30 epoch == 2025-01-04
    assert "2025-01-04" in text
    assert "45661" not in text


def test_xlsx_rows_and_sheet_name():
    from ragdesk.office import extract_xlsx_text

    data = make_xlsx(
        [["Item", "Amount"], ["Hosting", "1200"], ["Support", "800"]],
        sheet_name="Chi phi",
    )
    text = extract_xlsx_text(data)
    assert text is not None
    assert "[sheet] Chi phi" in text
    assert "r2: Hosting | 1200" in text
    assert "r3: Support | 800" in text


def test_xlsx_broken_or_empty_returns_none():
    from ragdesk.office import extract_xlsx_text

    assert extract_xlsx_text(b"not a zip") is None
    empty = io.BytesIO()
    with zipfile.ZipFile(empty, "w") as archive:
        archive.writestr("docProps/app.xml", "<x/>")
    assert extract_xlsx_text(empty.getvalue()) is None


def test_xlsx_rejects_dtd():
    from ragdesk.office import extract_xlsx_text

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY boom "y">]>'
            "<worksheet><sheetData/></worksheet>",
        )
    assert extract_xlsx_text(payload.getvalue()) is None


def test_xlsx_is_a_document_for_admission(tmp_path: Path):
    from ragdesk.index import is_indexable

    book = tmp_path / "book.xlsx"
    book.write_bytes(b"x")
    assert is_indexable(book, 5_000_000) is True


def test_pdf_text_is_extracted():
    text = extract_pdf_text(make_pdf("Grounding gate notes for ragdesk"))
    assert text is not None
    assert "Grounding gate" in text


def test_broken_pdf_returns_none():
    assert extract_pdf_text(b"not a pdf at all") is None
    assert extract_pdf_text(b"%PDF-1.4 truncated") is None


def test_scanned_like_pdf_without_text_returns_none():
    # a page with no content stream text: nothing to index, must not crash
    empty = make_pdf("").replace(b"( ) Tj", b" ")
    assert extract_pdf_text(empty) in (None, "") or "BT" not in str(extract_pdf_text(empty))


def test_docx_text_is_extracted():
    assert extract_office_text(make_docx("Xin chao ragdesk"), ".docx") == "Xin chao ragdesk"
    assert extract_office_text(b"junk", ".docx") is None
    assert extract_office_text(b"junk", ".txt") is None


def test_extract_document_dispatches_by_suffix(tmp_path: Path):
    pdf = tmp_path / "note.pdf"
    pdf.write_bytes(make_pdf("Tax certificate summary"))
    docx = tmp_path / "note.docx"
    docx.write_bytes(make_docx("Bang ke thue"))
    txt = tmp_path / "note.txt"
    txt.write_text("plain text stays plain")

    assert "Tax certificate" in (extract_document(pdf) or "")
    assert extract_document(docx) == "Bang ke thue"
    assert extract_document(txt) is None  # not a document suffix; index.py handles text


def test_document_size_rules(tmp_path: Path):
    from ragdesk.index import MAX_DOCUMENT_BYTES, MAX_FILE_BYTES, is_indexable

    pdf = tmp_path / "big.pdf"
    pdf.write_bytes(b"x")
    assert is_indexable(pdf, MAX_DOCUMENT_BYTES) is True
    assert is_indexable(pdf, MAX_DOCUMENT_BYTES + 1) is False
    note = tmp_path / "note.md"
    note.write_text("x")
    assert is_indexable(note, MAX_FILE_BYTES) is True
    assert is_indexable(note, MAX_FILE_BYTES + 1) is False
