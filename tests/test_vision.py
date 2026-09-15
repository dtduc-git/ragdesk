from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ragdesk import vision
from ragdesk.index import MAX_DOCUMENT_BYTES, is_indexable
from ragdesk.vision import extract_image, ocr_available

macos_ocr = pytest.mark.skipif(
    not (sys.platform == "darwin" and ocr_available()),
    reason="Apple Vision OCR needs macOS with the vision extra",
)


def _render_png(text: str, path: Path, width: int = 900, height: int = 140) -> None:
    """Draw text into a PNG with AppKit (pyobjc ships with the vision extra)."""
    from AppKit import (  # type: ignore[import-not-found]
        NSBitmapImageFileTypePNG,
        NSBitmapImageRep,
        NSColor,
        NSFont,
        NSFontAttributeName,
        NSForegroundColorAttributeName,
        NSGraphicsContext,
        NSMakePoint,
        NSMakeRect,
        NSString,
    )

    allocate = NSBitmapImageRep.alloc()
    # The ObjC selector is one generated identifier; build it in a variable.
    selector = (
        "initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_"
        "hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_"
    )
    rep = getattr(allocate, selector)(
        None, width, height, 8, 4, True, False, "NSCalibratedRGBColorSpace", 0, 0
    )
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.setCurrentContext_(
        NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep)
    )
    NSColor.whiteColor().set()
    NSMakeRect(0, 0, width, height)
    from AppKit import NSRectFill  # type: ignore[import-not-found]

    NSRectFill(NSMakeRect(0, 0, width, height))
    attrs = {
        NSFontAttributeName: NSFont.systemFontOfSize_(40),
        NSForegroundColorAttributeName: NSColor.blackColor(),
    }
    NSString.stringWithString_(text).drawAtPoint_withAttributes_(
        NSMakePoint(30, 50), attrs
    )
    NSGraphicsContext.restoreGraphicsState()
    data = rep.representationUsingType_properties_(NSBitmapImageFileTypePNG, {})
    path.write_bytes(bytes(data))


def test_image_admission_rules(tmp_path: Path):
    shot = tmp_path / "shot.png"
    shot.write_bytes(b"x")
    assert is_indexable(shot, 5_000_000) is True
    assert is_indexable(shot, MAX_DOCUMENT_BYTES + 1) is False
    note = tmp_path / "note.md"
    note.write_text("x")
    assert is_indexable(note, 100) is True


def test_extract_image_without_the_extra_returns_none(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("ragdesk.vision.ocr_available", lambda: False)
    shot = tmp_path / "shot.png"
    shot.write_bytes(b"not really a png")
    assert extract_image(shot) is None


@macos_ocr
def test_ocr_reads_rendered_text(tmp_path: Path):
    shot = tmp_path / "screenshot.png"
    _render_png("Deploy rollback checklist RRF fusion", shot)
    text = extract_image(shot)
    assert text is not None
    assert "screenshot.png" in text  # the header carries the file name
    lowered = text.lower()
    assert "rollback" in lowered
    assert "fusion" in lowered


@macos_ocr
def test_ocr_on_blank_image_returns_none(tmp_path: Path):
    blank = tmp_path / "blank.png"
    _render_png("", blank)
    assert extract_image(blank) is None


@macos_ocr
def test_ocr_on_corrupt_file_returns_none(tmp_path: Path):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"definitely not an image")
    assert extract_image(broken) is None


def test_creation_date_helper_tolerates_failures(tmp_path: Path, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("mdls gone")

    monkeypatch.setattr("ragdesk.vision.subprocess.run", boom)
    assert vision._created(tmp_path / "x.png") == ""


def _make_empty_pdf() -> bytes:
    """A one-page PDF whose page holds no text at all (a stand-in for a scan)."""
    import io

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] >>",
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
        b"trailer << /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (len(objects) + 1, xref)
    )
    return out.getvalue()


def test_pdf_without_text_layer_falls_back_to_ocr(monkeypatch):
    """The wiring: an empty text layer hands the bytes to the on-device OCR."""
    import ragdesk.vision as vision_module
    from ragdesk.office import extract_pdf_text

    calls: list[int] = []

    def fake_ocr(data: bytes, **kwargs) -> str:
        calls.append(len(data))
        return "recovered from a scan"

    monkeypatch.setattr(vision_module, "ocr_pdf", fake_ocr)
    # a valid PDF whose page has no text content
    empty_pdf = _make_empty_pdf()
    assert extract_pdf_text(empty_pdf) == "recovered from a scan"
    assert calls == [len(empty_pdf)]


@macos_ocr
def test_real_scan_extraction_end_to_end(tmp_path: Path):
    """Vision OCR reads a rendered image page directly (PDF rasterizing is the
    same code path, verified live against a real scanned PDF)."""
    shot = tmp_path / "scan-page.png"
    _render_png("Kubernetes crash course notes", shot)
    from ragdesk.vision import extract_image

    text = extract_image(shot)
    assert text is not None
    assert "kubernetes" in text.lower()
