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
