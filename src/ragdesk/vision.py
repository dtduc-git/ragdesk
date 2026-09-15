"""Image text extraction (OCR) via Apple's on-device Vision framework.

No model download, nothing leaves the machine; the ``vision`` extra pulls the
pyobjc bindings. On other platforms images are skipped like before.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".heic",
    ".heif",
    ".tiff",
    ".bmp",
    ".gif",
}
MAX_IMAGE_CHARS = 20_000
# Vision's identifiers: Vietnamese is "vi-VT" on macOS, not "vi-VN".
LANGUAGES = ("en-US", "vi-VT")


def ocr_available() -> bool:
    return (
        importlib.util.find_spec("Vision") is not None
        and importlib.util.find_spec("Quartz") is not None
    )


def _ocr_bytes(data: bytes, languages: tuple[str, ...] = LANGUAGES, scale: float = 2.0) -> str:
    import objc  # noqa: PLC0415 - optional extra
    import Quartz  # noqa: PLC0415
    import Vision  # noqa: PLC0415
    from Foundation import NSData  # noqa: PLC0415

    lines: list[str] = []

    def handler(request, error):  # noqa: ANN001 - ObjC callback signature
        if error:
            return
        for observation in request.results() or []:
            candidates = observation.topCandidates_(1)
            if candidates:
                lines.append(str(candidates[0].string()))

    request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(handler)
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)
    try:
        supported = request.supportedRecognitionLanguagesAndReturnError_(None)
        supported = supported[0] if isinstance(supported, tuple) else supported
        wanted = [code for code in languages if code in supported]
        if wanted:
            request.setRecognitionLanguages_(wanted)
    except Exception:  # noqa: BLE001 - older macOS: keep the default language
        pass

    payload = NSData.dataWithBytes_length_(data, len(data))
    image = Quartz.CIImage.imageWithData_(payload)
    if image is None:
        return ""
    if scale > 1.0:
        # Small screenshots OCR better upscaled.
        image = image.imageByApplyingTransform_(
            Quartz.CGAffineTransformMakeScale(scale, scale)
        )
    with objc.autorelease_pool():
        vision_handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(
            image, None
        )
        vision_handler.performRequests_error_([request], None)
    return "\n".join(lines)


def _created(path: Path) -> str:
    """Best-effort capture date from Spotlight metadata (macOS built-in)."""
    try:
        out = subprocess.run(
            ["mdls", "-name", "kMDItemContentCreationDate", "-raw", str(path)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""
    return out if out and out != "(null)" else ""


def extract_image_bytes(data: bytes, name: str) -> str | None:
    """OCR for connector payloads (OneDrive, repo tarballs) that hold bytes."""
    if not ocr_available():
        return None
    try:
        text = _ocr_bytes(data)
    except Exception:  # noqa: BLE001 - a broken image must never kill indexing
        return None
    return _wrap(text, name, "")


def extract_image(path: Path) -> str | None:
    """OCR text plus a small header (file name, capture date) for context."""
    if not ocr_available():
        return None
    try:
        text = _ocr_bytes(path.read_bytes())
    except Exception:  # noqa: BLE001 - a broken image must never kill indexing
        return None
    return _wrap(text, path.name, _created(path))


def _wrap(text: str, name: str, created: str) -> str | None:
    body = text.strip()
    if not body:
        return None
    header = f"[image] {name}"
    if created:
        header += f" (captured {created})"
    return f"{header}\n{body[:MAX_IMAGE_CHARS]}"
