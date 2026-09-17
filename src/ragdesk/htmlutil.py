"""Shared HTML → text helpers (Confluence storage format, crawled web pages)."""

from __future__ import annotations

import html
import re

_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_BLOCK_RE = re.compile(r"</(?:p|div|li|tr|h[1-6]|table|ul|ol|blockquote|pre|code)>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def html_to_text(source_html: str) -> str:
    """Convert HTML to readable plain text (keeps paragraph breaks)."""
    text = _BR_RE.sub("\n", source_html)
    text = _BLOCK_RE.sub("\n", text)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    lines: list[str] = []
    blank = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line:
            lines.append(line)
            blank = False
        elif not blank:
            lines.append("")
            blank = True
    return "\n".join(lines).strip()
