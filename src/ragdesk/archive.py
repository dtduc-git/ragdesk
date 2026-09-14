"""Shared tarball helpers for repository connectors (GitHub, GitLab)."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

from ragdesk.index import is_indexable


def tar_text_files(data: bytes, subdir: str = "") -> list[tuple[str, str]]:
    """Return ``(relative_path, text)`` for indexable files in a repo tarball.

    Repository archives wrap everything in one top-level ``<project>-<sha>/``
    directory, which is stripped here.
    """
    files: list[tuple[str, str]] = []
    prefix = Path(subdir) if subdir else None
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            rel = Path(*Path(member.name).parts[1:])  # strip '<owner>-<repo>-<sha>/'
            if not rel.parts:
                continue
            if prefix is not None and rel != prefix and prefix not in rel.parents:
                continue
            if not is_indexable(rel, member.size):
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            raw = extracted.read()
            if b"\x00" in raw[:1024]:
                continue
            files.append((str(rel), raw.decode("utf-8", errors="replace")))
    return files
