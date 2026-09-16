"""AWS S3 connector: index a bucket prefix through the ``aws`` CLI (read-only).

No SDK, no credentials of our own: the AWS CLI already lives on most machines
that talk to S3, and it already knows the profiles (``~/.aws``, SSO, env vars).
ragdesk runs ``aws s3 ls`` and ``aws s3 cp -`` and never stores a key — the
same posture as the GitHub connector reusing ``gh``.

Objects go through the shared admission rules (``is_indexable``) and the shared
extractor (``extract_bytes``), so PDFs, Office files, images (OCR) and plain
text all work exactly as they do for local files.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from ragdesk.embed import Embedder
from ragdesk.index import IndexStats, extract_bytes, index_document, is_indexable
from ragdesk.store import Store

DEFAULT_LIMIT = 500
LIST_TIMEOUT = 120.0
FETCH_TIMEOUT = 120.0


class S3Error(RuntimeError):
    """The aws CLI is missing, unauthenticated, or the bucket refused us."""


def aws_available() -> bool:
    return shutil.which("aws") is not None


def _run_aws(args: list[str], *, profile: str = "", timeout: float = LIST_TIMEOUT) -> bytes:
    if not aws_available():
        raise S3Error(
            "the aws CLI is not on PATH — install it (brew install awscli) and "
            "log in with `aws sso login` or `aws configure`"
        )
    command = ["aws", *args]
    if profile:
        command += ["--profile", profile]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise S3Error(f"cannot run aws: {exc}") from exc
    if completed.returncode != 0:
        message = completed.stderr.decode(errors="replace").strip().splitlines()
        detail = message[-1] if message else f"exit {completed.returncode}"
        raise S3Error(f"aws refused the request: {detail[:200]}")
    return completed.stdout


def parse_listing(output: str) -> list[tuple[str, int]]:
    """`aws s3 ls --recursive` lines → (key, size)."""
    entries: list[tuple[str, int]] = []
    for line in output.splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) < 4:
            continue
        try:
            size = int(parts[2])
        except ValueError:
            continue
        entries.append((parts[3].strip(), size))
    return entries


def sync_s3(
    store: Store,
    embedder: Embedder,
    *,
    bucket: str,
    prefix: str = "",
    profile: str = "",
    limit: int = DEFAULT_LIMIT,
    progress: Callable[[str, int, int], None] | None = None,
) -> IndexStats:
    """Index the objects under ``s3://bucket/prefix`` (newest listing first)."""
    bucket = bucket.strip()
    if not bucket:
        raise S3Error("bucket is required")
    store.ensure_embedder(embedder.name, embedder.dim)

    key_prefix = prefix.strip().lstrip("/")
    raw = _run_aws(
        ["s3", "ls", f"s3://{bucket}/{key_prefix}", "--recursive"], profile=profile
    )
    listing = parse_listing(raw.decode(errors="replace"))
    stats = IndexStats()
    fetched = 0
    for key, size in listing:
        stats.files_scanned += 1
        if not is_indexable(Path(key), size):
            stats.skip(Path(key), "unsupported or too large")
            continue
        if fetched >= limit:
            break
        fetched += 1
        if progress is not None:
            progress(f"fetching {Path(key).name}", stats.files_scanned, 0)
        try:
            data = _run_aws(
                ["s3", "cp", f"s3://{bucket}/{key}", "-"],
                profile=profile,
                timeout=FETCH_TIMEOUT,
            )
        except S3Error as exc:
            stats.skip(Path(key), str(exc)[:120])
            continue
        try:
            content = extract_bytes(data, Path(key).name)
        except Exception as exc:  # noqa: BLE001 - one bad object must not stop the sync
            stats.skip(Path(key), f"extract failed: {type(exc).__name__}")
            continue
        if content is None or not content.strip():
            stats.skip(Path(key), "no extractable text")
            continue
        chunks = index_document(
            store,
            embedder,
            source=f"s3:{bucket}",
            path=f"s3://{bucket}/{key}",
            content=content,
            metadata={"bucket": bucket, **({"prefix": key_prefix} if key_prefix else {})},
        )
        if chunks:
            stats.indexed += 1
            stats.chunks += chunks
        else:
            stats.unchanged += 1
    return stats
