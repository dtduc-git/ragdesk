"""GitLab connector: index a repository from its archive (read-only).

Auth: a personal access token (``glpat-…``) with the ``read_api`` scope. Works
with gitlab.com or a self-hosted instance via ``base_url``.

Token resolution: explicit -> ``GITLAB_TOKEN`` env -> saved connection.
Whole-repo dedupe: the archive digest is remembered per project/ref.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable

from ragdesk import credentials
from ragdesk.archive import tar_text_files
from ragdesk.embed import Embedder
from ragdesk.index import IndexStats, index_document
from ragdesk.store import Store

DEFAULT_BASE_URL = "https://gitlab.com"
USER_AGENT = "ragdesk-gitlab/0.1"


class GitLabError(RuntimeError):
    """GitLab API / auth failure."""


def resolve_token(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    env = os.environ.get("GITLAB_TOKEN")
    if env:
        return env
    stored = credentials.get("gitlab").get("token")
    return str(stored) if stored else None


def _headers(token: str) -> dict[str, str]:
    return {
        "PRIVATE-TOKEN": token,
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }


def _get_json(url: str, token: str, *, timeout: float = 60.0) -> dict:
    request = urllib.request.Request(url, headers=_headers(token))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise GitLabError(
                "GitLab rejected the token (401/403) — it needs the read_api scope."
            ) from exc
        if exc.code == 404:
            raise GitLabError(
                f"GitLab 404 for {url} — check the project path and the token's access."
            ) from exc
        raise GitLabError(f"GitLab API error {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise GitLabError(f"cannot reach GitLab: {exc}") from exc


def _get_bytes(url: str, token: str, *, timeout: float = 180.0) -> bytes:
    request = urllib.request.Request(url, headers=_headers(token))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise GitLabError(f"GitLab archive download failed (HTTP {exc.code})") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise GitLabError(f"cannot reach GitLab: {exc}") from exc


def whoami(
    token: str, *, base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0
) -> str:
    base = base_url.rstrip("/")
    payload = _get_json(f"{base}/api/v4/user", token, timeout=timeout)
    return str(payload.get("username") or payload.get("name") or "gitlab")


def project_info(
    project: str, token: str, *, base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0
) -> dict:
    base = base_url.rstrip("/")
    slug = urllib.parse.quote(project, safe="")
    payload = _get_json(f"{base}/api/v4/projects/{slug}", token, timeout=timeout)
    return {
        "id": payload.get("id"),
        "default_branch": payload.get("default_branch", ""),
        "path_with_namespace": payload.get("path_with_namespace", project),
    }


def _download_archive(
    project_id: int,
    ref: str,
    token: str,
    base_url: str,
    *,
    timeout: float = 180.0,
) -> bytes:
    base = base_url.rstrip("/")
    url = f"{base}/api/v4/projects/{project_id}/repository/archive.tar.gz"
    if ref:
        url += f"?sha={urllib.parse.quote(ref)}"
    return _get_bytes(url, token, timeout=timeout)


def sync_gitlab(
    store: Store,
    embedder: Embedder,
    *,
    project: str,
    token: str | None = None,
    ref: str = "",
    subdir: str = "",
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 180.0,
    progress: Callable[[str, int, int], None] | None = None,
) -> IndexStats:
    resolved = resolve_token(token)
    if resolved is None:
        raise GitLabError(
            "no GitLab token: connect the project in the app or set GITLAB_TOKEN "
            "(needs the read_api scope)"
        )
    store.ensure_embedder(embedder.name, embedder.dim)

    info = project_info(project, resolved, base_url=base_url)
    project_id = info.get("id")
    if not isinstance(project_id, int):
        raise GitLabError(f"cannot resolve project id for {project!r}")

    data = _download_archive(project_id, ref, resolved, base_url, timeout=timeout)
    digest = hashlib.sha256(data).hexdigest()
    host = urllib.parse.urlparse(base_url).netloc or base_url
    meta_key = f"gitlab.sha:{host}/{project}@{ref or 'default'}"
    if store.get_meta(meta_key) == digest:
        return IndexStats()

    stats = IndexStats()
    for rel, content in tar_text_files(data, subdir):
        stats.files_scanned += 1
        if progress is not None:
            progress(f"indexing {rel}", stats.files_scanned, 0)
        if not content.strip():
            stats.skipped += 1
            continue
        chunks = index_document(
            store,
            embedder,
            source=f"gitlab:{host}/{project}",
            path=f"gitlab://{host}/{project}/{rel}",
            content=content,
        )
        if chunks:
            stats.indexed += 1
            stats.chunks += chunks
        else:
            stats.unchanged += 1
    store.set_meta(meta_key, digest)
    return stats
