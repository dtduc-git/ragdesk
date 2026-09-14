"""Confluence connector: index pages from a space (read-only).

Auth: email + API token (Basic) for Confluence Cloud. ``api_path`` lets
Server/DC deployments override the REST path. Sync is full-fetch: content
hashes make re-indexing a no-op for unchanged pages.
"""

from __future__ import annotations

import base64
import html
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

from ragdesk import credentials
from ragdesk.embed import Embedder
from ragdesk.index import IndexStats, index_document
from ragdesk.store import Store

DEFAULT_API_PATH = "/wiki/rest/api/content/search"
SPACE_RE = re.compile(r"^[A-Za-z0-9_-]+$")

_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_BLOCK_RE = re.compile(
    r"</(?:p|div|li|tr|h[1-6]|table|ul|ol|blockquote|pre|code)>", re.IGNORECASE
)
_TAG_RE = re.compile(r"<[^>]+>")


class ConfluenceError(RuntimeError):
    """Confluence API / auth failure."""


def html_to_text(storage_html: str) -> str:
    """Convert Confluence storage-format HTML to readable plain text."""
    text = _BR_RE.sub("\n", storage_html)
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


def _get_json(url: str, auth: str, *, timeout: float = 60.0) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Basic {auth}",
            "Accept": "application/json",
            "User-Agent": "ragdesk",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise ConfluenceError(
                "Confluence rejected the credentials (401/403). Check email + API token."
            ) from exc
        raise ConfluenceError(f"Confluence API error {exc.code}") from exc


def _resolve_credentials(email: str | None, token: str | None) -> tuple[str, str]:
    stored = credentials.get("confluence")
    email = email or os.environ.get("CONFLUENCE_EMAIL") or stored.get("email")
    token = token or os.environ.get("CONFLUENCE_TOKEN") or stored.get("token")
    if not email or not token:
        raise ConfluenceError(
            "Confluence credentials missing: connect the site (base URL + email + API "
            "token) or set CONFLUENCE_EMAIL / CONFLUENCE_TOKEN"
        )
    return email, token


def whoami(
    base_url: str,
    email: str,
    token: str,
    *,
    api_path: str = "/wiki/rest/api/user/current",
    timeout: float = 30.0,
) -> str:
    """Validate site credentials and return a display name."""
    auth = base64.b64encode(f"{email}:{token}".encode()).decode()
    base = base_url.rstrip("/")
    payload = _get_json(f"{base}{api_path}", auth, timeout=timeout)
    return str(
        payload.get("displayName")
        or payload.get("publicName")
        or payload.get("email")
        or email
    )


def sync_confluence(
    store: Store,
    embedder: Embedder,
    *,
    base_url: str,
    space: str,
    email: str | None = None,
    token: str | None = None,
    api_path: str = DEFAULT_API_PATH,
    limit: int = 100,
    timeout: float = 60.0,
) -> IndexStats:
    """Fetch and index every page in a space (Cursorless: v1 ``_links.next``)."""
    if not SPACE_RE.match(space):
        raise ConfluenceError(f"invalid space key: {space!r}")
    email, token = _resolve_credentials(email, token)
    auth = base64.b64encode(f"{email}:{token}".encode()).decode()
    store.ensure_embedder(embedder.name, embedder.dim)

    base = base_url.rstrip("/")
    cql = urllib.parse.quote(f'space="{space}" and type=page')
    url = f"{base}{api_path}?cql={cql}&limit={limit}&expand=body.storage,version"

    stats = IndexStats()
    host = urllib.parse.urlparse(base).netloc or base
    while url:
        payload = _get_json(url, auth, timeout=timeout)
        for page in payload.get("results", []):
            stats.files_scanned += 1
            title = str(page.get("title", "untitled"))
            body = html_to_text(((page.get("body") or {}).get("storage") or {}).get("value", ""))
            content = f"# {title}\n\n{body}".strip()
            if not body.strip():
                stats.skipped += 1
                continue
            page_id = str(page.get("id", ""))
            chunks = index_document(
                store,
                embedder,
                source=f"confluence:{host}/{space}",
                path=f"confluence://{host}/{space}/{page_id}",
                content=content,
            )
            if chunks:
                stats.indexed += 1
                stats.chunks += chunks
            else:
                stats.unchanged += 1

        next_link = (payload.get("_links") or {}).get("next")
        url = urllib.parse.urljoin(base + "/", next_link) if next_link else ""
    return stats
