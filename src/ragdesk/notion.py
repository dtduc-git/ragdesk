"""Notion connector: index pages the integration has been granted (read-only).

Auth: a Notion internal integration token (``ntn_...`` / ``secret_...``). Create
one at https://www.notion.so/my-integrations and **share pages with it** —
Notion exposes only what you explicitly share.

Incremental: the page ``last_edited_time`` is remembered, so unchanged pages are
not re-fetched block by block.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections import deque

from ragdesk import credentials
from ragdesk.embed import Embedder
from ragdesk.index import IndexStats, index_document
from ragdesk.store import Store

API = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"
USER_AGENT = "ragdesk-notion/0.1"
PAGE_SIZE = 100
MAX_BLOCKS_PER_PAGE = 800


class NotionError(RuntimeError):
    """Notion API / auth failure."""


def resolve_token(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    env = os.environ.get("NOTION_TOKEN")
    if env:
        return env
    stored = credentials.get("notion").get("token")
    return str(stored) if stored else None


def _request(
    method: str,
    path: str,
    token: str,
    *,
    body: dict | None = None,
    timeout: float = 60.0,
) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"{API}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Notion-Version": NOTION_VERSION,
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:200]
        if exc.code in (401, 403):
            raise NotionError(
                "Notion rejected the token (401/403). Check the integration token and "
                "that the pages are shared with the integration."
            ) from exc
        raise NotionError(f"Notion API error {exc.code}: {detail!r}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise NotionError(f"cannot reach Notion: {exc}") from exc


def whoami(token: str, *, timeout: float = 30.0) -> str:
    payload = _request("GET", "/users/me", token, timeout=timeout)
    return str(payload.get("name") or payload.get("id") or "notion")


def _rich_text(items: list[dict] | None) -> str:
    return "".join(str(item.get("plain_text", "")) for item in items or [])


def _page_title(page: dict) -> str:
    for value in (page.get("properties") or {}).values():
        if isinstance(value, dict) and value.get("type") == "title":
            return _rich_text(value.get("title")) or "untitled"
    return "untitled"


def block_text(block: dict) -> str:
    """One block → markdown-ish text (empty string = skip)."""
    block_type = str(block.get("type", ""))
    payload = block.get(block_type) or {}
    if block_type == "child_page":
        return f"\n# {payload.get('title', 'untitled')}\n"
    if block_type in ("heading_1", "heading_2", "heading_3"):
        text = _rich_text(payload.get("rich_text"))
        return f"\n{'#' * int(block_type[-1])} {text}\n" if text else ""
    if block_type in (
        "paragraph",
        "quote",
        "callout",
        "toggle",
        "bulleted_list_item",
        "numbered_list_item",
        "to_do",
        "code",
    ):
        text = _rich_text(payload.get("rich_text"))
        if not text.strip():
            return ""
        if block_type == "bulleted_list_item":
            return f"- {text}"
        if block_type == "numbered_list_item":
            return f"1. {text}"
        if block_type == "to_do":
            return f"[{'x' if payload.get('checked') else ' '}] {text}"
        if block_type == "quote":
            return f"> {text}"
        if block_type == "code":
            return f"```{payload.get('language', '')}\n{text}\n```"
        return text
    if block_type == "table_row":
        cells = payload.get("cells") or []
        return " | ".join(_rich_text(cell) for cell in cells)
    return ""


def page_text(block_id: str, token: str, *, timeout: float = 60.0) -> str:
    """Walk a page's blocks depth-first (child pages are indexed separately)."""
    lines: list[str] = []
    queue: deque[str] = deque([block_id])
    visited = 0
    while queue and visited < MAX_BLOCKS_PER_PAGE:
        current = queue.popleft()
        cursor = ""
        while True:
            path = f"/blocks/{current}/children?page_size={PAGE_SIZE}"
            if cursor:
                path += f"&start_cursor={cursor}"
            payload = _request("GET", path, token, timeout=timeout)
            for block in payload.get("results", []):
                visited += 1
                line = block_text(block)
                if line:
                    lines.append(line)
                if block.get("has_children") and block.get("type") != "child_page":
                    queue.append(str(block.get("id")))
            if payload.get("has_more") and payload.get("next_cursor"):
                cursor = str(payload["next_cursor"])
                continue
            break
    return "\n".join(lines).strip()


def _slug(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:48] or "page"


def list_pages(token: str, *, limit: int = 500, timeout: float = 60.0) -> list[dict]:
    pages: list[dict] = []
    cursor = ""
    while len(pages) < limit:
        body: dict = {
            "filter": {"property": "object", "value": "page"},
            "page_size": PAGE_SIZE,
        }
        if cursor:
            body["start_cursor"] = cursor
        payload = _request("POST", "/search", token, body=body, timeout=timeout)
        pages.extend(payload.get("results", []))
        if not payload.get("has_more") or not payload.get("next_cursor"):
            break
        cursor = str(payload["next_cursor"])
    return pages[:limit]


def sync_notion(
    store: Store,
    embedder: Embedder,
    *,
    token: str | None = None,
    limit: int = 500,
    timeout: float = 60.0,
) -> IndexStats:
    resolved = resolve_token(token)
    if resolved is None:
        raise NotionError("no Notion token: connect the integration in the app or set NOTION_TOKEN")
    store.ensure_embedder(embedder.name, embedder.dim)

    stats = IndexStats()
    for page in list_pages(resolved, limit=limit, timeout=timeout):
        stats.files_scanned += 1
        page_id = str(page.get("id", ""))
        edited = str(page.get("last_edited_time", ""))
        meta_key = f"notion.edited:{page_id}"
        if edited and store.get_meta(meta_key) == edited:
            stats.unchanged += 1
            continue

        title = _page_title(page)
        text = page_text(page_id, resolved, timeout=timeout)
        if not text.strip():
            stats.skipped += 1
            if edited:  # remember empty pages too — don't re-fetch every sync
                store.set_meta(meta_key, edited)
            continue
        content = f"# {title}\n\nNotion: {page.get('url', '')}\n\n{text}".strip()
        chunks = index_document(
            store,
            embedder,
            source="notion",
            path=f"notion://{page_id}/{_slug(title)}",
            content=content,
        )
        if chunks:
            stats.indexed += 1
            stats.chunks += chunks
        else:
            stats.unchanged += 1
        if edited:
            store.set_meta(meta_key, edited)
    return stats
