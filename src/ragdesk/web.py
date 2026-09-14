"""Website source: crawl a docs site and index its pages (read-only).

A deliberately small breadth-first crawler: HTTP(S) only, same host by
default, HTML only, capped by ``max_pages`` and ``max_depth``. No JavaScript
rendering — for JS-heavy sites a managed crawler (e.g. Bedrock KB's web
connector) is the right tool; this one stays simple, local and dependency-free.
"""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from html.parser import HTMLParser

from ragdesk.embed import Embedder
from ragdesk.htmlutil import html_to_text
from ragdesk.index import IndexStats, index_document
from ragdesk.store import Store

USER_AGENT = "ragdesk-web/0.1 (+https://github.com/dtduc-git/ragdesk)"
MAX_BYTES = 1_000_000
DEFAULT_MAX_PAGES = 50
DEFAULT_MAX_DEPTH = 2


class WebError(RuntimeError):
    """Fetch / parse failure."""


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            for key, value in attrs:
                if key == "href" and value:
                    self.links.append(value)
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title and not self.title.strip():
            self.title = data.strip()


def _fetch(url: str, *, timeout: float = 30.0) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = (response.headers.get("Content-Type") or "").lower()
            if "html" not in content_type:
                raise WebError(f"not HTML ({content_type or 'unknown'}): {url}")
            raw = response.read(MAX_BYTES)
    except urllib.error.HTTPError as exc:
        raise WebError(f"HTTP {exc.code}: {url}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise WebError(f"cannot fetch {url}: {exc}") from exc
    return raw.decode("utf-8", errors="replace")


def crawl_site(
    store: Store,
    embedder: Embedder,
    *,
    start_url: str,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_depth: int = DEFAULT_MAX_DEPTH,
    same_host: bool = True,
    timeout: float = 30.0,
) -> IndexStats:
    """Breadth-first crawl from ``start_url``; index each HTML page."""
    parsed_start = urllib.parse.urlparse(start_url)
    if parsed_start.scheme not in ("http", "https") or not parsed_start.netloc:
        raise WebError(f"invalid start URL: {start_url!r}")
    store.ensure_embedder(embedder.name, embedder.dim)

    host = parsed_start.netloc
    seen: set[str] = {start_url}
    queue: deque[tuple[str, int]] = deque([(start_url, 0)])
    stats = IndexStats()

    while queue and stats.files_scanned < max_pages:
        url, depth = queue.popleft()
        try:
            page = _fetch(url, timeout=timeout)
        except WebError:
            stats.skipped += 1
            continue
        stats.files_scanned += 1

        parser = _PageParser()
        parser.feed(page)
        text = html_to_text(page)
        if not text.strip():
            stats.skipped += 1
            continue

        title = parser.title or url
        content = f"# {title}\n\nURL: {url}\n\n{text}".strip()
        path = f"web://{host}{urllib.parse.urlparse(url).path or '/'}"
        chunks = index_document(
            store, embedder, source=f"web:{host}", path=path, content=content
        )
        if chunks:
            stats.indexed += 1
            stats.chunks += chunks
        else:
            stats.unchanged += 1

        if depth < max_depth:
            for link in parser.links:
                absolute = urllib.parse.urldefrag(urllib.parse.urljoin(url, link))[0]
                target = urllib.parse.urlparse(absolute)
                if target.scheme not in ("http", "https"):
                    continue
                if same_host and target.netloc != host:
                    continue
                if absolute in seen:
                    continue
                seen.add(absolute)
                queue.append((absolute, depth + 1))
    return stats
