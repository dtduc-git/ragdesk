"""Confluence connector: index pages from a space (read-only).

Two auth paths:

- API token (Basic): site URL + email + token — simplest for personal Cloud use.
- Atlassian OAuth 2.0 (3LO) with PKCE + loopback (RFC 8252): one browser
  consent, then syncs go through ``api.atlassian.com`` with a refresh token.

Sync is full-fetch: content hashes make re-indexing a no-op for unchanged pages.
"""

from __future__ import annotations

import base64
import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from ragdesk import credentials, defaults
from ragdesk.embed import Embedder
from ragdesk.index import IndexStats, index_document
from ragdesk.oauth import new_state, pkce_pair, run_loopback
from ragdesk.store import Store

DEFAULT_API_PATH = "/wiki/rest/api/content/search"
SPACE_RE = re.compile(r"^[A-Za-z0-9_-]+$")

ATLASSIAN_AUTH = "https://auth.atlassian.com"
ATLASSIAN_API = "https://api.atlassian.com"
CONFLUENCE_SCOPES = (
    "read:confluence-content.all read:confluence-space.summary read:confluence-user offline_access"
)
DEFAULT_OAUTH_PORT = 8788

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


# --- HTTP helpers ---------------------------------------------------------------


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


def _get_json_bearer(url: str, token: str, *, timeout: float = 60.0) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
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
                "Atlassian rejected the access token — reconnect the site."
            ) from exc
        raise ConfluenceError(f"Confluence API error {exc.code}") from exc


def _post_form_json(url: str, data: dict[str, str], timeout: float = 60.0) -> dict:
    body = urllib.parse.urlencode(data).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise ConfluenceError(f"Atlassian OAuth error {exc.code}: {exc.read()[:200]!r}") from exc


# --- API-token auth -------------------------------------------------------------


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


# --- Atlassian OAuth 2.0 (3LO) ----------------------------------------------------


def oauth_authorize_url(client_id: str, redirect_uri: str, state: str, challenge: str) -> str:
    params = {
        "audience": "api.atlassian.com",
        "client_id": client_id,
        "scope": CONFLUENCE_SCOPES,
        "redirect_uri": redirect_uri,
        "state": state,
        "response_type": "code",
        "prompt": "consent",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    return f"{ATLASSIAN_AUTH}/authorize?{urllib.parse.urlencode(params)}"


def exchange_oauth_code(
    client_id: str,
    client_secret: str,
    code: str,
    verifier: str,
    redirect_uri: str,
) -> dict:
    return _post_form_json(
        f"{ATLASSIAN_AUTH}/oauth/token",
        {
            "grant_type": "authorization_code",
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
        },
    )


def refresh_oauth_token(client_id: str, client_secret: str, refresh_token: str) -> dict:
    return _post_form_json(
        f"{ATLASSIAN_AUTH}/oauth/token",
        {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
        },
    )


def accessible_resources(access_token: str, *, timeout: float = 60.0) -> list[dict]:
    payload = _get_json_bearer(
        f"{ATLASSIAN_API}/oauth/token/accessible-resources", access_token, timeout=timeout
    )
    return payload if isinstance(payload, list) else []


def resolve_oauth_client() -> tuple[str, str] | None:
    """(client_id, client_secret) from saved creds, env, or shipped defaults."""
    stored = credentials.get("confluence")
    client_id = (
        stored.get("client_id")
        or os.environ.get("RAGDESK_ATLASSIAN_CLIENT_ID")
        or defaults.ATLASSIAN_CLIENT_ID
    )
    client_secret = (
        stored.get("client_secret")
        or os.environ.get("RAGDESK_ATLASSIAN_CLIENT_SECRET")
        or defaults.ATLASSIAN_CLIENT_SECRET
    )
    if client_id and client_secret:
        return str(client_id), str(client_secret)
    return None


def oauth_ready() -> bool:
    return resolve_oauth_client() is not None


def connect_oauth(
    client_id: str,
    client_secret: str,
    *,
    port: int | None = None,
    timeout: float = 300.0,
) -> dict:
    """Browser consent via a loopback callback; returns the site + tokens."""
    verify_port = port or int(os.environ.get("RAGDESK_ATLASSIAN_PORT", DEFAULT_OAUTH_PORT))
    verifier, challenge = pkce_pair()
    state = new_state()
    redirect: dict[str, str] = {}

    def build_url(actual_port: int) -> str:
        redirect["uri"] = f"http://127.0.0.1:{actual_port}/callback"
        return oauth_authorize_url(client_id, redirect["uri"], state, challenge)

    try:
        result = run_loopback(build_url, port=verify_port, timeout=timeout)
    except RuntimeError as exc:
        raise ConfluenceError(
            f"{exc}. Register http://127.0.0.1:{verify_port}/callback as a redirect "
            "URL in your Atlassian app (or change RAGDESK_ATLASSIAN_PORT)."
        ) from exc
    if not result:
        raise ConfluenceError("Atlassian OAuth timed out — no callback received")
    if result.get("state") != state:
        raise ConfluenceError("Atlassian OAuth state mismatch — aborting")
    code = result.get("code", "")
    if not code:
        raise ConfluenceError(
            f"Atlassian rejected the authorization: {result.get('error', 'no code returned')}"
        )

    token = exchange_oauth_code(client_id, client_secret, code, verifier, redirect["uri"])
    access = str(token.get("access_token", ""))
    if not access:
        raise ConfluenceError("Atlassian token exchange returned no access_token")
    resources = accessible_resources(access)
    if not resources:
        raise ConfluenceError("no accessible Confluence sites for this account")
    site = resources[0]
    return {
        "cloud_id": str(site.get("id", "")),
        "site_url": str(site.get("url", "")),
        "site_name": str(site.get("name", "")),
        "access_token": access,
        "refresh_token": str(token.get("refresh_token", "")),
        "expires_in": int(token.get("expires_in", 3600)),
    }


def resolve_oauth_credentials() -> tuple[str, str] | None:
    """(api_base, bearer) using the stored OAuth session, refreshing when stale."""
    entry = credentials.get("confluence")
    cloud_id = str(entry.get("cloud_id", ""))
    refresh_token = str(entry.get("refresh_token", ""))
    if not cloud_id or not refresh_token:
        return None
    client = resolve_oauth_client()
    if client is None:
        raise ConfluenceError(
            "stored Atlassian session has no client credentials to refresh with — reconnect"
        )
    client_id, client_secret = client
    access = str(entry.get("access_token", ""))
    expires_at = float(entry.get("expires_at", 0) or 0)
    if not access or time.time() >= expires_at - 60:
        data = refresh_oauth_token(client_id, client_secret, refresh_token)
        access = str(data.get("access_token", ""))
        if not access:
            raise ConfluenceError("Atlassian token refresh failed — reconnect the site")
        credentials.set_provider(
            "confluence",
            {
                "access_token": access,
                "expires_at": time.time() + int(data.get("expires_in", 3600)),
            },
        )
    return f"{ATLASSIAN_API}/ex/confluence/{cloud_id}", access


# --- sync -------------------------------------------------------------------------


def sync_confluence(
    store: Store,
    embedder: Embedder,
    *,
    space: str,
    base_url: str = "",
    email: str | None = None,
    token: str | None = None,
    api_path: str = DEFAULT_API_PATH,
    limit: int = 100,
    timeout: float = 60.0,
    api_base: str = "",
    bearer: str = "",
    label_host: str = "",
) -> IndexStats:
    """Fetch and index every page in a space (walks v1 ``_links.next``)."""
    if not SPACE_RE.match(space):
        raise ConfluenceError(f"invalid space key: {space!r}")

    cql = urllib.parse.quote(f'space="{space}" and type=page')
    if bearer and api_base:
        base = api_base.rstrip("/")

        def fetch(target: str) -> dict:
            return _get_json_bearer(target, bearer, timeout=timeout)

    else:
        if not base_url:
            raise ConfluenceError(
                "Confluence needs a site URL (API-token mode) or an OAuth connection"
            )
        email, token = _resolve_credentials(email, token)
        auth = base64.b64encode(f"{email}:{token}".encode()).decode()
        base = base_url.rstrip("/")

        def fetch(target: str) -> dict:
            return _get_json(target, auth, timeout=timeout)

    store.ensure_embedder(embedder.name, embedder.dim)
    host = label_host or urllib.parse.urlparse(base).netloc or base
    url = f"{base}{api_path}?cql={cql}&limit={limit}&expand=body.storage,version"

    stats = IndexStats()
    while url:
        payload = fetch(url)
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
