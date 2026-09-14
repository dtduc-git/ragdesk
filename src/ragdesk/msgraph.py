"""Microsoft Graph connector: index OneDrive / SharePoint files (read-only).

Auth: **device flow** (RFC 8628) against Azure AD using a *public client* app
registration — no secret involved. Scopes: ``Files.Read.All`` (plus
``Sites.Read.All`` for SharePoint sites) and ``offline_access`` so ragdesk can
keep a refresh token.

Client id: create the app registration once (see ``docs/sources.md``) and set
``RAGDESK_MS_CLIENT_ID`` (or paste the id in the app card). Client ids are
public — they may ship in release builds via ``.env``/``buildenv``.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import deque
from io import BytesIO
from pathlib import Path

from ragdesk import credentials, defaults
from ragdesk.embed import Embedder
from ragdesk.index import IndexStats, index_document, is_indexable
from ragdesk.store import Store

TENANT = "common"
DEVICE_CODE_URL = f"https://login.microsoftonline.com/{TENANT}/oauth2/v2.0/devicecode"
TOKEN_URL = f"https://login.microsoftonline.com/{TENANT}/oauth2/v2.0/token"
GRAPH = "https://graph.microsoft.com/v1.0"
SCOPES = "Files.Read.All Sites.Read.All offline_access"
USER_AGENT = "ragdesk-msgraph/0.1"
MAX_ITEMS = 2000
OFFICE_SUFFIXES = {".docx", ".pptx"}


class MsGraphError(RuntimeError):
    """Microsoft Graph / Azure AD failure."""


def resolve_client_id(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    env = os.environ.get("RAGDESK_MS_CLIENT_ID")
    if env:
        return env
    stored = credentials.get("msgraph").get("client_id")
    if stored:
        return str(stored)
    shipped = defaults.MS_CLIENT_ID.strip()
    return shipped or None


# --- device flow (RFC 8628) -----------------------------------------------------


def _post_form(url: str, data: dict[str, str], *, timeout: float = 60.0) -> dict:
    body = urllib.parse.urlencode(data).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:300]
        try:
            parsed = json.loads(detail)
        except json.JSONDecodeError:
            parsed = {}
        error = str(parsed.get("error", "") or "")
        if error in ("authorization_pending", "slow_down"):
            return parsed
        raise MsGraphError(
            f"Azure AD rejected the request ({exc.code}): "
            f"{parsed.get('error_description', detail)!r}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise MsGraphError(f"cannot reach Azure AD: {exc}") from exc


def device_flow_start(client_id: str, *, timeout: float = 60.0) -> dict:
    return _post_form(
        DEVICE_CODE_URL, {"client_id": client_id, "scope": SCOPES}, timeout=timeout
    )


def device_flow_poll_once(
    client_id: str, device_code: str, *, timeout: float = 60.0
) -> tuple[str, dict]:
    """("token", token_payload) | ("pending", {}) | ("slow_down", {})."""
    payload = _post_form(
        TOKEN_URL,
        {
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "client_id": client_id,
            "device_code": device_code,
        },
        timeout=timeout,
    )
    if "access_token" in payload:
        return ("token", payload)
    error = str(payload.get("error", ""))
    if error == "authorization_pending":
        return ("pending", {})
    if error == "slow_down":
        return ("slow_down", {})
    raise MsGraphError(f"device flow failed: {error or 'unknown error'}")


def refresh_access_token(client_id: str, refresh_token: str, *, timeout: float = 60.0) -> dict:
    return _post_form(
        TOKEN_URL,
        {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "refresh_token": refresh_token,
            "scope": SCOPES,
        },
        timeout=timeout,
    )


def device_flow_connect(
    client_id: str, *, on_code=None, interval_default: float = 5.0
) -> dict:
    """Run the full device flow (start → wait → save tokens) for the CLI."""
    start = device_flow_start(client_id)
    if on_code is not None:
        on_code(start)
    interval = float(start.get("interval", interval_default) or interval_default)
    while True:
        time.sleep(max(interval, 1.0))
        status, payload = device_flow_poll_once(client_id, str(start.get("device_code", "")))
        if status == "pending":
            continue
        if status == "slow_down":
            interval += 5
            continue
        token = str(payload.get("access_token", ""))
        account = whoami(token)
        updates: dict = {"client_id": client_id, "account": account}
        if payload.get("refresh_token"):
            updates["refresh_token"] = str(payload["refresh_token"])
        credentials.set_provider("msgraph", updates)
        return payload


# --- Graph helpers ----------------------------------------------------------------


def _auth_request(url: str, token: str) -> urllib.request.Request:
    return urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT}
    )


def _get_json(url: str, token: str, *, timeout: float = 60.0) -> dict:
    try:
        with urllib.request.urlopen(_auth_request(url, token), timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise MsGraphError(
                "Microsoft Graph rejected the token (401/403) — reconnect and check the "
                "app permissions (Files.Read.All, Sites.Read.All)."
            ) from exc
        raise MsGraphError(f"Graph API error {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise MsGraphError(f"cannot reach Microsoft Graph: {exc}") from exc


def _get_bytes(url: str, token: str, *, timeout: float = 180.0) -> bytes:
    try:
        with urllib.request.urlopen(_auth_request(url, token), timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise MsGraphError(f"drive download failed (HTTP {exc.code})") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise MsGraphError(f"cannot reach Microsoft Graph: {exc}") from exc


def whoami(access_token: str, *, timeout: float = 30.0) -> str:
    payload = _get_json(
        f"{GRAPH}/me?$select=displayName,userPrincipalName", access_token, timeout=timeout
    )
    return str(
        payload.get("userPrincipalName") or payload.get("displayName") or "microsoft"
    )


def resolve_access_token(
    client_id: str | None = None,
    *,
    access_token: str = "",
    timeout: float = 60.0,
) -> str:
    """Reuse a provided/polled token, else refresh the stored one."""
    if access_token:
        return access_token
    entry = credentials.get("msgraph")
    refresh_token = str(entry.get("refresh_token", ""))
    if not refresh_token:
        raise MsGraphError("not connected: run the Microsoft device flow first")
    client = resolve_client_id(client_id)
    if client is None:
        raise MsGraphError("no Microsoft client id configured (RAGDESK_MS_CLIENT_ID)")
    payload = refresh_access_token(client, refresh_token, timeout=timeout)
    token = str(payload.get("access_token", ""))
    if not token:
        raise MsGraphError("token refresh returned no access_token — reconnect")
    updates: dict = {"access_token": token}
    rotated = payload.get("refresh_token")
    if rotated:  # Azure rotates refresh tokens sometimes; keep the newest
        updates["refresh_token"] = str(rotated)
    credentials.set_provider("msgraph", updates)
    return token


def extract_office_text(data: bytes, suffix: str) -> str | None:
    """Extract text from .docx / .pptx (both are zip+XML) without dependencies."""
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            if suffix == ".docx":
                xml = archive.read("word/document.xml").decode("utf-8", "replace")
            elif suffix == ".pptx":
                names = sorted(
                    name
                    for name in archive.namelist()
                    if name.startswith("ppt/slides/slide") and name.endswith(".xml")
                )
                xml = "\n".join(
                    archive.read(name).decode("utf-8", "replace") for name in names
                )
            else:
                return None
    except (KeyError, zipfile.BadZipFile, OSError):
        return None
    text = re.sub(r"</(?:w:p|a:p)>", "\n", xml)
    text = re.sub(r"<[^>]+>", "", text)
    import html as html_module

    return html_module.unescape(text).strip() or None


def _drive_base(access_token: str, site: str, timeout: float) -> str:
    if not site:
        return f"{GRAPH}/me/drive"
    payload = _get_json(f"{GRAPH}/sites/{site}", access_token, timeout=timeout)
    site_id = str(payload.get("id", ""))
    if not site_id:
        raise MsGraphError(f"cannot resolve SharePoint site {site!r}")
    return f"{GRAPH}/sites/{site_id}/drive"


def walk_drive(
    access_token: str,
    *,
    site: str = "",
    folder_id: str = "",
    max_items: int = MAX_ITEMS,
    timeout: float = 60.0,
) -> list[dict]:
    """Breadth-first file listing of a drive (folders are queues, not results)."""
    base = _drive_base(access_token, site, timeout)
    root = f"{base}/items/{folder_id}/children" if folder_id else f"{base}/root/children"
    queue: deque[str] = deque([root])
    files: list[dict] = []
    visited_folders = 0
    while queue and len(files) < max_items and visited_folders < 200:
        url = queue.popleft()
        visited_folders += 1
        while url:
            payload = _get_json(url, access_token, timeout=timeout)
            for item in payload.get("value", []):
                if "folder" in item:
                    queue.append(f"{base}/items/{item.get('id')}/children")
                    continue
                if "file" in item:
                    files.append(item)
                    if len(files) >= max_items:
                        break
            url = str(payload.get("@odata.nextLink", "") or "")
        url = ""
    return files[:max_items]


def sync_onedrive(
    store: Store,
    embedder: Embedder,
    *,
    site: str = "",
    folder_id: str = "",
    access_token: str = "",
    client_id: str | None = None,
    max_items: int = MAX_ITEMS,
    timeout: float = 60.0,
) -> IndexStats:
    token = resolve_access_token(
        client_id, access_token=access_token, timeout=timeout
    )
    store.ensure_embedder(embedder.name, embedder.dim)
    host = "sharepoint" if site else "onedrive"
    drive = _drive_base(token, site, timeout)
    stats = IndexStats()

    for item in walk_drive(
        token, site=site, folder_id=folder_id, max_items=max_items, timeout=timeout
    ):
        stats.files_scanned += 1
        name = str(item.get("name", "untitled"))
        suffix = Path(name).suffix.lower()
        size = int(item.get("size") or 0)
        text: str | None = None
        if suffix in OFFICE_SUFFIXES:
            raw = _get_bytes(f"{drive}/items/{item['id']}/content", token)
            text = extract_office_text(raw, suffix)
        elif is_indexable(Path(name), max(size, 1)):
            raw = _get_bytes(f"{drive}/items/{item['id']}/content", token)
            text = raw.decode("utf-8", errors="replace")
        if text is None or not text.strip():
            stats.skipped += 1
            continue

        path = f"{host}://{item.get('id', '')}/{name}"
        chunks = index_document(
            store, embedder, source=f"msgraph:{host}", path=path, content=text
        )
        if chunks:
            stats.indexed += 1
            stats.chunks += chunks
        else:
            stats.unchanged += 1
    return stats
