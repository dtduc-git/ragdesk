"""Google Drive connector: read-only sync with a bring-your-own OAuth client.

Uses the native-app OAuth flow (RFC 8252): PKCE + loopback redirect, opened in
the user's browser. Refresh token is stored 0600 under ``~/.config/ragdesk``.
Credentials can come from params or ``GDRIVE_CLIENT_ID`` / ``GDRIVE_CLIENT_SECRET``.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from ragdesk import credentials, defaults
from ragdesk.embed import Embedder
from ragdesk.index import IndexStats, index_document
from ragdesk.oauth import new_state, pkce_pair, run_loopback
from ragdesk.store import Store

SCOPES = "https://www.googleapis.com/auth/drive.readonly"
GOOGLE_AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN = "https://oauth2.googleapis.com/token"
DRIVE_API = "https://www.googleapis.com/drive/v3"
TOKEN_FILE = Path.home() / ".config" / "ragdesk" / "gdrive.json"

GOOGLE_DOC = "application/vnd.google-apps.document"
GOOGLE_SHEET = "application/vnd.google-apps.spreadsheet"
GOOGLE_SLIDES = "application/vnd.google-apps.presentation"
GOOGLE_FOLDER = "application/vnd.google-apps.folder"
EXPORT_MAP = {
    GOOGLE_DOC: "text/plain",
    GOOGLE_SHEET: "text/csv",
    GOOGLE_SLIDES: "text/plain",
}
TEXT_MIMES = {"application/json", "application/xml", "application/x-yaml"}


class GdriveError(RuntimeError):
    """Google Drive auth / API failure."""


# --- OAuth (RFC 8252 native app) -------------------------------------------------


def auth_url(client_id: str, redirect_uri: str, challenge: str, state: str) -> str:
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }
    return f"{GOOGLE_AUTH}?{urllib.parse.urlencode(params)}"


def _post_form(url: str, data: dict[str, str], timeout: float = 60.0) -> dict:
    body = urllib.parse.urlencode(data).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise GdriveError(f"Google OAuth error {exc.code}: {exc.read()[:200]!r}") from exc


def exchange_code(
    client_id: str,
    client_secret: str,
    code: str,
    verifier: str,
    redirect_uri: str,
) -> dict:
    payload = {
        "client_id": client_id,
        "code": code,
        "code_verifier": verifier,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    }
    if client_secret:
        payload["client_secret"] = client_secret
    return _post_form(GOOGLE_TOKEN, payload)


def refresh_access_token(client_id: str, client_secret: str, refresh_token: str) -> str:
    payload = {
        "client_id": client_id,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    if client_secret:
        payload["client_secret"] = client_secret
    data = _post_form(GOOGLE_TOKEN, payload)
    token = data.get("access_token")
    if not token:
        raise GdriveError("token refresh returned no access_token")
    return str(token)


def _callback_code(result: dict[str, str], expected_state: str) -> str:
    if not result:
        raise GdriveError("OAuth flow timed out (no redirect captured)")
    if result.get("state") != expected_state:
        raise GdriveError("OAuth state mismatch — aborting")
    if "code" not in result:
        raise GdriveError(f"OAuth error: {result.get('error', 'no code returned')}")
    return result["code"]


def run_loopback_flow(
    client_id: str,
    client_secret: str = "",
    *,
    token_file: Path = TOKEN_FILE,
    port: int = 0,
    timeout: float = 300.0,
) -> dict:
    """Open the browser, capture the loopback redirect, exchange the code."""
    verifier, challenge = pkce_pair()
    state = new_state()
    redirect: dict[str, str] = {}

    def build_url(actual_port: int) -> str:
        redirect["uri"] = f"http://127.0.0.1:{actual_port}/oauth/callback"
        return auth_url(client_id, redirect["uri"], challenge, state)

    result = run_loopback(build_url, port=port, path="/oauth/callback", timeout=timeout)
    code = _callback_code(result, state)
    payload = exchange_code(client_id, client_secret, code, verifier, redirect["uri"])
    save_token_file(payload, token_file)
    return payload


def save_token_file(payload: dict, path: Path = TOKEN_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    path.chmod(0o600)


def load_token_file(path: Path = TOKEN_FILE) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


# --- Drive API --------------------------------------------------------------------


def _auth_request(url: str, access_token: str) -> urllib.request.Request:
    return urllib.request.Request(
        url, headers={"Authorization": f"Bearer {access_token}", "User-Agent": "ragdesk"}
    )


def _get_json(url: str, access_token: str, *, timeout: float = 60.0) -> dict:
    try:
        with urllib.request.urlopen(_auth_request(url, access_token), timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise GdriveError(f"Drive API error {exc.code}: {exc.read()[:200]!r}") from exc


def _get_bytes(url: str, access_token: str, *, timeout: float = 120.0) -> bytes:
    try:
        with urllib.request.urlopen(_auth_request(url, access_token), timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise GdriveError(f"Drive download error {exc.code}") from exc


def _list_files(access_token: str, folder_id: str) -> Iterator[dict]:
    query = f"'{folder_id}' in parents and trashed=false" if folder_id else "trashed=false"
    page_token = ""
    while True:
        params = {
            "q": query,
            "pageSize": "100",
            "fields": "nextPageToken,files(id,name,mimeType)",
        }
        if page_token:
            params["pageToken"] = page_token
        payload = _get_json(f"{DRIVE_API}/files?{urllib.parse.urlencode(params)}", access_token)
        yield from payload.get("files", [])
        page_token = payload.get("nextPageToken", "")
        if not page_token:
            return


def _fetch_text(file: dict, access_token: str) -> str | None:
    file_id = str(file.get("id", ""))
    mime = str(file.get("mimeType", ""))
    if mime in EXPORT_MAP:
        export = urllib.parse.quote(EXPORT_MAP[mime])
        url = f"{DRIVE_API}/files/{file_id}/export?mimeType={export}"
        return _get_bytes(url, access_token).decode("utf-8", errors="replace")
    if mime.startswith("text/") or mime in TEXT_MIMES:
        return _get_bytes(f"{DRIVE_API}/files/{file_id}?alt=media", access_token).decode(
            "utf-8", errors="replace"
        )
    return None


def resolve_client_credentials(client_id: str = "", client_secret: str = "") -> tuple[str, str]:
    """(client_id, client_secret): explicit > env > saved > shipped defaults."""
    stored = credentials.get("gdrive")
    resolved_id = (
        client_id
        or os.environ.get("GDRIVE_CLIENT_ID")
        or str(stored.get("client_id", ""))
        or defaults.GOOGLE_CLIENT_ID
    )
    resolved_secret = (
        client_secret
        or os.environ.get("GDRIVE_CLIENT_SECRET")
        or str(stored.get("client_secret", ""))
        or defaults.GOOGLE_CLIENT_SECRET
    )
    return str(resolved_id), str(resolved_secret)


def whoami(access_token: str, *, timeout: float = 30.0) -> str:
    """Return the connected account's email address."""
    payload = _get_json(
        f"{DRIVE_API}/about?fields=user/emailAddress", access_token, timeout=timeout
    )
    return str((payload.get("user") or {}).get("emailAddress", ""))


def resolve_access_token(
    client_id: str = "",
    client_secret: str = "",
    *,
    access_token: str = "",
    token_file: Path = TOKEN_FILE,
    interactive: bool = True,
) -> str:
    if access_token:
        return access_token
    stored = load_token_file(token_file)
    if not stored.get("refresh_token"):
        if not interactive or not client_id:
            raise GdriveError(
                "no Google credentials: run the OAuth flow (needs a client ID) or pass an "
                "access token. Set GDRIVE_CLIENT_ID env for the browser flow."
            )
        stored = run_loopback_flow(client_id, client_secret, token_file=token_file)
    if stored.get("access_token") and not stored.get("refresh_token"):
        return str(stored["access_token"])
    return refresh_access_token(client_id, client_secret, str(stored["refresh_token"]))


def sync_gdrive(
    store: Store,
    embedder: Embedder,
    *,
    client_id: str = "",
    client_secret: str = "",
    access_token: str = "",
    folder_id: str = "",
    token_file: Path = TOKEN_FILE,
    interactive: bool = True,
) -> IndexStats:
    """Index Google Docs/Sheets/Slides + text files from Drive (reads only)."""
    client_id, client_secret = resolve_client_credentials(client_id, client_secret)
    token = resolve_access_token(
        client_id,
        client_secret,
        access_token=access_token,
        token_file=token_file,
        interactive=interactive,
    )
    store.ensure_embedder(embedder.name, embedder.dim)

    stats = IndexStats()
    for file in _list_files(token, folder_id):
        if file.get("mimeType") == GOOGLE_FOLDER:
            continue
        stats.files_scanned += 1
        file_id = str(file.get("id", ""))
        name = str(file.get("name", "untitled"))
        text = _fetch_text(file, token)
        if text is None or not text.strip():
            stats.skipped += 1
            continue
        content = f"# {name}\n\n{text}".strip()
        chunks = index_document(
            store,
            embedder,
            source="gdrive",
            path=f"gdrive://{file_id}/{name}",
            content=content,
        )
        if chunks:
            stats.indexed += 1
            stats.chunks += chunks
        else:
            stats.unchanged += 1
    return stats
