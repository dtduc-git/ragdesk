"""GitHub connector: index a repository from its tarball (read-only).

Token resolution: explicit token -> ``GITHUB_TOKEN``/``GH_TOKEN`` env ->
``gh auth token``. RFC 8628 device-flow helpers are provided for clients that
want to log in without the gh CLI.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from ragdesk import credentials
from ragdesk.embed import Embedder
from ragdesk.index import IndexStats, index_document, is_indexable
from ragdesk.store import Store

API = "https://api.github.com"
GH_HOST = "https://github.com"
USER_AGENT = "ragdesk"


class GitHubError(RuntimeError):
    """GitHub API / auth failure."""


# --- token resolution ---------------------------------------------------------


def token_source(explicit: str | None = None) -> tuple[str, str] | None:
    """Return ``(source, token)`` for the first available credential."""
    if explicit:
        return ("explicit", explicit)
    env = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if env:
        return ("env", env)
    stored = credentials.get("github").get("token")
    if stored:
        return ("credentials", str(stored))
    token = _token_from_gh()
    if token:
        return ("gh", token)
    return None


def resolve_token(explicit: str | None = None) -> str | None:
    found = token_source(explicit)
    return found[1] if found else None


def resolve_client_id(explicit: str | None = None) -> str | None:
    """OAuth client ID for the device flow (env or saved in credentials)."""
    if explicit:
        return explicit
    env = os.environ.get("RAGDESK_GITHUB_CLIENT_ID")
    if env:
        return env
    stored = credentials.get("github").get("client_id")
    return str(stored) if stored else None


def whoami(token: str, *, timeout: float = 30.0) -> str:
    """Validate a token against the API and return the login name."""
    request = urllib.request.Request(
        f"{API}/user",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise GitHubError(f"GitHub rejected the token (HTTP {exc.code})") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise GitHubError(f"cannot reach GitHub: {exc}") from exc
    return str(payload.get("login", ""))


def _token_from_gh() -> str | None:
    gh = shutil.which("gh")
    if not gh:
        return None
    try:
        result = subprocess.run(
            [gh, "auth", "token"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    token = result.stdout.strip()
    return token or None


# --- device flow (RFC 8628) -----------------------------------------------------


def _post_form(url: str, data: dict[str, str], timeout: float = 30.0) -> dict:
    body = urllib.parse.urlencode(data).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Accept": "application/json", "User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise GitHubError(
            f"GitHub rejected the request (HTTP {exc.code}) — check the OAuth client ID "
            "and that device flow is enabled for the app"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise GitHubError(f"cannot reach GitHub: {exc}") from exc


def device_flow_start(client_id: str, *, host: str = GH_HOST) -> dict:
    """Start the device flow; returns user_code, verification_uri, device_code..."""
    return _post_form(f"{host}/login/device/code", {"client_id": client_id, "scope": "repo"})


def device_flow_poll_once(
    client_id: str, device_code: str, *, host: str = GH_HOST
) -> tuple[str, str | None]:
    """One poll iteration: ("token", value) | ("pending", None) | ("slow_down", None)."""
    payload = _post_form(
        f"{host}/login/oauth/access_token",
        {
            "client_id": client_id,
            "device_code": device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        },
    )
    if "access_token" in payload:
        return ("token", str(payload["access_token"]))
    error = payload.get("error", "")
    if error == "authorization_pending":
        return ("pending", None)
    if error == "slow_down":
        return ("slow_down", None)
    raise GitHubError(f"device flow failed: {error or 'unknown error'}")


def device_flow_poll(
    client_id: str,
    device_code: str,
    *,
    host: str = GH_HOST,
    interval: float = 5.0,
    timeout: float = 300.0,
) -> str:
    """Poll until the user approves; returns the access token."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(interval)
        status, value = device_flow_poll_once(client_id, device_code, host=host)
        if status == "token" and value:
            return value
        if status == "slow_down":
            interval += 1.0
    raise GitHubError("device flow timed out")


# --- tarball sync ---------------------------------------------------------------


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return None


def _request(url: str, *, token: str | None = None) -> urllib.request.Request:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def _download_tarball(repo: str, ref: str, token: str, *, timeout: float = 180.0) -> bytes:
    url = f"{API}/repos/{repo}/tarball" + (f"/{urllib.parse.quote(ref)}" if ref else "")
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        opener.open(_request(url, token=token), timeout=timeout)
        raise GitHubError(f"unexpected non-redirect response for {repo}")
    except urllib.error.HTTPError as exc:
        if exc.code in (301, 302, 303, 307, 308):
            redirect = exc.headers.get("Location", "")
            if not redirect:
                raise GitHubError(f"empty tarball redirect for {repo}") from exc
        else:
            raise GitHubError(f"GitHub API error {exc.code} for {repo}") from exc
    # Signed codeload URL: no Authorization header needed (or wanted).
    with urllib.request.urlopen(redirect, timeout=timeout) as response:
        return response.read()


def _tar_members(data: bytes, subdir: str) -> list[tuple[str, str]]:
    """Return (relative_path, text) for indexable files in the tarball."""
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


def sync_github(
    store: Store,
    embedder: Embedder,
    *,
    repo: str,
    token: str | None = None,
    ref: str = "",
    subdir: str = "",
    timeout: float = 180.0,
) -> IndexStats:
    """Download and index a GitHub repository tarball (whole-repo sha skip)."""
    resolved = resolve_token(token)
    if resolved is None:
        raise GitHubError(
            "no GitHub token found: set GITHUB_TOKEN, run 'gh auth login', or pass a token"
        )
    store.ensure_embedder(embedder.name, embedder.dim)

    data = _download_tarball(repo, ref, resolved, timeout=timeout)
    digest = hashlib.sha256(data).hexdigest()
    meta_key = f"github.sha:{repo}@{ref or 'default'}"
    if store.get_meta(meta_key) == digest:
        return IndexStats()

    stats = IndexStats()
    for rel, content in _tar_members(data, subdir):
        stats.files_scanned += 1
        if not content.strip():
            stats.skipped += 1
            continue
        chunks = index_document(
            store,
            embedder,
            source=f"github:{repo}",
            path=f"github://{repo}/{rel}",
            content=content,
        )
        if chunks:
            stats.indexed += 1
            stats.chunks += chunks
        else:
            stats.unchanged += 1
    store.set_meta(meta_key, digest)
    return stats
