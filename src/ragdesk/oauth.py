"""Shared OAuth native-app helpers (RFC 8252): PKCE + loopback callback capture.

Used by the Google Drive and Atlassian (Confluence) connect flows.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import threading
import time
import urllib.parse
import webbrowser
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def pkce_pair() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).rstrip(b"=").decode()
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    return verifier, challenge


def new_state() -> str:
    return secrets.token_urlsafe(16)


class _CallbackHandler(BaseHTTPRequestHandler):
    result: dict[str, str] = {}
    callback_path = "/callback"

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != self.callback_path:
            self.send_response(404)
            self.end_headers()
            return
        params = urllib.parse.parse_qs(parsed.query)
        _CallbackHandler.result = {key: value[0] for key, value in params.items()}
        body = (
            b"<html><body><h3>ragdesk</h3>"
            b"<p>Authorization captured. You can close this tab.</p></body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


def run_loopback(
    build_url: Callable[[int], str],
    *,
    port: int = 0,
    path: str = "/callback",
    timeout: float = 300.0,
) -> dict[str, str]:
    """Serve exactly one OAuth callback on 127.0.0.1, open the browser, return
    the query params. ``build_url(actual_port)`` lets the caller embed the port
    in the redirect URI."""
    handler = type("BoundCallback", (_CallbackHandler,), {"callback_path": path})
    _CallbackHandler.result = {}
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError as exc:
        raise RuntimeError(
            f"cannot listen on 127.0.0.1:{port} for the OAuth callback: {exc}"
        ) from exc
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        webbrowser.open(build_url(server.server_port))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not _CallbackHandler.result:
            time.sleep(0.5)
        return dict(_CallbackHandler.result)
    finally:
        server.shutdown()
        server.server_close()
