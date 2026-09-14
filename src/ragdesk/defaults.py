"""Public OAuth client defaults shipped with ragdesk.

OAuth servers require every client to be a registered application — that is
the protocol, not a licensing quirk. The registrations are one-time and belong
to the maintainer; **users** never need to create anything when these are set.

Per-provider policy on what may live in this file (it is public):

- GitHub device flow: **client ID only** — device flow uses no client secret.
- Google (Desktop app): client ID + secret may ship; Google documents that
  installed apps cannot keep secrets (PKCE protects the flow), and this is
  common practice for OSS desktop tools.
- Atlassian 3LO: **client ID only at most; never the secret.** Atlassian has
  no installed-app model — the secret is a real confidential credential, and
  shipping it would let anyone impersonate the app. Users bring their own
  app credentials (stored locally, 0600) or use an API token instead.

Users can always override via the Sources UI (saved under
``~/.config/ragdesk/credentials.json``) or environment variables:
``RAGDESK_GITHUB_CLIENT_ID``, ``RAGDESK_ATLASSIAN_CLIENT_ID/SECRET``,
``GDRIVE_CLIENT_ID/SECRET``.

Release builds get their values from a local, gitignored ``.env`` via
``python -m ragdesk.buildenv`` (writes ``ragdesk/_build_env.py``, also
gitignored). A repository checkout therefore ships nothing.
"""

from __future__ import annotations


def shipped(const_name: str) -> str:
    """Value baked at build time by ``ragdesk.buildenv`` (empty in a checkout)."""
    try:
        from ragdesk import _build_env
    except ImportError:
        return ""
    return str(getattr(_build_env, const_name, "") or "")


GITHUB_CLIENT_ID = shipped("GITHUB_CLIENT_ID")

ATLASSIAN_CLIENT_ID = shipped("ATLASSIAN_CLIENT_ID")
ATLASSIAN_CLIENT_SECRET = shipped("ATLASSIAN_CLIENT_SECRET")

GOOGLE_CLIENT_ID = shipped("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = shipped("GOOGLE_CLIENT_SECRET")

MS_CLIENT_ID = shipped("MS_CLIENT_ID")
