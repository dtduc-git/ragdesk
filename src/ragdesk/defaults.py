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
"""

GITHUB_CLIENT_ID = "Ov23liI2iSz3dhYO9yaM"

ATLASSIAN_CLIENT_ID = ""
ATLASSIAN_CLIENT_SECRET = ""

GOOGLE_CLIENT_ID = ""
GOOGLE_CLIENT_SECRET = ""
