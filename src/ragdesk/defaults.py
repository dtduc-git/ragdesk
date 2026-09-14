"""Public OAuth client defaults shipped with ragdesk.

OAuth servers require every client to be a registered application — that is
the protocol, not a licensing quirk. The registrations are one-time and belong
to the maintainer; **users** never need to create anything when these are set.

Client IDs are public by definition. Client secrets for installed apps are,
per Google/Atlassian documentation, not confidential (native apps cannot keep
secrets; PKCE is what protects the flow) — they still must not be committed
together with user tokens.

Maintainers: fill these in at release time, or set the environment variables
(``RAGDESK_GITHUB_CLIENT_ID``, ``RAGDESK_ATLASSIAN_CLIENT_ID/SECRET``,
``GDRIVE_CLIENT_ID/SECRET``) / let users bring their own via the Sources UI.
"""

GITHUB_CLIENT_ID = "Ov23liI2iSz3dhYO9yaM"

ATLASSIAN_CLIENT_ID = ""
ATLASSIAN_CLIENT_SECRET = ""

GOOGLE_CLIENT_ID = ""
GOOGLE_CLIENT_SECRET = ""
