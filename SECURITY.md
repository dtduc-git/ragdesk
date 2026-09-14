# Security Policy

## Posture

- **Local-first**: the index (`.ragdesk/index.db`) never leaves your machine.
- **Read-only sources**: ragdesk only reads files; it never writes to indexed
  sources.
- **Local credentials**: connection credentials (GitHub token, Confluence
  site, Google OAuth client, Atlassian app credentials) are stored in
  `~/.config/ragdesk/credentials.json` with mode `0600`, and Google refresh
  tokens in `~/.config/ragdesk/gdrive.json` (`0600`). Disconnect from the
  Sources panel removes them.

## What may be public in this repository

`src/ragdesk/defaults.py` ships OAuth client defaults, and this repo is
public. Per-provider policy:

- **GitHub device flow** — client ID only; the device flow uses no client
  secret, so nothing confidential is involved.
- **Google (Desktop app type)** — client ID and secret may ship; Google
  documents that installed apps cannot keep secrets, and PKCE protects the
  flow. This matches common OSS practice.
- **Atlassian 3LO** — **never ship the client secret.** Atlassian has no
  installed-app model; the secret is a confidential credential and a public
  one lets anyone impersonate the app. Users bring their own app credentials
  (stored locally) or use a Confluence API token.

If a secret is ever committed by accident: rotate it at the provider first
(that invalidates the leaked value), then remove it from the tree. Rewriting
published git history is optional and usually unnecessary after rotation.
- **No telemetry**: no analytics, no phone-home. Network calls go only to the
  local Ollama server, unless you explicitly configure a cloud provider with
  your own key.
- **No execution**: indexed content is treated as data — never executed, never
  piped into a shell.

## Reporting

Report suspected vulnerabilities via GitHub Security Advisories
("Report a vulnerability" on the repository) rather than a public issue.
Please include reproduction steps and the affected version.
