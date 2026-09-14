# Security Policy

## Posture

- **Local-first**: the index (`.ragdesk/index.db`) never leaves your machine.
- **Read-only sources**: ragdesk only reads files; it never writes to indexed
  sources.
- **Local credentials**: connection credentials (GitHub token, Confluence
  site, Google OAuth client) are stored in
  `~/.config/ragdesk/credentials.json` with mode `0600`, and Google refresh
  tokens in `~/.config/ragdesk/gdrive.json` (`0600`). Disconnect from the
  Sources panel removes them.
- **No telemetry**: no analytics, no phone-home. Network calls go only to the
  local Ollama server, unless you explicitly configure a cloud provider with
  your own key.
- **No execution**: indexed content is treated as data — never executed, never
  piped into a shell.

## Reporting

Report suspected vulnerabilities via GitHub Security Advisories
("Report a vulnerability" on the repository) rather than a public issue.
Please include reproduction steps and the affected version.
