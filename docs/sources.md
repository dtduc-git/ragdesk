# Connecting your sources

Everything ragdesk indexes is read-only, and everything it stores stays on
your machine. Open the **Sources** tab in the desktop app (or use the CLI
equivalents below) and connect what you need.

| Source | Auth | Registration needed | Where credentials live |
|---|---|---|---|
| Local files / git repos | none | no | — |
| GitHub | device code (default), `gh` login, or a token | no (a client ID ships with the app) | token in `credentials.json` (0600) |
| Confluence | API token (default) or your own Atlassian OAuth app | no for API token | token / app credentials in `credentials.json` (0600) |
| Google Drive | browser consent (OAuth, PKCE) | no (a client ID ships with the app) | refresh token in `gdrive.json` (0600) |
| Notion | internal integration token | no (you create the integration, 1 min) | token in `credentials.json` (0600) |
| GitLab | personal access token (`read_api`) | no | token in `credentials.json` (0600) |
| Microsoft (OneDrive / SharePoint) | device flow (Azure public client) | one Azure app registration (~2 min) | refresh token in `credentials.json` (0600) |
| Website | none | no | — |

Credential resolution order everywhere: **explicit input → environment
variable → saved credentials (UI) → build-time baked value**. Disconnect in
the Sources card removes saved credentials.

---

## Local files

- **Desktop app**: `Choose folder…` / `Choose files…`, then **Index**.
  Incremental — unchanged files are skipped by content hash; deleted files
  stay in the index until you re-index.
- **CLI**: `ragdesk index ~/notes ~/repos/myrepo`

## GitHub

Read-only repository sync (whole-repo tarball, incremental by commit SHA).

1. Sources → GitHub → **Connect with a code**. GitHub shows an 8-character
   code in your browser (`github.com/login/device`) — approve once.
2. Then enter `owner/name` (optionally a branch or subfolder) and **Sync repo**.

Alternatives shown in the same card: **Use gh login** (reuses your `gh` CLI
session) or paste a personal access token.

CLI: `ragdesk github owner/name --ref main --subdir docs`

<details>
<summary>Self-hosting / BYO OAuth app (maintainers)</summary>

Create an OAuth App at `github.com/settings/applications/new`:
- Application name: ragdesk
- Homepage URL: any page you own
- Redirect URI: any valid URL (unused by device flow)
- ✅ **Enable Device Flow**
- Leave *Expire user access tokens* **unchecked** (non-expiring tokens)

Put the client ID in `RAGDESK_GITHUB_CLIENT_ID` (or the app's `.env`).
Device flow uses **no client secret**.
</details>

## Confluence

The default path is an **API token** — no app registration required.

1. Create a token at
   [id.atlassian.com → API tokens](https://id.atlassian.com/manage-profile/security/api-tokens).
2. Sources → Confluence → enter your site URL (`https://team.atlassian.net`),
   the email you use for Atlassian, and the token → **Connect with API token**.
3. Enter a space key (e.g. `DOCS`) → **Sync space**.

CLI: `ragdesk confluence DOCS --base-url https://team.atlassian.net`
(email/token from `CONFLUENCE_EMAIL` / `CONFLUENCE_TOKEN` or the saved card).

<details>
<summary>Atlassian OAuth (your own 3LO app — browser consent instead of a token)</summary>

1. `developer.atlassian.com` → Developer console → **Create app** →
   OAuth 2.0 integration. Name it, choose **Resource-level** access, accept terms.
2. **Permissions** → Confluence API → add:
   `read:confluence-content.all`, `read:confluence-space.summary`,
   `read:confluence-user`.
3. **Authorization** → Callback URL:
   `http://127.0.0.1:8788/callback`
4. **Settings** → copy Client ID + Secret.
5. In the app: Confluence → **Use Atlassian OAuth (your own app)** → paste both
   → **Save & connect**.

**The Atlassian secret is a real credential** — it is stored only on your
machine (`~/.config/ragdesk/credentials.json`, 0600) and is never shipped in
this repository or in release builds (the build baker refuses it on purpose).
The app is private by default: only the Atlassian account that created it can
authorize it until you enable *Distribution → Sharing*.
</details>

## Google Drive

Reads Google Docs/Sheets/Slides (exported to text) and text files. Nothing is
uploaded or modified.

1. Sources → Google Drive → **Connect Google Drive**.
2. The browser asks for consent. If you see *"Google hasn't verified this
   app"*, choose **Advanced → Go to ragdesk (unsafe)** — expected for an
   unverified app.
3. Optionally paste a folder ID (from
   `drive.google.com/drive/folders/<ID>`) → **Sync Drive**.
   ⚠️ Leaving the folder empty syncs **your entire Drive**.

CLI: `ragdesk gdrive --folder-id <ID>`

<details>
<summary>BYO OAuth client (or how the shipped one was created)</summary>

1. `console.cloud.google.com` → project → **APIs & Services → Library** →
   enable **Google Drive API**.
2. **Google Auth Platform → Branding**: app name, support email, developer
   contact, **home page URL and privacy policy URL** (both are required to
   publish; any page on a domain you control works while unverified), then add
   the domain under **Authorized domains**.
3. **Audience** → **Publish app** (confirm; skip verification — you stay
   "In production, unverified"). This matters: tokens minted while the app is
   in *Testing* expire after 7 days.
4. **Clients** → **Create client** → **Desktop app** → copy Client ID + Secret.
5. Put them in `GDRIVE_CLIENT_ID` / `GDRIVE_CLIENT_SECRET` (or `.env`).

Notes: unverified apps show a warning and are capped at 100 users over the
project's lifetime — fine for personal use and small teams. Desktop-app
client secrets are documented by Google as **not confidential** (installed
apps cannot keep secrets; PKCE protects the flow), which is why the build may
bake them into release artifacts.
</details>

## GitLab

Read-only repository sync (archive download, incremental by archive digest).

1. Create a personal access token with the **`read_api`** scope
   (GitLab → *Preferences → Access tokens*).
2. Sources → GitLab → paste the token (and a base URL if self-hosted) →
   **Connect GitLab**.
3. Enter `group/name` (optionally ref/subfolder) → **Sync repo**.

CLI: `ragdesk gitlab group/name --ref main --base-url https://gitlab.example.com`

## Microsoft (OneDrive & SharePoint)

Read-only sync of text files; `.docx` and `.pptx` are converted to text
locally (no upload, no conversion service).

1. **Azure app registration** (once): Azure Portal → *Microsoft Entra ID →
   App registrations → New registration* (any account type) → copy the
   **Application (client) ID**.
2. *Authentication* → **Allow public client flows: Yes** — the device flow
   needs this.
3. *API permissions* → Microsoft Graph → **delegated** `Files.Read.All` and
   `Sites.Read.All` (consent happens in the browser on first connect;
   `offline_access` is implicit).
4. Save the client ID: `RAGDESK_MS_CLIENT_ID` in `.env`, or paste it once in
   the app card.
5. Sources → Microsoft → **Connect Microsoft** → open
   `microsoft.com/devicelogin`, enter the code, approve.
6. **Sync Microsoft files**: leave both fields empty for all of OneDrive, or
   give a OneDrive folder ID / SharePoint site (`contoso.sharepoint.com:/sites/Team`).

CLI: `ragdesk msgraph --folder-id <ID>` (or `--site hostname:/sites/x`).

## Notion

Indexes the pages you explicitly share with an integration — Notion never
exposes anything else.

1. Create an internal integration at
   [notion.so/my-integrations](https://www.notion.so/my-integrations) and copy
   its token (`ntn_…`).
2. **Share pages**: open each page (or a parent page) in Notion → `•••` →
   *Connections* → add your integration. Child pages inherit the share.
3. Sources → Notion → paste the token → **Connect Notion** → **Sync pages**.

Re-syncs skip pages whose `last_edited_time` is unchanged, so only edits are
re-fetched. CLI: `ragdesk notion` (or `NOTION_TOKEN=… ragdesk notion`).

## Website (web crawl)

Index a docs site: same host, HTML only, capped pages and depth, no
JavaScript rendering.

- **Desktop app**: Sources → Website → URL + max pages + depth → **Crawl site**.
- **CLI**: `ragdesk web https://docs.example.com/ --max-pages 50 --depth 2`

## Troubleshooting

| Symptom | Fix |
|---|---|
| `GitHub rejected the request (HTTP 404)` on device login | The client ID is wrong or device flow isn't enabled for the app. |
| `device_flow_disabled` | Tick **Enable Device Flow** in the GitHub app settings. |
| Confluence `401/403` | API tokens are per-user; re-create the token and check the email matches. |
| Google warning screen | Expected while unverified: **Advanced → Go to ragdesk (unsafe)**. |
| Google token stops working after ~7 days | The consent screen is in *Testing*; publish it to production (see above). |
| Notion sync finds `0 pages` | The pages aren't shared with the integration — add the connection on each page. |
| GitLab `401/403` | The token is missing the `read_api` scope, or it expired. |
| Microsoft: *"AADSTS7000218: … public client flows"* | Enable **Allow public client flows** in the app registration (Authentication). |
| Microsoft: consent screen says *need admin approval* | Your tenant blocks user consent; an admin must approve the app (or use a personal account). |
| Something shows *"local server is not reachable"* | The Python server restarted; the app relaunches it within ~3 s, retry. If it repeats, check `~/.ragdesk/serve.log`. |
| `not HTML (...)` during crawl | The link points at a binary/JS-only asset; those are skipped by design. |
