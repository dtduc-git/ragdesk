const API = "http://127.0.0.1:8765";
const REFUSAL = "I could not find this in your indexed sources.";
const SUGGESTIONS = [
  "how does RRF fusion rank results?",
  "what does the grounding gate do?",
  "how do I install the onnx extra?",
];

type Hit = {
  path: string;
  source: string;
  ordinal: number;
  text: string;
  score: number;
  cosine: number;
  lanes: string;
};

type SourceStat = { source: string; documents: number; chunks: number; indexed_at: string };

type Status = {
  version: string;
  db: string;
  embedder: { name: string | null; dim: string | null };
  rerank: string;
  llm_model: string;
  preset: string;
  documents: number;
  chunks: number;
  sources: SourceStat[];
};

function $<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`missing element #${id}`);
  return element as T;
}

function escapeHtml(value: string): string {
  const replacements: Record<string, string> = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  };
  return value.replace(/[&<>"']/g, (character) => replacements[character] ?? character);
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${API}${path}`);
  const data: unknown = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error((data as { error?: string }).error ?? `HTTP ${response.status}`);
  }
  return data as T;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data: unknown = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error((data as { error?: string }).error ?? `HTTP ${response.status}`);
  }
  return data as T;
}

// --- toast --------------------------------------------------------------------

let toastTimer: number | undefined;

function toast(message: string): void {
  const element = $("toast");
  element.textContent = message;
  element.hidden = false;
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    element.hidden = true;
  }, 6000);
}

// --- tabs ---------------------------------------------------------------------

function activateTab(tab: string): void {
  document.querySelectorAll<HTMLElement>(".nav-item").forEach((item) => {
    item.classList.toggle("is-active", item.dataset.tab === tab);
  });
  document.querySelectorAll<HTMLElement>(".panel").forEach((panel) => {
    panel.classList.toggle("is-active", panel.id === `panel-${tab}`);
  });
}

document.querySelectorAll<HTMLButtonElement>(".nav-item").forEach((item) => {
  item.addEventListener("click", () => activateTab(item.dataset.tab ?? "chat"));
});

// --- status -------------------------------------------------------------------

let status: Status | null = null;

async function loadStatus(): Promise<void> {
  try {
    status = await get<Status>("/api/status");
    renderStatus();
  } catch {
    status = null;
    $("rail-meta").textContent = "server offline — is the indexer running?";
  }
}

function renderStatus(): void {
  if (!status) return;
  $("stat-docs").textContent = String(status.documents);
  $("stat-chunks").textContent = String(status.chunks);
  $("rail-meta").textContent = `${status.preset} preset · ${status.llm_model}`;

  const rows: Array<[string, string]> = [
    ["version", status.version],
    ["database", status.db],
    ["embedder", status.embedder.name ?? "not set"],
    ["rerank", status.rerank],
    ["model", status.llm_model],
    ["preset", status.preset],
  ];
  $("status-table").innerHTML = rows
    .map(
      ([key, value]) =>
        `<div class="status-row"><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd></div>`,
    )
    .join("");

  const table = $("indexed-table");
  if (status.sources.length === 0) {
    table.innerHTML = `<p class="muted">Nothing indexed yet. Add a source.</p>`;
    return;
  }
  const totalDocs = status.sources.reduce((sum, entry) => sum + entry.documents, 0);
  const totalChunks = status.sources.reduce((sum, entry) => sum + entry.chunks, 0);
  table.innerHTML = `
    <div class="stats-row stats-head">
      <span>Source</span><span>Documents</span><span>Chunks</span><span>Last indexed</span>
    </div>
    ${status.sources
      .map(
        (entry) => `<div class="stats-row">
          <span class="stats-source">${escapeHtml(entry.source)}</span>
          <span>${entry.documents}</span>
          <span>${entry.chunks}</span>
          <span>${escapeHtml((entry.indexed_at || "").slice(0, 16))}</span>
        </div>`,
      )
      .join("")}
    <div class="stats-row stats-total">
      <span>Total</span><span>${totalDocs}</span><span>${totalChunks}</span><span></span>
    </div>`;
}

// --- citations ----------------------------------------------------------------

function snippet(text: string, length: number): string {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > length ? `${flat.slice(0, length - 1)}…` : flat;
}

function citeCard(hit: Hit, rank: number): string {
  const openable = hit.source === "local";
  const pathTag = openable
    ? `<code class="cite-path" data-open-path="${escapeHtml(hit.path)}" title="Open file">${escapeHtml(hit.path)}</code>`
    : `<code class="cite-path">${escapeHtml(hit.path)}</code>`;
  return `<article class="cite">
    <span class="cite-rank">[${rank}]</span>
    <div class="cite-body">
      <div class="cite-top">
        ${pathTag}
        <span class="cite-score">${hit.cosine.toFixed(2)} · ${escapeHtml(hit.lanes)}</span>
      </div>
      <p class="cite-snippet">${escapeHtml(snippet(hit.text, 240))}</p>
    </div>
  </article>`;
}

async function openLocal(path: string): Promise<void> {
  try {
    const { openPath } = await import("@tauri-apps/plugin-opener");
    await openPath(path);
  } catch {
    // browser mode: opening local files is a desktop affordance
  }
}

document.addEventListener("click", (event) => {
  const target = event.target as HTMLElement;
  const path = target.closest<HTMLElement>("[data-open-path]")?.dataset.openPath;
  if (path) void openLocal(path);
});

// --- chat ---------------------------------------------------------------------

let streaming = false;

function appendUserMessage(query: string): void {
  $("chat-empty")?.remove();
  const message = document.createElement("div");
  message.className = "msg msg-user";
  message.textContent = query;
  $("chat-log").append(message);
}

function appendAssistantShell(): { answer: HTMLElement; cites: HTMLElement } {
  const message = document.createElement("div");
  message.className = "msg msg-assistant";
  message.innerHTML = `<div class="answer streaming"></div><div class="cites"></div>`;
  $("chat-log").append(message);
  message.scrollIntoView({ block: "end" });
  return {
    answer: message.querySelector<HTMLElement>(".answer") as HTMLElement,
    cites: message.querySelector<HTMLElement>(".cites") as HTMLElement,
  };
}

async function ask(query: string): Promise<void> {
  if (streaming || !status) {
    if (!status) toast("The local indexer is not reachable.");
    return;
  }
  streaming = true;
  $<HTMLButtonElement>("chat-send").disabled = true;
  appendUserMessage(query);
  const { answer, cites } = appendAssistantShell();

  try {
    const response = await fetch(`${API}/api/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 6 }),
    });
    if (!response.ok || !response.body) {
      const data: unknown = await response.json().catch(() => ({}));
      throw new Error((data as { error?: string }).error ?? `HTTP ${response.status}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finished = false;

    while (!finished) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline = buffer.indexOf("\n");
      while (newline >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (line) {
          const event = JSON.parse(line) as {
            delta?: string;
            done?: boolean;
            error?: string;
            hits?: Hit[];
          };
          if (event.delta) {
            answer.textContent += event.delta;
            answer.scrollIntoView({ block: "end" });
          }
          if (event.error) throw new Error(event.error);
          if (event.done) {
            finished = true;
            answer.classList.remove("streaming");
            answer.classList.toggle("is-refused", answer.textContent === REFUSAL);
            renderCites(cites, event.hits ?? []);
          }
        }
        newline = buffer.indexOf("\n");
      }
    }
  } catch (error) {
    answer.classList.remove("streaming");
    answer.classList.add("is-error");
    answer.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    streaming = false;
    $<HTMLButtonElement>("chat-send").disabled = false;
  }
}

function renderCites(container: HTMLElement, hits: Hit[]): void {
  if (hits.length === 0) return;
  container.innerHTML = `<p class="cites-label">${hits.length} source${hits.length === 1 ? "" : "s"}</p>${hits
    .map((hit, index) => citeCard(hit, index + 1))
    .join("")}`;
  container.querySelectorAll(".cite").forEach((card, index) => {
    (card as HTMLElement).style.setProperty("--stagger", `${index * 60}ms`);
  });
}

$<HTMLFormElement>("chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const query = $<HTMLTextAreaElement>("chat-input").value.trim();
  if (query) {
    $<HTMLTextAreaElement>("chat-input").value = "";
    void ask(query);
  }
});

$<HTMLTextAreaElement>("chat-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $<HTMLFormElement>("chat-form").requestSubmit();
  }
});

for (const suggestion of SUGGESTIONS) {
  const chip = document.createElement("button");
  chip.type = "button";
  chip.className = "chip";
  chip.textContent = suggestion;
  chip.addEventListener("click", () => void ask(suggestion));
  $("chat-suggestions").append(chip);
}

// --- search -------------------------------------------------------------------

$<HTMLFormElement>("search-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = $<HTMLInputElement>("search-input").value.trim();
  const results = $("search-results");
  if (!query) return;
  results.innerHTML = `<p class="muted">Searching…</p>`;
  try {
    const payload = await post<{ hits: Hit[] }>("/api/search", { query, top_k: 8 });
    if (payload.hits.length === 0) {
      results.innerHTML = `<p class="muted">Nothing matched. Try other words, or index more sources.</p>`;
      return;
    }
    results.innerHTML = payload.hits.map((hit, index) => citeCard(hit, index + 1)).join("");
  } catch (error) {
    results.innerHTML = `<p class="muted">${escapeHtml(error instanceof Error ? error.message : String(error))}</p>`;
  }
});

// --- sources ------------------------------------------------------------------

type Connections = {
  github: {
    connected: boolean;
    source: string | null;
    login: string;
    gh_available: boolean;
    device_flow_ready: boolean;
  };
  confluence: {
    connected: boolean;
    source: string | null;
    base_url: string;
    email: string;
    display_name: string;
    oauth_ready: boolean;
    oauth_connected: boolean;
    site_name: string;
    site_url: string;
  };
  gdrive: { connected: boolean; email: string; oauth_ready: boolean };
  notion: { connected: boolean; name: string };
};

type DeviceFlow = { userCode: string; verificationUri: string; interval: number };

let connections: Connections | null = null;
let selectedPaths: string[] = [];
let deviceFlow: DeviceFlow | null = null;
let deviceTimer: number | undefined;
const reveal: Record<string, boolean> = {};
const sourceResults: Record<string, string> = {};

const GITHUB_SOURCE_LABEL: Record<string, string> = {
  credentials: "saved token",
  env: "GITHUB_TOKEN env",
  gh: "gh CLI login",
  explicit: "token",
};

function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

async function openExternal(url: string): Promise<void> {
  if (isTauri()) {
    try {
      const { openUrl } = await import("@tauri-apps/plugin-opener");
      await openUrl(url);
      return;
    } catch {
      // fall through to a browser tab
    }
  }
  window.open(url, "_blank", "noopener");
}

function cancelDeviceFlow(): void {
  deviceFlow = null;
  window.clearTimeout(deviceTimer);
}

function startDevicePolling(): void {
  window.clearTimeout(deviceTimer);
  if (!deviceFlow) return;
  deviceTimer = window.setTimeout(async () => {
    if (!deviceFlow) return;
    try {
      const response = await post<{
        connected?: boolean;
        pending?: boolean;
        login?: string;
        interval?: number;
      }>("/api/connections/github/device/poll", {});
      if (response.connected) {
        const login = response.login ?? "github";
        cancelDeviceFlow();
        setResult("github", `connected — ${login}`);
        await loadConnections();
        return;
      }
      if (response.interval) deviceFlow.interval = response.interval;
      startDevicePolling();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      cancelDeviceFlow();
      setResult("github", message);
      toast(message);
      renderSources();
    }
  }, Math.max(deviceFlow.interval, 3) * 1000);
}

function setResult(kind: string, message: string): void {
  sourceResults[kind] = message;
  const element = document.querySelector<HTMLElement>(`[data-result="${kind}"]`);
  if (element) element.textContent = message;
}

function summarize(payload: Record<string, unknown>): string {
  const parts = ["scanned", "indexed", "unchanged", "skipped", "chunks"]
    .filter((key) => typeof payload[key] === "number")
    .map((key) => `${key} ${payload[key]}`);
  return parts.length ? parts.join(" · ") : "done";
}

async function loadConnections(): Promise<void> {
  try {
    connections = await get<Connections>("/api/connections");
  } catch {
    connections = null;
  }
  renderSources();
}

async function pickPaths(kind: "folder" | "files"): Promise<void> {
  try {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const selection = await open({
      directory: kind === "folder",
      multiple: kind === "files",
      title: kind === "folder" ? "Choose a folder to index" : "Choose files to index",
    });
    if (!selection) return;
    for (const path of Array.isArray(selection) ? selection : [selection]) {
      if (!selectedPaths.includes(path)) selectedPaths.push(path);
    }
    renderSources();
  } catch {
    toast("Folder picking works in the desktop app; paste a path instead.");
  }
}

function localCard(): string {
  const chips = selectedPaths
    .map(
      (path) =>
        `<span class="path-chip"><code>${escapeHtml(path)}</code><button type="button" class="chip-x" data-action="remove-path" data-path="${escapeHtml(path)}" aria-label="Remove path">×</button></span>`,
    )
    .join("");
  const picker = isTauri()
    ? `<div class="button-row">
         <button class="btn" type="button" data-action="pick-folder">Choose folder…</button>
         <button class="btn" type="button" data-action="pick-files">Choose files…</button>
       </div>`
    : "";
  const count = selectedPaths.length;
  return `<div class="source-card">
    <h3>Local files</h3>
    <p class="source-note">Pick folders or files — everything is read, never modified.</p>
    <div class="path-list">${chips || `<p class="muted">Nothing selected yet.</p>`}</div>
    ${picker}
    <form data-form="local-add" class="field-row">
      <input name="paths" placeholder="…or paste a path, comma-separated" />
      <button class="btn" type="submit">Add</button>
    </form>
    <form data-form="local-index">
      <button class="btn btn-primary" type="submit" ${count ? "" : "disabled"}>
        Index ${count ? `${count} path${count === 1 ? "" : "s"}` : "paths"}
      </button>
    </form>
    <p class="source-result" data-result="local"></p>
  </div>`;
}

async function startGithubDevice(): Promise<void> {
  setResult("github", "requesting a code…");
  try {
    const response = await post<{
      user_code: string;
      verification_uri: string;
      interval: number;
    }>("/api/connections/github/device/start", {});
    deviceFlow = {
      userCode: response.user_code,
      verificationUri: response.verification_uri,
      interval: response.interval || 5,
    };
    setResult("github", "enter the code shown in your browser");
    renderSources();
    void openExternal(deviceFlow.verificationUri);
    startDevicePolling();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setResult("github", message);
    toast(message);
  }
}

function githubCard(conn: Connections["github"]): string {
  if (!conn.connected) {
    let primary: string;
    if (deviceFlow) {
      primary = `<div class="device-box">
        <p class="source-note">Enter this code on GitHub:</p>
        <p class="device-code">${escapeHtml(deviceFlow.userCode)}</p>
        <div class="button-row">
          <button class="btn" type="button" data-action="open-device-page">Open GitHub</button>
          <button class="btn btn-quiet" type="button" data-action="github-device-cancel">Cancel</button>
        </div>
        <p class="muted">Waiting for approval…</p>
      </div>`;
    } else if (conn.device_flow_ready) {
      primary = `<button class="btn btn-primary" type="button" data-action="github-device-start">Connect with a code</button>`;
    } else {
      primary = `<button class="btn btn-primary" type="button" data-action="reveal-github">Connect with a code</button>`;
    }
    const setup =
      !deviceFlow && !conn.device_flow_ready && reveal.github
        ? `<form data-form="github-client-id" class="stack">
             <p class="source-note">Create a GitHub OAuth app with device flow enabled, then paste its client ID once.</p>
             <input name="client_id" placeholder="OAuth client ID" required />
             <button class="btn" type="submit">Save &amp; get a code</button>
           </form>`
        : "";
    const gh =
      conn.gh_available && !deviceFlow
        ? `<button class="btn btn-quiet" type="button" data-action="github-gh">Use gh login instead</button>`
        : "";
    return `<div class="source-card">
      <h3>GitHub</h3>
      <p class="source-note">One click: GitHub shows a code, you approve in the browser, done.</p>
      ${primary}
      ${setup}
      ${gh}
      <details class="alt">
        <summary>Use a personal access token</summary>
        <form data-form="github-connect" class="stack">
          <input name="token" type="password" placeholder="personal access token" />
          <button class="btn" type="submit">Connect with token</button>
        </form>
      </details>
      <p class="source-result" data-result="github"></p>
    </div>`;
  }
  const label = GITHUB_SOURCE_LABEL[conn.source ?? ""] ?? conn.source ?? "token";
  return `<div class="source-card is-connected">
    <h3>GitHub <span class="conn-badge">connected</span></h3>
    <p class="source-note">${conn.login ? escapeHtml(conn.login) : "token"} · via ${escapeHtml(label)}</p>
    <div class="button-row">
      <button class="btn btn-quiet" type="button" data-action="disconnect-github">Disconnect</button>
    </div>
    <form data-form="github-sync" class="stack">
      <input name="repo" placeholder="owner/name" required />
      <div class="field-row">
        <input name="ref" placeholder="branch (optional)" />
        <input name="subdir" placeholder="subfolder (optional)" />
      </div>
      <button class="btn btn-primary" type="submit">Sync repo</button>
    </form>
    <p class="source-result" data-result="github"></p>
  </div>`;
}

function confluenceCard(conn: Connections["confluence"]): string {
  if (!conn.connected) {
    const tokenForm = `<form data-form="confluence-connect" class="stack">
      <div class="field-row">
        <input name="base_url" placeholder="https://team.atlassian.net" required />
        <input name="email" placeholder="you@company.com" required />
      </div>
      <input name="token" type="password" placeholder="API token" required />
      <button class="btn btn-primary" type="submit">Connect with API token</button>
      <p class="source-note">
        Create one at
        <button class="link-like" type="button" data-action="open-atlassian-tokens">id.atlassian.com — API tokens</button>.
        No app registration needed.
      </p>
    </form>`;

    const oauthSetup = `<form data-form="confluence-oauth" class="stack">
      <p class="source-note">
        Create an OAuth 2.0 (3LO) app with redirect URL
        <code>http://127.0.0.1:8788/callback</code> —
        <button class="link-like" type="button" data-action="open-atlassian-console">open the developer console</button>
        — then paste its credentials. Stored locally, never in git.
      </p>
      <input name="client_id" placeholder="Atlassian client ID" required />
      <input name="client_secret" type="password" placeholder="Atlassian client secret" required />
      <button class="btn" type="submit">Save &amp; connect</button>
    </form>`;

    if (conn.oauth_ready) {
      return `<div class="source-card">
        <h3>Confluence</h3>
        <p class="source-note">One click: Atlassian asks for consent in your browser.</p>
        <button class="btn btn-primary" type="button" data-action="confluence-oauth-start">Connect with Atlassian</button>
        <details class="alt">
          <summary>Connect with a site URL + API token</summary>
          ${tokenForm}
        </details>
        <p class="source-result" data-result="confluence"></p>
      </div>`;
    }
    return `<div class="source-card">
      <h3>Confluence</h3>
      <p class="source-note">Use an API token (works for everyone), or bring your own Atlassian OAuth app for one-click consent.</p>
      ${tokenForm}
      <details class="alt">
        <summary>Use Atlassian OAuth (your own app)</summary>
        ${oauthSetup}
      </details>
      <p class="source-result" data-result="confluence"></p>
    </div>`;
  }
  const who = conn.site_name || conn.display_name || conn.email;
  const where = conn.site_url || conn.base_url;
  const via = conn.oauth_connected
    ? "Atlassian OAuth"
    : conn.source === "env"
      ? "env vars"
      : "API token";
  return `<div class="source-card is-connected">
    <h3>Confluence <span class="conn-badge">connected</span></h3>
    <p class="source-note">${escapeHtml(who)} · ${escapeHtml(where)} · ${via}</p>
    <div class="button-row">
      <button class="btn btn-quiet" type="button" data-action="disconnect-confluence">Disconnect</button>
    </div>
    <form data-form="confluence-sync" class="stack">
      <input name="space" placeholder="space key, e.g. DOCS" required />
      <button class="btn btn-primary" type="submit">Sync space</button>
    </form>
    <p class="source-result" data-result="confluence"></p>
  </div>`;
}

function gdriveCard(conn: Connections["gdrive"]): string {
  if (!conn.connected) {
    const primary = conn.oauth_ready
      ? `<button class="btn btn-primary" type="button" data-action="gdrive-oauth-start">Connect Google Drive</button>`
      : `<button class="btn btn-primary" type="button" data-action="reveal-gdrive">Connect Google Drive</button>`;
    const setup =
      !conn.oauth_ready && reveal.gdrive
        ? `<form data-form="gdrive-connect" class="stack">
             <p class="source-note">Create a Google OAuth client (type “Desktop app”) and paste its ID once.</p>
             <input name="client_id" placeholder="OAuth client ID" required />
             <input name="client_secret" type="password" placeholder="client secret (optional)" />
             <button class="btn" type="submit">Save &amp; connect</button>
           </form>`
        : "";
    return `<div class="source-card">
      <h3>Google Drive</h3>
      <p class="source-note">One click: Google asks for consent in your browser.</p>
      ${primary}
      ${setup}
      <p class="source-result" data-result="gdrive"></p>
    </div>`;
  }
  return `<div class="source-card is-connected">
    <h3>Google Drive <span class="conn-badge">connected</span></h3>
    <p class="source-note">${conn.email ? escapeHtml(conn.email) : "consent saved"}</p>
    <div class="button-row">
      <button class="btn btn-quiet" type="button" data-action="disconnect-gdrive">Disconnect</button>
    </div>
    <form data-form="gdrive-sync" class="stack">
      <input name="folder_id" placeholder="folder id (optional — all files by default)" />
      <button class="btn btn-primary" type="submit">Sync Drive</button>
    </form>
    <p class="source-result" data-result="gdrive"></p>
  </div>`;
}

function notionCard(conn: Connections["notion"]): string {
  if (!conn.connected) {
    return `<div class="source-card">
      <h3>Notion</h3>
      <p class="source-note">
        Create an internal integration at
        <button class="link-like" type="button" data-action="open-notion-integrations">notion.so/my-integrations</button>
        and share the pages with it — Notion exposes only what you share.
      </p>
      <form data-form="notion-connect" class="stack">
        <input name="token" type="password" placeholder="integration token (ntn_…)" required />
        <button class="btn btn-primary" type="submit">Connect Notion</button>
      </form>
      <p class="source-result" data-result="notion"></p>
    </div>`;
  }
  return `<div class="source-card is-connected">
    <h3>Notion <span class="conn-badge">connected</span></h3>
    <p class="source-note">${conn.name ? escapeHtml(conn.name) : "integration"} · only shared pages are indexed</p>
    <div class="button-row">
      <button class="btn btn-quiet" type="button" data-action="disconnect-notion">Disconnect</button>
    </div>
    <form data-form="notion-sync" class="stack">
      <button class="btn btn-primary" type="submit">Sync pages</button>
    </form>
    <p class="source-result" data-result="notion"></p>
  </div>`;
}

function webCard(): string {
  return `<div class="source-card">
    <h3>Website</h3>
    <p class="source-note">Crawl a docs site: same host, HTML only, capped pages. Read-only.</p>
    <form data-form="web-sync" class="stack">
      <input name="url" placeholder="https://docs.example.com/start" required />
      <div class="field-row">
        <input name="max_pages" type="number" min="1" max="500" placeholder="max pages (50)" />
        <input name="max_depth" type="number" min="0" max="5" placeholder="depth (2)" />
      </div>
      <button class="btn btn-primary" type="submit">Crawl site</button>
    </form>
    <p class="source-result" data-result="web"></p>
  </div>`;
}

function renderSources(): void {
  const github = connections?.github ?? {
    connected: false,
    source: null,
    login: "",
    gh_available: false,
    device_flow_ready: false,
  };
  const confluence = connections?.confluence ?? {
    connected: false,
    source: null,
    base_url: "",
    email: "",
    display_name: "",
    oauth_ready: false,
    oauth_connected: false,
    site_name: "",
    site_url: "",
  };
  const gdrive = connections?.gdrive ?? { connected: false, email: "", oauth_ready: false };
  const notion = connections?.notion ?? { connected: false, name: "" };
  $("source-grid").innerHTML =
    localCard() +
    githubCard(github) +
    confluenceCard(confluence) +
    gdriveCard(gdrive) +
    notionCard(notion) +
    webCard();
  for (const [kind, message] of Object.entries(sourceResults)) {
    const element = document.querySelector<HTMLElement>(`[data-result="${kind}"]`);
    if (element) element.textContent = message;
  }
}

$("source-grid").addEventListener("submit", async (event) => {
  const form = event.target as HTMLFormElement;
  if (!(form instanceof HTMLFormElement) || !form.dataset.form) return;
  event.preventDefault();
  const kind = form.dataset.form;
  const provider = kind.split("-")[0];
  const payload: Record<string, unknown> = {};
  new FormData(form).forEach((value, key) => {
    if (String(value).trim()) payload[key] = String(value).trim();
  });
  try {
    if (kind === "local-add") {
      for (const part of String(payload.paths ?? "")
        .split(",")
        .map((path) => path.trim())
        .filter(Boolean)) {
        if (!selectedPaths.includes(part)) selectedPaths.push(part);
      }
      renderSources();
      return;
    }
    if (kind === "local-index") {
      if (selectedPaths.length === 0) {
        setResult("local", "Choose at least one path first.");
        return;
      }
      setResult("local", "working…");
      const response = await post<Record<string, number>>("/api/index", {
        paths: selectedPaths,
      });
      selectedPaths = [];
      setResult("local", summarize(response));
      renderSources();
      await loadStatus();
      return;
    }
    if (kind === "github-client-id") {
      if (!payload.client_id) {
        setResult("github", "paste an OAuth client ID first");
        return;
      }
      await post("/api/connections/github/client-id", { client_id: payload.client_id });
      await loadConnections();
      void startGithubDevice();
      return;
    }
    const endpoints: Record<string, string> = {
      "github-connect": "/api/connections/github",
      "github-sync": "/api/sync/github",
      "confluence-connect": "/api/connections/confluence",
      "confluence-oauth": "/api/connections/confluence/oauth",
      "confluence-sync": "/api/sync/confluence",
      "gdrive-connect": "/api/connections/gdrive",
      "gdrive-sync": "/api/sync/gdrive",
      "notion-connect": "/api/connections/notion",
      "notion-sync": "/api/sync/notion",
      "web-sync": "/api/sync/web",
    };
    const endpoint = endpoints[kind];
    if (!endpoint) return;
    setResult(provider, "working…");
    const response = await post<Record<string, unknown>>(endpoint, payload);
    if (kind.endsWith("-connect") || kind.endsWith("-oauth")) {
      const detail = String(response.email ?? response.display_name ?? response.login ?? "");
      setResult(provider, `connected${detail ? ` — ${detail}` : ""}`);
      await loadConnections();
    } else {
      setResult(provider, summarize(response));
      await loadStatus();
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setResult(provider, message);
    toast(message);
  }
});

$("source-grid").addEventListener("click", async (event) => {
  const target = (event.target as HTMLElement).closest<HTMLElement>("[data-action]");
  if (!target) return;
  const action = target.dataset.action ?? "";
  if (action === "pick-folder") {
    void pickPaths("folder");
    return;
  }
  if (action === "pick-files") {
    void pickPaths("files");
    return;
  }
  if (action === "remove-path") {
    selectedPaths = selectedPaths.filter((path) => path !== target.dataset.path);
    renderSources();
    return;
  }
  if (action.startsWith("disconnect-")) {
    const provider = action.split("-")[1];
    if (provider === "github") cancelDeviceFlow();
    setResult(provider, "disconnecting…");
    try {
      await post(`/api/connections/${provider}/disconnect`, {});
      await loadConnections();
      if (provider === "github" && connections?.github.connected) {
        const label =
          GITHUB_SOURCE_LABEL[connections.github.source ?? ""] ??
          connections.github.source ??
          "an ambient source";
        setResult("github", `saved token removed — still connected via ${label}`);
      } else if (provider === "confluence" && connections?.confluence.connected) {
        setResult("confluence", "saved credentials removed — still connected via env vars");
      } else {
        setResult(provider, "disconnected");
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setResult(provider, message);
      toast(message);
    }
    return;
  }
  if (action === "github-device-start") {
    void startGithubDevice();
    return;
  }
  if (action === "reveal-github" || action === "reveal-gdrive") {
    reveal[action.replace("reveal-", "")] = true;
    renderSources();
    return;
  }
  if (action === "open-atlassian-tokens") {
    void openExternal("https://id.atlassian.com/manage-profile/security/api-tokens");
    return;
  }
  if (action === "open-atlassian-console") {
    void openExternal("https://developer.atlassian.com/console/myapps/");
    return;
  }
  if (action === "open-notion-integrations") {
    void openExternal("https://www.notion.so/my-integrations");
    return;
  }
  if (action === "gdrive-oauth-start" || action === "confluence-oauth-start") {
    const provider = action.startsWith("gdrive") ? "gdrive" : "confluence";
    const endpoint =
      provider === "gdrive"
        ? "/api/connections/gdrive"
        : "/api/connections/confluence/oauth";
    setResult(provider, "waiting for browser consent…");
    try {
      const response = await post<Record<string, unknown>>(endpoint, {});
      const detail = String(response.email ?? response.display_name ?? "");
      setResult(provider, `connected${detail ? ` — ${detail}` : ""}`);
      await loadConnections();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setResult(provider, message);
      toast(message);
    }
    return;
  }
  if (action === "github-device-cancel") {
    cancelDeviceFlow();
    renderSources();
    return;
  }
  if (action === "open-device-page") {
    if (deviceFlow) void openExternal(deviceFlow.verificationUri);
    return;
  }
  if (action === "github-gh") {
    setResult("github", "using gh login…");
    try {
      const response = await post<{ login?: string }>("/api/connections/github/gh", {});
      setResult("github", `connected — ${response.login ?? "gh"}`);
      await loadConnections();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setResult("github", message);
      toast(message);
    }
  }
});

// --- boot ---------------------------------------------------------------------

async function boot(): Promise<void> {
  // The desktop shell may still be starting the Python server: retry briefly.
  for (let attempt = 0; attempt < 25; attempt += 1) {
    try {
      status = await get<Status>("/api/status");
      break;
    } catch {
      await new Promise((resolve) => window.setTimeout(resolve, 400));
    }
  }
  if (status) {
    renderStatus();
    await loadConnections();
  } else {
    $("rail-meta").textContent = "server offline — start it with: ragdesk serve";
  }
}

void boot();

export {};
