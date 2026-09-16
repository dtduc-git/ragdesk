const API = window.location.protocol.startsWith("http")
  ? window.location.origin
  : "http://127.0.0.1:8765";
const REFUSAL = "I could not find this in your indexed sources.";
const SUGGESTIONS = [
  "how does RRF fusion rank results?",
  "what does the grounding gate do?",
  "how do I install the onnx extra?",
];

type Hit = {
  path: string;
  source?: string;
  ordinal?: number;
  text: string;
  score?: number;
  cosine?: number;
  lanes?: string;
  line?: number;
  metadata?: Record<string, string>;
};

type SourceStat = { source: string; documents: number; chunks: number; indexed_at: string };

type PathStat = { path: string; documents: number; chunks: number; indexed_at: string };

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
  local_paths: PathStat[];
  auto_index: { hours: number; last_run: string };
  watch_seconds: number;
  vaults: string[];
  memory: { models_loaded: boolean; idle_unload_minutes: number };
  hyde: boolean;
  notes_available: boolean;
  answer_length: string;
  embed_threads: number;
  activity: {
    running: boolean;
    kind: string;
    detail: string;
    done: number;
    total: number;
    started: number;
  };
  presets: Array<{ name: string; note: string; rerank: string; llm: string }>;
  llm: { kind: string; model: string; note: string };
  llm_setting: {
    preference: string;
    openai_host: string;
    openai_model: string;
    openai_key_set: boolean;
  };
  onboarded: boolean;
  system: {
    ram_gb: number;
    platform: string;
    suggested_preset: string;
    screenshot_dirs: string[];
  };
  llm_setup: {
    ollama_model: string;
    ollama_reachable: boolean;
    ollama_has_model: boolean;
    mlx_available: boolean;
    mlx_repo: string;
    mlx_cached: boolean;
    job: {
      running: boolean;
      kind: string;
      model: string;
      progress: number;
      detail: string;
      error: string;
    };
  };
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

function explainFetch(error: unknown): Error {
  if (error instanceof TypeError) {
    return new Error(
      "ragdesk's local server is not reachable right now — the app restarts it within a few seconds, retry",
    );
  }
  return error instanceof Error ? error : new Error(String(error));
}

async function get<T>(path: string): Promise<T> {
  try {
    const response = await fetch(`${API}${path}`);
    const data: unknown = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error((data as { error?: string }).error ?? `HTTP ${response.status}`);
    }
    return data as T;
  } catch (error) {
    throw explainFetch(error);
  }
}

async function post<T>(path: string, body: unknown): Promise<T> {
  try {
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
  } catch (error) {
    throw explainFetch(error);
  }
}

// --- theme --------------------------------------------------------------------

type Theme = "light" | "dark";

function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
  $("theme-toggle").textContent = theme === "dark" ? "Light theme" : "Dark theme";
  window.localStorage.setItem("ragdesk-theme", theme);
}

const storedTheme = window.localStorage.getItem("ragdesk-theme");
const initialTheme: Theme =
  storedTheme === "dark" || storedTheme === "light"
    ? storedTheme
    : window.matchMedia?.("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
applyTheme(initialTheme);

$("theme-toggle").addEventListener("click", () => {
  applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
});

function markSeg(containerId: string, value: number | string): void {
  document.querySelectorAll<HTMLElement>(`#${containerId} .seg-item`).forEach((item) => {
    item.classList.toggle("is-active", item.dataset.value === String(value));
  });
}

async function saveSetting(body: Record<string, number | string | boolean>): Promise<void> {
  try {
    await post("/api/settings", body);
    await loadStatus();
  } catch (error) {
    toast(error instanceof Error ? error.message : String(error));
    await loadStatus();
  }
}

document.querySelectorAll<HTMLElement>(".seg").forEach((group) => {
  group.addEventListener("click", (event) => {
    const item = (event.target as HTMLElement).closest<HTMLElement>(".seg-item");
    if (!item) return;
    const value = item.dataset.value ?? "";
    if (group.id === "auto-index-seg") void saveSetting({ auto_index_hours: Number(value) });
    if (group.id === "watch-seg") void saveSetting({ watch_seconds: Number(value) });
    if (group.id === "idle-unload-seg") void saveSetting({ idle_unload_minutes: Number(value) });
    if (group.id === "hyde-seg") void saveSetting({ hyde: Number(value) === 1 });
    if (group.id === "quiet-seg") void saveSetting({ embed_threads: Number(value) === 1 ? 4 : 0 });
    if (group.id === "answer-length-seg") void saveSetting({ answer_length: value });
    if (group.id === "preset-seg") void saveSetting({ preset: value });
    if (group.id === "backend-seg") {
      void saveSetting({ llm_preference: value });
      const form = $<HTMLFormElement>("openai-form");
      form.hidden = value !== "openai";
    }
  });
});

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
  if (tab === "indexed") {
    void loadHealth();
    void loadTopics();
    void loadRefusals();
    void loadDuplicates();
  }
  if (tab === "sources") void loadBookmarks();
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

function llmLabel(status: Status): string {
  if (status.llm.kind === "none") return "no LLM — see Settings";
  const short = status.llm.model.split("/").pop() ?? status.llm.model;
  return `${short} (${status.llm.kind})`;
}

function llmSetupButtons(status: Status): string {
  const options = status.llm_setup;
  const buttons: string[] = [];
  const repoName = options.mlx_repo.split("/").pop() ?? "model";
  if (options.mlx_available && options.mlx_repo && !options.mlx_cached) {
    buttons.push(
      `<button class="btn btn-primary" type="button" data-action="llm-setup-mlx">Download ${escapeHtml(repoName)} (in-process, one-time)</button>`,
    );
  }
  if (options.ollama_reachable && !options.ollama_has_model) {
    buttons.push(
      `<button class="btn" type="button" data-action="llm-setup-ollama">Pull ${escapeHtml(options.ollama_model)} with Ollama</button>`,
    );
  }
  return buttons.length ? `<div class="button-row">${buttons.join("")}</div>` : "";
}

function renderLlmSetup(status: Status): void {
  const box = $("llm-setup");
  const job = status.llm_setup.job;
  if (job.running) {
    const percent = Math.round(job.progress * 100);
    box.innerHTML = `
      <div class="progress"><div class="progress-bar" style="width:${percent}%"></div></div>
      <p class="source-result">${escapeHtml(job.detail || "working…")} · ${percent}% (${escapeHtml(job.model)})</p>`;
    startSetupPolling();
    return;
  }
  const error = job.error ? `<p class="source-result">${escapeHtml(job.error)}</p>` : "";
  const buttons = llmSetupButtons(status);
  if (status.llm.kind !== "none") {
    const pending =
      status.llm.kind === "mlx" && !status.llm_setup.mlx_cached
        ? `<p class="source-note">This preset's model is not downloaded yet — use the button in Machine preset below, or the first question will fetch it silently.</p>`
        : "";
    box.innerHTML = `
      <div class="status-line">
        <span class="status-model"><span class="dot is-hot"></span>${escapeHtml(status.llm.model)}</span>
        <span class="status-kind">${status.llm.kind === "mlx" ? "in-process on this machine" : "reused from your Ollama"}</span>
      </div>${pending}${error}`;
    return;
  }
  box.innerHTML =
    `<p class="source-note">Chat needs a local model — one-time download, stays on this machine. Indexing and search already work without it.</p>` +
    (buttons ||
      `<p class="source-note">Install one first: <code>uv tool install "ragdesk[mlx]"</code> (Apple Silicon), or start Ollama and pull <code>${escapeHtml(status.llm_setup.ollama_model)}</code>.</p>`) +
    error;
}

let setupTimer: number | undefined;

function startSetupPolling(): void {
  if (setupTimer) return;
  setupTimer = window.setInterval(async () => {
    await loadStatus();
    if (!status?.llm_setup.job.running) {
      window.clearInterval(setupTimer);
      setupTimer = undefined;
    }
  }, 2000);
}

let activityPoll: number | undefined;

function startActivityPolling(): void {
  if (activityPoll) return;
  activityPoll = window.setInterval(async () => {
    await loadStatus();
    if (!status?.activity.running) {
      window.clearInterval(activityPoll);
      activityPoll = undefined;
    }
  }, 2000);
}

function renderActivity(status: Status): void {
  const line = $("rail-activity");
  const activity = status.activity;
  if (!activity.running) {
    line.hidden = true;
    if (activityPoll) {
      window.clearInterval(activityPoll);
      activityPoll = undefined;
    }
    return;
  }
  const elapsed = Math.max(0, Math.round(Date.now() / 1000 - activity.started));
  line.hidden = false;
  line.textContent = `${activity.kind} · ${activity.detail}${activity.done ? ` (${activity.done})` : ""} · ${elapsed}s`;
  const cardKind = activity.kind.split(":")[0];
  if (cardKind === "github" || cardKind === "gitlab") {
    setResult(cardKind, `syncing — ${activity.detail} · ${elapsed}s`);
  } else if (cardKind === "index") {
    setResult("local", `indexing — ${activity.detail} · ${elapsed}s`);
  }
  startActivityPolling();
}

function renderStatus(): void {
  if (!status) return;
  if (wizardOpen) renderWizard();
  $("stat-docs").textContent = String(status.documents);
  $("stat-chunks").textContent = String(status.chunks);
  $("rail-meta").textContent = `${status.preset} preset · ${llmLabel(status)}`;

  const rows: Array<[string, string]> = [
    ["version", status.version],
    ["database", status.db],
    ["embedder", status.embedder.name ?? "not set"],
    ["rerank", status.rerank],
    ["llm", status.llm.kind === "none" ? status.llm.note : `${status.llm.model} (${status.llm.kind})`],
    ["preset", status.preset],
  ];
  $("status-table").innerHTML = rows
    .map(
      ([key, value]) =>
        `<div class="status-row"><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd></div>`,
    )
    .join("");

  const table = $("indexed-table");
  renderLlmSetup(status);
  markSeg("watch-seg", status.watch_seconds);
  const watched = (status.local_paths ?? []).filter((entry) => entry.documents > 0);
  $("watch-note").textContent = status.watch_seconds
    ? `checking ${watched.length || "your"} folder${watched.length === 1 ? "" : "s"} every ${
        status.watch_seconds < 60 ? `${status.watch_seconds}s` : `${Math.round(status.watch_seconds / 60)} min`
      } — changes and new files appear on their own`
    : "off — new or edited files wait for the next Auto re-index pass";
  markSeg("auto-index-seg", status.auto_index.hours);
  $("auto-index-last").textContent = status.auto_index.last_run
    ? `last run ${status.auto_index.last_run} UTC`
    : "never ran";
  markSeg("idle-unload-seg", status.memory.idle_unload_minutes);
  $("memory-dot").classList.toggle("is-hot", status.memory.models_loaded);
  $("memory-state").textContent = status.memory.models_loaded
    ? "in RAM — unloads after the idle stretch"
    : "released — the next question reloads them";
  markSeg("backend-seg", status.llm_setting.preference);
  $<HTMLFormElement>("openai-form").hidden = status.llm_setting.preference !== "openai";
  $("backend-note").textContent = status.llm_setting.openai_host
    ? `endpoint ${status.llm_setting.openai_host} · model ${status.llm_setting.openai_model}${status.llm_setting.openai_key_set ? " · key saved" : ""}`
    : "Auto keeps the local model; OpenAI-compatible covers LM Studio, llama.cpp, vLLM or OpenAI itself.";
  markSeg("answer-length-seg", status.answer_length);
  markSeg("hyde-seg", status.hyde ? 1 : 0);
  markSeg("quiet-seg", status.embed_threads ? 1 : 0);
  $("quiet-note").textContent = status.embed_threads
    ? `capped at ${status.embed_threads} threads — measured on this repo: 1.4× slower to index, ~35% of the CPU`
    : "every core — measured: 651 chunks in 97s at ~580% CPU; quiet mode takes 136s at ~390%";
  $("hyde-note").textContent = status.hyde
    ? `on — drafts with ${status.llm.kind === "none" ? "the local model (none found yet)" : status.llm.model}; adds 2-4s per question`
    : "off — measured on our golden set: recall@5 0.89 → 1.00 with it on, at +2-4s per question";
  markSeg("preset-seg", status.preset);
  const presetName = status.preset;
  const activePreset = (status.presets ?? []).find((entry) => entry.name === presetName);
  $("preset-note").textContent = activePreset
    ? `${activePreset.note} · rerank ${activePreset.rerank}`
    : "";
  const repoShort = status.llm_setup.mlx_repo.split("/").pop() ?? "";
  const needsDownload =
    status.llm.kind === "mlx" &&
    status.llm_setup.mlx_available &&
    !status.llm_setup.mlx_cached &&
    Boolean(status.llm_setup.mlx_repo);
  $("preset-cta").innerHTML = needsDownload
    ? `<button class="btn btn-primary" type="button" data-action="llm-setup-mlx">Download ${escapeHtml(repoShort)}</button>
       <span class="caption">one-time download for this preset — it stays on this machine</span>`
    : "";
  renderActivity(status);
  if (status.sources.length === 0) {
    table.innerHTML = `<p class="muted">Nothing indexed yet. Add a source.</p>`;
    return;
  }
  const totalDocs = status.sources.reduce((sum, entry) => sum + entry.documents, 0);
  const totalChunks = status.sources.reduce((sum, entry) => sum + entry.chunks, 0);
  const localPaths = status.local_paths ?? [];
  const maxDocs = Math.max(...status.sources.map((entry) => entry.documents), 1);
  const maxChunks = Math.max(...status.sources.map((entry) => entry.chunks), 1);
  const numberCell = (value: number, max: number) =>
    `<span class="stat-bar-cell">${value}<span class="stat-bar" style="--bar:${Math.round((value / max) * 100)}%"></span></span>`;
  table.innerHTML = `
    <div class="stats-row stats-head">
      <span>Source</span><span>Documents</span><span>Chunks</span><span>Last indexed</span>
    </div>
    ${status.sources
      .map((entry) => {
        const children =
          entry.source === "local"
            ? localPaths
                .map(
                  (child) => `<div class="stats-row stats-sub">
          <span class="stats-source" title="${escapeHtml(child.path)}">↳ ${escapeHtml(child.path)}</span>
          ${numberCell(child.documents, maxDocs)}
          ${numberCell(child.chunks, maxChunks)}
          <span>${escapeHtml((child.indexed_at || "").slice(0, 16))}</span>
        </div>`,
                )
                .join("")
            : "";
        return `<div class="stats-row">
          <span class="stats-source">${escapeHtml(entry.source)}</span>
          ${numberCell(entry.documents, maxDocs)}
          ${numberCell(entry.chunks, maxChunks)}
          <span>${escapeHtml((entry.indexed_at || "").slice(0, 16))}</span>
        </div>${children}`;
      })
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
  const where = `${hit.path}${hit.line && hit.line > 1 ? `:${hit.line}` : ""}`;
  const pathTag = openable
    ? `<code class="cite-path" data-open-path="${escapeHtml(hit.path)}" title="Open file">${escapeHtml(where)}</code>`
    : `<code class="cite-path">${escapeHtml(where)}</code>`;
  const chips = Object.entries(hit.metadata ?? {})
    .slice(0, 3)
    .map(
      ([key, value]) =>
        `<span class="meta-chip">${escapeHtml(key)}: ${escapeHtml(String(value))}</span>`,
    )
    .join("");
  // Older rows (written by early terminal chats) predate the full citation
  // shape: render what exists instead of throwing on a missing score.
  const score = typeof hit.cosine === "number" ? hit.cosine.toFixed(2) : "";
  return `<article class="cite" data-rank="${rank}">
    <span class="cite-rank">[${rank}]</span>
    <div class="cite-body">
      <div class="cite-top">
        ${pathTag}
        <span class="cite-score">${score}${hit.lanes ? `${score ? " " : ""}<em>${escapeHtml(hit.lanes)}</em>` : ""}</span>
      </div>
      ${chips ? `<div class="meta-chips">${chips}</div>` : ""}
      <p class="cite-snippet">${escapeHtml(snippet(hit.text, 240))}</p>
      <div class="cite-tools">
        <button class="action" type="button" data-related="${escapeHtml(hit.path)}">Related</button>
        <button class="action" type="button" data-backlinks="${escapeHtml(hit.path)}">Backlinks</button>
        <button class="action" type="button" data-scope="${escapeHtml(folderOf(hit.path))}" title="Ask only inside this folder">Only this folder</button>
      </div>
      <div class="cite-nav" hidden></div>
    </div>
  </article>`;
}

async function openLocal(path: string): Promise<void> {
  try {
    const { openPath } = await import("@tauri-apps/plugin-opener");
    await openPath(path);
  } catch (error) {
    try {
      await navigator.clipboard.writeText(path);
      toast(`Could not open the file — path copied instead (${path.split("/").pop()})`);
    } catch {
      toast(`Could not open ${path} (${error instanceof Error ? error.message : error})`);
    }
  }
}

document.addEventListener("click", (event) => {
  const target = event.target as HTMLElement;
  const path = target.closest<HTMLElement>("[data-open-path]")?.dataset.openPath;
  if (path) void openLocal(path);
});

// --- scope --------------------------------------------------------------------

let scopeFolder = "";

function folderOf(path: string): string {
  const cut = path.replace(/\\/g, "/").split("/");
  cut.pop();
  return cut.join("/") || "/";
}

function renderScopeChip(): void {
  $("scope-chip").hidden = !scopeFolder;
  $("scope-name").textContent = scopeFolder ? scopeFolder.split("/").pop() ?? scopeFolder : "";
}

function setScope(folder: string): void {
  scopeFolder = folder;
  renderScopeChip();
  toast(
    folder
      ? `Scoped to ${folder} — the next questions search only there`
      : "Scope cleared — searching everything again",
  );
}

$("scope-clear").addEventListener("click", () => setScope(""));

// --- chat ---------------------------------------------------------------------

let streaming = false;
let activeAbort: AbortController | null = null;
let entryCount = 0;

function nextEntryNo(): string {
  entryCount += 1;
  return String(entryCount).padStart(3, "0");
}

function appendUserMessage(query: string): void {
  document.getElementById("chat-empty")?.remove();
  const message = document.createElement("div");
  message.className = "msg msg-user";
  const no = document.createElement("span");
  no.className = "entry-no";
  no.textContent = nextEntryNo();
  const text = document.createElement("div");
  text.className = "user-text";
  text.textContent = query;
  message.append(no, text);
  $("chat-log").append(message);
}

function appendAssistantShell(): {
  answer: HTMLElement;
  cites: HTMLElement;
  status: HTMLElement;
  actions: HTMLElement;
  element: HTMLElement;
} {
  const message = document.createElement("div");
  message.className = "msg msg-assistant";
  const no = document.createElement("span");
  no.className = "entry-no";
  no.textContent = nextEntryNo();
  message.prepend(no);
  message.insertAdjacentHTML(
    "beforeend",
    `<div class="answer-status"></div><div class="answer streaming"></div><div class="answer-actions"></div><div class="cites"></div>`,
  );
  $("chat-log").append(message);
  message.scrollIntoView({ block: "end" });
  return {
    answer: message.querySelector<HTMLElement>(".answer") as HTMLElement,
    cites: message.querySelector<HTMLElement>(".cites") as HTMLElement,
    status: message.querySelector<HTMLElement>(".answer-status") as HTMLElement,
    actions: message.querySelector<HTMLElement>(".answer-actions") as HTMLElement,
    element: message,
  };
}

async function ask(rawQuery: string): Promise<void> {
  if (streaming || !status) {
    if (!status) toast("The local indexer is not reachable.");
    return;
  }
  const query = scopeFolder ? `folder:"${scopeFolder}" ${rawQuery}` : rawQuery;
  streaming = true;
  $<HTMLButtonElement>("chat-send").disabled = true;
  $("chat-stop").hidden = false;
  appendUserMessage(query);
  const { answer, cites, status: statusLine, actions, element: shellElement } =
    appendAssistantShell();
  const startedAt = Date.now();
  let phase = "working…";
  const ticker = window.setInterval(() => {
    const seconds = Math.round((Date.now() - startedAt) / 1000);
    statusLine.textContent = `${phase} ${seconds}s`;
  }, 1000);
  statusLine.textContent = "working…";
  activeAbort = new AbortController();

  try {
    const response = await fetch(`${API}/api/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 6, chat_id: currentChatId }),
      signal: activeAbort.signal,
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
            status?: string;
            done?: boolean;
            error?: string;
            hits?: Hit[];
            cached?: boolean;
            cached_question?: string;
            correction?: string;
            symbol?: string;
            chat_id?: number;
            answer_id?: number;
          };
          if (event.status) {
            phase = event.status;
            const seconds = Math.round((Date.now() - startedAt) / 1000);
            statusLine.textContent = `${phase} ${seconds}s`;
          }
          if (event.delta) {
            answer.textContent += event.delta;
            answer.scrollIntoView({ block: "end" });
          }
          if (event.error) throw new Error(event.error);
          if (event.done) {
            finished = true;
            window.clearInterval(ticker);
            statusLine.remove();
            answer.classList.remove("streaming");
            answer.classList.toggle("is-refused", answer.textContent === REFUSAL);
            renderCites(cites, event.hits ?? []);
            if (event.cached) {
              const badge = document.createElement("span");
              badge.className = "cache-badge";
              badge.textContent = "from cache";
              if (event.cached_question && event.cached_question !== query) {
                badge.title = `answered earlier for: ${event.cached_question}`;
              }
              answer.before(badge);
            }
            if (event.correction) {
              const badge = document.createElement("span");
              badge.className = "cache-badge";
              badge.textContent = "your fix";
              badge.title = `using your correction for: ${event.correction}`;
              answer.before(badge);
            }
            if (event.symbol) {
              const badge = document.createElement("span");
              badge.className = "cache-badge";
              badge.textContent = "call graph";
              badge.title = `symbol lookup for ${event.symbol} — definitions and call sites from the indexed code`;
              answer.before(badge);
            }
            if (event.chat_id) currentChatId = event.chat_id;
            if (event.answer_id) {
              renderAnswerActions(shellElement, actions, event.answer_id, 0, query);
              void autoVerify(actions, event.answer_id);
            }
            renderRichText(answer, answer.textContent ?? "");
            void renderDiagrams(answer, cites, answer.textContent ?? "");
            void loadChats();
          }
        }
        newline = buffer.indexOf("\n");
      }
    }
  } catch (error) {
    window.clearInterval(ticker);
    answer.classList.remove("streaming");
    if (error instanceof Error && error.name === "AbortError") {
      statusLine.textContent = "stopped — the partial answer stays above";
    } else {
      answer.classList.add("is-error");
      answer.textContent = error instanceof Error ? error.message : String(error);
      statusLine.remove();
    }
  } finally {
    streaming = false;
    activeAbort = null;
    $("chat-stop").hidden = true;
    $<HTMLButtonElement>("chat-send").disabled = false;
  }
}

$("chat-stop").addEventListener("click", () => {
  activeAbort?.abort();
});

function renderCites(container: HTMLElement, hits: Hit[]): void {
  if (hits.length === 0) return;
  container.innerHTML = `<p class="cites-label">${hits.length} source${hits.length === 1 ? "" : "s"}</p>${hits
    .map((hit, index) => citeCard(hit, index + 1))
    .join("")}`;
  container.querySelectorAll(".cite").forEach((card, index) => {
    (card as HTMLElement).style.setProperty("--stagger", `${index * 60}ms`);
  });
}

function growComposer(): void {
  const input = $<HTMLTextAreaElement>("chat-input");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 216)}px`;
}

$<HTMLTextAreaElement>("chat-input").addEventListener("input", growComposer);

$<HTMLFormElement>("chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const input = $<HTMLTextAreaElement>("chat-input");
  const query = input.value.trim();
  if (query) {
    input.value = "";
    growComposer();
    void ask(query);
  }
});

$<HTMLTextAreaElement>("chat-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $<HTMLFormElement>("chat-form").requestSubmit();
  }
});

function bindSuggestionChips(): void {
  const box = $("chat-suggestions");
  box.innerHTML = "";
  for (const suggestion of SUGGESTIONS) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = suggestion;
    chip.addEventListener("click", () => void ask(suggestion));
    box.append(chip);
  }
}

bindSuggestionChips();

// --- chat history -------------------------------------------------------------

type ChatMessage = {
  id: number;
  role: string;
  text: string;
  citations: Hit[];
  feedback?: number;
};
type ChatSummary = { id: number; title: string; updated_at: string; messages: number };

let currentChatId = 0;
let chatCache: ChatSummary[] = [];
const chatLogTemplate = $("chat-log").innerHTML;

function relativeWhen(stamp: string): string {
  const parsed = Date.parse(`${stamp.replace(" ", "T")}Z`);
  if (Number.isNaN(parsed)) return stamp.slice(0, 16);
  const minutes = Math.round((Date.now() - parsed) / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  if (minutes < 60 * 24) return `${Math.round(minutes / 60)}h ago`;
  return `${Math.round(minutes / (60 * 24))}d ago`;
}

function renderHistoryPanel(): void {
  const panel = $("chat-history-panel");
  const rows = chatCache
    .map(
      (chat) => `<div class="chat-history-row${chat.id === currentChatId ? " is-current" : ""}" data-chat="${chat.id}">
        <button class="chat-history-open" type="button" data-chat="${chat.id}">
          <span class="chat-history-title">${escapeHtml(chat.title)}</span>
          <span class="chat-history-when">${relativeWhen(chat.updated_at)} · ${chat.messages} msg${chat.messages === 1 ? "" : "s"}</span>
        </button>
        <button class="chat-history-delete" type="button" data-delete="${chat.id}" aria-label="Delete conversation">×</button>
      </div>`,
    )
    .join("");
  panel.innerHTML =
    `<p class="chat-history-head">Past conversations</p>` +
    rows +
    `<button class="chat-history-new" type="button" data-chat="0">＋ New conversation</button>` +
    (rows ? "" : `<p class="chat-history-empty">Nothing saved yet — ask something.</p>`);
}

async function loadChats(): Promise<ChatSummary[]> {
  try {
    const { chats } = await get<{ chats: ChatSummary[] }>("/api/chats");
    chatCache = chats;
    const current = chats.find((chat) => chat.id === currentChatId);
    $("chat-current").textContent = current ? current.title : "New conversation";
    renderHistoryPanel();
    return chats;
  } catch {
    return [];
  }
}

$("chat-history-btn").addEventListener("click", () => {
  const panel = $("chat-history-panel");
  panel.hidden = !panel.hidden;
  if (!panel.hidden) void loadChats();
});

$("chat-history-panel").addEventListener("click", async (event) => {
  const target = event.target as HTMLElement;
  const deleteId = Number(target.closest<HTMLElement>("[data-delete]")?.dataset.delete ?? 0);
  if (deleteId) {
    try {
      await post("/api/chats/delete", { chat_id: deleteId });
      if (deleteId === currentChatId) newChat();
      await loadChats();
    } catch (error) {
      toast(error instanceof Error ? error.message : String(error));
    }
    return;
  }
  const openId = Number(target.closest<HTMLElement>("[data-chat]")?.dataset.chat ?? -1);
  if (openId === -1) return;
  $("chat-history-panel").hidden = true;
  if (openId === 0) newChat();
  else void openChat(openId);
});

document.addEventListener("click", (event) => {
  const panel = $("chat-history-panel");
  if (panel.hidden) return;
  const target = event.target as HTMLElement;
  if (!target.closest(".chat-tools")) panel.hidden = true;
});

function resetChatLog(): void {
  entryCount = 0;
  $("chat-log").innerHTML = chatLogTemplate;
  bindSuggestionChips();
}

function renderChat(messages: ChatMessage[]): void {
  if (messages.length === 0) {
    resetChatLog();
    return;
  }
  entryCount = 0;
  $("chat-log").innerHTML = "";
  let question = "";
  for (const message of messages) {
    if (message.role === "user") {
      appendUserMessage(message.text);
      question = message.text;
      continue;
    }
    const { answer, cites, actions, element } = appendAssistantShell();
    answer.classList.remove("streaming");
    answer.textContent = message.text;
    answer.classList.toggle("is-refused", message.text === REFUSAL);
    renderAnswerActions(element, actions, message.id, message.feedback ?? 0, question);
    renderCites(cites, message.citations ?? []);
    renderRichText(answer, message.text);
    void renderDiagrams(answer, cites, message.text);
  }
}

async function openChat(id: number): Promise<void> {
  try {
    const detail = await get<{ messages: ChatMessage[] }>(`/api/chats/${id}`);
    currentChatId = id;
    renderChat(detail.messages);
    await loadChats();
  } catch (error) {
    toast(error instanceof Error ? error.message : String(error));
  }
}

function newChat(): void {
  currentChatId = 0;
  resetChatLog();
  void loadChats();
}

$("chat-new").addEventListener("click", () => newChat());

void loadChats().then((chats) => {
  if (chats.length > 0) void openChat(chats[0].id);
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
  gitlab: { connected: boolean; name: string; base_url: string };
  msgraph: { connected: boolean; account: string; client_id_set: boolean };
  email: { connected: boolean; host: string; user: string; folder: string };
};

type DeviceFlow = { userCode: string; verificationUri: string; interval: number };

let connections: Connections | null = null;
let selectedPaths: string[] = [];
const deviceFlows: Record<string, DeviceFlow> = {};
const deviceTimers: Record<string, number> = {};
const DEVICE_ENDPOINTS: Record<string, { start: string; poll: string }> = {
  github: {
    start: "/api/connections/github/device/start",
    poll: "/api/connections/github/device/poll",
  },
  msgraph: {
    start: "/api/connections/msgraph/device/start",
    poll: "/api/connections/msgraph/device/poll",
  },
};
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

function cancelDeviceFlow(provider: string): void {
  delete deviceFlows[provider];
  window.clearTimeout(deviceTimers[provider]);
}

function startDevicePolling(provider: string): void {
  window.clearTimeout(deviceTimers[provider]);
  const flow = deviceFlows[provider];
  if (!flow) return;
  const endpoints = DEVICE_ENDPOINTS[provider];
  deviceTimers[provider] = window.setTimeout(async () => {
    if (!deviceFlows[provider]) return;
    try {
      const response = await post<{
        connected?: boolean;
        pending?: boolean;
        login?: string;
        interval?: number;
      }>(endpoints.poll, {});
      if (response.connected) {
        const account = response.login ?? provider;
        cancelDeviceFlow(provider);
        setResult(provider, `connected — ${account}`);
        await loadConnections();
        return;
      }
      if (response.interval) deviceFlows[provider].interval = response.interval;
      startDevicePolling(provider);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      cancelDeviceFlow(provider);
      setResult(provider, message);
      toast(message);
      renderSources();
    }
  }, Math.max(flow.interval, 3) * 1000);
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
    if (wizardOpen) renderWizard();
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

async function startDevice(provider: string): Promise<void> {
  const labels: Record<string, string> = { github: "GitHub", msgraph: "Microsoft" };
  const label = labels[provider] ?? provider;
  setResult(provider, "requesting a code…");
  try {
    const response = await post<{
      user_code: string;
      verification_uri: string;
      interval: number;
    }>(DEVICE_ENDPOINTS[provider].start, {});
    deviceFlows[provider] = {
      userCode: response.user_code,
      verificationUri: response.verification_uri,
      interval: response.interval || 5,
    };
    setResult(provider, "enter the code shown in your browser");
    renderSources();
    void openExternal(response.verification_uri);
    startDevicePolling(provider);
    void label;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setResult(provider, message);
    toast(message);
  }
}

function deviceBox(provider: string, label: string): string {
  const flow = deviceFlows[provider];
  return `<div class="device-box">
        <p class="source-note">Enter this code on ${label}:</p>
        <p class="device-code">${escapeHtml(flow.userCode)}</p>
        <div class="button-row">
          <button class="btn" type="button" data-action="open-device-page" data-provider="${provider}">Open ${label}</button>
          <button class="btn btn-quiet" type="button" data-action="${provider}-device-cancel">Cancel</button>
        </div>
        <p class="muted">Waiting for approval…</p>
      </div>`;
}

function githubCard(conn: Connections["github"]): string {
  if (!conn.connected) {
    let primary: string;
    if (deviceFlows.github) {
      primary = deviceBox("github", "GitHub");
    } else if (conn.device_flow_ready) {
      primary = `<button class="btn btn-primary" type="button" data-action="github-device-start">Connect with a code</button>`;
    } else {
      primary = `<button class="btn btn-primary" type="button" data-action="reveal-github">Connect with a code</button>`;
    }
    const setup =
      !deviceFlows.github && !conn.device_flow_ready && reveal.github
        ? `<form data-form="github-client-id" class="stack">
             <p class="source-note">Create a GitHub OAuth app with device flow enabled, then paste its client ID once.</p>
             <input name="client_id" placeholder="OAuth client ID" required />
             <button class="btn" type="submit">Save &amp; get a code</button>
           </form>`
        : "";
    const gh =
      conn.gh_available && !deviceFlows.github
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

function gitlabCard(conn: Connections["gitlab"]): string {
  if (!conn.connected) {
    return `<div class="source-card">
      <h3>GitLab</h3>
      <p class="source-note">Read-only repository sync. Needs a personal access token with the <code>read_api</code> scope.</p>
      <form data-form="gitlab-connect" class="stack">
        <div class="field-row">
          <input name="token" type="password" placeholder="glpat-…" required />
          <input name="base_url" placeholder="gitlab.com (or self-hosted URL)" />
        </div>
        <button class="btn btn-primary" type="submit">Connect GitLab</button>
      </form>
      <p class="source-result" data-result="gitlab"></p>
    </div>`;
  }
  return `<div class="source-card is-connected">
    <h3>GitLab <span class="conn-badge">connected</span></h3>
    <p class="source-note">${conn.name ? escapeHtml(conn.name) : "token"} · ${escapeHtml(conn.base_url || "gitlab.com")}</p>
    <div class="button-row">
      <button class="btn btn-quiet" type="button" data-action="disconnect-gitlab">Disconnect</button>
    </div>
    <form data-form="gitlab-sync" class="stack">
      <input name="project" placeholder="group/name" required />
      <div class="field-row">
        <input name="ref" placeholder="branch (optional)" />
        <input name="subdir" placeholder="subfolder (optional)" />
      </div>
      <button class="btn btn-primary" type="submit">Sync repo</button>
    </form>
    <p class="source-result" data-result="gitlab"></p>
  </div>`;
}

function msgraphCard(conn: Connections["msgraph"]): string {
  if (!conn.connected) {
    let primary: string;
    if (deviceFlows.msgraph) {
      primary = deviceBox("msgraph", "Microsoft");
    } else if (conn.client_id_set) {
      primary = `<button class="btn btn-primary" type="button" data-action="msgraph-device-start">Connect Microsoft</button>`;
    } else {
      primary = `<button class="btn btn-primary" type="button" data-action="reveal-msgraph">Connect Microsoft</button>`;
    }
    const setup =
      !conn.client_id_set && reveal.msgraph
        ? `<form data-form="msgraph-client-id" class="stack">
             <p class="source-note">Create an Azure app registration (public client; delegated Files.Read.All, Sites.Read.All, offline_access) and paste its Application (client) ID once.</p>
             <input name="client_id" placeholder="Azure Application (client) ID" required />
             <button class="btn" type="submit">Save &amp; connect</button>
           </form>`
        : "";
    return `<div class="source-card">
      <h3>Microsoft</h3>
      <p class="source-note">OneDrive &amp; SharePoint, read-only. Microsoft shows a code, you approve in the browser.</p>
      ${primary}
      ${setup}
      <p class="source-result" data-result="msgraph"></p>
    </div>`;
  }
  return `<div class="source-card is-connected">
    <h3>Microsoft <span class="conn-badge">connected</span></h3>
    <p class="source-note">${conn.account ? escapeHtml(conn.account) : "account"} · OneDrive + SharePoint</p>
    <div class="button-row">
      <button class="btn btn-quiet" type="button" data-action="disconnect-msgraph">Disconnect</button>
    </div>
    <form data-form="msgraph-sync" class="stack">
      <div class="field-row">
        <input name="folder_id" placeholder="OneDrive folder id (optional)" />
        <input name="site" placeholder="site hostname:/sites/x (optional)" />
      </div>
      <button class="btn btn-primary" type="submit">Sync Microsoft files</button>
    </form>
    <p class="source-result" data-result="msgraph"></p>
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

function emailCard(conn: Connections["email"]): string {
  const mboxForm = `<form data-form="email-mbox" class="field-row">
      <input name="path" placeholder="/path/to/inbox.mbox (optional)" />
      <button class="btn btn-quiet" type="submit">Index mbox</button>
    </form>`;
  if (!conn.connected) {
    return `<div class="source-card">
      <h3>Email</h3>
      <p class="source-note">Read-only via IMAP: nothing is marked as read, nothing is deleted. Gmail needs an app password.</p>
      <form data-form="email-connect" class="stack">
        <div class="field-row">
          <input name="host" placeholder="imap.gmail.com" required />
          <input name="port" type="number" placeholder="993" />
        </div>
        <div class="field-row">
          <input name="user" placeholder="you@gmail.com" required />
          <input name="password" type="password" placeholder="app password" required />
        </div>
        <button class="btn btn-primary" type="submit">Connect email</button>
      </form>
      ${mboxForm}
      <p class="source-result" data-result="email"></p>
    </div>`;
  }
  return `<div class="source-card is-connected">
    <h3>Email <span class="conn-badge">connected</span></h3>
    <p class="source-note">${escapeHtml(conn.user)} · ${escapeHtml(conn.host)}</p>
    <div class="button-row">
      <button class="btn btn-quiet" type="button" data-action="disconnect-email">Disconnect</button>
    </div>
    <form data-form="email-sync" class="stack">
      <div class="field-row">
        <input name="folder" placeholder="INBOX" value="${escapeHtml(conn.folder || "INBOX")}" />
        <input name="limit" type="number" min="1" max="5000" placeholder="newest 200" />
      </div>
      <button class="btn btn-primary" type="submit">Sync email</button>
    </form>
    ${mboxForm}
    <p class="source-result" data-result="email"></p>
  </div>`;
}

function vaultCard(vaults: string[]): string {
  const list = vaults.length
    ? `<div class="memory-list">${vaults
        .map(
          (vault) =>
            `<div class="memory-item"><span class="bookmark-url" title="${escapeHtml(vault)}">${escapeHtml(vault)}</span></div>`,
        )
        .join("")}</div>`
    : `<p class="caption">no vault added yet</p>`;
  return `<div class="source-card">
    <h3>Obsidian vault</h3>
    <p class="source-note">Aliases and tags become searchable, front-matter stays filterable, and .obsidian/.trash are never read. Wikilinks already resolve through Backlinks.</p>
    ${list}
    <div class="button-row">
      <button class="btn" type="button" data-action="pick-vault">Choose vault…</button>
    </div>
    <form data-form="obsidian-add" class="field-row">
      <input name="path" placeholder="…or paste the vault path" />
      <button class="btn" type="submit">Add vault</button>
    </form>
    <p class="source-result" data-result="obsidian"></p>
  </div>`;
}

function notesCard(available: boolean): string {
  if (!available) return "";
  return `<div class="source-card">
    <h3>Apple Notes</h3>
    <p class="source-note">Read-only export through AppleScript. macOS asks once for permission to control Notes.</p>
    <button class="btn btn-primary" type="button" data-action="notes-sync">Sync notes</button>
    <p class="source-result" data-result="notes"></p>
  </div>`;
}

function webCard(): string {
  return `<div class="source-card">
    <h3>Website</h3>
    <p class="source-note">Save a single page, or crawl a docs site. HTML only, read-only.</p>
    <form data-form="web-sync" class="stack">
      <input name="url" placeholder="https://docs.example.com/start" required />
      <div class="field-row">
        <input name="max_pages" type="number" min="1" max="500" placeholder="max pages (50)" />
        <input name="max_depth" type="number" min="0" max="5" placeholder="depth (2)" />
      </div>
      <button class="btn btn-primary" type="submit">Crawl site</button>
    </form>
    <form data-form="web-save" class="field-row">
      <input name="url" placeholder="https://example.com/article" required />
      <button class="btn" type="submit">Save page</button>
    </form>
    <p class="source-result" data-result="web"></p>
    <p class="caption">Saved pages</p>
    <div class="memory-list" id="bookmark-list"></div>
  </div>`;
}

function filterSources(): void {
  const input = $<HTMLInputElement>("source-filter");
  const query = input.value.trim().toLowerCase();
  document
    .querySelectorAll<HTMLElement>("#source-grid .source-card")
    .forEach((card) => {
      card.classList.toggle(
        "is-filtered",
        Boolean(query) && !(card.textContent ?? "").toLowerCase().includes(query),
      );
    });
}

$("source-filter").addEventListener("input", filterSources);

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
  const gitlab = connections?.gitlab ?? { connected: false, name: "", base_url: "" };
  const msgraph = connections?.msgraph ?? {
    connected: false,
    account: "",
    client_id_set: false,
  };
  const email = connections?.email ?? {
    connected: false,
    host: "",
    user: "",
    folder: "INBOX",
  };
  $("source-grid").innerHTML =
    localCard() +
    githubCard(github) +
    gitlabCard(gitlab) +
    confluenceCard(confluence) +
    gdriveCard(gdrive) +
    msgraphCard(msgraph) +
    notionCard(notion) +
    emailCard(email) +
    vaultCard(status?.vaults ?? []) +
    notesCard(status?.notes_available ?? false) +
    webCard();
  for (const [kind, message] of Object.entries(sourceResults)) {
    const element = document.querySelector<HTMLElement>(`[data-result="${kind}"]`);
    if (element) element.textContent = message;
  }
  filterSources();
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
      void startDevice("github");
      return;
    }
    if (kind === "openai-config") {
      const host = String(payload.openai_host ?? "");
      const model = String(payload.openai_model ?? "");
      const key = String(payload.openai_api_key ?? "");
      if (!host || !model) {
        setResult("openai", "host and model are required");
        return;
      }
      await post("/api/settings", {
        openai_host: host,
        openai_model: model,
        llm_preference: "openai",
        ...(key ? { openai_api_key: key } : {}),
      });
      (form as HTMLFormElement).reset();
      toast("OpenAI-compatible endpoint saved");
      await loadStatus();
      return;
    }
    if (kind === "msgraph-client-id") {
      if (!payload.client_id) {
        setResult("msgraph", "paste the Azure client ID first");
        return;
      }
      await post("/api/connections/msgraph", { client_id: payload.client_id });
      await loadConnections();
      void startDevice("msgraph");
      return;
    }
    if (kind === "email-connect") {
      setResult("email", "checking the mailbox…");
      const response = await post<{ display_name: string }>("/api/connections/email", {
        host: String(payload.host ?? ""),
        user: String(payload.user ?? ""),
        password: String(payload.password ?? ""),
        port: payload.port ? Number(payload.port) : 0,
      });
      (form as HTMLFormElement).reset();
      setResult("email", `connected — ${response.display_name}`);
      await loadConnections();
      return;
    }
    if (kind === "email-sync") {
      setResult("email", "reading the mailbox…");
      const response = await post<Record<string, number>>("/api/sync/email", {
        folder: String(payload.folder ?? ""),
        limit: payload.limit ? Number(payload.limit) : 0,
      });
      setResult("email", summarize(response));
      await loadStatus();
      return;
    }
    if (kind === "email-mbox") {
      if (!payload.path) {
        setResult("email", "paste the path of an .mbox file first");
        return;
      }
      setResult("email", "reading the archive…");
      const response = await post<Record<string, number>>("/api/sync/email-mbox", {
        path: String(payload.path),
      });
      setResult("email", summarize(response));
      await loadStatus();
      return;
    }
    if (kind === "obsidian-add") {
      const path = String(payload.path ?? "");
      if (!path) {
        setResult("obsidian", "paste the vault path first");
        return;
      }
      (form as HTMLFormElement).reset();
      await addVault(path);
      return;
    }
    if (kind === "web-save") {
      const url = String(payload.url ?? "");
      if (!url) {
        setResult("web", "paste a URL first");
        return;
      }
      setResult("web", "saving…");
      const response = await post<Record<string, number>>("/api/save", { url });
      setResult(
        "web",
        response.indexed || response.unchanged
          ? summarize(response)
          : "fetched, but no readable text — not an HTML page?",
      );
      (form as HTMLFormElement).reset();
      await loadBookmarks();
      await loadStatus();
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
      "gitlab-connect": "/api/connections/gitlab",
      "gitlab-sync": "/api/sync/gitlab",
      "msgraph-sync": "/api/sync/msgraph",
      "notes-sync": "/api/sync/notes",
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

async function addVault(path?: string): Promise<void> {
  let chosen = path ?? "";
  if (!chosen) {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selection = await open({
        directory: true,
        multiple: false,
        title: "Choose an Obsidian vault",
      });
      if (!selection) return;
      chosen = Array.isArray(selection) ? selection[0] : selection;
    } catch {
      setResult("obsidian", "Folder picking works in the desktop app; paste the path instead.");
      return;
    }
  }
  setResult("obsidian", "indexing the vault…");
  try {
    const response = await post<{ is_vault: boolean } & Record<string, number>>(
      "/api/obsidian",
      { path: chosen },
    );
    setResult(
      "obsidian",
      `${summarize(response)}${response.is_vault ? "" : " (no .obsidian/ folder — indexed as a plain folder)"}`,
    );
    await loadStatus();
  } catch (error) {
    setResult("obsidian", error instanceof Error ? error.message : String(error));
  }
}

async function loadMcp(): Promise<void> {
  try {
    const info = await get<{ cli_on_path: boolean; cli_path: string }>("/api/mcp");
    $("mcp-dot").classList.toggle("is-hot", info.cli_on_path);
    $("mcp-state").textContent = info.cli_on_path
      ? `ready — clients run "ragdesk mcp" (${info.cli_path})`
      : "the ragdesk command is missing from PATH — click Install";
  } catch {
    $("mcp-state").textContent = "server offline";
  }
}

document.addEventListener("click", async (event) => {
  const element = event.target as HTMLElement;
  const mcp = element.closest<HTMLElement>("[data-mcp]");
  if (mcp) {
    try {
      const info = await get<{ cli_on_path: boolean; snippets: Record<string, string> }>("/api/mcp");
      let snippet = info.snippets[mcp.dataset.mcp ?? ""] ?? "";
      if (!info.cli_on_path) {
        await post("/api/mcp/install", {});
        snippet += "\n# or point the client at the ragdesk binary directly";
      }
      await navigator.clipboard.writeText(snippet);
      toast("MCP setup copied — paste it into your client");
      await loadMcp();
    } catch (error) {
      toast(error instanceof Error ? error.message : String(error));
    }
    return;
  }
  if (element.closest<HTMLElement>("#mcp-install")) {
    const result = await post<{ installed: boolean; reason: string; path: string }>(
      "/api/mcp/install",
      {},
    );
    toast(result.installed ? `Installed at ${result.path}` : `Nothing to do: ${result.reason}`);
    await loadMcp();
    return;
  }
  const use = element.closest<HTMLElement>("[data-use]");
  if (use) {
    await saveSetting({ llm_preference: use.dataset.use ?? "" });
    toast(`Answer engine: ${use.dataset.use}`);
    return;
  }
  const scope = element.closest<HTMLElement>("[data-scope]");
  if (scope?.dataset.scope) {
    setScope(scope.dataset.scope);
    return;
  }
  const related = element.closest<HTMLElement>("[data-related]");
  const backlinks = element.closest<HTMLElement>("[data-backlinks]");
  if (related || backlinks) {
    const path = (related ?? backlinks)?.dataset[related ? "related" : "backlinks"] ?? "";
    const box = (related ?? backlinks)?.closest(".cite-body")?.querySelector<HTMLElement>(".cite-nav");
    if (!path || !box) return;
    const kind = related ? "related" : "backlinks";
    box.hidden = false;
    box.textContent = "loading…";
    try {
      const payload = await get<Record<string, Array<{ path: string; score?: number; hits?: number }>>>(
        `/api/${kind}?path=${encodeURIComponent(path)}`,
      );
      const rows = payload[kind] ?? [];
      box.innerHTML = rows.length
        ? rows
            .map(
              (row) =>
                `<span class="cite-nav-item">${escapeHtml(row.path.split("/").pop() ?? row.path)}<em>${row.score ?? row.hits ?? ""}</em></span>`,
            )
            .join("")
        : `<span class="cite-nav-empty">nothing found</span>`;
    } catch (error) {
      box.textContent = error instanceof Error ? error.message : String(error);
    }
    return;
  }
  const target = element.closest<HTMLElement>("[data-action]");
  const action = target?.dataset.action ?? "";
  if (action === "pick-vault") {
    void addVault();
    return;
  }
  if (action === "notes-sync") {
    // Apple Notes has no form: the button is a data-action, so it never went
    // through the form dispatcher and the click did nothing at all.
    setResult("notes", "reading Apple Notes…");
    void post<Record<string, number>>("/api/sync/notes", {})
      .then(async (response) => {
        setResult("notes", summarize(response));
        await loadStatus();
      })
      .catch((error: unknown) => {
        const message = error instanceof Error ? error.message : String(error);
        setResult("notes", message);
      });
    return;
  }
  if (action !== "llm-setup-mlx" && action !== "llm-setup-ollama") return;
  const kind = action === "llm-setup-mlx" ? "mlx" : "ollama";
  try {
    await post("/api/llm/setup", { kind });
    toast(kind === "mlx" ? "Downloading the MLX model…" : "Pulling the model with Ollama…");
    await loadStatus();
  } catch (error) {
    toast(error instanceof Error ? error.message : String(error));
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
    if (provider === "github" || provider === "msgraph") cancelDeviceFlow(provider);
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
      } else if (provider === "github") {
        setResult("github", "disconnected — the gh CLI login is ignored until you reconnect");
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
  if (action.endsWith("-device-start") && action !== "open-device-page") {
    void startDevice(action.replace("-device-start", ""));
    return;
  }
  if (
    action === "reveal-github" ||
    action === "reveal-gdrive" ||
    action === "reveal-msgraph"
  ) {
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
  if (action.endsWith("-device-cancel")) {
    cancelDeviceFlow(action.replace("-device-cancel", ""));
    renderSources();
    return;
  }
  if (action === "open-device-page") {
    const provider = target.dataset.provider ?? "github";
    const flow = deviceFlows[provider];
    if (flow) void openExternal(flow.verificationUri);
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

// --- eval ---------------------------------------------------------------------

$("feedback-golden").addEventListener("click", async () => {
  try {
    const payload = await get<{ rows: number; jsonl: string }>("/api/feedback/golden");
    if (!payload.rows) {
      toast("No rated answers yet — use the ▲/▼ buttons in Chat");
      return;
    }
    const blob = new Blob([payload.jsonl], { type: "application/x-ndjson" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "golden-feedback.jsonl";
    link.click();
    URL.revokeObjectURL(link.href);
    toast(`Exported ${payload.rows} rated answers`);
  } catch (error) {
    toast(error instanceof Error ? error.message : String(error));
  }
});

$("eval-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const golden = $<HTMLInputElement>("eval-golden").value.trim();
  const results = $("eval-results");
  if (!golden) return;
  results.innerHTML = `<p class="muted">Running eval — embedding every query locally…</p>`;
  try {
    const response = await post<{
      metrics: Record<string, number>;
      queries: Array<Record<string, unknown>>;
    }>("/api/eval", { golden });
    const metrics = response.metrics;
    const misses = response.queries.filter((row) => Number(row["recall@5"]) < 1);
    results.innerHTML = `
      <div class="score-sheet">
        ${["recall@5", "ndcg@10", "mrr@10"]
          .map(
            (key) => `<div class="score">
              <span class="score-value">${Number(metrics[key] ?? 0).toFixed(3)}</span>
              <span class="score-key">${key}</span>
            </div>`,
          )
          .join("")}
      </div>
      <p class="caption">${Number(metrics.queries ?? 0)} queries · ${
        misses.length
          ? `${misses.length} missing: ${misses
              .map((row) => escapeHtml(String(row.query)))
              .join(" · ")}`
          : "every query found its sources"
      }</p>`;
  } catch (error) {
    results.innerHTML = `<p class="muted">${escapeHtml(
      error instanceof Error ? error.message : String(error),
    )}</p>`;
  }
});

// --- memory -------------------------------------------------------------------

type Memory = { id: number; text: string; created_at: string };

async function loadMemories(): Promise<void> {
  try {
    const { memories } = await get<{ memories: Memory[] }>("/api/memories");
    $("memory-list").innerHTML = memories.length
      ? memories
          .map(
            (memory) => `<div class="memory-item">
              <span>${escapeHtml(memory.text)}</span>
              <button class="memory-drop" type="button" data-id="${memory.id}" aria-label="Forget this">×</button>
            </div>`,
          )
          .join("")
      : `<p class="caption">nothing remembered yet</p>`;
  } catch {
    $("memory-list").innerHTML = "";
  }
}

$("memory-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = $<HTMLInputElement>("memory-input");
  const text = input.value.trim();
  if (!text) return;
  try {
    const result = await post<{ duplicate?: boolean }>("/api/memories", { text });
    input.value = "";
    $("memory-status").textContent = result.duplicate ? "already remembered" : "remembered";
    await loadMemories();
  } catch (error) {
    $("memory-status").textContent = error instanceof Error ? error.message : String(error);
  }
});

$("memory-list").addEventListener("click", async (event) => {
  const button = (event.target as HTMLElement).closest<HTMLElement>(".memory-drop");
  if (!button) return;
  try {
    await post("/api/memories/delete", { id: Number(button.dataset.id) });
    await loadMemories();
  } catch (error) {
    toast(error instanceof Error ? error.message : String(error));
  }
});

$("memory-extract").addEventListener("click", async () => {
  $("memory-status").textContent = "reading the latest chat…";
  try {
    const result = await post<{ added: string[] }>("/api/memories/extract", {
      chat_id: currentChatId,
    });
    $("memory-status").textContent = result.added.length
      ? `saved ${result.added.length} note${result.added.length === 1 ? "" : "s"}`
      : "nothing durable found in the latest chat";
    await loadMemories();
  } catch (error) {
    $("memory-status").textContent = error instanceof Error ? error.message : String(error);
  }
});

// --- corrections --------------------------------------------------------------

type Correction = { id: number; question: string; answer: string; created_at: string };

async function loadCorrections(): Promise<void> {
  try {
    const { corrections } = await get<{ corrections: Correction[] }>("/api/corrections");
    $("correction-list").innerHTML = corrections.length
      ? corrections
          .map(
            (correction) => `<div class="correction-item">
              <div class="correction-text">
                <p class="correction-q">${escapeHtml(correction.question)}</p>
                <p class="correction-a">${escapeHtml(snippet(correction.answer, 180))}</p>
              </div>
              <button class="memory-drop" type="button" data-id="${correction.id}" aria-label="Remove this correction">×</button>
            </div>`,
          )
          .join("")
      : `<p class="caption">no corrections yet — use Fix under an answer</p>`;
  } catch {
    $("correction-list").innerHTML = "";
  }
}

$("correction-list").addEventListener("click", async (event) => {
  const button = (event.target as HTMLElement).closest<HTMLElement>(".memory-drop");
  if (!button) return;
  try {
    await post("/api/corrections/delete", { id: Number(button.dataset.id) });
    await loadCorrections();
  } catch (error) {
    toast(error instanceof Error ? error.message : String(error));
  }
});

// --- bookmarks + duplicates ----------------------------------------------------

type Bookmark = { url: string; path: string; source: string; indexed_at: string };
type DuplicateCluster = { paths: string[]; shared_chunks: number; ratio: number };
const MAX_DUP_CLUSTERS = 5;
const MAX_DUP_PATHS = 3;

async function loadBookmarks(): Promise<void> {
  const box = document.getElementById("bookmark-list");
  if (!box) return;
  try {
    const { pages } = await get<{ pages: Bookmark[] }>("/api/bookmarks");
    box.innerHTML = pages.length
      ? pages
          .map(
            (page) => `<div class="memory-item">
              <span class="bookmark-url" title="${escapeHtml(page.path)}">${escapeHtml(page.url)}</span>
              <span class="bookmark-tools">
                <button class="action" type="button" data-open-url="${escapeHtml(page.url)}">Open</button>
                <button class="action" type="button" data-refresh-url="${escapeHtml(page.url)}">Refresh</button>
              </span>
            </div>`,
          )
          .join("")
      : `<p class="caption">nothing saved yet</p>`;
  } catch {
    box.innerHTML = "";
  }
}

async function loadDuplicates(): Promise<void> {
  const box = document.getElementById("duplicate-list");
  if (!box) return;
  box.innerHTML = `<p class="caption">checking…</p>`;
  try {
    const { clusters } = await get<{ clusters: DuplicateCluster[] }>("/api/duplicates");
    if (!clusters.length) {
      box.innerHTML = `<p class="caption">no duplicates found — every document is unique here</p>`;
      return;
    }
    const shownClusters = clusters.slice(0, MAX_DUP_CLUSTERS);
    box.innerHTML =
      shownClusters
        .map((cluster) => {
          const shown = cluster.paths.slice(0, MAX_DUP_PATHS);
          const hidden = cluster.paths.length - shown.length;
          return `<div class="dup-cluster">
            <p class="dup-meta">${cluster.paths.length} copies · ${cluster.shared_chunks} shared chunks · ${Math.round(cluster.ratio * 100)}% overlap</p>
            <ul class="dup-paths">
              ${shown
                .map(
                  (path) =>
                    `<li><code class="cite-path" data-open-path="${escapeHtml(path)}" title="${escapeHtml(path)}">${escapeHtml(path)}</code></li>`,
                )
                .join("")}
            </ul>
            ${hidden > 0 ? `<p class="caption">+${hidden} more ${hidden === 1 ? "copy" : "copies"}</p>` : ""}
          </div>`;
        })
        .join("") +
      (clusters.length > shownClusters.length
        ? `<p class="caption">+${clusters.length - shownClusters.length} more duplicate groups</p>`
        : "");
  } catch (error) {
    box.innerHTML = `<p class="caption">${escapeHtml(error instanceof Error ? error.message : String(error))}</p>`;
  }
}

async function openUrl(url: string): Promise<void> {
  if (isTauri()) {
    try {
      const { openUrl: open } = await import("@tauri-apps/plugin-opener");
      await open(url);
      return;
    } catch (error) {
      toast(error instanceof Error ? error.message : String(error));
      return;
    }
  }
  window.open(url, "_blank");
}

// --- bookmarks + duplicates handlers -------------------------------------------

document.addEventListener("click", (event) => {
  const element = event.target as HTMLElement;
  const openTarget = element.closest<HTMLElement>("[data-open-url]");
  if (openTarget?.dataset.openUrl) {
    void openUrl(openTarget.dataset.openUrl);
    return;
  }
  const refresh = element.closest<HTMLElement>("[data-refresh-url]");
  if (refresh?.dataset.refreshUrl) {
    const url = refresh.dataset.refreshUrl;
    refresh.textContent = "saving…";
    void post<{ indexed: number; unchanged: number }>("/api/save", { url })
      .then(async (result) => {
        toast(result.indexed ? `refreshed ${url}` : `already current: ${url}`);
        await loadBookmarks();
        await loadStatus();
      })
      .catch((error: unknown) => {
        toast(error instanceof Error ? error.message : String(error));
        refresh.textContent = "Refresh";
      });
  }
});

// --- health + never-index + backups --------------------------------------------

type Health = {
  documents: number;
  chunks: number;
  db_bytes: number;
  embedder: { index: string; current: string; matches: boolean };
  last_index: {
    at?: string;
    roots?: string[];
    files_scanned?: number;
    indexed?: number;
    unchanged?: number;
    skipped?: number;
    chunks?: number;
    skipped_samples?: Array<{ path: string; reason: string }>;
  };
  oldest: Array<{ path: string; mtime: number; indexed_at: string }>;
  never_index: { patterns: string[]; indexed_matches: string[] };
};

function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`;
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${bytes} B`;
}

function shortWhen(stamp: number | string): string {
  const date = typeof stamp === "number" ? new Date(stamp * 1000) : new Date(`${stamp}Z`);
  return Number.isNaN(date.getTime()) ? String(stamp).slice(0, 16) : date.toISOString().slice(0, 10);
}

async function loadHealth(): Promise<void> {
  const box = $("health-body");
  try {
    const health = await get<Health>("/api/health");
    const run = health.last_index;
    const samples = run.skipped_samples ?? [];
    const reasons = new Map<string, number>();
    for (const sample of samples) {
      reasons.set(sample.reason, (reasons.get(sample.reason) ?? 0) + 1);
    }
    const reasonLine = [...reasons.entries()]
      .map(([reason, count]) => `${count} ${reason}`)
      .join(" · ");
    box.innerHTML = `
      <p class="health-line ${health.embedder.matches ? "is-ok" : "is-warn"}">
        <span class="health-dot"></span>
        embedder ${escapeHtml(health.embedder.index || "unset")}${
          health.embedder.matches ? "" : ` — index built with ${escapeHtml(health.embedder.index)}, now ${escapeHtml(health.embedder.current)}`
        }
      </p>
      <p class="health-line">
        <span class="health-dot"></span>
        ${
          run.at
            ? `last local run ${escapeHtml(run.at)} · ${run.indexed ?? 0} indexed · ${run.unchanged ?? 0} unchanged · ${run.skipped ?? 0} skipped · +${run.chunks ?? 0} chunks`
            : "no local index run recorded yet"
        }
      </p>
      ${
        run.skipped
          ? `<details class="health-more">
              <summary>${run.skipped} skipped${reasonLine ? ` — ${escapeHtml(reasonLine)}` : ""}</summary>
              ${
                samples.length
                  ? `<ul class="health-list">${samples
                      .map(
                        (sample) =>
                          `<li><code data-open-path="${escapeHtml(sample.path)}" title="${escapeHtml(sample.path)}">${escapeHtml(sample.path.split("/").pop() ?? sample.path)}</code><span>${escapeHtml(sample.reason)}</span></li>`,
                      )
                      .join("")}</ul>${run.skipped > samples.length ? `<p class="caption">showing the first ${samples.length}</p>` : ""}`
                  : ""
              }
            </details>`
          : ""
      }
      ${
        health.oldest.length
          ? `<details class="health-more">
              <summary>oldest ${health.oldest.length} documents</summary>
              <ul class="health-list">${health.oldest
                .map(
                  (doc) =>
                    `<li><span class="health-when">${escapeHtml(shortWhen(doc.mtime))}</span><code data-open-path="${escapeHtml(doc.path)}" title="${escapeHtml(doc.path)}">${escapeHtml(doc.path)}</code></li>`,
                )
                .join("")}</ul>
            </details>`
          : ""
      }
      ${
        health.never_index.indexed_matches.length
          ? `<p class="health-line is-warn"><span class="health-dot"></span>${health.never_index.indexed_matches.length} document(s) still match a never-index pattern — save the patterns again to remove them</p>`
          : ""
      }
      <p class="health-line"><span class="health-dot"></span>${health.documents} documents · ${health.chunks} chunks · ${formatBytes(health.db_bytes)} on disk</p>`;
  } catch (error) {
    box.innerHTML = `<p class="caption">${escapeHtml(error instanceof Error ? error.message : String(error))}</p>`;
  }
}

async function loadNeverIndex(): Promise<void> {
  const box = $("never-index-list");
  try {
    const health = await get<Health>("/api/health");
    const patterns = health.never_index.patterns;
    box.innerHTML = patterns.length
      ? patterns
          .map(
            (pattern) => `<div class="memory-item">
              <span class="bookmark-url">${escapeHtml(pattern)}</span>
              <button class="memory-drop" type="button" data-never="${escapeHtml(pattern)}" aria-label="Stop this pattern">×</button>
            </div>`,
          )
          .join("")
      : `<p class="caption">nothing excluded</p>`;
  } catch {
    box.innerHTML = "";
  }
}

async function saveNeverIndex(patterns: string[]): Promise<void> {
  const result = await post<{ removed: string[] }>("/api/never-index", { patterns });
  $("never-index-status").textContent = result.removed.length
    ? `removed ${result.removed.length} document(s) from the index`
    : "patterns saved";
  await loadNeverIndex();
  await loadHealth();
  await loadStatus();
}

$("never-index-form")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = $<HTMLInputElement>("never-index-input");
  const pattern = input.value.trim();
  if (!pattern) return;
  try {
    const health = await get<Health>("/api/health");
    if (!health.never_index.patterns.includes(pattern)) {
      await saveNeverIndex([...health.never_index.patterns, pattern]);
    }
    input.value = "";
  } catch (error) {
    $("never-index-status").textContent =
      error instanceof Error ? error.message : String(error);
  }
});

$("never-index-list")?.addEventListener("click", async (event) => {
  const button = (event.target as HTMLElement).closest<HTMLElement>("[data-never]");
  if (!button) return;
  try {
    const health = await get<Health>("/api/health");
    await saveNeverIndex(
      health.never_index.patterns.filter((pattern) => pattern !== button.dataset.never),
    );
    $("never-index-status").textContent = "pattern removed — matching files return on the next index run";
  } catch (error) {
    $("never-index-status").textContent =
      error instanceof Error ? error.message : String(error);
  }
});

async function loadBackups(): Promise<void> {
  const box = $("backup-list");
  if (!box) return;
  try {
    const { backups } = await get<{ backups: Array<{ path: string; name: string; bytes: number; at: string }> }>(
      "/api/backups",
    );
    box.innerHTML = backups.length
      ? backups
          .map(
            (backup) => `<div class="memory-item">
              <span class="bookmark-url" title="${escapeHtml(backup.path)}">${escapeHtml(backup.at)} · ${formatBytes(backup.bytes)}</span>
              <span class="bookmark-tools">
                <button class="action" type="button" data-restore="${escapeHtml(backup.path)}">Restore</button>
              </span>
            </div>`,
          )
          .join("")
      : `<p class="caption">no backups yet</p>`;
  } catch {
    box.innerHTML = "";
  }
}

$("backup-now")?.addEventListener("click", async () => {
  $("backup-status").textContent = "working…";
  try {
    const result = await post<{ path: string; bytes: number }>("/api/backup", {});
    $("backup-status").textContent = `saved ${formatBytes(result.bytes)} — ${result.path.split("/").pop()}`;
    await loadBackups();
  } catch (error) {
    $("backup-status").textContent = error instanceof Error ? error.message : String(error);
  }
});

$("backup-list")?.addEventListener("click", async (event) => {
  const button = (event.target as HTMLElement).closest<HTMLElement>("[data-restore]");
  if (!button?.dataset.restore) return;
  const name = button.dataset.restore.split("/").pop();
  if (!window.confirm(`Restore the index from ${name}? A safety copy of the current index is kept.`)) return;
  $("backup-status").textContent = "restoring…";
  try {
    await post("/api/restore", { path: button.dataset.restore });
    window.location.reload();
  } catch (error) {
    $("backup-status").textContent = error instanceof Error ? error.message : String(error);
  }
});

type TopicCluster = { label: string; documents: number; paths: string[] };
type Refusal = {
  question: string;
  count: number;
  last_seen: string;
  resolved: boolean;
  best_hit?: string;
  best_cosine?: number;
};
const MAX_TOPIC_PATHS = 5;
const MAX_TOPIC_CLUSTERS = 15;

async function loadRefusals(): Promise<void> {
  const box = document.getElementById("refusal-list");
  if (!box) return;
  box.innerHTML = `<p class="caption">reading the transcripts…</p>`;
  try {
    const { rows } = await get<{ rows: Refusal[]; total: number }>("/api/refusals?probe=1");
    if (!rows.length) {
      box.innerHTML = `<p class="caption">nothing yet — every question so far found its sources</p>`;
      return;
    }
    box.innerHTML = rows
      .slice(0, 10)
      .map((row) => {
        const state = row.resolved
          ? `<span class="health-line is-ok"><span class="health-dot"></span>answered since</span>`
          : (row.best_cosine ?? 0) >= 0.45 && row.best_hit
            ? `<span class="caption">closest now: <code data-open-path="${escapeHtml(row.best_hit)}">${escapeHtml(row.best_hit.split("/").pop() ?? row.best_hit)}</code> (${row.best_cosine?.toFixed(2)}) — worth asking again</span>`
            : `<span class="caption">still nothing — add or connect a source</span>`;
        return `<div class="correction-item">
          <div class="correction-text">
            <p class="correction-q">${escapeHtml(row.question)}</p>
            <p class="correction-a">${state}</p>
          </div>
          <span class="dup-meta">${row.count}×</span>
        </div>`;
      })
      .join("");
  } catch (error) {
    box.innerHTML = `<p class="caption">${escapeHtml(error instanceof Error ? error.message : String(error))}</p>`;
  }
}

async function loadTopics(): Promise<void> {
  const box = document.getElementById("topic-list");
  if (!box) return;
  box.innerHTML = `<p class="caption">grouping documents…</p>`;
  try {
    const { clusters } = await get<{ clusters: TopicCluster[] }>("/api/topics");
    if (!clusters.length) {
      box.innerHTML = `<p class="caption">nothing indexed yet</p>`;
      return;
    }
    const multi = clusters.filter((cluster) => cluster.documents >= 2);
    const singles = clusters.filter((cluster) => cluster.documents < 2);
    const pool = multi.length ? multi : clusters;
    const shown = pool.slice(0, MAX_TOPIC_CLUSTERS);
    const hiddenClusters = pool.length - shown.length;
    box.innerHTML =
      shown
        .map((cluster, index) => {
          const paths = cluster.paths.slice(0, MAX_TOPIC_PATHS);
          const hidden = cluster.paths.length - paths.length;
          return `<details class="topic-cluster" ${index < 3 ? "open" : ""}>
            <summary><strong>${escapeHtml(cluster.label || "(mixed)")}</strong> <span>${cluster.documents} document${cluster.documents === 1 ? "" : "s"}</span></summary>
            <ul class="dup-paths">
              ${paths
                .map(
                  (path) =>
                    `<li><code class="cite-path" data-open-path="${escapeHtml(path)}" title="${escapeHtml(path)}">${escapeHtml(path)}</code></li>`,
                )
                .join("")}
            </ul>
            ${hidden > 0 ? `<p class="caption">+${hidden} more document${hidden === 1 ? "" : "s"}</p>` : ""}
          </details>`;
        })
        .join("") +
      (hiddenClusters > 0
        ? `<p class="caption">+${hiddenClusters} more clusters</p>`
        : "") +
      (multi.length && singles.length
        ? `<details class="topic-cluster">
            <summary><strong>${singles.length} single-document topic${singles.length === 1 ? "" : "s"}</strong></summary>
            <p class="caption">${singles.map((cluster) => escapeHtml(cluster.label || "(mixed)")).join(" · ")}</p>
          </details>`
        : "");
  } catch (error) {
    box.innerHTML = `<p class="caption">${escapeHtml(error instanceof Error ? error.message : String(error))}</p>`;
  }
}

type Bundle = { path: string; name: string; bytes: number; at: string };

async function loadBundles(): Promise<void> {
  const box = $("bundle-list");
  if (!box) return;
  try {
    const { bundles } = await get<{ bundles: Bundle[] }>("/api/bundles");
    box.innerHTML = bundles.length
      ? bundles
          .map(
            (bundle) => `<div class="memory-item">
              <span class="bookmark-url" title="${escapeHtml(bundle.path)}">${escapeHtml(bundle.name)} · ${formatBytes(bundle.bytes)}</span>
              <span class="bookmark-tools">
                <button class="action" type="button" data-import="${escapeHtml(bundle.path)}">Import</button>
              </span>
            </div>`,
          )
          .join("")
      : `<p class="caption">no bundles yet</p>`;
  } catch {
    box.innerHTML = "";
  }
}

$("bundle-export")?.addEventListener("click", async () => {
  $("bundle-status").textContent = "writing the bundle…";
  try {
    const result = await post<{ path: string; bytes: number; documents: number }>(
      "/api/export",
      {},
    );
    $("bundle-status").textContent = `exported ${result.documents} documents — ${result.path}`;
    await loadBundles();
  } catch (error) {
    $("bundle-status").textContent = error instanceof Error ? error.message : String(error);
  }
});

$("bundle-list")?.addEventListener("click", async (event) => {
  const button = (event.target as HTMLElement).closest<HTMLElement>("[data-import]");
  if (!button?.dataset.import) return;
  const name = button.dataset.import.split("/").pop();
  if (!window.confirm(`Replace this index with ${name}? A safety snapshot of the current index is kept.`)) return;
  $("bundle-status").textContent = "importing…";
  try {
    const result = await post<{ documents: number; safety_backup: string }>("/api/import", {
      path: button.dataset.import,
    });
    $("bundle-status").textContent = `imported ${result.documents} documents — reloading`;
    window.location.reload();
  } catch (error) {
    $("bundle-status").textContent = error instanceof Error ? error.message : String(error);
  }
});

function renderAnswerActions(
  message: HTMLElement,
  actions: HTMLElement,
  answerId: number,
  feedbackValue: number,
  question: string,
): void {
  if (!answerId) {
    actions.innerHTML = "";
    return;
  }
  actions.innerHTML = `
    <button class="action" type="button" data-thumb="1" aria-label="Good answer" class="${feedbackValue === 1 ? "is-on" : ""}">▲</button>
    <button class="action" type="button" data-thumb="-1" aria-label="Bad answer" class="${feedbackValue === -1 ? "is-on" : ""}">▼</button>
    <button class="action action-text" type="button" data-verify="1">Verify</button>
    ${question ? `<button class="action action-text" type="button" data-fix="1">Fix</button>` : ""}
    <span class="action-note" data-note></span>`;
  actions.querySelectorAll<HTMLElement>("[data-thumb]").forEach((button) => {
    button.addEventListener("click", async () => {
      const value =
        String(button.dataset.thumb) === "1" ? (feedbackValue === 1 ? 0 : 1) : feedbackValue === -1 ? 0 : -1;
      try {
        await post("/api/feedback", { message_id: answerId, value });
        feedbackValue = value;
        renderAnswerActions(message, actions, answerId, value, question);
        const note = actions.querySelector<HTMLElement>("[data-note]");
        if (note) note.textContent = value === 0 ? "" : "noted — thanks";
      } catch (error) {
        toast(error instanceof Error ? error.message : String(error));
      }
    });
  });
  const verify = actions.querySelector<HTMLElement>("[data-verify]");
  verify?.addEventListener("click", async () => {
    try {
      const verdict = await post<{
        grounded_ratio: number;
        citation_valid: boolean;
        sentences_detail: Array<{ text: string; ratio: number; grounded: boolean }>;
      }>("/api/verify", { message_id: answerId });
      const answer = message.querySelector<HTMLElement>(".answer");
      const original = answer?.textContent ?? "";
      if (answer) {
        answer.innerHTML = verdict.sentences_detail
          .map(
            (row) =>
              `<span class="sentence ${row.grounded ? "is-grounded" : "is-loose"}" title="grounded ${Math.round(row.ratio * 100)}%">${escapeHtml(row.text)}</span>`,
          )
          .join(" ");
      }
      const note = actions.querySelector<HTMLElement>("[data-note]");
      if (note) {
        note.textContent = `${Math.round(verdict.grounded_ratio * 100)}% grounded · citations ${verdict.citation_valid ? "valid" : "unclear"}`;
      }
      window.setTimeout(() => {
        if (answer) answer.textContent = original;
      }, 12000);
    } catch (error) {
      toast(error instanceof Error ? error.message : String(error));
    }
  });
  const fix = actions.querySelector<HTMLElement>("[data-fix]");
  fix?.addEventListener("click", () => {
    openFixEditor(message, actions, answerId, feedbackValue, question);
  });
}

function openFixEditor(
  message: HTMLElement,
  actions: HTMLElement,
  answerId: number,
  feedbackValue: number,
  question: string,
): void {
  const current = message.querySelector<HTMLElement>(".answer")?.textContent ?? "";
  const editor = document.createElement("div");
  editor.className = "fix-editor";
  editor.innerHTML = `
    <textarea class="fix-input" rows="5" aria-label="Corrected answer"></textarea>
    <div class="fix-row">
      <button class="action action-text is-on" type="button" data-save>Save correction</button>
      <button class="action action-text" type="button" data-cancel>Cancel</button>
      <span class="action-note" data-note></span>
    </div>`;
  const input = editor.querySelector<HTMLTextAreaElement>(".fix-input") as HTMLTextAreaElement;
  input.value = current;
  actions.innerHTML = "";
  actions.append(editor);
  input.focus();
  const restore = (note: string) => {
    renderAnswerActions(message, actions, answerId, feedbackValue, question);
    const target = actions.querySelector<HTMLElement>("[data-note]");
    if (target && note) target.textContent = note;
  };
  editor.querySelector<HTMLElement>("[data-save]")?.addEventListener("click", async () => {
    const corrected = input.value.trim();
    if (!corrected) return;
    try {
      await post("/api/corrections", { question, answer: corrected });
      void loadCorrections();
      restore("correction saved — matching questions reuse it");
    } catch (error) {
      toast(error instanceof Error ? error.message : String(error));
    }
  });
  editor.querySelector<HTMLElement>("[data-cancel]")?.addEventListener("click", () => restore(""));
}

async function autoVerify(actions: HTMLElement, answerId: number): Promise<void> {
  const note = actions.querySelector<HTMLElement>("[data-note]");
  if (!note) return;
  try {
    const verdict = await post<{ grounded_ratio: number; citation_valid: boolean }>(
      "/api/verify",
      { message_id: answerId },
    );
    note.textContent = `${Math.round(verdict.grounded_ratio * 100)}% grounded${
      verdict.citation_valid ? "" : " · citations unclear"
    }`;
  } catch {
    /* the manual Verify button still works */
  }
}

const CITE_REF_RE = /\[(\d+)\]/g;

function renderRichText(element: HTMLElement, text: string): void {
  // A deliberately tiny renderer: paragraphs, "- " bullets, **bold** and
  // clickable [n] references. Everything is escaped first, so no model output
  // can inject markup.
  const lines = escapeHtml(text).split("\n");
  const html: string[] = [];
  let inList = false;
  const inline = (value: string) =>
    value
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(CITE_REF_RE, '<sup class="cite-ref" data-cite="$1">$1</sup>');

  for (const raw of lines) {
    const line = raw.trimEnd();
    if (/^\s*-\s+/.test(line)) {
      if (!inList) {
        html.push("<ul>");
        inList = true;
      }
      html.push(`<li>${inline(line.replace(/^\s*-\s+/, ""))}</li>`);
      continue;
    }
    if (inList) {
      html.push("</ul>");
      inList = false;
    }
    if (line.trim()) html.push(`<p>${inline(line)}</p>`);
  }
  if (inList) html.push("</ul>");
  element.innerHTML = html.join("");
}

document.addEventListener("click", (event) => {
  const ref = (event.target as HTMLElement).closest<HTMLElement>(".cite-ref");
  if (!ref) return;
  const index = Number(ref.dataset.cite ?? 0);
  const card = document.querySelector<HTMLElement>(`.cite[data-rank="${index}"]`);
  if (card) {
    card.scrollIntoView({ block: "center", behavior: "smooth" });
    card.classList.add("is-flash");
    window.setTimeout(() => card.classList.remove("is-flash"), 1200);
  }
});

// --- diagrams -----------------------------------------------------------------

const MERMAID_FENCE = /```mermaid\s*\n([\s\S]*?)```/g;

async function renderDiagrams(
  answer: HTMLElement,
  container: HTMLElement,
  text: string,
): Promise<void> {
  const blocks = [...text.matchAll(MERMAID_FENCE)];
  if (blocks.length === 0) return;
  // The prose stays; the fence becomes the rendered figure below it.
  answer.textContent = text.replace(MERMAID_FENCE, "").trim();
  const { default: mermaid } = await import("mermaid");
  const styles = getComputedStyle(document.documentElement);
  const token = (name: string) => styles.getPropertyValue(name).trim();
  mermaid.initialize({
    startOnLoad: false,
    securityLevel: "strict",
    theme: "base",
    fontFamily: token("--font-sans") || "system-ui, sans-serif",
    themeVariables: {
      background: token("--card") || "#f6f7f2",
      primaryColor: token("--paper") || "#e6e8e0",
      primaryTextColor: token("--ink") || "#1d2830",
      primaryBorderColor: token("--rule") || "#c8cdc0",
      secondaryColor: token("--card") || "#f6f7f2",
      tertiaryColor: token("--paper") || "#e6e8e0",
      lineColor: token("--ink-soft") || "#57646c",
      textColor: token("--ink") || "#1d2830",
      fontSize: "13px",
    },
  });

  for (const [index, match] of blocks.entries()) {
    // Small models slip on keywords; repair the common ones before rendering.
    const code = match[1]
      .trim()
      .replace(/\bsubregion\b/g, "subgraph")
      .replace(/^\s*(flowchart|graph)\s+graph\b/gm, "$1 TD");
    const figure = document.createElement("figure");
    figure.className = "diagram";
    container.append(figure);
    try {
      const { svg } = await mermaid.render(`ragdesk-diagram-${Date.now()}-${index}`, code);
      figure.innerHTML = svg;
      const toolbar = document.createElement("div");
      toolbar.className = "diagram-tools";
      const download = document.createElement("button");
      download.type = "button";
      download.className = "btn btn-quiet";
      download.textContent = "Download SVG";
      download.addEventListener("click", () => {
        const blob = new Blob([figure.querySelector("svg")?.outerHTML ?? ""], {
          type: "image/svg+xml",
        });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = `ragdesk-diagram-${index + 1}.svg`;
        link.click();
        URL.revokeObjectURL(link.href);
      });
      const copy = document.createElement("button");
      copy.type = "button";
      copy.className = "btn btn-quiet";
      copy.textContent = "Copy Mermaid";
      copy.addEventListener("click", () => void navigator.clipboard.writeText(code));
      toolbar.append(download, copy);
      figure.append(toolbar);
    } catch {
      const pre = document.createElement("pre");
      pre.className = "diagram-source";
      pre.textContent = code;
      figure.append(pre);
    }
  }
}

// --- setup wizard -------------------------------------------------------------

type WizardStep = {
  title: string;
  lede: string;
  render: () => string;
  bind?: () => void;
  nextLabel?: string;
};

let wizardStep = 0;
let wizardOpen = false;

function wizardOpenForm(open: boolean): void {
  wizardOpen = open;
  $("wizard").hidden = !open;
  if (open) renderWizard();
}

function renderWizard(): void {
  if (!status) return;
  const steps = wizardSteps();
  const step = steps[Math.min(wizardStep, steps.length - 1)];
  $("wizard-title").textContent = step.title;
  $("wizard-lede").textContent = step.lede;
  $("wizard-body").innerHTML = step.render();
  $("wizard-steps").innerHTML = steps
    .map(
      (_entry, index) =>
        `<li class="${index === wizardStep ? "is-current" : index < wizardStep ? "is-done" : ""}">${index + 1}</li>`,
    )
    .join("");
  $("wizard-back").hidden = wizardStep === 0;
  $<HTMLButtonElement>("wizard-next").textContent =
    step.nextLabel ?? (wizardStep === steps.length - 1 ? "Finish" : "Continue");
  step.bind?.();
}

function wizardFolders(): string {
  const list = selectedPaths.length
    ? selectedPaths.map((path) => `<li><code>${escapeHtml(path)}</code></li>`).join("")
    : `<li class="muted">No folders chosen yet — you can add them later in Sources.</li>`;
  return `
    <p class="wizard-hint">Pick the folders ragdesk should keep indexed. Everything is read-only.</p>
    <div class="button-row">
      <button class="btn" type="button" id="wizard-pick-folder">Choose folder…</button>
      <button class="btn btn-quiet" type="button" id="wizard-pick-files">Choose files…</button>
      ${
        (status?.system.screenshot_dirs ?? [])
          .map(
            (dir) =>
              `<button class="btn btn-quiet" type="button" data-watch="${escapeHtml(dir)}">Watch screenshots (${escapeHtml(dir.split("/").pop() ?? "")})</button>`,
          )
          .join("")
      }
    </div>
    <ul class="wizard-list">${list}</ul>`;
}

function wizardModel(): string {
  const options = status?.llm_setup;
  const rows: string[] = [];
  const activeKind = status?.llm.kind ?? "none";
  const mlxAction = options?.mlx_cached
    ? activeKind === "mlx"
      ? `<span class="meta-chip">active</span>`
      : `<button class="btn" type="button" data-use="mlx">Use MLX</button>`
    : `<button class="btn btn-primary" type="button" data-action="llm-setup-mlx">Download</button>`;
  if (options?.mlx_available && options?.mlx_repo) {
    rows.push(
      `<div class="wizard-option">
        <div><strong>Local model (MLX)</strong><p>${options?.mlx_cached ? "Runs in-process on Apple Silicon — already downloaded." : "Runs in-process on Apple Silicon. One-time download."}</p></div>
        ${mlxAction}
      </div>`,
    );
  }
  if (options?.ollama_reachable) {
    const ollamaAction = options?.ollama_has_model
      ? activeKind === "ollama"
        ? `<span class="meta-chip">active</span>`
        : `<button class="btn" type="button" data-use="ollama">Use Ollama</button>`
      : `<button class="btn btn-primary" type="button" data-action="llm-setup-ollama">Pull model</button>`;
    rows.push(
      `<div class="wizard-option">
        <div><strong>Ollama</strong><p>${options?.ollama_has_model ? `${escapeHtml(options?.ollama_model ?? "")} is already pulled.` : `Pull ${escapeHtml(options?.ollama_model ?? "")} through your Ollama.`}</p></div>
        ${ollamaAction}
      </div>`,
    );
  }
  rows.push(
    `<div class="wizard-option">
      <div><strong>OpenAI-compatible endpoint</strong><p>LM Studio, llama.cpp, vLLM or OpenAI. Configure it in Settings when you need it.</p></div>
      <button class="btn btn-quiet" type="button" id="wizard-openai">Set up later</button>
    </div>`,
  );
  const job = status?.llm_setup.job;
  const running = job?.running
    ? `<div class="progress"><div class="progress-bar" style="width:${Math.round((job.progress ?? 0) * 100)}%"></div></div>
       <p class="caption">${escapeHtml(job.detail || "working…")} · ${Math.round((job.progress ?? 0) * 100)}%</p>`
    : "";
  return `
    <p class="wizard-hint">Chat needs a model. Indexing and search already work without one.</p>
    ${rows.join("")}
    ${running}
    <p class="caption" id="wizard-model-state">active: ${escapeHtml(llmLabel(status as Status))}</p>`;
}

function wizardPreset(): string {
  const suggested = status?.system.suggested_preset ?? "light";
  const current = status?.preset ?? "light";
  const note = (status?.presets ?? []).find((entry) => entry.name === current)?.note ?? "";
  return `
    <p class="wizard-hint">This machine has ${status?.system.ram_gb || "?"} GB of RAM — <strong>${suggested}</strong> fits it best. You are on <strong>${escapeHtml(current)}</strong>.</p>
    <div class="seg" id="wizard-preset-seg" role="group" aria-label="Machine preset">
      <button class="seg-item ${current === "light" ? "is-active" : ""}" type="button" data-value="light">Light</button>
      <button class="seg-item ${current === "balanced" ? "is-active" : ""}" type="button" data-value="balanced">Balanced</button>
      <button class="seg-item ${current === "quality" ? "is-active" : ""}" type="button" data-value="quality">Quality</button>
    </div>
    <p class="caption" id="wizard-preset-note">${escapeHtml(note)}</p>`;
}

function wizardIndex(): string {
  const activity = status?.activity;
  const running = activity?.running ?? false;
  const progress = running
    ? `<div class="progress"><div class="progress-bar" style="width:${
        activity?.total ? Math.round(((activity?.done ?? 0) / activity.total) * 100) : 15
      }%"></div></div>
       <p class="caption">${escapeHtml(activity?.detail ?? "working…")}${
         activity?.done ? ` (${activity.done}${activity.total ? "/" + activity.total : ""})` : ""
       }</p>`
    : "";
  return `
    <p class="wizard-hint">Ready. Index now and the first answers are a minute or two away — you can keep using Chat while it runs.</p>
    <ul class="wizard-list">
      <li>${selectedPaths.length} folder${selectedPaths.length === 1 ? "" : "s"} chosen</li>
      <li>preset <strong>${escapeHtml(status?.preset ?? "light")}</strong> · engine <strong>${escapeHtml(llmLabel(status as Status))}</strong></li>
      <li id="wizard-index-counts">${status?.documents ?? 0} documents indexed so far</li>
    </ul>
    <div class="button-row">
      <button class="btn btn-primary" type="button" id="wizard-index" ${running ? "disabled" : ""}>${
        running ? "Indexing…" : "Index now"
      }</button>
    </div>
    ${progress}
    <p class="caption" id="wizard-index-state"></p>`;
}

function wizardSteps(): WizardStep[] {
  return [
    {
      title: "Welcome to ragdesk",
      lede: "Your own knowledge base, running entirely on this machine.",
      render: () => `
        <ul class="wizard-list">
          <li>Index folders, PDFs, Office files, screenshots and code.</li>
          <li>Ask questions, get answers with citations — nothing leaves this machine.</li>
          <li>Works offline once your model is downloaded.</li>
        </ul>`,
      nextLabel: "Get started",
    },
    {
      title: "Choose what to index",
      lede: "Folders are read-only and can be changed any time.",
      render: wizardFolders,
      bind: () => {
        $("wizard-pick-folder").addEventListener("click", () => void pickPaths("folder"));
        $("wizard-pick-files").addEventListener("click", () => void pickPaths("files"));
        document
          .querySelectorAll<HTMLElement>("#wizard-body [data-watch]")
          .forEach((button) =>
            button.addEventListener("click", () => {
              const dir = button.dataset.watch ?? "";
              if (dir && !selectedPaths.includes(dir)) selectedPaths.push(dir);
              renderSources();
              renderWizard();
            }),
          );
      },
    },
    {
      title: "Pick an answer engine",
      lede: "ragdesk reuses what you already have, or downloads a model for you.",
      render: wizardModel,
      bind: () => {
        $("wizard-openai").addEventListener("click", () => {
          wizardOpenForm(false);
          activateTab("settings");
          markSeg("backend-seg", "openai");
          $<HTMLFormElement>("openai-form").hidden = false;
          $("openai-form").scrollIntoView({ block: "center" });
        });
      },
    },
    {
      title: "Size it to this machine",
      lede: "Bigger presets use a larger model and a reranker. Nothing is re-indexed.",
      render: wizardPreset,
      bind: () => {
        document.querySelectorAll<HTMLElement>("#wizard-preset-seg .seg-item").forEach((item) => {
          item.addEventListener("click", () => {
            markSeg("wizard-preset-seg", item.dataset.value ?? "");
            void saveSetting({ preset: item.dataset.value ?? "light" });
          });
        });
      },
    },
    {
      title: "Index and go",
      lede: "You can re-run this wizard from Settings whenever you like.",
      render: wizardIndex,
      bind: () => {
        $("wizard-finish").addEventListener("click", () => void finishWizard());
        $("wizard-index").addEventListener("click", async () => {
          if (selectedPaths.length === 0) {
            $("wizard-index-state").textContent =
              "No folders chosen — go back and pick one, or add them later in Sources.";
            return;
          }
          $("wizard-index-state").textContent = "Indexing — watching progress above…";
          startActivityPolling();
          try {
            const result = await post<Record<string, number>>("/api/index", {
              paths: selectedPaths,
            });
            selectedPaths = [];
            renderSources();
            await loadStatus();
            const skipped = Number(result.skipped ?? 0);
            $("wizard-index-state").textContent =
              `Indexed ${result.indexed ?? 0} files (${result.chunks ?? 0} chunks)` +
              (skipped ? ` — ${skipped} skipped (unsupported or no text)` : "") +
              ". Ask something in Chat.";
          } catch (error) {
            $("wizard-index-state").textContent =
              error instanceof Error ? error.message : String(error);
          }
        });
      },
      nextLabel: "Finish",
    },
  ];
}

async function finishWizard(): Promise<void> {
  wizardOpenForm(false);
  try {
    await post("/api/settings", { onboarded: true });
    await loadStatus();
    await loadChats();
  } catch {
    /* onboarding is a nicety; never block the app on it */
  }
}

$("run-wizard").addEventListener("click", () => {
  wizardStep = 0;
  wizardOpenForm(true);
});
$("wizard-skip").addEventListener("click", () => void finishWizard());
$("wizard-back").addEventListener("click", () => {
  wizardStep = Math.max(0, wizardStep - 1);
  renderWizard();
});
$<HTMLButtonElement>("wizard-next").addEventListener("click", async () => {
  const steps = wizardSteps();
  if (wizardStep >= steps.length - 1) {
    await finishWizard();
    return;
  }
  wizardStep += 1;
  renderWizard();
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
    void loadMemories();
    void loadCorrections();
    void loadNeverIndex();
    void loadBackups();
    void loadBundles();
    void loadMcp();
    if (!status.onboarded) {
      wizardStep = 0;
      wizardOpenForm(true);
    }
  } else {
    $("rail-meta").textContent = "server offline — start it with: ragdesk serve";
  }
}

// --- command palette + shortcuts ------------------------------------------------

type PaletteItem = {
  kind: string;
  title: string;
  sub?: string;
  run: () => void | Promise<void>;
};

const TABS = ["chat", "sources", "indexed", "settings"] as const;
let paletteResults: PaletteItem[] = [];
let paletteSelected = 0;
let paletteDocs: PaletteItem[] = [];
let paletteToken = 0;
let paletteTimer = 0;

function subsequence(query: string, text: string): boolean {
  let position = 0;
  for (const char of text) {
    if (char === query[position]) position += 1;
    if (position >= query.length) return true;
  }
  return false;
}

function paletteMatches(query: string, item: PaletteItem): boolean {
  if (!query) return true;
  const haystack = `${item.kind} ${item.title} ${item.sub ?? ""}`.toLowerCase();
  return haystack.includes(query) || subsequence(query, haystack);
}

function basePaletteItems(): PaletteItem[] {
  const items: PaletteItem[] = [
    { kind: "action", title: "New conversation", sub: "⌘N", run: () => newChat() },
    {
      kind: "action",
      title: "Toggle light / dark theme",
      run: () =>
        applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"),
    },
    {
      kind: "action",
      title: "Keyboard shortcuts",
      sub: "?",
      run: () => {
        $("shortcuts").hidden = false;
      },
    },
  ];
  TABS.forEach((tab, index) => {
    items.push({
      kind: "go",
      title: `Open ${tab[0].toUpperCase()}${tab.slice(1)}`,
      sub: `⌘${index + 1}`,
      run: () => activateTab(tab),
    });
  });
  for (const chat of chatCache) {
    items.push({
      kind: "conversation",
      title: chat.title,
      sub: relativeWhen(chat.updated_at),
      run: () => void openChat(chat.id),
    });
  }
  for (const source of status?.sources ?? []) {
    items.push({
      kind: "source",
      title: source.source,
      sub: `${source.documents} documents`,
      run: () => activateTab("indexed"),
    });
  }
  return items;
}

function renderPalette(): void {
  const list = $("palette-list");
  if (!paletteResults.length) {
    list.innerHTML = `<p class="palette-empty">nothing matches — try a file name, a source, or an action</p>`;
    return;
  }
  list.innerHTML = paletteResults
    .slice(0, 40)
    .map(
      (item, index) =>
        `<div class="palette-item${index === paletteSelected ? " is-selected" : ""}" data-index="${index}">
          <span class="palette-kind">${escapeHtml(item.kind)}</span>
          <span class="palette-title">${escapeHtml(item.title)}</span>
          ${item.sub ? `<span class="palette-sub">${escapeHtml(item.sub)}</span>` : ""}
        </div>`,
    )
    .join("");
  list.querySelector(".is-selected")?.scrollIntoView({ block: "nearest" });
}

function refreshPalette(): void {
  const query = $<HTMLInputElement>("palette-input").value.trim().toLowerCase();
  paletteResults = [...basePaletteItems(), ...paletteDocs].filter((item) =>
    paletteMatches(query, item),
  );
  paletteSelected = 0;
  renderPalette();
}

function movePalette(step: number): void {
  if (!paletteResults.length) return;
  paletteSelected = (paletteSelected + step + paletteResults.length) % paletteResults.length;
  renderPalette();
}

async function runPalette(index: number): Promise<void> {
  const item = paletteResults[index];
  closePalette();
  if (item) await item.run();
}

function openPalette(): void {
  $("palette").hidden = false;
  const input = $<HTMLInputElement>("palette-input");
  input.value = "";
  paletteDocs = [];
  refreshPalette();
  input.focus();
}

function closePalette(): void {
  $("palette").hidden = true;
  paletteToken += 1;
  window.clearTimeout(paletteTimer);
}

async function searchPaletteDocs(query: string): Promise<void> {
  const token = ++paletteToken;
  try {
    const response = await post<{ hits: Hit[] }>("/api/search", { query, top_k: 8 });
    if (token !== paletteToken) return;
    paletteDocs = (response.hits ?? []).map((hit) => ({
      kind: "document",
      title: hit.path.split("/").pop() ?? hit.path,
      sub: hit.path,
      run: () => void openLocal(hit.path),
    }));
  } catch {
    if (token !== paletteToken) return;
    paletteDocs = [];
  }
  if ($<HTMLInputElement>("palette-input").value.trim().toLowerCase() === query) {
    refreshPalette();
  }
}

$("palette-open").addEventListener("click", () => openPalette());

$("palette-input").addEventListener("input", () => {
  refreshPalette();
  window.clearTimeout(paletteTimer);
  const query = $<HTMLInputElement>("palette-input").value.trim().toLowerCase();
  if (query.length < 3) {
    paletteDocs = [];
    return;
  }
  paletteTimer = window.setTimeout(() => void searchPaletteDocs(query), 220);
});

$("palette-input").addEventListener("keydown", (event) => {
  if (event.key === "ArrowDown") {
    event.preventDefault();
    movePalette(1);
  } else if (event.key === "ArrowUp") {
    event.preventDefault();
    movePalette(-1);
  } else if (event.key === "Enter") {
    event.preventDefault();
    void runPalette(paletteSelected);
  } else if (event.key === "Escape") {
    event.preventDefault();
    closePalette();
  }
});

$("palette-list").addEventListener("click", (event) => {
  const row = (event.target as HTMLElement).closest<HTMLElement>("[data-index]");
  if (row) void runPalette(Number(row.dataset.index ?? 0));
});

$("palette").addEventListener("click", (event) => {
  if (event.target === $("palette")) closePalette();
});

$("shortcuts").addEventListener("click", (event) => {
  if (event.target === $("shortcuts")) $("shortcuts").hidden = true;
});

document.addEventListener("keydown", (event) => {
  const meta = event.metaKey || event.ctrlKey;
  const key = event.key.toLowerCase();
  if (meta && key === "k") {
    event.preventDefault();
    if ($("palette").hidden) openPalette();
    else closePalette();
    return;
  }
  if (meta && key === "n") {
    event.preventDefault();
    newChat();
    return;
  }
  if (meta && ["1", "2", "3", "4"].includes(key)) {
    event.preventDefault();
    activateTab(TABS[Number(key) - 1]);
    return;
  }
  if (event.key === "Escape") {
    if (!$("palette").hidden) {
      closePalette();
      return;
    }
    if (!$("shortcuts").hidden) {
      $("shortcuts").hidden = true;
      return;
    }
    if (streaming) activeAbort?.abort();
    return;
  }
  const target = event.target as HTMLElement | null;
  const typing = target ? ["INPUT", "TEXTAREA"].includes(target.tagName) : false;
  if (event.key === "?" && !typing) {
    event.preventDefault();
    $("shortcuts").hidden = false;
  }
});

void boot();

export {};
