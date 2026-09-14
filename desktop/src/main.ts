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

type SourceStat = { source: string; documents: number };

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

  const list = $("indexed-sources");
  if (status.sources.length === 0) {
    list.innerHTML = `<li class="muted">Nothing indexed yet. Add a source.</li>`;
  } else {
    list.innerHTML = status.sources
      .map(
        (entry) =>
          `<li><code>${escapeHtml(entry.source)}</code><span>${entry.documents} documents</span></li>`,
      )
      .join("");
  }
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

const SYNC_ENDPOINTS: Record<string, string> = {
  local: "/api/index",
  github: "/api/sync/github",
  confluence: "/api/sync/confluence",
  gdrive: "/api/sync/gdrive",
};

function renderSyncResult(element: HTMLElement, payload: Record<string, number>): void {
  const parts = ["scanned", "indexed", "unchanged", "skipped", "chunks"]
    .filter((key) => key in payload)
    .map((key) => `${key} ${payload[key]}`);
  element.textContent = parts.join(" · ") || "done";
}

document.querySelectorAll<HTMLFormElement>(".source-form").forEach((form) => {
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const kind = form.dataset.source ?? "";
    const result = form.querySelector<HTMLElement>(".source-result") as HTMLElement;
    const button = form.querySelector<HTMLButtonElement>("button");
    const data = new FormData(form);
    const payload: Record<string, unknown> = {};
    data.forEach((value, key) => {
      if (String(value).trim()) payload[key] = String(value).trim();
    });
    if (kind === "local") {
      payload.paths = String(payload.paths ?? "")
        .split(",")
        .map((part) => part.trim())
        .filter(Boolean);
      if ((payload.paths as string[]).length === 0) {
        result.textContent = "Add at least one path.";
        return;
      }
    }
    if (button) button.disabled = true;
    result.textContent = "working…";
    try {
      const response = await post<Record<string, number>>(SYNC_ENDPOINTS[kind], payload);
      renderSyncResult(result, response);
      await loadStatus();
    } catch (error) {
      result.textContent = error instanceof Error ? error.message : String(error);
      toast(result.textContent);
    } finally {
      if (button) button.disabled = false;
    }
  });
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
  } else {
    $("rail-meta").textContent = "server offline — start it with: ragdesk serve";
  }
}

void boot();

export {};
