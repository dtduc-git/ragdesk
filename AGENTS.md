# ragdesk — agent notes

Personal, local-first RAG over your own sources. Core is **stdlib-only Python**
(retrieval + eval); models are reached through a local Ollama server.

## Layout

- `src/ragdesk/` — package. `chunk` (paragraph-aware + overlap), `embed`
  (HashingEmbedder for CI, OllamaEmbedder, OnnxEmbedder = EmbeddingGemma int8
  with query/doc prompts; shared `tokenize`), `store` (SQLite: FTS5 + float32
  vectors + fail-closed embedder guard; keeps the chosen local roots in `meta`
  for the per-path Indexed breakdown via `local_paths()`; `duplicate_clusters`
  groups docs by exact chunk-hash containment — a copied or sliced file shares
  chunks, a topical neighbour does not, which document-level cosine cannot
  separate; `web_pages()` is the bookmark list), `search` (BM25 +
  dense + RRF; `retrieve` adds the optional rerank stage), `rerank`
  (LexicalReranker baseline; FastEmbedReranker + OnnxReranker = multilingual
  gte behind the `onnx` extra), `index` (incremental local files; `iter_files`
  prunes `SKIP_DIRS` during the walk so `target/`/`node_modules/` are never
  traversed; `index_document` takes optional `metadata` merged over any
  front-matter; one bad file becomes a skip with the reason, never a dead
  watcher), `office` (PDF/DOCX/PPTX/XLSX text extraction: docx/pptx/xlsx via
  zip+XML with zero deps — sheets keep `r<row>` refs, shared + inline strings;
  PDFs via pypdf, the only runtime dependency; XML with a DTD is refused;
  scanned PDFs return empty and are skipped, no OCR), `vision` (image OCR through Apple's
  on-device Vision framework via the `vision` extra: `en-US` + `vi-VT`, images
  upscaled 2× before recognition, header carries file name + Spotlight capture
  date; non-macOS or no extra → images skip as before), `github` / `gitlab` / `confluence` / `gdrive` / `notion` /
  `msgraph` (OneDrive + SharePoint, device flow) connectors, `email_source`
  (read-only email: `index_mbox` iterates one document per message,
  `sync_imap` uses `select(readonly=True)` + `BODY.PEEK` so nothing is ever
  marked read; headers + text body only, attachment names are listed not
  parsed; credentials under the `email` provider), `web` (same-host
  HTML crawl, capped pages/depth; `save_page` = one page, failures raise),
  `archive`
  (shared repo-tarball extraction), `htmlutil` (shared HTML→text),
  `evaluate` (recall@5 / nDCG@10 / MRR), `answer` (Ollama LLM: non-stream +
  stream, grounding gate), `credentials` (0600 store under `~/.config/ragdesk/`),
  `envfile` (.env loader for dev), `buildenv` (bakes .env into a gitignored
  `_build_env.py` for release builds), `mcp` (stdio MCP server exposing
  search/document/sources to Claude Code & co), `serve` (loopback JSON API:
  status/search/ask/index/sync + connections connect/disconnect + optional
  static UI; hosts the auto-index timer), `settings` (app config in
  `~/.config/ragdesk/settings.json`, 0600: `auto_index_hours`, default 1,
  0 = off), `llm` (answer backends + the no-double-download ladder: `auto`
  reuses a running Ollama that already has the preset model, else MLX
  in-process from the shared HF cache; `--llm`/`RAGDESK_LLM` override),
  `presets` (RAM tiers; each carries an Ollama tag and an `llm_mlx` repo;
  `POST /api/settings {preset}` applies a tier live — rerank and LLM swap, the
  shared embedder means no re-index; the choice persists in settings.json and
  beats the built-in default, with `--preset` still winning at launch),
  `cli`.
- UI identity (`desktop/src/styles.css`, tokens at the top): the "catalog
  drawer" — grey-green card stock (`--paper` desk, `--sheet` working sheet,
  `--card` insets), **violet library ink** (`--stamp`) for actions and stamps,
  **amber** (`--lamp`) for anything live or cited, Iowan/Charter display + SF
  interface + SF Mono for paths and figures. Structural devices carry meaning:
  the transcript is a ruled ledger with entry numbers in the gutter, the active
  rail tab is the pulled drawer, the Indexed stats are a ledger with
  proportional bars, and the composer is an index card with a violet top edge.
  Keep it restrained — hairlines and one shadow, no per-element shadows, no
  gradient decoration, and `prefers-reduced-motion` disables all motion.
  `main.ts` also owns the command palette (⌘K/⌘N/⌘1-4/?; documents come from
  `/api/search`), which must keep every `$("id")` it touches present in
  `index.html` (`tests/test_ui.py` enforces that). One more rule with a
  precedent: citations have ONE stored shape (`search.hit_to_dict`) — the
  terminal chats once wrote a thinner one and the GUI crashed on a missing
  `cosine` when it replayed them, so any writer must use the shared helper.
- `desktop/` — Tauri 2 shell (card-catalog UI; tabs: Chat, Sources, Indexed, Settings). Retrieval-only search lives in `ragdesk search` and `/api/search`;
  the Chat tab keeps its source list under every answer. Rust spawns `ragdesk serve`
  with `--db $HOME/.ragdesk/index.db`; a watchdog thread respawns it if it
  dies (skipping the respawn when another instance owns the port) and the
  child's output goes to `~/.ragdesk/serve.log`; env overrides: `RAGDESK_BIN`,
  `RAGDESK_DB`, `RAGDESK_LLM_MODEL`, `RAGDESK_PROJECT`. Browser mode:
  `ragdesk serve --ui desktop/dist`.
- Terminal surfaces (`tui.py`, `screen.py`): `ragdesk chat` is the line REPL
  (streaming + citations, history shared with the GUI); `ragdesk chat --server
  URL` attaches to a running server over its local API — no db, no embedder,
  no model load, the server owns all three; `ragdesk tui` is the full-screen
  Textual app behind the optional `tui` extra (`uv run --extra tui pytest
  tests/test_screen.py`; the test skips itself without the extra).
- Watcher + auto-index: `watch_loop` re-indexes the recorded local roots every
  `settings.watch_seconds` (default 60, 0 = off, UI card "Watch for changes")
  and `auto_index_loop` re-checks everything every `auto_index_hours` as the
  safety net; both take the mtime fast path, so a quiet pass is one stat per
  file. `POST /api/settings {watch_seconds}` clamps to 0-3600 and `/api/status`
  exposes it.
- Auto-index: `serve` runs a 60s timer; when `auto_index_hours` (Settings tab,
  default 1, 0 = off) has elapsed since `auto_index_last`, it re-indexes the
  recorded local roots via `run_auto_index` (connectors stay manual until
  their sync params are persisted).
- LLM wizard: `/api/llm/setup` {kind: mlx|ollama} starts a background download
  (`run_llm_setup`); progress + options ride on `/api/status.llm_setup`
  (job: running/progress/detail/error), the Settings tab renders them, and a
  finished job clears `state.llm` so the next ask re-resolves the ladder.
  `llm_setup_options` respects an explicit `--llm mlx:<repo>` override.
- Ask pipeline: `ask`/`ask_stream` parse `folder:`/`source:` filters, build a
  fingerprint (embedder + model + corpus revision) and try the exact cache key,
  then the semantic cache (`store.cache_nearest`, cosine ≥ 0.88 — calibrated:
  paraphrases score 0.91+, different intents ≤ 0.35), then retrieve → inject up
  to 3 similar memories (`store.memories`, cosine ≥ 0.35) and the last 3 turns
  → `answer*()`; every exchange is recorded in `chats`/`messages`, refusals and
  cache replays included. Chat/cache/memory tables live in the same SQLite
  file; `/api/chats` and `/api/memories` (+ `/api/memories/extract`) back the
  UI. A semantic hit reports `cached_question` so the UI can say what it matched.
- Answer engines: `llm.llm_preference` (Settings/wizard) picks Ollama, MLX, or an
  OpenAI-compatible endpoint; host/model live in settings.json and the API key
  in credentials (`openai`). `auto` ladder: ollama-with-model → configured
  endpoint → MLX.
- Diagrams: `answer.wants_diagram` detects intent (EN + VI words) and adds
  `DIAGRAM_NOTE` to the prompt, which pins the allowed Mermaid vocabulary;
  the UI strips the fence from the prose and renders the figure with mermaid
  (strict security, themed from the CSS tokens) plus SVG download. Small-model
  keyword slips (`subregion`) are repaired before rendering.
- Wizard: `/api/status.onboarded` gates the first-run overlay; `system_info()`
  reports RAM + suggested preset; `/api/settings {onboarded}` marks it done.
- Chunking: `chunk_text` is paragraph-aware and records `line_start` per chunk
  (citations show `file:line`). **Symbol-aware code chunking was tried three
  ways and rejected — do not rebuild it without new evidence.** Fresh-db A/B on
  the 12-query scoped golden (plain = recall 1.000 / nDCG 0.819 / MRR 0.757):
  1. per-symbol segmentation → 0.743 nDCG / 0.656 MRR (fragmentation raises BM25
     term density, small keyword-rich chunks crowd out the right document);
  2. symbol header inside each chunk → 0.788 / 0.715 (test files carry the
     symbol in their own names and out-rank the implementation);
  3. a symbol lane over definition sites (LIKE on names) → 0.523 / 0.419 (loose
     substring matching adds a weak lane and dilutes RRF).
  The path lane already answers "where is X defined". Real symbol intelligence
  is a call-graph index (defs + references + callers), a separate feature —
  not a chunking trick.
- Tuning tool: `scripts/bench.py` indexes once per chunking config and sweeps
  rerankers/weights over the same db (`--db`). Measured 2026-09-15 on the
  scoped golden: 600-char chunks rank better but lose recall; the multilingual
  `onnx` reranker lifts recall 0.833 → 0.917 while the English fastembed one
  drops it to 0.750; lane weights change nothing. Chunking stays 1000 chars
  (settings `chunk_chars`/`chunk_overlap`), weights stay equal, and the quality
  preset keeps `rerank: onnx`.
- Metadata: `index.parse_front_matter` reads a flat `--- key: value ---` header
  (md/txt), `documents.metadata` stores it as JSON, `parse_filters` turns any
  unrecognised `key:value` (minus URL schemes) into `Filters.meta`, and
  `_filter_sql` matches it through `json_extract` with a sanitised key. Hits
  carry the metadata for citation chips. `search.metadata_factor` is the soft
  boost (measure first — see the symbol-lane lesson): `authority: canonical`
  (or authoritative/official/high/true/yes/1) multiplies the fused score by
  `META_BOOST` 1.05, `status: draft` (or deprecated/archived/superseded/
  obsolete/wip) by `META_PENALTY` 0.90; no tag = 1.0, so the boost is opt-in
  and cannot re-rank an untagged corpus. `hybrid_search` decodes every lane's
  metadata JSON into the same dict shape (bm25/path lanes used to leak the raw
  string into `Hit.metadata`). Measured 2026-09-15: whole-repo index (93 docs /
  2408 chunks, absolute paths) bench none+onnx and the 24-query corpus copy
  both identical before/after (0.917/0.752/0.694 and 0.542/0.557/0.549);
  `tests/test_retrieval.py` proves the reorder canonical > plain > draft.
- Quiet indexing: `embed.session_options(threads)` builds every ONNX session
  (embedder *and* reranker) with `intra_op_num_threads`/`inter_op_num_threads=1`
  when capped; `settings.embed_threads` (0 = all cores) is read at session build
  and `embed.set_thread_override` lets `--embed-threads N` win for one process.
  `POST /api/settings {embed_threads}` unloads the embedder so the toggle takes
  effect immediately. Measured on the 651-chunk repo subset, cold index:
  all cores 97s @ ~580% CPU, 4 threads 136s @ ~390%, 2 threads 256s @ ~200% —
  `OMP_NUM_THREADS` does **not** work with the Python wheel, the code change is
  the only way to cap it.
- Embedding cache (`store.embed_cache`, key = sha1 of embedder name + dim + exact
  chunk text): `index._embed_with_cache` embeds only what is new, so a one-line
  edit costs one embedding instead of a whole document's worth and a duplicated
  file costs zero model calls. Measured on the 48-doc / 648-chunk repo subset:
  cold index 96.8s, wipe-the-documents-and-re-index 0.2s, and the golden
  numbers unchanged (0.917 / 0.783) — the cache must never alter a result, only
  the work it takes to produce one. It lives in the same SQLite file (one vector
  per chunk, ~3KB, pruned to 50k rows FIFO) and is disposable: delete the rows
  and the next index run re-embeds.
- Rejected after measuring: a **`tiny` LLM preset** (Qwen3.5-2B-4bit, ~1.2GB RAM
  vs the 4B's 2.5GB). The deterministic proxy looked *better* (grounded 1.000 vs
  0.953) but reading the answers showed the drop: the 2B opened the second
  question by re-answering the first (context bleed), missed the named flags and
  was visibly less structured. Lesson: `ground_answer` measures sentence overlap
  with the cited chunks, so a short extractive answer scores high — **always read
  the answers before shipping a model swap**, and keep the 4B.
- Ranking extras: `diversify` caps chunks per document (2 by default, backfills
  when a query is dominated by one file) and `recency_factor` adds a mild
  freshness nudge to the RRF score (RECENCY_WEIGHT).
- Smart retrieval (`serve._smart_retrieval` + `parse_smart_retrieval`): ONE
  local-model call that rewrites follow-ups, drafts the HyDE text and proposes
  up to two sub-queries (extra dense lanes). Runs only when it can pay off
  (history present, multi-part question, or hyde enabled) and is a no-op
  whenever no model resolves.
- Trust surface: `POST /api/feedback {message_id, value}` (▲/▼ stored on the
  message), `GET /api/feedback/golden` (rated answers exported as a golden
  JSONL), `POST /api/verify {message_id}` (`ground_answer_detail`: per-sentence
  grounded/loose verdicts against the stored citations, stopword-free overlap).
- Library surfaces: `GET /api/duplicates` (chunk-hash clusters, display-only —
  ranking is untouched by design), `POST /api/save {url}` + `GET /api/bookmarks`
  (one page saved with `web.save_page`, `metadata.url` keeps the real address),
  and the CLI mirrors: `ragdesk save <url>`, `ragdesk completions
  bash|zsh|fish`, `ragdesk man` (all in `complete.py`, generated from the
  argparse tree so they cannot drift; `complete._subparsers` reads argparse
  internals on purpose).
- Email surfaces: `POST /api/connections/email {host, port, user, password}`
  (validated by `email_source.whoami` before saving), `POST /api/sync/email
  {folder, limit}` (saved credentials; `limit` = newest N messages) and
  `POST /api/sync/email-mbox {path}`; CLI mirrors as `ragdesk email --mbox
  FILE` / `--imap HOST --user U` (password prompted, never an argv default).
  The Sources tab has the card; `credentials.email` holds the app password.
- Call graph (`symbols.py`): `parse_symbol_question` recognises "who calls X",
  "callers of X", "where is X defined" and the VI equivalents (returns
  `(name, certain)`; a plain lowercase word is only trusted when a definition
  exists). `find_symbol` scans the *stored chunk text* of code extensions for
  definitions (`chunk.SYMBOL_RE`), call sites (`X(`) and mentioning files — no
  AST, no index-time table, no re-index needed on old corpora; line numbers are
  chunk-relative (a file with runs of blank lines can be off by a line or two).
  `serve._symbol_lookup` runs before retrieval and returns a deterministic
  answer + `lanes="symbol"` hits, so symbol questions never reach the LLM and
  never touch the RRF lanes (graph stays at the navigation layer). The ask
  response carries `symbol` and the UI shows a "call graph" badge.
- Judge (`evaluate.judge_answer` + `JUDGE_PROMPT`): an opt-in LLM grader next
  to the overlap proxy — one call per answer asking for
  `{"supported", "total", "unsupported"}`; `parse_judge_reply` tolerates prose
  around the JSON and any unparseable reply stays `judged: False` (out of the
  aggregate, reported as coverage). `eval --judge` implies `--answers` and
  prints `judge faithfulness` + how many answers were judged.
- Portable bundle (`serve.run_export` / `run_import`): a zip with
  `manifest.json` + a SQLite-consistent `index.db` snapshot under
  `<db dir>/bundles/`; import validates the manifest, refuses a major-version
  gap or a different embedder (fail-closed: a silent mismatch would poison
  every future query), takes a safety snapshot, then uses the SQLite backup API
  to overwrite the live database in place. CLI `ragdesk export` / `import`,
  API `/api/export`, `/api/bundles`, `/api/import`, UI card in Settings.
  Credentials are never included in a bundle.
- Scope chip (UI): "Only this folder" on a citation sets a pending scope; the
  next questions are sent as `folder:"<dir>"` (quoted filter values — see
  `_FILTER_RE` in search.py), which is why paths with spaces work.
- Topic map picture (`scripts/topic_map.py`): numpy PCA over `_doc_vectors()`
  (script, not the stdlib core), colours from `topics.cluster_documents`,
  capped at `PER_CLUSTER` dots so a 200-copy pile cannot squash the map;
  writes a self-contained HTML with hover tooltips.
- Topic map (`topics.py`): `cluster_documents` is greedy leader clustering over
  `store._doc_vectors()` (the same average chunk vectors related-documents uses;
  pure Python, `MAX_DOCS` 2000 before it needs numpy) at cosine ≥ 0.75;
  `cluster_labels` attaches the most distinctive terms (cluster rate minus the
  corpus rate, sampled to 20k chunks) so the label is not just the commonest
  words. `GET /api/topics` powers the Indexed tab's Topics card, which shows
  multi-document clusters open, collapses single-document topics, and caps
  paths per cluster. Display only — never a ranking lane.
- Connector auto-sync: `settings.sync_jobs` holds `{id, provider, params}` entries
  (id = sha1 of provider+params, so the same sync is one job); the UI's
  "Keep in sync (hourly)" checkbox on a sync form posts to `/api/sync-jobs`
  after a successful sync. `SYNC_HANDLERS` in serve.py dispatches all nine
  connectors — github, gitlab, confluence, gdrive, msgraph, notion, email,
  web, s3 — pulling credentials from the same places the manual endpoints do
  (email's password from `credentials`, tokens resolved by each connector).
  `run_auto_index` runs local roots first, then every job, and returns a
  `connectors` list; a failing job is reported in it, never fatal.
  `GET /api/sync-jobs`, `POST /api/sync-jobs`, `/api/sync-jobs/delete`;
  `/api/status.sync_jobs` feeds the Settings card.
- S3 (`s3.py`): hand-rolled **SigV4 over stdlib** (`hashlib`/`hmac`/`urllib` +
  `xml.etree`) instead of boto3 or the aws CLI — a non-technical user should
  paste one key into the app (or read a public bucket with no key at all), not
  install a second tool. Key details: the canonical URI is the path **as sent**
  (S3 does not normalize it — matching botocore's `S3SigV4Auth`, cross-checked
  2026-09-16 on query strings, encoded keys with spaces, Unicode, `+` and
  custom endpoints), `x-amz-content-sha256` is the empty-payload hash, and
  **S3 XML carries a default namespace** so tags are stripped before parsing
  (finding `Contents` without that step silently returns zero objects — the
  first live run against a public bucket caught it). Credentials resolve
  explicit → `AWS_*` env → saved connection; custom endpoints are path-style.
  `/api/connections/s3` validates by listing before saving, `/api/sync/s3`
  indexes, `/api/status.s3` drives the card, `ragdesk s3 BUCKET` mirrors it.
  Verified live against the public `noaa-ghcn-pds` bucket: anonymous list,
  fetch, extract and search with no key stored.
- Email attachments (`email_source.attachments`): up to 10 per message, each
  decoded and pushed through `extract_bytes` (so OCR and sheet row-refs work);
  indexed as their own documents at `<base>::<message-key>::NN-<name>` with
  `kind: attachment`, while the message body keeps `[attachment: name]` lines.
  `IndexStats.attachments` counts them and flows into every sync summary.
- Reliability surfaces: `GET /api/health` (embedder match, last local run + `skipped_samples` from `store.last_index_report()`, oldest documents, db
  size, never-index patterns + docs still matching them), `POST
  /api/never-index {patterns}` (saves `settings.never_index` AND prunes matching
  documents — redaction, not just prevention; `index_paths` skips matching files
  as `never-index pattern`), and `GET /api/backups` + `POST /api/backup` +
  `POST /api/restore {path}` (`run_backup`/`restore_backup` use the SQLite
  backup API so they are safe while the app runs; a restore takes a safety
  snapshot first, and same-second snapshots get a `-N` suffix — without it the
  safety copy silently overwrote the backup being restored).
- **FTS wedge (fixed 2026-09-15, live incident):** a `chunks_fts` row whose
  chunk was gone collided with the next chunk id once ids were reused
  ("constraint failed" on insert), every re-index of that file failed, and the
  exception killed the watch + auto-index threads for the session. `Store._prune_orphan_fts_rows` repairs on every open (cheap count check),
  `index_paths` turns a per-file failure into a skip with the reason, and the
  serve loops log a failed pass instead of dying. If indexing "stops" again,
  check `~/.ragdesk/serve.log` first.
- Corrections: `corrections` table (`question`, `answer`, embedding of the
  question; `corrections_revision` for invalidation). The UI's Fix button under
  an answer opens an editor and `POST /api/corrections {question, answer}`;
  `GET /api/corrections` + `POST /api/corrections/delete` back the Settings
  card. On ask, `serve._corrections_for` tries every rewrite variant
  (`search_query` + `sub_queries`) and injects the nearest correction at
  cosine ≥ `CORRECTION_MIN_COSINE` 0.88 (same calibration as the semantic
  cache) — measured: a follow-up's raw text scored 0.416, its sub-query 0.896,
  so a single raw lookup is not enough. The block rides right before the
  question (`answer.CORRECTION_HEADER`, `ANSWER_PROMPT_VERSION` v4) because a
  4B model ignored it mid-prompt but followed it near the question — verify any
  wording change against the live model, not just a unit test. The API echoes
  `correction` (the matched question) on ask/ask_stream so the UI shows a "your
  fix" badge; the fingerprint carries `corrections_revision`, so adding a
  correction invalidates cached answers instead of replaying an uncorrected one.
- Retrieval lanes (`search.hybrid_search`): BM25 (FTS5), dense cosine, path
  tokens (file names), plus an optional HyDE dense lane — RRF-fused. HyDE text
  comes from `settings.hyde` (Settings toggle, off by default; measured:
  recall@5 0.889 → 1.000 on the repo golden set at +2-4s per question) and is
  skipped silently whenever no LLM resolves.
- Parent-child: `index.group_parents` groups child chunks (~4k chars) and
  `Hit.context` returns the parent for prompts; old rows are backfilled on
  Store open (`_backfill_parents`, no re-embedding) and fall back to the child.
- Eval: `evaluate` groups per-category metrics, `category_metrics` powers
  `--min-recall-category name=value` gates, and `ground_answer` is the
  deterministic faithfulness proxy (sentence overlap against cited chunks +
  citation range checks) used by `eval --answers`. Golden rows may carry
  `"history": [["user", "…"], ["assistant", "…"]]` for follow-ups; passing
  `rewrite_for(question, history)` makes the harness search the standalone
  rewrite too, and `eval --rewrite` prints the raw baseline next to it (the
  rewriter is `cli._standalone_rewrite`, the same prompt `serve` uses; it needs
  a resolvable local model or exits 2). `fixtures/golden_multiturn.jsonl`
  scores 0.926 → 1.000 nDCG / 0.900 → 1.000 MRR (raw → rewrite, Qwen3.5-4B MLX,
  2026-09-15). **Lesson: `SMART_RETRIEVAL_PROMPT` holds a JSON example, so its
  braces must be doubled for `.format()` — the un-doubled prompt raised
  `KeyError: '"standalone"'`, the bare `except` in `_smart_retrieval` swallowed
  it, and follow-up rewriting was silently off; those handlers now print the
  reason to stderr.**
- Connectors ingest payloads through `index.extract_bytes(data, name)`: one
  dispatcher for PDFs, office files, images (OCR) and plain text. Never decode
  raw bytes to text at a call site — `is_indexable` admits those types now, so
  a local decode would index image bytes as mojibake.
- Long jobs report progress through `state.activity` (owner-token guarded so
  concurrent runs cannot clobber each other): index/sync endpoints pass a
  progress callback into `index_paths`/`sync_github`/`sync_gitlab`, auto-index
  claims the slot too, and `/api/status.activity` feeds the rail + sync cards.
- `ask_stream` commits headers first and then emits `{"status": …}` lines
  (searching → HyDE → loading the model → thinking) before the deltas, so the
  UI shows phases + elapsed seconds and can abort with the Stop button.
- Idle unload: the serve timer releases the embedder session and the LLM after
  `idle_unload_minutes` (Settings; 0 = never) of no POSTs; ONNX workspace
  memory does not respond to arena/batch tuning, so dropping the session is
  the honest way to give RAM back.
- Connections: GitHub has three paths (device code with `RAGDESK_GITHUB_CLIENT_ID`
  or a saved client ID, `gh` login, or a pasted token); Confluence and GDrive
  connect flows validate before saving to `~/.config/ragdesk/credentials.json`
  (`0600`; override the dir with `RAGDESK_CONFIG_DIR` for tests).
- Credentials policy: `.env` (gitignored) → `python -m ragdesk.buildenv` bakes
  non-confidential values into `src/ragdesk/_build_env.py` (gitignored) for
  release artifacts. The Atlassian 3LO secret is refused by the baker — it stays
  per-user. User-facing setup guides: `docs/sources.md`.
- `fixtures/` — tiny corpus + two golden sets (fixtures, repo) for offline CI.
- `tests/` — pytest; always uses `HashingEmbedder` (never requires Ollama or
  network). Connector tests monkeypatch HTTP.

## Commands

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run ragdesk --embedder hash --db /tmp/eval.db index fixtures/docs
uv run ragdesk --embedder hash --db /tmp/eval.db eval --golden fixtures/golden.jsonl
# repo golden (real numbers; --embedder onnx downloads ~0.3GB on first run)
# NB: the golden scopes with folder:dtduc-git/ragdesk, so index an absolute path
uv run ragdesk --embedder hash --db /tmp/eval-repo.db index "$PWD"
uv run ragdesk --embedder hash --db /tmp/eval-repo.db eval --golden fixtures/golden_repo.jsonl
# desktop shell
cd desktop && npm install && npm run tauri build
# macOS: sign with a stable identity so the Documents/Desktop/Downloads TCC
# grant survives rebuilds — an ad-hoc build gets a new identity every time and
# macOS re-asks for folder access on the next run
APPLE_SIGNING_IDENTITY="Apple Development: <you> (<TEAMID>)" npm run tauri build
# release build with baked credentials
cp .env.example .env   # fill non-confidential values (see SECURITY.md)
uv run python -m ragdesk.buildenv && uv build
```

## Conventions

- Core stays dependency-free (argparse/sqlite3/urllib). New runtime deps need a
  good reason; ONNX/reranker deps belong in an optional extra.
- `HashingEmbedder` is for tests/CI only — never quote its numbers as quality.
- Never default the reranker to a CC-BY-NC model (jina-reranker-v2 is
  non-commercial). Defaults: `BAAI/bge-reranker-base` (MIT, fastembed) and
  `onnx-community/gte-multilingual-reranker-base` (Apache-2.0, `--rerank onnx`).
- Embedder/dimension mismatch must stay fail-closed (`Store.ensure_embedder`).
- Sources are read-only. No telemetry, ever.
- Eval numbers published in README must be reproducible from the repo.

## Roadmap order

sources (Notion → GitLab → OneDrive/SharePoint → S3) → multilingual reranker →
MCP server → release v0.1.0 (sidecar bundling + DMG release workflow + PyPI
trusted publisher) → eval badge automation → Windows/Linux builds.

## Notes

- If code is ever reused from OpsRAG (Apache-2.0), add the required NOTICE
  attribution before release.
- PyPI name `ragdesk` verified free (2026-09-14); re-check the similarity
  checker at publish time.
