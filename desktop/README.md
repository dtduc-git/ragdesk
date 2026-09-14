# ragdesk desktop

Tauri 2 shell for the local indexer. On launch it spawns `ragdesk serve`
(port 8765) and renders the card-catalog UI against it.

## Run

```bash
# from the repo root, one-time: make the CLI available to the shell
uv tool install ".[onnx]"

# dev (spawns `uv run ragdesk serve` automatically)
npm install
npm run tauri dev

# build a macOS app bundle
npm run tauri build
```

The Rust side looks for the server binary in this order:
`$RAGDESK_BIN` → `ragdesk` on PATH → `~/.local/bin/ragdesk` →
`uv run --project $RAGDESK_PROJECT (default: ..) ragdesk serve`.

Set `RAGDESK_LLM_MODEL` to override the preset's LLM (useful when the
preset model is not pulled yet).

## Browser mode

The same UI runs in a browser against a manually started server:

```bash
uv run ragdesk --embedder onnx serve --ui desktop/dist
# open http://127.0.0.1:8765/
```
