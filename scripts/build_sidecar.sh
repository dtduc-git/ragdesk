#!/usr/bin/env bash
# Build the bundled runtime: a relocatable CPython with ragdesk installed, so
# the .app works on a machine that has never run `uv tool install ragdesk`.
#
#   scripts/build_sidecar.sh            # ~500MB into desktop/src-tauri/sidecar
#   npm run tauri build                 # bakes it into Contents/Resources/python
#
# The Rust shell runs `<Resources>/python/bin/python3 -m ragdesk …` (module
# invocation: a venv's console-script shebang would point at build-time paths).
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$root/desktop/src-tauri/sidecar"
python_version="${RAGDESK_PYTHON:-3.12}"
extras="${RAGDESK_SIDECAR_EXTRAS:-onnx,mlx,vision}"

echo "==> interpreter (uv-managed standalone CPython $python_version)"
uv python install "$python_version" >/dev/null
py="$(uv python find "$python_version")"
py_root="$(cd "$(dirname "$py")/.." && pwd)"   # …/cpython-3.12.x-macos-aarch64-none
echo "    $py_root"

echo "==> copying the interpreter to $dest/python"
rm -rf "$dest"
mkdir -p "$dest"
rsync -a \
  --exclude 'lib/python*/test' \
  --exclude 'lib/python*/idlelib' \
  --exclude 'lib/python*/tkinter' \
  --exclude 'lib/python*/ensurepip' \
  --exclude 'share/' \
  --exclude 'include/' \
  "$py_root/" "$dest/python/"

echo "==> installing ragdesk[$extras] into it (no venv: the interpreter is ours)"
uv pip install --python "$dest/python/bin/python3" --no-cache "$root[$extras]"

echo "==> pruning caches"
find "$dest" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$dest" -name '*.pyc' -delete 2>/dev/null || true

echo "==> smoke test: the runtime serves on its own"
"$dest/python/bin/python3" -c "import ragdesk, sys; print('   ragdesk', ragdesk.__version__, 'on', sys.version.split()[0])"

echo "==> done: $(du -sh "$dest" | awk '{print $1}') in $dest"
