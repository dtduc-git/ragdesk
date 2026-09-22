"""Materialize ONNX files in the HuggingFace cache (workaround for v0.1.0).

onnxruntime validates that a model's external-data file (``*.onnx_data``)
lives inside the model's own directory. On caches that shard blobs into
``blobs/<xx>/<hash>`` the two files resolve into different directories and the
model refuses to load:

    [ONNXRuntimeError] : 1 : FAIL : External data path validation failed ...
    Error: External data path escapes model directory

ragdesk 0.1.1 loads models from a materialized copy and no longer needs this.
For a machine still on 0.1.0, run this once (the app's bundled Python works):

    "/Applications/ragdesk.app/Contents/Resources/python/bin/python3" \
        scripts/materialize_hf_models.py

Run it again after the app downloads a new model (embedder, reranker).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def materialize(root: Path) -> int:
    done = 0
    for link in root.rglob("*"):
        if not link.is_symlink():
            continue
        if not (link.name.endswith(".onnx") or link.name.endswith(".onnx_data")):
            continue
        target = link.resolve()
        if not target.is_file():
            continue
        link.unlink()
        try:
            os.link(target, link)  # same inode: no extra disk
        except OSError:
            shutil.copy2(target, link)
        done += 1
        print(f"  {link}")
    return done


def main() -> int:
    base = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
    root = base / "hub"
    if not root.is_dir():
        print(f"no HuggingFace cache at {root} — nothing to do")
        return 0
    print(f"materializing ONNX files under {root} ...")
    done = materialize(root)
    print(f"done: {done} files are now real files (restart ragdesk)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
