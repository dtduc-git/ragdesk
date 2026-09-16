#!/usr/bin/env bash
# The repo-golden eval CI runs: build the subset index the golden is about and
# score it. stdout = eval JSON (for the badge), stderr = the human report.
# Shared by .github/workflows/ci.yml (the gate) and eval.yml (badge + README).
#
#   scripts/eval_ci.sh --gate            # exit 1 when recall@5 < 0.8
#   scripts/eval_ci.sh --db=/tmp/x.db    # reuse another index
set -euo pipefail

gate=""
db=".ragdesk/eval-repo.db"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gate) gate="--min-recall 0.8"; shift ;;
    --db) db="$2"; shift 2 ;;
    --db=*) db="${1#--db=}"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

base="$(cd "$(dirname "$0")/.." && pwd)"
root="/tmp/ragdesk-ci/dtduc-git/ragdesk"
mkdir -p "$(dirname "$root")"
ln -sfn "$base" "$root"   # the golden scopes queries with folder:dtduc-git/ragdesk

uv run ragdesk --embedder onnx --db "$db" index \
  "$root/README.md" "$root/AGENTS.md" "$root/SECURITY.md" \
  "$root/.github" "$root/src" "$root/fixtures" >&2
# shellcheck disable=SC2086 - the gate flag is a fixed two-word string
uv run ragdesk --embedder onnx --db "$db" eval \
  --golden fixtures/golden_repo.jsonl $gate --json
