"""Refresh the published eval numbers: docs/eval.json (shields endpoint) and the
CI row in README.md between the ``<!-- eval-ci:start -->`` markers.

    scripts/eval_badge.py /tmp/eval.json     # the JSON printed by eval_ci.sh
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

START = "<!-- eval-ci:start -->"
END = "<!-- eval-ci:end -->"


def badge_payload(metrics: dict) -> dict:
    recall = float(metrics.get("recall@5") or 0.0)
    color = "brightgreen" if recall >= 0.9 else "green" if recall >= 0.8 else "orange"
    return {
        "schemaVersion": 1,
        "label": "retrieval recall@5",
        "message": f"{recall:.3f}",
        "color": color,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: eval_badge.py <eval-report.json>", file=sys.stderr)
        return 2
    report = json.loads(Path(argv[1]).read_text())
    metrics = report.get("metrics") or {}
    if "recall@5" not in metrics:
        print("no metrics in the report", file=sys.stderr)
        return 1

    badge_path = Path("docs/eval.json")
    badge_path.parent.mkdir(parents=True, exist_ok=True)
    badge_path.write_text(json.dumps(badge_payload(metrics), indent=2) + "\n")

    row = (
        "| repo docs+source subset (the CI run) / EmbeddingGemma int8 "
        f"| {float(metrics['recall@5']):.3f} "
        f"| {float(metrics['ndcg@10']):.3f} "
        f"| {float(metrics['mrr@10']):.3f} |"
    )
    readme = Path("README.md")
    text = readme.read_text()
    if START in text and END in text:
        head, rest = text.split(START, 1)
        _, tail = rest.split(END, 1)
        readme.write_text(f"{head}{START}\n{row}\n{END}{tail}")
    else:
        print("README markers missing — row not updated", file=sys.stderr)
    print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
