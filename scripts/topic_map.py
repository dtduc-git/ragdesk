"""Draw the corpus: every document a point, topic clusters as colours.

    python scripts/topic_map.py --db ~/.ragdesk/index.db --out /tmp/topics.html

The projection is a two-component PCA over the stored document vectors (numpy
is fine here — this is a script, not the stdlib-only core). Clusters come from
the same greedy leader algorithm the app uses, so the picture and the Topics
card agree. Open the HTML in a browser; hovering a point shows its path.
"""

from __future__ import annotations

import argparse
import html
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ragdesk.store import Store  # noqa: E402
from ragdesk.topics import cluster_documents, cluster_labels  # noqa: E402

PALETTE = [
    "#4a3d6b",
    "#a97a1e",
    "#2e6b58",
    "#a33b32",
    "#3a6ea5",
    "#7a4a8c",
    "#6b7a1e",
    "#8c5a3a",
    "#1e6b6b",
    "#8c3a6b",
    "#5a6b8c",
    "#3a8c5a",
]
MAX_POINTS = 1200
PER_CLUSTER = 12  # a 200-copy pile must not squash the rest of the map


def build_map(store: Store, *, limit: int = MAX_POINTS) -> dict:
    """Project document vectors to 2D and label the clusters."""
    import numpy as np

    vectors = store._doc_vectors()  # noqa: SLF001 - the shared average-vector helper
    items = sorted(vectors.items())
    if len(items) > limit:  # deterministic stride so the sample covers the corpus
        step = math.ceil(len(items) / limit)
        items = items[::step]
    if not items:
        return {"points": [], "clusters": [], "documents": 0, "sampled": False}

    # clusters first: a 200-copy pile must not squash the rest of the map
    clusters = cluster_documents(store)
    cluster_labels(store, clusters)
    cluster_of: dict[str, tuple[int, str, str]] = {}
    clustered: set[str] = set()
    for index, cluster in enumerate(clusters):
        color = PALETTE[index % len(PALETTE)]
        clustered.update(cluster["paths"])
        for path in cluster["paths"][:PER_CLUSTER]:
            cluster_of[path] = (index, color, cluster.get("label", ""))
    sampled = len(clustered) > len(cluster_of)
    if sampled:
        items = [item for item in items if item[1][0] not in clustered or item[1][0] in cluster_of]

    paths = [path for _doc_id, (path, _vector) in items]
    matrix = np.array([vector for _doc_id, (_path, vector) in items], dtype=np.float32)
    matrix -= matrix.mean(axis=0)
    # two leading principal components; SVD is cheap at this size
    left, singular, _vt = np.linalg.svd(matrix, full_matrices=False)
    coords = left[:, :2] * singular[:2]

    points = [
        {
            "path": path,
            "x": float(coords[position][0]),
            "y": float(coords[position][1]),
            "cluster": cluster_of.get(path, (-1, "#8b9295", "(unclustered)"))[0],
            "color": cluster_of.get(path, (-1, "#8b9295", "(unclustered)"))[1],
            "label": cluster_of.get(path, (-1, "#8b9295", "(unclustered)"))[2],
        }
        for position, path in enumerate(paths)
    ]
    legend = [
        {
            "label": cluster.get("label", "") or "(mixed)",
            "documents": len(cluster["paths"]),
            "color": PALETTE[index % len(PALETTE)],
        }
        for index, cluster in enumerate(clusters)
    ]
    return {
        "points": points,
        "clusters": legend,
        "documents": len(paths),
        "sampled": sampled,
    }


def render_html(artifact: dict, *, title: str = "ragdesk topic map") -> str:
    points = artifact["points"]
    width, height, pad = 960, 620, 30
    xs = [point["x"] for point in points] or [0.0]
    ys = [point["y"] for point in points] or [0.0]
    span_x = (max(xs) - min(xs)) or 1.0
    span_y = (max(ys) - min(ys)) or 1.0

    def to_svg(point: dict) -> tuple[float, float]:
        x = pad + (point["x"] - min(xs)) / span_x * (width - 2 * pad)
        y = height - pad - (point["y"] - min(ys)) / span_y * (height - 2 * pad)
        return x, y

    dots = "\n".join(
        f'<circle cx="{to_svg(point)[0]:.1f}" cy="{to_svg(point)[1]:.1f}" r="3.4" '
        f'fill="{point["color"]}" fill-opacity="0.78">'
        f"<title>{html.escape(point['path'])} — {html.escape(point['label'])}</title>"
        "</circle>"
        for point in points
    )
    legend = "\n".join(
        f'<li><span class="swatch" style="background:{entry["color"]}"></span>'
        f"{html.escape(entry['label'])} <em>{entry['documents']}</em></li>"
        for entry in artifact["clusters"]
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
  body {{ margin: 0; padding: 28px 32px; font: 14px/1.5 -apple-system, system-ui, sans-serif;
         color: #1b1f23; background: #f4f5f0; }}
  h1 {{ font-family: "Iowan Old Style", Georgia, serif; font-size: 24px; margin: 0 0 4px; }}
  p.lede {{ margin: 0 0 18px; color: #5b6367; }}
  .wrap {{ display: grid; grid-template-columns: 1fr 260px; gap: 26px; align-items: start; }}
  svg {{ background: #fff; border: 1px solid #c7cabf; border-radius: 10px;
        width: 100%; height: auto; }}
  ul {{ list-style: none; margin: 0; padding: 0; }}
  li {{ display: flex; gap: 8px; align-items: baseline; padding: 5px 0;
        border-bottom: 1px solid #e3e5dd; font-size: 13px; }}
  li em {{ margin-left: auto; color: #8b9295; font-style: normal;
          font-family: ui-monospace, monospace; }}
  .swatch {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="lede">{artifact["documents"]} points, {len(artifact["clusters"])} clusters —
hover a point for its path.{" Large clusters are sampled." if artifact.get("sampled") else ""}</p>
<div class="wrap">
  <svg viewBox="0 0 {width} {height}" role="img" aria-label="document map">
    <rect x="0" y="0" width="{width}" height="{height}" fill="none"/>
{dots}
  </svg>
  <ul>{legend}</ul>
</div>
</body></html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(Path.home() / ".ragdesk" / "index.db"))
    parser.add_argument("--out", default="topic-map.html")
    parser.add_argument("--limit", type=int, default=MAX_POINTS)
    args = parser.parse_args(argv)

    try:
        import numpy  # noqa: F401
    except ImportError:
        print(
            "numpy is required for the projection (it comes with the onnx extra)",
            file=sys.stderr,
        )
        return 2

    with Store(args.db) as store:
        artifact = build_map(store, limit=args.limit)
    if not artifact["documents"]:
        print("nothing indexed yet", file=sys.stderr)
        return 1
    out = Path(args.out).expanduser()
    out.write_text(render_html(artifact))
    print(f"wrote {out} ({artifact['documents']} documents, {len(artifact['clusters'])} clusters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
