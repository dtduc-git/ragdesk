"""The 2D map script: deterministic projection + a self-contained HTML page."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from ragdesk.embed import HashingEmbedder  # noqa: E402
from ragdesk.index import index_paths  # noqa: E402
from ragdesk.store import Store  # noqa: E402

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "topic_map.py"
spec = importlib.util.spec_from_file_location("topic_map_script", SCRIPT)
topic_map_script = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
assert spec and spec.loader
spec.loader.exec_module(topic_map_script)


def make_corpus(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    for index in range(4):
        (docs / f"tax-{index}.md").write_text(
            "Thuế thu nhập cá nhân: khấu trừ thuế thu nhập cá nhân theo biểu thuế. " * 6
        )
    for index in range(4):
        (docs / f"k8s-{index}.md").write_text(
            "Kubernetes deployment: kubectl apply the deployment manifest into the namespace. " * 6
        )
    return docs


def test_build_map_projects_and_clusters(tmp_path: Path):
    docs = make_corpus(tmp_path)
    with Store(tmp_path / "index.db") as store:
        index_paths(store, HashingEmbedder(dim=512), [docs])
        artifact = topic_map_script.build_map(store)

    assert artifact["documents"] == 8
    assert len(artifact["points"]) == 8
    assert all(
        isinstance(point["x"], float) and isinstance(point["y"], float)
        for point in artifact["points"]
    )
    assert len(artifact["clusters"]) == 2
    # the two topics land in different clusters, so different colours
    colors: dict[str, str] = {}
    for point in artifact["points"]:
        stem = Path(point["path"]).stem.split("-")[0]
        colors[stem] = point["color"]
    assert colors["tax"] != colors["k8s"]
    # coordinates actually vary (a real projection, not a constant)
    assert len({round(point["x"], 3) for point in artifact["points"]}) > 1


def test_render_html_is_self_contained(tmp_path: Path):
    docs = make_corpus(tmp_path)
    with Store(tmp_path / "index.db") as store:
        index_paths(store, HashingEmbedder(dim=512), [docs])
        artifact = topic_map_script.build_map(store)
    page = topic_map_script.render_html(artifact)
    assert page.startswith("<!doctype html>")
    assert "<svg" in page and page.count("<circle") == 8
    assert "tax-0.md" in page  # the tooltip carries the path
    assert "http" not in page.split("</style>")[1]  # no external assets
