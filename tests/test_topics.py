from __future__ import annotations

from pathlib import Path

from ragdesk.embed import HashingEmbedder
from ragdesk.index import index_paths
from ragdesk.store import Store
from ragdesk.topics import cluster_documents, topic_map

TAX_TEXT = (
    "Thuế thu nhập cá nhân: khấu trừ thuế thu nhập cá nhân theo biểu thuế lũy tiến. "
    "Tờ khai thuế thu nhập cá nhân nộp cho cơ quan thuế. "
) * 6

K8S_TEXT = (
    "Kubernetes deployment: kubectl apply the deployment manifest into the namespace. "
    "The pod restarts under the deployment controller inside the kubernetes cluster. "
) * 6


def make_corpus(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    for index in range(3):
        (docs / f"tax-{index}.md").write_text(TAX_TEXT + f"note {index}")
    for index in range(3):
        (docs / f"k8s-{index}.md").write_text(K8S_TEXT + f"note {index}")
    return docs


def test_cluster_documents_groups_by_topic(tmp_path: Path):
    docs = make_corpus(tmp_path)
    with Store(tmp_path / "index.db") as store:
        index_paths(store, HashingEmbedder(dim=512), [docs])
        clusters = cluster_documents(store)
    groups = sorted(
        sorted(Path(path).name.split("-")[0] for path in cluster["paths"])
        for cluster in clusters
    )
    assert groups == [["k8s", "k8s", "k8s"], ["tax", "tax", "tax"]]


def test_topic_map_labels_are_distinctive(tmp_path: Path):
    docs = make_corpus(tmp_path)
    with Store(tmp_path / "index.db") as store:
        index_paths(store, HashingEmbedder(dim=512), [docs])
        topics = topic_map(store)
    assert [topic["documents"] for topic in topics] == [3, 3]
    labels = " | ".join(str(topic["label"]) for topic in topics)
    assert "thuế" in labels or "thu" in labels, labels
    assert "kubernetes" in labels or "kubectl" in labels, labels
    for topic in topics:
        assert len(topic["paths"]) == 3


def test_topic_map_on_an_empty_index(tmp_path: Path):
    with Store(tmp_path / "index.db") as store:
        assert topic_map(store) == []
