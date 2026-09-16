from __future__ import annotations

from pathlib import Path

from ragdesk.embed import HashingEmbedder
from ragdesk.index import index_paths
from ragdesk.obsidian import is_vault, links_in, parse_aliases, vault_metadata
from ragdesk.search import hybrid_search
from ragdesk.store import Store


def test_parse_aliases_accepts_comma_and_bracket_lists():
    assert parse_aliases("Runbook, Deploy notes") == ["Runbook", "Deploy notes"]
    assert parse_aliases("[one, 'two']") == ["one", "two"]
    assert parse_aliases("") == []


def test_vault_metadata_merges_tags_and_prefixes_aliases():
    content = (
        "---\n"
        "aliases: Deploy Runbook, Rollback\n"
        "tags: platform, deploy\n"
        "service: payments\n"
        "---\n"
        "# Deploying\n\n"
        "Everything here is #infra and #deploy.\n"
    )
    metadata, body = vault_metadata("Notes", content)
    assert metadata["vault"] == "Notes"
    assert metadata["service"] == "payments"
    assert metadata["aliases"] == "Deploy Runbook,Rollback"
    assert metadata["tags"] == "platform,deploy,infra"
    assert body.startswith("Aliases: Deploy Runbook, Rollback")
    assert "---" not in body  # front-matter never reaches the chunker
    assert "#infra" in body


def test_links_in_dedupes_wikilinks():
    text = "See [[Deploy Runbook]] and [[Deploy Runbook|the runbook]] plus [[Ops#On-call]]."
    assert links_in(text) == ["Deploy Runbook", "Ops"]


def make_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "MyVault"
    (vault / ".obsidian").mkdir(parents=True)
    (vault / ".trash").mkdir()
    (vault / ".obsidian" / "workspace.json").write_text('{"config": "never indexed"}')
    (vault / ".trash" / "old-note.md").write_text("deleted note body")
    (vault / "deploy.md").write_text(
        "---\n"
        "aliases: Rollback Runbook\n"
        "tags: platform\n"
        "---\n"
        "# Deploying\n\n"
        "Canary first, roll back with kubectl rollout undo. See [[On-call]] #infra\n"
    )
    (vault / "oncall.md").write_text("Pages are acknowledged within 5 minutes.\n")
    return vault


def test_index_paths_skips_config_dirs_and_keeps_vault_metadata(tmp_path: Path, monkeypatch):
    vault = make_vault(tmp_path)
    monkeypatch.setattr(
        "ragdesk.settings.load",
        lambda path=None: {
            "chunk_chars": 1000,
            "chunk_overlap": 150,
            "never_index": [],
            "vaults": [str(vault)],
        },
    )
    embedder = HashingEmbedder(dim=512)
    with Store(tmp_path / "index.db") as store:
        stats = index_paths(store, embedder, [vault])
        assert stats.indexed == 2  # the config dirs contributed nothing
        docs = {row["path"]: row for row in store.documents()}
        deploy = docs[str(vault / "deploy.md")]
        assert deploy["metadata"]["vault"] == "MyVault"
        assert "platform" in deploy["metadata"]["tags"]
        assert "infra" in deploy["metadata"]["tags"]

        # the alias is searchable text, not just metadata
        hits = hybrid_search(store, embedder, "rollback runbook", top_k=3)
        assert hits and hits[0].path.endswith("deploy.md")
        assert all(".obsidian" not in row["path"] for row in store.documents())


def test_is_vault_detects_the_config_folder(tmp_path: Path):
    assert is_vault(str(make_vault(tmp_path))) is True
    plain = tmp_path / "plain"
    plain.mkdir()
    assert is_vault(str(plain)) is False
