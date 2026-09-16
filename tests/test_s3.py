from __future__ import annotations

from pathlib import Path

import pytest

from ragdesk.embed import HashingEmbedder
from ragdesk.s3 import S3Error, parse_listing, sync_s3
from ragdesk.search import hybrid_search
from ragdesk.store import Store

LISTING = """2026-09-01 10:00:00       1200 docs/deploy.md
2026-09-01 10:00:01       2400 docs/oncall.md
2026-09-01 10:00:02        900 archive/dump.zip
2026-09-01 10:00:03          0 docs/empty.md
"""


def test_parse_listing_reads_aws_output():
    assert parse_listing(LISTING) == [
        ("docs/deploy.md", 1200),
        ("docs/oncall.md", 2400),
        ("archive/dump.zip", 900),
        ("docs/empty.md", 0),
    ]
    assert parse_listing("") == []


def fake_aws(pages: dict[str, bytes]):
    calls: list[list[str]] = []

    def runner(args: list[str], *, profile: str = "", timeout: float = 120.0) -> bytes:
        calls.append(args)
        if args[:2] == ["s3", "ls"]:
            return LISTING.encode()
        if args[:2] == ["s3", "cp"]:
            key = args[2].split("/", 3)[3]
            return pages.get(key, b"")
        raise AssertionError(args)

    return runner, calls


def test_sync_s3_indexes_objects(tmp_path: Path, monkeypatch):
    runner, calls = fake_aws(
        {
            "docs/deploy.md": (
                b"Deploys use Terraform and Helm. Rollback with kubectl rollout undo."
            ),
            "docs/oncall.md": b"Pages are acknowledged within 5 minutes.",
            "archive/dump.zip": b"PK\x03\x04 not read",
            "docs/empty.md": b"   ",
        }
    )
    monkeypatch.setattr("ragdesk.s3._run_aws", runner)
    with Store(tmp_path / "index.db") as store:
        stats = sync_s3(store, HashingEmbedder(dim=512), bucket="my-bucket", prefix="docs/")
        assert stats.indexed == 2 and stats.chunks == 2
        assert stats.skipped == 2  # the .zip is unreadable, the empty one has no text
        reasons = {row["path"]: row["reason"] for row in stats.skipped_samples}
        assert reasons["archive/dump.zip"] == "unsupported or too large"
        assert reasons["docs/empty.md"] == "no extractable text"
        assert [row["path"] for row in store.documents()] == [
            "s3://my-bucket/docs/deploy.md",
            "s3://my-bucket/docs/oncall.md",
        ]
        rows = store.documents()
        assert rows[0]["metadata"]["bucket"] == "my-bucket"
        assert rows[0]["source"] == "s3:my-bucket"

        hits = hybrid_search(store, HashingEmbedder(dim=512), "terraform rollback", top_k=3)
        assert hits and hits[0].path.endswith("docs/deploy.md")

    assert calls[0][:3] == ["s3", "ls", "s3://my-bucket/docs/"]
    assert "--recursive" in calls[0]


def test_sync_s3_respects_the_limit(tmp_path: Path, monkeypatch):
    runner, _calls = fake_aws(
        {"docs/deploy.md": b"one", "docs/oncall.md": b"two"}
    )
    monkeypatch.setattr("ragdesk.s3._run_aws", runner)
    with Store(tmp_path / "index.db") as store:
        stats = sync_s3(store, HashingEmbedder(dim=256), bucket="b", limit=1)
    assert stats.indexed == 1


def test_sync_s3_without_the_cli_says_so(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("ragdesk.s3.shutil.which", lambda name: None)
    with Store(tmp_path / "index.db") as store:
        with pytest.raises(S3Error) as excinfo:
            sync_s3(store, HashingEmbedder(dim=64), bucket="b")
    assert "aws CLI is not on PATH" in str(excinfo.value)


def test_sync_s3_requires_a_bucket(tmp_path: Path):
    with Store(tmp_path / "index.db") as store:
        with pytest.raises(S3Error):
            sync_s3(store, HashingEmbedder(dim=64), bucket="  ")
