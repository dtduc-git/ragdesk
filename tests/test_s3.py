"""S3 connector: stdlib SigV4 signing, REST calls, and indexing.

The signature vectors below are byte-for-byte what botocore's ``S3SigV4Auth``
produces for the same inputs (cross-checked 2026-09-16 across query strings,
encoded keys with spaces, Unicode, ``+`` and custom endpoints), so the suite
stays offline while still pinning the algorithm AWS expects.
"""

from __future__ import annotations

import io
import urllib.error
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ragdesk.embed import HashingEmbedder
from ragdesk.s3 import (
    EMPTY_PAYLOAD_HASH,
    S3Error,
    get_object,
    list_objects,
    resolve_credentials,
    sign_request,
    sync_s3,
)
from ragdesk.store import Store

MOMENT = datetime(2026, 9, 16, 12, 0, 0, tzinfo=UTC)
CREDS = {
    "access_key": "AKIAIOSFODNN7EXAMPLE",
    "secret_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "session_token": "",
    "region": "us-east-1",
    "endpoint": "",
}

LISTING_PAGE = """<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <IsTruncated>true</IsTruncated>
  <NextContinuationToken>page2</NextContinuationToken>
  <Contents><Key>docs/deploy.md</Key><Size>1200</Size></Contents>
  <Contents><Key>docs/oncall.md</Key><Size>2400</Size></Contents>
  <Contents><Key>archive/dump.zip</Key><Size>900</Size></Contents>
</ListBucketResult>
"""

LISTING_PAGE_2 = """<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <IsTruncated>false</IsTruncated>
  <Contents><Key>docs/empty.md</Key><Size>0</Size></Contents>
</ListBucketResult>
"""


def test_signature_matches_botocore_vectors():
    url, headers = sign_request(
        "GET",
        "https://examplebucket.s3.us-east-1.amazonaws.com/?list-type=2&max-keys=2&prefix=docs%2F",
        CREDS,
        now=MOMENT,
    )
    assert headers["Authorization"] == (
        "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20260916/us-east-1/s3/aws4_request, "
        "SignedHeaders=host;x-amz-content-sha256;x-amz-date, "
        "Signature=f6924777e6583cc79f6aaa5d198f6a54742d78df8819973f2f302bc03efd9568"
    )
    assert headers["x-amz-date"] == "20260916T120000Z"
    assert headers["x-amz-content-sha256"] == EMPTY_PAYLOAD_HASH
    assert url.endswith("?list-type=2&max-keys=2&prefix=docs%2F")
    assert "host" not in {key.lower() for key in headers if key != "Authorization"}
    assert "Authorization" in headers and "User-Agent" in headers


def test_signature_covers_encoded_object_paths():
    _url, headers = sign_request(
        "GET",
        "https://examplebucket.s3.us-east-1.amazonaws.com/reports/2026%20Q1.pdf",
        CREDS,
        now=MOMENT,
    )
    assert headers["Authorization"].endswith(
        "Signature=c1751177b17c6176c2e7c2bf6357cb36ff74d9784c2712a5618aed0eda3015a7"
    )


def test_session_token_is_signed():
    _url, headers = sign_request(
        "GET",
        "https://examplebucket.s3.us-east-1.amazonaws.com/?list-type=2",
        {**CREDS, "session_token": "SESSIONTOKENEXAMPLE"},
        now=MOMENT,
    )
    assert "x-amz-security-token" in headers["Authorization"]
    assert headers["x-amz-security-token"] == "SESSIONTOKENEXAMPLE"


def test_resolve_credentials_precedence(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RAGDESK_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ENVKEY")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    from ragdesk import credentials

    credentials.set_provider(
        "s3", {"access_key": "STOREDKEY", "secret_key": "STOREDSECRET", "region": "ap-southeast-2"}
    )
    values = resolve_credentials()
    assert values["access_key"] == "ENVKEY"  # env beats the stored connection
    assert values["secret_key"] == "STOREDSECRET"  # ...per field
    explicit = resolve_credentials({"access_key": "EXPLICITKEY"})
    assert explicit["access_key"] == "EXPLICITKEY"
    assert explicit["region"] == "eu-west-1"


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def headers(self):
        return {}


def fake_transport(pages: dict[str, bytes], seen: list[str]):
    import urllib.request

    def opener(request, timeout=None):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        seen.append(url)
        for marker, body in pages.items():
            if marker in url:
                return FakeResponse(body)
        raise urllib.error.HTTPError(
            url,
            404,
            "Not Found",
            {},
            io.BytesIO(
                b'<Error xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                b"<Code>NoSuchKey</Code><Message>missing</Message></Error>"
            ),
        )

    return opener


def test_list_objects_follows_pagination(monkeypatch):
    seen: list[str] = []
    transport = fake_transport(
        {
            "continuation-token=page2": LISTING_PAGE_2.encode(),
            "list-type=2": LISTING_PAGE.encode(),
        },
        seen,
    )
    monkeypatch.setattr(urllib.request, "urlopen", transport)
    entries = list_objects(CREDS, "examplebucket", "docs/")
    assert entries == [
        ("docs/deploy.md", 1200),
        ("docs/oncall.md", 2400),
        ("archive/dump.zip", 900),
        ("docs/empty.md", 0),
    ]
    assert "list-type=2" in seen[0] and "prefix=docs%2F" in seen[0]
    assert "continuation-token=page2" in seen[1]


def test_list_objects_reports_http_errors(monkeypatch):
    import urllib.request

    def forbidden(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url,
            403,
            "Forbidden",
            {},
            io.BytesIO(
                b'<Error xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                b"<Message>Access Denied</Message></Error>"
            ),
        )

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    with pytest.raises(S3Error) as excinfo:
        list_objects(CREDS, "examplebucket")
    assert "access denied" in str(excinfo.value) and "Access Denied" in str(excinfo.value)


def test_anonymous_access_sends_no_authorization(monkeypatch):
    import urllib.request

    captured: list[dict] = []

    def opener(request, timeout=None):
        captured.append({key.lower(): value for key, value in request.headers.items()})
        return FakeResponse(LISTING_PAGE_2.encode())

    monkeypatch.setattr(urllib.request, "urlopen", opener)
    anonymous = {**CREDS, "access_key": "", "secret_key": ""}
    list_objects(anonymous, "public-bucket")
    assert "authorization" not in captured[0]


def test_get_object_encodes_each_segment(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(
        "urllib.request.urlopen",
        fake_transport({"reports/": b"pdf bytes", "list-type=2": LISTING_PAGE_2.encode()}, seen),
    )
    data = get_object(CREDS, "b", "reports/2026 Q1.pdf")
    assert data == b"pdf bytes"
    assert seen[0].endswith("/reports/2026%20Q1.pdf")


def test_sync_s3_indexes_objects(tmp_path: Path, monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(
        "urllib.request.urlopen",
        fake_transport(
            {
                "continuation-token=page2": LISTING_PAGE_2.encode(),
                "list-type=2": LISTING_PAGE.encode(),
                "docs/deploy.md": b"Deploys use Terraform and Helm. Rollback with kubectl undo.",
                "docs/oncall.md": b"Pages are acknowledged within 5 minutes.",
                "docs/empty.md": b"   ",
            },
            seen,
        ),
    )
    with Store(tmp_path / "index.db") as store:
        stats = sync_s3(store, HashingEmbedder(dim=512), bucket="examplebucket", prefix="docs/")
        assert stats.indexed == 2 and stats.chunks == 2
        assert [row["path"] for row in store.documents()] == [
            "s3://examplebucket/docs/deploy.md",
            "s3://examplebucket/docs/oncall.md",
        ]
    reasons = {row["path"]: row["reason"] for row in stats.skipped_samples}
    assert reasons["archive/dump.zip"] == "unsupported or too large"
    assert reasons["docs/empty.md"] == "no extractable text"


def test_sync_s3_respects_the_limit(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        fake_transport(
            {
                "list-type=2": LISTING_PAGE_2.encode(),
                "docs/empty.md": b"the only object it may touch",
            },
            [],
        ),
    )
    with Store(tmp_path / "index.db") as store:
        stats = sync_s3(store, HashingEmbedder(dim=256), bucket="b", limit=1)
        assert stats.indexed == 1
        assert len(store.documents()) == 1


def test_sync_s3_requires_a_bucket(tmp_path: Path):
    with Store(tmp_path / "index.db") as store:
        with pytest.raises(S3Error):
            sync_s3(store, HashingEmbedder(dim=64), bucket="  ")

