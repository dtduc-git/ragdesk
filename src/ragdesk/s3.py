"""S3 (and S3-compatible) connector: stdlib SigV4, credentials inside the app.

Why not the aws CLI: it would ask a non-technical user to install and configure
a second tool before ragdesk can read a bucket. Why not boto3: the core is
stdlib-only, and S3's REST API plus SigV4 signing is a bounded, well-specified
algorithm (``hashlib``/``hmac``/``urllib`` + ``xml.etree``). So the user pastes
an access key into the app once — and public buckets work with no key at all.

Credentials resolution: explicit → environment (``AWS_*``) → saved connection.
S3-compatible services (Cloudflare R2, Backblaze B2, MinIO, Wasabi) work by
setting a custom endpoint; those are path-style.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from ragdesk import credentials
from ragdesk.embed import Embedder
from ragdesk.index import IndexStats, extract_bytes, index_document, is_indexable
from ragdesk.store import Store

DEFAULT_REGION = "us-east-1"
DEFAULT_LIMIT = 500
MAX_PAGE_SIZE = 1000
REQUEST_TIMEOUT = 60.0
EMPTY_PAYLOAD_HASH = hashlib.sha256(b"").hexdigest()  # GET/DELETE carry no body
USER_AGENT = "ragdesk-s3/0.1"


class S3Error(RuntimeError):
    """The bucket, the key, or the endpoint refused us."""


# --- credentials ---------------------------------------------------------------


def resolve_credentials(explicit: dict | None = None) -> dict[str, str]:
    """explicit → AWS_* environment → saved connection (first non-empty wins)."""
    values = {
        "access_key": "",
        "secret_key": "",
        "session_token": "",
        "region": "",
        "endpoint": "",
    }
    values.update({key: str(value) for key, value in credentials.get("s3").items() if value})
    env = {
        "access_key": os.environ.get("AWS_ACCESS_KEY_ID", ""),
        "secret_key": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        "session_token": os.environ.get("AWS_SESSION_TOKEN", ""),
        "region": os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "",
        "endpoint": os.environ.get("AWS_ENDPOINT_URL", ""),
    }
    for source in (env, explicit or {}):
        for key, value in source.items():
            if str(value or "").strip():
                values[key] = str(value).strip()
    values["region"] = values["region"] or DEFAULT_REGION
    values["endpoint"] = values["endpoint"].rstrip("/")
    return values


# --- SigV4 ---------------------------------------------------------------------


def _quote(value: str) -> str:
    return urllib.parse.quote(str(value), safe="-_.~")


def _signing_key(secret: str, date: str, region: str, service: str) -> bytes:
    key = f"AWS4{secret}".encode()
    for part in (date, region, service, "aws4_request"):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    return key


def sign_request(
    method: str,
    url: str,
    credentials_map: dict[str, str],
    *,
    service: str = "s3",
    now: datetime | None = None,
    extra_headers: dict[str, str] | None = None,
) -> tuple[str, dict[str, str]]:
    """Return ``(url, headers)`` with a SigV4 Authorization header added.

    The query string is rebuilt in canonical (sorted, RFC 3986) form so the
    signed request and the canonical request cannot drift apart.
    """
    parsed = urllib.parse.urlparse(url)
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    amz_date = moment.strftime("%Y%m%dT%H%M%SZ")
    short_date = amz_date[:8]
    region = credentials_map.get("region") or DEFAULT_REGION

    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    canonical_query = "&".join(f"{_quote(k)}={_quote(v)}" for k, v in sorted(query))
    # S3 does not normalize the path: the canonical URI is exactly what we send
    # (callers build it with per-segment encoding), so no second encode here.
    canonical_path = parsed.path or "/"

    headers = {
        "host": parsed.netloc,
        "x-amz-date": amz_date,
        "x-amz-content-sha256": EMPTY_PAYLOAD_HASH,
    }
    token = credentials_map.get("session_token", "")
    if token:
        headers["x-amz-security-token"] = token
    for key, value in (extra_headers or {}).items():
        headers[key.lower()] = " ".join(str(value).split())

    canonical_headers = "".join(f"{key}:{headers[key]}\n" for key in sorted(headers))
    signed_names = ";".join(sorted(headers))
    canonical_request = "\n".join(
        [
            method,
            canonical_path,
            canonical_query,
            canonical_headers,
            signed_names,
            EMPTY_PAYLOAD_HASH,
        ]
    )
    scope = f"{short_date}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            scope,
            hashlib.sha256(canonical_request.encode()).hexdigest(),
        ]
    )
    signature = hmac.new(
        _signing_key(credentials_map["secret_key"], short_date, region, service),
        string_to_sign.encode(),
        hashlib.sha256,
    ).hexdigest()

    request_headers = {key: value for key, value in headers.items() if key != "host"}
    request_headers["Authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={credentials_map['access_key']}/{scope}, "
        f"SignedHeaders={signed_names}, Signature={signature}"
    )
    request_headers["User-Agent"] = USER_AGENT
    url_with_query = urllib.parse.urlunparse(
        parsed._replace(query=canonical_query, path=canonical_path)
    )
    return url_with_query, request_headers


def _request(
    method: str,
    url: str,
    credentials_map: dict[str, str],
    *,
    timeout: float = REQUEST_TIMEOUT,
) -> bytes:
    """One S3 request: signed when a key is present, anonymous otherwise."""
    headers = {"User-Agent": USER_AGENT}
    if credentials_map.get("access_key") and credentials_map.get("secret_key"):
        url, headers = sign_request(method, url, credentials_map)
    request = urllib.request.Request(url, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise S3Error(_explain_http_error(exc)) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise S3Error(f"cannot reach {urllib.parse.urlparse(url).netloc}: {exc}") from exc


def _strip_namespace(root: ET.Element) -> ET.Element:
    """S3 XML carries a default namespace; drop it so tags match plainly."""
    for element in root.iter():
        if "}" in element.tag:
            element.tag = element.tag.split("}", 1)[1]
    return root


def _explain_http_error(exc: urllib.error.HTTPError) -> str:
    code = exc.code
    detail = ""
    try:
        body = exc.read().decode(errors="replace")
        root = _strip_namespace(
            ET.fromstring(body)  # noqa: S314 - S3 error documents are small
        )
        detail = (root.findtext("Message") or root.findtext("Code") or "").strip()
    except (ET.ParseError, ValueError, OSError):
        pass
    if code == 403:
        base = "access denied — check the access key and its bucket permissions"
    elif code == 404:
        base = "no such bucket"
    elif code == 400:
        base = "bad request (wrong region, or a custom endpoint that needs path-style)"
    else:
        base = f"HTTP {code}"
    return f"{base}{f': {detail}' if detail else ''}"


# --- REST calls -----------------------------------------------------------------


def _bucket_url(credentials_map: dict[str, str], bucket: str) -> str:
    endpoint = credentials_map.get("endpoint", "")
    if endpoint:
        return f"{endpoint}/{bucket}"  # S3-compatible services: path-style
    return f"https://{bucket}.s3.{credentials_map.get('region') or DEFAULT_REGION}.amazonaws.com"


def list_objects(
    credentials_map: dict[str, str],
    bucket: str,
    prefix: str = "",
    *,
    limit: int = MAX_PAGE_SIZE,
) -> list[tuple[str, int]]:
    """(key, size) for up to ``limit`` objects under the prefix."""
    base = _bucket_url(credentials_map, bucket)
    out: list[tuple[str, int]] = []
    token = ""
    while len(out) < limit:
        params = {"list-type": "2", "max-keys": str(min(MAX_PAGE_SIZE, limit - len(out)))}
        if prefix:
            params["prefix"] = prefix
        if token:
            params["continuation-token"] = token
        query = urllib.parse.urlencode(params)
        body = _request("GET", f"{base}?{query}", credentials_map)
        try:
            root = _strip_namespace(
                ET.fromstring(body)  # noqa: S314 - S3 responses, size-bounded by the API
            )
        except ET.ParseError as exc:
            raise S3Error(f"unreadable listing response: {exc}") from exc
        for entry in root.findall(".//Contents"):
            key = entry.findtext("Key") or ""
            try:
                size = int(entry.findtext("Size") or 0)
            except ValueError:
                size = 0
            if key:
                out.append((key, size))
        truncated = (root.findtext("IsTruncated") or "").lower() == "true"
        token = root.findtext("NextContinuationToken") or ""
        if not truncated or not token:
            break
    return out[:limit]


def get_object(credentials_map: dict[str, str], bucket: str, key: str) -> bytes:
    quoted = "/".join(_quote(part) for part in key.split("/"))
    return _request("GET", f"{_bucket_url(credentials_map, bucket)}/{quoted}", credentials_map)


def probe(credentials_map: dict[str, str], bucket: str) -> bool:
    """One cheap listing: proves the key can read the bucket (or that it is public)."""
    return list_objects(credentials_map, bucket, limit=1) is not None


# --- indexing -------------------------------------------------------------------


def sync_s3(
    store: Store,
    embedder: Embedder,
    *,
    bucket: str,
    prefix: str = "",
    limit: int = DEFAULT_LIMIT,
    credentials_map: dict[str, str] | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> IndexStats:
    """Index the objects under ``s3://bucket/prefix``."""
    bucket = bucket.strip()
    if not bucket:
        raise S3Error("bucket is required")
    creds = credentials_map or resolve_credentials()
    store.ensure_embedder(embedder.name, embedder.dim)

    listing = list_objects(creds, bucket, prefix.strip().lstrip("/"), limit=limit)
    stats = IndexStats()
    for key, size in listing:
        stats.files_scanned += 1
        if not is_indexable(Path(key), size):
            stats.skip(Path(key), "unsupported or too large")
            continue
        if progress is not None:
            progress(f"fetching {Path(key).name}", stats.files_scanned, 0)
        try:
            data = get_object(creds, bucket, key)
        except S3Error as exc:
            stats.skip(Path(key), str(exc)[:120])
            continue
        try:
            content = extract_bytes(data, Path(key).name)
        except Exception as exc:  # noqa: BLE001 - one bad object must not stop the sync
            stats.skip(Path(key), f"extract failed: {type(exc).__name__}")
            continue
        if content is None or not content.strip():
            stats.skip(Path(key), "no extractable text")
            continue
        chunks = index_document(
            store,
            embedder,
            source=f"s3:{bucket}",
            path=f"s3://{bucket}/{key}",
            content=content,
            metadata={"bucket": bucket, **({"prefix": prefix} if prefix else {})},
        )
        if chunks:
            stats.indexed += 1
            stats.chunks += chunks
        else:
            stats.unchanged += 1
    return stats
