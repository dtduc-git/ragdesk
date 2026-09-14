from __future__ import annotations

from ragdesk.confluence import resolve_oauth_client
from ragdesk.gdrive import resolve_client_credentials
from ragdesk.github import resolve_client_id


def _clear(monkeypatch) -> None:
    for name in (
        "RAGDESK_GITHUB_CLIENT_ID",
        "RAGDESK_ATLASSIAN_CLIENT_ID",
        "RAGDESK_ATLASSIAN_CLIENT_SECRET",
        "GDRIVE_CLIENT_ID",
        "GDRIVE_CLIENT_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("ragdesk.github.credentials.get", lambda provider: {})
    monkeypatch.setattr("ragdesk.confluence.credentials.get", lambda provider: {})
    monkeypatch.setattr("ragdesk.gdrive.credentials.get", lambda provider: {})


def test_shipped_defaults_are_used(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setattr("ragdesk.defaults.GITHUB_CLIENT_ID", "shipped-gh")
    monkeypatch.setattr("ragdesk.defaults.ATLASSIAN_CLIENT_ID", "shipped-at")
    monkeypatch.setattr("ragdesk.defaults.ATLASSIAN_CLIENT_SECRET", "shipped-at-secret")
    monkeypatch.setattr("ragdesk.defaults.GOOGLE_CLIENT_ID", "shipped-g")
    monkeypatch.setattr("ragdesk.defaults.GOOGLE_CLIENT_SECRET", "shipped-g-secret")

    assert resolve_client_id() == "shipped-gh"
    assert resolve_oauth_client() == ("shipped-at", "shipped-at-secret")
    assert resolve_client_credentials() == ("shipped-g", "shipped-g-secret")


def test_env_wins_over_shipped_defaults(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setattr("ragdesk.defaults.GITHUB_CLIENT_ID", "shipped-gh")
    monkeypatch.setattr("ragdesk.defaults.GOOGLE_CLIENT_ID", "shipped-g")
    monkeypatch.setenv("RAGDESK_GITHUB_CLIENT_ID", "env-gh")
    monkeypatch.setenv("GDRIVE_CLIENT_ID", "env-g")
    assert resolve_client_id() == "env-gh"
    assert resolve_client_credentials()[0] == "env-g"


def test_saved_credentials_win_over_shipped_defaults(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setattr("ragdesk.defaults.ATLASSIAN_CLIENT_ID", "shipped-at")
    monkeypatch.setattr("ragdesk.defaults.ATLASSIAN_CLIENT_SECRET", "shipped-at-secret")
    monkeypatch.setattr(
        "ragdesk.confluence.credentials.get",
        lambda provider: {"client_id": "saved-at", "client_secret": "saved-secret"},
    )
    assert resolve_oauth_client() == ("saved-at", "saved-secret")


def test_no_defaults_means_no_client(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setattr("ragdesk.defaults.GITHUB_CLIENT_ID", "")
    monkeypatch.setattr("ragdesk.defaults.ATLASSIAN_CLIENT_ID", "")
    monkeypatch.setattr("ragdesk.defaults.ATLASSIAN_CLIENT_SECRET", "")
    monkeypatch.setattr("ragdesk.defaults.GOOGLE_CLIENT_ID", "")
    assert resolve_client_id() is None
    assert resolve_oauth_client() is None
    assert resolve_client_credentials() == ("", "")
