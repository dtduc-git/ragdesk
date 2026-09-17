from __future__ import annotations

from ragdesk import settings as app_settings
from ragdesk.embed import configured_threads


def test_quiet_indexing_is_the_default(monkeypatch, tmp_path):
    """Fresh installs leave the machine usable: 4 embedding threads, not every core."""
    assert app_settings.DEFAULTS["embed_threads"] == 4

    monkeypatch.setattr(app_settings, "settings_file", lambda: tmp_path / "none.json")
    assert app_settings.load()["embed_threads"] == 4
    assert configured_threads() == 4
