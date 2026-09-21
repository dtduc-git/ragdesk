from __future__ import annotations

import json
import threading

from ragdesk import settings as app_settings
from ragdesk.embed import configured_threads


def test_quiet_indexing_is_the_default(monkeypatch, tmp_path):
    """Fresh installs leave the machine usable: 4 embedding threads, not every core."""
    assert app_settings.DEFAULTS["embed_threads"] == 4

    monkeypatch.setattr(app_settings, "settings_file", lambda: tmp_path / "none.json")
    assert app_settings.load()["embed_threads"] == 4
    assert configured_threads() == 4


def test_concurrent_saves_do_not_drop_each_others_keys(tmp_path):
    """save() is read-modify-write: without a lock the last writer wins and loses keys."""
    path = tmp_path / "settings.json"
    failures: list[BaseException] = []

    def writer(name: str) -> None:
        try:
            for value in range(15):
                app_settings.save({name: value}, path)
        except BaseException as exc:  # noqa: BLE001 - reported below
            failures.append(exc)

    threads = [threading.Thread(target=writer, args=(f"key{index}",)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not failures
    saved = json.loads(path.read_text())
    assert all(f"key{index}" in saved for index in range(4))
