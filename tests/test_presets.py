from __future__ import annotations

import pytest

from ragdesk.presets import DEFAULT_PRESET, PRESETS, resolve


def test_resolve_fills_from_preset():
    settings = resolve("light")
    assert settings["embedder"] == "onnx"
    assert settings["rerank"] == "none"
    assert settings["llm"]


def test_explicit_flags_win_over_preset():
    settings = resolve("light", embedder="hash", rerank="lexical", llm="my-model")
    assert settings["embedder"] == "hash"
    assert settings["rerank"] == "lexical"
    assert settings["llm"] == "my-model"


def test_all_presets_resolve():
    for name in PRESETS:
        settings = resolve(name)
        assert settings["preset"] == name
        assert settings["embedder"]


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        resolve("nope")


def test_default_preset_is_the_lightest_tier():
    """Fresh installs (empty saved setting) must start on the cheapest setup."""
    from ragdesk import settings as settings_mod

    assert settings_mod.DEFAULTS["preset"] == ""          # nothing pinned at install
    assert DEFAULT_PRESET == "light"                       # …so this is what runs
    assert PRESETS["light"]["rerank"] == "none"            # lightest: no reranker
    assert PRESETS["balanced"]["rerank"] != "none"         # upgrades add weight
    assert PRESETS[DEFAULT_PRESET]["note"].startswith("8GB")
