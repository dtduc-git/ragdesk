from __future__ import annotations

import pytest

from ragdesk.presets import DEFAULT_PRESET, PRESETS, resolve


def test_resolve_fills_from_preset():
    settings = resolve("light")
    assert settings["embedder"] == "onnx"
    assert settings["rerank"].startswith("onnx:")
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

    assert settings_mod.DEFAULTS["preset"] == ""  # nothing pinned at install
    assert DEFAULT_PRESET == "light"  # …so this is what runs
    assert PRESETS["light"]["note"].startswith("8GB")
    # Light reranks with the small model only (measured: +34MB, ~0.8s/question);
    # the heavy one is what the quality tier pays RAM for.
    assert "mmarco" in PRESETS["light"]["rerank"]
    assert "mmarco" in PRESETS["balanced"]["rerank"]
    assert PRESETS["quality"]["rerank"] != PRESETS["light"]["rerank"]
    assert PRESETS[DEFAULT_PRESET]["llm"] != PRESETS["quality"]["llm"]


def test_every_preset_rerank_spec_resolves():
    from ragdesk.rerank import get_reranker, rerank_label

    for name, entry in PRESETS.items():
        assert get_reranker(entry["rerank"]) is not None, f"{name} rerank does not resolve"
        assert rerank_label(entry["rerank"]) != entry["rerank"], f"{name} lacks a short label"


def test_rerank_labels():
    from ragdesk.rerank import MMARCO_RERANK_REPO, rerank_label

    assert rerank_label("none") == "off"
    assert rerank_label("") == "off"
    assert rerank_label(f"onnx:{MMARCO_RERANK_REPO}") == "mmarco-mMiniLMv2"
    assert rerank_label("onnx:someone/odd-model") == "odd-model"
