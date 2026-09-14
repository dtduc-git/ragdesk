from __future__ import annotations

from pathlib import Path

from ragdesk.buildenv import CONST_ORDER, render_build_env, write_build_env


def test_write_build_env_maps_and_refuses_confidential(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text(
        "RAGDESK_GITHUB_CLIENT_ID=gh-1\n"
        "GDRIVE_CLIENT_ID=g-1\n"
        "GDRIVE_CLIENT_SECRET=g-secret\n"
        "RAGDESK_ATLASSIAN_CLIENT_ID=at-1\n"
        "RAGDESK_ATLASSIAN_CLIENT_SECRET=at-secret\n"
    )
    out = tmp_path / "_build_env.py"
    mapped = write_build_env(env, out)

    assert mapped["GITHUB_CLIENT_ID"] == "gh-1"
    assert mapped["GOOGLE_CLIENT_ID"] == "g-1"
    assert mapped["GOOGLE_CLIENT_SECRET"] == "g-secret"
    assert mapped["ATLASSIAN_CLIENT_ID"] == "at-1"
    assert mapped["ATLASSIAN_CLIENT_SECRET"] == ""  # confidential: refused

    text = out.read_text()
    assert "gh-1" in text
    assert "at-secret" not in text


def test_write_build_env_without_env_file(tmp_path: Path):
    out = tmp_path / "_build_env.py"
    mapped = write_build_env(tmp_path / "missing.env", out)
    assert all(value == "" for value in mapped.values())
    assert out.is_file()


def test_render_contains_all_constants():
    text = render_build_env({})
    for const in CONST_ORDER:
        assert f"{const} = " in text
