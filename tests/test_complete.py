from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ragdesk.cli import _build_parser
from ragdesk.complete import bash_script, fish_script, man_page, zsh_script

COMMANDS = ("index", "search", "ask", "chat", "eval", "stats", "save", "mcp", "serve")


def test_completion_scripts_cover_the_real_commands():
    parser = _build_parser()
    for emitter in (bash_script, zsh_script, fish_script):
        script = emitter(parser)
        for command in COMMANDS:
            assert command in script, (emitter.__name__, command)
    # bash/zsh write long flags literally; fish spells them -l "json"
    for script in (bash_script(parser), zsh_script(parser)):
        assert "--db" in script
        assert "--json" in script
    fish = fish_script(parser)
    assert '-l "db"' in fish
    assert '-l "json"' in fish


def test_completion_script_shapes():
    parser = _build_parser()
    assert "complete -o default -F _ragdesk ragdesk" in bash_script(parser)
    assert zsh_script(parser).startswith("#compdef ragdesk")
    assert 'complete -c ragdesk -n "__fish_seen_subcommand_from eval"' in fish_script(parser)


def test_man_page_is_roff_with_the_commands():
    parser = _build_parser()
    man = man_page(parser)
    assert man.startswith('.TH RAGDESK 1')
    assert ".SH COMMANDS" in man
    assert ".SH GLOBAL OPTIONS" in man
    assert "ragdesk serve" in man


def test_generated_scripts_parse_in_their_shells(tmp_path: Path):
    parser = _build_parser()
    for shell, script in (("bash", bash_script(parser)), ("zsh", zsh_script(parser))):
        binary = shutil.which(shell)
        if not binary:
            continue
        path = tmp_path / f"ragdesk.{shell}"
        path.write_text(script)
        result = subprocess.run(
            [binary, "-n", str(path)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, f"{shell}: {result.stderr}"
