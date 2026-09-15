"""Shell completions and a man page, generated from the real argparse tree.

Generated beats hand-written here: the command and flag lists cannot drift
from the CLI because they come from the same parser that runs it. No
dependency (shtab/argcomplete) — the emitters are plain strings.
"""

from __future__ import annotations

import argparse


def _subparsers(
    parser: argparse.ArgumentParser,
) -> tuple[dict[str, argparse.ArgumentParser], dict[str, str]]:
    """(command → parser, command → one-line help) from argparse's own table."""
    for action in parser._actions:  # noqa: SLF001 - argparse has no public API
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            help_by_name = {
                choice.dest: " ".join(str(choice.help or "").split())
                for choice in getattr(action, "_choices_actions", [])
            }
            return dict(action.choices), help_by_name
    return {}, {}


def _flags(parser: argparse.ArgumentParser) -> list[tuple[str, str]]:
    """(flag, help) for every option, in the order argparse declares them."""
    out: list[tuple[str, str]] = []
    for action in parser._actions:
        help_text = " ".join(str(action.help or "").split())
        for flag in action.option_strings:
            out.append((flag, help_text))
    return out


def _clean(text: str) -> str:
    return (
        text.replace('"', "'")
        .replace("'", "")
        .replace("[", "(")
        .replace("]", ")")
    )


def _zsh_opts(flags: list[tuple[str, str]]) -> str:
    parts = []
    for flag, help_text in flags:
        description = _clean(help_text)
        parts.append(f"'{flag}[{description}]'" if description else f"'{flag}'")
    return " ".join(parts)


def bash_script(parser: argparse.ArgumentParser) -> str:
    commands, _helps = _subparsers(parser)
    global_words = " ".join(flag for flag, _ in _flags(parser))
    cases = "\n".join(
        f'      {name}) words="{global_words} '
        f'{" ".join(flag for flag, _ in _flags(sub))}" ;;'
        for name, sub in commands.items()
    )
    return f"""# bash completion for ragdesk
# install: ragdesk completions bash > /usr/local/etc/bash_completion.d/ragdesk
_ragdesk() {{
  local cur words
  cur="${{COMP_WORDS[COMP_CWORD]}}"
  if [[ $COMP_CWORD -eq 1 ]]; then
    words="{' '.join(commands)} {global_words}"
  else
    words=""
    case "${{COMP_WORDS[1]}}" in
{cases}
    esac
  fi
  COMPREPLY=($(compgen -W "$words" -- "$cur"))
}}
complete -o default -F _ragdesk ragdesk
"""


def zsh_script(parser: argparse.ArgumentParser) -> str:
    commands, helps = _subparsers(parser)
    command_list = " ".join(
        f"'{name}:{_clean(helps.get(name, ''))}'" for name in commands
    )
    cases = "\n".join(
        f"    {name}) opts=({_zsh_opts(_flags(sub))}) ;;"
        for name, sub in commands.items()
    )
    return f"""#compdef ragdesk
_ragdesk() {{
  local -a commands opts options
  commands=({command_list})
  if (( CURRENT == 2 )); then
    _describe -t commands 'ragdesk command' commands
    return
  fi
  options=({_zsh_opts(_flags(parser))})
  case "${{words[2]}}" in
{cases}
    *) opts=() ;;
  esac
  _arguments -s "${{options[@]}}" "${{opts[@]}}" '*:file:_files'
}}
_ragdesk "$@"
"""


def _fish_flag(flag: str, help_text: str) -> str:
    if flag.startswith("--"):
        return f'-l "{flag[2:]}" -d "{_clean(help_text)}"'
    return f'-s "{flag[1:]}" -d "{_clean(help_text)}"'


def fish_script(parser: argparse.ArgumentParser) -> str:
    commands, helps = _subparsers(parser)
    lines = ["# fish completion for ragdesk"]
    for flag, help_text in _flags(parser):
        lines.append(
            f'complete -c ragdesk -n "__fish_use_subcommand" {_fish_flag(flag, help_text)}'
        )
    for name, sub in commands.items():
        lines.append(
            f'complete -c ragdesk -n "__fish_use_subcommand" -a "{name}" '
            f'-d "{_clean(helps.get(name, ""))}"'
        )
        for flag, help_text in _flags(sub):
            lines.append(
                f'complete -c ragdesk -n "__fish_seen_subcommand_from {name}" '
                f"{_fish_flag(flag, help_text)}"
            )
    return "\n".join(lines) + "\n"


def _roff(text: str) -> str:
    cleaned = " ".join(str(text).split()).replace("\\", "\\\\").replace("-", "\\-")
    if cleaned.startswith(".") or cleaned.startswith("'"):
        cleaned = "\\&" + cleaned
    return cleaned


def man_page(parser: argparse.ArgumentParser) -> str:
    commands, helps = _subparsers(parser)
    lines = [
        '.TH RAGDESK 1 "" "ragdesk" "User Commands"',
        ".SH NAME",
        "ragdesk \\- local-first retrieval over your own sources",
        ".SH SYNOPSIS",
        ".B ragdesk",
        "[\\fIglobal options\\fR] \\fIcommand\\fR [\\fIoptions\\fR]",
        ".SH DESCRIPTION",
        _roff(str(parser.description or "")),
        ".SH COMMANDS",
    ]
    for name in commands:
        lines += [
            ".TP",
            f".B {name}",
            _roff(helps.get(name) or "(no description)"),
        ]
    lines.append(".SH GLOBAL OPTIONS")
    for flag, help_text in _flags(parser):
        lines += [".TP", f".B {flag}", _roff(help_text or "(no description)")]
    lines += [
        ".SH EXAMPLES",
        ".nf",
        "ragdesk index ~/Documents",
        'ragdesk ask "when do access tokens expire"',
        "ragdesk save https://example.com/article",
        "ragdesk eval --golden fixtures/golden_repo.jsonl",
        "ragdesk serve",
        ".fi",
        ".SH SEE ALSO",
        "https://github.com/dtduc-git/ragdesk",
    ]
    return "\n".join(lines) + "\n"
