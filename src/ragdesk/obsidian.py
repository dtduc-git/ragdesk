"""Obsidian vault support: index a vault the way its owner wrote it.

A vault is a normal folder of markdown, so the indexer already handles it; what
this module adds is the vault's own vocabulary — front-matter `aliases` become
searchable text, `tags` and the vault name become filterable metadata, and the
config directories (``.obsidian``, ``.trash``) are never walked. Wikilinks stay
in the text where they are readable, and the existing backlinks lookup (FTS over
file names) already resolves them to the target note.
"""

from __future__ import annotations

import re

from ragdesk.index import parse_front_matter

WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)")   # [[Note]], [[Note|label]], [[Note#heading]]
TAG_RE = re.compile(r"(?<![\w/])#([A-Za-z][\w/-]{1,40})")
MAX_ALIASES = 12


def parse_aliases(raw: str) -> list[str]:
    """`aliases: one, two` and YAML-list-ish `[one, two]` both work."""
    cleaned = raw.strip().strip("[]")
    return [
        alias.strip().strip("'\"")
        for alias in cleaned.split(",")
        if alias.strip().strip("'\"")
    ][:MAX_ALIASES]


def vault_metadata(vault: str, content: str) -> tuple[dict[str, str], str]:
    """(metadata, content) for one note: tags, aliases and the vault name.

    Aliases are prepended as a plain line so retrieval can find the note by the
    name its owner actually uses; everything else lands in metadata for filters
    like ``vault:notes tag:kubernetes``.
    """
    meta, body = parse_front_matter(content)  # index_document sees the body only
    aliases = parse_aliases(meta.get("aliases", "") or meta.get("alias", ""))
    tags = [
        tag
        for tag in (meta.get("tags", "") or "").replace("[", "").replace("]", "").split(",")
        if tag.strip()
    ]
    tags += TAG_RE.findall(body)
    seen: list[str] = []
    for tag in tags:
        tag = tag.strip().strip("'\"")
        if tag and tag not in seen:
            seen.append(tag)
    metadata = {**meta, "vault": vault}
    if seen:
        metadata["tags"] = ",".join(seen[:12])
    if aliases:
        metadata["aliases"] = ",".join(aliases)
        body = f"Aliases: {', '.join(aliases)}\n\n{body}"
    return metadata, body


def vault_name(path: str) -> str:
    """The name shown in `vault:` filters: the folder's own name."""
    from pathlib import Path

    return Path(path).name or path


def is_vault(path: str) -> bool:
    from pathlib import Path

    return (Path(path) / ".obsidian").is_dir()


def links_in(content: str, limit: int = 30) -> list[str]:
    """Wikilink targets of one note (deduped) — used by the vault report."""
    seen: list[str] = []
    for match in WIKILINK_RE.findall(content):
        target = match.strip()
        if target and target not in seen:
            seen.append(target)
    return seen[:limit]
