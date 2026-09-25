"""Static consistency checks between the UI script and its HTML shell.

Cheap guard with a real precedent: the MCP card rendered nothing because its
status elements were never added to ``index.html``, so ``loadMcp()`` threw on
boot and the whole feature stayed invisible.
"""

from __future__ import annotations

import re
from pathlib import Path

DESKTOP = Path(__file__).resolve().parent.parent / "desktop"

# ids the wizard renders at runtime (and the bookmark list inside the web card)
DYNAMIC_IDS = {
    "bookmark-list",
    "wizard-finish",
    "wizard-index",
    "wizard-index-state",
    "wizard-openai",
    "wizard-pick-files",
    "wizard-pick-folder",
}


def test_every_referenced_element_id_exists_in_the_html():
    script = (DESKTOP / "src" / "main.ts").read_text()
    html = (DESKTOP / "index.html").read_text()
    referenced = set(re.findall(r'\$\("([a-zA-Z0-9_-]+)"\)', script))
    declared = set(re.findall(r'id="([a-zA-Z0-9_-]+)"', html))
    missing = sorted(referenced - declared - DYNAMIC_IDS)
    assert not missing, f"main.ts references elements that do not exist: {missing}"


def test_the_mcp_card_is_wired():
    html = (DESKTOP / "index.html").read_text()
    assert 'id="mcp-state"' in html and 'id="mcp-dot"' in html
    for kind in ("claude_code", "claude_desktop", "codex"):
        assert f'data-mcp="{kind}"' in html


SYNC_FORMS = [
    "github-sync",
    "gitlab-sync",
    "confluence-sync",
    "gdrive-sync",
    "msgraph-sync",
    "notion-sync",
    "email-sync",
    "web-sync",
    "s3-sync",
]


def test_every_sync_form_offers_keep_in_sync():
    """README promises the tick on every connector; four forms once lacked it."""
    script = (DESKTOP / "src" / "main.ts").read_text()
    for form in SYNC_FORMS:
        match = re.search(rf'<form data-form="{form}"(.*?)</form>', script, re.DOTALL)
        assert match, f"main.ts no longer renders the {form} form"
        assert "keepToggle()" in match.group(1), f"{form} lost its Keep in sync toggle"
