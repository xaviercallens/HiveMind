# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Offline tests for ``scripts/gen_todo.py`` (Issue #555)."""

from __future__ import annotations

import json
from pathlib import Path
import re

from scripts.gen_todo import (
    check_todo_file,
    main as gen_todo_main,
    render_todo_markdown,
    strip_season_ids_and_closing_keywords,
)

RECORDED_GH_FIXTURE = {
    "issues": [
        {
            "number": 493,
            "title": "[P0 Security] Cockpit steam/register-login rotates an existing handle's key without auth (IP-141)",
        },
        {
            "number": 522,
            "title": "[P1 Security] /api/astro/gemini-tts is an unauthenticated open proxy (IP-169)",
        },
        {
            "number": 555,
            "title": "[P2 Docs/CI] Generate TODO.md from the open issues in a script and check it in CI",
        },
        {
            "number": 487,
            "title": "[Incident] RTX 2070 failover orchestrator re-pointed all 42 producers to a dead tunnel",
        },
        {
            "number": 269,
            "title": "[S19-W1-9] Newcomer quickstart for Solver Bee + Stretch W1b NL-to-Spec Coder Translator",
        },
        {
            "number": 208,
            "title": "[Epic] Season 18 Operation Energy-Bees (HM-S18-AE-01..05): Fixes #191 and closes #166",
        },
    ],
    "prs": [
        {
            "number": 664,
            "title": "fix(swarm): clamp Multi-Krum token weights (HM-S35-04, resolves #582)",
            "author": {"login": "xaviercallens"},
            "isDraft": False,
        },
        {
            "number": 397,
            "title": "T4 audit: governor telemetry and report (Season 20)",
            "author": {"login": "xaviercallens"},
            "isDraft": True,
        },
    ],
}


def test_gen_todo_check_fails_when_issue_removed_and_passes_when_fresh(tmp_path: Path) -> None:
    """Negative control first: --check fails when one issue is removed from TODO.md and passes when freshly generated."""
    fixture_path = tmp_path / "gh_fixture.json"
    todo_path = tmp_path / "TODO.md"
    fixture_path.write_text(json.dumps(RECORDED_GH_FIXTURE), encoding="utf-8")

    # Generate fresh TODO.md
    rc_gen = gen_todo_main(
        ["--output", str(todo_path), "--fixture-json", str(fixture_path), "--date", "2026-11-06"]
    )
    assert rc_gen == 0

    # Positive control: --check passes on freshly generated TODO.md
    rc_ok = gen_todo_main(
        ["--output", str(todo_path), "--fixture-json", str(fixture_path), "--check"]
    )
    assert rc_ok == 0

    # Negative control: remove one issue line from TODO.md -> --check must exit non-zero
    lines = todo_path.read_text(encoding="utf-8").splitlines()
    mutated = "\n".join(line for line in lines if "issue 522:" not in line) + "\n"
    todo_path.write_text(mutated, encoding="utf-8")

    rc_fail = gen_todo_main(
        ["--output", str(todo_path), "--fixture-json", str(fixture_path), "--check"]
    )
    assert rc_fail != 0
    passed, reason = check_todo_file(
        todo_path, RECORDED_GH_FIXTURE["issues"], RECORDED_GH_FIXTURE["prs"]
    )
    assert passed is False
    assert "out of date" in reason


def test_gen_todo_strips_season_ids_and_closing_keywords() -> None:
    """Verify generator output contains no season IDs and no GitHub closing keywords before issue numbers."""
    rendered = render_todo_markdown(
        RECORDED_GH_FIXTURE["issues"],
        RECORDED_GH_FIXTURE["prs"],
        generated_date="2026-11-06",
    )
    assert "Open issues: 6. Open pull requests: 2 (1 draft)." in rendered
    assert "## P0: security or safety, fix first (1)" in rendered
    assert "## P1: a claim the code does not back, or a verifier that cannot fail (1)" in rendered
    assert "## P2: correctness and CI (1)" in rendered
    assert "## Incidents (1)" in rendered
    assert "## Other open issues (honesty, experiments, UX, federation) (2)" in rendered
    assert "- PR 397 (draft) by xaviercallens:" in rendered

    # No season IDs (S19, HM-S18, HM-S35, Season 18, Season 20)
    assert re.search(r"\b(?:HM-)?S\d+\b", rendered) is None
    assert re.search(r"\bSeason\s+\d+\b", rendered, flags=re.IGNORECASE) is None

    # No closing keywords followed by an issue number
    assert re.search(
        r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#?\d+\b",
        rendered,
        flags=re.IGNORECASE,
    ) is None
    assert strip_season_ids_and_closing_keywords("Fixes #555 in HM-S35-02") == "issue 555 in"
