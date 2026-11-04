# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""scripts/gen_todo.py builds TODO.md from recorded issue and pull request data and detects a stale file (no network)."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import pytest

_SPEC = importlib.util.spec_from_file_location("gen_todo", Path(__file__).resolve().parent.parent / "scripts" / "gen_todo.py")
assert _SPEC is not None and _SPEC.loader is not None
gen = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gen)

ISSUES: list[dict[str, Any]] = [
    {"number": 12, "title": "[P0 Security] key committed in a module"},
    {"number": 10, "title": "[P1 Swarm] gate fails open | HM-S16-06: on NaN"},
    {"number": 9, "title": "[Incident] orchestrator re-pointed producers"},
    {"number": 7, "title": "Join page misses the status table"},
]
PRS: list[dict[str, Any]] = [
    {"number": 40, "title": "fix: something", "author": {"login": "xaviercallens"}, "isDraft": False},
    {"number": 39, "title": "draft work", "author": {"login": "someone"}, "isDraft": True},
]


def test_groups_by_priority_and_cleans_titles() -> None:
    text = gen.build_todo(ISSUES, PRS, "2026-11-04")
    assert text.index("## P0") < text.index("## P1") < text.index("## Incidents") < text.index("## Other open issues")
    assert "- [ ] issue 12: [P0 Security] key committed in a module" in text
    assert "HM-S16" not in text and "|" not in text
    assert "Open issues: 4. Open pull requests: 2 (1 drafts)." in text
    assert "- PR 39 (draft) by someone: draft work" in text


def test_check_detects_a_removed_issue_and_ignores_the_date() -> None:
    committed = gen.build_todo(ISSUES, PRS, "2026-11-01")
    assert gen.is_current(committed, ISSUES, PRS)  # a later day alone is not a difference
    assert not gen.is_current(committed, ISSUES[:-1], PRS)
    assert not gen.is_current(committed, ISSUES, PRS[:-1])


def test_cli_writes_then_checks_from_recorded_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "i.json").write_text(json.dumps(ISSUES), encoding="utf-8")
    (tmp_path / "p.json").write_text(json.dumps(PRS), encoding="utf-8")
    todo = tmp_path / "TODO.md"
    base = ["--issues", str(tmp_path / "i.json"), "--prs", str(tmp_path / "p.json"), "--todo", str(todo)]
    assert gen.main(["--check", *base]) == 1  # no file yet
    assert gen.main(base) == 0 and re.search(r"^Generated \d{4}-\d{2}-\d{2} ", todo.read_text(encoding="utf-8"), re.M)
    assert gen.main(["--check", *base]) == 0
    (tmp_path / "i.json").write_text(json.dumps(ISSUES[:-1]), encoding="utf-8")
    assert gen.main(["--check", *base]) == 1
    assert "out of date" in capsys.readouterr().err
