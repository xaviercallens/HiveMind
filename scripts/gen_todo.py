# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Generate or verify ``TODO.md`` from open GitHub issues and pull requests (Issue #555).

Groups open issues by ``[P0]`` / ``[P1]`` / ``[P2]`` / ``[Incident]`` prefix, strips season IDs
and GitHub closing keywords before issue numbers, lists open pull requests, and provides a
``--check`` mode that exits non-zero when ``TODO.md`` differs from the generated text.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TODO_PATH = PROJECT_ROOT / "TODO.md"

_SEASON_TOKEN_RE = re.compile(
    r"\b(?:CAMP-S\d+(?:-[A-Za-z0-9_]+)*|HM-S\d+(?:-[A-Za-z0-9_]+)*|S\d+(?:-[A-Za-z0-9_]+)*|Season\s+\d+(?:\s*/\s*\d+)?)\b"
)
_CLOSING_KEYWORD_RE = re.compile(
    r"(?i)\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#?(\d+)\b"
)
_GENERATED_DATE_RE = re.compile(r"^Generated (\d{4}-\d{2}-\d{2}) from the repository's open GitHub issues", re.MULTILINE)


def strip_season_ids_and_closing_keywords(text: str) -> str:
    """Remove season identifiers and neutralize GitHub closing keywords before issue numbers."""
    cleaned = _SEASON_TOKEN_RE.sub("", str(text))
    cleaned = _CLOSING_KEYWORD_RE.sub(r"issue \1", cleaned)
    cleaned = re.sub(r"\[\s*[-:]+\s*", "[", cleaned)
    cleaned = re.sub(r"\(\s*[-:,]+\s*\)", "()", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()


def fetch_open_github_items(repo_root: Path = PROJECT_ROOT) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch open issues and open pull requests from GitHub via the ``gh`` CLI."""
    issue_proc = subprocess.run(
        ["gh", "issue", "list", "--state", "open", "--limit", "500", "--json", "number,title"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    pr_proc = subprocess.run(
        ["gh", "pr", "list", "--state", "open", "--limit", "200", "--json", "number,title,author,isDraft"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    issues = json.loads(issue_proc.stdout)
    prs = json.loads(pr_proc.stdout)
    return list(issues), list(prs)


def render_todo_markdown(
    issues: Sequence[Mapping[str, Any]],
    prs: Sequence[Mapping[str, Any]],
    *,
    generated_date: str | None = None,
) -> str:
    """Render deterministic ``TODO.md`` content from open issues and pull requests."""
    date_str = generated_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sorted_issues = sorted(issues, key=lambda item: int(item["number"]), reverse=True)
    sorted_prs = sorted(prs, key=lambda item: int(item["number"]), reverse=True)

    p0: list[str] = []
    p1: list[str] = []
    p2: list[str] = []
    incidents: list[str] = []
    other: list[str] = []

    for item in sorted_issues:
        num = int(item["number"])
        raw_title = str(item.get("title", "")).strip()
        clean_title = strip_season_ids_and_closing_keywords(raw_title)
        line = f"- [ ] issue {num}: {clean_title}"
        if raw_title.startswith("[P0"):
            p0.append(line)
        elif raw_title.startswith("[P1"):
            p1.append(line)
        elif raw_title.startswith("[P2"):
            p2.append(line)
        elif raw_title.startswith("[Incident]"):
            incidents.append(line)
        else:
            other.append(line)

    draft_count = sum(1 for pr in sorted_prs if bool(pr.get("isDraft")))
    draft_word = "draft" if draft_count == 1 else "drafts"

    pr_lines: list[str] = []
    for pr in sorted_prs:
        num = int(pr["number"])
        author_obj = pr.get("author")
        if isinstance(author_obj, Mapping):
            author_login = str(author_obj.get("login", "unknown"))
        else:
            author_login = str(author_obj or "unknown")
        draft_suffix = " (draft)" if bool(pr.get("isDraft")) else ""
        clean_pr_title = strip_season_ids_and_closing_keywords(str(pr.get("title", "")))[:120]
        pr_lines.append(f"- PR {num}{draft_suffix} by {author_login}: {clean_pr_title}")

    sections: list[str] = [
        "<!-- Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0. -->",
        "# TODO: what is still open",
        "",
        (
            f"Generated {date_str} from the repository's open GitHub issues and pull requests, "
            "not from memory. Nothing here claims that something works; it lists what is not finished. "
            "Re-generate it instead of hand-editing the counts."
        ),
        "",
        f"Open issues: {len(sorted_issues)}. Open pull requests: {len(sorted_prs)} ({draft_count} {draft_word}).",
        "",
        (
            "Rules for anyone taking an item: AGPL header on every file under src/hivemind; "
            "no stub bodies and no mock_, simulate_ or dummy_ identifiers outside tests/; "
            "a new test must fail on unfixed main before it passes; no fabricated numbers; "
            "merges are owner-only through the hourly gatekeeper."
        ),
        "",
        (
            "What is real and what is simulated across the codebase is audited in "
            "[docs/MVP_REALITY_AUDIT.md](docs/MVP_REALITY_AUDIT.md); the smoke path is in "
            "[docs/MVP_E2E_SMOKE.md](docs/MVP_E2E_SMOKE.md). Per-row status lives in the dated "
            "docs/IMPROVEMENT_PLAN_*.md files."
        ),
        "",
        f"## P0: security or safety, fix first ({len(p0)})",
        "",
        *p0,
        "",
        f"## P1: a claim the code does not back, or a verifier that cannot fail ({len(p1)})",
        "",
        *p1,
        "",
        f"## P2: correctness and CI ({len(p2)})",
        "",
        *p2,
        "",
        f"## Incidents ({len(incidents)})",
        "",
        *incidents,
        "",
        f"## Other open issues (honesty, experiments, UX, federation) ({len(other)})",
        "",
        *other,
        "",
        "## Open pull requests",
        "",
        "Owner pull requests are merged by the hourly gatekeeper when every check passes; the rest are held.",
        "",
        *pr_lines,
        "",
    ]
    return "\n".join(sections)


def check_todo_file(
    todo_path: Path,
    issues: Sequence[Mapping[str, Any]],
    prs: Sequence[Mapping[str, Any]],
    *,
    generated_date: str | None = None,
) -> tuple[bool, str]:
    """Verify that ``todo_path`` matches the freshly generated markdown."""
    if not todo_path.exists():
        return False, f"{todo_path} does not exist"
    current = todo_path.read_text(encoding="utf-8")
    effective_date = generated_date
    if effective_date is None:
        match = _GENERATED_DATE_RE.search(current)
        if match is not None:
            effective_date = match.group(1)
    expected = render_todo_markdown(issues, prs, generated_date=effective_date)
    if current != expected:
        return False, f"{todo_path} is out of date with open issues/PRs; re-run scripts/gen_todo.py"
    return True, "ok"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate or check TODO.md from open GitHub issues and PRs.")
    parser.add_argument("--output", type=Path, default=DEFAULT_TODO_PATH, help="Path to TODO.md")
    parser.add_argument("--check", action="store_true", help="Exit non-zero if TODO.md differs from generated text")
    parser.add_argument("--fixture-json", type=Path, default=None, help="Offline JSON fixture with issues and prs")
    parser.add_argument("--date", type=str, default=None, help="Override YYYY-MM-DD generation date")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.fixture_json is not None:
        payload = json.loads(args.fixture_json.read_text(encoding="utf-8"))
        issues = list(payload.get("issues", []))
        prs = list(payload.get("prs", []))
    else:
        issues, prs = fetch_open_github_items()

    if args.check:
        passed, reason = check_todo_file(args.output, issues, prs, generated_date=args.date)
        if not passed:
            print(reason, file=sys.stderr)
            return 1
        return 0

    rendered = render_todo_markdown(issues, prs, generated_date=args.date)
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
