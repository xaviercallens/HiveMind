# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Generate TODO.md from the open GitHub issues and pull requests, or check that the committed file is current.

    python scripts/gen_todo.py            # fetch with the gh CLI and rewrite TODO.md
    python scripts/gen_todo.py --check    # exit 1 when TODO.md differs from what the open issues and PRs produce
    python scripts/gen_todo.py --issues issues.json --prs prs.json   # use recorded data, no network

The date line is not part of the comparison, so a check does not fail only because a day went by.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
TODO_PATH = REPO_ROOT / "TODO.md"
GENERATED_LINE = re.compile(r"^Generated \d{4}-\d{2}-\d{2} ", re.M)
SEASON_ID = re.compile(r"\bHM-S\d+(?:-\d+)*\b[:,]?\s*")
SECTIONS: list[tuple[str, str]] = [
    ("P0", "P0: security or safety, fix first"),
    ("P1", "P1: a claim the code does not back, or a verifier that cannot fail"),
    ("P2", "P2: correctness and CI"),
    ("INC", "Incidents"),
    ("OTHER", "Other open issues (honesty, experiments, UX, federation)"),
]


def clean_title(title: str) -> str:
    """Drop season ids (docs rule) and table-breaking pipes from an issue or PR title."""
    return SEASON_ID.sub("", title).replace("|", "/").strip()


def priority_of(title: str) -> str:
    match = re.match(r"\[(P\d)", title)
    if match:
        return match.group(1) if match.group(1) in ("P0", "P1", "P2") else "OTHER"
    return "INC" if title.startswith("[Incident") else "OTHER"


def build_todo(issues: list[dict[str, Any]], prs: list[dict[str, Any]], today: str) -> str:
    """Return the TODO.md text for the given open issues and pull requests."""
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for issue in sorted(issues, key=lambda i: int(i["number"]), reverse=True):
        grouped[priority_of(str(issue["title"]))].append(issue)
    drafts = sum(1 for p in prs if p.get("isDraft"))
    lines = [
        "<!-- Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0. -->",
        "# TODO: what is still open",
        "",
        f"Generated {today} from the repository's open GitHub issues and pull requests, not from memory. Nothing here claims that something works; "
        "it lists what is not finished. Re-generate it with scripts/gen_todo.py instead of hand-editing the counts.",
        "",
        f"Open issues: {len(issues)}. Open pull requests: {len(prs)} ({drafts} drafts).",
        "",
        "Rules for anyone taking an item: AGPL header on every file under src/hivemind; no stub bodies and no mock_, simulate_ or dummy_ identifiers "
        "outside tests/; a new test must fail on unfixed main before it passes; no fabricated numbers; merges are owner-only through the hourly gatekeeper.",
        "",
        "What is real and what is simulated across the codebase is audited in [docs/MVP_REALITY_AUDIT.md](docs/MVP_REALITY_AUDIT.md); the smoke path is in "
        "[docs/MVP_E2E_SMOKE.md](docs/MVP_E2E_SMOKE.md). Per-row status lives in the dated docs/IMPROVEMENT_PLAN_*.md files.",
        "",
    ]
    for key, name in SECTIONS:
        if grouped[key]:
            lines += [f"## {name} ({len(grouped[key])})", ""]
            lines += [f"- [ ] issue {i['number']}: {clean_title(str(i['title']))}" for i in grouped[key]]
            lines.append("")
    lines += ["## Open pull requests", "", "Owner pull requests are merged by the hourly gatekeeper when every check passes; the rest are held.", ""]
    for pr in sorted(prs, key=lambda p: int(p["number"]), reverse=True):
        author = (pr.get("author") or {}).get("login", "unknown")
        lines.append(f"- PR {pr['number']}{' (draft)' if pr.get('isDraft') else ''} by {author}: {clean_title(str(pr['title']))[:110]}")
    return "\n".join(lines) + "\n"


def without_date(text: str) -> str:
    """The text with the day in the Generated line blanked, for comparison."""
    return GENERATED_LINE.sub("Generated DATE ", text)


def is_current(committed: str, issues: list[dict[str, Any]], prs: list[dict[str, Any]]) -> bool:
    """True when the committed TODO.md matches what the given issues and pull requests produce."""
    return without_date(committed) == without_date(build_todo(issues, prs, "1970-01-01"))


def fetch(kind: str) -> list[dict[str, Any]]:
    fields = "number,title" if kind == "issue" else "number,title,author,isDraft"
    out = subprocess.run(["gh", kind, "list", "--state", "open", "--limit", "1000", "--json", fields], capture_output=True, text=True, check=True, timeout=120)
    data = json.loads(out.stdout)
    if not isinstance(data, list):
        raise ValueError(f"gh {kind} list did not return a list")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="exit 1 when TODO.md is not current; write nothing")
    parser.add_argument("--issues", type=Path, help="JSON list of open issues (number, title) instead of calling gh")
    parser.add_argument("--prs", type=Path, help="JSON list of open pull requests (number, title, author, isDraft) instead of calling gh")
    parser.add_argument("--todo", type=Path, default=TODO_PATH, help="path of the TODO.md to write or check")
    args = parser.parse_args(argv)
    issues = json.loads(args.issues.read_text(encoding="utf-8")) if args.issues else fetch("issue")
    prs = json.loads(args.prs.read_text(encoding="utf-8")) if args.prs else fetch("pr")
    if args.check:
        if not args.todo.exists():
            print(f"{args.todo} does not exist", file=sys.stderr)
            return 1
        if is_current(args.todo.read_text(encoding="utf-8"), issues, prs):
            print(f"{args.todo.name} is current ({len(issues)} issues, {len(prs)} pull requests)")
            return 0
        print(f"{args.todo.name} is out of date: regenerate it with scripts/gen_todo.py", file=sys.stderr)
        return 1
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    args.todo.write_text(build_todo(issues, prs, today), encoding="utf-8")
    print(f"wrote {args.todo} ({len(issues)} issues, {len(prs)} pull requests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
