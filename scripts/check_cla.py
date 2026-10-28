#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Contributor License Agreement (CLA) verification utility for Project HiveMind (AGPL-3.0 Dual License).

Verifies that a Pull Request author has signed `.github/CLA.md` either:
1. By being listed in `.github/cla-signatures.json`, or
2. By posting the exact phrase `"I have read the CLA Document and I hereby sign the CLA"` in the PR body or comments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CLA_SIGNATURES_PATH = REPO_ROOT / ".github" / "cla-signatures.json"
REQUIRED_PHRASE = "I have read the CLA Document and I hereby sign the CLA"
ACCEPTED_PHRASES = (
    REQUIRED_PHRASE.lower(),
    "i have read and agree to the project hive mind contributor license agreement (cla.md v1.0)".lower(),
)


def verify_cla_signature(
    author: str,
    pr_body: str = "",
    comments: list[dict[str, str]] | None = None,
    signatures_path: Path = CLA_SIGNATURES_PATH,
) -> tuple[bool, str]:
    """Return (signed, reason) for a PR author."""
    ledger = json.loads(signatures_path.read_text(encoding="utf-8"))
    phrase = ledger.get("required_phrase", REQUIRED_PHRASE).strip().lower()
    accepted = {phrase, *ACCEPTED_PHRASES}
    signed_users = {u["github_login"].lower() for u in ledger.get("signed_contributors", [])}
    exempt_bots = {b.lower() for b in ledger.get("exempt_bots", [])}
    author_lower = author.strip().lower()

    if author_lower in signed_users:
        return True, f"@{author} is registered in .github/cla-signatures.json"
    if author_lower in exempt_bots:
        return True, f"@{author} is an exempt automated bot"
    body_lower = (pr_body or "").lower()
    if any(p in body_lower for p in accepted):
        return True, f"@{author} signed the CLA in the PR body"

    for c in comments or []:
        c_user = (c.get("user") or "").strip().lower()
        c_body = (c.get("body") or "").lower()
        if c_user == author_lower and any(p in c_body for p in accepted):
            return True, f"@{author} signed the CLA via PR comment"

    return (
        False,
        f"@{author} has not signed CLA.md (.github/CLA.md). Comment `{REQUIRED_PHRASE}` on the PR to sign.",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--author", required=True, help="GitHub login of PR author")
    ap.add_argument("--pr-body", default="", help="Pull request body text")
    args = ap.parse_args()
    ok, reason = verify_cla_signature(args.author, args.pr_body)
    print(json.dumps({"signed": ok, "author": args.author, "reason": reason}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
