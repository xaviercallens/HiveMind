#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""CLI Dispatcher for HiveMind v3.4.0 Pioneer Outreach & Nov 14–15, 2026 CrowdGPT Swarm Raid.

Coordinates outreach from `callensxavier@gmail.com` across:
- Gmail (`gmail mutate create-draft` with `--html` inline CSS + `.ics` attachment + strict `BCC` privacy)
- Live HiveMind AMCP / Kafka mesh (`broadcast` + all 16 pioneer community handles)
- GitHub (1-click Discussion, Community Call Issue, and Release `v3.4.0` URLs + optional `gh` publish)
- Discord (`#announcements` Markdown + optional `DISCORD_WEBHOOK_URL` webhook push)
- Reddit (`r/LocalLLaMA`, `r/BOINC`, `r/MachineLearning`, `r/homelab` 1-click submit links)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hivemind.router.community_outreach_v34 import (  # noqa: E402
    OWNER_SENDER_EMAIL,
    PUBLIC_OPENROUTER_URL,
    VERIFIED_PIONEER_EMAILS,
    build_v34_discord_outreach,
    build_v34_github_outreach,
    build_v34_pioneer_email,
    build_v34_reddit_outreach,
    publish_v34_amcp_pioneer_outreach,
    send_discord_webhook,
)

ICS_CALENDAR_PATH = REPO_ROOT / "docs" / "calendar" / "hivemind_crowdgpt_weekend_nov14_15_2026.ics"


def _discover_hivemind_api_key() -> str:
    env_key = os.environ.get("HIVEMIND_API_KEY", "").strip()
    if env_key:
        return env_key
    cfg_dir = Path.home() / ".config" / "hivemind"
    if cfg_dir.exists():
        for p in cfg_dir.rglob("*"):
            if p.is_file() and p.stat().st_size < 200_000:
                try:
                    txt = p.read_text(encoding="utf-8", errors="ignore")
                    m = re.search(r"(hm[-_][A-Za-z0-9_-]{16,})", txt)
                    if m:
                        return m.group(1)
                except OSError:
                    continue
    return ""


def try_create_gmail_draft(email_pkg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Create a ready-to-send Gmail draft on `callensxavier@gmail.com` if `gmail` CLI is authenticated."""
    out_dir.mkdir(parents=True, exist_ok=True)
    html_file = out_dir / "v34_pioneer_community_email.html"
    md_file = out_dir / "v34_pioneer_community_email.md"
    html_file.write_text(email_pkg["html_body"], encoding="utf-8")
    md_file.write_text(email_pkg["markdown_body"], encoding="utf-8")

    gmail_bin = shutil.which("gmail")
    if not gmail_bin:
        return {
            "draft_created": False,
            "reason": "gmail_cli_not_on_path",
            "html_file": str(html_file),
            "markdown_file": str(md_file),
            "gmail_web_compose_url": email_pkg["gmail_web_compose_url"],
        }

    cmd = [
        gmail_bin,
        "mutate",
        "create-draft",
        "--to",
        OWNER_SENDER_EMAIL,
        "--bcc",
        ",".join(email_pkg["bcc_emails"]),
        "--subject",
        email_pkg["subject"],
        "--html",
        "--body-file",
        str(html_file),
    ]
    if ICS_CALENDAR_PATH.exists():
        cmd.extend(["--file", str(ICS_CALENDAR_PATH)])

    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode == 0:
        return {
            "draft_created": True,
            "stdout": res.stdout.strip(),
            "to": OWNER_SENDER_EMAIL,
            "bcc_count": len(email_pkg["bcc_emails"]),
            "html_file": str(html_file),
        }
    return {
        "draft_created": False,
        "reason": (res.stderr or res.stdout).strip()[:300],
        "html_file": str(html_file),
        "markdown_file": str(md_file),
        "gmail_web_compose_url": email_pkg["gmail_web_compose_url"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prepare & dispatch HiveMind v3.4.0 multi-channel pioneer outreach")
    ap.add_argument("--out-dir", type=Path, default=Path.home() / ".cache" / "hivemind_outreach_v34")
    ap.add_argument("--create-gmail-draft", action="store_true", help="Create Gmail draft via gmail CLI if logged in")
    ap.add_argument("--publish-amcp", action="store_true", help="Send live AMCP messages to broadcast + 16 pioneers")
    ap.add_argument("--publish-discord", action="store_true", help="Post to DISCORD_WEBHOOK_URL if configured")
    ap.add_argument("--gateway", default=PUBLIC_OPENROUTER_URL)
    args = ap.parse_args(argv)

    email_pkg = build_v34_pioneer_email(
        sender_email=OWNER_SENDER_EMAIL,
        bcc_recipients=VERIFIED_PIONEER_EMAILS,
    )
    gh_pkg = build_v34_github_outreach()
    discord_pkg = build_v34_discord_outreach()
    reddit_pkg = build_v34_reddit_outreach()

    summary: dict[str, Any] = {
        "sender_email": OWNER_SENDER_EMAIL,
        "bcc_pioneer_count": len(email_pkg["bcc_emails"]),
        "bcc_pioneers": email_pkg["bcc_emails"],
        "gmail_web_compose_url": email_pkg["gmail_web_compose_url"],
        "github_urls": {
            "discussion_new_url": gh_pkg["discussion_new_url"],
            "issue_new_url": gh_pkg["issue_new_url"],
            "release_v340_new_url": gh_pkg["release_v340_new_url"],
            "raid_template_url": gh_pkg["raid_template_url"],
        },
        "discord_invite_url": discord_pkg["discord_invite_url"],
        "reddit_subreddits": [{"subreddit": r["subreddit"], "submit_url": r["submit_url"]} for r in reddit_pkg],
    }

    if args.create_gmail_draft:
        summary["gmail_draft"] = try_create_gmail_draft(email_pkg, args.out_dir)

    if args.publish_amcp:
        api_key = _discover_hivemind_api_key()
        if not api_key:
            raise RuntimeError("HIVEMIND_API_KEY not found in environment or ~/.config/hivemind")
        receipts = publish_v34_amcp_pioneer_outreach(gateway_url=args.gateway, api_key=api_key)
        summary["amcp_dispatched_count"] = len(receipts)
        summary["amcp_receipts_sample"] = receipts[:5]
        summary["amcp_all_ok"] = all(r.get("status") == "ok" for r in receipts)

    if args.publish_discord:
        wh = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
        if wh:
            summary["discord_webhook"] = send_discord_webhook(wh, discord_pkg["webhook_payload"])
        else:
            summary["discord_webhook"] = {"delivered": False, "reason": "DISCORD_WEBHOOK_URL_not_set"}

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
