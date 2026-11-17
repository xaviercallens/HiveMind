#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""CLI Dispatcher for the HiveMind v3.4.0 5-Phase Communication Plan & Nov 14–15, 2026 CrowdGPT Swarm Raid.

Coordinates outreach from `callensxavier@gmail.com` across:
- Web UI & JSON Plan Manifest (`v34_communication_plan_status.json`)
- Gmail (`gmail mutate create-draft` with `--html` inline CSS + `.ics` attachment + strict `BCC` privacy)
- Live HiveMind AMCP / Kafka mesh (`broadcast` proposal + `bee.report` + pioneer direct messages + `--all-fleet`)
- Live Campaign Quorum Poll (`/v1/hivemind/campaigns/poll`)
- GitHub (1-click Discussion, Community Call Issue, and Release `v3.4.0` URLs + Markdown bundle)
- Discord (`#announcements` Markdown + optional `DISCORD_WEBHOOK_URL` webhook push)
- Reddit (`r/LocalLLaMA`, `r/BOINC`, `r/MachineLearning`, `r/homelab` 1-click submit links + JSON bundle)
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
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hivemind.router.community_outreach_v34 import (  # noqa: E402
    OWNER_SENDER_EMAIL,
    PIONEER_COMMUNITY_HANDLES,
    PUBLIC_OPENROUTER_URL,
    VERIFIED_PIONEER_EMAILS,
    build_v34_communication_plan_status,
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


def _discover_additional_fleet_handles(gateway_url: str, max_extra: int = 8) -> list[str]:
    """Fetch additional community/veteran handles from `/v1/hivemind/producers/fleet` without exceeding rate limits."""
    url = gateway_url.rstrip("/") + "/v1/hivemind/producers/fleet"
    try:
        with urllib.request.urlopen(url, timeout=12.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []
    known = {h.lower() for h in PIONEER_COMMUNITY_HANDLES}
    skip_prefixes = ("a1_", "rtx2070_slot_", "e2e_", "kit-check-", "enduser-p-", "gcp-t4-", "t4-big-gpu")
    extras: list[str] = []
    for row in data.get("hiveminders", []):
        h = str(row.get("handle") or row.get("producer_id") or "").strip()
        low = h.lower()
        if not h or low in known or any(low.startswith(pfx) for pfx in skip_prefixes):
            continue
        if low.startswith(("veteran_", "user-", "user.")):
            extras.append(h)
            known.add(low)
            if len(extras) >= max_extra:
                break
    return extras


def register_weekend_campaign_poll(gateway_url: str, api_key: str) -> dict[str, Any]:
    """Register a weekend VRAM pledge on `hivemind.amcp.campaigns.v1` via `/v1/hivemind/campaigns/poll`."""
    endpoint = gateway_url.rstrip("/") + "/v1/hivemind/campaigns/poll"
    payload = {
        "subject_id": "SUBJ-06-LEAN4-FORMAL-PROOFS",
        "participant_kind": "gpu_producer",
        "pledged_vram_gb": 24,
        "availability_slot": "eu_evening_cet",
        "comment": (
            "HiveMind v3.4 x CrowdGPT Weekend Swarm Raid (14-15 Nov 2026): RTX 2070 8GB + GCP Spot T4 16GB "
            "pledged for Producer serving & 8x Sparse-MoE OpenDiLoCo validation."
        ),
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
            "User-Agent": "hivemind-v3.4-campaign-poll/1.1",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20.0) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return {
        "subject_id": payload["subject_id"],
        "pledged_vram_gb": payload["pledged_vram_gb"],
        "changed": body.get("registration", {}).get("changed"),
        "amcp_event_id": (body.get("amcp_event") or {}).get("event_id"),
    }


def export_outreach_bundle(
    out_dir: Path,
    email_pkg: dict[str, Any],
    gh_pkg: dict[str, Any],
    discord_pkg: dict[str, Any],
    reddit_pkg: list[dict[str, str]],
    plan_status: dict[str, Any],
) -> dict[str, str]:
    """Export ready-to-publish files for every communication channel."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "email_html": out_dir / "v34_pioneer_community_email.html",
        "email_md": out_dir / "v34_pioneer_community_email.md",
        "github_discussion_md": out_dir / "v34_github_discussion_and_release.md",
        "discord_announcement_md": out_dir / "v34_discord_announcement.md",
        "reddit_posts_json": out_dir / "v34_reddit_collab_posts.json",
        "plan_status_json": out_dir / "v34_communication_plan_status.json",
    }
    paths["email_html"].write_text(email_pkg["html_body"], encoding="utf-8")
    paths["email_md"].write_text(email_pkg["markdown_body"], encoding="utf-8")
    paths["github_discussion_md"].write_text(gh_pkg["discussion_body"], encoding="utf-8")
    paths["discord_announcement_md"].write_text(discord_pkg["markdown_post"], encoding="utf-8")
    paths["reddit_posts_json"].write_text(json.dumps(reddit_pkg, indent=2), encoding="utf-8")
    paths["plan_status_json"].write_text(json.dumps(plan_status, indent=2), encoding="utf-8")
    return {k: str(v) for k, v in paths.items()}


def try_create_gmail_draft(email_pkg: dict[str, Any], html_file: Path) -> dict[str, Any]:
    """Create a ready-to-send Gmail draft on `callensxavier@gmail.com` if `gmail` CLI is authenticated."""
    gmail_bin = shutil.which("gmail")
    if not gmail_bin:
        return {
            "draft_created": False,
            "reason": "gmail_cli_not_on_path",
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
        }
    return {
        "draft_created": False,
        "reason": (res.stderr or res.stdout).strip()[:300],
        "gmail_web_compose_url": email_pkg["gmail_web_compose_url"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Execute the HiveMind v3.4.0 5-Phase Community Communication Plan")
    ap.add_argument("--out-dir", type=Path, default=Path.home() / ".cache" / "hivemind_outreach_v34")
    ap.add_argument("--create-gmail-draft", action="store_true", help="Create Gmail draft via gmail CLI if logged in")
    ap.add_argument("--publish-amcp", action="store_true", help="Send live AMCP proposal, bee.report & pioneer messages")
    ap.add_argument("--all-fleet", action="store_true", help="Include veteran/user handles discovered on live fleet")
    ap.add_argument("--register-campaign-poll", action="store_true", help="Publish weekend VRAM pledge on campaigns.v1")
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
    plan_status = build_v34_communication_plan_status()

    exported = export_outreach_bundle(args.out_dir, email_pkg, gh_pkg, discord_pkg, reddit_pkg, plan_status)

    summary: dict[str, Any] = {
        "sender_email": OWNER_SENDER_EMAIL,
        "phases_count": len(plan_status["phases"]),
        "bcc_pioneer_count": len(email_pkg["bcc_emails"]),
        "bcc_pioneers": email_pkg["bcc_emails"],
        "exported_bundle": exported,
        "gmail_web_compose_url": email_pkg["gmail_web_compose_url"],
        "github_urls": plan_status["channels"]["github"],
        "discord_invite_url": discord_pkg["discord_invite_url"],
        "reddit_subreddits": plan_status["channels"]["reddit"],
    }

    if args.create_gmail_draft:
        summary["gmail_draft"] = try_create_gmail_draft(email_pkg, Path(exported["email_html"]))

    if args.publish_amcp or args.register_campaign_poll:
        api_key = _discover_hivemind_api_key()
        if not api_key:
            raise RuntimeError("HIVEMIND_API_KEY not found in environment or ~/.config/hivemind")
        if args.register_campaign_poll:
            summary["campaign_poll"] = register_weekend_campaign_poll(args.gateway, api_key)
        if args.publish_amcp:
            target_handles = list(PIONEER_COMMUNITY_HANDLES)
            if args.all_fleet:
                for extra in _discover_additional_fleet_handles(args.gateway, max_extra=6):
                    if extra not in target_handles:
                        target_handles.append(extra)
            receipts = publish_v34_amcp_pioneer_outreach(
                gateway_url=args.gateway,
                api_key=api_key,
                handles=target_handles,
                include_bee_report=True,
            )
            summary["amcp_dispatched_count"] = len(receipts)
            summary["amcp_receipts_sample"] = receipts[:6]
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
