#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Publisher & Launchpad CLI for the Weekend Nov 14–15, 2026 CrowdGPT-MoE x HiveMind Raid.

Automates:
1. Validating the `.ics` calendar, S21 dataset plan, MoE + Decoupled OpenDiLoCo manifest, and Raid Kit docs.
2. Emitting 1-click pre-filled GitHub web URLs for creating the PR, 4 Raid Track Issues, and GitHub Discussion.
3. Publishing via `gh` CLI (`--publish-github`) when authenticated (`gh auth status`).
4. Broadcasting signed AMCP/Kafka CloudEvents (`--publish-amcp`) to `broadcast` and `vxtzq` via
   `POST /v1/hivemind/amcp/publish` without ever logging or persisting API keys.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import urllib.parse
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hivemind.cli.campaign_calendar_quorum import (  # noqa: E402
    CROWDGPT_WEEKEND_NOV14_15_EVENTS,
    DEFAULT_CROWDGPT_WEEKEND_ICS_PATH,
)
from hivemind.swarm.crowdgpt_moe_opendiloco import (  # noqa: E402
    CROWDGPT_MOE_PUBLIC_MODEL_ID,
    CROWDGPT_WEEKEND_CAMPAIGN_ID,
    build_weekend_nov14_15_raid_manifest,
)

GITHUB_REPO = "xaviercallens/HiveMind"
DEFAULT_GATEWAY_URL = "https://hivemind-openrouter-1003063861791.us-central1.run.app"
RAID_KIT_DOC_PATH = REPO_ROOT / "docs" / "CROWDGPT_WE_NOV14_15_2026_MASSIVE_TRAINING_KIT.md"
ISSUE_TEMPLATE_PATH = REPO_ROOT / ".github" / "ISSUE_TEMPLATE" / "04_crowdgpt_weekend_moe_raid.yml"

WEEKEND_RAID_ISSUES: tuple[dict[str, Any], ...] = (
    {
        "track": "Track 1",
        "title": "[Weekend Raid · Track 1] CrowdGPT 8-Expert Sparse MoE Split & FoMoE sigma-Router Calibration (14–15 Nov 2026)",
        "labels": ["crowdgpt-weekend-raid", "moe-opendiloco", "domain-expert"],
        "body": (
            "## Track 1 — Function-Preserving 8-Expert Sparse MoE Split (`hivemind/crowd-moe-v1`)\n\n"
            "- **Campaign**: `CROWDGPT-WE-NOV14-15-2026-MOE-OPENDILOCO`\n"
            "- **Code**: `src/hivemind/swarm/crowdgpt_moe_opendiloco.py`\n"
            "- **Calendar**: `docs/calendar/hivemind_crowdgpt_weekend_nov14_15_2026.ics`\n\n"
            "### Deliverables\n"
            "- [x] Step-0 exact `CrowdFineGrainedMoEMLP` (`8 x 320 = 2560`) and `CrowdLoRAMoEMLP` (`8` SwiGLU LoRA experts).\n"
            "- [x] FoMoE non-competitive `sigmoid_independent_topk_routing` + Ghost-Expert Skip-Token routing for `8 GB` GPUs.\n"
            "- [ ] Execute Saturday Wave 1 (`14 Nov 09:00–13:00 UTC`) router warm-start across `E0..E7` (`CV < 0.35`).\n\n"
            "*(Non-Monetary Charter: HMT Reflexion Points carry zero financial value — reciprocal shared compute quota only.)*"
        ),
    },
    {
        "track": "Track 2",
        "title": "[Weekend Raid · Track 2] Decoupled OpenDiLoCo + RDA Quorum Syncer on GCP Spot T4 & Home Swarm",
        "labels": ["crowdgpt-weekend-raid", "moe-opendiloco", "gcp-cost-governance"],
        "body": (
            "## Track 2 — PrimeIntellect OpenDiLoCo + DeepMind Decoupled DiLoCo (`RDA`)\n\n"
            "- **Coordinator**: `CrowdOpenDiLoCoCoordinator` (`CPUOffloadedFlatBucketOuterState` `0 MiB` GPU overhead, `P=4` router + `E=8` expert buckets)\n"
            "- **T4 Spot Schedule**: `bash scripts/t4_training_window.sh weekend-raid` (`14:00–22:00 Europe/Paris` Sat 14 & Sun 15 Nov = `+$1.30` Spot, `$121.92/mo <= $150/mo`)\n"
            "- **Safety**: `>= 4,300 MiB` free VRAM floor on Tesla T4; zero writes to `/opt/hivemind/t4/`.\n\n"
            "### Deliverables\n"
            "- [x] Integrate `DecoupledDiLoCoQuorumSyncer` (`RDA`), `AdaptiveGraceWindowController`, and nocturnal `PA-CGAD`.\n"
            "- [ ] Complete Wave 2 (Sat 14 Nov `13:00–21:00 UTC`) & Wave 4 (Sun 15 Nov `13:00–18:00 UTC`) outer checkpoints with monotone validation-loss ratchet.\n"
        ),
    },
    {
        "track": "Track 3",
        "title": "[Weekend Raid · Track 3] Producer, Trainer & Prosumer Onboarding + 18 License-Clean S21 Shards",
        "labels": ["crowdgpt-weekend-raid", "community", "datasets"],
        "body": (
            "## Track 3 — HiveMinder Producers, Trainers & Prosumers + S21 Dataset Shards\n\n"
            "- **Dataset Registry**: `configs/datasets_s21.json` (`18` `ACCEPTED` AGPL-3.0 allowlisted datasets; `cc-by-nc-*` rejected; `13`-gram decontaminated).\n"
            "- **Live Endpoint**: `GET /v1/hivemind/opentrainer/crowdgpt-weekend-raid?handle=<handle>&role=prosumer&vram_mib=8192`\n\n"
            "### Deliverables\n"
            "- [x] Deterministic non-IID shard assignment across `E0..E7` and `8 GB / 12 GB / 16 GB / 24 GB` VRAM tiers.\n"
            "- [ ] Onboard `>= 25` active weekend Producer, Trainer, and Prosumer GPUs via Discord (`https://discord.gg/ZeFtZpKHf`) & GitHub.\n"
        ),
    },
    {
        "track": "Track 4",
        "title": "[Weekend Raid · Track 4] Sunday Wave-5 Blind A/B Arena & Vxtzq/Crowd-MoE-v1 Checkpoint Export",
        "labels": ["crowdgpt-weekend-raid", "evaluation", "release"],
        "body": (
            "## Track 4 — Sunday Blind A/B Arena (`hivemind/crowd-v1` vs `hivemind/crowd-moe-v1`)\n\n"
            "- **Window**: Sunday **15 Nov 2026 `18:00–21:00 UTC`** (`19:00–22:00 CET`)\n"
            "- **Partner Lead**: `@Vxtzq` (`105,000 HMT` testing balance active on HiveMind OpenRouter)\n\n"
            "### Deliverables\n"
            "- [ ] Run live side-by-side bilingual FR/EN + Python + GSM8K + ARC evaluation in the Blind A/B Arena.\n"
            "- [ ] Export signed `safetensors` SHA-256 checkpoint & `LossCheckReceipt` for Hugging Face / OpenRouter release.\n"
        ),
    },
)

DISCUSSION_TITLE = (
    "[Weekend Raid · 14–15 Nov 2026] Massive CrowdGPT-MoE x HiveMind Decoupled OpenDiLoCo Training "
    "— Calling All Producers, Trainers & Prosumers!"
)


def build_prefilled_github_urls(branch: str = "feat/crowdgpt-we-nov14-15-moe-opendiloco") -> dict[str, Any]:
    """Build 1-click pre-filled GitHub URLs for PR, 4 Track Issues, and Discussion."""
    pr_title = "feat(crowdgpt-moe): Weekend Nov 14–15 Massive Training Calendar, 8x MoE Split & Decoupled OpenDiLoCo"
    pr_url = (
        f"https://github.com/{GITHUB_REPO}/compare/main...{urllib.parse.quote(branch)}"
        f"?expand=1&title={urllib.parse.quote(pr_title)}"
    )
    issue_urls = []
    for item in WEEKEND_RAID_ISSUES:
        q = urllib.parse.urlencode(
            {
                "title": item["title"],
                "labels": ",".join(item["labels"]),
                "body": item["body"],
            }
        )
        issue_urls.append(
            {
                "track": item["track"],
                "title": item["title"],
                "url": f"https://github.com/{GITHUB_REPO}/issues/new?{q}",
            }
        )
    disc_q = urllib.parse.urlencode({"category": "announcements", "title": DISCUSSION_TITLE})
    return {
        "existing_pr_url": f"https://github.com/{GITHUB_REPO}/pull/871",
        "standalone_raid_pr_url": pr_url,
        "raid_registration_template_url": (
            f"https://github.com/{GITHUB_REPO}/issues/new?template=04_crowdgpt_weekend_moe_raid.yml"
        ),
        "track_issue_urls": issue_urls,
        "discussion_new_url": f"https://github.com/{GITHUB_REPO}/discussions/new?{disc_q}",
    }


def publish_amcp_raid_announcements(gateway_url: str, api_key: str) -> list[dict[str, Any]]:
    """Publish signed AMCP/Kafka weekend raid announcements to `broadcast` and `vxtzq`."""
    messages = [
        {
            "recipient_id": "broadcast",
            "topic_key": "human_agent",
            "event_type": "io.hivemind.amcp.dev.proposal",
            "sender_kind": "gpu_producer",
            "payload": {
                "title": "Weekend Raid (14-15 Nov 2026): CrowdGPT-MoE x HiveMind Decoupled OpenDiLoCo",
                "body": (
                    f"[{CROWDGPT_WEEKEND_CAMPAIGN_ID}] Massive 48h Weekend Swarm Training Raid across all Producers, "
                    f"Trainers & Prosumers (8 GB-24 GB GPUs) + GCP Spot T4 to train {CROWDGPT_MOE_PUBLIC_MODEL_ID}! "
                    "6 global quorum waves (Fri 13 Nov 18:00 UTC Preflight -> Sun 15 Nov 21:00 UTC Blind A/B Arena). "
                    "Step-0 exact 8-expert Sparse MoE split + FoMoE Ghost-Expert routing (<1.8 GiB VRAM on 8 GB GPUs) "
                    "+ PrimeIntellect OpenDiLoCo + DeepMind Decoupled DiLoCo (RDA) on 18 license-clean S21 datasets. "
                    "Non-monetary reciprocal HMT compute credits only."
                ),
                "area": "model",
                "refs": [
                    "https://github.com/xaviercallens/HiveMind/pull/871",
                    "https://discord.gg/ZeFtZpKHf",
                ],
            },
        },
        {
            "recipient_id": "vxtzq",
            "topic_key": "human_agent",
            "event_type": "io.hivemind.amcp.message",
            "sender_kind": "gpu_producer",
            "payload": {
                "text": (
                    "Salut @Vxtzq! Everything is ready for this weekend's Massive CrowdGPT-MoE x HiveMind Raid "
                    "(Sat 14 & Sun 15 Nov 2026): step-0 exact 8-expert Sparse MoE split of Vxtzq/Crowd-v1 "
                    "(CrowdFineGrainedMoEMLP + CrowdLoRAMoEMLP), Decoupled OpenDiLoCo + RDA Quorum on GCP Spot T4 "
                    "& home GPUs, .ics calendar, Discord briefing, and our joint Reddit collab draft in PR #871!"
                ),
            },
        },
    ]
    receipts: list[dict[str, Any]] = []
    endpoint = gateway_url.rstrip("/") + "/v1/hivemind/amcp/publish"
    for msg in messages:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(msg).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "hivemind-crowdgpt-weekend-raid/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        receipts.append(
            {
                "recipient_id": msg["recipient_id"],
                "event_id": body.get("event_id") or body.get("id"),
                "bus": body.get("bus"),
                "topic": body.get("topic"),
            }
        )
    return receipts


def try_publish_github() -> dict[str, Any]:
    """Create GitHub issues via `gh` CLI if authenticated; otherwise return clean instructions."""
    chk = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True, check=False)
    if chk.returncode != 0:
        return {
            "published": False,
            "reason": "gh_cli_not_authenticated",
            "hint": "Run `gh auth login` then re-run `--publish-github`, or click the pre-filled URLs from `--print-urls`.",
        }
    created_issues: list[str] = []
    for item in WEEKEND_RAID_ISSUES:
        cmd = [
            "gh",
            "issue",
            "create",
            "--repo",
            GITHUB_REPO,
            "--title",
            item["title"],
            "--body",
            item["body"],
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0:
            created_issues.append(res.stdout.strip())
    return {"published": True, "created_issues": created_issues}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate & publish the Nov 14–15, 2026 CrowdGPT-MoE Weekend Raid Kit")
    ap.add_argument("--dry-run", action="store_true", help="Validate all raid artifacts without network writes")
    ap.add_argument("--print-urls", action="store_true", help="Include 1-click pre-filled GitHub URLs")
    ap.add_argument("--publish-github", action="store_true", help="Create GitHub Issues via gh CLI if logged in")
    ap.add_argument("--publish-amcp", action="store_true", help="Broadcast signed AMCP/Kafka raid notices")
    ap.add_argument("--gateway", default=DEFAULT_GATEWAY_URL)
    ap.add_argument("--api-key-file", type=Path, default=None, help="Path to file containing HIVEMIND_API_KEY")
    ap.add_argument("--branch", default="feat/crowdgpt-we-nov14-15-moe-opendiloco")
    args = ap.parse_args(argv)

    if not DEFAULT_CROWDGPT_WEEKEND_ICS_PATH.exists():
        raise FileNotFoundError(f"Missing weekend ICS calendar: {DEFAULT_CROWDGPT_WEEKEND_ICS_PATH}")
    if not RAID_KIT_DOC_PATH.exists():
        raise FileNotFoundError(f"Missing weekend raid kit doc: {RAID_KIT_DOC_PATH}")
    if not ISSUE_TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Missing GitHub issue template: {ISSUE_TEMPLATE_PATH}")

    manifest = build_weekend_nov14_15_raid_manifest()
    report: dict[str, Any] = {
        "status": "ok",
        "campaign_id": CROWDGPT_WEEKEND_CAMPAIGN_ID,
        "ics_path": str(DEFAULT_CROWDGPT_WEEKEND_ICS_PATH.relative_to(REPO_ROOT)),
        "ics_vevents": len(CROWDGPT_WEEKEND_NOV14_15_EVENTS),
        "raid_kit_doc": str(RAID_KIT_DOC_PATH.relative_to(REPO_ROOT)),
        "issue_template": str(ISSUE_TEMPLATE_PATH.relative_to(REPO_ROOT)),
        "num_moe_experts": manifest["dataset_plan"]["num_experts"],
        "rejected_s21_datasets": len(manifest["dataset_plan"]["rejected_non_allowlisted_datasets"]),
        "weekend_spot_budget_usd": manifest["gcp_spot_t4_budget_governance"]["projected_total_monthly_usd"],
        "within_150_usd_ceiling": manifest["gcp_spot_t4_budget_governance"]["within_budget"],
    }

    if args.print_urls or args.dry_run:
        report["github_urls"] = build_prefilled_github_urls(branch=args.branch)

    if args.publish_github and not args.dry_run:
        report["github_publish"] = try_publish_github()

    if args.publish_amcp and not args.dry_run:
        key = os.environ.get("HIVEMIND_API_KEY", "").strip()
        if not key and args.api_key_file is not None and args.api_key_file.exists():
            key = args.api_key_file.read_text(encoding="utf-8").strip()
        if not key:
            raise RuntimeError("Set HIVEMIND_API_KEY or pass --api-key-file for --publish-amcp")
        report["amcp_receipts"] = publish_amcp_raid_announcements(args.gateway, key)

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
