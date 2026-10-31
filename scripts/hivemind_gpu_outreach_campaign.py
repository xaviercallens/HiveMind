#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Generate personalized bilingual (FR/EN) GPU Reconnection & Onboarding Campaign emails/messages
for registered HiveMinder prosumer & producer accounts from /data/router.db or the verified cohort list.

Usage:
  python scripts/hivemind_gpu_outreach_campaign.py --format markdown
  python scripts/hivemind_gpu_outreach_campaign.py --db /path/to/router.db --format json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_GATEWAY_URL = "https://hivemind-openrouter-1003063861791.us-central1.run.app"
DEFAULT_RUCHE_URL = "https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app"


@dataclass(frozen=True)
class OutreachRecipient:
    account_id: str
    handle: str
    email: str | None
    role: str
    os_platform: str
    known_gpu: str
    recommended_model_spec: str
    previous_tunnel_note: str


# Verified external & registered prosumer/producer cohort from /data/router.db forensic audit
VERIFIED_COHORT: tuple[OutreachRecipient, ...] = (
    OutreachRecipient(
        account_id="4203a7421cb018af01507cd08d101465",
        handle="tibrib",
        email=None,
        role="prosumer",
        os_platform="windows",
        known_gpu="NVIDIA GeForce RTX 4070 (12,282 MiB VRAM)",
        recommended_model_spec="coder=qwen2.5-coder:7b",
        previous_tunnel_note=(
            "Previously connected from Windows 10 x64 via Cloudflare tunnels "
            "(links-winter-swing-pubs.trycloudflare.com & robin-kernel-keyword-shaved.trycloudflare.com) "
            "serving qwen2.5-coder:7b in 4,535 MiB VRAM."
        ),
    ),
    OutreachRecipient(
        account_id="60a79a85fe77942ecbe36892d8e3ee6b",
        handle="ilenemaurin@gmail.com",
        email="ilenemaurin@gmail.com",
        role="prosumer",
        os_platform="windows",
        known_gpu="Consumer GPU (>= 6 GB VRAM) or CPU/BOINC Worker",
        recommended_model_spec="coder=qwen2.5-coder:7b",
        previous_tunnel_note="Registered prosumer account with 200 HMT balance ready for 1-click GPU producer onboarding.",
    ),
    OutreachRecipient(
        account_id="a35cdd56e5892104ea8a95eb35853895",
        handle="vxtzq",
        email="vxtzq@github.com",
        role="prosumer",
        os_platform="linux",
        known_gpu="CrowdGPT Partner GPU Node (Vxtzq/Crowd-v1)",
        recommended_model_spec="crowd-v1=qwen2.5-coder:7b-instruct",
        previous_tunnel_note="Registered CrowdGPT partner prosumer account for federated hivemind/crowd-v1 routing.",
    ),
)


def load_recipients_from_db(db_path: Path) -> list[OutreachRecipient]:
    """Load real user prosumer/producer accounts from a router.db SQLite file, merging with VERIFIED_COHORT metadata."""
    by_handle = {r.handle.lower(): r for r in VERIFIED_COHORT}
    if not db_path.exists():
        return list(VERIFIED_COHORT)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(accounts)").fetchall()}
        email_col = "email" if "email" in cols else "NULL AS email"
        rows = conn.execute(
            f"SELECT account_id, handle, role, {email_col} FROM accounts "
            "WHERE role IN ('producer', 'prosumer') AND kind = 'user' AND account_id NOT LIKE 'sys:%' "
            "ORDER BY created_utc ASC"
        ).fetchall()
    finally:
        conn.close()

    recipients: list[OutreachRecipient] = []
    seen_handles: set[str] = set()
    for row in rows:
        handle = str(row["handle"])
        h_low = handle.lower()
        seen_handles.add(h_low)
        if h_low in by_handle:
            recipients.append(by_handle[h_low])
            continue
        raw_email = str(row["email"] or "").strip() or (handle if "@" in handle else None)
        recipients.append(
            OutreachRecipient(
                account_id=str(row["account_id"]),
                handle=handle,
                email=raw_email,
                role=str(row["role"]),
                os_platform="linux",
                known_gpu="NVIDIA / Apple Silicon GPU (>= 6 GB VRAM)",
                recommended_model_spec="coder=qwen2.5-coder:7b",
                previous_tunnel_note="Registered HiveMinder prosumer account.",
            )
        )
    for preset in VERIFIED_COHORT:
        if preset.handle.lower() not in seen_handles:
            recipients.append(preset)
    return recipients


def render_bilingual_invitation(
    recipient: OutreachRecipient,
    *,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    ruche_url: str = DEFAULT_RUCHE_URL,
) -> dict[str, Any]:
    """Build a personalized FR + EN email/message payload with auto-start service & Night Training instructions."""
    autostart_cmd = (
        f"python scripts/hivemind_producer_autostart.py {recipient.os_platform} "
        f"--model {recipient.recommended_model_spec} --apply ./producer-autostart"
    )
    subject_fr = f"[HiveMind] Reconnectez votre GPU ({recipient.known_gpu}) en service automatique + Night Training 22h-07h"
    subject_en = f"[HiveMind] Keep your GPU ({recipient.known_gpu}) connected automatically + Night Training 22:00-07:00 CET"

    body_fr = f"""Bonjour {recipient.handle},

Merci de faire partie des pionniers de la ruche **Project HiveMind** (compte `{recipient.account_id[:12]}…`).
Contexte matériel : {recipient.previous_tunnel_note}

### 1. Nouveau : Service Auto-Start (votre GPU se reconnecte tout seul au redémarrage)
Pour éviter que votre tunnel Cloudflare ou votre agent ne s'arrête lorsque vous fermez votre terminal ou redémarrez votre PC, nous avons ajouté un installateur de service automatique (`systemd` sous Linux, tâche planifiée sous Windows 10/11, `launchd` sous macOS) :

```bash
{autostart_cmd}
```

- **Windows 10/11 (`tibrib`, etc.)** : exécutez ensuite `.\\producer-autostart\\install-producer-service.ps1` dans PowerShell. Votre `hivemind_producer_shield.py`, votre tunnel `cloudflared` et votre `hivemind_producer_agent.py` démarreront automatiquement en arrière-plan à chaque ouverture de session.
- **Linux (`systemd --user`)** : copiez `hivemind-producer.service` dans `~/.config/systemd/user/` puis lancez `systemctl --user enable --now hivemind-producer.service`.

### 2. Protection Gamer & Night Training (22h00 – 07h00 CET)
- **Priorité 100% à vos usages & jeux (< 1,5 s)** : configurez votre profil BOINC avec :
  `python scripts/hivemind_volunteer_config.py night --apply <BOINC_DATA_DIR>`
- **Entraînement Nocturne (22h00 – 07h00 Europe/Paris)** : pendant la nuit ou lorsque votre PC est inactif, votre GPU participe à l'entraînement QLoRA MoE distribué (`{ruche_url}/`) et gagne **+250 HMT** + **45 Cobblestones** par workunit vérifiée, en plus des HMT gagnés sur chaque requête OpenRouter (`{gateway_url}/join?lang=fr#share`).

À très vite sur le essaim,
Xavier Callens — Project HiveMind (`{gateway_url}/join?lang=fr`)
"""

    body_en = f"""Hi {recipient.handle},

Thank you for being an early contributor to **Project HiveMind** (account `{recipient.account_id[:12]}…`).
Hardware note: {recipient.previous_tunnel_note}

### 1. New: Auto-Start Background Service (Stay Connected Across Reboots)
So your Cloudflare tunnel and producer agent never drop when you close a terminal window or reboot your machine, you can now generate an auto-start background service for Windows 10/11, Linux (`systemd --user`), or macOS (`launchd`):

```bash
{autostart_cmd}
```

- **Windows 10/11**: run `.\\producer-autostart\\install-producer-service.ps1` in PowerShell to register `HiveMindProducerAgent` at logon with automatic tunnel recovery.
- **Linux**: copy `hivemind-producer.service` to `~/.config/systemd/user/` and run `systemctl --user enable --now hivemind-producer.service`.

### 2. Gamer Guard & Night Training (22:00 – 07:00 CET)
- **Zero game lag (< 1.5 s preemption)**: apply the stock BOINC night profile:
  `python scripts/hivemind_volunteer_config.py night --apply <BOINC_DATA_DIR>`
- **Overnight QLoRA MoE Training (22:00 – 07:00 Europe/Paris)**: while you sleep, your idle GPU folds scientific workunits and rank-4 QLoRA adapters on `{ruche_url}/`, earning **+250 HMT** + **45 Cobblestones** per verified workunit alongside live OpenRouter token settlements (`{gateway_url}/join?lang=en#share`).

See you on the swarm,
Xavier Callens — Project HiveMind (`{gateway_url}/join?lang=en`)
"""

    return {
        "recipient": asdict(recipient),
        "autostart_command": autostart_cmd,
        "subject_fr": subject_fr,
        "subject_en": subject_en,
        "body_fr": body_fr,
        "body_en": body_en,
    }


def build_campaign_report(
    db_path: Path | None = None,
    *,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    ruche_url: str = DEFAULT_RUCHE_URL,
) -> dict[str, Any]:
    recipients = load_recipients_from_db(db_path) if db_path is not None else list(VERIFIED_COHORT)
    invitations = [
        render_bilingual_invitation(r, gateway_url=gateway_url, ruche_url=ruche_url)
        for r in recipients
    ]
    return {
        "campaign_id": "HIVEMIND-GPU-RECONNECT-2026-11",
        "gateway_url": gateway_url,
        "ruche_url": ruche_url,
        "recipients_count": len(invitations),
        "invitations": invitations,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", help="optional path to router.db SQLite file")
    ap.add_argument("--format", choices=("markdown", "json"), default="markdown", help="output format")
    args = ap.parse_args(argv)

    report = build_campaign_report(Path(args.db).expanduser() if args.db else None)
    if args.format == "json":
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    print(f"# HiveMind End-User GPU Reconnection & Auto-Start Campaign ({report['recipients_count']} Recipients)\n")
    for idx, inv in enumerate(report["invitations"], start=1):
        rec = inv["recipient"]
        email_tag = f" (`{rec['email']}`)" if rec.get("email") else " (handle / forum DM)"
        print(f"## {idx}. `{rec['handle']}`{email_tag} — {rec['known_gpu']} [{rec['os_platform']}]")
        print(f"- **Auto-Start Command:** `{inv['autostart_command']}`")
        print(f"- **FR Subject:** {inv['subject_fr']}")
        print(f"- **EN Subject:** {inv['subject_en']}\n")
        print(inv["body_fr"])
        print("---\n")
        print(inv["body_en"])
        print("\n" + "=" * 72 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
