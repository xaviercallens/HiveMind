# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HiveMind v3.4.0 Multi-Channel Pioneer Thank-You, Reconnection & CrowdGPT Weekend Swarm Raid Outreach.

Prepares and dispatches coordinated community communications from ``callensxavier@gmail.com`` across:
1. **Email (Gmail CLI / BCC / SMTP)**: Inline-CSS HTML + Markdown + ``.ics`` calendar invite, strictly using ``BCC``
   for community recipients so no contributor's e-mail address is ever exposed to other recipients.
2. **GitHub (`xaviercallens/HiveMind` & `Vxtzq/CrowdGPT`)**: Pioneer contributor thank-you Discussion, Community Raid
   Issue, and `v3.4.0` Release announcement with 1-click pre-filled GitHub URLs.
3. **Discord (`https://discord.gg/ZeFtZpKHf`)**: Bilingual (EN/FR) `#announcements` & `#weekend-swarm-raid` post plus
   optional `DISCORD_WEBHOOK_URL` rich embed delivery.
4. **Reddit (`r/LocalLLaMA`, `r/BOINC`, `r/MachineLearning`, `r/homelab`)**: Co-authored `HiveMind v3.4 x CrowdGPT`
   collaboration posts & 1-click submit URLs.
5. **AMCP / Kafka Live Mesh (`hivemind.amcp.human_agent.v1`)**: Global broadcast + personalized direct messages to
   all 16 early human pioneer handles registered on the swarm.

Core Themes Covered Across Every Channel:
- Heartfelt thank-you to the first open-source contributors and pioneer GPU/end-user adopters.
- Opportunity to reconnect existing accounts (`hm-...` keys & `HMT` balances 100% preserved) on the new `v3.3/v3.4`
  bilingual (EN/FR) Web UI with real-time account ledger & peer-to-peer `HMT` transfers.
- Open-source collaboration proposal with CrowdGPT (`@Vxtzq`) uniting decentralized MoE serving/routing with
  from-scratch community pretraining (`Vxtzq/Crowd-v1` -> `hivemind/crowd-moe-v1`).
- Call to join as **GPU Producers, Trainers, or Prosumers** for the **Massive Weekend Swarm Raid (14–15 Nov 2026)** to
  jointly validate **both** open-source concepts at real community scale.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Sequence
import urllib.parse
import urllib.request

OWNER_SENDER_NAME: str = "Xavier Callens"
OWNER_SENDER_EMAIL: str = "callensxavier@gmail.com"
OWNER_FROM_HEADER: str = f"{OWNER_SENDER_NAME} <{OWNER_SENDER_EMAIL}>"

PUBLIC_OPENROUTER_URL: str = "https://hivemind-openrouter-1003063861791.us-central1.run.app"
PUBLIC_WEB_UI_URL: str = "https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app"
GITHUB_REPO: str = "xaviercallens/HiveMind"
GITHUB_REPO_URL: str = f"https://github.com/{GITHUB_REPO}"
CROWDGPT_REPO_URL: str = "https://github.com/Vxtzq/CrowdGPT"
CROWDGPT_HF_URL: str = "https://huggingface.co/Vxtzq/Crowd-v1"
DISCORD_INVITE_URL: str = "https://discord.gg/ZeFtZpKHf"
RELEASE_V340_URL: str = f"{GITHUB_REPO_URL}/releases/tag/v3.4.0"
PR_871_URL: str = f"{GITHUB_REPO_URL}/pull/871"
WEEKEND_RAID_DATES_LABEL: str = "Saturday 14 & Sunday 15 November 2026 (Preflight Friday 13 Nov 18:00 UTC / 19:00 CET)"

# Verified early Git contributors (`git shortlog -sne`) + pioneer email handles on HiveMind
VERIFIED_PIONEER_EMAILS: tuple[str, ...] = (
    "clark719664@gmail.com",
    "prmineo@gmail.com",
    "radial@gmail.com",
    "ilenemaurin@gmail.com",
)

# All 16 verified human pioneer & early-adopter handles registered on HiveMind
PIONEER_COMMUNITY_HANDLES: tuple[str, ...] = (
    "vxtzq",
    "clark719664",
    "pedromineo",
    "radialmonster",
    "tibrib",
    "tibrib-chat",
    "ilenemaurin@gmail.com",
    "jblam62122",
    "dr_elena_astro",
    "epic_unreal_dev",
    "reddit_llama_king",
    "steam_rtx4090_pro",
    "minder_kipping_rtx4080",
    "cybervalkyrie-fr-5ecb",
    "cmdr_rtx2070_steam",
    "audit_webgpu_pilot",
)

GITHUB_PIONEER_MENTIONS: tuple[str, ...] = (
    "@clark719664",
    "@pedromineo",
    "@radialmonster",
    "@Vxtzq",
)


def _validate_emails(emails: Sequence[str]) -> tuple[str, ...]:
    if isinstance(emails, (str, bytes)) or not isinstance(emails, Sequence):
        raise TypeError("emails must be a sequence of email strings")
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in emails:
        if not isinstance(raw, str):
            raise TypeError("every email address must be a string")
        addr = raw.strip()
        if not addr or "@" not in addr or any(c in addr for c in ("\r", "\n", "\x00", " ", ",")):
            raise ValueError(f"invalid email address: {raw!r}")
        low = addr.lower()
        if low != OWNER_SENDER_EMAIL.lower() and low not in seen:
            seen.add(low)
            cleaned.append(addr)
    return tuple(cleaned)


def build_v34_pioneer_email(
    *,
    sender_email: str = OWNER_SENDER_EMAIL,
    bcc_recipients: Sequence[str] = VERIFIED_PIONEER_EMAILS,
) -> dict[str, Any]:
    """Build the `v3.4.0` Pioneer Thank-You, Web UI Reconnect & Nov 14–15 CrowdGPT Raid email package.

    Enforces `To: callensxavier@gmail.com` and `BCC`-only community recipients so contributor addresses
    remain completely private. Uses inline CSS on all HTML tables and callouts for clean Gmail rendering.
    """
    if not isinstance(sender_email, str) or sender_email.strip().lower() != OWNER_SENDER_EMAIL:
        raise ValueError(f"sender_email must be {OWNER_SENDER_EMAIL!r}")
    bcc = _validate_emails(bcc_recipients)

    subject = (
        "Thank You HiveMind Pioneers! HiveMind v3.4 + New Web UI + CrowdGPT Massive Swarm Raid (14–15 Nov 2026)"
    )

    markdown_body = f"""# Thank You for Being Among the First HiveMind Pioneers & Open-Source Contributors!

Hi everyone,

I am writing personally from **{OWNER_SENDER_EMAIL}** to say a sincere **thank you** for being the very first open-source contributors, GPU producers, trainers, and early adopters of **Project HiveMind** (`SETI@home` & `BOINC` reimagined for open-source AI).

Thanks to your early commits, bug reports, GPU nodes, and live inference tests, our first 30 days crossed **55 registered HiveMinder nodes**, **3,617 settled requests**, and **363,655 tokens** across community GPUs!

---

## 1. What's New in HiveMind `v3.4.0` & Opportunity to Reconnect
- **Brand-New Hardened Bilingual Web UI (EN / FR)**: Redesigned account & live usage panel, OWASP 600k-iteration PBKDF2 security, live monthly/weekly token telemetry (`/telemetry`), and **500 HMT** welcome & pioneer bonuses.
- **Peer-to-Peer HiveMind Point (`HMT`) Transfers**: HiveMinders can now gift reciprocal `HMT` compute credits directly to fellow builders and model researchers (`POST /v1/hivemind/accounts/transfer`).
- **Seamless Reconnection (100% Preserved Balances & Keys)**: Your existing `hm-...` API key and `HMT` balance are fully preserved. Reconnecting your `8 GB–24 GB` GPU takes **1 command** via the hardened Producer Shield (`:8787` + `< 1.5s` automatic game preemption):
  - **Portal & Web UI**: {PUBLIC_OPENROUTER_URL}
  - **Release `v3.4.0` Notes**: {RELEASE_V340_URL}

---

## 2. Open-Source Collaboration Proposal: **HiveMind x CrowdGPT (`@Vxtzq`)**
We are teaming up with **`@Vxtzq` (CrowdGPT)** to bridge **two complementary open-source citizen-AI concepts**:
1. **HiveMind's Decentralized MoE Swarm & Reciprocal Compute Router**: Upcycling dense `Vxtzq/Crowd-v1` (`1.01B` + `CrowdEngram`) into **`hivemind/crowd-moe-v1`** (**1 Shared Anchor + 8 Routed Domain Experts**, `< 1.8 GiB` LoRA-MoE VRAM on `8 GB` GPUs) with OpenAI-compatible community serving.
2. **Decoupled OpenDiLoCo + Cryptographic Loss-Ratchet Co-Training**: Asynchronous inner AdamW steps across home GPUs + GCP Spot T4, aggregated via CPU-offloaded FP32 Nesterov outer momentum (`RDA` staleness damping + `PA-CGAD` cosine conflict gating + CPU-verifiable held-out `LossCheckReceipt`). In our 2-round CUDA preflight on `Crowd-v1` (`59c016e94875`), held-out cross-entropy dropped **`5.3042 -> 5.1742 -> 0.9668`**!

---

## 3. Join the **Massive Weekend Swarm Raid — 14 & 15 November 2026**!
This weekend (**{WEEKEND_RAID_DATES_LABEL}**), we are running a coordinated **48-hour Swarm Raid** to **validate BOTH new concepts live together**:
- **Join as a GPU Producer (`>= 6–8 GB VRAM`)**: Serve live `hivemind/crowd-v1`, `coder-lite`, or `granite-moe` completions and earn `85%` reciprocal `HMT` credits while keeping `< 1.5s` game preemption.
- **Join as a Decentralised Trainer (`8 GB–24 GB VRAM`)**: Claim one of the `8` Sparse-MoE domain experts (`E0..E7`) on `18` license-clean S21 datasets and submit OpenDiLoCo pseudo-gradients every outer round.
- **Join as a Prosumer / Evaluator**: Run blind A/B prompts on the new Web UI and verify held-out loss receipts!

### 6-Wave Global Raid Schedule (`.ics` Calendar Attached)
- **Wave 0 — Launch Review & Preflight**: Fri 13 Nov, `18:00–20:00 UTC` (`19:00–21:00 CET`)
- **Wave 1 — Step-0 8x MoE Split & Router Warm-Start**: Sat 14 Nov, `09:00–13:00 UTC`
- **Wave 2 — Trans-Atlantic Peak OpenDiLoCo (`E0..E7`)**: Sat 14 Nov, `13:00–21:00 UTC`
- **Wave 3 — Overnight Autonomous `PA-CGAD` Window**: Sat 14 Nov `21:00 UTC` -> Sun 15 Nov `08:00 UTC`
- **Wave 4 — Sunday Convergence & Producer Stress-Test**: Sun 15 Nov, `13:00–18:00 UTC`
- **Wave 5 — Blind A/B Arena & `Vxtzq/Crowd-MoE-v1` Promotion**: Sun 15 Nov, `18:00–20:00 UTC`

### Quick Links to Reconnect & Register Your Node
- **GitHub Raid Kit & PR #871**: {PR_871_URL}
- **1-Click Raid Node Registration Issue**: {GITHUB_REPO_URL}/issues/new?template=04_crowdgpt_weekend_moe_raid.yml
- **Discord Swarm Coordination (`#weekend-swarm-raid`)**: {DISCORD_INVITE_URL}
- **Live Gateway & Web UI**: {PUBLIC_OPENROUTER_URL}

*(Non-Monetary Open-Source Charter: `HMT` Reflexion Points carry zero financial value and represent reciprocal shared community compute quota only under AGPL-3.0.)*

---

### En Français (Résumé Rapide)
Un immense **merci** à nos premiers contributeurs open-source et pionniers GPU ! **HiveMind `v3.4.0`** est disponible avec une **nouvelle interface Web bilingue FR/EN**, le transfert pair-à-pair de points `HMT`, la reconnexion en 1 commande (vos clés `hm-...` et soldes `HMT` sont 100% conservés), et une **proposition de collaboration open-source avec CrowdGPT (`@Vxtzq`)**. Rejoignez-nous comme **Producteur GPU, Entraîneur OpenDiLoCo ou Prosumer** lors du ** Raid Swarm Massif ce week-end des 14 et 15 novembre 2026** pour valider ensemble ces **deux nouveaux concepts** à grande échelle !

Warm regards / À très vite sur La Ruche,  
**Xavier Callens**  
Founder & Maintainer, Project HiveMind  
{OWNER_SENDER_EMAIL} • {GITHUB_REPO_URL}
"""

    html_body = _build_inline_css_html_email(
        sender_email=OWNER_SENDER_EMAIL,
        subject=subject,
    )

    gmail_compose_params = urllib.parse.urlencode(
        {
            "authuser": OWNER_SENDER_EMAIL,
            "view": "cm",
            "fs": "1",
            "to": OWNER_SENDER_EMAIL,
            "bcc": ",".join(bcc),
            "su": subject,
            "body": markdown_body,
        }
    )
    gmail_web_compose_url = f"https://mail.google.com/mail/?{gmail_compose_params}"

    return {
        "from_header": OWNER_FROM_HEADER,
        "from_email": OWNER_SENDER_EMAIL,
        "to_email": OWNER_SENDER_EMAIL,
        "bcc_emails": list(bcc),
        "subject": subject,
        "markdown_body": markdown_body,
        "plain_text_body": markdown_body,
        "html_body": html_body,
        "gmail_web_compose_url": gmail_web_compose_url,
    }


def _build_inline_css_html_email(*, sender_email: str, subject: str) -> str:
    safe_sender = html.escape(sender_email)
    safe_subj = html.escape(subject)
    th_style = (
        "border:1px solid #334155;padding:10px 12px;background-color:#0f172a;"
        "color:#f8fafc;text-align:left;font-size:13px;"
    )
    td_style = "border:1px solid #cbd5e1;padding:9px 12px;color:#1e293b;font-size:13px;"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{safe_subj}</title>
</head>
<body style="margin:0;padding:24px;background-color:#f1f5f9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#0f172a;line-height:1.6;">
  <div style="max-width:680px;margin:0 auto;background-color:#ffffff;border:1px solid #e2e8f0;border-radius:12px;overflow:hidden;">
    <div style="background-color:#0f172a;color:#f8fafc;padding:24px 28px;border-bottom:4px solid #f59e0b;">
      <div style="font-size:12px;font-weight:700;letter-spacing:0.08em;color:#fbbf24;text-transform:uppercase;">Project HiveMind v3.4.0 &times; CrowdGPT Open-Source Collab</div>
      <h1 style="margin:8px 0 0 0;font-size:22px;line-height:1.3;color:#ffffff;">Thank You Pioneers! Reconnect on the New Web UI &amp; Join This Weekend's Massive Swarm Raid (14&ndash;15 Nov 2026)</h1>
    </div>

    <div style="padding:26px 28px;">
      <p style="margin-top:0;">Hi everyone,</p>
      <p>I am writing personally from <strong>{safe_sender}</strong> to thank you for being among the <strong>very first open-source contributors, GPU producers, trainers, and early adopters</strong> of <strong>Project HiveMind</strong> (<em>SETI@home &amp; BOINC reimagined for citizen open-source AI</em>).</p>
      <p>Because you stepped up early to test, commit code, and share idle GPUs, HiveMind grew to <strong>55 registered HiveMinder nodes</strong>, <strong>3,617 settled inference requests</strong>, and <strong>363,655 tokens</strong> in our very first month.</p>

      <h2 style="font-size:17px;color:#0f172a;border-bottom:2px solid #e2e8f0;padding-bottom:6px;margin-top:24px;">1. HiveMind v3.4.0: New Bilingual Web UI &amp; Instant Reconnection</h2>
      <ul style="padding-left:20px;margin:10px 0;">
        <li><strong>New Hardened Web UI (EN / FR)</strong>: Real-time Account &amp; Usage Ledger, monthly/weekly telemetry dashboard (<code>/telemetry</code>), OWASP 600k-iteration PBKDF2 authentication, and <strong>500 HMT</strong> welcome &amp; pioneer bonuses.</li>
        <li><strong>Peer-to-Peer <code>HMT</code> Transfers</strong>: Send reciprocal <code>HMT</code> compute points directly to fellow contributors and model researchers (<code>POST /v1/hivemind/accounts/transfer</code>).</li>
        <li><strong>100% Preserved Balances &amp; Keys</strong>: Your existing <code>hm-...</code> API key and <code>HMT</code> balance are intact. Reconnect any <code>8 GB&ndash;24 GB</code> GPU in 1 command with the Zero-Trust Producer Shield (<code>:8787</code> + <code>&lt; 1.5s</code> automatic gaming preemption).</li>
      </ul>

      <h2 style="font-size:17px;color:#0f172a;border-bottom:2px solid #e2e8f0;padding-bottom:6px;margin-top:24px;">2. Collaboration Proposal: HiveMind &times; CrowdGPT (<code>@Vxtzq</code>)</h2>
      <p>We are proposing an open-source collaboration with <strong><code>@Vxtzq</code> (CrowdGPT)</strong> that unites and validates <strong>both new community AI concepts together</strong>:</p>
      <ol style="padding-left:20px;margin:10px 0;">
        <li><strong>Concept A &mdash; Citizen GPU Producer Swarm &amp; Sparse-MoE Serving</strong>: Upcycling <code>Vxtzq/Crowd-v1</code> (<code>1.01B</code> + <code>CrowdEngram</code>) into <strong><code>hivemind/crowd-moe-v1</code></strong> (1 Shared Anchor + 8 Domain Specialists, <code>&lt; 1.8 GiB</code> LoRA-MoE VRAM on <code>8 GB</code> GPUs) routed live across home GPUs.</li>
        <li><strong>Concept B &mdash; Decoupled OpenDiLoCo + Verifiable Loss-Ratchet Training</strong>: Asynchronous local AdamW inner steps aggregated via CPU-offloaded FP32 Nesterov outer momentum (<code>RDA</code> staleness damping + <code>PA-CGAD</code> conflict gating + spot-checked <code>LossCheckReceipt</code>). In our 2-round CUDA preflight on <code>Crowd-v1</code>, held-out loss dropped <strong><code>5.3042 &rarr; 5.1742 &rarr; 0.9668</code></strong>!</li>
      </ol>

      <h2 style="font-size:17px;color:#0f172a;border-bottom:2px solid #e2e8f0;padding-bottom:6px;margin-top:24px;">3. Join the Massive Swarm Raid &mdash; 14 &amp; 15 November 2026</h2>
      <p>Join for 2 hours or all weekend as a <strong>GPU Producer</strong> (<code>&ge; 6&ndash;8 GB VRAM</code>), <strong>OpenDiLoCo Trainer</strong> (claim experts <code>E0..E7</code> on 18 license-clean S21 shards), or <strong>Prosumer Evaluator</strong>:</p>

      <table style="width:100%;border-collapse:collapse;margin:16px 0;">
        <thead>
          <tr>
            <th style="{th_style}">Wave</th>
            <th style="{th_style}">UTC Window</th>
            <th style="{th_style}">Focus</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style="{td_style}"><strong>Wave 0 (Preflight)</strong></td>
            <td style="{td_style}">Fri 13 Nov, 18:00&ndash;20:00 UTC</td>
            <td style="{td_style}">Launch review, Producer Shield reconnect &amp; <code>Crowd-v1</code> parity check</td>
          </tr>
          <tr>
            <td style="{td_style}"><strong>Wave 1 (MoE Split)</strong></td>
            <td style="{td_style}">Sat 14 Nov, 09:00&ndash;13:00 UTC</td>
            <td style="{td_style}">Step-0 exact 8-expert Sparse-MoE split &amp; router warm-start</td>
          </tr>
          <tr>
            <td style="{td_style}"><strong>Wave 2 (Peak Raid)</strong></td>
            <td style="{td_style}">Sat 14 Nov, 13:00&ndash;21:00 UTC</td>
            <td style="{td_style}">EU + Americas Decoupled OpenDiLoCo across <code>E0..E7</code> + Spot T4</td>
          </tr>
          <tr>
            <td style="{td_style}"><strong>Wave 3 (Nocturnal)</strong></td>
            <td style="{td_style}">Sat 21:00 &rarr; Sun 08:00 UTC</td>
            <td style="{td_style}">Overnight idle-GPU <code>PA-CGAD</code> deep training window</td>
          </tr>
          <tr>
            <td style="{td_style}"><strong>Wave 4 &amp; 5 (Arena)</strong></td>
            <td style="{td_style}">Sun 15 Nov, 13:00&ndash;20:00 UTC</td>
            <td style="{td_style}">Convergence rounds, Blind A/B Web UI Arena &amp; HF/OpenRouter promotion</td>
          </tr>
        </tbody>
      </table>

      <div style="margin:22px 0;text-align:center;">
        <a href="{GITHUB_REPO_URL}/issues/new?template=04_crowdgpt_weekend_moe_raid.yml" style="display:inline-block;background-color:#f59e0b;color:#0f172a;font-weight:700;text-decoration:none;padding:12px 20px;border-radius:8px;margin:4px;">Register Your GPU / Trainer Node</a>
        <a href="{PUBLIC_OPENROUTER_URL}" style="display:inline-block;background-color:#0f172a;color:#ffffff;font-weight:700;text-decoration:none;padding:12px 20px;border-radius:8px;margin:4px;">Open HiveMind v3.4 Web UI</a>
        <a href="{DISCORD_INVITE_URL}" style="display:inline-block;background-color:#4f46e5;color:#ffffff;font-weight:700;text-decoration:none;padding:12px 20px;border-radius:8px;margin:4px;">Join Discord Raid Room</a>
      </div>

      <div style="background-color:#f8fafc;border-left:4px solid #0f172a;padding:12px 16px;margin-top:20px;font-size:13px;color:#334155;">
        <strong>En Fran&ccedil;ais :</strong> Un immense <strong>merci</strong> &agrave; nos premiers contributeurs open-source et pionniers GPU ! <strong>HiveMind v3.4.0</strong> apporte la nouvelle interface Web bilingue FR/EN, le transfert pair-&agrave;-pair de points <code>HMT</code>, la reconnexion en 1 commande (vos cl&eacute;s <code>hm-...</code> et soldes <code>HMT</code> sont 100% conserv&eacute;s), et notre <strong>collaboration open-source avec CrowdGPT (<code>@Vxtzq</code>)</strong>. Rejoignez le <strong>Raid Swarm Massif des 14&ndash;15 novembre 2026</strong> comme Producteur GPU ou Entra&icirc;neur OpenDiLoCo pour valider ensemble ces deux nouveaux concepts ! (Calendrier <code>.ics</code> joint.)
      </div>

      <p style="font-size:12px;color:#64748b;margin-top:20px;"><em>Open-Source Non-Monetary Charter (AGPL-3.0): HMT Reflexion Points carry zero monetary value and represent reciprocal community compute credits only.</em></p>

      <p style="margin-bottom:0;">Warm regards / &Agrave; tr&egrave;s vite sur La Ruche,<br>
      <strong>Xavier Callens</strong> (<a href="mailto:{safe_sender}" style="color:#2563eb;">{safe_sender}</a>)<br>
      <a href="{RELEASE_V340_URL}" style="color:#2563eb;">HiveMind v3.4.0 Release</a> &bull; <a href="{PR_871_URL}" style="color:#2563eb;">CrowdGPT Raid PR #871</a> &bull; <a href="{DISCORD_INVITE_URL}" style="color:#2563eb;">Discord</a></p>
    </div>
  </div>
</body>
</html>
"""


def build_v34_github_outreach() -> dict[str, Any]:
    """Build GitHub Discussion, Pioneer Call Issue, and Release `v3.4.0` Markdown + 1-click URLs."""
    mentions = ", ".join(GITHUB_PIONEER_MENTIONS)
    pioneer_list = ", ".join(f"`{h}`" for h in PIONEER_COMMUNITY_HANDLES)

    discussion_title = (
        "🐝 Thank You Pioneers! HiveMind v3.4.0 + New Web UI + CrowdGPT Collab & Nov 14–15 Massive Swarm Raid"
    )
    discussion_body = f"""## Thank You to the First HiveMind Open-Source Contributors & Pioneer Adopters! 🙏

A personal note from **Xavier Callens (`{OWNER_SENDER_EMAIL}`)** to thank everyone who contributed code, opened issues, connected the first home GPUs, and tested early decentralized routing on **Project HiveMind** — including {mentions} and all **55 registered HiveMinder pioneer nodes** ({pioneer_list})!

In just 30 days, our community validated the foundations with **3,617 settled inference requests** and **363,655 tokens** across citizen GPUs.

---

### ✨ What's New in `v3.4.0` (Opportunity to Reconnect!)
1. **New Hardened Bilingual Web UI (`v3.3 / v3.4`, EN & FR)**:
   - Live Account & Usage Ledger panel, OWASP 600k-iteration PBKDF2-SHA256 sign-in, `/telemetry` weekly/monthly token dashboards, and **500 HMT** welcome & pioneer bonuses.
2. **Peer-to-Peer `HMT` Point Transfers**:
   - Transfer reciprocal `HMT` compute credits directly between HiveMinders (`POST /v1/hivemind/accounts/transfer`).
3. **1-Command Producer & Trainer Reconnection**:
   - Your existing `hm-...` API key and `HMT` balance are **100% preserved**. Reconnect via `scripts/hivemind_producer_shield.py` (`:8787`) + `scripts/hivemind_producer_agent.py` with `< 1.5s` gaming preemption.

---

### 🤝 Open-Source Collaboration Proposal: **HiveMind x CrowdGPT (`@Vxtzq`)**
We are proposing and launching a joint experiment with **`@Vxtzq` ([CrowdGPT]({CROWDGPT_REPO_URL}))** so this weekend's raid **validates BOTH open-source concepts simultaneously**:
- **Concept 1 — Decentralized Citizen-GPU Producer Swarm**: Serving `hivemind/crowd-v1` and `hivemind/crowd-moe-v1` through OpenAI-compatible weighted routing with reciprocal `HMT` accounting.
- **Concept 2 — 8x Sparse-MoE Upcycling + Decoupled OpenDiLoCo Co-Training**: Splitting `Vxtzq/Crowd-v1` (`1.01B` + `CrowdEngram`) into **1 Shared Anchor + 8 Domain Specialists (`E0..E7`)** (`< 1.8 GiB` LoRA-MoE VRAM on `8 GB` GPUs) trained asynchronously with **PrimeIntellect OpenDiLoCo + DeepMind `RDA` Quorum + CPU-verifiable `LossCheckReceipt`** (live 2-round CUDA preflight loss: **`5.3042 -> 5.1742 -> 0.9668`**).

---

### 🚀 Join the Massive Weekend Swarm Raid (`14–15 November 2026`)
- **Schedule**: 6 global waves from **Fri 13 Nov `18:00 UTC` (`19:00 CET`) Preflight** through **Sun 15 Nov `20:00 UTC` Blind A/B Arena** ([`.ics` calendar](docs/calendar/hivemind_crowdgpt_weekend_nov14_15_2026.ics)).
- **Register your GPU Producer / OpenDiLoCo Trainer node**: [Open Raid Registration Issue]({GITHUB_REPO_URL}/issues/new?template=04_crowdgpt_weekend_moe_raid.yml)
- **Live Portal**: {PUBLIC_OPENROUTER_URL} • **Discord**: {DISCORD_INVITE_URL} • **Contact**: `{OWNER_SENDER_EMAIL}`
"""

    issue_title = (
        "[Community Call] Thank You Pioneers — Reconnect on HiveMind v3.4 Web UI & Join Nov 14–15 CrowdGPT Swarm Raid"
    )
    disc_url = f"{GITHUB_REPO_URL}/discussions/new?" + urllib.parse.urlencode(
        {"category": "announcements", "title": discussion_title, "body": discussion_body}
    )
    issue_url = f"{GITHUB_REPO_URL}/issues/new?" + urllib.parse.urlencode(
        {"title": issue_title, "labels": "community,crowdgpt-weekend-raid", "body": discussion_body}
    )
    release_url = f"{GITHUB_REPO_URL}/releases/new?" + urllib.parse.urlencode(
        {
            "tag": "v3.4.0",
            "title": "HiveMind v3.4.0 — New Web UI, Pioneer Reconnect & CrowdGPT-MoE Weekend Swarm Raid",
            "body": discussion_body,
        }
    )
    return {
        "discussion_title": discussion_title,
        "discussion_body": discussion_body,
        "issue_title": issue_title,
        "issue_body": discussion_body,
        "discussion_new_url": disc_url,
        "issue_new_url": issue_url,
        "release_v340_new_url": release_url,
        "raid_template_url": f"{GITHUB_REPO_URL}/issues/new?template=04_crowdgpt_weekend_moe_raid.yml",
    }


def build_v34_discord_outreach() -> dict[str, Any]:
    """Build bilingual (EN/FR) Discord `#announcements` post and Discord Webhook JSON payload."""
    markdown_post = f"""@everyone 🐝 **THANK YOU PIONEERS — HIVEMIND `v3.4.0` + NEW WEB UI + CROWDGPT WEEKEND SWARM RAID (14–15 NOV 2026)!** 🚀

Huge personal thank-you from **Xavier (`{OWNER_SENDER_EMAIL}`)** to all our first open-source contributors, GPU producers, trainers, and early adopters (`55` registered nodes & `363,655+` tokens settled in Month 1)!

✨ **What's New & Why Reconnect Today:**
• **New Hardened EN/FR Web UI (`v3.3/v3.4`)**: Live Account & Usage panel, `/telemetry` rollups, `500 HMT` welcome/pioneer bonuses, and **Peer-to-Peer `HMT` Point Transfers** between HiveMinders!
• **1-Command Reconnect (Keys & Balances 100% Preserved)**: Bring your `8 GB–24 GB` GPU back online via `hivemind_producer_shield.py` (`:8787`, `< 1.5s` gaming preemption).

🤝 **Open-Source Collab Proposal — HiveMind x CrowdGPT (`@Vxtzq`)**:
This weekend (**Sat 14 & Sun 15 Nov 2026**, Preflight **Fri 13 Nov 18:00 UTC / 19:00 CET**), we are uniting **HiveMind** and **CrowdGPT (`Vxtzq/Crowd-v1`)** in a **48h Massive Swarm Raid** to **validate BOTH open-source concepts live**:
1️⃣ **Producer Swarm Concept**: OpenAI-compatible citizen-GPU serving + reciprocal `HMT` compute credits.
2️⃣ **Trainer Swarm Concept**: Step-0 exact **8-Expert Sparse-MoE Split (`hivemind/crowd-moe-v1`) + Decoupled OpenDiLoCo (`RDA` + Spot-Checked Loss Ratchet)** (`< 1.8 GiB` VRAM on `8 GB` GPUs; preflight CUDA loss **`5.3042 -> 5.1742 -> 0.9668`**!).

🔗 **Join as Producer, Trainer, or Prosumer:**
• **Web UI & Gateway**: <{PUBLIC_OPENROUTER_URL}>
• **Register Your Raid GPU (`E0..E7`)**: <{GITHUB_REPO_URL}/issues/new?template=04_crowdgpt_weekend_moe_raid.yml>
• **Release `v3.4.0` & Raid Kit**: <{RELEASE_V340_URL}> | <{PR_871_URL}>

🇫🇷 *Merci à tous nos pionniers ! Vos clés `hm-...` et soldes `HMT` sont conservés : reconnectez votre GPU sur la nouvelle Web UI FR/EN et rejoignez le Raid Swarm Massif CrowdGPT x HiveMind ces **14–15 novembre 2026** pour valider ensemble ces deux concepts open-source !*"""

    webhook_payload = {
        "username": "HiveMind v3.4 Swarm Coordinator",
        "content": markdown_post[:1990],
        "embeds": [
            {
                "title": "HiveMind v3.4.0 x CrowdGPT — Massive Weekend Swarm Raid (14–15 Nov 2026)",
                "url": RELEASE_V340_URL,
                "color": 16098851,
                "description": (
                    "Validate **both** citizen-GPU open-source concepts this weekend: "
                    "(1) Hardened Producer Swarm + New EN/FR Web UI + P2P `HMT` reciprocity, and "
                    "(2) 8-Expert Sparse-MoE Upcycling + Decoupled OpenDiLoCo co-training on `Vxtzq/Crowd-v1`."
                ),
                "fields": [
                    {
                        "name": "Preflight (Wave 0)",
                        "value": "Fri 13 Nov 2026 • 18:00–20:00 UTC (19:00–21:00 CET)",
                        "inline": True,
                    },
                    {
                        "name": "Main Raid (Waves 1–5)",
                        "value": "Sat 14 Nov 09:00 UTC -> Sun 15 Nov 20:00 UTC",
                        "inline": True,
                    },
                    {
                        "name": "Organizer Contact",
                        "value": f"{OWNER_SENDER_NAME} (`{OWNER_SENDER_EMAIL}`)",
                        "inline": False,
                    },
                ],
            }
        ],
    }
    return {
        "discord_invite_url": DISCORD_INVITE_URL,
        "markdown_post": markdown_post,
        "webhook_payload": webhook_payload,
    }


def build_v34_reddit_outreach() -> list[dict[str, str]]:
    """Build Reddit collab posts & 1-click submit links for `r/LocalLLaMA`, `r/BOINC`, `r/MachineLearning`, `r/homelab`."""
    subreddits = [
        (
            "LocalLLaMA",
            "[Open-Source Collab] HiveMind v3.4 x CrowdGPT: Upcycling a 1B Community LLM into an 8-Expert Sparse MoE via Decoupled OpenDiLoCo on Home GPUs (Weekend Raid Nov 14–15)",
        ),
        (
            "BOINC",
            "[Citizen Science AI] Thank You Early Volunteer GPUs! HiveMind@home v3.4 + New Web UI + Nov 14–15 CrowdGPT Distributed Training & Serving Raid",
        ),
        (
            "MachineLearning",
            "[P] Validating Asynchronous Sparse-MoE Upcycling + Decoupled DiLoCo (RDA + Spot-Checked Loss Ratchet) Across Heterogeneous Volunteer GPUs (Nov 14–15 Open Raid)",
        ),
        (
            "homelab",
            "[Open Source] Put Idle 8GB–24GB Home GPUs to Work: HiveMind v3.4 Zero-Trust Producer Shield + This Weekend's CrowdGPT-MoE Swarm Raid (Nov 14–15)",
        ),
    ]

    shared_body = f"""Hi everyone,

First, a huge **thank you** to everyone from Reddit and GitHub who jumped in over the past month as the first open-source contributors, testers, and volunteer GPU nodes on **Project HiveMind** (`AGPL-3.0`). With 55 registered community nodes and `363,655+` tokens routed in our first month, your feedback directly shaped **`v3.4.0`**.

### What We Just Shipped in HiveMind `v3.4.0`
1. **Rebuilt Bilingual Web UI (EN/FR) & P2P Compute Credits**: Hardened OWASP PBKDF2-600k auth, live `/telemetry` dashboards, preserved `hm-...` keys & balances for returning pioneers, and peer-to-peer reciprocal `HMT` transfers (strictly non-monetary compute quota — zero crypto/tokens for sale).
2. **Zero-Trust Home GPU Producer Shield (`:8787`)**: Share Ollama / PyTorch inference with `< 1.5s` automatic gaming preemption so your local GPU stays 100% yours whenever you launch a game or local job.

### Open-Source Collab Proposal: **HiveMind x CrowdGPT (`u/Vxtzq`)**
We are proposing and running a joint weekend experiment with **[CrowdGPT (`Vxtzq/Crowd-v1`)]({CROWDGPT_HF_URL})** on **Saturday 14 & Sunday 15 November 2026** (Preflight Friday 13 Nov `18:00 UTC`) to **empirically validate two open-source concepts at once**:

- **Concept 1 — Citizen-GPU Producer Serving**: Distributed OpenAI-compatible inference across `8 GB–24 GB` volunteer GPUs with verifiable reciprocity.
- **Concept 2 — 8x Sparse-MoE Upcycling + Decoupled OpenDiLoCo Training**:
  - Function-preserving upcycle of `Vxtzq/Crowd-v1` (`1.01B` dense + `CrowdEngram`) into **1 Shared Anchor + 8 Routed Domain Specialists (`hivemind/crowd-moe-v1`)** (`< 1.8 GiB` VRAM via LoRA-MoE on `8 GB` GPUs like RTX 2070 / 3060 / 4060).
  - Asynchronous inner AdamW steps aggregated via CPU-offloaded FP32 Nesterov outer momentum + DeepMind **Decoupled DiLoCo (`RDA`)** + **CPU Spot-Checked `LossCheckReceipt`** so poisoned updates never advance the checkpoint ratchet (2-round CUDA preflight held-out loss: **`5.3042 -> 5.1742 -> 0.9668`**).

### How to Join This Weekend (14–15 Nov 2026) as a Producer or Trainer
- **Code & `v3.4.0` Release (`AGPL-3.0`)**: {RELEASE_V340_URL}
- **Weekend Raid Architecture & `.ics` Schedule**: {PR_871_URL}
- **1-Click GPU / Trainer Node Signup**: {GITHUB_REPO_URL}/issues/new?template=04_crowdgpt_weekend_moe_raid.yml
- **Live Web UI & Telemetry**: {PUBLIC_OPENROUTER_URL}
- **Discord Coordination**: {DISCORD_INVITE_URL} (or reach me at `{OWNER_SENDER_EMAIL}`)

Would love your feedback, critiques on the `RDA` + spot-checked loss ratchet, and any idle GPUs (`8 GB+`) that want to join Waves 1–5 this weekend!
"""

    posts: list[dict[str, str]] = []
    for sub, title in subreddits:
        submit_url = f"https://www.reddit.com/r/{sub}/submit?" + urllib.parse.urlencode(
            {"selftext": "true", "title": title, "text": shared_body}
        )
        posts.append(
            {
                "subreddit": f"r/{sub}",
                "title": title,
                "body": shared_body,
                "submit_url": submit_url,
            }
        )
    return posts


def publish_v34_amcp_pioneer_outreach(
    *,
    gateway_url: str = PUBLIC_OPENROUTER_URL,
    api_key: str,
    handles: Sequence[str] = PIONEER_COMMUNITY_HANDLES,
) -> list[dict[str, Any]]:
    """Publish live AMCP v3.4 Pioneer Thank-You + Reconnect + Weekend Raid messages to `broadcast` & each pioneer."""
    if not isinstance(gateway_url, str) or not gateway_url.strip().startswith(("https://", "http://")):
        raise ValueError("gateway_url must be a valid http(s) URL")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("api_key must be a non-empty string")
    if isinstance(handles, (str, bytes)) or not isinstance(handles, Sequence):
        raise TypeError("handles must be a sequence of handle strings")

    endpoint = gateway_url.rstrip("/") + "/v1/hivemind/amcp/publish"
    messages: list[dict[str, Any]] = [
        {
            "recipient_id": "broadcast",
            "topic_key": "human_agent",
            "event_type": "io.hivemind.amcp.dev.proposal",
            "sender_kind": "gpu_producer",
            "payload": {
                "title": "Thank You Pioneers! HiveMind v3.4 Web UI + CrowdGPT Massive Swarm Raid (14-15 Nov 2026)",
                "body": (
                    f"From {OWNER_SENDER_NAME} ({OWNER_SENDER_EMAIL}): Huge thank you to our first 55 open-source "
                    "contributors, GPU producers, trainers, and adopters! HiveMind v3.4.0 is live with the new "
                    "bilingual EN/FR Web UI, P2P HMT point transfers, 1-command Producer Shield reconnection "
                    "(keys & HMT balances 100% preserved), and our joint CrowdGPT (@Vxtzq) collaboration. "
                    "Join as Producer or OpenDiLoCo Trainer this weekend (14-15 Nov 2026) to validate BOTH "
                    "citizen-GPU serving and 8-expert Sparse-MoE OpenDiLoCo training live!"
                ),
                "area": "model",
                "refs": [RELEASE_V340_URL, PR_871_URL, DISCORD_INVITE_URL],
            },
        }
    ]

    for raw_handle in handles:
        h = str(raw_handle).strip()
        if not h:
            continue
        messages.append(
            {
                "recipient_id": h,
                "topic_key": "human_agent",
                "event_type": "io.hivemind.amcp.message",
                "sender_kind": "gpu_producer",
                "payload": {
                    "text": (
                        f"Hi @{h}! Personal thank-you from Xavier ({OWNER_SENDER_EMAIL}) for being one of the very "
                        "first HiveMind pioneers! Your hm-... key and HMT balance are 100% preserved on the new v3.4 "
                        "EN/FR Web UI (with P2P HMT transfers + 500 HMT bonuses). We'd love to reconnect with you as "
                        "a GPU Producer or OpenDiLoCo Trainer for this weekend's HiveMind x CrowdGPT Massive Swarm "
                        f"Raid (Sat 14 - Sun 15 Nov 2026) to validate both open-source concepts together: {PR_871_URL}"
                    ),
                },
            }
        )

    receipts: list[dict[str, Any]] = []
    for msg in messages:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(msg).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json",
                "User-Agent": "hivemind-v3.4-pioneer-outreach/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        receipts.append(
            {
                "recipient_id": msg["recipient_id"],
                "event_id": body.get("event_id") or body.get("id"),
                "topic": body.get("topic"),
                "status": body.get("status", "ok"),
            }
        )
    return receipts


def send_discord_webhook(webhook_url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Post the `v3.4.0` Pioneer & Weekend Raid announcement to a Discord Webhook URL."""
    if not isinstance(webhook_url, str) or not webhook_url.strip().startswith("https://discord.com/api/webhooks/"):
        raise ValueError("webhook_url must be a valid https://discord.com/api/webhooks/... URL")
    body = payload if payload is not None else build_v34_discord_outreach()["webhook_payload"]
    req = urllib.request.Request(
        webhook_url.strip(),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "hivemind-v3.4-discord/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15.0) as resp:
        return {"status_code": resp.status, "delivered": 200 <= resp.status < 300}
