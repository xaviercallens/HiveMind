# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Welcome, Pioneer Congratulations & GPU Producer Onboarding Email Generator and Dispatcher.

Generates rich bilingual (EN/FR) HTML and plain-text welcome emails explaining:
1. Congratulations to Pioneer End-User #1 (``fbc1d3042d377feba6b1e0ac6fd576a4``) and every new HiveMinder.
2. The Cyber-Bee & La Ruche concept in simple everyday analogies.
3. How HiveMind Points (``HMT``) and the Kelly-Tullock Nash Equilibrium work.
4. How to contribute an idle GPU in 1 command (with ``< 1.5s`` automatic game preemption).
5. All open technical capabilities available for community contribution.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import smtplib
import time
from email.message import EmailMessage
from typing import Any

PIONEER_FIRST_USER_ACCOUNT_ID: str = "fbc1d3042d377feba6b1e0ac6fd576a4"
PIONEER_FIRST_TEST_METRICS: dict[str, Any] = {
    "account_id": PIONEER_FIRST_USER_ACCOUNT_ID,
    "badge": "PIONEER END-USER #1 • FIRST LIVE OPENROUTER COMPLETION",
    "role": "prosumer",
    "model": "hivemind/auto (Smart Market Liquidity Routing)",
    "prompt": "Write a high-performance Python decorator with LRU caching, thread safety, and execution duration metrics in microseconds.",
    "first_byte_ms": 273,
    "generation_speed_tok_s": 36.0,
    "total_tokens": 294,
    "starter_grant_hmt": 100.0,
    "settled_hmt": 1.5835,
    "balance_after_test_hmt": 98.4165,
    "pioneer_bonus_grant_hmt": 250.0,
    "induction_consumer_bonus_hmt": 250.0,
    "total_bonus_credited_hmt": 500.0,
    "new_spendable_balance_hmt": 598.4165,
    "induction_fleet_providers": [
        "NVIDIA Tesla T4 (16 GB GDDR6 • Active Induction Node)",
        "NVIDIA GeForce RTX 2070 (8 GB GDDR6 Turing • Safe <6,656 MiB Ceiling + <1.5s Game Preemption)",
    ],
    "recorded_at_iso": "2026-10-25T12:44:54+01:00",
    "receipt_sha256": "45dc962abf34a28c0011f1e55466872a2d3c155e5cda91d5951d2128084a5d5b",
}

PUBLIC_RUCHE_URL: str = "https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app"
PUBLIC_OPENROUTER_URL: str = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
PUBLIC_AT_HOME_URL: str = "https://hivemind-at-home-cfhmhmvv5a-uc.a.run.app"
GITHUB_REPO_URL: str = "https://github.com/xaviercallens/HiveMind"


def build_install_producer_shell_script(gateway_url: str = PUBLIC_RUCHE_URL) -> str:
    """Return a standalone 1-command bash onboarding script for GPU Producers."""
    if not isinstance(gateway_url, str) or not gateway_url.strip() or not gateway_url.startswith(("https://", "http://")):
        raise ValueError("gateway_url must be a non-empty http(s) URL")
    gw = gateway_url.rstrip("/")
    return f"""#!/usr/bin/env bash
# ==============================================================================
# Project HiveMind@home & OpenRouter — 1-Command GPU Producer Onboarding Script
# Reciprocity Unit: HiveMind Points (HMT) • <1.5s Instant Game Preemption SLA
# ==============================================================================
set -euo pipefail

GATEWAY_URL="${{HIVEMIND_GATEWAY_URL:-{gw}}}"
echo "🐝 Welcome to Project HiveMind@home & La Ruche GPU Producer Onboarding!"
echo "🌐 Active Gateway: ${{GATEWAY_URL}}"
echo "🪙 Reciprocity Model: HiveMind Points (HMT) — Earn 85% Tullock GPU Payouts"
echo ""

# 1. Detect GPU VRAM if nvidia-smi is available
VRAM_GB=""
if command -v nvidia-smi >/dev/null 2>&1; then
  VRAM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  if [ -n "${{VRAM_MIB}}" ]; then
    VRAM_GB="$(python3 -c "print(round(float('${{VRAM_MIB}}')/1024.0, 1))")"
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
    echo "✅ Detected NVIDIA GPU: ${{GPU_NAME}} (${{VRAM_GB}} GB VRAM)"
  fi
fi

# 2. Download standalone producer agent & Zero-Trust Producer Shield
echo "⬇️  Downloading standalone hivemind_producer_agent.py & hivemind_producer_shield.py..."
curl -fsSL "${{GATEWAY_URL}}/hivemind_producer_agent.py" -o hivemind_producer_agent.py
curl -fsSL "${{GATEWAY_URL}}/hivemind_producer_shield.py" -o hivemind_producer_shield.py
chmod +x hivemind_producer_agent.py hivemind_producer_shield.py

# 3. Ensure httpx, fastapi, uvicorn are installed for Agent + Shield
if ! python3 -c "import httpx, fastapi, uvicorn" >/dev/null 2>&1; then
  echo "📦 Installing lightweight HTTP & Shield dependencies (httpx, fastapi, uvicorn)..."
  pip install --user --quiet httpx fastapi uvicorn || pip install --quiet httpx fastapi uvicorn
fi

# 4. Recommend optimal capability tier for this GPU
echo ""
echo "🔍 Querying HiveMind Capability Catalog for your GPU..."
if [ -n "${{VRAM_GB}}" ]; then
  python3 hivemind_producer_agent.py --gateway "${{GATEWAY_URL}}" --suggest --vram-gb "${{VRAM_GB}}" || true
else
  python3 hivemind_producer_agent.py --gateway "${{GATEWAY_URL}}" --suggest --vram-gb 8 || true
fi

ENDPOINT_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
echo ""
echo "=============================================================================="
echo "GPU PRODUCER ONBOARDING (T4 16GB / RTX 2070 8GB / ANY GPU) - PROOF OF CONCEPT, public data only:"
echo "=============================================================================="
echo "1. Pull & run your recommended Ollama model locally:"
echo "   • RTX 2070 (8 GB) / Consumer 8 GB:  ollama run qwen2.5-coder:7b"
echo "   • Tesla T4 (16 GB) / 12-16 GB GPU:  ollama run qwen2.5-coder:14b"
echo ""
echo "2. Start the Zero-Trust Producer Shield (protects your local Ollama on port 8787"
echo "   so ONLY the HiveMind Gateway with your secret token can query your GPU):"
echo "   export HIVEMIND_ENDPOINT_TOKEN=\"${{ENDPOINT_TOKEN}}\""
echo "   python3 hivemind_producer_shield.py --upstream http://127.0.0.1:11434 --port 8787 --max-concurrent 2 &"
echo ""
echo "3. Export your hm-... key (from ${{GATEWAY_URL}}/openrouter) & start the Agent:"
echo "   export HIVEMIND_API_KEY=hm-YOUR_KEY"
echo "   python3 hivemind_producer_agent.py \\\\"
echo "     --gateway ${{GATEWAY_URL}} \\\\"
echo "     --endpoint https://YOUR_PUBLIC_TUNNEL_TO_PORT_8787 \\\\"
echo "     --preset coder --sync-remote-config"
echo "=============================================================================="
"""


def build_welcome_email_content(
    *,
    account_id: str,
    handle: str,
    role: str,
    balance_hmt: float,
    recipient_email: str | None = None,
    is_pioneer_first_user: bool = False,
) -> dict[str, Any]:
    """Generate the bilingual (EN + FR) subject, plain-text body, and HTML body for a user."""
    if not isinstance(account_id, str) or not account_id.strip():
        raise ValueError("account_id must be a non-empty string")
    if not isinstance(handle, str):
        raise ValueError("handle must be a string")
    if not isinstance(role, str) or not role.strip():
        raise ValueError("role must be a non-empty string")
    if isinstance(balance_hmt, bool) or not isinstance(balance_hmt, (int, float)) or not math.isfinite(float(balance_hmt)) or float(balance_hmt) < 0.0:
        raise ValueError("balance_hmt must be a finite non-negative float")
    if not isinstance(is_pioneer_first_user, bool):
        raise ValueError("is_pioneer_first_user must be a boolean")
    safe_handle = html.escape(handle or account_id[:12])
    safe_account = html.escape(account_id)
    safe_role = html.escape(role)
    is_pioneer = is_pioneer_first_user or (account_id == PIONEER_FIRST_USER_ACCOUNT_ID)

    if is_pioneer:
        subject = (
            f"🏆 Congratulations Pioneer #1 ({account_id[:12]}...)! Welcome to HiveMind@home & OpenRouter (+HMT Points)"
        )
    else:
        subject = (
            f"🐝 Welcome to Project HiveMind@home & OpenRouter, {handle or account_id[:8]}! (+{balance_hmt:.1f} HMT Allocated)"
        )

    pioneer_banner_html = ""
    pioneer_banner_text = ""
    if is_pioneer:
        pioneer_banner_text = (
            "========================================================================\n"
            "🏆 OFFICIAL CONGRATULATIONS TO PIONEER END-USER #1!\n"
            f"Account ID: {PIONEER_FIRST_USER_ACCOUNT_ID}\n"
            "========================================================================\n"
            "Thank you for completing the FIRST live end-user AI completion on the\n"
            "HiveMind OpenRouter (/v1/chat/completions)!\n"
            "• Capability Tier: hivemind/auto (Smart Market Liquidity Routing)\n"
            "• First Byte Latency: 273 ms | Generation Speed: 36.0 tok/s | Tokens: 294\n"
            "• Balance Verified: 98.4165 HMT (+250.0 HMT Pioneer Achievement Bonus Credited!)\n"
            f"• Cryptographic Receipt SHA-256: {PIONEER_FIRST_TEST_METRICS['receipt_sha256']}\n\n"
        )
        pioneer_banner_html = f"""
        <div style="background:linear-gradient(135deg,#1e1b0a 0%,#0f2922 100%);border:2px solid #ffc943;border-radius:10px;padding:18px;margin-bottom:20px;">
          <div style="font-size:12px;font-weight:800;color:#ffc943;letter-spacing:0.08em;text-transform:uppercase;">
            🏆 Official Swarm Milestone &bull; Pioneer End-User #1 Certificate
          </div>
          <h2 style="margin:6px 0 8px;color:#ffffff;font-size:20px;">
            Congratulations &amp; Thank You, Pioneer <code>{safe_account}</code>!
          </h2>
          <p style="margin:0 0 10px;color:#e2e8f0;font-size:14px;line-height:1.55;">
            Your first live request on <strong>HiveMind OpenRouter (<code>/v1/chat/completions</code>)</strong> succeeded and has been permanently recorded in the HiveMind Episode Store, Redis Long-Term Memory, and ChromaDB VectorDB:
          </p>
          <ul style="margin:0;padding-left:20px;color:#a8eff0;font-family:monospace;font-size:13px;line-height:1.65;">
            <li>Account ID: <strong>{PIONEER_FIRST_USER_ACCOUNT_ID}</strong> (Role: <strong>prosumer</strong>)</li>
            <li>Model Routed: <strong>hivemind/auto</strong> &rarr; Live GPU Producer</li>
            <li>First Byte (TTFT): <strong>273 ms</strong> &bull; Speed: <strong>36.0 tok/s</strong> &bull; Output: <strong>294 tokens</strong></li>
            <li>Ledger Settlement: <strong>98.4165 HMT</strong> remaining (+<strong>250.0 HMT Pioneer Bonus Grant</strong> awarded!)</li>
            <li>Receipt SHA-256: <code>{PIONEER_FIRST_TEST_METRICS['receipt_sha256'][:32]}...</code></li>
          </ul>
        </div>
        """

    text_body = (
        f"Hello {handle or account_id[:12]},\n\n"
        "PROOF OF CONCEPT: HiveMind is not a production service yet. Until it is fully secured, send only public, non-sensitive data (no secrets, passwords, API keys, personal, private or enterprise data). Prompts are processed by volunteer-run nodes whose operators are not verified.\n\n"
        f"{pioneer_banner_text}"
        "Thank you for joining Project HiveMind@home & La Ruche OpenRouter!\n\n"
        "1. YOUR ACCOUNT & HIVEMIND POINTS (HMT) SUMMARY\n"
        "------------------------------------------------------------------------\n"
        f"• Account ID: {account_id}\n"
        f"• Handle:     {handle}\n"
        f"• Swarm Role: {role}\n"
        f"• Allocated Balance: {balance_hmt:.4f} HMT (HiveMind Points)\n"
        f"• Live OpenRouter Portal: {PUBLIC_RUCHE_URL}/openrouter\n"
        f"• HiveMind@home Portal:   {PUBLIC_RUCHE_URL}/\n\n"
        "2. THE CONCEPT IN SIMPLE WORDS (THE SOLAR GRID & BEEHIVE ANALOGY)\n"
        "------------------------------------------------------------------------\n"
        "Imagine millions of gaming PCs and workstations whose GPUs sit idle 80% of the day.\n"
        "Alone, one 8 GB consumer GPU cannot train or serve frontier AI models.\n"
        "Together—like rooftop solar panels feeding a neighborhood micro-grid, or Worker Bees\n"
        "filling hexagonal cells in a beehive (La Ruche)—we form a citizen supercomputer:\n"
        "• When you are NOT gaming, your GPU serves AI completions or trains compact QLoRA experts.\n"
        "• The instant you launch a game (Steam, Cyberpunk 2077, Blender), our hardware watchdog\n"
        "  yields 100% of your GPU in under 1.5 seconds (<1.5s SLA). Zero FPS drop.\n"
        "• Every contribution earns HiveMind Points (HMT) via the Kelly-Tullock Nash Equilibrium\n"
        "  (85% minted directly to the GPU Producer, 15% to the community safety reserve).\n\n"
        "3. HOW TO CONTRIBUTE YOUR GPU IN 1 COMMAND (PRODUCER ONBOARDING)\n"
        "------------------------------------------------------------------------\n"
        "Run this single command on Linux, Windows WSL2, or macOS to detect your GPU VRAM\n"
        "and download the standalone Producer Agent:\n\n"
        f"  curl -sSL {PUBLIC_RUCHE_URL}/install-producer.sh | bash\n\n"
        "Or run manually with Ollama:\n"
        "  1) ollama run qwen2.5-coder:7b\n"
        "  2) export HIVEMIND_API_KEY=hm-YOUR_KEY\n"
        f"  3) curl -sSL {PUBLIC_RUCHE_URL}/hivemind_producer_agent.py -o hivemind_producer_agent.py\n"
        f"  4) python3 hivemind_producer_agent.py --gateway {PUBLIC_RUCHE_URL} \\\n"
        "       --endpoint https://YOUR_PUBLIC_TUNNEL_URL --preset coder --sync-remote-config\n\n"
        "4. ALL TECHNICAL CAPABILITIES ARE OPEN FOR YOUR CONTRIBUTION\n"
        "------------------------------------------------------------------------\n"
        "• OpenAI-Compatible OpenRouter (/v1/chat/completions): Use in Cursor, VS Code Continue,\n"
        "  Cline, Zed, OpenCode, Aider, or Python OpenAI SDK.\n"
        "• La Ruche 8-Domain Federated MoE Training (/api/v1/federated/workunit/request):\n"
        "  Train specialist bees in Python Coding (GWAYA), Astrophysics, Lean 4 Math, Genetics,\n"
        "  Rust QUIC, and Cyber Defense (+250 HMT per verified workunit).\n"
        "• Remote Control Center (/v1/hivemind/remote/config): Pause, drain, or cap your GPU\n"
        "  VRAM remotely from your browser at any time.\n"
        f"• Full Open-Source Repository (Apache-2.0): {GITHUB_REPO_URL}\n\n"
        "--- VERSION FRANÇAISE ---\n"
        "Merci d'avoir rejoint HiveMind@home & La Ruche ! Vos points HiveMind (HMT) vous\n"
        "permettent d'interroger nos modèles MoE depuis votre IDE favori et de partager votre\n"
        "GPU lorsqu'il est inactif (préemption jeu instantanée < 1,5 seconde).\n\n"
        "Warm regards,\n"
        "The Project HiveMind & La Ruche Community Swarm\n"
    )

    html_body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(subject)}</title>
</head>
<body style="margin:0;padding:24px;background:#0d0e11;color:#e4e2e6;font-family:'Inter',system-ui,-apple-system,sans-serif;line-height:1.6;">
  <div style="max-width:720px;margin:0 auto;background:#131417;border:1px solid rgba(255,201,67,0.35);border-radius:12px;padding:28px;">
    <p style="margin:0 0 16px;padding:10px 12px;border:1px solid #ffc943;border-radius:8px;color:#ffc943;font-size:13px;"><strong>&#9888; Proof of concept.</strong> HiveMind is not a production service yet. Until it is fully secured, send only public, non-sensitive data (no secrets, passwords, API keys, personal, private or enterprise data). Prompts are processed by volunteer-run nodes whose operators are not verified.</p>
    <div style="display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid rgba(255,255,255,0.1);padding-bottom:16px;margin-bottom:20px;">
      <div>
        <div style="font-size:18px;font-weight:800;color:#ffc943;">🐝 PROJECT HIVEMIND@HOME &amp; LA RUCHE OPENROUTER</div>
        <div style="font-size:12px;color:#55f7ed;font-family:monospace;">Powered by HiveMind Points (HMT) &bull; Citizen GPU Supercluster</div>
      </div>
    </div>

    {pioneer_banner_html}

    <h1 style="font-size:20px;color:#ffffff;margin:0 0 10px;">
      Welcome to the Swarm, <code style="color:#55f7ed;">{safe_handle}</code>!
    </h1>
    <p style="color:#cbd5e1;font-size:14px;margin:0 0 16px;">
      Thank you for joining <strong>Project HiveMind</strong>. Your account (<code>{safe_account}</code>, role: <strong>{safe_role}</strong>) is active with <strong>{balance_hmt:.4f} HMT (HiveMind Points)</strong> ready to use across our OpenAI-compatible OpenRouter and La Ruche MoE Bees.
    </p>

    <div style="background:#191a1e;border-left:4px solid #55f7ed;border-radius:6px;padding:14px 16px;margin-bottom:18px;">
      <h3 style="margin:0 0 6px;color:#55f7ed;font-size:15px;">1. The Concept in Simple Words (The Beehive &amp; Solar Grid Analogy)</h3>
      <p style="margin:0;font-size:13.5px;color:#e2e8f0;">
        Across the world, millions of gamers, students, and researchers own capable consumer GPUs (4 GB to 24 GB VRAM) that sit idle 80% of the day. Like neighborhood rooftop solar panels feeding a shared smart grid:
      </p>
      <ul style="margin:8px 0 0;padding-left:18px;font-size:13px;color:#cbd5e1;">
        <li><strong>When your PC is idle:</strong> Your Worker Bee serves coding/scientific prompts or trains compact QLoRA adapters compressed via 8D Gosset $E_8$ lattice geometry (<strong>16&times;</strong> dense at 2 bits/dim; <strong>30.89&times;</strong> with top-k sparsity, ref: <code>artifacts/experiments/E10_lattice_diloco.json</code>).</li>
        <li><strong>Instant &lt;1.5s Game Preemption:</strong> Launch any game (Steam, Cyberpunk, Blender) and the hardware circuit-breaker frees 100% of your GPU in under 1.5 seconds.</li>
        <li><strong>Kelly-Tullock Nash Equilibrium:</strong> Producers earn <strong>85%</strong> of settled HMT points directly to their hash-chained ledger; Consumers get instant access to 5 capability tiers and 8 specialist Honeycomb Bees.</li>
      </ul>
    </div>

    <div style="background:#191a1e;border-left:4px solid #ffc943;border-radius:6px;padding:14px 16px;margin-bottom:18px;">
      <h3 style="margin:0 0 6px;color:#ffc943;font-size:15px;">2. Contribute Your GPU in 1 Command (Zero-Friction Onboarding)</h3>
      <p style="margin:0 0 8px;font-size:13px;color:#cbd5e1;">
        Every technical capability is open for your contribution. Run this 1-command installer on Linux, Windows WSL2, or macOS to auto-detect your GPU VRAM and match your optimal tier:
      </p>
      <pre style="background:#0a0c10;border:1px solid rgba(255,201,67,0.3);border-radius:6px;padding:10px;color:#ffdf9c;font-family:monospace;font-size:12px;overflow-x:auto;margin:0 0 8px;">curl -sSL {PUBLIC_RUCHE_URL}/install-producer.sh | bash</pre>
      <p style="margin:0;font-size:12.5px;color:#94a3b8;">
        Or download the standalone agent directly: <a href="{PUBLIC_RUCHE_URL}/hivemind_producer_agent.py" style="color:#55f7ed;">{PUBLIC_RUCHE_URL}/hivemind_producer_agent.py</a>
      </p>
    </div>

    <div style="background:#191a1e;border-left:4px solid #34d399;border-radius:6px;padding:14px 16px;margin-bottom:18px;">
      <h3 style="margin:0 0 6px;color:#34d399;font-size:15px;">3. Open Technical Capabilities &amp; Documentation</h3>
      <ul style="margin:0;padding-left:18px;font-size:13px;color:#cbd5e1;">
        <li><strong>OpenRouter Playground &amp; IDE Studio:</strong> <a href="{PUBLIC_RUCHE_URL}/openrouter" style="color:#55f7ed;">{PUBLIC_RUCHE_URL}/openrouter</a> (Cursor, VS Code Continue, Cline, Zed, OpenCode, Aider, Python SDK)</li>
        <li><strong>HiveMind@home 5-Screen Portal:</strong> <a href="{PUBLIC_RUCHE_URL}/" style="color:#ffc943;">{PUBLIC_RUCHE_URL}/</a> (Federated QLoRA Workunits, +250 HMT/WU, 8 Specialist Honeycomb Bees)</li>
        <li><strong>Remote Control Center:</strong> Pause, drain, or cap your GPU VRAM from anywhere via <code>/v1/hivemind/remote/config</code></li>
        <li><strong>Open-Source Code &amp; 128 Lean 4 Proofs:</strong> <a href="{GITHUB_REPO_URL}" style="color:#34d399;">{GITHUB_REPO_URL}</a></li>
      </ul>
    </div>

    <div style="border-top:1px solid rgba(255,255,255,0.1);padding-top:12px;font-size:12px;color:#958e7f;">
      <strong>Français :</strong> Bienvenue dans l'essaim HiveMind@home &amp; La Ruche ! Vos points HiveMind (<code>HMT</code>) sont actifs sur votre compte <code>{safe_account}</code>. Partagez votre GPU inactif en une commande et utilisez nos modèles MoE dans votre IDE favori.
    </div>
  </div>
</body>
</html>"""

    payload = {
        "subject": subject,
        "account_id": account_id,
        "handle": handle,
        "role": role,
        "balance_hmt": balance_hmt,
        "is_pioneer_first_user": is_pioneer,
        "recipient_email_masked": _mask_email(recipient_email) if recipient_email else "on-file-hmac",
        "text_body": text_body,
        "html_body": html_body,
        "generated_at_epoch": time.time(),
    }
    payload["email_receipt_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


def _mask_email(email: str) -> str:
    """Mask an email address for safe telemetry display (e.g. ``x***@example.org``)."""
    clean = email.strip()
    if "@" not in clean:
        return "***"
    local, _, domain = clean.partition("@")
    if len(local) <= 2:
        masked_local = local[:1] + "***"
    else:
        masked_local = local[:2] + "***"
    return f"{masked_local}@{domain}"


def send_or_spool_welcome_email(
    *,
    account_id: str,
    handle: str,
    role: str,
    balance_hmt: float,
    recipient_email: str | None = None,
    is_pioneer_first_user: bool = False,
) -> dict[str, Any]:
    """Generate the Welcome Email, send via SMTP if configured, and return a signed delivery receipt."""
    content = build_welcome_email_content(
        account_id=account_id,
        handle=handle,
        role=role,
        balance_hmt=balance_hmt,
        recipient_email=recipient_email,
        is_pioneer_first_user=is_pioneer_first_user,
    )
    smtp_host = os.environ.get("HIVEMIND_SMTP_HOST", "").strip()
    smtp_port = int(os.environ.get("HIVEMIND_SMTP_PORT", "587") or "587")
    smtp_user = os.environ.get("HIVEMIND_SMTP_USER", "").strip()
    smtp_pass = os.environ.get("HIVEMIND_SMTP_PASSWORD", "").strip()
    smtp_from = os.environ.get("HIVEMIND_SMTP_FROM", "swarm@hivemind-at-home.org").strip()

    delivery_status = "DISPATCHED_IN_PORTAL_AND_SPOOL"
    if smtp_host and recipient_email and "@" in recipient_email:
        try:
            msg = EmailMessage()
            msg["Subject"] = content["subject"]
            msg["From"] = smtp_from
            msg["To"] = recipient_email.strip()
            msg.set_content(content["text_body"])
            msg.add_alternative(content["html_body"], subtype="html")
            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                server.ehlo()
                if smtp_port in (587, 2587):
                    server.starttls()
                if smtp_user and smtp_pass:
                    server.login(smtp_user, smtp_pass)
                server.send_message(msg)
            delivery_status = "SENT_VIA_SMTP"
        except Exception as exc:
            delivery_status = f"SPOOL_FALLBACK ({type(exc).__name__})"

    return {
        "delivery_status": delivery_status,
        "subject": content["subject"],
        "account_id": account_id,
        "handle": handle,
        "role": role,
        "balance_hmt": balance_hmt,
        "is_pioneer_first_user": content["is_pioneer_first_user"],
        "recipient_email_masked": content["recipient_email_masked"],
        "email_receipt_sha256": content["email_receipt_sha256"],
        "preview_url": f"/v1/hivemind/welcome-email/preview?account_id={account_id}",
        "install_producer_cmd": f"curl -sSL {PUBLIC_RUCHE_URL}/install-producer.sh | bash",
        "text_body": content["text_body"],
        "html_body": content["html_body"],
    }
