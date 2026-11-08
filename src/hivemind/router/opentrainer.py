# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""OpenTrainer: Central Intense-Training Coordinator & Firewall-Traversal Relay for HiveMind.

Unifies the four distributed training and volunteer compute layers built across the RTX 2070 (8 GB)
and GCP Tesla T4 (16 GB) fleet into one central component (`OpenTrainer`) accessible by accounts with
the ``trainer`` (or ``prosumer`` / ``producer``) role:

1. **Tier B — Outbound HTTPS Relay (`outbound-https-relay`, default for firewalled home GPUs):**
   - Home GPUs behind carrier/residential NAT or strict firewalls (e.g. Windows 10/11 RTX 2070 / RTX 4070)
     never open an inbound port and never bind ``0.0.0.0`` (per PR #401).
   - Nodes connect **outbound only over HTTPS (port 443)** to ``/v1/hivemind/opentrainer/heartbeat`` and
     ``/v1/hivemind/opentrainer/submit-delta``.
   - ``OpenTrainer`` bridges signed task offers and LoRA delta results directly onto the AMCP/Kafka event
     store (`hivemind.amcp.federation.v1` / `amcp_kafka_events`).

2. **Tier A — Signed Task Agents over AMCP & Kafka (`amcp-kafka-aglet`, PR #410):**
   - Implements the IBM Aglet life cycle (*dispatch* ``task_offer`` -> *arrival* ``check_offer`` ->
     *clone* ``task_claim`` -> *result* ``task_result`` -> *retract* ``task_abort``) **without mobile code**.
   - Code is pinned via CRLF-normalized SHA-256 (`code_pin`); tensor payloads use ``torch.load(..., weights_only=True)``
     and canonical SHA-256 verification (`tensors_sha256`).

3. **Two-Host & Multi-Host Co-Training Tunnel (`ssh-loopback-tunnel` / `wireguard-t4-hub`, PRs #401, #407, #410):**
   - Coordinates the **RTX 2070 (`XAVGAMSRV`, 8 GB, Windows 11)** and **GCP Tesla T4 (`socreateai-agora-hermes-node1`, 16 GB, Linux)**:
     * Stage 1 E8 Two-Host WAN Cell (PR #407): 12 rounds, ``K=2``, ``--merge rda``, 400 inner steps, seeds 1-3,
       paired difference ``+4.8e-07 +/- 8.3e-07`` vs single-host baseline, round latency ``p50 = 1.20 s``, ``p95 = 3.79 s``.
     * Stage 2 Tiny Granite-MoE NF4 QLoRA (PR #410): ``GraniteMoeForCausalLM`` (4 experts, top-2 routing), NF4 + LoRA
       (``r=4, alpha=8``) + output head, ``>= 4,300 MiB`` free VRAM guard, loss ``4.1721 -> 1.7772`` matching single-process
       baseline to ``0.0000`` difference while shuffled-label control stays at chance (``4.2034``).
     * Live 3-Broker Kafka + T4 Churn Soak (PR #410): 70 rounds (28.6 min) with ``SIGKILL`` learner restarts,
       62 rounds complete, 8 closed by quorum, 0 rejections, loss ``4.171 -> 1.448``.

4. **Tier C — BOINC Nocturnal Workunits & Truthful CUDA Probe (`boinc-night-workunit`, PRs #366, #399, #408):**
   - Integrates stock BOINC clients (`cuda_hivemind`, `cuda_hivemind_big`, `cpu_hivemind_solver`), gamer idle-GPU
     protection (`hivemind_volunteer_config.py`, PR #399), truthful gated CUDA device verification (PR #408),
     and monotone validation-loss ratchet (PR #366).
"""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

OPENTRAINER_VERSION = "1.0.0-s36"
OPENTRAINER_TOPIC = "hivemind.amcp.federation.v1"

VALID_OPENTRAINER_CAMPAIGN_IDS: frozenset[str] = frozenset(
    {
        "E8-STAGE2-TINY-GRANITE-MOE-QLORA",
        "opentrainer-rtx2070-t4-co-train-v1",
        "NIGHT-REDDIT-VISION-MOE-V1",
        "MASSIVE-SWARM-IMAGE-MOE-V1",
        "BOINC-MOE-GRANITE-6DOMAIN",
        "NIGHT-MOE-1-BEE-STARS",
        "NIGHT-MOE-2-EXOPLANET",
        "NIGHT-MOE-2-VERIFIED-MATHS",
        "NIGHT-MOE-2-CODE-AND-CYBER",
        "NIGHT-SCI-WAVE3-2026",
        "SUBJ-01-QWEN35-CODER-POLYGLOT",
        "SUBJ-02-RUSTY-SUNDIALS-NUMERIC",
        "SUBJ-03-ASTROPHYSICS-EXOPLANET",
        "SUBJ-04-LEAN4-FORMAL-PROOFS",
        "SUBJ-05-CYBER-DEFENSE-FIREWALL",
        "SUBJ-06-CHEM-BIO-MED-DISCOVERY",
    }
)
_BYOM_CAMPAIGN_RE = re.compile(r"^byom-[A-Za-z0-9._-]{1,100}$")


def is_allowed_opentrainer_campaign_id(
    campaign_id: str,
    registered_account_campaigns: Iterable[str] = (),
) -> bool:
    """Return True if ``campaign_id`` is in the canonical allow-list, matches ``byom-<name>``, or was registered in a heartbeat of that account (Issue #618 / IP-259)."""
    cid = str(campaign_id or "").strip()
    if not cid:
        return False
    if cid in VALID_OPENTRAINER_CAMPAIGN_IDS:
        return True
    if _BYOM_CAMPAIGN_RE.match(cid) is not None:
        return True
    acct_set = {str(c).strip() for c in registered_account_campaigns if c and str(c).strip()}
    return cid in acct_set

NETWORK_MODES: tuple[str, ...] = (
    "outbound-https-relay",
    "amcp-kafka-aglet",
    "ssh-loopback-tunnel",
    "wireguard-t4-hub",
    "wstunnel-https-wg",
    "headscale-tailscale",
    "nebula-mesh",
    "boinc-night-workunit",
)

SUPPORTED_STACKS: tuple[str, ...] = (
    "boinc",
    "amcp_kafka",
    "aglet_qlora",
    "diloco_dct",
    "stage2_granite_moe",
)

# Minimum free VRAM guard (in MiB) by hardware tier so intense training never starves the host OS or daytime inference.
DEFAULT_MIN_FREE_VRAM_MIB_T4 = 4300
DEFAULT_MIN_FREE_VRAM_MIB_8GB = 1536
DEFAULT_MAX_DELTA_L2 = 25.0


FIREWALL_TRAVERSAL_MODES: Sequence[dict[str, Any]] = (
    {
        "mode_id": "outbound-https-relay",
        "tier": "Tier B (Default — Zero Firewall / NAT Config)",
        "inbound_ports_required": 0,
        "protocol": "Outbound HTTPS (TCP/443) to /v1/hivemind/opentrainer/*",
        "how_it_works": (
            "The trainer node initiates outbound-only HTTPS requests to OpenTrainer on the Cloud Run gateway. "
            "OpenTrainer validates signed Aglet task claims and QLoRA/DiLoCo delta submissions and bridges them "
            "into the AMCP/Kafka event ledger. Works behind any home router, carrier-grade NAT (CGNAT), or corporate firewall."
        ),
        "recommended_for": "Home Windows/Linux/macOS GPUs (RTX 2070, RTX 3060, RTX 4070) without public IPs or UDP 51820.",
    },
    {
        "mode_id": "amcp-kafka-aglet",
        "tier": "Tier A (Direct AMCP v1.5 + Kafka Signed Task Agents)",
        "inbound_ports_required": 0,
        "protocol": "Outbound TCP/9092 over WireGuard overlay (10.77.0.0/24) to 3 Kafka brokers",
        "how_it_works": (
            "Learners subscribe to topic hivemind.amcp.federation.v1, verify HMAC-signed task_offer envelopes "
            "(code_pin SHA-256 with CRLF normalization + base/global tensor SHA-256, weights_only=True, 64 KiB cap), "
            "train inner_steps on their non-IID shard, and publish task_result deltas."
        ),
        "recommended_for": "Dedicated nodes on the 10.77.0.0/24 overlay (GCP Tesla T4 + RTX 2070 via WireGuard T4 Hub).",
    },
    {
        "mode_id": "ssh-loopback-tunnel",
        "tier": "Two-Host WAN Cell Tunnel (Loopback-Only Syncer)",
        "inbound_ports_required": 0,
        "protocol": "Outbound SSH port-forward (127.0.0.1:18766 -> 127.0.0.1:8765, permitopen restricted)",
        "how_it_works": (
            "Per PR #401 and PR #407, the syncer never binds 0.0.0.0; it listens strictly on 127.0.0.1:8765 / 10.77.0.4:8765. "
            "The remote RTX 2070 learner (XAVGAMSRV) forwards a local loopback port over SSH with a key restricted to "
            "permitopen=\"127.0.0.1:8765\" and no shell execution."
        ),
        "recommended_for": "Cross-host E8 Stage 1 (CPU RDA) and Stage 2 (NF4 QLoRA Tiny-MoE) experiments when UDP is blocked.",
    },
    {
        "mode_id": "wireguard-t4-hub",
        "tier": "Tier A Overlay Hub & Open-Source Mesh Alternatives",
        "inbound_ports_required": 0,
        "protocol": "Outbound UDP/51820 to 34.48.48.154:51820 (or wstunnel TCP/443, Headscale, Nebula MIT)",
        "how_it_works": (
            "The T4 VM acts as a single WireGuard hub (deploy/amcp/hm-wg-hub.sh) forwarding 10.77.0.5/32 (RTX 2070) "
            "to 10.77.0.0/24 (Kafka brokers) with iptables drop rules blocking VPC access. Where UDP/51820 is blocked, "
            "trainers use wstunnel (WireGuard over WebSocket/TLS), Headscale + Tailscale DERP, or Nebula certificates."
        ),
        "recommended_for": "Persistent multi-host co-training between XAVGAMSRV (RTX 2070) and socreateai-agora-hermes-node1 (Tesla T4).",
    },
    {
        "mode_id": "boinc-night-workunit",
        "tier": "Tier C (BOINC Nocturnal Workunits & Idle-GPU Worker)",
        "inbound_ports_required": 0,
        "protocol": "Outbound HTTPS (TCP/443) to HiveMind La Ruche BOINC Coordinator",
        "how_it_works": (
            "Standard BOINC client (boinccmd) and hivemind_boinc_idle_worker.py lease nocturnal workunits (22:00-07:00 Paris time), "
            "verify that folding ran on the gated CUDA device (PR #408), pause automatically within 4s of gamer input (PR #399), "
            "and enforce the monotone validation-loss ratchet before assimilating QLoRA deltas (PR #366)."
        ),
        "recommended_for": "Volunteer gamers and citizen-science trainers contributing overnight GPU cycles.",
    },
)


RTX2070_T4_INTEGRATION_LEDGER: Sequence[dict[str, Any]] = (
    {
        "pr_number": 410,
        "branch": "feat/e8-stage2-tiny-moe-qlora",
        "title": "Signed task agents over AMCP & Kafka (Aglet life cycle, no mobile code), Stage 2 Tiny-MoE QLoRA & live churn soak",
        "components": [
            "src/hivemind/swarm/federation_agents.py",
            "scripts/e8_amcp_round.py",
            "scripts/e8_qlora_moe_round.py",
            "deploy/amcp/hm-wg-hub.sh",
            "docs/FEDERATION_V2_AMCP_AGLETS.md",
            "docs/E8_STAGE2_TINY_MOE_QLORA.md",
            "docs/OVERLAY_OPEN_SOURCE_TUNNELS.md",
        ],
        "artifact_refs": [
            "docs/E8_STAGE2_TINY_MOE_QLORA.md",
            "docs/FEDERATION_V2_AMCP_AGLETS.md",
        ],
        "measured_results": {
            "stage2_qlora_moe_loss": "4.1721 -> 1.7772 (4 rounds, NF4 + LoRA r=4, alpha=8, 0.0000 diff vs single-process baseline)",
            "stage2_negative_control_shuffled_loss": "4.1721 -> 4.2034 (stays at chance ln(64)=4.159)",
            "amcp_kafka_live_cluster_6_rounds": "0.0000 loss difference vs single-process baseline across 3 Spot VM Kafka brokers + T4",
            "amcp_kafka_70_round_churn_soak": "62/70 complete, 8/70 closed by quorum during SIGKILL outages, 0 rejections, loss 4.171 -> 1.448",
            "vram_peak_allocated_mib": 27.8,
            "vram_free_floor_mib": 4300,
        },
    },
    {
        "pr_number": 409,
        "branch": "feat/s36-honest-gpu-count-and-producer-autostart",
        "title": "Honest physical GPU counting (#402), cross-platform auto-start service & GPU outreach campaign",
        "components": [
            "src/hivemind/router/accounts.py",
            "src/hivemind/router/t4_visibility.py",
            "scripts/hivemind_producer_autostart.py",
            "scripts/hivemind_gpu_outreach_campaign.py",
        ],
        "artifact_refs": [
            "src/hivemind/router/accounts.py",
            "src/hivemind/router/t4_visibility.py",
        ],
        "measured_results": {
            "distinct_physical_gpus_metric": "Normalizes endpoint URLs by _endpoint_host_root; separates distinct_physical_gpus from shared_host_endpoints",
            "autostart_platforms": "Linux (systemd --user), Windows 10/11 (Scheduled Task), macOS (launchd)",
        },
    },
    {
        "pr_number": 408,
        "branch": "fix/idle-worker-truthful-gpu-probe",
        "title": "Idle GPU worker reports READY only when the fold ran on the gated CUDA device",
        "components": ["scripts/hivemind_boinc_idle_worker.py"],
        "artifact_refs": ["scripts/hivemind_boinc_idle_worker.py"],
        "measured_results": {
            "truthful_cuda_probe": "Refuses READY status if PyTorch/CUDA falls back to CPU or fails the device gate",
        },
    },
    {
        "pr_number": 407,
        "branch": "artifacts/e8-two-host-wan-results",
        "title": "First real two-host E8 cell between GCP Tesla T4 and Windows 11 RTX 2070 (XAVGAMSRV)",
        "components": ["artifacts/experiments/e8_real/RESULTS_T4_SIDE.md"],
        "artifact_refs": ["artifacts/experiments/e8_real/RESULTS_T4_SIDE.md"],
        "measured_results": {
            "hosts": "Learner 0: socreateai-agora-hermes-node1 (Tesla T4, Linux) | Learner 1: XAVGAMSRV (RTX 2070, Windows 11)",
            "paired_loss_diff_vs_baseline": "+4.8e-07 +/- 8.3e-07 across seeds 1-3 (K=2, 12 rounds, 400 inner steps, RDA merge)",
            "round_latency_s": "p50 = 1.20 s, p95 = 3.79 s, max = 3.89 s (excluding join round 0)",
        },
    },
    {
        "pr_number": 401,
        "branch": "docs/e8-two-host-real-test",
        "title": "Real two-host test T4 & RTX 2070 per Foundation Paper; pilot syncer refuses 0.0.0.0 wildcard bind",
        "components": [
            "docs/E8_TWO_HOST_REAL_TEST.md",
            "scripts/run_experiment_E8_pilot.py",
            "tests/test_e8_pilot_network_safety.py",
        ],
        "artifact_refs": ["docs/E8_TWO_HOST_REAL_TEST.md"],
        "measured_results": {
            "network_safety": "Syncer binds 127.0.0.1 by default and raises ValueError on 0.0.0.0 / :: wildcard addresses",
        },
    },
    {
        "pr_number": 399,
        "branch": "feat/volunteer-gamer-config",
        "title": "Fail-safe gamer idle-GPU helper, stock-BOINC gamer settings & honest join page wording",
        "components": ["scripts/hivemind_volunteer_config.py", "src/hivemind/router/join_page.py"],
        "artifact_refs": ["scripts/hivemind_volunteer_config.py"],
        "measured_results": {
            "gamer_preemption": "Unmeasurable input defaults to playing (pauses GPU); generates cc_config.xml + global_prefs_override.xml",
        },
    },
    {
        "pr_number": 366,
        "branch": "fix/boinc-qlora-ratchet-and-mixed-domain-moe",
        "title": "BOINC QLoRA monotone validation-loss ratchet & mixed-domain MoE dispatch",
        "components": ["src/hivemind/boinc/real_client/assimilator.py"],
        "artifact_refs": ["src/hivemind/boinc/real_client/assimilator.py"],
        "measured_results": {
            "monotone_ratchet": "Rejects any outer-step checkpoint where E_val(W^(t+1)) > E_val(W^(t))",
        },
    },
)


def build_opentrainer_training_round_table(
    ledger_dir: Path | None = None,
    prefix: str = "hmq_",
) -> list[dict[str, Any]]:
    """Build a structured table of `(round, domain, eval_by_domain_before, eval_by_domain_after)` from ledger JSON artifacts (Audit C8)."""
    if ledger_dir is None or not ledger_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for p in sorted(ledger_dir.glob(f"*{prefix}*.json")):
        if p.name.startswith("workgen_state") or p.name in ("latest_adapter.json", "latest_adapter_big.json"):
            continue
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        wu_name = str(rec.get("wu_name") or rec.get("name") or p.stem)
        rows.append(
            {
                "round": wu_name,
                "domain": rec.get("domain", "unknown"),
                "eval_by_domain_before": dict(rec.get("eval_by_domain_before") or {}),
                "eval_by_domain_after": dict(rec.get("eval_by_domain_after") or {}),
                "eval_loss_before": rec.get("eval_loss_before"),
                "eval_loss_after": rec.get("eval_loss_after"),
                "sha256": rec.get("sha256") or rec.get("adapter_sha256"),
            }
        )
    return rows


AGLET_AMCP_LIFECYCLE_SPEC: Sequence[dict[str, str]] = (
    {
        "aglet_verb": "dispatch",
        "amcp_event_type": "io.hivemind.amcp.fed.task_offer",
        "direction": "OpenTrainer Coordinator -> Trainers (AMCP/Kafka or HTTPS /opentrainer/heartbeat)",
        "security_invariant": "Carries run_id, round, code_pin SHA-256, base_sha256, global_sha256, hyperparameters, deadline, quorum — NEVER executable code.",
    },
    {
        "aglet_verb": "arrival (onArrival)",
        "amcp_event_type": "check_offer (local verification gate)",
        "direction": "Trainer Local Verification before GPU execution",
        "security_invariant": "Verifies CRLF-normalized code_pin matches local git checkout, verifies base_sha256 & global_sha256 with weights_only=True, checks night window & free VRAM floor (>= 4,300 MiB on T4 / >= 1,536 MiB on 8 GB RTX 2070).",
    },
    {
        "aglet_verb": "clone",
        "amcp_event_type": "io.hivemind.amcp.fed.task_claim",
        "direction": "Trainer -> OpenTrainer Coordinator",
        "security_invariant": "Multiple independent trainers (e.g. Learner 0 on Tesla T4, Learner 1 on RTX 2070) claim and run the same offer on disjoint non-IID data shards.",
    },
    {
        "aglet_verb": "result",
        "amcp_event_type": "io.hivemind.amcp.fed.task_result",
        "direction": "Trainer -> OpenTrainer Coordinator (/v1/hivemind/opentrainer/submit-delta or AMCP bus)",
        "security_invariant": "Sends only the weight delta; coordinator verifies run_id, round, allowed learner list, global_sha256, code_pin, tensor shapes, finiteness (no NaN/Inf) and L2 norm bound before token-weighted outer step.",
    },
    {
        "aglet_verb": "retract",
        "amcp_event_type": "io.hivemind.amcp.fed.task_abort",
        "direction": "OpenTrainer Coordinator -> Trainers",
        "security_invariant": "Withdraws and re-offers a round if quorum is not reached by deadline; late work from stale rounds is never carried over (HM-S9-01).",
    },
)


DEFAULT_ROUND_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "e8_qlora_moe_round.py"


def compute_default_code_pin(script_path: Path | None = None) -> str:
    """Return a deterministic SHA-256 code pin for the OpenTrainer Stage-2 QLoRA / DiLoCo round script (IP-176)."""
    candidate = script_path if script_path is not None else DEFAULT_ROUND_SCRIPT_PATH
    if candidate.is_file():
        return hashlib.sha256(candidate.read_bytes()).hexdigest()
    return hashlib.sha256(b"hivemind-opentrainer-e8-qlora-moe-v1-2026-11-01").hexdigest()


def default_active_task_offer(
    now_utc: float | None = None,
    *,
    global_sha256: str | None = None,
    round_index: int | None = None,
    quorum: int = 2,
    script_path: Path | None = None,
) -> dict[str, Any]:
    """Return the current active OpenTrainer Aglet ``task_offer`` descriptor for connecting trainers."""
    now = time.time() if now_utc is None else float(now_utc)
    round_idx = int(round_index) if round_index is not None else (int(now // 1800) % 10000)
    pin = compute_default_code_pin(script_path=script_path)
    base_sha = hashlib.sha256(b"granite-moe-tiny-nf4-pcg64-seed1").hexdigest()
    global_sha = str(global_sha256).strip().lower() if global_sha256 else base_sha
    return {
        "event_type": "io.hivemind.amcp.fed.task_offer",
        "topic": OPENTRAINER_TOPIC,
        "run_id": "opentrainer-rtx2070-t4-co-train-v1",
        "round": round_idx,
        "algorithm": "e8_qlora_moe_v1",
        "model_id": "hivemind/granite-moe-qlora-nf4",
        "script_sha256": pin,
        "base_sha256": base_sha,
        "global_sha256": global_sha,
        "inner_steps": 40,
        "inner_lr": 0.01,
        "outer_lr": 0.5,
        "outer_mu": 0.5,
        "qlora_rank": 4,
        "qlora_alpha": 8,
        "quorum": max(1, int(quorum)),
        "max_delta_l2": DEFAULT_MAX_DELTA_L2,
        "min_free_vram_mib_t4": DEFAULT_MIN_FREE_VRAM_MIB_T4,
        "min_free_vram_mib_8gb": DEFAULT_MIN_FREE_VRAM_MIB_8GB,
        "deadline_unix": round(now + 1800.0, 3),
        "mobile_code_allowed": False,
        "weights_only_deserialization": True,
    }


def validate_trainer_delta_submission(
    *,
    run_id: str,
    round_index: int,
    delta_l2: float,
    eval_loss: float,
    tokens_trained: int,
    vram_free_mib: int,
    vram_total_mib: int,
    max_delta_l2: float = DEFAULT_MAX_DELTA_L2,
) -> dict[str, Any]:
    """Validate a trainer's outbound HTTPS delta submission against OpenTrainer safety & VRAM gates."""
    if not run_id or not run_id.strip():
        raise ValueError("run_id must be non-empty")
    if round_index < 0:
        raise ValueError("round_index must be non-negative")
    if not math.isfinite(delta_l2) or delta_l2 < 0.0:
        raise ValueError("delta_l2 must be a finite non-negative number")
    if delta_l2 > max_delta_l2:
        raise ValueError(f"delta_l2 ({delta_l2:.4f}) exceeds maximum bound ({max_delta_l2:.4f})")
    if not math.isfinite(eval_loss) or eval_loss < 0.0:
        raise ValueError("eval_loss must be a finite non-negative number")
    if tokens_trained <= 0:
        raise ValueError("tokens_trained must be > 0")
    min_floor = DEFAULT_MIN_FREE_VRAM_MIB_T4 if vram_total_mib >= 14000 else DEFAULT_MIN_FREE_VRAM_MIB_8GB
    if (vram_total_mib > 0 or vram_free_mib > 0) and vram_free_mib < min_floor:
        raise ValueError(
            f"vram_free_mib ({vram_free_mib} MiB) is below the required safety floor ({min_floor} MiB) for a {vram_total_mib} MiB GPU"
        )
    return {
        "accepted": True,
        "run_id": run_id.strip(),
        "round_index": int(round_index),
        "delta_l2": round(float(delta_l2), 6),
        "eval_loss": round(float(eval_loss), 6),
        "tokens_trained": int(tokens_trained),
        "vram_floor_required_mib": min_floor,
        "vram_headroom_ok": True,
    }


_FED_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_WG_PUBKEY_RE = re.compile(r"^[A-Za-z0-9+/]{43}=$")


def validate_federation_v2_trainer_registration(
    *,
    learner_id: str,
    gpu_model: str,
    vram_free_mib: int,
    vram_total_mib: int = 0,
    night_only_consent: bool = False,
    shard_index: int = 0,
    run_id: str = "opentrainer-rtx2070-t4-co-train-v1",
    pinned_commit: str = "HEAD",
    network_mode: str = "outbound-https-relay",
    wireguard_public_key: str | None = None,
    wireguard_private_key: str | None = None,
    amcp_hmac_keys: str | None = None,
    overlay_slot_ip: str = "10.77.0.10/32",
) -> dict[str, Any]:
    """Validate a Federation v2 Trainer registration against docs/FEDERATION_V2_AMCP_AGLETS.md Section 9.

    Enforces:
      - Valid learner_id (`^[A-Za-z0-9._-]{1,64}$`) and run_id (`^[A-Za-z0-9._-]{1,64}$`)
      - Truthful non-empty gpu_model and vram_free_mib >= safety floor (4,300 MiB on T4 / >=14 GB GPUs; 1,536 MiB on 8 GB GPUs)
      - Explicit night_only_consent=True (22:00-07:00 Europe/Paris window)
      - Rejects any WireGuard private key or shared HIVEMIND_AMCP_HMAC_KEYS secret
      - Validates WireGuard public key format (`^[A-Za-z0-9+/]{43}=$`) when provided or when network_mode == 'wireguard-t4-hub'
      - Never promises points or HMT payment and never includes shared signing keys in the response
    """
    if wireguard_private_key is not None and str(wireguard_private_key).strip():
        raise ValueError(
            "wireguard_private_key must never be submitted or leaves the trainer machine; provide only wireguard_public_key"
        )
    if amcp_hmac_keys is not None and str(amcp_hmac_keys).strip():
        raise ValueError(
            "HIVEMIND_AMCP_HMAC_KEYS must never be submitted, stored, or echoed in web registration; keys are exchanged out of band"
        )
    lid = str(learner_id or "").strip()
    if not _FED_ID_RE.match(lid):
        raise ValueError("learner_id must match ^[A-Za-z0-9._-]{1,64}$")
    rid = str(run_id or "").strip()
    if not _FED_ID_RE.match(rid):
        raise ValueError("run_id must match ^[A-Za-z0-9._-]{1,64}$")
    if isinstance(shard_index, bool) or not isinstance(shard_index, int) or shard_index < 0:
        raise ValueError("shard_index must be a non-negative integer")
    gpu_clean = str(gpu_model or "").strip()
    if not gpu_clean or gpu_clean.lower() in ("unknown", "not reported", "none"):
        raise ValueError("gpu_model must be explicitly reported")
    if night_only_consent is not True:
        raise ValueError("night_only_consent must be True (22:00-07:00 Europe/Paris window required)")
    if network_mode not in NETWORK_MODES:
        raise ValueError(f"network_mode must be one of {NETWORK_MODES}, got {network_mode!r}")
    if isinstance(vram_free_mib, bool) or not isinstance(vram_free_mib, int):
        raise ValueError("vram_free_mib must be an integer")
    is_t4_class = int(vram_total_mib or 0) >= 14000 or "t4" in gpu_clean.lower()
    min_floor = DEFAULT_MIN_FREE_VRAM_MIB_T4 if is_t4_class else DEFAULT_MIN_FREE_VRAM_MIB_8GB
    if vram_free_mib < min_floor:
        raise ValueError(
            f"vram_free_mib ({vram_free_mib} MiB) is below the required safety floor ({min_floor} MiB) for {gpu_clean}"
        )

    wg_pub_clean: str | None = None
    operator_wg_cmd: str | None = None
    if wireguard_public_key is not None and str(wireguard_public_key).strip():
        wg_pub_clean = str(wireguard_public_key).strip()
        if not _WG_PUBKEY_RE.match(wg_pub_clean):
            raise ValueError("wireguard_public_key must be a 44-character base64 Curve25519 public key")
        if wg_pub_clean == "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=":
            raise ValueError("wireguard_public_key cannot be an all-zero key")
        operator_wg_cmd = f"hm-wg-hub add-member {overlay_slot_ip} {wg_pub_clean}"
    elif network_mode == "wireguard-t4-hub":
        raise ValueError("wireguard_public_key is required when network_mode is 'wireguard-t4-hub'")

    pin = compute_default_code_pin()
    commit_str = str(pinned_commit or "HEAD").strip() or "HEAD"
    return {
        "object": "hivemind.opentrainer.federation_v2_trainer_registration",
        "accepted": True,
        "learner_id": lid,
        "shard_index": int(shard_index),
        "run_id": rid,
        "pinned_commit": commit_str,
        "code_pin": pin,
        "gpu_model": gpu_clean,
        "vram_free_mib": int(vram_free_mib),
        "vram_total_mib": int(vram_total_mib or 0),
        "vram_floor_required_mib": min_floor,
        "night_only_consent": True,
        "night_window": "22:00-07:00 Europe/Paris",
        "network_mode": network_mode,
        "wireguard_public_key": wg_pub_clean,
        "operator_wg_hub_command": operator_wg_cmd,
        "operator_coordinator_flag": f"--learners {lid}",
        "cli_check_command": "python scripts/hivemind_amcp.py --dev-keyring check",
        "cli_join_command": (
            f"python scripts/e8_amcp_round.py learner --run-id {rid} "
            f"--learner-id {lid} --index {int(shard_index)} --quant nf4 --device cuda --night-only"
        ),
        "points_or_payment_promised": False,
        "points_policy": (
            "No points or HMT are earned for Federation v2 rounds: a delta is not verified work until "
            "at least five independent learners and a Byzantine filter are active."
        ),
        "shared_signing_key_included": False,
        "auto_starts_training": False,
    }


COMMUNITY_HELP_AND_CONTRIBUTIVE_GUIDE: Sequence[dict[str, Any]] = (
    {
        "topic_id": "producer-openrouter-join",
        "category": "producer_openrouter",
        "title_en": "1. Join as an OpenRouter Inference Producer (Share Idle GPU & Earn HMT)",
        "title_fr": "1. Rejoindre comme Producteur d'Inférence OpenRouter (Partage GPU & HMT)",
        "summary_en": (
            "Run Ollama (>= 0.1.34, CVE-2024-37032 fixed), llama-server, or vLLM behind hivemind_producer_shield.py "
            "(HIVEMIND_ENDPOINT_TOKEN; never expose raw port 11434) and heartbeat your models to the HiveMind OpenRouter "
            "gateway. Whenever a local game or heavy workload starts, your GPU yields in < 1.5s."
        ),
        "summary_fr": (
            "Lancez Ollama (>= 0.1.34, CVE-2024-37032 corrigée), llama-server ou vLLM derrière hivemind_producer_shield.py "
            "(HIVEMIND_ENDPOINT_TOKEN ; n'exposez jamais le port 11434 brut) et envoyez vos heartbeats vers la passerelle "
            "HiveMind OpenRouter. Dès qu'un jeu ou une tâche lourde démarre, votre GPU libère la VRAM en < 1,5 s."
        ),
        "cli_command": (
            "HIVEMIND_ENDPOINT_TOKEN=<token> python scripts/hivemind_producer_shield.py --listen 127.0.0.1:8788 --ollama http://127.0.0.1:11434 && "
            "python scripts/hivemind_producer_agent.py "
            "--gateway https://hivemind-openrouter-1003063861791.us-central1.run.app --preset coder"
        ),
        "verification_endpoint": "/v1/hivemind/producers",
    },
    {
        "topic_id": "trainer-opentrainer-moe",
        "category": "trainer_opentrainer_moe",
        "title_en": "2. Join as an OpenTrainer Self-Training MoE Node (0 Inbound Ports)",
        "title_fr": "2. Contribuer à l'Auto-Entraînement MoE via OpenTrainer (0 Port Entrant)",
        "summary_en": (
            "Participate in Stage-2 Tiny Granite-MoE NF4 QLoRA and DiLoCo 2D-DCT outer steps over outbound HTTPS "
            "(TCP/443) or signed AMCP/Kafka Aglets (`hivemind.amcp.federation.v1`). Monotone validation-loss ratchet "
            "ensures only loss-improving rounds advance the checkpoint."
        ),
        "summary_fr": (
            "Participez à l'auto-entraînement Stage-2 Tiny Granite-MoE NF4 QLoRA et aux pas externes DiLoCo 2D-DCT "
            "via HTTPS sortant (TCP/443) ou Aglets signés AMCP/Kafka (`hivemind.amcp.federation.v1`). Le cliquet "
            "monotone rejette toute mise à jour qui dégrade la loss de validation."
        ),
        "cli_command": (
            "python scripts/hivemind_opentrainer_agent.py "
            "--gateway https://hivemind-openrouter-1003063861791.us-central1.run.app "
            "--network-mode outbound-https-relay --check-firewall --submit-delta --once"
        ),
        "verification_endpoint": "/v1/hivemind/opentrainer/status",
    },
    {
        "topic_id": "dual-role-failover-orchestrator",
        "category": "dual_producer_trainer_3h_cycle",
        "title_en": "3. Dual Producer + Trainer Co-Serving & 3-Hour MoE Self-Training Cycle",
        "title_fr": "3. Orchestrateur Hybride Producteur + Entraîneur & Cycle MoE 3 Heures",
        "summary_en": (
            "Every 3 hours, the Producer+Trainer MoE integration orchestrator verifies live OpenRouter inference "
            "(`hivemind/coder`, `hivemind/bee-stars` with NIGHT-MOE-1 LoRA, `hivemind/auto`), refreshes producer "
            "and OpenTrainer heartbeats, and executes verified QLoRA self-training rounds."
        ),
        "summary_fr": (
            "Toutes les 3 heures, l'orchestrateur d'intégration Producteur+Entraîneur vérifie le service d'inférence "
            "OpenRouter (`hivemind/coder`, `hivemind/bee-stars` avec adaptateur NIGHT-MOE-1 LoRA, `hivemind/auto`), "
            "rafraîchit les heartbeats Producteur/OpenTrainer et exécute des rondes d'auto-entraînement QLoRA."
        ),
        "cli_command": (
            "PYTHONPATH=src:scripts python scripts/run_3h_producer_trainer_moe_integration.py"
        ),
        "verification_endpoint": "/v1/hivemind/contributive-hub",
    },
    {
        "topic_id": "community-troubleshooting-help",
        "category": "community_mutual_aid",
        "title_en": "4. Community Help, Firewall Diagnostics & Honest Adapter Rules",
        "title_fr": "4. Entraide Communautaire, Diagnostic Pare-Feu & Règles d'Intégrité",
        "summary_en": (
            "Behind NAT/CGNAT or Windows Firewall? Use `outbound-https-relay` (TCP/443 outbound only). "
            "Note: `hivemind/bee-stars` requires the `NIGHT-MOE-1` LoRA adapter on `llama-server` (`/bee`); "
            "plain Ollama nodes should advertise base/coder models without `bee-stars`."
        ),
        "summary_fr": (
            "Derrière un NAT/CGNAT ou pare-feu Windows ? Utilisez `outbound-https-relay` (TCP/443 sortant uniquement). "
            "Important : `hivemind/bee-stars` requiert l'adaptateur LoRA `NIGHT-MOE-1` sur `llama-server` (`/bee`) ; "
            "un nœud Ollama standard ne doit pas annoncer `bee-stars` sans l'adaptateur."
        ),
        "cli_command": (
            "curl -sS https://hivemind-openrouter-1003063861791.us-central1.run.app/v1/hivemind/contributive-hub"
        ),
        "verification_endpoint": "/v1/hivemind/contributive-hub",
    },
    {
        "topic_id": "byom-2-to-4-local-gpus",
        "category": "byom_multi_gpu_co_training",
        "title_en": "5. Co-Train New or Custom Models (BYOM) with 2 to 4 Local GPUs",
        "title_fr": "5. Co-Entraîner de Nouveaux Modèles ou Vos Propres Modèles (BYOM) avec 2 à 4 GPUs",
        "summary_en": (
            "Join as a Trainer (`role=\"trainer\"`) to co-train shared HiveMind MoE models OR your own custom base "
            "models/LoRA adapters (`--campaign-id byom-<your-model>`) across 2 to 4 local GPUs (same rig via "
            "`CUDA_VISIBLE_DEVICES=0..3` or 2–4 LAN/WAN machines). Each GPU takes a disjoint non-IID data shard "
            "(`--learner-index 0..3 --num-learners K`) and merges outer steps via DiLoCo 2D-DCT + Multi-Krum."
        ),
        "summary_fr": (
            "Rejoignez comme Entraîneur (`role=\"trainer\"`) pour co-entraîner de nouveaux modèles communs OU vos "
            "propres modèles/adaptateurs LoRA (`--campaign-id byom-<votre-modele>`) avec 2 à 4 GPUs locaux (même "
            "machine via `CUDA_VISIBLE_DEVICES=0..3` ou 2 à 4 PC en réseau). Chaque GPU entraîne une partition "
            "disjointe (`--learner-index 0..3 --num-learners K`) et synchronise par pas externe DiLoCo 2D-DCT."
        ),
        "cli_command": (
            "for gpu in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$gpu python scripts/hivemind_opentrainer_agent.py "
            "--gateway https://hivemind-openrouter-1003063861791.us-central1.run.app "
            "--host-label RIG-GPU$gpu --learner-index $gpu --campaign-id byom-custom-moe-v1 "
            "--network-mode outbound-https-relay --submit-delta --once & done; wait"
        ),
        "verification_endpoint": "/v1/hivemind/opentrainer/status",
    },
)


MULTI_GPU_2_TO_4_BYOM_CO_TRAINING_PLAYBOOK: dict[str, Any] = {
    "title_en": "Co-Training New & Custom (BYOM) Models with 2 to 4 Local GPUs on OpenTrainer",
    "title_fr": "Co-Entraîner de Nouveaux Modèles ou Vos Propres Modèles (BYOM) avec 2 à 4 GPUs Locaux",
    "supported_topologies": [
        {
            "topology_id": "single_host_2_to_4_gpus",
            "label_en": "Topology A — 1 Workstation / Homelab Server with 2 to 4 GPUs (CUDA_VISIBLE_DEVICES=0..3)",
            "label_fr": "Topologie A — 1 Station / Serveur Homelab avec 2 à 4 GPUs (CUDA_VISIBLE_DEVICES=0..3)",
            "how_it_works": (
                "Pin one OpenTrainer worker per physical GPU (`CUDA_VISIBLE_DEVICES=0..K-1`), assign each worker "
                "a distinct `--host-label RIG-GPU$i` and `--learner-index $i`, and run H=20..40 inner QLoRA steps "
                "before outer-step DiLoCo / RDA aggregation."
            ),
        },
        {
            "topology_id": "multi_host_lan_or_wan_2_to_4_gpus",
            "label_en": "Topology B — 2 to 4 Separate Gaming PCs / Laptops Across LAN or Internet (0 Inbound Ports)",
            "label_fr": "Topologie B — 2 à 4 PC Gamers / Machines Séparées en LAN ou Internet (0 Port Entrant)",
            "how_it_works": (
                "Each PC runs `scripts/hivemind_opentrainer_agent.py --network-mode outbound-https-relay` (or local "
                "syncer `scripts/run_experiment_E8_pilot.py --role syncer --bind 127.0.0.1` over SSH loopback tunnel) "
                "and submits only compact FP16/2D-DCT LoRA deltas (~6–18 MB per round, never raw private datasets)."
            ),
        },
        {
            "topology_id": "bring_your_own_model_byom",
            "label_en": "Topology C — Bring Your Own Model (BYOM) / Custom Domain LoRA Campaign",
            "label_fr": "Topologie C — Entraîner Votre Propre Modèle (BYOM) / Campagne LoRA Personnalisée",
            "how_it_works": (
                "Specify a custom `--campaign-id byom-<your-project>` when submitting deltas to "
                "`/v1/hivemind/opentrainer/submit-delta`, or launch a private 2–4 GPU DiLoCo coordinator locally while "
                "optionally publishing the trained GGUF/LoRA bee onto HiveMind OpenRouter (`/v1/chat/completions`)."
            ),
        },
    ],
    "vram_budget_per_gpu_tier": {
        "6gb_to_8gb_gpus_e_g_rtx2070_3060_4060": "NF4 4-bit base + rank-4/rank-8 QLoRA (batch=1, grad_accum=8, seq_len=1024, >=1,536 MiB free VRAM safety floor)",
        "12gb_to_16gb_gpus_e_g_rtx3060_12g_4070ti_t4": "NF4/FP8 base + rank-16 QLoRA or 8-expert Tiny-MoE (batch=2, grad_accum=8, seq_len=2048, >=4,300 MiB free VRAM floor on T4)",
        "24gb_plus_gpus_e_g_rtx3090_4090_a5000": "14B–30B MoE expert shards + rank-32 QLoRA / teacher distillation",
    },
}


def summarize_moe_self_training_progress(
    *,
    recent_rounds: Sequence[Mapping[str, Any]],
    opentrainer_nodes: Sequence[Mapping[str, Any]],
    now_utc: float | None = None,
) -> dict[str, Any]:
    """Compute real, non-fabricated MoE self-training trajectory metrics from recorded OpenTrainer rounds."""
    now = time.time() if now_utc is None else float(now_utc)
    live_nodes = [n for n in opentrainer_nodes if (now - float(n.get("last_heartbeat_utc") or 0.0)) <= 300.0]
    ordered = sorted(
        recent_rounds,
        key=lambda r: (float(r.get("submitted_utc") or 0.0), int(r.get("round_index") or 0)),
    )
    eval_losses = [
        float(r["eval_loss"])
        for r in ordered
        if r.get("eval_loss") is not None and math.isfinite(float(r["eval_loss"]))
    ]
    total_tokens = sum(max(0, int(r.get("tokens_trained") or 0)) for r in ordered)
    campaigns = sorted({str(r.get("run_id") or "") for r in ordered if str(r.get("run_id") or "").strip()})
    handles = sorted({str(r.get("handle") or "") for r in ordered if str(r.get("handle") or "").strip()})
    latest_round = int(ordered[-1].get("round_index") or 0) if ordered else 0

    if eval_losses:
        initial_loss = round(eval_losses[0], 6)
        latest_loss = round(eval_losses[-1], 6)
        best_loss = round(min(eval_losses), 6)
        loss_delta = round(initial_loss - latest_loss, 6)
        ratchet_improved = latest_loss <= initial_loss
        data_source = "LIVE_RECORDED_OPENTRAINER_ROUNDS"
    else:
        initial_loss = None
        latest_loss = None
        best_loss = None
        loss_delta = 0.0
        ratchet_improved = False
        data_source = "NO_ROUNDS_RECORDED_YET"

    return {
        "object": "hivemind.opentrainer.moe_self_training_progress",
        "target_model": "hivemind/granite-moe-qlora-nf4",
        "architecture": "Stage-2 Tiny Granite-MoE NF4 QLoRA + DiLoCo 2D-DCT Outer Step + Monotone Loss Ratchet",
        "cadence": "every_3_hours_and_continuous_45s_micro_rounds",
        "data_source": data_source,
        "rounds_recorded": len(ordered),
        "latest_round_index": latest_round,
        "initial_eval_loss": initial_loss,
        "latest_eval_loss": latest_loss,
        "best_ratcheted_eval_loss": best_loss,
        "loss_improvement_delta": loss_delta,
        "monotone_ratchet_improved": ratchet_improved,
        "total_tokens_trained": total_tokens,
        "active_campaigns": campaigns,
        "participating_handles": handles,
        "active_trainer_nodes_count": len(live_nodes),
        "total_registered_trainer_nodes": len(opentrainer_nodes),
    }


def build_opentrainer_status_payload(
    *,
    trainer_accounts: Sequence[Mapping[str, Any]],
    opentrainer_nodes: Sequence[Mapping[str, Any]],
    recent_rounds: Sequence[Mapping[str, Any]],
    now_utc: float | None = None,
) -> dict[str, Any]:
    """Build the comprehensive ``/v1/hivemind/opentrainer/status`` payload."""
    now = time.time() if now_utc is None else float(now_utc)
    live_nodes = [n for n in opentrainer_nodes if (now - float(n.get("last_heartbeat_utc") or 0.0)) <= 300.0]
    distinct_hosts = {
        str(n.get("account_id") or n.get("host_label") or n.get("node_id") or "")
        for n in live_nodes
        if str(n.get("account_id") or n.get("host_label") or n.get("node_id") or "")
    }
    moe_progress = summarize_moe_self_training_progress(
        recent_rounds=recent_rounds,
        opentrainer_nodes=opentrainer_nodes,
        now_utc=now,
    )
    return {
        "object": "hivemind.opentrainer.status",
        "component": "OpenTrainer",
        "version": OPENTRAINER_VERSION,
        # The state is derived from the heartbeats seen in the last 300 s, never a literal: the coordinator answering
        # this request says nothing about trainers being connected.
        "status": "TRAINERS_LIVE" if live_nodes else "COORDINATOR_UP_NO_TRAINERS",
        "status_basis": f"{len(live_nodes)} node heartbeat(s) within 300 s of {round(now, 3)}",
        "updated_utc": round(now, 3),
        "portal_category": {
            "role": "trainer",
            "display_name": "Trainer (Intense GPU Training — OpenTrainer)",
            "signup_url_path": "/signup?role=trainer",
            "zero_private_user_prompts_on_trainer": True,
            "description": (
                "Provides dedicated or nocturnal GPU compute for intense distributed training (Stage-2 Tiny-MoE NF4 QLoRA, "
                "DiLoCo 2D-DCT outer steps, signed AMCP/Kafka Aglet rounds, and BOINC workunits) via OpenTrainer with "
                "zero-port-forwarding firewall traversal. Zero private user prompts ever touch a Trainer GPU (Folding@home model)."
            ),
            "registered_trainer_accounts_count": len(trainer_accounts),
            "registered_trainer_accounts": list(trainer_accounts),
        },
        "live_opentrainer_fleet": {
            "active_trainer_nodes": len(live_nodes),
            "distinct_physical_trainer_gpus": len(distinct_hosts),
            # IP-211: host labels are self-declared; the count is bounded by accounts x MAX_OPENTRAINER_NODES_PER_ACCOUNT, not verified GPUs.
            "distinct_physical_trainer_gpus_basis": "self-declared host_label, at most 8 per account (not verified hardware)",
            "total_registered_nodes": len(opentrainer_nodes),
            "nodes": list(opentrainer_nodes),
        },
        "moe_self_training_progress": moe_progress,
        "community_help_guide": list(COMMUNITY_HELP_AND_CONTRIBUTIVE_GUIDE),
        "multi_gpu_2_to_4_byom_playbook": MULTI_GPU_2_TO_4_BYOM_CO_TRAINING_PLAYBOOK,
        "active_task_offer": default_active_task_offer(now_utc=now),
        "recent_training_rounds": list(recent_rounds),
        "firewall_traversal_modes": list(FIREWALL_TRAVERSAL_MODES),
        "aglet_amcp_kafka_lifecycle": list(AGLET_AMCP_LIFECYCLE_SPEC),
        "rtx2070_t4_integration_ledger": list(RTX2070_T4_INTEGRATION_LEDGER),
    }
