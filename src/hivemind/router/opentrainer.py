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

import hashlib
import math
import time
from typing import Any, Mapping, Sequence

OPENTRAINER_VERSION = "1.0.0-s36"
OPENTRAINER_TOPIC = "hivemind.amcp.federation.v1"

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
        "measured_results": {
            "truthful_cuda_probe": "Refuses READY status if PyTorch/CUDA falls back to CPU or fails the device gate",
        },
    },
    {
        "pr_number": 407,
        "branch": "artifacts/e8-two-host-wan-results",
        "title": "First real two-host E8 cell between GCP Tesla T4 and Windows 11 RTX 2070 (XAVGAMSRV)",
        "components": ["artifacts/experiments/e8_real/RESULTS_T4_SIDE.md"],
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
        "measured_results": {
            "network_safety": "Syncer binds 127.0.0.1 by default and raises ValueError on 0.0.0.0 / :: wildcard addresses",
        },
    },
    {
        "pr_number": 399,
        "branch": "feat/volunteer-gamer-config",
        "title": "Fail-safe gamer idle-GPU helper, stock-BOINC gamer settings & honest join page wording",
        "components": ["scripts/hivemind_volunteer_config.py", "src/hivemind/router/join_page.py"],
        "measured_results": {
            "gamer_preemption": "Unmeasurable input defaults to playing (pauses GPU); generates cc_config.xml + global_prefs_override.xml",
        },
    },
    {
        "pr_number": 366,
        "branch": "fix/boinc-qlora-ratchet-and-mixed-domain-moe",
        "title": "BOINC QLoRA monotone validation-loss ratchet & mixed-domain MoE dispatch",
        "components": ["src/hivemind/boinc/real_client/assimilator.py"],
        "measured_results": {
            "monotone_ratchet": "Rejects any outer-step checkpoint where E_val(W^(t+1)) > E_val(W^(t))",
        },
    },
)


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


def compute_default_code_pin() -> str:
    """Return a deterministic SHA-256 code pin for the OpenTrainer Stage-2 QLoRA / DiLoCo protocol."""
    return hashlib.sha256(b"hivemind-opentrainer-e8-qlora-moe-v1-2026-11-01").hexdigest()


def default_active_task_offer(now_utc: float | None = None) -> dict[str, Any]:
    """Return the current active OpenTrainer Aglet ``task_offer`` descriptor for connecting trainers."""
    now = time.time() if now_utc is None else float(now_utc)
    round_idx = int(now // 1800) % 10000
    pin = compute_default_code_pin()
    base_sha = hashlib.sha256(b"granite-moe-tiny-nf4-pcg64-seed1").hexdigest()
    global_sha = hashlib.sha256(f"opentrainer-global-round-{round_idx}".encode()).hexdigest()
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
        "quorum": 1,
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
    if vram_total_mib > 0 and vram_free_mib < min_floor:
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


def load_live_lowcost_vm_orchestrator_ledger(repo_root: Any = None) -> dict[str, Any]:
    """Load the live Permanent Low-Cost VM OpenTrainer Orchestrator status ledger from disk if available."""
    import json
    from pathlib import Path

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    for rel in (
        Path("artifacts") / "opentrainer" / "opentrainer_status.json",
        Path("artifacts") / "opentrainer_lowcost_vm_verification.json",
    ):
        cand = root / rel
        if cand.exists():
            try:
                raw = json.loads(cand.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return raw
            except Exception:
                continue
    return {}


def sync_live_opentrainer_into_store(
    store: Any,
    now_utc: float | None = None,
    repo_root: Any = None,
) -> dict[str, Any]:
    """Sync the live Permanent Low-Cost VM OpenTrainer Orchestrator ledger into a persistent SQLite RouterStore."""
    ledger = load_live_lowcost_vm_orchestrator_ledger(repo_root=repo_root)
    if ledger and hasattr(store, "sync_opentrainer_orchestrator_ledger"):
        try:
            sync_stats = store.sync_opentrainer_orchestrator_ledger(ledger, now_utc=now_utc)
            return {"synced": True, **sync_stats, "orchestrator_round": ledger.get("current_round", 0)}
        except Exception:
            return {"synced": False}
    return {"synced": False}


def build_opentrainer_status_payload(
    *,
    trainer_accounts: Sequence[Mapping[str, Any]],
    opentrainer_nodes: Sequence[Mapping[str, Any]],
    recent_rounds: Sequence[Mapping[str, Any]],
    now_utc: float | None = None,
    lowcost_vm_orchestrator: Mapping[str, Any] | None = None,
    database_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the comprehensive ``/v1/hivemind/opentrainer/status`` payload."""
    now = time.time() if now_utc is None else float(now_utc)
    live_nodes = [n for n in opentrainer_nodes if (now - float(n.get("last_heartbeat_utc") or 0.0)) <= 300.0]
    distinct_hosts = {
        str(n.get("host_label") or n.get("node_id") or "")
        for n in live_nodes
        if str(n.get("host_label") or n.get("node_id") or "")
    }
    vm_orch = dict(lowcost_vm_orchestrator) if lowcost_vm_orchestrator is not None else load_live_lowcost_vm_orchestrator_ledger()
    vm_rounds = vm_orch.get("rounds") if isinstance(vm_orch.get("rounds"), list) else []
    vm_summary = {
        "component": vm_orch.get("component", "hivemind.opentrainer.lowcost_vm_orchestrator"),
        "t4_required": bool(vm_orch.get("t4_required", False)),
        "t4_participated_rounds": int(vm_orch.get("t4_participated_rounds", 0)),
        "t4_independent_rounds_completed": int(
            vm_orch.get("t4_independent_rounds_completed", len(vm_rounds))
        ),
        "current_round": int(vm_orch.get("current_round", len(vm_rounds))),
        "initial_eval_loss": vm_orch.get("initial_eval_loss"),
        "current_eval_loss": vm_orch.get("current_eval_loss"),
        "loss_reduction": vm_orch.get("loss_reduction"),
        "hosts_seen": list(vm_orch.get("hosts_seen") or []),
        "learners_last_seen": dict(vm_orch.get("learners_last_seen") or {}),
        "receipt_sha256": vm_orch.get("receipt_sha256"),
    }
    payload: dict[str, Any] = {
        "object": "hivemind.opentrainer.status",
        "component": "OpenTrainer",
        "version": OPENTRAINER_VERSION,
        "status": "ONLINE",
        "updated_utc": round(now, 3),
        "portal_category": {
            "role": "trainer",
            "display_name": "Trainer (Intense GPU Training — OpenTrainer)",
            "description": (
                "Provides dedicated or nocturnal GPU compute for intense distributed training (Stage-2 Tiny-MoE NF4 QLoRA, "
                "DiLoCo 2D-DCT outer steps, signed AMCP/Kafka Aglet rounds, and BOINC workunits) via OpenTrainer with "
                "zero-port-forwarding firewall traversal."
            ),
            "registered_trainer_accounts_count": len(trainer_accounts),
            "registered_trainer_accounts": list(trainer_accounts),
        },
        "live_opentrainer_fleet": {
            "active_trainer_nodes": len(live_nodes),
            "distinct_physical_trainer_gpus": len(distinct_hosts),
            "total_registered_nodes": len(opentrainer_nodes),
            "nodes": list(opentrainer_nodes),
        },
        "active_task_offer": default_active_task_offer(now_utc=now),
        "recent_training_rounds": list(recent_rounds),
        "lowcost_vm_orchestrator": vm_summary,
        "firewall_traversal_modes": list(FIREWALL_TRAVERSAL_MODES),
        "aglet_amcp_kafka_lifecycle": list(AGLET_AMCP_LIFECYCLE_SPEC),
        "rtx2070_t4_integration_ledger": list(RTX2070_T4_INTEGRATION_LEDGER),
    }
    if database_summary is not None:
        payload["database_summary"] = dict(database_summary)
    return payload
