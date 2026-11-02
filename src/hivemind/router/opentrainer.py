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


COMMUNITY_HELP_AND_CONTRIBUTIVE_GUIDE: Sequence[dict[str, Any]] = (
    {
        "topic_id": "producer-openrouter-join",
        "category": "producer_openrouter",
        "title_en": "1. Join as an OpenRouter Inference Producer (Share Idle GPU & Earn HMT)",
        "title_fr": "1. Rejoindre comme Producteur d'Inférence OpenRouter (Partage GPU & HMT)",
        "summary_en": (
            "Run Ollama, llama-server, or vLLM locally and heartbeat your models to the HiveMind OpenRouter gateway. "
            "Whenever a local game or heavy workload starts, your GPU yields in < 1.5s."
        ),
        "summary_fr": (
            "Lancez Ollama, llama-server ou vLLM localement et envoyez vos heartbeats vers la passerelle HiveMind OpenRouter. "
            "Dès qu'un jeu ou une tâche lourde démarre, votre GPU libère la VRAM en < 1,5 s."
        ),
        "cli_command": (
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


OPENROUTER_SMALL_INFRA_SPEC: dict[str, Any] = {
    "infrastructure_id": "openrouter-small-infra-v1",
    "title_en": "HiveMind OpenRouter Small Infrastructure — RTX 2070 Night-Training Contributor & Federated HiveMinders",
    "title_fr": "Petite Infrastructure OpenRouter HiveMind — Contributeur RTX 2070 d'Entraînement Nocturne & HiveMinders Fédérés",
    "gateway_component": {
        "service": "hivemind-openrouter",
        "cpu": "1 vCPU",
        "memory": "1 GiB",
        "persistence": "Litestream SQLite Single-Writer (RouterStore)",
        "public_url": "https://hivemind-openrouter-1003063861791.us-central1.run.app",
        "local_portal_url": "http://127.0.0.1:8095",
    },
    "orchestrator_component": {
        "service": "hm-opentrainer-1-cpu-sentinel",
        "bind": "127.0.0.1:8096",
        "vm_tier": "Permanent Low-Cost VM / e2-small CPU Sentinel",
        "t4_required": False,
        "role": "24/7 Quorum Sentinel, Non-IID Shard Dispatcher & DiLoCo/Multi-Krum Coordinator",
    },
    "rtx_primary_trainer": {
        "handle": "rtx2070-xavgamsrv",
        "host_label": "XAVGAMSRV-RTX2070",
        "gpu_name": "NVIDIA GeForce RTX 2070 (8 GB NF4 QLoRA)",
        "vram_total_mib": 8192,
        "vram_safety_floor_mib": DEFAULT_MIN_FREE_VRAM_MIB_8GB,
        "wddm_ceiling_mib": 6656,
        "role": "trainer",
        "badge": "🏆 Founding Pioneer Trainer #1",
    },
    "night_window_policy": {
        "deep_night_window_cet": "22:00-07:00 Europe/Paris (80-160 QLoRA inner steps + SwaT Sleep Consolidation)",
        "continuous_micro_round_cadence_s": 45,
        "evolution_sync_cadence_hours": 2,
        "moe_experimentator_cadence_hours": 4,
        "federation_protocol": "DiLoCo 2D-DCT Outer Step + Multi-Krum Byzantine Resilience + Monotone Held-Out Loss Ratchet",
    },
}

NIGHT_TRAINING_DOMAIN_SHARDS: tuple[dict[str, str], ...] = (
    {
        "shard_id": "shard-0-code-rust-python",
        "domain": "Code & Systems Engineering (Python / Rust / FastAPI)",
        "target_expert": "hivemind/coder",
    },
    {
        "shard_id": "shard-1-math-lean4-sympy",
        "domain": "Formal Verification & Symbolic Math (Lean 4 / SymPy)",
        "target_expert": "hivemind/bee-lean-prover",
    },
    {
        "shard_id": "shard-2-science-astro-pde",
        "domain": "Astrophysics BLS & Sundials PDE/ODE Simulation",
        "target_expert": "hivemind/bee-rusty-sundials",
    },
    {
        "shard_id": "shard-3-multilingual-i18n",
        "domain": "5-Language Citizen AI & SwaT Sleep Consolidation (EN/FR/ES/DE/ZH)",
        "target_expert": "hivemind/moe-general",
    },
)


def build_night_training_federation_plan(
    *,
    trainer_accounts: Sequence[Mapping[str, Any]],
    opentrainer_nodes: Sequence[Mapping[str, Any]],
    recent_rounds: Sequence[Mapping[str, Any]],
    hiveminder_accounts: Sequence[Mapping[str, Any]] = (),
    lowcost_vm_orchestrator: Mapping[str, Any] | None = None,
    now_utc: float | None = None,
) -> dict[str, Any]:
    """Build the OpenRouter Small-Infrastructure Night-Training Federation plan uniting the RTX 2070 and peer HiveMinders."""
    now = time.time() if now_utc is None else float(now_utc)
    vm_orch = dict(lowcost_vm_orchestrator) if lowcost_vm_orchestrator is not None else load_live_lowcost_vm_orchestrator_ledger()

    nodes_by_handle: dict[str, dict[str, Any]] = {}
    for n in opentrainer_nodes:
        h = str(n.get("handle") or "").strip().lower()
        if not h:
            continue
        existing = nodes_by_handle.get(h)
        if existing is None or float(n.get("last_heartbeat_utc") or 0.0) >= float(existing.get("last_heartbeat_utc") or 0.0):
            nodes_by_handle[h] = dict(n)

    # Collect all federated participants: RTX 2070 primary night contributor + OpenTrainer nodes + Trainer accounts + Producer/Prosumer HiveMinders
    ordered_handles: list[str] = []
    handle_meta: dict[str, dict[str, Any]] = {}

    def _register_candidate(
        handle_raw: str,
        *,
        role: str,
        host_label: str,
        gpu_name: str,
        vram_total_mib: int,
        vram_free_mib: int,
        network_mode: str,
        last_heartbeat_utc: float,
        contributor_kind: str,
    ) -> None:
        clean = handle_raw.strip()
        if not clean:
            return
        key = clean.lower()
        if key not in ordered_handles:
            ordered_handles.append(key)
            handle_meta[key] = {
                "handle": clean,
                "role": role,
                "host_label": host_label,
                "gpu_name": gpu_name,
                "vram_total_mib": int(vram_total_mib),
                "vram_free_mib": int(vram_free_mib),
                "network_mode": network_mode,
                "last_heartbeat_utc": float(last_heartbeat_utc),
                "contributor_kind": contributor_kind,
            }
        else:
            cur = handle_meta[key]
            if float(last_heartbeat_utc) > float(cur["last_heartbeat_utc"]):
                cur["last_heartbeat_utc"] = float(last_heartbeat_utc)
            if vram_total_mib > cur["vram_total_mib"]:
                cur["vram_total_mib"] = int(vram_total_mib)
                cur["vram_free_mib"] = int(vram_free_mib)
                cur["gpu_name"] = gpu_name

    # 1. Existing OpenTrainer nodes (putting RTX 2070 first if present)
    sorted_ot_nodes = sorted(
        opentrainer_nodes,
        key=lambda x: (
            0 if ("xavgamsrv" in str(x.get("handle", "")).lower() or "xavgamsrv" in str(x.get("host_label", "")).lower()) else (1 if "rtx" in str(x.get("handle", "")).lower() else 2),
            -float(x.get("last_heartbeat_utc") or 0.0),
        ),
    )
    for n in sorted_ot_nodes:
        h = str(n.get("handle") or "").strip()
        host_l = str(n.get("host_label") or h)
        gpu_n = str(n.get("gpu_name") or "NVIDIA GeForce RTX 2070")
        is_rtx = "rtx" in h.lower() or "rtx" in gpu_n.lower() or "xavgamsrv" in host_l.lower()
        is_sentinel = "sentinel" in h.lower() or "sentinel" in host_l.lower()
        c_kind = (
            "rtx_primary_night_trainer"
            if is_rtx
            else ("openrouter_small_infra_sentinel" if is_sentinel else "federated_hiveminder_trainer")
        )
        _register_candidate(
            h,
            role="trainer",
            host_label=host_l,
            gpu_name=gpu_n,
            vram_total_mib=int(n.get("vram_total_mib") or 8192),
            vram_free_mib=int(n.get("vram_free_mib") or 6400),
            network_mode=str(n.get("network_mode") or "outbound-https-relay"),
            last_heartbeat_utc=float(n.get("last_heartbeat_utc") or 0.0),
            contributor_kind=c_kind,
        )

    # 2. Registered trainer accounts not yet in opentrainer_nodes
    for t_acc in trainer_accounts:
        h = str(t_acc.get("handle") or "").strip()
        is_rtx = "rtx" in h.lower() or "xavgamsrv" in h.lower()
        _register_candidate(
            h,
            role=str(t_acc.get("role") or "trainer"),
            host_label="XAVGAMSRV-RTX2070" if is_rtx else f"{h}-node",
            gpu_name="NVIDIA GeForce RTX 2070 (8 GB NF4 QLoRA)" if is_rtx else "Federated HiveMinder GPU",
            vram_total_mib=8192,
            vram_free_mib=6400,
            network_mode="outbound-https-relay",
            last_heartbeat_utc=float(t_acc.get("created_utc") or 0.0),
            contributor_kind="rtx_primary_night_trainer" if is_rtx else "federated_hiveminder_trainer",
        )

    # 3. Registered Producer / Prosumer HiveMinders from OpenRouter small infrastructure
    for hm in hiveminder_accounts:
        h = str(hm.get("handle") or "").strip()
        role = str(hm.get("role") or "prosumer")
        if role not in ("trainer", "prosumer", "producer"):
            continue
        _register_candidate(
            h,
            role=role,
            host_label=f"{h}-openrouter-producer",
            gpu_name=f"Federated OpenRouter {role.capitalize()} GPU ({hm.get('execution_class') or 'cuda'})",
            vram_total_mib=8192,
            vram_free_mib=6400,
            network_mode="outbound-https-relay",
            last_heartbeat_utc=float(hm.get("last_heartbeat_utc") or 0.0),
            contributor_kind="federated_openrouter_hiveminder",
        )

    federated_roster: list[dict[str, Any]] = []
    num_participants = max(1, len(ordered_handles))
    for idx, key in enumerate(ordered_handles):
        meta = handle_meta[key]
        shard = NIGHT_TRAINING_DOMAIN_SHARDS[idx % len(NIGHT_TRAINING_DOMAIN_SHARDS)]
        vram_tot = int(meta["vram_total_mib"])
        if vram_tot <= 4096 or "sentinel" in meta["contributor_kind"]:
            qlora_profile = "CPU Sentinel / Rank-4 Aglet Quorum Validator"
        elif vram_tot <= 10240:
            qlora_profile = "NF4 4-bit Base + QLoRA (rank=4, alpha=8, batch=1, seq=1024, >=1536 MiB free floor)"
        elif vram_tot <= 16384:
            qlora_profile = "NF4/FP8 Base + QLoRA (rank=16, alpha=32, batch=2, seq=2048, >=4300 MiB free floor)"
        else:
            qlora_profile = "Full 14B-30B MoE Expert Shard + Rank-32 QLoRA / Teacher Distillation"
        hb_age_s = round(max(0.0, now - float(meta["last_heartbeat_utc"])), 3) if meta["last_heartbeat_utc"] > 0 else None
        hb_fresh = hb_age_s is not None and hb_age_s <= 300.0
        federated_roster.append(
            {
                "learner_id": f"l{idx}",
                "learner_index": idx,
                "total_learners": num_participants,
                "handle": meta["handle"],
                "role": meta["role"],
                "contributor_kind": meta["contributor_kind"],
                "host_label": meta["host_label"],
                "gpu_name": meta["gpu_name"],
                "vram_total_mib": vram_tot,
                "vram_free_mib": int(meta["vram_free_mib"]),
                "network_mode": meta["network_mode"],
                "shard_index": idx % len(NIGHT_TRAINING_DOMAIN_SHARDS),
                "shard_id": shard["shard_id"],
                "domain_focus": shard["domain"],
                "target_expert": shard["target_expert"],
                "qlora_profile": qlora_profile,
                "heartbeat_fresh": hb_fresh,
                "heartbeat_age_s": hb_age_s,
                "last_heartbeat_utc": round(float(meta["last_heartbeat_utc"]), 3),
            }
        )

    live_federated = [m for m in federated_roster if m["heartbeat_fresh"]]
    rtx_entry = next(
        (m for m in federated_roster if m["contributor_kind"] == "rtx_primary_night_trainer"),
        federated_roster[0] if federated_roster else None,
    )
    peer_minders = [
        m for m in federated_roster if rtx_entry is None or m["handle"].lower() != rtx_entry["handle"].lower()
    ]

    moe_prog = summarize_moe_self_training_progress(
        recent_rounds=recent_rounds,
        opentrainer_nodes=opentrainer_nodes,
        now_utc=now,
    )
    n_live = len(live_federated)
    # Byzantine tolerance f for Multi-Krum: 2f + 2 < n  =>  f = max(0, (n - 3) // 2)
    byzantine_f = max(0, (len(federated_roster) - 3) // 2)

    return {
        "object": "hivemind.opentrainer.night_training_federation",
        "infrastructure": OPENROUTER_SMALL_INFRA_SPEC,
        "status": "FEDERATED_NIGHT_SWARM_LIVE" if live_federated else "AWAITING_TRAINER_HEARTBEATS",
        "updated_utc": round(now, 3),
        "rtx_lead_trainer": rtx_entry,
        "rtx_integrated_as_night_trainer": rtx_entry is not None,
        "federated_peer_minders_count": len(peer_minders),
        "total_federated_trainers": len(federated_roster),
        "live_heartbeat_trainers_count": n_live,
        "quorum_satisfied": n_live >= 1 or int(vm_orch.get("current_round") or 0) > 0,
        "multi_krum_byzantine_tolerance_f": byzantine_f,
        "orchestrator_independent_rounds": int(vm_orch.get("current_round") or 0),
        "moe_self_training_progress": moe_prog,
        "domain_shards": list(NIGHT_TRAINING_DOMAIN_SHARDS),
        "federated_roster": federated_roster,
        "federate_cli_commands": {
            "rtx_lead_night_trainer": (
                "python scripts/hivemind_opentrainer_agent.py "
                "--gateway https://hivemind-openrouter-1003063861791.us-central1.run.app "
                '--host-label XAVGAMSRV-RTX2070 --gpu-name "NVIDIA GeForce RTX 2070 (8 GB NF4 QLoRA)" '
                "--join-night-federation --federate-night-round --once"
            ),
            "peer_hiveminder_join": (
                "python scripts/hivemind_opentrainer_agent.py "
                "--gateway https://hivemind-openrouter-1003063861791.us-central1.run.app "
                "--host-label <YOUR_RIG_NAME> --join-night-federation --submit-delta --once"
            ),
        },
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
        # Derived from live heartbeats, never a literal "ONLINE": "TRAINERS_LIVE" only when at least one trainer sent a
        # heartbeat within the 300 s window; otherwise the coordinator HTTP route is up with no trainers attached.
        "status": "TRAINERS_LIVE" if live_nodes else "COORDINATOR_UP_NO_TRAINERS",
        "status_basis": f"{len(live_nodes)} node heartbeat(s) within 300 s of {round(now, 3)}",
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
        "moe_self_training_progress": summarize_moe_self_training_progress(
            recent_rounds=recent_rounds,
            opentrainer_nodes=opentrainer_nodes,
            now_utc=now,
        ),
        "community_help_guide": list(COMMUNITY_HELP_AND_CONTRIBUTIVE_GUIDE),
        "multi_gpu_2_to_4_byom_playbook": MULTI_GPU_2_TO_4_BYOM_CO_TRAINING_PLAYBOOK,
        "active_task_offer": default_active_task_offer(now_utc=now),
        "recent_training_rounds": list(recent_rounds),
        "lowcost_vm_orchestrator": vm_summary,
        "openrouter_small_infra_night_federation": build_night_training_federation_plan(
            trainer_accounts=trainer_accounts,
            opentrainer_nodes=opentrainer_nodes,
            recent_rounds=recent_rounds,
            lowcost_vm_orchestrator=vm_orch,
            now_utc=now,
        ),
        "firewall_traversal_modes": list(FIREWALL_TRAVERSAL_MODES),
        "aglet_amcp_kafka_lifecycle": list(AGLET_AMCP_LIFECYCLE_SPEC),
        "rtx2070_t4_integration_ledger": list(RTX2070_T4_INTEGRATION_LEDGER),
    }
    if database_summary is not None:
        payload["database_summary"] = dict(database_summary)
    return payload
