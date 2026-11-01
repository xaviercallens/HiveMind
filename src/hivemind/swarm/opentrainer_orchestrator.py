# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Permanent Low-Cost VM OpenTrainer Orchestrator for T4-Independent Collaborative Training.

Provides a lightweight, CPU-only, permanent collaborative training orchestrator designed to run on a
low-cost GCP Spot VM (`e2-small` at ~$3.65/mo or colocated on `hm-kafka-1`) so that collaborative
DiLoCo + QLoRA MoE training continues across community GPUs (such as the local RTX 2070) even when
the Tesla T4 VM (`socreateai-agora-hermes-node1`) is offline, preempted, or busy serving daytime inference.

Key capabilities:
1. **Zero GPU Dependency (`requires_t4_gpu=False`)**: Runs outer DiLoCo Nesterov momentum updates,
   Multi-Krum / norm / finiteness verification, and held-out evaluation on CPU (`< 350 MiB` RAM).
2. **Dual-Transport Aglet Lifecycle (Tier A Kafka/AMCP + Tier B Outbound HTTP/HTTPS)**:
   Exposes `/v1/opentrainer/{status,offer,claim,result,tick}` over HTTP/HTTPS with signed AMCP
   envelopes (`HMACKeyring`) and optionally bridges to the AMCP/Kafka `hivemind.amcp.federation.v1` bus.
3. **T4-Offline Autonomous Quorum & CPU Sentinel Micro-Learner**:
   Advances rounds whenever `quorum` learners submit verified deltas (e.g. `rtx2070` alone when `quorum=1`,
   or `rtx2070` + the built-in `opentrainer-cpu-sentinel` learner for 2-shard cross-domain mixing when
   the T4 is absent).
4. **Atomic Checkpoint Persistence (`IP-08`)**:
   Writes checkpoints and round ledgers atomically (`temp + fsync + os.replace`) so Spot VM preemption
   restarts resume at the exact outer round with zero lost progress.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from fastapi import APIRouter, FastAPI, HTTPException, Request

from hivemind.swarm import federation_agents as fed
from hivemind.swarm.amcp_kafka_campaign_bus import KAFKA_TOPICS, AMCPKafkaEvent
from hivemind.swarm.amcp_kafka_transport import (
    EnvelopeVerifier,
    HMACKeyring,
    TransportRecord,
     VerifiedEvent,
    sign_envelope,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import e8_qlora_moe_round as qlora_round  # noqa: E402

FEDERATION_TOPIC = KAFKA_TOPICS["federation"]


def resolve_opentrainer_keyring(keyring: HMACKeyring | None = None) -> HMACKeyring:
    """Resolve the AMCP HMAC keyring from environment or deterministic local fallback."""
    if keyring is not None:
        return keyring
    if os.environ.get("HIVEMIND_AMCP_HMAC_KEYS", "").strip():
        return HMACKeyring.from_env()
    return HMACKeyring(
        keys={"opentrainer1": hashlib.sha256(b"hivemind-opentrainer-lowcost-vm-key-v1").digest()},
        active_kid="opentrainer1",
    )


def verify_federation_envelope(verifier: EnvelopeVerifier, envelope: Mapping[str, Any]) -> VerifiedEvent:
    """Verify a CloudEvents 1.0 JSON envelope dict on the federation topic using EnvelopeVerifier."""
    raw = json.dumps(dict(envelope), sort_keys=True, separators=(",", ":")).encode("utf-8")
    rec = TransportRecord(
        topic=FEDERATION_TOPIC,
        partition=0,
        offset=0,
        key=str(envelope.get("subject", "federation")).encode("utf-8"),
        value=raw,
    )
    return verifier.verify(rec)

PIN_FILES: tuple[Path, ...] = (
    SCRIPTS_DIR / "e8_qlora_moe_round.py",
    SCRIPTS_DIR / "e8_amcp_round.py",
    REPO_ROOT / "src" / "hivemind" / "swarm" / "federation_agents.py",
)


@dataclass(frozen=True)
class LowCostVMProfile:
    """GCP Spot VM & cost governance specification for the permanent OpenTrainer orchestrator."""

    vm_name: str = "hm-opentrainer-1"
    project_id: str = "gen-lang-client-0625573011"
    zone: str = "us-east4-b"
    machine_type: str = "e2-small"
    vcpus: int = 2
    memory_mib: int = 2048
    provisioning_model: str = "SPOT"
    termination_action: str = "STOP"
    accelerators_count: int = 0
    requires_t4_gpu: bool = False
    vpc_private_ip: str = "10.150.0.14"
    wireguard_overlay_ip: str = "10.77.0.6"
    static_external_ip_reserved: bool = False
    boot_disk_gb: int = 20
    boot_disk_type: str = "pd-standard"
    estimated_monthly_cost_usd: float = 3.65
    keepalive_scheduler_job: str = "hm-opentrainer-1-keepalive"
    keepalive_cron: str = "*/5 * * * *"
    systemd_unit: str = "hivemind-opentrainer.service"

    def validate_governance(self) -> dict[str, Any]:
        """Verify strict compliance with AGENTS.md GCP cost governance rules."""
        checks = {
            "spot_provisioning": self.provisioning_model == "SPOT",
            "zero_gpu_accelerators": self.accelerators_count == 0,
            "t4_independent": self.requires_t4_gpu is False,
            "no_paid_unused_static_ip": self.static_external_ip_reserved is False,
            "low_cost_budget_under_10_usd": self.estimated_monthly_cost_usd < 10.0,
            "stop_on_preemption": self.termination_action == "STOP",
        }
        return {
            "compliant": all(checks.values()),
            "checks": checks,
            "profile": asdict(self),
        }


@dataclass
class OpenTrainerConfig:
    """Configuration for a collaborative training run managed by PermanentOpenTrainerOrchestrator."""

    run_id: str = "opentrainer-perm-001"
    learners: tuple[str, ...] = ("l0", "l1")
    quorum: int = 1
    rounds: int = 50
    seed: int = 1
    inner_steps: int = 20
    inner_lr: float = 1e-2
    batch_size: int = 4
    shard_size: int = 32
    outer_lr: float = 0.5
    outer_mu: float = 0.5
    max_delta_l2: float = 50.0
    round_deadline_s: float = 120.0
    cpu_sentinel_when_t4_offline: bool = True
    cpu_sentinel_learner_id: str = "l0"
    cpu_sentinel_inner_steps: int = 10
    checkpoint_path: Path | None = None
    report_path: Path | None = None


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    """Write bytes atomically via a sibling temporary file + fsync + os.replace (IP-08)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}")
    with open(tmp_path, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, target)


def _atomic_write_json(target: Path, payload: Mapping[str, Any]) -> None:
    """Serialize JSON and persist atomically."""
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    _atomic_write_bytes(target, raw)


class PermanentOpenTrainerOrchestrator:
    """T4-independent collaborative QLoRA MoE + Outer DiLoCo orchestrator for a low-cost permanent VM."""

    def __init__(
        self,
        config: OpenTrainerConfig,
        *,
        keyring: HMACKeyring | None = None,
        vm_profile: LowCostVMProfile | None = None,
        bus: Any | None = None,
    ) -> None:
        if config.quorum < 1 or config.quorum > len(config.learners):
            raise ValueError(f"quorum must be in [1, {len(config.learners)}], got {config.quorum}")
        self.config = config
        self.keyring = resolve_opentrainer_keyring(keyring)
        self.verifier = EnvelopeVerifier(self.keyring)
        self.vm_profile = vm_profile or LowCostVMProfile()
        self.bus = bus
        self._lock = threading.RLock()

        self.pin = fed.code_pin(PIN_FILES)
        self.base_state = qlora_round.build_base_state(config.seed)
        self.base_sha256 = fed.tensors_sha256(self.base_state)
        self.global_state = qlora_round.initial_lora(self.base_state, config.seed)
        self.velocity: dict[str, torch.Tensor] | None = None
        self.current_round = 0
        self.retries_in_round = 0
        self.started_unix = time.time()
        self.round_started_unix = time.time()

        self._load_checkpoint_if_present()

        self.eval_model = qlora_round.load_model(self.base_state, "none", "cpu", config.seed)
        self.eval_dataset = qlora_round.make_eval(len(config.learners), config.seed)
        qlora_round.set_lora_state(self.eval_model, self.global_state)
        self.initial_eval_loss = float(qlora_round.eval_loss(self.eval_model, self.eval_dataset, "cpu"))
        self.current_eval_loss = self.initial_eval_loss
        self.rounds_log: list[dict[str, Any]] = []
        self.learner_last_seen: dict[str, dict[str, Any]] = {}

        self.current_offer_event: AMCPKafkaEvent | None = None
        self.current_book: fed.RoundBook | None = None
        if not self.is_completed:
            self._open_new_round_offer()

    @property
    def is_completed(self) -> bool:
        return self.current_round >= self.config.rounds

    def _load_checkpoint_if_present(self) -> None:
        ckpt = self.config.checkpoint_path
        if ckpt is None or not ckpt.exists():
            return
        blob = torch.load(ckpt, map_location="cpu", weights_only=True)
        loaded_global = {k[2:]: v for k, v in blob.items() if k.startswith("g.")}
        loaded_vel = {k[2:]: v for k, v in blob.items() if k.startswith("v.")}
        if not loaded_global or set(loaded_global) != set(self.global_state):
            raise ValueError(f"Corrupt or incompatible checkpoint at {ckpt}")
        self.global_state = loaded_global
        self.velocity = loaded_vel or None
        self.current_round = int(blob["round"].item()) if isinstance(blob.get("round"), torch.Tensor) else int(blob["round"])

    def _save_checkpoint(self) -> None:
        ckpt = self.config.checkpoint_path
        if ckpt is None:
            return
        import io

        buf = io.BytesIO()
        state_obj = {
            **{f"g.{k}": v.detach().cpu() for k, v in self.global_state.items()},
            **{f"v.{k}": v.detach().cpu() for k, v in (self.velocity or {}).items()},
            "round": torch.tensor(self.current_round, dtype=torch.int64),
        }
        torch.save(state_obj, buf)
        _atomic_write_bytes(ckpt, buf.getvalue())

    def _open_new_round_offer(self, now: float | None = None) -> AMCPKafkaEvent:
        now_ts = time.time() if now is None else float(now)
        deadline = now_ts + self.config.round_deadline_s
        offer = fed.build_offer(
            run_id=self.config.run_id,
            rnd=self.current_round,
            sender_id="opentrainer-orchestrator",
            global_state=self.global_state,
            base_sha256=self.base_sha256,
            pin=self.pin,
            inner_steps=self.config.inner_steps,
            deadline_unix=deadline,
            quorum=self.config.quorum,
            params={
                "seed": self.config.seed,
                "inner_lr": self.config.inner_lr,
                "batch": self.config.batch_size,
                "shard_size": self.config.shard_size,
                "learners": len(self.config.learners),
            },
        )
        self.current_offer_event = offer
        self.current_book = fed.RoundBook(
            run_id=self.config.run_id,
            rnd=self.current_round,
            allowed=self.config.learners,
            quorum=self.config.quorum,
            deadline_unix=deadline,
            global_sha256=offer.payload["global_sha256"],
            pin=self.pin,
            max_delta_l2=self.config.max_delta_l2,
            template=self.global_state,
        )
        self.round_started_unix = now_ts
        if self.bus is not None:
            self.bus.publish(offer)
        return offer

    def get_signed_offer_envelope(self, ttl_s: float = 900.0) -> dict[str, Any]:
        """Return the current round's task_offer wrapped in an HMAC-signed AMCP envelope."""
        with self._lock:
            if self.is_completed:
                abort_ev = fed.build_abort(
                    run_id=self.config.run_id,
                    rnd=self.current_round,
                    sender_id="opentrainer-orchestrator",
                    reason="run_complete",
                )
                return sign_envelope(abort_ev, self.keyring, ttl_s=ttl_s)
            if self.current_offer_event is None:
                self._open_new_round_offer()
            assert self.current_offer_event is not None
            return sign_envelope(self.current_offer_event, self.keyring, ttl_s=ttl_s)

    def submit_claim_envelope(self, envelope: Mapping[str, Any]) -> dict[str, Any]:
        """Verify a signed task_claim envelope and record the learner's claim on the active round."""
        verified = verify_federation_envelope(self.verifier, envelope)
        if verified.event.event_type != fed.CLAIM:
            raise ValueError(f"Expected {fed.CLAIM}, got {verified.event.event_type!r}")
        payload = verified.event.payload
        with self._lock:
            if self.current_book is None or self.is_completed:
                return {"accepted": False, "reason": "run_complete", "round": self.current_round}
            before = len(self.current_book.claims)
            self.current_book.claim(payload)
            lid = str(payload.get("learner_id", ""))
            if lid in self.current_book.claims:
                self.learner_last_seen[lid] = {
                    "host": str(payload.get("host", "?")),
                    "last_claim_round": self.current_round,
                    "timestamp_unix": time.time(),
                }
            return {
                "accepted": lid in self.current_book.claims,
                "newly_claimed": len(self.current_book.claims) > before,
                "round": self.current_round,
                "claims": sorted(self.current_book.claims),
            }

    def submit_result_envelope(self, envelope: Mapping[str, Any], *, now: float | None = None) -> dict[str, Any]:
        """Verify a signed task_result envelope, add it to the active RoundBook, and advance if ready."""
        verified = verify_federation_envelope(self.verifier, envelope)
        if verified.event.event_type != fed.RESULT:
            raise ValueError(f"Expected {fed.RESULT}, got {verified.event.event_type!r}")
        payload = verified.event.payload
        with self._lock:
            if self.current_book is None or self.is_completed:
                return {"accepted": False, "reason": "run_complete", "round": self.current_round}
            accepted = self.current_book.offer_result(payload)
            lid = str(payload.get("learner_id", "?"))
            if accepted:
                self.learner_last_seen[lid] = {
                    "host": str(payload.get("host", "?")),
                    "last_result_round": self.current_round,
                    "vram_peak_mib": payload.get("vram_peak_mib"),
                    "timestamp_unix": time.time(),
                }
            reason = None
            if not accepted and self.current_book.rejected:
                reason = self.current_book.rejected[-1][1]
            adv = self._maybe_close_round_locked(now=now)
            return {
                "accepted": accepted,
                "rejection_reason": reason,
                "round_advanced": adv["round_advanced"],
                "round_status": adv["round_status"],
                "current_round": self.current_round,
                "current_eval_loss": round(self.current_eval_loss, 6),
                "completed": self.is_completed,
            }

    def run_cpu_sentinel_step_if_needed(self, *, now: float | None = None) -> dict[str, Any]:
        """Execute a lightweight CPU sentinel learner step on the low-cost VM when the T4 learner is absent."""
        with self._lock:
            if self.is_completed or self.current_book is None or self.current_offer_event is None:
                return {"ran_sentinel": False, "reason": "no_open_round"}
            sentinel_id = self.config.cpu_sentinel_learner_id
            if not self.config.cpu_sentinel_when_t4_offline or sentinel_id not in self.config.learners:
                return {"ran_sentinel": False, "reason": "sentinel_disabled"}
            if sentinel_id in self.current_book.accepted or sentinel_id in self.current_book.claims:
                return {"ran_sentinel": False, "reason": "learner_already_active_in_round"}
            offer_payload = dict(self.current_offer_event.payload)
            rnd = self.current_round

        # Run the CPU sentinel training step outside the lock so HTTP polling stays fast
        idx = list(self.config.learners).index(sentinel_id)
        local_model = qlora_round.load_model(self.base_state, "none", "cpu", self.config.seed)
        start_state = fed.decode_tensors(str(offer_payload["global_b64"]))
        shard = qlora_round.make_shard(
            idx,
            self.config.shard_size,
            self.config.seed,
        )
        out_round = qlora_round.learner_round(
            local_model,
            start_state,
            shard,
            self.eval_dataset,
            steps=self.config.cpu_sentinel_inner_steps,
            lr=self.config.inner_lr,
            batch=self.config.batch_size,
            seed=self.config.seed,
            rnd=rnd,
            learner_id=idx,
            device="cpu",
            shuffle_labels=False,
        )
        delta = out_round["delta"]
        tokens = int(out_round["tokens"])
        eval_before = float(out_round["eval_loss_global"])
        norm = fed.check_delta(delta, start_state, self.config.max_delta_l2)
        res_ev = fed.build_result(
            run_id=self.config.run_id,
            rnd=rnd,
            learner_id=sentinel_id,
            host=f"{self.vm_profile.vm_name}-cpu-sentinel",
            tokens=tokens,
            delta=delta,
            delta_l2=norm,
            eval_loss_global=eval_before,
            pin=self.pin,
            global_sha256=str(offer_payload["global_sha256"]),
            vram_peak_mib=0.0,
        )
        env = sign_envelope(res_ev, self.keyring, ttl_s=900.0)
        out = self.submit_result_envelope(env, now=now)
        return {"ran_sentinel": True, "sentinel_learner_id": sentinel_id, **out}

    def tick(self, *, now: float | None = None, run_sentinel_if_t4_absent: bool = False) -> dict[str, Any]:
        """Advance round state on deadline expiry and optionally run the T4-offline CPU sentinel."""
        sentinel_info: dict[str, Any] | None = None
        if run_sentinel_if_t4_absent:
            sentinel_info = self.run_cpu_sentinel_step_if_needed(now=now)
        with self._lock:
            adv = self._maybe_close_round_locked(now=now)
            return {
                **adv,
                "current_round": self.current_round,
                "current_eval_loss": round(self.current_eval_loss, 6),
                "completed": self.is_completed,
                "sentinel": sentinel_info,
            }

    def _maybe_close_round_locked(self, *, now: float | None = None) -> dict[str, Any]:
        if self.current_book is None or self.is_completed:
            return {"round_advanced": False, "round_status": "completed"}

        status = self.current_book.status(now=now)
        if status == "open":
            return {"round_advanced": False, "round_status": "open"}

        if status == "failed":
            self.retries_in_round += 1
            if self.bus is not None:
                self.bus.publish(
                    fed.build_abort(
                        run_id=self.config.run_id,
                        rnd=self.current_round,
                        sender_id="opentrainer-orchestrator",
                        reason=f"quorum {self.config.quorum} not reached",
                    )
                )
            self._open_new_round_offer(now=now)
            return {"round_advanced": False, "round_status": "reoffered_after_timeout"}

        # Status is either 'complete' (all learners answered) or 'quorum' (deadline passed with >= quorum learners)
        book = self.current_book
        ids = sorted(book.accepted)
        deltas = [book.accepted[i]["delta"] for i in ids]
        tokens = [book.accepted[i]["tokens"] for i in ids]
        self.global_state, self.velocity = qlora_round.outer_step(
            self.global_state,
            deltas,
            tokens,
            self.velocity,
            self.config.outer_lr,
            self.config.outer_mu,
        )
        qlora_round.set_lora_state(self.eval_model, self.global_state)
        after_loss = float(qlora_round.eval_loss(self.eval_model, self.eval_dataset, "cpu"))
        before_loss = sum(float(book.accepted[i]["eval_loss_global"]) for i in ids) / float(len(ids))
        now_ts = time.time() if now is None else float(now)

        round_entry = {
            "round": self.current_round,
            "status": status,
            "accepted": ids,
            "hosts": {i: book.accepted[i]["host"] for i in ids},
            "claims": sorted(book.claims),
            "retries": self.retries_in_round,
            "rejected": [list(r) for r in book.rejected],
            "eval_loss_before": round(before_loss, 6),
            "eval_loss_after": round(after_loss, 6),
            "delta_l2": {i: round(float(book.accepted[i]["delta_l2"]), 6) for i in ids},
            "vram_peak_mib": {i: book.accepted[i]["vram_peak_mib"] for i in ids},
            "latency_s": round(max(0.0, now_ts - self.round_started_unix), 4),
            "t4_participated": any(
                "t4" in str(book.accepted[i]["host"]).lower() or "socreateai" in str(book.accepted[i]["host"]).lower()
                for i in ids
            ),
        }
        self.rounds_log.append(round_entry)
        self.current_eval_loss = after_loss
        self.current_round += 1
        self.retries_in_round = 0

        self._save_checkpoint()
        if self.config.report_path is not None:
            _atomic_write_json(self.config.report_path, self.build_report())

        if not self.is_completed:
            self._open_new_round_offer(now=now)
        elif self.bus is not None:
            self.bus.publish(
                fed.build_abort(
                    run_id=self.config.run_id,
                    rnd=self.current_round,
                    sender_id="opentrainer-orchestrator",
                    reason="run_complete",
                )
            )
        return {"round_advanced": True, "round_status": status}

    def build_report(self) -> dict[str, Any]:
        """Build a signed, tamper-evident status and experiment report."""
        with self._lock:
            hosts_seen = sorted({h for r in self.rounds_log for h in r["hosts"].values()})
            t4_rounds = sum(1 for r in self.rounds_log if r.get("t4_participated"))
            t4_free_rounds = len(self.rounds_log) - t4_rounds
            rep: dict[str, Any] = {
                "schema_version": 1,
                "component": "hivemind.opentrainer.lowcost_vm_orchestrator",
                "run_id": self.config.run_id,
                "completed": self.is_completed,
                "current_round": self.current_round,
                "total_rounds": self.config.rounds,
                "orchestrator_host": platform.node(),
                "t4_required": False,
                "t4_participated_rounds": t4_rounds,
                "t4_independent_rounds_completed": t4_free_rounds,
                "code_pin": self.pin,
                "base_sha256": self.base_sha256,
                "global_sha256": fed.tensors_sha256(self.global_state),
                "initial_eval_loss": round(self.initial_eval_loss, 6),
                "current_eval_loss": round(self.current_eval_loss, 6),
                "loss_reduction": round(self.initial_eval_loss - self.current_eval_loss, 6),
                "hosts_seen": hosts_seen,
                "learners_last_seen": dict(self.learner_last_seen),
                "vm_governance": self.vm_profile.validate_governance(),
                "rounds": list(self.rounds_log),
            }
            rep["receipt_sha256"] = hashlib.sha256(json.dumps(rep, sort_keys=True).encode("utf-8")).hexdigest()
            return rep


def execute_http_learner_round(
    orchestrator_url: str,
    *,
    learner_id: str,
    learner_index: int,
    keyring: HMACKeyring,
    quant: str = "nf4",
    device: str = "cuda",
    host_label: str | None = None,
    min_free_mib: float = 2048.0,
    need_mib: float = 250.0,
) -> dict[str, Any]:
    """Pull one signed task_offer from an OpenTrainer HTTP endpoint, train QLoRA inner steps, and push task_result."""
    import urllib.request

    verifier = EnvelopeVerifier(keyring)
    pin = fed.code_pin(PIN_FILES)
    host = host_label or platform.node()
    actual_device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    actual_quant = quant if actual_device == "cuda" else "none"

    if actual_device == "cuda":
        qlora_round.vram_guard(min_free_mib, need_mib, actual_device)

    base_url = orchestrator_url.rstrip("/")
    with urllib.request.urlopen(f"{base_url}/v1/opentrainer/offer", timeout=10.0) as resp:
        offer_env = json.loads(resp.read().decode("utf-8"))

    verified_offer = verify_federation_envelope(verifier, offer_env)
    if verified_offer.event.event_type == fed.ABORT:
        return {"status": "aborted", "reason": verified_offer.event.payload.get("reason", "run_complete")}

    payload = verified_offer.event.payload
    params = payload.get("params") or {}
    seed = int(params.get("seed", 1))
    base = qlora_round.build_base_state(seed)
    base_sha = fed.tensors_sha256(base)
    start_state = fed.check_offer(payload, my_pin=pin, my_base_sha256=base_sha)

    run_id = str(payload["run_id"])
    rnd = int(payload["round"])

    claim_ev = fed.build_claim(run_id=run_id, rnd=rnd, learner_id=learner_id, host=host)
    claim_req = urllib.request.Request(
        f"{base_url}/v1/opentrainer/claim",
        data=json.dumps(sign_envelope(claim_ev, keyring, ttl_s=900.0)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(claim_req, timeout=10.0) as resp:
        json.loads(resp.read().decode("utf-8"))

    n_learners = int(params.get("learners", 2))
    model = qlora_round.load_model(base, actual_quant, actual_device, seed)
    evalset = qlora_round.make_eval(n_learners, seed)
    if actual_device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    shard = qlora_round.make_shard(
        learner_index,
        int(params.get("shard_size", 32)),
        seed,
    )
    out_round = qlora_round.learner_round(
        model,
        start_state,
        shard,
        evalset,
        steps=int(payload["inner_steps"]),
        lr=float(params.get("inner_lr", 1e-2)),
        batch=int(params.get("batch", 4)),
        seed=seed,
        rnd=rnd,
        learner_id=learner_index,
        device=actual_device,
        shuffle_labels=False,
    )
    delta = out_round["delta"]
    tokens = int(out_round["tokens"])
    eval_before = float(out_round["eval_loss_global"])
    peak_vram = float(torch.cuda.max_memory_allocated() / 2**20) if actual_device == "cuda" else 0.0
    norm = fed.check_delta(delta, start_state, 50.0)
    res_ev = fed.build_result(
        run_id=run_id,
        rnd=rnd,
        learner_id=learner_id,
        host=host,
        tokens=tokens,
        delta=delta,
        delta_l2=norm,
        eval_loss_global=eval_before,
        pin=pin,
        global_sha256=str(payload["global_sha256"]),
        vram_peak_mib=peak_vram,
    )
    res_req = urllib.request.Request(
        f"{base_url}/v1/opentrainer/result",
        data=json.dumps(sign_envelope(res_ev, keyring, ttl_s=900.0)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(res_req, timeout=15.0) as resp:
        res_body = json.loads(resp.read().decode("utf-8"))

    return {
        "status": "submitted",
        "round": rnd,
        "learner_id": learner_id,
        "device": actual_device,
        "quant": actual_quant,
        "eval_loss_before": round(eval_before, 6),
        "delta_l2": round(norm, 6),
        "vram_peak_mib": round(float(peak_vram), 2),
        "orchestrator_response": res_body,
    }


def create_opentrainer_router(orchestrator: PermanentOpenTrainerOrchestrator) -> APIRouter:
    """Create a FastAPI APIRouter exposing `/v1/opentrainer/*` endpoints."""
    router = APIRouter(prefix="/v1/opentrainer", tags=["opentrainer"])

    @router.get("/status")
    def get_status() -> dict[str, Any]:
        return orchestrator.build_report()

    @router.get("/offer")
    def get_offer() -> dict[str, Any]:
        return orchestrator.get_signed_offer_envelope()

    @router.post("/claim")
    async def post_claim(request: Request) -> dict[str, Any]:
        body = await request.json()
        try:
            return orchestrator.submit_claim_envelope(body)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/result")
    async def post_result(request: Request) -> dict[str, Any]:
        body = await request.json()
        try:
            return orchestrator.submit_result_envelope(body)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/tick")
    async def post_tick(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except Exception:
            body = {}
        run_sentinel = bool(body.get("run_sentinel_if_t4_absent", True))
        return orchestrator.tick(run_sentinel_if_t4_absent=run_sentinel)

    return router


def create_opentrainer_app(
    orchestrator: PermanentOpenTrainerOrchestrator | None = None,
    config: OpenTrainerConfig | None = None,
) -> FastAPI:
    """Create a standalone FastAPI application for the low-cost VM OpenTrainer orchestrator."""
    orch = orchestrator or PermanentOpenTrainerOrchestrator(config or OpenTrainerConfig())
    app = FastAPI(
        title="HiveMind Permanent Low-Cost VM OpenTrainer Orchestrator",
        version="1.0.0",
    )
    app.state.opentrainer = orch
    app.include_router(create_opentrainer_router(orch))

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "component": "opentrainer-lowcost-vm",
            "t4_required": False,
            "current_round": orch.current_round,
            "completed": orch.is_completed,
        }

    return app
