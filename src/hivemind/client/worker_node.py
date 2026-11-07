# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Autonomous BOINC + Steam Worker Client Runtime.

Integrates:
- Steamworks AuthSessionTicket Sybil verification ($5+ Non-Limited account check)
- Win32 Zero-Intrusion Daemon (GetLastInputInfo + active game process preemption)
- Cryptographic Parquet shard verification (SHA-256 + HMAC)
- DiLoCo-QLoRA inner loop execution with micro-checkpointing & Top-k/FP8 delta compression
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from hivemind.boinc.workunit import BOINCWorkunit, verify_and_load_parquet_shard
from hivemind.client.daemon import ZeroIntrusionDaemon
from hivemind.client.steamworks import SteamAuthSessionTicket, SteamworksSDKBridge
from hivemind.config import ARTIFACTS_ROOT
from hivemind.core.diloco_inner import InnerLoopResult, run_diloco_inner_loop
from hivemind.core.quantization import load_lora_state_dict
from hivemind.swarm.strategy_optimizer import EF21DiLoCoErrorFeedbackBuffer


class HiveMindWorkerNode:
    """Consumer GPU Worker Node executing BOINC DiLoCo-QLoRA Workunits."""

    def __init__(
        self,
        worker_id: str,
        steam_id64: str,
        persona_name: str,
        model: nn.Module,
        steamworks: SteamworksSDKBridge | None = None,
        daemon: ZeroIntrusionDaemon | None = None,
        account_spend_usd: float = 49.99,
        enable_error_feedback: bool = False,
        error_feedback_stale_decay: float = 1.0,
        error_feedback_max_residual_norm: float | None = None,
    ) -> None:
        if not math.isfinite(float(account_spend_usd)) or float(account_spend_usd) < 0.0:
            raise ValueError("account_spend_usd must be finite and >= 0")
        if not math.isfinite(float(error_feedback_stale_decay)) or float(error_feedback_stale_decay) <= 0.0:
            raise ValueError("error_feedback_stale_decay must be finite and > 0")
        if error_feedback_max_residual_norm is not None and (
            not math.isfinite(float(error_feedback_max_residual_norm)) or float(error_feedback_max_residual_norm) <= 0.0
        ):
            raise ValueError("error_feedback_max_residual_norm must be finite and > 0")
        self.worker_id = worker_id
        self.steam_id64 = steam_id64
        self.persona_name = persona_name
        self.model = model
        self.steamworks = steamworks or SteamworksSDKBridge()
        self.daemon = daemon or ZeroIntrusionDaemon(enforce_os_idle_hook=False)
        self.auth_ticket: SteamAuthSessionTicket = self.steamworks.get_auth_session_ticket(
            steam_id64=steam_id64,
            account_spend_usd=account_spend_usd,
        )
        self.enable_error_feedback = enable_error_feedback
        self.error_feedback_stale_decay = error_feedback_stale_decay
        self.error_feedback_max_residual_norm = error_feedback_max_residual_norm
        self.error_feedback_buffers: dict[str, EF21DiLoCoErrorFeedbackBuffer] = {}
        self.last_ef21_telemetry: dict[str, dict[str, Any]] = {}

    def _error_feedback_for(self, branch_name: str, topk_ratio: float) -> EF21DiLoCoErrorFeedbackBuffer | None:
        """Return this worker's residual buffer for `branch_name`, creating it on first use.

        One buffer per branch: a residual is only meaningful against the branch whose
        pseudo-gradients produced it.
        """
        if not self.enable_error_feedback:
            return None
        buffer = self.error_feedback_buffers.get(branch_name)
        if buffer is None:
            buffer = EF21DiLoCoErrorFeedbackBuffer(
                topk_ratio=topk_ratio,
                stale_decay=self.error_feedback_stale_decay,
                max_residual_norm=self.error_feedback_max_residual_norm,
            )
            self.error_feedback_buffers[branch_name] = buffer
        return buffer

    def _record_ef21_telemetry(self, result: InnerLoopResult) -> None:
        """Store the measured residual state of the branch this upload came from."""
        if result.error_feedback_residual_norm is None:
            return
        buffer = self.error_feedback_buffers[result.branch_name]
        raw_norm = sum(float(t.detach().float().pow(2).sum().item()) for t in result.raw_delta.values()) ** 0.5
        residual = float(result.error_feedback_residual_norm)
        if not (math.isfinite(raw_norm) and math.isfinite(residual)):
            raise ValueError("EF21 telemetry norms must be finite")
        self.last_ef21_telemetry[result.branch_name] = {
            "worker_id": self.worker_id,
            "branch_name": result.branch_name,
            "outer_step": int(result.outer_step),
            "residual_norm": round(residual, 8),
            "raw_delta_norm": round(raw_norm, 8),
            "contraction_factor": round(residual / raw_norm, 8) if raw_norm > 0.0 else None,
            "clip_events": int(buffer.clip_events),
            "topk_ratio": float(buffer.topk_ratio),
            "stale_decay": float(buffer.stale_decay),
        }

    def ef21_telemetry(self) -> list[dict[str, Any]]:
        """Measured error-feedback state per branch, as of this worker's last compressed upload."""
        return [dict(entry) for entry in self.last_ef21_telemetry.values()]

    def execute_workunit(
        self,
        workunit: BOINCWorkunit,
        global_lora_state: dict[str, torch.Tensor],
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        micro_batch_size: int = 1,
        topk_ratio: float = 0.20,
        use_paged_8bit_adamw: bool = False,
        byzantine_poison_scale: float | None = None,
        use_muon_lora_preconditioner: bool = False,
        muon_ns_steps: int = 5,
    ) -> InnerLoopResult:
        """Verify Steam ticket & Parquet shard, execute H local AdamW steps, and emit compressed delta."""
        ok, msg = self.steamworks.verify_sybil_resistance(self.auth_ticket)
        if not ok:
            raise PermissionError(f"Worker {self.worker_id} blocked by Steam Sybil guard: {msg}")

        token_sequences = verify_and_load_parquet_shard(
            shard_path=workunit.data_shard_parquet,
            expected_sha256=workunit.shard_sha256,
            expected_hmac_signature=workunit.shard_hmac_signature,
            rng_seed=workunit.rng_seed,
        )

        load_lora_state_dict(self.model, global_lora_state)

        result = run_diloco_inner_loop(
            model=self.model,
            token_sequences=token_sequences,
            worker_id=self.worker_id,
            branch_name=workunit.branch_name,
            outer_step=workunit.outer_step,
            inner_steps_H=workunit.inner_steps_H,
            checkpoint_interval_s=workunit.spot_check_interval_s,
            lr=lr,
            weight_decay=weight_decay,
            rng_seed=workunit.rng_seed,
            micro_batch_size=micro_batch_size,
            topk_ratio=topk_ratio,
            use_paged_8bit_adamw=use_paged_8bit_adamw,
            should_preempt_fn=self.daemon.should_preempt,
            emergency_checkpoint_dir=ARTIFACTS_ROOT / "emergency_checkpoints",
            error_feedback=self._error_feedback_for(workunit.branch_name, topk_ratio),
            use_muon_lora_preconditioner=use_muon_lora_preconditioner,
            muon_ns_steps=muon_ns_steps,
        )
        self._record_ef21_telemetry(result)

        # Optional Byzantine attack injection for Multi-Krum & Spot-Check testing
        if byzantine_poison_scale is not None:
            for k in result.raw_delta:
                result.raw_delta[k] = -byzantine_poison_scale * result.raw_delta[k] + torch.randn_like(
                    result.raw_delta[k]
                ) * abs(byzantine_poison_scale)

        return result

    def get_seti_lifecycle_state(self, consumer_query_active: bool = False) -> str:
        """Return the current 3-state SETI-Swarm agent lifecycle state."""
        return self.daemon.resolve_seti_agent_state(consumer_query_active=consumer_query_active).value

    def execute_cas_shard(
        self,
        domain_name: str,
        token_sequences: torch.Tensor,
        global_lora_state: dict[str, torch.Tensor],
        outer_step: int = 1,
        inner_steps_H: int = 12,
        spot_check_interval_s: int = 6,
        lr: float = 5e-4,
        weight_decay: float = 0.01,
        rng_seed: int = 42,
        micro_batch_size: int = 2,
        topk_ratio: float = 0.50,
        use_muon_lora_preconditioner: bool = False,
        muon_ns_steps: int = 5,
    ) -> InnerLoopResult:
        """Execute a decentralized SETI-Swarm inner loop on a Content-Addressed Hugging Face shard."""
        ok, msg = self.steamworks.verify_sybil_resistance(self.auth_ticket)
        if not ok:
            raise PermissionError(f"Worker {self.worker_id} blocked by Steam Sybil guard: {msg}")

        self.daemon.resolve_seti_agent_state(consumer_query_active=False)
        load_lora_state_dict(self.model, global_lora_state)

        result = run_diloco_inner_loop(
            model=self.model,
            token_sequences=token_sequences,
            worker_id=self.worker_id,
            branch_name=domain_name,
            outer_step=outer_step,
            inner_steps_H=inner_steps_H,
            checkpoint_interval_s=spot_check_interval_s,
            lr=lr,
            weight_decay=weight_decay,
            rng_seed=rng_seed,
            micro_batch_size=micro_batch_size,
            topk_ratio=topk_ratio,
            use_paged_8bit_adamw=True,
            should_preempt_fn=self.daemon.should_preempt,
            emergency_checkpoint_dir=ARTIFACTS_ROOT / "emergency_checkpoints",
            error_feedback=self._error_feedback_for(domain_name, topk_ratio),
            use_muon_lora_preconditioner=use_muon_lora_preconditioner,
            muon_ns_steps=muon_ns_steps,
        )
        self._record_ef21_telemetry(result)
        return result

