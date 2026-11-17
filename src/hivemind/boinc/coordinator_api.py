# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Modified BOINC Coordinator Service & FastAPI REST Server.

Provides:
- Workunit (WU) scheduling across E=8 domain branches
- Cryptographically signed Parquet dataset sharding
- Deterministic Spot-Check verification (10% verification overhead)
- Multi-Krum + Nesterov outer-loop aggregation
- SteamID64 node reputation tracking & Compute-to-API credit minting
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from hivemind.client.steam_identity import IdentityError, SteamIdentityGate

from hivemind.boinc.validator import DeterministicSpotCheckValidator, SpotCheckVerdict
from hivemind.boinc.workunit import (
    BOINCWorkunit,
    create_signed_parquet_shard,
    verify_and_load_parquet_shard,
)
from hivemind.config import ARTIFACTS_ROOT, HiveMindConfig
from hivemind.core.btm_upcycle import ExpertBranchState, initialize_domain_branches
from hivemind.core.compression import compress_lora_delta, decompress_lora_delta
from hivemind.core.diloco_inner import InnerLoopResult
from hivemind.core.diloco_outer import DiLoCoOuterOptimizer, OuterStepResult
from hivemind.security.federated_learning_hardening import FederatedLearningSecurityGuard
from hivemind.security.swarm_runtime_hardening import (
    SwarmSecurityViolationError,
    SwarmTrainingAndStoreGuard,
)


@dataclass
class NodeReputationRecord:
    """Tracks reputation, validated FLOPs, and API credits for a SteamID64 participant."""

    steam_id64: str
    worker_id: str
    is_non_limited_steam_account: bool = True
    validated_workunits: int = 0
    rejected_workunits: int = 0
    contributed_tflops: float = 0.0
    minted_api_credits: float = 0.0
    reputation_score: float = 100.0
    flagged_byzantine: bool = False

    def __post_init__(self) -> None:
        for s_name in ("steam_id64", "worker_id"):
            val = getattr(self, s_name)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{s_name} must be a non-empty string")
        for b_name in ("is_non_limited_steam_account", "flagged_byzantine"):
            if not isinstance(getattr(self, b_name), bool):
                raise ValueError(f"{b_name} must be a boolean")
        for i_name in ("validated_workunits", "rejected_workunits"):
            val = getattr(self, i_name)
            if isinstance(val, bool) or not isinstance(val, int) or val < 0:
                raise ValueError(f"{i_name} must be a non-negative integer")
        for f_name in ("contributed_tflops", "minted_api_credits", "reputation_score"):
            val = getattr(self, f_name)
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not math.isfinite(float(val))
                or float(val) < 0.0
            ):
                raise ValueError(f"{f_name} must be a finite non-negative number")


class WorkunitRequestPayload(BaseModel):
    worker_id: str
    steam_id64: str
    preferred_branch: str | None = None
    vram_gb: float = 8.0


class BOINCCoordinatorService:
    """Central Coordinator managing E=8 BTM branches, Workunits, Validation, and Multi-Krum."""

    def __init__(
        self,
        config: HiveMindConfig,
        initial_lora_state: dict[str, torch.Tensor],
        base_model_sha256: str,
        shards_dir: Path | None = None,
        vocab_size: int = 4096,
        max_active_workunits_per_node: int = 8,
        max_total_active_workunits: int = 512,
        workunit_ttl_s: float = 3600.0,
    ) -> None:
        if not isinstance(base_model_sha256, str) or not base_model_sha256.strip():
            raise ValueError("base_model_sha256 must be a non-empty string")
        for iname, ival in (
            ("vocab_size", vocab_size),
            ("max_active_workunits_per_node", max_active_workunits_per_node),
            ("max_total_active_workunits", max_total_active_workunits),
        ):
            if isinstance(ival, bool) or not isinstance(ival, int) or ival < 1:
                raise ValueError(f"{iname} must be an integer >= 1")
        if (
            isinstance(workunit_ttl_s, bool)
            or not isinstance(workunit_ttl_s, (int, float))
            or not math.isfinite(float(workunit_ttl_s))
            or float(workunit_ttl_s) <= 0.0
        ):
            raise ValueError("workunit_ttl_s must be a positive finite number")
        self.config = config
        self.base_model_sha256 = base_model_sha256
        self.vocab_size = vocab_size
        self.shards_dir = shards_dir or (ARTIFACTS_ROOT / "shards")
        self.checkpoints_dir = ARTIFACTS_ROOT / "checkpoints"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.max_active_workunits_per_node = max(1, int(max_active_workunits_per_node))
        self.max_total_active_workunits = max(1, int(max_total_active_workunits))
        self.workunit_ttl_s = max(1.0, float(workunit_ttl_s))

        # Phase A: Initialize E=8 independent domain branches from identical W_0
        self.branches: dict[str, ExpertBranchState] = initialize_domain_branches(
            initial_lora_state=initial_lora_state,
            expert_domains=config.expert_domains,
        )
        self.validator = DeterministicSpotCheckValidator(
            tolerance_eps=config.spot_check_tolerance_eps
        )
        self.outer_optimizer = DiLoCoOuterOptimizer(
            outer_lr_eta=config.outer_lr_eta,
            outer_momentum_mu=config.outer_momentum_mu,
            max_byzantine_f=1,
            cosine_sigma=config.cosine_outlier_sigma,
        )
        self.fl_security_guard = FederatedLearningSecurityGuard()
        self.active_workunits: dict[str, BOINCWorkunit] = {}
        self._workunit_meta: dict[str, tuple[str, float]] = {}
        self._shard_cache: dict[tuple[str, int, int, int, int], tuple[Path, str, str]] = {}
        self.pending_branch_deltas: dict[str, dict[str, dict[str, torch.Tensor]]] = {
            dom: {} for dom in self.branches
        }
        self.reputation_ledger: dict[str, NodeReputationRecord] = {}
        self._rr_branch_idx = 0

    def prune_active_workunits(self, now_s: float | None = None) -> int:
        """Evict consumed, stale-outer-step, or TTL-expired entries from ``active_workunits`` (IP-249)."""
        if now_s is not None and (
            isinstance(now_s, bool) or not isinstance(now_s, (int, float)) or not math.isfinite(float(now_s))
        ):
            raise ValueError("now_s must be a finite number or None")
        now = time.monotonic() if now_s is None else float(now_s)
        consumed = self.fl_security_guard.consumed_workunit_ids
        to_remove: list[str] = []
        for wu_id, wu in list(self.active_workunits.items()):
            branch_state = self.branches.get(wu.branch_name)
            expected_step = (branch_state.outer_step + 1) if branch_state is not None else wu.outer_step
            meta = self._workunit_meta.get(wu_id)
            assigned_at = meta[1] if meta is not None else now
            if (
                wu_id in consumed
                or wu.outer_step != expected_step
                or (now - assigned_at) > self.workunit_ttl_s
            ):
                to_remove.append(wu_id)
        for wu_id in to_remove:
            self.active_workunits.pop(wu_id, None)
            self._workunit_meta.pop(wu_id, None)
        return len(to_remove)

    def register_or_get_node(
        self,
        steam_id64: str,
        worker_id: str,
        is_non_limited: bool = True,
    ) -> NodeReputationRecord:
        if not isinstance(is_non_limited, bool):
            raise ValueError("is_non_limited must be a bool")
        clean_sid = steam_id64.strip() if isinstance(steam_id64, str) else ""
        clean_wid = worker_id.strip() if isinstance(worker_id, str) else ""
        if (
            not clean_sid
            or not clean_wid
            or any(tok in clean_sid or tok in clean_wid for tok in ("..", "/", "\\", "\x00"))
        ):
            raise SwarmSecurityViolationError(
                "COORD-01",
                "INVALID_NODE_IDENTITY_OR_PATH_INJECTION",
                f"Invalid or path-traversal node identity: steam_id64='{steam_id64}', worker_id='{worker_id}'.",
            )
        if not is_non_limited:
            raise PermissionError(
                f"Steam account {steam_id64} is Limited (< $5 spent). Sybil defense blocks WU assignment."
            )
        if steam_id64 not in self.reputation_ledger:
            self.reputation_ledger[steam_id64] = NodeReputationRecord(
                steam_id64=steam_id64,
                worker_id=worker_id,
                is_non_limited_steam_account=is_non_limited,
            )
        rec = self.reputation_ledger[steam_id64]
        if rec.flagged_byzantine:
            raise PermissionError(
                f"Steam account {steam_id64} is flagged for Byzantine/verification failure."
            )
        return rec

    def assign_workunit(
        self,
        worker_id: str,
        steam_id64: str,
        branch_name: str | None = None,
        inner_steps_H: int | None = None,
        spot_check_interval_s: int | None = None,
        num_sequences: int = 32,
        seq_len: int = 64,
        rng_seed: int | None = None,
    ) -> tuple[BOINCWorkunit, dict[str, torch.Tensor]]:
        """Schedule a BOINC Workunit and generate a signed Parquet training shard."""
        if isinstance(num_sequences, bool) or not isinstance(num_sequences, int) or num_sequences < 1:
            raise ValueError("num_sequences must be an integer >= 1")
        if isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len < 1:
            raise ValueError("seq_len must be an integer >= 1")
        if inner_steps_H is not None and (
            isinstance(inner_steps_H, bool) or not isinstance(inner_steps_H, int) or inner_steps_H < 1
        ):
            raise ValueError("inner_steps_H must be an integer >= 1 or None")
        if spot_check_interval_s is not None and (
            isinstance(spot_check_interval_s, bool) or not isinstance(spot_check_interval_s, int) or spot_check_interval_s < 1
        ):
            raise ValueError("spot_check_interval_s must be an integer >= 1 or None")
        if rng_seed is not None and (isinstance(rng_seed, bool) or not isinstance(rng_seed, int)):
            raise ValueError("rng_seed must be an integer or None")
        self.register_or_get_node(steam_id64=steam_id64, worker_id=worker_id)

        domains = list(self.branches.keys())
        if branch_name is None or branch_name not in self.branches:
            branch_name = domains[self._rr_branch_idx % len(domains)]
            self._rr_branch_idx += 1

        branch_state = self.branches[branch_name]
        outer_step = branch_state.outer_step + 1
        wu_id = f"wu-{branch_name.lower()}-t{outer_step}-{uuid.uuid4().hex[:6]}"
        seed = rng_seed if rng_seed is not None else (1000 + outer_step * 137 + branch_state.branch_id * 17)

        now = time.monotonic()
        self.prune_active_workunits(now_s=now)

        node_wus = [
            (wid, meta[1])
            for wid, meta in self._workunit_meta.items()
            if meta[0] == steam_id64 and wid in self.active_workunits
        ]
        if len(node_wus) >= self.max_active_workunits_per_node:
            node_wus.sort(key=lambda item: item[1])
            for stale_wid, _ in node_wus[: len(node_wus) - self.max_active_workunits_per_node + 1]:
                self.active_workunits.pop(stale_wid, None)
                self._workunit_meta.pop(stale_wid, None)

        if len(self.active_workunits) >= self.max_total_active_workunits:
            all_wus = sorted(
                ((wid, self._workunit_meta.get(wid, ("", 0.0))[1]) for wid in self.active_workunits),
                key=lambda item: item[1],
            )
            for stale_wid, _ in all_wus[: len(all_wus) - self.max_total_active_workunits + 1]:
                self.active_workunits.pop(stale_wid, None)
                self._workunit_meta.pop(stale_wid, None)

        shard_key = (branch_name, outer_step, int(seed), int(num_sequences), int(seq_len))
        cached_shard = self._shard_cache.get(shard_key)
        if cached_shard is not None and cached_shard[0].exists():
            shard_path, shard_sha256, shard_sig = cached_shard
        else:
            shard_file = self.shards_dir / (
                f"shard-{branch_name.lower()}-t{outer_step}-s{seed}-n{num_sequences}-l{seq_len}.parquet"
            )
            shard_path, shard_sha256, shard_sig = create_signed_parquet_shard(
                shard_path=shard_file,
                branch_name=branch_name,
                rng_seed=seed,
                num_sequences=num_sequences,
                seq_len=seq_len,
                vocab_size=self.vocab_size,
            )
            self._shard_cache[shard_key] = (shard_path, shard_sha256, shard_sig)

        ckpt_path = self.checkpoints_dir / f"lora_{branch_name}_t{branch_state.outer_step}.pt"
        wu = BOINCWorkunit(
            wu_id=wu_id,
            branch_name=branch_name,
            outer_step=outer_step,
            base_model_id=self.base_model_sha256,
            lora_checkpoint_url=str(ckpt_path),
            data_shard_parquet=str(shard_path),
            shard_sha256=shard_sha256,
            shard_hmac_signature=shard_sig,
            rng_seed=seed,
            inner_steps_H=inner_steps_H or self.config.inner_steps_H,
            spot_check_interval_s=spot_check_interval_s or self.config.spot_check_interval_steps,
            seq_len=seq_len,
            num_sequences=num_sequences,
        )
        self.active_workunits[wu_id] = wu
        self._workunit_meta[wu_id] = (steam_id64, now)
        self.fl_security_guard.register_workunit_assignment(wu_id, worker_id=worker_id, steam_id64=steam_id64)
        return wu, {k: v.clone() for k, v in branch_state.lora_state.items()}

    def validate_and_record_submission(
        self,
        validator_model: nn.Module,
        workunit: BOINCWorkunit,
        submission: InnerLoopResult,
        steam_id64: str,
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        micro_batch_size: int = 1,
    ) -> SpotCheckVerdict:
        """Run Phase S-3 federated envelope verification + 50-step segment spot-check validation, update reputation, and queue delta for Multi-Krum."""
        if not isinstance(steam_id64, str) or not steam_id64.strip():
            raise ValueError("steam_id64 must be a non-empty string")
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(float(lr)) or float(lr) <= 0.0:
            raise ValueError("lr must be a positive finite float")
        if (
            isinstance(weight_decay, bool)
            or not isinstance(weight_decay, (int, float))
            or not math.isfinite(float(weight_decay))
            or float(weight_decay) < 0.0
        ):
            raise ValueError("weight_decay must be a non-negative finite float")
        if isinstance(micro_batch_size, bool) or not isinstance(micro_batch_size, int) or micro_batch_size < 1:
            raise ValueError("micro_batch_size must be a positive integer")
        branch_state = self.branches.get(workunit.branch_name)
        coord_outer_step = (branch_state.outer_step + 1) if branch_state is not None else workunit.outer_step
        self.fl_security_guard.check_workunit_submission_envelope(
            wu_id=workunit.wu_id,
            expected_branch=workunit.branch_name,
            submitted_branch=submission.branch_name,
            coordinator_outer_step=coord_outer_step,
            submitted_outer_step=submission.outer_step,
            steam_id64=steam_id64,
            worker_id=submission.worker_id,
            steps_completed=submission.steps_completed,
            expected_inner_steps=workunit.inner_steps_H,
            preempted_by_daemon=submission.preempted_by_daemon,
        )
        token_sequences = verify_and_load_parquet_shard(
            shard_path=workunit.data_shard_parquet,
            expected_sha256=workunit.shard_sha256,
            expected_hmac_signature=workunit.shard_hmac_signature,
            rng_seed=workunit.rng_seed,
        )
        verdict = self.validator.verify_submission(
            validator_model=validator_model,
            workunit=workunit,
            submission=submission,
            token_sequences=token_sequences,
            steam_id64=steam_id64,
            lr=lr,
            weight_decay=weight_decay,
            micro_batch_size=micro_batch_size,
            expected_init_state=branch_state.lora_state if branch_state is not None else None,
        )
        decompressed: dict[str, torch.Tensor] | None = None
        if verdict.passed:
            decompressed = decompress_lora_delta(submission.compressed_delta)
            expected_pkg = compress_lora_delta(
                delta_dict=submission.raw_delta,
                worker_id=submission.worker_id,
                branch_name=submission.branch_name,
                outer_step=submission.outer_step,
                inner_steps_completed=submission.steps_completed,
                topk_ratio=submission.compressed_delta.topk_ratio,
            )
            expected_decompressed = decompress_lora_delta(expected_pkg)
            compressed_ok = set(decompressed.keys()) == set(expected_decompressed.keys())
            if compressed_ok:
                if submission.error_feedback_residual_norm is None:
                    for k, exp_t in expected_decompressed.items():
                        dec_t = decompressed[k].detach().to(torch.float32).cpu()
                        ref_t = exp_t.detach().to(torch.float32).cpu()
                        raw_t = submission.raw_delta[k].detach().to(torch.float32).cpu()
                        if not bool(torch.isfinite(dec_t).all()):
                            compressed_ok = False
                            break
                        diff = float((dec_t - ref_t).abs().max().item()) if dec_t.numel() > 0 else 0.0
                        raw_mag = float(raw_t.abs().max().item()) if raw_t.numel() > 0 else 0.0
                        tol = max(self.validator.tolerance_eps, 1e-3 * raw_mag)
                        if not math.isfinite(diff) or diff > tol:
                            compressed_ok = False
                            break
                else:
                    for dec_t in decompressed.values():
                        if not bool(torch.isfinite(dec_t.detach().to(torch.float32).cpu()).all()):
                            compressed_ok = False
                            break
                    if compressed_ok:
                        raw_l2 = math.sqrt(
                            sum(
                                float(t.detach().to(torch.float32).cpu().pow(2).sum().item())
                                for t in submission.raw_delta.values()
                            )
                        )
                        decomp_l2 = math.sqrt(
                            sum(
                                float(t.detach().to(torch.float32).cpu().pow(2).sum().item())
                                for t in decompressed.values()
                            )
                        )
                        ef_norm = float(submission.error_feedback_residual_norm or 0.0)
                        if not math.isfinite(decomp_l2) or decomp_l2 > max(1e-5, 4.0 * raw_l2 + 2.0 * ef_norm):
                            compressed_ok = False
            if not compressed_ok:
                verdict.passed = False
                verdict.steam_account_flagged = True
                verdict.reason = "REJECTED: compressed_delta diverges from validated raw_delta (IP-77)"
                self.validator._flag_account(steam_id64, verdict.reason)

        # The verdict (pass or fail) is final for this wu_id: only now cast the FL-06 vote and consume it.
        self.fl_security_guard.record_workunit_submission(
            wu_id=workunit.wu_id,
            expected_branch=workunit.branch_name,
            coordinator_outer_step=coord_outer_step,
            steam_id64=steam_id64,
            worker_id=submission.worker_id,
        )
        rec = self.reputation_ledger.setdefault(
            steam_id64,
            NodeReputationRecord(steam_id64=steam_id64, worker_id=submission.worker_id),
        )
        if verdict.passed and decompressed is not None:
            step_fraction = min(1.0, max(0.0, submission.steps_completed / max(1, workunit.inner_steps_H)))
            rec.validated_workunits += 1
            rec.contributed_tflops = round(rec.contributed_tflops + 0.125 * submission.steps_completed, 3)
            rec.minted_api_credits = round(rec.minted_api_credits + 10.0 * step_fraction, 2)
            rec.reputation_score = min(100.0, round(rec.reputation_score + 2.5 * step_fraction, 2))
            self.pending_branch_deltas[workunit.branch_name][submission.worker_id] = decompressed
        else:
            rec.rejected_workunits += 1
            rec.reputation_score = 0.0
            rec.flagged_byzantine = True

        self.prune_active_workunits()
        return verdict

    def inject_raw_delta_for_aggregation(
        self,
        branch_name: str,
        worker_id: str,
        delta_dict: dict[str, torch.Tensor],
    ) -> None:
        """Directly queue a worker pseudo-gradient into the pending branch pool (for Multi-Krum tests)."""
        if branch_name not in self.branches:
            raise ValueError(f"Unknown branch_name {branch_name!r}")
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("worker_id must be a non-empty string")
        if not isinstance(delta_dict, dict) or not delta_dict:
            raise ValueError("delta_dict must be a non-empty dict of tensors")
        self.pending_branch_deltas[branch_name][worker_id] = delta_dict

    def step_branch_outer_loop(self, branch_name: str) -> OuterStepResult:
        """Execute Multi-Krum filtering + Nesterov outer update for `branch_name`."""
        if branch_name not in self.branches:
            raise ValueError(f"Unknown branch_name {branch_name!r}")
        branch = self.branches[branch_name]
        worker_deltas = self.pending_branch_deltas[branch_name]
        if not worker_deltas:
            raise ValueError(f"No validated worker deltas queued for branch '{branch_name}'.")

        next_step = branch.outer_step + 1
        res = self.outer_optimizer.step(
            branch_name=branch_name,
            outer_step=next_step,
            theta_prev=branch.lora_state,
            worker_deltas=worker_deltas,
        )
        branch.lora_state = res.theta_updated
        branch.outer_step = next_step
        self.pending_branch_deltas[branch_name] = {}
        self.prune_active_workunits()
        return res


LOGGER = logging.getLogger(__name__)


def _anonymous_id(steam_id64: str) -> str:
    """Stable, non-reversible label for a node in public listings (the raw SteamID64 stays inside the coordinator)."""
    if not isinstance(steam_id64, str) or not steam_id64.strip():
        raise ValueError("steam_id64 must be a non-empty string")
    return hashlib.sha256(b"hivemind-node:" + steam_id64.encode()).hexdigest()[:16]


def create_coordinator_fastapi_app(coordinator: BOINCCoordinatorService, identity: SteamIdentityGate | None = None) -> FastAPI:
    """Create the FastAPI application exposing the Modified BOINC Server endpoints.

    ``identity=None`` is development/simulation mode: ``steam_id64`` in the request body is taken on trust (tests, demos and local runs use synthetic ids).
    With a :class:`SteamIdentityGate` the app is meant to be reachable by others: a node must sign in with Steam (``/identity/steam/login``), and
    ``/workunit/request`` then requires the resulting token for the SAME SteamID64; Limited and VAC-banned accounts never get a token.
    """
    app = FastAPI(
        title="Project HiveMind — Modified BOINC Coordinator API",
        version="0.1.0",
        description="Workunit scheduling, signed Parquet sharding, Multi-Krum aggregation, and SteamID64 reputation.",
    )
    if identity is None:
        LOGGER.warning("coordinator started WITHOUT Steam identity verification: steam_id64 in requests is trusted (development mode only)")
    else:

        @app.get("/identity/steam/login")
        def steam_login() -> RedirectResponse:
            return RedirectResponse(identity.login_redirect(), status_code=302)

        @app.get("/identity/steam/callback")
        def steam_callback(request: Request) -> dict[str, Any]:
            try:
                steam_id64, token = identity.complete_login(dict(request.query_params))
            except IdentityError as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc
            return {"steam_id64": steam_id64, "identity_token": token, "expires_in_s": identity.tokens.ttl_s}

    @app.get("/status")
    def get_status() -> dict[str, Any]:
        return {
            "codename": coordinator.config.codename,
            "base_model_sha256": coordinator.base_model_sha256,
            "num_expert_branches": len(coordinator.branches),
            "branches": {
                name: {
                    "branch_id": b.branch_id,
                    "outer_step": b.outer_step,
                    "pending_submissions": len(coordinator.pending_branch_deltas[name]),
                }
                for name, b in coordinator.branches.items()
            },
            "active_workunits": len(coordinator.active_workunits),
            "registered_steam_nodes": len(coordinator.reputation_ledger),
        }

    def require_identity(request: Request, claimed_steam_id64: str | None = None) -> None:
        """No-op in development mode; with a gate, the request must carry a valid identity token (for ``claimed_steam_id64`` if one is named)."""
        if identity is None:
            return
        header = request.headers.get("authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else None
        try:
            if claimed_steam_id64 is None:
                identity.identify(token)
            else:
                identity.authorize(token, claimed_steam_id64)
        except IdentityError as exc:
            raise HTTPException(status_code=401 if not token else 403, detail=str(exc)) from exc

    @app.post("/workunit/request")
    def request_workunit(req: WorkunitRequestPayload, request: Request) -> dict[str, Any]:
        require_identity(request, req.steam_id64)
        try:
            wu, _ = coordinator.assign_workunit(
                worker_id=req.worker_id,
                steam_id64=req.steam_id64,
                branch_name=req.preferred_branch,
            )
            return {"status": "assigned", "workunit": wu.to_dict()}
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.get("/reputation")
    def get_reputation_ledger() -> dict[str, Any]:
        return {
            (sid if identity is None else _anonymous_id(sid)): {
                "worker_id": r.worker_id if identity is None else _anonymous_id(r.worker_id),
                "validated_workunits": r.validated_workunits,
                "rejected_workunits": r.rejected_workunits,
                "contributed_tflops": r.contributed_tflops,
                "minted_api_credits": r.minted_api_credits,
                "reputation_score": r.reputation_score,
                "flagged_byzantine": r.flagged_byzantine,
            }
            for sid, r in coordinator.reputation_ledger.items()
        }

    @app.post("/v1/chat/completions", response_model=None)
    def openai_chat_completions_gateway(payload: dict[str, Any], request: Request) -> Any:
        """M4 OpenAI-compatible `/v1/chat/completions` proxy with SSE streaming (`HM-M4-01`), OPoI 5% audit & L2 $HIVE voucher.

        NOTE: this is a SIMULATION. The completion comes from the Hack-Miner game probe, and the experts, routing weights and vouchers are illustrative
        (fixed provider names); it is not model inference. Real routed inference is the OpenRouter-style gateway in ``hivemind.router``.
        """
        require_identity(request)
        import json
        import re
        import time
        from fastapi.responses import StreamingResponse
        from hivemind.game.hf_hack_miner import execute_interactive_bounty_probe
        from hivemind.swarm.swarm_fabric import (
            HiveLayer2StateChannel,
            OptimisticProofOfInferenceAuditor,
            TokenTreeSpeculativeVerifier,
        )

        messages = payload.get("messages", [])
        user_prompt = str(
            messages[-1].get("content", "Hello Swarm")
            if messages and isinstance(messages[-1], dict)
            else payload.get("prompt", "Hello Swarm")
        )
        # Rule COORD-02: Screen gateway prompts against oversized DoS, sleeper-agent backdoor triggers, and payment solicitation
        if len(user_prompt) > 8192:
            raise HTTPException(
                status_code=400,
                detail="[COORD-02] Prompt exceeds 8,192-character gateway safety limit.",
            )
        gateway_guard = SwarmTrainingAndStoreGuard(min_entropy_bits=0.0)
        prompt_verdict = gateway_guard.validate_dataset_shard(
            "OpenAI_Gateway_Prompt",
            [user_prompt if len(user_prompt) >= 8 else f"{user_prompt}        "],
            raise_on_violation=False,
        )
        if not prompt_verdict.passed:
            raise HTTPException(
                status_code=400,
                detail=f"[COORD-02:{prompt_verdict.rule_id}] {prompt_verdict.message}",
            )
        model_name = str(payload.get("model", "hivemind-qwen2.5-sparse-moe-top2"))
        probe = execute_interactive_bounty_probe(
            gamer_tag=str(payload.get("user", "CyberValkyrie_FR")),
            prompt=user_prompt,
        )
        l2 = HiveLayer2StateChannel(
            consumer_id="seti-node-paris-01",
            provider_ids=["seti-node-lyon-02", "seti-node-nice-03"],
        )
        # No producer activation commitment reaches this endpoint, so no OPoI audit can be performed here. The previous
        # code compared a reference tensor with itself and reported the inevitable pass as an audit result.
        opoi = {
            "status": "NOT_PERFORMED",
            "reason": "no producer activation commitment is available on this endpoint; nothing was audited",
            "opoi_passed": None,
            "slashed": False,
            "auditor": OptimisticProofOfInferenceAuditor.__name__,
        }
        completion_id = f"chatcmpl-hivemind-{uuid.uuid4().hex[:10]}"
        created_ts = int(time.time())
        prompt_tokens = max(1, len(user_prompt.split()))

        if not payload.get("stream"):
            voucher = l2.sign_micro_voucher(tokens_generated=32)
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created_ts,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": probe["generated_solution"],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": 32,
                    "total_tokens": prompt_tokens + 32,
                },
                "hivemind_swarm_fabric": {
                    "top2_active_experts": probe["top2_active_experts"],
                    "expert_routing_weights": probe["expert_routing_weights"],
                    "tree_speculation_gamma": 6,
                    "opoi_audit": opoi,
                    "l2_state_channel_voucher": {
                        "channel_id": voucher.channel_id,
                        "nonce": voucher.nonce,
                        "cumulative_hive_spent": voucher.cumulative_hive_spent,
                        "signature_hmac": voucher.signature_hmac,
                    },
                },
            }

        tokens = re.findall(r"\S+\s*", str(probe["generated_solution"])) or [str(probe["generated_solution"])]
        batches = ["".join(tokens[i : i + 5]) for i in range(0, len(tokens), 5)]
        verifier = TokenTreeSpeculativeVerifier(gamma=6, branching_factor=2)

        def event_stream():
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_ts, 'model': model_name, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
            streamed_tokens = 0
            for idx, text_delta in enumerate(batches):
                batch_len = max(4, min(6, len(text_delta.split()) or 4))
                seq_ids = list(range(10, 10 + batch_len + 1))
                tree_res = verifier.verify_token_tree(
                    draft_tree_candidates=[seq_ids[:batch_len]],
                    target_reference_tokens=seq_ids,
                )
                l2.sign_micro_voucher(tokens_generated=batch_len)
                streamed_tokens += batch_len
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": text_delta}, "finish_reason": None}],
                    "speculative_batch": {
                        "batch_index": idx,
                        "tokens_in_chunk": batch_len,
                        "tree_depth_gamma": tree_res["tree_depth_gamma"],
                        "accepted_length": tree_res["accepted_length"],
                    },
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            voucher = l2.latest_voucher or l2.sign_micro_voucher(tokens_generated=32)
            closing = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": streamed_tokens,
                    "total_tokens": prompt_tokens + streamed_tokens,
                },
                "hivemind_swarm_fabric": {
                    "top2_active_experts": probe["top2_active_experts"],
                    "expert_routing_weights": probe["expert_routing_weights"],
                    "tree_speculation_gamma": 6,
                    "opoi_audit": opoi,
                    "l2_state_channel_voucher": {
                        "channel_id": voucher.channel_id,
                        "nonce": voucher.nonce,
                        "cumulative_tokens": voucher.cumulative_tokens,
                        "cumulative_hive_spent": voucher.cumulative_hive_spent,
                        "signature_hmac": voucher.signature_hmac,
                    },
                },
            }
            yield f"data: {json.dumps(closing)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


