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
import json
import logging
import os
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
from hivemind.core.compression import decompress_lora_delta
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


class WorkunitRequestPayload(BaseModel):
    worker_id: str
    steam_id64: str
    preferred_branch: str | None = None
    vram_gb: float = 8.0


class RealBoincLedgerBridge:
    """Reads and summarizes a real BOINC project server's ``hivemind_ledger`` directory."""

    def __init__(self, ledger_dir: Path) -> None:
        self.ledger_dir = Path(ledger_dir)

    def _read_json_if_exists(self, name: str) -> dict[str, Any] | None:
        p = self.ledger_dir / name
        if not p.is_file():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def summarize(self) -> dict[str, Any]:
        assimilated_total = 0
        assimilated_by_app: dict[str, int] = {}
        assimilated_by_domain: dict[str, int] = {}
        latest_eval_by_domain: dict[str, float] = {}

        jsonl_path = self.ledger_dir / "assimilated.jsonl"
        if jsonl_path.is_file():
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if not isinstance(entry, dict):
                    continue
                assimilated_total += 1
                app_name = str(entry.get("app", "unknown"))
                assimilated_by_app[app_name] = assimilated_by_app.get(app_name, 0) + 1
                dom = entry.get("domain")
                if isinstance(dom, str) and dom:
                    assimilated_by_domain[dom] = assimilated_by_domain.get(dom, 0) + 1
                    loss_after = entry.get("eval_loss_after")
                    if isinstance(loss_after, (int, float)):
                        latest_eval_by_domain[dom] = float(loss_after)
                eval_map = entry.get("eval_by_domain_after")
                if isinstance(eval_map, dict):
                    for d_key, d_val in eval_map.items():
                        if isinstance(d_val, (int, float)):
                            latest_eval_by_domain[str(d_key)] = float(d_val)

        verdicts_total = 0
        verdicts_valid = 0
        verdicts_rejected = 0
        verified_cobblestones_total = 0.0
        verified_flops_total = 0.0
        credits_by_host: dict[str, dict[str, Any]] = {}

        if self.ledger_dir.is_dir():
            for vfile in sorted(self.ledger_dir.glob("verdict_*.json")):
                try:
                    vdata = json.loads(vfile.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(vdata, dict):
                    continue
                verdicts_total += 1
                host = str(vdata.get("host", "host"))
                host_rec = credits_by_host.setdefault(
                    host,
                    {
                        "valid_workunits": 0,
                        "rejected_workunits": 0,
                        "cobblestones": 0.0,
                        "flops_granted": 0.0,
                    },
                )
                if vdata.get("valid"):
                    verdicts_valid += 1
                    host_rec["valid_workunits"] += 1
                    credit = vdata.get("grant") or vdata.get("credit") or {}
                    if isinstance(credit, dict):
                        cobb = float(credit.get("cobblestones", 0.0))
                        flops = float(credit.get("flops_granted", cobb * 432.0e9))
                        verified_cobblestones_total += cobb
                        verified_flops_total += flops
                        host_rec["cobblestones"] = round(host_rec["cobblestones"] + cobb, 6)
                        host_rec["flops_granted"] = round(host_rec["flops_granted"] + flops, 2)
                else:
                    verdicts_rejected += 1
                    host_rec["rejected_workunits"] += 1

        return {
            "ledger_dir": str(self.ledger_dir),
            "assimilated_total": assimilated_total,
            "assimilated_by_app": assimilated_by_app,
            "assimilated_by_domain": assimilated_by_domain,
            "latest_eval_by_domain": latest_eval_by_domain,
            "latest_adapter": self._read_json_if_exists("latest_adapter.json"),
            "latest_adapter_big": self._read_json_if_exists("latest_adapter_big.json"),
            "latest_solver_sweep": self._read_json_if_exists("latest_solver_sweep.json"),
            "verdicts_total": verdicts_total,
            "verdicts_valid": verdicts_valid,
            "verdicts_rejected": verdicts_rejected,
            "verified_cobblestones_total": round(verified_cobblestones_total, 6),
            "verified_flops_total": round(verified_flops_total, 2),
            "credits_by_host": credits_by_host,
        }


class BOINCCoordinatorService:
    """Central Coordinator managing E=8 BTM branches, Workunits, Validation, and Multi-Krum."""

    def __init__(
        self,
        config: HiveMindConfig,
        initial_lora_state: dict[str, torch.Tensor],
        base_model_sha256: str,
        shards_dir: Path | None = None,
        vocab_size: int = 4096,
        boinc_ledger_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.base_model_sha256 = base_model_sha256
        self.vocab_size = vocab_size
        self.shards_dir = shards_dir or (ARTIFACTS_ROOT / "shards")
        self.checkpoints_dir = ARTIFACTS_ROOT / "checkpoints"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.boinc_ledger_dir = Path(
            boinc_ledger_dir
            if boinc_ledger_dir is not None
            else os.environ.get("HIVEMIND_LEDGER_DIR", str(ARTIFACTS_ROOT / "boinc_ledger"))
        )
        self.ledger_bridge = RealBoincLedgerBridge(self.boinc_ledger_dir)
        self._synced_boinc_verdicts: set[str] = set()

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
        self.pending_branch_deltas: dict[str, dict[str, dict[str, torch.Tensor]]] = {
            dom: {} for dom in self.branches
        }
        self.reputation_ledger: dict[str, NodeReputationRecord] = {}
        self._rr_branch_idx = 0

    def register_or_get_node(
        self,
        steam_id64: str,
        worker_id: str,
        is_non_limited: bool = True,
    ) -> NodeReputationRecord:
        clean_sid = str(steam_id64).strip()
        clean_wid = str(worker_id).strip()
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
        self.register_or_get_node(steam_id64=steam_id64, worker_id=worker_id)

        domains = list(self.branches.keys())
        if branch_name is None or branch_name not in self.branches:
            branch_name = domains[self._rr_branch_idx % len(domains)]
            self._rr_branch_idx += 1

        branch_state = self.branches[branch_name]
        outer_step = branch_state.outer_step + 1
        wu_id = f"wu-{branch_name.lower()}-t{outer_step}-{uuid.uuid4().hex[:6]}"
        seed = rng_seed if rng_seed is not None else (1000 + outer_step * 137 + branch_state.branch_id * 17)

        shard_file = self.shards_dir / f"{wu_id}.parquet"
        shard_path, shard_sha256, shard_sig = create_signed_parquet_shard(
            shard_path=shard_file,
            branch_name=branch_name,
            rng_seed=seed,
            num_sequences=num_sequences,
            seq_len=seq_len,
            vocab_size=self.vocab_size,
        )

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
        )
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
        if verdict.passed:
            rec.validated_workunits += 1
            rec.contributed_tflops = round(rec.contributed_tflops + 0.125 * submission.steps_completed, 3)
            rec.minted_api_credits = round(rec.minted_api_credits + 10.0, 2)
            rec.reputation_score = min(100.0, round(rec.reputation_score + 2.5, 2))
            # Decompress uploaded Top-k + FP8 delta package on the server
            decompressed = decompress_lora_delta(submission.compressed_delta)
            self.pending_branch_deltas[workunit.branch_name][submission.worker_id] = decompressed
        else:
            rec.rejected_workunits += 1
            rec.reputation_score = 0.0
            rec.flagged_byzantine = True

        return verdict

    def inject_raw_delta_for_aggregation(
        self,
        branch_name: str,
        worker_id: str,
        delta_dict: dict[str, torch.Tensor],
    ) -> None:
        """Directly queue a worker pseudo-gradient into the pending branch pool (for Multi-Krum tests)."""
        self.pending_branch_deltas[branch_name][worker_id] = delta_dict

    def step_branch_outer_loop(self, branch_name: str) -> OuterStepResult:
        """Execute Multi-Krum filtering + Nesterov outer update for `branch_name`."""
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
        return res

    def get_boinc_ledger_summary(self) -> dict[str, Any]:
        """Return a structured summary of the real BOINC project server's ``hivemind_ledger`` directory."""
        return self.ledger_bridge.summarize()

    def sync_from_real_boinc_ledger(
        self,
        host_to_steam_id64: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Idempotently ingest validated/rejected BOINC verdicts from ``boinc_ledger_dir`` into ``reputation_ledger``."""
        mapping = host_to_steam_id64 or {}
        ingested = 0
        if self.boinc_ledger_dir.is_dir():
            for vfile in sorted(self.boinc_ledger_dir.glob("verdict_*.json")):
                if vfile.name in self._synced_boinc_verdicts:
                    continue
                try:
                    data = json.loads(vfile.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(data, dict):
                    continue
                host = str(data.get("host", "boinc-host"))
                steam_id64 = str(mapping.get(host, f"boinc:{host}"))
                rec = self.reputation_ledger.setdefault(
                    steam_id64,
                    NodeReputationRecord(steam_id64=steam_id64, worker_id=host),
                )
                if data.get("valid"):
                    rec.validated_workunits += 1
                    credit = data.get("grant") or data.get("credit") or {}
                    cobb = float(credit.get("cobblestones", 0.0)) if isinstance(credit, dict) else 0.0
                    flops = float(credit.get("flops_granted", cobb * 432.0e9)) if isinstance(credit, dict) else 0.0
                    rec.contributed_tflops = round(rec.contributed_tflops + flops / 1e12, 6)
                    rec.minted_api_credits = round(rec.minted_api_credits + max(cobb, 0.1), 4)
                    rec.reputation_score = min(100.0, round(rec.reputation_score + 2.5, 2))
                else:
                    rec.rejected_workunits += 1
                    rec.reputation_score = 0.0
                    rec.flagged_byzantine = True
                self._synced_boinc_verdicts.add(vfile.name)
                ingested += 1
        return {
            "ingested_verdicts": ingested,
            "total_tracked_verdicts": len(self._synced_boinc_verdicts),
            "registered_nodes": len(self.reputation_ledger),
        }


LOGGER = logging.getLogger(__name__)


def _anonymous_id(steam_id64: str) -> str:
    """Stable, non-reversible label for a node in public listings (the raw SteamID64 stays inside the coordinator)."""
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

    @app.get("/boinc/v1/ledger/summary")
    def get_boinc_ledger_summary() -> dict[str, Any]:
        return coordinator.get_boinc_ledger_summary()

    @app.post("/boinc/v1/ledger/sync")
    def sync_boinc_ledger(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = payload or {}
        host_map = body.get("host_to_steam_id64") if isinstance(body.get("host_to_steam_id64"), dict) else None
        return coordinator.sync_from_real_boinc_ledger(host_to_steam_id64=host_map)

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
        auditor = OptimisticProofOfInferenceAuditor(audit_probability=0.05)
        ref_act = torch.ones(1, 8, 64, dtype=torch.float16)
        opoi = auditor.verify_activation_commitment("seti-node-lyon-02", ref_act, ref_act)
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


