# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-25: autonomous WSP campaign scheduler and ledger."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.auto_wsp_campaign_scheduler import (
    ANSE_MAXIMUM_PAIN,
    DRIFT_TRIGGER_OBJECT,
    GENERATION_RECEIPT_OBJECT,
    GENESIS_RECEIPT_SHA256,
    LEDGER_SNAPSHOT_OBJECT,
    RECOMMENDED_S5_STAGES,
    SCHEMA_VERSION,
    SPLIT_ALLOCATION_OBJECT,
    AnseMonotonicityViolationError,
    AutoWSPCampaignLedger,
    DisjointSplitAllocator,
    DriftTelemetryWindow,
    LedgerChainError,
    SplitContaminationError,
    SplitQuadruple,
    VaultSealedError,
    compute_expert_norm_ratio,
    compute_payload_sha256,
    compute_state_sha256,
    compute_wsp_anse_energy,
    evaluate_drift_recalibration_trigger,
    verify_generation_receipt_hash,
    verify_ledger_chain,
    verify_shard_rows_disjoint,
)

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:
    ZeroStubAudit = None  # type: ignore[assignment,misc]


class _FallbackStubResult:
    """CI-safe stub audit result when optional ``gwaya`` is absent."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = tuple(violations)
        self.is_clean = len(violations) == 0
        self.penalty_energy = 0.0 if self.is_clean else 1e6


class _FallbackZeroStubAudit:
    """CI-safe fallback implementing both ``check_file`` and ``audit_python_code``."""

    def check_file(self, path: str | Path) -> list[str]:
        text = Path(path).read_text(encoding="utf-8")
        return list(self.audit_python_code(text).violations)

    def audit_python_code(self, code: str) -> _FallbackStubResult:
        tree = ast.parse(code)
        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                stmts = [
                    s
                    for s in node.body
                    if not (
                        isinstance(s, ast.Expr)
                        and isinstance(s.value, ast.Constant)
                        and isinstance(s.value.value, str)
                    )
                ]
                if not stmts or (len(stmts) == 1 and isinstance(stmts[0], ast.Pass)):
                    violations.append(f"stub function: {node.name}")
                if (
                    len(stmts) == 1
                    and isinstance(stmts[0], ast.Raise)
                    and isinstance(stmts[0].exc, ast.Call)
                    and isinstance(stmts[0].exc.func, ast.Name)
                    and stmts[0].exc.func.id == "NotImplementedError"
                ):
                    violations.append(f"NotImplementedError in {node.name}")
        return _FallbackStubResult(violations)


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULE_PATH = REPO_ROOT / "src" / "hivemind" / "swarm" / "auto_wsp_campaign_scheduler.py"
TEST_FILE_PATH = Path(__file__).resolve()
EXPECTED_AGPL_HEADER = (
    "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
)


def _load_module_from_path(name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_cpu_checkpoint_state(seed: int, scale: float = 0.05) -> dict[str, torch.Tensor]:
    """Create a deterministic CPU-only synthetic MoE checkpoint state dict."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return {
        "blocks.0.mlp.router.weight": scale * torch.randn((6, 16), generator=gen, dtype=torch.float32),
        "blocks.0.mlp.experts.0.w3_A": scale * torch.randn((4, 16), generator=gen, dtype=torch.float32),
        "blocks.0.mlp.experts.0.w3_B": scale * torch.randn((16, 4), generator=gen, dtype=torch.float32),
        "blocks.0.mlp.experts.5.w3_B": scale * torch.randn((16, 4), generator=gen, dtype=torch.float32),
        "ln.weight": torch.ones(16, dtype=torch.float32) + 0.01 * torch.randn(16, generator=gen),
        "global_step": torch.tensor([seed], dtype=torch.int64),
    }


def _materialize_shard_rows(skip_offset: int, stride: int, *, dataset_id: str = "fineweb_edu") -> list[dict[str, Any]]:
    """Build deterministic ``(dataset_id, row)`` descriptors for ``[skip_offset, skip_offset + stride)``."""
    return [{"dataset_id": dataset_id, "row": skip_offset + idx} for idx in range(stride)]


def test_s1_four_generation_allocation_zero_overlap_and_monotone_anse_chain() -> None:
    """(S1) 4-generation campaign certifies 0-overlap across 16 splits and monotone ANSE hash chain."""
    allocator = DisjointSplitAllocator(start_skip=40, split_stride=40, frozen_skip_floor=40)
    assert allocator.start_skip == 40
    assert allocator.split_stride == 40
    assert allocator.frozen_skip_floor == 40

    quads: list[SplitQuadruple] = []
    named_shard_splits: list[tuple[str, list[dict[str, Any]]]] = []
    for gen_idx in range(1, 5):
        gid = f"G_{gen_idx}"
        quad = allocator.allocate_generation(gid)
        quads.append(quad)
        assert quad.generation_index == gen_idx
        assert quad.to_dict()["object"] == SPLIT_ALLOCATION_OBJECT
        assert len(quad.intervals()) == 4
        for role_name, sk_off in quad.skip_offsets().items():
            named_shard_splits.append((f"{gid}_{role_name}", _materialize_shard_rows(sk_off, quad.split_stride)))

    # Verify exact expected non-overlapping progression across G_1..G_4
    assert quads[0].as_tuple() == (40, 80, 120, 160)
    assert quads[1].as_tuple() == (200, 240, 280, 320)
    assert quads[2].as_tuple() == (360, 400, 440, 480)
    assert quads[3].as_tuple() == (520, 560, 600, 640)

    alloc_manifest = allocator.verify_all_splits_disjoint()
    assert alloc_manifest["disjoint"] is True
    assert alloc_manifest["total_generations"] == 4
    assert alloc_manifest["total_splits"] == 16
    assert len(alloc_manifest["all_split_offsets"]) == 16
    assert len(set(alloc_manifest["all_split_offsets"])) == 16

    shard_report = verify_shard_rows_disjoint(named_shard_splits)
    assert shard_report["disjoint"] is True
    assert shard_report["split_count"] == 16
    assert shard_report["total_unique_rows"] == 16 * 40

    # Deterministic CPU checkpoints & hash-chained monotone ANSE ledger across G_1..G_4
    ledger = AutoWSPCampaignLedger(
        campaign_id="nightly-c2-wsp-campaign",
        split_allocator=allocator,
        general_guard_rel=0.01,
        min_energy_improvement=0.005,
        baseline_anse_energy=0.0,
    )
    assert ledger.campaign_id == "nightly-c2-wsp-campaign"
    assert ledger.head_receipt_sha256 == GENESIS_RECEIPT_SHA256
    assert ledger.champion_receipt is None
    assert ledger.min_energy_improvement == pytest.approx(0.005)
    assert ledger.baseline_anse_energy == pytest.approx(0.0)

    base_state = _make_cpu_checkpoint_state(1000)
    parent_sha = compute_state_sha256(base_state)
    base_nll_mean = 2.150

    gen_specs = [
        ("G_1", "WSP_S1_alphaedit_hybrid", 2.045, 0.0040, 0.0055, 1001, True),
        ("G_2", "WSP_S2_layer_localized", 1.985, 0.0030, 0.0040, 1002, True),
        ("G_3", "WSP_S3_qr_svd_surgery", 1.930, 0.0022, 0.0028, 1003, True),
        ("G_4", "WSP_S5_pass_synergy", 1.880, 0.0005, 0.0010, 1004, True),
    ]

    receipts = []
    expected_prev_hash = GENESIS_RECEIPT_SHA256
    for idx, (gid, arm, cand_nll, drift_rel, worst_rel, seed_val, consume_v) in enumerate(gen_specs):
        ckpt_sha = compute_state_sha256(_make_cpu_checkpoint_state(seed_val))
        rec = ledger.record_generation(
            generation_id=gid,
            arm_name=arm,
            split_quadruple=None if idx == 0 else quads[idx],
            nll_mean=cand_nll,
            base_nll_mean=base_nll_mean,
            drift_ratio=drift_rel,
            worst_domain_drift_ratio=worst_rel,
            checkpoint_sha256=ckpt_sha,
            parent_checkpoint_sha256=parent_sha,
            consume_virgin_vault=consume_v,
        )
        assert rec.object == GENERATION_RECEIPT_OBJECT
        assert rec.schema == SCHEMA_VERSION
        assert rec.prev_receipt_sha256 == expected_prev_hash
        assert rec.virgin_vault_sealed is True
        assert compute_payload_sha256(rec.core_payload()) == rec.receipt_sha256
        assert verify_generation_receipt_hash(rec) is True
        if idx == 0:
            assert rec.delta_anse_energy is None
        else:
            assert rec.delta_anse_energy is not None and rec.delta_anse_energy < -0.005
            assert rec.anse_energy < receipts[-1].anse_energy
        receipts.append(rec)
        expected_prev_hash = rec.receipt_sha256
        parent_sha = ckpt_sha

    assert len(ledger.entries) == 4
    assert ledger.champion_receipt == receipts[-1]
    assert ledger.head_receipt_sha256 == receipts[-1].receipt_sha256
    assert allocator.sealed_vault_skips == frozenset({160, 320, 480, 640})

    chain_report = ledger.verify_chain(raise_on_error=True)
    assert chain_report["valid"] is True
    assert chain_report["entry_count"] == 4
    assert chain_report["monotone_anse"] is True
    assert chain_report["total_splits_verified"] == 16
    assert chain_report["chain_head_sha256"] == receipts[-1].receipt_sha256

    snapshot = ledger.export_ledger()
    assert snapshot["object"] == LEDGER_SNAPSHOT_OBJECT
    restored = AutoWSPCampaignLedger.from_dict(snapshot)
    assert restored.head_receipt_sha256 == ledger.head_receipt_sha256
    assert restored.entries == ledger.entries
    assert restored.split_allocator.sealed_vault_skips == allocator.sealed_vault_skips


def test_s2_permanent_vault_sealing_blocks_consumed_skip_reuse() -> None:
    """(S2) Permanent vault sealing blocks any re-allocation or overlap with consumed skips."""
    allocator = DisjointSplitAllocator(start_skip=40, split_stride=40, frozen_skip_floor=40)
    q1 = allocator.allocate_generation("G_1")
    assert q1.virgin_vault_skip == 160
    assert allocator.is_vault_sealed(160) is False
    assert allocator.is_generation_sealed("G_1") is False

    eval_sha = compute_payload_sha256({"nll_mean": 1.91, "drift_ratio": 0.001, "anse_energy": -0.24})
    seal_receipt = allocator.seal_virgin_vault("G_1", evaluation_receipt_sha256=eval_sha, outcome="PROMOTED")
    assert seal_receipt["sealed"] is True
    assert seal_receipt["virgin_vault_skip"] == 160
    assert allocator.is_vault_sealed(160) is True
    assert allocator.is_vault_sealed(175) is True
    assert allocator.is_generation_sealed("G_1") is True

    # 1. Re-sealing the same generation fails closed with VaultSealedError
    with pytest.raises(VaultSealedError, match="already permanently sealed"):
        allocator.seal_virgin_vault("G_1")

    # 2. Attempting to re-allocate consumed skip=160 in ANY role raises VaultSealedError
    for bad_kwargs in (
        {"probe_skip": 160, "selection_skip": 200, "confirm_skip": 240, "virgin_vault_skip": 280},
        {"probe_skip": 200, "selection_skip": 160, "confirm_skip": 240, "virgin_vault_skip": 280},
        {"probe_skip": 200, "selection_skip": 240, "confirm_skip": 160, "virgin_vault_skip": 280},
        {"probe_skip": 200, "selection_skip": 240, "confirm_skip": 280, "virgin_vault_skip": 160},
        {"probe_skip": 180, "selection_skip": 240, "confirm_skip": 280, "virgin_vault_skip": 320},
    ):
        with pytest.raises(VaultSealedError, match="permanently sealed"):
            allocator.allocate_generation("G_2_bad", **bad_kwargs)

    # 3. Attempting to reuse non-vault reserved offsets or frozen offsets raises SplitContaminationError
    with pytest.raises(SplitContaminationError, match="overlaps existing"):
        allocator.allocate_generation(
            "G_2_overlap", probe_skip=40, selection_skip=200, confirm_skip=240, virgin_vault_skip=280
        )
    with pytest.raises(SplitContaminationError, match="frozen window"):
        allocator.allocate_generation(
            "G_2_frozen", probe_skip=0, selection_skip=200, confirm_skip=240, virgin_vault_skip=280
        )

    # 4. External consumed vault offset sealing blocks manual reuse AND makes auto allocation jump over it
    ext_seal = allocator.seal_consumed_skip_offset(240, reason="prior_campaign_vault", evaluation_receipt_sha256=eval_sha)
    assert ext_seal["virgin_vault_skip"] == 240
    with pytest.raises(VaultSealedError, match="already sealed"):
        allocator.seal_consumed_skip_offset(250)
    with pytest.raises(SplitContaminationError, match="already reserved"):
        allocator.seal_consumed_skip_offset(80)

    q2 = allocator.allocate_generation("G_2")
    assert 240 not in q2.as_tuple()
    assert q2.as_tuple() == (200, 280, 320, 360)

    # 5. Restored allocator preserves permanent vault seals across process restarts
    restored = DisjointSplitAllocator.from_state(allocator.export_state())
    assert restored.sealed_vault_skips == frozenset({160, 240})
    with pytest.raises(VaultSealedError):
        restored.allocate_generation(
            "G_3_bad", probe_skip=160, selection_skip=400, confirm_skip=440, virgin_vault_skip=480
        )

    # 6. Ledger also blocks recording a generation whose split touches a sealed skip (even via fresh forged quadruple)
    ledger = AutoWSPCampaignLedger("vault-seal-test", split_allocator=restored)
    ckpt_a = compute_state_sha256(_make_cpu_checkpoint_state(111))
    ckpt_b = compute_state_sha256(_make_cpu_checkpoint_state(112))
    # Recording G_1 into ledger links its existing seal without duplicating
    r1 = ledger.record_generation(
        generation_id="G_1",
        arm_name="WSP_S1",
        split_quadruple=q1,
        nll_mean=1.90,
        base_nll_mean=2.05,
        drift_ratio=0.002,
        checkpoint_sha256=ckpt_a,
        parent_checkpoint_sha256=ckpt_b,
        consume_virgin_vault=True,
    )
    assert r1.virgin_vault_sealed is True
    assert r1.vault_seal_receipt_sha256 == seal_receipt["receipt_sha256"]

    isolated_alloc = DisjointSplitAllocator(start_skip=160, split_stride=40, frozen_skip_floor=40)
    forged_q = isolated_alloc.allocate_generation("G_3")
    with pytest.raises(VaultSealedError):
        ledger.record_generation(
            generation_id="G_3",
            arm_name="WSP_S3",
            split_quadruple=forged_q,
            nll_mean=1.80,
            base_nll_mean=2.05,
            drift_ratio=0.001,
            checkpoint_sha256=ckpt_b,
            parent_checkpoint_sha256=ckpt_a,
        )


def test_s3_drift_triggered_vs_daytime_low_ram_deferred_recalibration() -> None:
    """(S3) Offline CPU S5 scheduling triggers on drift/rogue-norm and defers on daytime or low RAM."""
    # Deterministic CPU expert LoRA tensors: 2 zeroed ghost sinks + 3 normal specialists + 1 rogue spike
    gen = torch.Generator(device="cpu").manual_seed(20261125)
    healthy_experts = {
        "expert_0": torch.zeros((16, 4), dtype=torch.float32),
        "expert_1": torch.zeros((16, 4), dtype=torch.float32),
        "expert_2": 0.25 * torch.randn((16, 4), generator=gen),
        "expert_4": 0.25 * torch.randn((16, 4), generator=gen),
        "expert_5": 0.25 * torch.randn((16, 4), generator=gen),
    }
    healthy_ratio, _ = compute_expert_norm_ratio(healthy_experts)
    assert healthy_ratio < 2.0

    rogue_experts = dict(healthy_experts)
    rogue_experts["expert_3"] = 1.20 * torch.randn((16, 4), generator=gen)
    rogue_ratio, norms_map = compute_expert_norm_ratio(rogue_experts)
    assert rogue_ratio >= 2.5
    assert norms_map["expert_0"] == 0.0

    # 1. Healthy overnight window -> NO_ACTION_HEALTHY
    d_healthy = evaluate_drift_recalibration_trigger(
        DriftTelemetryWindow(
            drift_ratio=0.0025,
            observed_rogue_norm_ratio=healthy_ratio,
            free_ram_gib=15.0,
            nll_mean=1.94,
            window_id="win_healthy",
        ),
        general_drift_trigger_rel=0.008,
        rogue_expert_norm_ratio=2.5,
        min_free_ram_gib=11.0,
        is_overnight_window=True,
    )
    assert d_healthy.object == DRIFT_TRIGGER_OBJECT
    assert d_healthy.should_schedule is False
    assert d_healthy.trigger_fired is False
    assert d_healthy.deferred is False
    assert d_healthy.status == "NO_ACTION_HEALTHY"
    assert d_healthy.recommended_s5_stages == ()

    # 2. General drift >= 0.008 overnight with >= 11.0 GiB RAM -> SCHEDULED_OFFLINE_CPU_S5
    d_drift_sched = evaluate_drift_recalibration_trigger(
        {
            "window_id": "win_drift_night",
            "domain_nlls": {"E0_general_fineweb": 1.615, "E1_code": 1.400},
            "base_domain_nlls": {"E0_general_fineweb": 1.600, "E1_code": 1.450},
            "observed_rogue_norm_ratio": 1.2,
            "free_ram_gib": 12.8,
        },
        general_drift_trigger_rel=0.008,
        rogue_expert_norm_ratio=2.5,
        min_free_ram_gib=11.0,
        is_overnight_window=True,
    )
    assert d_drift_sched.should_schedule is True
    assert d_drift_sched.trigger_fired is True
    assert d_drift_sched.deferred is False
    assert d_drift_sched.status == "SCHEDULED_OFFLINE_CPU_S5"
    assert d_drift_sched.execution_mode == "cpu_only_offline_s5"
    assert d_drift_sched.trigger_reasons == ("general_drift_threshold_exceeded",)
    assert d_drift_sched.recommended_s5_stages == RECOMMENDED_S5_STAGES
    assert d_drift_sched.nll_mean == pytest.approx(1.5075)

    # 3. Rogue expert norm ratio >= 2.5 computed directly from sequence/dict overnight -> SCHEDULED_OFFLINE_CPU_S5
    d_rogue_sched = evaluate_drift_recalibration_trigger(
        {
            "window_id": "win_rogue_night",
            "current_general_nll": 1.602,
            "base_general_nll": 1.600,
            "expert_norms": rogue_experts,
            "available_ram_gib": 14.0,
        },
        is_overnight_window=True,
    )
    assert d_rogue_sched.should_schedule is True
    assert d_rogue_sched.trigger_reasons == ("rogue_expert_norm_ratio_exceeded",)

    # 4. Triggered during DAYTIME (is_overnight_window=False) -> DEFERRED_DAYTIME_WINDOW
    d_daytime = evaluate_drift_recalibration_trigger(
        {
            "window_id": "win_daytime",
            "general_drift_rel": 0.0091,
            "rogue_expert_norm_ratio": 3.1,
            "free_ram_gib": 16.0,
        },
        is_overnight_window=False,
    )
    assert d_daytime.should_schedule is False
    assert d_daytime.trigger_fired is True
    assert d_daytime.deferred is True
    assert d_daytime.status == "DEFERRED_DAYTIME_WINDOW"
    assert d_daytime.defer_reasons == ("outside_overnight_window",)
    assert set(d_daytime.trigger_reasons) == {
        "general_drift_threshold_exceeded",
        "rogue_expert_norm_ratio_exceeded",
    }

    # 5. Triggered overnight with LOW RAM (9.2 GiB < 11.0 GiB) -> DEFERRED_LOW_RAM
    d_low_ram = evaluate_drift_recalibration_trigger(
        {
            "window_id": "win_low_ram",
            "drift_ratio": 0.0090,
            "max_expert_norm_ratio": 1.4,
            "mem_available_gib": 9.2,
        },
        min_free_ram_gib=11.0,
        is_overnight_window=True,
    )
    assert d_low_ram.should_schedule is False
    assert d_low_ram.trigger_fired is True
    assert d_low_ram.deferred is True
    assert d_low_ram.status == "DEFERRED_LOW_RAM"
    assert d_low_ram.defer_reasons == ("insufficient_free_ram",)

    # 6. Triggered during DAYTIME + LOW RAM -> DEFERRED_DAYTIME_AND_LOW_RAM
    d_both = evaluate_drift_recalibration_trigger(
        {"drift_ratio": 0.011, "observed_rogue_norm_ratio": 1.2, "free_ram_gib": 7.5},
        is_overnight_window=False,
    )
    assert d_both.status == "DEFERRED_DAYTIME_AND_LOW_RAM"
    assert d_both.defer_reasons == ("outside_overnight_window", "insufficient_free_ram")

    # 7. Active eval lock / foreground game / CUDA enabled -> DEFERRED_EXECUTION_GUARD
    d_guards = evaluate_drift_recalibration_trigger(
        {
            "drift_ratio": 0.010,
            "observed_rogue_norm_ratio": 1.1,
            "free_ram_gib": 15.0,
            "eval_lock_held": True,
            "foreground_game_active": True,
            "cuda_disabled_for_s5": False,
        },
        is_overnight_window=True,
    )
    assert d_guards.status == "DEFERRED_EXECUTION_GUARD"
    assert d_guards.defer_reasons == (
        "active_eval_lock_held",
        "foreground_game_preempted",
        "cpu_only_isolation_required",
    )

    # Link scheduled decision into AutoWSPCampaignLedger provenance
    ledger = AutoWSPCampaignLedger("drift-triggered-campaign")
    r_trig = ledger.record_generation(
        generation_id="G_1",
        arm_name="WSP_S5_triggered",
        nll_mean=1.48,
        base_nll_mean=1.60,
        drift_ratio=0.001,
        checkpoint_sha256=compute_state_sha256(_make_cpu_checkpoint_state(501)),
        parent_checkpoint_sha256=compute_state_sha256(_make_cpu_checkpoint_state(500)),
        trigger_decision=d_drift_sched,
    )
    assert r_trig.trigger_decision_sha256 == d_drift_sched.receipt_sha256
    assert d_drift_sched.to_dict()["drift_ratio"] == pytest.approx(d_drift_sched.drift_ratio)


def test_s4_hash_chain_tamper_detection_and_monotone_anse_enforcement() -> None:
    """(S4) Ledger rejects ANSE regressions/guard breaches and detects any hash-chain tampering."""
    ledger = AutoWSPCampaignLedger(
        campaign_id="tamper-audit-campaign",
        general_guard_rel=0.01,
        min_energy_improvement=0.0,
        baseline_anse_energy=0.0,
    )
    ck0 = compute_state_sha256(_make_cpu_checkpoint_state(200))
    ck1 = compute_state_sha256(_make_cpu_checkpoint_state(201))
    ck2 = compute_state_sha256(_make_cpu_checkpoint_state(202))
    ck3 = compute_state_sha256(_make_cpu_checkpoint_state(203))
    ck4 = compute_state_sha256(_make_cpu_checkpoint_state(204))

    # Baseline ANSE violation on G_1 (nll_mean > base_nll_mean -> positive energy >= baseline 0.0)
    with pytest.raises(AnseMonotonicityViolationError, match="baseline ANSE"):
        ledger.record_generation(
            generation_id="G_1_bad_base",
            arm_name="WSP_bad",
            nll_mean=2.05,
            base_nll_mean=2.00,
            drift_ratio=0.001,
            checkpoint_sha256=ck1,
            parent_checkpoint_sha256=ck0,
        )
    assert ledger.rejected_count == 1
    assert len(ledger.entries) == 0

    # Guard breach (> +1.0% drift) -> ANSE Maximum Pain rejection
    with pytest.raises(AnseMonotonicityViolationError, match="failed domain guard"):
        ledger.record_generation(
            generation_id="G_1_pain",
            arm_name="WSP_pain",
            nll_mean=1.80,
            base_nll_mean=2.00,
            drift_ratio=0.012,
            checkpoint_sha256=ck1,
            parent_checkpoint_sha256=ck0,
        )
    assert ledger.rejected_count == 2

    # Valid G_1, G_2, G_3 progression (including testing soft margin penalty path on G_1 at +0.9% drift)
    e1, ok1, _ = compute_wsp_anse_energy(nll_mean=1.90, base_nll_mean=2.00, drift_ratio=0.009)
    assert ok1 is True and e1 == pytest.approx(-0.05)
    ledger.record_generation(
        generation_id="G_1",
        arm_name="WSP_S1",
        nll_mean=1.90,
        base_nll_mean=2.00,
        drift_ratio=0.009,
        checkpoint_sha256=ck1,
        parent_checkpoint_sha256=ck0,
    )
    ledger.record_generation(
        generation_id="G_2",
        arm_name="WSP_S2",
        nll_mean=1.86,
        base_nll_mean=2.00,
        drift_ratio=0.003,
        checkpoint_sha256=ck2,
        parent_checkpoint_sha256=ck1,
    )

    # Non-monotone G_3 attempt (equal or higher ANSE energy than G_2's -0.14) is refused atomically
    with pytest.raises(AnseMonotonicityViolationError, match="does not strictly improve"):
        ledger.record_generation(
            generation_id="G_3_regressed",
            arm_name="WSP_S3_bad",
            nll_mean=1.87,
            base_nll_mean=2.00,
            drift_ratio=0.003,
            checkpoint_sha256=ck3,
            parent_checkpoint_sha256=ck2,
        )
    assert len(ledger.entries) == 2
    assert ledger.rejected_count == 3

    # Valid G_3 succeeds without contamination from the rejected attempt
    ledger.record_generation(
        generation_id="G_3",
        arm_name="WSP_S3",
        nll_mean=1.82,
        base_nll_mean=2.00,
        drift_ratio=0.001,
        checkpoint_sha256=ck3,
        parent_checkpoint_sha256=ck2,
    )
    ledger.record_generation(
        generation_id="G_4",
        arm_name="WSP_S4",
        nll_mean=1.79,
        base_nll_mean=2.00,
        drift_ratio=0.0005,
        checkpoint_sha256=ck4,
        parent_checkpoint_sha256=ck3,
        consume_virgin_vault=False,
    )

    snapshot = ledger.export_ledger()
    assert verify_ledger_chain(snapshot["entries"])["valid"] is True

    # Tamper vector 1: Mutate receipt field without updating receipt_sha256
    tamper_1 = copy.deepcopy(snapshot["entries"])
    tamper_1[1]["anse_energy"] = -0.999
    assert verify_generation_receipt_hash(tamper_1[1]) is False
    res_1 = verify_ledger_chain(tamper_1)
    assert res_1["valid"] is False
    assert any("receipt_sha256 verification failed" in v for v in res_1["violations"])
    with pytest.raises(LedgerChainError):
        verify_ledger_chain(tamper_1, raise_on_error=True)

    # Tamper vector 2: Re-hash entry[1] cleanly without updating entry[2].prev_receipt_sha256
    tamper_2 = copy.deepcopy(snapshot["entries"])
    tamper_2[1]["nll_mean"] = 1.845
    core_2 = {k: v for k, v in tamper_2[1].items() if k != "receipt_sha256"}
    tamper_2[1]["receipt_sha256"] = compute_payload_sha256(core_2)
    res_2 = verify_ledger_chain(tamper_2)
    assert res_2["valid"] is False
    assert any("prev_receipt_sha256 mismatch" in v for v in res_2["violations"])

    # Tamper vector 3: Full re-hash of chain sneaking in ANSE regression + broken G_1 delta/guard/split
    tamper_3 = copy.deepcopy(snapshot["entries"])
    tamper_3[2]["anse_energy"] = -0.10  # Worse than entry[1] (-0.14)!
    tamper_3[2]["delta_anse_energy"] = 0.04
    for i in range(2, len(tamper_3)):
        if i > 0:
            tamper_3[i]["prev_receipt_sha256"] = tamper_3[i - 1]["receipt_sha256"]
        if i == 3:
            tamper_3[i]["delta_anse_energy"] = tamper_3[i]["anse_energy"] - tamper_3[i - 1]["anse_energy"]
        core_i = {k: v for k, v in tamper_3[i].items() if k != "receipt_sha256"}
        tamper_3[i]["receipt_sha256"] = compute_payload_sha256(core_i)
    res_3 = verify_ledger_chain(tamper_3)
    assert res_3["valid"] is False
    assert res_3["monotone_anse"] is False
    assert any("ANSE monotonicity violated" in v for v in res_3["violations"])

    # Tamper vector 4: Broken genesis, missing/corrupt fields, and split violations
    tamper_4 = copy.deepcopy(snapshot["entries"])
    tamper_4[0]["prev_receipt_sha256"] = "1" * 64
    tamper_4[0]["delta_anse_energy"] = -0.01
    tamper_4[0]["guard_ok"] = False
    tamper_4[0]["anse_energy"] = ANSE_MAXIMUM_PAIN
    tamper_4[0]["object"] = "bad"
    tamper_4[0]["campaign_id"] = "wrong_campaign"
    tamper_4[0]["generation_index"] = 99
    tamper_4[0]["split_stride"] = 20
    tamper_4[0]["probe_skip"] = 0
    tamper_4[0]["selection_skip"] = 40
    tamper_4[1]["confirm_skip"] = tamper_4[1]["probe_skip"]  # Intra-gen overlap
    tamper_4[1]["checkpoint_sha256"] = "bad_hex"
    tamper_4[2]["campaign_id"] = ""
    tamper_4[2]["anse_energy"] = float("nan")
    res_4 = verify_ledger_chain(tamper_4, campaign_id="tamper-audit-campaign")
    assert res_4["valid"] is False
    assert len(res_4["violations"]) >= 10

    # Tamper vector 5: Reusing G_1's permanently sealed virgin_vault_skip in G_2
    tamper_5 = copy.deepcopy(snapshot["entries"])
    tamper_5[1]["probe_skip"] = tamper_5[0]["virgin_vault_skip"]
    res_5 = verify_ledger_chain(tamper_5, campaign_id="tamper-audit-campaign")
    assert res_5["valid"] is False
    assert any("permanently sealed virgin vault" in v for v in res_5["violations"])

    bad_snap = copy.deepcopy(snapshot)
    bad_snap["head_receipt_sha256"] = "a" * 64
    with pytest.raises(LedgerChainError, match="snapshot_sha256 tamper detected"):
        AutoWSPCampaignLedger.from_dict(bad_snap)

    bad_snap_alloc = copy.deepcopy(snapshot)
    bad_snap_alloc["split_allocator_state"] = {"bad": True}
    core_alloc_bad = {k: v for k, v in bad_snap_alloc.items() if k != "snapshot_sha256"}
    bad_snap_alloc["snapshot_sha256"] = compute_payload_sha256(core_alloc_bad)
    with pytest.raises(LedgerChainError, match="split_allocator_state"):
        AutoWSPCampaignLedger.from_dict(bad_snap_alloc)

    bad_snap_entries = copy.deepcopy(snapshot)
    bad_snap_entries["entries"] = "not_a_list"
    core_ent_bad = {k: v for k, v in bad_snap_entries.items() if k != "snapshot_sha256"}
    bad_snap_entries["snapshot_sha256"] = compute_payload_sha256(core_ent_bad)
    with pytest.raises(LedgerChainError, match="must be a sequence"):
        AutoWSPCampaignLedger.from_dict(bad_snap_entries)

    bad_snap_resigned = copy.deepcopy(snapshot)
    bad_snap_resigned["head_receipt_sha256"] = "b" * 64
    core_snap = {k: v for k, v in bad_snap_resigned.items() if k != "snapshot_sha256"}
    bad_snap_resigned["snapshot_sha256"] = compute_payload_sha256(core_snap)
    with pytest.raises(LedgerChainError, match="head_receipt_sha256 mismatch"):
        AutoWSPCampaignLedger.from_dict(bad_snap_resigned)


def test_s5_validation_errors_and_security_stub_compliance_gates() -> None:
    """(S5) Validation edge cases plus ZeroStubAudit, PRSecurityScanner, lateral movement, and gatekeeper checks."""
    # 1. Helper & tensor validation branches
    with pytest.raises(TypeError):
        compute_state_sha256("not_a_dict")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        compute_state_sha256({})
    with pytest.raises(TypeError):
        compute_state_sha256({"w": [1.0]})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="non-finite"):
        compute_state_sha256({"w": torch.tensor([float("nan")])})
    assert len(compute_state_sha256({"empty": torch.empty(0), "u8": torch.ones(2, dtype=torch.uint8), "bf16": torch.ones(2, dtype=torch.bfloat16), "f64": torch.ones(2, dtype=torch.float64)})) == 64

    with pytest.raises(TypeError):
        compute_payload_sha256({"bad": object()})
    with pytest.raises(ValueError, match="non-finite"):
        compute_payload_sha256({"bad": float("inf")})

    with pytest.raises(ValueError, match="at least 2"):
        compute_expert_norm_ratio([1.0])
    with pytest.raises(ValueError, match="at least 2"):
        compute_expert_norm_ratio({"e0": 1.0})
    with pytest.raises(TypeError):
        compute_expert_norm_ratio("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty floating-point"):
        compute_expert_norm_ratio([torch.ones(2, dtype=torch.int64), torch.ones(2)])
    with pytest.raises(ValueError, match="non-finite"):
        compute_expert_norm_ratio([torch.tensor([float("nan")]), torch.ones(2)])
    assert compute_expert_norm_ratio([0.0, 0.0])[0] == 0.0
    assert compute_expert_norm_ratio([1.0, 2.0, 3.0, 4.0])[0] == pytest.approx(4.0 / 2.5)

    with pytest.raises(ValueError):
        verify_shard_rows_disjoint([])
    with pytest.raises(TypeError):
        verify_shard_rows_disjoint(["bad"])  # type: ignore[list-item]
    with pytest.raises(SplitContaminationError, match="Duplicate split name"):
        verify_shard_rows_disjoint([("a", [{"dataset_id": "d", "row": 0}]), ("a", [{"dataset_id": "d", "row": 1}])])
    with pytest.raises(ValueError, match="non-empty sequence"):
        verify_shard_rows_disjoint([("a", [])])
    with pytest.raises(TypeError):
        verify_shard_rows_disjoint([("a", ["bad"])])  # type: ignore[list-item]
    with pytest.raises(SplitContaminationError, match="duplicate"):
        verify_shard_rows_disjoint([("a", [{"dataset_id": "d", "row": 0}, {"dataset_id": "d", "row": 0}])])
    with pytest.raises(SplitContaminationError, match="Shard contamination"):
        verify_shard_rows_disjoint(
            [("a", [{"dataset_id": "d", "row": 0}]), ("b", [{"dataset_id": "d", "row": 0}])]
        )

    # 2. Allocator validation & tamper branches
    with pytest.raises(SplitContaminationError, match="frozen window"):
        DisjointSplitAllocator(start_skip=10, split_stride=40, frozen_skip_floor=40)
    alloc = DisjointSplitAllocator(start_skip=40, split_stride=40, frozen_skip_floor=40)
    with pytest.raises(ValueError, match="all four"):
        alloc.allocate_generation("G_partial", probe_skip=40)
    with pytest.raises(SplitContaminationError, match="Intra-generation"):
        alloc.allocate_generation("G_intra", probe_skip=40, selection_skip=60, confirm_skip=120, virgin_vault_skip=160)
    q1 = alloc.allocate_generation("G_1")
    with pytest.raises(SplitContaminationError, match="already has reserved"):
        alloc.allocate_generation("G_1")
    with pytest.raises(SplitContaminationError, match="Unknown generation_id"):
        alloc.seal_virgin_vault("G_missing")
    with pytest.raises(SplitContaminationError, match="frozen window"):
        alloc.seal_consumed_skip_offset(0)

    # Tampered SplitQuadruple digest caught in verify_all_splits_disjoint
    alloc._allocations["G_1"] = replace(q1, allocation_sha256="f" * 64)
    with pytest.raises(SplitContaminationError, match="Tampered SplitQuadruple"):
        alloc.verify_all_splits_disjoint()
    alloc._allocations["G_1"] = q1

    st = alloc.export_state()
    with pytest.raises(TypeError):
        DisjointSplitAllocator.from_state("bad")  # type: ignore[arg-type]
    bad_st = copy.deepcopy(st)
    bad_st["cursor"] = 999
    with pytest.raises(SplitContaminationError, match="state_sha256"):
        DisjointSplitAllocator.from_state(bad_st)
    bad_seal_st = copy.deepcopy(st)
    bad_seal_st["sealed_vaults"] = [{"receipt_sha256": "0" * 64, "virgin_vault_skip": 999}]
    bad_seal_st_core = {k: v for k, v in bad_seal_st.items() if k != "state_sha256"}
    bad_seal_st["state_sha256"] = compute_payload_sha256(bad_seal_st_core)
    with pytest.raises(VaultSealedError, match="Tampered virgin-vault seal"):
        DisjointSplitAllocator.from_state(bad_seal_st)

    # 3. Trigger & Ledger validation branches
    with pytest.raises(TypeError):
        evaluate_drift_recalibration_trigger("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        evaluate_drift_recalibration_trigger({"drift_ratio": 0.01, "observed_rogue_norm_ratio": 1.0, "free_ram_gib": 12.0}, is_overnight_window="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="drift_ratio"):
        evaluate_drift_recalibration_trigger({"observed_rogue_norm_ratio": 1.0, "free_ram_gib": 12.0})
    with pytest.raises(ValueError, match="identical domain keys"):
        evaluate_drift_recalibration_trigger({"domain_nlls": {"E0_a": 1.5}, "base_domain_nlls": {"E1_b": 1.5}, "observed_rogue_norm_ratio": 1.0, "free_ram_gib": 12.0})
    with pytest.raises(ValueError, match="general domain key"):
        evaluate_drift_recalibration_trigger({"domain_nlls": {"E1_b": 1.5}, "base_domain_nlls": {"E1_b": 1.5}, "observed_rogue_norm_ratio": 1.0, "free_ram_gib": 12.0})
    with pytest.raises(ValueError, match="observed_rogue_norm_ratio"):
        evaluate_drift_recalibration_trigger({"drift_ratio": 0.01, "free_ram_gib": 12.0})
    with pytest.raises(ValueError, match="free_ram_gib"):
        evaluate_drift_recalibration_trigger({"drift_ratio": 0.01, "observed_rogue_norm_ratio": 1.0})
    with pytest.raises(TypeError, match="real number"):
        evaluate_drift_recalibration_trigger(
            {"drift_ratio": 0.01, "observed_rogue_norm_ratio": 1.0, "free_ram_gib": 12.0},
            general_drift_trigger_rel=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="> 0.0"):
        evaluate_drift_recalibration_trigger(
            {"drift_ratio": 0.01, "observed_rogue_norm_ratio": 1.0, "free_ram_gib": 12.0},
            general_drift_trigger_rel=0.0,
        )
    with pytest.raises(ValueError, match=">= 1"):
        DisjointSplitAllocator(split_stride=0)
    with pytest.raises(TypeError):
        evaluate_drift_recalibration_trigger({"drift_ratio": 0.01, "observed_rogue_norm_ratio": 1.0, "free_ram_gib": 12.0, "eval_lock_held": "no"})
    with pytest.raises(ValueError, match="printable"):
        AutoWSPCampaignLedger(" \n ")
    with pytest.raises(TypeError):
        AutoWSPCampaignLedger("ok", split_allocator="bad")  # type: ignore[arg-type]

    lgr = AutoWSPCampaignLedger("edge-val")
    assert lgr.general_guard_rel == pytest.approx(0.01)
    assert verify_generation_receipt_hash({"receipt_sha256": "bad"}) is False
    assert verify_generation_receipt_hash({"receipt_sha256": "0" * 64, "object": "bad"}) is False
    assert verify_generation_receipt_hash({"receipt_sha256": "0" * 64, "object": GENERATION_RECEIPT_OBJECT, "schema": SCHEMA_VERSION, "bad": float("nan")}) is False
    with pytest.raises(TypeError):
        verify_generation_receipt_hash("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        verify_ledger_chain("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        AutoWSPCampaignLedger.from_dict("bad")  # type: ignore[arg-type]
    with pytest.raises(LedgerChainError, match="object or schema"):
        AutoWSPCampaignLedger.from_dict({"object": "bad"})
    with pytest.raises(LedgerChainError, match="Missing or malformed snapshot_sha256"):
        AutoWSPCampaignLedger.from_dict({"object": LEDGER_SNAPSHOT_OBJECT, "schema": SCHEMA_VERSION})

    valid_sha = "c" * 64
    with pytest.raises(TypeError, match="SplitQuadruple"):
        lgr.record_generation(
            generation_id="G_1",
            arm_name="a",
            split_quadruple="bad",  # type: ignore[arg-type]
            nll_mean=1.8,
            base_nll_mean=2.0,
            drift_ratio=0.001,
            checkpoint_sha256=valid_sha,
            parent_checkpoint_sha256=valid_sha,
        )
    with pytest.raises(TypeError, match="trigger_decision"):
        lgr.record_generation(
            generation_id="G_1",
            arm_name="a",
            nll_mean=1.8,
            base_nll_mean=2.0,
            drift_ratio=0.001,
            checkpoint_sha256=valid_sha,
            parent_checkpoint_sha256=valid_sha,
            trigger_decision="bad",  # type: ignore[arg-type]
        )
    lgr.record_generation(
        generation_id="G_1",
        arm_name="a",
        nll_mean=1.8,
        base_nll_mean=2.0,
        drift_ratio=0.001,
        checkpoint_sha256=valid_sha,
        parent_checkpoint_sha256=valid_sha,
        trigger_decision={"receipt_sha256": valid_sha},
    )
    with pytest.raises(LedgerChainError, match="already recorded"):
        lgr.record_generation(
            generation_id="G_1",
            arm_name="a",
            nll_mean=1.7,
            base_nll_mean=2.0,
            drift_ratio=0.001,
            checkpoint_sha256=valid_sha,
            parent_checkpoint_sha256=valid_sha,
        )
    with pytest.raises(SplitContaminationError, match="generation_id"):
        lgr.record_generation(
            generation_id="G_2",
            arm_name="b",
            split_quadruple=q1,
            nll_mean=1.7,
            base_nll_mean=2.0,
            drift_ratio=0.001,
            checkpoint_sha256=valid_sha,
            parent_checkpoint_sha256=valid_sha,
        )
    with pytest.raises(SplitContaminationError, match="split_stride"):
        lgr.record_generation(
            generation_id="G_2",
            arm_name="b",
            split_quadruple=replace(q1, generation_id="G_2", split_stride=20),
            nll_mean=1.7,
            base_nll_mean=2.0,
            drift_ratio=0.001,
            checkpoint_sha256=valid_sha,
            parent_checkpoint_sha256=valid_sha,
        )
    with pytest.raises(SplitContaminationError, match="Forged SplitQuadruple"):
        lgr.record_generation(
            generation_id="G_2",
            arm_name="b",
            split_quadruple=replace(q1, generation_id="G_2", allocation_sha256=valid_sha),
            nll_mean=1.7,
            base_nll_mean=2.0,
            drift_ratio=0.001,
            checkpoint_sha256=valid_sha,
            parent_checkpoint_sha256=valid_sha,
        )
    alloc_other = DisjointSplitAllocator(start_skip=240, split_stride=40, frozen_skip_floor=40)
    lgr.split_allocator.allocate_generation("G_2")
    q2_mismatch = alloc_other.allocate_generation("G_2")
    with pytest.raises(SplitContaminationError, match="mismatches allocator reservation"):
        lgr.record_generation(
            generation_id="G_2",
            arm_name="b",
            split_quadruple=q2_mismatch,
            nll_mean=1.7,
            base_nll_mean=2.0,
            drift_ratio=0.001,
            checkpoint_sha256=valid_sha,
            parent_checkpoint_sha256=valid_sha,
        )

    # 4. AGPL-3.0 header, ZeroStubAudit (CI-safe fallback + optional gwaya), PRSecurityScanner, lateral movement, FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn25", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn25", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/auto_wsp_campaign_scheduler.py", TARGET_MODULE_PATH),
        ("tests/test_auto_wsp_campaign_scheduler_scenarios.py", TEST_FILE_PATH),
    ):
        src_text = path_obj.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in src_text[:300], f"Missing AGPL-3.0 header in {rel}"

        assert fallback_auditor.check_file(path_obj) == []
        assert fallback_auditor.audit_python_code(src_text).is_clean is True

        if ZeroStubAudit is not None:
            stub_res = ZeroStubAudit().audit_python_code(src_text)
            assert stub_res.is_clean, f"ZeroStubAudit violations in {rel}: {stub_res.violations}"
            assert stub_res.penalty_energy == 0.0

        assert scanner.scan_python_ast_security(rel, src_text) == []
        assert scanner.scan_text_for_payment_solicitation(src_text, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(src_text, rel) == []
        assert scanner.scan_unicode_trojan_source(src_text, rel) == []
        assert lateral_guard.audit_source_for_lateral_movement(src_text, rel) == []

        synthetic_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(
            f"+{line}" for line in src_text.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []
        changed_files.append(
            {
                "filename": rel,
                "status": "added",
                "additions": len(src_text.splitlines()),
                "patch": "",
                "content": src_text,
            }
        )

    sample_receipt_json = json.dumps(
        {
            "object": GENERATION_RECEIPT_OBJECT,
            "nll_mean": 1.88,
            "drift_ratio": 0.0005,
            "anse_energy": -0.27,
            "virgin_vault_sealed": True,
        },
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+RECEIPT = {sample_receipt_json!r}"
        )
        == []
    )

    pr_audit = scanner.audit_pull_request(
        pr_number=1125,
        pr_title="feat(swarm): autonomous open-weight WSP campaign ledger and drift-triggered recalibration scheduler",
        pr_body="Implements NEURAL-CTRL-DYN-25 campaign ledger, disjoint split allocator, drift trigger, and scenario tests.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): autonomous open-weight WSP campaign ledger and drift-triggered recalibration scheduler"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
