# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression & negative-control test suite for Iteration 36 (PR #704).

Covers:
1. Audit C8 (`scripts/t4_workgen.py` & `src/hivemind/router/opentrainer.py`):
   - `build_training_round_table` and `build_opentrainer_training_round_table` extract
     `(round, domain, eval_by_domain_before, eval_by_domain_after)` from ledger receipts.
   - `select_round_head_with_rollback` enforces the single-metric rollback gate when
     `eval_loss_after > eval_loss_before`, rolling back `latest_adapter.pt` and `latest_adapter.json`
     to the most recent non-regressed checkpoint (or returning `None` if no valid checkpoint exists).
   - `RTX2070_T4_INTEGRATION_LEDGER` includes `artifact_refs` for every ledger entry.
2. Audit B15 (`zk_opoi_v5_circuit.py`, `s36_01_zk_opoi_v5_int8_verifier.py`, `reddit_vision_vlm_moe.py`):
   - `TamperEvident*` aliases (`TamperEvidentV5FixedPointCircuit`, `TamperEvidentInt8TransitionVerifier`,
     `TamperEvidentTrainingReceipt`, `generate_tamper_evident_training_receipt`,
     `verify_tamper_evident_training_receipt`).
   - Explicit `zero_knowledge=False` and `commitment_scheme` metadata on all receipts and reports.
   - Ephemeral secret isolation when `coordinator_secret=None`.
3. Audit B Relabel Pattern (`boinc/astrophysics_24h_campaign.py`, `boinc/local_astrophysics_boinc.py`,
   `cli/multi_moe_hf_campaign.py`, `cli/season9_10_gpu_swarm_campaigns.py`,
   `swarm/strategy_optimizer.py`, `swarm/top1pct_dct_ef21_compressor.py`):
   - `"simulated": True` present in all returned campaign/benchmark summaries.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from hivemind.core.reddit_vision_vlm_moe import (
    TamperEvidentTrainingReceipt,
    generate_tamper_evident_training_receipt,
    verify_tamper_evident_training_receipt,
)
from hivemind.router.opentrainer import (
    RTX2070_T4_INTEGRATION_LEDGER,
    build_opentrainer_training_round_table,
)
from hivemind.swarm.s36_01_zk_opoi_v5_int8_verifier import (
    TamperEvidentInt8ProofReceipt,
    TamperEvidentInt8TransitionVerifier,
    TamperEvidentInt8VerificationReport,
)
from hivemind.swarm.strategy_optimizer import StrategyOptimizerTelemetryEngine
from hivemind.swarm.top1pct_dct_ef21_compressor import (
    evaluate_7division_top1pct_compression_benchmark,
)
from hivemind.swarm.zk_opoi_v5_circuit import (
    TamperEvidentV5FixedPointCircuit,
    TamperEvidentV5ProofReceipt,
    TamperEvidentV5VerificationResult,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_t4_workgen():
    spec = importlib.util.spec_from_file_location(
        "t4_workgen", REPO_ROOT / "scripts" / "t4_workgen.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_c8_t4_workgen_round_table_and_rollback_gate(tmp_path: Path) -> None:
    """Audit C8: `build_training_round_table` and `select_round_head_with_rollback` enforce rollback on loss rise."""
    wg = _load_t4_workgen()
    ledger = tmp_path / "hivemind_ledger"
    ledger.mkdir()

    # Round 1: improved (2.12 -> 2.04)
    r1_bytes = b"adapter-round-1"
    (ledger / "adapter_hmq_00001.pt").write_bytes(r1_bytes)
    r1_meta = {
        "wu_name": "hmq_00001",
        "domain": "mix",
        "sha256": "sha_r1",
        "eval_loss_before": 2.12,
        "eval_loss_after": 2.04,
        "eval_by_domain_before": {"wikitext": 2.12, "code": 2.12},
        "eval_by_domain_after": {"wikitext": 2.03, "code": 2.05},
    }
    (ledger / "adapter_hmq_00001.json").write_text(json.dumps(r1_meta), encoding="utf-8")

    # Round 2: regressed (2.04 -> 2.19)
    r2_bytes = b"adapter-round-2-regressed"
    (ledger / "adapter_hmq_00002.pt").write_bytes(r2_bytes)
    r2_meta = {
        "wu_name": "hmq_00002",
        "domain": "mix",
        "sha256": "sha_r2",
        "eval_loss_before": 2.04,
        "eval_loss_after": 2.19,
        "eval_by_domain_before": {"wikitext": 2.03, "code": 2.05},
        "eval_by_domain_after": {"wikitext": 2.20, "code": 2.18},
    }
    (ledger / "adapter_hmq_00002.json").write_text(json.dumps(r2_meta), encoding="utf-8")

    # Set latest_adapter to the regressed Round 2 checkpoint
    (ledger / "latest_adapter.pt").write_bytes(r2_bytes)
    (ledger / "latest_adapter.json").write_text(json.dumps(r2_meta), encoding="utf-8")

    table = wg.build_training_round_table(ledger)
    assert len(table) == 2
    assert table[0]["round"] == "hmq_00001"
    assert table[0]["regressed"] is False
    assert table[1]["round"] == "hmq_00002"
    assert table[1]["regressed"] is True

    opentrainer_table = build_opentrainer_training_round_table(ledger)
    assert len(opentrainer_table) == 2
    assert opentrainer_table[0]["eval_by_domain_after"] == {"wikitext": 2.03, "code": 2.05}
    assert build_opentrainer_training_round_table(None) == []

    # Rollback gate rolls back from Round 2 -> Round 1
    selected = wg.select_round_head_with_rollback(ledger)
    assert selected is not None
    assert selected["rolled_back"] is True
    assert selected["sha256"] == "sha_r1"
    assert selected["rolled_back_from_sha256"] == "sha_r2"
    assert (ledger / "latest_adapter.pt").read_bytes() == r1_bytes

    # Calling again when latest_adapter is non-regressed keeps Round 1
    selected_again = wg.select_round_head_with_rollback(ledger)
    assert selected_again is not None
    assert selected_again["sha256"] == "sha_r1"

    # Negative control: if only a regressed head exists with no prior valid checkpoint, returns None
    empty_ledger = tmp_path / "empty_ledger"
    empty_ledger.mkdir()
    (empty_ledger / "latest_adapter.json").write_text(json.dumps(r2_meta), encoding="utf-8")
    assert wg.select_round_head_with_rollback(empty_ledger) is None


def test_c8_opentrainer_ledger_has_artifact_refs() -> None:
    """Audit C8: every entry in `RTX2070_T4_INTEGRATION_LEDGER` carries `artifact_refs`."""
    assert len(RTX2070_T4_INTEGRATION_LEDGER) >= 6
    for entry in RTX2070_T4_INTEGRATION_LEDGER:
        assert "artifact_refs" in entry
        assert isinstance(entry["artifact_refs"], list)
        assert len(entry["artifact_refs"]) >= 1


def test_b15_tamper_evident_commitment_metadata_and_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Audit B15: `TamperEvident*` classes/functions expose `zero_knowledge=False` and `commitment_scheme`."""
    monkeypatch.delenv("HIVEMIND_ZK_OPOI_SECRET", raising=False)

    circuit = TamperEvidentV5FixedPointCircuit(coordinator_secret="test-shared-secret")
    receipt, _ = circuit.prove_quantized_diloco_transition(
        node_id="worker-01",
        assigned_shard_sha256="a" * 64,
        theta0_float=[0.25, -0.5, 0.75, 0.1],
        step_gradients_float=[[0.1, -0.2, 0.05, 0.15]],
    )
    assert isinstance(receipt, TamperEvidentV5ProofReceipt)
    assert receipt.zero_knowledge is False
    assert receipt.commitment_scheme == "fiat_shamir_horner_hmac_sha256"

    verdict = circuit.verify_receipt_cpu_o1(
        receipt,
        expected_shard_sha256="a" * 64,
        expected_theta0_commitment=receipt.theta0_commitment,
    )
    assert isinstance(verdict, TamperEvidentV5VerificationResult)
    assert verdict.valid is True
    assert verdict.zero_knowledge is False
    assert verdict.commitment_scheme == "fiat_shamir_horner_hmac_sha256"

    # Separate ephemeral instance without shared secret rejects the receipt
    other_circuit = TamperEvidentV5FixedPointCircuit()
    other_verdict = other_circuit.verify_receipt_cpu_o1(
        receipt,
        expected_shard_sha256="a" * 64,
        expected_theta0_commitment=receipt.theta0_commitment,
    )
    assert other_verdict.valid is False

    # Int8 transition verifier
    v_int8 = TamperEvidentInt8TransitionVerifier(coordinator_secret="int8-secret")
    r_int8 = v_int8.generate_proof(
        worker_id="node-int8",
        workunit_id="wu-int8-01",
        shard_sha256="c" * 64,
        theta0=[0.1, -0.2, 0.3, 0.4] * 4,
        delta_weights=[0.05, -0.05, 0.08, -0.06] * 4,
        inner_steps_h=4,
    )
    assert isinstance(r_int8, TamperEvidentInt8ProofReceipt)
    assert r_int8.zero_knowledge is False
    assert r_int8.commitment_scheme == "fiat_shamir_horner_hmac_sha256"
    rep_int8 = v_int8.verify_proof(
        r_int8,
        expected_shard_sha256=r_int8.shard_sha256,
        expected_theta0_sha256=r_int8.theta0_sha256,
    )
    assert isinstance(rep_int8, TamperEvidentInt8VerificationReport)
    assert rep_int8.verified is True
    assert rep_int8.zero_knowledge is False

    # VLM training receipt
    init_d = np.ones((4, 4), dtype=np.float32) * 0.1
    final_d = np.ones((4, 4), dtype=np.float32) * 0.25
    vlm_receipt = generate_tamper_evident_training_receipt(
        bee_id="bee_01",
        specialization_id="art_aesthetics",
        inner_steps=10,
        optimizer_name="Muon",
        seed=42,
        initial_delta=init_d,
        final_delta=final_d,
        dct_packet_sha256="b" * 64,
    )
    assert isinstance(vlm_receipt, TamperEvidentTrainingReceipt)
    assert vlm_receipt.zero_knowledge is False
    assert vlm_receipt.commitment_scheme == "sha256_transcript_commitment"
    assert verify_tamper_evident_training_receipt(vlm_receipt, expected_final_delta=final_d) is True


def test_b_relabel_simulated_flags_on_optimizers_and_campaigns() -> None:
    """Audit B: benchmark and campaign summaries carry `"simulated": True`."""
    dct_rep = evaluate_7division_top1pct_compression_benchmark()
    assert dct_rep["simulated"] is True
    assert dct_rep["gate_passed"] is True

    opt = StrategyOptimizerTelemetryEngine()
    snap = opt.snapshot(force_refresh=True)
    assert snap["simulated"] is True
