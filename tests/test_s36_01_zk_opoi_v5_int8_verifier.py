"""Unit tests for HM-S36-01 Fixed-Point INT8/INT4 ZK-OPoI v5 Transition Verifier."""

from __future__ import annotations

from dataclasses import replace

import pytest

from hivemind.swarm.s36_01_zk_opoi_v5_int8_verifier import ZKOPoIV5Int8TransitionVerifier


def test_zk_opoi_v5_int8_proof_generation_and_under_5ms_verification() -> None:
    verifier = ZKOPoIV5Int8TransitionVerifier()
    theta0 = [0.03 * ((i % 11) - 5) for i in range(256)]
    delta = [0.008 * ((i % 7) - 3) for i in range(256)]
    receipt = verifier.generate_proof(
        worker_id="miner-rtx2070-01",
        workunit_id="wu-s36-001",
        shard_sha256="official_shard_sha256_001",
        theta0=theta0,
        delta_weights=delta,
        inner_steps_h=500,
        quant_bits=8,
    )
    report = verifier.verify_proof(
        receipt,
        expected_shard_sha256="official_shard_sha256_001",
        expected_theta0_sha256=receipt.theta0_sha256,
    )
    assert report.verified is True
    assert report.within_5ms_sla is True
    assert report.within_32kb_bound is True
    assert report.proof_bytes < 32_768
    assert report.rejection_reason is None


def test_zk_opoi_v5_rejects_tampered_shard_bilinear_or_norm_bounds() -> None:
    verifier = ZKOPoIV5Int8TransitionVerifier(max_l2_norm=5.0, max_linf_norm=1.0)
    theta0 = [0.02 * i for i in range(64)]
    delta = [0.01 * ((i % 5) - 2) for i in range(64)]
    receipt = verifier.generate_proof(
        worker_id="miner-02",
        workunit_id="wu-s36-002",
        shard_sha256="shard_abc",
        theta0=theta0,
        delta_weights=delta,
    )

    # 1. Wrong shard hash
    rep_shard = verifier.verify_proof(
        receipt,
        expected_shard_sha256="wrong_shard_hash",
        expected_theta0_sha256=receipt.theta0_sha256,
    )
    assert rep_shard.verified is False
    assert rep_shard.rejection_reason == "shard_sha256_mismatch"

    # 2. Tampered bilinear equation
    forged_bilinear = replace(receipt, bilinear_lhs_mod_p=receipt.bilinear_lhs_mod_p + 1)
    rep_bilin = verifier.verify_proof(
        forged_bilinear,
        expected_shard_sha256="shard_abc",
        expected_theta0_sha256=receipt.theta0_sha256,
    )
    assert rep_bilin.verified is False
    assert rep_bilin.rejection_reason == "bilinear_freivalds_equation_failed"

    # 3. Poisoned L2 norm exceeding safety bound
    poisoned_norm = replace(receipt, delta_l2_norm=99.0)
    rep_norm = verifier.verify_proof(
        poisoned_norm,
        expected_shard_sha256="shard_abc",
        expected_theta0_sha256=receipt.theta0_sha256,
    )
    assert rep_norm.verified is False
    assert rep_norm.rejection_reason == "delta_l2_norm_out_of_bounds"

    with pytest.raises(ValueError, match="quant_bits"):
        verifier.quantize_to_int8([0.1, 0.2], quant_bits=6)
