# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control forgery tests for #393 (IP-19): ZK-OPoI v5 witness & step binding."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import math
import pytest

from hivemind.swarm.s36_01_zk_opoi_v5_int8_verifier import ZKOPoIV5Int8TransitionVerifier
from hivemind.swarm.zk_opoi_v5_circuit import (
    ZKOPoIV5FixedPointCircuit,
    ZKOPoIV5ProofReceipt,
    _derive_fiat_shamir_challenge,
    _field_mod,
)


def test_zk_opoi_v5_circuit_rejects_forged_zero_step_or_self_signed_receipt() -> None:
    """#393 / IP-19: A forged receipt constructed without running honest steps must fail verification."""
    circuit = ZKOPoIV5FixedPointCircuit(quantization_bits=8)
    shard_hash = hashlib.sha256(b"official-shard-s30-01").hexdigest()
    theta0 = [math.sin(i * 0.1) * 0.5 for i in range(64)]
    grads = [[math.cos((i + s) * 0.07) * 0.08 for i in range(64)] for s in range(4)]

    honest_receipt, _ = circuit.prove_quantized_diloco_transition(
        node_id="bee-node-honest",
        assigned_shard_sha256=shard_hash,
        theta0_float=theta0,
        step_gradients_float=grads,
    )

    # Forgery 1: Lazy worker re-derives Fiat-Shamir challenge with forged polynomial evals & fake Merkle root
    fake_merkle = hashlib.sha256(b"forged-zero-work-merkle-root").hexdigest()
    fake_theta_final_comm = hashlib.sha256(b"forged-final-weights").hexdigest()
    forged_challenge = _derive_fiat_shamir_challenge(
        f"bee-node-lazy:{shard_hash}:{honest_receipt.theta0_commitment}:{fake_theta_final_comm}:{fake_merkle}"
    )
    forged_p0 = 123456789
    forged_pg = 9876543
    forged_pf = _field_mod(forged_p0 - forged_pg)

    forged_receipt = ZKOPoIV5ProofReceipt(
        node_id="bee-node-lazy",
        step_count_h=4,
        quantization_bits=8,
        assigned_shard_sha256=shard_hash,
        theta0_commitment=honest_receipt.theta0_commitment,
        theta_final_commitment=fake_theta_final_comm,
        fiat_shamir_challenge=forged_challenge,
        theta0_poly_eval=forged_p0,
        grad_accum_poly_eval=forged_pg,
        theta_final_poly_eval=forged_pf,
        delta_l2_norm_fp=100,
        max_allowed_l2_norm_fp=honest_receipt.max_allowed_l2_norm_fp,
        spectral_bound_fp=10,
        max_allowed_spectral_fp=honest_receipt.max_allowed_spectral_fp,
        witness_merkle_root=fake_merkle,
        proof_size_bytes=512,
    )
    verdict_forged = circuit.verify_receipt_cpu_o1(
        forged_receipt,
        expected_shard_sha256=shard_hash,
        expected_theta0_commitment=honest_receipt.theta0_commitment,
    )
    assert verdict_forged.valid is False

    # Forgery 2: Receipt claiming step_count_h = 0 or zero delta norm
    zero_step_receipt = replace(honest_receipt, step_count_h=0)
    assert (
        circuit.verify_receipt_cpu_o1(
            zero_step_receipt,
            expected_shard_sha256=shard_hash,
            expected_theta0_commitment=honest_receipt.theta0_commitment,
        ).valid
        is False
    )

    # Forgery 3: Prover rejects NaN/Inf or zero-gradient no-op steps
    with pytest.raises(ValueError):
        circuit.prove_quantized_diloco_transition(
            node_id="bee-nan",
            assigned_shard_sha256=shard_hash,
            theta0_float=theta0,
            step_gradients_float=[[math.nan] * 64],
        )
    with pytest.raises(ValueError):
        circuit.prove_quantized_diloco_transition(
            node_id="bee-zero",
            assigned_shard_sha256=shard_hash,
            theta0_float=theta0,
            step_gradients_float=[[0.0] * 64],
        )


def test_s36_01_verifier_rejects_zero_steps_and_tampered_sketch() -> None:
    """#393 / IP-19: ZKOPoIV5Int8TransitionVerifier must reject inner_steps_h <= 0 and non-finite inputs."""
    verifier = ZKOPoIV5Int8TransitionVerifier()
    shard = hashlib.sha256(b"shard-s36").hexdigest()
    theta0 = [0.1, -0.2, 0.3, -0.1, 0.05, 0.2, -0.15, 0.08]
    delta = [0.02, -0.01, 0.03, 0.01, -0.02, 0.01, 0.02, -0.01]

    with pytest.raises(ValueError):
        verifier.generate_proof(
            worker_id="w1",
            workunit_id="wu-0",
            shard_sha256=shard,
            theta0=theta0,
            delta_weights=delta,
            inner_steps_h=0,
        )

    receipt = verifier.generate_proof(
        worker_id="w1",
        workunit_id="wu-1",
        shard_sha256=shard,
        theta0=theta0,
        delta_weights=delta,
        inner_steps_h=10,
    )
    tampered_steps = replace(receipt, inner_steps_h=0)
    report = verifier.verify_proof(
        tampered_steps,
        expected_shard_sha256=shard,
        expected_theta0_sha256=receipt.theta0_sha256,
    )
    assert report.verified is False
