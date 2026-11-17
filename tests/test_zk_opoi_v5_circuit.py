"""Unit tests for HM-S30-01 ZK-OPoI v5 Fixed-Point INT8/INT4 Verifiable DiLoCo Circuit."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import math

from hivemind.swarm.zk_opoi_v5_circuit import (
    MAX_CPU_VERIFY_MS,
    MAX_PROOF_SIZE_BYTES,
    ZKOPoIV5FixedPointCircuit,
)


def test_honest_zk_opoi_v5_proof_verifies_in_sub_10ms_and_sub_32kb() -> None:
    circuit = ZKOPoIV5FixedPointCircuit(quantization_bits=8)
    shard_hash = hashlib.sha256(b"official-shard-s30-01").hexdigest()
    theta0 = [math.sin(i * 0.1) * 0.5 for i in range(128)]
    grads = [[math.cos((i + s) * 0.07) * 0.08 for i in range(128)] for s in range(8)]

    receipt, _ = circuit.prove_quantized_diloco_transition(
        node_id="bee-node-01",
        assigned_shard_sha256=shard_hash,
        theta0_float=theta0,
        step_gradients_float=grads,
    )
    verdict = circuit.verify_receipt_cpu_o1(
        receipt,
        expected_shard_sha256=shard_hash,
        expected_theta0_commitment=receipt.theta0_commitment,
    )

    assert verdict.valid is True
    assert verdict.reason == "zk_opoi_v5_verified_o1_cpu"
    assert verdict.verify_cpu_ms < MAX_CPU_VERIFY_MS
    assert verdict.proof_size_bytes < MAX_PROOF_SIZE_BYTES


def test_100pct_rejection_of_forged_shard_tampered_witness_and_norm_explosion() -> None:
    circuit = ZKOPoIV5FixedPointCircuit(quantization_bits=4)
    shard_hash = hashlib.sha256(b"official-shard-s30-01").hexdigest()
    theta0 = [0.25 * ((i % 5) - 2) for i in range(64)]
    grads = [[0.05 * ((i % 3) - 1) for i in range(64)] for _ in range(4)]

    receipt, _ = circuit.prove_quantized_diloco_transition(
        node_id="bee-node-02",
        assigned_shard_sha256=shard_hash,
        theta0_float=theta0,
        step_gradients_float=grads,
    )

    # 1. Forged dataset shard SHA-256
    v_bad_shard = circuit.verify_receipt_cpu_o1(
        receipt,
        expected_shard_sha256=hashlib.sha256(b"wrong-shard").hexdigest(),
        expected_theta0_commitment=receipt.theta0_commitment,
    )
    assert v_bad_shard.valid is False
    assert v_bad_shard.shard_commitment_ok is False

    # 2. Forged R1CS polynomial evaluation (lazy/poisoned worker tampering theta_final_poly_eval)
    forged_r1cs = replace(receipt, theta_final_poly_eval=(receipt.theta_final_poly_eval + 1337))
    v_bad_r1cs = circuit.verify_receipt_cpu_o1(
        forged_r1cs,
        expected_shard_sha256=shard_hash,
        expected_theta0_commitment=receipt.theta0_commitment,
    )
    assert v_bad_r1cs.valid is False
    assert v_bad_r1cs.r1cs_transition_ok is False

    # 3. Spectral / L2 poisoning norm bound violation
    forged_norm = replace(receipt, spectral_bound_fp=receipt.max_allowed_spectral_fp + 500)
    v_bad_norm = circuit.verify_receipt_cpu_o1(
        forged_norm,
        expected_shard_sha256=shard_hash,
        expected_theta0_commitment=receipt.theta0_commitment,
    )
    assert v_bad_norm.valid is False
    assert v_bad_norm.norm_bounds_ok is False
