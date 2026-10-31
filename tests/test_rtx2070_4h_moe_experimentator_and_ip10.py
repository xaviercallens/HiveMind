# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for PR-IP-10 (`IP-20`, `IP-34`, `IP-35`, `IP-36`, `IP-37`) and the 4-Hour RTX 2070 MoE Bee Experimentator."""

from __future__ import annotations

from pathlib import Path
import torch

from hivemind.core.compression import (
    compress_tensor_dct_sparse_2bit,
    decompress_tensor_dct_sparse_2bit,
)
from hivemind.core.diloco_outer import (
    DiLoCoOuterOptimizer,
    filter_byzantine_deltas_multikrum,
)
from hivemind.core.exoplanet_astro_vmoe_poc import (
    EXOPLANET_ASTRO_EXPERT_BRANCHES,
    FrozenAstroVisionBackbone,
    UpcycledAstroSparseVisionMoE,
    asymmetric_focal_loss,
    build_astrophysics_visual_shard,
)
from hivemind.core.quantization import quantize_tensor_nf4_double_quant
from scripts.run_4h_rtx2070_moe_bee_experimentator import (
    run_4h_rtx2070_moe_experimentator,
)


def test_ip20_attacker_first_does_not_quarantine_honest_workers() -> None:
    """IP-20: An attacker arriving first with bogus keys/shapes must be quarantined while all 5 honest workers pass."""
    deltas: dict[str, dict[str, torch.Tensor]] = {
        "00_attacker_first": {"bogus_key": torch.ones(5, 5, dtype=torch.float32) * 42.0},
    }
    for i in range(1, 6):
        deltas[f"honest_{i}"] = {
            "lora_A": torch.ones(8, 16, dtype=torch.float32) * (0.2 + 0.001 * i),
            "lora_B": torch.ones(16, 8, dtype=torch.float32) * (0.1 + 0.001 * i),
        }

    report = filter_byzantine_deltas_multikrum(deltas, max_byzantine_f=1)
    assert "00_attacker_first" in report.quarantined_pre_krum
    assert len(report.accepted_workers) == 5
    for i in range(1, 6):
        assert f"honest_{i}" in report.accepted_workers

    # Also verify DiLoCoOuterOptimizer.step uses theta_prev schema
    theta_prev = {
        "lora_A": torch.zeros(8, 16, dtype=torch.float32),
        "lora_B": torch.zeros(16, 8, dtype=torch.float32),
    }
    opt = DiLoCoOuterOptimizer(max_byzantine_f=1)
    step_res = opt.step("test_branch", 1, theta_prev, deltas)
    assert "00_attacker_first" in step_res.filter_report.quarantined_pre_krum
    assert len(step_res.filter_report.accepted_workers) == 5


def test_ip34_multikrum_clamps_f_when_k_below_2f_plus_3() -> None:
    """IP-34: K=4 with max_byzantine_f=1 must clamp effective f to 0 and report krum_guarantee=False."""
    deltas_4 = {
        f"w_{i}": {"lora_A": torch.ones(4, 4, dtype=torch.float32) * (0.5 + 0.01 * i)}
        for i in range(4)
    }
    rep_4 = filter_byzantine_deltas_multikrum(deltas_4, max_byzantine_f=1)
    assert rep_4.max_byzantine_f == 0
    assert rep_4.krum_guarantee is False

    deltas_5 = {
        f"w_{i}": {"lora_A": torch.ones(4, 4, dtype=torch.float32) * (0.5 + 0.01 * i)}
        for i in range(5)
    }
    rep_5 = filter_byzantine_deltas_multikrum(deltas_5, max_byzantine_f=1)
    assert rep_5.max_byzantine_f == 1
    assert rep_5.krum_guarantee is True


def test_ip35_load_balance_loss_has_gradient_and_updates_router() -> None:
    """IP-35: out['load_balance_loss_weighted'].requires_grad is True and router gradients differ with alpha=0.01 vs 0."""
    backbone = FrozenAstroVisionBackbone(
        patch_dim=48,
        hidden_dim=64,
        intermediate_dim=128,
        num_classes=12,
        seed=123,
    )
    zero_deltas = {b_id: torch.zeros(64, 64).numpy() for b_id in EXOPLANET_ASTRO_EXPERT_BRANCHES}
    vmoe_with_aux = UpcycledAstroSparseVisionMoE(
        backbone=backbone, branch_deltas=zero_deltas, top_k=2, load_balance_alpha=0.01
    )
    vmoe_no_aux = UpcycledAstroSparseVisionMoE(
        backbone=backbone, branch_deltas=zero_deltas, top_k=2, load_balance_alpha=0.0
    )
    with torch.no_grad():
        vmoe_no_aux.router.weight.copy_(vmoe_with_aux.router.weight)

    x, y = build_astrophysics_visual_shard("expert_1_stellar_transits", num_samples=12, patch_dim=48, seed=77)
    out_with = vmoe_with_aux(x)
    assert isinstance(out_with["load_balance_loss_weighted"], torch.Tensor)
    assert out_with["load_balance_loss_weighted"].requires_grad is True

    (asymmetric_focal_loss(out_with["logits"], y) + out_with["load_balance_loss_weighted"]).backward()
    out_no = vmoe_no_aux(x)
    (asymmetric_focal_loss(out_no["logits"], y) + out_no["load_balance_loss_weighted"]).backward()

    assert vmoe_with_aux.router.weight.grad is not None
    assert vmoe_no_aux.router.weight.grad is not None
    assert not torch.allclose(vmoe_with_aux.router.weight.grad, vmoe_no_aux.router.weight.grad)


def test_ip37_nf4_does_not_mutate_input_and_2bit_centroids_are_symmetric() -> None:
    """IP-37: quantize_tensor_nf4_double_quant leaves input unchanged; 2-bit centroids are symmetric; index_width uses padded size."""
    w = torch.randn(64, 64, dtype=torch.float32)
    w_before = w.clone()
    _, _ = quantize_tensor_nf4_double_quant(w, block_size=64)
    assert torch.equal(w, w_before)

    # Check index_width when original_numel <= 65536 but padded_shape product > 65536
    # 255 * 257 = 65535 <= 65536, but padded to multiples of 8 is 256 * 264 = 67584 > 65536
    rect = torch.randn(255, 257, dtype=torch.float32)
    payload = compress_tensor_dct_sparse_2bit(rect, topk_ratio=0.05, chunk_size=8)
    assert payload.index_width == 4

    # Check symmetric 2-bit centroids
    gen = torch.Generator().manual_seed(99)
    half = torch.randn(2048, generator=gen, dtype=torch.float32)
    sym = torch.cat([half, -half], dim=0).reshape(64, 64)
    sym_payload = compress_tensor_dct_sparse_2bit(sym, topk_ratio=1.0, chunk_size=8)
    sym_recon = decompress_tensor_dct_sparse_2bit(sym_payload)
    pos = sym > 0.2
    neg = sym < -0.2
    pos_err = float(((sym[pos] - sym_recon[pos]).pow(2).sum() / sym[pos].pow(2).sum()).item())
    neg_err = float(((sym[neg] - sym_recon[neg]).pow(2).sum() / sym[neg].pow(2).sum()).item())
    assert abs(pos_err - neg_err) / max(pos_err, neg_err) < 0.15


def test_ip19_forged_receipt_with_poisoned_weights_and_zero_steps_is_rejected() -> None:
    """IP-19: A forged ZKOPoIV5ProofReceipt with poisoned weights and 0 training steps must return valid=False."""
    import hashlib
    import json

    from hivemind.swarm.zk_opoi_v5_circuit import (
        ZKOPoIV5FixedPointCircuit,
        ZKOPoIV5ProofReceipt,
        _derive_fiat_shamir_challenge,
    )

    circuit = ZKOPoIV5FixedPointCircuit(quantization_bits=8)
    shard_sha = hashlib.sha256(b"official-shard").hexdigest()
    theta0_q = circuit.quantize_to_fixed_point([0.1, -0.2, 0.3, -0.4])
    theta0_comm = hashlib.sha256(json.dumps(theta0_q).encode("utf-8")).hexdigest()
    poisoned_q = [999999, -999999, 999999, -999999]
    poisoned_comm = hashlib.sha256(json.dumps(poisoned_q).encode("utf-8")).hexdigest()
    forged_merkle = hashlib.sha256(b"forged-zero-step-witness").hexdigest()
    forged_challenge = _derive_fiat_shamir_challenge(
        f"attacker-node:{shard_sha}:{theta0_comm}:{poisoned_comm}:{forged_merkle}"
    )

    forged_receipt = ZKOPoIV5ProofReceipt(
        node_id="attacker-node",
        step_count_h=0,
        quantization_bits=8,
        assigned_shard_sha256=shard_sha,
        theta0_commitment=theta0_comm,
        theta_final_commitment=poisoned_comm,
        fiat_shamir_challenge=forged_challenge,
        theta0_poly_eval=0,
        grad_accum_poly_eval=0,
        theta_final_poly_eval=0,
        delta_l2_norm_fp=0,
        max_allowed_l2_norm_fp=circuit.max_delta_l2_norm_fp,
        spectral_bound_fp=0,
        max_allowed_spectral_fp=circuit.max_spectral_fp,
        witness_merkle_root=forged_merkle,
        proof_size_bytes=512,
    )

    verdict = circuit.verify_receipt_cpu_o1(
        forged_receipt,
        expected_shard_sha256=shard_sha,
        expected_theta0_commitment=theta0_comm,
    )
    assert verdict.valid is False


def test_run_4h_rtx2070_moe_experimentator_end_to_end(tmp_path: Path) -> None:
    """Verify the full 4-Stage RTX 2070 MoE Bee Experimentator produces a passing signed ledger."""
    out_file = tmp_path / "rtx2070_4h_moe_experimentator_ledger.json"
    rep = run_4h_rtx2070_moe_experimentator(steps_per_domain=10, output_path=out_file)
    assert out_file.exists()
    assert rep["overall_experiment_passed"] is True
    assert rep["stage1_dry_run"]["dry_run_passed"] is True
    assert rep["stage2_small_qlora_training"]["all_domains_ratchet_passed"] is True
    assert rep["stage3_moe_scale_and_inference"]["all_wave3_routes_matched"] is True
    assert len(rep["receipt_sha256"]) == 64
