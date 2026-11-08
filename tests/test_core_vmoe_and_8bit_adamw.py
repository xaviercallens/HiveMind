# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for Blockwise8BitAdamW in DiLoCo inner loops and real V-MoE CWT router dispatch."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from hivemind.core.diloco_inner import (
    Blockwise8BitAdamW,
    build_inner_optimizer,
    dequantize_tensor_blockwise_8bit,
    quantize_tensor_blockwise_8bit,
    replay_inner_segment,
    run_diloco_inner_loop,
)
from hivemind.core.exoplanet_astro_vmoe_poc import (
    EXOPLANET_ASTRO_EXPERT_BRANCHES,
    FrozenAstroVisionBackbone,
    UpcycledAstroSparseVisionMoE,
    generate_phase_folded_cwt_scalogram,
    project_cwt_scalogram_to_visual_patches,
    synthesize_exoplanet_transit_and_router_response,
)
from hivemind.core.quantization import compute_state_dict_sha256


class _TinyLoRACausalLM(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_dim: int = 16, rank: int = 4) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.embed.weight.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.randn(rank, hidden_dim) * 0.05)
        self.lora_B = nn.Parameter(torch.zeros(hidden_dim, rank))
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.head.weight.requires_grad_(False)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        h = self.embed(input_ids)
        delta = (h @ self.lora_A.t()) @ self.lora_B.t()
        logits = self.head(h + delta)
        loss = torch.tensor(0.0, device=input_ids.device)
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
        return type("Out", (), {"loss": loss, "logits": logits})()


def test_blockwise_8bit_adamw_quantizes_moments_and_reduces_bytes() -> None:
    """Verify Blockwise8BitAdamW quantizes exp_avg/exp_avg_sq into 8-bit blocks with ~3.76x savings."""
    torch.manual_seed(123)
    param = nn.Parameter(torch.randn(64, 64))
    opt = build_inner_optimizer([param], lr=1e-3, use_paged_8bit_adamw=True)
    assert isinstance(opt, Blockwise8BitAdamW)
    assert getattr(opt, "is_paged_8bit", False) is True

    opt_fp32 = build_inner_optimizer([param], lr=1e-3, use_paged_8bit_adamw=False)
    assert getattr(opt_fp32, "is_paged_8bit", True) is False

    for _ in range(3):
        opt.zero_grad()
        loss = (param * param).sum()
        loss.backward()
        opt.step()

    st = opt.state[param]
    assert "exp_avg_q8" in st and st["exp_avg_q8"].dtype == torch.int8
    assert "exp_avg_sq_q8" in st and st["exp_avg_sq_q8"].dtype == torch.uint8
    q_bytes = opt.quantized_state_bytes()
    fp32_bytes = opt.fp32_state_bytes()
    assert q_bytes > 0
    assert fp32_bytes / q_bytes > 3.5

    # Verify round-trip blockwise quantization error is small
    sample = torch.randn(100, dtype=torch.float32)
    q, absmax, sh = quantize_tensor_blockwise_8bit(sample, block_size=64, signed=True)
    recon = dequantize_tensor_blockwise_8bit(q, absmax, sh, signed=True)
    assert recon.shape == sample.shape
    assert float((sample - recon).abs().max().item()) < 0.03


def test_diloco_inner_loop_and_replay_with_paged_8bit_adamw() -> None:
    """Verify run_diloco_inner_loop with use_paged_8bit_adamw=True is deterministically replayable."""
    model = _TinyLoRACausalLM()
    tokens = torch.randint(0, 32, (8, 12), dtype=torch.long)

    res = run_diloco_inner_loop(
        model=model,
        token_sequences=tokens,
        worker_id="worker-8bit",
        branch_name="math",
        inner_steps_H=6,
        checkpoint_interval_s=3,
        lr=1e-3,
        rng_seed=99,
        use_paged_8bit_adamw=True,
    )
    assert res.use_paged_8bit_adamw is True
    assert 3 in res.micro_checkpoints

    replayed = replay_inner_segment(
        model=model,
        start_lora_state=res.micro_checkpoints[0].state_snapshot,
        token_sequences=tokens,
        start_step=0,
        end_step=3,
        lr=1e-3,
        rng_seed=99,
        use_paged_8bit_adamw=True,
    )
    assert compute_state_dict_sha256(replayed) == res.micro_checkpoints[3].lora_sha256


def test_exoplanet_simulator_routes_cwt_patches_through_vmoe() -> None:
    """Verify synthesize_exoplanet_transit_and_router_response routes CWT patches through V-MoE."""
    flux, scalogram = generate_phase_folded_cwt_scalogram(
        transit_depth_ppm=2100.0,
        instrumental_noise_ppm=100.0,
        seed=42,
    )
    patches = project_cwt_scalogram_to_visual_patches(
        scalogram=scalogram,
        flux_ppm=flux,
        snr_single_transit=21.0,
    )
    assert tuple(patches.shape) == (1, 4, 48)

    backbone = FrozenAstroVisionBackbone(patch_dim=48, hidden_dim=64, intermediate_dim=128, seed=101)
    deltas = {b_id: np.zeros((64, 64), dtype=np.float32) for b_id in EXOPLANET_ASTRO_EXPERT_BRANCHES}
    vmoe = UpcycledAstroSparseVisionMoE(backbone=backbone, branch_deltas=deltas)

    sim_a = synthesize_exoplanet_transit_and_router_response(
        1.0, 0.2, instrumental_noise_ppm=100.0, seed=42, vmoe=vmoe
    )
    assert sim_a["visual_patch_shape"] == [1, 4, 48]
    assert len(sim_a["neural_router_logits"]) == 4
    assert sim_a["top1_expert"] == "expert_1_stellar_transits"
    assert sim_a["top1_probability"] > 0.85

    # Perturbing router weights must change neural_router_logits
    with torch.no_grad():
        vmoe.router.bias.add_(torch.tensor([0.5, -0.2, 0.1, -0.4]))
    sim_b = synthesize_exoplanet_transit_and_router_response(
        1.0, 0.2, instrumental_noise_ppm=100.0, seed=42, vmoe=vmoe
    )
    assert sim_a["neural_router_logits"] != sim_b["neural_router_logits"]
