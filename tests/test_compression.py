"""Unit tests for Top-k Sparsification & FP8 (E4M3) Pseudo-Gradient Compression."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from hivemind.core.compression import (
    compress_lora_delta,
    compress_tensor_dct_sparse_2bit,
    compress_tensor_topk_fp8,
    compute_pseudo_gradient_delta,
    decompress_lora_delta,
    decompress_tensor_topk_fp8,
)


def test_pseudo_gradient_computation() -> None:
    theta_init = {"lora_A": torch.ones((16, 64), dtype=torch.float32)}
    theta_final = {"lora_A": torch.full((16, 64), 0.75, dtype=torch.float32)}
    delta = compute_pseudo_gradient_delta(theta_init, theta_final)
    assert torch.allclose(delta["lora_A"], torch.full((16, 64), 0.25))


def test_topk_fp8_tensor_compression_and_reconstruction() -> None:
    torch.manual_seed(42)
    # Create a tensor with 20% dominant entries and 80% tiny noise
    base = torch.randn((64, 256), dtype=torch.float32) * 0.005
    top_indices = torch.randperm(base.numel())[: int(base.numel() * 0.20)]
    base.view(-1)[top_indices] = torch.randn(top_indices.numel()) * 2.5

    payload = compress_tensor_topk_fp8(base, topk_ratio=0.20)
    reconstructed = decompress_tensor_topk_fp8(payload)

    assert reconstructed.shape == base.shape
    cos_sim = F.cosine_similarity(base.reshape(1, -1), reconstructed.reshape(1, -1)).item()
    assert cos_sim > 0.99, f"Expected cosine similarity > 0.99, got {cos_sim}"


def test_lora_delta_package_wire_compression_ratio() -> None:
    torch.manual_seed(123)
    delta_dict = {
        "layers.0.self_attn.q_proj.lora_A.weight": torch.randn((64, 512), dtype=torch.float32),
        "layers.0.self_attn.q_proj.lora_B.weight": torch.randn((512, 64), dtype=torch.float32),
        "layers.0.mlp.gate_proj.lora_A.weight": torch.randn((64, 512), dtype=torch.float32),
        "layers.0.mlp.gate_proj.lora_B.weight": torch.randn((1024, 64), dtype=torch.float32),
    }
    pkg = compress_lora_delta(
        delta_dict=delta_dict,
        worker_id="worker-code-01",
        branch_name="Code",
        outer_step=1,
        inner_steps_completed=500,
        topk_ratio=0.20,
    )
    assert pkg.compression_ratio >= 3.0
    restored = decompress_lora_delta(pkg)
    assert set(restored.keys()) == set(delta_dict.keys())
    for k, orig in delta_dict.items():
        assert restored[k].shape == orig.shape


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("compress", [compress_tensor_topk_fp8, compress_tensor_dct_sparse_2bit])
def test_compressors_refuse_a_non_finite_delta_but_accept_a_finite_one(compress, bad) -> None:
    g = torch.Generator().manual_seed(5)
    ok = torch.randn(10, 10, generator=g)
    compress(ok)
    poisoned = ok.clone()
    poisoned[3, 4] = bad
    with pytest.raises(ValueError, match="NaN or inf"):
        compress(poisoned)
