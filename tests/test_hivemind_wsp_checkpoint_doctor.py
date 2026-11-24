# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-09: offline safetensors MoE surgery doctor and ANSE CLI."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:
    ZeroStubAudit = None

from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import CrowdV1RuntimeConfig, CrowdV1WithEngram
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    extract_moe_trainable_buckets,
    load_moe_trainable_buckets,
    upcycle_crowd_v1_to_sparse_moe,
)
from hivemind.swarm.layer_localized_attribution import compute_expert_weight_norms

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCTOR_PATH = REPO_ROOT / "scripts" / "hivemind_wsp_checkpoint_doctor.py"
TEST_FILE_PATH = Path(__file__).resolve()
EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."


def _load_module_from_path(name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


doctor = _load_module_from_path("hivemind_wsp_checkpoint_doctor", DOCTOR_PATH)


def _make_synthetic_moe_buckets(
    *,
    num_layers: int = 4,
    num_experts: int = 8,
    rank: int = 4,
    hidden_dim: int = 16,
    inter_dim: int = 24,
    num_router_fragments: int = 2,
    dtype: torch.dtype = torch.float32,
    seed: int = 20261124,
    b_scale: float = 0.02,
) -> dict[str, torch.Tensor]:
    """Build deterministic CPU CrowdLoRA bucket tensors matching ``extract_moe_trainable_buckets``."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    frag_span = max(1, math.ceil(num_layers / num_router_fragments))
    router_parts: dict[str, list[torch.Tensor]] = {}
    expert_parts: dict[str, list[torch.Tensor]] = {f"expert_{e}": [] for e in range(num_experts)}

    for l_idx in range(num_layers):
        f_idx = min(num_router_fragments - 1, l_idx // frag_span)
        r_key = f"router_frag_{f_idx}"
        r_w = (0.15 * torch.randn((num_experts, hidden_dim), generator=gen, dtype=torch.float32)).to(dtype)
        router_parts.setdefault(r_key, []).append(r_w.reshape(-1))

        for e_idx in range(num_experts):
            w1_a = (torch.randn((rank, hidden_dim), generator=gen, dtype=torch.float32) / math.sqrt(hidden_dim)).to(dtype)
            w1_b = (b_scale * torch.randn((inter_dim, rank), generator=gen, dtype=torch.float32)).to(dtype)
            w2_a = (torch.randn((rank, hidden_dim), generator=gen, dtype=torch.float32) / math.sqrt(hidden_dim)).to(dtype)
            w2_b = (b_scale * torch.randn((inter_dim, rank), generator=gen, dtype=torch.float32)).to(dtype)
            w3_a = (torch.randn((rank, inter_dim), generator=gen, dtype=torch.float32) / math.sqrt(inter_dim)).to(dtype)
            w3_b = (b_scale * torch.randn((hidden_dim, rank), generator=gen, dtype=torch.float32)).to(dtype)
            for factor in (w1_a, w1_b, w2_a, w2_b, w3_a, w3_b):
                expert_parts[f"expert_{e_idx}"].append(factor.reshape(-1))

    last_r_key = f"router_frag_{num_router_fragments - 1}"
    router_parts.setdefault(last_r_key, []).append(torch.ones(hidden_dim, dtype=dtype))
    router_parts[last_r_key].append(torch.zeros(hidden_dim, dtype=dtype))

    out: dict[str, torch.Tensor] = {}
    for k, parts in sorted(router_parts.items()):
        out[k] = torch.cat(parts, dim=0).contiguous()
    for e_idx in range(num_experts):
        k = f"expert_{e_idx}"
        out[k] = torch.cat(expert_parts[k], dim=0).contiguous()
    return out


def _slice_expert_factor(
    buckets: dict[str, torch.Tensor],
    *,
    expert_idx: int,
    layer_idx: int,
    factor_name: str,
    rank: int,
    hidden_dim: int,
    inter_dim: int,
) -> torch.Tensor:
    """Return a mutable slice view of one expert layer factor inside ``buckets``."""
    specs = (
        ("w1_A", rank * hidden_dim),
        ("w1_B", inter_dim * rank),
        ("w2_A", rank * hidden_dim),
        ("w2_B", inter_dim * rank),
        ("w3_A", rank * inter_dim),
        ("w3_B", hidden_dim * rank),
    )
    per_layer = sum(n for _, n in specs)
    base_off = layer_idx * per_layer
    off = 0
    for name, numel in specs:
        if name == factor_name:
            return buckets[f"expert_{expert_idx}"][base_off + off : base_off + off + numel]
        off += numel
    raise KeyError(factor_name)


# ---------------------------------------------------------------------------
# S1: Rogue High-Norm Expert 3 + LayerNorm Tail Drift + NaN & Spectral Spikes
# ---------------------------------------------------------------------------


def test_s1_detects_rogue_expert3_ln_drift_nan_and_spectral_spikes(tmp_path: Path) -> None:
    """Detect rogue expert_3 (||w1_B||=42.5), LN tail drift vs 1.0, NaNs, and spectral spikes."""
    num_layers, num_experts, rank, hidden_dim, inter_dim = 4, 8, 4, 16, 24
    buckets = _make_synthetic_moe_buckets(
        num_layers=num_layers,
        num_experts=num_experts,
        rank=rank,
        hidden_dim=hidden_dim,
        inter_dim=inter_dim,
        seed=101,
        b_scale=0.02,
    )

    # Healthy baseline check first
    healthy_rep = doctor.inspect_moe_bucket_safetensors(
        buckets,
        num_layers=num_layers,
        num_experts=num_experts,
        rank=rank,
        hidden_dim=hidden_dim,
        inter_dim=inter_dim,
    )
    assert healthy_rep["healthy"] is True
    assert healthy_rep["status"] == "HEALTHY"
    assert healthy_rep["health_failures"] == []
    assert healthy_rep["metrics"]["anse_energy"] < 1.0
    assert doctor.verify_report_provenance(healthy_rep) is True

    # Plant (1) rogue unscored expert_3 with exact global ||w1_B|| = 42.5 across 4 layers
    per_layer_norm = 42.5 / math.sqrt(float(num_layers))
    w1_b_numel = inter_dim * rank
    fill_val = per_layer_norm / math.sqrt(float(w1_b_numel))
    for l_idx in range(num_layers):
        sl_w1_b = _slice_expert_factor(
            buckets,
            expert_idx=3,
            layer_idx=l_idx,
            factor_name="w1_B",
            rank=rank,
            hidden_dim=hidden_dim,
            inter_dim=inter_dim,
        )
        sl_w1_b.fill_(fill_val)

    # Plant (2) LayerNorm tail drift vs 1.0 (weight shifted to 0.35, bias shifted by -0.60)
    last_r_key = "router_frag_1"
    buckets[last_r_key][-2 * hidden_dim : -hidden_dim].fill_(0.35)
    buckets[last_r_key][-hidden_dim:].fill_(-0.60)

    # Plant (3) NaN + Inf corruption in expert_5 layer 1 w2_A and router_frag_0
    sl_e5_w2a = _slice_expert_factor(
        buckets,
        expert_idx=5,
        layer_idx=1,
        factor_name="w2_A",
        rank=rank,
        hidden_dim=hidden_dim,
        inter_dim=inter_dim,
    )
    sl_e5_w2a[0] = float("nan")
    sl_e5_w2a[1] = float("inf")
    buckets["router_frag_0"][0] = float("nan")

    # Plant (4) Singular-value spectral spike in expert_2 layer 0 w3 (rank-1 spike > 10x median)
    sl_e2_w3b = _slice_expert_factor(
        buckets,
        expert_idx=2,
        layer_idx=0,
        factor_name="w3_B",
        rank=rank,
        hidden_dim=hidden_dim,
        inter_dim=inter_dim,
    ).reshape(hidden_dim, rank)
    sl_e2_w3a = _slice_expert_factor(
        buckets,
        expert_idx=2,
        layer_idx=0,
        factor_name="w3_A",
        rank=rank,
        hidden_dim=hidden_dim,
        inter_dim=inter_dim,
    ).reshape(rank, inter_dim)
    sl_e2_w3a.zero_()
    sl_e2_w3b.zero_()
    for m_idx in range(rank):
        sl_e2_w3a[m_idx, m_idx] = 1.0
        sl_e2_w3b[m_idx, m_idx] = 9.0 if m_idx == 0 else 0.1  # sigma_1 / median = 90.0 > 10.0

    # Plant (5) Router row norm outlier in layer 2 expert 6 (> 50.0)
    router_l2_start = 0
    buckets["router_frag_1"][router_l2_start + 6 * hidden_dim] = 75.0

    ckpt_path = tmp_path / "corrupt_moe_buckets.safetensors"
    save_file(buckets, str(ckpt_path))

    report = doctor.inspect_moe_bucket_safetensors(
        ckpt_path,
        num_layers=num_layers,
        num_experts=num_experts,
        rank=rank,
        hidden_dim=hidden_dim,
        inter_dim=inter_dim,
        scored_experts=(0, 1, 2, 5, 6, 7),
    )

    assert report["healthy"] is False
    assert report["status"] == "UNHEALTHY"
    assert report["source_type"] == "safetensors_file"
    assert doctor.verify_report_provenance(report) is True
    assert report["metrics"]["anse_energy"] >= doctor.ANSE_MAXIMUM_PAIN

    # Rogue expert_3 verification
    e3_info = report["experts"]["expert_3"]
    assert math.isclose(e3_info["w1_b_norm"], 42.5, rel_tol=1e-5)
    assert e3_info["is_scored_domain"] is False
    assert e3_info["is_high_norm_outlier"] is True
    assert e3_info["is_rogue_high_norm"] is True
    assert report["rogue_audit"]["rogue_high_norm_experts"] == [3]
    assert report["rogue_audit"]["recommended_sink_experts"] == [3]

    # Scored expert_2 with spectral spike is NOT misclassified as rogue_unscored
    assert report["experts"]["expert_2"]["is_scored_domain"] is True
    assert report["experts"]["expert_2"]["is_rogue_high_norm"] is False
    assert report["spectral_audit"]["spectral_spike_count"] >= 1
    assert any(sp["expert"] == 2 and sp["projection"] == "w3" for sp in report["spectral_audit"]["spectral_spikes"])

    # LayerNorm tail drift vs 1.0 verification
    ln_rep = report["layernorm_tail"]
    assert ln_rep["drift_failed"] is True
    assert math.isclose(ln_rep["ln_weight_mean"], 0.35, rel_tol=1e-5)
    assert math.isclose(ln_rep["ln_weight_drift_rms"], 0.65, rel_tol=1e-5)
    assert math.isclose(ln_rep["ln_bias_drift_rms"], 0.60, rel_tol=1e-5)

    # Non-finite summary & router norm outlier verification
    nf_sum = report["non_finite_summary"]
    assert nf_sum["has_non_finite"] is True
    assert nf_sum["total_non_finite_elements"] == 3
    assert nf_sum["nan_elements"] == 2
    assert nf_sum["inf_elements"] == 1
    assert report["router_audit"]["router_norm_outliers"]
    assert any(f.startswith("non_finite_corruption") for f in report["health_failures"])
    assert any(f.startswith("rogue_high_norm_experts") for f in report["health_failures"])
    assert any(f.startswith("layernorm_tail_drift") for f in report["health_failures"])
    assert any(f.startswith("spectral_spikes") for f in report["health_failures"])
    assert any(f.startswith("router_norm_outliers") for f in report["health_failures"])

    # JSON strict serialization safety (never emits NaN/Inf)
    json.dumps(report, allow_nan=False)

    # Cross-check exact norm parity against live CrowdV1WithEngram + CrowdLoRAMoEMLP
    cfg = CrowdV1RuntimeConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=24,
        max_position_embeddings=16,
        engram_buckets=0,
        tie_word_embeddings=False,
    )
    model = CrowdV1WithEngram(cfg)
    upcycle_crowd_v1_to_sparse_moe(model, num_experts=4, top_k=2, lora_rank=4, seed=77)
    with torch.no_grad():
        for blk in model.blocks:
            blk.mlp.experts["3"].w1_B.fill_(21.25 / math.sqrt(24.0 * 4.0))
        model.ln.weight.fill_(1.0)
        model.ln.bias.zero_()
    live_buckets = extract_moe_trainable_buckets(model, num_router_fragments=4)
    live_norms = compute_expert_weight_norms(model)
    live_rep = doctor.inspect_moe_bucket_safetensors(
        live_buckets,
        num_layers=2,
        num_experts=4,
        rank=4,
        hidden_dim=16,
        inter_dim=24,
    )
    assert math.isclose(live_rep["experts"]["expert_3"]["w1_b_norm"], live_norms[3]["w1_b_norm"], rel_tol=1e-5)
    assert live_rep["fomoe_zero_sink_summary"]["full_zero_sink_experts"] == [0, 1, 2]
    assert live_rep["fomoe_zero_sink_summary"]["active_experts"] == [3]


# ---------------------------------------------------------------------------
# S2: Layer-Localized Ghost-Sink + lambda_ln=0.65 Bitwise Preservation Surgery
# ---------------------------------------------------------------------------


def test_s2_layer_localized_ghost_sink_and_lambda_ln_surgery_bitwise_preservation() -> None:
    """Zero B-matrices on active layers [2, 3] + lambda_ln=0.65 while keeping layers [0, 1] bitwise identical."""
    num_layers, num_experts, rank, hidden_dim, inter_dim = 4, 4, 4, 16, 24
    buckets = _make_synthetic_moe_buckets(
        num_layers=num_layers,
        num_experts=num_experts,
        rank=rank,
        hidden_dim=hidden_dim,
        inter_dim=inter_dim,
        num_router_fragments=2,
        dtype=torch.bfloat16,
        seed=202,
        b_scale=0.35,
    )
    base_ln_w = torch.linspace(0.90, 1.10, hidden_dim, dtype=torch.float32)
    base_ln_b = torch.linspace(-0.05, 0.05, hidden_dim, dtype=torch.float32)

    # Plant drifted LN tail on buckets
    buckets["router_frag_1"][-2 * hidden_dim : -hidden_dim].copy_(torch.full((hidden_dim,), 0.40, dtype=torch.bfloat16))
    buckets["router_frag_1"][-hidden_dim:].copy_(torch.full((hidden_dim,), -0.80, dtype=torch.bfloat16))

    # Plant NaN in LN tail and active layer to verify clean repair without affecting preserved layers
    buckets["router_frag_1"][-1] = float("nan")
    orig_snapshot = {k: v.clone() for k, v in buckets.items()}

    sanitized, receipt = doctor.apply_offline_ghost_sink_and_ln_surgery(
        buckets,
        sink_experts=[3],
        active_layers=[2, 3],
        lambda_ln=0.65,
        base_ln_tail=(base_ln_w, base_ln_b),
        num_layers=num_layers,
        num_experts=num_experts,
        rank=rank,
        hidden_dim=hidden_dim,
        inter_dim=inter_dim,
    )

    # Input buckets must never be mutated in-place
    assert torch.equal(
        doctor._raw_bit_view(buckets["expert_3"]),
        doctor._raw_bit_view(orig_snapshot["expert_3"]),
    )

    # Non-sink experts (0, 1, 2) and non-tail router_frag_0 must be 100% bitwise identical
    for untouched_key in ("expert_0", "expert_1", "expert_2", "router_frag_0"):
        assert torch.equal(
            doctor._raw_bit_view(sanitized[untouched_key]),
            doctor._raw_bit_view(orig_snapshot[untouched_key]),
        )
        assert sanitized[untouched_key].dtype == torch.bfloat16

    # Router weight slice inside router_frag_1 must be 100% bitwise identical
    tail_off = sanitized["router_frag_1"].numel() - 2 * hidden_dim
    assert torch.equal(
        doctor._raw_bit_view(sanitized["router_frag_1"][:tail_off]),
        doctor._raw_bit_view(orig_snapshot["router_frag_1"][:tail_off]),
    )

    # For expert_3: layers 0 and 1 must be 100% bitwise identical; layers 2 and 3 have exact zero B and bitwise A
    for l_idx in (0, 1):
        for f_name in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B"):
            orig_sl = _slice_expert_factor(
                orig_snapshot,
                expert_idx=3,
                layer_idx=l_idx,
                factor_name=f_name,
                rank=rank,
                hidden_dim=hidden_dim,
                inter_dim=inter_dim,
            )
            new_sl = _slice_expert_factor(
                sanitized,
                expert_idx=3,
                layer_idx=l_idx,
                factor_name=f_name,
                rank=rank,
                hidden_dim=hidden_dim,
                inter_dim=inter_dim,
            )
            assert torch.equal(doctor._raw_bit_view(orig_sl), doctor._raw_bit_view(new_sl))

    for l_idx in (2, 3):
        for a_name in ("w1_A", "w2_A", "w3_A"):
            orig_sl = _slice_expert_factor(
                orig_snapshot,
                expert_idx=3,
                layer_idx=l_idx,
                factor_name=a_name,
                rank=rank,
                hidden_dim=hidden_dim,
                inter_dim=inter_dim,
            )
            new_sl = _slice_expert_factor(
                sanitized,
                expert_idx=3,
                layer_idx=l_idx,
                factor_name=a_name,
                rank=rank,
                hidden_dim=hidden_dim,
                inter_dim=inter_dim,
            )
            assert torch.equal(doctor._raw_bit_view(orig_sl), doctor._raw_bit_view(new_sl))
        for b_name in ("w1_B", "w2_B", "w3_B"):
            new_sl = _slice_expert_factor(
                sanitized,
                expert_idx=3,
                layer_idx=l_idx,
                factor_name=b_name,
                rank=rank,
                hidden_dim=hidden_dim,
                inter_dim=inter_dim,
            )
            assert torch.all(new_sl == 0)

    # Verify exact lambda_ln = 0.65 interpolation on LN tail (with NaN slot repaired to base_ln_b[-1])
    new_tail = sanitized["router_frag_1"][tail_off:].to(torch.float32)
    expected_w = (base_ln_w.to(torch.float64) + 0.65 * (0.40 - base_ln_w.to(torch.float64))).to(torch.bfloat16).to(torch.float32)
    curr_b_repaired = torch.full((hidden_dim,), -0.80, dtype=torch.bfloat16).to(torch.float64)
    curr_b_repaired[-1] = base_ln_b[-1].to(torch.float64)
    expected_b = (base_ln_b.to(torch.float64) + 0.65 * (curr_b_repaired - base_ln_b.to(torch.float64))).to(torch.bfloat16).to(torch.float32)
    assert torch.allclose(new_tail[:hidden_dim], expected_w, atol=1e-6, rtol=0.0)
    assert torch.allclose(new_tail[hidden_dim:], expected_b, atol=1e-6, rtol=0.0)

    # Verify receipt accounting & FoMoE partial zero-sink inspection
    assert receipt["non_active_layers_bitwise_preserved"] is True
    assert receipt["surgery_config"]["active_layers"] == [2, 3]
    assert receipt["surgery_config"]["preserved_layers"] == [0, 1]
    assert receipt["non_finite_repaired_elements"] == 1
    assert 0.0 < receipt["parameter_edit_ratio"] < 1.0
    assert doctor.verify_report_provenance(receipt) is True

    post_rep = doctor.inspect_moe_bucket_safetensors(
        sanitized,
        num_layers=num_layers,
        num_experts=num_experts,
        rank=rank,
        hidden_dim=hidden_dim,
        inter_dim=inter_dim,
        base_ln_tail=(base_ln_w, base_ln_b),
    )
    assert post_rep["experts"]["expert_3"]["fomoe_zero_sink_status"] == "partial_zero_sink"
    assert post_rep["experts"]["expert_3"]["zero_sink_layers"] == [2, 3]
    assert post_rep["experts"]["expert_3"]["nonzero_layers"] == [0, 1]

    # Verify geometry auto-inference & active_layers=None full zero-sink on CrowdV1WithEngram round-trip
    cfg = CrowdV1RuntimeConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=24,
        max_position_embeddings=16,
        engram_buckets=0,
        tie_word_embeddings=False,
    )
    model = CrowdV1WithEngram(cfg)
    upcycle_crowd_v1_to_sparse_moe(model, num_experts=4, top_k=2, lora_rank=4, seed=203)
    with torch.no_grad():
        for blk in model.blocks:
            blk.mlp.experts["3"].w1_B.fill_(1.25)
            blk.mlp.experts["3"].w2_B.fill_(0.50)
            blk.mlp.experts["3"].w3_B.fill_(-0.75)
        model.ln.weight.fill_(0.60)
        model.ln.bias.fill_(0.20)
    live_buckets = extract_moe_trainable_buckets(model, num_router_fragments=2)
    inferred_tail = torch.cat([torch.ones(16), torch.zeros(16)], dim=0)
    full_sunk, full_rec = doctor.apply_offline_ghost_sink_and_ln_surgery(
        live_buckets,
        sink_experts=[3],
        active_layers=None,
        lambda_ln=0.0,
        base_ln_tail=inferred_tail,
    )
    assert full_rec["surgery_config"]["active_layers"] == [0, 1]
    load_moe_trainable_buckets(model, full_sunk, num_router_fragments=2)
    x_in = torch.randn((4, 16), generator=torch.Generator("cpu").manual_seed(7))
    h1 = model.blocks[0].mlp.base_mlp.w1(x_in)
    h2 = model.blocks[0].mlp.base_mlp.w2(x_in)
    dy = model.blocks[0].mlp.experts["3"].forward_delta(x_in, h1, h2, model.blocks[0].mlp.base_mlp.w3.weight)
    assert torch.equal(dy, torch.zeros_like(dy))
    assert torch.allclose(model.ln.weight, torch.ones(16))
    assert torch.allclose(model.ln.bias, torch.zeros(16))


# ---------------------------------------------------------------------------
# S3: Safetensors Disk Round-Trip & SHA-256 Determinism
# ---------------------------------------------------------------------------


def test_s3_safetensors_disk_roundtrip_and_sha256_determinism(tmp_path: Path) -> None:
    """Verify deterministic safetensors serialization, reload parity, and SHA-256 provenance."""
    geom = {"num_layers": 2, "num_experts": 4, "rank": 4, "hidden_dim": 16, "inter_dim": 24}
    buckets = _make_synthetic_moe_buckets(**geom, num_router_fragments=2, dtype=torch.bfloat16, seed=303)
    sanitized, rec_mem = doctor.apply_offline_ghost_sink_and_ln_surgery(
        buckets,
        sink_experts=[2],
        active_layers=[1],
        lambda_ln=0.65,
        **geom,
    )

    path_a = tmp_path / "nested" / "ckpt_a.safetensors"
    path_b = tmp_path / "ckpt_b.safetensors"
    meta = {
        "hivemind_doctor_meta": json.dumps(
            {
                "num_layers": 2,
                "num_experts": 4,
                "rank": 4,
                "hidden_dim": 16,
                "inter_dim": 24,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    }
    sha_a = doctor.save_checkpoint_safetensors(sanitized, path_a, metadata=meta)
    sha_b = doctor.save_checkpoint_safetensors(sanitized, path_b, metadata=meta)
    assert sha_a == sha_b

    loaded, loaded_meta, loaded_sha = doctor.load_checkpoint_buckets(path_a)
    assert loaded_sha == sha_a
    assert loaded_meta == meta
    assert doctor.compute_buckets_sha256(loaded) == rec_mem["output_buckets_sha256"]
    for key in sanitized:
        assert torch.equal(doctor._raw_bit_view(sanitized[key]), doctor._raw_bit_view(loaded[key]))

    # Two independent inspections on same disk file must emit identical receipt_sha256
    rep1 = doctor.inspect_moe_bucket_safetensors(path_a, **geom)
    rep2 = doctor.inspect_moe_bucket_safetensors(path_a, **geom)
    assert rep1["buckets_sha256"] == rep2["buckets_sha256"] == rec_mem["output_buckets_sha256"]
    assert rep1["source_file_sha256"] == sha_a
    assert rep1["receipt_sha256"] == rep2["receipt_sha256"]
    assert doctor.verify_report_provenance(rep1) is True

    # Provenance verification rejects tampered or non-finite reports
    tampered = dict(rep1)
    tampered["healthy"] = False
    assert doctor.verify_report_provenance(tampered) is False
    bad_hex = dict(rep1)
    bad_hex["receipt_sha256"] = "short"
    assert doctor.verify_report_provenance(bad_hex) is False
    nan_rep = dict(rep1)
    nan_rep["metrics"] = {"anse_energy": float("nan")}
    assert doctor.verify_report_provenance(nan_rep) is False


# ---------------------------------------------------------------------------
# S4: CLI --inspect / --apply-surgery / --json / Markdown & Exit Codes
# ---------------------------------------------------------------------------


def test_s4_cli_inspect_apply_surgery_json_markdown_and_exit_codes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify CLI exit codes (0 healthy, 2 unhealthy), JSON/Markdown outputs, and error handling."""
    geom = {"num_layers": 4, "num_experts": 4, "rank": 4, "hidden_dim": 16, "inter_dim": 24}
    healthy_buckets = _make_synthetic_moe_buckets(**geom, num_router_fragments=2, seed=401, b_scale=0.02)
    healthy_path = tmp_path / "healthy.safetensors"
    doctor.save_checkpoint_safetensors(
        healthy_buckets,
        healthy_path,
        metadata={
            "hivemind_num_layers": "4",
            "hivemind_num_experts": "4",
            "hivemind_rank": "4",
            "hivemind_hidden_dim": "16",
            "hivemind_inter_dim": "24",
        },
    )

    # 1. Healthy checkpoint --inspect -> exit 0 (subprocess & in-process)
    sub_ok = subprocess.run(
        [sys.executable, str(DOCTOR_PATH), "--inspect", str(healthy_path), "--json"],
        cwd=str(REPO_ROOT),
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        capture_output=True,
        text=True,
        check=False,
    )
    assert sub_ok.returncode == 0
    sub_json = json.loads(sub_ok.stdout)
    assert sub_json["healthy"] is True
    assert doctor.verify_report_provenance(sub_json) is True

    # Markdown default output
    assert doctor.main(["--inspect", str(healthy_path), "--markdown"]) == 0
    md_out = capsys.readouterr().out
    assert "HiveMind WSP Checkpoint Doctor Inspection Report" in md_out
    assert "`HEALTHY`" in md_out

    # 2. Unhealthy checkpoint (rogue expert_3 + LN drift + NaN) --inspect -> exit 2
    unhealthy_buckets = _make_synthetic_moe_buckets(**geom, num_router_fragments=2, seed=402, b_scale=0.02)
    for l_idx in range(geom["num_layers"]):
        sl = _slice_expert_factor(
            unhealthy_buckets,
            expert_idx=3,
            layer_idx=l_idx,
            factor_name="w1_B",
            rank=4,
            hidden_dim=16,
            inter_dim=24,
        )
        sl.fill_((42.5 / math.sqrt(4.0)) / math.sqrt(float(sl.numel())))
    unhealthy_buckets["router_frag_1"][-32:-16].fill_(0.50)
    unhealthy_buckets["router_frag_1"][-16:].fill_(-0.40)
    unhealthy_buckets["expert_2"][0] = float("nan")

    base_path = tmp_path / "base.safetensors"
    save_file(healthy_buckets, str(base_path))
    unhealthy_path = tmp_path / "unhealthy.safetensors"
    save_file(unhealthy_buckets, str(unhealthy_path))

    out_inspect_json = tmp_path / "reports" / "inspect.json"
    out_inspect_md = tmp_path / "reports" / "inspect.md"
    rc_bad = doctor.main(
        [
            "--inspect",
            str(unhealthy_path),
            "--num-layers",
            "4",
            "--num-experts",
            "4",
            "--rank",
            "4",
            "--hidden-dim",
            "16",
            "--inter-dim",
            "24",
            "--scored-experts",
            "0,1,2",
            "--base-checkpoint",
            str(base_path),
            "--out-json",
            str(out_inspect_json),
            "--out-markdown",
            str(out_inspect_md),
            "--json",
        ]
    )
    assert rc_bad == 2
    bad_doc = json.loads(capsys.readouterr().out)
    assert bad_doc["healthy"] is False
    assert bad_doc["rogue_audit"]["rogue_high_norm_experts"] == [3]
    assert out_inspect_json.is_file() and out_inspect_md.is_file()

    # 3. --apply-surgery repairs rogue expert_3, NaN, and LN drift -> exit 0
    repaired_path = tmp_path / "repaired.safetensors"
    out_surg_json = tmp_path / "reports" / "surgery.json"
    out_surg_md = tmp_path / "reports" / "surgery.md"
    rc_fix = doctor.main(
        [
            "--apply-surgery",
            str(unhealthy_path),
            "--base-checkpoint",
            str(base_path),
            "--out-checkpoint",
            str(repaired_path),
            "--num-layers",
            "4",
            "--num-experts",
            "4",
            "--rank",
            "4",
            "--hidden-dim",
            "16",
            "--inter-dim",
            "24",
            "--lambda-ln",
            "0.05",
            "--out-json",
            str(out_surg_json),
            "--out-markdown",
            str(out_surg_md),
            "--json",
        ]
    )
    assert rc_fix == 0
    surg_doc = json.loads(capsys.readouterr().out)
    assert surg_doc["post_surgery_inspection"]["healthy"] is True
    assert surg_doc["non_finite_repaired_elements"] == 1
    assert surg_doc["surgery_config"]["applied_sink_experts"] == [3]
    assert doctor.verify_report_provenance(surg_doc) is True
    assert "HiveMind WSP Offline Ghost-Sink & LayerNorm Surgery Report" in out_surg_md.read_text(encoding="utf-8")

    # Saved repaired checkpoint now passes --inspect using embedded metadata alone
    assert doctor.main(["--inspect", str(repaired_path)]) == 0
    _ = capsys.readouterr()

    # 4. --apply-surgery with insufficient partial layers (rogue norm still >= 10) -> exit 2
    rc_still_bad = doctor.main(
        [
            "--apply-surgery",
            str(unhealthy_path),
            "--sink-experts",
            "3",
            "--active-layers",
            "3",
            "--lambda-ln",
            "0.65",
            "--num-layers",
            "4",
            "--num-experts",
            "4",
            "--rank",
            "4",
            "--hidden-dim",
            "16",
            "--inter-dim",
            "24",
        ]
    )
    assert rc_still_bad == 2
    assert "UNHEALTHY" in capsys.readouterr().out

    # 5. Error paths return exit code 2 cleanly
    assert doctor.main(["--inspect"]) == 2
    assert doctor.main([str(healthy_path)]) == 2
    assert doctor.main(["--inspect", str(tmp_path / "does_not_exist.safetensors")]) == 2
    corrupt_hdr = tmp_path / "not_safetensors.safetensors"
    corrupt_hdr.write_bytes(b"corrupt")
    assert doctor.main(["--inspect", str(corrupt_hdr), "--num-layers", "2", "--num-experts", "2", "--rank", "2", "--hidden-dim", "4", "--inter-dim", "8"]) == 2
    assert doctor.main(["--inspect", str(healthy_path), "--scored-experts", "bad_int"]) == 2
    assert doctor.main(["--apply-surgery", str(unhealthy_path), "--sink-experts", "none", "--no-sanitize-non-finite", "--num-layers", "4", "--num-experts", "4", "--rank", "4", "--hidden-dim", "16", "--inter-dim", "24"]) == 2


def test_validation_and_edge_cases(tmp_path: Path) -> None:
    """Cover geometry inference, single-mode spikes, error guards, and save/load failure paths."""
    geom = {"num_layers": 2, "num_experts": 4, "rank": 4, "hidden_dim": 16, "inter_dim": 24}
    buckets = _make_synthetic_moe_buckets(**geom, num_router_fragments=1, seed=505)

    # Geometry inference branches: num_layers given OR hidden_dim+inter_dim given
    assert doctor._infer_geometry(
        buckets,
        metadata={},
        base_ln_tail=None,
        num_layers=2,
        num_experts=None,
        rank=4,
        hidden_dim=None,
        inter_dim=None,
    ) == (2, 4, 4, 16, 24)
    assert doctor._infer_geometry(
        buckets,
        metadata={},
        base_ln_tail=None,
        num_layers=2,
        num_experts=4,
        rank=None,
        hidden_dim=16,
        inter_dim=24,
    ) == (2, 4, 4, 16, 24)

    # Single-mode rank-1 spectral spike & base_ln_tail 2D tensor form
    single_mode_b = {k: v.clone() for k, v in buckets.items()}
    sl_a = _slice_expert_factor(single_mode_b, expert_idx=1, layer_idx=0, factor_name="w1_A", rank=4, hidden_dim=16, inter_dim=24).reshape(4, 16)
    sl_b = _slice_expert_factor(single_mode_b, expert_idx=1, layer_idx=0, factor_name="w1_B", rank=4, hidden_dim=16, inter_dim=24).reshape(24, 4)
    sl_a.zero_()
    sl_b.zero_()
    sl_a[0, 0] = 1.0
    sl_b[0, 0] = 60.0  # s_pos has 1 mode with sigma_1 = 60.0 > single_mode_spike_norm=50.0
    rep_sm = doctor.inspect_moe_bucket_safetensors(
        single_mode_b,
        **geom,
        base_ln_tail=torch.stack([torch.ones(16), torch.zeros(16)], dim=0),
    )
    assert rep_sm["healthy"] is False
    assert rep_sm["spectral_audit"]["spectral_spike_count"] == 1

    # Missing sink expert inside sparse checkpoint is recorded gracefully
    sparse_b = {k: v.clone() for k, v in buckets.items() if k != "expert_2"}
    _, sparse_rec = doctor.apply_offline_ghost_sink_and_ln_surgery(
        sparse_b, sink_experts=[2], lambda_ln=1.0, **geom
    )
    assert sparse_rec["surgery_config"]["missing_sink_experts"] == [2]

    # Validation error checks
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.compute_buckets_sha256({})
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.compute_buckets_sha256({"expert_0": "bad"})  # type: ignore[dict-item]
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.verify_report_provenance("bad")  # type: ignore[arg-type]
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.save_checkpoint_safetensors({}, tmp_path / "empty.safetensors")
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.compute_file_sha256(tmp_path / "missing.bin")
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.render_markdown_report({"object": "unknown"})
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.inspect_moe_bucket_safetensors([], **geom)  # type: ignore[arg-type]
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.inspect_moe_bucket_safetensors({"": torch.ones(4)}, **geom)
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.inspect_moe_bucket_safetensors({"bad_key": torch.ones(4)}, **geom)
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.inspect_moe_bucket_safetensors({"expert_9": torch.ones(4), "router_frag_0": torch.ones(4)}, **geom)
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.inspect_moe_bucket_safetensors({"router_frag_0": torch.ones(4)}, **geom)
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.inspect_moe_bucket_safetensors({"expert_0": torch.ones(4)}, **geom)
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.inspect_moe_bucket_safetensors({"router_frag_0": torch.ones(3), "expert_0": torch.ones(4)}, **geom)
    with pytest.raises(doctor.CheckpointDoctorError):
        bad_r = dict(buckets)
        bad_r["router_frag_0"] = torch.ones(16 * 4 + 32)
        doctor.inspect_moe_bucket_safetensors(bad_r, **geom)
    with pytest.raises(doctor.CheckpointDoctorError):
        bad_e = dict(buckets)
        bad_e["expert_0"] = torch.ones(5)
        doctor.inspect_moe_bucket_safetensors(bad_e, **geom)
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.inspect_moe_bucket_safetensors(buckets, num_layers=True, num_experts=4, rank=4, hidden_dim=16, inter_dim=24)  # type: ignore[arg-type]
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.inspect_moe_bucket_safetensors(buckets, num_layers=0, num_experts=4, rank=4, hidden_dim=16, inter_dim=24)
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.inspect_moe_bucket_safetensors(buckets, **geom, rogue_b_norm_threshold=-1.0)
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.inspect_moe_bucket_safetensors(buckets, **geom, spectral_spike_ratio_threshold=0.0)
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.apply_offline_ghost_sink_and_ln_surgery(buckets, sink_experts=[0], lambda_ln=1.5, **geom)
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.apply_offline_ghost_sink_and_ln_surgery(buckets, sink_experts="0", **geom)  # type: ignore[arg-type]
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.apply_offline_ghost_sink_and_ln_surgery(buckets, sink_experts=[99], **geom)
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.apply_offline_ghost_sink_and_ln_surgery(buckets, sink_experts=[0], active_layers="0", **geom)  # type: ignore[arg-type]
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.apply_offline_ghost_sink_and_ln_surgery(buckets, sink_experts=[0], active_layers=[99], **geom)
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.apply_offline_ghost_sink_and_ln_surgery(buckets, sink_experts=[0], sanitize_non_finite="yes", **geom)  # type: ignore[arg-type]
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.apply_offline_ghost_sink_and_ln_surgery(buckets, sink_experts=[0], base_ln_tail=torch.ones(3), **geom)
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.apply_offline_ghost_sink_and_ln_surgery(buckets, sink_experts=[0], base_ln_tail=torch.tensor([float("nan")] * 32), **geom)
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.apply_offline_ghost_sink_and_ln_surgery(buckets, sink_experts=[0], base_ln_tail={"expert_0": torch.ones(4)}, **geom)
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.apply_offline_ghost_sink_and_ln_surgery(buckets, sink_experts=[0], base_ln_tail=(1, 2), **geom)  # type: ignore[arg-type]
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.apply_offline_ghost_sink_and_ln_surgery(buckets, sink_experts=[0], base_ln_tail=123, **geom)  # type: ignore[arg-type]
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.apply_offline_ghost_sink_and_ln_surgery({"router_frag_0": torch.ones(4)}, sink_experts=[0])

    empty_st = tmp_path / "empty.safetensors"
    save_file({}, str(empty_st))
    with pytest.raises(doctor.CheckpointDoctorError):
        doctor.load_checkpoint_buckets(empty_st)


# ---------------------------------------------------------------------------
# S5: Lateral Movement AST Audit, ZeroStubAudit, PRSecurityScanner & Gatekeeper
# ---------------------------------------------------------------------------


def test_s5_lateral_movement_zero_stub_pr_security_and_forbidden_added_cleanliness() -> None:
    """Verify 0 violations across lateral-movement, ZeroStubAudit, PRSecurityScanner, and gatekeeper."""
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn09", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn09", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []

    doctor_src = DOCTOR_PATH.read_text(encoding="utf-8")
    doctor_tree = ast.parse(doctor_src, filename="scripts/hivemind_wsp_checkpoint_doctor.py")
    imported_roots: set[str] = set()
    for node in ast.walk(doctor_tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots.isdisjoint({"urllib", "requests", "httpx", "socket", "aiohttp", "subprocess"})

    for rel, path_obj in (
        ("scripts/hivemind_wsp_checkpoint_doctor.py", DOCTOR_PATH),
        ("tests/test_hivemind_wsp_checkpoint_doctor.py", TEST_FILE_PATH),
    ):
        src_text = path_obj.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in src_text[:300], f"Missing AGPL-3.0 header in {rel}"

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

    # Assert output safety guard rejects forbidden pattern without literally placing it in source
    forbidden_payload = '{"' + "sco" + 're": 9}'
    with pytest.raises(doctor.CheckpointDoctorError, match="Output safety guard"):
        doctor.assert_safe_output(forbidden_payload)

    # Negative control: planted LAN URL fails lateral-movement scanner
    bad_host = ".".join(["192", "168", "1", "77"])
    bad_url = "ht" + f"tp://{bad_host}:8080/exfil"
    assert lateral_guard.audit_source_for_lateral_movement(doctor_src + f'\n_BAD = "{bad_url}"\n', "bad.py") != []

    pr_audit = scanner.audit_pull_request(
        pr_number=1109,
        pr_title="feat(swarm): deterministic safetensors open-weight MoE surgery doctor and offline ANSE preflight CLI",
        pr_body="Implements NEURAL-CTRL-DYN-09 offline CrowdLoRA safetensors checkpoint doctor and deterministic CPU scenario tests.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): deterministic safetensors open-weight MoE surgery doctor and offline ANSE preflight CLI"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
