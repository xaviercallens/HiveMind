"""Unit tests for HM-S29-02 Extreme Top-1% Quantized DCT + EF21 100x WAN Compressor."""

from __future__ import annotations

import math
import pytest

from hivemind.swarm.top1pct_dct_ef21_compressor import (
    Top1PctQuantizedDCTEF21Compressor,
    evaluate_7division_top1pct_compression_benchmark,
)


def test_top1pct_compressor_parameter_validation() -> None:
    with pytest.raises(ValueError):
        Top1PctQuantizedDCTEF21Compressor(top_k_fraction=0.5)
    with pytest.raises(ValueError):
        Top1PctQuantizedDCTEF21Compressor(quantization_bits=8)
    with pytest.raises(ValueError):
        Top1PctQuantizedDCTEF21Compressor(dampening_gamma=0.0)


def test_top1pct_compression_ratio_and_ef21_conservation() -> None:
    comp = Top1PctQuantizedDCTEF21Compressor(top_k_fraction=0.01, quantization_bits=4)
    assert comp.extra_gpu_vram_mib == 0.0

    raw = [math.cos(i * 0.15) * math.exp(-i / 1200.0) for i in range(4096)]
    pkt, recon = comp.compress_and_reconstruct("div1_photorealistic_diffusion", raw, step_index=1)

    assert pkt.total_elements == 4096
    assert pkt.retained_k == 41
    assert pkt.compression_ratio >= 100.0
    assert pkt.conservation_max_abs_err < 1e-9
    assert pkt.ef21_contraction_factor < 1.0
    assert len(recon) == 4096


def test_7division_vision_moe_benchmark_gate() -> None:
    report = evaluate_7division_top1pct_compression_benchmark()
    assert report["task_id"] == "HM-S29-02"
    assert report["gate_passed"] is True
    assert report["mean_compression_ratio"] >= 100.0
    assert report["max_held_out_loss_degradation_pct"] <= 0.5
    assert report["max_conservation_abs_error"] < 1e-9
    assert report["extra_gpu_vram_mib"] == 0.0
    assert len(report["divisions"]) == 7
