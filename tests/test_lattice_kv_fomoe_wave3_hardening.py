# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Hardening tests for lattice_quantizer, kv_lattice, fomoe_sigma_routing, and wave3_verifiers."""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from hivemind.compress.kv_lattice import (
    _cond_entropy_bits as kv_cond_entropy_bits,
    _entropy_bits as kv_entropy_bits,
    _joint_key,
    from_rows,
    lattice_quantize,
    to_rows,
)
from hivemind.compress.lattice_quantizer import (
    QuantizedDelta,
    _cond_entropy_bits as lq_cond_entropy_bits,
    _entropy_bits as lq_entropy_bits,
    _lattice_points_in_ball,
)
from hivemind.core.fomoe_sigma_routing import (
    SigmaMoERoutingResult,
    sigmoid_independent_topk_routing,
)
from hivemind.solverbee.wave3_verifiers import (
    _cert,
    verify_numeric_solver_trajectory,
)


def test_lattice_quantizer_entropy_and_quantized_delta_validation() -> None:
    c0 = np.array([0, 1, 0, 1], dtype=np.int64)
    c1 = np.array([0, 0, 1, 1], dtype=np.int64)
    assert math.isclose(lq_entropy_bits([c0]), 1.0, rel_tol=1e-9)
    assert math.isclose(lq_cond_entropy_bits(c0), 1.0, rel_tol=1e-9)
    assert math.isclose(lq_cond_entropy_bits(c0, c1), 1.0, rel_tol=1e-9)

    with pytest.raises(ValueError, match="cols must be a non-empty sequence"):
        lq_entropy_bits([])
    with pytest.raises(ValueError, match="cols arrays must be non-empty"):
        lq_entropy_bits([np.array([], dtype=np.int64)])
    with pytest.raises(ValueError, match="x must be a non-empty array"):
        lq_cond_entropy_bits(np.array([], dtype=np.int64))

    pts = _lattice_points_in_ball("A2", 1.5)
    assert pts.shape[1] == 2 and pts.shape[0] > 0
    with pytest.raises(KeyError):
        _lattice_points_in_ball("BAD", 1.5)
    for bad_r in (True, 0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="radius must be a strictly positive finite float"):
            _lattice_points_in_ball("A2", bad_r)  # type: ignore[arg-type]

    deq = torch.tensor([1.0, -1.0])
    res = torch.tensor([0.1, -0.1])
    qd = QuantizedDelta(dequantized=deq, rate_bits_per_dim=1.25, relative_mse=0.01, residual=res)
    assert qd.rate_bits_per_dim == 1.25

    with pytest.raises(ValueError, match="dequantized must be a non-empty"):
        QuantizedDelta(dequantized=torch.tensor([]), rate_bits_per_dim=1.0, relative_mse=0.0, residual=res)
    with pytest.raises(ValueError, match="residual must be a torch.Tensor matching"):
        QuantizedDelta(dequantized=deq, rate_bits_per_dim=1.0, relative_mse=0.0, residual=torch.tensor([0.1]))
    for bad_num in (True, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="rate_bits_per_dim"):
            QuantizedDelta(dequantized=deq, rate_bits_per_dim=bad_num, relative_mse=0.0, residual=res)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="relative_mse"):
            QuantizedDelta(dequantized=deq, rate_bits_per_dim=1.0, relative_mse=bad_num, residual=res)  # type: ignore[arg-type]


def test_kv_lattice_helpers_and_quantize_validation() -> None:
    k0 = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    k1 = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    assert math.isclose(kv_entropy_bits(k0), 1.0, rel_tol=1e-9)
    jk = _joint_key(k0, k1)
    assert jk.shape == k0.shape
    assert math.isclose(kv_cond_entropy_bits(k0), 1.0, rel_tol=1e-9)
    assert math.isclose(kv_cond_entropy_bits(k0, k1), 1.0, rel_tol=1e-9)

    with pytest.raises(ValueError, match="key must be a non-empty"):
        kv_entropy_bits(torch.tensor([], dtype=torch.int64))
    with pytest.raises(ValueError, match="cols must be non-empty"):
        _joint_key()
    with pytest.raises(ValueError, match="cols must be non-empty"):
        _joint_key(torch.tensor([], dtype=torch.int64))
    with pytest.raises(ValueError, match="x must be a non-empty"):
        kv_cond_entropy_bits(torch.tensor([], dtype=torch.int64))

    x = torch.randn(2, 8, dtype=torch.float64)
    rows, _ = to_rows("E8", x)
    rec = from_rows("E8", rows, x.shape)
    assert torch.allclose(rec, x)

    with pytest.raises(KeyError):
        from_rows("BAD", rows, x.shape)
    with pytest.raises(ValueError, match="rows must be a non-empty"):
        from_rows("E8", torch.tensor([]), x.shape)
    with pytest.raises(ValueError, match="shape must be a non-empty"):
        from_rows("E8", rows, ())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape must be a non-empty"):
        from_rows("E8", rows, (True, 8))  # type: ignore[arg-type]

    x_hat, rate = lattice_quantize("A2", x, 0.25, measure_rate=True)
    assert x_hat.shape == x.shape and math.isfinite(rate)
    with pytest.raises(ValueError, match="measure_rate must be a boolean"):
        lattice_quantize("A2", x, 0.25, measure_rate=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="x must be a non-empty"):
        lattice_quantize("A2", "bad", 0.25)  # type: ignore[arg-type]


def test_fomoe_sigma_routing_result_validation() -> None:
    logits = torch.tensor([[1.0, -0.5, 0.2, 2.0], [-1.0, 0.8, 1.5, -0.2]])
    res = sigmoid_independent_topk_routing(logits, top_k=2, resident_mask=torch.tensor([True, False, True, False]))
    assert res.total_experts == 4 and res.resident_experts == 2

    with pytest.raises(ValueError, match="expert_indices must be a non-empty 2D"):
        SigmaMoERoutingResult(
            expert_indices=torch.tensor([0, 1]),
            gate_values=res.gate_values,
            skipped_slots=res.skipped_slots,
            total_experts=4,
            resident_experts=2,
            top_k=2,
            ghost_assignment_fraction=0.5,
            ideal_skip_token_scaling=0.5,
            empirical_skip_token_scaling=0.5,
        )
    with pytest.raises(ValueError, match="gate_values must match"):
        SigmaMoERoutingResult(
            expert_indices=res.expert_indices,
            gate_values=torch.zeros(1, 2),
            skipped_slots=res.skipped_slots,
            total_experts=4,
            resident_experts=2,
            top_k=2,
            ghost_assignment_fraction=0.5,
            ideal_skip_token_scaling=0.5,
            empirical_skip_token_scaling=0.5,
        )
    with pytest.raises(ValueError, match="skipped_slots must match"):
        SigmaMoERoutingResult(
            expert_indices=res.expert_indices,
            gate_values=res.gate_values,
            skipped_slots=torch.zeros(1, 2, dtype=torch.bool),
            total_experts=4,
            resident_experts=2,
            top_k=2,
            ghost_assignment_fraction=0.5,
            ideal_skip_token_scaling=0.5,
            empirical_skip_token_scaling=0.5,
        )
    with pytest.raises(ValueError, match="total_experts must be an integer >= 1"):
        SigmaMoERoutingResult(
            expert_indices=res.expert_indices,
            gate_values=res.gate_values,
            skipped_slots=res.skipped_slots,
            total_experts=0,
            resident_experts=2,
            top_k=2,
            ghost_assignment_fraction=0.5,
            ideal_skip_token_scaling=0.5,
            empirical_skip_token_scaling=0.5,
        )
    with pytest.raises(ValueError, match="resident_experts must satisfy"):
        SigmaMoERoutingResult(
            expert_indices=res.expert_indices,
            gate_values=res.gate_values,
            skipped_slots=res.skipped_slots,
            total_experts=4,
            resident_experts=5,
            top_k=2,
            ghost_assignment_fraction=0.5,
            ideal_skip_token_scaling=0.5,
            empirical_skip_token_scaling=0.5,
        )
    with pytest.raises(ValueError, match="top_k must satisfy"):
        SigmaMoERoutingResult(
            expert_indices=res.expert_indices,
            gate_values=res.gate_values,
            skipped_slots=res.skipped_slots,
            total_experts=4,
            resident_experts=2,
            top_k=0,
            ghost_assignment_fraction=0.5,
            ideal_skip_token_scaling=0.5,
            empirical_skip_token_scaling=0.5,
        )
    with pytest.raises(ValueError, match="ghost_assignment_fraction must be a finite number"):
        SigmaMoERoutingResult(
            expert_indices=res.expert_indices,
            gate_values=res.gate_values,
            skipped_slots=res.skipped_slots,
            total_experts=4,
            resident_experts=2,
            top_k=2,
            ghost_assignment_fraction=1.2,
            ideal_skip_token_scaling=0.5,
            empirical_skip_token_scaling=0.5,
        )


def test_wave3_verifiers_cert_and_numeric_spec_id_validation() -> None:
    cert = _cert(
        "spec-01",
        "science_domain",
        [{"name": "c1", "status": "pass"}],
        [],
        {"k": 1},
    )
    assert cert["verdict"] == "VERIFIED" and cert["id"] == "spec-01"

    with pytest.raises(ValueError, match="spec_id must be a non-empty string"):
        _cert("   ", "science_domain", [], [], {})
    with pytest.raises(ValueError, match="domain must be a non-empty string"):
        _cert("spec-01", "", [], [], {})
    with pytest.raises(ValueError, match="checks must be a list"):
        _cert("spec-01", "science_domain", (), [], {})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reasons must be a list"):
        _cert("spec-01", "science_domain", [], (), {})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="payload_for_hash must be a dict"):
        _cert("spec-01", "science_domain", [], [], [])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unverified must be a boolean"):
        _cert("spec-01", "science_domain", [], [], {}, unverified=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="spec_id must be None or a non-empty string"):
        verify_numeric_solver_trajectory({}, [], [], spec_id="   ")
