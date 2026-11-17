"""Positive and negative controls for compress and economy input hardening (Iteration 110)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hivemind.compress.kv_lattice import (
    factorised_rate_t,
    is_a2_point,
    nearest_a2_t,
    nearest_e8_t,
    to_rows,
)
from hivemind.compress.lattice_quantizer import (
    LatticeDeltaQuantizer,
    SignDeltaQuantizer,
    calibrate_scale_for_rate,
    is_lattice_point,
    nearest_a2,
)
from hivemind.economy.compute_coin import (
    cobblestones_from_flops,
    kelly_equilibrium,
    tullock_equilibrium,
)
from hivemind.economy.international_launch_model import (
    compute_erlang_c_metrics,
    project_international_launch_wave,
)


def test_kv_lattice_positive_and_negative_controls() -> None:
    x2 = torch.tensor([[0.1, 0.2], [0.5, 0.8]], dtype=torch.float64)
    c2 = nearest_a2_t(x2)
    assert bool(is_a2_point(c2).all())

    x8 = torch.zeros((2, 8), dtype=torch.float64)
    c8 = nearest_e8_t(x8)
    assert c8.shape == (2, 8)

    rows, ctx = to_rows("A2", torch.ones((2, 4), dtype=torch.float64))
    assert rows.shape == (4, 2)
    assert factorised_rate_t("A2", nearest_a2_t(rows), ctx) >= 0.0

    with pytest.raises(ValueError, match="non-empty finite torch.Tensor"):
        nearest_e8_t("not-a-tensor")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-empty finite torch.Tensor"):
        nearest_a2_t(torch.tensor([[float("nan"), 0.0]]))

    with pytest.raises(ValueError, match="last dimension 2"):
        is_a2_point(torch.tensor([[float("nan"), 0.0]]))

    with pytest.raises(ValueError, match="non-empty 2D tensor"):
        factorised_rate_t("A2", [1, 2], ctx)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="1D tensor"):
        factorised_rate_t("A2", c2, [0, 1])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-empty"):
        to_rows("A2", torch.tensor([[float("nan"), 1.0]]))


def test_lattice_quantizer_positive_and_negative_controls() -> None:
    arr = np.array([[0.1, 0.2], [0.4, 0.9]])
    c = nearest_a2(arr)
    assert bool(np.all(is_lattice_point("A2", c)))

    lq = LatticeDeltaQuantizer("A2", scale=0.5)
    qd = lq.quantize(torch.tensor([0.1, 0.2, -0.3, 0.4]))
    assert qd.dequantized.shape == (4,)

    sq = SignDeltaQuantizer()
    sqd = sq.quantize(torch.tensor([0.1, -0.2]))
    assert sqd.dequantized.shape == (2,)

    with pytest.raises(ValueError, match="non-empty finite array"):
        nearest_a2(np.array([[np.nan, 0.0]]))

    with pytest.raises(ValueError, match="non-empty finite array"):
        is_lattice_point("Z", np.array([[np.inf, 0.0]]))

    with pytest.raises(ValueError, match="torch.Tensor"):
        lq.quantize([0.1, 0.2])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="torch.Tensor"):
        sq.quantize([0.1, 0.2])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="torch.Tensor"):
        calibrate_scale_for_rate("Z", [0.1, 0.2], 2.0)  # type: ignore[arg-type]


def test_compute_coin_positive_and_negative_controls() -> None:
    assert cobblestones_from_flops(432e9) == pytest.approx(1.0)
    keq = kelly_equilibrium([10.0, 20.0], capacity=2.0)
    assert keq.efficiency >= 0.75
    teq = tullock_equilibrium(100.0, [1.0, 2.0])
    assert teq.total_supply > 0.0

    with pytest.raises(ValueError, match="flops must be finite"):
        cobblestones_from_flops(True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="valuations must be finite"):
        kelly_equilibrium([True, 2.0])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="capacity and tol"):
        kelly_equilibrium([1.0, 2.0], capacity=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="costs must be finite"):
        tullock_equilibrium(True, [1.0, 2.0])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="costs must be finite"):
        tullock_equilibrium(100.0, [True, 2.0])  # type: ignore[list-item]


def test_international_launch_model_positive_and_negative_controls() -> None:
    diag = compute_erlang_c_metrics(active_producers_c=10, aggregate_tok_per_s=480.0, arrival_rate_req_per_s=0.5)
    assert diag.is_stable is True

    proj = project_international_launch_wave(wave_name="Wave-1", new_visitors=500)
    assert proj.new_signups >= 1

    with pytest.raises(ValueError, match="must be finite"):
        compute_erlang_c_metrics(active_producers_c=10, aggregate_tok_per_s=True, arrival_rate_req_per_s=0.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="signup_rate"):
        project_international_launch_wave(wave_name="Wave-1", new_visitors=500, signup_rate=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="uncontrolled_scraping_factor"):
        project_international_launch_wave(wave_name="Wave-1", new_visitors=500, uncontrolled_scraping_factor=True)  # type: ignore[arg-type]
