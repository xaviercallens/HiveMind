"""Boundary validation tests for kv_lattice, lattice_quantizer, auto_improvement, and base_model_catalog."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hivemind.compress.kv_lattice import (
    _closer_t,
    _cond_entropy_bits,
    _entropy_bits,
    _joint_key,
    factorised_rate_t,
    from_rows,
    integer_coordinates,
    is_a2_point,
    lattice_quantize,
    nearest_a2_t,
    nearest_dn_t,
    nearest_e8_t,
    nearest_zn_t,
    to_rows,
)
from hivemind.compress.lattice_quantizer import (
    LatticeDeltaQuantizer,
    QuantizedDelta,
    SignDeltaQuantizer,
    calibrate_scale_for_rate,
    factorised_rate_bits_per_dim,
    is_lattice_point,
    nearest_a2,
    nearest_dn,
    nearest_e8,
    nearest_zn,
)
from hivemind.core.auto_improvement import (
    NocturnalAutoImprovementScheduler,
    PointsRetrofitManager,
    RetrofitReport,
)
from hivemind.core.base_model_catalog import (
    GPU_TIERS,
    SERVE_LADDER,
    TRAIN_LADDER,
    _first_fit,
    load_catalog,
    main as catalog_main,
    recommend_upgrade_plan,
    render_plan,
)


def test_kv_lattice_rejects_bool_tensors_mismatched_shapes_and_bad_dimensions() -> None:
    bool_t = torch.tensor([[True, False]])
    for fn in (nearest_zn_t, nearest_dn_t, nearest_a2_t):
        with pytest.raises(ValueError):
            fn(bool_t)
    with pytest.raises(ValueError):
        nearest_e8_t(torch.ones((1, 8), dtype=torch.bool))
    with pytest.raises(ValueError):
        is_a2_point(bool_t)
    with pytest.raises(ValueError):
        _entropy_bits(bool_t)
    with pytest.raises(ValueError):
        _joint_key(bool_t)
    with pytest.raises(ValueError):
        _joint_key(torch.tensor([1, 2]), torch.tensor([1, 2, 3]))
    with pytest.raises(ValueError):
        _cond_entropy_bits(bool_t)

    a = torch.zeros(2, 2)
    b = torch.ones(2, 2)
    x = torch.full((2, 2), 0.25)
    assert torch.equal(_closer_t(a, b, x), a)
    with pytest.raises(ValueError, match="identical shapes"):
        _closer_t(a, torch.ones(2, 3), x)
    with pytest.raises(ValueError):
        _closer_t(bool_t, b, x)

    with pytest.raises(ValueError, match="dimension 8"):
        integer_coordinates("E8", torch.zeros(2, 4))
    with pytest.raises(ValueError, match="dimension 2"):
        integer_coordinates("A2", torch.zeros(2, 4))
    with pytest.raises(ValueError):
        integer_coordinates("Z", bool_t)

    with pytest.raises(ValueError):
        factorised_rate_t("Z", torch.zeros((2, 8), dtype=torch.bool), torch.zeros(2, dtype=torch.int64))
    with pytest.raises(ValueError):
        factorised_rate_t("Z", torch.zeros((2, 8)), torch.tensor([True, False]))

    with pytest.raises(ValueError):
        to_rows("A2", bool_t)
    with pytest.raises(ValueError):
        from_rows("A2", bool_t, torch.Size([1, 2]))
    with pytest.raises(ValueError, match="even head_dim"):
        from_rows("A2", torch.zeros(2, 2), torch.Size([1, 3]))
    with pytest.raises(ValueError, match="head_dim % 8 == 0"):
        from_rows("E8", torch.zeros(1, 8), torch.Size([1, 6]))
    with pytest.raises(ValueError):
        lattice_quantize("A2", bool_t, 0.5)


def test_lattice_quantizer_rejects_bool_arrays_and_wrong_dimensions() -> None:
    for bad in (True, "123", b"123", np.array([True, False])):
        with pytest.raises(ValueError):
            nearest_zn(bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            nearest_dn(bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            nearest_a2(bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            nearest_e8(bad)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="dimension 8"):
        is_lattice_point("E8", np.zeros((2, 4)))
    with pytest.raises(ValueError, match="dimension 2"):
        factorised_rate_bits_per_dim("A2", np.zeros((2, 4)))
    with pytest.raises(ValueError, match="dimension 8"):
        factorised_rate_bits_per_dim("E8", np.zeros((2, 4)))

    with pytest.raises(ValueError, match="dequantized"):
        QuantizedDelta(
            dequantized=torch.tensor([True, False]),
            rate_bits_per_dim=1.0,
            relative_mse=0.1,
            residual=torch.zeros(2),
        )
    with pytest.raises(ValueError, match="residual"):
        QuantizedDelta(
            dequantized=torch.zeros(2),
            rate_bits_per_dim=1.0,
            relative_mse=0.1,
            residual=torch.tensor([float("nan"), 0.0]),
        )

    lq = LatticeDeltaQuantizer("A2", scale=0.25)
    sq = SignDeltaQuantizer()
    with pytest.raises(ValueError, match="numeric torch.Tensor"):
        lq.quantize(torch.tensor([True, False]))
    with pytest.raises(ValueError, match="numeric torch.Tensor"):
        sq.quantize(torch.tensor([True, False]))
    with pytest.raises(ValueError, match="numeric torch.Tensor"):
        calibrate_scale_for_rate("A2", torch.tensor([True, False]), 1.5)


def test_auto_improvement_scheduler_and_retrofit_validation() -> None:
    sched = NocturnalAutoImprovementScheduler()
    for bad_tag in ("", "   ", "bad\x00tag", True, 123):
        with pytest.raises(ValueError, match="model_or_tag"):
            NocturnalAutoImprovementScheduler._canonical_base_model(bad_tag)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Mapping"):
        sched.filter_eligible_producers(["not-a-mapping"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="account_id"):
        sched.filter_eligible_producers([{"vram_gb": 8.0}])
    with pytest.raises(ValueError, match="account_id"):
        sched.filter_eligible_producers([{"account_id": "   ", "vram_gb": 8.0}])

    with pytest.raises(ValueError, match="base_model_id"):
        sched.create_domain_workunit("maths_formal", num_samples=2, inner_steps=2, base_model_id="   ")

    with pytest.raises(ValueError, match="num_samples"):
        sched._generate_synthetic_refinement_pairs("maths_formal", 0, 42)
    with pytest.raises(ValueError, match="seed"):
        sched._generate_synthetic_refinement_pairs("maths_formal", 2, -1)

    with pytest.raises(ValueError, match="credited_accounts"):
        RetrofitReport(
            total_workunits=1,
            newly_credited_count=1,
            already_credited_count=0,
            total_hmt_granted=10.0,
            total_tflops_attested=1.0,
            credited_accounts={"": 10.0},
            ledger_verified=True,
            timestamp_utc=1000.0,
        )
    with pytest.raises(ValueError, match="credited_accounts"):
        RetrofitReport(
            total_workunits=1,
            newly_credited_count=1,
            already_credited_count=0,
            total_hmt_granted=10.0,
            total_tflops_attested=1.0,
            credited_accounts={"acct-1": -1.0},
            ledger_verified=True,
            timestamp_utc=1000.0,
        )

    mgr = PointsRetrofitManager()
    with pytest.raises(ValueError, match="store"):
        mgr.retrofit_work_points(None, [])


def test_base_model_catalog_plan_and_cli_validation() -> None:
    with pytest.raises(ValueError, match="estimates"):
        _first_fit(["not-an-estimate"], 8000, 0.15)  # type: ignore[list-item]

    cat = load_catalog("configs/base_models_s21.json")
    with pytest.raises(ValueError, match="tiers"):
        recommend_upgrade_plan(cat, tiers=[])
    with pytest.raises(ValueError, match="train_ladder"):
        recommend_upgrade_plan(cat, train_ladder=[])
    with pytest.raises(ValueError, match="serve_ladder"):
        recommend_upgrade_plan(cat, serve_ladder=[])
    with pytest.raises(ValueError, match="margin"):
        recommend_upgrade_plan(cat, margin=1.0)
    with pytest.raises(ValueError, match="margin"):
        recommend_upgrade_plan(cat, margin=True)  # type: ignore[arg-type]

    plan = recommend_upgrade_plan(cat, tiers=GPU_TIERS[:1], train_ladder=TRAIN_LADDER[:1], serve_ladder=SERVE_LADDER[:1])
    rendered = render_plan(plan)
    assert "Upgrade plan" in rendered

    with pytest.raises(ValueError, match="missing required key"):
        render_plan({"kind": "x"})
    bad_plan = dict(plan, margin=True)
    with pytest.raises(ValueError, match="margin"):
        render_plan(bad_plan)

    for bad_argv in ("--json", b"--json", [123]):
        with pytest.raises(ValueError, match="argv"):
            catalog_main(bad_argv)  # type: ignore[arg-type]
