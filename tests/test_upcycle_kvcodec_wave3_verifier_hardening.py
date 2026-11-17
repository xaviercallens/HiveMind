# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit and negative-control tests for BTM upcycle, KV page codecs, serving experts, and Wave 3 verifiers."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from hivemind.compress.lattice_quantizer import nearest_a2, nearest_e8
from hivemind.core.btm_upcycle import (
    LearnedDepthPrior,
    SparseMoEMLPBlock,
    TopKMoERouter,
    _normalize_local_expert_indices,
    initialize_domain_branches,
    orthogonal_input_injection,
    project_lora_state_dict_to_rank,
    svd_project_lora_pair,
)
from hivemind.core.kv_page_codecs import (
    FP8E4M3PageCodec,
    IntMinMaxPageCodec,
    LatticePageCodec,
    PluggableKVPageCache,
)
from hivemind.core.serving_experts import load_gwaya_code_expert
from hivemind.solverbee.wave3_verifiers import (
    verify_astrophysics_transit_and_orbit,
    verify_genetic_elenchus_thermodynamics,
    verify_quantum_fluid_relaxation,
    verify_topology_tda_and_lattice,
)


def test_initialize_domain_branches_rejects_empty_duplicate_blank_and_nan() -> None:
    good_state = {"layers.0.mlp.gate_proj.lora_A.weight": torch.ones((4, 8))}
    branches = initialize_domain_branches(good_state, ["Code", "STEM"])
    assert list(branches.keys()) == ["Code", "STEM"]
    assert branches["Code"].branch_id == 0
    assert branches["STEM"].branch_id == 1

    with pytest.raises(ValueError, match="must not be empty"):
        initialize_domain_branches(good_state, [])

    with pytest.raises(ValueError, match="non-empty string"):
        initialize_domain_branches(good_state, ["Code", "   "])

    with pytest.raises(ValueError, match="duplicate"):
        initialize_domain_branches(good_state, ["Code", "Code"])

    nan_state = {"layers.0.mlp.gate_proj.lora_A.weight": torch.tensor([[1.0, float("nan")]])}
    with pytest.raises(ValueError, match="non-finite"):
        initialize_domain_branches(nan_state, ["Code"])


def test_svd_project_lora_pair_and_state_dict_validation() -> None:
    a = torch.randn((4, 8), dtype=torch.float32)
    b = torch.randn((8, 4), dtype=torch.float32)
    a_new, b_new, retained = svd_project_lora_pair(a, b, target_rank_r=2, scale_b=1.0)
    assert a_new.shape == (2, 8)
    assert b_new.shape == (8, 2)
    assert 0.0 <= retained <= 1.0

    for bad_rank in (True, False, 0, -2):
        with pytest.raises(ValueError, match="target_rank_r"):
            svd_project_lora_pair(a, b, target_rank_r=bad_rank)  # type: ignore[arg-type]

    for bad_scale in (True, 0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="scale_b"):
            svd_project_lora_pair(a, b, target_rank_r=2, scale_b=bad_scale)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="2-D"):
        svd_project_lora_pair(torch.ones(8), b, target_rank_r=2)

    with pytest.raises(ValueError, match="Incompatible"):
        svd_project_lora_pair(torch.randn((3, 8)), b, target_rank_r=2)

    a_nan = a.clone()
    a_nan[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        svd_project_lora_pair(a_nan, b, target_rank_r=2)

    state = {
        "layers.0.mlp.gate_proj.lora_A.weight": a,
        "layers.0.mlp.gate_proj.lora_B.weight": b,
    }
    with pytest.raises(ValueError, match="target_rank_r"):
        project_lora_state_dict_to_rank(state, target_rank_r=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="source_rank_r"):
        project_lora_state_dict_to_rank(state, target_rank_r=2, source_rank_r=True)  # type: ignore[arg-type]

    for bad_alpha in (True, 0.0, -16.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="source_lora_alpha"):
            project_lora_state_dict_to_rank(state, target_rank_r=2, source_lora_alpha=bad_alpha)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="target_lora_alpha"):
            project_lora_state_dict_to_rank(state, target_rank_r=2, target_lora_alpha=bad_alpha)  # type: ignore[arg-type]


def test_topk_router_local_expert_indices_and_hotplug_validation() -> None:
    router = TopKMoERouter(hidden_size=8, num_experts=4, top_k=2)
    w, idx, aux = router(torch.randn(3, 8))
    assert w.shape == (3, 2)
    assert idx.shape == (3, 2)
    assert torch.isfinite(aux).all()

    with pytest.raises(ValueError, match="hidden_size, num_experts, and top_k"):
        TopKMoERouter(hidden_size=True, num_experts=4, top_k=2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hidden_size, num_experts, and top_k"):
        TopKMoERouter(hidden_size=8, num_experts=0, top_k=2)
    with pytest.raises(ValueError, match="hidden_size, num_experts, and top_k"):
        TopKMoERouter(hidden_size=8, num_experts=4, top_k=False)  # type: ignore[arg-type]

    assert _normalize_local_expert_indices([2, 0, 2], 4) == (0, 2)
    with pytest.raises(ValueError, match="sequence of integers"):
        _normalize_local_expert_indices([True, False], 4)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="sequence of integers"):
        _normalize_local_expert_indices("01", 4)  # type: ignore[arg-type]

    experts = [nn.Linear(8, 8), nn.Linear(8, 8)]
    block = SparseMoEMLPBlock(hidden_size=8, experts=experts, expert_domains=["Code", "Math"], top_k=2)
    with pytest.raises(ValueError, match="non-empty string"):
        block.hotplug_expert("   ", nn.Linear(8, 8))
    with pytest.raises(ValueError, match="already registered"):
        block.hotplug_expert("Code", nn.Linear(8, 8))


def test_orthogonal_input_injection_and_learned_depth_prior_validation() -> None:
    h = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    u = torch.tensor([[2.0, 3.0, 0.0, 0.0]])
    out = orthogonal_input_injection(h, u)
    u_perp = out - h
    assert abs(float(torch.sum(u_perp * h).item())) < 1e-6

    with pytest.raises(ValueError, match="identical shapes"):
        orthogonal_input_injection(torch.ones(2, 4), torch.ones(3, 4))

    for bad_eps in (True, 0.0, -1e-8, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="eps"):
            orthogonal_input_injection(h, u, eps=bad_eps)  # type: ignore[arg-type]

    h_nan = h.clone()
    h_nan[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        orthogonal_input_injection(h_nan, u)

    with pytest.raises(ValueError, match="hidden_size >= 1"):
        LearnedDepthPrior(hidden_size=True, loop_steps=4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="entropy_coef and budget_coef"):
        LearnedDepthPrior(hidden_size=8, loop_steps=4, entropy_coef=float("nan"))
    with pytest.raises(ValueError, match="entropy_coef and budget_coef"):
        LearnedDepthPrior(hidden_size=8, loop_steps=4, budget_coef=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="target_budget_steps"):
        LearnedDepthPrior(hidden_size=8, loop_steps=4, target_budget_steps=0.0)
    with pytest.raises(ValueError, match="target_budget_steps"):
        LearnedDepthPrior(hidden_size=8, loop_steps=4, target_budget_steps=True)  # type: ignore[arg-type]


def test_kv_page_codecs_and_pluggable_cache_validation() -> None:
    for bad_bits in (True, False, 0, 17, 4.5):
        with pytest.raises(ValueError, match="bits"):
            IntMinMaxPageCodec(bad_bits)  # type: ignore[arg-type]

    for bad_scale in (True, 0.0, -0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="scale"):
            LatticePageCodec("E8", bad_scale)  # type: ignore[arg-type]

    for codec in (FP8E4M3PageCodec(), IntMinMaxPageCodec(4), LatticePageCodec("Z", 0.5)):
        with pytest.raises(ValueError, match="non-empty tensor"):
            codec.encode(torch.empty((0, 8)))
        with pytest.raises(ValueError, match="non-empty tensor"):
            codec.encode(torch.tensor(1.0))

    with pytest.raises(ValueError, match="max_pages"):
        PluggableKVPageCache(max_pages=0)
    with pytest.raises(ValueError, match="num_layers"):
        PluggableKVPageCache(num_layers=True)  # type: ignore[arg-type]

    cache = PluggableKVPageCache(max_pages=2)
    k = torch.randn(4, 16)
    v = torch.randn(4, 16)
    with pytest.raises(ValueError, match="layer_idx"):
        cache.write_kv_page(-1, 0, k, v)
    with pytest.raises(ValueError, match="page_idx"):
        cache.write_kv_page(0, True, k, v)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="matching shapes"):
        cache.write_kv_page(0, 0, k, torch.randn(4, 8))
    with pytest.raises(ValueError, match="layer_idx"):
        cache.read_kv_page(True, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="page_idx"):
        cache.read_kv_page(0, -1)


def test_serving_experts_load_gwaya_validation() -> None:
    with pytest.raises(ValueError, match="large_url"):
        load_gwaya_code_expert("   ")
    with pytest.raises(ValueError, match="large_model"):
        load_gwaya_code_expert("http://127.0.0.1:8080", large_model="")
    with pytest.raises(ValueError, match="small_url"):
        load_gwaya_code_expert("http://127.0.0.1:8080", small_url="   ")
    with pytest.raises(ValueError, match="small_model"):
        load_gwaya_code_expert("http://127.0.0.1:8080", small_model=" ")
    with pytest.raises(ValueError, match="small_first"):
        load_gwaya_code_expert("http://127.0.0.1:8080", small_first=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_rounds"):
        load_gwaya_code_expert("http://127.0.0.1:8080", max_rounds=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_rounds"):
        load_gwaya_code_expert("http://127.0.0.1:8080", max_rounds=0)


def test_wave3_verifiers_reject_abusive_tolerances_and_invalid_dimensions() -> None:
    # 1. Quantum fluid: loose/infinite/non-positive/bool force_rel_tol must fail closed
    for bad_tol in (float("inf"), float("nan"), 0.0, -1e-9, 0.5, True):
        qf_cert = verify_quantum_fluid_relaxation(
            energy_init=-120.5,
            energy_final=-125.9,
            max_free_force=1e-3,
            v_prime_scale=1.0,
            kappa=1.5,
            reported_defect_charge_sum=0,
            force_rel_tol=bad_tol,  # type: ignore[arg-type]
        )
        assert qf_cert["verdict"] == "FAILED"

    # 2. Astrophysics: loose/infinite depth_rel_tol or conservation_rel_tol or mismatched series length
    for bad_dtol in (float("inf"), float("nan"), 0.0, -0.01, 0.9, True):
        astro_cert = verify_astrophysics_transit_and_orbit(
            r_planet_earth=1.0,
            r_star_sun=1.0,
            reported_depth_ppm=84.0,
            energy_series=[-0.5, -0.5],
            ang_mom_series=[1.0, 1.0],
            depth_rel_tol=bad_dtol,  # type: ignore[arg-type]
        )
        assert astro_cert["verdict"] == "FAILED"

    for bad_ctol in (float("inf"), float("nan"), 0.0, -1e-5, 0.1, True):
        astro_cert2 = verify_astrophysics_transit_and_orbit(
            r_planet_earth=1.0,
            r_star_sun=1.0,
            reported_depth_ppm=84.0,
            energy_series=[-0.5, -0.5],
            ang_mom_series=[1.0, 1.0],
            conservation_rel_tol=bad_ctol,  # type: ignore[arg-type]
        )
        assert astro_cert2["verdict"] == "FAILED"

    mismatched_astro = verify_astrophysics_transit_and_orbit(
        r_planet_earth=1.0,
        r_star_sun=1.0,
        reported_depth_ppm=84.0,
        energy_series=[-0.5, -0.5, -0.5],
        ang_mom_series=[1.0, 1.0],
    )
    assert mismatched_astro["verdict"] == "UNVERIFIED"

    # 3. Genetic elenchus: loose/infinite/negative/bool prob_tol must fail closed
    vec8 = [0.4, -0.3, 0.6, 0.1, -0.2, 0.5, 0.0, -0.1]
    opt_e8 = nearest_e8(np.asarray(vec8, dtype=float).reshape(1, 8))[0].tolist()
    for bad_ptol in (float("inf"), float("nan"), -1e-9, 0.2, True):
        gen_cert = verify_genetic_elenchus_thermodynamics(
            prob_vector=[0.25, 0.5, 0.25],
            mean_work_kbt=3.4,
            delta_f_kbt=2.1,
            dicodon_vec_8d=vec8,
            reported_e8_codeword=opt_e8,
            prob_tol=bad_ptol,  # type: ignore[arg-type]
        )
        assert gen_cert["verdict"] == "FAILED"

    # 4. Topology TDA: bool/non-positive/non-int spatial_dim must return UNVERIFIED
    pt2 = [0.8, 0.5]
    opt_a2 = nearest_a2(np.asarray(pt2, dtype=float).reshape(1, 2))[0].tolist()
    for bad_dim in (True, False, 0, -2, 2.5):
        topo_cert = verify_topology_tda_and_lattice(
            betti_numbers=[1, 2, 1],
            reported_euler_char=0,
            point=pt2,
            lattice_name="A2",
            riesz_s=3.5,
            spatial_dim=bad_dim,  # type: ignore[arg-type]
            reported_nearest_point=opt_a2,
        )
        assert topo_cert["verdict"] == "UNVERIFIED"
