# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Validation, routing-edge and bucket round-trip tests for the CrowdGPT-MoE OpenDiLoCo module.

Complements `test_crowdgpt_moe_opendiloco_weekend.py` with the branches it does not reach: dataset-plan
refusals against a hand-built registry, shard assignment validation and the low-VRAM tier, both MoE MLPs
under `softmax_topk`, resident-expert masks and saturated routers, the fine-grained width-split upcycle,
exact extract/load bucket round trips, coordinator refusals and bucket filtering, and inner-step guards.
All on CPU with tiny models; no float result is compared across machines except exact identities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from hivemind.swarm.crowdgpt_live_runtime import CrowdRuntimeMLP, CrowdV1RuntimeConfig, CrowdV1WithEngram
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CROWDGPT_MOE_8_EXPERTS,
    CrowdFineGrainedMoEMLP,
    CrowdLoRAExpertAdapter,
    CrowdLoRAMoEMLP,
    CrowdOpenDiLoCoCoordinator,
    _switch_load_balancing_loss,
    assign_hiveminder_moe_shards,
    collect_moe_router_aux_loss,
    extract_moe_trainable_buckets,
    load_crowdgpt_moe_dataset_plan,
    load_moe_trainable_buckets,
    run_crowd_moe_opendiloco_inner_steps,
    upcycle_crowd_v1_to_sparse_moe,
)
from hivemind.swarm.decoupled_diloco import LearnerArrival


def _small_model(seed: int = 20261114) -> CrowdV1WithEngram:
    torch.manual_seed(seed)
    cfg = CrowdV1RuntimeConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        max_position_embeddings=64,
        engram_buckets=64,
    )
    model = CrowdV1WithEngram(cfg)
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim >= 2:
                p.mul_(0.02)
    return model


def _dense(seed: int = 7) -> CrowdRuntimeMLP:
    torch.manual_seed(seed)
    return CrowdRuntimeMLP(CrowdV1RuntimeConfig(hidden_size=32, intermediate_size=64))


def _stream(n: int = 160) -> torch.Tensor:
    return torch.tensor([(i * 5 + 7) % 64 for i in range(n)], dtype=torch.long)


# ---------------------------------------------------------------------------
# Dataset plan & shard assignment
# ---------------------------------------------------------------------------


def _write_registry(path: Path, *, drop_expert: int | None = None, verified: bool = True) -> Path:
    rows: list[dict[str, Any]] = []
    for spec in CROWDGPT_MOE_8_EXPERTS:
        for ds in spec.primary_datasets:
            status = "REJECTED" if spec.expert_index == drop_expert else "ACCEPTED"
            rows.append({"id": ds, "status": status, "license": "apache-2.0", "verified_signal": verified,
                         "revision_sha": "abc123", "config": "default"})
    rows.append({"id": "someone/non-commercial-set", "status": "REJECTED", "license": "cc-by-nc-4.0"})
    path.write_text(json.dumps({"registry": rows}), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"unverified_cap": 1.5}, "unverified_cap"),
        ({"unverified_cap": True}, "unverified_cap"),
        ({"seed": -1}, "seed"),
        ({"seed": 1.0}, "seed"),
    ],
)
def test_dataset_plan_rejects_invalid_policy_arguments(tmp_path: Path, kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        load_crowdgpt_moe_dataset_plan(_write_registry(tmp_path / "reg.json"), **kwargs)


def test_dataset_plan_refuses_an_expert_with_no_accepted_dataset(tmp_path: Path) -> None:
    reg = _write_registry(tmp_path / "reg.json", drop_expert=2)
    with pytest.raises(RuntimeError, match=CROWDGPT_MOE_8_EXPERTS[2].expert_id):
        load_crowdgpt_moe_dataset_plan(reg)


def test_dataset_plan_from_custom_registry_is_deterministic_and_lists_rejections(tmp_path: Path) -> None:
    reg = _write_registry(tmp_path / "reg.json", verified=False)
    plan = load_crowdgpt_moe_dataset_plan(reg, unverified_cap=0.2, seed=5)
    again = load_crowdgpt_moe_dataset_plan(reg, unverified_cap=0.2, seed=5)
    assert plan == again
    assert plan["num_experts"] == 8 and plan["seed"] == 5
    assert plan["rejected_non_allowlisted_datasets"] == ["someone/non-commercial-set"]
    by_id = {e["expert_id"]: e for e in plan["experts"]}
    general = by_id[CROWDGPT_MOE_8_EXPERTS[0].expert_id]
    assert general["has_execution_or_gold_verified_source"] is True  # fineweb anchor is always verified
    assert general["effective_unverified_cap"] == 0.2
    code = by_id[CROWDGPT_MOE_8_EXPERTS[1].expert_id]
    assert code["has_execution_or_gold_verified_source"] is False
    assert code["effective_unverified_cap"] == 1.0  # no verified source -> cap cannot apply
    assert all(s["license"] == "apache-2.0" for s in code["accepted_sources"])
    other_seed = load_crowdgpt_moe_dataset_plan(reg, unverified_cap=0.2, seed=6)
    assert other_seed["experts"][1]["deterministic_shard_prefix"] != code["deterministic_shard_prefix"]


@pytest.mark.parametrize(
    ("args", "kwargs", "match"),
    [
        (("@", "trainer", 8192), {}, "handle"),
        (("alice", "admin", 8192), {}, "role"),
        (("alice", "trainer", -1), {}, "vram_total_mib"),
        (("alice", "trainer", 8192.0), {}, "vram_total_mib"),
        (("alice", "trainer", 8192), {"num_experts": 1}, "num_experts"),
    ],
)
def test_shard_assignment_validation(args: tuple[Any, ...], kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        assign_hiveminder_moe_shards(*args, **kwargs)


def test_shard_assignment_low_vram_tier_and_handle_normalisation() -> None:
    a = assign_hiveminder_moe_shards("@Alice", "Prosumer", 4096)
    b = assign_hiveminder_moe_shards("alice", "prosumer", 4096)
    assert a["handle"] == "Alice" and a["role"] == "prosumer"
    assert a["resident_expert_indices"] == b["resident_expert_indices"]  # case-insensitive and '@'-stripped
    assert len(a["resident_expert_indices"]) == 1
    assert a["min_free_vram_floor_mib"] == 1024 and a["recommended_recipe"] == "light_1expert_lora_or_cpu_eval"
    big = assign_hiveminder_moe_shards("alice", "trainer", 16_000, num_experts=12)
    assert big["resident_expert_indices"] == list(range(12))
    assert len(big["resident_expert_ids"]) == len(CROWDGPT_MOE_8_EXPERTS)  # ids only exist for the canonical 8


# ---------------------------------------------------------------------------
# MoE MLPs
# ---------------------------------------------------------------------------


def test_switch_load_balancing_loss_is_zero_without_tokens_and_one_when_uniform() -> None:
    empty = _switch_load_balancing_loss(torch.zeros(0, 4), torch.zeros(0, 1, dtype=torch.long), 4)
    assert empty.item() == 0.0
    probs = torch.full((8, 4), 0.25)
    top = torch.arange(8).remainder(4).reshape(8, 1)
    assert _switch_load_balancing_loss(probs, top, 4).item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("kwargs", "exc", "match"),
    [
        ({"num_experts": 1}, ValueError, "num_experts"),
        ({"num_experts": True}, ValueError, "num_experts"),
        ({"top_k": 0}, ValueError, "top_k"),
        ({"top_k": 9}, ValueError, "top_k"),
        ({"routing_mode": "hash"}, ValueError, "routing_mode"),
        ({"num_experts": 6}, ValueError, "divisible"),
        ({"resident_expert_indices": (0, 8)}, ValueError, "out of"),
        ({"resident_expert_indices": ()}, ValueError, "At least one"),
    ],
)
def test_fine_grained_moe_validation(kwargs: dict[str, Any], exc: type, match: str) -> None:
    with pytest.raises(exc, match=match):
        CrowdFineGrainedMoEMLP(_dense(), **kwargs)


def test_fine_grained_moe_rejects_non_crowd_mlp() -> None:
    with pytest.raises(TypeError):
        CrowdFineGrainedMoEMLP(torch.nn.Linear(4, 4))  # type: ignore[arg-type]


def test_fine_grained_softmax_all_experts_matches_dense_and_ghost_fraction_counts_skips() -> None:
    dense = _dense()
    x = torch.randn(2, 5, 32)
    with torch.no_grad():
        y_dense = dense(x)
        full = CrowdFineGrainedMoEMLP(dense, num_experts=8, top_k=8, routing_mode="softmax_topk")
        # zero router -> uniform softmax 1/8 each, renormalised and scaled by E: every expert gets gate 1
        assert torch.allclose(full(x), y_dense, atol=1e-5)
        assert full.last_ghost_fraction == 0.0 and full.last_aux_loss.item() == pytest.approx(1.0)

        partial = CrowdFineGrainedMoEMLP(dense, num_experts=8, top_k=8, routing_mode="softmax_topk",
                                         resident_expert_indices=(0, 1))
        y_partial = partial(x)
        assert partial.last_ghost_fraction == pytest.approx(6 / 8)
        exact_two = CrowdFineGrainedMoEMLP(dense, num_experts=8, top_k=2, routing_mode="exact_dense_sum",
                                           resident_expert_indices=(0, 1))
        assert torch.allclose(y_partial, exact_two(x), atol=1e-6)  # only resident slices contribute


def test_fine_grained_router_preference_skips_unhit_experts_and_saturation_gives_zero() -> None:
    dense = _dense()
    x = torch.randn(6, 32)
    moe = CrowdFineGrainedMoEMLP(dense, num_experts=4, top_k=1, routing_mode="softmax_topk")
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[2].fill_(0.0)
        moe.router.weight[0, 0] = 50.0
        x[:, 0] = 1.0  # every token routes to expert 0
        y = moe(x)
        expected = moe._run_expert(0, x) * 4.0  # renormalised gate 1 x E
        assert torch.allclose(y, expected, atol=1e-5)

    sat = CrowdFineGrainedMoEMLP(dense, num_experts=4, top_k=2, routing_mode="sigma_topk")
    with torch.no_grad():
        sat.router.weight.fill_(-1e4)  # sigmoid underflows to exactly 0 for every expert
        xs = torch.ones(3, 32)
        out = sat(xs)
    assert torch.equal(out, torch.zeros_like(out)) and torch.isfinite(sat.last_aux_loss)


def test_lora_expert_adapter_validation_and_zero_delta_at_init() -> None:
    with pytest.raises(ValueError, match="rank"):
        CrowdLoRAExpertAdapter(8, 16, rank=0)
    ad = CrowdLoRAExpertAdapter(8, 16, rank=2, seed=3)
    again = CrowdLoRAExpertAdapter(8, 16, rank=2, seed=3)
    assert torch.equal(ad.w1_A, again.w1_A)  # seeded init is reproducible
    x = torch.randn(4, 8)
    h1, h2 = torch.randn(4, 16), torch.randn(4, 16)
    assert torch.count_nonzero(ad.forward_delta(x, h1, h2, torch.randn(8, 16))) == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_experts": 1}, "num_experts"),
        ({"top_k": 3, "num_experts": 2}, "top_k"),
        ({"routing_mode": "exact_dense_sum"}, "routing_mode"),
        ({"resident_expert_indices": ()}, "non-empty subset"),
        ({"resident_expert_indices": (-1,)}, "non-empty subset"),
    ],
)
def test_lora_moe_validation(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        CrowdLoRAMoEMLP(_dense(), **kwargs)


def test_lora_moe_rejects_non_crowd_mlp() -> None:
    with pytest.raises(TypeError):
        CrowdLoRAMoEMLP(torch.nn.Linear(4, 4))  # type: ignore[arg-type]


def test_lora_moe_softmax_step0_identity_and_router_saturation() -> None:
    dense = _dense()
    x = torch.randn(2, 4, 32)
    with torch.no_grad():
        y_dense = dense(x)
        moe = CrowdLoRAMoEMLP(dense, num_experts=4, top_k=2, rank=2, routing_mode="softmax_topk",
                              resident_expert_indices=(0, 3))
        assert torch.equal(moe(x), y_dense)
        assert 0.0 <= moe.last_ghost_fraction <= 1.0 and moe.last_sigma_report is None
        for e in moe.experts.values():
            e.w1_B.fill_(0.1)  # non-zero experts now change the output, but only through routed tokens
        for e in range(4):
            moe.router.weight[e].fill_(-1e4 - e)  # huge negative but distinct logits: resident expert 0 ranks first
        sat = moe(torch.ones(3, 32))
        # softmax is shift-invariant, so even saturated logits give a positive renormalised gate and expert 0 fires
        assert not torch.equal(sat, dense(torch.ones(3, 32)))
        assert moe.last_ghost_fraction == pytest.approx(0.5)  # top-2 = {0 resident, 1 ghost} for every token

        sig = CrowdLoRAMoEMLP(dense, num_experts=4, top_k=2, rank=2, routing_mode="sigma_topk")
        for e in sig.experts.values():
            e.w1_B.fill_(0.1)
        sig.router.weight.fill_(-1e4)  # every sigmoid gate is exactly 0: no expert runs
        xs = torch.ones(3, 32)
        assert torch.equal(sig(xs), dense(xs))


# ---------------------------------------------------------------------------
# Upcycling, aux loss and bucket round trips
# ---------------------------------------------------------------------------


def test_upcycle_rejects_unknown_split_and_double_upcycling() -> None:
    model = _small_model()
    with pytest.raises(ValueError, match="split_mode"):
        upcycle_crowd_v1_to_sparse_moe(model, split_mode="dense_copy")
    upcycle_crowd_v1_to_sparse_moe(model, num_experts=4, top_k=2, lora_rank=2, layer_indices=[1, 1])
    assert isinstance(model.blocks[0].mlp, CrowdRuntimeMLP) and isinstance(model.blocks[1].mlp, CrowdLoRAMoEMLP)
    with pytest.raises(TypeError, match="already CrowdLoRAMoEMLP"):
        upcycle_crowd_v1_to_sparse_moe(model, num_experts=4, top_k=2, lora_rank=2, layer_indices=[1])


def test_fine_grained_upcycle_trains_only_resident_experts_in_fp32() -> None:
    model = _small_model()
    rep = upcycle_crowd_v1_to_sparse_moe(model, split_mode="fine_grained_width_split", num_experts=4, top_k=2,
                                         routing_mode="softmax_topk", resident_expert_indices=[2, 0])
    assert rep.split_mode == "fine_grained_width_split" and rep.resident_expert_indices == (0, 2)
    assert rep.num_layers_upcycled == 2
    mlp = model.blocks[0].mlp
    assert isinstance(mlp, CrowdFineGrainedMoEMLP)
    assert mlp.experts_w1[0].weight.requires_grad and mlp.experts_w1[0].weight.dtype == torch.float32
    assert not mlp.experts_w1[1].weight.requires_grad
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert rep.trainable_parameters == trainable
    assert rep.trainable_vram_mib_fp32 == round(trainable * 4 / (1024.0 * 1024.0), 3)

    ids = _stream(20).reshape(2, 10)
    model(ids)
    aux = collect_moe_router_aux_loss(model)
    expected = sum(b.mlp.last_aux_loss for b in model.blocks) / 2
    assert torch.allclose(aux, expected)


def test_collect_aux_loss_is_zero_on_a_dense_model() -> None:
    assert collect_moe_router_aux_loss(_small_model()).item() == 0.0


def test_extract_rejects_invalid_fragment_count() -> None:
    for bad in (0, True, 1.5):
        with pytest.raises(ValueError, match="num_router_fragments"):
            extract_moe_trainable_buckets(_small_model(), num_router_fragments=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("split_mode", ["lora_domain_experts", "fine_grained_width_split"])
def test_bucket_extract_load_round_trip_is_exact(split_mode: str) -> None:
    src = _small_model(seed=1)
    upcycle_crowd_v1_to_sparse_moe(src, split_mode=split_mode, num_experts=4, top_k=2, lora_rank=2,
                                   resident_expert_indices=[1, 3])
    buckets = extract_moe_trainable_buckets(src, num_router_fragments=2)
    assert sorted(buckets) == ["expert_1", "expert_3", "router_frag_0", "router_frag_1"]
    g = torch.Generator().manual_seed(4)
    perturbed = {k: v + torch.randn(v.shape, generator=g) for k, v in buckets.items()}

    dst = _small_model(seed=1)
    upcycle_crowd_v1_to_sparse_moe(dst, split_mode=split_mode, num_experts=4, top_k=2, lora_rank=2,
                                   resident_expert_indices=[1, 3])
    load_moe_trainable_buckets(dst, perturbed, num_router_fragments=2)
    again = extract_moe_trainable_buckets(dst, num_router_fragments=2)
    assert again.keys() == perturbed.keys()
    assert all(torch.equal(again[k], perturbed[k]) for k in perturbed)


@pytest.mark.parametrize("split_mode", ["lora_domain_experts", "fine_grained_width_split"])
def test_bucket_load_skips_missing_buckets_and_frozen_parameters(split_mode: str) -> None:
    model = _small_model(seed=2)
    upcycle_crowd_v1_to_sparse_moe(model, split_mode=split_mode, num_experts=4, top_k=2, lora_rank=2,
                                   resident_expert_indices=[0, 1])
    mlp = model.blocks[0].mlp
    frozen = mlp.experts["0"].w1_A if split_mode == "lora_domain_experts" else mlp.experts_w1[0].weight
    frozen.requires_grad_(False)
    frozen_before = frozen.detach().clone()
    buckets = extract_moe_trainable_buckets(model, num_router_fragments=1)
    expert1_before = extract_moe_trainable_buckets(model, num_router_fragments=1)["expert_1"].clone()

    partial = {k: torch.full_like(v, 0.5) for k, v in buckets.items() if k != "expert_1"}
    load_moe_trainable_buckets(model, partial, num_router_fragments=1)
    after = extract_moe_trainable_buckets(model, num_router_fragments=1)
    assert torch.equal(after["expert_1"], expert1_before)  # absent bucket untouched
    assert torch.all(after["expert_0"] == 0.5) and torch.all(after["router_frag_0"] == 0.5)
    assert torch.equal(frozen.detach(), frozen_before)  # frozen tensor neither exported nor overwritten


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


def test_coordinator_refusals_and_in_memory_mode() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        CrowdOpenDiLoCoCoordinator({})
    with pytest.raises(ValueError, match="outer_optimizer_mode"):
        CrowdOpenDiLoCoCoordinator({"a": torch.zeros(3)}, outer_optimizer_mode="sgd")
    with pytest.raises(PermissionError):
        CrowdOpenDiLoCoCoordinator({"a": torch.zeros(3)}, work_dir="/opt/hivemind/t4/coord")
    coord = CrowdOpenDiLoCoCoordinator({"a": torch.arange(3, dtype=torch.float32)})
    assert coord.work_dir is None
    exported = coord.export_buckets()
    exported["a"].fill_(9.0)  # an export is a copy, never a view of the master
    assert torch.equal(coord.export_buckets()["a"], torch.arange(3, dtype=torch.float32))


def test_coordinator_ignores_unknown_and_empty_buckets_and_scales_to_large_cohorts() -> None:
    init = {"router_frag_0": torch.zeros(4), "expert_0": torch.ones(6)}
    coord = CrowdOpenDiLoCoCoordinator(init, min_quorum_k=2, merge_mode="rda", outer_lr=0.5, outer_momentum=0.0)
    sha0 = coord.master_sha256()
    n = 130  # more learners than the default 128-slot syncer: the syncer must be rebuilt, not raise
    pg = torch.full((6,), 0.1)
    subs = {
        "not_a_bucket": [(LearnerArrival("x", 0.1, 1, 10), torch.zeros(3), 0.0)],
        "router_frag_0": [],
        "expert_0": [(LearnerArrival(f"peer-{i}", 0.1 + i * 1e-4, 2, 64), pg, 0.0) for i in range(n)],
    }
    rec = coord.step_decoupled_round(subs)
    assert rec.fragments_synced == ("expert_0",) and rec.accepted_by_loss_ratchet is True
    tel = rec.quorum_telemetry["expert_0"]
    assert len(tel["admitted_learners"]) + tel["deferred_learners"] == n
    assert rec.round_index == 1 and rec.master_state_sha256 != sha0
    out = coord.export_buckets()
    assert torch.equal(out["router_frag_0"], torch.zeros(4))  # empty submission list leaves it untouched
    assert torch.all(out["expert_0"] < 1.0)  # identical pseudo-gradients move the master against them


def test_coordinator_enforces_the_configured_quorum() -> None:
    """One learner must not move a bucket when the quorum is 2 (the quorum used to be lowered to the submission count)."""
    coord = CrowdOpenDiLoCoCoordinator({"expert_0": torch.ones(5)}, min_quorum_k=2, outer_momentum=0.9)
    sha0 = coord.master_sha256()
    rec = coord.step_decoupled_round({"expert_0": [(LearnerArrival("solo", 0.1, 1, 8), torch.full((5,), 0.2), 0.0)]})
    assert coord.master_sha256() == sha0, "a single submission changed the master under a quorum of 2"
    assert "expert_0" not in rec.fragments_synced
    assert rec.quorum_telemetry["expert_0"]["status"] == "quorum_not_reached"
    assert rec.quorum_telemetry["expert_0"]["submissions"] == 1

    pair = [(LearnerArrival(f"l{i}", 0.1 + i * 1e-3, 1, 8), torch.full((5,), 0.2), 0.0) for i in range(2)]
    rec2 = coord.step_decoupled_round({"expert_0": pair})
    assert "expert_0" in rec2.fragments_synced and coord.master_sha256() != sha0


def test_coordinator_rolls_back_on_non_finite_eval_loss() -> None:
    coord = CrowdOpenDiLoCoCoordinator({"expert_0": torch.ones(5)}, min_quorum_k=1, outer_momentum=0.9)
    sha0 = coord.master_sha256()
    losses = iter([1.0, float("nan")])
    rec = coord.step_decoupled_round(
        {"expert_0": [(LearnerArrival("a", 0.1, 1, 8), torch.full((5,), 0.2), 0.0)]},
        eval_fn=lambda _b: next(losses),
    )
    assert rec.accepted_by_loss_ratchet is False and rec.eval_loss_after == rec.eval_loss_before == 1.0
    assert coord.master_sha256() == sha0 and rec.round_index == 0
    assert torch.count_nonzero(coord.nesterov_velocity["expert_0"]) == 0  # velocity restored too


# ---------------------------------------------------------------------------
# Inner steps
# ---------------------------------------------------------------------------


def test_inner_steps_refusals() -> None:
    model = _small_model()
    upcycle_crowd_v1_to_sparse_moe(model, num_experts=4, top_k=2, lora_rank=2)
    with pytest.raises(ValueError, match="steps"):
        run_crowd_moe_opendiloco_inner_steps(model, _stream(), steps=0)
    with pytest.raises(ValueError, match="seq_len \\+ 1"):
        run_crowd_moe_opendiloco_inner_steps(model, _stream(16), seq_len=16)
    frozen = _small_model()
    for p in frozen.parameters():
        p.requires_grad_(False)
    with pytest.raises(RuntimeError, match="zero trainable"):
        run_crowd_moe_opendiloco_inner_steps(frozen, _stream(), seq_len=8)


def test_inner_steps_wrap_a_short_stream_to_fill_the_batch() -> None:
    model = _small_model()
    upcycle_crowd_v1_to_sparse_moe(model, num_experts=4, top_k=2, lora_rank=2)
    before = extract_moe_trainable_buckets(model)
    out = run_crowd_moe_opendiloco_inner_steps(model, _stream(9).reshape(1, 9), steps=1, batch_size=2, seq_len=8,
                                              lr=1e-3)
    assert out["steps_completed"] == 1 and out["yielded"] is None
    assert out["tokens_processed"] == 2 * 7
    assert out["pseudo_grads"].keys() == before.keys()
    after = extract_moe_trainable_buckets(model)
    assert all(torch.equal(out["pseudo_grads"][k], before[k] - after[k]) for k in before)
    assert any(torch.count_nonzero(v) > 0 for v in out["pseudo_grads"].values())
