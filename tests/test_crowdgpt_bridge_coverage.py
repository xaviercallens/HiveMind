# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Wire-format, validation and gate-outcome tests for `hivemind.swarm.crowdgpt_bridge`.

Covers the refusals the existing bridge tests leave out: malformed and truncated delta streams, the
engram tail round trip, dict-shaped deltas, invalid audit dataclasses, the zero-participant halving rejection,
non-NVIDIA precision cohorts, forming-pool and resilience decisions, and the CrowdGPT `/fl/task` /
`/fl/submit` header and metadata guards. Pure CPU, no network.
"""

from __future__ import annotations

import gzip
import json
import struct
from typing import Any

import numpy as np
import pytest
import torch

from hivemind.swarm.crowdgpt_bridge import (
    CrowdGPTCrossCheckSpec,
    CrowdGPTDeltaPayload,
    GpuPrecisionCohort,
    LossRatchetDecision,
    MultiKrumAggregationResult,
    PoolDispatchDecision,
    WeightRmsGuardDecision,
    build_crowdgpt_client_hardware_headers,
    build_crowdgpt_submit_metadata,
    check_weight_rms_collapse_guard,
    chunked_pairwise_sq_l2,
    classify_gpu_precision_cohort,
    evaluate_client_resilience_policy,
    multi_krum_select_and_aggregate,
    pack_crowdgpt_delta_stream,
    parse_crowdgpt_task_crosscheck,
    resolve_forming_pool_starvation,
    unpack_crowdgpt_delta_stream,
)

SHA = "a" * 64


def _raw_stream(meta: dict[str, Any], body: bytes) -> bytes:
    pl = json.dumps(meta).encode("utf-8")
    return gzip.compress(struct.pack("<I", len(pl)) + pl + body)


# ---------------------------------------------------------------------------
# Delta stream wire format
# ---------------------------------------------------------------------------


def test_delta_payload_dataclass_refusals() -> None:
    bf = torch.zeros(2, dtype=torch.bfloat16)
    arr = np.zeros(0, dtype=np.uint32)
    with pytest.raises(TypeError, match="metadata"):
        CrowdGPTDeltaPayload([], bf, arr, arr, SHA)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bfloat16"):
        CrowdGPTDeltaPayload({}, torch.zeros(2), arr, arr, SHA)
    with pytest.raises(TypeError, match="numpy"):
        CrowdGPTDeltaPayload({}, bf, [1], arr, SHA)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="64-character"):
        CrowdGPTDeltaPayload({}, bf, arr, arr, "abc")


def test_pack_refusals_and_fp32_input_is_cast_to_bf16() -> None:
    with pytest.raises(TypeError, match="Mapping"):
        pack_crowdgpt_delta_stream([("a", 1)], torch.zeros(2))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="torch.Tensor"):
        pack_crowdgpt_delta_stream({}, np.zeros(2))  # type: ignore[arg-type]
    for bad in (10, -1, True):
        with pytest.raises(ValueError, match="compresslevel"):
            pack_crowdgpt_delta_stream({}, torch.zeros(2), compresslevel=bad)
    vals = torch.tensor([[0.5, -1.25], [3.0, 0.0]])
    out = unpack_crowdgpt_delta_stream(pack_crowdgpt_delta_stream({"taskId": "t"}, vals, compresslevel=0))
    assert torch.equal(out.base_delta_bf16, vals.reshape(-1).to(torch.bfloat16))
    assert out.metadata["hasEngram"] is False and out.metadata["engramSparseCount"] == 0


def test_engram_tail_round_trip_is_exact() -> None:
    base = torch.linspace(-1, 1, 10).to(torch.bfloat16)
    idx = np.array([3, 17, 4_000_000], dtype=np.uint32)
    vals = (np.arange(3 * 4, dtype=np.uint16) * 257).astype(np.uint16)
    blob = pack_crowdgpt_delta_stream({"taskId": "eng", "isDelta": False}, base, idx, vals)
    out = unpack_crowdgpt_delta_stream(blob, engram_row_dim=4)
    assert out.metadata["hasEngram"] is True and out.metadata["engramSparseCount"] == 3
    assert out.metadata["isDelta"] is False  # caller-supplied keys are kept, defaults only fill gaps
    assert torch.equal(out.base_delta_bf16, base)
    assert np.array_equal(out.engram_indices, idx) and np.array_equal(out.engram_values_uint16, vals)


@pytest.mark.parametrize(
    ("blob", "kwargs", "match"),
    [
        (b"", {}, "non-empty bytes"),
        ("text", {}, "non-empty bytes"),
        (gzip.compress(b"abc"), {}, "4-byte header"),
        (gzip.compress(struct.pack("<I", 50) + b"{}"), {}, "truncated inside JSON"),
        (_raw_stream({"hasEngram": True, "engramSparseCount": -1}, b""), {}, "cannot be negative"),
        (_raw_stream({"hasEngram": True, "engramSparseCount": 2}, b"\x00" * 10), {"engram_row_dim": 4},
         "sparse engram tail"),
        (_raw_stream({}, b"\x00\x00\x00"), {}, "divisible by 2"),
        (_raw_stream({}, b"\x00\x00"), {"engram_row_dim": 0}, "engram_row_dim"),
    ],
)
def test_unpack_refuses_malformed_streams(blob: Any, kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        unpack_crowdgpt_delta_stream(blob, **kwargs)


def test_unpack_ignores_sparse_count_when_engram_flag_is_off() -> None:
    out = unpack_crowdgpt_delta_stream(_raw_stream({"hasEngram": False, "engramSparseCount": 99}, b"\x80\x3f" * 3))
    assert out.base_delta_bf16.numel() == 3 and torch.all(out.base_delta_bf16 == 1.0)
    assert out.engram_indices.size == 0


# ---------------------------------------------------------------------------
# Distances, Multi-Krum, dataclass audits
# ---------------------------------------------------------------------------


def test_pairwise_distances_on_dict_deltas_use_sorted_keys() -> None:
    a = {"b": torch.tensor([1.0]), "a": torch.tensor([0.0, 0.0])}
    b = {"a": torch.tensor([3.0, 4.0]), "b": torch.tensor([1.0])}
    d = chunked_pairwise_sq_l2([a, b], chunk_elements=1)
    assert torch.equal(d, torch.tensor([[0.0, 25.0], [25.0, 0.0]]))
    assert torch.equal(chunked_pairwise_sq_l2([{}, {}]), torch.zeros(2, 2))


@pytest.mark.parametrize(
    ("deltas", "kwargs", "match"),
    [
        ([torch.zeros(2)], {"chunk_elements": 0}, "chunk_elements"),
        ([], {}, "non-empty"),
        ([torch.zeros(2), torch.zeros(3)], {}, "expected 2"),
        ([torch.zeros(2), torch.tensor([0.0, float("inf")])], {}, "non-finite"),
    ],
)
def test_pairwise_distance_refusals(deltas: list[torch.Tensor], kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        chunked_pairwise_sq_l2(deltas, **kwargs)


def test_multi_krum_refusals_and_single_learner() -> None:
    with pytest.raises(ValueError, match="At least one"):
        multi_krum_select_and_aggregate([])
    with pytest.raises(ValueError, match="length must match"):
        multi_krum_select_and_aggregate([torch.zeros(2), torch.zeros(2)], token_weights=[1])
    for bad in (0, -5, float("nan"), True, "10"):
        with pytest.raises(ValueError, match="finite and positive"):
            multi_krum_select_and_aggregate([torch.zeros(2)], token_weights=[bad])
    solo = multi_krum_select_and_aggregate([torch.tensor([1.0, 2.0])], token_weights=[0.4])
    assert solo.selected_indices == [0] and solo.effective_tokens == 1 and solo.byzantine_filter_active is False
    assert torch.equal(solo.aggregated_delta_bf16, torch.tensor([1.0, 2.0], dtype=torch.bfloat16))


def test_multi_krum_explicit_m_select_and_unweighted_tokens() -> None:
    deltas = [torch.tensor([0.0, 0.0]), torch.tensor([0.1, 0.0]), torch.tensor([0.0, 0.1]), torch.tensor([9.0, 9.0])]
    res = multi_krum_select_and_aggregate(deltas, f_byzantine=0, m_select=2)
    assert len(res.selected_indices) == 2 and 3 not in res.selected_indices
    assert res.effective_tokens == 2 and res.byzantine_filter_active is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"selected_indices": []},
        {"selected_indices": [True]},
        {"rejected_indices": (1,)},
        {"rejected_indices": [-1]},
        {"krum_scores": []},
        {"krum_scores": [float("nan")]},
        {"krum_scores": [-1.0]},
        {"aggregated_delta_bf16": torch.zeros(2)},
        {"effective_tokens": 0},
        {"byzantine_filter_active": 1},
    ],
)
def test_multi_krum_result_dataclass_refusals(kwargs: dict[str, Any]) -> None:
    base: dict[str, Any] = dict(selected_indices=[0], rejected_indices=[], krum_scores=[0.0],
                                aggregated_delta_bf16=torch.zeros(2, dtype=torch.bfloat16), effective_tokens=1,
                                byzantine_filter_active=False)
    base.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        MultiKrumAggregationResult(**base)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"accepted": 1},
        {"loss_before": "1.0"},
        {"loss_after": float("nan")},
        {"checkpoint_sha256": " "},
        {"reason": ""},
    ],
)
def test_loss_ratchet_decision_refusals(kwargs: dict[str, Any]) -> None:
    base: dict[str, Any] = dict(accepted=True, loss_before=1.0, loss_after=0.9, delta_loss=-0.1,
                                checkpoint_sha256=SHA, reason="ok")
    base.update(kwargs)
    with pytest.raises(ValueError):
        LossRatchetDecision(**base)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"accepted": "yes"},
        {"rms_old": "x"},
        {"rms_new": float("inf")},
        {"halving_residual_rms": -0.1},
        {"rms_ratio": 0.0},
        {"reason": " "},
    ],
)
def test_weight_rms_decision_refusals(kwargs: dict[str, Any]) -> None:
    base: dict[str, Any] = dict(accepted=True, rms_old=1.0, rms_new=1.0, rms_ratio=1.0, halving_residual_rms=0.5,
                                reason="ok")
    base.update(kwargs)
    with pytest.raises(ValueError):
        WeightRmsGuardDecision(**base)


def test_rejected_weight_rms_decision_may_carry_nan() -> None:
    d = WeightRmsGuardDecision(False, float("nan"), float("nan"), float("nan"), float("nan"), "REJECTED")
    assert d.accepted is False


# ---------------------------------------------------------------------------
# RMS collapse guard outcomes
# ---------------------------------------------------------------------------


def test_rms_guard_refusals_and_zero_participant_halving() -> None:
    old = torch.tensor([0.04, -0.02, 0.03, -0.05])
    for kwargs in ({"min_rms_ratio": 0.0}, {"max_rms_ratio": 0.5}, {"halving_residual_tol": -1.0},
                   {"min_rms_ratio": True}):
        with pytest.raises(ValueError, match="valid finite bounds"):
            check_weight_rms_collapse_guard(old, old, **kwargs)
    with pytest.raises(ValueError, match="identical element counts"):
        check_weight_rms_collapse_guard(old, old[:3])
    with pytest.raises(ValueError, match="identical element counts"):
        check_weight_rms_collapse_guard({}, {})
    with pytest.raises(ValueError, match="strictly positive"):
        check_weight_rms_collapse_guard(torch.zeros(4), old)

    halved = check_weight_rms_collapse_guard(old, 0.5 * old)
    assert halved.accepted is False and halved.reason.startswith("REJECTED_ZERO_PARTICIPANT_WEIGHT_HALVING")
    assert halved.rms_ratio == pytest.approx(0.5) and halved.halving_residual_rms == pytest.approx(0.0, abs=1e-12)
    dict_ok = check_weight_rms_collapse_guard({"w": old}, {"w": old * 1.1})
    assert dict_ok.accepted is True


# ---------------------------------------------------------------------------
# Precision cohorts, forming pools, resilience
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("vendor", "dtype", "expected_key"),
    [
        ("ROCm", "bf16", "amd_rocm:rocm:bf16:adamw_8bit"),
        ("hip", "auto", "amd_rocm:rocm:fp32:adamw_8bit"),
        ("Apple", "bf16", "apple_mps:mps:fp32:adamw_8bit"),
        ("intel-arc", "bf16", "intel-arc:generic:fp32:adamw_8bit"),
        ("cuda", "fp32", "nvidia:sm80_plus_ampere:fp32:adamw_8bit"),
    ],
)
def test_precision_cohorts_for_non_default_vendors(vendor: str, dtype: str, expected_key: str) -> None:
    cc = (8, 6) if vendor == "cuda" else (0, 0)
    cohort = classify_gpu_precision_cohort(vendor, cc, requested_dtype=dtype)
    assert cohort.cohort_key == expected_key and cohort.pre_ampere_fp32_enforced is False


def test_precision_cohort_refusals_and_empty_optimizer_default() -> None:
    with pytest.raises(ValueError, match="vendor"):
        classify_gpu_precision_cohort("  ")
    with pytest.raises(ValueError, match="compute_capability"):
        classify_gpu_precision_cohort("nvidia", (7,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="compute_capability"):
        classify_gpu_precision_cohort("nvidia", (7, -5))
    assert classify_gpu_precision_cohort("nvidia", (7, 5), optimizer_family=" ").optimizer_family == "adamw"
    for kwargs in ({"vendor": ""}, {"compute_capability": [7, 5]}, {"pre_ampere_fp32_enforced": 0}):
        base: dict[str, Any] = dict(vendor="nvidia", compute_capability=(7, 5), effective_dtype="fp32",
                                    optimizer_family="adamw", pre_ampere_fp32_enforced=True, cohort_key="k")
        base.update(kwargs)
        with pytest.raises(ValueError):
            GpuPrecisionCohort(**base)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": " "},
        {"cohort_key": ""},
        {"active_members": -1},
        {"target_pool_size": 0},
        {"requires_multi_krum": 1},
    ],
)
def test_pool_dispatch_decision_refusals(kwargs: dict[str, Any]) -> None:
    base: dict[str, Any] = dict(mode="FORMING_WAIT", cohort_key="k", active_members=1, target_pool_size=3,
                                requires_multi_krum=True, requires_loss_ratchet=True)
    base.update(kwargs)
    with pytest.raises(ValueError):
        PoolDispatchDecision(**base)


def test_forming_pool_modes_and_refusals() -> None:
    idle = resolve_forming_pool_starvation(current_members=0, waited_seconds=999.0)
    assert idle.mode == "IDLE_EMPTY_POOL" and idle.requires_multi_krum is False
    ready = resolve_forming_pool_starvation(current_members=4, target_pool_size=3, waited_seconds=0.0)
    assert ready.mode == "COHORT_QUORUM_READY" and ready.active_members == 4
    for kwargs in ({"current_members": 1.0}, {"target_pool_size": 0}, {"waited_seconds": -1.0},
                   {"forming_timeout_s": 0.0}):
        args: dict[str, Any] = dict(current_members=1, waited_seconds=1.0)
        args.update(kwargs)
        with pytest.raises(ValueError):
            resolve_forming_pool_starvation(**args)


def test_resilience_policy_continue_and_refusals() -> None:
    ok = evaluate_client_resilience_policy(batch_size=4, http_status=200)
    assert ok == {"action": "CONTINUE_NORMAL", "next_batch_size": 4, "redownload_checkpoint": False, "backoff_s": 0.0}
    capped = evaluate_client_resilience_policy(batch_size=1, http_status=429, retry_attempt=10, max_backoff_s=8.0)
    assert capped["backoff_s"] == 9.0  # 8 s cap + 12.5 % jitter
    for kwargs in ({"batch_size": 0}, {"retry_attempt": -1}, {"base_backoff_s": 0.0}, {"max_backoff_s": 1.0}):
        args: dict[str, Any] = dict(batch_size=1)
        args.update(kwargs)
        with pytest.raises(ValueError):
            evaluate_client_resilience_policy(**args)


# ---------------------------------------------------------------------------
# CrowdGPT headers, task parsing and submit metadata
# ---------------------------------------------------------------------------


def test_hardware_headers_refusals_and_sanitised_device_id() -> None:
    for kwargs, match in (
        ({"hiveminder_handle": "  "}, "hiveminder_handle"),
        ({"batch_size": 0}, "batch_size"),
        ({"vram_gb": 0.0}, "vram_gb"),
        ({"vram_gb": float("nan")}, "vram_gb"),
        ({"compute_capability": (7, "5")}, "compute_capability"),
    ):
        args: dict[str, Any] = dict(hiveminder_handle="alice")
        args.update(kwargs)
        with pytest.raises(ValueError, match=match):
            build_crowdgpt_client_hardware_headers(**args)
    h = build_crowdgpt_client_hardware_headers(hiveminder_handle="a b/c", device_suffix="", backend="rocm",
                                               can_fp32=False, can_8bit=False, engram_active=False)
    assert h["X-Client-Device"] == "hm-a-b-c-gpu0"
    assert h["X-Client-Backend"] == "ROCM"
    assert (h["X-Client-Optim-Can-Fp32"], h["X-Client-Optim-Can-8bit"], h["X-Client-Engram"]) == ("0", "0", "0")


def test_task_parsing_refusals_and_incomplete_crosscheck_falls_back() -> None:
    with pytest.raises(TypeError):
        parse_crowdgpt_task_crosscheck([("taskId", "x")])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="taskId"):
        parse_crowdgpt_task_crosscheck({"taskId": "  "})
    spec = parse_crowdgpt_task_crosscheck({
        "taskId": "t1",
        "crossCheck": {"enabled": True, "targetSeed": 1, "targetLocalSteps": 2, "targetBatchSize": 1,
                       "targetOptimizer": "adamw"},  # missing accum + autocast
        "shardConfig": {"slot": 5},
    })
    assert spec.cohort_mode is False and spec.target_seed is None and spec.target_optimizer == "any"
    assert spec.effective_slot == 5  # outside cohort mode the server-assigned slot applies
    odd = parse_crowdgpt_task_crosscheck({"taskId": "t2", "crossCheck": "yes", "shardConfig": [1]})
    assert odd.cohort_mode is False and odd.effective_slot == 0


def _spec(cohort: bool = False) -> CrowdGPTCrossCheckSpec:
    return CrowdGPTCrossCheckSpec(task_id="t", global_step=1, seq_len=8, cohort_mode=cohort,
                                  target_seed=3 if cohort else None, target_local_steps=4 if cohort else None,
                                  target_batch_size=2 if cohort else None, target_accum_steps=1 if cohort else None,
                                  target_autocast="fp32" if cohort else None, target_optimizer="any",
                                  effective_slot=0)


@pytest.mark.parametrize(
    ("kwargs", "exc", "match"),
    [
        ({"spec": {"task_id": "t"}}, TypeError, "CrowdGPTCrossCheckSpec"),
        ({"mean_loss": float("nan")}, ValueError, "mean_loss"),
        ({"last_step_loss": True}, ValueError, "last_step_loss"),
        ({"loss_std": -0.1}, ValueError, "non-negative"),
        ({"loss_samples": 0}, ValueError, "loss_samples"),
        ({"tokens_processed": 1.5}, ValueError, "tokens_processed"),
        ({"engram_sparse_count": -1}, ValueError, "engram_sparse_count"),
        ({"spec": "cohort", "actual_steps": 3}, ValueError, "targetLocalSteps=4"),
        ({"spec": "cohort", "actual_steps": 4, "actual_batch_size": 1}, ValueError, "targetBatchSize=2"),
    ],
)
def test_submit_metadata_refusals(kwargs: dict[str, Any], exc: type, match: str) -> None:
    args: dict[str, Any] = dict(spec=_spec(), mean_loss=1.0, last_step_loss=1.0, loss_std=0.0, loss_samples=1,
                                actual_steps=1, actual_batch_size=1, tokens_processed=8)
    args.update(kwargs)
    if args["spec"] == "cohort":
        args["spec"] = _spec(cohort=True)
    with pytest.raises(exc, match=match):
        build_crowdgpt_submit_metadata(**args)


def test_submit_metadata_engram_flags_follow_the_sparse_count() -> None:
    meta = build_crowdgpt_submit_metadata(spec=_spec(cohort=True), mean_loss=2.0, last_step_loss=1.5, loss_std=0.1,
                                          loss_samples=4, actual_steps=4, actual_batch_size=2, tokens_processed=64,
                                          has_engram=True, engram_sparse_count=0)
    assert meta["hasEngram"] is False and meta["engramSparseCount"] == 0
    assert meta["crossCheckUsed"] is True and meta["actualSeed"] == 3
