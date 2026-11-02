"""HM-S6-01 error-feedback buffer wired into the worker path and the BOINC spot-check."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hivemind.boinc.coordinator_api import BOINCCoordinatorService
from hivemind.boinc.validator import DeterministicSpotCheckValidator
from hivemind.boinc.workunit import (
    BOINCWorkunit,
    create_signed_parquet_shard,
    verify_and_load_parquet_shard,
)
from hivemind.client.worker_node import HiveMindWorkerNode
from hivemind.config import HiveMindConfig
from hivemind.core.compression import compress_lora_delta, decompress_lora_delta
from hivemind.core.diloco_inner import run_diloco_inner_loop
from hivemind.core.quantization import (
    attach_qlora_adapters,
    build_local_qwen2_backbone,
    extract_lora_state_dict,
)
from hivemind.swarm.strategy_optimizer import EF21DiLoCoErrorFeedbackBuffer

TOPK = 0.1


@pytest.fixture(scope="module")
def tiny_qwen_bundle():
    base_model, base_sha, _ = build_local_qwen2_backbone(
        seed=42,
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        device=torch.device("cpu"),
        use_4bit_nf4_if_cuda=False,
    )
    bundle = attach_qlora_adapters(
        base_model=base_model,
        rank_r=2,
        lora_alpha=4,
        gradient_checkpointing=False,
        seed=42,
    )
    return bundle, extract_lora_state_dict(bundle.model), base_sha


def _workunit(tmp_path: Path, base_sha: str, branch: str, outer_step: int, seed: int) -> BOINCWorkunit:
    path, sha256, sig = create_signed_parquet_shard(
        shard_path=tmp_path / f"{branch}_t{outer_step}_{seed}.parquet",
        branch_name=branch,
        rng_seed=seed,
        num_sequences=4,
        seq_len=8,
        vocab_size=128,
    )
    return BOINCWorkunit(
        wu_id=f"wu-{branch}-t{outer_step}-{seed}",
        branch_name=branch,
        outer_step=outer_step,
        base_model_id=base_sha,
        lora_checkpoint_url="local",
        data_shard_parquet=str(path),
        shard_sha256=sha256,
        shard_hmac_signature=sig,
        rng_seed=seed,
        inner_steps_H=2,
        spot_check_interval_s=1,
        seq_len=8,
        num_sequences=4,
    )


_steam_ids = iter(range(76561198000005600, 76561198000005700))


def _worker(bundle, worker_id: str, **kwargs) -> HiveMindWorkerNode:
    return HiveMindWorkerNode(
        worker_id=worker_id,
        steam_id64=str(next(_steam_ids)),
        persona_name=worker_id,
        model=bundle.model,
        **kwargs,
    )


def _clone(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in state.items()}


def _packages_equal(a, b) -> bool:
    da, db = decompress_lora_delta(a), decompress_lora_delta(b)
    return all(torch.equal(da[k], db[k]) for k in da)


def test_error_feedback_is_off_by_default(tiny_qwen_bundle, tmp_path: Path) -> None:
    bundle, init_lora, base_sha = tiny_qwen_bundle
    worker = _worker(bundle, "w-default")
    wu = _workunit(tmp_path, base_sha, "Code", outer_step=1, seed=11)

    res = worker.execute_workunit(wu, _clone(init_lora), topk_ratio=TOPK)

    assert worker.error_feedback_buffers == {}
    assert res.error_feedback_residual_norm is None
    plain = compress_lora_delta(
        res.raw_delta, "w-default", "Code", outer_step=1, inner_steps_completed=2, topk_ratio=TOPK
    )
    assert _packages_equal(res.compressed_delta, plain)


def test_worker_carries_residual_across_workunits_and_validator_accepts_every_round(
    tiny_qwen_bundle, tmp_path: Path
) -> None:
    bundle, init_lora, base_sha = tiny_qwen_bundle
    worker = _worker(bundle, "w-ef", enable_error_feedback=True)
    validator = DeterministicSpotCheckValidator(tolerance_eps=1e-4)
    global_state = _clone(init_lora)

    sent_total = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in init_lora.items()}
    raw_total = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in init_lora.items()}
    residual_before: dict[str, torch.Tensor] = {}

    for outer_step in (1, 2, 3):
        wu = _workunit(tmp_path, base_sha, "Code", outer_step=outer_step, seed=100 + outer_step)
        res = worker.execute_workunit(wu, _clone(global_state), topk_ratio=TOPK)
        buffer = worker.error_feedback_buffers["Code"]

        tokens = verify_and_load_parquet_shard(
            wu.data_shard_parquet, wu.shard_sha256, wu.shard_hmac_signature, rng_seed=wu.rng_seed
        )
        verdict = validator.verify_submission(
            validator_model=bundle.model,
            workunit=wu,
            submission=res,
            token_sequences=tokens,
            steam_id64=worker.steam_id64,
        )
        assert verdict.passed, verdict.reason
        assert verdict.hash_matched

        # The upload is C(raw + carried residual), not C(raw); from round 2 on they differ.
        corrected = {k: res.raw_delta[k] + residual_before.get(k, 0.0) for k in res.raw_delta}
        expected = compress_lora_delta(
            corrected, "w-ef", "Code", outer_step=outer_step, inner_steps_completed=2, topk_ratio=TOPK
        )
        assert _packages_equal(res.compressed_delta, expected)
        if residual_before:
            plain = compress_lora_delta(
                res.raw_delta, "w-ef", "Code", outer_step=outer_step, inner_steps_completed=2, topk_ratio=TOPK
            )
            assert not _packages_equal(res.compressed_delta, plain)

        sent = decompress_lora_delta(res.compressed_delta)
        for k in sent_total:
            sent_total[k] += sent[k]
            raw_total[k] += res.raw_delta[k]
            global_state[k] = global_state[k] - sent[k].to(global_state[k].dtype)

        assert res.error_feedback_residual_norm == pytest.approx(buffer.residual_norm())
        assert res.error_feedback_residual_norm > 0.0
        residual_before = {k: v.clone() for k, v in buffer.residuals.items()}

    assert worker.error_feedback_buffers["Code"].last_outer_step == 3
    for k in sent_total:
        assert torch.allclose(sent_total[k] + residual_before[k], raw_total[k], atol=1e-6)


def test_validator_still_rejects_tampered_submission_with_error_feedback(
    tiny_qwen_bundle, tmp_path: Path
) -> None:
    bundle, init_lora, base_sha = tiny_qwen_bundle
    worker = _worker(bundle, "w-ef-byz", enable_error_feedback=True)
    wu = _workunit(tmp_path, base_sha, "Code", outer_step=1, seed=303)

    res = worker.execute_workunit(wu, _clone(init_lora), topk_ratio=TOPK, byzantine_poison_scale=10.0)
    tokens = verify_and_load_parquet_shard(
        wu.data_shard_parquet, wu.shard_sha256, wu.shard_hmac_signature, rng_seed=wu.rng_seed
    )
    verdict = DeterministicSpotCheckValidator(tolerance_eps=1e-4).verify_submission(
        validator_model=bundle.model,
        workunit=wu,
        submission=res,
        token_sequences=tokens,
        steam_id64=worker.steam_id64,
    )
    assert verdict.passed is False


def test_coordinator_multi_round_with_error_feedback_workers(tiny_qwen_bundle, tmp_path: Path) -> None:
    bundle, init_lora, base_sha = tiny_qwen_bundle
    coordinator = BOINCCoordinatorService(
        config=HiveMindConfig.from_yaml(),
        initial_lora_state=init_lora,
        base_model_sha256=base_sha,
        shards_dir=tmp_path / "coord_shards",
        vocab_size=128,
    )
    workers = [_worker(bundle, f"w-ef-{i}", enable_error_feedback=True) for i in range(4)]

    for round_idx in (1, 2, 3):
        for i, worker in enumerate(workers):
            wu, state = coordinator.assign_workunit(
                worker_id=worker.worker_id,
                steam_id64=worker.steam_id64,
                branch_name="Code",
                inner_steps_H=2,
                spot_check_interval_s=1,
                num_sequences=4,
                seq_len=8,
                rng_seed=7000 + 100 * round_idx + i,
            )
            assert wu.outer_step == round_idx
            res = worker.execute_workunit(wu, state, topk_ratio=TOPK)
            verdict = coordinator.validate_and_record_submission(
                validator_model=bundle.model,
                workunit=wu,
                submission=res,
                steam_id64=worker.steam_id64,
            )
            assert verdict.passed, verdict.reason
            queued = coordinator.pending_branch_deltas["Code"][worker.worker_id]
            sent = decompress_lora_delta(res.compressed_delta)
            assert all(torch.equal(queued[k], sent[k]) for k in sent)
        assert coordinator.step_branch_outer_loop("Code").outer_step == round_idx

    for worker in workers:
        buffer = worker.error_feedback_buffers["Code"]
        assert buffer.last_outer_step == 3
        assert buffer.residual_norm() > 0.0


def test_worker_keeps_one_buffer_per_branch(tiny_qwen_bundle, tmp_path: Path) -> None:
    bundle, init_lora, base_sha = tiny_qwen_bundle
    worker = _worker(
        bundle,
        "w-ef-branches",
        enable_error_feedback=True,
        error_feedback_stale_decay=0.5,
        error_feedback_max_residual_norm=10.0,
    )
    worker.execute_workunit(_workunit(tmp_path, base_sha, "Code", 1, 41), _clone(init_lora), topk_ratio=TOPK)
    worker.execute_workunit(_workunit(tmp_path, base_sha, "STEM", 1, 42), _clone(init_lora), topk_ratio=TOPK)
    worker.execute_cas_shard(
        "Legal",
        torch.randint(0, 128, (4, 8), generator=torch.Generator().manual_seed(43)),
        _clone(init_lora),
        inner_steps_H=2,
        spot_check_interval_s=1,
        topk_ratio=TOPK,
    )

    assert set(worker.error_feedback_buffers) == {"Code", "STEM", "Legal"}
    code, stem = worker.error_feedback_buffers["Code"], worker.error_feedback_buffers["STEM"]
    assert code is not stem
    assert code.stale_decay == 0.5 and code.max_residual_norm == 10.0
    assert not all(torch.equal(code.residuals[k], stem.residuals[k]) for k in code.residuals)


def test_inner_loop_rejects_topk_ratio_that_disagrees_with_buffer(tiny_qwen_bundle) -> None:
    bundle, init_lora, _ = tiny_qwen_bundle
    with pytest.raises(ValueError, match="topk_ratio"):
        run_diloco_inner_loop(
            model=bundle.model,
            token_sequences=torch.zeros(4, 8, dtype=torch.long),
            worker_id="w-mismatch",
            branch_name="Code",
            inner_steps_H=1,
            checkpoint_interval_s=1,
            topk_ratio=0.2,
            error_feedback=EF21DiLoCoErrorFeedbackBuffer(topk_ratio=0.05),
        )


def test_worker_records_measured_ef21_telemetry_per_branch(tiny_qwen_bundle, tmp_path: Path) -> None:
    from hivemind.swarm.strategy_optimizer import StrategyOptimizerTelemetryEngine

    bundle, init_lora, base_sha = tiny_qwen_bundle
    plain = _worker(bundle, "w-ef-off")
    plain.execute_workunit(_workunit(tmp_path, base_sha, "Code", 1, 51), _clone(init_lora), topk_ratio=TOPK)
    assert plain.ef21_telemetry() == []

    worker = _worker(bundle, "w-ef-tele", enable_error_feedback=True)
    res = worker.execute_workunit(
        _workunit(tmp_path, base_sha, "Code", 2, 52), _clone(init_lora), topk_ratio=TOPK
    )
    honest_norm = sum(float(t.pow(2).sum()) for t in res.raw_delta.values()) ** 0.5
    poisoned = worker.execute_workunit(
        _workunit(tmp_path, base_sha, "STEM", 2, 53),
        _clone(init_lora),
        topk_ratio=TOPK,
        byzantine_poison_scale=10.0,
    )
    poisoned_norm = sum(float(t.pow(2).sum()) for t in poisoned.raw_delta.values()) ** 0.5

    by_branch = {e["branch_name"]: e for e in worker.ef21_telemetry()}
    code = by_branch["Code"]
    assert code["worker_id"] == "w-ef-tele" and code["outer_step"] == 2
    assert code["residual_norm"] == pytest.approx(res.error_feedback_residual_norm, abs=1e-6)
    assert code["raw_delta_norm"] == pytest.approx(honest_norm, rel=1e-5)
    assert code["contraction_factor"] == pytest.approx(code["residual_norm"] / code["raw_delta_norm"], rel=1e-5)
    assert code["topk_ratio"] == TOPK and code["clip_events"] == 0
    # Telemetry is taken before the Byzantine test hook rewrites raw_delta.
    assert by_branch["STEM"]["raw_delta_norm"] < poisoned_norm

    engine = StrategyOptimizerTelemetryEngine(seed=2026)
    served = engine.ingest_worker_ef21(worker)["ef21_live_telemetry"]["workers"]
    assert {(w["worker_id"], w["branch_name"]) for w in served} == {("w-ef-tele", "Code"), ("w-ef-tele", "STEM")}


def test_validator_rejects_nan_in_worker_checkpoint_and_in_raw_delta(tiny_qwen_bundle, tmp_path: Path) -> None:
    """NaN compares False against any tolerance, so an unguarded max-diff check would pass a poisoned state."""
    bundle, init_lora, base_sha = tiny_qwen_bundle
    worker = _worker(bundle, "w-nan")
    wu = _workunit(tmp_path, base_sha, "Code", outer_step=1, seed=404)
    res = worker.execute_workunit(wu, _clone(init_lora), topk_ratio=TOPK)
    tokens = verify_and_load_parquet_shard(
        wu.data_shard_parquet, wu.shard_sha256, wu.shard_hmac_signature, rng_seed=wu.rng_seed
    )
    validator = DeterministicSpotCheckValidator(tolerance_eps=1e-4)
    honest = validator.verify_submission(
        validator_model=bundle.model, workunit=wu, submission=res, token_sequences=tokens, steam_id64=worker.steam_id64
    )
    assert honest.passed, honest.reason  # positive control: the untouched submission verifies

    checked_step = sorted(s for s in res.micro_checkpoints if s > 0)[0]
    name = next(iter(res.micro_checkpoints[checked_step].state_snapshot))
    res.micro_checkpoints[checked_step].state_snapshot[name].view(-1)[0] = float("nan")
    poisoned_state = validator.verify_submission(
        validator_model=bundle.model, workunit=wu, submission=res, token_sequences=tokens, steam_id64="7656119900000001"
    )
    assert poisoned_state.passed is False
    assert poisoned_state.max_abs_divergence == float("inf")

    res2 = worker.execute_workunit(wu, _clone(init_lora), topk_ratio=TOPK)
    dname = next(iter(res2.raw_delta))
    res2.raw_delta[dname].view(-1)[0] = float("nan")
    poisoned_delta = validator.verify_submission(
        validator_model=bundle.model, workunit=wu, submission=res2, token_sequences=tokens, steam_id64="7656119900000002"
    )
    assert poisoned_delta.passed is False
