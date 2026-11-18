# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression and negative-control tests for PR-IP-19 (IP-76, IP-77, IP-79, IP-82, IP-83, IP-126)."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import time

import pytest
import torch

from hivemind.boinc.coordinator_api import BOINCCoordinatorService
from hivemind.boinc.validator import DeterministicSpotCheckValidator
from hivemind.boinc.workunit import (
    BOINCWorkunit,
    create_signed_parquet_shard,
    verify_and_load_parquet_shard,
)
from hivemind.client.steam_identity import (
    OPENID_ENDPOINT,
    IdentityError,
    SteamOpenIDVerifier,
)
from hivemind.config import HiveMindConfig
from hivemind.core.compression import compress_lora_delta, decompress_lora_delta
from hivemind.core.dev_image_video_moe_integration_benchmark import (
    DEV_IMAGE_VIDEO_BENCHMARK_SCENARIOS,
    run_intense_dev_image_video_moe_benchmark,
)
from hivemind.core.diloco_inner import MicroCheckpointRecord, run_diloco_inner_loop
from hivemind.core.quantization import (
    attach_qlora_adapters,
    build_local_qwen2_backbone,
    compute_state_dict_sha256,
    extract_lora_state_dict,
    load_lora_state_dict,
)
from hivemind.queenbee.dance import (
    DanceBoard,
    DanceRecord,
    derive_signer_secret,
    recruitment_weights,
)
from hivemind.queenbee.strategies import BanditStrategy, Outcome


def _build_tiny_qwen_bundle():
    device = torch.device("cpu")
    base_model, base_sha, _ = build_local_qwen2_backbone(
        seed=42,
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        device=device,
        use_4bit_nf4_if_cuda=False,
    )
    bundle = attach_qlora_adapters(
        base_model=base_model,
        rank_r=8,
        lora_alpha=16,
        gradient_checkpointing=False,
        seed=42,
    )
    init_lora = extract_lora_state_dict(bundle.model)
    return bundle, init_lora, base_sha


def test_ip76_validator_detects_overwritten_final_checkpoint_and_swapped_step0(tmp_path: Path) -> None:
    """IP-76: Overwriting checkpoint 100 + theta_final (or swapping micro_checkpoints[0]) must fail verification."""
    bundle, init_lora, base_sha = _build_tiny_qwen_bundle()
    shard_file = tmp_path / "shard_ip76.parquet"
    path, sha256, sig = create_signed_parquet_shard(
        shard_path=shard_file,
        branch_name="STEM",
        rng_seed=2026,
        num_sequences=8,
        seq_len=16,
        vocab_size=256,
    )
    tokens = verify_and_load_parquet_shard(path, sha256, sig, rng_seed=2026)

    wu = BOINCWorkunit(
        wu_id="wu-stem-ip76",
        branch_name="STEM",
        outer_step=1,
        base_model_id=base_sha,
        lora_checkpoint_url="local",
        data_shard_parquet=str(path),
        shard_sha256=sha256,
        shard_hmac_signature=sig,
        rng_seed=2026,
        inner_steps_H=100,
        spot_check_interval_s=50,
        seq_len=16,
        num_sequences=8,
    )

    load_lora_state_dict(bundle.model, init_lora)
    honest_res = run_diloco_inner_loop(
        model=bundle.model,
        token_sequences=tokens,
        worker_id="worker-ip76",
        branch_name="STEM",
        outer_step=1,
        inner_steps_H=100,
        checkpoint_interval_s=50,
        lr=1e-3,
        rng_seed=wu.rng_seed,
        micro_batch_size=2,
    )

    validator = DeterministicSpotCheckValidator(tolerance_eps=1e-4)
    verdict_ok = validator.verify_submission(
        validator_model=bundle.model,
        workunit=wu,
        submission=honest_res,
        token_sequences=tokens,
        steam_id64="76561198000000101",
        lr=1e-3,
        micro_batch_size=2,
        validator_rng_seed=0,
    )
    assert verdict_ok.passed is True

    # Case 1: Overwrite checkpoint 100 and theta_final with random but self-consistent tensors
    tampered_final = copy.deepcopy(honest_res)
    fake_final: dict[str, torch.Tensor] = {}
    fake_raw_delta: dict[str, torch.Tensor] = {}
    for k, init_t in tampered_final.theta_init.items():
        noise = torch.full_like(init_t, 0.05)
        fake_final[k] = (init_t - noise).clone()
        fake_raw_delta[k] = noise.clone()
    tampered_final.theta_final = fake_final
    tampered_final.raw_delta = fake_raw_delta
    tampered_final.micro_checkpoints[100] = MicroCheckpointRecord(
        step=100,
        lora_sha256=compute_state_dict_sha256(fake_final),
        loss_value=tampered_final.final_loss,
        state_snapshot=fake_final,
    )
    verdict_bad_final = validator.verify_submission(
        validator_model=bundle.model,
        workunit=wu,
        submission=tampered_final,
        token_sequences=tokens,
        steam_id64="76561198000000102",
        lr=1e-3,
        micro_batch_size=2,
        validator_rng_seed=0,
    )
    assert verdict_bad_final.passed is False
    assert verdict_bad_final.steam_account_flagged is True

    # Case 2: Swap micro_checkpoints[0] for a tampered initial state
    tampered_step0 = copy.deepcopy(honest_res)
    fake_step0 = {k: (v + 0.02).clone() for k, v in tampered_step0.theta_init.items()}
    tampered_step0.micro_checkpoints[0] = MicroCheckpointRecord(
        step=0,
        lora_sha256=compute_state_dict_sha256(fake_step0),
        loss_value=0.0,
        state_snapshot=fake_step0,
    )
    verdict_bad_step0 = validator.verify_submission(
        validator_model=bundle.model,
        workunit=wu,
        submission=tampered_step0,
        token_sequences=tokens,
        steam_id64="76561198000000103",
        lr=1e-3,
        micro_batch_size=2,
    )
    assert verdict_bad_step0.passed is False
    assert verdict_bad_step0.steam_account_flagged is True


def test_ip77_coordinator_rejects_inflated_compressed_delta(tmp_path: Path) -> None:
    """IP-77: Submitting valid checkpoints/raw_delta with compressed_delta = compress(100 * raw_delta) must be rejected."""
    bundle, init_lora, base_sha = _build_tiny_qwen_bundle()
    coordinator = BOINCCoordinatorService(
        config=HiveMindConfig.from_yaml(),
        initial_lora_state=init_lora,
        base_model_sha256=base_sha,
        shards_dir=tmp_path / "coord_shards_ip77",
        vocab_size=256,
    )
    steam_id = "76561198000000201"
    wu, state = coordinator.assign_workunit(
        worker_id="worker-ip77",
        steam_id64=steam_id,
        branch_name="Code",
        inner_steps_H=4,
        spot_check_interval_s=2,
        num_sequences=4,
        seq_len=16,
        rng_seed=7701,
    )
    load_lora_state_dict(bundle.model, state)
    honest_res = run_diloco_inner_loop(
        model=bundle.model,
        token_sequences=verify_and_load_parquet_shard(
            wu.data_shard_parquet, wu.shard_sha256, wu.shard_hmac_signature, rng_seed=wu.rng_seed
        ),
        worker_id="worker-ip77",
        branch_name="Code",
        outer_step=wu.outer_step,
        inner_steps_H=4,
        checkpoint_interval_s=2,
        lr=2e-4,
        rng_seed=wu.rng_seed,
        micro_batch_size=1,
    )

    inflated_delta = {k: (v * 100.0).clone() for k, v in honest_res.raw_delta.items()}
    poisoned_submission = copy.deepcopy(honest_res)
    poisoned_submission.compressed_delta = compress_lora_delta(
        delta_dict=inflated_delta,
        worker_id=honest_res.worker_id,
        branch_name=honest_res.branch_name,
        outer_step=honest_res.outer_step,
        inner_steps_completed=honest_res.steps_completed,
        topk_ratio=honest_res.compressed_delta.topk_ratio,
    )

    verdict = coordinator.validate_and_record_submission(
        validator_model=bundle.model,
        workunit=wu,
        submission=poisoned_submission,
        steam_id64=steam_id,
    )
    assert verdict.passed is False
    assert verdict.steam_account_flagged is True
    assert "worker-ip77" not in coordinator.pending_branch_deltas["Code"]
    assert coordinator.reputation_ledger[steam_id].flagged_byzantine is True


def test_ip79_dev_image_video_benchmark_marks_simulated_and_derives_p95() -> None:
    """IP-79: Benchmark marks synthetic metrics simulated/declared_constant and derives p95 from scenario latency table."""
    base_report = run_intense_dev_image_video_moe_benchmark()
    assert base_report["simulated"] is True
    assert base_report["declared_constant"] is True
    assert base_report["real_requests_executed"] == 0
    assert base_report["status"] != "COMPLETED_ALL_GATES_PASSED"

    slower_scenarios = [
        replace(s, moe_latency_ms=s.moe_latency_ms * 2.5)
        for s in DEV_IMAGE_VIDEO_BENCHMARK_SCENARIOS
    ]
    slower_report = run_intense_dev_image_video_moe_benchmark(scenarios=slower_scenarios)
    assert slower_report["overall_summary"]["p95_latency_ms"] > base_report["overall_summary"]["p95_latency_ms"] * 2.0
    assert (
        slower_report["overall_summary"]["developer_coding_3tome_gain"]["stress_test"]["p95_latency_ms"]
        > base_report["overall_summary"]["developer_coding_3tome_gain"]["stress_test"]["p95_latency_ms"] * 2.0
    )


def test_ip82_dance_record_replay_cap_and_per_signer_hmac_isolation(tmp_path: Path) -> None:
    """IP-82: Replaying [rec]*1000 gives weight <= cap_per_record, and signer keys cannot forge for a third identity."""
    master_secret = b"ip82-master-secret-32-bytes-long!"
    rec = DanceRecord("e1", "e1", "c1", 100, 95, 0.4, 10, "unit_tests", "scout-a").signed(master_secret)
    rep = recruitment_weights(
        [rec] * 1000,
        now_epoch=10,
        horizon_epochs=5,
        cap_per_record=0.35,
        secret=master_secret,
    )
    assert rep.weights["c1"]["e1"] <= 0.35 + 1e-9

    board = DanceBoard(tmp_path / "dance_ip82.jsonl")
    for _ in range(5):
        board.append(rec)
    loaded, bad = board.load()
    assert len(loaded) == 1 and bad == 0

    # Two distinct signers hold only their derived per-signer keys; neither can forge for a third expert/signer
    k_alice = derive_signer_secret(master_secret, "alice")
    k_bob = derive_signer_secret(master_secret, "bob")
    alice_valid = DanceRecord("alice", "alice", "c1", 50, 45, 0.3, 10, "unit_tests", "alice").signed(
        k_alice, prederived=True
    )
    assert alice_valid.verify(master_secret) is True
    assert alice_valid.verify({"alice": k_alice, "bob": k_bob}) is True

    forged_for_charlie_by_alice = DanceRecord(
        "e3", "e3", "c1", 50, 50, 0.3, 10, "unit_tests", "charlie"
    ).signed(k_alice, prederived=True)
    forged_for_charlie_by_bob = DanceRecord(
        "e3", "e3", "c1", 50, 50, 0.3, 10, "unit_tests", "charlie"
    ).signed(k_bob, prederived=True)
    assert forged_for_charlie_by_alice.verify(master_secret) is False
    assert forged_for_charlie_by_bob.verify(master_secret) is False
    rep_forged = recruitment_weights(
        [forged_for_charlie_by_alice, forged_for_charlie_by_bob],
        now_epoch=10,
        horizon_epochs=5,
        cap_per_record=0.5,
        secret=master_secret,
    )
    assert rep_forged.accepted == 0
    assert rep_forged.rejected["bad_signature"] == 2


def test_ip83_bandit_dance_records_report_only_observed_verified_successes() -> None:
    """IP-83: warm_start with w=1.0 followed by 3 verified failures must export dance_records()[0].successes == 0."""
    secret = b"ip83-dance-secret-32-bytes-long!"
    rec = DanceRecord("e1", "e1", "c1", 1000, 1000, 0.4, 10, "unit_tests", "scout-a").signed(secret)
    rep = recruitment_weights([rec], now_epoch=10, horizon_epochs=5, cap_per_record=1.0, secret=secret)
    bandit = BanditStrategy()
    bandit.warm_start(rep, pseudo_trials=20.0)

    for _ in range(3):
        bandit.observe(Outcome("c1", "e1", verified=True, success=False))

    exported = bandit.dance_records(
        served_by={"e1": "e1"},
        cost_per_1k={"e1": 0.4},
        epoch=11,
        verifier_kind="unit_tests",
        signer_id="client-1",
        secret=secret,
    )
    assert len(exported) == 1
    assert exported[0].n == 3
    assert exported[0].successes == 0


def test_ip126_steam_nonce_cache_resists_unverified_flood_and_enforces_per_ip_limit() -> None:
    """IP-126: 5000 invalid assertions with distinct fresh nonces must not fill the nonce cache or block a valid login."""
    return_to = "https://coord.example/identity/steam/callback"
    sid = "76561197960287930"
    now = 1_800_000_000.0
    state = {"valid": False}

    def stub_http(method: str, url: str, form=None):  # noqa: ANN001, ANN202
        return 200, f"ns:http://specs.openid.net/auth/2.0\nis_valid:{'true' if state['valid'] else 'false'}\n"

    verifier = SteamOpenIDVerifier(
        return_to,
        http=stub_http,
        clock=lambda: now,
        max_nonces=4096,
        max_attempts_per_ip=10,
    )

    def make_assertion(idx: int) -> dict[str, str]:
        nonce_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)) + f"n{idx:05d}"
        return {
            "openid.ns": "http://specs.openid.net/auth/2.0",
            "openid.mode": "id_res",
            "openid.op_endpoint": OPENID_ENDPOINT,
            "openid.claimed_id": f"https://steamcommunity.com/openid/id/{sid}",
            "openid.identity": f"https://steamcommunity.com/openid/id/{sid}",
            "openid.return_to": return_to,
            "openid.response_nonce": nonce_str,
            "openid.assoc_handle": "1234567890",
            "openid.signed": "signed,op_endpoint,claimed_id,identity,return_to,response_nonce,assoc_handle",
            "openid.sig": "c2ln",
        }

    for i in range(5000):
        with pytest.raises(IdentityError, match="Steam did not confirm"):
            verifier.verify_assertion(make_assertion(i))

    # A valid assertion must still succeed because invalid assertions never committed their nonces
    state["valid"] = True
    assert verifier.verify_assertion(make_assertion(99999)) == sid

    # Per-IP limit blocks abusive IP after max_attempts_per_ip without affecting another IP
    state["valid"] = False
    for i in range(10):
        with pytest.raises(IdentityError, match="Steam did not confirm"):
            verifier.verify_assertion(make_assertion(10000 + i), client_ip="203.0.113.9")
    with pytest.raises(IdentityError, match="too many sign-in attempts from this IP"):
        verifier.verify_assertion(make_assertion(10020), client_ip="203.0.113.9")

    state["valid"] = True
    assert verifier.verify_assertion(make_assertion(10021), client_ip="198.51.100.4") == sid
