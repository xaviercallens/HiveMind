"""Unit tests for BOINC Signed Parquet Shards & Deterministic Spot-Check Validator."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from hivemind.boinc.validator import DeterministicSpotCheckValidator
from hivemind.boinc.workunit import (
    BOINCWorkunit,
    create_signed_parquet_shard,
    verify_and_load_parquet_shard,
)
from hivemind.core.diloco_inner import run_diloco_inner_loop
from hivemind.core.quantization import (
    attach_qlora_adapters,
    build_local_qwen2_backbone,
    extract_lora_state_dict,
    load_lora_state_dict,
)


def test_signed_parquet_shard_verification_and_tamper_detection(tmp_path: Path) -> None:
    shard_file = tmp_path / "wu_test.parquet"
    path, sha256, sig = create_signed_parquet_shard(
        shard_path=shard_file,
        branch_name="Code",
        rng_seed=1337,
        num_sequences=16,
        seq_len=32,
        vocab_size=1024,
    )
    tokens = verify_and_load_parquet_shard(
        shard_path=path,
        expected_sha256=sha256,
        expected_hmac_signature=sig,
        rng_seed=1337,
    )
    assert tokens.shape == (16, 32)

    # Tampering with rng_seed or signature must raise ValueError
    with pytest.raises(ValueError, match="signature verification failed"):
        verify_and_load_parquet_shard(
            shard_path=path,
            expected_sha256=sha256,
            expected_hmac_signature=sig,
            rng_seed=9999,
        )


def test_deterministic_spot_check_validator_accepts_honest_and_flags_cheater(tmp_path: Path) -> None:
    device = torch.device("cpu")
    base_model, base_sha, _ = build_local_qwen2_backbone(
        seed=42,
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        device=device,
        use_4bit_nf4_if_cuda=False,
    )
    bundle = attach_qlora_adapters(
        base_model=base_model,
        rank_r=16,
        lora_alpha=32,
        gradient_checkpointing=False,
        seed=42,
    )
    init_lora = extract_lora_state_dict(bundle.model)

    shard_file = tmp_path / "shard_spotcheck.parquet"
    path, sha256, sig = create_signed_parquet_shard(
        shard_path=shard_file,
        branch_name="STEM",
        rng_seed=2026,
        num_sequences=8,
        seq_len=32,
        vocab_size=512,
    )
    tokens = verify_and_load_parquet_shard(path, sha256, sig, rng_seed=2026)

    wu = BOINCWorkunit(
        wu_id="wu-stem-t1",
        branch_name="STEM",
        outer_step=1,
        base_model_id=base_sha,
        lora_checkpoint_url="local",
        data_shard_parquet=str(path),
        shard_sha256=sha256,
        shard_hmac_signature=sig,
        rng_seed=2026,
        inner_steps_H=10,
        spot_check_interval_s=5,
        seq_len=32,
        num_sequences=8,
    )

    load_lora_state_dict(bundle.model, init_lora)
    honest_res = run_diloco_inner_loop(
        model=bundle.model,
        token_sequences=tokens,
        worker_id="honest-worker-1",
        branch_name="STEM",
        outer_step=1,
        inner_steps_H=10,
        checkpoint_interval_s=5,
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
        steam_id64="76561198000000001",
        lr=1e-3,
        micro_batch_size=2,
    )
    assert verdict_ok.passed is True
    assert verdict_ok.steam_account_flagged is False
    assert verdict_ok.max_abs_divergence <= 1e-4

    # Corrupt micro-checkpoint at step 5 to simulate hardware bit-flip or cheating client
    cheater_res = copy.deepcopy(honest_res)
    first_param = next(iter(cheater_res.micro_checkpoints[5].state_snapshot))
    cheater_res.micro_checkpoints[5].state_snapshot[first_param] += 0.01

    verdict_bad = validator.verify_submission(
        validator_model=bundle.model,
        workunit=wu,
        submission=cheater_res,
        token_sequences=tokens,
        steam_id64="76561198000009999",
        lr=1e-3,
        micro_batch_size=2,
    )
    assert verdict_bad.passed is False
    assert verdict_bad.steam_account_flagged is True
    assert "76561198000009999" in validator.flagged_steam_accounts

    # IP-218 negative control: verify_submission refuses a BOINCWorkunit with empty shard_sha256 or shard_hmac_signature
    import pytest

    wu_unsigned = copy.deepcopy(wu)
    wu_unsigned.shard_sha256 = ""
    with pytest.raises(ValueError, match="shard_sha256"):
        validator.verify_submission(
            validator_model=bundle.model,
            workunit=wu_unsigned,
            submission=honest_res,
            token_sequences=tokens,
            steam_id64="76561198000000001",
            lr=1e-3,
            micro_batch_size=2,
        )
