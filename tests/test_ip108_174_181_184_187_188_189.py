# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression & negative-control tests for IP-108, IP-174, IP-181, IP-184, IP-187, IP-188, and IP-189."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from hivemind.boinc.real_client.server_stack import Stack
from hivemind.boinc.validator import DeterministicSpotCheckValidator
from hivemind.boinc.workunit import BOINCWorkunit, create_signed_parquet_shard
from hivemind.client.worker_node import HiveMindWorkerNode
from hivemind.config import HiveMindConfig
from hivemind.core.diloco_inner import MicroCheckpointRecord
from hivemind.core.diloco_outer import DiLoCoOuterOptimizer, filter_byzantine_deltas_multikrum
from hivemind.core.quantization import (
    attach_qlora_adapters,
    build_local_qwen2_backbone,
    compute_state_dict_sha256,
    extract_lora_state_dict,
)
from hivemind.queenbee.knowledge_market import KnowledgeShare, admit_share
from hivemind.queenbee.strategies import PreferencePair, PreferencePairStrategy
from hivemind.swarm.dht_storage import ContentAddressedSwarmStore
from hivemind.swarm.ledger import HiveSwarmLedger


def test_ip108_and_ip181_verify_proof_of_storage_rejects_corrupted_or_mismatched_shard(
    tmp_path: Path,
) -> None:
    """IP-108 & IP-181 negative control:
    - Pin cid C for node A; calling verify_proof_of_storage('A', C, <unrelated file>) returns verified=False.
    - Corrupting a stored shard before mining results in verified=False and 0 storage reward in the PoUW block.
    """
    store = ContentAddressedSwarmStore(tmp_path / "cas_store", node_id="node-a")
    cid, shard_path, _ = store.store_hf_dataset_shard(
        node_id="node-a",
        domain_name="Astro_Transit",
        shard_id=1,
        samples=["Kepler-442b transit observation alpha", "TESS sector 14 lightcurve beta"],
    )
    prev_hash = "00" * 32

    # Positive control: intact shard verifies
    ok_receipt = store.verify_proof_of_storage("node-a", cid, shard_path, prev_hash)
    assert ok_receipt.verified is True
    assert ok_receipt.bytes_stored > 0

    # IP-108 negative control: unrelated file for pinned cid C must return verified=False
    unrelated = tmp_path / "unrelated.bin"
    unrelated.write_bytes(b"X" * 65536)
    bad_unrelated = store.verify_proof_of_storage("node-a", cid, unrelated, prev_hash)
    assert bad_unrelated.verified is False
    assert bad_unrelated.bytes_stored == 0

    # IP-181 negative control: corrupt the stored shard on disk before mining -> verified=False & 0 seeder reward
    shard_path.write_bytes(b"CORRUPTED_SHARD_PAYLOAD" * 512)
    corrupted_receipt = store.verify_proof_of_storage("node-a", cid, shard_path, prev_hash)
    assert corrupted_receipt.verified is False

    ledger = HiveSwarmLedger(ledger_path=tmp_path / "ledger.json")
    block = ledger.mine_pouw_block(
        outer_step=1,
        expert_cid="cid://sha256/" + "a" * 32,
        work_receipts=[],
        storage_receipts=[corrupted_receipt],
        validator_node_ids=["val-01"],
    )
    assert block.reward_allocations.get("node-a", 0.0) == 0.0


def test_ip174_compute_state_dict_sha256_hashes_all_base_weights() -> None:
    """IP-174 negative control: perturbing one base weight at an index not divisible by 256 must change the digest."""
    base_weight = torch.zeros(5000, dtype=torch.float32)
    state_dict = {"model.layers.0.mlp.down_proj.weight": base_weight.clone()}
    digest_before = compute_state_dict_sha256(state_dict)

    # Index 137 is not divisible by 256 (previously skipped by [::256] striding when numel > 4096)
    perturbed = base_weight.clone()
    perturbed[137] = 1e-3
    digest_after = compute_state_dict_sha256({"model.layers.0.mlp.down_proj.weight": perturbed})
    assert digest_before != digest_after


def test_ip184_two_deltas_with_opposing_byzantine_worker_rejects_second() -> None:
    """IP-184 negative control: with 2 worker deltas where second = -50x the first, the second is rejected."""
    honest = {"lora_A": torch.tensor([0.1, -0.2, 0.3, 0.4], dtype=torch.float32)}
    byzantine = {"lora_A": -50.0 * honest["lora_A"]}

    report = filter_byzantine_deltas_multikrum(
        {"honest-01": honest, "byz-02": byzantine},
        max_byzantine_f=1,
    )
    assert report.accepted_workers == ["honest-01"]
    assert "byz-02" in report.rejected_workers

    opt = DiLoCoOuterOptimizer(outer_lr_eta=0.5, outer_momentum_mu=0.0, max_byzantine_f=1)
    theta_init = {"lora_A": torch.ones(4, dtype=torch.float32)}
    res = opt.step("Code", 1, theta_init, {"honest-01": honest, "byz-02": byzantine})
    assert res.filter_report.accepted_workers == ["honest-01"]
    assert "byz-02" in res.filter_report.rejected_workers
    # Updated parameters move in the honest direction (theta_init - 0.5 * honest), not toward -50x
    assert torch.allclose(res.theta_updated["lora_A"], theta_init["lora_A"] - 0.5 * honest["lora_A"], atol=1e-5)


def test_ip187_two_fresh_boinc_stacks_generate_distinct_passwd_hashes(tmp_path: Path) -> None:
    """IP-187 negative control: two fresh stacks must not yield the same passwd_hash for the same email."""
    stack1 = Stack(tmp_path / "s1", tmp_path / "boinc_src", "python3", tmp_path / "repo")
    stack2 = Stack(tmp_path / "s2", tmp_path / "boinc_src", "python3", tmp_path / "repo")
    assert stack1.account_passwd_hash("node@local.invalid") != stack2.account_passwd_hash("node@local.invalid")


def test_ip188_validator_rejects_hash_mismatch_and_missing_step0_checkpoint(tmp_path: Path) -> None:
    """IP-188 negative control:
    - Honest submission with lora_sha256 = '0'*64 on the checked step -> passed=False (with hash_matched=False).
    - Missing step-0 checkpoint returns a rejection verdict instead of raising KeyError.
    """
    base_model, base_sha, _ = build_local_qwen2_backbone(
        seed=42,
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        device=torch.device("cpu"),
    )
    bundle = attach_qlora_adapters(
        base_model=base_model,
        rank_r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        gradient_checkpointing=False,
        seed=42,
    )
    init_state = extract_lora_state_dict(bundle.model)

    shard_path = tmp_path / "shard.parquet"
    _, shard_sha, shard_sig = create_signed_parquet_shard(
        shard_path=shard_path,
        branch_name="Code",
        rng_seed=101,
        num_sequences=4,
        seq_len=8,
        vocab_size=128,
    )
    wu = BOINCWorkunit(
        wu_id="wu-ip188",
        branch_name="Code",
        outer_step=1,
        base_model_id=base_sha,
        lora_checkpoint_url="local",
        data_shard_parquet=str(shard_path),
        shard_sha256=shard_sha,
        shard_hmac_signature=shard_sig,
        rng_seed=101,
        inner_steps_H=2,
        spot_check_interval_s=1,
        seq_len=8,
        num_sequences=4,
    )
    worker = HiveMindWorkerNode(
        worker_id="w-ip188",
        steam_id64="76561198000000188",
        persona_name="Tester",
        model=bundle.model,
    )
    sub = worker.execute_workunit(wu, init_state, lr=5e-4, micro_batch_size=2)
    tokens = torch.randint(0, 128, (4, 8), generator=torch.Generator().manual_seed(101))
    # Reload exact tokens from the signed shard
    from hivemind.boinc.workunit import verify_and_load_parquet_shard

    tokens = verify_and_load_parquet_shard(shard_path, shard_sha, shard_sig, 101)
    validator = DeterministicSpotCheckValidator(tolerance_eps=1e-4)

    # 1. Tamper lora_sha256 on the checked step to '0'*64 while keeping tensors intact
    checked_step = sorted(s for s in sub.micro_checkpoints if s > 0)[0]
    orig_ckpt = sub.micro_checkpoints[checked_step]
    sub.micro_checkpoints[checked_step] = MicroCheckpointRecord(
        step=orig_ckpt.step,
        loss_value=orig_ckpt.loss_value,
        lora_sha256="0" * 64,
        state_snapshot=orig_ckpt.state_snapshot,
    )
    verdict_bad_sha = validator.verify_submission(
        validator_model=bundle.model,
        workunit=wu,
        submission=sub,
        token_sequences=tokens,
        steam_id64="76561198000000188",
        lr=5e-4,
        micro_batch_size=2,
    )
    assert verdict_bad_sha.hash_matched is False
    assert verdict_bad_sha.passed is False
    assert verdict_bad_sha.steam_account_flagged is True

    # 2. Remove step-0 checkpoint -> returns passed=False instead of raising KeyError
    sub.micro_checkpoints.pop(0)
    verdict_no_step0 = validator.verify_submission(
        validator_model=bundle.model,
        workunit=wu,
        submission=sub,
        token_sequences=tokens,
        steam_id64="76561198000000188",
        lr=5e-4,
        micro_batch_size=2,
    )
    assert verdict_no_step0.passed is False
    assert verdict_no_step0.hash_matched is False
    assert "step-0" in verdict_no_step0.reason


def test_ip189_knowledge_share_and_preference_pair_require_verifier_receipt() -> None:
    """IP-189 negative control: admit_share(KnowledgeShare(verified=True), text) with no receipt must be refused."""
    text = "def square(x: int) -> int:\n    return x * x\n"
    share_no_receipt = KnowledgeShare(
        account_id="alice",
        kind="verified_code",
        licence="CC0-1.0",
        verified=True,
        epoch=1,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
    admission = admit_share(share_no_receipt, text)
    assert admission.admitted is False
    assert admission.reason == "missing_verifier_receipt"

    # With a valid 64-hex verifier receipt, admission succeeds
    valid_receipt = hashlib.sha256(f"verified:alice:{text}".encode("utf-8")).hexdigest()
    admission_ok = admit_share(share_no_receipt, text, verifier_receipt_sha256=valid_receipt)
    assert admission_ok.admitted is True

    # PreferencePair defaults to verified=False and refuses verified=True without a 64-hex receipt
    unverified_pair = PreferencePair("p1", "py", "bad", "good", "unit_tests")
    assert unverified_pair.verified is False
    with pytest.raises(ValueError, match="verifier_receipt_sha256"):
        PreferencePair("p1", "py", "bad", "good", "unit_tests", verified=True)

    strat = PreferencePairStrategy()
    assert not strat.observe_correction(
        "p1",
        "py",
        "bad",
        "good",
        verifier_kind="unit_tests",
        chosen_verified=True,
        verifier_receipt_sha256="not-a-valid-sha256",
    )
