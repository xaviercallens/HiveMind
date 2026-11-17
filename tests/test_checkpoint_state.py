# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression tests for IP-140 (checkpoint_state.py), IP-178 (workunit TFLOPs), and IP-180 (Wave-3 replay verifier)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from hivemind.boinc.workunit import BOINCWorkunit
from hivemind.cli.simulate_e2e import estimate_workunit_tflops
from hivemind.client.steamworks import SteamworksSDKBridge

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, rel_path: str):
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_ip140_checkpoint_collect_excludes_memory_by_default_and_blocks_email_notes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IP-140: collect() must yield no state/memory/* path by default and skip any note containing an e-mail."""
    cs = _load_script_module("checkpoint_state_mod", "scripts/checkpoint_state.py")

    fake_root = tmp_path / "repo"
    exp_dir = fake_root / "artifacts" / "experiments" / "E15"
    exp_dir.mkdir(parents=True)
    (exp_dir / "stage1.json").write_text('{"stage": 1, "status": "ok"}\n', encoding="utf-8")

    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    (mem_dir / "clean_note.md").write_text("# Campaign plan\nRun stage 1 then stage 2.\n", encoding="utf-8")
    (mem_dir / "private_note.md").write_text("Owner contact: user_note@example.org\n", encoding="utf-8")

    monkeypatch.setattr(cs, "ROOT", fake_root)
    monkeypatch.setattr(cs, "JOB_DIR", tmp_path / "no_jobs")
    monkeypatch.setattr(cs, "MEMORY_DIR", mem_dir)
    cs.SKIPPED_SECRET_LIKE.clear()

    # 1. Default collect() yields no state/memory/* paths even when MEMORY_DIR is populated
    default_items = dict(cs.collect())
    assert "state/artifacts/experiments/E15/stage1.json" in default_items
    assert not any(k.startswith("state/memory/") for k in default_items)

    # 2. Even with include_memory=True, a note containing an email address is blocked by SECRET_RE
    cs.SKIPPED_SECRET_LIKE.clear()
    mem_items = dict(cs.collect(include_memory=True))
    assert "state/memory/clean_note.md" in mem_items
    assert "state/memory/private_note.md" not in mem_items
    assert "state/memory/private_note.md" in cs.SKIPPED_SECRET_LIKE


def test_ip178_workunit_tflops_scales_with_inner_steps_and_tokens() -> None:
    """IP-178: PoC workunits with inner_steps=1 vs 12 must record distinct verified_tflops."""
    wu_1 = BOINCWorkunit(
        wu_id="wu-steps-1",
        branch_name="Code",
        outer_step=1,
        base_model_id="sha256:base",
        lora_checkpoint_url="local",
        data_shard_parquet="shard.parquet",
        shard_sha256="a" * 64,
        shard_hmac_signature="b" * 64,
        rng_seed=42,
        inner_steps_H=1,
        spot_check_interval_s=1,
        seq_len=48,
        num_sequences=8,
    )
    wu_12 = BOINCWorkunit(
        wu_id="wu-steps-12",
        branch_name="Code",
        outer_step=1,
        base_model_id="sha256:base",
        lora_checkpoint_url="local",
        data_shard_parquet="shard.parquet",
        shard_sha256="a" * 64,
        shard_hmac_signature="b" * 64,
        rng_seed=42,
        inner_steps_H=12,
        spot_check_interval_s=4,
        seq_len=48,
        num_sequences=8,
    )
    tflops_1 = estimate_workunit_tflops(wu_1, total_params=494_000_000, micro_batch_size=2)
    tflops_12 = estimate_workunit_tflops(wu_12, total_params=494_000_000, micro_batch_size=2)
    assert tflops_1 > 0.0
    assert tflops_12 != tflops_1
    assert tflops_12 == pytest.approx(12.0 * tflops_1, rel=1e-4)

    bridge = SteamworksSDKBridge()
    e1 = bridge.record_validated_workunit("76561198000000001", "Miner1", wu_1.wu_id, tflops_contributed=tflops_1)
    e12 = bridge.record_validated_workunit("76561198000000002", "Miner12", wu_12.wu_id, tflops_contributed=tflops_12)
    assert e1.verified_tflops != e12.verified_tflops
    assert e12.verified_tflops > e1.verified_tflops


def test_ip180_wave3_replay_examples_verify_formula_and_reject_corrupted_sum() -> None:
    """IP-180: changing triangular_sum total from n*(n+1)//2 to n*n must stop rows being stamped VERIFIED."""
    prep = _load_script_module("prepare_night_sci_wave3_mod", "scripts/prepare_night_sci_wave3_datasets.py")

    valid_rows = prep._general_replay_examples(4)
    assert len(valid_rows) == 4
    assert all(r["verdict"] == "VERIFIED" for r in valid_rows)

    corrupted_rows = prep._general_replay_examples(4, total_fn=lambda n: n * n)
    assert len(corrupted_rows) == 4
    assert all(r["verdict"] == "REJECTED" for r in corrupted_rows)
    assert all(r["certificate"]["matches_independent_sum"] is False for r in corrupted_rows)
