# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for PR #4:
- Thermodynamic ratchet in `boinc_server_scripts.assimilate` (`eval_loss_after <= best_loss`)
- Per-domain isolated adapter checkpointing (`latest_adapter_{domain}.pt/.json`)
- Mixed-domain 80/20 replay buffer (`build_mixed_domain_train_tokens`)
- Cosine learning-rate decay schedule (`cosine_segment_lr`)
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from hivemind.boinc.real_client import boinc_server_scripts
from hivemind.boinc.real_client.moe_app import (
    TRAIN_APP,
    build_mixed_domain_train_tokens,
    cosine_segment_lr,
)


def _write_dummy_wu_output(tmp_path: Path, wu_id: str, loss_before: float, loss_after: float, weight_val: float, domain: str = "math", domain_isolated: bool = False) -> tuple[str, str]:
    meta_path = tmp_path / f"{wu_id}_meta.json"
    pt_path = tmp_path / f"{wu_id}_out.pt"
    meta_path.write_text(
        json.dumps(
            {
                "wu_id": wu_id,
                "app": TRAIN_APP,
                "domain": domain,
                "domain_isolated_adapter": domain_isolated,
                "eval_loss_before": loss_before,
                "eval_loss_after": loss_after,
            }
        ),
        encoding="utf-8",
    )
    torch.save({"states": [{"lora_A": torch.full((4, 4), weight_val, dtype=torch.float32)}]}, pt_path)
    return str(meta_path), str(pt_path)


def test_assimilate_thermodynamic_ratchet_and_domain_isolation(tmp_path: Path, monkeypatch) -> None:
    ledger_dir = tmp_path / "ledger"
    monkeypatch.setattr(boinc_server_scripts, "LEDGER_DIR", ledger_dir)

    # Round 1: initial round (eval_loss_after = 2.20) -> accepted as chain head
    m1, p1 = _write_dummy_wu_output(tmp_path, "hmq_00001", 2.27, 2.20, 1.0, domain="math", domain_isolated=True)
    assert boinc_server_scripts.assimilate([m1, p1]) == 0
    head1 = json.loads((ledger_dir / "latest_adapter.json").read_text(encoding="utf-8"))
    assert head1["wu_id"] == "hmq_00001"
    assert head1["eval_loss_after"] == 2.20
    assert (ledger_dir / "latest_adapter_math.json").is_file()

    # Round 2: regression round (eval_loss_after = 2.28 > 2.20) -> held by ratchet, chain head stays hmq_00001
    m2, p2 = _write_dummy_wu_output(tmp_path, "hmq_00002", 2.20, 2.28, 2.0, domain="edu_web")
    assert boinc_server_scripts.assimilate([m2, p2]) == 0
    head2 = json.loads((ledger_dir / "latest_adapter.json").read_text(encoding="utf-8"))
    assert head2["wu_id"] == "hmq_00001"
    assert head2["eval_loss_after"] == 2.20
    assert (ledger_dir / "adapter_hmq_00002.pt").is_file()

    # Round 3: improved round (eval_loss_after = 2.13 <= 2.20) -> accepted by ratchet, chain head advances to hmq_00003
    m3, p3 = _write_dummy_wu_output(tmp_path, "hmq_00003", 2.20, 2.13, 3.0, domain="biomed")
    assert boinc_server_scripts.assimilate([m3, p3]) == 0
    head3 = json.loads((ledger_dir / "latest_adapter.json").read_text(encoding="utf-8"))
    assert head3["wu_id"] == "hmq_00003"
    assert head3["eval_loss_after"] == 2.13

    records = [
        json.loads(line)
        for line in (ledger_dir / "assimilated.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [r["ratchet_accepted"] for r in records] == [True, False, True]


def test_build_mixed_domain_train_tokens_80_20_replay() -> None:
    target_tokens = [[100, i] for i in range(10)]
    replay_pools = {
        "math": [[100, 99]],
        "biomed": [[200, 1], [200, 2]],
        "general": [[300, 1], [300, 2]],
    }
    mixed = build_mixed_domain_train_tokens(
        target_tokens,
        replay_pools,
        target_domain="math",
        replay_ratio=0.2,
        rng_seed=7,
    )
    assert len(mixed) == 10
    target_count = sum(1 for seq in mixed if seq[0] == 100)
    replay_count = sum(1 for seq in mixed if seq[0] in (200, 300))
    assert target_count == 8
    assert replay_count == 2


def test_cosine_segment_lr_monotonic_decay() -> None:
    lr_start = cosine_segment_lr(1e-4, 0, 4, round_idx=0, total_rounds=100, min_lr_scale=0.2)
    lr_mid = cosine_segment_lr(1e-4, 0, 4, round_idx=50, total_rounds=100, min_lr_scale=0.2)
    lr_end = cosine_segment_lr(1e-4, 4, 4, round_idx=99, total_rounds=100, min_lr_scale=0.2)
    assert abs(lr_start - 1e-4) < 1e-9
    assert abs(lr_mid - 6e-5) < 1e-6
    assert abs(lr_end - 2e-5) < 1e-9
    assert lr_start > lr_mid > lr_end
