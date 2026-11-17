# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression tests for IP-11/IP-12 (#381), IP-159 (#500), and IP-140 (#495)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from hivemind.boinc.real_client import boinc_server_scripts as bss
from hivemind.boinc.real_client import contributor_policy as cp
from hivemind.boinc.real_client import moe_bridge
from hivemind.boinc.real_client.moe_app import state_sha
from hivemind.boinc.workunit import sign_shard_bytes

K, SEG = 4, 10


def _write_train_artifacts(
    tmp_path: Path,
    wu_id: str,
    states: list[dict[str, torch.Tensor]],
    *,
    flops_claimed: float = 3.84e11,
    wall_seconds: float = 10.0,
    start_sha: str | None = None,
    tag: str = "",
) -> tuple[dict, Path, Path]:
    tokens = [[1, 2, 3, 4]] * 4
    sha, sig = sign_shard_bytes(json.dumps(tokens, separators=(",", ":")).encode(), 7)
    spec = {
        "wu_id": wu_id,
        "app": "hivemind_qlora_moe",
        "inner_steps": K * SEG,
        "segment_steps": SEG,
        "train_tokens": tokens,
        "rng_seed": 7,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "micro_batch": 1,
        "shard_sha256": sha,
        "shard_hmac_signature": sig,
        "start_adapter_sha256": start_sha or state_sha(states[0]),
        "model_id": "m",
        "rank": 8,
        "alpha": 16,
        "init_seed": 1,
    }
    meta = {
        "wu_id": wu_id,
        "app": "hivemind_qlora_moe",
        "state_sha256": [state_sha(s) for s in states],
        "flops_claimed": flops_claimed,
        "wall_seconds": wall_seconds,
    }
    suffix = f"_{tag}" if tag else ""
    mp = tmp_path / f"{wu_id}{suffix}_meta.json"
    sp = tmp_path / f"{wu_id}{suffix}_states.pt"
    mp.write_text(json.dumps(meta))
    torch.save({"states": states}, sp)
    return spec, mp, sp


def _patched_validator(monkeypatch: pytest.MonkeyPatch, states: list[dict[str, torch.Tensor]]) -> moe_bridge.MoEValidator:
    v = moe_bridge.MoEValidator()
    monkeypatch.setattr(
        v,
        "_ensure",
        lambda spec: setattr(v, "_init_state", states[0]) or setattr(v, "_model", object()),
    )
    monkeypatch.setattr(v, "_rng", type("R", (), {"randrange": staticmethod(lambda n: 0)})())

    def replay(model, start, tokens, a, b, **kw):
        k = a // SEG
        return {"w": states[k + 1]["w"].clone()}

    import hivemind.core.diloco_inner as di

    monkeypatch.setattr(di, "replay_inner_segment", replay)
    return v


def test_validate_rejects_wu_old_replayed_as_wu_new(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bss, "SPEC_DIR", tmp_path / "specs")
    monkeypatch.setattr(bss, "LEDGER_DIR", tmp_path / "ledger")
    bss.SPEC_DIR.mkdir()
    bss.LEDGER_DIR.mkdir()

    states_old = [{"w": torch.full((3,), float(i))} for i in range(K + 1)]
    states_new = [{"w": torch.full((3,), float(i + 10))} for i in range(K + 1)]
    spec_old, mp_old, sp_old = _write_train_artifacts(tmp_path, "wu_OLD", states_old)
    spec_new, _, _ = _write_train_artifacts(tmp_path, "wu_NEW", states_new)
    (bss.SPEC_DIR / "wu_OLD.json").write_text(json.dumps(spec_old))
    (bss.SPEC_DIR / "wu_NEW.json").write_text(json.dumps(spec_new))

    v = _patched_validator(monkeypatch, states_old)

    # DB lookup binds result 99 to wu_NEW (elapsed 12s); uploading wu_OLD files must be rejected
    directory = cp.BoincHostDirectory(lambda q: "wu_NEW\t12.0\n" if "workunit w" in q else "1\t10\n")
    rc = bss.run_validation([str(mp_old), str(sp_old), "99"], mv=v, directory=directory)
    assert rc == 1
    verdict_files = list(bss.LEDGER_DIR.glob("verdict_wu_NEW_*.json"))
    assert len(verdict_files) == 1
    verdict_doc = json.loads(verdict_files[0].read_text())
    assert verdict_doc["valid"] is False
    assert "mismatch" in verdict_doc["reason"]


def test_inflated_claim_earns_same_credit_as_honest_and_nan_wall_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = [{"w": torch.full((3,), float(i))} for i in range(K + 1)]
    # Spec FLOPs bound for 40 steps * 1 micro_batch * 4 tokens = 160 tokens -> 6 * 4e8 * 160 = 3.84e11 FLOPs
    spec, mp_honest, sp_honest = _write_train_artifacts(
        tmp_path, "wu_credit", states, flops_claimed=3.84e11, wall_seconds=10.0, tag="honest"
    )
    _, mp_inflated, sp_inflated = _write_train_artifacts(
        tmp_path, "wu_credit", states, flops_claimed=1e25, wall_seconds=1e6, tag="inflated"
    )
    _, mp_nan, sp_nan = _write_train_artifacts(
        tmp_path, "wu_credit", states, flops_claimed=1e30, wall_seconds=float("nan"), tag="nan"
    )

    v = _patched_validator(monkeypatch, states)
    verdict_honest = v.validate_train(spec, mp_honest, sp_honest, "h1", server_elapsed_seconds=10.0)
    verdict_inflated = v.validate_train(spec, mp_inflated, sp_inflated, "h1", server_elapsed_seconds=10.0)
    verdict_nan = v.validate_train(spec, mp_nan, sp_nan, "h1", server_elapsed_seconds=10.0)

    assert verdict_honest.valid is True
    assert verdict_inflated.valid is True
    assert verdict_inflated.grant["cobblestones"] == pytest.approx(verdict_honest.grant["cobblestones"])

    assert verdict_nan.valid is False
    assert verdict_nan.grant["cobblestones"] == 0.0


def test_assimilate_moves_chain_head_only_forward_and_rejects_rollback_or_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bss, "SPEC_DIR", tmp_path / "specs")
    monkeypatch.setattr(bss, "LEDGER_DIR", tmp_path / "ledger")
    bss.SPEC_DIR.mkdir()
    bss.LEDGER_DIR.mkdir()

    states_1 = [{"w": torch.full((3,), float(i))} for i in range(K + 1)]
    states_2 = [states_1[-1]] + [{"w": torch.full((3,), float(i + 10))} for i in range(1, K + 1)]

    spec_1, mp_1, sp_1 = _write_train_artifacts(
        tmp_path, "hmq_00001", states_1, start_sha=state_sha(states_1[0])
    )
    spec_2, mp_2, sp_2 = _write_train_artifacts(
        tmp_path, "hmq_00002", states_2, start_sha=state_sha(states_1[-1])
    )
    (bss.SPEC_DIR / "hmq_00001.json").write_text(json.dumps(spec_1))
    (bss.SPEC_DIR / "hmq_00002.json").write_text(json.dumps(spec_2))

    dir_1 = cp.BoincHostDirectory(lambda q: "hmq_00001\n")
    dir_2 = cp.BoincHostDirectory(lambda q: "hmq_00002\n")

    assert bss.assimilate(["101", str(mp_1), str(sp_1)], directory=dir_1) == 0
    assert json.loads((bss.LEDGER_DIR / "latest_adapter.json").read_text())["wu_id"] == "hmq_00001"

    assert bss.assimilate(["102", str(mp_2), str(sp_2)], directory=dir_2) == 0
    head_after_2 = json.loads((bss.LEDGER_DIR / "latest_adapter.json").read_text())
    assert head_after_2["wu_id"] == "hmq_00002"
    assert head_after_2["sha256"] == state_sha(states_2[-1])

    # Replaying hmq_00001 (or uploading hmq_00001 for wu_db_id 102) is rejected; chain head never moves backwards
    assert bss.assimilate(["101", str(mp_1), str(sp_1)], directory=dir_1) == 1
    assert bss.assimilate(["102", str(mp_1), str(sp_1)], directory=dir_2) == 1
    head_after_replay = json.loads((bss.LEDGER_DIR / "latest_adapter.json").read_text())
    assert head_after_replay["wu_id"] == "hmq_00002"
    assert head_after_replay["sha256"] == state_sha(states_2[-1])


def test_checkpoint_state_excludes_memory_notes_and_email_literals_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("checkpoint_state", root / "scripts" / "checkpoint_state.py")
    assert spec is not None and spec.loader is not None
    cs = importlib.util.module_from_spec(spec)
    sys.modules["checkpoint_state"] = cs
    spec.loader.exec_module(cs)

    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    (mem_dir / "note_plain.md").write_text(" Campaign status notes without personal addresses.\n", encoding="utf-8")
    (mem_dir / "note_email.md").write_text("Private contact: researcher@university.example\n", encoding="utf-8")
    monkeypatch.setattr(cs, "MEMORY_DIR", mem_dir)
    monkeypatch.setattr(cs, "ROOT", tmp_path)

    # Default collect() must exclude all state/memory/* paths
    default_items = cs.collect()
    assert not any(rel.startswith("state/memory/") for rel, _ in default_items)

    # Even with include_memory=True, notes containing email addresses are skipped by SECRET_RE
    opt_in_items = cs.collect(include_memory=True)
    opt_in_paths = [rel for rel, _ in opt_in_items]
    assert "state/memory/note_plain.md" in opt_in_paths
    assert "state/memory/note_email.md" not in opt_in_paths
