# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""--cuda-mem-cap-mib: a hard per-process cap so a trainer cannot push free VRAM below the floor (LL.md, lesson 1)."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _worker():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("t4_worker_cap", ROOT / "scripts" / "gcp_t4_crowd_diloco_worker.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_default_is_no_cap() -> None:
    w = _worker()
    assert w.resolve_cuda_mem_cap_mib("0", 10_091, 4300) == (0, "")


def test_auto_cap_keeps_the_floor_measured_2026_11_15() -> None:
    # T4 with only llama-server resident: 10,091 MiB free -> 10,091 - 4,300 - 700 = 5,091 MiB (the night run's cap)
    w = _worker()
    assert w.resolve_cuda_mem_cap_mib("auto", 10_091, 4300) == (5091, "")


def test_a_fixed_cap_that_would_break_the_floor_is_refused() -> None:
    w = _worker()
    cap, why = w.resolve_cuda_mem_cap_mib("6000", 10_091, 4300)
    assert cap == 0 and "at most 5091" in why
    assert w.resolve_cuda_mem_cap_mib("4000", 10_091, 4300) == (4000, "")


def test_too_little_headroom_or_bad_input_refuses_to_start() -> None:
    w = _worker()
    cap, why = w.resolve_cuda_mem_cap_mib("auto", 5_200, 4300)
    assert cap == 0 and "not starting" in why
    cap, why = w.resolve_cuda_mem_cap_mib("lots", 10_091, 4300)
    assert cap == 0 and "must be 0, auto" in why



def test_each_round_is_written_to_rounds_jsonl_as_it_ends(tmp_path) -> None:
    """A stopped run keeps every finished round (the 2026-11-16 run lost 250 rounds of results)."""
    import json

    w = _worker()
    w.append_round_record(tmp_path, {"round_index": 1, "accepted_by_loss_ratchet": False})
    w.append_round_record(tmp_path, {"round_index": 2, "accepted_by_loss_ratchet": True, "eval_loss_after": 4.7})
    w.append_round_record(None, {"round_index": 3})  # no work dir: nothing written, no error
    rows = [json.loads(x) for x in (tmp_path / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["round_index"] for r in rows] == [1, 2] and rows[1]["accepted_by_loss_ratchet"] is True
