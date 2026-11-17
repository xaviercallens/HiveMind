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



def test_inner_steps_continue_through_the_shard_from_round_to_round() -> None:
    """IP-293: each round must train on new windows, not on the shard's first steps*seq_len tokens again."""
    import inspect

    from hivemind.swarm import crowdgpt_moe_opendiloco as moe

    src = inspect.getsource(moe.run_crowd_moe_opendiloco_inner_steps)
    assert "start_offset" in inspect.signature(moe.run_crowd_moe_opendiloco_inner_steps).parameters
    assert "int(start_offset) + step_idx * seq_len" in src and '"next_offset"' in src
    worker_src = (ROOT / "scripts" / "gcp_t4_crowd_diloco_worker.py").read_text(encoding="utf-8")
    assert worker_src.count("start_offset=cursor_") == 2 and "--stop-after-rejections" in worker_src



def test_adaptive_outer_lr_backs_off_on_rejection_and_recovers_on_acceptance() -> None:
    w = _worker()
    kw = {"initial": 0.2, "decay": 0.5, "growth": 1.25, "floor": 0.02}
    lr = 0.2
    lr = w.next_outer_lr(lr, False, **kw)
    assert lr == 0.1
    for _ in range(10):
        lr = w.next_outer_lr(lr, False, **kw)
    assert lr == 0.02  # floor
    lr = w.next_outer_lr(lr, True, **kw)
    assert abs(lr - 0.025) < 1e-12
    assert w.next_outer_lr(0.19, True, **kw) == 0.2  # never above the starting rate
    off = {"initial": 0.2, "decay": 1.0, "growth": 1.0, "floor": 0.02}
    assert w.next_outer_lr(0.2, False, **off) == 0.2 and w.next_outer_lr(0.2, True, **off) == 0.2  # default: unchanged



def test_init_buckets_resume_checks_names_and_shapes(tmp_path) -> None:
    import pytest
    import torch
    from safetensors.torch import save_file

    w = _worker()
    template = {"expert_0": torch.zeros(4, 3), "router_0": torch.zeros(2)}
    ok = tmp_path / "round_7.safetensors"
    save_file({"expert_0": torch.ones(4, 3, dtype=torch.bfloat16), "router_0": torch.full((2,), 0.5, dtype=torch.bfloat16)}, str(ok))
    got = w.load_init_buckets(ok, template)
    assert got["expert_0"].dtype == torch.float32 and float(got["router_0"][0]) == 0.5
    bad = tmp_path / "bad.safetensors"
    save_file({"expert_0": torch.ones(4, 3)}, str(bad))
    with pytest.raises(ValueError, match="bucket names differ"):
        w.load_init_buckets(bad, template)
    shape = tmp_path / "shape.safetensors"
    save_file({"expert_0": torch.ones(5, 3), "router_0": torch.ones(2)}, str(shape))
    with pytest.raises(ValueError, match="shape"):
        w.load_init_buckets(shape, template)
