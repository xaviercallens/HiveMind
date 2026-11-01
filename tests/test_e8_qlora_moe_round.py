# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""E8 stage 2 (tiny MoE, LoRA deltas across machines): the outer step, the payload and bind safety, the VRAM guard, and a real HTTP round compared with the single-process baseline."""

from __future__ import annotations

import argparse
import base64
import io
import json
import socket
import sys
import threading
from pathlib import Path

import pytest
import torch

pytest.importorskip("peft")
pytest.importorskip("transformers")

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
import e8_qlora_moe_round as q  # noqa: E402

RAN: list[str] = []


def _marker() -> None:
    RAN.append("code ran")


class _Evil:
    def __reduce__(self):  # noqa: ANN204
        return (_marker, ())


# ----------------------------------------------------------------------------------------------------------------------- outer step


def test_outer_step_is_the_token_weighted_mean_then_momentum() -> None:
    g = {"w": torch.zeros(3)}
    d1, d2 = {"w": torch.tensor([1.0, 0.0, 0.0])}, {"w": torch.tensor([0.0, 4.0, 0.0])}
    new_g, v = q.outer_step(g, [d1, d2], [3, 1], None, eta=1.0, mu=0.5)
    assert torch.allclose(v["w"], torch.tensor([0.75, 1.0, 0.0]))  # (3*d1 + 1*d2) / 4
    assert torch.allclose(new_g["w"], v["w"])
    g2, v2 = q.outer_step(new_g, [{"w": torch.zeros(3)}], [1], v, eta=0.5, mu=0.5)
    assert torch.allclose(v2["w"], 0.5 * v["w"]) and torch.allclose(g2["w"], new_g["w"] + 0.5 * v2["w"])  # momentum keeps moving with zero deltas


def test_outer_step_rejects_bad_inputs() -> None:
    g = {"w": torch.zeros(1)}
    for deltas, weights in (([], []), ([{"w": torch.zeros(1)}], [0]), ([{"w": torch.zeros(1)}], [1, 2])):
        with pytest.raises(ValueError):
            q.outer_step(g, deltas, weights, None, 1.0, 0.0)


# ----------------------------------------------------------------------------------------------------------------------- safety


def test_payloads_roundtrip_and_a_payload_carrying_code_is_refused() -> None:
    d = {"a": torch.arange(4, dtype=torch.float32)}
    assert torch.equal(q.b64_to_tensors(q.tensors_to_b64(d))["a"], d["a"])
    buf = io.BytesIO()
    torch.save({"w": _Evil()}, buf)
    RAN.clear()
    with pytest.raises(Exception):  # noqa: B017, PT011
        q.b64_to_tensors(base64.b64encode(buf.getvalue()).decode("ascii"))
    assert RAN == []


@pytest.mark.parametrize("wide", ["0.0.0.0", "::", "[::]", ""])
def test_the_syncer_refuses_to_listen_on_every_interface(monkeypatch: pytest.MonkeyPatch, wide: str) -> None:
    with pytest.raises(SystemExit) as exc:
        q.main(["syncer", "--bind", wide, "--out", "x.json"])
    assert exc.value.code == 2


def test_the_vram_guard_never_lets_free_memory_fall_below_the_floor() -> None:
    assert q.vram_guard(4300, 600, "cpu", lambda: None) is None  # CPU runs are not subject to it
    assert q.vram_guard(4300, 600, "cuda", lambda: 10_000.0) == 10_000.0
    with pytest.raises(q.GuardError):
        q.vram_guard(4300, 600, "cuda", lambda: 4_800.0)  # 4,800 - 600 = 4,200 < 4,300
    assert q.vram_guard(4300, 600, "cuda", lambda: 4_900.0) == 4_900.0  # exactly the floor is allowed
    with pytest.raises(q.GuardError, match="cannot read"):
        q.vram_guard(4300, 600, "cuda", lambda: None)  # an unreadable GPU is a refusal, not a pass


def test_a_learner_that_would_break_the_guard_exits_with_3_before_touching_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(q, "free_vram_mib", lambda: 4_400.0)
    monkeypatch.setattr(q, "vram_guard", lambda m, n, d, free_fn=q.free_vram_mib: (_ for _ in ()).throw(q.GuardError("no")) if d == "cuda" else None)
    rc = q.main(["learner", "--syncer", "http://127.0.0.1:1", "--learner-id", "0", "--device", "cuda"])
    assert rc == 3


# ----------------------------------------------------------------------------------------------------------------------- data


def test_shards_are_deterministic_and_genuinely_different_per_learner() -> None:
    a, a2, b = q.make_shard(0, 16, 1), q.make_shard(0, 16, 1), q.make_shard(1, 16, 1)
    assert torch.equal(a, a2) and not torch.equal(a, b)
    assert a.shape == (16, q.SEQ_LEN) and int(a.max()) < q.VOCAB
    e = q.make_eval(2, 1, 8)
    assert e.shape[0] == 16 and not any(torch.equal(e[0], s) for s in a)  # the held-out set is not a training sequence


# ----------------------------------------------------------------------------------------------------------------------- learning


@pytest.fixture(scope="module")
def tiny() -> dict:
    base = q.build_base_state(1)
    return {"base": base, "lora": q.initial_lora(base, 1)}


def test_a_learner_round_returns_a_nonzero_delta_and_leaves_the_global_state_alone(tiny: dict) -> None:
    model = q.load_model(tiny["base"], "none", "cpu", 1)
    g_before = {k: v.clone() for k, v in tiny["lora"].items()}
    out = q.learner_round(model, tiny["lora"], q.make_shard(0, 64, 1), q.make_eval(2, 1, 8), steps=5, lr=2e-2, batch=8, seed=1, rnd=0, learner_id=0, device="cpu", shuffle_labels=False)
    assert set(out["delta"]) == set(tiny["lora"]) and out["delta_l2"] > 0 and out["tokens"] == 5 * 8 * q.SEQ_LEN
    assert all(torch.equal(tiny["lora"][k], g_before[k]) for k in g_before)


def test_real_labels_learn_and_shuffled_labels_do_not(tiny: dict) -> None:
    """Negative control: the same budget on shuffled targets must not give the improvement the real targets give."""
    shard, ev = q.make_shard(0, 128, 1), q.make_eval(1, 1, 16)
    gains = {}
    for shuffled in (False, True):
        model = q.load_model(tiny["base"], "none", "cpu", 1)
        q.set_lora_state(model, tiny["lora"])
        before = q.eval_loss(model, ev, "cpu")
        q.train_inner(model, shard, steps=30, lr=2e-2, batch=16, seed=3, device="cpu", shuffle_labels=shuffled)
        gains[shuffled] = before - q.eval_loss(model, ev, "cpu")
    assert gains[False] > 0.02 and gains[True] < 0.5 * gains[False], gains


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_a_real_http_round_across_two_learners_matches_the_single_process_baseline(tmp_path: Path) -> None:
    port = _free_port()
    common = dict(inner_steps=4, inner_lr=2e-2, batch=8, outer_lr=0.7, outer_mu=0.65, seed=1)
    srv = argparse.Namespace(bind="127.0.0.1", port=port, learners=2, rounds=2, timeout=240.0, out=str(tmp_path / "wan.json"), **common)
    rcs: dict[str, int] = {}
    t = threading.Thread(target=lambda: rcs.__setitem__("syncer", q.run_syncer(srv)), daemon=True)
    t.start()
    for _ in range(100):
        try:
            q.http_json(f"http://127.0.0.1:{port}/state")
            break
        except OSError:
            threading.Event().wait(0.1)
    ths = []
    for i in range(2):
        a = argparse.Namespace(syncer=f"http://127.0.0.1:{port}", learner_id=i, host_label=f"test-{i}", quant="none", device="cpu", shard_size=64, shuffle_labels=False,
                               min_free_mib=4300.0, need_mib=600.0, **common)
        th = threading.Thread(target=lambda a=a: rcs.__setitem__(f"l{a.learner_id}", q.run_learner(a)), daemon=True)
        th.start()
        ths.append(th)
    for th in ths:
        th.join(240)
    t.join(60)
    assert rcs == {"syncer": 0, "l0": 0, "l1": 0}
    wan = json.loads((tmp_path / "wan.json").read_text())
    base_args = argparse.Namespace(learners=2, rounds=2, quant="none", device="cpu", shard_size=64, shuffle_labels=False, min_free_mib=4300.0, need_mib=600.0,
                                   out=str(tmp_path / "base.json"), **common)
    assert q.run_baseline(base_args) == 0
    base = json.loads((tmp_path / "base.json").read_text())
    assert wan["completed"] and wan["hosts"] == {"0": "test-0", "1": "test-1"} and len(wan["rounds"]) == 2
    assert abs(wan["final_eval_loss_mean"] - base["final_eval_loss_mean"]) < 1e-4, (wan["final_eval_loss_mean"], base["final_eval_loss_mean"])
