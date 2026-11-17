"""Resumable checkpoints for the QLoRA expert trainer: exact resume, crash safety, and controls that can fail (CPU, tiny model)."""

from __future__ import annotations

import importlib.util
import math
import random
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ck = _load("e17_ckpt")
orch = _load("night_orchestrator")


class Preempted(Exception):
    """Stands for the Spot VM being stopped between two optimizer steps."""


def _seed(n: int) -> None:
    random.seed(n)
    torch.manual_seed(n)


def _mini_train(root: Path | None, *, ckpt_every: int, crash_after: int | None = None, restore_opt: bool = True,
                restore_rng: bool = True, reseed: int = 17) -> dict[str, torch.Tensor]:
    """The trainer's loop on a tiny CPU model: shuffled epochs, dropout (consumes torch RNG), AdamW, the same position/RNG
    bookkeeping through e17_ckpt. Returns the final weights."""
    _seed(reseed)
    n, accum, epochs = 10, 3, 3
    gen = torch.Generator().manual_seed(0)
    xs, ys = torch.randn(n, 4, generator=gen), torch.randn(n, 1, generator=gen)
    steps_per_epoch = math.ceil(n / accum)
    total = steps_per_epoch * epochs
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.3), torch.nn.Linear(8, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    step, start_epoch, start_s = 0, 0, 0
    found = ck.latest_checkpoint(root) if (root is not None and ckpt_every > 0) else None
    if found is not None:
        _, st = found
        model.load_state_dict(st["model"])
        if restore_opt:
            opt.load_state_dict(st["opt"])
        step, start_epoch, start_s = st["step"], st["next_epoch"], st["next_s"]
        if restore_rng:
            random.setstate(st["epoch_rng"])
            torch.set_rng_state(st["rng"]["torch_cpu"])
    model.train()
    epoch_rng = random.getstate()
    for epoch in range(start_epoch, epochs):
        epoch_rng = random.getstate()
        order = list(range(n))
        random.shuffle(order)
        for s in range(steps_per_epoch):
            if epoch == start_epoch and s < start_s:
                continue
            idx = order[s * accum:(s + 1) * accum]
            loss = torch.nn.functional.mse_loss(model(xs[idx]), ys[idx])
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            ne, ns = ck.next_position(epoch, s, steps_per_epoch)
            if root is not None and ckpt_every > 0 and step < total and step % ckpt_every == 0:
                ck.save_checkpoint(root, step, {"step": step, "next_epoch": ne, "next_s": ns,
                                                "epoch_rng": random.getstate() if ns == 0 else epoch_rng, "rng": ck.capture_rng(),
                                                "model": model.state_dict(), "opt": opt.state_dict()})
            if crash_after is not None and step == crash_after:
                raise Preempted
    return {k: v.clone() for k, v in model.state_dict().items()}


def _same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_resume_after_a_mid_run_preemption_is_bit_exact(tmp_path: Path) -> None:
    straight = _mini_train(None, ckpt_every=0)
    with pytest.raises(Preempted):
        _mini_train(tmp_path, ckpt_every=2, crash_after=5)  # last checkpoint is step 4, so step 5's work is lost and redone
    resumed = _mini_train(tmp_path, ckpt_every=2, reseed=999)  # a different seed: all state must come from the checkpoint
    assert ck.latest_checkpoint(tmp_path) is not None
    assert _same(resumed, straight)


def test_resume_across_an_epoch_boundary_is_bit_exact(tmp_path: Path) -> None:
    straight = _mini_train(None, ckpt_every=0)
    with pytest.raises(Preempted):
        _mini_train(tmp_path, ckpt_every=4, crash_after=6)  # steps_per_epoch is 4: checkpoint 4 sits exactly on the boundary
    st = ck.latest_checkpoint(tmp_path)[1]
    assert (st["next_epoch"], st["next_s"]) == (1, 0)
    assert _same(_mini_train(tmp_path, ckpt_every=4, reseed=5), straight)


def test_negative_controls_the_equality_check_can_fail(tmp_path: Path) -> None:
    straight = _mini_train(None, ckpt_every=0)
    for name, kwargs in (("optimizer", {"restore_opt": False}), ("rng", {"restore_rng": False})):
        root = tmp_path / name
        with pytest.raises(Preempted):
            _mini_train(root, ckpt_every=2, crash_after=5)
        broken = _mini_train(root, ckpt_every=2, reseed=999, **kwargs)
        assert not _same(broken, straight), f"resuming without the {name} state should NOT reproduce the run"


def test_rng_state_survives_the_safe_loader(tmp_path: Path) -> None:
    _seed(3)
    ck.save_checkpoint(tmp_path, 1, {"rng": ck.capture_rng(), "t": torch.arange(3)})
    expected = (random.random(), torch.rand(2))
    _seed(77)
    _, st = ck.latest_checkpoint(tmp_path)
    ck.restore_rng(st["rng"])
    assert random.random() == expected[0]
    assert torch.equal(torch.rand(2), expected[1])


def test_a_half_written_or_corrupt_checkpoint_falls_back_to_the_previous_one(tmp_path: Path) -> None:
    ck.save_checkpoint(tmp_path, 10, {"v": torch.tensor([10.0])})
    newest = ck.save_checkpoint(tmp_path, 20, {"v": torch.tensor([20.0])})
    assert ck.latest_checkpoint(tmp_path)[0] == 20
    (newest / ck.DONE_FILE).unlink()  # crash after the data was written but before the marker
    assert ck.latest_checkpoint(tmp_path)[0] == 10
    (newest / ck.DONE_FILE).write_text("20")
    (newest / ck.STATE_FILE).write_bytes(b"truncated")  # marker present but the file is damaged
    step, payload = ck.latest_checkpoint(tmp_path)
    assert step == 10 and torch.equal(payload["v"], torch.tensor([10.0]))
    (tmp_path / ".tmp-00000030-1").mkdir()  # a leftover temporary directory is never read
    assert ck.latest_checkpoint(tmp_path)[0] == 10


def test_only_the_newest_two_checkpoints_are_kept(tmp_path: Path) -> None:
    for s in (2, 4, 6, 8):
        ck.save_checkpoint(tmp_path, s, {"s": s})
    kept = sorted(p.name for p in tmp_path.iterdir())
    assert kept == [f"{ck.PREFIX}00000006", f"{ck.PREFIX}00000008"]
    assert ck.latest_checkpoint(tmp_path)[1]["s"] == 8


def test_fingerprint_changes_with_the_data_and_the_settings(tmp_path: Path) -> None:
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text('{"a": 1}\n')
    base: dict[str, Any] = dict(base="b", pairs_file=pairs, n_data=1, epochs=2, lr=1e-4, accum=16, warmup=10, max_len=768, seed=17)
    fp = ck.fingerprint(**base)
    assert fp == ck.fingerprint(**base)
    assert ck.fingerprint(**{**base, "lr": 2e-4}) != fp
    pairs.write_text('{"a": 2}\n')
    assert ck.fingerprint(**base) != fp  # same file name, different content


def test_completion_marker_and_the_orchestrator_flag(tmp_path: Path) -> None:
    assert not ck.is_complete(tmp_path / "adapter")
    ck.mark_complete(tmp_path / "adapter", {"steps": 5})
    assert ck.is_complete(tmp_path / "adapter")
    cmd = orch.train_cmd("packs/s/train.jsonl", "adapters/s", 2, 0.5)
    assert f"--ckpt-every {orch.CKPT_EVERY}" in cmd and orch.CKPT_EVERY > 0


def test_next_position_covers_every_step_exactly_once() -> None:
    hyp = pytest.importorskip("hypothesis")
    st = pytest.importorskip("hypothesis.strategies")

    @hyp.given(steps_per_epoch=st.integers(1, 12), epochs=st.integers(1, 6))
    def walk(steps_per_epoch: int, epochs: int) -> None:
        pos, seen = (0, 0), []
        while pos[0] < epochs:
            seen.append(pos)
            pos = ck.next_position(*pos, steps_per_epoch)
        assert seen == [(e, s) for e in range(epochs) for s in range(steps_per_epoch)]

    walk()
