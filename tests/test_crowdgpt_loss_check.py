"""GPU loss check for CrowdGPT, verified on a CPU backend: tiny SotaGPT-shaped model, real forward passes on both sides."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hivemind.swarm import crowdgpt_loss_check as lc  # noqa: E402

TINY = {"vocabSize": 97, "dim": 32, "nLayers": 2, "nHeads": 4, "nKvHeads": 2, "headDim": 8, "maxSeqLen": 64, "mlpHidden": 48, "weightTying": True}
W, N, SEED = 17, 10, 3
KEY, SECRET = b"evaluator-key", b"backend-secret"


def factory() -> lc.CrowdV1:
    return lc.CrowdV1(TINY)


def flat_weights(seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    m = factory()
    return torch.cat([p.detach().reshape(-1) for p in m.parameters()]).to(torch.bfloat16)


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    heldout = rng.integers(0, TINY["vocabSize"], size=600, dtype=np.uint32)
    base = flat_weights(1)
    candidate = (base.float() + 0.01 * torch.randn(base.shape, generator=torch.Generator().manual_seed(2))).to(torch.bfloat16)
    receipt = lc.make_receipt(evaluator_id="t4", key=KEY, base=base, candidate=candidate, heldout=heldout, window=W, n=N, seed=SEED,
                              model_factory=factory, device="cpu", round_id="r1")
    return heldout, base, candidate, receipt


def verify(receipt: lc.LossCheckReceipt, world, **kw) -> lc.LossCheckVerdict:
    heldout, base, candidate, _ = world
    args = dict(evaluator_keys={"t4": KEY}, server_secret=SECRET, expected_base_sha256=lc.weights_sha256(base),
                expected_candidate_sha256=lc.weights_sha256(candidate), heldout=heldout,
                recompute=lc.cpu_recompute_fn(factory, base, candidate, heldout, W), k=4, tol=1e-4, max_regression_tolerance=10.0, min_windows=8)
    args.update(kw)
    return lc.verify_receipt(receipt, **args)


def test_flat_layout_round_trips_and_ties_the_head() -> None:
    flat = flat_weights(5)
    m = factory()
    m.load_flat(flat)
    assert m.lm_head.weight.data_ptr() == m.wte.weight.data_ptr()  # one tensor, loaded once
    assert torch.equal(torch.cat([p.detach().reshape(-1) for p in m.parameters()]).to(torch.bfloat16), flat)
    with pytest.raises(ValueError):
        m.load_flat(flat[:-1])


def test_window_loss_matches_a_hand_computed_cross_entropy() -> None:
    m = factory()
    m.load_flat(flat_weights(7))
    toks = np.arange(40, dtype=np.uint32) % TINY["vocabSize"]
    got = lc.window_losses(m, toks, [5], 9)[0]
    t = torch.from_numpy(toks[5:14].astype(np.int64))[None]
    with torch.inference_mode():
        logp = torch.log_softmax(m(t[:, :-1])[0].float(), -1)
    expect = -float(logp[torch.arange(8), t[0, 1:]].mean())
    assert got == pytest.approx(expect, abs=1e-5) and 0.0 < got < math.log(TINY["vocabSize"]) + 3
    with pytest.raises(ValueError):
        lc.window_losses(m, np.array([0, 1, 500], dtype=np.uint32), [0], 3)  # token outside the vocabulary


def test_an_honest_receipt_is_accepted_with_its_spot_checks(world) -> None:
    v = verify(world[3], world)
    assert v.accepted and v.reason.startswith("ACCEPTED") and len(v.checked) == 4
    assert v.detail["max_abs_diff"] <= 1e-4 and v.delta_loss == pytest.approx(world[3].mean_after - world[3].mean_before)


def test_a_falsified_loss_is_caught_when_checked_and_missed_when_not(world) -> None:
    r = world[3]
    lie = lc._replace(r, losses_after=tuple(x - 0.5 if i == 0 else x for i, x in enumerate(r.losses_after))).signed(KEY)
    caught = verify(lie, world, k=2 * N)  # every number checked
    assert not caught.accepted and caught.reason.startswith("REJECTED_SPOT_CHECK") and "after window 0" in caught.reason
    missed = verify(lie, world, k=0)  # negative control: without spot checks the signed lie passes
    assert missed.accepted


def test_structural_checks_reject_before_any_forward_pass(world) -> None:
    heldout, base, candidate, r = world
    calls: list[str] = []

    def spy(which: str, starts: list[int]) -> list[float]:
        calls.append(which)
        return [0.0] * len(starts)

    cases = {
        "REJECTED_UNKNOWN_EVALUATOR": dict(evaluator_keys={"other": KEY}),
        "REJECTED_BAD_SIGNATURE": dict(evaluator_keys={"t4": b"wrong"}),
        "REJECTED_WRONG_WEIGHTS": dict(expected_candidate_sha256=lc.weights_sha256(base)),
        "REJECTED_WRONG_HELDOUT_DATA": dict(heldout=heldout[::-1].copy()),
        "REJECTED_TOO_FEW_WINDOWS": dict(min_windows=N + 1),
    }
    for reason, kw in cases.items():
        assert verify(r, world, recompute=spy, **kw).reason.startswith(reason), reason
    moved = lc._replace(r, starts=tuple(s + 1 for s in r.starts)).signed(KEY)
    assert verify(moved, world, recompute=spy).reason == "REJECTED_WINDOWS_NOT_FROM_SEED"
    for bad in (float("nan"), -0.1, 99.0):
        crazy = lc._replace(r, losses_before=(bad,) + r.losses_before[1:]).signed(KEY)
        assert verify(crazy, world, recompute=spy).reason == "REJECTED_IMPLAUSIBLE_LOSS"
    assert calls == []


def test_a_regression_holds_the_previous_head(world) -> None:
    v = verify(world[3], world, max_regression_tolerance=-10.0)
    assert not v.accepted and v.reason.startswith("HELD_PREVIOUS_HEAD") and len(v.checked) == 4


def test_spot_check_picks_depend_on_the_backend_secret() -> None:
    a = lc.spot_check_picks("c" * 64, b"s1", 20, 5)
    assert a == lc.spot_check_picks("c" * 64, b"s1", 20, 5) and len(set(a)) == 5
    others = {tuple(lc.spot_check_picks("c" * 64, bytes([i]), 20, 5)) for i in range(30)}
    assert len(others) > 20  # a different secret gives different picks: the evaluator cannot compute them
    assert all(w in ("before", "after") and 0 <= i < 20 for w, i in a)


@settings(max_examples=60, deadline=None)
@given(n=st.integers(2, 40), m=st.integers(0, 40), k=st.integers(0, 40))
def test_detection_probability_is_a_probability_and_monotone_in_k(n: int, m: int, k: int) -> None:
    m, k = min(m, n), min(k, n)
    p = lc.detection_probability(n, m, k)
    assert 0.0 <= p <= 1.0
    if k < n:
        assert lc.detection_probability(n, m, k + 1) >= p - 1e-12
    if m == 0 or k == 0:
        assert p == 0.0


def test_detection_probability_matches_simulation() -> None:
    rng = np.random.default_rng(1)
    hits = sum(bool(set(rng.choice(64, 4, replace=False)) & set(range(3))) for _ in range(20000))
    assert lc.detection_probability(64, 3, 4) == pytest.approx(hits / 20000, abs=0.01)
    assert lc.detection_probability(64, 3, 4) == pytest.approx(1 - math.comb(61, 4) / math.comb(64, 4))


def test_two_evaluators_must_agree(world) -> None:
    heldout, base, candidate, r = world
    r2 = lc.make_receipt(evaluator_id="rtx3090", key=b"k2", base=base, candidate=candidate, heldout=heldout, window=W, n=N, seed=SEED,
                         model_factory=factory, device="cpu")
    ok, why = lc.receipts_agree(r, r2, tol=1e-4)
    assert ok, why
    off = lc._replace(r2, losses_before=(r2.losses_before[0] + 0.01,) + r2.losses_before[1:])
    assert not lc.receipts_agree(r, off, tol=1e-4)[0] and not lc.receipts_agree(r, r)[0]


def test_receipt_json_round_trip_keeps_the_signature_valid(world) -> None:
    r = world[3]
    back = lc.LossCheckReceipt.from_json(r.to_json())
    assert back == r and back.verify_signature(KEY) and back.commitment() == r.commitment()
    with pytest.raises(ValueError):
        lc.LossCheckReceipt.from_json(r.to_json().replace(lc.RECEIPT_SCHEMA, "other"))


def test_heldout_windows_are_distinct_aligned_and_reproducible() -> None:
    s = lc.heldout_windows(1000, 50, 12, seed=9)
    assert s == lc.heldout_windows(1000, 50, 12, seed=9) and len(set(s)) == 12 and all(x % 50 == 0 and x + 50 <= 1000 for x in s)
    with pytest.raises(ValueError):
        lc.heldout_windows(100, 50, 3, seed=0)


def test_cli_evaluate_then_verify_end_to_end(tmp_path: Path, world, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    from safetensors.torch import save_file

    spec = importlib.util.spec_from_file_location("crowdgpt_loss_check_cli", Path(__file__).resolve().parents[1] / "scripts" / "crowdgpt_loss_check.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    heldout, base, candidate, _ = world
    real = lc.CrowdV1
    monkeypatch.setattr(cli.lc, "CrowdV1", lambda config=TINY: real(TINY))
    save_file({"weights": base.contiguous()}, str(tmp_path / "base.safetensors"))
    save_file({"weights": candidate.contiguous()}, str(tmp_path / "cand.safetensors"))
    (tmp_path / "held.bin").write_bytes(heldout.tobytes())
    (tmp_path / "k").write_bytes(KEY + b"\n")
    (tmp_path / "s").write_bytes(SECRET)
    common = ["--base", str(tmp_path / "base.safetensors"), "--candidate", str(tmp_path / "cand.safetensors"), "--heldout", str(tmp_path / "held.bin")]
    assert cli.main(["evaluate", *common, "--evaluator-id", "t4", "--key-file", str(tmp_path / "k"), "--out", str(tmp_path / "r.json"),
                     "--window", str(W), "--n", str(N), "--seed", "3", "--device", "cpu"]) == 0
    ver = ["verify", *common, "--receipt", str(tmp_path / "r.json"), "--evaluator", f"t4={tmp_path / 'k'}", "--secret-file", str(tmp_path / "s"),
           "--k", "3", "--tol", "1e-4", "--max-regression", "10"]
    assert cli.main(ver) == 0
    assert cli.main([*ver[:-2], "--max-regression", "-10"]) == 1  # the same receipt under a stricter ratchet is held


def test_hf_spec_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("crowdgpt_loss_check_cli2", Path(__file__).resolve().parents[1] / "scripts" / "crowdgpt_loss_check.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.parse_hf_spec("hf://Vxtzq/Crowd-v1@e4fba8b/model_bf16.safetensors") == ("Vxtzq/Crowd-v1", "e4fba8b", "model_bf16.safetensors")
    for bad in ("hf://Vxtzq/Crowd-v1/model.safetensors", "hf://Vxtzq/Crowd-v1@/x", "hf://Vxtzq/Crowd-v1@rev"):
        with pytest.raises(SystemExit):
            cli.parse_hf_spec(bad)
    assert cli.resolve("/local/file.safetensors") == Path("/local/file.safetensors")


def test_a_recompute_that_returns_too_few_or_non_finite_values_is_rejected(world) -> None:
    """The spot check must see every picked window: an empty, short or NaN recompute verifies nothing and must not be accepted."""
    _, _, _, receipt = world
    honest = lc.cpu_recompute_fn(factory, world[1], world[2], world[0], W)
    assert verify(receipt, world).accepted  # positive control: the real recompute is accepted
    assert not verify(receipt, world, recompute=lambda which, starts: []).accepted
    assert not verify(receipt, world, recompute=lambda which, starts: honest(which, starts)[:-1]).accepted
    assert not verify(receipt, world, recompute=lambda which, starts: [float("nan")] * len(starts)).accepted
    assert verify(receipt, world, recompute=lambda which, starts: [9.0] * len(starts)).reason.startswith("REJECTED_SPOT_CHECK")
