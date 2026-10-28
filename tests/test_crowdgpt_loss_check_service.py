"""CrowdGPT loss-check coordinator (backend, CPU) and HiveMind evaluator agent (GPU side), end to end on a tiny model."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hivemind.swarm import crowdgpt_loss_check as lc  # noqa: E402
from hivemind.swarm.crowdgpt_loss_check_service import LossCheckCoordinator, LossCheckJob  # noqa: E402

TINY = {"vocabSize": 97, "dim": 32, "nLayers": 2, "nHeads": 4, "nKvHeads": 2, "headDim": 8, "maxSeqLen": 64, "mlpHidden": 48, "weightTying": True}
W, N = 17, 10
KEYS = {"t4": b"k-t4", "rtx3090": b"k-3090"}


def factory() -> lc.CrowdV1:
    return lc.CrowdV1(TINY)


def flat(seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.cat([p.detach().reshape(-1) for p in factory().parameters()]).to(torch.bfloat16)


@pytest.fixture(scope="module")
def data():
    held = np.random.default_rng(0).integers(0, 97, size=600, dtype=np.uint32)
    base = flat(1)
    cand = (base.float() + 0.01 * torch.randn(base.shape, generator=torch.Generator().manual_seed(2))).to(torch.bfloat16)
    return base, cand, held


def coordinator(tmp_path: Path, data, clock=time.time, **kw) -> LossCheckCoordinator:
    base, cand, held = data
    args = dict(evaluator_keys=KEYS, server_secret=b"secret", state_path=tmp_path / "state.json", loader=lambda job: (base, cand, held),
                model_factory=factory, k=3, tol=1e-4, max_regression_tolerance=10.0, min_windows=8, clock=clock)
    args.update(kw)
    return LossCheckCoordinator(**args)


def open_job(c: LossCheckCoordinator, data, **kw) -> LossCheckJob:
    base, cand, held = data
    return c.open_job(base_url="https://x/base", base_sha256=lc.weights_sha256(base), candidate_url="https://x/cand",
                      candidate_sha256=lc.weights_sha256(cand), heldout_url="https://x/held", heldout_sha256=lc.tokens_sha256(held), window=W,
                      n=N, **kw)


def evaluate(job: LossCheckJob, data, who: str = "t4", **kw) -> lc.LossCheckReceipt:
    base, cand, held = data
    return lc.make_receipt(evaluator_id=who, key=KEYS[who], base=base, candidate=cand, heldout=held, window=job.window, n=job.n, seed=job.seed,
                           model_factory=factory, device="cpu", round_id=job.job_id, **kw)


def test_one_honest_receipt_decides_the_job_and_survives_a_restart(tmp_path: Path, data) -> None:
    c = coordinator(tmp_path, data)
    job = open_job(c, data)
    assert c.handle_get_job("t4") == (200, job.to_wire()) and c.handle_get_job("stranger") == (204, None)
    status, out = c.handle_post_receipt(evaluate(job, data).to_json())
    assert status == 200 and out["recorded"] and out["decision"]["accepted"] and out["decision"]["evaluators"] == ["t4"]
    assert c.handle_get_job("t4") == (204, None)  # decided: no more work
    again = coordinator(tmp_path, data)  # restart from the state file
    assert again.decision(job.job_id)["accepted"] is True
    assert again.handle_post_receipt(evaluate(job, data, "rtx3090").to_json())[0] == 409  # already decided


def test_forged_and_misdirected_receipts_never_count(tmp_path: Path, data) -> None:
    c = coordinator(tmp_path, data, k=2 * N)
    job = open_job(c, data)
    r = evaluate(job, data)
    lie = lc._replace(r, losses_after=tuple(x - 1.0 for x in r.losses_after)).signed(KEYS["t4"])
    status, out = c.handle_post_receipt(lie.to_json())
    assert status == 200 and out["recorded"] is False and out["verdict"]["reason"].startswith("REJECTED_SPOT_CHECK")
    assert c.decision(job.job_id) is None and c.handle_get_job("t4")[0] == 200  # the evaluator may still answer honestly
    assert c.handle_post_receipt(lc._replace(r, round_id="nope").signed(KEYS["t4"]).to_json())[0] == 404
    assert c.handle_post_receipt("{not json")[0] == 400
    other_seed = lc._replace(r, seed=r.seed + 1).signed(KEYS["t4"])
    assert c.handle_post_receipt(other_seed.to_json())[0] == 400
    assert c.handle_post_receipt(r.to_json())[1]["decision"]["accepted"]


def test_authenticate_before_loader_and_no_nan_in_json(tmp_path: Path, data) -> None:
    from fastapi.responses import JSONResponse

    base, cand, held = data
    load_calls = [0]

    def counting_loader(job: LossCheckJob):
        load_calls[0] += 1
        return base, cand, held

    c = coordinator(tmp_path, data, loader=counting_loader)
    job = open_job(c, data)
    honest = evaluate(job, data)

    # 1. Forged signature and unknown evaluator are rejected BEFORE touching the loader
    forged = lc._replace(honest, signature_hex="0" * 64)
    st_forged, out_forged = c.handle_post_receipt(forged.to_json())
    assert st_forged == 200 and out_forged["recorded"] is False
    assert out_forged["verdict"]["reason"] == "REJECTED_BAD_SIGNATURE"
    assert load_calls[0] == 0

    unknown_eval = lc._replace(honest, evaluator_id="unknown-node").signed(b"wrong-key")
    st_unk, out_unk = c.handle_post_receipt(unknown_eval.to_json())
    assert st_unk == 200 and out_unk["recorded"] is False
    assert out_unk["verdict"]["reason"] == "REJECTED_UNKNOWN_EVALUATOR"
    assert load_calls[0] == 0

    # 2. Signed receipt with wrong weights goes through verify_receipt and returns null (None) instead of NaN
    wrong_weights = lc._replace(honest, base_sha256="f" * 64).signed(KEYS["t4"])
    st_wrong, out_wrong = c.handle_post_receipt(wrong_weights.to_json())
    assert st_wrong == 200 and out_wrong["recorded"] is False
    assert out_wrong["verdict"]["reason"] == "REJECTED_WRONG_WEIGHTS"
    assert out_wrong["verdict"]["loss_before"] is None
    assert out_wrong["verdict"]["loss_after"] is None
    assert out_wrong["verdict"]["delta_loss"] is None
    assert load_calls[0] == 1
    # Strict JSON encoding and Starlette/FastAPI JSONResponse must succeed without ValueError
    assert json.dumps(out_wrong, allow_nan=False)
    assert JSONResponse(out_wrong, status_code=st_wrong).status_code == 200


def test_two_evaluators_required_must_agree(tmp_path: Path, data) -> None:
    c = coordinator(tmp_path, data, required_receipts=2)
    job = open_job(c, data)
    first = c.handle_post_receipt(evaluate(job, data, "t4").to_json())[1]
    assert first["recorded"] and "decision" not in first
    assert c.handle_post_receipt(evaluate(job, data, "t4").to_json())[0] == 409  # one vote per evaluator
    second = c.handle_post_receipt(evaluate(job, data, "rtx3090").to_json())[1]
    assert second["decision"]["accepted"] and sorted(second["decision"]["evaluators"]) == ["rtx3090", "t4"]


def test_disagreeing_evaluators_block_the_candidate(tmp_path: Path, data) -> None:
    c = coordinator(tmp_path, data, required_receipts=2, k=1, agreement_tol=1e-4)
    job = open_job(c, data)
    honest = evaluate(job, data, "t4")
    c.handle_post_receipt(honest.to_json())
    r2 = evaluate(job, data, "rtx3090")
    # shift numbers the spot check will not look at: find an index not among this receipt's picks
    for j in range(N):
        cand = lc._replace(r2, losses_before=tuple(x + 0.5 if i == j else x for i, x in enumerate(r2.losses_before))).signed(KEYS["rtx3090"])
        if ("before", j) not in lc.spot_check_picks(cand.commitment(), b"secret", N, 1):
            break
    out = c.handle_post_receipt(cand.to_json())[1]
    assert out["recorded"] and out["decision"]["accepted"] is False and out["decision"]["reason"].startswith("EVALUATORS_DISAGREE")


def test_expired_jobs_are_not_offered_or_accepted(tmp_path: Path, data) -> None:
    now = [1000.0]
    c = coordinator(tmp_path, data, clock=lambda: now[0], job_ttl_s=60)
    job = open_job(c, data)
    r = evaluate(job, data)
    now[0] += 61
    assert c.handle_get_job("t4") == (204, None) and c.handle_post_receipt(r.to_json())[0] == 410


def test_constructor_and_open_job_refuse_unworkable_settings(tmp_path: Path, data) -> None:
    with pytest.raises(ValueError):
        coordinator(tmp_path, data, required_receipts=3)
    c = coordinator(tmp_path, data)
    with pytest.raises(ValueError):
        c.open_job(base_url="u", base_sha256="a", candidate_url="u", candidate_sha256="b", heldout_url="u", heldout_sha256="c", n=4)
    j = open_job(c, data, job_id="fixed")
    with pytest.raises(ValueError):
        open_job(c, data, job_id="fixed")
    assert LossCheckJob.from_wire(j.to_wire()) == j


# ---------------------------------------------------------------- the evaluator agent


def load_agent():
    spec = importlib.util.spec_from_file_location("crowdgpt_loss_check_agent", ROOT / "scripts" / "crowdgpt_loss_check_agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_governor_gate(tmp_path: Path) -> None:
    ag = load_agent()
    f = tmp_path / "g.json"
    assert ag.governor_gate(None, {"training"}, 30)[0] is True
    assert ag.governor_gate(f, {"training"}, 30)[0] is False  # missing file: never use a shared GPU blind
    f.write_text(json.dumps({"state": "training"}))
    assert ag.governor_gate(f, {"training"}, 30)[0] is True
    f.write_text(json.dumps({"state": "yielded"}))
    assert ag.governor_gate(f, {"training"}, 30) == (False, "governor state 'yielded'")
    f.write_text(json.dumps({"state": "training"}))
    old = time.time() - 120
    os.utime(f, (old, old))
    assert ag.governor_gate(f, {"training"}, 30)[0] is False  # stale: the governor may be dead


def agent_args(tmp_path: Path, **kw) -> argparse.Namespace:
    (tmp_path / "key").write_bytes(KEYS["t4"])
    d = dict(server="https://crowd.example", evaluator_id="t4", key_file=str(tmp_path / "key"), device="cpu", dtype="float32", batch=1,
             cache_dir=tmp_path / "cache", governor_state=None, allowed_states=["training"], max_state_age_s=30.0)
    d.update(kw)
    return argparse.Namespace(**d)


def test_agent_runs_a_job_against_the_coordinator(tmp_path: Path, data, monkeypatch: pytest.MonkeyPatch) -> None:
    from safetensors.torch import save_file

    ag = load_agent()
    real = lc.CrowdV1
    monkeypatch.setattr(ag.lc, "CrowdV1", lambda config=TINY: real(TINY))
    base, cand, held = data
    files = {"https://x/base": tmp_path / "b.st", "https://x/cand": tmp_path / "c.st", "https://x/held": tmp_path / "h.bin"}
    save_file({"weights": base.contiguous()}, str(files["https://x/base"]))
    save_file({"weights": cand.contiguous()}, str(files["https://x/cand"]))
    files["https://x/held"].write_bytes(held.tobytes())
    monkeypatch.setattr(ag, "download", lambda url, dest, byte_range="": (dest.parent.mkdir(parents=True, exist_ok=True), dest.write_bytes(files[url].read_bytes()), dest)[2])
    c = coordinator(tmp_path / "srv", data)
    job = open_job(c, data)

    def get(url: str):
        status, body = c.handle_get_job(url.split("evaluator_id=")[1])
        return status, json.dumps(body).encode() if body else b""

    out = ag.run_once(agent_args(tmp_path), get=get, post=lambda url, body: c.handle_post_receipt(body))
    assert out["status"] == "submitted" and out["http"] == 200 and out["response"]["decision"]["accepted"]
    assert ag.run_once(agent_args(tmp_path), get=get, post=lambda u, b: (500, {}))["status"] == "no_job"


def test_agent_refuses_weights_that_do_not_match_the_job(tmp_path: Path, data, monkeypatch: pytest.MonkeyPatch) -> None:
    from safetensors.torch import save_file

    ag = load_agent()
    base, cand, held = data
    swapped = tmp_path / "swapped.st"
    save_file({"weights": cand.contiguous()}, str(swapped))  # served under the base URL
    monkeypatch.setattr(ag, "download", lambda url, dest, byte_range="": (dest.parent.mkdir(parents=True, exist_ok=True), dest.write_bytes(swapped.read_bytes()), dest)[2])
    with pytest.raises(ValueError, match="do not match"):
        ag.fetch_weights("https://x/base", lc.weights_sha256(base), tmp_path / "cache")
    assert not list((tmp_path / "cache").glob("*.safetensors"))  # the bad file is not kept in the cache


def test_agent_waits_when_the_gpu_is_serving_and_yields_mid_job(tmp_path: Path, data, monkeypatch: pytest.MonkeyPatch) -> None:
    ag = load_agent()
    g = tmp_path / "gov.json"
    g.write_text(json.dumps({"state": "yielded"}))
    calls: list[str] = []
    out = ag.run_once(agent_args(tmp_path, governor_state=g), get=lambda u: calls.append(u) or (204, b""), post=None)
    assert out["status"] == "waiting" and calls == []  # serving: the backend is not even asked for work
    m = factory()
    m.load_flat(flat(3))
    flips = iter([False, False, True])
    with pytest.raises(lc.Interrupted, match="stopped after 2 of 5"):
        lc.window_losses(m, data[2], [0, 17, 34, 51, 68], W, stop=lambda: next(flips))
