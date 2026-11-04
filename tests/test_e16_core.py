"""E16 core: Ollama client against the fake server, GPU readers, timing math, estimator numbers, H16a-f logic (incl. boundaries), P2 calibration, ladder, probes."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

_CUDA_VISIBLE_DEVICES_BEFORE = os.environ.get("CUDA_VISIBLE_DEVICES")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # for the imports below only; restored after them
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import e15_data as data  # noqa: E402
import e15_gate as gate_mod  # noqa: E402
import e16_fake_ollama as fk  # noqa: E402
import e16_ollama as oll  # noqa: E402
import run_experiment_E16_gwaya_lite_t4 as R  # noqa: E402
from e15_lib import mcnemar_exact, paired, wilson  # noqa: E402

# Restored right after the imports above: setting it at module level hid the GPU from every test module collected
# after this one (IP-179); the autouse fixture below keeps this module's own tests CPU-only at run time.
if _CUDA_VISIBLE_DEVICES_BEFORE is None:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _CUDA_VISIBLE_DEVICES_BEFORE


@pytest.fixture(autouse=True)
def _cpu_only_for_this_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def make_fake(**kw):
    tr, ev, _ = data.synthetic_sets()
    f = fk.FakeOllama({p.user_text: p.reference for p in tr + ev}, **kw)
    url = f.start()
    f.url = url
    return f, url


@pytest.fixture()
def fake():
    f, _ = make_fake()
    yield f
    f.stop()


def client_for(f, **kw):
    return oll.Client(f.url, timeout=10, retries=kw.pop("retries", 2), sleep=lambda s: None, **kw)


# ---------------------------------------------------------------- client
def test_chat_parsing_and_timing_fields(fake):
    c = client_for(fake)
    tr, ev, _ = data.synthetic_sets()
    p = ev[0]
    r = c.chat("qwen2.5-coder:3b", p.messages, {"num_ctx": 4096, "temperature": 0.0, "seed": 3, "num_predict": 128, "bogus": 1})
    assert r["text"] and set(oll.TIMING_FIELDS) <= set(r) and r["eval_count"] >= 1 and r["eval_duration"] > 0 and r["prompt_eval_count"] > 0
    assert r["load_duration"] > 0  # first call loads the model
    r2 = c.chat("qwen2.5-coder:3b", p.messages, {"num_ctx": 4096, "temperature": 0.0, "seed": 3})
    assert r2["load_duration"] == 0 and r2["total_duration"] >= r2["eval_duration"] + r2["prompt_eval_duration"]
    assert fake.calls[0]["ctx"] == 4096


def test_ps_tags_version(fake):
    c = client_for(fake)
    assert "qwen2.5-coder:7b-instruct" in c.tags() and c.version()
    assert c.ps() == []
    c.chat("qwen2.5-coder:7b-instruct", [{"role": "user", "content": "hi"}], {"num_ctx": 4096, "num_predict": 1})
    (m,) = c.ps()
    assert m["name"] == "qwen2.5-coder:7b-instruct" and m["gpu_pct"] == 100.0 and m["size_vram"] == m["size"] and m["context_length"] == 4096


def test_unload_verified_through_ps(fake):
    c = client_for(fake)
    c.chat("qwen2.5-coder:1.5b", [{"role": "user", "content": "hi"}], {"num_ctx": 2048, "num_predict": 1})
    assert c.unload("qwen2.5-coder:1.5b", wait_s=1.0) is True and c.ps() == []
    c.chat("qwen2.5-coder:1.5b", [{"role": "user", "content": "hi"}], {"num_ctx": 2048, "num_predict": 1})
    res = c.unload_all(wait_s=1.0)
    assert res["was_loaded"] == ["qwen2.5-coder:1.5b"] and res["still_loaded"] == []


def test_unload_stuck_model_is_reported_not_hidden():
    f, url = make_fake(stuck=["qwen2.5-coder:3b"])
    try:
        c = oll.Client(url, timeout=10, retries=0, sleep=lambda s: None)
        c.chat("qwen2.5-coder:3b", [{"role": "user", "content": "hi"}], {"num_ctx": 2048, "num_predict": 1})
        assert c.unload("qwen2.5-coder:3b", wait_s=0.05, poll_s=0.01) is False
        assert c.unload_all(wait_s=0.05)["still_loaded"] == ["qwen2.5-coder:3b"]
    finally:
        f.stop()


def test_localhost_only():
    for bad in ("http://example.com:11434", "http://10.0.0.5:11434", "https://127.0.0.1:11434", "http://127.0.0.1.evil.com"):
        with pytest.raises(oll.OllamaError):
            oll.Client(bad)
    oll.Client("http://localhost:11434")
    oll.Client("http://127.0.0.1:1234/")


def test_retries_max_two_then_clear_error():
    f, url = make_fake(fail_chat_after=0)
    sleeps = []
    try:
        c = oll.Client(url, timeout=5, retries=5, sleep=sleeps.append)  # asked for 5, capped at 2
        with pytest.raises(oll.OllamaError) as e:
            c.chat("qwen2.5-coder:3b", [{"role": "user", "content": "x"}], {"num_ctx": 2048})
        assert "3 attempts" in str(e.value) and len(sleeps) == 2 and f.n_chat == 3
    finally:
        f.stop()


def test_4xx_not_retried_and_clear_message():
    f, url = make_fake()
    try:
        c = oll.Client(url, timeout=5, retries=2, sleep=lambda s: None)
        with pytest.raises(oll.OllamaError) as e:
            c.chat("nope:1b", [{"role": "user", "content": "x"}], {})
        assert "404" in str(e.value) and "not found" in str(e.value)
    finally:
        f.stop()


def test_connection_refused_is_a_clear_error():
    c = oll.Client("http://127.0.0.1:9", timeout=2, retries=1, sleep=lambda s: None)
    with pytest.raises(oll.OllamaError) as e:
        c.tags()
    assert "is the Ollama daemon running" in str(e.value)


def test_gpu_readers_parse_nvidia_smi(monkeypatch):
    outs = {"memory.used": "  1234\n", "compute-apps": "4047, /usr/bin/python3, 4044\n999, /usr/local/bin/ollama, 5000\n", "name,memory.total": "Tesla T4, 15360, 535.1\n"}

    def fake_run(cmd, **kw):
        key = next(k for k in outs if any(k in a for a in cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout=outs[key], stderr="")

    monkeypatch.setattr(oll.subprocess, "run", fake_run)
    assert oll.gpu_memory_used_mib() == 1234
    procs = oll.gpu_compute_processes()
    assert [p["pid"] for p in procs] == [4047, 999] and procs[0]["used_memory_mib"] == 4044
    assert not oll.is_ollama_process(procs[0]) and oll.is_ollama_process(procs[1])
    assert oll.gpu_info()["memory_total_mib"] == 15360


def test_gpu_memory_error_is_clear(monkeypatch):
    monkeypatch.setattr(oll.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 9, stdout="", stderr="boom"))
    monkeypatch.setitem(sys.modules, "pynvml", None)
    with pytest.raises(oll.OllamaError):
        oll.gpu_memory_used_mib()


# ---------------------------------------------------------------- timing math / stats reuse
def test_timing_math():
    assert R.tok_per_s(100, 2_000_000_000) == 50.0 and R.tok_per_s(0, 5) is None and R.tok_per_s(5, 0) is None
    assert R.model_seconds({"total_duration": 5_000_000_000, "load_duration": 2_000_000_000}) == 3.0
    assert R.model_seconds({"total_duration": 1, "load_duration": 5}) == 0.0
    assert R.median([1.0, None, 3.0, 2.0]) == 2.0 and R.median([None]) is None
    assert R.seed_for("a", 0) == R.seed_for("a", 0) and R.seed_for("a", 0) != R.seed_for("a", 1) and 0 <= R.seed_for(5, 2) < 2 ** 31


def test_statistics_are_the_e15_functions():
    assert R.L.wilson is wilson and R.L.paired is paired and R.L.mcnemar_exact is mcnemar_exact
    assert mcnemar_exact(0, 5) == pytest.approx(2 / 32) and mcnemar_exact(0, 0) == 1.0
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.32
    pr = paired([True, False, False, True], [True, True, True, False])
    assert pr["b_only"] == 2 and pr["a_only"] == 1 and pr["diff"] == 0.25


def test_estimator_numbers_from_the_prereg():
    assert [R.estimator_mib(t, 4096) for t in ("1.5b", "3b", "7b")] == [1782, 2584, 5290]
    assert R.estimator_mib("7b", 8192) == 5514


def test_fill_prompt_is_deterministic_and_nonce_first():
    a, b = R.build_fill_prompt(["alpha beta", "gamma"], 3000, 1), R.build_fill_prompt(["alpha beta", "gamma"], 3000, 1)
    assert a == b and len(a) == 3000 and a.startswith("[fill 1]") and a.endswith("OK.")
    assert R.build_fill_prompt(["x"], 3000, 2)[:8] != a[:8]


# ---------------------------------------------------------------- hypotheses
def P(tier, ctx, measured, tps=None):
    return {"tier": tier, "ctx": ctx, "measured_mib": measured, "estimator_mib": R.estimator_mib(tier, ctx), "decode_tps_median": tps}


def test_h16a_boundary():
    assert R.h16a([P("7b", 4096, 6656)])["4096"]["status"] == "supported"
    assert R.h16a([P("7b", 4096, 6657)])["4096"]["status"] == "refuted"
    h = R.h16a([P("7b", 4096, 5000), P("7b", 8192, 7000)])
    assert h["4096"]["status"] == "supported" and h["8192"]["status"] == "refuted"
    assert R.h16a([P("3b", 4096, 100)])["4096"]["status"] == "not_evaluated"


def test_h16b_band_boundaries():
    row = lambda m: R.h16b([{"tier": "3b", "ctx": 4096, "measured_mib": m, "estimator_mib": 1000}])["status"]  # noqa: E731
    assert (row(800), row(1250), row(799), row(1251)) == ("adequate", "adequate", "discrepancy", "discrepancy")
    assert R.h16b([])["status"] == "not_evaluated"
    d = R.h16b([P("1.5b", 4096, 2500), P("7b", 4096, 5290)])
    assert d["status"] == "discrepancy" and d["discrepancies"][0]["tier"] == "1.5b" and d["discrepancies"][0]["ratio"] > 1.25


def test_h16c_h16d_logic():
    def pr(a_only, b_only, n=100):
        return {"n": n, "a_only": a_only, "b_only": b_only, "mcnemar_p": mcnemar_exact(a_only, b_only)}

    assert R.h16c(pr(0, 0)) == "supported" and R.h16c(pr(3, 3)) == "supported" and R.h16c(pr(5, 2)) == "inconclusive" and R.h16c(pr(2, 5)) == "supported"
    assert R.h16c(pr(12, 1)) == "refuted"
    assert R.h16d(pr(0, 0)) == "refuted" and R.h16d(pr(5, 2)) == "refuted"
    assert R.h16d(pr(1, 16)) == "supported"  # +15 points, p tiny
    assert R.h16d(pr(5, 10)) == "inconclusive"  # +5.0 points but p = 0.30
    assert R.h16d(pr(0, 4)) == "inconclusive"  # +4 points
    assert R.h16d(pr(0, 5)) == "inconclusive" and R.h16d(pr(0, 6)) == "supported"  # +5.0 points but p = 0.0625 / +6 with p = 0.031
    assert R.h16d(pr(0, 12, n=257)) == "inconclusive" and R.h16d(pr(0, 13, n=257)) == "supported"  # 12/257 = 4.67 points, 13/257 = 5.06 points


def test_h16e_boundary_and_h16f():
    assert R.h16e(80, 100) == "supported" and R.h16e(79, 100) == "refuted" and R.h16e(4, 5) == "supported" and R.h16e(0, 0) == "inconclusive"
    assert R.h16f([P("7b", 4096, 1, tps=20.0)])["status"] == "met" and R.h16f([P("7b", 4096, 1, tps=19.99)])["status"] == "not met"
    assert R.h16f([P("3b", 4096, 1, tps=50.0)])["status"] == "not_evaluated"


def test_gate_calibration_hand_example():
    def cand(v, f):
        return {"verified": v, "functional": f}

    recs = [{"candidates": [cand(True, True)], "verified_visible": True, "functional": True},
            {"candidates": [cand(False, False), cand(True, False)], "verified_visible": True, "functional": False},
            {"candidates": [cand(False, True), cand(False, False), cand(True, True)], "verified_visible": True, "functional": True},
            {"candidates": [cand(False, False)], "verified_visible": False, "functional": False}]
    g = R.gate_calibration(recs)
    assert (g["tp"], g["fp"], g["fn"], g["tn"]) == (2, 1, 1, 3) and g["precision"] == pytest.approx(2 / 3) and g["recall"] == pytest.approx(2 / 3)
    assert g["status_H16e"] == "refuted" and g["chosen_verified_problems"] == 3 and g["chosen_precision"] == pytest.approx(2 / 3)


def test_ladder_hand_example():
    def rec(v, f, s):
        return {"verified_visible": v, "functional": f, "model_s": s}

    l1 = {"a": rec(True, True, 1.0), "b": rec(False, False, 1.0), "c": rec(False, False, 1.0), "d": rec(True, False, 1.0)}
    l3 = {"a": rec(True, True, 2.0), "b": rec(True, True, 2.0), "c": rec(False, False, 2.0), "d": rec(True, True, 2.0)}
    l7 = {"a": rec(True, True, 4.0), "b": rec(True, True, 4.0), "c": rec(True, True, 4.0), "d": rec(True, True, 4.0)}
    r = R.ladder(l1, l3, l7)
    # a: 1.5B ok (1s); b: 1.5B fails -> 3B ok (1+2); c: 1.5B, 3B fail -> 7B ok (1+2+4); d: 1.5B verified but wrong (1s, stays wrong)
    assert r["escalated_to_3b"] == 2 and r["escalated_to_7b"] == 1
    assert r["ladder_passes"] == 3 and r["seven_b_only_passes"] == 4
    assert r["ladder_model_seconds"] == pytest.approx(1 + 3 + 7 + 1) and r["seven_b_only_model_seconds"] == 16.0
    assert r["ladder_over_7b_only_time"] == pytest.approx(12 / 16) and r["escalation_rate_3b"] == 0.5
    with pytest.raises(AssertionError):
        R.ladder(l1, {"a": l3["a"]}, l7)


# ---------------------------------------------------------------- probes with the fake memory reader
def ctx_for(tmp_path, fakesrv, url, extra=()):
    a = R.parse_args(["--smoke", "--out", str(tmp_path), *extra])
    cl = oll.Client(url, timeout=10, retries=0, sleep=lambda s: None)
    c = R.Ctx(a, cl, fakesrv.gpu_read, fakesrv.procs, fast=True)
    c.baseline_start = fakesrv.baseline
    return c


def test_probe_one_measures_delta_fill_and_returns_to_baseline(tmp_path):
    f, url = make_fake()
    try:
        c = ctx_for(tmp_path, f, url)
        r = R.probe_one(c, "7b", 4096)
        assert r["baseline_mib"] == 400 and r["delta_after_load_mib"] == r["mem_after_load_mib"] - 400
        assert r["measured_mib"] == max(r["delta_after_load_mib"], r["delta_after_fill_mib"]) and r["delta_after_fill_mib"] > r["delta_after_load_mib"]
        assert r["estimator_mib"] == 5290 and 0.8 <= r["ratio_measured_over_estimate"] <= 1.25
        assert r["fill"]["in_band"] and 0.80 <= r["fill"]["fraction_of_ctx"] <= 0.90
        assert r["gpu_offload_ok"] and r["gpu_pct"] == 100.0 and r["n_probe_prompts"] == 12 and len(r["decode_tps_all"]) <= 12
        assert r["decode_tps_median"] == pytest.approx(35.0, rel=1e-3) and r["prompt_eval_tps_median"] > 0
        assert r["unload_verified"] and r["returned_within_tol"] and r["clean"] and f.loaded == {}
    finally:
        f.stop()


def test_probe_flags_cpu_offload_and_unload_leak(tmp_path):
    f, url = make_fake(cpu_offload=True, leak_mib=900)
    try:
        r = R.probe_one(ctx_for(tmp_path, f, url), "3b", 4096)
        assert r["gpu_offload_ok"] is False and r["gpu_pct"] < 99.9
        assert r["returned_within_tol"] is False and r["mem_after_unload_mib"] >= 400 + 900
    finally:
        f.stop()


def test_idle_check_unloads_resident_model_and_passes(tmp_path):
    f, url = make_fake(preloaded=[("qwen2.5-coder:7b-instruct", 4096)])
    try:
        rec = R.idle_check(ctx_for(tmp_path, f, url))
        assert rec["loaded_at_start"] == ["qwen2.5-coder:7b-instruct"] and "qwen2.5-coder:7b-instruct" in f.ever_unloaded and rec["pass"] and rec["memory_used_mib"] == 400
    finally:
        f.stop()


def test_idle_check_fails_when_memory_is_busy(tmp_path):
    f, url = make_fake(baseline_mib=4047)
    try:
        rec = R.idle_check(ctx_for(tmp_path, f, url))
        assert rec["pass"] is False and rec["memory_used_mib"] == 4047
    finally:
        f.stop()


def test_final_unload_reports_failure_when_memory_does_not_return(tmp_path):
    f, url = make_fake(leak_mib=800)
    try:
        c = ctx_for(tmp_path, f, url)
        c.client.chat("qwen2.5-coder:3b", [{"role": "user", "content": "x"}], {"num_ctx": 2048, "num_predict": 1})
        rec = R.final_unload(c)
        assert rec["pass"] is False and rec["still_loaded"] == [] and rec["memory_used_mib"] >= 1200
    finally:
        f.stop()


def test_loop_one_fallback_and_lazy_candidates(tmp_path):
    """A model that never solves: loop = greedy + 2 samples + 3 repairs, falls back to the first greedy candidate, unverified.  A model that always solves: 1 call."""
    if not gate_mod.gate_available():
        pytest.skip("sandbox unavailable")
    f, url = make_fake(ability={"3b": 0.0}, repair_ability=0.0, wrong="stub")
    try:
        c = ctx_for(tmp_path / "a", f, url)
        p = c.ev[0]
        res = R.gen_call(c, "3b", p.messages, 4096, p.task_id, 0, False)
        cand, _ = R.eval_candidate(c, p, res, 0, "greedy")
        rec = R.run_loop_one(c, "3b", 4096, p, R.raw_record("3b", 4096, p, cand))
        assert rec["candidates_tried"] == 6 and rec["repair_rounds"] == 3 and not rec["verified_visible"] and rec["source"] == "fallback_first_greedy" and rec["text"] == cand["text"]
        assert len(rec["candidates"]) == 6 and abs(rec["model_s"] - sum(x["model_s"] for x in rec["candidates"])) < 1e-9
    finally:
        f.stop()
    f, url = make_fake(ability={"3b": 1.0})
    try:
        c = ctx_for(tmp_path / "b", f, url)
        p = c.ev[0]
        res = R.gen_call(c, "3b", p.messages, 4096, p.task_id, 0, False)
        cand, _ = R.eval_candidate(c, p, res, 0, "greedy")
        rec = R.run_loop_one(c, "3b", 4096, p, R.raw_record("3b", 4096, p, cand))
        assert rec["candidates_tried"] == 1 and rec["verified_visible"] and rec["functional"] and rec["extra_model_s"] == 0 and len(f.calls) == 1
    finally:
        f.stop()
