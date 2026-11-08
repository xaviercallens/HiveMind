"""E16 end to end against the in-process fake Ollama: smoke, resume, idle-GPU abort, unload-in-finally, fail-fast rules, refusals.  CPU only, no real daemon."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

_CUDA_VISIBLE_DEVICES_BEFORE = os.environ.get("CUDA_VISIBLE_DEVICES")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # for the imports below only; restored after them
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import e15_data as data  # noqa: E402
import e15_gate as gate_mod  # noqa: E402
import e16_fake_ollama as fk  # noqa: E402
import run_experiment_E16_gwaya_lite_t4 as R  # noqa: E402

# Restored right after the imports above: setting it at module level hid the GPU from every test module collected
# after this one (IP-179); the autouse fixture below keeps this module's own tests CPU-only at run time.
if _CUDA_VISIBLE_DEVICES_BEFORE is None:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _CUDA_VISIBLE_DEVICES_BEFORE


@pytest.fixture(autouse=True)
def _cpu_only_for_this_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

pytestmark = pytest.mark.skipif(not gate_mod.gate_available(), reason="sandbox unavailable")
P = "E16_gwaya_lite_t4"


def make_fake(**kw):
    tr, ev, _ = data.synthetic_sets()
    f = fk.FakeOllama({p.user_text: p.reference for p in tr + ev}, **kw)
    f.start()
    return f


def run(out, fake, *extra):
    return R.main(["--smoke", "--out", str(out), *extra], _fake=fake)


def stage(out, name):
    return json.loads((Path(out) / f"{P}_{name}.json").read_text())


def test_smoke_end_to_end_subprocess_under_60s(tmp_path):
    before = sorted(p.name for p in (ROOT / "artifacts" / "experiments").glob("E16*"))
    t0 = time.time()
    r = subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "scripts" / "run_experiment_E16_gwaya_lite_t4.py"), "--smoke", "--out", str(tmp_path / "o")], capture_output=True, text=True,
                       timeout=170, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, cwd="/")
    assert time.time() - t0 < 60, "smoke must stay under 60 s"
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    done = [l for l in r.stdout.splitlines() if l.startswith("E16 DONE")]
    assert len(done) == 1 and "stages_failed=0" in done[0] and "run_valid=True" in done[0]
    rep = json.loads((tmp_path / "o" / f"{P}_report.json").read_text())
    assert rep["smoke"] and rep["run_valid"] and set(rep["hypotheses"]) >= {"H16a", "H16b", "H16c", "H16d", "H16e", "H16f_descriptive"}
    assert len(rep["probes"]) == 6 and rep["escalation_ladder"]["n"] == 6 and rep["controls"]["unload"]["pass"] and rep["controls"]["idle_gpu"]["pass"]
    for t in ("1.5b", "3b", "7b"):
        assert {"R4096", "L4096", "paired_L_minus_R", "p2_gate_calibration", "determinism"} <= set(rep["analysis"]["tiers"][t])
    assert "R8192" in rep["analysis"]["tiers"]["7b"]
    after = sorted(p.name for p in (ROOT / "artifacts" / "experiments").glob("E16*"))
    assert before == after  # smoke never writes into artifacts/


def test_resume_skips_everything_and_makes_no_ollama_call(tmp_path):
    f = make_fake()
    try:
        assert run(tmp_path, f) == 0
        n_first = f.n_chat
        assert n_first > 0
        f.n_chat = 0
        f.calls.clear()
        assert run(tmp_path, f, "--resume") == 0
        assert f.n_chat == 0 and f.calls == []
        # accuracy stage file removed: the journals alone restore it without generating again
        (tmp_path / f"{P}_accuracy.json").unlink()
        assert run(tmp_path, f, "--resume") == 0 and f.n_chat == 0
        assert f.loaded == {}
    finally:
        f.stop()


def test_idle_gpu_abort_exit_3_and_no_generation(tmp_path):
    f = make_fake(baseline_mib=2000)
    try:
        assert run(tmp_path, f) == 3
        rec = stage(tmp_path, "preflight")
        assert rec["status"] == "aborted by registered fail-fast rule" and rec["idle_gpu_at_start"]["memory_used_mib"] == 2000
        assert f.n_chat == 0 and f.loaded == {}
        rep = json.loads((tmp_path / f"{P}_report.json").read_text())
        assert rep["run_valid"] is False and rep["controls"]["idle_gpu"]["pass"] is False
    finally:
        f.stop()


def test_resident_model_is_unloaded_first_then_run_proceeds(tmp_path):
    f = make_fake(preloaded=[("qwen2.5-coder:3b", 4096)])
    try:
        assert run(tmp_path, f, "--tiers", "1.5b", "--ctx", "4096") == 0
        assert "qwen2.5-coder:3b" in f.ever_unloaded and stage(tmp_path, "preflight")["idle_gpu_at_start"]["loaded_at_start"] == ["qwen2.5-coder:3b"]
    finally:
        f.stop()


def test_unload_in_finally_on_stage_error(tmp_path):
    """The 5th chat call fails (HTTP 500) while a model is resident: the stage fails (rc 1) and nothing stays loaded."""
    f = make_fake(fail_chat_after=4)
    try:
        rc = run(tmp_path, f, "--tiers", "7b", "--ctx", "4096")
        assert rc == 1 and f.loaded == {}
        assert stage(tmp_path, "probes")["status"] == "failed" and stage(tmp_path, "unload")["pass"] is True
    finally:
        f.stop()


def test_unload_in_finally_on_sigterm_like_exit(tmp_path, monkeypatch):
    f = make_fake()
    try:
        def boom(c):
            c.client.chat("qwen2.5-coder:7b-instruct", [{"role": "user", "content": "x"}], {"num_ctx": 4096, "num_predict": 1})
            assert f.loaded  # a model really is resident when the exit comes
            raise SystemExit(143)

        monkeypatch.setitem(R.STAGE_FUNCS, "probes", boom)
        with pytest.raises(SystemExit) as e:
            run(tmp_path, f, "--tiers", "7b", "--ctx", "4096")
        assert e.value.code == 143 and f.loaded == {}
        assert stage(tmp_path, "unload")["pass"] is True
    finally:
        f.stop()


def test_unload_control_fails_loudly_when_memory_does_not_return(tmp_path):
    f = make_fake(leak_mib=1000)
    try:
        rc = run(tmp_path, f, "--tiers", "1.5b", "--ctx", "4096")
        rep = json.loads((tmp_path / f"{P}_report.json").read_text())
        assert rep["controls"]["unload"]["pass"] is False and rep["run_valid"] is False and f.loaded == {}
        assert rc == 0  # the numbers exist; validity is flagged in the report and in the DONE line
    finally:
        f.stop()


def test_failfast_7b_decode_below_5_tok_s(tmp_path, capsys):
    f = make_fake(tps={"7b": 4.0})
    try:
        rc = run(tmp_path, f, "--tiers", "7b", "--ctx", "4096")
        out = capsys.readouterr().out
        assert rc == 3 and "aborted by registered fail-fast rule" in out and not [l for l in out.splitlines() if l.startswith("E16 DONE")]
        rec = stage(tmp_path, "probes")
        assert rec["status"] == "aborted by registered fail-fast rule" and "5.0 tok/s" in rec["reason"]
        assert not (tmp_path / f"{P}_accuracy.json").exists() and f.loaded == {}
    finally:
        f.stop()


def test_failfast_projected_time_over_max_hours(tmp_path):
    f = make_fake()
    try:
        rc = run(tmp_path, f, "--tiers", "1.5b", "--ctx", "4096", "--max-hours", "0.000001")
        assert rc == 3 and "max-hours" in stage(tmp_path, "accuracy")["reason"] and f.loaded == {}
    finally:
        f.stop()


def test_missing_model_gives_pull_command_and_exit_2(tmp_path, capsys):
    f = make_fake(missing=["qwen2.5-coder:7b-instruct"])
    try:
        rc = run(tmp_path, f)
        assert rc == 2 and "ollama pull qwen2.5-coder:7b-instruct" in capsys.readouterr().err and f.n_chat == 0
    finally:
        f.stop()


def test_refusals(tmp_path, monkeypatch, capsys):
    assert R.main(["--smoke"]) == 2  # smoke needs an explicit --out
    monkeypatch.setattr(R, "PREREG_PATH", tmp_path / "missing_prereg.json")
    assert R.main(["--out", str(tmp_path / "o")]) == 2 and "registration file" in capsys.readouterr().err
    monkeypatch.undo()
    assert R.main(["--out", str(tmp_path / "o2"), "--ollama-url", "http://8.8.8.8:11434"]) == 2
    assert "non-local" in capsys.readouterr().err
    assert R.main(["--smoke", "--out", str(tmp_path / "o3"), "--stages", "bogus"]) == 2


def test_estimate_prints_json(capsys):
    assert R.main(["--estimate"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["problems"] == 257 and set(d["per_tier"]) == {"1.5b", "3b", "7b"} and 0 < d["est_total_hours"] < 3.0


def test_prereg_file_exists_and_matches_the_estimator_constants():
    pr = json.loads((ROOT / "artifacts" / "experiments" / "E16_gwaya_lite_t4_prereg.json").read_text())
    assert "1782" in pr["what_is_measured"]["resource_probes"] and "5514" in pr["what_is_measured"]["resource_probes"]
    assert (R.estimator_mib("1.5b", 4096), R.estimator_mib("3b", 4096), R.estimator_mib("7b", 4096), R.estimator_mib("7b", 8192)) == (1782, 2584, 5290, 5514)
