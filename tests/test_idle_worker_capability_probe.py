# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Truthful GPU probe and BLOCKED verdict for the volunteer idle worker (`scripts/hivemind_boinc_idle_worker.py`).

External I/O (nvidia-smi subprocess, urllib) is faked with injected callables; the parser and the idle gate are pure
and covered with Hypothesis. A host without a GPU must produce BLOCKED, never an invented card or a fabricated QLoRA delta.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hivemind_boinc_idle_worker.py"

_spec = importlib.util.spec_from_file_location("hivemind_boinc_idle_worker_probe_tests", SCRIPT)
worker = importlib.util.module_from_spec(_spec)
sys.modules["hivemind_boinc_idle_worker_probe_tests"] = worker
_spec.loader.exec_module(worker)  # type: ignore[union-attr]

_REAL_PROBE_GPU = worker.probe_gpu  # captured before any monkeypatch replaces worker.probe_gpu
T4_LINE = "Tesla T4, 5, 52, 15360, 612, 28.4, 550.54.15"
INVENTED = (8192.0, 610.88, 9.8)


def _cp(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["nvidia-smi"], returncode=returncode, stdout=stdout, stderr=stderr)


def _ready_probe() -> Any:
    return _REAL_PROBE_GPU(run=lambda *a, **k: _cp(T4_LINE + "\n"), which=lambda _n: "/usr/bin/nvidia-smi")


def _blocked_probe() -> Any:
    return _REAL_PROBE_GPU(run=lambda *a, **k: _cp(), which=lambda _n: None)


# ---------------------------------------------------------------- parser: positive / negative controls


def test_parse_real_t4_line_positive_control() -> None:
    readings = worker.parse_nvidia_smi_csv(T4_LINE)
    assert len(readings) == 1
    g = readings[0]
    assert g.name == "Tesla T4"
    assert g.total_mib == 15360.0
    assert g.used_mib == 612.0
    assert g.utilization_pct == 5.0
    assert g.temp_c == 52.0
    assert g.power_w == 28.4
    assert g.driver_version == "550.54.15"


def test_parse_two_gpus_and_best_picks_larger_total() -> None:
    out = "GeForce GTX 1650, 1, 40, 4096, 100, 10.0, 550.1\n" + T4_LINE + "\n"
    readings = worker.parse_nvidia_smi_csv(out)
    assert [g.name for g in readings] == ["GeForce GTX 1650", "Tesla T4"]
    probe = worker.GpuProbe(status="READY", reason="ok", gpus=readings, detail="")
    assert probe.best().name == "Tesla T4"
    assert probe.best().total_mib == 15360.0


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "NVIDIA-SMI has failed because it could not communicate with the NVIDIA driver",
        "Tesla T4, 5, 52, 0, 0, 1, 1",
        "garbage,,,",
    ],
)
def test_parse_negative_controls_return_no_reading(bad: str) -> None:
    readings = worker.parse_nvidia_smi_csv(bad)
    assert len(readings) == 0
    for g in readings:
        assert g.total_mib not in INVENTED
        assert "RTX 2070" not in g.name


def test_parse_na_fields_become_none_not_literals() -> None:
    (g,) = worker.parse_nvidia_smi_csv("Tesla T4, [N/A], N/A, 15360, [N/A], [N/A], [N/A]")
    assert g.total_mib == 15360.0
    assert g.utilization_pct is None
    assert g.temp_c is None
    assert g.used_mib is None
    assert g.power_w is None
    assert g.driver_version is None


_NAME = st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), max_codepoint=127), min_size=1, max_size=12)
_OPT = st.one_of(st.none(), st.floats(min_value=0, max_value=1e4, allow_nan=False, allow_infinity=False))


@settings(max_examples=100, deadline=None)
@given(
    rows=st.lists(
        st.tuples(_NAME, _OPT, _OPT, st.floats(min_value=1, max_value=1e6), _OPT, _OPT),
        min_size=1,
        max_size=5,
    )
)
def test_parse_roundtrip_property(rows: list[tuple[Any, ...]]) -> None:
    def fmt(v: float | None) -> str:
        return "[N/A]" if v is None else repr(v)

    lines = [f"{n}, {fmt(u)}, {fmt(t)}, {tot!r}, {fmt(used)}, {fmt(pw)}, 550.1" for n, u, t, tot, used, pw in rows]
    parsed = worker.parse_nvidia_smi_csv("\n".join(lines))
    assert len(parsed) == len(rows)
    for g, (n, u, t, tot, used, pw) in zip(parsed, rows):
        assert g.name == n.strip()
        assert g.total_mib == tot
        assert g.utilization_pct == u
        assert g.temp_c == t
        assert g.used_mib == used
        assert g.power_w == pw


@settings(max_examples=200, deadline=None)
@given(text=st.text(max_size=300))
def test_parse_arbitrary_text_never_raises_and_only_real_rows(text: str) -> None:
    for g in worker.parse_nvidia_smi_csv(text):
        assert g.total_mib > 0
        assert g.name != ""


# ---------------------------------------------------------------- idle gate


@settings(max_examples=200, deadline=None)
@given(
    idle_s=st.floats(min_value=0, max_value=1e5),
    util=st.one_of(st.none(), st.floats(min_value=0, max_value=100)),
    temp=st.one_of(st.none(), st.floats(min_value=0, max_value=120)),
    thr=st.floats(min_value=0, max_value=100),
    max_load=st.floats(min_value=0, max_value=100),
    max_temp=st.floats(min_value=0, max_value=120),
)
def test_idle_gate_matches_definition_and_unknown_is_never_idle(
    idle_s: float, util: float | None, temp: float | None, thr: float, max_load: float, max_temp: float
) -> None:
    gpu = worker.GpuReading("Tesla T4", 15360.0, 0.0, util, temp, None, None)
    got = worker.gpu_is_idle_for_boinc(idle_s, gpu, thr, max_load, max_temp)
    if util is None or temp is None:
        assert got is False
    else:
        assert got == (idle_s >= thr and util < max_load and temp < max_temp)


def test_idle_gate_concrete_cases() -> None:
    def g(util: float | None, temp: float | None) -> Any:
        return worker.GpuReading("Tesla T4", 15360.0, 0.0, util, temp, None, None)

    assert worker.gpu_is_idle_for_boinc(10.0, g(5.0, 50.0), 4.0, 45.0) is True
    assert worker.gpu_is_idle_for_boinc(0.2, g(5.0, 50.0), 4.0, 45.0) is False
    assert worker.gpu_is_idle_for_boinc(10.0, g(90.0, 50.0), 4.0, 45.0) is False
    assert worker.gpu_is_idle_for_boinc(10.0, g(5.0, 90.0), 4.0, 45.0) is False
    assert worker.gpu_is_idle_for_boinc(10.0, g(None, 50.0), 4.0, 45.0) is False


# ---------------------------------------------------------------- probe_gpu with injected fakes


def test_probe_missing_binary_never_runs_command() -> None:
    calls: list[Any] = []

    def run(*a: Any, **k: Any) -> Any:
        calls.append(a)
        return _cp(T4_LINE)

    probe = worker.probe_gpu(run=run, which=lambda _n: None)
    assert probe.status == "BLOCKED"
    assert probe.reason == "nvidia_smi_missing"
    assert probe.best() is None
    assert len(calls) == 0


def test_probe_nonzero_returncode_keeps_stderr_detail() -> None:
    err = "NVIDIA-SMI has failed because it could not communicate with the NVIDIA driver. " + "x" * 400
    probe = worker.probe_gpu(run=lambda *a, **k: _cp(stderr=err, returncode=9), which=lambda _n: "/bin/nvidia-smi")
    assert probe.status == "BLOCKED"
    assert probe.reason == "nvidia_smi_failed"
    assert "could not communicate" in probe.detail
    assert len(probe.detail) <= 200


def test_probe_timeout_maps_to_timeout_reason() -> None:
    def run(*a: Any, **k: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5.0)

    probe = worker.probe_gpu(run=run, which=lambda _n: "/bin/nvidia-smi")
    assert (probe.status, probe.reason) == ("BLOCKED", "nvidia_smi_timeout")


def test_probe_oserror_maps_to_failed_reason() -> None:
    def run(*a: Any, **k: Any) -> Any:
        raise PermissionError("denied")

    probe = worker.probe_gpu(run=run, which=lambda _n: "/bin/nvidia-smi")
    assert (probe.status, probe.reason) == ("BLOCKED", "nvidia_smi_failed")
    assert "denied" in probe.detail


def test_probe_empty_stdout_is_no_gpu_rows() -> None:
    probe = worker.probe_gpu(run=lambda *a, **k: _cp(""), which=lambda _n: "/bin/nvidia-smi")
    assert (probe.status, probe.reason) == ("BLOCKED", "no_gpu_rows")


def test_probe_junk_stdout_is_unparseable() -> None:
    probe = worker.probe_gpu(run=lambda *a, **k: _cp("this is not csv\n"), which=lambda _n: "/bin/nvidia-smi")
    assert (probe.status, probe.reason) == ("BLOCKED", "unparseable_output")
    assert probe.gpus == ()


def test_probe_ready_and_argv_is_the_documented_query() -> None:
    seen: list[Any] = []

    def run(argv: Any, **k: Any) -> Any:
        seen.append(argv)
        return _cp(T4_LINE + "\n")

    probe = worker.probe_gpu(run=run, which=lambda _n: "/usr/bin/nvidia-smi")
    assert probe.status == "READY"
    assert probe.reason == "ok"
    assert probe.detail == ""
    assert probe.best().total_mib == 15360.0
    assert len(seen) == 1
    assert tuple(seen[0]) == worker.NVIDIA_SMI_QUERY
    assert worker.NVIDIA_SMI_QUERY[0] == "nvidia-smi"


def test_query_nvidia_gpu_status_unavailable_when_blocked_and_real_when_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker.shutil, "which", lambda _n: None)
    blocked = worker.query_nvidia_gpu_status()
    assert blocked["available"] is False and blocked["gpu_model"] == "unknown"
    assert blocked["utilization_pct"] >= 45.0 and blocked["temp_c"] >= 78.0  # busy and hot, so a caller that ignores the flag still does not start

    monkeypatch.setattr(worker.shutil, "which", lambda _n: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(worker.subprocess, "run", lambda *a, **k: _cp(T4_LINE + "\n"))
    gpu = worker.query_nvidia_gpu_status()
    assert gpu["available"] is True
    assert gpu["gpu_model"] == "Tesla T4"
    assert gpu["total_mib"] == 15360.0
    assert gpu["driver_version"] == "550.54.15"


# ---------------------------------------------------------------- cycle flows


def _cuda_fold_result(target_planet: str = "K2-18 b", n_pts: int = 16384, gpu: Any = None, cuda_lib: Any = None) -> dict[str, Any]:
    return {"backend": worker.CUDA_BACKEND, "cuda_device": gpu.name, "cuda_total_vram_mib": int(gpu.total_mib), "n_points_folded": n_pts, "vram_allocated_bytes": n_pts * 8,
            "kernel_exec_ms": 0.3, "recovered_transit_depth_ppm": 2789.0, "orbital_period_days": 32.9, "result_sha256": "c" * 64}


def _forbid(name: str) -> Callable[..., Any]:
    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError(f"{name} must not be called")

    return boom


def test_cycle_blocked_flow_only_pauses_and_never_folds_leases_or_submits(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(worker, "probe_gpu", lambda *a, **k: _blocked_probe())
    monkeypatch.setattr(worker, "set_boinc_client_gpu_mode", lambda enable: calls.append(enable) or "boinccmd_gpu_never")
    monkeypatch.setattr(worker, "execute_cuda_ptx_bls_fold", _forbid("execute_cuda_ptx_bls_fold"))
    monkeypatch.setattr(worker.urllib.request, "urlopen", _forbid("urlopen"))

    rep = worker.run_one_idle_gpu_boinc_cycle(force_fold=True, dry_run=True)
    assert rep["status"] == "BLOCKED"
    assert rep["reason"] == "nvidia_smi_missing"
    assert calls == [False]  # an unreadable GPU pauses BOINC (safe for a player), it never resumes it
    assert rep["boinc_driver"] == "boinccmd_gpu_never" and rep["boinc_touched"] is True
    assert rep["lease_attempted"] is False
    assert rep["submit_attempted"] is False
    assert rep["boinc_active"] is False and "cuda_fold" not in rep

    assert worker.main(["--dry-run"]) == 2
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "BLOCKED"
    assert worker.main(["--once"]) == 2
    assert calls == [False, False, False]


def test_cycle_ready_idle_folds_but_never_leases_or_submits(monkeypatch: pytest.MonkeyPatch) -> None:
    urls: list[str] = []

    def fake_urlopen(req: Any, timeout: float = 0.0) -> Any:
        url = req.full_url if hasattr(req, "full_url") else str(req)
        urls.append(url)
        assert not url.endswith("/workunit/request"), url
        assert not url.endswith("/workunit/submit"), url
        raise OSError("offline test double")

    monkeypatch.setattr(worker, "probe_gpu", lambda *a, **k: _ready_probe())
    monkeypatch.setattr(worker, "get_user_idle_seconds", lambda *a, **k: 30.0)
    monkeypatch.setattr(worker, "set_boinc_client_gpu_mode", lambda enable: "test_driver")
    monkeypatch.setattr(worker.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(worker, "execute_cuda_ptx_bls_fold", _cuda_fold_result)  # the CUDA driver is the external I/O; truthfulness is covered in test_idle_worker_cuda_truthfulness.py

    rep = worker.run_one_idle_gpu_boinc_cycle(commander="Cmdr_Test", idle_threshold_seconds=4.0)
    assert rep["status"] == "READY"
    assert rep["boinc_active"] is True
    assert rep["cuda_fold"]["n_points_folded"] == 16384
    assert rep["cuda_fold"]["backend"] == worker.CUDA_BACKEND and rep["cuda_fold"]["cuda_device"] == "Tesla T4"
    assert rep["qlora"] == {"status": "BLOCKED", "reason": "no_trainer_wired"}
    assert rep["gpu"]["gpu_model"] == "Tesla T4"
    assert rep["lease_attempted"] is False
    assert rep["submit_attempted"] is False
    assert len(urls) >= 1
    assert all(u.startswith("http://127.0.0.1:8095/") for u in urls)
    assert "qlora_outer_delta_summary" not in rep


def test_cycle_ready_but_user_active_pauses_with_zero_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker, "probe_gpu", lambda *a, **k: _ready_probe())
    monkeypatch.setattr(worker, "get_user_idle_seconds", lambda *a, **k: 0.1)
    monkeypatch.setattr(worker, "set_boinc_client_gpu_mode", lambda enable: "test_driver")
    monkeypatch.setattr(worker.urllib.request, "urlopen", _forbid("urlopen"))

    rep = worker.run_one_idle_gpu_boinc_cycle(idle_threshold_seconds=4.0)
    assert rep["boinc_active"] is False
    assert rep["cuda_fold"]["vram_allocated_bytes"] == 0
    assert rep["cuda_fold"]["n_points_folded"] == 0
    assert rep["qlora"] == {"status": "BLOCKED", "reason": "no_trainer_wired"}


def test_force_fold_bypasses_idle_timer_but_not_a_busy_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    busy = _REAL_PROBE_GPU(
        run=lambda *a, **k: _cp("Tesla T4, 97, 60, 15360, 9000, 70.0, 550.54.15\n"), which=lambda _n: "/bin/nvidia-smi"
    )
    monkeypatch.setattr(worker, "probe_gpu", lambda *a, **k: busy)
    monkeypatch.setattr(worker, "get_user_idle_seconds", lambda *a, **k: 0.1)
    monkeypatch.setattr(worker, "set_boinc_client_gpu_mode", lambda enable: "test_driver")
    monkeypatch.setattr(worker.urllib.request, "urlopen", _forbid("urlopen"))
    rep = worker.run_one_idle_gpu_boinc_cycle(force_fold=True)
    assert rep["boinc_active"] is False

    monkeypatch.setattr(worker, "probe_gpu", lambda *a, **k: _ready_probe())
    monkeypatch.setattr(worker, "execute_cuda_ptx_bls_fold", _cuda_fold_result)
    monkeypatch.setattr(worker.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    rep2 = worker.run_one_idle_gpu_boinc_cycle(force_fold=True, dry_run=True)
    assert rep2["boinc_active"] is True
    assert rep2["dry_run"] is True
    assert rep2["dry_run_passed"] is True


class _StopLoopError(Exception):
    """Raised by the fake sleep to break out of the worker loop."""


def test_loop_mode_prints_per_transition_and_backs_off_60s_when_blocked(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    blocked = {"status": "BLOCKED", "reason": "nvidia_smi_missing", "boinc_active": False}
    ready = {"status": "READY", "boinc_active": True}
    seq = iter([blocked, blocked, blocked, ready, ready])
    sleeps: list[float] = []

    def fake_cycle(**kwargs: Any) -> Any:
        return next(seq)

    def fake_sleep(s: float) -> None:
        sleeps.append(s)
        if len(sleeps) >= 5:
            raise _StopLoopError

    monkeypatch.setattr(worker, "run_one_idle_gpu_boinc_cycle", fake_cycle)
    monkeypatch.setattr(worker.time, "sleep", fake_sleep)
    with pytest.raises(_StopLoopError):
        worker.main([])
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["status"] == "BLOCKED"
    assert json.loads(lines[1])["status"] == "READY"
    assert sleeps == [60.0, 60.0, 60.0, 5.0, 5.0]


def test_script_loads_via_importlib_without_sys_modules_registration() -> None:
    """The gateway and the game cockpit exec the script without registering it; dataclasses must still build."""
    name = "hivemind_boinc_idle_worker_unregistered_probe"
    assert name not in sys.modules
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    assert name not in sys.modules
    assert mod.GpuReading("Tesla T4", 15360.0, None, None, None, None, None).total_mib == 15360.0
    assert mod.probe_gpu(run=lambda *a, **k: _cp(T4_LINE), which=lambda _n: "/bin/nvidia-smi").status == "READY"


# ---------------------------------------------------------------- no-fabrication static guard

FORBIDDEN_LITERALS = ("2.184", "1.712", "2.215", "1.798", "0.9996", "RTX 2070", "610.88")


def _docstring_constants(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                ids.add(id(body[0].value))
    return ids


def find_forbidden_literals(source: str) -> list[str]:
    """Forbidden literals appearing as code constants (comments are invisible to ast, docstrings are skipped)."""
    tree = ast.parse(source)
    skip = _docstring_constants(tree)
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or id(node) in skip:
            continue
        v = node.value
        for lit in FORBIDDEN_LITERALS:
            if isinstance(v, str) and lit in v:
                hits.append(lit)
            elif isinstance(v, float) and repr(v) == lit:
                hits.append(lit)
    return hits


def test_no_fabricated_gpu_or_qlora_constants_in_worker() -> None:
    assert not hasattr(worker, "compute_deterministic_qlora_outer_delta")
    assert find_forbidden_literals(SCRIPT.read_text(encoding="utf-8")) == []


def test_static_guard_can_fail_mutation_check() -> None:
    mutated = SCRIPT.read_text(encoding="utf-8") + '\nBAD = {"train_nll_before": 2.184, "gpu": "NVIDIA GeForce RTX 2070", "drv": "610.88"}\n'
    hits = find_forbidden_literals(mutated)
    assert "2.184" in hits
    assert "RTX 2070" in hits
    assert "610.88" in hits
    # a docstring mention alone is not flagged
    assert find_forbidden_literals('"""old value 2.184 on RTX 2070"""\nX = 1\n') == []
