"""CPU-only tests for scripts/t4_model_probe.py argument handling, status classification and checkpointing/temperature helpers. No torch is imported."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import threading
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "t4_model_probe.py"


def _load():
    spec = importlib.util.spec_from_file_location("t4_model_probe_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


probe = _load()


def test_default_argv_maps_to_the_historic_spec_and_options():
    a = probe.build_parser().parse_args(["ibm-granite/granite-3.1-1b-a400m-base"])
    assert probe.build_spec(a) == {"model_id": "ibm-granite/granite-3.1-1b-a400m-base", "init_seed": 7, "rank": 8, "alpha": 16}
    assert (a.cap_mib, a.segments, a.segment_steps, a.micro_batch, a.seq_len) == (3950, 3, 10, 2, 128)
    assert a.checkpointing is False and a.targets is None and a.max_gpu_temp_c == 83
    assert probe.parse_targets(a) is None


def test_rank_alone_scales_alpha_and_explicit_alpha_wins():
    p = probe.build_parser()
    assert probe.build_spec(p.parse_args(["m", "--rank", "32"]))["alpha"] == 64
    assert probe.build_spec(p.parse_args(["m", "--rank", "16", "--alpha", "16"])) == {"model_id": "m", "init_seed": 7, "rank": 16, "alpha": 16}


def test_new_flags_parse():
    a = probe.build_parser().parse_args(["m", "--checkpointing", "--targets", "q_proj, v_proj", "--max-gpu-temp-c", "80", "--seq-len", "256", "--micro-batch", "1"])
    assert a.checkpointing and a.max_gpu_temp_c == 80 and a.seq_len == 256 and a.micro_batch == 1
    assert probe.parse_targets(a) == ("q_proj", "v_proj")


def test_importing_the_probe_does_not_import_torch():
    code = f"import importlib.util as u,sys; s=u.spec_from_file_location('p', {str(SCRIPT)!r}); m=u.module_from_spec(s); sys.modules['p']=m; s.loader.exec_module(m); m.build_parser().parse_args(['x']); assert 'torch' not in sys.modules"
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("exc,status", [
    (RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"), "OOM"),
    (type("OutOfMemoryError", (RuntimeError,), {})("boom"), "OOM"),
    (probe.CheckpointingUnsupported("no"), "CHECKPOINTING_UNSUPPORTED"),
    (ValueError("unsupported architecture"), "ERROR"),
])
def test_classify_status(exc, status):
    assert probe.classify_status(exc) == status


class _Cfg:
    use_cache = True


class _Model:
    def __init__(self, enable=True, reports=True, raises=None):
        self.config, self._enable, self._reports, self._raises, self.kwargs = _Cfg(), enable, reports, raises, None
        if enable:
            self.gradient_checkpointing_enable = self._do

    def _do(self, gradient_checkpointing_kwargs=None):
        if self._raises:
            raise self._raises
        self.kwargs = gradient_checkpointing_kwargs
        self.is_gradient_checkpointing = self._reports


def test_enable_checkpointing_success_disables_kv_cache():
    m = _Model()
    probe.enable_checkpointing(m)
    assert m.kwargs == {"use_reentrant": False} and m.config.use_cache is False


@pytest.mark.parametrize("model", [_Model(enable=False), _Model(raises=ValueError("not supported")), _Model(reports=False)])
def test_checkpointing_unsupported_is_reported_not_ignored(model):
    with pytest.raises(probe.CheckpointingUnsupported):
        probe.enable_checkpointing(model)


def test_emitter_prints_once(capsys):
    e = probe.Emitter()
    assert e.emit({"a": 1}) is True and e.emit({"a": 2}) is False
    assert capsys.readouterr().out.count("\n") == 1


def test_watch_temperature_calls_on_hot_once_above_limit():
    temps, hot = iter([70, 80, 84, 90]), []
    stop = threading.Event()
    probe.watch_temperature(83, lambda: next(temps), hot.append, stop, interval_s=0.0)
    assert hot == [84]


def test_watch_temperature_ignores_read_errors_and_stops():
    stop, calls = threading.Event(), {"n": 0}

    def read():
        calls["n"] += 1
        if calls["n"] == 3:
            stop.set()
        raise OSError("nvidia-smi gone")

    probe.watch_temperature(83, read, lambda t: pytest.fail("must not fire"), stop, interval_s=0.0)
    assert calls["n"] >= 3
