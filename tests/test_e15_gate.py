"""E15 gate: vendored auditor pin, functional verdicts, and the sandbox containment probes (planted hostile candidates; CPU only, stdlib only).

The containment tests ARE the sandbox verification: they run hostile code, but only through ``e15_gate.gate`` (never directly).
"""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import e15_gate as g  # noqa: E402

VENDOR = ROOT / "third_party" / "gwaya"
needs_sandbox = pytest.mark.skipif(not g.gate_available(), reason="user-namespace sandbox unavailable on this host (all verdicts would be UNVERIFIED)")


# ---------------------------------------------------------------- vendored auditor
def test_vendored_hashes_match_provenance():
    prov = (VENDOR / "PROVENANCE.md").read_text()
    for name in ("ast_audit.py", "LICENSE"):
        digest = hashlib.sha256((VENDOR / name).read_bytes()).hexdigest()
        assert digest in prov, f"{name} sha256 {digest} not in PROVENANCE.md"
    assert g.AUDITOR_SHA256 in prov
    assert "no modifications" in prov.lower()
    assert "9c0001e8307d079ac834a86bc37e2d56703a4fb5" in prov


def test_vendored_auditor_imports_stdlib_only_without_heavy_deps():
    code = ("import sys, importlib.util\n"
            f"spec = importlib.util.spec_from_file_location('a', {str(VENDOR / 'ast_audit.py')!r})\n"
            "m = importlib.util.module_from_spec(spec); sys.modules['a'] = m; spec.loader.exec_module(m)\n"
            "heavy = [k for k in ('torch', 'numpy', 'transformers', 'gwaya') if k in sys.modules]\n"
            "assert not heavy, heavy\n"
            "assert m.ZeroStubAudit.audit_python_code('def f(x):\\n    return x\\n').is_clean\n"
            "print('ok')\n")
    r = subprocess.run([sys.executable, "-I", "-S", "-c", code], capture_output=True, text=True, timeout=60, env={"PATH": os.environ.get("PATH", "")})
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


def test_auditor_only_imports_stdlib():
    import ast as _ast
    tree = _ast.parse((VENDOR / "ast_audit.py").read_text())
    mods = {n.module if isinstance(n, _ast.ImportFrom) else a.name for n in _ast.walk(tree) if isinstance(n, (_ast.Import, _ast.ImportFrom)) for a in getattr(n, "names", [None])}
    assert mods <= {"__future__", "ast", "re", "dataclasses", "typing"}, mods


def test_gate_loads_pinned_auditor_and_fails_closed_on_tamper(monkeypatch, tmp_path):
    assert g.audit_stub("def f(x):\n    return x\n") == []
    fake = tmp_path / "ast_audit.py"
    fake.write_text((VENDOR / "ast_audit.py").read_text() + "\n# tampered\n")
    monkeypatch.setattr(g, "AUDITOR_PATH", fake)
    monkeypatch.setattr(g, "_AUDITOR", None)
    with pytest.raises(RuntimeError):
        g.load_auditor()


# ---------------------------------------------------------------- extraction / syntax / stub (no sandbox needed for the verdict logic)
def test_extract_last_python_block():
    t = "x\n```python\nA = 1\n```\ntext\n```python\nB = 2\n```\n"
    assert g.extract_code(t) == "B = 2"
    assert g.extract_code("no code here") is None
    assert g.extract_code("```python\nunterminated = 1") is None


@needs_sandbox
@pytest.mark.parametrize("name,code", [
    ("pass", "def f():\n    pass\n"),
    ("ellipsis", "def f():\n    ...\n"),
    ("not_implemented", "def f():\n    raise NotImplementedError\n"),
    ("return_none_only", "def f() -> int:\n    return None\n"),
    ("mock_name", "def mock_f():\n    return 1\ndef f():\n    return mock_f()\n"),
    ("fake_name", "def fake_helper():\n    return 1\ndef f():\n    return 1\n"),
    ("dummy_name", "def dummy_f():\n    return 1\n"),
    ("stub_name", "def stub_f():\n    return 1\n"),
    ("docstring_only", 'def f():\n    """todo"""\n'),
])
def test_stubs_rejected(name, code):
    r = g.gate(code, ["assert True"])
    assert r.verdict == g.REJECTED_STUB, (name, r)
    assert r.violations


@needs_sandbox
def test_syntax_error_rejected():
    assert g.gate("def f(:\n  return 1", ["assert True"]).verdict == g.REJECTED_SYNTAX
    assert g.gate("", ["assert True"]).verdict == g.REJECTED_SYNTAX


# ---------------------------------------------------------------- functional
@needs_sandbox
def test_correct_solution_accepted_and_wrong_rejected():
    ok = g.gate("def add(a, b):\n    return a + b\n", ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"])
    assert ok.verdict == g.ACCEPTED and ok.wall_s > 0
    bad = g.gate("def add(a, b):\n    return a - b\n", ["assert add(1, 2) == 3"])
    assert bad.verdict == g.REJECTED_TEST and "AssertionError" in bad.test_error
    assert "assert add(1, 2) == 3" in bad.test_error


@needs_sandbox
def test_fenced_input_setup_and_imports():
    r = g.gate("```python\nimport math\ndef area(r):\n    return math.pi * r * r\n```", ["assert abs(area(1) - K) < 1e-9"], setup="import math\nK = math.pi")
    assert r.verdict == g.ACCEPTED, r


@needs_sandbox
def test_every_test_must_pass_and_early_exit_is_not_acceptance():
    code = "def f():\n    return 1\n"
    assert g.gate(code, ["assert f() == 1", "assert f() == 2"]).verdict == g.REJECTED_TEST
    assert g.gate("import os\ndef f():\n    os._exit(0)\n", ["assert f() == 1"]).verdict == g.REJECTED_TEST
    assert g.gate("def f():\n    return 1\nprint('E15_ALL_PASSED 0 0')\n", ["assert f() == 2"]).verdict == g.REJECTED_TEST


@needs_sandbox
def test_runtime_error_and_timeout_are_test_rejections():
    assert g.gate("def f():\n    return 1 / 0\n", ["assert f() == 1"]).verdict == g.REJECTED_TEST
    t = g.gate("def f():\n    while True:\n        x = 1\n", ["assert f() == 1"], timeout_s=2)
    assert t.verdict == g.REJECTED_TEST and "timeout" in t.test_error


def test_sandbox_disabled_gives_unverified_for_everything():
    g.set_sandbox_enabled(False)
    try:
        assert not g.gate_available()
        for code in ("def f():\n    return 1\n", "def f():\n    pass\n", "def f(:\n"):
            r = g.gate(code, ["assert f() == 1"])
            assert r.verdict == g.UNVERIFIED and not r.accepted
    finally:
        g.set_sandbox_enabled(True)
    g.gate_available(refresh=True)


def test_garbled_sandbox_output_is_unverified(monkeypatch):
    monkeypatch.setattr(g, "_AVAILABLE", True)
    for fake in (g.SandboxOutcome(ok=False, error="garbled sandbox report"), ):
        monkeypatch.setattr(g, "sandbox_run", lambda *a, _f=fake, **k: _f)
        assert g.gate("def f():\n    return 1\n", ["assert f() == 1"]).verdict == g.UNVERIFIED


def test_sandbox_error_is_unverified(monkeypatch):
    monkeypatch.setattr(g, "_AVAILABLE", True)
    monkeypatch.setattr(g, "_find_exec", lambda names: None)
    assert g.gate("def f():\n    return 1\n", ["assert f() == 1"]).verdict == g.UNVERIFIED


# ---------------------------------------------------------------- containment (planted hostile candidates)
@needs_sandbox
def test_probe_benign_first_then_availability_selftest():
    assert g.gate("def f():\n    return 1\n", ["assert f() == 1"]).verdict == g.ACCEPTED
    assert g.gate_available(refresh=True)


@needs_sandbox
def test_python_interpreter_is_not_under_home():
    py = g.system_python()
    assert py and not py.startswith(os.path.expanduser("~"))


@needs_sandbox
def test_all_hostile_candidates_contained():
    res = g.run_containment_probes()
    assert res["benign_correct_solution"]["verdict"] == g.ACCEPTED
    bad = {k: v for k, v in res.items() if isinstance(v, dict) and not v["contained"]}
    assert not bad, bad
    assert res["auditor_unchanged"] and res["canary_intact"] and res["all_contained"]
    for name in ("read_zenodo_token", "read_claude_json", "read_canary_in_home", "tcp_loopback", "tcp_external", "alloc_4gib", "infinite_loop", "fork_bomb",
                 "write_100mb", "subprocess_cat_proc1_environ", "modify_vendored_auditor", "os_system_curl", "umount_home"):
        assert name in res and res[name]["verdict"] != g.ACCEPTED


@needs_sandbox
def test_canary_in_real_home_is_invisible_and_secret_never_returned():
    secret = secrets.token_hex(16)
    d = tempfile.mkdtemp(dir=os.path.expanduser("~"), prefix=".e15_test_canary_")
    try:
        canary = os.path.join(d, "s.txt")
        Path(canary).write_text(secret)
        prog = (f"import os, sys\nprint('exists', os.path.exists({canary!r}), os.path.exists({os.path.expanduser('~')!r}), os.listdir('/home'))\n"
                f"try:\n    print(open({canary!r}).read())\nexcept OSError as e:\n    print('blocked', type(e).__name__)\n")
        o = g.sandbox_run(prog)
        assert o.ok and o.rc == 0
        assert secret not in o.stdout and secret not in o.stderr
        assert "exists False False []" in o.stdout and "blocked FileNotFoundError" in o.stdout
        assert Path(canary).read_text() == secret
    finally:
        shutil.rmtree(d, ignore_errors=True)
    assert not os.path.exists(d)


@needs_sandbox
def test_sandbox_environment_is_empty_and_caps_dropped():
    prog = ("import os\nprint(sorted(os.environ))\nprint([l.split()[1] for l in open('/proc/self/status') if l.startswith(('CapEff', 'NoNewPrivs'))])\n"
            "print(os.getcwd(), os.getuid())\n")
    o = g.sandbox_run(prog)
    assert o.ok and o.rc == 0
    lines = o.stdout.strip().splitlines()
    assert set(ast.literal_eval(lines[0])) <= {"PATH", "LC_CTYPE"}  # LC_CTYPE is added by CPython's C-locale coercion, not inherited
    assert lines[1] == "['0000000000000000', '1']"
    assert lines[2].startswith("/tmp/w ")


@needs_sandbox
def test_output_truncated_and_background_children_killed():
    o = g.sandbox_run("import sys\nsys.stdout.write('A' * 5_000_000)\nsys.stdout.flush()\n")
    assert o.ok and len(o.stdout) <= g.OUT_LIMIT
    t0 = __import__("time").time()
    o2 = g.sandbox_run("import os, time\nif os.fork() == 0:\n    time.sleep(1000)\nprint('parent done')\n", timeout_s=5)
    assert o2.ok and "parent done" in o2.stdout and __import__("time").time() - t0 < 10


@needs_sandbox
def test_dev_is_minimal_no_gpu_nodes_and_devnull_works():
    prog = ("import os, subprocess\nprint(sorted(os.listdir('/dev')))\nprint(len(os.urandom(8)))\n"
            "open('/dev/null', 'w').write('x')\nprint(os.path.exists('/dev/nvidiactl'), os.path.exists('/dev/nvidia0'))\n"
            "print(subprocess.run(['true'], stdin=subprocess.DEVNULL).returncode)\n")
    o = g.sandbox_run(prog)
    assert o.ok and o.rc == 0, o
    lines = o.stdout.strip().splitlines()
    assert set(ast.literal_eval(lines[0])) == {"null", "zero", "full", "random", "urandom", "fd", "stdin", "stdout", "stderr"}
    assert lines[1] == "8" and lines[2] == "False False" and lines[3] == "0"
