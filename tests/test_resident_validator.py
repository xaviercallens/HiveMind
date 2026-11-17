"""Resident validator: model cache, UNIX-socket protocol, fallback and verdict equality with the one-shot path.

CPU only, fake loader/unloader. The module itself imports no torch; the equality test uses torch tensors for the toy states only.
"""

from __future__ import annotations

import json
import os
import socket
import stat
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hivemind.boinc.real_client import resident_validator as rv  # noqa: E402
from hivemind.boinc.real_client.resident_validator import ModelCache, ResidentValidatorClient, ResidentValidatorServer  # noqa: E402


def fake_cache(**kw):
    loads: list = []
    unloads: list = []

    def loader(key):
        loads.append(key)
        return {"key": key}

    cache = ModelCache(loader, unloads.append, sizer=lambda o: o["key"][1] if isinstance(o["key"], tuple) and isinstance(o["key"][1], int) else 0, **kw)
    return cache, loads, unloads


# ---------------------------------------------------------------------------------------------------------------- cache
def test_hit_miss_and_metrics() -> None:
    t = iter(range(0, 1000, 5))
    cache, loads, _ = fake_cache(max_entries=2, clock=lambda: float(next(t)))
    a = cache.checkout(("m", 1)); cache.release(("m", 1), a)
    a2 = cache.checkout(("m", 1)); cache.release(("m", 1), a2)
    assert a is a2 and loads == [("m", 1)]
    s = cache.stats()
    assert (s["hits"], s["misses"], s["loads"], s["evictions"]) == (1, 1, 1, 0)
    assert s["seconds_saved_estimate"] == 5.0  # one hit x the measured load time of the entry


def test_eviction_by_entries_is_lru() -> None:
    cache, loads, unloads = fake_cache(max_entries=2)
    for k in (("a", 1), ("b", 1)):
        with cache.acquire(k):
            pass
    with cache.acquire(("a", 1)):  # a is now most recently used
        pass
    with cache.acquire(("c", 1)):
        pass
    assert ("b", 1) not in cache and ("a", 1) in cache and ("c", 1) in cache
    assert unloads == [{"key": ("b", 1)}] and cache.stats()["evictions"] == 1


def test_eviction_by_bytes() -> None:
    cache, _, unloads = fake_cache(max_entries=10, max_bytes=100)
    for k in (("a", 60), ("b", 30), ("c", 50)):
        with cache.acquire(k):
            pass
    assert ("a", 60) not in cache and len(cache) == 2 and cache.stats()["bytes"] == 80
    assert unloads == [{"key": ("a", 60)}]


def test_single_oversized_entry_is_kept_alone_and_counted() -> None:
    cache, _, _ = fake_cache(max_entries=3, max_bytes=100)
    with cache.acquire(("a", 40)):
        pass
    with cache.acquire(("big", 500)):
        pass
    assert len(cache) == 1 and ("big", 500) in cache and cache.stats()["over_budget"] == 1


def test_concurrent_same_key_loads_once() -> None:
    gate = threading.Event()
    calls: list = []

    def slow(key):
        calls.append(key)
        gate.wait(5)
        return object()

    cache = ModelCache(slow, max_entries=2)
    got: list = []
    ts = [threading.Thread(target=lambda: got.append(cache.checkout("k"))) for _ in range(8)]
    for t in ts:
        t.start()
    time.sleep(0.2)
    gate.set()
    for t in ts:
        t.join(5)
    assert len(calls) == 1 and len({id(x) for x in got}) == 1 and len(got) == 8
    assert cache.stats()["loads"] == 1 and cache.stats()["hits"] == 7


def test_loader_failure_caches_nothing_and_isolates_keys() -> None:
    def loader(key):
        if key == "bad":
            raise RuntimeError("out of memory")
        return key

    cache = ModelCache(loader, max_entries=2)
    assert cache.checkout("good") == "good"
    for _ in range(2):  # no negative caching: it tries again
        with pytest.raises(RuntimeError, match="out of memory"):
            cache.checkout("bad")
    assert "bad" not in cache and "good" in cache and cache.checkout("good") == "good"
    assert cache.stats()["load_failures"] == 2 and cache.stats()["loads"] == 1


def test_evict_clear_and_deferred_unload_of_checked_out_entry() -> None:
    cache, _, unloads = fake_cache(max_entries=2)
    obj = cache.checkout(("a", 1))
    assert cache.evict(("a", 1)) is True and unloads == []  # still checked out: unloaded on release
    cache.release(("a", 1), obj)
    assert unloads == [obj] and cache.evict(("a", 1)) is False
    with cache.acquire(("b", 1)):
        pass
    with cache.acquire(("c", 1)):
        pass
    assert cache.clear() == 2 and len(cache) == 0 and len(unloads) == 3


def test_unloader_exception_does_not_break_cache(capsys) -> None:
    def bad_unload(o):
        raise RuntimeError("boom")

    cache = ModelCache(lambda k: k, bad_unload, max_entries=1)
    with cache.acquire("a"):
        pass
    with cache.acquire("b"):
        pass
    assert "b" in cache and "unloader failed" in capsys.readouterr().err


# -------------------------------------------------------------------------------------------------------- server/client
@pytest.fixture
def short_dir():
    import tempfile

    # AF_UNIX paths are limited to ~104 bytes: use a short temp dir (pytest's tmp_path can be too long)
    d = Path(tempfile.mkdtemp(prefix="rv"))
    os.chmod(d, 0o700)
    yield d
    import shutil

    shutil.rmtree(d, ignore_errors=True)


def make_server(short_dir: Path, validate_fn=None, cache=None, **kw):
    work = short_dir / "work"
    work.mkdir(exist_ok=True)
    srv = ResidentValidatorServer(short_dir / "sock" / "rv.sock", work, cache, validate_fn=validate_fn or (lambda files, log: (log("ok " + files[-1]), 0)[1]), **kw)
    srv.start()
    return srv, work


def test_round_trip_ping_stats_validate(short_dir: Path) -> None:
    cache, _, _ = fake_cache()
    srv, work = make_server(short_dir, cache=cache)
    try:
        c = ResidentValidatorClient(srv.socket_path, 10)
        assert c.ping()["pong"] is True
        f = work / "meta.json"
        f.write_text("{}")
        r = c.validate([str(f), "42"])
        assert r == {"ok": True, "rc": 0, "log": ["ok 42"]}
        st = c.stats()
        assert st["requests"]["validate"] == 1 and st["cache"]["entries"] == 0
    finally:
        srv.shutdown()


def test_nonzero_rc_and_log_are_relayed(short_dir: Path) -> None:
    srv, work = make_server(short_dir, validate_fn=lambda files, log: (log("REJECTED"), 1)[1])
    try:
        assert ResidentValidatorClient(srv.socket_path, 10).validate([str(work / "x"), "7"])["rc"] == 1
    finally:
        srv.shutdown()


def test_paths_outside_work_root_refused(short_dir: Path) -> None:
    ran: list = []
    srv, work = make_server(short_dir, validate_fn=lambda files, log: ran.append(files) or 0)
    try:
        c = ResidentValidatorClient(srv.socket_path, 10)
        (short_dir / "outside.json").write_text("{}")
        link = work / "link"
        link.symlink_to(short_dir / "outside.json")
        for bad in ("/etc/passwd", str(short_dir / "outside.json"), str(work / ".." / "outside.json"), str(link)):
            r = c.validate([bad, "1"])
            assert r["ok"] is False and "outside the work root" in r["error"], bad
        assert c.validate([])["ok"] is False and c.validate("x")["ok"] is False  # type: ignore[arg-type]
        assert ran == [] and c.stats()["requests"]["refused"] == 6
    finally:
        srv.shutdown()


def test_message_size_limit(short_dir: Path) -> None:
    srv, work = make_server(short_dir, max_message_bytes=2048)
    try:
        c = ResidentValidatorClient(srv.socket_path, 10, max_message_bytes=1 << 20)
        r = c.validate([str(work / ("a" * 5000)), "1"])
        assert r["ok"] is False and "exceeds the limit" in r["error"]
        # a hostile length prefix is refused without reading the body
        s = socket.socket(socket.AF_UNIX)
        s.settimeout(5)
        s.connect(str(srv.socket_path))
        s.sendall(struct.pack(">I", 0x7FFFFFFF))
        assert b"exceeds the limit" in rv.recv_msg(s)["error"].encode()
        s.close()
        # garbage JSON
        s = socket.socket(socket.AF_UNIX)
        s.settimeout(5)
        s.connect(str(srv.socket_path))
        s.sendall(struct.pack(">I", 3) + b"{{{")
        assert rv.recv_msg(s)["ok"] is False
        s.close()
        assert c.ping()["pong"]  # server survived
    finally:
        srv.shutdown()


def test_request_timeout(short_dir: Path) -> None:
    release = threading.Event()
    srv, work = make_server(short_dir, validate_fn=lambda files, log: (release.wait(5), 0)[1], request_timeout_s=0.3)
    try:
        r = ResidentValidatorClient(srv.socket_path, 10).validate([str(work / "x"), "1"])
        assert r["ok"] is False and "timeout" in r["error"]
        assert ResidentValidatorClient(srv.socket_path, 10).stats()["requests"]["timeouts"] == 1
    finally:
        release.set()
        srv.shutdown()


def test_validate_exception_is_reported_not_fatal(short_dir: Path) -> None:
    def boom(files, log):
        raise ValueError("kaput")

    srv, work = make_server(short_dir, validate_fn=boom)
    try:
        r = ResidentValidatorClient(srv.socket_path, 10).validate([str(work / "x"), "1"])
        assert r["ok"] is False and "kaput" in r["error"]
        assert ResidentValidatorClient(srv.socket_path, 10).ping()["pong"]
    finally:
        srv.shutdown()


def test_socket_permissions_and_no_tcp_listener(short_dir: Path) -> None:
    srv, _ = make_server(short_dir)
    try:
        assert stat.S_IMODE(srv.socket_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(srv.socket_path.parent.stat().st_mode) == 0o700
        assert stat.S_ISSOCK(srv.socket_path.stat().st_mode)
        assert srv._server.address_family == socket.AF_UNIX  # type: ignore[union-attr]
        # nothing of ours listens on TCP: the server has exactly one listening socket and it is AF_UNIX
        pid = os.getpid()
        tcp = []
        for fd in os.listdir(f"/proc/{pid}/fd"):
            try:
                target = os.readlink(f"/proc/{pid}/fd/{fd}")
            except OSError:
                continue
            if target.startswith("socket:"):
                tcp.append(target)
        inode_unix = {line.split()[6] for line in Path("/proc/net/unix").read_text().splitlines()[1:] if str(srv.socket_path) in line}
        assert inode_unix, "socket not found in /proc/net/unix"
        ours = {t[8:-1] for t in tcp}
        tcp_listen = set()
        for tbl in ("/proc/net/tcp", "/proc/net/tcp6"):
            if Path(tbl).exists():
                for line in Path(tbl).read_text().splitlines()[1:]:
                    cols = line.split()
                    if cols[3] == "0A":  # LISTEN
                        tcp_listen.add(cols[9])
        assert not (ours & tcp_listen), "this process holds a TCP listening socket"
    finally:
        srv.shutdown()


def test_refuses_loose_directory_and_live_duplicate(short_dir: Path) -> None:
    loose = short_dir / "loose"
    loose.mkdir()
    os.chmod(loose, 0o755)
    with pytest.raises(PermissionError):
        ResidentValidatorServer(loose / "s.sock", short_dir).start()
    srv, _ = make_server(short_dir)
    try:
        with pytest.raises(OSError, match="already listening"):
            ResidentValidatorServer(srv.socket_path, short_dir).start()
    finally:
        srv.shutdown()
    assert not srv.socket_path.exists()  # removed on shutdown


def test_stale_socket_is_replaced(short_dir: Path) -> None:
    d = short_dir / "sock"
    d.mkdir(mode=0o700)
    s = socket.socket(socket.AF_UNIX)
    s.bind(str(d / "rv.sock"))
    s.close()  # dead socket file left behind
    srv, _ = make_server(short_dir)
    try:
        assert ResidentValidatorClient(srv.socket_path, 5).ping()["pong"]
    finally:
        srv.shutdown()


def test_shutdown_unloads_models(short_dir: Path) -> None:
    cache, _, unloads = fake_cache(max_entries=3)
    srv, _ = make_server(short_dir, cache=cache)
    for k in (("a", 1), ("b", 1)):
        with cache.acquire(k):
            pass
    srv.shutdown()
    assert len(unloads) == 2 and len(cache) == 0


def test_sigterm_shuts_down_cleanly_and_serves_nothing_loaded(short_dir: Path) -> None:
    sock = short_dir / "s" / "rv.sock"
    (short_dir / "w").mkdir()
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": ""}
    p = subprocess.Popen([sys.executable, "-m", "hivemind.boinc.real_client.resident_validator", "serve", "--socket", str(sock), "--work-root", str(short_dir / "w")],
                         env=env, stderr=subprocess.PIPE)
    try:
        for _ in range(100):
            if sock.exists():
                break
            time.sleep(0.1)
        c = ResidentValidatorClient(sock, 5)
        assert c.ping()["pong"] and c.stats()["cache"]["entries"] == 0 and c.stats()["cache"]["loads"] == 0
        out = subprocess.run([sys.executable, "-m", "hivemind.boinc.real_client.resident_validator", "ping", "--socket", str(sock)], env=env, capture_output=True, text=True, timeout=30)
        assert out.returncode == 0 and '"pong": true' in out.stdout
        p.terminate()
        assert p.wait(15) == 0
        assert not sock.exists()
    finally:
        if p.poll() is None:
            p.kill()
    gone = subprocess.run([sys.executable, "-m", "hivemind.boinc.real_client.resident_validator", "stats", "--socket", str(sock)], env=env, capture_output=True, text=True, timeout=30)
    assert gone.returncode == 1


def test_module_imports_no_torch() -> None:
    code = "import sys; import hivemind.boinc.real_client.resident_validator; assert 'torch' not in sys.modules"
    r = subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ------------------------------------------------------------------------------------------------------------- fallback
def test_forward_falls_back_with_one_warning_when_socket_missing(short_dir: Path, capsys) -> None:
    assert rv.forward_validate(str(short_dir / "nope.sock"), ["a", "1"]) is None
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "running one-shot" in err[0]


def test_validate_entry_point_falls_back_to_one_shot(short_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from hivemind.boinc.real_client import boinc_server_scripts as bss

    seen: list = []
    monkeypatch.setattr(bss, "run_validation", lambda args, *a, **k: seen.append(list(args)) or 5)
    monkeypatch.setenv(rv.ENV_SOCKET, str(short_dir / "missing.sock"))
    assert bss.validate(["f.json", "g.pt", "9"]) == 5 and seen == [["f.json", "g.pt", "9"]]
    assert capsys.readouterr().err.count("running one-shot") == 1
    # flag form, socket from the flag, flag stripped before the one-shot path sees the arguments
    seen.clear()
    monkeypatch.delenv(rv.ENV_SOCKET)
    assert bss.validate([f"--resident={short_dir}/missing.sock", "f.json", "9"]) == 5 and seen == [["f.json", "9"]]
    # --resident with no socket anywhere: warn, one-shot
    seen.clear()
    assert bss.validate(["--resident", "f.json", "9"]) == 5 and seen == [["f.json", "9"]]
    assert "without a socket" in capsys.readouterr().err


def test_env_unset_is_exactly_the_one_shot_path(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from hivemind.boinc.real_client import boinc_server_scripts as bss

    monkeypatch.delenv(rv.ENV_SOCKET, raising=False)
    seen: list = []
    monkeypatch.setattr(bss, "run_validation", lambda args, *a, **k: seen.append((list(args), a, k)) or 0)
    monkeypatch.setitem(sys.modules, "hivemind.boinc.real_client.resident_validator", None)  # prove the module is never even imported
    assert bss.validate(["a.json", "b.pt", "3"]) == 0
    assert seen == [(["a.json", "b.pt", "3"], (), {})] and capsys.readouterr().err == ""


# ------------------------------------------------------------------------------------------ equal verdicts, both paths
def test_resident_and_one_shot_give_identical_verdicts(short_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    import hivemind.core.diloco_inner as di
    import hivemind.core.quantization as quant
    from hivemind.boinc.real_client import boinc_server_scripts as bss
    from hivemind.boinc.real_client import moe_app
    from hivemind.boinc.real_client.moe_app import state_sha
    from hivemind.boinc.workunit import sign_shard_bytes

    K, SEG = 3, 5
    work = short_dir / "work"
    work.mkdir()
    true = [{"w": torch.full((3,), float(i))} for i in range(K + 1)]
    tokens = [[1, 2, 3, 4]] * 4
    sha, sig = sign_shard_bytes(json.dumps(tokens, separators=(",", ":")).encode(), 7)
    spec_dir, ledgers = short_dir / "specs", {"one": short_dir / "l1", "res": short_dir / "l2"}
    spec_dir.mkdir()
    base_spec = {"app": "hivemind_qlora_moe", "inner_steps": K * SEG, "segment_steps": SEG, "train_tokens": tokens, "rng_seed": 7, "lr": 1e-3,
                 "weight_decay": 0.0, "micro_batch": 1, "shard_sha256": sha, "shard_hmac_signature": sig, "start_adapter_sha256": state_sha(true[0]),
                 "model_id": "fake/model", "rank": 8, "alpha": 16, "init_seed": 1}

    def make_result(wu: str, tamper: int | None) -> list[str]:
        claimed = [{"w": s["w"].clone()} for s in true]
        if tamper is not None:
            claimed[tamper + 1]["w"] += 5.0
        (spec_dir / f"{wu}.json").write_text(json.dumps({**base_spec, "wu_id": wu}))
        meta = {"wu_id": wu, "app": "hivemind_qlora_moe", "state_sha256": [state_sha(s) for s in claimed], "flops_claimed": 1e12, "wall_seconds": 10.0}
        mp, sp = work / f"{wu}_m.json", work / f"{wu}_s.pt"
        mp.write_text(json.dumps(meta))
        torch.save({"states": claimed}, sp)
        return [str(mp), str(sp), "77"]

    model_loads: list = []
    fake_model = object()

    def fake_load(spec, adapter_state=None):
        model_loads.append(spec["model_id"])
        return fake_model

    def replay(model, start, toks, a, b, **kw):
        assert model is fake_model
        return {"w": true[a // SEG + 1]["w"].clone()}  # what an honest replay produces

    monkeypatch.setattr(moe_app, "load_peft_model", fake_load)
    monkeypatch.setattr(quant, "extract_lora_state_dict", lambda m: true[0])
    monkeypatch.setattr(di, "replay_inner_segment", replay)
    monkeypatch.setattr(bss, "wait_for_idle", lambda *a, **k: 0.0)
    monkeypatch.setattr(bss, "SPEC_DIR", spec_dir)
    monkeypatch.setenv("HIVEMIND_COMMUNITY", "1")  # unknown account -> every segment replayed: deterministic

    cases = [("wu_ok", None), ("wu_bad", 1), ("wu_ok2", None)]
    results: dict[str, list] = {"one": [], "res": []}

    # one-shot: a fresh process-equivalent validator each time (what BOINC does today)
    monkeypatch.setattr(bss, "LEDGER_DIR", ledgers["one"])
    for wu, tamper in cases:
        lines: list[str] = []
        rc = bss.run_validation(make_result(wu, tamper), None, lines.append)
        results["one"].append((rc, lines))
    one_shot_loads = len(model_loads)
    model_loads.clear()

    # resident: same function, base model from the cache, over a real socket
    monkeypatch.setattr(bss, "LEDGER_DIR", ledgers["res"])
    cache = ModelCache(lambda key: (fake_load({"model_id": key[0]}), true[0]), lambda b: None, max_entries=1)
    srv = ResidentValidatorServer(short_dir / "sock" / "rv.sock", work, cache)
    srv.start()
    try:
        for wu, tamper in cases:
            args = make_result(wu, tamper)
            r = ResidentValidatorClient(srv.socket_path, 30).validate(args)
            assert r["ok"] is True
            results["res"].append((r["rc"], r["log"]))
    finally:
        srv.shutdown()

    assert results["one"] == results["res"]
    assert [rc for rc, _ in results["res"]] == [0, 1, 0] and "REJECTED segment 1" in results["res"][1][1][0]

    def verdicts(d: Path) -> dict[str, dict]:
        return {p.name.rsplit("_", 1)[0]: json.loads(p.read_text()) for p in d.glob("verdict_*.json")}

    v_one, v_res = verdicts(ledgers["one"]), verdicts(ledgers["res"])
    assert set(v_one) == {"verdict_wu_ok", "verdict_wu_bad", "verdict_wu_ok2"} and v_one == v_res
    # the point of the layer: three results, three loads one-shot, one load resident
    assert one_shot_loads == 3 and len(model_loads) == 1 and cache.stats()["loads"] == 1
