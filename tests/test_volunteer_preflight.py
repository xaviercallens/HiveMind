"""Tests for scripts/hivemind_volunteer_preflight.py: CPU only, deterministic, no network, no real nvidia-smi."""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("hivemind_volunteer_preflight", ROOT / "scripts" / "hivemind_volunteer_preflight.py")
pf = importlib.util.module_from_spec(_spec)
sys.modules["hivemind_volunteer_preflight"] = pf
_spec.loader.exec_module(pf)  # type: ignore[union-attr]

_REAL = pf.run_checks
NOW = datetime(2026, 10, 27, 23, 0, tzinfo=timezone.utc)
SCRUB = pf.Scrubber(home="/home/alice", user="alice", host="alicebox")


def smi(*rows: str, rc: int = 0):
    def runner(argv, timeout):
        assert argv[0] == "nvidia-smi" and timeout == 5
        return rc, "\n".join(rows) + "\n"
    return runner


def no_smi(argv, timeout):
    raise FileNotFoundError(argv[0])


RTX3060 = "NVIDIA GeForce RTX 3060, 12288, 512, 550.54, 8.6"
SMALL4 = "NVIDIA GeForce GTX 1050 Ti, 4096, 300, 535.1, 6.1"
GAME = "NVIDIA GeForce RTX 3060, 12288, 10500, 550.54, 8.6"
TINY2 = "NVIDIA GeForce GT 1030, 2048, 100, 535.1, 6.1"


def run(**kw):
    return _run(**kw)


def _run(**kw):
    base = dict(system="Linux", machine="x86_64", runner=smi(RTX3060), data_dir="/data", disk_free=lambda p: 50 * 10**9,
                ram_bytes=lambda: 16 * 2**30, cpus=8, which=lambda n: None, exists=lambda p: False, battery=lambda: {"present": False, "on_battery": None},
                scrub=SCRUB, now=NOW, py_version=(3, 11, 2))
    base.update(kw)
    return _REAL(**base)


def by_id(checks, i):
    return next(c for c in checks if c.id == i)


def test_ready_machine_rtx3060() -> None:
    cs = run()
    assert pf.verdict_of(cs) == "ready"
    g = by_id(cs, "gpu")
    assert g.status == "ok"
    assert g.data["tiers_servable_total"] == [
        "crowd-v1",
        "coder-lite",
        "gemma-core",
        "bee-stars",
        "mistral-core",
        "qwen35-core",
        "bee-rusty-sundials",
        "coder",
        "bee-lean-prover",
        "bee-ruche-moe",
        "coder-plus",
    ]
    assert g.data["tiers_too_big"] == ["moe-qwen35-bee", "moe-general", "moe-coder"]


def test_four_gb_card_trains_but_only_small_tiers() -> None:
    g = by_id(run(runner=smi(SMALL4)), "gpu")
    assert g.status == "ok" and g.data["tiers_servable_total"] == ["crowd-v1", "coder-lite"]


def test_two_gb_card_fails() -> None:
    assert by_id(run(runner=smi(TINY2)), "gpu").status == "fail"


def test_busy_gpu_warns() -> None:
    cs = run(runner=smi(GAME))
    g = by_id(cs, "gpu")
    assert g.status == "warn" and "game" in g.reason
    assert pf.verdict_of(cs) == "ready_with_warnings"


def test_no_gpu_fails_and_says_cpu_only_unsupported() -> None:
    cs = run(runner=no_smi)
    g = by_id(cs, "gpu")
    assert g.status == "fail" and "No NVIDIA GPU found" in g.reason and "CPU-only" in g.reason
    assert pf.verdict_of(cs) == "not_ready"


def test_allow_cpu_solver_succeeds_for_cpu_hivemind_solver() -> None:
    cs = run(runner=no_smi, allow_cpu_solver=True)
    g = by_id(cs, "gpu")
    assert g.status == "ok" and g.data["plan_class"] == "cpu_hivemind_solver"
    assert pf.verdict_of(cs) == "ready"


@pytest.mark.parametrize("bad", ["garbage", "a, b, c, d, e", "x, 100, 5000, 1, 1", "only,two"])
def test_malformed_output_warns_not_crashes(bad: str) -> None:
    assert by_id(run(runner=smi(bad)), "gpu").status == "warn"


def test_nonzero_exit_is_no_gpu() -> None:
    assert by_id(run(runner=smi("", rc=9)), "gpu").status == "fail"


def test_old_driver_without_compute_cap_falls_back() -> None:
    calls = []

    def runner(argv, timeout):
        calls.append(argv)
        return (2, "") if "compute_cap" in argv[1] else (0, "NVIDIA GeForce RTX 3060, 12288, 512, 470.1\n")
    assert by_id(run(runner=runner), "gpu").status == "ok" and len(calls) == 2


@pytest.mark.parametrize("free_gb,status", [(5, "fail"), (6.5, "fail"), (7, "warn"), (8.7, "warn"), (8.9, "ok"), (100, "ok")])
def test_disk_thresholds(free_gb: float, status: str) -> None:
    assert by_id(run(disk_free=lambda p: int(free_gb * 1e9)), "disk").status == status


def test_disk_error_warns() -> None:
    def boom(p):
        raise OSError("x")
    assert by_id(run(disk_free=boom), "disk").status == "warn"


def test_ram_cpu() -> None:
    assert by_id(run(ram_bytes=lambda: 4 * 2**30), "ram_cpu").status == "warn"
    assert by_id(run(cpus=2), "ram_cpu").status == "warn"
    assert by_id(run(ram_bytes=lambda: None), "ram_cpu").status == "ok"


@pytest.mark.parametrize("system,machine", [("Windows", "AMD64"), ("Darwin", "arm64"), ("Linux", "aarch64")])
def test_unsupported_platforms_fail(system: str, machine: str) -> None:
    c = by_id(run(system=system, machine=machine), "os")
    assert c.status == "fail"
    assert pf.verdict_of(run(system=system, machine=machine)) == "not_ready"


def test_boinc_detection() -> None:
    assert by_id(run(which=lambda n: "/usr/bin/" + n if n == "boinccmd" else None), "boinc").data["found"] == ["boinccmd"]
    assert "No BOINC client" in by_id(run(), "boinc").reason


def test_power_never_fails() -> None:
    for b in (None, {"present": True, "on_battery": True}, {"present": True, "on_battery": False}):
        c = by_id(run(battery=lambda b=b: b), "power")
        assert c.status == "ok"
    assert "inside" in by_id(run(battery=lambda: None), "power").reason  # 23:00 UTC is 00:00 Paris (CET) or 01:00 (CEST)


def test_exit_codes_and_aggregation(capsys) -> None:
    ok, warn, fail = (pf.Check("a", "a", s, "r") for s in ("ok", "warn", "fail"))
    assert pf.verdict_of([ok]) == "ready" and pf.verdict_of([ok, warn]) == "ready_with_warnings" and pf.verdict_of([ok, warn, fail]) == "not_ready"
    assert pf.EXIT == {"ready": 0, "ready_with_warnings": 1, "not_ready": 2}


def test_main_exit_code_and_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(pf, "default_runner", no_smi)
    monkeypatch.setattr(pf, "run_checks", lambda **kw: run(runner=no_smi))
    assert pf.main(["--json"]) == 2
    rep = json.loads(capsys.readouterr().out)
    assert rep["schema"] == pf.SCHEMA and rep["verdict"] == "not_ready"
    for c in rep["checks"]:
        assert set(c) == {"id", "title", "status", "reason", "what_to_do", "data"} and c["status"] in ("ok", "warn", "fail")
    monkeypatch.setattr(pf, "run_checks", lambda **kw: run())
    assert pf.main([]) == 0
    assert "Verdict: READY" in capsys.readouterr().out


def test_explain_text(capsys) -> None:
    assert pf.main(["--explain"]) == 0
    out = capsys.readouterr().out
    assert "sends nothing anywhere unless --project-url is given; it never installs anything" in out


# --- network / SSRF ---------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("url,frag", [
    ("http://example.org/", "https"),
    ("https://10.0.0.5/", "private"), ("https://127.0.0.1/", "private"), ("https://[::1]/", "private"), ("https://169.254.169.254/", "private"),
    ("https://[::ffff:10.0.0.1]/", "private"), ("https://[64:ff9b::7f00:1]/", "private"), ("https://user:pw@example.org/", "credentials"),
    ("ftp://example.org/", "https"), ("https://example.org:99999/", "valid"),
])
def test_url_refused(url: str, frag: str) -> None:
    prob, _ = pf.url_problem(url, lambda h: ["93.184.216.34"])
    assert prob is not None and frag in prob


def test_hostname_resolving_private_refused() -> None:
    assert pf.url_problem("https://evil.example/", lambda h: ["93.184.216.34", "192.168.1.2"])[0]
    assert pf.url_problem("https://good.example/", lambda h: ["93.184.216.34"]) == (None, ["93.184.216.34"])
    assert pf.url_problem("https://nx.example/", lambda h: (_ for _ in ()).throw(OSError()))[0]


def test_network_check_with_fake_fetcher() -> None:
    seen = []

    def fetcher(method, url, addrs, headers, timeout, max_bytes):
        seen.append((method, headers, timeout, max_bytes))
        return 206, {}, (0 if method == "HEAD" else 2 * 2**20), 1.0
    c = pf.check_network("https://good.example/p", lambda h: ["93.184.216.34"], fetcher)
    assert c.status == "ok" and c.data["first_download_minutes"] == 35  # 4.4e9 / 2097152 B/s / 60
    assert seen[0][0] == "HEAD" and seen[1][1]["Range"] == "bytes=0-2097151" and seen[1][2] == 10.0 and seen[1][3] == 2 * 2**20


def test_network_refusal_does_not_fetch() -> None:
    def fetcher(*a):
        raise AssertionError("must not fetch")
    for u in ("http://good.example/", "https://10.1.1.1/"):
        assert pf.check_network(u, lambda h: ["93.184.216.34"], fetcher).status == "fail"


def test_network_error_fails_cleanly() -> None:
    def fetcher(*a):
        raise OSError("down")
    assert pf.check_network("https://good.example/", lambda h: ["93.184.216.34"], fetcher).status == "fail"


def test_no_network_without_project_url(monkeypatch) -> None:
    def deny(*a, **k):
        raise AssertionError("network touched")
    monkeypatch.setattr(socket, "socket", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)
    monkeypatch.setattr(socket, "gethostbyname", deny)
    cs = pf.run_checks(runner=no_smi, data_dir="/tmp", which=lambda n: None, exists=lambda p: False, battery=lambda: None, ram_bytes=lambda: None,
                       scrub=SCRUB, now=NOW)
    assert all(c.id != "network" for c in cs)


# --- privacy ----------------------------------------------------------------------------------------------------------------------------------
def test_privacy_no_user_host_or_home_in_output() -> None:
    cs = run(data_dir="/home/alice/BOINC data for alicebox", runner=smi("alice GPU alicebox, 12288, 512, 550.54, 8.6"))
    rep = pf.build_report(cs, SCRUB)
    text = json.dumps(rep) + pf.render_table(rep)
    assert "alice" not in text and "/home/" not in text and "~/BOINC" in text


def test_default_scrubber_hides_real_names(monkeypatch) -> None:
    s = pf.Scrubber(home=str(Path.home()), user="someuser", host="somehost")
    assert s(f"{Path.home()}/x someuser somehost") == "~/x <redacted> <redacted>"


def test_no_gpu_uuid_requested() -> None:
    seen = []

    def runner(argv, timeout):
        seen.append(argv)
        return 0, RTX3060 + "\n"
    run(runner=runner)
    q = seen[0][1]
    assert "uuid" not in q and "serial" not in q


# --- facts kept in sync with the repo ---------------------------------------------------------------------------------------------------------
def test_tiers_match_catalog() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from hivemind.router.catalog import TIERS
    assert pf.TIERS == tuple((t.tier_id, t.min_vram_gb) for t in TIERS)


def test_ssrf_guard_matches_gateway() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from hivemind.router.gateway import _ip_is_global
    for a in ("8.8.8.8", "10.0.0.1", "127.0.0.1", "::1", "::ffff:10.0.0.1", "64:ff9b::808:808", "64:ff9b:1::1", "2001:4860:4860::8888", "fe80::1"):
        assert pf.ip_is_global(a) == _ip_is_global(a), a


def test_first_download_figures() -> None:
    assert pf.FIRST_DOWNLOAD_GB == 4.4 and (pf.RUNTIME_GB, pf.MODEL_GB) == (1.76, 2.67)
    assert pf.PLAN_CLASS_MIN_MB == 3500
    assert "<gpu_ram>3500000000</gpu_ram>" in (ROOT / "scripts" / "boinc_publish_app_version.py").read_text()
