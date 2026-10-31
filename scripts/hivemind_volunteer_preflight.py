#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Volunteer preflight: is this machine ready to join HiveMind as a BOINC volunteer?  Run it BEFORE installing anything.

Stdlib only, Python 3.10+, Linux / macOS / Windows.  It reads hardware facts and sends nothing anywhere unless ``--project-url`` is given;
it never installs, writes or changes anything.  Verdicts: ready (exit 0), ready_with_warnings (exit 1), not_ready (exit 2).

Where the facts come from (nothing here is invented; where the repo has no figure, the threshold is marked HEURISTIC):
* Platform: ``docs/BOINC_PRODUCTION_SERVER.md`` line "Apps": only platform ``x86_64-pc-linux-gnu`` is published; the launcher
  ``deploy/boinc_dist/launcher.sh.in`` is a bash script that needs tar, xz, flock (util-linux) and an NVIDIA driver.
* GPU floor: plan class ``cuda_hivemind`` = NVIDIA, >= 3,500 MB (``docs/BOINC_PRODUCTION_SERVER.md``; ``<gpu_ram>3500000000</gpu_ram>`` in
  ``scripts/boinc_publish_app_version.py``).  Measured training peak 3,409 MiB (``docs/BOINC_CORE_NODE.md``).
* First download about 4.4 GB = runtime 1.76 GB + model 2.67 GB + small files (figure given by the project owner for the portable
  distribution built by ``scripts/boinc_build_distribution.sh``; the repo files do not state it, re-measure it when the distribution changes).
* Tiers and ``min_vram_gb``: mirrored from ``src/hivemind/router/catalog.py`` (a test keeps the copy in sync).
* Night window 22:00-07:00 Europe/Paris for community work: ``src/hivemind/boinc/real_client/contributor_policy.py``.
* SSRF refusal (https only, public addresses only): mirrored from ``src/hivemind/router/gateway.py`` (``producer_url_problem``).
* RAM / CPU thresholds: HEURISTIC, the repo documents none.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import platform
import shutil
import socket
import ssl
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import urlsplit

SCHEMA = "hivemind.volunteer_preflight/1"

# --- facts (see module docstring for sources) -------------------------------------------------------------------------------------------------
SUPPORTED_PLATFORM = ("Linux", ("x86_64", "amd64"))  # x86_64-pc-linux-gnu
RUNTIME_GB, MODEL_GB, FIRST_DOWNLOAD_GB = 1.76, 2.67, 4.4  # decimal GB; small files make up the rest of the 4.4
DISK_FAIL_FACTOR, DISK_WARN_FACTOR = 1.5, 2.0
PLAN_CLASS_MIN_MB = 3500  # decimal MB, cuda_hivemind
TRAIN_PEAK_MIB = 3409
BUSY_FRACTION = 0.5  # HEURISTIC: more than half of the VRAM already used by something else
MIN_RAM_GB, MIN_CPUS = 8.0, 4  # HEURISTIC
NIGHT = ("Europe/Paris", 22, 7)
MIB_TO_MB = 1.048576
# (tier_id, min_vram_gb) in the order of src/hivemind/router/catalog.py TIERS
TIERS: tuple[tuple[str, float], ...] = (
    ("crowd-v1", 4.0),
    ("coder-lite", 4.0),
    ("gemma-core", 4.5),
    ("bee-stars", 5.5),
    ("mistral-core", 6.0),
    ("qwen35-core", 6.0),
    ("bee-rusty-sundials", 6.0),
    ("coder", 6.0),
    ("bee-lean-prover", 8.5),
    ("bee-ruche-moe", 10.0),
    ("coder-plus", 12.0),
    ("moe-qwen35-bee", 15.0),
    ("moe-general", 15.0),
    ("moe-coder", 24.0),
)
DATA_DIR_CANDIDATES = (
    "/var/lib/boinc-client", "/var/lib/boinc", "/Library/Application Support/BOINC Data",
    r"C:\ProgramData\BOINC", "~/BOINC", "~/.boinc",
)
BOINC_CLIENT_DIRS = (
    "/usr/bin", "/usr/local/bin", "/Applications/BOINCManager.app/Contents/MacOS", r"C:\Program Files\BOINC",
)

Runner = Callable[[Sequence[str], float], "tuple[int, str]"]


@dataclass
class Check:
    id: str
    title: str
    status: str  # ok | warn | fail
    reason: str
    todo: str = ""
    data: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "status": self.status, "reason": self.reason, "what_to_do": self.todo, "data": self.data or {}}


# --- privacy ----------------------------------------------------------------------------------------------------------------------------------
class Scrubber:
    """Replaces the home directory with '~' and any user or host name with a placeholder in every string that is printed."""

    def __init__(self, home: str | None = None, user: str | None = None, host: str | None = None) -> None:
        def _safe(f: Callable[[], str]) -> str:
            try:
                return f()
            except Exception:
                return ""
        self.home = home if home is not None else str(Path.home())
        self.names = [n for n in {user if user is not None else _safe(lambda: os.environ.get("USER") or os.environ.get("USERNAME") or ""),
                                  host if host is not None else _safe(socket.gethostname)} if len(n) >= 3]
        self.names.sort(key=len, reverse=True)

    def __call__(self, text: str) -> str:
        if self.home and self.home not in ("/", ""):
            text = text.replace(self.home, "~")
        for n in self.names:
            text = text.replace(n, "<redacted>")
        return text


def display_path(p: str | Path, scrub: Scrubber) -> str:
    return scrub(str(p))


# --- default system probes (all injectable) -------------------------------------------------------------------------------------------------
def default_runner(argv: Sequence[str], timeout: float) -> tuple[int, str]:
    exe = shutil.which(argv[0])
    if exe is None:
        raise FileNotFoundError(argv[0])
    try:
        proc = subprocess.run([exe, *argv[1:]], capture_output=True, text=True, timeout=timeout, check=False, shell=False)
    except subprocess.TimeoutExpired:
        return 124, ""
    return proc.returncode, proc.stdout


def default_disk_free(path: str) -> int:
    return shutil.disk_usage(path).free


def default_ram_bytes() -> int | None:
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        if sys.platform == "darwin":
            rc, out = default_runner(["sysctl", "-n", "hw.memsize"], 5)
            return int(out.strip()) if rc == 0 else None
        if sys.platform == "win32":
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("m", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong),
                            ("b", ctypes.c_ulonglong), ("c", ctypes.c_ulonglong), ("d", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong),
                            ("f", ctypes.c_ulonglong)]
            ms = _MS()
            ms.l = ctypes.sizeof(_MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):  # type: ignore[attr-defined]
                return int(ms.total)
    except Exception:
        return None
    return None


def default_battery() -> dict[str, Any] | None:
    """{'on_battery': bool|None, 'present': bool} best effort, or None when unknown."""
    try:
        if sys.platform.startswith("linux"):
            bats = sorted(Path("/sys/class/power_supply").glob("BAT*"))
            if not bats:
                return {"present": False, "on_battery": None}
            status = (bats[0] / "status").read_text().strip().lower()
            return {"present": True, "on_battery": status == "discharging"}
        if sys.platform == "darwin":
            rc, out = default_runner(["pmset", "-g", "batt"], 5)
            if rc != 0:
                return None
            return {"present": "InternalBattery" in out, "on_battery": "Battery Power" in out}
        if sys.platform == "win32":
            import ctypes

            class _SPS(ctypes.Structure):
                _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte), ("pct", ctypes.c_ubyte), ("sf", ctypes.c_ubyte),
                            ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]
            s = _SPS()
            if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):  # type: ignore[attr-defined]
                return {"present": s.flag != 128, "on_battery": s.ac == 0}
    except Exception:
        return None
    return None


# --- SSRF guard (mirror of src/hivemind/router/gateway.py) --------------------------------------------------------------------------------------
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_NAT64_LOCAL = ipaddress.ip_network("64:ff9b:1::/48")


def _embedded_ipv4(a: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    if a.ipv4_mapped is not None:
        return a.ipv4_mapped
    if a in _NAT64:
        return ipaddress.IPv4Address(int(a) & 0xFFFFFFFF)
    if a.sixtofour is not None:
        return a.sixtofour
    if a.teredo is not None:
        return a.teredo[1]
    if int(a) >> 32 == 0 and int(a) > 1:
        return ipaddress.IPv4Address(int(a) & 0xFFFFFFFF)
    return None


def ip_is_global(text: str) -> bool:
    a = ipaddress.ip_address(text.split("%", 1)[0])
    if isinstance(a, ipaddress.IPv6Address):
        if a in _NAT64_LOCAL:
            return False
        emb = _embedded_ipv4(a)
        if emb is not None:
            return emb.is_global
    return a.is_global


def default_resolver(host: str) -> list[str]:
    return sorted({str(i[4][0]) for i in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)})


def url_problem(url: str, resolver: Callable[[str], list[str]] = default_resolver) -> tuple[str | None, list[str]]:
    """(reason the URL is refused or None, resolved public addresses).  Nothing but DNS resolution happens here."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return "not a valid URL", []
    if parts.scheme != "https":
        return "only https:// URLs are accepted (http is refused)", []
    host = parts.hostname
    if not host or parts.username is not None or parts.password is not None:
        return "the URL must have a host and no credentials", []
    if port is not None and not 0 < port < 65536:
        return "invalid port", []
    try:
        ipaddress.ip_address(host.split("%", 1)[0])
        addrs = [host]
    except ValueError:
        try:
            addrs = list(resolver(host))
        except OSError:
            return "the host does not resolve", []
    if not addrs:
        return "the host does not resolve", []
    for a in addrs:
        try:
            if not ip_is_global(a):
                return "the host is a private or non-public address (refused)", []
        except ValueError:
            return "the host resolved to an invalid address", []
    return None, addrs


Fetcher = Callable[[str, str, list[str], "dict[str, str]", float, int], "tuple[int, dict[str, str], int, float]"]


def default_fetcher(method: str, url: str, addrs: list[str], headers: dict[str, str], timeout: float, max_bytes: int) -> tuple[int, dict[str, str], int, float]:
    """One https request pinned to an already-vetted address (no DNS rebinding, no redirects).  Returns (status, headers, bytes read, seconds)."""
    import http.client

    parts = urlsplit(url)
    host, port = parts.hostname or "", parts.port or 443
    path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    ctx = ssl.create_default_context()
    last: Exception | None = None
    for addr in addrs:
        try:
            raw = socket.create_connection((addr, port), timeout=timeout)
            break
        except OSError as exc:
            last = exc
    else:
        raise OSError(f"could not connect: {last}")
    try:
        conn = http.client.HTTPSConnection(host, port, timeout=timeout, context=ctx)
        conn.sock = ctx.wrap_socket(raw, server_hostname=host)
        t0 = time.monotonic()
        conn.request(method, path, headers={"Host": host, "User-Agent": "hivemind-volunteer-preflight", "Accept-Encoding": "identity", **headers})
        resp = conn.getresponse()
        got = 0
        if method != "HEAD":
            while got < max_bytes:
                chunk = resp.read(min(65536, max_bytes - got))
                if not chunk:
                    break
                got += len(chunk)
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, got, time.monotonic() - t0
    finally:
        raw.close()


# --- checks -----------------------------------------------------------------------------------------------------------------------------------
def check_os(system: str, machine: str) -> Check:
    ok_sys, ok_arch = SUPPORTED_PLATFORM
    arch_ok = machine.lower() in ok_arch
    friendly = {"Linux": "Linux", "Darwin": "macOS", "Windows": "Windows"}.get(system, system)
    data = {"system": friendly, "machine": machine}
    if system == ok_sys and arch_ok:
        return Check("os", "Operating system", "ok", f"{friendly} on {machine} matches the published app platform (x86_64-pc-linux-gnu).", "", data)
    if system == ok_sys:
        return Check("os", "Operating system", "fail", f"Linux on {machine}: only x86_64 builds are published.", "Use an x86_64 (64-bit Intel/AMD) Linux machine.", data)
    return Check("os", "Operating system", "fail", f"{friendly} is not supported: the only published app version is for x86_64 Linux (its launcher is a bash script).",
                 "Use an x86_64 Linux machine with an NVIDIA GPU. Windows and macOS builds do not exist yet.", data)


def check_python(version: tuple[int, int, int]) -> Check:
    s = ".".join(map(str, version[:3]))
    ok = version >= (3, 10)
    return Check("python", "Python (informational)", "ok" if ok else "warn",
                 f"Python {s}. The volunteer app ships its own Python 3.10, so yours is not used by BOINC tasks." if ok else f"Python {s} is old; this script needs 3.10+ to run again.",
                 "" if ok else "Install Python 3.10 or newer only if you want to re-run this script.", {"version": s})


def _parse_gpu(out: str) -> list[dict[str, Any]]:
    gpus = []
    for line in out.strip().splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 4:
            raise ValueError("unexpected nvidia-smi columns")
        total, used = float(cols[1]), float(cols[2])
        if total <= 0 or used < 0 or used > total * 1.01:
            raise ValueError("implausible memory figures")
        cap = cols[4] if len(cols) > 4 and cols[4] else None
        gpus.append({"name": cols[0], "total_mib": total, "used_mib": used, "free_mib": max(total - used, 0.0), "driver": cols[3], "compute_cap": cap})
    if not gpus:
        raise ValueError("no rows")
    return gpus


def tier_split(vram_gb: float) -> tuple[list[str], list[str]]:
    fit = [t for t, m in TIERS if m <= vram_gb]
    return fit, [t for t, m in TIERS if m > vram_gb]


def check_gpu(runner: Runner, allow_cpu_solver: bool = False) -> Check:
    q = "--query-gpu=name,memory.total,memory.used,driver_version,compute_cap"
    title = "NVIDIA GPU"
    cpu_ok = Check(
        "gpu",
        title,
        "ok",
        "No NVIDIA driver/GPU detected via nvidia-smi; validated for CPU/BLS transit & symbolic solver mode (BOINC plan class cpu_hivemind_solver via hivemind_boinc_idle_worker.py).",
        "Install the NVIDIA proprietary driver if you also want GPU QLoRA tasks (plan class cuda_hivemind).",
        {"cards": 0, "plan_class": "cpu_hivemind_solver", "tiers_servable_total": [], "tiers_servable_free_now": [], "tiers_too_big": [t for t, _ in TIERS]},
    )
    none = Check("gpu", title, "fail", "No NVIDIA GPU found (nvidia-smi is missing or reports no card). The HiveMind QLoRA app needs an NVIDIA GPU (BOINC plan class cuda_hivemind); CPU-only volunteering is not supported.",
                 "Install the NVIDIA driver if you have a card, or pass --allow-cpu-solver to validate for the Python BOINC/BLS CPU worker (plan class cpu_hivemind_solver).")
    try:
        rc, out = runner(["nvidia-smi", q, "--format=csv,noheader,nounits"], 5)
        if rc != 0:  # old drivers do not know compute_cap
            rc, out = runner(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader,nounits"], 5)
    except (OSError, subprocess.SubprocessError):
        return cpu_ok if allow_cpu_solver else none
    if rc != 0 or not out.strip():
        return cpu_ok if allow_cpu_solver else none
    try:
        gpus = _parse_gpu(out)
    except (ValueError, IndexError):
        return Check("gpu", title, "warn", "nvidia-smi answered but its output could not be understood, so the card was not checked.",
                     "Run nvidia-smi yourself; update the NVIDIA driver if it errors.")
    g = max(gpus, key=lambda x: x["free_mib"])
    total_mb, free_mb = g["total_mib"] * MIB_TO_MB, g["free_mib"] * MIB_TO_MB
    fit_total, no_fit = tier_split(g["total_mib"] / 1024)
    fit_free, _ = tier_split(g["free_mib"] / 1024)
    data = {"cards": len(gpus), "name": g["name"], "total_mib": round(g["total_mib"]), "free_mib": round(g["free_mib"]), "driver": g["driver"],
            "compute_cap": g["compute_cap"], "tiers_servable_total": fit_total, "tiers_servable_free_now": fit_free, "tiers_too_big": no_fit}
    tiers = f"Serving tiers it can hold when idle: {', '.join(fit_total) or 'none'}; too big: {', '.join(no_fit) or 'none'}."
    if total_mb < PLAN_CLASS_MIN_MB:
        return Check("gpu", title, "fail", f"{g['name']} has {g['total_mib']:.0f} MiB; the training app needs at least {PLAN_CLASS_MIN_MB} MB (it peaks near {TRAIN_PEAK_MIB} MiB). {tiers}",
                     "A card with 4 GB or more is needed.", data)
    busy = g["used_mib"] / g["total_mib"] > BUSY_FRACTION
    if free_mb < PLAN_CLASS_MIN_MB or busy:
        return Check("gpu", title, "warn", f"{g['name']}: {g['free_mib']:.0f} of {g['total_mib']:.0f} MiB free; another program (a game?) is using the card. Training needs about {TRAIN_PEAK_MIB} MiB. {tiers}",
                     "Close games and GPU-heavy apps, or let BOINC run only while the PC is idle.", data)
    return Check("gpu", title, "ok", f"{g['name']}: {g['free_mib']:.0f} of {g['total_mib']:.0f} MiB free, driver {g['driver']}. Enough for training (peak about {TRAIN_PEAK_MIB} MiB). {tiers}", "", data)


def detect_data_dir(exists: Callable[[str], bool]) -> str | None:
    for c in DATA_DIR_CANDIDATES:
        p = os.path.expanduser(c)
        if exists(p):
            return p
    return None


def check_disk(path: str, free_fn: Callable[[str], int], scrub: Scrubber) -> Check:
    need_gb = FIRST_DOWNLOAD_GB
    fail_gb, warn_gb = need_gb * DISK_FAIL_FACTOR, need_gb * DISK_WARN_FACTOR
    where = display_path(path, scrub)
    try:
        free_gb = free_fn(path) / 1e9
    except OSError:
        return Check("disk", "Free disk space", "warn", f"Could not read free space for {where}.", "Check there are about 9 GB free where BOINC stores its data.")
    data = {"path": where, "free_gb": round(free_gb, 2), "first_download_gb": need_gb, "fail_below_gb": round(fail_gb, 2), "warn_below_gb": round(warn_gb, 2)}
    why = f"First download is about {need_gb} GB (runtime {RUNTIME_GB} + model {MODEL_GB} + small files) and it is unpacked once more."
    if free_gb < fail_gb:
        return Check("disk", "Free disk space", "fail", f"{free_gb:.1f} GB free in {where}; need at least {fail_gb:.1f} GB. {why}", "Free up space or use --data-dir on a bigger disk.", data)
    if free_gb < warn_gb:
        return Check("disk", "Free disk space", "warn", f"{free_gb:.1f} GB free in {where}; {warn_gb:.1f} GB is comfortable. {why}", "Free a few GB so updates do not run you out of space.", data)
    return Check("disk", "Free disk space", "ok", f"{free_gb:.1f} GB free in {where}. {why}", "", data)


def check_ram_cpu(ram_bytes: int | None, cpus: int | None) -> Check:
    data = {"ram_gb": round(ram_bytes / 2**30, 1) if ram_bytes else None, "cpus": cpus}
    notes, warn = [], False
    if ram_bytes is None:
        notes.append("RAM unknown")
    elif ram_bytes / 2**30 < MIN_RAM_GB:
        warn = True
        notes.append(f"{ram_bytes / 2**30:.1f} GB RAM is below {MIN_RAM_GB:.0f} GB")
    else:
        notes.append(f"{ram_bytes / 2**30:.1f} GB RAM")
    if cpus is None:
        notes.append("CPU count unknown")
    elif cpus < MIN_CPUS:
        warn = True
        notes.append(f"{cpus} CPU threads is below {MIN_CPUS}")
    else:
        notes.append(f"{cpus} CPU threads")
    return Check("ram_cpu", "RAM and CPU", "warn" if warn else "ok", "; ".join(notes) + ". (Thresholds are the project's rule of thumb, not from the docs.)",
                 "Expect slow tasks and a sluggish PC; limit BOINC to part of the CPU." if warn else "", data)


def check_boinc(which: Callable[[str], str | None], exists: Callable[[str], bool]) -> Check:
    found = [n for n in ("boinccmd", "boinc") if which(n)]
    if not found:
        for d in BOINC_CLIENT_DIRS:
            for n in ("boinccmd", "boinc", "boinccmd.exe", "boinc.exe"):
                if exists(os.path.join(d, n)):
                    found.append(n)
    data = {"found": sorted(set(found))}
    if found:
        return Check("boinc", "Existing BOINC client (informational)", "ok", "A BOINC client is already installed.",
                     "Do not reinstall: add the HiveMind project from the BOINC Manager (Add project) once the project page is published.", data)
    return Check("boinc", "Existing BOINC client (informational)", "ok", "No BOINC client found.", "Install the BOINC client from your distribution or boinc.berkeley.edu, then add the HiveMind project.", data)


def check_power(battery: Callable[[], dict[str, Any] | None], now: datetime) -> Check:
    b = battery()
    parts = []
    if b is None:
        parts.append("battery state unknown")
    elif not b["present"]:
        parts.append("no battery (desktop)")
    elif b["on_battery"]:
        parts.append("running on battery: BOINC drains it quickly, plug in or pause BOINC on battery")
    else:
        parts.append("laptop on mains power: allow it to stay awake while tasks run")
    try:
        from zoneinfo import ZoneInfo
        tz, s, e = NIGHT
        local = now.astimezone(ZoneInfo(tz))
        inside = s <= local.hour < e or (s > e and (local.hour >= s or local.hour < e))
        parts.append(f"community work is offered {s}:00-{e:02d}:00 {tz} (now {'inside' if inside else 'outside'} that window)")
    except Exception:
        parts.append(f"community work is offered {NIGHT[1]}:00-{NIGHT[2]:02d}:00 {NIGHT[0]}")
    parts.append("disable sleep/hibernate while a task runs, or tasks pause until the PC wakes")
    return Check("power", "Laptop, battery and sleep (informational)", "ok", "; ".join(parts) + ".", "", {"battery": b})


def check_network(url: str, resolver: Callable[[str], list[str]], fetcher: Fetcher) -> Check:
    title = "Project server and download speed"
    problem, addrs = url_problem(url, resolver)
    if problem:
        return Check("network", title, "fail", f"Refused: {problem}.", "Give the official https:// project URL.")
    host = urlsplit(url).hostname or ""
    try:
        status, _h, _n, _t = fetcher("HEAD", url, addrs, {}, 10.0, 0)
        status2, _h2, got, secs = fetcher("GET", url, addrs, {"Range": "bytes=0-2097151"}, 10.0, 2 * 2**20)
    except (OSError, ssl.SSLError, ValueError) as exc:
        return Check("network", title, "fail", f"Could not reach {host} over https ({type(exc).__name__}).", "Check your internet connection and the URL.")
    if status >= 400 or status2 >= 400:
        return Check("network", title, "warn", f"{host} answered HTTP {max(status, status2)}.", "Check the project URL.", {"host": host, "http_status": max(status, status2)})
    if got <= 0 or secs <= 0:
        return Check("network", title, "warn", f"{host} reachable (HTTP {status}) but the speed test returned no data.", "", {"host": host, "http_status": status})
    mbps = got * 8 / secs / 1e6
    minutes = FIRST_DOWNLOAD_GB * 1e9 / (got / secs) / 60
    st = "warn" if minutes > 120 else "ok"
    return Check("network", title, st, f"{host} reachable (HTTP {status}); {mbps:.0f} Mbit/s over {got / 2**20:.1f} MiB; the first download of about {FIRST_DOWNLOAD_GB} GB would take about {minutes:.0f} minutes.",
                 "A slow link works, BOINC resumes interrupted downloads." if st == "warn" else "", {"host": host, "http_status": status, "mbit_per_s": round(mbps, 1), "first_download_minutes": round(minutes)})


# --- orchestration ----------------------------------------------------------------------------------------------------------------------------
def verdict_of(checks: Sequence[Check]) -> str:
    if any(c.status == "fail" for c in checks):
        return "not_ready"
    return "ready_with_warnings" if any(c.status == "warn" for c in checks) else "ready"


EXIT = {"ready": 0, "ready_with_warnings": 1, "not_ready": 2}

EXPLAIN = """HiveMind volunteer preflight: what it reads.
This script reads hardware facts and sends nothing anywhere unless --project-url is given; it never installs anything.
  os        platform.system() / machine(): is this x86_64 Linux, the only published app platform?
  python    the Python version running this script.
  gpu       runs `nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,compute_cap` (5 s limit). No GPU UUID, no serial.
  disk      free bytes in the data directory (BOINC data dir if present, else your home directory, or --data-dir). Nothing is written.
  ram_cpu   total RAM and CPU thread count.
  boinc     whether boinccmd/boinc are on PATH or in standard install folders.
  power     battery present / on battery, and the community night window. Never fails.
  network   ONLY with --project-url: https only, public addresses only; a HEAD and one ranged GET of at most 2 MiB (10 s limits).
Never printed or stored: hostname, user name, MAC address, serial numbers, GPU UUID; your home directory is shown as ~.
"""


def run_checks(*, system: str | None = None, machine: str | None = None, runner: Runner = default_runner, data_dir: str | None = None,
               disk_free: Callable[[str], int] = default_disk_free, ram_bytes: Callable[[], int | None] = default_ram_bytes,
               cpus: int | None = None, which: Callable[[str], str | None] = shutil.which, exists: Callable[[str], bool] = os.path.exists,
               battery: Callable[[], dict[str, Any] | None] = default_battery, project_url: str | None = None,
               resolver: Callable[[str], list[str]] = default_resolver, fetcher: Fetcher = default_fetcher, scrub: Scrubber | None = None,
               now: datetime | None = None, py_version: tuple[int, int, int] | None = None,
               allow_cpu_solver: bool = False) -> list[Check]:
    scrub = scrub or Scrubber()
    target = data_dir or detect_data_dir(exists) or str(Path.home())
    checks = [
        check_os(system or platform.system(), machine or platform.machine()),
        check_python(py_version or sys.version_info[:3]),
        check_gpu(runner, allow_cpu_solver=allow_cpu_solver),
        check_disk(target, disk_free, scrub),
        check_ram_cpu(ram_bytes(), cpus if cpus is not None else os.cpu_count()),
        check_boinc(which, exists),
        check_power(battery, now or datetime.now(timezone.utc)),
    ]
    if project_url:
        checks.append(check_network(project_url, resolver, fetcher))
    return checks


def build_report(checks: Sequence[Check], scrub: Scrubber) -> dict[str, Any]:
    rep = {"schema": SCHEMA, "verdict": verdict_of(checks), "checks": [c.to_dict() for c in checks]}
    return json.loads(scrub(json.dumps(rep)))


def render_table(report: dict[str, Any], color: bool = False) -> str:
    tags = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}
    cols = {"ok": "32", "warn": "33", "fail": "31"}
    lines = ["HiveMind volunteer preflight", "=" * 28, ""]
    for c in report["checks"]:
        tag = f"[{tags[c['status']]:^4}]"
        if color:
            tag = f"\x1b[{cols[c['status']]}m{tag}\x1b[0m"
        lines.append(f"{tag} {c['title']}")
        lines.append(f"       {c['reason']}")
        if c["what_to_do"]:
            lines.append(f"       -> {c['what_to_do']}")
    v = report["verdict"]
    lines += ["", f"Verdict: {v.replace('_', ' ').upper()}",
              {"ready": "This machine looks ready.", "ready_with_warnings": "You can join, but read the warnings above first.",
               "not_ready": "This machine cannot join yet: fix the FAIL lines above."}[v]]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check this machine before joining HiveMind as a BOINC volunteer. Installs nothing; no network unless --project-url.")
    ap.add_argument("--data-dir", help="directory BOINC will use (default: the BOINC data dir if found, else your home directory)")
    ap.add_argument("--project-url", help="optional https URL of the project; enables the only network access of this script")
    ap.add_argument("--allow-cpu-solver", action="store_true", help="validate CPU/BLS solver mode (plan class cpu_hivemind_solver) when nvidia-smi is absent")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--no-color", action="store_true", help="plain output (default is already plain unless stdout is a terminal)")
    ap.add_argument("--explain", action="store_true", help="print what each check reads, then exit")
    args = ap.parse_args(argv)
    if args.explain:
        print(EXPLAIN)
        return 0
    scrub = Scrubber()
    checks = run_checks(data_dir=args.data_dir, project_url=args.project_url, scrub=scrub, allow_cpu_solver=args.allow_cpu_solver)
    report = build_report(checks, scrub)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_table(report, color=sys.stdout.isatty() and not args.no_color and os.name != "nt"))
    return EXIT[report["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
