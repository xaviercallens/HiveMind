#!/usr/bin/env python3
"""Telemetry + watchdog for the T4 node. Standard library only.

Every ``--interval-s`` it records one JSON line (GPU, host, nginx in-flight, unit states, gateway-visible tiers, governor
state, QLoRA progress from the BOINC ledger) and, when run as root, restarts a managed unit that is enabled but not active
(at most 3 restarts per unit per 10 minutes). It polls the GCE metadata server every few seconds for a preemption notice and
writes an urgent event line plus a ``latest.json`` snapshot the moment one appears (a Spot VM gets ~30 s).

Files (``--out-dir``): ``t4_node.jsonl`` (rotated at 20 MB, 5 kept), ``latest.json``, ``events.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from collections import Counter, defaultdict, deque
from pathlib import Path

GATEWAY = "https://hivemind-openrouter-1003063861791.us-central1.run.app"
# restarted by the watchdog when enabled but not active
MANAGED = [
    "ollama", "nginx", "hivemind-live-shield-user", "hivemind-live-agent-user", "hivemind-boinc-db", "hivemind-boinc-apache",
    "hivemind-boinc-daemons", "hivemind-boinc-client", "hivemind-t4-governor", "hivemind-t4-workgen",
]
# reported only: these still run as session-started processes until the owner cuts them over to the units, and a unit that
# races a live process for its port would only fail; pass --manage-legacy after the cutover
LEGACY = ["hivemind-live-bee-llama", "hivemind-live-shield-bee", "hivemind-live-agent-bee", "hivemind-live-shield-coder", "hivemind-live-agent-coder"]


def sh(cmd: list[str], timeout: int = 10) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def http(url: str, headers: dict | None = None, timeout: float = 4) -> tuple[int, str]:
    try:
        req = urllib.request.Request(url, headers=headers or {})
        r = urllib.request.urlopen(req, timeout=timeout)
        return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception:
        return 0, ""


def metadata(path: str) -> str:
    return http(f"http://metadata.google.internal/computeMetadata/v1/{path}", {"Metadata-Flavor": "Google"}, 2)[1].strip()


def unit_states() -> dict[str, dict]:
    known = set(sh(["systemctl", "list-unit-files", "--no-legend", "--type=service"]).replace(".service", "").split())
    out = {}
    for u in MANAGED + LEGACY:
        if u in known:
            out[u] = {"active": sh(["systemctl", "is-active", u]), "enabled": sh(["systemctl", "is-enabled", u])}
    return out


def gpu() -> dict:
    q = sh(["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu,power.draw", "--format=csv,noheader,nounits"]).split(",")
    apps = [ln.split(",") for ln in sh(["nvidia-smi", "--query-compute-apps=pid,name,used_memory", "--format=csv,noheader,nounits"]).splitlines() if ln.strip()]
    if len(q) < 5:
        return {"error": "nvidia-smi"}
    return {"used_mib": int(q[0]), "total_mib": int(q[1]), "temp_c": int(q[2]), "util_pct": int(q[3]), "power_w": float(q[4]),
            "procs": [{"pid": a[0].strip(), "name": Path(a[1].strip()).name, "mib": int(a[2])} for a in apps]}


def nginx_counts() -> dict:
    code, txt = http("http://127.0.0.1:8091/nginx_status", timeout=2)
    if code != 200:
        return {"ok": False}
    nums = [int(x) for x in txt.replace(":", " ").split() if x.isdigit()]
    # layout: Active connections N / server accepts handled requests / A H R / Reading R Writing W Waiting K
    return {"ok": True, "active": nums[0], "requests_total": nums[3], "writing": nums[5], "in_flight": max(nums[5] - 1, 0)}


def gateway_view() -> dict:
    code, body = http(f"{GATEWAY}/v1/models", timeout=8)
    tiers = sorted(m["id"] for m in json.loads(body).get("data", [])) if code == 200 else []
    return {"models_http": code, "live_tiers": tiers, "health_http": http(f"{GATEWAY}/health", timeout=8)[0]}


def policy_drift(expected: str | None, actual: str | None, dropins: list[str]) -> dict:
    """The owner's rule: the governor changes only through the reviewed repo template and the installer. Compare what the installer recorded
    with what is running, and list systemd drop-in overrides (which silently beat the unit file). Report only; never fix automatically."""
    return {"expected_args": expected, "actual_args": actual, "args_match": expected is not None and actual == expected,
            "dropins": sorted(dropins), "drift": (expected is not None and actual != expected) or bool(dropins)}


def code_drift(manifest_text: str, read_hash) -> list[str]:
    """Files whose current sha256 differs from the manifest the installer wrote (``sha256sum`` format: '<hash>  <relative path>').
    ``read_hash(relpath)`` returns the current hash, or None when the file is missing (reported as drift too)."""
    drift = []
    for ln in manifest_text.splitlines():
        parts = ln.split()
        if len(parts) == 2 and read_hash(parts[1]) != parts[0]:
            drift.append(parts[1])
    return sorted(drift)


def governor_policy() -> dict:
    import glob
    import hashlib

    exp = Path("/opt/hivemind/t4/expected_governor_args.txt")
    expected = exp.read_text().strip() if exp.exists() else None
    actual = None
    pid = sh(["systemctl", "show", "-p", "MainPID", "--value", "hivemind-t4-governor.service"])
    if pid.isdigit() and int(pid) > 0:
        try:
            actual = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode().split("t4_governor.py", 1)[1].strip()
        except (OSError, IndexError):
            pass
    out = policy_drift(expected, actual, glob.glob("/etc/systemd/system/hivemind-*.service.d/*"))

    def current(rel: str) -> str | None:
        try:
            return hashlib.sha256((Path("/opt/hivemind/t4") / rel).read_bytes()).hexdigest()
        except OSError:
            return None

    man = Path("/opt/hivemind/t4/expected_code_hashes.txt")
    out["code_drift"] = code_drift(man.read_text(), current) if man.exists() else []
    out["drift"] = bool(out["drift"] or out["code_drift"])  # an edited governor script defeats the arguments just as well as a drop-in
    return out


class ChatLog:
    """Counts completed chat-completion requests from nginx's access log since the previous poll (the monitor runs as root; the log is
    adm-only). In-flight sampling every 15 s misses sub-second requests; this is the real request count. It does NOT give latency
    (the default log format has no request time). Handles rotation (new inode or a shorter file)."""

    PAT = None

    def __init__(self, path: Path = Path("/var/log/nginx/access.log")) -> None:
        import re

        self.path, self.inode, self.offset = path, None, None
        ChatLog.PAT = ChatLog.PAT or re.compile(r'"POST (\S+)[^"]*" (\d{3}) ')

    def poll(self) -> dict:
        try:
            st_ = self.path.stat()
        except OSError:
            return {"ok": False}
        if self.inode != st_.st_ino or (self.offset is not None and st_.st_size < self.offset):
            first = self.inode is None
            self.inode, self.offset = st_.st_ino, (st_.st_size if first else 0)  # first poll starts "now"; after rotation read the new file from 0
            if first:
                return {"ok": True, "n": 0, "by": {}}
        by: Counter = Counter()
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            data = f.read()
            self.offset = f.tell()
        for ln in data.decode(errors="replace").splitlines():
            m = ChatLog.PAT.search(ln)
            if m and "chat/completions" in m.group(1):
                p = m.group(1)
                route = "bee" if p.startswith("/bee/") else "u" if p.startswith("/u/") else "coder"
                by[f"{route}:{m.group(2)}"] += 1
        return {"ok": True, "n": sum(by.values()), "by": dict(by)}


def progress(ledger: Path) -> dict:
    out = {"assimilated": 0, "last": None}
    f = ledger / "assimilated.jsonl"
    if f.exists():
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        out["assimilated"] = len(rows)
        keys = ("wu_id", "app", "domain", "adapter_sha256", "eval_loss_before", "eval_loss_after", "eval_by_domain_after")
        out["last"] = {k: rows[-1].get(k) for k in keys} if rows else None
        train = [r for r in rows if r.get("app") == "hivemind_qlora_moe"]
        out["train_rounds"] = len(train)
        out["mean_eval_loss_trend"] = [round(r["eval_loss_after"], 4) for r in train[-10:] if r.get("eval_loss_after") is not None]
    return out


def rotate(path: Path, limit: int = 20_000_000, keep: int = 5) -> None:
    if path.exists() and path.stat().st_size > limit:
        for i in range(keep - 1, 0, -1):
            if (path.parent / f"{path.name}.{i}").exists():
                os.replace(path.parent / f"{path.name}.{i}", path.parent / f"{path.name}.{i + 1}")
        os.replace(path, path.parent / f"{path.name}.1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/hivemind_telemetry"))
    ap.add_argument("--governor-state", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/hivemind_telemetry/governor_state.json"))
    ap.add_argument("--ledger", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/boinc/stack/project/hmproj/hivemind_ledger"))
    ap.add_argument("--interval-s", type=float, default=15.0)
    ap.add_argument("--no-restart", action="store_true")
    ap.add_argument("--manage-legacy", action="store_true", help="also restart the bee/coder units (only after cutting the session-started processes over)")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    can_restart = os.geteuid() == 0 and not a.no_restart
    restarts: dict[str, deque] = defaultdict(deque)
    last_full, preempt_logged, prev_req, prev_t = 0.0, False, None, None
    chat_log = ChatLog()
    last_drift = None

    def event(kind: str, **kw) -> None:
        with open(a.out_dir / "events.jsonl", "a") as f:
            f.write(json.dumps({"t": round(time.time(), 2), "event": kind, **kw}) + "\n")

    event("monitor_start", restart_enabled=can_restart, boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())
    while True:
        now = time.time()
        pre = metadata("instance/preempted") == "TRUE" or metadata("instance/maintenance-event") not in ("", "NONE")
        if pre and not preempt_logged:
            preempt_logged = True
            event("PREEMPTION_NOTICE", maintenance=metadata("instance/maintenance-event"), gpu=gpu(), nginx=nginx_counts())
        if now - last_full >= a.interval_s:
            last_full = now
            units = unit_states()
            ng = nginx_counts()
            rate = None
            if ng.get("ok") and prev_req is not None and prev_t:
                rate = round((ng["requests_total"] - prev_req) / max(now - prev_t, 1e-6), 3)
            if ng.get("ok"):
                prev_req, prev_t = ng["requests_total"], now
            snap = {"t": round(now, 2), "uptime_s": float(Path("/proc/uptime").read_text().split()[0]), "loadavg": os.getloadavg(),
                    "disk_free_gb": {p: round(os.statvfs(p).f_bavail * os.statvfs(p).f_frsize / 1e9, 1) for p in ("/", "/mnt/disks/disk-socrateai-local-1")},
                    "gpu": gpu(), "nginx": {**ng, "req_per_s": rate}, "chat_completions_since_last": chat_log.poll(), "governor_policy": (pol := governor_policy()), "units": units, "gateway": gateway_view(),
                    "governor": json.loads(a.governor_state.read_text()) if a.governor_state.exists() else None,
                    "qlora_progress": progress(a.ledger), "preemption_notice": pre}
            if pol["drift"] != last_drift:  # one event per change, not one per snapshot
                last_drift = pol["drift"]
                event("GOVERNOR_POLICY_DRIFT" if pol["drift"] else "governor_policy_ok", **{k: pol[k] for k in ("expected_args", "actual_args", "dropins", "code_drift")})
            rotate(a.out_dir / "t4_node.jsonl")
            with open(a.out_dir / "t4_node.jsonl", "a") as f:
                f.write(json.dumps(snap) + "\n")
            (a.out_dir / "latest.json").write_text(json.dumps(snap, indent=1))
            for u, s in units.items():
                if u in LEGACY and not a.manage_legacy:
                    continue
                if s["enabled"] == "enabled" and s["active"] not in ("active", "activating") and can_restart:
                    q = restarts[u]
                    while q and now - q[0] > 600:
                        q.popleft()
                    if len(q) < 3:
                        q.append(now)
                        sh(["systemctl", "restart", u], 60)
                        event("unit_restart", unit=u, was=s["active"], after=sh(["systemctl", "is-active", u]))
                    else:
                        event("unit_restart_suppressed", unit=u, reason="3 restarts in 10 min")
            if a.once:
                return 0
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
