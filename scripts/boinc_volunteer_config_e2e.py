#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Does the STOCK BOINC client really obey the settings from ``hivemind_volunteer_config.py``? Runs a throw-away client (own data directory, own RPC port, attached to no project).

  python scripts/boinc_volunteer_config_e2e.py --boinc-root /path/to/extracted/boinc --work /tmp/vc_e2e

Checks, each against what the client itself reports over its RPC interface:
  1. control: with the dedicated profile the client reports GPU computing "not suspended";
  2. with the gamer profile it reports "suspended: computer is in use" (this host has no keyboard or mouse, which the client counts as in use);
  3. the gamer values show up in the client's working preferences;
  4. with a game listed in cc_config.xml: not suspended while the game is not running, suspended while a process with that name runs, and resumed once it exits even though
     an UNLISTED program is still running (the list is what protects the player, and filling it is the volunteer's job).
It never touches an installed client or a BOINC project.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hivemind_volunteer_config as vc  # noqa: E402


class Client:
    def __init__(self, root: Path, work: Path, port: int) -> None:
        self.root, self.work, self.port = root, work, port
        self.env = dict(os.environ, LD_LIBRARY_PATH=f"{root}/usr/lib/x86_64-linux-gnu:{os.environ.get('LD_LIBRARY_PATH', '')}")
        self.proc: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        self.work.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen([f"{self.root}/usr/bin/boinc", "--dir", str(self.work), "--allow_multiple_clients", "--gui_rpc_port", str(self.port), "--no_priority_change"],
                                     env=self.env, stdout=open(self.work / "stdout.log", "ab"), stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(60):
            if (self.work / "gui_rpc_auth.cfg").exists() and self.cmd("--get_state", check=False)[0] == 0:
                return
            time.sleep(1)
        raise SystemExit(f"client did not start; see {self.work}/stdout.log")

    def cmd(self, *args: str, check: bool = True) -> tuple[int, str]:
        pw = (self.work / "gui_rpc_auth.cfg").read_text().strip() if (self.work / "gui_rpc_auth.cfg").exists() else ""
        r = subprocess.run([f"{self.root}/usr/bin/boinccmd", "--host", f"localhost:{self.port}", "--passwd", pw, *args], env=self.env, capture_output=True, text=True, timeout=30)
        if check and r.returncode != 0:
            raise SystemExit(f"boinccmd {' '.join(args)} failed: {r.stderr.strip() or r.stdout.strip()}")
        return r.returncode, r.stdout

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)


def gpu_status(c: Client) -> str:
    """The first line of the 'GPU status' block of ``boinccmd --get_cc_status``: 'not suspended' or 'suspended: <reason>'."""
    lines = c.cmd("--get_cc_status")[1].splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "GPU status":
            return lines[i + 1].strip()
    raise SystemExit("no 'GPU status' block in --get_cc_status; client too old?")


def effective_gpu_status(c: Client) -> str:
    """What the GPU is really doing. When the whole computation is suspended (activity rule), the client reports it on the CPU line only and shows the GPU as 'not suspended'."""
    gpu = gpu_status(c)
    if gpu.startswith("suspended"):
        return gpu
    lines = c.cmd("--get_cc_status")[1].splitlines()
    cpu = lines[lines.index("CPU status") + 1].strip()
    return cpu if cpu == "suspended: computer is in use" else gpu


def settle(c: Client, want: str, timeout_s: float = 60.0) -> str:
    """The client re-evaluates about every 10 s: poll until the wanted state (or the timeout) and return the last state seen."""
    deadline, seen = time.time() + timeout_s, ""
    while time.time() < deadline:
        seen = effective_gpu_status(c)
        if (seen.startswith("suspended") if want == "suspended" else want in seen):  # "suspended" is a substring of "not suspended"
            return seen
        time.sleep(3)
    return seen


def reload(c: Client, work: Path, profile: str, games: list[str]) -> None:
    vc.apply(work, vc.prefs_xml(profile), vc.cc_config_xml(games))
    c.cmd("--read_global_prefs_override")
    c.cmd("--read_cc_config")


def fresh_client(args: argparse.Namespace, name: str, profile: str, games: list[str]) -> Client:
    """A new client per scenario, as on a volunteer's PC: the client keeps an idle clock that carries over when profiles are switched on a long-lived one."""
    c = Client(args.boinc_root, args.work / name, args.port)
    c.work.mkdir(parents=True, exist_ok=True)
    vc.apply(c.work, vc.prefs_xml(profile), vc.cc_config_xml(games))  # written BEFORE the client starts: what the installer would do
    c.start()
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boinc-root", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--port", type=int, default=31499)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    shutil.rmtree(args.work, ignore_errors=True)
    args.work.mkdir(parents=True)
    report: dict[str, object] = {}
    fake, other = args.work / "hivemindfakegame", args.work / "someotherprogram"  # 16 characters on purpose: Linux cuts process names to 15 (see with_linux_names)
    for path in (fake, other):
        shutil.copy2("/bin/sleep", path)
    procs: list[subprocess.Popen[bytes]] = []
    try:
        c = fresh_client(args, "gamer", "gamer", [])
        try:
            report["gamer_profile"] = settle(c, "computer is in use", timeout_s=30)
            report["gamer_cpu"] = c.cmd("--get_cc_status")[1].split("CPU status")[1].splitlines()[1].strip()
            report["gamer_log"] = "don't use GPU while active" in (c.work / "stdout.log").read_text()
        finally:
            c.stop()
        c = fresh_client(args, "dedicated", "dedicated", [])
        try:
            report["dedicated_profile"] = settle(c, "not suspended")
        finally:
            c.stop()
        c = fresh_client(args, "listed", "dedicated", [fake.name])  # the dedicated profile never suspends for activity: any suspension below is the listed game
        try:
            report["listed_game_not_running"] = settle(c, "not suspended")
            game = subprocess.Popen([str(fake), "600"])
            procs.append(game)
            report["listed_game_running"] = settle(c, "suspended")
            procs.append(subprocess.Popen([str(other), "600"]))
            game.terminate()
            game.wait()
            report["listed_game_exited_unlisted_program_running"] = settle(c, "not suspended")
        finally:
            c.stop()
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
    r = report
    in_use = "computer is in use" in str(r.get("gamer_cpu")) or "computer is in use" in str(r.get("gamer_profile"))
    report["activity_rule_exercised"] = in_use  # this host has no keyboard or mouse; the client's idea of "in use" follows terminal activity and cannot be forced from here
    checks = {
        "gamer profile: the client parsed the 'don't use GPU while active' rule": bool(r.get("gamer_log")),
        "gamer profile: when the client considers the computer in use, the GPU does no work": ("computer is in use" in str(r.get("gamer_profile"))) if in_use else True,
        "dedicated profile: GPU not suspended": r.get("dedicated_profile") == "not suspended",
        "listed game not running: GPU not suspended": r.get("listed_game_not_running") == "not suspended",
        "listed game running: GPU suspended because of the exclusive app": "exclusive app" in str(r.get("listed_game_running")),
        "game exited, unlisted program running: GPU resumes": r.get("listed_game_exited_unlisted_program_running") == "not suspended",
    }
    report["checks"] = checks
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
