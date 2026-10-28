"""The BOINC daemon supervisor: run the real script against a fake database and fake daemons.

Every fake daemon carries a per-test tag that the script is told to require (HM_PROC_TAG), so the tests also pass on the T4 itself,
where the real daemons are running.

Real incident behind it: after a Spot preemption `bin/start` ran before the database socket answered, the feeder and transitioner exited, and
nothing started them again, so no work was served. These tests cover: all running, restart of a dead daemon, a database that comes up late,
a database that never comes up (daemons must NOT be started into it), a daemon that never starts, and the watchdog's guard for a stopped unit.
"""

from __future__ import annotations

import os
import signal
import subprocess
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy" / "t4_node" / "node" / "bin" / "boinc_ensure_daemons.sh"
DAEMONS = [  # argv as `ps` shows the real ones
    "feeder -d 3", "transitioner -d 3", "file_deleter -d 3",
    "script_validator -d 3 --app hivemind_qlora_moe --init_script x", "script_assimilator -d 3 --app hivemind_qlora_moe --script x",
    "script_validator -d 3 --app hivemind_moe_infer --init_script x", "script_assimilator -d 3 --app hivemind_moe_infer --script x",
]


class Box:
    def __init__(self, tmp: Path) -> None:
        self.tmp, self.tag = tmp, uuid.uuid4().hex
        self.starts = tmp / "starts.log"
        self.admin_calls = tmp / "admin.log"
        self.pids: list[int] = []

    def spawn(self, argv0: str) -> None:
        p = subprocess.Popen(["bash", "-c", f'exec -a "{argv0} {self.tag}" sleep 60'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.pids.append(p.pid)

    def spawn_all(self, skip: tuple[str, ...] = ()) -> None:
        for d in DAEMONS:
            if not d.startswith(skip):
                self.spawn(d)

    def admin(self, fail_first: int) -> Path:
        """A fake mariadb-admin that fails (exit 1) for its first `fail_first` calls, then succeeds."""
        f = self.tmp / "admin.sh"
        f.write_text(f'#!/bin/bash\necho x >> {self.admin_calls}\nn=$(wc -l < {self.admin_calls})\n[ "$n" -le {fail_first} ] && exit 1\nexit 0\n')
        f.chmod(0o755)
        return f

    def env_script(self, spawns: tuple[str, ...]) -> Path:
        """A fake boinc_project_env.sh: records each `bin/start` and, if `spawns` is non-empty, starts those daemons."""
        f = self.tmp / "env.sh"
        lines = "".join(f'(exec -a "{d} {self.tag}" sleep 60 >/dev/null 2>&1 &)\n' for d in spawns)
        f.write_text(f"#!/bin/bash\necho \"$@\" >> {self.starts}\n{lines}")
        f.chmod(0o755)
        return f

    def run(self, *args: str, admin: Path, env_sh: Path, extra: dict[str, str] | None = None, timeout: int = 60) -> subprocess.CompletedProcess:
        sock = self.tmp / "mysqld.sock"
        sock.touch()
        # the script tests `-S` (a socket), so make a real one
        import socket

        sock.unlink()
        s = socket.socket(socket.AF_UNIX)
        s.bind(str(sock))
        env = dict(os.environ, HM_DB_SOCK=str(sock), HM_MARIADB_ADMIN=str(admin), HM_ENV_SCRIPT=str(env_sh), HM_STEP_S="1", HM_DB_WAIT_S="3",
                   HM_SETTLE_S="0", HM_MAX_TRIES="3", HM_PROC_TAG=self.tag, **(extra or {}))
        try:
            return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=timeout)
        finally:
            s.close()

    def n_starts(self) -> int:
        return len(self.starts.read_text().splitlines()) if self.starts.exists() else 0

    def cleanup(self) -> None:
        out = subprocess.run(["pgrep", "-f", self.tag], capture_output=True, text=True).stdout.split()
        for pid in [*map(int, out), *self.pids]:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.fixture
def box(tmp_path: Path):
    b = Box(tmp_path)
    yield b
    b.cleanup()


def test_nothing_is_started_when_every_daemon_is_running(box: Box) -> None:
    box.spawn_all()
    r = box.run("--verify", admin=box.admin(0), env_sh=box.env_script(()))
    assert r.returncode == 0 and "all required daemons are running" in r.stdout and box.n_starts() == 0


def test_a_dead_feeder_and_transitioner_are_started_again(box: Box) -> None:
    box.spawn_all(skip=("feeder", "transitioner"))  # exactly the state found on the T4
    r = box.run("--verify", admin=box.admin(0), env_sh=box.env_script(("feeder -d 3", "transitioner -d 3")))
    assert r.returncode == 0 and box.n_starts() == 1
    assert "not running: ^feeder ^transitioner" in " ".join(r.stdout.split())


def test_a_database_that_comes_up_late_delays_the_start_instead_of_losing_it(box: Box) -> None:
    box.spawn_all(skip=("feeder", "transitioner"))
    r = box.run("--verify", admin=box.admin(fail_first=2), env_sh=box.env_script(("feeder -d 3", "transitioner -d 3")))
    assert r.returncode == 0 and box.n_starts() == 1
    assert len(box.admin_calls.read_text().splitlines()) >= 3  # polled until it answered, only then started


def test_daemons_are_never_started_into_a_database_that_does_not_answer(box: Box) -> None:
    box.spawn_all(skip=("feeder", "transitioner"))
    r = box.run("--verify", admin=box.admin(fail_first=10**6), env_sh=box.env_script(("feeder -d 3", "transitioner -d 3")))
    assert r.returncode == 2 and box.n_starts() == 0 and "not starting daemons into a dead database" in r.stdout


def test_a_daemon_that_never_starts_is_retried_then_reported(box: Box) -> None:
    box.spawn_all(skip=("transitioner",))
    r = box.run("--verify", admin=box.admin(0), env_sh=box.env_script(()))  # bin/start spawns nothing
    assert r.returncode == 1 and box.n_starts() == 3 and "still not running after 3 attempts" in r.stdout and "transitioner" in r.stdout


def test_a_validator_for_the_big_app_does_not_satisfy_the_standard_app_check(box: Box) -> None:
    box.spawn_all(skip=("script_validator -d 3 --app hivemind_qlora_moe",))
    box.spawn("script_validator -d 3 --app hivemind_qlora_moe_big --init_script x")  # same prefix, different app
    r = box.run("--verify", admin=box.admin(0), env_sh=box.env_script(()))
    assert r.returncode == 1 and "script_validator" in r.stdout and "hivemind_qlora_moe" in r.stdout


def test_wait_db_succeeds_and_times_out(box: Box) -> None:
    assert box.run("--wait-db", "3", admin=box.admin(1), env_sh=box.env_script(())).returncode == 0
    (box.tmp / "admin.log").unlink()
    assert box.run("--wait-db", "2", admin=box.admin(10**6), env_sh=box.env_script(())).returncode == 1


def test_watchdog_leaves_a_deliberately_stopped_unit_alone(box: Box) -> None:
    r = box.run("--watchdog", admin=box.admin(0), env_sh=box.env_script(("feeder -d 3",)), extra={"HM_UNIT_ACTIVE_CMD": "false"})
    assert r.returncode == 0 and box.n_starts() == 0 and "nothing to do" in r.stdout


def test_watchdog_restarts_a_dead_daemon_when_the_unit_is_active(box: Box) -> None:
    box.spawn_all(skip=("transitioner",))
    r = box.run("--watchdog", admin=box.admin(0), env_sh=box.env_script(("transitioner -d 3",)), extra={"HM_UNIT_ACTIVE_CMD": "true"})
    assert r.returncode == 0 and box.n_starts() == 1


def test_watchdog_with_the_database_down_only_warns(box: Box) -> None:
    box.spawn_all(skip=("feeder",))
    r = box.run("--watchdog", admin=box.admin(10**6), env_sh=box.env_script(("feeder -d 3",)), extra={"HM_UNIT_ACTIVE_CMD": "true"})
    assert r.returncode == 0 and box.n_starts() == 0 and "dead database" in r.stdout


def test_unknown_mode_is_a_usage_error(box: Box) -> None:
    assert box.run("--bogus", admin=box.admin(0), env_sh=box.env_script(())).returncode == 64


def test_unit_files_wire_the_checks_in() -> None:
    node = ROOT / "deploy" / "t4_node" / "node"
    db, dm = (node / "hivemind-boinc-db.service").read_text(), (node / "hivemind-boinc-daemons.service").read_text()
    assert "ExecStartPost=/opt/hivemind/t4/bin/boinc_ensure_daemons.sh --wait-db" in db
    assert "ExecStartPre=/opt/hivemind/t4/bin/boinc_ensure_daemons.sh --wait-db" in dm and "ExecStartPost=/opt/hivemind/t4/bin/boinc_ensure_daemons.sh --verify" in dm
    assert "Restart=on-failure" in dm
    assert "--watchdog" in (node / "hivemind-boinc-watchdog.service").read_text()
    assert "OnUnitActiveSec=2min" in (node / "hivemind-boinc-watchdog.timer").read_text()
    inst = (ROOT / "deploy" / "t4_node" / "install_node.sh").read_text()
    assert "node/*.timer" in inst and "hivemind-boinc-watchdog.timer" in inst


def test_untagged_processes_do_not_satisfy_the_check(box: Box) -> None:
    """Hermeticity: daemons with the right name but without this test's tag (as on a host running the real ones) are not counted."""
    for d in DAEMONS:
        p = subprocess.Popen(["bash", "-c", f'exec -a "{d} other-{box.tag[:8]}" sleep 60'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        box.pids.append(p.pid)
    r = box.run("--verify", admin=box.admin(0), env_sh=box.env_script(()))
    assert r.returncode == 1 and box.n_starts() == 3
    assert "^feeder" in r.stdout and "still not running after 3 attempts" in r.stdout
