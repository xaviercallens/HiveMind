# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Run ONE binary (the Solver Bee CLI) under the same isolation primitives as the E15 gate (stdlib only).

The design follows ``scripts/e15_gate.py`` (``sandbox_run`` / ``LAUNCHER_SRC``), which is NOT modified or imported: ``unshare --user --map-root-user --mount
--net --pid --ipc --uts --fork --kill-child`` starts a small launcher (system ``python3 -I -S -B``). As the namespace root the launcher
  1. reads the binary (and checks its sha256) BEFORE hiding the host, so a binary under /home still works,
  2. mounts propagation private and puts a fresh tmpfs over /home, /root, /tmp, /var/tmp, /run, /dev/shm, /mnt, /media, /srv and the real home / cwd,
     a minimal /dev and a fresh /proc of the new pid namespace,
  3. writes the binary copy to the tmpfs work dir, drops ALL capabilities, sets no_new_privs and non-dumpable, and verifies that from /proc/self/status,
  4. starts the binary: scrubbed environment (PATH only), stdin from a sealed memfd (no pipe, no deadlock), own session, RLIMIT_AS / CPU / NPROC / FSIZE /
     NOFILE / CORE,
  5. enforces the wall clock and the stdout cap (kill of the whole process group), keeps a bounded stderr tail, reports ONE json line on its own stdout.
The network namespace has only a down ``lo``. The pid namespace dies with the launcher, so strays are killed.

KNOWN CAVEATS (same as E15): no seccomp filter, shared host kernel, no cgroup memory / pids limit (RLIMIT_AS per process; NPROC is counted per real uid),
and the work tmpfs is writable (bounded by ``tmpfs_mb``) although only the binary copy is meant to be there. The "memory" limit label is a heuristic
(the kernel does not report an RLIMIT_AS hit): it is set when stderr shows an allocation failure.

If user namespaces are unavailable the runner REFUSES (``allow_unsandboxed=False`` default). With ``allow_unsandboxed=True`` it runs the SAME launcher without
namespaces (rlimits, scrubbed env, process-group kill, no_new_privs only) and every outcome says ``sandbox='none'``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

HIDE_DIRS = ("/home", "/root", "/tmp", "/var/tmp", "/run", "/dev/shm", "/mnt", "/media", "/srv")
SANDBOX_MARK = "SOLVERBEE_SANDBOX_RESULT "
HOST_GRACE_S = 20.0
SANDBOX_NAMESPACE, SANDBOX_NONE = "namespace", "none"
STDIN_LIMIT_BYTES = 64 * 1024
STDOUT_LIMIT_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class Limits:
    timeout_s: float = 30.0
    mem_mb: int = 2048  # RLIMIT_AS
    nproc: int = 128
    fsize_bytes: int = 16 * 1024 * 1024
    nofile: int = 64
    stdout_limit: int = STDOUT_LIMIT_BYTES
    stdin_limit: int = STDIN_LIMIT_BYTES
    stderr_tail: int = 4096
    tmpfs_mb: int = 64  # room for the binary copy is added on top
    allow_unsandboxed: bool = False
    force_none: bool = False  # run WITHOUT namespaces even when they are available (the MCP server's --allow-unsandboxed mode); every outcome says 'none'

    def __post_init__(self) -> None:
        if isinstance(self.timeout_s, bool) or not isinstance(self.timeout_s, (int, float)) or not math.isfinite(float(self.timeout_s)) or float(self.timeout_s) <= 0.0:
            raise ValueError("timeout_s must be a strictly positive finite float")
        for attr in ("mem_mb", "nproc", "fsize_bytes", "nofile", "stdout_limit", "stderr_tail", "tmpfs_mb"):
            val = getattr(self, attr)
            if isinstance(val, bool) or not isinstance(val, int) or val < 1:
                raise ValueError(f"{attr} must be an integer >= 1")
        if isinstance(self.stdin_limit, bool) or not isinstance(self.stdin_limit, int) or self.stdin_limit < 0:
            raise ValueError("stdin_limit must be an integer >= 0")
        if not isinstance(self.allow_unsandboxed, bool) or not isinstance(self.force_none, bool):
            raise ValueError("allow_unsandboxed and force_none must be booleans")


@dataclass
class SandboxOutcome:
    ok: bool  # the binary was started and a report was obtained (NOT "exit code 0")
    error: str = ""  # why not ok (refusal, unavailable sandbox, launcher failure)
    rc: Optional[int] = None  # negative = killed by that signal
    stdout: bytes = b""
    stderr_tail: str = ""
    wall_s: float = 0.0
    limit: Optional[str] = None  # timeout | stdout | cpu | memory | None
    sandbox: str = SANDBOX_NONE  # namespace | none (what was ACTIVE for this run)
    refused: bool = False  # refused before running anything (sandbox unavailable / bad input)

    def __post_init__(self) -> None:
        for field_name in ("ok", "refused"):
            val = getattr(self, field_name)
            if not isinstance(val, bool):
                raise ValueError(f"{field_name} must be a boolean")
        for field_name in ("error", "stderr_tail"):
            val = getattr(self, field_name)
            if not isinstance(val, str):
                raise ValueError(f"{field_name} must be a string")
        if self.rc is not None and (isinstance(self.rc, bool) or not isinstance(self.rc, int)):
            raise ValueError("rc must be an integer or None")
        if not isinstance(self.stdout, bytes):
            raise ValueError("stdout must be bytes")
        if isinstance(self.wall_s, bool) or not isinstance(self.wall_s, (int, float)) or not math.isfinite(float(self.wall_s)) or float(self.wall_s) < 0.0:
            raise ValueError("wall_s must be a finite non-negative number")
        if self.limit is not None and (not isinstance(self.limit, str) or not self.limit.strip()):
            raise ValueError("limit must be a non-empty string or None")
        if self.sandbox not in (SANDBOX_NAMESPACE, SANDBOX_NONE):
            raise ValueError(f"sandbox must be {SANDBOX_NAMESPACE!r} or {SANDBOX_NONE!r}")


LAUNCHER_SRC = r'''
import base64, ctypes, hashlib, json, os, resource, selectors, signal, subprocess, sys, time

MARK = @MARK@

def emit(d):
    sys.stdout.write("\n" + MARK + json.dumps(d) + "\n")
    sys.stdout.flush()

def main():
    req = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    ns = req["ns"]
    with open(req["binary"], "rb") as f:  # read before the host is hidden
        blob = f.read()
    if hashlib.sha256(blob).hexdigest() != req["binary_sha256"]:
        raise RuntimeError("binary changed between hashing and start")
    stdin_bytes = base64.b64decode(req["stdin_b64"])
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    work = req["workdir"]
    if ns:
        libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_ulong, ctypes.c_char_p]

        def mount(src, tgt, fs, flags, data):
            if libc.mount(src.encode() if src else None, tgt.encode(), fs.encode() if fs else None, flags, data.encode() if data else None) != 0:
                e = ctypes.get_errno()
                raise OSError(e, "mount %s: %s" % (tgt, os.strerror(e)))

        MS_NOSUID, MS_NODEV, MS_NOEXEC, MS_REC, MS_PRIVATE, MS_BIND = 2, 4, 8, 16384, 1 << 18, 4096
        mount(None, "/", None, MS_REC | MS_PRIVATE, None)
        for d in req["hide"]:
            if os.path.isdir(d) and not os.path.islink(d):
                if d == "/tmp":
                    mount("tmpfs", d, "tmpfs", MS_NOSUID | MS_NODEV, "size=%dm,mode=1777" % req["tmpfs_mb"])
                else:
                    mount("tmpfs", d, "tmpfs", MS_NOSUID | MS_NODEV | MS_NOEXEC, "size=1m,mode=0755")
        os.makedirs(work, exist_ok=True)
        os.chdir(work)
        stage = work + "/.devstage"
        os.makedirs(stage, exist_ok=True)
        mount("tmpfs", stage, "tmpfs", MS_NOSUID, "size=1m,mode=0755")
        for n in ("null", "zero", "full", "random", "urandom"):
            open(stage + "/" + n, "w").close()
            mount("/dev/" + n, stage + "/" + n, None, MS_BIND, None)
        for link, tgt in (("fd", "/proc/self/fd"), ("stdin", "/proc/self/fd/0"), ("stdout", "/proc/self/fd/1"), ("stderr", "/proc/self/fd/2")):
            os.symlink(tgt, stage + "/" + link)
        mount(stage, "/dev", None, MS_BIND | MS_REC, None)
        try:
            mount("proc", "/proc", "proc", MS_NOSUID | MS_NODEV | MS_NOEXEC, None)
        except OSError:
            mount("tmpfs", "/proc", "tmpfs", MS_NOSUID | MS_NODEV | MS_NOEXEC, "size=1m")
        exe = work + "/solver"
        with open(exe, "wb") as f:
            f.write(blob)
        os.chmod(exe, 0o500)
        for cap in range(0, 64):
            libc.prctl(24, cap, 0, 0, 0)  # PR_CAPBSET_DROP

        class Hdr(ctypes.Structure):
            _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

        class Dat(ctypes.Structure):
            _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32), ("inheritable", ctypes.c_uint32)]

        hdr, dat = Hdr(0x20080522, 0), (Dat * 2)()
        libc.capset.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        if libc.capset(ctypes.byref(hdr), dat) != 0:
            raise OSError(ctypes.get_errno(), "capset")
    else:
        exe = req["binary"]
        os.chdir("/")
    libc.prctl(38, 1, 0, 0, 0)  # PR_SET_NO_NEW_PRIVS
    libc.prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE 0
    st = open("/proc/self/status").read()
    caps = {l.split(":")[0]: l.split(":")[1].strip() for l in st.splitlines() if l.startswith(("CapEff", "CapPrm", "CapBnd", "CapInh", "NoNewPrivs"))}
    if caps.get("NoNewPrivs") != "1" or (ns and any(int(caps.get(k, "1"), 16) != 0 for k in ("CapEff", "CapPrm", "CapBnd", "CapInh"))):
        raise RuntimeError("privileges not fully dropped: %r" % caps)

    mfd = os.memfd_create("solver_stdin")
    os.write(mfd, stdin_bytes)
    os.lseek(mfd, 0, os.SEEK_SET)
    mem = int(req["mem_mb"]) * 1024 * 1024
    cpu = int(req["timeout_s"]) + 2

    def pre():
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_NPROC, (req["nproc"], req["nproc"]))
        resource.setrlimit(resource.RLIMIT_FSIZE, (req["fsize"], req["fsize"]))
        resource.setrlimit(resource.RLIMIT_NOFILE, (req["nofile"], req["nofile"]))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    t0 = time.monotonic()
    p = subprocess.Popen([exe] + list(req["args"]), cwd=work if ns else "/", env={"PATH": "/usr/bin:/bin"}, stdin=mfd, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, close_fds=True, preexec_fn=pre, start_new_session=True)
    os.close(mfd)
    lim, tail = req["out_limit"], req["stderr_tail"]
    out, err = bytearray(), bytearray()
    bufs = {p.stdout: out, p.stderr: err}
    sel = selectors.DefaultSelector()
    for s in bufs:
        os.set_blocking(s.fileno(), False)
        sel.register(s, selectors.EVENT_READ)
    limit, end_seen = None, None

    def kill():
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            pass

    while sel.get_map():
        left = req["timeout_s"] - (time.monotonic() - t0)
        if p.poll() is not None:
            if end_seen is None:
                end_seen = time.monotonic()
            if time.monotonic() - end_seen > 0.3:
                break
        if left <= 0 and p.poll() is None and limit is None:
            limit = "timeout"
            kill()
        for key, _ in sel.select(timeout=0.05):
            try:
                chunk = os.read(key.fileobj.fileno(), 65536)
            except BlockingIOError:
                continue
            if not chunk:
                sel.unregister(key.fileobj)
                continue
            b = bufs[key.fileobj]
            b += chunk
            if b is out and len(b) > lim and limit is None:
                limit = "stdout"
                kill()
            if b is err and len(b) > tail:
                del b[: len(b) - tail]
        if limit is not None and p.poll() is not None:
            break
    kill()
    p.wait()
    wall = time.monotonic() - t0
    errtxt = bytes(err).decode("utf-8", "replace")
    low = errtxt.lower()
    if limit is None and p.returncode == -int(signal.SIGXCPU):
        limit = "cpu"
    elif limit is None and p.returncode is not None and p.returncode < 0 and wall >= cpu:
        limit = "cpu"
    elif limit is None and any(m in low for m in ("memoryerror", "memory allocation of", "out of memory", "cannot allocate memory")):
        limit = "memory"
    emit({"ok": True, "rc": p.returncode, "limit": limit, "wall_s": wall, "stdout_b64": base64.b64encode(bytes(out[:lim])).decode(), "stderr": errtxt})

try:
    main()
except BaseException as e:  # noqa: BLE001  report, never crash silently
    emit({"ok": False, "error": type(e).__name__ + ": " + str(e)})
'''


def _find_exec(names: Sequence[str]) -> Optional[str]:
    for n in names:
        p = n if os.path.isabs(n) else shutil.which(n)
        if p and os.access(p, os.X_OK):
            return p
    return None


def system_python() -> Optional[str]:
    """A python3 outside /home (the launcher runs after /home is hidden). Same rule as E15."""
    for p in ("/usr/bin/python3", "/usr/local/bin/python3", "/bin/python3"):
        if os.access(p, os.X_OK):
            return p
    return None


def _hide_list() -> List[str]:
    hide = [d for d in HIDE_DIRS if os.path.isdir(d) and not os.path.islink(d)]
    for extra in (os.path.realpath(os.path.expanduser("~")), os.path.realpath(os.getcwd())):
        if os.path.isdir(extra) and extra != "/" and not any(extra == h or extra.startswith(h + os.sep) for h in hide):
            hide.append(extra)
    return hide


_NS_CACHE: Optional[bool] = None
_NS_WHY = ""
_NS_LOCK = threading.Lock()


def _unshare_cmd(unshare: str) -> List[str]:
    return [unshare, "--user", "--map-root-user", "--mount", "--net", "--pid", "--ipc", "--uts", "--fork", "--kill-child", "--"]


def namespace_available(refresh: bool = False) -> bool:
    """True if a user+mount+net+pid namespace can be created here (probed once)."""
    if not isinstance(refresh, bool):
        raise ValueError("refresh must be a boolean")
    global _NS_CACHE, _NS_WHY
    with _NS_LOCK:
        if _NS_CACHE is not None and not refresh:
            return _NS_CACHE
        unshare, py = _find_exec(("unshare",)), system_python()
        if unshare is None or py is None:
            _NS_CACHE, _NS_WHY = False, "unshare or a system python3 not found"
            return False
        try:
            r = subprocess.run(_unshare_cmd(unshare) + [py, "-I", "-S", "-B", "-c", "print(1)"], capture_output=True, timeout=15, env={"PATH": "/usr/bin:/bin"}, cwd="/")
            ok = r.returncode == 0 and r.stdout.strip() == b"1"
            _NS_CACHE, _NS_WHY = ok, "" if ok else "unshare failed: " + r.stderr.decode("utf-8", "replace")[-200:].strip()
        except (OSError, subprocess.SubprocessError) as exc:
            _NS_CACHE, _NS_WHY = False, f"unshare probe failed: {type(exc).__name__}"
        return _NS_CACHE


def availability_reason() -> str:
    return _NS_WHY


def file_sha256(path: str) -> str:
    if not isinstance(path, str) or not path.strip() or any(ord(c) < 32 or ord(c) == 127 for c in path):
        raise ValueError("path must be a non-empty string")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _refuse(msg: str) -> SandboxOutcome:
    return SandboxOutcome(ok=False, error=msg, refused=True)


def run_binary(argv_binary: Sequence[str], stdin_bytes: bytes, *, limits: Limits = Limits(), expected_sha256: Optional[str] = None) -> SandboxOutcome:
    """Run ``argv_binary[0]`` with arguments ``argv_binary[1:]`` and ``stdin_bytes`` under the limits. Never raises for runtime problems; see SandboxOutcome."""
    t0 = time.monotonic()
    if not isinstance(limits, Limits):
        raise ValueError("limits must be a Limits instance")
    if (
        isinstance(argv_binary, (str, bytes))
        or not isinstance(argv_binary, Sequence)
        or not argv_binary
        or not all(isinstance(a, str) and "\x00" not in a for a in argv_binary)
        or not argv_binary[0].strip()
    ):
        return _refuse("empty command")
    if not isinstance(stdin_bytes, (bytes, bytearray)):
        return _refuse("stdin_bytes must be bytes")
    if len(stdin_bytes) > limits.stdin_limit:
        return _refuse(f"stdin of {len(stdin_bytes)} bytes exceeds the {limits.stdin_limit}-byte limit")
    if expected_sha256 is not None:
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256.strip()) != 64
            or not all(c in "0123456789abcdefABCDEF" for c in expected_sha256.strip())
        ):
            return _refuse("expected_sha256 must be a 64-character hex string")
    binary = os.path.realpath(argv_binary[0])
    try:
        sha = file_sha256(binary)
    except OSError as exc:
        return _refuse(f"cannot read binary: {exc.strerror}")
    if expected_sha256 is not None and sha != expected_sha256.strip().lower():
        return _refuse("binary sha256 does not match the expected value")
    if not os.access(binary, os.X_OK):
        return _refuse("binary is not executable")
    py = system_python()
    if py is None:
        return _refuse("no system python3 for the launcher")
    ns = namespace_available() and not limits.force_none
    if not ns and not limits.allow_unsandboxed:
        return _refuse("user namespaces unavailable and allow_unsandboxed is off (" + (_NS_WHY or "probe failed") + ")")
    mode = SANDBOX_NAMESPACE if ns else SANDBOX_NONE
    req = {"ns": ns, "binary": binary, "binary_sha256": sha, "args": list(argv_binary[1:]), "stdin_b64": base64.b64encode(stdin_bytes).decode(),
           "timeout_s": float(limits.timeout_s), "mem_mb": int(limits.mem_mb), "nproc": int(limits.nproc), "fsize": int(limits.fsize_bytes),
           "nofile": int(limits.nofile), "out_limit": int(limits.stdout_limit), "stderr_tail": int(limits.stderr_tail), "hide": _hide_list() if ns else [],
           "workdir": "/tmp/w", "tmpfs_mb": int(limits.tmpfs_mb) + os.path.getsize(binary) // (1024 * 1024) + 1}
    launcher = ["-I", "-S", "-B", "-c", LAUNCHER_SRC.replace("@MARK@", repr(SANDBOX_MARK))]
    cmd = (_unshare_cmd(_find_exec(("unshare",)) or "unshare") if ns else []) + [py] + launcher
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={"PATH": "/usr/bin:/bin"}, cwd="/", start_new_session=True)
    except OSError as exc:
        return SandboxOutcome(ok=False, error=f"cannot start launcher: {exc.strerror}", sandbox=mode, wall_s=time.monotonic() - t0)
    try:
        out, err = p.communicate(json.dumps(req).encode("utf-8"), timeout=float(limits.timeout_s) + HOST_GRACE_S)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            pass
        p.communicate()
        return SandboxOutcome(ok=False, error="host-side deadline exceeded (launcher hung)", sandbox=mode, limit="timeout", wall_s=time.monotonic() - t0)
    line = next((l for l in reversed(out.decode("utf-8", "replace").splitlines()) if l.startswith(SANDBOX_MARK)), None)
    if line is None:
        return SandboxOutcome(ok=False, error="no sandbox report; launcher rc=%s stderr=%s" % (p.returncode, err.decode("utf-8", "replace")[-300:]), sandbox=mode,
                              wall_s=time.monotonic() - t0)
    try:
        d = json.loads(line[len(SANDBOX_MARK):])
        if not isinstance(d, dict):
            raise ValueError("not an object")
        if not d.get("ok"):
            return SandboxOutcome(ok=False, error=str(d.get("error", "sandbox error"))[:300], sandbox=mode, wall_s=time.monotonic() - t0)
        return SandboxOutcome(ok=True, rc=d["rc"], stdout=base64.b64decode(d["stdout_b64"]), stderr_tail=str(d["stderr"])[-limits.stderr_tail:], wall_s=float(d["wall_s"]),
                              limit=d.get("limit"), sandbox=mode)
    except (KeyError, TypeError, ValueError):
        return SandboxOutcome(ok=False, error="garbled sandbox report", sandbox=mode, wall_s=time.monotonic() - t0)
