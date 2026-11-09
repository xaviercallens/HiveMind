# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Optional resident validator: keep the validator's base model loaded across results (OFF by default).

``hivemind_validate`` is a one-shot process started by BOINC's ``script_validator`` for every result, so the base model is
loaded again each time. This module adds (1) :class:`ModelCache`, a thread-safe LRU cache of loaded models, (2) a small
local server/client over a UNIX domain socket (no TCP listener, nothing on any network interface) with a length-prefixed
JSON protocol, and (3) :func:`forward_validate`, which the one-shot script uses when ``HIVEMIND_RESIDENT_VALIDATOR_SOCKET``
is set. If the server is not reachable the one-shot path runs as before (one warning line); a validation never fails
because the resident server is down.

The server runs the SAME function as the one-shot path (``boinc_server_scripts.run_validation``); only the source of the
base model differs. What this does NOT solve: GPU memory shared between the resident model and a volunteer's task.

No torch import at module import time. Only JSON crosses the socket.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import signal
import socket
import socketserver
import stat
import struct
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Hashable, Iterator

ENV_SOCKET = "HIVEMIND_RESIDENT_VALIDATOR_SOCKET"
ENV_WORK_ROOT = "HIVEMIND_RESIDENT_WORK_ROOT"
MAX_MESSAGE_BYTES = 1 << 20
DEFAULT_REQUEST_TIMEOUT_S = 3600.0


# ----------------------------------------------------------------------------------------------------------- cache
class ModelCache:
    """LRU cache of loaded models keyed by e.g. ``(model_id, quantisation signature, device)``.

    ``loader(key)`` returns the object, ``unloader(obj)`` frees it, ``sizer(obj)`` its bytes (default 0). Budgets are
    ``max_entries`` and (optionally) ``max_bytes``; least-recently-used entries are evicted first. A single entry larger than
    ``max_bytes`` is kept alone after everything else is evicted (the budget is soft for that one case; ``stats()`` counts it as
    ``over_budget``). Concurrent requests for one key load once (per-key lock); a loader that raises caches nothing and does
    not affect other keys. An entry that is evicted while checked out is unloaded when its last user releases it.
    """

    def __init__(
        self,
        loader: Callable[[Hashable], Any],
        unloader: Callable[[Any], None] = lambda obj: None,
        *,
        max_entries: int = 1,
        max_bytes: int | None = None,
        sizer: Callable[[Any], int] = lambda obj: 0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if max_bytes is not None and (isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1):
            raise ValueError("max_bytes must be a positive integer when provided")
        if not callable(loader) or not callable(unloader) or not callable(sizer) or not callable(clock):
            raise ValueError("loader, unloader, sizer, and clock must be callable")
        self._loader, self._unloader, self._sizer, self._clock = loader, unloader, sizer, clock
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self._lock = threading.RLock()
        self._entries: OrderedDict[Hashable, dict[str, Any]] = OrderedDict()
        self._key_locks: dict[Hashable, threading.Lock] = {}
        self._zombies: list[dict[str, Any]] = []
        self._m = {"hits": 0, "misses": 0, "loads": 0, "load_failures": 0, "evictions": 0, "over_budget": 0, "seconds_saved_estimate": 0.0}

    # -- internals
    def _drop(self, key: Hashable, count_eviction: bool) -> None:
        """Remove ``key`` (caller holds ``self._lock``); unload now, or later if still checked out."""
        e = self._entries.pop(key)
        if count_eviction:
            self._m["evictions"] += 1
        if e["refs"] > 0:
            e["zombie"] = True
            self._zombies.append(e)
        else:
            self._safe_unload(e["obj"])

    def _safe_unload(self, obj: Any) -> None:
        try:
            self._unloader(obj)
        except Exception as exc:  # an unloader failure must not break the cache
            print(f"resident_validator: unloader failed: {exc}", file=sys.stderr)

    def _enforce_budgets(self, keep: Hashable) -> None:
        def used() -> int:
            return sum(e["bytes"] for e in self._entries.values())

        while len(self._entries) > self.max_entries or (self.max_bytes is not None and used() > self.max_bytes and len(self._entries) > 1):
            victim = next((k for k in self._entries if k != keep), None)
            if victim is None:
                break
            self._drop(victim, True)
        if self.max_bytes is not None and used() > self.max_bytes:
            self._m["over_budget"] += 1

    # -- API
    def checkout(self, key: Hashable) -> Any:
        """Return the model for ``key`` (loading it on a miss) and pin it until :meth:`release`."""
        with self._lock:
            e = self._entries.get(key)
            if e is not None:
                self._entries.move_to_end(key)
                e["refs"] += 1
                self._m["hits"] += 1
                self._m["seconds_saved_estimate"] += e["load_s"]
                return e["obj"]
            klock = self._key_locks.setdefault(key, threading.Lock())
        with klock:
            with self._lock:  # another thread may have loaded it while we waited
                e = self._entries.get(key)
                if e is not None:
                    self._entries.move_to_end(key)
                    e["refs"] += 1
                    self._m["hits"] += 1
                    self._m["seconds_saved_estimate"] += e["load_s"]
                    return e["obj"]
                self._m["misses"] += 1
            t0 = self._clock()
            try:
                obj = self._loader(key)
            except BaseException:
                with self._lock:
                    self._m["load_failures"] += 1
                raise
            load_s = self._clock() - t0
            try:
                nbytes = int(self._sizer(obj))
            except Exception:
                nbytes = 0
            with self._lock:
                self._entries[key] = {"obj": obj, "bytes": nbytes, "load_s": load_s, "refs": 1, "zombie": False}
                self._m["loads"] += 1
                self._enforce_budgets(key)
            return obj

    def release(self, key: Hashable, obj: Any = None) -> None:
        with self._lock:
            e = self._entries.get(key)
            if e is not None and (obj is None or e["obj"] is obj):
                e["refs"] = max(0, e["refs"] - 1)
                return
            for z in list(self._zombies):
                if obj is None or z["obj"] is obj:
                    z["refs"] -= 1
                    if z["refs"] <= 0:
                        self._zombies.remove(z)
                        self._safe_unload(z["obj"])
                    return

    @contextlib.contextmanager
    def acquire(self, key: Hashable) -> Iterator[Any]:
        obj = self.checkout(key)
        try:
            yield obj
        finally:
            self.release(key, obj)

    def evict(self, key: Hashable) -> bool:
        with self._lock:
            if key not in self._entries:
                return False
            self._drop(key, True)
            return True

    def clear(self) -> int:
        """Unload everything (checked-out entries are unloaded on release). Returns the number of entries dropped."""
        with self._lock:
            keys = list(self._entries)
            for k in keys:
                self._drop(k, False)
            return len(keys)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, key: Hashable) -> bool:
        with self._lock:
            return key in self._entries

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {**self._m, "entries": len(self._entries), "bytes": sum(e["bytes"] for e in self._entries.values()),
                    "keys": [repr(k) for k in self._entries], "max_entries": self.max_entries, "max_bytes": self.max_bytes}


# ----------------------------------------------------------------------------------------------------- real model hooks
def model_key(spec: dict[str, Any], device: str = "cuda:0") -> tuple[str, str, str]:
    """Cache key of the validator's base model: model id, quantisation + LoRA-init signature, device."""
    if not isinstance(spec, dict) or not all(k in spec for k in ("model_id", "rank", "alpha", "init_seed")):
        raise ValueError("spec must be a dict containing model_id, rank, alpha, and init_seed")
    if not isinstance(spec["model_id"], str) or not spec["model_id"].strip():
        raise ValueError("spec['model_id'] must be a non-empty string")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty string")
    sig = f"nf4-dq-fp16|r{spec['rank']}|a{spec['alpha']}|s{spec['init_seed']}"
    return (str(spec["model_id"]), sig, device)


def _real_unloader(bundle: Any) -> None:  # pragma: no cover - needs a GPU stack
    del bundle
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _real_sizer(bundle: Any) -> int:  # pragma: no cover - needs a GPU stack
    try:
        return int(sum(p.numel() * p.element_size() for p in bundle[0].parameters()))
    except Exception:
        return 0


def _make_validator(cache: ModelCache, on_spec: Callable[[Hashable, dict[str, Any]], None] | None = None) -> Any:
    """A ``MoEValidator`` whose model is taken from ``cache`` (and pinned while in use) instead of being loaded per process."""
    from hivemind.boinc.real_client.moe_bridge import MoEValidator

    class CachedMoEValidator(MoEValidator):
        _held: tuple[Hashable, Any] | None = None

        def _ensure(self, spec: dict[str, Any]) -> None:  # called under MoEValidator's lock
            key = model_key(spec)
            if on_spec is not None:
                on_spec(key, spec)
            held, self._held = self._held, None
            bundle = cache.checkout(key)
            if held is not None:
                cache.release(*held)
            self._held = (key, bundle)
            self._model, self._init_state = bundle
            self._model_key = key

    return CachedMoEValidator()


def default_cache(max_entries: int = 1, max_bytes: int | None = None) -> ModelCache:  # pragma: no cover - needs a GPU stack
    """Cache backed by the real loader; the spec is recovered from a side table filled by the validator."""
    holder: dict[Hashable, dict[str, Any]] = {}

    def loader(key: Hashable) -> Any:
        from hivemind.boinc.real_client.moe_app import load_peft_model
        from hivemind.core.quantization import extract_lora_state_dict

        spec = holder[key]
        model = load_peft_model(spec)
        return (model, extract_lora_state_dict(model))

    cache = ModelCache(loader, _real_unloader, max_entries=max_entries, max_bytes=max_bytes, sizer=_real_sizer)
    cache.spec_table = holder  # type: ignore[attr-defined]
    return cache


# ------------------------------------------------------------------------------------------------------------ protocol
class ProtocolError(Exception):
    pass


def send_msg(sock: socket.socket, obj: dict[str, Any], limit: int = MAX_MESSAGE_BYTES) -> None:
    if not isinstance(obj, dict):
        raise ProtocolError("message must be a dict")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    data = json.dumps(obj, sort_keys=True).encode()
    if len(data) > limit:
        raise ProtocolError(f"message of {len(data)} bytes exceeds the limit of {limit}")
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ProtocolError("connection closed mid-message")
        buf += chunk
    return bytes(buf)


def recv_msg(sock: socket.socket, limit: int = MAX_MESSAGE_BYTES) -> dict[str, Any]:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    (n,) = struct.unpack(">I", _recv_exact(sock, 4))
    if n > limit:
        raise ProtocolError(f"message of {n} bytes exceeds the limit of {limit}")  # body is never read
    try:
        obj = json.loads(_recv_exact(sock, n).decode())
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProtocolError(f"invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ProtocolError("message must be a JSON object")
    return obj


# -------------------------------------------------------------------------------------------------------------- server
class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        srv: ResidentValidatorServer = self.server.owner  # type: ignore[attr-defined]
        self.request.settimeout(srv.io_timeout_s)
        try:
            req = recv_msg(self.request, srv.max_message_bytes)
        except (ProtocolError, OSError) as exc:
            self._reply(srv, {"ok": False, "error": f"bad request: {exc}"})
            return
        try:
            resp = srv.dispatch(req)
        except Exception as exc:  # never let a handler bug kill the server
            resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        self._reply(srv, resp)

    def _reply(self, srv: "ResidentValidatorServer", resp: dict[str, Any]) -> None:
        try:
            send_msg(self.request, resp, srv.max_message_bytes)
        except (ProtocolError, OSError):
            pass


class _UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = False


class ResidentValidatorServer:
    """UNIX-socket validator service. ``validate_fn(files, log) -> rc`` defaults to ``boinc_server_scripts.run_validation`` with a
    validator whose base model comes from ``cache``. Validations are serialised (one GPU consumer at a time)."""

    def __init__(
        self,
        socket_path: str | os.PathLike[str],
        work_root: str | os.PathLike[str],
        cache: ModelCache | None = None,
        *,
        validate_fn: Callable[[list[str], Callable[[str], None]], int] | None = None,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        io_timeout_s: float = 30.0,
        max_message_bytes: int = MAX_MESSAGE_BYTES,
    ) -> None:
        if isinstance(request_timeout_s, bool) or not isinstance(request_timeout_s, (int, float)) or not math.isfinite(float(request_timeout_s)) or float(request_timeout_s) <= 0.0:
            raise ValueError("request_timeout_s must be a finite positive number")
        if isinstance(io_timeout_s, bool) or not isinstance(io_timeout_s, (int, float)) or not math.isfinite(float(io_timeout_s)) or float(io_timeout_s) <= 0.0:
            raise ValueError("io_timeout_s must be a finite positive number")
        if isinstance(max_message_bytes, bool) or not isinstance(max_message_bytes, int) or max_message_bytes < 1:
            raise ValueError("max_message_bytes must be a positive integer")
        if not isinstance(socket_path, (str, os.PathLike)) or not str(socket_path).strip():
            raise ValueError("socket_path must be a non-empty path")
        if not isinstance(work_root, (str, os.PathLike)) or not str(work_root).strip():
            raise ValueError("work_root must be a non-empty path")
        if cache is not None and not isinstance(cache, ModelCache):
            raise ValueError("cache must be a ModelCache or None")
        if validate_fn is not None and not callable(validate_fn):
            raise ValueError("validate_fn must be callable or None")
        self.socket_path = Path(socket_path)
        self.work_root = Path(work_root).resolve()
        self.cache = cache
        self._validate_fn = validate_fn
        self.request_timeout_s, self.io_timeout_s, self.max_message_bytes = request_timeout_s, io_timeout_s, max_message_bytes
        self._validate_lock = threading.Lock()
        self._server: _UnixServer | None = None
        self._started = time.time()
        self._mv: Any = None
        self._stop_event = threading.Event()
        self._counts = {"validate": 0, "refused": 0, "timeouts": 0}

    # -- lifecycle
    def _prepare_socket_dir(self) -> None:
        d = self.socket_path.parent
        if not d.exists():
            d.mkdir(mode=0o700, parents=True)
            os.chmod(d, 0o700)
        else:
            st = d.stat()
            if st.st_uid != os.getuid() or st.st_mode & 0o077:
                raise PermissionError(f"socket directory {d} must be owned by this user with mode 0700")
        if self.socket_path.exists() or self.socket_path.is_symlink():
            if not stat.S_ISSOCK(self.socket_path.lstat().st_mode):
                raise FileExistsError(f"{self.socket_path} exists and is not a socket")
            probe = socket.socket(socket.AF_UNIX)
            try:
                probe.settimeout(1.0)
                probe.connect(str(self.socket_path))
            except OSError:
                self.socket_path.unlink()  # stale socket of a dead server
            else:
                raise OSError(f"another resident validator is already listening on {self.socket_path}")
            finally:
                probe.close()

    def start(self) -> None:
        """Bind and start serving in a background thread."""
        self._prepare_socket_dir()
        old = os.umask(0o177)
        try:
            self._server = _UnixServer(str(self.socket_path), _Handler)
        finally:
            os.umask(old)
        os.chmod(self.socket_path, 0o600)
        self._server.owner = self  # type: ignore[attr-defined]
        threading.Thread(target=self._server.serve_forever, name="resident-validator", daemon=True).start()

    def serve_forever(self) -> None:
        """Run until :meth:`shutdown` (SIGTERM/SIGINT call it when the CLI is used)."""
        self.start()
        self._stop_event.wait()
        self.shutdown()

    def request_stop(self) -> None:
        self._stop_event.set()  # safe inside a signal handler

    def shutdown(self) -> None:
        srv, self._server = self._server, None
        if srv is not None:
            srv.shutdown()
            srv.server_close()
        with contextlib.suppress(OSError):
            self.socket_path.unlink()
        if self.cache is not None:
            self.cache.clear()  # unloads every model

    # -- requests
    def _check_paths(self, files: Any) -> list[str]:
        if not isinstance(files, list) or not files or not all(isinstance(f, str) for f in files):
            raise ValueError("files must be a non-empty list of strings")
        for f in files:
            if f.isdigit():
                continue  # the result id (a bare integer), not a path
            p = Path(os.path.realpath(f))
            if not (p == self.work_root or self.work_root in p.parents):
                raise PermissionError(f"path outside the work root: {f}")
        return files

    def dispatch(self, req: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(req, dict):
            raise ValueError("req must be a dict")
        op = req.get("op")
        if op == "ping":
            return {"ok": True, "pong": True, "pid": os.getpid()}
        if op == "stats":
            return {"ok": True, "uptime_s": round(time.time() - self._started, 3), "requests": dict(self._counts),
                    "cache": self.cache.stats() if self.cache is not None else None}
        if op == "validate":
            try:
                files = self._check_paths(req.get("files"))
            except (ValueError, PermissionError) as exc:
                self._counts["refused"] += 1
                return {"ok": False, "error": str(exc)}
            return self._run_validate(files)
        return {"ok": False, "error": f"unknown op {op!r}"}

    def _default_validate(self, files: list[str], log: Callable[[str], None]) -> int:
        from hivemind.boinc.real_client import boinc_server_scripts as bss

        assert self.cache is not None
        if self._mv is None:
            table = getattr(self.cache, "spec_table", None)  # lets the real loader see the spec of the key it is asked for
            self._mv = _make_validator(self.cache, (lambda k, sp: table.__setitem__(k, sp)) if table is not None else None)
        return bss.run_validation(files, self._mv, log)

    def _run_validate(self, files: list[str]) -> dict[str, Any]:
        fn = self._validate_fn or self._default_validate
        lines: list[str] = []
        result: dict[str, Any] = {}

        def job() -> None:
            try:
                with self._validate_lock:
                    result["rc"] = int(fn(files, lines.append))
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"

        t = threading.Thread(target=job, daemon=True)
        t.start()
        t.join(self.request_timeout_s)
        if t.is_alive():
            self._counts["timeouts"] += 1
            return {"ok": False, "error": f"timeout after {self.request_timeout_s}s"}
        self._counts["validate"] += 1
        if "error" in result:
            return {"ok": False, "error": result["error"]}
        return {"ok": True, "rc": result["rc"], "log": lines}


# -------------------------------------------------------------------------------------------------------------- client
class ResidentValidatorClient:
    def __init__(self, socket_path: str | os.PathLike[str], timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S + 60.0, max_message_bytes: int = MAX_MESSAGE_BYTES) -> None:
        if not str(socket_path).strip():
            raise ValueError("socket_path must be non-empty")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(float(timeout_s)) or float(timeout_s) <= 0.0:
            raise ValueError("timeout_s must be a finite positive number")
        if isinstance(max_message_bytes, bool) or not isinstance(max_message_bytes, int) or max_message_bytes < 1:
            raise ValueError("max_message_bytes must be a positive integer")
        self.socket_path, self.timeout_s, self.max_message_bytes = str(socket_path), timeout_s, max_message_bytes

    def _call(self, req: dict[str, Any]) -> dict[str, Any]:
        s = socket.socket(socket.AF_UNIX)
        try:
            s.settimeout(self.timeout_s)
            s.connect(self.socket_path)
            send_msg(s, req, self.max_message_bytes)
            return recv_msg(s, self.max_message_bytes)
        finally:
            s.close()

    def ping(self) -> dict[str, Any]:
        return self._call({"op": "ping"})

    def stats(self) -> dict[str, Any]:
        return self._call({"op": "stats"})

    def validate(self, files: list[str]) -> dict[str, Any]:
        return self._call({"op": "validate", "files": list(files)})


def forward_validate(socket_path: str, args: list[str]) -> int | None:
    """Ask a resident server to validate ``args`` (the ``hivemind_validate`` arguments). Returns its exit status, or ``None`` after
    a single warning line when the server is unreachable or fails, so the caller falls back to the one-shot path."""
    try:
        resp = ResidentValidatorClient(socket_path, float(os.environ.get("HIVEMIND_RESIDENT_TIMEOUT_S", DEFAULT_REQUEST_TIMEOUT_S + 60.0))).validate(args)
    except (OSError, ProtocolError) as exc:
        print(f"hivemind_validate: resident validator unavailable ({type(exc).__name__}: {exc}); running one-shot", file=sys.stderr)
        return None
    if not resp.get("ok") or "rc" not in resp:
        print(f"hivemind_validate: resident validator refused or failed ({resp.get('error')}); running one-shot", file=sys.stderr)
        return None
    for line in resp.get("log", []):
        print(line, file=sys.stderr)
    return int(resp["rc"])


# ----------------------------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m hivemind.boinc.real_client.resident_validator", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["serve", "ping", "stats"])
    ap.add_argument("--socket", default=os.environ.get(ENV_SOCKET), help=f"UNIX socket path (env {ENV_SOCKET})")
    ap.add_argument("--work-root", default=os.environ.get(ENV_WORK_ROOT), help=f"serve: only files below this directory are validated (env {ENV_WORK_ROOT})")
    ap.add_argument("--max-entries", type=int, default=1)
    ap.add_argument("--max-bytes", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT_S, help="serve: per-request timeout, seconds")
    ap.add_argument("--preload", metavar="MODEL_ID", help="serve: load this model at start (default: load nothing until the first request)")
    ap.add_argument("--rank", type=int, default=8, help="--preload: LoRA rank (must match the workunits' spec)")
    ap.add_argument("--alpha", type=int, default=16, help="--preload: LoRA alpha")
    ap.add_argument("--init-seed", type=int, default=42, help="--preload: LoRA init seed")
    a = ap.parse_args(argv)
    if not a.socket:
        ap.error(f"--socket (or {ENV_SOCKET}) is required")
    if a.cmd in ("ping", "stats"):
        try:
            c = ResidentValidatorClient(a.socket, 10.0)
            print(json.dumps(c.ping() if a.cmd == "ping" else c.stats(), indent=2, sort_keys=True))
            return 0
        except (OSError, ProtocolError) as exc:
            print(f"unreachable: {exc}", file=sys.stderr)
            return 1
    if not a.work_root:
        ap.error(f"serve needs --work-root (or {ENV_WORK_ROOT})")
    cache = default_cache(a.max_entries, a.max_bytes)
    srv = ResidentValidatorServer(a.socket, a.work_root, cache, request_timeout_s=a.timeout)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: srv.request_stop())
    print(f"resident validator listening on {a.socket} (work root {srv.work_root}); loads nothing until the first request", file=sys.stderr)
    if a.preload:
        spec = {"model_id": a.preload, "rank": a.rank, "alpha": a.alpha, "init_seed": a.init_seed}
        key = model_key(spec)
        cache.spec_table[key] = spec  # type: ignore[attr-defined]
        cache.release(key, cache.checkout(key))
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
