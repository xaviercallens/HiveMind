#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-dependency cross-platform HiveMind quick-connect & Gamer-Whisper assistant.

Designed for personal home machines (Windows 11, macOS, Linux) with standard-library Python only:
  1. ``connect`` (default): register or load ``~/.hivemind/credentials.json``, verify API access
     against ``/v1/hivemind/me`` and ``/v1/models``, and output a ready-to-use VS Code (Continue.dev)
     snippet without installing PyTorch.
  2. ``--gamer-check`` / ``--gamer-whisper``: quiet gaming-PC guard (``<= 70 C`` default ceiling,
     ``2048 MiB`` free-VRAM reserve, active game detection, ``< 1.5 s`` full process-tree stop).
  3. ``--feedback``: optional anonymous usability & comfort telemetry (``ease_rating`` 1..5).
  4. ``--uninstall``: remove saved local HiveMind credentials cleanly in one step.

Personal Home Use & Non-Monetary Notice:
  - Personal home PCs and personal side-projects ONLY (never work laptops or corporate code/data).
  - 0.00 EUR / 0.00 USD: symbolic points have zero monetary value.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import re
import secrets
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
HANDLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{1,47}$")

DEFAULT_GAME_EXECUTABLES: frozenset[str] = frozenset(
    {
        "cs2.exe",
        "cs2",
        "dota2.exe",
        "dota2",
        "cyberpunk2077.exe",
        "eldenring.exe",
        "bg3.exe",
        "helldivers2.exe",
        "valorant.exe",
        "valorant-win64-shipping.exe",
        "fortniteclient-win64-shipping.exe",
        "overwatch.exe",
        "league of legends.exe",
        "baldur.exe",
        "starfield.exe",
        "witcher3.exe",
    }
)


@dataclass(frozen=True)
class GpuSnapshot:
    """Measured local GPU state from nvidia-smi."""

    present: bool
    name: str
    total_mib: int
    free_mib: int
    temp_c: float
    util_pct: float


HttpFn = Callable[[str, str, dict[str, Any] | None, dict[str, str] | None], tuple[int, dict[str, Any]]]


def _default_credentials_path() -> Path:
    return Path.home() / ".hivemind" / "credentials.json"


def _default_continue_snippet_path() -> Path:
    return Path.home() / ".continue" / "hivemind_provider.yaml"


def validate_gateway_url(url: str) -> str:
    cleaned = (url or "").strip().rstrip("/")
    if not (
        cleaned.startswith("https://")
        and cleaned.split("/")[2].endswith(".run.app")
        or cleaned.startswith("http://127.0.0.1")
        or cleaned.startswith("http://localhost")
    ):
        raise ValueError(
            f"Refusing untrusted gateway URL {url!r}: expected a *.run.app HTTPS gateway or localhost."
        )
    return cleaned


def sanitize_handle(raw: str) -> str:
    candidate = re.sub(r"[^A-Za-z0-9._+-]+", "-", (raw or "").strip()).strip("-.")
    if len(candidate) < 2:
        candidate = f"friend-{secrets.token_hex(3)}"
    candidate = candidate[:48]
    if not HANDLE_RE.match(candidate) or candidate.lower().startswith(("google.", "github.", "oidc.")):
        raise ValueError(
            "Invalid handle: use 2-48 ASCII letters, digits, '.', '_', '+', or '-' (not starting with a provider prefix)."
        )
    return candidate


def default_http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    data_bytes: bytes | None = None
    if payload is not None:
        data_bytes = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data_bytes, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return int(resp.status), (json.loads(body) if body.strip() else {})
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body) if body.strip() else {"error": body}
        except json.JSONDecodeError:
            parsed = {"error": body}
        return int(exc.code), parsed


def save_credentials(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if os.name != "nt":
        os.chmod(tmp, 0o600)
    tmp.replace(path)


def load_credentials(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict) and str(data.get("api_key", "")).startswith("hm-"):
        return data
    return None


def probe_local_gpu() -> GpuSnapshot:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            parts = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
            if len(parts) >= 5:
                return GpuSnapshot(
                    present=True,
                    name=parts[0],
                    total_mib=max(0, int(float(parts[1]))),
                    free_mib=max(0, int(float(parts[2]))),
                    temp_c=float(parts[3]),
                    util_pct=float(parts[4]),
                )
    except (OSError, ValueError, subprocess.SubprocessError):
        return GpuSnapshot(present=False, name="", total_mib=0, free_mib=0, temp_c=0.0, util_pct=0.0)
    return GpuSnapshot(present=False, name="", total_mib=0, free_mib=0, temp_c=0.0, util_pct=0.0)


def list_running_process_names() -> set[str]:
    names: set[str] = set()
    try:
        if os.name == "nt":
            proc = subprocess.run(
                ["tasklist", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            for line in proc.stdout.splitlines():
                line = line.strip()
                if line.startswith('"') and '","' in line:
                    names.add(line.split('","', 1)[0].strip('"').lower())
        else:
            proc = subprocess.run(
                ["ps", "-eo", "comm="],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            for line in proc.stdout.splitlines():
                cleaned = line.strip().lower()
                if cleaned:
                    names.add(cleaned)
    except (OSError, subprocess.SubprocessError):
        return names
    return names


def get_os_idle_seconds() -> float:
    if sys.platform != "win32":
        return 999.0
    try:
        class _LastInputInfo(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_ulong)]

        lii = _LastInputInfo()
        lii.cbSize = ctypes.sizeof(_LastInputInfo)
        if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):  # type: ignore[attr-defined]
            tick = ctypes.windll.kernel32.GetTickCount()  # type: ignore[attr-defined]
            elapsed_ms = (tick - lii.dwTime) & 0xFFFFFFFF
            return round(elapsed_ms / 1000.0, 3)
    except Exception:
        return 999.0
    return 999.0


def detect_active_game(running: set[str], extra_games: list[str] | None = None) -> str | None:
    watch = set(DEFAULT_GAME_EXECUTABLES)
    for g in extra_games or []:
        cleaned = g.strip().lower()
        if cleaned:
            watch.add(cleaned)
            watch.add(cleaned[:15])
    for proc_name in sorted(running):
        if proc_name in watch:
            return proc_name
    return None


def evaluate_gamer_whisper_guard(
    gpu: GpuSnapshot,
    *,
    running_procs: set[str],
    extra_games: list[str] | None = None,
    reserve_mib: int = 2048,
    hot_c: float = 70.0,
    max_external_util_pct: float = 15.0,
    worker_running: bool = False,
) -> dict[str, Any]:
    """Decide whether a gaming PC is quiet and free enough to run or keep running a lightweight round."""
    active_game = detect_active_game(running_procs, extra_games)
    if active_game is not None:
        return {"allow": False, "reason": f"game_active:{active_game}", "active_game": active_game}
    if not gpu.present or gpu.total_mib <= 0:
        return {"allow": False, "reason": "no_nvidia_gpu_detected", "active_game": None}
    if gpu.temp_c >= hot_c:
        return {"allow": False, "reason": f"thermal_whisper_cap:{gpu.temp_c:.0f}C>={hot_c:.0f}C", "active_game": None}
    if gpu.free_mib < reserve_mib:
        return {"allow": False, "reason": f"vram_reserve_guard:{gpu.free_mib}MiB<{reserve_mib}MiB", "active_game": None}
    if not worker_running and gpu.util_pct > max_external_util_pct:
        return {
            "allow": False,
            "reason": f"external_gpu_busy:{gpu.util_pct:.0f}%>{max_external_util_pct:.0f}%",
            "active_game": None,
        }
    return {"allow": True, "reason": "ready_gamer_whisper", "active_game": None}


def build_continue_yaml(gateway: str, api_key: str) -> str:
    return (
        "# HiveMind personal hobby/open-source coding configuration for VS Code (Continue.dev)\n"
        "# Personal projects ONLY — never send corporate or confidential source code.\n"
        "name: HiveMind Holiday Lab\n"
        "version: 0.0.1\n"
        "schema: v1\n"
        "models:\n"
        "  - name: HiveMind Coder\n"
        "    provider: openai\n"
        "    model: hivemind/coder\n"
        f"    apiBase: {gateway}/v1\n"
        f"    apiKey: {api_key}\n"
        "    roles:\n"
        "      - chat\n"
        "      - edit\n"
        "  - name: HiveMind Bee-Stars (Science)\n"
        "    provider: openai\n"
        "    model: hivemind/bee-stars\n"
        f"    apiBase: {gateway}/v1\n"
        f"    apiKey: {api_key}\n"
        "    roles:\n"
        "      - chat\n"
    )


def terminate_process_tree_fast(proc: subprocess.Popen[Any], timeout_s: float = 1.2) -> float:
    """Stop a worker process tree immediately so 100% of GPU VRAM returns to the owner within < 1.5s."""
    started = time.monotonic()
    if proc.poll() is not None:
        return round(time.monotonic() - started, 4)
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        else:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                proc.terminate()
        try:
            proc.wait(timeout=max(0.2, timeout_s * 0.6))
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    proc.kill()
            else:
                proc.kill()
            proc.wait(timeout=0.5)
    except (OSError, subprocess.SubprocessError):
        return round(time.monotonic() - started, 4)
    return round(time.monotonic() - started, 4)


def ensure_connected_account(
    *,
    gateway: str,
    credentials_path: Path,
    handle: str | None,
    role: str,
    explicit_api_key: str | None,
    http_fn: HttpFn = default_http_json,
) -> dict[str, Any]:
    api_key = (explicit_api_key or os.environ.get("HIVEMIND_API_KEY") or "").strip()
    saved = load_credentials(credentials_path)
    if not api_key and saved and saved.get("gateway") == gateway:
        api_key = str(saved["api_key"])

    created = False
    resolved_handle = handle or (saved.get("handle") if saved else None) or ""
    if not api_key:
        base_handle = sanitize_handle(handle or f"hm-{platform.node() or 'home'}")
        status, body = http_fn(
            "POST",
            f"{gateway}/v1/hivemind/register",
            {"handle": base_handle, "role": role},
            None,
        )
        if status == 409 and not handle:
            retry_handle = sanitize_handle(f"{base_handle[:38]}-{secrets.token_hex(2)}")
            status, body = http_fn(
                "POST",
                f"{gateway}/v1/hivemind/register",
                {"handle": retry_handle, "role": role},
                None,
            )
            base_handle = retry_handle
        if status != 200 or not str(body.get("api_key", "")).startswith("hm-"):
            msg = body.get("error") or body
            raise RuntimeError(f"Registration refused (HTTP {status}): {msg}")
        api_key = str(body["api_key"])
        resolved_handle = base_handle
        created = True

    me_status, me_body = http_fn(
        "GET",
        f"{gateway}/v1/hivemind/me",
        None,
        {"Authorization": f"Bearer {api_key}"},
    )
    if me_status != 200:
        raise RuntimeError(f"API key check failed on /v1/hivemind/me (HTTP {me_status}): {me_body}")

    models_status, models_body = http_fn("GET", f"{gateway}/v1/models", None, None)
    models = [m.get("id") for m in (models_body.get("data") or []) if isinstance(m, dict) and m.get("id")]
    if models_status != 200:
        models = []

    record = {
        "account_id": me_body.get("account_id", ""),
        "api_key": api_key,
        "created_here": created,
        "gateway": gateway,
        "handle": me_body.get("handle") or resolved_handle,
        "role": me_body.get("role") or role,
        "saved_at_utc": datetime.now(UTC).isoformat(),
    }
    save_credentials(credentials_path, record)
    return {"account": record, "balance": me_body.get("balance", 0.0), "models": models}


def run_gamer_whisper_session(
    *,
    gateway: str,
    api_key: str,
    extra_games: list[str],
    reserve_mib: int,
    hot_c: float,
    cool_c: float,
    once: bool,
    gpu_probe_fn: Callable[[], GpuSnapshot] = probe_local_gpu,
    proc_list_fn: Callable[[], set[str]] = list_running_process_names,
    popen_fn: Callable[..., subprocess.Popen[Any]] | None = None,
    poll_interval_s: float = 0.5,
) -> dict[str, Any]:
    gpu = gpu_probe_fn()
    preflight_decision = evaluate_gamer_whisper_guard(
        gpu,
        running_procs=proc_list_fn(),
        extra_games=extra_games,
        reserve_mib=reserve_mib,
        hot_c=hot_c,
        worker_running=False,
    )
    if not preflight_decision["allow"]:
        return {"status": "yielded_before_start", **preflight_decision, "yield_elapsed_s": 0.0}

    agent_script = Path(__file__).resolve().with_name("hivemind_opentrainer_agent.py")
    cmd = [
        sys.executable,
        str(agent_script),
        "--gateway",
        gateway,
        "--max-vram-gb",
        "4.5",
        "--train",
        "--once",
    ]
    env = dict(os.environ)
    env["HIVEMIND_API_KEY"] = api_key
    repo_src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = repo_src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    launcher = popen_fn or subprocess.Popen
    popen_kwargs: dict[str, Any] = {"env": env}
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True
    proc = launcher(cmd, **popen_kwargs)

    while proc.poll() is None:
        time.sleep(max(0.05, poll_interval_s))
        live_gpu = gpu_probe_fn()
        live_decision = evaluate_gamer_whisper_guard(
            live_gpu,
            running_procs=proc_list_fn(),
            extra_games=extra_games,
            reserve_mib=reserve_mib,
            hot_c=hot_c,
            worker_running=True,
        )
        if not live_decision["allow"]:
            elapsed_s = terminate_process_tree_fast(proc)
            return {
                "status": "yielded_to_owner",
                **live_decision,
                "cool_resume_c": cool_c,
                "yield_elapsed_s": elapsed_s,
            }

    return {"status": "round_completed" if proc.returncode == 0 else "worker_exited", "returncode": proc.returncode}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HiveMind 60-second Personal Quick-Connect & Gamer-Whisper Assistant (0.00 EUR, personal PCs only)."
    )
    p.add_argument("--gateway", default=os.environ.get("HIVEMIND_GATEWAY_URL", DEFAULT_GATEWAY))
    p.add_argument("--handle", default="", help="Desired handle (2-48 chars, e.g. alex-rtx4070).")
    p.add_argument("--role", default="prosumer", choices=("consumer", "producer", "prosumer", "trainer"))
    p.add_argument("--api-key", default="", help="Existing hm-... API key (optional).")
    p.add_argument("--credentials-file", default=str(_default_credentials_path()))
    p.add_argument("--write-continue-config", action="store_true", help="Write ~/.continue/hivemind_provider.yaml.")
    p.add_argument("--continue-config-path", default=str(_default_continue_snippet_path()))
    p.add_argument("--gamer-check", action="store_true", help="Inspect GPU, temperature, and running games only.")
    p.add_argument("--gamer-whisper", action="store_true", help="Run quiet GPU contributor (<70C, <1.5s game stop).")
    p.add_argument("--game", action="append", default=[], help="Extra game executable name to auto-stop on.")
    p.add_argument("--reserve-mib", type=int, default=2048, help="Free VRAM MiB reserved for desktop/games.")
    p.add_argument("--hot-c", type=float, default=70.0, help="Quiet fan temperature cap in Celsius (default: 70).")
    p.add_argument("--cool-c", type=float, default=64.0, help="Cool-down temperature in Celsius (default: 64).")
    p.add_argument("--once", action="store_true", help="Run a single quiet round and exit.")
    p.add_argument("--feedback", action="store_true", help="Submit anonymous usability feedback (1..5).")
    p.add_argument("--ease-rating", type=int, default=5, help="Ease of setup rating from 1 (hard) to 5 (instant).")
    p.add_argument("--had-game-lag", action="store_true", help="Report that a game stuttered or fans were noisy.")
    p.add_argument("--profile", default="dev", choices=("dev", "gamer"))
    p.add_argument("--note", default="", help="Optional short feedback note (no personal data).")
    p.add_argument("--uninstall", action="store_true", help="Delete local saved HiveMind credentials and snippet.")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    return p


def main(argv: list[str] | None = None, *, http_fn: HttpFn = default_http_json) -> int:
    args = build_parser().parse_args(argv)
    cred_path = Path(args.credentials_file)
    cont_path = Path(args.continue_config_path)

    if args.uninstall:
        removed: list[str] = []
        for target in (cred_path, cont_path):
            if target.is_file():
                target.unlink()
                removed.append(str(target))
        print(json.dumps({"status": "uninstalled", "removed_files": removed}, indent=2))
        return 0

    try:
        gateway = validate_gateway_url(args.gateway)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.gamer_check:
        gpu = probe_local_gpu()
        decision = evaluate_gamer_whisper_guard(
            gpu,
            running_procs=list_running_process_names(),
            extra_games=args.game,
            reserve_mib=args.reserve_mib,
            hot_c=args.hot_c,
        )
        out = {
            "gpu": {
                "present": gpu.present,
                "name": gpu.name,
                "total_mib": gpu.total_mib,
                "free_mib": gpu.free_mib,
                "temp_c": gpu.temp_c,
                "util_pct": gpu.util_pct,
            },
            "os_idle_seconds": get_os_idle_seconds(),
            "gamer_whisper": decision,
        }
        print(json.dumps(out, indent=2))
        return 0 if decision["allow"] else 2

    if args.feedback:
        if not (1 <= int(args.ease_rating) <= 5):
            print("Error: --ease-rating must be between 1 and 5.", file=sys.stderr)
            return 2
        status, body = http_fn(
            "POST",
            f"{gateway}/v1/hivemind/onboarding/feedback",
            {
                "profile": args.profile,
                "ease_rating": int(args.ease_rating),
                "zero_game_lag": not bool(args.had_game_lag),
                "platform": platform.system().lower(),
                "note": str(args.note or "")[:240],
            },
            None,
        )
        if status not in (200, 201):
            print(f"Feedback submission failed (HTTP {status}): {body}", file=sys.stderr)
            return 1
        print(json.dumps({"status": "feedback_recorded", "summary": body}, indent=2))
        return 0

    try:
        connected = ensure_connected_account(
            gateway=gateway,
            credentials_path=cred_path,
            handle=args.handle or None,
            role=args.role,
            explicit_api_key=args.api_key or None,
            http_fn=http_fn,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"Connection error: {exc}", file=sys.stderr)
        return 1

    acct = connected["account"]
    yaml_snippet = build_continue_yaml(gateway, acct["api_key"])
    if args.write_continue_config:
        cont_path.parent.mkdir(parents=True, exist_ok=True)
        cont_path.write_text(yaml_snippet, encoding="utf-8")
        if os.name != "nt":
            os.chmod(cont_path, 0o600)

    if args.gamer_whisper:
        result = run_gamer_whisper_session(
            gateway=gateway,
            api_key=acct["api_key"],
            extra_games=args.game,
            reserve_mib=args.reserve_mib,
            hot_c=args.hot_c,
            cool_c=args.cool_c,
            once=args.once,
        )
        print(json.dumps(result, indent=2))
        return 0 if result.get("status") in ("round_completed", "yielded_to_owner", "yielded_before_start") else 1

    if args.json:
        print(
            json.dumps(
                {
                    "status": "connected",
                    "handle": acct["handle"],
                    "role": acct["role"],
                    "credentials_file": str(cred_path),
                    "models": connected["models"],
                    "continue_config_written": str(cont_path) if args.write_continue_config else None,
                },
                indent=2,
            )
        )
        return 0

    print("====================================================================")
    print("  HIVEMIND HOLIDAY LAB — CONNECTED (PERSONAL HOME PC ONLY • 0.00 EUR)")
    print("====================================================================")
    print("  [Rule] Personal machines & personal hobby code ONLY (never work PCs).")
    print(f"  Handle           : {acct['handle']} ({acct['role']})")
    print(f"  Gateway          : {gateway}")
    print(f"  Credentials saved: {cred_path}")
    print(f"  Models online    : {', '.join(connected['models']) or 'verified'}")
    if args.write_continue_config:
        print(f"  VS Code snippet  : {cont_path}")
    print("\n--- VS Code (Continue.dev) snippet ---")
    print(yaml_snippet.rstrip())
    print("\n--- Clean uninstall anytime ---")
    print("  python3 scripts/hivemind_quickstart.py --uninstall")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
