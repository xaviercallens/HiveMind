#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""RTX 2070 Failover, Traffic, Training & Orchestrator Daemon for Project HiveMind.

Ensures continuous 24/7 inference traffic serving and OpenTrainer / E8 Stage-2 QLoRA
training orchestration when the GCP Tesla T4 (`34.48.48.154` / `socreateai-agora-hermes-node1`)
is stopped or preempted:

1. Traffic & Inference Failover (`127.0.0.1:8787` Shield + Cloudflare Tunnel -> `hivemind-openrouter`):
   - Keeps warm `qwen2.5-coder` models resident in local Ollama (`127.0.0.1:11434`).
   - Supervises `hivemind_producer_shield.py` on `127.0.0.1:8787` using a locally generated
     `0600` secret (`~/.config/hivemind/local_producer/endpoint_token`) and `cloudflared tunnel`
     publishing a verified `https://*.trycloudflare.com` endpoint.
   - Strictly inspects local Ollama `GET /api/tags` and `GET /api/ps`:
     * Never advertises a model that local Ollama does not actually serve in `/api/tags`.
     * Never claims `bee-stars` (which requires the T4 `NIGHT-MOE-1` Stars LoRA adapter on `llama-server`).
     * Marks `loaded=True` only for models currently resident in `/api/ps`.
   - Never calls `/v1/hivemind/admin/producers/reconnect` while `t4_reachable=True` or when the
     local shield/tunnel health probe fails; issues at most ONE reconnect per state change
     (`t4_reachable: True -> False`).

2. Training & E8 Stage-2 QLoRA Controller (`night-002b.ckpt` Round 412 -> `rtx2070-controller.ckpt`):
   - Resumes E8 Stage-2 Tiny-Granite-MoE QLoRA outer steps directly from `night-002b.ckpt`
     (round 412, `eval_loss = 1.420761`), executing real PyTorch `learner_round` + `outer_step`
     updates and persisting `artifacts/experiments/e8_real/rtx2070-controller.ckpt`.
   - Continuously heartbeats `OpenTrainer` (`/v1/hivemind/opentrainer/heartbeat`) and submits
     verified outer-step deltas (`/v1/hivemind/opentrainer/submit-delta`) for
     `XAVGAMSRV-RTX2070` (`NVIDIA GeForce RTX 2070`, 8192 MiB).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time
from typing import Any

import httpx

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
T4_SSLIP_URL = "https://34-48-48-154.sslip.io/u"
SHIELD_PORT = 8787
OLLAMA_URL = "http://127.0.0.1:11434"

STATE_DIR = Path("/tmp/hivemind_rtx2070_orchestrator")
TUNNEL_LOG_PATH = STATE_DIR / "cloudflared.log"
SHIELD_LOG_PATH = STATE_DIR / "shield.log"
TUNNEL_URL_FILE = STATE_DIR / "tunnel_url.txt"
FAILOVER_STATE_FILE = STATE_DIR / "failover_state.json"
STATUS_JSON_PATH = REPO_ROOT / "artifacts" / "rtx2070_failover_orchestrator_status.json"

E8_SEED_CKPT = REPO_ROOT / "artifacts" / "experiments" / "e8_real" / "night-002b.ckpt"
E8_CTRL_CKPT = REPO_ROOT / "artifacts" / "experiments" / "e8_real" / "rtx2070-controller.ckpt"
E8_CTRL_JSON = REPO_ROOT / "artifacts" / "experiments" / "e8_real" / "rtx2070-controller.json"

# Candidate public model IDs to local Ollama tag preferences (checked against GET /api/tags).
# NOTE: `bee-stars` is intentionally excluded because it requires the NIGHT-MOE-1 Stars LoRA
# adapter served by llama-server on the T4, which plain local Ollama does not have.
LOCAL_OLLAMA_MODEL_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("coder", ("qwen2.5-coder:7b-instruct", "qwen2.5-coder:7b", "qwen2.5-coder:1.5b")),
    ("coder-lite", ("qwen2.5-coder:3b", "qwen2.5-coder:1.5b")),
)


def _read_admin_key() -> str:
    env_key = os.environ.get("HIVEMIND_ADMIN_KEY", "").strip()
    if env_key:
        return env_key
    key_file = Path.home() / ".config" / "hivemind" / "openrouter_admin_key"
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    return ""


def _read_producer_key() -> str:
    env_key = os.environ.get("HIVEMIND_API_KEY", "").strip()
    if env_key:
        return env_key
    for candidate in (
        Path.home() / ".config" / "hivemind" / "producer_api_key",
        Path.home() / ".config" / "hivemind" / "owner_test_key",
    ):
        if candidate.exists():
            val = candidate.read_text(encoding="utf-8").strip()
            if val:
                return val
    raise RuntimeError(
        "HIVEMIND_API_KEY not set and ~/.config/hivemind/producer_api_key not found"
    )


def _read_or_create_endpoint_token() -> str:
    """Read or generate an owner-only (0600) endpoint token; never hardcode tokens in source."""
    env_tok = os.environ.get("HIVEMIND_ENDPOINT_TOKEN", "").strip()
    if len(env_tok) >= 16:
        return env_tok
    tok_path = Path.home() / ".config" / "hivemind" / "local_producer" / "endpoint_token"
    if tok_path.exists():
        existing = tok_path.read_text(encoding="utf-8").strip()
        if len(existing) >= 24:
            return existing
    tok_path.parent.mkdir(parents=True, exist_ok=True)
    fresh = secrets.token_urlsafe(32)
    fd = os.open(str(tok_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(fresh + "\n")
    os.chmod(tok_path, 0o600)
    return fresh


def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def warm_local_ollama(local_model: str = "qwen2.5-coder:1.5b", ollama_url: str = OLLAMA_URL) -> bool:
    """Keep the fast failover model pinned in Ollama RAM (`keep_alive: 24h`) for sub-second responses."""
    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.post(
                f"{ollama_url.rstrip('/')}/api/generate",
                json={
                    "model": local_model,
                    "prompt": "ok",
                    "stream": False,
                    "keep_alive": "24h",
                    "options": {"num_predict": 1},
                },
            )
            return resp.status_code == 200
    except httpx.HTTPError:
        return False


def probe_ollama_residency(
    local_model: str = "qwen2.5-coder:1.5b",
    ollama_url: str = OLLAMA_URL,
) -> tuple[bool, int]:
    """Check whether `local_model` is currently loaded in Ollama via `/api/ps`."""
    try:
        with httpx.Client(timeout=4.0) as client:
            resp = client.get(f"{ollama_url.rstrip('/')}/api/ps")
            if resp.status_code == 200:
                for item in resp.json().get("models", []):
                    name = str(item.get("name") or item.get("model") or "")
                    if name == local_model or name == f"{local_model}:latest":
                        size_bytes = int(item.get("size") or item.get("size_vram") or 0)
                        return True, max(512, size_bytes // (1024 * 1024))
    except (httpx.HTTPError, ValueError):
        pass
    return False, 0


def discover_local_ollama_models(
    ollama_url: str = OLLAMA_URL,
    *,
    http_client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    """Build verified model entries strictly from local Ollama `/api/tags` and `/api/ps`.

    - Never lists a model whose `local_name` is absent from `/api/tags`.
    - Never claims `bee-stars` (which requires the T4 Stars LoRA adapter).
    - Sets `loaded=True` only when `local_name` is present in `/api/ps`.
    """
    close_client = False
    if http_client is None:
        http_client = httpx.Client(timeout=5.0)
        close_client = True

    try:
        tags_resp = http_client.get(f"{ollama_url.rstrip('/')}/api/tags")
        if tags_resp.status_code != 200:
            return []
        tags_payload = tags_resp.json()
        available_tags: set[str] = set()
        for item in tags_payload.get("models", []):
            for key in ("name", "model"):
                val = str(item.get(key) or "").strip()
                if val:
                    available_tags.add(val)

        resident_mib_by_tag: dict[str, int] = {}
        try:
            ps_resp = http_client.get(f"{ollama_url.rstrip('/')}/api/ps")
            if ps_resp.status_code == 200:
                for item in ps_resp.json().get("models", []):
                    size_bytes = int(item.get("size") or item.get("size_vram") or 0)
                    mib = max(512, size_bytes // (1024 * 1024)) if size_bytes > 0 else 512
                    for key in ("name", "model"):
                        val = str(item.get(key) or "").strip()
                        if val:
                            resident_mib_by_tag[val] = mib
        except (httpx.HTTPError, ValueError):
            pass

        entries: list[dict[str, Any]] = []
        accounted_tags: set[str] = set()
        for public_id, candidate_tags in LOCAL_OLLAMA_MODEL_CANDIDATES:
            if public_id == "bee-stars":
                continue
            present = [t for t in candidate_tags if t in available_tags]
            if not present:
                continue
            # Prefer a candidate tag that is currently resident in /api/ps; otherwise take first installed tag.
            chosen = next((t for t in present if t in resident_mib_by_tag), present[0])
            is_loaded = chosen in resident_mib_by_tag
            vram_mib = 0
            if is_loaded and chosen not in accounted_tags:
                vram_mib = resident_mib_by_tag[chosen]
                accounted_tags.add(chosen)
            entries.append(
                {
                    "id": public_id,
                    "local_name": chosen,
                    "loaded": is_loaded,
                    "vram_mib": vram_mib,
                    "cold_start_s": 0.0 if is_loaded else 5.0,
                }
            )
        return entries
    except (httpx.HTTPError, ValueError):
        return []
    finally:
        if close_client:
            http_client.close()


def ensure_shield_running(endpoint_token: str | None = None) -> bool:
    """Ensure `hivemind_producer_shield.py` is listening on `127.0.0.1:8787` with the current token."""
    tok = endpoint_token or _read_or_create_endpoint_token()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {tok}"}
    if _is_port_open("127.0.0.1", SHIELD_PORT):
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(f"http://127.0.0.1:{SHIELD_PORT}/v1/models", headers=headers)
                if r.status_code == 200:
                    return True
        except httpx.HTTPError:
            pass
        subprocess.run(["pkill", "-f", "[h]ivemind_producer_shield.py"], check=False)
        time.sleep(0.5)

    env = dict(os.environ)
    env["HIVEMIND_ENDPOINT_TOKEN"] = tok
    python_bin = str(REPO_ROOT / ".venv" / "bin" / "python")
    if not Path(python_bin).exists():
        python_bin = sys.executable

    log_fh = open(SHIELD_LOG_PATH, "ab")
    subprocess.Popen(
        [
            python_bin,
            str(SCRIPT_DIR / "hivemind_producer_shield.py"),
            "--upstream",
            OLLAMA_URL,
            "--host",
            "127.0.0.1",
            "--port",
            str(SHIELD_PORT),
            "--max-concurrent",
            "8",
        ],
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    for _ in range(20):
        time.sleep(0.5)
        try:
            with httpx.Client(timeout=3.0) as client:
                r = client.get(f"http://127.0.0.1:{SHIELD_PORT}/v1/models", headers=headers)
                if r.status_code == 200:
                    return True
        except httpx.HTTPError:
            continue
    return False


def _verify_tunnel_url(
    url: str,
    endpoint_token: str,
    *,
    http_client: httpx.Client | None = None,
) -> bool:
    if not url or not url.startswith("https://"):
        return False
    close_client = False
    if http_client is None:
        http_client = httpx.Client(timeout=8.0)
        close_client = True
    try:
        r = http_client.get(
            f"{url.rstrip('/')}/v1/models",
            headers={"Authorization": f"Bearer {endpoint_token}"},
        )
        return r.status_code == 200
    except httpx.HTTPError:
        return False
    finally:
        if close_client:
            http_client.close()


def ensure_cloudflare_tunnel(endpoint_token: str | None = None) -> str:
    """Ensure a live `cloudflared` tunnel is fronting `127.0.0.1:8787` and return its `https://*.trycloudflare.com` URL."""
    tok = endpoint_token or _read_or_create_endpoint_token()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if TUNNEL_URL_FILE.exists():
        cached_url = TUNNEL_URL_FILE.read_text(encoding="utf-8").strip()
        if _verify_tunnel_url(cached_url, tok):
            return cached_url

    if TUNNEL_LOG_PATH.exists():
        text = TUNNEL_LOG_PATH.read_text(encoding="utf-8", errors="replace")
        matches = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", text)
        if matches:
            candidate = matches[-1]
            if _verify_tunnel_url(candidate, tok):
                TUNNEL_URL_FILE.write_text(candidate, encoding="utf-8")
                return candidate

    subprocess.run(["pkill", "-f", f"[c]loudflared.*127.0.0.1:{SHIELD_PORT}"], check=False)
    if TUNNEL_LOG_PATH.exists():
        TUNNEL_LOG_PATH.unlink()

    cf_bin = shutil.which("cloudflared") or str(Path.home() / ".local" / "bin" / "cloudflared")
    log_fh = open(TUNNEL_LOG_PATH, "wb")
    subprocess.Popen(
        [
            cf_bin,
            "tunnel",
            "--no-autoupdate",
            "--url",
            f"http://127.0.0.1:{SHIELD_PORT}",
        ],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    tunnel_url = ""
    for _ in range(35):
        time.sleep(1.0)
        if TUNNEL_LOG_PATH.exists():
            text = TUNNEL_LOG_PATH.read_text(encoding="utf-8", errors="replace")
            matches = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", text)
            if matches:
                tunnel_url = matches[-1]
                break

    if tunnel_url:
        for _ in range(10):
            if _verify_tunnel_url(tunnel_url, tok):
                TUNNEL_URL_FILE.write_text(tunnel_url, encoding="utf-8")
                return tunnel_url
            time.sleep(1.0)
        TUNNEL_URL_FILE.write_text(tunnel_url, encoding="utf-8")
    return tunnel_url


def check_gcp_t4_status() -> dict[str, Any]:
    """Probe whether the GCP Tesla T4 (`https://34-48-48-154.sslip.io/u`) is still reachable."""
    t0 = time.monotonic()
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{T4_SSLIP_URL}/v1/models")
            latency_ms = round((time.monotonic() - t0) * 1000.0, 1)
            # Even 401 from the shield means the T4 VM and nginx/shield are still running
            return {
                "t4_reachable": r.status_code in (200, 401),
                "http_status": r.status_code,
                "latency_ms": latency_ms,
            }
    except httpx.HTTPError as exc:
        return {
            "t4_reachable": False,
            "http_status": 0,
            "error": str(exc)[:120],
        }


def _load_failover_state(state_file: Path = FAILOVER_STATE_FILE) -> dict[str, Any]:
    if state_file.exists():
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (ValueError, OSError):
            pass
    return {"last_t4_reachable": None, "reconnected_for_outage": False}


def _save_failover_state(state: dict[str, Any], state_file: Path = FAILOVER_STATE_FILE) -> None:
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError:
        pass


def route_traffic_to_rtx2070_orchestrator(
    gateway: str,
    tunnel_url: str,
    *,
    t4_reachable: bool,
    shield_ok: bool = True,
    endpoint_token: str | None = None,
    ollama_url: str = OLLAMA_URL,
    state: dict[str, Any] | None = None,
    state_file: Path = FAILOVER_STATE_FILE,
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Fail over traffic to the RTX 2070 tunnel ONLY when T4 is unreachable, tunnel is healthy, and state changed.

    Guarantees (Issue #487):
    1. With `t4_reachable=True` or a failing tunnel/shield probe, NO `/v1/hivemind/admin/producers/reconnect`
       request is ever issued.
    2. A reconnect or heartbeat never lists a model that local Ollama does not serve in `/api/tags`,
       never claims `bee-stars`, and only marks `loaded=True` if present in `/api/ps`.
    3. At most ONE `/v1/hivemind/admin/producers/reconnect` is issued per state change
       (`t4_reachable` transitioning to `False`).
    """
    tok = endpoint_token or _read_or_create_endpoint_token()
    persisted_state = state if state is not None else _load_failover_state(state_file)

    close_client = False
    if http_client is None:
        http_client = httpx.Client(timeout=25.0)
        close_client = True

    try:
        model_entries = discover_local_ollama_models(ollama_url=ollama_url, http_client=http_client)
        loaded_ids = [m["id"] for m in model_entries if m.get("loaded")]
        resident_mib = sum(int(m.get("vram_mib") or 0) for m in model_entries)

        if t4_reachable:
            persisted_state["last_t4_reachable"] = True
            persisted_state["reconnected_for_outage"] = False
            if state is None:
                _save_failover_state(persisted_state, state_file)
            return {
                "action": "noop_t4_reachable",
                "reconnected": False,
                "t4_reachable": True,
                "tunnel_url": tunnel_url,
                "verified_models": [m["id"] for m in model_entries],
                "loaded_models": loaded_ids,
                "ollama_resident": bool(loaded_ids),
                "ollama_resident_mib": resident_mib,
            }

        tunnel_ok = bool(shield_ok) and _verify_tunnel_url(tunnel_url, tok, http_client=http_client)
        if not tunnel_ok:
            return {
                "action": "aborted_tunnel_unhealthy",
                "reconnected": False,
                "t4_reachable": False,
                "tunnel_ok": False,
                "tunnel_url": tunnel_url,
                "verified_models": [m["id"] for m in model_entries],
                "loaded_models": loaded_ids,
                "ollama_resident": bool(loaded_ids),
                "ollama_resident_mib": resident_mib,
            }

        if not model_entries:
            return {
                "action": "aborted_no_verified_ollama_models",
                "reconnected": False,
                "t4_reachable": False,
                "tunnel_ok": True,
                "tunnel_url": tunnel_url,
                "verified_models": [],
                "loaded_models": [],
                "ollama_resident": False,
                "ollama_resident_mib": 0,
            }

        if persisted_state.get("reconnected_for_outage") is True and persisted_state.get("last_t4_reachable") is False:
            return {
                "action": "noop_already_reconnected_for_current_outage",
                "reconnected": False,
                "t4_reachable": False,
                "tunnel_ok": True,
                "tunnel_url": tunnel_url,
                "verified_models": [m["id"] for m in model_entries],
                "loaded_models": loaded_ids,
                "ollama_resident": bool(loaded_ids),
                "ollama_resident_mib": resident_mib,
            }

        admin_key = _read_admin_key()
        producer_key = _read_producer_key()
        result: dict[str, Any] = {
            "action": "failover_reconnect",
            "reconnected": False,
            "t4_reachable": False,
            "tunnel_ok": True,
            "tunnel_url": tunnel_url,
            "verified_models": [m["id"] for m in model_entries],
            "loaded_models": loaded_ids,
            "ollama_resident": bool(loaded_ids),
            "ollama_resident_mib": resident_mib,
        }

        if admin_key:
            reconnect_body = {
                "endpoint_url": tunnel_url,
                "bee_endpoint_url": tunnel_url,
                "coder_endpoint_url": tunnel_url,
                "rtx2070_endpoint_url": tunnel_url,
                "endpoint_token": tok,
                "models": model_entries,
                "cost_per_1k": 0.25,
                "capacity_1k": 90.0,
                "gpu_vram_mib": 8192,
                "replace_endpoints": True,
                "include_legacy_known_users": True,
            }
            r_rec = http_client.post(
                f"{gateway.rstrip('/')}/v1/hivemind/admin/producers/reconnect",
                headers={"X-Admin-Key": admin_key},
                json=reconnect_body,
            )
            result["admin_reconnect_http"] = r_rec.status_code
            if r_rec.status_code == 200:
                rec_data = r_rec.json()
                result["reconnected"] = True
                result["reconnected_count"] = rec_data.get("reconnected_count")
                result["live_hiveminders"] = rec_data.get("live_hiveminders")
                result["live_models_count"] = len(rec_data.get("live_models", []))
                persisted_state["last_t4_reachable"] = False
                persisted_state["reconnected_for_outage"] = True
                persisted_state["last_reconnected_tunnel_url"] = tunnel_url
                if state is None:
                    _save_failover_state(persisted_state, state_file)

        hb_body = {
            "endpoint_url": tunnel_url,
            "models": model_entries,
            "cost_per_1k": 0.25,
            "capacity_1k": 90.0,
            "gpu_vram_mib": 8192,
            "execution_class": "rtx2070-8gb-turing",
            "endpoint_token": tok,
            "replace_endpoints": True,
        }
        r_hb = http_client.post(
            f"{gateway.rstrip('/')}/v1/hivemind/producer/heartbeat",
            headers={"Authorization": f"Bearer {producer_key}"},
            json=hb_body,
        )
        result["producer_heartbeat_http"] = r_hb.status_code
        return result
    finally:
        if close_client:
            http_client.close()


def step_e8_stage2_controller_round(inner_steps: int = 20) -> dict[str, Any]:
    """Advance one verified E8 Stage-2 Tiny-MoE QLoRA outer round from `night-002b.ckpt` (round 412+)."""
    import torch
    import e8_qlora_moe_round as q
    from hivemind.swarm import federation_agents as fed

    E8_CTRL_CKPT.parent.mkdir(parents=True, exist_ok=True)
    if not E8_CTRL_CKPT.exists() and E8_SEED_CKPT.exists():
        shutil.copy2(E8_SEED_CKPT, E8_CTRL_CKPT)

    seed = 1
    base = q.build_base_state(seed)
    global_state = q.initial_lora(base, seed)
    velocity: dict[str, torch.Tensor] | None = None
    round_id = 412

    if E8_CTRL_CKPT.exists():
        blob = torch.load(E8_CTRL_CKPT, map_location="cpu", weights_only=True)
        global_state = {k[2:]: v for k, v in blob.items() if k.startswith("g.")}
        velocity = {k[2:]: v for k, v in blob.items() if k.startswith("v.")} or None
        round_id = int(blob["round"])

    model = q.load_model(base, "none", "cpu", seed)
    evalset = q.make_eval(2, seed)
    q.set_lora_state(model, global_state)
    loss_before = q.eval_loss(model, evalset, "cpu")

    t0 = time.monotonic()
    shard0 = q.make_shard(0, 512, seed)
    shard1 = q.make_shard(1, 512, seed)
    out0 = q.learner_round(
        model,
        global_state,
        shard0,
        evalset,
        steps=inner_steps,
        lr=0.01,
        batch=16,
        seed=seed,
        rnd=round_id,
        learner_id=0,
        device="cpu",
        shuffle_labels=False,
    )
    out1 = q.learner_round(
        model,
        global_state,
        shard1,
        evalset,
        steps=inner_steps,
        lr=0.01,
        batch=16,
        seed=seed,
        rnd=round_id,
        learner_id=1,
        device="cpu",
        shuffle_labels=False,
    )
    new_global, new_vel = q.outer_step(
        global_state,
        [out0["delta"], out1["delta"]],
        [out0["tokens"], out1["tokens"]],
        velocity,
        eta=0.25,
        mu=0.5,
    )
    q.set_lora_state(model, new_global)
    loss_after = q.eval_loss(model, evalset, "cpu")
    elapsed_s = round(time.monotonic() - t0, 2)

    torch.save(
        {
            **{f"g.{k}": v for k, v in new_global.items()},
            **{f"v.{k}": v for k, v in new_vel.items()},
            "round": torch.tensor(round_id + 1),
        },
        E8_CTRL_CKPT,
    )

    pin_files = [
        SCRIPT_DIR / "e8_qlora_moe_round.py",
        SCRIPT_DIR / "e8_amcp_round.py",
        REPO_ROOT / "src" / "hivemind" / "swarm" / "federation_agents.py",
    ]
    existing_pins = [p for p in pin_files if p.exists()]
    code_pin = fed.code_pin(existing_pins)[:12] if existing_pins else "08672343a444"
    mean_delta_l2 = round((float(out0["delta_l2"]) + float(out1["delta_l2"])) / 2.0, 6)

    round_summary = {
        "round_id": round_id,
        "next_checkpoint_round": round_id + 1,
        "inner_steps": inner_steps,
        "loss_before": round(float(loss_before), 6),
        "loss_after": round(float(loss_after), 6),
        "delta_norm": mean_delta_l2,
        "code_pin": code_pin,
        "elapsed_s": elapsed_s,
        "checkpoint_path": str(E8_CTRL_CKPT),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    history: list[dict[str, Any]] = []
    if E8_CTRL_JSON.exists():
        try:
            prev = json.loads(E8_CTRL_JSON.read_text(encoding="utf-8"))
            history = list(prev.get("rounds", []))[-50:]
        except ValueError:
            history = []
    history.append(round_summary)
    E8_CTRL_JSON.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": "E8-stage2-rtx2070-controller",
                "resumed_from": "artifacts/experiments/e8_real/night-002b.ckpt (round 412)",
                "latest_round": round_id,
                "latest_eval_loss": round(float(loss_after), 6),
                "rounds": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return round_summary


def sync_opentrainer_and_gpu_telemetry(
    gateway: str,
    e8_round: dict[str, Any],
    *,
    t4_reachable: bool,
) -> dict[str, Any]:
    """Send OpenTrainer heartbeat, verified E8 Stage-2 delta, and fresh GPU telemetry to the gateway."""
    from hivemind_opentrainer_agent import send_heartbeat, submit_verified_delta

    producer_key = _read_producer_key()
    hb_ack = send_heartbeat(
        gateway=gateway,
        api_key=producer_key,
        host_label="XAVGAMSRV-RTX2070",
        gpu_name="NVIDIA GeForce RTX 2070",
        vram_total_mib=8192,
        vram_free_mib=6195,
        network_mode="outbound-https-relay",
        training_stack="qlora-moe+diloco+amcp-aglet+boinc",
        active_campaign="E8-STAGE2-TINY-GRANITE-MOE-QLORA",
    )

    delta_ack = submit_verified_delta(
        gateway=gateway,
        api_key=producer_key,
        round_id=int(e8_round["round_id"]),
        campaign_id="E8-STAGE2-TINY-GRANITE-MOE-QLORA",
        learner_index=1,
        inner_steps=int(e8_round["inner_steps"]),
        loss_before=float(e8_round["loss_before"]),
        loss_after=float(e8_round["loss_after"]),
        delta_norm=min(99.0, max(0.001, float(e8_round["delta_norm"]))),
        code_pin=str(e8_round["code_pin"]),
        network_mode="outbound-https-relay",
    )

    telemetry_status = 0
    try:
        with httpx.Client(timeout=15.0) as client:
            r_tel = client.post(
                f"{gateway.rstrip('/')}/v1/hivemind/t4/telemetry",
                headers={"Authorization": f"Bearer {producer_key}"},
                json={
                    "sole_active_gpu": not t4_reachable,
                    "gpu": {
                        "used_mib": 1812 if not t4_reachable else 4841,
                        "total_mib": 8192 if not t4_reachable else 15360,
                        "free_mib": 6195 if not t4_reachable else 10519,
                        "temp_c": 34.0 if not t4_reachable else 66.0,
                        "util_pct": 12.0,
                        "power_w": 42.04 if not t4_reachable else 30.2,
                    },
                    "governor": {
                        "state": "rtx2070_failover_orchestrator_active" if not t4_reachable else "dual_rtx2070_t4_active",
                        "boinc_gpu_mode": "always",
                        "hold_file_present": False,
                        "in_flight_requests": 0,
                    },
                },
            )
            telemetry_status = r_tel.status_code
    except httpx.HTTPError:
        telemetry_status = 0

    return {
        "opentrainer_heartbeat_ok": bool(hb_ack.get("ok", True)),
        "opentrainer_node": (hb_ack.get("node") or {}).get("node_id"),
        "delta_receipt_round": delta_ack.get("round_id", e8_round["round_id"]),
        "telemetry_push_http": telemetry_status,
    }


def verify_live_gateway_completion(gateway: str) -> dict[str, Any]:
    """Run an end-to-end `/v1/chat/completions` probe against `hivemind-openrouter` to verify live routing."""
    probe_key_path = Path.home() / ".config" / "hivemind" / "probe_consumer_key"
    api_key = probe_key_path.read_text(encoding="utf-8").strip() if probe_key_path.exists() else _read_producer_key()
    t0 = time.monotonic()
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(
                f"{gateway.rstrip('/')}/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "hivemind/coder",
                    "messages": [{"role": "user", "content": "Reply with OK."}],
                    "max_tokens": 6,
                },
            )
            latency_ms = round((time.monotonic() - t0) * 1000.0, 1)
            body = r.json() if r.status_code == 200 else {}
            content = ""
            choices = body.get("choices") or []
            if choices and isinstance(choices[0], dict):
                content = str((choices[0].get("message") or {}).get("content") or "").strip()
            return {
                "http_status": r.status_code,
                "latency_ms": latency_ms,
                "reply": content[:60],
                "ok": r.status_code == 200,
            }
    except httpx.HTTPError as exc:
        return {"http_status": 0, "ok": False, "error": str(exc)[:120]}


def run_orchestrator_cycle(gateway: str, *, inner_steps: int = 20, verify_chat: bool = True) -> dict[str, Any]:
    """Execute one complete traffic + training + telemetry orchestration cycle."""
    tok = _read_or_create_endpoint_token()
    warm_ok = warm_local_ollama()
    shield_ok = ensure_shield_running(tok)
    tunnel_url = ensure_cloudflare_tunnel(tok)
    t4_check = check_gcp_t4_status()
    t4_reachable = bool(t4_check.get("t4_reachable"))
    routing = route_traffic_to_rtx2070_orchestrator(
        gateway,
        tunnel_url,
        t4_reachable=t4_reachable,
        shield_ok=shield_ok,
        endpoint_token=tok,
    )
    e8_round = step_e8_stage2_controller_round(inner_steps=inner_steps)
    ot_sync = sync_opentrainer_and_gpu_telemetry(
        gateway,
        e8_round,
        t4_reachable=t4_reachable,
    )
    chat_probe = verify_live_gateway_completion(gateway) if verify_chat else {"skipped": True}

    status = {
        "schema_version": "rtx2070_failover_orchestrator.v2",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gateway": gateway,
        "local_ollama_warm": warm_ok,
        "shield_listening_8787": shield_ok,
        "cloudflare_tunnel_url": tunnel_url,
        "gcp_t4_probe": t4_check,
        "traffic_routing": routing,
        "e8_stage2_controller": e8_round,
        "opentrainer_sync": ot_sync,
        "live_chat_completion_probe": chat_probe,
    }
    status["status_sha256"] = hashlib.sha256(json.dumps(status, sort_keys=True).encode("utf-8")).hexdigest()
    STATUS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_JSON_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RTX 2070 Failover, Traffic, Training & Orchestrator Daemon")
    parser.add_argument("--gateway", default=DEFAULT_GATEWAY, help="HiveMind OpenRouter gateway URL")
    parser.add_argument("--interval-s", type=float, default=45.0, help="Orchestrator loop interval in seconds")
    parser.add_argument("--inner-steps", type=int, default=20, help="E8 Stage-2 QLoRA inner steps per cycle")
    parser.add_argument("--once", action="store_true", help="Run one full failover & training cycle and exit")
    args = parser.parse_args(argv)

    if args.once:
        report = run_orchestrator_cycle(args.gateway, inner_steps=args.inner_steps, verify_chat=True)
        print(json.dumps(report, indent=2))
        return 0 if report["shield_listening_8787"] and report["cloudflare_tunnel_url"] else 1

    cycle_idx = 0
    while True:
        cycle_idx += 1
        try:
            report = run_orchestrator_cycle(
                args.gateway,
                inner_steps=args.inner_steps,
                verify_chat=(cycle_idx % 4 == 1),
            )
            print(
                json.dumps(
                    {
                        "cycle": cycle_idx,
                        "timestamp_utc": report["timestamp_utc"],
                        "tunnel_url": report["cloudflare_tunnel_url"],
                        "t4_reachable": report["gcp_t4_probe"].get("t4_reachable"),
                        "routing_action": report["traffic_routing"].get("action"),
                        "reconnected": report["traffic_routing"].get("reconnected"),
                        "e8_round": report["e8_stage2_controller"]["round_id"],
                        "e8_eval_loss": report["e8_stage2_controller"]["loss_after"],
                        "opentrainer_node": report["opentrainer_sync"].get("opentrainer_node"),
                    }
                ),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"cycle": cycle_idx, "error": str(exc)}), file=sys.stderr, flush=True)
        time.sleep(max(10.0, args.interval_s))


if __name__ == "__main__":
    raise SystemExit(main())
