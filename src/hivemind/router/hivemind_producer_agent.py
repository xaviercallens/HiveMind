#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HiveMind producer agent: heartbeats a local OpenAI-compatible server into the OpenRouter gateway.

The agent does not run inference. It advertises an endpoint you already run (Ollama, vLLM or llama.cpp server), the
models behind it, a price per 1k tokens and a capacity per epoch. The gateway routes consumer requests to live
producers and credits HMT for the tokens they serve. Only dependency: ``pip install httpx``.

Pick a model for your card (no account needed):
    python hivemind_producer_agent.py --gateway https://GATEWAY --suggest --vram-gb 12
    python hivemind_producer_agent.py --gateway https://GATEWAY --suggest --detect-vram

Start sharing (preset = a capability tier from the gateway's catalog; price and capacity default from the tier):
    export HIVEMIND_API_KEY=hm-...                 # your producer key, shown once at registration
    export HIVEMIND_ENDPOINT_TOKEN=$(python -c "import secrets;print(secrets.token_urlsafe(32))")
    python hivemind_producer_agent.py --gateway https://GATEWAY --endpoint https://YOUR_PUBLIC_URL --preset coder

Or fully manual:
    python hivemind_producer_agent.py --gateway https://GATEWAY --endpoint https://YOUR_PUBLIC_URL \
        --model coder=qwen2.5-coder:7b --cost-per-1k 0.4 --capacity-1k 60

The API key is read from the environment only, so it never appears in shell history or process listings. The endpoint
must be reachable over https at a public address; the gateway's SSRF guard rejects anything else. If you set
HIVEMIND_ENDPOINT_TOKEN, the gateway presents it as ``Authorization: Bearer`` to your endpoint, so put the producer
shield (scripts/hivemind_producer_shield.py) in front of your model server and only the gateway can use your GPU.
Before the first heartbeat the agent checks that your endpoint really lists the model; use --skip-preflight to skip.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

import httpx

HEARTBEAT_PATH: str = "/v1/hivemind/producer/heartbeat"
CATALOG_PATH: str = "/v1/hivemind/catalog"
REMOTE_CONFIG_PATH: str = "/v1/hivemind/remote/config"
API_KEY_ENV: str = "HIVEMIND_API_KEY"
ENDPOINT_TOKEN_ENV: str = "HIVEMIND_ENDPOINT_TOKEN"
DEFAULT_EXECUTION_CLASS: str = "gpu"
DEFAULT_INTERVAL_S: float = 60.0
HTTP_TIMEOUT_S: float = 30.0
DEFAULT_LOCAL_ENDPOINT: str = "http://127.0.0.1:11434"  # the local Ollama; read-only residency probe only
RESIDENCY_TIMEOUT_S: float = 2.0
DEFAULT_COLD_START_S: float = 60.0

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def parse_model_spec(raw: str) -> dict[str, str]:
    """Parse one ``--model id=local_name`` argument into a heartbeat model entry."""
    if not isinstance(raw, str):
        raise argparse.ArgumentTypeError(f"--model expects id=local_name, got {raw!r}")
    model_id, separator, local_name = raw.partition("=")
    model_id = model_id.strip()
    local_name = local_name.strip()
    if not separator or not model_id or not local_name:
        raise argparse.ArgumentTypeError(f"--model expects id=local_name, got {raw!r}")
    return {"id": model_id, "local_name": local_name}


def probe_residency(
    local_endpoint: str, *, transport: httpx.BaseTransport | None = None, timeout_s: float = RESIDENCY_TIMEOUT_S
) -> dict[str, int] | None:
    """Which models the LOCAL Ollama holds in memory right now: ``{name: vram_mib}`` from ``GET /api/ps``, or ``None``.

    Read-only and best effort: any failure (server down, timeout, odd JSON) returns ``None`` so that the heartbeat simply reports nothing.
    """
    if not isinstance(local_endpoint, str) or not local_endpoint.strip():
        raise ValueError("local_endpoint must be a non-empty string")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(float(timeout_s)) or float(timeout_s) <= 0.0:
        raise ValueError("timeout_s must be a positive finite number")
    try:
        with httpx.Client(timeout=timeout_s, transport=transport, follow_redirects=False) as client:
            response = client.get(local_endpoint.rstrip("/") + "/api/ps")
            response.raise_for_status()
            listed = response.json().get("models")
    except (httpx.HTTPError, ValueError, AttributeError):
        return None
    if not isinstance(listed, list):
        return None
    resident: dict[str, int] = {}
    for item in listed:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("model")
        size = item.get("size_vram")
        if not isinstance(name, str) or not name:
            continue
        mib = int(size) // (1024 * 1024) if isinstance(size, (int, float)) and not isinstance(size, bool) and size >= 0 else 0
        resident[name] = mib
    return resident


def probe_model_sizes_mib(
    local_endpoint: str, *, transport: httpx.BaseTransport | None = None, timeout_s: float = RESIDENCY_TIMEOUT_S
) -> dict[str, int] | None:
    """Return ``{model_name: disk_weight_mib}`` from Ollama ``GET /api/tags``, or ``None`` if unavailable."""
    if not isinstance(local_endpoint, str) or not local_endpoint.strip():
        raise ValueError("local_endpoint must be a non-empty string")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(float(timeout_s)) or float(timeout_s) <= 0.0:
        raise ValueError("timeout_s must be a positive finite number")
    try:
        with httpx.Client(timeout=timeout_s, transport=transport, follow_redirects=False) as client:
            response = client.get(local_endpoint.rstrip("/") + "/api/tags")
            response.raise_for_status()
            listed = response.json().get("models")
    except (httpx.HTTPError, ValueError, AttributeError):
        return None
    if not isinstance(listed, list):
        return None
    sizes: dict[str, int] = {}
    for item in listed:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("model")
        size = item.get("size")
        if not isinstance(name, str) or not name:
            continue
        if isinstance(size, (int, float)) and not isinstance(size, bool) and size >= 0:
            sizes[name] = int(size) // (1024 * 1024)
    return sizes


def filter_models_by_vram_budget(
    models: Sequence[dict[str, str]],
    model_sizes_mib: Mapping[str, int] | None,
    *,
    gpu_vram_mib: int | None,
    pinned_vram_mib: int = 0,
    min_free_vram_mib: int = 0,
) -> tuple[list[dict[str, str]], list[str]]:
    """Filter out models whose on-disk weight size exceeds the safe VRAM budget after pinned daemons and free headroom.

    Returns ``(kept_models, dropped_descriptions)``. If ``model_sizes_mib`` or ``gpu_vram_mib`` is ``None``, all models
    are kept unchanged.
    """
    if not isinstance(models, Sequence) or isinstance(models, (str, bytes)):
        raise TypeError("models must be a Sequence of model dicts")
    if model_sizes_mib is not None and not isinstance(model_sizes_mib, Mapping):
        raise TypeError("model_sizes_mib must be a Mapping or None")
    if model_sizes_mib is None or gpu_vram_mib is None or gpu_vram_mib <= 0:
        return [dict(m) for m in models], []
    budget_mib = max(0, int(gpu_vram_mib) - max(0, int(pinned_vram_mib)) - max(0, int(min_free_vram_mib)))
    kept: list[dict[str, str]] = []
    dropped: list[str] = []
    for entry in models:
        local = entry["local_name"]
        size_mib = model_sizes_mib.get(local)
        if size_mib is None:
            size_mib = model_sizes_mib.get(f"{local}:latest")
        if size_mib is not None and size_mib > budget_mib:
            dropped.append(f"{entry['id']}={local} ({size_mib} MiB > {budget_mib} MiB budget)")
            continue
        kept.append(dict(entry))
    return kept, dropped


def apply_residency(
    payload: dict[str, Any], resident: dict[str, int] | None, *, cold_start_s: float, gpu_vram_mib: int | None
) -> dict[str, Any]:
    """Add the optional residency fields to ``payload`` (in place and returned). ``resident=None`` (probe failed) adds nothing per model."""
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")
    if resident is not None and not isinstance(resident, Mapping):
        raise TypeError("resident must be a Mapping or None")
    if isinstance(cold_start_s, bool) or not isinstance(cold_start_s, (int, float)) or not math.isfinite(float(cold_start_s)) or float(cold_start_s) < 0.0:
        raise ValueError("cold_start_s must be a non-negative finite number")
    if gpu_vram_mib is not None and gpu_vram_mib > 0:
        payload["gpu_vram_mib"] = int(gpu_vram_mib)
    if resident is None:
        return payload
    for entry in payload["models"]:
        local = entry["local_name"]
        hit = local if local in resident else (f"{local}:latest" if f"{local}:latest" in resident else None)
        if hit is not None:
            entry["loaded"] = True
            entry["vram_mib"] = resident[hit]
        else:
            entry["loaded"] = False
            entry["cold_start_s"] = cold_start_s
    return payload


MAX_CONTROL_VRAM_MIB: int = 1_048_576  # same ceiling the gateway applies to card sizes


def parse_control(raw: object) -> dict[str, Any]:
    """Validate the ``control`` object of a heartbeat ack. Only ``paused`` and ``vram_cap_mib`` are ever read; anything else is ignored.

    Raises ``ValueError`` for a malformed object (the caller ignores it and logs once). No command, path or process information is accepted.
    """
    if not isinstance(raw, dict):
        raise ValueError("control is not an object")
    paused = raw.get("paused")
    cap = raw.get("vram_cap_mib")
    if not isinstance(paused, bool):
        raise ValueError("control.paused is not a boolean")
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= MAX_CONTROL_VRAM_MIB):
        raise ValueError("control.vram_cap_mib is not a sane positive integer")
    version = raw.get("version")
    return {"paused": paused, "vram_cap_mib": cap, "version": version if isinstance(version, int) and not isinstance(version, bool) else None}


def write_control_state(path: str, control: dict[str, Any]) -> None:
    """Atomically publish the control state for the local shield (owner-only file, mode 0600). Raises ``OSError`` on failure."""
    if not isinstance(path, (str, os.PathLike)) or not str(path).strip():
        raise ValueError("path must be a non-empty path")
    if not isinstance(control, Mapping):
        raise TypeError("control must be a Mapping")
    body = json.dumps({"paused": control["paused"], "vram_cap_mib": control["vram_cap_mib"], "version": control.get("version")}, sort_keys=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(body)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_control_state(path: str) -> dict[str, Any] | None:
    """The control state saved by an earlier run, or ``None`` when there is none or it is unreadable."""
    try:
        with open(path, encoding="utf-8") as handle:
            return parse_control(json.loads(handle.read(4096)))
    except (OSError, ValueError):
        return None


def apply_control(payload: dict[str, Any], control: dict[str, Any] | None) -> dict[str, Any]:
    """Make the heartbeat obey the remote control state (in place and returned). ``None`` changes nothing.

    Paused: advertise zero capacity and mark ``paused``. VRAM cap: models reported loaded whose ``vram_mib`` would not fit (kept in order while the sum
    fits) are left out of the heartbeat, and the cap is reported back as ``vram_cap_mib``; if none is left the capacity is zero.
    """
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")
    if control is not None and not isinstance(control, Mapping):
        raise TypeError("control must be a Mapping or None")
    if control is None:
        return payload
    if control["paused"]:
        payload["capacity_1k"] = 0.0
        payload["paused"] = True
    cap = control["vram_cap_mib"]
    if cap is not None:
        payload["vram_cap_mib"] = cap
        used = 0
        kept: list[dict[str, Any]] = []
        for entry in payload["models"]:
            need = entry.get("vram_mib") if entry.get("loaded") is True else None
            if isinstance(need, int) and not isinstance(need, bool):
                if used + need > cap:
                    continue
                used += need
            kept.append(entry)
        if not kept:
            payload["capacity_1k"] = 0.0  # the gateway wants at least one model entry; zero capacity advertises nothing
        else:
            payload["models"] = kept
    return payload


def build_heartbeat_payload(
    *,
    endpoint: str,
    models: list[dict[str, str]],
    cost_per_1k: float,
    capacity_1k: float,
    execution_class: str,
    endpoint_token: str | None = None,
) -> dict[str, Any]:
    """Build the JSON body for POST /v1/hivemind/producer/heartbeat."""
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("endpoint must be a non-empty string")
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list")
    if isinstance(cost_per_1k, bool) or not isinstance(cost_per_1k, (int, float)) or not math.isfinite(float(cost_per_1k)) or float(cost_per_1k) < 0.0:
        raise ValueError("cost_per_1k must be a finite non-negative number")
    if isinstance(capacity_1k, bool) or not isinstance(capacity_1k, (int, float)) or not math.isfinite(float(capacity_1k)) or float(capacity_1k) < 0.0:
        raise ValueError("capacity_1k must be a finite non-negative number")
    if not isinstance(execution_class, str) or not execution_class.strip():
        raise ValueError("execution_class must be a non-empty string")
    payload: dict[str, Any] = {
        "endpoint_url": endpoint,
        "models": [dict(model) for model in models],
        "cost_per_1k": cost_per_1k,
        "capacity_1k": capacity_1k,
        "execution_class": execution_class,
    }
    if endpoint_token:
        payload["endpoint_token"] = endpoint_token
    return payload


def send_heartbeat(
    client: httpx.Client, gateway: str, api_key: str, payload: dict[str, Any]
) -> dict[str, Any]:
    """POST one heartbeat and return the gateway's JSON object. Raises on HTTP or JSON errors."""
    if not callable(getattr(client, "post", None)):
        raise TypeError("client must expose a callable post method")
    if not isinstance(gateway, str) or not gateway.strip():
        raise ValueError("gateway must be a non-empty string")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("api_key must be a non-empty string")
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a Mapping")
    response = client.post(
        gateway.rstrip("/") + HEARTBEAT_PATH,
        json=payload,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError("heartbeat response is not a JSON object")
    return body


def fetch_catalog(client: httpx.Client, gateway: str) -> list[dict[str, Any]]:
    """The gateway's public capability tiers (smallest GPU requirement first)."""
    if not callable(getattr(client, "get", None)):
        raise TypeError("client must expose a callable get method")
    if not isinstance(gateway, str) or not gateway.strip():
        raise ValueError("gateway must be a non-empty string")
    response = client.get(gateway.rstrip("/") + CATALOG_PATH)
    response.raise_for_status()
    tiers = response.json().get("tiers")
    if not isinstance(tiers, list) or not tiers:
        raise ValueError("catalog response has no tiers")
    return [tier for tier in tiers if isinstance(tier, dict)]


def fetch_remote_config(client: httpx.Client, gateway: str, api_key: str) -> dict[str, Any]:
    """Fetch the producer's remote control configuration from GET /v1/hivemind/remote/config."""
    if not callable(getattr(client, "get", None)):
        raise TypeError("client must expose a callable get method")
    if not isinstance(gateway, str) or not gateway.strip():
        raise ValueError("gateway must be a non-empty string")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("api_key must be a non-empty string")
    response = client.get(
        gateway.rstrip("/") + REMOTE_CONFIG_PATH,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    response.raise_for_status()
    rc = response.json().get("remote_config")
    if not isinstance(rc, dict) or not rc:
        raise ValueError("remote config response has no remote_config object")
    return dict(rc)


def recommend(tiers: Sequence[dict[str, Any]], vram_gb: float) -> dict[str, Any] | None:
    """The biggest tier whose recommended minimum memory fits ``vram_gb``, or ``None``."""
    if isinstance(vram_gb, bool) or not isinstance(vram_gb, (int, float)) or not math.isfinite(float(vram_gb)) or float(vram_gb) < 0.0:
        raise ValueError("vram_gb must be a finite non-negative number")
    fitting = [tier for tier in tiers if float(tier["min_vram_gb"]) <= vram_gb]
    return max(fitting, key=lambda tier: float(tier["min_vram_gb"])) if fitting else None


def detect_vram_gb(run: Runner = subprocess.run) -> float | None:
    """Largest NVIDIA card's memory in GB via ``nvidia-smi``, or ``None`` when it cannot be read.

    ``nvidia-smi`` can fail even on a working card when the driver is unreachable from the current user; pass
    ``--vram-gb`` in that case.
    """
    if not callable(run):
        raise TypeError("run must be callable")
    try:
        result = run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    values: list[float] = []
    for line in result.stdout.splitlines():
        try:
            values.append(float(line.strip()) / 1024.0)  # MiB -> GiB
        except ValueError:
            continue
    return max(values) if values else None


def preflight(client: httpx.Client, endpoint: str, token: str | None, local_names: Sequence[str]) -> list[str]:
    """Names in ``local_names`` that the endpoint's ``/v1/models`` does not list (empty list = ready).

    Raises ``httpx.HTTPError`` or ``ValueError`` when the endpoint cannot be read at all.
    """
    if not callable(getattr(client, "get", None)):
        raise TypeError("client must expose a callable get method")
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("endpoint must be a non-empty string")
    if not isinstance(local_names, Sequence) or isinstance(local_names, (str, bytes)):
        raise TypeError("local_names must be a Sequence of strings")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = client.get(endpoint.rstrip("/") + "/v1/models", headers=headers)
    response.raise_for_status()
    data = response.json().get("data")
    if not isinstance(data, list):
        raise ValueError("endpoint /v1/models did not return a model list")
    served = {str(item.get("id")) for item in data if isinstance(item, dict)}
    return [name for name in local_names if name not in served and f"{name}:latest" not in served]


POC_NOTICE = (
    "Proof of concept (PoC): not a production service. Until HiveMind is fully secured, prompts you serve and requests you make "
    "must contain only public, non-sensitive data (no secrets, personal, private or enterprise data); the people using your node "
    "are anonymous, and nothing yet verifies producers or their output."
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Advertise a local OpenAI-compatible server to the HiveMind OpenRouter gateway."
    )
    parser.add_argument("--gateway", required=True, help="Gateway base URL, e.g. https://host")
    parser.add_argument("--endpoint", help="Public https base URL of your OpenAI-compatible server (gateway appends /v1/...)")
    parser.add_argument("--preset", metavar="TIER", help="Serve a catalog tier (e.g. coder); sets model, price and capacity")
    parser.add_argument("--local-name", help="With --preset: the name your server uses for the tier's model, if not the recommended tag")
    parser.add_argument(
        "--model", action="append", default=[], type=parse_model_spec, metavar="ID=LOCAL_NAME",
        help="Public model id = the name your server uses. Repeat for several models.",
    )
    parser.add_argument("--cost-per-1k", type=float, help="Your cost in HMT per 1,000 tokens (>= 0); defaults from --preset")
    parser.add_argument("--capacity-1k", type=float, help="Capacity per epoch in 1,000-token units (> 0); defaults from --preset")
    parser.add_argument("--execution-class", default=DEFAULT_EXECUTION_CLASS, help=f"Hardware class label, default {DEFAULT_EXECUTION_CLASS!r}")
    parser.add_argument("--interval-s", type=float, default=DEFAULT_INTERVAL_S, help=f"Seconds between heartbeats (default {DEFAULT_INTERVAL_S:g})")
    parser.add_argument("--once", action="store_true", help="Send a single heartbeat, print the gateway's JSON response and exit")
    parser.add_argument("--suggest", action="store_true", help="Print the best tier for --vram-gb / --detect-vram and exit (no key needed)")
    parser.add_argument("--vram-gb", type=float, help="Your GPU memory in GB (for --suggest)")
    parser.add_argument("--detect-vram", action="store_true", help="Read GPU memory from nvidia-smi (for --suggest)")
    parser.add_argument("--residency", action="store_true", help="Report live GPU VRAM residency on each heartbeat (enabled by default unless --no-residency)")
    parser.add_argument("--no-residency", action="store_true", help="Do not ask the local model server which models are loaded; heartbeats then carry no residency data")
    parser.add_argument("--local-endpoint", default=DEFAULT_LOCAL_ENDPOINT, help=f"Local Ollama base URL for the read-only GET /api/ps residency probe (default {DEFAULT_LOCAL_ENDPOINT})")
    parser.add_argument("--routed-upstream", action="append", default=[], metavar="LOCAL_NAME=URL", help="Mark a non-Ollama routed model (e.g. hf.co/Vxtzq/Crowd-v1=http://127.0.0.1:8791) as loaded while its local /healthz or /v1/models answers 200")
    parser.add_argument("--cold-start-s", type=float, default=DEFAULT_COLD_START_S, help=f"Your estimate of the seconds needed to load a model that is not resident (default {DEFAULT_COLD_START_S:g}, max 600)")
    parser.add_argument("--gpu-vram-mib", type=int, help="Card memory in MiB to report with residency (default: read from nvidia-smi when possible)")
    parser.add_argument("--pinned-vram-mib", type=int, default=0, help="VRAM in MiB reserved by pinned daemons such as llama-server (default 0)")
    parser.add_argument("--min-free-vram-mib", type=int, default=0, help="Minimum free VRAM headroom in MiB to preserve when filtering models (e.g. 4300 on Tesla T4)")
    parser.add_argument("--control-state-file", metavar="PATH", help="Where to publish the remote pause/VRAM-cap state (owner-only file, mode 0600) for the producer shield's --control-state-file; omit to not write it")
    parser.add_argument("--sync-remote-config", action="store_true", help="Sync node_state and min_price_per_1k from /v1/hivemind/remote/config before each heartbeat")
    parser.add_argument("--skip-preflight", action="store_true", help="Do not check that your endpoint lists the model before the first heartbeat")
    return parser


def _suggest(client: httpx.Client, args: argparse.Namespace, parser: argparse.ArgumentParser, runner: Runner) -> int:
    vram = args.vram_gb
    if vram is None and args.detect_vram:
        vram = detect_vram_gb(runner)
        if vram is None:
            parser.error("could not read GPU memory with nvidia-smi; pass --vram-gb")
    if vram is None or vram < 0:
        parser.error("--suggest needs --vram-gb N (N >= 0) or --detect-vram")
    try:
        tiers = fetch_catalog(client, args.gateway)
    except (httpx.HTTPError, ValueError) as exc:
        print(f"could not read the catalog: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    best = recommend(tiers, vram)
    result: dict[str, Any] = {"vram_gb": round(vram, 2), "recommended": best,
                              "fits": [tier["tier_id"] for tier in tiers if float(tier["min_vram_gb"]) <= vram]}
    if best is None:
        result["advice"] = "No tier fits this card; you can still use HiveMind as a consumer."
    else:
        result["next"] = f"ollama pull {best['ollama_tag']}   then run with --preset {best['tier_id']}"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    runner: Runner = subprocess.run,
    residency_transport: httpx.BaseTransport | None = None,
) -> int:
    """Run the producer agent. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    print(f"NOTICE: {POC_NOTICE}", file=sys.stderr)
    gateway_url = urlsplit(args.gateway)
    if gateway_url.scheme != "https" and not (gateway_url.scheme == "http" and gateway_url.hostname in ("localhost", "127.0.0.1", "::1")):
        parser.error("--gateway must be https (plain http is allowed only for localhost): your API key is sent in every heartbeat")

    with httpx.Client(timeout=HTTP_TIMEOUT_S, transport=transport) as client:
        if args.suggest:
            return _suggest(client, args, parser, runner)

        api_key = os.environ.get(API_KEY_ENV, "")
        if not api_key:
            parser.error(f"set the {API_KEY_ENV} environment variable to your producer API key")
        if not args.endpoint:
            parser.error("--endpoint is required")
        endpoint_url = urlsplit(args.endpoint)
        if endpoint_url.scheme != "https" and not (endpoint_url.scheme == "http" and endpoint_url.hostname in ("localhost", "127.0.0.1", "::1")):
            parser.error("--endpoint must be https (plain http is allowed only for localhost): the gateway sends your endpoint token with every request")
        models: list[dict[str, str]] = list(args.model)
        cost, capacity = args.cost_per_1k, args.capacity_1k
        if args.preset:
            try:
                tiers = fetch_catalog(client, args.gateway)
            except (httpx.HTTPError, ValueError) as exc:
                print(f"could not read the catalog: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 1
            preset_key = args.preset.removeprefix("hivemind/")
            if preset_key in ("layerstorm-frontier-moe", "layerstorm-moe", "moe-layerstorm"):
                preset_key = "moe-general"
            tier = next((t for t in tiers if t.get("tier_id") == preset_key), None)
            if tier is None:
                parser.error(f"unknown preset {args.preset!r}; choose one of: {', '.join(str(t.get('tier_id')) for t in tiers)}")
            models.append({"id": str(tier["tier_id"]), "local_name": args.local_name or str(tier["ollama_tag"])})
            cost = float(tier["suggested_cost_per_1k"]) if cost is None else cost
            capacity = float(tier["suggested_capacity_1k"]) if capacity is None else capacity
        if not models:
            parser.error("give --preset TIER or at least one --model ID=LOCAL_NAME")
        if cost is None or capacity is None:
            parser.error("--cost-per-1k and --capacity-1k are required without --preset")
        if cost < 0:
            parser.error("--cost-per-1k must be >= 0")
        if capacity <= 0:
            parser.error("--capacity-1k must be > 0")
        if args.interval_s <= 0:
            parser.error("--interval-s must be > 0")

        token = os.environ.get(ENDPOINT_TOKEN_ENV) or None
        if not args.skip_preflight:
            try:
                missing = preflight(client, args.endpoint, token, [m["local_name"] for m in models])
            except (httpx.HTTPError, ValueError) as exc:
                print(f"preflight failed: cannot read {args.endpoint}/v1/models ({type(exc).__name__}: {exc}). "
                      "Check the tunnel/shield, the endpoint token, or use --skip-preflight.", file=sys.stderr)
                return 1
            if missing:
                print(f"preflight failed: your endpoint does not list {missing}. Pull it first (for Ollama: ollama pull NAME) "
                      "or pass --local-name / --model with the name your server uses.", file=sys.stderr)
                return 1

        if not 0 <= args.cold_start_s <= 600:
            parser.error("--cold-start-s must be between 0 and 600")
        gpu_mib: int | None = None
        if not args.no_residency:
            gpu_mib = args.gpu_vram_mib
            if gpu_mib is None:
                detected = detect_vram_gb(runner)
                gpu_mib = int(detected * 1024) if detected is not None else None
            if (args.min_free_vram_mib > 0 or args.pinned_vram_mib > 0) and gpu_mib is not None:
                sizes_mib = probe_model_sizes_mib(args.local_endpoint, transport=residency_transport)
                kept_models, dropped_models = filter_models_by_vram_budget(
                    models,
                    sizes_mib,
                    gpu_vram_mib=gpu_mib,
                    pinned_vram_mib=args.pinned_vram_mib,
                    min_free_vram_mib=args.min_free_vram_mib,
                )
                if dropped_models:
                    print(
                        f"vram guard: dropping models exceeding {gpu_mib} MiB (pinned={args.pinned_vram_mib} MiB, "
                        f"min_free={args.min_free_vram_mib} MiB): {', '.join(dropped_models)}",
                        file=sys.stderr,
                        flush=True,
                    )
                if not kept_models:
                    print("vram guard: no advertised models fit within the safe VRAM budget; refusing to start.", file=sys.stderr)
                    return 1
                models = kept_models

        routed_upstreams: dict[str, str] = {}
        for item in getattr(args, "routed_upstream", ()):
            r_name, sep, r_url = str(item).partition("=")
            if sep and r_name.strip() and r_url.strip().startswith(("http://127.0.0.1", "http://localhost")):
                routed_upstreams[r_name.strip()] = r_url.strip().rstrip("/")

        def make_payload() -> dict[str, Any]:
            built = build_heartbeat_payload(
                endpoint=args.endpoint, models=models, cost_per_1k=cost, capacity_1k=capacity,
                execution_class=args.execution_class, endpoint_token=token,
            )
            if args.no_residency:
                return built
            resident = probe_residency(args.local_endpoint, transport=residency_transport)
            if routed_upstreams:
                merged_res = dict(resident or {})
                with httpx.Client(timeout=RESIDENCY_TIMEOUT_S, transport=residency_transport, follow_redirects=False) as rclient:
                    for r_name, r_url in routed_upstreams.items():
                        try:
                            hr = rclient.get(r_url + "/healthz")
                            if hr.status_code == 200:
                                hjson = hr.json() if hr.content else {}
                                dev_info = hjson.get("device") if isinstance(hjson, dict) else {}
                                est_vram = dev_info.get("estimated_model_vram_mib", 0) if isinstance(dev_info, dict) else 0
                                merged_res[r_name] = int(est_vram) if isinstance(est_vram, int) and est_vram >= 0 else 0
                                continue
                            mr = rclient.get(r_url + "/v1/models")
                            if mr.status_code == 200:
                                merged_res[r_name] = 0
                        except (httpx.HTTPError, ValueError):
                            pass
                resident = merged_res if merged_res or resident is not None else None
            return apply_residency(built, resident, cold_start_s=args.cold_start_s, gpu_vram_mib=gpu_mib)

        # Remote control (HM-S16-05): the gateway's heartbeat ack may carry {"paused", "vram_cap_mib"}; the next heartbeat obeys it and the state is
        # published for the shield. Only those two values are ever read. A state saved by an earlier run is honoured until the gateway says otherwise.
        control: dict[str, Any] | None = read_control_state(args.control_state_file) if args.control_state_file else None
        warned = False

        def beat() -> dict[str, Any]:
            nonlocal control, warned
            current_payload = make_payload()
            if args.sync_remote_config:
                rc = fetch_remote_config(client, args.gateway, api_key)
                if rc.get("node_state") == "paused":
                    return {"status": "paused_by_remote_control", "remote_config": rc}
                current_payload["cost_per_1k"] = max(float(current_payload["cost_per_1k"]), float(rc.get("min_price_per_1k", current_payload["cost_per_1k"])))
            body = send_heartbeat(client, args.gateway, api_key, apply_control(current_payload, control))
            if "control" in body:
                try:
                    received = parse_control(body["control"])
                except ValueError as exc:
                    if not warned:
                        warned = True
                        print(f"ignoring malformed control object in the heartbeat ack ({exc}); continuing as before", file=sys.stderr, flush=True)
                else:
                    control = received
                    if args.control_state_file:
                        try:
                            write_control_state(args.control_state_file, received)
                        except OSError as exc:
                            print(f"could not write {args.control_state_file}: {type(exc).__name__}", file=sys.stderr, flush=True)
            return body

        if args.once:
            try:
                body = beat()
            except (httpx.HTTPError, ValueError) as exc:
                print(f"heartbeat failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 1
            print(json.dumps(body, indent=2, sort_keys=True))
            return 0

        try:
            while True:
                try:
                    body = beat()
                    if body.get("status") == "paused_by_remote_control":
                        print("remote control: node_state is paused, skipping heartbeat", flush=True)
                    else:
                        print(f"heartbeat ok: {json.dumps(body, sort_keys=True)}", flush=True)
                except (httpx.HTTPError, ValueError) as exc:
                    print(f"heartbeat failed, retrying: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                time.sleep(args.interval_s)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
