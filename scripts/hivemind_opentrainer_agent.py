#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""OpenTrainer CLI Agent — Firewall-Friendly Outbound GPU Co-Training (BOINC + AMCP + Kafka + Signed Aglets).

Connects a trainer GPU (e.g. Windows 11 RTX 2070 ``XAVGAMSRV`` 8 GB, RTX 4070 12 GB, or Linux Tesla T4 16 GB)
to the central ``OpenTrainer`` component on ``hivemind-openrouter`` over **outbound HTTPS (TCP/443)** with:
  - Zero inbound firewall ports opened
  - Zero ``0.0.0.0`` listeners
  - Signed Aglet task-offer verification (``dispatch`` -> ``arrival`` -> ``clone`` -> ``result`` -> ``retract``)
  - Optional Stage-2 QLoRA / DiLoCo delta submission
  - ``--crowdgpt``: lend the GPU to CrowdGPT's Crowd-v1 through CrowdGPT's signed client, with
    automatic pause/stop (see docs/OPENTRAINER_CROWDGPT_CONTRIBUTOR.md)

Usage:
  export HIVEMIND_API_KEY=<your hm-... trainer key>
  python scripts/hivemind_opentrainer_agent.py \\
    --gateway https://hivemind-openrouter-1003063861791.us-central1.run.app \\
    --network-mode outbound-https-relay \\
    --check-firewall \\
    --once
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_GATEWAY = "https://hivemind-openrouter-1003063861791.us-central1.run.app"
DEFAULT_PINNED_CROWD_REVISION = "58a33099424efc0a07fbdc16eaaf99a07dc95fb9"
DEFAULT_TRAINER_WORK_DIR = os.path.join(os.path.expanduser("~"), ".cache", "hivemind", "opentrainer")
REPO_ROOT = Path(__file__).resolve().parents[1]


def detect_local_gpu() -> tuple[str, int, int]:
    """Query ``nvidia-smi`` for GPU name, total VRAM (MiB), and free VRAM (MiB); return ``('', 0, 0)`` if absent."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            first_line = proc.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first_line.split(",")]
            if len(parts) >= 3:
                total_mib = int(float(parts[1]))
                free_mib = int(float(parts[2]))
                if total_mib > 0 and free_mib >= 0:
                    return parts[0], max(512, total_mib), min(max(512, total_mib), max(0, free_mib))
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return "", 0, 0


def verify_firewall_traversal(gateway: str, network_mode: str) -> dict[str, Any]:
    """Verify outbound HTTPS connectivity to OpenTrainer with zero inbound listener requirements."""
    url = gateway.rstrip("/") + "/v1/hivemind/opentrainer/status"
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=15) as resp:
        latency_ms = round((time.monotonic() - started) * 1000.0, 2)
        payload = json.loads(resp.read().decode("utf-8"))
    modes = payload.get("firewall_traversal_modes") or []
    if isinstance(modes, list):
        mode_spec = next((m for m in modes if m.get("mode_id") == network_mode), modes[0] if modes else {})
    else:
        mode_spec = modes.get(network_mode) or modes.get("outbound-https-relay") or {}
    offer = payload.get("active_task_offer") or {}
    return {
        "ok": True,
        "gateway": gateway,
        "network_mode": network_mode,
        "outbound_https_reachable": True,
        "roundtrip_latency_ms": latency_ms,
        "inbound_ports_required": int(mode_spec.get("inbound_ports_required", 0)),
        "zero_bind_all_interfaces": True,
        "opentrainer_version": payload.get("version", "1.0.0-s36"),
        "amcp_topic": offer.get("topic") or payload.get("amcp_topic", "hivemind.amcp.federation.v1"),
    }


def send_heartbeat(
    *,
    gateway: str,
    api_key: str,
    host_label: str,
    gpu_name: str,
    vram_total_mib: int,
    vram_free_mib: int,
    network_mode: str,
    training_stack: str,
    active_campaign: str,
) -> dict[str, Any]:
    url = gateway.rstrip("/") + "/v1/hivemind/opentrainer/heartbeat"
    body = json.dumps(
        {
            "host_label": host_label,
            "gpu_name": gpu_name,
            "vram_total_mib": vram_total_mib,
            "vram_free_mib": vram_free_mib,
            "network_mode": network_mode,
            "training_stack": training_stack,
            "active_campaign": active_campaign,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_round_report(report_path: str) -> dict[str, Any]:
    """Read measured round results from a JSON report and hash the real delta file bytes.

    The report must contain finite ``loss_before``, ``loss_after`` and ``delta_norm`` numbers and a
    ``delta_path`` string. A relative ``delta_path`` is resolved against the report's directory.
    Raises ``ValueError`` when the report or the delta file is missing or malformed.
    """
    try:
        with open(report_path, "r", encoding="utf-8") as fh:
            report = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"round report unreadable: {report_path}: {exc}") from exc
    if not isinstance(report, dict):
        raise ValueError("round report must be a JSON object")
    measured: dict[str, float] = {}
    for key in ("loss_before", "loss_after", "delta_norm"):
        value = report.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"round report field {key!r} must be a finite number")
        measured[key] = float(value)
    delta_path_raw = report.get("delta_path")
    if not isinstance(delta_path_raw, str) or not delta_path_raw.strip():
        raise ValueError("round report field 'delta_path' must be a non-empty string")
    delta_path = delta_path_raw if os.path.isabs(delta_path_raw) else os.path.join(os.path.dirname(os.path.abspath(report_path)), delta_path_raw)
    try:
        with open(delta_path, "rb") as fh:
            delta_bytes = fh.read()
    except OSError as exc:
        raise ValueError(f"delta file unreadable: {delta_path}: {exc}") from exc
    if not delta_bytes:
        raise ValueError(f"delta file is empty: {delta_path}")
    return {
        **measured,
        "delta_sha256": hashlib.sha256(delta_bytes).hexdigest(),
        "delta_bytes": len(delta_bytes),
    }


def _safetensors_named_key_count(path: Path) -> int:
    """Return the count of non-metadata tensor keys in a safetensors header, or 0 on raw/invalid files."""
    try:
        size = path.stat().st_size
        if size < 10:
            return 0
        with path.open("rb") as fh:
            raw_n = fh.read(8)
            if len(raw_n) != 8:
                return 0
            (header_len,) = struct.unpack("<Q", raw_n)
            if header_len <= 0 or header_len > 32 * 1024 * 1024 or 8 + header_len > size:
                return 0
            header = json.loads(fh.read(header_len).decode("utf-8"))
        if not isinstance(header, dict):
            return 0
        return sum(1 for k in header if k != "__metadata__")
    except (OSError, struct.error, UnicodeDecodeError, ValueError, TypeError):
        return 0


def _ensure_src_on_path() -> None:
    src_dir = str(REPO_ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _fetch_personalized_assignment(gateway: str, handle: str, vram_free_mib: int) -> dict[str, Any]:
    """Fetch personalized expert shard assignment from the gateway with safe fallback for older/stub gateways."""
    from urllib.parse import urlencode

    qs = urlencode({"handle": handle, "role": "trainer", "vram_mib": max(0, int(vram_free_mib))})
    url = f"{gateway.rstrip('/')}/v1/hivemind/opentrainer/crowdgpt-weekend-raid?{qs}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assign = payload.get("personalized_assignment") if isinstance(payload, dict) else None
        if isinstance(assign, dict) and assign.get("resident_expert_indices"):
            return assign
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError):
        pass
    return {"resident_expert_indices": [0, 1], "recommended_inner_steps_h": 8}


def _resolve_pinned_crowd_v1_files(
    *,
    cache_dir: str,
    pinned_revision: str,
    offline: bool,
) -> tuple[Path, Path | None, str]:
    """Locate or download pinned multi-key `Vxtzq/Crowd-v1` `model.safetensors` + `tokenizer.json`."""
    roots: list[Path] = []
    if cache_dir.strip():
        roots.append(Path(cache_dir).expanduser())
    for env_key in ("HF_HUB_CACHE", "HF_HOME"):
        val = os.environ.get(env_key, "").strip()
        if val:
            roots.append(Path(val) / ("hub" if env_key == "HF_HOME" else ""))
    roots.append(Path("/mnt/disks/disk-socrateai-local-1/hf_hub_cache"))
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    rev = pinned_revision.strip() or DEFAULT_PINNED_CROWD_REVISION
    cand_folders: list[Path] = []
    for root in roots:
        for base in (root, root / "models--Vxtzq--Crowd-v1" / "snapshots" / rev, root / "snapshots" / rev):
            if base.is_dir():
                cand_folders.append(base)
        snaps = root / "models--Vxtzq--Crowd-v1" / "snapshots"
        if snaps.is_dir():
            for child in sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if child.is_dir() and child not in cand_folders:
                    cand_folders.append(child)

    for folder in cand_folders:
        for fname in ("model.safetensors", "model_bf16.safetensors", "model_fp32.safetensors"):
            wp = folder / fname
            if wp.is_file() and _safetensors_named_key_count(wp) >= 16:
                tok = folder / "tokenizer.json"
                return wp, (tok if tok.is_file() else None), folder.name

    if offline:
        raise RuntimeError(
            f"No cached multi-key Vxtzq/Crowd-v1 safetensors found for revision {rev!r} under {[str(r) for r in roots]}."
        )

    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download pinned Vxtzq/Crowd-v1 weights (or pass --cache-dir / --small-arch)."
        ) from exc

    dl_cache = cache_dir.strip() or None
    wp = Path(hf_hub_download(repo_id="Vxtzq/Crowd-v1", filename="model.safetensors", revision=rev, cache_dir=dl_cache))
    if _safetensors_named_key_count(wp) < 16:
        raise RuntimeError(f"Downloaded {wp} lacks named parameter keys (got {_safetensors_named_key_count(wp)}).")
    tok: Path | None = None
    try:
        tok = Path(hf_hub_download(repo_id="Vxtzq/Crowd-v1", filename="tokenizer.json", revision=rev, cache_dir=dl_cache))
    except Exception:
        tok = None
    return wp, tok, rev


def _stream_named_safetensors(model: Any, safetensors_path: Path) -> set[str]:
    """Stream-load matching named tensors from a safetensors file into `model` without mmap bloat."""
    import torch

    dt_map = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    param_map = dict(model.named_parameters())
    matched: set[str] = set()
    with safetensors_path.open("rb") as fh, torch.no_grad():
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len).decode("utf-8"))
        data_base = 8 + header_len
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            target = param_map.get(key)
            if target is None or str(meta.get("dtype")) not in dt_map:
                continue
            lo, hi = int(meta["data_offsets"][0]), int(meta["data_offsets"][1])
            nbytes = hi - lo
            src_dt = dt_map[str(meta["dtype"])]
            shape = tuple(int(d) for d in meta["shape"])
            if tuple(target.shape) != shape:
                continue
            fh.seek(data_base + lo)
            buf = bytearray(fh.read(nbytes))
            src_t = torch.frombuffer(buf, dtype=src_dt).reshape(shape)
            target.copy_(src_t.to(device=target.device, dtype=target.dtype))
            del src_t, buf
            matched.add(key)
    return matched


def _load_public_token_streams(
    *,
    s21_shard_dir: str,
    tokenizer_path: Path | None,
    vocab_size: int,
    seq_len: int,
    offline: bool,
) -> tuple[Any, Any, str]:
    """Load real S21 tokenized shards or public open-corpus tokens + held-out eval stream."""
    import torch

    if s21_shard_dir.strip():
        sd = Path(s21_shard_dir).expanduser()
        if (sd / "manifest.json").is_file() and (sd / "trainer_shard.u32.bin").is_file():
            tr_raw = torch.frombuffer(bytearray((sd / "trainer_shard.u32.bin").read_bytes()), dtype=torch.int32).to(torch.long)
            he_file = sd / ("held_out_shard.u32.bin" if (sd / "held_out_shard.u32.bin").is_file() else "trainer_shard.u32.bin")
            he_raw = torch.frombuffer(bytearray(he_file.read_bytes()), dtype=torch.int32).to(torch.long)
            return tr_raw.clamp(0, vocab_size - 1), he_raw.clamp(0, vocab_size - 1), f"s21_shards:{sd}"

    docs: list[str] = []
    token_source = "public_open_docs"
    if not offline:
        ds_url = (
            "https://datasets-server.huggingface.co/rows"
            "?dataset=HuggingFaceFW%2Ffineweb-edu&config=sample-10BT&split=train&offset=0&length=12"
        )
        try:
            req = urllib.request.Request(ds_url, headers={"Accept": "application/json"}, method="GET")
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            for row in data.get("rows") or []:
                txt = ((row.get("row") or {}).get("text") or "").strip()
                if len(txt) >= 64:
                    docs.append(txt)
            if docs:
                token_source = "hf_datasets_server:HuggingFaceFW/fineweb-edu"
        except Exception:
            docs = []

    if not docs:
        for rel in ("README.md", "docs/TRAINER_JOIN.md"):
            p = REPO_ROOT / rel
            if p.is_file():
                docs.append(p.read_text(encoding="utf-8", errors="replace")[:8192])
    if not docs:
        docs = [
            "Open decentralized federated language modeling combines local inner optimization steps with "
            "sparse mixture-of-experts specialization and held-out validation loss verification across nodes. "
            * 16
        ]

    ids: list[int] = []
    if tokenizer_path is not None and tokenizer_path.is_file() and vocab_size > 256:
        try:
            from tokenizers import Tokenizer  # type: ignore[import-not-found]

            tok = Tokenizer.from_file(str(tokenizer_path))
            for doc in docs:
                ids.extend(int(t) % vocab_size for t in tok.encode(doc).ids)
            token_source = f"{token_source}+hf_tokenizer"
        except Exception:
            ids = []
    if len(ids) < max(seq_len * 4, 256):
        joined = "\n".join(docs).encode("utf-8", errors="ignore")
        ids = [b % min(256, vocab_size) for b in joined]
        token_source = f"{token_source}+utf8_bytes"

    need = max(seq_len * 8, 512)
    while len(ids) < need:
        ids.extend(ids)
    split = max(seq_len * 2, int(len(ids) * 0.8))
    train_tok = torch.tensor(ids[:split], dtype=torch.long)
    eval_tok = torch.tensor(ids[split : split + max(seq_len * 2, 128)], dtype=torch.long)
    if eval_tok.numel() < seq_len + 1:
        eval_tok = train_tok[: max(seq_len * 2, 128)].clone()
    return train_tok, eval_tok, token_source


def execute_local_trainer_round(
    *,
    gateway: str,
    host_label: str,
    round_id: int,
    work_dir: str,
    cache_dir: str,
    s21_shard_dir: str,
    pinned_revision: str,
    inner_steps: int,
    seq_len: int,
    lr: float,
    vram_free_mib: int,
    vram_total_mib: int,
    max_vram_gb: float,
    small_arch: bool,
    cpu_only: bool,
    offline: bool,
) -> dict[str, Any]:
    """Run one real PyTorch Sparse-MoE OpenDiLoCo inner round and persist measured artifacts for submission."""
    _ensure_src_on_path()
    import torch
    from safetensors.torch import save_file

    from hivemind.swarm.crowdgpt_moe_opendiloco import (
        run_crowd_moe_opendiloco_inner_steps,
        upcycle_crowd_v1_to_sparse_moe,
    )
    from hivemind.swarm.gcp_t4_crowd_diloco import (
        CrowdV1RuntimeConfig,
        CrowdV1WithEngram,
        forward_crowd_v1_checkpointed,
    )

    assign = _fetch_personalized_assignment(gateway, host_label, vram_free_mib)
    raw_experts = assign.get("resident_expert_indices") or [0, 1]
    res_experts = tuple(sorted({int(x) for x in raw_experts}))[:4] or (0, 1)

    use_cuda = (not cpu_only) and (not small_arch) and torch.cuda.is_available()
    exec_dev = "cuda" if use_cuda else "cpu"
    if use_cuda and max_vram_gb > 0.0 and vram_total_mib > 0:
        frac = max(0.1, min(0.95, (float(max_vram_gb) * 1024.0) / float(vram_total_mib)))
        try:
            torch.cuda.set_per_process_memory_fraction(frac, 0)
        except Exception:
            pass

    if small_arch:
        torch.manual_seed(20261115 + round_id)
        cfg = CrowdV1RuntimeConfig(
            vocab_size=256,
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            intermediate_size=256,
            max_position_embeddings=max(256, seq_len * 2),
            engram_buckets=512,
        )
        model = CrowdV1WithEngram(cfg)
        tokenizer_path: Path | None = None
        rev_used = "local-small-arch"
        matched_count = len(dict(model.named_parameters()))
    else:
        weights_path, tokenizer_path, rev_used = _resolve_pinned_crowd_v1_files(
            cache_dir=cache_dir,
            pinned_revision=pinned_revision,
            offline=offline,
        )
        cfg = CrowdV1RuntimeConfig()
        dt = torch.bfloat16 if (use_cuda and torch.cuda.get_device_capability(0)[0] >= 8) else (torch.float16 if use_cuda else torch.float32)
        prev_dt = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dt)
            with torch.device("meta"):
                model = CrowdV1WithEngram(cfg)
        finally:
            torch.set_default_dtype(prev_dt)
        model.engram.to_empty(device="cpu")
        model.engram.to(dtype=torch.bfloat16)
        model.wte.to_empty(device=exec_dev)
        model.blocks.to_empty(device=exec_dev)
        model.ln.to_empty(device=exec_dev)
        if not cfg.tie_word_embeddings:
            model.head.to_empty(device=exec_dev)
        else:
            model.head.weight = model.wte.weight
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim))
        freqs = torch.outer(torch.arange(cfg.max_position_embeddings, dtype=torch.float32), inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        model.register_buffer("cos_cached", emb.cos()[None, None].to(device=exec_dev), persistent=False)
        model.register_buffer("sin_cached", emb.sin()[None, None].to(device=exec_dev), persistent=False)
        matched = _stream_named_safetensors(model, weights_path)
        matched_count = len(matched)
        if matched_count == 0:
            raise RuntimeError(f"Refusing uninitialized Crowd-v1 weights: 0 keys matched in {weights_path}")

    split_report = upcycle_crowd_v1_to_sparse_moe(
        model,
        split_mode="lora_domain_experts",
        num_experts=8,
        top_k=2,
        lora_rank=8,
        routing_mode="sigma_topk",
        resident_expert_indices=res_experts,
    )
    train_tok, eval_tok, token_source = _load_public_token_streams(
        s21_shard_dir=s21_shard_dir,
        tokenizer_path=tokenizer_path,
        vocab_size=model.cfg.vocab_size,
        seq_len=seq_len,
        offline=offline,
    )

    eval_win = min(model.cfg.max_position_embeddings, max(4, min(seq_len, int(eval_tok.numel()))))
    eval_batch = eval_tok[:eval_win].reshape(1, eval_win).to(device=exec_dev, dtype=torch.long)

    def _measure_eval_loss() -> float:
        model.eval()
        with torch.no_grad():
            _, l_t = forward_crowd_v1_checkpointed(
                model,
                eval_batch,
                labels=eval_batch,
                use_grad_checkpointing=False,
                token_chunk_size=min(256, eval_win),
            )
            return float(l_t.item()) if l_t is not None else 0.0

    loss_before = _measure_eval_loss()
    inner_out = run_crowd_moe_opendiloco_inner_steps(
        model,
        train_tok,
        steps=max(1, int(inner_steps)),
        batch_size=1,
        seq_len=max(4, int(seq_len)),
        lr=float(lr),
        num_router_fragments=4,
        device=exec_dev,
    )
    loss_after = _measure_eval_loss()

    pseudo_grads: dict[str, Any] = inner_out["pseudo_grads"]
    delta_norm = float(math.sqrt(sum(float(v.float().pow(2).sum().item()) for v in pseudo_grads.values())))

    out_dir = Path(work_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    delta_path = out_dir / f"opentrainer_round_{round_id}.safetensors"
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in pseudo_grads.items()}, str(delta_path))

    report_path = out_dir / f"opentrainer_round_{round_id}_report.json"
    report_obj = {
        "round_id": int(round_id),
        "experimental_derivative": True,
        "base_model": "Vxtzq/Crowd-v1",
        "pinned_hf_revision": rev_used,
        "weights_matched_keys": matched_count,
        "device": exec_dev,
        "resident_experts": list(res_experts),
        "trainable_parameters": split_report.trainable_parameters,
        "inner_steps": int(inner_out["steps_completed"]),
        "tokens_processed": int(inner_out["tokens_processed"]),
        "token_source": token_source,
        "loss_before": round(loss_before, 6),
        "loss_after": round(loss_after, 6),
        "delta_norm": round(max(1e-6, delta_norm), 6),
        "delta_path": delta_path.name,
    }
    report_path.write_text(json.dumps(report_obj, indent=2, sort_keys=True), encoding="utf-8")
    loaded = load_round_report(str(report_path))
    return {
        **loaded,
        "report_path": str(report_path),
        "delta_file": str(delta_path),
        "resident_experts": list(res_experts),
        "token_source": token_source,
        "pinned_hf_revision": rev_used,
        "device": exec_dev,
    }


def submit_verified_delta(
    *,
    gateway: str,
    api_key: str,
    round_id: int,
    campaign_id: str,
    learner_index: int,
    inner_steps: int,
    loss_before: float,
    loss_after: float,
    delta_norm: float,
    delta_sha256: str,
    code_pin: str,
    network_mode: str,
) -> dict[str, Any]:
    url = gateway.rstrip("/") + "/v1/hivemind/opentrainer/submit-delta"
    body = json.dumps(
        {
            "round_id": round_id,
            "campaign_id": campaign_id,
            "learner_index": learner_index,
            "inner_steps": inner_steps,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "delta_norm": delta_norm,
            "delta_sha256": delta_sha256,
            "code_pin": code_pin,
            "network_mode": network_mode,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HiveMind OpenTrainer Agent — Firewall-Friendly Outbound GPU Co-Training (BOINC + AMCP + Kafka + Aglets)."
    )
    parser.add_argument(
        "--gateway",
        "--server-url",
        dest="gateway",
        default=os.environ.get("HIVEMIND_GATEWAY_URL", DEFAULT_GATEWAY),
        help="OpenTrainer gateway URL (also accepts --server-url)",
    )
    parser.add_argument(
        "--network-mode",
        default="outbound-https-relay",
        choices=("outbound-https-relay", "amcp-kafka-aglet", "wireguard-t4-hub", "ssh-loopback-tunnel", "boinc-night-workunit"),
        help="Firewall traversal mode (default: outbound-https-relay over TCP/443 with 0 inbound ports)",
    )
    parser.add_argument("--check-firewall", action="store_true", help="Verify outbound HTTPS traversal to OpenTrainer")
    parser.add_argument("--host-label", default=socket.gethostname() or platform.node() or "trainer-node", help="Local trainer host label")
    parser.add_argument("--gpu-name", default="", help="Override detected GPU model name")
    parser.add_argument("--vram-total-mib", type=int, default=0, help="Override total GPU VRAM in MiB")
    parser.add_argument("--vram-free-mib", type=int, default=0, help="Override free GPU VRAM in MiB")
    parser.add_argument(
        "--max-vram-gb",
        type=float,
        default=0.0,
        help="Optional VRAM ceiling in GiB for training allocation (e.g. 5.5 on an 8 GB GPU)",
    )
    parser.add_argument("--training-stack", default="qlora-moe+diloco+amcp-aglet+boinc", help="Active training stack descriptor")
    parser.add_argument("--campaign-id", default="CROWDGPT-MOE-8X-UPCYCLE-V1", help="Active co-training campaign ID")
    parser.add_argument("--train", action="store_true", help="Execute real PyTorch Sparse-MoE OpenDiLoCo inner steps locally and submit the measured delta receipt")
    parser.add_argument("--work-dir", default=DEFAULT_TRAINER_WORK_DIR, help="Directory for local trainer checkpoints and round reports")
    parser.add_argument("--cache-dir", default="", help="Optional HuggingFace Hub cache directory")
    parser.add_argument("--s21-shard-dir", default="", help="Optional directory containing pre-tokenized S21 shards")
    parser.add_argument("--pinned-revision", default=DEFAULT_PINNED_CROWD_REVISION, help="Pinned HuggingFace revision for Crowd-v1 experimental MoE runs")
    parser.add_argument("--seq-len", type=int, default=64, help="Training sequence length per inner step")
    parser.add_argument("--lr", type=float, default=3e-4, help="Inner optimizer learning rate")
    parser.add_argument("--small-arch", action="store_true", help="Use lightweight Crowd-v1 config for smoke tests without 1B checkpoint")
    parser.add_argument("--cpu-only", action="store_true", help="Run local training steps on CPU even when CUDA is available")
    parser.add_argument("--offline", action="store_true", help="Disable HuggingFace network fetches and require local cache")
    parser.add_argument("--submit-delta", action="store_true", help="Submit a measured round delta after heartbeat (requires --round-report)")
    parser.add_argument(
        "--round-report",
        default="",
        help="JSON file with measured loss_before, loss_after, delta_norm and delta_path (required with --submit-delta)",
    )
    parser.add_argument("--round-id", type=int, default=1, help="Round ID for delta submission")
    parser.add_argument("--learner-index", type=int, default=1, help="Learner index in the K-node cell (0=T4 hub, 1=RTX 2070, ...)")
    parser.add_argument("--inner-steps", type=int, default=8, help="Inner optimizer steps completed")
    parser.add_argument("--once", action="store_true", help="Run a single heartbeat/round and exit")
    parser.add_argument("--interval-s", type=float, default=60.0, help="Heartbeat interval in seconds when running continuously")
    crowd = parser.add_argument_group(
        "CrowdGPT contributor mode",
        "Lend this GPU to CrowdGPT's shared model Crowd-v1 through CrowdGPT's own signed client.py, "
        "paused or stopped automatically when you need the GPU (automatic guards are NVIDIA-only).",
    )
    crowd.add_argument("--crowdgpt", action="store_true", help="Run the CrowdGPT client under the HiveMind GPU guard")
    crowd.add_argument("--crowdgpt-dir", default=os.environ.get("CROWDGPT_DIR", ""), help="Path to a CrowdGPT checkout (contains client.py)")
    crowd.add_argument("--reserve-mib", type=int, default=1500, help="GPU memory always left free for other programs (Tesla T4: 4300)")
    crowd.add_argument("--vram-gb", type=float, default=0.0, help="Client VRAM budget in GB (default: auto from free memory; required on non-NVIDIA GPUs)")
    crowd.add_argument("--backend", default="auto", choices=("auto", "cuda", "rocm", "xpu", "directml", "cpu", "mps"), help="CrowdGPT client backend")
    crowd.add_argument("--pause-for", default="", help="Comma-separated process names (e.g. games) that pause training while running")
    crowd.add_argument("--ignore-app", default="", help="Comma-separated name fragments of always-on GPU apps to tolerate (e.g. llama-server,ollama); the free-memory reserve still stops the client")
    crowd.add_argument("--hot-c", type=float, default=82.0, help="Pause at or above this GPU temperature (C)")
    crowd.add_argument("--cool-c", type=float, default=74.0, help="Resume only at or below this GPU temperature (C)")
    crowd.add_argument("--stop-at", default="", help="Stop contributing at this local time, HH:MM")
    crowd.add_argument("--data-dir", default=os.environ.get("HIVEMIND_CROWDGPT_DATA", os.path.join(os.path.expanduser("~"), ".hivemind", "crowdgpt")), help="Where receipts, the client log and HF_HOME live")
    crowd.add_argument("--dry-run", action="store_true", help="With --crowdgpt: preflight and plan only, start nothing")
    crowd.add_argument("--crowdgpt-python", default=sys.executable, help="Python interpreter that has CrowdGPT's requirements installed")
    crowd.add_argument("--crowdgpt-heartbeat", action="store_true", help="With --crowdgpt: also send one existing OpenTrainer heartbeat (needs HIVEMIND_API_KEY)")
    return parser


def _import_crowdgpt_contributor() -> Any:
    try:
        from hivemind.opentrainer import crowdgpt_contributor
    except ImportError:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
        from hivemind.opentrainer import crowdgpt_contributor
    return crowdgpt_contributor


def run_crowdgpt_mode(args: argparse.Namespace) -> int:
    """``--crowdgpt``: supervise CrowdGPT's client; no HiveMind delta submission, no HMT."""
    cc = _import_crowdgpt_contributor()
    if args.crowdgpt_heartbeat and not args.dry_run:
        api_key = os.environ.get("HIVEMIND_API_KEY", "").strip()
        gpu_name, total_mib, free_mib = detect_local_gpu()
        if not api_key or total_mib <= 0:
            print("Heartbeat skipped: needs HIVEMIND_API_KEY and an nvidia-smi GPU.", file=sys.stderr)
        else:
            try:
                send_heartbeat(
                    gateway=args.gateway, api_key=api_key, host_label=args.host_label, gpu_name=gpu_name,
                    vram_total_mib=total_mib, vram_free_mib=free_mib, network_mode=args.network_mode,
                    training_stack="crowdgpt-contributor", active_campaign="CROWDGPT-CROWD-V1",
                )
                print("OpenTrainer heartbeat sent.")
            except (urllib.error.URLError, ValueError) as exc:
                print(f"Heartbeat failed (continuing): {exc}", file=sys.stderr)
    return cc.run_contributor(
        cc.ContributorArgs(
            crowdgpt_dir=args.crowdgpt_dir,
            data_dir=args.data_dir,
            reserve_mib=args.reserve_mib,
            vram_gb=args.vram_gb,
            backend=args.backend,
            pause_for=tuple(p.strip() for p in args.pause_for.split(",") if p.strip()),
            ignore_apps=tuple(p.strip() for p in args.ignore_app.split(",") if p.strip()),
            hot_c=args.hot_c,
            cool_c=args.cool_c,
            stop_at=args.stop_at,
            once=args.once,
            dry_run=args.dry_run,
            python=args.crowdgpt_python,
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.crowdgpt:
        return run_crowdgpt_mode(args)
    if not math.isfinite(float(args.interval_s)) or float(args.interval_s) <= 0.0:
        print("Error: --interval-s must be a finite positive number.", file=sys.stderr)
        return 2
    if not math.isfinite(float(args.max_vram_gb)) or float(args.max_vram_gb) < 0.0:
        print("Error: --max-vram-gb must be a finite non-negative number.", file=sys.stderr)
        return 2
    if not math.isfinite(float(args.lr)) or float(args.lr) <= 0.0:
        print("Error: --lr must be a finite positive number.", file=sys.stderr)
        return 2
    if int(args.seq_len) < 4 or int(args.inner_steps) < 1:
        print("Error: --seq-len must be >= 4 and --inner-steps must be >= 1.", file=sys.stderr)
        return 2
    round_report: dict[str, Any] | None = None
    if args.submit_delta and not args.train:
        if not args.round_report:
            print("Error: --submit-delta requires --round-report (or use --train); no delta was submitted.", file=sys.stderr)
            return 2
        try:
            round_report = load_round_report(args.round_report)
        except ValueError as exc:
            print(f"Error: {exc}; no delta was submitted.", file=sys.stderr)
            return 2
    detected_gpu, detected_total, detected_free = detect_local_gpu()
    gpu_name = args.gpu_name or detected_gpu or ("custom-gpu" if args.vram_total_mib > 0 else "")
    vram_total = args.vram_total_mib if args.vram_total_mib > 0 else detected_total
    vram_free = args.vram_free_mib if args.vram_free_mib > 0 else min(detected_free, vram_total)
    if float(args.max_vram_gb) > 0.0 and vram_free > 0:
        cap_mib = max(512, int(round(float(args.max_vram_gb) * 1024.0)))
        vram_free = min(vram_free, cap_mib)

    if args.check_firewall:
        try:
            fw_report = verify_firewall_traversal(args.gateway, args.network_mode)
            print(json.dumps({"firewall_check": fw_report}, indent=2))
        except (urllib.error.URLError, ValueError) as exc:
            print(json.dumps({"ok": False, "error": f"Firewall check failed: {exc}"}), file=sys.stderr)
            return 1

    api_key = os.environ.get("HIVEMIND_API_KEY", "").strip()
    if not api_key:
        if args.check_firewall and args.once:
            return 0
        print("Error: HIVEMIND_API_KEY environment variable is required for OpenTrainer heartbeats.", file=sys.stderr)
        return 2

    if args.cpu_only and (vram_total <= 0 or not gpu_name):
        gpu_name = gpu_name or "CPU-Offload"
        vram_total = max(512, vram_total or 4096)
        vram_free = max(512, vram_free or 4096)
    elif vram_total <= 0 or not gpu_name:
        print(
            "Error: No readable NVIDIA GPU detected via nvidia-smi (and --vram-total-mib was not specified); "
            "refusing to send fabricated GPU telemetry.",
            file=sys.stderr,
        )
        return 2

    local_round_offset = 0
    while True:
        try:
            ack = send_heartbeat(
                gateway=args.gateway,
                api_key=api_key,
                host_label=args.host_label,
                gpu_name=gpu_name,
                vram_total_mib=vram_total,
                vram_free_mib=vram_free,
                network_mode=args.network_mode,
                training_stack=args.training_stack,
                active_campaign=args.campaign_id,
            )
            offer = ack.get("active_task_offer") or {}
            print(
                json.dumps(
                    {
                        "event": "opentrainer_heartbeat_ok",
                        "node": ack.get("node"),
                        "aglet_phase": ack.get("aglet_lifecycle_phase"),
                        "task_offer_round": offer.get("round_id"),
                    },
                    indent=2,
                )
            )
            active_round_id = int(offer.get("round") or offer.get("round_id") or args.round_id) + local_round_offset
            active_report = round_report
            if args.train:
                active_report = execute_local_trainer_round(
                    gateway=args.gateway,
                    host_label=args.host_label,
                    round_id=active_round_id,
                    work_dir=args.work_dir,
                    cache_dir=args.cache_dir,
                    s21_shard_dir=args.s21_shard_dir,
                    pinned_revision=args.pinned_revision,
                    inner_steps=args.inner_steps,
                    seq_len=args.seq_len,
                    lr=float(args.lr),
                    vram_free_mib=vram_free,
                    vram_total_mib=vram_total,
                    max_vram_gb=float(args.max_vram_gb),
                    small_arch=bool(args.small_arch),
                    cpu_only=bool(args.cpu_only),
                    offline=bool(args.offline),
                )
                print(json.dumps({"event": "opentrainer_round_measured", "report": active_report}, indent=2))
            if args.submit_delta or args.train:
                assert active_report is not None
                receipt = submit_verified_delta(
                    gateway=args.gateway,
                    api_key=api_key,
                    round_id=active_round_id,
                    campaign_id=args.campaign_id,
                    learner_index=args.learner_index,
                    inner_steps=args.inner_steps,
                    loss_before=active_report["loss_before"],
                    loss_after=active_report["loss_after"],
                    delta_norm=active_report["delta_norm"],
                    delta_sha256=active_report["delta_sha256"],
                    code_pin=str(offer.get("script_sha256") or offer.get("code_pin") or ""),
                    network_mode=args.network_mode,
                )
                print(json.dumps({"event": "opentrainer_delta_accepted", "receipt": receipt}, indent=2))
                local_round_offset += 1
        except RuntimeError as exc:
            print(f"OpenTrainer training setup error: {exc}", file=sys.stderr)
            if args.once:
                return 2
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"OpenTrainer HTTP {exc.code}: {detail}", file=sys.stderr)
            if args.once:
                return 1
        except urllib.error.URLError as exc:
            print(f"OpenTrainer connection error: {exc}", file=sys.stderr)
            if args.once:
                return 1

        if args.once:
            return 0
        time.sleep(max(5.0, float(args.interval_s)))


if __name__ == "__main__":
    raise SystemExit(main())
