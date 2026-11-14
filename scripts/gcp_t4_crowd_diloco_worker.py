#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""GCP Spot Tesla T4 & Multi-GPU Swarm DiLoCo Worker for `GCPCrowd` (`Vxtzq/Crowd-v1` x HiveMind).

Modes:
  1. `budget-check`: Verifies Spot `n1-standard-8` + Tesla T4 schedule against the `$150/mo` `AGENTS.md` cap.
  2. `render-systemd`: Prints or writes the serving-first `hivemind-gcpcrowd-diloco.service` template
     (writes strictly to `/mnt/disks/disk-socrateai-local-1/gcpcrowd`, never `/opt/hivemind/t4/`).
  3. `co-train-round`: Executes a verified multi-learner DiLoCo outer Nesterov round (`Learner 0 = GCP T4`,
     `Learner 1 = RTX 2070` + optional community cohorts) on real `Vxtzq/Crowd-v1` weights (or synthetic
     smoke architecture), produces a signed `hivemind-t4` `LossCheckReceipt`, and optionally submits the
     round receipt to HiveMind OpenTrainer (`POST /v1/hivemind/opentrainer/submit-delta`).
  4. `once` / `daemon`: Runs the continuous serving-first T4 DiLoCo inner learner governed by
     `T4GovernorYieldGuard` (`HOLD`, `governor_state.json`, `nginx` `stub_status`, GCE Spot 30s preemption,
     and `>= 4,300 MiB` free VRAM floor).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
import urllib.request

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hivemind.router.opentrainer import DEFAULT_MIN_FREE_VRAM_MIB_T4  # noqa: E402
from hivemind.swarm.crowdgpt_bridge import DEFAULT_HIVEMIND_GATEWAY_URL  # noqa: E402
from hivemind.swarm.crowdgpt_live_runtime import (  # noqa: E402
    CrowdGPTHuggingFaceSync,
    CrowdV1RuntimeConfig,
    CrowdV1WithEngram,
    load_crowd_v1_safetensors_into_model,
)
from hivemind.swarm.gcp_t4_crowd_diloco import (  # noqa: E402
    DEFAULT_GCP_T4_WORK_DIR_POSIX,
    DEFAULT_T4_GOVERNOR_STATE_POSIX,
    DEFAULT_T4_HOLD_FILE_POSIX,
    DEFAULT_T4_NGINX_STATUS_URL,
    GCPCROWD_CAMPAIGN_ID,
    GcpCrowdDiLoCoOuterHub,
    T4GovernorYieldGuard,
    assert_not_protected_t4_path,
    compute_gcp_t4_spot_crowd_budget,
    render_gcp_t4_diloco_systemd_unit,
    run_gcp_t4_diloco_inner_loop,
)


def _submit_opentrainer_receipt(
    *,
    gateway_url: str,
    api_key: str,
    campaign_id: str,
    round_index: int,
    delta_l2: float,
    loss_before: float,
    eval_loss: float,
    delta_sha256: str,
    inner_steps: int,
    vram_free_mib: int,
    vram_total_mib: int,
) -> dict[str, Any]:
    import urllib.error

    base_url = gateway_url.rstrip("/")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "hivemind-gcpcrowd-t4-diloco/1.0",
    }
    hb_payload = {
        "host_label": "crowd-moe-opendiloco-worker",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-Offload",
        "vram_total_mib": int(vram_total_mib),
        "vram_free_mib": min(int(vram_free_mib), int(vram_total_mib)),
        "network_mode": "outbound-https-relay",
        "training_stack": "opendiloco+decoupled_diloco_rda",
        "active_campaign": campaign_id,
    }
    try:
        hb_req = urllib.request.Request(
            f"{base_url}/v1/hivemind/opentrainer/heartbeat",
            data=json.dumps(hb_payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(hb_req, timeout=15.0):
            pass
    except Exception:
        pass

    payload = {
        "round_id": int(round_index),
        "campaign_id": campaign_id,
        "inner_steps": max(1, int(inner_steps)),
        "loss_before": float(loss_before),
        "loss_after": float(eval_loss),
        "delta_norm": min(24.5, max(1e-6, float(delta_l2))),
        "delta_sha256": str(delta_sha256),
        "network_mode": "outbound-https-relay",
        "vram_free_mib": int(vram_free_mib),
        "vram_total_mib": int(vram_total_mib),
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/hivemind/opentrainer/submit-delta",
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        return {"accepted": False, "http_status": exc.code, "detail": err_body}


def _stream_safetensors_no_mmap(model: CrowdV1WithEngram, safetensors_path: Path) -> str:
    """Stream `model.safetensors` tensor-by-tensor via `seek`/`read` (zero 2 GB `MapViewOfFile` commit spike)."""
    import struct

    dt_map = {
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
    }
    h_hash = hashlib.sha256()
    param_map = dict(model.named_parameters())
    with safetensors_path.open("rb") as fh, torch.no_grad():
        raw_n = fh.read(8)
        h_hash.update(raw_n)
        (header_len,) = struct.unpack("<Q", raw_n)
        header_bytes = fh.read(header_len)
        h_hash.update(header_bytes)
        header = json.loads(header_bytes.decode("utf-8"))
        data_base = 8 + header_len

        for key, meta in header.items():
            if key == "__metadata__":
                continue
            target = param_map.get(key)
            if target is None:
                continue
            lo, hi = int(meta["data_offsets"][0]), int(meta["data_offsets"][1])
            nbytes = hi - lo
            src_dt = dt_map[str(meta["dtype"])]
            shape = tuple(int(d) for d in meta["shape"])
            fh.seek(data_base + lo)
            if len(shape) == 2 and nbytes > 64 * 1024 * 1024:
                elem_b = 2 if src_dt in (torch.bfloat16, torch.float16) else 4
                row_b = shape[1] * elem_b
                chunk_rows = max(1, (32 * 1024 * 1024) // row_b)
                for r0 in range(0, shape[0], chunk_rows):
                    r1 = min(shape[0], r0 + chunk_rows)
                    cbuf = bytearray(fh.read((r1 - r0) * row_b))
                    if r0 == 0:
                        h_hash.update(cbuf[: min(len(cbuf), 4096)])
                    part = torch.frombuffer(cbuf, dtype=src_dt).reshape((r1 - r0, shape[1]))
                    target[r0:r1].copy_(part.to(device=target.device, dtype=target.dtype))
                    del part, cbuf
            else:
                buf = bytearray(fh.read(nbytes))
                h_hash.update(buf[: min(len(buf), 4096)])
                src_t = torch.frombuffer(buf, dtype=src_dt).reshape(shape)
                target.copy_(src_t.to(device=target.device, dtype=target.dtype))
                del src_t, buf
    return h_hash.hexdigest()


def _load_or_build_model(
    *,
    cache_dir: Path | None,
    use_small_arch: bool,
    direct_trunk_device: str = "cpu",
    direct_trunk_dtype: torch.dtype = torch.bfloat16,
) -> tuple[CrowdV1WithEngram, dict[str, Any]]:
    if use_small_arch:
        torch.manual_seed(20261110)
        cfg = CrowdV1RuntimeConfig(
            vocab_size=256,
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            intermediate_size=256,
            max_position_embeddings=256,
            engram_buckets=512,
        )
        model = CrowdV1WithEngram(cfg)
        with torch.no_grad():
            for p in model.parameters():
                if p.ndim >= 2:
                    p.mul_(0.02)
        return model, {"source": "synthetic_small_arch", "hf_revision": "local-smoke", "global_step": 0}

    sync = CrowdGPTHuggingFaceSync(cache_dir=cache_dir)
    cached_candidates = sorted(sync.cache_dir.glob("model_*.safetensors")) if sync.cache_dir.exists() else []
    if cached_candidates:
        weights_path = cached_candidates[-1]
        rev = weights_path.stem.replace("model_", "")
        step = 348
    else:
        files = sync.ensure_latest_files(force_check=True)
        weights_path = Path(files["weights_path"])
        rev = str(files["hf_revision"])
        step = int(files["global_step"])

    cfg = CrowdV1RuntimeConfig()
    prev_dt = torch.get_default_dtype()
    try:
        torch.set_default_dtype(direct_trunk_dtype)
        with torch.device("meta"):
            model = CrowdV1WithEngram(cfg)
    finally:
        torch.set_default_dtype(prev_dt)

    # Materialize Engram in host RAM (bfloat16) and trunk directly on target device without kaiming_uniform_ RAM spikes
    model.engram.to_empty(device="cpu")
    model.engram.to(dtype=torch.bfloat16)
    model.wte.to_empty(device=direct_trunk_device)
    model.blocks.to_empty(device=direct_trunk_device)
    model.ln.to_empty(device=direct_trunk_device)
    if not cfg.tie_word_embeddings:
        model.head.to_empty(device=direct_trunk_device)
    else:
        model.head.weight = model.wte.weight

    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim))
    freqs = torch.outer(torch.arange(cfg.max_position_embeddings, dtype=torch.float32), inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    model.register_buffer("cos_cached", emb.cos()[None, None].to(device=direct_trunk_device), persistent=False)
    model.register_buffer("sin_cached", emb.sin()[None, None].to(device=direct_trunk_device), persistent=False)

    sha = _stream_safetensors_no_mmap(model, weights_path)
    return model, {
        "source": str(weights_path),
        "hf_revision": rev,
        "global_step": step,
        "weights_sha256": sha,
    }


def _build_token_shards(vocab_size: int, shard_tokens: int = 512) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build repeatable structured non-IID token shards + held-out validation stream."""
    base_pattern = torch.tensor(
        [(i * 7 + 13) % max(16, min(vocab_size, 256)) for i in range(16)],
        dtype=torch.long,
    )
    repeats = max(8, (shard_tokens // 16) + 4)
    stream = base_pattern.repeat(repeats)
    shard_t4 = stream[:shard_tokens].clone()
    shard_2070 = torch.roll(stream, shifts=4)[:shard_tokens].clone()
    held_out = stream[: max(128, shard_tokens // 2)].clone()
    return shard_t4, shard_2070, held_out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GCP Spot Tesla T4 DiLoCo Co-Trainer for GCPCrowd")
    ap.add_argument(
        "--mode",
        choices=("budget-check", "render-systemd", "co-train-round", "moe-opendiloco-round", "once", "daemon"),
        default="co-train-round",
    )
    ap.add_argument(
        "--moe-split-mode",
        choices=("lora_domain_experts", "fine_grained_width_split"),
        default="lora_domain_experts",
    )
    ap.add_argument(
        "--outer-optimizer",
        choices=("opendiloco_nesterov", "pa_cgad"),
        default="opendiloco_nesterov",
    )
    ap.add_argument("--resident-experts", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--work-dir", type=Path, default=Path(DEFAULT_GCP_T4_WORK_DIR_POSIX))
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--governor-state", type=Path, default=None)
    ap.add_argument("--hold-file", type=Path, default=None)
    ap.add_argument("--nginx-status-url", default=None)
    ap.add_argument("--require-governor-training", action="store_true")
    ap.add_argument("--check-spot-preemption", action="store_true")
    ap.add_argument("--min-free-vram-mib", type=int, default=DEFAULT_MIN_FREE_VRAM_MIB_T4)
    ap.add_argument(
        "--recipe",
        choices=("auto", "full_trunk_8bit_ckpt", "block_window_diloco", "qlora_graft_diloco", "cpu_fp32"),
        default="auto",
    )
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--inner-steps", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--outer-lr", type=float, default=0.7)
    ap.add_argument("--outer-momentum", type=float, default=0.9)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--window-blocks", type=int, default=6)
    ap.add_argument("--small-arch", action="store_true", help="Use lightweight test config instead of 1B HF checkpoint")
    ap.add_argument("--prefer-cuda", action="store_true", default=True)
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--gateway", default=DEFAULT_HIVEMIND_GATEWAY_URL)
    ap.add_argument("--submit-opentrainer", action="store_true")
    ap.add_argument("--real-data", action="store_true",
                    help="train on real rows of the 8-expert dataset plan instead of the synthetic pattern (needs --tokenizer-json)")
    ap.add_argument("--tokenizer-json", type=Path, default=None, help="Crowd-v1 tokenizer.json (tokenizers library format)")
    ap.add_argument("--data-cache-dir", type=Path, default=None, help="where fetched dataset rows are cached (default: <work-dir>/rows)")
    ap.add_argument("--train-tokens-per-expert", type=int, default=16_384)
    ap.add_argument("--heldout-tokens-per-expert", type=int, default=2_048)
    ap.add_argument("--eval-windows", type=int, default=2, help="held-out windows per evaluation (synthetic default 2)")
    ap.add_argument("--campaign-id", default=GCPCROWD_CAMPAIGN_ID)
    ap.add_argument("--no-save-safetensors", action="store_true")
    ap.add_argument("--systemd-out", type=Path, default=None)
    ap.add_argument("--poll-s", type=float, default=15.0)
    args = ap.parse_args(argv)

    if args.mode == "budget-check":
        plan = compute_gcp_t4_spot_crowd_budget()
        print(json.dumps(asdict(plan), indent=2, sort_keys=True))
        return 0 if plan.within_budget else 1

    safe_work = assert_not_protected_t4_path(args.work_dir)
    if args.mode == "render-systemd":
        unit_txt = render_gcp_t4_diloco_systemd_unit(
            work_dir=str(safe_work),
            governor_state=str(args.governor_state or DEFAULT_T4_GOVERNOR_STATE_POSIX),
            hold_file=str(args.hold_file or DEFAULT_T4_HOLD_FILE_POSIX),
            nginx_status_url=str(args.nginx_status_url or DEFAULT_T4_NGINX_STATUS_URL),
        )
        if args.systemd_out is not None:
            out_p = assert_not_protected_t4_path(args.systemd_out)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(unit_txt, encoding="utf-8")
        else:
            print(unit_txt, end="")
        return 0

    prefer_cuda = bool(args.prefer_cuda and not args.cpu_only)
    eff_min_free = int(args.min_free_vram_mib)
    if prefer_cuda and torch.cuda.is_available():
        _, total_b = torch.cuda.mem_get_info(0)
        total_m = int(total_b // (1024 * 1024))
        if total_m < 14_000 and eff_min_free == DEFAULT_MIN_FREE_VRAM_MIB_T4:
            eff_min_free = 1536

    if prefer_cuda and torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0).upper().split() and eff_min_free < 4300:
        # AGENTS.md: a Tesla T4 always keeps >= 4,300 MiB free; this is a rule for that card, not a tuning knob (as in PR #876)
        print(f"refused: --min-free-vram-mib {eff_min_free} is below the 4300 MiB T4 floor", file=sys.stderr)
        return 2

    guard = T4GovernorYieldGuard(
        hold_file=args.hold_file,
        governor_state_file=args.governor_state,
        require_governor_training_state=args.require_governor_training,
        nginx_status_url=args.nginx_status_url,
        check_gce_spot_metadata=args.check_spot_preemption,
        min_free_vram_mib=eff_min_free,
    )

    init_dev = "cpu"
    init_dt = torch.bfloat16
    if args.mode == "moe-opendiloco-round" and prefer_cuda and torch.cuda.is_available():
        fb0, _ = torch.cuda.mem_get_info(0)
        est0 = 1850 if args.moe_split_mode == "lora_domain_experts" else 2450
        if int(fb0 // (1024 * 1024)) - est0 >= eff_min_free:
            init_dev = "cuda"
            init_dt = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16

    model, rev_meta = _load_or_build_model(
        cache_dir=args.cache_dir,
        use_small_arch=args.small_arch,
        direct_trunk_device=init_dev,
        direct_trunk_dtype=init_dt,
    )
    data_manifest: dict[str, Any] | None = None
    if args.real_data:
        if args.tokenizer_json is None or not Path(args.tokenizer_json).is_file():
            print("refused: --real-data needs --tokenizer-json pointing to the Crowd-v1 tokenizer.json", file=sys.stderr)
            return 2
        from tokenizers import Tokenizer

        from hivemind.swarm.crowdgpt_moe_opendiloco import load_crowdgpt_moe_dataset_plan
        from hivemind.swarm.crowdgpt_real_shards import build_real_expert_shards

        tok = Tokenizer.from_file(str(args.tokenizer_json))
        shard_t4, shard_2070, held_out, data_manifest = build_real_expert_shards(
            load_crowdgpt_moe_dataset_plan(),
            lambda s: tok.encode(s).ids,
            model.cfg.vocab_size,
            cache_dir=Path(args.data_cache_dir) if args.data_cache_dir else Path(args.work_dir) / "rows",
            train_tokens_per_expert=args.train_tokens_per_expert,
            heldout_tokens_per_expert=args.heldout_tokens_per_expert,
        )
        Path(args.work_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.work_dir) / "real_data_manifest.json").write_text(json.dumps(data_manifest, indent=2) + "\n", encoding="utf-8")
    else:
        shard_t4, shard_2070, held_out = _build_token_shards(model.cfg.vocab_size, shard_tokens=max(256, args.seq_len * 6))

    if args.mode == "moe-opendiloco-round":
        from safetensors.torch import save_file

        from hivemind.swarm.crowdgpt_moe_opendiloco import (
            CrowdOpenDiLoCoCoordinator,
            extract_moe_trainable_buckets,
            load_moe_trainable_buckets,
            run_crowd_moe_opendiloco_inner_steps,
            upcycle_crowd_v1_to_sparse_moe,
        )
        from hivemind.swarm.decoupled_diloco import LearnerArrival
        from hivemind.swarm.gcp_t4_crowd_diloco import forward_crowd_v1_checkpointed

        res_experts = tuple(sorted({int(x.strip()) for x in str(args.resident_experts).split(",") if x.strip()}))
        exec_dev = init_dev if (init_dev == "cuda" and not args.small_arch) else "cpu"
        base_dt = init_dt if exec_dev == "cuda" else torch.float32
        if exec_dev != "cuda" and prefer_cuda and torch.cuda.is_available():
            free_b, _ = torch.cuda.mem_get_info(0)
            free_m = int(free_b // (1024 * 1024))
            est_mib = 1850 if args.moe_split_mode == "lora_domain_experts" else 2450
            if free_m - est_mib >= eff_min_free:
                exec_dev = "cuda"
                base_dt = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
                model.place_for_inference(device="cuda", dtype=base_dt, offload_engram_to_cpu=True)

        split_report = upcycle_crowd_v1_to_sparse_moe(
            model,
            split_mode=args.moe_split_mode,
            num_experts=8,
            top_k=2,
            lora_rank=args.lora_rank,
            routing_mode="sigma_topk",
            resident_expert_indices=res_experts,
        )
        init_buckets = extract_moe_trainable_buckets(model, num_router_fragments=4)
        coord = CrowdOpenDiLoCoCoordinator(
            init_buckets,
            min_quorum_k=2,
            merge_mode="rda",
            outer_optimizer_mode=args.outer_optimizer,
            outer_lr=args.outer_lr,
            outer_momentum=args.outer_momentum,
            work_dir=safe_work,
        )

        win_len = min(model.cfg.max_position_embeddings, max(4, min(args.seq_len, 64)))
        n_win = min(max(1, int(args.eval_windows)), max(1, (int(held_out.numel()) - 1) // win_len))
        eval_windows = torch.stack(
            [held_out[i * win_len : (i + 1) * win_len] for i in range(n_win)]
        ).to(device=exec_dev, dtype=torch.long)

        def _eval_buckets(cand_buckets: dict[str, torch.Tensor]) -> float:
            load_moe_trainable_buckets(model, cand_buckets, num_router_fragments=4)
            model.eval()
            losses: list[float] = []
            with torch.no_grad():
                for w_idx in range(n_win):
                    w_tok = eval_windows[w_idx : w_idx + 1]
                    _, l_t = forward_crowd_v1_checkpointed(
                        model,
                        w_tok,
                        labels=w_tok,
                        use_grad_checkpointing=False,
                        token_chunk_size=min(256, win_len),
                    )
                    losses.append(float(l_t.item()) if l_t is not None else 0.0)
            return sum(losses) / max(1, len(losses))

        moe_rounds: list[dict[str, Any]] = []
        for rnd in range(max(1, args.rounds)):
            master_snap = coord.export_buckets()

            # Learner 0: GCP Spot Tesla T4 / Primary Cohort
            load_moe_trainable_buckets(model, master_snap, num_router_fragments=4)
            t0 = time.monotonic()
            out_t4 = run_crowd_moe_opendiloco_inner_steps(
                model,
                shard_t4,
                steps=args.inner_steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                lr=args.lr,
                device=exec_dev,
                yield_guard=guard,
            )
            dt_t4 = max(0.02, time.monotonic() - t0)

            # Learner 1: XAVGAMSRV RTX 2070 / Secondary Non-IID Cohort
            load_moe_trainable_buckets(model, master_snap, num_router_fragments=4)
            t1 = time.monotonic()
            out_2070 = run_crowd_moe_opendiloco_inner_steps(
                model,
                shard_2070,
                steps=args.inner_steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                lr=args.lr,
                device=exec_dev,
                yield_guard=guard,
            )
            dt_2070 = max(0.03, time.monotonic() - t1)
            yielded = out_t4.get("yielded") or out_2070.get("yielded")
            if yielded:
                # HOLD file, live traffic, Spot preemption or the VRAM floor: stop training, do not aggregate a partial round
                moe_rounds.append({"round": rnd + 1, "yielded": yielded})
                print(f"yielded before round {rnd + 1} completed: {yielded}", file=sys.stderr)
                break

            subs: dict[str, list[tuple[LearnerArrival, torch.Tensor, float]]] = {}
            for b_key in master_snap:
                arr0 = LearnerArrival("gcp-t4-hermes-node1", dt_t4, out_t4["steps_completed"], out_t4["tokens_processed"])
                arr1 = LearnerArrival("xavgamsrv-rtx2070", dt_t4 + dt_2070 * 0.25, out_2070["steps_completed"], out_2070["tokens_processed"])
                subs[b_key] = [
                    (arr0, out_t4["pseudo_grads"][b_key], 0.0),
                    (arr1, out_2070["pseudo_grads"][b_key], 0.0),
                ]

            receipt = coord.step_decoupled_round(subs, eval_fn=_eval_buckets)
            load_moe_trainable_buckets(model, coord.export_buckets(), num_router_fragments=4)

            ckpt_sha = receipt.master_state_sha256
            if not args.no_save_safetensors:
                safe_work.mkdir(parents=True, exist_ok=True)
                st_path = safe_work / f"crowd_moe_opendiloco_round_{rnd + 1}.safetensors"
                save_file({k: v.to(torch.bfloat16).contiguous() for k, v in coord.export_buckets().items()}, str(st_path))
                ckpt_sha = hashlib.sha256(st_path.read_bytes()).hexdigest()

            total_l2 = math.sqrt(
                sum(float(info.get("merged_delta_norm", 0.0)) ** 2 for info in receipt.quorum_telemetry.values())
            )
            ot_resp = None
            api_key = os.environ.get("HIVEMIND_API_KEY", "").strip()
            if args.submit_opentrainer and api_key:
                free_m = 4800
                total_m = 15360
                if torch.cuda.is_available():
                    fb, tb = torch.cuda.mem_get_info(0)
                    free_m = max(eff_min_free, int(fb // (1024 * 1024)))
                    total_m = int(tb // (1024 * 1024))
                ot_resp = _submit_opentrainer_receipt(
                    gateway_url=args.gateway,
                    api_key=api_key,
                    campaign_id=args.campaign_id,
                    round_index=rnd + 1,
                    delta_l2=max(1e-4, total_l2),
                    loss_before=receipt.eval_loss_before,
                    eval_loss=receipt.eval_loss_after,
                    delta_sha256=ckpt_sha,
                    inner_steps=out_t4["steps_completed"] + out_2070["steps_completed"],
                    vram_free_mib=free_m,
                    vram_total_mib=total_m,
                )

            moe_rounds.append(
                {
                    "round_index": rnd + 1,
                    "device": exec_dev,
                    "accepted_by_loss_ratchet": receipt.accepted_by_loss_ratchet,
                    "eval_loss_before": receipt.eval_loss_before,
                    "eval_loss_after": receipt.eval_loss_after,
                    "outer_delta_l2": round(total_l2, 6),
                    "fragments_synced": list(receipt.fragments_synced),
                    "checkpoint_sha256": ckpt_sha,
                    "opentrainer_submission": ot_resp,
                }
            )

        print(
            json.dumps(
                {
                    "status": "yielded" if moe_rounds and moe_rounds[-1].get("yielded") else "ok",
                    "mode": "moe-opendiloco-round",
                    "campaign_id": args.campaign_id,
                    "revision_meta": rev_meta,
                    "moe_split_report": asdict(split_report),
                    "rounds": moe_rounds,
                    "data": ("real rows of the dataset plan (see real_data_manifest.json)" if data_manifest is not None
                             else "synthetic repeating token pattern: an infrastructure test, not training"),
                    "learners_note": "both learners run sequentially on this one device",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.mode == "co-train-round":
        secret = hashlib.sha256(b"hivemind-gcpcrowd-t4-receipt-key").digest()
        hub = GcpCrowdDiLoCoOuterHub(
            model.export_flat_base_bf16(),
            outer_lr=args.outer_lr,
            outer_momentum=args.outer_momentum,
            f_byzantine=1,
            work_dir=safe_work,
            evaluator_id="hivemind-t4",
            evaluator_secret=secret,
            clone_initial=False,
        )
        if prefer_cuda and torch.cuda.is_available():
            import gc
            model.place_for_inference(device="cuda", dtype=torch.float16, offload_engram_to_cpu=True)
            gc.collect()
        round_summaries: list[dict[str, Any]] = []
        for rnd in range(max(1, args.rounds)):
            if rnd > 0:
                model.load_flat_tensors(hub.theta_bf16)
            res_t4 = run_gcp_t4_diloco_inner_loop(
                model,
                shard_t4,
                learner_id="gcp-t4-hermes-node1",
                hiveminder_handle="gcp-t4-hermes",
                device_suffix="t4spot",
                round_index=rnd,
                inner_steps=args.inner_steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                learning_rate=args.lr,
                requested_mode=args.recipe,
                window_blocks=args.window_blocks,
                lora_rank=args.lora_rank,
                prefer_cuda=prefer_cuda,
                work_dir=safe_work,
                yield_guard=guard,
                pack_stream=False,
                base_before_bf16_ref=hub.theta_bf16,
            )

            model.load_flat_tensors(hub.theta_bf16)
            res_2070 = run_gcp_t4_diloco_inner_loop(
                model,
                shard_2070,
                learner_id="xavgamsrv-rtx2070",
                hiveminder_handle="xavgamsrv",
                device_suffix="rtx2070",
                round_index=rnd,
                inner_steps=args.inner_steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                learning_rate=args.lr,
                requested_mode="qlora_graft_diloco" if args.recipe == "auto" else args.recipe,
                lora_rank=args.lora_rank,
                prefer_cuda=prefer_cuda,
                work_dir=safe_work,
                yield_guard=guard,
                pack_stream=False,
                base_before_bf16_ref=hub.theta_bf16,
            )

            outer = hub.step(
                [res_t4, res_2070],
                eval_model=model,
                held_out_tokens=held_out,
                eval_window_len=min(args.seq_len, 64),
                eval_num_windows=2,
                prefer_cuda_eval=prefer_cuda,
                save_safetensors=not args.no_save_safetensors,
                clear_input_deltas=True,
            )

            ot_resp: dict[str, Any] | None = None
            api_key = os.environ.get("HIVEMIND_API_KEY", "").strip()
            if args.submit_opentrainer and api_key:
                free_m = res_t4.recipe_decision.vram_free_mib or 10_500
                total_m = res_t4.recipe_decision.vram_total_mib or 15_360
                if total_m >= 14_000:
                    free_m = max(free_m, DEFAULT_MIN_FREE_VRAM_MIB_T4)
                else:
                    free_m = max(free_m, 1536)
                ot_resp = _submit_opentrainer_receipt(
                    gateway_url=args.gateway,
                    api_key=api_key,
                    campaign_id=args.campaign_id,
                    round_index=outer.round_index,
                    delta_l2=outer.outer_delta_l2,
                    loss_before=outer.loss_before,
                    eval_loss=outer.loss_after,
                    delta_sha256=outer.checkpoint_sha256,
                    inner_steps=max(1, res_t4.steps_completed + res_2070.steps_completed),
                    vram_free_mib=free_m,
                    vram_total_mib=total_m,
                )

            round_summaries.append(
                {
                    "round_index": outer.round_index,
                    "accepted": outer.accepted,
                    "loss_before": outer.loss_before,
                    "loss_after": outer.loss_after,
                    "outer_delta_l2": outer.outer_delta_l2,
                    "effective_tokens": outer.effective_tokens,
                    "selected_learners": outer.selected_learners,
                    "t4_recipe": res_t4.recipe_decision.recipe,
                    "rtx2070_recipe": res_2070.recipe_decision.recipe,
                    "rms_reason": outer.rms_guard.reason,
                    "ratchet_reason": outer.ratchet_decision.reason,
                    "checkpoint_sha256": outer.checkpoint_sha256,
                    "opentrainer_submission": ot_resp,
                }
            )

        out_report = {
            "status": "ok",
            "campaign_id": args.campaign_id,
            "revision_meta": rev_meta,
            "budget_plan": asdict(compute_gcp_t4_spot_crowd_budget()),
            "rounds": round_summaries,
        }
        print(json.dumps(out_report, indent=2, sort_keys=True))
        return 0

    rnd = 0
    while True:
        res = run_gcp_t4_diloco_inner_loop(
            model,
            shard_t4,
            learner_id="gcp-t4-hermes-node1",
            hiveminder_handle="gcp-t4-hermes",
            device_suffix="t4spot",
            round_index=rnd,
            inner_steps=args.inner_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            learning_rate=args.lr,
            requested_mode=args.recipe,
            window_blocks=args.window_blocks,
            lora_rank=args.lora_rank,
            prefer_cuda=prefer_cuda,
            work_dir=safe_work,
            yield_guard=guard,
        )
        print(
            json.dumps(
                {
                    "round": rnd,
                    "recipe": res.recipe_decision.recipe,
                    "steps_completed": res.steps_completed,
                    "preempted": res.preempted,
                    "trigger": res.preemption_trigger,
                    "yield_latency_ms": res.yield_latency_ms,
                    "loss_before": res.loss_before,
                    "loss_after": res.loss_after,
                    "delta_l2": res.delta_l2,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not res.preempted:
            rnd += 1
        if args.mode == "once":
            return 0
        time.sleep(max(1.0, float(args.poll_s)))


if __name__ == "__main__":
    raise SystemExit(main())
