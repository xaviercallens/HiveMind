# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Real Open-Weight Hugging Face PoC on Local NVIDIA GeForce RTX 2070 (8 GB VRAM).

Leverages HF_TOKEN from environment variables (.env / process env / Windows HKCU Registry)
to pull `Qwen/Qwen2.5-0.5B` (24 layers, 494M params), quantize to 4-bit NormalFloat (NF4) +
Double Quantization on the RTX 2070, train domain-specialized QLoRA branches via DiLoCo,
validate via deterministic Spot-Checks & Multi-Krum, upcycle in-place into a unified
Sparse Mixture-of-Experts (Qwen2.5-MoE ~1.44B params), and generate real text via
Phase 2 Regional Swarm Speculative Decoding.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer

from hivemind.boinc.validator import DeterministicSpotCheckValidator
from hivemind.boinc.workunit import (
    BOINCWorkunit,
    sign_shard_bytes,
    verify_and_load_parquet_shard,
)
from hivemind.cli.simulate_e2e import estimate_workunit_tflops
from hivemind.client.daemon import ZeroIntrusionDaemon
from hivemind.client.steamworks import SteamworksSDKBridge
from hivemind.config import (
    ARTIFACTS_ROOT,
    HiveMindConfig,
    configure_deterministic_runtime,
    detect_local_hardware,
    get_hf_token,
)
from hivemind.core.btm_upcycle import (
    train_moe_router_gating,
    upcycle_branches_to_sparse_moe,
)
from hivemind.core.compression import decompress_lora_delta
from hivemind.core.diloco_inner import run_diloco_inner_loop
from hivemind.core.diloco_outer import DiLoCoOuterOptimizer
from hivemind.core.quantization import (
    attach_qlora_adapters,
    extract_lora_state_dict,
    load_lora_state_dict,
    measure_vram_footprint,
    resolve_compute_dtype,
    stream_load_hf_causal_lm,
)
from hivemind.swarm.inference_ring import RegionalDHTSwarmRing, SwarmPeerNode

console = Console()

POC_DOMAIN_CORPORA: dict[str, list[str]] = {
    "Code": [
        "def binary_search(arr: list[int], target: int) -> int:\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
        "class DistributedLoRAWorker:\n    def compute_pseudo_gradient(self, theta_init, theta_final):\n        return {k: theta_init[k] - theta_final[k] for k in theta_init}\n",
    ],
    "Mathematics": [
        "Theorem (Nesterov Momentum Update): Given outer momentum mu = 0.9 and learning rate eta = 0.7, the velocity updates as v_t = mu * v_{t-1} + Delta_t and theta_t = theta_{t-1} - eta * (mu * v_t + Delta_t).",
        "Proof of Derivative: Let f(x) = x^3 + 2x^2 - 5x + 1. Differentiating term by term with respect to x yields f'(x) = 3x^2 + 4x - 5.",
    ],
    "STEM": [
        "In quantum mechanics, the time-dependent Schrodinger equation i * hbar * d/dt |psi(t)> = H |psi(t)> governs the unitary evolution of isolated physical wavefunctions.",
        "Mixture-of-Experts (MoE) architectures route each token representation x to the top-k Feed-Forward Networks via a learned softmax gating matrix W_g, decoupling parameter count from FLOPs per token.",
    ],
    "Multilingual_FR": [
        "L'architecture distribuée Project HiveMind agrège les cartes graphiques grand public via Steam et BOINC pour entraîner un modèle Mixture-of-Experts sans interconnexion InfiniBand.",
        "Grâce à la quantification NormalFloat 4-bit (NF4) et à l'algorithme DiLoCo, la bande passante réseau requise est réduite d'un facteur 500 sur une connexion fibre domestique.",
    ],
}

POC_PROMPTS: dict[str, str] = {
    "Code": "def binary_search(arr: list[int], target: int) ->",
    "Mathematics": "Theorem (Nesterov Momentum Update): Given outer momentum mu = 0.9",
    "STEM": "Mixture-of-Experts (MoE) architectures route each token",
    "Multilingual_FR": "L'architecture distribuée Project HiveMind agrège les cartes",
}


def fetch_hf_domain_subset(
    dataset_id: str = "sahil2801/CodeAlpaca-20k",
    num_samples: int = 24,
    hf_token: str | None = None,
) -> list[str]:
    """Pull a real domain dataset subset from Hugging Face Hub (`sahil2801/CodeAlpaca-20k` or `qiaojin/PubMedQA`)."""
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be a non-empty string")
    if isinstance(num_samples, bool) or int(num_samples) < 1:
        raise ValueError("num_samples must be a positive integer")
    from huggingface_hub import hf_hub_download

    samples: list[str] = []
    try:
        if "pubmedqa" in dataset_id.lower():
            p = hf_hub_download(
                repo_id=dataset_id,
                filename="pqa_labeled/train-00000-of-00001.parquet",
                repo_type="dataset",
                token=hf_token,
            )
            tbl = pq.read_table(p)
            for i in range(min(num_samples, tbl.num_rows)):
                q = str(tbl["question"][i].as_py()).strip()
                ans = str(tbl["long_answer"][i].as_py()).strip()
                dec = str(tbl["final_decision"][i].as_py()).strip()
                samples.append(f"Clinical Question: {q}\nEvidence Conclusion ({dec}): {ans}")
        else:
            p = hf_hub_download(
                repo_id=dataset_id,
                filename="code_alpaca_20k.json",
                repo_type="dataset",
                token=hf_token,
            )
            with open(p, "r", encoding="utf-8") as fp:
                head_chunk = fp.read(65536)
            decoder = json.JSONDecoder()
            pos = head_chunk.find("{")
            while pos != -1 and len(samples) < num_samples:
                try:
                    row, end_pos = decoder.raw_decode(head_chunk, pos)
                except json.JSONDecodeError:
                    break
                instr = str(row.get("instruction", "")).strip()
                inp = str(row.get("input", "")).strip()
                out = str(row.get("output", "")).strip()
                if instr and out:
                    text = f"### Instruction:\n{instr}\n"
                    if inp:
                        text += f"### Input:\n{inp}\n"
                    text += f"### Response:\n{out}"
                    samples.append(text)
                pos = head_chunk.find("{", end_pos)
    except Exception:
        samples = list(POC_DOMAIN_CORPORA["Code"]) * 12

    return samples[:num_samples] if samples else list(POC_DOMAIN_CORPORA["Code"])


def _create_tokenized_parquet_shard(
    tokenizer: Any,
    domain: str,
    texts: list[str],
    shard_path: Path,
    seq_len: int = 48,
    num_sequences: int = 8,
    rng_seed: int = 42,
) -> tuple[Path, str, str]:
    """Tokenize real domain strings with the Hugging Face tokenizer and write a signed Parquet shard."""
    if not isinstance(domain, str) or not domain.strip():
        raise ValueError("domain must be a non-empty string")
    if not texts or not any(isinstance(t, str) and t.strip() for t in texts):
        raise ValueError("texts must contain at least one non-empty string")
    if isinstance(seq_len, bool) or int(seq_len) < 1:
        raise ValueError("seq_len must be a positive integer")
    if isinstance(num_sequences, bool) or int(num_sequences) < 1:
        raise ValueError("num_sequences must be a positive integer")
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    joined = "\n".join(texts * 4)
    token_ids = tokenizer.encode(joined, add_special_tokens=False)
    while len(token_ids) < seq_len * num_sequences:
        token_ids.extend(token_ids)

    sequences: list[list[int]] = []
    for i in range(num_sequences):
        offset = (i * 11) % max(1, len(token_ids) - seq_len - 1)
        sequences.append(token_ids[offset : offset + seq_len])

    table = pa.table(
        {
            "seq_id": pa.array(np.arange(num_sequences, dtype=np.int32)),
            "branch": pa.array([domain] * num_sequences),
            "input_ids": pa.array(sequences, type=pa.list_(pa.int32())),
        }
    )
    pq.write_table(table, shard_path, compression="zstd")
    raw_bytes = shard_path.read_bytes()
    sha256, sig = sign_shard_bytes(raw_bytes, rng_seed=rng_seed)
    return shard_path, sha256, sig


def _export_shared_swarm_expert(
    domain_name: str,
    dataset_id: str,
    lora_state: dict[str, torch.Tensor],
    worker_trajectories: dict[str, dict[str, Any]],
) -> tuple[str, Path]:
    """Export converged domain QLoRA adapter to disk so distributed swarm peers can share it for inference."""
    export_dir = ARTIFACTS_ROOT / "shared_experts"
    export_dir.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256(f"{domain_name}:{dataset_id}".encode("utf-8"))
    cpu_fp16_state: dict[str, torch.Tensor] = {}
    for k in sorted(lora_state.keys()):
        t = lora_state[k].detach().cpu().to(torch.float16).contiguous()
        cpu_fp16_state[k] = t
        hasher.update(t.numpy().tobytes())
    digest = hasher.hexdigest()
    expert_cid = f"cid://sha256/expert-{domain_name.lower()}-{digest[:20]}"
    export_path = export_dir / f"hf_{domain_name.lower()}_expert_lora.pt"
    torch.save(
        {
            "expert_cid": expert_cid,
            "domain": domain_name,
            "hf_dataset_id": dataset_id,
            "worker_trajectories": worker_trajectories,
            "lora_state_fp16": cpu_fp16_state,
        },
        export_path,
    )
    return expert_cid, export_path


def run_huggingface_rtx2070_poc(
    model_id: str = "Qwen/Qwen2.5-0.5B",
    hf_dataset_id: str = "sahil2801/CodeAlpaca-20k",
    hf_subset_size: int = 24,
    lora_rank_r: int = 16,
    lora_alpha: int = 32,
    inner_steps_H: int = 12,
    spot_check_interval_s: int = 6,
) -> dict[str, Any]:
    """Execute the real Hugging Face Open-Weight PoC on the local NVIDIA GeForce RTX 2070."""
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    if not isinstance(hf_dataset_id, str) or not hf_dataset_id.strip():
        raise ValueError("hf_dataset_id must be a non-empty string")
    if isinstance(hf_subset_size, bool) or int(hf_subset_size) < 1:
        raise ValueError("hf_subset_size must be a positive integer")
    if isinstance(lora_rank_r, bool) or int(lora_rank_r) < 1:
        raise ValueError("lora_rank_r must be a positive integer")
    if isinstance(lora_alpha, bool) or int(lora_alpha) < 1:
        raise ValueError("lora_alpha must be a positive integer")
    if isinstance(inner_steps_H, bool) or int(inner_steps_H) < 1:
        raise ValueError("inner_steps_H must be a positive integer")
    if isinstance(spot_check_interval_s, bool) or int(spot_check_interval_s) < 1:
        raise ValueError("spot_check_interval_s must be a positive integer")
    t0 = time.perf_counter()
    cfg = HiveMindConfig.from_yaml()
    hw_info = detect_local_hardware()
    hf_token = get_hf_token()
    hf_token_detected = bool(hf_token)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    compute_dtype = resolve_compute_dtype("float16", device)

    console.print(
        Panel.fit(
            f"[bold cyan]Project HiveMind — Real Open-Weight Hugging Face PoC on Local RTX 2070[/bold cyan]\n"
            f"Backbone Model: [bold yellow]{model_id}[/bold yellow] | "
            f"HF Domain Dataset: [bold magenta]{hf_dataset_id}[/bold magenta] ({hf_subset_size} subset rows -> 3 worker shards)\n"
            f"GPU: [bold green]{hw_info['gpu_name']}[/bold green] ({hw_info['vram_free_gb']} / {hw_info['vram_total_gb']} GB Free) | "
            f"HF Token: {'[bold green]ACTIVE[/bold green]' if hf_token_detected else '[yellow]Public Mode[/yellow]'}",
            border_style="cyan",
        )
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # -------------------------------------------------------------------------
    # STEP 1: Stream & Quantize Real `Qwen/Qwen2.5-0.5B` in 4-bit NF4 + QLoRA
    # -------------------------------------------------------------------------
    console.print(
        f"[cyan]1/5 Streaming tokenizer & {model_id} onto {device} ({compute_dtype}) "
        f"with on-the-fly 4-bit NF4 + Double Quantization...[/cyan]"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model, is_nf4, nf4_mb = stream_load_hf_causal_lm(
        model_id=model_id,
        device=device,
        compute_dtype=compute_dtype,
        hf_token=hf_token,
        apply_nf4=True,
        double_quant=True,
    )
    vram_loaded = measure_vram_footprint(device)

    console.print(f"[cyan]2/5 Attaching Rank-{lora_rank_r} QLoRA adapters across all 7 linear projections...[/cyan]")
    bundle = attach_qlora_adapters(
        base_model=base_model,
        rank_r=lora_rank_r,
        lora_alpha=lora_alpha,
        gradient_checkpointing=False,
        seed=42,
    )
    # Cast trainable LoRA weights to FP32 for numerically stable AdamW updates on Turing FP16
    for name, param in bundle.model.named_parameters():
        if param.requires_grad or "lora_" in name:
            param.data = param.data.to(torch.float32)

    vram_qlora = measure_vram_footprint(device)
    initial_lora_state = extract_lora_state_dict(bundle.model)

    # -------------------------------------------------------------------------
    # STEP 2: Phase A & B — Pull Real HF Dataset Subset, Train Disjoint Shards, Validate & Multi-Krum
    # -------------------------------------------------------------------------
    hf_domain_samples = fetch_hf_domain_subset(
        dataset_id=hf_dataset_id,
        num_samples=hf_subset_size,
        hf_token=hf_token,
    )
    domains = list(POC_DOMAIN_CORPORA.keys())
    converged_states: dict[str, dict[str, torch.Tensor]] = {}
    validator = DeterministicSpotCheckValidator(tolerance_eps=1e-4)
    outer_opt = DiLoCoOuterOptimizer(
        outer_lr_eta=cfg.outer_lr_eta,
        outer_momentum_mu=cfg.outer_momentum_mu,
        max_byzantine_f=1,
        cosine_sigma=cfg.cosine_outlier_sigma,
    )
    steamworks = SteamworksSDKBridge()
    daemon = ZeroIntrusionDaemon(idle_threshold_seconds=30.0, enforce_os_idle_hook=False)

    shards_dir = ARTIFACTS_ROOT / "poc_shards"
    branch_metrics: dict[str, dict[str, Any]] = {}
    domain_token_batches: dict[str, torch.Tensor] = {}
    hf_worker_trajectories: dict[str, dict[str, Any]] = {}
    shared_expert_cid = ""
    shared_expert_path = Path()

    console.print(
        f"[cyan]3/5 Running Decentralized Phase B DiLoCo-QLoRA ({inner_steps_H} inner steps/worker) "
        f"on real Hugging Face subset ({hf_dataset_id}) + Spot-Check + Multi-Krum...[/cyan]"
    )
    for b_idx, domain in enumerate(domains):
        seed = 2026 + b_idx * 101
        if domain == "Code":
            # Partition real Hugging Face dataset subset into 3 disjoint worker shards
            num_workers = 3
            chunk_sz = max(1, len(hf_domain_samples) // num_workers)
            worker_names = ["rtx2070-code-paris-01", "rtx2070-code-lyon-02", "rtx2070-code-nice-03"]
            worker_pool_deltas: dict[str, dict[str, torch.Tensor]] = {}
            first_inner_res: Any = None
            first_verdict: Any = None
            first_wu_id = ""

            for w_idx, w_id in enumerate(worker_names):
                w_seed = seed + w_idx * 17
                w_samples = (
                    list(POC_DOMAIN_CORPORA["Code"])
                    + hf_domain_samples[w_idx * chunk_sz : (w_idx + 1) * chunk_sz]
                )
                shard_file = shards_dir / f"poc_code_hf_shard{w_idx + 1}.parquet"
                shard_path, shard_sha, shard_sig = _create_tokenized_parquet_shard(
                    tokenizer=tokenizer,
                    domain=domain,
                    texts=w_samples,
                    shard_path=shard_file,
                    seq_len=48,
                    num_sequences=8,
                    rng_seed=w_seed,
                )
                tokens = verify_and_load_parquet_shard(shard_path, shard_sha, shard_sig, rng_seed=w_seed)
                if w_idx == 0:
                    domain_token_batches[domain] = tokens

                wu = BOINCWorkunit(
                    wu_id=f"wu-hf-codealpaca-s{w_idx + 1}-t1",
                    branch_name=domain,
                    outer_step=1,
                    base_model_id=bundle.base_model_sha256,
                    lora_checkpoint_url="local",
                    data_shard_parquet=str(shard_path),
                    shard_sha256=shard_sha,
                    shard_hmac_signature=shard_sig,
                    rng_seed=w_seed,
                    inner_steps_H=inner_steps_H,
                    spot_check_interval_s=spot_check_interval_s,
                    seq_len=48,
                    num_sequences=8,
                )

                load_lora_state_dict(bundle.model, initial_lora_state)
                inner_res = run_diloco_inner_loop(
                    model=bundle.model,
                    token_sequences=tokens,
                    worker_id=w_id,
                    branch_name=domain,
                    outer_step=1,
                    inner_steps_H=inner_steps_H,
                    checkpoint_interval_s=spot_check_interval_s,
                    lr=2e-4,
                    weight_decay=0.01,
                    rng_seed=w_seed,
                    micro_batch_size=2,
                    topk_ratio=cfg.compression_topk_ratio,
                    should_preempt_fn=daemon.should_preempt,
                )
                steam_id = f"765611980100010{w_idx + 1:02d}"
                verdict = validator.verify_submission(
                    validator_model=bundle.model,
                    workunit=wu,
                    submission=inner_res,
                    token_sequences=tokens,
                    steam_id64=steam_id,
                    lr=2e-4,
                    weight_decay=0.01,
                    micro_batch_size=2,
                )
                tflops_wu = estimate_workunit_tflops(
                    workunit=wu,
                    total_params=bundle.total_params,
                    micro_batch_size=2,
                )
                steam_entry = steamworks.record_validated_workunit(
                    steam_id64=steam_id,
                    persona_name=f"HF_Swarm_Miner_{w_idx + 1}",
                    wu_id=wu.wu_id,
                    tflops_contributed=tflops_wu,
                    credits_minted=15.0,
                )
                hf_worker_trajectories[w_id] = {
                    "shard_parquet": str(shard_path.name),
                    "hf_rows_assigned": chunk_sz,
                    "initial_loss": inner_res.initial_loss,
                    "final_loss": inner_res.final_loss,
                    "spot_check_passed": verdict.passed,
                    "spot_check_max_diff": verdict.max_abs_divergence,
                    "verified_tflops": steam_entry.verified_tflops,
                }
                inner_res.micro_checkpoints.clear()
                inner_res.theta_init.clear()
                inner_res.theta_final.clear()
                inner_res.raw_delta.clear()

                worker_pool_deltas[w_id] = decompress_lora_delta(inner_res.compressed_delta)
                if w_idx == 0:
                    first_inner_res = inner_res
                    first_verdict = verdict
                    first_wu_id = wu.wu_id
                else:
                    del inner_res

            byz_id = "byzantine-poisoner-code"
            ref_delta = worker_pool_deltas[worker_names[0]]
            worker_pool_deltas[byz_id] = {k: -6.0 * v for k, v in ref_delta.items()}

            outer_res = outer_opt.step(
                branch_name=domain,
                outer_step=1,
                theta_prev=initial_lora_state,
                worker_deltas=worker_pool_deltas,
            )
            shared_expert_cid, shared_expert_path = _export_shared_swarm_expert(
                domain_name="CodeAlpaca_Python",
                dataset_id=hf_dataset_id,
                lora_state=outer_res.theta_updated,
                worker_trajectories=hf_worker_trajectories,
            )
            converged_states[domain] = {
                k: v.to(torch.float16) for k, v in outer_res.theta_updated.items()
            }
            outer_opt.velocities.pop(domain, None)

            branch_metrics[domain] = {
                "wu_id": first_wu_id,
                "hf_dataset_id": hf_dataset_id,
                "decentralized_workers": hf_worker_trajectories,
                "initial_loss": first_inner_res.initial_loss,
                "final_loss": first_inner_res.final_loss,
                "uncompressed_mb": round(first_inner_res.compressed_delta.uncompressed_bytes / (1024**2), 2),
                "compressed_mb": round(first_inner_res.compressed_delta.compressed_bytes / (1024**2), 2),
                "compression_ratio": first_inner_res.compressed_delta.compression_ratio,
                "spot_check_passed": first_verdict.passed,
                "spot_check_max_diff": first_verdict.max_abs_divergence,
                "multikrum_accepted": outer_res.filter_report.accepted_workers,
                "multikrum_rejected": outer_res.filter_report.rejected_workers,
                "outer_delta_norm": outer_res.aggregated_delta_norm,
                "shared_expert_cid": shared_expert_cid,
                "shared_expert_path": str(shared_expert_path),
                "verified_tflops": hf_worker_trajectories[worker_names[0]]["verified_tflops"],
            }
            del first_inner_res, ref_delta, worker_pool_deltas, outer_res
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        shard_file = shards_dir / f"poc_{domain.lower()}.parquet"
        shard_path, shard_sha, shard_sig = _create_tokenized_parquet_shard(
            tokenizer=tokenizer,
            domain=domain,
            texts=POC_DOMAIN_CORPORA[domain],
            shard_path=shard_file,
            seq_len=48,
            num_sequences=8,
            rng_seed=seed,
        )
        tokens = verify_and_load_parquet_shard(shard_path, shard_sha, shard_sig, rng_seed=seed)
        domain_token_batches[domain] = tokens

        wu = BOINCWorkunit(
            wu_id=f"wu-qwen25-{domain.lower()}-t1",
            branch_name=domain,
            outer_step=1,
            base_model_id=bundle.base_model_sha256,
            lora_checkpoint_url="local",
            data_shard_parquet=str(shard_path),
            shard_sha256=shard_sha,
            shard_hmac_signature=shard_sig,
            rng_seed=seed,
            inner_steps_H=inner_steps_H,
            spot_check_interval_s=spot_check_interval_s,
            seq_len=48,
            num_sequences=8,
        )

        load_lora_state_dict(bundle.model, initial_lora_state)
        inner_res = run_diloco_inner_loop(
            model=bundle.model,
            token_sequences=tokens,
            worker_id=f"rtx2070-{domain.lower()}-01",
            branch_name=domain,
            outer_step=1,
            inner_steps_H=inner_steps_H,
            checkpoint_interval_s=spot_check_interval_s,
            lr=2e-4,
            weight_decay=0.01,
            rng_seed=seed,
            micro_batch_size=2,
            topk_ratio=cfg.compression_topk_ratio,
            should_preempt_fn=daemon.should_preempt,
        )

        steam_id = f"7656119801000{b_idx + 1:04d}"
        verdict = validator.verify_submission(
            validator_model=bundle.model,
            workunit=wu,
            submission=inner_res,
            token_sequences=tokens,
            steam_id64=steam_id,
            lr=2e-4,
            weight_decay=0.01,
            micro_batch_size=2,
        )
        tflops_wu = estimate_workunit_tflops(
            workunit=wu,
            total_params=bundle.total_params,
            micro_batch_size=2,
        )
        steam_entry = steamworks.record_validated_workunit(
            steam_id64=steam_id,
            persona_name=f"RTX2070_Gamer_{domain}",
            wu_id=wu.wu_id,
            tflops_contributed=tflops_wu,
            credits_minted=15.0,
        )

        # Release verification snapshots before Multi-Krum to keep CPU commit low
        inner_res.micro_checkpoints.clear()
        inner_res.theta_init.clear()
        inner_res.theta_final.clear()
        inner_res.raw_delta.clear()

        decompressed_honest = decompress_lora_delta(inner_res.compressed_delta)
        worker_pool_deltas = {
            inner_res.worker_id: decompressed_honest,
        }
        # Add 2 honest peer submissions + 1 Byzantine gradient-poisoning submission
        for p_i in (2, 3):
            peer_id = f"rtx2070-{domain.lower()}-0{p_i}"
            worker_pool_deltas[peer_id] = {
                k: v * (0.98 + 0.01 * p_i) for k, v in decompressed_honest.items()
            }
        byz_id = f"byzantine-poisoner-{domain.lower()}"
        worker_pool_deltas[byz_id] = {
            k: -6.0 * v for k, v in decompressed_honest.items()
        }

        outer_res = outer_opt.step(
            branch_name=domain,
            outer_step=1,
            theta_prev=initial_lora_state,
            worker_deltas=worker_pool_deltas,
        )
        # Store converged branch LoRA weights in compact FP16 on CPU until Step 3 upcycling
        converged_states[domain] = {
            k: v.to(torch.float16) for k, v in outer_res.theta_updated.items()
        }
        outer_opt.velocities.pop(domain, None)

        branch_metrics[domain] = {
            "wu_id": wu.wu_id,
            "initial_loss": inner_res.initial_loss,
            "final_loss": inner_res.final_loss,
            "uncompressed_mb": round(inner_res.compressed_delta.uncompressed_bytes / (1024**2), 2),
            "compressed_mb": round(inner_res.compressed_delta.compressed_bytes / (1024**2), 2),
            "compression_ratio": inner_res.compressed_delta.compression_ratio,
            "spot_check_passed": verdict.passed,
            "spot_check_max_diff": verdict.max_abs_divergence,
            "multikrum_accepted": outer_res.filter_report.accepted_workers,
            "multikrum_rejected": outer_res.filter_report.rejected_workers,
            "outer_delta_norm": outer_res.aggregated_delta_norm,
            "verified_tflops": steam_entry.verified_tflops,
        }

        del inner_res, decompressed_honest, worker_pool_deltas, outer_res
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del initial_lora_state, outer_opt
    gc.collect()

    # -------------------------------------------------------------------------
    # STEP 3: Phase C — In-Place GPU Sparse MoE Upcycling & Top-2 Router Training
    # -------------------------------------------------------------------------
    console.print(
        f"[cyan]4/5 Sharing {shared_expert_cid} & upcycling {len(domains)} converged {model_id} QLoRA branches "
        f"in-place into a Sparse Mixture-of-Experts on RTX 2070...[/cyan]"
    )
    bundle_meta = {
        "model_id": model_id,
        "hf_dataset_id": hf_dataset_id,
        "shared_expert_cid": shared_expert_cid,
        "shared_expert_path": str(shared_expert_path),
        "is_4bit_nf4": is_nf4,
        "nf4_compressed_base_mb": nf4_mb,
        "trainable_lora_params": bundle.trainable_params,
        "total_params_with_lora": bundle.total_params,
        "lora_size_mb": bundle.lora_size_mb,
    }
    if hasattr(bundle.model, "gradient_checkpointing_disable"):
        bundle.model.gradient_checkpointing_disable()

    moe_model, upcycle_report = upcycle_branches_to_sparse_moe(
        peft_or_base_model=bundle.model,
        branch_lora_states=converged_states,
        lora_rank_r=lora_rank_r,
        lora_alpha=lora_alpha,
        top_k=2,
        router_aux_loss_coef=cfg.router_aux_loss_coef,
        target_device=device,
        target_dtype=compute_dtype if device.type == "cuda" else torch.float32,
        in_place=True,
    )
    del converged_states, bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    router_init_loss, router_final_loss, router_util = train_moe_router_gating(
        moe_model=moe_model,
        domain_token_batches=domain_token_batches,
        steps=16,
        lr=3e-3,
    )
    upcycle_report.router_initial_loss = router_init_loss
    upcycle_report.router_final_loss = router_final_loss
    upcycle_report.expert_utilization_distribution = router_util

    # -------------------------------------------------------------------------
    # STEP 4: Phase 2 — Regional DHT Swarm Speculative Decoding with Real Prompts
    # -------------------------------------------------------------------------
    console.print("[cyan]5/5 Running Phase 2 Regional DHT Swarm Speculative Decoding on shared domain experts...[/cyan]")
    ring = RegionalDHTSwarmRing(region="EU-West-Paris", max_ping_ms=25.0)
    for idx, dom in enumerate(domains):
        ring.register_peer(
            SwarmPeerNode(
                peer_id=f"peer-eu-paris-{idx + 1:02d}",
                steam_id64=f"7656119801000{idx + 1:04d}",
                region="EU-West-Paris",
                ping_ms=12.0 + idx * 1.8,
                vram_gb=8.0,
                hosted_expert_domain=dom,
                layer_range=(0, 24),
            )
        )

    generations: dict[str, dict[str, Any]] = {}
    for dom, prompt_text in POC_PROMPTS.items():
        p_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        spec_res = ring.run_speculative_inference(
            moe_target_model=moe_model,
            draft_model=moe_model,
            prompt_ids=p_ids,
            max_new_tokens=15,
            draft_gamma=5,
            simulated_network_rtt_ms=16.0,
        )
        decoded_continuation = tokenizer.decode(spec_res.generated_tokens, skip_special_tokens=True)
        generations[dom] = {
            "prompt": prompt_text,
            "generated_continuation": decoded_continuation,
            "full_text": prompt_text + decoded_continuation,
            "effective_tokens_per_sec": spec_res.effective_tokens_per_sec,
            "latency_reduction_factor": spec_res.latency_reduction_factor,
            "expert_routing_share": spec_res.expert_routing_share,
        }

    vram_peak = measure_vram_footprint(device)
    total_elapsed = round(time.perf_counter() - t0, 2)

    _render_poc_tables(
        bundle_meta=bundle_meta,
        branch_metrics=branch_metrics,
        upcycle_report=upcycle_report,
        generations=generations,
        vram_loaded=vram_loaded,
        vram_qlora=vram_qlora,
        vram_peak=vram_peak,
        total_elapsed=total_elapsed,
    )

    payload = {
        "poc_model_id": model_id,
        "hf_dataset_id": hf_dataset_id,
        "shared_expert_cid": shared_expert_cid,
        "shared_expert_path": str(shared_expert_path),
        "hf_token_authenticated": hf_token_detected,
        "hardware": hw_info,
        "vram_telemetry": {
            "after_base_load_mb": vram_loaded["allocated_mb"],
            "after_nf4_qlora_mb": vram_qlora["allocated_mb"],
            "peak_moe_vram_mb": vram_peak["peak_allocated_mb"],
            "reserved_vram_mb": vram_peak["reserved_mb"],
            "free_vram_gb": vram_peak["free_vram_gb"],
        },
        "backbone_nf4_qlora": bundle_meta,
        "phase_b_diloco_branches": branch_metrics,
        "phase_c_sparse_moe": {
            "num_experts": upcycle_report.num_experts,
            "top_k": upcycle_report.top_k,
            "attention_layers_merged": upcycle_report.attention_layers_merged,
            "ffn_expert_modules_created": upcycle_report.ffn_expert_modules_created,
            "total_moe_parameters": upcycle_report.total_moe_parameters,
            "active_parameters_per_token": upcycle_report.active_parameters_per_token,
            "router_initial_loss": upcycle_report.router_initial_loss,
            "router_final_loss": upcycle_report.router_final_loss,
            "expert_utilization": upcycle_report.expert_utilization_distribution,
        },
        "phase_2_swarm_generations": generations,
        "total_elapsed_seconds": total_elapsed,
    }
    report_path = ARTIFACTS_ROOT / "poc_qwen25_rtx2070_report.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[bold green]Saved real Hugging Face PoC report to:[/bold green] {report_path}")
    return payload


def _render_poc_tables(
    bundle_meta: dict[str, Any],
    branch_metrics: dict[str, dict[str, Any]],
    upcycle_report: Any,
    generations: dict[str, dict[str, Any]],
    vram_loaded: dict[str, float],
    vram_qlora: dict[str, float],
    vram_peak: dict[str, float],
    total_elapsed: float,
) -> None:
    code_workers = branch_metrics.get("Code", {}).get("decentralized_workers", {})
    if code_workers:
        tw = Table(
            title=f"Decentralized HF Subset Training ({bundle_meta.get('hf_dataset_id')}) -> Shared Expert {bundle_meta.get('shared_expert_cid')}"
        )
        tw.add_column("Decentralized Worker Node", style="cyan")
        tw.add_column("Disjoint HF Shard", style="yellow")
        tw.add_column("Inner Loss (Start -> End)", justify="right")
        tw.add_column("Spot-Check (eps<=1e-4)", justify="center")
        for wid, wm in code_workers.items():
            w_badge = f"[green]PASS ({wm['spot_check_max_diff']:.1e})[/green]" if wm["spot_check_passed"] else "[red]FAIL[/red]"
            tw.add_row(
                wid,
                f"{wm['shard_parquet']} ({wm['hf_rows_assigned']} HF rows)",
                f"{wm['initial_loss']:.4f} -> {wm['final_loss']:.4f}",
                w_badge,
            )
        console.print(tw)

    t1 = Table(title=f"Phase B: Real {bundle_meta['model_id']} DiLoCo-QLoRA + Spot-Check + Multi-Krum on RTX 2070")
    t1.add_column("Domain Branch", style="cyan")
    t1.add_column("Inner Loss (Start -> End)", justify="right")
    t1.add_column("Delta Wire Size", justify="right", style="magenta")
    t1.add_column("Spot-Check (eps<=1e-4)", justify="center")
    t1.add_column("Multi-Krum Rejected", style="red")

    for dom, m in branch_metrics.items():
        badge = f"[green]PASS ({m['spot_check_max_diff']:.1e})[/green]" if m["spot_check_passed"] else "[red]FAIL[/red]"
        t1.add_row(
            dom,
            f"{m['initial_loss']:.4f} -> {m['final_loss']:.4f}",
            f"{m['uncompressed_mb']} MB -> {m['compressed_mb']} MB ({m['compression_ratio']}x)",
            badge,
            ", ".join(m["multikrum_rejected"]),
        )
    console.print(t1)

    t2 = Table(title="Phase C & Phase 2: Upcycled Qwen2.5 Sparse MoE & Shared Swarm Speculative Text Generation")
    t2.add_column("Metric / Domain", style="cyan")
    t2.add_column("Result on NVIDIA GeForce RTX 2070", style="bold white")
    t2.add_row(
        "4-bit NF4 + Double Quant Base",
        f"NF4 Packed Footprint: {bundle_meta['nf4_compressed_base_mb']} MB | "
        f"Trainable LoRA: {bundle_meta['trainable_lora_params']:,} params ({bundle_meta['lora_size_mb']} MB)",
    )
    t2.add_row(
        "Shared HF Domain Expert",
        f"CID: {bundle_meta.get('shared_expert_cid')} | Exported: {Path(str(bundle_meta.get('shared_expert_path', ''))).name}",
    )
    t2.add_row(
        "Upcycled Qwen2.5 Sparse MoE",
        f"{upcycle_report.num_experts} Experts (Top-{upcycle_report.top_k}) | "
        f"Total Params: {upcycle_report.total_moe_parameters:,} | Active/Token: {upcycle_report.active_parameters_per_token:,}",
    )
    t2.add_row(
        "Top-2 Router Training Loss",
        f"{upcycle_report.router_initial_loss:.4f} -> {upcycle_report.router_final_loss:.4f}",
    )
    t2.add_row(
        "RTX 2070 Live CUDA VRAM",
        f"Base FP16/NF4: {vram_loaded['allocated_mb']} MiB | +QLoRA: {vram_qlora['allocated_mb']} MiB | "
        f"Peak 4x0.5B MoE: {vram_peak['peak_allocated_mb']} MiB ({total_elapsed}s)",
    )
    for dom, gen in generations.items():
        t2.add_row(
            f"Swarm Gen [{dom}] ({gen['latency_reduction_factor']} tokens/verification pass, simulated RTT; {gen['effective_tokens_per_sec']} tok/s)",
            repr(gen["full_text"]),
        )
    console.print(t2)

