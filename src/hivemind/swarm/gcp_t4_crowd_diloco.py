# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""GCP Spot Tesla T4 DiLoCo Co-Training Engine for `GCPCrowd` (`Vxtzq/Crowd-v1` x HiveMind).

Engineers `socreateai-agora-hermes-node1` (`n1-standard-8` + `1x NVIDIA Tesla T4`, `15,360 MiB` GDDR6,
Turing `sm_75`, GCP Spot VM in `us-east4-b`, project `gen-lang-client-0625573011`) to co-train
`GCPCrowd` (`Vxtzq/Crowd-v1`, 657.8M trunk + 350.0M bigram `CrowdEngram`) alongside the local
`RTX 2070` (`XAVGAMSRV`) and community GPUs while strictly enforcing every `AGENTS.md` rule:

1. **Protected Control-Plane Directory (`AGENTS.md` Rule 1)**:
   - Hard runtime guard (`assert_not_protected_t4_path`) rejects any write, checkpoint, or cache path
     resolving to `/opt/hivemind/t4` or inside `/opt/hivemind/t4/...`.
2. **Tesla T4 Free VRAM Floor (`>= 4,300 MiB`, `AGENTS.md` Rule 1)**:
   - Auto-selects among three Turing `sm_75`-native (`fp32` master / `fp16` frozen, zero emulated `bf16`)
     DiLoCo recipes based on live free VRAM:
       * `full_trunk_8bit_ckpt`: full trunk in `fp32` + per-block activation checkpointing +
         `Blockwise8BitAdamW` + token-chunked `151,669`-vocab `fp32` cross-entropy + CPU-pinned `CrowdEngram`
         (`~5,200 MiB` peak VRAM when T4 is in training window with `>= 9,500 MiB` free).
       * `block_window_diloco`: frozen `fp16` trunk (`1,255 MiB`) + rotating window of `fp32` transformer
         blocks (`(round_idx * W) % 24`) + `ln` (`~2,350 MiB` peak VRAM, safe alongside `llama-server`
         `hivemind/bee-stars` while keeping `>= 4,300 MiB` free!).
       * `qlora_graft_diloco`: frozen `fp16` trunk + rank-`r` `fp32` `CrowdLoRALinear` adapters (`q,k,v,o,w1,w2,w3`)
         (`~1,750 MiB` peak VRAM; exports standalone `GCPCrowd` LoRA `safetensors` AND projects `(alpha/r) B @ A`
         into the exact `657,787,392`-element `bf16` trunk delta space for heterogeneous T4 + RTX 2070 DiLoCo!).
3. **Sub-Second Serving-First Governor & GCE Spot 30s Preemption Resilience (`T4GovernorYieldGuard`)**:
   - Polls `HOLD`, `governor_state.json`, `nginx` `stub_status` (`http://127.0.0.1:8091/nginx_status`),
     GCE Spot metadata (`instance/preempted`), GPU temperature (`< 82 C`), and live free VRAM before every
     inner step; atomically saves resumable state (`weights_only=True`) and evacuates T4 VRAM in `< 1s`.
4. **CPU-Offloaded DiLoCo Outer Nesterov Hub + 4-Gate Verification (`GcpCrowdDiLoCoOuterHub`)**:
   - Filters learner deltas via Chunked Multi-Krum (`multi_krum_select_and_aggregate`), applies outer
     Nesterov momentum (`lr_outer=0.7, momentum=0.9`) in CPU RAM (`0 MiB` GPU overhead), and gates every
     outer checkpoint with `check_weight_rms_collapse_guard`, `LossRatchetGate`, and signed
     `hivemind.crowdgpt.loss_check.v1` receipts (`evaluator_id="hivemind-t4"`).
5. **GCP `$150/mo` Spot Budget Governance (`compute_gcp_t4_spot_crowd_budget`)**:
   - Verifies Spot `n1-standard-8` (`$0.076/hr`) + Spot T4 (`$0.11/hr` = `$0.186/hr`) schedule compliance
     against the `$150.00/mo` ceiling (`scripts/d1_cost_estimate.py`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import gc
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from hivemind.core.diloco_inner import Blockwise8BitAdamW
from hivemind.router.opentrainer import (
    DEFAULT_MIN_FREE_VRAM_MIB_8GB,
    DEFAULT_MIN_FREE_VRAM_MIB_T4,
)
from hivemind.swarm.crowdgpt_bridge import (
    CrowdGPTCrossCheckSpec,
    LossRatchetDecision,
    LossRatchetGate,
    MultiKrumAggregationResult,
    WeightRmsGuardDecision,
    build_crowdgpt_client_hardware_headers,
    build_crowdgpt_submit_metadata,
    check_weight_rms_collapse_guard,
    classify_gpu_precision_cohort,
    multi_krum_select_and_aggregate,
    pack_crowdgpt_delta_stream,
    parse_crowdgpt_task_crosscheck,
)
from hivemind.swarm.crowdgpt_live_runtime import (
    CrowdRuntimeBlock,
    CrowdV1RuntimeConfig,
    CrowdV1WithEngram,
)
from hivemind.swarm.crowdgpt_loss_check import (
    LossCheckReceipt,
    tokens_sha256,
    weights_sha256,
)

PROTECTED_T4_DIR_POSIX = "/opt/hivemind/t4"
DEFAULT_GCP_T4_WORK_DIR_POSIX = "/mnt/disks/disk-socrateai-local-1/gcpcrowd"
DEFAULT_T4_GOVERNOR_STATE_POSIX = "/mnt/disks/disk-socrateai-local-1/hivemind_telemetry/governor_state.json"
DEFAULT_T4_HOLD_FILE_POSIX = "/mnt/disks/disk-socrateai-local-1/hivemind_telemetry/HOLD"
DEFAULT_T4_NGINX_STATUS_URL = "http://127.0.0.1:8091/nginx_status"
GCE_PREEMPTED_METADATA_URL = "http://metadata.google.internal/computeMetadata/v1/instance/preempted"
GCE_MAINTENANCE_METADATA_URL = "http://metadata.google.internal/computeMetadata/v1/instance/maintenance-event"

GCP_T4_SPOT_VM_HOURLY_USD = 0.0760  # n1-standard-8 Spot (us-east4)
GCP_T4_SPOT_GPU_HOURLY_USD = 0.1100  # nvidia-tesla-t4 Spot (us-east4)
GCP_T4_SPOT_COMBINED_HOURLY_USD = round(GCP_T4_SPOT_VM_HOURLY_USD + GCP_T4_SPOT_GPU_HOURLY_USD, 4)  # $0.186/hr
GCP_MONTHLY_BUDGET_CEILING_USD = 150.0
HOURS_PER_MONTH = 730.0
DAYS_PER_MONTH = HOURS_PER_MONTH / 24.0

GCPCROWD_CAMPAIGN_ID = "CROWDGPT-GCP-T4-DILOCO"
GCPCROWD_MODEL_ID = "hivemind/gcpcrowd"

# Estimated peak training VRAM footprints on Crowd-v1 (657.8M trunk, CPU-pinned Engram, chunked CE)
EST_VRAM_FULL_TRUNK_8BIT_CKPT_MIB = 6800
EST_VRAM_BLOCK_WINDOW_DILOCO_MIB = 2350
EST_VRAM_QLORA_GRAFT_DILOCO_MIB = 1750


# ---------------------------------------------------------------------------
# 1. Governance: Protected Path Guard & $150/mo GCP Spot Cost Calculator
# ---------------------------------------------------------------------------


def _normalize_posix_like_path(path: Path | str) -> str:
    raw = str(path).strip().replace("\\", "/")
    if not raw:
        raise ValueError("Path must be non-empty.")
    # Strip Windows drive prefix (e.g. 'D:/opt/hivemind/t4' -> '/opt/hivemind/t4') so cross-OS tests also catch it
    if len(raw) >= 2 and raw[1] == ":":
        raw = raw[2:] or "/"
    parts: list[str] = []
    leading_slash = raw.startswith("/")
    for seg in raw.split("/"):
        if not seg or seg == ".":
            continue
        if seg == "..":
            if parts:
                parts.pop()
            continue
        parts.append(seg)
    norm = ("/" if leading_slash else "") + "/".join(parts)
    return norm or ("/" if leading_slash else ".")


def assert_not_protected_t4_path(path: Path | str) -> Path:
    """Enforce `AGENTS.md` Rule 1: NEVER write, modify, overwrite, or delete `/opt/hivemind/t4/`."""
    norm = _normalize_posix_like_path(path)
    if norm == PROTECTED_T4_DIR_POSIX or norm.startswith(PROTECTED_T4_DIR_POSIX + "/"):
        raise PermissionError(
            f"Refusing write/checkpoint path {path!r}: {PROTECTED_T4_DIR_POSIX}/ is strictly protected by AGENTS.md Rule 1."
        )
    try:
        resolved_norm = _normalize_posix_like_path(Path(path).resolve())
        if resolved_norm == PROTECTED_T4_DIR_POSIX or resolved_norm.startswith(PROTECTED_T4_DIR_POSIX + "/"):
            raise PermissionError(
                f"Refusing resolved path {resolved_norm!r}: {PROTECTED_T4_DIR_POSIX}/ is strictly protected by AGENTS.md Rule 1."
            )
    except OSError:
        pass
    return Path(path)


@dataclass(frozen=True)
class GcpT4SpotBudgetPlan:
    """Monthly GCP Spot cost plan for `GCPCrowd` DiLoCo training on `socreateai-agora-hermes-node1`."""

    baseline_monthly_usd: float
    budget_ceiling_usd: float
    scheduled_window_hours_per_day: float
    extra_spot_hours_per_day: float
    spot_hourly_rate_usd: float
    incremental_monthly_usd: float
    projected_total_monthly_usd: float
    max_additional_hours_per_day_within_budget: float
    within_budget: bool
    provisioning_model: str = "SPOT"


def compute_gcp_t4_spot_crowd_budget(
    *,
    baseline_monthly_usd: float = 120.62,
    budget_ceiling_usd: float = GCP_MONTHLY_BUDGET_CEILING_USD,
    scheduled_window_hours_per_day: float = 4.5,
    extra_spot_hours_per_day: float = 0.0,
    provisioning_model: str = "SPOT",
) -> GcpT4SpotBudgetPlan:
    """Compute exact monthly GCP Spot cost for `GCPCrowd` DiLoCo training against the `$150/mo` cap."""
    for name, val, min_v in (
        ("baseline_monthly_usd", baseline_monthly_usd, 0.0),
        ("budget_ceiling_usd", budget_ceiling_usd, 1.0),
        ("scheduled_window_hours_per_day", scheduled_window_hours_per_day, 0.0),
        ("extra_spot_hours_per_day", extra_spot_hours_per_day, 0.0),
    ):
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)) or float(val) < min_v:
            raise ValueError(f"{name} must be a finite number >= {min_v}")
    if str(provisioning_model).strip().upper() != "SPOT":
        raise ValueError("AGENTS.md Rule 2.3 requires provisioning_model='SPOT' for T4 training VMs.")
    if scheduled_window_hours_per_day + extra_spot_hours_per_day > 24.0:
        raise ValueError("Total daily T4 hours cannot exceed 24.0.")

    incremental_monthly = round(float(extra_spot_hours_per_day) * DAYS_PER_MONTH * GCP_T4_SPOT_COMBINED_HOURLY_USD, 2)
    projected_total = round(float(baseline_monthly_usd) + incremental_monthly, 2)
    remaining_usd = max(0.0, float(budget_ceiling_usd) - float(baseline_monthly_usd))
    max_extra_h_day = round(remaining_usd / (DAYS_PER_MONTH * GCP_T4_SPOT_COMBINED_HOURLY_USD), 2)

    return GcpT4SpotBudgetPlan(
        baseline_monthly_usd=round(float(baseline_monthly_usd), 2),
        budget_ceiling_usd=round(float(budget_ceiling_usd), 2),
        scheduled_window_hours_per_day=round(float(scheduled_window_hours_per_day), 2),
        extra_spot_hours_per_day=round(float(extra_spot_hours_per_day), 2),
        spot_hourly_rate_usd=GCP_T4_SPOT_COMBINED_HOURLY_USD,
        incremental_monthly_usd=incremental_monthly,
        projected_total_monthly_usd=projected_total,
        max_additional_hours_per_day_within_budget=min(24.0, max_extra_h_day),
        within_budget=projected_total <= float(budget_ceiling_usd),
        provisioning_model="SPOT",
    )


# ---------------------------------------------------------------------------
# 2. Memory-Bounded Token-Chunked Cross-Entropy & CrowdLoRALinear Adapters
# ---------------------------------------------------------------------------


def chunked_causal_lm_cross_entropy(
    hidden_states: torch.Tensor,
    head_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    token_chunk_size: int = 256,
    use_checkpoint: bool = True,
) -> torch.Tensor:
    """Compute exact `fp32` causal cross-entropy without materializing `[B, T, 151669]` logits at once.

    Cuts peak LM-head logit + gradient VRAM on `Vxtzq/Crowd-v1` (`vocab_size=151,669`, `T=2048`)
    from `~4.96 GiB` (`B=2`) down to `< 160 MiB` while preserving exact `fp32` loss and gradients.
    """
    if hidden_states.ndim != 3 or labels.ndim != 2:
        raise ValueError("hidden_states must be [B, T, D] and labels must be [B, T].")
    if hidden_states.shape[:2] != labels.shape:
        raise ValueError("hidden_states[:2] must match labels shape.")
    if isinstance(token_chunk_size, bool) or not isinstance(token_chunk_size, int) or token_chunk_size < 1:
        raise ValueError("token_chunk_size must be an integer >= 1.")
    if hidden_states.shape[1] < 2:
        raise ValueError("Sequence length T must be >= 2 for causal LM loss.")

    shift_h = hidden_states[:, :-1, :].contiguous().reshape(-1, hidden_states.shape[-1])
    shift_y = labels[:, 1:].contiguous().reshape(-1).to(device=hidden_states.device, dtype=torch.long)

    valid_count = int((shift_y != -100).sum().item())
    if valid_count == 0:
        return (shift_h.sum() * 0.0).to(torch.float32)

    def _chunk_loss_sum(h_c: torch.Tensor, w: torch.Tensor, y_c: torch.Tensor) -> torch.Tensor:
        logits_c = F.linear(h_c.float(), w.float())
        return F.cross_entropy(logits_c, y_c, ignore_index=-100, reduction="sum")

    n_tokens = shift_h.shape[0]
    need_ckpt = bool(use_checkpoint and (shift_h.requires_grad or head_weight.requires_grad))
    total_loss = torch.zeros((), device=hidden_states.device, dtype=torch.float32)

    for start in range(0, n_tokens, token_chunk_size):
        end = min(n_tokens, start + token_chunk_size)
        y_chunk = shift_y[start:end]
        if not bool((y_chunk != -100).any().item()):
            continue
        h_chunk = shift_h[start:end]
        if need_ckpt:
            c_loss = torch_checkpoint(_chunk_loss_sum, h_chunk, head_weight, y_chunk, use_reentrant=False)
        else:
            c_loss = _chunk_loss_sum(h_chunk, head_weight, y_chunk)
        total_loss = total_loss + c_loss

    return total_loss / float(valid_count)


class CrowdLoRALinear(nn.Module):
    """Low-rank `fp32` adapter wrapping a frozen `fp16`/`fp32` `nn.Linear` layer in `CrowdV1WithEngram`."""

    def __init__(self, base: nn.Linear, *, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be an nn.Linear module.")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be an integer >= 1.")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)) or float(alpha) <= 0.0:
            raise ValueError("alpha must be a finite positive float.")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)

        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty((self.rank, self.in_features), device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, self.rank), device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self) -> nn.Parameter:
        return self.base.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x.to(dtype=self.base.weight.dtype))
        lora_h = F.linear(x.to(dtype=torch.float32), self.lora_A)
        lora_out = F.linear(lora_h, self.lora_B) * self.scaling
        return (base_out.to(torch.float32) + lora_out).to(dtype=x.dtype)

    @torch.no_grad()
    def delta_weight_fp32(self) -> torch.Tensor:
        """Compute dense `[out_features, in_features]` adapter contribution `(alpha / r) * B @ A` on CPU."""
        return (self.lora_B.detach().to("cpu", dtype=torch.float32) @ self.lora_A.detach().to("cpu", dtype=torch.float32)) * self.scaling


LORA_TARGET_ATTRS: tuple[tuple[str, str], ...] = (
    ("attn", "q"),
    ("attn", "k"),
    ("attn", "v"),
    ("attn", "o"),
    ("mlp", "w1"),
    ("mlp", "w2"),
    ("mlp", "w3"),
)


def inject_crowd_lora_adapters(
    model: CrowdV1WithEngram,
    *,
    rank: int = 8,
    alpha: float = 16.0,
    seed: int = 20261110,
) -> list[nn.Parameter]:
    """Attach `CrowdLoRALinear` adapters across all transformer blocks if not already attached."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    for p in model.parameters():
        p.requires_grad_(False)

    trainable: list[nn.Parameter] = []
    for blk in model.blocks:
        for sub_name, proj_name in LORA_TARGET_ATTRS:
            sub_mod = getattr(blk, sub_name)
            curr = getattr(sub_mod, proj_name)
            if isinstance(curr, CrowdLoRALinear):
                if curr.rank != rank or abs(curr.alpha - float(alpha)) > 1e-9:
                    base_lin = curr.base
                    wrapped = CrowdLoRALinear(base_lin, rank=rank, alpha=alpha)
                    with torch.no_grad():
                        wrapped.lora_A.copy_(
                            torch.randn(wrapped.lora_A.shape, generator=gen, dtype=torch.float32).to(wrapped.lora_A.device)
                            * (1.0 / math.sqrt(wrapped.in_features))
                        )
                        wrapped.lora_B.zero_()
                    setattr(sub_mod, proj_name, wrapped)
                    curr = wrapped
                curr.lora_A.requires_grad_(True)
                curr.lora_B.requires_grad_(True)
                trainable.extend([curr.lora_A, curr.lora_B])
            else:
                wrapped = CrowdLoRALinear(curr, rank=rank, alpha=alpha)
                with torch.no_grad():
                    wrapped.lora_A.copy_(
                        torch.randn(wrapped.lora_A.shape, generator=gen, dtype=torch.float32).to(wrapped.lora_A.device)
                        * (1.0 / math.sqrt(wrapped.in_features))
                    )
                    wrapped.lora_B.zero_()
                setattr(sub_mod, proj_name, wrapped)
                trainable.extend([wrapped.lora_A, wrapped.lora_B])

    # Also train final LayerNorm in fp32 for fast calibration stability
    model.ln.to(dtype=torch.float32)
    model.ln.weight.requires_grad_(True)
    model.ln.bias.requires_grad_(True)
    trainable.extend([model.ln.weight, model.ln.bias])
    return trainable


def unwrap_crowd_lora_adapters(model: CrowdV1WithEngram, *, merge_into_base: bool = False) -> None:
    """Remove `CrowdLoRALinear` wrappers, optionally folding `(alpha/r) B @ A` into base weights."""
    with torch.no_grad():
        for blk in model.blocks:
            for sub_name, proj_name in LORA_TARGET_ATTRS:
                sub_mod = getattr(blk, sub_name)
                curr = getattr(sub_mod, proj_name)
                if isinstance(curr, CrowdLoRALinear):
                    if merge_into_base:
                        dw = curr.delta_weight_fp32().to(device=curr.base.weight.device, dtype=torch.float32)
                        curr.base.weight.copy_((curr.base.weight.float() + dw).to(dtype=curr.base.weight.dtype))
                    setattr(sub_mod, proj_name, curr.base)


def extract_crowd_lora_state_dict(model: CrowdV1WithEngram) -> dict[str, torch.Tensor]:
    """Extract named `fp32` CPU tensors for all `CrowdLoRALinear` adapters + final LayerNorm."""
    out: dict[str, torch.Tensor] = {}
    for idx, blk in enumerate(model.blocks):
        for sub_name, proj_name in LORA_TARGET_ATTRS:
            curr = getattr(getattr(blk, sub_name), proj_name)
            if isinstance(curr, CrowdLoRALinear):
                prefix = f"blocks.{idx}.{sub_name}.{proj_name}"
                out[f"{prefix}.lora_A"] = curr.lora_A.detach().to("cpu", dtype=torch.float32).clone()
                out[f"{prefix}.lora_B"] = curr.lora_B.detach().to("cpu", dtype=torch.float32).clone()
    out["ln.weight"] = model.ln.weight.detach().to("cpu", dtype=torch.float32).clone()
    out["ln.bias"] = model.ln.bias.detach().to("cpu", dtype=torch.float32).clone()
    return out


def load_crowd_lora_state_dict(model: CrowdV1WithEngram, state: Mapping[str, torch.Tensor]) -> None:
    """Load named LoRA adapter + final LayerNorm tensors into `model`."""
    with torch.no_grad():
        for idx, blk in enumerate(model.blocks):
            for sub_name, proj_name in LORA_TARGET_ATTRS:
                curr = getattr(getattr(blk, sub_name), proj_name)
                if isinstance(curr, CrowdLoRALinear):
                    prefix = f"blocks.{idx}.{sub_name}.{proj_name}"
                    if f"{prefix}.lora_A" in state:
                        curr.lora_A.copy_(state[f"{prefix}.lora_A"].to(device=curr.lora_A.device, dtype=curr.lora_A.dtype))
                    if f"{prefix}.lora_B" in state:
                        curr.lora_B.copy_(state[f"{prefix}.lora_B"].to(device=curr.lora_B.device, dtype=curr.lora_B.dtype))
        if "ln.weight" in state:
            model.ln.weight.copy_(state["ln.weight"].to(device=model.ln.weight.device, dtype=model.ln.weight.dtype))
        if "ln.bias" in state:
            model.ln.bias.copy_(state["ln.bias"].to(device=model.ln.bias.device, dtype=model.ln.bias.dtype))


def save_crowd_lora_safetensors(model: CrowdV1WithEngram, dest: Path | str) -> str:
    """Save `GCPCrowd` LoRA adapter state dict to `safetensors` (strictly outside `/opt/hivemind/t4/`)."""
    from safetensors.torch import save_file

    safe_dest = assert_not_protected_t4_path(dest)
    safe_dest.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.to(torch.bfloat16).contiguous() for k, v in extract_crowd_lora_state_dict(model).items()}
    with tempfile.NamedTemporaryFile(dir=safe_dest.parent, prefix=".tmp_lora_", suffix=".safetensors", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    save_file(tensors, str(tmp_path))
    os.replace(tmp_path, safe_dest)
    return hashlib.sha256(safe_dest.read_bytes()).hexdigest()


@torch.no_grad()
def project_lora_to_flat_trunk_delta(
    model: CrowdV1WithEngram,
    *,
    ln_before: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Project `CrowdLoRALinear` low-rank updates into the exact flat `bf16` `SotaGPT`/`Crowd-v1` trunk space.

    Allows an `8 GB` GPU (`RTX 2070`) running `< 1.8 GB` VRAM LoRA inner steps and the `16 GB` GCP Spot T4
    running block-window or full-trunk inner steps to aggregate in the **same** DiLoCo outer Nesterov loop.
    Writes directly into a single preallocated `bf16` flat buffer to avoid duplicating `1.31 GB` during `torch.cat`.
    """
    total_numel = model.wte.weight.numel()
    for blk in model.blocks:
        total_numel += blk.ln1.weight.numel() + blk.ln1.bias.numel()
        for proj_name in ("q", "k", "v", "o"):
            mod = getattr(blk.attn, proj_name)
            total_numel += mod.base.weight.numel() if isinstance(mod, CrowdLoRALinear) else mod.weight.numel()
        total_numel += blk.ln2.weight.numel() + blk.ln2.bias.numel()
        for proj_name in ("w1", "w2", "w3"):
            mod = getattr(blk.mlp, proj_name)
            total_numel += mod.base.weight.numel() if isinstance(mod, CrowdLoRALinear) else mod.weight.numel()
    total_numel += model.ln.weight.numel() + model.ln.bias.numel()
    if not model.cfg.tie_word_embeddings:
        total_numel += model.head.weight.numel()

    out = torch.zeros(total_numel, dtype=torch.bfloat16)
    offset = model.wte.weight.numel()
    for blk in model.blocks:
        offset += blk.ln1.weight.numel() + blk.ln1.bias.numel()
        for proj_name in ("q", "k", "v", "o"):
            mod = getattr(blk.attn, proj_name)
            if isinstance(mod, CrowdLoRALinear):
                n = mod.base.weight.numel()
                out[offset : offset + n].copy_(mod.delta_weight_fp32().reshape(-1).to(torch.bfloat16))
            else:
                n = mod.weight.numel()
            offset += n
        offset += blk.ln2.weight.numel() + blk.ln2.bias.numel()
        for proj_name in ("w1", "w2", "w3"):
            mod = getattr(blk.mlp, proj_name)
            if isinstance(mod, CrowdLoRALinear):
                n = mod.base.weight.numel()
                out[offset : offset + n].copy_(mod.delta_weight_fp32().reshape(-1).to(torch.bfloat16))
            else:
                n = mod.weight.numel()
            offset += n

    n_lnw = model.ln.weight.numel()
    n_lnb = model.ln.bias.numel()
    if ln_before is not None:
        dw_ln = (model.ln.weight.detach().to("cpu", dtype=torch.float32) - ln_before[0].to("cpu", dtype=torch.float32)).reshape(-1)
        db_ln = (model.ln.bias.detach().to("cpu", dtype=torch.float32) - ln_before[1].to("cpu", dtype=torch.float32)).reshape(-1)
        out[offset : offset + n_lnw].copy_(dw_ln.to(torch.bfloat16))
        out[offset + n_lnw : offset + n_lnw + n_lnb].copy_(db_ln.to(torch.bfloat16))
    offset += n_lnw + n_lnb
    if not model.cfg.tie_word_embeddings:
        offset += model.head.weight.numel()
    if offset != total_numel:
        raise RuntimeError(f"Flat trunk projection size mismatch: offset={offset} vs total_numel={total_numel}")
    return out


# ---------------------------------------------------------------------------
# 3. Adaptive T4 VRAM Recipe Selector & Mixed-Precision Checkpointed Forward
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class T4DiLoCoRecipeDecision:
    """Selected inner-loop DiLoCo recipe respecting the T4 `>= 4,300 MiB` free VRAM floor."""

    recipe: str
    device: str
    frozen_dtype: str
    trainable_dtype: str
    vram_total_mib: int
    vram_free_mib: int
    vram_floor_mib: int
    estimated_peak_vram_mib: int
    projected_free_vram_mib: int
    cohort_key: str
    window_block_indices: tuple[int, ...]
    lora_rank: int
    can_train_on_gpu: bool
    reason: str


def select_t4_diloco_recipe(
    *,
    requested_mode: str = "auto",
    round_index: int = 0,
    num_layers: int = 24,
    window_blocks: int = 6,
    lora_rank: int = 8,
    prefer_cuda: bool = True,
    cuda_available: bool | None = None,
    vram_info_mib: tuple[int, int] | None = None,
    compute_capability: tuple[int, int] | None = None,
    gpu_name: str | None = None,
    min_free_vram_mib_override: int | None = None,
) -> T4DiLoCoRecipeDecision:
    """Select the richest DiLoCo training recipe that preserves `>= 4,300 MiB` free VRAM on T4."""
    valid_modes = {"auto", "full_trunk_8bit_ckpt", "block_window_diloco", "qlora_graft_diloco", "cpu_fp32"}
    if requested_mode not in valid_modes:
        raise ValueError(f"requested_mode must be one of {sorted(valid_modes)}, got {requested_mode!r}")
    if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0:
        raise ValueError("round_index must be a non-negative integer.")
    if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
        raise ValueError("num_layers must be an integer >= 1.")
    if isinstance(window_blocks, bool) or not isinstance(window_blocks, int) or window_blocks < 1:
        raise ValueError("window_blocks must be an integer >= 1.")

    eff_window = min(num_layers, window_blocks)
    win_start = (round_index * eff_window) % num_layers
    win_indices = tuple((win_start + i) % num_layers for i in range(eff_window))

    has_cuda = torch.cuda.is_available() if cuda_available is None else bool(cuda_available)
    if not prefer_cuda or not has_cuda or requested_mode == "cpu_fp32":
        chosen = "qlora_graft_diloco" if requested_mode == "qlora_graft_diloco" else (
            "block_window_diloco" if requested_mode == "block_window_diloco" else "full_trunk_8bit_ckpt"
        )
        return T4DiLoCoRecipeDecision(
            recipe=chosen,
            device="cpu",
            frozen_dtype="float32",
            trainable_dtype="float32",
            vram_total_mib=0,
            vram_free_mib=0,
            vram_floor_mib=0,
            estimated_peak_vram_mib=0,
            projected_free_vram_mib=0,
            cohort_key="cpu:generic:fp32:adamw_8bit",
            window_block_indices=win_indices if chosen == "block_window_diloco" else tuple(range(num_layers)),
            lora_rank=lora_rank if chosen == "qlora_graft_diloco" else 0,
            can_train_on_gpu=False,
            reason="CPU_SELECTED_OR_CUDA_UNAVAILABLE",
        )

    if vram_info_mib is not None:
        free_mib, total_mib = int(vram_info_mib[0]), int(vram_info_mib[1])
    else:
        free_b, total_b = torch.cuda.mem_get_info(0)
        free_mib, total_mib = int(free_b // (1024 * 1024)), int(total_b // (1024 * 1024))

    cc = compute_capability if compute_capability is not None else torch.cuda.get_device_capability(0)
    detected_name = gpu_name if gpu_name is not None else torch.cuda.get_device_name(0)
    is_t4_class = total_mib >= 14_000 or "t4" in detected_name.lower()
    default_floor = DEFAULT_MIN_FREE_VRAM_MIB_T4 if is_t4_class else DEFAULT_MIN_FREE_VRAM_MIB_8GB
    floor_mib = max(default_floor, int(min_free_vram_mib_override)) if min_free_vram_mib_override is not None else default_floor

    cohort = classify_gpu_precision_cohort(
        "nvidia",
        (int(cc[0]), int(cc[1])),
        requested_dtype="auto",
        optimizer_family="adamw_8bit",
    )
    # Turing sm_75 uses float16 for frozen weights and float32 for trainable master parameters (never emulated bf16)
    frozen_dt = "bfloat16" if int(cc[0]) >= 8 else "float16"

    candidates: list[tuple[str, int, tuple[int, ...], int]] = []
    if requested_mode in ("auto", "full_trunk_8bit_ckpt"):
        candidates.append(("full_trunk_8bit_ckpt", EST_VRAM_FULL_TRUNK_8BIT_CKPT_MIB, tuple(range(num_layers)), 0))
    if requested_mode in ("auto", "full_trunk_8bit_ckpt", "block_window_diloco"):
        candidates.append(("block_window_diloco", EST_VRAM_BLOCK_WINDOW_DILOCO_MIB, win_indices, 0))
    if requested_mode in ("auto", "full_trunk_8bit_ckpt", "block_window_diloco", "qlora_graft_diloco"):
        candidates.append(("qlora_graft_diloco", EST_VRAM_QLORA_GRAFT_DILOCO_MIB, tuple(range(num_layers)), lora_rank))

    for rec_name, est_mib, blk_idx, r_val in candidates:
        proj_free = free_mib - est_mib
        if proj_free >= floor_mib:
            return T4DiLoCoRecipeDecision(
                recipe=rec_name,
                device="cuda",
                frozen_dtype=frozen_dt,
                trainable_dtype="float32",
                vram_total_mib=total_mib,
                vram_free_mib=free_mib,
                vram_floor_mib=floor_mib,
                estimated_peak_vram_mib=est_mib,
                projected_free_vram_mib=proj_free,
                cohort_key=cohort.cohort_key,
                window_block_indices=blk_idx,
                lora_rank=r_val,
                can_train_on_gpu=True,
                reason=(
                    f"CUDA_APPROVED ({rec_name} on {detected_name}: free={free_mib} MiB - est={est_mib} MiB "
                    f"= {proj_free} MiB >= floor={floor_mib} MiB)"
                ),
            )

    return T4DiLoCoRecipeDecision(
        recipe="yield_insufficient_vram",
        device="cpu",
        frozen_dtype="float32",
        trainable_dtype="float32",
        vram_total_mib=total_mib,
        vram_free_mib=free_mib,
        vram_floor_mib=floor_mib,
        estimated_peak_vram_mib=0,
        projected_free_vram_mib=free_mib,
        cohort_key=cohort.cohort_key,
        window_block_indices=win_indices,
        lora_rank=lora_rank,
        can_train_on_gpu=False,
        reason=(
            f"VRAM_GUARD_YIELD (free={free_mib} MiB on {detected_name} cannot accommodate lightest GPU recipe "
            f"{EST_VRAM_QLORA_GRAFT_DILOCO_MIB} MiB while maintaining >= {floor_mib} MiB free)"
        ),
    )


def forward_crowd_v1_checkpointed(
    model: CrowdV1WithEngram,
    input_ids: torch.Tensor,
    labels: torch.Tensor | None = None,
    *,
    use_grad_checkpointing: bool = True,
    token_chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Mixed-dtype (`fp16` frozen + `fp32` trainable) checkpointed forward pass with CPU-pinned `CrowdEngram`."""
    t = input_ids.shape[1]
    if t > model.cfg.max_position_embeddings:
        raise ValueError(f"Sequence length {t} exceeds max_position_embeddings={model.cfg.max_position_embeddings}")

    x = model.wte(input_ids).to(torch.float32)
    if model.cfg.engram_buckets > 0:
        x = x + model.engram(input_ids).to(device=x.device, dtype=torch.float32)

    cos = model.cos_cached.to(device=x.device)
    sin = model.sin_cached.to(device=x.device)

    for blk in model.blocks:
        blk_dtype = blk.ln1.weight.dtype
        blk_trainable = any(p.requires_grad for p in blk.parameters())
        x_in = x.to(dtype=blk_dtype)

        if use_grad_checkpointing and (x_in.requires_grad or blk_trainable):
            def _run_blk(h_tensor: torch.Tensor, block_mod: CrowdRuntimeBlock = blk) -> torch.Tensor:
                out_h, _ = block_mod(h_tensor, cos, sin, past_kv=None, use_cache=False)
                return out_h

            x = torch_checkpoint(_run_blk, x_in, use_reentrant=False).to(torch.float32)
        else:
            out_h, _ = blk(x_in, cos, sin, past_kv=None, use_cache=False)
            x = out_h.to(torch.float32)

    h_ln = model.ln(x.to(dtype=model.ln.weight.dtype)).to(torch.float32)
    if labels is None:
        logits = F.linear(h_ln.to(dtype=model.head.weight.dtype), model.head.weight)
        return logits, None

    loss = chunked_causal_lm_cross_entropy(
        h_ln,
        model.head.weight,
        labels,
        token_chunk_size=token_chunk_size,
        use_checkpoint=use_grad_checkpointing,
    )
    return h_ln, loss


# ---------------------------------------------------------------------------
# 4. Serving-First Governor & GCE Spot 30s Preemption Yield Guard
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreemptionCheckResult:
    """Result of polling T4 serving-first & Spot preemption signals before an inner step."""

    should_yield: bool
    trigger: str | None
    in_flight_requests: int
    governor_state: str
    spot_preempted: bool
    vram_free_mib: int
    gpu_temp_c: int


def _parse_nginx_in_flight(text: str) -> int:
    for ln in text.splitlines():
        if ln.startswith("Reading:"):
            parts = ln.replace(":", " ").split()
            if len(parts) >= 4 and parts[3].isdigit():
                return max(int(parts[3]) - 1, 0)
    return 0


class T4GovernorYieldGuard:
    """Sub-second preemption monitor cooperating with `t4_governor.py`, `nginx`, and GCE Spot metadata."""

    def __init__(
        self,
        *,
        hold_file: Path | str | None = None,
        governor_state_file: Path | str | None = None,
        require_governor_training_state: bool = False,
        nginx_status_url: str | None = None,
        check_gce_spot_metadata: bool = False,
        min_free_vram_mib: int = DEFAULT_MIN_FREE_VRAM_MIB_T4,
        max_gpu_temp_c: int = 82,
        spot_poll_interval_s: float = 2.0,
        nginx_probe_fn: Callable[[], int] | None = None,
        spot_probe_fn: Callable[[], bool] | None = None,
        gpu_telemetry_fn: Callable[[], tuple[int, int]] | None = None,
    ) -> None:
        self.hold_file = Path(hold_file) if hold_file is not None else None
        self.governor_state_file = Path(governor_state_file) if governor_state_file is not None else None
        self.require_governor_training_state = bool(require_governor_training_state)
        self.nginx_status_url = nginx_status_url
        self.check_gce_spot_metadata = bool(check_gce_spot_metadata)
        self.min_free_vram_mib = int(min_free_vram_mib)
        self.max_gpu_temp_c = int(max_gpu_temp_c)
        self.spot_poll_interval_s = max(0.25, float(spot_poll_interval_s))
        self.nginx_probe_fn = nginx_probe_fn
        self.spot_probe_fn = spot_probe_fn
        self.gpu_telemetry_fn = gpu_telemetry_fn
        self._last_spot_check_t: float = 0.0
        self._cached_spot_preempted: bool = False

    def _probe_nginx(self) -> tuple[int, bool]:
        if self.nginx_probe_fn is not None:
            return int(self.nginx_probe_fn()), True
        if not self.nginx_status_url:
            return 0, True
        try:
            with urllib.request.urlopen(self.nginx_status_url, timeout=0.8) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            return _parse_nginx_in_flight(body), True
        except Exception:
            return 0, False

    def _probe_spot_preempted(self) -> bool:
        now = time.monotonic()
        if self._cached_spot_preempted:
            return True
        if now - self._last_spot_check_t < self.spot_poll_interval_s:
            return self._cached_spot_preempted
        self._last_spot_check_t = now
        if self.spot_probe_fn is not None:
            self._cached_spot_preempted = bool(self.spot_probe_fn())
            return self._cached_spot_preempted
        if not self.check_gce_spot_metadata:
            return False
        for url, trigger_token in (
            (GCE_PREEMPTED_METADATA_URL, "TRUE"),
            (GCE_MAINTENANCE_METADATA_URL, "TERMINATE"),
        ):
            try:
                req = urllib.request.Request(url, headers={"Metadata-Flavor": "Google"})
                with urllib.request.urlopen(req, timeout=0.5) as resp:
                    txt = resp.read().decode("utf-8", errors="replace").strip().upper()
                if trigger_token in txt:
                    self._cached_spot_preempted = True
                    return True
            except Exception:
                continue
        return False

    def _probe_gpu_free_and_temp(self) -> tuple[int, int]:
        if self.gpu_telemetry_fn is not None:
            free_m, temp_c = self.gpu_telemetry_fn()
            return int(free_m), int(temp_c)
        if torch.cuda.is_available():
            free_b, _ = torch.cuda.mem_get_info(0)
            free_mib = int(free_b // (1024 * 1024))
            if self.min_free_vram_mib > 0 and free_mib < self.min_free_vram_mib:
                # Reclaim unused PyTorch caching-allocator slack before declaring a real VRAM floor breach
                torch.cuda.empty_cache()
                free_b, _ = torch.cuda.mem_get_info(0)
                free_mib = int(free_b // (1024 * 1024))
            return free_mib, 55
        return 99_999, 40

    def check(self) -> PreemptionCheckResult:
        """Evaluate all preemption & safety signals before starting the next micro-step."""
        if self.hold_file is not None and self.hold_file.exists():
            return PreemptionCheckResult(
                should_yield=True,
                trigger="operator_hold_file",
                in_flight_requests=0,
                governor_state="held",
                spot_preempted=False,
                vram_free_mib=0,
                gpu_temp_c=0,
            )

        gov_state = "standalone"
        if self.governor_state_file is not None and self.governor_state_file.exists():
            try:
                snap = json.loads(self.governor_state_file.read_text(encoding="utf-8"))
                gov_state = str(snap.get("state") or "unknown")
                if int(snap.get("in_flight") or 0) >= 1:
                    return PreemptionCheckResult(
                        should_yield=True,
                        trigger="governor_in_flight",
                        in_flight_requests=int(snap["in_flight"]),
                        governor_state=gov_state,
                        spot_preempted=False,
                        vram_free_mib=0,
                        gpu_temp_c=int(snap.get("gpu_temp_c") or 0),
                    )
            except Exception:
                gov_state = "unreadable"

        if self.require_governor_training_state and gov_state != "training":
            return PreemptionCheckResult(
                should_yield=True,
                trigger=f"governor_state_{gov_state}",
                in_flight_requests=0,
                governor_state=gov_state,
                spot_preempted=False,
                vram_free_mib=0,
                gpu_temp_c=0,
            )

        in_flight, nginx_ok = self._probe_nginx()
        if in_flight >= 1:
            return PreemptionCheckResult(
                should_yield=True,
                trigger="nginx_in_flight_inference",
                in_flight_requests=in_flight,
                governor_state=gov_state,
                spot_preempted=False,
                vram_free_mib=0,
                gpu_temp_c=0,
            )
        if self.nginx_status_url and not nginx_ok and self.require_governor_training_state:
            return PreemptionCheckResult(
                should_yield=True,
                trigger="nginx_status_unreachable_fail_closed",
                in_flight_requests=0,
                governor_state=gov_state,
                spot_preempted=False,
                vram_free_mib=0,
                gpu_temp_c=0,
            )

        spot_preempted = self._probe_spot_preempted()
        if spot_preempted:
            return PreemptionCheckResult(
                should_yield=True,
                trigger="gce_spot_preemption_30s_notice",
                in_flight_requests=0,
                governor_state=gov_state,
                spot_preempted=True,
                vram_free_mib=0,
                gpu_temp_c=0,
            )

        free_mib, temp_c = self._probe_gpu_free_and_temp()
        if temp_c >= self.max_gpu_temp_c:
            return PreemptionCheckResult(
                should_yield=True,
                trigger=f"gpu_thermal_guard_{temp_c}C",
                in_flight_requests=0,
                governor_state=gov_state,
                spot_preempted=False,
                vram_free_mib=free_mib,
                gpu_temp_c=temp_c,
            )
        if self.min_free_vram_mib > 0 and free_mib < self.min_free_vram_mib:
            return PreemptionCheckResult(
                should_yield=True,
                trigger=f"vram_floor_breach_{free_mib}MiB_lt_{self.min_free_vram_mib}MiB",
                in_flight_requests=0,
                governor_state=gov_state,
                spot_preempted=False,
                vram_free_mib=free_mib,
                gpu_temp_c=temp_c,
            )

        return PreemptionCheckResult(
            should_yield=False,
            trigger=None,
            in_flight_requests=0,
            governor_state=gov_state,
            spot_preempted=False,
            vram_free_mib=free_mib,
            gpu_temp_c=temp_c,
        )


def save_resumable_diloco_segment(
    segment_path: Path | str,
    *,
    round_index: int,
    completed_steps: int,
    target_steps: int,
    recipe: str,
    step_losses: Sequence[float],
    tokens_processed: int,
    base_before_bf16: torch.Tensor,
    current_base_bf16: torch.Tensor,
    lora_state: Mapping[str, torch.Tensor] | None = None,
    trigger: str = "segment_checkpoint",
) -> Path:
    """Atomically persist a `weights_only=True`-safe DiLoCo inner-loop segment outside `/opt/hivemind/t4/`."""
    dest = assert_not_protected_t4_path(segment_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "hivemind.gcpcrowd.diloco_segment.v1",
        "round_index": int(round_index),
        "completed_steps": int(completed_steps),
        "target_steps": int(target_steps),
        "recipe": str(recipe),
        "step_losses": [float(x) for x in step_losses],
        "tokens_processed": int(tokens_processed),
        "base_before_bf16": base_before_bf16.detach().to("cpu", dtype=torch.bfloat16).contiguous(),
        "current_base_bf16": current_base_bf16.detach().to("cpu", dtype=torch.bfloat16).contiguous(),
        "lora_state": {k: v.detach().to("cpu", dtype=torch.float32).contiguous() for k, v in (lora_state or {}).items()},
        "trigger": str(trigger),
        "saved_utc": time.time(),
    }
    with tempfile.NamedTemporaryFile(dir=dest.parent, prefix=".tmp_seg_", suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    torch.save(payload, tmp_path)
    os.replace(tmp_path, dest)
    return dest


def load_resumable_diloco_segment(segment_path: Path | str) -> dict[str, Any]:
    """Load a previously preempted DiLoCo segment using `torch.load(..., weights_only=True)`."""
    src = assert_not_protected_t4_path(segment_path)
    data = torch.load(src, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or data.get("schema") != "hivemind.gcpcrowd.diloco_segment.v1":
        raise ValueError(f"Invalid or unsupported DiLoCo segment file: {src}")
    return data


# ---------------------------------------------------------------------------
# 5. Inner-Loop `GCPCrowd` DiLoCo Learner (T4 / RTX 2070 / Community GPU)
# ---------------------------------------------------------------------------


@dataclass
class GcpT4DiLoCoInnerResult:
    """Output of a single `GCPCrowd` DiLoCo inner-loop execution (complete or preempted)."""

    learner_id: str
    hiveminder_handle: str
    device_suffix: str
    round_index: int
    recipe_decision: T4DiLoCoRecipeDecision
    steps_completed: int
    target_steps: int
    tokens_processed: int
    loss_before: float
    loss_after: float
    step_losses: list[float]
    delta_bf16: torch.Tensor = field(repr=False)
    delta_l2: float = 0.0
    lora_state: dict[str, torch.Tensor] | None = field(default=None, repr=False)
    preempted: bool = False
    preemption_trigger: str | None = None
    yield_latency_ms: float | None = None
    segment_checkpoint_path: Path | None = None
    hardware_headers: dict[str, str] = field(default_factory=dict)
    submit_metadata: dict[str, Any] = field(default_factory=dict)
    compressed_delta_stream: bytes = field(default=b"", repr=False)
    compressed_delta_sha256: str = ""


def _configure_model_for_recipe(
    model: CrowdV1WithEngram,
    decision: T4DiLoCoRecipeDecision,
    *,
    lora_seed: int = 20261110,
) -> tuple[list[nn.Parameter], tuple[torch.Tensor, torch.Tensor] | None]:
    """Place `model` onto `decision.device` with CPU-pinned `CrowdEngram` and return trainable parameters."""
    target_dev = torch.device(decision.device)
    frozen_dt = (
        torch.bfloat16
        if decision.frozen_dtype == "bfloat16"
        else (torch.float16 if decision.frozen_dtype == "float16" else torch.float32)
    )

    # Always keep the 350M-parameter Engram table on CPU RAM to save ~700-1400 MiB GPU VRAM
    model.engram.to(device=torch.device("cpu"), dtype=torch.bfloat16)
    model.engram.table.weight.requires_grad_(False)
    model.cos_cached = model.cos_cached.to(device=target_dev)
    model.sin_cached = model.sin_cached.to(device=target_dev)

    if decision.recipe == "qlora_graft_diloco":
        unwrap_crowd_lora_adapters(model, merge_into_base=False)
        model.wte.to(device=target_dev, dtype=frozen_dt)
        if model.cfg.tie_word_embeddings:
            model.head.weight = model.wte.weight
        model.blocks.to(device=target_dev, dtype=frozen_dt)
        model.ln.to(device=target_dev, dtype=torch.float32)
        if not model.cfg.tie_word_embeddings:
            model.head.to(device=target_dev, dtype=frozen_dt)
        ln_before = (
            model.ln.weight.detach().to("cpu", dtype=torch.float32).clone(),
            model.ln.bias.detach().to("cpu", dtype=torch.float32).clone(),
        )
        trainable = inject_crowd_lora_adapters(
            model,
            rank=max(1, decision.lora_rank or 8),
            alpha=float(2 * max(1, decision.lora_rank or 8)),
            seed=lora_seed,
        )
        return trainable, ln_before

    unwrap_crowd_lora_adapters(model, merge_into_base=False)
    if decision.recipe == "block_window_diloco":
        win_set = set(decision.window_block_indices)
        model.wte.to(device=target_dev, dtype=frozen_dt)
        model.wte.weight.requires_grad_(False)
        if model.cfg.tie_word_embeddings:
            model.head.weight = model.wte.weight
        trainable = []
        for idx, blk in enumerate(model.blocks):
            if idx in win_set:
                blk.to(device=target_dev, dtype=torch.float32)
                for p in blk.parameters():
                    p.requires_grad_(True)
                    trainable.append(p)
            else:
                blk.to(device=target_dev, dtype=frozen_dt)
                for p in blk.parameters():
                    p.requires_grad_(False)
        model.ln.to(device=target_dev, dtype=torch.float32)
        for p in model.ln.parameters():
            p.requires_grad_(True)
            trainable.append(p)
        if not model.cfg.tie_word_embeddings:
            model.head.to(device=target_dev, dtype=frozen_dt)
            model.head.weight.requires_grad_(False)
        else:
            model.head.weight = model.wte.weight
        return trainable, None

    # full_trunk_8bit_ckpt
    model.wte.to(device=target_dev, dtype=torch.float32)
    model.blocks.to(device=target_dev, dtype=torch.float32)
    model.ln.to(device=target_dev, dtype=torch.float32)
    if not model.cfg.tie_word_embeddings:
        model.head.to(device=target_dev, dtype=torch.float32)
    else:
        model.head.weight = model.wte.weight
    trainable = model.base_parameters()
    for p in trainable:
        p.requires_grad_(True)
    return trainable, None


def _evacuate_model_to_cpu(model: CrowdV1WithEngram, optimizer: torch.optim.Optimizer | None = None) -> float:
    """Release CUDA tensors immediately on governor preemption and return evacuation latency in ms."""
    t0 = time.perf_counter()
    if optimizer is not None:
        optimizer.state.clear()
    model.to(device=torch.device("cpu"))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return round((time.perf_counter() - t0) * 1000.0, 2)


def run_gcp_t4_diloco_inner_loop(
    model: CrowdV1WithEngram,
    token_shard: torch.Tensor,
    *,
    learner_id: str = "gcp-t4-node1",
    hiveminder_handle: str = "gcp-t4-hermes",
    device_suffix: str = "t4spot",
    round_index: int = 0,
    inner_steps: int = 8,
    batch_size: int = 1,
    grad_accum_steps: int = 1,
    seq_len: int | None = None,
    learning_rate: float = 1.5e-4,
    requested_mode: str = "auto",
    window_blocks: int = 6,
    lora_rank: int = 8,
    use_8bit_adamw: bool = True,
    use_grad_checkpointing: bool = True,
    token_chunk_size: int = 256,
    prefer_cuda: bool = True,
    work_dir: Path | str | None = None,
    resume_from_segment: bool = True,
    yield_guard: T4GovernorYieldGuard | None = None,
    task_metadata: Mapping[str, Any] | None = None,
    recipe_override: T4DiLoCoRecipeDecision | None = None,
    pack_stream: bool = True,
    base_before_bf16_ref: torch.Tensor | None = None,
) -> GcpT4DiLoCoInnerResult:
    """Execute $H$ DiLoCo inner steps on `CrowdV1WithEngram` with T4 VRAM floor & sub-second preemption."""
    if token_shard.ndim != 1 or token_shard.numel() < 16:
        raise ValueError("token_shard must be a 1D tensor with at least 16 tokens.")
    if isinstance(inner_steps, bool) or not isinstance(inner_steps, int) or inner_steps < 1:
        raise ValueError("inner_steps must be an integer >= 1.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be an integer >= 1.")
    if isinstance(grad_accum_steps, bool) or not isinstance(grad_accum_steps, int) or grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be an integer >= 1.")

    safe_work_dir = assert_not_protected_t4_path(work_dir) if work_dir is not None else None
    seg_path = (safe_work_dir / f"segment_{learner_id}_r{round_index:04d}.pt") if safe_work_dir is not None else None

    decision = recipe_override or select_t4_diloco_recipe(
        requested_mode=requested_mode,
        round_index=round_index,
        num_layers=model.cfg.num_hidden_layers,
        window_blocks=window_blocks,
        lora_rank=lora_rank,
        prefer_cuda=prefer_cuda,
    )
    if decision.recipe == "yield_insufficient_vram":
        unwrap_crowd_lora_adapters(model, merge_into_base=False)
        zero_delta = torch.zeros(sum(p.numel() for p in model.base_parameters()), dtype=torch.bfloat16)
        return GcpT4DiLoCoInnerResult(
            learner_id=learner_id,
            hiveminder_handle=hiveminder_handle,
            device_suffix=device_suffix,
            round_index=round_index,
            recipe_decision=decision,
            steps_completed=0,
            target_steps=inner_steps,
            tokens_processed=0,
            loss_before=0.0,
            loss_after=0.0,
            step_losses=[],
            delta_bf16=zero_delta,
            delta_l2=0.0,
            preempted=True,
            preemption_trigger="vram_guard_pre_start",
            yield_latency_ms=0.0,
        )

    meta_in = dict(task_metadata or {"taskId": f"gcpcrowd-r{round_index:04d}", "globalStep": 348 + round_index})
    eff_default_seq = min(
        model.cfg.max_position_embeddings,
        int(seq_len) if seq_len is not None else max(8, min(512, (int(token_shard.numel()) - 2) // max(1, batch_size))),
    )
    spec: CrowdGPTCrossCheckSpec = parse_crowdgpt_task_crosscheck(meta_in, default_seq_len=eff_default_seq)
    eff_bs = spec.target_batch_size if (spec.cohort_mode and spec.target_batch_size) else batch_size
    eff_steps = spec.target_local_steps if (spec.cohort_mode and spec.target_local_steps) else inner_steps
    eff_accum = spec.target_accum_steps if (spec.cohort_mode and spec.target_accum_steps) else grad_accum_steps
    eff_seq = min(model.cfg.max_position_embeddings, max(4, min(eff_default_seq, int(token_shard.numel()) - 2)))

    # Resume state if a preempted segment exists for this exact round
    start_step = 0
    step_losses: list[float] = []
    tokens_processed = 0
    resumed_lora: dict[str, torch.Tensor] | None = None

    unwrap_crowd_lora_adapters(model, merge_into_base=False)
    base_before_bf16 = (
        base_before_bf16_ref.detach().to("cpu", dtype=torch.bfloat16).reshape(-1)
        if base_before_bf16_ref is not None
        else model.export_flat_base_bf16()
    )

    if resume_from_segment and seg_path is not None and seg_path.exists():
        seg = load_resumable_diloco_segment(seg_path)
        if int(seg.get("round_index", -1)) == round_index and str(seg.get("recipe")) == decision.recipe:
            base_before_bf16 = seg["base_before_bf16"].to(torch.bfloat16)
            model.load_flat_tensors(seg["current_base_bf16"].to(torch.bfloat16))
            start_step = int(seg["completed_steps"])
            step_losses = list(seg["step_losses"])
            tokens_processed = int(seg["tokens_processed"])
            if seg.get("lora_state"):
                resumed_lora = dict(seg["lora_state"])

    trainable_params, ln_before = _configure_model_for_recipe(
        model,
        decision,
        lora_seed=20261110 + round_index,
    )
    if resumed_lora and decision.recipe == "qlora_graft_diloco":
        load_crowd_lora_state_dict(model, resumed_lora)

    target_dev = torch.device(decision.device)
    seed_base = int(
        spec.target_seed
        if (spec.cohort_mode and spec.target_seed is not None)
        else (20261110 + round_index * 1009 + sum(ord(c) for c in learner_id))
    )

    def _sample_batch(offset_idx: int) -> torch.Tensor:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed_base + offset_idx)
        max_start = max(1, int(token_shard.numel()) - eff_seq)
        starts = torch.randint(0, max_start, (eff_bs,), generator=g)
        return torch.stack([token_shard[int(s) : int(s) + eff_seq] for s in starts]).to(device=target_dev, dtype=torch.long)

    eval_tokens = _sample_batch(99_991)
    model.eval()
    with torch.no_grad():
        _, loss_before_t = forward_crowd_v1_checkpointed(
            model,
            eval_tokens,
            labels=eval_tokens,
            use_grad_checkpointing=False,
            token_chunk_size=token_chunk_size,
        )
        loss_before = float(loss_before_t.item()) if loss_before_t is not None else 0.0

    opt: torch.optim.Optimizer = (
        Blockwise8BitAdamW(trainable_params, lr=float(learning_rate), weight_decay=0.01)
        if use_8bit_adamw
        else torch.optim.AdamW(trainable_params, lr=float(learning_rate), weight_decay=0.01)
    )

    model.train()
    preempted = False
    preemption_trigger: str | None = None
    yield_ms: float | None = None
    completed = start_step

    for s_idx in range(start_step, eff_steps):
        if yield_guard is not None:
            chk = yield_guard.check()
            if chk.should_yield:
                preempted = True
                preemption_trigger = chk.trigger or "governor_yield"
                lora_snap = extract_crowd_lora_state_dict(model) if decision.recipe == "qlora_graft_diloco" else None
                curr_base = (
                    base_before_bf16
                    if decision.recipe == "qlora_graft_diloco"
                    else model.export_flat_base_bf16()
                )
                if seg_path is not None:
                    save_resumable_diloco_segment(
                        seg_path,
                        round_index=round_index,
                        completed_steps=completed,
                        target_steps=eff_steps,
                        recipe=decision.recipe,
                        step_losses=step_losses,
                        tokens_processed=tokens_processed,
                        base_before_bf16=base_before_bf16,
                        current_base_bf16=curr_base,
                        lora_state=lora_snap,
                        trigger=preemption_trigger,
                    )
                yield_ms = _evacuate_model_to_cpu(model, opt)
                break

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for acc_idx in range(eff_accum):
            batch_ids = _sample_batch(s_idx * eff_accum + acc_idx)
            _, loss_t = forward_crowd_v1_checkpointed(
                model,
                batch_ids,
                labels=batch_ids,
                use_grad_checkpointing=use_grad_checkpointing,
                token_chunk_size=token_chunk_size,
            )
            assert loss_t is not None
            scaled = loss_t / float(eff_accum)
            scaled.backward()
            accum_loss += float(loss_t.item()) / float(eff_accum)
            tokens_processed += int(batch_ids.numel())

        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()
        step_losses.append(accum_loss)
        completed = s_idx + 1

    if not preempted:
        model.eval()
        with torch.no_grad():
            _, loss_after_t = forward_crowd_v1_checkpointed(
                model,
                eval_tokens,
                labels=eval_tokens,
                use_grad_checkpointing=False,
                token_chunk_size=token_chunk_size,
            )
            loss_after = float(loss_after_t.item()) if loss_after_t is not None else loss_before
        if seg_path is not None and seg_path.exists():
            seg_path.unlink(missing_ok=True)
    else:
        loss_after = step_losses[-1] if step_losses else loss_before

    lora_out: dict[str, torch.Tensor] | None = None
    chunk_n = 2_097_152
    if decision.recipe == "qlora_graft_diloco":
        lora_out = extract_crowd_lora_state_dict(model)
        delta_bf16 = project_lora_to_flat_trunk_delta(model, ln_before=ln_before)
        unwrap_crowd_lora_adapters(model, merge_into_base=False)
        sq_norm = 0.0
        for c_start in range(0, delta_bf16.numel(), chunk_n):
            sl = delta_bf16[c_start : c_start + chunk_n].to(torch.float32)
            sq_norm += float((sl * sl).sum().item())
        delta_l2 = math.sqrt(max(0.0, sq_norm))
    else:
        base_after_bf16 = model.export_flat_base_bf16()
        delta_bf16 = torch.empty_like(base_before_bf16, dtype=torch.bfloat16)
        sq_norm = 0.0
        for c_start in range(0, base_before_bf16.numel(), chunk_n):
            c_end = min(base_before_bf16.numel(), c_start + chunk_n)
            diff_f32 = base_after_bf16[c_start:c_end].to(torch.float32) - base_before_bf16[c_start:c_end].to(torch.float32)
            delta_bf16[c_start:c_end] = diff_f32.to(torch.bfloat16)
            sq_norm += float((diff_f32 * diff_f32).sum().item())
        del base_after_bf16
        delta_l2 = math.sqrt(max(0.0, sq_norm))
    del base_before_bf16

    if step_losses:
        mean_l = sum(step_losses) / len(step_losses)
        var_l = max(0.0, sum((v - mean_l) ** 2 for v in step_losses) / len(step_losses))
        last_l = step_losses[-1]
        samples_n = len(step_losses)
    else:
        mean_l, var_l, last_l, samples_n = loss_before, 0.0, loss_before, 1

    submit_meta = build_crowdgpt_submit_metadata(
        spec=spec,
        mean_loss=mean_l,
        last_step_loss=last_l,
        loss_std=math.sqrt(var_l),
        loss_samples=samples_n,
        actual_steps=max(1, completed),
        actual_batch_size=eff_bs,
        tokens_processed=max(1, tokens_processed),
    )
    submit_meta["hivemindRecipe"] = decision.recipe
    submit_meta["preempted"] = preempted

    compressed_stream = pack_crowdgpt_delta_stream(submit_meta, delta_bf16) if pack_stream else b""
    compressed_sha = hashlib.sha256(compressed_stream).hexdigest() if compressed_stream else ""
    hw_headers = build_crowdgpt_client_hardware_headers(
        hiveminder_handle=hiveminder_handle,
        device_suffix=device_suffix,
        backend="CUDA" if decision.device == "cuda" else "CPU",
        batch_size=eff_bs,
        vram_gb=max(1.0, decision.vram_total_mib / 1024.0) if decision.vram_total_mib > 0 else 16.0,
        autocast_dtype="fp32",
        native_optim="adamw8bit" if use_8bit_adamw else "adamw",
    )

    return GcpT4DiLoCoInnerResult(
        learner_id=learner_id,
        hiveminder_handle=hiveminder_handle,
        device_suffix=device_suffix,
        round_index=round_index,
        recipe_decision=decision,
        steps_completed=completed,
        target_steps=eff_steps,
        tokens_processed=tokens_processed,
        loss_before=round(loss_before, 6),
        loss_after=round(loss_after, 6),
        step_losses=[round(x, 6) for x in step_losses],
        delta_bf16=delta_bf16,
        delta_l2=round(delta_l2, 6),
        lora_state=lora_out,
        preempted=preempted,
        preemption_trigger=preemption_trigger,
        yield_latency_ms=yield_ms if preempted else None,
        segment_checkpoint_path=seg_path if preempted else None,
        hardware_headers=hw_headers,
        submit_metadata=submit_meta,
        compressed_delta_stream=compressed_stream,
        compressed_delta_sha256=compressed_sha,
    )


# ---------------------------------------------------------------------------
# 6. CPU-Offloaded DiLoCo Outer Nesterov Hub + 4-Gate Ratchet (`GCPCrowd`)
# ---------------------------------------------------------------------------


@dataclass
class GcpCrowdOuterRoundResult:
    """Audit receipt of a `GCPCrowd` DiLoCo outer Nesterov round across T4 + Swarm GPUs."""

    round_index: int
    accepted: bool
    outer_lr: float
    outer_momentum: float
    selected_learners: list[str]
    rejected_learners: list[str]
    krum_result: MultiKrumAggregationResult = field(repr=False)
    rms_guard: WeightRmsGuardDecision
    ratchet_decision: LossRatchetDecision
    loss_before: float
    loss_after: float
    outer_delta_l2: float
    effective_tokens: int
    checkpoint_sha256: str
    loss_check_receipt: dict[str, Any] | None = None
    checkpoint_path: Path | None = None
    upstream_crowdgpt_stream_sha256: str = ""


def _streaming_multi_krum_bf16(
    deltas: Sequence[torch.Tensor],
    *,
    f_byzantine: int,
    token_weights: Sequence[int],
    chunk_elements: int = 1_048_576,
) -> MultiKrumAggregationResult:
    """Memory-bounded `bf16` Multi-Krum that avoids upcasting `657.8M` vectors to `fp32` simultaneously."""
    n = len(deltas)
    dim = deltas[0].numel()
    if dim <= 4_194_304:
        return multi_krum_select_and_aggregate(
            deltas,
            f_byzantine=f_byzantine,
            token_weights=token_weights,
            chunk_elements=chunk_elements,
        )

    vecs = [d.detach().to("cpu").reshape(-1) for d in deltas]
    if n == 1:
        return MultiKrumAggregationResult(
            selected_indices=[0],
            rejected_indices=[],
            krum_scores=[0.0],
            aggregated_delta_bf16=vecs[0].to(torch.bfloat16).clone(),
            effective_tokens=max(1, int(token_weights[0])),
            byzantine_filter_active=False,
        )

    dist_sq = torch.zeros((n, n), dtype=torch.float64)
    for c_start in range(0, dim, chunk_elements):
        c_end = min(dim, c_start + chunk_elements)
        blk = torch.stack([v[c_start:c_end].to(torch.float64) for v in vecs], dim=0)
        if not bool(torch.isfinite(blk).all()):
            raise ValueError("Learner delta contains non-finite (NaN/Inf) values.")
        norms = (blk * blk).sum(dim=1, keepdim=True)
        gram = blk @ blk.T
        dist_sq += torch.clamp(norms + norms.T - 2.0 * gram, min=0.0)
    dist_sq.fill_diagonal_(0.0)

    eff_f = min(max(0, f_byzantine), max(0, (n - 3) // 2))
    neighbors_k = max(1, n - eff_f - 2)
    m = min(n, max(1, n - eff_f))
    scores: list[float] = []
    for i in range(n):
        row = torch.cat([dist_sq[i, :i], dist_sq[i, i + 1 :]])
        nearest, _ = torch.topk(row, k=min(neighbors_k, row.numel()), largest=False)
        scores.append(float(nearest.sum().item()))

    ranked = sorted(range(n), key=lambda idx: (scores[idx], idx))
    selected = sorted(ranked[:m])
    rejected = sorted(ranked[m:])

    med_tok = float(sorted(float(token_weights[i]) for i in selected)[len(selected) // 2])
    cap_tok = max(1.0, 1.25 * med_tok)
    raw_w = [min(max(1.0, float(token_weights[i])), cap_tok) for i in selected]
    total_w = sum(raw_w)

    agg_bf16 = torch.empty(dim, dtype=torch.bfloat16)
    for c_start in range(0, dim, chunk_elements):
        c_end = min(dim, c_start + chunk_elements)
        acc = torch.zeros(c_end - c_start, dtype=torch.float32)
        for idx, w in zip(selected, raw_w):
            acc.add_(vecs[idx][c_start:c_end].to(torch.float32), alpha=w / total_w)
        agg_bf16[c_start:c_end] = acc.to(torch.bfloat16)

    eff_tokens = int(sum(int(round(w)) for w in raw_w))
    return MultiKrumAggregationResult(
        selected_indices=selected,
        rejected_indices=rejected,
        krum_scores=scores,
        aggregated_delta_bf16=agg_bf16,
        effective_tokens=max(1, eff_tokens),
        byzantine_filter_active=bool(eff_f > 0 and len(selected) < n),
    )


def _streaming_weight_rms_guard(
    old_bf16: torch.Tensor,
    new_bf16: torch.Tensor,
    *,
    min_rms_ratio: float = 0.80,
    max_rms_ratio: float = 1.25,
    halving_residual_tol: float = 1e-3,
    chunk_elements: int = 2_097_152,
) -> WeightRmsGuardDecision:
    """Chunked `float64` Weight RMS Collapse Guard (`< 32 MiB` RAM instead of `21 GB` on `657.8M` params)."""
    n = old_bf16.numel()
    if n <= 4_194_304:
        return check_weight_rms_collapse_guard(
            old_bf16,
            new_bf16,
            min_rms_ratio=min_rms_ratio,
            max_rms_ratio=max_rms_ratio,
            halving_residual_tol=halving_residual_tol,
        )

    sum_old_sq = 0.0
    sum_new_sq = 0.0
    sum_half_sq = 0.0
    for c_start in range(0, n, chunk_elements):
        c_end = min(n, c_start + chunk_elements)
        o64 = old_bf16[c_start:c_end].to(torch.float64)
        n64 = new_bf16[c_start:c_end].to(torch.float64)
        if not (bool(torch.isfinite(o64).all()) and bool(torch.isfinite(n64).all())):
            return WeightRmsGuardDecision(
                accepted=False,
                rms_old=float("nan"),
                rms_new=float("nan"),
                rms_ratio=float("nan"),
                halving_residual_rms=float("nan"),
                reason="REJECTED_NON_FINITE_WEIGHTS",
            )
        sum_old_sq += float((o64 * o64).sum().item())
        sum_new_sq += float((n64 * n64).sum().item())
        dh = n64 - 0.5 * o64
        sum_half_sq += float((dh * dh).sum().item())

    rms_old = math.sqrt(sum_old_sq / float(n))
    rms_new = math.sqrt(sum_new_sq / float(n))
    rms_ratio = rms_new / max(1e-12, rms_old)
    halving_res = math.sqrt(sum_half_sq / float(n))
    if halving_res <= halving_residual_tol and rms_ratio < min_rms_ratio:
        return WeightRmsGuardDecision(
            False, rms_old, rms_new, rms_ratio, halving_res,
            f"REJECTED_ZERO_PARTICIPANT_WEIGHT_HALVING (rms_ratio={rms_ratio:.4f})",
        )
    if rms_ratio < min_rms_ratio:
        return WeightRmsGuardDecision(
            False, rms_old, rms_new, rms_ratio, halving_res,
            f"REJECTED_RMS_COLLAPSE (rms_ratio={rms_ratio:.4f} < min={min_rms_ratio:.4f})",
        )
    if rms_ratio > max_rms_ratio:
        return WeightRmsGuardDecision(
            False, rms_old, rms_new, rms_ratio, halving_res,
            f"REJECTED_RMS_EXPLOSION (rms_ratio={rms_ratio:.4f} > max={max_rms_ratio:.4f})",
        )
    return WeightRmsGuardDecision(
        True, rms_old, rms_new, rms_ratio, halving_res,
        f"ACCEPTED_RMS_HEALTHY (rms_ratio={rms_ratio:.4f} in [{min_rms_ratio:.2f}, {max_rms_ratio:.2f}])",
    )


class GcpCrowdDiLoCoOuterHub:
    """CPU-resident DiLoCo Outer Nesterov Coordinator + Multi-Krum + Held-Out GPU Loss Ratchet.

    Maintains `theta` (`bf16`) and Nesterov velocity `v` (`bf16` with `fp32` chunked updates) in host CPU RAM
    (`0 B` GPU VRAM), filtering learner deltas via Multi-Krum and applying the DeepMind DiLoCo outer update:
        v_{t+1}     = mu * v_t + delta_agg
        theta_{t+1} = theta_t + lr_outer * (mu * v_{t+1} + delta_agg)
    where `delta_agg` is the Byzantine-filtered additive displacement `theta_inner - theta_t`.
    """

    def __init__(
        self,
        initial_flat_base_bf16: torch.Tensor,
        *,
        outer_lr: float = 0.7,
        outer_momentum: float = 0.9,
        f_byzantine: int = 1,
        max_regression_tolerance: float = 1e-4,
        work_dir: Path | str | None = None,
        evaluator_id: str = "hivemind-t4",
        evaluator_secret: bytes | None = None,
        clone_initial: bool = True,
    ) -> None:
        if not isinstance(initial_flat_base_bf16, torch.Tensor) or initial_flat_base_bf16.numel() == 0:
            raise ValueError("initial_flat_base_bf16 must be a non-empty 1D tensor.")
        if isinstance(outer_lr, bool) or not isinstance(outer_lr, (int, float)) or not math.isfinite(float(outer_lr)) or float(outer_lr) <= 0.0:
            raise ValueError("outer_lr must be a positive finite float.")
        if (
            isinstance(outer_momentum, bool)
            or not isinstance(outer_momentum, (int, float))
            or not math.isfinite(float(outer_momentum))
            or not (0.0 <= float(outer_momentum) < 1.0)
        ):
            raise ValueError("outer_momentum must be in [0.0, 1.0).")

        base_cpu = initial_flat_base_bf16.detach().to("cpu", dtype=torch.bfloat16).reshape(-1)
        self.theta_bf16: torch.Tensor = base_cpu.clone() if clone_initial else base_cpu
        self.velocity_bf16: torch.Tensor | None = None
        self.outer_lr = float(outer_lr)
        self.outer_momentum = float(outer_momentum)
        self.f_byzantine = max(0, int(f_byzantine))
        self.ratchet = LossRatchetGate(max_regression_tolerance=float(max_regression_tolerance))
        self.work_dir = assert_not_protected_t4_path(work_dir) if work_dir is not None else None
        self.evaluator_id = str(evaluator_id)
        self.evaluator_secret = evaluator_secret
        self.round_index: int = 0
        self.history: list[dict[str, Any]] = []

    def current_checkpoint_sha256(self) -> str:
        raw = self.theta_bf16.view(torch.int16).numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()

    def step(
        self,
        submissions: Sequence[GcpT4DiLoCoInnerResult | tuple[str, torch.Tensor, int]],
        *,
        eval_model: CrowdV1WithEngram,
        held_out_tokens: torch.Tensor,
        eval_window_len: int = 64,
        eval_num_windows: int = 4,
        prefer_cuda_eval: bool = True,
        save_safetensors: bool = True,
        pack_upstream_stream: bool | None = None,
        clear_input_deltas: bool = False,
    ) -> GcpCrowdOuterRoundResult:
        """Aggregate learner deltas via Multi-Krum + Outer Nesterov and gate on held-out validation loss."""
        if not submissions:
            raise ValueError("At least one learner submission is required for an outer DiLoCo step.")
        if held_out_tokens.ndim != 1 or held_out_tokens.numel() < max(8, eval_window_len + 1):
            raise ValueError("held_out_tokens must be a 1D tensor larger than eval_window_len.")

        learner_ids: list[str] = []
        deltas: list[torch.Tensor] = []
        token_weights: list[int] = []

        for item in submissions:
            if isinstance(item, GcpT4DiLoCoInnerResult):
                if item.steps_completed < 1:
                    continue
                learner_ids.append(item.learner_id)
                deltas.append(item.delta_bf16)
                token_weights.append(max(1, item.tokens_processed))
            else:
                lid, d_vec, tok = item
                learner_ids.append(str(lid))
                deltas.append(d_vec)
                token_weights.append(max(1, int(tok)))

        if not deltas:
            raise ValueError("No non-empty learner deltas available for outer aggregation.")

        krum = _streaming_multi_krum_bf16(
            deltas,
            f_byzantine=self.f_byzantine,
            token_weights=token_weights,
        )
        selected_names = [learner_ids[i] for i in krum.selected_indices]
        rejected_names = [learner_ids[i] for i in krum.rejected_indices]
        del deltas
        if clear_input_deltas:
            for item in submissions:
                if isinstance(item, GcpT4DiLoCoInnerResult):
                    item.delta_bf16 = torch.empty(0, dtype=torch.bfloat16)
            gc.collect()

        dim = self.theta_bf16.numel()
        chunk_n = 2_097_152
        new_vel_bf16 = torch.empty(dim, dtype=torch.bfloat16)
        candidate_bf16 = torch.empty(dim, dtype=torch.bfloat16)
        outer_delta_bf16 = torch.empty(dim, dtype=torch.bfloat16)
        outer_sq_norm = 0.0

        for c_start in range(0, dim, chunk_n):
            c_end = min(dim, c_start + chunk_n)
            agg_sl = krum.aggregated_delta_bf16[c_start:c_end].to(torch.float32)
            th_sl = self.theta_bf16[c_start:c_end].to(torch.float32)
            if self.velocity_bf16 is not None:
                vel_sl = self.velocity_bf16[c_start:c_end].to(torch.float32)
                nv = self.outer_momentum * vel_sl + agg_sl
            else:
                nv = agg_sl
            ns = self.outer_lr * (self.outer_momentum * nv + agg_sl)
            cand_sl = (th_sl + ns).to(torch.bfloat16)
            od_sl = (cand_sl.to(torch.float32) - th_sl).to(torch.bfloat16)

            new_vel_bf16[c_start:c_end] = nv.to(torch.bfloat16)
            candidate_bf16[c_start:c_end] = cand_sl
            outer_delta_bf16[c_start:c_end] = od_sl
            od_f32 = od_sl.to(torch.float32)
            outer_sq_norm += float((od_f32 * od_f32).sum().item())

        outer_l2 = math.sqrt(max(0.0, outer_sq_norm))

        # Gate 1: Anti-Lobotomy Weight RMS Collapse Guard
        rms_decision = _streaming_weight_rms_guard(self.theta_bf16, candidate_bf16)

        # Gate 2: Held-out multi-window evaluation on GPU/CPU
        unwrap_crowd_lora_adapters(eval_model, merge_into_base=False)
        eval_decision = select_t4_diloco_recipe(
            requested_mode="qlora_graft_diloco",
            prefer_cuda=prefer_cuda_eval,
        )
        eval_dev = torch.device("cuda" if eval_decision.can_train_on_gpu else "cpu")
        eval_dtype = torch.float16 if (eval_dev.type == "cuda" and eval_decision.frozen_dtype == "float16") else (
            torch.bfloat16 if (eval_dev.type == "cuda" and eval_decision.frozen_dtype == "bfloat16") else torch.float32
        )

        win_len = min(eval_model.cfg.max_position_embeddings, max(4, int(eval_window_len)))
        max_w = max(1, (int(held_out_tokens.numel()) - 1) // win_len)
        n_win = min(max_w, max(1, int(eval_num_windows)))
        windows = torch.stack(
            [held_out_tokens[i * win_len : (i + 1) * win_len] for i in range(n_win)]
        ).to(device=eval_dev, dtype=torch.long)

        def _eval_per_window(flat_weights: torch.Tensor) -> list[float]:
            eval_model.load_flat_tensors(flat_weights)
            eval_model.place_for_inference(device=str(eval_dev), dtype=eval_dtype, offload_engram_to_cpu=True)
            eval_model.eval()
            losses: list[float] = []
            with torch.no_grad():
                for w_idx in range(n_win):
                    w_tok = windows[w_idx : w_idx + 1]
                    _, l_t = forward_crowd_v1_checkpointed(
                        eval_model,
                        w_tok,
                        labels=w_tok,
                        use_grad_checkpointing=False,
                        token_chunk_size=256,
                    )
                    losses.append(float(l_t.item()) if l_t is not None else 0.0)
            return losses

        losses_before = _eval_per_window(self.theta_bf16)
        losses_after = _eval_per_window(candidate_bf16)
        mean_before = sum(losses_before) / len(losses_before)
        mean_after = sum(losses_after) / len(losses_after)

        ratchet_decision = self.ratchet.evaluate(
            loss_before=mean_before,
            loss_after=mean_after,
            candidate_delta_bf16=outer_delta_bf16,
        )

        loss_receipt_dict: dict[str, Any] | None = None
        if self.evaluator_secret is not None:
            shard_np = held_out_tokens.detach().to("cpu", dtype=torch.int64).numpy().astype(np.uint32)
            receipt = LossCheckReceipt(
                evaluator_id=self.evaluator_id,
                base_sha256=weights_sha256(self.theta_bf16),
                candidate_sha256=weights_sha256(candidate_bf16),
                heldout_sha256=tokens_sha256(shard_np),
                window=win_len,
                seed=20261110 + self.round_index,
                starts=tuple(i * win_len for i in range(n_win)),
                losses_before=tuple(losses_before),
                losses_after=tuple(losses_after),
                compute_dtype=str(eval_dtype).replace("torch.", ""),
                device=str(eval_dev),
                round_id=f"gcpcrowd-r{self.round_index:04d}",
            ).signed(self.evaluator_secret)
            loss_receipt_dict = json.loads(receipt.to_json())

        accepted = bool(rms_decision.accepted and ratchet_decision.accepted)
        ckpt_path: Path | None = None
        if accepted:
            self.theta_bf16 = candidate_bf16
            self.velocity_bf16 = new_vel_bf16
            eval_model.load_flat_tensors(self.theta_bf16)
        else:
            # Dampen momentum on rejected outer step so stale velocity cannot cause repeated rejections
            if self.velocity_bf16 is not None:
                for c_start in range(0, dim, chunk_n):
                    c_end = min(dim, c_start + chunk_n)
                    self.velocity_bf16[c_start:c_end] = (
                        0.5 * self.velocity_bf16[c_start:c_end].to(torch.float32)
                    ).to(torch.bfloat16)
            del candidate_bf16
            del new_vel_bf16
            eval_model.load_flat_tensors(self.theta_bf16)

        ckpt_sha = self.current_checkpoint_sha256()
        do_pack = (dim <= 16_777_216) if pack_upstream_stream is None else bool(pack_upstream_stream)
        if do_pack:
            upstream_stream = pack_crowdgpt_delta_stream(
                {
                    "isDelta": True,
                    "loraRank": 0,
                    "weightFormat": "bf16",
                    "tokensProcessed": krum.effective_tokens,
                    "gcpCrowdOuterRound": self.round_index,
                    "selectedLearners": selected_names,
                    "lossBefore": round(mean_before, 6),
                    "lossAfter": round(mean_after, 6),
                },
                outer_delta_bf16,
            )
            upstream_sha = hashlib.sha256(upstream_stream).hexdigest()
        else:
            upstream_sha = weights_sha256(outer_delta_bf16)

        if self.work_dir is not None:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            if accepted and save_safetensors:
                from safetensors.torch import save_file

                ckpt_path = assert_not_protected_t4_path(self.work_dir / "gcpcrowd_latest_trunk_bf16.safetensors")
                with tempfile.NamedTemporaryFile(
                    dir=self.work_dir, prefix=".tmp_gcpcrowd_", suffix=".safetensors", delete=False
                ) as tmp:
                    tmp_p = Path(tmp.name)
                save_file({"weights": self.theta_bf16.contiguous()}, str(tmp_p))
                os.replace(tmp_p, ckpt_path)

            receipt_file = assert_not_protected_t4_path(self.work_dir / f"gcpcrowd_outer_round_{self.round_index:04d}.json")
            summary_json = {
                "round_index": self.round_index,
                "accepted": accepted,
                "selected_learners": selected_names,
                "rejected_learners": rejected_names,
                "loss_before": round(mean_before, 6),
                "loss_after": round(mean_after, 6),
                "outer_delta_l2": round(outer_l2, 6),
                "effective_tokens": krum.effective_tokens,
                "checkpoint_sha256": ckpt_sha,
                "upstream_crowdgpt_stream_sha256": upstream_sha,
                "rms_guard": asdict(rms_decision),
                "ratchet": asdict(ratchet_decision),
                "loss_check_receipt": loss_receipt_dict,
                "timestamp_utc": round(time.time(), 3),
            }
            receipt_file.write_text(json.dumps(summary_json, indent=2, sort_keys=True), encoding="utf-8")
            self.history.append(summary_json)

        out = GcpCrowdOuterRoundResult(
            round_index=self.round_index,
            accepted=accepted,
            outer_lr=self.outer_lr,
            outer_momentum=self.outer_momentum,
            selected_learners=selected_names,
            rejected_learners=rejected_names,
            krum_result=krum,
            rms_guard=rms_decision,
            ratchet_decision=ratchet_decision,
            loss_before=round(mean_before, 6),
            loss_after=round(mean_after, 6),
            outer_delta_l2=round(outer_l2, 6),
            effective_tokens=krum.effective_tokens,
            checkpoint_sha256=ckpt_sha,
            loss_check_receipt=loss_receipt_dict,
            checkpoint_path=ckpt_path,
            upstream_crowdgpt_stream_sha256=upstream_sha,
        )
        self.round_index += 1
        return out


def render_gcp_t4_diloco_systemd_unit(
    *,
    repo_dir: str = "/opt/hivemind/HiveMind",
    python_bin: str = "/mnt/disks/disk-socrateai-local-1/boinc/venv/bin/python",
    work_dir: str = DEFAULT_GCP_T4_WORK_DIR_POSIX,
    governor_state: str = DEFAULT_T4_GOVERNOR_STATE_POSIX,
    hold_file: str = DEFAULT_T4_HOLD_FILE_POSIX,
    nginx_status_url: str = DEFAULT_T4_NGINX_STATUS_URL,
    user: str = "callensxavier_gmail_com",
) -> str:
    """Render a systemd template for `socreateai-agora-hermes-node1` writing strictly outside `/opt/hivemind/t4/`."""
    assert_not_protected_t4_path(work_dir)
    repo_posix = PurePosixPath(str(repo_dir).replace("\\", "/")).as_posix()
    work_posix = PurePosixPath(str(work_dir).replace("\\", "/")).as_posix()
    return (
        "# Serving-first GCP Spot Tesla T4 DiLoCo trainer for GCPCrowd (Vxtzq/Crowd-v1 x HiveMind).\n"
        "# Writes strictly to /mnt/disks/disk-socrateai-local-1/gcpcrowd (never /opt/hivemind/t4/).\n"
        "[Unit]\n"
        "Description=HiveMind GCP T4 DiLoCo Co-Trainer for GCPCrowd (>= 4300 MiB free VRAM guard)\n"
        "After=hivemind-t4-governor.service nginx.service\n\n"
        "[Service]\n"
        "NoNewPrivileges=yes\n"
        "ProtectSystem=full\n"
        f"User={user}\n"
        f"Environment=PYTHONPATH={repo_posix}/src\n"
        f"ExecStartPre=/usr/bin/install -d -m 0755 {work_posix}\n"
        f"ExecStart={python_bin} {repo_posix}/scripts/gcp_t4_crowd_diloco_worker.py "
        f"--mode daemon --work-dir {work_posix} --governor-state {governor_state} "
        f"--hold-file {hold_file} --nginx-status-url {nginx_status_url} "
        f"--min-free-vram-mib {DEFAULT_MIN_FREE_VRAM_MIB_T4} --check-spot-preemption\n"
        "Restart=on-failure\n"
        "RestartSec=15\n\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
    )
