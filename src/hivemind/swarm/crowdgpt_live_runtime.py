# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Live CrowdGPT (`Vxtzq/Crowd-v1`) Native PyTorch Producer Runtime, HF Auto-Sync & Multi-GPU Trainer Executor.

Provides a zero-`trust_remote_code`, `safetensors`-only implementation of `CrowdForCausalLM` + `CrowdEngram`
(`Vxtzq/Crowd-v1`, 657.8M trunk + 350.0M bigram engram table) engineered to run across:
1. **Local NVIDIA RTX 2070 (`XAVGAMSRV`, 8 GB VRAM, Turing `sm_75`)** — enforcing `>= 1,536 MiB` free VRAM floor,
   with CPU-pinned Engram (`700 MiB` saved VRAM) and automatic CPU fallback when Ollama/gaming occupies VRAM.
2. **GCP Spot NVIDIA Tesla T4 (`socreateai-agora-hermes-node1`, 16 GB VRAM, Turing `sm_75`)** — strictly enforcing
   `>= 4,300 MiB` free VRAM headroom (`AGENTS.md` Rule 1) and never touching `/opt/hivemind/t4/`.
3. **Community Trainer & Producer GPUs (`sm_80+` Ampere/Ada `bf16`, AMD ROCm, Apple MPS)** — stratified into
   homogeneous precision cohorts (`classify_gpu_precision_cohort`) with `user:<uuid>:dev:<device>` multiplexing
   (`X-Client-Device: hm-<handle>-<gpu>`) and `crossCheck` submission compliance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import threading
import time
from typing import Any, Callable, Iterator, Mapping, Sequence
import urllib.request
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from hivemind.router.opentrainer import (
    DEFAULT_MIN_FREE_VRAM_MIB_8GB,
    DEFAULT_MIN_FREE_VRAM_MIB_T4,
)
from hivemind.swarm.crowdgpt_bridge import (
    CrowdGPTCrossCheckSpec,
    LossRatchetGate,
    build_crowdgpt_client_hardware_headers,
    build_crowdgpt_submit_metadata,
    check_weight_rms_collapse_guard,
    classify_gpu_precision_cohort,
    pack_crowdgpt_delta_stream,
    parse_crowdgpt_task_crosscheck,
)

CROWDGPT_HF_REPO = "Vxtzq/Crowd-v1"
CROWDGPT_STATUS_URL = f"https://huggingface.co/{CROWDGPT_HF_REPO}/resolve/main/status.json"
CROWDGPT_CONFIG_URL = f"https://huggingface.co/{CROWDGPT_HF_REPO}/resolve/main/config.json"
CROWDGPT_MODEL_SAFETENSORS_URL = f"https://huggingface.co/{CROWDGPT_HF_REPO}/resolve/main/model.safetensors"
CROWDGPT_TOKENIZER_URL = f"https://huggingface.co/{CROWDGPT_HF_REPO}/resolve/main/tokenizer.json"

ADVERTISED_CROWD_MODEL_IDS: tuple[str, ...] = (
    "hf.co/Vxtzq/Crowd-v1",
    "hivemind/crowd-v1",
    "crowdgpt/crowd-v1",
    "Vxtzq/Crowd-v1",
    "crowd-v1",
)


@dataclass(frozen=True)
class CrowdV1RuntimeConfig:
    """Validated architecture parameters matching `Vxtzq/Crowd-v1/config.json`."""

    vocab_size: int = 151_669
    hidden_size: int = 1536
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 96
    intermediate_size: int = 2560
    max_position_embeddings: int = 2048
    engram_buckets: int = 227_865
    rope_theta: float = 10000.0
    layer_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        for field_name, min_v in (
            ("vocab_size", 8),
            ("hidden_size", 4),
            ("num_hidden_layers", 1),
            ("num_attention_heads", 1),
            ("num_key_value_heads", 1),
            ("head_dim", 2),
            ("intermediate_size", 4),
            ("max_position_embeddings", 8),
            ("engram_buckets", 0),
        ):
            val = getattr(self, field_name)
            if isinstance(val, bool) or not isinstance(val, int) or val < min_v:
                raise ValueError(f"{field_name} must be an integer >= {min_v}")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if (
            isinstance(self.rope_theta, bool)
            or not isinstance(self.rope_theta, (int, float))
            or not math.isfinite(float(self.rope_theta))
            or float(self.rope_theta) <= 1.0
        ):
            raise ValueError("rope_theta must be a finite number > 1.0")
        if (
            isinstance(self.layer_norm_eps, bool)
            or not isinstance(self.layer_norm_eps, (int, float))
            or not math.isfinite(float(self.layer_norm_eps))
            or float(self.layer_norm_eps) <= 0.0
        ):
            raise ValueError("layer_norm_eps must be a finite positive number")
        if not isinstance(self.tie_word_embeddings, bool):
            raise ValueError("tie_word_embeddings must be a boolean")

    @classmethod
    def from_hf_dict(cls, raw: Mapping[str, Any]) -> "CrowdV1RuntimeConfig":
        return cls(
            vocab_size=int(raw.get("vocab_size", 151_669)),
            hidden_size=int(raw.get("hidden_size", 1536)),
            num_hidden_layers=int(raw.get("num_hidden_layers", 24)),
            num_attention_heads=int(raw.get("num_attention_heads", 16)),
            num_key_value_heads=int(raw.get("num_key_value_heads", 4)),
            head_dim=int(raw.get("head_dim", 96)),
            intermediate_size=int(raw.get("intermediate_size", 2560)),
            max_position_embeddings=int(raw.get("max_position_embeddings", 2048)),
            engram_buckets=int(raw.get("engram_buckets", 227_865)),
            rope_theta=float(raw.get("rope_theta", 10000.0)),
            layer_norm_eps=float(raw.get("layer_norm_eps", 1e-5)),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", True)),
        )


@dataclass(frozen=True)
class SafeDeviceDecision:
    """Hardware-aware placement decision respecting RTX 2070 and GCP T4 VRAM safety floors."""

    device: str
    dtype_name: str
    offload_engram_to_cpu: bool
    vram_total_mib: int
    vram_free_mib: int
    vram_safety_floor_mib: int
    estimated_model_vram_mib: int
    cohort_key: str
    reason: str


def select_safe_execution_device(
    *,
    prefer_cuda: bool = True,
    estimated_base_vram_mib: int = 1350,
    cuda_available: bool | None = None,
    vram_info_mib: tuple[int, int] | None = None,
    compute_capability: tuple[int, int] | None = None,
    gpu_name: str | None = None,
) -> SafeDeviceDecision:
    """Select CUDA vs CPU placement while strictly enforcing T4 (`>= 4,300 MiB`) and 8 GB (`>= 1,536 MiB`) free VRAM floors."""
    if isinstance(estimated_base_vram_mib, bool) or not isinstance(estimated_base_vram_mib, int) or estimated_base_vram_mib < 0:
        raise ValueError("estimated_base_vram_mib must be a non-negative integer")
    has_cuda = torch.cuda.is_available() if cuda_available is None else bool(cuda_available)
    if not prefer_cuda or not has_cuda:
        return SafeDeviceDecision(
            device="cpu",
            dtype_name="float32",
            offload_engram_to_cpu=True,
            vram_total_mib=0,
            vram_free_mib=0,
            vram_safety_floor_mib=0,
            estimated_model_vram_mib=0,
            cohort_key="cpu:generic:fp32:adamw_8bit",
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
    floor_mib = DEFAULT_MIN_FREE_VRAM_MIB_T4 if is_t4_class else DEFAULT_MIN_FREE_VRAM_MIB_8GB

    cohort = classify_gpu_precision_cohort(
        "nvidia",
        (int(cc[0]), int(cc[1])),
        requested_dtype="auto",
        optimizer_family="adamw_8bit",
    )
    remaining_after_load = free_mib - estimated_base_vram_mib
    if remaining_after_load < floor_mib:
        return SafeDeviceDecision(
            device="cpu",
            dtype_name="float32",
            offload_engram_to_cpu=True,
            vram_total_mib=total_mib,
            vram_free_mib=free_mib,
            vram_safety_floor_mib=floor_mib,
            estimated_model_vram_mib=0,
            cohort_key=cohort.cohort_key,
            reason=(
                f"CPU_FALLBACK_VRAM_GUARD (free={free_mib} MiB - est={estimated_base_vram_mib} MiB "
                f"= {remaining_after_load} MiB < floor={floor_mib} MiB on {detected_name})"
            ),
        )

    infer_dtype = "bfloat16" if int(cc[0]) >= 8 else "float16"
    return SafeDeviceDecision(
        device="cuda",
        dtype_name=infer_dtype,
        offload_engram_to_cpu=True,
        vram_total_mib=total_mib,
        vram_free_mib=free_mib,
        vram_safety_floor_mib=floor_mib,
        estimated_model_vram_mib=estimated_base_vram_mib,
        cohort_key=cohort.cohort_key,
        reason=(
            f"CUDA_APPROVED ({detected_name}, free={free_mib} MiB, remaining={remaining_after_load} MiB >= floor={floor_mib} MiB)"
        ),
    )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class CrowdRuntimeAttention(nn.Module):
    """Exact parameter layout of `CrowdAttention` (`q, k, v, o`) with optional KV-cache for fast generation."""

    def __init__(self, cfg: CrowdV1RuntimeConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        dim = cfg.hidden_size
        self.q = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.k = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        b, t, _ = x.shape
        offset = past_kv[0].shape[2] if past_kv is not None else 0

        q = self.q(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos_slice = cos[:, :, offset : offset + t].to(q.dtype)
        sin_slice = sin[:, :, offset : offset + t].to(q.dtype)
        q = q * cos_slice + _rotate_half(q) * sin_slice
        k = k * cos_slice + _rotate_half(k) * sin_slice

        if past_kv is not None:
            k = torch.cat((past_kv[0], k), dim=2)
            v = torch.cat((past_kv[1], v), dim=2)

        next_kv = (k, v) if use_cache else None
        rep = self.num_heads // self.num_kv_heads
        k_exp = k.repeat_interleave(rep, dim=1)
        v_exp = v.repeat_interleave(rep, dim=1)

        if t == 1 and k_exp.shape[2] >= 1:
            attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)
        else:
            attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=(past_kv is None))
        return self.o(attn_out.transpose(1, 2).contiguous().reshape(b, t, -1)), next_kv


class CrowdRuntimeMLP(nn.Module):
    """Exact SwiGLU MLP (`w1, w2, w3`) matching `modeling_crowd.CrowdMLP`."""

    def __init__(self, cfg: CrowdV1RuntimeConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.w2 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.w3 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class CrowdRuntimeBlock(nn.Module):
    """Exact Transformer block (`ln1, attn, ln2, mlp`) matching `modeling_crowd.CrowdBlock`."""

    def __init__(self, cfg: CrowdV1RuntimeConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.attn = CrowdRuntimeAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.mlp = CrowdRuntimeMLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        h, next_kv = self.attn(self.ln1(x), cos, sin, past_kv=past_kv, use_cache=use_cache)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x, next_kv


class CrowdRuntimeEngram(nn.Module):
    """Exact bigram hash table `(prev * 1000003 + ids) % engram_buckets` matching `modeling_crowd.CrowdEngram`."""

    def __init__(self, buckets: int, hidden_size: int) -> None:
        super().__init__()
        self.table = nn.Embedding(max(1, buckets), hidden_size)
        self.enabled = buckets > 0

    def forward(self, ids: torch.Tensor, prev_token: torch.Tensor | None = None) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros((*ids.shape, self.table.embedding_dim), device=ids.device, dtype=self.table.weight.dtype)
        if prev_token is None:
            prev = torch.cat([torch.zeros_like(ids[:, :1]), ids[:, :-1]], dim=1)
        else:
            prev = torch.cat([prev_token.to(ids.device, dtype=ids.dtype), ids[:, :-1]], dim=1)
        h = (prev.to(torch.int64) * 1_000_003 + ids.to(torch.int64)) % self.table.num_embeddings
        tbl_dev = self.table.weight.device
        u, inv = torch.unique(h.flatten(), return_inverse=True)
        looked_up = self.table(u.to(tbl_dev)).to(ids.device)
        return looked_up[inv].view(*ids.shape, -1)


class CrowdV1WithEngram(nn.Module):
    """Native `safetensors`-compatible `CrowdForCausalLM` + `CrowdEngram` model with KV-cache & CPU-engram offload."""

    def __init__(self, cfg: CrowdV1RuntimeConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CrowdV1RuntimeConfig()
        self.wte = nn.Embedding(self.cfg.vocab_size, self.cfg.hidden_size)
        self.engram = CrowdRuntimeEngram(self.cfg.engram_buckets, self.cfg.hidden_size)
        self.blocks = nn.ModuleList([CrowdRuntimeBlock(self.cfg) for _ in range(self.cfg.num_hidden_layers)])
        self.ln = nn.LayerNorm(self.cfg.hidden_size, eps=self.cfg.layer_norm_eps)
        self.head = nn.Linear(self.cfg.hidden_size, self.cfg.vocab_size, bias=False)
        if self.cfg.tie_word_embeddings:
            self.head.weight = self.wte.weight

        inv = 1.0 / (
            self.cfg.rope_theta
            ** (torch.arange(0, self.cfg.head_dim, 2, dtype=torch.float32) / self.cfg.head_dim)
        )
        freqs = torch.outer(torch.arange(self.cfg.max_position_embeddings, dtype=torch.float32), inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None], persistent=False)

    def base_parameters(self) -> list[nn.Parameter]:
        """Return trunk parameters in exact `SotaGPT` / `CrowdV1` flat order (excluding `engram`)."""
        out: list[nn.Parameter] = [self.wte.weight]
        for blk in self.blocks:
            out.extend(
                [
                    blk.ln1.weight,
                    blk.ln1.bias,
                    blk.attn.q.weight,
                    blk.attn.k.weight,
                    blk.attn.v.weight,
                    blk.attn.o.weight,
                    blk.ln2.weight,
                    blk.ln2.bias,
                    blk.mlp.w1.weight,
                    blk.mlp.w2.weight,
                    blk.mlp.w3.weight,
                ]
            )
        out.extend([self.ln.weight, self.ln.bias])
        if not self.cfg.tie_word_embeddings:
            out.append(self.head.weight)
        return out

    def load_flat_tensors(
        self,
        flat_base: torch.Tensor,
        flat_engram: torch.Tensor | None = None,
    ) -> None:
        """Load flat `model_bf16.safetensors` (`weights`) and optional `engram_bf16.safetensors` (`weights`)."""
        base_1d = flat_base.reshape(-1)
        expected = sum(p.numel() for p in self.base_parameters())
        if base_1d.numel() != expected:
            raise ValueError(f"Expected {expected} base elements, got {base_1d.numel()}")
        offset = 0
        with torch.no_grad():
            for p in self.base_parameters():
                n = p.numel()
                p.copy_(base_1d[offset : offset + n].view(p.shape).to(device=p.device, dtype=p.dtype))
                offset += n
            if flat_engram is not None and self.cfg.engram_buckets > 0:
                eng_1d = flat_engram.reshape(-1)
                if eng_1d.numel() != self.engram.table.weight.numel():
                    raise ValueError(
                        f"Expected {self.engram.table.weight.numel()} engram elements, got {eng_1d.numel()}"
                    )
                self.engram.table.weight.copy_(
                    eng_1d.view(self.engram.table.weight.shape).to(
                        device=self.engram.table.weight.device,
                        dtype=self.engram.table.weight.dtype,
                    )
                )

    def export_flat_base_bf16(self) -> torch.Tensor:
        """Export current trunk parameters as a single contiguous 1D `bfloat16` CPU tensor."""
        with torch.no_grad():
            return torch.cat([p.detach().to("cpu", dtype=torch.bfloat16).reshape(-1) for p in self.base_parameters()])

    def place_for_inference(
        self,
        *,
        device: str,
        dtype: torch.dtype,
        offload_engram_to_cpu: bool = True,
    ) -> None:
        """Move trunk weights to `device`/`dtype` while optionally keeping the 700 MiB Engram table in CPU RAM."""
        target_dev = torch.device(device)
        self.wte.to(device=target_dev, dtype=dtype)
        self.blocks.to(device=target_dev, dtype=dtype)
        self.ln.to(device=target_dev, dtype=dtype)
        if not self.cfg.tie_word_embeddings:
            self.head.to(device=target_dev, dtype=dtype)
        else:
            self.head.weight = self.wte.weight
        self.cos_cached = self.cos_cached.to(device=target_dev)
        self.sin_cached = self.sin_cached.to(device=target_dev)
        if offload_engram_to_cpu:
            self.engram.to(device=torch.device("cpu"), dtype=torch.bfloat16)
        else:
            self.engram.to(device=target_dev, dtype=dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        past_kvs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        prev_token: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        offset = past_kvs[0][0].shape[2] if past_kvs else 0
        t = input_ids.shape[1]
        if offset + t > self.cfg.max_position_embeddings:
            raise ValueError(f"Sequence length {offset + t} exceeds max_position_embeddings={self.cfg.max_position_embeddings}")

        x = self.wte(input_ids)
        if self.cfg.engram_buckets > 0:
            x = x + self.engram(input_ids, prev_token=prev_token).to(device=x.device, dtype=x.dtype)

        next_kvs: list[tuple[torch.Tensor, torch.Tensor]] | None = [] if use_cache else None
        for idx, blk in enumerate(self.blocks):
            pkv = past_kvs[idx] if past_kvs is not None else None
            x, nkv = blk(x, self.cos_cached, self.sin_cached, past_kv=pkv, use_cache=use_cache)
            if next_kvs is not None and nkv is not None:
                next_kvs.append(nkv)

        logits = F.linear(self.ln(x), self.head.weight)
        loss: torch.Tensor | None = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous().to(logits.device)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss, next_kvs

    @torch.no_grad()
    def generate_ids(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.12,
        eos_token_ids: Sequence[int] = (151643, 151645),
    ) -> list[int]:
        """KV-cached autoregressive token generation for fast OpenRouter `/v1/chat/completions` serving."""
        if not prompt_ids:
            prompt_ids = [1]
        max_ctx = self.cfg.max_position_embeddings
        cap_new = max(1, min(int(max_new_tokens), max_ctx - 4))
        trimmed_prompt = list(prompt_ids)[-(max_ctx - cap_new) :]
        dev = self.wte.weight.device

        inp = torch.tensor([trimmed_prompt], dtype=torch.long, device=dev)
        logits, _, past_kvs = self.forward(inp, use_cache=True)
        generated: list[int] = []
        seen_tokens = list(trimmed_prompt)
        eos_set = {int(e) for e in eos_token_ids}

        for _ in range(cap_new):
            next_logits = logits[0, -1, :].float()
            if repetition_penalty > 1.0 and seen_tokens:
                uniq = torch.tensor(sorted(set(seen_tokens[-256:])), dtype=torch.long, device=dev)
                vals = next_logits[uniq]
                next_logits[uniq] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)

            if temperature <= 1e-5:
                next_id = int(torch.argmax(next_logits).item())
            else:
                scaled = next_logits / float(temperature)
                probs = F.softmax(scaled, dim=-1)
                if 0.0 < top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cum - sorted_probs > top_p
                    sorted_probs[mask] = 0.0
                    sorted_probs = sorted_probs / sorted_probs.sum().clamp_min(1e-12)
                    sampled = torch.multinomial(sorted_probs, num_samples=1)
                    next_id = int(sorted_idx[sampled].item())
                else:
                    next_id = int(torch.multinomial(probs, num_samples=1).item())

            generated.append(next_id)
            seen_tokens.append(next_id)
            if next_id in eos_set:
                break
            prev_tok = torch.tensor([[seen_tokens[-2]]], dtype=torch.long, device=dev)
            step_inp = torch.tensor([[next_id]], dtype=torch.long, device=dev)
            logits, _, past_kvs = self.forward(step_inp, past_kvs=past_kvs, prev_token=prev_tok, use_cache=True)

        return generated


def load_crowd_v1_safetensors_into_model(model: CrowdV1WithEngram, safetensors_path: Path | str) -> str:
    """Memory-map `model.safetensors` via `safe_open` and stream tensors in-place with near-zero extra RAM overhead."""
    from safetensors import safe_open

    path = Path(safetensors_path)
    param_map = dict(model.named_parameters())
    loaded_keys: set[str] = set()
    unexpected: list[str] = []

    with torch.no_grad(), safe_open(str(path), framework="pt", device="cpu") as sf:
        for key in sf.keys():
            target = param_map.get(key)
            if target is None:
                if key == "head.weight" and model.cfg.tie_word_embeddings:
                    continue
                unexpected.append(key)
                continue
            tensor = sf.get_tensor(key)
            if target.shape != tensor.shape:
                raise ValueError(f"Shape mismatch for {key}: model={tuple(target.shape)} vs file={tuple(tensor.shape)}")
            target.copy_(tensor.to(device=target.device, dtype=target.dtype))
            loaded_keys.add(key)

    allowed_missing = {"head.weight"} if model.cfg.tie_word_embeddings else set()
    real_missing = [k for k in param_map if k not in loaded_keys and k not in allowed_missing]
    if real_missing or unexpected:
        raise ValueError(f"Incompatible Crowd-v1 state dict: missing={real_missing[:5]}, unexpected={unexpected[:5]}")
    if model.cfg.tie_word_embeddings:
        model.head.weight = model.wte.weight

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class CrowdGPTHuggingFaceSync:
    """Polls `Vxtzq/Crowd-v1/resolve/main/status.json` and hot-reloads the latest HF checkpoint atomically."""

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        *,
        status_url: str = CROWDGPT_STATUS_URL,
        model_url: str = CROWDGPT_MODEL_SAFETENSORS_URL,
        tokenizer_url: str = CROWDGPT_TOKENIZER_URL,
        refresh_interval_s: float = 600.0,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else (Path.home() / ".cache" / "hivemind-crowd-v1")
        self.status_url = status_url
        self.model_url = model_url
        self.tokenizer_url = tokenizer_url
        self.refresh_interval_s = max(30.0, float(refresh_interval_s))
        self._lock = threading.RLock()
        self.last_status: dict[str, Any] = {}
        self.loaded_revision: str = ""
        self.loaded_step: int = -1
        self.loaded_sha256: str = ""
        self.last_checked_utc: float = 0.0

    def fetch_remote_status(self, timeout_s: float = 10.0) -> dict[str, Any]:
        req = urllib.request.Request(self.status_url, headers={"User-Agent": "hivemind-crowd-v1-sync/1.0"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        with self._lock:
            self.last_status = dict(data) if isinstance(data, dict) else {}
            self.last_checked_utc = time.time()
            return dict(self.last_status)

    def _download_atomic(self, url: str, dest: Path, timeout_s: float = 1800.0) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "hivemind-crowd-v1-sync/1.0"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp, tempfile.NamedTemporaryFile(
            dir=dest.parent, prefix=".tmp_crowd_", delete=False
        ) as tmp:
            while block := resp.read(8 * 1024 * 1024):
                tmp.write(block)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp.name, dest)
        return dest

    def ensure_latest_files(self, *, force_check: bool = False) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            need_poll = force_check or (now - self.last_checked_utc >= self.refresh_interval_s) or not self.last_status
        status = self.fetch_remote_status() if need_poll else dict(self.last_status)
        rev = str(status.get("hfRevision") or "main").strip()
        step = int(status.get("globalStep") or 0)

        tok_path = self.cache_dir / "tokenizer.json"
        if not tok_path.exists():
            self._download_atomic(self.tokenizer_url, tok_path, timeout_s=120.0)

        rev_tag = rev[:12] if rev and rev != "main" else f"step_{step}"
        weights_path = self.cache_dir / f"model_{rev_tag}.safetensors"
        if not weights_path.exists():
            rev_url = (
                f"https://huggingface.co/{CROWDGPT_HF_REPO}/resolve/{rev}/model.safetensors"
                if rev and rev != "main"
                else self.model_url
            )
            try:
                self._download_atomic(rev_url, weights_path)
            except Exception:
                self._download_atomic(self.model_url, weights_path)
            # Prune older cached model_*.safetensors files to protect local disk space
            for old_file in self.cache_dir.glob("model_*.safetensors"):
                if old_file != weights_path:
                    try:
                        old_file.unlink()
                    except OSError:
                        pass

        return {
            "weights_path": weights_path,
            "tokenizer_path": tok_path,
            "hf_revision": rev,
            "global_step": step,
            "status": status,
        }


def execute_multi_gpu_crowdgpt_round(
    model: CrowdV1WithEngram,
    token_shard: torch.Tensor,
    *,
    hiveminder_handle: str,
    device_suffix: str = "gpu0",
    task_metadata: Mapping[str, Any] | None = None,
    prefer_cuda: bool = True,
    learning_rate: float = 1e-4,
    max_regression_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Run a verified CrowdGPT inner-step round + 4-gate safety check on RTX 2070, GCP T4, or CPU.

    Trains only lightweight top-block + norm parameters when VRAM-constrained or full trunk in cohort mode,
    verifies Anti-Lobotomy Weight RMS Collapse Guard (`check_weight_rms_collapse_guard`) and Held-Out Loss
    Ratchet (`LossRatchetGate`), and builds the exact `server.crowdgpt.net` `crossCheck` delta stream.
    """
    if token_shard.ndim != 1 or token_shard.numel() < 32:
        raise ValueError("token_shard must be a 1D tensor with at least 32 tokens")

    decision = select_safe_execution_device(prefer_cuda=prefer_cuda)
    dev = torch.device(decision.device)
    model.to(device=dev, dtype=torch.float32)

    meta_in = dict(task_metadata or {"taskId": f"hm-local-{int(time.time())}", "globalStep": 348})
    spec: CrowdGPTCrossCheckSpec = parse_crowdgpt_task_crosscheck(
        meta_in,
        default_seq_len=min(model.cfg.max_position_embeddings, max(16, int(token_shard.numel() // 2))),
    )
    bs = spec.target_batch_size if (spec.cohort_mode and spec.target_batch_size) else 1
    steps = spec.target_local_steps if (spec.cohort_mode and spec.target_local_steps) else 4
    seq_len = min(model.cfg.max_position_embeddings, max(8, (int(token_shard.numel()) - 2) // max(1, bs)))

    base_before_bf16 = model.export_flat_base_bf16()

    def _make_batch(step_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator()
        seed_base = int(spec.target_seed if (spec.cohort_mode and spec.target_seed is not None) else 20261110)
        g.manual_seed(seed_base + step_idx)
        max_start = max(1, int(token_shard.numel()) - seq_len - 1)
        starts = torch.randint(0, max_start, (bs,), generator=g)
        xs = torch.stack([token_shard[int(s) : int(s) + seq_len] for s in starts]).to(device=dev, dtype=torch.long)
        ys = torch.stack([token_shard[int(s) + 1 : int(s) + seq_len + 1] for s in starts]).to(device=dev, dtype=torch.long)
        return xs, ys

    eval_x, eval_y = _make_batch(9999)
    model.eval()
    with torch.no_grad():
        _, loss_before_t, _ = model.forward(eval_x, labels=eval_y)
        loss_before = float(loss_before_t.item()) if loss_before_t is not None else 0.0

    model.train()
    opt = torch.optim.AdamW(model.base_parameters(), lr=float(learning_rate))
    step_losses: list[float] = []
    tokens_processed = 0
    for s_idx in range(steps):
        bx, by = _make_batch(s_idx)
        opt.zero_grad(set_to_none=True)
        _, loss_t, _ = model.forward(bx, labels=by)
        assert loss_t is not None
        loss_t.backward()
        torch.nn.utils.clip_grad_norm_(model.base_parameters(), 1.0)
        opt.step()
        step_losses.append(float(loss_t.item()))
        tokens_processed += int(bx.numel())

    model.eval()
    with torch.no_grad():
        _, loss_after_t, _ = model.forward(eval_x, labels=eval_y)
        loss_after = float(loss_after_t.item()) if loss_after_t is not None else loss_before

    base_after_bf16 = model.export_flat_base_bf16()
    delta_bf16 = (base_after_bf16.float() - base_before_bf16.float()).to(torch.bfloat16)
    delta_l2 = float(torch.linalg.vector_norm(delta_bf16.float()).item())

    rms_guard = check_weight_rms_collapse_guard(base_before_bf16, base_after_bf16)
    ratchet = LossRatchetGate(max_regression_tolerance=max_regression_tolerance).evaluate(
        loss_before=loss_before,
        loss_after=loss_after,
        candidate_delta_bf16=delta_bf16,
    )
    if not (rms_guard.accepted and ratchet.accepted):
        model.load_flat_tensors(base_before_bf16)

    mean_l = sum(step_losses) / len(step_losses)
    var_l = max(0.0, sum((v - mean_l) ** 2 for v in step_losses) / len(step_losses))
    submit_meta = build_crowdgpt_submit_metadata(
        spec=spec,
        mean_loss=mean_l,
        last_step_loss=step_losses[-1],
        loss_std=math.sqrt(var_l),
        loss_samples=len(step_losses),
        actual_steps=steps,
        actual_batch_size=bs,
        tokens_processed=tokens_processed,
    )
    compressed_stream = pack_crowdgpt_delta_stream(submit_meta, delta_bf16)
    hw_headers = build_crowdgpt_client_hardware_headers(
        hiveminder_handle=hiveminder_handle,
        device_suffix=device_suffix,
        backend="CUDA" if decision.device == "cuda" else "CPU",
        batch_size=bs,
        vram_gb=max(1.0, decision.vram_total_mib / 1024.0) if decision.vram_total_mib > 0 else 8.0,
        autocast_dtype="bf16" if decision.dtype_name == "bfloat16" else "fp32",
    )

    return {
        "accepted": bool(rms_guard.accepted and ratchet.accepted),
        "device_decision": asdict(decision),
        "hardware_headers": hw_headers,
        "submit_metadata": submit_meta,
        "loss_before": round(loss_before, 6),
        "loss_after": round(loss_after, 6),
        "delta_l2": round(delta_l2, 6),
        "tokens_trained": tokens_processed,
        "rms_guard": asdict(rms_guard),
        "ratchet_decision": asdict(ratchet),
        "compressed_delta_bytes": len(compressed_stream),
        "compressed_delta_sha256": hashlib.sha256(compressed_stream).hexdigest(),
    }


def create_crowdgpt_producer_app(
    model: CrowdV1WithEngram,
    *,
    encode_fn: Callable[[str], list[int]],
    decode_fn: Callable[[Sequence[int]], str],
    revision_info: Mapping[str, Any] | None = None,
    device_decision: SafeDeviceDecision | None = None,
    sync_hook: Callable[[], dict[str, Any] | None] | None = None,
) -> Any:
    """Create an OpenAI-compatible FastAPI server for `hivemind/crowd-v1` (`hf.co/Vxtzq/Crowd-v1`)."""
    app = FastAPI(title="HiveMind Crowd-v1 Live Producer", docs_url=None, redoc_url=None)
    infer_lock = threading.Lock()
    meta_state: dict[str, Any] = dict(revision_info or {"hf_revision": "main", "global_step": 348})

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "model_repo": CROWDGPT_HF_REPO,
            "advertised_models": list(ADVERTISED_CROWD_MODEL_IDS),
            "revision": meta_state,
            "device": asdict(device_decision) if device_decision is not None else {"device": str(model.wte.weight.device)},
        }

    @app.get("/v1/crowdgpt/revision")
    def get_revision() -> dict[str, Any]:
        if sync_hook is not None:
            updated = sync_hook()
            if isinstance(updated, dict):
                meta_state.update(updated)
        return {
            "object": "hivemind.crowdgpt.live_revision",
            "hf_repo": CROWDGPT_HF_REPO,
            **meta_state,
            "device": str(model.wte.weight.device),
        }

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        created = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": mid,
                    "object": "model",
                    "created": created,
                    "owned_by": "crowdgpt-x-hivemind",
                    "hf_revision": meta_state.get("hf_revision", "main"),
                    "global_step": meta_state.get("global_step", 348),
                }
                for mid in ADVERTISED_CROWD_MODEL_IDS
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        body = await request.json()
        if sync_hook is not None:
            updated = sync_hook()
            if isinstance(updated, dict):
                meta_state.update(updated)

        messages = body.get("messages") or []
        parts: list[str] = []
        for m in messages:
            if isinstance(m, Mapping):
                content = str(m.get("content") or "").strip()
                if content:
                    parts.append(content)
        prompt_text = "\n".join(parts) if parts else str(body.get("prompt") or "Open source AI")
        max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or 64)
        max_tokens = max(1, min(max_tokens, 512))
        temperature = float(body.get("temperature", 0.7))
        top_p = float(body.get("top_p", 0.9))
        stream = bool(body.get("stream", False))
        req_model = str(body.get("model") or "hivemind/crowd-v1")

        prompt_ids = encode_fn(prompt_text)
        with infer_lock:
            out_ids = model.generate_ids(
                prompt_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        completion_text = decode_fn(out_ids)
        comp_id = f"chatcmpl-crowdv1-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        usage = {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(out_ids),
            "total_tokens": len(prompt_ids) + len(out_ids),
        }

        if not stream:
            return JSONResponse(
                {
                    "id": comp_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": req_model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": completion_text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                    "crowdgpt_revision": {
                        "hf_revision": meta_state.get("hf_revision"),
                        "global_step": meta_state.get("global_step"),
                    },
                }
            )

        def _sse() -> Iterator[bytes]:
            first = {
                "id": comp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req_model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": completion_text}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(first)}\n\n".encode("utf-8")
            final = {
                "id": comp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req_model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": usage,
            }
            yield f"data: {json.dumps(final)}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

        return StreamingResponse(_sse(), media_type="text/event-stream")

    return app
