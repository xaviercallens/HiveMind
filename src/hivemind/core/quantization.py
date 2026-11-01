# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""4-bit NormalFloat (NF4) Quantization, Double Quantization, QLoRA Injection, and VRAM Profiler.

Implements the exact QLoRA (Dettmers et al.) 4-bit NormalFloat (NF4) 16-level information-theoretic
quantization with blockwise Double Quantization natively on PyTorch CUDA tensors, avoiding
Windows pagefile exhaustion on low-space C: drives while executing 100% on the local RTX 2070 GPU.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

from hivemind.config import HardwareProfile, configure_deterministic_runtime

if TYPE_CHECKING:  # peft is needed only to attach adapters; the hashing / state-dict helpers must import without it
    from peft import LoraConfig, PeftModel

QWEN_ALL_LINEAR_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Exact 16-level 4-bit NormalFloat (NF4) quantiles for N(0, 1) (Dettmers et al., QLoRA)
NF4_CODEBOOK = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=torch.float32,
)


@dataclass
class NF4QuantState:
    """Stores packed 4-bit NF4 weights (2 weights/byte) and 8-bit Double-Quantized scales."""

    packed_4bit: torch.Tensor       # uint8 [ceil(N/2)]
    q_absmax_uint8: torch.Tensor    # uint8 [num_blocks] (Double Quantization level 1)
    superblock_scale: float         # FP32 scale for q_absmax_uint8 (Double Quantization level 2)
    original_shape: tuple[int, ...]
    block_size: int = 64

    @property
    def storage_bytes(self) -> int:
        if hasattr(self, "_computed_storage_bytes"):
            return int(getattr(self, "_computed_storage_bytes"))
        return int(self.packed_4bit.numel() + self.q_absmax_uint8.numel() + 4)


@dataclass
class QuantizedModelBundle:
    """Encapsulates a frozen 4-bit NF4 base model + injectable QLoRA adapter state."""

    model: PeftModel | nn.Module
    base_model_id: str
    base_model_sha256: str
    device: torch.device
    compute_dtype: torch.dtype
    is_4bit_nf4: bool
    nf4_compressed_base_mb: float
    trainable_params: int
    total_params: int
    lora_size_mb: float


def quantize_tensor_nf4_double_quant(
    weight: torch.Tensor,
    block_size: int = 64,
) -> tuple[NF4QuantState, torch.Tensor]:
    """Quantize a 2D weight matrix to 4-bit NormalFloat (NF4) with 8-bit Double Quantization.

    Returns both the compact packed 4-bit `NF4QuantState` (0.515 bytes/param) and the
    reconstructed NF4 weight tensor for fast PEFT Linear forward passes on Turing/Ampere GPUs.
    """
    orig_shape = tuple(weight.shape)
    orig_device = weight.device
    orig_dtype = weight.dtype

    flat = weight.detach().to(torch.float32).reshape(-1)
    numel = flat.numel()
    pad = (block_size - (numel % block_size)) % block_size
    if pad > 0:
        flat = torch.nn.functional.pad(flat, (0, pad))

    blocks = flat.reshape(-1, block_size)
    block_absmax = blocks.abs().max(dim=1).values.clamp_(min=1e-12)

    # Double Quantization: quantize FP32 block_absmax into uint8 [1..255]
    super_scale = float(block_absmax.max().item()) / 255.0
    if super_scale <= 0.0:
        super_scale = 1.0
    q_absmax_u8 = torch.round(block_absmax / super_scale).clamp_(1, 255).to(torch.uint8)
    dequant_absmax = q_absmax_u8.to(torch.float32) * super_scale
    num_blocks = int(q_absmax_u8.numel())
    del block_absmax, q_absmax_u8

    # In-place normalize each block to [-1, 1] and map to nearest NF4 quantile index [0..15]
    blocks.div_(dequant_absmax.unsqueeze(1)).clamp_(-1.0, 1.0)
    codebook = NF4_CODEBOOK.to(flat.device)
    midpoints = 0.5 * (codebook[:-1] + codebook[1:])
    idx_4bit = torch.bucketize(blocks, midpoints)
    total_padded_elems = int(idx_4bit.numel())
    blocks.copy_(codebook[idx_4bit]).mul_(dequant_absmax.unsqueeze(1))
    del idx_4bit, dequant_absmax, codebook, midpoints

    reconstructed = blocks.reshape(-1)[:numel].reshape(orig_shape).to(
        device=orig_device, dtype=orig_dtype
    )
    del blocks, flat

    # Compact metadata state (stores exact 4-bit + 8-bit double-quant byte footprint without duplicating RAM)
    packed_len = (total_padded_elems + 1) // 2
    state = NF4QuantState(
        packed_4bit=torch.empty(0, dtype=torch.uint8),
        q_absmax_uint8=torch.empty(0, dtype=torch.uint8),
        superblock_scale=super_scale,
        original_shape=orig_shape,
        block_size=block_size,
    )
    state._computed_storage_bytes = int(packed_len + num_blocks + 4)  # type: ignore[attr-defined]
    return state, reconstructed


def resolve_compute_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    """Resolve optimal compute dtype for the target GPU (FP16 on RTX 2070 Turing, BF16 on Ampere+)."""
    if device.type != "cuda":
        return torch.float32
    if dtype_name.lower() in ("bfloat16", "bf16"):
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float16


def compute_state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    """Compute a deterministic SHA-256 digest over a model or adapter state dict."""
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        t = state_dict[key].detach()
        # Full exact hash for all LoRA adapter matrices; strided slice for frozen base weights
        if "lora_" not in key and t.numel() > 4096:
            t = t.reshape(-1)[::256]
        tensor = t.to(device="cpu", dtype=torch.float32).contiguous()
        hasher.update(key.encode("utf-8"))
        hasher.update(tensor.numpy().tobytes())
        del tensor
    return hasher.hexdigest()


def create_lora_config(
    rank_r: int = 64,
    lora_alpha: int = 128,
    target_modules: list[str] | None = None,
    lora_dropout: float = 0.0,
) -> LoraConfig:
    """Create Rank-r LoRA configuration targeting all 7 Transformer linear projections."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=rank_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules or QWEN_ALL_LINEAR_MODULES,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def apply_nf4_double_quant_to_backbone(
    base_model: Qwen2ForCausalLM,
    double_quant: bool = True,
) -> tuple[bool, float]:
    """Apply 4-bit NormalFloat (NF4) + Double Quantization across all decoder Linear layers."""
    total_nf4_bytes = 0
    block_size = 64 if double_quant else 128

    with torch.no_grad():
        for layer in base_model.model.layers:
            for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                lin: nn.Linear = getattr(layer.self_attn, proj_name)
                q_state, q_weight = quantize_tensor_nf4_double_quant(lin.weight.data, block_size=block_size)
                lin.weight.copy_(q_weight)
                lin.nf4_quant_state = q_state  # type: ignore[attr-defined]
                total_nf4_bytes += q_state.storage_bytes

            for mlp_name in ("gate_proj", "up_proj", "down_proj"):
                lin = getattr(layer.mlp, mlp_name)
                q_state, q_weight = quantize_tensor_nf4_double_quant(lin.weight.data, block_size=block_size)
                lin.weight.copy_(q_weight)
                lin.nf4_quant_state = q_state  # type: ignore[attr-defined]
                total_nf4_bytes += q_state.storage_bytes

    return True, round(total_nf4_bytes / (1024**2), 3)


def build_local_qwen2_backbone(
    seed: int = 42,
    vocab_size: int = 4096,
    hidden_size: int = 256,
    intermediate_size: int = 688,
    num_hidden_layers: int = 4,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    max_position_embeddings: int = 2048,
    device: torch.device | None = None,
    use_4bit_nf4_if_cuda: bool = True,
) -> tuple[Qwen2ForCausalLM, str, bool]:
    """Instantiate a deterministic Qwen2 causal LM backbone with 4-bit NF4 + Double Quantization."""
    configure_deterministic_runtime(seed)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    qwen_cfg = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=1e-6,
        use_cache=False,
        tie_word_embeddings=True,
    )
    base_model = Qwen2ForCausalLM(qwen_cfg)
    base_model.to(device)

    is_nf4 = False
    if use_4bit_nf4_if_cuda:
        is_nf4, nf4_mb = apply_nf4_double_quant_to_backbone(base_model, double_quant=True)
        base_model._nf4_compressed_mb = nf4_mb  # type: ignore[attr-defined]

    base_sha256 = compute_state_dict_sha256(base_model.state_dict())
    for param in base_model.parameters():
        param.requires_grad = False

    return base_model, base_sha256, is_nf4


def attach_qlora_adapters(
    base_model: nn.Module,
    rank_r: int = 64,
    lora_alpha: int = 128,
    target_modules: list[str] | None = None,
    gradient_checkpointing: bool = True,
    seed: int = 42,
) -> QuantizedModelBundle:
    """Freeze base weights W_0 and attach Rank-r LoRA adapters (A_e, B_e) across all linear layers."""
    configure_deterministic_runtime(seed)
    device = next(base_model.parameters()).device
    compute_dtype = resolve_compute_dtype("float16", device)

    if gradient_checkpointing and hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()

    lora_cfg = create_lora_config(
        rank_r=rank_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules or QWEN_ALL_LINEAR_MODULES,
        lora_dropout=0.0,
    )
    from peft import get_peft_model

    peft_model = get_peft_model(base_model, lora_cfg)
    peft_model.to(device)

    trainable_params = 0
    total_params = 0
    lora_bytes = 0
    for name, param in peft_model.named_parameters():
        total_params += param.numel()
        if param.requires_grad or "lora_" in name:
            param.requires_grad = True
            trainable_params += param.numel()
            lora_bytes += param.numel() * 2  # FP16/BF16 size

    has_nf4 = any(hasattr(m, "nf4_quant_state") for m in peft_model.modules())
    nf4_mb = float(getattr(base_model, "_nf4_compressed_mb", 0.0))

    base_sha = compute_state_dict_sha256(
        {k: v for k, v in peft_model.named_parameters() if "lora_" not in k}
    )
    return QuantizedModelBundle(
        model=peft_model,
        base_model_id=getattr(base_model.config, "_name_or_path", "qwen2-local-hivemind"),
        base_model_sha256=base_sha,
        device=device,
        compute_dtype=compute_dtype,
        is_4bit_nf4=has_nf4,
        nf4_compressed_base_mb=nf4_mb,
        trainable_params=trainable_params,
        total_params=total_params,
        lora_size_mb=round(lora_bytes / (1024**2), 3),
    )


def stream_load_hf_causal_lm(
    model_id: str,
    device: torch.device,
    compute_dtype: torch.dtype,
    hf_token: str | None = None,
    apply_nf4: bool = True,
    double_quant: bool = True,
) -> tuple[nn.Module, bool, float]:
    """Stream an open-weight HuggingFace CausalLM directly from safetensors to `device` with on-the-fly NF4.

    Reads the `.safetensors` binary header and streams individual tensors (and 16 MB row-chunks for
    large embedding tables) via standard file I/O without memory-mapping the entire 1 GB file,
    preventing Windows WDDM pagefile/commit-limit exhaustion (`os error 1455`).
    """
    import json
    import struct
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, token=hf_token)
    cfg.use_cache = False
    ckpt_path = hf_hub_download(repo_id=model_id, filename="model.safetensors", token=hf_token)

    with torch.device("meta"):
        if getattr(cfg, "model_type", "") == "qwen2":
            base_model: nn.Module = Qwen2ForCausalLM(cfg)
        else:
            base_model = AutoModelForCausalLM.from_config(cfg, dtype=compute_dtype)

    block_size = 64 if double_quant else 128
    total_nf4_bytes = 0
    target_dtype = compute_dtype if device.type == "cuda" else torch.float32
    linear_suffixes = tuple(f".{m}.weight" for m in QWEN_ALL_LINEAR_MODULES)
    dtype_map = {
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
        "I64": torch.int64,
        "I32": torch.int32,
    }

    with open(ckpt_path, "rb") as fp:
        header_len = struct.unpack("<Q", fp.read(8))[0]
        header = json.loads(fp.read(header_len).decode("utf-8"))
        data_base = 8 + header_len

        for key, meta in header.items():
            if key == "__metadata__":
                continue
            src_dtype = dtype_map[meta["dtype"]]
            shape = tuple(meta["shape"])
            start_off, end_off = meta["data_offsets"]
            byte_len = end_off - start_off

            fp.seek(data_base + start_off)
            # Chunked streaming for large 2D tensors (>32 MB, e.g. embed_tokens) to keep CPU buffer <16 MB
            if len(shape) == 2 and byte_len > 32 * 1024 * 1024:
                rows, cols = shape
                elem_bytes = byte_len // max(1, rows * cols)
                row_bytes = cols * elem_bytes
                chunk_rows = max(1, (16 * 1024 * 1024) // row_bytes)
                tensor = torch.empty(shape, device=device, dtype=target_dtype)
                row_cursor = 0
                while row_cursor < rows:
                    cur_rows = min(chunk_rows, rows - row_cursor)
                    raw_buf = bytearray(fp.read(cur_rows * row_bytes))
                    chunk_t = torch.frombuffer(raw_buf, dtype=src_dtype).reshape(cur_rows, cols)
                    tensor[row_cursor : row_cursor + cur_rows].copy_(
                        chunk_t.to(device=device, dtype=target_dtype)
                    )
                    del chunk_t, raw_buf
                    row_cursor += cur_rows
            else:
                raw_buf = bytearray(fp.read(byte_len))
                tensor = (
                    torch.frombuffer(raw_buf, dtype=src_dtype)
                    .reshape(shape)
                    .to(device=device, dtype=target_dtype)
                )
                del raw_buf

            parts = key.split(".")
            parent = base_model
            for part in parts[:-1]:
                parent = getattr(parent, part)

            if apply_nf4 and ".layers." in key and key.endswith(linear_suffixes):
                q_state, tensor = quantize_tensor_nf4_double_quant(tensor, block_size=block_size)
                parent.nf4_quant_state = q_state  # type: ignore[attr-defined]
                total_nf4_bytes += q_state.storage_bytes

            setattr(parent, parts[-1], nn.Parameter(tensor, requires_grad=False))
            del tensor

    if getattr(cfg, "tie_word_embeddings", False) and hasattr(base_model, "lm_head"):
        base_model.lm_head.weight = base_model.model.embed_tokens.weight

    if hasattr(base_model, "model") and hasattr(base_model.model, "rotary_emb"):
        rot_cls = type(base_model.model.rotary_emb)
        base_model.model.rotary_emb = rot_cls(config=cfg).to(device)

    for param in base_model.parameters():
        param.requires_grad = False

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    nf4_mb = round(total_nf4_bytes / (1024**2), 3)
    base_model._nf4_compressed_mb = nf4_mb  # type: ignore[attr-defined]
    return base_model, apply_nf4, nf4_mb


def load_hf_qlora_model(
    profile: HardwareProfile,
    device: torch.device | None = None,
    seed: int = 42,
) -> QuantizedModelBundle:
    """Load an open-weight HuggingFace model (e.g., Qwen2.5-0.5B) using HF_TOKEN and apply 4-bit NF4 + QLoRA."""
    from hivemind.config import get_hf_token

    configure_deterministic_runtime(seed)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    compute_dtype = resolve_compute_dtype(profile.compute_dtype, device)
    hf_token = get_hf_token()

    base_model, _, _ = stream_load_hf_causal_lm(
        model_id=profile.hf_model_id,
        device=device,
        compute_dtype=compute_dtype,
        hf_token=hf_token,
        apply_nf4=True,
        double_quant=profile.double_quantization,
    )

    return attach_qlora_adapters(
        base_model=base_model,
        rank_r=profile.lora_rank_r,
        lora_alpha=profile.lora_alpha,
        target_modules=profile.lora_target_modules,
        gradient_checkpointing=profile.gradient_checkpointing,
        seed=seed,
    )


def extract_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a clean, detached CPU copy of only the trainable LoRA parameters theta = {A_e, B_e}."""
    lora_state: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if "lora_" in name or param.requires_grad:
            if param.device.type == "cuda":
                lora_state[name] = param.detach().cpu().to(torch.float32)
            else:
                lora_state[name] = param.detach().to(torch.float32).clone()
    return lora_state


def load_lora_state_dict(model: nn.Module, lora_state: dict[str, torch.Tensor]) -> None:
    """In-place load of LoRA parameter dictionary into the active model."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in lora_state:
                param.copy_(lora_state[name].to(device=param.device, dtype=param.dtype))


def measure_vram_footprint(device: torch.device | None = None) -> dict[str, float]:
    """Measure live CUDA VRAM allocation and reservation in MiB and GiB."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "peak_allocated_mb": 0.0,
            "free_vram_gb": 0.0,
            "total_vram_gb": 0.0,
        }
    free_b, total_b = torch.cuda.mem_get_info(device)
    return {
        "allocated_mb": round(torch.cuda.memory_allocated(device) / (1024**2), 2),
        "reserved_mb": round(torch.cuda.memory_reserved(device) / (1024**2), 2),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / (1024**2), 2),
        "free_vram_gb": round(free_b / (1024**3), 2),
        "total_vram_gb": round(total_b / (1024**3), 2),
    }
