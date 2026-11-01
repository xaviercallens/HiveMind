# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Hardware & Algorithmic Configuration for Project HiveMind.

Automatically configures local disk caches on D:\\xdev\\DecentralizedAI\\.cache to protect C: drive,
enforces deterministic CUDA execution (CUBLAS_WORKSPACE_CONFIG=:4096:8), and manages dual hardware
profiles for the local NVIDIA GeForce RTX 2070 (8 GB VRAM, Turing SM 7.5) and 12 GB reference nodes.

Season 4 (`HM-S4-01`) extends the runtime to **multi-vendor accelerators**: NVIDIA CUDA, AMD ROCm/HIP
(`gfx1100`/`gfx1030` Radeon RX 7000/6000) and Apple Silicon Metal Performance Shaders (`MPS`) with
unified memory, exposing a single `detect_gpu_backend()` contract for the rest of the code base.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = PROJECT_ROOT / ".cache"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"


def ensure_local_drive_caches() -> dict[str, str]:
    """Redirect all heavy caches (HuggingFace, PyTorch, TMP, uv) to D:\\xdev\\DecentralizedAI\\.cache."""
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
        load_dotenv(Path.home() / ".env", override=False)
    except Exception:
        _ = None

    cache_dirs = {
        "HF_HOME": str(CACHE_ROOT / "huggingface"),
        "HF_HUB_CACHE": str(CACHE_ROOT / "huggingface" / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(CACHE_ROOT / "huggingface" / "hub"),
        "TORCH_HOME": str(CACHE_ROOT / "torch"),
        "UV_CACHE_DIR": str(CACHE_ROOT / "uv"),
        "TMP": str(CACHE_ROOT / "tmp"),
        "TEMP": str(CACHE_ROOT / "tmp"),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    }
    for env_key, dir_path in cache_dirs.items():
        if env_key not in ("CUBLAS_WORKSPACE_CONFIG", "HF_HUB_DISABLE_SYMLINKS_WARNING"):
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            os.environ[env_key] = dir_path
        else:
            os.environ.setdefault(env_key, dir_path)
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    return cache_dirs


def get_hf_token() -> str | None:
    """Resolve Hugging Face token from environment variables (.env, process env, or Windows HKCU)."""
    env_keys = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_TOKEN")
    for key in env_keys:
        val = os.environ.get(key)
        if val and val.strip():
            token = val.strip()
            os.environ["HF_TOKEN"] = token
            return token

    # Also check Windows User/Machine Registry in case `setx HF_TOKEN ...` was run in another shell
    if os.name == "nt":
        import winreg

        for hive, subkey in (
            (winreg.HKEY_CURRENT_USER, r"Environment"),
            (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
        ):
            try:
                with winreg.OpenKey(hive, subkey) as reg_key:
                    for k in env_keys:
                        try:
                            val, _ = winreg.QueryValueEx(reg_key, k)
                            if isinstance(val, str) and val.strip():
                                token = val.strip()
                                os.environ["HF_TOKEN"] = token
                                return token
                        except FileNotFoundError:
                            continue
            except OSError:
                continue

    try:
        from huggingface_hub import get_token

        tok = get_token()
        if tok:
            os.environ["HF_TOKEN"] = tok
            return tok
    except Exception:
        _ = None
    return None


# Apply cache redirection and token resolution immediately on import
ensure_local_drive_caches()
get_hf_token()


@dataclass
class VRAMBudget:
    """VRAM Allocation Budget for a DiLoCo-QLoRA Worker Node."""

    frozen_base_nf4_gb: float
    lora_trainable_r64_gb: float
    paged_adamw_8bit_gb: float
    activations_kv_cache_gb: float
    cuda_context_overhead_gb: float

    @property
    def total_peak_vram_gb(self) -> float:
        return round(
            self.frozen_base_nf4_gb
            + self.lora_trainable_r64_gb
            + self.paged_adamw_8bit_gb
            + self.activations_kv_cache_gb
            + self.cuda_context_overhead_gb,
            2,
        )


@dataclass
class HardwareProfile:
    """Hardware and Model Profile for Project HiveMind."""

    name: str
    description: str
    target_vram_gb: float
    available_vram_budget_gb: float
    hf_model_id: str
    compute_dtype: str = "float16"
    quantization_4bit: str = "nf4"
    double_quantization: bool = True
    lora_rank_r: int = 64
    lora_alpha: int = 128
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    micro_batch_size: int = 1
    seq_len: int = 512
    gradient_checkpointing: bool = True
    sim_inner_steps_H: int = 100
    sim_spot_check_interval_s: int = 25
    sim_workers_per_branch: int = 5
    sim_byzantine_workers: int = 1
    vram_budget: VRAMBudget = field(
        default_factory=lambda: VRAMBudget(
            frozen_base_nf4_gb=0.35,
            lora_trainable_r64_gb=0.08,
            paged_adamw_8bit_gb=0.08,
            activations_kv_cache_gb=0.35,
            cuda_context_overhead_gb=0.60,
        )
    )


@dataclass
class HiveMindConfig:
    """Master Configuration for Project HiveMind."""

    codename: str = "Project HiveMind"
    deterministic_cuda: bool = True
    cublas_workspace_config: str = ":4096:8"
    spot_check_interval_steps: int = 50
    spot_check_tolerance_eps: float = 1e-4
    idle_threshold_seconds: float = 30.0
    emergency_checkpoint_timeout_sec: float = 1.5
    expert_domains: list[str] = field(
        default_factory=lambda: [
            "Code",
            "Mathematics",
            "Creative_Writing",
            "STEM",
            "Legal_Medical",
            "Multilingual_FR",
            "Reasoning",
            "General",
        ]
    )
    inner_steps_H: int = 500
    inner_lr: float = 2e-4
    inner_weight_decay: float = 0.01
    outer_lr_eta: float = 0.7
    outer_momentum_mu: float = 0.9
    compression_topk_ratio: float = 0.20
    compression_dtype: str = "fp8_e4m3fn"
    multikrum_max_byzantine_fraction: float = 0.25
    cosine_outlier_sigma: float = 2.0
    num_experts_E: int = 8
    router_top_k: int = 2
    router_aux_loss_coef: float = 0.01
    profiles: dict[str, HardwareProfile] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: Path | str | None = None) -> HiveMindConfig:
        if yaml_path is None:
            yaml_path = PROJECT_ROOT / "configs" / "rtx2070_dual_profile.yaml"
        path = Path(yaml_path)
        if not path.exists():
            return cls()

        with path.open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        sys_cfg = raw.get("system", {})
        diloco_cfg = raw.get("diloco", {})
        btm_cfg = raw.get("btm_moe", {})
        raw_profiles = raw.get("profiles", {})

        profiles: dict[str, HardwareProfile] = {}
        for pname, pdata in raw_profiles.items():
            vb_raw = pdata.get("vram_breakdown_estimate_gb")
            if vb_raw:
                vram_budget = VRAMBudget(
                    frozen_base_nf4_gb=float(vb_raw["frozen_base_nf4_gb"]),
                    lora_trainable_r64_gb=float(vb_raw["lora_trainable_r64_gb"]),
                    paged_adamw_8bit_gb=float(vb_raw["paged_adamw_8bit_gb"]),
                    activations_kv_cache_gb=float(vb_raw["activations_kv_cache_gb"]),
                    cuda_context_overhead_gb=float(vb_raw["cuda_context_overhead_gb"]),
                )
            else:
                vram_budget = VRAMBudget(
                    frozen_base_nf4_gb=0.35,
                    lora_trainable_r64_gb=0.08,
                    paged_adamw_8bit_gb=0.08,
                    activations_kv_cache_gb=0.35,
                    cuda_context_overhead_gb=0.60,
                )

            profiles[pname] = HardwareProfile(
                name=pname,
                description=pdata.get("description", pname),
                target_vram_gb=float(pdata.get("target_vram_gb", 8.0)),
                available_vram_budget_gb=float(pdata.get("available_vram_budget_gb", 6.4)),
                hf_model_id=pdata.get("hf_model_id", "Qwen/Qwen2.5-0.5B"),
                compute_dtype=pdata.get("compute_dtype", "float16"),
                quantization_4bit=pdata.get("quantization_4bit", "nf4"),
                double_quantization=bool(pdata.get("double_quantization", True)),
                lora_rank_r=int(pdata.get("lora_rank_r", 64)),
                lora_alpha=int(pdata.get("lora_alpha", 128)),
                lora_target_modules=pdata.get(
                    "lora_target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                ),
                micro_batch_size=int(pdata.get("micro_batch_size", 1)),
                seq_len=int(pdata.get("seq_len", 512)),
                gradient_checkpointing=bool(pdata.get("gradient_checkpointing", True)),
                sim_inner_steps_H=int(pdata.get("sim_inner_steps_H", 100)),
                sim_spot_check_interval_s=int(pdata.get("sim_spot_check_interval_s", 25)),
                sim_workers_per_branch=int(pdata.get("sim_workers_per_branch", 5)),
                sim_byzantine_workers=int(pdata.get("sim_byzantine_workers", 1)),
                vram_budget=vram_budget,
            )

        return cls(
            codename=sys_cfg.get("codename", "Project HiveMind"),
            deterministic_cuda=bool(sys_cfg.get("deterministic_cuda", True)),
            cublas_workspace_config=sys_cfg.get("cublas_workspace_config", ":4096:8"),
            spot_check_interval_steps=int(sys_cfg.get("spot_check_interval_steps", 50)),
            spot_check_tolerance_eps=float(sys_cfg.get("spot_check_tolerance_eps", 1e-4)),
            idle_threshold_seconds=float(sys_cfg.get("idle_threshold_seconds", 30.0)),
            emergency_checkpoint_timeout_sec=float(
                sys_cfg.get("emergency_checkpoint_timeout_sec", 1.5)
            ),
            expert_domains=sys_cfg.get(
                "expert_domains",
                [
                    "Code",
                    "Mathematics",
                    "Creative_Writing",
                    "STEM",
                    "Legal_Medical",
                    "Multilingual_FR",
                    "Reasoning",
                    "General",
                ],
            ),
            inner_steps_H=int(diloco_cfg.get("inner_steps_H", 500)),
            inner_lr=float(diloco_cfg.get("inner_lr", 2e-4)),
            inner_weight_decay=float(diloco_cfg.get("inner_weight_decay", 0.01)),
            outer_lr_eta=float(diloco_cfg.get("outer_lr_eta", 0.7)),
            outer_momentum_mu=float(diloco_cfg.get("outer_momentum_mu", 0.9)),
            compression_topk_ratio=float(diloco_cfg.get("compression_topk_ratio", 0.20)),
            compression_dtype=diloco_cfg.get("compression_dtype", "fp8_e4m3fn"),
            multikrum_max_byzantine_fraction=float(
                diloco_cfg.get("multikrum_max_byzantine_fraction", 0.25)
            ),
            cosine_outlier_sigma=float(diloco_cfg.get("cosine_outlier_sigma", 2.0)),
            num_experts_E=int(btm_cfg.get("num_experts_E", 8)),
            router_top_k=int(btm_cfg.get("router_top_k", 2)),
            router_aux_loss_coef=float(btm_cfg.get("router_aux_loss_coef", 0.01)),
            profiles=profiles,
        )


# ============================================================================
# HM-S4-01: Multi-Vendor GPU Backend Detection (CUDA / ROCm / MPS / CPU)
# ============================================================================

GPU_BACKEND_CUDA = "cuda"    # NVIDIA CUDA (SM 7.5 / 8.6 / 12.0)
GPU_BACKEND_ROCM = "rocm"    # AMD ROCm / HIP (gfx1100 RDNA3, gfx1030 RDNA2)
GPU_BACKEND_MPS = "mps"      # Apple Silicon Metal Performance Shaders (unified memory)
GPU_BACKEND_CPU = "cpu"      # CPU-only fallback

GPU_VENDOR_NVIDIA = "nvidia"
GPU_VENDOR_AMD = "amd"
GPU_VENDOR_APPLE = "apple"
GPU_VENDOR_CPU = "cpu"
GPU_VENDOR_UNKNOWN = "unknown"

VALID_GPU_BACKENDS = (GPU_BACKEND_CUDA, GPU_BACKEND_ROCM, GPU_BACKEND_MPS, GPU_BACKEND_CPU)


def is_rocm_available() -> bool:
    """True when the active PyTorch build is an AMD ROCm/HIP build (`torch.version.hip` set)."""
    import torch

    return bool(getattr(torch.version, "hip", None))


def is_mps_available() -> bool:
    """True when the active PyTorch build exposes the Apple Metal (`MPS`) backend."""
    import torch

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        return False
    try:
        return bool(mps_backend.is_available())
    except Exception:
        return False


def is_mps_built() -> bool:
    """True when the PyTorch wheel was compiled with Metal (`MPS`) support."""
    import torch

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        return False
    try:
        return bool(mps_backend.is_built())
    except Exception:
        return False


def resolve_default_device():
    """Resolve the best available accelerator: NVIDIA/AMD CUDA, then Apple MPS, then CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def detect_gpu_backend(device: Any | None = None) -> str:
    """Resolve the effective GPU backend for `device`.

    Returns one of `cuda` (NVIDIA), `rocm` (AMD HIP), `mps` (Apple Metal) or `cpu`.
    ROCm builds expose the `torch.cuda` namespace, so they are discriminated through
    `torch.version.hip` rather than through the device type.
    """
    import torch

    if device is None:
        device = resolve_default_device()
    dev_type = torch.device(device).type
    if dev_type == "cuda":
        return GPU_BACKEND_ROCM if is_rocm_available() else GPU_BACKEND_CUDA
    if dev_type == "mps":
        return GPU_BACKEND_MPS
    return GPU_BACKEND_CPU


def detect_gpu_vendor(backend: str) -> str:
    """Map a resolved GPU backend onto its silicon vendor."""
    if backend == GPU_BACKEND_CUDA:
        return GPU_VENDOR_NVIDIA
    if backend == GPU_BACKEND_ROCM:
        return GPU_VENDOR_AMD
    if backend == GPU_BACKEND_MPS:
        return GPU_VENDOR_APPLE
    if backend == GPU_BACKEND_CPU:
        return GPU_VENDOR_CPU
    return GPU_VENDOR_UNKNOWN


def detect_rocm_gcn_arch(device: Any | None = None) -> str | None:
    """Return the AMD GCN/RDNA architecture identifier (e.g. `gfx1100`, `gfx1030`)."""
    import torch

    if not is_rocm_available() or not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(0 if device is None else device)
    except Exception:
        return None
    arch = getattr(props, "gcnArchName", None)
    if arch:
        return str(arch)
    # Fall back to the `major`/`minor` pair exposed by ROCm device properties.
    major = getattr(props, "major", None)
    if major is None:
        return None
    return f"gfx{int(major):02x}{int(getattr(props, 'minor', 0)):02x}"


def rocm_supports_native_bf16(gcn_arch: str | None) -> bool:
    """Whether an AMD GCN/RDNA target exposes native BF16 matrix operations.

    RDNA3 (`gfx11xx`, Radeon RX 7000) and CDNA2+ (`gfx9xx`) do; RDNA2 (`gfx10xx`, Radeon RX 6000)
    only exposes FP16 matrix paths and therefore falls back to `float16` compute.
    """
    if not gcn_arch:
        return False  # Unknown target: stay on the universally supported FP16 path.
    arch = str(gcn_arch).lower()
    if not arch.startswith("gfx"):
        return False
    family = arch[3:5]
    if family == "9":  # CDNA / CDNA2 / CDNA3 (MI100 / MI200 / MI300)
        return True
    if family == "11":  # RDNA3 (gfx1100 = Radeon RX 7000 series)
        return True
    if family == "12":  # RDNA4 (gfx1200 = Radeon RX 9000 series)
        return True
    if family == "10":  # RDNA2 (gfx1030 = Radeon RX 6000 series) - FP16 only
        return False
    return False


def configure_deterministic_runtime(seed: int = 42, warn_only: bool = False) -> dict[str, Any]:
    """Enforce deterministic CUDA/CPU kernels for BOINC Spot-Check Verification (IP-43).

    Defaults ``warn_only=False`` so non-deterministic CUDA/CPU kernels fail closed
    during cryptographic spot-check replay rather than silently emitting a warning.
    Also records whether ``PYTHONHASHSEED`` was already set to ``str(seed)`` at
    interpreter startup (mutating ``os.environ["PYTHONHASHSEED"]`` after startup
    only affects child processes).
    """
    import torch

    hashseed_preconfigured = os.environ.get("PYTHONHASHSEED") == str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=bool(warn_only))
    return {
        "seed": int(seed),
        "warn_only": bool(warn_only),
        "hashseed_preconfigured": hashseed_preconfigured,
        "deterministic_algorithms": True,
    }


def detect_local_hardware() -> dict[str, Any]:
    """Inspect local GPU (CUDA / ROCm / MPS) and system RAM for node assignment.

    HM-S4-01: the probe now discriminates the **four** accelerator classes used by the swarm:
    NVIDIA CUDA, AMD ROCm/HIP (`gfx1100`/`gfx1030`), Apple Silicon MPS (unified memory) and CPU.
    """
    import subprocess
    import psutil
    import torch

    vm = psutil.virtual_memory()
    mps_available = is_mps_available()
    rocm_available = is_rocm_available()
    backend = detect_gpu_backend()
    info: dict[str, Any] = {
        "system_ram_total_gb": round(vm.total / (1024**3), 2),
        "system_ram_available_gb": round(vm.available / (1024**3), 2),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": "CPU (CUDA not linked in active Python)"
        if not torch.cuda.is_available() and not mps_available
        else "",
        "vram_total_gb": 0.0,
        "vram_free_gb": 0.0,
        "compute_capability": None,
        "supports_native_bf16": False,
        "recommended_dtype": "float16",
        "assigned_backbone_tier": "7B/9B (8GB Consumer Node)",
        # --- HM-S4-01: multi-vendor accelerator telemetry ---
        "gpu_backend": backend,
        "gpu_vendor": detect_gpu_vendor(backend),
        "gpu_arch": None,
        "mps_available": mps_available,
        "mps_built": is_mps_built(),
        "rocm_available": rocm_available,
        "hip_version": getattr(torch.version, "hip", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "unified_memory": False,
        "zero_copy_staging_supported": False,
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free_b, total_b = torch.cuda.mem_get_info(0)
        cc = f"{props.major}.{props.minor}"
        supports_bf16 = props.major >= 8
        info.update(
            {
                "gpu_name": props.name,
                "vram_total_gb": round(total_b / (1024**3), 2),
                "vram_free_gb": round(free_b / (1024**3), 2),
                "compute_capability": cc,
                "supports_native_bf16": supports_bf16,
                "recommended_dtype": "bfloat16" if supports_bf16 else "float16",
                "assigned_backbone_tier": (
                    "14B (12GB+ Target Node)"
                    if (total_b / (1024**3)) >= 11.5
                    else "7B/9B or 8x0.5B/1.5B MoE (8GB Consumer Node)"
                ),
            }
        )
        if rocm_available:
            # --- AMD ROCm / HIP (Radeon RX 7000 `gfx1100` & RX 6000 `gfx1030`) ---
            gcn_arch = detect_rocm_gcn_arch(0)
            rocm_bf16 = rocm_supports_native_bf16(gcn_arch)
            info.update(
                {
                    "gpu_backend": GPU_BACKEND_ROCM,
                    "gpu_vendor": GPU_VENDOR_AMD,
                    "gpu_arch": gcn_arch,
                    "compute_capability": gcn_arch or cc,
                    "supports_native_bf16": rocm_bf16,
                    "recommended_dtype": "bfloat16" if rocm_bf16 else "float16",
                    "unified_memory": bool(getattr(props, "unified_memory", False)),
                    "zero_copy_staging_supported": bool(
                        getattr(props, "unified_memory", False)
                    ),
                }
            )
    elif mps_available:
        # --- Apple Silicon Metal Performance Shaders (unified memory) ---
        import platform

        chip = platform.processor() or platform.machine() or "Apple Silicon"
        ram_gb = vm.total / (1024**3)
        info.update(
            {
                "gpu_name": f"Apple {chip} (Metal Performance Shaders)",
                "gpu_backend": GPU_BACKEND_MPS,
                "gpu_vendor": GPU_VENDOR_APPLE,
                "gpu_arch": chip,
                "vram_total_gb": round(ram_gb, 2),
                "vram_free_gb": round(vm.available / (1024**3), 2),
                "compute_capability": "mps",
                "supports_native_bf16": is_mps_built(),
                "recommended_dtype": "float16",
                "unified_memory": True,
                "zero_copy_staging_supported": True,
                "assigned_backbone_tier": (
                    "32B/70B (64GB+ Apple Silicon Unified Memory Node)"
                    if ram_gb >= 64
                    else "14B (32GB Apple Silicon Unified Memory Node)"
                    if ram_gb >= 32
                    else "7B/9B (16GB Apple Silicon Unified Memory Node)"
                ),
            }
        )
    else:
        # Query nvidia-smi if torch is currently CPU-only
        try:
            res = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                parts = [p.strip() for p in res.stdout.strip().split(",")]
                if len(parts) >= 4:
                    total_gb = round(float(parts[1]) / 1024.0, 2)
                    free_gb = round(float(parts[2]) / 1024.0, 2)
                    cc = parts[3]
                    major = int(cc.split(".")[0]) if "." in cc else 7
                    info.update(
                        {
                            "gpu_name": f"{parts[0]} (Detected via NVML/nvidia-smi)",
                            "gpu_vendor": GPU_VENDOR_NVIDIA,
                            "vram_total_gb": total_gb,
                            "vram_free_gb": free_gb,
                            "compute_capability": cc,
                            "supports_native_bf16": major >= 8,
                            "recommended_dtype": "bfloat16" if major >= 8 else "float16",
                        }
                    )
        except Exception:
            _ = None

    info["compute_backend"] = info["gpu_backend"]
    return info
