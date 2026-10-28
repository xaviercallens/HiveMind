# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Season 21 (``S21-basemodels``): base-model catalog, VRAM estimator, promotion gate and per-tier upgrade plan.

The question this module answers is "which base model should a HiveMind GPU tier train and serve next, and when may a
candidate replace the incumbent ``Qwen/Qwen2.5-Coder-7B-Instruct``?". It is split in four parts that never invent a number:

1. **Catalog.** Every factual field of a model (revision sha, parameter count, safetensors bytes, license, architecture) is
   read from the Hugging Face Hub: ``/api/models/<id>?blobs=true`` for the metadata, ``config.json`` at the pinned revision for
   the architecture, and the safetensors *headers* (HTTP range requests, no weights downloaded) for the per-component
   parameter split. Each entry carries the retrieval time and the revision sha. A model id that the Hub does not serve is
   recorded under ``not_found`` with the HTTP status and left out of ``models``. ``scripts/refresh_base_model_catalog.py`` runs
   the fetch and writes ``configs/base_models_s21.json``; the HTTP getter is injected so tests mock only the network.
2. **VRAM estimator.** Analytic, CPU-only, and labelled ``ESTIMATE`` everywhere. For 4-bit NF4 QLoRA training it sums the
   quantised linear weights, the 16-bit embeddings / norms (and, when loaded, the vision and multi-token-prediction towers),
   LoRA weights + gradients + optimizer state, activations at the given sequence length and micro-batch (with or without
   gradient checkpointing), the logits, one de-quantisation buffer and a CUDA-context allowance. For Q4_0 / Q8_0 GGUF serving
   it sums the quantised weights, the KV cache of the full-attention layers, the recurrent state of linear-attention layers,
   a compute buffer and the same context allowance. The constants are the ones ``scripts/qlora_vram_estimate.py`` uses and
   cites. A fit verdict adds a stated safety margin (default 20 %, the floor that script never goes below) before comparing
   against a GPU budget. :func:`calibrate_from_measurement` records a real measured peak against an estimate and the ratio.
3. **Promotion gate.** Exact McNemar test (two-sided binomial on the discordant pairs) on paired held-out outcomes of the
   candidate and the incumbent. ``PROMOTE`` requires p < alpha AND candidate wins > losses AND no regression on a separate
   guard pack. Identical outcomes have no discordant pairs, p = 1, and never promote.
4. **Upgrade plan.** Derived by code from the catalog and the estimator for each GPU tier: the largest candidate whose
   training recipe fits with the margin, the largest model whose serving fits, and whether the incumbent still fits. The plan
   only nominates a candidate for the promotion gate; "larger" is never taken to mean "better".

Nothing here runs on a GPU, trains, or downloads weights.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "hivemind.base_model_catalog/v1"
HF_BASE = "https://huggingface.co"
INCUMBENT_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
INCUMBENT, CANDIDATE, REFERENCE, S19_REFERENCED = "incumbent", "candidate", "reference", "s19_referenced"

# (model id, role). The S19 ids are the ones ``qwen35_usage_moe_orchestrator.QWEN35_HARDWARE_TIERS`` names as backbones: they
# are checked so the catalog says whether they exist, but they are never recommended (only CANDIDATE models are).
REQUESTED_MODELS: tuple[tuple[str, str], ...] = (
    (INCUMBENT_ID, INCUMBENT),
    ("Qwen/Qwen3.5-0.8B", CANDIDATE),
    ("Qwen/Qwen3.5-4B", CANDIDATE),
    ("Qwen/Qwen3.5-9B", CANDIDATE),
    ("Qwen/Qwen2.5-Coder-14B-Instruct", REFERENCE),
    ("Qwen/Qwen3.5-9B-Base", S19_REFERENCED),
    ("Qwen/Qwen3.5-Coder-3B", S19_REFERENCED),
    ("Qwen/Qwen3.5-Coder-14B-Instruct", S19_REFERENCED),
    ("Qwen/Qwen3.5-Coder-30B-A3B-MoE", S19_REFERENCED),
)

HttpGet = Callable[[str, Mapping[str, str]], tuple[int, bytes]]
MIB = 2**20
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


# ---------------------------------------------------------------------------------------------------------------------------
# 1. Catalog: fetch from the Hugging Face Hub
# ---------------------------------------------------------------------------------------------------------------------------


def urllib_get(url: str, headers: Mapping[str, str]) -> tuple[int, bytes]:
    """Plain HTTPS GET returning (status, body); an HTTP error status is returned, not raised."""
    req = urllib.request.Request(url, headers={"User-Agent": "hivemind-base-model-catalog", **dict(headers)})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return int(resp.status), resp.read()
    except urllib.error.HTTPError as err:
        return int(err.code), err.read() if err.fp is not None else b""


def _get_json(http_get: HttpGet, url: str) -> tuple[int, Any]:
    status, body = http_get(url, {})
    if status != 200:
        return status, None
    return status, json.loads(body.decode("utf-8"))


def read_safetensors_header(http_get: HttpGet, url: str) -> dict[str, tuple[list[int], str]]:
    """Tensor name -> (shape, dtype) of one safetensors file, from two range requests (8-byte length, then the JSON header)."""
    status, head = http_get(url, {"Range": "bytes=0-7"})
    if status not in (200, 206) or len(head) < 8:
        raise RuntimeError(f"safetensors length prefix unavailable for {url} (HTTP {status})")
    n = int.from_bytes(head[:8], "little")
    status, raw = http_get(url, {"Range": f"bytes=8-{7 + n}"})
    if status not in (200, 206):
        raise RuntimeError(f"safetensors header unavailable for {url} (HTTP {status})")
    out: dict[str, tuple[list[int], str]] = {}
    for name, meta in json.loads(raw[:n].decode("utf-8")).items():
        if name != "__metadata__":
            out[name] = (list(meta["shape"]), str(meta["dtype"]))
    return out


def classify_tensor(name: str, shape: Sequence[int]) -> str:
    """Component of a checkpoint tensor: vision / mtp / embedding / linear (an nn.Linear weight NF4 would quantise) / other."""
    if ".visual." in name or name.startswith(("visual.", "vision_tower.", "model.vision")):
        return "vision"
    if name.startswith("mtp.") or ".mtp." in name:
        return "mtp"
    if re.search(r"(embed_tokens|lm_head|\bwte\b|\bwpe\b)", name):
        return "embedding"
    if len(shape) == 2 and name.endswith(".weight"):
        return "linear"
    return "other"


def summarize_header(tensors: Mapping[str, tuple[Sequence[int], str]]) -> dict[str, int]:
    """Parameter split per component plus the LoRA geometry (sum of in+out over the text linear layers) from header shapes."""
    acc = {"params_total": 0, "params_text_linear": 0, "params_embedding": 0, "params_text_other": 0, "params_vision": 0,
           "params_mtp": 0, "lora_in_plus_out_all_linear": 0, "lora_in_plus_out_mlp": 0, "max_linear_elements": 0,
           "n_tensors": 0}
    for name, (shape, _dtype) in tensors.items():
        n = math.prod(int(d) for d in shape)
        acc["params_total"] += n
        acc["n_tensors"] += 1
        kind = classify_tensor(name, shape)
        key = {"vision": "params_vision", "mtp": "params_mtp", "embedding": "params_embedding",
               "linear": "params_text_linear"}.get(kind, "params_text_other")
        acc[key] += n
        if kind == "linear":
            in_plus_out = int(shape[0]) + int(shape[1])
            acc["lora_in_plus_out_all_linear"] += in_plus_out
            if ".mlp." in name:
                acc["lora_in_plus_out_mlp"] += in_plus_out
            acc["max_linear_elements"] = max(acc["max_linear_elements"], n)
    return acc


def architecture_from_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Text-model geometry from a ``config.json`` (``text_config`` when the model is multimodal)."""
    t = dict(cfg.get("text_config") or cfg)
    hidden, layers = int(t["hidden_size"]), int(t["num_hidden_layers"])
    heads = int(t["num_attention_heads"])
    layer_types = list(t.get("layer_types") or [])
    n_full = sum(1 for x in layer_types if x == "full_attention") if layer_types else layers
    return {
        "model_type": str(t.get("model_type") or cfg.get("model_type") or ""),
        "architectures": list(cfg.get("architectures") or []),
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "intermediate_size": int(t["intermediate_size"]),
        "vocab_size": int(t["vocab_size"]),
        "num_attention_heads": heads,
        "num_key_value_heads": int(t.get("num_key_value_heads") or heads),
        "head_dim": int(t.get("head_dim") or hidden // heads),
        "tie_word_embeddings": bool(t.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False))),
        "n_full_attention_layers": n_full,
        "n_linear_attention_layers": layers - n_full,
        "linear_num_key_heads": int(t.get("linear_num_key_heads") or 0),
        "linear_num_value_heads": int(t.get("linear_num_value_heads") or 0),
        "linear_key_head_dim": int(t.get("linear_key_head_dim") or 0),
        "linear_value_head_dim": int(t.get("linear_value_head_dim") or 0),
        "linear_conv_kernel_dim": int(t.get("linear_conv_kernel_dim") or 0),
        "has_vision_config": "vision_config" in cfg,
        "max_position_embeddings": int(t.get("max_position_embeddings") or 0),
    }


def fetch_catalog_entry(model_id: str, role: str, http_get: HttpGet, retrieved_at: str) -> dict[str, Any]:
    """One catalog entry fetched from the Hub, or ``{"exists": False, "http_status": ...}`` when the id is not served."""
    api_url = f"{HF_BASE}/api/models/{model_id}?blobs=true"
    status, info = _get_json(http_get, api_url)
    if status != 200 or not isinstance(info, dict):
        return {"model_id": model_id, "role": role, "exists": False, "http_status": status, "retrieved_at": retrieved_at,
                "source": api_url,
                "note": "the Hub did not serve this id (it answers 401 for ids that do not exist or are private/gated)"}
    sha = str(info["sha"])
    siblings = info.get("siblings") or []
    st_files = [s for s in siblings if str(s.get("rfilename", "")).endswith(".safetensors")]
    card = info.get("cardData") or {}
    license_tags = [str(t).split(":", 1)[1] for t in info.get("tags") or [] if str(t).startswith("license:")]
    cfg_url = f"{HF_BASE}/{model_id}/resolve/{sha}/config.json"
    cfg_status, cfg = _get_json(http_get, cfg_url)
    if cfg_status != 200 or not isinstance(cfg, dict):
        raise RuntimeError(f"config.json unavailable for {model_id}@{sha} (HTTP {cfg_status})")
    tensors: dict[str, tuple[list[int], str]] = {}
    for s in st_files:
        tensors.update(read_safetensors_header(http_get, f"{HF_BASE}/{model_id}/resolve/{sha}/{s['rfilename']}"))
    st = info.get("safetensors") or {}
    return {
        "model_id": model_id,
        "role": role,
        "exists": True,
        "http_status": status,
        "retrieved_at": retrieved_at,
        "revision_sha": sha,
        "last_modified": info.get("lastModified"),
        "license": card.get("license") or (license_tags[0] if license_tags else None),
        "gated": info.get("gated"),
        "pipeline_tag": info.get("pipeline_tag"),
        "library_name": info.get("library_name"),
        "api_parameters_total": st.get("total"),
        "api_parameters_by_dtype": st.get("parameters"),
        "safetensors_total_bytes": sum(int(s.get("size") or 0) for s in st_files),
        "safetensors_files": [{"name": s["rfilename"], "bytes": s.get("size"), "lfs_sha256": (s.get("lfs") or {}).get("sha256")}
                              for s in st_files],
        "architecture": architecture_from_config(cfg),
        "header": summarize_header(tensors),
        "sources": {"api": api_url, "config": cfg_url, "headers": "HTTP range requests on each safetensors file at the revision"},
    }


def build_catalog(http_get: HttpGet, requested: Sequence[tuple[str, str]] = REQUESTED_MODELS,
                  now: datetime | None = None) -> dict[str, Any]:
    """Fetch every requested model; ids the Hub does not serve go to ``not_found`` instead of ``models``."""
    stamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    models: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model_id, role in requested:
        entry = fetch_catalog_entry(model_id, role, http_get, stamp)
        (models if entry["exists"] else missing).append(entry)
    return {"schema": SCHEMA, "generated_by": "scripts/refresh_base_model_catalog.py", "retrieved_at": stamp,
            "incumbent": INCUMBENT_ID, "models": models, "not_found": missing}


def validate_catalog(catalog: Mapping[str, Any]) -> list[str]:
    """Problems that make a catalog unusable (an empty list means it passes). Header totals must match the Hub's own count."""
    problems: list[str] = []
    if catalog.get("schema") != SCHEMA:
        problems.append(f"schema is {catalog.get('schema')!r}, expected {SCHEMA!r}")
    models = list(catalog.get("models") or [])
    ids = [m.get("model_id") for m in models]
    if len(set(ids)) != len(ids):
        problems.append("duplicate model ids")
    if not any(m.get("role") == INCUMBENT for m in models):
        problems.append("no incumbent model in the catalog")
    for m in models:
        mid = m.get("model_id")
        if not _SHA_RE.match(str(m.get("revision_sha") or "")):
            problems.append(f"{mid}: revision_sha is not a 40-hex commit sha")
        if not m.get("retrieved_at"):
            problems.append(f"{mid}: no retrieval date")
        head = m.get("header") or {}
        api_total = m.get("api_parameters_total")
        if not head.get("params_total"):
            problems.append(f"{mid}: no safetensors header summary")
        elif api_total is None:
            # Without the Hub's own count the header cross-check cannot run; a skipped check must not pass silently.
            problems.append(f"{mid}: no Hub safetensors total to cross-check the header parameter total against")
        elif int(api_total) != int(head["params_total"]):
            problems.append(f"{mid}: header parameter total {head['params_total']} != Hub safetensors total {api_total}")
        if not m.get("license"):
            problems.append(f"{mid}: license missing")
    return problems


def load_catalog(path: str | Path) -> dict[str, Any]:
    """Read and validate a catalog file; raises ``ValueError`` listing every problem."""
    catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = validate_catalog(catalog)
    if problems:
        raise ValueError("invalid base-model catalog: " + "; ".join(problems))
    return catalog


# ---------------------------------------------------------------------------------------------------------------------------
# 2. VRAM estimator (an ESTIMATE, never a measurement)
# ---------------------------------------------------------------------------------------------------------------------------

# Bytes per NF4-quantised parameter including the double-quantisation state. Same constant as scripts/t4_model_candidates.py;
# docs/T4_NIGHT_MODEL_DECISION.md reports its header prediction for Qwen3-4B (2,579 MiB) against 2,578 MiB measured resident.
NF4_BYTES = 0.53
FP16_BYTES = 2.0
LORA_BYTES = 4.0  # fp32 LoRA weights
GRAD_BYTES = 4.0  # fp32 LoRA gradients
OPTIMIZER_BYTES: dict[str, float] = {"adamw": 8.0, "adamw8bit": 2.0, "sgd": 0.0}
# Activation constants and context allowance as in scripts/qlora_vram_estimate.py (the latter is 3716 process peak minus
# 3409 torch-allocated peak, both measured on the T4: deploy/t4_node/live/README.md and docs/T4_NIGHT_MODEL_DECISION.md).
ATTN_ELEMS = 10.0
MLP_ELEMS = 3.0
LOGIT_BYTES = 10.0
CONTEXT_ALLOWANCE_MIB = 3716 - 3409
DEFAULT_MARGIN = 0.20  # scripts/qlora_vram_estimate.py never claims a tighter margin than 20 percent
# GGUF block layouts (ggml): Q4_0 = 32 weights in 16 nibble bytes + one fp16 scale; Q8_0 = 32 int8 bytes + one fp16 scale.
GGUF_BLOCKS: dict[str, tuple[int, int]] = {"Q4_0": (32, 16 + 2), "Q8_0": (32, 32 + 2)}
SERVE_UBATCH = 512  # llama.cpp default micro-batch; the compute buffer is sized for it

FITS, MARGINAL, DOES_NOT_FIT = "FITS", "MARGINAL", "DOES_NOT_FIT"


def gguf_bytes_per_weight(quant: str) -> float:
    """Bytes per weight of a GGUF quant type, derived from its block layout (Q4_0 -> 0.5625, Q8_0 -> 1.0625)."""
    if quant not in GGUF_BLOCKS:
        raise ValueError(f"quant must be one of {sorted(GGUF_BLOCKS)}")
    weights, nbytes = GGUF_BLOCKS[quant]
    return nbytes / weights


@dataclass(frozen=True)
class ModelGeometry:
    """What the estimator needs, taken from one catalog entry (header split + config geometry)."""

    model_id: str
    params_text_linear: int
    params_embedding: int
    params_text_other: int
    params_vision: int
    params_mtp: int
    lora_in_plus_out_all_linear: int
    lora_in_plus_out_mlp: int
    max_linear_elements: int
    hidden: int
    layers: int
    intermediate: int
    vocab: int
    n_kv_heads: int
    head_dim: int
    n_full_attention_layers: int
    n_linear_attention_layers: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int

    @property
    def params_text(self) -> int:
        return self.params_text_linear + self.params_embedding + self.params_text_other


def geometry_from_entry(entry: Mapping[str, Any]) -> ModelGeometry:
    h, a = entry["header"], entry["architecture"]
    return ModelGeometry(
        model_id=str(entry["model_id"]), params_text_linear=int(h["params_text_linear"]), params_embedding=int(h["params_embedding"]),
        params_text_other=int(h["params_text_other"]), params_vision=int(h["params_vision"]), params_mtp=int(h["params_mtp"]),
        lora_in_plus_out_all_linear=int(h["lora_in_plus_out_all_linear"]), lora_in_plus_out_mlp=int(h["lora_in_plus_out_mlp"]),
        max_linear_elements=int(h["max_linear_elements"]), hidden=int(a["hidden_size"]), layers=int(a["num_hidden_layers"]),
        intermediate=int(a["intermediate_size"]), vocab=int(a["vocab_size"]), n_kv_heads=int(a["num_key_value_heads"]),
        head_dim=int(a["head_dim"]), n_full_attention_layers=int(a["n_full_attention_layers"]),
        n_linear_attention_layers=int(a["n_linear_attention_layers"]), linear_num_key_heads=int(a["linear_num_key_heads"]),
        linear_num_value_heads=int(a["linear_num_value_heads"]), linear_key_head_dim=int(a["linear_key_head_dim"]),
        linear_value_head_dim=int(a["linear_value_head_dim"]), linear_conv_kernel_dim=int(a["linear_conv_kernel_dim"]),
    )


@dataclass(frozen=True)
class TrainRecipe:
    """One NF4 QLoRA training configuration. ``targets`` is ``all_linear`` (every text nn.Linear) or ``mlp``."""

    seq_len: int = 512
    micro_batch: int = 1
    lora_rank: int = 16
    targets: str = "all_linear"
    gradient_checkpointing: bool = False
    optimizer: str = "adamw"
    act_scale: float = 1.0  # uncalibrated; scripts/qlora_vram_estimate.py fits one when the calibration models are cached
    load_aux_towers: bool = True  # vision + MTP weights loaded at 16-bit (conservative: a text-only loader may drop them)

    def __post_init__(self) -> None:
        if self.seq_len < 1 or self.micro_batch < 1 or self.lora_rank < 1:
            raise ValueError("seq_len, micro_batch and lora_rank must be >= 1")
        if self.targets not in ("all_linear", "mlp"):
            raise ValueError("targets must be 'all_linear' or 'mlp'")
        if self.optimizer not in OPTIMIZER_BYTES:
            raise ValueError(f"optimizer must be one of {sorted(OPTIMIZER_BYTES)}")
        if self.act_scale <= 0:
            raise ValueError("act_scale must be > 0")

    @property
    def label(self) -> str:
        ck = " ckpt" if self.gradient_checkpointing else ""
        return f"seq{self.seq_len} mb{self.micro_batch} r{self.lora_rank} {self.targets}{ck}"


@dataclass(frozen=True)
class ServeConfig:
    quant: str = "Q4_0"
    context_len: int = 4096
    n_parallel: int = 1

    def __post_init__(self) -> None:
        gguf_bytes_per_weight(self.quant)
        if self.context_len < 1 or self.n_parallel < 1:
            raise ValueError("context_len and n_parallel must be >= 1")

    @property
    def label(self) -> str:
        return f"{self.quant} ctx{self.context_len} x{self.n_parallel}"


@dataclass(frozen=True)
class VramEstimate:
    model_id: str
    mode: str  # "qlora_nf4_training" | "gguf_serving"
    config: str
    components_mib: dict[str, float]
    total_mib: float
    kind: str = "ESTIMATE"
    note: str = "analytic estimate; nothing ran on a GPU"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_qlora_training(g: ModelGeometry, r: TrainRecipe) -> VramEstimate:
    """Peak-VRAM ESTIMATE for NF4 QLoRA training of ``g`` with recipe ``r`` (process level, context allowance included)."""
    lora_params = r.lora_rank * (g.lora_in_plus_out_all_linear if r.targets == "all_linear" else g.lora_in_plus_out_mlp)
    tokens = r.seq_len * r.micro_batch
    per_layer = tokens * (ATTN_ELEMS * g.hidden + MLP_ELEMS * g.intermediate) * FP16_BYTES / MIB
    if r.gradient_checkpointing:
        layer_acts = g.layers * tokens * g.hidden * FP16_BYTES / MIB + per_layer
    else:
        layer_acts = g.layers * per_layer
    comps = {
        "nf4_weights": g.params_text_linear * NF4_BYTES / MIB,
        "fp16_embeddings_norms": (g.params_embedding + g.params_text_other) * FP16_BYTES / MIB,
        "fp16_aux_towers": (g.params_vision + g.params_mtp) * FP16_BYTES / MIB if r.load_aux_towers else 0.0,
        "lora_weights_grads_optimizer": lora_params * (LORA_BYTES + GRAD_BYTES + OPTIMIZER_BYTES[r.optimizer]) / MIB,
        "activations": r.act_scale * layer_acts,
        "logits": r.act_scale * tokens * g.vocab * LOGIT_BYTES / MIB,
        "dequant_buffer": 2 * FP16_BYTES * g.max_linear_elements / MIB,
        "cuda_context_allowance": float(CONTEXT_ALLOWANCE_MIB),
    }
    return VramEstimate(g.model_id, "qlora_nf4_training", r.label, {k: round(v, 1) for k, v in comps.items()}, round(sum(comps.values()), 1))


def estimate_gguf_serving(g: ModelGeometry, s: ServeConfig) -> VramEstimate:
    """VRAM ESTIMATE for serving the text model as a GGUF of type ``s.quant`` (vision projector not loaded)."""
    bpw = gguf_bytes_per_weight(s.quant)
    seqs = s.n_parallel
    kv = seqs * s.context_len * 2 * g.n_full_attention_layers * g.n_kv_heads * g.head_dim * FP16_BYTES / MIB
    ssm = g.linear_num_value_heads * g.linear_key_head_dim * g.linear_value_head_dim * 4
    conv_ch = 2 * g.linear_num_key_heads * g.linear_key_head_dim + g.linear_num_value_heads * g.linear_value_head_dim
    conv = max(g.linear_conv_kernel_dim - 1, 0) * conv_ch * 4
    comps = {
        "quantised_weights": (g.params_text_linear + g.params_embedding) * bpw / MIB,
        "fp32_norms_other": g.params_text_other * 4 / MIB,
        "kv_cache_fp16": kv,
        "linear_attention_state": seqs * g.n_linear_attention_layers * (ssm + conv) / MIB,
        "compute_buffer": SERVE_UBATCH * (g.vocab + 4 * g.hidden + 2 * g.intermediate) * 4 / MIB,
        "cuda_context_allowance": float(CONTEXT_ALLOWANCE_MIB),
    }
    return VramEstimate(g.model_id, "gguf_serving", s.label, {k: round(v, 1) for k, v in comps.items()}, round(sum(comps.values()), 1))


@dataclass(frozen=True)
class FitVerdict:
    verdict: str
    fits: bool
    estimate_mib: float
    required_with_margin_mib: float
    budget_mib: float
    margin: float


def fit_verdict(estimate_mib: float, budget_mib: float, margin: float = DEFAULT_MARGIN) -> FitVerdict:
    """FITS when estimate x (1 + margin) <= budget; MARGINAL when only the bare estimate fits; else DOES_NOT_FIT."""
    if margin < 0 or budget_mib <= 0 or estimate_mib < 0:
        raise ValueError("margin >= 0, budget > 0 and estimate >= 0 are required")
    need = estimate_mib * (1.0 + margin)
    verdict = FITS if need <= budget_mib else (MARGINAL if estimate_mib <= budget_mib else DOES_NOT_FIT)
    return FitVerdict(verdict, verdict == FITS, round(estimate_mib, 1), round(need, 1), float(budget_mib), margin)


@dataclass(frozen=True)
class CalibrationRecord:
    model_id: str
    mode: str
    config: str
    estimate_mib: float
    measured_peak_mib: float
    measured_kind: str
    ratio_measured_to_estimate: float
    within_margin: bool
    margin: float
    gpu: str
    source: str
    recorded_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MEASURED_KINDS = ("process_peak", "torch_max_allocated")


def calibrate_from_measurement(estimate: VramEstimate, measured_peak_mib: float, *, measured_kind: str, gpu: str, source: str,
                               margin: float = DEFAULT_MARGIN, now: datetime | None = None) -> CalibrationRecord:
    """Record a REAL measured peak against an estimate. ``source`` must say where the number came from (log path, command).

    ``torch_max_allocated`` excludes the CUDA context, so it is compared against the estimate minus the context allowance.
    ``within_margin`` is False when the measurement exceeds the estimate by more than the margin: the estimator undershot.
    """
    if measured_peak_mib <= 0 or not math.isfinite(measured_peak_mib):
        raise ValueError("measured_peak_mib must be a positive finite number")
    if measured_kind not in MEASURED_KINDS:
        raise ValueError(f"measured_kind must be one of {MEASURED_KINDS}")
    if not source.strip() or not gpu.strip():
        raise ValueError("a calibration needs the GPU and the source of the measurement")
    est = estimate.total_mib
    if measured_kind == "torch_max_allocated":
        est = est - estimate.components_mib.get("cuda_context_allowance", 0.0)
    if est <= 0:
        raise ValueError("estimate must be positive")
    ratio = measured_peak_mib / est
    stamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return CalibrationRecord(estimate.model_id, estimate.mode, estimate.config, round(est, 1), float(measured_peak_mib), measured_kind,
                             round(ratio, 4), ratio <= 1.0 + margin, margin, gpu, source, stamp)


def append_calibration(path: str | Path, record: CalibrationRecord) -> int:
    """Append a calibration record to a JSON list file; returns the number of records now stored."""
    p = Path(path)
    rows = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    if not isinstance(rows, list):
        raise TypeError(f"{p} does not hold a JSON list")
    rows.append(record.to_dict())
    p.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    return len(rows)


# ---------------------------------------------------------------------------------------------------------------------------
# 3. Promotion gate: exact McNemar on paired held-out outcomes + guard-pack non-regression
# ---------------------------------------------------------------------------------------------------------------------------

PROMOTE, HOLD, INSUFFICIENT = "PROMOTE", "HOLD", "INSUFFICIENT"


def binom_half_lower_tail(k: int, n: int) -> float:
    """P(X <= k) for X ~ Binomial(n, 1/2), exact (integer arithmetic until the final division)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.comb(n, i) for i in range(k + 1)) / 2**n


def mcnemar_exact_p(losses: int, wins: int) -> float:
    """Two-sided exact McNemar p from the discordant counts (same definition as ``swarm.verified_update_gate.exact_paired_p``)."""
    if losses < 0 or wins < 0:
        raise ValueError("discordant counts must be >= 0")
    n = losses + wins
    return 1.0 if n == 0 else min(1.0, 2 * binom_half_lower_tail(min(losses, wins), n))


@dataclass(frozen=True)
class PromotionPolicy:
    alpha: float = 0.05
    min_items: int = 50
    min_guard_items: int = 20
    tolerated_guard_net_loss: int = 0  # guard losses minus guard wins allowed

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if self.min_items < 1 or self.min_guard_items < 1 or self.tolerated_guard_net_loss < 0:
            raise ValueError("min_items, min_guard_items >= 1 and tolerated_guard_net_loss >= 0 are required")


DEFAULT_PROMOTION_POLICY = PromotionPolicy()


@dataclass(frozen=True)
class PromotionReceipt:
    decision: str
    reasons: list[str]
    n_items: int
    wins: int  # candidate correct, incumbent wrong
    losses: int  # candidate wrong, incumbent correct
    p_two_sided: float
    guard_items: int
    guard_wins: int
    guard_losses: int
    guard_p_regression: float  # one-sided exact P(X >= guard_losses), X ~ Binomial(guard discordant, 1/2)
    policy: dict[str, Any] = field(default_factory=dict)

    @property
    def promoted(self) -> bool:
        return self.decision == PROMOTE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _discordant(candidate: Mapping[str, bool], incumbent: Mapping[str, bool]) -> tuple[int, int, int, bool]:
    for name, m in (("candidate", candidate), ("incumbent", incumbent)):
        bad = [k for k, v in m.items() if not isinstance(v, bool)]
        if bad:
            raise ValueError(f"{name} outcomes must be booleans (first offending item: {bad[0]!r})")
    same_keys = set(candidate) == set(incumbent)
    keys = set(candidate) & set(incumbent)
    wins = sum(1 for k in keys if candidate[k] and not incumbent[k])
    losses = sum(1 for k in keys if incumbent[k] and not candidate[k])
    return len(keys), wins, losses, same_keys


def promotion_gate(candidate: Mapping[str, bool], incumbent: Mapping[str, bool], guard_candidate: Mapping[str, bool],
                   guard_incumbent: Mapping[str, bool], policy: PromotionPolicy = DEFAULT_PROMOTION_POLICY) -> PromotionReceipt:
    """Decide whether a candidate base model replaces the incumbent. Outcomes are per item id: True = correct.

    PROMOTE iff (1) both packs pair exactly with enough items, (2) exact two-sided McNemar p < alpha on the held-out pack,
    (3) candidate wins > losses there, and (4) on the guard pack losses - wins <= tolerated and no one-sided significant
    regression. A pairing or size failure is INSUFFICIENT; anything else that is not a promotion is HOLD.
    """
    n, wins, losses, paired = _discordant(candidate, incumbent)
    gn, gwins, glosses, gpaired = _discordant(guard_candidate, guard_incumbent)
    p = mcnemar_exact_p(losses, wins)
    gd = gwins + glosses
    g_p_reg = 1.0 - binom_half_lower_tail(glosses - 1, gd) if gd else 1.0
    reasons: list[str] = []
    decision = PROMOTE
    if not paired or not gpaired:
        reasons.append("item ids differ between candidate and incumbent (held-out or guard pack)")
        decision = INSUFFICIENT
    if n < policy.min_items or gn < policy.min_guard_items:
        reasons.append(f"too few paired items: held-out {n} (need {policy.min_items}), guard {gn} (need {policy.min_guard_items})")
        decision = INSUFFICIENT
    if decision == PROMOTE:
        if not p < policy.alpha:
            reasons.append(f"held-out difference not significant: exact McNemar p={p:.4g} >= alpha={policy.alpha}")
            decision = HOLD
        if not wins > losses:
            reasons.append(f"candidate does not win more than it loses on held-out ({wins} wins, {losses} losses)")
            decision = HOLD
        if glosses - gwins > policy.tolerated_guard_net_loss:
            reasons.append(f"guard-pack regression: net loss {glosses - gwins} > tolerated {policy.tolerated_guard_net_loss}")
            decision = HOLD
        elif glosses > gwins and g_p_reg < policy.alpha:
            reasons.append(f"significant guard-pack regression (one-sided p={g_p_reg:.4g})")
            decision = HOLD
    if decision == PROMOTE:
        reasons.append(f"candidate better on held-out (p={p:.4g}, {wins} wins vs {losses} losses) with no guard regression")
    return PromotionReceipt(decision, reasons, n, wins, losses, p, gn, gwins, glosses, g_p_reg, asdict(policy))


# ---------------------------------------------------------------------------------------------------------------------------
# 4. Per-tier upgrade plan derived from the catalog and the estimator
# ---------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class GpuTier:
    tier_id: str
    budget_mib: int
    source: str


# Budgets with their origin. The std/big/xl values equal QWEN35_HARDWARE_TIERS[*]["max_wddm_safe_mib"] (a test asserts it).
GPU_TIERS: tuple[GpuTier, ...] = (
    GpuTier("t4_serving_resident", 4300, "T4 governor --min-free-mib 4300 while serving stays up (docs/T4_NIGHT_MODEL_DECISION.md)"),
    GpuTier("std_8gb", 6656, "qwen35_usage_moe_orchestrator.QWEN35_HARDWARE_TIERS std_8gb max_wddm_safe_mib"),
    GpuTier("big_12gb", 11264, "qwen35_usage_moe_orchestrator.QWEN35_HARDWARE_TIERS big_12gb max_wddm_safe_mib"),
    GpuTier("t4_serving_stopped", 15360, "T4 total memory with serving stopped (S21 task statement)"),
    GpuTier("xl_16gb_plus", 22528, "qwen35_usage_moe_orchestrator.QWEN35_HARDWARE_TIERS xl_16gb_plus max_wddm_safe_mib"),
)

# Tried in order, most ambitious first; the order follows the T4 decision doc's plan (micro-batch 1, then checkpointing,
# then a shorter sequence).
TRAIN_LADDER: tuple[TrainRecipe, ...] = (
    TrainRecipe(seq_len=512, micro_batch=2, lora_rank=16),
    TrainRecipe(seq_len=512, micro_batch=1, lora_rank=16),
    TrainRecipe(seq_len=512, micro_batch=1, lora_rank=16, gradient_checkpointing=True),
    TrainRecipe(seq_len=256, micro_batch=1, lora_rank=16, gradient_checkpointing=True),
)
SERVE_LADDER: tuple[ServeConfig, ...] = (ServeConfig("Q8_0", 4096), ServeConfig("Q4_0", 4096))


def _first_fit(estimates: Sequence[VramEstimate], budget: int, margin: float) -> dict[str, Any] | None:
    for est in estimates:
        v = fit_verdict(est.total_mib, budget, margin)
        if v.fits:
            return {"config": est.config, "estimate_mib": est.total_mib, "required_with_margin_mib": v.required_with_margin_mib}
    return None


def recommend_upgrade_plan(catalog: Mapping[str, Any], tiers: Sequence[GpuTier] = GPU_TIERS,
                           train_ladder: Sequence[TrainRecipe] = TRAIN_LADDER, serve_ladder: Sequence[ServeConfig] = SERVE_LADDER,
                           margin: float = DEFAULT_MARGIN) -> dict[str, Any]:
    """For each tier: what fits to train and serve, and which candidate (if any) goes to the promotion gate."""
    problems = validate_catalog(catalog)
    if problems:
        raise ValueError("invalid base-model catalog: " + "; ".join(problems))
    entries = [m for m in catalog["models"] if m["role"] in (INCUMBENT, CANDIDATE, REFERENCE)]
    geoms = {m["model_id"]: geometry_from_entry(m) for m in entries}
    roles = {m["model_id"]: m["role"] for m in entries}
    train = {mid: [estimate_qlora_training(g, r) for r in train_ladder] for mid, g in geoms.items()}
    serve = {mid: [estimate_gguf_serving(g, s) for s in serve_ladder] for mid, g in geoms.items()}
    by_size = sorted(geoms, key=lambda mid: -geoms[mid].params_text)
    incumbent = str(catalog["incumbent"])
    plan: list[dict[str, Any]] = []
    for tier in tiers:
        rows = {mid: {"role": roles[mid], "params_text": geoms[mid].params_text,
                      "train": _first_fit(train[mid], tier.budget_mib, margin),
                      "serve": _first_fit(serve[mid], tier.budget_mib, margin)} for mid in by_size}
        train_cands = [mid for mid in by_size if roles[mid] == CANDIDATE and rows[mid]["train"]]
        serve_cands = [mid for mid in by_size if roles[mid] == CANDIDATE and rows[mid]["serve"]]
        inc = rows.get(incumbent, {"train": None, "serve": None})
        pick = train_cands[0] if train_cands else None
        if pick is None:
            action = "KEEP_INCUMBENT: no candidate's training recipe fits this tier with the margin"
        elif inc["train"] is None:
            action = f"EVALUATE {pick}: the incumbent does not fit training here; promote only through promotion_gate"
        else:
            action = f"EVALUATE {pick} against {incumbent} with promotion_gate before any switch"
        plan.append({"tier_id": tier.tier_id, "budget_mib": tier.budget_mib, "budget_source": tier.source,
                     "train_candidate": pick, "serve_candidate": serve_cands[0] if serve_cands else None,
                     "incumbent_train": inc["train"], "incumbent_serve": inc["serve"], "action": action, "models": rows})
    return {"kind": "ESTIMATE-derived plan", "margin": margin, "catalog_retrieved_at": catalog.get("retrieved_at"),
            "incumbent": incumbent, "train_ladder": [r.label for r in train_ladder], "serve_ladder": [s.label for s in serve_ladder],
            "tiers": plan,
            "not_found": [m["model_id"] for m in catalog.get("not_found") or []]}


def render_plan(plan: Mapping[str, Any]) -> str:
    lines = [f"Upgrade plan ({plan['kind']}, margin {plan['margin']:.0%}, catalog retrieved {plan['catalog_retrieved_at']})"]
    for t in plan["tiers"]:
        lines.append(f"- {t['tier_id']} ({t['budget_mib']} MiB): train={t['train_candidate']} serve={t['serve_candidate']} :: {t['action']}")
        for mid, row in t["models"].items():
            tr = f"{row['train']['config']} {row['train']['estimate_mib']:.0f} MiB" if row["train"] else "no recipe fits"
            sv = f"{row['serve']['config']} {row['serve']['estimate_mib']:.0f} MiB" if row["serve"] else "does not fit"
            lines.append(f"    {mid:36s} [{row['role']}] train: {tr}; serve: {sv}")
    if plan["not_found"]:
        lines.append("Not on the Hub (left out): " + ", ".join(plan["not_found"]))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Print the per-tier base-model upgrade plan (an ESTIMATE) from a catalog file.")
    ap.add_argument("--catalog", default="configs/base_models_s21.json")
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    plan = recommend_upgrade_plan(load_catalog(a.catalog), margin=a.margin)
    sys.stdout.write((json.dumps(plan, indent=1) if a.json else render_plan(plan)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
