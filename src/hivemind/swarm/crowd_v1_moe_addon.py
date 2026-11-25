# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Serve a Crowd-v1 MoE LoRA add-on on top of a pinned `Vxtzq/Crowd-v1` base.

The add-on is the 12-bucket file the OpenDiLoCo MoE campaigns write (`expert_0..7`, `router_frag_0..3`; the last router
fragment also holds the final LayerNorm). It only means something on the exact base revision it was trained and judged
on, so a manifest pins that revision, the base weights sha256, the split and the add-on sha256, and
:func:`apply_moe_addon` refuses anything that does not match.

Order matters: the bucket loader copies only into parameters that still require grad, so the model must be upcycled
and loaded **before** anything freezes it or moves it for inference. :func:`apply_moe_addon` does the upcycle, the
load and a read-back check (every bucket must read back equal to the file) and only then freezes the parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import torch

MANIFEST_SCHEMA = "hivemind.crowd_v1_moe_addon.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REV = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_SPLIT_KEYS = (
    "split_mode",
    "num_experts",
    "top_k",
    "routing_mode",
    "lora_rank",
    "lora_alpha",
    "resident_expert_indices",
    "num_router_fragments",
    "num_layers_upcycled",
)
READBACK_TOLERANCE = 1e-6


class AddonIntegrityError(ValueError):
    """The add-on, its manifest or the base it is applied to does not match what was judged."""


@dataclass(frozen=True)
class MoEAddonManifest:
    addon_id: str
    addon_file: str
    addon_sha256: str
    base_hf_repo: str
    base_hf_revision: str
    base_weights_sha256: str
    tokenizer_sha256: str
    split: Mapping[str, Any]
    advertised_model_ids: tuple[str, ...]
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MANIFEST_SCHEMA,
            "addon_id": self.addon_id,
            "addon_file": self.addon_file,
            "addon_sha256": self.addon_sha256,
            "base": {
                "hf_repo": self.base_hf_repo,
                "hf_revision": self.base_hf_revision,
                "weights_sha256": self.base_weights_sha256,
                "tokenizer_sha256": self.tokenizer_sha256,
            },
            "split": dict(self.split),
            "advertised_model_ids": list(self.advertised_model_ids),
            "evidence": dict(self.evidence),
        }


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_file_sha256(path: Path | str, expected: str, what: str) -> str:
    """Return the file's sha256, or raise if it differs from ``expected``."""
    got = sha256_file(path)
    if got != expected:
        raise AddonIntegrityError(f"{what} {path}: sha256 {got} does not match the manifest ({expected})")
    return got


def parse_addon_manifest(data: Mapping[str, Any]) -> MoEAddonManifest:
    if data.get("schema") != MANIFEST_SCHEMA:
        raise AddonIntegrityError(f"manifest schema must be {MANIFEST_SCHEMA!r}, got {data.get('schema')!r}")
    base = data.get("base")
    split = data.get("split")
    if not isinstance(base, Mapping) or not isinstance(split, Mapping):
        raise AddonIntegrityError("manifest needs 'base' and 'split' objects")
    for key, value, pattern in (
        ("addon_sha256", data.get("addon_sha256"), _SHA256),
        ("base.weights_sha256", base.get("weights_sha256"), _SHA256),
        ("base.tokenizer_sha256", base.get("tokenizer_sha256"), _SHA256),
        ("base.hf_revision", base.get("hf_revision"), _REV),
    ):
        if not isinstance(value, str) or not pattern.match(value):
            raise AddonIntegrityError(f"manifest {key} is missing or malformed")
    missing = [k for k in REQUIRED_SPLIT_KEYS if k not in split]
    if missing:
        raise AddonIntegrityError(f"manifest split is missing {missing}")
    if split["split_mode"] != "lora_domain_experts":
        raise AddonIntegrityError("only the lora_domain_experts split can be served as an add-on")
    ids = data.get("advertised_model_ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(i, str) and i for i in ids):
        raise AddonIntegrityError("manifest advertised_model_ids must be a non-empty list of strings")
    base_ids = {"hivemind/crowd-v1", "crowd-v1", "crowdgpt/crowd-v1", "Vxtzq/Crowd-v1", "hf.co/Vxtzq/Crowd-v1"}
    if base_ids & set(ids):
        raise AddonIntegrityError("an add-on must not advertise the base model's IDs")
    addon_file = data.get("addon_file")
    if not isinstance(addon_file, str) or not addon_file or "/" in addon_file or addon_file.startswith("."):
        raise AddonIntegrityError("manifest addon_file must be a plain file name")
    return MoEAddonManifest(
        addon_id=str(data.get("addon_id") or ""),
        addon_file=addon_file,
        addon_sha256=str(data["addon_sha256"]),
        base_hf_repo=str(base.get("hf_repo") or "Vxtzq/Crowd-v1"),
        base_hf_revision=str(base["hf_revision"]),
        base_weights_sha256=str(base["weights_sha256"]),
        tokenizer_sha256=str(base["tokenizer_sha256"]),
        split=dict(split),
        advertised_model_ids=tuple(ids),
        evidence=dict(data.get("evidence") or {}),
    )


def load_addon_manifest(path: Path | str) -> MoEAddonManifest:
    return parse_addon_manifest(json.loads(Path(path).read_text(encoding="utf-8")))


def check_buckets_against_template(saved: Mapping[str, torch.Tensor], template: Mapping[str, torch.Tensor]) -> None:
    """Every bucket name and shape must match the freshly upcycled model's buckets."""
    if set(saved) != set(template):
        missing, extra = sorted(set(template) - set(saved)), sorted(set(saved) - set(template))
        raise AddonIntegrityError(f"add-on bucket names differ (missing {missing[:4]}, extra {extra[:4]})")
    for k, v in template.items():
        if tuple(saved[k].shape) != tuple(v.shape):
            raise AddonIntegrityError(f"add-on bucket {k} has shape {tuple(saved[k].shape)}, expected {tuple(v.shape)}")


def apply_moe_addon(
    model: torch.nn.Module,
    manifest: MoEAddonManifest,
    addon_path: Path | str,
    *,
    verify_sha: bool = True,
) -> dict[str, Any]:
    """Upcycle ``model`` (a dense, base-loaded ``CrowdV1WithEngram``) in place and load the add-on.

    Must run before ``eval()``/``place_for_inference``. Freezes every parameter at the end.
    """
    from safetensors.torch import load_file

    from hivemind.swarm.crowdgpt_moe_opendiloco import (
        extract_moe_trainable_buckets,
        load_moe_trainable_buckets,
        upcycle_crowd_v1_to_sparse_moe,
    )

    addon_sha = verify_file_sha256(addon_path, manifest.addon_sha256, "add-on") if verify_sha else sha256_file(addon_path)
    split = manifest.split
    nfrag = int(split["num_router_fragments"])
    rep = upcycle_crowd_v1_to_sparse_moe(
        model,
        split_mode=str(split["split_mode"]),
        num_experts=int(split["num_experts"]),
        top_k=int(split["top_k"]),
        lora_rank=int(split["lora_rank"]),
        lora_alpha=float(split["lora_alpha"]),
        routing_mode=str(split["routing_mode"]),
        resident_expert_indices=tuple(int(i) for i in split["resident_expert_indices"]),
    )
    if int(rep.num_layers_upcycled) != int(split["num_layers_upcycled"]):
        raise AddonIntegrityError(
            f"upcycled {rep.num_layers_upcycled} layers, the add-on was trained with {split['num_layers_upcycled']}"
        )
    template = extract_moe_trainable_buckets(model, num_router_fragments=nfrag)
    saved = load_file(str(addon_path))
    check_buckets_against_template(saved, template)
    buckets = {k: saved[k].to(dtype=torch.float32) for k in template}
    expert_norm = sum(float(v.norm()) for k, v in buckets.items() if k.startswith("expert_"))
    if expert_norm == 0.0:
        raise AddonIntegrityError("every expert bucket is zero: the add-on would be a no-op")
    with torch.no_grad():
        load_moe_trainable_buckets(model, buckets, num_router_fragments=nfrag)
    readback = extract_moe_trainable_buckets(model, num_router_fragments=nfrag)
    worst = max(float((readback[k] - buckets[k]).abs().max()) for k in buckets)
    if worst > READBACK_TOLERANCE:
        raise AddonIntegrityError(f"add-on did not load: a bucket reads back {worst:.3g} away from the file")
    for p in model.parameters():
        p.requires_grad_(False)
    return {
        "addon_id": manifest.addon_id,
        "addon_sha256": addon_sha,
        "num_layers_upcycled": int(rep.num_layers_upcycled),
        "addon_parameters": int(sum(v.numel() for v in buckets.values())),
        "buckets": sorted(buckets),
        "readback_max_abs_diff": worst,
    }


def zero_router_layers(model: torch.nn.Module) -> int:
    """Number of MoE layers whose router weight is exactly zero (every expert ties; top-k is a device tie-break)."""
    from hivemind.swarm.crowdgpt_moe_opendiloco import CrowdLoRAMoEMLP

    return sum(1 for blk in model.blocks
               if isinstance(blk.mlp, CrowdLoRAMoEMLP) and not bool(blk.mlp.router.weight.detach().any()))


def device_tie_pair(device: torch.device | str, dtype: torch.dtype, num_experts: int, top_k: int) -> list[int]:
    """The experts top-k picks on ``device``/``dtype`` when every router logit is zero (the routing's own expression)."""
    scores = torch.sigmoid(torch.zeros((4, int(num_experts)), device=device, dtype=dtype))
    idx = torch.topk(scores, k=int(top_k), dim=-1).indices
    rows = {tuple(int(i) for i in r) for r in idx.tolist()}
    if len(rows) != 1:
        raise AddonIntegrityError(f"top-k tie-break is not stable on {device}: {sorted(rows)}")
    return list(next(iter(rows)))


def check_zero_router_tie(model: torch.nn.Module, manifest: MoEAddonManifest) -> dict[str, Any]:
    """A zero router routes every token by the device's tie-break, so the served experts depend on the device.

    Passes when no router is zero, or when this device and dtype pick the pair recorded in the manifest
    (``evidence.judged_runtime.zero_router_tie_pair``, measured where the add-on was judged). Fails closed otherwise.
    """
    n_zero = zero_router_layers(model)
    if n_zero == 0:
        return {"ok": True, "zero_router_layers": 0}
    ref = model.blocks[0].mlp.router.weight
    pair = device_tie_pair(ref.device, ref.dtype, int(manifest.split["num_experts"]), int(manifest.split["top_k"]))
    judged = (manifest.evidence.get("judged_runtime") or {}).get("zero_router_tie_pair")
    ok = judged is not None and sorted(int(i) for i in judged) == sorted(pair)
    return {"ok": ok, "zero_router_layers": n_zero, "tie_pair_here": pair, "judged_tie_pair": judged,
            "device": str(ref.device), "dtype": str(ref.dtype)}


def pinned_base_urls(manifest: MoEAddonManifest) -> dict[str, str]:
    """Hugging Face URLs of the exact base revision the add-on was judged on (never ``main``)."""
    root = f"https://huggingface.co/{manifest.base_hf_repo}/resolve/{manifest.base_hf_revision}"
    return {"weights": f"{root}/model.safetensors", "tokenizer": f"{root}/tokenizer.json"}


def fetch_pinned_base(manifest: MoEAddonManifest, cache_dir: Path | str, timeout_s: float = 1800.0) -> dict[str, Path]:
    """Download the pinned base weights and tokenizer into ``cache_dir`` if absent, and verify both sha256."""
    import urllib.request

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    tag = manifest.base_hf_revision[:12]
    out = {"weights": cache / f"model_{tag}.safetensors", "tokenizer": cache / f"tokenizer_{tag}.json"}
    for key, url in pinned_base_urls(manifest).items():
        dest = out[key]
        if not dest.exists():
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=timeout_s) as resp, open(tmp, "wb") as fh:  # noqa: S310 (fixed https host)
                while True:
                    b = resp.read(1 << 20)
                    if not b:
                        break
                    fh.write(b)
            tmp.replace(dest)
    verify_file_sha256(out["weights"], manifest.base_weights_sha256, "base weights")
    verify_file_sha256(out["tokenizer"], manifest.tokenizer_sha256, "tokenizer")
    return out


def build_crowd_v1_moe_model(
    manifest: MoEAddonManifest,
    base_weights: Path | str,
    addon_path: Path | str | None,
    *,
    build_dtype: torch.dtype = torch.float32,
    verify_base_sha: bool = True,
    cfg: Any | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Dense Crowd-v1 on the pinned base, then (if ``addon_path``) upcycled with the add-on loaded. Not yet placed.

    With ``addon_path=None`` the model is the plain dense base (the reference arm for acceptance checks).
    """
    from hivemind.swarm.crowdgpt_live_runtime import (
        CrowdV1RuntimeConfig,
        CrowdV1WithEngram,
        load_crowd_v1_safetensors_into_model,
    )

    if verify_base_sha:
        verify_file_sha256(base_weights, manifest.base_weights_sha256, "base weights")
    prev = torch.get_default_dtype()
    torch.set_default_dtype(build_dtype)
    try:
        model = CrowdV1WithEngram(cfg if cfg is not None else CrowdV1RuntimeConfig())
    finally:
        torch.set_default_dtype(prev)
    base_sha = load_crowd_v1_safetensors_into_model(model, base_weights)
    info: dict[str, Any] = {"base_hf_revision": manifest.base_hf_revision, "base_weights_sha256": base_sha, "addon": None}
    if addon_path is not None:
        info["addon"] = apply_moe_addon(model, manifest, addon_path)
    else:
        for p in model.parameters():
            p.requires_grad_(False)
    model.eval()
    return model, info
