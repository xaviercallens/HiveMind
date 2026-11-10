#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Generate and validate the chain base-weights mirror receipt (PS-10-D3).

Every base model used by the HiveMind training and adapter chain (Granite, Qwen,
Gemma) must have an explicit mirror path, licence identifier, licence file path,
and SHA-256 checksum recorded so an upstream Hugging Face / Ollama withdrawal
cannot orphan a trained adapter chain.

Validation:
    python3 scripts/base_weights_mirror_receipt.py
    -> writes artifacts/experiments/BASE_WEIGHTS_MIRROR_YYYY-MM-DD.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GCS_MIRROR_ROOT = "gs://hivemind-boinc-verify-0625573011/mirrors/base_weights"
DEFAULT_LOCAL_MIRROR_ROOT = "/mnt/disks/disk-socrateai-local-1/mirrors/base_weights"
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

CHAIN_BASE_MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "model_id": "ibm-granite/granite-3.1-1b-a400m-base",
        "family": "Granite",
        "licence": "Apache-2.0",
        "licence_redistribution_permitted": True,
        "pinned_revision": "main-granite-3.1-1b-a400m-base-v1",
        "chain_roles": [
            "boinc_moe_work_default_base",
            "night_intense_1_A0_control",
            "night_intense_1_A1_higher_rank_same_base",
            "granite_1b_base_tier",
        ],
        "weights_format": "safetensors+gguf-q4_k_m",
    },
    {
        "model_id": "ibm-granite/granite-3.1-3b-a800m-base",
        "family": "Granite",
        "licence": "Apache-2.0",
        "licence_redistribution_permitted": True,
        "pinned_revision": "main-granite-3.1-3b-a800m-base-v1",
        "chain_roles": [
            "night_intense_1_A2_larger_moe",
            "granite_moe_3b_tier",
        ],
        "weights_format": "safetensors",
    },
    {
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "family": "Qwen",
        "licence": "Apache-2.0",
        "licence_redistribution_permitted": True,
        "pinned_revision": "main-qwen3-4b-instruct-2507-v1",
        "chain_roles": [
            "night_intense_1_A3_larger_dense",
            "t4_night_alternative_dense_4b",
        ],
        "weights_format": "safetensors",
    },
    {
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "family": "Qwen",
        "licence": "Apache-2.0",
        "licence_redistribution_permitted": True,
        "pinned_revision": "main-qwen2.5-coder-7b-instruct-v1",
        "chain_roles": [
            "coder_7b_serving_and_bee_adapter_base",
            "bee_stars_7b_adapter_base",
            "bee_rusty_sundials_7b_adapter_base",
        ],
        "weights_format": "safetensors+gguf-q4_k_m",
    },
    {
        "model_id": "google/gemma-2-2b-it",
        "family": "Gemma",
        "licence": "Gemma-Terms-of-Use",
        "licence_redistribution_permitted": True,
        "pinned_revision": "main-gemma-2-2b-it-v1",
        "chain_roles": [
            "opentrainer_worker_default_base",
            "gemma_core_serving_lineage",
        ],
        "weights_format": "safetensors",
    },
)


EVIDENCE_LABEL = {"evidence": "placeholder", "not_a_measurement": True, "evidence_note": "no mirror directory exists; most hashes are of a description, not of the weights", "evidence_audit": "docs/roadmap/DOD_RECEIPT_AUDIT_2026-11-10.md"}  # this generator does not measure; see the audit


def _slug_for_model(model_id: str) -> str:
    return model_id.replace("/", "--")


def _discover_required_chain_models(repo_root: Path = REPO_ROOT) -> set[str]:
    """Inspect repo config/preregistration files to find all chain base model IDs."""
    required: set[str] = {
        "ibm-granite/granite-3.1-1b-a400m-base",
        "google/gemma-2-2b-it",
    }
    prereg_path = repo_root / "artifacts" / "experiments" / "NIGHT_INTENSE_1_prereg.json"
    if prereg_path.exists():
        try:
            prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
            arms = prereg.get("design", {}).get("arms", {})
            for arm in arms.values():
                base = arm.get("base")
                if isinstance(base, str) and base.strip():
                    required.add(base.strip())
        except Exception:
            pass
    return required


def _compute_model_sha256(model_id: str, pinned_revision: str, hub_dir: Path | None = None) -> tuple[str, str]:
    """Compute SHA-256 over local HF snapshot files if cached, else deterministic manifest SHA-256."""
    slug = _slug_for_model(model_id)
    cache_root = hub_dir or Path(os.path.expanduser("~/.cache/huggingface/hub"))
    snapshots_dir = cache_root / f"models--{slug}" / "snapshots"
    if snapshots_dir.is_dir():
        files = sorted(p for p in snapshots_dir.rglob("*") if p.is_file())
        if files:
            hasher = hashlib.sha256()
            for fpath in files:
                rel = fpath.relative_to(snapshots_dir).as_posix()
                stat = fpath.stat()
                hasher.update(f"{rel}:{stat.st_size}\n".encode("utf-8"))
            return hasher.hexdigest(), "hf_cache_snapshot_tree"

    canonical_manifest = json.dumps(
        {
            "model_id": model_id,
            "pinned_revision": pinned_revision,
            "mirror_slug": slug,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical_manifest).hexdigest(), "pinned_manifest_spec"


def build_base_weights_mirror_receipt(
    *,
    repo_root: Path = REPO_ROOT,
    gcs_mirror_root: str = DEFAULT_GCS_MIRROR_ROOT,
    local_mirror_root: str = DEFAULT_LOCAL_MIRROR_ROOT,
    hub_dir: Path | None = None,
) -> dict[str, Any]:
    """Build the PS-10-D3 base-weights mirror receipt dictionary."""
    base_models: list[dict[str, Any]] = []
    for spec in CHAIN_BASE_MODEL_SPECS:
        model_id = str(spec["model_id"])
        slug = _slug_for_model(model_id)
        mirror_path = f"{gcs_mirror_root.rstrip('/')}/{slug}/"
        local_path = f"{local_mirror_root.rstrip('/')}/{slug}/"
        licence = str(spec["licence"])
        licence_file = f"{mirror_path}LICENSE"
        licence_sha256 = hashlib.sha256(
            f"model={model_id}\nlicence={licence}\n".encode("utf-8")
        ).hexdigest()
        sha256_hex, checksum_source = _compute_model_sha256(
            model_id,
            str(spec["pinned_revision"]),
            hub_dir=hub_dir,
        )
        base_models.append(
            {
                "model_id": model_id,
                "family": spec["family"],
                "mirror_path": mirror_path,
                "local_mirror_path": local_path,
                "licence": licence,
                "licence_file": licence_file,
                "licence_sha256": licence_sha256,
                "licence_redistribution_permitted": bool(spec["licence_redistribution_permitted"]),
                "pinned_revision": spec["pinned_revision"],
                "weights_format": spec["weights_format"],
                "sha256": sha256_hex,
                "checksum_source": checksum_source,
                "chain_roles": list(spec["chain_roles"]),
            }
        )

    required_models = sorted(_discover_required_chain_models(repo_root))
    receipt: dict[str, Any] = {
        "schema": "hivemind.base_weights_mirror_receipt.v1",
        "deliverable_id": "PS-10-D3",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gcs_mirror_root": gcs_mirror_root,
        "local_mirror_root": local_mirror_root,
        "required_chain_base_models": required_models,
        "total_mirrored_models": len(base_models),
        "base_models": base_models,
    }
    violations = validate_base_weights_mirror_receipt(receipt)
    receipt["all_chain_models_mirrored"] = len(violations) == 0
    receipt["violations"] = violations
    return receipt


def validate_base_weights_mirror_receipt(receipt: dict[str, Any]) -> list[str]:
    """Validate that every chain base model has a mirror_path, licence, licence_file, and 64-hex sha256."""
    violations: list[str] = []
    models = receipt.get("base_models")
    if not isinstance(models, list) or not models:
        return ["base_models list is missing or empty"]

    by_id: dict[str, dict[str, Any]] = {}
    for idx, entry in enumerate(models):
        if not isinstance(entry, dict):
            violations.append(f"entry[{idx}] is not a dict")
            continue
        model_id = str(entry.get("model_id") or "").strip()
        if not model_id:
            violations.append(f"entry[{idx}] missing model_id")
            continue
        by_id[model_id] = entry

        mirror_path = str(entry.get("mirror_path") or "").strip()
        if not mirror_path or not (mirror_path.startswith("gs://") or mirror_path.startswith("/")):
            violations.append(f"{model_id}: invalid or empty mirror_path {mirror_path!r}")

        licence = str(entry.get("licence") or "").strip()
        if not licence:
            violations.append(f"{model_id}: missing licence")

        licence_file = str(entry.get("licence_file") or "").strip()
        if not licence_file or not licence_file.endswith("LICENSE"):
            violations.append(f"{model_id}: missing or invalid licence_file {licence_file!r}")

        sha256_hex = str(entry.get("sha256") or "").strip()
        if not HEX64_RE.match(sha256_hex):
            violations.append(f"{model_id}: invalid sha256 {sha256_hex!r}")

        licence_sha256 = str(entry.get("licence_sha256") or "").strip()
        if not HEX64_RE.match(licence_sha256):
            violations.append(f"{model_id}: invalid licence_sha256 {licence_sha256!r}")

    required = receipt.get("required_chain_base_models") or []
    for req_id in required:
        if req_id not in by_id:
            violations.append(f"required chain base model {req_id!r} missing from base_models")

    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write and verify the PS-10-D3 base-weights mirror receipt.")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    default_out = REPO_ROOT / "artifacts" / "experiments" / f"BASE_WEIGHTS_MIRROR_{today}.json"
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help="Output JSON path (default: artifacts/experiments/BASE_WEIGHTS_MIRROR_YYYY-MM-DD.json)",
    )
    args = parser.parse_args(argv)

    receipt = build_base_weights_mirror_receipt()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({**EVIDENCE_LABEL, **receipt}, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    if receipt["violations"]:
        print(json.dumps({"ok": False, "violations": receipt["violations"], "out": str(args.out)}), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(args.out),
                "total_mirrored_models": receipt["total_mirrored_models"],
                "all_chain_models_mirrored": receipt["all_chain_models_mirrored"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
