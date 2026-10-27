"""Training profiles for the QLoRA MoE app: the standard one for small GPUs and a larger one for GPUs with more memory.

BOINC hands a workunit to any host that can run one of the app's versions, and a workunit cannot name a plan class. The only way to
keep large workunits away from small GPUs is therefore a separate app whose single version has a high ``min_gpu_ram_mb``: the
scheduler then never offers it to a host below that threshold. Each profile is such an app, with its own plan class, its own
workunit-name prefix and its own adapter chain (a rank-16 adapter cannot continue a rank-8 one).

This module has no heavy imports so the server scripts, the work generator and the app can all read it.
"""

from __future__ import annotations

from typing import Any

STANDARD, BIG = "standard", "big"
TOKENS_PER_ROUND_REFERENCE = 80 * 2 * 128  # tokens of the standard round whose flops estimate (4.8e14) the project already uses
FLOPS_REFERENCE = 4.8e14

PROFILES: dict[str, dict[str, Any]] = {
    STANDARD: {
        "app": "hivemind_qlora_moe", "plan_class": "cuda_hivemind", "min_gpu_ram_mb": 3500, "prefix": "hmq_", "chain_stem": "latest_adapter",
        "seq_len": 128, "n_seq": 64, "inner_steps": 80, "segment_steps": 10, "micro_batch": 2, "rank": 8, "alpha": 16,
    },
    BIG: {
        "app": "hivemind_qlora_moe_big", "plan_class": "cuda_hivemind_big", "min_gpu_ram_mb": 10000, "prefix": "hmqb_", "chain_stem": "latest_adapter_big",
        "seq_len": 512, "n_seq": 128, "inner_steps": 80, "segment_steps": 10, "micro_batch": 4, "rank": 16, "alpha": 32,
    },
}
TRAIN_APPS: tuple[str, ...] = tuple(p["app"] for p in PROFILES.values())


def profile_for_app(app: str) -> dict[str, Any]:
    for p in PROFILES.values():
        if p["app"] == app:
            return p
    raise KeyError(f"not a training app: {app!r}")


def flops_estimate(profile: str) -> float:
    """Scaled from the project's reference estimate by tokens per round, so a bigger round is not under-budgeted."""
    p = PROFILES[profile]
    return FLOPS_REFERENCE * (p["inner_steps"] * p["micro_batch"] * p["seq_len"]) / TOKENS_PER_ROUND_REFERENCE


def profiles_for_gpu(total_mib: int) -> list[str]:
    """Profiles a GPU with this much memory may run, biggest first (empty when it is below even the standard floor)."""
    ranked = sorted(PROFILES, key=lambda n: -PROFILES[n]["min_gpu_ram_mb"])
    return [n for n in ranked if total_mib >= PROFILES[n]["min_gpu_ram_mb"]]
