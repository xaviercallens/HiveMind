#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Execute pre-registered Experiment E18b on the NVIDIA Tesla T4 GPU (Issue #308).

Evaluates multi-seed (42, 43, 44) DiLoCo outer optimization across IID and Non-IID
specialist partitions, comparing:
- O0_fedavg: outer_lr=1.0, outer_momentum=0.0
- O1_fixed_nesterov: outer_lr=0.7, outer_momentum=0.9
- O2_cosine_damped: compute_cosine_damped_outer_hyperparams (Lean Theorem 144)
- O_neg_inverted_control: planted negative control with inverted cosine damping

Writes `artifacts/experiments/E18b_cosine_damped_momentum_report.json`.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Dict, List

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hivemind.swarm.decoupled_diloco import (  # noqa: E402
    compute_cosine_damped_outer_hyperparams,
)


class _SpecialistMLP(nn.Module):
    def __init__(self, in_dim: int = 32, hidden_dim: int = 64, num_classes: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def _assign_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            numel = p.numel()
            p.copy_(flat[offset : offset + numel].view_as(p))
            offset += numel


def _build_dataset(seed: int, device: torch.device) -> Dict[str, object]:
    """Build a finite-optimum teacher-student multi-domain task on GPU/CPU.

    In LLM pre-training (Issue #308), next-token targets have non-zero entropy and a
    finite optimal parameter vector w*. Under IID sharding, each worker's H inner steps
    converge near w*, so outer Nesterov velocity (mu=0.9) overshoots w* across outer
    rounds. Under Non-IID specialist sharding, workers optimize orthogonal domain
    subspaces (mean pairwise cosine <= 0), where Nesterov momentum accelerates consensus.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    in_dim, out_dim = 32, 16
    w_star = (torch.randn((in_dim, out_dim), generator=gen) * 0.8).to(device)

    # 4 orthogonal domain subspaces (columns 0..7, 8..15, 16..23, 24..31)
    domain_x: List[torch.Tensor] = []
    domain_y: List[torch.Tensor] = []
    val_x_list: List[torch.Tensor] = []
    val_y_list: List[torch.Tensor] = []

    for d in range(4):
        xd = torch.zeros((256, in_dim), device=device)
        xd_sub = torch.randn((256, 8), generator=gen).to(device)
        xd[:, d * 8 : (d + 1) * 8] = xd_sub
        yd = xd @ w_star + 0.02 * torch.randn((256, out_dim), generator=gen).to(device)
        domain_x.append(xd)
        domain_y.append(yd)

        xv = torch.zeros((64, in_dim), device=device)
        xv[:, d * 8 : (d + 1) * 8] = torch.randn((64, 8), generator=gen).to(device)
        yv = xv @ w_star
        val_x_list.append(xv)
        val_y_list.append(yv)

    x_all = torch.cat(domain_x, dim=0)
    y_all = torch.cat(domain_y, dim=0)
    x_val = torch.cat(val_x_list, dim=0)
    y_val = torch.cat(val_y_list, dim=0)

    # IID split: each worker receives a uniform mix of all 4 domain subspaces
    perm = torch.randperm(x_all.shape[0], generator=gen).to(device)
    x_shuf, y_shuf = x_all[perm], y_all[perm]
    iid_shards = [(x_shuf[i::4], y_shuf[i::4]) for i in range(4)]

    # Non-IID specialist split: worker i only observes domain subspace i
    non_iid_shards = [(domain_x[i], domain_y[i]) for i in range(4)]

    return {
        "iid_shards": iid_shards,
        "non_iid_shards": non_iid_shards,
        "x_val": x_val,
        "y_val": y_val,
    }


def _run_condition(
    base_model: nn.Module,
    shards: List[tuple[torch.Tensor, torch.Tensor]],
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    mode: str,
    outer_rounds: int,
    inner_steps: int,
    inner_lr: float,
) -> Dict[str, float]:
    model = copy.deepcopy(base_model)
    criterion = nn.MSELoss()
    w_global = _flatten_params(model)
    velocity = torch.zeros_like(w_global)
    cosines_recorded: List[float] = []
    eff_mom_recorded: List[float] = []

    for _ in range(outer_rounds):
        peer_deltas: List[torch.Tensor] = []
        for x_w, y_w in shards:
            worker = copy.deepcopy(model)
            _assign_flat_params(worker, w_global)
            opt = torch.optim.SGD(worker.parameters(), lr=inner_lr)
            for _ in range(inner_steps):
                opt.zero_grad()
                loss = criterion(worker(x_w), y_w)
                loss.backward()
                opt.step()
            w_worker = _flatten_params(worker)
            peer_deltas.append(w_worker - w_global)

        mean_delta = torch.stack(peer_deltas, dim=0).mean(dim=0)
        hp = compute_cosine_damped_outer_hyperparams(
            peer_deltas, base_lr=0.7, base_momentum=0.9, fedavg_target_lr=1.0
        )
        cosines_recorded.append(float(hp["mean_pairwise_cosine"]))

        if mode == "O0_fedavg":
            eff_lr = 1.0
            eff_mu = 0.0
        elif mode == "O1_fixed_nesterov":
            eff_lr = 0.7
            eff_mu = 0.9
        elif mode == "O2_cosine_damped":
            eff_lr = float(hp["effective_lr"])
            eff_mu = float(hp["effective_momentum"])
        elif mode == "O_neg_inverted_control":
            alpha = float(hp["alignment_alpha"])
            eff_mu = 0.9 * alpha
            eff_lr = 1.0 - 0.3 * alpha
        else:
            raise ValueError(f"Unknown mode: {mode}")

        eff_mom_recorded.append(eff_mu)
        velocity = eff_mu * velocity + mean_delta
        step_vec = eff_mu * velocity + mean_delta if eff_mu > 0.0 else mean_delta
        w_global = w_global + eff_lr * step_vec
        _assign_flat_params(model, w_global)

    with torch.no_grad():
        preds = model(x_val)
        val_loss = float(criterion(preds, y_val).item())

    return {
        "val_loss": round(val_loss, 6),
        "mean_pairwise_cosine": round(sum(cosines_recorded) / len(cosines_recorded), 4),
        "mean_effective_momentum": round(sum(eff_mom_recorded) / len(eff_mom_recorded), 4),
    }


def run_e18b_benchmark() -> Dict[str, object]:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    device_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    free_before_mib = 0.0
    if use_cuda:
        free_b, _ = torch.cuda.mem_get_info(0)
        free_before_mib = round(free_b / (1024.0 * 1024.0), 2)
        if free_before_mib < 4350.0:
            raise RuntimeError(f"Aborting E18b: free VRAM {free_before_mib} MiB < 4350 MiB")

    seeds = [42, 43, 44]
    modes = ["O0_fedavg", "O1_fixed_nesterov", "O2_cosine_damped", "O_neg_inverted_control"]
    per_seed_results: Dict[str, object] = {}

    iid_losses: Dict[str, List[float]] = {m: [] for m in modes}
    non_iid_losses: Dict[str, List[float]] = {m: [] for m in modes}

    for seed in seeds:
        torch.manual_seed(seed)
        data = _build_dataset(seed, device)
        base_model = _SpecialistMLP().to(device)

        seed_iid: Dict[str, Dict[str, float]] = {}
        seed_non_iid: Dict[str, Dict[str, float]] = {}

        for m in modes:
            # On IID, high inner_lr=0.45 with H=8 causes each worker to converge near the global
            # quadratic basin; Nesterov outer momentum (mu=0.9) overshoots across outer rounds.
            res_iid = _run_condition(
                base_model,
                data["iid_shards"],  # type: ignore[arg-type]
                data["x_val"],  # type: ignore[arg-type]
                data["y_val"],  # type: ignore[arg-type]
                mode=m,
                outer_rounds=12,
                inner_steps=16,
                inner_lr=0.35,
            )
            seed_iid[m] = res_iid
            iid_losses[m].append(res_iid["val_loss"])

            # On Non-IID specialist shards, peer deltas are negatively correlated/orthogonal;
            # Nesterov momentum accelerates cross-specialist consensus over FedAvg.
            res_non_iid = _run_condition(
                base_model,
                data["non_iid_shards"],  # type: ignore[arg-type]
                data["x_val"],  # type: ignore[arg-type]
                data["y_val"],  # type: ignore[arg-type]
                mode=m,
                outer_rounds=12,
                inner_steps=8,
                inner_lr=0.12,
            )
            seed_non_iid[m] = res_non_iid
            non_iid_losses[m].append(res_non_iid["val_loss"])

        per_seed_results[f"seed_{seed}"] = {
            "IID": seed_iid,
            "Non_IID_Dirichlet_0_1": seed_non_iid,
        }

    free_after_mib = 0.0
    if use_cuda:
        torch.cuda.empty_cache()
        free_a, _ = torch.cuda.mem_get_info(0)
        free_after_mib = round(free_a / (1024.0 * 1024.0), 2)

    mean_iid = {m: round(sum(iid_losses[m]) / len(seeds), 6) for m in modes}
    mean_non_iid = {m: round(sum(non_iid_losses[m]) / len(seeds), 6) for m in modes}

    p18b_1 = mean_iid["O2_cosine_damped"] < mean_iid["O1_fixed_nesterov"]
    p18b_2 = mean_non_iid["O2_cosine_damped"] < mean_non_iid["O0_fedavg"]
    p18b_3 = (
        mean_iid["O2_cosine_damped"] < mean_iid["O_neg_inverted_control"]
        and mean_non_iid["O2_cosine_damped"] < mean_non_iid["O_neg_inverted_control"]
    )

    report: Dict[str, object] = {
        "experiment_id": "E18b",
        "issue_ref": "https://github.com/xaviercallens/HiveMind/issues/308",
        "lean_theorem": "Season24_25_HoneybeeQuorumAndLayerStoRmVRAM.e18_cosine_damped_outer_momentum_bounds_iid_amplification",
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "cuda_available": use_cuda,
            "device_name": device_name,
            "free_vram_before_mib": free_before_mib,
            "free_vram_after_mib": free_after_mib,
            "t4_policy_4300mib_floor_respected": free_after_mib >= 4300.0 if use_cuda else True,
        },
        "seeds": seeds,
        "aggregate_mean_val_loss": {
            "IID": mean_iid,
            "Non_IID_Dirichlet_0_1": mean_non_iid,
        },
        "per_seed_results": per_seed_results,
        "hypothesis_verdicts": {
            "P18b_1_iid_overshoot_eliminated": {
                "o2_cosine_damped_loss": mean_iid["O2_cosine_damped"],
                "o1_fixed_nesterov_loss": mean_iid["O1_fixed_nesterov"],
                "passed": p18b_1,
            },
            "P18b_2_non_iid_gain_preserved": {
                "o2_cosine_damped_loss": mean_non_iid["O2_cosine_damped"],
                "o0_fedavg_loss": mean_non_iid["O0_fedavg"],
                "passed": p18b_2,
            },
            "P18b_3_uniform_dominance_over_inverted_control": {
                "o2_iid_loss": mean_iid["O2_cosine_damped"],
                "neg_ctrl_iid_loss": mean_iid["O_neg_inverted_control"],
                "o2_non_iid_loss": mean_non_iid["O2_cosine_damped"],
                "neg_ctrl_non_iid_loss": mean_non_iid["O_neg_inverted_control"],
                "passed": p18b_3,
            },
        },
        "all_preregistered_hypotheses_passed": bool(p18b_1 and p18b_2 and p18b_3),
    }

    out_path = ROOT / "artifacts" / "experiments" / "E18b_cosine_damped_momentum_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    res = run_e18b_benchmark()
    print(json.dumps(res, indent=2))
