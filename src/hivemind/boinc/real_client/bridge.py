# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Bridge between a BOINC result and HiveMind's existing verification and credit code.

BOINC validates by quorum (the same workunit is computed twice and the results compared). HiveMind replaces the
second full computation with a spot-check replay (``validator.py``): here the validator re-runs the workunit
itself and compares the volunteer's adapter state in signed 64-bit Q16 fixed point
(:func:`~hivemind.boinc.validator.verify_fixed_point_q16_digest`).

* same execution class (same GPU model and torch build) must match **exactly** (``max_ulp_drift=0``);
* a different class (CPU replay of a GPU result) may drift by at most one Q16 unit, the E12 rule.

Credit is the repository's ``verified_credit`` (cobblestones, capped by the class peak), so the BOINC credit
column and the HiveMind ledger cannot disagree.
"""

from __future__ import annotations

import base64
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hivemind.boinc.real_client.app import run
from hivemind.boinc.validator import verify_fixed_point_q16_digest
from hivemind.economy.compute_coin import CreditClaim, CreditGrant, verified_credit

# Measured peak of the class, in GFLOPS, for the credit cap. Conservative: well above what the tiny adapter
# reaches, so the cap only bites on an inflated claim.
CLASS_PEAK_GFLOPS = {"Tesla_T4": 8000.0, "cpu_x86_64": 200.0}


@dataclass(frozen=True)
class Verdict:
    wu_id: str
    host: str
    valid: bool
    reason: str
    same_class_exact: bool | None
    cross_class_max_ulp: float | None
    grant: dict[str, Any]
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def credit(self) -> float:
        return float(self.grant.get("cobblestones", 0.0))

    @property
    def pflops_seconds(self) -> float:
        return float(self.grant.get("cobblestones", 0.0)) * 432.0e-6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _state(doc: dict[str, Any]):
    import numpy as np
    import torch

    shapes = {"a": (int(doc["spec"]["rank"]), int(doc["spec"]["d_in"])), "b": (int(doc["spec"]["d_out"]), int(doc["spec"]["rank"]))}
    return {
        k: torch.from_numpy(np.frombuffer(base64.b64decode(doc["state_b64"][k]), dtype=np.float32).reshape(shapes[k]).copy())
        for k in ("a", "b")
    }


def _replay(spec: dict[str, Any], device: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as d:
        r = run(spec, device, Path(d) / "ckpt.pt")
    r["spec"] = spec
    return r


def _peak(fingerprint_class: str) -> float:
    return CLASS_PEAK_GFLOPS.get(fingerprint_class, 50.0)


def validate_result(spec: dict[str, Any], result: dict[str, Any], host: str, *, replay_devices: tuple[str, ...] = ("cpu", "cuda")) -> Verdict:
    """Spot-check one volunteer result against independent replays and compute the credit grant."""
    wu = spec["wu_id"]

    def reject(reason: str, exact: bool | None = None, ulp: float | None = None) -> Verdict:
        claim = CreditClaim(host, wu, 0.0, 1.0, 1.0, False, False)
        return Verdict(wu, host, False, reason, exact, ulp, verified_credit(claim).to_dict())

    if result.get("wu_id") != wu or result.get("app") != "hivemind_inner_loop":
        return reject("result is not for this workunit/app")
    if not (result.get("final_loss", float("inf")) < result.get("initial_loss", 0.0)):
        return reject("no learning: final loss is not below initial loss")
    try:
        claimed = _state({**result, "spec": spec})
    except Exception as exc:  # malformed upload is a rejection, never a crash of the validator
        return reject(f"unreadable state: {exc}")

    volunteer_class = result["fingerprint"]["accelerator_class"]
    exact: bool | None = None
    worst_ulp = 0.0
    cross_seen = False
    for dev in replay_devices:
        if dev == "cuda":
            import torch

            if not torch.cuda.is_available():
                continue
        ref = _replay(spec, dev)
        same_class = ref["fingerprint"]["accelerator_class"] == volunteer_class and ref["fingerprint"]["torch_build"] == result["fingerprint"]["torch_build"]
        cmp = verify_fixed_point_q16_digest(claimed, _state(ref), max_ulp_drift=0 if same_class else 1)
        if same_class:
            exact = bool(cmp["exact_digest_match"])
            if not cmp["passed"]:
                return reject(f"same-class replay on {dev} differs (max Q16 ulp {cmp['max_ulp_diff']})", exact, float(cmp["max_ulp_diff"]))
        else:
            cross_seen = True
            worst_ulp = max(worst_ulp, float(cmp["max_ulp_diff"]))
            if not cmp["passed"]:
                return reject(f"cross-class replay on {dev} differs by {cmp['max_ulp_diff']} Q16 units (> 1)", exact, float(cmp["max_ulp_diff"]))
    claim = CreditClaim(
        host, wu, float(result["flops_claimed"]), max(float(result["wall_seconds"]), 1e-6), _peak(volunteer_class), True, True
    )
    grant: CreditGrant = verified_credit(claim)
    return Verdict(wu, host, True, "verified", exact, worst_ulp if cross_seen else None, grant.to_dict())
