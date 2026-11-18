# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HM-S33-02 — Multi-Bee Debate & Cross-Verification Arena over AMCP Bus.

Coordinates a 3-Bee scientific & formal cross-verification debate on ``hivemind.amcp.bee_coord.v1``:
1. ``hivemind/bee-stars`` proposes an astrophysical hypothesis or orbital/transit model.
2. ``hivemind/bee-rusty-sundials`` verifies numerical ODE/PDE stability and physical conservation laws.
3. ``hivemind/bee-lean-prover`` verifies formal algebraic/logical soundness and boundary invariants.
Enforces fail-closed veto: if either the numerical solver bee or the Lean 4 prover bee flags an invariant
violation, the arena rejects the hypothesis regardless of narrative fluency.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class BeeDebateTurn:
    """Single Bee's structured assessment in an AMCP cross-verification debate."""

    bee_id: str
    role: str
    passed_invariants: bool
    confidence: float
    residual_error: float
    evidence_summary: str

    def __post_init__(self) -> None:
        if not isinstance(self.bee_id, str) or not self.bee_id.strip():
            raise ValueError("bee_id must be a non-empty string")
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError("role must be a non-empty string")
        if not isinstance(self.passed_invariants, bool):
            raise ValueError("passed_invariants must be a boolean")
        for num_field in ("confidence", "residual_error"):
            val = getattr(self, num_field)
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"{num_field} must be a number")
        if not isinstance(self.evidence_summary, str):
            raise ValueError("evidence_summary must be a string")


@dataclass(frozen=True)
class MultiBeeDebateVerdict:
    """Consensus & fail-closed verification verdict from the Multi-Bee AMCP Arena."""

    subject_id: str
    accepted: bool
    vetoed_by: tuple[str, ...]
    bayesian_posterior_confidence: float
    max_residual_error: float
    amcp_topic: str
    turns: tuple[BeeDebateTurn, ...]
    transcript_sha256: str

    def __post_init__(self) -> None:
        for s_field in ("subject_id", "amcp_topic"):
            val = getattr(self, s_field)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{s_field} must be a non-empty string")
        if not isinstance(self.accepted, bool):
            raise ValueError("accepted must be a boolean")
        if not isinstance(self.vetoed_by, tuple) or any(not isinstance(v, str) or not v.strip() for v in self.vetoed_by):
            raise ValueError("vetoed_by must be a tuple of non-empty strings")
        if (
            isinstance(self.bayesian_posterior_confidence, bool)
            or not isinstance(self.bayesian_posterior_confidence, (int, float))
            or not math.isfinite(float(self.bayesian_posterior_confidence))
            or not (0.0 <= float(self.bayesian_posterior_confidence) <= 1.0)
        ):
            raise ValueError("bayesian_posterior_confidence must be a finite number in [0.0, 1.0]")
        if (
            isinstance(self.max_residual_error, bool)
            or not isinstance(self.max_residual_error, (int, float))
            or math.isnan(float(self.max_residual_error))
            or float(self.max_residual_error) < 0.0
        ):
            raise ValueError("max_residual_error must be a non-negative number")
        if not isinstance(self.turns, tuple) or len(self.turns) < 2 or any(not isinstance(t, BeeDebateTurn) for t in self.turns):
            raise ValueError("turns must be a tuple of at least 2 BeeDebateTurn instances")
        if (
            not isinstance(self.transcript_sha256, str)
            or len(self.transcript_sha256) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in self.transcript_sha256)
        ):
            raise ValueError("transcript_sha256 must be a 64-character hex string")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MultiBeeAMCPDebateArena:
    """Fail-closed 3-Bee cross-verification arena over ``hivemind.amcp.bee_coord.v1``."""

    AMCP_TOPIC = "hivemind.amcp.bee_coord.v1"

    def __init__(self, *, residual_ceiling: float = 1e-4, min_posterior: float = 0.80) -> None:
        if (
            isinstance(residual_ceiling, bool)
            or not isinstance(residual_ceiling, (int, float))
            or not math.isfinite(residual_ceiling)
            or residual_ceiling < 0.0
        ):
            raise ValueError("residual_ceiling must be a non-negative finite number")
        if (
            isinstance(min_posterior, bool)
            or not isinstance(min_posterior, (int, float))
            or not math.isfinite(min_posterior)
            or not (0.0 < float(min_posterior) <= 1.0)
        ):
            raise ValueError("min_posterior must be a finite number in (0.0, 1.0]")
        self.residual_ceiling = float(residual_ceiling)
        self.min_posterior = float(min_posterior)

    @staticmethod
    def compute_bayesian_posterior(confidences: Sequence[float]) -> float:
        """Combine independent Bee confidences via log-odds Bayesian update with uniform prior."""
        if isinstance(confidences, (str, bytes)) or not isinstance(confidences, Sequence) or len(confidences) == 0:
            raise ValueError("confidences must be a non-empty sequence of numbers")
        if any(isinstance(c, bool) or not isinstance(c, (int, float)) for c in confidences):
            raise ValueError("confidences must contain only numbers")
        prod_p = 1.0
        prod_q = 1.0
        for c in confidences:
            val = float(c)
            if not math.isfinite(val):
                p = 1e-4
            else:
                p = min(max(val, 1e-4), 1.0 - 1e-4)
            prod_p *= p
            prod_q *= (1.0 - p)
        return prod_p / max(prod_p + prod_q, 1e-15)

    def adjudicate_debate(
        self,
        subject_id: str,
        turns: Sequence[BeeDebateTurn],
    ) -> MultiBeeDebateVerdict:
        """Adjudicate a multi-bee debate with strict fail-closed solver/prover veto."""
        if not isinstance(subject_id, str) or not subject_id.strip():
            raise ValueError("subject_id must be a non-empty string")
        if not isinstance(turns, (list, tuple)) or len(turns) < 2:
            raise ValueError("at least 2 specialist bees must participate in the AMCP debate")
        if any(not isinstance(t, BeeDebateTurn) for t in turns):
            raise ValueError("all turns must be BeeDebateTurn instances")

        vetoes: list[str] = []
        finite_residuals: list[float] = []
        has_nonfinite_residual = False
        for t in turns:
            res = float(t.residual_error)
            conf = float(t.confidence)
            if (
                not t.passed_invariants
                or not math.isfinite(res)
                or res < 0.0
                or res > self.residual_ceiling
                or not math.isfinite(conf)
                or conf < 0.0
                or conf > 1.0
            ):
                vetoes.append(t.bee_id)
            if math.isfinite(res):
                finite_residuals.append(res)
            else:
                has_nonfinite_residual = True

        posterior = self.compute_bayesian_posterior([t.confidence for t in turns])
        max_res = float("inf") if has_nonfinite_residual else max(finite_residuals, default=0.0)
        accepted = (len(vetoes) == 0) and (posterior >= self.min_posterior)

        canonical = json.dumps(
            {
                "subject_id": subject_id,
                "accepted": accepted,
                "vetoes": vetoes,
                "posterior": round(posterior, 8),
                "turns": [asdict(t) for t in turns],
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return MultiBeeDebateVerdict(
            subject_id=subject_id,
            accepted=accepted,
            vetoed_by=tuple(vetoes),
            bayesian_posterior_confidence=posterior,
            max_residual_error=max_res,
            amcp_topic=self.AMCP_TOPIC,
            turns=tuple(turns),
            transcript_sha256=digest,
        )
