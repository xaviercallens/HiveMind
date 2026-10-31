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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MultiBeeAMCPDebateArena:
    """Fail-closed 3-Bee cross-verification arena over ``hivemind.amcp.bee_coord.v1``."""

    AMCP_TOPIC = "hivemind.amcp.bee_coord.v1"

    def __init__(self, *, residual_ceiling: float = 1e-4, min_posterior: float = 0.80) -> None:
        self.residual_ceiling = float(residual_ceiling)
        self.min_posterior = float(min_posterior)

    @staticmethod
    def compute_bayesian_posterior(confidences: Sequence[float]) -> float:
        """Combine independent Bee confidences via log-odds Bayesian update with uniform prior."""
        log_odds = 0.0
        for c in confidences:
            p = min(max(float(c), 1e-4), 1.0 - 1e-4)
            log_odds += (p / (1.0 - p))
        # Normalize log-odds product: P = prod(p_i) / (prod(p_i) + prod(1 - p_i))
        prod_p = 1.0
        prod_q = 1.0
        for c in confidences:
            p = min(max(float(c), 1e-4), 1.0 - 1e-4)
            prod_p *= p
            prod_q *= (1.0 - p)
        return prod_p / max(prod_p + prod_q, 1e-15)

    def adjudicate_debate(
        self,
        subject_id: str,
        turns: Sequence[BeeDebateTurn],
    ) -> MultiBeeDebateVerdict:
        """Adjudicate a multi-bee debate with strict fail-closed solver/prover veto."""
        if len(turns) < 2:
            raise ValueError("at least 2 specialist bees must participate in the AMCP debate")

        vetoes: list[str] = []
        for t in turns:
            if not t.passed_invariants or t.residual_error > self.residual_ceiling:
                vetoes.append(t.bee_id)

        posterior = self.compute_bayesian_posterior([t.confidence for t in turns])
        max_res = max(t.residual_error for t in turns)
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
