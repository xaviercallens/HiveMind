# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HM-S32-02 — Lifelong Decentralized Ratchet & Anti-Forgetting Anchor Buffer for Continuous arXiv/JWST Ingestion.

Implements the nocturnal lifelong learning ratchet for Project HiveMind:
1. Maintains an immutable Golden Anchor Validation Buffer across core domains (Astrophysics, Coding, Lean 4, Physics).
2. Evaluates candidate nocturnal QLoRA / DiLoCo checkpoints against both the new daily corpus (arXiv, JWST, TESS)
   and the Golden Anchor Validation Buffer.
3. Enforces the strict Anti-Catastrophic-Forgetting Ratchet Invariant:
   a candidate adapter is promoted if and only if ``new_domain_loss < baseline_domain_loss`` AND
   ``(candidate_anchor_perplexity - baseline_anchor_perplexity) / baseline_anchor_perplexity <= max_anchor_drift_ratio``
   (default ``0.005`` = ``0.5%``).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

MAX_ANCHOR_PERPLEXITY_DRIFT_RATIO = 0.005


@dataclass(frozen=True)
class AnchorEvalSample:
    """Golden anchor evaluation sample protecting prior capabilities against catastrophic forgetting."""

    sample_id: str
    domain: str
    baseline_nll: float
    candidate_nll: float
    weight: float = 1.0


@dataclass(frozen=True)
class DailyCorpusIngestionBatch:
    """Daily nocturnal scientific/coding corpus batch (e.g. arXiv astro-ph, JWST MAST, TESS light curves)."""

    batch_id: str
    source: str
    document_count: int
    baseline_domain_nll: float
    candidate_domain_nll: float


@dataclass(frozen=True)
class LifelongRatchetVerdict:
    """Deterministic promotion/rejection verdict from the Lifelong Decentralized Ratchet."""

    checkpoint_id: str
    promoted: bool
    reason: str
    baseline_anchor_perplexity: float
    candidate_anchor_perplexity: float
    relative_anchor_perplexity_drift: float
    max_allowed_anchor_drift: float
    domain_nll_delta: float
    ewc_fisher_penalty: float
    ratchet_certificate_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LifelongDecentralizedRatchet:
    """Nocturnal anti-forgetting ratchet with EWC Fisher regularization and anchor perplexity gate."""

    def __init__(
        self,
        *,
        max_anchor_drift_ratio: float = MAX_ANCHOR_PERPLEXITY_DRIFT_RATIO,
        ewc_lambda: float = 0.15,
    ) -> None:
        if max_anchor_drift_ratio < 0.0 or not math.isfinite(max_anchor_drift_ratio):
            raise ValueError("max_anchor_drift_ratio must be finite and non-negative")
        if ewc_lambda < 0.0 or not math.isfinite(ewc_lambda):
            raise ValueError("ewc_lambda must be finite and non-negative")
        self.max_anchor_drift_ratio = float(max_anchor_drift_ratio)
        self.ewc_lambda = float(ewc_lambda)

    @staticmethod
    def weighted_perplexity(samples: Sequence[AnchorEvalSample], *, use_candidate: bool) -> float:
        if not samples:
            raise ValueError("samples must be non-empty")
        total_w = sum(max(0.0, s.weight) for s in samples)
        if total_w <= 0.0:
            raise ValueError("sum of sample weights must be positive")
        mean_nll = sum(
            max(0.0, s.weight) * (s.candidate_nll if use_candidate else s.baseline_nll)
            for s in samples
        ) / total_w
        return math.exp(min(mean_nll, 50.0))

    def compute_ewc_penalty(
        self,
        fisher_diag: Sequence[float],
        anchor_params: Sequence[float],
        candidate_params: Sequence[float],
    ) -> float:
        """Compute Elastic Weight Consolidation (EWC) quadratic penalty ``(lambda/2) * sum(F_i * (theta_i - theta*_i)^2)``."""
        if not (len(fisher_diag) == len(anchor_params) == len(candidate_params)):
            raise ValueError("fisher_diag, anchor_params, and candidate_params must have equal length")
        quad = sum(
            max(0.0, float(f)) * ((float(c) - float(a)) ** 2)
            for f, a, c in zip(fisher_diag, anchor_params, candidate_params)
        )
        return 0.5 * self.ewc_lambda * quad

    def evaluate_checkpoint(
        self,
        checkpoint_id: str,
        anchor_samples: Sequence[AnchorEvalSample],
        daily_batch: DailyCorpusIngestionBatch,
        *,
        fisher_diag: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
        anchor_params: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
        candidate_params: Sequence[float] = (0.01, -0.01, 0.005, 0.0),
    ) -> LifelongRatchetVerdict:
        """Evaluate whether ``checkpoint_id`` passes the anti-forgetting anchor gate and improves on ``daily_batch``."""
        base_ppl = self.weighted_perplexity(anchor_samples, use_candidate=False)
        cand_ppl = self.weighted_perplexity(anchor_samples, use_candidate=True)
        rel_drift = (cand_ppl - base_ppl) / max(base_ppl, 1e-12)
        ewc_pen = self.compute_ewc_penalty(fisher_diag, anchor_params, candidate_params)
        domain_delta = daily_batch.candidate_domain_nll + ewc_pen - daily_batch.baseline_domain_nll

        if rel_drift > self.max_anchor_drift_ratio:
            promoted = False
            reason = (
                f"REJECTED_ANCHOR_DRIFT: relative anchor perplexity drift {rel_drift:.6f} "
                f"exceeds ceiling {self.max_anchor_drift_ratio:.6f}"
            )
        elif domain_delta >= 0.0:
            promoted = False
            reason = (
                f"REJECTED_NO_DOMAIN_GAIN: EWC-adjusted domain NLL delta {domain_delta:.6f} >= 0"
            )
        else:
            promoted = True
            reason = "PROMOTED_MONOTONIC_RATCHET: anchor perplexity preserved and daily corpus NLL improved"

        canonical = json.dumps(
            {
                "checkpoint_id": checkpoint_id,
                "promoted": promoted,
                "base_ppl": round(base_ppl, 8),
                "cand_ppl": round(cand_ppl, 8),
                "domain_delta": round(domain_delta, 8),
            },
            sort_keys=True,
        )
        cert = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return LifelongRatchetVerdict(
            checkpoint_id=checkpoint_id,
            promoted=promoted,
            reason=reason,
            baseline_anchor_perplexity=base_ppl,
            candidate_anchor_perplexity=cand_ppl,
            relative_anchor_perplexity_drift=rel_drift,
            max_allowed_anchor_drift=self.max_anchor_drift_ratio,
            domain_nll_delta=domain_delta,
            ewc_fisher_penalty=ewc_pen,
            ratchet_certificate_sha256=cert,
        )
