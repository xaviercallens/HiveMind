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
from typing import Any, Optional, Sequence

MAX_ANCHOR_PERPLEXITY_DRIFT_RATIO = 0.005


@dataclass(frozen=True)
class AnchorEvalSample:
    """Golden anchor evaluation sample protecting prior capabilities against catastrophic forgetting."""

    sample_id: str
    domain: str
    baseline_nll: float
    candidate_nll: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")
        if not isinstance(self.domain, str) or not self.domain.strip():
            raise ValueError("domain must be a non-empty string")
        for field_name in ("baseline_nll", "candidate_nll", "weight"):
            val = getattr(self, field_name)
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"{field_name} must be a number")


@dataclass(frozen=True)
class DailyCorpusIngestionBatch:
    """Daily nocturnal scientific/coding corpus batch (e.g. arXiv astro-ph, JWST MAST, TESS light curves)."""

    batch_id: str
    source: str
    document_count: int
    baseline_domain_nll: float
    candidate_domain_nll: float

    def __post_init__(self) -> None:
        if not isinstance(self.batch_id, str) or not self.batch_id.strip():
            raise ValueError("batch_id must be a non-empty string")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")
        if isinstance(self.document_count, bool) or not isinstance(self.document_count, int) or self.document_count < 1:
            raise ValueError("document_count must be a positive integer")
        for field_name in ("baseline_domain_nll", "candidate_domain_nll"):
            val = getattr(self, field_name)
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"{field_name} must be a number")


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
    ewc_fisher_penalty: Optional[float]
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
        if (
            isinstance(max_anchor_drift_ratio, bool)
            or not isinstance(max_anchor_drift_ratio, (int, float))
            or max_anchor_drift_ratio < 0.0
            or not math.isfinite(max_anchor_drift_ratio)
        ):
            raise ValueError("max_anchor_drift_ratio must be finite and non-negative")
        if (
            isinstance(ewc_lambda, bool)
            or not isinstance(ewc_lambda, (int, float))
            or ewc_lambda < 0.0
            or not math.isfinite(ewc_lambda)
        ):
            raise ValueError("ewc_lambda must be finite and non-negative")
        self.max_anchor_drift_ratio = float(max_anchor_drift_ratio)
        self.ewc_lambda = float(ewc_lambda)

    @staticmethod
    def weighted_perplexity(samples: Sequence[AnchorEvalSample], *, use_candidate: bool) -> float:
        if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence) or not samples:
            raise ValueError("samples must be non-empty")
        for s in samples:
            if not isinstance(s, AnchorEvalSample):
                raise TypeError("each sample must be an AnchorEvalSample")
            if not math.isfinite(float(s.weight)):
                raise ValueError(f"non-finite weight for anchor sample {s.sample_id!r}: {s.weight!r}")
            value = s.candidate_nll if use_candidate else s.baseline_nll
            if not math.isfinite(value):
                raise ValueError(f"non-finite NLL for anchor sample {s.sample_id!r}: {value!r}")
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
        for seq_name, seq in (
            ("fisher_diag", fisher_diag),
            ("anchor_params", anchor_params),
            ("candidate_params", candidate_params),
        ):
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in seq):
                raise ValueError(f"{seq_name} must contain only finite numbers")
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
        fisher_diag: Optional[Sequence[float]] = None,
        anchor_params: Optional[Sequence[float]] = None,
        candidate_params: Optional[Sequence[float]] = None,
    ) -> LifelongRatchetVerdict:
        """Evaluate whether ``checkpoint_id`` passes the anti-forgetting anchor gate and improves on ``daily_batch``.

        The EWC penalty is computed only when ``fisher_diag``, ``anchor_params`` and ``candidate_params`` are all
        supplied; when all three are absent it is ``None`` and the domain delta is the raw NLL change. No parameter
        vector is invented. Supplying only some of them raises ``ValueError``.
        """
        if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
            raise ValueError("checkpoint_id must be a non-empty string")
        if not isinstance(daily_batch, DailyCorpusIngestionBatch):
            raise ValueError("daily_batch must be a DailyCorpusIngestionBatch")
        for name, value in (
            ("candidate_domain_nll", daily_batch.candidate_domain_nll),
            ("baseline_domain_nll", daily_batch.baseline_domain_nll),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        ewc_inputs = (fisher_diag, anchor_params, candidate_params)
        ewc_pen: Optional[float]
        if all(v is None for v in ewc_inputs):
            ewc_pen = None
        elif any(v is None for v in ewc_inputs):
            raise ValueError("fisher_diag, anchor_params and candidate_params must be supplied together")
        else:
            ewc_pen = self.compute_ewc_penalty(fisher_diag, anchor_params, candidate_params)  # type: ignore[arg-type]
        base_ppl = self.weighted_perplexity(anchor_samples, use_candidate=False)
        cand_ppl = self.weighted_perplexity(anchor_samples, use_candidate=True)
        rel_drift = (cand_ppl - base_ppl) / max(base_ppl, 1e-12)
        domain_delta = daily_batch.candidate_domain_nll + (ewc_pen or 0.0) - daily_batch.baseline_domain_nll

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
