"""Dance records: how an expert's verified results are reported and how routers recruit toward them.

A honeybee's waggle dance encodes a direction, a distance and a quality, recruits followers in proportion to that
quality, and fades when the source is exhausted. Here a record carries the task cluster (direction), the cost
(distance) and verified trials and successes (quality). The aggregator turns records into recruitment weights with
three rules, each with a Lean theorem in `HiveMindSpec.ALIX_QueenBee`:

- fading: a record older than the horizon has weight 0 (Theorem 129);
- influence cap: one record carries at most `cap` of weight, so `k` records carry at most `k * cap` (Theorems 130-131);
- verified only: `n` and `successes` count outcomes a verifier judged; the score is a lower confidence bound that
  cannot rise by padding trials (Theorems 132-133).

Records are HMAC-signed by the identity that reports them and must name the expert that actually served the task
(`served_by == expert_id`), so a producer cannot dance for someone else's work. Records hold aggregate counts only:
no prompts, no code.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from hivemind.queenbee import RECORD_SCHEMA

VERIFIER_KINDS: frozenset[str] = frozenset({"unit_tests", "numeric", "mcq", "label", "lean"})
Z95 = 1.959963984540054
SCALE = 1_000_000  # integer scale used by the Lean model; weights below are floats in [0, cap]


@dataclass(frozen=True)
class DanceRecord:
    """Aggregate, signed statistics about one expert on one task cluster for one epoch."""

    expert_id: str
    served_by: str
    cluster_id: str
    n: int
    successes: int
    cost_per_1k: float
    epoch: int
    verifier_kind: str
    signer_id: str
    signature_hex: str = ""

    def __post_init__(self) -> None:
        if not self.expert_id or not self.cluster_id or not self.signer_id:
            raise ValueError("expert_id, cluster_id and signer_id must be non-empty")
        if self.n < 0 or self.successes < 0 or self.successes > self.n:
            raise ValueError(f"need 0 <= successes <= n, got successes={self.successes} n={self.n}")
        if not math.isfinite(self.cost_per_1k) or self.cost_per_1k < 0.0:
            raise ValueError("cost_per_1k must be finite and >= 0")
        if self.epoch < 0:
            raise ValueError("epoch must be >= 0")
        if self.verifier_kind not in VERIFIER_KINDS:
            raise ValueError(f"verifier_kind must be one of {sorted(VERIFIER_KINDS)}")

    def payload(self) -> bytes:
        """Canonical bytes that are signed: every field except the signature, plus the schema id."""
        body = asdict(self)
        body.pop("signature_hex")
        body["schema"] = RECORD_SCHEMA
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def signed(self, secret: bytes) -> "DanceRecord":
        return replace(self, signature_hex=hmac.new(secret, self.payload(), hashlib.sha256).hexdigest())

    def verify(self, secret: bytes) -> bool:
        expected = hmac.new(secret, self.payload(), hashlib.sha256).hexdigest()
        return bool(self.signature_hex) and hmac.compare_digest(expected, self.signature_hex)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"schema": RECORD_SCHEMA}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DanceRecord":
        fields = {k: data[k] for k in ("expert_id", "served_by", "cluster_id", "n", "successes", "cost_per_1k", "epoch",
                                        "verifier_kind", "signer_id")}
        return cls(signature_hex=str(data.get("signature_hex", "")), **fields)


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson's score interval; (0.0, 1.0) when there are no trials."""
    if n < 0 or successes < 0 or successes > n:
        raise ValueError("need 0 <= successes <= n")
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def wilson_lower(successes: int, n: int, z: float = Z95) -> float:
    """Lower end of Wilson's score interval; 0.0 when there are no trials."""
    return wilson_interval(successes, n, z)[0]


@dataclass(frozen=True)
class RecruitmentReport:
    """Weights per cluster and expert, and why records were dropped (counts only)."""

    weights: Mapping[str, Mapping[str, float]]
    accepted: int
    rejected: Mapping[str, int]

    def probabilities(self, cluster_id: str, candidates: Iterable[str], floor: float = 0.0) -> dict[str, float]:
        """Recruitment probabilities over `candidates`: weight plus `floor` (exploration mass), normalised. A cluster or
        candidate with no record gets only the floor; with every weight zero the distribution is uniform."""
        if floor < 0.0 or not math.isfinite(floor):
            raise ValueError("floor must be finite and >= 0")
        names = list(dict.fromkeys(candidates))
        if not names:
            raise ValueError("candidates must be non-empty")
        per = self.weights.get(cluster_id, {})
        raw = {c: per.get(c, 0.0) + floor for c in names}
        total = sum(raw.values())
        if total <= 0.0:
            return {c: 1.0 / len(names) for c in names}
        return {c: v / total for c, v in raw.items()}

    def to_dict(self) -> dict[str, Any]:
        return {"weights": {c: dict(e) for c, e in self.weights.items()}, "accepted": self.accepted, "rejected": dict(self.rejected)}


def recruitment_weights(
    records: Iterable[DanceRecord],
    *,
    now_epoch: int,
    horizon_epochs: int,
    cap_per_record: float,
    secret: bytes,
    z: float = Z95,
    scout_bounds: Mapping[tuple[str, str], float] | None = None,
) -> RecruitmentReport:
    """Aggregate records into recruitment weights.

    A record is dropped (and counted under a reason) when its signature fails, when the reporter dances for an expert it
    did not serve, or when it is older than `horizon_epochs`. Each surviving record contributes
    `min(cap_per_record, wilson_lower(successes, n))` to its (cluster, expert) cell: the lower bound is the pessimistic
    score of Theorems 132-133 and the cap is Theorems 130-131. The horizon follows Theorem 129.

    `scout_bounds` maps (cluster, expert) to the upper end of a scout's own measured interval; a record whose lower bound
    exceeds it claims more than an independent probe could see and is dropped (`exceeds_scout_bound`). Cells without a
    scout bound are not checked.
    """
    if horizon_epochs < 0 or now_epoch < 0:
        raise ValueError("now_epoch and horizon_epochs must be >= 0")
    if not math.isfinite(cap_per_record) or cap_per_record <= 0.0:
        raise ValueError("cap_per_record must be finite and > 0")
    weights: dict[str, dict[str, float]] = {}
    rejected = {"bad_signature": 0, "not_served_by_reporter": 0, "expired": 0, "future_epoch": 0, "exceeds_scout_bound": 0}
    accepted = 0
    for r in records:
        if not r.verify(secret):
            rejected["bad_signature"] += 1
            continue
        if r.served_by != r.expert_id:
            rejected["not_served_by_reporter"] += 1
            continue
        if r.epoch > now_epoch:
            rejected["future_epoch"] += 1
            continue
        if now_epoch - r.epoch > horizon_epochs:
            rejected["expired"] += 1
            continue
        lower = wilson_lower(r.successes, r.n, z)
        bound = None if scout_bounds is None else scout_bounds.get((r.cluster_id, r.expert_id))
        if bound is not None and lower > bound:
            rejected["exceeds_scout_bound"] += 1
            continue
        w = min(cap_per_record, lower)
        cell = weights.setdefault(r.cluster_id, {})
        cell[r.expert_id] = cell.get(r.expert_id, 0.0) + w
        accepted += 1
    return RecruitmentReport(weights, accepted, rejected)


class DanceBoard:
    """Append-only JSONL store of records. Reading never trusts a line: malformed lines are counted and skipped."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def append(self, record: DanceRecord) -> None:
        if not record.signature_hex:
            raise ValueError("refusing to store an unsigned record")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

    def load(self) -> tuple[list[DanceRecord], int]:
        """(records, malformed_line_count)."""
        if not self.path.exists():
            return [], 0
        out: list[DanceRecord] = []
        bad = 0
        for line in self.path.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if data.get("schema") != RECORD_SCHEMA:
                    raise ValueError("schema")
                out.append(DanceRecord.from_dict(data))
            except (ValueError, KeyError, TypeError):
                bad += 1
        return out, bad
