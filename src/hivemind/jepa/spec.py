# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Versioned, validated configuration of the JEPA-style auxiliary objectives (part of a signed workunit).

A validator that replays a training segment must compute exactly the same loss as the producer. So every knob that
changes the loss lives here; the spec is serialized canonically and hashed (`spec_sha256`) so it can be bound into a
workunit signature. A client that does not know `version` must refuse the workunit rather than train something else.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

SPEC_VERSION = 1
MODES = ("none", "stp", "pairs")
STP_VARIANTS = ("geodesic", "random_span")


@dataclass(frozen=True)
class JepaSpec:
    """`mode`:
    - "none": plain next-token loss (default; the spec is then a no-op);
    - "stp": Semantic Tube Prediction (arXiv 2602.22617), single forward pass, no paired views;
    - "pairs": LLM-JEPA (arXiv 2509.14252) on paired views, e.g. question <-> answer, docstring <-> code.

    `stp_variant`:
    - "geodesic": the paper's loss, 1 - cos(h_t - h_r, h_r - h_s) for random s < r < t (used for its results);
    - "random_span": the variant in the released code (`stp.py --linear=random_span`), patch displacement vs the
      displacement of the rest. Not described in the paper; not assumed equivalent.
    """

    version: int = SPEC_VERSION
    mode: str = "none"
    lam: float = 0.02                 # paper Table 2: 0.02 on NL-RX-SYNTH; per-task range 0.0025-0.16
    stp_variant: str = "geodesic"
    stp_samples: int = 1              # index triples per sequence
    min_segment: int = 8              # shortest document segment eligible for STP (tokens)
    eos_id: int = -1                  # document separator in packed streams; -1 = the whole row is one document
    stopgrad: bool = True             # pairs: no gradient through the target view (measured to stop collapse)
    center: bool = False              # pairs: subtract the batch mean before the cosine
    dropout: float = 0.0              # skip the auxiliary term on this fraction of steps (hash of seed, step)
    seed: int = 0
    collapse_floor: float = 0.05      # health: an auxiliary term below this is flagged as a possible shortcut

    def __post_init__(self) -> None:
        if self.version != SPEC_VERSION:
            raise ValueError(f"unsupported JepaSpec version {self.version} (this client knows {SPEC_VERSION})")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.stp_variant not in STP_VARIANTS:
            raise ValueError(f"stp_variant must be one of {STP_VARIANTS}")
        for name in ("lam", "dropout", "collapse_floor"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v < 0:
                raise ValueError(f"{name} must be a finite number >= 0")
        if self.dropout > 1:
            raise ValueError("dropout must be in [0, 1]")
        for name in ("stp_samples", "min_segment"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        if self.min_segment < 3:
            raise ValueError("min_segment must be >= 3 (STP needs s < r < t)")
        if isinstance(self.eos_id, bool) or not isinstance(self.eos_id, int) or self.eos_id < -1:
            raise ValueError("eos_id must be an integer >= -1")

    @property
    def active(self) -> bool:
        return self.mode != "none" and self.lam > 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: dict) -> "JepaSpec":
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown JepaSpec fields: {sorted(unknown)}")
        return cls(**d)

    def spec_sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()


def step_uniform(seed: int, step: int, row: int, k: int, salt: str = "hivemind-jepa") -> float:
    """A deterministic number in [0, 1) from (seed, step, row, k): never a device RNG, so a validator replays it."""
    h = hashlib.sha256(f"{salt}:{int(seed)}:{int(step)}:{int(row)}:{int(k)}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def apply_aux(spec: JepaSpec, step: int) -> bool:
    """Whether the auxiliary term is applied at this step (loss dropout, deterministic)."""
    if not spec.active:
        return False
    return spec.dropout == 0 or step_uniform(spec.seed, step, -1, 0, "dropout") >= spec.dropout
