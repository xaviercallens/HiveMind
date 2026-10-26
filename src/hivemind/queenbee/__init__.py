"""QueenBee: split work into verifiable sub-tasks, route them to federated experts, learn from verified outcomes.

Public name: QueenBee. Algorithm name used in the Lean specification (`HiveMindSpec.ALIX_QueenBee`) and in record
headers: ALIX. Modules:

- `dance`: signed, aggregate "dance records" (task cluster, cost, verified successes) and the recruitment weights
  routers derive from them (fading, per-identity cap, verified-only). Theorems 129-134 of the Lean spec.
- `strategies`: the three local-learning strategies behind one `StrategyManager`: a Thompson-sampling bandit over
  experts (the default), verified preference pairs for a local adapter (with the 30% dilution cap, Theorems 135-136),
  and optional sharing of clipped, noised adapter deltas through Multi-Krum plus the DiLoCo quorum.
- `knowledge_market`: end users share verified knowledge as public data and receive a lump-sum HMT bonus that
  leaves the Kelly equilibrium unchanged (Theorems 137-139).

Nothing here calls a model: every strategy is driven by outcomes a verifier produced, and every function is pure or
writes only to a path the caller names.
"""

from __future__ import annotations

ALGORITHM_NAME = "ALIX"
PUBLIC_NAME = "QueenBee"
RECORD_SCHEMA = "hivemind-queenbee-dance/v1"

__all__ = ["ALGORITHM_NAME", "PUBLIC_NAME", "RECORD_SCHEMA"]
