# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HiveMind JEPA-LLM: JEPA-style auxiliary objectives for language-model training (opt-in, not wired by default).

- `spec`: versioned, hashable configuration (`JepaSpec`) for signed, replayable workunits.
- `stp`: Semantic Tube Prediction (arXiv 2602.22617), single pass, no paired views; document-aware and deterministic.
- `pairs`: LLM-JEPA on paired views (arXiv 2509.14252), with the anti-collapse defaults measured in HiveMind PR #907.
- `hf`: one-pass next-token + STP loss for Hugging Face / PEFT causal LMs.

Design and pre-registered experiment plan: docs/HIVEMIND_JEPA_LLM_PROPOSAL.md.
"""

from hivemind.jepa.spec import SPEC_VERSION, JepaSpec, apply_aux, step_uniform

__all__ = ["SPEC_VERSION", "JepaSpec", "apply_aux", "step_uniform"]
