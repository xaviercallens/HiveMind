"""Security exceptions shared by the owner-order guardrail and the GPU runtime hardening.

Kept free of torch so that the PR security scanner (and the lightweight CI that runs it) can import the
owner-order checks without the GPU runtime. Found by PR #118 (clark719664), 2026-10-24.
"""

from __future__ import annotations


class SwarmSecurityViolationError(RuntimeError):
    """Raised when a GPU training job, LoRA tensor, or CAS data store operation violates swarm security rules."""

    def __init__(self, rule_id: str, category: str, message: str) -> None:
        super().__init__(f"[{rule_id}:{category}] {message}")
        self.rule_id = rule_id
        self.category = category
        self.message = message
