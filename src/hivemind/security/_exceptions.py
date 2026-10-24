"""Shared security exceptions without optional GPU runtime dependencies."""

from __future__ import annotations


class SwarmSecurityViolationError(RuntimeError):
    """Raised when a GPU training job, LoRA tensor, or CAS data store operation violates swarm security rules."""

    def __init__(self, rule_id: str, category: str, message: str) -> None:
        super().__init__(f"[{rule_id}:{category}] {message}")
        self.rule_id = rule_id
        self.category = category
        self.message = message
