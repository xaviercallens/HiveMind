# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Blinded inference canary probe with Wilson score lower-bound quarantine (PS-05-D1).

Evaluates serving producers with blinded canary prompts that are wire-indistinguishable
from normal ``/v1/chat/completions`` traffic, scores each producer using the exact
Wilson score 95% lower confidence bound, and measures junk-node detection rate and
honest-node false-positive rate against pre-registered thresholds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
import random
import re
from typing import Any, Callable

Z_95 = 1.959963984540054
DEFAULT_MIN_TRIALS = 10
DEFAULT_WILSON_LB_THRESHOLD = 0.60
PREREG_MIN_DETECTION_RATE = 0.95
PREREG_MAX_FALSE_POSITIVE_RATE = 0.05


def wilson_confidence_interval(
    successes: int,
    total: int,
    z: float = Z_95,
) -> tuple[float, float]:
    """Compute the exact Wilson score confidence interval ``(lower, upper)`` in ``[0, 1]``."""
    if total <= 0:
        return 0.0, 0.0
    if successes < 0 or successes > total:
        raise ValueError(f"successes ({successes}) must be in [0, total ({total})]")
    n = float(total)
    p_hat = float(successes) / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    radicand = max(0.0, (p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n)
    half_width = (z * math.sqrt(radicand)) / denom
    low = max(0.0, min(1.0, center - half_width))
    high = max(0.0, min(1.0, center + half_width))
    return low, high


def wilson_lower_bound(successes: int, total: int, z: float = Z_95) -> float:
    """Return the lower bound of the Wilson score interval for ``successes / total``."""
    low, _ = wilson_confidence_interval(successes, total, z=z)
    return low


@dataclass(frozen=True)
class CanaryPrompt:
    """A blinded canary prompt with a deterministic answer verifier."""

    canary_id: str
    prompt: str
    expected_pattern: str
    category: str

    def to_chat_request(self, model: str = "hivemind/coder") -> dict[str, Any]:
        """Format as an ordinary OpenAI-compatible chat request (blinded: no canary header)."""
        return {
            "model": model,
            "messages": [{"role": "user", "content": self.prompt}],
            "temperature": 0.0,
            "max_tokens": 32,
        }

    def verify_answer(self, text: str) -> bool:
        if not isinstance(text, str) or not text.strip():
            return False
        return re.search(self.expected_pattern, text.strip(), flags=re.IGNORECASE) is not None


BLINDED_CANARY_BANK: tuple[CanaryPrompt, ...] = (
    CanaryPrompt(
        canary_id="canary-arith-17x19",
        prompt="Compute 17 * 19 and reply with the integer only.",
        expected_pattern=r"\b323\b",
        category="arithmetic",
    ),
    CanaryPrompt(
        canary_id="canary-python-len",
        prompt="What does len('hivemind') evaluate to in Python? Reply with the integer only.",
        expected_pattern=r"\b8\b",
        category="code",
    ),
    CanaryPrompt(
        canary_id="canary-sha-hex-len",
        prompt="How many hexadecimal characters are in a standard SHA-256 digest? Reply with the integer.",
        expected_pattern=r"\b64\b",
        category="crypto",
    ),
    CanaryPrompt(
        canary_id="canary-derivative-x3",
        prompt="What is the derivative of x^3 with respect to x at x = 2? Reply with the integer only.",
        expected_pattern=r"\b12\b",
        category="math",
    ),
    CanaryPrompt(
        canary_id="canary-http-unauthorized",
        prompt="What is the standard HTTP status code for Unauthorized? Reply with the 3-digit integer.",
        expected_pattern=r"\b401\b",
        category="systems",
    ),
    CanaryPrompt(
        canary_id="canary-binary-42",
        prompt="Write the decimal integer 42 in binary without prefix.",
        expected_pattern=r"\b101010\b",
        category="cs",
    ),
    CanaryPrompt(
        canary_id="canary-prime-after-29",
        prompt="What is the smallest prime number strictly greater than 29?",
        expected_pattern=r"\b31\b",
        category="math",
    ),
    CanaryPrompt(
        canary_id="canary-json-bool",
        prompt="In strict RFC 8259 JSON, is the boolean true written lowercase 'true' or capitalized 'True'?",
        expected_pattern=r"\btrue\b",
        category="code",
    ),
)


@dataclass
class NodeCanaryState:
    """Per-node canary evaluation state and quarantine decision."""

    node_id: str
    planted_role: str  # "honest" or junk mode name
    successes: int = 0
    total: int = 0
    min_trials: int = DEFAULT_MIN_TRIALS
    wilson_lb_threshold: float = DEFAULT_WILSON_LB_THRESHOLD

    @property
    def empirical_pass_rate(self) -> float:
        return float(self.successes) / float(self.total) if self.total > 0 else 0.0

    @property
    def wilson_lb(self) -> float:
        return wilson_lower_bound(self.successes, self.total)

    @property
    def quarantined(self) -> bool:
        return self.total >= self.min_trials and self.wilson_lb < self.wilson_lb_threshold

    def record(self, passed: bool) -> None:
        self.total += 1
        if passed:
            self.successes += 1

    def to_dict(self) -> dict[str, Any]:
        low, high = wilson_confidence_interval(self.successes, self.total)
        return {
            "node_id": self.node_id,
            "planted_role": self.planted_role,
            "successes": self.successes,
            "total": self.total,
            "empirical_pass_rate": round(self.empirical_pass_rate, 6),
            "wilson_lb": round(low, 6),
            "wilson_ub": round(high, 6),
            "quarantined": self.quarantined,
        }


def run_planted_canary_evaluation(
    *,
    num_honest_nodes: int = 40,
    num_planted_junk_nodes: int = 20,
    trials_per_node: int = 20,
    honest_pass_prob: float = 0.96,
    seed: int = 20261107,
    min_trials: int = DEFAULT_MIN_TRIALS,
    wilson_lb_threshold: float = DEFAULT_WILSON_LB_THRESHOLD,
) -> dict[str, Any]:
    """Run a blinded canary campaign over honest nodes and 4 classes of planted junk nodes."""
    rng = random.Random(seed)
    canonical_answers = {
        "canary-arith-17x19": "323",
        "canary-python-len": "8",
        "canary-sha-hex-len": "64",
        "canary-derivative-x3": "12",
        "canary-http-unauthorized": "401",
        "canary-binary-42": "101010",
        "canary-prime-after-29": "31",
        "canary-json-bool": "true",
    }

    junk_modes: tuple[tuple[str, Callable[[CanaryPrompt], str]], ...] = (
        ("junk_constant_ok", lambda _c: "OK"),
        ("junk_empty_or_whitespace", lambda _c: "   "),
        ("junk_random_number", lambda _c: str(rng.randint(1000, 9999))),
        (
            "junk_lazy_half_guesser",
            lambda c: canonical_answers[c.canary_id] if rng.random() < 0.25 else "I do not know",
        ),
    )

    states: list[NodeCanaryState] = []
    for i in range(num_honest_nodes):
        state = NodeCanaryState(
            node_id=f"node-honest-{i:03d}",
            planted_role="honest",
            min_trials=min_trials,
            wilson_lb_threshold=wilson_lb_threshold,
        )
        for t in range(trials_per_node):
            canary = BLINDED_CANARY_BANK[t % len(BLINDED_CANARY_BANK)]
            ans = canonical_answers[canary.canary_id] if rng.random() < honest_pass_prob else "transient_timeout"
            state.record(canary.verify_answer(ans))
        states.append(state)

    for j in range(num_planted_junk_nodes):
        mode_name, responder = junk_modes[j % len(junk_modes)]
        state = NodeCanaryState(
            node_id=f"node-planted-junk-{j:03d}",
            planted_role=mode_name,
            min_trials=min_trials,
            wilson_lb_threshold=wilson_lb_threshold,
        )
        for t in range(trials_per_node):
            canary = BLINDED_CANARY_BANK[t % len(BLINDED_CANARY_BANK)]
            ans = responder(canary)
            state.record(canary.verify_answer(ans))
        states.append(state)

    honest_states = [s for s in states if s.planted_role == "honest"]
    junk_states = [s for s in states if s.planted_role != "honest"]

    detected_junk = sum(1 for s in junk_states if s.quarantined)
    false_positives = sum(1 for s in honest_states if s.quarantined)

    det_rate = float(detected_junk) / float(len(junk_states)) if junk_states else 0.0
    fp_rate = float(false_positives) / float(len(honest_states)) if honest_states else 0.0
    det_ci = wilson_confidence_interval(detected_junk, len(junk_states))
    fp_ci = wilson_confidence_interval(false_positives, len(honest_states))

    passes_thresholds = (
        det_rate >= PREREG_MIN_DETECTION_RATE and fp_rate <= PREREG_MAX_FALSE_POSITIVE_RATE
    )

    return {
        "schema": "hivemind.canary_probe_receipt.v1",
        "deliverable_id": "PS-05-D1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": seed,
        "parameters": {
            "num_honest_nodes": num_honest_nodes,
            "num_planted_junk_nodes": num_planted_junk_nodes,
            "trials_per_node": trials_per_node,
            "min_trials": min_trials,
            "wilson_lb_threshold": wilson_lb_threshold,
            "preregistered_min_detection_rate": PREREG_MIN_DETECTION_RATE,
            "preregistered_max_false_positive_rate": PREREG_MAX_FALSE_POSITIVE_RATE,
        },
        "summary": {
            "planted_junk_detected": detected_junk,
            "planted_junk_total": len(junk_states),
            "detection_rate": round(det_rate, 6),
            "detection_rate_ci95": [round(det_ci[0], 6), round(det_ci[1], 6)],
            "honest_false_positives": false_positives,
            "honest_total": len(honest_states),
            "false_positive_rate": round(fp_rate, 6),
            "false_positive_rate_ci95": [round(fp_ci[0], 6), round(fp_ci[1], 6)],
            "passes_preregistered_thresholds": passes_thresholds,
        },
        "nodes": [s.to_dict() for s in states],
    }
