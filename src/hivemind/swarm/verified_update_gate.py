# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Level B of the GWAYA Code expert: a sandbox pass rate on held-out Python tasks gates acceptance of a LoRA / DiLoCo update.

A candidate update (for example the E15 LoRA on Qwen2.5-Coder-1.5B) is accepted only if, on the SAME held-out tasks, it does
not do worse than the current baseline when both are judged by the fail-closed GWAYA-style sandbox (``scripts/e15_gate.py``).
This module is torch-free and has no I/O of its own: :func:`decide` is a pure function of two outcome maps, and
:func:`evaluate_tasks` turns a generator plus an injected sandbox gate into such a map.

Outcomes per task are ``"pass"``, ``"fail"`` or ``"unverified"`` (the sandbox could not be established, errored, or the
generator crashed). An unverified result is never counted as a pass. To stay conservative in both directions an unverified
CANDIDATE result is scored as a failure and an unverified BASELINE result is scored as a pass, so uncertainty can only work
against the update. Too many unverified tasks block the decision altogether.

The test is the exact paired (McNemar / sign) test on the discordant tasks: ``b`` = baseline passes and candidate fails,
``c`` = baseline fails and candidate passes. Under "no difference" each discordant task is a fair coin. Rules, in order:

1. the two maps must cover the same task keys, with at least ``min_n`` tasks, else ``INSUFFICIENT``;
2. more than ``max_unverified_fraction`` of the tasks unverified (in either map) is ``INSUFFICIENT``;
3. a net loss beyond ``tolerated_net_loss`` tasks (``b - c``) is ``REJECT``, and so is a one-sided significant regression;
4. if ``require_improvement`` is set the candidate must beat the baseline with one-sided exact p < ``alpha``, else ``REJECT``;
5. otherwise ``ACCEPT``. With the defaults an identical candidate is accepted (non-regression policy).

Only ``ACCEPT`` admits an update. This gate says nothing about quality beyond the held-out tasks and the visible tests: a pass
means the sandboxed tests passed, not that the code is correct.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

PASS, FAIL, UNVERIFIED = "pass", "fail", "unverified"
OUTCOMES = (PASS, FAIL, UNVERIFIED)
ACCEPT, REJECT, INSUFFICIENT = "ACCEPT", "REJECT", "INSUFFICIENT"


@dataclass(frozen=True)
class UpdatePolicy:
    alpha: float = 0.05
    min_n: int = 30
    max_unverified_fraction: float = 0.05
    tolerated_net_loss: int = 0  # allowed value of (b - c) in tasks; 0 = the candidate may not lose on net
    require_improvement: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, (int, float))
            or not 0.0 < float(self.alpha) < 1.0
        ):
            raise ValueError("alpha must be in (0, 1)")
        if isinstance(self.min_n, bool) or not isinstance(self.min_n, int) or self.min_n < 1:
            raise ValueError("min_n must be >= 1")
        if (
            isinstance(self.max_unverified_fraction, bool)
            or not isinstance(self.max_unverified_fraction, (int, float))
            or not 0.0 <= float(self.max_unverified_fraction) <= 1.0
        ):
            raise ValueError("max_unverified_fraction must be in [0, 1]")
        if (
            isinstance(self.tolerated_net_loss, bool)
            or not isinstance(self.tolerated_net_loss, int)
            or self.tolerated_net_loss < 0
        ):
            raise ValueError("tolerated_net_loss must be >= 0")
        if not isinstance(self.require_improvement, bool):
            raise ValueError("require_improvement must be a bool")


@dataclass(frozen=True)
class UpdateReceipt:
    decision: str
    reasons: List[str]
    n: int
    b: int  # baseline pass, candidate fail (regressions)
    c: int  # baseline fail, candidate pass (improvements)
    both_pass: int
    both_fail: int
    baseline_pass: int
    candidate_pass: int
    unverified_baseline: int
    unverified_candidate: int
    unverified_fraction: float
    p_two_sided: float
    p_regression: float  # one-sided exact: P(X >= b) for X ~ Binomial(b + c, 1/2)
    p_improvement: float  # one-sided exact: P(X >= c)
    policy: Dict[str, Any] = field(default_factory=dict)
    receipt_sha256: str = ""

    @property
    def accepted(self) -> bool:
        return self.decision == ACCEPT

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _upper_tail(k: int, n: int) -> float:
    """P(X >= k) for X ~ Binomial(n, 1/2), exact."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def _lower_tail(k: int, n: int) -> float:
    """P(X <= k) for X ~ Binomial(n, 1/2), exact."""
    return sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n


def exact_paired_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p from the discordant counts (same value as ``scripts/e15_lib.mcnemar_exact``)."""
    if (
        isinstance(b, bool)
        or not isinstance(b, int)
        or b < 0
        or isinstance(c, bool)
        or not isinstance(c, int)
        or c < 0
    ):
        raise ValueError("b and c must be non-negative integers")
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * _lower_tail(min(b, c), n))


def _seal(receipt: UpdateReceipt) -> UpdateReceipt:
    body = receipt.to_dict()
    body["receipt_sha256"] = ""
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return UpdateReceipt(**{**body, "receipt_sha256": digest})


def decide(baseline: Mapping[str, str], candidate: Mapping[str, str], policy: UpdatePolicy = UpdatePolicy()) -> UpdateReceipt:
    """Accept or refuse a candidate update from per-task sandbox outcomes on the same held-out tasks (see module docstring)."""
    for name, outcomes in (("baseline", baseline), ("candidate", candidate)):
        bad = sorted({v for v in outcomes.values() if v not in OUTCOMES})
        if bad:
            raise ValueError(f"{name} has outcomes outside {OUTCOMES}: {bad}")
    reasons: List[str] = []
    keys_b, keys_c = set(baseline), set(candidate)
    shared = sorted(keys_b & keys_c)
    if keys_b != keys_c:
        reasons.append(f"task sets differ ({len(keys_b - keys_c)} only in baseline, {len(keys_c - keys_b)} only in candidate)")
    n = len(shared)
    base_scored = [baseline[k] in (PASS, UNVERIFIED) for k in shared]  # unverified baseline scores as a pass
    cand_scored = [candidate[k] == PASS for k in shared]  # unverified candidate scores as a fail
    b = sum(1 for x, y in zip(base_scored, cand_scored) if x and not y)
    c = sum(1 for x, y in zip(base_scored, cand_scored) if y and not x)
    both_pass = sum(1 for x, y in zip(base_scored, cand_scored) if x and y)
    ub = sum(1 for k in shared if baseline[k] == UNVERIFIED)
    uc = sum(1 for k in shared if candidate[k] == UNVERIFIED)
    unverified_tasks = sum(1 for k in shared if baseline[k] == UNVERIFIED or candidate[k] == UNVERIFIED)
    frac = unverified_tasks / n if n else 1.0
    decision = ACCEPT
    if keys_b != keys_c:
        decision = INSUFFICIENT
    if n < policy.min_n:
        reasons.append(f"only {n} shared tasks, need at least {policy.min_n}")
        decision = INSUFFICIENT
    if frac > policy.max_unverified_fraction:
        reasons.append(f"{unverified_tasks}/{n} tasks unverified ({frac:.3f} > {policy.max_unverified_fraction})")
        decision = INSUFFICIENT
    d = b + c
    p_regress, p_improve = _upper_tail(b, d), _upper_tail(c, d)
    p_two = exact_paired_p(b, c)
    if decision == ACCEPT:
        if b - c > policy.tolerated_net_loss:
            reasons.append(f"net loss {b - c} tasks (b={b}, c={c}) exceeds tolerated {policy.tolerated_net_loss}")
            decision = REJECT
        elif b > c and p_regress < policy.alpha:
            reasons.append(f"significant regression (one-sided exact p={p_regress:.4g} < {policy.alpha})")
            decision = REJECT
        elif policy.require_improvement and not (c > b and p_improve < policy.alpha):
            reasons.append(f"improvement required but not shown (one-sided exact p={p_improve:.4g}, alpha={policy.alpha})")
            decision = REJECT
    if decision == ACCEPT:
        reasons.append("no regression on the held-out tasks" if b >= c else "candidate did not lose on net")
    receipt = UpdateReceipt(decision, reasons, n, b, c, both_pass, n - both_pass - b - c, sum(base_scored), sum(cand_scored), ub, uc, frac,
                            p_two, p_regress, p_improve, asdict(policy))
    return _seal(receipt)


@dataclass(frozen=True)
class HeldOutTask:
    key: str
    prompt: str
    tests: Sequence[str]
    setup: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("key must be a non-empty string")
        if not isinstance(self.prompt, str):
            raise ValueError("prompt must be a string")
        if isinstance(self.tests, (str, bytes)) or not isinstance(self.tests, Sequence) or not self.tests:
            raise ValueError("tests must be a non-empty sequence of strings")
        if not isinstance(self.setup, str):
            raise ValueError("setup must be a string")


_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S | re.I)


def extract_code(text: str) -> str:
    """The last fenced code block of a model reply, or the whole reply when it has none."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    blocks = _FENCE.findall(text)
    return (blocks[-1] if blocks else text).strip()


def evaluate_tasks(tasks: Sequence[HeldOutTask], generate: Callable[[HeldOutTask], str], gate: Callable[[str, Sequence[str], str], Any],
                   extract: Callable[[str], str] = extract_code) -> Dict[str, str]:
    """Run ``generate`` on each task and judge the code with ``gate(code, tests, setup)`` (e.g. ``e15_gate.gate``).

    ``gate`` must return an object with a ``verdict`` string: ``ACCEPTED`` is a pass, ``UNVERIFIED`` is unverified, every other
    verdict is a fail. A generator that raises, an empty extraction, or a gate that raises is ``unverified`` and never a pass.
    """
    keys = [t.key for t in tasks]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate task keys")
    out: Dict[str, str] = {}
    for task in tasks:
        try:
            code = extract(generate(task))
        except Exception:  # an external generator failure says nothing about the model's code
            out[task.key] = UNVERIFIED
            continue
        if not code.strip():
            out[task.key] = FAIL  # the model answered, with nothing runnable
            continue
        try:
            verdict = str(gate(code, task.tests, task.setup).verdict)
        except Exception:
            out[task.key] = UNVERIFIED
            continue
        out[task.key] = PASS if verdict == "ACCEPTED" else UNVERIFIED if verdict == "UNVERIFIED" else FAIL
    return out


def gate_update(tasks: Sequence[HeldOutTask], baseline_generate: Callable[[HeldOutTask], str], candidate_generate: Callable[[HeldOutTask], str],
                gate: Callable[[str, Sequence[str], str], Any], policy: UpdatePolicy = UpdatePolicy()) -> UpdateReceipt:
    """Evaluate baseline and candidate on the same held-out tasks and decide."""
    return decide(evaluate_tasks(tasks, baseline_generate, gate), evaluate_tasks(tasks, candidate_generate, gate), policy)


__all__ = ["ACCEPT", "FAIL", "HeldOutTask", "INSUFFICIENT", "PASS", "REJECT", "UNVERIFIED", "UpdatePolicy", "UpdateReceipt", "decide",
           "evaluate_tasks", "exact_paired_p", "extract_code", "gate_update"]
