# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Autopoietic Neuro-Symbolic Energy Model (ANSE) for HiveMind Specialist Bees (Season 18).

Ports the measured, gate-verified mechanisms from `xaviercallens/AutoevolveAI` (`docs/EVOLUTION_LAB.md`)
to HiveMind's Specialist Bees (`hivemind/coder`, `coder-lite`, `bee-stars`, `crowd-v1`, and La Ruche MoE):
1. Graded Physical Energy Function `E_Bee(x, y)` with `MAX_PAIN_ENERGY = 1e6` fail-closed penalty
   on syntax errors, AST stubs, `sorry` axioms, or timeouts (Phase 1 Reality Engine).
2. Low-Tier Hardness Directives `D1` (`<200`-char error budget), `D3` (Best-So-Far selection +
   energy-monotonic early stop), and `D6` (Live Energy-Gap DPO/RFT preference pair export).
3. Two-Split BOINC Loss-Plateau Detector (`HM-S18-AE-02`) and Phase 3 Autopoietic Bee Promotion Gate
   (`HM-S18-AE-04`, rejecting fast-but-wrong children and A/A noise).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import math
import statistics
from typing import Any, Mapping, Sequence

MAX_PAIN_ENERGY: float = 1_000_000.0
LATENCY_SCALE_MS: float = 100.0
MEMORY_SCALE_MB: float = 64.0
COST_FRACTION_OF_ONE_TEST: float = 0.99


@dataclass(frozen=True)
class BeeEnergyEvaluation:
    """Measured physical energy receipt for a single Bee response candidate."""

    candidate_id: str
    response_text: str
    tests_passed: int
    tests_total: int
    latency_ms: float
    memory_mb: float
    invariant_residual: float
    is_stub_or_invalid: bool
    failure_reason: str | None
    total_energy: float
    receipt_sha256: str

    @property
    def passed_all_tests(self) -> bool:
        return (
            not self.is_stub_or_invalid
            and self.tests_total > 0
            and self.tests_passed == self.tests_total
            and math.isfinite(self.invariant_residual)
            and 0.0 <= self.invariant_residual <= 1e-4
        )


def _strip_lean4_comments(text: str) -> str:
    """Strip Lean 4 line (`-- ...`) and nested block (`/- ... -/`) comments before token scanning (`IP-44`)."""
    out: list[str] = []
    i = 0
    n = len(text)
    depth = 0
    while i < n:
        if i + 1 < n and text[i : i + 2] == "/-":
            depth += 1
            i += 2
            continue
        if depth > 0 and i + 1 < n and text[i : i + 2] == "-/":
            depth -= 1
            i += 2
            continue
        if depth == 0 and i + 1 < n and text[i : i + 2] == "--":
            nl = text.find("\n", i + 2)
            if nl == -1:
                break
            i = nl
            continue
        if depth == 0:
            out.append(text[i])
        i += 1
    return "".join(out)


def _has_ast_stub_or_syntax_error(code: str, *, domain: str = "code") -> tuple[bool, str | None]:
    stripped = code.strip()
    if not stripped:
        return True, "EMPTY_RESPONSE"
    if domain in ("lean4", "math_lean4"):
        import re

        uncommented = _strip_lean4_comments(stripped)
        if re.search(r"\b(sorry|admit)\b", uncommented) or "sorry" in stripped:
            return True, "LEAN4_SORRY_DETECTED"
        if re.search(r"\b(axiom|opaque|unsafe|partial)\b", uncommented):
            return True, "LEAN4_AXIOM_DETECTED"
    if domain not in ("code", "code_python", "python"):
        return False, None
    try:
        tree = ast.parse(stripped)
    except SyntaxError as exc:
        return True, f"SYNTAX_ERROR:{exc.msg}"

    funcs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for fn in funcs:
        body = [
            stmt
            for stmt in fn.body
            if not (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            )
        ]
        if not body:
            return True, f"AST_STUB_EMPTY_FUNC:{fn.name}"
        if len(body) == 1:
            stmt = body[0]
            if isinstance(stmt, ast.Pass):
                return True, f"AST_STUB_PASS:{fn.name}"
            if (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value is Ellipsis
            ):
                return True, f"AST_STUB_ELLIPSIS:{fn.name}"
            if isinstance(stmt, ast.Raise):
                return True, f"AST_STUB_RAISE_ONLY:{fn.name}"
    return False, None


def evaluate_bee_candidate_energy(
    *,
    candidate_id: str,
    response_text: str,
    tests_passed: int,
    tests_total: int,
    latency_ms: float = 0.0,
    memory_mb: float = 0.0,
    invariant_residual: float = 0.0,
    timed_out_or_crashed: bool = False,
    error_message: str | None = None,
    domain: str = "code",
    w_fail: float = 100.0,
    w_inv: float = 50.0,
    w_lat: float = 1.0,
    w_mem: float = 0.5,
) -> BeeEnergyEvaluation:
    """Compute the deterministic ANSE physical energy `E_Bee(x, y)` for a Bee candidate."""
    raw_digest = hashlib.sha256(
        f"{candidate_id}|{tests_passed}/{tests_total}|{latency_ms:.3f}|{memory_mb:.3f}|{invariant_residual:.6f}|{response_text}".encode(
            "utf-8"
        )
    ).hexdigest()

    if timed_out_or_crashed:
        return BeeEnergyEvaluation(
            candidate_id=candidate_id,
            response_text=response_text,
            tests_passed=0,
            tests_total=max(1, tests_total),
            latency_ms=max(0.0, float(latency_ms)),
            memory_mb=max(0.0, float(memory_mb)),
            invariant_residual=float(invariant_residual),
            is_stub_or_invalid=True,
            failure_reason=error_message or "SANDBOX_TIMEOUT_OR_CRASH",
            total_energy=MAX_PAIN_ENERGY,
            receipt_sha256=raw_digest,
        )

    has_ast_defect, ast_failure_reason = _has_ast_stub_or_syntax_error(response_text, domain=domain)
    if has_ast_defect or tests_total <= 0:
        return BeeEnergyEvaluation(
            candidate_id=candidate_id,
            response_text=response_text,
            tests_passed=max(0, int(tests_passed)),
            tests_total=max(0, int(tests_total)),
            latency_ms=max(0.0, float(latency_ms)) if math.isfinite(float(latency_ms)) else 0.0,
            memory_mb=max(0.0, float(memory_mb)) if math.isfinite(float(memory_mb)) else 0.0,
            invariant_residual=float(invariant_residual),
            is_stub_or_invalid=True,
            failure_reason=ast_failure_reason or error_message or "NO_HIDDEN_TESTS",
            total_energy=MAX_PAIN_ENERGY,
            receipt_sha256=raw_digest,
        )

    clamped_passed = max(0, min(int(tests_total), int(tests_passed)))
    fail_ratio = float(tests_total - clamped_passed) / float(tests_total)
    inv_term = max(0.0, float(invariant_residual))

    # Correctness dominates cost (review fix, 2026-10-26): latency and memory are squashed into [0, 1) and scaled
    # to stay strictly below the energy of ONE failed hidden test, so a correct-but-slow answer can never rank
    # below a partly-wrong fast one. Cost only orders candidates of equal correctness.
    lat = max(0.0, float(latency_ms))
    mem = max(0.0, float(memory_mb))
    lat_sq = lat / (lat + LATENCY_SCALE_MS)
    mem_sq = mem / (mem + MEMORY_SCALE_MB)
    cost_weight = max(1e-9, float(w_lat) + float(w_mem))
    cost_bound = COST_FRACTION_OF_ONE_TEST * float(w_fail) / float(tests_total)
    cost_term = cost_bound * (float(w_lat) * lat_sq + float(w_mem) * mem_sq) / cost_weight

    energy = w_fail * fail_ratio + w_inv * inv_term + cost_term
    return BeeEnergyEvaluation(
        candidate_id=candidate_id,
        response_text=response_text,
        tests_passed=clamped_passed,
        tests_total=int(tests_total),
        latency_ms=max(0.0, float(latency_ms)),
        memory_mb=max(0.0, float(memory_mb)),
        invariant_residual=inv_term,
        is_stub_or_invalid=False,
        failure_reason=error_message if clamped_passed < tests_total else None,
        total_energy=round(float(energy), 6),
        receipt_sha256=raw_digest,
    )


def compress_error_trace_d1(
    failing_code: str,
    stderr_or_reason: str,
    *,
    max_chars: int = 200,
) -> str:
    """Directive D1: Compress failing code + stderr to `< max_chars` to prevent small-Bee attention dilution."""
    budget = max(32, int(max_chars))
    err_clean = " ".join((stderr_or_reason or "test_failed").strip().split())
    code_clean = " ".join((failing_code or "").strip().split())
    combined = f"ERR: {err_clean} | CODE: {code_clean}"
    if len(combined) <= budget:
        return combined
    return combined[: budget - 3] + "..."


def select_best_so_far_d3(
    attempts: Sequence[BeeEnergyEvaluation],
    *,
    early_stop_energy_ceiling: float = 60.0,
) -> tuple[BeeEnergyEvaluation, bool]:
    """Directive D3: Return `(best_attempt_so_far, should_stop_early)`.

    Never returns a degraded final retry when an earlier attempt had lower energy,
    and triggers early stop when `E_1 > early_stop_energy_ceiling` or `E_t > E_{t-1}`.
    """
    if not attempts:
        raise ValueError("attempts must be non-empty.")
    best = min(attempts, key=lambda a: a.total_energy)
    if best.passed_all_tests:
        return best, True
    if len(attempts) == 1:
        return best, bool(attempts[0].total_energy > early_stop_energy_ceiling)
    diverged = attempts[-1].total_energy >= attempts[-2].total_energy
    exceeded_ceiling = attempts[0].total_energy > early_stop_energy_ceiling
    return best, bool(diverged or exceeded_ceiling)


def harvest_energy_gap_dpo_pair_d6(
    prompt: str,
    attempts: Sequence[BeeEnergyEvaluation],
    *,
    min_delta_energy: float = 10.0,
) -> dict[str, Any] | None:
    """Directive D6: Export a verified `(chosen_low_E, rejected_high_E)` pair when `Delta E >= min_delta_energy`."""
    if len(attempts) < 2:
        return None
    best = min(attempts, key=lambda a: a.total_energy)
    worst = max(attempts, key=lambda a: a.total_energy)
    if not best.passed_all_tests:
        return None
    delta_e = float(worst.total_energy - best.total_energy)
    if delta_e < float(min_delta_energy):
        return None
    pair_sha = hashlib.sha256(
        f"{prompt}|{best.receipt_sha256}|{worst.receipt_sha256}".encode("utf-8")
    ).hexdigest()
    return {
        "prompt": prompt,
        "chosen_response": best.response_text,
        "rejected_response": worst.response_text,
        "chosen_energy": best.total_energy,
        "rejected_energy": worst.total_energy,
        "delta_energy": round(delta_e, 6),
        "pair_sha256": pair_sha,
    }


def detect_boinc_loss_plateau(
    round_losses: Sequence[float],
    *,
    block_size: int = 10,
    min_block_gain: float = 0.005,
    max_tail_std: float = 0.006,
) -> dict[str, Any]:
    """Detect the 30-round BOINC QLoRA loss plateau (`HM-S18-AE-02`) to trigger curriculum rotation."""
    n = len(round_losses)
    if n < 2 * block_size:
        return {
            "plateau_detected": False,
            "rounds_evaluated": n,
            "reason": "INSUFFICIENT_ROUNDS",
        }
    prev_block = round_losses[-2 * block_size : -block_size]
    curr_block = round_losses[-block_size:]
    block_gain = float(statistics.mean(prev_block) - statistics.mean(curr_block))
    tail_window = round_losses[-min(n, 2 * block_size) :]
    tail_std = float(statistics.pstdev(tail_window))
    plateau = bool(block_gain < min_block_gain and tail_std <= max_tail_std)
    return {
        "plateau_detected": plateau,
        "rounds_evaluated": n,
        "block_gain": round(block_gain, 6),
        "tail_std": round(tail_std, 6),
        "recommended_action": (
            "ROTATE_CURRICULUM_OR_COSINE_LR_RESTART" if plateau else "CONTINUE_CURRENT_CHAIN"
        ),
    }


def exact_mcnemar_p(only_child_right: int, only_parent_right: int) -> float:
    """Two-sided exact McNemar p-value on the discordant pairs (binomial, p = 0.5)."""
    b, c = int(only_child_right), int(only_parent_right)
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative.")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / float(2**n)
    return min(1.0, 2.0 * tail)


def calibrate_noise_margin(aa_median_deltas: Sequence[float]) -> float:
    """Noise margin from measured A/A runs (parent against itself): the largest |median Delta E| observed.

    The margin must come from A/A measurements on the same host and suite, never from a constant.
    """
    if len(aa_median_deltas) < 3:
        raise ValueError("at least 3 A/A runs are needed to calibrate the noise margin.")
    return float(max(abs(float(d)) for d in aa_median_deltas))


@dataclass(frozen=True)
class AutopoieticPromotionVerdict:
    """Phase 3 Autopoietic Bee Promotion Gate verdict."""

    promoted: bool
    median_delta_energy: float
    parent_regressions_failed: int
    shuffled_control_gained: bool
    reason: str
    mcnemar_p: float | None = None


def evaluate_child_bee_promotion(
    *,
    parent_energies: Sequence[float],
    child_energies: Sequence[float],
    noise_margin: float,
    parent_regressions_failed: int = 0,
    shuffled_control_gained: bool = False,
    paired_discordant: tuple[int, int] | None = None,
    alpha: float = 0.05,
) -> AutopoieticPromotionVerdict:
    """Promote a child Bee adapter only if every condition holds.

    1. ``parent_regressions_failed == 0`` (no item the parent verified is lost: rejects fast-but-wrong children);
    2. the shuffled-label control did not gain (else the gain is not attributable to the data);
    3. when ``paired_discordant = (only_child_right, only_parent_right)`` is given, the exact McNemar test is
       significant at ``alpha`` in the child's favour (primary rule, same as HM-S14-01);
    4. ``median(E_child - E_parent) < -noise_margin``, with ``noise_margin`` from :func:`calibrate_noise_margin`.
    """
    if len(parent_energies) == 0 or len(parent_energies) != len(child_energies):
        raise ValueError("parent_energies and child_energies must be non-empty and equal length.")
    if not math.isfinite(float(noise_margin)) or float(noise_margin) <= 0.0:
        raise ValueError("noise_margin must be a strictly positive finite float from calibrate_noise_margin.")
    if not all(math.isfinite(float(p)) and math.isfinite(float(c)) for p, c in zip(parent_energies, child_energies)):
        raise ValueError("parent_energies and child_energies must contain only finite floats.")
    deltas = [float(c - p) for p, c in zip(parent_energies, child_energies)]
    med_delta = float(statistics.median(deltas))
    p_value: float | None = None
    if paired_discordant is not None:
        p_value = exact_mcnemar_p(*paired_discordant)
        child_better = paired_discordant[0] > paired_discordant[1]
        if parent_regressions_failed == 0 and not shuffled_control_gained and not (child_better and p_value < alpha):
            return AutopoieticPromotionVerdict(
                promoted=False,
                median_delta_energy=med_delta,
                parent_regressions_failed=0,
                shuffled_control_gained=False,
                reason=f"REJECTED_NOT_SIGNIFICANT (McNemar p={p_value:.4g}, discordant={paired_discordant})",
                mcnemar_p=p_value,
            )

    if parent_regressions_failed > 0:
        return AutopoieticPromotionVerdict(
            promoted=False,
            median_delta_energy=med_delta,
            parent_regressions_failed=int(parent_regressions_failed),
            shuffled_control_gained=bool(shuffled_control_gained),
            reason=f"REJECTED_FAST_BUT_WRONG ({parent_regressions_failed} parent regressions failed)",
            mcnemar_p=p_value,
        )
    if shuffled_control_gained:
        return AutopoieticPromotionVerdict(
            promoted=False,
            median_delta_energy=med_delta,
            parent_regressions_failed=0,
            shuffled_control_gained=True,
            reason="VOID_SHUFFLED_CONTROL_GAINED",
            mcnemar_p=p_value,
        )
    if med_delta >= -abs(float(noise_margin)):
        return AutopoieticPromotionVerdict(
            promoted=False,
            median_delta_energy=med_delta,
            parent_regressions_failed=0,
            shuffled_control_gained=False,
            reason=f"REJECTED_WITHIN_NOISE_MARGIN (median_delta={med_delta:+.4f} >= {-abs(noise_margin):+.4f})",
            mcnemar_p=p_value,
        )
    if paired_discordant is None:
        return AutopoieticPromotionVerdict(
            promoted=False,
            median_delta_energy=med_delta,
            parent_regressions_failed=0,
            shuffled_control_gained=False,
            reason="REJECTED_MISSING_PAIRED_DISCORDANT_MCNEMAR",
            mcnemar_p=None,
        )
    return AutopoieticPromotionVerdict(
        promoted=True,
        median_delta_energy=med_delta,
        parent_regressions_failed=0,
        shuffled_control_gained=False,
        reason=f"PROMOTED_AUTOPOIETIC_CHILD (median_delta={med_delta:+.4f} < {-abs(noise_margin):+.4f})",
        mcnemar_p=p_value,
    )
