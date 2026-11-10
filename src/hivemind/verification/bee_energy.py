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

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string")
        if not isinstance(self.response_text, str):
            raise ValueError("response_text must be a string")
        if isinstance(self.tests_passed, bool) or not isinstance(self.tests_passed, int) or self.tests_passed < 0:
            raise ValueError("tests_passed must be a non-negative integer")
        if isinstance(self.tests_total, bool) or not isinstance(self.tests_total, int) or self.tests_total < 0:
            raise ValueError("tests_total must be a non-negative integer")
        if not isinstance(self.is_stub_or_invalid, bool):
            raise ValueError("is_stub_or_invalid must be a boolean")
        if (
            isinstance(self.total_energy, bool)
            or not isinstance(self.total_energy, (int, float))
            or float(self.total_energy) < 0.0
            or math.isinf(float(self.total_energy))
        ):
            raise ValueError("total_energy must be a finite number >= 0.0")
        if not isinstance(self.receipt_sha256, str) or len(self.receipt_sha256.strip()) != 64:
            raise ValueError("receipt_sha256 must be a 64-character SHA-256 hex digest")

    @property
    def passed_all_tests(self) -> bool:
        return (
            not self.is_stub_or_invalid
            and self.tests_total > 0
            and self.tests_passed == self.tests_total
            and math.isfinite(self.invariant_residual)
            and 0.0 <= self.invariant_residual <= 1e-4
        )


def _strip_lean4_comments(source: str) -> str:
    """Strip Lean 4 block comments (`/- ... -/`) and line comments (`-- ...`)."""
    import re

    if not isinstance(source, str):
        raise ValueError("source must be a string")
    no_block = re.sub(r"/-[\s\S]*?-/", " ", source)
    return "\n".join(line.split("--", 1)[0] for line in no_block.splitlines())


def _has_ast_stub_or_syntax_error(code: str, *, domain: str = "code") -> tuple[bool, str | None]:
    import re

    if not isinstance(code, str):
        raise ValueError("code must be a string")
    if not isinstance(domain, str) or not domain.strip():
        raise ValueError("domain must be a non-empty string")
    stripped = code.strip()
    if not stripped:
        return True, "EMPTY_RESPONSE"
    if domain in ("lean4", "math_lean4"):
        cleaned_lean = _strip_lean4_comments(stripped)
        if not cleaned_lean.strip():
            return True, "EMPTY_RESPONSE"
        if re.search(r"\b(sorry|admit)\b", cleaned_lean):
            return True, "LEAN4_SORRY_DETECTED"
        m_ax = re.search(r"\b(axiom|constant|unsafe|implemented_by|extern)\b", cleaned_lean)
        if m_ax is not None:
            return True, f"LEAN4_AXIOM_DETECTED:{m_ax.group(1)}"
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
        if all(
            isinstance(stmt, ast.Pass)
            or (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value is Ellipsis
            )
            for stmt in body
        ):
            first = body[0]
            if isinstance(first, ast.Pass):
                return True, f"AST_STUB_PASS:{fn.name}"
            return True, f"AST_STUB_ELLIPSIS:{fn.name}"
        last_stmt = body[-1]
        if isinstance(last_stmt, ast.Raise):
            if len(body) == 1:
                return True, f"AST_STUB_RAISE_ONLY:{fn.name}"
            exc_node = last_stmt.exc
            exc_name = None
            if isinstance(exc_node, ast.Name):
                exc_name = exc_node.id
            elif isinstance(exc_node, ast.Call) and isinstance(exc_node.func, ast.Name):
                exc_name = exc_node.func.id
            if exc_name == "NotImplementedError":
                return True, f"AST_STUB_RAISE_ONLY:{fn.name}"
        if (
            isinstance(last_stmt, ast.Return)
            and isinstance(last_stmt.value, ast.Name)
            and last_stmt.value.id == "NotImplemented"
        ):
            return True, f"AST_STUB_RETURN_NOT_IMPLEMENTED:{fn.name}"
    return False, None


def _build_ast_namespace(source_text: str) -> dict[str, Any]:
    """Interpret pure numerical/functional AST definitions without dynamic code execution."""
    import operator

    bin_ops: dict[type[ast.operator], Any] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    unary_ops: dict[type[ast.unaryop], Any] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
        ast.Not: operator.not_,
    }
    cmp_ops: dict[type[ast.cmpop], Any] = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }
    safe_builtins: dict[str, Any] = {
        "abs": abs,
        "min": min,
        "max": max,
        "len": len,
        "range": range,
        "float": float,
        "int": int,
        "round": round,
        "sum": sum,
    }
    ns: dict[str, Any] = {"__builtins__": safe_builtins}

    def _eval_expr(node: ast.AST, env: dict[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id in ns:
                return ns[node.id]
            if node.id in safe_builtins:
                return safe_builtins[node.id]
            raise NameError(node.id)
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary_ops:
            return unary_ops[type(node.op)](_eval_expr(node.operand, env))
        if isinstance(node, ast.BinOp) and type(node.op) in bin_ops:
            return bin_ops[type(node.op)](_eval_expr(node.left, env), _eval_expr(node.right, env))
        if isinstance(node, ast.Tuple):
            return tuple(_eval_expr(elt, env) for elt in node.elts)
        if isinstance(node, ast.List):
            return [_eval_expr(elt, env) for elt in node.elts]
        if isinstance(node, ast.Compare):
            left_val = _eval_expr(node.left, env)
            for op, comp in zip(node.ops, node.comparators):
                right_val = _eval_expr(comp, env)
                if type(op) not in cmp_ops or not cmp_ops[type(op)](left_val, right_val):
                    return False
                left_val = right_val
            return True
        if isinstance(node, ast.IfExp):
            return _eval_expr(node.body, env) if _eval_expr(node.test, env) else _eval_expr(node.orelse, env)
        if isinstance(node, ast.Call):
            fn = _eval_expr(node.func, env)
            args = [_eval_expr(a, env) for a in node.args]
            kwargs = {kw.arg: _eval_expr(kw.value, env) for kw in node.keywords if kw.arg is not None}
            return fn(*args, **kwargs)
        raise ValueError(f"Unsupported AST expression: {type(node).__name__}")

    def _make_fn(fn_def: ast.FunctionDef) -> Any:
        arg_names = [a.arg for a in fn_def.args.args]

        def _runner(*call_args: Any) -> Any:
            local_env: dict[str, Any] = dict(zip(arg_names, call_args))
            for stmt in fn_def.body:
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    continue
                if isinstance(stmt, ast.Assign):
                    val = _eval_expr(stmt.value, local_env)
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            local_env[target.id] = val
                        elif isinstance(target, ast.Tuple):
                            for sub_t, sub_v in zip(target.elts, val):
                                if isinstance(sub_t, ast.Name):
                                    local_env[sub_t.id] = sub_v
                elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                    cur = local_env[stmt.target.id]
                    rhs = _eval_expr(stmt.value, local_env)
                    local_env[stmt.target.id] = bin_ops[type(stmt.op)](cur, rhs)
                elif isinstance(stmt, ast.If):
                    branch = stmt.body if _eval_expr(stmt.test, local_env) else stmt.orelse
                    for b_stmt in branch:
                        if isinstance(b_stmt, ast.Return):
                            return _eval_expr(b_stmt.value, local_env) if b_stmt.value is not None else None
                elif isinstance(stmt, ast.Return):
                    return _eval_expr(stmt.value, local_env) if stmt.value is not None else None
            return None

        return _runner

    tree = ast.parse(source_text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            ns[node.name] = _make_fn(node)
    return ns


def evaluate_bee_candidate_energy(
    *,
    candidate_id: str,
    response_text: str,
    tests_passed: int = 0,
    tests_total: int = 0,
    test_callables: Sequence[Any] | None = None,
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
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("candidate_id must be a non-empty string.")
    if not isinstance(response_text, str):
        raise ValueError("response_text must be a string.")
    if not isinstance(domain, str) or not domain.strip():
        raise ValueError("domain must be a non-empty string.")
    if not isinstance(timed_out_or_crashed, bool):
        raise ValueError("timed_out_or_crashed must be a boolean.")
    if isinstance(tests_passed, bool) or isinstance(tests_total, bool):
        raise ValueError("tests_passed and tests_total must be integers, not booleans.")
    if isinstance(latency_ms, bool) or isinstance(memory_mb, bool) or isinstance(invariant_residual, bool):
        raise ValueError("latency_ms, memory_mb, and invariant_residual must be numeric, not booleans.")
    if (
        isinstance(w_fail, bool)
        or isinstance(w_inv, bool)
        or isinstance(w_lat, bool)
        or isinstance(w_mem, bool)
        or not (
            math.isfinite(float(w_fail))
            and float(w_fail) > 0.0
            and math.isfinite(float(w_inv))
            and float(w_inv) >= 0.0
            and math.isfinite(float(w_lat))
            and float(w_lat) >= 0.0
            and math.isfinite(float(w_mem))
            and float(w_mem) >= 0.0
        )
    ):
        raise ValueError("energy weights w_fail (> 0) and w_inv, w_lat, w_mem (>= 0) must be finite.")
    is_stub, stub_reason = _has_ast_stub_or_syntax_error(response_text, domain=domain)
    if test_callables is not None and not timed_out_or_crashed and not is_stub:
        tests_total = len(test_callables)
        passed_count = 0
        try:
            ns = _build_ast_namespace(response_text)
            for tc in test_callables:
                try:
                    if bool(tc(ns)):
                        passed_count += 1
                except Exception:
                    pass
        except Exception as exc:
            timed_out_or_crashed = True
            error_message = f"CANDIDATE_EXEC_ERROR:{exc}"
        tests_passed = passed_count
    elif test_callables is not None and is_stub and tests_total <= 0:
        tests_total = len(test_callables)
        tests_passed = 0

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

    is_stub, stub_reason = _has_ast_stub_or_syntax_error(response_text, domain=domain)
    metrics_valid = (
        math.isfinite(float(invariant_residual))
        and float(invariant_residual) >= 0.0
        and math.isfinite(float(latency_ms))
        and float(latency_ms) >= 0.0
        and math.isfinite(float(memory_mb))
        and float(memory_mb) >= 0.0
    )
    if is_stub or tests_total <= 0 or not metrics_valid:
        return BeeEnergyEvaluation(
            candidate_id=candidate_id,
            response_text=response_text,
            tests_passed=max(0, int(tests_passed)),
            tests_total=max(0, int(tests_total)),
            latency_ms=max(0.0, float(latency_ms)) if math.isfinite(float(latency_ms)) else 0.0,
            memory_mb=max(0.0, float(memory_mb)) if math.isfinite(float(memory_mb)) else 0.0,
            invariant_residual=float(invariant_residual),
            is_stub_or_invalid=True,
            failure_reason=stub_reason or error_message or (
                "INVALID_NONFINITE_OR_NEGATIVE_METRIC" if not metrics_valid else "NO_HIDDEN_TESTS"
            ),
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
    if not isinstance(failing_code, str) or not isinstance(stderr_or_reason, str):
        raise ValueError("failing_code and stderr_or_reason must be strings.")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError("max_chars must be a positive integer.")
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
    if isinstance(early_stop_energy_ceiling, bool) or not isinstance(early_stop_energy_ceiling, (int, float)) or not math.isfinite(float(early_stop_energy_ceiling)) or float(early_stop_energy_ceiling) <= 0.0:
        raise ValueError("early_stop_energy_ceiling must be a strictly positive finite float.")
    if any(isinstance(a.total_energy, bool) or not math.isfinite(float(a.total_energy)) for a in attempts):
        raise ValueError("all attempt energies must be finite.")
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
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")
    if isinstance(min_delta_energy, bool) or not isinstance(min_delta_energy, (int, float)) or not math.isfinite(float(min_delta_energy)) or float(min_delta_energy) <= 0.0:
        raise ValueError("min_delta_energy must be a strictly positive finite float.")
    if any(isinstance(a.total_energy, bool) or not math.isfinite(float(a.total_energy)) for a in attempts):
        raise ValueError("all attempt energies must be finite.")
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
    if isinstance(block_size, bool) or not isinstance(block_size, int) or int(block_size) < 1:
        raise ValueError("block_size must be >= 1.")
    if (
        isinstance(min_block_gain, bool)
        or isinstance(max_tail_std, bool)
        or not isinstance(min_block_gain, (int, float))
        or not isinstance(max_tail_std, (int, float))
        or not math.isfinite(float(min_block_gain))
        or not math.isfinite(float(max_tail_std))
        or float(max_tail_std) < 0.0
    ):
        raise ValueError("min_block_gain and max_tail_std (>= 0) must be finite floats.")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)) for x in round_losses):
        raise ValueError("round_losses must contain only finite floats.")
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
    if isinstance(only_child_right, bool) or isinstance(only_parent_right, bool) or not isinstance(only_child_right, int) or not isinstance(only_parent_right, int):
        raise ValueError("discordant counts must be non-negative integers.")
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
    if any(isinstance(d, bool) or not isinstance(d, (int, float)) or not math.isfinite(float(d)) for d in aa_median_deltas):
        raise ValueError("aa_median_deltas must contain only finite floats.")
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

    def __post_init__(self) -> None:
        if not isinstance(self.promoted, bool):
            raise ValueError("promoted must be a boolean")
        if (
            isinstance(self.median_delta_energy, bool)
            or not isinstance(self.median_delta_energy, (int, float))
            or not math.isfinite(float(self.median_delta_energy))
        ):
            raise ValueError("median_delta_energy must be a finite number")
        if (
            isinstance(self.parent_regressions_failed, bool)
            or not isinstance(self.parent_regressions_failed, int)
            or self.parent_regressions_failed < 0
        ):
            raise ValueError("parent_regressions_failed must be a non-negative integer")
        if not isinstance(self.shuffled_control_gained, bool):
            raise ValueError("shuffled_control_gained must be a boolean")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")


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
    if isinstance(parent_regressions_failed, bool) or not isinstance(parent_regressions_failed, int) or int(parent_regressions_failed) < 0:
        raise ValueError("parent_regressions_failed must be >= 0.")
    if not isinstance(shuffled_control_gained, bool):
        raise ValueError("shuffled_control_gained must be a boolean.")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)) or not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be a finite float in (0.0, 1.0).")
    if isinstance(noise_margin, bool) or not isinstance(noise_margin, (int, float)) or not math.isfinite(float(noise_margin)) or float(noise_margin) <= 0.0:
        raise ValueError("noise_margin must be a strictly positive finite float from calibrate_noise_margin.")
    if not all(
        not isinstance(p, bool)
        and not isinstance(c, bool)
        and isinstance(p, (int, float))
        and isinstance(c, (int, float))
        and math.isfinite(float(p))
        and math.isfinite(float(c))
        for p, c in zip(parent_energies, child_energies)
    ):
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
