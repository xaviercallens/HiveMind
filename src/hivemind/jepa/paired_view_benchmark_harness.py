# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Pre-Registered E15-J and P2 Paired-View Benchmark Harness (Paper v4.4.0 Section 15.4 J2).

Implements Opportunity J2 (DOI ``10.5281/zenodo.24553269``) strictly on CPU using Python stdlib
and NumPy (zero network calls, zero subprocess invocations, zero stubs):

1. **Natural Paired-View Builders (Python & Lean 4):**
   - Python ``docstring/specification <-> function body`` pairs extracted via AST or explicit
     specification/body inputs, rejecting stub or malformed code.
   - Lean 4 ``theorem signature <-> proof term`` pairs extracted with comment-aware ``:=``
     splitting and zero-``sorry`` / ``admit`` / ``sorryAx`` verification.
   - Zero-overlap split decontamination and verification across ``item_id``, normalized
     ``view_a_spec``, normalized ``view_b_target``, and canonical ``pair_sha256``.
2. **Hash-Deterministic Loss Dropout & Shuffled-Pair Control (Arm S):**
   - Reuses :func:`hivemind.jepa.spec.step_uniform` and :func:`hivemind.jepa.spec.apply_aux`
     to evaluate ``sha256(seed, step, row, k)`` loss dropout deterministically.
   - Constructs a strict derangement control (Arm ``S``) via Sattolo's cyclic permutation
     where zero items remain paired with their own or identical target view, enabling
     compute-matched diagnostics against Arm ``A`` (SFT) and Arm ``J`` (JEPA).
3. **Pre-Registered Promotion Gate (Exact McNemar & >= +3.0 pts):**
   - Evaluates paired binary task outcomes across Arms ``A``, ``J``, and ``S`` using
     :func:`hivemind.verification.bee_energy.exact_mcnemar_p`, enforcing ``>= +3.0`` pts,
     two-sided McNemar ``p < 0.05``, negative-control non-significance, general-NLL
     guard rail (``<= +0.05`` nat), and cosine non-collapse, emitting canonical SHA-256 receipts.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from hivemind.jepa.spec import JepaSpec, apply_aux, step_uniform
from hivemind.verification.bee_energy import exact_mcnemar_p

E15J_SCHEMA: str = "hivemind.jepa.paired_view_benchmark/1"
E15J_RECEIPT_OBJECT: str = "hivemind.jepa.e15j_p2_gate_receipt"

VALID_DOMAINS: tuple[str, ...] = ("python_code", "lean4_proof")
VALID_SPLITS: tuple[str, ...] = ("train", "eval", "holdout")
VALID_ARMS: tuple[str, ...] = ("A", "J", "S")
ARM_DESCRIPTIONS: dict[str, str] = {
    "A": "SFT",
    "J": "JEPA",
    "S": "SHUFFLED_CONTROL",
}

DEFAULT_MBPP_EVAL_N: int = 257
DEFAULT_MIN_GAIN_PTS: float = 3.0
DEFAULT_MCNEMAR_ALPHA: float = 0.05
DEFAULT_MAX_GENERAL_NLL_DELTA: float = 0.05
DEFAULT_COLLAPSE_FLOOR: float = 0.05
DEFAULT_PREDICTOR_TOKENS_K: int = 1
DEFAULT_INFONCE_TEMPERATURE: float = 0.07

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_LEAN_DECL_RE = re.compile(r"\b(?:theorem|lemma|def|instance)\b")
_LEAN_HOLE_RE = re.compile(r"\b(?:sorry|admit|sorryAx)\b")


# ---------------------------------------------------------------------------
# 1. Validation, Normalization & Hashing
# ---------------------------------------------------------------------------


def _require_non_empty_str(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{name} must be a non-empty string")
    return stripped


def _require_int(value: Any, name: str, *, min_value: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return int(value)


def _require_finite_float(
    value: Any,
    name: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    strict_min: bool = False,
    strict_max: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if min_value is not None:
        if (strict_min and out <= min_value) or (not strict_min and out < min_value):
            op = ">" if strict_min else ">="
            raise ValueError(f"{name} must be {op} {min_value}, got {out}")
    if max_value is not None:
        if (strict_max and out >= max_value) or (not strict_max and out > max_value):
            op = "<" if strict_max else "<="
            raise ValueError(f"{name} must be {op} {max_value}, got {out}")
    return out


def _require_bool_sequence(values: Sequence[bool], name: str) -> tuple[bool, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of booleans")
    if not values:
        raise ValueError(f"{name} must be non-empty")
    out: list[bool] = []
    for idx, val in enumerate(values):
        if not isinstance(val, bool):
            raise TypeError(f"{name}[{idx}] must be bool, got {type(val).__name__}")
        out.append(val)
    return tuple(out)


def normalize_view_text(text: str) -> str:
    """Normalize line endings and whitespace for canonical decontamination and hashing."""
    cleaned = _require_non_empty_str(text, "text").replace("\r\n", "\n")
    return re.sub(r"\s+", " ", cleaned).strip()


def sha256_text(text: str) -> str:
    """Return canonical SHA-256 hex digest of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_payload_sha256(payload: Any) -> str:
    """Return canonical lowercase SHA-256 hex digest for a JSON-serializable payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PairedViewHarnessConfig:
    """Pre-registered configuration for E15-J and P2 paired-view benchmarking."""

    jepa_spec: JepaSpec = field(
        default_factory=lambda: JepaSpec(
            mode="pairs",
            lam=1.0,
            stopgrad=True,
            center=False,
            dropout=0.0,
            seed=0,
            collapse_floor=DEFAULT_COLLAPSE_FLOOR,
        )
    )
    k_predictor_tokens: int = DEFAULT_PREDICTOR_TOKENS_K
    min_gain_pts: float = DEFAULT_MIN_GAIN_PTS
    mcnemar_alpha: float = DEFAULT_MCNEMAR_ALPHA
    max_general_nll_delta: float = DEFAULT_MAX_GENERAL_NLL_DELTA
    require_j_beats_s: bool = True
    infonce_temperature: float = DEFAULT_INFONCE_TEMPERATURE

    def __post_init__(self) -> None:
        if not isinstance(self.jepa_spec, JepaSpec):
            raise TypeError("jepa_spec must be a JepaSpec")
        if self.jepa_spec.mode != "pairs":
            raise ValueError(f"PairedViewHarnessConfig requires jepa_spec.mode == 'pairs', got {self.jepa_spec.mode!r}")
        _require_int(self.k_predictor_tokens, "k_predictor_tokens", min_value=0)
        _require_finite_float(self.min_gain_pts, "min_gain_pts", min_value=0.0, max_value=100.0)
        _require_finite_float(
            self.mcnemar_alpha, "mcnemar_alpha", min_value=0.0, max_value=1.0, strict_min=True, strict_max=True
        )
        _require_finite_float(self.max_general_nll_delta, "max_general_nll_delta", min_value=0.0)
        if not isinstance(self.require_j_beats_s, bool):
            raise TypeError("require_j_beats_s must be a bool")
        _require_finite_float(self.infonce_temperature, "infonce_temperature", min_value=0.0, strict_min=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "jepa_spec": asdict(self.jepa_spec),
            "jepa_spec_sha256": self.jepa_spec.spec_sha256(),
            "k_predictor_tokens": self.k_predictor_tokens,
            "min_gain_pts": self.min_gain_pts,
            "mcnemar_alpha": self.mcnemar_alpha,
            "max_general_nll_delta": self.max_general_nll_delta,
            "require_j_beats_s": self.require_j_beats_s,
            "infonce_temperature": self.infonce_temperature,
        }

    def config_sha256(self) -> str:
        return compute_payload_sha256(self.to_dict())


# ---------------------------------------------------------------------------
# 2. Paired-View Extractors, Records & Zero-Overlap Split Verification
# ---------------------------------------------------------------------------


def _is_python_stub_body(fn_node: ast.FunctionDef | ast.AsyncFunctionDef, body_stmts: list[ast.stmt]) -> bool:
    if not body_stmts:
        return True
    if len(body_stmts) == 1:
        stmt = body_stmts[0]
        if isinstance(stmt, ast.Pass):
            return True
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is Ellipsis:
            return True
        if isinstance(stmt, ast.Raise) and stmt.exc is not None:
            exc = stmt.exc
            if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                return True
            if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) and exc.func.id == "NotImplementedError":
                return True
        if fn_node.returns is not None and isinstance(stmt, ast.Return):
            if stmt.value is None or (isinstance(stmt.value, ast.Constant) and stmt.value.value is None):
                return True
    return False


def extract_python_paired_view(
    source_code: str,
    *,
    fallback_spec: str = "",
    function_name: str | None = None,
) -> tuple[str, str]:
    """Extract normalized ``(docstring_or_spec, function_body)`` from Python source via AST."""
    code = _require_non_empty_str(source_code, "source_code")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Invalid Python source: {exc}") from exc

    candidates = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if function_name is not None:
        target_name = _require_non_empty_str(function_name, "function_name")
        matched = [n for n in candidates if n.name == target_name]
        if not matched:
            raise ValueError(f"Function {target_name!r} not found in Python source")
        fn_node = matched[0]
    elif candidates:
        fn_node = candidates[0]
    else:
        raise ValueError("No top-level Python function found in source_code")

    has_docstring = (
        bool(fn_node.body)
        and isinstance(fn_node.body[0], ast.Expr)
        and isinstance(fn_node.body[0].value, ast.Constant)
        and isinstance(fn_node.body[0].value.value, str)
    )
    doc_text = (ast.get_docstring(fn_node, clean=True) or "").strip() if has_docstring else ""
    spec_text = doc_text if doc_text else fallback_spec.strip()
    if not spec_text:
        raise ValueError(f"Function {fn_node.name!r} has neither a docstring nor a fallback_spec")

    body_stmts = list(fn_node.body[1:] if has_docstring else fn_node.body)
    if _is_python_stub_body(fn_node, body_stmts):
        raise ValueError(f"Function {fn_node.name!r} has an empty or placeholder body")

    fn_clean = copy.copy(fn_node)
    fn_clean.body = body_stmts
    body_source = ast.unparse(fn_clean)
    return spec_text, body_source


def _strip_lean4_comments(lean_source: str) -> str:
    """Strip Lean 4 ``--`` line comments and nested ``/- ... -/`` block comments."""
    out: list[str] = []
    n = len(lean_source)
    i = 0
    block_depth = 0
    in_string = False

    while i < n:
        ch = lean_source[i]
        nxt = lean_source[i + 1] if i + 1 < n else ""

        if block_depth > 0:
            if ch == "/" and nxt == "-":
                block_depth += 1
                i += 2
            elif ch == "-" and nxt == "/":
                block_depth -= 1
                i += 2
                if block_depth == 0:
                    out.append(" ")
            else:
                if ch == "\n":
                    out.append("\n")
                i += 1
            continue

        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(nxt)
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
        elif ch == "-" and nxt == "-":
            while i < n and lean_source[i] != "\n":
                i += 1
            out.append(" ")
        elif ch == "/" and nxt == "-":
            block_depth = 1
            i += 2
        else:
            out.append(ch)
            i += 1

    if block_depth != 0:
        raise ValueError("Unterminated Lean 4 block comment '/-'")
    return "".join(out)


def _find_top_level_assign(cleaned_lean: str) -> int:
    """Locate the first top-level ``:=`` separator outside brackets and string literals."""
    depth = 0
    in_string = False
    i = 0
    n = len(cleaned_lean)
    while i < n:
        ch = cleaned_lean[i]
        if in_string:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            i += 1
        elif ch in "([{":
            depth += 1
            i += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
            i += 1
        elif depth == 0 and cleaned_lean[i : i + 2] == ":=":
            return i
        else:
            i += 1
    return -1


def extract_lean4_paired_view(lean_source: str) -> tuple[str, str]:
    """Extract ``(theorem_signature, proof_term)`` from Lean 4 declaration without ``sorry``."""
    raw = _require_non_empty_str(lean_source, "lean_source")
    cleaned = _strip_lean4_comments(raw)
    hole_match = _LEAN_HOLE_RE.search(cleaned)
    if hole_match:
        raise ValueError(f"Lean 4 proof contains forbidden hole {hole_match.group(0)!r}")
    if not _LEAN_DECL_RE.search(cleaned):
        raise ValueError("Lean 4 declaration must contain theorem, lemma, def, or instance")
    assign_idx = _find_top_level_assign(cleaned)
    if assign_idx < 0:
        raise ValueError("Lean 4 declaration missing top-level ':=' proof separator")

    sig_text = "\n".join(line.rstrip() for line in cleaned[:assign_idx].strip().splitlines()).strip()
    proof_text = "\n".join(line.rstrip() for line in cleaned[assign_idx + 2 :].strip().splitlines()).strip()
    if not sig_text or not proof_text:
        raise ValueError("Lean 4 declaration must have non-empty signature and proof term")
    return sig_text, proof_text


@dataclass(frozen=True)
class PairedViewRecord:
    """Canonical paired-view training or evaluation item for E15-J and P2."""

    item_id: str
    domain: str
    split: str
    view_a_spec: str
    view_b_target: str
    k_predictor_tokens: int = DEFAULT_PREDICTOR_TOKENS_K
    source: str = "e15_j"
    view_a_sha256: str = field(init=False)
    view_b_sha256: str = field(init=False)
    pair_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        item_id = _require_non_empty_str(self.item_id, "item_id")
        object.__setattr__(self, "item_id", item_id)
        if self.domain not in VALID_DOMAINS:
            raise ValueError(f"domain must be one of {VALID_DOMAINS}, got {self.domain!r}")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {self.split!r}")
        view_a = self.view_a_spec.replace("\r\n", "\n").strip()
        view_b = self.view_b_target.replace("\r\n", "\n").strip()
        if not view_a or not view_b:
            raise ValueError("view_a_spec and view_b_target must be non-empty")
        object.__setattr__(self, "view_a_spec", view_a)
        object.__setattr__(self, "view_b_target", view_b)
        _require_int(self.k_predictor_tokens, "k_predictor_tokens", min_value=0)
        object.__setattr__(self, "source", _require_non_empty_str(self.source, "source"))

        va_hash = sha256_text(normalize_view_text(view_a))
        vb_hash = sha256_text(normalize_view_text(view_b))
        pair_hash = compute_payload_sha256(
            {"domain": self.domain, "view_a_sha256": va_hash, "view_b_sha256": vb_hash}
        )
        object.__setattr__(self, "view_a_sha256", va_hash)
        object.__setattr__(self, "view_b_sha256", vb_hash)
        object.__setattr__(self, "pair_sha256", pair_hash)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "domain": self.domain,
            "split": self.split,
            "view_a_spec": self.view_a_spec,
            "view_b_target": self.view_b_target,
            "k_predictor_tokens": self.k_predictor_tokens,
            "source": self.source,
            "view_a_sha256": self.view_a_sha256,
            "view_b_sha256": self.view_b_sha256,
            "pair_sha256": self.pair_sha256,
        }


def build_python_paired_record(
    item_id: str,
    *,
    source_code: str | None = None,
    docstring_or_spec: str | None = None,
    function_body: str | None = None,
    function_name: str | None = None,
    split: str = "train",
    k_predictor_tokens: int = DEFAULT_PREDICTOR_TOKENS_K,
    source: str = "mbpp",
) -> PairedViewRecord:
    """Build a deterministic Python ``docstring/spec <-> function body`` paired-view record."""
    if source_code is not None:
        if function_body is not None:
            raise ValueError("Provide either source_code or explicit function_body, not both")
        fallback = docstring_or_spec or ""
        spec_text, body_text = extract_python_paired_view(
            source_code, fallback_spec=fallback, function_name=function_name
        )
    else:
        spec_text = _require_non_empty_str(docstring_or_spec, "docstring_or_spec")
        body_text = _require_non_empty_str(function_body, "function_body")
    return PairedViewRecord(
        item_id=item_id,
        domain="python_code",
        split=split,
        view_a_spec=spec_text,
        view_b_target=body_text,
        k_predictor_tokens=k_predictor_tokens,
        source=source,
    )


def build_lean4_paired_record(
    item_id: str,
    *,
    lean_source: str | None = None,
    theorem_signature: str | None = None,
    proof_term: str | None = None,
    split: str = "train",
    k_predictor_tokens: int = DEFAULT_PREDICTOR_TOKENS_K,
    source: str = "granite_lean4",
) -> PairedViewRecord:
    """Build a deterministic Lean 4 ``theorem signature <-> proof term`` paired-view record."""
    if lean_source is not None:
        if theorem_signature is not None or proof_term is not None:
            raise ValueError("Provide either lean_source or (theorem_signature, proof_term), not both")
        sig_text, proof_text = extract_lean4_paired_view(lean_source)
    else:
        sig_text = _require_non_empty_str(theorem_signature, "theorem_signature")
        proof_text = _require_non_empty_str(proof_term, "proof_term")
        cleaned = _strip_lean4_comments(f"{sig_text} := {proof_text}")
        hole_match = _LEAN_HOLE_RE.search(cleaned)
        if hole_match:
            raise ValueError(f"Lean 4 proof contains forbidden hole {hole_match.group(0)!r}")
    return PairedViewRecord(
        item_id=item_id,
        domain="lean4_proof",
        split=split,
        view_a_spec=sig_text,
        view_b_target=proof_text,
        k_predictor_tokens=k_predictor_tokens,
        source=source,
    )


def build_paired_view_records(
    items: Sequence[Mapping[str, Any]],
    *,
    default_k_predictor_tokens: int = DEFAULT_PREDICTOR_TOKENS_K,
) -> tuple[PairedViewRecord, ...]:
    """Deterministically construct and ID-deduplicate a batch of Python or Lean 4 records."""
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)) or not items:
        raise ValueError("items must be a non-empty sequence of mappings")
    _require_int(default_k_predictor_tokens, "default_k_predictor_tokens", min_value=0)

    records: list[PairedViewRecord] = []
    seen_ids: set[str] = set()
    for idx, raw in enumerate(items):
        if not isinstance(raw, Mapping):
            raise TypeError(f"items[{idx}] must be a Mapping")
        domain = raw.get("domain")
        item_id = _require_non_empty_str(raw.get("item_id"), f"items[{idx}].item_id")
        if item_id in seen_ids:
            raise ValueError(f"Duplicate item_id {item_id!r} in build_paired_view_records")
        seen_ids.add(item_id)
        split = str(raw.get("split", "train"))
        k_tok = raw.get("k_predictor_tokens", default_k_predictor_tokens)

        if domain == "python_code":
            rec = build_python_paired_record(
                item_id,
                source_code=raw.get("source_code"),
                docstring_or_spec=raw.get("docstring_or_spec"),
                function_body=raw.get("function_body"),
                function_name=raw.get("function_name"),
                split=split,
                k_predictor_tokens=k_tok,
                source=str(raw.get("source", "mbpp")),
            )
        elif domain == "lean4_proof":
            rec = build_lean4_paired_record(
                item_id,
                lean_source=raw.get("lean_source"),
                theorem_signature=raw.get("theorem_signature"),
                proof_term=raw.get("proof_term"),
                split=split,
                k_predictor_tokens=k_tok,
                source=str(raw.get("source", "granite_lean4")),
            )
        else:
            raise ValueError(f"Unsupported domain {domain!r} at items[{idx}]")
        records.append(rec)
    return tuple(records)


@dataclass(frozen=True)
class SplitIntegrityReport:
    """Zero-overlap leakage verification report between train and eval/holdout splits."""

    train_count: int
    eval_count: int
    item_id_overlap: tuple[str, ...]
    view_a_overlap_count: int
    view_b_overlap_count: int
    pair_sha256_overlap_count: int
    is_disjoint: bool
    report_sha256: str


def decontaminate_paired_records(
    train_records: Sequence[PairedViewRecord],
    eval_records: Sequence[PairedViewRecord],
) -> tuple[tuple[PairedViewRecord, ...], tuple[str, ...]]:
    """Drop train records overlapping ``eval_records`` on ID, normalized view_a, or view_b."""
    eval_ids = {r.item_id for r in eval_records}
    eval_a = {r.view_a_sha256 for r in eval_records}
    eval_b = {r.view_b_sha256 for r in eval_records}
    eval_p = {r.pair_sha256 for r in eval_records}

    kept: list[PairedViewRecord] = []
    dropped: list[str] = []
    for rec in train_records:
        collides = (
            rec.item_id in eval_ids
            or rec.view_a_sha256 in eval_a
            or rec.view_b_sha256 in eval_b
            or rec.pair_sha256 in eval_p
        )
        if collides:
            dropped.append(rec.item_id)
        else:
            kept.append(rec)
    return tuple(kept), tuple(dropped)


def verify_zero_overlap_splits(
    train_records: Sequence[PairedViewRecord],
    eval_records: Sequence[PairedViewRecord],
    *,
    raise_on_overlap: bool = True,
) -> SplitIntegrityReport:
    """Verify zero ID, specification, target-body, or pair-hash overlap across splits."""
    if not train_records or not eval_records:
        raise ValueError("train_records and eval_records must both be non-empty")

    for split_name, recs in (("train_records", train_records), ("eval_records", eval_records)):
        seen: set[str] = set()
        for r in recs:
            if not isinstance(r, PairedViewRecord):
                raise TypeError(f"{split_name} must contain PairedViewRecord instances")
            if r.item_id in seen:
                raise ValueError(f"Duplicate item_id {r.item_id!r} within {split_name}")
            seen.add(r.item_id)

    t_ids = {r.item_id for r in train_records}
    e_ids = {r.item_id for r in eval_records}
    t_a = {r.view_a_sha256 for r in train_records}
    e_a = {r.view_a_sha256 for r in eval_records}
    t_b = {r.view_b_sha256 for r in train_records}
    e_b = {r.view_b_sha256 for r in eval_records}
    t_p = {r.pair_sha256 for r in train_records}
    e_p = {r.pair_sha256 for r in eval_records}

    id_overlap = tuple(sorted(t_ids & e_ids))
    a_overlap = len(t_a & e_a)
    b_overlap = len(t_b & e_b)
    p_overlap = len(t_p & e_p)
    is_disjoint = (len(id_overlap) == 0) and (a_overlap == 0) and (b_overlap == 0) and (p_overlap == 0)

    core = {
        "train_count": len(train_records),
        "eval_count": len(eval_records),
        "item_id_overlap": list(id_overlap),
        "view_a_overlap_count": a_overlap,
        "view_b_overlap_count": b_overlap,
        "pair_sha256_overlap_count": p_overlap,
        "is_disjoint": is_disjoint,
    }
    report = SplitIntegrityReport(
        train_count=len(train_records),
        eval_count=len(eval_records),
        item_id_overlap=id_overlap,
        view_a_overlap_count=a_overlap,
        view_b_overlap_count=b_overlap,
        pair_sha256_overlap_count=p_overlap,
        is_disjoint=is_disjoint,
        report_sha256=compute_payload_sha256(core),
    )
    if raise_on_overlap and not is_disjoint:
        raise ValueError(
            "Zero-overlap split violation: "
            f"ids={len(id_overlap)}, view_a={a_overlap}, view_b={b_overlap}, pair={p_overlap}"
        )
    return report


# ---------------------------------------------------------------------------
# 3. Hash-Deterministic Loss Dropout & Shuffled-Pair Control (Arm S)
# ---------------------------------------------------------------------------


def dropout_draw(seed: int, step: int, row: int, k: int, *, salt: str = "dropout") -> float:
    """Deterministic ``[0, 1)`` draw from ``sha256(salt, seed, step, row, k)``."""
    s = _require_int(seed, "seed")
    st = _require_int(step, "step", min_value=0)
    rw = _require_int(row, "row", min_value=-1)
    kk = _require_int(k, "k", min_value=0)
    clean_salt = _require_non_empty_str(salt, "salt")
    return step_uniform(s, st, rw, kk, salt=clean_salt)


def should_apply_jepa_loss(spec: JepaSpec, step: int, *, row: int = -1, k: int = 0) -> bool:
    """Deterministic replayable loss-dropout predicate using ``sha256(seed, step, row, k)``."""
    if not isinstance(spec, JepaSpec):
        raise TypeError("spec must be a JepaSpec")
    st = _require_int(step, "step", min_value=0)
    rw = _require_int(row, "row", min_value=-1)
    kk = _require_int(k, "k", min_value=0)
    if not spec.active:
        return False
    if rw == -1 and kk == 0:
        return apply_aux(spec, st)
    return spec.dropout == 0.0 or step_uniform(spec.seed, st, rw, kk, "dropout") >= spec.dropout


def compute_step_dropout_mask(
    spec: JepaSpec,
    step: int,
    n_rows: int,
    *,
    k_predictor_tokens: int = DEFAULT_PREDICTOR_TOKENS_K,
) -> tuple[bool, ...]:
    """Return deterministic per-row JEPA active mask for ``(spec.seed, step, row, k)``."""
    if not isinstance(spec, JepaSpec):
        raise TypeError("spec must be a JepaSpec")
    st = _require_int(step, "step", min_value=0)
    n = _require_int(n_rows, "n_rows", min_value=1)
    kk = _require_int(k_predictor_tokens, "k_predictor_tokens", min_value=0)
    return tuple(should_apply_jepa_loss(spec, st, row=row, k=kk) for row in range(n))


@dataclass(frozen=True)
class ShuffledPairRecord:
    """Shuffled-pair negative control (Arm S) record with strict derangement guarantee."""

    item_id: str
    domain: str
    split: str
    view_a_spec: str
    shuffled_view_b_target: str
    source_target_item_id: str
    original_index: int
    donor_index: int
    k_predictor_tokens: int
    pair_sha256: str


def _derange_indices(n: int, *, seed: int, step: int, k: int) -> list[int]:
    """Generate a single-cycle Sattolo derangement of ``range(n)`` via ``sha256(seed, step, row, k)``."""
    _require_int(n, "n", min_value=2)
    s = _require_int(seed, "seed")
    st = _require_int(step, "step", min_value=0)
    kk = _require_int(k, "k", min_value=0)

    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        u = step_uniform(s, st, i, kk, salt="e15j_derangement")
        j = min(i - 1, int(u * i))
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def build_shuffled_pair_derangement(
    records: Sequence[PairedViewRecord],
    *,
    seed: int,
    step: int = 0,
    k_predictor_tokens: int = DEFAULT_PREDICTOR_TOKENS_K,
) -> tuple[ShuffledPairRecord, ...]:
    """Construct Arm S negative-control pairs with a strict target-distinct derangement."""
    n = len(records)
    if n < 2:
        raise ValueError("build_shuffled_pair_derangement requires at least 2 records")
    for idx, rec in enumerate(records):
        if not isinstance(rec, PairedViewRecord):
            raise TypeError(f"records[{idx}] must be a PairedViewRecord")

    target_hashes = [r.view_b_sha256 for r in records]
    freq = Counter(target_hashes)
    if len(freq) < 2 or max(freq.values()) > n // 2:
        raise ValueError(
            "Target views lack sufficient diversity for a target-distinct derangement "
            f"(max_freq={max(freq.values())}, n={n})"
        )

    perm = _derange_indices(n, seed=seed, step=step, k=k_predictor_tokens)

    # Repair any coincidental hash collision across distinct items with identical view_b strings.
    for i in range(n):
        if target_hashes[perm[i]] != target_hashes[i]:
            continue
        for m in range(n):
            if m == i or perm[m] == i or perm[i] == m:
                continue
            if target_hashes[perm[m]] != target_hashes[i] and target_hashes[perm[i]] != target_hashes[m]:
                perm[i], perm[m] = perm[m], perm[i]
                break

    out: list[ShuffledPairRecord] = []
    for i, rec in enumerate(records):
        donor_idx = perm[i]
        donor = records[donor_idx]
        if donor_idx == i or donor.view_b_sha256 == rec.view_b_sha256:
            raise RuntimeError(f"Derangement post-condition violated at index {i}")
        shuf_hash = compute_payload_sha256(
            {"domain": rec.domain, "view_a_sha256": rec.view_a_sha256, "view_b_sha256": donor.view_b_sha256}
        )
        out.append(
            ShuffledPairRecord(
                item_id=rec.item_id,
                domain=rec.domain,
                split=rec.split,
                view_a_spec=rec.view_a_spec,
                shuffled_view_b_target=donor.view_b_target,
                source_target_item_id=donor.item_id,
                original_index=i,
                donor_index=donor_idx,
                k_predictor_tokens=rec.k_predictor_tokens,
                pair_sha256=shuf_hash,
            )
        )
    return tuple(out)


def build_three_arm_schedule(
    records: Sequence[PairedViewRecord],
    *,
    config: PairedViewHarnessConfig | None = None,
    step: int = 0,
) -> dict[str, Any]:
    """Assemble pre-registered Arm A (SFT), Arm J (JEPA), and Arm S (shuffled control) batches."""
    if not records:
        raise ValueError("records must be non-empty")
    cfg = config if config is not None else PairedViewHarnessConfig()
    st = _require_int(step, "step", min_value=0)

    dropout_mask = compute_step_dropout_mask(
        cfg.jepa_spec,
        st,
        len(records),
        k_predictor_tokens=cfg.k_predictor_tokens,
    )
    shuffled = build_shuffled_pair_derangement(
        records,
        seed=cfg.jepa_spec.seed,
        step=st,
        k_predictor_tokens=cfg.k_predictor_tokens,
    )
    arm_a_items = [
        {"item_id": r.item_id, "domain": r.domain, "prompt": r.view_a_spec, "target": r.view_b_target, "k": 0}
        for r in records
    ]
    arm_j_items = [
        {
            "item_id": r.item_id,
            "domain": r.domain,
            "view_a_spec": r.view_a_spec,
            "view_b_target": r.view_b_target,
            "k_predictor_tokens": cfg.k_predictor_tokens,
            "aux_active": active,
            "pair_sha256": r.pair_sha256,
        }
        for r, active in zip(records, dropout_mask, strict=True)
    ]
    arm_s_items = [
        {
            "item_id": s.item_id,
            "domain": s.domain,
            "view_a_spec": s.view_a_spec,
            "shuffled_view_b_target": s.shuffled_view_b_target,
            "source_target_item_id": s.source_target_item_id,
            "donor_index": s.donor_index,
            "k_predictor_tokens": cfg.k_predictor_tokens,
            "aux_active": active,
            "pair_sha256": s.pair_sha256,
        }
        for s, active in zip(shuffled, dropout_mask, strict=True)
    ]
    payload = {
        "schema": E15J_SCHEMA,
        "step": st,
        "config_sha256": cfg.config_sha256(),
        "n_records": len(records),
        "active_aux_count": sum(1 for x in dropout_mask if x),
        "arms": {
            "A": {"label": ARM_DESCRIPTIONS["A"], "lam": 0.0, "k_predictor_tokens": 0, "items": arm_a_items},
            "J": {
                "label": ARM_DESCRIPTIONS["J"],
                "lam": cfg.jepa_spec.lam,
                "k_predictor_tokens": cfg.k_predictor_tokens,
                "items": arm_j_items,
            },
            "S": {
                "label": ARM_DESCRIPTIONS["S"],
                "lam": cfg.jepa_spec.lam,
                "k_predictor_tokens": cfg.k_predictor_tokens,
                "items": arm_s_items,
            },
        },
    }
    payload["schedule_sha256"] = compute_payload_sha256(payload)
    return payload


def _validate_embeddings_2d(arr: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr).__name__}")
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 2 or out.shape[0] < 2 or out.shape[1] < 1:
        raise ValueError(f"{name} must be 2D with shape (N >= 2, D >= 1), got {out.shape}")
    if not bool(np.isfinite(out).all()):
        raise ValueError(f"{name} contains non-finite values")
    return out


def _unit_rows(mat: np.ndarray, name: str) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    if bool((norms <= 1e-12).any()):
        raise ValueError(f"{name} contains zero-norm row vectors")
    return mat / norms


def _cosine_and_infonce(
    a_unit: np.ndarray,
    b_unit: np.ndarray,
    *,
    temperature: float,
) -> tuple[float, float]:
    pair_cos = np.sum(a_unit * b_unit, axis=1)
    one_minus_cos = float(np.mean(1.0 - pair_cos))
    logits = (a_unit @ b_unit.T) / temperature
    logits_shift = logits - np.max(logits, axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(logits_shift), axis=1))
    diag_logits = np.diag(logits_shift)
    infonce_loss = float(np.mean(log_sum_exp - diag_logits))
    return one_minus_cos, infonce_loss


def compute_paired_view_alignment_diagnostics(
    view_a_embeddings: np.ndarray,
    view_b_embeddings: np.ndarray,
    *,
    spec: JepaSpec | None = None,
    step: int = 0,
    k_predictor_tokens: int = DEFAULT_PREDICTOR_TOKENS_K,
    infonce: bool = False,
    temperature: float = DEFAULT_INFONCE_TEMPERATURE,
) -> dict[str, Any]:
    """Compute CPU NumPy JEPA alignment, hash dropout, InfoNCE, and Arm S derangement contrast."""
    sp = spec if spec is not None else JepaSpec(mode="pairs", lam=1.0, stopgrad=True)
    if not isinstance(sp, JepaSpec):
        raise TypeError("spec must be a JepaSpec")
    temp = _require_finite_float(temperature, "temperature", min_value=0.0, strict_min=True)
    a = _validate_embeddings_2d(view_a_embeddings, "view_a_embeddings")
    b = _validate_embeddings_2d(view_b_embeddings, "view_b_embeddings")
    if a.shape != b.shape:
        raise ValueError(f"view_a_embeddings shape {a.shape} != view_b_embeddings shape {b.shape}")

    n = int(a.shape[0])
    mask = compute_step_dropout_mask(sp, step, n, k_predictor_tokens=k_predictor_tokens)
    active_idx = [i for i, keep in enumerate(mask) if keep]

    if sp.center:
        mu = np.mean(np.concatenate([a, b], axis=0), axis=0, keepdims=True)
        a_eval = a - mu
        b_eval = b - mu
    else:
        a_eval = a
        b_eval = b

    a_unit = _unit_rows(a_eval, "view_a_embeddings")
    b_unit = _unit_rows(b_eval, "view_b_embeddings")

    perm = _derange_indices(n, seed=sp.seed, step=step, k=k_predictor_tokens)
    b_shuf_unit = b_unit[perm]

    matched_1m_cos, matched_infonce = _cosine_and_infonce(a_unit, b_unit, temperature=temp)
    shuf_1m_cos, shuf_infonce = _cosine_and_infonce(a_unit, b_shuf_unit, temperature=temp)

    if active_idx and sp.active:
        idx_arr = np.asarray(active_idx, dtype=np.int64)
        if infonce:
            logits = (a_unit[idx_arr] @ b_unit.T) / temp
            logits_shift = logits - np.max(logits, axis=1, keepdims=True)
            row_losses = np.log(np.sum(np.exp(logits_shift), axis=1)) - logits_shift[
                np.arange(len(active_idx)), idx_arr
            ]
            raw_aux = float(np.mean(row_losses))
        else:
            raw_aux = float(np.mean(1.0 - np.sum(a_unit[idx_arr] * b_unit[idx_arr], axis=1)))
        weighted_aux = float(sp.lam * raw_aux)
    else:
        raw_aux = 0.0
        weighted_aux = 0.0

    return {
        "n_pairs": n,
        "active_pairs": len(active_idx),
        "dropped_pairs": n - len(active_idx),
        "stopgrad": sp.stopgrad,
        "centered": sp.center,
        "infonce": bool(infonce),
        "matched_one_minus_cos": matched_1m_cos,
        "shuffled_one_minus_cos": shuf_1m_cos,
        "alignment_margin": float(shuf_1m_cos - matched_1m_cos),
        "matched_infonce_loss": matched_infonce,
        "shuffled_infonce_loss": shuf_infonce,
        "raw_aux_loss": raw_aux,
        "weighted_aux_loss": weighted_aux,
        "collapse_detected": bool(matched_1m_cos < sp.collapse_floor),
    }


# ---------------------------------------------------------------------------
# 4. Exact Paired McNemar & >= +3.0 pts Promotion Gate (E15-J & P2)
# ---------------------------------------------------------------------------


def compute_exact_mcnemar_p(candidate_only_right: int, baseline_only_right: int) -> float:
    """Two-sided exact McNemar p-value reusing ``hivemind.verification.bee_energy.exact_mcnemar_p``."""
    b = _require_int(candidate_only_right, "candidate_only_right", min_value=0)
    c = _require_int(baseline_only_right, "baseline_only_right", min_value=0)
    n = b + c
    if n < 1024:
        return exact_mcnemar_p(b, c)
    k = min(b, c)
    log2 = math.log(2.0)
    log_n_fact = math.lgamma(n + 1)
    tail = sum(math.exp(log_n_fact - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * log2) for i in range(k + 1))
    return min(1.0, 2.0 * tail)


@dataclass(frozen=True)
class PairedComparisonStats:
    """Paired contingency counts, point gain, and exact two-sided McNemar p-value."""

    arm_candidate: str
    arm_baseline: str
    n_items: int
    baseline_passed: int
    candidate_passed: int
    baseline_pass_rate_pct: float
    candidate_pass_rate_pct: float
    delta_pts: float
    candidate_only_right: int
    baseline_only_right: int
    both_right: int
    both_wrong: int
    discordant_count: int
    mcnemar_p: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_paired_arms(
    baseline_outcomes: Sequence[bool],
    candidate_outcomes: Sequence[bool],
    *,
    arm_baseline: str = "A",
    arm_candidate: str = "J",
) -> PairedComparisonStats:
    """Compare two equal-length boolean pass vectors on identical aligned items."""
    base = _require_bool_sequence(baseline_outcomes, f"arm_{arm_baseline}_outcomes")
    cand = _require_bool_sequence(candidate_outcomes, f"arm_{arm_candidate}_outcomes")
    if len(base) != len(cand):
        raise ValueError(f"Arm lengths must match: {len(base)} ({arm_baseline}) != {len(cand)} ({arm_candidate})")

    n = len(base)
    cand_only = sum(1 for b_ok, c_ok in zip(base, cand, strict=True) if c_ok and not b_ok)
    base_only = sum(1 for b_ok, c_ok in zip(base, cand, strict=True) if b_ok and not c_ok)
    both_right = sum(1 for b_ok, c_ok in zip(base, cand, strict=True) if b_ok and c_ok)
    both_wrong = n - cand_only - base_only - both_right

    base_sum = sum(1 for x in base if x)
    cand_sum = sum(1 for x in cand if x)
    base_pct = (100.0 * base_sum) / n
    cand_pct = (100.0 * cand_sum) / n
    delta_pts = (100.0 * (cand_sum - base_sum)) / n

    return PairedComparisonStats(
        arm_candidate=arm_candidate,
        arm_baseline=arm_baseline,
        n_items=n,
        baseline_passed=base_sum,
        candidate_passed=cand_sum,
        baseline_pass_rate_pct=base_pct,
        candidate_pass_rate_pct=cand_pct,
        delta_pts=delta_pts,
        candidate_only_right=cand_only,
        baseline_only_right=base_only,
        both_right=both_right,
        both_wrong=both_wrong,
        discordant_count=cand_only + base_only,
        mcnemar_p=compute_exact_mcnemar_p(cand_only, base_only),
    )


def evaluate_e15j_p2_promotion_gate(
    item_ids: Sequence[str],
    arm_a_passed: Sequence[bool],
    arm_j_passed: Sequence[bool],
    arm_s_passed: Sequence[bool],
    *,
    arm_a_general_nll: float | None = None,
    arm_j_general_nll: float | None = None,
    arm_s_general_nll: float | None = None,
    arm_j_one_minus_cos: float | None = None,
    split_integrity: SplitIntegrityReport | None = None,
    config: PairedViewHarnessConfig | None = None,
    domain: str = "python_code",
    experiment_id: str = "E15-J",
) -> dict[str, Any]:
    """Evaluate Paper §15.4 J2 promotion gate across Arm A (SFT), Arm J (JEPA), and Arm S (Control)."""
    cfg = config if config is not None else PairedViewHarnessConfig()
    if domain not in VALID_DOMAINS:
        raise ValueError(f"domain must be one of {VALID_DOMAINS}, got {domain!r}")
    exp_id = _require_non_empty_str(experiment_id, "experiment_id")
    if not isinstance(item_ids, Sequence) or isinstance(item_ids, (str, bytes)) or not item_ids:
        raise ValueError("item_ids must be a non-empty sequence of strings")

    clean_ids = [_require_non_empty_str(x, f"item_ids[{idx}]") for idx, x in enumerate(item_ids)]
    if len(set(clean_ids)) != len(clean_ids):
        raise ValueError("item_ids must be unique")

    a_ok = _require_bool_sequence(arm_a_passed, "arm_a_passed")
    j_ok = _require_bool_sequence(arm_j_passed, "arm_j_passed")
    s_ok = _require_bool_sequence(arm_s_passed, "arm_s_passed")
    n = len(clean_ids)
    if not (len(a_ok) == len(j_ok) == len(s_ok) == n):
        raise ValueError(
            f"All arms and item_ids must have equal length (ids={n}, A={len(a_ok)}, J={len(j_ok)}, S={len(s_ok)})"
        )

    if (arm_a_general_nll is None) != (arm_j_general_nll is None):
        raise ValueError("arm_a_general_nll and arm_j_general_nll must both be provided or both omitted")
    gen_nll_delta: float | None = None
    if arm_a_general_nll is not None and arm_j_general_nll is not None:
        a_nll = _require_finite_float(arm_a_general_nll, "arm_a_general_nll", min_value=0.0)
        j_nll = _require_finite_float(arm_j_general_nll, "arm_j_general_nll", min_value=0.0)
        gen_nll_delta = j_nll - a_nll
    s_nll_val = (
        _require_finite_float(arm_s_general_nll, "arm_s_general_nll", min_value=0.0)
        if arm_s_general_nll is not None
        else None
    )
    j_1m_cos = (
        _require_finite_float(arm_j_one_minus_cos, "arm_j_one_minus_cos", min_value=0.0, max_value=2.0)
        if arm_j_one_minus_cos is not None
        else None
    )

    if split_integrity is not None and not isinstance(split_integrity, SplitIntegrityReport):
        raise TypeError("split_integrity must be a SplitIntegrityReport or None")

    j_vs_a = compare_paired_arms(a_ok, j_ok, arm_baseline="A", arm_candidate="J")
    s_vs_a = compare_paired_arms(a_ok, s_ok, arm_baseline="A", arm_candidate="S")
    j_vs_s = compare_paired_arms(s_ok, j_ok, arm_baseline="S", arm_candidate="J")

    split_disjoint_ok = True if split_integrity is None else split_integrity.is_disjoint
    gain_threshold_met = (100.0 * (j_vs_a.candidate_passed - j_vs_a.baseline_passed)) >= (
        cfg.min_gain_pts * n - 1e-12
    )
    mcnemar_j_vs_a_significant = (
        j_vs_a.candidate_only_right > j_vs_a.baseline_only_right and j_vs_a.mcnemar_p < cfg.mcnemar_alpha
    )
    shuffled_control_gained = (
        s_vs_a.candidate_only_right > s_vs_a.baseline_only_right and s_vs_a.mcnemar_p < cfg.mcnemar_alpha
    )
    j_beats_shuffled_control = (
        j_vs_s.candidate_only_right > j_vs_s.baseline_only_right and j_vs_s.mcnemar_p < cfg.mcnemar_alpha
    )
    general_guard_ok = (gen_nll_delta is None) or (gen_nll_delta <= cfg.max_general_nll_delta + 1e-12)
    collapse_guard_ok = (j_1m_cos is None) or (j_1m_cos >= cfg.jepa_spec.collapse_floor)

    rejection_reasons: list[str] = []
    if not split_disjoint_ok:
        rejection_reasons.append("split_leakage_detected")
    if shuffled_control_gained:
        rejection_reasons.append("shuffled_control_gained_significantly")
    if not general_guard_ok:
        rejection_reasons.append("general_nll_guard_exceeded")
    if not collapse_guard_ok:
        rejection_reasons.append("jepa_cosine_collapse_detected")
    if not gain_threshold_met:
        rejection_reasons.append("gain_below_min_pts_threshold")
    if not mcnemar_j_vs_a_significant:
        rejection_reasons.append("mcnemar_j_vs_a_not_significant")
    if cfg.require_j_beats_s and not j_beats_shuffled_control:
        rejection_reasons.append("jepa_did_not_beat_shuffled_control")

    if not split_disjoint_ok:
        verdict = "INVALID_SPLIT_LEAKAGE"
    elif shuffled_control_gained:
        verdict = "INVALID_SHUFFLED_CONTROL"
    elif not general_guard_ok or not collapse_guard_ok:
        verdict = "REJECTED_GUARD_OR_COLLAPSE"
    elif (
        j_vs_a.delta_pts < 0.0
        and j_vs_a.baseline_only_right > j_vs_a.candidate_only_right
        and j_vs_a.mcnemar_p < cfg.mcnemar_alpha
    ):
        verdict = "REFUTED_REGRESSION"
    elif not rejection_reasons:
        verdict = "PROMOTED"
    else:
        verdict = "INCONCLUSIVE"

    promoted = verdict == "PROMOTED"
    receipt: dict[str, Any] = {
        "schema": E15J_SCHEMA,
        "object": E15J_RECEIPT_OBJECT,
        "experiment_id": exp_id,
        "domain": domain,
        "n_items": n,
        "item_ids_sha256": compute_payload_sha256(clean_ids),
        "config": cfg.to_dict(),
        "config_sha256": cfg.config_sha256(),
        "j_vs_a": j_vs_a.to_dict(),
        "s_vs_a": s_vs_a.to_dict(),
        "j_vs_s": j_vs_s.to_dict(),
        "arm_a_general_nll": arm_a_general_nll,
        "arm_j_general_nll": arm_j_general_nll,
        "arm_s_general_nll": s_nll_val,
        "general_nll_delta": gen_nll_delta,
        "arm_j_one_minus_cos": j_1m_cos,
        "split_integrity_sha256": split_integrity.report_sha256 if split_integrity is not None else None,
        "split_disjoint_ok": split_disjoint_ok,
        "gain_threshold_met": gain_threshold_met,
        "mcnemar_j_vs_a_significant": mcnemar_j_vs_a_significant,
        "shuffled_control_gained": shuffled_control_gained,
        "j_beats_shuffled_control": j_beats_shuffled_control,
        "general_guard_ok": general_guard_ok,
        "collapse_guard_ok": collapse_guard_ok,
        "verdict": verdict,
        "promoted": promoted,
        "p2_gate_open": promoted,
        "rejection_reasons": rejection_reasons,
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return receipt


def verify_e15j_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Verify canonical SHA-256 provenance and E15-J / P2 gate invariants on a receipt."""
    if not isinstance(receipt, Mapping):
        return False
    if receipt.get("schema") != E15J_SCHEMA or receipt.get("object") != E15J_RECEIPT_OBJECT:
        return False
    for key in ("receipt_sha256", "config_sha256", "item_ids_sha256"):
        val = receipt.get(key)
        if not isinstance(val, str) or not _SHA256_HEX_RE.match(val):
            return False

    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        if compute_payload_sha256(core) != receipt["receipt_sha256"]:
            return False
    except (TypeError, ValueError):
        return False

    promoted = receipt.get("promoted")
    p2_gate_open = receipt.get("p2_gate_open")
    verdict = receipt.get("verdict")
    if not isinstance(promoted, bool) or not isinstance(p2_gate_open, bool):
        return False
    if promoted != p2_gate_open or promoted != (verdict == "PROMOTED"):
        return False
    if not promoted:
        return True

    cfg = receipt.get("config")
    j_vs_a = receipt.get("j_vs_a")
    if not isinstance(cfg, Mapping) or not isinstance(j_vs_a, Mapping):
        return False
    min_pts = float(cfg.get("min_gain_pts", DEFAULT_MIN_GAIN_PTS))
    alpha = float(cfg.get("mcnemar_alpha", DEFAULT_MCNEMAR_ALPHA))
    return (
        float(j_vs_a.get("delta_pts", -1.0)) >= min_pts - 1e-9
        and float(j_vs_a.get("mcnemar_p", 1.0)) < alpha
        and int(j_vs_a.get("candidate_only_right", 0)) > int(j_vs_a.get("baseline_only_right", 0))
        and receipt.get("shuffled_control_gained") is False
        and receipt.get("general_guard_ok") is True
        and receipt.get("collapse_guard_ok") is True
        and receipt.get("split_disjoint_ok") is True
        and receipt.get("rejection_reasons") == []
    )


__all__ = [
    "ARM_DESCRIPTIONS",
    "DEFAULT_COLLAPSE_FLOOR",
    "DEFAULT_INFONCE_TEMPERATURE",
    "DEFAULT_MAX_GENERAL_NLL_DELTA",
    "DEFAULT_MBPP_EVAL_N",
    "DEFAULT_MCNEMAR_ALPHA",
    "DEFAULT_MIN_GAIN_PTS",
    "DEFAULT_PREDICTOR_TOKENS_K",
    "E15J_RECEIPT_OBJECT",
    "E15J_SCHEMA",
    "PairedComparisonStats",
    "PairedViewHarnessConfig",
    "PairedViewRecord",
    "ShuffledPairRecord",
    "SplitIntegrityReport",
    "VALID_ARMS",
    "VALID_DOMAINS",
    "VALID_SPLITS",
    "build_lean4_paired_record",
    "build_paired_view_records",
    "build_python_paired_record",
    "build_shuffled_pair_derangement",
    "build_three_arm_schedule",
    "compare_paired_arms",
    "compute_exact_mcnemar_p",
    "compute_paired_view_alignment_diagnostics",
    "compute_payload_sha256",
    "compute_step_dropout_mask",
    "decontaminate_paired_records",
    "dropout_draw",
    "evaluate_e15j_p2_promotion_gate",
    "extract_lean4_paired_view",
    "extract_python_paired_view",
    "normalize_view_text",
    "sha256_text",
    "should_apply_jepa_loss",
    "verify_e15j_receipt_provenance",
    "verify_zero_overlap_splits",
]
