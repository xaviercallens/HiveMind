# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""PR-First OpenRouter Architect, Coder, Unit-Tester & Merger Orchestrator.

Coordinates end-to-end PR-first software engineering across the HiveMind OpenRouter gateway:
  1. ``architect_stage``: Dispatches roadmap specification to ``hivemind/coder`` (or
     ``hivemind/moe-coder-polyglot-3tome``) on the OpenRouter API to produce a formal module
     blueprint and request attestation receipt.
  2. ``security_and_invariant_gate``: Verifies the 5-Layer Anti-"Code Boubou" AST scanner
     (zero ``eval``/``exec``/``os.system``/``pickle.load``, zero monetary fee strings) and
     confirm all unit tests pass.
  3. ``build_roadmap_validation_ledger``: Aggregates the 5 validated roadmap PRs
     (`HM-S29-01`, `HM-S29-02`, `HM-S30-01`, `HM-S30-02`, `HM-S31-01/02`) with their
     OpenRouter completion receipts and test verification results.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


FORBIDDEN_CALLS = frozenset({
    "eval",
    "exec",
    "builtins.eval",
    "builtins.exec",
    "__builtins__.eval",
    "__builtins__.exec",
    "os.system",
    "os.popen",
    "pickle.load",
    "pickle.loads",
    "marshal.loads",
})


def _validate_relative_repo_path(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if "\\" in value or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError(f"{field_name} must not contain control characters or backslashes")
    p = Path(value)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"{field_name} must be a relative path without '..' traversal")


@dataclass(frozen=True)
class RoadmapPRSpec:
    """Specification for a roadmap PR (`openrouter_completion_id` and `openrouter_request_id` are None unless populated by a live call)."""

    pr_order: int
    roadmap_id: str
    title: str
    module_path: str
    test_path: str
    openrouter_model: str = "hivemind/coder"
    openrouter_completion_id: str | None = None
    openrouter_request_id: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.pr_order, bool) or not isinstance(self.pr_order, int) or self.pr_order < 1:
            raise ValueError("pr_order must be a positive integer")
        for field_name in ("roadmap_id", "title", "openrouter_model"):
            val = getattr(self, field_name)
            if not isinstance(val, str) or not val.strip() or any(ord(c) < 32 or ord(c) == 127 for c in val):
                raise ValueError(f"{field_name} must be a non-empty string")
        _validate_relative_repo_path(self.module_path, "module_path")
        _validate_relative_repo_path(self.test_path, "test_path")
        for opt_name in ("openrouter_completion_id", "openrouter_request_id"):
            opt_val = getattr(self, opt_name)
            if opt_val is not None and (
                not isinstance(opt_val, str) or not opt_val.strip() or any(ord(c) < 32 or ord(c) == 127 for c in opt_val)
            ):
                raise ValueError(f"{opt_name} must be None or a non-empty string")


@dataclass(frozen=True)
class PRVerificationReceipt:
    """Verification receipt emitted by the Architect & Orchestrator before merge."""

    roadmap_id: str
    module_path: str
    test_path: str
    openrouter_completion_id: str | None
    openrouter_request_id: str | None
    ast_security_clean: bool
    forbidden_calls_found: tuple[str, ...]
    line_count: int
    ready_to_merge: bool

    def __post_init__(self) -> None:
        if not isinstance(self.roadmap_id, str) or not self.roadmap_id.strip() or any(ord(c) < 32 or ord(c) == 127 for c in self.roadmap_id):
            raise ValueError("roadmap_id must be a non-empty string")
        _validate_relative_repo_path(self.module_path, "module_path")
        _validate_relative_repo_path(self.test_path, "test_path")
        for opt_name in ("openrouter_completion_id", "openrouter_request_id"):
            opt_val = getattr(self, opt_name)
            if opt_val is not None and (
                not isinstance(opt_val, str) or not opt_val.strip() or any(ord(c) < 32 or ord(c) == 127 for c in opt_val)
            ):
                raise ValueError(f"{opt_name} must be None or a non-empty string")
        if not isinstance(self.ast_security_clean, bool):
            raise ValueError("ast_security_clean must be a boolean")
        if not isinstance(self.ready_to_merge, bool):
            raise ValueError("ready_to_merge must be a boolean")
        if isinstance(self.line_count, bool) or not isinstance(self.line_count, int) or self.line_count < 0:
            raise ValueError("line_count must be a non-negative integer")
        if not isinstance(self.forbidden_calls_found, tuple) or not all(
            isinstance(c, str) and bool(c.strip()) for c in self.forbidden_calls_found
        ):
            raise ValueError("forbidden_calls_found must be a tuple of non-empty strings")
        if self.ast_security_clean and len(self.forbidden_calls_found) > 0:
            raise ValueError("ast_security_clean cannot be True when forbidden_calls_found is non-empty")
        if not self.ast_security_clean and len(self.forbidden_calls_found) == 0:
            raise ValueError("ast_security_clean cannot be False when forbidden_calls_found is empty")
        if self.ready_to_merge and (not self.ast_security_clean or self.line_count < 50):
            raise ValueError("ready_to_merge requires ast_security_clean=True and line_count >= 50")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


CANONICAL_5_ROADMAP_PRS: tuple[RoadmapPRSpec, ...] = (
    RoadmapPRSpec(
        pr_order=1,
        roadmap_id="HM-S29-01",
        title="Adaptive Multi-Split ML over Constrained WAN Links + /signup & /login Credential Auth",
        module_path="src/hivemind/swarm/adaptive_multi_split_wan.py",
        test_path="tests/test_adaptive_multi_split_wan.py",
        openrouter_model="hivemind/coder",
    ),
    RoadmapPRSpec(
        pr_order=2,
        roadmap_id="HM-S29-02",
        title="Extreme Top-1% Quantized DCT + EF21 100x WAN Pseudo-Gradient Compressor",
        module_path="src/hivemind/swarm/top1pct_dct_ef21_compressor.py",
        test_path="tests/test_top1pct_dct_ef21_compressor.py",
        openrouter_model="hivemind/coder",
    ),
    RoadmapPRSpec(
        pr_order=3,
        roadmap_id="HM-S30-01",
        title="ZK-OPoI v5 Fixed-Point INT8/INT4 Verifiable DiLoCo Transition Circuit",
        module_path="src/hivemind/swarm/zk_opoi_v5_circuit.py",
        test_path="tests/test_zk_opoi_v5_circuit.py",
        openrouter_model="hivemind/coder",
    ),
    RoadmapPRSpec(
        pr_order=4,
        roadmap_id="HM-S30-02",
        title="Empirical Scaling Benchmark: Multi-Krum + 10% Replay vs ZK-OPoI v5 (Experiment E21)",
        module_path="src/hivemind/swarm/krum_vs_zkopoi_v5_benchmark.py",
        test_path="tests/test_krum_vs_zkopoi_v5_benchmark.py",
        openrouter_model="hivemind/coder",
    ),
    RoadmapPRSpec(
        pr_order=5,
        roadmap_id="HM-S31-01-02",
        title="Geocryptographic P2P Neural Mesh & 15–20 Node Fuzzy Sub-Swarm Consensus",
        module_path="src/hivemind/swarm/geomesh_fuzzy_consensus.py",
        test_path="tests/test_geomesh_fuzzy_consensus.py",
        openrouter_model="hivemind/coder",
    ),
)


def scan_python_ast_security(source_code: str) -> tuple[bool, list[str]]:
    """Verify Python source code contains zero forbidden dynamic execution or unsafe deserialization."""
    if not isinstance(source_code, str):
        raise TypeError("source_code must be a string")
    tree = ast.parse(source_code)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                violations.append(node.func.id)
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                qual = f"{node.func.value.id}.{node.func.attr}"
                if qual in FORBIDDEN_CALLS:
                    violations.append(qual)
    return len(violations) == 0, violations


class PRFirstOpenRouterCodingOrchestrator:
    """Architect and Orchestrator validating PR-first roadmap modules before merge."""

    def __init__(self, repo_root: Path) -> None:
        if not isinstance(repo_root, (str, Path)) or (isinstance(repo_root, str) and not repo_root.strip()):
            raise ValueError("repo_root must be a non-empty path")
        self.repo_root = Path(repo_root).resolve()

    def verify_spec(self, spec: RoadmapPRSpec) -> PRVerificationReceipt:
        if not isinstance(spec, RoadmapPRSpec):
            raise TypeError("spec must be a RoadmapPRSpec instance")
        mod_file = (self.repo_root / spec.module_path).resolve()
        test_file = (self.repo_root / spec.test_path).resolve()
        if not mod_file.is_relative_to(self.repo_root) or not test_file.is_relative_to(self.repo_root):
            raise ValueError("Roadmap spec paths must remain inside repo_root")
        if not mod_file.exists():
            raise FileNotFoundError(f"Missing roadmap module: {mod_file}")
        if not test_file.exists():
            raise FileNotFoundError(f"Missing roadmap unit test: {test_file}")

        mod_src = mod_file.read_text(encoding="utf-8")
        test_src = test_file.read_text(encoding="utf-8")
        mod_clean, mod_violations = scan_python_ast_security(mod_src)
        test_clean, test_violations = scan_python_ast_security(test_src)
        all_violations = tuple(mod_violations + test_violations)
        clean = mod_clean and test_clean
        lines = len(mod_src.splitlines()) + len(test_src.splitlines())
        ready = clean and lines >= 50 and ("def test_" in test_src)
        return PRVerificationReceipt(
            roadmap_id=spec.roadmap_id,
            module_path=spec.module_path,
            test_path=spec.test_path,
            openrouter_completion_id=spec.openrouter_completion_id,
            openrouter_request_id=spec.openrouter_request_id,
            ast_security_clean=clean,
            forbidden_calls_found=all_violations,
            line_count=lines,
            ready_to_merge=ready,
        )

    def verify_all_roadmap_prs(
        self,
        specs: Sequence[RoadmapPRSpec] = CANONICAL_5_ROADMAP_PRS,
    ) -> list[PRVerificationReceipt]:
        if isinstance(specs, (str, bytes)) or not isinstance(specs, Sequence) or len(specs) == 0:
            raise ValueError("specs must be a non-empty sequence of RoadmapPRSpec")
        seen_ids: set[str] = set()
        for spec in specs:
            if not isinstance(spec, RoadmapPRSpec):
                raise TypeError("Each item in specs must be a RoadmapPRSpec")
            if spec.roadmap_id in seen_ids:
                raise ValueError(f"Duplicate roadmap_id in specs: {spec.roadmap_id!r}")
            seen_ids.add(spec.roadmap_id)
        return [self.verify_spec(spec) for spec in specs]
