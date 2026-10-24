"""Project HiveMind — Repository, Pull Request, Runtime Swarm, Federated Learning, GPU Hardware & Red-Team Guardrail Security Package.

Provides:
1. Automated 5-Layer Pull Request & Repository Security Scanner (`pr_security_scanner.py`)
2. Phase S-2 Swarm GPU Training & Content-Addressed Data Store Runtime Hardening (`swarm_runtime_hardening.py`)
3. Phase S-3 Federated Learning & Byzantine-Resilient DiLoCo Hardening (`federated_learning_hardening.py`)
4. Phase S-4 Federated GPU Execution & Hardware Safety Circuit-Breaker (`gpu_execution_circuit_breaker.py`)
5. Phase S-5 25-Vector End-to-End Multi-Layer Attack Simulation & Guardrail Verification (`guardrail_attack_simulator.py`)

Exports are resolved lazily (PEP 562) so the lightweight PR security scanner can be imported by the
security gate without installing the full Torch/GPU runtime stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_MODULES: dict[str, tuple[str, ...]] = {
    "hivemind.security._exceptions": (
        "SwarmSecurityViolationError",
    ),
    "hivemind.security.federated_learning_hardening": (
        "DPGradientSanitizerConfig",
        "FederatedDeltaAuditReport",
        "FederatedLearningSecurityGuard",
        "run_federated_hardening_self_test",
    ),
    "hivemind.security.gpu_execution_circuit_breaker": (
        "CircuitBreakerStatus",
        "GPUExecutionCircuitBreaker",
        "GPUHardwareTelemetry",
        "query_live_gpu_hardware_telemetry",
        "run_gpu_circuit_breaker_self_test",
    ),
    "hivemind.security.guardrail_attack_simulator": (
        "SimulatedAttackScenarioResult",
        "SwarmGuardrailAttackSimulator",
        "run_guardrail_attack_simulation",
    ),
    "hivemind.security.owner_order_guardrail": (
        "AUTHORIZED_OWNER_DOMAIN",
        "AUTHORIZED_OWNER_EMAIL",
        "AUTHORIZED_OWNER_GITHUB_LOGIN",
        "AUTHORIZED_OWNER_LOCAL_PART",
        "EmailIdentityVerification",
        "OwnerOrderDecision",
        "OwnerOrderFilterBatchResult",
        "OwnerOrderGuardrail",
        "OwnerOrderRequest",
        "RequestIntentCategory",
        "RequestIntentClassification",
        "classify_request_intent",
        "evaluate_owner_order_request",
        "filter_authorized_owner_orders",
        "normalize_and_verify_owner_email",
        "verify_local_git_owner_identity",
    ),
    "hivemind.security.pr_security_scanner": (
        "PRSecurityAuditResult",
        "PRSecurityScanner",
        "SecurityFinding",
        "SecuritySeverity",
        "run_cli_security_scan",
    ),
    "hivemind.security.swarm_runtime_hardening": (
        "SwarmHardeningVerdict",
        "SwarmTrainingAndStoreGuard",
        "compute_shannon_entropy",
        "run_swarm_hardening_self_test",
    ),
}
_LAZY_EXPORTS: dict[str, str] = {
    symbol: module_name
    for module_name, symbols in _LAZY_MODULES.items()
    for symbol in symbols
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Import a security symbol on first access instead of eagerly loading Torch modules."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
