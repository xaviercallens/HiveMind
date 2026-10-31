# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HM-S35-05: Unified XDEV Portfolio Registry & 4-Layer Cross-Project Execution Pipeline.

Distinguishes:
- `original_core_ip`: Xavier Callens' original research & systems frameworks (`AMCP`, `HiveMind`,
  `AutoevolveAI/ANSE`, `GWAYA-GwenLayaOpenModel`, `OptimalEnergyAI`, `SocrateAI-Scientific-Elenchus`,
  `elenchus-solvers-mcp`, `rusty-SUNDIALS`, `SocrateAI-Lean-Lib`, `OpenAI-NSE-Epistemic-Audit`,
  `SocrateAI-Scientific-MechanicaFluidorum`, `SocrateAI-Scientific-QuantumFluids`, `runux-ai-runtime`,
  `rust-linux-mini-kernel`).
- `community_fork_bridge`: GitHub forks prefixed with `x`/`X` (`XSwaT`, `xLayerStoRm`, `xensorSharp`,
  `xlaya`, `XHillock`, `xdatabricks-agent-skills`, `xskills-for-copilot-studio`, `xanything-llm`,
  `Xoptics-lab`, `xProCompta`, `xfibo`) integrated as interoperable bridges around the core stack.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Sequence

from hivemind.swarm.s35_01_optimal_energy_amcp_xswat_codec import (
    benchmark_a2_rope_plane_quantizer,
    pack_amcp_xswat_latent_kv_envelope,
)
from hivemind.swarm.s35_02_gwaya_laya_failclosed_gate import run_gwaya_a3_sample_and_repair
from hivemind.swarm.s35_03_elenchus_leanlib_sundials_r64 import (
    audit_lean4_source_elenchus,
    verify_robertson_sundials_tier_x,
)
from hivemind.swarm.s35_04_xlayerstorm_xensorsharp_bridge import (
    compute_xlayerstorm_turboquant_schedule,
)


@dataclass(frozen=True)
class PortfolioProjectNode:
    """Metadata and integration role for a project in the unified ecosystem."""

    name: str
    origin_type: str  # "original_core_ip" | "community_fork_bridge"
    layer: str  # "L1_Orchestration_Latent" | "L2_Compute_Lattice" | "L3_FailClosed_Gate" | "L4_Formal_Physics_Audit" | "B2B_Vertical"
    local_xdev_dir: str
    github_repo: str
    role_in_unified_stack: str
    locally_present: bool


@dataclass(frozen=True)
class UnifiedEcosystemExecutionReceipt:
    """End-to-end execution receipt across all 4 layers of the unified stack."""

    original_core_count: int
    fork_bridge_count: int
    local_xdev_verified_count: int
    layer1_kv_compression_ratio: float
    layer1_a2_rope_mse_gain_pct: float
    layer2_peak_citizen_vram_mib: float
    layer3_gwaya_a3_verified: bool
    layer3_final_anse_energy: float
    layer4_lean_kernel_clean: bool
    layer4_sundials_tier_x_accepted: bool
    all_layers_passed: bool
    merkle_root_sha256: str


def classify_repo_origin(repo_name: str) -> str:
    """Classify a repository name: `x`/`X`-prefixed forks vs original core IP."""
    if len(repo_name) >= 2 and repo_name[0] in ("x", "X") and repo_name not in ("xaviercallens.github.io",):
        return "community_fork_bridge"
    return "original_core_ip"


def _default_xdev_root() -> Path:
    """Resolve XDEV_ROOT from environment or relative to repository root without hardcoding user paths."""
    import os

    env_val = os.environ.get("XDEV_ROOT", "").strip()
    if env_val:
        return Path(env_val)
    parents = Path(__file__).resolve().parents
    return parents[7] if len(parents) > 7 else Path.cwd()


def build_xdev_portfolio_catalog(xdev_root: Path | None = None) -> tuple[PortfolioProjectNode, ...]:
    """Build the complete catalog of original core IP projects and `x*` fork bridges."""
    if xdev_root is None:
        xdev_root = _default_xdev_root()
    specs = (
        (
            "AMCP",
            "L1_Orchestration_Latent",
            "AutoevolveAI",
            "xaviercallens/amcp-v1.5-opensource",
            "Distributed event-driven Agent Mesh Communication Protocol & P2P bus",
        ),
        (
            "HiveMind",
            "L2_Compute_Lattice",
            "AutoevolveAI/sub_projects/hivemind/HiveMind",
            "xaviercallens/HiveMind",
            "Decentralized <1.5 GB consumer GPU swarm, OpenRouter & Kelly HMT ledger",
        ),
        (
            "AutoevolveAI",
            "L3_FailClosed_Gate",
            "AutoevolveAI",
            "xaviercallens/AutoevolveAI",
            "ANSE physical Energy Function (E=10^6 fail-closed) & Phase 1-3 Evolution Lab",
        ),
        (
            "GWAYA-GwenLayaOpenModel",
            "L3_FailClosed_Gate",
            "GWAYA-GwenLayaOpenModel",
            "xaviercallens/GWAYA-GwenLayaOpenModel",
            "Zero-Stub AST gate, Laya calibrated selective abstention & Arm A3 repair loop",
        ),
        (
            "OptimalEnergyAI",
            "L2_Compute_Lattice",
            "OptimalEnergyAI",
            "xaviercallens/OptimalEnergyAI",
            "Universal optimality of A2 triangular & E8 Gosset lattices for RoPE & KV quantization",
        ),
        (
            "SocrateAI-Scientific-Elenchus",
            "L4_Formal_Physics_Audit",
            "SocrateAI-Scientific-Elenchus",
            "xaviercallens/SocrateAI-Scientific-Elenchus",
            "3-Layer Socratic cross-examination harness (Lean 4 + ODE + physical regime)",
        ),
        (
            "elenchus-solvers-mcp",
            "L4_Formal_Physics_Audit",
            "elenchus-solvers-mcp",
            "xaviercallens/elenchus-solvers-mcp",
            "Tier-X stdio JSON-RPC MCP server with R-64 argument provenance & stub refusal",
        ),
        (
            "rusty-SUNDIALS",
            "L4_Formal_Physics_Audit",
            "rusty-SUNDIALS",
            "xaviercallens/rusty-SUNDIALS",
            "Pure-Rust formally verified CVODE stiff ODE solver (44K LOC C -> safe Rust)",
        ),
        (
            "SocrateAI-Lean-Lib",
            "L4_Formal_Physics_Audit",
            "SocrateAI-Lean-Lib",
            "xaviercallens/SocrateAI-Lean-Lib",
            "14-domain, 40+ module Lean 4 scientific foundation library (zero sorry, zero Mathlib)",
        ),
        (
            "OpenAI-NSE-Epistemic-Audit",
            "L4_Formal_Physics_Audit",
            "OpenAI-NSE-Epistemic-Audit",
            "xaviercallens/OpenAI-NSE-Epistemic-Audit",
            "Physical Knudsen-number & thermodynamic audit of fluid PDE blow-up proofs",
        ),
        (
            "runux-ai-runtime",
            "L2_Compute_Lattice",
            "runux-ai-runtime",
            "xaviercallens/runux-ai-runtime",
            "Memory-safe Rust runtime & TurboQuant KV-cache compression for RISC-V & TPUs",
        ),
        (
            "XSwaT",
            "L1_Orchestration_Latent",
            "",
            "xaviercallens/XSwaT",
            "Fork bridge: Prefill-free cross-agent latent KV-cache state transfer over AMCP",
        ),
        (
            "xLayerStoRm",
            "L2_Compute_Lattice",
            "",
            "xaviercallens/xLayerStoRm",
            "Fork bridge: Double-buffered PCIe MoE expert streaming under <=1.5 GB GPU VRAM",
        ),
        (
            "xensorSharp",
            "L2_Compute_Lattice",
            "",
            "xaviercallens/xensorSharp",
            "Fork bridge: Native .NET/Windows/macOS/iOS GGUF citizen worker without WSL2",
        ),
        (
            "xlaya",
            "L1_Orchestration_Latent",
            "",
            "xaviercallens/xlaya",
            "Fork bridge: Single-forward-pass non-autoregressive System-1 decision router",
        ),
        (
            "XHillock",
            "L1_Orchestration_Latent",
            "",
            "xaviercallens/XHillock",
            "Fork bridge: Local 0-VRAM Hyperdimensional (HDC/VSA) & Hebbian agent memory",
        ),
        (
            "xdatabricks-agent-skills",
            "B2B_Vertical",
            "",
            "xaviercallens/xdatabricks-agent-skills",
            "Fork bridge: B2B Databricks Spark/Delta agent skills for Claude Code & Cursor",
        ),
        (
            "xskills-for-copilot-studio",
            "B2B_Vertical",
            "",
            "xaviercallens/xskills-for-copilot-studio",
            "Fork bridge: B2B Microsoft Copilot Studio YAML validation skills",
        ),
        (
            "xProCompta",
            "B2B_Vertical",
            "",
            "xaviercallens/xProCompta",
            "Fork bridge: Sovereign French SME accounting document workflow on local HiveMind node",
        ),
    )

    nodes: list[PortfolioProjectNode] = []
    for name, layer, rel_dir, gh_repo, role in specs:
        origin = classify_repo_origin(name)
        is_present = bool(rel_dir) and (xdev_root / rel_dir).exists()
        nodes.append(
            PortfolioProjectNode(
                name=name,
                origin_type=origin,
                layer=layer,
                local_xdev_dir=str(xdev_root / rel_dir) if rel_dir else "",
                github_repo=gh_repo,
                role_in_unified_stack=role,
                locally_present=is_present,
            )
        )
    return tuple(nodes)


def run_unified_4layer_ecosystem_pipeline(
    xdev_root: Path | None = None,
) -> UnifiedEcosystemExecutionReceipt:
    """Execute the end-to-end 4-Layer integration pipeline across all core & fork-bridge modules."""
    if xdev_root is None:
        xdev_root = _default_xdev_root()
    catalog = build_xdev_portfolio_catalog(xdev_root=xdev_root)
    orig_cnt = sum(1 for n in catalog if n.origin_type == "original_core_ip")
    fork_cnt = sum(1 for n in catalog if n.origin_type == "community_fork_bridge")
    local_cnt = sum(1 for n in catalog if n.locally_present)

    # Layer 1: OptimalEnergyAI A2-RoPE + AMCP x XSwaT E8 Latent KV-Cache Envelope
    rope_pairs = [
        ((i % 19 - 9) * 0.037, ((i // 19) % 19 - 9) * 0.037)
        for i in range(361)
    ]
    a2_report = benchmark_a2_rope_plane_quantizer(rope_pairs, scale=0.12)
    kv_blocks = [
        [math.sin(0.31 * (k * 8 + d)) * 0.65 for d in range(8)]
        for k in range(16)
    ]
    amcp_env = pack_amcp_xswat_latent_kv_envelope(
        sender_agent_id="amcp://bee-coder-proposer",
        receiver_agent_id="amcp://bee-lean-verifier",
        kv_vectors_8d=kv_blocks,
        scale=0.25,
        bits_per_dim=2.5,
    )

    # Layer 2: xLayerStoRm PCIe MoE Streaming + RunuX TurboQuant Schedule under <= 1,536 MiB
    pcie_plan = compute_xlayerstorm_turboquant_schedule()

    # Layer 3: GWAYA Fail-Closed Anti-Stub Gate + Arm A3 Sample-and-Repair Loop
    candidates: Sequence[str] = (
        "def compute_conservation_residual(m0: float, m1: float) -> float:\n    pass\n",
        "def compute_conservation_residual(m0: float, m1: float) -> float:\n    return abs(m1 - m0)\n",
    )
    a3_report = run_gwaya_a3_sample_and_repair(
        candidates=candidates,
        public_test_fn=lambda code: "abs(m1 - m0)" in code,
        hidden_test_fn=lambda code: "pass" not in code and "abs(" in code,
    )

    # Layer 4: Elenchus Comment-Stripped Lean 4 Audit + rusty-SUNDIALS Tier-X R-64 Provenance
    sample_lean = (
        "/- **Kernel Certified:** 0 sorry, 0 admit. -/\n"
        "-- No sorry allowed in SocrateAI-Lean-Lib\n"
        "theorem k3_euler_characteristic : 24 = 24 := by rfl\n"
    )
    lean_audit = audit_lean4_source_elenchus(sample_lean)
    args = {"problem": "robertson", "rtol": 1e-9, "atol": 1e-12, "max_order": 5}
    sundials_receipt = verify_robertson_sundials_tier_x(sent_arguments=args, received_arguments=args)

    all_passed = (
        a2_report.beats_z2_in_mse
        and amcp_env.compression_ratio >= 6.0
        and pcie_plan.within_citizen_vram_cap
        and a3_report.final_a3_passed
        and lean_audit.is_kernel_clean
        and sundials_receipt.accepted_by_elenchus_bridge
    )

    merkle_payload = (
        f"{amcp_env.payload_sha256}|{pcie_plan.peak_active_vram_mib}|"
        f"{a3_report.final_energy}|{lean_audit.classification}|{sundials_receipt.provenance_sha256}"
    )
    merkle_root = hashlib.sha256(merkle_payload.encode("utf-8")).hexdigest()

    return UnifiedEcosystemExecutionReceipt(
        original_core_count=orig_cnt,
        fork_bridge_count=fork_cnt,
        local_xdev_verified_count=local_cnt,
        layer1_kv_compression_ratio=amcp_env.compression_ratio,
        layer1_a2_rope_mse_gain_pct=a2_report.empirical_mse_reduction_pct,
        layer2_peak_citizen_vram_mib=pcie_plan.peak_active_vram_mib,
        layer3_gwaya_a3_verified=a3_report.final_a3_passed,
        layer3_final_anse_energy=a3_report.final_energy,
        layer4_lean_kernel_clean=lean_audit.is_kernel_clean,
        layer4_sundials_tier_x_accepted=sundials_receipt.accepted_by_elenchus_bridge,
        all_layers_passed=all_passed,
        merkle_root_sha256=merkle_root,
    )
