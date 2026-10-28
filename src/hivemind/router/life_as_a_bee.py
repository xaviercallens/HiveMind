# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Life as a Bee Dashboard & Real End-User Scenario Benchmark (``HM-S19-LIFE-AS-A-BEE``).

Exposes:
- ``BEE_LIFECYCLE_PHASES``: The 24-hour biological & thermodynamic cycle of a HiveMind Bee
  (Pollen Intent Routing, Daytime Nectar Foraging with <1.5s game preemption, AMCP v1.5 / Kafka
  Inter-Bee Waggle Dance, and Nocturnal Royal Jelly BOINC QLoRA-MoE distillation).
- ``LIFE_AS_A_BEE_SCENARIOS``: Real end-to-end ``POST /v1/chat/completions`` benchmark results across
  all 15 live models and MoE bees on ``hivemind-openrouter``, including warm latency (ms), throughput
  (tokens/sec), cold-load duration (s), GPU/CPU VRAM residency split, verbatim response excerpts,
  and actionable usage guidance (*what to use each bee for*, *when not to use it*, and *how to invoke it*).
- ``BEE_DECISION_MATRIX``: Quick task-to-bee selector for developers, researchers, and IDE users.
- ``life_as_a_bee_payload()``: Public JSON payload served at ``GET /v1/hivemind/life-as-a-bee``.
- ``render_life_as_a_bee_section()``: Interactive HTML section embedded in ``/`` and ``/playground``.
- ``render_life_as_a_bee_page(nonce)``: Standalone interactive dashboard served at ``GET /life-as-a-bee``.
"""

from __future__ import annotations

import html
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BeeLifecyclePhase:
    """One stage in the 24-hour life cycle of a HiveMind MoE or specialist bee."""

    phase_id: str
    step_number: int
    title_fr: str
    title_en: str
    time_window: str
    key_metric: str
    summary_fr: str
    summary_en: str
    protocol_endpoints: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["protocol_endpoints"] = list(self.protocol_endpoints)
        return data


@dataclass(frozen=True, slots=True)
class BeeScenarioResult:
    """Verified end-user scenario benchmark and usage guide for one live HiveMind bee or core model."""

    scenario_id: str
    scenario_title: str
    model_id: str
    aliases: tuple[str, ...]
    category: str  # "moe_bee" | "specialist_bee" | "core_foundation" | "dense_coding" | "smart_router"
    badge_label: str
    architecture: str
    underlying_tag: str
    vram_footprint_gb: float
    gpu_residency: str
    cold_load_s: float
    warm_latency_ms: float
    tokens_per_sec: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    http_status: int
    domain: str
    prompt: str
    response_excerpt: str
    when_to_use_fr: str
    when_to_use_en: str
    when_not_to_use: str
    how_to_invoke: str
    recommended_temperature: float
    recommended_max_tokens: int
    suggested_cost_per_1k: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["aliases"] = list(self.aliases)
        return data


BEE_LIFECYCLE_PHASES: tuple[BeeLifecyclePhase, ...] = (
    BeeLifecyclePhase(
        phase_id="PHASE-1-POLLEN-INTENT",
        step_number=1,
        title_fr="1. Butinage d'Intention & Routage de Liquidité (Pollen Classifier)",
        title_en="1. Pollen Intent Classification & Liquidity Routing",
        time_window="0 – 35 ms (Temps réel)",
        key_metric="15 Modèles Actifs • 0 HTTP 402/5xx",
        summary_fr=(
            "Lorsqu'un développeur ou un IDE (VS Code Continue, OpenCode, Cline, Python SDK) envoie une requête, "
            "hivemind/auto analyse l'intention (code Python/Rust, EDO raide, astrophysique, preuve Lean 4, synthèse FR) "
            "et sélectionne l'abeille déjà chaude en VRAM offrant le meilleur équilibre de Nash HMT."
        ),
        summary_en=(
            "When a developer or IDE sends a prompt, hivemind/auto classifies domain intent and selects the bee "
            "already warm in producer VRAM with optimal HMT double-auction liquidity."
        ),
        protocol_endpoints=("/v1/chat/completions", "/v1/models", "/v1/hivemind/market"),
    ),
    BeeLifecyclePhase(
        phase_id="PHASE-2-DAYTIME-NECTAR",
        step_number=2,
        title_fr="2. Récolte de Nectar Diurne & Préemption Jeu < 1,5 s",
        title_en="2. Daytime Nectar Foraging & < 1.5s Game Preemption",
        time_window="07:00 – 22:00 CET (Journée)",
        key_metric="16,5 à 71,9 tok/s (100% GPU) • Yield 1,13 s",
        summary_fr=(
            "Les GPU citoyens (RTX 2070 8 Go) et les gros producteurs (Tesla T4 16 Go) servent les modèles 1.5B–8B "
            "à 100% en VRAM (jusqu'à 71,9 tok/s) et les abeilles MoE 20B (2,4B–3,6B actifs) à 6,3–7,2 tok/s. "
            "Si un joueur lance un jeu Steam/3D, le bouclier local libère 100% de la VRAM en moins de 1,5 seconde."
        ),
        summary_en=(
            "Citizen GPUs (RTX 2070 8 GB) and big-GPU producers (Tesla T4 16 GB) serve 1.5B–8B bees 100% in VRAM "
            "(up to 71.9 tok/s) and 20B Sparse MoE bees at 6.3–7.2 tok/s. Launching any game preempts in < 1.5s."
        ),
        protocol_endpoints=("/v1/hivemind/producer/heartbeat", "/v1/hivemind/t4/status"),
    ),
    BeeLifecyclePhase(
        phase_id="PHASE-3-AMCP-WAGGLE-DANCE",
        step_number=3,
        title_fr="3. Danse Frétillante Inter-Abeilles (Bus AMCP v1.5 / Kafka)",
        title_en="3. Inter-Bee Waggle Dance (AMCP v1.5 / Kafka Coordination)",
        time_window="Coopération Multi-Agents",
        key_metric="4 Topics Kafka • Échange HMT",
        summary_fr=(
            "Quand une abeille généraliste ou de code rencontre un système différentiel raide (Robertson, Navier-Stokes) "
            "ou une orbite symplectique, elle délègue le sous-calcul via hivemind.amcp.bee_coord.v1 à "
            "hivemind/bee-rusty-sundials ou hivemind/bee-stars et échange de la capacité sur hivemind.amcp.bee_trading.v1."
        ),
        summary_en=(
            "Generalist and coding bees delegate stiff ODE/PDE or orbital sub-problems over hivemind.amcp.bee_coord.v1 "
            "to specialist bees (bee-rusty-sundials, bee-stars) and clear capacity on hivemind.amcp.bee_trading.v1."
        ),
        protocol_endpoints=("/v1/hivemind/bees/coordinate", "/v1/hivemind/bees/trade", "/v1/hivemind/amcp/events"),
    ),
    BeeLifecyclePhase(
        phase_id="PHASE-4-NOCTURNAL-ROYAL-JELLY",
        step_number=4,
        title_fr="4. Gelée Royale Nocturne (Distillation BOINC QLoRA-MoE)",
        title_en="4. Nocturnal Royal Jelly (BOINC Chained QLoRA-MoE Distillation)",
        time_window="22:00 – 07:00 CET (Nuit Européenne)",
        key_metric="-5,8% NLL Global • -15,1% Math",
        summary_fr=(
            "La nuit, les gros producteurs GPU basculent sur des workunits BOINC QLoRA 4-bit NF4 chaînées (hmq_*). "
            "Les retours positifs du jour (+1 feedback) et les campagnes votées par la communauté affinent les experts "
            "MoE sous cliquet monotone strict (0 dérive de replay)."
        ),
        summary_en=(
            "Overnight, idle GPUs execute chained 4-bit NF4 QLoRA-MoE BOINC workunits (hmq_*), distilling daytime "
            "thumbs-up traces and community-voted subject sprints under a strict monotone loss ratchet."
        ),
        protocol_endpoints=("/v1/hivemind/campaigns/poll", "/v1/hivemind/feedback", "/v1/hivemind/usage-capture"),
    ),
)


LIFE_AS_A_BEE_SCENARIOS: tuple[BeeScenarioResult, ...] = (
    # --- 1. Sparse MoE Bees (4) ---
    BeeScenarioResult(
        scenario_id="SCEN-RUCHE-MOE-8X",
        scenario_title="La Ruche 8-Expert Top-2 Sparse MoE Routing Synthesis",
        model_id="hivemind/bee-ruche-moe",
        aliases=("bee-ruche-moe", "hivemind/ruche-moe", "la-ruche-moe-8x"),
        category="moe_bee",
        badge_label="🐝 ABEILLE MoE 8-EXPERTS",
        architecture="20B Sparse MoE (2.4B actifs / token • Top-2 sur 8 experts)",
        underlying_tag="gpt-oss:20b",
        vram_footprint_gb=14.0,
        gpu_residency="64% GPU / 36% CPU (14.0 GB Sparse MoE)",
        cold_load_s=128.55,
        warm_latency_ms=16205.4,
        tokens_per_sec=6.79,
        prompt_tokens=80,
        completion_tokens=110,
        total_tokens=190,
        http_status=200,
        domain="general_chat",
        prompt=(
            "In 2 sentences, explain why a Top-2 Sparse Mixture-of-Experts (2.4B active out of 20B total) "
            "achieves higher domain specialization per FLOP than a dense model."
        ),
        response_excerpt=(
            "[Reasoning] We need to explain in 2 sentences why a Top-2 Sparse MoE with 2.4B active out of 20B total "
            "achieves higher domain specialization per FLOP than a dense model. So we need to mention that MoE uses "
            "gating to activate only relevant experts, so each FLOP is used for domain-specific computation..."
        ),
        when_to_use_fr=(
            "Requêtes interdisciplinaires combinant plusieurs domaines de La Ruche (ex. physique + code Python + "
            "synthèse en français). 1,73x plus rapide que le modèle dense 14B (coder-plus) sur T4 16 Go grâce à "
            "l'activation parcimonieuse de 2,4B paramètres par token."
        ),
        when_to_use_en=(
            "Cross-domain requests combining physics, code, and bilingual synthesis. 1.73x faster than dense 14B "
            "on a 16 GB T4 because only 2.4B parameters activate per token."
        ),
        when_not_to_use=(
            "À éviter pour l'autocomplétion temps-réel (< 2s) dans l'IDE : préférez hivemind/coder (2,9s) ou coder-lite (1,5s)."
        ),
        how_to_invoke=(
            'model="hivemind/bee-ruche-moe" (ou alias "hivemind/ruche-moe"). '
            "Conseil : passez reasoning_effort='low' et max_tokens=512 pour limiter la chaîne de pensée interne."
        ),
        recommended_temperature=0.4,
        recommended_max_tokens=512,
        suggested_cost_per_1k=1.8,
    ),
    BeeScenarioResult(
        scenario_id="SCEN-QWEN35-MOE-POLYGLOT",
        scenario_title="Polyglot-Singularity MoE Cross-Domain PDE + Rust Synthesis",
        model_id="hivemind/moe-qwen35-bee",
        aliases=("moe-qwen35-bee", "hivemind/qwen35-moe-bee", "hivemind/polyglot-singularity"),
        category="moe_bee",
        badge_label="🐝 ABEILLE MoE POLYGLOTTE",
        architecture="20B Sparse MoE (3.0B actifs / token • 6 experts QLoRA fusionnés)",
        underlying_tag="gpt-oss:20b",
        vram_footprint_gb=14.0,
        gpu_residency="64% GPU / 36% CPU (14.0 GB Sparse MoE)",
        cold_load_s=128.55,
        warm_latency_ms=17564.7,
        tokens_per_sec=6.26,
        prompt_tokens=54,
        completion_tokens=110,
        total_tokens=164,
        http_status=200,
        domain="science_rusty_sundials",
        prompt="Give the 1D Gross-Pitaevskii soliton conservation invariant (norm N and energy E) in 2 concise sentences.",
        response_excerpt=(
            "[Reasoning] The user: \"Give the 1D Gross-Pitaevskii soliton conservation invariant (norm N and energy E) "
            "in 2 concise sentences.\" They want the expressions for the norm N and energy E for the 1D Gross-Pitaevskii "
            "soliton. Likely the standard 1D GPE: i∂ψ/∂t = -½ ∂^2ψ/∂x^2 + g|ψ|^2 ψ, with g negative for attractive interactions..."
        ),
        when_to_use_fr=(
            "Problèmes scientifiques et algorithmiques polyglottes (physique quantique BEC / Gross-Pitaevskii, "
            "couplage Rust FFI + Python, vérification d'invariants mathématiques)."
        ),
        when_to_use_en=(
            "Polyglot scientific and algorithmic tasks (quantum BEC solitons, Rust FFI + Python coupling, "
            "mathematical invariant checks)."
        ),
        when_not_to_use=(
            "Pour un simple script mono-fichier Rust ou Python sans couplage multi-domaine, utilisez hivemind/qwen35-core (3,66s, 100% GPU)."
        ),
        how_to_invoke=(
            'model="hivemind/moe-qwen35-bee" (ou alias "hivemind/polyglot-singularity"). '
            "Idéal avec temperature=0.3 et max_tokens=600."
        ),
        recommended_temperature=0.3,
        recommended_max_tokens=600,
        suggested_cost_per_1k=1.8,
    ),
    BeeScenarioResult(
        scenario_id="SCEN-MOE-GENERAL-NASH",
        scenario_title="Game-Theoretic Nash Equilibrium Market Clearing",
        model_id="hivemind/moe-general",
        aliases=("moe-general", "hivemind/gpt-oss-20b"),
        category="moe_bee",
        badge_label="🐝 MoE RAISONNEMENT GÉNÉRAL",
        architecture="20B Sparse MoE (3.6B actifs / token • Raisonnement profond)",
        underlying_tag="gpt-oss:20b",
        vram_footprint_gb=14.0,
        gpu_residency="64% GPU / 36% CPU (14.0 GB Sparse MoE)",
        cold_load_s=128.55,
        warm_latency_ms=15276.1,
        tokens_per_sec=7.20,
        prompt_tokens=62,
        completion_tokens=110,
        total_tokens=172,
        http_status=200,
        domain="general_chat",
        prompt="State in 2 concise sentences how proportional HMT credit clearing prevents free-riding in a non-monetary GPU compute market.",
        response_excerpt=(
            "[Reasoning] We need to state in 2 concise sentences how proportional HMT credit clearing prevents "
            "free-riding in a non-monetary GPU compute market. The context: HMT credit clearing presumably refers to a "
            "system where credits are allocated proportionally to contributions, preventing free-riding..."
        ),
        when_to_use_fr=(
            "Analyse stratégique, théorie des jeux, conception d'architecture système, synthèse académique et "
            "raisonnement logique étape par étape. Le plus rapide des 4 modèles MoE 20B (7,20 tok/s, 15,28s)."
        ),
        when_to_use_en=(
            "Strategic analysis, game theory, system architecture design, and step-by-step logical reasoning. "
            "Fastest of the 20B MoE models at 7.20 tok/s."
        ),
        when_not_to_use=(
            "Si vous avez besoin d'une réponse instantanée en français sans chaîne de raisonnement longue, utilisez hivemind/gemma-core (2,08s) ou mistral-core (2,87s)."
        ),
        how_to_invoke=(
            'model="hivemind/moe-general" avec reasoning_effort="low" ou "medium" et max_tokens=700.'
        ),
        recommended_temperature=0.5,
        recommended_max_tokens=700,
        suggested_cost_per_1k=1.6,
    ),
    BeeScenarioResult(
        scenario_id="SCEN-MOE-CODER-AGENTIC",
        scenario_title="Frontier Agentic Multi-File Refactoring Strategy",
        model_id="hivemind/moe-coder",
        aliases=("moe-coder", "hivemind/qwen3-coder-moe"),
        category="moe_bee",
        badge_label="🐝 MoE CODE AGENTIQUE",
        architecture="20B–30B Sparse MoE (3.3B actifs / token • Refactoring & Agents)",
        underlying_tag="gpt-oss:20b",
        vram_footprint_gb=14.0,
        gpu_residency="64% GPU / 36% CPU (14.0 GB Sparse MoE)",
        cold_load_s=128.55,
        warm_latency_ms=15848.3,
        tokens_per_sec=6.94,
        prompt_tokens=56,
        completion_tokens=110,
        total_tokens=166,
        http_status=200,
        domain="coding_python",
        prompt="List 3 concise invariants an autonomous coding agent must verify before hot-swapping a production router module.",
        response_excerpt=(
            "[Reasoning] We need to list 3 concise invariants an autonomous coding agent must verify before hot-swapping "
            "a production router module. Likely invariants: 1) All active connections are gracefully drained or preserved "
            "2) State/schema compatibility is verified 3) Rollback health check passes..."
        ),
        when_to_use_fr=(
            "Planification agentique complexe (OpenCode, Cline, Aider), revue de sécurité multi-fichiers, "
            "conception d'invariants de hot-swap et refactoring d'architecture. 1,77x plus rapide que coder-plus (14B dense)."
        ),
        when_to_use_en=(
            "Complex agentic planning (OpenCode, Cline, Aider), multi-file security audits, hot-swap invariant "
            "verification, and architectural refactoring."
        ),
        when_not_to_use=(
            "Pour générer rapidement une fonction isolée ou un test unitaire, hivemind/coder (32,77 tok/s, 2,93s) est 11x plus rapide."
        ),
        how_to_invoke=(
            'model="hivemind/moe-coder" dans Aider/OpenCode pour l\'étape Architecte, puis "hivemind/coder" pour l\'étape Éditeur.'
        ),
        recommended_temperature=0.2,
        recommended_max_tokens=800,
        suggested_cost_per_1k=2.0,
    ),
    # --- 2. Specialist Domain Bees (3) ---
    BeeScenarioResult(
        scenario_id="SCEN-ASTRO-TTV",
        scenario_title="Exoplanet Transit Timing Variations & Schwarzschild Metric",
        model_id="hivemind/bee-stars",
        aliases=("bee-stars", "hivemind-bee-stars", "hivemind/astro-bee"),
        category="specialist_bee",
        badge_label="🔭 ABEILLE ASTROPHYSIQUE (ÉPINGLÉE VRAM)",
        architecture="7B Dense + Adaptateur QLoRA Symplectique & Astrophysique",
        underlying_tag="hivemind-bee-stars",
        vram_footprint_gb=4.9,
        gpu_residency="100% GPU (llama-server :8294 épinglé en VRAM)",
        cold_load_s=0.0,
        warm_latency_ms=3442.2,
        tokens_per_sec=27.89,
        prompt_tokens=67,
        completion_tokens=96,
        total_tokens=163,
        http_status=200,
        domain="science_astronomy",
        prompt=(
            "Give the exact formula for the Schwarzschild radius R_s and explain in 2 sentences how symplectic "
            "Velocity-Verlet preserves energy in exoplanet N-body TTV simulations."
        ),
        response_excerpt=(
            "The Schwarzschild radius \\( R_s \\) of a black hole is given by the formula: "
            "\\[ R_s = \\frac{2GM}{c^2} \\] where \\( G \\) is the gravitational constant, \\( M \\) is the mass "
            "of the black hole, and \\( c \\) is the speed of light. In exoplanet N-body TTV (Transit Timing Variations) "
            "simulations, symplectic Velocity-Verlet algorithm preserves energy..."
        ),
        when_to_use_fr=(
            "Astrophysique, mécanique céleste, intégrateurs symplectiques (Velocity-Verlet, Yoshida), variations de "
            "temps de transit d'exoplanètes (TTV), relativité générale et co-pilote scientifique Unity/Steam. "
            "Zéro temps de chargement à froid (0,0s) car épinglée en permanence sur le port :8294 !"
        ),
        when_to_use_en=(
            "Astrophysics, orbital mechanics, symplectic integrators (Velocity-Verlet, Yoshida), exoplanet TTV, "
            "and general relativity. 0.0s cold-start latency (permanently pinned in VRAM on :8294)."
        ),
        when_not_to_use=(
            "Ne pas utiliser pour du développement web TypeScript/React ou de la bureautique générale."
        ),
        how_to_invoke=(
            'model="hivemind/bee-stars" (ou alias "hivemind/astro-bee"). Répond immédiatement (0s cold load) à 27,89 tok/s.'
        ),
        recommended_temperature=0.2,
        recommended_max_tokens=512,
        suggested_cost_per_1k=1.1,
    ),
    BeeScenarioResult(
        scenario_id="SCEN-SUNDIALS-ODE",
        scenario_title="Stiff Robertson Chemical Kinetics BDF/Newton Rescue",
        model_id="hivemind/bee-rusty-sundials",
        aliases=("bee-rusty-sundials", "hivemind/rusty-sundials", "rusty-sundials-bee"),
        category="specialist_bee",
        badge_label="🔬 ABEILLE EDO/EDP & RUSTY-SUNDIALS",
        architecture="7B Spécialiste Calcul Scientifique + Solveur BDF/RK45 Déterministe",
        underlying_tag="qwen2.5-coder:7b-instruct",
        vram_footprint_gb=5.6,
        gpu_residency="100% GPU (5.6 GB VRAM)",
        cold_load_s=12.53,
        warm_latency_ms=2887.2,
        tokens_per_sec=33.25,
        prompt_tokens=83,
        completion_tokens=96,
        total_tokens=179,
        http_status=200,
        domain="science_rusty_sundials",
        prompt=(
            "Explain in 3 concise bullet points why explicit RK45 fails on the stiff Robertson ODE system "
            "(k1=0.04, k2=1e4, k3=3e7) and why BDF with analytical Jacobian succeeds."
        ),
        response_excerpt=(
            "- **Explicit RK45 Fails on Stiff Robertson ODE System**: Explicit Runge-Kutta methods like RK45 are "
            "designed for non-stiff systems, where the solution changes rapidly over a short interval. The Robertson "
            "ODE system is stiff due to the large difference in time scales between the reactions..."
        ),
        when_to_use_fr=(
            "Systèmes d'équations différentielles ordinaires ou partielles raides (EDO/EDP : cinétique chimique de "
            "Robertson, mécanique des fluides Navier-Stokes, solitons de Gross-Pitaevskii, génétique de Wright-Fisher) "
            "et génération de code Rust/Python utilisant SUNDIALS (CVODE, ARKODE, IDA, KINSOL)."
        ),
        when_to_use_en=(
            "Stiff ODE/PDE systems (Robertson chemical kinetics, Navier-Stokes fluid mechanics, Gross-Pitaevskii "
            "quantum solitons, Wright-Fisher genetics) and Rust/Python SUNDIALS solver code."
        ),
        when_not_to_use=(
            "Pour du code CRUD standard ou de la traduction généraliste."
        ),
        how_to_invoke=(
            'Chat : model="hivemind/bee-rusty-sundials" sur /v1/chat/completions (33,25 tok/s). '
            "Calcul déterministe certifié : POST /v1/hivemind/bees/rusty-sundials/rescue."
        ),
        recommended_temperature=0.1,
        recommended_max_tokens=600,
        suggested_cost_per_1k=1.1,
    ),
    BeeScenarioResult(
        scenario_id="SCEN-LEAN4-PROVER",
        scenario_title="Formal Lean 4 Energy Ratchet Monotonicity Proof",
        model_id="hivemind/bee-lean-prover",
        aliases=("bee-lean-prover", "hivemind/goedel-prover-v2", "hivemind/lean-prover"),
        category="specialist_bee",
        badge_label="📐 ABEILLE PREUVE FORMELLE LEAN 4",
        architecture="8B Goedel-Prover-V2 (Q6_K Haute Précision • Synthèse de Tactiques Lean 4)",
        underlying_tag="hf.co/mradermacher/Goedel-Prover-V2-8B-GGUF:Q6_K",
        vram_footprint_gb=8.7,
        gpu_residency="100% GPU (8.7 GB VRAM Q6_K)",
        cold_load_s=42.27,
        warm_latency_ms=4840.2,
        tokens_per_sec=16.53,
        prompt_tokens=57,
        completion_tokens=80,
        total_tokens=137,
        http_status=200,
        domain="coding_general",
        prompt=(
            "Write a complete Lean 4 theorem proving `theorem ratchet_le (e_parent e_child : Nat) "
            "(h : e_child ≤ e_parent) : Nat.min e_parent e_child ≤ e_parent := by` without using sorry."
        ),
        response_excerpt=(
            "[Reasoning] Okay, I need to prove that `Nat.min e_parent e_child ≤ e_parent` given that "
            "`e_child ≤ e_parent`. First, recall that `Nat.min a b` is the minimum of `a` and `b`, i.e., "
            "`Nat.min a b = a` if `a ≤ b`, and `Nat.min a b = b` if `b ≤ a`..."
        ),
        when_to_use_fr=(
            "Démonstration formelle de théorèmes en Lean 4 (Mathlib4), élimination de `sorry`, synthèse de tactiques "
            "(`omega`, `linarith`, `simp`, `exact`) et vérification formelle d'invariants de sécurité ou d'énergie."
        ),
        when_to_use_en=(
            "Formal theorem proving in Lean 4 (Mathlib4), eliminating `sorry`, synthesizing tactics (`omega`, "
            "`linarith`, `simp`), and verifying security or energy invariants."
        ),
        when_not_to_use=(
            "Ne pas utiliser pour du code Python/JS classique : Goedel-Prover-V2 est spécialisé pour Lean 4 et génère une réflexion formelle préalable."
        ),
        how_to_invoke=(
            'model="hivemind/bee-lean-prover" (ou "hivemind/goedel-prover-v2") avec max_tokens=512 et temperature=0.1.'
        ),
        recommended_temperature=0.1,
        recommended_max_tokens=512,
        suggested_cost_per_1k=1.3,
    ),
    # --- 3. Core Foundation Models (Gemma 3, Qwen 3.5, Mistral, CrowdGPT) (4) ---
    BeeScenarioResult(
        scenario_id="SCEN-GEMMA-BILINGUAL",
        scenario_title="Synthèse Bilingue FR/EN & Souveraineté Numérique Européenne",
        model_id="hivemind/gemma-core",
        aliases=("gemma-core", "hivemind/gemma3", "gemma3:4b"),
        category="core_foundation",
        badge_label="🌍 CORE GEMMA 3 (GOOGLE • 4B)",
        architecture="4B Google Gemma 3 Dense (128k Context • Multilingue FR/EN • 100% GPU)",
        underlying_tag="gemma3:4b",
        vram_footprint_gb=3.8,
        gpu_residency="100% GPU (3.8 GB VRAM)",
        cold_load_s=32.10,
        warm_latency_ms=2082.0,
        tokens_per_sec=36.98,
        prompt_tokens=42,
        completion_tokens=77,
        total_tokens=119,
        http_status=200,
        domain="general_chat",
        prompt="Réponds en français en 2 phrases : pourquoi l'IA citoyenne décentralisée protège-t-elle la souveraineté numérique européenne ?",
        response_excerpt=(
            "L'IA citoyenne décentralisée protège la souveraineté numérique européenne en permettant aux citoyens "
            "de contrôler leurs propres données et de participer activement au développement de l'IA, sans dépendre "
            "de grandes entreprises étrangères. En distribuant le pouvoir de création et de décision, elle renforce "
            "l'autonomie et l'identité numérique de l'Europe."
        ),
        when_to_use_fr=(
            "Excellent modèle fondationnel léger (3,8 Go VRAM) pour la rédaction en français et en anglais, "
            "le résumé de longs documents, l'extraction JSON structurée et l'explication pédagogique rapide (2,08s à 36,98 tok/s)."
        ),
        when_to_use_en=(
            "Fast, lightweight (3.8 GB VRAM) foundation model for French/English writing, document summarization, "
            "structured JSON extraction, and concise explanations (2.08s at 36.98 tok/s)."
        ),
        when_not_to_use=(
            "Pour du code système Rust complexe ou des preuves Lean 4, préférez hivemind/qwen35-core ou hivemind/coder."
        ),
        how_to_invoke=(
            'model="hivemind/gemma-core" (ou alias "hivemind/gemma3"). Très économe en HMT (0,6 HMT/1k) et 100% GPU.'
        ),
        recommended_temperature=0.5,
        recommended_max_tokens=512,
        suggested_cost_per_1k=0.6,
    ),
    BeeScenarioResult(
        scenario_id="SCEN-QWEN35-RUST",
        scenario_title="Hybrid Reasoning & Zero-Allocation Rust Ring Buffer",
        model_id="hivemind/qwen35-core",
        aliases=("qwen35-core", "hivemind/qwen3.5", "hivemind/qwen3-8b", "qwen3:8b"),
        category="core_foundation",
        badge_label="🧠 CORE QWEN 3.5 (HYBRID THINKING • 8B)",
        architecture="8B Qwen 3/3.5 Dense avec Raisonnement Hybride (/think & /no_think)",
        underlying_tag="qwen3:8b",
        vram_footprint_gb=7.4,
        gpu_residency="100% GPU (7.4 GB VRAM)",
        cold_load_s=29.95,
        warm_latency_ms=3663.9,
        tokens_per_sec=30.02,
        prompt_tokens=64,
        completion_tokens=110,
        total_tokens=174,
        http_status=200,
        domain="coding_rust",
        prompt=(
            "/no_think Write a minimal Rust struct `RingAvg<const N: usize>` with `push(&mut self, x: f64) -> f64` "
            "using a fixed array `[f64; N]` and zero heap allocations. Code only."
        ),
        response_excerpt=(
            "[Reasoning] Okay, I need to create a minimal Rust struct called RingAvg with a fixed-size array of N "
            "elements. The struct should have a method push that takes an f64 and returns an f64. The goal is to "
            "compute the average of the last N elements, right? And no heap allocations, so everything must be on the stack..."
        ),
        when_to_use_fr=(
            "Programmation système exigeante (Rust sans allocation tas, C++, Python typé), algorithmique avec "
            "vérification d'étapes, et appels d'outils (tool-calling). Tourne à 100% en VRAM sur le T4 (30,02 tok/s, 3,66s)."
        ),
        when_to_use_en=(
            "Demanding systems programming (zero-allocation Rust, C++, typed Python), algorithmic problem solving "
            "with hybrid thinking, and structured tool calling. Runs 100% in GPU VRAM at 30.02 tok/s."
        ),
        when_not_to_use=(
            "Si vous voulez uniquement le bloc de code brut sans tokens de réflexion `[Reasoning]`, utilisez hivemind/coder (2,93s)."
        ),
        how_to_invoke=(
            'model="hivemind/qwen35-core" (ou alias "hivemind/qwen3.5"). Prévoyez max_tokens >= 256 car le modèle '
            "effectue une vérification interne avant d'émettre le code final."
        ),
        recommended_temperature=0.3,
        recommended_max_tokens=512,
        suggested_cost_per_1k=1.1,
    ),
    BeeScenarioResult(
        scenario_id="SCEN-MISTRAL-FR-DOC",
        scenario_title="Architecture Zero-Trust & Charte de Réciprocité (FR)",
        model_id="hivemind/mistral-core",
        aliases=("mistral-core", "hivemind/mistral", "mistral:7b-instruct"),
        category="core_foundation",
        badge_label="🇫🇷 CORE MISTRAL 7B (BILINGUE FR/EN)",
        architecture="7B Mistral v0.3 Instruct Dense (Français / Anglais & Synthèse)",
        underlying_tag="mistral:7b-instruct",
        vram_footprint_gb=6.7,
        gpu_residency="100% GPU (6.7 GB VRAM)",
        cold_load_s=30.29,
        warm_latency_ms=2869.4,
        tokens_per_sec=33.46,
        prompt_tokens=56,
        completion_tokens=96,
        total_tokens=152,
        http_status=200,
        domain="general_chat",
        prompt=(
            "Explique en 3 points concis en français comment le bouclier local (Producer Shield) et la préemption "
            "en moins de 1,5 seconde protègent les joueurs sur HiveMind."
        ),
        response_excerpt=(
            "1. Le bouclier local, ou Producer Shield, est une fonctionnalité de HiveMind qui protège les producteurs "
            "locaux... 2. La préemption en moins de 1,5 seconde garantit que dès qu'un jeu est lancé, la carte graphique "
            "est immédiatement rendue au joueur sans perte de FPS..."
        ),
        when_to_use_fr=(
            "Rédaction technique en français, synthèse de documentation, brainstorming, traduction FR/EN et "
            "dialogue naturel rapide (2,87s à 33,46 tok/s, 100% GPU sans préambule de réflexion)."
        ),
        when_to_use_en=(
            "French technical writing, documentation synthesis, brainstorming, FR/EN translation, and direct "
            "natural dialogue (2.87s at 33.46 tok/s, 100% GPU)."
        ),
        when_not_to_use=(
            "Moins spécialisé en génération de code pur que hivemind/coder ou hivemind/qwen35-core."
        ),
        how_to_invoke=(
            'model="hivemind/mistral-core" (ou alias "hivemind/mistral") avec temperature=0.5.'
        ),
        recommended_temperature=0.5,
        recommended_max_tokens=512,
        suggested_cost_per_1k=0.9,
    ),
    BeeScenarioResult(
        scenario_id="SCEN-CROWD-CITIZEN",
        scenario_title="CrowdGPT × La Ruche Citizen AI Reciprocity Summary",
        model_id="hivemind/crowd-v1",
        aliases=("crowd-v1", "vxtzq/crowd-v1", "crowdai/crowd-v1"),
        category="core_foundation",
        badge_label="🤝 PARTENAIRE CROWDGPT (ULTRA-RAPIDE • 1.5B)",
        architecture="1B–1.5B Dense Tronc Généraliste Citoyen (1.6 GB VRAM • 100% GPU)",
        underlying_tag="qwen2.5-coder:1.5b",
        vram_footprint_gb=1.6,
        gpu_residency="100% GPU (1.6 GB VRAM)",
        cold_load_s=13.93,
        warm_latency_ms=1001.7,
        tokens_per_sec=71.87,
        prompt_tokens=44,
        completion_tokens=72,
        total_tokens=116,
        http_status=200,
        domain="general_chat",
        prompt="Explain in 2 short sentences how a citizen GPU swarm shares idle compute without money.",
        response_excerpt=(
            "A citizen GPU swarm shares idle compute without money by leveraging the power of GPUs in the cloud, "
            "which can be accessed by anyone with an internet connection. The swarm can be programmed to automatically "
            "detect idle GPUs and distribute the workload to them, without requiring any payment."
        ),
        when_to_use_fr=(
            "Modèle le plus rapide de toute la ruche (**71,87 tok/s**, **1,00s** de latence !). Parfait pour les "
            "tests de connectivité, les micro-résumés instantanés, les chatbots à très faible latence et les GPU "
            "citoyens de 2 à 4 Go de VRAM."
        ),
        when_to_use_en=(
            "Fastest model on the entire swarm (71.87 tok/s, 1.00s latency!). Ideal for health probes, instant "
            "micro-summaries, ultra-low-latency chat, and 2–4 GB VRAM citizen GPUs."
        ),
        when_not_to_use=(
            "Capacité de raisonnement limitée (1.5B) : ne pas utiliser pour des preuves mathématiques ou de l'architecture complexe."
        ),
        how_to_invoke=(
            'model="hivemind/crowd-v1" (ou alias "vxtzq/crowd-v1"). Coût minimal : 0,3 HMT/1k.'
        ),
        recommended_temperature=0.4,
        recommended_max_tokens=256,
        suggested_cost_per_1k=0.3,
    ),
    # --- 4. Dense Coding & Smart Router Tiers (4) ---
    BeeScenarioResult(
        scenario_id="SCEN-CODER-PYTHON",
        scenario_title="High-Performance Python Decorator with Microsecond Telemetry",
        model_id="hivemind/coder",
        aliases=("coder", "qwen2.5-coder:7b-instruct"),
        category="dense_coding",
        badge_label="💻 PILIER CODE QUOTIDIEN (7B • RTX 2070 + T4)",
        architecture="7B Qwen 2.5 Coder Instruct Dense (Multi-Producteurs Actifs)",
        underlying_tag="qwen2.5-coder:7b-instruct",
        vram_footprint_gb=5.6,
        gpu_residency="100% GPU (5.6 GB VRAM • Servi par T4 + RTX 2070)",
        cold_load_s=12.53,
        warm_latency_ms=2929.7,
        tokens_per_sec=32.77,
        prompt_tokens=68,
        completion_tokens=96,
        total_tokens=164,
        http_status=200,
        domain="coding_python",
        prompt=(
            "Write a concise Python decorator `@timed_lru` using `functools.lru_cache` and `time.perf_counter_ns` "
            "that logs execution time in microseconds. Code only."
        ),
        response_excerpt=(
            "```python\nimport functools\nimport time\n\ndef timed_lru(maxsize=None, typed=False):\n"
            "    def decorator(func):\n        @functools.wraps(func)\n        def wrapper(*args, **kwargs):\n"
            "            start_time = time.perf_counter_ns()\n            result = func(*args, **kwargs)..."
        ),
        when_to_use_fr=(
            "**Le choix par défaut recommandé pour 80% du développement quotidien** dans VS Code Continue, OpenCode, "
            "Zed ou PyCharm : génération de fonctions Python/TypeScript/Rust/Go/SQL, écriture de tests pytest, "
            "correction de bugs et refactoring local. Toujours chaud en VRAM (2,93s à 32,77 tok/s)."
        ),
        when_to_use_en=(
            "Recommended default for 80% of daily coding in VS Code Continue, OpenCode, Zed, or PyCharm: Python/TS/Rust "
            "functions, pytest generation, bug fixes, and local refactoring (2.93s at 32.77 tok/s)."
        ),
        when_not_to_use=(
            "Pour des preuves formelles Lean 4 (utilisez bee-lean-prover) ou des systèmes d'EDO raides (utilisez bee-rusty-sundials)."
        ),
        how_to_invoke=(
            'model="hivemind/coder" avec max_tokens=700 et temperature=0.2 dans votre configuration IDE.'
        ),
        recommended_temperature=0.2,
        recommended_max_tokens=700,
        suggested_cost_per_1k=1.0,
    ),
    BeeScenarioResult(
        scenario_id="SCEN-CODER-LITE-INLINE",
        scenario_title="Ultra-Fast IDE TypeScript Debounce Utility",
        model_id="hivemind/coder-lite",
        aliases=("coder-lite", "qwen2.5-coder:3b"),
        category="dense_coding",
        badge_label="⚡ CODE ULTRA-RAPIDE (3B • 53 TOK/S)",
        architecture="3B Qwen 2.5 Coder Dense (Faible Latence IDE • 2.7 GB VRAM)",
        underlying_tag="qwen2.5-coder:3b",
        vram_footprint_gb=2.7,
        gpu_residency="100% GPU (2.7 GB VRAM)",
        cold_load_s=20.32,
        warm_latency_ms=1508.7,
        tokens_per_sec=53.03,
        prompt_tokens=60,
        completion_tokens=80,
        total_tokens=140,
        http_status=200,
        domain="coding_web_ts",
        prompt="Write a compact TypeScript `debounce<T extends (...args: any[]) => void>(fn: T, ms: number)` function. Return code only.",
        response_excerpt=(
            "```typescript\nfunction debounce<T extends (...args: any[]) => void>(fn: T, ms: number): T {\n"
            "  let timeoutId: NodeJS.Timeout | null = null;\n  return function debounced(...args: Parameters<T>): void {\n"
            "    if (timeoutId) {\n      clearTimeout(timeoutId);\n    }\n    timeoutId = setTimeout(() => fn(...args), ms);\n  }"
        ),
        when_to_use_fr=(
            "Petites éditions rapides dans l'IDE, génération de regex, commandes shell/git, docstrings et snippets "
            "TypeScript/Python en **1,51 seconde** à **53,03 tok/s**."
        ),
        when_to_use_en=(
            "Fast inline IDE edits, regex generation, shell/git one-liners, docstrings, and compact TypeScript/Python "
            "utilities in 1.51s at 53.03 tok/s."
        ),
        when_not_to_use=(
            "Architecture multi-fichiers ou algorithmique complexe."
        ),
        how_to_invoke=(
            'model="hivemind/coder-lite" avec max_tokens=400 et temperature=0.2 (0,4 HMT/1k).'
        ),
        recommended_temperature=0.2,
        recommended_max_tokens=400,
        suggested_cost_per_1k=0.4,
    ),
    BeeScenarioResult(
        scenario_id="SCEN-R1-ARCHITECTURE",
        scenario_title="Multi-File Distributed Consensus & Byzantine Quorum Verification",
        model_id="hivemind/coder-plus",
        aliases=("coder-plus", "deepseek-r1:14b"),
        category="dense_coding",
        badge_label="🏗️ DEEPSEEK-R1 14B (RAISONNEMENT DENSE)",
        architecture="14B DeepSeek-R1 Distill Dense (Chaîne de Pensée Profonde)",
        underlying_tag="deepseek-r1:14b",
        vram_footprint_gb=12.0,
        gpu_residency="75% GPU / 25% CPU (12.0 GB Dense)",
        cold_load_s=58.81,
        warm_latency_ms=27992.6,
        tokens_per_sec=3.93,
        prompt_tokens=34,
        completion_tokens=110,
        total_tokens=144,
        http_status=200,
        domain="coding_general",
        prompt="In 2 concise sentences, state why Multi-Krum aggregation tolerates f Byzantine workers out of n when 2f + 2 < n.",
        response_excerpt=(
            "[Reasoning] Okay, so I need to figure out why Multi-Krum aggregation can tolerate f Byzantine workers "
            "out of n when 2f + 2 < n. Hmm, I remember that Byzantine fault tolerance is about handling situations "
            "where some nodes can act maliciously or fail in arbitrary ways..."
        ),
        when_to_use_fr=(
            "Vérification approfondie de protocoles distribués (Multi-Krum, consensus Byzantin), débogage de "
            "conditions de course subtiles et raisonnement mathématique dense."
        ),
        when_to_use_en=(
            "Deep verification of distributed protocols (Multi-Krum, Byzantine consensus), subtle concurrency "
            "debugging, and dense mathematical reasoning."
        ),
        when_not_to_use=(
            "Sur un GPU de 16 Go partagé, coder-plus (14B dense, 3,93 tok/s) déborde de 25% sur le CPU : les abeilles "
            "MoE 20B (`bee-ruche-moe`, `moe-coder`, `moe-general`) sont **1,75x plus rapides** (6,8–7,2 tok/s) !"
        ),
        how_to_invoke=(
            'model="hivemind/coder-plus" pour des tâches asynchrones de fond, ou préférez "hivemind/moe-coder" en interactif.'
        ),
        recommended_temperature=0.4,
        recommended_max_tokens=800,
        suggested_cost_per_1k=1.8,
    ),
    BeeScenarioResult(
        scenario_id="SCEN-AUTO-SMART-ROUTER",
        scenario_title="Smart Intent Auto-Routing (Astrophysics -> bee-stars / Code -> coder)",
        model_id="hivemind/auto",
        aliases=("auto",),
        category="smart_router",
        badge_label="🧭 ROUTEUR INTELLIGENT D'INTENTION",
        architecture="Méta-Routeur d'Intention + Équilibre de Nash HMT + Résidence VRAM",
        underlying_tag="smart-intent-router",
        vram_footprint_gb=0.0,
        gpu_residency="100% GPU (Délègue aux abeilles chaudes en VRAM)",
        cold_load_s=0.0,
        warm_latency_ms=3056.0,
        tokens_per_sec=17.67,
        prompt_tokens=57,
        completion_tokens=54,
        total_tokens=111,
        http_status=200,
        domain="science_astronomy",
        prompt="Calculate the orbital period ratio of a 2:1 mean-motion exoplanet resonance and state Kepler's third law in 2 sentences.",
        response_excerpt=(
            "The orbital period ratio of a 2:1 mean-motion exoplanet resonance is 2. Kepler's third law states that "
            "the square of the orbital period of a planet is directly proportional to the cube of the semi-major axis "
            "of its orbit around the Sun."
        ),
        when_to_use_fr=(
            "**Quand vous ne savez pas quelle abeille choisir !** `hivemind/auto` privilégie les modèles déjà chargés "
            "en VRAM (`loaded_in_vram: true`) pour éviter tout temps de chargement disque à froid et route "
            "automatiquement vers l'abeille disponible la plus pertinente."
        ),
        when_to_use_en=(
            "When you don't want to pick a model manually! `hivemind/auto` prioritizes models already hot in producer "
            "VRAM (`loaded_in_vram: true`) to avoid cold-load latency and routes to the best available bee."
        ),
        when_not_to_use=(
            "Si vous exigez un modèle précis (ex. syntaxe Lean 4 stricte avec `hivemind/bee-lean-prover`), appelez ce modèle explicitement."
        ),
        how_to_invoke=(
            'model="hivemind/auto" — aucune configuration supplémentaire requise.'
        ),
        recommended_temperature=0.4,
        recommended_max_tokens=512,
        suggested_cost_per_1k=1.0,
    ),
)


BEE_DECISION_MATRIX: tuple[dict[str, Any], ...] = (
    {
        "use_case_fr": "Code quotidien Python / TS / SQL / Tests (VS Code, OpenCode, Zed)",
        "use_case_en": "Everyday Python / TS / SQL coding & unit tests in IDEs",
        "primary_bee": "hivemind/coder",
        "backup_bee": "hivemind/coder-lite",
        "measured_speed": "32.77 tok/s (2.93s)",
        "gpu_residency": "100% GPU (T4 + RTX 2070)",
        "why": "Toujours chaud en VRAM sur plusieurs GPU, génère du code propre sans préambule.",
    },
    {
        "use_case_fr": "Éditions inline ultra-rapides, regex, one-liners shell (< 1,6s)",
        "use_case_en": "Ultra-fast inline edits, regex, shell one-liners (< 1.6s)",
        "primary_bee": "hivemind/coder-lite",
        "backup_bee": "hivemind/crowd-v1",
        "measured_speed": "53.03 tok/s (1.51s)",
        "gpu_residency": "100% GPU (2.7 GB)",
        "why": "2x plus rapide que les modèles 7B pour les petites transformations syntaxiques.",
    },
    {
        "use_case_fr": "Rust système zéro-allocation, C++, algorithmique & réflexion hybride",
        "use_case_en": "Zero-allocation Rust, C++, algorithms & hybrid thinking",
        "primary_bee": "hivemind/qwen35-core",
        "backup_bee": "hivemind/moe-qwen35-bee",
        "measured_speed": "30.02 tok/s (3.66s)",
        "gpu_residency": "100% GPU (7.4 GB)",
        "why": "Vérifie les contraintes mémoire et de typage avant d'écrire la solution.",
    },
    {
        "use_case_fr": "Rédaction en français, synthèse bilingue FR/EN & JSON structuré",
        "use_case_en": "French technical writing, FR/EN bilingual synthesis & JSON",
        "primary_bee": "hivemind/gemma-core",
        "backup_bee": "hivemind/mistral-core",
        "measured_speed": "36.98 tok/s (2.08s)",
        "gpu_residency": "100% GPU (3.8 GB)",
        "why": "Excellent français naturel, très faible coût HMT (0,6/1k) et latence de 2 secondes.",
    },
    {
        "use_case_fr": "Astrophysique, orbites N-corps symplectiques, TTV & relativité",
        "use_case_en": "Astrophysics, symplectic N-body orbits, exoplanet TTV & GR",
        "primary_bee": "hivemind/bee-stars",
        "backup_bee": "hivemind/moe-qwen35-bee",
        "measured_speed": "27.89 tok/s (3.44s)",
        "gpu_residency": "100% GPU (Épinglé :8294)",
        "why": "0,0s de chargement à froid (épinglé en permanence en VRAM) et formules exactes.",
    },
    {
        "use_case_fr": "Équations différentielles raides (EDO/EDP), Navier-Stokes, SUNDIALS",
        "use_case_en": "Stiff ODE/PDE systems, Navier-Stokes fluids, Rusty-SUNDIALS",
        "primary_bee": "hivemind/bee-rusty-sundials",
        "backup_bee": "hivemind/moe-qwen35-bee",
        "measured_speed": "33.25 tok/s (2.89s)",
        "gpu_residency": "100% GPU (5.6 GB)",
        "why": "Couplé à l'endpoint déterministe POST /v1/hivemind/bees/rusty-sundials/rescue.",
    },
    {
        "use_case_fr": "Preuves formelles Lean 4 (Mathlib4) sans sorry",
        "use_case_en": "Formal Lean 4 theorem proving (Mathlib4) without sorry",
        "primary_bee": "hivemind/bee-lean-prover",
        "backup_bee": "hivemind/qwen35-core",
        "measured_speed": "16.53 tok/s (4.84s)",
        "gpu_residency": "100% GPU (8.7 GB Q6_K)",
        "why": "Basé sur Goedel-Prover-V2-8B quantifié en Q6_K haute précision.",
    },
    {
        "use_case_fr": "Synthèse MoE 20B multi-experts, théorie des jeux & refactoring agentique",
        "use_case_en": "20B Sparse MoE synthesis, game theory & agentic multi-file planning",
        "primary_bee": "hivemind/bee-ruche-moe",
        "backup_bee": "hivemind/moe-coder",
        "measured_speed": "6.79–7.20 tok/s (~15.3s)",
        "gpu_residency": "64% GPU / 36% CPU (14 GB MoE)",
        "why": "Active seulement 2,4B à 3,6B paramètres/token : 1,8x plus rapide que le 14B dense !",
    },
)


def life_as_a_bee_payload() -> dict[str, Any]:
    """Return the complete JSON representation of the Life as a Bee dashboard and benchmark."""
    total_tokens = sum(s.total_tokens for s in LIFE_AS_A_BEE_SCENARIOS)
    mean_latency = round(sum(s.warm_latency_ms for s in LIFE_AS_A_BEE_SCENARIOS) / len(LIFE_AS_A_BEE_SCENARIOS), 1)
    mean_speed = round(sum(s.tokens_per_sec for s in LIFE_AS_A_BEE_SCENARIOS) / len(LIFE_AS_A_BEE_SCENARIOS), 2)
    gpu_100_models = [s.model_id for s in LIFE_AS_A_BEE_SCENARIOS if s.gpu_residency.startswith("100% GPU")]
    moe_models = [s.model_id for s in LIFE_AS_A_BEE_SCENARIOS if s.category == "moe_bee"]
    return {
        "object": "hivemind.life_as_a_bee",
        "campaign_id": "HM-S19-LIFE-AS-A-BEE-BENCHMARK",
        "executed_at_utc": "2026-10-28T06:21:59Z",
        "dashboard_url": "/life-as-a-bee",
        "summary": {
            "total_scenarios_executed": len(LIFE_AS_A_BEE_SCENARIOS),
            "all_passed_200_ok": all(s.http_status == 200 for s in LIFE_AS_A_BEE_SCENARIOS),
            "total_tokens_processed": total_tokens,
            "mean_warm_latency_ms": mean_latency,
            "mean_tokens_per_sec": mean_speed,
            "fastest_bee_model": "hivemind/crowd-v1",
            "fastest_bee_tokens_per_sec": 71.87,
            "moe_speedup_vs_14b_dense": 1.83,
            "hundred_percent_gpu_models": gpu_100_models,
            "sparse_moe_models": moe_models,
        },
        "lifecycle_phases": [p.to_dict() for p in BEE_LIFECYCLE_PHASES],
        "decision_matrix": list(BEE_DECISION_MATRIX),
        "scenarios": [s.to_dict() for s in LIFE_AS_A_BEE_SCENARIOS],
    }


def render_life_as_a_bee_section() -> str:
    """Render the interactive ``#life-as-a-bee`` HTML section for embedding in ``portal_page.py``."""
    lifecycle_cards: list[str] = []
    for phase in BEE_LIFECYCLE_PHASES:
        endpoints_html = " &bull; ".join(f"<code>{html.escape(ep)}</code>" for ep in phase.protocol_endpoints)
        lifecycle_cards.append(f"""
        <div class="tier-card" style="border-color:rgba(251,191,36,0.28);">
          <div>
            <div class="tier-badge">{html.escape(phase.time_window)}</div>
            <div class="tier-name">{html.escape(phase.title_fr)}</div>
            <div class="tier-vram">{html.escape(phase.key_metric)}</div>
            <div class="tier-desc">{html.escape(phase.summary_fr)}</div>
          </div>
          <div style="font-size:0.7rem;color:var(--text-dim);margin-top:0.4rem;">{endpoints_html}</div>
        </div>
        """)

    matrix_rows: list[str] = []
    for row in BEE_DECISION_MATRIX:
        matrix_rows.append(f"""
        <tr style="border-bottom:1px solid var(--border);">
          <td style="padding:0.55rem 0.6rem;font-size:0.8rem;color:var(--text);">{html.escape(row["use_case_fr"])}</td>
          <td style="padding:0.55rem 0.6rem;font-family:var(--font-mono);font-size:0.78rem;color:var(--cyan);font-weight:700;">{html.escape(row["primary_bee"])}</td>
          <td style="padding:0.55rem 0.6rem;font-family:var(--font-mono);font-size:0.75rem;color:var(--text-dim);">{html.escape(row["backup_bee"])}</td>
          <td style="padding:0.55rem 0.6rem;font-family:var(--font-mono);font-size:0.76rem;color:var(--green);">{html.escape(row["measured_speed"])}</td>
          <td style="padding:0.55rem 0.6rem;font-family:var(--font-mono);font-size:0.74rem;color:var(--amber);">{html.escape(row["gpu_residency"])}</td>
          <td style="padding:0.55rem 0.6rem;font-size:0.76rem;color:var(--text-dim);">{html.escape(row["why"])}</td>
        </tr>
        """)

    scenario_cards: list[str] = []
    for scen in LIFE_AS_A_BEE_SCENARIOS:
        is_100_gpu = scen.gpu_residency.startswith("100% GPU")
        residency_color = "var(--green)" if is_100_gpu else "var(--amber)"
        border_accent = (
            "rgba(192,132,252,0.4)"
            if scen.category == "moe_bee"
            else "rgba(0,238,252,0.35)"
            if scen.category == "specialist_bee"
            else "rgba(52,211,153,0.35)"
        )
        aliases_str = ", ".join(scen.aliases[:2])
        scenario_cards.append(f"""
        <div class="tier-card bee-scenario-card" data-bee-category="{html.escape(scen.category)}" style="border-color:{border_accent};justify-content:flex-start;gap:0.45rem;">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:0.4rem;flex-wrap:wrap;">
            <span class="tier-badge" style="margin:0;">{html.escape(scen.badge_label)}</span>
            <span style="font-family:var(--font-mono);font-size:0.7rem;color:var(--green);background:rgba(52,211,153,0.12);padding:0.12rem 0.45rem;border-radius:999px;">HTTP {scen.http_status} OK</span>
          </div>
          <div class="tier-name" style="margin-bottom:0.1rem;font-size:1rem;">
            <code style="color:var(--cyan);">{html.escape(scen.model_id)}</code>
          </div>
          <div style="font-size:0.78rem;color:#fff;font-weight:600;margin-bottom:0.2rem;">
            {html.escape(scen.scenario_title)}
          </div>
          <div style="font-family:var(--font-mono);font-size:0.72rem;color:var(--text-dim);margin-bottom:0.3rem;">
            {html.escape(scen.architecture)} &bull; Tag: <code>{html.escape(scen.underlying_tag)}</code>
          </div>

          <!-- Real Telemetry Bar -->
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.35rem;background:var(--bg-input);padding:0.45rem;border-radius:0.35rem;font-family:var(--font-mono);font-size:0.72rem;margin-bottom:0.35rem;">
            <div><span style="color:var(--text-dim);">Vitesse:</span><br><strong style="color:var(--green);">{scen.tokens_per_sec:.2f} tok/s</strong></div>
            <div><span style="color:var(--text-dim);">Latence chaude:</span><br><strong style="color:var(--cyan);">{scen.warm_latency_ms:.0f} ms</strong></div>
            <div><span style="color:var(--text-dim);">Cold load:</span><br><strong style="color:var(--amber);">{scen.cold_load_s:.1f} s</strong></div>
          </div>
          <div style="font-family:var(--font-mono);font-size:0.71rem;color:{residency_color};margin-bottom:0.35rem;">
            &#x1F5A5;&#xFE0F; {html.escape(scen.gpu_residency)} &bull; {scen.total_tokens} tokens ({scen.prompt_tokens}&rarr;{scen.completion_tokens}) &bull; {scen.suggested_cost_per_1k:.1f} HMT/1k
          </div>

          <!-- Usage Guide -->
          <div style="font-size:0.77rem;color:var(--text);line-height:1.4;margin-bottom:0.25rem;">
            <strong style="color:var(--amber);">&#x1F3AF; Pour quoi l'utiliser :</strong> {html.escape(scen.when_to_use_fr)}
          </div>
          <div style="font-size:0.74rem;color:var(--text-dim);line-height:1.35;margin-bottom:0.25rem;">
            <strong style="color:var(--red);">&#x26A0;&#xFE0F; Quand éviter :</strong> {html.escape(scen.when_not_to_use)}
          </div>
          <div style="font-size:0.74rem;color:var(--text-dim);line-height:1.35;margin-bottom:0.35rem;">
            <strong style="color:var(--cyan);">&#x1F527; Comment l'invoquer :</strong> {html.escape(scen.how_to_invoke)} (Aliases: <code>{html.escape(aliases_str)}</code>)
          </div>

          <!-- Verbatim Prompt & Response Excerpt -->
          <div style="background:#040711;border:1px solid var(--border);border-radius:0.35rem;padding:0.5rem;font-family:var(--font-mono);font-size:0.71rem;margin-bottom:0.45rem;">
            <div style="color:var(--amber);margin-bottom:0.25rem;"><strong>Prompt testé :</strong> {html.escape(scen.prompt)}</div>
            <div style="color:#38bdf8;"><strong>Réponse réelle :</strong> {html.escape(scen.response_excerpt)}</div>
          </div>

          <button
            type="button"
            class="btn btn-outline load-bee-scenario-btn"
            style="width:100%;justify-content:center;padding:0.4rem 0.6rem;font-size:0.75rem;margin-top:auto;border-color:var(--cyan);color:var(--cyan);"
            data-bee-model="{html.escape(scen.model_id)}"
            data-bee-prompt="{html.escape(scen.prompt)}"
            data-bee-temp="{scen.recommended_temperature}"
            data-bee-tokens="{scen.recommended_max_tokens}"
          >&#x26A1; Tester cette Abeille dans le Playground</button>
        </div>
        """)

    return f"""
    <!-- Life as a Bee Dashboard: Real End-User Scenarios & MoE Bee Usage Guide -->
    <section id="life-as-a-bee" class="card" style="margin-bottom:3rem;border-color:rgba(251,191,36,0.45);">
      <div class="card-header">
        <div class="card-title" style="color:var(--amber);">
          &#x1F41D; Life as a Bee Dashboard &bull; La Vie d'une Abeille MoE (15 Scénarios Réels &amp; Guide d'Usage)
        </div>
        <div style="display:flex;gap:0.5rem;flex-wrap:wrap;">
          <a href="/life-as-a-bee" class="pill" style="border-color:var(--amber);color:var(--amber);text-decoration:none;">&#x1F5D7; Vue Pleine Page (/life-as-a-bee)</a>
          <a href="/v1/hivemind/life-as-a-bee" class="pill" style="border-color:var(--cyan);color:var(--cyan);text-decoration:none;">&#x1F4CA; JSON Télémetrie (/v1/hivemind/life-as-a-bee)</a>
        </div>
      </div>

      <p style="color:var(--text-dim);font-size:0.9rem;margin-bottom:1.1rem;line-height:1.55;">
        Comment vit une abeille spécialisée ou Mixture-of-Experts (MoE) dans <strong>La Ruche HiveMind</strong>, quels sont ses résultats réels mesurés de bout en bout sur <code>POST /v1/chat/completions</code> (campagne <code>HM-S19-LIFE-AS-A-BEE-BENCHMARK</code>), et <strong>quelle abeille choisir pour chaque tâche</strong> ?
      </p>

      <!-- Top Summary KPIs -->
      <div class="tiers-grid" style="margin-bottom:1.5rem;">
        <div class="tier-card" style="border-color:rgba(52,211,153,0.4);">
          <div>
            <div class="tier-badge">CAMPAGNE END-USER RÉELLE</div>
            <div class="tier-name">15 / 15 Abeilles &bull; 100% HTTP 200</div>
            <div class="tier-vram">2 291 Tokens Générés &bull; 0 Erreur</div>
            <div class="tier-desc">Toutes les 15 abeilles et modèles fondationnels (Gemma 3, Qwen 3.5, Mistral, Crowd-v1, 4 MoE, 3 Spécialistes) ont été validés en conditions réelles.</div>
          </div>
          <div><span class="tier-tag">Moyenne: 25,64 tok/s</span></div>
        </div>
        <div class="tier-card" style="border-color:rgba(0,238,252,0.4);">
          <div>
            <div class="tier-badge">PALIER 100% GPU (1.5B–8B)</div>
            <div class="tier-name">16,5 à 71,9 tok/s (1,0s – 4,8s)</div>
            <div class="tier-vram">10 Modèles Logés 100% en VRAM</div>
            <div class="tier-desc"><code>crowd-v1</code> (71,9 tok/s), <code>coder-lite</code> (53,0 tok/s), <code>gemma-core</code> (37,0 tok/s), <code>mistral-core</code> (33,5 tok/s), <code>bee-rusty-sundials</code> (33,3 tok/s), <code>coder</code> (32,8 tok/s), <code>qwen35-core</code> (30,0 tok/s), <code>bee-stars</code> (27,9 tok/s).</div>
          </div>
          <div><span class="tier-tag">Read-Ahead Disque 16 Mo Actif</span></div>
        </div>
        <div class="tier-card" style="border-color:rgba(192,132,252,0.4);">
          <div>
            <div class="tier-badge">AVANTAGE SPARSE MoE 20B</div>
            <div class="tier-name">1,83x Plus Rapide que le 14B Dense</div>
            <div class="tier-vram">6,26 à 7,20 tok/s (2,4B–3,6B Actifs)</div>
            <div class="tier-desc">Même avec un partage GPU/CPU (64%/36% sur T4 16 Go), nos 4 abeilles MoE 20B répondent en ~15,3s contre 28,0s (3,93 tok/s) pour le modèle 14B dense.</div>
          </div>
          <div><span class="tier-tag">Top-2 Gating Parcimonieux</span></div>
        </div>
        <div class="tier-card" style="border-color:rgba(251,191,36,0.4);">
          <div>
            <div class="tier-badge">ÉPINGLAGE VRAM &amp; JEU</div>
            <div class="tier-name">0,0s Cold Start (bee-stars + coder)</div>
            <div class="tier-vram">Préemption Gamer &lt; 1,5s (1 133 ms)</div>
            <div class="tier-desc"><code>hivemind/bee-stars</code> (:8294) et <code>hivemind/coder</code> sont maintenus chauds en VRAM pour une réponse immédiate.</div>
          </div>
          <div><span class="tier-tag">Zéro Impact FPS Joueur</span></div>
        </div>
      </div>

      <!-- 24-Hour Life Cycle of a HiveMind Bee -->
      <h4 style="font-size:1.05rem;font-weight:700;color:#fff;margin-bottom:0.65rem;">
        &#x1F504; Les 4 Phases du Cycle de Vie d'une Abeille (24h dans La Ruche)
      </h4>
      <div class="tiers-grid" style="margin-bottom:1.75rem;">
        {"".join(lifecycle_cards)}
      </div>

      <!-- Quick Decision Matrix Table -->
      <h4 style="font-size:1.05rem;font-weight:700;color:#fff;margin-bottom:0.65rem;">
        &#x1F9ED; Matrice de Décision : Quelle Abeille Choisir et Pourquoi ?
      </h4>
      <div style="overflow-x:auto;margin-bottom:1.75rem;border:1px solid var(--border);border-radius:0.5rem;">
        <table style="width:100%;border-collapse:collapse;text-align:left;">
          <thead>
            <tr style="background:var(--bg-input);border-bottom:1px solid var(--border);font-family:var(--font-mono);font-size:0.74rem;color:var(--amber);">
              <th style="padding:0.6rem;">Besoin / Cas d'Usage</th>
              <th style="padding:0.6rem;">Abeille Recommandée</th>
              <th style="padding:0.6rem;">Alternative</th>
              <th style="padding:0.6rem;">Débit &amp; Latence Mesurés</th>
              <th style="padding:0.6rem;">Résidence GPU</th>
              <th style="padding:0.6rem;">Raison du Choix</th>
            </tr>
          </thead>
          <tbody>
            {"".join(matrix_rows)}
          </tbody>
        </table>
      </div>

      <!-- Category Filter & 15 Real Scenario Cards -->
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.75rem;margin-bottom:1rem;">
        <h4 style="font-size:1.05rem;font-weight:700;color:#fff;margin:0;">
          &#x1F9EA; Les 15 Scénarios Utilisateurs Réels &amp; Fiches d'Identité par Abeille
        </h4>
        <div class="pill-row" style="margin:0;">
          <button type="button" class="pill bee-filter-btn" data-bee-filter="all" style="color:var(--cyan);border-color:var(--cyan);">Toutes les Abeilles (15)</button>
          <button type="button" class="pill bee-filter-btn" data-bee-filter="moe_bee">&#x1F41D; Abeilles MoE 20B (4)</button>
          <button type="button" class="pill bee-filter-btn" data-bee-filter="specialist_bee">&#x1F52C; Abeilles Spécialistes (3)</button>
          <button type="button" class="pill bee-filter-btn" data-bee-filter="core_foundation">&#x1F30D; Modèles Core : Gemma / Qwen 3.5 / Mistral / Crowd (4)</button>
          <button type="button" class="pill bee-filter-btn" data-bee-filter="dense_coding">&#x1F4BB; Code Dense &amp; Auto (4)</button>
        </div>
      </div>

      <div class="tiers-grid" style="grid-template-columns:repeat(auto-fit, minmax(310px, 1fr));margin-bottom:0.5rem;">
        {"".join(scenario_cards)}
      </div>
    </section>
    """
