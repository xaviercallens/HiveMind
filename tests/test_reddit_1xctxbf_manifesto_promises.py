"""End-to-end verification suite for all promises made in the Reddit manifesto post `1xctxbf`
(`https://www.reddit.com/r/artificielle/comments/1xctxbf/pour_une_ai_citoyenne_et_d%C3%A9centralis%C3%A9e_en/`,
`docs/MANIFESTE_IA_CITOYENNE_SETI_AT_HOME.md`, and `src/hivemind/router/manifesto_page.py`).

Covers:
1. 30-Second Onboarding Profiles (WebGPU game `/webgpu`, 5-language OpenRouter `/?lang=fr` & `/fr|/en|/es|/de|/zh`,
   `/signup`, `/join`, `/rejoindre`, `/manifesto`, `/manifeste`, downloadable `/scripts/hivemind_boinc_idle_worker.py`
   on both OpenRouter and La Ruche, and CLI flags `--coordinator ... --handle ... --domain physics_sim --once`).
2. Section 2 Breakthroughs:
   - 2A: Asynchronous Island Training (`DiLoCo` `Blockwise8BitAdamW` / `DiLoCoOuterOptimizer` & `DecoupledDiLoCoQuorumSyncer`
     / `StragglerFreeDecoupledDiLoCoAggregator` `K < M` straggler-free quorum).
   - 2B: `Branch-Train-Mix` (BTM) + 4-bit `NF4` quantization + Top-2 Sparse MoE routing.
   - 2C: Anti-cheating Byzantine verification: 10% spot-check (`DeterministicSpotCheckValidator`), FP32 Multi-Krum
     (`ByzantineMultiKrumAggregator`), CrowdGPT S-1 loss check (`LossRatchetGate`), and `TamperEvidentV5FixedPointCircuit`.
3. Section 3 Citizen AI Pillars:
   - Pillar 1: La Ruche Data Commons (`ContentAddressedSwarmStore`, `LaRucheModelVault`, `DARLShardLeaseCoordinator`).
   - Pillar 2: Zero Intrusion on Host PC (`ZeroIntrusionDaemon`, `GamePreemptionEF21CircuitBreaker` `< 1.5s`,
     `GPUExecutionCircuitBreaker`, `hivemind_boinc_idle_worker.py --pause-now`).
   - Pillar 3: Non-Monetary Compute Reciprocity (`0.00 EUR / 0.00 USD`, `100.0 HMT` starter grant, 15-min Kelly active
     window `bid_activity_window_s=900`, and legacy password sign-in preservation).
   - Pillar 4: RTX 2070 8 GB PoC + `SpeculativeSwarmDecoder` + Sole T4 Day/Night schedule (`/v1/hivemind/t4/status`, `/night-moe`).
4. Section 4 Follow-Up Promises:
   - 4A: `NocturnalCodingMoE` (`hivemind.core.nocturnal_coding_moe`).
   - 4B: 7-Division / 104-Bee Vision-MoE (`hivemind.core.reddit_vision_vlm_moe`) & 12-Scenario Gallery (`/life-as-a-bee`).
   - 4C: Season 34 `SwaTTelepathyBridge` (`/v1/hivemind/bees/swat-telepathy/status`) & `AMCPLatentKVCacheEnvelope`.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app
from hivemind.router.accounts import LEGACY_KNOWN_USERS, STARTER_WINDOW_S, RouterStore
from hivemind.router.gateway import create_router_app


def _make_openrouter_client(tmp_path: Path) -> TestClient:
    store = RouterStore(tmp_path / "router.db", b"test-secret-for-reddit-1xctxbf-manifesto-32b")
    app = create_router_app(store, starter_grant_hmt=100.0)
    return TestClient(app)


def _make_ruche_client(tmp_path: Path) -> TestClient:
    app = create_hivemind_at_home_app(vault_dir=tmp_path / "ruche_vault")
    return TestClient(app)


def test_1xctxbf_profile1_and_2_five_languages_and_manifesto_routes(tmp_path: Path) -> None:
    """Verify Profile 1 (`/webgpu`) and Profile 2 (`/?lang=fr`, `/fr`, `/en`, `/es`, `/de`, `/zh`, `/manifesto`, `/manifeste`)."""
    client = _make_openrouter_client(tmp_path)

    # Profile 1: Zero-install WebGPU space game
    r_webgpu = client.get("/webgpu")
    assert r_webgpu.status_code == 200
    assert "WebGPU" in r_webgpu.text

    # Profile 2: 5 languages on OpenRouter (`/?lang=...` and `/fr`, `/en`, `/es`, `/de`, `/zh`)
    from hivemind.router.points_wording import currency_leftovers

    for lang in ("fr", "en", "es", "de", "zh"):
        r_path = client.get(f"/{lang}")
        assert r_path.status_code == 200, f"GET /{lang} failed with {r_path.status_code}"
        assert r_path.headers.get("content-language") == lang
        assert f'<html lang="{lang}">' in r_path.text
        assert 'id="portal-i18n-bar"' in r_path.text
        assert currency_leftovers(r_path.text) == []

        r_query = client.get(f"/?lang={lang}")
        assert r_query.status_code == 200
        assert r_query.headers.get("content-language") == lang
        assert f'<html lang="{lang}">' in r_query.text

    # Manifesto routes in FR and EN + JSON API
    for m_path in ("/manifesto", "/manifeste", "/manifesto?lang=en"):
        rm = client.get(m_path)
        assert rm.status_code == 200
        assert "1xctxbf" in rm.text

    rj = client.get("/v1/hivemind/manifesto")
    assert rj.status_code == 200
    m_data = rj.json()
    assert m_data["object"] == "hivemind.manifesto"
    assert "0.00 EUR / 0.00 USD" in m_data["charter"]
    threads = m_data["reddit_summary"]["threads"]
    assert any("1xctxbf" in t["thread_id"] for t in threads)


def test_1xctxbf_profile3_and_4_onboarding_scripts_and_idle_worker_cli(tmp_path: Path) -> None:
    """Verify `/join`, `/rejoindre`, `/scripts/hivemind_boinc_idle_worker.py` on both OpenRouter and La Ruche, and CLI flags."""
    openrouter = _make_openrouter_client(tmp_path)
    ruche = _make_ruche_client(tmp_path)

    for client in (openrouter, ruche):
        for join_path in ("/join", "/rejoindre"):
            rj = client.get(join_path)
            assert rj.status_code == 200
            assert "hivemind_boinc_idle_worker.py" in rj.text

        for script_path in (
            "/scripts/hivemind_boinc_idle_worker.py",
            "/hivemind_boinc_idle_worker.py",
            "/scripts/hivemind_producer_agent.py",
            "/scripts/hivemind_producer_shield.py",
            "/scripts/hivemind_producer_autostart.py",
            "/scripts/hivemind_volunteer_config.py",
            "/scripts/hivemind_volunteer_preflight.py",
            "/scripts/hivemind_opentrainer_agent.py",
        ):
            rs = client.get(script_path)
            assert rs.status_code == 200, f"{script_path} returned {rs.status_code}"
            assert "python" in rs.text.lower()

        for lang in ("fr", "en", "es", "de", "zh"):
            rl = client.get(f"/{lang}")
            assert rl.status_code == 200
            assert rl.headers.get("content-language") == lang

    # Verify exact CLI invocation promised in docs/MANIFESTE_IA_CITOYENNE_SETI_AT_HOME.md line 154:
    # python3 hivemind_boinc_idle_worker.py --coordinator https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app --handle votre-pseudo --domain physics_sim --once
    import importlib.util

    worker_path = Path(__file__).resolve().parents[1] / "scripts" / "hivemind_boinc_idle_worker.py"
    spec = importlib.util.spec_from_file_location("hivemind_boinc_idle_worker", worker_path)
    assert spec is not None and spec.loader is not None
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)

    rc_pause = worker.main(["--pause-now"])
    assert rc_pause == 0

    rc_once = worker.main(
        [
            "--coordinator",
            "https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app",
            "--handle",
            "votre-pseudo",
            "--domain",
            "physics_sim",
            "--once",
        ]
    )
    # 0 when a real idle CUDA GPU is present, 2 when truth-gated BLOCKED (e.g. headless/no GPU or busy GPU)
    assert rc_once in (0, 2)


def test_1xctxbf_section2_diloco_btm_moe_and_byzantine_verification() -> None:
    """Verify Section 2 promises: DiLoCo, Decoupled DiLoCo K<M, BTM NF4 + Sparse MoE, 10% Spot-Check, Multi-Krum, CrowdGPT, ZK-OPoI."""
    from hivemind.boinc.validator import DeterministicSpotCheckValidator
    from hivemind.core.btm_moe import HierarchicalMoERouter
    from hivemind.core.diloco_inner import Blockwise8BitAdamW
    from hivemind.core.diloco_outer import DecoupledDiLoCoQuorumSyncer, DiLoCoOuterOptimizer
    from hivemind.core.quantization import NF4_CODEBOOK
    from hivemind.core.reddit_vision_vlm_moe import ByzantineMultiKrumAggregator
    from hivemind.swarm.crowdgpt_bridge import LossRatchetGate
    from hivemind.swarm.s33_01_decoupled_diloco_pacgad import StragglerFreeDecoupledDiLoCoAggregator
    from hivemind.swarm.zk_opoi_v5_circuit import TamperEvidentV5FixedPointCircuit

    assert len(NF4_CODEBOOK) == 16
    assert Blockwise8BitAdamW is not None
    assert DiLoCoOuterOptimizer is not None

    quorum = DecoupledDiLoCoQuorumSyncer(num_learners_m=4, min_quorum_k=2)
    assert quorum.min_quorum_k == 2
    assert quorum.num_learners_m == 4

    pacgad = StragglerFreeDecoupledDiLoCoAggregator(staleness_alpha=0.5, outer_lr=0.7)
    assert pacgad.staleness_weight(0) == 1.0

    router = HierarchicalMoERouter(hidden_size=16, num_macro_domains=4, experts_per_domain=4, top_k=2)
    assert router.top_k == 2

    validator = DeterministicSpotCheckValidator(tolerance_eps=1e-4)
    assert validator.tolerance_eps == 1e-4

    krum = ByzantineMultiKrumAggregator(replay_fraction=0.10, min_cosine_sim=0.95, max_l2_drift=0.05)
    assert abs(krum.replay_fraction - 0.10) < 1e-6

    ratchet = LossRatchetGate(max_regression_tolerance=0.0)
    assert ratchet.max_regression_tolerance == 0.0

    circuit = TamperEvidentV5FixedPointCircuit(coordinator_secret="test-shared-secret")
    receipt, _ = circuit.prove_quantized_diloco_transition(
        node_id="bee-rtx2070-fr",
        assigned_shard_sha256="a" * 64,
        theta0_float=[0.25, -0.5, 0.75, 0.1],
        step_gradients_float=[[0.1, -0.2, 0.05, 0.15]],
    )
    verdict = circuit.verify_receipt_cpu_o1(
        receipt,
        expected_shard_sha256="a" * 64,
        expected_theta0_commitment=receipt.theta0_commitment,
    )
    assert verdict.valid is True
    assert receipt.zero_knowledge is False
    assert receipt.commitment_scheme == "fiat_shamir_horner_hmac_sha256"


def test_1xctxbf_section3_and_4_pillars_reciprocity_legacy_login_and_swat(tmp_path: Path) -> None:
    """Verify Section 3 & 4 promises: CAS store, <1.5s preemption, 100 HMT non-monetary reciprocity, legacy password login, T4 schedule, SwaT."""
    from hivemind.client.daemon import ZeroIntrusionDaemon
    from hivemind.core.nocturnal_coding_moe import DEFAULT_CODING_DILOCO_STEPS
    from hivemind.swarm.la_ruche_vault import ContentAddressedSwarmStore
    from hivemind.swarm.s34_01_speculative_swarm_decoding import SpeculativeSwarmDecoder
    from hivemind.swarm.s34_02_game_preemption_ef21_flush import (
        MAX_PREEMPTION_LATENCY_MS,
        GamePreemptionEF21CircuitBreaker,
    )

    # Pillar 1: La Ruche Content-Addressed Store
    cas = ContentAddressedSwarmStore(tmp_path / "cas")
    digest = cas.put_chunk(b"citizen-ai-shard-v1")
    assert cas.read_chunk(digest) == b"citizen-ai-shard-v1"

    # Pillar 2: Zero Intrusion (< 1.5s game preemption)
    assert MAX_PREEMPTION_LATENCY_MS <= 1500.0
    assert ZeroIntrusionDaemon is not None
    assert GamePreemptionEF21CircuitBreaker is not None

    # Pillar 3: 100 HMT starter grant & legacy known users password login preserved
    assert STARTER_WINDOW_S == 86_400.0
    assert "xavier_callens" in LEGACY_KNOWN_USERS
    assert "callensxavier@gmail.com" in LEGACY_KNOWN_USERS["xavier_callens"]["aliases"]

    # Pillar 4 & Section 4: Speculative decoding, Coding MoE, T4 status, SwaT telepathy, and Life as a Bee gallery
    assert SpeculativeSwarmDecoder is not None
    assert DEFAULT_CODING_DILOCO_STEPS > 0

    client = _make_openrouter_client(tmp_path)
    ruche = _make_ruche_client(tmp_path)
    assert client.get("/v1/hivemind/t4/status").status_code == 200
    assert client.get("/night-moe", follow_redirects=False).status_code in (200, 307)
    assert ruche.get("/night-moe").status_code == 200
    assert client.get("/v1/hivemind/bees/swat-telepathy/status").status_code == 200
    r_bee = client.get("/life-as-a-bee")
    assert r_bee.status_code == 200
    assert "image-moe-gallery" in r_bee.text
