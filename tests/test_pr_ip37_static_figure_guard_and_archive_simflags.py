# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression & negative-control test suite for Iteration 37 (PR #706).

Covers:
1. Audit C2 (`src/hivemind/router/portal_page.py` & `src/hivemind/router/life_as_a_bee.py`):
   - Static-number guard on `portal_page.py:1521-1523` and `life_as_a_bee.py:2161`:
     replaces hardcoded `"4 Active Producers &bull; 7 Live Models"` and
     `"240 Reqs Stress Burst &bull; p50 139.0 ms"` with `shown_figure()` over `live_telemetry`.
2. Audit D (`image_moe_inference_gallery_benchmark.py`, `dynamic_diurnal_moe_engine.py`,
   `s35_03_elenchus_leanlib_sundials_r64.py`, `s35_05_unified_xdev_ecosystem_bridge.py`,
   `boinc/real_client/app.py`, and `scripts/e8_qlora_moe_round.py`):
   - Verifies `"simulated": True` / `simulated=True` / `"validator_fixture": True` flags on
     all Audit D simulation/fixture modules and verifies `scripts/e8_qlora_moe_round.py` exists.
"""

from __future__ import annotations

from pathlib import Path

from hivemind.boinc.real_client import app as boinc_real_app
from hivemind.core.image_moe_inference_gallery_benchmark import (
    run_intensive_image_moe_inference_stress_campaign,
)
from hivemind.router.life_as_a_bee import render_life_as_a_bee_section
from hivemind.router.opentrainer import DEFAULT_ROUND_SCRIPT_PATH
from hivemind.router.portal_page import render_portal_page
from hivemind.swarm.dynamic_diurnal_moe_engine import DiurnalMoETrainingScheduler
from hivemind.swarm.s35_03_elenchus_leanlib_sundials_r64 import (
    verify_robertson_sundials_tier_x,
)
from hivemind.swarm.s35_05_unified_xdev_ecosystem_bridge import (
    run_unified_4layer_ecosystem_pipeline,
)


def test_c2_portal_and_life_as_a_bee_use_shown_figure_over_live_telemetry() -> None:
    """Audit C2: `portal_page.py` and `life_as_a_bee.py` render `n/a` when telemetry is absent and actual numbers when provided."""
    html_empty = render_portal_page(None, "nonce-c2", live_telemetry=None)
    assert "4 Active Producers &bull; 7 Live Models" not in html_empty
    assert "240 Reqs Stress Burst &bull; p50 139.0 ms" not in html_empty
    assert "n/a Active Producers &bull; n/a Live Models" in html_empty
    assert "n/a Reqs (24h) &bull; p50 n/a ms" in html_empty

    live_payload = {
        "producers": {"live_hiveminders": 9},
        "models": {"live_models_count": 12},
        "usage_24h": {"total_requests": 318, "p50_latency_ms": 142.6},
    }
    html_live = render_portal_page(None, "nonce-c2-live", live_telemetry=live_payload)
    assert "9 Active Producers &bull; 12 Live Models" in html_live
    assert "318 Reqs (24h) &bull; p50 142.6 ms" in html_live

    bee_live = render_life_as_a_bee_section(live_telemetry=live_payload)
    assert "318 Reqs (24h) &bull; p50 142.6 ms &bull; Micro-VRAM &lt;= 1,410 MiB" in bee_live


def test_d_simulation_and_validator_fixture_flags(tmp_path: Path) -> None:
    """Audit D: simulation and validator fixture modules carry explicit `simulated` / `validator_fixture` flags."""
    assert DEFAULT_ROUND_SCRIPT_PATH.exists(), "scripts/e8_qlora_moe_round.py must exist"

    img_rep = run_intensive_image_moe_inference_stress_campaign(
        concurrent_requests_per_node=2,
        output_gallery_dir=tmp_path / "gal",
    )
    assert img_rep["simulated"] is True

    sched = DiurnalMoETrainingScheduler()
    quorum_plan = sched.plan_producer_quorum_moe_serving(active_records=[])
    assert quorum_plan["simulated"] is True

    r64_args = {
        "problem": "robertson",
        "rtol": 1e-9,
        "atol": 1e-12,
        "max_order": 5,
        "y_final": [0.2083340149701255, 0.0000008333360770, 0.7916651516937975],
    }
    r64_receipt = verify_robertson_sundials_tier_x(
        sent_arguments=r64_args,
        received_arguments=dict(r64_args),
    )
    assert r64_receipt.simulated is True

    u_receipt = run_unified_4layer_ecosystem_pipeline(xdev_root=tmp_path)
    assert u_receipt.simulated is True

    assert boinc_real_app.VALIDATOR_FIXTURE_ONLY is True
    wu_spec = {
        "wu_id": "wu-fixture-01",
        "seed": 7,
        "d_in": 8,
        "d_out": 8,
        "rank": 2,
        "batch": 2,
        "steps": 2,
        "lr": 0.05,
        "spot_check_interval": 1,
    }
    wu_res = boinc_real_app.run(wu_spec, "cpu", tmp_path / "ckpt.pt")
    assert wu_res["validator_fixture"] is True
