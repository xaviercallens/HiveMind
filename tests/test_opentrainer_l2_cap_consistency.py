"""One delta L2 cap on every OpenTrainer path (router offer and validation, orchestrator, AMCP round CLI).

The cap had three values (25 / 50 / 100). 25.0 was chosen from the 2,026 honest delta norms recorded in
artifacts/experiments/e8_real (max 15.4). This test fails if the defaults drift apart again.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hivemind.router import opentrainer as rot  # noqa: E402


def test_router_offer_and_validation_use_the_same_cap() -> None:
    assert rot.DEFAULT_MAX_DELTA_L2 == 25.0
    assert rot.default_active_task_offer()["max_delta_l2"] == rot.DEFAULT_MAX_DELTA_L2


def test_orchestrator_default_equals_the_router_cap() -> None:
    from hivemind.swarm.opentrainer_orchestrator import OpenTrainerConfig

    assert OpenTrainerConfig().max_delta_l2 == rot.DEFAULT_MAX_DELTA_L2


def test_amcp_round_cli_default_equals_the_router_cap() -> None:
    src = (ROOT / "scripts/e8_amcp_round.py").read_text()
    m = re.search(r'"--max-delta-l2", type=float, default=([0-9.]+)', src)
    assert m and float(m.group(1)) == rot.DEFAULT_MAX_DELTA_L2


def test_cap_leaves_headroom_over_recorded_honest_deltas() -> None:
    import json

    worst = 0.0
    for f in (ROOT / "artifacts/experiments/e8_real").glob("*.json"):
        def walk(o):
            nonlocal worst
            if isinstance(o, dict):
                for k, v in o.items():
                    if k == "delta_l2":
                        vals = v.values() if isinstance(v, dict) else [v]
                        worst = max([worst] + [float(x) for x in vals if isinstance(x, (int, float))])
                    else:
                        walk(v)
            elif isinstance(o, list):
                for x in o:
                    walk(x)
        try:
            walk(json.loads(f.read_text()))
        except ValueError:
            continue
    assert 0.0 < worst < rot.DEFAULT_MAX_DELTA_L2, f"largest recorded honest delta {worst} must stay under the cap"


def test_validation_rejects_just_above_and_accepts_at_the_cap() -> None:
    import pytest

    args = dict(run_id="E8-STAGE2-TINY-GRANITE-MOE-QLORA", round_index=1, eval_loss=1.0, tokens_trained=512,
                vram_free_mib=8000, vram_total_mib=16000)
    rot.validate_trainer_delta_submission(delta_l2=rot.DEFAULT_MAX_DELTA_L2, **args)
    with pytest.raises(ValueError, match="exceeds maximum bound"):
        rot.validate_trainer_delta_submission(delta_l2=rot.DEFAULT_MAX_DELTA_L2 + 0.01, **args)
