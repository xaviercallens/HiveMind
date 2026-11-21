# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Option 2 (owner, 2026-11-17): training rows through the gateway with the dedicated Hugging Face account."""
from __future__ import annotations

import pytest

from hivemind.router.training_data_proxy import TRAINING_DATASETS, TrainingDataError, TrainingRowsProxy


def test_allowlist_is_the_plan_plus_local_extras_minus_benchmarks() -> None:
    from hivemind.swarm.crowdgpt_moe_opendiloco import load_crowdgpt_moe_dataset_plan
    from hivemind.swarm.crowdgpt_real_shards import LOCAL_DATASET_FILES

    plan = load_crowdgpt_moe_dataset_plan()
    banned = set(plan.get("decontamination_benchmarks", []))
    ids = {s["dataset_id"] for e in plan["experts"] for s in e.get("accepted_sources", [])}
    assert TRAINING_DATASETS == (ids | set(LOCAL_DATASET_FILES)) - banned
    assert not TRAINING_DATASETS & banned


def test_proxy_validates_caches_and_uses_the_dedicated_token() -> None:
    calls = []

    def fetch(url, headers, timeout):
        calls.append((url, dict(headers)))
        return {"rows": [{"row": {"text": "hello"}}, {"bad": 1}], "num_rows_total": 10}

    clock = [0.0]
    proxy = TrainingRowsProxy("hf_dedicated_token", fetch=fetch, ttl_s=60, clock=lambda: clock[0])
    out = proxy.rows("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", 40, 12)
    assert out["rows"] == [{"row": {"text": "hello"}}] and out["upstream_authenticated"] is True and out["cached"] is False
    assert "Bearer hf_dedicated_token" == calls[0][1]["Authorization"] and "datasets-server.huggingface.co/rows" in calls[0][0]
    assert "hf_dedicated_token" not in str(out)
    assert proxy.rows("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", 40, 12)["cached"] is True and len(calls) == 1
    clock[0] = 61.0
    proxy.rows("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", 40, 12)
    assert len(calls) == 2
    for bad in (("allenai/ai2_arc", "default", "train", 0, 1, 404), ("HuggingFaceFW/fineweb-edu", "../x", "train", 0, 1, 400),
                ("HuggingFaceFW/fineweb-edu", "default", "train", 0, 101, 400), ("HuggingFaceFW/fineweb-edu", "default", "train", -1, 1, 400)):
        with pytest.raises(TrainingDataError) as exc:
            proxy.rows(*bad[:5])
        assert exc.value.status == bad[5]
    anon = TrainingRowsProxy("", fetch=fetch)
    anon.rows("sahil2801/CodeAlpaca-20k", "default", "train", 0, 5)
    assert "Authorization" not in calls[-1][1] and anon.authenticated is False


def test_upstream_failure_is_a_502_without_leaking(tmp_path) -> None:
    def boom(url, headers, timeout):
        raise OSError(f"failed with {headers}")

    proxy = TrainingRowsProxy("hf_secret", fetch=boom)
    with pytest.raises(TrainingDataError) as exc:
        proxy.rows("HuggingFaceFW/fineweb-edu", "default", "train", 0, 1)
    assert exc.value.status == 502 and "hf_secret" not in str(exc.value)


def test_gateway_endpoint_needs_a_key_and_serves_rows(tmp_path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    import hivemind.router.training_data_proxy as tdp
    from hivemind.router.accounts import RouterStore
    from hivemind.router.gateway import create_router_app

    store = RouterStore(tmp_path / "r.db", b"k" * 32)
    _, key = store.register("vol-1", "consumer")
    proxy = tdp.TrainingRowsProxy("", fetch=lambda url, headers, timeout: {"rows": [{"row": {"text": "row"}}]})
    monkeypatch.setattr(tdp, "TrainingRowsProxy", lambda *a, **k: proxy)
    client = TestClient(create_router_app(store, allow_private_producers=True))
    url = "/v1/hivemind/training-data/rows?dataset=HuggingFaceFW/fineweb-edu&config=sample-10BT&offset=0&length=2"
    assert client.get(url).status_code == 401
    r = client.get(url, headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200 and r.json()["rows"] == [{"row": {"text": "row"}}]
    bad = client.get("/v1/hivemind/training-data/rows?dataset=openai/openai_humaneval", headers={"Authorization": f"Bearer {key}"})
    assert bad.status_code == 404
