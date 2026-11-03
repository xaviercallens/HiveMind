"""Tests for HM-S16-06: heartbeat residency fields, loaded-model routing, explicit ``model_loading`` status.

Producers are replaced by httpx.MockTransport (see test_router_gateway); nothing here touches a real model server or GPU.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from hivemind.router.accounts import SYS_ESCROW
from tests.test_router_gateway import COST, FakeProducer, World, _build_world

MODEL = "qwen"


@dataclass
class TimeoutProducer(FakeProducer):
    """A backend that never answers in time."""

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        raise httpx.ReadTimeout("slow", request=request)


@pytest.fixture()
def w() -> World:
    return _build_world(warmup_unloaded=False)


def beat(
    world: World,
    key: str,
    host: str,
    *,
    loaded: bool | None = None,
    vram_mib: int | None = None,
    cold: float | None = None,
    gpu: int | None = None,
    capacity: float = 100.0,
    extra_models: dict[str, dict[str, Any]] | None = None,
) -> httpx.Response:
    entry: dict[str, Any] = {"id": MODEL, "local_name": "q"}
    for name, value in (("loaded", loaded), ("vram_mib", vram_mib), ("cold_start_s", cold)):
        if value is not None:
            entry[name] = value
    models = [entry] + [{"id": mid, "local_name": mid, **fields} for mid, fields in (extra_models or {}).items()]
    body: dict[str, Any] = {"endpoint_url": f"https://{host}", "models": models, "cost_per_1k": COST, "capacity_1k": capacity, "execution_class": "t4"}
    if gpu is not None:
        body["gpu_vram_mib"] = gpu
    return world.app_client.post("/v1/hivemind/producer/heartbeat", headers={"Authorization": f"Bearer {key}"}, json=body)


def add_producer(world: World, host: str, agent: FakeProducer | None = None, **fields: Any) -> tuple[str, str, FakeProducer]:
    account_id, key = world.register("producer")
    fake = agent if agent is not None else FakeProducer()
    world.producers[host] = fake
    response = beat(world, key, host, **fields)
    assert response.status_code == 200, response.text
    return account_id, key, fake


def no_charge(world: World, consumer_id: str, funds: float) -> None:
    assert world.store.balance(consumer_id) == funds
    assert world.store.balance(SYS_ESCROW) == 0.0
    assert world.store.active_hold_count(consumer_id) == 0
    assert world.store.recent_usage(consumer_id) == []


# ------------------------------------------------------------------------------------------ heartbeat schema
def test_heartbeat_accepts_and_stores_the_residency_fields(w: World) -> None:
    add_producer(w, "a.test", loaded=True, vram_mib=5200, cold=12.5, gpu=15360)
    record = w.store.active_producers(MODEL, w.clock(), 120)[0]
    assert record.model_state(MODEL) is True
    assert record.cold_start_s(MODEL) == 12.5
    assert record.residency is not None and record.residency["gpu_vram_mib"] == 15360


def test_heartbeat_ack_reports_what_is_loaded(w: World) -> None:
    _, key = w.register("producer")
    body = beat(w, key, "a.test", loaded=True, vram_mib=100, gpu=1000).json()
    assert body["residency"]["loaded"] == [MODEL]


@pytest.mark.parametrize(
    "fields",
    [{"vram_mib": -1}, {"cold_start_s": -0.5}, {"gpu_vram_mib": -4}, {"vram_mib": 10**12}, {"loaded": "yes"}, {"loaded": 1}, {"vram_mib": 1.5}],
)
def test_heartbeat_rejects_bad_residency_values(w: World, fields: dict[str, Any]) -> None:
    _, key = w.register("producer")
    body: dict[str, Any] = {"endpoint_url": "https://a.test", "models": [{"id": MODEL, "local_name": "q"}], "cost_per_1k": COST, "capacity_1k": 10.0}
    for name, value in fields.items():
        if name == "gpu_vram_mib":
            body[name] = value
        else:
            body["models"][0][name] = value
    response = w.app_client.post("/v1/hivemind/producer/heartbeat", headers={"Authorization": f"Bearer {key}"}, json=body)
    assert response.status_code == 400, response.text


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_heartbeat_rejects_non_finite_cold_start(w: World, literal: str) -> None:
    _, key = w.register("producer")
    raw = '{"endpoint_url":"https://a.test","models":[{"id":"qwen","local_name":"q","cold_start_s":%s}],"cost_per_1k":0.4,"capacity_1k":10}' % literal
    response = w.app_client.post(
        "/v1/hivemind/producer/heartbeat", headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, content=raw
    )
    assert response.status_code == 400, response.text


def test_cold_start_is_clamped_to_ten_minutes(w: World) -> None:
    add_producer(w, "a.test", loaded=False, cold=99999.0)
    assert w.store.active_producers(MODEL, w.clock(), 120)[0].cold_start_s(MODEL) == 600.0


# ------------------------------------------------------------------------------------------ routing preference
def test_loaded_producer_is_preferred_over_unknown_and_not_loaded(w: World) -> None:
    _, ckey = w.consumer(50.0)
    _, _, cold = add_producer(w, "cold.test", loaded=False, cold=5.0)
    _, _, unknown = add_producer(w, "unknown.test")
    _, _, hot = add_producer(w, "hot.test", loaded=True)
    for _ in range(6):
        assert w.chat(ckey, f"hivemind/{MODEL}").status_code == 200
    assert len(hot.calls) == 6 and not unknown.calls and not cold.calls


def test_unknown_ranks_after_loaded_but_before_not_loaded(w: World) -> None:
    _, ckey = w.consumer(50.0)
    _, _, cold = add_producer(w, "cold.test", loaded=False, cold=5.0)
    _, _, unknown = add_producer(w, "unknown.test")
    _, _, hot = add_producer(w, "hot.test", loaded=True, agent=FakeProducer(explode=True))
    for _ in range(4):
        assert w.chat(ckey, f"hivemind/{MODEL}").status_code == 200  # hot fails first, the fallback is the unknown producer
    assert len(hot.calls) == 4 and len(unknown.calls) == 4 and not cold.calls


# ------------------------------------------------------------------------------------------ model_loading
def test_all_known_not_loaded_above_budget_is_503_model_loading_without_charge(w: World) -> None:
    cid, ckey = w.consumer(10.0)
    _, _, fake = add_producer(w, "cold.test", loaded=False, cold=90.0)
    response = w.chat(ckey, f"hivemind/{MODEL}")
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "model_loading" and "90" in error["message"]
    assert response.headers["Retry-After"] == "90"
    assert not fake.calls  # nothing forwarded blindly
    no_charge(w, cid, 10.0)


def test_retry_after_rounds_up_and_uses_the_shortest_cold_start(w: World) -> None:
    _, ckey = w.consumer(10.0)
    add_producer(w, "a.test", loaded=False, cold=200.0)
    add_producer(w, "b.test", loaded=False, cold=30.2)
    assert w.chat(ckey, f"hivemind/{MODEL}").headers["Retry-After"] == "31"


def test_missing_cold_start_estimate_assumes_a_slow_load(w: World) -> None:
    _, ckey = w.consumer(10.0)
    add_producer(w, "a.test", loaded=False)
    response = w.chat(ckey, f"hivemind/{MODEL}")
    assert response.status_code == 503 and response.headers["Retry-After"] == "60"


def test_short_cold_start_below_the_budget_is_forwarded_and_charged(w: World) -> None:
    cid, ckey = w.consumer(10.0)
    _, _, fake = add_producer(w, "cold.test", loaded=False, cold=5.0)
    response = w.chat(ckey, f"hivemind/{MODEL}")
    assert response.status_code == 200 and len(fake.calls) == 1
    assert w.store.balance(cid) < 10.0


def test_budget_is_a_constructor_parameter() -> None:
    world = _build_world(cold_start_wait_budget_s=100.0, warmup_unloaded=False)
    _, ckey = world.consumer(10.0)
    _, _, fake = add_producer(world, "cold.test", loaded=False, cold=90.0)
    assert world.chat(ckey, f"hivemind/{MODEL}").status_code == 200 and len(fake.calls) == 1


def test_one_loaded_producer_means_the_request_is_served_not_deferred(w: World) -> None:
    _, ckey = w.consumer(10.0)
    add_producer(w, "cold.test", loaded=False, cold=300.0)
    _, _, hot = add_producer(w, "hot.test", loaded=True)
    assert w.chat(ckey, f"hivemind/{MODEL}").status_code == 200 and len(hot.calls) == 1


def test_unloaded_model_request_starts_a_background_warmup() -> None:
    world = _build_world(warmup_unloaded=True)
    _, ckey = world.consumer(10.0)
    _, _, fake = add_producer(world, "cold.test", loaded=False, cold=90.0)
    assert world.chat(ckey, f"hivemind/{MODEL}").status_code == 503
    deadline = time.monotonic() + 3.0
    while not fake.calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(fake.calls) == 1
    sent = json.loads(fake.calls[0].content)
    assert sent["max_tokens"] == 1 and sent["model"] == "q"
    # a second request inside the interval does not start another load
    assert world.chat(ckey, f"hivemind/{MODEL}").status_code == 503
    time.sleep(0.1)
    assert len(fake.calls) == 1


# ------------------------------------------------------------------------------------------ upstream timeouts
def test_timeout_on_a_known_not_loaded_model_is_model_loading(w: World) -> None:
    cid, ckey = w.consumer(10.0)
    add_producer(w, "slow.test", loaded=False, cold=5.0, agent=TimeoutProducer())  # short cold start: forwarded, then times out
    response = w.chat(ckey, f"hivemind/{MODEL}")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_loading" and response.headers["Retry-After"] == "5"
    no_charge(w, cid, 10.0)


def test_timeout_on_an_unknown_residency_producer_is_a_plain_504_without_a_cause(w: World) -> None:
    cid, ckey = w.consumer(10.0)
    add_producer(w, "slow.test", agent=TimeoutProducer())
    response = w.chat(ckey, f"hivemind/{MODEL}")
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "upstream_timeout" and "loading" not in response.json()["error"]["message"]
    assert "Retry-After" not in response.headers
    no_charge(w, cid, 10.0)


def test_timeout_on_a_loaded_model_is_a_clear_504(w: World) -> None:
    cid, ckey = w.consumer(10.0)
    add_producer(w, "slow.test", loaded=True, agent=TimeoutProducer())
    response = w.chat(ckey, f"hivemind/{MODEL}")
    assert response.status_code == 504 and response.json()["error"]["code"] == "upstream_timeout"
    no_charge(w, cid, 10.0)


def test_streaming_timeout_distinguishes_known_not_loaded_from_unknown(w: World) -> None:
    cid, ckey = w.consumer(10.0)
    add_producer(w, "slow.test", agent=TimeoutProducer())
    response = w.chat(ckey, f"hivemind/{MODEL}", stream=True)
    assert response.status_code == 504 and response.json()["error"]["code"] == "upstream_timeout"
    no_charge(w, cid, 10.0)
    world = _build_world(warmup_unloaded=False)
    cid2, ckey2 = world.consumer(10.0)
    add_producer(world, "slow2.test", loaded=False, cold=5.0, agent=TimeoutProducer())
    response = world.chat(ckey2, f"hivemind/{MODEL}", stream=True)
    assert response.status_code == 503 and response.json()["error"]["code"] == "model_loading"
    no_charge(world, cid2, 10.0)


def test_a_plain_connection_failure_stays_502(w: World) -> None:
    _, ckey = w.consumer(10.0)
    add_producer(w, "down.test", loaded=True, agent=FakeProducer(explode=True))
    response = w.chat(ckey, f"hivemind/{MODEL}")
    assert response.status_code == 502 and response.json()["error"]["code"] == "upstream_failure"


# ------------------------------------------------------------------------------------------ declared capacity vs the card
def test_loaded_models_that_exceed_the_card_are_clamped_and_explained(w: World) -> None:
    _, key = w.register("producer")
    response = beat(
        w, key, "a.test", loaded=True, vram_mib=4800, gpu=10000, capacity=100.0,
        extra_models={"coder": {"loaded": True, "vram_mib": 5300}},
    )
    assert response.status_code == 200
    ack = response.json()["residency"]
    assert ack["loaded"] == [MODEL] and ack["effective_capacity_1k"] == 50.0 and "coder" in ack["capacity_note"]
    now = w.clock()
    record = w.store.active_producers(MODEL, now, 120)[0]
    assert record.capacity_1k == 50.0
    assert record.model_state(MODEL) is True and record.model_state("coder") is False
    assert record.residency is not None and "exceed" in record.residency["capacity_note"]


def test_loaded_models_that_fit_keep_full_capacity(w: World) -> None:
    _, key = w.register("producer")
    response = beat(w, key, "a.test", loaded=True, vram_mib=4800, gpu=15360, extra_models={"coder": {"loaded": True, "vram_mib": 5200}})
    ack = response.json()["residency"]
    assert ack["effective_capacity_1k"] == 100.0 and ack["capacity_note"] is None and ack["loaded"] == ["coder", MODEL]


def test_the_operator_capacity_clamp_is_unchanged_by_residency() -> None:
    world = _build_world(max_producer_capacity_1k=10.0, warmup_unloaded=False)
    _, key = world.register("producer")
    assert beat(world, key, "a.test", loaded=True, vram_mib=100, gpu=1000, capacity=500.0).status_code == 200
    _, ckey = world.consumer(10.0)
    world.producers["a.test"] = FakeProducer()
    assert world.chat(ckey, f"hivemind/{MODEL}").status_code == 200
    assert world.store.active_producers(MODEL, world.clock(), 120)[0].capacity_1k == 500.0  # stored as declared; the clamp is applied when pricing


# ------------------------------------------------------------------------------------------ public surface and compatibility
def test_models_endpoint_exposes_only_an_aggregate_loaded_flag(w: World) -> None:
    add_producer(w, "hot.test", loaded=True)
    data = {item["id"]: item for item in w.app_client.get("/v1/models").json()["data"]}
    assert data[f"hivemind/{MODEL}"]["loaded"] is True
    assert "loaded" not in data["hivemind/auto"]
    assert "hot.test" not in json.dumps(data)


def test_old_heartbeat_without_residency_behaves_as_before(w: World) -> None:
    cid, ckey = w.consumer(10.0)
    _, key, fake = w.producer("old.test", {MODEL: "q"})
    body = w.beat(key, "old.test", {MODEL: "q"}, COST, 100.0).json()
    assert "residency" not in body
    record = w.store.active_producers(MODEL, w.clock(), 120)[0]
    assert record.residency is None and record.model_state(MODEL) is None and record.cold_start_s(MODEL) is None
    assert w.chat(ckey, f"hivemind/{MODEL}").status_code == 200 and len(fake.calls) == 1
    data = {item["id"]: item for item in w.app_client.get("/v1/models").json()["data"]}
    assert "loaded" not in data[f"hivemind/{MODEL}"]
    assert w.store.balance(cid) < 10.0


def test_a_later_heartbeat_without_residency_clears_the_report(w: World) -> None:
    _, key, _ = add_producer(w, "a.test", loaded=False, cold=300.0)
    assert w.beat(key, "a.test", {MODEL: "q"}, COST, 100.0).status_code == 200
    assert w.store.active_producers(MODEL, w.clock(), 120)[0].model_state(MODEL) is None


def test_residency_survives_a_reopened_database_and_old_tables_gain_the_column(tmp_path: Any) -> None:
    import sqlite3

    from hivemind.router import RouterStore

    path = tmp_path / "r.sqlite"
    store = RouterStore(path, b"g" * 32)
    account_id, _ = store.register("p@producer.test", "producer")
    store.upsert_producer(
        account_id, "https://a.test", [{"id": MODEL, "local_name": "q"}], 0.4, 10.0, now_utc=1000.0,
        residency={"gpu_vram_mib": 100, "capacity_note": None, "models": {MODEL: {"loaded": True, "vram_mib": 5, "cold_start_s": None}}},
    )
    store.close()
    reopened = RouterStore(path, b"g" * 32)
    assert reopened.active_producers(MODEL, 1001.0, 120)[0].model_state(MODEL) is True
    assert reopened.residency_summary(1001.0, 120) == {MODEL: True}
    reopened.close()
    # a database from before this change has no residency_json column: it is added in place and old rows read as "unknown"
    legacy = sqlite3.connect(path)
    legacy.execute("ALTER TABLE producer_endpoints DROP COLUMN residency_json")
    legacy.commit()
    legacy.close()
    upgraded = RouterStore(path, b"g" * 32)
    record = upgraded.active_producers(MODEL, 1001.0, 120)[0]
    assert record.residency is None and record.model_state(MODEL) is None


# ------------------------------------------------------------------------------------------ warm-up never evicts
def _wait_calls(fake: FakeProducer, n: int) -> None:
    deadline = time.monotonic() + 3.0
    while len(fake.calls) < n and time.monotonic() < deadline:
        time.sleep(0.01)


def test_no_warmup_when_the_target_has_another_model_loaded() -> None:
    world = _build_world(warmup_unloaded=True)
    _, ckey = world.consumer(10.0)
    _, _, fake = add_producer(world, "busy.test", loaded=False, cold=90.0, extra_models={"coder": {"loaded": True, "vram_mib": 5200}})
    response = world.chat(ckey, f"hivemind/{MODEL}")
    assert response.status_code == 503 and response.json()["error"]["code"] == "model_loading"
    time.sleep(0.2)
    assert not fake.calls  # Ollama would have evicted the resident coder


def test_warmup_goes_to_an_idle_server_and_skips_a_busy_one() -> None:
    world = _build_world(warmup_unloaded=True)
    _, ckey = world.consumer(10.0)
    _, _, busy = add_producer(world, "busy.test", loaded=False, cold=30.0, extra_models={"coder": {"loaded": True, "vram_mib": 5200}})
    _, _, idle = add_producer(world, "idle.test", loaded=False, cold=90.0, extra_models={"coder": {"loaded": False}})
    assert world.chat(ckey, f"hivemind/{MODEL}").status_code == 503
    _wait_calls(idle, 1)
    assert len(idle.calls) == 1 and not busy.calls


def test_warmup_rate_limit_holds_and_expires() -> None:
    world = _build_world(warmup_unloaded=True)
    _, ckey = world.consumer(10.0)
    _, _, fake = add_producer(world, "idle.test", loaded=False, cold=90.0)
    assert world.chat(ckey, f"hivemind/{MODEL}").status_code == 503
    _wait_calls(fake, 1)
    assert world.chat(ckey, f"hivemind/{MODEL}").status_code == 503
    time.sleep(0.2)
    assert len(fake.calls) == 1
    world.clock.now += 31.0  # inside the 120 s producer TTL, past the 30 s warm-up interval
    assert world.chat(ckey, f"hivemind/{MODEL}").status_code == 503
    _wait_calls(fake, 2)
    assert len(fake.calls) == 2


def test_a_cheaper_producer_that_unloaded_the_model_does_not_block_a_warm_dearer_one(w: World) -> None:
    """2026-11-03 incident: the cheapest producer reported coder not loaded, the posted-price equilibrium selected it alone
    and the residency gate answered 503 while a warm producer for the same model stood idle."""
    cid, ckey = w.consumer(50.0)
    _, _, warm = add_producer(w, "warm.test", loaded=True)
    cheap_id, cheap_key = w.register("producer")
    cheap = FakeProducer()
    w.producers["cheap.test"] = cheap
    body = {
        "endpoint_url": "https://cheap.test",
        "models": [{"id": MODEL, "local_name": "q", "loaded": False, "cold_start_s": 90.0}],
        "cost_per_1k": COST / 4.0, "capacity_1k": 100.0, "execution_class": "t4",
    }
    assert w.app_client.post("/v1/hivemind/producer/heartbeat", headers={"Authorization": f"Bearer {cheap_key}"}, json=body).status_code == 200
    response = w.chat(ckey, f"hivemind/{MODEL}")
    assert response.status_code == 200, response.text
    assert len(warm.calls) == 1 and not cheap.calls
    assert w.store.balance(cid) < 50.0  # the request was served and charged
    # when the only producers left are all cold, the explicit model_loading answer is unchanged
    w2 = _build_world(warmup_unloaded=False)
    _, ckey2 = w2.consumer(10.0)
    _, _, cold = add_producer(w2, "cold.test", loaded=False, cold=90.0)
    r2 = w2.chat(ckey2, f"hivemind/{MODEL}")
    assert r2.status_code == 503 and r2.json()["error"]["code"] == "model_loading" and not cold.calls
