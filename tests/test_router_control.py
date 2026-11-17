"""Remote control (HM-S16-05): pause and VRAM cap are enforced in routing, obeyed by the agent and answered by the shield."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from hivemind.router.accounts import RouterStore
from tests.test_router_gateway import ADMIN_KEY, COST, FakeProducer, World, _build_world

MODEL = "qwen"
ROOT = Path(__file__).resolve().parents[1] / "src" / "hivemind" / "router"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"hivemind.router.{name}", ROOT / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent = _load("hivemind_producer_agent")
shield = _load("hivemind_producer_shield")

CONTROL = "/v1/hivemind/producer/control"


def auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def beat(world: World, key: str, host: str, *, capacity: float = 100.0, models: list[dict[str, Any]] | None = None, gpu: int | None = None) -> httpx.Response:
    body: dict[str, Any] = {
        "endpoint_url": f"https://{host}", "models": models or [{"id": MODEL, "local_name": "q"}],
        "cost_per_1k": COST, "capacity_1k": capacity, "execution_class": "t4",
    }
    if gpu is not None:
        body["gpu_vram_mib"] = gpu
    return world.app_client.post("/v1/hivemind/producer/heartbeat", headers=auth(key), json=body)


def set_control(world: World, key: str, **body: Any) -> httpx.Response:
    return world.app_client.post(CONTROL, headers=auth(key), json=body)


@pytest.fixture()
def w() -> World:
    return _build_world(warmup_unloaded=False)


def live(world: World) -> list[Any]:
    return world.store.active_producers(MODEL, world.clock(), 120)


# ----------------------------------------------------------------------------------------------- gateway
def test_pause_excludes_the_producer_from_routing_and_supply_at_once_and_resume_restores_it(w: World) -> None:
    _, ckey = w.consumer(50.0)
    _, pkey, fake = w.producer("a.test", {MODEL: "q"})
    assert w.chat(ckey).status_code == 200 and len(fake.calls) == 1
    assert MODEL in w.store.live_models(w.clock(), 120)

    response = set_control(w, pkey, paused=True)
    assert response.status_code == 200 and response.json()["control"]["paused"] is True
    assert live(w) == [] and w.store.live_models(w.clock(), 120) == []
    assert w.chat(ckey).status_code in (404, 503) and len(fake.calls) == 1  # zero further requests reached the producer (the model has no live supply)
    assert w.app_client.get("/v1/models", headers=auth(ckey)).status_code == 200
    # the agent keeps heartbeating: still excluded, and the ack carries the state
    ack = beat(w, pkey, "a.test").json()
    assert ack["control"]["paused"] is True and ack["control"]["acknowledged"] is False
    assert w.chat(ckey).status_code in (404, 503) and len(fake.calls) == 1

    assert set_control(w, pkey, paused=False).status_code == 200
    ack = beat(w, pkey, "a.test").json()
    assert ack["control"]["paused"] is False and ack["control"]["acknowledged"] is True
    assert len(live(w)) == 1 and w.chat(ckey).status_code == 200 and len(fake.calls) == 2


def test_pausing_one_producer_leaves_the_others_serving(w: World) -> None:
    _, ckey = w.consumer(50.0)
    _, akey, fa = w.producer("a.test", {MODEL: "q"})
    _, _, fb = w.producer("b.test", {MODEL: "q"})
    set_control(w, akey, paused=True)
    for _ in range(4):
        assert w.chat(ckey).status_code == 200
    assert len(fa.calls) == 0 and len(fb.calls) == 4
    assert [r.endpoint_url for r in live(w)] == ["https://b.test"]


def test_vram_cap_scales_capacity_like_the_card_clamp(w: World) -> None:
    account_id, key = w.register("producer")
    models = [{"id": MODEL, "local_name": "q", "loaded": True, "vram_mib": 4800}, {"id": "coder", "local_name": "c", "loaded": True, "vram_mib": 5300}]
    assert beat(w, key, "a.test", models=models, gpu=15360).json()["residency"]["effective_capacity_1k"] == 100.0
    assert live(w)[0].capacity_1k == 100.0
    assert set_control(w, key, paused=False, vram_cap_mib=5000).status_code == 200
    record = live(w)[0]
    assert record.capacity_1k == 50.0 and record.model_state(MODEL) is True and record.model_state("coder") is False
    assert record.residency is not None and "remote VRAM cap" in record.residency["capacity_note"] and "coder" in record.residency["capacity_note"]
    assert beat(w, key, "a.test", models=models, gpu=15360).json()["control"]["vram_cap_mib"] == 5000
    assert set_control(w, key, paused=False, vram_cap_mib=20000).status_code == 200  # a cap above the loaded sum changes nothing
    assert live(w)[0].capacity_1k == 100.0
    assert set_control(w, key, paused=False).status_code == 200  # clearing the cap
    control = w.store.producer_control(account_id)
    assert live(w)[0].capacity_1k == 100.0 and control is not None and control["vram_cap_mib"] is None


def test_no_control_state_means_the_old_ack_and_behaviour(w: World) -> None:
    _, key = w.register("producer")
    ack = beat(w, key, "a.test").json()
    assert "control" not in ack and len(live(w)) == 1
    assert w.app_client.get(CONTROL, headers=auth(key)).json()["control"] is None


def test_setting_control_needs_a_key_and_only_the_own_key_or_the_admin_key_works(w: World) -> None:
    pid, pkey, _ = w.producer("a.test", {MODEL: "q"})
    oid, okey, _ = w.producer("b.test", {MODEL: "q"})
    _, ckey = w.consumer(1.0)
    assert w.app_client.post(CONTROL, json={"paused": True}).status_code == 401
    assert w.app_client.post(CONTROL, headers=auth("hm-nope"), json={"paused": True}).status_code == 401
    assert set_control(w, ckey, paused=True).status_code == 403  # a consumer has nothing to pause
    assert set_control(w, okey, paused=True, account_id=pid).status_code == 403  # a stranger cannot pause someone else
    assert len(live(w)) == 2
    assert w.app_client.post(CONTROL, headers={"X-Admin-Key": "wrong"}, json={"paused": True, "account_id": pid}).status_code == 403
    assert set_control(w, pkey, paused=True, account_id=pid).status_code == 200  # naming yourself is fine
    assert [r.account_id for r in live(w)] == [oid]
    admin = w.app_client.post(CONTROL, headers={"X-Admin-Key": ADMIN_KEY}, json={"paused": True, "account_id": oid})
    assert admin.status_code == 200 and live(w) == []
    assert w.app_client.post(CONTROL, headers={"X-Admin-Key": ADMIN_KEY}, json={"paused": False}).status_code == 400  # admin must name the account
    assert w.app_client.post(CONTROL, headers={"X-Admin-Key": ADMIN_KEY}, json={"paused": False, "account_id": "ffff"}).status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"paused": "yes"}, {"paused": 1}, {}, {"paused": True, "vram_cap_mib": -1}, {"paused": True, "vram_cap_mib": 0},
        {"paused": True, "vram_cap_mib": 1.5}, {"paused": True, "vram_cap_mib": "8000"}, {"paused": True, "vram_cap_mib": 10**9},
        {"paused": True, "vram_cap_mib": True}, {"paused": True, "command": "rm -rf /"}, {"paused": True, "path": "/etc/passwd"},
    ],
)
def test_malformed_control_values_are_400(w: World, body: dict[str, Any]) -> None:
    account_id, key = w.register("producer")
    assert set_control(w, key, **body).status_code == 400
    assert w.store.producer_control(account_id) is None


def test_nan_and_infinity_are_rejected(w: World) -> None:
    _, key = w.register("producer")
    for literal in ("NaN", "Infinity", "-Infinity"):
        response = w.app_client.post(CONTROL, headers={**auth(key), "Content-Type": "application/json"}, content=f'{{"paused": false, "vram_cap_mib": {literal}}}')
        assert response.status_code == 400


def test_control_survives_a_reopened_database(tmp_path: Any) -> None:
    path = tmp_path / "r.db"
    secret = b"s" * 32
    store = RouterStore(str(path), secret)
    account_id, _ = store.register("p@x.test", "producer")
    store.set_producer_control(account_id, True, 4096)
    again = RouterStore(str(path), secret)
    control = again.producer_control(account_id)
    assert control is not None and control["paused"] is True and control["vram_cap_mib"] == 4096 and control["version"] == 1
    updated = again.set_producer_control(account_id, False, None)
    assert updated["version"] == 2 and updated["paused"] is False


# ----------------------------------------------------------------------------------------------- agent
AGENT_ARGS = ["--gateway", "https://gw.example.org", "--endpoint", "https://prod.example.org", "--model", "qwen=q", "--model", "coder=c",
              "--cost-per-1k", "0.4", "--capacity-1k", "50", "--skip-preflight", "--no-residency"]


def fake_gateway(acks: list[Any], seen: list[dict[str, Any]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=acks.pop(0) if len(acks) > 1 else acks[0])

    return httpx.MockTransport(handler)


def run_once(monkeypatch: pytest.MonkeyPatch, state: Path, ack: dict[str, Any], seen: list[dict[str, Any]]) -> int:
    monkeypatch.setenv(agent.API_KEY_ENV, "hm-test-key-for-producer")
    return int(agent.main([*AGENT_ARGS, "--once", "--control-state-file", str(state)], transport=fake_gateway([ack], seen)))


def ack_with(**control: Any) -> dict[str, Any]:
    return {"account_id": "a", "status": "live", "control": control}


def test_paused_ack_makes_the_next_heartbeat_advertise_nothing_and_flips_the_shield_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    state = tmp_path / "control.json"
    seen: list[dict[str, Any]] = []
    assert run_once(monkeypatch, state, ack_with(paused=True, vram_cap_mib=None, version=1), seen) == 0
    assert seen[0]["capacity_1k"] == 50.0 and "paused" not in seen[0]  # this beat could not know yet
    assert json.loads(state.read_text())["paused"] is True and stat.S_IMODE(os.stat(state).st_mode) == 0o600
    assert run_once(monkeypatch, state, ack_with(paused=True, vram_cap_mib=None, version=1), seen) == 0
    assert seen[1]["capacity_1k"] == 0.0 and seen[1]["paused"] is True
    assert run_once(monkeypatch, state, ack_with(paused=False, vram_cap_mib=None, version=2), seen) == 0
    assert json.loads(state.read_text())["paused"] is False
    assert run_once(monkeypatch, state, ack_with(paused=False, vram_cap_mib=None, version=2), seen) == 0
    assert seen[3]["capacity_1k"] == 50.0 and "paused" not in seen[3]  # resumed on the next heartbeat


def test_vram_cap_drops_an_over_cap_model_and_is_reported_back(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    state = tmp_path / "control.json"
    payload = agent.apply_control(
        {"capacity_1k": 50.0, "models": [{"id": "qwen", "local_name": "q", "loaded": True, "vram_mib": 4800},
                                         {"id": "coder", "local_name": "c", "loaded": True, "vram_mib": 5300},
                                         {"id": "cold", "local_name": "x", "loaded": False}]},
        {"paused": False, "vram_cap_mib": 5000},
    )
    assert [m["id"] for m in payload["models"]] == ["qwen", "cold"] and payload["vram_cap_mib"] == 5000 and payload["capacity_1k"] == 50.0
    only_big = agent.apply_control({"capacity_1k": 9.0, "models": [{"id": "coder", "local_name": "c", "loaded": True, "vram_mib": 5300}]}, {"paused": False, "vram_cap_mib": 5000})
    assert only_big["capacity_1k"] == 0.0 and len(only_big["models"]) == 1
    plain = {"capacity_1k": 5.0, "models": [{"id": "a", "local_name": "a"}]}
    assert agent.apply_control(dict(plain), None) == plain
    assert agent.apply_control(dict(plain), {"paused": False, "vram_cap_mib": None}) == plain
    # through main(): the cap learned from an ack is reported back in the next heartbeat, and clearing it resumes the full set
    seen: list[dict[str, Any]] = []
    assert run_once(monkeypatch, state, ack_with(paused=False, vram_cap_mib=5000, version=1), seen) == 0
    assert run_once(monkeypatch, state, ack_with(paused=False, vram_cap_mib=None, version=2), seen) == 0
    assert seen[1]["vram_cap_mib"] == 5000
    assert run_once(monkeypatch, state, ack_with(paused=False, vram_cap_mib=None, version=2), seen) == 0
    assert "vram_cap_mib" not in seen[2]


@pytest.mark.parametrize(
    "bad",
    [None, "paused", [], 7, {}, {"paused": "true"}, {"paused": 3}, {"paused": True, "vram_cap_mib": -5}, {"paused": True, "vram_cap_mib": "x"},
     {"paused": False, "vram_cap_mib": 10**12}, {"paused": True, "vram_cap_mib": True}],
)
def test_malformed_control_is_ignored_and_logged_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], bad: Any) -> None:
    state = tmp_path / "control.json"
    monkeypatch.setenv(agent.API_KEY_ENV, "hm-test-key-for-producer")
    seen: list[dict[str, Any]] = []
    calls = {"n": 0}

    real_sleep = agent.time.sleep

    def fake_sleep(sec: float) -> None:
        if threading.current_thread() is not threading.main_thread():
            real_sleep(sec)
            return
        calls["n"] += 1
        if calls["n"] >= 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(agent.time, "sleep", fake_sleep)
    transport = fake_gateway([{"status": "live", "control": bad}], seen)
    code = agent.main([*AGENT_ARGS, "--control-state-file", str(state)], transport=transport)
    assert code == 0 and len(seen) == 2 and seen[1]["capacity_1k"] == 50.0 and not state.exists()
    assert capsys.readouterr().err.count("ignoring malformed control") == 1


def test_a_valid_control_object_is_reduced_to_pause_and_cap_only() -> None:
    parsed = agent.parse_control({"paused": True, "vram_cap_mib": 4096, "version": 3, "command": "rm -rf /", "path": "/etc/shadow"})
    assert parsed == {"paused": True, "vram_cap_mib": 4096, "version": 3}


# ----------------------------------------------------------------------------------------------- shield
TOKEN = "shield-token-" + "z" * 20
CHAT = {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}


def make_shield(state: Path) -> tuple[TestClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{"id": "q"}]})
        return httpx.Response(200, headers={"content-type": "application/json"}, stream=httpx.ByteStream(b'{"choices":[{"message":{"content":"ok"}}]}'))

    app = shield.create_shield_app("http://up.local:11434", TOKEN, transport=httpx.MockTransport(handler), control_state_file=str(state))
    return TestClient(app), seen


def test_shield_answers_503_producer_paused_for_chat_but_keeps_models_working(tmp_path: Path) -> None:
    state = tmp_path / "control.json"
    client, seen = make_shield(state)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    assert client.post("/v1/chat/completions", json=CHAT, headers=headers).status_code == 200  # no file: today's behaviour
    state.write_text(json.dumps({"paused": True, "vram_cap_mib": None}))
    before = len(seen)
    response = client.post("/v1/chat/completions", json=CHAT, headers=headers)
    assert response.status_code == 503 and response.json()["error"]["code"] == "producer_paused" and len(seen) == before
    assert client.get("/v1/models", headers=headers).status_code == 200
    assert client.post("/v1/chat/completions", json=CHAT).status_code == 401  # auth still comes first
    state.write_text(json.dumps({"paused": False, "vram_cap_mib": None}))
    assert client.post("/v1/chat/completions", json=CHAT, headers=headers).status_code == 200
    state.write_text("not json")  # unreadable state means not paused
    assert client.post("/v1/chat/completions", json=CHAT, headers=headers).status_code == 200


# ----------------------------------------------------------------------------------------------- integration
def test_end_to_end_pause_heartbeat_shield_503_resume_shield_forwards(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    w = _build_world(warmup_unloaded=False)
    _, ckey = w.consumer(50.0)
    _, pkey = w.register("producer")
    backend = FakeProducer()
    w.producers["prod.example.org"] = backend
    state = tmp_path / "control.json"
    shield_client, _ = make_shield(state)

    def to_gateway(request: httpx.Request) -> httpx.Response:  # the agent talks to the in-process gateway app
        reply = w.app_client.request(request.method, request.url.path, headers={"Authorization": request.headers["authorization"], "Content-Type": "application/json"}, content=request.content)
        return httpx.Response(reply.status_code, content=reply.content, headers={"content-type": "application/json"})

    monkeypatch.setenv(agent.API_KEY_ENV, pkey)
    heads = {"Authorization": f"Bearer {TOKEN}"}

    def tick() -> None:  # one agent iteration against the gateway
        assert agent.main([*AGENT_ARGS, "--once", "--control-state-file", str(state)], transport=httpx.MockTransport(to_gateway)) == 0

    tick()  # serving
    assert shield_client.post("/v1/chat/completions", json=CHAT, headers=heads).status_code == 200
    assert w.chat(ckey).status_code == 200 and len(backend.calls) == 1

    assert set_control(w, pkey, paused=True).status_code == 200
    assert w.chat(ckey).status_code in (404, 503) and len(backend.calls) == 1  # the router stops at once
    tick()  # the heartbeat ack delivers the pause to the agent, which tells the shield
    paused = shield_client.post("/v1/chat/completions", json=CHAT, headers=heads)
    assert paused.status_code == 503 and paused.json()["error"]["code"] == "producer_paused"
    assert shield_client.get("/v1/models", headers=heads).status_code == 200
    tick()  # a paused agent advertises nothing
    assert live(w) == [] and w.chat(ckey).status_code in (404, 503)

    assert set_control(w, pkey, paused=False).status_code == 200
    tick()  # the ack delivers the resume
    assert shield_client.post("/v1/chat/completions", json=CHAT, headers=heads).status_code == 200
    tick()
    assert len(live(w)) == 1 and live(w)[0].capacity_1k == 50.0
    assert w.chat(ckey).status_code == 200 and len(backend.calls) == 2
