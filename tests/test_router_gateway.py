"""Tests for hivemind.router.gateway. Producers are external I/O, so they are replaced by httpx.MockTransport."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from hivemind.router import RouterStore, create_router_app
from hivemind.router import gateway as gateway_module
from hivemind.router.gateway import (
    DEFAULT_MAX_TOKENS,
    UpstreamLimitError,
    has_completion_choice,
    limited_lines,
    read_limited,
    request_token_limit,
    SlidingWindowLimiter,
    WeightedRoundRobin,
    completion_byte_count,
    estimate_tokens,
    meter_usage,
    producer_url_problem,
    prompt_byte_count,
)
from hivemind.router.market import ConsumerBid, FeePolicy, ProducerOffer, clear_market

SECRET = b"g" * 32
ADMIN_KEY = "admin-secret-key"
ADMIN_HASH = hmac.new(SECRET, ADMIN_KEY.encode(), hashlib.sha256).hexdigest()
FEE = FeePolicy()
PROMPT = "hello"
ANSWER = "hi there!"  # 9 bytes
COST = 0.4
POSTED_PRICE = COST / (1.0 - FEE.fee_rate) * (1.0 + FEE.posted_margin)


class FakeClock:
    def __init__(self) -> None:
        self.now: float = time.time()

    def __call__(self) -> float:
        return self.now


@dataclass
class FakeProducer:
    """A scripted OpenAI-compatible backend served through httpx.MockTransport."""

    answer: str = ANSWER
    reported_prompt: int = 3
    reported_completion: int = 3
    status: int = 200
    explode: bool = False
    calls: list[httpx.Request] = field(default_factory=list)

    def completion(self, model: str) -> dict[str, Any]:
        return {
            "id": "chatcmpl-up",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": self.answer}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": self.reported_prompt,
                "completion_tokens": self.reported_completion,
                "total_tokens": self.reported_prompt + self.reported_completion,
            },
        }

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if self.explode:
            raise httpx.ConnectError("producer down", request=request)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "boom"}})
        body = json.loads(request.content)
        if body.get("stream"):
            pieces = [self.answer[i : i + 3] for i in range(0, len(self.answer), 3)]
            lines = [
                "data: " + json.dumps({"id": "c1", "object": "chat.completion.chunk", "model": body["model"], "choices": [{"index": 0, "delta": {"content": p}}]})
                for p in pieces
            ]
            lines.append(
                "data: " + json.dumps({"id": "c1", "object": "chat.completion.chunk", "model": body["model"], "choices": [], "usage": {"prompt_tokens": self.reported_prompt, "completion_tokens": self.reported_completion}})
            )
            lines.append("data: [DONE]")
            return httpx.Response(200, content=("\n\n".join(lines) + "\n\n").encode(), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=self.completion(body["model"]))


@dataclass
class World:
    store: RouterStore
    app_client: TestClient
    clock: FakeClock
    producers: dict[str, FakeProducer]
    counter: int = 0

    def register(self, role: str) -> tuple[str, str]:
        self.counter += 1
        response = self.app_client.post("/v1/hivemind/register", json={"handle": f"user{self.counter}@{role}.test", "role": role})
        assert response.status_code == 200, response.text
        body = response.json()
        return body["account_id"], body["api_key"]

    def consumer(self, funds: float = 100.0) -> tuple[str, str]:
        account_id, key = self.register("consumer")
        if funds > 0:
            self.store.grant(account_id, funds, "test")
        return account_id, key

    def producer(self, host: str, models: dict[str, str], *, cost: float = COST, capacity: float = 100.0, agent: FakeProducer | None = None) -> tuple[str, str, FakeProducer]:
        account_id, key = self.register("producer")
        fake = agent if agent is not None else FakeProducer()
        self.producers[host] = fake
        response = self.beat(key, host, models, cost, capacity)
        assert response.status_code == 200, response.text
        return account_id, key, fake

    def beat(self, key: str, host: str, models: dict[str, str], cost: float, capacity: float) -> httpx.Response:
        return self.app_client.post(
            "/v1/hivemind/producer/heartbeat",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "endpoint_url": f"https://{host}",
                "models": [{"id": mid, "local_name": local} for mid, local in models.items()],
                "cost_per_1k": cost,
                "capacity_1k": capacity,
                "execution_class": "t4",
            },
        )

    def chat(self, key: str, model: str = "hivemind/qwen", **extra: Any) -> httpx.Response:
        payload = {"model": model, "messages": [{"role": "user", "content": PROMPT}], **extra}
        return self.app_client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=payload)


def _build_world(*, allow_private: bool = True, resolver: Callable[[str], list[str]] | None = None, **app_kwargs: Any) -> World:
    store = RouterStore(":memory:", SECRET)
    clock = FakeClock()
    producers: dict[str, FakeProducer] = {}

    def route(request: httpx.Request) -> httpx.Response:
        agent = producers.get(request.url.host)
        if agent is None:
            raise httpx.ConnectError("unknown host", request=request)
        return agent.handle(request)

    app = create_router_app(
        store,
        http_client=httpx.Client(transport=httpx.MockTransport(route)),
        allow_private_producers=allow_private,
        admin_key_hash=ADMIN_HASH,
        clock=clock,
        resolver=resolver,
        **app_kwargs,
    )
    return World(store, TestClient(app), clock, producers)


@pytest.fixture()
def world() -> World:
    return _build_world()


def _expected_billed(answer: str = ANSWER) -> int:
    prompt_bytes = len(PROMPT.encode())
    return min(prompt_bytes, 2 * estimate_tokens(prompt_bytes)) + len(answer.encode())


# ---------------------------------------------------------------------------------------------------------
# pricing, charging, crediting
# ---------------------------------------------------------------------------------------------------------


def test_chat_charges_consumer_and_credits_producer_with_fee_split(world: World) -> None:
    consumer_id, key = world.consumer(10.0)
    producer_id, _, fake = world.producer("prod-a.test", {"qwen": "qwen3:8b"})

    response = world.chat(key)

    assert response.status_code == 200, response.text
    body = response.json()
    info = body["hivemind"]
    assert body["model"] == "hivemind/qwen"
    assert body["choices"][0]["message"]["content"] == ANSWER
    assert info["producer_id"] == producer_id
    assert info["mechanism"] == "posted_price"
    assert info["price_per_1k"] == pytest.approx(POSTED_PRICE)
    billed = 6  # honest report: 3 prompt + 3 completion, below the byte caps
    charge = POSTED_PRICE * billed / 1000.0
    assert info["charged_hmt"] == pytest.approx(charge)
    assert world.store.balance(consumer_id) == pytest.approx(10.0 - charge)
    assert world.store.balance(producer_id) == pytest.approx(charge * (1.0 - FEE.fee_rate))
    assert world.store.balance("sys:validators") == pytest.approx(charge * FEE.validator_share)
    assert world.store.balance("sys:burn") == pytest.approx(charge * FEE.burn_share)
    assert world.store.total_balances() == pytest.approx(world.store.total_minted(), abs=1e-9)
    assert world.store.verify_chain()
    forwarded = json.loads(fake.calls[0].content)
    assert forwarded["model"] == "qwen3:8b"
    assert "authorization" not in fake.calls[0].headers


def test_two_consumers_clear_at_the_kelly_price(world: World) -> None:
    _, key = world.consumer(10.0)
    world.consumer(10.0)
    world.producer("prod-a.test", {"qwen": "qwen3:8b"})

    response = world.chat(key)

    assert response.status_code == 200, response.text
    info = response.json()["hivemind"]
    assert info["mechanism"] == "kelly"
    assert info["price_per_1k"] == pytest.approx(1.0)  # v = 2, 2 -> omega = 1
    market = world.app_client.get("/v1/hivemind/market", params={"model": "qwen"}, headers={"Authorization": f"Bearer {key}"}).json()
    assert market["equilibrium"]["price_per_1k"] == pytest.approx(1.0)
    assert market["certificate"]["win_win"] is True


def test_inflation_attempt_is_billed_by_the_byte_cap(world: World) -> None:
    consumer_id, key = world.consumer(10.0)
    world.producer("prod-a.test", {"qwen": "q"}, agent=FakeProducer(reported_prompt=30, reported_completion=30))

    response = world.chat(key)

    assert response.status_code == 200, response.text
    info = response.json()["hivemind"]
    cap = _expected_billed()
    assert info["charged_hmt"] == pytest.approx(POSTED_PRICE * cap / 1000.0)
    assert info["charged_hmt"] < POSTED_PRICE * 60 / 1000.0
    assert cap <= len(PROMPT.encode()) + len(ANSWER.encode())
    row = world.store.recent_usage(consumer_id, 1)[0]
    assert row["billed_tokens"] == cap


def test_max_tokens_caps_the_completion_part(world: World) -> None:
    consumer_id, key = world.consumer(10.0)
    world.producer("prod-a.test", {"qwen": "q"}, agent=FakeProducer(reported_prompt=3, reported_completion=500))

    response = world.chat(key, max_tokens=4)

    assert response.status_code == 200, response.text
    row = world.store.recent_usage(consumer_id, 1)[0]
    assert row["completion_tokens"] == 4
    assert row["billed_tokens"] == 3 + 4


def test_long_answer_is_clamped_to_the_balance_and_never_overdraws(world: World) -> None:
    consumer_id, key = world.consumer(0.51)
    long_answer = "x" * 5000
    world.producer("prod-a.test", {"qwen": "q"}, agent=FakeProducer(answer=long_answer, reported_prompt=5, reported_completion=5000))

    response = world.chat(key)  # reserve (1024 default tokens) fits in 0.51, the real bill would not

    assert response.status_code == 200, response.text
    assert world.store.balance(consumer_id) >= 0.0
    assert response.json()["hivemind"]["charged_hmt"] <= 0.51 + 1e-12
    assert world.store.min_balance() >= 0.0
    assert world.store.verify_chain()


# ---------------------------------------------------------------------------------------------------------
# refusals and failures
# ---------------------------------------------------------------------------------------------------------


def test_402_on_low_balance_and_producer_is_never_called(world: World) -> None:
    consumer_id, key = world.consumer(0.0)
    _, _, fake = world.producer("prod-a.test", {"qwen": "q"})

    response = world.chat(key)

    assert response.status_code == 402
    error = response.json()["error"]
    assert error["code"] == "insufficient_balance"
    assert set(error) == {"message", "type", "code"}
    assert fake.calls == []
    assert world.store.balance(consumer_id) == 0.0


def test_402_when_the_price_exceeds_the_consumer_max_price(world: World) -> None:
    _, key = world.consumer(10.0)
    world.producer("prod-a.test", {"qwen": "q"})
    set_price = world.app_client.post("/v1/hivemind/consumer/max-price", headers={"Authorization": f"Bearer {key}"}, json={"max_price_per_1k": 0.1})
    assert set_price.status_code == 200

    response = world.chat(key)

    assert response.status_code == 402
    assert response.json()["error"]["code"] == "price_above_max_price"


def test_502_without_charge_when_every_producer_is_down(world: World) -> None:
    consumer_id, key = world.consumer(10.0)
    producer_id, _, _ = world.producer("prod-a.test", {"qwen": "q"}, agent=FakeProducer(explode=True))

    response = world.chat(key)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_failure"
    assert world.store.balance(consumer_id) == 10.0
    assert world.store.balance(producer_id) == 0.0
    assert world.store.recent_usage(consumer_id) == []


def test_failover_to_the_second_producer(world: World) -> None:
    _, key = world.consumer(10.0)
    bad_id, _, bad = world.producer("prod-bad.test", {"qwen": "q"}, capacity=1000.0, agent=FakeProducer(status=500))
    good_id, _, good = world.producer("prod-good.test", {"qwen": "q"}, capacity=1.0)

    response = world.chat(key)

    assert response.status_code == 200, response.text
    assert len(bad.calls) == 1  # the heavy producer is tried first, fails, and the request moves on
    assert response.json()["hivemind"]["producer_id"] == good_id
    assert world.store.balance(good_id) > 0.0
    assert world.store.balance(bad_id) == 0.0
    assert len(good.calls) == 1


def test_unknown_model_bad_body_and_auth_errors(world: World) -> None:
    _, key = world.consumer(10.0)
    world.producer("prod-a.test", {"qwen": "q"})

    missing = world.chat(key, model="hivemind/nope")
    no_messages = world.app_client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {key}"}, json={"model": "hivemind/qwen"})
    anonymous = world.app_client.post("/v1/chat/completions", json={"model": "hivemind/qwen", "messages": [{"role": "user", "content": "x"}]})
    wrong_key = world.chat("hm-not-a-real-key")

    assert missing.status_code == 404 and missing.json()["error"]["code"] == "model_not_found"
    assert no_messages.status_code == 400 and "error" in no_messages.json()
    assert anonymous.status_code == 401 and anonymous.json()["error"]["code"] == "invalid_api_key"
    assert wrong_key.status_code == 401


# ---------------------------------------------------------------------------------------------------------
# streaming
# ---------------------------------------------------------------------------------------------------------


def _read_sse(response: httpx.Response) -> list[str]:
    return [line[len("data: ") :] for line in response.iter_lines() if line.startswith("data: ")]


def test_streaming_proxies_chunks_and_settles_after_the_stream(world: World) -> None:
    consumer_id, key = world.consumer(10.0)
    producer_id, _, fake = world.producer("prod-a.test", {"qwen": "q"})

    with world.app_client.stream(
        "POST", "/v1/chat/completions", headers={"Authorization": f"Bearer {key}"},
        json={"model": "hivemind/qwen", "stream": True, "messages": [{"role": "user", "content": PROMPT}]},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _read_sse(response)

    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert text == ANSWER
    assert all(c["model"] == "hivemind/qwen" for c in chunks)
    summary = chunks[-1]["hivemind"]
    assert summary["producer_id"] == producer_id
    billed = 6
    assert summary["charged_hmt"] == pytest.approx(POSTED_PRICE * billed / 1000.0)
    assert world.store.balance(consumer_id) == pytest.approx(10.0 - summary["charged_hmt"])
    assert world.store.balance(producer_id) == pytest.approx(summary["charged_hmt"] * (1.0 - FEE.fee_rate))
    assert json.loads(fake.calls[0].content)["stream_options"] == {"include_usage": True}
    assert world.store.verify_chain()


def test_streaming_inflated_usage_is_capped_by_streamed_bytes(world: World) -> None:
    consumer_id, key = world.consumer(10.0)
    world.producer("prod-a.test", {"qwen": "q"}, agent=FakeProducer(reported_prompt=900, reported_completion=900))

    with world.app_client.stream(
        "POST", "/v1/chat/completions", headers={"Authorization": f"Bearer {key}"},
        json={"model": "hivemind/qwen", "stream": True, "messages": [{"role": "user", "content": PROMPT}]},
    ) as response:
        _read_sse(response)

    assert world.store.recent_usage(consumer_id, 1)[0]["billed_tokens"] == _expected_billed()


def test_streaming_fails_over_before_the_first_byte(world: World) -> None:
    _, key = world.consumer(10.0)
    world.producer("prod-bad.test", {"qwen": "q"}, capacity=1000.0, agent=FakeProducer(explode=True))
    good_id, _, _ = world.producer("prod-good.test", {"qwen": "q"}, capacity=1.0)

    with world.app_client.stream(
        "POST", "/v1/chat/completions", headers={"Authorization": f"Bearer {key}"},
        json={"model": "hivemind/qwen", "stream": True, "messages": [{"role": "user", "content": PROMPT}]},
    ) as response:
        assert response.status_code == 200
        events = _read_sse(response)

    assert json.loads(events[-2])["hivemind"]["producer_id"] == good_id
    assert world.store.balance(good_id) > 0.0


def test_streaming_502_when_all_producers_fail_and_no_charge(world: World) -> None:
    consumer_id, key = world.consumer(10.0)
    world.producer("prod-a.test", {"qwen": "q"}, agent=FakeProducer(status=503))

    response = world.chat(key, stream=True)

    assert response.status_code == 502
    assert world.store.balance(consumer_id) == 10.0


# ---------------------------------------------------------------------------------------------------------
# official OpenAI client
# ---------------------------------------------------------------------------------------------------------


def test_official_openai_client_works_against_the_gateway(world: World) -> None:
    openai = pytest.importorskip("openai")
    consumer_id, key = world.consumer(10.0)
    world.producer("prod-a.test", {"qwen": "qwen3:8b"})
    sdk = openai.OpenAI(base_url="http://testserver/v1", api_key=key, http_client=world.app_client, max_retries=0)

    listed = [m.id for m in sdk.models.list().data]
    completion = sdk.chat.completions.create(model="hivemind/qwen", messages=[{"role": "user", "content": PROMPT}])
    streamed = sdk.chat.completions.create(model="hivemind/qwen", messages=[{"role": "user", "content": PROMPT}], stream=True)
    pieces = [c.choices[0].delta.content or "" for c in streamed if c.choices]

    assert "hivemind/qwen" in listed and "hivemind/auto" in listed
    assert completion.choices[0].message.content == ANSWER
    assert completion.model == "hivemind/qwen"
    assert "".join(pieces) == ANSWER
    assert len(world.store.recent_usage(consumer_id)) == 2
    with pytest.raises(openai.AuthenticationError):
        openai.OpenAI(base_url="http://testserver/v1", api_key="hm-wrong", http_client=world.app_client, max_retries=0).chat.completions.create(
            model="hivemind/qwen", messages=[{"role": "user", "content": PROMPT}]
        )


# ---------------------------------------------------------------------------------------------------------
# models, routing, liveness
# ---------------------------------------------------------------------------------------------------------


def test_models_lists_live_models_and_auto_picks_the_largest_supply(world: World) -> None:
    _, key = world.consumer(10.0)
    world.producer("prod-a.test", {"small": "s"}, capacity=5.0)
    world.producer("prod-b.test", {"big": "b"}, capacity=500.0)

    listing = world.app_client.get("/v1/models").json()
    response = world.chat(key, model="hivemind/auto")

    assert listing["object"] == "list"
    assert [m["id"] for m in listing["data"]] == ["hivemind/auto", "hivemind/big", "hivemind/small"]
    assert response.status_code == 200
    assert response.json()["model"] == "hivemind/big"


def test_a_producer_that_stops_heartbeating_disappears(world: World) -> None:
    _, key = world.consumer(10.0)
    world.producer("prod-a.test", {"qwen": "q"})
    assert world.chat(key).status_code == 200

    world.clock.now += 121.0

    assert [m["id"] for m in world.app_client.get("/v1/models").json()["data"]] == ["hivemind/auto"]
    assert world.chat(key).status_code == 404
    assert world.chat(key, model="hivemind/auto").status_code == 503


def test_weighted_round_robin_follows_capacity_across_requests(world: World) -> None:
    _, key = world.consumer(50.0)
    heavy_id, _, _ = world.producer("prod-heavy.test", {"qwen": "q"}, capacity=3.0)
    light_id, _, _ = world.producer("prod-light.test", {"qwen": "q"}, capacity=1.0)

    served = [world.chat(key).json()["hivemind"]["producer_id"] for _ in range(8)]

    assert served.count(heavy_id) == 6
    assert served.count(light_id) == 2


def test_market_endpoint_needs_auth_and_hides_consumer_identities(world: World) -> None:
    consumer_id, key = world.consumer(1.0)
    world.consumer(1.0)
    world.producer("prod-a.test", {"qwen": "q"})

    assert world.app_client.get("/v1/hivemind/market", params={"model": "hivemind/qwen"}).status_code == 401
    response = world.app_client.get("/v1/hivemind/market", params={"model": "hivemind/qwen"}, headers={"Authorization": f"Bearer {key}"})

    assert response.status_code == 200
    body = response.json()
    assert body["certificate"]["win_win"] is True
    assert body["equilibrium"]["mechanism"] == "kelly"
    assert consumer_id not in response.text
    assert world.app_client.get("/v1/hivemind/market", params={"model": "nope"}, headers={"Authorization": f"Bearer {key}"}).status_code == 404


# ---------------------------------------------------------------------------------------------------------
# registration, accounts, admin
# ---------------------------------------------------------------------------------------------------------


def test_register_me_and_duplicate_handle(world: World) -> None:
    created = world.app_client.post("/v1/hivemind/register", json={"handle": "Alice@Example.org", "role": "consumer"})
    again = world.app_client.post("/v1/hivemind/register", json={"handle": "alice@example.org", "role": "consumer"})
    bad_role = world.app_client.post("/v1/hivemind/register", json={"handle": "bob", "role": "admin"})
    bad_handle = world.app_client.post("/v1/hivemind/register", json={"handle": "no spaces allowed", "role": "consumer"})
    key = created.json()["api_key"]
    me = world.app_client.get("/v1/hivemind/me", headers={"Authorization": f"Bearer {key}"})

    assert created.status_code == 200 and key.startswith("hm-")
    assert again.status_code == 409
    assert bad_role.status_code == 400 and bad_handle.status_code == 400
    assert me.json()["role"] == "consumer" and me.json()["balance"] == 0.0
    assert me.json()["handle"] == "alice@example.org"


def test_registration_is_rate_limited_per_client(world: World) -> None:
    statuses = [
        world.app_client.post("/v1/hivemind/register", json={"handle": f"rate{i}", "role": "consumer"}).status_code for i in range(12)
    ]

    assert statuses[:10] == [200] * 10
    assert statuses[10:] == [429, 429]
    world.clock.now += 61.0
    assert world.app_client.post("/v1/hivemind/register", json={"handle": "rate-later", "role": "consumer"}).status_code == 200


def test_admin_grant_requires_the_admin_key(world: World) -> None:
    account_id, key = world.consumer(0.0)

    denied = world.app_client.post("/v1/hivemind/admin/grant", json={"account_id": account_id, "amount": 5})
    wrong = world.app_client.post("/v1/hivemind/admin/grant", headers={"X-Admin-Key": "nope"}, json={"account_id": account_id, "amount": 5})
    granted = world.app_client.post("/v1/hivemind/admin/grant", headers={"X-Admin-Key": ADMIN_KEY}, json={"account_id": account_id, "amount": 5, "ref": "promo"})
    unknown = world.app_client.post("/v1/hivemind/admin/grant", headers={"X-Admin-Key": ADMIN_KEY}, json={"account_id": "ffff", "amount": 5})

    assert denied.status_code == 403 and wrong.status_code == 403
    assert granted.status_code == 200 and granted.json()["balance"] == 5.0
    assert unknown.status_code == 404
    assert world.store.balance(account_id) == 5.0
    store_without_admin = create_router_app(RouterStore(":memory:", SECRET))
    assert TestClient(store_without_admin).post("/v1/hivemind/admin/grant", headers={"X-Admin-Key": ADMIN_KEY}, json={"handle": "x", "amount": 1}).status_code == 403
    assert key.startswith("hm-")


def test_only_producers_may_heartbeat_and_healthz(world: World) -> None:
    _, consumer_key = world.consumer(0.0)

    refused = world.beat(consumer_key, "prod-a.test", {"qwen": "q"}, COST, 10.0)

    assert refused.status_code == 403
    assert world.app_client.get("/healthz").json() == {"status": "ok"}
    invalid = world.app_client.post("/v1/hivemind/consumer/max-price", headers={"Authorization": f"Bearer {consumer_key}"}, json={"max_price_per_1k": -1})
    assert invalid.status_code == 400


# ---------------------------------------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1",
        "http://8.8.8.8",
        "https://10.0.0.1",
        "https://192.168.1.5:8443",
        "https://169.254.169.254",
        "https://[::1]",
        "https://[::ffff:127.0.0.1]",
        "https://user:pw@8.8.8.8",
        "ftp://8.8.8.8",
    ],
)
def test_heartbeat_rejects_unsafe_endpoints(url: str) -> None:
    strict = _build_world(allow_private=False)
    _, key = strict.register("producer")

    response = strict.app_client.post(
        "/v1/hivemind/producer/heartbeat",
        headers={"Authorization": f"Bearer {key}"},
        json={"endpoint_url": url, "models": [{"id": "qwen", "local_name": "q"}], "cost_per_1k": 0.4, "capacity_1k": 10},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_endpoint"
    assert strict.store.active_producers("qwen", strict.clock(), 120) == []


def test_heartbeat_accepts_global_addresses_and_checks_dns_results() -> None:
    answers = {"good.example": ["8.8.8.8"], "rebind.example": ["8.8.8.8", "127.0.0.1"], "localhost": ["127.0.0.1"]}

    def resolver(host: str) -> list[str]:
        return answers[host]

    strict = _build_world(allow_private=False, resolver=resolver)
    _, key = strict.register("producer")

    def beat(url: str) -> int:
        return strict.app_client.post(
            "/v1/hivemind/producer/heartbeat",
            headers={"Authorization": f"Bearer {key}"},
            json={"endpoint_url": url, "models": [{"id": "qwen", "local_name": "q"}], "cost_per_1k": 0.4, "capacity_1k": 10},
        ).status_code

    assert beat("https://good.example") == 200
    assert beat("https://rebind.example") == 400
    assert beat("https://localhost") == 400
    assert beat("https://8.8.8.8") == 200


def test_forward_time_recheck_skips_a_producer_whose_dns_turned_private() -> None:
    answers = {"prod-a.test": ["8.8.8.8"]}
    strict = _build_world(allow_private=False, resolver=lambda host: answers[host])
    _, key = strict.consumer(10.0)
    strict.producer("prod-a.test", {"qwen": "q"})
    assert strict.chat(key).status_code == 200

    answers["prod-a.test"] = ["10.0.0.7"]
    response = strict.chat(key)

    assert response.status_code == 502
    assert len(strict.producers["prod-a.test"].calls) == 1


@given(st.ip_addresses(v=4))
@settings(max_examples=200, deadline=None)
def test_producer_url_problem_matches_ip_globality(address: Any) -> None:
    problem = producer_url_problem(f"https://{address}", allow_private=False)

    assert (problem is None) == address.is_global
    assert producer_url_problem(f"http://{address}", allow_private=False) is not None
    assert producer_url_problem(f"http://{address}", allow_private=True) is None


# ---------------------------------------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------------------------------------


@given(
    st.integers(0, 10**6), st.integers(0, 10**6), st.integers(0, 10**5), st.integers(0, 10**5), st.one_of(st.none(), st.integers(1, 10**5))
)
def test_meter_usage_never_exceeds_reports_bytes_or_max_tokens(rp: int, rc: int, pb: int, cb: int, max_tokens: int | None) -> None:
    prompt, completion, billed = meter_usage(rp, rc, pb, cb, max_tokens)

    assert billed == prompt + completion
    assert 0 <= billed <= min(rp + rc, pb + cb)
    assert prompt <= min(rp, pb) and completion <= min(rc, cb)
    assert completion <= (DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens)


@given(st.integers(0, 10**4), st.integers(0, 10**4), st.integers(0, 10**4), st.integers(0, 10**4))
def test_meter_usage_is_identity_for_honest_reports(prompt_bytes: int, completion_bytes: int, extra_p: int, extra_c: int) -> None:
    honest_completion = min(completion_bytes, DEFAULT_MAX_TOKENS)
    prompt, completion, billed = meter_usage(prompt_bytes, completion_bytes, prompt_bytes + extra_p, completion_bytes + extra_c, None)

    assert (prompt, completion, billed) == (prompt, honest_completion, prompt + honest_completion)
    assert prompt <= prompt_bytes and prompt <= 2 * estimate_tokens(prompt_bytes + extra_p)
    assert estimate_tokens(prompt_bytes) <= prompt_bytes


def test_byte_counters_handle_parts_tools_and_deltas() -> None:
    body = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "héllo"}, {"type": "text", "text": "!"}]},
            {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}]},
        ],
    }
    plain = prompt_byte_count({"messages": body["messages"][:1]})
    answer = {"choices": [{"message": {"content": "ab", "reasoning_content": "cd", "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}]}}]}
    chunk = {"choices": [{"delta": {"content": "xyz"}}, {"delta": {}}]}

    assert plain == len("héllo!".encode())
    assert prompt_byte_count(body) > plain
    assert completion_byte_count(answer, delta=False) == len("abcdf{}")
    assert completion_byte_count(chunk, delta=True) == 3
    assert completion_byte_count({"choices": "garbage"}, delta=True) == 0


@given(st.dictionaries(st.sampled_from(["a", "b", "c", "d"]), st.integers(1, 6), min_size=1))
def test_weighted_round_robin_serves_each_candidate_in_proportion(weights: dict[str, int]) -> None:
    rotation = WeightedRoundRobin()
    total = sum(weights.values())

    picks = [rotation.order("m", {k: float(v) for k, v in weights.items()})[0] for _ in range(total)]

    assert {k: picks.count(k) for k in weights} == weights
    assert sorted(rotation.order("m", {k: float(v) for k, v in weights.items()})) == sorted(weights)


def test_weighted_round_robin_is_deterministic_and_empty_safe() -> None:
    first = WeightedRoundRobin()
    second = WeightedRoundRobin()
    weights = {"x": 2.0, "y": 5.0, "z": 1.0}

    assert [first.order("m", weights)[0] for _ in range(10)] == [second.order("m", weights)[0] for _ in range(10)]
    assert first.order("m", {}) == []


def test_sliding_window_limiter_blocks_then_recovers() -> None:
    limiter = SlidingWindowLimiter(3, 10.0)

    assert [limiter.allow("ip", 0.0) for _ in range(4)] == [True, True, True, False]
    assert limiter.allow("other", 0.0) is True
    assert limiter.allow("ip", 10.5) is True


# ---------------------------------------------------------------------------------------------------------
# security regressions (adversarial review findings)
# ---------------------------------------------------------------------------------------------------------


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def test_unfunded_sybil_bids_cannot_move_the_market_price(world: World) -> None:
    """Finding 1: zero-balance accounts used to set the Kelly price for everyone."""
    honest = [world.consumer(50.0) for _ in range(3)]
    sybils = [world.consumer(0.0) for _ in range(2)]
    world.producer("prod-a.test", {"qwen": "q"})
    for _, sybil_key in sybils:
        rejected = world.app_client.post("/v1/hivemind/consumer/max-price", headers=_auth(sybil_key), json={"max_price_per_1k": 1_000_000.0})
        assert rejected.status_code == 400  # ceiling lowered: 1e6 is no longer accepted
        accepted = world.app_client.post("/v1/hivemind/consumer/max-price", headers=_auth(sybil_key), json={"max_price_per_1k": gateway_module.MAX_PRICE_CEILING})
        assert accepted.status_code == 200
    sybil_ids = {account_id for account_id, _ in sybils}

    legacy = world.store.consumer_max_prices()  # what the gateway used to feed clear_market
    assert sybil_ids <= set(legacy)
    offers = [ProducerOffer("p", COST, 100.0)]
    legacy_price = clear_market("qwen", [ConsumerBid(a, v) for a, v in legacy.items()], offers).price_per_1k
    assert legacy_price > 50.0  # the attack is real against the unfiltered bid set

    response = world.chat(honest[0][1])
    assert response.status_code == 200, response.text
    assert response.json()["hivemind"]["price_per_1k"] < 5.0
    assert sybil_ids.isdisjoint(world.store.funded_consumer_bids(DEFAULT_MAX_TOKENS))


def test_a_funded_bid_is_capped_by_what_the_balance_can_pay(world: World) -> None:
    account_id, key = world.consumer(1.0)
    world.app_client.post("/v1/hivemind/consumer/max-price", headers=_auth(key), json={"max_price_per_1k": 900.0})

    bids = world.store.funded_consumer_bids(DEFAULT_MAX_TOKENS)

    assert bids[account_id] == pytest.approx(1.0 * 1000.0 / DEFAULT_MAX_TOKENS)
    assert bids[account_id] < 900.0


def test_parallel_requests_cannot_spend_one_balance_twice(world: World) -> None:
    """Finding 2: the reserve was checked but not held, so concurrent requests all passed and settled for free."""
    consumer_id, key = world.consumer(0.6)  # one request's reserve is about 0.51
    nested: list[httpx.Response] = []
    fired: list[bool] = []

    class Reentrant(FakeProducer):
        def handle(self, request: httpx.Request) -> httpx.Response:
            if not fired:  # fire a second request on the same balance while the first is being served
                fired.append(True)
                nested.append(world.chat(key))
            return super().handle(request)

    world.producer("prod-a.test", {"qwen": "q"}, agent=Reentrant())

    outer = world.chat(key)

    assert outer.status_code == 200, outer.text
    assert nested[0].status_code == 402
    assert nested[0].json()["error"]["code"] == "insufficient_balance"
    charge = outer.json()["hivemind"]["charged_hmt"]
    assert charge > 0.0
    assert world.store.balance(consumer_id) == pytest.approx(0.6 - charge)
    assert world.store.active_hold_count(consumer_id) == 0
    assert world.store.total_balances() == pytest.approx(world.store.total_minted(), abs=1e-9)
    assert world.store.verify_chain()


def test_hold_is_released_when_every_producer_fails(world: World) -> None:
    consumer_id, key = world.consumer(5.0)
    world.producer("prod-a.test", {"qwen": "q"}, agent=FakeProducer(explode=True))

    response = world.chat(key)

    assert response.status_code == 502
    assert world.store.balance(consumer_id) == 5.0
    assert world.store.active_hold_count(consumer_id) == 0


@pytest.mark.parametrize("limits", [{"max_completion_tokens": 1, "max_tokens": 4000}, {"max_completion_tokens": 4000, "max_tokens": 1}])
def test_conflicting_limit_fields_use_the_minimum_and_are_forwarded_as_such(world: World, limits: dict[str, int]) -> None:
    """Finding 3: billing read max_completion_tokens while the producer honoured the larger max_tokens."""
    consumer_id, key = world.consumer(10.0)
    _, _, fake = world.producer("prod-a.test", {"qwen": "q"}, agent=FakeProducer(reported_prompt=3, reported_completion=3))

    response = world.chat(key, options={"num_predict": 4000, "temperature": 0}, **limits)

    assert response.status_code == 200, response.text
    forwarded = json.loads(fake.calls[0].content)
    assert forwarded["max_tokens"] == 1 and forwarded["max_completion_tokens"] == 1
    assert forwarded["options"] == {"temperature": 0}
    assert world.store.recent_usage(consumer_id, 1)[0]["completion_tokens"] <= 1


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("bad", [None, 0, -3, "9", 1.5, True])
def test_null_or_malformed_limits_are_rejected(world: World, field: str, bad: Any) -> None:
    _, key = world.consumer(10.0)
    _, _, fake = world.producer("prod-a.test", {"qwen": "q"})

    response = world.chat(key, **{field: bad})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert fake.calls == []


def test_missing_limit_is_forwarded_and_billed_as_the_default(world: World) -> None:
    """Finding 4: with no max_tokens a producer could pad the answer and bill without bound."""
    consumer_id, key = world.consumer(1000.0)
    padding = "x" * 20_000
    _, _, fake = world.producer("prod-a.test", {"qwen": "q"}, agent=FakeProducer(answer=padding, reported_prompt=5, reported_completion=10**9))

    response = world.chat(key)

    assert response.status_code == 200, response.text
    assert json.loads(fake.calls[0].content)["max_tokens"] == DEFAULT_MAX_TOKENS
    row = world.store.recent_usage(consumer_id, 1)[0]
    assert row["completion_tokens"] == DEFAULT_MAX_TOKENS
    assert row["charge"] <= response.json()["hivemind"]["price_per_1k"] * (len(PROMPT) + DEFAULT_MAX_TOKENS) / 1000.0 + 1e-12
    assert meter_usage(10**9, 10**9, 4000, 10_000_000, None)[2] <= 2 * estimate_tokens(4000) + DEFAULT_MAX_TOKENS


@given(st.dictionaries(st.sampled_from(["max_tokens", "max_completion_tokens"]), st.integers(1, 10**6)))
def test_request_token_limit_is_the_minimum_of_the_present_fields(fields: dict[str, int]) -> None:
    limit = request_token_limit(fields)

    assert limit == (min(fields.values()) if fields else DEFAULT_MAX_TOKENS)
    assert all(limit <= v for v in fields.values())


@pytest.mark.parametrize("extra", [{"n": 64}, {"best_of": 8}, {"n": 2, "max_tokens": 100}])
def test_multiple_choices_are_rejected(world: World, extra: dict[str, Any]) -> None:
    """Finding 6: n>1 multiplied the producer's work while the billing cap stayed at max_tokens."""
    _, key = world.consumer(10.0)
    _, _, fake = world.producer("prod-a.test", {"qwen": "q"})

    response = world.chat(key, **extra)

    assert response.status_code == 400
    assert fake.calls == []
    assert world.chat(key, n=1).status_code == 200


def test_oversized_producer_response_is_a_failure_without_charge(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding 5: the whole upstream body used to be buffered with no cap."""
    monkeypatch.setattr(gateway_module, "MAX_PRODUCER_RESPONSE_BYTES", 200)
    consumer_id, key = world.consumer(10.0)
    world.producer("prod-a.test", {"qwen": "q"}, agent=FakeProducer(answer="y" * 5000))

    response = world.chat(key)

    assert response.status_code == 502
    assert world.store.balance(consumer_id) == 10.0
    assert world.store.recent_usage(consumer_id, 1) == []


def test_slow_producer_misses_the_wall_clock_deadline(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway_module, "PRODUCER_DEADLINE_S", -1.0)
    consumer_id, key = world.consumer(10.0)
    world.producer("prod-a.test", {"qwen": "q"})

    assert world.chat(key).status_code == 502
    assert world.store.balance(consumer_id) == 10.0


def test_streaming_line_without_newline_is_cut_off_and_not_billed(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway_module, "MAX_SSE_LINE_BYTES", 100)
    consumer_id, key = world.consumer(10.0)
    world.producer("prod-a.test", {"qwen": "q"}, agent=FakeProducer(answer="z" * 3000))

    with world.app_client.stream("POST", "/v1/chat/completions", headers=_auth(key), json={"model": "hivemind/qwen", "messages": [{"role": "user", "content": PROMPT}], "stream": True}) as response:
        text = "".join(response.iter_text())

    assert "upstream_failure" in text
    assert world.store.recent_usage(consumer_id, 1)[0]["billed_tokens"] == 0
    assert world.store.balance(consumer_id) == 10.0
    assert world.store.active_hold_count(consumer_id) == 0


@given(st.lists(st.text(alphabet=st.characters(blacklist_characters="\r\n", blacklist_categories=("Cs",)), max_size=40), max_size=12))
def test_limited_lines_roundtrips_lines_under_the_cap(lines: list[str]) -> None:
    payload = "".join(line + "\n" for line in lines).encode()
    response = httpx.Response(200, content=payload)

    got = list(limited_lines(response, max_bytes=len(payload) + 1, max_line=10_000, deadline=time.monotonic() + 60))

    assert got == lines
    with pytest.raises(UpstreamLimitError):
        list(limited_lines(httpx.Response(200, content=payload + b"q" * 50), max_bytes=len(payload) + 1, max_line=10_000, deadline=time.monotonic() + 60))
    with pytest.raises(UpstreamLimitError):
        read_limited(httpx.Response(200, content=b"abcdef"), max_bytes=5, deadline=time.monotonic() + 60)
    assert read_limited(httpx.Response(200, content=b"abcdef"), max_bytes=6, deadline=time.monotonic() + 60) == b"abcdef"


def test_empty_choices_are_a_failure_and_fail_over(world: World) -> None:
    """Finding 11: {"usage": ...} with no choices used to be settled as a completed request."""
    consumer_id, key = world.consumer(10.0)

    class Hollow(FakeProducer):
        def completion(self, model: str) -> dict[str, Any]:
            return {"usage": {"prompt_tokens": 999_999_999, "completion_tokens": 0}}

    world.producer("prod-bad.test", {"qwen": "q"}, capacity=1000.0, agent=Hollow())
    good_id, _, _ = world.producer("prod-good.test", {"qwen": "q"}, capacity=1.0)

    response = world.chat(key)

    assert response.status_code == 200, response.text
    assert response.json()["hivemind"]["producer_id"] == good_id
    assert world.store.recent_usage(consumer_id, 1)[0]["producer_id"] == good_id

    assert has_completion_choice({"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]})
    assert not has_completion_choice({"choices": []})
    assert not has_completion_choice({"choices": [{"message": {"content": "x"}, "finish_reason": None}]})
    assert not has_completion_choice({"usage": {}})


def test_streaming_without_any_choice_chunk_is_not_billed(world: World) -> None:
    consumer_id, key = world.consumer(10.0)

    class SilentStream(FakeProducer):
        def handle(self, request: httpx.Request) -> httpx.Response:
            body = "data: " + json.dumps({"id": "c", "choices": [], "usage": {"prompt_tokens": 10**9, "completion_tokens": 0}}) + "\n\ndata: [DONE]\n\n"
            return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})

    world.producer("prod-a.test", {"qwen": "q"}, agent=SilentStream())

    with world.app_client.stream("POST", "/v1/chat/completions", headers=_auth(key), json={"model": "hivemind/qwen", "messages": [{"role": "user", "content": PROMPT}], "stream": True}) as response:
        "".join(response.iter_text())

    assert world.store.recent_usage(consumer_id, 1)[0]["billed_tokens"] == 0
    assert world.store.balance(consumer_id) == 10.0


def test_declared_capacity_is_clamped_for_round_robin() -> None:
    """Finding 7: a self-declared 1e12 capacity took ~100% of the traffic."""
    world = _build_world(max_producer_capacity_1k=10.0)
    _, key = world.consumer(1000.0)
    greedy_id, _, _ = world.producer("greedy.test", {"qwen": "q"}, capacity=1e12)
    honest_id, _, _ = world.producer("honest.test", {"qwen": "q"}, capacity=10.0)

    served = [world.chat(key).json()["hivemind"]["producer_id"] for _ in range(20)]

    assert served.count(honest_id) >= 8
    assert served.count(greedy_id) <= 12


def test_auto_model_prefers_more_distinct_producers_over_declared_supply(world: World) -> None:
    _, key = world.consumer(100.0)
    world.producer("whale.test", {"whale-model": "w"}, capacity=1e11)
    world.producer("a.test", {"crowd-model": "c"}, capacity=1.0)
    world.producer("b.test", {"crowd-model": "c"}, capacity=1.0)

    response = world.chat(key, model="hivemind/auto")

    assert response.status_code == 200, response.text
    assert response.json()["model"] == "hivemind/crowd-model"


def test_market_is_cached_per_epoch(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding 8: certify() is quadratic in bidders and ran on every GET."""
    calls: list[int] = []
    real = gateway_module.certify

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(gateway_module, "certify", counting)
    _, key = world.consumer(10.0)
    world.consumer(10.0)
    world.producer("prod-a.test", {"qwen": "q"})

    first = world.app_client.get("/v1/hivemind/market", params={"model": "qwen"}, headers=_auth(key))
    second = world.app_client.get("/v1/hivemind/market", params={"model": "qwen"}, headers=_auth(key))
    world.clock.now += gateway_module.MARKET_CACHE_EPOCH_S + 1
    third = world.app_client.get("/v1/hivemind/market", params={"model": "qwen"}, headers=_auth(key))

    assert first.status_code == second.status_code == third.status_code == 200
    assert first.json() == second.json()
    assert len(calls) == 2


def test_oversized_bodies_are_refused_before_authentication_or_parsing(world: World) -> None:
    """Finding 9: the size check ran after FastAPI had already parsed the body, and trusted Content-Length."""
    limit = gateway_module.MAX_BODY_BYTES

    def chunks() -> Any:
        for _ in range(5):
            yield b"[" * (limit // 4)

    chunked = world.app_client.post("/v1/chat/completions", content=chunks())  # no Content-Length, no key
    declared = world.app_client.post("/v1/chat/completions", content=b"x" * (limit + 1))
    small_anonymous = world.app_client.post("/v1/chat/completions", content=b"{}")

    assert chunked.status_code == 413 and chunked.json()["error"]["code"] == "request_too_large"
    assert declared.status_code == 413
    assert small_anonymous.status_code == 401


def test_registration_limit_keys_on_the_trusted_forwarded_hop() -> None:
    """Finding 10: behind Cloud Run every caller shared the proxy's address; a forged left-most XFF must not help either."""
    world = _build_world(trusted_proxy_hops=1)

    def register(n: int, real_ip: str, forged: str) -> int:
        return world.app_client.post(
            "/v1/hivemind/register", headers={"X-Forwarded-For": f"{forged}, {real_ip}"}, json={"handle": f"sybil{n}@x.test", "role": "consumer"}
        ).status_code

    attacker = [register(n, "203.0.113.7", f"198.51.100.{n}") for n in range(12)]
    bystander = register(99, "203.0.113.99", "203.0.113.7")

    assert attacker[:10] == [200] * 10
    assert attacker[10:] == [429, 429]
    assert bystander == 200


def test_deploy_files_trust_exactly_the_proxy_hops_not_a_wildcard() -> None:
    from pathlib import Path

    deploy = Path(__file__).resolve().parent.parent / "deploy" / "gcp_openrouter"
    commands = [line for line in (deploy / "Dockerfile").read_text().splitlines() if not line.lstrip().startswith("#")]
    text = "\n".join(commands)

    assert "--forwarded-allow-ips" not in text and "--proxy-headers" not in text
    assert "HIVEMIND_ROUTER_TRUSTED_PROXY_HOPS=1" in text
    assert "trusted_proxy_hops=hops" in (deploy / "server.py").read_text()


@pytest.mark.parametrize("address", [
    "64:ff9b::a9fe:a9fe",   # NAT64 of 169.254.169.254 (cloud metadata)
    "64:ff9b::7f00:1",      # NAT64 of 127.0.0.1
    "::127.0.0.1",          # deprecated IPv4-compatible loopback
    "::10.0.0.1",           # IPv4-compatible private
    "::ffff:10.0.0.1",      # IPv4-mapped private
    "2002:7f00:1::",        # 6to4 embedding 127.0.0.1
    "2002:0a00:0001::",     # 6to4 embedding 10.0.0.1
    "64:ff9b:1::1",         # local-use NAT64 prefix
])
def test_ssrf_guard_rejects_ipv6_forms_that_smuggle_private_ipv4(address: str) -> None:
    assert gateway_module._ip_is_global(address) is False
    assert producer_url_problem(f"https://[{address}]/", allow_private=False) == "endpoint_url resolves to a non-public address"


@pytest.mark.parametrize("address", ["8.8.8.8", "2606:4700:4700::1111", "64:ff9b::808:808", "::ffff:8.8.8.8"])
def test_ssrf_guard_still_accepts_public_addresses_including_public_nat64(address: str) -> None:
    assert gateway_module._ip_is_global(address) is True


def test_only_allow_listed_request_fields_reach_the_producer(world: World) -> None:
    _, key = world.consumer(1000.0)
    _, _, fake = world.producer("prod-a.test", {"qwen": "q"})
    smuggled = {"documents": ["x" * 5000], "chat_template": "{{ evil }}", "functions": [{"name": "f"}], "response_format": {"type": "json_schema"}, "temperature": 0.2}

    response = world.chat(key, **smuggled)

    assert response.status_code == 200, response.text
    forwarded = json.loads(fake.calls[0].content)
    assert forwarded["temperature"] == 0.2 and forwarded["model"] == "q"
    for name in ("documents", "chat_template", "functions", "response_format"):
        assert name not in forwarded


def test_container_requirements_cover_every_third_party_import_of_the_router() -> None:
    """A missing dependency only shows up at container start (it did once: cryptography), so check the files statically."""
    import ast
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    files = [*(root / "src/hivemind/router").glob("*.py"), root / "src/hivemind/economy/compute_coin.py",
             root / "src/hivemind/security/hivemind_firewall.py", root / "src/hivemind/security/_exceptions.py", root / "deploy/gcp_openrouter/server.py"]
    imported: set[str] = set()
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
    third_party = {name for name in imported if name not in sys.stdlib_module_names and name != "hivemind"}
    requirements = (root / "deploy/gcp_openrouter/requirements.txt").read_text(encoding="utf-8").lower()
    declared = {line.split("[")[0].split(">")[0].split("=")[0].strip() for line in requirements.splitlines() if line.strip()}
    provided_by = {"starlette": "fastapi"}  # starlette ships with fastapi
    missing = {name for name in third_party if provided_by.get(name, name) not in declared}
    assert third_party >= {"fastapi", "httpx", "cryptography"}, third_party
    assert not missing, missing


def test_health_is_served_on_both_paths(world: World) -> None:
    for path in ("/health", "/healthz"):
        response = world.app_client.get(path)
        assert response.status_code == 200 and response.json() == {"status": "ok"}


def test_catalog_is_public_and_lists_the_moe_tiers(world: World) -> None:
    response = world.app_client.get("/v1/hivemind/catalog")  # no key
    assert response.status_code == 200
    tiers = {t["tier_id"]: t for t in response.json()["tiers"]}
    assert tiers["moe-coder"]["moe"] is True and tiers["coder"]["moe"] is False
    assert tiers["coder-lite"]["min_vram_gb"] < tiers["moe-coder"]["min_vram_gb"]


def _beat_with_token(world: World, key: str, host: str, token: str | None) -> httpx.Response:
    body: dict[str, Any] = {"endpoint_url": f"https://{host}", "models": [{"id": "qwen", "local_name": "q"}], "cost_per_1k": COST, "capacity_1k": 100.0}
    if token is not None:
        body["endpoint_token"] = token
    return world.app_client.post("/v1/hivemind/producer/heartbeat", headers={"Authorization": f"Bearer {key}"}, json=body)


def test_endpoint_token_is_presented_to_the_producer_and_never_echoed(world: World) -> None:
    _, key = world.register("producer")
    fake = FakeProducer()
    world.producers["gpu.test"] = fake
    token = "tok-" + "x" * 30
    assert _beat_with_token(world, key, "gpu.test", token).status_code == 200
    _, consumer_key = world.consumer(100.0)

    reply = world.chat(consumer_key)

    assert reply.status_code == 200, reply.text
    assert fake.calls[0].headers["authorization"] == f"Bearer {token}"
    me = world.app_client.get("/v1/hivemind/me", headers={"Authorization": f"Bearer {key}"})
    market = world.app_client.get("/v1/hivemind/market?model=hivemind/qwen", headers={"Authorization": f"Bearer {consumer_key}"})
    assert token not in reply.text and token not in me.text and token not in market.text


def test_no_endpoint_token_means_no_authorization_header_and_short_tokens_are_rejected(world: World) -> None:
    _, key = world.register("producer")
    fake = FakeProducer()
    world.producers["gpu.test"] = fake
    assert _beat_with_token(world, key, "gpu.test", None).status_code == 200
    _, consumer_key = world.consumer(100.0)
    assert world.chat(consumer_key).status_code == 200
    assert "authorization" not in fake.calls[0].headers
    assert _beat_with_token(world, key, "gpu.test", "short").status_code in (400, 422)
    assert _beat_with_token(world, key, "gpu.test", "has space " + "y" * 20).status_code in (400, 422)
