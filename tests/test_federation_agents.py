# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Signed task agents for co-training over AMCP: nothing is run, aggregated or trusted before the checks pass, and one bad participant never stops a round."""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import pytest
import torch

from hivemind.swarm import federation_agents as fed
from hivemind.swarm.amcp_kafka_campaign_bus import KAFKA_TOPICS
from hivemind.swarm.amcp_kafka_transport import (
    AMCPConsumer,
    AMCPPublisher,
    EnvelopeVerifier,
    HMACKeyring,
    InMemoryTransport,
    SchemaError,
    validate_event_dict,
)

KEY = bytes(range(32))
OTHER = bytes(range(50, 82))


def ring(key: bytes = KEY) -> HMACKeyring:
    return HMACKeyring(keys={"k1": key}, active_kid="k1")


def state(scale: float = 1.0) -> dict[str, torch.Tensor]:
    return {"a.lora_A": torch.arange(6, dtype=torch.float32).reshape(2, 3) * scale, "b.lora_B": torch.ones(4) * scale}


def zero_like(s: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: torch.zeros_like(v) for k, v in s.items()}


PIN, BASE = "p" * 64, "b" * 64


def offer_payload(**kw) -> dict:  # noqa: ANN003
    g = state()
    args = dict(run_id="run1", rnd=3, sender_id="coord", global_state=g, base_sha256=BASE, pin=PIN, inner_steps=40, deadline_unix=time.time() + 60, quorum=1)
    args.update(kw)
    return fed.build_offer(**args).payload


# ----------------------------------------------------------------------------------------------------------------------- pins and hashes


def test_the_code_pin_changes_with_one_byte_and_with_the_file_name(tmp_path: Path) -> None:
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_text("x = 1\n")
    b.write_text("y = 2\n")
    base = fed.code_pin([a, b])
    assert fed.code_pin([a, b]) == base and fed.code_pin([b, a]) != base
    a.write_text("x = 2\n")
    assert fed.code_pin([a, b]) != base


def test_a_windows_checkout_with_crlf_line_endings_pins_the_same_as_a_linux_one(tmp_path: Path) -> None:
    lf, crlf = tmp_path / "lf", tmp_path / "crlf"
    lf.mkdir()
    crlf.mkdir()
    (lf / "a.py").write_bytes(b"x = 1\ny = 2\n")
    (crlf / "a.py").write_bytes(b"x = 1\r\ny = 2\r\n")
    assert fed.code_pin([lf / "a.py"]) == fed.code_pin([crlf / "a.py"])


def test_the_tensor_hash_ignores_insertion_order_and_sees_every_value() -> None:
    s = state()
    assert fed.tensors_sha256(s) == fed.tensors_sha256(dict(reversed(list(s.items()))))
    t = state()
    t["b.lora_B"][2] += 1e-6
    assert fed.tensors_sha256(t) != fed.tensors_sha256(s)


def test_payload_codec_roundtrips_and_never_executes_code() -> None:
    s = state()
    back = fed.decode_tensors(fed.encode_tensors(s))
    assert all(torch.equal(back[k], s[k]) for k in s)
    with pytest.raises(Exception):  # noqa: B017, PT011
        import base64
        import io

        class Evil:
            def __reduce__(self):  # noqa: ANN204
                return (print, ("pwned",))

        buf = io.BytesIO()
        torch.save({"w": Evil()}, buf)
        fed.decode_tensors(base64.b64encode(buf.getvalue()).decode())


# ----------------------------------------------------------------------------------------------------------------------- the schema and the bus


def test_the_federation_topic_exists_and_the_schema_refuses_an_unknown_algorithm() -> None:
    topic = KAFKA_TOPICS["federation"]
    ev = fed.build_offer(run_id="r", rnd=0, sender_id="c", global_state=state(), base_sha256=BASE, pin=PIN, inner_steps=1, deadline_unix=time.time() + 5, quorum=1)
    validate_event_dict(dataclasses.asdict(ev), topic)
    bad = dataclasses.asdict(ev)
    bad["payload"] = dict(bad["payload"], algo="anything-else")
    with pytest.raises(SchemaError):
        validate_event_dict(bad, topic)
    other = dataclasses.asdict(ev)
    other["event_type"] = "io.hivemind.amcp.fed.run_this_code"
    with pytest.raises(SchemaError):
        validate_event_dict(other, topic)


def test_a_signed_offer_travels_and_a_message_signed_with_a_foreign_key_is_ignored() -> None:
    t = InMemoryTransport(num_partitions=2)
    got: list = []
    con = AMCPConsumer(t, EnvelopeVerifier(ring()), topics=["federation"])
    con.register(fed.OFFER, got.append)
    mine = fed.build_offer(run_id="r", rnd=0, sender_id="c", global_state=state(), base_sha256=BASE, pin=PIN, inner_steps=1, deadline_unix=time.time() + 30, quorum=1)
    AMCPPublisher(t, ring()).publish(mine)
    AMCPPublisher(t, ring(OTHER)).publish(mine)  # an outsider who does not hold the keyring
    for _ in range(3):
        con.run_once()
    assert len(got) == 1 and got[0].event.payload["run_id"] == "r"


# ----------------------------------------------------------------------------------------------------------------------- learner side: verify before running


def test_a_good_offer_yields_the_global_state() -> None:
    p = offer_payload()
    s = fed.check_offer(p, my_pin=PIN, my_base_sha256=BASE, run_id="run1")
    assert fed.tensors_sha256(s) == fed.tensors_sha256(state())


@pytest.mark.parametrize("change,match", [
    (dict(my_pin="q" * 64), "different code version"),
    (dict(my_base_sha256="c" * 64), "base model"),
    (dict(run_id="other"), "this host joined"),
    (dict(now=time.time() + 3600), "deadline has passed"),
])
def test_an_offer_is_refused_before_anything_runs(change: dict, match: str) -> None:
    kw = dict(my_pin=PIN, my_base_sha256=BASE, run_id="run1")
    kw.update(change)
    with pytest.raises(fed.OfferRefused, match=match):
        fed.check_offer(offer_payload(), **kw)


def test_an_offer_whose_state_does_not_match_its_hash_is_refused() -> None:
    p = offer_payload()
    p["global_b64"] = fed.encode_tensors(state(2.0))  # a different state under the original hash
    with pytest.raises(fed.OfferRefused, match="does not hash"):
        fed.check_offer(p, my_pin=PIN, my_base_sha256=BASE)


def test_a_state_too_big_for_one_envelope_is_refused_with_an_actionable_message() -> None:
    big = {"w": torch.randn(40_000)}  # 160 kB of float32
    with pytest.raises(fed.FederationError, match="chunking or an object store"):
        fed.build_offer(run_id="r", rnd=0, sender_id="c", global_state=big, base_sha256=BASE, pin=PIN, inner_steps=1, deadline_unix=time.time() + 5, quorum=1)


# ----------------------------------------------------------------------------------------------------------------------- coordinator side: check before aggregating


def book(**kw) -> fed.RoundBook:  # noqa: ANN003
    g = state()
    args = dict(run_id="run1", rnd=3, allowed=["l0", "l1"], quorum=1, deadline_unix=time.time() + 60, global_sha256=fed.tensors_sha256(g), pin=PIN, max_delta_l2=100.0, template=g)
    args.update(kw)
    return fed.RoundBook(**args)


def result_payload(who: str = "l0", delta: dict | None = None, **kw) -> dict:  # noqa: ANN003
    g = state()
    d = delta if delta is not None else {k: v * 0.01 for k, v in g.items()}
    args = dict(run_id="run1", rnd=3, learner_id=who, host="h", tokens=100, delta=d, delta_l2=1.0, eval_loss_global=4.0, pin=PIN, global_sha256=fed.tensors_sha256(g))
    args.update(kw)
    return fed.build_result(**args).payload


def test_a_good_result_is_accepted_and_a_duplicate_is_not() -> None:
    b = book()
    assert b.offer_result(result_payload("l0")) and "l0" in b.accepted
    assert not b.offer_result(result_payload("l0")) and b.rejected[-1][1] == "duplicate result for this round"


@pytest.mark.parametrize("mutate,match", [
    (dict(run_id="other"), "another run"),
    (dict(rnd=2), "open round"),
    (dict(who="mallory"), "allowed list"),
    (dict(global_sha256="0" * 64), "different global state"),
    (dict(pin="z" * 64), "different code version"),
])
def test_results_that_do_not_belong_to_this_round_are_rejected_with_the_reason(mutate: dict, match: str) -> None:
    b = book()
    assert not b.offer_result(result_payload(**mutate))
    assert match in b.rejected[-1][1] and not b.accepted


def test_a_poisoned_delta_cannot_enter_the_round() -> None:
    g = state()
    cases = {
        "NaN": {**{k: v * 0.01 for k, v in g.items()}, "b.lora_B": torch.tensor([float("nan")] * 4)},
        "shape": {**{k: v * 0.01 for k, v in g.items()}, "b.lora_B": torch.ones(5)},
        "norm": {k: v * 1e6 for k, v in g.items()},
        "names": {"x": torch.ones(1)},
    }
    for label, bad in cases.items():
        b = book()
        assert not b.offer_result(result_payload("l0", delta=bad)), label
        assert b.rejected and not b.accepted, label
    b = book()
    assert not b.offer_result({"run_id": "run1", "round": 3, "learner_id": "l0", "global_sha256": fed.tensors_sha256(g), "script_sha256": PIN, "tokens": 5, "delta_b64": "not base64 !!"})
    assert "unreadable" in b.rejected[-1][1]


def test_one_bad_participant_does_not_stop_the_round() -> None:
    b = book()
    b.offer_result(result_payload("mallory"))
    assert b.offer_result(result_payload("l1")) and set(b.accepted) == {"l1"}


def test_round_status_complete_open_quorum_failed() -> None:
    now = time.time()
    b = book(deadline_unix=now + 60, quorum=1)
    assert b.status(now) == "open"
    b.offer_result(result_payload("l0"))
    assert b.status(now) == "open" and b.status(now + 61) == "quorum"
    b.offer_result(result_payload("l1"))
    assert b.status(now) == "complete"
    assert book(deadline_unix=now + 60, quorum=2).status(now + 61) == "failed"


# ----------------------------------------------------------------------------------------------------------------------- the remaining paths


def test_a_run_id_with_unsafe_characters_is_refused() -> None:
    with pytest.raises(fed.FederationError, match="run_id"):
        fed.build_offer(run_id="bad id; rm", rnd=0, sender_id="c", global_state=state(), base_sha256=BASE, pin=PIN, inner_steps=1, deadline_unix=time.time() + 5, quorum=1)


def test_an_offer_for_an_unknown_algorithm_is_refused() -> None:
    p = offer_payload()
    p["algo"] = "something_else"
    with pytest.raises(fed.OfferRefused, match="unknown algorithm"):
        fed.check_offer(p, my_pin=PIN, my_base_sha256=BASE)


def test_a_result_with_no_tokens_is_rejected() -> None:
    b = book()
    assert not b.offer_result(result_payload("l0", tokens=0)) and "no tokens" in b.rejected[-1][1]


def test_claim_and_abort_events_pass_the_schema_and_the_result_can_carry_peak_vram() -> None:
    topic = KAFKA_TOPICS["federation"]
    for ev in (fed.build_claim(run_id="r", rnd=1, learner_id="l0", host="h"), fed.build_abort(run_id="r", rnd=1, sender_id="c", reason="x" * 500)):
        validate_event_dict(dataclasses.asdict(ev), topic)
    assert len(fed.build_abort(run_id="r", rnd=1, sender_id="c", reason="x" * 500).payload["reason"]) == 128
    assert result_payload("l0", vram_peak_mib=27.75)["vram_peak_mib"] == 27.75
    assert "vram_peak_mib" not in result_payload("l0")


def test_a_round_counts_only_claims_that_belong_to_it() -> None:
    b = book()
    b.claim({"run_id": "run1", "round": 3, "learner_id": "l0"})
    b.claim({"run_id": "run1", "round": 2, "learner_id": "l1"})  # another round
    b.claim({"run_id": "other", "round": 3, "learner_id": "l1"})  # another run
    b.claim({"run_id": "run1", "round": 3, "learner_id": "mallory"})  # not on the allowed list
    assert b.claims == {"l0"}
