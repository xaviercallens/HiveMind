# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Co-training over AMCP, end to end on CPU with an in-process fan-out bus: exactness against the single-process baseline, a missing learner, a poisoned delta, a stale code pin."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import torch

pytest.importorskip("peft")
pytest.importorskip("transformers")

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
import e8_amcp_round as r  # noqa: E402
import e8_qlora_moe_round as q  # noqa: E402

from hivemind.swarm import federation_agents as fed  # noqa: E402
from hivemind.swarm.amcp_kafka_transport import HMACKeyring, TransportRecord  # noqa: E402

RING = HMACKeyring(keys={"k1": bytes(range(32))}, active_kid="k1")


class Hub:
    """One shared log; every participant gets its own view with its own read position, like separate Kafka consumer groups."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.records: list[TransportRecord] = []

    def view(self) -> "View":
        return View(self)


class View:
    def __init__(self, hub: Hub) -> None:
        self.hub, self.pos, self.topics = hub, 0, ()

    def publish(self, topic: str, key: bytes, value: bytes) -> dict:
        with self.hub.lock:
            off = len(self.hub.records)
            self.hub.records.append(TransportRecord(topic, 0, off, bytes(key), bytes(value)))
        return {"topic": topic, "partition": 0, "offset": off}

    def subscribe(self, topics) -> None:  # noqa: ANN001
        self.topics = tuple(topics)

    def poll(self, max_records: int, timeout_s: float) -> list[TransportRecord]:
        out: list[TransportRecord] = []
        with self.hub.lock:
            while self.pos < len(self.hub.records) and len(out) < max_records:
                rec = self.hub.records[self.pos]
                self.pos += 1
                if rec.topic in self.topics:
                    out.append(rec)
        if not out:
            time.sleep(min(0.05, timeout_s))
        return out

    def commit(self, record: TransportRecord) -> None:
        pass

    def seek(self, topic: str, partition: int, offset: int) -> None:
        self.pos = offset

    def committed_offset(self, topic: str, partition: int):  # noqa: ANN201
        return None

    def close(self) -> None:
        pass


def parse(*argv: str) -> argparse.Namespace:
    return r.build_parser().parse_args(list(argv))


COORD = ["--inner-lr", "2e-2", "--batch", "8", "--shard-size", "64", "--seed", "1"]


def coordinator_args(tmp: Path, run: str, learners: list[str], rounds: int = 2, quorum: int = 1, deadline: float = 120.0, steps: int = 4) -> argparse.Namespace:
    return parse("coordinator", "--run-id", run, "--learners", *learners, "--rounds", str(rounds), "--quorum", str(quorum), "--round-deadline-s", str(deadline), "--inner-steps", str(steps), *COORD, "--out", str(tmp / f"{run}.json"))


def learner_args(run: str, who: str, index: int, **extra: str) -> argparse.Namespace:
    argv = ["learner", "--run-id", run, "--learner-id", who, "--index", str(index), "--device", "cpu", "--quant", "none", "--governor-log", "", *extra.get("argv", "").split()]
    return parse(*argv)


def run_threads(hub: Hub, coord: argparse.Namespace, learners: list[argparse.Namespace], timeout: float = 240.0) -> dict:
    rcs: dict = {}
    ths = [threading.Thread(target=lambda: rcs.__setitem__("coord", r.run_coordinator(coord, hub.view(), RING)), daemon=True)]
    for a in learners:
        ths.append(threading.Thread(target=lambda a=a: rcs.__setitem__(a.learner_id, r.run_learner(a, hub.view(), RING)), daemon=True))
    for t in ths:
        t.start()
    for t in ths:
        t.join(timeout)
    return rcs


def baseline_final(tmp: Path, learners: int, rounds: int) -> float:
    a = argparse.Namespace(learners=learners, rounds=rounds, quant="none", device="cpu", shard_size=64, shuffle_labels=False, min_free_mib=4300.0, need_mib=600.0, out=str(tmp / "base.json"),
                           inner_steps=4, inner_lr=2e-2, batch=8, outer_lr=0.7, outer_mu=0.65, seed=1)
    assert q.run_baseline(a) == 0
    return json.loads((tmp / "base.json").read_text())["final_eval_loss_mean"]


# ----------------------------------------------------------------------------------------------------------------------- end to end


def test_two_learners_over_amcp_match_the_single_process_baseline(tmp_path: Path) -> None:
    hub = Hub()
    rcs = run_threads(hub, coordinator_args(tmp_path, "rt1", ["l0", "l1"]), [learner_args("rt1", "l0", 0), learner_args("rt1", "l1", 1)])
    assert rcs == {"coord": 0, "l0": 0, "l1": 0}, rcs
    rep = json.loads((tmp_path / "rt1.json").read_text())
    assert rep["completed"] and [x["status"] for x in rep["rounds"]] == ["complete", "complete"] and rep["hosts_seen"]
    assert abs(rep["final_eval_loss"] - baseline_final(tmp_path, 2, 2)) < 1e-4, (rep["final_eval_loss"],)


def test_a_missing_learner_does_not_stop_the_run_the_quorum_closes_the_round(tmp_path: Path) -> None:
    hub = Hub()
    rcs = run_threads(hub, coordinator_args(tmp_path, "rt2", ["l0", "l1"], rounds=3, quorum=1, deadline=4.0, steps=25), [learner_args("rt2", "l0", 0)])  # l1 never shows up
    assert rcs == {"coord": 0, "l0": 0}, rcs
    rep = json.loads((tmp_path / "rt2.json").read_text())
    assert [x["status"] for x in rep["rounds"]] == ["quorum"] * 3 and all(x["accepted"] == ["l0"] for x in rep["rounds"])
    # The held-out set mixes both learners' data; l1 never contributed, so its half is NOT learned (no carry of late work): the loss on the mixture may well rise. That cost is real
    # and is documented, so this test asserts the protocol guarantees (rounds close at the quorum, the run finishes, the loss stays finite), not an improvement.
    assert rep["completed"] and rep["final_eval_loss"] == rep["final_eval_loss"] and rep["final_eval_loss"] < 50


def test_a_poisoned_delta_from_a_keyholder_is_rejected_and_the_honest_learner_carries_the_round(tmp_path: Path) -> None:
    hub = Hub()
    stop = threading.Event()

    def evil() -> None:
        bus = r.Bus(hub.view(), RING, "rt3")
        pin = fed.code_pin(r.PIN_FILES)
        while not stop.is_set():
            for ev in bus.poll():
                if ev.event_type == fed.OFFER:
                    g = fed.decode_tensors(ev.payload["global_b64"])
                    bad = {k: torch.full_like(v, float("nan")) for k, v in g.items()}
                    bus.publish(fed.build_result(run_id="rt3", rnd=int(ev.payload["round"]), learner_id="evil", host="evil-host", tokens=10, delta=bad, delta_l2=1.0, eval_loss_global=4.0,
                                                 pin=pin, global_sha256=ev.payload["global_sha256"]))
                if ev.event_type == fed.ABORT:
                    return

    t = threading.Thread(target=evil, daemon=True)
    t.start()
    rcs = run_threads(hub, coordinator_args(tmp_path, "rt3", ["l0", "evil"], rounds=2, quorum=1, deadline=5.0, steps=25), [learner_args("rt3", "l0", 0)])
    stop.set()
    assert rcs == {"coord": 0, "l0": 0}, rcs
    rep = json.loads((tmp_path / "rt3.json").read_text())
    assert all(x["accepted"] == ["l0"] for x in rep["rounds"])
    assert any(who == "evil" and "NaN" in why for x in rep["rounds"] for who, why in x["rejected"])
    assert rep["completed"] and rep["final_eval_loss"] == rep["final_eval_loss"] and rep["final_eval_loss"] < 50  # finite: the NaN never reached the global state


def test_a_learner_whose_code_differs_refuses_the_task_and_sends_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = Hub()
    pin_there = fed.code_pin(r.PIN_FILES)
    base = q.build_base_state(1)
    pub = r.Bus(hub.view(), RING, "rt4")
    pub.publish(fed.build_offer(run_id="rt4", rnd=0, sender_id="coordinator", global_state=q.initial_lora(base, 1), base_sha256=fed.tensors_sha256(base), pin="f" * 64, inner_steps=2,
                                deadline_unix=time.time() + 60, quorum=1, params={"seed": 1, "shard_size": 64, "learners": 1}))
    assert pin_there != "f" * 64
    rc = r.run_learner(parse("learner", "--run-id", "rt4", "--learner-id", "l0", "--device", "cpu", "--governor-log", "", "--idle-exit-s", "2"), hub.view(), RING)
    assert rc == 0
    types = [json.loads(rec.value)["type"] for rec in hub.records]
    assert fed.RESULT not in types and fed.CLAIM not in types


# ----------------------------------------------------------------------------------------------------------------------- gating


def test_the_night_window_is_22_to_07_paris() -> None:
    z = ZoneInfo("Europe/Paris")
    for hour, expect in ((21, False), (22, True), (23, True), (0, True), (6, True), (7, False), (12, False)):
        assert r.in_night_window(datetime(2026, 11, 1, hour, 30, tzinfo=z)) is expect, hour


def test_the_governor_log_says_whether_inference_is_in_flight(tmp_path: Path) -> None:
    log = tmp_path / "gov.jsonl"
    assert r.governor_busy(str(log)) is None  # no log: the check does not apply
    log.write_text('{"event":"snapshot","in_flight":0}\n{"event":"snapshot","in_flight":2}\n')
    assert r.governor_busy(str(log)) is True
    log.write_text('{"event":"snapshot","in_flight":2}\n{"event":"snapshot","in_flight":0}\n')
    assert r.governor_busy(str(log)) is False
    log.write_text("not json\n")
    assert r.governor_busy(str(log)) is None


def test_the_gate_never_lets_a_learner_claim_when_it_must_not(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = tmp_path / "gov.jsonl"
    log.write_text('{"in_flight":1}\n')
    z = ZoneInfo("Europe/Paris")
    a = parse("learner", "--run-id", "x", "--learner-id", "l0", "--night-only", "--governor-log", str(log))
    assert r.gate(a, "cpu", datetime(2026, 11, 1, 12, 0, tzinfo=z)) == (False, "outside the night window")
    assert r.gate(a, "cpu", datetime(2026, 11, 1, 23, 0, tzinfo=z))[1] == "inference in flight (governor)"
    log.write_text('{"in_flight":0}\n')
    assert r.gate(a, "cpu", datetime(2026, 11, 1, 23, 0, tzinfo=z)) == (True, "ok")
    monkeypatch.setattr(q, "vram_guard", lambda *a, **k: (_ for _ in ()).throw(q.GuardError("4,400 MiB free")))
    ok, why = r.gate(a, "cuda", datetime(2026, 11, 1, 23, 0, tzinfo=z))
    assert not ok and why.startswith("vram:")


def test_a_learner_that_declined_an_offer_takes_the_re_offer_of_the_same_round(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The first night run stopped after 13 rounds: inference was busy for one offer, the learners marked that round done and ignored every re-offer of it."""
    calls = {"n": 0}

    def busy_once(args, device, now=None):  # noqa: ANN001, ANN202
        calls["n"] += 1
        return (False, "inference in flight (governor)") if calls["n"] == 1 else (True, "ok")

    monkeypatch.setattr(r, "gate", busy_once)
    hub = Hub()
    rcs = run_threads(hub, coordinator_args(tmp_path, "rt5", ["l0"], rounds=2, quorum=1, deadline=4.0), [learner_args("rt5", "l0", 0)])
    assert rcs == {"coord": 0, "l0": 0}, rcs
    rep = json.loads((tmp_path / "rt5.json").read_text())
    assert rep["completed"] and rep["rounds"][0]["retries"] == 1 and rep["rounds"][0]["accepted"] == ["l0"]
    assert calls["n"] >= 2


def test_a_result_delivered_before_a_re_offer_still_counts_after_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cross-machine run B: the VM answered round 0, the T4 learner declined (VRAM floor); on the re-offer the coordinator dropped the VM's valid result and the VM, having
    answered already, did not send it again, so the quorum of 2 could never be reached."""
    seen = {"l0": 0}

    def l0_busy_once(args, device, now=None):  # noqa: ANN001, ANN202
        if args.learner_id == "l0":
            seen["l0"] += 1
            if seen["l0"] == 1:
                return (False, "vram: test")
        return (True, "ok")

    monkeypatch.setattr(r, "gate", l0_busy_once)
    hub = Hub()
    rcs = run_threads(hub, coordinator_args(tmp_path, "rt6", ["l0", "l1"], rounds=1, quorum=2, deadline=6.0), [learner_args("rt6", "l0", 0), learner_args("rt6", "l1", 1)])
    assert rcs == {"coord": 0, "l0": 0, "l1": 0}, rcs
    rep = json.loads((tmp_path / "rt6.json").read_text())
    assert rep["completed"] and rep["rounds"][0]["status"] == "complete" and rep["rounds"][0]["accepted"] == ["l0", "l1"] and rep["rounds"][0]["retries"] == 1


# ----------------------------------------------------------------------------------------------------------------------- per-learner keys

K_COORD, K_L0, K_L1 = bytes(range(32)), bytes(range(40, 72)), bytes(range(80, 112))


def test_with_per_learner_keys_nobody_can_submit_a_result_in_another_learners_name(tmp_path: Path) -> None:
    hub = Hub()
    coord_ring = HMACKeyring(keys={"coord": K_COORD, "lk-l0": K_L0, "lk-l1": K_L1}, active_kid="coord")
    l0_ring = HMACKeyring(keys={"coord": K_COORD, "lk-l0": K_L0}, active_kid="lk-l0")
    rogue_ring = HMACKeyring(keys={"coord": K_COORD, "lk-l1": K_L1}, active_kid="lk-l1")  # holds l1's key, pretends to be l0
    stop = threading.Event()

    def rogue() -> None:
        bus = r.Bus(hub.view(), rogue_ring, "rt7")
        pin = fed.code_pin(r.PIN_FILES)
        while not stop.is_set():
            for ev in bus.poll():
                if ev.event_type == fed.OFFER:
                    g = fed.decode_tensors(ev.payload["global_b64"])
                    bus.publish(fed.build_result(run_id="rt7", rnd=int(ev.payload["round"]), learner_id="l0", host="rogue", tokens=10,
                                                 delta={k: v * 0.0 for k, v in g.items()}, delta_l2=0.0, eval_loss_global=4.0, pin=pin, global_sha256=ev.payload["global_sha256"]))
                if ev.event_type == fed.ABORT:
                    return

    coord = parse("coordinator", "--run-id", "rt7", "--learners", "l0", "l1", "--rounds", "1", "--quorum", "1", "--round-deadline-s", "6", "--inner-steps", "4", *COORD,
                  "--learner-key-prefix", "lk-", "--out", str(tmp_path / "rt7.json"))
    rcs: dict = {}
    t_rogue = threading.Thread(target=rogue, daemon=True)
    t_l0 = threading.Thread(target=lambda: rcs.__setitem__("l0", r.run_learner(learner_args("rt7", "l0", 0, argv="--coordinator-key coord"), hub.view(), l0_ring)), daemon=True)
    t_c = threading.Thread(target=lambda: rcs.__setitem__("coord", r.run_coordinator(coord, hub.view(), coord_ring)), daemon=True)
    for t in (t_rogue, t_l0, t_c):
        t.start()
    t_c.join(240)
    t_l0.join(60)
    stop.set()
    assert rcs == {"coord": 0, "l0": 0}, rcs
    rnd = json.loads((tmp_path / "rt7.json").read_text())["rounds"][0]
    assert rnd["accepted"] == ["l0"] and rnd["hosts"]["l0"] != "rogue"  # the accepted l0 result is the honest one
    assert any(who == "l0" and "not the key issued to 'l0'" in why for who, why in rnd["rejected"])


def test_a_learner_pinned_to_the_coordinator_key_ignores_offers_signed_by_anyone_else(tmp_path: Path) -> None:
    hub = Hub()
    other = HMACKeyring(keys={"coord": K_COORD, "lk-l1": K_L1}, active_kid="lk-l1")
    l0_ring = HMACKeyring(keys={"coord": K_COORD, "lk-l0": K_L0, "lk-l1": K_L1}, active_kid="lk-l0")  # can verify l1's signature, but must not obey it
    base = q.build_base_state(1)
    r.Bus(hub.view(), other, "rt8").publish(fed.build_offer(run_id="rt8", rnd=0, sender_id="coordinator", global_state=q.initial_lora(base, 1), base_sha256=fed.tensors_sha256(base),
                                                            pin=fed.code_pin(r.PIN_FILES), inner_steps=2, deadline_unix=time.time() + 60, quorum=1,
                                                            params={"seed": 1, "shard_size": 64, "learners": 1}))
    rc = r.run_learner(parse("learner", "--run-id", "rt8", "--learner-id", "l0", "--device", "cpu", "--governor-log", "", "--idle-exit-s", "2", "--coordinator-key", "coord"), hub.view(), l0_ring)
    assert rc == 0
    types = [json.loads(rec.value)["type"] for rec in hub.records]
    assert fed.CLAIM not in types and fed.RESULT not in types
