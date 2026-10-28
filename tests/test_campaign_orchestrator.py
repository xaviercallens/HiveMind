# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for the S21 campaign orchestrator: signed joins, DST-correct night windows and the dream-phase planner."""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from hivemind.swarm.amcp_kafka_campaign_bus import AMCPKafkaEvent, InMemoryKafkaBridge
from hivemind.swarm.campaign_orchestrator import (
    CampaignOrchestrator,
    CampaignPolicy,
    CampaignRequest,
    DomainEvidence,
    DreamPolicy,
    check_governor_state,
    load_episode_ledger,
    max_unverified_rows,
    night_window,
    parse_campaign_utc,
    plan_dream_phase,
    sign_campaign_request,
    verify_campaign_request_signature,
)

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HAVE_HYPOTHESIS = True
except ImportError:  # pragma: no cover - hypothesis is in the dev group
    HAVE_HYPOTHESIS = False

SECRET = b"s21-orch-test-secret-0123456789ab"
SUBJ1 = "SUBJ-01-PYTHON-RUST-CODER"  # 2026-10-28 18:00Z .. 2026-10-30 22:00Z, domain_key coding_python
SUBJ2 = "SUBJ-02-RUSTY-SUNDIALS-FLUID-ODE"  # 2026-10-31 16:00Z .. 2026-11-02 22:00Z


def utc(text: str) -> float:
    return datetime.fromisoformat(text).timestamp()


NOW = utc("2026-10-28T21:30:00Z")  # inside SUBJ-01 and inside the Paris night (22:30 CET)


def req(pid: str = "rtx-producer-1", *, kind: str = "gpu_producer", campaign: str = SUBJ1, vram: int = 15360,
        slot: str = "eu_evening_cet", now: float = NOW, nonce: str = "nonce-0001", action: str = "join",
        secret: bytes = SECRET) -> CampaignRequest:
    r = CampaignRequest(action, pid, kind, campaign, vram, slot, now, nonce)
    return sign_campaign_request(r, secret)


def orch(tmp_path: Path, policy: CampaignPolicy | None = None, publish: Any = None) -> tuple[CampaignOrchestrator, InMemoryKafkaBridge]:
    bridge = InMemoryKafkaBridge()
    return CampaignOrchestrator(tmp_path / "state.json", SECRET, publish or bridge.publish, policy=policy), bridge


# ---------------------------------------------------------------------------------------------------------------- signing


def test_signature_roundtrip_and_tamper_detection() -> None:
    good = req()
    assert verify_campaign_request_signature(good, SECRET)
    tampered = CampaignRequest(**{**good.__dict__, "declared_vram_mib": 81920})
    assert not verify_campaign_request_signature(tampered, SECRET)
    assert not verify_campaign_request_signature(good, b"another-secret-of-32-bytes-000000")
    assert CampaignRequest.from_dict(good.__dict__) == good


def test_short_secret_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 16 bytes"):
        CampaignOrchestrator(tmp_path / "s.json", b"short", InMemoryKafkaBridge().publish)
    assert not (tmp_path / "s.json").exists()


def test_parse_campaign_utc() -> None:
    assert parse_campaign_utc("20261028T180000Z") == utc("2026-10-28T18:00:00Z")
    with pytest.raises(ValueError):
        parse_campaign_utc("2026-10-28 18:00")


# ----------------------------------------------------------------------------------------------------------------- joins


def test_join_admitted_persisted_and_event_emitted(tmp_path: Path) -> None:
    o, bridge = orch(tmp_path)
    d = o.handle(req(), NOW)
    assert d.decision == "ADMITTED" and d.accepted
    assert d.delivery["status"] == "DELIVERED" and d.delivery["topic"] == "hivemind.amcp.campaigns.v1"
    events = bridge.recent("campaigns")
    assert [e["event_type"] for e in events] == ["io.hivemind.amcp.campaign.admitted"]
    assert events[0]["recipient_id"] == "rtx-producer-1" and events[0]["correlation_id"] == "nonce-0001"
    reloaded = CampaignOrchestrator(tmp_path / "state.json", SECRET, bridge.publish)
    members = reloaded.active_members(NOW, SUBJ1)
    assert [m["participant_id"] for m in members] == ["rtx-producer-1"]
    assert members[0]["expires_at_utc"] == utc("2026-10-30T22:00:00Z")
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert d.to_dict()["accepted"] is True


def test_forged_join_rejected_and_state_unchanged(tmp_path: Path) -> None:
    o, bridge = orch(tmp_path)
    forged = req(secret=b"attacker-guessed-secret-000000000")
    d = o.handle(forged, NOW)
    assert d.decision == "REJECTED" and d.reason == "invalid signature"
    assert o.active_members(NOW) == []
    assert bridge.recent("campaigns")[0]["event_type"] == "io.hivemind.amcp.campaign.rejected"
    # the forged nonce was not recorded, so the genuine request with that nonce still goes through
    assert o.handle(req(), NOW).decision == "ADMITTED"


def test_duplicate_join_is_idempotent(tmp_path: Path) -> None:
    o, bridge = orch(tmp_path, CampaignPolicy(max_members_per_campaign=1))
    assert o.handle(req(nonce="nonce-0001"), NOW).decision == "ADMITTED"
    again = o.handle(req(nonce="nonce-0002"), NOW + 5)
    assert again.decision == "ALREADY_MEMBER" and again.accepted
    assert len(o.active_members(NOW, SUBJ1)) == 1
    assert [e["event_type"].rsplit(".", 1)[1] for e in bridge.recent("campaigns")] == ["already_member", "admitted"]


def test_exact_resend_returns_stored_decision_without_new_event(tmp_path: Path) -> None:
    o, bridge = orch(tmp_path)
    first = o.handle(req(), NOW)
    resend = o.handle(req(), NOW + 1)
    assert first.decision == resend.decision == "ADMITTED"
    assert "duplicate" in resend.reason and resend.event == {}
    assert len(bridge.recent("campaigns")) == 1


def test_nonce_reuse_with_different_content_rejected(tmp_path: Path) -> None:
    o, _ = orch(tmp_path)
    o.handle(req(pid="alice"), NOW)
    d = o.handle(req(pid="mallory"), NOW)
    assert d.decision == "REJECTED" and "nonce reused" in d.reason
    assert [m["participant_id"] for m in o.active_members(NOW)] == ["alice"]


def test_over_capacity_rejected(tmp_path: Path) -> None:
    o, _ = orch(tmp_path, CampaignPolicy(max_members_per_campaign=2))
    assert o.handle(req("a", nonce="nonce-000a"), NOW).decision == "ADMITTED"
    assert o.handle(req("b", nonce="nonce-000b"), NOW).decision == "ADMITTED"
    third = o.handle(req("c", nonce="nonce-000c"), NOW)
    assert third.decision == "REJECTED" and "campaign full (2/2)" in third.reason
    assert len(o.active_members(NOW, SUBJ1)) == 2


def test_participant_quota_across_campaigns(tmp_path: Path) -> None:
    policy = CampaignPolicy(max_active_campaigns_per_participant=1, join_lead_s=4 * 86400)  # SUBJ-02 opens for joins early
    o, _ = orch(tmp_path, policy)
    assert o.handle(req(campaign=SUBJ1, nonce="nonce-0001"), NOW).decision == "ADMITTED"
    d = o.handle(req(campaign=SUBJ2, nonce="nonce-0002"), NOW)
    assert d.decision == "REJECTED" and d.reason.startswith("quota")
    assert o.handle(req("other", campaign=SUBJ2, nonce="nonce-0003"), NOW).decision == "ADMITTED"


def test_campaign_must_exist_and_be_open(tmp_path: Path) -> None:
    o, _ = orch(tmp_path)
    assert o.handle(req(campaign="SUBJ-99-NOPE"), NOW).reason == "unknown campaign 'SUBJ-99-NOPE'"
    early = o.handle(req(campaign=SUBJ2, nonce="nonce-0002"), NOW)  # SUBJ-02 starts 2026-10-31, no lead time by default
    assert early.decision == "REJECTED" and "not open" in early.reason
    late_now = utc("2026-10-30T22:00:00Z")  # SUBJ-01 end instant: closed
    late = o.handle(req(now=late_now, nonce="nonce-0003"), late_now)
    assert late.decision == "REJECTED" and "not open" in late.reason
    assert o.active_members(NOW) == []


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"kind": "wizard"}, "unknown participant_kind"),
        ({"slot": "mars_evening"}, "unknown availability"),
        ({"vram": -1}, "declared_vram_mib"),
        ({"vram": 2_000_000}, "declared_vram_mib"),
        ({"pid": "bad id with spaces"}, "participant_id"),
        ({"nonce": "short"}, "nonce must be"),
        ({"action": "promote"}, "unknown action"),
        ({"now": NOW - 3600}, "allowed skew"),
    ],
)
def test_invalid_fields_rejected(tmp_path: Path, kwargs: dict[str, Any], fragment: str) -> None:
    o, bridge = orch(tmp_path)
    d = o.handle(req(**kwargs), NOW)
    assert d.decision == "REJECTED" and fragment in d.reason
    assert o.active_members(NOW) == [] and bridge.recent("campaigns")[0]["payload"]["decision"] == "REJECTED"


def test_leave_and_rejoin(tmp_path: Path) -> None:
    o, _ = orch(tmp_path)
    o.handle(req(nonce="nonce-0001"), NOW)
    left = o.handle(req(action="leave", nonce="nonce-0002"), NOW + 10)
    assert left.decision == "LEFT" and o.active_members(NOW + 10) == []
    again = o.handle(req(action="leave", nonce="nonce-0003"), NOW + 20)
    assert again.decision == "NOT_MEMBER" and not again.accepted
    assert o.handle(req(nonce="nonce-0004"), NOW + 30).decision == "ADMITTED"


def test_membership_expires_when_campaign_ends(tmp_path: Path) -> None:
    o, bridge = orch(tmp_path)
    o.handle(req(), NOW)
    after = utc("2026-10-30T22:00:01Z")
    expired = o.expire(after)
    assert [d.decision for d in expired] == ["EXPIRED"]
    assert o.active_members(after) == [] and o.state["memberships"][f"{SUBJ1}|rtx-producer-1"]["status"] == "expired"
    assert bridge.recent("campaigns")[0]["event_type"] == "io.hivemind.amcp.campaign.expired"
    assert o.expire(after + 1) == []


def test_bus_outage_is_reported_and_decision_kept(tmp_path: Path) -> None:
    def broken(event: AMCPKafkaEvent) -> dict[str, Any]:
        raise ConnectionError("broker unreachable")

    o, _ = orch(tmp_path, publish=broken)
    d = o.handle(req(), NOW)
    assert d.decision == "ADMITTED" and d.delivery["status"] == "FAILED"
    assert "ConnectionError" in d.delivery["error"]
    assert len(CampaignOrchestrator(tmp_path / "state.json", SECRET, broken).active_members(NOW)) == 1


def test_corrupt_state_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text(json.dumps({"schema_version": 99, "memberships": {}}))
    with pytest.raises(ValueError, match="schema-v1"):
        CampaignOrchestrator(tmp_path / "state.json", SECRET, InMemoryKafkaBridge().publish)
    (tmp_path / "state.json").write_text("{not json")
    with pytest.raises(ValueError):
        CampaignOrchestrator(tmp_path / "state.json", SECRET, InMemoryKafkaBridge().publish)


def test_old_nonces_are_pruned(tmp_path: Path) -> None:
    o, _ = orch(tmp_path)
    o.handle(req(nonce="nonce-0001"), NOW)
    later = NOW + 3 * 900
    o.handle(req("b", now=later, nonce="nonce-0002"), later)
    assert set(o.state["nonces"]) == {"nonce-0002"}
    assert o.state["nonces"]["nonce-0002"]["decision"] == "ADMITTED"


# ---------------------------------------------------------------------------------------------------------- night window


def test_ordinary_night_is_nine_hours() -> None:
    w = night_window(utc("2026-10-28T12:00:00Z"))  # Paris midday, CET (UTC+1): next window is upcoming
    assert w.start_utc == datetime(2026, 10, 28, 21, 0, tzinfo=UTC)
    assert w.end_utc == datetime(2026, 10, 29, 6, 0, tzinfo=UTC)
    assert w.duration_h == 9.0 and not w.contains(utc("2026-10-28T12:00:00Z"))


def test_spring_forward_night_is_eight_hours() -> None:
    # last Sunday of March 2026 is the 29th: 02:00 CET -> 03:00 CEST inside the night of 28->29 March
    w = night_window(utc("2026-03-28T23:00:00Z"))
    assert w.start_utc == datetime(2026, 3, 28, 21, 0, tzinfo=UTC)
    assert w.end_utc == datetime(2026, 3, 29, 5, 0, tzinfo=UTC)
    assert w.duration_h == 8.0 and w.contains(utc("2026-03-28T23:00:00Z"))


def test_fall_back_night_is_ten_hours() -> None:
    # last Sunday of October 2026 is the 25th: 03:00 CEST -> 02:00 CET inside the night of 24->25 October
    w = night_window(utc("2026-10-25T01:30:00Z"))  # after midnight: the window that started yesterday evening
    assert w.start_utc == datetime(2026, 10, 24, 20, 0, tzinfo=UTC)
    assert w.end_utc == datetime(2026, 10, 25, 6, 0, tzinfo=UTC)
    assert w.duration_h == 10.0
    assert w.to_dict()["start_local"].endswith("+02:00") and w.to_dict()["end_local"].endswith("+01:00")


def test_window_edges_and_same_day_windows() -> None:
    end = utc("2026-10-29T06:00:00Z")
    assert night_window(end - 1).end_utc.timestamp() == end  # last second of the night: still the current one
    assert night_window(end).start_utc == datetime(2026, 10, 29, 21, 0, tzinfo=UTC)  # end is exclusive
    day = night_window(utc("2026-10-28T20:00:00Z"), "UTC", 1, 5)  # non-wrapping window already over today: tomorrow's
    assert day.start_utc == datetime(2026, 10, 29, 1, 0, tzinfo=UTC) and day.duration_h == 4.0
    with pytest.raises(ValueError):
        night_window(NOW, start_h=7, end_h=7)
    with pytest.raises(ValueError):
        night_window(NOW, start_h=25)


if HAVE_HYPOTHESIS:

    @settings(max_examples=300, deadline=None)
    @given(st.integers(min_value=int(utc("2024-01-01T00:00:00Z")), max_value=int(utc("2030-12-31T00:00:00Z"))))
    def test_window_contains_now_or_starts_within_a_day(now: int) -> None:
        w = night_window(float(now))
        assert w.duration_h in (8.0, 9.0, 10.0)
        assert w.contains(now) or 0 < w.start_utc.timestamp() - now <= 24 * 3600
        assert w.start.hour == 22 and w.end.hour == 7

    @given(st.integers(min_value=0, max_value=10**9))
    def test_unverified_cap_is_exact_and_maximal(v: int) -> None:
        u = max_unverified_rows(v)
        assert 10 * u <= 3 * (v + u)  # at most 30% of the mix
        assert 10 * (u + 1) > 3 * (v + u + 1)  # and one more row would break the cap


def test_unverified_cap_examples_and_bad_input() -> None:
    assert max_unverified_rows(700) == 300 and max_unverified_rows(0) == 0
    assert max_unverified_rows(6) == 2  # 2/8 = 25%; 3/9 = 33% would exceed
    with pytest.raises(ValueError):
        max_unverified_rows(-1)
    with pytest.raises(ValueError):
        max_unverified_rows(10, 10, 10)


# ------------------------------------------------------------------------------------------------------- ledger, governor


def test_ledger_loading_validates(tmp_path: Path) -> None:
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps({"domains": [{"domain": "stars", "verified_accept": 900, "heldout_eval_items": 317}]}))
    (ev,) = load_episode_ledger(p)
    assert ev.new_verified == 900 and ev.unverified == 0 and ev.task_kind == "label"
    bad_rows = [
        [{"domain": "stars", "verified_accept": 1}, {"domain": "stars", "verified_accept": 2}],
        [{"domain": "stars", "verified_accept": -5}],
        [{"domain": "stars", "verified_accept": 5, "task_kind": "poetry"}],
        [{"domain": "Bad Name", "verified_accept": 5}],
    ]
    for rows in bad_rows:
        p.write_text(json.dumps({"domains": rows}))
        with pytest.raises(ValueError):
            load_episode_ledger(p)
    p.write_text(json.dumps([1, 2]))
    with pytest.raises(ValueError, match="'domains' list"):
        load_episode_ledger(p)


def write_governor(path: Path, state: str = "yielded", t: float | None = NOW) -> Path:
    snap: dict[str, Any] = {"state": state}
    if t is not None:
        snap["t"] = t
    path.write_text(json.dumps(snap))
    return path


def test_governor_state_checks(tmp_path: Path) -> None:
    g = tmp_path / "gov.json"
    assert check_governor_state(None, NOW, 900)[0] == "no producer governor state path given"
    assert "not readable" in check_governor_state(g, NOW, 900)[0]
    g.write_text("{oops")
    assert "not readable" in check_governor_state(g, NOW, 900)[0]
    g.write_text(json.dumps({"t": NOW}))
    assert "no 'state'" in check_governor_state(g, NOW, 900)[0]
    write_governor(g, t=None)
    assert "no timestamp" in check_governor_state(g, NOW, 900)[0]
    write_governor(g, t=NOW - 3600)
    assert "old" in check_governor_state(g, NOW, 900)[0]
    write_governor(g, state="held")
    assert "held" in check_governor_state(g, NOW, 900)[0]
    write_governor(g, state="training")
    reason, seen = check_governor_state(g, NOW, 900)
    assert reason == "" and seen == {"state": "training", "age_s": 0.0}


# ------------------------------------------------------------------------------------------------------------ dream plan


def member(pid: str, vram: int, *, slot: str = "eu_evening_cet", campaign: str = SUBJ1, kind: str = "gpu_producer") -> dict[str, Any]:
    return {"campaign_id": campaign, "participant_id": pid, "participant_kind": kind, "declared_vram_mib": vram,
            "availability": slot, "status": "active", "expires_at_utc": utc("2026-10-30T22:00:00Z")}


LEDGER = [
    DomainEvidence("coding_python", verified_accept=1200, verified_reject=400, unverified=2000, heldout_eval_items=500),
    DomainEvidence("stars", verified_accept=900, verified_reject=50, unverified=0, heldout_eval_items=317),
    DomainEvidence("cyber", verified_accept=40, verified_reject=0, unverified=1200, heldout_eval_items=600),
]


def plan(tmp_path: Path, members: list[dict[str, Any]] | None = None, ledger: list[DomainEvidence] | None = None,
         now: float = NOW, policy: DreamPolicy | None = None, hold: bool = False, governor: bool = True) -> dict[str, Any]:
    hold_file = tmp_path / "HOLD"
    if hold:
        hold_file.write_text("")
    gov = write_governor(tmp_path / "gov.json", t=now) if governor else tmp_path / "missing_gov.json"
    return plan_dream_phase(
        now_utc=now, members=members if members is not None else [member("t4-paris", 15360), member("rtx2070", 8192)],
        ledger=ledger if ledger is not None else LEDGER, hold_file=hold_file, governor_state_path=gov, policy=policy,
    )


def test_plan_assigns_experts_with_controls_and_gates(tmp_path: Path) -> None:
    p = plan(tmp_path)
    assert p["status"] == "PLANNED" and p["blocked_reasons"] == [] and p["executes_anything"] is False
    by_domain = {j["domain"]: j for j in p["jobs"]}
    assert set(by_domain) == {"coding_python", "stars"}
    decisions = {d["domain"]: d["decision"] for d in p["domains_considered"]}
    assert decisions == {"coding_python": "SELECTED", "stars": "SELECTED", "cyber": "REFUSED"}
    coding = by_domain["coding_python"]
    assert coding["producer_id"] == "t4-paris" and coding["profile"] == "big" and coding["campaign_id"] == SUBJ1
    assert coding["training_mix"] == {"verified": 1200, "unverified": 514, "unverified_fraction": round(514 / 1714, 6)}
    assert coding["training_mix"]["unverified_fraction"] <= 0.3
    stages = [s["stage"] for s in coding["stages"]]
    assert stages == ["train_adapter", "control_shuffled_label", "eval_heldout", "promotion_gate"]
    assert coding["stages"][1]["seed"] == coding["stages"][0]["seed"]
    assert coding["stages"][-1]["auto_promote"] is False and "control_shuffled_label" in coding["stages"][-1]["requires"][0]
    assert by_domain["stars"]["producer_id"] == "rtx2070" and by_domain["stars"]["profile"] == "standard"
    assert p["night_window"]["in_window_now"] is True and p["night_window"]["duration_h"] == 9.0


def test_plan_is_deterministic(tmp_path: Path) -> None:
    a, b = plan(tmp_path), plan(tmp_path)
    assert a["plan_sha256"] == b["plan_sha256"]
    c = plan(tmp_path, ledger=LEDGER[:1])
    assert c["plan_sha256"] != a["plan_sha256"] and len(c["jobs"]) == 1


def test_hold_file_blocks(tmp_path: Path) -> None:
    p = plan(tmp_path, hold=True)
    assert p["status"] == "BLOCKED" and p["jobs"] == []
    assert any("HOLD" in r for r in p["blocked_reasons"])
    assert p["preconditions"]["hold_file"]["blocking"] is not None


def test_unreadable_governor_blocks(tmp_path: Path) -> None:
    p = plan(tmp_path, governor=False)
    assert p["status"] == "BLOCKED" and p["jobs"] == []
    assert any("governor state not readable" in r for r in p["blocked_reasons"])


def test_no_fitting_participant_blocks(tmp_path: Path) -> None:
    small = [member("gtx1050", 2048), member("ai-agent", 0, kind="ai_agent"), member("apac", 24576, slot="apac_evening_jst")]
    p = plan(tmp_path, members=small)
    assert p["status"] == "BLOCKED" and p["jobs"] == []
    assert any(r.startswith("no participant fits") for r in p["blocked_reasons"])
    reasons = {r["participant_id"]: r["decision"] for r in p["participants_considered"]}
    assert reasons["gtx1050"].startswith("NO_FIT") and reasons["apac"].startswith("NOT_AVAILABLE")
    assert "ai-agent" not in reasons


def test_no_qualifying_domain_blocks(tmp_path: Path) -> None:
    p = plan(tmp_path, ledger=[DomainEvidence("cyber", 40, 0, 5000, 600), DomainEvidence("stars", 900, 0, 0, 100)])
    assert p["status"] == "BLOCKED" and any("no domain qualifies" in r for r in p["blocked_reasons"])
    reasons = {d["domain"]: d["reason"] for d in p["domains_considered"]}
    assert "new verified" in reasons["cyber"] and "held-out" in reasons["stars"]


def test_consumed_episodes_do_not_count_again(tmp_path: Path) -> None:
    used = DomainEvidence("stars", verified_accept=900, verified_reject=0, unverified=0, heldout_eval_items=400, consumed_verified=800)
    p = plan(tmp_path, ledger=[used])
    assert p["status"] == "BLOCKED"
    assert p["domains_considered"][0]["new_verified"] == 100


def test_generative_domain_gets_cross_domain_control(tmp_path: Path) -> None:
    gen = DomainEvidence("maths", verified_accept=1000, verified_reject=0, unverified=0, heldout_eval_items=500, task_kind="generative")
    alone = plan(tmp_path, ledger=[gen])
    assert alone["status"] == "BLOCKED" and "cross-domain control donor" in alone["domains_considered"][0]["reason"]
    paired = plan(tmp_path, ledger=[gen, LEDGER[1]])
    maths = next(j for j in paired["jobs"] if j["domain"] == "maths")
    ctrl = {s["stage"]: s for s in maths["stages"]}
    assert ctrl["control_cross_domain"]["donor_domain"] == "stars" and ctrl["control_shuffled_label"]["can_fail"] is False
    assert "control_cross_domain" in ctrl["promotion_gate"]["requires"][0]
    assert maths["budget_h"] == DreamPolicy().job_budget_h(2)


def test_little_night_left_leaves_domains_unassigned(tmp_path: Path) -> None:
    late = utc("2026-10-29T05:30:00Z")  # 06:30 CET: half an hour of night left
    p = plan(tmp_path, now=late)
    assert p["status"] == "BLOCKED" and p["night_window"]["usable_h"] == 0.5
    assert {u["domain"] for u in p["unassigned_domains"]} == {"coding_python", "stars"}


def test_max_jobs_defers_and_one_producer_runs_jobs_back_to_back(tmp_path: Path) -> None:
    p = plan(tmp_path, members=[member("t4-paris", 15360)], policy=DreamPolicy(max_jobs=1))
    assert [d["decision"] for d in p["domains_considered"]] == ["SELECTED", "DEFERRED", "REFUSED"]
    assert len(p["jobs"]) == 1
    q = plan(tmp_path, members=[member("t4-paris", 15360)])
    jobs = sorted(q["jobs"], key=lambda j: j["slot_start_utc"])
    assert len(jobs) == 2 and jobs[0]["slot_end_utc"] == jobs[1]["slot_start_utc"]
    assert jobs[0]["slot_start_utc"] == datetime.fromtimestamp(NOW, UTC).isoformat()


def test_producer_vram_is_the_minimum_across_memberships(tmp_path: Path) -> None:
    ms = [member("p1", 15360), member("p1", 8192, campaign=SUBJ2)]
    p = plan(tmp_path, members=ms, ledger=LEDGER[:1])
    assert p["participants_considered"][0]["declared_vram_mib"] == 8192
    assert p["jobs"][0]["profile"] == "standard" and p["jobs"][0]["lora_rank"] == 8


# ------------------------------------------------------------------------------------------------------------------- CLI


def load_cli() -> Any:
    spec = importlib.util.spec_from_file_location("dream_phase_plan", Path(__file__).resolve().parents[1] / "scripts" / "dream_phase_plan.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cli_files(tmp_path: Path) -> list[str]:
    o, _ = orch(tmp_path)
    assert o.handle(req("t4-paris"), NOW).decision == "ADMITTED"
    (tmp_path / "ledger.json").write_text(json.dumps({"domains": [
        {"domain": "stars", "verified_accept": 900, "verified_reject": 50, "unverified": 0, "heldout_eval_items": 317}]}))
    write_governor(tmp_path / "gov.json")
    return ["--dry-run", "--state", str(tmp_path / "state.json"), "--ledger", str(tmp_path / "ledger.json"),
            "--governor-state", str(tmp_path / "gov.json"), "--hold-file", str(tmp_path / "HOLD"), "--now", "2026-10-28T21:30:00Z"]


def test_cli_dry_run_plans_and_writes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cli = load_cli()
    args = cli_files(tmp_path)
    assert cli.main(args + ["--out", str(tmp_path / "plan.json")]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "PLANNED" and printed["jobs"][0]["producer_id"] == "t4-paris"
    assert json.loads((tmp_path / "plan.json").read_text())["plan_sha256"] == printed["plan_sha256"]


def test_cli_blocked_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cli = load_cli()
    args = cli_files(tmp_path)
    (tmp_path / "HOLD").write_text("")
    assert cli.main(args) == 2
    assert "HOLD" in " ".join(json.loads(capsys.readouterr().out)["blocked_reasons"])
    (tmp_path / "HOLD").unlink()
    (tmp_path / "ledger.json").write_text("{")
    (tmp_path / "state.json").write_text(json.dumps({"schema_version": 1}))
    assert cli.main(args) == 2
    reasons = json.loads(capsys.readouterr().out)["blocked_reasons"]
    assert any("ledger not usable" in r for r in reasons) and any("not a schema-v1" in r for r in reasons)
    missing = [a if a != str(tmp_path / "state.json") else str(tmp_path / "nope.json") for a in args]
    assert cli.main(missing) == 2
    assert any("campaign state not readable" in r for r in json.loads(capsys.readouterr().out)["blocked_reasons"])


def test_cli_requires_dry_run_and_explicit_utc(tmp_path: Path) -> None:
    cli = load_cli()
    args = cli_files(tmp_path)
    with pytest.raises(SystemExit):
        cli.main([a for a in args if a != "--dry-run"])
    with pytest.raises(SystemExit, match="UTC offset"):
        cli.main(args[:-1] + ["2026-10-28T21:30:00"])
    assert isinstance(cli.parse_now(None), float) and cli.parse_now("2026-10-28T21:30:00Z") == NOW


def test_atomic_write_keeps_old_file_on_failure(tmp_path: Path) -> None:
    from hivemind.swarm.campaign_orchestrator import atomic_write_json

    target = tmp_path / "x.json"
    atomic_write_json(target, {"v": 1})
    with pytest.raises(TypeError):
        atomic_write_json(target, {"v": object()})
    assert json.loads(target.read_text()) == {"v": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


# ------------------------------------------------------------------------------------------------- review regressions


def test_generative_job_keeps_cross_domain_control_when_donor_is_deferred(tmp_path: Path) -> None:
    # max_jobs=1 picks only the generative domain; the donor (stars) is DEFERRED but its verified rows still exist.
    # Before the fix the job fell back to the shuffled-label control, which cannot fail on a generative task.
    gen = DomainEvidence("maths", verified_accept=1000, verified_reject=0, unverified=0, heldout_eval_items=500, task_kind="generative")
    p = plan(tmp_path, members=[member("t4-paris", 15360)], ledger=[gen, LEDGER[1]], policy=DreamPolicy(max_jobs=1))
    assert [d["decision"] for d in p["domains_considered"]] == ["SELECTED", "DEFERRED"]
    (maths,) = p["jobs"]
    stages = {s["stage"]: s for s in maths["stages"]}
    assert stages["control_cross_domain"]["donor_domain"] == "stars" and stages["control_cross_domain"]["can_fail"] is True
    assert "control_cross_domain" in stages["promotion_gate"]["requires"][0]
    assert "control_shuffled_label" not in stages["promotion_gate"]["requires"][0]


def test_governor_snapshot_from_the_future_blocks(tmp_path: Path) -> None:
    g = write_governor(tmp_path / "gov.json", t=NOW + 3600)
    reason, seen = check_governor_state(g, NOW, 900)
    assert "in the future" in reason and seen["age_s"] == -3600.0
    write_governor(g, t=NOW + 30)  # small clock skew is tolerated
    assert check_governor_state(g, NOW, 900)[0] == ""


def test_per_participant_keys_stop_impersonation(tmp_path: Path) -> None:
    alice_key, bob_key = b"alice-key-0123456789abcdef", b"bob-key-0123456789abcdef00"
    bridge = InMemoryKafkaBridge()
    o = CampaignOrchestrator(tmp_path / "state.json", {"alice": alice_key, "bob": bob_key}, bridge.publish)
    assert o.handle(req("alice", nonce="nonce-0001", secret=alice_key), NOW).decision == "ADMITTED"
    # bob holds a valid key, but not alice's: he can neither make her leave nor join under her name elsewhere
    spoof = o.handle(req("alice", action="leave", nonce="nonce-0002", secret=bob_key), NOW)
    assert spoof.decision == "REJECTED" and spoof.reason == "invalid signature"
    unknown = o.handle(req("carol", nonce="nonce-0003", secret=alice_key), NOW)
    assert unknown.decision == "REJECTED" and unknown.reason == "invalid signature"
    assert [m["participant_id"] for m in o.active_members(NOW)] == ["alice"]
    with pytest.raises(ValueError, match="at least 16 bytes"):
        CampaignOrchestrator(tmp_path / "s2.json", {"alice": b"short"}, bridge.publish)
    with pytest.raises(ValueError, match="at least 16 bytes"):
        CampaignOrchestrator(tmp_path / "s3.json", {}, bridge.publish)
