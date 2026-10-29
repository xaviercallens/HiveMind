# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Campaign orchestrator: signed campaign joins, and a night "dream phase" planner for new MoE experts (S21 ``orch``).

Two parts, both fail-closed and both side-effect free beyond their own state file and the AMCP publish callable they are given.

1. **Join campaign.** A participant (``human_researcher``, ``gpu_producer``, ``ai_agent``, ``bee_specialist``) sends a request
   naming one of the ``JOINT_SUBJECT_CAMPAIGNS``, its declared VRAM and its availability slot, signed with HMAC-SHA256 over a
   canonical JSON encoding. :class:`CampaignOrchestrator` checks, in order: field validity, freshness of ``issued_at_utc``,
   signature, nonce reuse, campaign existence, the campaign being open now, the participant's quota of concurrent campaigns, and
   the campaign's capacity. A repeated join of an active member is idempotent (``ALREADY_MEMBER``, capacity untouched); an
   exact resend of an already-decided request returns the stored decision; a nonce reused with different content is rejected.
   Leave is the same signed request with ``action="leave"``; memberships expire when their campaign ends. Every decision,
   including rejections, is emitted as an AMCP event through the publish callable passed in (``AMCPKafkaEvent -> metadata``,
   e.g. ``InMemoryKafkaBridge().publish`` from :mod:`hivemind.swarm.amcp_kafka_campaign_bus`; a real Kafka transport's publish
   can be swapped in without touching this module). State is written atomically (temp file + fsync + ``os.replace``).

2. **Dream phase planner.** :func:`plan_dream_phase` produces a plan JSON for the next night window, 22:00-07:00 Europe/Paris by
   default, computed with ``zoneinfo`` so the window is 8 h on the last Sunday of March and 10 h on the last Sunday of October.
   From a ledger of *verified* episode counts per domain it selects which new experts to train: a domain needs a minimum number
   of new verified episodes and enough held-out eval items to power the gate, and the unverified rows mixed in are capped so they
   are at most 30% of the training mix (the cap is computed in integers: ``u <= floor(3 v / 7)``). Each selected expert is
   assigned to a joined ``gpu_producer`` whose declared VRAM fits a BOINC training profile (``profiles_for_gpu``) and whose
   remaining night time covers the job budget. Every job carries a shuffled-label control, and for generative domains a
   cross-domain control too (a shuffled numeric answer can never be right, so that control cannot fail: NIGHT-MOE-2 section 1),
   then a held-out eval gate, then a promotion gate that never auto-promotes.
   The plan is BLOCKED, with every reason listed and no jobs, if the operator HOLD file exists, the producer governor state is
   unreadable, stale or ``held``, no domain qualifies, or no participant fits. The planner executes nothing on any GPU.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from hivemind.boinc.real_client.moe_profiles import PROFILES, profiles_for_gpu
from hivemind.swarm.amcp_kafka_campaign_bus import (
    JOINT_SUBJECT_CAMPAIGNS,
    AMCPKafkaEvent,
    SubjectCampaign,
    build_amcp_kafka_event,
)

PARTICIPANT_KINDS: frozenset[str] = frozenset({"human_researcher", "gpu_producer", "ai_agent", "bee_specialist"})
AVAILABILITY_SLOTS: frozenset[str] = frozenset({"eu_evening_cet", "us_evening_est", "apac_evening_jst", "24x7_autonomous"})
ACTIONS: frozenset[str] = frozenset({"join", "leave"})
ORCHESTRATOR_ID = "hivemind/campaign-orchestrator"
STATE_SCHEMA_VERSION = 1
MAX_DECLARED_VRAM_MIB = 1_048_576  # 1 TiB: anything above is a typo or a lie, not a GPU
MIN_SECRET_BYTES = 16
_PARTICIPANT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@-]{0,63}$")
_NONCE_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_DOMAIN_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,47}$")

NIGHT_TZ = "Europe/Paris"
NIGHT_START_H = 22
NIGHT_END_H = 7
# the dilution cap of AutoevolveAI scripts/ltm_learning_mix.py: unverified rows <= 30% of the training mix
MAX_UNVERIFIED_NUM, MAX_UNVERIFIED_DEN = 3, 10


# What the orchestrator needs from a bus: one callable that delivers an event and returns broker metadata
# (topic, partition, offset). ``InMemoryKafkaBridge().publish`` today; the S21 Kafka transport's publish later.
PublishFn = Callable[[AMCPKafkaEvent], Mapping[str, Any]]
# Signing keys: one shared secret (every key holder can sign for any participant_id: only for a trusted gateway that has already
# authenticated the participant), or a mapping participant_id -> that participant's own secret (a participant can then sign
# only its own joins and leaves; a participant without a key is rejected).
SecretSource = bytes | Mapping[str, bytes]


# ----------------------------------------------------------------------------------------------------------------------
# Join campaign
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CampaignRequest:
    """A signed join or leave request. ``signature`` is HMAC-SHA256 (hex) over :meth:`canonical_bytes`."""

    action: str
    participant_id: str
    participant_kind: str
    campaign_id: str
    declared_vram_mib: int
    availability: str
    issued_at_utc: float
    nonce: str
    signature: str = ""

    def canonical_bytes(self) -> bytes:
        body = {k: v for k, v in asdict(self).items() if k != "signature"}
        return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CampaignRequest:
        return cls(
            action=str(data.get("action", "join")),
            participant_id=str(data["participant_id"]),
            participant_kind=str(data["participant_kind"]),
            campaign_id=str(data["campaign_id"]),
            declared_vram_mib=int(data.get("declared_vram_mib", 0)),
            availability=str(data.get("availability", "24x7_autonomous")),
            issued_at_utc=float(data["issued_at_utc"]),
            nonce=str(data["nonce"]),
            signature=str(data.get("signature", "")),
        )


def sign_campaign_request(request: CampaignRequest, secret: bytes) -> CampaignRequest:
    """Return a copy of ``request`` carrying a valid HMAC-SHA256 signature under ``secret``."""
    sig = hmac.new(secret, request.canonical_bytes(), hashlib.sha256).hexdigest()
    return CampaignRequest(**{**asdict(request), "signature": sig})


def verify_campaign_request_signature(request: CampaignRequest, secret: bytes) -> bool:
    expected = hmac.new(secret, request.canonical_bytes(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, request.signature)


def parse_campaign_utc(stamp: str) -> float:
    """``20261028T180000Z`` (the iCalendar UTC form used by ``SubjectCampaign``) -> POSIX seconds."""
    return datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC).timestamp()


@dataclass(frozen=True)
class CampaignPolicy:
    """Admission limits. ``join_lead_s`` lets people join that long before a campaign starts (0 = only while it runs)."""

    max_members_per_campaign: int = 64
    max_active_campaigns_per_participant: int = 2
    request_max_skew_s: float = 900.0
    join_lead_s: float = 0.0


@dataclass(frozen=True)
class JoinDecision:
    decision: str  # ADMITTED | ALREADY_MEMBER | LEFT | NOT_MEMBER | REJECTED
    reason: str
    campaign_id: str
    participant_id: str
    event: dict[str, Any] = field(default_factory=dict)
    delivery: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.decision in ("ADMITTED", "ALREADY_MEMBER", "LEFT")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"accepted": self.accepted}


def _membership_key(campaign_id: str, participant_id: str) -> str:
    return f"{campaign_id}|{participant_id}"


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON so a reader sees either the old file or the new one, never a torn write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class CampaignOrchestrator:
    """Admission control for joint subject campaigns, persisted to one JSON state file."""

    def __init__(
        self,
        state_path: Path | str,
        secret: SecretSource,
        publish: PublishFn,
        *,
        campaigns: Sequence[SubjectCampaign] = JOINT_SUBJECT_CAMPAIGNS,
        policy: CampaignPolicy | None = None,
    ) -> None:
        keys = [secret] if isinstance(secret, bytes) else list(secret.values())
        if not keys or any(not isinstance(k, bytes) or len(k) < MIN_SECRET_BYTES for k in keys):
            raise ValueError(f"HMAC secret must be at least {MIN_SECRET_BYTES} bytes")
        self.state_path = Path(state_path)
        self._secret: bytes | dict[str, bytes] = secret if isinstance(secret, bytes) else dict(secret)
        self.publish = publish
        self.campaigns: dict[str, SubjectCampaign] = {c.subject_id: c for c in campaigns}
        self.policy = policy or CampaignPolicy()
        self.state = self._load_state()

    # -- persistence ---------------------------------------------------------------------------------------------------
    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"schema_version": STATE_SCHEMA_VERSION, "memberships": {}, "nonces": {}}
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        if (
            not isinstance(data, dict)
            or data.get("schema_version") != STATE_SCHEMA_VERSION
            or not isinstance(data.get("memberships"), dict)
            or not isinstance(data.get("nonces"), dict)
        ):
            raise ValueError(f"campaign state {self.state_path} is not a schema-v{STATE_SCHEMA_VERSION} orchestrator state")
        return data

    def _save(self) -> None:
        atomic_write_json(self.state_path, self.state)

    # -- queries -------------------------------------------------------------------------------------------------------
    def campaign_window(self, campaign_id: str) -> tuple[float, float]:
        c = self.campaigns[campaign_id]
        return parse_campaign_utc(c.dtstart_utc), parse_campaign_utc(c.dtend_utc)

    def is_open(self, campaign_id: str, now_utc: float) -> bool:
        start, end = self.campaign_window(campaign_id)
        return start - self.policy.join_lead_s <= now_utc < end

    def active_members(self, now_utc: float, campaign_id: str | None = None) -> list[dict[str, Any]]:
        out = [
            dict(m)
            for m in self.state["memberships"].values()
            if m["status"] == "active" and m["expires_at_utc"] > now_utc and (campaign_id is None or m["campaign_id"] == campaign_id)
        ]
        return sorted(out, key=lambda m: (m["campaign_id"], m["participant_id"]))

    # -- transitions ---------------------------------------------------------------------------------------------------
    def expire(self, now_utc: float) -> list[JoinDecision]:
        """Mark memberships whose campaign has ended as expired and emit one event each."""
        decisions: list[JoinDecision] = []
        for m in self.state["memberships"].values():
            if m["status"] == "active" and m["expires_at_utc"] <= now_utc:
                m["status"] = "expired"
                m["ended_at_utc"] = now_utc
                decisions.append(
                    JoinDecision("EXPIRED", "campaign ended", m["campaign_id"], m["participant_id"])
                )
        if decisions:
            self._save()
        return [self._emit(d, now_utc, nonce=None) for d in decisions]

    def handle(self, request: CampaignRequest, now_utc: float) -> JoinDecision:
        """Decide a join or leave request, persist the outcome, and emit it as an AMCP event."""
        self.expire(now_utc)
        self._prune_nonces(now_utc)
        problem = self._validate(request, now_utc)
        if problem:
            return self._emit(JoinDecision("REJECTED", problem, request.campaign_id, request.participant_id), now_utc, request.nonce)
        seen = self.state["nonces"].get(request.nonce)
        if seen is not None:
            if seen["digest"] != request.digest():
                return self._emit(
                    JoinDecision("REJECTED", "nonce reused with different content", request.campaign_id, request.participant_id),
                    now_utc,
                    request.nonce,
                )
            # an exact resend (network retry): same answer, no second state change, no second event
            return JoinDecision(seen["decision"], "duplicate of an already-decided request", request.campaign_id, request.participant_id)
        decision = self._join(request, now_utc) if request.action == "join" else self._leave(request, now_utc)
        self.state["nonces"][request.nonce] = {"digest": request.digest(), "decision": decision.decision, "utc": now_utc}
        self._save()
        return self._emit(decision, now_utc, request.nonce)

    def _validate(self, r: CampaignRequest, now_utc: float) -> str:
        if r.action not in ACTIONS:
            return f"unknown action {r.action!r}"
        if not _PARTICIPANT_ID_RE.match(r.participant_id):
            return "participant_id is not a valid identifier"
        if r.participant_kind not in PARTICIPANT_KINDS:
            return f"unknown participant_kind {r.participant_kind!r}"
        if r.availability not in AVAILABILITY_SLOTS:
            return f"unknown availability slot {r.availability!r}"
        if not 0 <= r.declared_vram_mib <= MAX_DECLARED_VRAM_MIB:
            return f"declared_vram_mib {r.declared_vram_mib} outside [0, {MAX_DECLARED_VRAM_MIB}]"
        if not _NONCE_RE.match(r.nonce):
            return "nonce must be 8-64 characters of [A-Za-z0-9_-]"
        if abs(now_utc - r.issued_at_utc) > self.policy.request_max_skew_s:
            return f"request issued {now_utc - r.issued_at_utc:.0f} s from now; allowed skew is {self.policy.request_max_skew_s:.0f} s"
        key = self._secret if isinstance(self._secret, bytes) else self._secret.get(r.participant_id)
        if key is None:
            return "invalid signature"  # no key on file for this participant: same answer as a bad signature
        if not verify_campaign_request_signature(r, key):
            return "invalid signature"
        if r.campaign_id not in self.campaigns:
            return f"unknown campaign {r.campaign_id!r}"
        return ""

    def _join(self, r: CampaignRequest, now_utc: float) -> JoinDecision:
        key = _membership_key(r.campaign_id, r.participant_id)
        current = self.state["memberships"].get(key)
        if current is not None and current["status"] == "active":
            return JoinDecision("ALREADY_MEMBER", "already an active member; nothing changed", r.campaign_id, r.participant_id)
        if not self.is_open(r.campaign_id, now_utc):
            start, end = self.campaign_window(r.campaign_id)
            return JoinDecision(
                "REJECTED", f"campaign not open now (accepts joins from {start - self.policy.join_lead_s:.0f} until {end:.0f} UTC s)",
                r.campaign_id, r.participant_id,
            )
        mine = [m for m in self.active_members(now_utc) if m["participant_id"] == r.participant_id]
        if len(mine) >= self.policy.max_active_campaigns_per_participant:
            return JoinDecision(
                "REJECTED", f"quota: already in {len(mine)} campaigns (max {self.policy.max_active_campaigns_per_participant})",
                r.campaign_id, r.participant_id,
            )
        members = self.active_members(now_utc, r.campaign_id)
        if len(members) >= self.policy.max_members_per_campaign:
            return JoinDecision(
                "REJECTED", f"campaign full ({len(members)}/{self.policy.max_members_per_campaign})", r.campaign_id, r.participant_id
            )
        _, end = self.campaign_window(r.campaign_id)
        self.state["memberships"][key] = {
            "campaign_id": r.campaign_id,
            "participant_id": r.participant_id,
            "participant_kind": r.participant_kind,
            "declared_vram_mib": r.declared_vram_mib,
            "availability": r.availability,
            "joined_at_utc": now_utc,
            "expires_at_utc": end,
            "status": "active",
        }
        return JoinDecision("ADMITTED", "admitted", r.campaign_id, r.participant_id)

    def _leave(self, r: CampaignRequest, now_utc: float) -> JoinDecision:
        m = self.state["memberships"].get(_membership_key(r.campaign_id, r.participant_id))
        if m is None or m["status"] != "active":
            return JoinDecision("NOT_MEMBER", "no active membership to leave", r.campaign_id, r.participant_id)
        m["status"] = "left"
        m["ended_at_utc"] = now_utc
        return JoinDecision("LEFT", "left the campaign", r.campaign_id, r.participant_id)

    def _prune_nonces(self, now_utc: float) -> None:
        horizon = 2 * self.policy.request_max_skew_s
        self.state["nonces"] = {n: v for n, v in self.state["nonces"].items() if now_utc - float(v["utc"]) <= horizon}

    def _emit(self, d: JoinDecision, now_utc: float, nonce: str | None) -> JoinDecision:
        event = build_amcp_kafka_event(
            topic_key="campaigns",
            event_type=f"io.hivemind.amcp.campaign.{d.decision.lower()}",
            sender_id=ORCHESTRATOR_ID,
            sender_kind="orchestrator",
            recipient_id=d.participant_id,
            subject_id=d.campaign_id,
            correlation_id=nonce,
            payload={"decision": d.decision, "reason": d.reason, "campaign_id": d.campaign_id, "participant_id": d.participant_id},
            now_utc=now_utc,
        )
        try:
            delivery = {"status": "DELIVERED", **dict(self.publish(event))}
        except Exception as exc:  # noqa: BLE001 - the decision is already durable; a bus outage is reported, not raised
            delivery = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"[:200]}
        return JoinDecision(d.decision, d.reason, d.campaign_id, d.participant_id, event.to_dict(), delivery)


# ----------------------------------------------------------------------------------------------------------------------
# Night window (DST-correct)
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NightWindow:
    start: datetime  # aware, in the night's timezone
    end: datetime
    tz: str

    @property
    def start_utc(self) -> datetime:
        return self.start.astimezone(UTC)

    @property
    def end_utc(self) -> datetime:
        return self.end.astimezone(UTC)

    @property
    def duration_h(self) -> float:
        # subtract in UTC: two datetimes sharing one ZoneInfo subtract as wall-clock times and would miss the DST hour
        return (self.end_utc - self.start_utc).total_seconds() / 3600.0

    def contains(self, now_utc: float) -> bool:
        return self.start_utc.timestamp() <= now_utc < self.end_utc.timestamp()

    def to_dict(self) -> dict[str, Any]:
        return {
            "tz": self.tz,
            "start_local": self.start.isoformat(),
            "end_local": self.end.isoformat(),
            "start_utc": self.start_utc.isoformat(),
            "end_utc": self.end_utc.isoformat(),
            "duration_h": round(self.duration_h, 4),
        }


def night_window(now_utc: float, tz: str = NIGHT_TZ, start_h: int = NIGHT_START_H, end_h: int = NIGHT_END_H) -> NightWindow:
    """The night window containing ``now_utc``, or else the next one. Wraps past midnight when ``start_h > end_h``."""
    if start_h == end_h or not (0 <= start_h <= 23 and 0 <= end_h <= 23):
        raise ValueError(f"invalid night window {start_h}:00-{end_h}:00")
    zone = ZoneInfo(tz)
    local_day = datetime.fromtimestamp(now_utc, zone).date()

    def window_starting(d: date) -> NightWindow:
        start = datetime.combine(d, time(start_h), tzinfo=zone)
        end = datetime.combine(d + timedelta(days=1) if end_h < start_h else d, time(end_h), tzinfo=zone)
        return NightWindow(start, end, tz)

    for offset in (-1, 0):
        w = window_starting(local_day + timedelta(days=offset))
        if w.end_utc.timestamp() > now_utc:
            return w
    return window_starting(local_day + timedelta(days=1))  # a same-day window that already ended today: tomorrow's


# ----------------------------------------------------------------------------------------------------------------------
# Dream phase planner
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainEvidence:
    """Real verdict counts for one domain. ``consumed_verified`` = verified episodes already used by an earlier expert."""

    domain: str
    verified_accept: int
    verified_reject: int
    unverified: int
    heldout_eval_items: int
    task_kind: str = "label"  # "label" (classification-style) or "generative"
    consumed_verified: int = 0

    @property
    def new_verified(self) -> int:
        return max(0, self.verified_accept - self.consumed_verified)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> DomainEvidence:
        ev = cls(
            domain=str(d["domain"]),
            verified_accept=int(d["verified_accept"]),
            verified_reject=int(d.get("verified_reject", 0)),
            unverified=int(d.get("unverified", 0)),
            heldout_eval_items=int(d.get("heldout_eval_items", 0)),
            task_kind=str(d.get("task_kind", "label")),
            consumed_verified=int(d.get("consumed_verified", 0)),
        )
        if not _DOMAIN_RE.match(ev.domain):
            raise ValueError(f"invalid domain name {ev.domain!r}")
        if ev.task_kind not in ("label", "generative"):
            raise ValueError(f"{ev.domain}: task_kind must be 'label' or 'generative'")
        counts = (ev.verified_accept, ev.verified_reject, ev.unverified, ev.heldout_eval_items, ev.consumed_verified)
        if min(counts) < 0:
            raise ValueError(f"{ev.domain}: counts must be non-negative")
        return ev


def load_episode_ledger(path: Path | str) -> list[DomainEvidence]:
    """Read ``{"domains": [{domain, verified_accept, verified_reject, unverified, heldout_eval_items, ...}]}``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("domains") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise ValueError("ledger must be an object with a 'domains' list")
    out = [DomainEvidence.from_dict(r) for r in rows]
    names = [e.domain for e in out]
    if len(set(names)) != len(names):
        raise ValueError("ledger lists a domain twice")
    return out


def max_unverified_rows(verified_rows: int, num: int = MAX_UNVERIFIED_NUM, den: int = MAX_UNVERIFIED_DEN) -> int:
    """Largest ``u`` with ``u / (verified_rows + u) <= num / den``, in exact integer arithmetic."""
    if verified_rows < 0 or not 0 <= num < den:
        raise ValueError("need verified_rows >= 0 and 0 <= num < den")
    return (num * verified_rows) // (den - num)


@dataclass(frozen=True)
class DreamPolicy:
    """Selection and budget rules. Stage caps default to the NIGHT-MOE-1 orchestrator's largest caps (train 1.0 h,
    control 0.25 h) plus an eval allowance; they are budgets, not measured durations."""

    min_new_verified: int = 200
    min_heldout_eval_items: int = 300  # NIGHT-MOE-2: n >= 300 per pack for the primary paired test
    max_jobs: int = 8
    train_cap_h: float = 1.0
    control_cap_h: float = 0.25
    eval_cap_h: float = 0.75
    governor_max_age_s: float = 900.0
    governor_max_future_s: float = 60.0  # a snapshot stamped later than this ahead of now would never go stale
    eligible_availability: frozenset[str] = frozenset({"eu_evening_cet", "24x7_autonomous"})
    tz: str = NIGHT_TZ
    start_h: int = NIGHT_START_H
    end_h: int = NIGHT_END_H

    def job_budget_h(self, n_controls: int) -> float:
        return self.train_cap_h + n_controls * self.control_cap_h + (1 + n_controls) * self.eval_cap_h


def check_hold_file(hold_file: Path | None) -> str:
    if hold_file is not None and hold_file.exists():
        return f"operator HOLD file present at {hold_file}"
    return ""


def check_governor_state(
    path: Path | None, now_utc: float, max_age_s: float, max_future_s: float = 60.0
) -> tuple[str, dict[str, Any]]:
    """Return (blocking reason or "", what was read). Fail-closed: anything unreadable blocks."""
    if path is None:
        return "no producer governor state path given", {}
    try:
        snap = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"producer governor state not readable: {type(exc).__name__}", {}
    if not isinstance(snap, dict) or not isinstance(snap.get("state"), str):
        return "producer governor state has no 'state' field", {}
    t = snap.get("t")
    if not isinstance(t, (int, float)):
        return "producer governor state has no timestamp 't'", {"state": snap["state"]}
    age = now_utc - float(t)
    seen = {"state": snap["state"], "age_s": round(age, 1)}
    if age > max_age_s:
        return f"producer governor state is {age:.0f} s old (max {max_age_s:.0f} s)", seen
    if -age > max_future_s:
        return f"producer governor state is stamped {-age:.0f} s in the future (max {max_future_s:.0f} s): clock or file is wrong", seen
    if snap["state"] == "held":
        return "producer governor is held by the operator", seen
    return "", seen


def _seed(night_date: str, domain: str, salt: str) -> int:
    return int(hashlib.sha256(f"{night_date}:{domain}:{salt}".encode()).hexdigest()[:8], 16)


def _select_domains(
    ledger: Sequence[DomainEvidence], policy: DreamPolicy
) -> tuple[list[dict[str, Any]], list[DomainEvidence], list[str]]:
    """Return (report rows, domains picked tonight, every qualifying domain in rank order).

    The third list is the donor pool for cross-domain controls: a donor needs qualifying verified rows, not a job tonight, so a
    domain DEFERRED by ``max_jobs`` is still a valid donor."""
    considered: list[dict[str, Any]] = []
    picked: list[DomainEvidence] = []
    qualified_names = {
        e.domain for e in ledger if e.new_verified >= policy.min_new_verified and e.heldout_eval_items >= policy.min_heldout_eval_items
    }
    ranked = sorted(ledger, key=lambda x: (-x.new_verified, x.domain))
    for e in ranked:
        cap = max_unverified_rows(e.new_verified)
        used_u = min(e.unverified, cap)
        row: dict[str, Any] = {
            "domain": e.domain,
            "task_kind": e.task_kind,
            "new_verified": e.new_verified,
            "verified_reject": e.verified_reject,
            "unverified_available": e.unverified,
            "unverified_included": used_u,
            "unverified_dropped": e.unverified - used_u,
            "unverified_fraction": round(used_u / (e.new_verified + used_u), 6) if e.new_verified + used_u else 0.0,
            "heldout_eval_items": e.heldout_eval_items,
        }
        if e.new_verified < policy.min_new_verified:
            row |= {"decision": "REFUSED", "reason": f"{e.new_verified} new verified episodes < minimum {policy.min_new_verified}"}
        elif e.heldout_eval_items < policy.min_heldout_eval_items:
            row |= {"decision": "REFUSED", "reason": f"{e.heldout_eval_items} held-out eval items < minimum {policy.min_heldout_eval_items}"}
        elif e.task_kind == "generative" and not (qualified_names - {e.domain}):
            row |= {"decision": "REFUSED", "reason": "generative domain needs a cross-domain control donor and no other domain qualifies"}
        elif len(picked) >= policy.max_jobs:
            row |= {"decision": "DEFERRED", "reason": f"max_jobs {policy.max_jobs} reached"}
        else:
            row |= {"decision": "SELECTED", "reason": "qualifies"}
            picked.append(e)
        considered.append(row)
    return considered, picked, [e.domain for e in ranked if e.domain in qualified_names]


def _producers(members: Iterable[Mapping[str, Any]], policy: DreamPolicy, campaigns: Mapping[str, SubjectCampaign]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    for m in members:
        if m.get("participant_kind") != "gpu_producer" or m.get("status", "active") != "active":
            continue
        p = by_id.setdefault(m["participant_id"], {"participant_id": m["participant_id"], "vram_mib": int(m["declared_vram_mib"]),
                                                   "availability": set(), "domains": set(), "campaigns": set()})
        p["vram_mib"] = min(p["vram_mib"], int(m["declared_vram_mib"]))  # conservative across memberships
        p["availability"].add(m["availability"])
        p["campaigns"].add(m["campaign_id"])
        if m["campaign_id"] in campaigns:
            p["domains"].add(campaigns[m["campaign_id"]].domain_key)
    fit: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []
    for pid in sorted(by_id):
        p = by_id[pid]
        profiles = profiles_for_gpu(p["vram_mib"])
        row = {"participant_id": pid, "declared_vram_mib": p["vram_mib"], "availability": sorted(p["availability"]),
               "campaigns": sorted(p["campaigns"]), "profiles": profiles}
        if not profiles:
            row["decision"] = f"NO_FIT: {p['vram_mib']} MiB below every training profile floor"
        elif not (p["availability"] & policy.eligible_availability):
            row["decision"] = f"NOT_AVAILABLE: slots {sorted(p['availability'])} not in {sorted(policy.eligible_availability)}"
        else:
            row["decision"] = "ELIGIBLE"
            fit.append({**p, "profile": profiles[0], "used_h": 0.0})
        report.append(row)
    return report, fit


def plan_dream_phase(
    *,
    now_utc: float,
    members: Sequence[Mapping[str, Any]],
    ledger: Sequence[DomainEvidence],
    hold_file: Path | None,
    governor_state_path: Path | None,
    policy: DreamPolicy | None = None,
    campaigns: Sequence[SubjectCampaign] = JOINT_SUBJECT_CAMPAIGNS,
    extra_blocked_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the night's expert-training plan. Pure planning: it reads the HOLD file and governor state, nothing else."""
    pol = policy or DreamPolicy()
    window = night_window(now_utc, pol.tz, pol.start_h, pol.end_h)
    usable_from = max(now_utc, window.start_utc.timestamp())
    usable_h = (window.end_utc.timestamp() - usable_from) / 3600.0
    blocked = list(extra_blocked_reasons)
    hold_reason = check_hold_file(hold_file)
    gov_reason, gov_seen = check_governor_state(governor_state_path, now_utc, pol.governor_max_age_s, pol.governor_max_future_s)
    blocked += [r for r in (hold_reason, gov_reason) if r]

    considered, picked, qualified = _select_domains(ledger, pol)
    if not picked:
        blocked.append(f"no domain qualifies (min {pol.min_new_verified} new verified episodes, {pol.min_heldout_eval_items} held-out items)")
    camp_map = {c.subject_id: c for c in campaigns}
    producer_report, producers = _producers(members, pol, camp_map)

    night_date = window.start.date().isoformat()
    jobs: list[dict[str, Any]] = []
    unassigned: list[dict[str, Any]] = []
    for e in picked:
        donor = next((d for d in qualified if d != e.domain), None) if e.task_kind == "generative" else None
        if e.task_kind == "generative" and donor is None:  # pragma: no cover - _select_domains refuses this; never plan a gate that cannot fail
            raise RuntimeError(f"generative domain {e.domain} selected without a cross-domain control donor")
        n_controls = 2 if donor else 1
        budget = pol.job_budget_h(n_controls)
        candidates = [p for p in producers if usable_h - p["used_h"] >= budget]
        if not candidates:
            unassigned.append({"domain": e.domain, "budget_h": budget, "reason": "no eligible producer has this much night time left"})
            continue
        candidates.sort(key=lambda p: (e.domain not in p["domains"], p["used_h"], -p["vram_mib"], p["participant_id"]))
        p = candidates[0]
        jobs.append(_job(e, p, donor, budget, night_date, usable_from, considered, camp_map))
        p["used_h"] += budget
    if picked and not jobs:
        blocked.append("no participant fits: no joined gpu_producer with a fitting VRAM profile, an eligible slot and enough night time")
    status = "BLOCKED" if blocked else "PLANNED"
    plan: dict[str, Any] = {
        "object": "hivemind.dream_phase_plan",
        "schema_version": 1,
        "status": status,
        "blocked_reasons": blocked,
        "generated_at_utc": datetime.fromtimestamp(now_utc, UTC).isoformat(),
        "night_window": window.to_dict() | {"in_window_now": window.contains(now_utc), "usable_h": round(usable_h, 4)},
        "preconditions": {
            "hold_file": {"path": str(hold_file) if hold_file else None, "blocking": hold_reason or None},
            "governor": {"path": str(governor_state_path) if governor_state_path else None, "blocking": gov_reason or None, **gov_seen},
        },
        "policy": {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in asdict(pol).items()}
        | {"max_unverified_fraction": f"{MAX_UNVERIFIED_NUM}/{MAX_UNVERIFIED_DEN}"},
        "domains_considered": considered,
        "participants_considered": producer_report,
        "jobs": jobs if status == "PLANNED" else [],
        "unassigned_domains": unassigned,
        "executes_anything": False,
    }
    plan["plan_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True, default=str).encode()).hexdigest()
    return plan


def select_dream_teacher_for_job(domain: str, vram_mib: int) -> dict[str, Any]:
    """Select the low-bit (1.58-bit / 2-bit / 3-bit) Dream Phase teacher for a job's domain and producer VRAM."""
    dom = domain.strip().lower()
    is_code = any(k in dom for k in ("code", "coding", "swe", "rust", "python", "cyber", "formal", "lean", "sys"))
    if is_code and vram_mib >= 14500:
        return {
            "teacher_id": "tiel-coder-35b-a3b-iq3-xxs",
            "hf_repo": "peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF",
            "gguf_file": "Tiel-Coder-35B-A3B-UD-IQ3_XXS.gguf",
            "quant_bits": 3.06,
            "weights_gib": 12.30,
            "params_b": 35.0,
            "active_params_b": 3.0,
            "moe": True,
            "execution_mode": "full_gpu_offload_16gb",
        }
    if is_code and vram_mib >= 6000:
        return {
            "teacher_id": "tiel-coder-35b-a3b-iq3-xxs",
            "hf_repo": "peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF",
            "gguf_file": "Tiel-Coder-35B-A3B-UD-IQ3_XXS.gguf",
            "quant_bits": 3.06,
            "weights_gib": 12.30,
            "params_b": 35.0,
            "active_params_b": 3.0,
            "moe": True,
            "execution_mode": "moe_expert_cpu_ram_split_wddm_safe",
        }
    if vram_mib >= 12000:
        return {
            "teacher_id": "ternary-bonsai-2-27b-pq2",
            "hf_repo": "prism-ml/Ternary-Bonsai-2-27B-gguf",
            "gguf_file": "Ternary-Bonsai-2-27B-PQ2_0.gguf",
            "quant_bits": 2.13,
            "weights_gib": 6.71,
            "params_b": 27.36,
            "active_params_b": 27.36,
            "moe": False,
            "execution_mode": "full_gpu_offload_co_resident_student",
        }
    if vram_mib >= 6500:
        return {
            "teacher_id": "ternary-bonsai-2-27b-ptq1",
            "hf_repo": "prism-ml/Ternary-Bonsai-2-27B-gguf",
            "gguf_file": "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
            "quant_bits": 1.75,
            "weights_gib": 5.54,
            "params_b": 27.36,
            "active_params_b": 27.36,
            "moe": False,
            "execution_mode": "full_gpu_offload_8gb_wddm_safe",
        }
    return {
        "teacher_id": "ternary-bonsai-2-9b-pq2",
        "hf_repo": "prism-ml/Ternary-Bonsai-2-9B-gguf",
        "gguf_file": "Ternary-Bonsai-2-9B-PQ2_0.gguf",
        "quant_bits": 2.13,
        "weights_gib": 2.25,
        "params_b": 9.0,
        "active_params_b": 9.0,
        "moe": False,
        "execution_mode": "compact_edge_gpu_offload",
    }


def _job(
    e: DomainEvidence,
    p: Mapping[str, Any],
    donor: str | None,
    budget: float,
    night_date: str,
    usable_from: float,
    considered: Sequence[Mapping[str, Any]],
    campaigns: Mapping[str, SubjectCampaign],
) -> dict[str, Any]:
    row = next(c for c in considered if c["domain"] == e.domain)
    prof = PROFILES[p["profile"]]
    seed = _seed(night_date, e.domain, "train")
    start = usable_from + p["used_h"] * 3600.0
    campaign = next((cid for cid in sorted(p["campaigns"]) if cid in campaigns and campaigns[cid].domain_key == e.domain), None)
    stages: list[dict[str, Any]] = [
        {"stage": "train_adapter", "rows_verified": e.new_verified, "rows_unverified": row["unverified_included"], "seed": seed},
        {
            "stage": "control_shuffled_label",
            "rows_verified": e.new_verified,
            "rows_unverified": row["unverified_included"],
            "seed": seed,
            "label_permutation_seed": _seed(night_date, e.domain, "shuffle"),
            "can_fail": e.task_kind == "label",
            "note": "same rows, recipe and seed; labels permuted across rows, so label marginals are kept",
        },
    ]
    if donor:
        stages.append({"stage": "control_cross_domain", "donor_domain": donor, "seed": seed, "can_fail": True,
                       "note": "same recipe trained on another domain's verified rows; a shuffled generative answer cannot score, this can"})
    primary_control = "control_cross_domain" if donor else "control_shuffled_label"
    stages += [
        {
            "stage": "eval_heldout",
            "items": e.heldout_eval_items,
            "systems": ["base", "train_adapter"] + [s["stage"] for s in stages if s["stage"].startswith("control_")],
            "test": "exact paired McNemar",
            "alpha": 0.05,
            "report": ["per-system accuracy", "unparsable counts", "discordant pair counts"],
        },
        {
            "stage": "promotion_gate",
            "auto_promote": False,
            "status": "PENDING_EVAL",
            "requires": [
                f"train_adapter beats {primary_control} on the held-out set (primary, p < 0.05, positive difference)",
                "train_adapter beats base on items both parsed (secondary)",
                "verified update gate passes on the other domains' packs (no interference regression)",
                "held-out items are disjoint from training rows (decontamination report attached)",
            ],
        },
    ]
    return {
        "job_id": f"dream-{night_date}-{e.domain}",
        "expert_id": f"hivemind/expert-{e.domain}-{night_date.replace('-', '')}",
        "domain": e.domain,
        "task_kind": e.task_kind,
        "campaign_id": campaign,
        "producer_id": p["participant_id"],
        "producer_vram_mib": p["vram_mib"],
        "profile": p["profile"],
        "boinc_app": prof["app"],
        "plan_class": prof["plan_class"],
        "workunit_prefix": prof["prefix"],
        "min_gpu_ram_mb": prof["min_gpu_ram_mb"],
        "lora_rank": prof["rank"],
        "dream_teacher": select_dream_teacher_for_job(e.domain, int(p["vram_mib"])),
        "training_mix": {
            "verified": e.new_verified,
            "unverified": row["unverified_included"],
            "unverified_fraction": row["unverified_fraction"],
        },
        "budget_h": round(budget, 4),
        "slot_start_utc": datetime.fromtimestamp(start, UTC).isoformat(),
        "slot_end_utc": datetime.fromtimestamp(start + budget * 3600.0, UTC).isoformat(),
        "stages": stages,
    }
