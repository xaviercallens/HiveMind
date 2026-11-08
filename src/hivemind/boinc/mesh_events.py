# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""BOINC Validator -> AMCP Mesh Event Bridge (PS-08-D1).

Publishes ``validated`` and ``assimilated`` work-unit events from the BOINC
validator onto AMCP mesh topics so volunteer serving and training hosts observe
the newest validated adapter pointer directly from the mesh, equal to the
BOINC chain ledger's pointer, while keeping mesh events strictly read-only on
the credit/HMT ledger (no mesh event can mint, burn, or alter credit).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Callable, Mapping

TOPIC_VALIDATED = "hivemind.boinc.workunit.validated"
TOPIC_ASSIMILATED = "hivemind.boinc.workunit.assimilated"
ALLOWED_EVENT_TYPES = frozenset({"validated", "assimilated"})
_HEX_CHARS = frozenset("0123456789abcdefABCDEF")
FORBIDDEN_CREDIT_FIELDS = frozenset(
    {
        "credit",
        "credit_delta",
        "verified_credit",
        "hmt_delta",
        "hmt_amount",
        "balance",
        "grant_credit",
        "mint_hmt",
    }
)


def _is_sha256_hex(val: object) -> bool:
    return isinstance(val, str) and len(val) == 64 and all(c in _HEX_CHARS for c in val)


class ReadOnlyLedgerViolationError(ValueError):
    """Raised when an AMCP mesh event attempts to carry or trigger a credit/HMT mutation."""


@dataclass(frozen=True)
class AdapterPointer:
    """Canonical pointer to the newest validated and assimilated LoRA adapter on a chain."""

    chain_stem: str
    round_index: int
    adapter_rel_path: str
    adapter_sha256: str
    wu_name: str
    domain: str
    timestamp_utc: str

    def __post_init__(self) -> None:
        for fname in ("chain_stem", "adapter_rel_path", "wu_name", "domain", "timestamp_utc"):
            val = getattr(self, fname)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{fname} must be a non-empty string")
        if isinstance(self.round_index, bool) or not isinstance(self.round_index, int) or self.round_index < 0:
            raise ValueError("round_index must be a non-negative integer")
        if not _is_sha256_hex(self.adapter_sha256):
            raise ValueError("adapter_sha256 must be a 64-character hex string")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BOINCMeshEvent:
    """Immutable event published onto AMCP mesh topics by the BOINC validator/assimilator."""

    event_id: str
    topic: str
    event_type: str
    wu_name: str
    host_id: str
    chain_stem: str
    round_index: int
    adapter_rel_path: str
    adapter_sha256: str
    domain: str
    timestamp_utc: str
    payload_sha256: str

    def __post_init__(self) -> None:
        for fname in ("event_id", "topic", "wu_name", "host_id", "chain_stem", "adapter_rel_path", "domain", "timestamp_utc"):
            val = getattr(self, fname)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{fname} must be a non-empty string")
        if self.event_type not in ALLOWED_EVENT_TYPES:
            raise ValueError(f"event_type must be one of {sorted(ALLOWED_EVENT_TYPES)}")
        if isinstance(self.round_index, bool) or not isinstance(self.round_index, int) or self.round_index < 0:
            raise ValueError("round_index must be a non-negative integer")
        if not _is_sha256_hex(self.adapter_sha256):
            raise ValueError("adapter_sha256 must be a 64-character hex string")
        if not _is_sha256_hex(self.payload_sha256):
            raise ValueError("payload_sha256 must be a 64-character hex string")

    def to_AdapterPointer(self) -> AdapterPointer:
        return AdapterPointer(
            chain_stem=self.chain_stem,
            round_index=self.round_index,
            adapter_rel_path=self.adapter_rel_path,
            adapter_sha256=self.adapter_sha256,
            wu_name=self.wu_name,
            domain=self.domain,
            timestamp_utc=self.timestamp_utc,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BOINCChainLedger:
    """Authoritative BOINC chain state and volunteer credit ledger."""

    def __init__(self, initial_credits: Mapping[str, float] | None = None) -> None:
        if initial_credits is not None and not isinstance(initial_credits, Mapping):
            raise TypeError("initial_credits must be a Mapping or None")
        self._credits: dict[str, float] = {}
        for k, v in (initial_credits or {}).items():
            if not isinstance(k, str) or not k.strip():
                raise ValueError("credit account key must be a non-empty string")
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) < 0.0:
                raise ValueError("credit balance must be a non-negative finite number")
            self._credits[k] = float(v)
        self._validated_wus: dict[str, AdapterPointer] = {}
        self._latest_pointers: dict[str, AdapterPointer] = {}

    def credit_snapshot(self) -> dict[str, float]:
        """Return an immutable copy of all volunteer credit balances."""
        return dict(self._credits)

    def credit_digest(self) -> str:
        """Compute a deterministic SHA-256 digest of the credit ledger state."""
        raw = json.dumps(self._credits, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def record_validated_workunit(self, pointer: AdapterPointer) -> None:
        """Record that a workunit passed replay validation (does not advance assimilated chain tip yet)."""
        if not isinstance(pointer, AdapterPointer):
            raise TypeError("pointer must be an AdapterPointer")
        self._validated_wus[pointer.wu_name] = pointer

    def record_assimilated_workunit(self, pointer: AdapterPointer) -> None:
        """Advance the chain's assimilated adapter pointer once validated."""
        if not isinstance(pointer, AdapterPointer):
            raise TypeError("pointer must be an AdapterPointer")
        if pointer.wu_name not in self._validated_wus:
            raise ValueError(f"Cannot assimilate unvalidated workunit {pointer.wu_name!r}")
        current = self._latest_pointers.get(pointer.chain_stem)
        if current is None or pointer.round_index >= current.round_index:
            self._latest_pointers[pointer.chain_stem] = pointer

    def latest_adapter_pointer(self, chain_stem: str = "latest_adapter") -> AdapterPointer | None:
        if not isinstance(chain_stem, str) or not chain_stem.strip():
            raise ValueError("chain_stem must be a non-empty string")
        return self._latest_pointers.get(chain_stem)


class AMCPMeshEventBus:
    """In-memory / topic-routed AMCP event bus enforcing read-only credit invariants."""

    def __init__(self, ledger: BOINCChainLedger) -> None:
        if not isinstance(ledger, BOINCChainLedger):
            raise TypeError("ledger must be a BOINCChainLedger")
        self._ledger = ledger
        self._subscribers: dict[str, list[Callable[[BOINCMeshEvent], None]]] = {
            TOPIC_VALIDATED: [],
            TOPIC_ASSIMILATED: [],
        }
        self._published_events: list[BOINCMeshEvent] = []

    @property
    def published_events(self) -> tuple[BOINCMeshEvent, ...]:
        return tuple(self._published_events)

    def subscribe(self, topic: str, handler: Callable[[BOINCMeshEvent], None]) -> None:
        if topic not in self._subscribers:
            raise ValueError(f"Unsupported AMCP BOINC topic: {topic!r}")
        if not callable(handler):
            raise TypeError("handler must be callable")
        self._subscribers[topic].append(handler)

    def publish_raw_dict(self, raw: Mapping[str, Any]) -> BOINCMeshEvent:
        """Validate a raw event dictionary, reject any credit fields, and publish onto the AMCP topic."""
        if not isinstance(raw, Mapping):
            raise ValueError("Event payload must be a mapping")
        forbidden_present = FORBIDDEN_CREDIT_FIELDS.intersection(raw.keys())
        if forbidden_present:
            raise ReadOnlyLedgerViolationError(
                f"AMCP mesh events are read-only on the ledger; forbidden credit fields: {sorted(forbidden_present)}"
            )

        event_type = str(raw.get("event_type") or "").strip()
        if event_type not in ALLOWED_EVENT_TYPES:
            raise ValueError(f"Invalid event_type {event_type!r}; expected one of {sorted(ALLOWED_EVENT_TYPES)}")

        topic = TOPIC_VALIDATED if event_type == "validated" else TOPIC_ASSIMILATED
        wu_name = str(raw.get("wu_name") or "").strip()
        host_id = str(raw.get("host_id") or "").strip()
        chain_stem = str(raw.get("chain_stem") or "latest_adapter").strip()
        raw_round = raw.get("round_index", -1)
        if isinstance(raw_round, bool) or not isinstance(raw_round, int):
            raise ValueError("wu_name, host_id, chain_stem, and non-negative round_index are required")
        round_index = int(raw_round)
        adapter_rel_path = str(raw.get("adapter_rel_path") or "").strip()
        adapter_sha256 = str(raw.get("adapter_sha256") or "").strip()
        domain = str(raw.get("domain") or "general").strip()
        timestamp_utc = str(raw.get("timestamp_utc") or "").strip()

        if not wu_name or not host_id or not chain_stem or round_index < 0:
            raise ValueError("wu_name, host_id, chain_stem, and non-negative round_index are required")
        if not adapter_rel_path or not _is_sha256_hex(adapter_sha256):
            raise ValueError("adapter_rel_path and 64-hex adapter_sha256 are required")

        body_for_hash = {
            "topic": topic,
            "event_type": event_type,
            "wu_name": wu_name,
            "host_id": host_id,
            "chain_stem": chain_stem,
            "round_index": round_index,
            "adapter_rel_path": adapter_rel_path,
            "adapter_sha256": adapter_sha256,
            "domain": domain,
            "timestamp_utc": timestamp_utc,
        }
        payload_sha256 = hashlib.sha256(
            json.dumps(body_for_hash, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        event = BOINCMeshEvent(
            event_id=f"amcp-boinc-{event_type}-{wu_name}-r{round_index}",
            topic=topic,
            event_type=event_type,
            wu_name=wu_name,
            host_id=host_id,
            chain_stem=chain_stem,
            round_index=round_index,
            adapter_rel_path=adapter_rel_path,
            adapter_sha256=adapter_sha256,
            domain=domain,
            timestamp_utc=timestamp_utc,
            payload_sha256=payload_sha256,
        )
        return self._dispatch(event)

    def _dispatch(self, event: BOINCMeshEvent) -> BOINCMeshEvent:
        digest_before = self._ledger.credit_digest()
        self._published_events.append(event)
        for handler in self._subscribers[event.topic]:
            handler(event)
        digest_after = self._ledger.credit_digest()
        if digest_before != digest_after:
            raise ReadOnlyLedgerViolationError(
                "AMCP mesh event dispatch mutated the credit ledger; mesh events must be strictly read-only"
            )
        return event


class AMCPMeshHostView:
    """Host-side subscriber view tracking validated and assimilated adapter pointers from the AMCP mesh."""

    def __init__(self, host_id: str) -> None:
        if not isinstance(host_id, str) or not host_id.strip():
            raise ValueError("host_id must be a non-empty string")
        self.host_id = host_id
        self._validated_by_wu: dict[str, AdapterPointer] = {}
        self._latest_by_chain: dict[str, AdapterPointer] = {}
        self.events_received: int = 0

    def attach(self, bus: AMCPMeshEventBus) -> None:
        if not isinstance(bus, AMCPMeshEventBus):
            raise TypeError("bus must be an AMCPMeshEventBus")
        bus.subscribe(TOPIC_VALIDATED, self.on_event)
        bus.subscribe(TOPIC_ASSIMILATED, self.on_event)

    def on_event(self, event: BOINCMeshEvent) -> None:
        if not isinstance(event, BOINCMeshEvent):
            raise TypeError("event must be a BOINCMeshEvent")
        self.events_received += 1
        pointer = event.to_AdapterPointer()
        if event.event_type == "validated":
            self._validated_by_wu[event.wu_name] = pointer
            return
        if event.event_type == "assimilated":
            validated_ptr = self._validated_by_wu.get(event.wu_name)
            if validated_ptr is None or validated_ptr.adapter_sha256 != pointer.adapter_sha256:
                raise ValueError(
                    f"Host {self.host_id!r} rejected assimilated event for {event.wu_name!r}: no matching validated event"
                )
            current = self._latest_by_chain.get(pointer.chain_stem)
            if current is None or pointer.round_index >= current.round_index:
                self._latest_by_chain[pointer.chain_stem] = pointer

    def latest_adapter_pointer(self, chain_stem: str = "latest_adapter") -> AdapterPointer | None:
        if not isinstance(chain_stem, str) or not chain_stem.strip():
            raise ValueError("chain_stem must be a non-empty string")
        return self._latest_by_chain.get(chain_stem)


def publish_validated_and_assimilated_round(
    *,
    ledger: BOINCChainLedger,
    bus: AMCPMeshEventBus,
    wu_name: str,
    host_id: str,
    chain_stem: str,
    round_index: int,
    adapter_rel_path: str,
    adapter_sha256: str,
    domain: str,
    timestamp_utc: str,
) -> tuple[BOINCMeshEvent, BOINCMeshEvent]:
    """Record a validated + assimilated workunit in the BOINC chain ledger and publish both AMCP mesh events."""
    pointer = AdapterPointer(
        chain_stem=chain_stem,
        round_index=round_index,
        adapter_rel_path=adapter_rel_path,
        adapter_sha256=adapter_sha256,
        wu_name=wu_name,
        domain=domain,
        timestamp_utc=timestamp_utc,
    )
    ledger.record_validated_workunit(pointer)
    ev_val = bus.publish_raw_dict(
        {
            "event_type": "validated",
            "wu_name": wu_name,
            "host_id": host_id,
            "chain_stem": chain_stem,
            "round_index": round_index,
            "adapter_rel_path": adapter_rel_path,
            "adapter_sha256": adapter_sha256,
            "domain": domain,
            "timestamp_utc": timestamp_utc,
        }
    )
    ledger.record_assimilated_workunit(pointer)
    ev_assim = bus.publish_raw_dict(
        {
            "event_type": "assimilated",
            "wu_name": wu_name,
            "host_id": host_id,
            "chain_stem": chain_stem,
            "round_index": round_index,
            "adapter_rel_path": adapter_rel_path,
            "adapter_sha256": adapter_sha256,
            "domain": domain,
            "timestamp_utc": timestamp_utc,
        }
    )
    return ev_val, ev_assim
