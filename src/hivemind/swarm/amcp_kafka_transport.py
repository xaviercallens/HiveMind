# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""AMCP v1.5 Kafka transport: signed CloudEvents, dead-lettering, replay protection and an at-least-once consumer.

`amcp_kafka_campaign_bus` defines the AMCP event (`AMCPKafkaEvent`), the four topics and an in-process
ring buffer (`InMemoryKafkaBridge`) that the gateway's public feed reads. Nothing there talks to a broker.
This module is the transport layer underneath it:

* ``Transport`` protocol (publish / subscribe / poll / commit / seek / close) with two implementations:
  ``InMemoryTransport`` (partitioned logs, consumer-group offsets, retention cap; deterministic, for tests and
  single-process use) and ``KafkaBrokerTransport`` (``confluent-kafka``, optional extra ``kafka``, imported lazily,
  configured from ``HIVEMIND_KAFKA_*`` environment variables, idempotent producer with ``acks=all``).
* Wire format: a CloudEvents 1.0 structured-mode JSON envelope whose ``data`` is the AMCP event. Two signed
  extension attributes (``hmkeyid``, ``hmexpiry``) and the signature ``hmsig`` = HMAC-SHA256 over the canonical
  JSON (sorted keys, no whitespace, UTF-8) of the envelope without ``hmsig``. Keys rotate via a key-id keyring.
* Consume path: size cap -> JSON -> CloudEvents shape -> signature -> per-topic schema -> expiry/skew.
  Anything that fails goes to ``hivemind.amcp.dlq.v1`` with a reason and the raw bytes; it never reaches a handler.
* Delivery: at-least-once. The offset is committed only after the handler returns (or after the record was
  dead-lettered). A failing handler causes a seek back to the failed offset and the rest of that partition's batch
  is held, so per-key order is kept; after ``max_attempts`` the record is dead-lettered as ``handler_failed``.
  A bounded event-id window drops redeliveries of events whose handler already succeeded (the duplicate is
  committed, not re-handled). The window TTL must cover the maximum event lifetime plus clock skew, so an
  event old enough to have left the window is already rejected as expired.
* Backpressure: the consumer fetches at most ``max_batch`` records per iteration and fetches nothing new until
  those are handled or rewound, so the broker, not process memory, holds the backlog.
* Partition key (per-key order): campaigns and human-agent traffic by ``subject_id`` (the campaign), bee
  coordination by ``correlation_id`` (one hand-off conversation, so a request precedes its reply), bee trading
  by ``sender_id`` (the bee's own order sequence).

Readiness of the real broker path is reported by ``kafka_transport_readiness()`` as READY or BLOCKED with the
reason (missing ``HIVEMIND_KAFKA_BOOTSTRAP`` or missing client library); nothing falls back silently.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import importlib
import json
import math
import os
import re
import threading
import time
import zlib
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import ModuleType
from typing import Any, Literal, Protocol

from hivemind.swarm.amcp_kafka_campaign_bus import (
    KAFKA_TOPICS,
    SUBJECT_BY_ID,
    AMCPKafkaEvent,
    InMemoryKafkaBridge,
)

DLQ_TOPIC = "hivemind.amcp.dlq.v1"
AMCP_TOPICS: frozenset[str] = frozenset(KAFKA_TOPICS.values())
TOPIC_KEY_BY_NAME: dict[str, str] = {name: key for key, name in KAFKA_TOPICS.items()}
EVENT_TYPE_PREFIX = "io.hivemind.amcp."
MAX_ENVELOPE_BYTES = 64 * 1024
MIN_HMAC_KEY_BYTES = 32
DEFAULT_EVENT_TTL_S = 300.0
MAX_EVENT_TTL_S = 3600.0
MAX_CLOCK_SKEW_S = 30.0
DLQ_VALUE_CAP_BYTES = 64 * 1024
SENDER_KINDS: frozenset[str] = frozenset({"human_researcher", "gpu_producer", "ai_agent", "bee_specialist"})
AVAILABILITY_SLOTS: tuple[str, ...] = ("eu_evening_cet", "us_evening_est", "apac_evening_jst", "24x7_autonomous")
_KID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_SIG_RE = re.compile(r"^[0-9a-f]{64}$")

RejectReason = Literal[
    "oversize",
    "malformed",
    "not_cloudevent",
    "unsigned",
    "unknown_key_id",
    "bad_signature",
    "envelope_mismatch",
    "topic_mismatch",
    "schema_invalid",
    "ttl_too_long",
    "expired",
    "future_dated",
]


class TransportError(RuntimeError):
    """A broker or client operation failed (delivery not confirmed, poll error, commit error)."""


class TransportUnavailable(RuntimeError):
    """The optional Kafka client library is not installed."""


class KafkaConfigError(ValueError):
    """The ``HIVEMIND_KAFKA_*`` / ``HIVEMIND_AMCP_HMAC_*`` configuration is missing or unsafe."""


class SchemaError(ValueError):
    """An AMCP event does not satisfy the schema of its topic."""


class EnvelopeRejected(Exception):
    """A consumed record failed verification; ``reason`` is the dead-letter reason code."""

    def __init__(self, reason: RejectReason, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason: RejectReason = reason
        self.detail = detail


# ---------------------------------------------------------------------------------------------------------------
# Transport protocol and records
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TransportRecord:
    """One consumed record with its coordinates; ``offset`` is the record's own offset (commit uses offset + 1)."""

    topic: str
    partition: int
    offset: int
    key: bytes | None
    value: bytes


class Transport(Protocol):
    """Minimal broker surface used by the AMCP publisher and consumer (structural; any object with these works).

    * ``publish(topic, key, value)``: append and return ``{"topic", "partition", "offset"}`` once acknowledged.
    * ``subscribe(topics)``: join the consumer group for ``topics``.
    * ``poll(max_records, timeout_s)``: up to ``max_records`` records, in per-partition order.
    * ``commit(record)``: synchronously commit ``record.offset + 1`` for the record's partition.
    * ``seek(topic, partition, offset)``: move the fetch position so ``offset`` is delivered again.
    * ``close()``: flush pending sends and release the client.
    """

    publish: Callable[[str, bytes, bytes], dict[str, Any]]
    subscribe: Callable[[Iterable[str]], None]
    poll: Callable[[int, float], list[TransportRecord]]
    commit: Callable[[TransportRecord], None]
    seek: Callable[[str, int, int], None]
    close: Callable[[], None]


class InMemoryTransport:
    """Partitioned in-process log with one consumer group: Kafka semantics without a broker.

    Partitioning is ``crc32(key) % num_partitions`` so a key always lands on the same partition. Each partition
    keeps at most ``retention_per_partition`` records (oldest dropped, offsets keep increasing). ``poll`` never
    blocks; ``timeout_s`` is accepted for protocol compatibility.
    """

    def __init__(self, num_partitions: int = 4, retention_per_partition: int = 10_000) -> None:
        if num_partitions < 1:
            raise ValueError("num_partitions must be >= 1")
        if retention_per_partition < 1:
            raise ValueError("retention_per_partition must be >= 1")
        self.num_partitions = num_partitions
        self._retention = retention_per_partition
        self._lock = threading.Lock()
        self._logs: dict[tuple[str, int], deque[TransportRecord]] = {}
        self._next_offset: dict[tuple[str, int], int] = {}
        self._position: dict[tuple[str, int], int] = {}
        self._committed: dict[tuple[str, int], int] = {}
        self._subscribed: tuple[str, ...] = ()
        self._closed = False

    def partition_for(self, key: bytes) -> int:
        return zlib.crc32(key) % self.num_partitions

    def publish(self, topic: str, key: bytes, value: bytes) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise TransportError("transport is closed")
            partition = self.partition_for(key)
            tp = (topic, partition)
            offset = self._next_offset.get(tp, 0)
            log = self._logs.setdefault(tp, deque())
            log.append(TransportRecord(topic, partition, offset, bytes(key), bytes(value)))
            if len(log) > self._retention:
                log.popleft()
            self._next_offset[tp] = offset + 1
        return {"topic": topic, "partition": partition, "offset": offset}

    def subscribe(self, topics: Iterable[str]) -> None:
        with self._lock:
            self._subscribed = tuple(dict.fromkeys(topics))

    def poll(self, max_records: int, timeout_s: float) -> list[TransportRecord]:
        out: list[TransportRecord] = []
        with self._lock:
            if self._closed:
                raise TransportError("transport is closed")
            for topic in self._subscribed:
                for partition in range(self.num_partitions):
                    tp = (topic, partition)
                    log = self._logs.get(tp)
                    if not log:
                        continue
                    base = log[0].offset
                    pos = max(self._position.get(tp, self._committed.get(tp, base)), base)
                    while pos - base < len(log) and len(out) < max_records:
                        out.append(log[pos - base])
                        pos += 1
                    self._position[tp] = pos
                    if len(out) >= max_records:
                        return out
        return out

    def commit(self, record: TransportRecord) -> None:
        with self._lock:
            self._committed[(record.topic, record.partition)] = record.offset + 1

    def seek(self, topic: str, partition: int, offset: int) -> None:
        with self._lock:
            self._position[(topic, partition)] = offset

    def committed_offset(self, topic: str, partition: int) -> int | None:
        """Next offset the group would resume from on this partition, or None if nothing was committed."""
        with self._lock:
            return self._committed.get((topic, partition))

    def records(self, topic: str) -> list[TransportRecord]:
        """All retained records of ``topic`` across partitions (inspection helper, e.g. for the DLQ)."""
        with self._lock:
            return [r for (t, _p), log in sorted(self._logs.items()) if t == topic for r in log]

    def close(self) -> None:
        with self._lock:
            self._closed = True


# ---------------------------------------------------------------------------------------------------------------
# Real broker transport (confluent-kafka, optional)
# ---------------------------------------------------------------------------------------------------------------

_SECURITY_PROTOCOLS = frozenset({"PLAINTEXT", "SSL", "SASL_SSL", "SASL_PLAINTEXT"})
_SASL_MECHANISMS = frozenset({"PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512"})


@dataclass(frozen=True)
class KafkaSettings:
    """Broker connection settings resolved from ``HIVEMIND_KAFKA_*`` environment variables."""

    bootstrap: str
    group_id: str = "hivemind-amcp"
    client_id: str = "hivemind-amcp"
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: str | None = None
    sasl_username: str | None = None
    sasl_password: str | None = field(default=None, repr=False)
    ssl_ca_location: str | None = None

    def __post_init__(self) -> None:
        if not self.bootstrap.strip():
            raise KafkaConfigError("HIVEMIND_KAFKA_BOOTSTRAP is empty")
        if self.security_protocol not in _SECURITY_PROTOCOLS:
            raise KafkaConfigError(f"unsupported security protocol {self.security_protocol!r}")
        uses_sasl = self.security_protocol.startswith("SASL_")
        if uses_sasl:
            if self.sasl_mechanism not in _SASL_MECHANISMS:
                raise KafkaConfigError(f"unsupported SASL mechanism {self.sasl_mechanism!r}")
            if not self.sasl_username or not self.sasl_password:
                raise KafkaConfigError("SASL requires HIVEMIND_KAFKA_SASL_USERNAME and HIVEMIND_KAFKA_SASL_PASSWORD")
            if self.security_protocol == "SASL_PLAINTEXT" and self.sasl_mechanism == "PLAIN":
                raise KafkaConfigError("SASL PLAIN over SASL_PLAINTEXT would send the password in clear text")
        elif self.sasl_username or self.sasl_password:
            raise KafkaConfigError("SASL credentials given but security protocol is not SASL_*; they would be ignored")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> KafkaSettings:
        src = os.environ if env is None else env
        bootstrap = src.get("HIVEMIND_KAFKA_BOOTSTRAP", "").strip()
        if not bootstrap:
            raise KafkaConfigError("HIVEMIND_KAFKA_BOOTSTRAP is not set")
        username = src.get("HIVEMIND_KAFKA_SASL_USERNAME") or None
        default_protocol = "SASL_SSL" if username else "PLAINTEXT"
        protocol = (src.get("HIVEMIND_KAFKA_SECURITY_PROTOCOL") or default_protocol).upper()
        mechanism = src.get("HIVEMIND_KAFKA_SASL_MECHANISM") or ("SCRAM-SHA-512" if protocol.startswith("SASL_") else None)
        return cls(
            bootstrap=bootstrap,
            group_id=src.get("HIVEMIND_KAFKA_GROUP_ID") or "hivemind-amcp",
            client_id=src.get("HIVEMIND_KAFKA_CLIENT_ID") or "hivemind-amcp",
            security_protocol=protocol,
            sasl_mechanism=mechanism.upper() if mechanism else None,
            sasl_username=username,
            sasl_password=src.get("HIVEMIND_KAFKA_SASL_PASSWORD") or None,
            ssl_ca_location=src.get("HIVEMIND_KAFKA_SSL_CA_LOCATION") or None,
        )

    def _common_config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "bootstrap.servers": self.bootstrap,
            "client.id": self.client_id,
            "security.protocol": self.security_protocol,
        }
        if self.security_protocol.startswith("SASL_"):
            cfg["sasl.mechanism"] = self.sasl_mechanism
            cfg["sasl.username"] = self.sasl_username
            cfg["sasl.password"] = self.sasl_password
        if self.ssl_ca_location:
            cfg["ssl.ca.location"] = self.ssl_ca_location
        return cfg

    def producer_config(self) -> dict[str, Any]:
        cfg = self._common_config()
        cfg.update(
            {
                "enable.idempotence": True,
                "acks": "all",
                "max.in.flight.requests.per.connection": 5,
                "delivery.timeout.ms": 30_000,
                "linger.ms": 5,
            }
        )
        return cfg

    def consumer_config(self) -> dict[str, Any]:
        cfg = self._common_config()
        cfg.update(
            {
                "group.id": self.group_id,
                "enable.auto.commit": False,
                "enable.auto.offset.store": False,
                "auto.offset.reset": "earliest",
                "isolation.level": "read_committed",
            }
        )
        return cfg


def load_kafka_client() -> ModuleType:
    """Import ``confluent_kafka`` lazily; raise ``TransportUnavailable`` with the install hint if it is absent."""
    try:
        return importlib.import_module("confluent_kafka")
    except ImportError as exc:
        raise TransportUnavailable(
            "confluent-kafka is not installed; install the optional extra: pip install 'project-hivemind[kafka]'"
        ) from exc


def kafka_transport_readiness(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Report whether the real broker path can run here: READY, or BLOCKED with the reason. Does not connect."""
    try:
        settings = KafkaSettings.from_env(env)
    except KafkaConfigError as exc:
        return {"status": "BLOCKED", "reason": str(exc)}
    try:
        mod = load_kafka_client()
    except TransportUnavailable as exc:
        return {"status": "BLOCKED", "reason": str(exc)}
    version = getattr(mod, "version", None)
    return {
        "status": "READY",
        "bootstrap": settings.bootstrap,
        "security_protocol": settings.security_protocol,
        "client_version": version()[0] if callable(version) else "unknown",
    }


class KafkaBrokerTransport:
    """``Transport`` over a real Kafka-compatible broker through ``confluent-kafka``.

    ``publish`` is synchronous: it waits for the delivery report (``acks=all`` on an idempotent producer) and
    raises ``TransportError`` if the broker did not confirm within ``flush_timeout_s``. The consumer is created on
    ``subscribe`` with auto-commit disabled; ``commit`` is synchronous and explicit.
    """

    def __init__(
        self,
        settings: KafkaSettings,
        *,
        client_module: ModuleType | None = None,
        flush_timeout_s: float = 10.0,
    ) -> None:
        self.settings = settings
        self._mod = client_module if client_module is not None else load_kafka_client()
        self._flush_timeout_s = float(flush_timeout_s)
        self._producer: Any = self._mod.Producer(settings.producer_config())
        self._consumer: Any = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> KafkaBrokerTransport:
        return cls(KafkaSettings.from_env(env))

    def publish(self, topic: str, key: bytes, value: bytes) -> dict[str, Any]:
        report: dict[str, Any] = {}

        def _on_delivery(err: Any, msg: Any) -> None:
            report["err"] = err
            report["msg"] = msg

        try:
            self._producer.produce(topic, key=key, value=value, on_delivery=_on_delivery)
            remaining = self._producer.flush(self._flush_timeout_s)
        except (BufferError, self._mod.KafkaException) as exc:
            raise TransportError(f"produce to {topic} failed: {exc}") from exc
        if remaining or "msg" not in report:
            raise TransportError(f"delivery to {topic} not confirmed within {self._flush_timeout_s}s")
        if report["err"] is not None:
            raise TransportError(f"delivery to {topic} failed: {report['err']}")
        msg = report["msg"]
        return {"topic": msg.topic(), "partition": msg.partition(), "offset": msg.offset()}

    def subscribe(self, topics: Iterable[str]) -> None:
        if self._consumer is None:
            self._consumer = self._mod.Consumer(self.settings.consumer_config())
        self._consumer.subscribe(list(topics))

    def _require_consumer(self) -> Any:
        if self._consumer is None:
            raise TransportError("subscribe() must be called before consuming")
        return self._consumer

    def poll(self, max_records: int, timeout_s: float) -> list[TransportRecord]:
        consumer = self._require_consumer()
        try:
            messages = consumer.consume(num_messages=max_records, timeout=timeout_s)
        except self._mod.KafkaException as exc:
            raise TransportError(f"consume failed: {exc}") from exc
        out: list[TransportRecord] = []
        for msg in messages:
            err = msg.error()
            if err is not None:
                if err.code() == self._mod.KafkaError._PARTITION_EOF:
                    continue
                raise TransportError(f"consume error: {err}")
            out.append(TransportRecord(msg.topic(), msg.partition(), msg.offset(), msg.key(), msg.value() or b""))
        return out

    def commit(self, record: TransportRecord) -> None:
        consumer = self._require_consumer()
        tp = self._mod.TopicPartition(record.topic, record.partition, record.offset + 1)
        try:
            consumer.commit(offsets=[tp], asynchronous=False)
        except self._mod.KafkaException as exc:
            raise TransportError(f"commit failed: {exc}") from exc

    def seek(self, topic: str, partition: int, offset: int) -> None:
        consumer = self._require_consumer()
        try:
            consumer.seek(self._mod.TopicPartition(topic, partition, offset))
        except self._mod.KafkaException as exc:
            raise TransportError(f"seek failed: {exc}") from exc

    def close(self) -> None:
        self._producer.flush(self._flush_timeout_s)
        if self._consumer is not None:
            self._consumer.close()
            self._consumer = None


# ---------------------------------------------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------------------------------------------


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON bytes: sorted keys, no whitespace, UTF-8, NaN/Infinity refused."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class HMACKeyring:
    """Signing keys by key id; ``active_kid`` signs, every key in ``keys`` verifies (rotation window)."""

    keys: Mapping[str, bytes] = field(repr=False)
    active_kid: str

    def __post_init__(self) -> None:
        if not self.keys:
            raise KafkaConfigError("keyring is empty")
        for kid, key in self.keys.items():
            if not _KID_RE.match(kid):
                raise KafkaConfigError(f"invalid key id {kid!r}")
            if len(key) < MIN_HMAC_KEY_BYTES:
                raise KafkaConfigError(f"key {kid!r} is shorter than {MIN_HMAC_KEY_BYTES} bytes")
        if self.active_kid not in self.keys:
            raise KafkaConfigError(f"active key id {self.active_kid!r} is not in the keyring")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> HMACKeyring:
        """Parse ``HIVEMIND_AMCP_HMAC_KEYS="kid1:<hex>,kid2:<hex>"`` and ``HIVEMIND_AMCP_HMAC_ACTIVE_KID``."""
        src = os.environ if env is None else env
        raw = src.get("HIVEMIND_AMCP_HMAC_KEYS", "").strip()
        if not raw:
            raise KafkaConfigError("HIVEMIND_AMCP_HMAC_KEYS is not set")
        keys: dict[str, bytes] = {}
        for item in raw.split(","):
            kid, sep, hex_key = item.strip().partition(":")
            if not sep:
                raise KafkaConfigError("HIVEMIND_AMCP_HMAC_KEYS entries must be kid:hexkey")
            try:
                keys[kid] = bytes.fromhex(hex_key)
            except ValueError as exc:
                raise KafkaConfigError(f"key {kid!r} is not valid hex") from exc
        active = src.get("HIVEMIND_AMCP_HMAC_ACTIVE_KID") or next(iter(keys))
        return cls(keys=keys, active_kid=active)

    def sign(self, message: bytes, kid: str | None = None) -> str:
        key = self.keys[kid or self.active_kid]
        return hmac.new(key, message, hashlib.sha256).hexdigest()


def _rfc3339(utc: float) -> str:
    return datetime.fromtimestamp(utc, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def sign_envelope(event: AMCPKafkaEvent, keyring: HMACKeyring, ttl_s: float = DEFAULT_EVENT_TTL_S) -> dict[str, Any]:
    """Wrap ``event`` in a CloudEvents 1.0 envelope signed with the keyring's active key."""
    if not (0.0 < ttl_s <= MAX_EVENT_TTL_S):
        raise ValueError(f"ttl_s must be in (0, {MAX_EVENT_TTL_S}]")
    envelope: dict[str, Any] = {
        "specversion": "1.0",
        "id": event.event_id,
        "source": f"hivemind://{event.sender_kind}/{event.sender_id}",
        "type": event.event_type,
        "subject": event.subject_id,
        "time": _rfc3339(event.utc),
        "datacontenttype": "application/json",
        "data": event.to_dict(),
        "hmkeyid": keyring.active_kid,
        "hmexpiry": round(event.utc + ttl_s, 6),
    }
    envelope["hmsig"] = keyring.sign(canonical_json(envelope))
    return envelope


# ---------------------------------------------------------------------------------------------------------------
# Per-topic schema
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldRule:
    """Constraint on one payload field."""

    kind: Literal["str", "number"]
    choices: tuple[str, ...] | None = None
    ge: float | None = None
    gt: float | None = None
    le: float | None = None
    max_len: int = 256

    def check(self, name: str, value: Any) -> None:
        if self.kind == "str":
            if not isinstance(value, str):
                raise SchemaError(f"payload.{name} must be a string")
            if len(value) > self.max_len:
                raise SchemaError(f"payload.{name} longer than {self.max_len}")
            if self.choices is not None and value not in self.choices:
                raise SchemaError(f"payload.{name}={value!r} not in {list(self.choices)}")
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise SchemaError(f"payload.{name} must be a finite number")
        if self.ge is not None and value < self.ge:
            raise SchemaError(f"payload.{name} must be >= {self.ge}")
        if self.gt is not None and value <= self.gt:
            raise SchemaError(f"payload.{name} must be > {self.gt}")
        if self.le is not None and value > self.le:
            raise SchemaError(f"payload.{name} must be <= {self.le}")


@dataclass(frozen=True)
class TopicSchema:
    """Schema of one AMCP topic: which event types it carries and the payload rules of each."""

    topic_key: str
    payloads: Mapping[str, Mapping[str, FieldRule]]
    open_event_types: bool = False
    require_known_subject: bool = False


_TRADE_FIELDS: dict[str, FieldRule] = {
    "order_id": FieldRule("str", max_len=64),
    "capacity_1k": FieldRule("number", gt=0.0, le=100_000.0),
    "price_per_1k": FieldRule("number", gt=0.0),
}

TOPIC_SCHEMAS: dict[str, TopicSchema] = {
    KAFKA_TOPICS["campaigns"]: TopicSchema(
        topic_key="campaigns",
        payloads={
            "io.hivemind.amcp.campaign.poll_registered": {
                "pledged_vram_gb": FieldRule("number", ge=0.0, le=1024.0),
                "availability_slot": FieldRule("str", choices=AVAILABILITY_SLOTS),
                "comment": FieldRule("str", max_len=256),
            },
        },
        require_known_subject=True,
    ),
    KAFKA_TOPICS["human_agent"]: TopicSchema(topic_key="human_agent", payloads={}, open_event_types=True),
    KAFKA_TOPICS["bee_coord"]: TopicSchema(
        topic_key="bee_coord",
        payloads={
            "io.hivemind.amcp.bee.handoff_request": {
                "task_domain": FieldRule("str", max_len=64),
                "language": FieldRule("str", max_len=32),
                "delegated_action": FieldRule("str", max_len=128),
            },
            "io.hivemind.amcp.bee.handoff_certified": {
                "solver_family": FieldRule("str", max_len=128),
                "domain": FieldRule("str", max_len=64),
                "recommended_crate": FieldRule("str", max_len=128),
                "relative_invariant_drift": FieldRule("number", ge=0.0),
                "improvement_factor": FieldRule("number"),
                "verdict": FieldRule("str", choices=("PASS", "FAIL")),
            },
        },
    ),
    KAFKA_TOPICS["bee_trading"]: TopicSchema(
        topic_key="bee_trading",
        payloads={
            "io.hivemind.amcp.trade.offer": {**_TRADE_FIELDS, "side": FieldRule("str", choices=("OFFER",))},
            "io.hivemind.amcp.trade.bid": {**_TRADE_FIELDS, "side": FieldRule("str", choices=("BID",))},
        },
    ),
}

_EVENT_STR_FIELDS: dict[str, int] = {
    "event_id": 64,
    "topic": 128,
    "event_type": 128,
    "sender_id": 64,
    "sender_kind": 32,
    "recipient_id": 64,
    "subject_id": 64,
    "correlation_id": 64,
    "specversion": 8,
    "protocol": 32,
}
_EVENT_FIELDS: frozenset[str] = frozenset({*_EVENT_STR_FIELDS, "payload", "utc"})


def validate_event_dict(data: Any, topic: str) -> AMCPKafkaEvent:
    """Validate an AMCP event dict against the schema of ``topic`` and return the typed event."""
    if not isinstance(data, dict):
        raise SchemaError("data must be a JSON object")
    keys = set(data)
    if keys != _EVENT_FIELDS:
        missing, extra = sorted(_EVENT_FIELDS - keys), sorted(keys - _EVENT_FIELDS)
        raise SchemaError(f"event fields mismatch (missing={missing}, unexpected={extra})")
    for name, max_len in _EVENT_STR_FIELDS.items():
        value = data[name]
        if not isinstance(value, str) or not value or len(value) > max_len:
            raise SchemaError(f"{name} must be a non-empty string of at most {max_len} chars")
    utc = data["utc"]
    if isinstance(utc, bool) or not isinstance(utc, (int, float)) or not math.isfinite(utc):
        raise SchemaError("utc must be a finite number")
    if not isinstance(data["payload"], dict):
        raise SchemaError("payload must be a JSON object")
    if data["specversion"] != "1.0":
        raise SchemaError("specversion must be 1.0")
    if data["topic"] != topic:
        raise SchemaError(f"event topic {data['topic']!r} does not match {topic!r}")
    schema = TOPIC_SCHEMAS.get(topic)
    if schema is None:
        raise SchemaError(f"no schema for topic {topic!r}")
    if data["sender_kind"] not in SENDER_KINDS:
        raise SchemaError(f"sender_kind {data['sender_kind']!r} not in {sorted(SENDER_KINDS)}")
    event_type = data["event_type"]
    if not event_type.startswith(EVENT_TYPE_PREFIX):
        raise SchemaError(f"event_type must start with {EVENT_TYPE_PREFIX}")
    if schema.require_known_subject and data["subject_id"] not in SUBJECT_BY_ID:
        raise SchemaError(f"unknown campaign subject {data['subject_id']!r}")
    rules = schema.payloads.get(event_type)
    if rules is None:
        if not schema.open_event_types:
            raise SchemaError(f"event_type {event_type!r} is not allowed on {topic}")
    else:
        payload = data["payload"]
        missing_fields = sorted(set(rules) - set(payload))
        if missing_fields:
            raise SchemaError(f"payload missing {missing_fields}")
        for name, rule in rules.items():
            rule.check(name, payload[name])
    return AMCPKafkaEvent(**data)


def partition_key_for(event: AMCPKafkaEvent) -> str:
    """Kafka record key: the campaign (subject) or bee identity whose events must stay ordered."""
    topic_key = TOPIC_KEY_BY_NAME.get(event.topic)
    if topic_key == "bee_coord":
        return event.correlation_id
    if topic_key == "bee_trading":
        return event.sender_id
    return event.subject_id


# ---------------------------------------------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifiedEvent:
    """An event that passed signature, schema and freshness checks, with where it came from."""

    event: AMCPKafkaEvent
    key_id: str
    record: TransportRecord


class EnvelopeVerifier:
    """Verify consumed records; raise ``EnvelopeRejected`` with a dead-letter reason on any failure."""

    def __init__(
        self,
        keyring: HMACKeyring,
        *,
        max_ttl_s: float = MAX_EVENT_TTL_S,
        max_skew_s: float = MAX_CLOCK_SKEW_S,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if max_ttl_s <= 0 or max_skew_s < 0:
            raise ValueError("max_ttl_s must be > 0 and max_skew_s >= 0")
        self.keyring = keyring
        self.max_ttl_s = float(max_ttl_s)
        self.max_skew_s = float(max_skew_s)
        self._clock = clock

    def verify(self, record: TransportRecord) -> VerifiedEvent:
        if len(record.value) > MAX_ENVELOPE_BYTES:
            raise EnvelopeRejected("oversize", f"{len(record.value)} bytes > {MAX_ENVELOPE_BYTES}")
        try:
            envelope = json.loads(record.value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EnvelopeRejected("malformed", str(exc)[:200]) from exc
        if not isinstance(envelope, dict):
            raise EnvelopeRejected("malformed", "envelope is not a JSON object")
        if envelope.get("specversion") != "1.0" or not all(k in envelope for k in ("id", "type", "source", "data")):
            raise EnvelopeRejected("not_cloudevent", "missing specversion 1.0 / id / type / source / data")
        sig = envelope.get("hmsig")
        kid = envelope.get("hmkeyid")
        if sig is None or kid is None:
            raise EnvelopeRejected("unsigned", "hmsig or hmkeyid missing")
        if not isinstance(kid, str) or kid not in self.keyring.keys:
            raise EnvelopeRejected("unknown_key_id", f"key id {str(kid)[:64]!r} not in keyring")
        if not isinstance(sig, str) or not _SIG_RE.match(sig):
            raise EnvelopeRejected("bad_signature", "hmsig is not a 64-char lowercase hex digest")
        unsigned = {k: v for k, v in envelope.items() if k != "hmsig"}
        try:
            expected = self.keyring.sign(canonical_json(unsigned), kid)
        except ValueError as exc:
            raise EnvelopeRejected("malformed", f"not canonicalisable: {exc}") from exc
        if not hmac.compare_digest(expected, sig):
            raise EnvelopeRejected("bad_signature", "HMAC-SHA256 mismatch")
        data = envelope["data"]
        if isinstance(data, dict) and data.get("topic") != record.topic:
            raise EnvelopeRejected("topic_mismatch", f"signed for {data.get('topic')!r}, arrived on {record.topic!r}")
        try:
            event = validate_event_dict(data, record.topic)
        except SchemaError as exc:
            raise EnvelopeRejected("schema_invalid", str(exc)[:300]) from exc
        if (envelope["id"], envelope["type"], envelope.get("subject")) != (
            event.event_id,
            event.event_type,
            event.subject_id,
        ):
            raise EnvelopeRejected("envelope_mismatch", "id/type/subject differ from data")
        expiry = envelope.get("hmexpiry")
        if isinstance(expiry, bool) or not isinstance(expiry, (int, float)) or not math.isfinite(expiry):
            raise EnvelopeRejected("unsigned", "hmexpiry missing or not a finite number")
        if expiry - event.utc > self.max_ttl_s:
            raise EnvelopeRejected("ttl_too_long", f"lifetime {expiry - event.utc:.0f}s > {self.max_ttl_s:.0f}s")
        now = self._clock()
        if event.utc > now + self.max_skew_s:
            raise EnvelopeRejected("future_dated", f"event time is {event.utc - now:.1f}s in the future")
        if now > expiry:
            raise EnvelopeRejected("expired", f"expired {now - expiry:.1f}s ago")
        return VerifiedEvent(event=event, key_id=kid, record=record)


# ---------------------------------------------------------------------------------------------------------------
# Publisher, dead letters, replay window
# ---------------------------------------------------------------------------------------------------------------


class AMCPPublisher:
    """Validate, sign and publish AMCP events; optionally mirror them into the gateway's read-model bridge."""

    def __init__(
        self,
        transport: Transport,
        keyring: HMACKeyring,
        *,
        ttl_s: float = DEFAULT_EVENT_TTL_S,
        mirror: InMemoryKafkaBridge | None = None,
    ) -> None:
        if not (0.0 < ttl_s <= MAX_EVENT_TTL_S):
            raise ValueError(f"ttl_s must be in (0, {MAX_EVENT_TTL_S}]")
        self.transport = transport
        self.keyring = keyring
        self.ttl_s = float(ttl_s)
        self.mirror = mirror

    def publish(self, event: AMCPKafkaEvent) -> dict[str, Any]:
        validate_event_dict(event.to_dict(), event.topic)
        value = canonical_json(sign_envelope(event, self.keyring, self.ttl_s))
        if len(value) > MAX_ENVELOPE_BYTES:
            raise SchemaError(f"signed envelope is {len(value)} bytes > {MAX_ENVELOPE_BYTES}")
        key = partition_key_for(event)
        ack = self.transport.publish(event.topic, key.encode("utf-8"), value)
        if self.mirror is not None:
            self.mirror.publish(event)
        return {**ack, "event_id": event.event_id, "key": key, "key_id": self.keyring.active_kid}


def build_dead_letter(record: TransportRecord, reason: str, detail: str, now_utc: float) -> bytes:
    """DLQ record: reason, detail, source coordinates and the raw key/value (base64, value capped at 64 KiB)."""
    value = record.value[:DLQ_VALUE_CAP_BYTES]
    return canonical_json(
        {
            "specversion": "1.0",
            "type": "io.hivemind.amcp.dead_letter",
            "reason": reason,
            "detail": detail[:512],
            "source_topic": record.topic,
            "source_partition": record.partition,
            "source_offset": record.offset,
            "key_b64": base64.b64encode(record.key or b"").decode("ascii"),
            "value_b64": base64.b64encode(value).decode("ascii"),
            "value_truncated": len(record.value) > DLQ_VALUE_CAP_BYTES,
            "utc": now_utc,
        }
    )


class ReplayWindow:
    """Bounded set of handled event ids with a TTL; oldest entries are evicted first when full."""

    def __init__(self, capacity: int = 100_000, ttl_s: float = MAX_EVENT_TTL_S + MAX_CLOCK_SKEW_S) -> None:
        if capacity < 1 or ttl_s <= 0:
            raise ValueError("capacity must be >= 1 and ttl_s > 0")
        self.capacity = capacity
        self.ttl_s = float(ttl_s)
        self.evictions = 0
        self._seen: OrderedDict[str, float] = OrderedDict()

    def _prune(self, now: float) -> None:
        while self._seen:
            oldest_id, seen_at = next(iter(self._seen.items()))
            if now - seen_at <= self.ttl_s:
                break
            del self._seen[oldest_id]

    def seen(self, event_id: str, now: float) -> bool:
        self._prune(now)
        return event_id in self._seen

    def mark(self, event_id: str, now: float) -> None:
        self._seen[event_id] = now
        self._seen.move_to_end(event_id)
        while len(self._seen) > self.capacity:
            self._seen.popitem(last=False)
            self.evictions += 1

    def __len__(self) -> int:
        return len(self._seen)


# ---------------------------------------------------------------------------------------------------------------
# Consumer loop
# ---------------------------------------------------------------------------------------------------------------

Handler = Callable[[VerifiedEvent], None]


@dataclass
class ConsumerStats:
    """Counters since the consumer was created."""

    polled: int = 0
    delivered: int = 0
    duplicates_dropped: int = 0
    handler_failures: int = 0
    rewinds: int = 0
    committed: int = 0
    commit_errors: int = 0
    dlq_errors: int = 0
    dead_lettered: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "polled": self.polled,
            "delivered": self.delivered,
            "duplicates_dropped": self.duplicates_dropped,
            "handler_failures": self.handler_failures,
            "rewinds": self.rewinds,
            "committed": self.committed,
            "commit_errors": self.commit_errors,
            "dlq_errors": self.dlq_errors,
            "dead_lettered": dict(self.dead_lettered),
        }


class AMCPConsumer:
    """At-least-once AMCP consumer: verify, dedupe, dispatch to a registered handler, then commit."""

    def __init__(
        self,
        transport: Transport,
        verifier: EnvelopeVerifier,
        *,
        topics: Iterable[str] | None = None,
        replay_window: ReplayWindow | None = None,
        max_batch: int = 64,
        max_attempts: int = 3,
        poll_timeout_s: float = 0.5,
        dlq_topic: str = DLQ_TOPIC,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not 1 <= max_batch <= 10_000:
            raise ValueError("max_batch must be in [1, 10000]")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        window = replay_window if replay_window is not None else ReplayWindow()
        if window.ttl_s < verifier.max_ttl_s + verifier.max_skew_s:
            raise ValueError("replay window TTL must cover max event lifetime + clock skew, or replays can slip through")
        resolved = tuple(KAFKA_TOPICS.get(t, t) for t in (topics if topics is not None else KAFKA_TOPICS.values()))
        unknown = [t for t in resolved if t not in TOPIC_SCHEMAS]
        if unknown or not resolved:
            raise ValueError(f"topics must be AMCP topics with a schema; unknown: {unknown}")
        if dlq_topic in resolved:
            raise ValueError("the consumer must not subscribe to its own dead-letter topic")
        self.transport = transport
        self.verifier = verifier
        self.topics = resolved
        self.window = window
        self.max_batch = max_batch
        self.max_attempts = max_attempts
        self.poll_timeout_s = float(poll_timeout_s)
        self.dlq_topic = dlq_topic
        self.stats = ConsumerStats()
        self._clock = clock
        self._handlers: dict[str, Handler] = {}
        self._topic_handlers: dict[str, Handler] = {}
        self._attempts: OrderedDict[tuple[str, int, int], int] = OrderedDict()
        self._stop = threading.Event()
        self._subscribed = False

    def register(self, event_type: str, handler: Handler) -> None:
        """Route one event type to ``handler``; registering the same type twice is an error."""
        if not event_type.startswith(EVENT_TYPE_PREFIX):
            raise ValueError(f"event_type must start with {EVENT_TYPE_PREFIX}")
        if event_type in self._handlers:
            raise ValueError(f"handler already registered for {event_type}")
        self._handlers[event_type] = handler

    def register_topic(self, topic: str, handler: Handler) -> None:
        """Fallback handler for every event type on ``topic`` (key or full name) without an exact handler."""
        name = KAFKA_TOPICS.get(topic, topic)
        if name not in self.topics:
            raise ValueError(f"{name} is not one of this consumer's topics")
        if name in self._topic_handlers:
            raise ValueError(f"topic handler already registered for {name}")
        self._topic_handlers[name] = handler

    def stop(self) -> None:
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def _dead_letter(self, record: TransportRecord, reason: str, detail: str) -> bool:
        try:
            self.transport.publish(self.dlq_topic, record.key or b"", build_dead_letter(record, reason, detail, self._clock()))
        except TransportError:
            self.stats.dlq_errors += 1
            return False
        self.stats.dead_lettered[reason] = self.stats.dead_lettered.get(reason, 0) + 1
        return True

    def _commit(self, record: TransportRecord) -> None:
        try:
            self.transport.commit(record)
        except TransportError:
            self.stats.commit_errors += 1
            return
        self.stats.committed += 1

    def _rewind(self, record: TransportRecord, held: set[tuple[str, int]]) -> None:
        self.transport.seek(record.topic, record.partition, record.offset)
        held.add((record.topic, record.partition))
        self.stats.rewinds += 1

    def _bump_attempts(self, coords: tuple[str, int, int]) -> int:
        count = self._attempts.pop(coords, 0) + 1
        self._attempts[coords] = count
        while len(self._attempts) > 10_000:
            self._attempts.popitem(last=False)
        return count

    def _process(self, record: TransportRecord, held: set[tuple[str, int]]) -> None:
        try:
            verified = self.verifier.verify(record)
        except EnvelopeRejected as rej:
            if self._dead_letter(record, rej.reason, rej.detail):
                self._commit(record)
            else:
                self._rewind(record, held)
            return
        event = verified.event
        if self.window.seen(event.event_id, self._clock()):
            self.stats.duplicates_dropped += 1
            self._commit(record)
            return
        handler = self._handlers.get(event.event_type) or self._topic_handlers.get(record.topic)
        if handler is None:
            if self._dead_letter(record, "no_handler", f"no handler for {event.event_type}"):
                self._commit(record)
            else:
                self._rewind(record, held)
            return
        coords = (record.topic, record.partition, record.offset)
        try:
            handler(verified)
        except Exception as exc:  # noqa: BLE001 - any handler failure must keep the offset uncommitted
            self.stats.handler_failures += 1
            if self._bump_attempts(coords) >= self.max_attempts:
                detail = f"{type(exc).__name__}: {exc}"[:512]
                if self._dead_letter(record, "handler_failed", detail):
                    self._attempts.pop(coords, None)
                    self._commit(record)
                    return
            self._rewind(record, held)
            return
        self._attempts.pop(coords, None)
        self.window.mark(event.event_id, self._clock())
        self.stats.delivered += 1
        self._commit(record)

    def run_once(self) -> int:
        """Poll one bounded batch and process it; return the number of records polled."""
        if not self._subscribed:
            self.transport.subscribe(self.topics)
            self._subscribed = True
        records = self.transport.poll(self.max_batch, self.poll_timeout_s)
        self.stats.polled += len(records)
        held: set[tuple[str, int]] = set()
        for record in records:
            if self._stop.is_set():
                self._rewind(record, held)
                break
            if (record.topic, record.partition) in held:
                continue
            self._process(record, held)
        return len(records)

    def run(self, max_batches: int | None = None, idle_sleep_s: float = 0.0) -> ConsumerStats:
        """Loop until ``stop()`` (or ``max_batches`` polls); sleep ``idle_sleep_s`` after an empty poll."""
        batches = 0
        while not self._stop.is_set() and (max_batches is None or batches < max_batches):
            got = self.run_once()
            batches += 1
            if got == 0 and idle_sleep_s > 0:
                self._stop.wait(idle_sleep_s)
        return self.stats
