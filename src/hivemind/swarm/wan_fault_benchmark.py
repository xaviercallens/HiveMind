# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HM-S5-05: multi-host WAN QUIC 0-RTT & RFC 5766 TURN fault-injection benchmark.

The benchmark drives the real transports of this repository through a deterministic WAN
impairment:

- `hivemind.swarm.quic_link`: a real `aioquic` TLS 1.3 0-RTT resumption (RFC 9001 §4.6) and
  the libp2p multistream-select WebSocket fallback (RFC 6455).
- `hivemind.swarm.nat_traversal`: a real RFC 5766 TURN relay, both its Send/Data indication
  path and its ChannelBind/ChannelData path (RFC 5766 §11).
- `hivemind.swarm.quic_transport`: the monotonic 0-RTT strike register
  (`AsyncQUICGeoRingTransport.verify_0rtt_ticket_anti_replay`) and the BBRv1-inspired
  congestion controller (`AdaptiveCongestionController`).

Every transport runs behind `nat_traversal.DeterministicFaultProxy`, a seeded UDP proxy that
drops, delays and reorders datagrams and can rebind the client's NAT mapping. The resulting
telemetry is written to `artifacts/wan_quic_turn_fault_report.json` and reports, per fault
profile, the p50/p95/p99 frame delivery latency, the 0-RTT strike-register rejection rate and
the BBRv1 pacing adaptation.

Run it with `python -m hivemind.swarm.wan_fault_benchmark`.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import math
import os
import platform
import socket
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hivemind.swarm.nat_traversal import (
    Address,
    DeterministicFaultProxy,
    FaultProfile,
    StunError,
    TurnClient,
    TurnRelayServer,
)
from hivemind.swarm.quic_link import (
    SERVER_NAME,
    FrameEnvelope,
    LinkEndpoint,
    LinkIdentity,
    LinkProtocolError,
    QuicHiddenStateServer,
    RemoteHiddenStateLink,
    WebSocketHiddenStateServer,
    quic_0rtt_resumption_probe,
    quic_send_frames,
    websocket_send_frames,
)

REPORT_SCHEMA = "hivemind.wan_quic_turn_fault_report/v1"
REPORT_NAME = "wan_quic_turn_fault_report.json"
DEFAULT_LOSS_RATES: Tuple[float, ...] = (0.0, 0.01, 0.03, 0.05)
DEFAULT_REORDER_RATE = 0.02
DEFAULT_JITTER_MS = 1.0
DEFAULT_FRAMES = 24
DEFAULT_FRAME_BYTES = 512
DEFAULT_SEED = 47001
STRIKE_PEER_ID = "wan-fault-bench-peer"
ENVELOPE_OVERHEAD = 12


# ---------------------------------------------------------------------------
# Telemetry helpers
# ---------------------------------------------------------------------------
def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile of `values` (deterministic, without interpolation)."""
    if isinstance(values, (str, bytes)):
        raise ValueError("values must be a sequence of numbers, not str/bytes")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(float(fraction))
        or not (0.0 < float(fraction) <= 1.0)
    ):
        raise ValueError("percentile fraction must be a finite number in (0, 1]")
    if not values:
        return 0.0
    clean: List[float] = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise ValueError(f"values must contain only finite numbers, got {v!r}")
        clean.append(float(v))
    ordered = sorted(clean)
    index = max(0, min(len(ordered) - 1, math.ceil(float(fraction) * len(ordered)) - 1))
    return ordered[index]


def latency_summary(values: Sequence[float]) -> Dict[str, float]:
    """Summarise a latency sample with the p50/p95/p99 the report has to carry."""
    if isinstance(values, (str, bytes)):
        raise ValueError("values must be a sequence of numbers, not str/bytes")
    if not values:
        return {
            "count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0,
            "min_ms": 0.0, "max_ms": 0.0, "mean_ms": 0.0,
        }
    clean: List[float] = []
    for v in values:
        if (
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(float(v))
            or float(v) < 0.0
        ):
            raise ValueError(f"latencies must be finite and >= 0, got {v!r}")
        clean.append(float(v))
    ordered = sorted(clean)
    return {
        "count": len(ordered),
        "p50_ms": round(percentile(ordered, 0.50), 3),
        "p95_ms": round(percentile(ordered, 0.95), 3),
        "p99_ms": round(percentile(ordered, 0.99), 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
        "mean_ms": round(sum(ordered) / len(ordered), 3),
    }


def _throughput_mbps(payload_bytes: int, elapsed_s: float) -> float:
    """Payload throughput of a transfer in Mbit/s."""
    if isinstance(payload_bytes, bool) or not isinstance(payload_bytes, int) or payload_bytes < 0:
        raise ValueError("payload_bytes must be a non-negative integer")
    if (
        isinstance(elapsed_s, bool)
        or not isinstance(elapsed_s, (int, float))
        or not math.isfinite(float(elapsed_s))
    ):
        raise ValueError("elapsed_s must be a finite number")
    if elapsed_s <= 0:
        return 0.0
    return round((payload_bytes * 8.0) / elapsed_s / 1_000_000.0, 4)


def _unused_udp_port(host: str = "127.0.0.1") -> int:
    """A UDP port nothing listens on (used to force the WebSocket fallback)."""
    if not isinstance(host, str) or not host.strip():
        raise ValueError("host must be a non-empty string")
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.bind((host, 0))
        return probe.getsockname()[1]
    finally:
        probe.close()


@dataclass(frozen=True)
class FaultScenario:
    """One deterministic fault profile replayed against every transport."""

    scenario_id: str
    profile: FaultProfile
    nat_rebinds: int = 1
    frames: int = DEFAULT_FRAMES
    frame_bytes: int = DEFAULT_FRAME_BYTES

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise ValueError("scenario_id must be a non-empty string")
        if not isinstance(self.profile, FaultProfile):
            raise ValueError("profile must be a FaultProfile instance")
        if isinstance(self.nat_rebinds, bool) or not isinstance(self.nat_rebinds, int) or self.nat_rebinds < 0:
            raise ValueError("nat_rebinds must be an integer >= 0")
        if isinstance(self.frames, bool) or not isinstance(self.frames, int) or self.frames < 1:
            raise ValueError("frames must be an integer >= 1")
        if isinstance(self.frame_bytes, bool) or not isinstance(self.frame_bytes, int) or self.frame_bytes < 1:
            raise ValueError("frame_bytes must be an integer >= 1")

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict view of the scenario (JSON friendly)."""
        return {
            "scenario_id": self.scenario_id,
            "packet_loss_rate": float(self.profile.loss_rate),
            "reorder_rate": float(self.profile.reorder_rate),
            "jitter_ms": float(self.profile.jitter_ms),
            "seed": int(self.profile.seed),
            "nat_rebind_events": int(self.nat_rebinds),
            "frames_per_batch": int(self.frames),
            "frame_payload_bytes": int(self.frame_bytes),
        }


@dataclass(frozen=True)
class ExternalEndpoints:
    """Remote receivers, so the same suite can run from another host (multi-host mode).

    The environment variables are optional: without them the benchmark runs every transport
    locally and simulates the WAN path with the deterministic fault proxy.

    - `HIVEMIND_WAN_QUIC_HOST` / `HIVEMIND_WAN_QUIC_PORT`: a remote QUIC hidden-state
      receiver. `HIVEMIND_WAN_QUIC_CA_PEM` (a PEM or a path to one) and
      `HIVEMIND_WAN_QUIC_SERVER_NAME` describe its trust anchor.
    - `HIVEMIND_WAN_TURN_HOST` / `HIVEMIND_WAN_TURN_PORT`: a remote RFC 5766 TURN server,
      with `HIVEMIND_WAN_TURN_USERNAME` and `HIVEMIND_WAN_TURN_PASSWORD`.
    """

    quic_host: Optional[str] = None
    quic_port: Optional[int] = None
    quic_ca_pem: Optional[str] = None
    quic_server_name: Optional[str] = None
    turn_host: Optional[str] = None
    turn_port: Optional[int] = None
    turn_username: Optional[str] = None
    turn_password: Optional[str] = None

    def __post_init__(self) -> None:
        for s_name in (
            "quic_host",
            "quic_ca_pem",
            "quic_server_name",
            "turn_host",
            "turn_username",
            "turn_password",
        ):
            val = getattr(self, s_name)
            if val is not None and (not isinstance(val, str) or not val.strip()):
                raise ValueError(f"{s_name} must be None or a non-empty string")
        for p_name in ("quic_port", "turn_port"):
            port = getattr(self, p_name)
            if port is not None and (isinstance(port, bool) or not isinstance(port, int) or not (1 <= port <= 65535)):
                raise ValueError(f"{p_name} must be None or an integer in 1..65535")

    @property
    def has_quic(self) -> bool:
        """True when a remote QUIC receiver is configured."""
        return bool(self.quic_host and self.quic_port and self.quic_ca_pem)

    @property
    def has_turn(self) -> bool:
        """True when a remote TURN server is configured."""
        return bool(self.turn_host and self.turn_port and self.turn_username)

    @classmethod
    def from_environment(cls, environ: Optional[Dict[str, str]] = None) -> "ExternalEndpoints":
        """Read the optional remote endpoints from the environment."""
        env = environ if environ is not None else dict(os.environ)
        ca_pem = env.get("HIVEMIND_WAN_QUIC_CA_PEM")
        if ca_pem and "-----BEGIN" not in ca_pem:
            candidate = Path(ca_pem)
            ca_pem = candidate.read_text(encoding="utf-8") if candidate.is_file() else None
        turn_port = env.get("HIVEMIND_WAN_TURN_PORT")
        quic_port = env.get("HIVEMIND_WAN_QUIC_PORT")
        return cls(
            quic_host=env.get("HIVEMIND_WAN_QUIC_HOST"),
            quic_port=int(quic_port) if quic_port and quic_port.isdigit() else None,
            quic_ca_pem=ca_pem,
            quic_server_name=env.get("HIVEMIND_WAN_QUIC_SERVER_NAME"),
            turn_host=env.get("HIVEMIND_WAN_TURN_HOST"),
            turn_port=int(turn_port) if turn_port and turn_port.isdigit() else None,
            turn_username=env.get("HIVEMIND_WAN_TURN_USERNAME"),
            turn_password=env.get("HIVEMIND_WAN_TURN_PASSWORD"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict view of the endpoints (JSON friendly, without secrets)."""
        return {
            "mode": "multi_host" if (self.has_quic or self.has_turn) else "local_wan_simulation",
            "quic_endpoint": (
                f"{self.quic_host}:{self.quic_port}" if self.has_quic else None
            ),
            "turn_endpoint": (
                f"{self.turn_host}:{self.turn_port}" if self.has_turn else None
            ),
        }


def default_scenarios(
    loss_rates: Sequence[float] = DEFAULT_LOSS_RATES,
    seed: int = DEFAULT_SEED,
    frames: int = DEFAULT_FRAMES,
    frame_bytes: int = DEFAULT_FRAME_BYTES,
) -> List[FaultScenario]:
    """The `0%`, `1%`, `3%` and `5%` packet-loss profiles the bounty asks for."""
    if isinstance(loss_rates, (str, bytes)) or not loss_rates:
        raise ValueError("loss_rates must be a non-empty sequence of floats in [0.0, 1.0)")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be an integer >= 0")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        raise ValueError("frames must be an integer >= 1")
    if isinstance(frame_bytes, bool) or not isinstance(frame_bytes, int) or frame_bytes < 1:
        raise ValueError("frame_bytes must be an integer >= 1")
    scenarios = []
    for index, loss in enumerate(loss_rates):
        if (
            isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
            or float(loss) < 0.0
            or float(loss) >= 1.0
        ):
            raise ValueError(f"loss_rate must be a finite number in [0.0, 1.0), got {loss!r}")
        scenarios.append(
            FaultScenario(
                scenario_id=f"loss_{int(round(loss * 100))}pct",
                profile=FaultProfile(
                    loss_rate=float(loss),
                    reorder_rate=DEFAULT_REORDER_RATE,
                    jitter_ms=DEFAULT_JITTER_MS,
                    seed=seed + index,
                    label=f"{int(round(loss * 100))}% loss",
                ),
                frames=frames,
                frame_bytes=frame_bytes,
            )
        )
    return scenarios


# ---------------------------------------------------------------------------
# 0-RTT strike register
# ---------------------------------------------------------------------------
class StrikeRegister:
    """The application-level monotonic 0-RTT strike register of the swarm transport.

    A client that retries a 0-RTT send after an unacknowledged batch replays its nonce.
    The register must reject that replay (`verify_0rtt_ticket_anti_replay`), which is what
    the report counts as the 0-RTT strike-register rejection rate.
    """

    def __init__(self, peer_id: str = STRIKE_PEER_ID) -> None:
        if not isinstance(peer_id, str) or not peer_id.strip():
            raise ValueError("peer_id must be a non-empty string")
        from hivemind.swarm.quic_transport import AsyncQUICGeoRingTransport

        self.peer_id = peer_id
        self._transport = AsyncQUICGeoRingTransport()
        self._transport.issue_0rtt_ticket(peer_id)
        self.nonce = 0
        self.attempts = 0
        self.accepted = 0
        self.rejected = 0
        self.replay_probes = 0
        self.reasons: List[str] = []

    def open_session(self) -> int:
        """Register a fresh nonce for a new 0-RTT attempt."""
        self.nonce += 1
        return self.nonce

    def register(self, nonce: int) -> bool:
        """Offer `nonce` to the strike register and record the outcome."""
        if isinstance(nonce, bool) or not isinstance(nonce, int) or nonce < 1:
            raise ValueError("nonce must be an integer >= 1")
        self.attempts += 1
        accepted, reason = self._transport.verify_0rtt_ticket_anti_replay(self.peer_id, nonce)
        if accepted:
            self.accepted += 1
        else:
            self.rejected += 1
            self.reasons.append(reason)
        return accepted

    def replay(self, nonce: int) -> bool:
        """Offer a nonce that was already used: the register has to reject it."""
        if isinstance(nonce, bool) or not isinstance(nonce, int) or nonce < 1:
            raise ValueError("nonce must be an integer >= 1")
        self.replay_probes += 1
        return self.register(nonce)

    @property
    def rejection_rate(self) -> float:
        """Share of the offered nonces the strike register rejected."""
        if self.attempts == 0:
            return 0.0
        return round(self.rejected / self.attempts, 4)

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict view of the register (JSON friendly)."""
        return {
            "attempts": int(self.attempts),
            "accepted": int(self.accepted),
            "rejected": int(self.rejected),
            "replay_probes": int(self.replay_probes),
            "rejection_rate": self.rejection_rate,
            "rejection_reasons": self.reasons[:16],
        }


# ---------------------------------------------------------------------------
# Transport legs
# ---------------------------------------------------------------------------
def _envelope_frames(count: int, payload_bytes: int) -> List[bytes]:
    """Build `count` timestamped frames; the timestamp is taken when the batch is built."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a non-negative integer")
    if isinstance(payload_bytes, bool) or not isinstance(payload_bytes, int) or payload_bytes < 1:
        raise ValueError("payload_bytes must be an integer >= 1")
    now = time.time()
    return [
        FrameEnvelope(index, now, b"\x5a" * max(1, payload_bytes)).encode()
        for index in range(count)
    ]


def _decode_latency(payload: bytes) -> Optional[float]:
    """The delivery latency of a timestamped payload, or `None` when it is not one."""
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("payload must be bytes")
    try:
        envelope = FrameEnvelope.decode(bytes(payload))
    except (LinkProtocolError, TypeError):
        return None
    return (time.time() - envelope.sent_at) * 1000.0


async def _run_quic_leg(
    scenario: FaultScenario,
    strike: StrikeRegister,
    server: Optional[QuicHiddenStateServer],
    proxy: DeterministicFaultProxy,
    timeout: float = 3.0,
    ca_pem: Optional[str] = None,
    server_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Measure a real `aioquic` 0-RTT resumption behind the fault proxy.

    With a local `server` the per-frame delivery latency is measured at the receiver; with a
    remote receiver (multi-host mode) only the sender side can be observed, so the batch
    acknowledgement latency is reported instead.
    """
    if not isinstance(scenario, FaultScenario):
        raise TypeError("scenario must be a FaultScenario")
    if not isinstance(strike, StrikeRegister):
        raise TypeError("strike must be a StrikeRegister")
    if proxy is None or not hasattr(proxy, "client_address") or not callable(getattr(proxy, "stats", None)):
        raise TypeError("proxy must be a DeterministicFaultProxy")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(float(timeout)) or float(timeout) <= 0.0:
        raise ValueError("timeout must be a finite positive number")
    identity = server.identity if server is not None else None
    if ca_pem is None:
        ca_pem = identity.ca_pem
    if server_name is None:
        server_name = SERVER_NAME
    latencies: List[float] = []
    started = time.perf_counter()

    # The first handshake hands out the single-use TLS 1.3 session ticket.
    ticket = None
    warmup_errors: List[str] = []
    for _attempt in range(5):
        try:
            warmup = await quic_send_frames(
                proxy.client_address[0], proxy.client_address[1], [b"warmup"],
                ca_pem, timeout=timeout, server_name=server_name,
            )
        except (TimeoutError, ConnectionError, OSError) as exc:
            warmup_errors.append(type(exc).__name__)
            continue
        if warmup.new_ticket is None:
            # The NewSessionTicket datagram was lost: ask for a fresh one.
            warmup_errors.append("session_ticket_lost")
            continue
        ticket = warmup.new_ticket
        break
    if ticket is None:
        return {
            "ok": False,
            "error": "session_ticket_unavailable",
            "warmup_errors": warmup_errors,
            "attempts": 0,
            "frames_sent": 0,
            "frames_delivered": 0,
            "session_resumed": False,
            "early_data_accepted": False,
            "server_early_frames": 0,
            "handshake_latency_ms": None,
            "resumption_latency_ms": None,
            "frame_delivery_latency_ms": latency_summary([]),
            "throughput_mbps": 0.0,
            "elapsed_s": round(time.perf_counter() - started, 6),
            "proxy_stats": proxy.stats(),
            "_legs": [],
        }

    frames_sent = 0
    attempts = 0
    probe_error: Optional[str] = None
    resumed = False
    early_accepted = False
    early_frames = 0
    delivered = 0
    handshake_latency: Optional[float] = None
    resumption_latency: Optional[float] = None
    for _attempt in range(3):
        attempts += 1
        nonce = strike.open_session()
        strike.register(nonce)
        batch = _envelope_frames(scenario.frames, scenario.frame_bytes)
        frames_sent += len(batch)
        if server is not None:
            probe = await quic_0rtt_resumption_probe(
                server,
                frame_factory=lambda batch=batch: batch,
                client_target=proxy.client_address,
                session_ticket=ticket,
                timeout=timeout,
            )
            latencies.extend(probe.frame_latencies_ms)
            delivered = probe.frames_delivered
            ok = probe.ok
        else:
            # Multi-host mode: the receiver is remote, so only the sender can be observed.
            probe_error = None
            ok = False
            try:
                report = await quic_send_frames(
                    proxy.client_address[0], proxy.client_address[1], batch, ca_pem,
                    session_ticket=ticket, timeout=timeout, server_name=server_name,
                )
                delivered = report.delivered
                ok = True
                resumed = report.session_resumed
                early_accepted = report.early_data_accepted
                handshake_latency = (
                    report.handshake_latency_s * 1000.0
                    if report.handshake_latency_s is not None
                    else None
                )
                ticket = report.new_ticket or ticket
            except (TimeoutError, ConnectionError, OSError) as exc:
                probe_error = type(exc).__name__
        # A client that retries a 0-RTT batch replays its nonce: the strike register has to
        # reject that replay, whether the attempt succeeded or not.
        strike.replay(nonce)
        if ok:
            if server is not None:
                resumed = probe.session_resumed
                early_accepted = probe.early_data_accepted
                early_frames = max(early_frames, probe.server_early_frames)
                handshake_latency = probe.handshake_latency_ms
                resumption_latency = probe.resumption_latency_ms
                ticket = probe.ticket or ticket
            break
        # The ticket was consumed by the failed attempt: obtain a fresh one.
        try:
            warmup = await quic_send_frames(
                proxy.client_address[0], proxy.client_address[1], [b"warmup"],
                ca_pem, timeout=timeout, server_name=server_name,
            )
            ticket = warmup.new_ticket
        except (TimeoutError, ConnectionError, OSError) as exc:
            probe_error = probe_error or type(exc).__name__
            break

    elapsed_s = time.perf_counter() - started
    return {
        "ok": bool(latencies) and probe_error is None,
        "error": probe_error,
        "attempts": attempts,
        "frames_sent": frames_sent,
        "frames_delivered": delivered,
        "session_resumed": resumed,
        "early_data_accepted": early_accepted,
        "server_early_frames": early_frames,
        "handshake_latency_ms": (
            round(handshake_latency, 3) if handshake_latency is not None else None
        ),
        "resumption_latency_ms": (
            round(resumption_latency, 3) if resumption_latency is not None else None
        ),
        "frame_delivery_latency_ms": latency_summary(latencies),
        "throughput_mbps": _throughput_mbps(
            len(latencies) * (scenario.frame_bytes + ENVELOPE_OVERHEAD), elapsed_s
        ),
        "elapsed_s": round(elapsed_s, 6),
        "proxy_stats": proxy.stats(),
        "_legs": [
            {
                "leg": "quic_0rtt",
                "latencies": list(latencies),
                "bytes": len(latencies) * (scenario.frame_bytes + ENVELOPE_OVERHEAD),
                "elapsed_s": elapsed_s,
                "frames_sent": frames_sent,
                "frames_delivered": len(latencies),
            }
        ],
    }


def _relay_frames(
    sender: TurnClient,
    receiver: TurnClient,
    peer: Address,
    scenario: FaultScenario,
    timeout: float,
    channel: bool = False,
) -> List[float]:
    """Relay `scenario.frames` payloads and return the per-frame delivery latency."""
    if (
        sender is None
        or receiver is None
        or not (callable(getattr(sender, "send_to", None)) or callable(getattr(sender, "send_channel", None)))
        or not callable(getattr(receiver, "recv_from", None))
    ):
        raise TypeError("sender and receiver must be TurnClient instances")
    if not isinstance(scenario, FaultScenario):
        raise TypeError("scenario must be a FaultScenario")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(float(timeout)) or float(timeout) <= 0.0:
        raise ValueError("timeout must be a finite positive number")
    if not isinstance(channel, bool):
        raise TypeError("channel must be a boolean")
    payloads = _envelope_frames(scenario.frames, scenario.frame_bytes)
    for payload in payloads:
        if channel:
            sender.send_channel(peer, payload)
        else:
            sender.send_to(peer, payload)
    latencies: List[float] = []
    deadline = time.monotonic() + max(1.0, min(timeout, 2.0))
    while len(latencies) < len(payloads) and time.monotonic() < deadline:
        try:
            _peer, data = receiver.recv_from(timeout=max(0.1, deadline - time.monotonic()))
        except StunError:
            break
        latency = _decode_latency(data)
        if latency is None:
            continue
        latencies.append(latency)
    return latencies


def _run_turn_leg(
    scenario: FaultScenario,
    timeout: float = 3.0,
    endpoints: Optional[ExternalEndpoints] = None,
) -> Dict[str, Any]:
    """Measure the RFC 5766 TURN relay (data indications and channels) behind the proxy.

    Without `endpoints` the relay runs locally; with a remote TURN server the same suite
    measures a real multi-host relay path.
    """
    if not isinstance(scenario, FaultScenario):
        raise TypeError("scenario must be a FaultScenario")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(float(timeout)) or float(timeout) <= 0.0:
        raise ValueError("timeout must be a finite positive number")
    if endpoints is not None and not (hasattr(endpoints, "has_turn") or hasattr(endpoints, "has_quic")):
        raise TypeError("endpoints must be an ExternalEndpoints or None")
    endpoints = endpoints if endpoints is not None else ExternalEndpoints()
    started = time.perf_counter()
    with ExitStack() as stack:
        turn: Optional[TurnRelayServer] = None
        if endpoints.has_turn:
            turn_address = (endpoints.turn_host, endpoints.turn_port)
            credentials = [
                (endpoints.turn_username, endpoints.turn_password or ""),
                (endpoints.turn_username, endpoints.turn_password or ""),
            ]
        else:
            turn = stack.enter_context(
                TurnRelayServer({"alice": "s3cret", "bob": "hunter2"})
            )
            turn_address = turn.address
            credentials = [("alice", "s3cret"), ("bob", "hunter2")]
        alice_proxy = DeterministicFaultProxy(turn_address, scenario.profile).start()
        bob_proxy = DeterministicFaultProxy(turn_address, scenario.profile).start()
        alice = TurnClient(
            alice_proxy.client_address, credentials[0][0], credentials[0][1], timeout=timeout
        )
        bob = TurnClient(
            bob_proxy.client_address, credentials[1][0], credentials[1][1], timeout=timeout
        )
        try:
            relay_a = alice.allocate()
            relay_b = bob.allocate()
            alice.create_permission(relay_b[0])
            bob.create_permission(relay_a[0])

            # Data indication path (RFC 5766 §10.1).
            data_latencies = _relay_frames(alice, bob, relay_b, scenario, timeout)
            data_elapsed = time.perf_counter() - started
            # Channel path (RFC 5766 §11): ChannelBind plus ChannelData messages.
            channel_number = alice.bind_channel(relay_b)
            bob.bind_channel(relay_a)
            channel_started = time.perf_counter()
            channel_latencies = _relay_frames(
                alice, bob, relay_b, scenario, timeout, channel=True
            )
            channel_elapsed = time.perf_counter() - channel_started

            # NAT rebinding: the relay loses the allocation of the old mapping.
            rebinds = 0
            reallocations = 0
            post_rebind_latencies: List[float] = []
            lost_after_rebind = 0
            for _ in range(max(0, scenario.nat_rebinds)):
                rebinds += 1
                alice_proxy.rebind()
                # The allocation of the old mapping is gone: the relay drops the frames.
                dropped = _relay_frames(alice, bob, relay_b, scenario, timeout=0.5)
                lost_after_rebind += scenario.frames - len(dropped)
                try:
                    relay_a = alice.allocate()
                    reallocations += 1
                    alice.create_permission(relay_b[0])
                    post_rebind_latencies.extend(
                        _relay_frames(alice, bob, relay_b, scenario, timeout)
                    )
                except StunError:
                    break

            elapsed_s = time.perf_counter() - started
            return {
                "ok": bool(data_latencies),
                "error": None,
                "allocate_ok": True,
                "relayed_addresses": [list(relay_a), list(relay_b)],
                "channel_number": channel_number,
                "channels_bound": len(turn.channels) if turn is not None else 0,
                "frames_sent": 3 * scenario.frames,
                "frames_delivered": len(data_latencies) + len(channel_latencies),
                "data_relay": {
                    "frames_delivered": len(data_latencies),
                    "frame_delivery_latency_ms": latency_summary(data_latencies),
                    "throughput_mbps": _throughput_mbps(
                        len(data_latencies) * (scenario.frame_bytes + ENVELOPE_OVERHEAD),
                        data_elapsed,
                    ),
                },
                "channel_relay": {
                    "frames_delivered": len(channel_latencies),
                    "frame_delivery_latency_ms": latency_summary(channel_latencies),
                    "throughput_mbps": _throughput_mbps(
                        len(channel_latencies) * (scenario.frame_bytes + ENVELOPE_OVERHEAD),
                        channel_elapsed,
                    ),
                },
                "nat_rebind_events": rebinds,
                "reallocations_after_rebind": reallocations,
                "frames_lost_after_rebind": lost_after_rebind,
                "post_rebind_frames_delivered": len(post_rebind_latencies),
                "post_rebind_latency_ms": latency_summary(post_rebind_latencies),
                "elapsed_s": round(elapsed_s, 6),
                "proxy_stats": {"alice": alice_proxy.stats(), "bob": bob_proxy.stats()},
                "_legs": [
                    {
                        "leg": "turn_data",
                        "latencies": list(data_latencies),
                        "bytes": len(data_latencies) * (scenario.frame_bytes + ENVELOPE_OVERHEAD),
                        "elapsed_s": data_elapsed,
                        "frames_sent": scenario.frames,
                        "frames_delivered": len(data_latencies),
                    },
                    {
                        "leg": "turn_channel",
                        "latencies": list(channel_latencies),
                        "bytes": len(channel_latencies) * (scenario.frame_bytes + ENVELOPE_OVERHEAD),
                        "elapsed_s": channel_elapsed,
                        "frames_sent": scenario.frames,
                        "frames_delivered": len(channel_latencies),
                    },
                    {
                        "leg": "turn_after_rebind",
                        "latencies": list(post_rebind_latencies),
                        "bytes": len(post_rebind_latencies)
                        * (scenario.frame_bytes + ENVELOPE_OVERHEAD),
                        "elapsed_s": max(1e-9, elapsed_s),
                        "frames_sent": scenario.frames,
                        "frames_delivered": len(post_rebind_latencies),
                    },
                ],
            }
        except StunError as exc:
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "allocate_ok": False,
                "frames_sent": 0,
                "frames_delivered": 0,
                "data_relay": {
                    "frames_delivered": 0,
                    "frame_delivery_latency_ms": latency_summary([]),
                    "throughput_mbps": 0.0,
                },
                "channel_relay": {
                    "frames_delivered": 0,
                    "frame_delivery_latency_ms": latency_summary([]),
                    "throughput_mbps": 0.0,
                },
                "nat_rebind_events": 0,
                "reallocations_after_rebind": 0,
                "frames_lost_after_rebind": 0,
                "post_rebind_frames_delivered": 0,
                "post_rebind_latency_ms": latency_summary([]),
                "elapsed_s": round(time.perf_counter() - started, 6),
                "proxy_stats": {"alice": alice_proxy.stats(), "bob": bob_proxy.stats()},
                "_legs": [],
            }
        finally:
            alice.close()
            bob.close()
            alice_proxy.close()
            bob_proxy.close()


async def _run_websocket_leg(
    scenario: FaultScenario,
    ca_pem: str,
    lossy_quic_target: Address,
    timeout: float = 3.0,
    fallback_timeout: float = 0.5,
) -> Dict[str, Any]:
    """Measure the libp2p WebSocket fallback of `quic_link`."""
    if not isinstance(scenario, FaultScenario):
        raise TypeError("scenario must be a FaultScenario")
    if not isinstance(ca_pem, str) or not ca_pem.strip():
        raise ValueError("ca_pem must be a non-empty string")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or float(timeout) <= 0.0
        or isinstance(fallback_timeout, bool)
        or not isinstance(fallback_timeout, (int, float))
        or not math.isfinite(float(fallback_timeout))
        or float(fallback_timeout) <= 0.0
    ):
        raise ValueError("timeout and fallback_timeout must be finite positive numbers")
    delivered: List[float] = []

    def on_frame(frame: bytes) -> None:
        latency = _decode_latency(frame)
        if latency is not None:
            delivered.append(latency)

    server = await WebSocketHiddenStateServer("127.0.0.1", on_frame=on_frame).start()
    started = time.perf_counter()
    try:
        # The WebSocket path on its own: the libp2p fallback's delivery latency.
        frames = _envelope_frames(scenario.frames, scenario.frame_bytes)
        sent_at = time.perf_counter()
        acknowledged = await websocket_send_frames(server.url, frames, timeout=timeout)
        elapsed = time.perf_counter() - sent_at
        fallback_latencies = list(delivered)
        delivered.clear()

        # The QUIC path is unreachable, so the link has to fall back to the WebSocket.
        dead_port = _unused_udp_port()
        endpoint = LinkEndpoint(
            host="127.0.0.1", port=dead_port, ca_pem=ca_pem, websocket_url=server.url
        )
        link = RemoteHiddenStateLink(endpoint, timeout=fallback_timeout)
        link_frames = _envelope_frames(scenario.frames, scenario.frame_bytes)
        fallback_run = await link.send_frames(link_frames)
        delivered.clear()

        # The same link, but with the QUIC path pointing at the lossy proxy.
        lossy_endpoint = LinkEndpoint(
            host="127.0.0.1", port=lossy_quic_target[1], ca_pem=ca_pem, websocket_url=server.url
        )
        lossy_link = RemoteHiddenStateLink(lossy_endpoint, timeout=fallback_timeout)
        lossy_frames = _envelope_frames(scenario.frames, scenario.frame_bytes)
        lossy_error = None
        lossy_protocol = None
        lossy_delivered = 0
        lossy_reason = None
        try:
            lossy_run = await lossy_link.send_frames(lossy_frames)
            lossy_protocol = lossy_run.protocol
            lossy_delivered = lossy_run.frames_delivered
            lossy_reason = lossy_run.fallback_reason
        except (TimeoutError, ConnectionError, OSError) as exc:
            lossy_error = type(exc).__name__
        lossy_latencies = list(delivered)
        delivered.clear()

        return {
            "ok": acknowledged == len(frames) and fallback_run.protocol == "libp2p_websocket",
            "protocol": fallback_run.protocol,
            "fallback_reason": fallback_run.fallback_reason,
            "frames_sent": len(frames),
            "frames_delivered": acknowledged,
            "frame_delivery_latency_ms": latency_summary(fallback_latencies),
            "throughput_mbps": _throughput_mbps(
                acknowledged * (scenario.frame_bytes + ENVELOPE_OVERHEAD), elapsed
            ),
            "fallback": {
                "protocol": fallback_run.protocol,
                "fallback_reason": fallback_run.fallback_reason,
                "frames_sent": fallback_run.frames_sent,
                "frames_delivered": fallback_run.frames_delivered,
                "note": "the WebSocket fallback carries the batch when UDP is unusable",
            },
            "lossy_quic_path": {
                "protocol": lossy_protocol,
                "fallback_reason": lossy_reason,
                "error": lossy_error,
                "frames_delivered": lossy_delivered,
                "frame_delivery_latency_ms": latency_summary(lossy_latencies),
                "note": "an unacknowledged QUIC batch is never replayed automatically",
            },
            "elapsed_s": round(time.perf_counter() - started, 6),
            "proxy_stats": {},
            "_legs": [
                {
                    "leg": "libp2p_websocket",
                    "latencies": list(fallback_latencies),
                    "bytes": acknowledged * (scenario.frame_bytes + ENVELOPE_OVERHEAD),
                    "elapsed_s": elapsed,
                    "frames_sent": len(frames),
                    "frames_delivered": acknowledged,
                }
            ],
        }
    finally:
        await server.close()


def _bbr_pacing(
    samples: Sequence[Dict[str, Any]],
    frames_sent: int,
    frames_delivered: int,
) -> Dict[str, Any]:
    """Feed the measured path into the BBRv1-inspired controller and report its adaptation.

    `samples` are the per-leg measurements (`leg`, `latencies`, `bytes`, `elapsed_s`,
    `frames_sent`, `frames_delivered`); the controller is fed leg by leg, so the report
    shows how the pacing window adapts while the benchmark walks the transports.
    """
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise TypeError("samples must be a sequence of dicts")
    if (
        isinstance(frames_sent, bool)
        or not isinstance(frames_sent, int)
        or frames_sent < 0
        or isinstance(frames_delivered, bool)
        or not isinstance(frames_delivered, int)
        or frames_delivered < 0
    ):
        raise ValueError("frames_sent and frames_delivered must be non-negative integers")
    from hivemind.swarm.quic_transport import AdaptiveCongestionController

    controller = AdaptiveCongestionController()
    windows: List[Dict[str, Any]] = []
    for sample in samples:
        for rtt in sample["latencies"]:
            controller.record_rtt_sample(rtt)
        if sample["elapsed_s"] > 0:
            controller.record_delivery(max(1, sample["bytes"]), sample["elapsed_s"] * 1000.0)
        controller.record_loss(
            max(0, sample["frames_sent"] - sample["frames_delivered"]),
            max(1, sample["frames_sent"]),
        )
        window = controller.compute_target_window().to_dict()
        window["leg"] = sample["leg"]
        windows.append(window)

    telemetry = dict(windows[-1]) if windows else AdaptiveCongestionController(
    ).compute_target_window().to_dict()
    telemetry.update(
        {
            "model": "bbr_v1_delivery_rate_max_filter",
            "frames_sent": int(frames_sent),
            "frames_delivered": int(frames_delivered),
            "frames_lost": max(0, int(frames_sent) - int(frames_delivered)),
            "rtprop_ms": round(controller.rtprop_ms, 3),
            "jitter_ms": round(controller.jitter_ms, 3),
            "pacing_samples": windows,
        }
    )
    return telemetry


# ---------------------------------------------------------------------------
# Scenario and benchmark runners
# ---------------------------------------------------------------------------
async def run_fault_scenario(
    scenario: FaultScenario,
    timeout: float = 3.0,
    endpoints: Optional[ExternalEndpoints] = None,
) -> Dict[str, Any]:
    """Run every transport leg of one fault profile and return its telemetry."""
    if not isinstance(scenario, FaultScenario):
        raise TypeError("scenario must be a FaultScenario")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(float(timeout)) or float(timeout) <= 0.0:
        raise ValueError("timeout must be a finite positive number")
    if endpoints is not None and not (hasattr(endpoints, "has_turn") or hasattr(endpoints, "has_quic")):
        raise TypeError("endpoints must be an ExternalEndpoints or None")
    endpoints = endpoints if endpoints is not None else ExternalEndpoints()
    strike = StrikeRegister()
    identity = LinkIdentity.generate()
    server: Optional[QuicHiddenStateServer] = None
    if endpoints.has_quic:
        upstream = (endpoints.quic_host, int(endpoints.quic_port))
        ca_pem = endpoints.quic_ca_pem
        server_name = endpoints.quic_server_name or SERVER_NAME
    else:
        server = await QuicHiddenStateServer(identity, "127.0.0.1", 0).start()
        upstream = (server.host, server.port)
        ca_pem = identity.ca_pem
        server_name = SERVER_NAME
    proxy = DeterministicFaultProxy(upstream, scenario.profile).start()
    try:
        quic = await _run_quic_leg(
            scenario, strike, server, proxy, timeout=timeout, ca_pem=ca_pem,
            server_name=server_name,
        )
        turn = await asyncio.to_thread(_run_turn_leg, scenario, timeout, endpoints)
        websocket = await _run_websocket_leg(
            scenario, identity.ca_pem, proxy.client_address, timeout=timeout
        )
    finally:
        proxy.close()
        if server is not None:
            server.close()

    legs: List[Dict[str, Any]] = []
    for part in (quic, turn, websocket):
        legs.extend(part.pop("_legs", []))
    latencies: List[float] = []
    for leg in legs:
        latencies.extend(leg["latencies"])

    frames_sent = (
        int(quic.get("frames_sent", 0))
        + int(turn.get("frames_sent", 0))
        + int(websocket.get("frames_sent", 0))
    )
    frames_delivered = (
        int(quic.get("frames_delivered", 0))
        + int(turn.get("frames_delivered", 0))
        + int(websocket.get("frames_delivered", 0))
    )
    bytes_delivered = frames_delivered * (scenario.frame_bytes + ENVELOPE_OVERHEAD)
    elapsed_s = max(
        float(quic.get("elapsed_s", 0.0) or 0.0)
        + float(turn.get("elapsed_s", 0.0) or 0.0)
        + float(websocket.get("elapsed_s", 0.0) or 0.0),
        1e-9,
    )

    pacing = _bbr_pacing(legs, frames_sent, frames_delivered)

    return {
        **scenario.to_dict(),
        "frames_sent": frames_sent,
        "frames_delivered": frames_delivered,
        "bytes_delivered": bytes_delivered,
        "elapsed_s": round(elapsed_s, 6),
        "frame_delivery_latency_ms": latency_summary(latencies),
        "quic_0rtt": quic,
        "turn_relay": turn,
        "websocket_fallback": websocket,
        "zero_rtt_strike_register": strike.to_dict(),
        "bbr_v1_pacing": pacing,
    }


async def run_wan_quic_turn_fault_benchmark(
    scenarios: Optional[Sequence[FaultScenario]] = None,
    timeout: float = 3.0,
    endpoints: Optional[ExternalEndpoints] = None,
) -> Dict[str, Any]:
    """Run the whole benchmark and return the structured telemetry report."""
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(float(timeout)) or float(timeout) <= 0.0:
        raise ValueError("timeout must be a finite positive number")
    if scenarios is None:
        scenarios = default_scenarios()
    elif isinstance(scenarios, (str, bytes)) or not isinstance(scenarios, Sequence) or not all(isinstance(s, FaultScenario) for s in scenarios):
        raise TypeError("scenarios must be a sequence of FaultScenario instances")
    if endpoints is not None and not (hasattr(endpoints, "has_turn") or hasattr(endpoints, "has_quic")):
        raise TypeError("endpoints must be an ExternalEndpoints or None")
    endpoints = endpoints if endpoints is not None else ExternalEndpoints.from_environment()
    results: List[Dict[str, Any]] = []
    for scenario in scenarios:
        results.append(
            await run_fault_scenario(scenario, timeout=timeout, endpoints=endpoints)
        )

    baseline = results[0] if results else {}
    worst = max(
        results,
        key=lambda item: item.get("frame_delivery_latency_ms", {}).get("p99_ms", 0.0),
        default={},
    )
    register_totals = {
        "attempts": sum(item["zero_rtt_strike_register"]["attempts"] for item in results),
        "accepted": sum(item["zero_rtt_strike_register"]["accepted"] for item in results),
        "rejected": sum(item["zero_rtt_strike_register"]["rejected"] for item in results),
    }
    register_totals["rejection_rate"] = round(
        register_totals["rejected"] / max(1, register_totals["attempts"]), 4
    )
    summary = {
        "scenarios": len(results),
        "packet_loss_rates": [item["packet_loss_rate"] for item in results],
        "baseline_scenario": baseline.get("scenario_id"),
        "worst_case_scenario": worst.get("scenario_id"),
        "worst_case_p99_frame_latency_ms": worst.get("frame_delivery_latency_ms", {}).get(
            "p99_ms", 0.0
        ),
        "frame_delivery_latency_ms_by_scenario": {
            item["scenario_id"]: {
                "p50_ms": item["frame_delivery_latency_ms"]["p50_ms"],
                "p95_ms": item["frame_delivery_latency_ms"]["p95_ms"],
                "p99_ms": item["frame_delivery_latency_ms"]["p99_ms"],
            }
            for item in results
        },
        "zero_rtt_strike_register": register_totals,
        "bbr_v1_pacing_adaptation": _pacing_adaptation(results),
    }
    return {
        "schema": REPORT_SCHEMA,
        "bounty": "HM-S5-05",
        "title": "Multi-host WAN QUIC 0-RTT & RFC 5766 TURN fault-injection benchmark",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "environment": _environment(),
        "endpoints": endpoints.to_dict(),
        "fault_profiles": [scenario.to_dict() for scenario in scenarios],
        "scenarios": results,
        "summary": summary,
    }


def _pacing_adaptation(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare the BBRv1 window of the clean path with the most impaired one."""
    if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
        raise TypeError("results must be a sequence of dicts")
    if not results:
        return {}
    baseline = results[0].get("bbr_v1_pacing", {})
    impaired = max(
        results, key=lambda item: item.get("packet_loss_rate", 0.0), default=results[-1]
    )
    worst = impaired.get("bbr_v1_pacing", {})
    return {
        "baseline_scenario": results[0].get("scenario_id"),
        "baseline_cwnd_bytes": baseline.get("cwnd_bytes"),
        "baseline_gain": baseline.get("gain"),
        "baseline_btl_bw_bps": baseline.get("btl_bw_bps"),
        "impaired_scenario": impaired.get("scenario_id"),
        "impaired_cwnd_bytes": worst.get("cwnd_bytes"),
        "impaired_gain": worst.get("gain"),
        "impaired_btl_bw_bps": worst.get("btl_bw_bps"),
        "impaired_loss_ratio": worst.get("loss_ratio"),
    }


def _environment() -> Dict[str, Any]:
    """Describe the runtime the benchmark ran on (JSON friendly)."""
    versions: Dict[str, str] = {}
    for name in ("aioquic", "websockets", "cryptography"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:  # pragma: no cover - the dependency is optional at runtime
            versions[name] = "unavailable"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": versions,
    }


def report_path(root: Optional[Path] = None) -> Path:
    """The canonical location of the telemetry report."""
    base = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    return base / "artifacts" / REPORT_NAME


def write_fault_report(report: Dict[str, Any], path: Optional[Path] = None) -> Path:
    """Write the telemetry report as JSON and return the path it was written to."""
    if not isinstance(report, dict):
        raise TypeError("report must be a dict")
    target = Path(path) if path is not None else report_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return target


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the benchmark and write `artifacts/wan_quic_turn_fault_report.json`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=None, help="path of the JSON report")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--frame-bytes", type=int, default=DEFAULT_FRAME_BYTES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--loss-rates",
        default=",".join(str(rate) for rate in DEFAULT_LOSS_RATES),
        help="comma separated packet loss rates",
    )
    args = parser.parse_args(argv)
    loss_rates = [float(item) for item in args.loss_rates.split(",") if item.strip()]
    scenarios = default_scenarios(
        loss_rates=loss_rates, seed=args.seed, frames=args.frames, frame_bytes=args.frame_bytes
    )
    report = asyncio.run(run_wan_quic_turn_fault_benchmark(scenarios))
    target = write_fault_report(report, Path(args.output) if args.output else None)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
