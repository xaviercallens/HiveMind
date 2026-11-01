# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""QUIC hidden-state link with TLS 1.3 0-RTT resumption and a libp2p WebSocket fallback (`HM-M1-01`).

- Frames travel on a QUIC (RFC 9000) stream, length-prefixed, via `aioquic`.
- The server issues TLS 1.3 session tickets. A client that presents one resumes the session and
  sends its frames as 0-RTT early data (RFC 9001 §4.6), before the handshake finishes. Tickets
  are single-use on the server, the RFC 8446 §8.1 defence against replayed 0-RTT data.
- When the initial QUIC handshake cannot complete, a configured WebSocket (RFC 6455) can
  carry frames after libp2p multistream-select negotiation. Uncertain deliveries are never
  automatically replayed. Plaintext WebSocket is restricted to literal loopback addresses.

Clients authenticate the receiver with a configured CA: `LinkIdentity.generate()` creates an
ephemeral CA and server certificate. `RemoteHiddenStateLink` sends to an existing receiver,
including across processes, without starting a local server or implementing NAT traversal.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import functools
import inspect
import ipaddress
import math
import socket
import ssl
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from aioquic.asyncio import serve
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.connection import QuicConnection
from aioquic.quic.events import ConnectionTerminated, HandshakeCompleted, QuicEvent, StreamDataReceived
from aioquic.tls import SessionTicket
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from websockets.asyncio.client import connect as ws_connect
from websockets.asyncio.server import serve as ws_serve
from websockets.exceptions import ConnectionClosed

ALPN = "hivemind-hidden-state/1"
SERVER_NAME = "hivemind.local"
MULTISTREAM = "/multistream/1.0.0"
LIBP2P_PROTOCOL = "/hivemind/hidden-state/1.0.0"
_LENGTH = struct.Struct("!I")
ENVELOPE_HEADER = struct.Struct("!dI")
"""Frame envelope header: the send timestamp (seconds) and the frame sequence number."""
MAX_FRAME_BYTES = 16 * 1024 * 1024
MAX_BATCH_BYTES = 64 * 1024 * 1024
MAX_BATCH_FRAMES = 4096
MAX_OPEN_STREAMS = 16
MAX_PROTOCOL_BYTES = 1024
MAX_SESSION_TICKETS = 1024


@contextlib.asynccontextmanager
async def _async_timeout(delay: Optional[float]):
    native = getattr(asyncio, "timeout", None)
    if native is not None:
        async with native(delay):
            yield
        return
    if delay is None:
        yield
        return
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    timed_out = False

    def _on_timeout() -> None:
        nonlocal timed_out
        timed_out = True
        if task is not None:
            task.cancel()

    handle = loop.call_later(delay, _on_timeout)
    try:
        yield
    except asyncio.CancelledError as exc:
        if timed_out:
            raise TimeoutError from exc
        raise
    finally:
        handle.cancel()
STREAM_TIMEOUT_S = 10.0

FrameHandler = Callable[[bytes], None]


class LinkProtocolError(ConnectionError):
    """The peer sent invalid framing or an invalid delivery acknowledgement."""


class DeliveryUncertainError(ConnectionError):
    """Some frames may have arrived; automatically replaying the batch is unsafe."""


def _validate_timeout(timeout: float) -> None:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("link timeout must be finite and positive")


def _validate_frames(frames: Sequence[bytes]) -> List[bytes]:
    if len(frames) > MAX_BATCH_FRAMES:
        raise ValueError("too many hidden-state frames in one batch")
    batch = list(frames)
    total = 0
    for frame in batch:
        if not isinstance(frame, bytes):
            raise TypeError("hidden-state frames must be bytes")
        if not 0 < len(frame) <= MAX_FRAME_BYTES:
            raise ValueError("hidden-state frame is empty or exceeds the size limit")
        total += len(frame) + _LENGTH.size
        if total > MAX_BATCH_BYTES:
            raise ValueError("hidden-state batch exceeds the size limit")
    return batch


def _is_loopback_ip(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_websocket_url(url: str) -> str:
    parsed = urlsplit(url)
    if (parsed.scheme not in ("ws", "wss") or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.fragment):
        raise ValueError("expected a ws/wss endpoint without userinfo or a fragment")
    if parsed.port is not None and not 1 <= parsed.port <= 65535:
        raise ValueError("invalid WebSocket port")
    if parsed.scheme == "ws" and not _is_loopback_ip(parsed.hostname):
        raise ValueError("plaintext WebSocket is restricted to literal loopback IPs; use wss")
    return parsed.scheme


@dataclass
class LinkIdentity:
    """An ephemeral CA plus a server certificate (SAN: `SERVER_NAME`, 127.0.0.1) it signed."""

    ca_certificate: x509.Certificate
    certificate: x509.Certificate
    private_key: ec.EllipticCurvePrivateKey

    @property
    def ca_pem(self) -> str:
        return self.ca_certificate.public_bytes(serialization.Encoding.PEM).decode("ascii")

    @classmethod
    def generate(cls, server_name: str = SERVER_NAME, days: int = 7) -> "LinkIdentity":
        now = datetime.datetime.now(datetime.timezone.utc)
        ca_key = ec.generate_private_key(ec.SECP256R1())
        ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "HiveMind link CA")])
        ca_ski = x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key())
        ca_certificate = (
            x509.CertificateBuilder()
            .subject_name(ca_name)
            .issuer_name(ca_name)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=days))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=False, content_commitment=False, key_encipherment=False,
                    data_encipherment=False, key_agreement=False, key_cert_sign=True,
                    crl_sign=True, encipher_only=False, decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(ca_ski, critical=False)
            .sign(ca_key, hashes.SHA256())
        )
        key = ec.generate_private_key(ec.SECP256R1())
        certificate = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, server_name)]))
            .issuer_name(ca_name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=days))
            .add_extension(
                x509.SubjectAlternativeName(
                    [x509.DNSName(server_name), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
                ),
                critical=False,
            )
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_subject_key_identifier(ca_ski), critical=False
            )
            .sign(ca_key, hashes.SHA256())
        )
        return cls(ca_certificate, certificate, key)


def _drain_frames(buffer: bytearray) -> List[bytes]:
    frames = []
    offset = 0
    while len(buffer) - offset >= _LENGTH.size:
        (length,) = _LENGTH.unpack_from(buffer, offset)
        if not 0 < length <= MAX_FRAME_BYTES:
            raise LinkProtocolError("invalid hidden-state frame length")
        if len(buffer) - offset < _LENGTH.size + length:
            break
        if len(frames) >= MAX_BATCH_FRAMES:
            raise LinkProtocolError("too many hidden-state frames")
        frames.append(bytes(buffer[offset + _LENGTH.size : offset + _LENGTH.size + length]))
        offset += _LENGTH.size + length
    del buffer[:offset]
    return frames


@dataclass(frozen=True)
class FrameEnvelope:
    """A hidden-state frame that carries its own send timestamp and sequence number.

    The envelope is the frame payload, so the length-prefixed QUIC framing is unchanged.
    It lets a receiver measure the delivery latency of every single frame, which the WAN
    fault benchmark reports as p50/p95/p99 (`HM-S5-05`).
    """

    sequence: int
    sent_at: float
    payload: bytes

    def encode(self) -> bytes:
        """Serialise the envelope."""
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("frame sequence number must be a non-negative integer")
        if not math.isfinite(self.sent_at):
            raise ValueError("frame send timestamp must be finite")
        return ENVELOPE_HEADER.pack(self.sent_at, self.sequence) + self.payload

    @classmethod
    def decode(cls, frame: bytes) -> "FrameEnvelope":
        """Parse an envelope, rejecting truncated or malformed frames."""
        if not isinstance(frame, (bytes, bytearray)):
            raise TypeError("hidden-state frames must be bytes")
        if len(frame) < ENVELOPE_HEADER.size:
            raise LinkProtocolError("truncated frame envelope")
        sent_at, sequence = ENVELOPE_HEADER.unpack_from(frame, 0)
        if not math.isfinite(sent_at) or sent_at <= 0:
            raise LinkProtocolError("invalid frame envelope timestamp")
        return cls(sequence, sent_at, bytes(frame[ENVELOPE_HEADER.size :]))


class _ServerProtocol(QuicConnectionProtocol):
    def __init__(self, *args, link: "QuicHiddenStateServer", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._link = link
        self._handshake_done = False
        self._buffers: Dict[int, bytearray] = {}
        self._counts: Dict[int, int] = {}
        self._totals: Dict[int, int] = {}
        self._deadlines: Dict[int, asyncio.TimerHandle] = {}
        self._failed = False

    def close(self, *args, **kwargs) -> None:
        for timer in self._deadlines.values():
            timer.cancel()
        self._deadlines.clear()
        self._buffers.clear()
        self._counts.clear()
        self._totals.clear()
        super().close(*args, **kwargs)

    def _fail(self, reason: str) -> None:
        self._failed = True
        for timer in self._deadlines.values():
            timer.cancel()
        self._deadlines.clear()
        self._buffers.clear()
        self._counts.clear()
        self._totals.clear()
        self.close(error_code=0x100, reason_phrase=reason)

    def quic_event_received(self, event: QuicEvent) -> None:
        if self._failed:
            return
        if isinstance(event, HandshakeCompleted):
            self._handshake_done = True
            self._link.handshakes.append(
                {"session_resumed": event.session_resumed, "early_data_accepted": event.early_data_accepted}
            )
            del self._link.handshakes[:-128]
        elif isinstance(event, StreamDataReceived):
            if event.stream_id % 4 != 0:
                self._fail("hidden-state frames require a client bidirectional stream")
                return
            if event.stream_id not in self._buffers:
                if len(self._buffers) >= MAX_OPEN_STREAMS:
                    self._fail("too many open hidden-state streams")
                    return
                self._deadlines[event.stream_id] = asyncio.get_running_loop().call_later(
                    STREAM_TIMEOUT_S, self._fail, "hidden-state stream timed out"
                )
            total = self._totals.get(event.stream_id, 0) + len(event.data)
            if total > MAX_BATCH_BYTES:
                self._fail("hidden-state batch exceeds size limit")
                return
            self._totals[event.stream_id] = total
            buffer = self._buffers.setdefault(event.stream_id, bytearray())
            buffer += event.data
            try:
                frames = _drain_frames(buffer)
                count = self._counts.get(event.stream_id, 0) + len(frames)
                if count > MAX_BATCH_FRAMES or sum(map(len, self._buffers.values())) > MAX_BATCH_BYTES:
                    raise LinkProtocolError("hidden-state stream limits exceeded")
                if event.end_stream and buffer:
                    raise LinkProtocolError("truncated hidden-state stream")
                for frame in frames:
                    self._link.deliver(frame)
                    if not self._handshake_done:
                        self._link.early_frames += 1
                self._counts[event.stream_id] = count
            except Exception:
                self._fail("invalid hidden-state stream or failed frame handler")
                return
            if event.end_stream:
                count = self._counts.pop(event.stream_id, 0)
                self._buffers.pop(event.stream_id, None)
                self._totals.pop(event.stream_id, None)
                self._deadlines.pop(event.stream_id).cancel()
                self._quic.send_stream_data(event.stream_id, _LENGTH.pack(count), end_stream=True)
                self.transmit()
        elif isinstance(event, ConnectionTerminated):
            for timer in self._deadlines.values():
                timer.cancel()
            self._deadlines.clear()
            self._buffers.clear()
            self._counts.clear()
            self._totals.clear()


class QuicHiddenStateServer:
    """QUIC endpoint that receives length-prefixed hidden-state frames and acknowledges counts."""

    def __init__(
        self,
        identity: LinkIdentity,
        host: str = "127.0.0.1",
        port: int = 0,
        on_frame: Optional[FrameHandler] = None,
    ) -> None:
        self.identity = identity
        self.host = host
        self.port = port
        self.on_frame = on_frame
        self.tickets: Dict[bytes, SessionTicket] = {}
        self.handshakes: List[Dict[str, bool]] = []
        self.frames_received = 0
        self.early_frames = 0
        """Frames that arrived before this server's handshake completed, i.e. as 0-RTT data."""
        self._server = None

    def deliver(self, frame: bytes) -> None:
        if self.on_frame is not None:
            self.on_frame(frame)
        self.frames_received += 1

    def _store_ticket(self, ticket: SessionTicket) -> None:
        for token, saved in list(self.tickets.items()):
            if not saved.is_valid:
                del self.tickets[token]
        self.tickets[ticket.ticket] = ticket
        while len(self.tickets) > MAX_SESSION_TICKETS:
            del self.tickets[next(iter(self.tickets))]

    def _consume_ticket(self, token: bytes) -> Optional[SessionTicket]:
        ticket = self.tickets.pop(token, None)
        return ticket if ticket is not None and ticket.is_valid else None

    async def start(self) -> "QuicHiddenStateServer":
        if self._server is not None:
            raise RuntimeError("QUIC hidden-state server is already started")
        configuration = QuicConfiguration(is_client=False, alpn_protocols=[ALPN])
        configuration.certificate = self.identity.certificate
        configuration.private_key = self.identity.private_key
        self._server = await serve(
            self.host,
            self.port,
            configuration=configuration,
            create_protocol=functools.partial(_ServerProtocol, link=self),
            session_ticket_fetcher=self._consume_ticket,
            session_ticket_handler=self._store_ticket,
        )
        self.port = self._server._transport.get_extra_info("sockname")[1]
        return self

    def close(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None
        self.tickets.clear()


class _ClientProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.handshake: Optional[HandshakeCompleted] = None
        self.reply: asyncio.Future = asyncio.get_running_loop().create_future()
        self.connected: asyncio.Future = asyncio.get_running_loop().create_future()
        self._buffer = bytearray()
        self.stream_id: Optional[int] = None
        self.started_at: Optional[float] = None
        self.handshake_latency_s: Optional[float] = None

    async def wait_connected(self) -> None:
        """Wait for the TLS handshake with a waiter this link owns.

        `aioquic` shields its own waiter, so a connection that is terminated after the wait
        was cancelled logs "exception in shielded future". The link keeps its own waiter, so
        a cancelled wait stays silent and the termination is reported to the caller.
        """
        await self.connected

    def quic_event_received(self, event: QuicEvent) -> None:
        if isinstance(event, HandshakeCompleted):
            self.handshake = event
            if self.started_at is not None:
                self.handshake_latency_s = time.perf_counter() - self.started_at
            if not self.connected.done():
                self.connected.set_result(None)
        elif isinstance(event, StreamDataReceived):
            if event.stream_id != self.stream_id or len(self._buffer) + len(event.data) > _LENGTH.size:
                if not self.reply.done():
                    self.reply.set_exception(LinkProtocolError("invalid QUIC acknowledgement stream or length"))
                self.close(error_code=0x100, reason_phrase="invalid acknowledgement")
                return
            self._buffer += event.data
            if event.end_stream and not self.reply.done():
                if len(self._buffer) != _LENGTH.size:
                    self.reply.set_exception(LinkProtocolError("truncated QUIC acknowledgement"))
                else:
                    self.reply.set_result(bytes(self._buffer))
        elif isinstance(event, ConnectionTerminated):
            reason = f"QUIC connection closed: {event.reason_phrase}"
            if not self.reply.done():
                self.reply.set_exception(ConnectionError(reason))
            if not self.connected.done():
                self.connected.set_exception(ConnectionError(reason))


@dataclass
class QuicSendReport:
    delivered: int
    elapsed_s: float
    session_resumed: bool
    early_data_accepted: bool
    new_ticket: Optional[SessionTicket]
    handshake_latency_s: Optional[float] = None
    """Time from the first byte on the wire to the completed TLS handshake."""


async def quic_send_frames(
    host: str,
    port: int,
    frames: Sequence[bytes],
    ca_pem: str,
    session_ticket: Optional[SessionTicket] = None,
    server_name: str = SERVER_NAME,
    timeout: float = 3.0,
) -> QuicSendReport:
    """Send `frames` on one QUIC stream and wait for the server's delivered-frame count.

    With a `session_ticket` the frames are written before the handshake completes, so the
    client sends them as 0-RTT early data. A pre-delivery `TimeoutError`/`ConnectionError` can
    permit fallback. `DeliveryUncertainError` must not trigger an automatic batch replay.
    """
    _validate_timeout(timeout)
    batch = _validate_frames(frames)
    if session_ticket is not None and not session_ticket.is_valid:
        session_ticket = None
    configuration = QuicConfiguration(is_client=True, alpn_protocols=[ALPN], server_name=server_name)
    configuration.load_verify_locations(cadata=ca_pem.encode("ascii"))
    configuration.session_ticket = session_ticket
    tickets: List[SessionTicket] = []
    connection = QuicConnection(configuration=configuration, session_ticket_handler=tickets.append)
    loop = asyncio.get_running_loop()
    transport = protocol = None
    succeeded = False
    attempted_delivery = False
    t0 = time.perf_counter()
    try:
        async with _async_timeout(timeout):
            addresses = await loop.getaddrinfo(host, port, family=socket.AF_INET, type=socket.SOCK_DGRAM)
            transport, protocol = await loop.create_datagram_endpoint(
                lambda: _ClientProtocol(connection), local_addr=("0.0.0.0", 0), family=socket.AF_INET
            )
            protocol.started_at = time.perf_counter()
            protocol.connect(addresses[0][4])
            if session_ticket is None:
                # No application bytes are queued before the initial authenticated handshake.
                # A failure here is the only safe automatic-fallback boundary.
                await protocol.wait_connected()
            stream_id = connection.get_next_available_stream_id()
            protocol.stream_id = stream_id
            payload = b"".join(_LENGTH.pack(len(frame)) + frame for frame in batch)
            attempted_delivery = True
            connection.send_stream_data(stream_id, payload, end_stream=True)
            protocol.transmit()
            reply = await protocol.reply
            while protocol.handshake is None:
                await asyncio.sleep(0.002)
            if len(reply) != _LENGTH.size or _LENGTH.unpack(reply)[0] != len(batch):
                raise LinkProtocolError("QUIC acknowledgement does not match the sent frame count")
            succeeded = True
            handshake = protocol.handshake
            return QuicSendReport(
                delivered=len(batch),
                elapsed_s=time.perf_counter() - t0,
                session_resumed=handshake.session_resumed,
                early_data_accepted=handshake.early_data_accepted,
                new_ticket=tickets[-1] if tickets else None,
                handshake_latency_s=protocol.handshake_latency_s,
            )
    except (TimeoutError, ConnectionError, OSError) as exc:
        if attempted_delivery:
            raise DeliveryUncertainError("QUIC delivery was not confirmed; batch was not replayed") from exc
        raise
    finally:
        if protocol is not None:
            protocol.close()
            try:
                if succeeded:
                    try:
                        await asyncio.wait_for(protocol.wait_closed(), min(timeout, 0.25))
                    except TimeoutError:
                        pass
            finally:
                if not protocol.reply.done():
                    protocol.reply.cancel()
                elif not protocol.reply.cancelled():
                    protocol.reply.exception()
                if not protocol.connected.done():
                    protocol.connected.cancel()
                elif not protocol.connected.cancelled():
                    protocol.connected.exception()
                if transport is not None:
                    transport.close()


@dataclass
class Quic0RttProbe:
    """Outcome of one real `aioquic` TLS 1.3 0-RTT resumption probe."""

    frames_sent: int
    frames_delivered: int
    session_resumed: bool
    early_data_accepted: bool
    server_early_frames: int
    handshake_latency_ms: Optional[float]
    resumption_latency_ms: Optional[float]
    frame_latencies_ms: List[float] = field(default_factory=list)
    elapsed_s: float = 0.0
    ticket: Optional[SessionTicket] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        """True only when every frame was delivered, the server resumed the TLS session and
        it accepted the 0-RTT early data. A probe that fell back to a full 1-RTT handshake
        delivered its frames but measured no 0-RTT, so it is not ok."""
        return (
            self.error is None
            and self.frames_delivered == self.frames_sent
            and self.session_resumed
            and self.early_data_accepted
        )


async def quic_0rtt_resumption_probe(
    server: "QuicHiddenStateServer",
    frames: Optional[Sequence[bytes]] = None,
    *,
    frame_factory: Optional[Callable[[], Sequence[bytes]]] = None,
    client_target: Optional[Tuple[str, int]] = None,
    session_ticket: Optional[SessionTicket] = None,
    timeout: float = 3.0,
    server_name: str = SERVER_NAME,
) -> Quic0RttProbe:
    """Resume a QUIC session with 0-RTT and time the delivery of every frame.

    `client_target` overrides the address the client dials, so the probe can run through a
    deterministic fault-injection proxy. Without a `session_ticket` the probe performs a first
    handshake to obtain one and then resumes the session with it; `frame_factory` builds the
    frames right before that resumed send, so their send timestamps stay meaningful.
    """
    _validate_timeout(timeout)
    if frames is None and frame_factory is None:
        raise ValueError("the probe needs frames or a frame factory")
    host, port = client_target if client_target is not None else (server.host, server.port)
    latencies: List[float] = []
    first_delivery_at: Optional[float] = None
    early_frames_before = server.early_frames
    original = server.on_frame
    timed_frames: set = set()

    def observer(frame: bytes) -> None:
        # Every frame reaches the server's own handler; only the probe's envelope frames
        # are timed, so unrelated traffic is neither dropped nor mistaken for an envelope.
        nonlocal first_delivery_at
        if frame in timed_frames:
            now = time.time()
            try:
                envelope = FrameEnvelope.decode(frame)
            except (LinkProtocolError, TypeError):
                envelope = None
            # The server shares the sender's clock, so a timestamp in the future means the
            # bytes merely happened to parse as an envelope.
            if envelope is not None and envelope.sent_at <= now:
                latencies.append((now - envelope.sent_at) * 1000.0)
                if first_delivery_at is None:
                    first_delivery_at = now
        if original is not None:
            original(frame)

    server.on_frame = observer
    started = time.perf_counter()
    started_wall = time.time()
    try:
        ticket = session_ticket
        if ticket is None:
            # The warm-up frame is shorter than an envelope, so it is never mistaken for one.
            warmup = await quic_send_frames(
                host, port, [b"warmup"], server.identity.ca_pem,
                timeout=timeout, server_name=server_name,
            )
            ticket = warmup.new_ticket
        batch = _validate_frames(frame_factory() if frame_factory is not None else frames)
        timed_frames.update(batch)
        report = await quic_send_frames(
            host, port, batch, server.identity.ca_pem, session_ticket=ticket,
            timeout=timeout, server_name=server_name,
        )
        return Quic0RttProbe(
            frames_sent=len(batch),
            frames_delivered=report.delivered,
            session_resumed=report.session_resumed,
            early_data_accepted=report.early_data_accepted,
            server_early_frames=max(0, server.early_frames - early_frames_before),
            handshake_latency_ms=(
                report.handshake_latency_s * 1000.0 if report.handshake_latency_s is not None else None
            ),
            resumption_latency_ms=(
                (first_delivery_at - started_wall) * 1000.0 if first_delivery_at is not None else None
            ),
            frame_latencies_ms=list(latencies),
            elapsed_s=time.perf_counter() - started,
            ticket=report.new_ticket,
        )
    except (TimeoutError, ConnectionError, OSError) as exc:
        return Quic0RttProbe(
            frames_sent=len(latencies),
            frames_delivered=len(latencies),
            session_resumed=False,
            early_data_accepted=False,
            server_early_frames=max(0, server.early_frames - early_frames_before),
            handshake_latency_ms=None,
            resumption_latency_ms=(
                (first_delivery_at - started_wall) * 1000.0 if first_delivery_at is not None else None
            ),
            frame_latencies_ms=list(latencies),
            elapsed_s=time.perf_counter() - started,
            error=type(exc).__name__,
        )
    finally:
        server.on_frame = original


def _uvarint(value: int) -> bytes:
    if not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF:
        raise ValueError("multistream length must be an unsigned 32-bit integer")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _multistream_message(text: str) -> bytes:
    if not text or "\n" in text:
        raise ValueError("invalid multistream protocol name")
    payload = (text + "\n").encode("utf-8")
    if len(payload) > MAX_PROTOCOL_BYTES:
        raise ValueError("multistream protocol name exceeds size limit")
    return _uvarint(len(payload)) + payload


def _parse_multistream(data: bytes) -> str:
    if not isinstance(data, bytes) or not data or len(data) > MAX_PROTOCOL_BYTES + 5:
        raise LinkProtocolError("invalid multistream message type or size")
    length = 0
    for index, byte in enumerate(data[:5], start=1):
        length |= (byte & 0x7F) << (7 * (index - 1))
        if not byte & 0x80:
            if length > MAX_PROTOCOL_BYTES or data[:index] != _uvarint(length):
                raise LinkProtocolError("invalid multistream length prefix")
            payload = data[index:]
            if len(payload) != length or length < 2 or not payload.endswith(b"\n") or b"\n" in payload[:-1]:
                raise LinkProtocolError("truncated or noncanonical multistream message")
            try:
                return payload[:-1].decode("utf-8")
            except UnicodeDecodeError as exc:
                raise LinkProtocolError("multistream protocol name is not UTF-8") from exc
    raise LinkProtocolError("unterminated multistream length prefix")


class WebSocketHiddenStateServer:
    """Bounded multistream WebSocket receiver; non-loopback listeners require TLS."""

    def __init__(
        self, host: str = "127.0.0.1", port: int = 0, on_frame: Optional[FrameHandler] = None,
        *, ssl_context: Optional[ssl.SSLContext] = None, timeout: float = STREAM_TIMEOUT_S,
    ) -> None:
        _validate_timeout(timeout)
        if ssl_context is None and not _is_loopback_ip(host):
            raise ValueError("non-loopback WebSocket listeners require TLS")
        self.host = host
        self.port = port
        self.on_frame = on_frame
        self.frames_received = 0
        self.ssl_context = ssl_context
        self.timeout = timeout
        self._server = None

    async def start(self) -> "WebSocketHiddenStateServer":
        if self._server is not None:
            raise RuntimeError("WebSocket hidden-state server is already started")
        self._server = await ws_serve(
            self._handle, self.host, self.port, ssl=self.ssl_context,
            max_size=MAX_FRAME_BYTES, max_queue=4, compression=None,
            open_timeout=self.timeout, close_timeout=0.25,
        )
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    @property
    def url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{'wss' if self.ssl_context is not None else 'ws'}://{host}:{self.port}"

    async def _handle(self, websocket) -> None:
        try:
            async with _async_timeout(self.timeout):
                if _parse_multistream(await websocket.recv()) != MULTISTREAM:
                    raise LinkProtocolError("unsupported multistream version")
                await websocket.send(_multistream_message(MULTISTREAM))
                if _parse_multistream(await websocket.recv()) != LIBP2P_PROTOCOL:
                    await websocket.send(_multistream_message("na"))
                    raise LinkProtocolError("unsupported hidden-state protocol")
                await websocket.send(_multistream_message(LIBP2P_PROTOCOL))
                count, total = 0, 0
                while True:
                    message = await websocket.recv()
                    if not isinstance(message, bytes):
                        raise LinkProtocolError("hidden-state frames must be binary")
                    if message == b"":
                        break
                    count += 1
                    total += len(message) + _LENGTH.size
                    if count > MAX_BATCH_FRAMES or total > MAX_BATCH_BYTES or len(message) > MAX_FRAME_BYTES:
                        raise LinkProtocolError("hidden-state batch limits exceeded")
                    if self.on_frame is not None:
                        self.on_frame(message)
                    self.frames_received += 1
                await websocket.send(_LENGTH.pack(count))
        except ConnectionClosed:
            pass
        except (TimeoutError, LinkProtocolError):
            await websocket.close(code=1002, reason="invalid or timed-out hidden-state stream")
        except Exception:
            await websocket.close(code=1011, reason="hidden-state frame handler failed")

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None


async def websocket_send_frames(
    url: str, frames: Sequence[bytes], timeout: float = 5.0, *,
    ssl_context: Optional[ssl.SSLContext] = None, server_name: Optional[str] = None,
) -> int:
    """Negotiate `LIBP2P_PROTOCOL` via multistream-select, stream frames, return delivered count."""
    _validate_timeout(timeout)
    scheme = _validate_websocket_url(url)
    batch = _validate_frames(frames)
    options = {}
    # Newer websockets versions otherwise inherit ambient proxy settings, even
    # for an explicitly configured peer. Older supported versions lack this option.
    if "proxy" in inspect.signature(ws_connect).parameters:
        options["proxy"] = None
    if scheme == "wss":
        context = ssl_context or ssl.create_default_context()
        if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
            raise ValueError("WebSocket TLS must verify both the certificate and hostname")
        options["ssl"] = context
        if server_name is not None:
            options["server_hostname"] = server_name
    elif ssl_context is not None or server_name is not None:
        raise ValueError("TLS options require a wss endpoint")
    async with _async_timeout(timeout):
        async with ws_connect(
            url, max_size=MAX_PROTOCOL_BYTES + 5, max_queue=4, compression=None,
            open_timeout=timeout, close_timeout=0.25, **options,
        ) as websocket:
            await websocket.send(_multistream_message(MULTISTREAM))
            await websocket.send(_multistream_message(LIBP2P_PROTOCOL))
            header = _parse_multistream(await websocket.recv())
            selected = _parse_multistream(await websocket.recv())
            if header != MULTISTREAM or selected != LIBP2P_PROTOCOL:
                raise LinkProtocolError(f"libp2p negotiation failed: {header!r} / {selected!r}")
            for frame in batch:
                await websocket.send(frame)
            await websocket.send(b"")
            reply = await websocket.recv()
            if not isinstance(reply, bytes) or len(reply) != _LENGTH.size or _LENGTH.unpack(reply)[0] != len(batch):
                raise LinkProtocolError("WebSocket acknowledgement does not match the sent frame count")
            return len(batch)


@dataclass
class LinkRun:
    """Acknowledged delivery outcome of a local demonstration or configured remote send."""

    protocol: str
    """`quic_0rtt` (resumed, early data accepted), `quic` (no 0-RTT) or `libp2p_websocket`."""
    frames_sent: int
    frames_delivered: int
    elapsed_s: float
    session_resumed: bool = False
    early_data_accepted: bool = False
    server_early_frames: int = 0
    fallback_reason: Optional[str] = None
    handshakes: List[Dict[str, bool]] = field(default_factory=list)

    @property
    def frames_per_sec(self) -> float:
        return self.frames_delivered / max(self.elapsed_s, 1e-9)


@dataclass(frozen=True)
class LinkEndpoint:
    """Address and explicitly trusted CA of an existing hidden-state receiver.

    The receiver may be in another process or on a reachable remote host. This
    configuration does not discover peers, open NAT mappings or provision TLS.
    """

    host: str
    port: int
    ca_pem: str
    server_name: str = SERVER_NAME
    websocket_url: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.host or not self.server_name or not self.ca_pem:
            raise ValueError("endpoint host, server name and trusted CA are required")
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise ValueError("endpoint port must be between 1 and 65535")
        try:
            ip = ipaddress.ip_address(self.host)
        except ValueError:
            pass  # DNS names are resolved asynchronously when sending.
        else:
            if ip.is_unspecified or ip.is_multicast:
                raise ValueError("endpoint requires a specific unicast host")
        if self.websocket_url is not None:
            _validate_websocket_url(self.websocket_url)


class RemoteHiddenStateLink:
    """Send batches to an existing receiver, retaining one single-use TLS ticket.

    Connections are closed after each batch; the receiver stays up. Ticket state
    is protected independently of asyncio loops, including callers using a new
    loop per send. No warmup frame or local receiver is created.
    """

    def __init__(self, endpoint: LinkEndpoint, timeout: float = 3.0) -> None:
        _validate_timeout(timeout)
        self.endpoint = endpoint
        self.timeout = timeout
        self._ticket: Optional[SessionTicket] = None
        self._ticket_lock = threading.Lock()

    async def send_frames(self, frames: Sequence[bytes]) -> LinkRun:
        batch = _validate_frames(frames)
        with self._ticket_lock:
            ticket, self._ticket = self._ticket, None
        try:
            report = await quic_send_frames(
                self.endpoint.host, self.endpoint.port, batch, self.endpoint.ca_pem,
                session_ticket=ticket, server_name=self.endpoint.server_name, timeout=self.timeout,
            )
        except DeliveryUncertainError:
            raise  # A fresh transport cannot safely replay a potentially delivered batch.
        except (TimeoutError, ConnectionError, OSError) as exc:
            if self.endpoint.websocket_url is None:
                raise
            tls_options = {}
            if _validate_websocket_url(self.endpoint.websocket_url) == "wss":
                tls_options["ssl_context"] = ssl.create_default_context(cadata=self.endpoint.ca_pem)
                tls_options["server_name"] = self.endpoint.server_name
            started = time.perf_counter()
            delivered = await websocket_send_frames(
                self.endpoint.websocket_url, batch, timeout=self.timeout, **tls_options
            )
            return LinkRun(
                protocol="libp2p_websocket", frames_sent=len(batch), frames_delivered=delivered,
                elapsed_s=time.perf_counter() - started, fallback_reason=type(exc).__name__,
            )
        with self._ticket_lock:
            self._ticket = report.new_ticket
        return LinkRun(
            protocol="quic_0rtt" if report.early_data_accepted else "quic",
            frames_sent=len(batch), frames_delivered=report.delivered, elapsed_s=report.elapsed_s,
            session_resumed=report.session_resumed, early_data_accepted=report.early_data_accepted,
        )


def _unused_udp_port(host: str) -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind((host, 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


async def run_local_link(
    frames: Sequence[bytes],
    host: str = "127.0.0.1",
    on_frame: Optional[FrameHandler] = None,
    udp_blocked: bool = False,
    quic_timeout: float = 2.0,
) -> LinkRun:
    """Start QUIC + WebSocket servers on `host` and push `frames` the way a peer would.

    A first connection obtains a session ticket; the second resumes with it and sends `frames`
    as 0-RTT early data. With `udp_blocked` the client is pointed at a UDP port nothing listens
    on, so the QUIC handshake genuinely times out and the WebSocket fallback carries the frames.
    """
    _validate_timeout(quic_timeout)
    _validate_frames(frames)
    if not _is_loopback_ip(host):
        raise ValueError("run_local_link is a loopback demonstration; configure a LinkEndpoint for remote sends")
    identity = LinkIdentity.generate()
    quic_server = await QuicHiddenStateServer(identity, host, on_frame=on_frame).start()
    ws_server = WebSocketHiddenStateServer(host, on_frame=on_frame)
    try:
        await ws_server.start()
        quic_port = _unused_udp_port(host) if udp_blocked else quic_server.port
        try:
            warmup = await quic_send_frames(host, quic_port, [b"warmup"], identity.ca_pem, timeout=quic_timeout)
            report = await quic_send_frames(
                host, quic_port, frames, identity.ca_pem, session_ticket=warmup.new_ticket, timeout=quic_timeout
            )
        except DeliveryUncertainError:
            raise
        except (asyncio.TimeoutError, ConnectionError, OSError) as exc:
            t0 = time.perf_counter()
            delivered = await websocket_send_frames(ws_server.url, frames)
            return LinkRun(
                protocol="libp2p_websocket",
                frames_sent=len(frames),
                frames_delivered=delivered,
                elapsed_s=time.perf_counter() - t0,
                fallback_reason=type(exc).__name__,
            )
        return LinkRun(
            protocol="quic_0rtt" if report.early_data_accepted else "quic",
            frames_sent=len(frames),
            frames_delivered=report.delivered,
            elapsed_s=report.elapsed_s,
            session_resumed=report.session_resumed,
            early_data_accepted=report.early_data_accepted,
            server_early_frames=quic_server.early_frames,
            handshakes=list(quic_server.handshakes),
        )
    finally:
        quic_server.close()
        await ws_server.close()
