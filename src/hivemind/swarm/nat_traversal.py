# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""RFC 5389 STUN, RFC 5766 TURN and UDP hole punching for HiveMind swarm peers (`HM-S3-01`).

Everything in this module speaks the real wire protocols over real UDP sockets:

- `StunMessage` encodes and decodes RFC 5389 messages, including MESSAGE-INTEGRITY
  (HMAC-SHA1 with short- or long-term credentials) and FINGERPRINT. It reproduces the
  RFC 5769 test vectors byte for byte.
- `stun_binding` sends Binding requests with RFC 5389 §7.2.1 retransmission and returns the
  XOR-MAPPED-ADDRESS (server-reflexive transport address).
- `classify_nat_mapping` applies the RFC 5780 §4.3 mapping-behaviour test against two STUN
  servers to tell endpoint-independent (cone) mappings from address-dependent (symmetric) ones.
- `TurnClient` performs RFC 5766 Allocate (401 long-term-credential challenge, 438 stale-nonce
  retry), CreatePermission, Refresh and Send/Data indications.
- `udp_hole_punch` runs ICE-style (RFC 8445) simultaneous-open connectivity checks.
- `StunServer` and `TurnRelayServer` let any HiveMind node act as a STUN reflector or a TURN
  relay for peers stuck behind symmetric NATs.

Passwords must already be SASLprep-normalised (RFC 4013); plain ASCII passwords need no change.
Only IPv4 transport is used for sockets; IPv6 addresses are still encoded and decoded.
"""

from __future__ import annotations

import hashlib
import heapq
import hmac
import ipaddress
import itertools
import math
import random
import secrets
import select
import socket
import struct
import threading
import time
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

Address = Tuple[str, int]
ServerSpec = Union[str, Address]

MAGIC_COOKIE = 0x2112A442
FINGERPRINT_XOR = 0x5354554E
HEADER = struct.Struct("!HHI12s")

BINDING = 0x001
ALLOCATE = 0x003
REFRESH = 0x004
SEND = 0x006
DATA = 0x007
CREATE_PERMISSION = 0x008
CHANNEL_BIND = 0x009

REQUEST = 0b00
INDICATION = 0b01
SUCCESS = 0b10
ERROR = 0b11

MAPPED_ADDRESS = 0x0001
USERNAME = 0x0006
MESSAGE_INTEGRITY = 0x0008
ERROR_CODE = 0x0009
CHANNEL_NUMBER = 0x000C
LIFETIME = 0x000D
CHANNEL_LIFETIME = 0x000F
XOR_PEER_ADDRESS = 0x0012
DATA_ATTR = 0x0013
REALM = 0x0014
NONCE = 0x0015
XOR_RELAYED_ADDRESS = 0x0016
REQUESTED_TRANSPORT = 0x0019
XOR_MAPPED_ADDRESS = 0x0020
SOFTWARE = 0x8022
FINGERPRINT = 0x8028

CHANNEL_NUMBER_MIN = 0x4000
"""Lowest TURN channel number (RFC 5766 §11.3: the two most significant bits are 01)."""
CHANNEL_NUMBER_MAX = 0x7FFF
CHANNEL_DATA_FLAG = 0x40
"""ChannelData messages start with the two bits 01, unlike STUN messages (00)."""
CHANNEL_HEADER = struct.Struct("!HH")

UDP_PROTOCOL_NUMBER = 17
SOFTWARE_NAME = "HiveMind STUN/TURN"
PERMISSION_LIFETIME_S = 300.0
TRANSACTION_CACHE_LIFETIME_S = 40.0
TRANSACTION_CACHE_LIMIT = 1024
MAX_PENDING_INDICATIONS = 256
DEFAULT_STUN_SERVERS: Tuple[str, ...] = ("stun.l.google.com:19302", "stun.cloudflare.com:3478")
"""Public STUN servers used when a caller does not name one."""


class StunError(Exception):
    """A STUN/TURN transaction failed: timeout, error response or failed integrity check."""

    def __init__(self, message: str, code: Optional[int] = None) -> None:
        super().__init__(message)
        self.code = code


def message_type(method: int, cls: int) -> int:
    """Interleave the 12-bit method and 2-bit class into the 14-bit STUN message type."""
    return (
        ((method & 0x0F80) << 2)
        | ((method & 0x0070) << 1)
        | (method & 0x000F)
        | ((cls & 0b10) << 7)
        | ((cls & 0b01) << 4)
    )


def split_message_type(msg_type: int) -> Tuple[int, int]:
    method = (msg_type & 0x000F) | ((msg_type & 0x00E0) >> 1) | ((msg_type & 0x3E00) >> 2)
    cls = ((msg_type & 0x0100) >> 7) | ((msg_type & 0x0010) >> 4)
    return method, cls


def long_term_key(username: str, realm: str, password: str) -> bytes:
    """RFC 5389 §15.4 long-term credential key: MD5(username ":" realm ":" password)."""
    return hashlib.md5(f"{username}:{realm}:{password}".encode("utf-8")).digest()


def short_term_key(password: str) -> bytes:
    """RFC 5389 §15.4 short-term credential key: the password itself."""
    return password.encode("utf-8")


def encode_xor_address(addr: Address, transaction_id: bytes) -> bytes:
    ip = ipaddress.ip_address(addr[0])
    port = addr[1] ^ (MAGIC_COOKIE >> 16)
    if ip.version == 4:
        return struct.pack("!BBHI", 0, 0x01, port, int(ip) ^ MAGIC_COOKIE)
    key = int.from_bytes(struct.pack("!I", MAGIC_COOKIE) + transaction_id, "big")
    return struct.pack("!BBH", 0, 0x02, port) + (int(ip) ^ key).to_bytes(16, "big")


def decode_xor_address(value: bytes, transaction_id: bytes) -> Address:
    family = _address_family(value)
    if len(transaction_id) != 12:
        raise StunError("invalid STUN transaction ID length")
    port = struct.unpack("!H", value[2:4])[0] ^ (MAGIC_COOKIE >> 16)
    if family == 0x01:
        ip: Union[ipaddress.IPv4Address, ipaddress.IPv6Address] = ipaddress.IPv4Address(
            struct.unpack("!I", value[4:8])[0] ^ MAGIC_COOKIE
        )
    elif family == 0x02:
        key = int.from_bytes(struct.pack("!I", MAGIC_COOKIE) + transaction_id, "big")
        ip = ipaddress.IPv6Address(int.from_bytes(value[4:20], "big") ^ key)
    else:
        raise StunError(f"unknown address family {family:#x}")
    return str(ip), port


def decode_plain_address(value: bytes) -> Address:
    family = _address_family(value)
    port = struct.unpack("!H", value[2:4])[0]
    if family == 0x01:
        return str(ipaddress.IPv4Address(value[4:8])), port
    if family == 0x02:
        return str(ipaddress.IPv6Address(value[4:20])), port
    raise StunError(f"unknown address family {family:#x}")


def _address_family(value: bytes) -> int:
    if len(value) < 4:
        raise StunError("truncated STUN address")
    family = value[1]
    expected = {0x01: 8, 0x02: 20}.get(family)
    if expected is None or len(value) != expected:
        raise StunError("invalid STUN address family or length")
    return family


def is_channel_data(data: bytes) -> bool:
    """True when `data` is a ChannelData message rather than a STUN message (RFC 5766 §11.4)."""
    return len(data) >= CHANNEL_HEADER.size and bool(data[0] & CHANNEL_DATA_FLAG) and not data[0] & 0x80


def encode_channel_data(channel_number: int, payload: bytes) -> bytes:
    """Serialise a ChannelData message: channel number, length and the application data."""
    if isinstance(channel_number, bool) or not isinstance(channel_number, int):
        raise TypeError("TURN channel number must be an integer")
    if not 0 <= channel_number <= 0xFFFF:
        raise ValueError("TURN channel number must fit in 16 bits")
    if len(payload) > 0xFFFF:
        raise ValueError("TURN channel payload exceeds the 16-bit length field")
    return CHANNEL_HEADER.pack(channel_number, len(payload)) + payload


def encode_channel_number(number: int) -> bytes:
    """Encode a CHANNEL-NUMBER attribute (RFC 5766 §14.1: 16-bit number + 16-bit RFFU = 0)."""
    if isinstance(number, bool) or not isinstance(number, int):
        raise ValueError("channel number must be an integer")
    if not CHANNEL_NUMBER_MIN <= number <= CHANNEL_NUMBER_MAX:
        raise ValueError("channel number must be in 0x4000-0x7FFF (RFC 5766 §11)")
    return struct.pack("!HH", number, 0)


def decode_channel_number(value: bytes) -> int:
    """Decode a CHANNEL-NUMBER attribute; the RFFU field is ignored on reception (§14.1)."""
    if len(value) != 4:
        raise StunError("CHANNEL-NUMBER must be 4 bytes long")
    number = struct.unpack("!H", value[:2])[0]
    if not CHANNEL_NUMBER_MIN <= number <= CHANNEL_NUMBER_MAX:
        raise StunError("channel number outside 0x4000-0x7FFF")
    return number


def decode_channel_data(data: bytes) -> Tuple[int, bytes]:
    """Parse a ChannelData message into `(channel_number, payload)`."""
    if not is_channel_data(data):
        raise StunError("not a ChannelData message")
    channel_number, length = CHANNEL_HEADER.unpack_from(data, 0)
    payload = data[CHANNEL_HEADER.size : CHANNEL_HEADER.size + length]
    if len(payload) != length:
        raise StunError("truncated ChannelData message")
    return channel_number, payload


def encode_error_code(code: int, reason: str) -> bytes:
    return struct.pack("!HBB", 0, code // 100, code % 100) + reason.encode("utf-8")


def decode_error_code(value: Optional[bytes]) -> Tuple[int, str]:
    if value is None:
        return 0, ""
    if len(value) < 4 or not 3 <= (value[2] & 0x07) <= 6 or value[3] > 99:
        raise StunError("invalid STUN ERROR-CODE")
    return (value[2] & 0x07) * 100 + value[3], value[4:].decode("utf-8", "replace")


def _wire_attributes(data: bytes) -> List[Tuple[int, int, bytes]]:
    """Validate one UDP STUN frame before exposing any attribute slices."""
    if len(data) < HEADER.size:
        raise StunError("datagram shorter than a STUN header")
    mtype, length, cookie, _ = HEADER.unpack_from(data)
    if mtype & 0xC000 or cookie != MAGIC_COOKIE or length % 4 or len(data) != HEADER.size + length:
        raise StunError("not a STUN message")
    entries = []
    offset, end = HEADER.size, len(data)
    while offset < end:
        if offset + 4 > end:
            raise StunError("truncated STUN attribute header")
        attr_type, attr_len = struct.unpack_from("!HH", data, offset)
        next_offset = offset + 4 + attr_len + (-attr_len % 4)
        if next_offset > end:
            raise StunError("STUN attribute exceeds message bounds")
        if attr_type == FINGERPRINT and (attr_len != 4 or next_offset != end):
            raise StunError("FINGERPRINT must be a final four-byte attribute")
        entries.append((offset, attr_type, data[offset + 4 : offset + 4 + attr_len]))
        offset = next_offset
    return entries


def _validate_attribute(attr_type: int, value: bytes) -> None:
    fixed_lengths = {MESSAGE_INTEGRITY: 20, FINGERPRINT: 4, LIFETIME: 4, REQUESTED_TRANSPORT: 4}
    if attr_type in fixed_lengths and len(value) != fixed_lengths[attr_type]:
        raise StunError(f"invalid STUN attribute {attr_type:#06x} length")
    if attr_type in (MAPPED_ADDRESS, XOR_MAPPED_ADDRESS, XOR_PEER_ADDRESS, XOR_RELAYED_ADDRESS):
        _address_family(value)
    if attr_type == ERROR_CODE:
        decode_error_code(value)


def _pack_attribute(attr_type: int, value: bytes, pad_byte: bytes) -> bytes:
    return struct.pack("!HH", attr_type, len(value)) + value + pad_byte * (-len(value) % 4)


@dataclass
class StunMessage:
    """A STUN message: method, class, transaction id and ordered attributes."""

    method: int
    cls: int
    transaction_id: bytes = field(default_factory=lambda: secrets.token_bytes(12))
    attributes: List[Tuple[int, bytes]] = field(default_factory=list)

    def add(self, attr_type: int, value: bytes) -> "StunMessage":
        self.attributes.append((attr_type, value))
        return self

    def get(self, attr_type: int) -> Optional[bytes]:
        for current, value in self.attributes:
            if current == attr_type:
                return value
        return None

    def encode(
        self,
        integrity_key: Optional[bytes] = None,
        fingerprint: bool = False,
        pad_byte: bytes = b"\x00",
    ) -> bytes:
        """Serialise the message, appending MESSAGE-INTEGRITY and/or FINGERPRINT last.

        RFC 5389 lets padding bytes take any value; `pad_byte` exists so the RFC 5769 vectors
        (which pad with spaces) can be reproduced exactly.
        """
        mtype = message_type(self.method, self.cls)
        body = b"".join(
            _pack_attribute(t, v, pad_byte)
            for t, v in self.attributes
            if t not in (MESSAGE_INTEGRITY, FINGERPRINT)
        )
        if integrity_key is not None:
            header = HEADER.pack(mtype, len(body) + 24, MAGIC_COOKIE, self.transaction_id)
            mac = hmac.new(integrity_key, header + body, hashlib.sha1).digest()
            body += _pack_attribute(MESSAGE_INTEGRITY, mac, pad_byte)
        if fingerprint:
            header = HEADER.pack(mtype, len(body) + 8, MAGIC_COOKIE, self.transaction_id)
            crc = (zlib.crc32(header + body) & 0xFFFFFFFF) ^ FINGERPRINT_XOR
            body += _pack_attribute(FINGERPRINT, struct.pack("!I", crc), pad_byte)
        return HEADER.pack(mtype, len(body), MAGIC_COOKIE, self.transaction_id) + body

    @classmethod
    def decode(cls, data: bytes) -> "StunMessage":
        entries = _wire_attributes(data)
        mtype, _, _, transaction_id = HEADER.unpack_from(data)
        method, klass = split_message_type(mtype)
        message = cls(method, klass, transaction_id)
        after_integrity = False
        for offset, attr_type, value in entries:
            # RFC 5389 section 15.4: attributes after the first MAC are not
            # authenticated and MUST NOT influence the message's semantics.
            if after_integrity and attr_type != FINGERPRINT:
                continue
            _validate_attribute(attr_type, value)
            if attr_type == FINGERPRINT:
                expected = (zlib.crc32(data[:offset]) & 0xFFFFFFFF) ^ FINGERPRINT_XOR
                if struct.unpack("!I", value)[0] != expected:
                    raise StunError("STUN FINGERPRINT mismatch")
            message.attributes.append((attr_type, value))
            after_integrity = after_integrity or attr_type == MESSAGE_INTEGRITY
        return message


def verify_message_integrity(raw: bytes, key: bytes) -> bool:
    """Check MESSAGE-INTEGRITY per RFC 5389 §15.4 (length field covers up to the MAC)."""
    try:
        StunMessage.decode(raw)
        entries = _wire_attributes(raw)
    except StunError:
        return False
    mtype = struct.unpack_from("!H", raw)[0]
    for offset, attr_type, value in entries:
        if attr_type == MESSAGE_INTEGRITY:
            header = struct.pack("!HH", mtype, offset + 24 - HEADER.size) + raw[4:HEADER.size]
            expected = hmac.new(key, header + raw[HEADER.size : offset], hashlib.sha1).digest()
            return hmac.compare_digest(expected, value)
    return False


def verify_fingerprint(raw: bytes) -> bool:
    """Check the trailing FINGERPRINT attribute (CRC-32 XOR 0x5354554E, RFC 5389 §15.5)."""
    try:
        entries = _wire_attributes(raw)
    except StunError:
        return False
    if not entries or entries[-1][1] != FINGERPRINT:
        return False
    expected = (zlib.crc32(raw[:-8]) & 0xFFFFFFFF) ^ FINGERPRINT_XOR
    return struct.unpack_from("!I", raw, len(raw) - 4)[0] == expected


def udp_socket(bind: Address = ("0.0.0.0", 0)) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if hasattr(socket, "SIO_UDP_CONNRESET"):
        # Windows otherwise turns an ICMP port-unreachable into an error on the next recvfrom.
        sock.ioctl(socket.SIO_UDP_CONNRESET, False)
    sock.bind(bind)
    return sock


def resolve(server: ServerSpec) -> Address:
    """Resolve `"host:port"` (or a `(host, port)` tuple) to an IPv4 transport address."""
    if isinstance(server, tuple):
        host, port = server[0], int(server[1])
    else:
        host, _, port_text = server.rpartition(":")
        host, port = host.strip("[]"), int(port_text)
    info = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_DGRAM)
    return info[0][4][0], info[0][4][1]


def outbound_interface_ip(target: Address) -> str:
    """Local IP the OS would use to reach `target` (UDP connect sends no packets)."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(target)
        return probe.getsockname()[0]
    finally:
        probe.close()


def stun_transaction(
    sock: socket.socket,
    server: Address,
    request: bytes,
    transaction_id: bytes,
    timeout: float = 3.0,
    initial_rto: float = 0.1,
    on_other: Optional[Callable[[bytes, Address], None]] = None,
    response_validator: Optional[Callable[[StunMessage, bytes], bool]] = None,
) -> Tuple[StunMessage, bytes, float]:
    """Send `request` with RFC 5389 §7.2.1 retransmission (RTO doubling) and await its response.

    Datagrams that are not the matching response are handed to `on_other`, so a TURN client
    sharing the socket does not lose Data indications. Source, method, response class and
    transaction ID must match. A validator can additionally require authentication; invalid
    replies are discarded without ending retransmission. Returns `(message, raw, rtt_ms)`.
    """
    sent = StunMessage.decode(request)
    if sent.cls != REQUEST or sent.transaction_id != transaction_id:
        raise StunError("transaction does not match its STUN request")
    if timeout <= 0 or initial_rto <= 0:
        raise ValueError("STUN timeout and initial RTO must be positive")
    deadline = time.monotonic() + timeout
    rto = initial_rto
    next_send = time.monotonic()
    t0 = time.perf_counter()
    while True:
        now = time.monotonic()
        if now >= deadline:
            raise StunError(f"no STUN response from {server[0]}:{server[1]} within {timeout:.1f}s")
        if now >= next_send:
            sock.sendto(request, server)
            next_send = now + rto
            rto *= 2
        wait = max(0.0, min(next_send, deadline) - time.monotonic())
        ready, _, _ = select.select([sock], [], [], wait)
        if not ready:
            continue
        try:
            data, addr = sock.recvfrom(65535)
        except OSError:
            continue
        try:
            message = StunMessage.decode(data)
        except StunError:
            message = None
        if (
            message is None
            or addr != server
            or message.transaction_id != transaction_id
            or message.method != sent.method
            or message.cls not in (SUCCESS, ERROR)
        ):
            if on_other is not None:
                on_other(data, addr)
            continue
        if response_validator is not None and not response_validator(message, data):
            continue
        return message, data, (time.perf_counter() - t0) * 1000.0


@dataclass
class StunBindingResult:
    server: Address
    local: Address
    mapped: Address
    rtt_ms: float


def stun_binding(sock: socket.socket, server: ServerSpec, timeout: float = 3.0) -> StunBindingResult:
    """Send an RFC 5389 Binding request and return the server-reflexive address it reports."""
    server_addr = resolve(server)
    request = StunMessage(BINDING, REQUEST).add(SOFTWARE, SOFTWARE_NAME.encode("utf-8"))
    message, _, rtt_ms = stun_transaction(
        sock, server_addr, request.encode(fingerprint=True), request.transaction_id, timeout
    )
    if message.cls == ERROR:
        code, reason = decode_error_code(message.get(ERROR_CODE))
        raise StunError(f"Binding error {code} {reason}", code)
    xor_value = message.get(XOR_MAPPED_ADDRESS)
    plain_value = message.get(MAPPED_ADDRESS)
    if xor_value is not None:
        mapped = decode_xor_address(xor_value, message.transaction_id)
    elif plain_value is not None:
        mapped = decode_plain_address(plain_value)
    else:
        raise StunError("Binding response carried no mapped address")
    return StunBindingResult(server_addr, sock.getsockname(), mapped, rtt_ms)


@dataclass
class NatMapping:
    """RFC 5780 §4.3 mapping behaviour observed for one local socket."""

    behaviour: str
    """`no_nat`, `endpoint_independent`, `address_dependent` or `unknown`."""
    local: Address
    mapped: List[Address]
    servers: List[Address]
    rtt_ms: List[float] = field(default_factory=list)

    @property
    def nat_type(self) -> str:
        return {
            "no_nat": "open",
            "endpoint_independent": "cone",
            "address_dependent": "symmetric",
        }.get(self.behaviour, "unknown")

    @property
    def hole_punchable(self) -> bool:
        return self.behaviour in ("no_nat", "endpoint_independent")


def classify_nat_mapping(
    sock: socket.socket, servers: Sequence[ServerSpec], timeout: float = 2.0
) -> NatMapping:
    """Compare the mappings two STUN servers report for the same local socket."""
    results: List[StunBindingResult] = []
    for server in servers:
        try:
            results.append(stun_binding(sock, server, timeout))
        except (StunError, OSError):
            continue
        if len(results) == 2:
            break
    if not results:
        raise StunError("no STUN server answered")
    local_ip, local_port = sock.getsockname()
    if local_ip in ("0.0.0.0", ""):
        local_ip = outbound_interface_ip(results[0].server)
    mapped = [r.mapped for r in results]
    if mapped[0] == (local_ip, local_port):
        behaviour = "no_nat"
    elif len(mapped) < 2:
        behaviour = "unknown"
    elif mapped[0] == mapped[1]:
        behaviour = "endpoint_independent"
    else:
        behaviour = "address_dependent"
    return NatMapping(
        behaviour, (local_ip, local_port), mapped, [r.server for r in results], [r.rtt_ms for r in results]
    )


PUNCH_MAGIC = b"HMPUNCH1"


@dataclass
class PunchResult:
    succeeded: bool
    peer: Optional[Address]
    rtt_ms: Optional[float]
    probes_sent: int


def udp_hole_punch(
    sock: socket.socket,
    candidates: Sequence[Address],
    session: bytes,
    timeout: float = 2.0,
    interval: float = 0.01,
    linger: float = 0.05,
) -> PunchResult:
    """Simultaneous-open UDP hole punching with ICE-style connectivity checks (RFC 8445 §7.2).

    Both peers call this at the same time with each other's candidates and a shared `session`
    id from signalling. Each side keeps probing every candidate; a probe that gets through is
    answered from the address it arrived on, which is the hole the peer's NAT just opened. A
    side succeeds when it receives that answer, then lingers briefly to answer late probes.
    """
    syn = PUNCH_MAGIC + b"S" + session
    ack = PUNCH_MAGIC + b"A" + session
    deadline = time.monotonic() + timeout
    t0 = time.perf_counter()
    probes = 0
    next_probe = 0.0
    winner: Optional[Address] = None
    rtt_ms: Optional[float] = None
    while time.monotonic() < deadline:
        now = time.monotonic()
        if winner is None and now >= next_probe:
            for candidate in candidates:
                try:
                    sock.sendto(syn, candidate)
                    probes += 1
                except OSError:
                    continue
            next_probe = now + interval
        wake = min(next_probe, deadline) if winner is None else deadline
        try:
            ready, _, _ = select.select([sock], [], [], max(0.0, wake - time.monotonic()))
        except (OSError, ValueError):
            break
        if not ready:
            continue
        try:
            data, addr = sock.recvfrom(2048)
        except OSError:
            continue
        if data == syn:
            try:
                sock.sendto(ack, addr)
            except OSError:
                continue
        elif data == ack and winner is None:
            winner = addr
            rtt_ms = (time.perf_counter() - t0) * 1000.0
            deadline = time.monotonic() + linger
    return PunchResult(winner is not None, winner, rtt_ms, probes)


class TurnClient:
    """RFC 5766 TURN client over UDP using long-term credentials.

    Besides the Send/Data indication path the client can bind a channel to a peer
    (RFC 5766 §11.2) and then exchange ChannelData messages (§11.4) with it.
    """

    def __init__(
        self,
        server: ServerSpec,
        username: str,
        password: str,
        sock: Optional[socket.socket] = None,
        timeout: float = 3.0,
    ) -> None:
        self.server = resolve(server)
        self.username = username
        self.password = password
        self.sock = sock if sock is not None else udp_socket()
        self._owns_socket = sock is None
        self.timeout = timeout
        self.realm: Optional[str] = None
        self.nonce: Optional[bytes] = None
        self.key: Optional[bytes] = None
        self.relayed: Optional[Address] = None
        self.mapped: Optional[Address] = None
        self.lifetime = 0
        self.channels: Dict[Address, int] = {}
        """Channels bound by this client, keyed by the peer transport address."""
        self._channel_peers: Dict[int, Address] = {}
        self._inbox: List[Tuple[Address, bytes]] = []

    def _on_other(self, data: bytes, addr: Address) -> None:
        self._handle_datagram(data, addr)

    def _handle_datagram(self, data: bytes, addr: Address) -> None:
        """Absorb one datagram: a ChannelData message or a STUN Data indication."""
        if addr != self.server:
            return
        if is_channel_data(data):
            try:
                channel_number, payload = decode_channel_data(data)
            except StunError:
                return
            peer = self._channel_peers.get(channel_number)
            if peer is not None and len(self._inbox) < MAX_PENDING_INDICATIONS:
                self._inbox.append((peer, payload))
            return
        try:
            message = StunMessage.decode(data)
        except StunError:
            return
        if message.method == DATA and message.cls == INDICATION:
            peer, payload = message.get(XOR_PEER_ADDRESS), message.get(DATA_ATTR)
            if peer is not None and payload is not None and len(self._inbox) < MAX_PENDING_INDICATIONS:
                self._inbox.append((decode_xor_address(peer, message.transaction_id), payload))

    def _valid_response(self, message: StunMessage, raw: bytes) -> bool:
        """Permit unsigned authentication challenges only, never unsigned results."""
        if message.get(MESSAGE_INTEGRITY) is not None:
            key = self.key
            # A server may sign the first 401 with the realm it is introducing.
            if key is None and message.cls == ERROR:
                realm = message.get(REALM)
                if realm:
                    try:
                        key = long_term_key(self.username, realm.decode("utf-8"), self.password)
                    except UnicodeDecodeError:
                        return False
            return key is not None and verify_message_integrity(raw, key)
        if message.cls != ERROR:
            return False
        code, _ = decode_error_code(message.get(ERROR_CODE))
        realm, nonce = message.get(REALM), message.get(NONCE)
        if code not in (401, 438) or not realm or not nonce or len(realm) > 763 or len(nonce) > 763:
            return False
        try:
            challenge_realm = realm.decode("utf-8")
        except UnicodeDecodeError:
            return False
        if self.realm is not None and challenge_realm != self.realm:
            return False
        return code == 401 or self.key is not None

    def _request(
        self, method: int, attributes: Sequence[Tuple[int, Callable[[bytes], bytes]]]
    ) -> StunMessage:
        retried_stale_nonce = False
        while True:
            request = StunMessage(method, REQUEST)
            for attr_type, build in attributes:
                request.add(attr_type, build(request.transaction_id))
            if self.key is not None and self.realm is not None and self.nonce is not None:
                request.add(USERNAME, self.username.encode("utf-8"))
                request.add(REALM, self.realm.encode("utf-8"))
                request.add(NONCE, self.nonce)
                raw = request.encode(integrity_key=self.key, fingerprint=True)
            else:
                raw = request.encode(fingerprint=True)
            message, _, _ = stun_transaction(
                self.sock, self.server, raw, request.transaction_id, self.timeout,
                on_other=self._on_other, response_validator=self._valid_response,
            )
            if message.cls == SUCCESS:
                return message
            code, reason = decode_error_code(message.get(ERROR_CODE))
            fresh_nonce = message.get(NONCE)
            had_credentials = self.key is not None
            if code in (401, 438) and fresh_nonce is not None:
                # Changing only the nonce does not authorize retrying a 401.
                # With fixed user credentials this is a terminal rejection.
                if had_credentials and code == 401:
                    raise StunError("TURN server rejected the credentials (401)", code)
                if code == 438 and (retried_stale_nonce or fresh_nonce == self.nonce):
                    raise StunError("TURN server keeps reporting a stale nonce", code)
                retried_stale_nonce = retried_stale_nonce or code == 438
                realm = message.get(REALM)
                if realm is not None:
                    try:
                        challenge_realm = realm.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise StunError("TURN challenge carried an invalid REALM", code) from exc
                    if self.realm is not None and self.realm != challenge_realm:
                        raise StunError("TURN challenge changed REALM", code)
                    self.realm = challenge_realm
                if not self.realm or not fresh_nonce or len(fresh_nonce) > 763 or len(realm or b"") > 763:
                    raise StunError("TURN challenge carried an invalid REALM or NONCE", code)
                self.nonce = fresh_nonce
                self.key = long_term_key(self.username, self.realm, self.password)
                continue
            raise StunError(f"TURN request {method:#05x} failed: {code} {reason}", code)

    def allocate(self, lifetime: int = 600) -> Address:
        message = self._request(
            ALLOCATE,
            [
                (REQUESTED_TRANSPORT, lambda _t: struct.pack("!B3x", UDP_PROTOCOL_NUMBER)),
                (LIFETIME, lambda _t: struct.pack("!I", lifetime)),
            ],
        )
        relayed = message.get(XOR_RELAYED_ADDRESS)
        if relayed is None:
            raise StunError("Allocate success carried no XOR-RELAYED-ADDRESS")
        self.relayed = decode_xor_address(relayed, message.transaction_id)
        mapped = message.get(XOR_MAPPED_ADDRESS)
        if mapped is not None:
            self.mapped = decode_xor_address(mapped, message.transaction_id)
        granted = message.get(LIFETIME)
        self.lifetime = struct.unpack("!I", granted)[0] if granted else lifetime
        return self.relayed

    def create_permission(self, peer_ip: str) -> None:
        self._request(
            CREATE_PERMISSION,
            [(XOR_PEER_ADDRESS, lambda txid: encode_xor_address((peer_ip, 0), txid))],
        )

    def refresh(self, lifetime: int = 600) -> int:
        message = self._request(REFRESH, [(LIFETIME, lambda _t: struct.pack("!I", lifetime))])
        granted = message.get(LIFETIME)
        self.lifetime = struct.unpack("!I", granted)[0] if granted else lifetime
        return self.lifetime

    def send_to(self, peer: Address, payload: bytes) -> None:
        """Relay `payload` to `peer` through the allocation (Send indication, RFC 5766 §10.1)."""
        indication = StunMessage(SEND, INDICATION)
        indication.add(XOR_PEER_ADDRESS, encode_xor_address(peer, indication.transaction_id))
        indication.add(DATA_ATTR, payload)
        self.sock.sendto(indication.encode(fingerprint=True), self.server)

    def bind_channel(self, peer: Address, channel_number: Optional[int] = None) -> int:
        """Bind a channel to `peer` (RFC 5766 §11.1) and return the channel number.

        The client picks the number in 0x4000-0x7FFF and sends it with XOR-PEER-ADDRESS;
        the server only confirms. Without `channel_number` the client reuses the number
        already bound to `peer` (a refresh, §11.2) or the lowest free one. Binding also
        installs the peer permission, so no separate CreatePermission is needed.
        """
        if channel_number is None:
            channel_number = self.channels.get(peer)
            if channel_number is None:
                channel_number = self._free_channel_number()
        encoded_number = encode_channel_number(channel_number)
        self._request(
            CHANNEL_BIND,
            [
                (CHANNEL_NUMBER, lambda _txid: encoded_number),
                (XOR_PEER_ADDRESS, lambda txid: encode_xor_address(peer, txid)),
            ],
        )
        self.channels[peer] = channel_number
        self._channel_peers[channel_number] = peer
        return channel_number

    def _free_channel_number(self) -> int:
        for number in range(CHANNEL_NUMBER_MIN, CHANNEL_NUMBER_MAX + 1):
            if number not in self._channel_peers:
                return number
        raise StunError("every TURN channel number of this allocation is bound")

    def send_channel(self, peer: Address, payload: bytes) -> None:
        """Send `payload` to `peer` over the channel bound to it (RFC 5766 §11.4)."""
        channel_number = self.channels.get(peer)
        if channel_number is None:
            raise StunError("no TURN channel is bound to that peer")
        self.sock.sendto(encode_channel_data(channel_number, payload), self.server)

    def recv_from(self, timeout: float = 2.0) -> Tuple[Address, bytes]:
        """Wait for the next relayed payload and return `(peer, payload)`."""
        deadline = time.monotonic() + timeout
        while not self._inbox:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise StunError("no relayed data before the timeout")
            ready, _, _ = select.select([self.sock], [], [], remaining)
            if not ready:
                continue
            try:
                data, addr = self.sock.recvfrom(65535)
            except OSError:
                continue
            self._on_other(data, addr)
        return self._inbox.pop(0)

    def close(self) -> None:
        """Release the allocation (Refresh with LIFETIME 0) and close an owned socket."""
        try:
            if self.relayed is not None:
                self.refresh(0)
        except (StunError, OSError):
            pass
        finally:
            self.relayed = None
            self.channels.clear()
            self._channel_peers.clear()
            if self._owns_socket:
                self.sock.close()


class StunServer:
    """RFC 5389 Binding responder serving on a background thread."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.sock = udp_socket((host, port))
        self.address: Address = self.sock.getsockname()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "StunServer":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    @property
    def url(self) -> str:
        return f"{self.address[0]}:{self.address[1]}"

    def start(self) -> "StunServer":
        self._thread = threading.Thread(target=self._serve, name=type(self).__name__, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._close_sockets()

    def _close_sockets(self) -> None:
        self.sock.close()

    def _sockets(self) -> List[socket.socket]:
        return [self.sock]

    def _serve(self) -> None:
        while not self._stop.is_set():
            ready, _, _ = select.select(self._sockets(), [], [], 0.05)
            for sock in ready:
                self._on_readable(sock)
            self._housekeeping()

    def _housekeeping(self) -> None:
        """Periodic hook of the base UDP server: the STUN server keeps no expiring state, so there is
        nothing to reap here; the TURN subclass overrides this to expire allocations and channels.
        The base implementation only counts ticks so the serve loop's cadence is observable."""
        self.housekeeping_ticks = getattr(self, "housekeeping_ticks", 0) + 1

    def _on_readable(self, sock: socket.socket) -> None:
        try:
            data, addr = sock.recvfrom(65535)
        except OSError:
            return
        self.handle_datagram(data, addr)

    def mapped_address_for(self, addr: Address) -> Address:
        """The reflexive address reported to `addr` (the address the datagram came from)."""
        return addr

    def handle_datagram(self, data: bytes, addr: Address) -> None:
        try:
            message = StunMessage.decode(data)
        except StunError:
            return
        if message.method == BINDING and message.cls == REQUEST:
            response = StunMessage(BINDING, SUCCESS, message.transaction_id)
            response.add(
                XOR_MAPPED_ADDRESS,
                encode_xor_address(self.mapped_address_for(addr), message.transaction_id),
            )
            response.add(SOFTWARE, SOFTWARE_NAME.encode("utf-8"))
            self.sock.sendto(response.encode(fingerprint=True), addr)


@dataclass
class _Allocation:
    client: Address
    username: str
    relay: socket.socket
    expires: float
    permissions: Dict[str, float] = field(default_factory=dict)

    def permits(self, ip: str) -> bool:
        return self.permissions.get(ip, 0.0) > time.monotonic()

    def permit(self, ip: str) -> None:
        self.permissions[ip] = time.monotonic() + PERMISSION_LIFETIME_S


@dataclass
class _Channel:
    """A channel bound between an allocation and one peer (RFC 5766 §11)."""

    number: int
    peer: Address
    allocation: _Allocation
    expires: float


class TurnRelayServer(StunServer):
    """Minimal RFC 5766 TURN relay (UDP allocations, permissions, Send/Data indications).

    Also answers Binding requests, like production TURN servers do. Intended for HiveMind nodes
    that volunteer as relays; TCP allocations are not implemented, but channels are:
    `ChannelBind` (RFC 5766 §11.2) allocates a channel number and `ChannelData` (§11.4)
    carries the payload without a STUN wrapper.
    For a wildcard bind, the relay address defaults to the interface used to reach the client.
    Set `advertised_ip` explicitly when a NAT forwards the relay's UDP ports to a public address.
    """

    _DENIED_PRIVATE_NETWORKS = (
        ipaddress.IPv4Network("10.0.0.0/8"),
        ipaddress.IPv4Network("172.16.0.0/12"),
        ipaddress.IPv4Network("192.168.0.0/16"),
        ipaddress.IPv4Network("100.64.0.0/10"),
    )

    def __init__(
        self,
        users: Dict[str, str],
        host: str = "127.0.0.1",
        port: int = 0,
        realm: str = "hivemind.local",
        max_lifetime: int = 3600,
        advertised_ip: Optional[str] = None,
        channel_lifetime: int = 600,
        max_allocations_per_user: int = 16,
        max_total_allocations: int = 256,
        allow_private_peers: bool = False,
        allow_loopback_peers: bool = True,
    ) -> None:
        if int(max_allocations_per_user) < 1 or int(max_total_allocations) < 1:
            raise ValueError("allocation quotas must be >= 1")
        if advertised_ip is not None:
            advertised_ip = self._validate_relay_ip(advertised_ip)
        super().__init__(host, port)
        self.users = dict(users)
        self.realm = realm
        self.max_lifetime = max_lifetime
        self.channel_lifetime = int(channel_lifetime)
        self.max_allocations_per_user = int(max_allocations_per_user)
        self.max_total_allocations = int(max_total_allocations)
        self.allow_private_peers = bool(allow_private_peers)
        self.allow_loopback_peers = bool(allow_loopback_peers)
        self.nonce = secrets.token_hex(16).encode("ascii")
        self.allocations: Dict[Address, _Allocation] = {}
        self._by_relay: Dict[socket.socket, _Allocation] = {}
        self.channels: Dict[Tuple[Address, int], _Channel] = {}
        """Bound channels, keyed by `(allocation client, channel number)`.

        RFC 5766 §11 scopes channel numbers to one allocation, so two clients may both bind
        0x4000.
        """
        self._channel_by_peer: Dict[Tuple[Address, Address], int] = {}
        self.advertised_ip = advertised_ip
        self._responses: OrderedDict[
            Tuple[Address, int, bytes], Tuple[float, bytes, bytes]
        ] = OrderedDict()

    @staticmethod
    def _validate_relay_ip(value: str) -> str:
        ip = ipaddress.ip_address(value)
        if ip.version != 4 or ip.is_unspecified or ip.is_multicast or str(ip) == "255.255.255.255":
            raise ValueError("advertised relay IP must be a specific unicast IPv4 address")
        return str(ip)

    def _is_forbidden_peer_ip(self, ip: ipaddress.IPv4Address) -> bool:
        if ip.is_unspecified or ip.is_multicast or ip.is_link_local or str(ip) == "255.255.255.255":
            return True
        if not self.allow_loopback_peers and ip.is_loopback:
            return True
        if not self.allow_private_peers and any(ip in net for net in self._DENIED_PRIVATE_NETWORKS):
            return True
        return False

    def _sockets(self) -> List[socket.socket]:
        return [self.sock, *self._by_relay]

    def _close_sockets(self) -> None:
        for allocation in list(self.allocations.values()):
            allocation.relay.close()
        self.allocations.clear()
        self._by_relay.clear()
        self.channels.clear()
        self._channel_by_peer.clear()
        self._responses.clear()
        super()._close_sockets()

    def _housekeeping(self) -> None:
        now = time.monotonic()
        for client, allocation in list(self.allocations.items()):
            if allocation.expires <= now:
                self._release(client)
        for key, channel in list(self.channels.items()):
            if channel.expires <= now:
                self._release_channel(key)
        while self._responses and next(iter(self._responses.values()))[0] <= now:
            self._responses.popitem(last=False)

    def _release(self, client: Address) -> None:
        allocation = self.allocations.pop(client, None)
        if allocation is not None:
            self._by_relay.pop(allocation.relay, None)
            allocation.relay.close()
            for key, channel in list(self.channels.items()):
                if channel.allocation is allocation:
                    self._release_channel(key)

    def _release_channel(self, key: Tuple[Address, int]) -> None:
        channel = self.channels.pop(key, None)
        if channel is not None:
            self._channel_by_peer.pop((channel.allocation.client, channel.peer), None)

    def _live_channel(self, client: Address, number: int) -> Optional["_Channel"]:
        channel = self.channels.get((client, number))
        if channel is None:
            return None
        if channel.expires <= time.monotonic():
            self._release_channel((client, number))
            return None
        return channel

    def _on_readable(self, sock: socket.socket) -> None:
        if sock is self.sock:
            super()._on_readable(sock)
            return
        allocation = self._by_relay.get(sock)
        try:
            payload, peer = sock.recvfrom(65535)
        except OSError:
            return
        if allocation is None or not allocation.permits(peer[0]):
            return  # RFC 5766 §10.3: drop data from peers without a permission
        channel_number = self._channel_by_peer.get((allocation.client, peer))
        channel = (
            self._live_channel(allocation.client, channel_number)
            if channel_number is not None else None
        )
        if channel is not None:
            message = encode_channel_data(channel.number, payload)
        else:
            indication = StunMessage(DATA, INDICATION)
            indication.add(XOR_PEER_ADDRESS, encode_xor_address(peer, indication.transaction_id))
            indication.add(DATA_ATTR, payload)
            message = indication.encode(fingerprint=True)
        try:
            self.sock.sendto(message, allocation.client)
        except OSError:
            pass  # An oversized peer datagram or unreachable client must not stop the relay.

    def handle_datagram(self, data: bytes, addr: Address) -> None:
        if is_channel_data(data):
            self._on_channel_data(data, addr)
            return
        try:
            message = StunMessage.decode(data)
        except StunError:
            return
        if message.cls == REQUEST and message.method == BINDING:
            super().handle_datagram(data, addr)
        elif message.cls == INDICATION and message.method == SEND:
            self._on_send(message, addr)
        elif message.cls == REQUEST and message.method in (
            ALLOCATE, REFRESH, CREATE_PERMISSION, CHANNEL_BIND
        ):
            cache_key = (addr, message.method, message.transaction_id)
            cached = self._responses.get(cache_key)
            if cached is not None:
                expires, digest, response = cached
                if expires > time.monotonic():
                    # A transaction ID cannot authorize a different request. Replay only
                    # the exact original datagram, including its authentication bytes.
                    if hmac.compare_digest(digest, hashlib.sha256(data).digest()):
                        self.sock.sendto(response, addr)
                    return
                del self._responses[cache_key]
            self._on_authenticated_request(message, data, addr)

    @staticmethod
    def _error_response(request: StunMessage, code: int, reason: str) -> StunMessage:
        return StunMessage(request.method, ERROR, request.transaction_id).add(
            ERROR_CODE, encode_error_code(code, reason)
        )

    def _send_error(
        self, request: StunMessage, addr: Address, code: int, reason: str, challenge: bool = False
    ) -> None:
        response = self._error_response(request, code, reason)
        if challenge:
            response.add(REALM, self.realm.encode("utf-8"))
            response.add(NONCE, self.nonce)
        self.sock.sendto(response.encode(fingerprint=True), addr)

    def _on_authenticated_request(self, request: StunMessage, raw: bytes, addr: Address) -> None:
        username = request.get(USERNAME)
        if request.get(MESSAGE_INTEGRITY) is None:
            self._send_error(request, addr, 401, "Unauthorized", challenge=True)
            return
        if username is None or request.get(REALM) is None or request.get(NONCE) is None:
            self._send_error(request, addr, 400, "Bad Request")
            return
        try:
            user = username.decode("utf-8")
        except UnicodeDecodeError:
            self._send_error(request, addr, 401, "Unauthorized", challenge=True)
            return
        if user not in self.users or request.get(REALM) != self.realm.encode("utf-8"):
            self._send_error(request, addr, 401, "Unauthorized", challenge=True)
            return
        if request.get(NONCE) != self.nonce:
            self._send_error(request, addr, 438, "Stale Nonce", challenge=True)
            return
        key = long_term_key(user, self.realm, self.users[user])
        if not verify_message_integrity(raw, key):
            self._send_error(request, addr, 401, "Unauthorized", challenge=True)
            return
        handler = {
            ALLOCATE: self._allocate,
            REFRESH: self._refresh,
            CREATE_PERMISSION: self._create_permission,
            CHANNEL_BIND: self._channel_bind,
        }[request.method]
        response = handler(request, addr, user)
        if response is not None:
            response.add(SOFTWARE, SOFTWARE_NAME.encode("utf-8"))
            encoded = response.encode(integrity_key=key, fingerprint=True)
            cache_key = (addr, request.method, request.transaction_id)
            self._responses[cache_key] = (
                time.monotonic() + TRANSACTION_CACHE_LIFETIME_S, hashlib.sha256(raw).digest(), encoded
            )
            while len(self._responses) > TRANSACTION_CACHE_LIMIT:
                self._responses.popitem(last=False)
            self.sock.sendto(encoded, addr)

    def _requested_lifetime(self, request: StunMessage) -> int:
        value = request.get(LIFETIME)
        requested = struct.unpack("!I", value)[0] if value else 600
        return min(requested, self.max_lifetime)

    def _allocate(self, request: StunMessage, addr: Address, user: str) -> Optional[StunMessage]:
        self._housekeeping()
        if addr in self.allocations:
            return self._error_response(request, 437, "Allocation Mismatch")
        transport = request.get(REQUESTED_TRANSPORT)
        if transport is None or transport[0] != UDP_PROTOCOL_NUMBER:
            return self._error_response(request, 442, "Unsupported Transport Protocol")
        if len(self.allocations) >= self.max_total_allocations:
            return self._error_response(request, 508, "Insufficient Capacity")
        user_allocations = sum(1 for a in self.allocations.values() if a.username == user)
        if user_allocations >= self.max_allocations_per_user:
            return self._error_response(request, 486, "Allocation Quota Reached")
        lifetime = self._requested_lifetime(request)
        try:
            relay_ip = self.advertised_ip or self.address[0]
            if relay_ip == "0.0.0.0":
                relay_ip = outbound_interface_ip(addr)
            relay_ip = self._validate_relay_ip(relay_ip)
            relay = udp_socket((self.address[0], 0))
        except (OSError, ValueError):
            return self._error_response(request, 508, "Insufficient Capacity")
        allocation = _Allocation(addr, user, relay, time.monotonic() + lifetime)
        self.allocations[addr] = allocation
        self._by_relay[relay] = allocation
        response = StunMessage(ALLOCATE, SUCCESS, request.transaction_id)
        response.add(
            XOR_RELAYED_ADDRESS,
            encode_xor_address((relay_ip, relay.getsockname()[1]), request.transaction_id),
        )
        response.add(LIFETIME, struct.pack("!I", lifetime))
        response.add(XOR_MAPPED_ADDRESS, encode_xor_address(addr, request.transaction_id))
        return response

    def _refresh(self, request: StunMessage, addr: Address, user: str) -> Optional[StunMessage]:
        allocation = self.allocations.get(addr)
        if allocation is None:
            return self._error_response(request, 437, "Allocation Mismatch")
        if allocation.username != user:
            return self._error_response(request, 441, "Wrong Credentials")
        lifetime = self._requested_lifetime(request)
        if lifetime == 0:
            self._release(addr)
        else:
            allocation.expires = time.monotonic() + lifetime
        response = StunMessage(REFRESH, SUCCESS, request.transaction_id)
        response.add(LIFETIME, struct.pack("!I", lifetime))
        return response

    def _create_permission(
        self, request: StunMessage, addr: Address, user: str
    ) -> Optional[StunMessage]:
        allocation = self.allocations.get(addr)
        peer = request.get(XOR_PEER_ADDRESS)
        if allocation is None:
            return self._error_response(request, 437, "Allocation Mismatch")
        if allocation.username != user:
            return self._error_response(request, 441, "Wrong Credentials")
        if peer is None:
            return self._error_response(request, 400, "Bad Request")
        peer_ip = decode_xor_address(peer, request.transaction_id)[0]
        ip_obj = ipaddress.ip_address(peer_ip)
        if ip_obj.version != 4:
            return self._error_response(request, 443, "Peer Address Family Mismatch")
        if self._is_forbidden_peer_ip(ip_obj):
            return self._error_response(request, 403, "Forbidden")
        allocation.permit(peer_ip)
        return StunMessage(CREATE_PERMISSION, SUCCESS, request.transaction_id)

    def _channel_bind(self, request: StunMessage, addr: Address, user: str) -> Optional[StunMessage]:
        """Bind the client's channel number to a peer (RFC 5766 §11.2).

        The request must carry both CHANNEL-NUMBER (0x4000-0x7FFF) and XOR-PEER-ADDRESS.
        The number may not already be bound to another peer, nor the peer to another number,
        in this allocation; re-binding the same pair refreshes the channel. A successful bind
        also installs or refreshes the permission for the peer's IP address, and the success
        response carries no attributes.
        """
        allocation = self.allocations.get(addr)
        if allocation is None:
            return self._error_response(request, 437, "Allocation Mismatch")
        if allocation.username != user:
            return self._error_response(request, 441, "Wrong Credentials")
        peer = request.get(XOR_PEER_ADDRESS)
        number_attr = request.get(CHANNEL_NUMBER)
        if peer is None or number_attr is None:
            return self._error_response(request, 400, "Bad Request")
        try:
            number = decode_channel_number(number_attr)
        except StunError:
            return self._error_response(request, 400, "Bad Request")
        peer_addr = decode_xor_address(peer, request.transaction_id)
        ip_obj = ipaddress.ip_address(peer_addr[0])
        if ip_obj.version != 4:
            return self._error_response(request, 443, "Peer Address Family Mismatch")
        if self._is_forbidden_peer_ip(ip_obj):
            return self._error_response(request, 403, "Forbidden")
        by_number = self._live_channel(addr, number)
        if by_number is not None and by_number.peer != peer_addr:
            return self._error_response(request, 400, "Bad Request")
        peer_number = self._channel_by_peer.get((addr, peer_addr))
        if (
            peer_number is not None
            and peer_number != number
            and self._live_channel(addr, peer_number) is not None
        ):
            return self._error_response(request, 400, "Bad Request")
        lifetime = min(self.channel_lifetime, self.max_lifetime)
        self.channels[(addr, number)] = _Channel(
            number, peer_addr, allocation, time.monotonic() + lifetime
        )
        self._channel_by_peer[(addr, peer_addr)] = number
        allocation.permit(peer_addr[0])
        return StunMessage(CHANNEL_BIND, SUCCESS, request.transaction_id)

    def _on_channel_data(self, data: bytes, addr: Address) -> None:
        """Relay a ChannelData message to the peer the channel is bound to (RFC 5766 §11.4)."""
        try:
            number, payload = decode_channel_data(data)
        except StunError:
            return
        channel = self._live_channel(addr, number)
        if channel is None:
            return  # Unbound, expired, or not this client's channel.
        if not channel.allocation.permits(channel.peer[0]):
            return  # RFC 5766 §10.3: drop data from peers without a permission.
        try:
            channel.allocation.relay.sendto(payload, channel.peer)
        except OSError:
            pass

    def _on_send(self, indication: StunMessage, addr: Address) -> None:
        allocation = self.allocations.get(addr)
        peer = indication.get(XOR_PEER_ADDRESS)
        payload = indication.get(DATA_ATTR)
        if allocation is None or peer is None or payload is None:
            return
        peer_addr = decode_xor_address(peer, indication.transaction_id)
        if ipaddress.ip_address(peer_addr[0]).version == 4 and allocation.permits(peer_addr[0]):
            try:
                allocation.relay.sendto(payload, peer_addr)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# HM-S5-05: deterministic WAN fault injection (loss, reordering, jitter, rebinding)
# ---------------------------------------------------------------------------

MAX_HELD_PACKET_S = 0.05
"""How long a reordered datagram may be held before it is released anyway."""


@dataclass(frozen=True)
class FaultProfile:
    """A deterministic packet impairment applied by `DeterministicFaultProxy`.

    The profile fully describes the impairment, so a benchmark run is reproducible: with the
    same seed the proxy drops, delays and reorders the same datagrams in the same per-direction
    order. Each direction draws from its own PRNG stream, so the decision sequence never
    depends on how the two directions interleave in time.
    """

    loss_rate: float = 0.0
    reorder_rate: float = 0.0
    jitter_ms: float = 0.0
    seed: int = 0
    label: str = ""

    def __post_init__(self) -> None:
        for name, value in (("loss_rate", self.loss_rate), ("reorder_rate", self.reorder_rate)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a probability in [0, 1]")
        if isinstance(self.jitter_ms, bool) or not isinstance(self.jitter_ms, (int, float)):
            raise TypeError("jitter_ms must be a number")
        if not math.isfinite(self.jitter_ms) or self.jitter_ms < 0:
            raise ValueError("jitter_ms must be a finite, non-negative delay")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")

    def to_dict(self) -> Dict[str, object]:
        """Plain-dict view of the profile (JSON friendly)."""
        return {
            "label": self.label,
            "loss_rate": float(self.loss_rate),
            "reorder_rate": float(self.reorder_rate),
            "jitter_ms": float(self.jitter_ms),
            "seed": int(self.seed),
        }


@dataclass
class _HeldPacket:
    """A datagram held back so that the next one overtakes it."""

    data: bytes
    delay_ms: float
    deadline: float


class DeterministicFaultProxy:
    """A UDP relay that applies a deterministic loss/reorder/jitter profile to one path.

    The client dials `client_address`; the proxy forwards every datagram to the upstream
    server from a socket of its own and returns the replies. Impairments are drawn from a
    seeded PRNG per direction:

    - `loss_rate`: the datagram is dropped.
    - `reorder_rate`: the datagram is held back and released after the next datagram of the
      same direction, so the receiver observes the two swapped.
    - `jitter_ms`: an extra delay drawn uniformly from `[0, jitter_ms]`.

    `rebind()` models a NAT rebinding: the upstream socket is replaced by a socket with a new
    port, so the server observes a new client address while the previous mapping stops
    receiving replies, exactly like a rebinding NAT in front of the client.
    """

    def __init__(
        self,
        upstream: Address,
        profile: Optional[FaultProfile] = None,
        host: str = "127.0.0.1",
    ) -> None:
        self.upstream = upstream
        self.profile = profile if profile is not None else FaultProfile()
        self.host = host
        self.rebinds = 0
        self.dropped = 0
        self.reordered = 0
        self.forwarded = 0
        self.held = 0
        self._client_sock: Optional[socket.socket] = None
        self._upstream_sock: Optional[socket.socket] = None
        self._pending_upstream_sock: Optional[socket.socket] = None
        self._client_addr: Optional[Address] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._rng_up = random.Random(self.profile.seed)
        self._rng_down = random.Random(self.profile.seed ^ 0x5F3759DF)
        self._pending: List[Tuple[float, int, bytes, Address, socket.socket]] = []
        self._sequence = itertools.count()
        self._held: Dict[str, Optional[_HeldPacket]] = {"up": None, "down": None}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> "DeterministicFaultProxy":
        """Bind the sockets and start the relay thread."""
        if self._thread is not None:
            raise RuntimeError("fault proxy is already started")
        self._client_sock = udp_socket((self.host, 0))
        self._upstream_sock = udp_socket((self.host, 0))
        self._stop.clear()
        self._thread = threading.Thread(target=self._serve, name="FaultProxy", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        """Stop the relay thread and release the sockets."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)
        for attribute in ("_client_sock", "_upstream_sock", "_pending_upstream_sock"):
            sock = getattr(self, attribute)
            setattr(self, attribute, None)
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

    def __enter__(self) -> "DeterministicFaultProxy":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Addresses
    # ------------------------------------------------------------------
    @property
    def client_address(self) -> Address:
        """The address a client must dial to reach the upstream server through the proxy."""
        if self._client_sock is None:
            raise RuntimeError("fault proxy is not started")
        return self._client_sock.getsockname()

    @property
    def upstream_address(self) -> Address:
        """The address the upstream server sees as the client's (it changes on a rebinding)."""
        if self._upstream_sock is None:
            raise RuntimeError("fault proxy is not started")
        return self._upstream_sock.getsockname()

    def rebind(self) -> Address:
        """Simulate a NAT rebinding and return the new server-visible client address."""
        if self._client_sock is None:
            raise RuntimeError("fault proxy is not started")
        new_sock = udp_socket((self.host, 0))
        with self._lock:
            self._pending_upstream_sock = new_sock
            self.rebinds += 1
        return new_sock.getsockname()

    # ------------------------------------------------------------------
    # Relay loop
    # ------------------------------------------------------------------
    def _serve(self) -> None:
        while not self._stop.is_set():
            self._apply_pending_rebind()
            socks = [sock for sock in (self._client_sock, self._upstream_sock) if sock is not None]
            if not socks:
                time.sleep(0.005)
                continue
            try:
                ready, _, _ = select.select(socks, [], [], self._select_timeout())
            except (OSError, ValueError):
                continue
            for sock in ready:
                try:
                    data, addr = sock.recvfrom(65535)
                except OSError:
                    continue
                if sock is self._client_sock:
                    self._forward("up", data, addr)
                else:
                    self._forward("down", data, addr)
            self._flush()
            self._release_expired()

    def _apply_pending_rebind(self) -> None:
        with self._lock:
            new_sock = self._pending_upstream_sock
            if new_sock is None:
                return
            self._pending_upstream_sock = None
            old_sock = self._upstream_sock
            self._upstream_sock = new_sock
        if old_sock is not None:
            try:
                old_sock.close()
            except OSError:
                pass

    def _select_timeout(self) -> float:
        """Wait no longer than the next scheduled event (and never longer than 20 ms)."""
        now = time.monotonic()
        deadlines = [now + 0.02]
        with self._lock:
            if self._pending:
                deadlines.append(self._pending[0][0])
            for held in self._held.values():
                if held is not None:
                    deadlines.append(held.deadline)
        return max(0.0, min(deadlines) - now)

    def _decide(self, direction: str) -> Tuple[bool, bool, float]:
        """Draw the impairment of the next datagram: `(drop, reorder, delay_ms)`."""
        rng = self._rng_up if direction == "up" else self._rng_down
        drop = rng.random() < self.profile.loss_rate
        reorder = (not drop) and rng.random() < self.profile.reorder_rate
        delay = rng.uniform(0.0, self.profile.jitter_ms) if self.profile.jitter_ms > 0 else 0.0
        return drop, reorder, delay

    def _forward(self, direction: str, data: bytes, addr: Address) -> None:
        with self._lock:
            drop, reorder, delay_ms = self._decide(direction)
            if drop:
                self.dropped += 1
                return
            if direction == "up":
                out_sock, dest = self._upstream_sock, self.upstream
                self._client_addr = addr
            else:
                out_sock, dest = self._client_sock, self._client_addr
            if out_sock is None or dest is None:
                return
            held = self._held[direction]
            if held is not None:
                # The held datagram was overtaken: the current one goes out first and the
                # held one follows it immediately.
                self._held[direction] = None
                current_due = time.monotonic() + max(0.0, delay_ms) / 1000.0
                self._schedule(out_sock, dest, data, delay_ms, due=current_due)
                self._schedule(out_sock, dest, held.data, held.delay_ms, due=current_due + 1e-6)
                self.reordered += 1
                return
            if reorder:
                self._held[direction] = _HeldPacket(
                    data, delay_ms, time.monotonic() + MAX_HELD_PACKET_S
                )
                self.held += 1
                return
            self._schedule(out_sock, dest, data, delay_ms)

    def _schedule(
        self,
        sock: socket.socket,
        dest: Address,
        data: bytes,
        delay_ms: float,
        due: Optional[float] = None,
    ) -> None:
        if due is None:
            due = time.monotonic() + max(0.0, delay_ms) / 1000.0
        heapq.heappush(self._pending, (due, next(self._sequence), data, dest, sock))

    def _flush(self) -> None:
        now = time.monotonic()
        while self._pending and self._pending[0][0] <= now:
            _, _, data, dest, sock = heapq.heappop(self._pending)
            try:
                sock.sendto(data, dest)
                self.forwarded += 1
            except OSError:
                pass  # The socket was closed by a rebinding or by close().

    def _release_expired(self) -> None:
        now = time.monotonic()
        with self._lock:
            for direction, held in list(self._held.items()):
                if held is None or held.deadline > now:
                    continue
                self._held[direction] = None
                if direction == "up":
                    out_sock, dest = self._upstream_sock, self.upstream
                else:
                    out_sock, dest = self._client_sock, self._client_addr
                if out_sock is not None and dest is not None:
                    self._schedule(out_sock, dest, held.data, held.delay_ms)

    def stats(self) -> Dict[str, int]:
        """Counters describing what the proxy did (dropped, reordered and forwarded datagrams)."""
        return {
            "dropped": int(self.dropped),
            "reordered": int(self.reordered),
            "held": int(self.held),
            "forwarded": int(self.forwarded),
            "rebinds": int(self.rebinds),
        }
