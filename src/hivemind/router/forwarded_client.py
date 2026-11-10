# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Signed forwarding of the end user's IP from the web portal to the gateway (v3.2).

Every portal call reaches the gateway from the portal's own egress address, so without this all visitors would share one
rate-limit bucket and one attacker could lock everybody out. The portal adds ``X-HiveMind-Forwarded-Client:
<ip>|<unix seconds>|<hex HMAC-SHA256>``, keyed with the secret both services already share (``HIVEMIND_ROUTER_SECRET``). The
gateway uses the address only when the signature verifies and the timestamp is within ``MAX_SKEW_S``; a forged or stale
header is ignored, and the TCP peer is used as before.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import math

HEADER = "x-hivemind-forwarded-client"
MAX_SKEW_S = 60.0
_DOMAIN = b"hivemind-forwarded-client-v1:"


def _mac(secret: bytes, ip: str, ts: int) -> str:
    if not isinstance(secret, (bytes, bytearray)) or not secret:
        raise ValueError("secret must be non-empty bytes")
    if not isinstance(ip, str) or not ip.strip() or "|" in ip:
        raise ValueError("ip must be a non-empty string without '|'")
    if isinstance(ts, bool) or not isinstance(ts, int) or ts < 0:
        raise ValueError("ts must be a non-negative integer")
    return hmac.new(bytes(secret), _DOMAIN + f"{ip}|{ts}".encode("ascii"), hashlib.sha256).hexdigest()


def sign(secret: bytes, ip: str, now: float) -> str:
    if not isinstance(secret, bytes) or not secret:
        raise ValueError("secret must be non-empty bytes")
    if not isinstance(ip, str) or not ip.strip() or "|" in ip:
        raise ValueError("ip must be a non-empty string without '|'")
    ip_clean = ip.strip()
    if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(float(now)) or float(now) < 0.0:
        raise ValueError("now must be a finite non-negative number")
    ts = int(now)
    return f"{ip_clean}|{ts}|{_mac(secret, ip_clean, ts)}"


def verify(secret: bytes, header: str, now: float) -> str | None:
    """The forwarded IP if ``header`` is authentic and fresh, else ``None``."""
    if not isinstance(secret, bytes) or not secret or not isinstance(header, str):
        return None
    if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(float(now)):
        return None
    try:
        ip, ts_text, mac = header.strip().split("|")
        ipaddress.ip_address(ip)
        ts = int(ts_text)
    except ValueError:
        return None
    if abs(now - ts) > MAX_SKEW_S:
        return None
    return ip if hmac.compare_digest(mac, _mac(secret, ip, ts)) else None
