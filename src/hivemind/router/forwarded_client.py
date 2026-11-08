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

HEADER = "x-hivemind-forwarded-client"
MAX_SKEW_S = 60.0
_DOMAIN = b"hivemind-forwarded-client-v1:"


def _mac(secret: bytes, ip: str, ts: int) -> str:
    return hmac.new(secret, _DOMAIN + f"{ip}|{ts}".encode("ascii"), hashlib.sha256).hexdigest()


def sign(secret: bytes, ip: str, now: float) -> str:
    ts = int(now)
    return f"{ip}|{ts}|{_mac(secret, ip, ts)}"


def verify(secret: bytes, header: str, now: float) -> str | None:
    """The forwarded IP if ``header`` is authentic and fresh, else ``None``."""
    try:
        ip, ts_text, mac = header.strip().split("|")
        ipaddress.ip_address(ip)
        ts = int(ts_text)
    except ValueError:
        return None
    if abs(now - ts) > MAX_SKEW_S:
        return None
    return ip if hmac.compare_digest(mac, _mac(secret, ip, ts)) else None
