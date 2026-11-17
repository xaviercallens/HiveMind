# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""A remote service's state is the result of a bounded probe, never a literal.

`/boinc/status` used to answer ``status: "ONLINE"`` for the remote Ruche without asking it (IP-177). The probe here
is a GET with a short timeout, cached for a few seconds so a page polling the router does not hammer the Ruche;
every answer carries the URL probed, the time and the latency, or the reason it could not be reached.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import httpx

PROBE_TIMEOUT_S = 2.5
PROBE_CACHE_S = 20.0
_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_LOCK = threading.Lock()


def probe_remote_status(url: str, *, timeout_s: float = PROBE_TIMEOUT_S, now_utc: float | None = None) -> dict[str, Any]:
    """GET ``url`` once; ONLINE on a 2xx answer, HTTP_<code> otherwise, UNREACHABLE on a transport error or timeout."""
    if not isinstance(url, str) or not url.strip().startswith(("http://", "https://")):
        raise ValueError("url must be a non-empty http/https URL")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(float(timeout_s)) or float(timeout_s) <= 0.0:
        raise ValueError("timeout_s must be a finite positive number")
    if now_utc is not None and (isinstance(now_utc, bool) or not isinstance(now_utc, (int, float)) or not math.isfinite(float(now_utc))):
        raise ValueError("now_utc must be a finite number when provided")
    now = time.time() if now_utc is None else float(now_utc)
    with _LOCK:
        hit = _CACHE.get(url)
        if hit is not None and now - hit[0] <= PROBE_CACHE_S:
            return dict(hit[1], probe_cached=True)
    started = time.perf_counter()
    try:
        resp = httpx.get(url, timeout=timeout_s, follow_redirects=True)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 1)
        # ONLINE only for a 2xx answer: a deleted Cloud Run service answers 404 and a login wall 3xx, neither is "serving" (IP-213).
        status = "ONLINE" if 200 <= resp.status_code < 300 else f"HTTP_{resp.status_code}"
        result: dict[str, Any] = {"status": status, "http_status": resp.status_code, "probe_latency_ms": latency_ms, "probe_error": None}
    except httpx.HTTPError as exc:
        latency_ms = round((time.perf_counter() - started) * 1000.0, 1)
        result = {"status": "UNREACHABLE", "http_status": None, "probe_latency_ms": latency_ms, "probe_error": type(exc).__name__}
    result.update({"probe_url": url, "probed_at_utc": round(now, 3), "probe_timeout_s": timeout_s, "probe_cached": False,
                   "status_basis": f"GET {url} with a {timeout_s} s timeout at {round(now, 3)}"})
    with _LOCK:
        _CACHE[url] = (now, result)
    return result
