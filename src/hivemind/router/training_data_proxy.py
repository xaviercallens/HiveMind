# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Training rows through the gateway (owner decision 2026-11-17, option 2).

Volunteers may read public training rows either straight from Hugging Face's datasets-server (option 1) or through
the gateway (this module). The gateway path uses ONE dedicated, shared Hugging Face account: its token is read from
``HIVEMIND_HF_TOKEN`` (set from Secret Manager by the owner) and is never returned or logged. Without a token the
gateway fetches anonymously, like option 1.

Only datasets of the training plan (configs/datasets_s21.json accepted sources) and the licensed local extras are
served; benchmark sets on the decontamination list never are. ``tests/test_training_data_proxy.py`` checks that this
list equals the plan's sources plus the local extras, minus the benchmarks.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

DATASETS_SERVER_ROWS = "https://datasets-server.huggingface.co/rows"
TRAINING_DATASETS: frozenset[str] = frozenset({
    "AI-MO/NuminaMath-CoT",
    "CyberNative/Code_Vulnerability_Security_DPO",
    "HuggingFaceFW/fineweb-edu",
    "OneAstronomy/sspp-dr9-stellar-spectra",
    "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset",
    "bigcode/self-oss-instruct-sc2-exec-filter-50k",
    "bingbangboom/exoplanet-transit-detection",
    "ise-uiuc/Magicoder-OSS-Instruct-75K",
    "jeff-vincent/astronomy-textbook-qa-context",
    "juliensimon/nasa-exoplanets",
    "maximus-21/PhysicsQA",
    "meta-math/MetaMathQA",
    "nuhmanpk/cybersecurity-controls-instructions",
    "nvidia/OpenCodeInstruct",
    "nvidia/OpenMathInstruct-2",
    "sahil2801/CodeAlpaca-20k",
})
_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
MAX_LENGTH = 100  # the datasets-server maximum per request
MAX_OFFSET = 50_000_000


class TrainingDataError(ValueError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _urllib_fetch(url: str, headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 - fixed https host
        return json.loads(resp.read().decode("utf-8"))


class TrainingRowsProxy:
    """Validated, cached passthrough to the datasets-server, authenticated with the dedicated account when configured."""

    def __init__(self, token: str | None = None, *, fetch: Callable[[str, dict[str, str], float], dict[str, Any]] = _urllib_fetch,
                 cache_entries: int = 512, ttl_s: float = 6 * 3600.0, clock: Callable[[], float] = time.monotonic):
        self._token = (token if token is not None else os.environ.get("HIVEMIND_HF_TOKEN", "")).strip()
        self._fetch = fetch
        self._cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._max = int(cache_entries)
        self._ttl = float(ttl_s)
        self._clock = clock
        self._lock = threading.Lock()

    @property
    def authenticated(self) -> bool:
        return bool(self._token)

    def rows(self, dataset: str, config: str, split: str, offset: int, length: int) -> dict[str, Any]:
        if dataset not in TRAINING_DATASETS:
            raise TrainingDataError(404, "dataset is not in the training plan (benchmarks are never served)")
        if not _NAME_RE.match(config or "") or not _NAME_RE.match(split or ""):
            raise TrainingDataError(400, "config and split must be 1-64 characters of letters, digits, '_', '.', '-'")
        if not (0 <= int(offset) <= MAX_OFFSET) or not (1 <= int(length) <= MAX_LENGTH):
            raise TrainingDataError(400, f"offset must be 0..{MAX_OFFSET} and length 1..{MAX_LENGTH}")
        key = f"{dataset}|{config}|{split}|{int(offset)}|{int(length)}"
        now = self._clock()
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None and now - hit[0] <= self._ttl:
                self._cache.move_to_end(key)
                return {**hit[1], "cached": True}
        q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": int(offset), "length": int(length)})
        headers = {"User-Agent": "HiveMind-Gateway-TrainingRows/1", "Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            body = self._fetch(f"{DATASETS_SERVER_ROWS}?{q}", headers, 30.0)
        except Exception as exc:  # noqa: BLE001 - upstream errors become a 502 without leaking headers
            raise TrainingDataError(502, f"upstream datasets-server error: {type(exc).__name__}") from None
        if not isinstance(body, dict) or not isinstance(body.get("rows"), list):
            raise TrainingDataError(502, "unexpected datasets-server answer")
        out = {"dataset": dataset, "config": config, "split": split, "offset": int(offset),
               "rows": [r for r in body["rows"] if isinstance(r, dict) and isinstance(r.get("row"), dict)],
               "num_rows_total": body.get("num_rows_total"), "source": "hivemind_gateway_proxy",
               "upstream_authenticated": self.authenticated, "cached": False}
        with self._lock:
            self._cache[key] = (now, out)
            self._cache.move_to_end(key)
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)
        return out
