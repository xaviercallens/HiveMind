# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Architecture-matched spot-check validator assignment (roadmap HM-S10-08).

The eps = 1e-4 replay rule of DeterministicSpotCheckValidator presumes worker and validator run the
same floating-point program (paper/hivemind_foundation_article.tex, Section 4.1, "architecture-matched").
This module assigns a validator of the same accelerator class as the worker, names the fallback when
none exists, and keeps a per-class-pair eps table. Experiment E12 (RQ11,
scripts/run_experiment_E12_replay_divergence.py) measures the divergence; from_e12_report() records
those numbers beside eps and NEVER changes eps itself. Cross-class eps is None (not comparable).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

DEVICE_CLASSES = frozenset({"cuda", "rocm", "mps", "cpu", "tpu"})
DEFAULT_EPS = 1e-4

ClassKey = tuple[str, "str | None", str, str]


def _major_minor(version: str) -> str:
    """'2.4.1+cu121' -> '2.4'."""
    if not isinstance(version, str) or not version.strip():
        raise ValueError("version must be a non-empty string")
    parts = version.split("+")[0].split(".")
    return ".".join(parts[:2])


@dataclass(frozen=True)
class AcceleratorFingerprint:
    """Hardware/software identity relevant to floating-point replay equality."""

    device_class: str
    device_name: str
    compute_capability: str | None
    torch_version: str  # major.minor
    accel_runtime: str | None  # CUDA / ROCm / Metal runtime version, if any
    dtype: str

    def __post_init__(self) -> None:
        if self.device_class not in DEVICE_CLASSES:
            raise ValueError(f"device_class must be one of {sorted(DEVICE_CLASSES)}, got {self.device_class!r}")
        for fname in ("device_name", "torch_version", "dtype"):
            val = getattr(self, fname)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{fname} must be a non-empty string")
        for opt_name in ("compute_capability", "accel_runtime"):
            val = getattr(self, opt_name)
            if val is not None and (not isinstance(val, str) or not val.strip()):
                raise ValueError(f"{opt_name} must be a non-empty string or None")

    def class_key(self) -> ClassKey:
        return (self.device_class, self.compute_capability, self.torch_version, self.dtype)

    @classmethod
    def from_local(cls, dtype: str = "float32") -> "AcceleratorFingerprint":
        """Fingerprint of this host; torch is imported lazily and optional."""
        if not isinstance(dtype, str) or not dtype.strip():
            raise ValueError("dtype must be a non-empty string")
        try:
            import torch
        except ImportError:
            return cls("cpu", "unknown", None, "none", None, dtype)
        tv = _major_minor(torch.__version__)
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            hip = getattr(torch.version, "hip", None)
            return cls(
                "rocm" if hip else "cuda",
                torch.cuda.get_device_name(0),
                f"{major}.{minor}",
                tv,
                hip or torch.version.cuda,
                dtype,
            )
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return cls("mps", "apple-mps", None, tv, None, dtype)
        return cls("cpu", "cpu", None, tv, None, dtype)


class FallbackPolicy(str, Enum):
    """Fallbacks when no architecture-matched validator exists (paper Section 4.1)."""

    REJECT_UNMATCHED = "reject_unmatched"
    FIXED_POINT_REPLAY = "fixed_point_replay"
    LOSS_TRAJECTORY_CHECK = "loss_trajectory_check"


class ValidatorRegistry:
    """Registered validators by fingerprint; deterministic matched assignment."""

    def __init__(self, default_fallback: FallbackPolicy = FallbackPolicy.REJECT_UNMATCHED) -> None:
        if not isinstance(default_fallback, FallbackPolicy):
            raise ValueError("default_fallback must be a FallbackPolicy")
        self.default_fallback = default_fallback
        self._validators: dict[str, AcceleratorFingerprint] = {}

    def register(self, validator_id: str, fingerprint: AcceleratorFingerprint) -> None:
        if not isinstance(validator_id, str) or not validator_id.strip():
            raise ValueError("validator_id must be a non-empty string")
        if not isinstance(fingerprint, AcceleratorFingerprint):
            raise TypeError("fingerprint must be an AcceleratorFingerprint")
        self._validators[validator_id] = fingerprint

    def assign(
        self,
        worker_fp: AcceleratorFingerprint,
        rng_seed: int,
        exclude: Iterable[str] = (),
    ) -> tuple[str | None, FallbackPolicy | None, str]:
        """Return (validator_id, policy, reason). policy is None on an exact match."""
        if not isinstance(worker_fp, AcceleratorFingerprint):
            raise TypeError("worker_fp must be an AcceleratorFingerprint")
        if isinstance(rng_seed, bool) or not isinstance(rng_seed, int):
            raise ValueError("rng_seed must be an integer")
        if isinstance(exclude, (str, bytes)):
            raise ValueError("exclude must be an iterable of validator_id strings, not str/bytes")
        key = worker_fp.class_key()
        skip = set(exclude)
        matches = sorted(v for v, fp in self._validators.items() if fp.class_key() == key and v not in skip)
        if matches:
            chosen = random.Random(rng_seed).choice(matches)  # sorted -> deterministic
            return chosen, None, f"exact class match {key} ({len(matches)} candidate(s))"
        return None, self.default_fallback, f"no validator with class key {key}; fallback {self.default_fallback.value}"


def _validate_class_key(key: Any, name: str = "class_key") -> ClassKey:
    if not isinstance(key, tuple) or len(key) != 4:
        raise ValueError(f"{name} must be a 4-tuple (device_class, compute_capability, torch_version, dtype)")
    dev_cls, cc, tv, dt = key
    if dev_cls not in DEVICE_CLASSES:
        raise ValueError(f"{name}[0] (device_class) must be one of {sorted(DEVICE_CLASSES)}")
    if cc is not None and (not isinstance(cc, str) or not cc.strip()):
        raise ValueError(f"{name}[1] (compute_capability) must be None or a non-empty string")
    if not isinstance(tv, str) or not tv.strip():
        raise ValueError(f"{name}[2] (torch_version) must be a non-empty string")
    if not isinstance(dt, str) or not dt.strip():
        raise ValueError(f"{name}[3] (dtype) must be a non-empty string")
    return key


class EpsilonTable:
    """Per class-pair eps. Same class: 1e-4 default; cross class: None (not comparable)."""

    def __init__(self, same_class_eps: float = DEFAULT_EPS) -> None:
        if (
            isinstance(same_class_eps, bool)
            or not isinstance(same_class_eps, (int, float))
            or not math.isfinite(float(same_class_eps))
            or float(same_class_eps) <= 0.0
        ):
            raise ValueError("same_class_eps must be a positive finite number")
        self.same_class_eps = float(same_class_eps)
        self._overrides: dict[tuple[ClassKey, ClassKey], float | None] = {}
        self.measured: dict[str, dict[str, Any]] = {}  # E12 cell -> measurement, stored beside eps
        self.flags: list[str] = []

    def set_eps(self, a: ClassKey, b: ClassKey, eps: float | None) -> None:
        _validate_class_key(a, "a")
        _validate_class_key(b, "b")
        if eps is not None and (
            isinstance(eps, bool)
            or not isinstance(eps, (int, float))
            or not math.isfinite(float(eps))
            or float(eps) <= 0.0
        ):
            raise ValueError("eps must be None or a positive finite number")
        self._overrides[tuple(sorted((a, b), key=repr))] = float(eps) if eps is not None else None  # type: ignore[assignment]

    def eps_for(self, a: ClassKey, b: ClassKey) -> float | None:
        _validate_class_key(a, "a")
        _validate_class_key(b, "b")
        pair = tuple(sorted((a, b), key=repr))
        if pair in self._overrides:
            return self._overrides[pair]  # type: ignore[index]
        return self.same_class_eps if a == b else None

    @classmethod
    def from_e12_report(cls, path: str | Path) -> "EpsilonTable":
        """Record E12 measured divergences; eps stays at its default (report eps is noted, not applied)."""
        if not isinstance(path, (str, Path)) or not str(path).strip():
            raise ValueError("path must be a non-empty Path or string")
        table = cls()
        p = Path(path)
        if not p.exists():
            return table
        report = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError("E12 report must be a JSON object")
        eps = report.get("eps", DEFAULT_EPS)
        kinds = {
            "A_same_process_T4": "same_class",
            "B_separate_process_T4": "same_class",
            "C_cross_T4_vs_cpu": "cross_class",
            "D_proxy_fp64_snap_T4_vs_cpu": "cross_class",
            "E_tamper": "tamper",
        }
        for cell, s in (report.get("summary") or {}).items():
            table.measured[cell] = {"kind": kinds.get(cell, "unknown"), "max_linf": s.get("max_linf"), "n": s.get("n")}
            if kinds.get(cell) == "same_class" and s.get("max_linf") is not None and s["max_linf"] > eps:
                table.flags.append(
                    f"{cell}: measured same-class max_linf {s['max_linf']:.3e} exceeds eps {eps:.1e}; "
                    "eps NOT changed, review required"
                )
        return table
