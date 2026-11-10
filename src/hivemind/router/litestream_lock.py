# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Single-writer Litestream replica lock with generation preconditions (HM-S16-10 / #197).

When the router ledger is replicated to a single GCS path via Litestream, starting a second revision or canary with
the same replica URL risks two concurrent SQLite writers. ``LitestreamWriterLock`` enforces single-writer exclusivity
using an atomic generation precondition (``if_generation_match=0`` on creation, ``if_generation_match=<held>`` on
renewal or release). A second instance that cannot acquire the lock either refuses to start or enters read-only mode
so only the lock holder can mutate the ledger.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union


class WriterLockBusyError(RuntimeError):
    """Raised when a second instance attempts to acquire or write under a held Litestream replica lock."""


@dataclass(frozen=True)
class LockGenerationRecord:
    replica_url: str
    owner_id: str
    generation: int
    acquired_utc: float
    lease_expires_utc: float

    def __post_init__(self) -> None:
        for field_name in ("replica_url", "owner_id"):
            val = getattr(self, field_name)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if isinstance(self.generation, bool) or not isinstance(self.generation, int) or self.generation < 0:
            raise ValueError("generation must be a non-negative integer")
        for field_name in ("acquired_utc", "lease_expires_utc"):
            val = getattr(self, field_name)
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)) or float(val) < 0.0:
                raise ValueError(f"{field_name} must be a finite non-negative number")
        if float(self.lease_expires_utc) < float(self.acquired_utc):
            raise ValueError("lease_expires_utc must be >= acquired_utc")


class InMemoryGenerationStore:
    """Thread-safe atomic generation-precondition store modeling GCS ``ifGenerationMatch`` semantics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._objects: dict[str, tuple[int, dict[str, Any]]] = {}
        self._next_gen = 1

    def put_if_generation_match(
        self,
        key: str,
        payload: dict[str, Any],
        *,
        if_generation_match: int,
    ) -> int:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("key must be a non-empty string")
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dict")
        if isinstance(if_generation_match, bool) or not isinstance(if_generation_match, int) or if_generation_match < 0:
            raise ValueError("if_generation_match must be a non-negative integer")
        with self._lock:
            current = self._objects.get(key)
            current_gen = 0 if current is None else current[0]
            if current_gen != int(if_generation_match):
                raise WriterLockBusyError(
                    f"generation precondition failed for {key!r}: expected {if_generation_match}, actual {current_gen}"
                )
            new_gen = self._next_gen
            self._next_gen += 1
            self._objects[key] = (new_gen, dict(payload))
            return new_gen

    def get_record(self, key: str) -> tuple[int, dict[str, Any]] | None:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("key must be a non-empty string")
        with self._lock:
            entry = self._objects.get(key)
            if entry is None:
                return None
            return entry[0], dict(entry[1])

    def delete_if_generation_match(self, key: str, *, if_generation_match: int) -> bool:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("key must be a non-empty string")
        if isinstance(if_generation_match, bool) or not isinstance(if_generation_match, int) or if_generation_match < 0:
            raise ValueError("if_generation_match must be a non-negative integer")
        with self._lock:
            current = self._objects.get(key)
            current_gen = 0 if current is None else current[0]
            if current_gen != int(if_generation_match):
                raise WriterLockBusyError(
                    f"delete precondition failed for {key!r}: expected {if_generation_match}, actual {current_gen}"
                )
            if current is not None:
                del self._objects[key]
                return True
            return False


class FileGenerationStore:
    """Filesystem-backed generation-precondition store for single-host / shared-volume Litestream locks."""

    def __init__(self, root_dir: Path | str) -> None:
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path_for(self, key: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in key)
        return self._root / f"{safe}.lock.json"

    def put_if_generation_match(
        self,
        key: str,
        payload: dict[str, Any],
        *,
        if_generation_match: int,
    ) -> int:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("key must be a non-empty string")
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dict")
        if isinstance(if_generation_match, bool) or not isinstance(if_generation_match, int) or if_generation_match < 0:
            raise ValueError("if_generation_match must be a non-negative integer")
        path = self._path_for(key)
        with self._lock:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                current_gen = int(data.get("generation", 0))
            else:
                current_gen = 0
            if current_gen != int(if_generation_match):
                raise WriterLockBusyError(
                    f"generation precondition failed for {key!r}: expected {if_generation_match}, actual {current_gen}"
                )
            new_gen = current_gen + 1
            doc = {"generation": new_gen, "payload": dict(payload)}
            tmp = path.with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}")
            tmp.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
            tmp.replace(path)
            return new_gen

    def get_record(self, key: str) -> tuple[int, dict[str, Any]] | None:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("key must be a non-empty string")
        path = self._path_for(key)
        with self._lock:
            if not path.exists():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            return int(data["generation"]), dict(data.get("payload") or {})

    def delete_if_generation_match(self, key: str, *, if_generation_match: int) -> bool:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("key must be a non-empty string")
        if isinstance(if_generation_match, bool) or not isinstance(if_generation_match, int) or if_generation_match < 0:
            raise ValueError("if_generation_match must be a non-negative integer")
        path = self._path_for(key)
        with self._lock:
            if not path.exists():
                current_gen = 0
            else:
                data = json.loads(path.read_text(encoding="utf-8"))
                current_gen = int(data.get("generation", 0))
            if current_gen != int(if_generation_match):
                raise WriterLockBusyError(
                    f"delete precondition failed for {key!r}: expected {if_generation_match}, actual {current_gen}"
                )
            if path.exists():
                path.unlink()
                return True
            return False


GenerationBackend = Union[InMemoryGenerationStore, FileGenerationStore]


class LitestreamWriterLock:
    """Enforces single-writer ownership for one Litestream replica URL."""

    def __init__(
        self,
        replica_url: str,
        owner_id: str,
        backend: GenerationBackend,
        *,
        lease_ttl_s: float = 300.0,
        fail_closed: bool = False,
    ) -> None:
        if not isinstance(replica_url, str):
            raise ValueError("replica_url must be a string")
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("owner_id must be a non-empty string")
        if isinstance(lease_ttl_s, bool) or not isinstance(lease_ttl_s, (int, float)) or not math.isfinite(float(lease_ttl_s)) or float(lease_ttl_s) <= 0.0:
            raise ValueError("lease_ttl_s must be a finite positive number")
        if not isinstance(fail_closed, bool):
            raise ValueError("fail_closed must be a boolean")
        self.replica_url = str(replica_url).strip()
        self.owner_id = owner_id.strip()
        self._backend = backend
        self.lease_ttl_s = max(1.0, float(lease_ttl_s))
        self.fail_closed = bool(fail_closed)
        self._generation: int | None = None
        self._read_only = True

    @property
    def lock_key(self) -> str:
        return f"litestream_writer_lock::{self.replica_url}"

    @property
    def is_writer(self) -> bool:
        return self._generation is not None and not self._read_only

    @property
    def read_only(self) -> bool:
        return self._read_only

    @property
    def generation(self) -> int | None:
        return self._generation

    def acquire(self, now_utc: float | None = None) -> bool:
        """Attempt to claim single-writer ownership via ``if_generation_match=0`` (or expired lease takeover)."""
        if now_utc is not None and (isinstance(now_utc, bool) or not isinstance(now_utc, (int, float)) or not math.isfinite(float(now_utc))):
            raise ValueError("now_utc must be a finite number when provided")
        if not self.replica_url:
            # Ephemeral canary with empty replica URL does not contend on a shared replica
            self._generation = 0
            self._read_only = False
            return True
        now = time.time() if now_utc is None else float(now_utc)
        existing = self._backend.get_record(self.lock_key)
        expected_gen = 0
        if existing is not None:
            held_gen, meta = existing
            if str(meta.get("owner_id")) == self.owner_id:
                expected_gen = held_gen
            elif float(meta.get("lease_expires_utc", 0.0)) <= now:
                expected_gen = held_gen
        payload = {
            "replica_url": self.replica_url,
            "owner_id": self.owner_id,
            "acquired_utc": now,
            "lease_expires_utc": now + self.lease_ttl_s,
        }
        try:
            gen = self._backend.put_if_generation_match(
                self.lock_key,
                payload,
                if_generation_match=expected_gen,
            )
            self._generation = gen
            self._read_only = False
            return True
        except WriterLockBusyError:
            self._generation = None
            self._read_only = True
            if self.fail_closed:
                raise
            return False

    def assert_writable(self) -> None:
        """Raise ``WriterLockBusyError`` if this instance does not hold the single-writer lock."""
        if not self.is_writer:
            raise WriterLockBusyError(
                f"RouterStore instance {self.owner_id!r} is read-only: replica {self.replica_url!r} is locked by another writer"
            )

    def release(self) -> bool:
        if not self.replica_url or self._generation is None:
            self._generation = None
            self._read_only = True
            return False
        gen = self._generation
        self._generation = None
        self._read_only = True
        return self._backend.delete_if_generation_match(self.lock_key, if_generation_match=gen)
