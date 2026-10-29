# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Accounts, API keys and the HMT (HiveMind points) ledger for the HiveMind OpenRouter, backed by SQLite.

Design notes
------------
* API keys are ``"hm-" + token_urlsafe(32)`` and are only ever stored as ``HMAC-SHA256(secret, key)``.
* ``account_id`` is ``HMAC-SHA256(secret, normalise_identity(handle))[:32]``.
* The ledger is append-only and hash-chained: ``entry_hash = sha256(prev_hash | canonical_json(row))``.
  HMT is created only by :meth:`RouterStore.grant`; :meth:`RouterStore.settle` moves it between accounts, so
  ``sum(balances) == total_minted()`` always holds (the producer share is the remainder of the charge, which makes
  the split conserve exactly in floating point up to rounding of the subtraction).
* Every mutation runs in ``BEGIN IMMEDIATE`` under a process-wide :class:`threading.Lock`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import secrets
import sqlite3
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from hivemind.security.hivemind_firewall import normalise_identity

ROLES: tuple[str, ...] = ("consumer", "producer", "prosumer", "admin")
PRODUCER_ROLES: tuple[str, ...] = ("producer", "prosumer")
SYS_VALIDATORS = "sys:validators"
SYS_BURN = "sys:burn"
SYS_ESCROW = "sys:escrow"
SYSTEM_ACCOUNTS: tuple[str, ...] = (SYS_VALIDATORS, SYS_BURN)
_ESCROW_ACCOUNTS: tuple[str, ...] = (*SYSTEM_ACCOUNTS, SYS_ESCROW)
HOLD_TTL_S = 900.0
MAX_INFLIGHT_HOLDS = 16
DEFAULT_MAX_PRICE_PER_1K = 2.0
GENESIS_HASH = "0" * 64
MIN_SECRET_BYTES = 32

DEFAULT_VALIDATOR_SHARE = 0.10
DEFAULT_BURN_SHARE = 0.05

_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts(
    account_id TEXT PRIMARY KEY,
    handle TEXT NOT NULL UNIQUE,
    role TEXT NOT NULL CHECK (role IN ('consumer','producer','prosumer','admin')),
    key_hash TEXT NOT NULL UNIQUE,
    created_utc REAL NOT NULL,
    max_price_per_1k REAL NOT NULL,
    is_test INTEGER NOT NULL DEFAULT 0,
    kind TEXT NOT NULL DEFAULT 'user'
);
CREATE TABLE IF NOT EXISTS starter_grant_ips(
    ip_hash TEXT NOT NULL,
    utc REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS starter_grant_ips_hash ON starter_grant_ips(ip_hash, utc);
CREATE TABLE IF NOT EXISTS ledger(
    seq INTEGER PRIMARY KEY,
    account_id TEXT NOT NULL,
    delta REAL NOT NULL,
    kind TEXT NOT NULL,
    ref TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    entry_hash TEXT NOT NULL,
    utc REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ledger_account ON ledger(account_id);
CREATE TABLE IF NOT EXISTS producers(
    account_id TEXT PRIMARY KEY,
    endpoint_url TEXT NOT NULL,
    models_json TEXT NOT NULL,
    cost_per_1k REAL NOT NULL,
    capacity_1k REAL NOT NULL,
    last_heartbeat_utc REAL NOT NULL,
    execution_class TEXT NOT NULL,
    endpoint_token TEXT
);
CREATE TABLE IF NOT EXISTS producer_endpoints(
    account_id TEXT NOT NULL,
    endpoint_url TEXT NOT NULL,
    models_json TEXT NOT NULL,
    cost_per_1k REAL NOT NULL,
    capacity_1k REAL NOT NULL,
    last_heartbeat_utc REAL NOT NULL,
    execution_class TEXT NOT NULL,
    endpoint_token TEXT,
    residency_json TEXT,
    PRIMARY KEY(account_id, endpoint_url)
);
CREATE TABLE IF NOT EXISTS producer_control(
    account_id TEXT PRIMARY KEY,
    paused INTEGER NOT NULL DEFAULT 0,
    vram_cap_mib INTEGER,
    updated_at REAL NOT NULL,
    version INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS usage(
    request_id TEXT PRIMARY KEY,
    consumer_id TEXT NOT NULL,
    producer_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    billed_tokens INTEGER NOT NULL,
    price_per_1k REAL NOT NULL,
    charge REAL NOT NULL,
    utc REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS holds(
    request_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    amount REAL NOT NULL,
    utc REAL NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('active','released'))
);
CREATE INDEX IF NOT EXISTS holds_account ON holds(account_id, state);
CREATE TABLE IF NOT EXISTS usage_telemetry(
    request_id TEXT PRIMARY KEY,
    consumer_id TEXT NOT NULL,
    producer_id TEXT NOT NULL,
    requested_model TEXT NOT NULL,
    routed_model TEXT NOT NULL,
    routing_mode TEXT NOT NULL,
    task_domain TEXT NOT NULL,
    language TEXT NOT NULL,
    client_sdk TEXT NOT NULL,
    app_title TEXT NOT NULL,
    is_stream INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    billed_tokens INTEGER NOT NULL,
    charged_hmt REAL NOT NULL,
    utc REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_telemetry_utc ON usage_telemetry(utc);
CREATE TABLE IF NOT EXISTS usage_feedback(
    request_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),
    task_domain TEXT NOT NULL,
    comment TEXT NOT NULL DEFAULT '',
    utc REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS campaign_poll_votes(
    account_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    participant_handle TEXT NOT NULL,
    participant_kind TEXT NOT NULL,
    pledged_vram_gb REAL NOT NULL DEFAULT 0.0,
    availability_slot TEXT NOT NULL DEFAULT 'eu_evening_cet',
    comment TEXT NOT NULL DEFAULT '',
    utc REAL NOT NULL,
    PRIMARY KEY(account_id, subject_id)
);
CREATE TABLE IF NOT EXISTS amcp_kafka_events(
    event_id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    event_type TEXT NOT NULL,
    sender_id TEXT NOT NULL,
    sender_kind TEXT NOT NULL,
    recipient_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    correlation_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    utc REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS amcp_kafka_events_topic_utc ON amcp_kafka_events(topic, utc);
CREATE TABLE IF NOT EXISTS bee_trade_orders(
    order_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    bee_id TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('OFFER', 'BID')),
    subject_id TEXT NOT NULL,
    capacity_1k REAL NOT NULL,
    price_per_1k REAL NOT NULL,
    utc REAL NOT NULL
);
"""


KIND_USER = "user"
ACCOUNT_KINDS = (KIND_USER, "test", "probe")
STARTER_WINDOW_S = 86_400.0


class InsufficientBalance(Exception):
    """Raised by :meth:`RouterStore.settle` when the consumer cannot cover the charge (nothing is written)."""


class TooManyHolds(Exception):
    """Raised by :meth:`RouterStore.hold` when the account already has the maximum number of in-flight holds."""


class DuplicateRequest(ValueError):
    """Raised when a ``request_id`` has already been settled."""


class TooManyEndpoints(ValueError):
    """Raised by :meth:`RouterStore.upsert_producer` when an account already advertises ``MAX_ENDPOINTS_PER_ACCOUNT`` endpoints."""


MAX_ENDPOINTS_PER_ACCOUNT = 8  # one account may run several GPUs / servers; bounded so a heartbeat flood cannot grow the table without limit
STALE_ENDPOINT_PURGE_S = 86_400.0  # an endpoint silent for a day is forgotten (it left the pool long ago) and frees its slot


# Handles of the form ``<provider>.<keyed hash>`` belong to identity-provider sign-ins; nobody can register one by hand,
# so a Google account cannot be squatted by typing its handle into the open registration form.
IDENTITY_PROVIDERS: tuple[str, ...] = ("google",)  # always reserved; a store can enable more with ``allow_identity_provider``
RESERVED_HANDLE_PREFIXES: tuple[str, ...] = tuple(f"{provider}." for provider in IDENTITY_PROVIDERS)
IDENTITY_PROVIDER_NAME_RE = re.compile(r"^[a-z][a-z0-9]{1,15}$")


class ReservedHandle(ValueError):
    """The handle belongs to an identity provider and cannot be registered by hand."""


class HandleTaken(ValueError):
    """Raised when registering a handle (after normalisation) that already exists."""


class UnknownAccount(KeyError):
    """Raised when an operation references an account that does not exist."""


class FeeLike(Protocol):
    """Duck type of ``hivemind.router.market.FeePolicy``: only the two shares are used for settlement."""

    validator_share: float
    burn_share: float


@dataclass(frozen=True)
class Account:
    account_id: str
    handle: str
    role: str
    created_utc: float
    max_price_per_1k: float
    kind: str = "user"

    @property
    def is_test(self) -> bool:
        """True for every account kind that is not a real user ('test' or 'probe')."""
        return self.kind != KIND_USER


@dataclass(frozen=True)
class ProducerRecord:
    account_id: str
    endpoint_url: str
    models: tuple[tuple[str, str], ...]  # (public model id, producer-local model name)
    cost_per_1k: float
    capacity_1k: float
    last_heartbeat_utc: float
    execution_class: str
    endpoint_token: str | None = None  # secret the gateway presents to the producer's own server; never serialised
    # When an account advertises several live endpoints for a model, the record is the ACCOUNT's offer (capacity summed, cost the capacity-weighted mean,
    # the market sees one producer) and ``endpoints`` holds one single-endpoint record per server for dispatch. Empty for a one-endpoint account.
    endpoints: tuple[ProducerRecord, ...] = field(default=(), compare=False)
    # Optional residency report of THIS endpoint (HM-S16-06): ``{"gpu_vram_mib": int | None, "capacity_note": str | None,
    # "models": {public_id: {"loaded": bool | None, "vram_mib": int | None, "cold_start_s": float | None}}}``. ``None`` = the producer
    # reports nothing (old agent); such a producer is "unknown", never penalised. Always ``None`` on an account-level aggregate record.
    residency: Mapping[str, Any] | None = field(default=None, compare=False)

    def __repr__(self) -> str:
        return (f"ProducerRecord(account_id={self.account_id!r}, endpoint_url={self.endpoint_url!r}, models={self.models!r}, "
                f"cost_per_1k={self.cost_per_1k!r}, capacity_1k={self.capacity_1k!r}, execution_class={self.execution_class!r})")

    def per_endpoint(self) -> tuple[ProducerRecord, ...]:
        """The servers to try for this account: itself when it has one, otherwise one record per endpoint."""
        return self.endpoints or (self,)

    def _residency_entry(self, model_id: str) -> Mapping[str, Any] | None:
        models = self.residency.get("models") if self.residency else None
        entry = models.get(model_id) if isinstance(models, Mapping) else None
        return entry if isinstance(entry, Mapping) else None

    def model_state(self, model_id: str) -> bool | None:
        """``True`` = reported loaded, ``False`` = reported not loaded, ``None`` = the producer does not say."""
        entry = self._residency_entry(model_id)
        loaded = entry.get("loaded") if entry else None
        return loaded if isinstance(loaded, bool) else None

    def cold_start_s(self, model_id: str) -> float | None:
        """The producer's own estimate of the seconds needed to load ``model_id`` (``None`` when not reported)."""
        entry = self._residency_entry(model_id)
        value = entry.get("cold_start_s") if entry else None
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def local_name(self, model_id: str) -> str | None:
        """Return the producer's local name for ``model_id`` or ``None`` if it does not serve it."""
        for public, local in self.models:
            if public == model_id:
                return local
        return None


def _control_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {"paused": bool(row["paused"]), "vram_cap_mib": None if row["vram_cap_mib"] is None else int(row["vram_cap_mib"]),
            "updated_at": float(row["updated_at"]), "version": int(row["version"])}


def _load_residency(raw: object) -> dict[str, Any] | None:
    """Decode a stored residency report; anything malformed counts as "not reported"."""
    if raw is None:
        return None
    try:
        value = json.loads(str(raw))
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def fit_loaded_models(models: dict[str, dict[str, Any]], limit_mib: int, capacity_1k: float, what: str) -> tuple[float, str | None]:
    """Clamp the models reported loaded to ``limit_mib`` of video memory (shared by the card check and the remote VRAM cap, HM-S16-05/06).

    ``models`` is the residency ``models`` mapping and is changed in place: loaded models are kept in the order listed while their ``vram_mib`` fits
    and the rest are demoted to ``loaded = False``. A model with no declared size cannot be checked and is taken at face value. Returns the capacity
    scaled by the share of loaded models that were kept, and a ``capacity_note`` (``None`` when nothing had to be dropped).
    """
    loaded_ids = [m for m, v in models.items() if v.get("loaded") is True]
    used = 0
    kept = 0
    dropped: list[str] = []
    for model_id in loaded_ids:
        need = models[model_id].get("vram_mib")
        if need is None:
            kept += 1
        elif used + need <= limit_mib:
            used += need
            kept += 1
        else:
            dropped.append(model_id)
            models[model_id]["loaded"] = False
    if not dropped:
        return capacity_1k, None
    effective = capacity_1k * kept / len(loaded_ids)
    note = (
        f"declared loaded models exceed the {limit_mib} MiB {what}; treated as not loaded: {', '.join(dropped)}; "
        f"capacity scaled {capacity_1k:g} -> {effective:g}"
    )[:512]
    return effective, note


@dataclass(frozen=True)
class Settlement:
    request_id: str
    charge: float
    producer_credit: float
    validator_fee: float
    burn: float
    consumer_balance: float
    billed_tokens: int = 0


def _canonical_row(seq: int, account_id: str, delta: float, kind: str, ref: str, utc: float) -> str:
    return json.dumps(
        {"seq": seq, "account_id": account_id, "delta": delta, "kind": kind, "ref": ref, "utc": utc},
        sort_keys=True,
        separators=(",", ":"),
    )


def _entry_hash(prev_hash: str, canonical: str) -> str:
    return hashlib.sha256(f"{prev_hash}|{canonical}".encode()).hexdigest()


def _check_kind(kind: str) -> str:
    if kind not in ACCOUNT_KINDS:
        raise ValueError(f"kind must be one of {ACCOUNT_KINDS}")
    return kind


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


class RouterStore:
    """SQLite-backed accounts, API keys, producer registry, usage log and hash-chained HMT ledger."""

    def __init__(self, path: Path | str, secret: bytes) -> None:
        if not isinstance(secret, (bytes, bytearray)) or len(secret) < MIN_SECRET_BYTES:
            raise ValueError(f"secret must be at least {MIN_SECRET_BYTES} bytes")
        self._secret = bytes(secret)
        self._identity_providers: set[str] = set(IDENTITY_PROVIDERS)
        self._lock = threading.Lock()
        target = ":memory:" if str(path) == ":memory:" else str(path)
        self._conn = sqlite3.connect(target, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        if target != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._tx():
            for statement in _SCHEMA.split(";"):
                if statement.strip():
                    self._conn.execute(statement)
            # Databases created before the endpoint token existed are upgraded in place (additive, NULL for old rows).
            columns = {str(row["name"]) for row in self._conn.execute("PRAGMA table_info(producers)")}
            if "endpoint_token" not in columns:
                self._conn.execute("ALTER TABLE producers ADD COLUMN endpoint_token TEXT")
            # Databases created before model residency reporting (HM-S16-06) gain the nullable column in place.
            endpoint_columns = {str(row["name"]) for row in self._conn.execute("PRAGMA table_info(producer_endpoints)")}
            if "residency_json" not in endpoint_columns:
                self._conn.execute("ALTER TABLE producer_endpoints ADD COLUMN residency_json TEXT")
            # Databases created before the test/probe flag existed gain it in place; every existing account stays a real one.
            account_columns = {str(row["name"]) for row in self._conn.execute("PRAGMA table_info(accounts)")}
            if "is_test" not in account_columns:
                self._conn.execute("ALTER TABLE accounts ADD COLUMN is_test INTEGER NOT NULL DEFAULT 0")
            # Databases created before account kinds (HM-S16-07) gain the column in place: flagged accounts become 'test', all others 'user'.
            if "kind" not in account_columns:
                self._conn.execute("ALTER TABLE accounts ADD COLUMN kind TEXT NOT NULL DEFAULT 'user'")
                self._conn.execute("UPDATE accounts SET kind='test' WHERE is_test=1")
            # One account may advertise several endpoints (several GPUs / servers): rows of the old one-per-account table move across once.
            self._conn.execute(
                "INSERT OR IGNORE INTO producer_endpoints(account_id, endpoint_url, models_json, cost_per_1k, capacity_1k, last_heartbeat_utc, "
                "execution_class, endpoint_token) SELECT account_id, rtrim(endpoint_url, '/'), models_json, cost_per_1k, capacity_1k, last_heartbeat_utc, "
                "execution_class, endpoint_token FROM producers"
            )
            for sys_id in _ESCROW_ACCOUNTS:
                self._conn.execute(
                    "INSERT OR IGNORE INTO accounts(account_id, handle, role, key_hash, created_utc, max_price_per_1k) VALUES (?,?,?,?,?,?)",
                    (sys_id, sys_id, "admin", f"!{sys_id}", time.time(), 0.0),
                )

    # ------------------------------------------------------------------ plumbing
    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def _hmac_hex(self, message: str) -> str:
        return hmac.new(self._secret, message.encode(), hashlib.sha256).hexdigest()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _append(
        conn: sqlite3.Connection, account_id: str, delta: float, kind: str, ref: str, utc: float
    ) -> None:
        delta = float(delta) + 0.0  # normalise -0.0 to 0.0 so the hashed JSON survives the SQLite round trip
        last = conn.execute("SELECT seq, entry_hash FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()
        seq = 1 if last is None else int(last["seq"]) + 1
        prev = GENESIS_HASH if last is None else str(last["entry_hash"])
        digest = _entry_hash(prev, _canonical_row(seq, account_id, delta, kind, ref, utc))
        conn.execute(
            "INSERT INTO ledger(seq, account_id, delta, kind, ref, prev_hash, entry_hash, utc) VALUES (?,?,?,?,?,?,?,?)",
            (seq, account_id, delta, kind, ref, prev, digest, utc),
        )

    @staticmethod
    def _balance(conn: sqlite3.Connection, account_id: str) -> float:
        row = conn.execute("SELECT COALESCE(SUM(delta), 0.0) AS b FROM ledger WHERE account_id=?", (account_id,)).fetchone()
        return float(row["b"])

    @staticmethod
    def _row_account(row: sqlite3.Row) -> Account:
        return Account(
            account_id=str(row["account_id"]),
            handle=str(row["handle"]),
            role=str(row["role"]),
            created_utc=float(row["created_utc"]),
            max_price_per_1k=float(row["max_price_per_1k"]),
            kind=str(row["kind"]),
        )

    # ------------------------------------------------------------------ accounts
    def account_id_for(self, handle: str) -> str:
        """Deterministic account id for a handle (after identity normalisation)."""
        return self._hmac_hex(normalise_identity(handle))[:32]

    def register(self, handle: str, role: str, *, is_test: bool = False, kind: str | None = None) -> tuple[str, str]:
        """Create an account; returns ``(account_id, api_key)``. The key is shown once and never stored.

        ``kind`` ('user' by default, 'test' or 'probe') keeps an account off the market and out of the user counts;
        ``is_test=True`` is the older spelling of ``kind='test'``. The store does not authorise that choice; the gateway
        lets only an admin key ask for it.
        """
        kind = _check_kind(kind if kind is not None else ("test" if is_test else KIND_USER))
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        normalised = normalise_identity(handle)
        if normalised.startswith(tuple(f"{provider}." for provider in sorted(self._identity_providers))):
            raise ReservedHandle(f"handle prefix is reserved: {normalised}")
        return self._create_account(normalised, role, kind)

    def allow_identity_provider(self, name: str) -> None:
        """Enable sign-in through another OpenID Connect provider and reserve its handle prefix (``<name>.``) at once.

        Call it before any handle can be registered under that prefix: an existing account already named ``<name>.x``
        would be indistinguishable from a sign-in account.
        """
        if not IDENTITY_PROVIDER_NAME_RE.match(name):
            raise ValueError("provider name must be 2-16 lowercase letters or digits, starting with a letter")
        with self._tx() as conn:
            clash = conn.execute("SELECT 1 FROM accounts WHERE handle LIKE ? LIMIT 1", (f"{name}.%",)).fetchone()
        if clash is not None and name not in self._identity_providers:
            raise ReservedHandle(f"an account already uses the handle prefix {name}.")
        self._identity_providers.add(name)

    def sign_in_with_identity(self, provider: str, subject: str, role: str = "prosumer") -> tuple[str, str, bool]:
        """Create or log in the account that belongs to an identity-provider subject.

        Returns ``(account_id, api_key, created)``. The handle is a keyed hash of ``provider:subject``, so the provider's
        opaque id is not stored in clear and cannot be guessed. A returning user keeps their account, balance and role;
        each sign-in issues a NEW key and the previous one stops working, which doubles as lost-key recovery.
        """
        if provider not in self._identity_providers:
            raise ValueError(f"unknown identity provider {provider!r}")
        if not isinstance(subject, str) or not subject or len(subject) > 255:
            raise ValueError("subject must be a non-empty string of at most 255 characters")
        if role not in ROLES or role == "admin":
            raise ValueError(f"role must be one of {tuple(r for r in ROLES if r != 'admin')}")
        handle = f"{provider}.{self._hmac_hex(f'{provider}:{subject}')[:32]}"
        account_id = self._hmac_hex(handle)[:32]
        new_key = "hm-" + secrets.token_urlsafe(32)
        with self._tx() as conn:
            row = conn.execute("SELECT role FROM accounts WHERE account_id=?", (account_id,)).fetchone()
            if row is not None:
                conn.execute("UPDATE accounts SET key_hash=? WHERE account_id=?", (self._hmac_hex(new_key), account_id))
                return account_id, new_key, False
            conn.execute(
                "INSERT INTO accounts(account_id, handle, role, key_hash, created_utc, max_price_per_1k) VALUES (?,?,?,?,?,?)",
                (account_id, handle, role, self._hmac_hex(new_key), time.time(), DEFAULT_MAX_PRICE_PER_1K),
            )
        return account_id, new_key, True

    def _create_account(self, normalised: str, role: str, kind: str = KIND_USER) -> tuple[str, str]:
        account_id = self._hmac_hex(normalised)[:32]
        api_key = "hm-" + secrets.token_urlsafe(32)
        with self._tx() as conn:
            exists = conn.execute(
                "SELECT 1 FROM accounts WHERE account_id=? OR handle=?", (account_id, normalised)
            ).fetchone()
            if exists is not None:
                raise HandleTaken(f"handle already registered: {normalised}")
            conn.execute(
                "INSERT INTO accounts(account_id, handle, role, key_hash, created_utc, max_price_per_1k, is_test, kind) VALUES (?,?,?,?,?,?,?,?)",
                (account_id, normalised, role, self._hmac_hex(api_key), time.time(), DEFAULT_MAX_PRICE_PER_1K, int(kind != KIND_USER), kind),
            )
        return account_id, api_key

    def authenticate(self, api_key: str) -> Account | None:
        """Return the account owning ``api_key`` or ``None``; hashes are compared in constant time."""
        if not isinstance(api_key, str) or not api_key.startswith("hm-"):
            return None
        digest = self._hmac_hex(api_key)
        with self._lock:
            row = self._conn.execute("SELECT * FROM accounts WHERE key_hash=?", (digest,)).fetchone()
        if row is None or not hmac.compare_digest(str(row["key_hash"]), digest):
            return None
        return self._row_account(row)

    def get_account(self, account_id: str) -> Account | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM accounts WHERE account_id=?", (account_id,)).fetchone()
        return None if row is None else self._row_account(row)

    def consumer_max_prices(self) -> dict[str, float]:
        """Declared max price of every non-system, non-test account that may consume (everything but pure producers)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT account_id, max_price_per_1k FROM accounts WHERE role != 'producer' AND account_id NOT LIKE 'sys:%' "
                "AND kind = 'user' ORDER BY account_id"
            ).fetchall()
        return {str(r["account_id"]): float(r["max_price_per_1k"]) for r in rows}

    def funded_consumer_bids(self, expected_tokens: int, active_since: float | None = None) -> dict[str, float]:
        """Effective bids of accounts that can actually pay: ``min(max_price, balance * 1000 / expected_tokens)``.

        Accounts with a non-positive balance are omitted, so anonymous zero-balance sybils cannot move the price,
        and a funded account cannot bid above what its balance could pay for one ``expected_tokens`` request.
        With ``active_since`` set, only accounts with a usage row or a hold at or after that time bid: every sign-up
        is funded by the starter grant, so without it idle accounts that once raised their max price keep the
        clearing price above a new account's default max price and its first request is refused.
        Test and probe accounts (see :meth:`set_kind`) never bid, whatever their balance, price or activity.
        """
        tokens = max(1, int(expected_tokens))
        with self._lock:
            active: set[str] | None = None
            if active_since is not None:
                since = _finite(active_since, "active_since")
                active = {str(r[0]) for r in self._conn.execute(
                    "SELECT consumer_id FROM usage WHERE utc >= ? UNION SELECT account_id FROM holds WHERE utc >= ?",
                    (since, since),
                ).fetchall()}
            rows = self._conn.execute(
                "SELECT a.account_id AS account_id, a.max_price_per_1k AS price, COALESCE(SUM(l.delta), 0.0) AS bal "
                "FROM accounts a LEFT JOIN ledger l ON l.account_id = a.account_id "
                "WHERE a.role != 'producer' AND a.account_id NOT LIKE 'sys:%' AND a.kind = 'user' "
                "GROUP BY a.account_id ORDER BY a.account_id"
            ).fetchall()
        bids: dict[str, float] = {}
        for row in rows:
            balance = float(row["bal"])
            if balance > 0.0 and (active is None or str(row["account_id"]) in active):
                bids[str(row["account_id"])] = min(float(row["price"]), balance * 1000.0 / tokens)
        return bids

    def set_max_price(self, account_id: str, max_price_per_1k: float) -> None:
        price = _finite(max_price_per_1k, "max_price_per_1k")
        if price <= 0:
            raise ValueError("max_price_per_1k must be positive")
        with self._tx() as conn:
            cur = conn.execute("UPDATE accounts SET max_price_per_1k=? WHERE account_id=?", (price, account_id))
            if cur.rowcount != 1:
                raise UnknownAccount(account_id)

    def set_kind(self, account_id: str, kind: str) -> None:
        """Set an account's kind ('user', 'test' or 'probe'). Raises :class:`UnknownAccount` for an unknown or system id."""
        kind = _check_kind(kind)
        with self._tx() as conn:
            cur = conn.execute(
                "UPDATE accounts SET kind=?, is_test=? WHERE account_id=? AND account_id NOT LIKE 'sys:%'",
                (kind, int(kind != KIND_USER), account_id),
            )
            if cur.rowcount != 1:
                raise UnknownAccount(account_id)

    def set_test_flag(self, account_id: str, is_test: bool) -> None:
        """Mark (or unmark) an account as a test account; the older boolean spelling of :meth:`set_kind`."""
        self.set_kind(account_id, "test" if is_test else KIND_USER)

    def non_user_account_ids(self) -> frozenset[str]:
        """Ids of every test or probe account: their bids and their supply stay out of pricing."""
        with self._lock:
            rows = self._conn.execute("SELECT account_id FROM accounts WHERE kind != 'user'").fetchall()
        return frozenset(str(r[0]) for r in rows)

    def user_counts(self, since: float) -> dict[str, dict[str, int]]:
        """Real users, test accounts and probe accounts counted apart: ``total``, ``created`` and ``active`` (a usage row or a hold) at or after ``since``.

        System accounts are not users and are never counted.
        """
        cutoff = _finite(since, "since")
        counts = {name: {"total": 0, "created": 0, "active": 0} for name in ("real", "test", "probe")}
        with self._lock:
            active = {str(r[0]) for r in self._conn.execute(
                "SELECT consumer_id FROM usage WHERE utc >= ? UNION SELECT account_id FROM holds WHERE utc >= ?", (cutoff, cutoff),
            ).fetchall()}
            rows = self._conn.execute("SELECT account_id, kind, created_utc FROM accounts WHERE account_id NOT LIKE 'sys:%'").fetchall()
        for row in rows:
            bucket = counts["real" if row["kind"] == KIND_USER else str(row["kind"])]
            bucket["total"] += 1
            bucket["created"] += float(row["created_utc"]) >= cutoff
            bucket["active"] += str(row["account_id"]) in active
        return counts

    def claim_starter_slot(self, client_ip: str, now_utc: float, cap: int, window_s: float = STARTER_WINDOW_S) -> bool:
        """Reserve one starter grant for ``client_ip`` unless ``cap`` were already given to it in the last ``window_s``.

        Only a salted keyed hash of the address is stored (HMAC with the store secret), next to a timestamp; entries older
        than the window are deleted on every call. The raw address is never written or logged.
        """
        now = _finite(now_utc, "now_utc")
        digest = self._hmac_hex("starter-ip:" + client_ip)
        with self._tx() as conn:
            conn.execute("DELETE FROM starter_grant_ips WHERE utc <= ?", (now - window_s,))
            used = int(conn.execute("SELECT COUNT(*) FROM starter_grant_ips WHERE ip_hash=?", (digest,)).fetchone()[0])
            if used >= cap:
                return False
            conn.execute("INSERT INTO starter_grant_ips(ip_hash, utc) VALUES (?,?)", (digest, now))
        return True

    # ------------------------------------------------------------------ ledger
    def balance(self, account_id: str) -> float:
        with self._lock:
            return self._balance(self._conn, account_id)

    def grant(self, account_id: str, amount: float, ref: str) -> None:
        """Admin mint: the only operation that creates HMT."""
        value = _finite(amount, "amount")
        if value <= 0:
            raise ValueError("grant amount must be positive")
        with self._tx() as conn:
            if conn.execute("SELECT 1 FROM accounts WHERE account_id=?", (account_id,)).fetchone() is None:
                raise UnknownAccount(account_id)
            self._append(conn, account_id, value, "grant", ref, time.time())

    def total_minted(self) -> float:
        with self._lock:
            row = self._conn.execute("SELECT COALESCE(SUM(delta), 0.0) AS m FROM ledger WHERE kind='grant'").fetchone()
        return float(row["m"])

    def total_balances(self) -> float:
        """Sum of every balance, system accounts included (equals ``total_minted()`` unless the ledger is corrupt)."""
        with self._lock:
            row = self._conn.execute("SELECT COALESCE(SUM(delta), 0.0) AS s FROM ledger").fetchone()
        return float(row["s"])

    def min_balance(self) -> float:
        """Smallest balance over all accounts that appear in the ledger (0.0 when the ledger is empty)."""
        with self._lock:
            rows = self._conn.execute("SELECT SUM(delta) AS b FROM ledger GROUP BY account_id").fetchall()
        return min((float(r["b"]) for r in rows), default=0.0)

    def verify_chain(self) -> bool:
        """Recompute every hash; ``False`` if any row was altered, removed, reordered or re-linked."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, account_id, delta, kind, ref, prev_hash, entry_hash, utc FROM ledger ORDER BY seq"
            ).fetchall()
        prev = GENESIS_HASH
        expected_seq = 1
        for row in rows:
            seq = int(row["seq"])
            if seq != expected_seq or str(row["prev_hash"]) != prev:
                return False
            canonical = _canonical_row(seq, str(row["account_id"]), float(row["delta"]), str(row["kind"]), str(row["ref"]), float(row["utc"]))
            digest = _entry_hash(prev, canonical)
            if not hmac.compare_digest(digest, str(row["entry_hash"])):
                return False
            prev = digest
            expected_seq += 1
        return True

    # ------------------------------------------------------------------ escrow
    def hold(self, request_id: str, account_id: str, amount: float, *, max_inflight: int = MAX_INFLIGHT_HOLDS,
             ttl_s: float = HOLD_TTL_S, now_utc: float | None = None) -> None:
        """Escrow ``amount`` from the account (debit it, credit ``sys:escrow``) in one transaction.

        Raises :class:`InsufficientBalance` if the balance cannot cover it (balances already net of other holds, so
        concurrent requests cannot spend one balance twice) and :class:`TooManyHolds` beyond ``max_inflight`` active
        holds. Holds older than ``ttl_s`` seconds are released first, so a request that never settled cannot lock
        funds forever.
        """
        value = _finite(amount, "amount")
        if value < 0:
            raise ValueError("hold amount must be non-negative")
        now = time.time() if now_utc is None else float(now_utc)
        with self._tx() as conn:
            if conn.execute("SELECT 1 FROM accounts WHERE account_id=?", (account_id,)).fetchone() is None:
                raise UnknownAccount(account_id)
            stale = conn.execute(
                "SELECT request_id FROM holds WHERE account_id=? AND state='active' AND utc < ?", (account_id, now - ttl_s)
            ).fetchall()
            for row in stale:
                self._release(conn, str(row["request_id"]), now)
            active = conn.execute("SELECT COUNT(*) AS n FROM holds WHERE account_id=? AND state='active'", (account_id,)).fetchone()
            if int(active["n"]) >= max_inflight:
                raise TooManyHolds(f"{active['n']} in-flight holds")
            if conn.execute("SELECT 1 FROM holds WHERE request_id=?", (request_id,)).fetchone() is not None:
                raise DuplicateRequest(request_id)
            balance = self._balance(conn, account_id)
            if balance < value:
                raise InsufficientBalance(f"balance {balance:.9f} < hold {value:.9f}")
            ref = f"req:{request_id}"
            self._append(conn, account_id, -value, "hold", ref, now)
            self._append(conn, SYS_ESCROW, value, "hold", ref, now)
            conn.execute("INSERT INTO holds VALUES (?,?,?,?,'active')", (request_id, account_id, value, now))

    @classmethod
    def _release(cls, conn: sqlite3.Connection, request_id: str, now: float) -> float | None:
        row = conn.execute("SELECT account_id, amount FROM holds WHERE request_id=? AND state='active'", (request_id,)).fetchone()
        if row is None:
            return None
        amount = float(row["amount"])
        ref = f"req:{request_id}"
        cls._append(conn, SYS_ESCROW, -amount, "hold_release", ref, now)
        cls._append(conn, str(row["account_id"]), amount, "hold_release", ref, now)
        conn.execute("UPDATE holds SET state='released' WHERE request_id=?", (request_id,))
        return amount

    def release_hold(self, request_id: str) -> bool:
        """Return an unsettled hold to its account. Idempotent: ``False`` when there is nothing to release."""
        with self._tx() as conn:
            return self._release(conn, request_id, time.time()) is not None

    def active_hold_count(self, account_id: str) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM holds WHERE account_id=? AND state='active'", (account_id,)).fetchone()
        return int(row["n"])

    def settle(
        self,
        request_id: str,
        consumer_id: str,
        producer_id: str,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        billed_tokens: int,
        price_per_1k: float,
        fee: FeeLike | float | None = None,
        burn_share: float | None = None,
    ) -> Settlement:
        """Atomically debit the consumer and credit producer, ``sys:validators`` and ``sys:burn``.

        ``fee`` is a FeePolicy-like object (``validator_share``/``burn_share`` attributes), or a plain float
        ``validator_share`` with ``burn_share`` given separately. ``None`` selects the 0.10 / 0.05 defaults.
        """
        if fee is None:
            validator_share, burn = DEFAULT_VALIDATOR_SHARE, DEFAULT_BURN_SHARE
        elif isinstance(fee, (int, float)):
            validator_share = float(fee)
            burn = DEFAULT_BURN_SHARE if burn_share is None else float(burn_share)
        else:
            validator_share, burn = float(fee.validator_share), float(fee.burn_share)
        _finite(validator_share, "validator_share")
        _finite(burn, "burn_share")
        if validator_share < 0 or burn < 0 or validator_share + burn >= 1.0:
            raise ValueError("fee shares must be non-negative and sum to less than 1")
        price = _finite(price_per_1k, "price_per_1k")
        if price < 0:
            raise ValueError("price_per_1k must be non-negative")
        if min(prompt_tokens, completion_tokens, billed_tokens) < 0:
            raise ValueError("token counts must be non-negative")
        now = time.time()
        with self._tx() as conn:
            for acct in (consumer_id, producer_id):
                if conn.execute("SELECT 1 FROM accounts WHERE account_id=?", (acct,)).fetchone() is None:
                    raise UnknownAccount(acct)
            if conn.execute("SELECT 1 FROM usage WHERE request_id=?", (request_id,)).fetchone() is not None:
                raise DuplicateRequest(request_id)
            held = self._release(conn, request_id, now)
            billed = int(billed_tokens)
            charge = price * billed / 1000.0
            if held is not None and charge > held:
                # the escrow is the ceiling: bill what it can pay for, never a free remainder
                billed = int(math.floor(held * 1000.0 / price)) if price > 0 else billed
                charge = min(price * billed / 1000.0, held)
                prompt_tokens = min(int(prompt_tokens), billed)
                completion_tokens = billed - int(prompt_tokens)
            validator_fee = charge * validator_share
            burn_amount = charge * burn
            producer_credit = charge - validator_fee - burn_amount
            balance = self._balance(conn, consumer_id)
            if balance < charge:
                raise InsufficientBalance(f"balance {balance:.9f} < charge {charge:.9f}")
            ref = f"req:{request_id}"
            self._append(conn, consumer_id, -charge, "debit", ref, now)
            self._append(conn, producer_id, producer_credit, "producer_credit", ref, now)
            self._append(conn, SYS_VALIDATORS, validator_fee, "validator_fee", ref, now)
            self._append(conn, SYS_BURN, burn_amount, "burn", ref, now)
            conn.execute(
                "INSERT INTO usage VALUES (?,?,?,?,?,?,?,?,?,?)",
                (request_id, consumer_id, producer_id, model_id, int(prompt_tokens), int(completion_tokens), billed, price, charge, now),
            )
            remaining = self._balance(conn, consumer_id)
        return Settlement(request_id, charge, producer_credit, validator_fee, burn_amount, remaining, billed)

    def recent_usage(self, account_id: str, limit: int = 20) -> list[dict[str, object]]:
        """Most recent usage rows where the account was consumer or producer."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM usage WHERE consumer_id=? OR producer_id=? ORDER BY utc DESC, rowid DESC LIMIT ?",
                (account_id, account_id, int(limit)),
            ).fetchall()
        return [{key: row[key] for key in row.keys()} for row in rows]

    # ------------------------------------------------------------------ producers
    def upsert_producer(
        self,
        account_id: str,
        endpoint_url: str,
        models: Sequence[Mapping[str, str]],
        cost_per_1k: float,
        capacity_1k: float,
        execution_class: str = "unknown",
        now_utc: float | None = None,
        endpoint_token: str | None = None,
        residency: Mapping[str, Any] | None = None,
    ) -> int:
        """Insert or refresh ONE endpoint of a producer account; doubles as the heartbeat. Returns how many endpoints the account now advertises.

        An account may advertise several endpoints (several GPUs or servers, each with its own models, cost and capacity): endpoints are keyed
        by ``(account, endpoint_url)``, so a heartbeat refreshes only its own endpoint and never evicts the others. At most
        ``MAX_ENDPOINTS_PER_ACCOUNT`` per account (``TooManyEndpoints``); endpoints silent for a day are forgotten first.

        ``endpoint_token`` is the bearer secret the gateway presents to the producer's own server so that only the
        gateway can use it; a heartbeat without one clears any earlier token.

        ``residency`` is the validated, clamped residency report (see ``ProducerRecord.residency``); a heartbeat without one clears the earlier report.
        """
        cost = _finite(cost_per_1k, "cost_per_1k")
        capacity = _finite(capacity_1k, "capacity_1k")
        if cost < 0 or capacity < 0:
            raise ValueError("cost_per_1k and capacity_1k must be non-negative")
        pairs = [{"id": str(m["id"]), "local_name": str(m.get("local_name", m["id"]))} for m in models]
        stamp = time.time() if now_utc is None else float(now_utc)
        with self._tx() as conn:
            row = conn.execute("SELECT role FROM accounts WHERE account_id=?", (account_id,)).fetchone()
            if row is None:
                raise UnknownAccount(account_id)
            if str(row["role"]) not in PRODUCER_ROLES:
                raise ValueError("account role cannot produce")
            url = endpoint_url.rstrip("/")
            conn.execute("DELETE FROM producer_endpoints WHERE account_id=? AND last_heartbeat_utc < ?", (account_id, stamp - STALE_ENDPOINT_PURGE_S))
            known = {str(r["endpoint_url"]) for r in conn.execute("SELECT endpoint_url FROM producer_endpoints WHERE account_id=?", (account_id,))}
            if url not in known and len(known) >= MAX_ENDPOINTS_PER_ACCOUNT:
                raise TooManyEndpoints(f"an account may advertise at most {MAX_ENDPOINTS_PER_ACCOUNT} endpoints")
            conn.execute(
                "INSERT INTO producer_endpoints(account_id, endpoint_url, models_json, cost_per_1k, capacity_1k, last_heartbeat_utc, "
                "execution_class, endpoint_token, residency_json) VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(account_id, endpoint_url) DO UPDATE SET "
                "models_json=excluded.models_json, cost_per_1k=excluded.cost_per_1k, "
                "capacity_1k=excluded.capacity_1k, last_heartbeat_utc=excluded.last_heartbeat_utc, "
                "execution_class=excluded.execution_class, endpoint_token=excluded.endpoint_token, residency_json=excluded.residency_json",
                (account_id, url, json.dumps(pairs, sort_keys=True), cost, capacity, stamp, execution_class, endpoint_token,
                 None if residency is None else json.dumps(residency, sort_keys=True)),
            )
            return len(known | {url})

    # ------------------------------------------------------------------ remote control (HM-S16-05)
    def set_producer_control(self, account_id: str, paused: bool, vram_cap_mib: int | None, now_utc: float | None = None) -> dict[str, Any]:
        """Replace the remote-control state of a producer account (pause flag and optional VRAM cap) and return it. Unknown account: ``UnknownAccount``."""
        if vram_cap_mib is not None and (isinstance(vram_cap_mib, bool) or not isinstance(vram_cap_mib, int) or vram_cap_mib < 1):
            raise ValueError("vram_cap_mib must be a positive integer or None")
        stamp = time.time() if now_utc is None else float(now_utc)
        with self._tx() as conn:
            if conn.execute("SELECT 1 FROM accounts WHERE account_id=?", (account_id,)).fetchone() is None:
                raise UnknownAccount(account_id)
            conn.execute(
                "INSERT INTO producer_control(account_id, paused, vram_cap_mib, updated_at, version) VALUES (?,?,?,?,1) "
                "ON CONFLICT(account_id) DO UPDATE SET paused=excluded.paused, vram_cap_mib=excluded.vram_cap_mib, "
                "updated_at=excluded.updated_at, version=producer_control.version+1",
                (account_id, 1 if paused else 0, vram_cap_mib, stamp),
            )
        control = self.producer_control(account_id)
        assert control is not None
        return control

    def producer_control(self, account_id: str) -> dict[str, Any] | None:
        """The stored control state of one account, or ``None`` when none was ever set (such an account behaves exactly as before)."""
        with self._lock:
            row = self._conn.execute("SELECT * FROM producer_control WHERE account_id=?", (account_id,)).fetchone()
        return None if row is None else _control_dict(row)

    def _control_by_account(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM producer_control WHERE paused=1 OR vram_cap_mib IS NOT NULL").fetchall()
        return {str(row["account_id"]): _control_dict(row) for row in rows}

    def active_producers(self, model_id: str, now_utc: float, ttl_s: float = 120) -> list[ProducerRecord]:
        """One record per producer ACCOUNT serving ``model_id`` with at least one endpoint whose heartbeat is within ``ttl_s`` seconds of ``now_utc``.

        An account with several live endpoints for the model is one market participant: its record carries the summed capacity and the
        capacity-weighted mean cost, and ``endpoints`` lists the single-endpoint records the gateway dispatches to.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM producer_endpoints WHERE last_heartbeat_utc >= ? ORDER BY account_id, endpoint_url", (now_utc - ttl_s,)
            ).fetchall()
        controls = self._control_by_account()
        by_account: dict[str, list[ProducerRecord]] = {}
        for row in rows:
            control = controls.get(str(row["account_id"]))
            if control is not None and control["paused"]:
                continue  # remote pause: never selected and not counted as supply, whatever the agent does
            models = tuple((str(m["id"]), str(m["local_name"])) for m in json.loads(str(row["models_json"])))
            if all(public != model_id for public, _ in models):
                continue
            residency = _load_residency(row["residency_json"])
            capacity_1k = float(row["capacity_1k"])
            cap = None if control is None else control["vram_cap_mib"]
            if cap is not None and residency is not None and isinstance(residency.get("models"), dict):
                # remote VRAM cap: same clamp as the card check (a fresh copy is decoded per call, so changing it here is safe)
                capacity_1k, note = fit_loaded_models(residency["models"], cap, capacity_1k, "remote VRAM cap")
                if note is not None:
                    residency["capacity_note"] = note
            by_account.setdefault(str(row["account_id"]), []).append(
                ProducerRecord(
                    account_id=str(row["account_id"]),
                    endpoint_url=str(row["endpoint_url"]),
                    models=models,
                    cost_per_1k=float(row["cost_per_1k"]),
                    capacity_1k=capacity_1k,
                    last_heartbeat_utc=float(row["last_heartbeat_utc"]),
                    execution_class=str(row["execution_class"]),
                    endpoint_token=None if row["endpoint_token"] is None else str(row["endpoint_token"]),
                    residency=residency,
                )
            )
        out: list[ProducerRecord] = []
        for account_id in sorted(by_account):
            servers = by_account[account_id]
            if len(servers) == 1:
                out.append(servers[0])
                continue
            capacity = math.fsum(s.capacity_1k for s in servers)
            cost = math.fsum(s.cost_per_1k * s.capacity_1k for s in servers) / capacity if capacity > 0.0 else min(s.cost_per_1k for s in servers)
            classes = {s.execution_class for s in servers}
            first = servers[0]
            out.append(
                ProducerRecord(
                    account_id=account_id,
                    endpoint_url=first.endpoint_url,
                    models=first.models,
                    cost_per_1k=cost,
                    capacity_1k=capacity,
                    last_heartbeat_utc=max(s.last_heartbeat_utc for s in servers),
                    execution_class=first.execution_class if len(classes) == 1 else "mixed",
                    endpoint_token=first.endpoint_token,
                    endpoints=tuple(servers),
                )
            )
        return out

    def residency_summary(self, now_utc: float, ttl_s: float = 120) -> dict[str, bool]:
        """Per live model that at least one live endpoint reports residency for: whether any endpoint has it loaded. No producer identities."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT residency_json FROM producer_endpoints WHERE last_heartbeat_utc >= ? AND residency_json IS NOT NULL "
                "AND account_id NOT IN (SELECT account_id FROM producer_control WHERE paused=1)", (now_utc - ttl_s,)
            ).fetchall()
        out: dict[str, bool] = {}
        for row in rows:
            residency = _load_residency(row["residency_json"])
            models = residency.get("models") if residency else None
            if not isinstance(models, Mapping):
                continue
            for model_id, entry in models.items():
                if isinstance(entry, Mapping) and isinstance(entry.get("loaded"), bool):
                    out[str(model_id)] = out.get(str(model_id), False) or bool(entry["loaded"])
        return out

    def live_models(self, now_utc: float, ttl_s: float = 120) -> list[str]:
        """Sorted public model ids served by at least one producer whose heartbeat is within ``ttl_s`` seconds."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT models_json FROM producer_endpoints WHERE last_heartbeat_utc >= ? "
                "AND account_id NOT IN (SELECT account_id FROM producer_control WHERE paused=1)", (now_utc - ttl_s,)
            ).fetchall()
        found: set[str] = set()
        for row in rows:
            found.update(str(m["id"]) for m in json.loads(str(row["models_json"])))
        return sorted(found)

    def key_digest(self, text: str) -> str:
        """``HMAC-SHA256(secret, text)`` as hex: the digest stored for API keys and expected for the admin key."""
        return self._hmac_hex(text)

    # ------------------------------------------------------------------ usage capture & feedback (HM-S19-USAGE)
    def record_usage_telemetry(
        self,
        *,
        request_id: str,
        consumer_id: str,
        producer_id: str,
        requested_model: str,
        routed_model: str,
        routing_mode: str,
        task_domain: str,
        language: str,
        client_sdk: str,
        app_title: str,
        is_stream: bool,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        billed_tokens: int,
        charged_hmt: float,
        now_utc: float | None = None,
    ) -> None:
        """Store privacy-preserving usage metadata for one settled request (never raw prompt text or IP)."""
        stamp = time.time() if now_utc is None else float(now_utc)
        lat = max(0.0, float(latency_ms)) if math.isfinite(latency_ms) else 0.0
        with self._tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO usage_telemetry(request_id, consumer_id, producer_id, requested_model, "
                "routed_model, routing_mode, task_domain, language, client_sdk, app_title, is_stream, latency_ms, "
                "prompt_tokens, completion_tokens, billed_tokens, charged_hmt, utc) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(request_id),
                    str(consumer_id),
                    str(producer_id),
                    str(requested_model)[:128],
                    str(routed_model)[:128],
                    str(routing_mode)[:32],
                    str(task_domain)[:64],
                    str(language)[:16],
                    str(client_sdk)[:64],
                    str(app_title)[:64],
                    1 if is_stream else 0,
                    lat,
                    int(prompt_tokens),
                    int(completion_tokens),
                    int(billed_tokens),
                    float(charged_hmt),
                    stamp,
                ),
            )

    def record_usage_feedback(
        self,
        request_id: str,
        account_id: str,
        rating: int,
        comment: str = "",
        *,
        now_utc: float | None = None,
    ) -> dict[str, Any]:
        """Record a consumer's thumbs-up (+1) or thumbs-down (-1) rating on their own settled request."""
        if rating not in (-1, 1):
            raise ValueError("rating must be 1 or -1")
        stamp = time.time() if now_utc is None else float(now_utc)
        clean_comment = str(comment).strip()[:256]
        with self._tx() as conn:
            usage_row = conn.execute(
                "SELECT consumer_id, model_id FROM usage WHERE request_id=?",
                (request_id,),
            ).fetchone()
            if usage_row is None or str(usage_row["consumer_id"]) != account_id:
                raise KeyError(request_id)
            tel_row = conn.execute(
                "SELECT task_domain FROM usage_telemetry WHERE request_id=?",
                (request_id,),
            ).fetchone()
            domain = str(tel_row["task_domain"]) if tel_row is not None else "general_chat"
            conn.execute(
                "INSERT INTO usage_feedback(request_id, account_id, rating, task_domain, comment, utc) "
                "VALUES (?,?,?,?,?,?) ON CONFLICT(request_id) DO UPDATE SET "
                "rating=excluded.rating, comment=excluded.comment, utc=excluded.utc",
                (request_id, account_id, int(rating), domain, clean_comment, stamp),
            )
        return {
            "request_id": request_id,
            "account_id": account_id,
            "rating": int(rating),
            "task_domain": domain,
            "comment": clean_comment,
            "utc": stamp,
        }

    def usage_capture_summary(
        self,
        *,
        now_utc: float | None = None,
        window_s: float = 86_400.0,
        include_non_user: bool = False,
    ) -> dict[str, Any]:
        """Aggregate anonymized usage telemetry over the last ``window_s`` seconds."""
        from hivemind.router.usage_capture import compute_nocturnal_moe_workunit_weights

        now = time.time() if now_utc is None else float(now_utc)
        since = now - max(0.0, float(window_s))
        with self._lock:
            if include_non_user:
                rows = self._conn.execute(
                    "SELECT * FROM usage_telemetry WHERE utc >= ? ORDER BY utc DESC",
                    (since,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT t.* FROM usage_telemetry t JOIN accounts a ON a.account_id = t.consumer_id "
                    "WHERE t.utc >= ? AND a.kind = 'user' ORDER BY t.utc DESC",
                    (since,),
                ).fetchall()
            feedback_rows = self._conn.execute(
                "SELECT rating, task_domain FROM usage_feedback WHERE utc >= ?",
                (since,),
            ).fetchall()

        by_domain: dict[str, dict[str, Any]] = {}
        by_language: dict[str, int] = {}
        by_client_sdk: dict[str, int] = {}
        by_model: dict[str, dict[str, Any]] = {}
        by_routing_mode: dict[str, int] = {}
        domain_tokens: dict[str, int] = {}
        latencies: list[float] = []
        total_tokens = 0
        total_hmt = 0.0
        stream_count = 0

        for r in rows:
            domain = str(r["task_domain"])
            lang = str(r["language"])
            sdk = str(r["client_sdk"])
            model = str(r["routed_model"])
            mode = str(r["routing_mode"])
            tokens = int(r["billed_tokens"])
            hmt = float(r["charged_hmt"])
            lat = float(r["latency_ms"])

            total_tokens += tokens
            total_hmt += hmt
            if int(r["is_stream"]):
                stream_count += 1
            latencies.append(lat)

            d_entry = by_domain.setdefault(domain, {"requests": 0, "billed_tokens": 0})
            d_entry["requests"] += 1
            d_entry["billed_tokens"] += tokens
            domain_tokens[domain] = domain_tokens.get(domain, 0) + tokens

            by_language[lang] = by_language.get(lang, 0) + 1
            by_client_sdk[sdk] = by_client_sdk.get(sdk, 0) + 1
            by_routing_mode[mode] = by_routing_mode.get(mode, 0) + 1

            m_entry = by_model.setdefault(model, {"requests": 0, "billed_tokens": 0, "charged_hmt": 0.0})
            m_entry["requests"] += 1
            m_entry["billed_tokens"] += tokens
            m_entry["charged_hmt"] = round(m_entry["charged_hmt"] + hmt, 6)

        latencies.sort()
        if latencies:
            p50 = latencies[int(0.50 * (len(latencies) - 1))]
            p95 = latencies[int(0.95 * (len(latencies) - 1))]
            mean_lat = sum(latencies) / len(latencies)
        else:
            p50, p95, mean_lat = 0.0, 0.0, 0.0

        thumbs_up = sum(1 for f in feedback_rows if int(f["rating"]) == 1)
        thumbs_down = sum(1 for f in feedback_rows if int(f["rating"]) == -1)
        total_fb = thumbs_up + thumbs_down

        return {
            "object": "hivemind.usage_capture",
            "window_s": float(window_s),
            "total_requests": len(rows),
            "stream_requests": stream_count,
            "total_billed_tokens": total_tokens,
            "total_charged_hmt": round(total_hmt, 6),
            "by_task_domain": by_domain,
            "by_language": by_language,
            "by_client_sdk": by_client_sdk,
            "by_routed_model": by_model,
            "by_routing_mode": by_routing_mode,
            "latency_ms": {
                "p50": round(p50, 2),
                "p95": round(p95, 2),
                "mean": round(mean_lat, 2),
            },
            "feedback": {
                "thumbs_up": thumbs_up,
                "thumbs_down": thumbs_down,
                "total_ratings": total_fb,
                "satisfaction_rate": round(thumbs_up / total_fb, 4) if total_fb > 0 else 1.0,
            },
            "nocturnal_moe_workunit_weights": compute_nocturnal_moe_workunit_weights(domain_tokens),
        }

    # ------------------------------------------------------------------ joint campaigns, poll & AMCP/Kafka bus (HM-S19-CAMPAIGNS)
    def record_campaign_poll_vote(
        self,
        *,
        account_id: str,
        participant_handle: str,
        participant_kind: str,
        subject_id: str,
        pledged_vram_gb: float = 0.0,
        availability_slot: str = "eu_evening_cet",
        comment: str = "",
        now_utc: float | None = None,
    ) -> dict[str, Any]:
        """Insert or update a participant's interest & VRAM pledge for one Subject Campaign."""
        stamp = time.time() if now_utc is None else float(now_utc)
        vram = max(0.0, min(float(pledged_vram_gb), 1024.0)) if math.isfinite(pledged_vram_gb) else 0.0
        clean_comment = str(comment).strip()[:256]
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO campaign_poll_votes(account_id, subject_id, participant_handle, participant_kind, "
                "pledged_vram_gb, availability_slot, comment, utc) VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(account_id, subject_id) DO UPDATE SET "
                "participant_kind=excluded.participant_kind, pledged_vram_gb=excluded.pledged_vram_gb, "
                "availability_slot=excluded.availability_slot, comment=excluded.comment, utc=excluded.utc",
                (
                    str(account_id),
                    str(subject_id)[:64],
                    str(participant_handle)[:64],
                    str(participant_kind)[:32],
                    vram,
                    str(availability_slot)[:32],
                    clean_comment,
                    stamp,
                ),
            )
        return {
            "account_id": account_id,
            "subject_id": subject_id,
            "participant_handle": participant_handle,
            "participant_kind": participant_kind,
            "pledged_vram_gb": vram,
            "availability_slot": availability_slot,
            "comment": clean_comment,
            "utc": stamp,
        }

    def list_campaign_poll_votes(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM campaign_poll_votes ORDER BY utc DESC").fetchall()
        return [{k: row[k] for k in row.keys()} for row in rows]

    def record_amcp_kafka_event(
        self,
        *,
        event_id: str,
        topic: str,
        event_type: str,
        sender_id: str,
        sender_kind: str,
        recipient_id: str,
        subject_id: str,
        correlation_id: str,
        payload: Mapping[str, Any],
        now_utc: float | None = None,
    ) -> None:
        stamp = time.time() if now_utc is None else float(now_utc)
        with self._tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO amcp_kafka_events(event_id, topic, event_type, sender_id, sender_kind, "
                "recipient_id, subject_id, correlation_id, payload_json, utc) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    str(event_id)[:64],
                    str(topic)[:64],
                    str(event_type)[:64],
                    str(sender_id)[:64],
                    str(sender_kind)[:32],
                    str(recipient_id)[:64],
                    str(subject_id)[:64],
                    str(correlation_id)[:64],
                    json.dumps(dict(payload), sort_keys=True)[:4096],
                    stamp,
                ),
            )

    def list_amcp_kafka_events(self, topic: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        lim = max(1, min(int(limit), 200))
        with self._lock:
            if topic:
                rows = self._conn.execute(
                    "SELECT * FROM amcp_kafka_events WHERE topic=? ORDER BY utc DESC LIMIT ?",
                    (str(topic), lim),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM amcp_kafka_events ORDER BY utc DESC LIMIT ?",
                    (lim,),
                ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            item = {k: r[k] for k in r.keys() if k != "payload_json"}
            try:
                item["payload"] = json.loads(str(r["payload_json"]))
            except ValueError:
                item["payload"] = {}
            out.append(item)
        return out

    def record_bee_trade_order(
        self,
        *,
        order_id: str,
        account_id: str,
        bee_id: str,
        side: str,
        subject_id: str,
        capacity_1k: float,
        price_per_1k: float,
        now_utc: float | None = None,
    ) -> dict[str, Any]:
        stamp = time.time() if now_utc is None else float(now_utc)
        side_up = str(side).upper()
        if side_up not in ("OFFER", "BID"):
            raise ValueError("side must be OFFER or BID")
        cap = _finite(capacity_1k, "capacity_1k")
        price = _finite(price_per_1k, "price_per_1k")
        if cap <= 0.0 or price <= 0.0:
            raise ValueError("capacity_1k and price_per_1k must be positive")
        with self._tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO bee_trade_orders(order_id, account_id, bee_id, side, subject_id, "
                "capacity_1k, price_per_1k, utc) VALUES (?,?,?,?,?,?,?,?)",
                (str(order_id)[:64], str(account_id), str(bee_id)[:64], side_up, str(subject_id)[:64], cap, price, stamp),
            )
        return {
            "order_id": order_id,
            "account_id": account_id,
            "bee_id": bee_id,
            "side": side_up,
            "subject_id": subject_id,
            "capacity_1k": cap,
            "price_per_1k": price,
            "utc": stamp,
        }

    def list_bee_trade_orders(self, window_s: float = 86_400.0, now_utc: float | None = None) -> list[dict[str, Any]]:
        now = time.time() if now_utc is None else float(now_utc)
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM bee_trade_orders WHERE utc >= ? ORDER BY utc ASC",
                (now - max(0.0, float(window_s)),),
            ).fetchall()
        return [{k: row[k] for k in row.keys()} for row in rows]

    def producer_fleet_scale_status(
        self,
        now_utc: float | None = None,
        ttl_s: float = 120.0,
        target_producers: int = 100,
    ) -> dict[str, Any]:
        """Aggregate live producer accounts, endpoints, execution classes, and residency progress toward the 100-node gate."""
        now = time.time() if now_utc is None else float(now_utc)
        target = max(1, int(target_producers))
        with self._lock:
            rows = self._conn.execute(
                "SELECT account_id, endpoint_url, models_json, capacity_1k, execution_class, residency_json "
                "FROM producer_endpoints WHERE last_heartbeat_utc >= ? "
                "AND account_id NOT IN (SELECT account_id FROM producer_control WHERE paused=1)",
                (now - float(ttl_s),),
            ).fetchall()
            paused_row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM producer_control WHERE paused=1"
            ).fetchone()
        paused_count = int(paused_row["n"]) if paused_row is not None else 0
        accounts: set[str] = set()
        exec_classes: dict[str, int] = {}
        live_models: set[str] = set()
        warm_models: set[str] = set()
        residency_endpoints = 0
        total_capacity_1k = 0.0

        for row in rows:
            accounts.add(str(row["account_id"]))
            ec = str(row["execution_class"] or "ollama")
            exec_classes[ec] = exec_classes.get(ec, 0) + 1
            total_capacity_1k += float(row["capacity_1k"] or 0.0)
            try:
                for m in json.loads(str(row["models_json"])):
                    if isinstance(m, Mapping) and "id" in m:
                        live_models.add(str(m["id"]))
            except ValueError:
                pass
            residency = _load_residency(row["residency_json"])
            if residency is not None:
                residency_endpoints += 1
                rmodels = residency.get("models")
                if isinstance(rmodels, Mapping):
                    for mid, entry in rmodels.items():
                        if isinstance(entry, Mapping) and entry.get("loaded") is True:
                            warm_models.add(str(mid))

        live_acc_count = len(accounts)
        return {
            "target_producers": target,
            "live_producer_accounts": live_acc_count,
            "live_endpoints": len(rows),
            "paused_producer_accounts": paused_count,
            "scale_readiness_pct": round(min(100.0, (100.0 * live_acc_count) / target), 2),
            "target_reached": live_acc_count >= target,
            "total_capacity_1k": round(total_capacity_1k, 2),
            "execution_class_breakdown": exec_classes,
            "residency_reporting_endpoints": residency_endpoints,
            "warm_loaded_models_count": len(warm_models),
            "live_models_count": len(live_models),
        }



