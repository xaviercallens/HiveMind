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
import secrets
import sqlite3
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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
    max_price_per_1k REAL NOT NULL
);
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
"""


class InsufficientBalance(Exception):
    """Raised by :meth:`RouterStore.settle` when the consumer cannot cover the charge (nothing is written)."""


class TooManyHolds(Exception):
    """Raised by :meth:`RouterStore.hold` when the account already has the maximum number of in-flight holds."""


class DuplicateRequest(ValueError):
    """Raised when a ``request_id`` has already been settled."""


# Handles of the form ``<provider>.<keyed hash>`` belong to identity-provider sign-ins; nobody can register one by hand,
# so a Google account cannot be squatted by typing its handle into the open registration form.
IDENTITY_PROVIDERS: tuple[str, ...] = ("google",)
RESERVED_HANDLE_PREFIXES: tuple[str, ...] = tuple(f"{provider}." for provider in IDENTITY_PROVIDERS)


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

    def __repr__(self) -> str:
        return (f"ProducerRecord(account_id={self.account_id!r}, endpoint_url={self.endpoint_url!r}, models={self.models!r}, "
                f"cost_per_1k={self.cost_per_1k!r}, capacity_1k={self.capacity_1k!r}, execution_class={self.execution_class!r})")

    def local_name(self, model_id: str) -> str | None:
        """Return the producer's local name for ``model_id`` or ``None`` if it does not serve it."""
        for public, local in self.models:
            if public == model_id:
                return local
        return None


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
            for sys_id in _ESCROW_ACCOUNTS:
                self._conn.execute(
                    "INSERT OR IGNORE INTO accounts VALUES (?,?,?,?,?,?)",
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
        )

    # ------------------------------------------------------------------ accounts
    def account_id_for(self, handle: str) -> str:
        """Deterministic account id for a handle (after identity normalisation)."""
        return self._hmac_hex(normalise_identity(handle))[:32]

    def register(self, handle: str, role: str) -> tuple[str, str]:
        """Create an account; returns ``(account_id, api_key)``. The key is shown once and never stored."""
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        normalised = normalise_identity(handle)
        if normalised.startswith(RESERVED_HANDLE_PREFIXES):
            raise ReservedHandle(f"handle prefix is reserved: {normalised}")
        return self._create_account(normalised, role)

    def sign_in_with_identity(self, provider: str, subject: str, role: str = "prosumer") -> tuple[str, str, bool]:
        """Create or log in the account that belongs to an identity-provider subject.

        Returns ``(account_id, api_key, created)``. The handle is a keyed hash of ``provider:subject``, so the provider's
        opaque id is not stored in clear and cannot be guessed. A returning user keeps their account, balance and role;
        each sign-in issues a NEW key and the previous one stops working, which doubles as lost-key recovery.
        """
        if provider not in IDENTITY_PROVIDERS:
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
                "INSERT INTO accounts VALUES (?,?,?,?,?,?)",
                (account_id, handle, role, self._hmac_hex(new_key), time.time(), DEFAULT_MAX_PRICE_PER_1K),
            )
        return account_id, new_key, True

    def _create_account(self, normalised: str, role: str) -> tuple[str, str]:
        account_id = self._hmac_hex(normalised)[:32]
        api_key = "hm-" + secrets.token_urlsafe(32)
        with self._tx() as conn:
            exists = conn.execute(
                "SELECT 1 FROM accounts WHERE account_id=? OR handle=?", (account_id, normalised)
            ).fetchone()
            if exists is not None:
                raise HandleTaken(f"handle already registered: {normalised}")
            conn.execute(
                "INSERT INTO accounts VALUES (?,?,?,?,?,?)",
                (account_id, normalised, role, self._hmac_hex(api_key), time.time(), DEFAULT_MAX_PRICE_PER_1K),
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
        """Declared max price of every non-system account that may consume (everything but pure producers)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT account_id, max_price_per_1k FROM accounts WHERE role != 'producer' AND account_id NOT LIKE 'sys:%' "
                "ORDER BY account_id"
            ).fetchall()
        return {str(r["account_id"]): float(r["max_price_per_1k"]) for r in rows}

    def funded_consumer_bids(self, expected_tokens: int) -> dict[str, float]:
        """Effective bids of accounts that can actually pay: ``min(max_price, balance * 1000 / expected_tokens)``.

        Accounts with a non-positive balance are omitted, so anonymous zero-balance sybils cannot move the price,
        and a funded account cannot bid above what its balance could pay for one ``expected_tokens`` request.
        """
        tokens = max(1, int(expected_tokens))
        with self._lock:
            rows = self._conn.execute(
                "SELECT a.account_id AS account_id, a.max_price_per_1k AS price, COALESCE(SUM(l.delta), 0.0) AS bal "
                "FROM accounts a LEFT JOIN ledger l ON l.account_id = a.account_id "
                "WHERE a.role != 'producer' AND a.account_id NOT LIKE 'sys:%' "
                "GROUP BY a.account_id ORDER BY a.account_id"
            ).fetchall()
        bids: dict[str, float] = {}
        for row in rows:
            balance = float(row["bal"])
            if balance > 0.0:
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
    ) -> None:
        """Insert or refresh a producer's advertisement; doubles as the heartbeat.

        ``endpoint_token`` is the bearer secret the gateway presents to the producer's own server so that only the
        gateway can use it; a heartbeat without one clears any earlier token.
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
            conn.execute(
                "INSERT INTO producers(account_id, endpoint_url, models_json, cost_per_1k, capacity_1k, last_heartbeat_utc, "
                "execution_class, endpoint_token) VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(account_id) DO UPDATE SET "
                "endpoint_url=excluded.endpoint_url, models_json=excluded.models_json, cost_per_1k=excluded.cost_per_1k, "
                "capacity_1k=excluded.capacity_1k, last_heartbeat_utc=excluded.last_heartbeat_utc, "
                "execution_class=excluded.execution_class, endpoint_token=excluded.endpoint_token",
                (account_id, endpoint_url, json.dumps(pairs, sort_keys=True), cost, capacity, stamp, execution_class, endpoint_token),
            )

    def active_producers(self, model_id: str, now_utc: float, ttl_s: float = 120) -> list[ProducerRecord]:
        """Producers serving ``model_id`` whose last heartbeat is within ``ttl_s`` seconds of ``now_utc``."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM producers WHERE last_heartbeat_utc >= ? ORDER BY account_id", (now_utc - ttl_s,)
            ).fetchall()
        out: list[ProducerRecord] = []
        for row in rows:
            models = tuple((str(m["id"]), str(m["local_name"])) for m in json.loads(str(row["models_json"])))
            if all(public != model_id for public, _ in models):
                continue
            out.append(
                ProducerRecord(
                    account_id=str(row["account_id"]),
                    endpoint_url=str(row["endpoint_url"]),
                    models=models,
                    cost_per_1k=float(row["cost_per_1k"]),
                    capacity_1k=float(row["capacity_1k"]),
                    last_heartbeat_utc=float(row["last_heartbeat_utc"]),
                    execution_class=str(row["execution_class"]),
                    endpoint_token=None if row["endpoint_token"] is None else str(row["endpoint_token"]),
                )
            )
        return out

    def live_models(self, now_utc: float, ttl_s: float = 120) -> list[str]:
        """Sorted public model ids served by at least one producer whose heartbeat is within ``ttl_s`` seconds."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT models_json FROM producers WHERE last_heartbeat_utc >= ?", (now_utc - ttl_s,)
            ).fetchall()
        found: set[str] = set()
        for row in rows:
            found.update(str(m["id"]) for m in json.loads(str(row["models_json"])))
        return sorted(found)

    def key_digest(self, text: str) -> str:
        """``HMAC-SHA256(secret, text)`` as hex: the digest stored for API keys and expected for the admin key."""
        return self._hmac_hex(text)
