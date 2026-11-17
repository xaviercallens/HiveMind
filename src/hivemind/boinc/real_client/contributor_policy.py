# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Who may contribute to the QLoRA MoE training, when, and how closely their results are checked.

Outside contributors (any BOINC host whose account is not an owner account) can train for the project. Three rules keep that safe:

- **When**: community work is only offered during the European night (22:00-07:00 Europe/Paris), when end-user inference on the shared
  GPU is lowest; the work generator can be told to create work only inside that window.
- **How closely**: the validator normally replays ONE random segment of a result. A new contributor's result is replayed in FULL
  (every segment) until that host has ``probation_valid_results`` validated results, because a single random segment lets a result
  that is wrong in the other segments through. Owner accounts keep the single-segment check.
- **Fail closed**: if the host or account of a result cannot be determined, the result gets the full replay.

Community mode is opt-in (``HIVEMIND_COMMUNITY=1``); without it the validator behaves exactly as before. No heavy imports here.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Mapping, Protocol
from zoneinfo import ZoneInfo

NIGHT_TZ, NIGHT_START_H, NIGHT_END_H = "Europe/Paris", 22, 7
DEFAULT_PROBATION = 5


def in_european_night(now: datetime, tz: str = NIGHT_TZ, start_h: int = NIGHT_START_H, end_h: int = NIGHT_END_H) -> bool:
    """True when ``now`` falls in [start_h, end_h) local time of ``tz`` (the window wraps midnight). ``now`` must be timezone-aware."""
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if not isinstance(tz, str) or not tz.strip():
        raise ValueError("tz must be a non-empty timezone name")
    if isinstance(start_h, bool) or not isinstance(start_h, int) or not (0 <= start_h <= 23):
        raise ValueError("start_h and end_h must be integers in [0, 23]")
    if isinstance(end_h, bool) or not isinstance(end_h, int) or not (0 <= end_h <= 23):
        raise ValueError("start_h and end_h must be integers in [0, 23]")
    h = now.astimezone(ZoneInfo(tz)).hour
    return start_h <= h < end_h if start_h <= end_h else (h >= start_h or h < end_h)


@dataclass(frozen=True)
class ContributorPolicy:
    probation_valid_results: int = DEFAULT_PROBATION
    owner_user_ids: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        if isinstance(self.probation_valid_results, bool) or not isinstance(self.probation_valid_results, int) or self.probation_valid_results < 0:
            raise ValueError("probation_valid_results must be a non-negative integer")
        if not all(isinstance(u, int) and not isinstance(u, bool) and u >= 0 for u in self.owner_user_ids):
            raise ValueError("owner_user_ids must contain non-negative integers")

    def replay_all(self, user_id: int | None, valid_results: int) -> bool:
        """Whether every segment of this result must be replayed."""
        if isinstance(valid_results, bool) or not isinstance(valid_results, int) or valid_results < 0:
            raise ValueError("valid_results must be a non-negative integer")
        if user_id is None:  # unknown account: fail closed
            return True
        if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id < 0:
            raise ValueError("user_id must be None or a non-negative integer")
        if user_id in self.owner_user_ids:
            return False
        return valid_results < self.probation_valid_results

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ContributorPolicy":
        env = os.environ if env is None else env
        ids = frozenset(int(x) for x in re.split(r"[,\s]+", env.get("HIVEMIND_OWNER_USER_IDS", "").strip()) if x.isdigit())
        probation = env.get("HIVEMIND_PROBATION_RESULTS", "")
        return cls(int(probation) if probation.isdigit() else DEFAULT_PROBATION, ids)


class HostDirectory(Protocol):
    def lookup(self, result_id: int | str) -> tuple[int | None, int]:
        """(BOINC user id, validated results of the host that returned the result); (None, 0) when unknown."""
        return None, 0


class BoincHostDirectory:
    """Looks a result up in the project database through ``sql`` (the work generator's ``Stack.sql``: a query string in, tab text out)."""

    def __init__(self, sql: Callable[[str], str]) -> None:
        if not callable(sql):
            raise ValueError("sql must be callable")
        self._sql = sql

    def lookup(self, result_id: int | str) -> tuple[int | None, int]:
        if isinstance(result_id, bool):
            return None, 0
        try:
            rid = int(result_id)  # int() also refuses anything that is not a plain number, so nothing else reaches the query
            out = self._sql(
                "select h.userid, (select count(*) from result r2 where r2.hostid = h.id and r2.validate_state = 1) "
                f"from result r join host h on h.id = r.hostid where r.id = {rid};"
            )
            user, valid = out.strip().splitlines()[-1].split()[:2]
            return int(user), int(valid)
        except Exception:  # no row, bad output, database down: unknown, which the policy treats as untrusted
            return None, 0

    def result_workunit_and_elapsed(self, result_id: int | str) -> tuple[str | None, float | None]:
        """Return ``(workunit_name, server_elapsed_seconds)`` for ``result_id`` by joining ``result`` to ``workunit``."""
        if isinstance(result_id, bool):
            return None, None
        try:
            rid = int(result_id)
            out = self._sql(
                "select w.name, greatest(r.received_time - r.sent_time, 0) "
                f"from result r join workunit w on w.id = r.workunitid where r.id = {rid};"
            )
            parts = out.strip().splitlines()[-1].split()
            if not parts:
                return None, None
            wu_name = parts[0].strip() or None
            elapsed = float(parts[1]) if len(parts) >= 2 else None
            return wu_name, (elapsed if elapsed is not None and elapsed > 0.0 else None)
        except Exception:
            return None, None

    def workunit_name(self, wu_db_id: int | str) -> str | None:
        """Return the canonical BOINC ``workunit.name`` for ``wu_db_id``."""
        if isinstance(wu_db_id, bool):
            return None
        try:
            wid = int(wu_db_id)
            out = self._sql(f"select w.name from workunit w where w.id = {wid};")
            name = out.strip().splitlines()[-1].split()[0].strip()
            return name or None
        except Exception:
            return None


def replay_all_for_result(result_id: int | str | None, directory: HostDirectory | None, env: Mapping[str, str] | None = None) -> bool:
    """Full replay for this result? False unless community mode is on (``HIVEMIND_COMMUNITY=1``), so existing behaviour is unchanged."""
    if result_id is not None and (isinstance(result_id, bool) or not isinstance(result_id, (int, str))):
        raise ValueError("result_id must be None, int, or str")
    if env is not None and not isinstance(env, Mapping):
        raise TypeError("env must be None or a Mapping")
    env = os.environ if env is None else env
    if env.get("HIVEMIND_COMMUNITY") != "1":
        return False
    if result_id is None or directory is None:
        return True
    user, valid = directory.lookup(result_id)
    return ContributorPolicy.from_env(env).replay_all(user, valid)
