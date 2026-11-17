# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Cross-service invariant behind the dashboard's password check (PRs #904, #905).

The La Ruche dashboard asks the gateway to verify a seeded account's password BY THAT ACCOUNT'S HANDLE, and accepts the
answer when it names one of the account's handles. That is safe only while a stranger cannot get the gateway to accept
the stranger's own password for that handle, e.g. by registering any spelling of it first. This test does exactly that
for every seeded dashboard account (owner, MasterMinders) and fails if it works, which would happen if a seeded account
were added or renamed without its handle being reserved on the gateway.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app
from hivemind.router.accounts import RouterStore

SECRET = b"0123456789abcdef0123456789abcdef"
STRANGER_PASSWORD = "Stranger#Password2026"


def _norm(value: object) -> str:
    return str(value or "").lstrip("@").lower().replace("-", "_")


def _seeded(tmp_path: Path) -> dict[str, tuple[str, set[str]]]:
    """seed key -> (handle the dashboard asks the gateway about, handles it accepts in the answer)."""
    app = create_hivemind_at_home_app(vault_dir=tmp_path / "vault")
    out: dict[str, tuple[str, set[str]]] = {}
    for key, rec in app.state.auth_state["users"].items():
        handle = _norm(rec.get("handle"))
        out[key] = (handle, {handle, _norm(key)})
    return out


def _gateway() -> RouterStore:
    store = RouterStore(":memory:", SECRET)
    store.ensure_legacy_hiveminders_registered()
    return store


def _stranger_gets_in(store: RouterStore, query: str, accepted: set[str]) -> str | None:
    """Register every spelling a stranger could try, then ask the gateway as the dashboard does."""
    spellings = {h for a in accepted for h in (a, a.replace("_", "-"))} | {query, query.replace("_", "-")}
    for spelling in sorted(spellings):
        try:
            store.register_with_credentials(spelling, STRANGER_PASSWORD, email=f"{spelling}.stranger@example.org")
        except Exception:  # noqa: BLE001 - taken, reserved or invalid: the stranger cannot hold this spelling
            continue
    try:
        acct, _key, _created = store.login_with_credentials(query, STRANGER_PASSWORD)
    except Exception:  # noqa: BLE001 - refused: the stranger's password is not accepted for this handle
        return None
    return acct.handle if _norm(acct.handle) in accepted else None


def test_no_stranger_can_get_the_gateway_to_vouch_for_a_seeded_dashboard_account(tmp_path: Path) -> None:
    seeded = _seeded(tmp_path)
    assert len(seeded) >= 3, "expected the owner and MasterMinder seed records"
    store = _gateway()
    open_accounts = {key: got for key, (query, accepted) in seeded.items() if (got := _stranger_gets_in(store, query, accepted))}
    assert not open_accounts, (
        "a stranger can set these seeded dashboard accounts' passwords through the gateway; reserve the handle in "
        f"LEGACY_KNOWN_USERS (handle or aliases): {open_accounts}"
    )


def test_the_check_catches_an_unreserved_handle() -> None:
    """Negative control: for a handle no legacy user reserves, the stranger does get in, so the main test would fail."""
    assert _stranger_gets_in(_gateway(), "unreserved_newcomer_2026", {"unreserved_newcomer_2026"}) == "unreserved_newcomer_2026"


@pytest.mark.parametrize("key", ["xavier_callens", "vxtzq", "jblam62122"])
def test_the_known_admin_seeds_are_covered(tmp_path: Path, key: str) -> None:
    assert key in _seeded(tmp_path)
