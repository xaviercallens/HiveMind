"""Outside contributors on the QLoRA MoE training: the European-night window, trust tiers, host lookup and the full-replay validator."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hivemind.boinc.real_client import boinc_server_scripts as bss  # noqa: E402
from hivemind.boinc.real_client import contributor_policy as cp  # noqa: E402
from hivemind.boinc.real_client import moe_bridge  # noqa: E402
from hivemind.boinc.real_client.moe_app import state_sha  # noqa: E402
from hivemind.boinc.workunit import sign_shard_bytes  # noqa: E402


def utc(y: int, mo: int, d: int, h: int, mi: int = 0) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


def test_night_window_follows_paris_time_including_daylight_saving() -> None:
    # 26 Oct 2026 is winter time (UTC+1): 22:00 Paris = 21:00Z, 07:00 Paris = 06:00Z
    assert cp.in_european_night(utc(2026, 10, 26, 21, 0)) and not cp.in_european_night(utc(2026, 10, 26, 20, 59))
    assert cp.in_european_night(utc(2026, 10, 27, 5, 59)) and not cp.in_european_night(utc(2026, 10, 27, 6, 0))
    # 1 Jul 2026 is summer time (UTC+2): 22:00 Paris = 20:00Z
    assert cp.in_european_night(utc(2026, 7, 1, 20, 0)) and not cp.in_european_night(utc(2026, 7, 1, 19, 59))
    assert not cp.in_european_night(utc(2026, 7, 1, 12, 0))
    with pytest.raises(ValueError):
        cp.in_european_night(datetime(2026, 10, 26, 23, 0))  # naive time: refuse to guess a zone


def test_night_window_covers_nine_hours_a_day() -> None:
    day = [utc(2026, 11, 3, 0) + timedelta(minutes=m) for m in range(0, 24 * 60, 15)]
    assert sum(cp.in_european_night(t) for t in day) * 15 == 9 * 60


def test_policy_tiers() -> None:
    pol = cp.ContributorPolicy(probation_valid_results=5, owner_user_ids=frozenset({1, 2}))
    assert pol.replay_all(None, 999) is True  # unknown account fails closed even with a long history
    assert pol.replay_all(1, 0) is False and pol.replay_all(2, 0) is False  # owner accounts: normal spot check
    assert [pol.replay_all(77, n) for n in (0, 4, 5, 6)] == [True, True, False, False]  # a contributor graduates after 5 validated results


def test_policy_from_env() -> None:
    pol = cp.ContributorPolicy.from_env({"HIVEMIND_OWNER_USER_IDS": "3, 4 x", "HIVEMIND_PROBATION_RESULTS": "2"})
    assert pol.owner_user_ids == frozenset({3, 4}) and pol.probation_valid_results == 2
    assert cp.ContributorPolicy.from_env({}) == cp.ContributorPolicy(5, frozenset())


def test_host_directory_parses_and_fails_closed() -> None:
    seen: list[str] = []
    ok = cp.BoincHostDirectory(lambda q: (seen.append(q), "warn\n7\t3\n")[1])
    assert ok.lookup("41") == (7, 3) and "r.id = 41;" in seen[0]
    assert cp.BoincHostDirectory(lambda q: "").lookup(5) == (None, 0)  # no row
    assert cp.BoincHostDirectory(lambda q: "garbage").lookup(5) == (None, 0)

    def boom(q: str) -> str:
        raise RuntimeError("db down")

    assert cp.BoincHostDirectory(boom).lookup(5) == (None, 0)
    assert cp.BoincHostDirectory(lambda q: seen.append(q) or "1 1").lookup("1; drop table result") == (None, 0)
    assert all("drop" not in q for q in seen)  # a non-numeric id never reaches the query


class FakeDir:
    def __init__(self, user, valid) -> None:
        self.user, self.valid, self.calls = user, valid, 0

    def lookup(self, rid):
        self.calls += 1
        return self.user, self.valid


def test_community_mode_is_opt_in() -> None:
    d = FakeDir(77, 0)
    assert cp.replay_all_for_result(5, d, {}) is False and d.calls == 0  # off: legacy behaviour, no database lookup
    on = {"HIVEMIND_COMMUNITY": "1", "HIVEMIND_OWNER_USER_IDS": "1"}
    assert cp.replay_all_for_result(5, d, on) is True  # new outside contributor
    assert cp.replay_all_for_result(5, FakeDir(1, 0), on) is False  # owner
    assert cp.replay_all_for_result(5, FakeDir(77, 9), on) is False  # graduated
    assert cp.replay_all_for_result(None, d, on) is True and cp.replay_all_for_result(5, None, on) is True  # unknown: fail closed


# ---- validator: replay every segment vs one random segment, with the model replaced by an honest reference

K, SEG = 4, 10


def _result(tmp_path: Path, tamper_segment: int | None):
    """A chain of K+1 states. With tamper_segment=t the claimed end state of segment t is wrong (hashes match what was uploaded)."""
    true = [{"w": torch.full((3,), float(i))} for i in range(K + 1)]
    claimed = [{"w": s["w"].clone()} for s in true]
    if tamper_segment is not None:
        claimed[tamper_segment + 1]["w"] += 5.0
    tokens = [[1, 2, 3, 4]] * 4
    sha, sig = sign_shard_bytes(json.dumps(tokens, separators=(",", ":")).encode(), 7)
    spec = {"wu_id": "hmq_x", "app": "hivemind_qlora_moe", "inner_steps": K * SEG, "segment_steps": SEG, "train_tokens": tokens, "rng_seed": 7,
            "lr": 1e-3, "weight_decay": 0.0, "micro_batch": 1, "shard_sha256": sha, "shard_hmac_signature": sig,
            "start_adapter_sha256": state_sha(claimed[0]), "model_id": "m", "rank": 8, "alpha": 16, "init_seed": 1}
    meta = {"wu_id": "hmq_x", "app": "hivemind_qlora_moe", "state_sha256": [state_sha(s) for s in claimed], "flops_claimed": 1e12, "wall_seconds": 10.0}
    mp, sp = tmp_path / "m.json", tmp_path / "s.pt"
    mp.write_text(json.dumps(meta))
    torch.save({"states": claimed}, sp)
    return spec, mp, sp, true


def _validator(monkeypatch: pytest.MonkeyPatch, true: list[dict], calls: list[int], first_pick: int) -> moe_bridge.MoEValidator:
    v = moe_bridge.MoEValidator()
    monkeypatch.setattr(v, "_ensure", lambda spec: setattr(v, "_init_state", true[0]) or setattr(v, "_model", object()))
    monkeypatch.setattr(v, "_rng", type("R", (), {"randrange": staticmethod(lambda n: first_pick)})())

    def replay(model, start, tokens, a, b, **kw):
        k = a // SEG
        calls.append(k)
        return {"w": true[k + 1]["w"].clone()}  # what an honest replay of segment k produces

    import hivemind.core.diloco_inner as di

    monkeypatch.setattr(di, "replay_inner_segment", replay)
    return v


def test_full_replay_catches_a_bad_segment_that_one_random_segment_misses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec, mp, sp, true = _result(tmp_path, tamper_segment=2)
    calls: list[int] = []
    v = _validator(monkeypatch, true, calls, first_pick=0)  # the random pick lands on a good segment
    single = v.validate_train(spec, mp, sp, "h")
    assert single.valid is True and calls == [0]  # the weakness: one random segment lets the bad result through
    calls.clear()
    full = v.validate_train(spec, mp, sp, "h", replay_all=True)
    assert full.valid is False and calls == [0, 1, 2, 3] and "segment 2" in full.reason
    assert full.detail["full_replay"] is True and full.detail["replayed_segments"] == K and full.detail["overhead_pct"] == 100.0


def test_full_replay_accepts_an_honest_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec, mp, sp, true = _result(tmp_path, tamper_segment=None)
    calls: list[int] = []
    v = _validator(monkeypatch, true, calls, first_pick=1)
    res = v.validate_train(spec, mp, sp, "h", replay_all=True)
    assert res.valid is True and calls == [0, 1, 2, 3] and "verified all 4 segments" in res.reason
    calls.clear()
    legacy = v.validate_train(spec, mp, sp, "h")
    assert legacy.valid is True and calls == [1] and legacy.detail["overhead_pct"] == 25.0 and "verified segment 1/4" in legacy.reason


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeStack:
    def __init__(self, *a, **k) -> None:
        self.root = Path(k.get("root", "."))
        self.dbname = "db"
        self.made: list[str] = []

    def ensure_chain_template(self, app: str = "x") -> str:
        return f"{app}_chain_in"

    def sql(self, q: str) -> str:
        return "0\n"

    def create_work(self, app, name, inputs, fpops, command_line="", template=None) -> None:
        self.made.append(name)


def test_workgen_night_only_creates_nothing_by_day_and_one_unit_by_night(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wg = _load("t4_workgen")
    made: list[str] = []
    monkeypatch.setattr(wg, "Stack", lambda *a, **k: type("S", (_FakeStack,), {"create_work": lambda self, app, name, *r, **kw: made.append(name)})(root=tmp_path / "stack"))
    monkeypatch.setattr(wg, "build_train_spec", lambda name, **k: {"name": name})
    (tmp_path / "stack" / "hivemind_specs").mkdir(parents=True)
    argv = ["t4_workgen.py", "--night-only", "--once", "--stack-base", str(tmp_path), "--boinc-src", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(wg, "_now", lambda: utc(2026, 10, 27, 12, 0))  # 13:00 Paris
    assert wg.main() == 0 and made == []
    monkeypatch.setattr(wg, "_now", lambda: utc(2026, 10, 27, 23, 30))  # 00:30 Paris
    assert wg.main() == 0 and made == ["hmq_00001"]
    made.clear()
    monkeypatch.setattr(sys, "argv", [a for a in argv if a != "--night-only"])
    monkeypatch.setattr(wg, "_now", lambda: utc(2026, 10, 27, 12, 0))
    assert wg.main() == 0 and made == ["hmq_00002"]  # without the flag the window does not apply (existing behaviour)


def test_scripts_host_directory_needs_both_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIVEMIND_STACK_BASE", raising=False)
    monkeypatch.delenv("HIVEMIND_BOINC_SRC", raising=False)
    assert bss._host_directory() is None
