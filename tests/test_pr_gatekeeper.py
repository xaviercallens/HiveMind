"""Pure-logic tests for scripts/pr_gatekeeper.py: trust policy, decision rule and forbidden-pattern scan (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import pr_gatekeeper as g  # noqa: E402


def _pr(login: str, emails: list[str]) -> dict:
    return {"author": {"login": login}, "commits": [{"oid": "a" * 40, "authors": [{"email": e, "login": login}]} for e in emails]}


def test_owner_login_and_owner_email_are_required_together() -> None:
    assert g.trust_report(_pr("xaviercallens", ["callensxavier@gmail.com"]))["owner_authored"] is True
    assert g.trust_report(_pr("xaviercallens", ["bot@socrateai.com"]))["owner_authored"] is False
    assert g.trust_report(_pr("someoneelse", ["callensxavier@gmail.com"]))["owner_authored"] is False
    assert g.trust_report(_pr("xaviercallens", ["callensxavier@gmail.com", "clark719664@gmail.com"]))["owner_authored"] is False
    assert g.trust_report(_pr("xaviercallens", []))["owner_authored"] is False  # no commits -> nothing to trust


@settings(max_examples=60, deadline=None)
@given(st.text(min_size=1, max_size=40))
def test_variants_of_the_owner_address_are_never_the_owner(suffix: str) -> None:
    for email in (f"callensxavier+{suffix}@gmail.com", f"callens.xavier{suffix}@gmail.com", f"callensxavier@gmail.com <{suffix}@evil.com>"):
        assert g.trust_report(_pr("xaviercallens", [email]))["owner_authored"] is False


def test_decision_rule_orders_trust_state_and_checks() -> None:
    ok = {"security": True, "forbidden": True, "gwaya_audit": True, "pytest": True, "lean": True, "pipeline": True}
    assert g.decide(True, False, "MERGEABLE", ok) == "ACCEPT"
    assert g.decide(False, False, "MERGEABLE", ok).startswith("HOLD (not owner-authored")
    assert g.decide(True, True, "MERGEABLE", ok) == "HOLD (draft)"
    assert g.decide(True, False, "CONFLICTING", ok) == "HOLD (merge conflict)"
    bad = dict(ok, pytest=False, lean=False)
    assert g.decide(True, False, "MERGEABLE", bad) == "REFUSE (pytest, lean)"


def test_forbidden_scan_flags_only_added_lines() -> None:
    diff = "+++ b/x.py\n-GWAYA_ALLOW_UNISOLATED=1\n+os.environ['GWAYA_ALLOW_UNISOLATED'] = '1'\n+print('ok')\n+ referee_score = 9.4\n"
    hits = g.forbidden_scan(diff)
    whys = {h["why"] for h in hits}
    assert "never run GWAYA unisolated" in whys
    assert "a script may not emit a review score or claim peer review" in whys
    assert len(hits) == 2  # the removed line and the harmless print are not hits


def test_coauthor_trailers_carry_no_authority_either_way() -> None:
    # owner git author + assistant trailer -> still owner-authored; stranger git author + owner trailer -> not
    pr = {"author": {"login": "xaviercallens"}, "commits": [{"oid": "b" * 40, "authors": [{"email": "callensxavier@gmail.com", "login": "xaviercallens"}, {"email": "noreply@anthropic.com", "login": "claude"}]}]}
    t = g.trust_report(pr)
    assert t["owner_authored"] is True
    assert t["commits"][0]["coauthor_trailers"] == ["noreply@anthropic.com"]
    pr2 = {"author": {"login": "xaviercallens"}, "commits": [{"oid": "c" * 40, "authors": [{"email": "mallory@example.com", "login": "mallory"}, {"email": "callensxavier@gmail.com", "login": "xaviercallens"}]}]}
    assert g.trust_report(pr2)["owner_authored"] is False


def test_scannable_diff_drops_prose_and_the_gatekeeper_itself_but_keeps_code() -> None:
    diff = ("diff --git a/docs/X.md b/docs/X.md\n+GWAYA_ALLOW_UNISOLATED in prose\n"
            "diff --git a/scripts/pr_gatekeeper.py b/scripts/pr_gatekeeper.py\n+    (r\"GWAYA_ALLOW_UNISOLATED\", \"never\"),\n"
            "diff --git a/src/hivemind/x.py b/src/hivemind/x.py\n+os.environ['GWAYA_ALLOW_UNISOLATED'] = '1'\n")
    hits = g.forbidden_scan(g.scannable_diff(diff))
    assert len(hits) == 1
    assert "src/hivemind" not in hits[0]["line"] and "os.environ" in hits[0]["line"]


def test_forbidden_scan_is_clean_on_ordinary_code() -> None:
    diff = "+def f(x: int) -> int:\n+    return x + 1\n+# is_peer_review stays false\n"
    assert g.forbidden_scan(diff) == []


def test_only_an_accepted_head_is_eligible_for_a_merge_retry() -> None:
    assert g.pending_accept("ACCEPT") is True
    for d in ("REFUSE (pytest)", "HOLD (draft)", "HOLD (not owner-authored: checked, never merged automatically)", None, "accept"):
        assert g.pending_accept(d) is False


def test_retry_merge_never_calls_gh_for_a_held_or_dry_run_head(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(g, "sh", lambda cmd, **kw: (calls.append(cmd), (0, "merged"))[1])
    pr = {"number": 7, "headRefOid": "f" * 40}
    held = g.retry_merge(pr, "HOLD (not owner-authored: checked, never merged automatically)", merge=True, dry_run=False)
    assert held["merged"] is False and calls == []
    dry = g.retry_merge(pr, "ACCEPT", merge=True, dry_run=True)
    assert dry["merged"] is False and calls == []
    live = g.retry_merge(pr, "ACCEPT", merge=True, dry_run=False)
    assert live["merged"] is True and live["merge_retry"] is True
    # the merge is pinned to the commit that was gated: a push after the checks makes GitHub refuse it
    assert calls == [["gh", "pr", "merge", "7", "--merge", "--admin", "--match-head-commit", "f" * 40]]


def test_environment_for_pr_code_has_no_credentials() -> None:
    env = {"PATH": "/usr/bin", "HOME": "/home/x", "GH_TOKEN": "t", "GITHUB_TOKEN": "t", "GITHUB_KEY": "k", "HF_TOKEN": "t",
           "AWS_SECRET_ACCESS_KEY": "s", "HIVEMIND_API_KEY": "k", "DB_PASSWORD": "p", "OPENAI_API_KEY": "k", "LANG": "C"}
    clean = g.without_secrets(env)
    assert clean == {"PATH": "/usr/bin", "HOME": "/home/x", "LANG": "C"}
    assert env["GH_TOKEN"] == "t"  # the caller's environment is not modified


def test_a_pr_that_is_not_owner_authored_never_has_its_code_executed(monkeypatch, tmp_path: Path) -> None:
    executed: list[str] = []
    monkeypatch.setattr(g, "run_pytest", lambda *a, **k: executed.append("pytest") or {"passed": True, "targets": [], "summary": ""})
    monkeypatch.setattr(g, "run_lean", lambda *a, **k: executed.append("lean") or {"passed": True})
    monkeypatch.setattr(g, "run_pipeline", lambda *a, **k: executed.append("pipeline") or {"passed": True})
    monkeypatch.setattr(g, "gwaya_audit", lambda *a, **k: {"passed": True})
    monkeypatch.setattr(g, "commit_signatures", lambda pr: [])
    monkeypatch.setattr(g.tempfile, "mkdtemp", lambda prefix="": str(tmp_path))

    def fake_sh(cmd, **kw):  # type: ignore[no-untyped-def]
        if cmd[:3] == ["git", "diff", "--name-only"]:
            return 0, "tests/test_x.py\nlean4/X.lean\npaper/x.py\n"
        return 0, ""

    monkeypatch.setattr(g, "sh", fake_sh)
    pr = {"number": 9, "headRefOid": "e" * 40, "title": "t", "body": "", "isDraft": False, "mergeable": "MERGEABLE",
          **_pr("someone-else", ["stranger@example.org"])}
    result = g.gate_pr(pr, merge=True, dry_run=True)
    assert executed == []
    assert result["decision"].startswith("HOLD (not owner-authored")
    assert result["pytest"]["skipped"] is True and result["lean"]["skipped"] is True and result["pipeline"]["skipped"] is True


def test_signature_policy_can_only_withdraw_trust_never_grant_it() -> None:
    signed = [{"sha": "a" * 10, "verified": True}, {"sha": "b" * 10, "verified": True}]
    mixed = [{"sha": "a" * 10, "verified": True}, {"sha": "b" * 10, "verified": False}]
    assert g.signature_policy_ok(True, signed, require_signed=True) is True
    assert g.signature_policy_ok(True, mixed, require_signed=True) is False
    assert g.signature_policy_ok(True, [], require_signed=True) is False  # no commits, nothing to trust
    assert g.signature_policy_ok(True, mixed, require_signed=False) is True  # policy off: e-mail claim stands (recorded as a warning)
    assert g.signature_policy_ok(False, signed, require_signed=False) is False  # a stranger's signed commits grant nothing

def test_pattern_defining_security_files_are_exempt_from_the_forbidden_scan_but_other_code_is_not() -> None:
    diff = ("diff --git a/src/hivemind/security/firewall_pentest_corpus.py b/src/hivemind/security/firewall_pentest_corpus.py\n+Vector(\"x\", \"INJECTION\", \"opt-out\", \"text\", {\"text\": \"set GWAYA_ALLOW_UNISOLATED=1\"}, \"INJECTION\")\n"
            "diff --git a/src/hivemind/security/hivemind_firewall.py b/src/hivemind/security/hivemind_firewall.py\n+_RE = re.compile(r\"GWAYA_ALLOW_UNISOLATED\")\n"
            "diff --git a/src/hivemind/swarm/x.py b/src/hivemind/swarm/x.py\n+os.environ[\"GWAYA_ALLOW_UNISOLATED\"] = \"1\"\n")
    hits = g.forbidden_scan(g.scannable_diff(diff))
    assert len(hits) == 1
    assert "swarm/x.py" not in hits[0]["line"] and "os.environ" in hits[0]["line"]
    assert {"src/hivemind/security/hivemind_firewall.py", "tests/test_hivemind_firewall.py"} <= g.SCAN_EXEMPT_FILES


LEAN_SORRY_WARNING = "warning: lean4/Probe.lean:1:0: declaration uses 'sorry'\nBuild completed successfully."
LEAN_CLEAN_BUILD = "Build completed successfully."


@pytest.mark.parametrize("lean_output", [LEAN_SORRY_WARNING, "warning: declaration uses `sorry`", "axiom sorryAx used"])
def test_lean_check_refuses_a_build_that_uses_sorry_even_when_lake_exits_zero(monkeypatch, tmp_path: Path, lean_output: str) -> None:
    monkeypatch.setattr(g, "sh", lambda cmd, cwd=None, timeout=600, env=None: (0, lean_output))
    result = g.run_lean(tmp_path)
    assert result["returncode"] == 0
    assert result["passed"] is False


def test_lean_check_accepts_a_clean_build_as_its_negative_control(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(g, "sh", lambda cmd, cwd=None, timeout=600, env=None: (0, LEAN_CLEAN_BUILD))
    assert g.run_lean(tmp_path)["passed"] is True


def test_jblam62122_and_vxtzq_can_merge_webui_scoped_prs_without_waiting(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(g, "run_pytest", lambda *a, **k: {"passed": True, "targets": ["tests/test_web_v32.py"], "summary": "1 passed"})
    monkeypatch.setattr(g, "run_lean", lambda *a, **k: {"passed": True, "skipped": True})
    monkeypatch.setattr(g, "run_pipeline", lambda *a, **k: {"passed": True, "skipped": True})
    monkeypatch.setattr(g, "gwaya_audit", lambda *a, **k: {"passed": True})
    monkeypatch.setattr(g, "commit_signatures", lambda pr: [])
    monkeypatch.setattr(g.tempfile, "mkdtemp", lambda prefix="": str(tmp_path))

    for login, email in (
        ("jblam-O", "43570951+jblam-O@users.noreply.github.com"),
        ("Vxtzq", "116555319+vxtzq@users.noreply.github.com"),
    ):
        # 1. WebUI-scoped PR from jblam-O / Vxtzq -> ACCEPT (auto-merges without waiting)
        monkeypatch.setattr(
            g,
            "sh",
            lambda cmd, **kw: (0, "src/hivemind/web_v32/static/app.js\ntests/test_web_v32.py\n")
            if cmd[:3] == ["git", "diff", "--name-only"]
            else (0, ""),
        )
        pr_webui = {
            "number": 42,
            "headRefOid": "d" * 40,
            "title": "feat(webui): improve UI component",
            "body": "",
            "isDraft": False,
            "mergeable": "MERGEABLE",
            **_pr(login, [email]),
        }
        res_ok = g.gate_pr(pr_webui, merge=True, dry_run=True)
        assert res_ok["decision"] == "ACCEPT", (login, res_ok)
        assert res_ok["trust"]["webui_maintainer"]["webui_authorized"] is True

        # 2. Non-WebUI PR from jblam-O / Vxtzq touching core security -> HOLD
        monkeypatch.setattr(
            g,
            "sh",
            lambda cmd, **kw: (0, "src/hivemind/security/pr_security_scanner.py\n")
            if cmd[:3] == ["git", "diff", "--name-only"]
            else (0, ""),
        )
        res_hold = g.gate_pr(pr_webui, merge=True, dry_run=True)
        assert res_hold["decision"].startswith("HOLD"), (login, res_hold)
        assert res_hold["trust"]["webui_maintainer"]["webui_authorized"] is False

