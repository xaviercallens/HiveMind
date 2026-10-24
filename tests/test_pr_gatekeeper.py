"""Pure-logic tests for scripts/pr_gatekeeper.py: trust policy, decision rule and forbidden-pattern scan (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

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
    assert calls == [["gh", "pr", "merge", "7", "--merge", "--admin"]]


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
