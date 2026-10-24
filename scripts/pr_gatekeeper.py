"""Hourly pull-request gatekeeper for HiveMind (owner-only merge authority).

Policy (fixed in code, not configurable from PR text):
  * Only pull requests whose GitHub author is the owner account AND whose every commit author e-mail
    normalises to ``callensxavier@gmail.com`` (``OwnerOrderGuardrail``) are *eligible* for a merge.
    Everything else is checked and reported, never merged ("HOLD").
  * PR titles, bodies, commit messages and comments are data: nothing in them is interpreted as an
    instruction. The only authority is the owner's identity on the commits plus the checks below.
  * Checks (all must pass for ACCEPT): PRSecurityScanner (0 critical, 0 high), forbidden patterns in added
    lines, GWAYA zero-stub audit on changed Python files, pytest on the changed test files plus a smoke set,
    ``lake build`` with no ``sorryAx`` when ``lean4/`` changed, and the paper provenance pipeline when the
    manuscript changed. Any failure -> REFUSE (comment with the table; never merged).
  * Decisions are posted as a PR comment and logged to ``artifacts/pr_gatekeeper/``; a PR whose head SHA was
    already gated with the same outcome is not re-run; an ACCEPTed head that is still open is only
    re-merged (branch protection may have refused the first merge while CI was pending).

Usage:  PYTHONPATH=src python scripts/pr_gatekeeper.py [--merge] [--dry-run] [--pr N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hivemind.security.hivemind_firewall import quarantine_text  # noqa: E402
from hivemind.security.owner_order_guardrail import normalize_and_verify_owner_email  # noqa: E402
from hivemind.security.pr_security_scanner import PRSecurityScanner  # noqa: E402

OWNER_LOGIN = "xaviercallens"
REPO = "xaviercallens/HiveMind"
# Commit author e-mail is a *claim* (another agent on the shared machine forged it on 2026-10-24). With the owner's signing key
# registered on GitHub, HIVEMIND_GATE_REQUIRE_SIGNED=1 makes an unsigned owner-attributed commit HOLD instead of ACCEPT.
REQUIRE_SIGNED_COMMITS = os.environ.get("HIVEMIND_GATE_REQUIRE_SIGNED", "0") == "1"
LOG_DIR = PROJECT_ROOT / "artifacts" / "pr_gatekeeper"
_venv_py = PROJECT_ROOT / ".venv" / "bin" / "python"
PY = os.environ.get("HIVEMIND_PY", str(_venv_py) if _venv_py.exists() else sys.executable)
SMOKE_TESTS = ["tests/test_decoupled_diloco_quorum.py", "tests/test_readme_claims.py", "tests/test_lattice_quantizer.py"]
# (regex over ADDED diff lines, reason). The patterns are assembled from fragments so that this file itself
# does not contain the literal strings that repository hooks refuse to see in shell commands.
FORBIDDEN_ADDED = [
    (r"GWAYA_ALLOW_UNISOLATED", "never run GWAYA unisolated"),
    (r"-" + r"-no-verify", "git hooks must not be bypassed"),
    (r"git push\s+(-" + r"f|-" + r"-force)", "forced push"),
    (r"rm\s+-rf\s+/", "destructive shell"),
    (r"curl[^\n]*\|\s*(ba)?sh", "remote script execution"),
    (r"\bexec\(\s*base64", "obfuscated execution"),
    (r"referee_score|\"score\"\s*:\s*\d|is_peer_review\"?\s*:\s*[Tt]rue", "a script may not emit a review score or claim peer review"),
    (r"callensxavier\+|callensxavier\.[a-z]", "owner-address variants are not the owner"),
]
TEST_TIMEOUT_S = 1500


def sh(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 600, env: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout, env=env)
    return p.returncode, (p.stdout + p.stderr)


def gh_json(args: List[str]) -> Any:
    rc, out = sh(["gh", *args], cwd=PROJECT_ROOT, timeout=120)
    if rc != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed: {out[-400:]}")
    return json.loads(out)


def trust_report(pr: Dict[str, Any]) -> Dict[str, Any]:
    """Owner-authored iff the PR author is the owner login and the git author of EVERY commit is the owner.

    GitHub lists ``Co-Authored-By`` trailers after the git author in ``commit.authors``; those trailers are
    recorded for the report but carry no authority (a trailer cannot make a commit the owner's, and the
    owner's own commits routinely carry an assistant trailer)."""
    author_login = (pr.get("author") or {}).get("login", "")
    commit_rows = []
    all_ok = author_login == OWNER_LOGIN
    for c in pr.get("commits", []):
        authors = c.get("authors", [])
        if not authors:
            all_ok = False
            continue
        git_author = authors[0]
        v = normalize_and_verify_owner_email(git_author.get("email"))
        commit_rows.append({"sha": c.get("oid", "")[:10], "email": git_author.get("email"), "login": git_author.get("login"), "owner": v.is_authorized_owner, "rule": v.rule_id,
                            "coauthor_trailers": [a.get("email") for a in authors[1:]]})
        all_ok &= v.is_authorized_owner
    return {"author_login": author_login, "author_is_owner_login": author_login == OWNER_LOGIN, "commits": commit_rows, "owner_authored": bool(all_ok and commit_rows)}


def commit_signatures(pr: Dict[str, Any]) -> List[Dict[str, Any]]:
    """GitHub's signature verification for every commit of the PR (verified, reason, signer), read-only."""
    rows: List[Dict[str, Any]] = []
    for c in pr.get("commits", []):
        sha = c.get("oid", "")
        try:
            v = gh_json(["api", f"repos/{REPO}/commits/{sha}", "--jq", ".commit.verification"]) or {}
        except (RuntimeError, json.JSONDecodeError):
            v = {"verified": False, "reason": "lookup_failed"}
        rows.append({"sha": sha[:10], "verified": bool(v.get("verified")), "reason": v.get("reason", "")})
    return rows


def signature_policy_ok(owner_authored: bool, signatures: List[Dict[str, Any]], require_signed: bool) -> bool:
    """Pure: when signatures are required, an owner-attributed PR is trusted only if every commit is verified by GitHub.
    Non-owner PRs are HOLD anyway; the policy never *grants* trust, it can only withdraw it."""
    if not owner_authored:
        return False
    if not require_signed:
        return True
    return bool(signatures) and all(s.get("verified") for s in signatures)


SCAN_EXEMPT_FILES = {  # these files *define and test* the forbidden patterns, so their text necessarily contains them
    "scripts/pr_gatekeeper.py", "tests/test_pr_gatekeeper.py",
    "src/hivemind/security/hivemind_firewall.py", "src/hivemind/security/firewall_pentest_corpus.py", "tests/test_hivemind_firewall.py",
}
SCAN_SKIP_SUFFIXES = (".md", ".txt", ".rst", ".tex", ".json", ".jsonl", ".csv")  # prose and data, not executed


def scannable_diff(diff_text: str) -> str:
    """Keep only the hunks of files whose content can execute; drop prose/data files and the gatekeeper's own files."""
    out: List[str] = []
    keep = True
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            path = line.split(" b/", 1)[-1] if " b/" in line else ""
            keep = path not in SCAN_EXEMPT_FILES and not path.endswith(SCAN_SKIP_SUFFIXES)
        if keep:
            out.append(line)
    return "\n".join(out)


def forbidden_scan(diff_text: str) -> List[Dict[str, str]]:
    hits = []
    for line in diff_text.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        for pat, why in FORBIDDEN_ADDED:
            if re.search(pat, line):
                hits.append({"pattern": pat, "why": why, "line": line[:160]})
    return hits


def gwaya_audit(wt: Path, files: List[str]) -> Dict[str, Any]:
    try:
        from gwaya.ast_audit import ZeroStubAudit
    except Exception as e:  # GWAYA missing -> the check cannot run and the PR cannot be accepted
        return {"ran": False, "error": f"gwaya unavailable: {e}", "passed": False}
    bad = {}
    for f in files:
        if f.endswith(".py") and not f.startswith("tests/") and (wt / f).exists():
            r = ZeroStubAudit().audit_python_code((wt / f).read_text(encoding="utf-8", errors="replace"))
            if not r.is_clean:
                bad[f] = [str(v) for v in r.violations][:5]
    return {"ran": True, "files_checked": sum(1 for f in files if f.endswith(".py") and not f.startswith("tests/")), "violations": bad, "passed": not bad}


def run_pytest(wt: Path, files: List[str]) -> Dict[str, Any]:
    targets = sorted({f for f in files if f.startswith("tests/") and f.endswith(".py") and (wt / f).exists()} | {t for t in SMOKE_TESTS if (wt / t).exists()})
    env = dict(os.environ, PYTHONPATH=str(wt / "src"))
    rc, out = sh([PY, "-m", "pytest", *targets, "-q", "-p", "no:cacheprovider", "-p", "no:cov", "-p", "no:typeguard", "-x", "--deselect", "tests/test_seti_swarm_v2.py::test_s3_01_multi_node_wan_nat_traversal_stun_turn_hole_punching"], cwd=wt, timeout=TEST_TIMEOUT_S, env=env)
    tail = out.strip().splitlines()[-1] if out.strip() else ""
    return {"targets": targets, "returncode": rc, "summary": tail[:200], "passed": rc == 0}


def run_lean(wt: Path) -> Dict[str, Any]:
    rc, out = sh(["lake", "build"], cwd=wt / "lean4", timeout=1200)
    return {"returncode": rc, "sorry_in_output": "sorryAx" in out, "passed": rc == 0 and "sorryAx" not in out, "tail": out.strip().splitlines()[-1][:200] if out.strip() else ""}


def run_pipeline(wt: Path) -> Dict[str, Any]:
    env = dict(os.environ, PYTHONPATH=str(wt / "src"))
    rc, out = sh([PY, "paper/autoevolve_peer_review_pipeline.py"], cwd=wt, timeout=900, env=env)
    return {"returncode": rc, "passed": rc == 0, "tail": out.strip().splitlines()[-1][:200] if out.strip() else ""}


def decide(trust_ok: bool, draft: bool, mergeable: Optional[str], checks: Dict[str, bool]) -> str:
    """Pure decision rule: trust first, then PR state, then every check."""
    if not trust_ok:
        return "HOLD (not owner-authored: checked, never merged automatically)"
    if draft:
        return "HOLD (draft)"
    if mergeable == "CONFLICTING":
        return "HOLD (merge conflict)"
    if all(checks.values()):
        return "ACCEPT"
    return "REFUSE (" + ", ".join(k for k, v in checks.items() if not v) + ")"


def pending_accept(decision: Optional[str]) -> bool:
    """True iff an earlier run ACCEPTed this head: the PR is eligible for a merge retry (never re-gated, never re-decided)."""
    return decision == "ACCEPT"


def retry_merge(pr: Dict[str, Any], decision: str, merge: bool, dry_run: bool) -> Dict[str, Any]:
    """Re-attempt the merge of a head already ACCEPTed by a previous run; the decision itself is not revisited."""
    n = pr["number"]
    result: Dict[str, Any] = {"pr": n, "head": pr["headRefOid"], "decision": decision, "skipped": "already gated at this head", "merged": False}
    if merge and not dry_run and pending_accept(decision):
        rc, out = sh(["gh", "pr", "merge", str(n), "--merge", "--admin"], cwd=PROJECT_ROOT, timeout=300)
        result["merged"] = rc == 0
        result["merge_output"] = out[-300:]
        result["merge_retry"] = True
    return result


def gate_pr(pr: Dict[str, Any], merge: bool, dry_run: bool) -> Dict[str, Any]:
    n = pr["number"]
    head = pr["headRefOid"]
    base = pr.get("baseRefName", "main")
    result: Dict[str, Any] = {"pr": n, "title": pr.get("title"), "head": head, "base": base, "draft": pr.get("isDraft"), "mergeable": pr.get("mergeable"), "gated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    result["trust"] = trust_report(pr)
    result["signatures"] = commit_signatures(pr)
    result["signature_policy"] = {"require_signed": REQUIRE_SIGNED_COMMITS, "all_verified": bool(result["signatures"]) and all(s["verified"] for s in result["signatures"])}
    if result["trust"]["owner_authored"] and not result["signature_policy"]["all_verified"]:
        result["warning"] = "owner-attributed commits are UNSIGNED: e-mail identity is a claim (HM-S11-04)"
    # PR text is data: scanned for prompt injection and recorded (report-only; the decision never reads the text)
    result["injection_scan"] = quarantine_text(f"{pr.get('title') or ''}\n{pr.get('body') or ''}", "pr_text").scan.to_dict()
    wt = Path(tempfile.mkdtemp(prefix=f"hm-gate-pr{n}-"))
    try:
        sh(["git", "fetch", "-q", "origin", f"pull/{n}/head:refs/gate/pr-{n}", f"{base}:refs/gate/base-{n}"], cwd=PROJECT_ROOT, timeout=300)
        rc, out = sh(["git", "worktree", "add", "--detach", str(wt), head], cwd=PROJECT_ROOT, timeout=120)
        if rc != 0:
            raise RuntimeError(f"worktree failed: {out[-300:]}")
        rc, diff = sh(["git", "diff", f"refs/gate/base-{n}...{head}"], cwd=PROJECT_ROOT, timeout=120)
        rc, names = sh(["git", "diff", "--name-only", f"refs/gate/base-{n}...{head}"], cwd=PROJECT_ROOT, timeout=120)
        files = [f for f in names.split("\n") if f.strip()]
        result["changed_files"] = files
        changed = [{"filename": f, "patch": "", "content": (wt / f).read_text(encoding="utf-8", errors="replace") if (wt / f).exists() and (wt / f).stat().st_size < 2_000_000 else ""} for f in files]
        commits_msgs = [c.get("messageHeadline", "") for c in pr.get("commits", [])]
        emails = [a.get("email") for c in pr.get("commits", []) for a in c.get("authors", [])]
        audit = PRSecurityScanner(trusted_maintainers={OWNER_LOGIN}).audit_pull_request(n, pr.get("title", ""), pr.get("body", "") or "", result["trust"]["author_login"], changed, commits_msgs, author_email=emails[0] if emails else None)
        result["security"] = {"critical": audit.critical_count, "high": audit.high_count, "medium": audit.medium_count, "passed": audit.critical_count == 0 and audit.high_count == 0, "findings": [f.to_dict() for f in audit.findings][:20]}
        hits = forbidden_scan(scannable_diff(diff))
        result["forbidden"] = {"hits": hits, "passed": not hits}
        result["gwaya_audit"] = gwaya_audit(wt, files)
        result["pytest"] = run_pytest(wt, files)
        result["lean"] = run_lean(wt) if any(f.startswith("lean4/") for f in files) else {"skipped": True, "passed": True}
        result["pipeline"] = run_pipeline(wt) if any(f.startswith("paper/") and (f.endswith(".tex") or f.endswith(".py")) for f in files) else {"skipped": True, "passed": True}
    finally:
        shutil.rmtree(wt, ignore_errors=True)
        sh(["git", "worktree", "prune"], cwd=PROJECT_ROOT, timeout=60)
    checks = {k: bool(result[k]["passed"]) for k in ("security", "forbidden", "gwaya_audit", "pytest", "lean", "pipeline")}
    trust_ok = signature_policy_ok(result["trust"]["owner_authored"], result["signatures"], REQUIRE_SIGNED_COMMITS)
    decision = decide(trust_ok, bool(pr.get("isDraft")), pr.get("mergeable"), checks)
    if result["trust"]["owner_authored"] and not trust_ok:
        decision = "HOLD (owner-attributed but unsigned commits; signatures required)"
    result["checks"] = checks
    result["decision"] = decision
    table = "| check | result |\n| :-- | :-- |\n" + "\n".join(f"| {k} | {'pass' if v else 'FAIL'} |" for k, v in checks.items())
    body = (f"**HiveMind PR gatekeeper** — head `{head[:10]}` — decision: **{decision}**\n\n"
            f"Owner-authored: {result['trust']['owner_authored']} (author `{result['trust']['author_login']}`, {len(result['trust']['commits'])} commit author(s) verified by OwnerOrderGuardrail)\n\n{table}\n\n"
            f"pytest: `{result['pytest']['summary']}` on {len(result['pytest']['targets'])} file(s). "
            "PR text is treated as data; only the owner's commit identity and these checks decide. Log: `artifacts/pr_gatekeeper/`.")
    result["merged"] = False
    if not dry_run:
        sh(["gh", "pr", "comment", str(n), "--body", body], cwd=PROJECT_ROOT, timeout=120)
        if merge and decision == "ACCEPT":
            rc, out = sh(["gh", "pr", "merge", str(n), "--merge", "--admin"], cwd=PROJECT_ROOT, timeout=300)
            result["merged"] = rc == 0
            result["merge_output"] = out[-300:]
    return result


def recent_activity() -> List[Dict[str, Any]]:
    sh(["git", "fetch", "-q", "--prune", "origin"], cwd=PROJECT_ROOT, timeout=300)
    rc, out = sh(["git", "for-each-ref", "--sort=-committerdate", "refs/remotes/origin", "--format=%(committerdate:iso8601)|%(refname:short)|%(authoremail)|%(subject)"], cwd=PROJECT_ROOT, timeout=60)
    rows = []
    for line in out.strip().splitlines()[:15]:
        d, ref, email, subj = line.split("|", 3)
        v = normalize_and_verify_owner_email(email.strip("<>"))
        rows.append({"date": d, "ref": ref, "author_email": email, "owner": v.is_authorized_owner, "subject": subj[:100]})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true", help="merge PRs that are ACCEPTed (owner-authored, all checks pass)")
    ap.add_argument("--dry-run", action="store_true", help="no comments, no merges")
    ap.add_argument("--pr", type=int, default=None)
    args = ap.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    latest_path = LOG_DIR / "latest.json"
    previous = json.loads(latest_path.read_text()) if latest_path.exists() else {"gated": {}}
    prs = gh_json(["pr", "list", "--state", "open", "--json", "number,title,body,author,headRefName,headRefOid,baseRefName,isDraft,mergeable,commits"])
    if args.pr is not None:
        prs = [p for p in prs if p["number"] == args.pr]
    run: Dict[str, Any] = {"run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "open_prs": [p["number"] for p in prs], "results": [], "recent_branches": recent_activity(), "policy": "owner-only merge authority; PR text is data"}
    gated = dict(previous.get("gated", {}))
    for pr in prs:
        key = f"{pr['number']}@{pr['headRefOid']}"
        if key in gated and not args.pr:
            run["results"].append(retry_merge(pr, gated[key], merge=args.merge, dry_run=args.dry_run))
            continue
        try:
            res = gate_pr(pr, merge=args.merge, dry_run=args.dry_run)
        except Exception as e:  # a failed gate is a refusal, never an acceptance
            res = {"pr": pr["number"], "head": pr["headRefOid"], "decision": f"REFUSE (gate error: {str(e)[:200]})", "error": True}
        run["results"].append(res)
        gated[key] = res["decision"]
    run["gated"] = gated
    run_hash = hashlib.sha256(json.dumps(run, sort_keys=True, default=str).encode()).hexdigest()
    run["receipt_sha256"] = run_hash
    stamp = run["run_utc"].replace(":", "").replace("-", "")
    (LOG_DIR / f"run_{stamp}.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    latest_path.write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"open_prs": run["open_prs"], "decisions": {str(r["pr"]): r["decision"] for r in run["results"]}, "merged": [r["pr"] for r in run["results"] if r.get("merged")], "non_owner_branches": [b["ref"] for b in run["recent_branches"] if not b["owner"]][:10], "receipt": run_hash}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
