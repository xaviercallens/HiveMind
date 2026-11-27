"""Offline command-adapter and report regressions; every external command is mocked."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hivemind.cli import community_orchestrator as orchestrator


@pytest.mark.parametrize("returncode, stdout", [(1, "failure"), (0, "")])
def test_gh_read_adapters_handle_failure_or_empty_output(monkeypatch, returncode, stdout):
    run = Mock(return_value=SimpleNamespace(returncode=returncode, stdout=stdout))
    monkeypatch.setattr(orchestrator.subprocess, "run", run)
    assert orchestrator._fetch_open_prs_via_gh("fixture/repo") == []
    assert orchestrator._fetch_discussions_via_gh("fixture", "repo") == []
    assert run.call_count == 2
    assert all(call.args[0][0] == "gh" for call in run.call_args_list)


def test_gh_read_adapters_normalize_metadata_and_missing_optional_fields(monkeypatch):
    prs = [{"number": 7, "changedFiles": 2}]
    files = [
        {"filename": "src/worker.py", "additions": 2, "deletions": 1,
         "patch": "@@ -1 +1,2 @@\n-old_value = 0\n+new_value = 1\n+other_value = 2"},
        {"filename": "src/config.py", "additions": 1, "deletions": 0,
         "patch": "@@ -0,0 +1 @@\n+enabled = True"},
    ]
    discussion = {"id": "discussion-id", "number": 3}
    run = Mock(side_effect=[
        SimpleNamespace(returncode=0, stdout=json.dumps(prs)),
        SimpleNamespace(returncode=0, stdout=json.dumps([[files[0]], [files[1]]])),
        SimpleNamespace(returncode=0, stdout=json.dumps({"data": {"repository": {"discussions": {"nodes": [discussion]}}}})),
    ])
    monkeypatch.setattr(orchestrator.subprocess, "run", run)
    normalized = orchestrator._fetch_open_prs_via_gh("fixture/repo")
    assert normalized[0]["number"] == 7
    assert normalized[0]["author"] == "unknown"
    assert normalized[0]["body"] == ""
    assert normalized[0]["mergeable"] == "UNKNOWN"
    assert normalized[0]["isDraft"] is False
    assert normalized[0]["files"] == files
    assert normalized[0]["diff_retrieved"] is True
    assert run.call_args_list[1].args[0] == [
        "gh", "api", "--paginate", "--slurp", "repos/fixture/repo/pulls/7/files?per_page=100",
    ]
    assert orchestrator._fetch_discussions_via_gh("fixture", "repo") == [discussion]


def test_general_discussion_reply_contains_wiki_and_idempotency_marker():
    plan = orchestrator.compose_discussion_reply(
        {"id": "discussion-id", "number": 8, "title": "Hello", "body": "Where can I learn more?", "author": {"login": "reader"}, "comments": {"nodes": []}}
    )
    assert plan["needs_reply"] is True
    assert plan["reply_category"] == "general_community_support"
    assert orchestrator.ORCHESTRATOR_REPLY_MARKER in plan["reply_markdown"]
    assert "reader" in plan["reply_markdown"]
    assert plan["recommended_wiki_pages"]


@pytest.mark.parametrize("returncode, files", [
    (1, []),
    (0, []),
    (0, [{"filename": "src/worker.py", "additions": 1, "deletions": 0}]),
    (0, [{"filename": "src/worker.py", "additions": 2, "deletions": 0, "patch": "+enabled = True"}]),
])
def test_pr_audit_holds_incomplete_diff(monkeypatch, returncode, files):
    prs = [{"number": 7, "changedFiles": 1, "mergeable": "MERGEABLE"}]
    run = Mock(side_effect=[
        SimpleNamespace(returncode=0, stdout=json.dumps(prs)),
        SimpleNamespace(returncode=returncode, stdout=json.dumps([files])),
    ])
    monkeypatch.setattr(orchestrator.subprocess, "run", run)
    pr, = orchestrator._fetch_open_prs_via_gh("fixture/repo")
    decision = orchestrator.evaluate_pull_request_for_merge(pr)
    assert decision["verdict"] == "HOLD_FOR_INCOMPLETE_DIFF"
    assert decision["diff_retrieved"] is False
    assert decision["security_passed"] is False
    assert decision["can_merge"] is False


def test_quality_audit_handles_corrupt_report_files(tmp_path, monkeypatch):
    for relative in (
        "artifacts/architecture_hypothesis_verification_report.json",
        "artifacts/decoupled_diloco_quorum_check.json",
        "artifacts/paper_v3_experiments_report.json",
        "paper/PEER_REVIEW_REPORT.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{invalid json", encoding="utf-8")
    monkeypatch.setattr(orchestrator, "verify_local_git_owner_identity", lambda root: {"git_config_email_matches_owner": False, "signature_verified": False, "allow_local_git_directive_authority": False})
    report = orchestrator.run_hourly_quality_and_experiment_audit(tmp_path)
    assert report["hypotheses_passed"] == "0/0"
    assert report["decoupled_diloco_h11_passed"] is False
    assert report["experimental_suites_verified"] == "0/4"
    assert report["paper_technical_gate_pass_rate"] == 0.0


@pytest.mark.parametrize("execute_live, auto_merge, returncode", [(False, True, 0), (True, False, 0), (True, True, 0), (True, True, 1)])
def test_cycle_command_dispatch_and_receipts_are_offline(monkeypatch, tmp_path, execute_live, auto_merge, returncode):
    run = Mock(return_value=SimpleNamespace(returncode=returncode, stdout=""))
    monkeypatch.setattr(orchestrator.subprocess, "run", run)
    monkeypatch.setattr(orchestrator, "_fetch_open_prs_via_gh", lambda repo: [{"number": 7}])
    monkeypatch.setattr(orchestrator, "evaluate_pull_request_for_merge", lambda pr, scanner: {"pr_number": 7, "can_merge": True, "review_markdown": "fixture review"})
    monkeypatch.setattr(orchestrator, "_fetch_discussions_via_gh", lambda owner, name: [{"id": "discussion-id"}, {"id": "already-answered"}])
    monkeypatch.setattr(orchestrator, "compose_discussion_reply", lambda discussion: {"discussion_id": discussion["id"], "needs_reply": discussion["id"] == "discussion-id", "reply_markdown": "fixture reply"})
    monkeypatch.setattr(orchestrator, "run_hourly_quality_and_experiment_audit", lambda: {"fixture": True})
    output = tmp_path / "nested" / "cycle.json"
    report = orchestrator.run_orchestrator_cycle("fixture/repo", execute_live, auto_merge, output)
    assert report["prs_merged_live"] == []
    assert report["discussion_replies_posted"] == int(execute_live and returncode == 0)
    assert report["pr_decisions"][0]["merged_live"] is False
    assert report["discussion_actions"][1]["posted_live"] is False
    assert run.call_count == int(execute_live)
    if execute_live:
        discussion_command = run.call_args_list[-1].args[0]
        assert discussion_command[:3] == ["gh", "api", "graphql"]
        assert "discId=discussion-id" in discussion_command
        assert "body=fixture reply" in discussion_command
    for call in run.call_args_list:
        assert call.kwargs == {"capture_output": True, "text": True, "check": False}
    assert json.loads(output.read_text(encoding="utf-8"))["cycle_sha256"] == report["cycle_sha256"]
    assert len(report["cycle_sha256"]) == 64


def test_cycle_rejects_invalid_repository_before_external_calls(monkeypatch):
    run = Mock(side_effect=AssertionError("unexpected external command"))
    monkeypatch.setattr(orchestrator.subprocess, "run", run)
    with pytest.raises(ValueError, match="owner/name"):
        orchestrator.run_orchestrator_cycle("invalid", output_path=None)
    run.assert_not_called()


def test_main_forwards_flags_and_prints_json(monkeypatch, tmp_path, capsys):
    cycle = Mock(return_value={"fixture": True})
    monkeypatch.setattr(orchestrator, "run_orchestrator_cycle", cycle)
    target = tmp_path / "cycle.json"
    assert orchestrator.main(["--repo", "fixture/repo", "--execute-live", "--no-auto-merge", "--output", str(target)]) == 0
    cycle.assert_called_once_with(repo="fixture/repo", execute_live=True, output_path=target)
    assert json.loads(capsys.readouterr().out) == {"fixture": True}
