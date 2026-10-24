"""Serialization, scan I/O failures, and storage boundary regressions."""

import json
from pathlib import Path

import pytest
import torch

import hivemind.security as security
from hivemind.security import swarm_runtime_hardening as hardening
from hivemind.security.owner_order_guardrail import (
    OwnerOrderGuardrail,
    classify_request_intent,
)
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.security.federated_learning_hardening import FederatedLearningSecurityGuard


def test_owner_receipts_serialize_nested_decisions_without_losing_fields():
    guard = OwnerOrderGuardrail()
    request = guard.coerce_request(
        {
            "id": "external",
            "author_email": "contributor@example.invalid",
            "title": "Please update the code",
            "body": "Change this function",
            "files": [{"filename": "src/worker.py"}, {"path": "README.md"}, "docs/help.md", {}],
        }
    )
    assert request.changed_files == ("src/worker.py", "README.md", "docs/help.md")
    classification = classify_request_intent(request.content, request.changed_files)
    encoded = classification.to_dict()
    assert encoded["intent_category"] == classification.intent_category.value
    assert encoded["matched_triggers"] == list(classification.matched_triggers)
    assert encoded["protected_paths_touched"] == list(classification.protected_paths_touched)
    result = guard.filter_requests([request])
    receipt = result.to_dict()
    assert receipt["total_evaluated"] == receipt["rejected_count"] == 1
    assert receipt["accepted_request_ids"] == []
    decision = result.decisions[0]
    assert receipt["decisions"][0] == decision.to_dict()
    assert receipt["decisions"][0]["matched_triggers"] == list(decision.matched_triggers)
    assert json.loads(json.dumps(receipt))["batch_receipt_sha256"] == result.batch_receipt_sha256


def test_unknown_security_export_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute"):
        getattr(security, "missing_security_export_for_test")


def test_security_directory_advertises_all_public_exports():
    names = dir(security)
    assert names == sorted(set(names))
    assert set(security.__all__) <= set(names)


@pytest.mark.parametrize(
    "method",
    ["scan_text_for_payment_solicitation", "scan_unicode_trojan_source", "scan_text_for_secrets_and_exfiltration"],
)
def test_empty_scanner_input_has_no_findings(method):
    assert getattr(PRSecurityScanner(), method)("", file_path="fixture.txt") == []


def test_named_safe_yaml_loader_is_not_an_unsafe_deserialization_finding():
    findings = PRSecurityScanner().scan_python_ast_security(
        file_path="src/reader.py",
        source_code="import yaml\nfrom yaml import SafeLoader\nvalue = yaml.load(payload, Loader=SafeLoader)\n",
    )
    assert not any(f.category == "AST_UNSAFE_DESERIALIZATION" for f in findings)
    assert findings == []


def test_source_without_patch_counts_reported_additions():
    result = PRSecurityScanner().audit_pull_request(
        1,
        "Add implementation",
        "A small computation implementation",
        "contributor",
        [{"filename": "src/hivemind/worker.rs", "additions": 3}],
    )
    assert result.functional_lines_added == 3
    assert result.scanned_files_count == 1


def test_repository_scan_handles_missing_directories_cache_and_unreadable_files(tmp_path, monkeypatch):
    source = tmp_path / "src" / "hivemind"
    source.mkdir(parents=True)
    unreadable = source / "unreadable.py"
    unreadable.write_text("value = 1\n", encoding="utf-8")
    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"not source")
    original = Path.read_text

    def read_text(path, *args, **kwargs):
        if path == unreadable:
            raise OSError("fixture read failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    result = PRSecurityScanner().audit_repository(tmp_path)
    assert result.scanned_files_count == 1
    assert result.passed
    assert result.findings == []


def test_exempt_scanner_source_still_requires_valid_python(tmp_path):
    source = tmp_path / "src" / "hivemind" / "security" / "pr_security_scanner.py"
    source.parent.mkdir(parents=True)
    source.write_text("def broken(:\n", encoding="utf-8")
    result = PRSecurityScanner().audit_repository(tmp_path)
    assert result.blocked_merge
    assert result.scanned_files_count == 1
    assert [(f.rule_id, f.category) for f in result.findings] == [("SEC-00", "PYTHON_SYNTAX_ERROR")]


@pytest.mark.parametrize(
    "limit, value, samples, category",
    [
        ("MAX_SAMPLES_PER_SHARD", 1, ["first sample", "second sample"], "SHARD_SAMPLE_COUNT_EXCEEDED"),
        ("MAX_SAMPLE_CHARS", 4, ["first sample", "second sample"], "OVERSIZED_TRAINING_SAMPLE"),
        ("MAX_SHARD_BYTES", 1, ["first sample", "second sample"], "SHARD_BYTE_QUOTA_EXCEEDED"),
    ],
)
def test_dataset_quotas_reject_before_training(limit, value, samples, category, monkeypatch):
    monkeypatch.setattr(hardening, limit, value)
    verdict = hardening.SwarmTrainingAndStoreGuard().validate_dataset_shard(
        "dataset", samples, raise_on_violation=False
    )
    assert not verdict.passed
    assert verdict.rule_id == "DS-04"
    assert verdict.category == category


@pytest.mark.parametrize("relative", ["", "/absolute", "C:\\data", "bad\x00name"])
def test_cas_rejects_absolute_drive_empty_and_null_paths(tmp_path, relative):
    verdict, target = hardening.SwarmTrainingAndStoreGuard().validate_cas_relative_path(
        tmp_path, relative, raise_on_violation=False
    )
    assert not verdict.passed
    assert verdict.rule_id == "CAS-01"
    assert target == tmp_path.resolve()


def test_cas_symlink_must_not_escape_its_root(tmp_path):
    root = tmp_path / "cas"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    verdict, target = hardening.SwarmTrainingAndStoreGuard().validate_cas_relative_path(
        root, "link/payload", raise_on_violation=False
    )
    assert not verdict.passed
    assert verdict.category == "CAS_PATH_TRAVERSAL_BLOCKED"
    assert target == root.resolve()


@pytest.mark.parametrize("trim_fraction", [0.5, 1.0])
def test_overlarge_trim_fraction_falls_back_to_nonempty_mean(trim_fraction):
    guard = FederatedLearningSecurityGuard(trimmed_mean_beta=trim_fraction)
    values = {str(i): {"weight": torch.tensor([1.0, 2.0])} for i in range(4)}
    aggregated, _ = guard.clip_and_aggregate_surviving_deltas(values)
    torch.testing.assert_close(aggregated["weight"], torch.tensor([1.0, 2.0]))
