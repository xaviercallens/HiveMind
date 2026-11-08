# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for scripts/publish_chain_to_gcs.py (read-only BOINC QLoRA-MoE adapter chain publisher)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "publish_chain_to_gcs.py"
spec = importlib.util.spec_from_file_location("publish_chain_to_gcs", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
pub = importlib.util.module_from_spec(spec)
sys.modules["publish_chain_to_gcs"] = pub
spec.loader.exec_module(pub)


def test_wu_number_and_sha256_streaming(tmp_path: Path) -> None:
    p = tmp_path / "adapter_hmq_00180.pt"
    payload = b"hivemind-adapter-payload-00180"
    p.write_bytes(payload)
    assert pub.wu_number(p) == 180
    assert pub.wu_number(tmp_path / "unrelated.txt") == -1
    assert pub.sha256(p) == hashlib.sha256(payload).hexdigest()


def test_readme_reports_median_gain_and_train_to_serve_gate_notice() -> None:
    rows = [
        {"wu_id": "hmq_00177", "eval_loss_before": 2.10, "eval_loss_after": 2.04},
        {"wu_id": "hmq_00178", "eval_loss_before": 2.04, "eval_loss_after": 2.02},
        {"wu_id": "hmq_00179", "eval_loss_before": 2.02, "eval_loss_after": 2.05},
    ]
    md = pub.readme("gs://hm-vault/chain", rows, 176, 6)
    assert "hmq_00177 to hmq_00179 (3)" in md
    assert "median +0.0200" in md
    assert "docs/TRAIN_TO_SERVE_GATE.md" in md
    assert "weights_only=True" in md


def test_main_refuses_public_bucket_without_allow_public_and_stages_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "hivemind_ledger"
    ledger.mkdir()
    (ledger / "assimilated.jsonl").write_text(
        json.dumps({"wu_id": "hmq_00177", "eval_loss_before": 2.1, "eval_loss_after": 2.0}) + "\n",
        encoding="utf-8",
    )
    (ledger / "adapter_hmq_00177.pt").write_bytes(b"adapter-177")
    (ledger / "verdict_hmq_00177.json").write_text('{"accepted": true}', encoding="utf-8")

    # Negative control: public IAM binding is refused with exit code 3 unless --allow-public is passed.
    monkeypatch.setattr(
        pub,
        "gcloud",
        lambda *args, **kwargs: json.dumps({"bindings": [{"members": ["allUsers"]}]}),
    )
    monkeypatch.setattr(sys, "argv", ["publish_chain_to_gcs.py", "--ledger", str(ledger), "--dest", "gs://pub-bkt/pfx"])
    assert pub.main() == 3

    # Positive control: private bucket with --dry-run succeeds (0) without uploading files.
    calls: list[tuple[Any, ...]] = []

    def _private_gcloud(*args: Any, **kwargs: Any) -> str:
        calls.append(args)
        if args[:3] == ("storage", "buckets", "get-iam-policy"):
            return json.dumps({"bindings": [{"members": ["user:callensxavier@gmail.com"]}]})
        return ""

    monkeypatch.setattr(pub, "gcloud", _private_gcloud)
    monkeypatch.setattr(
        sys,
        "argv",
        ["publish_chain_to_gcs.py", "--ledger", str(ledger), "--dest", "gs://priv-bkt/pfx", "--dry-run"],
    )
    assert pub.main() == 0
    assert all(c[:2] != ("storage", "cp") for c in calls)
