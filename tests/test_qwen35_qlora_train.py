"""CPU tests for scripts/qwen35_qlora_train.py: base pinning, input validation, step plan and the serving guard (no network, no GPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import qwen35_qlora_train as q  # noqa: E402


def _write_pairs(path: Path, n: int, bad_at: int | None = None) -> Path:
    lines = []
    for i in range(n):
        rec = {"messages": [{"role": "user", "content": f"question {i}"}], "completion": f"answer {i}"}
        if i == bad_at:
            rec = {"messages": [{"role": "user", "content": "question"}], "completion": "   "}
        lines.append(json.dumps(rec))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_resolve_base_returns_the_pinned_revision_from_the_catalogue() -> None:
    model_id, sha = q.resolve_base("Qwen/Qwen3.5-0.8B")
    assert model_id == "Qwen/Qwen3.5-0.8B"
    assert sha == "2fc06364715b967f1860aea9cf38778875588b17"


def test_resolve_base_refuses_a_model_that_is_not_in_the_catalogue() -> None:
    with pytest.raises(ValueError, match="is not in"):
        q.resolve_base("Qwen/Not-A-Model")


def test_resolve_base_refuses_an_unpinned_revision(tmp_path: Path) -> None:
    catalogue = tmp_path / "catalogue.json"
    catalogue.write_text(json.dumps({"models": [{"model_id": "Qwen/Qwen3.5-0.8B", "revision_sha": ""}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned 40-hex"):
        q.resolve_base("Qwen/Qwen3.5-0.8B", catalogue=catalogue)


def test_load_pairs_refuses_fewer_than_fifty_records(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 50"):
        q.load_pairs(_write_pairs(tmp_path / "pairs.jsonl", 49))


def test_load_pairs_accepts_fifty_valid_records(tmp_path: Path) -> None:
    assert len(q.load_pairs(_write_pairs(tmp_path / "pairs.jsonl", 50))) == 50


def test_load_pairs_refuses_a_blank_completion(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="completion"):
        q.load_pairs(_write_pairs(tmp_path / "pairs.jsonl", 60, bad_at=7))


def test_training_plan_counts_steps_per_epoch_and_in_total() -> None:
    assert q.training_plan(100, 16, 3) == {"steps_per_epoch": 7, "total_steps": 21}


def test_training_plan_changes_with_the_epoch_count() -> None:
    assert q.training_plan(100, 16, 3) != q.training_plan(100, 16, 1)


def test_gpu_run_is_refused_without_the_serving_confirmation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pairs = _write_pairs(tmp_path / "pairs.jsonl", 60)
    calls: list[object] = []
    monkeypatch.setattr(q.subprocess, "run", lambda *a, **k: calls.append("run"))
    monkeypatch.setattr(q, "fetch_snapshot", lambda *a, **k: calls.append("fetch"))
    assert q.main(["--size", "0.8B", "--pairs", str(pairs), "--out", str(tmp_path / "out")]) == 4
    assert calls == []


def test_dry_run_prints_the_plan_and_neither_downloads_nor_trains(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    pairs = _write_pairs(tmp_path / "pairs.jsonl", 60)
    calls: list[object] = []
    monkeypatch.setattr(q.subprocess, "run", lambda *a, **k: calls.append("run"))
    monkeypatch.setattr(q, "fetch_snapshot", lambda *a, **k: calls.append("fetch"))
    assert q.main(["--size", "0.8B", "--pairs", str(pairs), "--out", str(tmp_path / "out"), "--dry-run", "--smoke"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["epochs"] == 1 and printed["max_hours"] == 0.25 and printed["total_steps"] == 4
    assert calls == []
