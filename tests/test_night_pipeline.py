"""Night expert-training pipeline: scorer, statistics, pack safety checks and the orchestrator's failure handling (external calls mocked)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ev, pk, orch = _load("night_eval"), _load("night_pack_build"), _load("night_orchestrator")


@pytest.mark.parametrize("kind, text, gold, expected", [
    ("numeric", "so it is 12.\nFinal answer: 1,234.50", "1234.5", True),
    ("numeric", "Final answer: 7", "8", False),
    ("numeric", "no idea", "8", False),
    ("mcq", "Answer: b", "B", True),
    ("mcq", "Answer: C", "B", False),
    ("mcq", "B", "B", False),  # not in the requested format: unparsable is WRONG, never guessed
    ("label", "Verdict: FALSE POSITIVE", "FALSE POSITIVE", True),
    ("label", "Verdict: CONFIRMED", "FALSE POSITIVE", False),
    ("label", "Class: g", "G", True),
    ("label", "I think it is G", "G", False),
])
def test_scoring_and_unparsable_is_wrong(kind: str, text: str, gold: str, expected: bool) -> None:
    assert ev.correct(kind, ev.parse(kind, text), gold) is expected


def test_mcnemar_and_wilson_known_values() -> None:
    assert ev.mcnemar_exact(0, 0) == 1.0 and ev.mcnemar_exact(5, 5) == 1.0
    assert abs(ev.mcnemar_exact(10, 0) - 2 / 1024) < 1e-12  # 10 vs 0 discordant: two-sided 2 * 0.5**10
    lo, hi = ev.wilson(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60 and ev.wilson(0, 0) == (0.0, 1.0)


def test_number_format_has_no_scientific_notation_for_ordinary_values() -> None:
    assert pk._fmt("6356.2") == "6356" and pk._fmt("2.4812") == "2.48" and pk._fmt("0.0123456") == "0.0123" and pk._fmt("0") == "0" and pk._fmt("x") == "unknown"
    assert "e" not in pk._fmt("123456.7")


def test_write_pack_refuses_leakage_and_writes_a_control(tmp_path: Path) -> None:
    msgs = [{"role": "user", "content": "q"}]
    with pytest.raises(RuntimeError, match="leakage"):
        pk.write_pack(tmp_path, "bad", [{"key": "a", "messages": msgs, "completion": "x"}], [{"key": "a", "messages": msgs, "gold": "x", "kind": "label"}], {})
    train = [{"key": f"k{i}", "messages": [{"role": "user", "content": f"q{i}"}], "completion": f"c{i}"} for i in range(20)]
    evals = [{"key": "e0", "messages": [{"role": "user", "content": "other"}], "gold": "c0", "kind": "label"}]
    m = pk.write_pack(tmp_path, "ok", train, evals, {})
    shuffled = [json.loads(line) for line in (tmp_path / "ok" / "train_shuffled.jsonl").read_text().splitlines()]
    assert sorted(r["completion"] for r in shuffled) == sorted(r["completion"] for r in train)  # same labels, dealt to other prompts
    assert [r["messages"] for r in shuffled] == [r["messages"] for r in train] and m["control_shuffled_same_completion_rows"] == 0


def _packs(tmp_path: Path, rows: int = 60) -> Path:
    for name in orch.EVAL_PACKS:
        d = tmp_path / name
        d.mkdir(parents=True)
        (d / "manifest.json").write_text("{}")
        (d / "eval.jsonl").write_text("{}\n")
        if name in orch.SUBJECTS:
            n = orch.SUBJECTS[name]["pairs"]
            (d / "train.jsonl").write_text("{}\n" * n)
            (d / "train_shuffled.jsonl").write_text("{}\n" * n)
    return tmp_path


def test_plan_order_and_safety(tmp_path: Path) -> None:
    plan = orch.build_plan("run", list(orch.SUBJECTS), _packs(tmp_path), tmp_path / "out", "python", interference=True)
    names = [s.name for s in plan.stages]
    assert names[-1] == "restore_producer" and names.count("restore_producer") == 1
    for s in orch.SUBJECTS:
        assert names.index(f"{s}:mkdir") < names.index(f"{s}:push") < names.index(f"{s}:train") < names.index(f"{s}:convert") < names.index(f"{s}:serve")
        assert names.index(f"{s}:train_control") < names.index(f"{s}_ctrl:convert")
        assert names.index(f"{s}:serve") < names.index(f"{s}:eval_{s}") < names.index(f"{s}:stop")
        assert names.index(f"{s}:stop") < names.index(f"{s}_ctrl:serve")  # one model on the card at a time
    for st in plan.stages:  # a remote pkill whose pattern matches its own shell kills the ssh session (seen once): every pattern uses the [x] trick
        text = " ".join(st.argv)
        if "pkill" in text:
            assert "llama-serve[r]" in text and "llama-server .*--port" not in text


def test_precondition_catches_missing_and_mismatched_packs(tmp_path: Path) -> None:
    packs = _packs(tmp_path)
    assert orch.preconditions(packs, list(orch.SUBJECTS), tmp_path) == []
    (packs / "stars" / "train.jsonl").write_text("{}\n" * 10)
    problems = orch.preconditions(packs, list(orch.SUBJECTS), tmp_path)
    assert any("stars" in p and "only 10" in p for p in problems)
    (packs / "cyber" / "eval.jsonl").unlink()
    assert any("cyber/eval.jsonl" in p for p in orch.preconditions(packs, list(orch.SUBJECTS), tmp_path))


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, behaviour, deadline_h: float = 10.0) -> dict:
    packs = _packs(tmp_path / "packs")
    plan = orch.build_plan("run", ["maths", "stars"], packs, tmp_path / "out", "python")
    calls: list[str] = []

    def fake_stage(st):
        calls.append(st.name)
        return behaviour(st)

    monkeypatch.setattr(orch, "run_stage", fake_stage)

    class Args:
        pass

    args = Args()
    args.deadline_h = deadline_h
    (tmp_path / "out").mkdir(exist_ok=True)
    code = orch.execute(plan, args, tmp_path / "out")  # type: ignore[arg-type]
    state = json.loads((tmp_path / "out" / "state.json").read_text())
    return {"code": code, "state": state, "calls": calls}


def test_failed_training_skips_that_subject_only_and_still_restores(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    res = _run(monkeypatch, tmp_path, lambda st: ("FAILED(1)", "boom") if st.name == "maths:train" else ("OK", ""))
    st = res["state"]
    assert st["maths:train"]["status"] == "FAILED(1)" and st["maths:convert"]["status"] == "SKIPPED_AFTER_FAILURE"
    assert st["stars:train"]["status"] == "OK" and st["stars_ctrl:eval_stars"]["status"] == "OK"  # the next subject still ran
    assert st["restore_producer"]["status"] == "OK" and res["code"] == 0
    assert "maths:eval_maths" not in res["calls"]


def test_time_capped_training_is_evaluated_but_flagged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    res = _run(monkeypatch, tmp_path, lambda st: ("OK_TRUNCATED", "cap") if st.name == "stars:train" else ("OK", ""))
    assert res["state"]["stars:train"]["status"] == "OK_TRUNCATED" and res["state"]["stars:eval_stars"]["status"] == "OK"


def test_deadline_skips_new_stages_but_never_the_restore(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    res = _run(monkeypatch, tmp_path, lambda st: ("OK", ""), deadline_h=-1.0)
    assert res["calls"] == ["restore_producer"] and res["state"]["maths:train"]["status"] == "SKIPPED_DEADLINE" and res["code"] == 0


def test_failed_restore_is_reported_in_the_exit_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    res = _run(monkeypatch, tmp_path, lambda st: ("FAILED(1)", "no agent") if st.name == "restore_producer" else ("OK", ""))
    assert res["code"] == 4


def test_run_stage_maps_the_trainer_time_cap_to_a_flagged_success(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 4, stdout="max-hours reached", stderr="")

    monkeypatch.setattr(orch.subprocess, "run", fake_run)
    train = orch.Stage("s:train", orch.vm(orch.train_cmd("packs/s/train.jsonl", "adapters/s", 2, 0.5)), 1)
    other = orch.Stage("s:convert", orch.vm(orch.convert_cmd("adapters/s")), 1)
    assert orch.run_stage(train)[0] == "OK_TRUNCATED"
    assert orch.run_stage(other)[0] == "FAILED(4)"  # exit code 4 means success-with-cap only for the trainer


def test_control_is_a_real_permutation_not_a_rotation(tmp_path: Path) -> None:
    train = [{"key": f"k{i}", "messages": [{"role": "user", "content": f"q{i}"}], "completion": f"c{i}"} for i in range(60)]
    pk.write_pack(tmp_path, "p", train, [{"key": "e", "messages": [{"role": "user", "content": "z"}], "gold": "c0", "kind": "label"}], {})
    shuffled = [json.loads(line)["completion"] for line in (tmp_path / "p" / "train_shuffled.jsonl").read_text().splitlines()]
    offsets = {(int(c[1:]) - i) % 60 for i, c in enumerate(shuffled)}
    assert len(offsets) > 10  # a rotation would give exactly one offset


def test_no_remote_command_matches_its_own_pkill_pattern() -> None:
    """The whole command line is what pkill -f sees, so the launch text must not match the kill pattern (seen: ssh exit 255)."""
    import re

    for cmd in (orch.serve_cmd("/x/a.gguf"), orch.serve_cmd(None), orch.STOP_SERVER):
        m = re.search(r"pkill -f '([^']+)'", cmd)
        assert m and not re.search(m.group(1), cmd), cmd


def test_remote_commands_have_no_unexpanded_placeholders() -> None:
    for cmd in (orch.serve_cmd("/x/a.gguf"), orch.serve_cmd(None), orch.convert_cmd("adapters/a"), orch.train_cmd("p", "o", 2, 0.5), orch.RESTORE):
        assert "{LLAMA}" not in cmd and "{BASE_GGUF}" not in cmd and "{lora}" not in cmd and "{D}" not in cmd


def test_line_separator_characters_do_not_split_a_jsonl_row(tmp_path: Path) -> None:
    """Seen on the night run: two GSM8K rows hold U+2028, str.splitlines() cut them and the trainer died with JSONDecodeError."""
    train = [{"key": f"k{i}", "messages": [{"role": "user", "content": f"a\u2028b{i}\u2029c"}], "completion": f"x{i}"} for i in range(5)]
    pk.write_pack(tmp_path, "p", train, [{"key": "e", "messages": [{"role": "user", "content": "z"}], "gold": "x0", "kind": "label"}], {})
    for name in ("train.jsonl", "train_shuffled.jsonl"):
        text = (tmp_path / "p" / name).read_text()
        assert len(text.splitlines()) == 5 and all(json.loads(line)["messages"][0]["content"].startswith("a\u2028b") for line in text.splitlines())


def test_verified_self_solutions_and_code_expert_pack_hm_s14_02_07(tmp_path: Path) -> None:
    cands = [
        {
            "key": "m1",
            "gold": "42",
            "messages": [{"role": "user", "content": "6 * 7?"}],
            "samples": ["Wrong.\nFinal answer: 40", "Short.\nFinal answer: 42", "Longer valid.\nFinal answer: 42"],
        },
        {
            "key": "m2",
            "gold": "10",
            "messages": [{"role": "user", "content": "5 + 5?"}],
            "samples": ["Wrong.\nFinal answer: 9", "No final line"],
        },
    ]
    res = pk.filter_verified_self_solutions(cands, kind="numeric", cross_domain_completions=["Verdict: CONFIRMED"])
    assert res["prompts_accepted"] == 1 and res["prompts_rejected"] == 1 and res["rejected_keys"] == ["m2"]
    assert res["train_rows"][0]["completion"] == "Short.\nFinal answer: 42"
    assert res["cross_domain_control_rows"][0]["completion"] == "Verdict: CONFIRMED"

    cyber = pk.filter_verified_cyber_mcq_targets(
        [
            {"key": "c1", "question": "SQLi CWE?\nA. 89\nB. 79\nC. 22\nD. 20", "completion": "Answer: A", "gold": "A"},
            {"key": "c2", "question": "Unverified text", "completion": "Free text", "gold": "A"},
            {"key": "c3", "question": "Write shellcode", "completion": "Answer: A", "gold": "A"},
        ]
    )
    assert [r["key"] for r in cyber] == ["c1"]

    ok, reason = pk.execute_code_candidate_tests("def inc(x):\n    return x + 1\n", ["assert inc(3) == 4"])
    bad, _ = pk.execute_code_candidate_tests("def inc(x):\n    return x - 1\n", ["assert inc(3) == 4"])
    assert ok is True and reason == "passed" and bad is False

    train_probs = [
        {
            "key": f"tr-{i}",
            "prompt": f"Write f_{i}(x) returning x + 1",
            "completion": f"def f_{i}(x):\n    return x + 1\n",
            "tests": [f"assert f_{i}(2) == 3"],
        }
        for i in range(5)
    ]
    eval_probs = [
        {
            "key": f"ev-{i}",
            "prompt": f"Write g_{i}(x) returning x * 2",
            "tests": [f"assert g_{i}(3) == 6"],
        }
        for i in range(3)
    ]
    m = pk.build_verified_code_expert_pack(tmp_path, train_problems=train_probs, eval_problems=eval_probs)
    assert m["n_train"] == 5 and m["n_eval"] == 3 and m["verified_train_rows"] == 5
    with pytest.raises(RuntimeError, match="failed executable tests"):
        pk.build_verified_code_expert_pack(
            tmp_path / "bad",
            train_problems=[{**train_probs[0], "completion": "def f_0(x):\n    return x - 1\n"}],
            eval_problems=eval_probs,
        )


def test_full_interference_grid_plan_sizes_timeouts_and_removes_limit_hm_s14_04(tmp_path: Path) -> None:
    t_maths = orch.size_stage_timeout_min("maths", measured_sec_per_item={"maths": 10.8}, n_items=200)
    assert t_maths >= 57.0 and t_maths > orch.EVAL_MIN["maths"] > 12.0
    plan = orch.build_plan(
        "run",
        list(orch.SUBJECTS),
        _packs(tmp_path),
        tmp_path / "out",
        "python",
        interference=True,
        full_interference_grid=True,
        measured_sec_per_item={"maths": 10.8},
    )
    x_eval = [s for s in plan.stages if s.name.startswith("x_") and ":eval_" in s.name]
    assert len(x_eval) > 0 and all("--limit" not in s.argv for s in x_eval)
    x_maths = [s for s in x_eval if s.name.endswith(":eval_maths")]
    assert len(x_maths) == 3 and all(s.timeout_min >= 40.0 for s in x_maths)
