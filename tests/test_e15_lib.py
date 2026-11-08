"""E15 helpers (CPU only): statistics, AST dedup, the GWAYA loop logic with a scripted generator, LoRA merge equivalence, schedule, data splits, fail-fast and prereg guards."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
_CUDA_VISIBLE_DEVICES_BEFORE = os.environ.get("CUDA_VISIBLE_DEVICES")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # for the imports below only; restored after them

import e15_data as data  # noqa: E402
import e15_gate as gate_mod  # noqa: E402
import e15_lib as L  # noqa: E402

# Restored right after the imports above: setting it at module level hid the GPU from every test module collected
# after this one (IP-179); the autouse fixture below keeps this module's own tests CPU-only at run time.
if _CUDA_VISIBLE_DEVICES_BEFORE is None:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _CUDA_VISIBLE_DEVICES_BEFORE


@pytest.fixture(autouse=True)
def _cpu_only_for_this_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

needs_sandbox = pytest.mark.skipif(not gate_mod.gate_available(), reason="sandbox unavailable")


# ---------------------------------------------------------------- statistics
def test_wilson_known_values():
    lo, hi = L.wilson(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3
    lo, hi = L.wilson(0, 10)
    assert lo == pytest.approx(0, abs=1e-9) and 0.25 < hi < 0.31
    assert math.isnan(L.wilson(0, 0)[0])


def test_mcnemar_exact():
    assert L.mcnemar_exact(0, 0) == 1.0
    assert L.mcnemar_exact(5, 5) == 1.0
    assert L.mcnemar_exact(0, 10) == pytest.approx(2 * 0.5 ** 10)
    assert L.mcnemar_exact(2, 8) == pytest.approx(2 * sum(math.comb(10, i) for i in range(3)) / 1024)
    r = L.paired([True, True, False, False, False], [True, False, True, True, True])
    assert (r["a_only"], r["b_only"], r["discordant"]) == (1, 3, 4) and r["diff"] == pytest.approx(0.4)


def test_lr_schedule():
    assert L.lr_at(0, 100, 2e-4, 10) == pytest.approx(2e-5)
    assert L.lr_at(9, 100, 2e-4, 10) == pytest.approx(2e-4)
    assert L.lr_at(10, 100, 2e-4, 10) == pytest.approx(2e-4)
    assert L.lr_at(99, 100, 2e-4, 10) < 1e-6
    assert all(L.lr_at(s, 100, 2e-4, 10) >= 0 for s in range(100))


def test_design_registered_values():
    d = L.DESIGN
    assert (d["lora"]["r"], d["lora"]["alpha"]) == (16, 32.0) and len(d["lora"]["targets"]) == 7
    t = d["train"]
    assert (t["epochs"], t["lr"], t["warmup"], t["wd"], t["micro_batch"], t["grad_accum"], t["max_seq"]) == (3, 2e-4, 10, 0.0, 4, 4, 768)
    assert d["rft"]["n_samples"] == 8 and d["rft"]["keep_per_problem"] == 2 and d["rft"]["fail_fast_min_problems"] == 150
    assert d["vram_cap_gib"] == 7.5 and d["vram_limit_gib"] == 8.0


# ---------------------------------------------------------------- dedup
def test_norm_ast_and_select_kept():
    a = "def f(x):\n    '''doc'''\n    return x + 1\n"
    b = "def f(x):\n    # comment\n    return x+1\n"
    c = "def f(x):\n    y = x + 1\n    return y\n"
    assert L.norm_ast(a) == L.norm_ast(b) != L.norm_ast(c)
    kept = L.select_kept([a, b, c, "def f(x:\n"], 2)
    assert kept == [c, b]  # shortest first; a is a duplicate of b (docstring / comment / spacing differences only)
    assert L.select_kept([a, b], 2) == [b]
    assert L.select_kept([a, b, c], 1) == [c]


# ---------------------------------------------------------------- scripted generation for the loop logic
class FakeTok:
    def encode_prompt(self, messages):
        return [len(messages[-1]["content"]) % 200 + 2]


class FakeGen:
    """Replies depend on the key suffix: candidates '#c0'/'#c1' (samples) and '#fixN' (repairs) come from a script keyed by problem."""

    def __init__(self, script):
        self.script, self.seconds, self.tokens, self.calls = script, 0.0, 0, []

    def run(self, items, sample):
        self.calls.append((len(items), sample))
        out = {}
        for key, _ in items:
            out[key] = (self.script.get(key, "no code"), 10)
            self.tokens += 10
        self.seconds += 0.5
        return out


def block(code):
    return f"```python\n{code}\n```"


@needs_sandbox
def test_gwaya_loop_selection_rules(tmp_path):
    tr, ev, he = data.synthetic_sets()
    p0, p1, p2, p3 = ev[0], ev[1], ev[2], ev[3]  # add_two? use their own references / wrong code
    wrong = block("def wrong(x):\n    return x\n")
    greedy = {p.key: {"text": wrong, "tokens": 5, "functional": False} for p in (p0, p1, p2, p3)}
    script = {
        f"{p0.key}#c1": block(p0.reference),  # p0: second sample (3rd candidate) passes -> no repair, tried = 3
        f"{p1.key}#fix2": block(p1.reference),  # p1: no candidate passes, repair round 2 passes
        f"{p2.key}#c0": block("def x(:"),  # p2: nothing ever passes -> fallback to first greedy
    }
    gen = FakeGen(script)
    j = L.Journal(tmp_path / "j.jsonl", False)
    recs = L.gwaya_loop(gen, FakeTok(), [p0, p1, p2, p3], [greedy[p.key] for p in (p0, p1, p2, p3)], j, "CX", workers=2)
    by = {r["problem"]: r for r in recs}
    a, b, c = by[p0.key], by[p1.key], by[p2.key]
    assert a["functional"] and a["source"] == "candidate" and a["candidates_tried"] == 3 and a["repair_rounds"] == 0 and a["verified_visible"]
    assert b["functional"] and b["source"] == "repair2" and b["repair_rounds"] == 2 and b["candidates_tried"] == 5
    assert not c["functional"] and c["source"] == "fallback_first_greedy" and c["repair_rounds"] == 3 and not c["verified_visible"] and c["text"] == wrong
    # tokens: greedy 5 + 2 samples 10 each (+ 10 per repair round)
    assert a["tokens"] == 25 and b["tokens"] == 25 + 20 and c["tokens"] == 25 + 30
    # resume: nothing regenerated for known keys
    n_calls = len(gen.calls)
    L.gwaya_loop(gen, FakeTok(), [p0], [greedy[p0.key]], j, "CX", workers=2)
    assert len(gen.calls) == n_calls


@needs_sandbox
def test_gate_uses_only_visible_test_in_loop(tmp_path):
    """A candidate that passes the visible test but fails a hidden one is 'verified' by the loop and judged wrong by the final judgement."""
    p = data.synthetic_sets()[1][0]  # add_two-like with 3 tests: visible = first
    hacked = block("def add_two(a, b):\n    return 3\n")
    gen = FakeGen({})
    j = L.Journal(tmp_path / "j.jsonl", False)
    prob = data.Problem(key="v", source="synthetic", task_id=1, user_text="x", visible_tests=["assert add_two(1, 2) == 3"],
                        all_tests=["assert add_two(1, 2) == 3", "assert add_two(5, 5) == 10"], messages=p.messages)
    recs = L.gwaya_loop(gen, FakeTok(), [prob], [{"text": hacked, "tokens": 3}], j, "CY", workers=1)
    assert recs[0]["verified_visible"] and not recs[0]["functional"] and recs[0]["candidates_tried"] == 1


# ---------------------------------------------------------------- LoRA merge equivalence on the tiny model
def test_merge_equals_lora_forward():
    model, tk = L.load_base("tiny", "cpu")
    plain, _ = L.load_base("tiny", "cpu")
    L.lib.attach_lora(model, L.LORA_TARGETS, L.LORA_R, L.LORA_ALPHA)
    L.lib.reset_lora_all(model, 3)
    g = torch.Generator().manual_seed(0)
    for m in model.modules():
        if isinstance(m, L.lib.LoRALinear):
            m.lora_B.data = torch.randn(m.lora_B.shape, generator=g) * 0.05
    st = L.lora_state(model)
    assert len(st) == 2 * 7 * 2 and all(v.dtype == torch.float32 for v in st.values())
    assert L.merge_lora_into_base(plain, st) == 14
    x = torch.randint(0, 256, (2, 12))
    with torch.no_grad():
        a, b = model(input_ids=x).logits, plain(input_ids=x).logits
    assert torch.allclose(a, b, atol=1e-4), float((a - b).abs().max())


def test_train_lora_reduces_loss_and_masks_prompt():
    model, tk = L.load_base("tiny", "cpu")
    cfg = dict(L.DESIGN["train"])
    cfg.update(epochs=3, lr=5e-3, warmup=1, grad_ckpt=True)
    pairs = [(tk.encode_prompt([{"role": "user", "content": f"q{i}"}]), tk.encode_completion(f"```python\ndef f():\n    return {i}\n```")) for i in range(8)]
    out = L.train_lora(model, tk, pairs, "cpu", False, cfg, progress=lambda m: None)
    assert out["steps"] == 3 * math.ceil(8 / 16) and out["skipped_nonfinite_steps"] == 0
    assert out["epoch_mean_ce"][-1] < out["epoch_mean_ce"][0]
    assert any(bool((v != 0).any()) for k, v in L.lora_state(model).items() if k.endswith("lora_B"))


# ---------------------------------------------------------------- data
def test_synthetic_sets_disjoint_and_references_valid():
    tr, ev, he = data.synthetic_sets()
    rep = data.assert_no_leakage(tr, ev)
    assert rep["id_overlap"] == 0 and (len(tr), len(ev), len(he)) == (6, 6, 2)


def test_leakage_assertion_fires():
    tr, ev, _ = data.synthetic_sets()
    with pytest.raises(AssertionError):
        data.assert_no_leakage(tr + [ev[0]], ev)
    dup = data.Problem(key="d", source="s", task_id=99999, user_text="  " + ev[1].user_text.upper(), visible_tests=[], all_tests=[])
    with pytest.raises(AssertionError):
        data.assert_no_leakage(tr + [dup], ev)


def test_prompt_format():
    tr, ev, he = data.synthetic_sets()
    m = ev[0].messages
    assert m[0] == {"role": "system", "content": "You are a careful Python programming assistant. Reply with exactly one ```python code block containing the complete solution."}
    assert m[1]["content"].endswith("Your code should pass this test:\n" + ev[0].visible_tests[0]) and ev[0].visible_tests[0] == ev[0].all_tests[0]
    assert he[0].messages[1]["content"] == he[0].user_text  # HumanEval: the user turn is the prompt


needs_cache = pytest.mark.skipif(not (data.HUB / "datasets--google-research-datasets--mbpp").exists() or not (data.HUB / "datasets--openai--openai_humaneval").exists(),
                                 reason="MBPP / HumanEval not in the HF cache")


@needs_cache
def test_real_splits_counts_ranges_and_decontamination():
    train, ev, he, leak = data.load_all()
    assert len(ev) == 257 and len(he) == 164 and leak["train_raw"] == 464
    assert leak["train_after_decontamination"] == len(train) == 464 - len(leak["removed_for_text_overlap"])
    assert all(511 <= p.task_id <= 974 for p in train) and all(11 <= p.task_id <= 510 for p in ev)
    assert not ({p.task_id for p in train} & {p.task_id for p in ev})
    assert all(len(p.visible_tests) == 1 and len(p.all_tests) >= 3 for p in ev[:20])
    assert ev[0].messages[1]["content"].count("assert") >= 1


@needs_cache
@needs_sandbox
def test_reference_solutions_pass_the_gate_sample():
    train, ev, he, _ = data.load_all()
    sample = train[:12] + he[:12]
    res = L.gate_many([(p.reference, p.all_tests, p.setup, True) for p in sample], 4)
    assert sum(r.accepted for r in res) >= len(sample) - 1, [(p.key, r.verdict) for p, r in zip(sample, res) if not r.accepted]


# ---------------------------------------------------------------- run script guards
def run_script(*args, env_extra=None):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "scripts" / "run_experiment_E15_gwaya_rft.py"), *args], capture_output=True, text=True, timeout=170, env=env, cwd="/")


def test_full_run_refuses_without_prereg(tmp_path):
    code = ("import sys\nsys.path.insert(0, %r)\nimport run_experiment_E15_gwaya_rft as m\nfrom pathlib import Path\n"
            "m.PREREG_PATH = Path(%r)\nsys.exit(m.main(['--out', %r]))\n") % (str(ROOT / "scripts"), str(tmp_path / "nope.json"), str(tmp_path / "o"))
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True, timeout=120, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, cwd="/")
    assert r.returncode == 2 and "refuses to start" in r.stderr and not (tmp_path / "o").exists()


def test_estimate_prints_plan():
    r = run_script("--estimate")
    assert r.returncode == 0
    d = json.loads(r.stdout)
    assert d["planned_generations"]["RFT samples"] == 462 * 8 and d["total_generations"] > 4000 and d["est_total_hours_excluding_gating_and_load"] > 0
    assert d["assumed_tokens_per_s_per_sequence"] == 25.0 and d["assumed_batch"] == 16


def test_smoke_requires_explicit_out():
    r = run_script("--smoke")
    assert r.returncode != 0
