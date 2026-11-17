"""CPU-only unit tests for helper functions of scripts/run_experiment_E12_replay_divergence.py (no model, no GPU)."""

import os

_CUDA_VISIBLE_DEVICES_BEFORE = os.environ.get("CUDA_VISIBLE_DEVICES")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # for the imports below only; restored after them

import math  # noqa: E402
import sys  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import run_experiment_E12_replay_divergence as e12  # noqa: E402

# Restored right after the imports above: setting it at module level hid the GPU from every test module collected
# after this one (IP-179); the autouse fixture below keeps this module's own tests CPU-only at run time.
if _CUDA_VISIBLE_DEVICES_BEFORE is None:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _CUDA_VISIBLE_DEVICES_BEFORE


@pytest.fixture(autouse=True)
def _cpu_only_for_this_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def _state_a():
    return {"w": torch.tensor([3.0, 4.0]), "v": torch.tensor([[1.0, -2.0]])}


def _state_b():
    return {"w": torch.tensor([3.0, 4.5]), "v": torch.tensor([[1.25, -2.0]])}


def test_linf_zero_divergence():
    a = _state_a()
    b = {k: t.clone() for k, t in a.items()}
    assert e12.linf(a, b) == (0.0, 0.0)


def test_linf_known_max_abs_and_relative_l2():
    a, b = _state_a(), _state_b()
    m, rel = e12.linf(a, b)
    # abs diffs: w -> [0, 0.5], v -> [0.25, 0]; max = 0.5
    assert m == pytest.approx(0.5, abs=1e-12)
    # sum of squared diffs = 0.25 + 0.0625 = 0.3125; sum of squares of a = 9 + 16 + 1 + 4 = 30
    assert rel == pytest.approx(math.sqrt(0.3125 / 30.0), rel=1e-12)


def test_compare_zero_and_nonzero_divergence():
    a, b = _state_a(), _state_b()
    x = {"seg": {10: _state_a(), 50: a}}
    y_same = {"seg": {10: {k: t.clone() for k, t in a.items()}, 50: {k: t.clone() for k, t in a.items()}}}
    rows = e12.compare(x, y_same)
    assert len(rows) == 1
    assert rows[0]["linf_at_50"] == 0.0
    assert rows[0]["rel_l2_at_50"] == 0.0

    y_div = {"seg": {10: _state_a(), 50: b}}
    rows = e12.compare(x, y_div)
    assert rows[0]["linf_at_50"] == pytest.approx(0.5, abs=1e-12)
    assert rows[0]["rel_l2_at_50"] == pytest.approx(math.sqrt(0.3125 / 30.0), rel=1e-12)
    assert rows[0]["linf_curve"]["10"] == 0.0
    assert rows[0]["linf_curve"]["50"] == pytest.approx(0.5, abs=1e-12)


def test_lora_linear_forward_equals_base_when_b_is_zero():
    base = nn.Linear(16, 8)
    gen = torch.Generator().manual_seed(0)
    lora = e12.LoRALinear(base, r=4, alpha=8, gen=gen)
    assert float(lora.lora_B.abs().sum()) == 0.0
    x = torch.randn(3, 16, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = base(x)
        out = lora(x)
    assert torch.equal(out, ref)


def test_lora_linear_trainable_params_and_frozen_base():
    base = nn.Linear(16, 8)
    gen = torch.Generator().manual_seed(0)
    lora = e12.LoRALinear(base, r=4, alpha=8, gen=gen)
    trainable = [n for n, p in lora.named_parameters() if p.requires_grad]
    assert trainable, "expected trainable LoRA parameters"
    assert all("lora_" in n for n in trainable)
    assert set(trainable) == {"lora_A", "lora_B"}
    assert lora.base.weight.requires_grad is False
    assert lora.base.bias.requires_grad is False
    assert lora.lora_A.shape == (4, 16)
    assert lora.lora_B.shape == (8, 4)
    assert lora.scaling == pytest.approx(2.0)
