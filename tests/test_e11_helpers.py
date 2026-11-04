"""CPU-only unit tests for helper functions of scripts/run_experiment_E11_kv_lattice.py (no model, no GPU)."""

import os

_CUDA_VISIBLE_DEVICES_BEFORE = os.environ.get("CUDA_VISIBLE_DEVICES")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # for the imports below only; restored after them

import math  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import run_experiment_E11_kv_lattice as e11  # noqa: E402

# Restored right after the imports above: setting it at module level hid the GPU from every test module collected
# after this one (IP-179); the autouse fixture below keeps this module's own tests CPU-only at run time.
if _CUDA_VISIBLE_DEVICES_BEFORE is None:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _CUDA_VISIBLE_DEVICES_BEFORE


@pytest.fixture(autouse=True)
def _cpu_only_for_this_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_int_codec_levels_and_range(bits):
    gen = torch.Generator().manual_seed(bits)
    x = torch.randn(4, 64, generator=gen) * 3.0 + 0.5
    xh = e11.int_codec(x, bits)
    assert xh.dtype == x.dtype
    assert xh.shape == x.shape
    for r in range(x.shape[0]):
        row = xh[r]
        assert torch.unique(row).numel() <= 2**bits
        lo = float(x[r].min().half().float())
        hi = float(x[r].max().half().float())
        assert float(row.min()) >= lo - 1e-5
        assert float(row.max()) <= hi + 1e-5


def test_fp8_codec_relative_error_on_moderate_magnitudes():
    gen = torch.Generator().manual_seed(0)
    mag = 10.0 ** (torch.rand(20000, generator=gen) * 2.0 - 1.0)  # log-uniform in [0.1, 10]
    sign = torch.where(torch.rand(20000, generator=gen) < 0.5, -1.0, 1.0)
    x = (mag * sign).float()
    xh = e11.fp8_codec(x)
    assert xh.shape == x.shape
    rel = ((xh.double() - x.double()).abs() / x.double().abs())
    assert float(rel.max()) < 0.07


@pytest.mark.parametrize("transform", ["none", "klt", "kltc"])
def test_transform_round_trip(transform):
    gen = torch.Generator().manual_seed(7)
    capture = {(0, 0): [torch.randn(1, 2, 300, 64, generator=gen, dtype=torch.float64)]}
    klt = e11.klt_bases(capture)
    tbl = klt[(0, 0)]
    x = torch.randn(1, 2, 300, 64, generator=gen, dtype=torch.float64)
    y = e11.transform_fwd(transform, tbl, x)
    xr = e11.transform_inv(transform, tbl, y)
    assert xr.shape == x.shape
    assert torch.allclose(xr, x, atol=1e-10, rtol=0.0)


def test_window_indices_per_seed_in_range_and_disjoint():
    n = 292
    sets = {}
    for seed in (1, 2, 3):
        idx = e11.window_indices(n, seed)
        assert len(idx) == e11.WINDOWS_PER_SEED == 8
        assert len(set(idx)) == 8
        assert all(0 <= i < n for i in idx)
        sets[seed] = set(idx)
    assert sets[1].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[3])
    assert sets[2].isdisjoint(sets[3])


def test_mean_ci_three_known_values():
    xs = [1.0, 2.0, 4.0]
    m, h = e11.mean_ci(xs)
    assert m == pytest.approx(7.0 / 3.0, abs=1e-12)
    sd = math.sqrt(7.0 / 3.0)  # sample sd of [1, 2, 4]
    assert sd == pytest.approx(statistics.stdev(xs), abs=1e-12)
    assert h == pytest.approx(4.302652729911275 * sd / math.sqrt(3), rel=1e-12)


def test_mean_ci_single_value_has_zero_width():
    assert e11.mean_ci([5.0]) == (5.0, 0.0)


def test_cell_names_contents():
    cells = e11.cell_names()
    assert len(cells) == len(set(cells))
    assert "none" in cells
    assert "fp8" in cells
    assert "A2@8|none" in cells
    pattern = re.compile(r"^(Z|A2|E8)@[234]\|(none|klt)$")
    lattice = [c for c in cells if pattern.match(c)]
    assert len(lattice) == 18
    assert all(("|klt" in c) or ("|none" in c) for c in lattice)
