"""CPU tests for the pluggable KV page codecs (roadmap HM-S10-05). FP8 stays the default; no adoption claim."""

from __future__ import annotations

import pytest
import torch

from hivemind.core.btm_moe import FP8PagedKVCache
from hivemind.core.kv_page_codecs import (
    FP8E4M3PageCodec,
    IntMinMaxPageCodec,
    KVPageCodec,
    LatticePageCodec,
    PluggableKVPageCache,
)

D = 16


def _page(seed: int, tokens: int = 8, heads: int = 2) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(tokens, heads, D, generator=g) * 3.0


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


def test_protocol_membership():
    for c in (FP8E4M3PageCodec(), IntMinMaxPageCodec(4), LatticePageCodec("Z", 0.5)):
        assert isinstance(c, KVPageCodec)


def test_default_codec_is_fp8():
    assert isinstance(PluggableKVPageCache().codec, FP8E4M3PageCodec)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_default_bit_identical_to_fp8_paged_cache(dtype):
    ref, new = FP8PagedKVCache(max_pages=4), PluggableKVPageCache(max_pages=4)
    for i in range(3):
        k, v = _page(i), _page(100 + i)
        r_info = ref.write_kv_page(0, i, k, v)
        n_info = new.write_kv_page(0, i, k, v)
        assert r_info["fp8_paged_bytes"] == n_info["fp8_paged_bytes"]
        assert r_info["compression_ratio"] == n_info["compression_ratio"]
        rk, rv = ref.read_kv_page(0, i, dtype)
        nk, nv = new.read_kv_page(0, i, dtype)
        assert torch.equal(rk, nk) and torch.equal(rv, nv)
    assert ref.total_kv_cache_bytes() == new.total_kv_cache_bytes()


def test_int_error_ordering():
    x = _page(1)
    errs = {}
    for b in (2, 4, 8):
        c = IntMinMaxPageCodec(b)
        errs[b] = _rel(c.decode(c.encode(x), torch.float32), x)
    assert errs[2] > errs[4] > errs[8]


@pytest.mark.parametrize("kind", ["Z", "A2", "E8"])
def test_lattice_tiny_scale_reproduces(kind):
    x = _page(2)
    c = LatticePageCodec(kind, 1e-4)
    y = c.decode(c.encode(x), torch.float32)
    assert y.shape == x.shape
    assert _rel(y, x) < 1e-3


@pytest.mark.parametrize("kind", ["Z", "A2", "E8"])
def test_lattice_coarser_scale_larger_error(kind):
    x = _page(3)
    e = [_rel(LatticePageCodec(kind, s).decode(LatticePageCodec(kind, s).encode(x), torch.float32), x) for s in (0.1, 1.0)]
    assert e[0] < e[1]


def test_payload_bits_sanity():
    x = _page(4)
    rows, n = x.numel() // D, x.numel()
    assert FP8E4M3PageCodec().payload_bits(FP8E4M3PageCodec().encode(x)) == n * 8 + rows * 16
    for b in (2, 4):
        c = IntMinMaxPageCodec(b)
        assert c.payload_bits(c.encode(x)) == n * b + rows * 32
    for kind in ("Z", "A2", "E8"):
        c = LatticePageCodec(kind, 1.0)
        p = c.encode(x)
        bits = c.payload_bits(p)
        assert bits == pytest.approx(p["rate"] * n + rows * 16)
        assert bits > rows * 16  # rate is positive
        assert bits < n * 16 + rows * 16  # far below fp16 at this scale


def test_pluggable_cache_with_other_codecs_round_trips():
    for codec in (IntMinMaxPageCodec(4), LatticePageCodec("E8", 0.5)):
        cache = PluggableKVPageCache(codec=codec)
        k, v = _page(5), _page(6)
        info = cache.write_kv_page(1, 0, k, v)
        assert info["codec"] == codec.name
        rk, rv = cache.read_kv_page(1, 0, torch.float32)
        assert rk.shape == k.shape and rv.shape == v.shape
        assert cache.total_kv_cache_bytes() > 0


def test_eviction_mirrors_fp8_paged_cache():
    ref, new = FP8PagedKVCache(max_pages=2), PluggableKVPageCache(max_pages=2)
    for cache in (ref, new):
        for p in range(3):
            cache.write_kv_page(0, p, _page(p), _page(10 + p))
        # rewriting an existing key must not evict
        cache.write_kv_page(0, 2, _page(7), _page(8))
    for cache in (ref, new):
        with pytest.raises(KeyError):
            cache.read_kv_page(0, 0)
        cache.read_kv_page(0, 1)
        cache.read_kv_page(0, 2)
    assert list(ref._pages_k) == list(new._pages_k)
    assert ref.total_kv_cache_bytes() == new.total_kv_cache_bytes()


@pytest.mark.parametrize("codec", [IntMinMaxPageCodec(8), LatticePageCodec("Z", 1.0), LatticePageCodec("A2", 1.0), LatticePageCodec("E8", 1.0)])
def test_encode_refuses_nan_and_fp16_overflow_but_accepts_a_normal_page(codec) -> None:
    g = torch.Generator().manual_seed(0)
    ok = torch.randn(4, 16, generator=g)
    assert torch.isfinite(codec.decode(codec.encode(ok), torch.float32)).all()
    nan_page = ok.clone()
    nan_page[0, 0] = float("nan")
    with pytest.raises(ValueError):
        codec.encode(nan_page)
    with pytest.raises(ValueError):
        codec.encode(ok * 1e6)  # RMS / min / max above fp16's 65504 overflow the stored side information


@pytest.mark.parametrize("codec", [IntMinMaxPageCodec(8), LatticePageCodec("Z", 1.0), LatticePageCodec("A2", 1.0), LatticePageCodec("E8", 1.0)])
def test_encode_refuses_a_page_that_would_decode_to_zeros_but_keeps_an_all_zero_page(codec) -> None:
    g = torch.Generator().manual_seed(1)
    with pytest.raises(ValueError):
        codec.encode(torch.randn(4, 16, generator=g) * 1e-8)
    zero = codec.decode(codec.encode(torch.zeros(4, 16)), torch.float32)  # an all-zero page is legitimate and round-trips
    assert torch.count_nonzero(zero) == 0
    ok = torch.randn(4, 16, generator=g)
    assert torch.count_nonzero(codec.decode(codec.encode(ok), torch.float32)) > 0


def test_fp8_codec_refuses_nan_fp16_scale_overflow_and_underflow_but_keeps_normal_and_zero_pages() -> None:
    codec = FP8E4M3PageCodec()
    g = torch.Generator().manual_seed(3)
    ok = torch.randn(4, 16, generator=g)
    assert torch.isfinite(codec.decode(codec.encode(ok), torch.float32)).all()
    assert torch.count_nonzero(codec.decode(codec.encode(ok), torch.float32)) > 0
    nan_page = ok.clone()
    nan_page[1, 2] = float("nan")
    with pytest.raises(ValueError):
        codec.encode(nan_page)
    inf_page = ok.clone()
    inf_page[0, 0] = float("inf")
    with pytest.raises(ValueError):
        codec.encode(inf_page)
    with pytest.raises(ValueError):
        codec.encode(ok * 1e9)  # the per-row scale (absmax / 448) is above fp16's 65504
    with pytest.raises(ValueError):
        codec.encode(ok * 1e-9)  # the per-row scale rounds to 0 in fp16: the page would decode to zeros
    zero = codec.decode(codec.encode(torch.zeros(4, 16)), torch.float32)
    assert torch.isfinite(zero).all() and torch.count_nonzero(zero) == 0  # was NaN: scale 0 in fp16 gave 0/0
    mixed = ok.clone()
    mixed[2] = 0.0
    dec = codec.decode(codec.encode(mixed), torch.float32)
    assert torch.isfinite(dec).all() and torch.count_nonzero(dec[2]) == 0 and torch.count_nonzero(dec[0]) > 0


def test_a_refused_write_does_not_evict_a_cached_page() -> None:
    cache = PluggableKVPageCache(max_pages=1, codec=IntMinMaxPageCodec(8))
    g = torch.Generator().manual_seed(4)
    good_k, good_v = torch.randn(4, 16, generator=g), torch.randn(4, 16, generator=g)
    cache.write_kv_page(0, 0, good_k, good_v)
    bad = good_k.clone()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError):
        cache.write_kv_page(0, 1, bad, good_v)
    with pytest.raises(ValueError):
        cache.write_kv_page(0, 1, good_k, bad)  # the V half is refused after K encoded: still nothing may change
    assert list(cache._pages_k) == [(0, 0)] and list(cache._pages_v) == [(0, 0)]
    k, v = cache.read_kv_page(0, 0, torch.float32)
    assert torch.isfinite(k).all() and torch.isfinite(v).all()
    cache.write_kv_page(0, 1, good_v, good_k)  # a good write still evicts the oldest page
    assert list(cache._pages_k) == [(0, 1)]
