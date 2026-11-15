"""HiveMind JEPA-LLM library (src/hivemind/jepa): CPU-only tests, synthetic tensors and a tiny random model, no download."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hivemind.jepa import pairs, stp  # noqa: E402
from hivemind.jepa.spec import JepaSpec, apply_aux, step_uniform  # noqa: E402

EOS = 0


# ------------------------------------------------------------------------------------------------------------ spec


def test_spec_validation_hash_and_version() -> None:
    s = JepaSpec(mode="stp", lam=0.02, eos_id=EOS)
    assert s.active and JepaSpec().active is False
    assert s.spec_sha256() == JepaSpec.from_dict(__import__("json").loads(s.to_json())).spec_sha256()
    assert s.spec_sha256() != JepaSpec(mode="stp", lam=0.04, eos_id=EOS).spec_sha256()
    for bad in (dict(version=2), dict(mode="x"), dict(stp_variant="y"), dict(lam=-1), dict(dropout=1.5),
                dict(min_segment=2), dict(stp_samples=0), dict(eos_id=-2), dict(lam=float("nan"))):
        with pytest.raises(ValueError):
            JepaSpec(**bad)
    with pytest.raises(ValueError):
        JepaSpec.from_dict({"mode": "stp", "surprise": 1})


def test_loss_dropout_is_deterministic_and_has_the_right_rate() -> None:
    s = JepaSpec(mode="stp", lam=0.02, dropout=0.5, seed=3)
    a = [apply_aux(s, i) for i in range(2000)]
    assert a == [apply_aux(s, i) for i in range(2000)]
    assert abs(sum(a) / 2000 - 0.5) < 0.04
    assert all(apply_aux(JepaSpec(mode="stp", lam=0.02), i) for i in range(10))
    assert not apply_aux(JepaSpec(mode="stp", lam=0.0), 0)
    assert 0.0 <= step_uniform(1, 2, 3, 4) < 1.0


# ------------------------------------------------------------------------------------------- documents and triples


def test_document_segments_split_on_eos_and_drop_short_ones() -> None:
    row = [5, 6, 7, 8, EOS, 9, 9, EOS, 1, 2, 3, 4, 5, 6]
    assert stp.document_segments(row, EOS, 3) == [(0, 4), (8, 14)]
    assert stp.document_segments(row, -1, 3) == [(0, 14)]
    assert stp.document_segments([1, 2], -1, 3) == []
    with pytest.raises(ValueError):
        stp.document_segments(row, EOS, 2)


def test_triples_are_ordered_inside_one_document_and_deterministic() -> None:
    row = [7] * 9 + [EOS] + [3] * 4 + [EOS] + [5] * 20
    segs = stp.document_segments(row, EOS, 3)
    for seed in range(5):
        for step in range(60):
            trips = stp.sample_triples(segs, 3, seed, step, 0)
            assert trips == stp.sample_triples(segs, 3, seed, step, 0)
            for s, r, t in trips:
                assert s < r < t
                assert any(a <= s and t < b for a, b in segs)  # never across an EOS
                assert row[s] != EOS and row[r] != EOS and row[t] != EOS
    assert stp.sample_triples([], 2, 0, 0, 0) == []


# ---------------------------------------------------------------------------------------------- geodesic (paper) loss


def test_geodesic_loss_on_known_trajectories() -> None:
    T, D = 12, 6
    v = torch.randn(D)
    straight = (torch.arange(T).float()[:, None] * v + torch.randn(D))[None]
    loss, n = stp.geodesic_loss(straight, [[(0, 4, 9), (2, 3, 11)]])
    assert n == 2 and float(loss) < 1e-5  # a straight line is a perfect geodesic
    back = straight.clone()
    back[0, 9] = back[0, 4] - (back[0, 4] - back[0, 0])  # h_t - h_r = -(h_r - h_s)
    loss, _ = stp.geodesic_loss(back, [[(0, 4, 9)]])
    assert abs(float(loss) - 2.0) < 1e-5
    h = torch.randn(2, T, D)
    loss, _ = stp.geodesic_loss(h, [[(1, 2, 5)], [(0, 6, 7)]])
    want = [1 - torch.nn.functional.cosine_similarity(h[i, t] - h[i, r], h[i, r] - h[i, s], dim=0)
            for i, (s, r, t) in enumerate([(1, 2, 5), (0, 6, 7)])]
    assert abs(float(loss) - float(sum(want) / 2)) < 1e-6
    assert stp.geodesic_loss(h, [[], []]) == (None, 0)


def test_stp_term_on_packed_rows_and_health() -> None:
    spec = JepaSpec(mode="stp", lam=0.02, eos_id=EOS, min_segment=4, stp_samples=2, seed=1)
    ids = torch.tensor([[3, 3, 3, 3, 3, EOS, 4, 4], [EOS, 1, EOS, 2, EOS, 3, EOS, 4]])
    h = torch.randn(2, 8, 5, requires_grad=True)
    loss, info = stp.stp_term(spec, h, ids, step=7)
    assert info == {"triples": 2, "rows_without_segment": 1}
    loss.backward()
    assert float(h.grad[1].abs().sum()) == 0.0  # the row with only 1-token documents contributes nothing
    assert stp.health(0.01, spec) == "collapse" and stp.health(0.6, spec) == "ok"
    with pytest.raises(ValueError):
        stp.stp_term(JepaSpec(mode="stp", lam=0.02, stp_variant="random_span"), h, ids, 0)


# ------------------------------------------------------------------------------------- released-code random_span variant


def test_random_span_split_matches_a_hand_computed_case() -> None:
    # user tokens [2, 5), assistant tokens [7, 10): combined length 6. Patch [2, 4) covers user token 4 and assistant 7.
    segs = [(2, 5), (7, 10)]
    before, patch, after = stp._split(segs, 2, 4)
    assert (before, patch, after) == ([(2, 4)], [(4, 5), (7, 8)], [(8, 10)])
    h = torch.randn(1, 12, 4)
    cos, w = stp.random_span_cos(h, 0, segs, 2, 4)
    rest = (h[0, 3] - h[0, 1]) + (h[0, 9] - h[0, 7])
    pat = (h[0, 4] - h[0, 3]) + (h[0, 7] - h[0, 6])
    assert w == 1.0 and abs(float(cos) - float(torch.nn.functional.cosine_similarity(rest, pat, dim=0))) < 1e-6
    _, wc = stp.random_span_cos(h, 0, segs, 2, 4, "cosine_like")
    assert abs(wc - 2 * 4 * 2 / (16 + 4)) < 1e-12
    agg = stp.random_span_aggregate([torch.tensor(0.5), torch.tensor(-0.5)], [3.0, 1.0])
    assert abs(float(agg) - (1 - (1.5 - 0.5) / 4)) < 1e-6
    with pytest.raises(ValueError):
        stp.random_span_cos(h, 0, segs, 0, 6)


# ------------------------------------------------------------------------------------------------------- paired views


def test_pair_term_stopgrad_center_and_infonce() -> None:
    spec = JepaSpec(mode="pairs", lam=0.1)  # stopgrad on by default
    a = torch.randn(4, 8, requires_grad=True)
    b = torch.randn(4, 8, requires_grad=True)
    pairs.pair_term(spec, a, b).backward()
    assert float(a.grad.abs().sum()) > 0 and b.grad is None
    same = torch.ones(3, 8) * 5
    c = same + 0.01 * torch.randn(3, 8)
    assert float(pairs.pair_term(spec, same + 0.01 * torch.randn(3, 8), c)) < 0.01
    assert float(pairs.pair_term(JepaSpec(mode="pairs", lam=0.1, center=True), same + 0.01 * torch.randn(3, 8), c)) > 0.3
    x = torch.randn(8, 16)
    assert abs(float(pairs.pair_term(spec, x, torch.randn(8, 16), infonce=True)) - math.log(8)) < 2.5
    m = torch.tensor([[1, 1, 0], [1, 1, 1]])
    hh = torch.randn(2, 3, 4)
    assert torch.equal(pairs.last_token_states(hh, m), hh[[0, 1], [1, 2]])


# ------------------------------------------------------------------------------------------- one-pass HF integration


@pytest.fixture(scope="module")
def tiny_model():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.LlamaConfig(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM(cfg).eval()


def test_one_pass_loss_equals_hf_loss_when_stp_is_off_and_adds_lam_times_stp(tiny_model) -> None:
    from hivemind.jepa.hf import ntp_and_stp

    torch.manual_seed(1)
    ids = torch.randint(1, 50, (2, 24))
    ids[0, 10] = EOS
    labels = ids.clone()
    labels[:, :3] = -100
    ref = tiny_model(input_ids=ids, labels=labels).loss
    off, info = ntp_and_stp(tiny_model, ids, labels, JepaSpec(), step=0)
    assert abs(float(off) - float(ref)) < 1e-5 and info["aux"] is None
    spec = JepaSpec(mode="stp", lam=0.05, eos_id=EOS, min_segment=4, stp_samples=2, seed=2)
    on, info = ntp_and_stp(tiny_model, ids, labels, spec, step=5)
    assert info["triples"] == 4 and info["health"] in ("ok", "collapse")
    assert abs(float(on) - (info["ntp"] + 0.05 * info["aux"])) < 1e-5
    on2, info2 = ntp_and_stp(tiny_model, ids, labels, spec, step=5)
    assert float(on2) == float(on) and info2 == info  # replayable: same step, same loss
    on.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in tiny_model.parameters())
