"""Negative-control boundary-validation tests for QueenBee (dance, knowledge_market)
and SolverBee (verifier, science_bee).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from hivemind.queenbee.dance import (
    DanceBoard,
    DanceRecord,
    RecruitmentReport,
    derive_signer_secret,
    dual_scout_bounds,
    recruitment_weights,
)
from hivemind.queenbee.knowledge_market import (
    Admission,
    KnowledgeMarket,
    KnowledgeSettlement,
    KnowledgeShare,
    admit_share,
    certify_knowledge_bonus,
    firewall_scanner,
    knowledge_points,
    settle_knowledge_bonus,
)
from hivemind.router.market import FeePolicy, MarketEquilibrium
from hivemind.solverbee.science_bee import (
     BackendError,
    BaselineBackend,
    SlotBackend,
    _increasing,
    _slot_times,
    _times,
)
from hivemind.solverbee.verifier import (
    _CheckError,
    _check_invariants,
    _inv_energy,
    _inv_linear,
    _inv_lv_first_integral,
    _inv_quadratic_form,
    _tolerances,
    _validate_trajectory,
)


def _sample_record() -> DanceRecord:
    return DanceRecord(
        expert_id="exp-1",
        served_by="exp-1",
        cluster_id="code",
        n=10,
        successes=8,
        cost_per_1k=0.05,
        epoch=5,
        verifier_kind="unit_tests",
        signer_id="node-1",
    )


def test_dance_record_and_board_boundary_validation(tmp_path: Path) -> None:
    rec = _sample_record()
    signed = rec.signed(b"master-secret")
    assert signed.verify(b"master-secret") is True

    with pytest.raises(ValueError, match="non-empty strings"):
        DanceRecord(
            expert_id=123,  # type: ignore[arg-type]
            served_by="exp-1",
            cluster_id="code",
            n=10,
            successes=8,
            cost_per_1k=0.05,
            epoch=5,
            verifier_kind="unit_tests",
            signer_id="node-1",
        )
    with pytest.raises(ValueError, match="signature_hex"):
        DanceRecord(
            expert_id="exp-1",
            served_by="exp-1",
            cluster_id="code",
            n=10,
            successes=8,
            cost_per_1k=0.05,
            epoch=5,
            verifier_kind="unit_tests",
            signer_id="node-1",
            signature_hex=123,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="prederived"):
        rec.signed(b"master-secret", prederived="no")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="secret"):
        rec.signed(b"")
    with pytest.raises(ValueError, match="secret"):
        signed.verify("not-bytes")  # type: ignore[arg-type]
    assert signed.verify({"node-1": b""}) is False

    with pytest.raises(ValueError, match="data must be a mapping"):
        DanceRecord.from_dict([])  # type: ignore[arg-type]
    d = signed.to_dict()
    d["signature_hex"] = 999
    with pytest.raises(ValueError, match="signature_hex"):
        DanceRecord.from_dict(d)

    with pytest.raises(ValueError, match="master_secret"):
        derive_signer_secret("not-bytes", "node-1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="signer_id"):
        derive_signer_secret(b"secret", 123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="records"):
        dual_scout_bounds("not-records", ["node-1"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scout_ids"):
        dual_scout_bounds([signed], [""])
    with pytest.raises(ValueError, match="secret"):
        dual_scout_bounds([signed], ["node-1"], secret=b"")
    with pytest.raises(ValueError, match="DanceRecord"):
        dual_scout_bounds(["bad"], ["node-1"])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="z must be finite"):
        recruitment_weights([signed], now_epoch=5, horizon_epochs=2, cap_per_record=0.5, secret=b"s", z=0.0)
    with pytest.raises(ValueError, match="max_trials_per_signer"):
        recruitment_weights(
            [signed],
            now_epoch=5,
            horizon_epochs=2,
            cap_per_record=0.5,
            secret=b"s",
            max_trials_per_signer=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="secret"):
        recruitment_weights([signed], now_epoch=5, horizon_epochs=2, cap_per_record=0.5, secret=b"")
    with pytest.raises(ValueError, match="scout_bounds"):
        recruitment_weights(
            [signed],
            now_epoch=5,
            horizon_epochs=2,
            cap_per_record=0.5,
            secret=b"s",
            scout_bounds={("code", "exp-1"): -0.1},
        )
    with pytest.raises(ValueError, match="DanceRecord"):
        recruitment_weights(["bad"], now_epoch=5, horizon_epochs=2, cap_per_record=0.5, secret=b"s")  # type: ignore[list-item]

    rep = recruitment_weights([signed], now_epoch=5, horizon_epochs=2, cap_per_record=0.5, secret=b"master-secret")
    with pytest.raises(ValueError, match="cluster_id"):
        rep.probabilities("", ["exp-1"])

    with pytest.raises(ValueError, match="path"):
        DanceBoard("")
    board_path = tmp_path / "dance.jsonl"
    board = DanceBoard(board_path)
    with pytest.raises(ValueError, match="DanceRecord"):
        board.append("not-a-record")  # type: ignore[arg-type]
    board.append(signed)
    with board_path.open("a", encoding="utf-8") as fh:
        fh.write("[1, 2, 3]\n123\n")
    loaded, bad_cnt = board.load()
    assert len(loaded) == 1
    assert bad_cnt == 2


def test_knowledge_market_boundary_validation() -> None:
    txt = "def f(x): return x + 1"
    sha = hashlib.sha256(txt.encode("utf-8")).hexdigest()
    rcpt = "a" * 64
    share = KnowledgeShare(
        account_id="acc-1",
        kind="verified_code",
        licence="MIT",
        verified=True,
        epoch=1,
        text_sha256=sha,
        verifier_receipt_sha256=rcpt,
    )
    with pytest.raises(ValueError, match="account_id"):
        KnowledgeShare(
            account_id=None,  # type: ignore[arg-type]
            kind="verified_code",
            licence="MIT",
            verified=True,
            epoch=1,
            text_sha256=sha,
        )
    with pytest.raises(ValueError, match="licence"):
        KnowledgeShare(
            account_id="acc-1",
            kind="verified_code",
            licence="",
            verified=True,
            epoch=1,
            text_sha256=sha,
        )
    with pytest.raises(ValueError, match="verifier_receipt_sha256"):
        KnowledgeShare(
            account_id="acc-1",
            kind="verified_code",
            licence="MIT",
            verified=True,
            epoch=1,
            text_sha256=sha,
            verifier_receipt_sha256=123,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="text must be a string"):
        firewall_scanner(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scanner"):
        admit_share(share, txt, scanner="not-callable")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seen_shares"):
        admit_share(share, txt, seen_shares=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="epoch_counts"):
        admit_share(share, txt, epoch_counts=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="verifier_receipt_sha256"):
        admit_share(share, txt, verifier_receipt_sha256=123)  # type: ignore[arg-type]

    adm = admit_share(share, txt, scanner=lambda _t: True)
    assert isinstance(adm, Admission) and adm.admitted is True

    with pytest.raises(ValueError, match="admissions"):
        knowledge_points("not-admissions")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Admission"):
        knowledge_points(["not-an-admission"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="scanner"):
        KnowledgeMarket(1, scanner="not-callable")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="epoch"):
        KnowledgeSettlement(-1, 10.0, 0.2, 2.0, {}, {})
    with pytest.raises(ValueError, match="points"):
        settle_knowledge_bonus(1, 10.0, 0.2, {"acc-1": 1.5})  # type: ignore[dict-item]

    settlement = settle_knowledge_bonus(1, 10.0, 0.2, {"acc-1": 2})
    fee = FeePolicy()
    eq = MarketEquilibrium(
        model_id="m1",
        mechanism="kelly",
        price_per_1k=1.0,
        supply_1k=10.0,
        consumer_shares={"c1": 1.0},
        consumer_surplus={"c1": 5.0},
        active_producers=("p1",),
        producer_profit={"p1": 4.0},
        producer_revenue_per_1k=0.85,
        fee=fee,
    )
    cert = certify_knowledge_bonus(eq, eq, settlement, fee, 100.0, True, True)
    assert cert.ok is True
    with pytest.raises(ValueError, match="MarketEquilibrium"):
        certify_knowledge_bonus("bad", eq, settlement, fee, 100.0, True, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="KnowledgeSettlement"):
        certify_knowledge_bonus(eq, eq, "bad", fee, 100.0, True, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="FeePolicy"):
        certify_knowledge_bonus(eq, eq, settlement, "bad", 100.0, True, True)  # type: ignore[arg-type]


def test_solverbee_verifier_and_science_bee_boundary_validation() -> None:
    with pytest.raises(_CheckError, match="res and spec"):
        _validate_trajectory([], {}, "result")  # type: ignore[arg-type]
    with pytest.raises(_CheckError, match="what"):
        _validate_trajectory({}, {}, "")
    with pytest.raises(_CheckError, match="spec must be a dict"):
        _tolerances([], 2)  # type: ignore[arg-type]
    with pytest.raises(_CheckError, match="n must be a positive integer"):
        _tolerances({}, 0)

    with pytest.raises(_CheckError, match="inv and spec"):
        _inv_linear([], {}, [1.0])  # type: ignore[arg-type]
    with pytest.raises(_CheckError, match="inv and spec"):
        _inv_quadratic_form({}, {}, [])
    with pytest.raises(_CheckError, match="inv and spec"):
        _inv_energy({}, {}, [float("nan")])
    with pytest.raises(_CheckError, match="inv and spec"):
        _inv_lv_first_integral([], {}, [1.0, 2.0])  # type: ignore[arg-type]
    with pytest.raises(_CheckError, match="invariants must be a list"):
        _check_invariants({}, {}, {"invariants": "not-a-list"})
    with pytest.raises(_CheckError, match="each invariant must be an object"):
        _check_invariants({}, {}, {"invariants": ["not-a-dict"]})

    with pytest.raises(ValueError, match="text must be a string"):
        _times(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="text must be a string"):
        _slot_times(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ts must be a list"):
        _increasing([1.0, float("nan")])
    assert _increasing([0.5, 1.0, 2.0]) == [0.5, 1.0, 2.0]

    with pytest.raises(BackendError, match="messages"):
        BaselineBackend().complete([])
    with pytest.raises(BackendError, match="messages"):
        SlotBackend().complete([])
