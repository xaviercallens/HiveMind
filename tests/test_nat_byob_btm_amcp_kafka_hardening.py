# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and validation hardening tests for nat_traversal, byob, btm_upcycle, and amcp_kafka_transport."""

from __future__ import annotations

import pytest
import torch

from hivemind.core.btm_upcycle import (
    ExpertBranchState,
    LoopStepLoRAAdapter,
    _extract_ab_pair,
    measure_moe_vram_footprint_gb,
)
from hivemind.swarm.amcp_kafka_transport import (
    HMACKeyring,
    KafkaBrokerTransport,
    KafkaSettings,
    TransportRecord,
    build_dead_letter,
    partition_key_for,
)
from hivemind.swarm.byob import (
    Reason,
    ReasonCode,
    ServedBase,
    check_endpoint_url,
    hmac_hex,
    verify_signature,
)
from hivemind.swarm.nat_traversal import (
    BINDING,
    REQUEST,
    decode_xor_address,
    encode_xor_address,
    long_term_key,
    message_type,
    short_term_key,
    split_message_type,
)


def test_nat_traversal_helpers_validation() -> None:
    with pytest.raises(ValueError, match="method"):
        message_type(-1, 0)
    with pytest.raises(ValueError, match="method"):
        message_type(True, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cls"):
        message_type(BINDING, 4)
    mt = message_type(BINDING, REQUEST)
    assert split_message_type(mt) == (BINDING, REQUEST)
    with pytest.raises(ValueError, match="msg_type"):
        split_message_type(-1)
    with pytest.raises(ValueError, match="msg_type"):
        split_message_type(False)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="username"):
        long_term_key(123, "r", "p")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="password"):
        short_term_key(None)  # type: ignore[arg-type]
    assert len(long_term_key("u", "r", "p")) == 16
    assert short_term_key("secret") == b"secret"

    txid = b"0123456789ab"
    with pytest.raises(ValueError, match="addr"):
        encode_xor_address(("127.0.0.1", 70000), txid)
    with pytest.raises(ValueError, match="transaction_id"):
        encode_xor_address(("127.0.0.1", 3478), b"short")
    enc = encode_xor_address(("127.0.0.1", 3478), txid)
    assert decode_xor_address(enc, txid) == ("127.0.0.1", 3478)


def test_byob_dataclasses_and_helpers_validation() -> None:
    with pytest.raises(ValueError, match="hf_repo_id"):
        ServedBase("", "coder")
    with pytest.raises(ValueError, match="tier_id"):
        ServedBase("Qwen/Qwen2.5-Coder-7B-Instruct", " ")

    with pytest.raises(TypeError, match="code"):
        Reason("NOT_A_CODE", "detail")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="detail"):
        Reason(ReasonCode.SIGNATURE_MISSING, "   ")
    r = Reason(ReasonCode.SIGNATURE_MISSING, "missing")
    assert r.to_dict()["code"] == "SIGNATURE_MISSING"

    with pytest.raises(ValueError, match="HMAC key"):
        hmac_hex("not-bytes", b"payload")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="payload"):
        hmac_hex(b"k" * 32, "not-bytes")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="manifest"):
        verify_signature("not-manifest", {})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="url"):
        check_endpoint_url(123)  # type: ignore[arg-type]


def test_btm_upcycle_dataclasses_and_helpers_validation() -> None:
    with pytest.raises(ValueError, match="branch_id"):
        ExpertBranchState(branch_id=-1, domain_name="Code", outer_step=0, lora_state={})
    with pytest.raises(ValueError, match="domain_name"):
        ExpertBranchState(branch_id=0, domain_name=" ", outer_step=0, lora_state={})
    with pytest.raises(ValueError, match="outer_step"):
        ExpertBranchState(branch_id=0, domain_name="Code", outer_step=True, lora_state={})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lora_state"):
        ExpertBranchState(branch_id=0, domain_name="Code", outer_step=0, lora_state=[])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="lora_state"):
        _extract_ab_pair("bad", 0, "self_attn.q_proj")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="layer_idx"):
        _extract_ab_pair({}, -1, "self_attn.q_proj")
    with pytest.raises(ValueError, match="subpath"):
        _extract_ab_pair({}, 0, "")

    with pytest.raises(ValueError, match="hidden_size"):
        LoopStepLoRAAdapter(0, 4, torch.float32, torch.device("cpu"))
    with pytest.raises(ValueError, match="rank"):
        LoopStepLoRAAdapter(8, False, torch.float32, torch.device("cpu"))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="moe_model"):
        measure_moe_vram_footprint_gb("not-a-model")  # type: ignore[arg-type]


def test_amcp_kafka_transport_validation() -> None:
    settings = KafkaSettings(bootstrap="localhost:9092", security_protocol="PLAINTEXT")
    with pytest.raises(TypeError, match="settings"):
        KafkaBrokerTransport("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="flush_timeout_s"):
        KafkaBrokerTransport(settings, flush_timeout_s=0.0)

    kr = HMACKeyring(keys={"k1": b"a" * 32}, active_kid="k1")
    with pytest.raises(TypeError, match="message"):
        kr.sign("not-bytes")  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="unknown key id"):
        kr.sign(b"msg", kid="unknown")

    with pytest.raises(TypeError, match="event"):
        partition_key_for("not-an-event")  # type: ignore[arg-type]

    rec = TransportRecord(topic="t", partition=0, offset=1, key=b"k", value=b"v")
    with pytest.raises(TypeError, match="record"):
        build_dead_letter("bad", "malformed", "detail", 1700000000.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reason"):
        build_dead_letter(rec, " ", "detail", 1700000000.0)
    with pytest.raises(TypeError, match="detail"):
        build_dead_letter(rec, "malformed", 123, 1700000000.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="now_utc"):
        build_dead_letter(rec, "malformed", "detail", float("inf"))
