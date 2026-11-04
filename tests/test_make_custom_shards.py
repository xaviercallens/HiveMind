# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""MVP criterion B1: custom JSONL -> signed shards + disjoint held-out split, with no network and with tamper detection."""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_custom_shards as mcs  # noqa: E402

from hivemind.boinc.real_client.moe_work import load_custom_shards  # noqa: E402

SECRET = b"a-long-random-test-secret-123456"


def fake_tok(text: str) -> list[int]:
    """One token per whitespace word; ids come from the word hash so equal words give equal ids."""
    return [int.from_bytes(w.encode()[:3].ljust(3, b"_"), "big") % 50_000 for w in text.split()]


def _docs(n: int = 200) -> list[str]:
    return [f"document number {i} " + " ".join(f"w{i}x{j}" for j in range(30)) for i in range(n)]


def test_split_is_disjoint_deterministic_and_order_independent() -> None:
    docs = _docs() + _docs(20)  # 20 exact duplicates
    train, held = mcs.split_docs(docs, 0.2, seed=7)
    assert set(train).isdisjoint(held) and len(train) + len(held) == 200  # duplicates collapse, nobody is in both
    assert 20 <= len(held) <= 60
    assert mcs.split_docs(list(reversed(docs)), 0.2, seed=7) == (train, held)
    assert mcs.split_docs(docs, 0.2, seed=8) != (train, held)  # the seed changes the split


def test_build_writes_verifiable_shards(tmp_path: Path) -> None:
    out = tmp_path / "shards"
    m = mcs.build(_docs(), fake_tok, out, domain="mydata", seq_len=32, heldout_frac=0.2, seed=7, secret=SECRET, tokenizer_name="fake", secret_is_default=False)
    assert m["chunks"]["train"] > 10 and m["chunks"]["heldout"] >= 1 and m["domain"] == "mydata"
    train, held, manifest = load_custom_shards(out, SECRET)
    assert all(len(c) == 32 for c in train + held) and manifest["signed_with_default_public_secret"] is False
    flat = lambda chunks: {t for c in chunks for t in c}  # noqa: E731
    # the document split is disjoint, so training and held-out chunks are cut from different documents
    assert len(train) == m["chunks"]["train"] and len(held) == m["chunks"]["heldout"]


def test_a_flipped_byte_or_wrong_secret_is_rejected(tmp_path: Path) -> None:
    out = tmp_path / "shards"
    mcs.build(_docs(), fake_tok, out, domain="d", seq_len=32, heldout_frac=0.2, seed=7, secret=SECRET, tokenizer_name="fake", secret_is_default=False)
    with pytest.raises(ValueError, match="HMAC"):
        load_custom_shards(out, b"another-secret-entirely-xyz")
    raw = bytearray((out / "train.json").read_bytes())
    raw[5] = ord("9") if raw[5] != ord("9") else ord("8")
    (out / "train.json").write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="sha256"):
        load_custom_shards(out, SECRET)


def test_building_cannot_open_a_network_connection(tmp_path: Path) -> None:
    with mcs.no_network():
        with pytest.raises(RuntimeError, match="network access is blocked"):
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("127.0.0.1", 9))
        mcs.build(_docs(), fake_tok, tmp_path / "s", domain="d", seq_len=32, heldout_frac=0.2, seed=7, secret=SECRET, tokenizer_name="fake", secret_is_default=False)
    assert socket.socket.__name__ == "socket"  # the real class is restored afterwards


def test_too_little_text_is_an_error_not_an_empty_shard(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="too little text"):
        mcs.build(["just a few words here"] * 3, fake_tok, tmp_path / "s", domain="d", seq_len=32, heldout_frac=0.2, seed=7, secret=SECRET, tokenizer_name="fake", secret_is_default=False)


def test_a_secret_is_required_unless_explicitly_waived(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIVEMIND_SHARD_SECRET", raising=False)
    with pytest.raises(ValueError, match="HIVEMIND_SHARD_SECRET"):
        mcs.secret_from_env(False)
    assert mcs.secret_from_env(True)  # the public default, only on request
    monkeypatch.setenv("HIVEMIND_SHARD_SECRET", "short")
    with pytest.raises(ValueError, match="16"):
        mcs.secret_from_env(False)


def test_jsonl_reader_skips_blank_rows_and_reports_bad_lines(tmp_path: Path) -> None:
    p = tmp_path / "d.jsonl"
    p.write_text('{"text": "hello world"}\n\n{"other": 1}\n{"text": "  "}\n{"text": "second doc"}\n', encoding="utf-8")
    assert mcs.read_jsonl(p, "text") == ["hello world", "second doc"]
    p.write_text('{"text": "ok"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        mcs.read_jsonl(p, "text")


def test_end_to_end_with_the_real_granite_tokenizer_if_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("transformers")
    from hivemind.boinc.real_client import moe_work as mw

    hub = Path.home() / ".cache/huggingface"
    cached = list((Path("/mnt/disks/disk-socrateai-local-1/.cache/huggingface/hub")).glob("models--ibm-granite--granite-3.1-1b-a400m-base/snapshots/*/tokenizer.json")) \
        + list((hub / "hub").glob("models--ibm-granite--granite-3.1-1b-a400m-base/snapshots/*/tokenizer.json"))
    if not cached:
        pytest.skip("granite tokenizer not in the local cache")
    snap = str(cached[0].parent)
    jsonl = tmp_path / "mine.jsonl"
    jsonl.write_text("\n".join(json.dumps({"text": f"Recipe {i}: mix flour, water and salt, knead for ten minutes, then rest the dough. " * 6}) for i in range(60)), encoding="utf-8")
    monkeypatch.setenv("HIVEMIND_SHARD_SECRET", SECRET.decode())
    assert mcs.main(["--jsonl", str(jsonl), "--domain", "baking", "--tokenizer", snap, "--out", str(tmp_path / "o"), "--seq-len", "64"]) == 0
    train, held, manifest = mw.load_custom_shards(tmp_path / "o", SECRET)
    assert manifest["domain"] == "baking" and train and held and all(len(c) == 64 for c in train)
