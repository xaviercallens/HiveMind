#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""MVP criterion B1: turn YOUR dataset (JSONL) into signed training shards plus a held-out split, entirely on your machine.

    export HIVEMIND_SHARD_SECRET=<a long random secret only the coordinator and validators know>
    python scripts/make_custom_shards.py --jsonl my.jsonl --field text --tokenizer ibm-granite/granite-3.1-1b-a400m-base --domain mydata --out shards/mydata

Output directory: ``train.json`` and ``heldout.json`` (lists of token-id chunks of ``--seq-len``), and ``manifest.json`` (counts, sha256 of each file, the HMAC signature of the
training shard, tokenizer, seed). Documents are split by the hash of their text, so a document is in exactly one split and exact duplicates never straddle them.
Nothing is downloaded: the tokenizer is loaded from the local cache (HF_HUB_OFFLINE=1 is forced) and the build is wrapped so that opening a network socket fails.
Use the shards with ``build_train_spec(..., custom_dir=...)`` (src/hivemind/boinc/real_client/moe_work.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hivemind.boinc.workunit import DEFAULT_COORDINATOR_SECRET, sign_shard_bytes  # noqa: E402


def read_jsonl(path: Path, field: str) -> list[str]:
    docs: list[str] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{n}: not valid JSON ({exc})") from None
        value = row.get(field) if isinstance(row, dict) else None
        if isinstance(value, str) and value.strip():
            docs.append(value.strip())
    return docs


def split_docs(docs: list[str], heldout_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Deterministic split by document hash: a document is in exactly one side, exact duplicates collapse, order does not matter."""
    if not 0.0 < heldout_frac < 1.0:
        raise ValueError("heldout_frac must be between 0 and 1")
    train: dict[str, str] = {}
    held: dict[str, str] = {}
    for d in docs:
        h = hashlib.sha256(f"{seed}:{d}".encode()).hexdigest()
        (held if int(h[:8], 16) / 0xFFFFFFFF < heldout_frac else train)[h] = d
    return [train[k] for k in sorted(train)], [held[k] for k in sorted(held)]


def to_chunks(tok_fn: Callable[[str], list[int]], docs: list[str], seq_len: int) -> list[list[int]]:
    """Concatenate documents (blank-line separated, like the built-in domains) and cut whole ``seq_len`` chunks; the ragged tail is dropped."""
    ids: list[int] = []
    for d in docs:
        ids += tok_fn(d + "\n\n")
    return [ids[i : i + seq_len] for i in range(0, len(ids) - seq_len + 1, seq_len)]


@contextmanager
def no_network() -> Iterator[None]:
    """Any attempt to open a socket while building raises: the guarantee that no data leaves the machine is enforced, not just promised."""
    real = socket.socket

    class _Blocked(real):  # type: ignore[misc, valid-type]
        def connect(self, *a: Any, **k: Any) -> None:
            raise RuntimeError("network access is blocked while building custom shards")

        connect_ex = connect  # type: ignore[assignment]

    socket.socket = _Blocked  # type: ignore[misc]
    try:
        yield
    finally:
        socket.socket = real  # type: ignore[misc]


def secret_from_env(allow_default: bool) -> bytes:
    raw = os.environ.get("HIVEMIND_SHARD_SECRET", "")
    if raw:
        if len(raw) < 16:
            raise ValueError("HIVEMIND_SHARD_SECRET must be at least 16 characters")
        return raw.encode()
    if allow_default:
        return DEFAULT_COORDINATOR_SECRET  # a public constant: the signature then proves integrity against accidents only, not against an attacker
    raise ValueError("set HIVEMIND_SHARD_SECRET (or pass --insecure-default-secret for a local test)")


def build(docs: list[str], tok_fn: Callable[[str], list[int]], out: Path, *, domain: str, seq_len: int, heldout_frac: float, seed: int, secret: bytes,
          tokenizer_name: str, secret_is_default: bool) -> dict[str, Any]:
    train_docs, held_docs = split_docs(docs, heldout_frac, seed)
    train, held = to_chunks(tok_fn, train_docs, seq_len), to_chunks(tok_fn, held_docs, seq_len)
    if not train or not held:
        raise ValueError(f"too little text: {len(train)} train chunks and {len(held)} held-out chunks of {seq_len} tokens (need at least one of each)")
    out.mkdir(parents=True, exist_ok=True)
    train_bytes = json.dumps(train, separators=(",", ":")).encode()  # the exact bytes build_train_spec signs
    held_bytes = json.dumps(held, separators=(",", ":")).encode()
    (out / "train.json").write_bytes(train_bytes)
    (out / "heldout.json").write_bytes(held_bytes)
    train_sha, train_sig = sign_shard_bytes(train_bytes, seed, secret)
    manifest = {
        "schema": "hivemind.custom_shards.v1", "domain": domain, "tokenizer": tokenizer_name, "seq_len": seq_len, "seed": seed,
        "documents": {"total": len(docs), "train": len(train_docs), "heldout": len(held_docs)},
        "chunks": {"train": len(train), "heldout": len(held)},
        "train_sha256": train_sha, "train_hmac_signature": train_sig, "heldout_sha256": hashlib.sha256(held_bytes).hexdigest(),
        "signed_with_default_public_secret": secret_is_default,
        "offline": True,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--field", default="text")
    ap.add_argument("--tokenizer", default="ibm-granite/granite-3.1-1b-a400m-base")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--heldout-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--insecure-default-secret", action="store_true")
    args = ap.parse_args(argv)

    os.environ["HF_HUB_OFFLINE"] = "1"
    secret = secret_from_env(args.insecure_default_secret)
    docs = read_jsonl(args.jsonl, args.field)
    with no_network():
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        manifest = build(docs, lambda t: tok(t).input_ids, args.out, domain=args.domain, seq_len=args.seq_len, heldout_frac=args.heldout_frac,
                         seed=args.seed, secret=secret, tokenizer_name=args.tokenizer, secret_is_default=not os.environ.get("HIVEMIND_SHARD_SECRET"))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
