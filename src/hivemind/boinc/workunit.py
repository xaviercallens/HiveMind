# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""BOINC Workunit (WU) Specification & Cryptographically Signed Parquet Shards.

Encapsulates:
- base_model_id: SHA-256 hash of the immutable 4-bit base model
- lora_checkpoint_url: Current outer-step adapter weights theta^(t-1)
- data_shard_parquet: Pre-tokenized, cryptographically signed training sequences
- rng_seed: Fixed deterministic seed for dropout and data shuffling
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

DEFAULT_COORDINATOR_SECRET = b"ProjectHiveMind-BOINC-Coordinator-Integrity-Tag-v1"


@dataclass
class BOINCWorkunit:
    """Modified BOINC Workunit descriptor distributed to Steam worker nodes."""

    wu_id: str
    branch_name: str
    outer_step: int
    base_model_id: str
    lora_checkpoint_url: str
    data_shard_parquet: str
    shard_sha256: str
    shard_hmac_signature: str
    rng_seed: int
    inner_steps_H: int = 500
    spot_check_interval_s: int = 50
    seq_len: int = 2048
    num_sequences: int = 2048

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sign_shard_bytes(
    parquet_bytes: bytes,
    rng_seed: int,
    secret_key: bytes = DEFAULT_COORDINATOR_SECRET,
) -> tuple[str, str]:
    """Compute SHA-256 digest and HMAC-SHA256 cryptographic signature over a Parquet shard."""
    if not secret_key:
        raise ValueError("secret_key must be non-empty bytes for BOINC shard HMAC signing")
    shard_sha256 = hashlib.sha256(parquet_bytes).hexdigest()
    msg = parquet_bytes + f":seed={rng_seed}".encode("utf-8")
    signature = hmac.new(secret_key, msg, hashlib.sha256).hexdigest()
    return shard_sha256, signature


def create_signed_parquet_shard(
    shard_path: Path | str,
    branch_name: str,
    rng_seed: int,
    num_sequences: int = 64,
    seq_len: int = 128,
    vocab_size: int = 4096,
    secret_key: bytes = DEFAULT_COORDINATOR_SECRET,
) -> tuple[Path, str, str]:
    """Generate a deterministic, pre-tokenized domain training shard in Parquet format and sign it."""
    path = Path(shard_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create domain-specific deterministic token distributions so each branch learns distinct patterns
    domain_offset = int(hashlib.sha256(branch_name.encode("utf-8")).hexdigest()[:8], 16) % 997
    rng = np.random.default_rng(seed=rng_seed + domain_offset)

    # Structured motif tokens per domain so training measurably reduces cross-entropy loss
    base_motif = rng.integers(low=16, high=vocab_size - 16, size=(8,), dtype=np.int32)
    sequences = np.zeros((num_sequences, seq_len), dtype=np.int32)
    for i in range(num_sequences):
        tiled = np.resize(np.roll(base_motif, i % 8), (seq_len,))
        noise_mask = rng.random(seq_len) < 0.08
        noise_tokens = rng.integers(low=1, high=vocab_size, size=(seq_len,), dtype=np.int32)
        sequences[i] = np.where(noise_mask, noise_tokens, tiled)

    table = pa.table(
        {
            "seq_id": pa.array(np.arange(num_sequences, dtype=np.int32)),
            "branch": pa.array([branch_name] * num_sequences),
            "input_ids": pa.array(sequences.tolist(), type=pa.list_(pa.int32())),
        }
    )
    pq.write_table(table, path, compression="zstd")

    raw_bytes = path.read_bytes()
    shard_sha256, signature = sign_shard_bytes(raw_bytes, rng_seed=rng_seed, secret_key=secret_key)
    return path, shard_sha256, signature


def verify_and_load_parquet_shard(
    shard_path: Path | str,
    expected_sha256: str,
    expected_hmac_signature: str,
    rng_seed: int,
    secret_key: bytes = DEFAULT_COORDINATOR_SECRET,
) -> torch.Tensor:
    """Verify cryptographic signature of a BOINC Parquet shard and load token sequences as a Tensor."""
    path = Path(shard_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet data shard not found: {path}")

    raw_bytes = path.read_bytes()
    actual_sha256, actual_sig = sign_shard_bytes(
        raw_bytes, rng_seed=rng_seed, secret_key=secret_key
    )
    if not hmac.compare_digest(actual_sha256, expected_sha256):
        raise ValueError(
            f"Parquet shard SHA-256 mismatch! Expected {expected_sha256[:12]}..., got {actual_sha256[:12]}..."
        )
    if not hmac.compare_digest(actual_sig, expected_hmac_signature):
        raise ValueError("Parquet shard cryptographic HMAC signature verification failed!")

    table = pq.read_table(path)
    token_lists = table["input_ids"].to_pylist()
    return torch.tensor(token_lists, dtype=torch.long)
