# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Hardening and negative-control test suite for node_federation_plan, workunit, litestream_lock, and remote_probe."""

from __future__ import annotations

from pathlib import Path

import pytest

from hivemind.boinc import node_federation_plan, workunit
from hivemind.router import litestream_lock, remote_probe


def test_node_federation_plan_validation() -> None:
    assert node_federation_plan.classify_plan_class(None) == "cpu_hivemind_solver"
    assert node_federation_plan.classify_plan_class(8192.0) == "cuda_hivemind"
    assert node_federation_plan.classify_plan_class(16384.0) == "cuda_hivemind_big"

    with pytest.raises(ValueError, match="total_mib must be None or a finite non-negative number"):
        node_federation_plan.classify_plan_class(-1.0)
    with pytest.raises(ValueError, match="total_mib must be None or a finite non-negative number"):
        node_federation_plan.classify_plan_class(float("nan"))
    with pytest.raises(ValueError, match="total_mib must be None or a finite non-negative number"):
        node_federation_plan.classify_plan_class(True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="usable_mib must be a non-negative integer"):
        node_federation_plan.pick_bee(-1)
    with pytest.raises(ValueError, match="usable_mib must be a non-negative integer"):
        node_federation_plan.pick_bee(True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="coordinator must be a non-empty http/https URL"):
        node_federation_plan.build_plan("RTX 2070", 8192.0, coordinator="ftp://bad", handle="node1")
    with pytest.raises(ValueError, match="handle must be a non-empty string"):
        node_federation_plan.build_plan("RTX 2070", 8192.0, coordinator="https://example.com", handle="  ")
    with pytest.raises(ValueError, match="overlay_ip must be None or a non-empty string"):
        node_federation_plan.build_plan("RTX 2070", 8192.0, coordinator="https://example.com", handle="node1", overlay_ip=" ")


def test_workunit_dataclass_and_parquet_shard_validation(tmp_path: Path) -> None:
    wu = workunit.BOINCWorkunit(
        wu_id="wu_1",
        branch_name="math",
        outer_step=0,
        base_model_id="sha256:abc",
        lora_checkpoint_url="https://example.com/lora.pt",
        data_shard_parquet="shard.parquet",
        shard_sha256="a" * 64,
        shard_hmac_signature="b" * 64,
        rng_seed=42,
    )
    assert wu.to_dict()["wu_id"] == "wu_1"

    with pytest.raises(ValueError, match="wu_id must be a non-empty string"):
        workunit.BOINCWorkunit("", "math", 0, "m", "u", "p", "s", "h", 42)
    with pytest.raises(ValueError, match="outer_step must be a non-negative integer"):
        workunit.BOINCWorkunit("wu", "math", -1, "m", "u", "p", "s", "h", 42)
    with pytest.raises(ValueError, match="rng_seed must be an integer"):
        workunit.BOINCWorkunit("wu", "math", 0, "m", "u", "p", "s", "h", True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="inner_steps_H must be a positive integer"):
        workunit.BOINCWorkunit("wu", "math", 0, "m", "u", "p", "s", "h", 42, inner_steps_H=0)

    with pytest.raises(ValueError, match="parquet_bytes must be bytes"):
        workunit.sign_shard_bytes("not-bytes", 7)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rng_seed must be an integer"):
        workunit.sign_shard_bytes(b"abc", False)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="branch_name must be a non-empty string"):
        workunit.create_signed_parquet_shard(tmp_path / "s.parquet", "  ", 7)
    with pytest.raises(ValueError, match="num_sequences must be a positive integer"):
        workunit.create_signed_parquet_shard(tmp_path / "s.parquet", "math", 7, num_sequences=0)
    with pytest.raises(ValueError, match="seq_len must be a positive integer"):
        workunit.create_signed_parquet_shard(tmp_path / "s.parquet", "math", 7, seq_len=0)
    with pytest.raises(ValueError, match="vocab_size must be an integer >= 33"):
        workunit.create_signed_parquet_shard(tmp_path / "s.parquet", "math", 7, vocab_size=32)


def test_litestream_lock_store_and_writer_lock_validation(tmp_path: Path) -> None:
    mem = litestream_lock.InMemoryGenerationStore()
    fstore = litestream_lock.FileGenerationStore(tmp_path / "locks")

    for store in (mem, fstore):
        with pytest.raises(ValueError, match="key must be a non-empty string"):
            store.put_if_generation_match("  ", {}, if_generation_match=0)
        with pytest.raises(ValueError, match="payload must be a dict"):
            store.put_if_generation_match("k", "bad", if_generation_match=0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="if_generation_match must be a non-negative integer"):
            store.put_if_generation_match("k", {}, if_generation_match=-1)
        with pytest.raises(ValueError, match="if_generation_match must be a non-negative integer"):
            store.delete_if_generation_match("k", if_generation_match=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="owner_id must be a non-empty string"):
        litestream_lock.LitestreamWriterLock("gs://bucket/db", "   ", mem)
    with pytest.raises(ValueError, match="lease_ttl_s must be a finite positive number"):
        litestream_lock.LitestreamWriterLock("gs://bucket/db", "owner-1", mem, lease_ttl_s=0.0)
    with pytest.raises(ValueError, match="lease_ttl_s must be a finite positive number"):
        litestream_lock.LitestreamWriterLock("gs://bucket/db", "owner-1", mem, lease_ttl_s=float("nan"))

    lock = litestream_lock.LitestreamWriterLock("gs://bucket/db", "owner-1", mem)
    with pytest.raises(ValueError, match="now_utc must be a finite number"):
        lock.acquire(now_utc=float("inf"))
    assert lock.acquire(now_utc=1000.0) is True
    assert lock.release() is True


def test_remote_probe_validation() -> None:
    with pytest.raises(ValueError, match="url must be a non-empty http/https URL"):
        remote_probe.probe_remote_status("   ")
    with pytest.raises(ValueError, match="url must be a non-empty http/https URL"):
        remote_probe.probe_remote_status("file:///etc/passwd")
    with pytest.raises(ValueError, match="timeout_s must be a finite positive number"):
        remote_probe.probe_remote_status("https://example.com", timeout_s=0.0)
    with pytest.raises(ValueError, match="timeout_s must be a finite positive number"):
        remote_probe.probe_remote_status("https://example.com", timeout_s=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="now_utc must be a finite number"):
        remote_probe.probe_remote_status("https://example.com", now_utc=float("nan"))
