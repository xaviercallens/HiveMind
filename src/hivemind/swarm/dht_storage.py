# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Phase 2.0: Distributed Disk Vault (Kademlia DHT + Content-Addressed Storage) & Proof-of-Storage.

Stores sharded Hugging Face datasets and converged LoRA domain adapters on local NVMe/SSD storage
(`D:\\xdev\\DecentralizedAI\\.cache\\swarm_store`), tracks R=3 peer replication pins, and answers
block-hash-seeded Merkle Byte-Range Proof-of-Storage challenges.

Season 5 / Wave 4 (`HM-S5-01`) extends the vault with **resumable 64 KB Content-Addressed Chunk
Seeding**: exported community adapter bundles are split into 64 KB SHA-256 chunks, advertised
through a Merkle-rooted seed manifest, replicated across peer seeders and streamed (with resume
support) by downloading peers, with corrupted chunks rejected before reassembly.

Phase S-2 Runtime Hardening integrates `SwarmTrainingAndStoreGuard` directly into CAS storage,
checkpoint deserialization (`weights_only=True` + cryptographic CID digest verification),
path-traversal confinement, and dataset shard anti-poisoning checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from hivemind.security.swarm_runtime_hardening import (
    MAX_BUNDLE_BYTES,
    MAX_CHUNK_BYTES,
    SwarmTrainingAndStoreGuard,
)
from hivemind.swarm.ledger import StorageProofReceipt

CHUNK_SIZE_BYTES = 64 * 1024
"""Default 64 KB content-addressed chunk granularity used for multi-peer bundle seeding."""

PARTIAL_DIR_NAME = ".hivemind_cas_partial"
"""Staging folder holding partially downloaded chunks so transfers can resume."""


class ChunkIntegrityError(RuntimeError):
    """Raised when a chunked CAS transfer fails SHA-256 or Merkle-root verification."""


@dataclass
class ChunkedFileEntry:
    """Deterministic 64 KB chunking plan of a single file inside a seeded bundle."""

    relative_path: str
    size: int
    sha256: str
    chunk_size: int
    num_chunks: int
    chunk_hashes: List[str]
    merkle_root: str
    chunk_seeders: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.relative_path, str) or not self.relative_path.strip():
            raise ValueError("relative_path must be a non-empty string")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ValueError("size must be a non-negative integer")
        if isinstance(self.chunk_size, bool) or not isinstance(self.chunk_size, int) or self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if isinstance(self.num_chunks, bool) or not isinstance(self.num_chunks, int) or self.num_chunks < 0:
            raise ValueError("num_chunks must be a non-negative integer")
        if not isinstance(self.sha256, str) or not self.sha256.strip():
            raise ValueError("sha256 must be a non-empty string")
        if not isinstance(self.merkle_root, str) or not self.merkle_root.strip():
            raise ValueError("merkle_root must be a non-empty string")
        if not isinstance(self.chunk_hashes, list) or len(self.chunk_hashes) != self.num_chunks:
            raise ValueError("chunk_hashes must be a list of length num_chunks")
        if not isinstance(self.chunk_seeders, list):
            raise TypeError("chunk_seeders must be a list")


@dataclass
class BundleSeedManifest:
    """Merkle-rooted advertisement of a bundle's content-addressed chunks.

    The manifest is the contract shared between the originating node and every peer
    seeder: it pins the exact chunk digests, the per-file and bundle-wide Merkle roots,
    and the peer node id hosting each individual chunk.
    """

    bundle_id: str
    source_node_id: str
    created_at_epoch: float
    chunk_size: int
    total_bytes: int
    num_chunks: int
    bundle_merkle_root: str
    files: List[ChunkedFileEntry]
    seeders: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.bundle_id, str) or not self.bundle_id.strip():
            raise ValueError("bundle_id must be a non-empty string")
        if not isinstance(self.source_node_id, str) or not self.source_node_id.strip():
            raise ValueError("source_node_id must be a non-empty string")
        if (
            isinstance(self.created_at_epoch, bool)
            or not isinstance(self.created_at_epoch, (int, float))
            or not math.isfinite(float(self.created_at_epoch))
        ):
            raise ValueError("created_at_epoch must be a finite number")
        if isinstance(self.chunk_size, bool) or not isinstance(self.chunk_size, int) or self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if isinstance(self.total_bytes, bool) or not isinstance(self.total_bytes, int) or self.total_bytes < 0:
            raise ValueError("total_bytes must be a non-negative integer")
        if isinstance(self.num_chunks, bool) or not isinstance(self.num_chunks, int) or self.num_chunks < 0:
            raise ValueError("num_chunks must be a non-negative integer")
        if not isinstance(self.bundle_merkle_root, str) or not self.bundle_merkle_root.strip():
            raise ValueError("bundle_merkle_root must be a non-empty string")
        if not isinstance(self.files, list) or any(not isinstance(f, ChunkedFileEntry) for f in self.files):
            raise TypeError("files must be a list of ChunkedFileEntry instances")
        if not isinstance(self.seeders, list):
            raise TypeError("seeders must be a list")


def _lease_clock(now_s: Optional[float]) -> float:
    """The lease clock: wall time, or a finite caller-supplied time. A NaN `now_s` makes every `now > expires_at_s`
    comparison False, so a lease granted at 0.0 would commit (and never be reclaimed) at any later time."""
    if now_s is None:
        return time.time()
    if isinstance(now_s, bool) or not isinstance(now_s, (int, float)):
        raise ValueError("now_s must be finite")
    now = float(now_s)
    if not math.isfinite(now):
        raise ValueError("now_s must be finite")
    return now


def compute_lease_ttl_s(alpha: float, t_step_s: float, rtt_s: float) -> float:
    """HM-S9-06: Compute heartbeat data-lease TTL `tau_lease = alpha * t_step_s + rtt_s`."""
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or alpha <= 0.0 or not (alpha < float("inf")):
        raise ValueError("alpha must be > 0 and finite")
    if isinstance(t_step_s, bool) or not isinstance(t_step_s, (int, float)) or t_step_s <= 0.0 or not (t_step_s < float("inf")):
        raise ValueError("t_step_s must be > 0 and finite")
    if isinstance(rtt_s, bool) or not isinstance(rtt_s, (int, float)) or rtt_s < 0.0 or not (rtt_s < float("inf")):
        raise ValueError("rtt_s must be >= 0 and finite")
    return float(alpha) * float(t_step_s) + float(rtt_s)


@dataclass
class DataShardLease:
    """HM-S9-06: Heartbeat-backed dataset shard lease preventing duplicate commits or dropped shards."""

    lease_id: str
    shard_cid: str
    worker_id: str
    granted_at_s: float
    ttl_s: float
    expires_at_s: float
    status: str = "active"  # "active" | "committed" | "reclaimed"
    committed_at_s: Optional[float] = None
    reclaimed_at_s: Optional[float] = None
    heartbeat_count: int = 0

    def __post_init__(self) -> None:
        for str_name in ("lease_id", "shard_cid", "worker_id"):
            sval = getattr(self, str_name)
            if not isinstance(sval, str) or not sval.strip():
                raise ValueError(f"{str_name} must be a non-empty string")
        for f_name in ("granted_at_s", "expires_at_s"):
            fval = getattr(self, f_name)
            if isinstance(fval, bool) or not isinstance(fval, (int, float)) or not math.isfinite(float(fval)):
                raise ValueError(f"{f_name} must be a finite number")
        if isinstance(self.ttl_s, bool) or not isinstance(self.ttl_s, (int, float)) or not math.isfinite(float(self.ttl_s)) or float(self.ttl_s) <= 0.0:
            raise ValueError("ttl_s must be a positive finite number")
        if self.status not in ("active", "committed", "reclaimed"):
            raise ValueError("status must be 'active', 'committed', or 'reclaimed'")
        if isinstance(self.heartbeat_count, bool) or not isinstance(self.heartbeat_count, int) or self.heartbeat_count < 0:
            raise ValueError("heartbeat_count must be a non-negative integer")


class ContentAddressedSwarmStore:
    """Local Agent Disk Vault & Kademlia-style Content-Addressed Storage (CAS).

    Stores sharded Hugging Face datasets and converged LoRA domain adapters on disk,
    generates deterministic Merkle Proof-of-Storage receipts, and (Season 5) seeds and
    streams exported community adapter bundles as resumable 64 KB content-addressed chunks.
    """

    SHARD_BLOCK_ALIGNMENT_BYTES = 16_384
    CHUNK_SIZE_BYTES = CHUNK_SIZE_BYTES

    def __init__(
        self,
        root_dir: Path,
        replication_factor: int = 3,
        node_id: Optional[str] = None,
        security_guard: Optional[SwarmTrainingAndStoreGuard] = None,
    ) -> None:
        if isinstance(replication_factor, bool) or not isinstance(replication_factor, int) or replication_factor < 1:
            raise ValueError("replication_factor must be an integer >= 1")
        if security_guard is not None and not isinstance(security_guard, SwarmTrainingAndStoreGuard):
            raise TypeError("security_guard must be a SwarmTrainingAndStoreGuard or None")
        self.root_dir = Path(root_dir)
        self.replication_factor = replication_factor
        self.shards_dir = self.root_dir / "dataset_shards"
        self.experts_dir = self.root_dir / "lora_experts"
        self.chunks_dir = self.root_dir / "cas_chunks"
        self.bundle_seeds_dir = self.root_dir / "bundle_seeds"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.experts_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.bundle_seeds_dir.mkdir(parents=True, exist_ok=True)
        self.node_pins: Dict[str, List[str]] = {}
        self.cid_to_path: Dict[str, Path] = {}
        self.node_id = str(node_id) if node_id else f"cas-peer-{self.root_dir.name}"
        self._guard = security_guard or SwarmTrainingAndStoreGuard()
        # HM-S9-06: Heartbeat-backed data shard leasing state
        self._shard_leases: Dict[str, DataShardLease] = {}
        self._active_shard_lease_by_cid: Dict[str, str] = {}
        self._committed_shards: Dict[str, str] = {}
        self._reclaim_queue: List[str] = []
        self._lease_seq: int = 0

    def store_hf_dataset_shard(
        self,
        node_id: str,
        domain_name: str,
        shard_id: int,
        samples: List[str],
    ) -> Tuple[str, Path, int]:
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError("node_id must be a non-empty string")
        if isinstance(shard_id, bool) or not isinstance(shard_id, int) or shard_id < 0:
            raise ValueError("shard_id must be a non-negative integer")
        # Phase S-2 Hardening: Validate dataset shard against poisoning, payment spam, and low-entropy farming
        self._guard.validate_dataset_shard(
            domain_name=domain_name,
            samples=samples,
            raise_on_violation=True,
        )

        payload = {
            "domain": domain_name,
            "shard_id": shard_id,
            "num_samples": len(samples),
            "samples": samples,
        }
        raw_json = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        # Align shard container to a fixed 16 KB CAS block boundary for uniform quota accounting
        rem = len(raw_json) % self.SHARD_BLOCK_ALIGNMENT_BYTES
        if rem != 0:
            pad_len = self.SHARD_BLOCK_ALIGNMENT_BYTES - rem
            raw_bytes = raw_json + (b" " * pad_len)
        else:
            raw_bytes = raw_json

        digest = hashlib.sha256(raw_bytes).hexdigest()
        cid = f"cid://sha256/{digest[:32]}"

        safe_domain = self._safe_bundle_id(domain_name)
        _, shard_path = self._guard.validate_cas_relative_path(
            self.shards_dir,
            f"{safe_domain}_shard_{int(shard_id)}_{digest[:12]}.json",
            raise_on_violation=True,
        )
        shard_path.write_bytes(raw_bytes)

        self.node_pins.setdefault(node_id, []).append(cid)
        self.cid_to_path[cid] = shard_path
        return cid, shard_path, len(raw_bytes)

    def verify_proof_of_storage(
        self,
        node_id: str,
        shard_cid: str,
        shard_path: Path,
        prev_block_hash: str,
        chunk_size: int = 256,
        expected_sha256: Optional[str] = None,
    ) -> StorageProofReceipt:
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError("node_id must be a non-empty string")
        if not isinstance(shard_cid, str) or not shard_cid.strip():
            raise ValueError("shard_cid must be a non-empty string")
        if not isinstance(prev_block_hash, str) or not prev_block_hash.strip():
            raise ValueError("prev_block_hash must be a non-empty string")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        p = Path(shard_path)
        if not p.is_file():
            return StorageProofReceipt(
                node_id=node_id,
                shard_cid=shard_cid,
                challenged_chunk_index=0,
                merkle_proof_hash="",
                bytes_stored=0,
                verified=False,
            )
        raw_bytes = p.read_bytes()
        actual_digest = hashlib.sha256(raw_bytes).hexdigest()
        cid_hex = shard_cid.removeprefix("cid://sha256/") if str(shard_cid).startswith("cid://sha256/") else ""
        cid_matches = bool(len(cid_hex) >= 16 and actual_digest.startswith(cid_hex))
        sha_matches = True if expected_sha256 is None else (actual_digest == expected_sha256)
        pinned = (shard_cid in self.node_pins.get(node_id, [])) or (
            expected_sha256 is not None and sha_matches
        )
        is_verified = bool(pinned and cid_matches and sha_matches)
        chunks = [
            raw_bytes[i : i + chunk_size]
            for i in range(0, max(1, len(raw_bytes)), chunk_size)
        ]
        challenge_seed = hashlib.sha256(
            f"{prev_block_hash}:{node_id}:{shard_cid}".encode("utf-8")
        ).hexdigest()
        challenged_idx = int(challenge_seed, 16) % len(chunks)

        challenged_chunk = chunks[challenged_idx]
        merkle_proof = hashlib.sha256(
            challenged_chunk + prev_block_hash.encode("utf-8")
        ).hexdigest()

        return StorageProofReceipt(
            node_id=node_id,
            shard_cid=shard_cid,
            challenged_chunk_index=challenged_idx,
            merkle_proof_hash=merkle_proof,
            bytes_stored=len(raw_bytes) if is_verified else 0,
            verified=is_verified,
        )

    def publish_lora_expert_checkpoint(
        self,
        domain_name: str,
        lora_state_dict: Dict[str, torch.Tensor],
        metadata: Dict[str, object],
    ) -> Tuple[str, Path]:
        # Phase S-2 Hardening: Validate LoRA tensors for NaN/Inf bombs and Frobenius norm ceilings
        self._guard.validate_lora_tensors(
            lora_state_dict=lora_state_dict,
            domain_name=domain_name,
            raise_on_violation=True,
        )

        serializable_meta = json.dumps(metadata, sort_keys=True).encode("utf-8")
        hasher = hashlib.sha256(serializable_meta)
        cpu_state: Dict[str, torch.Tensor] = {}
        for k in sorted(lora_state_dict.keys()):
            t = lora_state_dict[k].detach().cpu().to(torch.float16).contiguous()
            cpu_state[k] = t
            hasher.update(t.numpy().tobytes())

        digest = hasher.hexdigest()
        safe_dom = self._safe_bundle_id(domain_name.lower())
        expert_cid = f"cid://sha256/expert-{safe_dom}-{digest[:24]}"
        _, out_path = self._guard.validate_cas_relative_path(
            self.experts_dir,
            f"expert_{safe_dom}_{digest[:12]}.pt",
            raise_on_violation=True,
        )
        torch.save({"cid": expert_cid, "metadata": metadata, "lora_state": cpu_state}, out_path)
        self.cid_to_path[expert_cid] = out_path
        return expert_cid, out_path

    def load_lora_expert_checkpoint(
        self,
        expert_cid_or_path: str | Path,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[str, Dict[str, object], Dict[str, torch.Tensor]]:
        """Retrieve a published LoRA domain expert from the local CAS vault by CID or file path.

        Phase S-2 Hardening: Uses `weights_only=True` exclusively to block arbitrary pickle RCE,
        recomputes the content-addressed digest over `metadata` + `lora_state`, and validates
        all loaded tensors through `SwarmTrainingAndStoreGuard.validate_lora_tensors`.
        """
        expected_cid: Optional[str] = None
        if isinstance(expert_cid_or_path, Path) or not str(expert_cid_or_path).startswith("cid://"):
            ckpt_path = Path(expert_cid_or_path)
        else:
            cid_str = str(expert_cid_or_path)
            expected_cid = cid_str
            if cid_str in self.cid_to_path:
                ckpt_path = self.cid_to_path[cid_str]
            else:
                raise FileNotFoundError(f"Expert CID {cid_str} not found in local CAS index.")

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ChunkIntegrityError("Loaded checkpoint payload is not a dictionary.")

        cid = str(payload.get("cid", ""))
        if expected_cid is not None and cid != expected_cid:
            raise ChunkIntegrityError(
                f"Checkpoint CID mismatch: expected '{expected_cid}', found '{cid}'."
            )

        meta = dict(payload.get("metadata", {}))
        raw_state: Dict[str, torch.Tensor] = payload.get("lora_state", {})

        # Verify cryptographic CID digest against metadata + tensor bytes. A payload whose cid is not an expert cid has
        # nothing to verify against, so it is refused instead of loaded unchecked.
        if not cid.startswith("cid://sha256/expert-"):
            raise ChunkIntegrityError(f"Checkpoint cid '{cid}' is not a content-addressed expert cid.")
        serializable_meta = json.dumps(meta, sort_keys=True).encode("utf-8")
        hasher = hashlib.sha256(serializable_meta)
        for k in sorted(raw_state.keys()):
            t_fp16 = raw_state[k].detach().cpu().to(torch.float16).contiguous()
            hasher.update(t_fp16.numpy().tobytes())
        recomputed_prefix = hasher.hexdigest()[:24]
        if not cid.endswith(f"-{recomputed_prefix}"):
            raise ChunkIntegrityError(
                f"Cryptographic CID digest mismatch for '{cid}' (expected suffix '-{recomputed_prefix}')."
            )

        lora_state = {k: v.to(dtype=dtype) for k, v in raw_state.items()}
        self._guard.validate_lora_tensors(
            lora_state_dict=lora_state,
            domain_name=cid or ckpt_path.stem,
            raise_on_violation=True,
        )
        return cid, meta, lora_state

    # ------------------------------------------------------------------
    # Season 5 (HM-S5-01): Resumable 64 KB Content-Addressed Chunk Seeding
    # ------------------------------------------------------------------

    @staticmethod
    def compute_merkle_root(leaf_hex_digests: Sequence[str]) -> str:
        """Compute the deterministic binary Merkle root over SHA-256 leaf digests.

        Leaves are the raw SHA-256 digests of the chunk (or file) payloads, hex encoded.
        Odd nodes are duplicated on every level, mirroring the `HiveSwarmLedger` Merkle
        convention so both modules can be audited with the same tooling.
        """
        if isinstance(leaf_hex_digests, (str, bytes)) or not isinstance(leaf_hex_digests, Sequence):
            raise ValueError("leaf_hex_digests must be a sequence of hex digest strings, not str/bytes")
        if not leaf_hex_digests:
            return hashlib.sha256(b"hivemind-empty-cas-merkle").hexdigest()
        for digest in leaf_hex_digests:
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("each leaf digest must be a 64-character hex string")
        layer = [bytes.fromhex(str(digest)) for digest in leaf_hex_digests]
        while len(layer) > 1:
            if len(layer) % 2 == 1:
                layer.append(layer[-1])
            layer = [
                hashlib.sha256(layer[i] + layer[i + 1]).digest()
                for i in range(0, len(layer), 2)
            ]
        return layer[0].hex()

    def put_chunk(self, data: bytes) -> str:
        """Store a chunk blob in the local CAS chunk vault and return its SHA-256 digest."""
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("Chunk payload must be a bytes-like object.")
        payload = bytes(data)
        if len(payload) > MAX_CHUNK_BYTES:
            raise ChunkIntegrityError(
                f"Chunk size {len(payload)} bytes exceeds MAX_CHUNK_BYTES ({MAX_CHUNK_BYTES})."
            )
        digest = hashlib.sha256(payload).hexdigest()
        chunk_path = self.chunks_dir / f"{digest}.chunk"
        if not chunk_path.exists():
            chunk_path.write_bytes(payload)
        return digest

    def has_chunk(self, chunk_hash: str) -> bool:
        """Return whether the local chunk vault already holds the given chunk digest."""
        self._guard.validate_sha256_hex(chunk_hash, context="has_chunk", raise_on_violation=True)
        return (self.chunks_dir / f"{chunk_hash}.chunk").exists()

    def read_chunk(self, chunk_hash: str) -> bytes:
        """Read a chunk blob out of the local CAS chunk vault."""
        self._guard.validate_sha256_hex(chunk_hash, context="read_chunk", raise_on_violation=True)
        chunk_path = self.chunks_dir / f"{chunk_hash}.chunk"
        if not chunk_path.exists():
            raise FileNotFoundError(
                f"Chunk {chunk_hash} is not present in the local CAS chunk vault."
            )
        return chunk_path.read_bytes()

    @staticmethod
    def _iter_file_chunks(file_path: Path, chunk_size: int) -> Iterable[bytes]:
        """Stream a file as fixed-size byte blocks without loading it fully into memory."""
        with Path(file_path).open("rb") as handle:
            while True:
                block = handle.read(chunk_size)
                if not block:
                    return
                yield block

    def plan_bundle_chunks(
        self,
        bundle_dir: Path,
        chunk_size: int = CHUNK_SIZE_BYTES,
    ) -> List[ChunkedFileEntry]:
        """Deterministically split every file of `bundle_dir` into `chunk_size` SHA-256 chunks.

        The plan is purely derived from the bundle bytes, so any peer can recompute the very
        same chunk digests and Merkle roots without talking to the originating node.
        """
        bundle_path = Path(bundle_dir)
        if not bundle_path.is_dir():
            raise FileNotFoundError(f"Bundle directory '{bundle_path}' does not exist.")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0 or chunk_size > MAX_CHUNK_BYTES:
            raise ValueError(f"chunk_size must be in (0, {MAX_CHUNK_BYTES}].")

        entries: List[ChunkedFileEntry] = []
        for file_path in sorted(p for p in bundle_path.rglob("*") if p.is_file()):
            relative = file_path.relative_to(bundle_path)
            parts = relative.parts
            if any(part.startswith(".") for part in parts):
                # Skip hidden metadata (e.g. the resumable partial staging folder).
                continue
            self._guard.validate_cas_relative_path(
                bundle_path, relative.as_posix(), raise_on_violation=True
            )
            file_hasher = hashlib.sha256()
            chunk_hashes: List[str] = []
            size = 0
            for block in self._iter_file_chunks(file_path, chunk_size):
                file_hasher.update(block)
                size += len(block)
                chunk_hashes.append(hashlib.sha256(block).hexdigest())
            entries.append(
                ChunkedFileEntry(
                    relative_path=relative.as_posix(),
                    size=size,
                    sha256=file_hasher.hexdigest(),
                    chunk_size=chunk_size,
                    num_chunks=len(chunk_hashes),
                    chunk_hashes=chunk_hashes,
                    merkle_root=self.compute_merkle_root(chunk_hashes),
                )
            )
        return entries

    def seed_bundle_chunks(
        self,
        bundle_dir: Path,
        source_node_id: str,
        bundle_id: Optional[str] = None,
        seeders: Optional[List[str]] = None,
        chunk_size: int = CHUNK_SIZE_BYTES,
    ) -> BundleSeedManifest:
        """Split an exported bundle into 64 KB chunks, store them locally and publish a manifest."""
        if not isinstance(source_node_id, str) or not source_node_id.strip():
            raise ValueError("source_node_id must be a non-empty string")
        if seeders is not None and (
            isinstance(seeders, (str, bytes))
            or not isinstance(seeders, Sequence)
            or any(not isinstance(s, str) or not s.strip() for s in seeders)
        ):
            raise ValueError("seeders must be a sequence of non-empty node ID strings")
        bundle_path = Path(bundle_dir)
        entries = self.plan_bundle_chunks(bundle_path, chunk_size=chunk_size)
        seeder_ids = [str(seeder) for seeder in (seeders or [source_node_id])] or [str(source_node_id)]

        for entry in entries:
            assignments: List[str] = []
            file_path = bundle_path / entry.relative_path
            for index, block in enumerate(self._iter_file_chunks(file_path, entry.chunk_size)):
                digest = hashlib.sha256(block).hexdigest()
                if index >= len(entry.chunk_hashes) or digest != entry.chunk_hashes[index]:
                    raise ChunkIntegrityError(
                        f"Bundle '{bundle_path}' changed while being chunked "
                        f"(chunk {index} of '{entry.relative_path}' digest mismatch)."
                    )
                self.put_chunk(block)
                assignments.append(seeder_ids[index % len(seeder_ids)])
            entry.chunk_seeders = assignments

        manifest = BundleSeedManifest(
            bundle_id=str(bundle_id or bundle_path.name),
            source_node_id=str(source_node_id),
            created_at_epoch=time.time(),
            chunk_size=chunk_size,
            total_bytes=sum(entry.size for entry in entries),
            num_chunks=sum(entry.num_chunks for entry in entries),
            bundle_merkle_root=self.compute_merkle_root([entry.sha256 for entry in entries]),
            files=entries,
            seeders=seeder_ids,
        )
        self.save_seed_manifest(manifest)
        return manifest

    def save_seed_manifest(self, manifest: BundleSeedManifest) -> Path:
        """Persist a bundle seed manifest inside the local CAS vault."""
        self.bundle_seeds_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.bundle_seeds_dir / f"{self._safe_bundle_id(manifest.bundle_id)}.seed_manifest.json"
        manifest_path.write_text(
            json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8"
        )
        return manifest_path

    def load_seed_manifest(self, bundle_id: str) -> BundleSeedManifest:
        """Load a previously advertised bundle seed manifest from the local CAS vault."""
        if not isinstance(bundle_id, str) or not bundle_id.strip():
            raise ValueError("bundle_id must be a non-empty string")
        manifest_path = self.bundle_seeds_dir / f"{self._safe_bundle_id(bundle_id)}.seed_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No seed manifest for bundle '{bundle_id}' in the local CAS vault."
            )
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw["files"] = [ChunkedFileEntry(**entry) for entry in raw.get("files", [])]
        return BundleSeedManifest(**raw)

    def replicate_seed_manifest(
        self,
        manifest: BundleSeedManifest,
        source_store: "ContentAddressedSwarmStore",
        node_id: str,
    ) -> int:
        """Copy the chunks assigned to `node_id` from `source_store` into this node's vault."""
        copied = 0
        for entry in manifest.files:
            for index, chunk_hash in enumerate(entry.chunk_hashes):
                self._guard.validate_sha256_hex(chunk_hash, context="replicate_chunk", raise_on_violation=True)
                assigned = (
                    entry.chunk_seeders[index]
                    if index < len(entry.chunk_seeders)
                    else str(node_id)
                )
                if assigned != str(node_id):
                    continue
                if self.has_chunk(chunk_hash):
                    continue
                payload = source_store.read_chunk(chunk_hash)
                if hashlib.sha256(payload).hexdigest() != chunk_hash:
                    raise ChunkIntegrityError(
                        f"Seeder '{source_store.node_id}' served a corrupted chunk {chunk_hash}."
                    )
                self.put_chunk(payload)
                copied += 1
        self.save_seed_manifest(manifest)
        return copied

    def download_bundle_chunks(
        self,
        manifest: BundleSeedManifest,
        target_dir: Path,
        peer_stores: Optional[List["ContentAddressedSwarmStore"]] = None,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """Stream a seeded bundle from peer seeders with 64 KB chunk verification, path confinement & resume."""
        if int(manifest.total_bytes) > MAX_BUNDLE_BYTES:
            raise ChunkIntegrityError(
                f"Bundle total_bytes ({manifest.total_bytes}) exceeds MAX_BUNDLE_BYTES ({MAX_BUNDLE_BYTES})."
            )

        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        staging_root = target_path / PARTIAL_DIR_NAME
        peers = list(peer_stores or [])

        summary: Dict[str, Any] = {
            "bundle_id": manifest.bundle_id,
            "target_dir": str(target_path),
            "files_written": [],
            "total_bytes": 0,
            "chunks_downloaded": 0,
            "chunks_reused": 0,
            "bytes_downloaded": 0,
            "bytes_reused": 0,
            "corrupt_chunks_rejected": 0,
            "chunk_sources": {},
            "seeders_contacted": [],
            "bundle_merkle_root": manifest.bundle_merkle_root,
            "verified": False,
        }

        for entry in manifest.files:
            # Phase S-2 Hardening: Strictly prevent path traversal outside target_path
            _, file_target = self._guard.validate_cas_relative_path(
                target_path, entry.relative_path, raise_on_violation=True
            )
            file_target.parent.mkdir(parents=True, exist_ok=True)
            if file_target.exists() and hashlib.sha256(file_target.read_bytes()).hexdigest() == entry.sha256:
                summary["files_written"].append(entry.relative_path)
                summary["total_bytes"] += entry.size
                continue

            _, stage_dir = self._guard.validate_cas_relative_path(
                staging_root, entry.relative_path, raise_on_violation=True
            )
            stage_dir.mkdir(parents=True, exist_ok=True)
            assembled = bytearray()
            for index, expected_hash in enumerate(entry.chunk_hashes):
                self._guard.validate_sha256_hex(
                    expected_hash, context=f"{entry.relative_path}#{index}", raise_on_violation=True
                )
                staged_path = stage_dir / f"{index:06d}.chunk"
                chunk_bytes: Optional[bytes] = None
                if resume and staged_path.exists():
                    candidate = staged_path.read_bytes()
                    if hashlib.sha256(candidate).hexdigest() == expected_hash:
                        chunk_bytes = candidate
                        summary["chunks_reused"] += 1
                        summary["bytes_reused"] += len(candidate)
                        summary["chunk_sources"][f"{entry.relative_path}#{index}"] = "resume-cache"
                    else:
                        staged_path.unlink(missing_ok=True)
                if chunk_bytes is None:
                    chunk_bytes, source_id = self._fetch_verified_chunk(
                        expected_hash=expected_hash,
                        entry=entry,
                        index=index,
                        peers=peers,
                        summary=summary,
                    )
                    staged_path.write_bytes(chunk_bytes)
                    summary["chunks_downloaded"] += 1
                    summary["bytes_downloaded"] += len(chunk_bytes)
                    summary["chunk_sources"][f"{entry.relative_path}#{index}"] = source_id
                    if source_id not in summary["seeders_contacted"]:
                        summary["seeders_contacted"].append(source_id)
                assembled.extend(chunk_bytes)

            payload = bytes(assembled)
            if len(payload) != entry.size:
                raise ChunkIntegrityError(
                    f"Reassembled '{entry.relative_path}' has {len(payload)} bytes, expected {entry.size}."
                )
            if hashlib.sha256(payload).hexdigest() != entry.sha256:
                raise ChunkIntegrityError(
                    f"SHA-256 mismatch after reassembling '{entry.relative_path}'."
                )
            recomputed = self.compute_merkle_root(
                [
                    hashlib.sha256(payload[offset : offset + entry.chunk_size]).hexdigest()
                    for offset in range(0, len(payload), entry.chunk_size)
                ]
            )
            if recomputed != entry.merkle_root:
                raise ChunkIntegrityError(
                    f"Merkle root mismatch after reassembling '{entry.relative_path}'."
                )
            file_target.write_bytes(payload)
            summary["files_written"].append(entry.relative_path)
            summary["total_bytes"] += entry.size
            shutil.rmtree(stage_dir, ignore_errors=True)

        if staging_root.exists() and not any(staging_root.iterdir()):
            staging_root.rmdir()

        written_digests = [
            hashlib.sha256((target_path / relative).read_bytes()).hexdigest()
            for relative in summary["files_written"]
        ]
        recomputed_bundle_root = self.compute_merkle_root(written_digests)
        if recomputed_bundle_root != manifest.bundle_merkle_root:
            raise ChunkIntegrityError(
                "Bundle Merkle root mismatch after chunked reassembly "
                f"({recomputed_bundle_root} != {manifest.bundle_merkle_root})."
            )
        summary["verified"] = True
        summary["bundle_merkle_root"] = recomputed_bundle_root
        return summary

    def _fetch_verified_chunk(
        self,
        expected_hash: str,
        entry: ChunkedFileEntry,
        index: int,
        peers: List["ContentAddressedSwarmStore"],
        summary: Dict[str, Any],
    ) -> Tuple[bytes, str]:
        """Fetch a single chunk from the advertised seeder (or any peer) and verify its digest."""
        candidates: List["ContentAddressedSwarmStore"] = []
        assigned = entry.chunk_seeders[index] if index < len(entry.chunk_seeders) else None
        if assigned:
            for peer in peers:
                if peer.node_id == assigned:
                    candidates.append(peer)
        if self.has_chunk(expected_hash):
            candidates.append(self)
        for peer in peers:
            if not any(peer is candidate for candidate in candidates):
                candidates.append(peer)

        for candidate in candidates:
            try:
                payload = candidate.read_chunk(expected_hash)
            except Exception:
                # Seeder is unreachable or does not host the chunk: try the next peer.
                continue
            if hashlib.sha256(payload).hexdigest() != expected_hash:
                summary["corrupt_chunks_rejected"] += 1
                continue
            return payload, candidate.node_id

        raise ChunkIntegrityError(
            f"Unable to fetch chunk #{index} ({expected_hash}) of '{entry.relative_path}': "
            "no seeder returned a valid copy (corrupted chunks were rejected)."
        )

    @staticmethod
    def _safe_bundle_id(bundle_id: str) -> str:
        """Normalize a bundle id so it can be used as a portable file name."""
        return (
            str(bundle_id)
            .replace("..", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )

    # ------------------------------------------------------------------
    # Season 9 (HM-S9-06): Heartbeat-Backed Data Shard Leasing
    # ------------------------------------------------------------------

    def grant_shard_lease(
        self,
        shard_cid: str,
        worker_id: str,
        alpha: float = 2.0,
        t_step_s: float = 1.0,
        rtt_s: float = 0.2,
        now_s: Optional[float] = None,
    ) -> DataShardLease:
        """Grant an exclusive heartbeat-backed lease on `shard_cid` with TTL `alpha * t_step_s + rtt_s`."""
        cid = shard_cid.strip() if isinstance(shard_cid, str) else ""
        wid = worker_id.strip() if isinstance(worker_id, str) else ""
        if not cid or not wid:
            raise ValueError("shard_cid and worker_id must be non-empty")
        if cid in self._committed_shards:
            raise ValueError(f"Shard {cid!r} has already been committed and cannot be re-leased")
        if cid in self._active_shard_lease_by_cid:
            active_id = self._active_shard_lease_by_cid[cid]
            raise ValueError(f"Shard {cid!r} already has an active lease ({active_id!r})")

        now = _lease_clock(now_s)
        ttl = compute_lease_ttl_s(alpha=alpha, t_step_s=t_step_s, rtt_s=rtt_s)
        self._lease_seq += 1
        lease_id = f"lease-{self._lease_seq:06d}-{hashlib.sha256(f'{cid}:{wid}:{self._lease_seq}'.encode('utf-8')).hexdigest()[:10]}"
        lease = DataShardLease(
            lease_id=lease_id,
            shard_cid=cid,
            worker_id=wid,
            granted_at_s=now,
            ttl_s=ttl,
            expires_at_s=now + ttl,
            status="active",
        )
        self._shard_leases[lease_id] = lease
        self._active_shard_lease_by_cid[cid] = lease_id
        if cid in self._reclaim_queue:
            self._reclaim_queue.remove(cid)
        return lease

    def heartbeat_shard_lease(
        self,
        lease_id: str,
        now_s: Optional[float] = None,
    ) -> DataShardLease:
        """Refresh an active shard lease by extending `expires_at_s = now + ttl_s`."""
        if lease_id not in self._shard_leases:
            raise KeyError(f"Unknown lease_id {lease_id!r}")
        lease = self._shard_leases[lease_id]
        if lease.status != "active":
            raise ValueError(f"Cannot heartbeat lease {lease_id!r} in status {lease.status!r}")
        now = _lease_clock(now_s)
        if now > lease.expires_at_s:
            raise ValueError(f"Lease {lease_id!r} expired at {lease.expires_at_s:.3f}s (now={now:.3f}s)")
        lease.expires_at_s = now + lease.ttl_s
        lease.heartbeat_count += 1
        return lease

    def commit_shard_lease(
        self,
        lease_id: str,
        worker_id: str,
        now_s: Optional[float] = None,
    ) -> DataShardLease:
        """Commit a leased shard batch, permanently preventing duplicate commits."""
        if lease_id not in self._shard_leases:
            raise KeyError(f"Unknown lease_id {lease_id!r}")
        lease = self._shard_leases[lease_id]
        if lease.status != "active":
            raise ValueError(f"Lease {lease_id!r} is not active (status={lease.status!r})")
        if lease.worker_id != str(worker_id):
            raise ValueError(
                f"Worker {worker_id!r} does not own lease {lease_id!r} (owner={lease.worker_id!r})"
            )
        now = _lease_clock(now_s)
        if now > lease.expires_at_s:
            raise ValueError(f"Cannot commit expired lease {lease_id!r} (now={now:.3f}s > {lease.expires_at_s:.3f}s)")
        if lease.shard_cid in self._committed_shards:
            raise ValueError(f"Shard {lease.shard_cid!r} was already committed")

        lease.status = "committed"
        lease.committed_at_s = now
        self._committed_shards[lease.shard_cid] = lease.lease_id
        self._active_shard_lease_by_cid.pop(lease.shard_cid, None)
        return lease

    def reclaim_expired_leases(
        self,
        now_s: Optional[float] = None,
    ) -> List[DataShardLease]:
        """Reclaim all active leases whose `expires_at_s < now` and return them to the reclaim queue."""
        now = _lease_clock(now_s)
        reclaimed: List[DataShardLease] = []
        for lease in self._shard_leases.values():
            if lease.status == "active" and now > lease.expires_at_s:
                lease.status = "reclaimed"
                lease.reclaimed_at_s = now
                self._active_shard_lease_by_cid.pop(lease.shard_cid, None)
                if (
                    lease.shard_cid not in self._committed_shards
                    and lease.shard_cid not in self._reclaim_queue
                ):
                    self._reclaim_queue.append(lease.shard_cid)
                reclaimed.append(lease)
        return reclaimed

    def lease_ledger_summary(self) -> Dict[str, Any]:
        """Return exact conservation telemetry (`granted == committed + reclaimed + active`, 0 duplicates)."""
        total_granted = len(self._shard_leases)
        committed_count = sum(1 for l in self._shard_leases.values() if l.status == "committed")
        reclaimed_count = sum(1 for l in self._shard_leases.values() if l.status == "reclaimed")
        active_count = sum(1 for l in self._shard_leases.values() if l.status == "active")
        unique_committed = len(self._committed_shards)
        duplicate_commits = committed_count - unique_committed
        conservation = (
            total_granted == (committed_count + reclaimed_count + active_count)
            and committed_count == (total_granted - reclaimed_count - active_count)
            and duplicate_commits == 0
        )
        return {
            "strategy_id": "HM-S9-06-HEARTBEAT-DATA-SHARD-LEASING",
            "total_granted": total_granted,
            "committed_count": committed_count,
            "reclaimed_count": reclaimed_count,
            "active_count": active_count,
            "unique_committed_shards": unique_committed,
            "duplicate_commits": duplicate_commits,
            "reclaim_queue_depth": len(self._reclaim_queue),
            "conservation_holds": bool(conservation),
        }

