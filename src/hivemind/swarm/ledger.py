# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Phase 2.5: Bitcoin-Inspired Proof-of-Useful-Work (PoUW) Cryptographic Ledger (`$HIVE`).

Implements:
- 21,000,000 `$HIVE` maximum supply cap with 50.0 `$HIVE` initial block reward and epoch halving
  every 10,000 outer DiLoCo steps.
- Multi-pool block reward distribution across:
    1. GPU Training Miners (60% training + 15% inference reserve pool = 75%)
    2. Dataset & Expert Storage Seeders (15% Proof-of-Storage pool)
    3. Spot-Check Validators (10% Proof-of-Verification pool)
- Automatic Byzantine slashing (50% balance slash, reputation penalty, and `slashed_registry` ban)
  when a worker fails Deterministic Spot-Check replay or Multi-Krum filtering.
- Circular-economy P2P inference micro-fee settlement (`settle_swarm_inference`).
- Phase S-5 Ledger Hardening (`LEDGER-01` .. `LEDGER-04`):
    * Blocks negative/zero token counts or negative fee rates in `settle_swarm_inference`.
    * Blocks negative/non-finite `credit_reward` amounts and bars slashed Byzantine nodes from
      receiving rewards or provider fees.
    * Deduplicates per-block `WorkReceipt`, `StorageProofReceipt`, and validator IDs to prevent
      receipt-flooding inflation, and enforces the 21,000,000 `$HIVE` supply ceiling.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from hivemind.security.swarm_runtime_hardening import SwarmSecurityViolationError


@dataclass
class WorkReceipt:
    """Cryptographic receipt of a worker node's DiLoCo-QLoRA inner-loop contribution."""

    node_id: str
    steam_id64: str
    branch_name: str
    flops_contributed: float
    cosine_alignment: float
    spot_check_passed: bool
    multikrum_accepted: bool
    delta_sha256: str


@dataclass
class StorageProofReceipt:
    """Merkle-tree byte-range Proof-of-Storage receipt for a pinned Hugging Face shard."""

    node_id: str
    shard_cid: str
    challenged_chunk_index: int
    merkle_proof_hash: str
    bytes_stored: int
    verified: bool


@dataclass
class InferenceSettlementReceipt:
    """Micro-fee settlement receipt for a P2P speculative MoE inference session."""

    consumer_node_id: str
    expert_provider_nodes: List[str]
    domain_expert: str
    verified_tokens: int
    rtt_speedup: Optional[float]
    fee_paid_hive: float


@dataclass
class HivePoUWBlock:
    """Single block in the `$HIVE` Proof-of-Useful-Work hash-chain."""

    block_height: int
    prev_block_hash: str
    timestamp: float
    outer_step: int
    expert_cid: str
    merkle_root: str
    total_block_reward: float
    reward_allocations: Dict[str, float]
    slashed_nodes: List[str]
    block_hash: str = ""

    def compute_hash(self) -> str:
        payload = {
            "block_height": self.block_height,
            "prev_block_hash": self.prev_block_hash,
            "timestamp": round(self.timestamp, 4),
            "outer_step": self.outer_step,
            "expert_cid": self.expert_cid,
            "merkle_root": self.merkle_root,
            "total_block_reward": round(self.total_block_reward, 8),
            "reward_allocations": {
                k: round(v, 8) for k, v in sorted(self.reward_allocations.items())
            },
            "slashed_nodes": sorted(self.slashed_nodes),
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


class HiveSwarmLedger:
    """Bitcoin-inspired Proof-of-Useful-Work (PoUW) Ledger for Project HiveMind.

    Rewards verified DiLoCo-QLoRA training, Merkle dataset storage, and MoE inference.
    """

    MAX_SUPPLY_HIVE = 21_000_000.0
    INITIAL_BLOCK_REWARD = 50.0
    HALVING_INTERVAL_BLOCKS = 10_000

    def __init__(self, ledger_path: Optional[Path] = None) -> None:
        self.ledger_path = Path(ledger_path) if ledger_path is not None else None
        self.chain: List[HivePoUWBlock] = []
        self.balances: Dict[str, float] = {}
        self.reputation_scores: Dict[str, int] = {}
        self.slashed_registry: List[str] = []
        self.settlement_history: List[InferenceSettlementReceipt] = []
        if self.ledger_path is not None and self.ledger_path.exists():
            self._load()
        else:
            self._init_genesis_block()

    def _load(self) -> None:
        assert self.ledger_path is not None
        try:
            raw = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            self.chain = [HivePoUWBlock(**b) for b in raw.get("chain", [])]
            self.balances = {str(k): float(v) for k, v in raw.get("balances", {}).items()}
            self.reputation_scores = {
                str(k): int(v) for k, v in raw.get("reputation_scores", {}).items()
            }
            self.slashed_registry = [str(x) for x in raw.get("slashed_registry", [])]
            self.settlement_history = [
                InferenceSettlementReceipt(**s) for s in raw.get("settlement_history", [])
            ]
        except Exception as exc:
            raise SwarmSecurityViolationError(
                "LEDGER-06",
                "CORRUPTED_LEDGER_FILE",
                f"Failed to parse persisted ledger at {self.ledger_path}: {exc}",
            ) from exc
        if not self.verify_chain_integrity():
            raise SwarmSecurityViolationError(
                "LEDGER-06",
                "CORRUPTED_LEDGER_CHAIN",
                f"Persisted ledger at {self.ledger_path} failed cryptographic block hash verification.",
            )

    def _init_genesis_block(self) -> None:
        genesis = HivePoUWBlock(
            block_height=0,
            prev_block_hash="0" * 64,
            timestamp=1700000000.0,
            outer_step=0,
            expert_cid="cid://sha256/genesis-qwen25-nf4-backbone",
            merkle_root=hashlib.sha256(b"HiveMind-SETI-Swarm-Genesis").hexdigest(),
            total_block_reward=0.0,
            reward_allocations={},
            slashed_nodes=[],
        )
        genesis.block_hash = genesis.compute_hash()
        self.chain.append(genesis)

    def total_circulating_supply(self) -> float:
        """Return total `$HIVE` currently held across all node balances."""
        return round(sum(self.balances.values()), 6)

    def block_reward_at_height(self, height: int) -> float:
        if height < 0:
            return 0.0
        halvings = height // self.HALVING_INTERVAL_BLOCKS
        if halvings >= 64:
            return 0.0
        raw_reward = self.INITIAL_BLOCK_REWARD * (0.5**halvings)
        remaining_cap = max(0.0, self.MAX_SUPPLY_HIVE - self.total_circulating_supply())
        return round(min(raw_reward, remaining_cap), 6)

    @staticmethod
    def _compute_merkle_root(items: List[str]) -> str:
        if not items:
            return hashlib.sha256(b"empty").hexdigest()
        layer = [hashlib.sha256(x.encode("utf-8")).hexdigest() for x in items]
        while len(layer) > 1:
            if len(layer) % 2 == 1:
                layer.append(layer[-1])
            next_layer = []
            for i in range(0, len(layer), 2):
                combined = (layer[i] + layer[i + 1]).encode("utf-8")
                next_layer.append(hashlib.sha256(combined).hexdigest())
            layer = next_layer
        return layer[0]

    def mine_pouw_block(
        self,
        outer_step: int,
        expert_cid: str,
        work_receipts: List[WorkReceipt],
        storage_receipts: List[StorageProofReceipt],
        validator_node_ids: List[str],
    ) -> HivePoUWBlock:
        height = len(self.chain)
        reward = self.block_reward_at_height(height)
        allocations: Dict[str, float] = {}
        slashed: List[str] = []

        # Rule LEDGER-03: Deduplicate work receipts per node_id & reject negative/NaN FLOPs
        dedup_work: List[WorkReceipt] = []
        seen_workers: set[str] = set()
        for w in work_receipts:
            if not math.isfinite(float(w.flops_contributed)) or float(w.flops_contributed) < 0.0:
                raise SwarmSecurityViolationError(
                    "LEDGER-03",
                    "INVALID_NEGATIVE_OR_NAN_FLOPS_RECEIPT",
                    f"Worker '{w.node_id}' submitted invalid flops_contributed={w.flops_contributed}.",
                )
            if w.node_id in seen_workers:
                raise SwarmSecurityViolationError(
                    "LEDGER-03",
                    "DUPLICATE_WORK_RECEIPT_IN_BLOCK",
                    f"Duplicate WorkReceipt for worker '{w.node_id}' in block height {height}.",
                )
            seen_workers.add(w.node_id)
            dedup_work.append(w)

        # 1. Process GPU Training Miners (60% of Block Reward + 15% Inference Reserve Pool)
        gpu_pool = 0.75 * reward
        valid_workers = [
            w
            for w in dedup_work
            if w.spot_check_passed
            and w.multikrum_accepted
            and w.node_id not in self.slashed_registry
            and float(w.flops_contributed) > 0.0
        ]
        for w in dedup_work:
            if not w.spot_check_passed or not w.multikrum_accepted:
                slashed.append(w.node_id)
                if w.node_id not in self.slashed_registry:
                    self.slashed_registry.append(w.node_id)
                self.reputation_scores[w.node_id] = self.reputation_scores.get(w.node_id, 100) - 50
                # Slash 50% of any existing balance for Byzantine behavior
                if w.node_id in self.balances:
                    self.balances[w.node_id] = round(self.balances[w.node_id] * 0.5, 6)
                else:
                    self.balances[w.node_id] = 0.0

        total_weighted_flops = sum(
            w.flops_contributed * max(0.05, w.cosine_alignment) for w in valid_workers
        )
        if total_weighted_flops > 0:
            for w in valid_workers:
                weight = (
                    w.flops_contributed * max(0.05, w.cosine_alignment)
                ) / total_weighted_flops
                earned = round(gpu_pool * weight, 6)
                allocations[w.node_id] = round(allocations.get(w.node_id, 0.0) + earned, 6)
                self.reputation_scores[w.node_id] = self.reputation_scores.get(w.node_id, 100) + 10

        # 2. Process Proof-of-Storage Seeders (15% of Block Reward)
        storage_pool = 0.15 * reward
        seen_seeders: set[str] = set()
        valid_seeders: List[StorageProofReceipt] = []
        for s in storage_receipts:
            if int(s.bytes_stored) < 0:
                raise SwarmSecurityViolationError(
                    "LEDGER-03",
                    "NEGATIVE_STORAGE_BYTES_RECEIPT",
                    f"Seeder '{s.node_id}' submitted negative bytes_stored={s.bytes_stored}.",
                )
            if s.verified and s.node_id not in self.slashed_registry and s.node_id not in seen_seeders and int(s.bytes_stored) > 0:
                seen_seeders.add(s.node_id)
                valid_seeders.append(s)

        total_bytes = sum(s.bytes_stored for s in valid_seeders)
        if total_bytes > 0:
            for s in valid_seeders:
                earned = round(storage_pool * (s.bytes_stored / total_bytes), 6)
                allocations[s.node_id] = round(allocations.get(s.node_id, 0.0) + earned, 6)

        # 3. Process Spot-Check Validators (10% of Block Reward)
        validator_pool = 0.10 * reward
        unique_validators = [
            vid for vid in dict.fromkeys(validator_node_ids) if vid not in self.slashed_registry
        ]
        if unique_validators:
            per_val = round(validator_pool / len(unique_validators), 6)
            for vid in unique_validators:
                allocations[vid] = round(allocations.get(vid, 0.0) + per_val, 6)

        # Commit balances
        for node_id, amount in allocations.items():
            self.balances[node_id] = round(self.balances.get(node_id, 0.0) + amount, 6)

        merkle_leaves = (
            [w.delta_sha256 for w in dedup_work]
            + [s.merkle_proof_hash for s in storage_receipts]
            + [expert_cid]
        )
        merkle_root = self._compute_merkle_root(merkle_leaves)

        block = HivePoUWBlock(
            block_height=height,
            prev_block_hash=self.chain[-1].block_hash,
            timestamp=time.time(),
            outer_step=outer_step,
            expert_cid=expert_cid,
            merkle_root=merkle_root,
            total_block_reward=reward,
            reward_allocations=allocations,
            slashed_nodes=slashed,
        )
        block.block_hash = block.compute_hash()
        self.chain.append(block)
        self._persist()
        return block

    def settle_swarm_inference(
        self,
        consumer_node_id: str,
        provider_node_ids: List[str],
        domain_expert: str,
        verified_tokens: int,
        rtt_speedup: Optional[float],
        rate_per_token: float = 0.005,
    ) -> InferenceSettlementReceipt:
        # Rule LEDGER-01: Prevent negative token / negative rate balance minting exploit
        if int(verified_tokens) <= 0:
            raise SwarmSecurityViolationError(
                "LEDGER-01",
                "NEGATIVE_OR_ZERO_INFERENCE_TOKENS",
                f"verified_tokens ({verified_tokens}) must be a positive integer >= 1.",
            )
        if not math.isfinite(float(rate_per_token)) or float(rate_per_token) <= 0.0 or float(rate_per_token) > 10.0:
            raise SwarmSecurityViolationError(
                "LEDGER-01",
                "INVALID_INFERENCE_FEE_RATE",
                f"rate_per_token ({rate_per_token}) must be finite and within (0.0, 10.0].",
            )
        if not provider_node_ids:
            raise SwarmSecurityViolationError(
                "LEDGER-01",
                "EMPTY_INFERENCE_PROVIDERS",
                "provider_node_ids must contain at least one active expert provider node.",
            )
        for pid in provider_node_ids:
            if pid in self.slashed_registry:
                raise SwarmSecurityViolationError(
                    "LEDGER-01",
                    "SLASHED_PROVIDER_SETTLEMENT_BLOCKED",
                    f"Provider node '{pid}' is in the Byzantine slashed_registry and cannot collect fees.",
                )

        total_fee = round(int(verified_tokens) * float(rate_per_token), 6)
        consumer_bal = self.balances.get(consumer_node_id, 0.0)
        if consumer_bal < total_fee:
            raise ValueError(
                f"Node {consumer_node_id} has insufficient $HIVE ({consumer_bal} < {total_fee})"
            )

        self.balances[consumer_node_id] = round(consumer_bal - total_fee, 6)
        share = round(total_fee / len(provider_node_ids), 6)
        for pid in provider_node_ids:
            self.balances[pid] = round(self.balances.get(pid, 0.0) + share, 6)

        receipt = InferenceSettlementReceipt(
            consumer_node_id=consumer_node_id,
            expert_provider_nodes=provider_node_ids,
            domain_expert=domain_expert,
            verified_tokens=int(verified_tokens),
            rtt_speedup=rtt_speedup,
            fee_paid_hive=total_fee,
        )
        self.settlement_history.append(receipt)
        self._persist()
        return receipt

    def credit_reward(
        self,
        node_id: str,
        amount_hive: float,
        reason: str = "",
    ) -> float:
        """Credit a peer reuse royalty or community bundle reward to `node_id`."""
        del reason
        if not node_id or not str(node_id).strip():
            raise SwarmSecurityViolationError(
                "LEDGER-02",
                "EMPTY_REWARD_NODE_ID",
                "Cannot credit $HIVE reward to an empty node_id.",
            )
        if node_id in self.slashed_registry:
            raise SwarmSecurityViolationError(
                "LEDGER-02",
                "SLASHED_NODE_REWARD_BLOCKED",
                f"Node '{node_id}' is in the Byzantine slashed_registry and cannot receive rewards.",
            )
        if not math.isfinite(float(amount_hive)) or float(amount_hive) <= 0.0:
            raise SwarmSecurityViolationError(
                "LEDGER-02",
                "NON_POSITIVE_OR_NON_FINITE_REWARD",
                f"amount_hive ({amount_hive}) must be finite and strictly positive (> 0.0).",
            )
        if self.total_circulating_supply() + float(amount_hive) > self.MAX_SUPPLY_HIVE:
            raise SwarmSecurityViolationError(
                "LEDGER-04",
                "MAX_SUPPLY_CAP_EXCEEDED",
                f"Crediting {amount_hive} $HIVE would exceed the 21,000,000 $HIVE supply cap.",
            )
        self.balances[node_id] = round(self.balances.get(node_id, 0.0) + float(amount_hive), 6)
        self.reputation_scores[node_id] = self.reputation_scores.get(node_id, 100) + 5
        self._persist()
        return self.balances[node_id]

    def get_balance(self, node_id: str) -> float:
        """Return the current `$HIVE` balance for `node_id`."""
        return round(self.balances.get(node_id, 0.0), 6)

    def verify_chain_integrity(self) -> bool:
        """Cryptographically verify every block hash and prev_block_hash link from genesis."""
        if not self.chain:
            return False
        for i, block in enumerate(self.chain):
            if block.compute_hash() != block.block_hash:
                return False
            if i > 0 and block.prev_block_hash != self.chain[i - 1].block_hash:
                return False
        return True

    def _persist(self) -> None:
        if self.ledger_path is not None:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "chain": [asdict(b) for b in self.chain],
                "balances": self.balances,
                "reputation_scores": self.reputation_scores,
                "slashed_registry": self.slashed_registry,
                "settlement_history": [asdict(s) for s in self.settlement_history],
            }
            payload = json.dumps(data, indent=2)
            tmp_path: Optional[Path] = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=str(self.ledger_path.parent),
                    prefix=f".{self.ledger_path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as tmp_file:
                    tmp_path = Path(tmp_file.name)
                    tmp_file.write(payload)
                    tmp_file.flush()
                    os.fsync(tmp_file.fileno())
                os.replace(tmp_path, self.ledger_path)
            except Exception:
                if tmp_path is not None and tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                raise


@dataclass
class IndemnityReceipt:
    """HM-S6-03: settlement record for one consumer indemnified after a proven batch fault."""

    fault_id: str
    commitment_root: str
    step: int
    layer: int
    fault_reason: str
    faulty_peer_id: str
    faulty_guild_id: str
    consumer_node_id: str
    claimed_hive: float
    paid_from_peer_stake: float
    paid_from_reserve: float
    indemnified_hive: float
    shortfall_hive: float
    stake_surplus_to_reserve: float
    settlement_ms: float


class GuildSlashingInsurancePool:
    """HM-S6-03: cross-guild stake pool that indemnifies consumers when a batch fault is proven.

    Peers bond `$HIVE` stake and guilds fund a shared reserve; both are transfers out of
    `HiveSwarmLedger.balances`, so nothing is minted and
    `ledger.total_circulating_supply() + pool.total_holdings()` is conserved. A claim is paid
    only after the pool re-runs the auditor's batch challenge itself, first from the faulty
    peer's slashed stake and then from the shared reserve. Settlement is a local, synchronous
    ledger update: `IndemnityReceipt.settlement_ms` measures it, and there is no on-chain step.
    """

    def __init__(self, ledger: HiveSwarmLedger, max_claim_hive: float = 1_000.0) -> None:
        self.ledger = ledger
        self.max_claim_hive = self._require_amount(max_claim_hive, "INVALID_CLAIM_CAP")
        self.peer_stakes: Dict[str, float] = {}
        self.peer_guilds: Dict[str, str] = {}
        self.reserve_hive = 0.0
        self.guild_reserve_contributions: Dict[str, float] = {}
        self.receipts: Dict[str, IndemnityReceipt] = {}

    @staticmethod
    def _require_amount(amount_hive: float, category: str) -> float:
        try:
            value = float(amount_hive)
        except (TypeError, ValueError):
            value = float("nan")
        if not math.isfinite(value) or value <= 0.0:
            raise SwarmSecurityViolationError(
                "LEDGER-05", category, f"amount ({amount_hive}) must be finite and > 0.0"
            )
        return round(value, 6)

    def _debit(self, node_id: str, amount: float) -> None:
        if not str(node_id).strip():
            raise SwarmSecurityViolationError("LEDGER-05", "EMPTY_NODE_ID", "node_id must be non-empty")
        if node_id in self.ledger.slashed_registry:
            raise SwarmSecurityViolationError(
                "LEDGER-05", "SLASHED_NODE_BLOCKED", f"Node '{node_id}' is in the slashed_registry."
            )
        balance = self.ledger.balances.get(node_id, 0.0)
        if balance < amount:
            raise SwarmSecurityViolationError(
                "LEDGER-05",
                "INSUFFICIENT_BALANCE",
                f"Node '{node_id}' holds {balance} $HIVE, needs {amount}.",
            )
        self.ledger.balances[node_id] = round(balance - amount, 6)

    def bond_stake(self, guild_id: str, peer_id: str, amount_hive: float) -> float:
        """Move `amount_hive` from `peer_id`'s balance into its bonded stake."""
        if not str(guild_id).strip():
            raise SwarmSecurityViolationError("LEDGER-05", "EMPTY_GUILD_ID", "guild_id must be non-empty")
        bonded_guild = self.peer_guilds.get(peer_id)
        if bonded_guild is not None and bonded_guild != guild_id:
            raise SwarmSecurityViolationError(
                "LEDGER-05",
                "PEER_ALREADY_BONDED_ELSEWHERE",
                f"Peer '{peer_id}' is bonded to guild '{bonded_guild}'.",
            )
        amount = self._require_amount(amount_hive, "NON_POSITIVE_OR_NON_FINITE_STAKE")
        self._debit(peer_id, amount)
        self.peer_guilds[peer_id] = guild_id
        self.peer_stakes[peer_id] = round(self.peer_stakes.get(peer_id, 0.0) + amount, 6)
        self.ledger._persist()
        return self.peer_stakes[peer_id]

    def contribute_reserve(self, guild_id: str, contributor_id: str, amount_hive: float) -> float:
        """Move `amount_hive` from `contributor_id`'s balance into the shared cross-guild reserve."""
        if not str(guild_id).strip():
            raise SwarmSecurityViolationError("LEDGER-05", "EMPTY_GUILD_ID", "guild_id must be non-empty")
        amount = self._require_amount(amount_hive, "NON_POSITIVE_OR_NON_FINITE_RESERVE")
        self._debit(contributor_id, amount)
        self.reserve_hive = round(self.reserve_hive + amount, 6)
        self.guild_reserve_contributions[guild_id] = round(
            self.guild_reserve_contributions.get(guild_id, 0.0) + amount, 6
        )
        self.ledger._persist()
        return self.reserve_hive

    def total_holdings(self) -> float:
        """`$HIVE` held by the pool: every bonded stake plus the shared reserve."""
        return round(sum(self.peer_stakes.values()) + self.reserve_hive, 6)

    def indemnify_proven_fault(
        self,
        auditor: Any,
        commitment: Any,
        opening: Any,
        claimed_activation: Any,
        recomputed_activation: Any,
        consumer_node_id: str,
        claimed_hive: float,
    ) -> IndemnityReceipt:
        """Re-verify one challenged leaf with `auditor`, then slash the peer and pay the consumer.

        One indemnity is paid per batch commitment root, however many of its leaves are bad.
        """
        verdict = auditor.verify_speculative_batch_challenge(
            commitment, opening, claimed_activation, recomputed_activation
        )
        if not verdict.get("fault_proven"):
            raise SwarmSecurityViolationError(
                "LEDGER-05", "FAULT_NOT_PROVEN", "The challenged leaf opened and passed the audit."
            )
        fault_id = hashlib.sha256(
            f"hivemind/opoi-batch/fault|{commitment.root}".encode("ascii")
        ).hexdigest()
        if fault_id in self.receipts:
            raise SwarmSecurityViolationError(
                "LEDGER-05", "DUPLICATE_FAULT_CLAIM", f"Batch {commitment.root[:16]} was already indemnified."
            )
        peer_id = str(commitment.peer_id)
        if peer_id not in self.peer_stakes:
            raise SwarmSecurityViolationError(
                "LEDGER-05", "UNBONDED_PEER", f"Peer '{peer_id}' has no bonded stake in this pool."
            )
        if not str(consumer_node_id).strip():
            raise SwarmSecurityViolationError("LEDGER-05", "EMPTY_NODE_ID", "consumer_node_id must be non-empty")
        if consumer_node_id == peer_id:
            raise SwarmSecurityViolationError(
                "LEDGER-05", "SELF_INDEMNIFICATION_BLOCKED", "A peer cannot claim against its own fault."
            )
        if consumer_node_id in self.ledger.slashed_registry:
            raise SwarmSecurityViolationError(
                "LEDGER-05", "SLASHED_NODE_BLOCKED", f"Node '{consumer_node_id}' is in the slashed_registry."
            )
        claimed = self._require_amount(claimed_hive, "NON_POSITIVE_OR_NON_FINITE_CLAIM")
        if claimed > self.max_claim_hive:
            raise SwarmSecurityViolationError(
                "LEDGER-05",
                "CLAIM_EXCEEDS_COVERAGE_CAP",
                f"claimed {claimed} $HIVE exceeds the {self.max_claim_hive} $HIVE coverage cap.",
            )

        started = time.perf_counter()
        stake = self.peer_stakes.pop(peer_id)
        from_stake = round(min(stake, claimed), 6)
        from_reserve = round(min(self.reserve_hive, claimed - from_stake), 6)
        surplus = round(stake - from_stake, 6)
        self.reserve_hive = round(self.reserve_hive - from_reserve + surplus, 6)
        paid = round(from_stake + from_reserve, 6)
        self.ledger.balances[consumer_node_id] = round(
            self.ledger.balances.get(consumer_node_id, 0.0) + paid, 6
        )
        if peer_id not in self.ledger.slashed_registry:
            self.ledger.slashed_registry.append(peer_id)
        self.ledger.reputation_scores[peer_id] = 0
        settlement_ms = (time.perf_counter() - started) * 1000.0

        receipt = IndemnityReceipt(
            fault_id=fault_id,
            commitment_root=str(commitment.root),
            step=int(verdict["step"]),
            layer=int(verdict["layer"]),
            fault_reason=str(verdict.get("fault_reason")),
            faulty_peer_id=peer_id,
            faulty_guild_id=self.peer_guilds.get(peer_id, ""),
            consumer_node_id=consumer_node_id,
            claimed_hive=claimed,
            paid_from_peer_stake=from_stake,
            paid_from_reserve=from_reserve,
            indemnified_hive=paid,
            shortfall_hive=round(claimed - paid, 6),
            stake_surplus_to_reserve=surplus,
            settlement_ms=round(settlement_ms, 4),
        )
        self.receipts[fault_id] = receipt
        self.ledger._persist()
        return receipt

    def snapshot(self) -> Dict[str, Any]:
        """JSON-serializable view of stakes, reserve and settled indemnities."""
        return {
            "peer_stakes": dict(self.peer_stakes),
            "peer_guilds": dict(self.peer_guilds),
            "reserve_hive": self.reserve_hive,
            "guild_reserve_contributions": dict(self.guild_reserve_contributions),
            "total_holdings": self.total_holdings(),
            "max_claim_hive": self.max_claim_hive,
            "indemnities": [asdict(r) for r in self.receipts.values()],
        }
