# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Phase 2: Regional DHT Swarm Router & 5-Token Speculative Decoding Engine.

Once an epoch of the 8x Sparse MoE is finalized, idle Steam nodes switch to Inference Mode:
- Residential fiber nodes (ping < 25 ms via STUN/TURN) form regional rings (e.g., EU-West-Paris).
- Local 0.5B draft model proposes gamma=5 speculative tokens locally.
- The P2P Sparse MoE swarm verifies all 5 tokens in a single parallel forward pass,
  dividing network latency overhead by 5 to achieve 15-25 tokens/sec over consumer broadband.

Season 4 Milestone `HM-S4-02` (Issue #27) upgrades the flat regional ring into a 2-Tier
Kademlia Super-Ring: the regional Geo-Ring stays the Tier-1 `< 20 ms` data plane while a
`ContinentalSuperRingRouter` elects *Continental Bridge* nodes per continent (`EU`, `US`,
`APAC`) that aggregate the Tier-1 rings into a single WAN-transit Tier-2 plane.

Phase S-5 Inference Ring Hardening (`RING-01` .. `RING-02`):
- Rejects spoofed (`ping_ms <= 0`), high-packet-loss (`> 15%`), high-jitter (`> 20 ms`),
  or insufficient-VRAM (`< 4.0 GB`) peers and deduplicates `peer_id` registrations.
- Enforces speculative decoding envelope (`1 <= draft_gamma <= 16`, `1 <= max_new_tokens <= 512`)
  and verifies finiteness (`torch.isfinite`) of MoE verification logits.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Sequence

import torch
import torch.nn as nn

from hivemind.core.btm_upcycle import SparseMoEMLPBlock, UpcycledSparseMoEModel
from hivemind.security.swarm_runtime_hardening import SwarmSecurityViolationError

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a circular runtime import.
    from hivemind.swarm.quic_transport import ContinentalSuperRingRouter
    from hivemind.swarm.swarm_fabric import PinnedRAMExpertDMAStreamer


def prefix_trie_branch_probabilities(
    draft_branches: Sequence[Sequence[int]],
    branch_step_probabilities: Sequence[Sequence[float]],
) -> tuple[list[tuple[int, ...]], list[float]]:
    """Prefix-trie nodes of the draft tree and a probability per node that sums to 1.

    Each node's raw weight is the product of the draft model's step probabilities along its
    prefix (the root's empty product is 1), so a prefix shared by several branches is counted
    once. The weights are normalised over the whole trie, root included, because the root
    position is on the verified path with certainty.
    """
    from hivemind.swarm.swarm_fabric import TokenTreeSpeculativeVerifier

    if len(draft_branches) != len(branch_step_probabilities):
        raise ValueError("one probability list per draft branch is required")
    depth = max((len(b) for b in draft_branches), default=0)
    verifier = TokenTreeSpeculativeVerifier()
    # The M3 verifier clamps its constructor gamma to [6, 8]; the ring allows [1, 16].
    verifier.gamma = max(1, depth)
    nodes = [tuple(n) for n in verifier.build_causal_tree_attention_mask(
        [list(b) for b in draft_branches], device="cpu"
    )["node_prefixes"]]

    raw: Dict[tuple[int, ...], float] = {(): 1.0}
    for branch, steps in zip(draft_branches, branch_step_probabilities):
        if len(steps) != len(branch):
            raise ValueError("each draft branch needs exactly one step probability per token")
        product = 1.0
        for i, p in enumerate(steps):
            product *= float(p)
            raw.setdefault(tuple(branch[: i + 1]), product)
    total = sum(raw[n] for n in nodes)
    if total <= 0.0:
        raise ValueError("draft step probabilities must give the trie positive total weight")
    return nodes, [raw[n] / total for n in nodes]


def _moe_layer_parents(model: UpcycledSparseMoEModel) -> list[nn.Module]:
    """The decoder layer that owns each MoE block, in block order."""
    blocks = model.get_moe_blocks()
    owner: Dict[int, nn.Module] = {}
    for mod in model.backbone.modules():
        for child in mod.children():
            if isinstance(child, SparseMoEMLPBlock):
                owner[id(child)] = mod
    return [owner[id(b)] for b in blocks]


@dataclass
class SwarmPeerNode:
    """Represents an idle gamer GPU hosting a subset of Sparse MoE expert layers in 4-bit."""

    peer_id: str
    steam_id64: str
    region: str
    ping_ms: float
    vram_gb: float
    hosted_expert_domain: str
    layer_range: tuple[int, int]
    stun_verified: bool = False
    jitter_ms: float = 0.0
    packet_loss_ratio: float = 0.0

    @property
    def eligible_for_low_latency_ring(self) -> bool:
        return (
            0.0 < self.ping_ms < 25.0
            and self.vram_gb >= 4.0
            and 0.0 <= self.packet_loss_ratio <= 0.15
            and 0.0 <= self.jitter_ms <= 20.0
        )

    @property
    def continent(self) -> str:
        """Continental Super-Ring bucket (`EU`/`US`/`APAC`) of this regional peer (`HM-S4-02`)."""
        from hivemind.swarm.quic_transport import classify_continent

        return classify_continent(self.region)


@dataclass
class SpeculativeSwarmResult:
    """Telemetry and output from a Regional P2P Swarm Speculative Decoding session."""

    region_ring: str
    active_ring_peers: int
    mean_ring_ping_ms: float
    prompt_length: int
    generated_tokens: list[int]
    draft_gamma: int
    swarm_verification_passes: int
    mean_acceptance_rate: float
    effective_tokens_per_sec: float
    latency_reduction_factor: float
    latency_basis: str = (
        "latency_reduction_factor = generated tokens / verification passes under a simulated "
        "per-pass RTT; effective_tokens_per_sec adds that simulated RTT to measured local compute "
        "time. Neither is a measured network speedup."
    )
    expert_routing_share: dict[str, float] = field(default_factory=dict)
    routing_tier: str = "regional_geo_ring"
    cross_continent_hops: int = 0
    estimated_route_rtt_ms: float = 0.0
    # HM-S6-02 token-tree expert prefetch. Hits/uses are exact counts on the verified path;
    # the latency figures inside each log entry's `streamer_report` are the streamer's simulation.
    expert_prefetch_enabled: bool = False
    expert_prefetch_calls: int = 0
    expert_prefetch_hits: int = 0
    expert_prefetch_expert_uses: int = 0
    expert_prefetch_hit_rate: float | None = None
    expert_prefetch_log: list[dict[str, Any]] = field(default_factory=list)
    expert_prefetch_basis: str = "disabled"


class RegionalDHTSwarmRing:
    """Petals-style DHT Swarm coordinator grouping low-latency (<25ms) peers into regional rings."""

    def __init__(self, region: str = "EU-West-Paris", max_ping_ms: float = 25.0) -> None:
        self.region = region
        self.max_ping_ms = max_ping_ms
        self.peers: list[SwarmPeerNode] = []
        self.super_ring_router: "ContinentalSuperRingRouter | None" = None
        self.hierarchical_routes: list[Dict[str, Any]] = []

    def register_peer(self, peer: SwarmPeerNode) -> bool:
        # Rule RING-01: Reject spoofed ping (<= 0), high packet loss (> 15%), high jitter (> 20ms), or low VRAM (< 4GB)
        if not peer.peer_id or not str(peer.peer_id).strip() or not peer.steam_id64:
            return False
        if not math.isfinite(float(peer.ping_ms)) or float(peer.ping_ms) <= 0.0:
            return False
        if not math.isfinite(float(peer.vram_gb)) or float(peer.vram_gb) < 4.0:
            return False
        if not math.isfinite(float(peer.packet_loss_ratio)) or not (0.0 <= float(peer.packet_loss_ratio) <= 0.15):
            return False
        if not math.isfinite(float(peer.jitter_ms)) or not (0.0 <= float(peer.jitter_ms) <= 20.0):
            return False

        if peer.region == self.region and peer.ping_ms < self.max_ping_ms:
            self.peers = [p for p in self.peers if p.peer_id != peer.peer_id]
            self.peers.append(peer)
            self._sync_peer_to_super_ring(peer)
            return True
        return False

    # ------------------------------------------------------------------
    # HM-S4-02: 2-Tier Kademlia Continental Super-Ring
    # ------------------------------------------------------------------
    def attach_continental_super_ring(self, router: "ContinentalSuperRingRouter") -> None:
        """Bind a 2-Tier Continental Super-Ring router to this regional ring (`HM-S4-02`).

        Every peer already registered in the Tier-1 regional ring is mirrored into the
        Super-Ring so that Continental Bridge elections see the full regional membership.
        """
        self.super_ring_router = router
        for peer in self.peers:
            self._sync_peer_to_super_ring(peer)

    def _ensure_super_ring(self) -> "ContinentalSuperRingRouter":
        """Return the attached Super-Ring router, lazily creating a default one."""
        if self.super_ring_router is None:
            from hivemind.swarm.quic_transport import ContinentalSuperRingRouter

            self.super_ring_router = ContinentalSuperRingRouter()
        return self.super_ring_router

    def _sync_peer_to_super_ring(self, peer: SwarmPeerNode) -> bool:
        """Mirror a Tier-1 peer into the Tier-2 Super-Ring (respecting the `< 20 ms` SLA)."""
        if self.super_ring_router is None:
            return False
        from hivemind.swarm.quic_transport import SuperRingPeerNode

        node = SuperRingPeerNode(
            peer_id=peer.peer_id,
            region=peer.region,
            rtt_ms=float(peer.ping_ms),
            jitter_ms=float(peer.jitter_ms),
            packet_loss_ratio=float(peer.packet_loss_ratio),
            stun_verified=bool(peer.stun_verified),
        )
        return self.super_ring_router.register_peer(node)

    def register_super_ring_peer(self, peer: SwarmPeerNode) -> bool:
        """Admit a peer of *any* continent into the Tier-2 Continental Super-Ring.

        Unlike `register_peer`, the peer does not need to belong to this regional ring: the
        Super-Ring aggregates `EU`/`US`/`APAC` rings into one WAN-transit plane (`HM-S4-02`).
        """
        router = self._ensure_super_ring()
        from hivemind.swarm.quic_transport import SuperRingPeerNode

        node = SuperRingPeerNode(
            peer_id=peer.peer_id,
            region=peer.region,
            rtt_ms=float(peer.ping_ms),
            jitter_ms=float(peer.jitter_ms),
            packet_loss_ratio=float(peer.packet_loss_ratio),
            stun_verified=bool(peer.stun_verified),
        )
        return router.register_peer(node)

    def resolve_hierarchical_route(
        self, source_peer_id: str, target_peer_id: str
    ) -> Dict[str, Any]:
        """Resolve the 2-Tier path (`Continental Bridge -> Regional Geo-Ring`) between two peers."""
        router = self._ensure_super_ring()
        route = router.route(source_peer_id, target_peer_id)
        self.hierarchical_routes.append(route)
        return route

    def route_to_region(self, source_peer_id: str, target_region: str) -> Dict[str, Any]:
        """Route towards the lowest-RTT peer of `target_region` via the Super-Ring."""
        router = self._ensure_super_ring()
        target = router.find_peer_in_region(target_region)
        if target is None:
            raise KeyError(f"No Super-Ring peer registered for region '{target_region}'.")
        return self.resolve_hierarchical_route(source_peer_id, target.peer_id)

    def run_speculative_inference(
        self,
        moe_target_model: UpcycledSparseMoEModel,
        draft_model: nn.Module,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 20,
        draft_gamma: int = 5,
        simulated_network_rtt_ms: float = 18.5,
        expert_streamer: "PinnedRAMExpertDMAStreamer | None" = None,
        expert_names: Sequence[str] | None = None,
        max_prefetch_experts: int = 2,
    ) -> SpeculativeSwarmResult:
        """Generate tokens using local draft speculation (gamma=5) + single-pass MoE swarm verification.

        With `expert_streamer` set (HM-S6-02), every verification pass also prefetches experts for
        each MoE layer `L+1` from the residual stream leaving layer `L`: the draft chain's prefix
        trie gives one branch per prefix, weighted by the product of the draft's step
        probabilities, and `prefetch_token_tree_lookahead` is issued before layer `L+1` runs.
        The result records what was prefetched and the exact share of expert uses on the
        verified path that were among them. Default off: the generated tokens are unchanged.
        """
        # Rule RING-02: Enforce speculative decoding bounds to prevent unbounded forward-pass DoS
        if int(draft_gamma) < 1 or int(draft_gamma) > 16:
            raise SwarmSecurityViolationError(
                "RING-02",
                "SPECULATIVE_GAMMA_OUT_OF_BOUNDS",
                f"draft_gamma ({draft_gamma}) must be within [1, 16].",
            )
        if int(max_new_tokens) < 1 or int(max_new_tokens) > 512:
            raise SwarmSecurityViolationError(
                "RING-02",
                "MAX_NEW_TOKENS_OUT_OF_BOUNDS",
                f"max_new_tokens ({max_new_tokens}) must be within [1, 512].",
            )
        if prompt_ids.numel() == 0:
            raise SwarmSecurityViolationError(
                "RING-02",
                "EMPTY_PROMPT_TENSOR",
                "prompt_ids must contain at least 1 token.",
            )

        device = next(moe_target_model.parameters()).device
        moe_target_model.eval()
        draft_model.eval()
        moe_target_model.reset_routing_telemetry()

        seq = prompt_ids.to(device, dtype=torch.long)
        if seq.dim() == 1:
            seq = seq.unsqueeze(0)
        prompt_len = seq.size(1)

        generated: list[int] = []
        verification_passes = 0
        accepted_total = 0
        proposed_total = 0

        prefetch_on = expert_streamer is not None
        prefetch_calls = 0
        prefetch_hits = 0
        prefetch_uses = 0
        prefetch_log: list[dict[str, Any]] = []
        prefetch_basis = "disabled"
        if prefetch_on:
            names = list(expert_names) if expert_names is not None else list(moe_target_model.expert_domains)
            if len(names) != moe_target_model.num_experts:
                raise ValueError(
                    f"expert_names has {len(names)} entries but the target model has "
                    f"{moe_target_model.num_experts} experts."
                )
            target_blocks = moe_target_model.get_moe_blocks()
            for blk in target_blocks[1:]:
                if not hasattr(blk.router, "predict_candidate_experts"):
                    raise TypeError(
                        "expert prefetch needs a router with predict_candidate_experts() on every "
                        "MoE layer after the first (install one with hotplug_three_tier_experts)."
                    )
            layer_parents = _moe_layer_parents(moe_target_model)
            if len(target_blocks) < 2:
                prefetch_basis = (
                    "enabled but idle: the target model has no next MoE layer to prefetch for"
                )
            else:
                prefetch_basis = (
                    "measured: exact count of (position, expert-slot) uses on the verified path at "
                    "each prefetched layer that were in that layer's prefetched set; the latency "
                    "figures inside expert_prefetch_log[*].streamer_report are the streamer's "
                    "simulated figures, not measured stalls"
                )

        t0 = time.perf_counter()
        with torch.no_grad():
            while len(generated) < max_new_tokens:
                # 1. Local Draft Pass (Top-1 single-expert mode) proposes gamma=5 tokens with zero network RTT
                draft_seq = seq.clone()
                draft_tokens: list[int] = []
                draft_step_probs: list[float] = []
                if isinstance(draft_model, UpcycledSparseMoEModel):
                    for blk in draft_model.get_moe_blocks():
                        blk.top_k = 1
                        blk.router.top_k = 1
                for _ in range(draft_gamma):
                    d_out = draft_model(input_ids=draft_seq)
                    next_tok = int(torch.argmax(d_out.logits[:, -1, :], dim=-1).item())
                    draft_tokens.append(next_tok)
                    if prefetch_on:
                        d_probs = torch.softmax(d_out.logits[0, -1, :].to(torch.float32), dim=-1)
                        draft_step_probs.append(float(d_probs[next_tok].item()))
                    draft_seq = torch.cat(
                        [draft_seq, torch.tensor([[next_tok]], device=device, dtype=torch.long)],
                        dim=1,
                    )

                # 2. P2P Sparse MoE Swarm (Top-2 multi-expert mode) verifies all gamma=5 tokens in ONE forward pass
                if isinstance(moe_target_model, UpcycledSparseMoEModel):
                    for blk in moe_target_model.get_moe_blocks():
                        blk.top_k = moe_target_model.top_k
                        blk.router.top_k = moe_target_model.top_k
                verification_passes += 1
                start_pos = seq.size(1) - 1
                hooks: list[Any] = []
                pass_reports: list[dict[str, Any]] = []
                try:
                    if prefetch_on and len(target_blocks) >= 2:
                        nodes, branch_probs = prefix_trie_branch_probabilities([draft_tokens], [draft_step_probs])
                        # Trie node with prefix length j sits at sequence position start_pos + j.
                        tree_positions = [start_pos + len(n) for n in nodes]
                        for nxt in range(1, len(target_blocks)):
                            hooks.append(
                                layer_parents[nxt].register_forward_pre_hook(
                                    self._make_prefetch_hook(
                                        expert_streamer, target_blocks[nxt].router, names, nxt,
                                        nodes, branch_probs, tree_positions, max_prefetch_experts,
                                        pass_reports,
                                    ),
                                    with_kwargs=True,
                                )
                            )
                            hooks.append(
                                target_blocks[nxt].router.register_forward_hook(
                                    self._make_router_capture_hook(pass_reports, nxt)
                                )
                            )
                    m_out = moe_target_model(input_ids=draft_seq)
                finally:
                    for h in hooks:
                        h.remove()
                if not bool(torch.isfinite(m_out.logits).all().item()):
                    raise SwarmSecurityViolationError(
                        "RING-02",
                        "POISONED_MOE_RING_LOGITS_DETECTED",
                        "Non-finite (NaN/Inf) logits returned by P2P Sparse MoE verification pass.",
                    )
                target_logits = m_out.logits[:, start_pos : start_pos + draft_gamma + 1, :].squeeze(0)
                target_probs = torch.softmax(target_logits.to(torch.float32), dim=-1)
                target_preds = torch.argmax(target_probs, dim=-1)

                draft_accepted_in_pass = 0
                emitted_before_bonus = 0
                for i, d_tok in enumerate(draft_tokens):
                    proposed_total += 1
                    t_tok = int(target_preds[i].item())
                    max_p = float(target_probs[i, t_tok].item())
                    draft_p = float(target_probs[i, d_tok].item())
                    # Speculative rejection sampling criterion: accept if matches argmax or within high-density support
                    if d_tok == t_tok or (max_p > 0 and (draft_p / max_p) >= 0.65):
                        generated.append(d_tok)
                        accepted_total += 1
                        draft_accepted_in_pass += 1
                        emitted_before_bonus += 1
                        if len(generated) >= max_new_tokens:
                            break
                    else:
                        # Accept the target MoE correction token and resync draft sequence;
                        # rejected draft tokens do not count as accepted draft proposals (IP-114).
                        generated.append(t_tok)
                        emitted_before_bonus += 1
                        break

                # Only if all gamma draft tokens were genuinely accepted do we emit a bonus token at draft_gamma (IP-114)
                if draft_accepted_in_pass == draft_gamma and len(generated) < max_new_tokens:
                    bonus_tok = int(target_preds[draft_gamma].item())
                    generated.append(bonus_tok)
                    emitted_before_bonus += 1

                # 3. Score the prefetch: the verified path consumed positions start_pos + i for every
                #    token emitted this pass, so those positions' router picks are the experts actually used.
                for rep in pass_reports:
                    prefetch_calls += 1
                    used_by_pos: list[list[str]] = []
                    picks = rep.pop("_router_picks", None)
                    if picks is not None:
                        for i in range(emitted_before_bonus):
                            pos = start_pos + i
                            if pos < picks.size(0):
                                used_by_pos.append([names[int(e)] for e in picks[pos].tolist()])
                    prefetched = set(rep["prefetched_experts"])
                    hits = sum(1 for row in used_by_pos for e in row if e in prefetched)
                    uses = sum(len(row) for row in used_by_pos)
                    prefetch_hits += hits
                    prefetch_uses += uses
                    rep.update(
                        {
                            "pass": verification_passes,
                            "used_experts_by_position": used_by_pos,
                            "hits": hits,
                            "uses": uses,
                        }
                    )
                    prefetch_log.append(rep)

                seq = torch.cat(
                    [
                        prompt_ids.to(device, dtype=torch.long).reshape(1, -1),
                        torch.tensor([generated], device=device, dtype=torch.long),
                    ],
                    dim=1,
                )

        compute_elapsed_sec = time.perf_counter() - t0
        # Account for single-pass regional ring RTT (1 RTT per verification pass instead of 1 RTT per token)
        network_elapsed_sec = verification_passes * (simulated_network_rtt_ms / 1000.0)
        total_time_sec = max(1e-4, compute_elapsed_sec + network_elapsed_sec)
        tps = round(len(generated) / total_time_sec, 2)

        mean_ping = (
            round(sum(p.ping_ms for p in self.peers) / len(self.peers), 2)
            if self.peers
            else simulated_network_rtt_ms
        )

        return SpeculativeSwarmResult(
            region_ring=self.region,
            active_ring_peers=len(self.peers),
            mean_ring_ping_ms=mean_ping,
            prompt_length=prompt_len,
            generated_tokens=generated[:max_new_tokens],
            draft_gamma=draft_gamma,
            swarm_verification_passes=verification_passes,
            mean_acceptance_rate=round(accepted_total / max(1, proposed_total), 3),
            effective_tokens_per_sec=tps,
            latency_reduction_factor=round(len(generated[:max_new_tokens]) / max(1, verification_passes), 2),
            expert_routing_share=moe_target_model.get_expert_utilization(),
            expert_prefetch_enabled=prefetch_on,
            expert_prefetch_calls=prefetch_calls,
            expert_prefetch_hits=prefetch_hits,
            expert_prefetch_expert_uses=prefetch_uses,
            expert_prefetch_hit_rate=(
                round(prefetch_hits / prefetch_uses, 4) if prefetch_on and prefetch_uses > 0 else None
            ),
            expert_prefetch_log=prefetch_log,
            expert_prefetch_basis=prefetch_basis,
        )

    @staticmethod
    def _make_prefetch_hook(
        streamer: "PinnedRAMExpertDMAStreamer",
        next_router: nn.Module,
        names: list[str],
        next_layer: int,
        nodes: list[tuple[int, ...]],
        branch_probs: list[float],
        tree_positions: list[int],
        max_prefetch_experts: int,
        sink: list[dict[str, Any]],
    ):
        """Pre-hook on decoder layer `next_layer`: its input is the residual stream leaving layer
        `next_layer - 1`, the earliest point the next MoE layer's demand can be predicted, so the
        prefetch is issued here, before that layer's attention runs."""

        def hook(_module: nn.Module, args: tuple, kwargs: dict) -> None:
            hs = args[0] if args else kwargs["hidden_states"]
            if hs.dim() != 3 or hs.size(0) != 1:
                raise ValueError("expert prefetch expects a single-sequence [1, T, d] residual stream")
            branches = [hs[:, pos, :] for pos in tree_positions if pos < hs.size(1)]
            probs = [p for pos, p in zip(tree_positions, branch_probs) if pos < hs.size(1)]
            report = streamer.prefetch_token_tree_lookahead(
                next_router, branches, probs, names,
                layer_idx=next_layer - 1, max_prefetch_experts=max_prefetch_experts,
            )
            sink.append(
                {
                    "prefetched_layer": next_layer,
                    "branch_prefixes": [list(n) for n in nodes],
                    "branch_probabilities": list(branch_probs),
                    "prefetched_experts": list(report["prefetched_experts"]),
                    "streamer_report": report,
                }
            )

        return hook

    @staticmethod
    def _make_router_capture_hook(sink: list[dict[str, Any]], next_layer: int):
        """Forward hook on the next layer's router: records the experts it really selected."""

        def hook(_module: nn.Module, _inputs: tuple, output: tuple) -> None:
            for rep in sink:
                if rep.get("prefetched_layer") == next_layer and "_router_picks" not in rep:
                    rep["_router_picks"] = output[1].detach().clone()
                    break

        return hook
