"""Synthetic regression LoRA/CAS mechanics retained for compatibility.

Despite this module's historical name, these helpers do not train a language
model on the supplied text or fetch the named HF datasets. That includes the
Season 3 domains and the Season 6 astrophysics "classifier" domains: both fit
seeded Gaussian regression targets, so no classification accuracy is measured.
Phase S-2 runtime checks (`SwarmTrainingAndStoreGuard`) still validate requests,
dataset shards and LoRA tensors. Use ``hivemind.data.token_training_eval`` for
the separate measured token smoke test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from hivemind.core.btm_moe import (
    SEASON_3_DOMAIN_EXPERT_SPECS,
    RegisteredDomainExpert,
    Season3DomainExpertRegistry,
    UpcycledSparseMoEModel,
)
from hivemind.core.compression import (
    compress_lora_delta,
    compute_pseudo_gradient_delta,
    decompress_lora_delta,
)
from hivemind.security.swarm_runtime_hardening import SwarmTrainingAndStoreGuard
from hivemind.swarm.dht_storage import ContentAddressedSwarmStore

CURATED_SEASON3_DOMAIN_CORPORA: Dict[str, List[str]] = {
    "Rust_Systems_Security": [
        "pub fn increment_arc_counter(counter: &std::sync::Arc<std::sync::Mutex<u64>>, step: u64) -> Result<u64, &'static str> { let mut guard = counter.lock().map_err(|_| \"poisoned\")?; *guard += step; Ok(*guard) }",
        "pub fn zero_copy_frame_slice(buf: &[u8]) -> Result<(u32, &[u8]), &'static str> { if buf.len() < 4 { return Err(\"truncated\"); } let hdr = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]); Ok((hdr, &buf[4..])) }",
        "pub struct PinnedDmaSlot<'a> { pub buf: &'a mut [u8] } impl<'a> Drop for PinnedDmaSlot<'a> { fn drop(&mut self) { self.buf.fill(0); } }",
    ],
    "Solidity_Smart_Contracts": [
        "pragma solidity ^0.8.24; contract HiveEscrowVault { mapping(address => uint256) public balances; bool private locked; modifier nonReentrant() { require(!locked, 'REENTRANCY'); locked = true; _; locked = false; } }",
        "function settleVoucher(address provider, uint256 amount) external nonReentrant { require(balances[msg.sender] >= amount, 'INSUFFICIENT_ESCROW'); balances[msg.sender] -= amount; balances[provider] += amount; }",
        "function slashByzantineWorker(address worker, bytes32 fraudProofRoot) external { require(fraudProofRoot != bytes32(0), 'EMPTY_PROOF'); balances[worker] = 0; }",
    ],
    "Biomedical_Clinical_QA": [
        "### Clinical Question: Does mitochondrial cytochrome-c release trigger caspase-9 activation during intrinsic apoptosis? ### Evidence Conclusion (yes): Cytochrome-c binds Apaf-1 to assemble the apoptosome and activate procaspase-9.",
        "### Clinical Question: Does oxidative phosphorylation across the inner mitochondrial membrane synthesize cellular ATP via F1F0-ATP synthase? ### Evidence Conclusion (yes): Proton motive force drives rotary catalysis in complex V.",
        "### Clinical Question: Do randomized double-blind clinical trials reduce observer bias in oncology biomarker endpoints? ### Evidence Conclusion (yes): Allocation concealment and double blinding prevent systematic assessment bias.",
    ],
    "Lean4_Theorem_Prover": [
        "theorem prefix_trie_dedup_node_bound (branches gamma sharedPrefix : Nat) (h1 : 2 <= branches) (h2 : 1 <= sharedPrefix) (h3 : sharedPrefix <= gamma) : 1 + sharedPrefix + branches * (gamma - sharedPrefix) < 1 + branches * gamma := by omega",
        "theorem rfc9001_0rtt_monotonic_nonce_anti_replay (lastAcceptedNonce replayedNonce : Nat) (h : replayedNonce <= lastAcceptedNonce) : ¬ (lastAcceptedNonce < replayedNonce) := by omega",
        "theorem l2_state_channel_settlement_conservation (escrow spend : Nat) (h : spend <= escrow) : spend + (escrow - spend) = escrow := by omega",
    ],
    "Kepler_Exoplanet_Transit_Classifier": [
        "### Kepler/TESS Exoplanet Transit Observation:\nOrbital Period: 5.604 days | Planet Radius: 1.25 R_earth | Transit Depth: 225.0 ppm (SNR=20.6) | Host Star Teff: 4847 K (logg=4.58, R*=0.74 R_sun, KepMag=13.71)\n### Planet Disposition & Classification:\nDisposition: CONFIRMED | Planetary Class: Rocky Terrestrial / Earth-Size World",
        "### Kepler/TESS Exoplanet Transit Observation:\nOrbital Period: 0.639 days | Planet Radius: 162.15 R_earth | Transit Depth: 141000.0 ppm (SNR=206.3) | Host Star Teff: 6424 K (logg=4.39, R*=1.18 R_sun, KepMag=14.07)\n### Planet Disposition & Classification:\nDisposition: FALSE POSITIVE | Planetary Class: Eclipsing Binary / False Positive Transit",
        "### Kepler/TESS Exoplanet Transit Observation:\nOrbital Period: 24.644 days | Planet Radius: 1.85 R_earth | Insolation: 3.30 S_earth | Host Star Teff: 3896 K (logg=4.72, R*=0.56 R_sun, KepMag=11.57)\n### Planet Disposition & Classification:\nDisposition: CONFIRMED | Planetary Class: Super-Earth Exoplanet",
    ],
    "NASA_Exoplanet_Taxonomy_Classifier": [
        "### NASA Exoplanet Archive Observation (TOI-5806 b around TOI-5806):\nDiscovery: Transit (TESS) | Period: 3.186 d | Radius: 15.13 R_earth | Mass: 721.47 M_earth | Teq: 1621 K | Host Star Teff: 6730 K, Distance: 119.2 pc\n### Exoplanet Taxonomy Classification:\nPhysical Class: Jovian Gas Giant | Thermal Regime: Hot Inner Irradiated Orbit",
        "### NASA Exoplanet Archive Observation (TOI-938 b around TOI-938):\nDiscovery: Transit (TESS) | Period: 8.807 d | Radius: 4.82 R_earth | Host Star Teff: 5829 K, R*=0.99 R_sun, Distance: 215.0 pc\n### Exoplanet Taxonomy Classification:\nPhysical Class: Sub-Neptune / Ice Giant | Thermal Regime: Hot Inner Irradiated Orbit",
        "### NASA Exoplanet Archive Observation (HIP 39330 b around HIP 39330):\nDiscovery: Radial Velocity | Period: 4650.0 d | Semi-Major Axis: 5.05 AU | Mass: 530.77 M_earth | Host Star Teff: 5748 K, Distance: 119.2 pc\n### Exoplanet Taxonomy Classification:\nPhysical Class: Jovian Gas Giant | Thermal Regime: Cold Outer Giant Orbit",
    ],
    "SDSS_Stellar_MK_Spectral_Classifier": [
        "### SDSS DR12 Stellar Spectrum (SpecID 21143, Plate 2316-580):\nCoordinates: RA=130.846, Dec=+54.661 | Teff: 6190.7 K | Surface Gravity log(g): 3.98 dex | Metallicity [Fe/H]: -0.41 dex | Radial Velocity: -38.3 km/s (SNR=43.4)\n### Stellar MK Classification:\nSpectral Type: F-Type Yellow-White Star | Luminosity Class: Class IV (Subgiant Turnoff) | Galactic Population: Thin-Disk Population I",
        "### SDSS DR12 Stellar Spectrum (SpecID 4346, Plate 1039-125):\nCoordinates: RA=195.870, Dec=+53.365 | Teff: 5382.4 K | Surface Gravity log(g): 4.27 dex | Metallicity [Fe/H]: -1.60 dex | Radial Velocity: -155.8 km/s (SNR=24.8)\n### Stellar MK Classification:\nSpectral Type: G-Type Solar-Analog Star | Luminosity Class: Class V (Main-Sequence Dwarf) | Galactic Population: Galactic Halo Metal-Poor Population II",
        "### SDSS DR12 Stellar Spectrum (SpecID 9019, Plate 1507-38):\nCoordinates: RA=34.897, Dec=-0.256 | Teff: 6168.2 K | Surface Gravity log(g): 3.92 dex | Metallicity [Fe/H]: -1.38 dex | Radial Velocity: -265.4 km/s (SNR=34.3)\n### Stellar MK Classification:\nSpectral Type: F-Type Yellow-White Star | Luminosity Class: Class IV (Subgiant Turnoff) | Galactic Population: Galactic Halo Metal-Poor Population II",
    ],
    "JWST_Atmosphere_Biosignature_Classifier": [
        "### JWST Exoplanet Transmission Spectroscopy (EXO-8821 around G-Type Star, 45.2 ly):\nTransit Depth: 1250 ppm | log(H2O): -4.5 | log(CH4): -6.2 | log(CO2): -5.1 | Cloud Deck: 0.10 bar | Teq: 285 K | Habitability Index: 0.82\n### Atmospheric Biosignature Classification:\nHabitability Class: Class-I High-Habitability Temperate World | Biosignature Status: Strong H2O + CH4/CO2 Chemical Disequilibrium Candidate",
        "### JWST Exoplanet Transmission Spectroscopy (EXO-4410 around M-Dwarf Star, 18.4 ly):\nTransit Depth: 3100 ppm | log(H2O): -5.8 | log(CH4): -8.1 | log(CO2): -3.9 | Cloud Deck: 0.85 bar | Teq: 495 K | Habitability Index: 0.21\n### Atmospheric Biosignature Classification:\nHabitability Class: Class-III Sterile Non-Habitable Atmosphere | Biosignature Status: CO2-Dominated Venusian Runaway Greenhouse",
        "### JWST Exoplanet Transmission Spectroscopy (EXO-6109 around K-Type Star, 31.8 ly):\nTransit Depth: 1680 ppm | log(H2O): -4.9 | log(CH4): -6.8 | log(CO2): -4.8 | Cloud Deck: 0.30 bar | Teq: 258 K | Habitability Index: 0.64\n### Atmospheric Biosignature Classification:\nHabitability Class: Class-II Marginal Sub-Temperate Atmosphere | Biosignature Status: Moderate H2O/CO2 Absorption in Cold Outer Habitable Zone",
    ],
}

ASTROPHYSICS_DOMAIN_EXPERT_SPECS: Dict[str, Dict[str, Any]] = {
    "Kepler_Exoplanet_Transit_Classifier": {
        "dataset_id": "mzrk36/exoplanets-unified-dataset",
        "lora_rank_r": 16,
        "lora_alpha": 32,
    },
    "NASA_Exoplanet_Taxonomy_Classifier": {
        "dataset_id": "juliensimon/nasa-exoplanets",
        "lora_rank_r": 16,
        "lora_alpha": 32,
    },
    "SDSS_Stellar_MK_Spectral_Classifier": {
        "dataset_id": "BrunoBarreto/sdss_dr12_stars_regression",
        "lora_rank_r": 16,
        "lora_alpha": 32,
    },
    "JWST_Atmosphere_Biosignature_Classifier": {
        "dataset_id": "Shoriful025/exoplanet_atmospheric_spectroscopy_results",
        "lora_rank_r": 16,
        "lora_alpha": 32,
    },
}


def _encode_synthetic_regression_batch(
    samples: Sequence[str],
    in_features: int,
    out_features: int,
    seed: int,
    is_holdout: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate seeded Gaussian regression tensors; text content is NOT consumed.

    Only the sample count affects the batch size. This is a synthetic fixture,
    not an encoding, semantic representation or tokenized corpus.
    """
    import math

    gen_manifold = torch.Generator().manual_seed(seed)
    manifold_dim = min(4, in_features, out_features)
    basis_u = torch.randn(
        (manifold_dim, in_features), generator=gen_manifold, dtype=torch.float32
    ) / math.sqrt(float(max(1, manifold_dim)))
    target_w = (
        torch.randn((out_features, in_features), generator=gen_manifold, dtype=torch.float32) * 0.22
    )

    sample_seed = seed + (9999 if is_holdout else 0)
    gen_samples = torch.Generator().manual_seed(sample_seed)
    batch_size = max(12, len(samples) * 4)
    z = torch.randn((batch_size, manifold_dim), generator=gen_samples, dtype=torch.float32)
    x = z @ basis_u
    y = x @ target_w.T
    return x, y


# Historical private alias; retained for callers, never a text encoder.
_encode_corpus_batch = _encode_synthetic_regression_batch


def train_single_domain_lora_adapter(
    domain_name: str,
    reference_lora_state: Dict[str, torch.Tensor],
    lora_rank_r: int = 16,
    lora_alpha: int = 32,
    inner_steps: int = 12,
    lr: float = 0.11,
    topk_ratio: float = 0.05,
    custom_corpus: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Fit a synthetic regression adapter matching the reference parameter shapes.

    Names and signatures are retained for game/demo callers. No domain-language
    capability is measured. Phase S-2 runtime GPU and dataset checks still run.
    The returned state blends raw and reconstructed weights; it is NOT a
    compression-only reconstruction.
    """
    import hashlib

    guard = SwarmTrainingAndStoreGuard()
    spec = SEASON_3_DOMAIN_EXPERT_SPECS.get(
        domain_name,
        ASTROPHYSICS_DOMAIN_EXPERT_SPECS.get(
            domain_name,
            {
                "dataset_id": f"hivemind/{domain_name.lower()}",
                "lora_rank_r": lora_rank_r,
                "lora_alpha": lora_alpha,
            },
        ),
    )
    corpus = list(
        custom_corpus
        if custom_corpus is not None
        else CURATED_SEASON3_DOMAIN_CORPORA.get(
            domain_name,
            CURATED_SEASON3_DOMAIN_CORPORA["Rust_Systems_Security"],
        )
    )

    # Phase S-2 Hardening Gate: Verify GPU training scope, dataset shard anti-poisoning, and input tensors
    guard.validate_gpu_training_request(
        domain_name=domain_name,
        dataset_id=str(spec["dataset_id"]),
        lora_rank_r=lora_rank_r,
        lora_alpha=lora_alpha,
        inner_steps=inner_steps,
        lr=lr,
        topk_ratio=topk_ratio,
        raise_on_violation=True,
    )
    guard.validate_dataset_shard(
        domain_name=domain_name,
        samples=corpus,
        raise_on_violation=True,
    )
    guard.validate_lora_tensors(
        lora_state_dict=reference_lora_state,
        domain_name=f"{domain_name}:init",
        raise_on_violation=True,
    )

    domain_seed = int(hashlib.sha256(domain_name.encode("utf-8")).hexdigest()[:8], 16) % 10_000 + 101

    prefix_groups: Dict[str, Dict[str, torch.Tensor]] = {}
    for k, v in reference_lora_state.items():
        if "lora_A" in k:
            prefix = k.split("lora_A")[0]
            prefix_groups.setdefault(prefix, {})["A"] = v.detach().clone().to(torch.float32)
            prefix_groups[prefix]["key_A"] = k  # type: ignore[assignment]
        elif "lora_B" in k:
            prefix = k.split("lora_B")[0]
            prefix_groups.setdefault(prefix, {})["B"] = v.detach().clone().to(torch.float32)
            prefix_groups[prefix]["key_B"] = k  # type: ignore[assignment]

    init_state = {k: v.detach().clone().to(torch.float32) for k, v in reference_lora_state.items()}
    trained_state: Dict[str, torch.Tensor] = {}

    initial_losses: List[float] = []
    final_losses: List[float] = []
    scaling = float(lora_alpha) / float(max(1, lora_rank_r))

    for idx, (prefix, grp) in enumerate(sorted(prefix_groups.items())):
        a_init = grp["A"]
        b_init = grp["B"]
        r_dim, in_dim = int(a_init.shape[0]), int(a_init.shape[1])
        out_dim = int(b_init.shape[0])

        x, target_y = _encode_synthetic_regression_batch(corpus, in_dim, out_dim, seed=domain_seed + idx * 17)
        lora_a = nn.Parameter(a_init.clone())
        lora_b = nn.Parameter(b_init.clone())
        if torch.allclose(lora_b.data, torch.zeros_like(lora_b.data)):
            gen = torch.Generator().manual_seed(domain_seed + idx)
            lora_a.data.copy_(
                torch.randn(
                    lora_a.data.shape,
                    generator=gen,
                    dtype=lora_a.data.dtype,
                    device=lora_a.data.device,
                )
                * 0.05
            )

        opt = torch.optim.SGD([lora_a, lora_b], lr=lr, momentum=0.85)
        for step in range(max(2, inner_steps)):
            opt.zero_grad()
            pred = (x @ lora_a.T @ lora_b.T) * scaling
            loss = torch.mean((pred - target_y) ** 2)
            if step == 0:
                initial_losses.append(float(loss.item()))
            loss.backward()
            opt.step()
        with torch.no_grad():
            final_pred = (x @ lora_a.T @ lora_b.T) * scaling
            final_losses.append(float(torch.mean((final_pred - target_y) ** 2).item()))

        trained_state[str(grp["key_A"])] = lora_a.detach().clone()
        trained_state[str(grp["key_B"])] = lora_b.detach().clone()

    for k, v in reference_lora_state.items():
        if k not in trained_state:
            # Only matched adapter pairs were optimized; preserve other reference tensors.
            trained_state[k] = v.detach().clone().to(torch.float32)

    raw_delta = compute_pseudo_gradient_delta(init_state, trained_state)
    delta_pkg = compress_lora_delta(
        delta_dict=raw_delta,
        worker_id=f"s3-trainer-{domain_name.lower()}",
        branch_name=domain_name,
        outer_step=1,
        inner_steps_completed=max(2, inner_steps),
        topk_ratio=topk_ratio,
    )
    reconstructed_delta = decompress_lora_delta(delta_pkg, device=torch.device("cpu"), dtype=torch.float32)
    converged_state = {
        k: (
            0.92 * trained_state[k]
            + 0.08 * (init_state[k] - reconstructed_delta.get(k, torch.zeros_like(init_state[k])))
        ).to(torch.float16)
        for k in init_state
    }

    guard.validate_lora_tensors(
        lora_state_dict=converged_state,
        domain_name=f"{domain_name}:converged",
        raise_on_violation=True,
    )

    init_loss_mean = round(sum(initial_losses) / max(1, len(initial_losses)), 6)
    final_loss_mean = round(sum(final_losses) / max(1, len(final_losses)), 6)
    metrics = {
        "domain_name": domain_name,
        "measurement_kind": "synthetic_regression",
        "loss_metric": "mean_squared_error",
        "text_content_used": False,
        "dataset_id": "synthetic/seeded-gaussian-regression-v1",
        "requested_dataset_id_not_loaded": spec["dataset_id"],
        "returned_state_provenance": "0.92 raw-trained + 0.08 reconstructed; not compression-only",
        "byte_count_basis": "compression tensor accounting; not serialized wire bytes",
        "lora_rank_r": int(reference_lora_state[next(iter(reference_lora_state))].shape[0]),
        "lora_alpha": int(lora_alpha),
        "inner_steps": int(max(2, inner_steps)),
        "requested_inner_steps": int(inner_steps),
        "initial_loss": init_loss_mean,
        "final_loss": final_loss_mean,
        "loss_decreased": final_loss_mean < init_loss_mean,
        "topk_ratio": float(topk_ratio),
        "compression_ratio": round(delta_pkg.compression_ratio, 2),
        "compressed_bytes": int(delta_pkg.compressed_bytes),
        "uncompressed_bytes": int(delta_pkg.uncompressed_bytes),
    }
    return converged_state, metrics


def train_and_publish_season3_domain_experts(
    store: ContentAddressedSwarmStore,
    reference_lora_state: Dict[str, torch.Tensor],
    moe_model: Optional[UpcycledSparseMoEModel] = None,
    domains: Optional[Sequence[str]] = None,
    topk_ratio: float = 0.05,
    inner_steps: int = 12,
) -> Tuple[Season3DomainExpertRegistry, Dict[str, Any]]:
    """
    Run synthetic regression mechanics for named demo domains and store artifacts.

    Text shards are stored but not used to construct training inputs/targets.
    The optional hot-plug exercises model wiring, not domain capability.
    """
    target_domains = list(domains) if domains is not None else list(SEASON_3_DOMAIN_EXPERT_SPECS.keys())
    registry = Season3DomainExpertRegistry()
    receipts: Dict[str, Dict[str, Any]] = {}

    for idx, domain_name in enumerate(target_domains):
        corpus = CURATED_SEASON3_DOMAIN_CORPORA.get(
            domain_name,
            CURATED_SEASON3_DOMAIN_CORPORA["Rust_Systems_Security"],
        )
        shard_cid, _, _ = store.store_hf_dataset_shard(
            node_id=f"s3-expert-node-{idx + 1:02d}",
            domain_name=domain_name,
            shard_id=idx,
            samples=corpus,
        )
        lora_state, metrics = train_single_domain_lora_adapter(
            domain_name=domain_name,
            reference_lora_state=reference_lora_state,
            inner_steps=inner_steps,
            topk_ratio=topk_ratio,
        )
        expert_cid, checkpoint_path = store.publish_lora_expert_checkpoint(
            domain_name=domain_name,
            lora_state_dict=lora_state,
            metadata=metrics,
        )
        reg_expert = RegisteredDomainExpert(
            domain_name=domain_name,
            dataset_id=str(metrics["dataset_id"]),
            expert_cid=expert_cid,
            shard_cid=shard_cid,
            lora_rank_r=int(metrics["lora_rank_r"]),
            lora_alpha=int(metrics["lora_alpha"]),
            initial_loss=float(metrics["initial_loss"]),
            final_loss=float(metrics["final_loss"]),
            compression_ratio=float(metrics["compression_ratio"]),
            compressed_bytes=int(metrics["compressed_bytes"]),
            lora_state_dict=lora_state,
        )
        registry.register_expert(reg_expert)
        receipts[domain_name] = {
            **metrics,
            "shard_cid": shard_cid,
            "expert_cid": expert_cid,
            "checkpoint_path": str(checkpoint_path),
        }

    hotplug_summary = None
    if moe_model is not None:
        hotplug_summary = registry.hot_plug_all_into_moe(moe_model)

    return registry, {
        "measurement_kind": "synthetic_regression",
        "num_domains_trained": len(receipts),
        "domains": list(receipts.keys()),
        "receipts": receipts,
        "hotplug_summary": hotplug_summary,
    }


def train_and_publish_astrophysics_domain_experts(
    store: ContentAddressedSwarmStore,
    reference_lora_state: Dict[str, torch.Tensor],
    moe_model: Optional[UpcycledSparseMoEModel] = None,
    domains: Optional[Sequence[str]] = None,
    topk_ratio: float = 0.05,
    inner_steps: int = 12,
) -> Tuple[Season3DomainExpertRegistry, Dict[str, Any]]:
    """
    Run the synthetic regression mechanics for the named astrophysics demo domains.

    The domain text shards are stored but not used to build inputs/targets, so no
    planet or star classification accuracy is measured. The optional hot-plug
    exercises model wiring, not domain capability.
    """
    target_domains = (
        list(domains) if domains is not None else list(ASTROPHYSICS_DOMAIN_EXPERT_SPECS.keys())
    )
    return train_and_publish_season3_domain_experts(
        store=store,
        reference_lora_state=reference_lora_state,
        moe_model=moe_model,
        domains=target_domains,
        topk_ratio=topk_ratio,
        inner_steps=inner_steps,
    )
