"""La Ruche (`HiveMind Ruche`) — Community Beehive MoE Model Vault & Post-QLoRA Hub (`v2.6.0`).

Provides a content-addressed, cryptographically verified Beehive (`La Ruche`) storage
and model-serving vault where trained QLoRA / LoRA domain experts from community GPU
contributors are validated, stored in hexagonal Honeycomb Cells (`Alveoles`), merged into
an upcycled Branch-Train-Mix (BTM) Sparse Mixture-of-Experts (`UpcycledSparseMoEModel`),
and published as a ready-to-use community model bundle (`safetensors` + Hugging Face
Model Card + 1-line Python loader).

Charter Invariant:
    Strictly non-monetary open-source scientific project (`0.00 EUR / 0.00 USD`). All
    compute reciprocity credits (`PTS` / `$HIVE`) are non-financial accounting units for
    reciprocal swarm inference priority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save

from hivemind.core.btm_moe import (
    hotplug_three_tier_experts,
    project_lora_rank_svd_with_report,
)
from hivemind.core.btm_upcycle import (
    UpcycledSparseMoEModel,
    upcycle_branches_to_sparse_moe,
)
from hivemind.core.quantization import (
    attach_qlora_adapters,
    build_local_qwen2_backbone,
)
from hivemind.swarm.dht_storage import ContentAddressedSwarmStore


# Axial hexagonal coordinates (q, r) for the 8 specialist Honeycomb Cells (Alveoles)
# arranged around the central Queen/Router core (0, 0) in Conway-Sloane A2 lattice geometry.
RUCHE_ALVEOLE_HEX_LAYOUT: Tuple[Dict[str, Any], ...] = (
    {
        "cell_id": "ALVEOLE-01-CODE-PYTHON",
        "domain_name": "code_python",
        "display_title": "Python & Numeric Kernels",
        "hex_q": 1,
        "hex_r": 0,
        "accent_hex": "#38bdf8",
    },
    {
        "cell_id": "ALVEOLE-02-MATH-REASONING",
        "domain_name": "math_reasoning",
        "display_title": "Algebraic & Symbolic Math",
        "hex_q": 1,
        "hex_r": -1,
        "accent_hex": "#f59e0b",
    },
    {
        "cell_id": "ALVEOLE-03-PHYSICS-SIM",
        "domain_name": "physics_sim",
        "display_title": "Computational Physics & Astro",
        "hex_q": 0,
        "hex_r": -1,
        "accent_hex": "#a855f7",
    },
    {
        "cell_id": "ALVEOLE-04-SYSTEMS-RUST",
        "domain_name": "systems_rust",
        "display_title": "Rust Systems & Zero-Copy QUIC",
        "hex_q": -1,
        "hex_r": 0,
        "accent_hex": "#f97316",
    },
    {
        "cell_id": "ALVEOLE-05-CYBER-SEC",
        "domain_name": "cyber_sec",
        "display_title": "Byzantine & Zero-Trust Security",
        "hex_q": -1,
        "hex_r": 1,
        "accent_hex": "#ef4444",
    },
    {
        "cell_id": "ALVEOLE-06-BIO-MED",
        "domain_name": "bio_med",
        "display_title": "Clinical Genomics & Proteomics",
        "hex_q": 0,
        "hex_r": 1,
        "accent_hex": "#10b981",
    },
    {
        "cell_id": "ALVEOLE-07-GENERAL-CHAT",
        "domain_name": "general_chat",
        "display_title": "Multilingual Scientific Synthesis",
        "hex_q": 2,
        "hex_r": -1,
        "accent_hex": "#ec4899",
    },
    {
        "cell_id": "ALVEOLE-08-FORMAL-LEAN4",
        "domain_name": "formal_lean4",
        "display_title": "Lean 4 Formal Theorem Proving",
        "hex_q": -2,
        "hex_r": 1,
        "accent_hex": "#eab308",
    },
)


@dataclass
class RucheHoneycombCell:
    """Metadata and cryptographic receipt for one Honeycomb Cell (`Alveole`) in La Ruche."""

    cell_id: str
    domain_name: str
    display_title: str
    hex_q: int
    hex_r: int
    accent_hex: str
    contributor_handle: str
    hardware_device: str
    dataset_shard_cid: str
    expert_cas_cid: str
    safetensors_rel_path: str
    safetensors_sha256: str
    safetensors_bytes: int
    source_rank_r: int
    canonical_rank_r: int
    spectral_energy_retained: float
    train_nll_before: float
    train_nll_after: float
    holdout_nll_before: float
    holdout_nll_after: float
    holdout_perplexity_before: float
    holdout_perplexity_after: float
    perplexity_reduction_pct: float
    wire_codec: str
    deposited_at_epoch: float
    verified: bool = True


@dataclass
class RucheCommunityMoERelease:
    """Manifest for a merged Community BTM Sparse MoE model exported from La Ruche."""

    release_id: str
    release_version: str
    base_architecture: str
    num_domains: int
    top_k_active: int
    canonical_rank_r: int
    hidden_size: int
    vocab_size: int
    router_type: str
    three_tier_enabled: bool
    ortho_inj_enabled: bool
    combined_safetensors_path: str
    combined_safetensors_sha256: str
    combined_safetensors_bytes: int
    model_card_path: str
    manifest_path: str
    mean_holdout_perplexity_before: float
    mean_holdout_perplexity_after: float
    mean_perplexity_reduction_pct: float
    contributors: List[str]
    cells: List[RucheHoneycombCell]
    created_at_epoch: float
    charter_notice: str = (
        "Strictly non-monetary open-source project (0.00 EUR / 0.00 USD). "
        "All experts and MoE weights are openly shared under reciprocal community compute."
    )


def _serialize_tensors_safetensors(tensors: Dict[str, torch.Tensor]) -> bytes:
    clean = {
        str(k): v.detach().cpu().to(dtype=torch.float32).contiguous().clone()
        for k, v in sorted(tensors.items())
    }
    return safetensors_save(clean)


def _verify_and_measure_adapter_svd(
    adapter_tensors: Dict[str, torch.Tensor],
    canonical_rank_r: int = 4,
) -> Tuple[int, float]:
    """Validate finite non-zero adapter tensors and compute SVD spectral energy retention."""
    if not adapter_tensors:
        raise ValueError("adapter_tensors cannot be empty")

    total_norm = 0.0
    for name, tensor in adapter_tensors.items():
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Non-finite values detected in adapter tensor '{name}'")
        total_norm += float(torch.norm(tensor.float()).item())

    if total_norm <= 1e-9:
        raise ValueError("Rejected silent/zero LoRA adapter in La Ruche gatekeeper")

    a_keys = [k for k in sorted(adapter_tensors.keys()) if "lora_A" in k]
    b_keys = [k for k in sorted(adapter_tensors.keys()) if "lora_B" in k]
    if a_keys and b_keys:
        raw_a = adapter_tensors[a_keys[0]].detach().cpu().float()
        source_rank = int(raw_a.shape[0])
        _, svd_report = project_lora_rank_svd_with_report(
            adapter_tensors,
            target_rank_r=min(canonical_rank_r, source_rank),
            source_lora_alpha=float(2 * source_rank),
            target_lora_alpha=float(2 * min(canonical_rank_r, source_rank)),
        )
        return source_rank, float(svd_report.mean_spectral_energy_retained)

    return canonical_rank_r, 1.0


class LaRucheModelVault:
    """Content-Addressed Beehive (`La Ruche`) Vault for QLoRA Experts and Community BTM MoE Models."""

    def __init__(
        self,
        vault_dir: Path | str,
        hidden_size: int = 32,
        vocab_size: int = 259,
        canonical_rank_r: int = 4,
    ) -> None:
        self.vault_dir = Path(vault_dir)
        self.cells_dir = self.vault_dir / "cells"
        self.release_dir = self.vault_dir / "community_moe_release"
        self.cas_store = ContentAddressedSwarmStore(self.vault_dir / "cas_dht")
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.canonical_rank_r = int(canonical_rank_r)

        self.cells_dir.mkdir(parents=True, exist_ok=True)
        self.release_dir.mkdir(parents=True, exist_ok=True)

        self._layout_by_domain: Dict[str, Dict[str, Any]] = {
            item["domain_name"]: dict(item) for item in RUCHE_ALVEOLE_HEX_LAYOUT
        }
        self.cells: Dict[str, RucheHoneycombCell] = {}
        self.cell_tensors: Dict[str, Dict[str, torch.Tensor]] = {}

    def deposit_qlora_expert(
        self,
        domain_name: str,
        adapter_tensors: Dict[str, torch.Tensor],
        contributor_handle: str,
        hardware_device: str,
        dataset_shard_cid: str,
        train_nll_before: float,
        train_nll_after: float,
        holdout_nll_before: float,
        holdout_nll_after: float,
        holdout_perplexity_before: float,
        holdout_perplexity_after: float,
        wire_codec: str = "e8_2bit",
    ) -> RucheHoneycombCell:
        """Verify and deposit a trained QLoRA domain expert into its Honeycomb Cell (`Alveole`)."""
        if domain_name not in self._layout_by_domain:
            raise ValueError(
                f"Unknown La Ruche domain '{domain_name}'. Expected one of {sorted(self._layout_by_domain)}"
            )
        if holdout_nll_after >= holdout_nll_before:
            raise ValueError(
                f"La Ruche quality gate rejected expert for '{domain_name}': "
                f"holdout_nll_after ({holdout_nll_after:.4f}) >= holdout_nll_before ({holdout_nll_before:.4f})"
            )

        source_rank, spectral_retained = _verify_and_measure_adapter_svd(
            adapter_tensors=adapter_tensors,
            canonical_rank_r=self.canonical_rank_r,
        )

        clean_state: Dict[str, torch.Tensor] = {
            str(k): v.detach().cpu().float().contiguous()
            for k, v in sorted(adapter_tensors.items())
        }
        raw_bytes = _serialize_tensors_safetensors(clean_state)
        file_sha256 = hashlib.sha256(raw_bytes).hexdigest()

        expert_cid, _ = self.cas_store.publish_lora_expert_checkpoint(
            domain_name=domain_name,
            lora_state_dict=clean_state,
            metadata={
                "contributor_handle": contributor_handle,
                "dataset_shard_cid": dataset_shard_cid,
                "safetensors_sha256": file_sha256,
            },
        )

        domain_dir = self.cells_dir / domain_name
        domain_dir.mkdir(parents=True, exist_ok=True)
        safetensors_filename = f"alveole_{domain_name}_{file_sha256[:12]}.safetensors"
        safetensors_path = domain_dir / safetensors_filename
        safetensors_path.write_bytes(raw_bytes)

        ppl_reduction_pct = (
            100.0
            * (holdout_perplexity_before - holdout_perplexity_after)
            / max(1e-9, holdout_perplexity_before)
        )
        layout = self._layout_by_domain[domain_name]

        cell = RucheHoneycombCell(
            cell_id=str(layout["cell_id"]),
            domain_name=domain_name,
            display_title=str(layout["display_title"]),
            hex_q=int(layout["hex_q"]),
            hex_r=int(layout["hex_r"]),
            accent_hex=str(layout["accent_hex"]),
            contributor_handle=contributor_handle,
            hardware_device=hardware_device,
            dataset_shard_cid=dataset_shard_cid,
            expert_cas_cid=expert_cid,
            safetensors_rel_path=str(safetensors_path.relative_to(self.vault_dir)),
            safetensors_sha256=file_sha256,
            safetensors_bytes=len(raw_bytes),
            source_rank_r=source_rank,
            canonical_rank_r=self.canonical_rank_r,
            spectral_energy_retained=round(spectral_retained, 6),
            train_nll_before=round(float(train_nll_before), 6),
            train_nll_after=round(float(train_nll_after), 6),
            holdout_nll_before=round(float(holdout_nll_before), 6),
            holdout_nll_after=round(float(holdout_nll_after), 6),
            holdout_perplexity_before=round(float(holdout_perplexity_before), 4),
            holdout_perplexity_after=round(float(holdout_perplexity_after), 4),
            perplexity_reduction_pct=round(float(ppl_reduction_pct), 2),
            wire_codec=wire_codec,
            deposited_at_epoch=time.time(),
            verified=True,
        )
        self.cells[domain_name] = cell
        self.cell_tensors[domain_name] = clean_state
        return cell

    def assemble_community_btm_moe(
        self,
        release_version: str = "v2.6.0",
        top_k: int = 2,
        seed: int = 20261024,
    ) -> Tuple[UpcycledSparseMoEModel, RucheCommunityMoERelease]:
        """Merge all deposited Honeycomb Cells into a Community BTM Sparse MoE Model & export bundle."""
        if not self.cells:
            raise ValueError("Cannot assemble Community MoE: La Ruche has no deposited cells")

        ordered_domains = [
            item["domain_name"]
            for item in RUCHE_ALVEOLE_HEX_LAYOUT
            if item["domain_name"] in self.cells
        ]

        base_model, _, _ = build_local_qwen2_backbone(
            seed=seed,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.hidden_size * 2,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            device=torch.device("cpu"),
            use_4bit_nf4_if_cuda=False,
        )
        peft_bundle = attach_qlora_adapters(
            base_model=base_model,
            rank_r=self.canonical_rank_r,
            lora_alpha=self.canonical_rank_r * 2,
            gradient_checkpointing=False,
            seed=seed,
        )

        branch_lora_states: Dict[str, Dict[str, torch.Tensor]] = {
            dom: self.cell_tensors[dom] for dom in ordered_domains
        }
        moe_model, _ = upcycle_branches_to_sparse_moe(
            peft_bundle.model,
            branch_lora_states,
            lora_rank_r=self.canonical_rank_r,
            lora_alpha=self.canonical_rank_r * 2,
            in_place=False,
        )
        moe_model.set_routing_mode("sigma_moe")
        moe_model.set_peri_ln(True)
        moe_model.moe_blocks = moe_model.get_moe_blocks()  # type: ignore[attr-defined]
        moe_model.hidden_size = self.hidden_size  # type: ignore[attr-defined]
        hotplug_three_tier_experts(
            moe_model,
            {},
            canonical_rank_r=self.canonical_rank_r,
            num_super_clusters=1,
            macro_per_super=1,
        )

        # Save all domain adapter states + MoE router states into the combined safetensors release
        combined_tensors: Dict[str, torch.Tensor] = {}
        for dom in ordered_domains:
            for k, v in self.cell_tensors[dom].items():
                combined_tensors[f"branch.{dom}.{k}"] = v.detach().cpu().float().contiguous()
        for k, v in moe_model.state_dict().items():
            if v.is_floating_point():
                combined_tensors[f"moe.{k}"] = v.detach().cpu().float().contiguous()

        combined_bytes = _serialize_tensors_safetensors(combined_tensors)
        combined_sha256 = hashlib.sha256(combined_bytes).hexdigest()
        combined_path = self.release_dir / "hivemind_la_ruche_community_moe.safetensors"
        combined_path.write_bytes(combined_bytes)

        ordered_cells = [self.cells[d] for d in ordered_domains]
        mean_ppl_before = sum(c.holdout_perplexity_before for c in ordered_cells) / len(ordered_cells)
        mean_ppl_after = sum(c.holdout_perplexity_after for c in ordered_cells) / len(ordered_cells)
        mean_ppl_red = sum(c.perplexity_reduction_pct for c in ordered_cells) / len(ordered_cells)
        contributors = sorted({c.contributor_handle for c in ordered_cells})

        release_id = f"RUCHE-MOE-{release_version.upper().replace('.', '_')}-{combined_sha256[:8]}"
        manifest_path = self.release_dir / "ruche_manifest.json"
        model_card_path = self.release_dir / "README.md"

        release = RucheCommunityMoERelease(
            release_id=release_id,
            release_version=release_version,
            base_architecture="HiveMind-BTM-UpcycledSparseMoE-8xQLoRA",
            num_domains=len(ordered_cells),
            top_k_active=min(top_k, len(ordered_cells)),
            canonical_rank_r=self.canonical_rank_r,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            router_type="SigmaMoE-PeriLN + ThreeTierHotColdHierarchy",
            three_tier_enabled=True,
            ortho_inj_enabled=True,
            combined_safetensors_path=str(combined_path.relative_to(self.vault_dir)),
            combined_safetensors_sha256=combined_sha256,
            combined_safetensors_bytes=len(combined_bytes),
            model_card_path=str(model_card_path.relative_to(self.vault_dir)),
            manifest_path=str(manifest_path.relative_to(self.vault_dir)),
            mean_holdout_perplexity_before=round(mean_ppl_before, 4),
            mean_holdout_perplexity_after=round(mean_ppl_after, 4),
            mean_perplexity_reduction_pct=round(mean_ppl_red, 2),
            contributors=contributors,
            cells=ordered_cells,
            created_at_epoch=time.time(),
        )

        manifest_path.write_text(
            json.dumps(asdict(release), indent=2) + "\n", encoding="utf-8"
        )
        model_card_path.write_text(
            self._render_community_model_card(release), encoding="utf-8"
        )
        return moe_model, release

    def _render_community_model_card(self, release: RucheCommunityMoERelease) -> str:
        rows = []
        for cell in release.cells:
            rows.append(
                f"| `{cell.cell_id}` | `{cell.domain_name}` | `{cell.contributor_handle}` | "
                f"`{cell.hardware_device}` | `{cell.holdout_perplexity_before:.2f} -> {cell.holdout_perplexity_after:.2f}` "
                f"(`-{cell.perplexity_reduction_pct:.2f}%`) | `{cell.wire_codec}` | `{cell.safetensors_sha256[:12]}...` |"
            )
        table_md = "\n".join(rows)
        return (
            "---\n"
            "library_name: hivemind-la-ruche\n"
            "license: apache-2.0\n"
            "tags:\n"
            "  - hivemind\n"
            "  - mixture-of-experts\n"
            "  - qlora\n"
            "  - diloco\n"
            "  - la-ruche\n"
            "---\n\n"
            f"# 🐝 La Ruche (`{release.release_id}`) — Community BTM Sparse MoE Model Vault\n\n"
            f"- **Architecture:** `{release.base_architecture}` (`{release.num_domains}` Honeycomb Cells, Top-`{release.top_k_active}` active)\n"
            f"- **Router & Depth Core:** `{release.router_type}` + `OrthogonalInputInjector`\n"
            f"- **Combined Safetensors SHA-256:** `{release.combined_safetensors_sha256}` (`{release.combined_safetensors_bytes}` bytes)\n"
            f"- **Mean Holdout Perplexity:** `{release.mean_holdout_perplexity_before:.2f} -> {release.mean_holdout_perplexity_after:.2f}` (`-{release.mean_perplexity_reduction_pct:.2f}%` reduction)\n"
            f"- **Contributors:** {', '.join(release.contributors)}\n"
            f"- **Charter:** {release.charter_notice}\n\n"
            "## Honeycomb Cells (`Alvéoles`) Inventory\n\n"
            "| Cell ID | Domain | Contributor | Hardware | Holdout PPL (`Before -> After`) | Wire Codec | SHA-256 |\n"
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
            f"{table_md}\n\n"
            "## Quickstart: Load & Run the Community MoE Model\n\n"
            "```python\n"
            "import torch\n"
            "from hivemind.swarm.la_ruche_vault import load_community_moe_from_ruche\n\n"
            "moe_model, manifest = load_community_moe_from_ruche('artifacts/la_ruche_vault')\n"
            "tokens = torch.tensor([[2, 5, 9, 14, 21, 33, 44, 55]])\n"
            "outputs = moe_model(input_ids=tokens)\n"
            "print('Logits shape:', outputs.logits.shape)\n"
            "```\n"
        )


def load_community_moe_from_ruche(
    vault_dir: Path | str,
    seed: int = 20261024,
) -> Tuple[UpcycledSparseMoEModel, Dict[str, Any]]:
    """Load the merged Community BTM Sparse MoE model and manifest from a La Ruche vault directory."""
    vault_path = Path(vault_dir)
    manifest_path = vault_path / "community_moe_release" / "ruche_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"La Ruche manifest not found at {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    safetensors_path = vault_path / str(manifest["combined_safetensors_path"])
    raw_bytes = safetensors_path.read_bytes()
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha256 != manifest["combined_safetensors_sha256"]:
        raise ValueError(
            f"La Ruche safetensors SHA-256 mismatch: expected {manifest['combined_safetensors_sha256']}, got {actual_sha256}"
        )

    tensors = safetensors_load(raw_bytes)
    rank_r = int(manifest["canonical_rank_r"])
    top_k = int(manifest["top_k_active"])
    hidden_size = int(manifest["hidden_size"])
    vocab_size = int(manifest["vocab_size"])

    base_model, _, _ = build_local_qwen2_backbone(
        seed=seed,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        device=torch.device("cpu"),
        use_4bit_nf4_if_cuda=False,
    )
    peft_bundle = attach_qlora_adapters(
        base_model=base_model,
        rank_r=rank_r,
        lora_alpha=rank_r * 2,
        gradient_checkpointing=False,
        seed=seed,
    )

    branch_lora_states: Dict[str, Dict[str, torch.Tensor]] = {}
    for cell_info in manifest["cells"]:
        dom = str(cell_info["domain_name"])
        prefix = f"branch.{dom}."
        branch_lora_states[dom] = {
            k[len(prefix) :]: v for k, v in tensors.items() if k.startswith(prefix)
        }

    moe_model, _ = upcycle_branches_to_sparse_moe(
        peft_bundle.model,
        branch_lora_states,
        lora_rank_r=rank_r,
        lora_alpha=rank_r * 2,
        in_place=False,
    )
    moe_model.set_routing_mode("sigma_moe")
    moe_model.set_peri_ln(True)

    # Restore exact saved MoE parameters
    current_sd = moe_model.state_dict()
    for k in list(current_sd.keys()):
        saved_key = f"moe.{k}"
        if saved_key in tensors and current_sd[k].shape == tensors[saved_key].shape:
            current_sd[k] = tensors[saved_key]
    moe_model.load_state_dict(current_sd, strict=False)
    return moe_model, manifest


SOCRATEAI_GCS_LA_RUCHE_BUCKET = "gs://socrateai-hivemind-la-ruche-vault"
SOCRATEAI_GCS_DATALAKE_BUCKET = (
    "gs://socrateai-datalake-gen-lang-client-0625573011/hivemind_la_ruche_vault"
)
SOCRATEAI_GCS_PUBLIC_BASE_URL = (
    "https://storage.googleapis.com/socrateai-hivemind-la-ruche-vault"
)


def load_community_moe_from_gcs_or_https(
    cache_dir: Path | str = Path("/tmp/hivemind_la_ruche_gcs_cache"),
    gcs_public_base_url: str = SOCRATEAI_GCS_PUBLIC_BASE_URL,
    seed: int = 20261024,
) -> Tuple[UpcycledSparseMoEModel, Dict[str, Any]]:
    """Download the Community BTM Sparse MoE model from SocrateAI GCP Cloud Storage (`La Ruche`) and instantiate it."""
    import urllib.request

    cache_path = Path(cache_dir)
    release_dir = cache_path / "community_moe_release"
    release_dir.mkdir(parents=True, exist_ok=True)

    base = gcs_public_base_url.rstrip("/")
    manifest_url = f"{base}/community_moe_release/ruche_manifest.json"
    req_m = urllib.request.Request(
        manifest_url, headers={"User-Agent": "HiveMind-LaRuche-GCS-Loader/2.6.0"}
    )
    with urllib.request.urlopen(req_m, timeout=30.0) as resp_m:
        manifest_bytes = resp_m.read()

    manifest = json.loads(manifest_bytes.decode("utf-8"))
    manifest_file = release_dir / "ruche_manifest.json"
    manifest_file.write_bytes(manifest_bytes)

    rel_st_path = str(
        manifest.get(
            "combined_safetensors_path",
            "community_moe_release/hivemind_la_ruche_community_moe.safetensors",
        )
    )
    st_file = cache_path / rel_st_path
    st_file.parent.mkdir(parents=True, exist_ok=True)

    expected_sha256 = str(manifest["combined_safetensors_sha256"])
    need_download = True
    if st_file.exists():
        existing_sha256 = hashlib.sha256(st_file.read_bytes()).hexdigest()
        if existing_sha256 == expected_sha256:
            need_download = False

    if need_download:
        st_url = f"{base}/{rel_st_path}"
        req_st = urllib.request.Request(
            st_url, headers={"User-Agent": "HiveMind-LaRuche-GCS-Loader/2.6.0"}
        )
        with urllib.request.urlopen(req_st, timeout=60.0) as resp_st:
            raw_bytes = resp_st.read()
        dl_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        if dl_sha256 != expected_sha256:
            raise ValueError(
                f"SocrateAI GCS La Ruche safetensors SHA-256 mismatch: expected {expected_sha256}, got {dl_sha256}"
            )
        st_file.write_bytes(raw_bytes)

    return load_community_moe_from_ruche(cache_path, seed=seed)
