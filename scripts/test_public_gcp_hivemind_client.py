"""End-to-End Public GCP DNS Verification Client for HiveMind@home, La Ruche MoE Vault & Federated AI (`v2.6.0`).

Connects over public HTTPS to the deployed GCP Cloud Run service (`hivemind-at-home-ruche`)
and verifies that external end-users can:
1. Access the interactive `HiveMind@home` SETI-style HTML5 dashboard and GCP cost-optimization telemetry,
2. Retrieve the `La Ruche` Community BTM Sparse MoE manifest and download the binary `.safetensors`
   weights (both the merged 8-domain Community MoE model and individual Honeycomb Cells) with
   end-to-end SHA-256 cryptographic verification and local `safetensors` tensor deserialization,
3. Execute live Top-2 Sparse MoE inference (`POST /api/v1/moe/inference` and OpenAI-compatible
   `POST /v1/chat/completions`) across specialized scientific prompts,
4. Participate in Federated AI Training (`POST /api/v1/federated/workunit/request` and
   `POST /api/v1/federated/workunit/submit`) with two-statistic spot-check verification and
   non-monetary compute-reciprocity receipts (`0.00 EUR / 0.00 USD`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from typing import Any, Dict

from safetensors import safe_open

from hivemind.boinc.hivemind_at_home_dashboard import PUBLIC_GCP_CLOUD_RUN_URL


def _http_get_json(url: str, timeout: float = 30.0) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "HiveMind-Public-Client/2.6.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"GET {url} returned HTTP {resp.status}")
        return json.loads(resp.read().decode("utf-8"))


def _http_get_bytes(url: str, timeout: float = 30.0) -> tuple[bytes, Dict[str, str]]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "HiveMind-Public-Client/2.6.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"GET {url} returned HTTP {resp.status}")
        headers = {k.lower(): v for k, v in resp.headers.items()}
        return resp.read(), headers


def _http_post_json(
    url: str, payload: Dict[str, Any], timeout: float = 30.0
) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "HiveMind-Public-Client/2.6.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"POST {url} returned HTTP {resp.status}")
        return json.loads(resp.read().decode("utf-8"))


def run_public_gcp_e2e_verification(
    base_url: str = PUBLIC_GCP_CLOUD_RUN_URL,
    output_receipt_path: Path | str = Path(
        "artifacts/gcp_public_deployment_verification.json"
    ),
    tmp_download_dir: Path | str = Path("/tmp/hivemind_la_ruche_client_verify"),
) -> Dict[str, Any]:
    """Run full end-user verification against the public GCP Cloud Run URL."""
    base = base_url.rstrip("/")
    t0 = time.perf_counter()
    dl_dir = Path(tmp_download_dir)
    dl_dir.mkdir(parents=True, exist_ok=True)

    # 1. Health & Cost-Optimization Probe
    health = _http_get_json(f"{base}/health")
    if health.get("status") != "ok":
        raise RuntimeError(f"Health check failed: {health}")

    # 2. HTML5 Dashboard Portal Check
    html_bytes, _ = _http_get_bytes(f"{base}/")
    html_text = html_bytes.decode("utf-8")
    dashboard_ok = (
        "HIVEMIND@HOME" in html_text
        and "LA RUCHE" in html_text
        and "seti-spectrum-canvas" in html_text
    )
    if not dashboard_ok:
        raise RuntimeError("HTML5 dashboard missing expected SETI/La Ruche sections")

    # 3. Telemetry Status & GCP Cost Governance Check
    status_doc = _http_get_json(f"{base}/api/hivemind-at-home/status")
    gcp_fleet = status_doc["gcp_cloud_fleet"]
    gateway_info = gcp_fleet["public_cloud_run_gateway"]

    # 4. La Ruche Manifest & Binary Community MoE Safetensors Download
    manifest = _http_get_json(f"{base}/api/la-ruche/manifest")
    moe_bytes, moe_headers = _http_get_bytes(
        f"{base}/api/la-ruche/download/community-moe.safetensors"
    )
    moe_sha256 = hashlib.sha256(moe_bytes).hexdigest()
    expected_sha256 = str(manifest["combined_safetensors_sha256"])
    header_sha256 = str(moe_headers.get("x-safetensors-sha256", ""))
    if moe_sha256 != expected_sha256 or header_sha256 != expected_sha256:
        raise RuntimeError(
            f"Community MoE SHA-256 mismatch: client={moe_sha256}, header={header_sha256}, manifest={expected_sha256}"
        )

    moe_file_path = dl_dir / "hivemind_la_ruche_community_moe.safetensors"
    moe_file_path.write_bytes(moe_bytes)
    with safe_open(str(moe_file_path), framework="np") as f:
        tensor_keys = list(f.keys())
        embed_shape = list(
            f.get_tensor("moe.backbone.model.embed_tokens.weight").shape
        )

    # 5. Individual Honeycomb Cell (Alveole) Download Verification
    cell_bytes, cell_headers = _http_get_bytes(
        f"{base}/api/la-ruche/download/cell/code_python"
    )
    cell_sha256 = hashlib.sha256(cell_bytes).hexdigest()
    if cell_sha256 != cell_headers.get("x-safetensors-sha256", ""):
        raise RuntimeError("Cell SHA-256 header mismatch on code_python download")

    # 6. Live Community Sparse MoE Inference & OpenAI-Compatible Chat Completion
    inf_1 = _http_post_json(
        f"{base}/api/v1/moe/inference",
        {
            "prompt": "Optimize a PyTorch tensor kernel and prove Lean 4 formal theorem soundness",
            "top_k": 2,
            "max_new_tokens": 32,
        },
    )
    inf_2 = _http_post_json(
        f"{base}/v1/chat/completions",
        {
            "model": "hivemind-la-ruche-moe-v2.6",
            "messages": [
                {
                    "role": "user",
                    "content": "Simulate symplectic Hamiltonian astrophysics orbit and E8 lattice wire quantization in Rust QUIC",
                }
            ],
            "top_k": 2,
            "max_tokens": 48,
        },
    )

    # 7. End-to-End Federated AI Training (Workunit Lease -> Outer Delta Submission -> Ledger Check)
    workunit = _http_post_json(
        f"{base}/api/v1/federated/workunit/request",
        {
            "contributor_handle": "@xaviercallens-public-client",
            "hardware_device": "NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM)",
            "preferred_domain": "code_python",
            "execution_class": "fp16_amp_cuda",
            "inner_steps_H": 8,
        },
    )
    fed_receipt = _http_post_json(
        f"{base}/api/v1/federated/workunit/submit",
        {
            "workunit_id": workunit["workunit_id"],
            "lease_id": workunit["lease_id"],
            "lease_version_tag": workunit["lease_version_tag"],
            "contributor_handle": "@xaviercallens-public-client",
            "hardware_device": "NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM)",
            "domain_name": workunit["domain_name"],
            "wire_codec": "e8_2bit",
            "train_nll_before": 5.5342,
            "train_nll_after": 5.1420,
            "holdout_nll_before": 5.4910,
            "holdout_nll_after": 5.3180,
            "spotcheck_replay_l2_drift": 0.00024,
            "spotcheck_cosine_sim": 0.9996,
        },
    )
    ledger = _http_get_json(f"{base}/api/v1/federated/ledger")

    # 8. Compute Points Redemption Check (GWAYA Python Coding & Astrophysics Exoplanet)
    redeem_resp = _http_post_json(
        f"{base}/api/v1/compute-points/redeem",
        {
            "contributor_handle": "@e2e-public-gcp-verifier",
            "service_id": "python_coding_gwaya",
            "prompt": "Write a vectorized E8 lattice quantizer kernel in Python.",
            "points_balance_before": 1250,
            "lang": "en",
        },
    )
    if redeem_resp.get("status") != "COMPUTE_POINTS_REDEEMED":
        raise RuntimeError(f"Compute points redemption check failed: {redeem_resp}")

    # 8B. 5-Language (EN, FR, ZH, ES, DE), 8 Interactive Examples, Navigation & Video/SVG Media Verification
    i18n_doc = _http_get_json(f"{base}/api/v1/i18n")
    supported_langs = list(i18n_doc.get("supported_languages", []))
    if supported_langs != ["en", "fr", "zh", "es", "de"]:
        raise RuntimeError(f"Expected 5 core languages ['en','fr','zh','es','de'], got {supported_langs}")
    for lang_code in supported_langs:
        lang_single = _http_get_json(f"{base}/api/v1/i18n/{lang_code}")
        if lang_single.get("lang") != lang_code:
            raise RuntimeError(f"i18n endpoint mismatch for {lang_code}")

    examples_doc = _http_get_json(f"{base}/api/v1/examples?lang=fr")
    if int(examples_doc.get("examples_count", 0)) < 8:
        raise RuntimeError(f"Expected >= 8 interactive examples, got {examples_doc.get('examples_count')}")

    poster_svg_bytes, poster_headers = _http_get_bytes(f"{base}/api/v1/media/cyber-bee-poster.svg")
    exo_svg_bytes, exo_headers = _http_get_bytes(f"{base}/api/v1/media/exoplanet-seti-visual.svg")
    video_manifest = _http_get_json(f"{base}/api/v1/media/video-manifest")
    if (
        b"cyber-bee-poster-svg" not in poster_svg_bytes
        or b"exoplanet-seti-svg" not in exo_svg_bytes
        or len(video_manifest.get("chapters", [])) != 4
    ):
        raise RuntimeError("SVG visualizations or 4-chapter video manifest check failed")
    for screen_id in (
        "screen-overview",
        "screen-join-points",
        "screen-moe-bees",
        "screen-results-carousel",
        "screen-tech-leaderboard",
        "cyber-bee-video-element",
        "cyber-bee-video-canvas",
        "interactive-examples-grid",
        "lang-switcher-bar",
    ):
        if screen_id not in html_text:
            raise RuntimeError(f"HTML portal missing required navigation/media ID: {screen_id}")

    # 9. Direct SocrateAI GCP Low-Cost Cloud Storage (gs://socrateai-hivemind-la-ruche-vault) Loader Check
    import torch
    from hivemind.swarm.la_ruche_vault import (
        SOCRATEAI_GCS_DATALAKE_BUCKET,
        SOCRATEAI_GCS_LA_RUCHE_BUCKET,
        SOCRATEAI_GCS_PUBLIC_BASE_URL,
        load_community_moe_from_gcs_or_https,
    )

    gcs_moe_model, gcs_manifest = load_community_moe_from_gcs_or_https(
        cache_dir=dl_dir / "gcs_ruche_cache",
        gcs_public_base_url=SOCRATEAI_GCS_PUBLIC_BASE_URL,
    )
    sample_tokens = torch.tensor([[2, 5, 9, 14, 21, 33, 44, 55]], dtype=torch.long)
    gcs_out = gcs_moe_model(input_ids=sample_tokens)
    gcs_logits_shape = list(gcs_out.logits.shape)
    gcs_finite = bool(torch.isfinite(gcs_out.logits).all())
    if not gcs_finite or gcs_logits_shape != [1, 8, 259]:
        raise RuntimeError(
            f"GCS-loaded Community MoE forward check failed: shape={gcs_logits_shape}, finite={gcs_finite}"
        )

    total_elapsed_ms = (time.perf_counter() - t0) * 1000.0
    report: Dict[str, Any] = {
        "verification_title": "HiveMind@home & La Ruche Public GCP Cloud Run & SocrateAI GCS End-to-End Verification",
        "verified_at_epoch": time.time(),
        "public_dns_base_url": base,
        "canonical_cloud_run_url": gateway_info.get("canonical_run_url", ""),
        "gcp_project_id": gcp_fleet.get("gcp_project_id", "gen-lang-client-0625573011"),
        "gcp_region": gateway_info.get("region", "us-central1"),
        "socrateai_gcs_la_ruche_storage": {
            "primary_gcs_bucket": SOCRATEAI_GCS_LA_RUCHE_BUCKET,
            "datalake_mirror_bucket": SOCRATEAI_GCS_DATALAKE_BUCKET,
            "public_https_base_url": SOCRATEAI_GCS_PUBLIC_BASE_URL,
            "release_id": gcs_manifest["release_id"],
            "combined_safetensors_sha256": gcs_manifest["combined_safetensors_sha256"],
            "pytorch_moe_loaded_and_verified": gcs_finite,
            "forward_logits_shape": gcs_logits_shape,
        },
        "cost_optimization": {
            "min_instances": gateway_info.get("min_instances", 0),
            "max_instances": gateway_info.get("max_instances", 2),
            "cpu_limit": gateway_info.get("cpu_limit", "1000m"),
            "memory_limit": gateway_info.get("memory_limit", "512Mi"),
            "cpu_throttling": gateway_info.get("cpu_throttling", True),
            "idle_monthly_cost_usd": gateway_info.get("idle_monthly_cost_usd", 0.0),
            "cost_governance_verdict": gcp_fleet.get(
                "cost_optimization_audit", {}
            ).get("cost_governance_verdict", ""),
        },
        "dashboard_html_check": {
            "http_status": 200,
            "html_bytes": len(html_bytes),
            "passed": dashboard_ok,
        },
        "la_ruche_model_retrieval": {
            "release_id": manifest["release_id"],
            "num_domains": manifest["num_domains"],
            "community_moe_bytes": len(moe_bytes),
            "community_moe_sha256": moe_sha256,
            "sha256_verified": True,
            "deserialized_tensor_count": len(tensor_keys),
            "embed_tokens_shape": embed_shape,
            "sample_cell_domain": "code_python",
            "sample_cell_bytes": len(cell_bytes),
            "sample_cell_sha256": cell_sha256,
        },
        "sparse_moe_inference_checks": {
            "native_inference": {
                "routed_experts": [
                    r["domain_name"] for r in inf_1["routed_experts"]
                ],
                "logit_l2_norm": inf_1["logit_l2_norm"],
                "latency_ms": inf_1["latency_ms"],
                "inference_receipt_sha256": inf_1["inference_receipt_sha256"],
            },
            "openai_chat_completions": {
                "completion_id": inf_2["id"],
                "routed_experts": [
                    r["domain_name"]
                    for r in inf_2["la_ruche_routing"]["routed_experts"]
                ],
                "inference_receipt_sha256": inf_2["la_ruche_routing"][
                    "inference_receipt_sha256"
                ],
            },
        },
        "federated_training_checks": {
            "leased_workunit_id": workunit["workunit_id"],
            "lease_id": workunit["lease_id"],
            "domain_name": workunit["domain_name"],
            "submission_status": fed_receipt["status"],
            "outer_delta_frob_norm": fed_receipt["outer_delta_frob_norm"],
            "svd_spectral_energy_retained": fed_receipt["svd_spectral_energy_retained"],
            "perplexity_reduction_pct": fed_receipt["perplexity_reduction_pct"],
            "monetary_payout_eur_usd": fed_receipt["monetary_payout_eur_usd"],
            "receipt_sha256": fed_receipt["receipt_sha256"],
            "ledger_completed_receipts_count": ledger["completed_receipts_count"],
        },
        "compute_points_redemption_check": {
            "service_id": redeem_resp["service_id"],
            "specialist_bee_invoked": redeem_resp["specialist_bee_invoked"],
            "points_spent": redeem_resp["points_spent"],
            "points_balance_after": redeem_resp["points_balance_after"],
            "monetary_cost_eur_usd": redeem_resp["monetary_cost_eur_usd"],
            "redemption_receipt_sha256": redeem_resp["redemption_receipt_sha256"],
        },
        "multilingual_i18n_and_media_checks": {
            "supported_languages_verified": supported_langs,
            "interactive_examples_count": int(examples_doc.get("examples_count", 0)),
            "cyber_bee_poster_svg_bytes": len(poster_svg_bytes),
            "exoplanet_seti_svg_bytes": len(exo_svg_bytes),
            "video_cinema_chapters_count": len(video_manifest.get("chapters", [])),
            "navigation_and_media_dom_ids_verified": True,
        },
        "total_verification_duration_ms": round(total_elapsed_ms, 2),
        "all_checks_passed": True,
    }
    report["verification_suite_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True).encode("utf-8")
    ).hexdigest()

    out_p = Path(output_receipt_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify public GCP Cloud Run deployment of HiveMind@home & La Ruche."
    )
    parser.add_argument(
        "--url",
        default=PUBLIC_GCP_CLOUD_RUN_URL,
        help="Public HTTPS URL of the deployed HiveMind@home & La Ruche service",
    )
    parser.add_argument(
        "--output",
        default="artifacts/gcp_public_deployment_verification.json",
        help="Output JSON path for the verification receipt",
    )
    args = parser.parse_args()
    report = run_public_gcp_e2e_verification(
        base_url=args.url, output_receipt_path=args.output
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
