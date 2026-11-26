"""Deposit the HiveMind foundation article and its evidence bundle on Zenodo.

Flow (idempotent, verification-first, after the lessons in the AutoevolveAI workspace memory):
  1. create a draft deposition (or a new version of ``--new-version-of``) and read the
     pre-reserved DOI;
  2. write the DOI into ``paper/zenodo_doi.tex`` so the PDF can carry it, then (caller
     recompiles) upload the bundle;
  3. ``--publish`` publishes; afterwards the published record is fetched and every file's
     checksum is compared with the local file. A write that returns an HTTP 5xx is **not**
     retried blindly: the deposition state is re-read first.
Reads the token from ``ZENODO_TOKEN`` or ``~/.config/zenodo/token`` and never prints it.

Usage:
  PYTHONPATH=src python scripts/publish_hivemind_paper_zenodo.py --stage draft     # create draft, upload, write DOI file
  PYTHONPATH=src python scripts/publish_hivemind_paper_zenodo.py --stage upload --deposition-id N   # re-upload files into existing draft
  PYTHONPATH=src python scripts/publish_hivemind_paper_zenodo.py --stage publish --deposition-id N  # publish + verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZENODO = "https://zenodo.org/api"
RECEIPT_PATH = PROJECT_ROOT / "paper" / "ZENODO_DEPOSIT_RECEIPT.json"
SOURCE_COMMIT = os.environ.get("HIVEMIND_SOURCE_COMMIT", "unspecified")
DOI_TEX_PATH = PROJECT_ROOT / "paper" / "zenodo_doi.tex"

BUNDLE: List[Path] = [
    PROJECT_ROOT / "paper" / "hivemind_foundation_article.pdf",
    PROJECT_ROOT / "paper" / "hivemind_foundation_article.tex",
    PROJECT_ROOT / "paper" / "PEER_REVIEW_REPORT.json",
    PROJECT_ROOT / "docs" / "RESEARCH_PROGRAM.md",
    PROJECT_ROOT / "docs" / "MOE_DILOCO_REVIEW_2026-10-23.md",
    PROJECT_ROOT / "lean4" / "HiveMindSpec.lean",
    PROJECT_ROOT / "artifacts" / "architecture_hypothesis_verification_report.json",
    PROJECT_ROOT / "artifacts" / "paper_v3_experiments_report.json",
    PROJECT_ROOT / "artifacts" / "decoupled_diloco_quorum_check.json",
    PROJECT_ROOT / "artifacts" / "experiments" / "E5_toy_prereg.json",
    PROJECT_ROOT / "artifacts" / "experiments" / "E5_toy_report.json",
    PROJECT_ROOT / "artifacts" / "experiments" / "E5_het_prereg.json",
    PROJECT_ROOT / "artifacts" / "experiments" / "E5_het_calibration.json",
    PROJECT_ROOT / "artifacts" / "experiments" / "E5_het_report.json",
    PROJECT_ROOT / "artifacts" / "experiments" / "E10_lattice_prereg.json",
    PROJECT_ROOT / "artifacts" / "experiments" / "E10_lattice_report.json",
    PROJECT_ROOT / "artifacts" / "lean_independent_runner_check.json",
    PROJECT_ROOT / "artifacts" / "lean_rda_multiplier_check.json",
    PROJECT_ROOT / "lean4" / "mathlib" / "RDAMultiplier.lean",
    PROJECT_ROOT / "artifacts" / "experiments" / "E12_pilot_prereg.json",
    PROJECT_ROOT / "artifacts" / "experiments" / "E12_pilot_report.json",
    PROJECT_ROOT / "ROADMAP.md",
    PROJECT_ROOT / "paper" / "review" / "external_review_v3.5.0_2026-10-23.md",
    PROJECT_ROOT / "paper" / "review" / "response_to_external_review_v3.5.0.md",
    PROJECT_ROOT / "artifacts" / "experiments" / "E8_pilot_report.json",
    PROJECT_ROOT / "artifacts" / "predictions_ledger.json",
    PROJECT_ROOT / "artifacts" / "gwaya_stub_audit.json",
    PROJECT_ROOT / "CITATION.cff",
    PROJECT_ROOT / "artifacts" / "gwaya_lite_live_check.json",
    PROJECT_ROOT / "docs" / "benchmarks" / "T4_10h_campaign_report.md",
    PROJECT_ROOT / "docs" / "review" / "t4_branch_review_2026-10-24.md",
    PROJECT_ROOT / "docs" / "community" / "GWAYA_T4_DEPLOYMENT.md",
    PROJECT_ROOT / "docs" / "community" / "GWAYA_LITE_ASSISTANT.md",
    PROJECT_ROOT / "docs" / "community" / "T4_REPLICATION_KIT.md",
    PROJECT_ROOT / "artifacts" / "live_wan_swarm_production_report.json",
    PROJECT_ROOT / "artifacts" / "nocturnal_20_moe_stress_test_report.json",
    PROJECT_ROOT / "artifacts" / "nocturnal_large_moe_20_campaign_report.json",
    PROJECT_ROOT / "docs" / "ROADMAP_AMCP_AGENT_MESH_KAFKA_EVOLUTION.md",
    PROJECT_ROOT / "docs" / "VISION_AND_NEXTGEN_SWARM_ROADMAP.md",
    PROJECT_ROOT / "artifacts" / "nocturnal_reddit_vision_moe_report.json",
    PROJECT_ROOT / "artifacts" / "exoplanet_astro_vmoe_poc_report.json",
    PROJECT_ROOT / "artifacts" / "massive_swarm_image_moe_report.json",
]
# v3.8.0: every receipt behind E10b, E11, E11b, E12 (T4), E13, E14, E15 and E16 (preregistrations, amendments, raw and reports); partial files and superseded E11b/7B raw dumps excluded to respect Zenodo's 100-file-per-record cap
_T4_PREFIXES = ("E10b_", "E11_", "E11b_", "E12_replay_", "E13_", "E14_", "E15_", "E16_")
_SUPERSEDED_RAW = {
    "E11_raw_qwen7b.json",
    "E11b_raw_gemma4-e4b_fineweb-edu_w1024.json",
    "E11b_raw_qwen0.5b_fineweb-edu_w1024.json",
    "E11b_raw_qwen7b_fineweb-edu_w1024.json",
}
BUNDLE += sorted(
    p for p in (PROJECT_ROOT / "artifacts" / "experiments").glob("*.json")
    if p.name.startswith(_T4_PREFIXES) and not p.name.endswith(".partial.json") and p.name not in _SUPERSEDED_RAW and p not in BUNDLE
    and "callensxavier_gmail_com" not in p.read_text(encoding="utf-8", errors="ignore")  # run logs carrying the machine username stay out of the public record
)
CARRIED_PATH = PROJECT_ROOT / "paper" / "zenodo_carried_over_v3.7.1.json"
CARRIED: Dict[str, str] = json.loads(CARRIED_PATH.read_text(encoding="utf-8"))["md5"] if CARRIED_PATH.exists() else {}


def load_token() -> str:
    if os.environ.get("ZENODO_TOKEN"):
        return os.environ["ZENODO_TOKEN"].strip()
    for cand in (Path.home() / ".config" / "zenodo" / "token", Path.home() / ".zenodo_token"):
        if cand.exists():
            txt = cand.read_text(encoding="utf-8").strip()
            if "=" in txt:
                txt = txt.split("=", 1)[1].strip().strip('"').strip("'")
            if txt:
                return txt
    raise RuntimeError("No Zenodo token in ZENODO_TOKEN or ~/.config/zenodo/token")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def request(method: str, url: str, token: str, data: Optional[bytes] = None, content_type: str = "application/json") -> Dict[str, Any]:
    req = urllib.request.Request(url, data=data, method=method, headers={"Authorization": f"Bearer {token}", "Content-Type": content_type})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            body = r.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"Zenodo {method} {url} -> HTTP {e.code}: {detail}") from e


def get_deposition(dep_id: int, token: str) -> Dict[str, Any]:
    return request("GET", f"{ZENODO}/deposit/depositions/{dep_id}", token)


def metadata(version: str, doi_note: str) -> Dict[str, Any]:
    desc = (
        "<p>Preprint and evidence bundle of <b>Project HiveMind (Version 4.4.0)</b>: a system design for decentralized sparse "
        "Mixture-of-Experts training and closed-loop neuro-symbolic weight control on 8 GB consumer GPUs, integrating "
        "Decoupled DiLoCo quorum aggregation (Google DeepMind), PA-CGAD staleness gating, FoMoE partial expert "
        "replication, DeMo/SparseLoCo 2-bit compression, MuLoCo inner preconditioning, a looped MoE backbone, "
        "the biomimetic <b>Bi-Layer Honeybee Agent Mesh (AMCP-Kafka &amp; ABC MoE)</b>, multi-division <b>Vision-MoE (V-MoE)</b>, "
        "live two-host E8/A1 &amp; planetary-scale readiness (PS-01..PS-11), the <b>Reine-Cortex (QueenCortex) zero-gradient NeuroController</b> "
        "with <b>Ph&eacute;romones Royales</b> closed-loop governance, real-life <b>LLM-JEPA / Semantic Tube Prediction (STP)</b> diagnostics and opportunities, "
        "and <b>188 zero-sorry Lean 4 declarations</b> (171 in HiveMindSpec.lean + 14 arbitrary-dimension real-valued theorems in ReineCortexNeuroController.lean + 3 Mathlib RDA theorems).</p>"
        "<p><b>Version 4.4.0 Highlights (Reine-Cortex NeuroController, Ph&eacute;romones Royales Architecture, Real-Life Multi-Model Vaults &amp; LLM-JEPA Opportunities):</b><br/>"
        "1. <i>Four-Tier Eusocial Governance Architecture (La Ruche / Gel&eacute;e Royale / Ph&eacute;romones Royales):</i> "
        "Unifies <b>La Ruche</b> (8-Alv&eacute;ole MoE content vault), <b>Les Abeilles Butineuses</b> (volunteer consumer-GPU DiLoCo/QLoRA foragers), "
        "<b>Les Abeilles Nourrices</b> (0 MiB GPU VRAM CPU weight-space surgeons running Trophallaxie incremental TIES differencing, associative low-RSS (sB)(AP_0) AlphaEdit prose-nullspace projection, "
        "prefix-cached Leave-One-Alv&eacute;ole-Out causal knockout, and Riemannian Fisher-whitened ker(G_D0) + safe-&tau;&ge;1 folded RMSNorm calibration), and "
        "<b>La Reine (Reine-Cortex / QueenCortex)</b> enforcing monotone ANSE Operculation (&Delta;E &lt; 0, D_0 &le; +1.0%, 10^6 Maximum Pain atomic rollback, SingleShotVaultLock) "
        "and broadcasting signed <b>QueenPheromoneDirective</b> packets governing anchor freezing, Ghost Sink mitosis, router mode, and normalized Boltzmann/UCB next-round foraging weights p_d.<br/>"
        "2. <i>Real-Life Multi-Model Results Across 6 Open-Weight Model Families &amp; 10 Locked Holdout Vaults (0 MiB GPU VRAM, 0 Gradient Steps):</i> "
        "Resolves Campaign 2's B3 general-prose gate on CrowdGPT-MoE-8x-3B (<b>PILOT1</b> vault_skip160: <b>-0.554969 nats</b> at E_0=+0.119%, 6.35&times; deeper than uniform interpolation; "
        "21-way disjoint <b>PILOT6</b> vault_skip800: <b>-0.535510 nats</b>), rescues discarded Campaign 3 C3 Arm A (<b>PILOT5</b>: <b>-0.159464 nats</b>, 6/6&lt;0% across all splits), "
        "rescues failing sequentially chained IBM Granite-3.1-1B 32-expert MoE (+0.17297 &rarr; <b>-0.008324 nats</b>, ECE -20.59%) and Granite-3.1-3B 40-expert MoE (+0.06116 &rarr; <b>-0.01287 nats</b> while cutting host RSS <b>-65.5%</b> from 20.6 to 7.1 GiB), "
        "reduces dense Qwen2.5-1.5B ECE by <b>-46.8%</b> at 0.0000995% parameter DOF and Qwen3.5-0.8B 4-gram reasoning loops by <b>-15.4%</b>, and improves live 8-Alv&eacute;ole La Ruche v2.6.0 (<b>PILOT10</b>: <b>-0.009267 nats</b>, PPL 261.46 &rarr; 259.05, ||G_D0 &delta;w||_2 = 1.16e-9) while emitting real Ph&eacute;romones Royales (anchoring code_python/math_reasoning, flagging physics_sim/systems_rust Ghost Sinks, prioritizing formal_lean4 at 25.94%).<br/>"
        "3. <i>Expanded Lean 4 Formal Verification (188 Zero-sorry Declarations):</i> Expands lean4/HiveMindSpec.lean from 166 to <b>171</b> theorems (Season 35 WSP invariants), paired with <b>14</b> arbitrary-dimension real-valued theorems and exact obstructions (+2624/4225 &gt; 0 unwhitened Fisher ascent; L=8 &gt; 0 cross-arm graft collapse) in ReineCortexNeuroController.lean (companion monograph DOI 10.5281/zenodo.24542441) plus 3 Mathlib RDA theorems.<br/>"
        "4. <i>Real-Life LLM-JEPA &amp; STP Diagnostics + Four Falsifiable Reine-Cortex &times; JEPA Opportunities (J1&ndash;J4):</i> "
        "Consolidates measured Tesla T4 receipts (P1 cosine collapse +0.90 nat prose drift, 600s autoresearch H1&ndash;H6 anti-collapse screen restoring safety &le;3.446 nat, E-STP-1/1b/2 keeping gate M4 closed, P0 domain deficits led by Python code +3.84 nat and math +2.44 nat) and formulates four falsifiable opportunities: (J1) P_0-nullspace post-JEPA Gel&eacute;e Royale rescue, (J2) pre-registered E15-J/P2 paired-view docstring&harr;code and theorem&harr;Lean 4 proof fine-tuning, (J3) frozen latent JEPA world-model critic E_JEPA inside ANSE Operculation and Ph&eacute;romones Royales, and (J4) hash-deterministic BOINC JepaSpec replay, JEPA-encoded MoE routing, and Gen-3 D1&ndash;D5 research slices.<br/>"
        "5. <i>Version 4.3.0 &amp; Earlier Foundations Retained:</i> Live two-host E8/A1 verification, November 2026 field campaigns (Section 14), E20 &amp; &gt;200-node international launch forecast, Vision-MoE &amp; 18-month roadmap, Bi-Layer Honeybee Agent Mesh, and 47-node WAN swarm validation.</p>"
        "<p>The JSON artifacts in this record (including <code>v4.4.0_reine_cortex_neurocontroller_jepa_bundle.zip</code>) are the sole source of every number in the PDF (scripted provenance lock PROV-01..PROV-87 in paper/autoevolve_peer_review_pipeline.py; <code>is_peer_review: false</code>). Source commit: "
        + SOURCE_COMMIT + " of the HiveMind repository on GitHub (xaviercallens/HiveMind).</p>"
        f"<p>{doi_note}</p>"
    )
    return {
        "metadata": {
            "title": "Project HiveMind: A System Design for Decentralized Sparse Mixture-of-Experts Training with DiLoCo-Style Outer Optimization on 8 GB Consumer GPUs",
            "upload_type": "publication",
            "publication_type": "preprint",
            "description": desc,
            "creators": [{"name": "Callens, Xavier", "affiliation": "Project HiveMind / AutoevolveAI"}],
            "keywords": [
                "decentralized training", "DiLoCo", "Decoupled DiLoCo", "mixture-of-experts", "FoMoE",
                "consumer GPU", "Byzantine fault tolerance", "Lean 4", "agent mesh", "honeybee swarm",
                "Reine-Cortex", "NeuroController", "Weight-Space Programming", "Pheromones Royales",
                "LLM-JEPA", "Semantic Tube Prediction", "AlphaEdit", "Vision-MoE", "Split Learning",
            ],
            "license": "cc-by-4.0",
            "related_identifiers": [
                {"identifier": "https://github.com/xaviercallens/HiveMind", "relation": "isSupplementTo", "scheme": "url"},
                {"identifier": "10.5281/zenodo.24542441", "relation": "isSupplementedBy", "scheme": "doi"},
                {"identifier": "10.5281/zenodo.24542440", "relation": "isSupplementedBy", "scheme": "doi"},
            ],
            "version": version,
            "notes": "Preprint v4.4.0. Adds Section 15: full integration of the Reine-Cortex (QueenCortex) zero-gradient NeuroController, La Ruche / Gelee Royale / Pheromones Royales four-tier architecture, 10 single-shot certified holdout vaults across 6 model families at 0 MiB GPU VRAM, 188 zero-sorry Lean 4 declarations (171 HiveMindSpec + 14 ReineCortexNeuroController + 3 Mathlib RDA), companion Zenodo monograph 10.5281/zenodo.24542441, and real-life LLM-JEPA / STP diagnostics paired with four falsifiable Reine-Cortex x LLM-JEPA opportunities (J1-J4).",
        }
    }


def upload_files(dep: Dict[str, Any], token: str, files: List[Path]) -> List[Dict[str, Any]]:
    bucket = dep["links"]["bucket"]
    existing = {f["filename"]: f for f in get_deposition(dep["id"], token).get("files", [])}
    target_names = {f.name for f in files} | set(CARRIED.keys())
    for name, info in list(existing.items()):
        if name not in target_names:
            print(f"Deleting superseded file from draft: {name}")
            request("DELETE", f"{ZENODO}/deposit/depositions/{dep['id']}/files/{info['id']}", token)
            del existing[name]
    uploaded: List[Dict[str, Any]] = []
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f)
        local_md5 = md5(f)
        if f.name in existing and existing[f.name].get("checksum", "").replace("md5:", "") == local_md5:
            uploaded.append({"filename": f.name, "md5": local_md5, "status": "already_present"})
            continue
        if f.name in existing:
            request("DELETE", f"{ZENODO}/deposit/depositions/{dep['id']}/files/{existing[f.name]['id']}", token)
        with open(f, "rb") as fh:
            try:
                request("PUT", f"{bucket}/{f.name}", token, fh.read(), "application/octet-stream")
            except RuntimeError as e:
                if "HTTP 5" not in str(e):
                    raise
                # verify-before-retry: a 5xx may still have stored the file
                now = {x["filename"]: x for x in get_deposition(dep["id"], token).get("files", [])}
                if f.name not in now or now[f.name].get("checksum", "").replace("md5:", "") != local_md5:
                    raise
        uploaded.append({"filename": f.name, "md5": local_md5, "status": "uploaded"})
    # final verification of the draft's file list
    now = {x["filename"]: x for x in get_deposition(dep["id"], token).get("files", [])}
    for f in files:
        remote = now.get(f.name, {}).get("checksum", "").replace("md5:", "")
        if remote != md5(f):
            raise RuntimeError(f"checksum mismatch after upload for {f.name}: remote={remote}")
    return uploaded


def write_receipt(payload: Dict[str, Any]) -> None:
    payload["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    RECEIPT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["draft", "upload", "publish"], required=True)
    ap.add_argument("--deposition-id", type=int, default=None)
    ap.add_argument("--new-version-of", type=int, default=None, help="existing record id to create a new version of")
    ap.add_argument("--version", default="4.4.0")
    args = ap.parse_args()
    token = load_token()

    if args.stage == "draft":
        if args.new_version_of is not None:
            nv = request("POST", f"{ZENODO}/deposit/depositions/{args.new_version_of}/actions/newversion", token)
            dep = request("GET", nv["links"]["latest_draft"], token)
        else:
            dep = request("POST", f"{ZENODO}/deposit/depositions", token, b"{}")
        dep = request("PUT", f"{ZENODO}/deposit/depositions/{dep['id']}", token,
                      json.dumps(metadata(args.version, "DOI is pre-reserved at draft time and printed in the PDF footer.")).encode("utf-8"))
        doi = dep.get("metadata", {}).get("prereserve_doi", {}).get("doi") or dep.get("doi")
        if not doi:
            raise RuntimeError("no pre-reserved DOI returned")
        DOI_TEX_PATH.write_text("\\newcommand{\\zenodoDOI}{" + doi + "}\n", encoding="utf-8")
        write_receipt({"stage": "draft", "deposition_id": dep["id"], "prereserved_doi": doi, "state": dep.get("state"),
                       "html": dep.get("links", {}).get("html"), "files": []})
        print(json.dumps({"deposition_id": dep["id"], "prereserved_doi": doi, "state": dep.get("state")}, indent=2))
        print("Now recompile the PDF (it reads paper/zenodo_doi.tex) and run --stage upload.")
        return 0

    if args.deposition_id is None:
        raise SystemExit("--deposition-id required for upload/publish")
    dep = get_deposition(args.deposition_id, token)

    if args.stage == "upload":
        dep = request("PUT", f"{ZENODO}/deposit/depositions/{dep['id']}", token,
                      json.dumps(metadata(args.version, f"DOI: {dep.get('metadata', {}).get('prereserve_doi', {}).get('doi') or dep.get('doi')}")).encode("utf-8"))
        uploaded = upload_files(dep, token, BUNDLE)
        receipt = json.loads(RECEIPT_PATH.read_text()) if RECEIPT_PATH.exists() else {}
        receipt.update({"stage": "upload", "deposition_id": dep["id"], "files": uploaded,
                        "local_sha256": {f.name: sha256(f) for f in BUNDLE}, "state": dep.get("state")})
        write_receipt(receipt)
        print(json.dumps({"deposition_id": dep["id"], "uploaded": [(u["filename"], u["status"]) for u in uploaded]}, indent=2))
        return 0

    # publish
    if dep.get("state") == "done" or dep.get("submitted"):
        print("deposition already published; verifying only")
    else:
        try:
            dep = request("POST", f"{ZENODO}/deposit/depositions/{dep['id']}/actions/publish", token)
        except RuntimeError as e:
            if "HTTP 5" not in str(e):
                raise
            dep = get_deposition(args.deposition_id, token)  # a 5xx may still have published
            if not dep.get("submitted"):
                raise
    rec_id = dep.get("record_id") or dep.get("id")
    record = request("GET", f"{ZENODO}/records/{rec_id}", token)
    remote = {f["key"]: f.get("checksum", "").replace("md5:", "") for f in record.get("files", [])}
    mismatches = [f.name for f in BUNDLE if remote.get(f.name) != md5(f)]
    mismatches += [n for n, m in CARRIED.items() if remote.get(n) != m]
    receipt = json.loads(RECEIPT_PATH.read_text()) if RECEIPT_PATH.exists() else {}
    receipt.update({
        "stage": "published", "deposition_id": dep["id"], "record_id": rec_id, "doi": dep.get("doi"),
        "doi_url": dep.get("links", {}).get("doi"), "html": dep.get("links", {}).get("record_html") or dep.get("links", {}).get("html"),
        "conceptdoi": dep.get("conceptdoi"), "state": dep.get("state"), "submitted": dep.get("submitted"),
        "remote_md5": remote, "checksum_mismatches": mismatches, "verified": not mismatches and set(remote) == ({f.name for f in BUNDLE} | set(CARRIED)),
    })
    write_receipt(receipt)
    print(json.dumps({k: receipt[k] for k in ("doi", "record_id", "state", "verified", "checksum_mismatches")}, indent=2))
    return 0 if receipt["verified"] else 1


if __name__ == "__main__":
    sys.exit(main())
