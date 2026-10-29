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
]
# v3.8.0: every receipt behind E10b, E11, E11b, E12 (T4), E13, E14, E15 and E16 (preregistrations, amendments, raw and reports); partial files excluded
_T4_PREFIXES = ("E10b_", "E11_", "E11b_", "E12_replay_", "E13_", "E14_", "E15_", "E16_")
BUNDLE += sorted(
    p for p in (PROJECT_ROOT / "artifacts" / "experiments").glob("*.json")
    if p.name.startswith(_T4_PREFIXES) and not p.name.endswith(".partial.json") and p not in BUNDLE
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
        "<p>Preprint and evidence bundle of <b>Project HiveMind</b>: a system design for decentralized sparse "
        "Mixture-of-Experts training with DiLoCo-style outer optimization on 8 GB consumer GPUs, integrating "
        "Decoupled DiLoCo quorum aggregation (Google DeepMind), PA-CGAD staleness gating, FoMoE partial expert "
        "replication, DeMo/SparseLoCo 2-bit compression, MuLoCo inner preconditioning, a looped MoE backbone, "
        "the biomimetic <b>Bi-Layer Honeybee Agent Mesh (AMCP-Kafka & ABC MoE)</b>, and 141 Lean 4 theorems compiling without sorry.</p>"
        "<p><b>Version 4.0.0 Highlights:</b><br/>"
        "1. <i>Bi-Layer Honeybee Agent Mesh:</i> Layer 1 Nesting Scrutiny & Waggle Dance Quorum Sensing on Kafka (CloudEvents 1.0, duration tau_0 * Q, MCL recruitment, partition lease lock) "
        "and Layer 2 Artificial Bee Colony (ABC) MoE Loop (Employed exploitation, Observer fitness critique with finite abandonment limit L_limit=2, Scout tool calls for Z3/SciPy/Lean 4/Web Search) "
        "plus the P2P Swarm Nectar Cache (SHA-256 deduplication, 0.005 HMT royalties; 60% reduction in duplicate queries).<br/>"
        "2. <i>20 Sectoral Hugging Face MoE Adapters:</i> +277.06 TFLOPs of attested compute across 4 hardware tiers (RTX 3090, T4, RTX 4070, RTX 2070), expanding the active fleet to 35 models total "
        "across 4 Macro-Cluster Super-MoEs with +33.1% mean accuracy gain and 0 VRAM breaches.<br/>"
        "3. <i>Dynamic Diurnal Engine & Long-Tail Intent Scanner:</i> Timezone-aware scheduling prioritizing rare domain requests via S_d = (1/sqrt(N_d+1)) * Complexity_d * RecencyBonus_d.<br/>"
        "4. <i>Bring Your Own Bee (BYOB):</i> Formal 4-stage attestation lifecycle illustrated with 3 live exemplars (bee-quantum-simd, bee-reverse-cve, bee-genomics-crispr).<br/>"
        "5. <i>Live 47-Node WAN Swarm:</i> 5,632.11 BOINC Cobblestones (2.433 PFLOPs), 4,525.0 TFLOPs-h cumulative, 1,427 OpenRouter completions (196.8 ms p50), and 0.0% execution-class-matched false rejection rate.</p>"
        "<p>The JSON artifacts in this bundle are the sole source of every number in the PDF (provenance lock PROV-01..PROV-82 in paper/autoevolve_peer_review_pipeline.py). Source commit: "
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
                "consumer GPU", "Byzantine fault tolerance", "Lean 4", "agent mesh", "honeybee swarm", "ABC algorithm", "Kafka AMCP",
            ],
            "license": "cc-by-4.0",
            "related_identifiers": [
                {"identifier": "https://github.com/xaviercallens/HiveMind", "relation": "isSupplementTo", "scheme": "url"},
            ],
            "version": version,
            "notes": "Preprint v4.0.0. Live 47-node WAN swarm and 20-adapter nocturnal MoE campaign telemetry. See Section 10 and Section 11 of the PDF.",
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
    ap.add_argument("--version", default="4.0.0")
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
