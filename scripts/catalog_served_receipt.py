#!/usr/bin/env python3
"""Receipt: what the public catalog serves versus what it says is trained.

Reads ``src/hivemind/router/catalog.py`` statically (no import, no network) and writes a JSON
receipt the paper's provenance lock can check. It records, for the serving tiers, which Ollama
tag backs each one and whether that tag is a HiveMind-built artefact; and, for the public ids,
how many carry a "trained" status while being served by an off-the-shelf tag, and how many
still name a Hugging Face repository under an account that was never published.

Usage: python3 scripts/catalog_served_receipt.py [--catalog PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
UNPUBLISHED_HF_ACCOUNT = "xaviercallens/"


def _lit(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def build_receipt(catalog_path: pathlib.Path) -> dict:
    src = catalog_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fields: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ModelTier":
            fields = [b.target.id for b in node.body if isinstance(b, ast.AnnAssign)]
    tiers: list[dict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "ModelTier":
            d = dict(zip(fields, [_lit(a) for a in node.args]))
            d.update({k.arg: _lit(k.value) for k in node.keywords if k.arg})
            tiers.append({"tier_id": d.get("tier_id"), "ollama_tag": d.get("ollama_tag"), "title": d.get("title"),
                          "moe": d.get("moe"), "hivemind_built_tag": str(d.get("ollama_tag", "")).startswith("hivemind-")})
    tag_by_tier = {t["tier_id"]: t["ollama_tag"] for t in tiers}
    publics: list[dict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            kws = {k.arg: _lit(k.value) for k in node.keywords if k.arg}
            if "public_id" in kws and "status" in kws:
                tier_id = kws.get("tier_id")
                tag = tag_by_tier.get(tier_id)
                publics.append({
                    "public_id": kws["public_id"], "status": kws["status"], "tier_id": tier_id,
                    "served_by_tag": tag, "hivemind_built_tag": bool(tag) and tag.startswith("hivemind-"),
                    "hf_repo": kws.get("hf_repo"), "tokens_trained_b": kws.get("tokens_trained_b"),
                    "shard_format": kws.get("shard_format"),
                    "hf_repo_under_unpublished_account": str(kws.get("hf_repo") or "").startswith(UNPUBLISHED_HF_ACCOUNT),
                })
    status_counts = Counter(p["status"] for p in publics)
    live_served_off_the_shelf = [p["public_id"] for p in publics
                                 if str(p["status"]).startswith("LIVE") and not p["hivemind_built_tag"]]
    unpublished = [p["public_id"] for p in publics if p["hf_repo_under_unpublished_account"]]
    qlora_claim = [p["public_id"] for p in publics if "qlora" in str(p["shard_format"] or "").lower()]
    return {
        "receipt": "CATALOG_SERVED",
        "catalog_path": str(catalog_path.relative_to(ROOT)) if catalog_path.is_relative_to(ROOT) else str(catalog_path),
        "catalog_sha256": hashlib.sha256(src.encode("utf-8")).hexdigest(),
        "serving_tiers": {
            "count": len(tiers),
            "distinct_ollama_tags": sorted({t["ollama_tag"] for t in tiers if t["ollama_tag"]}),
            "hivemind_built_tags": sorted({t["ollama_tag"] for t in tiers if t["hivemind_built_tag"]}),
            "tiers": tiers,
        },
        "public_ids": {
            "count": len(publics),
            "status_counts": dict(sorted(status_counts.items(), key=lambda kv: str(kv[0]))),
            "live_status_served_by_off_the_shelf_tag": {"count": len(live_served_off_the_shelf), "ids": sorted(live_served_off_the_shelf)},
            "qlora_shard_format_claimed": {"count": len(qlora_claim), "ids": sorted(qlora_claim)},
            "hf_repo_under_unpublished_account": {"account_prefix": UNPUBLISHED_HF_ACCOUNT, "count": len(unpublished), "ids": sorted(unpublished)},
            "entries": publics,
        },
        "reading": (
            "A public id with a LIVE status is served by the Ollama tag of its tier_id. Only tags starting with 'hivemind-' "
            "name a HiveMind-built artefact. tokens_trained_b and shard_format are catalog metadata, not evidence of training; "
            "the BOINC chain receipt (CHAIN_STATUS) is the only training evidence in this repository."
        ),
    }


def runtime_receipt() -> dict:
    """Import the catalog and measure what the gateway serves after ``_truthful()`` has rewritten the unpublished entries."""
    import importlib
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    cat = importlib.import_module("hivemind.router.catalog")
    tags = {t.tier_id: t.ollama_tag for t in cat.TIERS}
    rows = []
    for m in cat.PREWIRED_PARTNER_MODELS:
        tag = getattr(m, "served_by", "") or tags.get(m.tier_id, "")
        rows.append({"public_id": m.public_id, "status": m.status, "served_by": tag, "hivemind_built_tag": str(tag).startswith("hivemind-"),
                     "hf_repo": m.hf_repo, "tokens_trained_b": m.tokens_trained_b, "title_planned": str(m.title).startswith("Planned, not trained yet")})
    status_counts = Counter(r["status"] for r in rows)
    live_off = [r["public_id"] for r in rows if str(r["status"]).startswith("LIVE") and not r["hivemind_built_tag"]]
    trained_claim = [r["public_id"] for r in rows if (r["tokens_trained_b"] or 0) > 0 and not r["hivemind_built_tag"] and str(r["status"]).startswith("LIVE")]
    moe_status_off = [r["public_id"] for r in rows if r["status"] in ("LIVE_NOCTURNAL_MOE", "LIVE_WAVE2_SCIENTIFIC_MOE") and not r["hivemind_built_tag"]]
    unpublished_rt = [r["public_id"] for r in rows if str(r["hf_repo"] or "").startswith(UNPUBLISHED_HF_ACCOUNT)]
    return {
        "note": "measured after import: _truthful() in catalog.py rewrites entries whose hf_repo is in UNPUBLISHED_HF_REPOS",
        "count": len(rows),
        "status_counts": dict(sorted(status_counts.items(), key=lambda kv: str(kv[0]))),
        "live_status_served_by_off_the_shelf_tag": {"count": len(live_off), "ids": sorted(live_off)},
        "live_status_with_tokens_trained_claim_off_the_shelf": {"count": len(trained_claim), "ids": sorted(trained_claim)},
        "moe_trained_status_served_by_off_the_shelf_tag": {"count": len(moe_status_off), "ids": sorted(moe_status_off)},
        "hf_repo_under_unpublished_account_at_runtime": {"count": len(unpublished_rt), "ids": sorted(unpublished_rt)},
        "relabelled_planned_not_trained": sum(1 for r in rows if r["title_planned"]),
        "entries": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalog", default=str(ROOT / "src/hivemind/router/catalog.py"))
    ap.add_argument("--out", default=str(ROOT / "artifacts/experiments/CATALOG_SERVED_2026-11-07.json"))
    a = ap.parse_args()
    receipt = build_receipt(pathlib.Path(a.catalog))
    receipt["source_literals_note"] = "serving_tiers and public_ids above are read from the source text; the runtime section below is what the gateway exposes"
    try:
        receipt["runtime"] = runtime_receipt()
    except Exception as exc:  # the receipt still records the static view
        receipt["runtime"] = {"error": f"{type(exc).__name__}: {exc}"}
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    s, p = receipt["serving_tiers"], receipt["public_ids"]
    print(f"serving tiers {s['count']}, distinct tags {len(s['distinct_ollama_tags'])}, hivemind-built tags {s['hivemind_built_tags']}")
    print(f"public ids {p['count']}; statuses {p['status_counts']}")
    print(f"LIVE status served by off-the-shelf tag: {p['live_status_served_by_off_the_shelf_tag']['count']}")
    print(f"qlora shard format claimed: {p['qlora_shard_format_claimed']['count']}")
    print(f"hf_repo under unpublished account: {p['hf_repo_under_unpublished_account']['count']} {p['hf_repo_under_unpublished_account']['ids']}")
    r = receipt["runtime"]
    if "error" in r:
        print("runtime: ERROR", r["error"])
    else:
        print(f"runtime: {r['count']} entries; statuses {r['status_counts']}; LIVE off-the-shelf {r['live_status_served_by_off_the_shelf_tag']['count']}; "
              f"MoE-trained status off-the-shelf {r['moe_trained_status_served_by_off_the_shelf_tag']['count']}; tokens-trained claim {r['live_status_with_tokens_trained_claim_off_the_shelf']['count']}; "
              f"unpublished at runtime {r['hf_repo_under_unpublished_account_at_runtime']['count']}; relabelled planned {r['relabelled_planned_not_trained']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
