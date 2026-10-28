#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Refresh ``configs/datasets_s21.json`` from the Hugging Face API and validate a streamed slice of every accepted source.

Steps (design: ``docs/S21_DATASETS.md``; library: ``hivemind.data.dataset_registry``):
1. For each candidate in ``configs/datasets_s21_candidates.json`` query ``https://huggingface.co/api/datasets/<id>`` and record
   licence (from cardData), revision sha, size and the retrieval time. Unusable licences are recorded as REJECTED with the reason.
2. For every ACCEPTED source, read the split size from datasets-server ``/size`` and stream the first ``slice_rows`` rows through
   ``/rows`` (no full download); map them to the night-pack train format and report PASSED / FAILED / BLOCKED with real counts.
3. Decontaminate the mapped slices against the held-out eval packs (``--eval-packs``, built by scripts/night_pack_build.py) and the
   reference eval sets listed in the candidates file, then run the mixture planner on the slices (``--slice-total`` rows).

Slices, the eval index summary and the full provenance manifest go to ``--work`` (keep it outside the repository); only the
registry and the per-source counts are written to ``--out``.

Usage: python scripts/refresh_dataset_registry.py --work DIR [--eval-packs DIR] [--out configs/datasets_s21.json] [--only id,id] [--no-slice]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hivemind.data.dataset_registry import (
    ACCEPTED,
    LICENSE_ALLOWLIST,
    PASSED,
    EvalIndex,
    MixtureConfig,
    SourceSpec,
    build_mixture,
    default_fetch,
    load_candidates,
    registry_entry,
    retrying_fetch,
    slice_rows,
    split_size,
    utc_now,
    validate_slice,
)

REPO = Path(__file__).resolve().parent.parent
FETCH = retrying_fetch(default_fetch)


def cached_rows_fetch(cache: Path) -> Callable[[str], bytes]:
    """Cache datasets-server ``/rows`` pages on disk (a rerun then re-reads identical bytes, same sha256); Hub metadata is never cached."""
    cache.mkdir(parents=True, exist_ok=True)

    def fetch(url: str) -> bytes:
        if "/rows?" not in url:
            return FETCH(url)
        path = cache / hashlib.sha256(url.encode()).hexdigest()
        if path.exists():
            return path.read_bytes()
        data = FETCH(url)
        path.write_bytes(data)
        return data

    return fetch


def source_label(spec: SourceSpec) -> str:
    return f"{spec.dataset_id}@{spec.config}/{spec.split}"


def build_index(doc: dict[str, Any], eval_packs: Path | None, fetch: Callable[[str], bytes]) -> tuple[EvalIndex, list[dict[str, Any]]]:
    index = EvalIndex()
    notes: list[dict[str, Any]] = []
    if eval_packs:
        for pack in sorted(p for p in eval_packs.iterdir() if (p / "eval.jsonl").exists()):
            index.add_eval_pack(pack)
            notes.append({"eval_pack": pack.name, "sha256": hashlib.sha256((pack / "eval.jsonl").read_bytes()).hexdigest()})
    for ref in doc.get("decontamination_references", []):
        spec = SourceSpec(ref["id"], "reference", ref["config"], ref["split"], {"kind": "fields"}, False, "eval reference", int(ref["rows"]))
        res = slice_rows(spec, fetch)
        texts = ["\n".join(str(r.get(f, "")) for f in ref["text_fields"]) for r in res.rows]
        index.add(f"reference:{source_label(spec)}", texts)
        notes.append({"reference": source_label(spec), "status": res.status, "rows": len(res.rows), "reason": res.reason,
                      "shards_sha256": [s.sha256 for s in res.shards]})
    return index, notes


def eval_index_gaps(notes: list[dict[str, Any]], eval_packs_given: bool) -> list[str]:
    """What the decontamination index is missing: no night packs, or a reference eval set that was not fully read.

    Decontamination against a partial index under-removes silently, so the gaps are written next to the counts."""
    gaps = [] if eval_packs_given and any("eval_pack" in n for n in notes) else ["no night-pack eval.jsonl was indexed (--eval-packs)"]
    gaps += [f"reference {n['reference']} is {n['status']}: {n['reason']}" for n in notes if "reference" in n and n["status"] != PASSED]
    return gaps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(REPO / "configs" / "datasets_s21_candidates.json"))
    ap.add_argument("--out", default=str(REPO / "configs" / "datasets_s21.json"))
    ap.add_argument("--work", required=True, help="scratch directory for slices and the manifest (outside the repo)")
    ap.add_argument("--eval-packs", default="", help="night pack directory (scripts/night_pack_build.py --out)")
    ap.add_argument("--only", default="")
    ap.add_argument("--no-slice", action="store_true")
    ap.add_argument("--slice-total", type=int, default=600)
    args = ap.parse_args()

    cand_path = Path(args.candidates)
    doc, specs = load_candidates(cand_path)
    if args.only:
        keep = set(args.only.split(","))
        specs = [s for s in specs if s.dataset_id in keep]
    work = Path(args.work)
    (work / "slices").mkdir(parents=True, exist_ok=True)
    rows_fetch = cached_rows_fetch(work / "rows_cache")

    entries: list[dict[str, Any]] = []
    rows_by_source: dict[str, list[dict[str, Any]]] = {}
    shards = []
    for spec in specs:
        entry = registry_entry(spec, FETCH, utc_now())
        if entry["status"] == ACCEPTED and not args.no_slice:
            entry["split_size"] = split_size(spec, FETCH)
            res = slice_rows(spec, rows_fetch)
            check: dict[str, Any] = {"status": res.status, "reason": res.reason, "rows_read": len(res.rows)}
            if res.rows:
                report, mapped = validate_slice(spec, res.rows, doc["domains"][spec.domain]["system_prompt"])
                check = {**report, **({"status": res.status, "reason": res.reason} if res.status != PASSED else {})}
                if report["status"] == PASSED:
                    rows_by_source[source_label(spec)] = mapped
                    out = work / "slices" / (source_label(spec).replace("/", "__") + ".jsonl")
                    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in mapped), encoding="utf-8")
            check["slice_endpoint"] = "datasets-server /rows (current parquet conversion; not pinned to revision_sha)"
            check["shards"] = [s.to_dict() for s in res.shards]
            shards.extend(res.shards)
            entry["slice_check"] = check
        entries.append(entry)
        print(f"{entry['status']:9s} {source_label(spec):70s} {entry.get('license')} {entry.get('slice_check', {}).get('status', '')} "
              f"{entry.get('slice_check', {}).get('rows_mapped', '')} {entry.get('reason', '')[:80]}", flush=True)

    result: dict[str, Any] = {
        "about": "Generated by scripts/refresh_dataset_registry.py from the Hugging Face Hub API and datasets-server. Do not edit by hand.",
        "candidates_file": str(cand_path.relative_to(REPO)) if cand_path.is_relative_to(REPO) else str(cand_path),
        "candidates_sha256": hashlib.sha256(cand_path.read_bytes()).hexdigest(), "generated_utc": utc_now(),
        "policy": {"license_allowlist": LICENSE_ALLOWLIST, **doc["mixture"]},
        "summary": {"candidates": len(entries), "accepted": sum(e["status"] == ACCEPTED for e in entries),
                    "rejected": sum(e["status"] != ACCEPTED for e in entries),
                    "slice_passed": sum(e.get("slice_check", {}).get("status") == PASSED for e in entries)},
        "registry": entries,
    }
    if rows_by_source:
        index, notes = build_index(doc, Path(args.eval_packs) if args.eval_packs else None, rows_fetch)
        by_label = {source_label(s): s for s in specs}
        mcfg = MixtureConfig(total_rows=args.slice_total, domain_weights={d: v["weight"] for d, v in doc["domains"].items()},
                             per_source_cap=doc["mixture"]["per_source_cap"], unverified_cap=doc["mixture"]["unverified_cap"],
                             seed=doc["mixture"]["seed"])
        mix, manifest = build_mixture(rows_by_source, {k: by_label[k].verified_signal for k in rows_by_source},
                                      {k: by_label[k].domain for k in rows_by_source}, mcfg, index, shards)
        manifest["eval_index_inputs"] = notes
        gaps = eval_index_gaps(notes, bool(args.eval_packs))
        manifest["eval_index_gaps"] = gaps
        for g in gaps:
            print(f"WARNING: decontamination index incomplete: {g}", file=sys.stderr)
        (work / "slice_mixture.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in mix), encoding="utf-8")
        (work / "slice_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        result["slice_validation"] = {
            "note": "decontamination and mixture run on the streamed slices only (a pipeline check, not the training mixture)",
            "eval_index": manifest["decontamination"] | {"inputs": notes}, "plan": manifest["plan"],
            "mixture_rows": manifest["rows"], "mixture_sha256": manifest["mixture_sha256"],
            "eval_index_complete": not gaps, "eval_index_gaps": gaps}
        print(json.dumps({"decontamination": manifest["decontamination"]["per_source"], "plan_domains": manifest["plan"]["domains"],
                          "unverified_fraction": manifest["plan"]["unverified_fraction"]}, indent=1))
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
