#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Serving acceptance for a Crowd-v1 MoE add-on: does the producer's load path reproduce the judged numbers?

Builds the model exactly as `hivemind_crowd_v1_moe_producer.py` does on CPU (`build_crowd_v1_moe_model`, then
`place_for_inference(cpu, float32)`), re-selects the first ``--per-expert`` chunks of each expert with the frozen
mid-row judge's own selection code and spec, and compares, chunk by chunk, with the judge's receipt:

* the add-on arm against the receipt's ``moe_nll`` (both through the judge's forward and through ``model.forward``,
  the path ``generate_ids`` serves with);
* the dense arm (no add-on) against the receipt's ``base_nll``.

It also generates a few tokens through the KV-cache path. CPU only; it refuses to run with a visible GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hivemind.swarm.crowd_v1_moe_addon import build_crowd_v1_moe_model, load_addon_manifest  # noqa: E402


def _load_judge(path: Path):
    spec = importlib.util.spec_from_file_location("crowdgpt_moe_paired_eval", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["crowdgpt_moe_paired_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


def forward_nll(model: torch.nn.Module, ids: list[int], score_from: int) -> float:
    """Mean NLL of targets ids[score_from:] from ``model.forward`` logits (the serving path)."""
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits, _, _ = model(x)
    start = max(1, score_from)
    return float(F.cross_entropy(logits[0, start - 1 : len(ids) - 1].float(), x[0, start:], reduction="mean"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", type=Path, default=REPO_ROOT / "deploy/crowd_v1_moe/e50_r00_manifest.json")
    ap.add_argument("--addon", type=Path, required=True)
    ap.add_argument("--base-weights", type=Path, required=True)
    ap.add_argument("--tokenizer-json", type=Path, required=True)
    ap.add_argument("--judge-spec", type=Path, required=True, help="the spec the receipt was scored with")
    ap.add_argument("--receipt", type=Path, required=True, help="the judge's full receipt (has per-chunk NLLs)")
    ap.add_argument("--rows-cache-dir", type=Path, required=True)
    ap.add_argument("--judge-script", type=Path, default=REPO_ROOT / "scripts/crowdgpt_moe_paired_eval.py")
    ap.add_argument("--per-expert", type=int, default=2)
    ap.add_argument("--tolerance-nats", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if torch.cuda.is_available():
        print("refused: run with CUDA_VISIBLE_DEVICES= (CPU only)", file=sys.stderr)
        return 2
    torch.set_num_threads(args.threads)
    t0 = time.time()
    judge = _load_judge(args.judge_script)
    manifest = load_addon_manifest(args.manifest)
    spec = json.loads(args.judge_spec.read_text(encoding="utf-8"))
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    if receipt["moe"]["checkpoint_sha256"] != manifest.addon_sha256:
        print("refused: the receipt scored a different add-on", file=sys.stderr)
        return 3
    if receipt["base"]["weights_sha256"] != manifest.base_weights_sha256:
        print("refused: the receipt scored a different base", file=sys.stderr)
        return 3
    ref = {(c["dataset_id"], int(c["row"])): c for c in receipt["chunks"]}

    from tokenizers import Tokenizer

    from hivemind.swarm.crowdgpt_real_shards import LocalHFCacheFetcher, _cached_rows, fetch_rows_datasets_server

    data_manifest = json.loads(Path(spec["data"]["manifest"]).read_text(encoding="utf-8"))
    local = LocalHFCacheFetcher()
    cache = args.rows_cache_dir
    tok = Tokenizer.from_file(str(args.tokenizer_json))
    ch = spec["chunking"]
    chunks, _sel = judge.select_test_chunks(
        data_manifest,
        {"local_hf_cache": lambda ds, cfg, sp, off, n: _cached_rows(cache, local, ds, cfg, sp, off, n),
         "datasets_server": lambda ds, cfg, sp, off, n: _cached_rows(cache, fetch_rows_datasets_server, ds, cfg, sp, off, n)},
        lambda s: tok.encode(s).ids,
        chunks_per_expert=args.per_expert, chunk_tokens=int(ch["chunk_tokens"]),
        scan_cap_rows=int(ch.get("scan_cap_rows_per_source", 2000)), chunk_offset=int(ch.get("chunk_offset", 0)),
        skip_eligible=int(ch.get("skip_eligible_rows", 0)))
    missing = [(c["dataset_id"], c["row"]) for c in chunks if (c["dataset_id"], int(c["row"])) not in ref]
    if missing:
        print(f"refused: selected chunks not in the receipt: {missing[:4]}", file=sys.stderr)
        return 4

    rows: list[dict] = [{"expert_id": c["expert_id"], "dataset_id": c["dataset_id"], "row": int(c["row"])} for c in chunks]
    arms = {"dense": None, "addon": args.addon}
    gen = {}
    for arm, addon in arms.items():
        model, info = build_crowd_v1_moe_model(manifest, args.base_weights, addon, build_dtype=torch.float32)
        model.place_for_inference(device=torch.device("cpu"), dtype=torch.float32, offload_engram_to_cpu=False)
        model.eval()
        want_key = "moe_nll" if addon else "base_nll"
        for r, c in zip(rows, chunks):
            sf = int(c.get("score_from", 0))
            r[f"{arm}_judge_forward"] = judge.chunk_nll(model, c["ids"], score_from=sf)
            r[f"{arm}_serving_forward"] = forward_nll(model, list(c["ids"]), sf)
            r[f"{arm}_receipt"] = float(ref[(c["dataset_id"], int(c["row"]))][want_key])
        prompt = tok.encode("The star Betelgeuse is").ids
        with torch.no_grad():
            out = model.generate_ids(prompt, max_new_tokens=12, temperature=0.0, top_p=1.0)
        gen[arm] = {"n_tokens": len(out), "text": tok.decode([int(i) for i in out], skip_special_tokens=True)}
        if addon:
            gen["addon_load_report"] = info["addon"]
        del model

    def worst(arm: str, kind: str) -> float:
        return max(abs(r[f"{arm}_{kind}"] - r[f"{arm}_receipt"]) for r in rows)

    checks = {f"{a}_{k}_max_abs_diff_nats": worst(a, k) for a in arms for k in ("judge_forward", "serving_forward")}
    passed = all(v <= args.tolerance_nats for v in checks.values()) and gen["addon"]["n_tokens"] > 0
    result = {
        "schema": "hivemind.crowd_v1_moe_serving_acceptance.v1",
        "addon_id": manifest.addon_id,
        "addon_sha256": manifest.addon_sha256,
        "base_weights_sha256": manifest.base_weights_sha256,
        "receipt": str(args.receipt.name),
        "judge_spec_sha256": receipt.get("preregistration", {}).get("spec_sha256"),
        "device": "cpu",
        "dtype": "float32",
        "n_chunks": len(rows),
        "tolerance_nats": args.tolerance_nats,
        "checks": checks,
        "passed": passed,
        "generation_smoke": gen,
        "chunks": rows,
        "wall_seconds": round(time.time() - t0, 1),
        "not_checked": "GPU half-precision serving; the producer HTTP layer (covered by unit tests)",
    }
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("n_chunks", "checks", "passed", "wall_seconds")}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
