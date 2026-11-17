#!/usr/bin/env python3
"""Download the Hugging Face parquet files behind the QLoRA MoE training domains into the local HF hub cache.

The work generator (src/hivemind/boinc/real_client/moe_work.py) reads only from this cache (HF_HUB_OFFLINE=1), so a host that creates
workunits must run this once. Every file is pinned to a dataset commit, so the training and held-out slices are reproducible.
Usage: python scripts/moe_fetch_domains.py [--cache DIR] [--only domain,domain] [--list]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# repo -> (pinned commit, files). Converted-parquet repos use the refs/convert/parquet commit.
DATASETS: dict[str, tuple[str, tuple[str, ...]]] = {
    "allenai/sciq": ("2c94ad3e1aafab77146f384e23536f97a4849815", ("data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet")),
    "openlifescienceai/medmcqa": ("91c6572c454088bf71b679ad90aa8dffcd0d5868", ("data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet")),
    "lex_glue": ("c23fdff1a6bf74e0e1a71cb86f1e781d37da888c", ("case_hold/train-00000-of-00001.parquet", "case_hold/test-00000-of-00001.parquet")),
    "FinGPT/fingpt-fiqa_qa": ("c65953e39aaea4779efc54805c64769429c554c8", ("data/train-00000-of-00001-ab79bf9300210e98.parquet",)),
    "ccdv/arxiv-summarization": ("240aaf1a969b3f8cd0ade6986bfad0cd730ee288", ("document/validation-00000-of-00001.parquet", "document/test-00000-of-00001.parquet")),
    "flytech/python-codes-25k": ("956d3f9eb97c74850f0d5f8b0d755aeb474409d7", ("default/train/0000.parquet",)),
    "CyberNative/Code_Vulnerability_Security_DPO": ("deafb4ff7cad9fc87f1fda0f373766eb7a9c5c0a", ("default/train/0000.parquet",)),
    "camel-ai/physics": ("db2cabc56050dad732a03674b47bdeaa9595ced0", ("default/train/0000.parquet",)),
    "camel-ai/chemistry": ("69f5cea0e307af555293cfabc1d6f7784cbc6705", ("default/train/0000.parquet",)),
    "AI-MO/NuminaMath-CoT": ("9d8d210c9f6a36c8f3cd84045668c9b7800ef517", ("data/train-00000-of-00005.parquet", "data/test-00000-of-00001.parquet")),
}


def cache_dir_name(repo: str) -> str:
    return "datasets--" + repo.replace("/", "--")


def fetch(repo: str, cache: Path) -> list[Path]:
    from huggingface_hub import hf_hub_download

    sha, files = DATASETS[repo]
    return [Path(hf_hub_download(repo, f, repo_type="dataset", revision=sha, cache_dir=str(cache))) for f in files]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/huggingface/hub"))
    ap.add_argument("--only", default="", help="comma-separated dataset repos (default: all)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    repos = [r for r in a.only.split(",") if r] or list(DATASETS)
    unknown = [r for r in repos if r not in DATASETS]
    if unknown:
        print(f"unknown dataset(s): {unknown}", file=sys.stderr)
        return 2
    for r in repos:
        if a.list:
            print(r, DATASETS[r][0][:8], *DATASETS[r][1])
            continue
        for p in fetch(r, Path(a.cache)):
            print(f"ok {r} {p.name} {p.stat().st_size} bytes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
