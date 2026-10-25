p = "scripts/publish_hivemind_paper_zenodo.py"
s = open(p, encoding="utf-8").read()

# 1. default version
old = 'ap.add_argument("--version", default="3.7.2")'
assert s.count(old) == 1
s = s.replace(old, 'ap.add_argument("--version", default="3.8.0")')

# 2. bundle: extend with the T4 campaign evidence (flat unique names)
marker = '    PROJECT_ROOT / "CITATION.cff",\n]\n'
assert s.count(marker) == 1
extra = '''    PROJECT_ROOT / "CITATION.cff",
    PROJECT_ROOT / "artifacts" / "gwaya_lite_live_check.json",
    PROJECT_ROOT / "docs" / "benchmarks" / "T4_10h_campaign_report.md",
    PROJECT_ROOT / "docs" / "review" / "t4_branch_review_2026-10-24.md",
    PROJECT_ROOT / "docs" / "community" / "GWAYA_T4_DEPLOYMENT.md",
    PROJECT_ROOT / "docs" / "community" / "GWAYA_LITE_ASSISTANT.md",
    PROJECT_ROOT / "docs" / "community" / "T4_REPLICATION_KIT.md",
]
# v3.8.0: every receipt behind E10b, E11, E11b, E12 (T4), E13, E14, E15 and E16 (preregistrations, amendments, raw and reports); partial files excluded
_T4_PREFIXES = ("E10b_", "E11_", "E11b_", "E12_replay_", "E13_", "E14_", "E15_", "E16_", "T4_benchmark_")
BUNDLE += sorted(
    p for p in (PROJECT_ROOT / "artifacts" / "experiments").glob("*.json")
    if p.name.startswith(_T4_PREFIXES) and not p.name.endswith(".partial.json") and p not in BUNDLE
)
'''
s = s.replace(marker, extra)

# 3. description
a = s.index("    desc = (")
end_marker = '        f"<p>{doi_note}</p>"\n    )'
b = s.index(end_marker) + len(end_marker)
new = '''    desc = (
        "<p>Preprint and evidence bundle of <b>Project HiveMind</b>: a system design for decentralized sparse "
        "Mixture-of-Experts training with DiLoCo-style outer optimization on 8 GB consumer GPUs, integrating "
        "Decoupled DiLoCo quorum aggregation (Google DeepMind), PA-CGAD staleness gating, FoMoE partial expert "
        "replication, DeMo/SparseLoCo 2-bit compression, MuLoCo inner preconditioning and a looped MoE backbone, "
        "with 128 Lean 4 theorems that compile without sorry (bookkeeping lemmas; two of them are flagged in the paper as not "
        "constraining the mechanism they are named after).</p>"
        "<p><b>Scope statement (please read before citing).</b> This preprint has <b>not</b> been peer reviewed. "
        "All experiments ran on a single host or, for E8-pilot, on two hosts over the public internet. E1-E4 use a single seed and "
        "show that the mechanisms behave as specified; they do not estimate effect sizes. From E5 on the experiments are pre-registered with "
        "controls (three seeds where stated). Version 3.8.0 adds the single-GPU campaign on a Tesla T4: E10b (invalid: a control failed), "
        "E11 (lattice codecs for the KV cache; the Qwen2.5-Coder-7B row is invalid because its negative control failed and its verdicts are withdrawn), "
        "E12 (replay divergence), E13 (a swarm simulated on one GPU over localhost HTTP, peers are threads sharing one model except in one run; "
        "the registered viability prediction P13b is refuted), E14 (fair quorum, supported on a dense 0.5B model, inconclusive on Granite-1B), "
        "E15 and E16 (a verification-gated coding assistant under 8 GB). Refuted, inconclusive and invalid outcomes are reported as such. "
        "No wide-area, multi-host training claim has been measured. The JSON artifacts in this bundle are the sole source of every number in "
        "the PDF (provenance lock PROV-01..PROV-73 in paper/autoevolve_peer_review_pipeline.py). Source commit: "
        + SOURCE_COMMIT + " of the HiveMind repository on GitHub (xaviercallens/HiveMind).</p>"
        f"<p>{doi_note}</p>"
    )'''
s = s[:a] + new + s[b:]

# 4. source commit constant, supplied by the operator through the environment
anchor = 'RECEIPT_PATH = PROJECT_ROOT / "paper" / "ZENODO_DEPOSIT_RECEIPT.json"'
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + '\nSOURCE_COMMIT = os.environ.get("HIVEMIND_SOURCE_COMMIT", "unspecified")')
open(p, "w", encoding="utf-8").write(s)
print("patched")
