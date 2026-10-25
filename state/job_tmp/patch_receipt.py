import json

R = "paper/ZENODO_DEPOSIT_RECEIPT.json"
r = json.load(open(R, encoding="utf-8"))
assert r["doi"] == "10.5281/zenodo.23649757" and r["verified"] is True, r.get("doi")
r["verification_note"] = (
    "Published 2026-10-25 as 10.5281/zenodo.23649757 (version 3.8.0, 99 files), new version of 10.5281/zenodo.23607597. "
    "Every uploaded file matches the local tree (checksum_mismatches empty) and the eight raw E12 replay tensors (e12/*.pt) were carried over "
    "unchanged from v3.7.1 and verified against the md5 values in paper/zenodo_carried_over_v3.7.1.json. "
    "Source: main at fefa68eb582b (PR #150 merged) plus the DOI footer and the deposit script. "
    "Known differences from main: (1) lean4/HiveMindSpec.lean in the record is the 128-theorem version (md5 59e17dfc55d0101ed88dbf5c0c801db3, identical to the v3.7.1 file) "
    "that the paper and the independent compile receipt describe; main has since added 13 theorems (commit cffc194, QueenBee), not covered by this record; "
    "paper/PEER_REVIEW_REPORT.json was generated against the 128-theorem file. (2) The run logs T4_benchmark_queue_log.json and the E10b/E11/E11b partial files are not in the record "
    "(the log carries the machine username). (3) The E11 Qwen2.5-Coder-7B row is withdrawn in the paper (negative control failed); its raw receipt is in the record."
)
json.dump(r, open(R, "w", encoding="utf-8"), indent=2)
print("receipt note added; files in record:", len(r["remote_md5"]))
