p = "CHANGELOG.md"
s = open(p, encoding="utf-8").read()
head = "# Changelog\n"
assert s.startswith(head)
entry = (
    "\n## Zenodo deposit of manuscript v3.8.0 — 2026-10-25\n\n"
    "- Published as 10.5281/zenodo.23649757 (new version of 10.5281/zenodo.23607597, concept DOI 10.5281/zenodo.23597697; 99 files; checksums verified against the local tree and, for the eight carried-over E12 replay tensors, against the v3.7.1 receipt). "
    "Not peer reviewed; the record bundles the E10b-E16 preregistrations, amendments, raw data and reports, the GWAYA-Lite live check and the T4 campaign report.\n"
    "- `scripts/publish_hivemind_paper_zenodo.py`: default version 3.8.0, extended bundle, an honest description (128 theorems, invalid rows named), carried-over file verification, and a filter that keeps run logs containing the machine username out of the public record. `paper/zenodo_carried_over_v3.7.1.json` lists the carried-over md5 values.\n"
    "- Known differences from main (also in `paper/ZENODO_DEPOSIT_RECEIPT.json`): the Lean file in the record is the 128-theorem version the paper describes (main has since added 13 theorems); the T4 queue run log is not in the record.\n"
)
open(p, "w", encoding="utf-8").write(head + entry + s[len(head):])
print("ok")
