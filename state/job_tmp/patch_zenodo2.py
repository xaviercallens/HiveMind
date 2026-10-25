import json

R = "paper/ZENODO_DEPOSIT_RECEIPT.json"
rec = json.load(open(R, encoding="utf-8"))
pt = {k: v for k, v in rec["remote_md5"].items() if k.endswith(".pt")}
assert len(pt) == 8, pt
out = {
    "note": "Files carried over unchanged from Zenodo record 10.5281/zenodo.23607597 (v3.7.1) into v3.8.0; they are not in every checkout (git-ignored raw replay tensors). Their md5 values are those the v3.7.1 receipt recorded after publication. The v3.8.0 publish stage verifies the new record against these.",
    "source_record": rec["doi"],
    "md5": pt,
}
json.dump(out, open("paper/zenodo_carried_over_v3.7.1.json", "w", encoding="utf-8"), indent=2)

p = "scripts/publish_hivemind_paper_zenodo.py"
s = open(p, encoding="utf-8").read()

# drop the missing raw tensors from the upload bundle
for name in ("lora_init", "ref", "same_process_repeat", "threads_1", "threads_8", "remote_cpu", "tampered", "bf16_autocast"):
    line = f'    PROJECT_ROOT / "artifacts" / "experiments" / "e12" / "{name}.pt",\n'
    assert s.count(line) == 1, name
    s = s.replace(line, "")

# carried-over registry, placed after the bundle extension
anchor = "    if p.name.startswith(_T4_PREFIXES) and not p.name.endswith(\".partial.json\") and p not in BUNDLE\n)\n"
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + '''CARRIED_PATH = PROJECT_ROOT / "paper" / "zenodo_carried_over_v3.7.1.json"
CARRIED: Dict[str, str] = json.loads(CARRIED_PATH.read_text(encoding="utf-8"))["md5"] if CARRIED_PATH.exists() else {}
''')

# verification covers the carried-over files too
old = '    mismatches = [f.name for f in BUNDLE if remote.get(f.name) != md5(f)]\n'
assert s.count(old) == 1
s = s.replace(old, old + '    mismatches += [n for n, m in CARRIED.items() if remote.get(n) != m]\n')
old2 = '"verified": not mismatches and len(remote) == len(BUNDLE),'
assert s.count(old2) == 1
s = s.replace(old2, '"verified": not mismatches and set(remote) == ({f.name for f in BUNDLE} | set(CARRIED)),')
open(p, "w", encoding="utf-8").write(s)
print("ok", len(pt), "carried")
