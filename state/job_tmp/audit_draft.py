import importlib.util

sp = importlib.util.spec_from_file_location("z", "scripts/publish_hivemind_paper_zenodo.py")
z = importlib.util.module_from_spec(sp)
sp.loader.exec_module(z)
tok = z.load_token()
dep = z.get_deposition(23649757, tok)
files = {f["filename"]: f.get("checksum", "") for f in dep.get("files", [])}
want = {f.name: z.md5(f) for f in z.BUNDLE}
bad = [n for n, m in want.items() if files.get(n) != m]
carried_bad = [n for n, m in z.CARRIED.items() if files.get(n) != m]
extra = sorted(set(files) - set(want) - set(z.CARRIED))
missing = sorted((set(want) | set(z.CARRIED)) - set(files))
md = dep["metadata"]
print("state:", dep["state"], "| submitted:", dep.get("submitted"))
print("files on draft:", len(files), "| expected:", len(want) + len(z.CARRIED))
print("checksum mismatches (uploaded):", bad)
print("checksum mismatches (carried over):", carried_bad)
print("unexpected extra files:", extra)
print("missing files:", missing)
print("version:", md.get("version"), "| license:", md.get("license"), "| prereserved DOI:", md.get("prereserve_doi", {}).get("doi"))
d = md.get("description", "")
for k in ("128 Lean 4 theorems", "PROV-01..PROV-73", "withdrawn", "13 theorems", "fefa68eb582b", "not</b> been peer reviewed"):
    print("description contains", repr(k), ":", k in d)
