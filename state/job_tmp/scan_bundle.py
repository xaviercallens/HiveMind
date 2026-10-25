import importlib.util
import re

sp = importlib.util.spec_from_file_location("z", "scripts/publish_hivemind_paper_zenodo.py")
z = importlib.util.module_from_spec(sp)
sp.loader.exec_module(z)
pats = [r"callensxavier", r"gmail", r"/home/[A-Za-z0-9_]+", r"hm-[A-Za-z0-9_-]{16,}", r"ghp_[A-Za-z0-9]{20,}", r"AIza[0-9A-Za-z_-]{20,}", r"BEGIN [A-Z ]*PRIVATE KEY", r"Bearer [A-Za-z0-9._-]{20,}"]
hits = {}
for f in z.BUNDLE:
    if f.suffix in (".pdf", ".pt"):
        continue
    t = f.read_text(encoding="utf-8", errors="ignore")
    for p in pats:
        n = len(re.findall(p, t))
        if n:
            hits.setdefault(f.name, []).append((p, n))
for k, v in sorted(hits.items()):
    print(k, v)
print("files with hits:", len(hits))
