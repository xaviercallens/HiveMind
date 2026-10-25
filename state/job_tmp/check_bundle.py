import collections
import importlib.util

sp = importlib.util.spec_from_file_location("z", "scripts/publish_hivemind_paper_zenodo.py")
z = importlib.util.module_from_spec(sp)
sp.loader.exec_module(z)
names = [f.name for f in z.BUNDLE]
print("files", len(names))
print("duplicate names", [n for n, c in collections.Counter(names).items() if c > 1])
print("missing", [str(f) for f in z.BUNDLE if not f.exists()])
print("MB", round(sum(f.stat().st_size for f in z.BUNDLE if f.exists()) / 1e6, 1))
print("version default ok; source commit const:", z.SOURCE_COMMIT)
