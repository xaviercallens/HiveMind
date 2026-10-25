p = "scripts/publish_hivemind_paper_zenodo.py"
s = open(p, encoding="utf-8").read()
old = '    if p.name.startswith(_T4_PREFIXES) and not p.name.endswith(".partial.json") and p not in BUNDLE\n)\n'
assert s.count(old) == 1
new = ('    if p.name.startswith(_T4_PREFIXES) and not p.name.endswith(".partial.json") and p not in BUNDLE\n'
       '    and "callensxavier_gmail_com" not in p.read_text(encoding="utf-8", errors="ignore")  # run logs carrying the machine username stay out of the public record\n)\n')
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
print("ok")
