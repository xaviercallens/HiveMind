p = "scripts/publish_hivemind_paper_zenodo.py"
s = open(p, encoding="utf-8").read()
old = '"constraining the mechanism they are named after).</p>"'
assert s.count(old) == 1
new = ('"constraining the mechanism they are named after). The Lean file in this record is the 128-theorem version that the paper and the independent compile receipt describe; "\n'
       '        "the repository has since added 13 theorems (an unrelated change), which this record does not cover.</p>"')
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
print("ok")
