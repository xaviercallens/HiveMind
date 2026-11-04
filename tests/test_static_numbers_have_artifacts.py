# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""MVP criterion F1 guard: public page modules must not gain new typed-in figures (latency, speed, request counts) that have no artifact behind them.

The page modules already contain such figures from earlier reports (``tests/data/page_literals_baseline.json`` records how many per file). This is a RATCHET:
the count per file may only go down; a new typed-in figure fails the test. To keep a figure, prove it (put it in ``measured_results.json`` with its artifact and render it
from there) or lower the baseline when you delete one: ``python tests/test_static_numbers_have_artifacts.py --write-baseline``."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted((ROOT / "src/hivemind/router").glob("*page*.py")) + [ROOT / "src/hivemind/router/life_as_a_bee.py"]
BASELINE = Path(__file__).with_name("data") / "page_literals_baseline.json"
# a number with a unit that makes it a performance or traffic claim
FIGURE = re.compile(r"\d[\d.,]*\s?(?:ms\b|tok/s|HTTP Reqs|Completions|\bs/round)")


def count_figures(text: str) -> int:
    return len(FIGURE.findall(text))


def current() -> dict[str, int]:
    return {p.name: count_figures(p.read_text(encoding="utf-8", errors="ignore")) for p in PAGES if p.exists()}


def test_page_modules_gain_no_new_typed_in_figures() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    for name, n in current().items():
        assert n <= baseline.get(name, 0), (
            f"{name} has {n} typed-in figures, baseline {baseline.get(name, 0)}: render new figures from measured_results.json, "
            "or run --write-baseline after DELETING figures"
        )


def test_the_guard_can_fail() -> None:
    clean = "<p>Latency not measured</p>"
    assert count_figures(clean) == 0
    assert count_figures(clean + "<p>p50 198.1 ms and 71.87 tok/s</p>") == 2  # negative control: an injected figure is seen


def test_the_benchmarks_page_has_none() -> None:
    assert current().get("benchmarks_page.py", 0) == 0


if __name__ == "__main__":
    if "--write-baseline" in sys.argv:
        BASELINE.parent.mkdir(exist_ok=True)
        BASELINE.write_text(json.dumps(current(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(open(BASELINE).read())
