"""The launch-wave model is a projection with stated assumptions, not a simulation of observed data (IP-22, IP-23)."""

from __future__ import annotations

import ast
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "src" / "hivemind" / "economy" / "international_launch_model.py"


def _defined_and_referenced_names(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def test_no_identifier_in_the_launch_model_starts_with_simulate() -> None:
    names = _defined_and_referenced_names(MODULE.read_text(encoding="utf-8"))
    assert sorted(n for n in names if n.startswith("simulate_")) == []


def test_the_projection_is_exported_under_its_new_name() -> None:
    names = _defined_and_referenced_names(MODULE.read_text(encoding="utf-8"))
    assert "project_international_launch_wave" in names
