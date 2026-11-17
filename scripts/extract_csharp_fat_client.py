#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Extract and verify ``NATIVE_CSHARP_FAT_CLIENT_SOURCE`` from ``fat_steam_astro_builder.py`` (#564).

Writes ``HiveMind_Exoplanet_Hunter.cs``, verifies lexical/brace/string/entrypoint syntax (so a broken C# string or
unbalanced delimiter fails immediately even on hosts without Mono/.NET), and invokes ``mcs`` or ``csc`` when present
(reporting an explicit ``SKIP: no C# compiler installed`` status rather than a silent pass when absent).
"""

from __future__ import annotations

import argparse
import ast
import shutil
import subprocess
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILDER_PATH = PROJECT_ROOT / "src" / "hivemind" / "game" / "fat_steam_astro_builder.py"


def extract_csharp_source(builder_path: Path = DEFAULT_BUILDER_PATH) -> str:
    """Extract ``NATIVE_CSHARP_FAT_CLIENT_SOURCE`` from ``fat_steam_astro_builder.py`` via Python AST."""
    tree = ast.parse(builder_path.read_text(encoding="utf-8"), filename=str(builder_path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "NATIVE_CSHARP_FAT_CLIENT_SOURCE":
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        return node.value.value
    raise ValueError(f"NATIVE_CSHARP_FAT_CLIENT_SOURCE not found in {builder_path}")


def validate_csharp_syntax(source: str) -> dict[str, Any]:
    """Verify balanced braces/brackets/parentheses, closed strings/comments, namespace, and Main entrypoint."""
    errors: list[str] = []
    if "namespace HiveMind.ExoplanetHunterClient" not in source:
        errors.append("missing namespace HiveMind.ExoplanetHunterClient")
    if "static void Main(" not in source:
        errors.append("missing static void Main entrypoint")

    stack: list[tuple[str, int]] = []
    pairs = {"{": "}", "(": ")", "[": "]"}
    closing = {"}": "{", ")": "(", "]": "["}
    i = 0
    n = len(source)
    line_no = 1

    while i < n:
        ch = source[i]
        if ch == "\n":
            line_no += 1
            i += 1
            continue
        # Line comment
        if ch == "/" and i + 1 < n and source[i + 1] == "/":
            nl = source.find("\n", i + 2)
            if nl == -1:
                break
            i = nl
            continue
        # Block comment
        if ch == "/" and i + 1 < n and source[i + 1] == "*":
            end = source.find("*/", i + 2)
            if end == -1:
                errors.append(f"unclosed block comment starting at line {line_no}")
                break
            line_no += source[i : end + 2].count("\n")
            i = end + 2
            continue
        # Verbatim string @"..."
        if ch == "@" and i + 1 < n and source[i + 1] == '"':
            i += 2
            closed = False
            while i < n:
                if source[i] == "\n":
                    line_no += 1
                if source[i] == '"':
                    if i + 1 < n and source[i + 1] == '"':
                        i += 2
                        continue
                    i += 1
                    closed = True
                    break
                i += 1
            if not closed:
                errors.append(f"unclosed verbatim string literal at line {line_no}")
            continue
        # Regular string "..."
        if ch == '"':
            start_line = line_no
            i += 1
            closed = False
            while i < n:
                if source[i] == "\\":
                    i += 2
                    continue
                if source[i] == "\n":
                    errors.append(f"unterminated string literal on line {start_line}")
                    break
                if source[i] == '"':
                    i += 1
                    closed = True
                    break
                i += 1
            if not closed and not errors:
                errors.append(f"unterminated string literal at EOF (started line {start_line})")
            continue
        # Character literal '...'
        if ch == "'":
            i += 1
            while i < n and source[i] != "'" and source[i] != "\n":
                if source[i] == "\\":
                    i += 2
                else:
                    i += 1
            if i < n and source[i] == "'":
                i += 1
            else:
                errors.append(f"unterminated char literal on line {line_no}")
            continue

        if ch in pairs:
            stack.append((ch, line_no))
        elif ch in closing:
            if not stack or stack[-1][0] != closing[ch]:
                errors.append(f"unmatched closing {ch!r} at line {line_no}")
            else:
                stack.pop()
        i += 1

    if stack:
        top_ch, top_line = stack[-1]
        errors.append(f"unclosed {top_ch!r} opened at line {top_line} ({len(stack)} unclosed total)")

    return {
        "valid": len(errors) == 0,
        "line_count": line_no,
        "char_count": n,
        "errors": errors,
    }


def write_and_check_csharp(
    out_path: Path,
    *,
    source_override: str | None = None,
    builder_path: Path = DEFAULT_BUILDER_PATH,
) -> dict[str, Any]:
    """Write ``HiveMind_Exoplanet_Hunter.cs``, validate syntax, and compile if ``mcs`` or ``csc`` is installed."""
    source = source_override if source_override is not None else extract_csharp_source(builder_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(source, encoding="utf-8")
    syntax = validate_csharp_syntax(source)
    if not syntax["valid"]:
        return {
            "passed": False,
            "syntax": syntax,
            "compiler": None,
            "compiled": False,
            "skipped_no_compiler": False,
            "message": f"Syntax validation failed: {syntax['errors']}",
        }

    compiler = shutil.which("mcs") or shutil.which("csc")
    if compiler is None:
        return {
            "passed": True,
            "syntax": syntax,
            "compiler": None,
            "compiled": False,
            "skipped_no_compiler": True,
            "message": "SKIP: no C# compiler (mcs/csc) installed on host; structural syntax validation passed",
        }

    out_exe = out_path.with_suffix(".exe")
    proc = subprocess.run(
        [
            compiler,
            f"-out:{out_exe}",
            "-r:System.Drawing.dll",
            "-r:System.Windows.Forms.dll",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    ok = proc.returncode == 0
    return {
        "passed": ok,
        "syntax": syntax,
        "compiler": compiler,
        "compiled": ok,
        "skipped_no_compiler": False,
        "message": (proc.stdout + "\n" + proc.stderr).strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("/tmp/HiveMind_Exoplanet_Hunter.cs"))
    args = parser.parse_args()
    result = write_and_check_csharp(args.out)
    print(result["message"])
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
