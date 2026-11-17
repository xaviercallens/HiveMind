#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Solver Bee MCP server over stdio (Model Context Protocol, newline-delimited JSON-RPC 2.0). See docs/solver_bee/MCP.md.

    solver_bee_mcp.py --binary PATH [--problems DIR] [--timeout 15] [--allow-unsandboxed]

Stdout carries protocol messages only; logs and diagnostics go to stderr. Without user namespaces the server refuses to start unless
--allow-unsandboxed is given, and then every result says sandbox 'none'.

Exit codes: 0 end of input, 3 usage or environment error.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hivemind.solverbee import mcp, sandbox  # noqa: E402

DEFAULT_PROBLEMS = ROOT / "solvers" / "problems"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True, help="path of the hivemind-solver executable (pinned build)")
    ap.add_argument("--problems", default=str(DEFAULT_PROBLEMS) if DEFAULT_PROBLEMS.is_dir() else None,
                    help="directory of hivemind-solver-problem/1 files, read-only (default: solvers/problems in this repository)")
    ap.add_argument("--timeout", type=float, default=mcp.DEFAULT_TIMEOUT_S, help="wall-clock seconds per solver run (a solve makes at most two)")
    ap.add_argument("--allow-unsandboxed", action="store_true",
                    help="run the solver WITHOUT namespaces (rlimits and a scrubbed environment only); every result is flagged sandbox 'none'")
    return ap


def main(argv: list | None = None, stdin=None, stdout=None) -> int:
    args = build_parser().parse_args(argv)
    if not Path(args.binary).is_file():
        print("solver_bee_mcp: --binary is not a file", file=sys.stderr)
        return 3
    if args.problems is not None and not Path(args.problems).is_dir():
        print("solver_bee_mcp: --problems is not a directory", file=sys.stderr)
        return 3
    if not args.timeout > 0:
        print("solver_bee_mcp: --timeout must be > 0", file=sys.stderr)
        return 3
    if not args.allow_unsandboxed and not sandbox.namespace_available():
        print("solver_bee_mcp: user namespaces are unavailable (" + (sandbox.availability_reason() or "probe failed") + "). "
              "Refusing to run the solver unsandboxed; pass --allow-unsandboxed to accept sandbox 'none'.", file=sys.stderr)
        return 3
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="solver_bee_mcp %(levelname)s %(message)s")
    server = mcp.build_server(args.binary, problems_dir=args.problems, timeout_s=args.timeout, allow_unsandboxed=args.allow_unsandboxed)

    # Stdout purity: the protocol gets a private copy of fd 1; fd 1 and sys.stdout now point at stderr, so a stray print or an inheriting
    # child cannot corrupt the protocol stream.
    if stdout is None:
        proto_fd = os.dup(1)
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        stdout = os.fdopen(proto_fd, "wb", buffering=0)
    src = stdin if stdin is not None else sys.stdin.buffer
    logging.getLogger("solver_bee_mcp").info("ready: sandbox=%s problems=%s", server.sandbox_mode, args.problems or "none")
    while True:
        raw = mcp.read_frame(src)
        if raw is None:
            break
        if not raw.strip():
            continue
        resp = server.handle_line(raw)
        if resp is not None:
            stdout.write(mcp.encode(resp))
            stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
