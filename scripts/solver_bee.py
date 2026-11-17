#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Solver Bee service launcher (loopback only by default). See docs/solver_bee/SERVICE.md.

    solver_bee.py serve --binary PATH [--problems DIR] [--port 8100] [--host 127.0.0.1] [--max-concurrent 2] [--timeout 30]
                        [--allow-unsandboxed] [--nonlocal --api-key-file FILE]

Exit codes: 0 clean stop, 3 usage / environment error.
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hivemind.solverbee import sandbox, service  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the HTTP service")
    s.add_argument("--binary", required=True, help="path of the hivemind-solver executable")
    s.add_argument("--problems", help="directory of hivemind-solver-problem/1 files (references for the verifier)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8100)
    s.add_argument("--max-concurrent", type=int, default=2)
    s.add_argument("--timeout", type=float, default=30.0, help="wall-clock seconds per solver run")
    s.add_argument("--allow-unsandboxed", action="store_true", help="run WITHOUT namespaces when they are unavailable (flagged sandbox:'none' everywhere)")
    s.add_argument("--nonlocal", "--allow-nonlocal", dest="nonlocal_", action="store_true", help="allow a non-loopback bind (requires --api-key-file)")
    s.add_argument("--api-key-file", help="file holding the Bearer token (>= 16 chars)")
    return ap


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        key = service.check_bind(args.host, args.nonlocal_, args.api_key_file)
        if args.max_concurrent < 1 or args.timeout <= 0:
            raise service.ServiceError("--max-concurrent must be >= 1 and --timeout > 0")
        if not Path(args.binary).is_file():
            raise service.ServiceError("--binary is not a file")
        if args.problems and not Path(args.problems).is_dir():
            raise service.ServiceError("--problems is not a directory")
    except service.ServiceError as exc:
        print(f"solver_bee: {exc}", file=sys.stderr)
        return 3
    if not sandbox.namespace_available() and not args.allow_unsandboxed:
        print("solver_bee: warning: user namespaces unavailable; every solve will be refused (UNVERIFIED) unless --allow-unsandboxed is given", file=sys.stderr)
    svc = service.SolverBeeService(args.binary, problems_dir=args.problems, max_concurrent=args.max_concurrent, timeout_s=args.timeout,
                                   allow_unsandboxed=args.allow_unsandboxed, api_key=key, loopback_only=service._is_loopback_host(args.host),
                                   log=lambda m: print(m, file=sys.stderr, flush=True))
    httpd = service.build_server(svc, args.host, args.port)

    def _stop(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _stop)
    print(f"solver_bee: listening on {args.host}:{httpd.server_address[1]} (max {args.max_concurrent} concurrent, auth={'on' if key else 'off'})", file=sys.stderr, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    return cmd_serve(args)


if __name__ == "__main__":
    sys.exit(main())
