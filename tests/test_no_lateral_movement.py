# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""PS-02-D3 — No-lateral-movement static and runtime guard for volunteer node binaries.

Verifies that:
1. Volunteer node entrypoints (``hivemind_producer_agent.py``, ``hivemind_producer_shield.py``,
   ``hivemind_boinc_idle_worker.py``, ``hivemind_opentrainer_worker.py``, ``hivemind_opentrainer_agent.py``)
   contain zero outbound connections to LAN peers (RFC1918), arbitrary third-party hosts, raw
   ``socket.connect`` / ``socket.create_connection`` calls, or lateral-movement libraries/binaries
   (``paramiko``, ``ssh``, ``nmap``, ``smbclient``, etc.). Only loopback (``127.0.0.1`` / ``localhost``),
   the configured HiveMind gateway (``*.run.app``), and ``storage.googleapis.com`` are permitted.
2. ``hivemind_producer_shield.py`` refuses non-loopback ``--upstream`` and ``--route`` targets at runtime.
3. Negative control: when a planted outbound connection (LAN peer URL, external URL, raw socket connect,
   or SSH library import) is injected into any of the audited files, the scanner flags it.
"""

from __future__ import annotations

import ast
import ipaddress
from pathlib import Path
import re
from urllib.parse import urlparse

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

AUDITED_VOLUNTEER_FILES: tuple[str, ...] = (
    "src/hivemind/router/hivemind_producer_agent.py",
    "src/hivemind/router/hivemind_producer_shield.py",
    "scripts/hivemind_boinc_idle_worker.py",
    "scripts/hivemind_opentrainer_worker.py",
    "scripts/hivemind_opentrainer_agent.py",
)

FORBIDDEN_IMPORTS: frozenset[str] = frozenset(
    {
        "paramiko",
        "fabric",
        "asyncssh",
        "telnetlib",
        "smbprotocol",
        "pywinrm",
        "impacket",
        "scapy",
        "nmap",
        "ftplib",
    }
)

FORBIDDEN_SUBPROCESS_BINARIES: frozenset[str] = frozenset(
    {
        "ssh",
        "scp",
        "sftp",
        "nmap",
        "nc",
        "netcat",
        "ncat",
        "smbclient",
        "winrm",
        "arp",
        "masscan",
    }
)

ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_DOC_PLACEHOLDERS: frozenset[str] = frozenset({"host", "gateway", "your_public_url", "127.0.0.1:port"})

_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def _docstring_node_ids(tree: ast.AST) -> set[int]:
    """Return AST node ids of module, class, and function docstring Constant nodes."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                val = node.body[0].value
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    ids.add(id(val))
    return ids


def _is_allowed_host(host: str) -> bool:
    h = host.strip().lower().strip("[]")
    if not h or h in ALLOWED_LOOPBACK_HOSTS or h in ALLOWED_DOC_PLACEHOLDERS:
        return True
    if h.endswith(".run.app") or h == "storage.googleapis.com" or h.endswith(".googleapis.com"):
        return True
    return False


def audit_source_for_lateral_movement(source_text: str, filename: str = "<memory>") -> list[str]:
    """Return a list of lateral-movement or unauthorized-egress violations found in ``source_text``."""
    tree = ast.parse(source_text, filename=filename)
    doc_ids = _docstring_node_ids(tree)
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                if root_pkg in FORBIDDEN_IMPORTS:
                    violations.append(f"{filename}:{node.lineno}: forbidden lateral-movement import {alias.name!r}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            root_pkg = node.module.split(".")[0]
            if root_pkg in FORBIDDEN_IMPORTS:
                violations.append(f"{filename}:{node.lineno}: forbidden lateral-movement import from {node.module!r}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr in ("connect", "connect_ex", "create_connection", "sendto"):
                violations.append(f"{filename}:{node.lineno}: raw socket call .{node.func.attr}() is forbidden in volunteer node code")
            if node.args and isinstance(node.args[0], (ast.List, ast.Tuple)) and node.args[0].elts:
                first_elt = node.args[0].elts[0]
                if isinstance(first_elt, ast.Constant) and isinstance(first_elt.value, str):
                    cmd_bin = Path(first_elt.value).name.lower()
                    if cmd_bin in FORBIDDEN_SUBPROCESS_BINARIES:
                        violations.append(f"{filename}:{node.lineno}: forbidden lateral-movement subprocess binary {cmd_bin!r}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in doc_ids:
            text = node.value.strip()
            if "://" in text:
                for token in text.split():
                    if "://" not in token:
                        continue
                    cleaned = token.strip(",;()'\"")
                    if "=" in cleaned and not cleaned.startswith(("http://", "https://", "ws://", "wss://")):
                        cleaned = cleaned.split("=", 1)[1]
                    parsed = urlparse(cleaned)
                    if parsed.scheme in ("http", "https", "ws", "wss", "ftp", "ssh", "smb"):
                        host = (parsed.hostname or "").lower()
                        if not _is_allowed_host(host):
                            violations.append(f"{filename}:{node.lineno}: unauthorized outbound URL target {cleaned!r} (host={host!r})")
            for ip_match in _IPV4_RE.findall(text):
                try:
                    ip_obj = ipaddress.ip_address(ip_match)
                except ValueError:
                    continue
                if not ip_obj.is_loopback and (ip_obj.is_private or ip_obj.is_link_local or not _is_allowed_host(ip_match)):
                    violations.append(f"{filename}:{node.lineno}: forbidden non-loopback IP literal {ip_match!r}")

    return violations


@pytest.mark.parametrize("rel_path", AUDITED_VOLUNTEER_FILES)
def test_volunteer_binaries_have_zero_lateral_movement_or_unauthorized_egress(rel_path: str) -> None:
    """Every volunteer binary only connects to loopback, the gateway (*.run.app), or GCS."""
    full_path = REPO_ROOT / rel_path
    source = full_path.read_text(encoding="utf-8")
    violations = audit_source_for_lateral_movement(source, filename=rel_path)
    assert violations == [], f"Unexpected lateral movement / egress violations in {rel_path}: {violations}"


def test_planted_outbound_connections_fail_the_scanner() -> None:
    """Negative control (PS-02-D3): planting a LAN peer URL, external URL, raw socket connect, or SSH import fails."""
    clean_source = (REPO_ROOT / "src/hivemind/router/hivemind_producer_agent.py").read_text(encoding="utf-8")
    assert audit_source_for_lateral_movement(clean_source, "hivemind_producer_agent.py") == []

    planted_snippets = [
        '\n_PLANTED_LAN = "http://192.168.1.42:11434/api/tags"\n',
        '\n_PLANTED_EXTERNAL = "https://c2.attacker.example.com/beacon"\n',
        '\nimport socket as _s\n_s.create_connection(("10.0.0.55", 22))\n',
        "\nimport paramiko\n",
        '\nimport subprocess as _sp\n_sp.run(["ssh", "192.168.1.10", "id"])\n',
    ]
    for snippet in planted_snippets:
        poisoned = clean_source + snippet
        found = audit_source_for_lateral_movement(poisoned, "hivemind_producer_agent.py")
        assert len(found) >= 1, f"Scanner failed to catch planted lateral-movement snippet: {snippet!r}"


def test_producer_shield_rejects_non_loopback_upstream_and_routes_at_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime guard: hivemind_producer_shield.main refuses non-loopback --upstream or --route targets."""
    from hivemind.router import hivemind_producer_shield as shield

    monkeypatch.setenv("HIVEMIND_ENDPOINT_TOKEN", "secret-token-1234567890")
    with pytest.raises(SystemExit):
        shield.main(["--upstream", "http://192.168.1.42:11434"])

    with pytest.raises(SystemExit):
        shield.main(["--route", "coder=http://10.0.0.55:8080"])

