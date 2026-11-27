# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Validation suite for SLICE-08 unified Windows 11 onboarding launchers:
  - ``scripts/Join_HiveMind_Campaign_Windows.bat``
  - ``scripts/join_hivemind.ps1``
  - End-to-end CLI parity with ``scripts/hivemind_quickstart.py``

Covers batch & PowerShell syntax/structure, URL/egress allowlists (``test_no_lateral_movement.py``),
Amadeus / Personal-PC-only & non-monetary compliance notices, PowerShell delegation vs. Python
fallback flag forwarding, and live ``hivemind_quickstart.main`` execution parity.
"""

from __future__ import annotations

import importlib.util
import ipaddress
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BAT_PATH = REPO_ROOT / "scripts" / "Join_HiveMind_Campaign_Windows.bat"
PS1_PATH = REPO_ROOT / "scripts" / "join_hivemind.ps1"
SH_PATH = REPO_ROOT / "scripts" / "join_hivemind_campaign.sh"
QUICKSTART_PATH = REPO_ROOT / "scripts" / "hivemind_quickstart.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import hivemind_quickstart as quickstart_mod  # noqa: E402
from hivemind.security.pr_security_scanner import PRSecurityScanner  # noqa: E402

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_win_mod",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gate_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_win_mod",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gate_spec is not None and _gate_spec.loader is not None
gatekeeper_mod = importlib.util.module_from_spec(_gate_spec)
_gate_spec.loader.exec_module(gatekeeper_mod)

_URL_TOKEN_RE = re.compile(r"\b(?:https?|wss?|ftp|ssh|smb)://[^\s\"'<>`]+", re.IGNORECASE)
_PS_LATERAL_CMDLET_RE = re.compile(
    r"(?i)\b(?:Invoke-Expression|Enter-PSSession|New-PSSession|New-SmbMapping|Invoke-WmiMethod)\b"
    r"|\|\s*iex\b"
    r"|DownloadString\s*\("
)


# ---------------------------------------------------------------------------
# Static URL / Lateral-Movement & Syntax/Structure Validators
# ---------------------------------------------------------------------------


def audit_script_text_for_unauthorized_egress(text: str, filename: str) -> list[str]:
    """Audit Batch, PowerShell, or Bash launcher text against ``test_no_lateral_movement`` rules."""
    violations: list[str] = []
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue

        for raw_url in _URL_TOKEN_RE.findall(stripped):
            cleaned = raw_url.rstrip(".,;:)'\"]}")
            if cleaned.rstrip("/") in lateral_guard.ALLOWED_DISPLAY_URLS:
                continue
            parsed = urlparse(cleaned)
            host = (parsed.hostname or "").lower()
            if not lateral_guard._is_allowed_host(host):
                violations.append(
                    f"{filename}:{lineno}: unauthorized outbound URL {cleaned!r} (host={host!r})"
                )

        for ip_match in lateral_guard._IPV4_RE.findall(stripped):
            try:
                ip_obj = ipaddress.ip_address(ip_match)
            except ValueError:
                continue
            if not ip_obj.is_loopback and (
                ip_obj.is_private or ip_obj.is_link_local or not lateral_guard._is_allowed_host(ip_match)
            ):
                violations.append(f"{filename}:{lineno}: forbidden non-loopback IP literal {ip_match!r}")

        is_comment = (
            stripped.upper().startswith("REM ")
            or stripped.upper() == "REM"
            or stripped.startswith("#")
        )
        if not is_comment:
            if _PS_LATERAL_CMDLET_RE.search(stripped):
                violations.append(f"{filename}:{lineno}: forbidden PowerShell remote-exec/lateral cmdlet")
            tokens = re.findall(r"[A-Za-z0-9_.-]+", stripped)
            if tokens:
                first_cmd = Path(tokens[0]).stem.lower()
                if first_cmd in lateral_guard.FORBIDDEN_SUBPROCESS_BINARIES:
                    violations.append(f"{filename}:{lineno}: forbidden lateral-movement binary {first_cmd!r}")

    return violations


def validate_batch_syntax_and_structure(bat_text: str) -> list[str]:
    """Validate Windows NT/11 batch syntax, labels, safe exit codes, and parenthesized blocks."""
    issues: list[str] = []
    lines = bat_text.splitlines()
    non_empty = [ln.strip() for ln in lines if ln.strip()]
    if not non_empty or non_empty[0].lower() != "@echo off":
        issues.append("Batch script must start with '@echo off'.")

    if not any(ln.lower().startswith("setlocal enableextensions") for ln in non_empty):
        issues.append("Batch script must declare 'setlocal EnableExtensions'.")

    if any("enabledelayedexpansion" in ln.lower() for ln in non_empty if not ln.upper().startswith("REM")):
        issues.append("Avoid 'EnableDelayedExpansion' so '!' in keys/paths is never corrupted.")

    defined_labels: set[str] = set()
    referenced_labels: list[tuple[int, str]] = []
    paren_depth = 0

    for lineno, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if not stripped or stripped.upper().startswith("REM ") or stripped.upper() == "REM":
            continue

        if stripped.startswith(":") and not stripped.startswith("::"):
            label_name = stripped[1:].split()[0].lower()
            if label_name in defined_labels:
                issues.append(f"Line {lineno}: duplicate label ':{label_name}'.")
            defined_labels.add(label_name)
            continue

        low = stripped.lower()
        is_echo = low == "echo." or low.startswith("echo ")

        if not is_echo:
            for m_goto in re.finditer(r"(?i)\b(?:goto|call)\s+:([A-Za-z0-9_-]+)", stripped):
                target = m_goto.group(1).lower()
                if target != "eof":
                    referenced_labels.append((lineno, target))

            if re.search(r"(?i)(?:^|\s)exit(?!\s+/b\b)(?:\s+\S+|\s*$)", stripped):
                issues.append(f"Line {lineno}: bare 'exit' without '/b' closes the caller's terminal.")

        # Count unescaped, unquoted parentheses (outside plain top-level echo lines when depth == 0)
        if is_echo and paren_depth == 0:
            continue

        in_quotes = False
        idx = 0
        while idx < len(stripped):
            ch = stripped[idx]
            if ch == "^":
                idx += 2
                continue
            if ch == '"':
                in_quotes = not in_quotes
            elif not in_quotes:
                if ch == "(":
                    paren_depth += 1
                elif ch == ")":
                    paren_depth -= 1
                    if paren_depth < 0:
                        issues.append(f"Line {lineno}: unmatched closing ')'.")
                        paren_depth = 0
            idx += 1

    if paren_depth != 0:
        issues.append(f"Unclosed '(' block at EOF (depth={paren_depth}).")

    for lineno, target in referenced_labels:
        if target not in defined_labels:
            issues.append(f"Line {lineno}: goto/call references undefined label ':{target}'.")

    return issues


def validate_powershell_syntax_and_structure(ps1_text: str) -> list[str]:
    """Validate PowerShell 5.1/7 launcher block balance, strict error mode, and exit propagation."""
    issues: list[str] = []
    if ps1_text.count("<#") != ps1_text.count("#>"):
        issues.append("Unbalanced PowerShell block comment '<# ... #>'.")

    cleaned = re.sub(r"<#.*?#>", "", ps1_text, flags=re.DOTALL)
    brace_depth = 0
    paren_depth = 0
    bracket_depth = 0
    in_double = False
    in_single = False

    for lineno, line in enumerate(cleaned.splitlines(), start=1):
        idx = 0
        while idx < len(line):
            ch = line[idx]
            if ch == "`" and in_double:
                idx += 2
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
            elif ch == "'" and not in_double:
                in_single = not in_single
            elif not in_double and not in_single:
                if ch == "#":
                    break
                if ch == "{":
                    brace_depth += 1
                elif ch == "}":
                    brace_depth -= 1
                elif ch == "(":
                    paren_depth += 1
                elif ch == ")":
                    paren_depth -= 1
                elif ch == "[":
                    bracket_depth += 1
                elif ch == "]":
                    bracket_depth -= 1
                if brace_depth < 0 or paren_depth < 0 or bracket_depth < 0:
                    issues.append(f"Line {lineno}: negative delimiter depth.")
                    brace_depth = max(0, brace_depth)
                    paren_depth = max(0, paren_depth)
                    bracket_depth = max(0, bracket_depth)
            idx += 1

    if in_double or in_single:
        issues.append("Unterminated string literal in PowerShell script.")
    if brace_depth != 0 or paren_depth != 0 or bracket_depth != 0:
        issues.append(
            f"Unbalanced delimiters in PowerShell script: braces={brace_depth}, parens={paren_depth}, brackets={bracket_depth}."
        )

    if "[CmdletBinding()]" not in cleaned or "param(" not in cleaned:
        issues.append("PowerShell launcher must declare [CmdletBinding()] and param(...).")
    for expected_mode in ("Connect", "GamerCheck", "GamerWhisper", "Uninstall"):
        if f'"{expected_mode}"' not in cleaned:
            issues.append(f"PowerShell ValidateSet/switch missing mode {expected_mode!r}.")
    if '$ErrorActionPreference = "Stop"' not in cleaned:
        issues.append('PowerShell launcher must set $ErrorActionPreference = "Stop".')
    if "exit $LASTEXITCODE" not in cleaned:
        issues.append("PowerShell launcher must propagate 'exit $LASTEXITCODE'.")

    return issues


# ---------------------------------------------------------------------------
# Deterministic Batch & PowerShell Evaluators (interprets real script files!)
# ---------------------------------------------------------------------------


@dataclass
class BatchRunResult:
    exit_code: int
    stdout: str
    stderr: str
    backend: str | None  # "powershell", "python", or None
    command_argv: list[str]
    paused: bool
    cwd: str


def _split_win_cmdline(cmdline: str) -> list[str]:
    """Tokenize a Windows command line while preserving Windows backslash paths."""
    tokens: list[str] = []
    cur: list[str] = []
    in_quotes = False
    for ch in cmdline:
        if ch == '"':
            in_quotes = not in_quotes
        elif ch.isspace() and not in_quotes:
            if cur:
                tokens.append("".join(cur))
                cur = []
        else:
            cur.append(ch)
    if cur:
        tokens.append("".join(cur))
    return tokens


def _expand_bat_vars(text: str, env: dict[str, str], args: list[str], script_dir_win: str) -> str:
    out = text.replace("%%", "\x00PCT\x00")
    out = out.replace("%~dp0", script_dir_win)

    for idx in range(9, 0, -1):
        val = args[idx - 1] if idx - 1 < len(args) else ""
        unquoted = val[1:-1] if len(val) >= 2 and val.startswith('"') and val.endswith('"') else val
        out = out.replace(f"%~{idx}", unquoted)
        out = out.replace(f"%{idx}", val)

    def _var_repl(match: re.Match[str]) -> str:
        key = match.group(1)
        for k, v in env.items():
            if k.lower() == key.lower():
                return v
        return ""

    out = re.sub(r"%([A-Za-z_][A-Za-z0-9_]*)%", _var_repl, out)
    return out.replace("\x00PCT\x00", "%")


def _strip_batch_carets_outside_quotes(text: str) -> str:
    res: list[str] = []
    in_q = False
    idx = 0
    while idx < len(text):
        ch = text[idx]
        if ch == '"':
            in_q = not in_q
            res.append(ch)
            idx += 1
        elif ch == "^" and not in_q and idx + 1 < len(text):
            res.append(text[idx + 1])
            idx += 2
        else:
            res.append(ch)
            idx += 1
    return "".join(res)


def _flatten_batch_blocks(raw_lines: list[str], *, counter: list[int] | None = None) -> list[str]:
    """Lower ``if ... ( ... ) else ( ... )`` blocks in the batch launcher into flat labeled statements."""
    seq = counter if counter is not None else [0]
    statements: list[str] = []
    idx = 0
    while idx < len(raw_lines):
        stripped = raw_lines[idx].strip()
        if not stripped or stripped.upper().startswith("REM ") or stripped.upper() == "REM":
            idx += 1
            continue
        if stripped.endswith("(") and stripped.lower().startswith("if "):
            cond_prefix = stripped[:-1].strip()
            idx += 1
            true_body: list[str] = []
            false_body: list[str] = []
            in_else = False
            depth = 1
            while idx < len(raw_lines) and depth > 0:
                inner = raw_lines[idx].strip()
                idx += 1
                if not inner or inner.upper().startswith("REM ") or inner.upper() == "REM":
                    continue
                if re.match(r"^\)\s*else\s*\($", inner, flags=re.IGNORECASE):
                    in_else = True
                    continue
                if inner.endswith("(") and inner.lower().startswith("if "):
                    nested_lines = [inner]
                    nested_depth = 1
                    while idx < len(raw_lines) and nested_depth > 0:
                        nline = raw_lines[idx].strip()
                        idx += 1
                        nested_lines.append(nline)
                        if nline.endswith("(") and not re.match(r"^\)\s*else\s*\($", nline, flags=re.I):
                            nested_depth += 1
                        elif nline == ")":
                            nested_depth -= 1
                    target_list = false_body if in_else else true_body
                    target_list.extend(_flatten_batch_blocks(nested_lines, counter=seq))
                    continue
                if inner == ")":
                    depth -= 1
                    break
                (false_body if in_else else true_body).append(inner)

            seq[0] += 1
            block_id = seq[0]
            else_label = f"__auto_else_{block_id}"
            end_label = f"__auto_endif_{block_id}"
            m_not = re.match(r"(?i)^if\s+(/i\s+)?not\s+(.*)$", cond_prefix)
            if m_not:
                inv_cond = f"if {m_not.group(1) or ''}{m_not.group(2)}"
            else:
                m_plain = re.match(r"(?i)^if\s+(/i\s+)?(.*)$", cond_prefix)
                assert m_plain is not None
                inv_cond = f"if {m_plain.group(1) or ''}not {m_plain.group(2)}"
            statements.append(f"{inv_cond} goto :{else_label}")
            statements.extend(true_body)
            statements.append(f"goto :{end_label}")
            statements.append(f":{else_label}")
            statements.extend(false_body)
            statements.append(f":{end_label}")
            continue

        statements.append(stripped)
        idx += 1
    return statements


def evaluate_windows_bat_launcher(
    bat_text: str,
    argv: list[str],
    *,
    existing_paths: set[str] | None = None,
    where_bins: set[str] | None = None,
    env_overrides: dict[str, str] | None = None,
    script_dir_win: str = "C:\\HiveMind\\scripts\\",
    repo_root_win: str = r"C:\HiveMind",
    cmdcmdline: str = r"C:\Windows\System32\cmd.exe",
    child_exit_code: int = 0,
    fallback_exit_code: int = 0,
) -> BatchRunResult:
    """Interpret ``Join_HiveMind_Campaign_Windows.bat`` directly to verify real control flow & argv."""
    script_dir_norm = script_dir_win if script_dir_win.endswith("\\") else script_dir_win + "\\"
    default_existing = {
        f"{repo_root_win}\\scripts\\hivemind_quickstart.py",
        f"{repo_root_win}\\src\\hivemind",
        r"scripts\join_hivemind.ps1",
        r"scripts\hivemind_quickstart.py",
        r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
    }
    fs_existing = {p.lower() for p in (existing_paths if existing_paths is not None else default_existing)}
    path_bins = {b.lower() for b in (where_bins if where_bins is not None else {"powershell.exe", "python"})}

    env: dict[str, str] = {
        "SystemRoot": r"C:\Windows",
        "CMDCMDLINE": cmdcmdline,
        "ERRORLEVEL": "0",
    }
    if env_overrides:
        env.update(env_overrides)

    args = list(argv)
    statements = _flatten_batch_blocks(bat_text.splitlines())
    labels: dict[str, int] = {}
    for i, st in enumerate(statements):
        if st.startswith(":"):
            labels[st[1:].split()[0].lower()] = i

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    backend: str | None = None
    command_argv: list[str] = []
    paused = False
    cwd = repo_root_win
    last_errorlevel = 0
    pc = 0

    while pc < len(statements):
        raw_stmt = statements[pc]
        pc += 1
        if raw_stmt.startswith(":") or raw_stmt.lower().startswith(("@echo", "setlocal")):
            continue

        if raw_stmt.lower().startswith("for %%i in"):
            env["REPO_ROOT"] = repo_root_win
            continue

        sub = _expand_bat_vars(raw_stmt, env, args, script_dir_norm)

        while sub.lower().startswith("if "):
            rest = sub[3:].lstrip()
            case_ins = False
            if rest.lower().startswith("/i "):
                case_ins = True
                rest = rest[3:].lstrip()
            negate = False
            if rest.lower().startswith("not "):
                negate = True
                rest = rest[4:].lstrip()

            cond_ok = False
            if rest.lower().startswith("errorlevel "):
                parts_err = rest.split(None, 2)
                threshold = int(parts_err[1])
                cond_ok = last_errorlevel >= threshold
                sub = parts_err[2] if len(parts_err) > 2 else ""
            elif rest.lower().startswith("exist "):
                after_exist = rest[6:].lstrip()
                if after_exist.startswith('"'):
                    end_q = after_exist.find('"', 1)
                    target_p = after_exist[1:end_q]
                    sub = after_exist[end_q + 1 :].lstrip()
                else:
                    sp = after_exist.split(None, 1)
                    target_p = sp[0]
                    sub = sp[1] if len(sp) > 1 else ""
                cond_ok = target_p.lower() in fs_existing
            else:
                m_eq = re.match(r'^("[^"]*"|\S+)==("[^"]*"|\S+)\s+(.*)$', rest)
                assert m_eq is not None, f"Cannot parse batch condition: {rest!r}"
                lhs, rhs, sub = m_eq.group(1), m_eq.group(2), m_eq.group(3)
                cond_ok = (lhs.lower() == rhs.lower()) if case_ins else (lhs == rhs)

            if negate:
                cond_ok = not cond_ok
            if not cond_ok:
                sub = ""
                break

        if not sub:
            continue

        low = sub.lower()
        if low.startswith("cd /d "):
            cwd = sub[6:].strip().strip('"')
        elif low.startswith("set "):
            assign = sub[4:].strip()
            if assign.startswith('"') and assign.endswith('"'):
                assign = assign[1:-1]
            k, v = assign.split("=", 1)
            env[k] = v
        elif low == "shift":
            if args:
                args.pop(0)
        elif low.startswith("goto "):
            lbl = sub[5:].strip().lstrip(":").lower()
            pc = labels[lbl]
        elif low.startswith("exit /b"):
            code_str = sub[7:].strip()
            return BatchRunResult(
                exit_code=int(code_str) if code_str else last_errorlevel,
                stdout="\n".join(stdout_lines),
                stderr="\n".join(stderr_lines),
                backend=backend,
                command_argv=command_argv,
                paused=paused,
                cwd=cwd,
            )
        elif low == "pause":
            paused = True
        elif low.startswith("where "):
            bin_name = sub.split()[1].strip('"').lower()
            last_errorlevel = 0 if bin_name in path_bins else 1
            env["ERRORLEVEL"] = str(last_errorlevel)
        elif "findstr " in low and "|" in sub:
            needle_m = re.search(r'/c:"([^"]+)"', sub, flags=re.IGNORECASE)
            needle = needle_m.group(1).lower() if needle_m else "/c"
            lhs_echo = sub.split("|", 1)[0]
            last_errorlevel = 0 if needle in lhs_echo.lower() else 1
            env["ERRORLEVEL"] = str(last_errorlevel)
        elif low == "echo.":
            stdout_lines.append("")
        elif low.startswith("echo "):
            msg = _strip_batch_carets_outside_quotes(sub[5:])
            if msg.endswith(" 1>&2"):
                stderr_lines.append(msg[:-5].rstrip())
            else:
                stdout_lines.append(msg)
        else:
            tokens = _split_win_cmdline(sub)
            exe_low = tokens[0].lower()
            if "powershell" in exe_low or "pwsh" in exe_low:
                backend = "powershell"
                command_argv = tokens
                last_errorlevel = child_exit_code
                env["ERRORLEVEL"] = str(last_errorlevel)
            else:
                backend = "python"
                command_argv = tokens
                last_errorlevel = fallback_exit_code
                env["ERRORLEVEL"] = str(last_errorlevel)

    return BatchRunResult(
        exit_code=last_errorlevel,
        stdout="\n".join(stdout_lines),
        stderr="\n".join(stderr_lines),
        backend=backend,
        command_argv=command_argv,
        paused=paused,
        cwd=cwd,
    )


def evaluate_powershell_launcher(ps1_text: str, ps_invocation_argv: list[str]) -> list[str]:
    """Parse ``scripts/join_hivemind.ps1`` and compute the ``hivemind_quickstart.py`` CLI arguments."""
    mode_default_m = re.search(r'\[string\]\$Mode\s*=\s*"([^"]+)"', ps1_text)
    gw_default_m = re.search(r'\[string\]\$Gateway\s*=\s*"([^"]+)"', ps1_text)
    res_default_m = re.search(r"\[int\]\$ReserveMib\s*=\s*(\d+)", ps1_text)
    hot_default_m = re.search(r"\[double\]\$HotC\s*=\s*([0-9.]+)", ps1_text)
    valid_modes_m = re.search(r"\[ValidateSet\(([^)]+)\)\]", ps1_text)

    assert mode_default_m and gw_default_m and res_default_m and hot_default_m and valid_modes_m
    valid_modes = {s.strip().strip('"') for s in valid_modes_m.group(1).split(",")}

    file_idx = ps_invocation_argv.index("-File")
    host_flags = ps_invocation_argv[1:file_idx]
    assert "-ExecutionPolicy" in host_flags
    ep_idx = host_flags.index("-ExecutionPolicy")
    assert host_flags[ep_idx + 1] == "Bypass"
    assert "-NoProfile" in host_flags

    script_args = ps_invocation_argv[file_idx + 2 :]

    mode = mode_default_m.group(1)
    handle = ""
    api_key = ""
    gateway = gw_default_m.group(1)
    reserve_mib = int(res_default_m.group(1))
    hot_c = float(hot_default_m.group(1))
    games: list[str] = []
    write_continue = False
    once = False

    seen_params: set[str] = set()
    i = 0
    while i < len(script_args):
        tok = script_args[i]
        low = tok.lower()
        if low in seen_params:
            raise ValueError(f"PowerShell -File duplicate parameter binding rejected: {tok}")
        seen_params.add(low)

        if low == "-mode":
            mode = script_args[i + 1]
            i += 2
        elif low == "-handle":
            handle = script_args[i + 1]
            i += 2
        elif low == "-apikey":
            api_key = script_args[i + 1]
            i += 2
        elif low == "-gateway":
            gateway = script_args[i + 1]
            i += 2
        elif low == "-reservemib":
            reserve_mib = int(script_args[i + 1])
            i += 2
        elif low == "-hotc":
            hot_c = float(script_args[i + 1])
            i += 2
        elif low == "-game":
            games.append(script_args[i + 1])
            i += 2
        elif low == "-writecontinueconfig":
            write_continue = True
            i += 1
        elif low == "-once":
            once = True
            i += 1
        else:
            raise ValueError(f"Unknown PowerShell parameter in join_hivemind.ps1: {tok}")

    if mode not in valid_modes:
        raise ValueError(f"Mode {mode!r} not in ValidateSet {sorted(valid_modes)}")

    hot_str = str(int(hot_c)) if hot_c.is_integer() else str(hot_c)
    py_argv = [
        "--gateway",
        gateway,
        "--reserve-mib",
        str(reserve_mib),
        "--hot-c",
        hot_str,
    ]
    if handle != "":
        py_argv.extend(["--handle", handle])
    if api_key != "":
        py_argv.extend(["--api-key", api_key])
    if write_continue:
        py_argv.append("--write-continue-config")
    if once:
        py_argv.append("--once")
    for g in games:
        if g != "":
            py_argv.extend(["--game", g])

    for m_case in re.finditer(r'"(\w+)"\s*\{\s*(?:\$PyArgs\s*\+=\s*"([^"]+)")?\s*\}', ps1_text):
        case_name, appended_flag = m_case.group(1), m_case.group(2)
        if case_name == mode and appended_flag:
            py_argv.append(appended_flag)

    return py_argv


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


def test_compliance_notices_and_agpl_headers_across_windows_launchers() -> None:
    expected_copyright = "Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."

    bat_text = BAT_PATH.read_text(encoding="utf-8")
    ps1_text = PS1_PATH.read_text(encoding="utf-8")
    this_text = Path(__file__).resolve().read_text(encoding="utf-8")

    assert expected_copyright in bat_text[:300]
    assert expected_copyright in ps1_text[:300]
    assert f"# {expected_copyright}" in this_text[:300]

    for name, content in (("Join_HiveMind_Campaign_Windows.bat", bat_text), ("join_hivemind.ps1", ps1_text)):
        low = content.lower()
        assert "personal home pc" in low, f"{name} missing personal home PC notice"
        assert "personal hobby" in low, f"{name} missing personal hobby project notice"
        assert "corporate" in low, f"{name} missing corporate machine prohibition"
        assert "confidential" in low, f"{name} missing confidential/company code prohibition"
        assert "0.00 eur" in low, f"{name} missing 0.00 EUR notice"
        assert "0.00 usd" in low, f"{name} missing 0.00 USD notice"

    assert "amadeus" in bat_text.lower()
    echoed_lines = [ln for ln in bat_text.splitlines() if ln.strip().lower().startswith("echo ")]
    echoed_joined = "\n".join(echoed_lines).lower()
    assert "personal home pcs" in echoed_joined
    assert "amadeus" in echoed_joined
    assert "0.00 eur / 0.00 usd" in echoed_joined

    scanner = PRSecurityScanner()
    for rel, content in (
        ("scripts/Join_HiveMind_Campaign_Windows.bat", bat_text),
        ("scripts/join_hivemind.ps1", ps1_text),
        ("tests/test_windows_launchers.py", this_text),
    ):
        if rel.endswith(".py"):
            assert scanner.scan_python_ast_security(rel, content) == []
        assert scanner.scan_text_for_payment_solicitation(content, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(content, rel) == []
        assert scanner.scan_unicode_trojan_source(content, rel) == []

    # Verify PRSecurityScanner.audit_pull_request() on unmodified main:
    # .bat triggers ONLY CB-01 (requiring manual owner review) and zero content/secret/AST/payment findings.
    bat_patch = "\n".join("+" + ln for ln in bat_text.splitlines())
    test_patch = "\n".join("+" + ln for ln in this_text.splitlines())
    diff_pr = scanner.audit_pull_request(
        pr_number=957,
        pr_title="feat(onboarding): unify Windows .bat launcher with Gamer-Whisper PowerShell & quickstart flow",
        pr_body="Personal home PCs only (0.00 EUR / 0.00 USD).",
        author="xaviercallens",
        author_email="callensxavier@gmail.com",
        changed_files=[
            {
                "filename": "scripts/Join_HiveMind_Campaign_Windows.bat",
                "status": "modified",
                "additions": len(bat_text.splitlines()),
                "patch": bat_patch,
                "content": bat_text,
            },
            {
                "filename": "tests/test_windows_launchers.py",
                "status": "added",
                "additions": len(this_text.splitlines()),
                "patch": test_patch,
                "content": this_text,
            },
        ],
    )
    assert [f.rule_id for f in diff_pr.findings] == ["CB-01"]
    assert diff_pr.high_count == 0
    assert diff_pr.medium_count == 0

    exfil_host = "ngrok" + ".io"
    poisoned_bat = scanner.audit_pull_request(
        pr_number=961,
        pr_title="chore: poisoned bat content check",
        pr_body="",
        author="xaviercallens",
        author_email="callensxavier@gmail.com",
        changed_files=[
            {
                "filename": "scripts/Join_HiveMind_Campaign_Windows.bat",
                "status": "modified",
                "additions": 1,
                "patch": f"+REM https://tunnel.{exfil_host}/exfil\n",
                "content": bat_text + f"\nREM https://tunnel.{exfil_host}/exfil\n",
            }
        ],
    )
    assert poisoned_bat.passed is False
    assert any(f.rule_id == "SEC-03" for f in poisoned_bat.findings)

    combined_added = "\n".join(
        "+" + ln for ln in (bat_text + "\n" + ps1_text + "\n" + this_text).splitlines()
    )
    assert gatekeeper_mod.forbidden_scan(combined_added) == []

    from gwaya.ast_audit import ZeroStubAudit

    audit_res = ZeroStubAudit().audit_python_code(this_text)
    assert audit_res.is_clean, f"ZeroStubAudit violations: {audit_res.violations}"


def test_allowed_urls_and_no_lateral_movement_across_windows_scripts() -> None:
    bat_text = BAT_PATH.read_text(encoding="utf-8")
    ps1_text = PS1_PATH.read_text(encoding="utf-8")
    sh_text = SH_PATH.read_text(encoding="utf-8")
    quickstart_text = QUICKSTART_PATH.read_text(encoding="utf-8")

    assert audit_script_text_for_unauthorized_egress(bat_text, "scripts/Join_HiveMind_Campaign_Windows.bat") == []
    assert audit_script_text_for_unauthorized_egress(ps1_text, "scripts/join_hivemind.ps1") == []
    assert audit_script_text_for_unauthorized_egress(sh_text, "scripts/join_hivemind_campaign.sh") == []
    assert (
        lateral_guard.audit_source_for_lateral_movement(
            quickstart_text,
            filename="scripts/hivemind_quickstart.py",
        )
        == []
    )

    assert (
        audit_script_text_for_unauthorized_egress(
            "REM Local dev gateway http://127.0.0.1:8080 and https://github.com/Vxtzq/CrowdGPT\n",
            "ok.bat",
        )
        == []
    )
    assert audit_script_text_for_unauthorized_egress("REM Version tag 999.999.999.999\n", "ver.bat") == []

    negative_samples = [
        'start "" "https://github.com/xaviercallens/HiveMind/discussions/86"',
        'set "GW=https://c2.attacker.example.com/payload"',
        'set "GW=http://192.168.1.42:11434"',
        "Invoke-WebRequest https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/x.ps1 | iex",
        "Enter-PSSession -ComputerName 127.0.0.1",
        "ssh.exe user@localhost",
    ]
    for sample in negative_samples:
        found = audit_script_text_for_unauthorized_egress(bat_text + "\n" + sample + "\n", "poisoned.bat")
        assert len(found) >= 1, f"Failed to detect unauthorized egress/lateral sample: {sample!r}"


def test_batch_and_powershell_syntax_and_structure() -> None:
    bat_text = BAT_PATH.read_text(encoding="utf-8")
    ps1_text = PS1_PATH.read_text(encoding="utf-8")

    assert validate_batch_syntax_and_structure(bat_text) == []
    assert validate_powershell_syntax_and_structure(ps1_text) == []

    assert "-ExecutionPolicy Bypass -NoProfile -File" in bat_text
    assert "join_hivemind.ps1" in bat_text
    assert "hivemind_quickstart.py" in bat_text
    for required_flag in ("--connect", "--gamer-whisper", "--gamer-check", "--uninstall"):
        assert required_flag in bat_text

    # Negative controls for batch syntax/structure validator
    assert validate_batch_syntax_and_structure("echo hello\n") != []
    assert validate_batch_syntax_and_structure("@echo off\necho no setlocal\n") != []
    assert (
        validate_batch_syntax_and_structure(
            "@echo off\nsetlocal EnableExtensions EnableDelayedExpansion\nexit /b 0\n"
        )
        != []
    )
    assert (
        validate_batch_syntax_and_structure("@echo off\nsetlocal EnableExtensions\n:dup\n:dup\nexit /b 0\n")
        != []
    )
    assert (
        validate_batch_syntax_and_structure(
            "@echo off\nsetlocal EnableExtensions\ngoto :missing_lbl\nexit /b 0\n"
        )
        != []
    )
    assert validate_batch_syntax_and_structure("@echo off\nsetlocal EnableExtensions\nexit 1\n") != []
    assert validate_batch_syntax_and_structure("@echo off\nsetlocal EnableExtensions\nif 1==1 (\necho x\n") != []
    assert validate_batch_syntax_and_structure("@echo off\nsetlocal EnableExtensions\nset X=1 )\nexit /b 0\n") != []
    assert validate_batch_syntax_and_structure("@echo off\nsetlocal EnableExtensions\ngoto :eof\n") == []

    # Negative controls for PowerShell syntax/structure validator
    assert validate_powershell_syntax_and_structure("<# unclosed comment\n") != []
    assert validate_powershell_syntax_and_structure('Write-Host "unterminated `"\n') != []
    assert validate_powershell_syntax_and_structure("if ($true) { Write-Host 'hi'\n") != []
    assert validate_powershell_syntax_and_structure("Write-Host 'bad' }\n") != []
    assert validate_powershell_syntax_and_structure("Write-Host 'minimal'\n") != []
    no_stop = ps1_text.replace('$ErrorActionPreference = "Stop"', '$ErrorActionPreference = "Continue"')
    assert validate_powershell_syntax_and_structure(no_stop) != []
    no_exit = ps1_text.replace("exit $LASTEXITCODE", "exit 0")
    assert validate_powershell_syntax_and_structure(no_exit) != []
    missing_mode = ps1_text.replace('"Uninstall"', '"RemovedMode"')
    assert validate_powershell_syntax_and_structure(missing_mode) != []


@pytest.mark.parametrize(
    ("bat_argv", "expected_mode", "expect_write_continue", "expect_once"),
    [
        ([], "Connect", True, False),
        (["--connect"], "Connect", True, False),
        (["--connect", "--no-write-continue-config"], "Connect", False, False),
        (["--gamer-whisper"], "GamerWhisper", False, True),
        (["--gamer-whisper", "--continuous"], "GamerWhisper", False, False),
        (["--gamer-check"], "GamerCheck", False, False),
        (["--uninstall"], "Uninstall", False, False),
        (["-Mode", "Connect", "-Once"], "Connect", True, True),
        (["--mode", "gamer-whisper"], "GamerWhisper", False, True),
        (["-Mode", "gamer-check", "-WriteContinueConfig"], "GamerCheck", True, False),
        (["-Mode", "Uninstall"], "Uninstall", False, False),
    ],
)
def test_batch_to_powershell_and_python_fallback_flag_parity(
    bat_argv: list[str],
    expected_mode: str,
    expect_write_continue: bool,
    expect_once: bool,
) -> None:
    bat_text = BAT_PATH.read_text(encoding="utf-8")
    ps1_text = PS1_PATH.read_text(encoding="utf-8")
    parser = quickstart_mod.build_parser()

    # 1. Primary path: Windows 11 PowerShell delegation
    ps_run = evaluate_windows_bat_launcher(bat_text, bat_argv)
    assert ps_run.exit_code == 0
    assert ps_run.backend == "powershell"
    assert "amadeus" in ps_run.stdout.lower()
    assert "0.00 EUR / 0.00 USD" in ps_run.stdout
    assert ps_run.command_argv[1:4] == ["-ExecutionPolicy", "Bypass", "-NoProfile"]
    assert ps_run.command_argv[4:6] == ["-File", r"scripts\join_hivemind.ps1"]
    mode_idx = ps_run.command_argv.index("-Mode")
    assert ps_run.command_argv[mode_idx + 1] == expected_mode

    ps_resolved_py_argv = evaluate_powershell_launcher(ps1_text, ps_run.command_argv)

    # 2. Fallback path: Python standard-library hivemind_quickstart.py (no PowerShell installed)
    py_run = evaluate_windows_bat_launcher(
        bat_text,
        bat_argv,
        existing_paths={
            r"C:\HiveMind\scripts\hivemind_quickstart.py",
            r"C:\HiveMind\src\hivemind",
            r"scripts\join_hivemind.ps1",
            r"scripts\hivemind_quickstart.py",
        },
        where_bins={"python"},
    )
    assert py_run.exit_code == 0
    assert py_run.backend == "python"
    assert py_run.command_argv[0] == "python"
    assert py_run.command_argv[1] == r"scripts\hivemind_quickstart.py"
    fallback_py_argv = py_run.command_argv[2:]

    # 3. Commuting-diagram check: both paths parse to the EXACT same hivemind_quickstart Namespace
    ns_from_ps1 = parser.parse_args(ps_resolved_py_argv)
    ns_from_bat = parser.parse_args(fallback_py_argv)
    assert ns_from_ps1 == ns_from_bat
    assert ns_from_ps1.write_continue_config is expect_write_continue
    assert ns_from_ps1.once is expect_once
    assert ns_from_ps1.gamer_whisper is (expected_mode == "GamerWhisper")
    assert ns_from_ps1.gamer_check is (expected_mode == "GamerCheck")
    assert ns_from_ps1.uninstall is (expected_mode == "Uninstall")


def test_explorer_double_click_pause_options_and_fallback_chains() -> None:
    bat_text = BAT_PATH.read_text(encoding="utf-8")
    ps1_text = PS1_PATH.read_text(encoding="utf-8")
    parser = quickstart_mod.build_parser()

    # 1. Double-clicking in Windows 11 Explorer (0 args + cmd.exe /c) pauses so window stays readable
    dbl_click = evaluate_windows_bat_launcher(
        bat_text,
        [],
        cmdcmdline=r'C:\Windows\System32\cmd.exe /c ""C:\HiveMind\scripts\Join_HiveMind_Campaign_Windows.bat" "',
    )
    assert dbl_click.exit_code == 0
    assert dbl_click.backend == "powershell"
    assert dbl_click.paused is True

    no_pause_env = evaluate_windows_bat_launcher(
        bat_text,
        [],
        env_overrides={"HIVEMIND_NO_PAUSE": "1"},
        cmdcmdline=r'C:\Windows\System32\cmd.exe /c ""C:\HiveMind\scripts\Join_HiveMind_Campaign_Windows.bat" "',
    )
    assert no_pause_env.paused is False

    no_pause_flag = evaluate_windows_bat_launcher(
        bat_text,
        ["--no-pause"],
        cmdcmdline=r'C:\Windows\System32\cmd.exe /c ""C:\HiveMind\scripts\Join_HiveMind_Campaign_Windows.bat" "',
    )
    assert no_pause_flag.paused is False

    # 2. Value-flag forwarding (--handle, --api-key, --gateway, --reserve-mib, --hot-c, single --game)
    rich_argv = [
        "--gamer-whisper",
        "--handle",
        "alex-rtx4070",
        "--api-key",
        "hm-local-test-key-99",
        "--gateway",
        "http://localhost:8080",
        "--reserve-mib",
        "3072",
        "--hot-c",
        "68.5",
        "--game",
        "cyberpunk2077.exe",
    ]
    ps_rich = evaluate_windows_bat_launcher(bat_text, rich_argv)
    py_rich = evaluate_windows_bat_launcher(
        bat_text,
        rich_argv,
        env_overrides={"HIVEMIND_FORCE_PYTHON_FALLBACK": "1"},
    )
    assert ps_rich.backend == "powershell"
    assert py_rich.backend == "python"
    ns_ps_rich = parser.parse_args(evaluate_powershell_launcher(ps1_text, ps_rich.command_argv))
    ns_py_rich = parser.parse_args(py_rich.command_argv[2:])
    assert ns_ps_rich == ns_py_rich
    assert ns_ps_rich.handle == "alex-rtx4070"
    assert ns_ps_rich.api_key == "hm-local-test-key-99"
    assert ns_ps_rich.gateway == "http://localhost:8080"
    assert ns_ps_rich.reserve_mib == 3072
    assert ns_ps_rich.hot_c == 68.5
    assert ns_ps_rich.game == ["cyberpunk2077.exe"]

    ps_alias_argv = [
        "-Mode",
        "GamerWhisper",
        "--once",
        "-Gateway",
        "http://127.0.0.1:9000",
        "-Handle",
        "marie-home",
        "-ApiKey",
        "hm-local-test-key-77",
        "-ReserveMib",
        "2560",
        "-HotC",
        "66",
        "-Game",
        "bg3.exe",
        "--write-continue-config",
    ]
    alias_res = evaluate_windows_bat_launcher(bat_text, ps_alias_argv)
    ns_alias = parser.parse_args(evaluate_powershell_launcher(ps1_text, alias_res.command_argv))
    assert ns_alias.gateway == "http://127.0.0.1:9000"
    assert ns_alias.handle == "marie-home"
    assert ns_alias.reserve_mib == 2560
    assert ns_alias.game == ["bg3.exe"]
    assert ns_alias.once is True
    assert ns_alias.write_continue_config is True

    # 3. Multiple --game flags route safely to Python fallback so PowerShell -File never errors on duplicate -Game
    multi_game = evaluate_windows_bat_launcher(
        bat_text,
        ["--gamer-whisper", "--game", "cs2.exe", "--game", "dota2.exe"],
    )
    assert multi_game.backend == "python"
    ns_multi = parser.parse_args(multi_game.command_argv[2:])
    assert ns_multi.game == ["cs2.exe", "dota2.exe"]

    # 4. --json suppresses ASCII banner on stdout and routes pass-through flags to Python
    json_run = evaluate_windows_bat_launcher(
        bat_text,
        ["--connect", "--json", "--role", "consumer"],
    )
    assert json_run.backend == "python"
    assert json_run.stdout == ""
    ns_json = parser.parse_args(json_run.command_argv[2:])
    assert ns_json.json is True
    assert ns_json.role == "consumer"

    # 5. PowerShell discovery fallback: System32 missing -> powershell.exe on PATH -> pwsh.exe on PATH -> 9009 fallback
    base_fs_no_sys_ps = {
        r"C:\HiveMind\scripts\hivemind_quickstart.py",
        r"scripts\join_hivemind.ps1",
        r"scripts\hivemind_quickstart.py",
    }
    ps_path_run = evaluate_windows_bat_launcher(
        bat_text,
        ["--connect"],
        existing_paths=base_fs_no_sys_ps,
        where_bins={"powershell.exe"},
    )
    assert ps_path_run.backend == "powershell"
    assert ps_path_run.command_argv[0] == "powershell.exe"

    pwsh_run = evaluate_windows_bat_launcher(
        bat_text,
        ["--connect"],
        existing_paths=base_fs_no_sys_ps,
        where_bins={"pwsh.exe"},
    )
    assert pwsh_run.backend == "powershell"
    assert pwsh_run.command_argv[0] == "pwsh.exe"

    venv_fs = base_fs_no_sys_ps | {r"C:\HiveMind\.venv\Scripts\python.exe"}
    fallback_9009 = evaluate_windows_bat_launcher(
        bat_text,
        ["--connect"],
        existing_paths=venv_fs,
        where_bins={"pwsh.exe"},
        child_exit_code=9009,
        fallback_exit_code=0,
    )
    assert fallback_9009.backend == "python"
    assert fallback_9009.command_argv[0] == r"C:\HiveMind\.venv\Scripts\python.exe"
    assert fallback_9009.exit_code == 0

    # Standalone Downloads folder (no repo root) + Windows 'py' launcher fallback
    standalone_fs = {
        r"C:\Users\Alex\Downloads\hivemind_quickstart.py",
    }
    py_launcher_run = evaluate_windows_bat_launcher(
        bat_text,
        ["--connect"],
        existing_paths=standalone_fs,
        where_bins={"py"},
        script_dir_win=r"C:\Users\Alex\Downloads",
        repo_root_win=r"C:\Users\Alex",
    )
    assert py_launcher_run.backend == "python"
    assert py_launcher_run.cwd == "C:\\Users\\Alex\\Downloads\\"
    assert py_launcher_run.command_argv[0] == "py"
    assert py_launcher_run.command_argv[1] == r"C:\Users\Alex\Downloads\hivemind_quickstart.py"

    # Missing Python everywhere -> exit code 2
    no_py_run = evaluate_windows_bat_launcher(
        bat_text,
        ["--connect"],
        existing_paths=standalone_fs,
        where_bins=set(),
        script_dir_win=r"C:\Users\Alex\Downloads",
        repo_root_win=r"C:\Users\Alex",
    )
    assert no_py_run.exit_code == 2
    assert "Python 3.10+" in no_py_run.stderr

    # Missing hivemind_quickstart.py in fallback mode -> exit code 2
    missing_qs_run = evaluate_windows_bat_launcher(
        bat_text,
        ["--connect"],
        existing_paths=set(),
        where_bins={"python"},
    )
    assert missing_qs_run.exit_code == 2
    assert "Missing" in missing_qs_run.stderr


def test_help_and_cli_error_validation_branches() -> None:
    bat_text = BAT_PATH.read_text(encoding="utf-8")
    ps1_text = PS1_PATH.read_text(encoding="utf-8")

    for help_flag in ("--help", "-h", "/?"):
        res = evaluate_windows_bat_launcher(bat_text, [help_flag])
        assert res.exit_code == 0
        assert res.backend is None
        assert "--connect" in res.stdout
        assert "--gamer-whisper" in res.stdout
        assert "--uninstall" in res.stdout
        assert "0.00 EUR / 0.00 USD" in res.stdout

    for flag_missing_val in (
        ["-Mode"],
        ["--gateway"],
        ["--handle"],
        ["--api-key"],
        ["--reserve-mib"],
        ["--hot-c"],
        ["--game"],
    ):
        err_res = evaluate_windows_bat_launcher(bat_text, flag_missing_val)
        assert err_res.exit_code == 2
        assert "Missing required value" in err_res.stderr

    bad_mode = evaluate_windows_bat_launcher(bat_text, ["-Mode", "UnsupportedMode"])
    assert bad_mode.exit_code == 2
    assert "Invalid mode" in bad_mode.stderr

    # Minimal script without trailing exit /b reaches natural EOF return cleanly
    eof_res = evaluate_windows_bat_launcher("@echo off\nsetlocal EnableExtensions\necho ok\n", [])
    assert eof_res.exit_code == 0
    assert eof_res.stdout == "ok"

    base_ps_cmd = ["powershell.exe", "-ExecutionPolicy", "Bypass", "-NoProfile", "-File", "join_hivemind.ps1"]
    with pytest.raises(ValueError, match="duplicate parameter"):
        evaluate_powershell_launcher(ps1_text, [*base_ps_cmd, "-Game", "a.exe", "-Game", "b.exe"])
    with pytest.raises(ValueError, match="Unknown PowerShell parameter"):
        evaluate_powershell_launcher(ps1_text, [*base_ps_cmd, "-UnknownSwitch"])
    with pytest.raises(ValueError, match="not in ValidateSet"):
        evaluate_powershell_launcher(ps1_text, [*base_ps_cmd, "-Mode", "Bogus"])


def test_end_to_end_execution_of_resolved_quickstart_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute the exact ``hivemind_quickstart.py`` argv resolved from BOTH Windows paths end-to-end."""
    monkeypatch.delenv("HIVEMIND_API_KEY", raising=False)
    bat_text = BAT_PATH.read_text(encoding="utf-8")
    ps1_text = PS1_PATH.read_text(encoding="utf-8")

    cred_file = tmp_path / ".hivemind" / "credentials.json"
    cont_file = tmp_path / ".continue" / "hivemind_provider.yaml"
    monkeypatch.setattr(quickstart_mod, "_default_credentials_path", lambda: cred_file)
    monkeypatch.setattr(quickstart_mod, "_default_continue_snippet_path", lambda: cont_file)

    def _gateway_http_responder(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        if url.endswith("/v1/hivemind/register"):
            return 200, {"account_id": "acc_win11", "api_key": "hm-win11-key-123456", "role": "prosumer"}
        if url.endswith("/v1/hivemind/me"):
            assert headers and headers.get("Authorization") == "Bearer hm-win11-key-123456"
            return 200, {"account_id": "acc_win11", "handle": "win11-gamer", "role": "prosumer", "balance": 0.0}
        if url.endswith("/v1/models"):
            return 200, {"data": [{"id": "hivemind/coder"}]}
        return 404, {"error": "not_found"}

    # 1. Double-click / Connect via PowerShell delegation -> registers & writes Continue YAML
    ps_connect = evaluate_windows_bat_launcher(bat_text, ["--handle", "win11-gamer"])
    py_argv_connect = evaluate_powershell_launcher(ps1_text, ps_connect.command_argv)
    assert quickstart_mod.main(py_argv_connect, http_fn=_gateway_http_responder) == 0
    assert cred_file.is_file()
    assert cont_file.is_file()
    assert json.loads(cred_file.read_text(encoding="utf-8"))["api_key"] == "hm-win11-key-123456"

    # 2. Gamer-Whisper via Python fallback -> yields cleanly before start when a Steam game is active
    monkeypatch.setattr(
        quickstart_mod,
        "probe_local_gpu",
        lambda: quickstart_mod.GpuSnapshot(
            present=True,
            name="NVIDIA GeForce RTX 4070",
            total_mib=12288,
            free_mib=10240,
            temp_c=50.0,
            util_pct=4.0,
        ),
    )
    monkeypatch.setattr(quickstart_mod, "list_running_process_names", lambda: {"steam.exe", "cs2.exe"})

    py_whisper = evaluate_windows_bat_launcher(
        bat_text,
        ["--gamer-whisper"],
        env_overrides={"HIVEMIND_FORCE_PYTHON_FALLBACK": "1"},
    )
    assert quickstart_mod.main(py_whisper.command_argv[2:], http_fn=_gateway_http_responder) == 0

    # 3. Uninstall via PowerShell delegation -> removes credentials & Continue config cleanly
    ps_uninst = evaluate_windows_bat_launcher(bat_text, ["--uninstall"])
    py_argv_uninst = evaluate_powershell_launcher(ps1_text, ps_uninst.command_argv)
    assert quickstart_mod.main(py_argv_uninst, http_fn=_gateway_http_responder) == 0
    assert not cred_file.exists()
    assert not cont_file.exists()

    # 4. Cross-platform parity check with Linux/macOS join_hivemind_campaign.sh --help
    sh_help = subprocess.run(["bash", str(SH_PATH), "--help"], capture_output=True, text=True, timeout=10, check=False)
    bat_help = evaluate_windows_bat_launcher(bat_text, ["--help"])
    assert sh_help.returncode == 0
    for shared_flag in ("--connect", "--gamer-whisper"):
        assert shared_flag in sh_help.stdout and shared_flag in bat_help.stdout
