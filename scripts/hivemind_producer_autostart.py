#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Generate cross-platform auto-start background service files (Linux systemd, Windows Scheduled Task, macOS launchd)
for the HiveMind GPU Producer Kit so volunteer GPUs automatically reconnect across reboots and terminal closures.

Usage:
  python scripts/hivemind_producer_autostart.py linux --model coder=qwen2.5-coder:7b
  python scripts/hivemind_producer_autostart.py windows --model coder=qwen2.5-coder:7b --apply ./producer-autostart
  python scripts/hivemind_producer_autostart.py macos --model coder=qwen2.5-coder:7b
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT_GATEWAY_URL = "https://hivemind-openrouter-1003063861791.us-central1.run.app"
PLATFORMS = ("linux", "windows", "macos")
_MODEL_SPEC_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}=[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_URL_RE = re.compile(r"^https?://[A-Za-z0-9._:-]+(?:/[A-Za-z0-9._/-]*)?$")


class AutostartConfigError(ValueError):
    """Raised when an invalid model mapping or URL is passed to the autostart generator."""


def validate_model_spec(spec: str) -> str:
    cleaned = spec.strip()
    if not _MODEL_SPEC_RE.match(cleaned) or ".." in cleaned:
        raise AutostartConfigError(
            f"invalid --model {spec!r}; expected PUBLIC_TIER=LOCAL_OLLAMA_TAG (e.g. coder=qwen2.5-coder:7b)"
        )
    return cleaned


def validate_url(url: str, field_name: str = "url") -> str:
    cleaned = url.strip().rstrip("/")
    if not _URL_RE.match(cleaned) or ".." in cleaned:
        raise AutostartConfigError(f"invalid {field_name}: {url!r}")
    return cleaned


def render_linux_systemd_bundle(
    *,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    models: list[str] | None = None,
    shield_port: int = 8787,
    ollama_url: str = "http://127.0.0.1:11434",
) -> dict[str, str]:
    """Return {'hivemind-producer-runner.sh': ..., 'hivemind-producer.service': ...} for Linux systemd --user."""
    gw = validate_url(gateway_url, "gateway_url")
    ol = validate_url(ollama_url, "ollama_url")
    model_specs = [validate_model_spec(m) for m in (models or ["coder=qwen2.5-coder:7b"])]
    model_flags = " ".join(f"--model {m}" for m in model_specs)
    allow_flags = " ".join(f"--allow-model {m.split('=', 1)[1]}" for m in model_specs)
    port = max(1024, min(int(shield_port), 65535))

    runner_sh = f"""#!/usr/bin/env bash
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
set -euo pipefail

ENV_FILE="${{HIVEMIND_ENV_FILE:-$HOME/.config/hivemind/producer.env}}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

: "${{HIVEMIND_API_KEY:?Set HIVEMIND_API_KEY in $ENV_FILE}}"
if [[ -z "${{HIVEMIND_ENDPOINT_TOKEN:-}}" ]]; then
  HIVEMIND_ENDPOINT_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  export HIVEMIND_ENDPOINT_TOKEN
fi
HIVEMIND_GATEWAY="${{HIVEMIND_GATEWAY:-{gw}}}"
HIVEMIND_OLLAMA="${{HIVEMIND_OLLAMA:-{ol}}}"
SHIELD_PORT="{port}"
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cleanup() {{
  [[ -n "${{SHIELD_PID:-}}" ]] && kill "$SHIELD_PID" 2>/dev/null || true
  [[ -n "${{TUNNEL_PID:-}}" ]] && kill "$TUNNEL_PID" 2>/dev/null || true
}}
trap cleanup EXIT INT TERM

python3 "$REPO_ROOT/scripts/hivemind_producer_shield.py" \\
  --upstream "$HIVEMIND_OLLAMA" \\
  --port "${{SHIELD_PORT}}" \\
  {allow_flags} &
SHIELD_PID=$!

if [[ -z "${{HIVEMIND_ENDPOINT_URL:-}}" ]]; then
  TUNNEL_LOG="$(mktemp -t hivemind-cloudflared.XXXXXX.log)"
  cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:${{SHIELD_PORT}}" >"$TUNNEL_LOG" 2>&1 &
  TUNNEL_PID=$!
  for _ in $(seq 1 30); do
    HIVEMIND_ENDPOINT_URL="$(grep -Eo 'https://[a-z0-9-]+\\.trycloudflare\\.com' "$TUNNEL_LOG" | head -n 1 || true)"
    [[ -n "$HIVEMIND_ENDPOINT_URL" ]] && break
    sleep 1
  done
  if [[ -z "$HIVEMIND_ENDPOINT_URL" ]]; then
    echo "error: cloudflared did not publish a trycloudflare.com URL within 30s" >&2
    exit 1
  fi
fi

exec python3 "$REPO_ROOT/scripts/hivemind_producer_agent.py" \\
  --gateway "$HIVEMIND_GATEWAY" \\
  --endpoint "$HIVEMIND_ENDPOINT_URL" \\
  --local-endpoint "$HIVEMIND_OLLAMA" \\
  {model_flags}
"""

    service_unit = """[Unit]
Description=HiveMind OpenRouter Volunteer GPU Producer (Shield + Tunnel + Heartbeat Agent)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=%h/.local/share/hivemind/hivemind-producer-runner.sh
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
"""
    return {
        "hivemind-producer-runner.sh": runner_sh,
        "hivemind-producer.service": service_unit,
    }


def render_windows_powershell_bundle(
    *,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    models: list[str] | None = None,
    shield_port: int = 8787,
    ollama_url: str = "http://127.0.0.1:11434",
) -> dict[str, str]:
    """Return {'run-producer-service.ps1': ..., 'install-producer-service.ps1': ...} for Windows 10/11."""
    gw = validate_url(gateway_url, "gateway_url")
    ol = validate_url(ollama_url, "ollama_url")
    model_specs = [validate_model_spec(m) for m in (models or ["coder=qwen2.5-coder:7b"])]
    ps_model_args = ", ".join(f'"{m}"' for m in model_specs)
    ps_allow_args = ", ".join(f'"{m.split("=", 1)[1]}"' for m in model_specs)
    port = max(1024, min(int(shield_port), 65535))

    run_ps1 = f"""# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
$ErrorActionPreference = "Stop"
$EnvFile = Join-Path $HOME ".hivemind\\producer.env.ps1"
if (Test-Path $EnvFile) {{
    . $EnvFile
}}
if (-not $env:HIVEMIND_API_KEY) {{
    Write-Error "HIVEMIND_API_KEY must be set in $EnvFile"
    exit 1
}}
if (-not $env:HIVEMIND_ENDPOINT_TOKEN) {{
    $env:HIVEMIND_ENDPOINT_TOKEN = (python -c "import secrets; print(secrets.token_urlsafe(32))").Trim()
}}
$Gateway = if ($env:HIVEMIND_GATEWAY) {{ $env:HIVEMIND_GATEWAY }} else {{ "{gw}" }}
$Ollama = if ($env:HIVEMIND_OLLAMA) {{ $env:HIVEMIND_OLLAMA }} else {{ "{ol}" }}
$ShieldPort = {port}
$RepoRoot = Split-Path -Parent $PSScriptRoot

$AllowModels = @({ps_allow_args})
$ShieldArgs = @(
    "$RepoRoot\\scripts\\hivemind_producer_shield.py",
    "--upstream", $Ollama,
    "--port", "$ShieldPort"
)
foreach ($am in $AllowModels) {{
    $ShieldArgs += @("--allow-model", $am)
}}
$ShieldProc = Start-Process -FilePath "python" -ArgumentList $ShieldArgs -PassThru -WindowStyle Hidden

$TunnelLog = Join-Path $env:TEMP "hivemind-cloudflared.log"
if (Test-Path $TunnelLog) {{ Remove-Item $TunnelLog -Force }}
$TunnelProc = Start-Process -FilePath "cloudflared" -ArgumentList @(
    "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:$ShieldPort", "--logfile", $TunnelLog
) -PassThru -WindowStyle Hidden

$EndpointUrl = $env:HIVEMIND_ENDPOINT_URL
for ($i = 0; $i -lt 30 -and -not $EndpointUrl; $i++) {{
    Start-Sleep -Seconds 1
    if (Test-Path $TunnelLog) {{
        $Match = Select-String -Path $TunnelLog -Pattern "https://[a-z0-9-]+\\.trycloudflare\\.com" | Select-Object -First 1
        if ($Match) {{ $EndpointUrl = $Match.Matches[0].Value }}
    }}
}}
if (-not $EndpointUrl) {{
    Stop-Process -Id $ShieldProc.Id -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $TunnelProc.Id -Force -ErrorAction SilentlyContinue
    Write-Error "cloudflared did not publish a trycloudflare.com URL within 30s"
    exit 1
}}

$Models = @({ps_model_args})
$AgentArgs = @(
    "$RepoRoot\\scripts\\hivemind_producer_agent.py",
    "--gateway", $Gateway,
    "--endpoint", $EndpointUrl,
    "--local-endpoint", $Ollama
)
foreach ($m in $Models) {{
    $AgentArgs += @("--model", $m)
}}
try {{
    & python @AgentArgs
}} finally {{
    Stop-Process -Id $ShieldProc.Id -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $TunnelProc.Id -Force -ErrorAction SilentlyContinue
}}
"""

    install_ps1 = """# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
# Registers the HiveMind Producer Agent as a Windows Scheduled Task that starts automatically at user logon
# and restarts automatically if the network or Cloudflare tunnel resets.
$ErrorActionPreference = "Stop"
$RunnerPath = Join-Path $PSScriptRoot "run-producer-service.ps1"
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$RunnerPath`""
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName "HiveMindProducerAgent" -Action $Action -Trigger $Trigger -Settings $Settings -Description "HiveMind Volunteer GPU Producer (Shield + Cloudflare Tunnel + Heartbeat Agent)" -Force
Start-ScheduledTask -TaskName "HiveMindProducerAgent"
Write-Host "HiveMindProducerAgent installed and started. Check status with: Get-ScheduledTask -TaskName HiveMindProducerAgent"
"""
    return {
        "run-producer-service.ps1": run_ps1,
        "install-producer-service.ps1": install_ps1,
    }


def render_macos_launchd_bundle(
    *,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    models: list[str] | None = None,
    shield_port: int = 8787,
    ollama_url: str = "http://127.0.0.1:11434",
) -> dict[str, str]:
    """Return {'org.hivemind.producer.plist': ..., 'hivemind-producer-runner.sh': ...} for macOS launchd."""
    linux_bundle = render_linux_systemd_bundle(
        gateway_url=gateway_url,
        models=models,
        shield_port=shield_port,
        ollama_url=ollama_url,
    )
    plist = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>org.hivemind.producer</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-lc</string>
        <string>$HOME/.local/share/hivemind/hivemind-producer-runner.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>10</integer>
</dict>
</plist>
"""
    return {
        "hivemind-producer-runner.sh": linux_bundle["hivemind-producer-runner.sh"],
        "org.hivemind.producer.plist": plist,
    }


def generate_autostart_bundle(
    platform: str,
    *,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    models: list[str] | None = None,
    shield_port: int = 8787,
    ollama_url: str = "http://127.0.0.1:11434",
) -> dict[str, str]:
    plat = platform.strip().lower()
    if plat == "linux":
        return render_linux_systemd_bundle(gateway_url=gateway_url, models=models, shield_port=shield_port, ollama_url=ollama_url)
    if plat == "windows":
        return render_windows_powershell_bundle(gateway_url=gateway_url, models=models, shield_port=shield_port, ollama_url=ollama_url)
    if plat == "macos":
        return render_macos_launchd_bundle(gateway_url=gateway_url, models=models, shield_port=shield_port, ollama_url=ollama_url)
    raise AutostartConfigError(f"unsupported platform {platform!r}; choose one of {', '.join(PLATFORMS)}")


def write_autostart_bundle(dest: Path, files: dict[str, str]) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for filename, content in files.items():
        target = dest / filename
        target.write_text(content, encoding="utf-8")
        if filename.endswith(".sh"):
            target.chmod(0o755)
        written.append(str(target))
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("platform", choices=PLATFORMS, help="target operating system (linux, windows, macos)")
    ap.add_argument("--gateway", default=DEFAULT_GATEWAY_URL, help="HiveMind OpenRouter URL")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434", help="local Ollama upstream URL")
    ap.add_argument("--shield-port", type=int, default=8787, help="local loopback port for hivemind_producer_shield.py")
    ap.add_argument("--model", action="append", default=[], help="PUBLIC_TIER=LOCAL_OLLAMA_TAG (repeatable, default: coder=qwen2.5-coder:7b)")
    ap.add_argument("--apply", metavar="DEST_DIR", help="write the generated service files into DEST_DIR")
    args = ap.parse_args(argv)

    try:
        files = generate_autostart_bundle(
            args.platform,
            gateway_url=args.gateway,
            models=args.model or ["coder=qwen2.5-coder:7b"],
            shield_port=args.shield_port,
            ollama_url=args.ollama,
        )
        if args.apply:
            paths = write_autostart_bundle(Path(args.apply).expanduser(), files)
            for p in paths:
                print(f"wrote {p}")
        else:
            for name, content in files.items():
                print(f"# --- {name} ---\n{content}")
    except AutostartConfigError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
