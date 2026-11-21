# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
<#
.SYNOPSIS
    HiveMind Native Windows 11 Quick-Connect & Gamer-Whisper Launcher (Personal Home PCs Only).

.DESCRIPTION
    1. Mode Connect (default): registers or loads ~/.hivemind/credentials.json and prints/writes
       a ready-to-use VS Code (Continue.dev) configuration in under 60 seconds (no PyTorch needed).
    2. Mode GamerCheck: inspects NVIDIA GPU, temperature, and running Steam/PC games.
    3. Mode GamerWhisper: runs a quiet GPU contributor round (<= 70 C cap, 2048 MiB free VRAM,
       instant full stop < 1.5 s if a game starts).
    4. Mode Uninstall: removes saved local HiveMind credentials cleanly.

    STRICT RULE: Personal home PCs and personal hobby projects ONLY. Never install on corporate
    machines or send confidential/company code. 0.00 EUR / 0.00 USD (zero monetary value).
#>
[CmdletBinding()]
param(
    [ValidateSet("Connect", "GamerCheck", "GamerWhisper", "Uninstall")]
    [string]$Mode = "Connect",

    [string]$Handle = "",
    [string]$ApiKey = "",
    [string]$Gateway = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
    [int]$ReserveMib = 2048,
    [double]$HotC = 70.0,
    [string[]]$Game = @(),
    [switch]$WriteContinueConfig,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ConnectPy = Join-Path $ScriptDir "hivemind_quickstart.py"

if (-not (Test-Path $ConnectPy)) {
    Write-Error "Missing $ConnectPy alongside join_hivemind.ps1."
    exit 2
}

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  PROJECT HIVEMIND - WINDOWS 11 PERSONAL QUICK-CONNECT & GAMER WHISPER" -ForegroundColor Cyan
Write-Host "  [Rule] Personal home PC & personal hobby projects ONLY (0.00 EUR)" -ForegroundColor Yellow
Write-Host "============================================================================"

$PythonExe = $null
$RepoVenvPy = Join-Path (Split-Path -Parent $ScriptDir) ".venv\Scripts\python.exe"
if (Test-Path $RepoVenvPy) {
    $PythonExe = $RepoVenvPy
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonExe = (Get-Command python).Source
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonExe = "py"
} else {
    Write-Error "Python 3.10+ was not found on PATH. Install Python from python.org or winget install Python.Python.3.12."
    exit 2
}

$PyArgs = @($ConnectPy, "--gateway", $Gateway, "--reserve-mib", "$ReserveMib", "--hot-c", "$HotC")
if ($Handle -ne "") { $PyArgs += @("--handle", $Handle) }
if ($ApiKey -ne "") { $PyArgs += @("--api-key", $ApiKey) }
if ($WriteContinueConfig) { $PyArgs += "--write-continue-config" }
if ($Once) { $PyArgs += "--once" }
foreach ($g in $Game) {
    if ($g -ne "") { $PyArgs += @("--game", $g) }
}

switch ($Mode) {
    "Uninstall"    { $PyArgs += "--uninstall" }
    "GamerCheck"   { $PyArgs += "--gamer-check" }
    "GamerWhisper" { $PyArgs += "--gamer-whisper" }
    "Connect"      { }
}

& $PythonExe @PyArgs
exit $LASTEXITCODE
