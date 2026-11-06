# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
<#
.SYNOPSIS
  HiveMind@home (v2.13.0) — Official BOINC Client + Idle-GPU Worker Installer for Windows 10/11.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -Command "iwr -useb https://xaviercallens.github.io/HiveMindatHome.github.io/boinc/Install_BOINC_HiveMind_Windows.ps1 | iex"
#>
param(
  [string]$Handle = "@BOINC_Win64_Cruncher",
  [string]$CoordinatorUrl = "https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app",
  [string]$WeakAccountKey = "hm_boinc_weak_key_v213_open_science"
)

$ErrorActionPreference = "Continue"
$InstallDir = Join-Path $env:USERPROFILE ".hivemind_boinc"
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host " 🐝 HiveMind@home (v2.13.0) — Windows BOINC Client & Idle-GPU Worker Setup" -ForegroundColor Yellow
Write-Host "    Contributor Handle : $Handle"
Write-Host "    Project Master URL : $CoordinatorUrl/"
Write-Host "    VRAM Budget Cap    : 6,656 MiB | Gaming Preemption: < 412 ms"
Write-Host "==============================================================================" -ForegroundColor Cyan

# 1. Check / Install Official Berkeley BOINC Client via winget
$BoincCmdPath = "C:\Program Files\BOINC\boinccmd.exe"
if (-not (Test-Path $BoincCmdPath)) {
  Write-Host "[1/5] Installing official Berkeley BOINC Client via winget (UCBerkeley.BOINC)..." -ForegroundColor Green
  if (Get-Command winget -ErrorAction SilentlyContinue) {
    winget install --id UCBerkeley.BOINC -e --accept-package-agreements --accept-source-agreements
  } else {
    Write-Host "  [!] winget not found; you can install Berkeley BOINC from https://boinc.berkeley.edu/download.php" -ForegroundColor Yellow
  }
} else {
  Write-Host "[1/5] Found official Berkeley BOINC Client at: $BoincCmdPath" -ForegroundColor Green
}

# 2. Write cc_config.xml and app_config.xml
Write-Host "[2/5] Generating BOINC cc_config.xml & app_config.xml in $InstallDir..." -ForegroundColor Green
$CcConfigXml = @"
<cc_config>
  <log_flags>
    <task>1</task>
    <file_xfer>1</file_xfer>
    <sched_ops>1</sched_ops>
  </log_flags>
  <options>
    <use_all_gpus>1</use_all_gpus>
    <allow_remote_gui_rpc>0</allow_remote_gui_rpc>
  </options>
</cc_config>
"@
Set-Content -Path (Join-Path $InstallDir "cc_config.xml") -Value $CcConfigXml -Encoding UTF8

$AppConfigXml = @"
<app_config>
  <app>
    <name>hivemind_astro_bls</name>
    <max_concurrent>2</max_concurrent>
    <gpu_versions>
      <gpu_usage>0.5</gpu_usage>
      <cpu_usage>0.25</cpu_usage>
    </gpu_versions>
  </app>
  <app>
    <name>hivemind_sundials_ode</name>
    <max_concurrent>4</max_concurrent>
    <gpu_versions>
      <gpu_usage>0.5</gpu_usage>
      <cpu_usage>0.5</cpu_usage>
    </gpu_versions>
  </app>
  <project_max_concurrent>4</project_max_concurrent>
</app_config>
"@
Set-Content -Path (Join-Path $InstallDir "app_config.xml") -Value $AppConfigXml -Encoding UTF8

$BoincDataDir = "C:\ProgramData\BOINC"
if (Test-Path $BoincDataDir) {
  Copy-Item -Path (Join-Path $InstallDir "cc_config.xml") -Destination (Join-Path $BoincDataDir "cc_config.xml") -Force -ErrorAction SilentlyContinue
}

# 3. Attach BOINC Client to HiveMind@home
if (Test-Path $BoincCmdPath) {
  Write-Host "[3/5] Attaching BOINC Client to $CoordinatorUrl/ ..." -ForegroundColor Green
  & $BoincCmdPath --project_attach "$CoordinatorUrl/" "$WeakAccountKey" 2>$null
  & $BoincCmdPath --read_cc_config 2>$null
  & $BoincCmdPath --set_gpu_mode auto 2>$null
} else {
  Write-Host "[3/5] Configuring standalone CUDA PTX / CPU BOINC worker mode..." -ForegroundColor Green
}

# 4. Download hivemind_boinc_idle_worker.py
$WorkerScript = Join-Path $InstallDir "hivemind_boinc_idle_worker.py"
Write-Host "[4/5] Downloading HiveMind BOINC Idle-GPU Worker to $WorkerScript ..." -ForegroundColor Green
Invoke-WebRequest -Uri "https://xaviercallens.github.io/HiveMindatHome.github.io/boinc/hivemind_boinc_idle_worker.py" -OutFile $WorkerScript -UseBasicParsing

# 5. Execute 1 Verified Workunit & Display Receipt
Write-Host "[5/5] Executing first verified BOINC BLS Exoplanet Transit Workunit (--once --force-fold)..." -ForegroundColor Green
python $WorkerScript --coordinator $CoordinatorUrl --handle $Handle --force-fold --once

Write-Host ""
Write-Host "✅ HiveMind@home BOINC Windows Setup Complete!" -ForegroundColor Green
Write-Host "   • Run background idle-GPU daemon : python `"$WorkerScript`" --handle `"$Handle`""
Write-Host "   • Download Native Win64 GUI .exe : https://github.com/xaviercallens/HiveMind/releases/latest/download/HiveMind_Exoplanet_Hunter.exe"
Write-Host "   • Live BOINC Leaderboard         : $CoordinatorUrl/community"
