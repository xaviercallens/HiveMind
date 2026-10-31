# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
# Build and deploy the 5-language (EN, FR, ES, DE, ZH) HiveMind OpenRouter Cloud Run overlay.
# Usage: .\deploy\router_overlay\deploy_openrouter_i18n.ps1 [-Tag "i18n-5lang"]
param(
    [string]$Tag = "i18n-5lang",
    [string]$ProjectId = "gen-lang-client-0625573011",
    [string]$Region = "us-central1",
    [string]$ServiceName = "hivemind-openrouter"
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $Here "..\..")
$RouterDir = Join-Path $RepoRoot "src\hivemind\router"
$Image = "$Region-docker.pkg.dev/$ProjectId/cloud-run-source-deploy/${ServiceName}:$Tag"

$ActiveAccount = (gcloud auth list --filter="status:ACTIVE" --format="value(account)" 2>$null)
if (-not $ActiveAccount) {
    Write-Host "[HiveMind OpenRouter] No active gcloud account detected. Launching 'gcloud auth login'..." -ForegroundColor Yellow
    gcloud auth login
}

$StageDir = Join-Path ([System.IO.Path]::GetTempPath()) ("hivemind-router-overlay-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Path $StageDir -Force | Out-Null

try {
    Copy-Item (Join-Path $Here "Dockerfile") $StageDir
    Copy-Item (Join-Path $Here "server.py") $StageDir
    foreach ($mod in @("points_wording.py", "join_page.py", "life_as_a_bee.py", "portal_page.py", "signup_page.py")) {
        Copy-Item (Join-Path $RouterDir $mod) $StageDir
    }

    Write-Host "[1/2] Building Cloud Run overlay image: $Image ..." -ForegroundColor Cyan
    gcloud builds submit --project $ProjectId --region $Region --tag $Image $StageDir

    Write-Host "[2/2] Updating Cloud Run service $ServiceName in $Region ..." -ForegroundColor Cyan
    gcloud run services update $ServiceName --project $ProjectId --region $Region --image $Image

    Write-Host "[OK] https://hivemind-openrouter-1003063861791.us-central1.run.app/ is now live in 5 languages (EN, FR, ES, DE, ZH)!" -ForegroundColor Green
} finally {
    if (Test-Path $StageDir) {
        Remove-Item -Path $StageDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
