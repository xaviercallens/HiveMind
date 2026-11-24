# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
<#
.SYNOPSIS
    HiveMind Native Windows 11 Quick-Connect, Doctor, DeploySmoke & Gamer-Whisper Launcher (Personal Home PCs Only).

.DESCRIPTION
    1. Mode Connect (default): registers or loads ~/.hivemind/credentials.json and prints/writes
       a ready-to-use VS Code (Continue.dev) configuration in under 60 seconds (no PyTorch needed).
    2. Mode GamerCheck: inspects NVIDIA GPU, temperature, and running Steam/PC games.
    3. Mode GamerWhisper: runs a quiet GPU contributor round (<= 70 C cap, 2048 MiB free VRAM,
       instant full stop < 1.5 s if a game starts).
    4. Mode Doctor: runs hivemind_doctor.py --noel holiday preflight, yield/thermal & fleet checks.
    5. Mode DeploySmoke: runs noel_deploy_smoke_and_win11_verifier.py --smoke-only against $Gateway.
    6. Mode Uninstall: removes saved local HiveMind credentials cleanly (always permitted).

    STRICT RULE: Personal home PCs and personal hobby/open-source projects ONLY. Never install on
    corporate machines or send confidential/company code. 0.00 EUR / 0.00 USD (AGPL-3.0).
#>
[CmdletBinding()]
param(
    [ValidateSet("Connect", "GamerCheck", "GamerWhisper", "Doctor", "DeploySmoke", "Uninstall")]
    [string]$Mode = "Connect",

    [string]$Handle = "",
    [string]$ApiKey = "",
    [string]$Gateway = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
    [int]$ReserveMib = 2048,
    [double]$HotC = 70.0,
    [string[]]$Game = @(),
    [switch]$WriteContinueConfig,
    [switch]$Once,

    [ValidateSet("en", "fr")]
    [string]$Lang = "en",

    [switch]$SmokeRoutes,
    [switch]$CheckDeployParity,
    [switch]$PostSetupTelemetry,

    [ValidateRange(0, 5)]
    [int]$EaseRating = 0,

    [switch]$HadGameLag,
    [switch]$FlushFeedbackQueue,
    [switch]$Bootstrap,
    [switch]$Json,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

$PersonalPcNoticeEn = "100% personal home PCs & personal hobby/open-source projects only (Amadeus / Personal-PC rule: 0.00 EUR / 0.00 USD, AGPL-3.0)."
$PersonalPcNoticeFr = "100 % PC personnels a domicile et projets hobby/open-source uniquement (regle Amadeus / PC perso : 0,00 EUR / 0,00 USD, AGPL-3.0)."

$AllowedBootstrapScripts = @(
    "hivemind_quickstart.py",
    "hivemind_doctor.py",
    "noel_deploy_smoke_and_win11_verifier.py",
    "noel_post_setup_probe.py",
    "noel_cloud_run_parity_gate.py",
    "noel_vscode_autoconfig.py",
    "noel_gamer_whisper_selftest.py",
    "join_hivemind.ps1"
)

function Format-MaskedApiKey {
    param([string]$RawKey)
    if ([string]::IsNullOrWhiteSpace($RawKey)) {
        return ""
    }
    $clean = $RawKey.Trim()
    if ($clean.Length -lt 12) {
        return "<redacted>"
    }
    $tail = $clean.Substring($clean.Length - 4)
    if ($clean.StartsWith("hm_live_")) {
        return "hm_live_***$tail"
    }
    if ($clean.StartsWith("hm-")) {
        return "hm-***$tail"
    }
    return "<redacted>"
}

function Get-GitConfigFilesForPath {
    param([string]$StartDir)
    $configs = New-Object System.Collections.Generic.List[string]
    if ([string]::IsNullOrWhiteSpace($StartDir) -or -not (Test-Path -LiteralPath $StartDir)) {
        return $configs
    }
    $curr = (Resolve-Path -LiteralPath $StartDir).Path
    while ($curr) {
        $dotGit = Join-Path $curr ".git"
        if (Test-Path -LiteralPath $dotGit -PathType Container) {
            $cfg = Join-Path $dotGit "config"
            if (Test-Path -LiteralPath $cfg -PathType Leaf) {
                $configs.Add($cfg)
            }
            break
        }
        if (Test-Path -LiteralPath $dotGit -PathType Leaf) {
            $lines = Get-Content -LiteralPath $dotGit -ErrorAction SilentlyContinue
            if ($lines -and $lines[0] -match '^\s*gitdir:\s*(.+)$') {
                $rawGitDir = $Matches[1].Trim()
                $gitDir = if ([System.IO.Path]::IsPathRooted($rawGitDir)) {
                    $rawGitDir
                } else {
                    [System.IO.Path]::GetFullPath((Join-Path $curr $rawGitDir))
                }
                $directCfg = Join-Path $gitDir "config"
                if (Test-Path -LiteralPath $directCfg -PathType Leaf) {
                    $configs.Add($directCfg)
                }
                $commonDirFile = Join-Path $gitDir "commondir"
                if (Test-Path -LiteralPath $commonDirFile -PathType Leaf) {
                    $commonRel = (Get-Content -LiteralPath $commonDirFile -Raw -ErrorAction SilentlyContinue)
                    if ($commonRel) {
                        $commonRel = $commonRel.Trim()
                        $commonCfg = [System.IO.Path]::GetFullPath((Join-Path $gitDir (Join-Path $commonRel "config")))
                        if ((Test-Path -LiteralPath $commonCfg -PathType Leaf) -and (-not $configs.Contains($commonCfg))) {
                            $configs.Add($commonCfg)
                        }
                    }
                }
            }
            break
        }
        $parent = Split-Path -Parent $curr
        if ([string]::IsNullOrEmpty($parent) -or $parent -eq $curr) {
            break
        }
        $curr = $parent
    }
    return $configs
}

function Get-CorporateComplianceIndicators {
    $corpRegex = '(?i)(amadeus|(?:^|[^a-z])corp(?:orate|s)?(?:$|[^a-z])|[a-z0-9]corp(?:$|[^a-z])|enterprise|intranet|(?:^|[^a-z])internal(?:$|[^a-z])|[a-z0-9]internal(?:$|[^a-z])|zscaler|netskope|bluecoat|fortinet|palo[-_.]?alto)'
    $indicators = New-Object System.Collections.Generic.List[string]

    foreach ($entry in (Get-ChildItem -Path Env: -ErrorAction SilentlyContinue)) {
        $k = [string]$entry.Name
        $v = [string]$entry.Value
        if ([string]::IsNullOrWhiteSpace($k) -or [string]::IsNullOrWhiteSpace($v)) {
            continue
        }
        $ku = $k.ToUpperInvariant()
        if ($ku.StartsWith("AMADEUS")) {
            if (-not $indicators.Contains("env:AMADEUS*")) { $indicators.Add("env:AMADEUS*") }
        } elseif ($ku -eq "ZSCALER_CERT_PATH") {
            if (-not $indicators.Contains("env:ZSCALER_CERT_PATH")) { $indicators.Add("env:ZSCALER_CERT_PATH") }
        } elseif ($ku -eq "USERDNSDOMAIN") {
            if (-not $indicators.Contains("env:USERDNSDOMAIN")) { $indicators.Add("env:USERDNSDOMAIN") }
        } elseif ($ku -in @("USERDOMAIN", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "COMPUTERNAME", "HOSTNAME", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")) {
            if ($v -match $corpRegex) {
                $tag = "env:$ku"
                if (-not $indicators.Contains($tag)) { $indicators.Add($tag) }
            }
        }
    }

    foreach ($wsPath in @($PWD.Path, $ScriptDir)) {
        if (-not [string]::IsNullOrWhiteSpace($wsPath) -and ($wsPath -match $corpRegex)) {
            if (-not $indicators.Contains("workspace_path")) { $indicators.Add("workspace_path") }
        }
    }

    $checkedCfgs = New-Object System.Collections.Generic.HashSet[string]
    foreach ($rootCandidate in @($PWD.Path, $RepoRoot)) {
        foreach ($cfgFile in (Get-GitConfigFilesForPath -StartDir $rootCandidate)) {
            if (-not $checkedCfgs.Add($cfgFile)) {
                continue
            }
            $cfgLines = Get-Content -LiteralPath $cfgFile -ErrorAction SilentlyContinue
            foreach ($line in $cfgLines) {
                if ($line -match '^\s*(?:push)?(?:url|insteadof|pushinsteadof)\s*=\s*(.+)$' -or $line -match '^\s*\[url\s+"([^"]+)"\]') {
                    $remoteVal = $Matches[1].Trim()
                    if ($remoteVal -match $corpRegex) {
                        if (-not $indicators.Contains("git_remote")) { $indicators.Add("git_remote") }
                    }
                }
            }
        }
    }

    return $indicators
}

function Resolve-ValidatedGateway {
    param([string]$RawGateway)
    if ([string]::IsNullOrWhiteSpace($RawGateway)) {
        throw "Gateway URL must not be empty."
    }
    $cleaned = $RawGateway.Trim().TrimEnd("/")
    if ($cleaned -match '\s' -or $cleaned -match '[\x00-\x1f\x7f]') {
        throw "Gateway URL must not contain whitespace or control characters."
    }
    [Uri]$parsed = $null
    if (-not [Uri]::TryCreate($cleaned, [UriKind]::Absolute, [ref]$parsed)) {
        throw "Gateway URL is not a valid absolute URI."
    }
    if (-not [string]::IsNullOrEmpty($parsed.UserInfo) -or -not [string]::IsNullOrEmpty($parsed.Query) -or -not [string]::IsNullOrEmpty($parsed.Fragment)) {
        throw "Gateway URL must not embed credentials, queries, or fragments."
    }
    if ($parsed.AbsolutePath -ne "" -and $parsed.AbsolutePath -ne "/") {
        throw "Gateway URL must be a bare origin without a path suffix."
    }
    $hostLower = $parsed.Host.ToLowerInvariant().Trim("[", "]")
    $schemeLower = $parsed.Scheme.ToLowerInvariant()
    $isLoopback = $hostLower -in @("127.0.0.1", "localhost", "::1")
    $isRunApp = $hostLower -match '^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+run\.app$'
    if ($isLoopback) {
        if ($schemeLower -notin @("http", "https")) {
            throw "Loopback gateway URL must use http or https."
        }
    } elseif ($isRunApp) {
        if ($schemeLower -ne "https") {
            throw "Remote Cloud Run gateway URL must use https."
        }
    } else {
        throw "Untrusted gateway host; expected https://*.run.app or loopback."
    }
    return $cleaned
}

function Resolve-PythonExecutable {
    $winVenvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    $posixVenvPy = Join-Path $RepoRoot ".venv/bin/python"
    if (Test-Path -LiteralPath $winVenvPy -PathType Leaf) {
        return $winVenvPy
    }
    if (Test-Path -LiteralPath $posixVenvPy -PathType Leaf) {
        return $posixVenvPy
    }
    foreach ($cmdName in @("python", "python3", "py")) {
        $cmd = Get-Command $cmdName -ErrorAction SilentlyContinue
        if ($cmd) {
            return $cmd.Source
        }
    }
    return $null
}

# 1. Local Personal-PC & Amadeus Compliance Gate (fail closed with exit 2; ALWAYS permit -Mode Uninstall)
if ($Mode -ne "Uninstall") {
    $corpHits = Get-CorporateComplianceIndicators
    if ($corpHits.Count -gt 0) {
        $hitList = ($corpHits -join ", ")
        $blockMsg = if ($Lang -eq "fr") {
            "REFUS DE CONFORMITE ($hitList) : environnement d'entreprise detecte (0 appel reseau, 0 ecriture). $PersonalPcNoticeFr"
        } else {
            "COMPLIANCE BLOCKED ($hitList): corporate environment indicator detected (0 network calls, 0 writes). $PersonalPcNoticeEn"
        }
        if ($Json) {
            $blockedObj = [ordered]@{
                compliance_ok       = $false
                corporate_detected  = $true
                corporate_indicators = @($corpHits)
                personal_pc_only    = $true
                financial_promise   = "0.00 EUR / 0.00 USD"
                license             = "AGPL-3.0"
                notice_en           = $PersonalPcNoticeEn
                notice_fr           = $PersonalPcNoticeFr
                exit_code           = 2
                summary             = $blockMsg
            }
            Write-Output ($blockedObj | ConvertTo-Json -Depth 4)
        } else {
            Write-Error $blockMsg
        }
        exit 2
    }
}

# 2. Validate Gateway allowlist before any -Bootstrap download or Python execution
try {
    $CleanGateway = Resolve-ValidatedGateway -RawGateway $Gateway
} catch {
    $gwErr = $_.Exception.Message
    if ($Json) {
        $errObj = [ordered]@{
            compliance_ok     = $true
            gateway_valid     = $false
            personal_pc_only  = $true
            financial_promise = "0.00 EUR / 0.00 USD"
            license           = "AGPL-3.0"
            exit_code         = 2
            error             = $gwErr
        }
        Write-Output ($errObj | ConvertTo-Json -Depth 4)
    } else {
        Write-Error $gwErr
    }
    exit 2
}

if (-not $Json) {
    Write-Host "============================================================================" -ForegroundColor Cyan
    Write-Host "  PROJECT HIVEMIND - WINDOWS 11 PERSONAL QUICK-CONNECT & GAMER WHISPER" -ForegroundColor Cyan
    Write-Host "  [Rule] Personal home PC & personal hobby projects ONLY (0.00 EUR / 0.00 USD)" -ForegroundColor Yellow
    Write-Host "============================================================================"
}

# 3. Route Mode to allowlisted target script
$TargetScriptName = switch ($Mode) {
    "Doctor"      { "hivemind_doctor.py" }
    "DeploySmoke" { "noel_deploy_smoke_and_win11_verifier.py" }
    default       { "hivemind_quickstart.py" }
}

if (($TargetScriptName -notmatch '^[A-Za-z0-9_]+\.(?:py|ps1)$') -or ($AllowedBootstrapScripts -notcontains $TargetScriptName)) {
    Write-Error "Refused non-allowlisted target script: $TargetScriptName"
    exit 2
}

$TargetScript = Join-Path $ScriptDir $TargetScriptName
$BootstrapPlanned = $false

if (-not (Test-Path -LiteralPath $TargetScript -PathType Leaf)) {
    if ($Bootstrap) {
        if ($DryRun) {
            $BootstrapPlanned = $true
        } else {
            $BootstrapUri = "$CleanGateway/scripts/$TargetScriptName"
            try {
                Invoke-RestMethod -Uri $BootstrapUri -Method Get -OutFile $TargetScript -TimeoutSec 15 | Out-Null
            } catch {
                Write-Error "Bootstrap download failed for $TargetScriptName from ${CleanGateway}: $($_.Exception.Message)"
                exit 2
            }
            if (-not (Test-Path -LiteralPath $TargetScript -PathType Leaf) -or (Get-Item -LiteralPath $TargetScript).Length -le 0) {
                Write-Error "Bootstrap download produced an empty or missing file at $TargetScript."
                exit 2
            }
        }
    } elseif (-not $DryRun) {
        Write-Error "Missing $TargetScript alongside join_hivemind.ps1. Pass -Bootstrap to fetch from $CleanGateway/scripts/$TargetScriptName."
        exit 2
    }
}

# 4. Build sanitized argument list matching the target script's CLI contract
$PyArgs = New-Object System.Collections.Generic.List[string]
$PyArgs.Add($TargetScript)

switch ($Mode) {
    "Doctor" {
        $PyArgs.Add("--noel")
        $PyArgs.Add("--gateway"); $PyArgs.Add($CleanGateway)
        $PyArgs.Add("--lang"); $PyArgs.Add($Lang)
        $PyArgs.Add("--max-temp-c"); $PyArgs.Add("$HotC")
        foreach ($g in $Game) {
            if (-not [string]::IsNullOrWhiteSpace($g)) {
                $PyArgs.Add("--game"); $PyArgs.Add($g)
            }
        }
        if ($PostSetupTelemetry) {
            $PyArgs.Add("--submit-ces")
            if ($EaseRating -ge 1) {
                $PyArgs.Add("--ease-rating"); $PyArgs.Add("$EaseRating")
            }
            if ($HadGameLag) {
                $PyArgs.Add("--had-game-lag")
            }
        }
        if ($Json) { $PyArgs.Add("--json") }
    }
    "DeploySmoke" {
        $PyArgs.Add("--smoke-only")
        $PyArgs.Add("--gateway"); $PyArgs.Add($CleanGateway)
        $PyArgs.Add("--lang"); $PyArgs.Add($Lang)
        if ($PostSetupTelemetry) { $PyArgs.Add("--probe-feedback-post") }
        if ($Json) { $PyArgs.Add("--json") }
    }
    "Uninstall" {
        $PyArgs.Add("--uninstall")
        if ($Json) { $PyArgs.Add("--json") }
    }
    default {
        $PyArgs.Add("--gateway"); $PyArgs.Add($CleanGateway)
        $PyArgs.Add("--reserve-mib"); $PyArgs.Add("$ReserveMib")
        $PyArgs.Add("--hot-c"); $PyArgs.Add("$HotC")
        $PyArgs.Add("--lang"); $PyArgs.Add($Lang)
        if (-not [string]::IsNullOrWhiteSpace($Handle)) { $PyArgs.Add("--handle"); $PyArgs.Add($Handle) }
        if (-not [string]::IsNullOrWhiteSpace($ApiKey)) { $PyArgs.Add("--api-key"); $PyArgs.Add($ApiKey) }
        if ($WriteContinueConfig) { $PyArgs.Add("--write-continue-config") }
        if ($Once) { $PyArgs.Add("--once") }
        foreach ($g in $Game) {
            if (-not [string]::IsNullOrWhiteSpace($g)) {
                $PyArgs.Add("--game"); $PyArgs.Add($g)
            }
        }
        if ($SmokeRoutes) { $PyArgs.Add("--smoke-routes") }
        if ($CheckDeployParity) { $PyArgs.Add("--check-deploy-parity") }
        if ($PostSetupTelemetry) { $PyArgs.Add("--post-setup-telemetry") }
        if ($EaseRating -ge 1) { $PyArgs.Add("--ease-rating"); $PyArgs.Add("$EaseRating") }
        if ($HadGameLag) { $PyArgs.Add("--had-game-lag") }
        if ($FlushFeedbackQueue) { $PyArgs.Add("--flush-feedback-queue") }
        if ($Json) { $PyArgs.Add("--json") }
        if ($Mode -eq "GamerCheck") { $PyArgs.Add("--gamer-check") }
        if ($Mode -eq "GamerWhisper") { $PyArgs.Add("--gamer-whisper") }
    }
}

$DisplayArgs = New-Object System.Collections.Generic.List[string]
$MaskNext = $false
foreach ($argItem in $PyArgs) {
    if ($MaskNext) {
        $DisplayArgs.Add((Format-MaskedApiKey -RawKey $argItem))
        $MaskNext = $false
    } else {
        $DisplayArgs.Add($argItem)
        if ($argItem -eq "--api-key") {
            $MaskNext = $true
        }
    }
}

$PythonExe = Resolve-PythonExecutable

if ($DryRun) {
    $DryObj = [ordered]@{
        dry_run           = $true
        compliance_ok     = $true
        mode              = $Mode
        lang              = $Lang
        gateway           = $CleanGateway
        target_script     = $TargetScriptName
        bootstrap_planned = $BootstrapPlanned
        python_found      = ($null -ne $PythonExe)
        argv              = @($DisplayArgs)
        personal_pc_only  = $true
        financial_promise = "0.00 EUR / 0.00 USD"
        license           = "AGPL-3.0"
    }
    if ($Json) {
        Write-Output ($DryObj | ConvertTo-Json -Depth 4)
    } else {
        Write-Host "[DryRun] Mode=$Mode Target=$TargetScriptName Gateway=$CleanGateway (0.00 EUR / 0.00 USD)"
        Write-Host "[DryRun] Planned argv: $($DisplayArgs -join ' ')"
    }
    exit 0
}

if ($null -eq $PythonExe) {
    Write-Error "Python 3.10+ was not found on PATH. Install Python from python.org or winget install Python.Python.3.12."
    exit 2
}

& $PythonExe @PyArgs
exit $LASTEXITCODE
