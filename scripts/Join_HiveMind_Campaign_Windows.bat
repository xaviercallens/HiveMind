@echo off
REM ============================================================================
REM Project HiveMind - 1-Click Non-Technical User Launcher (Windows)
REM Operation Polyglot-Singularity-128 (8-Domain Hugging Face & 128-Expert MoE)
REM
REM NON-MONETARY NOTICE (0.00 EUR / 0.00 USD):
REM No financial payment, money, or cryptocurrency is ever offered or paid.
REM Symbolic 'points' (PTS / $HIVE) confer exclusively reciprocal computing
REM processing benefits when the live production HiveMind Swarm is deployed.
REM ============================================================================

title Project HiveMind - 1-Click Community Campaign Launcher (v1.8.0)
cd /d "%~dp0\.."

echo ============================================================================
echo   PROJECT HIVEMIND - 1-CLICK CITIZEN SCIENCE ^& GPU CAMPAIGN LAUNCHER
echo   Operation Polyglot-Singularity-128 (Safe ^< 1.5 GB VRAM Envelope)
echo ============================================================================
echo.
echo [LEGAL NOTICE] 0.00 EUR / 0.00 USD: There is NO financial payment or money.
echo Earned points (PTS / $HIVE) grant only reciprocal AI computing processing
echo benefits when the live production HiveMind Swarm is deployed and shared.
echo.

set /p HANDLE="Enter your GitHub username or nickname (press Enter for '%USERNAME%'): "
if "%HANDLE%"=="" set HANDLE=%USERNAME%

echo.
echo [Step 1/2] Running the 8-Domain Hugging Face ^& 128-Expert Multi-MoE Campaign...
if exist ".venv\Scripts\python.exe" (
    set PYTHON_BIN=.venv\Scripts\python.exe
) else (
    set PYTHON_BIN=python
)

set PYTHONPATH=src
%PYTHON_BIN% scripts\run_multi_moe_hf_campaign.py --contributor-handle "%HANDLE%" --tier standard > artifacts\latest_user_campaign_summary.md
echo. >> artifacts\latest_user_campaign_summary.md
%PYTHON_BIN% scripts\run_season9_10_gpu_swarm_campaigns.py --contributor-handle "%HANDLE%" >> artifacts\latest_user_campaign_summary.md
type artifacts\latest_user_campaign_summary.md

if exist "%SystemRoot%\System32\clip.exe" (
    type artifacts\latest_user_campaign_summary.md | clip
    echo.
    echo [Step 2/2] SUCCESS! Your Campaign Receipt has been COPIED to your clipboard!
    echo Just press Ctrl+V in the GitHub Discussion page that opens now:
) else (
    echo.
    echo [Step 2/2] SUCCESS! Copy the table above and paste it into GitHub Discussions:
)

echo   https://github.com/xaviercallens/HiveMind/discussions/86
start "" "https://github.com/xaviercallens/HiveMind/discussions/86"
echo.
pause
