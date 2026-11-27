@echo off
REM Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
REM ============================================================================
REM Project HiveMind - Unified Windows 11 Quick-Connect & Gamer-Whisper Launcher
REM
REM Delegates cleanly on Windows 11 to scripts\join_hivemind.ps1
REM   (-ExecutionPolicy Bypass -NoProfile -File scripts\join_hivemind.ps1)
REM or falls back to Python standard-library scripts\hivemind_quickstart.py
REM in Connect (default), GamerWhisper, GamerCheck, or Uninstall mode.
REM
REM AMADEUS / PERSONAL-PC COMPLIANCE & NON-MONETARY NOTICE (0.00 EUR / 0.00 USD):
REM   - 100% personal home PCs and personal hobby/open-source projects ONLY.
REM   - NEVER install or run on corporate/work machines (Amadeus compliance rule)
REM     and NEVER send confidential, proprietary, or company code/data.
REM   - 0.00 EUR / 0.00 USD: zero monetary value. Symbolic points confer
REM     exclusively reciprocal open-source compute benefits.
REM ============================================================================
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"

if exist "%REPO_ROOT%\scripts\hivemind_quickstart.py" (
    cd /d "%REPO_ROOT%"
    set "PS1_SCRIPT=scripts\join_hivemind.ps1"
    set "QUICKSTART_PY=scripts\hivemind_quickstart.py"
) else (
    cd /d "%SCRIPT_DIR%"
    set "PS1_SCRIPT=%SCRIPT_DIR%join_hivemind.ps1"
    set "QUICKSTART_PY=%SCRIPT_DIR%hivemind_quickstart.py"
)

if exist "%REPO_ROOT%\src\hivemind" (
    if "%PYTHONPATH%"=="" set "PYTHONPATH=%REPO_ROOT%\src"
)

set "HAD_CLI_ARGS=0"
if not "%~1"=="" set "HAD_CLI_ARGS=1"

set "MODE=Connect"
set "GATEWAY_URL=%HIVEMIND_GATEWAY_URL%"
if "%GATEWAY_URL%"=="" set "GATEWAY_URL=https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
set "RESERVE_MIB=%HIVEMIND_RESERVE_MIB%"
if "%RESERVE_MIB%"=="" set "RESERVE_MIB=2048"
set "HOT_C=%HIVEMIND_HOT_C%"
if "%HOT_C%"=="" set "HOT_C=70"

set "HANDLE_VAL="
set "API_KEY_VAL="
set "GAME_VAL="
set "WRITE_CONTINUE=default"
set "ONCE_MODE=default"
set "NO_PAUSE=0"
set "QUIET_JSON=0"
set "NEEDS_PYTHON_DIRECT=0"
set "PY_EXTRA_ARGS="

:parse_args
if "%~1"=="" goto :after_parse

if /i "%~1"=="-h" goto :show_help
if /i "%~1"=="--help" goto :show_help
if "%~1"=="/?" goto :show_help

if /i "%~1"=="--connect" (
    set "MODE=Connect"
    shift
    goto :parse_args
)
if /i "%~1"=="--gamer-whisper" (
    set "MODE=GamerWhisper"
    shift
    goto :parse_args
)
if /i "%~1"=="--gamer-check" (
    set "MODE=GamerCheck"
    shift
    goto :parse_args
)
if /i "%~1"=="--uninstall" (
    set "MODE=Uninstall"
    shift
    goto :parse_args
)

if /i "%~1"=="-Mode" goto :arg_mode
if /i "%~1"=="--mode" goto :arg_mode
if /i "%~1"=="--gateway" goto :arg_gateway
if /i "%~1"=="-Gateway" goto :arg_gateway
if /i "%~1"=="--handle" goto :arg_handle
if /i "%~1"=="-Handle" goto :arg_handle
if /i "%~1"=="--api-key" goto :arg_apikey
if /i "%~1"=="-ApiKey" goto :arg_apikey
if /i "%~1"=="--reserve-mib" goto :arg_reserve
if /i "%~1"=="-ReserveMib" goto :arg_reserve
if /i "%~1"=="--hot-c" goto :arg_hotc
if /i "%~1"=="-HotC" goto :arg_hotc
if /i "%~1"=="--game" goto :arg_game
if /i "%~1"=="-Game" goto :arg_game

if /i "%~1"=="--write-continue-config" (
    set "WRITE_CONTINUE=1"
    shift
    goto :parse_args
)
if /i "%~1"=="-WriteContinueConfig" (
    set "WRITE_CONTINUE=1"
    shift
    goto :parse_args
)
if /i "%~1"=="--no-write-continue-config" (
    set "WRITE_CONTINUE=0"
    shift
    goto :parse_args
)
if /i "%~1"=="--once" (
    set "ONCE_MODE=1"
    shift
    goto :parse_args
)
if /i "%~1"=="-Once" (
    set "ONCE_MODE=1"
    shift
    goto :parse_args
)
if /i "%~1"=="--continuous" (
    set "ONCE_MODE=0"
    shift
    goto :parse_args
)
if /i "%~1"=="--no-pause" (
    set "NO_PAUSE=1"
    shift
    goto :parse_args
)
if /i "%~1"=="--json" (
    set "QUIET_JSON=1"
    set "NEEDS_PYTHON_DIRECT=1"
    set "PY_EXTRA_ARGS=%PY_EXTRA_ARGS% --json"
    shift
    goto :parse_args
)

set "NEEDS_PYTHON_DIRECT=1"
set "PY_EXTRA_ARGS=%PY_EXTRA_ARGS% "%~1""
shift
goto :parse_args

:arg_mode
if "%~2"=="" goto :missing_value
if /i "%~2"=="Connect" (
    set "MODE=Connect"
    shift
    shift
    goto :parse_args
)
if /i "%~2"=="GamerWhisper" (
    set "MODE=GamerWhisper"
    shift
    shift
    goto :parse_args
)
if /i "%~2"=="gamer-whisper" (
    set "MODE=GamerWhisper"
    shift
    shift
    goto :parse_args
)
if /i "%~2"=="GamerCheck" (
    set "MODE=GamerCheck"
    shift
    shift
    goto :parse_args
)
if /i "%~2"=="gamer-check" (
    set "MODE=GamerCheck"
    shift
    shift
    goto :parse_args
)
if /i "%~2"=="Uninstall" (
    set "MODE=Uninstall"
    shift
    shift
    goto :parse_args
)
echo [ERROR] Invalid mode "%~2". Expected Connect, GamerWhisper, GamerCheck, or Uninstall. 1>&2
exit /b 2

:arg_gateway
if "%~2"=="" goto :missing_value
set "GATEWAY_URL=%~2"
shift
shift
goto :parse_args

:arg_handle
if "%~2"=="" goto :missing_value
set "HANDLE_VAL=%~2"
shift
shift
goto :parse_args

:arg_apikey
if "%~2"=="" goto :missing_value
set "API_KEY_VAL=%~2"
shift
shift
goto :parse_args

:arg_reserve
if "%~2"=="" goto :missing_value
set "RESERVE_MIB=%~2"
shift
shift
goto :parse_args

:arg_hotc
if "%~2"=="" goto :missing_value
set "HOT_C=%~2"
shift
shift
goto :parse_args

:arg_game
if "%~2"=="" goto :missing_value
if not "%GAME_VAL%"=="" set "NEEDS_PYTHON_DIRECT=1"
set "GAME_VAL=%~2"
set "PY_EXTRA_ARGS=%PY_EXTRA_ARGS% --game "%~2""
shift
shift
goto :parse_args

:missing_value
echo [ERROR] Missing required value after "%~1". Run with --help for usage. 1>&2
exit /b 2

:show_help
echo ============================================================================
echo   PROJECT HIVEMIND - WINDOWS 11 QUICK-CONNECT ^& GAMER-WHISPER LAUNCHER
echo ============================================================================
echo   [COMPLIANCE] 100%% personal home PCs ^& personal hobby/open-source projects ONLY.
echo   [AMADEUS RULE] NEVER install on corporate/work PCs or send confidential/company code.
echo   [NON-MONETARY] 0.00 EUR / 0.00 USD (zero monetary value; reciprocal compute only).
echo.
echo Usage:
echo   scripts\Join_HiveMind_Campaign_Windows.bat [options]
echo.
echo Modes (double-clicking with no flags runs --connect):
echo   --connect                  Register/load ~/.hivemind/credentials.json ^& VS Code config (default)
echo   --gamer-whisper            Quiet gaming PC mode (^<= 70 C cap, 2048 MiB reserve, ^< 1.5s game stop)
echo   --gamer-check              Inspect NVIDIA GPU, temperature, and running Steam/PC games only
echo   --uninstall                Remove local saved HiveMind credentials ^& Continue snippet cleanly
echo.
echo Options:
echo   --handle NAME              Desired personal handle (2-48 chars)
echo   --api-key KEY              Existing hm-... personal API key
echo   --gateway URL              HiveMind HTTPS *.run.app gateway or localhost
echo   --reserve-mib MIB          Free VRAM MiB reserved for desktop/games (default: 2048)
echo   --hot-c CELSIUS            Quiet fan temperature cap in Celsius (default: 70)
echo   --game EXE                 Extra game executable to auto-yield on (e.g. cs2.exe)
echo   --write-continue-config    Write ~/.continue/hivemind_provider.yaml
echo   --no-write-continue-config Print Continue.dev snippet without writing ~/.continue file
echo   --once                     Run a single quiet Gamer-Whisper round and exit
echo   --continuous               Keep Gamer-Whisper running across rounds instead of --once
echo   --no-pause                 Never pause before exit
echo   -h, --help, /?             Show this help message and exit
exit /b 0

:after_parse
if "%WRITE_CONTINUE%"=="default" (
    if /i "%MODE%"=="Connect" (
        set "WRITE_CONTINUE=1"
    ) else (
        set "WRITE_CONTINUE=0"
    )
)
if "%ONCE_MODE%"=="default" (
    if /i "%MODE%"=="GamerWhisper" (
        set "ONCE_MODE=1"
    ) else (
        set "ONCE_MODE=0"
    )
)

if "%QUIET_JSON%"=="0" (
    echo ============================================================================
    echo   PROJECT HIVEMIND - WINDOWS 11 QUICK-CONNECT ^& GAMER-WHISPER LAUNCHER
    echo   [COMPLIANCE] 100%% personal home PCs ^& personal hobby/open-source projects ONLY.
    echo   [AMADEUS RULE] NEVER run on corporate/work PCs or send confidential/company code.
    echo   [NON-MONETARY] 0.00 EUR / 0.00 USD: zero monetary value ^(reciprocal compute only^).
    echo ============================================================================
)

set "PS_HANDLE_ARG="
if not "%HANDLE_VAL%"=="" set "PS_HANDLE_ARG=-Handle "%HANDLE_VAL%""
set "PS_APIKEY_ARG="
if not "%API_KEY_VAL%"=="" set "PS_APIKEY_ARG=-ApiKey "%API_KEY_VAL%""
set "PS_GAME_ARG="
if not "%GAME_VAL%"=="" set "PS_GAME_ARG=-Game "%GAME_VAL%""
set "PS_WRITE_ARG="
if "%WRITE_CONTINUE%"=="1" set "PS_WRITE_ARG=-WriteContinueConfig"
set "PS_ONCE_ARG="
if "%ONCE_MODE%"=="1" set "PS_ONCE_ARG=-Once"

set "PY_MODE_ARG="
if /i "%MODE%"=="GamerWhisper" set "PY_MODE_ARG=--gamer-whisper"
if /i "%MODE%"=="GamerCheck" set "PY_MODE_ARG=--gamer-check"
if /i "%MODE%"=="Uninstall" set "PY_MODE_ARG=--uninstall"
set "PY_HANDLE_ARG="
if not "%HANDLE_VAL%"=="" set "PY_HANDLE_ARG=--handle "%HANDLE_VAL%""
set "PY_APIKEY_ARG="
if not "%API_KEY_VAL%"=="" set "PY_APIKEY_ARG=--api-key "%API_KEY_VAL%""
set "PY_WRITE_ARG="
if "%WRITE_CONTINUE%"=="1" set "PY_WRITE_ARG=--write-continue-config"
set "PY_ONCE_ARG="
if "%ONCE_MODE%"=="1" set "PY_ONCE_ARG=--once"

if "%HIVEMIND_FORCE_PYTHON_FALLBACK%"=="1" goto :run_python_fallback
if "%NEEDS_PYTHON_DIRECT%"=="1" goto :run_python_fallback
if not exist "%PS1_SCRIPT%" goto :run_python_fallback

set "PS_EXE="
if exist "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" (
    set "PS_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
)
if "%PS_EXE%"=="" (
    where powershell.exe >nul 2>nul
    if not errorlevel 1 set "PS_EXE=powershell.exe"
)
if "%PS_EXE%"=="" (
    where pwsh.exe >nul 2>nul
    if not errorlevel 1 set "PS_EXE=pwsh.exe"
)
if "%PS_EXE%"=="" goto :run_python_fallback

:run_powershell
"%PS_EXE%" -ExecutionPolicy Bypass -NoProfile -File "%PS1_SCRIPT%" -Mode %MODE% -Gateway "%GATEWAY_URL%" -ReserveMib %RESERVE_MIB% -HotC %HOT_C% %PS_HANDLE_ARG% %PS_APIKEY_ARG% %PS_WRITE_ARG% %PS_ONCE_ARG% %PS_GAME_ARG%
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="9009" goto :run_python_fallback
goto :finish

:run_python_fallback
if not exist "%QUICKSTART_PY%" (
    echo [ERROR] Missing "%QUICKSTART_PY%" alongside Join_HiveMind_Campaign_Windows.bat. 1>&2
    set "EXIT_CODE=2"
    goto :finish
)

set "PYTHON_BIN="
if exist "%REPO_ROOT%\.venv\Scripts\python.exe" set "PYTHON_BIN=%REPO_ROOT%\.venv\Scripts\python.exe"
if "%PYTHON_BIN%"=="" (
    where python >nul 2>nul
    if not errorlevel 1 set "PYTHON_BIN=python"
)
if "%PYTHON_BIN%"=="" (
    where py >nul 2>nul
    if not errorlevel 1 set "PYTHON_BIN=py"
)
if "%PYTHON_BIN%"=="" (
    echo [ERROR] Python 3.10+ was not found on PATH. Install Python from python.org or run: winget install Python.Python.3.12 1>&2
    set "EXIT_CODE=2"
    goto :finish
)

REM Connect is the default mode of scripts\hivemind_quickstart.py (--connect -> Connect mode)
"%PYTHON_BIN%" "%QUICKSTART_PY%" --gateway "%GATEWAY_URL%" --reserve-mib %RESERVE_MIB% --hot-c %HOT_C% %PY_MODE_ARG% %PY_HANDLE_ARG% %PY_APIKEY_ARG% %PY_WRITE_ARG% %PY_ONCE_ARG% %PY_EXTRA_ARGS%
set "EXIT_CODE=%ERRORLEVEL%"
goto :finish

:finish
if "%NO_PAUSE%"=="1" exit /b %EXIT_CODE%
if "%HIVEMIND_NO_PAUSE%"=="1" exit /b %EXIT_CODE%
if "%HAD_CLI_ARGS%"=="1" exit /b %EXIT_CODE%
echo %CMDCMDLINE% | findstr /i /c:" /c " >nul 2>nul
if not errorlevel 1 pause
exit /b %EXIT_CODE%
