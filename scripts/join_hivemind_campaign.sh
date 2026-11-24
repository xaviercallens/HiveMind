#!/usr/bin/env bash
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
# ============================================================================
# Project HiveMind - Turnkey Citizen-Science Volunteer Onboarding (macOS / Linux)
# Runs real hardware/VRAM preflight -> Quick-Connect / Gamer-Whisper / Doctor /
# DeploySmoke / VS Code Setup / Gamer-Service / OpenTrainer with strict
# free-VRAM reserve & thermal guard (auto-yield < 1.5 s).
#
# PERSONAL HOME PC & NON-MONETARY NOTICE (0.00 EUR / 0.00 USD):
# Personal home machines and personal hobby/open-source projects ONLY (never
# corporate hardware or work code). No financial payment is ever offered/paid.
# ============================================================================
set -euo pipefail

CALLER_PWD="$PWD"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

GATEWAY_URL="${HIVEMIND_GATEWAY_URL:-https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app}"
RESERVE_MIB="${HIVEMIND_RESERVE_MIB:-1536}"
RESERVE_MIB_SET=0
HOT_C="${HIVEMIND_HOT_C:-82}"
HOT_C_SET=0
COOL_C="${HIVEMIND_COOL_C:-74}"
MODE="${HIVEMIND_JOIN_MODE:-opentrainer}" # connect | gamer-whisper | gamer-check | doctor | deploy-smoke | vscode-setup | gamer-service | uninstall | opentrainer | train | crowdgpt | legacy-demo
ONCE_FLAG="--once"
HANDLE=""
API_KEY=""
LANG_CODE="en"
EXPECTED_SHA=""
EASE_RATING=0
HAD_GAME_LAG=0
WRITE_CONTINUE_CONFIG=0
SMOKE_ROUTES=0
CHECK_DEPLOY_PARITY=0
POST_SETUP_TELEMETRY=0
FLUSH_FEEDBACK_QUEUE=0
BOOTSTRAP=0
DRY_RUN=0
JSON_OUT=0
GAMES=()
EXTRA_ARGS=()

PERSONAL_PC_NOTICE_EN="100% personal home PCs & personal hobby/open-source projects only (Amadeus / Personal-PC rule: 0.00 EUR / 0.00 USD, AGPL-3.0)."
PERSONAL_PC_NOTICE_FR="100 % PC personnels a domicile et projets hobby/open-source uniquement (regle Amadeus / PC perso : 0,00 EUR / 0,00 USD, AGPL-3.0)."

ALLOWED_BOOTSTRAP_SCRIPTS=(
    "hivemind_boinc_idle_worker.py"
    "hivemind_connect.py"
    "hivemind_doctor.py"
    "hivemind_quickstart.py"
    "hivemind_producer_agent.py"
    "hivemind_producer_shield.py"
    "hivemind_producer_autostart.py"
    "hivemind_volunteer_config.py"
    "hivemind_volunteer_preflight.py"
    "hivemind_opentrainer_agent.py"
    "join_hivemind.ps1"
    "noel_canary_cohort_rehearsal.py"
    "noel_cloud_run_parity_gate.py"
    "noel_gamer_whisper_selftest.py"
    "noel_post_setup_probe.py"
    "noel_scale_load_harness.py"
    "noel_vscode_autoconfig.py"
)

print_help() {
    cat <<'EOF'
Usage: ./scripts/join_hivemind_campaign.sh [options]

Turnkey 1-command HiveMind Personal Onboarding (Personal Home PCs Only • 0.00 EUR / 0.00 USD):
  1. Quick < 60 s Developer Connect (`--connect`): gets/loads ~/.hivemind/credentials.json & VS Code config
  2. Quiet PC Gamer Mode (`--gamer-whisper`): <= 70 C cap, 2048 MiB free VRAM, < 1.5 s stop on Steam game
  3. Holiday Preflight & Smoke (`--doctor`, `--deploy-smoke`), VS Code (`--vscode-setup`), Gamer Service (`--gamer-service`)
  4. OpenTrainer / CrowdGPT GPU contributor (`--train` or `--crowdgpt`)

Options:
  --connect                Register/load personal API key in < 60 s and write VS Code (Continue.dev) snippet
  --gamer-whisper          Quiet home gaming PC mode (70 C cap, 2048 MiB reserve, instant stop on game launch)
  --gamer-check            Inspect NVIDIA GPU, temperature, and running games without starting training
  --doctor                 Run hivemind_doctor.py --noel holiday preflight, yield/thermal & fleet checks
  --deploy-smoke           Run Cloud Run holiday smoke gate (locally or parity gate with --bootstrap)
  --vscode-setup           Run < 60 s VS Code setup (quick preset locally or autoconfig with --bootstrap)
  --gamer-service          Run opt-in user-space idle-resume GPU service (noel_idle_resume_service.py)
  --uninstall              Cleanly remove saved local HiveMind credentials/config (always permitted)
  --train                  Run a real OpenTrainer round (auto-loads ~/.hivemind/credentials.json if present)
  --crowdgpt               Contribute to Crowd-v1 via CrowdGPT client under HiveMind VRAM & thermal guards
  --continuous             Keep running heartbeats/training loop instead of single `--once` pass
  --once                   Run a single cycle and exit
  --legacy-demo            Run legacy local Season 9/10 simulation demo
  --gateway URL            HiveMind gateway bare origin (https://*.run.app or loopback)
  --handle HANDLE          Personal contributor handle (2-48 chars)
  --api-key KEY            Existing personal hm-... / hm_live_... API key (masked in dry-run outputs)
  --reserve-mib N          Free VRAM MiB always reserved for your desktop/games (default: 1536, gamer-whisper: 2048)
  --hot-c N                Thermal cap in Celsius (default: 82, gamer-whisper/doctor/service: 70)
  --cool-c N               Cool-down resume threshold in Celsius (default: 74)
  --game EXE               Extra game executable name to auto-yield for (repeatable)
  --expected-sha SHA       Expected Cloud Run manifest build_sha
  --write-continue-config  Write ~/.continue/hivemind_provider.yaml
  --smoke-routes           Run zero-inference Cloud Run holiday route smoke check
  --check-deploy-parity    Run Cloud Run deploy parity check
  --post-setup-telemetry   Submit non-blocking zero-PII CES onboarding feedback
  --ease-rating N          CES ease-of-setup rating from 1 to 5
  --had-game-lag           Report that game stutter/lag occurred (zero_game_lag=false)
  --flush-feedback-queue   Flush spooled offline CES feedback entries idempotently
  --bootstrap              Fetch missing allowlisted helper from $GATEWAY_URL/scripts/<name> via stdlib ProxyHandler({})
  --lang en|fr             Output language (default: en)
  --dry-run                Validate compliance/gateway and print planned execution without network calls or writes
  --json                   Emit machine-readable JSON output
  -h, --help               Show this help and exit

STRICT RULE: 100% personal home PCs & personal hobby/open-source projects ONLY (0.00 EUR / 0.00 USD, AGPL-3.0).
EOF
}

json_escape() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    s="${s//$'\n'/\\n}"
    s="${s//$'\r'/\\r}"
    s="${s//$'\t'/\\t}"
    printf '"%s"' "$s"
}

json_bool() {
    if [ "$1" -eq 1 ]; then
        printf 'true'
    else
        printf 'false'
    fi
}

trim_ws() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

to_lower() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

to_upper() {
    printf '%s' "$1" | tr '[:lower:]' '[:upper:]'
}

format_masked_api_key() {
    local raw=""
    raw="$(trim_ws "$1")"
    if [ -z "$raw" ]; then
        printf ''
        return 0
    fi
    local len="${#raw}"
    if [ "$len" -lt 12 ]; then
        printf '<redacted>'
        return 0
    fi
    local tail="${raw:len-4:4}"
    case "$raw" in
        hm_live_*) printf 'hm_live_***%s' "$tail" ;;
        hm-*)      printf 'hm-***%s' "$tail" ;;
        *)         printf '<redacted>' ;;
    esac
}

redact_home_prefix() {
    local val="$1"
    if [ -n "${HOME:-}" ] && [ "$HOME" != "/" ]; then
        case "$val" in
            "$HOME")
                printf '~'
                return 0
                ;;
            "$HOME"/*)
                printf '~%s' "${val#"$HOME"}"
                return 0
                ;;
        esac
    fi
    printf '%s' "$val"
}

emit_error_or_json() {
    local exit_code="$1"
    local field_flag="$2"
    local msg="$3"
    if [ "$JSON_OUT" -eq 1 ]; then
        printf '{\n'
        printf '  "compliance_ok": %s,\n' "$([ "$field_flag" = "compliance" ] && printf 'false' || printf 'true')"
        if [ "$field_flag" = "gateway" ]; then
            printf '  "gateway_valid": false,\n'
        fi
        printf '  "personal_pc_only": true,\n'
        printf '  "financial_promise": "0.00 EUR / 0.00 USD",\n'
        printf '  "license": "AGPL-3.0",\n'
        printf '  "exit_code": %s,\n' "$exit_code"
        printf '  "error": %s\n' "$(json_escape "$msg")"
        printf '}\n'
    else
        printf 'Error: %s\n' "$msg" >&2
    fi
    exit "$exit_code"
}

require_value_arg() {
    local flag="$1"
    local count="$2"
    if [ "$count" -lt 2 ]; then
        emit_error_or_json 2 "validation" "Option ${flag} requires a value."
    fi
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --help|-h)
            print_help
            exit 0
            ;;
        --connect) MODE="connect" ;;
        --gamer-whisper) MODE="gamer-whisper" ;;
        --gamer-check) MODE="gamer-check" ;;
        --doctor) MODE="doctor" ;;
        --deploy-smoke) MODE="deploy-smoke" ;;
        --vscode-setup) MODE="vscode-setup" ;;
        --gamer-service) MODE="gamer-service" ;;
        --uninstall) MODE="uninstall" ;;
        --train) MODE="train" ;;
        --crowdgpt) MODE="crowdgpt" ;;
        --legacy-demo) MODE="legacy-demo" ;;
        --continuous) ONCE_FLAG="" ;;
        --once) ONCE_FLAG="--once" ;;
        --write-continue-config) WRITE_CONTINUE_CONFIG=1 ;;
        --smoke-routes) SMOKE_ROUTES=1 ;;
        --check-deploy-parity) CHECK_DEPLOY_PARITY=1 ;;
        --post-setup-telemetry) POST_SETUP_TELEMETRY=1 ;;
        --had-game-lag) HAD_GAME_LAG=1 ;;
        --flush-feedback-queue) FLUSH_FEEDBACK_QUEUE=1 ;;
        --bootstrap) BOOTSTRAP=1 ;;
        --dry-run) DRY_RUN=1 ;;
        --json) JSON_OUT=1 ;;
        --reserve-mib)
            require_value_arg "$1" "$#"
            shift
            RESERVE_MIB="$1"
            RESERVE_MIB_SET=1
            ;;
        --reserve-mib=*)
            RESERVE_MIB="${1#*=}"
            RESERVE_MIB_SET=1
            ;;
        --gateway)
            require_value_arg "$1" "$#"
            shift
            GATEWAY_URL="$1"
            ;;
        --gateway=*)
            GATEWAY_URL="${1#*=}"
            ;;
        --handle)
            require_value_arg "$1" "$#"
            shift
            HANDLE="$1"
            ;;
        --handle=*)
            HANDLE="${1#*=}"
            ;;
        --api-key)
            require_value_arg "$1" "$#"
            shift
            API_KEY="$1"
            ;;
        --api-key=*)
            API_KEY="${1#*=}"
            ;;
        --lang)
            require_value_arg "$1" "$#"
            shift
            LANG_CODE="$1"
            ;;
        --lang=*)
            LANG_CODE="${1#*=}"
            ;;
        --ease-rating)
            require_value_arg "$1" "$#"
            shift
            EASE_RATING="$1"
            ;;
        --ease-rating=*)
            EASE_RATING="${1#*=}"
            ;;
        --hot-c)
            require_value_arg "$1" "$#"
            shift
            HOT_C="$1"
            HOT_C_SET=1
            ;;
        --hot-c=*)
            HOT_C="${1#*=}"
            HOT_C_SET=1
            ;;
        --cool-c)
            require_value_arg "$1" "$#"
            shift
            COOL_C="$1"
            ;;
        --cool-c=*)
            COOL_C="${1#*=}"
            ;;
        --game)
            require_value_arg "$1" "$#"
            shift
            GAMES+=("$1")
            ;;
        --game=*)
            GAMES+=("${1#*=}")
            ;;
        --expected-sha)
            require_value_arg "$1" "$#"
            shift
            EXPECTED_SHA="$1"
            ;;
        --expected-sha=*)
            EXPECTED_SHA="${1#*=}"
            ;;
        *)
            EXTRA_ARGS+=("$1")
            ;;
    esac
    shift
done

# Normalize and validate language
LANG_LOW="$(printf '%s' "$LANG_CODE" | tr '[:upper:]' '[:lower:]' | tr '_' '-')"
case "$LANG_LOW" in
    en|en-us|en-gb) LANG_CODE="en" ;;
    fr|fr-fr|fr-be|fr-ca|fr-ch) LANG_CODE="fr" ;;
    *) emit_error_or_json 2 "validation" "Unsupported --lang '${LANG_CODE}'; expected 'en' or 'fr'." ;;
esac

# Validate ease rating if provided (0 means unset)
if ! [[ "$EASE_RATING" =~ ^[0-9]+$ ]] || [ "$EASE_RATING" -lt 0 ] || [ "$EASE_RATING" -gt 5 ]; then
    emit_error_or_json 2 "validation" "Invalid --ease-rating '${EASE_RATING}'; expected integer 1..5."
fi

# Route standalone quickstart utility switches if no explicit mode was picked
if [ "$MODE" = "opentrainer" ]; then
    if [ "$SMOKE_ROUTES" -eq 1 ] || [ "$CHECK_DEPLOY_PARITY" -eq 1 ] || [ "$FLUSH_FEEDBACK_QUEUE" -eq 1 ]; then
        MODE="connect"
    fi
fi

# 1. Local Personal-PC & Amadeus Compliance Gate (fail closed exit 2; ALWAYS permit --uninstall)
CORP_REGEX='amadeus|(^|[^a-z0-9])corp(orate|s)?($|[^a-z0-9])|[a-z0-9]corp($|[^a-z0-9])|enterprise|intranet|(^|[^a-z0-9])internal($|[^a-z0-9])|[a-z0-9]internal($|[^a-z0-9])|zscaler|netskope|bluecoat|fortinet|palo[-_.]?alto'
CORP_HITS=()
CHECKED_GIT_CFGS=()

add_corp_hit() {
    local tag="$1"
    local existing=""
    for existing in "${CORP_HITS[@]+"${CORP_HITS[@]}"}"; do
        if [ "$existing" = "$tag" ]; then
            return 0
        fi
    done
    CORP_HITS+=("$tag")
}

check_git_cfg_already_seen() {
    local cand="$1"
    local existing=""
    for existing in "${CHECKED_GIT_CFGS[@]+"${CHECKED_GIT_CFGS[@]}"}"; do
        if [ "$existing" = "$cand" ]; then
            return 0
        fi
    done
    CHECKED_GIT_CFGS+=("$cand")
    return 1
}

scan_git_config_file() {
    local cfg_path="$1"
    [ -f "$cfg_path" ] || return 0
    if check_git_cfg_already_seen "$cfg_path"; then
        return 0
    fi
    local line=""
    local line_low=""
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            *[Uu][Rr][Ll]*|*[Ii][Nn][Ss][Tt][Ee][Aa][Dd][Oo][Ff]*)
                line_low="$(to_lower "$line")"
                if [[ "$line_low" =~ ^[[:space:]]*(push)?(url|insteadof|pushinsteadof)[[:space:]]*= ]] || [[ "$line_low" =~ ^[[:space:]]*\[url[[:space:]]+\" ]]; then
                    if [[ "$line_low" =~ $CORP_REGEX ]]; then
                        add_corp_hit "git_remote"
                    fi
                fi
                ;;
        esac
    done < "$cfg_path"
}

resolve_git_configs_for_dir() {
    local curr="$1"
    [ -n "$curr" ] && [ -d "$curr" ] || return 0
    curr="$(cd "$curr" 2>/dev/null && pwd || printf '%s' "$curr")"
    while [ -n "$curr" ]; do
        if [ -d "$curr/.git" ]; then
            scan_git_config_file "$curr/.git/config"
            break
        elif [ -f "$curr/.git" ]; then
            local first_line=""
            IFS= read -r first_line < "$curr/.git" || first_line=""
            local first_low=""
            first_low="$(to_lower "$first_line")"
            if [[ "$first_low" =~ ^[[:space:]]*gitdir:[[:space:]]*(.+)$ ]]; then
                local raw_gitdir="${first_line#*:}"
                raw_gitdir="$(trim_ws "$raw_gitdir")"
                local gitdir="$raw_gitdir"
                if [[ "$raw_gitdir" != /* ]]; then
                    gitdir="$(cd "$curr/$raw_gitdir" 2>/dev/null && pwd || printf '%s' "$curr/$raw_gitdir")"
                fi
                scan_git_config_file "$gitdir/config"
                if [ -f "$gitdir/commondir" ]; then
                    local common_rel=""
                    IFS= read -r common_rel < "$gitdir/commondir" || common_rel=""
                    common_rel="$(trim_ws "$common_rel")"
                    if [ -n "$common_rel" ]; then
                        local common_dir="$common_rel"
                        if [[ "$common_rel" != /* ]]; then
                            common_dir="$(cd "$gitdir/$common_rel" 2>/dev/null && pwd || printf '%s' "$gitdir/$common_rel")"
                        fi
                        scan_git_config_file "$common_dir/config"
                    fi
                fi
            fi
            break
        fi
        local parent=""
        parent="$(dirname "$curr")"
        if [ -z "$parent" ] || [ "$parent" = "$curr" ]; then
            break
        fi
        curr="$parent"
    done
}

check_personal_pc_compliance() {
    local k=""
    local ku=""
    local v=""
    local v_low=""

    while IFS= read -r k; do
        [ -n "$k" ] || continue
        case "$k" in
            [Aa][Mm][Aa][Dd][Ee][Uu][Ss]*|[Zz][Ss][Cc][Aa][Ll][Ee][Rr]_[Cc][Ee][Rr][Tt]_[Pp][Aa][Tt][Hh]|[Uu][Ss][Ee][Rr][Dd][Nn][Ss][Dd][Oo][Mm][Aa][Ii][Nn]|[Uu][Ss][Ee][Rr][Dd][Oo][Mm][Aa][Ii][Nn]|[Hh][Tt][Tt][Pp]_[Pp][Rr][Oo][Xx][Yy]|[Hh][Tt][Tt][Pp][Ss]_[Pp][Rr][Oo][Xx][Yy]|[Aa][Ll][Ll]_[Pp][Rr][Oo][Xx][Yy]|[Nn][Oo]_[Pp][Rr][Oo][Xx][Yy]|[Cc][Oo][Mm][Pp][Uu][Tt][Ee][Rr][Nn][Aa][Mm][Ee]|[Hh][Oo][Ss][Tt][Nn][Aa][Mm][Ee]|[Ss][Ss][Ll]_[Cc][Ee][Rr][Tt]_[Ff][Ii][Ll][Ee]|[Rr][Ee][Qq][Uu][Ee][Ss][Tt][Ss]_[Cc][Aa]_[Bb][Uu][Nn][Dd][Ll][Ee]|[Cc][Uu][Rr][Ll]_[Cc][Aa]_[Bb][Uu][Nn][Dd][Ll][Ee])
                v="${!k-}"
                v="$(trim_ws "$v")"
                [ -n "$v" ] || continue
                ku="$(to_upper "$k")"
                case "$ku" in
                    AMADEUS*)
                        add_corp_hit "env:AMADEUS*"
                        ;;
                    ZSCALER_CERT_PATH)
                        add_corp_hit "env:ZSCALER_CERT_PATH"
                        ;;
                    USERDNSDOMAIN)
                        add_corp_hit "env:USERDNSDOMAIN"
                        ;;
                    *)
                        v_low="$(to_lower "$v")"
                        if [[ "$v_low" =~ $CORP_REGEX ]]; then
                            add_corp_hit "env:${ku}"
                        fi
                        ;;
                esac
                ;;
        esac
    done < <(compgen -e)

    # Also inspect non-exported HOSTNAME if set in shell
    if [ -n "${HOSTNAME:-}" ]; then
        v_low="$(to_lower "$HOSTNAME")"
        if [[ "$v_low" =~ $CORP_REGEX ]]; then
            add_corp_hit "env:HOSTNAME"
        fi
    fi

    local ws_candidate=""
    local ws_low=""
    for ws_candidate in "$CALLER_PWD" "$SCRIPT_DIR" "$ROOT_DIR"; do
        [ -n "$ws_candidate" ] || continue
        ws_low="$(to_lower "$ws_candidate")"
        if [[ "$ws_low" =~ $CORP_REGEX ]]; then
            add_corp_hit "workspace_path"
        fi
    done

    resolve_git_configs_for_dir "$CALLER_PWD"
    resolve_git_configs_for_dir "$ROOT_DIR"

    if [ "${#CORP_HITS[@]}" -gt 0 ]; then
        local hit_list=""
        local h=""
        for h in "${CORP_HITS[@]}"; do
            hit_list="${hit_list:+${hit_list}, }${h}"
        done
        local block_msg=""
        if [ "$LANG_CODE" = "fr" ]; then
            block_msg="REFUS DE CONFORMITE (${hit_list}) : environnement d'entreprise detecte (0 appel reseau, 0 ecriture). ${PERSONAL_PC_NOTICE_FR}"
        else
            block_msg="COMPLIANCE BLOCKED (${hit_list}): corporate environment indicator detected (0 network calls, 0 writes). ${PERSONAL_PC_NOTICE_EN}"
        fi
        if [ "$JSON_OUT" -eq 1 ]; then
            printf '{\n'
            printf '  "compliance_ok": false,\n'
            printf '  "corporate_detected": true,\n'
            printf '  "corporate_indicators": ['
            local first=1
            for h in "${CORP_HITS[@]}"; do
                [ "$first" -eq 1 ] && first=0 || printf ', '
                json_escape "$h"
            done
            printf '],\n'
            printf '  "personal_pc_only": true,\n'
            printf '  "financial_promise": "0.00 EUR / 0.00 USD",\n'
            printf '  "license": "AGPL-3.0",\n'
            printf '  "notice_en": %s,\n' "$(json_escape "$PERSONAL_PC_NOTICE_EN")"
            printf '  "notice_fr": %s,\n' "$(json_escape "$PERSONAL_PC_NOTICE_FR")"
            printf '  "exit_code": 2,\n'
            printf '  "summary": %s\n' "$(json_escape "$block_msg")"
            printf '}\n'
        else
            printf '%s\n' "$block_msg" >&2
        fi
        exit 2
    fi
}

if [ "$MODE" != "uninstall" ]; then
    check_personal_pc_compliance
fi

# 2. Validate bare-origin Gateway URL allowlist (https://*.run.app or loopback)
validate_gateway_url() {
    local raw="$1"
    local trimmed=""
    trimmed="$(trim_ws "$raw")"
    if [ -z "$trimmed" ]; then
        emit_error_or_json 2 "gateway" "Gateway URL must not be empty."
    fi
    if [[ "$raw" =~ [[:space:]] ]] || [[ "$raw" =~ [[:cntrl:]] ]]; then
        emit_error_or_json 2 "gateway" "Gateway URL must not contain whitespace or control characters."
    fi
    case "$trimmed" in
        *\?*|*\#*|*@*|*\\*)
            emit_error_or_json 2 "gateway" "Gateway URL must not embed credentials, queries, or fragments."
            ;;
    esac
    while [[ "$trimmed" == */ ]] && [[ "$trimmed" != *"://" ]]; do
        trimmed="${trimmed%/}"
    done
    local low=""
    low="$(to_lower "$trimmed")"
    local scheme=""
    local rest=""
    case "$low" in
        https://*)
            scheme="https"
            rest="${low#https://}"
            ;;
        http://*)
            scheme="http"
            rest="${low#http://}"
            ;;
        *)
            emit_error_or_json 2 "gateway" "Gateway URL must use https:// (or http:// on loopback)."
            ;;
    esac
    if [ -z "$rest" ] || [[ "$rest" == */* ]]; then
        emit_error_or_json 2 "gateway" "Gateway URL must be a bare origin without a path suffix."
    fi
    local host=""
    local port=""
    if [[ "$rest" == \[* ]]; then
        if [[ "$rest" =~ ^\[::1\](:([0-9]+))?$ ]]; then
            host="::1"
            port="${BASH_REMATCH[2]:-}"
        else
            emit_error_or_json 2 "gateway" "Untrusted or malformed IPv6 gateway host; expected loopback [::1]."
        fi
    else
        case "$rest" in
            *:*)
                host="${rest%%:*}"
                port="${rest#*:}"
                if [[ "$port" == *:* ]]; then
                    emit_error_or_json 2 "gateway" "Gateway URL contains an invalid port."
                fi
                ;;
            *)
                host="$rest"
                port=""
                ;;
        esac
    fi
    if [ -n "$port" ]; then
        if ! [[ "$port" =~ ^[0-9]+$ ]] || [ "$((10#$port))" -lt 1 ] || [ "$((10#$port))" -gt 65535 ]; then
            emit_error_or_json 2 "gateway" "Gateway URL port out of range."
        fi
    fi
    if [ "$host" = "127.0.0.1" ] || [ "$host" = "localhost" ] || [ "$host" = "::1" ]; then
        CLEAN_GATEWAY="$trimmed"
        return 0
    fi
    if [[ "$host" =~ ^([a-z0-9]([a-z0-9-]*[a-z0-9])?\.)+run\.app$ ]]; then
        if [ "$scheme" != "https" ]; then
            emit_error_or_json 2 "gateway" "Remote Cloud Run gateway URL must use https."
        fi
        CLEAN_GATEWAY="$trimmed"
        return 0
    fi
    emit_error_or_json 2 "gateway" "Untrusted gateway host; expected https://*.run.app or loopback."
}

CLEAN_GATEWAY=""
validate_gateway_url "$GATEWAY_URL"
GATEWAY_URL="$CLEAN_GATEWAY"

# Resolve Python interpreter
PYTHON_BIN=""
if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif [ -x "$ROOT_DIR/../../HiveMind/.venv/bin/python" ]; then
    PYTHON_BIN="$ROOT_DIR/../../HiveMind/.venv/bin/python"
elif [ -x "$ROOT_DIR/../../../.venv/bin/python" ]; then
    PYTHON_BIN="$ROOT_DIR/../../../.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
fi

if [ "$JSON_OUT" -eq 0 ]; then
    echo "============================================================================"
    echo "  PROJECT HIVEMIND - TURNKEY PERSONAL ONBOARDING (POSIX / macOS / Linux)"
    echo "  Gateway: ${GATEWAY_URL}"
    echo "============================================================================"
    echo "[RULE] Personal home PC & personal hobby/open-source projects ONLY (never work laptops)."
    echo "[LEGAL NOTICE] 0.00 EUR / 0.00 USD: No money is paid; points confer reciprocal compute (AGPL-3.0)."
    echo ""
fi

is_allowlisted_bootstrap_script() {
    local cand="$1"
    if ! [[ "$cand" =~ ^[A-Za-z0-9_]+\.(py|ps1)$ ]]; then
        return 1
    fi
    local item=""
    for item in "${ALLOWED_BOOTSTRAP_SCRIPTS[@]}"; do
        if [ "$item" = "$cand" ]; then
            return 0
        fi
    done
    return 1
}

bootstrap_fetch_script() {
    local script_name="$1"
    local dest_path="$2"
    if ! is_allowlisted_bootstrap_script "$script_name"; then
        emit_error_or_json 2 "bootstrap" "Refused non-allowlisted bootstrap script: ${script_name}."
    fi
    if [ -z "$PYTHON_BIN" ]; then
        emit_error_or_json 2 "python" "Python 3.10+ was not found on PATH; cannot bootstrap ${script_name}."
    fi
    local url="${GATEWAY_URL}/scripts/${script_name}"
    if ! "$PYTHON_BIN" - "$url" "$dest_path" <<'PYEOF'
import os
import sys
import urllib.error
import urllib.request

url = sys.argv[1]
dest = sys.argv[2]
tmp_path = f"{dest}.tmp.{os.getpid()}"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
req = urllib.request.Request(url, headers={"User-Agent": "hivemind-posix-bootstrap/1.0"})
try:
    with opener.open(req, timeout=15.0) as resp:
        if getattr(resp, "status", 200) != 200:
            raise SystemExit(2)
        payload = resp.read(2 * 1024 * 1024)
except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
    print(f"Bootstrap download failed for {url}: {exc}", file=sys.stderr)
    raise SystemExit(2)
if not payload:
    print(f"Bootstrap download returned empty body for {url}", file=sys.stderr)
    raise SystemExit(2)
os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
with open(tmp_path, "wb") as handle:
    handle.write(payload)
os.chmod(tmp_path, 0o700)
os.replace(tmp_path, dest)
PYEOF
    then
        emit_error_or_json 2 "bootstrap" "Bootstrap download failed for ${script_name} from ${GATEWAY_URL}."
    fi
    if [ ! -s "$dest_path" ]; then
        emit_error_or_json 2 "bootstrap" "Bootstrap download produced an empty or missing file at ${dest_path}."
    fi
}

ensure_target_script_available() {
    local script_name="$1"
    local target_path="$2"
    if [ -f "$target_path" ]; then
        BOOTSTRAP_PLANNED=0
        return 0
    fi
    if [ "$BOOTSTRAP" -eq 1 ]; then
        if ! is_allowlisted_bootstrap_script "$script_name"; then
            emit_error_or_json 2 "bootstrap" "Refused non-allowlisted bootstrap script: ${script_name}."
        fi
        if [ "$DRY_RUN" -eq 1 ]; then
            BOOTSTRAP_PLANNED=1
            return 0
        fi
        bootstrap_fetch_script "$script_name" "$target_path"
        BOOTSTRAP_PLANNED=0
        return 0
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        BOOTSTRAP_PLANNED=0
        return 0
    fi
    emit_error_or_json 2 "missing_script" "Missing ${target_path} alongside join_hivemind_campaign.sh. Pass --bootstrap to fetch from ${GATEWAY_URL}/scripts/${script_name}."
}

# 3. Route MODE to target script & build sanitized CLI args
TARGET_SCRIPT_NAME=""
PY_ARGS=()
BOOTSTRAP_PLANNED=0

case "$MODE" in
    doctor)
        TARGET_SCRIPT_NAME="hivemind_doctor.py"
        DOC_HOT_C="$( [ "$HOT_C_SET" -eq 1 ] && printf '%s' "$HOT_C" || printf '%s' "${HIVEMIND_HOT_C:-70.0}" )"
        PY_ARGS+=("--noel" "--gateway" "$GATEWAY_URL" "--lang" "$LANG_CODE" "--max-temp-c" "$DOC_HOT_C")
        for g in "${GAMES[@]+"${GAMES[@]}"}"; do
            [ -n "$g" ] && PY_ARGS+=("--game" "$g")
        done
        if [ "$POST_SETUP_TELEMETRY" -eq 1 ]; then
            PY_ARGS+=("--submit-ces")
            [ "$EASE_RATING" -ge 1 ] && PY_ARGS+=("--ease-rating" "$EASE_RATING")
            [ "$HAD_GAME_LAG" -eq 1 ] && PY_ARGS+=("--had-game-lag")
        fi
        [ "$JSON_OUT" -eq 1 ] && PY_ARGS+=("--json")
        PY_ARGS+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")
        ;;
    deploy-smoke)
        if [ "$BOOTSTRAP" -eq 1 ]; then
            TARGET_SCRIPT_NAME="noel_cloud_run_parity_gate.py"
            PY_ARGS+=("--post-deploy-gate" "--gateway" "$GATEWAY_URL" "--lang" "$LANG_CODE" "--enforce-personal-pc")
        else
            TARGET_SCRIPT_NAME="noel_posix_launcher_and_smoke_verifier.py"
            PY_ARGS+=("--smoke-only" "--gateway" "$GATEWAY_URL" "--lang" "$LANG_CODE")
        fi
        [ -n "$EXPECTED_SHA" ] && PY_ARGS+=("--expected-sha" "$EXPECTED_SHA")
        [ "$POST_SETUP_TELEMETRY" -eq 1 ] && PY_ARGS+=("--probe-feedback-post")
        [ "$JSON_OUT" -eq 1 ] && PY_ARGS+=("--json")
        PY_ARGS+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")
        ;;
    vscode-setup)
        if [ "$BOOTSTRAP" -eq 1 ]; then
            TARGET_SCRIPT_NAME="noel_vscode_autoconfig.py"
            PY_ARGS+=("--quick-connect" "--gateway" "$GATEWAY_URL" "--lang" "$LANG_CODE")
            [ -n "$HANDLE" ] && PY_ARGS+=("--handle" "$HANDLE")
            [ -n "$API_KEY" ] && PY_ARGS+=("--api-key" "$API_KEY")
            [ "$POST_SETUP_TELEMETRY" -eq 1 ] && PY_ARGS+=("--post-setup-telemetry")
            [ "$EASE_RATING" -ge 1 ] && PY_ARGS+=("--ease-rating" "$EASE_RATING")
            [ "$HAD_GAME_LAG" -eq 1 ] && PY_ARGS+=("--had-game-lag")
        else
            TARGET_SCRIPT_NAME="noel_vscode_quick_preset.py"
            PY_ARGS+=("--vscode-setup" "--gateway" "$GATEWAY_URL" "--lang" "$LANG_CODE")
            [ -n "$HANDLE" ] && PY_ARGS+=("--handle" "$HANDLE")
            [ -n "$EXPECTED_SHA" ] && PY_ARGS+=("--expected-sha" "$EXPECTED_SHA")
            [ "$WRITE_CONTINUE_CONFIG" -eq 1 ] && PY_ARGS+=("--write-continue-config")
            [ "$POST_SETUP_TELEMETRY" -eq 1 ] && PY_ARGS+=("--submit-ces")
            [ "$EASE_RATING" -ge 1 ] && PY_ARGS+=("--ease-rating" "$EASE_RATING")
            [ "$HAD_GAME_LAG" -eq 1 ] && PY_ARGS+=("--had-game-lag")
        fi
        [ "$JSON_OUT" -eq 1 ] && PY_ARGS+=("--json")
        PY_ARGS+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")
        ;;
    gamer-service)
        SVC_RESERVE="$( [ "$RESERVE_MIB_SET" -eq 1 ] && printf '%s' "$RESERVE_MIB" || printf '%s' "${HIVEMIND_RESERVE_MIB:-2048}" )"
        SVC_HOT="$( [ "$HOT_C_SET" -eq 1 ] && printf '%s' "$HOT_C" || printf '%s' "${HIVEMIND_HOT_C:-70.0}" )"
        if [ "$BOOTSTRAP" -eq 1 ]; then
            TARGET_SCRIPT_NAME="noel_gamer_whisper_selftest.py"
            PY_ARGS+=("--noel-verify" "--gateway" "$GATEWAY_URL" "--lang" "$LANG_CODE" "--min-free-vram-mib" "$SVC_RESERVE" "--max-temp-c" "$SVC_HOT")
            [ -n "$API_KEY" ] && PY_ARGS+=("--api-key" "$API_KEY")
            [ -n "$EXPECTED_SHA" ] && PY_ARGS+=("--expected-sha" "$EXPECTED_SHA")
            for g in "${GAMES[@]+"${GAMES[@]}"}"; do
                [ -n "$g" ] && PY_ARGS+=("--game" "$g")
            done
            if [ "$POST_SETUP_TELEMETRY" -eq 1 ]; then
                PY_ARGS+=("--submit-feedback")
                [ "$EASE_RATING" -ge 1 ] && PY_ARGS+=("--ease-rating" "$EASE_RATING")
                [ "$HAD_GAME_LAG" -eq 1 ] && PY_ARGS+=("--had-game-lag")
            fi
        else
            TARGET_SCRIPT_NAME="noel_idle_resume_service.py"
            PY_ARGS+=("--gateway" "$GATEWAY_URL" "--lang" "$LANG_CODE" "--min-free-vram-mib" "$SVC_RESERVE" "--max-temp-c" "$SVC_HOT" "--tick" "--yield-drill" "--check-fleet")
            [ -n "$HANDLE" ] && PY_ARGS+=("--handle" "$HANDLE")
            [ -n "$EXPECTED_SHA" ] && PY_ARGS+=("--expected-sha" "$EXPECTED_SHA")
            if [ "$POST_SETUP_TELEMETRY" -eq 1 ]; then
                PY_ARGS+=("--submit-ces")
                [ "$EASE_RATING" -ge 1 ] && PY_ARGS+=("--ease-rating" "$EASE_RATING")
                [ "$HAD_GAME_LAG" -eq 1 ] && PY_ARGS+=("--had-game-lag")
            fi
        fi
        [ "$JSON_OUT" -eq 1 ] && PY_ARGS+=("--json")
        PY_ARGS+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")
        ;;
    uninstall)
        TARGET_SCRIPT_NAME="hivemind_quickstart.py"
        PY_ARGS+=("--uninstall")
        [ "$JSON_OUT" -eq 1 ] && PY_ARGS+=("--json")
        PY_ARGS+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")
        ;;
    connect|gamer-whisper|gamer-check)
        TARGET_SCRIPT_NAME="hivemind_quickstart.py"
        PY_ARGS+=("--gateway" "$GATEWAY_URL" "--lang" "$LANG_CODE")
        [ -n "$HANDLE" ] && PY_ARGS+=("--handle" "$HANDLE")
        [ -n "$API_KEY" ] && PY_ARGS+=("--api-key" "$API_KEY")
        if [ "$MODE" = "connect" ] || [ "$WRITE_CONTINUE_CONFIG" -eq 1 ]; then
            PY_ARGS+=("--write-continue-config")
        fi
        if [ "$MODE" = "gamer-whisper" ]; then
            WHISPER_RESERVE="$( [ "$RESERVE_MIB_SET" -eq 1 ] && printf '%s' "$RESERVE_MIB" || printf '%s' "${HIVEMIND_RESERVE_MIB:-2048}" )"
            WHISPER_HOT="$( [ "$HOT_C_SET" -eq 1 ] && printf '%s' "$HOT_C" || printf '%s' "${HIVEMIND_HOT_C:-70}" )"
            PY_ARGS+=("--gamer-whisper" "--reserve-mib" "$WHISPER_RESERVE" "--hot-c" "$WHISPER_HOT")
            [ -n "$ONCE_FLAG" ] && PY_ARGS+=("$ONCE_FLAG")
        elif [ "$MODE" = "gamer-check" ]; then
            WHISPER_RESERVE="$( [ "$RESERVE_MIB_SET" -eq 1 ] && printf '%s' "$RESERVE_MIB" || printf '%s' "${HIVEMIND_RESERVE_MIB:-2048}" )"
            WHISPER_HOT="$( [ "$HOT_C_SET" -eq 1 ] && printf '%s' "$HOT_C" || printf '%s' "${HIVEMIND_HOT_C:-70}" )"
            PY_ARGS+=("--gamer-check" "--reserve-mib" "$WHISPER_RESERVE" "--hot-c" "$WHISPER_HOT")
        fi
        for g in "${GAMES[@]+"${GAMES[@]}"}"; do
            [ -n "$g" ] && PY_ARGS+=("--game" "$g")
        done
        [ -n "$EXPECTED_SHA" ] && PY_ARGS+=("--expected-sha" "$EXPECTED_SHA")
        [ "$SMOKE_ROUTES" -eq 1 ] && PY_ARGS+=("--smoke-routes")
        [ "$CHECK_DEPLOY_PARITY" -eq 1 ] && PY_ARGS+=("--check-deploy-parity")
        [ "$POST_SETUP_TELEMETRY" -eq 1 ] && PY_ARGS+=("--post-setup-telemetry")
        [ "$EASE_RATING" -ge 1 ] && PY_ARGS+=("--ease-rating" "$EASE_RATING")
        [ "$HAD_GAME_LAG" -eq 1 ] && PY_ARGS+=("--had-game-lag")
        [ "$FLUSH_FEEDBACK_QUEUE" -eq 1 ] && PY_ARGS+=("--flush-feedback-queue")
        [ "$JSON_OUT" -eq 1 ] && PY_ARGS+=("--json")
        PY_ARGS+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")
        ;;
    legacy-demo)
        TARGET_SCRIPT_NAME="run_multi_moe_hf_campaign.py"
        DEMO_HANDLE="${HANDLE:-${USER:-citizen-scientist}}"
        PY_ARGS+=("--contributor-handle" "$DEMO_HANDLE" "--tier" "standard")
        PY_ARGS+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")
        ;;
    crowdgpt)
        TARGET_SCRIPT_NAME="hivemind_opentrainer_agent.py"
        PY_ARGS+=("--gateway" "$GATEWAY_URL" "--crowdgpt" "--reserve-mib" "$RESERVE_MIB" "--hot-c" "$HOT_C" "--cool-c" "$COOL_C")
        [ -n "$ONCE_FLAG" ] && PY_ARGS+=("$ONCE_FLAG")
        PY_ARGS+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")
        ;;
    train)
        TARGET_SCRIPT_NAME="hivemind_opentrainer_agent.py"
        PY_ARGS+=("--gateway" "$GATEWAY_URL" "--max-vram-gb" "5.5" "--train")
        [ -n "$ONCE_FLAG" ] && PY_ARGS+=("$ONCE_FLAG")
        PY_ARGS+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")
        ;;
    opentrainer|*)
        TARGET_SCRIPT_NAME="hivemind_opentrainer_agent.py"
        PY_ARGS+=("--gateway" "$GATEWAY_URL" "--check-firewall")
        [ -n "$ONCE_FLAG" ] && PY_ARGS+=("$ONCE_FLAG")
        PY_ARGS+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")
        ;;
esac

TARGET_SCRIPT="$SCRIPT_DIR/$TARGET_SCRIPT_NAME"
ensure_target_script_available "$TARGET_SCRIPT_NAME" "$TARGET_SCRIPT"

# Build masked/redacted display args for --dry-run
DISPLAY_ARGS=("$(redact_home_prefix "$TARGET_SCRIPT")")
MASK_NEXT=0
for arg_item in "${PY_ARGS[@]+"${PY_ARGS[@]}"}"; do
    if [ "$MASK_NEXT" -eq 1 ]; then
        DISPLAY_ARGS+=("$(format_masked_api_key "$arg_item")")
        MASK_NEXT=0
    else
        case "$arg_item" in
            --api-key=*)
                DISPLAY_ARGS+=("--api-key=$(format_masked_api_key "${arg_item#*=}")")
                ;;
            --api-key)
                DISPLAY_ARGS+=("$arg_item")
                MASK_NEXT=1
                ;;
            *)
                DISPLAY_ARGS+=("$(redact_home_prefix "$arg_item")")
                ;;
        esac
    fi
done

if [ "$DRY_RUN" -eq 1 ]; then
    PYTHON_FOUND=0
    [ -n "$PYTHON_BIN" ] && PYTHON_FOUND=1
    if [ "$JSON_OUT" -eq 1 ]; then
        printf '{\n'
        printf '  "dry_run": true,\n'
        printf '  "compliance_ok": true,\n'
        printf '  "mode": %s,\n' "$(json_escape "$MODE")"
        printf '  "lang": %s,\n' "$(json_escape "$LANG_CODE")"
        printf '  "gateway": %s,\n' "$(json_escape "$GATEWAY_URL")"
        printf '  "target_script": %s,\n' "$(json_escape "$TARGET_SCRIPT_NAME")"
        printf '  "bootstrap": %s,\n' "$(json_bool "$BOOTSTRAP")"
        printf '  "bootstrap_planned": %s,\n' "$(json_bool "$BOOTSTRAP_PLANNED")"
        printf '  "python_found": %s,\n' "$(json_bool "$PYTHON_FOUND")"
        if [ -n "$API_KEY" ]; then
            printf '  "masked_api_key": %s,\n' "$(json_escape "$(format_masked_api_key "$API_KEY")")"
        fi
        printf '  "argv": ['
        first=1
        for d_item in "${DISPLAY_ARGS[@]+"${DISPLAY_ARGS[@]}"}"; do
            [ "$first" -eq 1 ] && first=0 || printf ', '
            json_escape "$d_item"
        done
        printf '],\n'
        printf '  "personal_pc_only": true,\n'
        printf '  "financial_promise": "0.00 EUR / 0.00 USD",\n'
        printf '  "license": "AGPL-3.0"\n'
        printf '}\n'
    else
        printf '[DryRun] Mode=%s Target=%s Gateway=%s (0.00 EUR / 0.00 USD)\n' "$MODE" "$TARGET_SCRIPT_NAME" "$GATEWAY_URL"
        printf '[DryRun] Planned argv:'
        for d_item in "${DISPLAY_ARGS[@]+"${DISPLAY_ARGS[@]}"}"; do
            printf ' %s' "$d_item"
        done
        printf '\n'
    fi
    exit 0
fi

if [ -z "$PYTHON_BIN" ]; then
    emit_error_or_json 2 "python" "Python 3.10+ was not found on PATH. Install Python 3.10+ before running."
fi

if [ -d "$ROOT_DIR/src/hivemind" ]; then
    export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
fi

if [ -n "$API_KEY" ]; then
    export HIVEMIND_API_KEY="$API_KEY"
fi

case "$MODE" in
    doctor|deploy-smoke|vscode-setup|gamer-service|uninstall|connect|gamer-whisper|gamer-check)
        exec "$PYTHON_BIN" "$TARGET_SCRIPT" "${PY_ARGS[@]+"${PY_ARGS[@]}"}"
        ;;
    legacy-demo)
        mkdir -p "$ROOT_DIR/artifacts"
        cd "$ROOT_DIR"
        exec "$PYTHON_BIN" "$TARGET_SCRIPT" "${PY_ARGS[@]+"${PY_ARGS[@]}"}"
        ;;
esac

# Auto-load saved key from ~/.hivemind/credentials.json AFTER compliance check passed
if [ -z "${HIVEMIND_API_KEY:-}" ] && [ -f "$HOME/.hivemind/credentials.json" ]; then
    SAVED_KEY="$("$PYTHON_BIN" -c 'import json, os; p=os.path.expanduser("~/.hivemind/credentials.json"); d=json.load(open(p)); print(d.get("api_key",""))' 2>/dev/null || true)"
    if [[ "$SAVED_KEY" == hm-* ]] || [[ "$SAVED_KEY" == hm_live_* ]]; then
        export HIVEMIND_API_KEY="$SAVED_KEY"
    fi
fi

PREFLIGHT_SCRIPT="$SCRIPT_DIR/hivemind_volunteer_preflight.py"
ensure_target_script_available "hivemind_volunteer_preflight.py" "$PREFLIGHT_SCRIPT"

echo "[Step 1/2] Running volunteer hardware & VRAM preflight..."
set +e
"$PYTHON_BIN" "$PREFLIGHT_SCRIPT" --project-url "$GATEWAY_URL"
PREFLIGHT_RC=$?
set -e
if [ "$PREFLIGHT_RC" -eq 2 ]; then
    echo "Preflight reported not_ready (exit 2). Fix the FAIL items above before joining." >&2
    exit 2
fi

echo ""
echo "[Step 2/2] Running OpenTrainer volunteer check / contributor (${MODE})..."
if [ "$MODE" = "train" ] && [ -z "${HIVEMIND_API_KEY:-}" ]; then
    QS_SCRIPT="$SCRIPT_DIR/hivemind_quickstart.py"
    ensure_target_script_available "hivemind_quickstart.py" "$QS_SCRIPT"
    "$PYTHON_BIN" "$QS_SCRIPT" --gateway "$GATEWAY_URL" --json >/dev/null
    SAVED_KEY="$("$PYTHON_BIN" -c 'import json, os; p=os.path.expanduser("~/.hivemind/credentials.json"); d=json.load(open(p)); print(d.get("api_key",""))' 2>/dev/null || true)"
    export HIVEMIND_API_KEY="$SAVED_KEY"
fi

if [ "$MODE" = "opentrainer" ] && [ -z "${HIVEMIND_API_KEY:-}" ]; then
    echo "No HIVEMIND_API_KEY set; running outbound firewall traversal check only."
    echo "Run ./scripts/join_hivemind_campaign.sh --connect to register your personal key in < 60 s."
    exec "$PYTHON_BIN" "$TARGET_SCRIPT" --gateway "$GATEWAY_URL" --check-firewall --once "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
fi

exec "$PYTHON_BIN" "$TARGET_SCRIPT" "${PY_ARGS[@]+"${PY_ARGS[@]}"}"
