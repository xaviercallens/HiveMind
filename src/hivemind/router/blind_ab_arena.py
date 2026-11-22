# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-VRAM Blind A/B Evaluation Arena (`/arena-noel`) for Holiday Checkpoints.

Serves curated deterministic side-by-side tasks across four scientific & systems domains
(Python, Rust, Lean 4, and exoplanet transit analysis) comparing the pre-holiday base
checkpoint against the holiday swarm QLoRA checkpoint with per-request randomized blind
`A` / `B` labels so live T4 / volunteer GPU VRAM (`0 MiB`) is never touched.

Exposes:
* Strict per-request nonce-pinned Content-Security-Policy (`blind_ab_arena_csp`).
* Bilingual French (`/arena-noel` default) and English (`?lang=en` / `Accept-Language`) HTML UI
  with `textContent`-only DOM updates (`render_blind_ab_arena_page`).
* Stateless HMAC-masked blind ballot signing & verification (`sign_blind_ballot`,
  `verify_blind_ballot`) so browser DOM inspection cannot reveal checkpoint identity prior to voting.
* Thread-safe vote recording & statistical aggregation (`BlindArenaCollector`,
  `wilson_95_lower_bound`) computing exact vote counts, `win_ratio`, and the 95% Wilson
  confidence interval lower bound without ever emitting a JSON `'score'` key.
* Strict Amadeus / Personal-PC compliance rule (100% personal home PCs & personal open-source/hobby
  projects only, `0.00 EUR / 0.00 USD`).
"""

from __future__ import annotations

import hashlib
import hmac
import html
import math
import random
import re
import secrets
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

from hivemind.router.telemetry_figures import NOT_AVAILABLE

DEFAULT_GATEWAY_URL = "https://hivemind-openrouter-1003063861791.us-central1.run.app"
SUPPORTED_ARENA_LANGS: tuple[str, ...] = ("fr", "en")
VALID_ARENA_PROFILES: frozenset[str] = frozenset({"dev", "gamer"})
VALID_CHOSEN_SIDES: frozenset[str] = frozenset({"A", "B", "tie"})
VALID_HOLIDAY_SIDES: frozenset[str] = frozenset({"A", "B"})
MAX_ARENA_VOTES = 2000
MAX_ARENA_VOTES_CEILING = 50_000
MAX_SECRET_BYTES = 4096
MAX_QUERY_LANG_CHARS = 64
MAX_ACCEPT_LANGUAGE_CHARS = 512
MAX_ACCEPT_LANGUAGE_PARTS = 32
MAX_TASK_ID_CHARS = 64
MAX_CHOSEN_SIDE_CHARS = 16
MAX_PROFILE_CHARS = 32
MAX_BALLOT_TOKEN_CHARS = 128
MAX_GATEWAY_URL_CHARS = 256
MAX_ROUTE_PATH_CHARS = 128
MAX_RENDER_CARDS = 16
MAX_CARD_TEXT_CHARS = 16_384
Z_WILSON_95 = 1.959963984540054

_NONCE_RE = re.compile(r"\A[A-Za-z0-9_+=-]{8,80}\Z", re.ASCII)
_SALT_RE = re.compile(r"\A[0-9a-f]{8,32}\Z", re.ASCII)
_MAC_RE = re.compile(r"\A[0-9a-f]{24}\Z", re.ASCII)
_TASK_ID_RE = re.compile(r"\A[a-z0-9][a-z0-9-]{0,63}\Z", re.ASCII)
_BALLOT_SAFE_RE = re.compile(r"\A[A-Za-z0-9._-]{8,128}\Z", re.ASCII)
_SAFE_PATH_RE = re.compile(r"\A/[A-Za-z0-9_./-]{0,127}\Z", re.ASCII)
_Q_PARAM_RE = re.compile(r"\A\s*q\s*=\s*([01](?:\.[0-9]{0,3})?)\s*\Z", re.IGNORECASE | re.ASCII)
_FORBIDDEN_URL_CHARS: frozenset[str] = frozenset(
    {" ", "\t", "\r", "\n", '"', "'", "<", ">", "`", ";", "|", "&", "$", "\\"}
)

PERSONAL_PC_RULE_FR = (
    "Règle stricte d'intégrité personnelle : usage 100 % sur PC personnel à domicile et projets personnels / open-source "
    "uniquement (jamais sur un poste professionnel, un réseau d'entreprise, ni avec du code ou des données d'employeur). "
    "Projet citoyen AGPL-3.0 • 0,00 EUR / 0,00 USD • 0 Mio de VRAM GPU sollicitée."
)
PERSONAL_PC_RULE_EN = (
    "Strict Personal-PC Compliance Rule: 100% personal home PCs and personal hobby/open-source projects ONLY "
    "(never on corporate laptops, employer networks, or with company code/data). "
    "Citizen-science AGPL-3.0 • 0.00 EUR / 0.00 USD • 0 MiB live GPU VRAM touched."
)

CURATED_ARENA_TASKS: tuple[dict[str, str], ...] = (
    {
        "task_id": "python-sliding-median",
        "domain": "python",
        "badge_fr": "Python 3.11 • Stdlib",
        "badge_en": "Python 3.11 • Stdlib",
        "default_holiday_side": "A",
        "title_fr": "Médiane glissante déterministe avec garde NaN / bool",
        "title_en": "Deterministic sliding-window median with NaN / bool guard",
        "prompt_fr": (
            "Écrire `rolling_median(values, window)` en bibliothèque standard Python pure : "
            "rejeter les booléens (`isinstance(True, int)`), les flottants `NaN`/`inf` et `window < 1`, "
            "puis renvoyer la médiane exacte de chaque fenêtre complète."
        ),
        "prompt_en": (
            "Write `rolling_median(values, window)` using Python stdlib only: reject booleans "
            "(`isinstance(True, int)`), non-finite `NaN`/`inf` floats, and `window < 1`, then "
            "return the exact median of every full sliding window."
        ),
        "holiday_checkpoint_output": (
            "import math\n"
            "from collections.abc import Sequence\n\n"
            "def rolling_median(values: Sequence[float], window: int) -> list[float]:\n"
            "    if isinstance(window, bool) or not isinstance(window, int) or window < 1:\n"
            "        raise ValueError('window must be an integer >= 1')\n"
            "    clean: list[float] = []\n"
            "    for item in values:\n"
            "        if isinstance(item, bool) or not isinstance(item, (int, float)):\n"
            "            raise ValueError('all elements must be real numbers, not bool')\n"
            "        fval = float(item)\n"
            "        if not math.isfinite(fval):\n"
            "            raise ValueError('all elements must be finite')\n"
            "        clean.append(fval)\n"
            "    if len(clean) < window:\n"
            "        return []\n"
            "    half = window // 2\n"
            "    out: list[float] = []\n"
            "    for idx in range(window, len(clean) + 1):\n"
            "        ordered = sorted(clean[idx - window : idx])\n"
            "        if window % 2 == 1:\n"
            "            out.append(ordered[half])\n"
            "        else:\n"
            "            out.append(0.5 * (ordered[half - 1] + ordered[half]))\n"
            "    return out"
        ),
        "base_checkpoint_output": (
            "import statistics\n\n"
            "def rolling_median(values, window):\n"
            "    # Direct slice loop over input sequence\n"
            "    if len(values) < window:\n"
            "        return []\n"
            "    medians = []\n"
            "    for idx in range(len(values) - window + 1):\n"
            "        chunk = [float(x) for x in values[idx : idx + window]]\n"
            "        medians.append(statistics.median(chunk))\n"
            "    return medians"
        ),
    },
    {
        "task_id": "rust-frame-decoder",
        "domain": "rust",
        "badge_fr": "Rust 2021 • Zéro Allocation",
        "badge_en": "Rust 2021 • Zero Allocation",
        "default_holiday_side": "B",
        "title_fr": "Décodeur d'en-tête binaire sans panique ni allocation",
        "title_en": "Panic-free zero-allocation binary frame header decoder",
        "prompt_fr": (
            "Implémenter `decode_frame(buf: &[u8]) -> Result<(u8, &[u8]), FrameError>` lisant "
            "1 octet de version et une taille big-endian `u16`, sans indexation susceptible de paniquer "
            "sur paquet tronqué et avec arithmétique vérifiée."
        ),
        "prompt_en": (
            "Implement `decode_frame(buf: &[u8]) -> Result<(u8, &[u8]), FrameError>` reading a "
            "1-byte version and big-endian `u16` payload length with zero heap allocations, "
            "checked bounds, and zero panic paths on truncated packets."
        ),
        "holiday_checkpoint_output": (
            "#[derive(Debug, Clone, Copy, PartialEq, Eq)]\n"
            "pub enum FrameError {\n"
            "    TruncatedHeader,\n"
            "    TruncatedPayload { expected: usize, actual: usize },\n"
            "    UnsupportedVersion(u8),\n"
            "}\n\n"
            "pub fn decode_frame(buf: &[u8]) -> Result<(u8, &[u8]), FrameError> {\n"
            "    let header = buf.get(..3).ok_or(FrameError::TruncatedHeader)?;\n"
            "    let version = header[0];\n"
            "    if version == 0 || version > 2 {\n"
            "        return Err(FrameError::UnsupportedVersion(version));\n"
            "    }\n"
            "    let payload_len = usize::from(u16::from_be_bytes([header[1], header[2]]));\n"
            "    let end = 3usize.checked_add(payload_len).ok_or(FrameError::TruncatedHeader)?;\n"
            "    let payload = buf.get(3..end).ok_or(FrameError::TruncatedPayload {\n"
            "        expected: payload_len,\n"
            "        actual: buf.len().saturating_sub(3),\n"
            "    })?;\n"
            "    Ok((version, payload))\n"
            "}"
        ),
        "base_checkpoint_output": (
            "#[derive(Debug, PartialEq, Eq)]\n"
            "pub enum FrameError {\n"
            "    TooShort,\n"
            "}\n\n"
            "pub fn decode_frame(buf: &[u8]) -> Result<(u8, &[u8]), FrameError> {\n"
            "    let version = buf[0];\n"
            "    let len = u16::from_be_bytes([buf[1], buf[2]]) as usize;\n"
            "    if buf.len() < 3 + len {\n"
            "        return Err(FrameError::TooShort);\n"
            "    }\n"
            "    Ok((version, &buf[3..3 + len]))\n"
            "}"
        ),
    },
    {
        "task_id": "lean4-energy-descent",
        "domain": "lean4",
        "badge_fr": "Lean 4 • Preuve Formelle",
        "badge_en": "Lean 4 • Formal Proof",
        "default_holiday_side": "A",
        "title_fr": "Monotonie stricte de descente d'énergie discrète",
        "title_en": "Strict discrete Lyapunov energy descent monotonicity",
        "prompt_fr": (
            "Prouver en Lean 4 sans axiome externe : si `e2 + delta <= e1`, `e1 <= e0` et `0 < delta` "
            "sur `Nat`, alors la décroissance stricte `e2 < e0` est garantie."
        ),
        "prompt_en": (
            "Prove in Lean 4 with zero extra axioms: given `e2 + delta <= e1`, `e1 <= e0`, and "
            "`0 < delta` over `Nat`, strict energy descent `e2 < e0` holds."
        ),
        "holiday_checkpoint_output": (
            "theorem strict_energy_descent\n"
            "    (e0 e1 e2 delta : Nat)\n"
            "    (h_step : e2 + delta <= e1)\n"
            "    (h_prev : e1 <= e0)\n"
            "    (h_pos  : 0 < delta) :\n"
            "    e2 < e0 := by\n"
            "  have h_lt_sum : e2 < e2 + delta := Nat.lt_add_of_pos_right h_pos\n"
            "  have h_lt_e1  : e2 < e1 := Nat.lt_of_lt_of_le h_lt_sum h_step\n"
            "  exact Nat.lt_of_lt_of_le h_lt_e1 h_prev"
        ),
        "base_checkpoint_output": (
            "theorem weak_energy_bound\n"
            "    (e0 e1 e2 : Nat)\n"
            "    (h_step : e2 <= e1)\n"
            "    (h_prev : e1 <= e0) :\n"
            "    e2 <= e0 := by\n"
            "  -- Proves only non-strict inequality; omits positive delta margin\n"
            "  exact Nat.le_trans h_step h_prev"
        ),
    },
    {
        "task_id": "exoplanet-transit-bls",
        "domain": "exoplanet",
        "badge_fr": "Astrophysique • Transit Exoplanète",
        "badge_en": "Astrophysics • Exoplanet Transit",
        "default_holiday_side": "B",
        "title_fr": "Profondeur relative ΔF/F et rapport signal/bruit d'un transit",
        "title_en": "Relative transit depth ΔF/F and folded SNR on light curves",
        "prompt_fr": (
            "Écrire `analyze_transit_dip(flux, in_transit)` estimant le flux hors-transit par médiane "
            "robuste aux éruptions stellaires, la profondeur relative en ppm et le SNR replié "
            "`depth / (sigma_oot / sqrt(n_in))` en rejetant toute variance nulle ou flux <= 0."
        ),
        "prompt_en": (
            "Write `analyze_transit_dip(flux, in_transit)` computing a flare-robust median "
            "out-of-transit baseline, relative transit depth in ppm, and folded SNR "
            "`depth / (sigma_oot / sqrt(n_in))`, rejecting non-positive flux or zero baseline scatter."
        ),
        "holiday_checkpoint_output": (
            "import math\n"
            "from collections.abc import Sequence\n\n"
            "def analyze_transit_dip(\n"
            "    flux: Sequence[float],\n"
            "    in_transit: Sequence[bool],\n"
            ") -> dict[str, float]:\n"
            "    if len(flux) != len(in_transit) or len(flux) < 4:\n"
            "        raise ValueError('flux and in_transit must have matching length >= 4')\n"
            "    oot: list[float] = []\n"
            "    dip: list[float] = []\n"
            "    for f_raw, flag in zip(flux, in_transit):\n"
            "        if isinstance(f_raw, bool) or not isinstance(flag, bool):\n"
            "            raise ValueError('invalid sample or transit mask type')\n"
            "        f_val = float(f_raw)\n"
            "        if not math.isfinite(f_val) or f_val <= 0.0:\n"
            "            raise ValueError('flux samples must be finite and > 0')\n"
            "        (dip if flag else oot).append(f_val)\n"
            "    if len(oot) < 2 or len(dip) < 1:\n"
            "        raise ValueError('need >= 2 out-of-transit and >= 1 in-transit samples')\n"
            "    oot_sorted = sorted(oot)\n"
            "    mid = len(oot_sorted) // 2\n"
            "    baseline = (\n"
            "        oot_sorted[mid]\n"
            "        if len(oot_sorted) % 2 == 1\n"
            "        else 0.5 * (oot_sorted[mid - 1] + oot_sorted[mid])\n"
            "    )\n"
            "    norm_oot = [x / baseline for x in oot]\n"
            "    mean_oot = sum(norm_oot) / len(norm_oot)\n"
            "    var_oot = sum((x - mean_oot) ** 2 for x in norm_oot) / (len(norm_oot) - 1)\n"
            "    sigma_oot = math.sqrt(var_oot)\n"
            "    if sigma_oot <= 0.0:\n"
            "        raise ValueError('out-of-transit scatter must be > 0')\n"
            "    mean_in = (sum(dip) / len(dip)) / baseline\n"
            "    rel_depth = max(0.0, 1.0 - mean_in)\n"
            "    snr = rel_depth / (sigma_oot / math.sqrt(len(dip)))\n"
            "    return {\n"
            "        'baseline_flux': round(baseline, 6),\n"
            "        'relative_depth_ppm': round(rel_depth * 1_000_000.0, 2),\n"
            "        'transit_snr': round(snr, 4),\n"
            "    }"
        ),
        "base_checkpoint_output": (
            "import math\n\n"
            "def analyze_transit_dip(flux, in_transit):\n"
            "    oot = [f for f, flag in zip(flux, in_transit) if not flag]\n"
            "    dip = [f for f, flag in zip(flux, in_transit) if flag]\n"
            "    # Arithmetic mean baseline sensitive to stellar flare outliers\n"
            "    baseline = sum(oot) / len(oot)\n"
            "    rel_depth = (baseline - (sum(dip) / len(dip))) / baseline\n"
            "    sigma = math.sqrt(sum((x / baseline - 1.0) ** 2 for x in oot) / len(oot))\n"
            "    snr = rel_depth / (sigma / math.sqrt(len(dip)))\n"
            "    return {'relative_depth_ppm': rel_depth * 1e6, 'transit_snr': snr}"
        ),
    },
)

ARENA_TASKS_BY_ID: dict[str, dict[str, str]] = {t["task_id"]: t for t in CURATED_ARENA_TASKS}


def blind_ab_arena_csp(nonce: str) -> str:
    """Return a strict per-request nonce-pinned Content-Security-Policy for `/arena-noel`."""
    if isinstance(nonce, bool) or not isinstance(nonce, str) or not _NONCE_RE.fullmatch(nonce):
        raise ValueError("nonce must be 8-80 URL-safe / base64 characters")
    return (
        "default-src 'none'; "
        f"style-src 'nonce-{nonce}'; "
        f"script-src 'nonce-{nonce}'; "
        "connect-src 'self'; "
        "img-src 'self' data:; "
        "form-action 'self'; "
        "base-uri 'none'; "
        "frame-ancestors 'none'"
    )


def pick_arena_language(
    query_lang: str | None,
    accept_language: str | None,
    default: str = "fr",
) -> str:
    """Resolve `'fr'` or `'en'` from `?lang=`, `Accept-Language`, or `default`."""
    if query_lang is not None and (isinstance(query_lang, bool) or not isinstance(query_lang, str)):
        raise ValueError("query_lang must be a string or None")
    if accept_language is not None and (isinstance(accept_language, bool) or not isinstance(accept_language, str)):
        raise ValueError("accept_language must be a string or None")
    if isinstance(default, bool) or not isinstance(default, str) or default not in SUPPORTED_ARENA_LANGS:
        raise ValueError("default must be 'fr' or 'en'")

    if query_lang:
        q_bounded = query_lang[:MAX_QUERY_LANG_CHARS]
        if q_bounded.isascii():
            q_primary = q_bounded.strip().lower().split("-")[0].split("_")[0]
            if q_primary in SUPPORTED_ARENA_LANGS:
                return q_primary

    best: tuple[float, int, str] | None = None
    raw_accept = (accept_language or "")[:MAX_ACCEPT_LANGUAGE_CHARS]
    for position, part in enumerate(raw_accept.split(",", MAX_ACCEPT_LANGUAGE_PARTS)[:MAX_ACCEPT_LANGUAGE_PARTS]):
        if not part.isascii():
            continue
        tag, _, params = part.strip().partition(";")
        primary = tag.strip().lower().split("-")[0]
        if primary not in SUPPORTED_ARENA_LANGS:
            continue
        quality = 1.0
        if "q" in params.lower():
            match = _Q_PARAM_RE.fullmatch(params)
            quality = float(match.group(1)) if match else 0.0
        if quality > 0 and (best is None or (quality, -position) > (best[0], -best[1])):
            best = (quality, position, primary)
    return best[2] if best else default


def wilson_95_lower_bound(wins: int, trials: int, z: float = Z_WILSON_95) -> float:
    """Compute the exact binomial 95% Wilson confidence interval lower bound (`0.0` when `trials == 0`)."""
    if isinstance(wins, bool) or not isinstance(wins, int) or isinstance(trials, bool) or not isinstance(trials, int):
        raise ValueError("wins and trials must be integers")
    if trials < 0 or wins < 0 or wins > trials:
        raise ValueError("0 <= wins <= trials is required")
    if isinstance(z, bool) or not isinstance(z, (int, float)) or not math.isfinite(float(z)) or float(z) <= 0.0:
        raise ValueError("z must be a finite positive float")
    if trials == 0 or wins == 0:
        return 0.0
    n = float(trials)
    p_hat = float(wins) / n
    z_val = float(z)
    z2 = z_val * z_val
    denom = 1.0 + z2 / n
    center = p_hat + z2 / (2.0 * n)
    radicand = max(0.0, (p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))
    return min(1.0, max(0.0, (center - z_val * math.sqrt(radicand)) / denom))


def _validate_secret(secret: bytes) -> bytes:
    if not isinstance(secret, (bytes, bytearray)) or len(secret) < 16 or len(secret) > MAX_SECRET_BYTES:
        raise ValueError("secret must be between 16 and 4096 bytes")
    return bytes(secret)


def sign_blind_ballot(
    task_id: str,
    holiday_side: str,
    secret: bytes,
    *,
    salt: str | None = None,
) -> str:
    """Create an HMAC-masked blind ballot token hiding `holiday_side` from client DOM inspection."""
    if (
        isinstance(task_id, bool)
        or not isinstance(task_id, str)
        or len(task_id) > MAX_TASK_ID_CHARS
        or task_id not in ARENA_TASKS_BY_ID
    ):
        raise ValueError("unknown task_id for blind ballot")
    if isinstance(holiday_side, bool) or not isinstance(holiday_side, str) or holiday_side not in VALID_HOLIDAY_SIDES:
        raise ValueError("holiday_side must be 'A' or 'B'")
    key = _validate_secret(secret)
    if salt is None:
        clean_salt = secrets.token_hex(6)
    else:
        if isinstance(salt, bool) or not isinstance(salt, str) or not _SALT_RE.fullmatch(salt):
            raise ValueError("salt must be 8-32 lowercase hex characters")
        clean_salt = salt

    pad_bit = hmac.new(key, f"arena-pad:{clean_salt}:{task_id}".encode(), hashlib.sha256).digest()[0] & 1
    raw_bit = 1 if holiday_side == "A" else 0
    masked_bit = raw_bit ^ pad_bit
    mac = hmac.new(
        key,
        f"arena-mac:{clean_salt}:{task_id}:{masked_bit}".encode(),
        hashlib.sha256,
    ).hexdigest()[:24]
    return f"{clean_salt}.{masked_bit}.{mac}"


def verify_blind_ballot(ballot_token: str, task_id: str, secret: bytes) -> str:
    """Verify an HMAC-masked `ballot_token` for `task_id` and return the original `holiday_side` (`'A'` or `'B'`)."""
    if (
        isinstance(task_id, bool)
        or not isinstance(task_id, str)
        or len(task_id) > MAX_TASK_ID_CHARS
        or task_id not in ARENA_TASKS_BY_ID
    ):
        raise ValueError("unknown task_id for blind ballot")
    if isinstance(ballot_token, bool) or not isinstance(ballot_token, str):
        raise ValueError("ballot_token must be a string")
    key = _validate_secret(secret)
    if len(ballot_token) > MAX_BALLOT_TOKEN_CHARS or not ballot_token.isascii():
        raise ValueError("malformed ballot_token")
    parts = ballot_token.strip().split(".")
    if len(parts) != 3:
        raise ValueError("malformed ballot_token")
    salt, masked_str, mac = parts
    if not _SALT_RE.fullmatch(salt) or masked_str not in ("0", "1") or not _MAC_RE.fullmatch(mac):
        raise ValueError("malformed ballot_token fields")
    masked_bit = int(masked_str)
    expected_mac = hmac.new(
        key,
        f"arena-mac:{salt}:{task_id}:{masked_bit}".encode(),
        hashlib.sha256,
    ).hexdigest()[:24]
    if not hmac.compare_digest(mac, expected_mac):
        raise ValueError("invalid ballot_token signature")
    pad_bit = hmac.new(key, f"arena-pad:{salt}:{task_id}".encode(), hashlib.sha256).digest()[0] & 1
    raw_bit = masked_bit ^ pad_bit
    return "A" if raw_bit == 1 else "B"


def build_blind_task_cards(
    *,
    lang: str = "fr",
    rng: random.Random | None = None,
    secret: bytes | None = None,
) -> list[dict[str, str]]:
    """Build curated side-by-side blind A/B task cards with randomized `A` / `B` checkpoint placement."""
    if isinstance(lang, bool) or not isinstance(lang, str):
        raise ValueError("lang must be a string")
    is_fr = lang[:MAX_QUERY_LANG_CHARS].strip().lower().startswith("fr")
    active_rng: random.Random = rng if rng is not None else secrets.SystemRandom()
    active_secret = _validate_secret(secret) if secret is not None else secrets.token_bytes(32)

    cards: list[dict[str, str]] = []
    for task in CURATED_ARENA_TASKS:
        holiday_side = active_rng.choice(("A", "B"))
        if holiday_side == "A":
            cand_a = task["holiday_checkpoint_output"]
            cand_b = task["base_checkpoint_output"]
        else:
            cand_a = task["base_checkpoint_output"]
            cand_b = task["holiday_checkpoint_output"]

        cards.append(
            {
                "task_id": task["task_id"],
                "domain": task["domain"],
                "badge": task["badge_fr"] if is_fr else task["badge_en"],
                "title": task["title_fr"] if is_fr else task["title_en"],
                "prompt": task["prompt_fr"] if is_fr else task["prompt_en"],
                "candidate_a": cand_a,
                "candidate_b": cand_b,
                "ballot_token": sign_blind_ballot(task["task_id"], holiday_side, active_secret),
            }
        )
    return cards


def _normalize_chosen_side(chosen_side: str) -> str:
    if (
        isinstance(chosen_side, bool)
        or not isinstance(chosen_side, str)
        or len(chosen_side) > MAX_CHOSEN_SIDE_CHARS
        or not chosen_side.isascii()
    ):
        raise ValueError("chosen_side must be 'A', 'B', or 'tie'")
    cleaned = chosen_side.strip()
    if cleaned in ("A", "a"):
        return "A"
    if cleaned in ("B", "b"):
        return "B"
    if cleaned.lower() == "tie":
        return "tie"
    raise ValueError("chosen_side must be 'A', 'B', or 'tie'")


def _bucket_stats_from_counts(holiday_wins: int, baseline_wins: int, ties: int) -> dict[str, Any]:
    total = holiday_wins + baseline_wins + ties
    decisive = holiday_wins + baseline_wins
    win_ratio = round(holiday_wins / decisive, 4) if decisive > 0 else None
    wilson_lb = round(wilson_95_lower_bound(holiday_wins, decisive), 4)
    return {
        "total_votes": total,
        "decisive_votes": decisive,
        "holiday_wins": holiday_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "win_ratio": win_ratio,
        "wilson_95_lower_bound": wilson_lb,
    }


class BlindArenaCollector:
    """Thread-safe bounded zero-VRAM collector for blind A/B holiday checkpoint evaluations."""

    def __init__(
        self,
        *,
        secret: bytes | None = None,
        max_votes: int = MAX_ARENA_VOTES,
    ) -> None:
        if (
            isinstance(max_votes, bool)
            or not isinstance(max_votes, int)
            or max_votes < 1
            or max_votes > MAX_ARENA_VOTES_CEILING
        ):
            raise ValueError(f"max_votes must be an integer between 1 and {MAX_ARENA_VOTES_CEILING}")
        self._secret = _validate_secret(secret) if secret is not None else secrets.token_bytes(32)
        self._max_votes = max_votes
        self._lock = threading.Lock()
        self._votes: deque[tuple[str, str, str, str]] = deque(maxlen=max_votes)
        self._holiday_wins = 0
        self._baseline_wins = 0
        self._ties = 0
        self._side_counts: dict[str, int] = {"A": 0, "B": 0, "tie": 0}
        self._by_profile: dict[str, int] = {"dev": 0, "gamer": 0}
        self._by_task_counts: dict[str, list[int]] = {tid: [0, 0, 0] for tid in ARENA_TASKS_BY_ID}
        self._lifetime_votes = 0
        self._evicted_votes = 0

    def window_diagnostics(self) -> dict[str, int]:
        """Return atomic ring-buffer bounds and lifetime/eviction counters."""
        with self._lock:
            return {
                "max_votes": self._max_votes,
                "retained_votes": len(self._votes),
                "lifetime_votes": self._lifetime_votes,
                "evicted_votes": self._evicted_votes,
            }

    def issue_ballot(self, task_id: str, holiday_side: str, *, salt: str | None = None) -> str:
        """Issue an HMAC-masked blind ballot token bound to this collector's secret."""
        return sign_blind_ballot(task_id, holiday_side, self._secret, salt=salt)

    def build_cards(
        self,
        *,
        lang: str = "fr",
        rng: random.Random | None = None,
    ) -> list[dict[str, str]]:
        """Build per-request randomized blind A/B task cards signed with this collector's secret."""
        return build_blind_task_cards(lang=lang, rng=rng, secret=self._secret)

    def _apply_delta_unlocked(
        self,
        task_id: str,
        chosen_side: str,
        profile: str,
        preferred: str,
        sign: int,
    ) -> None:
        task_bucket = self._by_task_counts[task_id]
        if preferred == "holiday_checkpoint":
            self._holiday_wins += sign
            task_bucket[0] += sign
        elif preferred == "baseline_checkpoint":
            self._baseline_wins += sign
            task_bucket[1] += sign
        else:
            self._ties += sign
            task_bucket[2] += sign
        self._side_counts[chosen_side] += sign
        self._by_profile[profile] += sign

    def _summary_unlocked(self) -> dict[str, Any]:
        overall = _bucket_stats_from_counts(self._holiday_wins, self._baseline_wins, self._ties)
        by_task: dict[str, dict[str, Any]] = {}
        for task_id, task in ARENA_TASKS_BY_ID.items():
            hw, bw, ti = self._by_task_counts[task_id]
            by_task[task_id] = {
                "domain": task["domain"],
                **_bucket_stats_from_counts(hw, bw, ti),
            }
        return {
            "object": "hivemind.blind_ab_arena.summary",
            "campaign": "noel-2026-holiday-checkpoint-arena",
            "zero_vram_inference": True,
            **overall,
            "chosen_side_counts": dict(self._side_counts),
            "by_profile": dict(self._by_profile),
            "by_task": by_task,
            "compliance": {
                "personal_pc_only": True,
                "corporate_hardware_or_code_forbidden": True,
                "monetary_cost_eur": "0.00 EUR",
                "monetary_cost_usd": "0.00 USD",
                "license": "AGPL-3.0",
                "zero_live_gpu_vram": True,
                "rule_fr": PERSONAL_PC_RULE_FR,
                "rule_en": PERSONAL_PC_RULE_EN,
            },
        }

    def record_vote(
        self,
        *,
        task_id: str,
        chosen_side: str,
        profile: str,
        ballot_token: str | None = None,
        holiday_side: str | None = None,
    ) -> dict[str, Any]:
        """Validate and record a blind A/B evaluation vote and return atomic summary + receipt."""
        if (
            isinstance(task_id, bool)
            or not isinstance(task_id, str)
            or len(task_id) > MAX_TASK_ID_CHARS
            or not task_id.isascii()
        ):
            raise ValueError("task_id must be a known curated task identifier")
        norm_task_id = task_id.strip()
        task = ARENA_TASKS_BY_ID.get(norm_task_id)
        if task is None:
            raise ValueError("unknown curated task_id")

        norm_side = _normalize_chosen_side(chosen_side)

        if (
            isinstance(profile, bool)
            or not isinstance(profile, str)
            or len(profile) > MAX_PROFILE_CHARS
            or not profile.isascii()
        ):
            raise ValueError("profile must be 'dev' or 'gamer'")
        norm_profile = profile.strip().lower()
        if norm_profile not in VALID_ARENA_PROFILES:
            raise ValueError("profile must be 'dev' or 'gamer'")

        if ballot_token is not None and holiday_side is not None:
            raise ValueError("provide at most one of ballot_token or holiday_side")

        if ballot_token is not None:
            resolved_holiday_side = verify_blind_ballot(ballot_token, norm_task_id, self._secret)
        elif holiday_side is not None:
            if isinstance(holiday_side, bool) or not isinstance(holiday_side, str) or holiday_side not in VALID_HOLIDAY_SIDES:
                raise ValueError("holiday_side must be 'A' or 'B'")
            resolved_holiday_side = holiday_side
        else:
            resolved_holiday_side = task["default_holiday_side"]

        if norm_side == "tie":
            preferred_checkpoint = "tie"
        elif norm_side == resolved_holiday_side:
            preferred_checkpoint = "holiday_checkpoint"
        else:
            preferred_checkpoint = "baseline_checkpoint"

        with self._lock:
            if len(self._votes) == self._max_votes:
                old_tid, old_side, old_prof, old_pref = self._votes.popleft()
                self._apply_delta_unlocked(old_tid, old_side, old_prof, old_pref, -1)
                self._evicted_votes += 1
            self._votes.append((norm_task_id, norm_side, norm_profile, preferred_checkpoint))
            self._apply_delta_unlocked(norm_task_id, norm_side, norm_profile, preferred_checkpoint, 1)
            self._lifetime_votes += 1
            summary_doc = self._summary_unlocked()

        return {
            **summary_doc,
            "recorded": True,
            "last_vote": {
                "task_id": norm_task_id,
                "domain": task["domain"],
                "chosen_side": norm_side,
                "profile": norm_profile,
                "revealed_holiday_side": resolved_holiday_side,
                "preferred_checkpoint": preferred_checkpoint,
            },
        }

    def summary(self) -> dict[str, Any]:
        """Compute exact vote counts, `win_ratio`, and `wilson_95_lower_bound` across recorded votes."""
        with self._lock:
            return self._summary_unlocked()


def _validate_gateway_url(gateway_url: str) -> str:
    if (
        isinstance(gateway_url, bool)
        or not isinstance(gateway_url, str)
        or len(gateway_url) > MAX_GATEWAY_URL_CHARS
        or not gateway_url.isascii()
        or not gateway_url.strip()
    ):
        raise ValueError("gateway_url must be a non-empty http(s) URL")
    cleaned = gateway_url.strip().rstrip("/")
    if any(ord(ch) < 33 or ord(ch) == 127 or ch in _FORBIDDEN_URL_CHARS for ch in cleaned):
        raise ValueError("gateway_url contains forbidden characters")
    parsed = urlparse(cleaned)
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("gateway_url has an invalid port") from None
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path not in ("", "/")
        or parsed.params
        or parsed.query
        or parsed.fragment
        or (port is not None and not (1 <= port <= 65535))
    ):
        raise ValueError("gateway_url must be a clean http(s) origin without credentials")
    return cleaned


def _validate_route_path(path: str) -> str:
    if (
        isinstance(path, bool)
        or not isinstance(path, str)
        or len(path) > MAX_ROUTE_PATH_CHARS
        or not path.startswith("/")
        or path.startswith("//")
        or ".." in path.split("/")
        or not _SAFE_PATH_RE.fullmatch(path)
    ):
        raise ValueError("path must be a clean URL path starting with '/'")
    return path


def _sanitize_cards_for_render(cards: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    if isinstance(cards, (str, bytes, bytearray)) or not isinstance(cards, Sequence) or len(cards) > MAX_RENDER_CARDS:
        raise ValueError("cards must be a bounded sequence of task card mappings")
    clean_cards: list[dict[str, str]] = []
    required_text_keys = ("badge", "title", "prompt", "candidate_a", "candidate_b")
    for card in cards:
        if not isinstance(card, Mapping):
            raise ValueError("each card must be a Mapping")
        tid = card.get("task_id")
        ballot = card.get("ballot_token")
        if isinstance(tid, bool) or not isinstance(tid, str) or not _TASK_ID_RE.fullmatch(tid):
            raise ValueError("card task_id must be a clean lowercase slug")
        if isinstance(ballot, bool) or not isinstance(ballot, str) or not _BALLOT_SAFE_RE.fullmatch(ballot):
            raise ValueError("card ballot_token must be a bounded ASCII token")
        normalized: dict[str, str] = {"task_id": tid, "ballot_token": ballot}
        for key in required_text_keys:
            val = card.get(key)
            if isinstance(val, bool) or not isinstance(val, str) or len(val) > MAX_CARD_TEXT_CHARS or "\x00" in val:
                raise ValueError(f"card {key} must be a bounded string without null bytes")
            normalized[key] = val
        clean_cards.append(normalized)
    return clean_cards


def _format_win_ratio(win_ratio: Any) -> str:
    if win_ratio is None or isinstance(win_ratio, bool):
        return NOT_AVAILABLE
    if isinstance(win_ratio, (int, float)) and math.isfinite(float(win_ratio)):
        return f"{float(win_ratio) * 100.0:.1f}%"
    return NOT_AVAILABLE


def _format_wilson(lb: Any) -> str:
    if lb is None or isinstance(lb, bool):
        return NOT_AVAILABLE
    if isinstance(lb, (int, float)) and math.isfinite(float(lb)):
        return f"{float(lb):.4f}"
    return NOT_AVAILABLE


def _format_count(val: Any) -> str:
    if val is None or isinstance(val, bool):
        return NOT_AVAILABLE
    if isinstance(val, (int, float)) and math.isfinite(float(val)) and float(val) == int(val):
        return str(int(val))
    return NOT_AVAILABLE


# Client-side controller: strictly uses DOM textContent / setAttribute / className (zero HTML string sinks).
_ARENA_NOEL_JS = r"""(function () {
  var root = document.getElementById('arenaNoelApp');
  var isFr = root && root.getAttribute('data-lang') === 'fr';
  var selectedProfile = 'dev';

  function byId(id) {
    return document.getElementById(id);
  }

  function setText(id, val) {
    var el = byId(id);
    if (el) el.textContent = String(val);
  }

  function fmtCount(v) {
    if (v === null || v === undefined || typeof v === 'boolean') return 'n/a';
    var n = Number(v);
    if (!Number.isFinite(n)) return 'n/a';
    return String(Math.trunc(n));
  }

  function fmtRatio(v) {
    if (v === null || v === undefined || typeof v === 'boolean') return 'n/a';
    var n = Number(v);
    if (!Number.isFinite(n)) return 'n/a';
    return (n * 100).toFixed(1) + '%';
  }

  function fmtWilson(v) {
    if (v === null || v === undefined || typeof v === 'boolean') return 'n/a';
    var n = Number(v);
    if (!Number.isFinite(n)) return 'n/a';
    return n.toFixed(4);
  }

  function selectProfile(prof) {
    if (prof !== 'dev' && prof !== 'gamer') return;
    selectedProfile = prof;
    ['dev', 'gamer'].forEach(function (p) {
      var btn = byId('profileBtn-' + p);
      if (btn) {
        var active = p === prof;
        btn.setAttribute('aria-pressed', active ? 'true' : 'false');
        btn.className = active ? 'prof-btn active' : 'prof-btn';
      }
    });
    setText(
      'arenaProfileStatus',
      isFr
        ? 'Profil évaluateur actif : ' + prof + ' (anonyme • 0,00 EUR).'
        : 'Active evaluator profile: ' + prof + ' (anonymous • 0.00 EUR).'
    );
  }

  function applySummary(d) {
    if (!d || typeof d !== 'object') return;
    setText('arenaTotalVotes', fmtCount(d.total_votes));
    setText(
      'arenaRecordBreakdown',
      fmtCount(d.holiday_wins) + ' / ' + fmtCount(d.baseline_wins) + ' / ' + fmtCount(d.ties)
    );
    setText('arenaWinRatio', fmtRatio(d.win_ratio));
    setText('arenaWilsonLower', fmtWilson(d.wilson_95_lower_bound));
    var prof = d.by_profile || {};
    setText(
      'arenaProfileBreakdown',
      'dev: ' + fmtCount(prof.dev) + ' | gamer: ' + fmtCount(prof.gamer)
    );
  }

  function refreshSummary() {
    if (typeof fetch !== 'function') return Promise.resolve();
    setText(
      'arenaRefreshNote',
      isFr ? 'Actualisation du bilan en direct…' : 'Refreshing live arena summary…'
    );
    return fetch('/v1/hivemind/arena/summary', { cache: 'no-store' })
      .then(function (r) {
        if (!r || !r.ok) throw new Error('summary_' + (r ? r.status : 'err'));
        return r.json();
      })
      .then(function (d) {
        applySummary(d);
        setText(
          'arenaRefreshNote',
          isFr
            ? 'Bilan synchronisé depuis /v1/hivemind/arena/summary (0 Mio VRAM sollicitée).'
            : 'Summary synced from /v1/hivemind/arena/summary (0 MiB VRAM touched).'
        );
      })
      .catch(function () {
        setText(
          'arenaRefreshNote',
          isFr
            ? 'Lecture hors-ligne : derniers compteurs conservés (aucun chiffre inventé).'
            : 'Offline view: keeping last received counts (no figures invented).'
        );
      });
  }

  function submitVote(taskId, chosenSide) {
    if (typeof fetch !== 'function') return Promise.resolve();
    if (chosenSide !== 'A' && chosenSide !== 'B' && chosenSide !== 'tie') return Promise.resolve();
    var card = byId('arenaCard-' + taskId);
    var ballot = card ? card.getAttribute('data-ballot') : null;
    var payload = {
      task_id: taskId,
      chosen_side: chosenSide,
      profile: selectedProfile
    };
    if (ballot) payload.ballot_token = ballot;

    setText(
      'voteStatus-' + taskId,
      isFr ? 'Envoi du vote aveugle…' : 'Submitting blind vote…'
    );

    return fetch('/v1/hivemind/arena/vote', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
      .then(function (r) {
        if (!r || !r.ok) throw new Error('vote_' + (r ? r.status : 'err'));
        return r.json();
      })
      .then(function (d) {
        applySummary(d);
        var lv = (d && d.last_vote) || {};
        var holSide = (lv.revealed_holiday_side === 'A' || lv.revealed_holiday_side === 'B')
          ? lv.revealed_holiday_side
          : '?';
        var baseSide = holSide === 'A' ? 'B' : (holSide === 'B' ? 'A' : '?');
        ['A', 'B', 'Tie'].forEach(function (suffix) {
          var b = byId('voteBtn' + suffix + '-' + taskId);
          if (b) {
            var expected = suffix === 'Tie' ? 'tie' : suffix;
            var picked = expected === chosenSide;
            b.setAttribute('aria-pressed', picked ? 'true' : 'false');
            b.className = picked ? 'vote-btn active' : 'vote-btn';
          }
        });
        setText(
          'voteStatus-' + taskId,
          isFr
            ? 'Vote enregistré (' + chosenSide + ') — Révélation : Candidat ' + holSide + ' = Checkpoint Noël, Candidat ' + baseSide + ' = Base de départ.'
            : 'Vote recorded (' + chosenSide + ') — Reveal: Candidate ' + holSide + ' = Holiday Checkpoint, Candidate ' + baseSide + ' = Starting Base.'
        );
      })
      .catch(function () {
        setText(
          'voteStatus-' + taskId,
          isFr
            ? 'Échec de l’enregistrement du vote — réessaie dans un instant.'
            : 'Could not record vote — please retry in a moment.'
        );
      });
  }

  ['dev', 'gamer'].forEach(function (p) {
    var btn = byId('profileBtn-' + p);
    if (btn) {
      btn.addEventListener('click', function () {
        selectProfile(p);
      });
    }
  });

  var refreshBtn = byId('refreshArenaBtn');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', function () {
      return refreshSummary();
    });
  }

  [
    'python-sliding-median',
    'rust-frame-decoder',
    'lean4-energy-descent',
    'exoplanet-transit-bls'
  ].forEach(function (taskId) {
    [
      ['A', 'A'],
      ['B', 'B'],
      ['Tie', 'tie']
    ].forEach(function (pair) {
      var btn = byId('voteBtn' + pair[0] + '-' + taskId);
      if (btn) {
        btn.addEventListener('click', function () {
          return submitVote(taskId, pair[1]);
        });
      }
    });
  });
})();"""


def render_blind_ab_arena_page(
    lang: str = "fr",
    nonce: str = "ArenaNoelDefaultNonce2026",
    *,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    path: str = "/arena-noel",
    arena_summary: Mapping[str, Any] | None = None,
    cards: Sequence[Mapping[str, str]] | None = None,
    rng: random.Random | None = None,
    secret: bytes | None = None,
) -> str:
    """Render the strict-CSP bilingual (`/arena-noel`) zero-VRAM Blind A/B Evaluation Arena."""
    if isinstance(lang, bool) or not isinstance(lang, str):
        raise ValueError("lang must be a string")
    blind_ab_arena_csp(nonce)
    safe_gw = html.escape(_validate_gateway_url(gateway_url))
    safe_path = html.escape(_validate_route_path(path))
    if arena_summary is not None and not isinstance(arena_summary, Mapping):
        raise ValueError("arena_summary must be a Mapping or None")

    is_fr = lang[:MAX_QUERY_LANG_CHARS].strip().lower().startswith("fr")
    lang_code = "fr" if is_fr else "en"

    active_cards = (
        _sanitize_cards_for_render(cards)
        if cards is not None
        else build_blind_task_cards(lang=lang_code, rng=rng, secret=secret)
    )

    sm = arena_summary or {}
    raw_prof = sm.get("by_profile") if isinstance(sm.get("by_profile"), Mapping) else {}

    total_txt = html.escape(_format_count(sm.get("total_votes")) if arena_summary is not None else NOT_AVAILABLE)
    record_txt = html.escape(
        f"{_format_count(sm.get('holiday_wins'))} / {_format_count(sm.get('baseline_wins'))} / {_format_count(sm.get('ties'))}"
        if arena_summary is not None
        else f"{NOT_AVAILABLE} / {NOT_AVAILABLE} / {NOT_AVAILABLE}"
    )
    ratio_txt = html.escape(_format_win_ratio(sm.get("win_ratio")) if arena_summary is not None else NOT_AVAILABLE)
    wilson_txt = html.escape(
        _format_wilson(sm.get("wilson_95_lower_bound")) if arena_summary is not None else NOT_AVAILABLE
    )
    prof_txt = html.escape(
        f"dev: {_format_count(raw_prof.get('dev'))} | gamer: {_format_count(raw_prof.get('gamer'))}"
        if arena_summary is not None
        else f"dev: {NOT_AVAILABLE} | gamer: {NOT_AVAILABLE}"
    )

    if is_fr:
        title = "HiveMind — Arène Aveugle A/B de Noël (Zéro VRAM)"
        subtitle = (
            "Compare à l’aveugle le modèle de départ et le checkpoint entraîné pendant les vacances sur 4 tâches "
            "déterministes (Python, Rust, Lean 4, transit d’exoplanète) — sans toucher à la mémoire vidéo T4."
        )
        rule_text = PERSONAL_PC_RULE_FR
        summary_h2 = "1. Bilan Statistique en Direct (Ratio de Victoire & Borne Inférieure de Wilson 95 %)"
        summary_desc = (
            "Chaque vote met à jour les compteurs exacts, le ratio de victoire sur votes décisifs et la borne "
            "inférieure de confiance de Wilson à 95 % depuis /v1/hivemind/arena/summary."
        )
        lbl_total = "Votes enregistrés"
        lbl_record = "Noël / Base / Égalité"
        lbl_ratio = "Ratio victoire Noël"
        lbl_wilson = "Borne inf. Wilson 95 %"
        lbl_profiles = "Répartition profils"
        btn_refresh = "Actualiser le bilan"
        refresh_note = "Source : /v1/hivemind/arena/summary (instantanés déterministes • 0 Mio VRAM)."
        prof_h2 = "2. Profil Évaluateur Anonyme"
        prof_dev = "Profil Développeur (dev)"
        prof_gamer = "Profil Gamer PC (gamer)"
        prof_status = "Profil évaluateur actif : dev (anonyme • 0,00 EUR)."
        tasks_h2 = "3. Comparaison Aveugle Côte-à-Côte (Étiquettes A / B Tirées au Sort)"
        tasks_desc = (
            "Pour chaque défi ci-dessous, l’ordre entre Candidat A et Candidat B est tiré au sort à chaque chargement. "
            "L’identité du checkpoint n’est révélée qu’après ton vote."
        )
        cand_a_hdr = "Candidat A (aveugle)"
        cand_b_hdr = "Candidat B (aveugle)"
        vote_a_lbl = "Voter Candidat A"
        vote_b_lbl = "Voter Candidat B"
        vote_tie_lbl = "Égalité"
        vote_idle = "En attente de ton vote aveugle (étiquettes A/B masquées par HMAC)."
        cli_hint_lbl = "Tester aussi son PC en lecture seule (Python standard) :"
        lang_switch = f'<a href="{safe_path}?lang=en">English version</a>'
    else:
        title = "HiveMind — Zero-VRAM Holiday Blind A/B Evaluation Arena"
        subtitle = (
            "Blindly compare the starting base checkpoint against the holiday-trained checkpoint across 4 curated "
            "deterministic tasks (Python, Rust, Lean 4, exoplanet transit analysis) — touching 0 MiB of T4 VRAM."
        )
        rule_text = PERSONAL_PC_RULE_EN
        summary_h2 = "1. Live Blind Arena Summary (Win Ratio & 95% Wilson Lower Bound)"
        summary_desc = (
            "Every blind vote updates exact counts, decisive win ratio, and the conservative 95% Wilson confidence "
            "lower bound directly from /v1/hivemind/arena/summary."
        )
        lbl_total = "Total Votes"
        lbl_record = "Holiday / Base / Tie"
        lbl_ratio = "Holiday Win Ratio"
        lbl_wilson = "Wilson 95% Lower Bound"
        lbl_profiles = "By Evaluator Profile"
        btn_refresh = "Refresh Live Summary"
        refresh_note = "Source: /v1/hivemind/arena/summary (deterministic snapshots • 0 MiB VRAM)."
        prof_h2 = "2. Anonymous Evaluator Profile"
        prof_dev = "Developer Profile (dev)"
        prof_gamer = "PC Gamer Profile (gamer)"
        prof_status = "Active evaluator profile: dev (anonymous • 0.00 EUR)."
        tasks_h2 = "3. Randomized Side-by-Side Blind Tasks (Labels A / B Shuffled Per Load)"
        tasks_desc = (
            "For every challenge below, Candidate A and Candidate B are randomly assigned on each page load. "
            "Checkpoint identity stays hidden until after you cast your vote."
        )
        cand_a_hdr = "Candidate A (blind)"
        cand_b_hdr = "Candidate B (blind)"
        vote_a_lbl = "Prefer Candidate A"
        vote_b_lbl = "Prefer Candidate B"
        vote_tie_lbl = "Tie"
        vote_idle = "Awaiting your blind vote (A/B assignment HMAC-masked)."
        cli_hint_lbl = "Optional read-only local preflight check (Python stdlib only):"
        lang_switch = f'<a href="{safe_path}?lang=fr">Version française</a>'

    task_sections_html: list[str] = []
    for card in active_cards:
        tid = html.escape(card["task_id"])
        badge = html.escape(card["badge"])
        ctitle = html.escape(card["title"])
        cprompt = html.escape(card["prompt"])
        cand_a = html.escape(card["candidate_a"])
        cand_b = html.escape(card["candidate_b"])
        ballot = html.escape(card["ballot_token"])
        task_sections_html.append(
            f'<article id="arenaCard-{tid}" class="task-card" data-task-id="{tid}" data-ballot="{ballot}">'
            f'<div class="task-head"><span class="badge">{badge}</span><h3>{ctitle}</h3></div>'
            f'<p class="prompt">{cprompt}</p>'
            f'<div class="ab-grid">'
            f'<div class="ab-col"><h4>{html.escape(cand_a_hdr)}</h4><pre id="candA-{tid}">{cand_a}</pre></div>'
            f'<div class="ab-col"><h4>{html.escape(cand_b_hdr)}</h4><pre id="candB-{tid}">{cand_b}</pre></div>'
            f"</div>"
            f'<div class="vote-bar">'
            f'<button type="button" id="voteBtnA-{tid}" class="vote-btn" aria-pressed="false">{html.escape(vote_a_lbl)}</button>'
            f'<button type="button" id="voteBtnB-{tid}" class="vote-btn" aria-pressed="false">{html.escape(vote_b_lbl)}</button>'
            f'<button type="button" id="voteBtnTie-{tid}" class="vote-btn" aria-pressed="false">{html.escape(vote_tie_lbl)}</button>'
            f"</div>"
            f'<p id="voteStatus-{tid}" class="muted">{html.escape(vote_idle)}</p>'
            f"</article>"
        )

    safe_nonce = html.escape(nonce)
    return f"""<!DOCTYPE html>
<html lang="{lang_code}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style nonce="{safe_nonce}">
    body {{ font-family: system-ui, -apple-system, sans-serif; background: #0b1120; color: #e2e8f0; margin: 0; padding: 22px; line-height: 1.5; }}
    .wrap {{ max-width: 1080px; margin: 0 auto; }}
    nav.topnav {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 16px; font-size: 0.9rem; }}
    a {{ color: #7dd3fc; }}
    h1 {{ font-size: 1.65rem; margin: 0 0 6px 0; color: #f8fafc; }}
    .sub {{ color: #94a3b8; margin: 0 0 14px 0; }}
    .rule {{ background: #1e293b; border-left: 4px solid #f59e0b; padding: 12px 16px; border-radius: 6px; margin-bottom: 18px; font-size: 0.92rem; }}
    .panel {{ background: #111827; border: 1px solid #334155; border-radius: 10px; padding: 18px; margin-bottom: 18px; }}
    .panel h2 {{ font-size: 1.15rem; margin: 0 0 8px 0; color: #38bdf8; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-top: 12px; }}
    .stat-box {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 12px; }}
    .stat-val {{ font-size: 1.4rem; font-weight: 700; color: #f8fafc; display: block; }}
    .stat-lbl {{ font-size: 0.85rem; color: #cbd5e1; }}
    .prof-row, .vote-bar {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }}
    .prof-btn, .vote-btn, .act-btn {{ background: #1e293b; color: #e2e8f0; border: 1px solid #38bdf8; border-radius: 6px; padding: 8px 14px; font-size: 0.88rem; font-weight: 600; cursor: pointer; }}
    .prof-btn.active, .vote-btn.active, .act-btn {{ background: #0284c7; color: #ffffff; border-color: #7dd3fc; }}
    .act-btn {{ margin-top: 12px; }}
    .task-card {{ background: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 16px; margin-top: 14px; }}
    .task-head {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
    .task-head h3 {{ margin: 0; font-size: 1.05rem; color: #f8fafc; }}
    .badge {{ background: #1e3a8a; color: #bfdbfe; border-radius: 999px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600; }}
    .prompt {{ color: #cbd5e1; font-size: 0.92rem; margin: 8px 0 12px 0; }}
    .ab-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }}
    .ab-col {{ background: #020617; border: 1px solid #1e293b; border-radius: 8px; padding: 10px; min-width: 0; }}
    .ab-col h4 {{ margin: 0 0 6px 0; font-size: 0.88rem; color: #7dd3fc; }}
    pre {{ background: #020617; color: #a7f3d0; margin: 0; padding: 8px; border-radius: 6px; overflow-x: auto; font-size: 0.82rem; line-height: 1.42; }}
    .muted {{ color: #94a3b8; font-size: 0.86rem; margin: 8px 0 0 0; }}
    footer.foot {{ display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px; color: #94a3b8; font-size: 0.86rem; margin-top: 16px; }}
  </style>
</head>
<body>
  <main id="arenaNoelApp" class="wrap" data-lang="{lang_code}">
    <nav class="topnav" aria-label="HiveMind navigation">
      <a href="/noel">/noel</a>
      <a href="/quick">/quick</a>
      <a href="/raid-noel">/raid-noel</a>
      <a href="/arena-noel">/arena-noel</a>
      <a href="/signup">/signup</a>
      <a href="/join">/join</a>
      <a href="/portal">/portal</a>
      <span>•</span>
      {lang_switch}
    </nav>

    <h1>{html.escape(title)}</h1>
    <p class="sub">{html.escape(subtitle)}</p>
    <div id="personalPcComplianceBanner" class="rule" role="note">{html.escape(rule_text)}</div>

    <section class="panel" aria-labelledby="arenaSummaryHeading">
      <h2 id="arenaSummaryHeading">{html.escape(summary_h2)}</h2>
      <p class="muted">{html.escape(summary_desc)}</p>
      <div class="stats-grid">
        <div class="stat-box">
          <span class="stat-lbl">{html.escape(lbl_total)}</span>
          <span id="arenaTotalVotes" class="stat-val">{total_txt}</span>
        </div>
        <div class="stat-box">
          <span class="stat-lbl">{html.escape(lbl_record)}</span>
          <span id="arenaRecordBreakdown" class="stat-val">{record_txt}</span>
        </div>
        <div class="stat-box">
          <span class="stat-lbl">{html.escape(lbl_ratio)}</span>
          <span id="arenaWinRatio" class="stat-val">{ratio_txt}</span>
        </div>
        <div class="stat-box">
          <span class="stat-lbl">{html.escape(lbl_wilson)}</span>
          <span id="arenaWilsonLower" class="stat-val">{wilson_txt}</span>
        </div>
        <div class="stat-box">
          <span class="stat-lbl">{html.escape(lbl_profiles)}</span>
          <span id="arenaProfileBreakdown" class="stat-val">{prof_txt}</span>
        </div>
      </div>
      <button type="button" id="refreshArenaBtn" class="act-btn">{html.escape(btn_refresh)}</button>
      <p id="arenaRefreshNote" class="muted">{html.escape(refresh_note)}</p>
    </section>

    <section class="panel" aria-labelledby="arenaProfileHeading">
      <h2 id="arenaProfileHeading">{html.escape(prof_h2)}</h2>
      <div class="prof-row">
        <button type="button" id="profileBtn-dev" class="prof-btn active" aria-pressed="true">{html.escape(prof_dev)}</button>
        <button type="button" id="profileBtn-gamer" class="prof-btn" aria-pressed="false">{html.escape(prof_gamer)}</button>
      </div>
      <p id="arenaProfileStatus" class="muted">{html.escape(prof_status)}</p>
    </section>

    <section class="panel" aria-labelledby="arenaTasksHeading">
      <h2 id="arenaTasksHeading">{html.escape(tasks_h2)}</h2>
      <p class="muted">{html.escape(tasks_desc)}</p>
      {"".join(task_sections_html)}
    </section>

    <footer class="foot">
      <span>{html.escape(cli_hint_lbl)} <code>python3 scripts/hivemind_doctor.py --gateway {safe_gw}</code></span>
      <span>AGPL-3.0 • 0.00 EUR / 0.00 USD</span>
    </footer>
  </main>
  <script nonce="{safe_nonce}">
{_ARENA_NOEL_JS}
  </script>
</body>
</html>
"""
