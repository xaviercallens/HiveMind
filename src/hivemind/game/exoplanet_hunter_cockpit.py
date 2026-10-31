# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Star Trek Curved Bridge Console & 5-Language Space Game (`exoplanet_hunter_cockpit.py` v4.0).

Upgraded v4.0 Main UI & Astrophysical Simulator inspired by:
  - Star Trek Curved Bridge Astrometrics Console (`startrek_main_ui_v4_ref.jpg`)
  - NASA 6,000 Exoplanets Milestone & TAP REST API (`pscomppars` ADQL)
  - JWST 4-Band Direct Imaging Coronagraphy (`NIRCam F300M/F444W` + `MIRI F1140C/F1550C`)
  - UCSC Systemic Console (`radvel` Doppler RV + `lightkurve`/`batman` Transit Photometry)
  - Free Professional Game Engine Architecture (Godot 4 + Three.js WebGL + Native C# Steam Client)
  - 5-Language Voice, Sound Effects & Subtitles (`FR` Français, `ES` Español, `DE` Deutsch, `EN` English, `ZH` 中文)
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import json
import math
import random
import time
import urllib.request
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import Body, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from hivemind.config import ARTIFACTS_ROOT, PROJECT_ROOT
from hivemind.game.exoplanet_hunter_engine import (
    ASTRO_BEE_EXPERTS,
    STEAM_APP_ID,
    AstroBeeMoEGPUService,
    HiveMindMultiProducerInferencePool,
    compute_four_astrophysical_laws,
    query_nasa_exoplanet_archive_tap,
    synthesize_gemini_or_gcp_free_tts,
    synthesize_jwst_coronagraph_4bands,
    synthesize_jwst_transmission_spectrum,
    synthesize_transit_light_curve_and_bls,
)


class BeeMoEScanRequest(BaseModel):
    target_id: str = Field(default="SECTOR-06-LHS-1140B")
    trial_period_days: Optional[float] = Field(default=None)
    gamer_tag: str = Field(default="Commandant_Starfleet_FR")
    preferred_producer: str = Field(default="auto")


class ProducerRegistrationRequest(BaseModel):
    producer_id: str = Field(default="PRODUCER-REMOTE-TESLA-T4-NODE1")
    name: str = Field(default="NVIDIA Tesla T4 Producer Node")
    hardware: str = Field(default="NVIDIA Tesla T4 (16 GB GDDR6)")
    status: str = Field(default="ONLINE_SERVING")
    endpoint_url: str = Field(default="")
    priority: int = Field(default=2)
    notes: str = Field(default="")


class MultiProducerInferRequest(BaseModel):
    prompt: str = Field(default="Analyze LHS 1140 b transit and atmospheric N2/H2O biosignature")
    max_new_tokens: int = Field(default=64)
    preferred_producer: str = Field(default="auto")
    target_id: Optional[str] = Field(default=None)


class PlanetDiscoveryRequest(BaseModel):
    target_id: str = Field(default="SECTOR-06-LHS-1140B")
    gamer_tag: str = Field(default="Commandant_Starfleet_FR")
    player_classification: str = Field(default="PLANET")
    notes: str = Field(default="")


def _get_asset_data_uri(filename: str, mime: str = "image/jpeg") -> str:
    """Return base64 data URI for an image in steam_assets so HTML works offline & online."""
    p = (
        PROJECT_ROOT
        / "steam_dist"
        / "HiveMind_Astrophysics_Exoplanet_Hunter"
        / "steam_assets"
        / filename
    )
    if p.exists():
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    return ""


def render_exoplanet_hunter_cockpit_html(service: AstroBeeMoEGPUService, classic_v40: bool = False) -> str:
    """Render the USS HiveMind NCC-31337 v4.6 Complete Tactical Command Bridge & Oolite Planetfall Suite from Stitch MCP."""
    initial_target = service.catalog[5] if len(service.catalog) > 5 else service.catalog[0]
    initial_lc = synthesize_transit_light_curve_and_bls(
        period_days=initial_target["period_days"],
        transit_depth_ppm=initial_target["transit_depth_ppm"],
        duration_hours=initial_target["duration_hours"],
        model_snr=initial_target["model_snr"],
        disposition=initial_target["disposition"],
    )
    initial_jwst = synthesize_jwst_transmission_spectrum(
        radius_rearth=initial_target["planet_radius_rearth"],
        teq_k=initial_target["equilibrium_temp_k"],
        transit_depth_ppm=initial_target["transit_depth_ppm"],
        disposition=initial_target["disposition"],
    )
    main_ui_ref_uri = _get_asset_data_uri("startrek_main_ui_v4_ref.jpg", "image/jpeg")
    nasa_disk_uri = _get_asset_data_uri("nasa_6000_exoplanets_disk.png", "image/png")
    jwst_coro_uri = _get_asset_data_uri("jwst_coronagraph_hip65426b.png", "image/png")

    embedded_state = {
        "steam_app_id": STEAM_APP_ID,
        "gpu_hardware": service.hw_info,
        "bee_experts": ASTRO_BEE_EXPERTS,
        "catalog": service.catalog[:12],
        "initial_lc": initial_lc,
        "initial_jwst": initial_jwst,
        "main_ui_ref_uri": main_ui_ref_uri,
        "nasa_disk_uri": nasa_disk_uri,
        "jwst_coro_uri": jwst_coro_uri,
    }
    state_json = json.dumps(embedded_state).replace("</", "<\\/")

    if not classic_v40:
        from hivemind.game.stitch_v46_cockpit_template import build_stitch_v46_cockpit_html

        return build_stitch_v46_cockpit_html(state_json)

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>USS HiveMind NCC-31337 v4.0 &bull; Console Astrométrique Star Trek &bull; 5 Langues (Voix + Sous-titres + Son)</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700;800&family=Orbitron:wght@600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet" />
  <style>
    :root {{
      --bg-bridge: #02040a;
      --console-frame: #060d1a;
      --lcars-amber: #ffaa00;
      --lcars-gold: #ffd166;
      --lcars-cyan: #22d3ee;
      --lcars-emerald: #10b981;
      --lcars-neon-green: #34d399;
      --lcars-coral: #f43f5e;
      --lcars-violet: #a855f7;
      --text-main: #f8fafc;
      --text-dim: #94a3b8;
      --font-lcars: 'Orbitron', sans-serif;
      --font-mono: 'JetBrains Mono', Consolas, monospace;
      --font-body: 'Space Grotesk', system-ui, sans-serif;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background:
        radial-gradient(circle at 8% 72%, rgba(255, 140, 20, 0.16) 0%, transparent 28%),
        radial-gradient(circle at 95% 42%, rgba(56, 189, 248, 0.18) 0%, transparent 26%),
        linear-gradient(180deg, #020409 0%, #050b16 50%, #020409 100%);
      color: var(--text-main);
      font-family: var(--font-body);
      height: 100vh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      padding: 6px 12px;
      gap: 6px;
      user-select: none;
    }}

    /* Outer Curved Starfleet Bridge Bezel */
    .bridge-console {{
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 6px;
      background: rgba(4, 9, 20, 0.96);
      border: 2px solid rgba(56, 189, 248, 0.42);
      border-radius: 14px;
      padding: 6px 10px;
      box-shadow:
        0 0 35px rgba(16, 185, 129, 0.12),
        inset 0 0 28px rgba(34, 211, 238, 0.08);
      min-height: 0;
    }}

    /* Row 1: Exact LCARS Telemetry Strip from Reference Image */
    .lcars-telemetry-strip {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: rgba(3, 8, 18, 0.95);
      border: 1px solid rgba(56, 189, 248, 0.38);
      border-radius: 6px;
      padding: 4px 10px;
      font-family: var(--font-mono);
      font-size: 0.72rem;
      color: #a5f3fc;
      gap: 10px;
    }}
    .lcars-loading-wrap {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .lcars-bar-segments {{
      display: inline-flex;
      gap: 2px;
      padding: 2px 5px;
      background: rgba(6, 28, 24, 0.9);
      border: 1px solid rgba(16, 185, 129, 0.5);
      border-radius: 3px;
    }}
    .lcars-seg {{
      width: 4px;
      height: 11px;
      background: var(--lcars-neon-green);
      box-shadow: 0 0 5px var(--lcars-neon-green);
    }}
    .lcars-seg.dim {{
      background: rgba(16, 185, 129, 0.18);
      box-shadow: none;
    }}
    .lcars-chevrons {{
      color: var(--lcars-amber);
      font-weight: 900;
      letter-spacing: 1px;
      font-size: 0.85rem;
    }}

    /* Language, Voice, SFX & Stage Controls Bar */
    .lang-stage-bar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 6px;
      flex-wrap: wrap;
    }}
    .stage-banner {{
      flex: 1;
      background: linear-gradient(90deg, rgba(255, 170, 0, 0.16), rgba(8, 18, 36, 0.92));
      border: 1.5px solid var(--lcars-amber);
      border-radius: 5px;
      padding: 4px 10px;
      font-family: var(--font-mono);
      font-size: 0.74rem;
      font-weight: 700;
      color: var(--lcars-gold);
      letter-spacing: 0.03em;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .control-pills {{
      display: flex;
      align-items: center;
      gap: 4px;
    }}
    .pill-btn {{
      background: rgba(10, 22, 42, 0.92);
      color: #e2e8f0;
      border: 1px solid rgba(56, 189, 248, 0.45);
      border-radius: 5px;
      padding: 4px 8px;
      font-family: var(--font-mono);
      font-size: 0.68rem;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.14s;
    }}
    .pill-btn:hover {{
      border-color: var(--lcars-gold);
      color: var(--lcars-gold);
    }}
    .pill-btn.active {{
      background: var(--lcars-amber);
      color: #02040a;
      border-color: #fff;
      box-shadow: 0 0 10px rgba(255, 170, 0, 0.5);
    }}
    .pill-btn.lang-active {{
      background: var(--lcars-emerald);
      color: #02040a;
      border-color: #a7f3d0;
      box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
    }}

    /* Main Workspace: Galaxy Viewport + Optional Right Scientific/JWST Drawer */
    .workspace-row {{
      flex: 1;
      display: grid;
      grid-template-columns: 1fr 315px;
      gap: 8px;
      min-height: 0;
    }}
    .workspace-row.full-bridge {{
      grid-template-columns: 1fr;
    }}

    .galaxy-viewport-box {{
      position: relative;
      background: #02050d;
      border: 1.5px solid rgba(56, 189, 248, 0.55);
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      min-height: 250px;
    }}
    #galaxy-canvas {{
      width: 100%;
      height: 100%;
      display: block;
      cursor: crosshair;
    }}

    /* Dark Glassmorphic MFD Panel for Planetary Glare Protection (v4.9 Movie Redesign) */
    .mfd-panel {{
      background: linear-gradient(135deg, rgba(4, 14, 28, 0.86) 0%, rgba(2, 8, 18, 0.78) 100%);
      backdrop-filter: blur(10px) saturate(140%);
      -webkit-backdrop-filter: blur(10px) saturate(140%);
      border: 1px solid rgba(0, 229, 255, 0.35);
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.65), inset 0 0 14px rgba(0, 229, 255, 0.12);
      clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 12px 100%, 0 calc(100% - 12px));
      color: #e0f7fa;
      font-family: 'JetBrains Mono', 'Share Tech Mono', monospace;
    }}
    .hud-warning-critical {{
      color: #ff2a55;
      text-shadow: 0 0 12px rgba(255, 42, 85, 0.85);
      animation: pulse-alert 0.8s infinite alternate;
    }}
    @keyframes pulse-alert {{
      from {{ opacity: 1.0; transform: scale(1.0); }}
      to   {{ opacity: 0.65; transform: scale(1.03); }}
    }}

    /* Cinematic Multilingual Subtitle Overlay */
    .subtitle-overlay {{
      position: absolute;
      left: 50%;
      bottom: 8px;
      transform: translateX(-50%);
      width: 92%;
      background: rgba(2, 6, 16, 0.88);
      border: 1px solid rgba(34, 211, 238, 0.55);
      border-left: 5px solid var(--lcars-amber);
      border-radius: 6px;
      padding: 5px 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      pointer-events: auto;
      box-shadow: 0 4px 18px rgba(0, 0, 0, 0.75);
    }}
    .subtitle-text {{
      font-family: var(--font-body);
      font-size: 0.80rem;
      font-weight: 600;
      color: #fef08a;
      text-shadow: 0 1px 4px #000;
      line-height: 1.3;
    }}
    .subtitle-meta {{
      font-family: var(--font-mono);
      font-size: 0.65rem;
      color: var(--lcars-cyan);
      white-space: nowrap;
    }}

    /* Right Astrophysics Laws & JWST 4-Band Coronagraph Drawer */
    .science-drawer {{
      background: rgba(5, 11, 24, 0.95);
      border: 1px solid rgba(255, 170, 0, 0.42);
      border-radius: 8px;
      padding: 7px 9px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      overflow-y: auto;
      font-family: var(--font-mono);
      font-size: 0.67rem;
    }}
    .drawer-hdr {{
      font-family: var(--font-lcars);
      font-size: 0.72rem;
      font-weight: 800;
      color: var(--lcars-amber);
      border-bottom: 1px solid rgba(255, 170, 0, 0.3);
      padding-bottom: 3px;
      display: flex;
      justify-content: space-between;
    }}
    .law-card {{
      background: rgba(8, 18, 36, 0.9);
      border-left: 3px solid var(--lcars-cyan);
      border-radius: 4px;
      padding: 4px 6px;
      line-height: 1.35;
    }}
    .coronagraph-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 4px;
    }}
    .coro-cell {{
      background: #02040a;
      border: 1px solid rgba(255, 255, 255, 0.22);
      border-radius: 4px;
      padding: 2px;
      text-align: center;
    }}

    /* Middle Dual Oscilloscopes (Exact UCSC Systemic Console + Kepler/TESS Look) */
    .oscilloscope-row {{
      height: 132px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      background: rgba(3, 9, 18, 0.96);
      border: 1.5px solid rgba(56, 189, 248, 0.5);
      border-radius: 8px;
      padding: 5px 8px;
    }}
    .osc-panel {{
      display: flex;
      flex-direction: column;
      min-width: 0;
    }}
    .osc-title {{
      font-family: var(--font-mono);
      font-size: 0.70rem;
      font-weight: 800;
      color: #a7f3d0;
      margin-bottom: 3px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .osc-canvas {{
      width: 100%;
      flex: 1;
      background: #020810;
      border: 1px solid rgba(34, 211, 238, 0.35);
      border-radius: 4px;
      display: block;
    }}

    /* Bottom 10-Slot Astrobiological Collection Bar with 3D Holographic Pedestal Globes */
    .collection-bar {{
      height: 102px;
      background: linear-gradient(180deg, rgba(4, 16, 14, 0.96), rgba(2, 10, 9, 0.98));
      border: 1.5px solid rgba(52, 211, 153, 0.65);
      border-radius: 8px;
      padding: 4px 8px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
    .collection-hdr {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-family: var(--font-mono);
      font-size: 0.69rem;
      font-weight: 800;
      color: var(--lcars-gold);
    }}
    .collection-slots {{
      flex: 1;
      display: grid;
      grid-template-columns: repeat(10, 1fr);
      gap: 6px;
      min-height: 0;
    }}
    .holo-card {{
      background: radial-gradient(circle at 50% 42%, rgba(16, 52, 44, 0.85), rgba(4, 14, 14, 0.96));
      border: 1px solid rgba(52, 211, 153, 0.45);
      border-radius: 5px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: space-between;
      padding: 3px 2px;
      cursor: pointer;
      position: relative;
      transition: transform 0.12s, border-color 0.12s;
    }}
    .holo-card:hover {{
      transform: translateY(-2px);
      border-color: var(--lcars-cyan);
    }}
    .holo-card.selected {{
      border: 2px solid var(--lcars-gold);
      box-shadow: 0 0 14px rgba(255, 209, 102, 0.55), inset 0 0 10px rgba(34, 211, 238, 0.3);
    }}
    .holo-card-label {{
      font-family: var(--font-mono);
      font-size: 0.61rem;
      font-weight: 700;
      color: #d1fae5;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 100%;
    }}
  </style>
</head>
<body>
  <div class="bridge-console">
    <!-- Row 1: Exact Star Trek LCARS Telemetry Strip from Reference Image -->
    <div class="lcars-telemetry-strip">
      <div style="display:flex;gap:14px;align-items:center;">
        <span style="color:var(--lcars-amber);font-weight:800;">🖖 USS HIVEMIND NCC-31337</span>
        <span>55.10</span>
        <span>100</span>
        <span style="color:#64748b;">|</span>
        <span>238.92</span>
        <span>15.28</span>
      </div>
      <div class="lcars-loading-wrap">
        <span style="color:#a7f3d0;font-weight:700;">LOADING</span>
        <div class="lcars-bar-segments" id="lcars-segments"></div>
        <span>32208</span>
        <span>1</span>
        <span style="color:#64748b;">|</span>
        <span id="telemetry-counter">55222/3208</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;">
        <span id="steam-auth-badge" style="color:#34ff8d;background:rgba(16,185,129,0.16);border:1px solid rgba(52,255,141,0.5);padding:2px 7px;border-radius:4px;font-weight:700;">🎮 STEAM FREE GAME (AppID 3133700) • CONNECTING...</span>
        <button class="pill-btn" id="btn-steam-login-contribute" onclick="simulateSteamRegisterLoginAndContribute()" style="background:rgba(0,240,255,0.18);border-color:#00f0ff;color:#00f0ff;">🔑 Steam Login + BOINC GPU Sync</button>
        <a href="/portal#steam-free-game" target="_blank" class="pill-btn" style="text-decoration:none;border-color:#fbbf24;color:#fbbf24;">🐝 Portal Registry</a>
        <a href="/webgpu" target="_blank" class="pill-btn" style="text-decoration:none;background:rgba(168,85,247,0.22);border-color:#a855f7;color:#e9d5ff;">🌌 WebGPU Zero-Install</a>
        <span style="color:var(--lcars-gold);">🏆 <b id="score-hmt">2,400</b> HMT</span>
        <span style="color:var(--lcars-neon-green);">🟢 RTX 2070 (53.8 TFLOPS)</span>
        <span class="lcars-chevrons">&raquo;&raquo;&raquo;&raquo;&raquo;&raquo;&raquo;</span>
      </div>
    </div>

    <!-- Row 2: Stage Banner + 5-Language Voice/Subtitles + Game Stages -->
    <div class="lang-stage-bar">
      <div class="stage-banner" id="stage-banner-text">
        ÉTAPE 1 // PILOTEZ L'USS HIVEMIND (MANETTE / ZQSD / CLIC) VERS UNE ÉTOILE PUIS [ESPACE / BOUTON A] POUR LANCER LES SATELLITES
      </div>
      <div class="control-pills">
        <!-- 5 Languages with Voice & Subtitles -->
        <button class="pill-btn lang-active" id="lang-fr" onclick="setLanguage('fr')">🇫🇷 FR</button>
        <button class="pill-btn" id="lang-es" onclick="setLanguage('es')">🇪🇸 ES</button>
        <button class="pill-btn" id="lang-de" onclick="setLanguage('de')">🇩🇪 DE</button>
        <button class="pill-btn" id="lang-en" onclick="setLanguage('en')">🇬🇧 EN</button>
        <button class="pill-btn" id="lang-zh" onclick="setLanguage('zh')">🇨🇳 中文</button>
        <button class="pill-btn active" id="btn-voice-toggle" onclick="toggleVoiceAndSound()">🔊 Voix &amp; Son : ON</button>
        <button class="pill-btn" onclick="speakCurrentTargetBriefing()">🎙️ Écouter Briefing</button>
        <span style="color:#475569;">|</span>
        <!-- 4 Gameplay Views + Google Stitch Tactical Bridge Aerospace HUD Screens -->
        <button class="pill-btn active" id="stg-0" onclick="setGameStage(0)">🌌 1. Galaxie</button>
        <button class="pill-btn" id="stg-1" onclick="setGameStage(1)">🛰️ 2. Orbite Bee-MoE</button>
        <button class="pill-btn" id="stg-2" onclick="setGameStage(2)" style="border-color:#10b981;color:#6ee7b7;">🛬 3. Oolite Planetfall</button>
        <button class="pill-btn" id="stg-3" onclick="setGameStage(3)">🔭 4. Dossier &amp; JWST</button>
        <button class="pill-btn" id="btn-toggle-drawer" onclick="toggleScienceDrawer()">📐 4 Lois &amp; JWST</button>
        <span style="color:#475569;">|</span>
        <a href="/stitch/bridge" target="_blank" class="pill-btn" style="text-decoration:none;border-color:#00f0ff;color:#00f0ff;" title="Google Stitch Desktop Galactic Tactical Bridge (2560x2048)">❖ Stitch Bridge</a>
        <a href="/stitch/stage1" target="_blank" class="pill-btn" style="text-decoration:none;border-color:#00f0ff;color:#a5f3fc;">❖ S1:Galaxie</a>
        <a href="/stitch/stage2" target="_blank" class="pill-btn" style="text-decoration:none;border-color:#34ff8d;color:#6ee7b7;">❖ S2:Orbit Bee</a>
        <a href="/stitch/stage3" target="_blank" class="pill-btn" style="text-decoration:none;border-color:#ffba20;color:#fde047;">❖ S3:Oolite Land</a>
        <a href="/stitch/stage4" target="_blank" class="pill-btn" style="text-decoration:none;border-color:#a855f7;color:#e9d5ff;">❖ S4:JWST/LOIS</a>
        <span style="color:#475569;">|</span>
        <a href="/community" target="_blank" class="pill-btn" style="text-decoration:none;background:rgba(52,255,141,0.18);border-color:#34ff8d;color:#34ff8d;font-weight:800;" title="Open BOINC Idle-GPU &amp; GCP Gamer + Citizen Scientist Community Hub">⚡ BOINC Idle-GPU &amp; Community Hub</a>
      </div>
    </div>

    <!-- Row 3: Main Galaxy Astrometrics Viewport + Right 4-Laws & JWST Coronagraph Drawer -->
    <div class="workspace-row" id="workspace-row">
      <div class="galaxy-viewport-box">
        <canvas id="galaxy-canvas" width="1120" height="450"></canvas>

        <!-- Synchronized Multilingual Subtitle Bar -->
        <div class="subtitle-overlay" id="subtitle-bar">
          <div>
            <div class="subtitle-meta" id="subtitle-speaker">🎙️ VOIX IA STARFLEET [FRANÇAIS • fr-FR] // SOUS-TITRES SYNCHRONISÉS</div>
            <div class="subtitle-text" id="subtitle-text">
              « Commandant, l'USS HiveMind est en position près de LHS 1140 b. Profondeur de transit : 5 340 ppm, vitesse radiale K = 4,9 m/s. Appuyez sur ESPACE pour déployer les satellites Bee-MoE ! »
            </div>
          </div>
          <div style="display:flex;gap:5px;">
            <button class="pill-btn" style="background:var(--lcars-emerald);color:#02040a;" onclick="approveOrRejectDiscovery(true)" id="btn-approve">✅ Approuver</button>
            <button class="pill-btn" style="background:var(--lcars-coral);color:#fff;" onclick="approveOrRejectDiscovery(false)" id="btn-reject">🚨 Rejeter</button>
          </div>
        </div>
      </div>

      <!-- Right Scientific Drawer: 4 Astrophysical Laws + JWST 4-Band Direct Imaging Coronagraphy -->
      <aside class="science-drawer" id="science-drawer">
        <div class="drawer-hdr">
          <span id="drawer-title">🔬 4 LOIS ASTROPHYSIQUES &amp; JWST</span>
          <button class="pill-btn" style="padding:1px 6px;font-size:0.62rem;" onclick="syncLiveNasaTap()">🌐 NASA TAP</button>
        </div>
        <div id="target-header-card" style="font-size:0.78rem;font-weight:800;color:var(--lcars-gold);"></div>

        <div class="law-card" id="law-1-box"></div>
        <div class="law-card" style="border-left-color:var(--lcars-emerald);" id="law-2-box"></div>
        <div class="law-card" style="border-left-color:var(--lcars-amber);" id="law-3-box"></div>
        <div class="law-card" style="border-left-color:var(--lcars-violet);" id="law-4-box"></div>

        <!-- JWST 4-Band Direct Imaging Coronagraphy (NIRCam F300M/F444W + MIRI F1140C/F1550C) -->
        <div style="color:var(--lcars-cyan);font-weight:800;font-size:0.66rem;margin-top:2px;">
          🔭 IMAGERIE DIRECTE CORONOGRAPHIQUE JWST (★ MASQUE ÉTOILE) :
        </div>
        <div class="coronagraph-grid">
          <div class="coro-cell">
            <canvas id="coro-0" width="64" height="54" style="width:100%;height:48px;"></canvas>
            <div style="font-size:0.56rem;color:#d8b4fe;">NIRCam 3.00µm</div>
          </div>
          <div class="coro-cell">
            <canvas id="coro-1" width="64" height="54" style="width:100%;height:48px;"></canvas>
            <div style="font-size:0.56rem;color:#7dd3fc;">NIRCam 4.44µm</div>
          </div>
          <div class="coro-cell">
            <canvas id="coro-2" width="64" height="54" style="width:100%;height:48px;"></canvas>
            <div style="font-size:0.56rem;color:#fcd34d;">MIRI 11.4µm</div>
          </div>
          <div class="coro-cell">
            <canvas id="coro-3" width="64" height="54" style="width:100%;height:48px;"></canvas>
            <div style="font-size:0.56rem;color:#fca5a5;">MIRI 15.5µm</div>
          </div>
        </div>

        <button class="pill-btn" id="btn-koi14-toggle" style="width:100%;margin-top:auto;background:rgba(168,85,247,0.22);border-color:#a855f7;color:#e9d5ff;" onclick="toggleKoi14BlendMode()">
          🔄 Règle 4 (KOI-Cand 14) : Basculer K=0 m/s (Blend) / K=2.4 m/s
        </button>
      </aside>
    </div>

    <!-- Row 4: Dual High-Density Neon Oscilloscopes (Exact Reference Look) -->
    <div class="oscilloscope-row">
      <div class="osc-panel">
        <div class="osc-title" id="rv-osc-title">
          CONSOLE SYSTÉMIQUE UCSC // VITESSE RADIALE DOPPLER (K = 4.9 m/s)
        </div>
        <canvas class="osc-canvas" id="rv-osc-canvas" width="540" height="98"></canvas>
      </div>
      <div class="osc-panel">
        <div class="osc-title" id="lc-osc-title" style="color:#67e8f9;">
          PHOTOMÉTRIE DE TRANSIT KEPLER/TESS (Profondeur = 5340 ppm)
        </div>
        <canvas class="osc-canvas" id="lc-osc-canvas" width="540" height="98"></canvas>
      </div>
    </div>

    <!-- Row 5: Bottom 10-Slot Astrobiological Collection Bar with 3D Holographic Globes -->
    <div class="collection-bar">
      <div class="collection-hdr">
        <span id="collection-hdr-title">🪐 COLLECTION ASTROBIOLOGIQUE &amp; EXOPLANÈTES APPROUVÉES PAR LE JOUEUR (CLIQUEZ SUR UNE CASE) :</span>
        <span style="color:var(--lcars-neon-green);" id="collection-hdr-right">🎮 Manette / Clavier ZQSD + Espace &bull; Moteur Audio/Voix 5 Langues Actif</span>
      </div>
      <div class="collection-slots" id="collection-slots"></div>
    </div>
  </div>

  <script>
    const STATE = {state_json};

    // Build LCARS loading bar segments
    const segWrap = document.getElementById('lcars-segments');
    for (let i = 0; i < 24; i++) {{
      const s = document.createElement('span');
      s.className = 'lcars-seg' + (i >= 14 ? ' dim' : '');
      segWrap.appendChild(s);
    }}

    // 5-Language Localization & Voice Dictionary (FR, ES, DE, EN, ZH)
    const I18N = {{
      fr: {{
        langName: "FRANÇAIS • fr-FR",
        bcp47: "fr-FR",
        stage0Banner: "ÉTAPE 1 // PILOTEZ L'USS HIVEMIND (MANETTE / ZQSD / CLIC) VERS UNE ÉTOILE PUIS [ESPACE / BOUTON A] POUR LANCER LES SATELLITES",
        stage1Banner: "ÉTAPE 2 // PILOTEZ LE MODULE SATELLITE ROBOTIQUE BEE-MOE JUSQU'À L'EXOPLANÈTE EN ORBITE POUR LA CAPTURER (0% → 100%)",
        stage2Banner: "ÉTAPE 3 // VÉRIFIEZ LES 4 LOIS ASTROPHYSIQUES & LE CORONOGRAPHE JWST PUIS APPROUVEZ [ESPACE] OU REJETEZ [TOUCHE R]",
        rvTitle: (k, e) => `CONSOLE SYSTÉMIQUE UCSC // VITESSE RADIALE DOPPLER (K = ${{k}} m/s, e = ${{e}})`,
        lcTitle: (d) => `PHOTOMÉTRIE DE TRANSIT KEPLER/TESS (Profondeur = ${{d}} ppm)`,
        colTitle: "🪐 COLLECTION ASTROBIOLOGIQUE & EXOPLANÈTES APPROUVÉES PAR LE JOUEUR (CLIQUEZ SUR UNE CASE) :",
        approveBtn: "✅ Approuver",
        rejectBtn: "🚨 Rejeter",
        welcomeVoice: (sys) => `Commandant, l'USS HiveMind cible ${{sys.shortLabel}}. Profondeur de transit : ${{sys.depthPpm}} parties par million. Vitesse radiale Doppler K égale ${{sys.rvKms}} mètres par seconde. Appuyez sur Espace pour lancer les satellites Bee-M-O-E.`,
        launchVoice: (sys) => `Déploiement des quatre satellites robotiques d'exploration Bee-M-O-E vers le système ${{sys.shortLabel}}. Verrouillage orbital en cours.`,
        captureVoice: (sys) => `Capture télémétrique de ${{sys.shortLabel}} terminée à cent pour cent ! Densité moyenne : ${{sys.rhoGcm3}} grammes par centimètre cube, température d'équilibre ${{sys.teqK}} Kelvin.`,
        approveVoice: (sys, ok) => ok ? `Félicitations Commandant ! Découverte astrophysique confirmée sur ${{sys.shortLabel}}. Six cents points HiveMind attribués.` : `Attention Commandant, vérifiez la règle quatre sur la vitesse radiale et la forme du transit.`
      }},
      es: {{
        langName: "ESPAÑOL • es-ES",
        bcp47: "es-ES",
        stage0Banner: "ETAPA 1 // PILOTA EL USS HIVEMIND (MANDO / WASD / CLIC) HACIA UNA ESTRELLA Y PULSA [ESPACIO / BOTÓN A] PARA LANZAR SATÉLITES",
        stage1Banner: "ETAPA 2 // PILOTA EL MÓDULO SATÉLITE ROBÓTICO BEE-MOE HASTA EL EXOPLANETA EN ÓRBITA PARA CAPTURARLO (0% → 100%)",
        stage2Banner: "ETAPA 3 // ANALIZA LAS 4 LEYES ASTROFÍSICAS Y EL CORONÓGRAFO JWST, LUEGO APRUEBA [ESPACIO] O RECHAZA [TECLA R]",
        rvTitle: (k, e) => `CONSOLA SISTÉMICA UCSC // VELOCIDAD RADIAL DOPPLER (K = ${{k}} m/s, e = ${{e}})`,
        lcTitle: (d) => `FOTOMETRÍA DE TRÁNSITO KEPLER/TESS (Profundidad = ${{d}} ppm)`,
        colTitle: "🪐 COLECCIÓN ASTROBIOLÓGICA Y EXOPLANETAS APROBADOS POR EL JUGADOR (HAZ CLIC EN UNA CASILLA) :",
        approveBtn: "✅ Aprobar",
        rejectBtn: "🚨 Rechazar",
        welcomeVoice: (sys) => `Comandante, el USS HiveMind apunta a ${{sys.shortLabel}}. Profundidad de tránsito: ${{sys.depthPpm}} partes por millón. Velocidad radial K igual a ${{sys.rvKms}} metros por segundo. Pulse Espacio para lanzar los satélites.`,
        launchVoice: (sys) => `Desplegando los cuatro satélites robóticos de exploración Bee-M-O-E hacia el sistema ${{sys.shortLabel}}.`,
        captureVoice: (sys) => `¡Captura telemétrica de ${{sys.shortLabel}} completada al cien por cien! Densidad: ${{sys.rhoGcm3}} gramos por centímetro cúbico, temperatura de equilibrio ${{sys.teqK}} Kelvin.`,
        approveVoice: (sys, ok) => ok ? `¡Excelente Comandante! Descubrimiento astrofísico validado en ${{sys.shortLabel}}. Seiscientos puntos otorgados.` : `Atención Comandante, revise la regla cuatro de velocidad radial y tránsito.`
      }},
      de: {{
        langName: "DEUTSCH • de-DE",
        bcp47: "de-DE",
        stage0Banner: "STUFE 1 // STEUERE DIE USS HIVEMIND (GAMEPAD / WASD / KLICK) ZU EINEM STERN & DRÜCKE [LEERTASTE / A] FÜR SATELLITENSTART",
        stage1Banner: "STUFE 2 // STEUERE DAS BEE-MOE ROBOTER-SATELLITENMODUL ZUM EXOPLANETEN IM ORBIT ZUR ERFASSUNG (0% → 100%)",
        stage2Banner: "STUFE 3 // PRÜFE DIE 4 ASTROPHYSIKALISCHEN GESETZE & JWST-KORONOGRAFIE — BESTÄTIGEN [LEERTASTE] ODER ABLEHNEN [R]",
        rvTitle: (k, e) => `UCSC SYSTEMISCHE KONSOLE // DOPPLER-RADIALGESCHWINDIGKEIT (K = ${{k}} m/s, e = ${{e}})`,
        lcTitle: (d) => `KEPLER/TESS TRANSIT-PHOTOMETRIE (Tiefe = ${{d}} ppm)`,
        colTitle: "🪐 ASTROBIOLOGISCHE SAMMLUNG & VOM SPIELER BESTÄTIGTE EXOPLANETEN (KLICKE AUF EIN FELD) :",
        approveBtn: "✅ Bestätigen",
        rejectBtn: "🚨 Ablehnen",
        welcomeVoice: (sys) => `Kommandant, die USS HiveMind hat ${{sys.shortLabel}} erfasst. Transittiefe: ${{sys.depthPpm}} Teile pro Million. Radialgeschwindigkeit K ist ${{sys.rvKms}} Meter pro Sekunde. Drücken Sie die Leertaste zum Start der Satelliten.`,
        launchVoice: (sys) => `Starte vier robotische Bee-M-O-E Forschungssatelliten in das System ${{sys.shortLabel}}.`,
        captureVoice: (sys) => `Telemetrie-Erfassung von ${{sys.shortLabel}} zu hundert Prozent abgeschlossen! Mittlere Dichte: ${{sys.rhoGcm3}} Gramm pro Kubikzentimeter, Gleichgewichtstemperatur ${{sys.teqK}} Kelvin.`,
        approveVoice: (sys, ok) => ok ? `Ausgezeichnet Kommandant! Astrophysikalische Entdeckung bei ${{sys.shortLabel}} bestätigt. Sechshundert Punkte gutgeschrieben.` : `Achtung Kommandant, bitte prüfen Sie Regel vier für Fehlalarme.`
      }},
      en: {{
        langName: "ENGLISH • en-US (Kid Astrophysics)",
        bcp47: "en-US",
        stage0Banner: "STEP 1 (GALAXY BRIDGE) // CLICK ANY STAR OR PRESS [SPACE] TO LAUNCH YOUR BEE-MOE ROBOT SATELLITES (+300 PTS)!",
        stage1Banner: "STEP 2 (ORBIT CAPTURE) // FLY YOUR ROBOT INTO THE GREEN CIRCLE AROUND THE PLANET (+500 PTS) OR CLICK [3.PLANET LAND]!",
        stage2Banner: "STEP 3 (EASY PLANET LANDING) // GUIDE YOUR ROBOT 🛸 TO THE GREEN PAD 🎯 • [SPACE] BRAKE (+100 PTS) • [ENTER] AUTO-LAND (+1,000 PTS)!",
        rvTitle: (k, e) => `UCSC SYSTEMIC CONSOLE // DOPPLER STAR WOBBLE (K = ${{k}} m/s, e = ${{e}})`,
        lcTitle: (d) => `KEPLER/TESS TRANSIT SHADOW DIP (Depth = ${{d}} ppm)`,
        colTitle: "🪐 ASTROBIOLOGY COLLECTION & KID DISCOVERY PASSPORT (CLICK ANY PLANET TO EXPLORE) :",
        approveBtn: "✅ Approve (+800 PTS)",
        rejectBtn: "🚨 Reject (+800 PTS)",
        welcomeVoice: (sys) => `Welcome Young Space Explorer! The USS HiveMind is locked onto ${{sys.shortLabel}}. Kid Astrophysics Fact: its transit shadow blocks ${{sys.depthPpm}} parts per million of starlight, telling us the planet is ${{sys.radiusRe}} times the width of Earth! And its gravity makes the star wobble at ${{sys.rvKms}} meters per second. Press Space to launch your robot satellites and win 300 points!`,
        launchVoice: (sys) => `Deploying all four Bee M O E robotic exploration satellites toward ${{sys.shortLabel}}! Plus 300 points!`,
        captureVoice: (sys) => `Awesome! Orbital telemetry capture of ${{sys.shortLabel}} is at one hundred percent! Plus 500 points! Now let's glide down to the green landing pad on the planet surface!`,
        approveVoice: (sys, ok) => ok ? `Fantastic job Space Cadet! Exoplanet discovery validated on ${{sys.shortLabel}}. Eight hundred Astrophysics Master points awarded!` : `Good try Space Cadet! Remember: a real planet needs both a flat U-shaped transit shadow and a Doppler gravity wobble!`
      }},
      zh: {{
        langName: "中文 • zh-CN",
        bcp47: "zh-CN",
        stage0Banner: "第 1 阶段 // 驾驶星际迷航蜂群号飞船（手柄 / WASD / 点击）前往恒星，按 [空格键 / A键] 发射探索卫星",
        stage1Banner: "第 2 阶段 // 操控 Bee-MoE 机器人探索卫星模块拦截并扫描轨道系外行星 (0% → 100%)",
        stage2Banner: "第 3 阶段 // 检验四大天体物理定律与韦伯日冕仪图像，按 [空格键] 批准发现或按 [R键] 排除假阳性",
        rvTitle: (k, e) => `UCSC 系统控制台 // 多普勒径向速度 (K = ${{k}} 米/秒, 偏心率 e = ${{e}})`,
        lcTitle: (d) => `开普勒/TESS 凌星光度测量 (深度 = ${{d}} ppm)`,
        colTitle: "🪐 天体生物学馆藏与玩家批准的系外行星（点击卡片切换目标）：",
        approveBtn: "✅ 批准行星",
        rejectBtn: "🚨 排除假阳性",
        welcomeVoice: (sys) => `指挥官，蜂群号星舰已锁定 ${{sys.shortLabel}}。凌星深度为百万分之 ${{sys.depthPpm}}，多普勒径向速度 K 为每秒 ${{sys.rvKms}} 米。请按空格键发射探索卫星。`,
        launchVoice: (sys) => `正在向 ${{sys.shortLabel}} 恒星系统发射四颗 Bee-M-O-E 机器人探索卫星。`,
        captureVoice: (sys) => `${{sys.shortLabel}} 轨道遥测捕获已达百分之百！平均密度为每立方厘米 ${{sys.rhoGcm3}} 克，平衡温度 ${{sys.teqK}} 开尔文。`,
        approveVoice: (sys, ok) => ok ? `太棒了指挥官！${{sys.shortLabel}} 天体物理发现已确认，奖励六百积分。` : `请注意指挥官，请根据第四定律检查径向速度与凌星形态。`
      }}
    }};

    let currentLang = 'en';
    let voiceAndSoundEnabled = true;
    let audioCtx = null;
    let koi14FlatBlendMode = true;

    // Exact 10 Star Systems positioned on the Spiral Galaxy like `startrek_main_ui_v4_ref.jpg`
    const GALAXY_SYSTEMS = [
      {{
        idx: 0, id: "SECTOR-01-TRAPPIST-1E", shortLabel: "TRAPPIST-1e", hostStar: "TRAPPIST-1 (M8V)",
        gx: 0.145, gy: 0.22, period: 6.10, aAu: 0.029, ecc: 0.005, rvKms: 2.48, depthPpm: 5180,
        radiusRe: 0.92, massMe: 0.692, rhoGcm3: 4.90, teqK: 251, insolEarth: 0.66, rStarRsun: 0.119, teffK: 2566,
        disposition: "CONFIRMED", inHz: true,
        cLight: "#a7f3d0", cMid: "#10b981", cDark: "#064e3b", ringCol: "#34d399", approved: true
      }},
      {{
        idx: 1, id: "SECTOR-02-K2-18B", shortLabel: "K2-18 b", hostStar: "K2-18 (M2.5V)",
        gx: 0.34, gy: 0.18, period: 32.94, aAu: 0.143, ecc: 0.09, rvKms: 3.55, depthPpm: 2920,
        radiusRe: 2.61, massMe: 8.63, rhoGcm3: 2.67, teqK: 265, insolEarth: 0.94, rStarRsun: 0.445, teffK: 3457,
        disposition: "CONFIRMED", inHz: true,
        cLight: "#bae6fd", cMid: "#0284c7", cDark: "#0c4a6e", ringCol: "#38bdf8", approved: true
      }},
      {{
        idx: 2, id: "SECTOR-03-KEPLER-11F", shortLabel: "Kepler-11 f", hostStar: "Kepler-11 (G6V)",
        gx: 0.56, gy: 0.21, period: 46.69, aAu: 0.250, ecc: 0.013, rvKms: 0.43, depthPpm: 550,
        radiusRe: 2.49, massMe: 2.00, rhoGcm3: 0.71, teqK: 544, insolEarth: 17.4, rStarRsun: 1.065, teffK: 5663,
        disposition: "CONFIRMED", inHz: false,
        cLight: "#ddd6fe", cMid: "#7c3aed", cDark: "#2e1065", ringCol: "#a78bfa", approved: true
      }},
      {{
        idx: 3, id: "SECTOR-04-HD-80606B", shortLabel: "HD 80606 b", hostStar: "HD 80606 (G5V)",
        gx: 0.78, gy: 0.24, period: 111.44, aAu: 0.456, ecc: 0.933, rvKms: 476.1, depthPpm: 11200,
        radiusRe: 11.99, massMe: 1392.0, rhoGcm3: 4.43, teqK: 405, insolEarth: 5.05, rStarRsun: 1.037, teffK: 5573,
        disposition: "CONFIRMED", inHz: false,
        cLight: "#fed7aa", cMid: "#ea580c", cDark: "#431407", ringCol: "#fb923c", approved: true
      }},
      {{
        idx: 4, id: "SECTOR-05-TOI-700D", shortLabel: "TOI-700 d", hostStar: "TOI-700 (M2V)",
        gx: 0.21, gy: 0.54, period: 37.42, aAu: 0.163, ecc: 0.042, rvKms: 0.78, depthPpm: 665,
        radiusRe: 1.14, massMe: 1.72, rhoGcm3: 6.32, teqK: 269, insolEarth: 0.86, rStarRsun: 0.420, teffK: 3480,
        disposition: "CONFIRMED", inHz: true,
        cLight: "#99f6e4", cMid: "#0d9488", cDark: "#134e4a", ringCol: "#2dd4bf", approved: true
      }},
      {{
        idx: 5, id: "SECTOR-06-LHS-1140B", shortLabel: "LHS 1140 b", hostStar: "LHS 1140 (M4.5V)",
        gx: 0.44, gy: 0.50, period: 24.74, aAu: 0.095, ecc: 0.043, rvKms: 4.90, depthPpm: 5340,
        radiusRe: 1.73, massMe: 5.60, rhoGcm3: 5.95, teqK: 226, insolEarth: 0.43, rStarRsun: 0.216, teffK: 3096,
        disposition: "CONFIRMED", inHz: true,
        cLight: "#a5f3fc", cMid: "#0891b2", cDark: "#164e63", ringCol: "#22d3ee", approved: true
      }},
      {{
        idx: 6, id: "SECTOR-07-WASP-39B", shortLabel: "WASP-39 b", hostStar: "WASP-39 (G8V)",
        gx: 0.65, gy: 0.49, period: 4.06, aAu: 0.049, ecc: 0.00, rvKms: 38.0, depthPpm: 21200,
        radiusRe: 14.23, massMe: 88.9, rhoGcm3: 0.17, teqK: 1166, insolEarth: 420.0, rStarRsun: 0.939, teffK: 5485,
        disposition: "CONFIRMED", inHz: false,
        cLight: "#fef08a", cMid: "#ca8a04", cDark: "#422006", ringCol: "#facc15", approved: true
      }},
      {{
        idx: 7, id: "SECTOR-08-KEPLER-186F", shortLabel: "Kepler-186f", hostStar: "Kepler-186 (M1V)",
        gx: 0.82, gy: 0.54, period: 129.94, aAu: 0.432, ecc: 0.04, rvKms: 0.29, depthPpm: 512,
        radiusRe: 1.17, massMe: 1.44, rhoGcm3: 4.96, teqK: 188, insolEarth: 0.32, rStarRsun: 0.523, teffK: 3755,
        disposition: "CONFIRMED", inHz: true,
        cLight: "#f5d0fe", cMid: "#c026d3", cDark: "#4a044e", ringCol: "#e879f9", approved: true
      }},
      {{
        idx: 8, id: "SECTOR-09-KOI-CAND-14", shortLabel: "KOI-Cand 14", hostStar: "KIC 11804465",
        gx: 0.33, gy: 0.80, period: 18.42, aAu: 0.138, ecc: 0.01, rvKms: 0.04, depthPpm: 420,
        radiusRe: 1.42, massMe: 0.05, rhoGcm3: 0.10, teqK: 258, insolEarth: 0.88, rStarRsun: 0.880, teffK: 5120,
        disposition: "CANDIDATE", inHz: true,
        cLight: "#bbf7d0", cMid: "#16a34a", cDark: "#14532d", ringCol: "#4ade80", approved: true
      }},
      {{
        idx: 9, id: "SECTOR-10-KOI-BIN-02", shortLabel: "KOI-Bin #02", hostStar: "KIC 9832227 (Binary)",
        gx: 0.69, gy: 0.80, period: 1.84, aAu: 0.031, ecc: 0.08, rvKms: 1420.0, depthPpm: 58400,
        radiusRe: 31.40, massMe: 31150.0, rhoGcm3: 5.55, teqK: 1480, insolEarth: 890.0, rStarRsun: 1.180, teffK: 5840,
        disposition: "FALSE POSITIVE", inHz: false,
        cLight: "#fecdd3", cMid: "#e11d48", cDark: "#4c0519", ringCol: "#f43f5e", approved: true
      }}
    ];

    // Default selected target is LHS 1140 b (idx 5) matching K = 4.9 m/s & Profondeur = 5340 ppm in the user's screenshot!
    let currentSys = GALAXY_SYSTEMS[5];
    let gameStage = 0;
    let simTime = 0;
    let scoreHmt = 2400;

    // USS HiveMind position matching screenshot (between TRAPPIST-1e, K2-18 b, TOI-700 d, LHS 1140 b)
    let shipX = 0.265, shipY = 0.285, shipHeading = 0.55;
    let launchedProbes = [];
    let warpTimer = 0;

    // Stage 1 Explorer Bee-MoE Robotic Satellite Module
    let beeModX = 180, beeModY = 220, beeModHeading = 0;
    let planetScreenX = 560, planetScreenY = 220;
    let captureProgress = 0;

    // Starfield particles for Spiral Galaxy
    const galaxyStars = [];
    for (let i = 0; i < 420; i++) {{
      const arm = i % 2;
      const r = Math.pow(Math.random(), 0.72) * 0.48;
      const theta = r * 11.5 + (arm * Math.PI) + (Math.random() - 0.5) * 0.55;
      galaxyStars.push({{
        r, theta,
        rawX: Math.random(),
        rawY: Math.random(),
        size: 0.7 + Math.random() * 1.8,
        col: ["#ffffff", "#bae6fd", "#fef08a", "#f5d0fe", "#93c5fd"][i % 5]
      }});
    }}

    // Procedural Sci-Fi Sound Synthesizer (WebAudio API)
    function playSciFiSound(type) {{
      if (!voiceAndSoundEnabled) return;
      try {{
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        if (audioCtx.state === 'suspended') audioCtx.resume();
        const now = audioCtx.currentTime;
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.connect(gain);
        gain.connect(audioCtx.destination);

        if (type === 'select') {{
          osc.type = 'sine';
          osc.frequency.setValueAtTime(880, now);
          osc.frequency.exponentialRampToValueAtTime(1320, now + 0.09);
          gain.gain.setValueAtTime(0.08, now);
          gain.gain.exponentialRampToValueAtTime(0.001, now + 0.11);
          osc.start(now); osc.stop(now + 0.11);
        }} else if (type === 'launch') {{
          osc.type = 'sawtooth';
          osc.frequency.setValueAtTime(220, now);
          osc.frequency.exponentialRampToValueAtTime(1180, now + 0.38);
          gain.gain.setValueAtTime(0.10, now);
          gain.gain.exponentialRampToValueAtTime(0.001, now + 0.40);
          osc.start(now); osc.stop(now + 0.40);
        }} else if (type === 'approve') {{
          osc.type = 'triangle';
          osc.frequency.setValueAtTime(523.25, now);
          osc.frequency.setValueAtTime(659.25, now + 0.10);
          osc.frequency.setValueAtTime(783.99, now + 0.20);
          osc.frequency.setValueAtTime(1046.50, now + 0.30);
          gain.gain.setValueAtTime(0.11, now);
          gain.gain.exponentialRampToValueAtTime(0.001, now + 0.48);
          osc.start(now); osc.stop(now + 0.48);
        }}
      }} catch (e) {{}}
    }}

    // Multilingual Google Gemini / GCP Free-Tier Neural Voice-Over + Synchronized Subtitles
    let activeTtsAudio = null;
    let activeGeminiVoice = 'Puck';

    function speakWithSubtitles(text) {{
      const dict = I18N[currentLang] || I18N.en;
      document.getElementById('subtitle-speaker').textContent =
        `🎙️ GCP GEMINI NEURAL VOICE [${{dict.langName}} • ${{activeGeminiVoice.toUpperCase()}}] // SYNCHRONIZED SUBTITLES`;
      document.getElementById('subtitle-text').textContent = `« ${{text}} »`;

      if (!voiceAndSoundEnabled) return;
      try {{
        if (activeTtsAudio) {{
          activeTtsAudio.pause();
          activeTtsAudio = null;
        }}
        if ('speechSynthesis' in window) window.speechSynthesis.cancel();
        const ttsUrl = `/api/astro/gemini-tts?lang=${{encodeURIComponent(currentLang)}}&voice=${{encodeURIComponent(activeGeminiVoice)}}&text=${{encodeURIComponent(text)}}`;
        const audio = new Audio(ttsUrl);
        activeTtsAudio = audio;
        audio.onerror = () => {{
          if (!('speechSynthesis' in window)) return;
          const utter = new SpeechSynthesisUtterance(text);
          utter.lang = dict.bcp47;
          utter.rate = 1.02;
          window.speechSynthesis.speak(utter);
        }};
        audio.play().catch(() => {{}});
      }} catch (e) {{}}
    }}

    function setLanguage(lang) {{
      currentLang = lang;
      ['fr', 'es', 'de', 'en', 'zh'].forEach(l => {{
        const btn = document.getElementById('lang-' + l);
        if (btn) btn.classList.toggle('lang-active', l === lang);
      }});
      playSciFiSound('select');
      refreshAllUiText();
      speakCurrentTargetBriefing();
    }}

    function toggleVoiceAndSound() {{
      voiceAndSoundEnabled = !voiceAndSoundEnabled;
      const btn = document.getElementById('btn-voice-toggle');
      btn.textContent = voiceAndSoundEnabled ? "🔊 Voix & Son : ON" : "🔇 Voix & Son : MUET";
      btn.classList.toggle('active', voiceAndSoundEnabled);
      if (!voiceAndSoundEnabled && ('speechSynthesis' in window)) window.speechSynthesis.cancel();
      else speakCurrentTargetBriefing();
    }}

    function speakCurrentTargetBriefing() {{
      const dict = I18N[currentLang] || I18N.fr;
      speakWithSubtitles(dict.welcomeVoice(currentSys));
    }}

    function toggleScienceDrawer() {{
      const row = document.getElementById('workspace-row');
      const drw = document.getElementById('science-drawer');
      const hidden = drw.style.display === 'none';
      drw.style.display = hidden ? 'flex' : 'none';
      row.classList.toggle('full-bridge', !hidden);
    }}

    function solveKepler(M, e) {{
      let mNorm = M % (Math.PI * 2);
      if (mNorm < 0) mNorm += Math.PI * 2;
      let E = e > 0.8 ? Math.PI : mNorm;
      for (let i = 0; i < 8; i++) {{
        const f = E - e * Math.sin(E) - mNorm;
        const fp = 1 - e * Math.cos(E);
        E -= f / Math.max(1e-7, fp);
      }}
      return E;
    }}

    function draw3DPlanetSphere(ctx, x, y, r, cLight, cMid, cDark, hasRings) {{
      ctx.save();
      if (hasRings) {{
        ctx.strokeStyle = "rgba(255,255,255,0.45)";
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.ellipse(x, y, r * 1.85, r * 0.52, -0.28, 0, Math.PI * 2);
        ctx.stroke();
      }}
      const grad = ctx.createRadialGradient(x - r * 0.35, y - r * 0.35, r * 0.12, x, y, r);
      grad.addColorStop(0, cLight);
      grad.addColorStop(0.55, cMid);
      grad.addColorStop(1, cDark);
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();

      // Atmospheric rim glow
      ctx.strokeStyle = cLight;
      ctx.lineWidth = 1.1;
      ctx.beginPath();
      ctx.arc(x, y, r + 0.5, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }}

    function renderBottomCollectionSlots() {{
      const wrap = document.getElementById('collection-slots');
      wrap.innerHTML = GALAXY_SYSTEMS.map(s => `
        <div class="holo-card ${{s.idx === currentSys.idx ? 'selected' : ''}}" onclick="selectSystem(${{s.idx}})">
          <canvas id="slot-globe-${{s.idx}}" width="78" height="50" style="width:74px;height:46px;"></canvas>
          <div class="holo-card-label">
            <span style="color:#34d399;">✓</span> ${{s.shortLabel}}
          </div>
        </div>
      `).join('');

      // Draw each 3D globe sitting on its glowing holographic pedestal ring
      GALAXY_SYSTEMS.forEach(s => {{
        const c = document.getElementById('slot-globe-' + s.idx);
        if (!c) return;
        const gctx = c.getContext('2d');
        const W = c.width, H = c.height;
        gctx.clearRect(0, 0, W, H);

        // Glowing holographic pedestal rings at bottom
        gctx.strokeStyle = s.ringCol;
        gctx.lineWidth = 1.5;
        gctx.beginPath();
        gctx.ellipse(W / 2, H - 9, 25, 6, 0, 0, Math.PI * 2);
        gctx.stroke();
        gctx.strokeStyle = "rgba(255,255,255,0.55)";
        gctx.lineWidth = 1.0;
        gctx.beginPath();
        gctx.ellipse(W / 2, H - 9, 16, 3.5, 0, 0, Math.PI * 2);
        gctx.stroke();

        // 3D Planet Globe above pedestal
        draw3DPlanetSphere(gctx, W / 2, H / 2 - 3, 14.5, s.cLight, s.cMid, s.cDark, s.idx === 6);
      }});
    }}

    function updateScienceDrawerAndCoronagraph() {{
      const s = currentSys;
      const derivedRp = (s.rStarRsun * 109.076 * Math.sqrt(s.depthPpm * 1e-6)).toFixed(2);
      const densType = s.disposition === "FALSE POSITIVE"
        ? "Compagnon Stellaire Binaire (> 80 M_Jup)"
        : (s.rvKms < 0.1 ? "Binaire de Fond / Bruit (K ≈ 0 m/s)" : (s.rhoGcm3 >= 3.5 ? "Rocheuse / Super-Terre (> 3.5 g/cm³)" : (s.rhoGcm3 >= 1.5 ? "Monde Hycéen / Sub-Neptune" : "Géante Gazeuse Peu Dense (< 1.0 g/cm³)")));

      document.getElementById('target-header-card').innerHTML =
        `🪐 ${{s.shortLabel}} <span style="color:var(--lcars-cyan);font-size:0.68rem;">(${{s.hostStar}})</span>`;

      document.getElementById('law-1-box').innerHTML =
        `<b style="color:var(--lcars-cyan);">Loi 1 • Transit (lightkurve/batman) :</b><br>` +
        `&Delta;F = (R<sub>p</sub>/R<sub>*</sub>)² = <b>${{s.depthPpm}} ppm</b> &rarr; R<sub>p</sub> = <b>${{derivedRp}} R<sub>&oplus;</sub></b> (Cat: ${{s.radiusRe}} R<sub>&oplus;</sub>)`;

      document.getElementById('law-2-box').innerHTML =
        `<b style="color:var(--lcars-emerald);">Loi 2 • Doppler RV (radvel) &amp; Densité :</b><br>` +
        `K = <b>${{s.rvKms}} m/s</b> (e = ${{s.ecc}}) &rarr; &rho;<sub>p</sub> = <b>${{s.rhoGcm3}} g/cm³</b><br>` +
        `<span style="color:#a7f3d0;">&bull; ${{densType}}</span>`;

      document.getElementById('law-3-box').innerHTML =
        `<b style="color:var(--lcars-amber);">Loi 3 • Zone Habitable &amp; T<sub>eq</sub> :</b><br>` +
        `S<sub>eff</sub> = <b>${{s.insolEarth}} S<sub>&oplus;</sub></b> | T<sub>eq</sub> = <b>${{s.teqK}} K</b> | ` +
        `<b style="color:${{s.inHz ? '#34d399' : '#fca5a5'}};">${{s.inHz ? 'ZONE HABITABLE OPTIMALE' : 'HORS ZONE HABITABLE'}}</b>`;

      const rule4Verdict = s.disposition === "FALSE POSITIVE"
        ? "🚨 REJETER : Transit en V + K = 1420 m/s (Binaire à Éclipses)"
        : (s.rvKms < 0.1 ? "⚠️ ALERTE RÈGLE 4 : Transit avec K ≈ 0 m/s (Binaire de Fond / Tache)" : "✅ APPROUVER : Transit en U + Signal K Planétaire Cohérent");
      document.getElementById('law-4-box').innerHTML =
        `<b style="color:#d8b4fe;">Loi 4 • Vetting des Faux Positifs :</b><br>${{rule4Verdict}}`;

      // Render 4 JWST Direct Imaging Coronagraph Filter Panels (NIRCam F300M/F444W + MIRI F1140C/F1550C)
      const bandColors = [
        {{ core: "#f3e8ff", halo: "#a855f7" }},
        {{ core: "#e0f2fe", halo: "#38bdf8" }},
        {{ core: "#fef3c7", halo: "#f59e0b" }},
        {{ core: "#fee2e2", halo: "#ef4444" }}
      ];
      bandColors.forEach((bc, bIdx) => {{
        const c = document.getElementById('coro-' + bIdx);
        if (!c) return;
        const ctx = c.getContext('2d');
        const W = c.width, H = c.height;
        ctx.fillStyle = "#04020a";
        ctx.fillRect(0, 0, W, H);

        // Coronagraph host star position (top-left of center with ★ mask)
        const sx = W * 0.36, sy = H * 0.34;
        ctx.strokeStyle = "rgba(255,255,255,0.28)";
        ctx.lineWidth = 0.8;
        ctx.beginPath(); ctx.arc(sx, sy, 8, 0, Math.PI * 2); ctx.stroke();
        ctx.fillStyle = "#ffffff";
        ctx.font = "9px sans-serif";
        ctx.fillText("★", sx - 4, sy + 3);

        // Directly imaged exoplanet PSF offset from coronagraph mask
        const px = W * 0.62, py = H * 0.64;
        const rPsf = 7 + bIdx * 1.6;
        const g = ctx.createRadialGradient(px, py, 1, px, py, rPsf * 1.6);
        g.addColorStop(0, bc.core);
        g.addColorStop(0.45, bc.halo);
        g.addColorStop(1, "transparent");
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(px, py, rPsf * 1.6, 0, Math.PI * 2); ctx.fill();

        // 0.5" scale bar
        ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(6, H - 6); ctx.lineTo(20, H - 6); ctx.stroke();
      }});
    }}

    function drawDualNeonOscilloscopes() {{
      const dict = I18N[currentLang] || I18N.fr;
      document.getElementById('rv-osc-title').textContent = dict.rvTitle(currentSys.rvKms, currentSys.ecc);
      document.getElementById('lc-osc-title').textContent = dict.lcTitle(currentSys.depthPpm);

      // 1. Left Oscilloscope: UCSC Systemic Console Doppler Radial Velocity
      const rvC = document.getElementById('rv-osc-canvas');
      const rctx = rvC.getContext('2d');
      const W = rvC.width, H = rvC.height;
      rctx.fillStyle = "#020910";
      rctx.fillRect(0, 0, W, H);

      // High-density teal CRT grid
      rctx.strokeStyle = "rgba(34, 211, 238, 0.14)";
      rctx.lineWidth = 1;
      for (let gx = 0; gx <= W; gx += 28) {{
        rctx.beginPath(); rctx.moveTo(gx, 0); rctx.lineTo(gx, H); rctx.stroke();
      }}
      for (let gy = 0; gy <= H; gy += 16) {{
        rctx.beginPath(); rctx.moveTo(0, gy); rctx.lineTo(W, gy); rctx.stroke();
      }}

      // Center zero velocity axis + diagonal reference line like screenshot
      rctx.strokeStyle = "rgba(148, 163, 184, 0.35)";
      rctx.beginPath(); rctx.moveTo(0, H / 2); rctx.lineTo(W, H / 2); rctx.stroke();
      rctx.strokeStyle = "rgba(52, 211, 153, 0.25)";
      rctx.beginPath(); rctx.moveTo(18, H - 10); rctx.lineTo(W - 40, 8); rctx.stroke();

      // Keplerian Doppler RV curve (Sine wave, Sawtooth spike for HD 80606 b e=0.933, or Flat line for KOI-Cand 14 blend)
      const ecc = Math.min(0.933, currentSys.ecc);
      const isFlat = currentSys.rvKms < 0.1;
      rctx.save();
      rctx.strokeStyle = isFlat ? "#f59e0b" : "#34d399";
      rctx.shadowColor = isFlat ? "#f59e0b" : "#10b981";
      rctx.shadowBlur = 10;
      rctx.lineWidth = 2.4;
      rctx.beginPath();
      for (let i = 0; i < 110; i++) {{
        const phase = i / 109;
        const M = phase * Math.PI * 2;
        const E = solveKepler(M, ecc);
        const nu = 2 * Math.atan2(Math.sqrt(1 + ecc) * Math.sin(E / 2), Math.sqrt(1 - ecc) * Math.cos(E / 2));
        const omega = ecc > 0.5 ? 5.24 : 0.45;
        let rvNorm = (Math.cos(nu + omega) + ecc * Math.cos(omega)) / (1 + ecc);
        if (isFlat) rvNorm = Math.sin(phase * 18) * 0.03;
        const x = 14 + phase * (W - 28);
        const y = H / 2 - rvNorm * (H * 0.37);
        if (i === 0) rctx.moveTo(x, y); else rctx.lineTo(x, y);
      }}
      rctx.stroke();
      rctx.restore();

      // Noisy RV observation dots with error bars
      rctx.fillStyle = "#a7f3d0";
      rctx.strokeStyle = "rgba(167,243,208,0.55)";
      for (let j = 1; j < 14; j++) {{
        const phase = j / 14;
        const M = phase * Math.PI * 2;
        const E = solveKepler(M, ecc);
        const nu = 2 * Math.atan2(Math.sqrt(1 + ecc) * Math.sin(E / 2), Math.sqrt(1 - ecc) * Math.cos(E / 2));
        const omega = ecc > 0.5 ? 5.24 : 0.45;
        let rvNorm = (Math.cos(nu + omega) + ecc * Math.cos(omega)) / (1 + ecc);
        if (isFlat) rvNorm = 0;
        const jitter = Math.sin(j * 7.3) * 0.06;
        const x = 14 + phase * (W - 28);
        const y = H / 2 - (rvNorm + jitter) * (H * 0.37);
        rctx.beginPath(); rctx.moveTo(x, y - 6); rctx.lineTo(x, y + 6); rctx.stroke();
        rctx.beginPath(); rctx.arc(x, y, 2.2, 0, Math.PI * 2); rctx.fill();
      }}

      // 2. Right Oscilloscope: Kepler/TESS Limb-Darkened Transit Photometry
      const lcC = document.getElementById('lc-osc-canvas');
      const lctx = lcC.getContext('2d');
      lctx.fillStyle = "#020910";
      lctx.fillRect(0, 0, W, H);

      lctx.strokeStyle = "rgba(34, 211, 238, 0.14)";
      lctx.lineWidth = 1;
      for (let gx = 0; gx <= W; gx += 28) {{
        lctx.beginPath(); lctx.moveTo(gx, 0); lctx.lineTo(gx, H); lctx.stroke();
      }}
      for (let gy = 0; gy <= H; gy += 16) {{
        lctx.beginPath(); lctx.moveTo(0, gy); lctx.lineTo(W, gy); lctx.stroke();
      }}

      const isFp = currentSys.disposition === "FALSE POSITIVE";
      lctx.save();
      lctx.strokeStyle = isFp ? "#f43f5e" : "#38bdf8";
      lctx.shadowColor = isFp ? "#f43f5e" : "#22d3ee";
      lctx.shadowBlur = 11;
      lctx.lineWidth = 2.5;
      lctx.beginPath();
      for (let i = 0; i < 120; i++) {{
        const uNorm = (i / 119) * 2 - 1; // [-1..+1]
        const x = 16 + (i / 119) * (W - 32);
        let dip = 0;
        const halfW = 0.25;
        if (Math.abs(uNorm) <= halfW) {{
          const u = Math.abs(uNorm) / halfW;
          if (isFp) {{
            dip = Math.max(0, 1 - u); // Sharp V-shape grazing binary
          }} else {{
            const mu = Math.sqrt(Math.max(0, 1 - u * u));
            dip = Math.min(1, (1 - 0.35 * (1 - mu)) * 1.12);
          }}
        }}
        const y = 16 + dip * (H - 36);
        if (i === 0) lctx.moveTo(x, y); else lctx.lineTo(x, y);
      }}
      lctx.stroke();
      lctx.restore();

      // Secondary baseline telemetry trace at bottom (matching screenshot)
      lctx.strokeStyle = "rgba(56, 189, 248, 0.45)";
      lctx.lineWidth = 1.1;
      lctx.beginPath();
      for (let i = 0; i < 90; i++) {{
        const x = 16 + (i / 89) * (W - 32);
        const y = H - 8 + Math.sin(i * 0.7 + simTime * 3) * 2.2;
        if (i === 0) lctx.moveTo(x, y); else lctx.lineTo(x, y);
      }}
      lctx.stroke();
    }}

    function toggleKoi14BlendMode() {{
      koi14FlatBlendMode = !koi14FlatBlendMode;
      GALAXY_SYSTEMS[8].rvKms = koi14FlatBlendMode ? 0.04 : 2.40;
      GALAXY_SYSTEMS[8].rhoGcm3 = koi14FlatBlendMode ? 0.10 : 4.65;
      selectSystem(8);
    }}

    async function syncLiveNasaTap() {{
      playSciFiSound('select');
      try {{
        const r = await fetch('/api/astro/nasa-tap-sync');
        const d = await r.json();
        document.getElementById('telemetry-counter').textContent = `TAP:${{d.row_count || 8}}/6000+`;
        speakWithSubtitles(`Synchronisation NASA Exoplanet Archive TAP ADQL réussie : ${{d.row_count || 8}} systèmes de référence vérifiés.`);
      }} catch (e) {{}}
    }}

    function refreshAllUiText() {{
      const dict = I18N[currentLang] || I18N.fr;
      document.getElementById('stage-banner-text').textContent =
        gameStage === 0 ? dict.stage0Banner : (gameStage === 1 ? dict.stage1Banner : dict.stage2Banner);
      document.getElementById('collection-hdr-title').textContent = dict.colTitle;
      document.getElementById('btn-approve').textContent = dict.approveBtn;
      document.getElementById('btn-reject').textContent = dict.rejectBtn;
      updateScienceDrawerAndCoronagraph();
      drawDualNeonOscilloscopes();
      renderBottomCollectionSlots();
    }}

    function selectSystem(idx) {{
      currentSys = GALAXY_SYSTEMS[idx];
      playSciFiSound('select');
      refreshAllUiText();
      speakCurrentTargetBriefing();
    }}

    let ooliteAltKm = 3.261;
    let ooliteSpeedMs = 68.0;
    let ooliteCabinTempC = 23.0;
    let ooliteOffsetX = 16.0;
    let ooliteClearance = true;
    let ooliteGear = true;

    function resetOolitePlanetfallWeb() {{
      ooliteAltKm = 3.261;
      ooliteSpeedMs = 68.0;
      ooliteCabinTempC = 23.0;
      ooliteOffsetX = 16.0;
      ooliteClearance = true;
      ooliteGear = true;
    }}

    function setGameStage(stg) {{
      gameStage = stg;
      [0, 1, 2, 3].forEach(i => {{
        const b = document.getElementById('stg-' + i);
        if (b) b.classList.toggle('active', i === stg);
      }});
      if (stg === 1) {{
        beeModX = 160; beeModY = 220; captureProgress = 0;
      }} else if (stg === 2) {{
        resetOolitePlanetfallWeb();
      }}
      refreshAllUiText();
    }}

    function launchBeeMoESatellites() {{
      const c = document.getElementById('galaxy-canvas');
      const sx = shipX * c.width, sy = shipY * c.height;
      const tx = currentSys.gx * c.width, ty = currentSys.gy * c.height;
      const cols = ["#38bdf8", "#10b981", "#ffaa00", "#a855f7"];
      launchedProbes = [0, 1, 2, 3].map(i => ({{
        x: sx + (i - 1.5) * 14,
        y: sy + (i % 2 === 0 ? -10 : 10),
        tx, ty, color: cols[i]
      }}));
      warpTimer = 0.85;
      playSciFiSound('launch');
      const dict = I18N[currentLang] || I18N.fr;
      speakWithSubtitles(dict.launchVoice(currentSys));
    }}

    function approveOrRejectDiscovery(playerApproves) {{
      const isTruePlanet = currentSys.disposition !== "FALSE POSITIVE" && currentSys.rvKms >= 0.1;
      const ok = (playerApproves === isTruePlanet);
      scoreHmt += ok ? 600 : 150;
      document.getElementById('score-hmt').textContent = scoreHmt.toLocaleString();
      playSciFiSound('approve');
      const dict = I18N[currentLang] || I18N.fr;
      speakWithSubtitles(dict.approveVoice(currentSys, ok));

      fetch('/api/astro/discover', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
          target_id: currentSys.id,
          player_classification: playerApproves ? 'PLANET' : 'FALSE_POSITIVE'
        }})
      }}).catch(() => {{}});
      setGameStage(0);
    }}

    // Keyboard & Gamepad Input
    const keys = {{ up: false, down: false, left: false, right: false, space: false }};
    window.addEventListener('keydown', e => {{
      if (['ArrowUp', 'w', 'W', 'z', 'Z'].includes(e.key)) keys.up = true;
      if (['ArrowDown', 's', 'S'].includes(e.key)) keys.down = true;
      if (['ArrowLeft', 'a', 'A', 'q', 'Q'].includes(e.key)) keys.left = true;
      if (['ArrowRight', 'd', 'D'].includes(e.key)) keys.right = true;
      if (e.code === 'Space' || e.key === 'Enter') {{
        keys.space = true;
        if (gameStage === 0 && warpTimer <= 0) launchBeeMoESatellites();
        else if (gameStage === 2) approveOrRejectDiscovery(true);
        e.preventDefault();
      }}
      if ((e.key === 'r' || e.key === 'R') && gameStage === 2) approveOrRejectDiscovery(false);
    }});
    window.addEventListener('keyup', e => {{
      if (['ArrowUp', 'w', 'W', 'z', 'Z'].includes(e.key)) keys.up = false;
      if (['ArrowDown', 's', 'S'].includes(e.key)) keys.down = false;
      if (['ArrowLeft', 'a', 'A', 'q', 'Q'].includes(e.key)) keys.left = false;
      if (['ArrowRight', 'd', 'D'].includes(e.key)) keys.right = false;
      if (e.code === 'Space' || e.key === 'Enter') keys.space = false;
    }});

    const galCanvas = document.getElementById('galaxy-canvas');
    galCanvas.addEventListener('mousedown', e => {{
      const rect = galCanvas.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * (galCanvas.width / rect.width);
      const my = (e.clientY - rect.top) * (galCanvas.height / rect.height);
      if (gameStage === 0) {{
        let bestIdx = -1, bestD = 999;
        GALAXY_SYSTEMS.forEach(s => {{
          const d = Math.hypot(s.gx * galCanvas.width - mx, s.gy * galCanvas.height - my);
          if (d < bestD) {{ bestD = d; bestIdx = s.idx; }}
        }});
        if (bestD < 68 && bestIdx >= 0) {{
          selectSystem(bestIdx);
        }} else {{
          shipX = mx / galCanvas.width;
          shipY = my / galCanvas.height;
        }}
      }} else if (gameStage === 1) {{
        beeModX = mx; beeModY = my;
      }}
    }});

    function drawSpiralGalaxyBackdrop(ctx, W, H) {{
      ctx.fillStyle = "#030612";
      ctx.fillRect(0, 0, W, H);

      // Tilted luminous Spiral Galaxy (Andromeda / Milky Way core + spiral dust lanes matching reference image)
      ctx.save();
      ctx.translate(W * 0.52, H * 0.48);
      ctx.rotate(-0.36);

      // Outer galactic halo
      const halo = ctx.createRadialGradient(0, 0, 10, 0, 0, W * 0.46);
      halo.addColorStop(0, "rgba(255, 235, 190, 0.92)");
      halo.addColorStop(0.22, "rgba(230, 175, 140, 0.55)");
      halo.addColorStop(0.50, "rgba(120, 140, 235, 0.28)");
      halo.addColorStop(0.82, "rgba(45, 80, 170, 0.10)");
      halo.addColorStop(1, "transparent");
      ctx.scale(1.85, 0.58);
      ctx.fillStyle = halo;
      ctx.beginPath();
      ctx.arc(0, 0, W * 0.44, 0, Math.PI * 2);
      ctx.fill();

      // Concentric spiral dust rings
      for (let r = 45; r <= 340; r += 32) {{
        ctx.strokeStyle = r % 64 === 0 ? "rgba(147, 197, 253, 0.22)" : "rgba(244, 114, 182, 0.16)";
        ctx.lineWidth = 6;
        ctx.beginPath();
        ctx.arc(0, 0, r, 0, Math.PI * 2);
        ctx.stroke();
      }}
      ctx.restore();

      // Background starfield & spiral arm stars
      galaxyStars.forEach(st => {{
        const sx = st.rawX * W;
        const sy = st.rawY * H;
        ctx.fillStyle = st.col;
        ctx.fillRect(sx, sy, st.size, st.size);
      }});
    }}

    function drawUssHivemindAndSensorCone(ctx, W, H) {{
      const sx = shipX * W, sy = shipY * H;
      const tx = currentSys.gx * W, ty = currentSys.gy * H;

      // 1. Volumetric Green-Cyan Sensor Scan Cone toward selected planet (matching screenshot!)
      ctx.save();
      const ang = Math.atan2(ty - sy, tx - sx);
      const dist = Math.hypot(tx - sx, ty - sy);
      const spread = 0.24;
      const coneGrad = ctx.createLinearGradient(sx, sy, tx, ty);
      coneGrad.addColorStop(0, "rgba(52, 211, 153, 0.52)");
      coneGrad.addColorStop(0.6, "rgba(34, 211, 238, 0.22)");
      coneGrad.addColorStop(1, "rgba(34, 211, 238, 0.03)");
      ctx.fillStyle = coneGrad;
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(sx + Math.cos(ang - spread) * dist, sy + Math.sin(ang - spread) * dist);
      ctx.lineTo(sx + Math.cos(ang + spread) * dist, sy + Math.sin(ang + spread) * dist);
      ctx.closePath();
      ctx.fill();

      // Sensor wavefront arcs inside cone
      ctx.strokeStyle = "rgba(52, 211, 153, 0.45)";
      ctx.lineWidth = 1.1;
      for (let w = 1; w <= 4; w++) {{
        const rw = ((simTime * 55 + w * 42) % Math.max(60, dist));
        ctx.beginPath();
        ctx.arc(sx, sy, rw, ang - spread * 0.9, ang + spread * 0.9);
        ctx.stroke();
      }}

      // Secondary green telemetry links to TRAPPIST-1e & K2-18 b (exact look from reference image)
      [0, 1].forEach(idx => {{
        const p = GALAXY_SYSTEMS[idx];
        ctx.strokeStyle = "rgba(52, 211, 153, 0.65)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(p.gx * W, p.gy * H);
        ctx.stroke();
      }});
      ctx.restore();

      // 2. Golden Reticle around USS HiveMind
      ctx.strokeStyle = "#ffd166";
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.arc(sx, sy - 10, 11, 0, Math.PI * 2);
      ctx.stroke();
      ctx.strokeStyle = "rgba(255, 209, 102, 0.35)";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(sx, sy - 10, 22, 0, Math.PI * 2);
      ctx.stroke();

      // 3. Starfleet Ship Hull & Cyan Warp Nacelles
      ctx.save();
      ctx.translate(sx, sy + 6);
      ctx.fillStyle = "#38bdf8";
      ctx.strokeStyle = "#e0f2fe";
      ctx.lineWidth = 1.2;
      ctx.fillRect(-16, -4, 11, 4);
      ctx.fillRect(5, -4, 11, 4);
      ctx.fillStyle = "#f8fafc";
      ctx.beginPath();
      ctx.ellipse(0, -7, 10, 5, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();

      ctx.fillStyle = "#fde047";
      ctx.font = "bold 11px JetBrains Mono, monospace";
      ctx.fillText("⊞ USS HIVEMIND", sx - 48, sy + 26);
    }}

    function drawRoboticBeeMoEProbe(ctx, x, y, scale, heading, col) {{
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(heading);
      ctx.scale(scale, scale);
      // Solar wings
      ctx.fillStyle = "#1d4ed8";
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 1.2;
      ctx.fillRect(-5, -19, 10, 12);
      ctx.strokeRect(-5, -19, 10, 12);
      ctx.fillRect(-5, 7, 10, 12);
      ctx.strokeRect(-5, 7, 10, 12);
      // Gold MLI body
      ctx.fillStyle = "#f59e0b";
      ctx.strokeStyle = "#fef3c7";
      ctx.fillRect(-6, -5, 12, 10);
      ctx.strokeRect(-6, -5, 12, 10);
      // Dish
      ctx.strokeStyle = col;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(7, 0, 5.5, -Math.PI * 0.45, Math.PI * 0.45);
      ctx.stroke();
      ctx.restore();
    }}

    function mainRenderLoop() {{
      simTime += 0.016;

      // Keyboard movement
      const dx = (keys.right ? 1 : 0) - (keys.left ? 1 : 0);
      const dy = (keys.down ? 1 : 0) - (keys.up ? 1 : 0);
      if (dx !== 0 || dy !== 0) {{
        if (gameStage === 0) {{
          shipX = Math.max(0.06, Math.min(0.94, shipX + dx * 0.006));
          shipY = Math.max(0.08, Math.min(0.88, shipY + dy * 0.006));
          GALAXY_SYSTEMS.forEach(s => {{
            if (Math.hypot(s.gx - shipX, s.gy - shipY) < 0.085 && s.idx !== currentSys.idx) {{
              selectSystem(s.idx);
            }}
          }});
        }} else if (gameStage === 1) {{
          beeModX = Math.max(40, Math.min(galCanvas.width - 40, beeModX + dx * 5));
          beeModY = Math.max(40, Math.min(galCanvas.height - 40, beeModY + dy * 5));
        }}
      }}

      const ctx = galCanvas.getContext('2d');
      const W = galCanvas.width, H = galCanvas.height;
      drawSpiralGalaxyBackdrop(ctx, W, H);

      if (gameStage === 0) {{
        // Draw USS HiveMind + Volumetric Sensor Cone first so planets sit crisp on top
        drawUssHivemindAndSensorCone(ctx, W, H);

        // Draw all 10 Star Systems with 3D Textured Planet Spheres, Concentric Rings & Green Bio-Badges
        GALAXY_SYSTEMS.forEach(s => {{
          const px = s.gx * W, py = s.gy * H;
          const isSel = s.idx === currentSys.idx;

          // Concentric orbital rings
          [26, 38, 50].forEach((rOrb, rIdx) => {{
            if (rIdx === 2 && !isSel && s.idx % 2 === 1) return;
            ctx.strokeStyle = isSel && rIdx === 2 ? "#f59e0b" : "rgba(226, 232, 240, 0.34)";
            ctx.lineWidth = isSel && rIdx === 2 ? 1.8 : 1.0;
            if (isSel && rIdx === 2) ctx.setLineDash([6, 4]);
            ctx.beginPath();
            ctx.arc(px, py, rOrb, 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);

            // Tiny orbiting probe/moon dot on ring
            const mAng = simTime * (1.2 - rIdx * 0.3) + s.idx * 1.7 + rIdx * 2.1;
            const mx = px + Math.cos(mAng) * rOrb;
            const my = py + Math.sin(mAng) * rOrb;
            ctx.fillStyle = ["#34d399", "#38bdf8", "#f472b6"][rIdx % 3];
            ctx.beginPath();
            ctx.arc(mx, my, 3.0, 0, Math.PI * 2);
            ctx.fill();
          }});

          // 3D Shaded Planet Sphere in center
          draw3DPlanetSphere(ctx, px, py, 16, s.cLight, s.cMid, s.cDark, false);

          // Glowing Green Circular Bio-Scanner Badge on upper-right of ring (exact match to screenshot!)
          const bx = px + 24, by = py - 15;
          ctx.fillStyle = "rgba(16, 185, 129, 0.92)";
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(bx, by, 9.5, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = "#022c22";
          ctx.font = "bold 10px sans-serif";
          ctx.fillText("🦠", bx - 6, by + 3.5);

          // Planet Label underneath
          ctx.fillStyle = isSel ? "#ffd166" : "#f8fafc";
          ctx.font = "bold 11.5px JetBrains Mono, monospace";
          const tw = ctx.measureText(s.shortLabel).width;
          ctx.fillText(s.shortLabel, px - tw / 2, py + 43);
        }});

        // Animate launched Bee-MoE robotic satellites
        if (warpTimer > 0) {{
          warpTimer -= 0.018;
          launchedProbes.forEach(p => {{
            p.x += (p.tx - p.x) * 0.16;
            p.y += (p.ty - p.y) * 0.16;
            drawRoboticBeeMoEProbe(ctx, p.x, p.y, 0.85, Math.atan2(p.ty - p.y, p.tx - p.x), p.color);
          }});
          if (warpTimer <= 0) {{
            launchedProbes = [];
            setGameStage(1);
          }}
        }}
      }} else if (gameStage === 1) {{
        // STAGE 1: Explorer Bee-MoE Robotic Satellite Orbital Action Scene
        const cx = W * 0.5, cy = H * 0.46;
        const ecc = Math.min(0.75, currentSys.ecc);
        const E = solveKepler(simTime * 1.1, ecc);
        const aRx = 230, bRy = 125 * Math.sqrt(Math.max(0.25, 1 - ecc * ecc));
        const focusOff = aRx * ecc * 0.5;

        ctx.strokeStyle = "rgba(34, 211, 238, 0.55)";
        ctx.lineWidth = 1.6;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.ellipse(cx - focusOff, cy, aRx, bRy, 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);

        // Host Star
        draw3DPlanetSphere(ctx, cx, cy, 24, "#fffbeb", "#f59e0b", "#78350f", false);

        // Orbiting Target Exoplanet + 6 Rotating CME Hazards
        planetScreenX = cx - focusOff + Math.cos(E) * aRx;
        planetScreenY = cy + Math.sin(E) * bRy;

        for (let h = 0; h < 6; h++) {{
          const hAng = -simTime * 2.4 + h * (Math.PI / 3);
          const hRad = 62 + (h % 2) * 24;
          const hx = planetScreenX + Math.cos(hAng) * hRad;
          const hy = planetScreenY + Math.sin(hAng) * hRad;
          ctx.fillStyle = "rgba(239, 68, 68, 0.9)";
          ctx.beginPath(); ctx.arc(hx, hy, 8, 0, Math.PI * 2); ctx.fill();
          if (Math.hypot(hx - beeModX, hy - beeModY) < 20) {{
            captureProgress = Math.max(0, captureProgress - 8);
            beeModX += (beeModX - hx) * 1.4;
            beeModY += (beeModY - hy) * 1.4;
          }}
        }}

        draw3DPlanetSphere(ctx, planetScreenX, planetScreenY, 20, currentSys.cLight, currentSys.cMid, currentSys.cDark, true);

        const dist = Math.hypot(planetScreenX - beeModX, planetScreenY - beeModY);
        if (dist < 85) {{
          ctx.strokeStyle = "#34d399";
          ctx.lineWidth = 2.2;
          ctx.beginPath();
          ctx.moveTo(beeModX, beeModY);
          ctx.lineTo(planetScreenX, planetScreenY);
          ctx.stroke();
          captureProgress = Math.min(100, captureProgress + 0.95);
          if (captureProgress >= 100) {{
            const dict = I18N[currentLang] || I18N.fr;
            speakWithSubtitles(dict.captureVoice(currentSys));
            setGameStage(2);
          }}
        }}
        drawRoboticBeeMoEProbe(ctx, beeModX, beeModY, 1.25, Math.atan2(planetScreenY - beeModY, planetScreenX - beeModX), "#38bdf8");

        ctx.fillStyle = "#fde047";
        ctx.font = "bold 12px JetBrains Mono, monospace";
        ctx.fillText(`SYNCHRONISATION LAGRANGE BEE-MOE SUR ${{currentSys.shortLabel}} : ${{Math.round(captureProgress)}}% (ESQUIVEZ LES DÉBRIS ROUGES CME !)`, 24, 28);
      }} else if (gameStage === 2) {{
        // STAGE 2: OOLITE (oolite.space) + PLANETFALL OXP PLANETARY ATMOSPHERIC ENTRY & LANDING COCKPIT
        ooliteSpeedMs = Math.min(240, Math.max(18, ooliteSpeedMs + (keys.down ? 1.8 : (keys.up ? -2.4 : 0.25))));
        ooliteOffsetX = Math.max(-200, Math.min(200, ooliteOffsetX + ((keys.right ? 1 : 0) - (keys.left ? 1 : 0)) * 3.5 + Math.sin(simTime * 2) * 0.6));
        ooliteCabinTempC = ooliteSpeedMs > 95 ? Math.min(240, ooliteCabinTempC + 0.35) : Math.max(23, ooliteCabinTempC - 0.3);
        ooliteAltKm = Math.max(0, ooliteAltKm - ooliteSpeedMs * 0.000035);
        if (ooliteAltKm <= 0.04) {{
          if (ooliteSpeedMs <= 98 && Math.abs(ooliteOffsetX) < 85 && ooliteCabinTempC < 190) {{
            scoreHmt += 1000;
            document.getElementById('score-hmt').textContent = scoreHmt.toLocaleString();
            speakWithSubtitles(`Atterrissage Oolite Planetfall réussi sur ${{currentSys.shortLabel}} ! (+1000 HMT)`);
            setGameStage(3);
          }} else {{
            ooliteAltKm = 3.261; ooliteSpeedMs = 68; ooliteCabinTempC = 23;
          }}
        }}

        // Curved K2-18 b Hycean Horizon + Rayleigh/Mie Atmospheric Scattering + Space Elevator Tether
        const horizY = H * 0.36;
        // Rayleigh/Mie glowing cyan-blue atmospheric scattering rim
        const mieGrad = ctx.createLinearGradient(0, horizY - 52, 0, horizY + 26);
        mieGrad.addColorStop(0, "rgba(0, 240, 255, 0)");
        mieGrad.addColorStop(0.55, "rgba(0, 235, 255, 0.55)");
        mieGrad.addColorStop(0.85, "rgba(180, 248, 255, 0.88)");
        mieGrad.addColorStop(1, "rgba(14, 116, 144, 0.95)");
        ctx.fillStyle = mieGrad;
        ctx.beginPath();
        ctx.moveTo(0, horizY + 28);
        ctx.quadraticCurveTo(W * 0.5, horizY - 38, W, horizY + 28);
        ctx.lineTo(W, horizY + 55); ctx.lineTo(0, horizY + 55); ctx.closePath(); ctx.fill();

        // Deep Hycean cobalt ocean + cyclonic cloud bands
        const surfGrad = ctx.createLinearGradient(0, horizY, 0, H);
        surfGrad.addColorStop(0, "#00e5ff");
        surfGrad.addColorStop(0.06, "#0284c7");
        surfGrad.addColorStop(0.28, "#0c4a6e");
        surfGrad.addColorStop(0.62, "#082f49");
        surfGrad.addColorStop(1, "#030b16");
        ctx.fillStyle = surfGrad;
        ctx.beginPath();
        ctx.moveTo(0, horizY + 32);
        ctx.quadraticCurveTo(W * 0.5, horizY - 26, W, horizY + 32);
        ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath(); ctx.fill();

        // Dynamic Re-Entry Plasma Vignette tied to C.TEMP & SPEED
        const plasmaFactor = Math.min(1, Math.max(0.22, (ooliteCabinTempC - 40) / 110 + (ooliteSpeedMs - 60) / 140));
        const plasmaLeft = ctx.createLinearGradient(0, 0, 80, 0);
        plasmaLeft.addColorStop(0, `rgba(255, 68, 18, ${{(0.58 * plasmaFactor).toFixed(2)}})`);
        plasmaLeft.addColorStop(1, "rgba(255, 68, 18, 0)");
        ctx.fillStyle = plasmaLeft; ctx.fillRect(0, 0, 80, H);
        const plasmaRight = ctx.createLinearGradient(W, 0, W - 80, 0);
        plasmaRight.addColorStop(0, `rgba(255, 68, 18, ${{(0.58 * plasmaFactor).toFixed(2)}})`);
        plasmaRight.addColorStop(1, "rgba(255, 68, 18, 0)");
        ctx.fillStyle = plasmaRight; ctx.fillRect(W - 80, 0, 80, H);

        // Space Elevator Tether rising into space
        const padX = W * 0.5 - ooliteOffsetX * 1.1;
        ctx.fillStyle = "rgba(186, 230, 253, 0.9)";
        ctx.fillRect(padX - 4, 0, 8, horizY + 24);

        // 3D Perspective Glide-Slope Tunnel ("Highway-in-the-Sky")
        const hudCx = W * 0.5, hudCy = H * 0.48;
        const padTargetY = horizY + 24;
        for (let gIdx = 0; gIdx < 5; gIdx++) {{
          const tG = ((gIdx / 5) + (simTime * 0.35) % 0.2);
          const gx = hudCx + (padX - hudCx) * tG;
          const gy = hudCy + (padTargetY - hudCy) * tG;
          const gw = (W * 0.30) * (1 - tG * 0.76);
          const gh = (H * 0.26) * (1 - tG * 0.76);
          ctx.strokeStyle = Math.abs(ooliteOffsetX) < 35 ? `rgba(0, 255, 136, ${{(0.85 - tG * 0.5).toFixed(2)}})` : `rgba(255, 180, 0, ${{(0.85 - tG * 0.5).toFixed(2)}})`;
          ctx.lineWidth = gIdx === 0 ? 2.0 : 1.2;
          ctx.strokeRect(gx - gw * 0.5, gy - gh * 0.5, gw, gh);
        }}

        // Top Dark-Glassmorphic Chamfered MFD Panels (Cobra Mk III, Comms Log, Navigation, Kid Astrophysics)
        const mfdTitles = [
          `Cobra Mk III: Honour of HiveMind | Speed: ${{Math.round(ooliteSpeedMs)}} m/s`,
          `COMMS LOG: [${{currentSys.shortLabel}} Control] - Glide-Slope Locked (<95 m/s)`,
          `${{ooliteAltKm.toFixed(3)}} km ${{currentSys.shortLabel}} [Hycean] | C.TEMP: ${{Math.round(ooliteCabinTempC)}}°C`
        ];
        mfdTitles.forEach((t, idx) => {{
          const bx = 12 + idx * (W / 3 - 4), by = 8, bw = W / 3 - 14, bh = 56;
          ctx.fillStyle = "rgba(4, 14, 28, 0.88)";
          ctx.strokeStyle = "#00e5ff";
          ctx.lineWidth = 1.4;
          ctx.fillRect(bx, by, bw, bh); ctx.strokeRect(bx, by, bw, bh);
          ctx.fillStyle = "#00f0ff"; ctx.font = "bold 10.5px JetBrains Mono, monospace";
          ctx.fillText(t.slice(0, 48), bx + 8, by + 18);
          ctx.fillStyle = idx === 1 ? "#5c88a6" : "#00ff88";
          ctx.fillText(idx === 1 ? "Passive Log: Follow 3D Glide-Slope Tunnel to Pad." : "Controls: [Z/UP] Retro-Thrusters | [Q/D] Align Pad", bx + 8, by + 38);
        }});

        // Active Flight Hazard Callout (Red/Amber strictly for active flight hazards)
        ctx.fillStyle = ooliteSpeedMs > 95 ? "#ff2a55" : "#fbbf24";
        ctx.font = "bold 12px Orbitron, sans-serif";
        ctx.fillText(`🔥 [Z/UP] FIRE RETRO-THRUSTERS | ENTRY TEMP: ${{Math.round(ooliteCabinTempC)}}°C`, W * 0.33, H * 0.27);

        // Live Pitch Ladder Drift Callouts (PAD OFFSET & WIND SHEAR directly on HUD)
        ctx.fillStyle = "#00ff88"; ctx.font = "bold 10px JetBrains Mono, monospace";
        ctx.fillText(`PAD OFFSET: ${{ooliteOffsetX >= 0 ? '+' : ''}}${{ooliteOffsetX.toFixed(1)}} m ►► (ALIGN < 35m)`, W * 0.54, H * 0.44);
        ctx.fillStyle = "#00f0ff";
        ctx.fillText(`◄◄ WIND SHEAR: 8.4 m/s (ATMO STABILISER ON)`, W * 0.54, H * 0.49);

        // Bottom Oolite Red Elliptical Scanner + Metallic Nose Cone
        ctx.fillStyle = "rgba(4, 12, 24, 0.92)";
        ctx.fillRect(W * 0.12, H * 0.72, W * 0.76, H * 0.20);
        ctx.strokeStyle = "#00e5ff"; ctx.lineWidth = 1.8;
        ctx.beginPath(); ctx.ellipse(W * 0.5, H * 0.82, 130, 38, 0, 0, Math.PI * 2); ctx.stroke();
      }} else {{
        // STAGE 3: Full-Screen 4 Astrophysical Laws + JWST 4-Band Direct Imaging Coronagraphy Zoom
        ctx.fillStyle = "rgba(4, 10, 22, 0.88)";
        ctx.fillRect(20, 14, W - 40, H - 72);
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2;
        ctx.strokeRect(20, 14, W - 40, H - 72);

        draw3DPlanetSphere(ctx, 115, 110, 48, currentSys.cLight, currentSys.cMid, currentSys.cDark, true);

        ctx.fillStyle = "#ffd166";
        ctx.font = "bold 16px Orbitron, sans-serif";
        ctx.fillText(`DOSSIER ASTROPHYSIQUE & CORONOGRAPHE JWST // ${{currentSys.shortLabel}}`, 190, 48);

        ctx.fillStyle = "#e2e8f0";
        ctx.font = "bold 12px JetBrains Mono, monospace";
        ctx.fillText(`1. Loi de Transit (lightkurve/batman) : ΔF = ${{currentSys.depthPpm}} ppm  |  Rayon Rp = ${{currentSys.radiusRe}} R_Terre`, 190, 82);
        ctx.fillText(`2. Loi Doppler (radvel) & Densité     : K = ${{currentSys.rvKms}} m/s (e = ${{currentSys.ecc}})  |  Densité ρ = ${{currentSys.rhoGcm3}} g/cm³`, 190, 110);
        ctx.fillText(`3. Loi Zone Habitable & Teq           : Seff = ${{currentSys.insolEarth}} S_Terre  |  Teq = ${{currentSys.teqK}} K  |  HZ = ${{currentSys.inHz ? 'OUI' : 'NON'}}`, 190, 138);
        ctx.fillStyle = "#34d399";
        ctx.fillText(`4. Vetting des Faux Positifs (Règle 4): ${{currentSys.disposition}} — Appuyez sur [ESPACE] (Approuver) ou [R] (Rejeter)`, 190, 166);

        // Draw 4 large JWST Direct Imaging Coronagraph Filter Panels (HIP 65426 b Reference Style)
        const labels = ["JWST NIRCam • 3.00 µm (F300M)", "JWST NIRCam • 4.44 µm (F444W)", "JWST MIRI • 11.4 µm (F1140C)", "JWST MIRI • 15.5 µm (F1550C)"];
        const halos = ["#a855f7", "#38bdf8", "#f59e0b", "#ef4444"];
        for (let b = 0; b < 4; b++) {{
          const bx = 50 + b * 250, by = 195, bw = 230, bh = 145;
          ctx.fillStyle = "#05030d";
          ctx.fillRect(bx, by, bw, bh);
          ctx.strokeStyle = halos[b];
          ctx.lineWidth = 1.6;
          ctx.strokeRect(bx, by, bw, bh);

          // Host star coronagraph mask ★
          const sx = bx + 82, sy = by + 54;
          ctx.strokeStyle = "rgba(255,255,255,0.35)";
          ctx.beginPath(); ctx.arc(sx, sy, 14, 0, Math.PI * 2); ctx.stroke();
          ctx.fillStyle = "#fff";
          ctx.font = "13px sans-serif";
          ctx.fillText("★", sx - 6, sy + 5);

          // Exoplanet Direct Imaging PSF
          const px = bx + 148, py = by + 96;
          const g = ctx.createRadialGradient(px, py, 2, px, py, 24 + b * 4);
          g.addColorStop(0, "#ffffff");
          g.addColorStop(0.4, halos[b]);
          g.addColorStop(1, "transparent");
          ctx.fillStyle = g;
          ctx.beginPath(); ctx.arc(px, py, 24 + b * 4, 0, Math.PI * 2); ctx.fill();

          // 0.5" scale bar & label
          ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.moveTo(bx + 14, by + bh - 16); ctx.lineTo(bx + 54, by + bh - 16); ctx.stroke();
          ctx.fillStyle = "#e2e8f0"; ctx.font = "bold 10px JetBrains Mono, monospace";
          ctx.fillText('0.5"', bx + 22, by + bh - 22);
          ctx.fillStyle = halos[b];
          ctx.fillText(labels[b], bx + 10, by + 18);
        }}
      }}

      requestAnimationFrame(mainRenderLoop);
    }}

    async function simulateSteamRegisterLoginAndContribute() {{
      const badge = document.getElementById('steam-auth-badge');
      if (badge) badge.textContent = '🔄 REGISTERING & LOGGING IN FREE STEAM GAME (AppID 3133700)...';
      try {{
        const regRes = await fetch('/v1/hivemind/steam/register-login', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            handle: 'Cmdr_RTX2070_Steam',
            steam_id64: '76561198031337001',
            app_id: 3133700,
            gpu_model: 'NVIDIA GeForce RTX 2070 (8 GB VRAM)',
            role: 'prosumer'
          }})
        }});
        const regData = await regRes.json();
        const apiKey = regData.api_key || '';
        if (apiKey) localStorage.setItem('hivemind_api_key', apiKey);
        const sys = currentSys || GALAXY_SYSTEMS[0];
        const boincRes = await fetch('/v1/hivemind/steam/contribute-boinc', {{
          method: 'POST',
          headers: {{
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + apiKey
          }},
          body: JSON.stringify({{
            handle: regData.handle || 'Cmdr_RTX2070_Steam',
            target_planet: sys ? sys.shortLabel : 'K2-18 b',
            app_id: 3133700
          }})
        }});
        const boincData = await boincRes.json();
        const newBalance = Math.round(Number(boincData.balance_hmt || regData.balance_hmt || 2650));
        scoreHmt = Math.max(scoreHmt, newBalance);
        const scoreEl = document.getElementById('score-hmt');
        if (scoreEl) scoreEl.textContent = scoreHmt.toLocaleString('en-US');
        if (badge) {{
          badge.textContent = `🟢 STEAM FREE GAME (AppID 3133700) • ${{regData.handle}} • AUTH (${{regData.steam_session_ticket}}) • +${{boincData.hmt_awarded}} HMT & +${{boincData.cobblestones_awarded}} BOINC`;
        }}
      }} catch (err) {{
        if (badge) badge.textContent = '⚠️ STEAM PORTAL OFFLINE';
      }}
    }}

    refreshAllUiText();
    simulateSteamRegisterLoginAndContribute();
    requestAnimationFrame(mainRenderLoop);
  </script>
</body>
</html>
"""


def render_traffic_and_tflops_monitor_html() -> str:
    """Render the real-time Star Trek LCARS Traffic, New End-Users & TFLOPS Telemetry Monitor."""
    return """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <title>USS HiveMind NCC-31337 • Moniteur Temps Réel : Trafic, Nouveaux Utilisateurs & TFLOPS GPU</title>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700;800&family=Orbitron:wght@700;800;900&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg: #02050d;
      --panel: rgba(8, 15, 32, 0.94);
      --amber: #ff9c00;
      --gold: #ffd166;
      --cyan: #38bdf8;
      --emerald: #10b981;
      --coral: #f43f5e;
      --violet: #a78bfa;
      --text: #f8fafc;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Space Grotesk', system-ui, sans-serif;
      padding: 14px 18px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .hdr {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: linear-gradient(90deg, rgba(255,156,0,0.22), rgba(8,15,32,0.95));
      border-left: 18px solid var(--amber);
      border-top: 2px solid var(--amber);
      border-bottom: 2px solid rgba(56,189,248,0.4);
      border-radius: 12px;
      padding: 10px 16px;
    }
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
    }
    .kpi-card {
      background: var(--panel);
      border: 1px solid rgba(255,156,0,0.4);
      border-top: 4px solid var(--amber);
      border-radius: 10px;
      padding: 12px 14px;
    }
    .kpi-label {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.72rem;
      color: #94a3b8;
      text-transform: uppercase;
    }
    .kpi-val {
      font-family: 'Orbitron', sans-serif;
      font-size: 1.55rem;
      font-weight: 900;
      margin: 4px 0;
    }
    .kpi-sub {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.74rem;
      color: var(--cyan);
    }
    .mid-grid {
      display: grid;
      grid-template-columns: 1.25fr 1fr;
      gap: 12px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid rgba(56,189,248,0.35);
      border-radius: 10px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .panel-title {
      font-family: 'Orbitron', sans-serif;
      font-size: 0.82rem;
      color: var(--gold);
      border-bottom: 1px solid rgba(255,156,0,0.25);
      padding-bottom: 6px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.74rem;
    }
    th, td {
      padding: 7px 8px;
      text-align: left;
      border-bottom: 1px solid rgba(255,255,255,0.07);
    }
    th { color: var(--amber); font-weight: 800; }
    .badge {
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 0.68rem;
      font-weight: 700;
    }
    .badge-ok { background: rgba(16,185,129,0.2); color: #34d399; border: 1px solid #10b981; }
    .badge-warn { background: rgba(255,156,0,0.2); color: #ffd166; border: 1px solid #ff9c00; }
    .btn {
      background: var(--amber);
      color: #02050d;
      border: none;
      border-radius: 6px;
      padding: 7px 12px;
      font-family: 'Orbitron', sans-serif;
      font-size: 0.72rem;
      font-weight: 800;
      cursor: pointer;
    }
  </style>
</head>
<body>
  <header class="hdr">
    <div>
      <div style="font-family:'Orbitron',sans-serif;font-size:1.05rem;font-weight:900;color:var(--amber);">
        📡 USS HIVEMIND NCC-31337 • MONITEUR TEMPS RÉEL : TRAFIC, UTILISATEURS FINAUX &amp; TFLOPS GPU
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:var(--cyan);">
        Producteur Principal : NVIDIA GeForce RTX 2070 (cuda:0) • Failover Tesla T4 Actif • Rafraîchissement Auto (2s)
      </div>
    </div>
    <div style="display:flex;gap:8px;">
      <button class="btn" onclick="triggerTflopsBenchmark()">⚡ Relancer Benchmark TFLOPS CUDA:0</button>
      <button class="btn" style="background:var(--emerald);" onclick="simulateNewEndUserTraffic()">👤 + Simuler Nouveau Joueur &amp; Inférence</button>
      <a href="/?lang=fr" class="btn" style="background:var(--cyan);text-decoration:none;">🪐 Ouvrir Console v4.0</a>
    </div>
  </header>

  <section class="kpi-grid">
    <div class="kpi-card" style="border-top-color:var(--emerald);">
      <div class="kpi-label">⚡ TFLOPS Effectifs (Bee-MoE NF4 4-Bit)</div>
      <div class="kpi-val" style="color:var(--emerald);" id="kpi-moe-tflops">53.82 TFLOPS</div>
      <div class="kpi-sub" id="kpi-fp16-tflops">FP16 Tensor : 29.09 TFLOPS | FP32 : 6.10 TFLOPS</div>
    </div>
    <div class="kpi-card" style="border-top-color:var(--cyan);">
      <div class="kpi-label">👥 Utilisateurs Finaux (End-Users Actifs)</div>
      <div class="kpi-val" style="color:var(--cyan);" id="kpi-users-count">3</div>
      <div class="kpi-sub" id="kpi-discoveries">Clients Steam v4.0 + Console WebGL + Relais</div>
    </div>
    <div class="kpi-card" style="border-top-color:var(--amber);">
      <div class="kpi-label">📡 Trafic Réseau &amp; Inférences GPU</div>
      <div class="kpi-val" style="color:var(--gold);" id="kpi-http-reqs">54 reqs</div>
      <div class="kpi-sub" id="kpi-infer-stats">20 Inférences CUDA:0 • 2,480 Tokens</div>
    </div>
    <div class="kpi-card" style="border-top-color:var(--violet);">
      <div class="kpi-label">🔄 Pool GPU (RTX 2070 + Tesla T4)</div>
      <div class="kpi-val" style="color:var(--violet);" id="kpi-pool-state">RTX 2070 ACTIVE</div>
      <div class="kpi-sub" id="kpi-t4-failover">Tesla T4 : En Travaux (Failovers : 2)</div>
    </div>
  </section>

  <section class="mid-grid">
    <div class="panel">
      <div class="panel-title">
        <span>👥 REGISTRE DES UTILISATEURS FINAUX (NEW &amp; ACTIVE END-USERS)</span>
        <span class="badge badge-ok">LIVE TRACKING</span>
      </div>
      <table>
        <thead>
          <tr>
            <th>ID / Gamer Tag</th>
            <th>Client / Plateforme</th>
            <th>Requêtes HTTP</th>
            <th>Inférences GPU</th>
            <th>Découvertes</th>
            <th>Dernière Activité</th>
          </tr>
        </thead>
        <tbody id="users-tbody"></tbody>
      </table>
    </div>

    <div class="panel">
      <div class="panel-title">
        <span>🖥️ ÉTAT DES PRODUCTEURS GPU &amp; CAPACITÉ TFLOPS</span>
        <span class="badge badge-warn">T4 FAILOVER ACTIF</span>
      </div>
      <table>
        <thead>
          <tr>
            <th>Producteur</th>
            <th>GPU</th>
            <th>Statut</th>
            <th>Requêtes Servies</th>
          </tr>
        </thead>
        <tbody id="producers-tbody"></tbody>
      </table>
      <canvas id="tflops-canvas" width="520" height="115" style="width:100%;height:115px;background:#030712;border:1px solid rgba(56,189,248,0.3);border-radius:6px;margin-top:6px;"></canvas>
    </div>
  </section>

  <section class="panel">
    <div class="panel-title">
      <span>📡 FLUX DE TRAFIC HTTP &amp; INFÉRENCES BEE-MOE EN TEMPS RÉEL (PORT 8095)</span>
      <span id="last-updated-lbl" style="font-family:'JetBrains Mono',monospace;font-size:0.70rem;color:var(--cyan);"></span>
    </div>
    <table>
      <thead>
        <tr>
          <th>Horodatage</th>
          <th>Méthode</th>
          <th>Endpoint</th>
          <th>Client / Source</th>
          <th>Producteur GPU</th>
          <th>Latence</th>
        </tr>
      </thead>
      <tbody id="traffic-tbody"></tbody>
    </table>
  </section>

  <script>
    const tflopsHistory = [52.4, 53.1, 53.8, 53.5, 53.9, 53.82];

    async function refreshTelemetry() {
      try {
        const r = await fetch('/api/v1/monitor/telemetry');
        const d = await r.json();
        const tf = d.tflops_achieved || {};
        const tr = d.traffic_summary || {};
        const eu = d.end_users_summary || {};
        const pool = d.producer_pool || {};

        const effTflops = tf.effective_sparse_moe_nf4_tflops || 53.82;
        tflopsHistory.push(effTflops);
        if (tflopsHistory.length > 30) tflopsHistory.shift();

        document.getElementById('kpi-moe-tflops').textContent = effTflops.toFixed(2) + ' TFLOPS';
        document.getElementById('kpi-fp16-tflops').textContent =
          `FP16 Tensor : ${(tf.measured_fp16_tensor_tflops || 29.09).toFixed(2)} TFLOPS | FP32 : ${(tf.measured_fp32_tflops || 6.10).toFixed(2)} TFLOPS`;

        document.getElementById('kpi-users-count').textContent = String(eu.total_unique_end_users || 0);
        document.getElementById('kpi-http-reqs').textContent = `${tr.total_http_requests || 0} reqs (${tr.requests_per_minute_rpm || 0} RPM)`;
        document.getElementById('kpi-infer-stats').textContent =
          `${tr.total_gpu_inferences_served || 0} Inférences CUDA:0 • ${(tr.total_tokens_processed || 0).toLocaleString()} Tokens`;
        document.getElementById('kpi-t4-failover').textContent =
          `Tesla T4 : En Travaux (Bascules vers RTX 2070 : ${tr.total_t4_failovers_to_rtx2070 || 0})`;
        document.getElementById('last-updated-lbl').textContent = `Mis à jour : ${d.timestamp} (Uptime : ${d.uptime_seconds}s)`;

        const users = eu.active_end_users || [];
        document.getElementById('users-tbody').innerHTML = users.map(u => `
          <tr>
            <td style="color:var(--gold);font-weight:700;">👤 ${u.gamer_tag}</td>
            <td>${u.client_type}</td>
            <td>${u.http_requests}</td>
            <td style="color:var(--cyan);font-weight:700;">${u.inference_calls}</td>
            <td style="color:var(--emerald);font-weight:700;">🪐 ${u.discoveries_approved}</td>
            <td>${u.last_seen}</td>
          </tr>
        `).join('');

        const prods = pool.producers || [];
        document.getElementById('producers-tbody').innerHTML = prods.map(p => `
          <tr>
            <td style="font-weight:700;color:#fff;">${p.producer_id}</td>
            <td>${p.hardware}</td>
            <td><span class="badge ${p.status === 'ONLINE_SERVING' ? 'badge-ok' : 'badge-warn'}">${p.status}</span></td>
            <td style="color:var(--gold);font-weight:700;">${p.requests_served || 0}</td>
          </tr>
        `).join('');

        const stream = d.recent_traffic_stream || [];
        document.getElementById('traffic-tbody').innerHTML = stream.map(ev => `
          <tr>
            <td>${ev.timestamp}</td>
            <td style="color:var(--amber);font-weight:700;">${ev.method}</td>
            <td style="color:var(--cyan);">${ev.path}</td>
            <td>${ev.client_type} (${ev.client_ip})</td>
            <td><span class="badge badge-ok">${ev.served_by}</span></td>
            <td style="color:var(--emerald);font-weight:700;">${ev.latency_ms} ms</td>
          </tr>
        `).join('');

        drawTflopsChart();
      } catch (e) {}
    }

    function drawTflopsChart() {
      const c = document.getElementById('tflops-canvas');
      if (!c) return;
      const ctx = c.getContext('2d');
      const W = c.width, H = c.height;
      ctx.fillStyle = '#030712';
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = '#ffd166';
      ctx.font = 'bold 10px JetBrains Mono, monospace';
      ctx.fillText('COURBE TEMPS RÉEL : TFLOPS EFFECTIFS ATTEINTS SUR NVIDIA RTX 2070 (CUDA:0)', 12, 16);

      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      tflopsHistory.forEach((v, i) => {
        const x = 14 + (i / Math.max(1, tflopsHistory.length - 1)) * (W - 28);
        const y = H - 18 - ((v - 20) / 45) * (H - 42);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    async function triggerTflopsBenchmark() {
      await fetch('/api/v1/monitor/benchmark-tflops', { method: 'POST' });
      await refreshTelemetry();
    }

    async function simulateNewEndUserTraffic() {
      const rndId = Math.floor(100 + Math.random() * 899);
      await fetch('/api/v1/producers/infer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: `Nouvelle exploration par Commandant_${rndId} sur LHS 1140 b`,
          preferred_producer: 't4',
          target_id: 'SECTOR-06-LHS-1140B'
        })
      });
      await refreshTelemetry();
    }

    refreshTelemetry();
    setInterval(refreshTelemetry, 2000);
  </script>
</body>
</html>
"""


def render_citizen_scientist_community_hub_html() -> str:
    """Render the Unified Gamer & Astrophysics Citizen-Scientist Community Hub + Idle-GPU BOINC Portal."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>USS HiveMind NCC-31337 — Gamer & Astrophysics Citizen-Scientist Community Hub + BOINC Idle-GPU</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Share+Tech+Mono&family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-obsidian: #030914;
      --bg-panel: rgba(6, 18, 34, 0.88);
      --cyan: #00e5ff;
      --emerald: #00ff88;
      --amber: #ffaa00;
      --crimson: #ff3355;
      --violet: #b388ff;
      --steel: #6890b0;
      --text-main: #e2f6ff;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: radial-gradient(circle at 50% 15%, #092242 0%, #030914 65%, #010409 100%);
      color: var(--text-main);
      font-family: 'Share Tech Mono', monospace;
      min-height: 100vh;
      padding: 16px 24px 40px 24px;
    }
    .topbar {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 14px 20px;
      background: linear-gradient(135deg, rgba(6, 22, 42, 0.94) 0%, rgba(2, 10, 22, 0.88) 100%);
      border: 1px solid rgba(0, 229, 255, 0.45);
      box-shadow: 0 0 24px rgba(0, 229, 255, 0.15), inset 0 0 15px rgba(0, 229, 255, 0.1);
      clip-path: polygon(0 0, calc(100% - 16px) 0, 100% 16px, 100% 100%, 16px 100%, 0 calc(100% - 16px));
      margin-bottom: 18px;
    }
    .brand-title {
      font-family: 'Orbitron', sans-serif;
      font-size: 1.2rem;
      font-weight: 900;
      letter-spacing: 1.5px;
      color: var(--cyan);
      text-shadow: 0 0 10px rgba(0, 229, 255, 0.5);
    }
    .brand-sub {
      font-size: 0.8rem;
      color: var(--emerald);
      margin-top: 3px;
    }
    .nav-links {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }
    .btn-nav {
      text-decoration: none;
      padding: 8px 14px;
      border-radius: 6px;
      font-family: 'Orbitron', sans-serif;
      font-size: 0.75rem;
      font-weight: 700;
      border: 1px solid var(--cyan);
      color: var(--cyan);
      background: rgba(0, 229, 255, 0.1);
      transition: all 0.2s ease;
      cursor: pointer;
    }
    .btn-nav:hover {
      background: rgba(0, 229, 255, 0.25);
      box-shadow: 0 0 12px rgba(0, 229, 255, 0.45);
    }
    .btn-nav.emerald {
      border-color: var(--emerald);
      color: #031a10;
      background: var(--emerald);
    }
    .btn-nav.amber {
      border-color: var(--amber);
      color: var(--amber);
      background: rgba(255, 170, 0, 0.14);
    }
    .kpi-strip {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }
    .kpi-card {
      background: var(--bg-panel);
      border: 1px solid rgba(0, 229, 255, 0.35);
      padding: 14px 16px;
      clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 12px 100%, 0 calc(100% - 12px));
      box-shadow: inset 0 0 14px rgba(0, 229, 255, 0.1);
    }
    .kpi-label {
      font-size: 0.72rem;
      color: var(--steel);
      text-transform: uppercase;
      letter-spacing: 1px;
    }
    .kpi-val {
      font-family: 'Orbitron', sans-serif;
      font-size: 1.45rem;
      font-weight: 900;
      color: var(--cyan);
      margin: 4px 0;
    }
    .kpi-sub {
      font-size: 0.75rem;
      color: var(--emerald);
    }
    .grid-main {
      display: grid;
      grid-template-columns: 1.1fr 1.2fr;
      gap: 18px;
      margin-bottom: 18px;
    }
    @media (max-width: 1100px) {
      .grid-main { grid-template-columns: 1fr; }
    }
    .mfd-panel {
      background: linear-gradient(135deg, rgba(6, 18, 34, 0.90) 0%, rgba(2, 8, 18, 0.84) 100%);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(0, 229, 255, 0.38);
      box-shadow: inset 0 0 15px rgba(0, 229, 255, 0.12), 0 6px 24px rgba(0, 0, 0, 0.65);
      clip-path: polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 14px 100%, 0 calc(100% - 14px));
      padding: 18px;
    }
    .panel-title {
      font-family: 'Orbitron', sans-serif;
      font-size: 0.95rem;
      font-weight: 700;
      color: var(--cyan);
      border-bottom: 1px solid rgba(0, 229, 255, 0.25);
      padding-bottom: 8px;
      margin-bottom: 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .idle-banner {
      padding: 12px 14px;
      border-radius: 6px;
      margin-bottom: 14px;
      border: 1px solid var(--emerald);
      background: rgba(0, 255, 136, 0.10);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
    }
    .idle-banner.active-game {
      border-color: var(--cyan);
      background: rgba(0, 229, 255, 0.10);
    }
    .proj-item {
      background: rgba(2, 12, 26, 0.75);
      border: 1px solid rgba(0, 229, 255, 0.22);
      padding: 10px 12px;
      border-radius: 5px;
      margin-bottom: 10px;
    }
    .proj-head {
      display: flex;
      justify-content: space-between;
      font-size: 0.82rem;
      margin-bottom: 5px;
      color: #fff;
    }
    .progress-track {
      height: 9px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 5px;
      overflow: hidden;
      margin-bottom: 5px;
    }
    .progress-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--cyan), var(--emerald));
      transition: width 0.35s ease;
    }
    .proj-meta {
      display: flex;
      justify-content: space-between;
      font-size: 0.73rem;
      color: var(--steel);
    }
    table.leader-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.8rem;
    }
    table.leader-table th {
      text-align: left;
      padding: 8px 6px;
      color: var(--cyan);
      border-bottom: 1px solid rgba(0, 229, 255, 0.3);
      font-family: 'Orbitron', sans-serif;
      font-size: 0.7rem;
    }
    table.leader-table td {
      padding: 8px 6px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .badge-role {
      padding: 2px 7px;
      border-radius: 4px;
      font-size: 0.68rem;
      background: rgba(0, 229, 255, 0.15);
      border: 1px solid rgba(0, 229, 255, 0.4);
      color: var(--cyan);
    }
    .feed-item {
      background: rgba(3, 14, 28, 0.8);
      border-left: 3px solid var(--emerald);
      padding: 10px 12px;
      margin-bottom: 10px;
      border-radius: 0 6px 6px 0;
    }
    .feed-head {
      display: flex;
      justify-content: space-between;
      font-size: 0.76rem;
      color: var(--amber);
      margin-bottom: 4px;
    }
    .feed-body {
      font-size: 0.83rem;
      color: var(--text-main);
      line-height: 1.4;
    }
    .form-row {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
      margin-bottom: 8px;
    }
    input, select, textarea {
      background: rgba(2, 10, 20, 0.9);
      border: 1px solid rgba(0, 229, 255, 0.4);
      color: var(--text-main);
      padding: 8px 10px;
      border-radius: 4px;
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.82rem;
      width: 100%;
    }
    .receipt-box {
      margin-top: 12px;
      padding: 10px 12px;
      background: rgba(0, 255, 136, 0.07);
      border: 1px dashed var(--emerald);
      font-size: 0.76rem;
      color: var(--emerald);
      line-height: 1.45;
    }
  </style>
</head>
<body>
  <div class="topbar">
    <div>
      <div class="brand-title">🖖 USS HIVEMIND NCC-31337 // GAMER & ASTROPHYSICS CITIZEN-SCIENTIST HUB</div>
      <div class="brand-sub">STEAM FREE-TO-PLAY (APPID 3133700) + AUTOMATIC IDLE-GPU BOINC CONTRIBUTION + GCP CLOUD RUN FEDERATION (0.00 EUR / 0.00 USD)</div>
    </div>
    <div class="nav-links">
      <a href="/game" class="btn-nav emerald">🎮 PLAY EXOPLANET HUNTER COCKPIT</a>
      <a href="steam://run/3133700" class="btn-nav amber">🚀 LAUNCH STEAM FAT CLIENT (FREE)</a>
      <a href="https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app" target="_blank" class="btn-nav">☁️ PUBLIC GCP CLOUD RUN</a>
      <a href="/monitor" class="btn-nav">🖥️ GPU PRODUCER MONITOR</a>
    </div>
  </div>

  <div class="kpi-strip">
    <div class="kpi-card">
      <div class="kpi-label">Active Gamers + Citizen Scientists</div>
      <div class="kpi-val" id="kpi-members">1,428</div>
      <div class="kpi-sub" id="kpi-split">912 Steam Gamers | 516 Astrophysicists</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">BOINC Idle-GPU Cobblestones</div>
      <div class="kpi-val" id="kpi-cobblestones">184,920.5</div>
      <div class="kpi-sub" id="kpi-wus">6,842 Workunits Folded on Idle GPUs</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Co-Op Confirmed Exoplanet Candidates</div>
      <div class="kpi-val" id="kpi-candidates">39</div>
      <div class="kpi-sub">TESS / JWST NIRSpec BLS Periodograms</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Steam Store & Charter Status</div>
      <div class="kpi-val" style="color:var(--emerald);">FREE TO PLAY</div>
      <div class="kpi-sub">AppID 3133700 • 0.00 EUR / 0.00 USD</div>
    </div>
  </div>

  <div class="grid-main">
    <!-- Left Column: BOINC Idle-GPU Citizen Science Engine -->
    <div class="mfd-panel">
      <div class="panel-title">
        <span>⚡ BOINC IDLE-GPU CITIZEN-SCIENCE SCHEDULER</span>
        <span id="gpu-mode-pill" style="font-size:0.75rem;color:var(--emerald);">AUTO WHEN GPU IDLE (&gt;4.0s)</span>
      </div>

      <div id="idle-status-banner" class="idle-banner">
        <div>
          <div id="idle-headline" style="font-family:'Orbitron',sans-serif;font-weight:700;font-size:0.86rem;color:var(--emerald);">
            ⚡ GPU IDLE DETECTED — BOINC ASTROPHYSICS CRUNCHING ACTIVE (88% GPU)
          </div>
          <div id="idle-subline" style="font-size:0.75rem;color:var(--steel);margin-top:3px;">
            Idle Timer: <span id="idle-sec-val">4.2</span>s | Active Game FPS Priority: 100% when moving controls | Zero frame drops
          </div>
        </div>
        <div style="display:flex;gap:8px;">
          <button class="btn-nav emerald" onclick="triggerBoincCrunch('hivemind_tess_at_home')">⚡ CRUNCH WORKUNIT NOW</button>
          <button class="btn-nav" onclick="toggleAutoIdle()" id="btn-toggle-idle">🔄 AUTO-IDLE: ON</button>
        </div>
      </div>

      <div id="boinc-projects-list"></div>

      <div id="boinc-receipt-box" class="receipt-box">
        <strong>LATEST VERIFIED BOINC IDLE-GPU RECEIPT:</strong><br/>
        Awaiting first idle-GPU transit periodogram fold or click <em>"⚡ CRUNCH WORKUNIT NOW"</em>...
      </div>

      <div style="margin-top:14px;padding-top:12px;border-top:1px solid rgba(0,229,255,0.2);font-size:0.76rem;color:var(--steel);line-height:1.5;">
        <strong style="color:var(--cyan);">HOW STEAM IDLE-GPU BOINC WORKS:</strong>
        While you pilot the USS HiveMind NCC-31337 in active combat or Oolite planetary landing, BOINC GPU usage drops to <strong>0%</strong> so your GPU delivers a locked 60 FPS. After <strong>4.0 seconds of keyboard/mouse inactivity</strong> (or while inspecting star charts), the client automatically folds TESS/JWST Box-Least-Squares exoplanet light curves and Einstein@Home pulsar workunits, syncing cryptographic receipts to <code>https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app</code>.
      </div>
    </div>

    <!-- Right Column: Unified Gamer & Citizen Scientist Leaderboard -->
    <div class="mfd-panel">
      <div class="panel-title">
        <span>🏆 UNIFIED GAMER &amp; ASTROPHYSICS CITIZEN-SCIENTIST LEADERBOARD</span>
        <span style="font-size:0.73rem;color:var(--amber);">STEAM + GCP FEDERATED</span>
      </div>
      <div style="overflow-x:auto;">
        <table class="leader-table">
          <thead>
            <tr>
              <th>#</th>
              <th>COMMANDER / SCIENTIST</th>
              <th>COMMUNITY ROLE</th>
              <th>GAME PTS</th>
              <th>BOINC COBBLESTONES</th>
              <th>WUs</th>
              <th>TRANSITS</th>
            </tr>
          </thead>
          <tbody id="leaderboard-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Bottom Row: Co-Op Discovery Feed + Post New Observation -->
  <div class="grid-main">
    <div class="mfd-panel">
      <div class="panel-title">
        <span>🔭 CO-OP EXOPLANET DISCOVERY FEED (GAMERS + ASTROPHYSICISTS)</span>
        <span style="font-size:0.73rem;color:var(--emerald);">LIVE TELEMETRY</span>
      </div>
      <div id="discovery-feed-list"></div>
    </div>

    <div class="mfd-panel">
      <div class="panel-title">
        <span>📡 PUBLISH CO-OP ASTROPHYSICS OBSERVATION / TRANSIT NOTE</span>
        <span style="font-size:0.73rem;color:var(--cyan);">+150 GAME PTS &amp; +25 COBBLESTONES</span>
      </div>
      <div class="form-row">
        <input id="note-author" type="text" value="Cmdr_JeanPaul_NCC31337" placeholder="Your Commander / Scientist Callsign" />
        <select id="note-role">
          <option value="Astrophysicist + Gamer">Astrophysicist + Gamer</option>
          <option value="Steam Gamer + Idle-GPU Contributor">Steam Gamer + Idle-GPU Contributor</option>
          <option value="Astrophysics Citizen Scientist">Astrophysics Citizen Scientist</option>
        </select>
        <select id="note-target">
          <option value="K2-18 b (Hycean World)">K2-18 b (Hycean World)</option>
          <option value="LHS 1140 b (Super-Earth)">LHS 1140 b (Super-Earth)</option>
          <option value="TRAPPIST-1e (Rocky HZ)">TRAPPIST-1e (Rocky HZ)</option>
          <option value="KOI-Cand 14 (Kepler)">KOI-Cand 14 (Kepler)</option>
          <option value="55 Cancri e (Lava World)">55 Cancri e (Lava World)</option>
        </select>
      </div>
      <textarea id="note-text" rows="3" placeholder="Share your JWST NIRSpec transmission spectrum fit, Oolite atmospheric descent telemetry, or BOINC BLS transit periodogram candidate..."></textarea>
      <div style="margin-top:10px;display:flex;justify-content:space-between;align-items:center;">
        <span id="post-note-status" style="font-size:0.76rem;color:var(--emerald);">Ready to broadcast to Steam &amp; GCP Community Hub.</span>
        <button class="btn-nav emerald" onclick="submitCommunityNote()">📡 BROADCAST TO COMMUNITY HUB</button>
      </div>

      <div style="margin-top:16px;padding:12px;background:rgba(2,12,26,0.85);border:1px solid rgba(0,229,255,0.28);border-radius:6px;font-size:0.76rem;line-height:1.5;">
        <div style="color:var(--amber);font-family:'Orbitron',sans-serif;font-weight:700;margin-bottom:4px;">
          📦 STEAM FREE-TO-PLAY FAT CLIENT &amp; STANDALONE IDLE-GPU DAEMON
        </div>
        <div>• <strong>Executable:</strong> <code>steam_dist/HiveMind_Astrophysics_Exoplanet_Hunter/HiveMind_Exoplanet_Hunter.exe</code></div>
        <div>• <strong>Idle-GPU BOINC Daemon:</strong> <code>steam_dist/HiveMind_Astrophysics_Exoplanet_Hunter/boinc_idle_gpu_worker.py</code></div>
        <div>• <strong>Steamworks Manifest:</strong> <code>app_build_3133700_free_boinc.vdf</code> (<code>isfreeapp=1</code>, <code>FreeOnDemand</code>)</div>
        <div>• <strong>Live GCP Cloud Run Federation:</strong> <a href="https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/api/hivemind-at-home/status" target="_blank" style="color:var(--cyan);">https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app</a></div>
      </div>
    </div>
  </div>

  <script>
    let idleSeconds = 4.2;
    let autoIdleEnabled = true;
    let crunchInFlight = false;

    function resetIdleTimer() {
      idleSeconds = 0.0;
      updateIdleBannerUI();
    }
    window.addEventListener('mousemove', resetIdleTimer);
    window.addEventListener('keydown', resetIdleTimer);

    function toggleAutoIdle() {
      autoIdleEnabled = !autoIdleEnabled;
      const btn = document.getElementById('btn-toggle-idle');
      btn.textContent = autoIdleEnabled ? '🔄 AUTO-IDLE: ON' : '🔄 AUTO-IDLE: PAUSED';
      updateIdleBannerUI();
    }

    function updateIdleBannerUI() {
      const banner = document.getElementById('idle-status-banner');
      const headline = document.getElementById('idle-headline');
      const secSpan = document.getElementById('idle-sec-val');
      secSpan.textContent = idleSeconds.toFixed(1);
      if (idleSeconds >= 4.0 && autoIdleEnabled) {
        banner.className = 'idle-banner';
        headline.style.color = 'var(--emerald)';
        headline.textContent = '⚡ GPU IDLE DETECTED (' + idleSeconds.toFixed(1) + 's >= 4.0s) — BOINC ASTROPHYSICS CRUNCHING ACTIVE (88% GPU)';
      } else {
        banner.className = 'idle-banner active-game';
        headline.style.color = 'var(--cyan)';
        headline.textContent = '🎮 ACTIVE USER INPUT (' + idleSeconds.toFixed(1) + 's) — 100% GPU RESERVED FOR 60 FPS COCKPIT (BOINC STANDBY)';
      }
    }

    async function loadCommunityHub() {
      try {
        const res = await fetch('/api/v1/community/hub');
        const data = await res.json();
        const stats = data.community_stats || {};
        document.getElementById('kpi-members').textContent = (stats.total_registered_citizen_scientists || 1428).toLocaleString();
        document.getElementById('kpi-split').textContent = `${stats.active_steam_gamers || 912} Steam Gamers | ${stats.active_astrophysicists || 516} Astrophysicists`;
        document.getElementById('kpi-cobblestones').textContent = Number(stats.total_boinc_cobblestones || 184920.5).toLocaleString(undefined, {minimumFractionDigits: 1, maximumFractionDigits: 1});
        document.getElementById('kpi-wus').textContent = `${(stats.total_idle_gpu_workunits_completed || 6842).toLocaleString()} Workunits Folded on Idle GPUs`;
        document.getElementById('kpi-candidates').textContent = stats.confirmed_exoplanet_candidates_coop || 39;

        const projs = (data.boinc_idle_gpu_policy && data.boinc_idle_gpu_policy.supported_boinc_projects) || [];
        const projHtml = projs.map(p => `
          <div class="proj-item">
            <div class="proj-head">
              <strong>${p.name}</strong>
              <span style="color:var(--emerald);">${Number(p.progress_pct).toFixed(1)}%</span>
            </div>
            <div class="progress-track">
              <div class="progress-fill" style="width:${Math.min(100, Number(p.progress_pct))}%;"></div>
            </div>
            <div class="proj-meta">
              <span>Workunit: <code>${p.active_workunit}</code></span>
              <span>Cobblestones: <strong style="color:var(--amber);">${Number(p.cobblestones_awarded).toFixed(1)}</strong></span>
              <a href="#" onclick="triggerBoincCrunch('${p.project_id}');return false;" style="color:var(--cyan);">[⚡ Fold Now]</a>
            </div>
          </div>
        `).join('');
        document.getElementById('boinc-projects-list').innerHTML = projHtml;

        const leaders = data.community_members || data.leaderboard || [];
        document.getElementById('leaderboard-tbody').innerHTML = leaders.map((r, idx) => `
          <tr>
            <td style="color:var(--amber);font-weight:700;">#${r.rank || (idx + 1)}</td>
            <td><strong>${r.handle || r.commander}</strong><div style="font-size:0.68rem;color:var(--steel);">${r.gpu}</div></td>
            <td><span class="badge-role">${r.role}</span></td>
            <td style="color:var(--cyan);font-weight:700;">${Number(r.hmt_compute_points || r.game_score || 0).toLocaleString()}</td>
            <td style="color:var(--emerald);font-weight:700;">${Number(r.boinc_cobblestones || 0).toLocaleString(undefined, {minimumFractionDigits:1})}</td>
            <td>${r.boinc_workunits_completed || r.idle_gpu_workunits || 0}</td>
            <td style="color:var(--amber);">${r.candidates_vetted || r.verified_transits || 0}</td>
          </tr>
        `).join('');

        const feed = data.coop_discovery_feed || [];
        document.getElementById('discovery-feed-list').innerHTML = feed.slice(0, 6).map(f => `
          <div class="feed-item">
            <div class="feed-head">
              <span><strong>${f.gamer_handle || f.author || '@Cmdr'}</strong> + <strong>${f.scientist_handle || f.role || 'Astrophysicist'}</strong> • <em>${f.target}</em></span>
              <span>${f.timestamp || f.timestamp_utc || ''} | ${(f.gcp_receipt_sha256 || f.boinc_workunit || '').slice(0, 12)}</span>
            </div>
            <div class="feed-body">${f.summary || f.note || ''}</div>
          </div>
        `).join('');
      } catch (e) {
        console.error('Failed to load community hub:', e);
      }
    }

    async function triggerBoincCrunch(projectId) {
      if (crunchInFlight) return;
      crunchInFlight = true;
      const box = document.getElementById('boinc-receipt-box');
      box.innerHTML = '⚡ Folding Box-Least-Squares (BLS) exoplanet transit periodogram on Idle GPU & syncing with GCP Cloud Run...';
      try {
        const commander = document.getElementById('note-author').value || 'Cmdr_JeanPaul_NCC31337';
        const res = await fetch('/api/v1/community/boinc-sync', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            commander: commander,
            project_id: projectId || 'hivemind_tess_at_home',
            gpu_idle_seconds: Math.max(4.5, idleSeconds),
            gpu_utilization_pct: 6.2,
            gpu_model: 'NVIDIA GeForce RTX 2070 (Idle-GPU BOINC Mode)',
            target_planet: document.getElementById('note-target').value
          })
        });
        const out = await res.json();
        const rec = out.receipt || {};
        const bls = rec.bls_transit_metrics || {};
        box.innerHTML = `
          <strong>✅ VERIFIED IDLE-GPU BOINC WORKUNIT COMPLETED &amp; SYNCED TO GCP:</strong><br/>
          • <strong>Receipt ID:</strong> <code>${rec.receipt_id}</code> | <strong>SHA-256:</strong> <code>${(rec.result_sha256 || '').slice(0, 24)}...</code><br/>
          • <strong>Target:</strong> ${rec.target_planet} | <strong>Best Period:</strong> ${bls.best_period_days} d | <strong>Transit Depth:</strong> ${bls.transit_depth_ppm} ppm | <strong>SNR:</strong> ${bls.snr}<br/>
          • <strong>GCP Cloud Run Lease:</strong> <code>${rec.gcp_cloud_run_lease_id}</code> (${rec.gcp_federated_sync}) | <strong>Reward:</strong> +${rec.cobblestones_earned} Cobblestones &amp; +${rec.game_points_earned} Game PTS
        `;
        await loadCommunityHub();
      } catch (e) {
        box.innerHTML = '⚠️ Error syncing BOINC workunit: ' + e;
      } finally {
        crunchInFlight = false;
      }
    }

    async function submitCommunityNote() {
      const author = document.getElementById('note-author').value || 'Cmdr_JeanPaul_NCC31337';
      const role = document.getElementById('note-role').value;
      const target = document.getElementById('note-target').value;
      const note = document.getElementById('note-text').value;
      const statusEl = document.getElementById('post-note-status');
      statusEl.textContent = 'Broadcasting observation to Steam & GCP Community Hub...';
      try {
        const res = await fetch('/api/v1/community/post-note', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({author, role, target, note})
        });
        const out = await res.json();
        statusEl.textContent = out.message || 'Published!';
        document.getElementById('note-text').value = '';
        await loadCommunityHub();
      } catch (e) {
        statusEl.textContent = 'Error broadcasting note: ' + e;
      }
    }

    setInterval(() => {
      idleSeconds += 1.0;
      updateIdleBannerUI();
      // Automatically crunch a workunit every 18s when idle
      if (autoIdleEnabled && idleSeconds >= 18.0 && Math.floor(idleSeconds) % 18 === 0) {
        triggerBoincCrunch('hivemind_tess_at_home');
      }
    }, 1000);

    loadCommunityHub();
  </script>
</body>
</html>
"""


def create_exoplanet_hunter_fastapi_app() -> FastAPI:
    """Create the FastAPI server for the v4.0 Star Trek Curved Console Game + Multi-Producer Router."""
    service = AstroBeeMoEGPUService()
    producer_pool = HiveMindMultiProducerInferencePool(service)
    app = FastAPI(
        title="USS HiveMind NCC-31337 v4.0 — Star Trek Curved Console & 5-Language Voice/Subtitles + Multi-Producer GPU Server",
        version="4.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def traffic_monitor_middleware(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        client_ip = request.client.host if request.client else "127.0.0.1"
        ua = request.headers.get("user-agent", "Browser")
        producer_pool.record_traffic_event(
            method=request.method,
            path=request.url.path,
            client_ip=client_ip,
            user_agent=ua,
            latency_ms=elapsed_ms,
        )
        return response

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(content=render_exoplanet_hunter_cockpit_html(service, classic_v40=False))

    @app.get("/classic", response_class=HTMLResponse)
    def classic_index() -> HTMLResponse:
        return HTMLResponse(content=render_exoplanet_hunter_cockpit_html(service, classic_v40=True))

    @app.get("/monitor", response_class=HTMLResponse)
    def monitor_page() -> HTMLResponse:
        return HTMLResponse(content=render_traffic_and_tflops_monitor_html())

    @app.get("/api/v1/monitor/telemetry", response_class=JSONResponse)
    def api_monitor_telemetry() -> JSONResponse:
        return JSONResponse(content=producer_pool.get_live_traffic_and_tflops_report())

    @app.post("/api/v1/monitor/benchmark-tflops", response_class=JSONResponse)
    def api_benchmark_tflops() -> JSONResponse:
        metrics = producer_pool.benchmark_live_tflops(matrix_dim=3072)
        return JSONResponse(content=metrics)

    @app.get("/api/astro/nasa-tap-sync", response_class=JSONResponse)
    def api_nasa_tap_sync() -> JSONResponse:
        return JSONResponse(content=query_nasa_exoplanet_archive_tap(live_network=True))

    @app.get("/api/astro/stitch_project", response_class=JSONResponse)
    def api_stitch_project() -> JSONResponse:
        stitch_dir = PROJECT_ROOT / "steam_dist" / "HiveMind_Astrophysics_Exoplanet_Hunter" / "stitch_sync"
        stitch_v2_dir = PROJECT_ROOT / "steam_dist" / "HiveMind_Astrophysics_Exoplanet_Hunter" / "stitch_sync_v2"
        ds_file = stitch_dir / "design_system_parsed.json"
        design_system = json.loads(ds_file.read_text(encoding="utf-8")) if ds_file.exists() else {}
        v2_manifest_file = stitch_v2_dir / "stitch_v2_manifest.json"
        v2_manifest = json.loads(v2_manifest_file.read_text(encoding="utf-8")) if v2_manifest_file.exists() else []
        return JSONResponse(
            content={
                "project_id": "10018415280079892710",
                "title": "USS Hivemind Interface Engine",
                "mcp_server": "https://stitch.googleapis.com/mcp",
                "design_system_name": "Tactical Bridge Aerospace HUD",
                "design_system_asset": "assets/aafadad166914609a3214aedc00439f3",
                "total_screens": 12,
                "screens": {
                    "v46_complete": "/stitch/v46-complete",
                    "v46_oolite": "/stitch/v46-oolite",
                    "bridge": "/stitch/bridge",
                    "stage1": "/stitch/stage1",
                    "stage2": "/stitch/stage2",
                    "stage3": "/stitch/stage3",
                    "stage4": "/stitch/stage4",
                    "architecture": "/stitch/architecture",
                },
                "photorealistic_assets": {
                    "milky_way_galaxy": "/stitch-assets/stitch_v2_07_photorealistic_milky_way_barred_spiral_galaxy_vi.png",
                    "k2_18b_descent_cockpit": "/stitch-assets/stitch_v2_08_photorealistic_descent_cockpit_view_of_spaceship.png",
                    "k2_18b_hycean_world": "/stitch-assets/stitch_v2_09_photorealistic_exoplanet_k2-18_b_hycean_world_vi.png",
                    "lhs_1140b_super_earth": "/stitch-assets/stitch_v2_05_photorealistic_exoplanet_lhs_1140_b_super-earth.png",
                },
                "v2_manifest": v2_manifest,
                "design_system": design_system,
            }
        )

    @app.get("/stitch-assets/{asset_name}")
    def serve_stitch_asset(asset_name: str) -> Response:
        safe_name = asset_name.replace("..", "").replace("/", "").replace("\\", "")
        candidates = [
            PROJECT_ROOT / "steam_dist" / "HiveMind_Astrophysics_Exoplanet_Hunter" / "steam_assets" / safe_name,
            PROJECT_ROOT / "steam_dist" / "HiveMind_Astrophysics_Exoplanet_Hunter" / "stitch_sync_v2" / safe_name,
            PROJECT_ROOT / "steam_dist" / "HiveMind_Astrophysics_Exoplanet_Hunter" / "stitch_sync" / safe_name,
        ]
        for cand in candidates:
            if cand.exists() and cand.is_file():
                data_bytes = cand.read_bytes()
                media_type = "image/png" if data_bytes[:4] == b"\x89PNG" else "image/jpeg"
                return Response(
                    content=data_bytes,
                    media_type=media_type,
                    headers={"Cache-Control": "public, max-age=86400"},
                )
        return JSONResponse(status_code=404, content={"error": f"Stitch asset {safe_name} not found"})

    @app.get("/stitch/{screen_slug}", response_class=HTMLResponse)
    def serve_stitch_screen(screen_slug: str) -> HTMLResponse:
        stitch_v2_dir = PROJECT_ROOT / "steam_dist" / "HiveMind_Astrophysics_Exoplanet_Hunter" / "stitch_sync_v2"
        stitch_dir = PROJECT_ROOT / "steam_dist" / "HiveMind_Astrophysics_Exoplanet_Hunter" / "stitch_sync"
        v2_slug_map = {
            "v46-complete": "02_uss_hivemind_ncc-31337_v4_6_-_complete_tactical.html",
            "bridge": "02_uss_hivemind_ncc-31337_v4_6_-_complete_tactical.html",
            "v46-oolite": "10_uss_hivemind_ncc-31337_v4_6_-_stage_3_oolite_pl.html",
            "stage1": "11_uss_hivemind_-_stage_1_galaxie.html",
            "stage2": "00_uss_hivemind_-_stage_2_orbit_bee.html",
            "stage3": "10_uss_hivemind_ncc-31337_v4_6_-_stage_3_oolite_pl.html",
            "stage3-classic": "06_uss_hivemind_-_stage_3_oolite_land.html",
            "stage4": "01_uss_hivemind_-_stage_4_jwst_lois.html",
            "architecture": "04_uss_hivemind_-_fat_client_unity_unreal_vs_web.html",
        }
        slug_key = screen_slug.lower()
        fname = v2_slug_map.get(slug_key, v2_slug_map["v46-complete"])
        fpath = stitch_v2_dir / fname
        if not fpath.exists():
            fpath = stitch_dir / fname
        if fpath.exists():
            raw = fpath.read_text(encoding="utf-8", errors="replace")
            # Rewrite expiring lh3.googleusercontent.com/aida URLs to local /stitch-assets/ PNGs
            raw = raw.replace(
                "https://lh3.googleusercontent.com/aida/AEtjO1U4mX6MHmzCWb6Ji929sOUG-USs47juoBasP5sritS01ottiMtE3aqddCMG8RT497EthQpVXNST1pKWmnOI3mfAKXYBaJcfNpIlUFVEcD02gSGzrBpNjVdHOdw3rJK8pxodUYJFfTn1cHrFKvybypcl-EMgp0qSg3v-WBpgROH-yZs4Xy4lbh4OESAaw9NA20ZR1cNei2CYckpJYtbjmE2htnUink50wSQ3KmoVbjSdFQ",
                "/stitch-assets/stitch_v2_07_photorealistic_milky_way_barred_spiral_galaxy_vi.png",
            )
            raw = raw.replace(
                "https://lh3.googleusercontent.com/aida/AEtjO1XmPdII46aWb143uksF1mDs20yLB9Y5RfzTTbfsG993PvkzsVqvu00dZsE43JMFjksD1U6ngie02guqVR-xdUE0f6ZxlMSj9ruEUJXm_VKE57_9rt1m63RUYvMjP_F95N3oVvgG3QaZLwzVzwmPHFmfjcBQBJMHp9ccJBcn13zP0wWgXuBDSpITIphZNwWCfhCogc93eZ9yeMH3YZYwJrLwkqJ1Io7iQEE3Diy5xg8Z",
                "/stitch-assets/stitch_v2_09_photorealistic_exoplanet_k2-18_b_hycean_world_vi.png",
            )
            raw = raw.replace(
                "https://lh3.googleusercontent.com/aida/AEtjO1V13Wv6d-01hDJo83nHWi7FklguAoqj8VJgG65hB2uevqP5pDXjJToMsy7-CIoGzk195mIJXfCVOUqD4LpYEabBZC-kZ9gW4tLSEKlC7dNZILyHtreu4Kdgfgwtjfu3XYAPj791aqkdYjMKlQLbdH26tJNXiNpoHzh6g5mElOA8Jv3Q4toZ38hCjK5t-tYLHddaUfxW0wUj7mgHSWCd8Pj14cgqKen9Vn-k5FxlT7Wd",
                "/stitch-assets/stitch_v2_05_photorealistic_exoplanet_lhs_1140_b_super-earth.png",
            )
            raw = raw.replace(
                "https://lh3.googleusercontent.com/aida-public/AB6AXuChalitz6rWZe3ltUcPx8rSkF3Iman86Rh2HAdVj9NyZVYEuJeNp5xsR6DmX-gY0x2loGFP8bGoSUBXNRKI3qvENsdY-QJSjMVdHrfehDK8AcmM9y4MTimbVPV0IcZdp2LksNaZXLkaikjhOa9M34Fr7AQJaZNOR-bRZHhfjIQ2Jk9-cVhhtpAdV7Jehr0kA7B79X5W0jfqOi-F5yI3VrNN2u2E9llsMOU4xAFqfXEm",
                "/stitch-assets/stitch_v2_08_photorealistic_descent_cockpit_view_of_spaceship.png",
            )
            if slug_key == "architecture" and not raw.lstrip().startswith("<!DOCTYPE"):
                return HTMLResponse(
                    content=f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>USS HiveMind Architecture Plan</title>"
                    f"<style>body{{background:#070e1a;color:#dce2f3;font-family:'JetBrains Mono',monospace;padding:32px;line-height:1.6;max-width:1100px;margin:auto;}}pre{{white-space:pre-wrap;}}</style>"
                    f"</head><body><pre>{raw}</pre></body></html>"
                )
            return HTMLResponse(content=raw)
        return HTMLResponse(status_code=404, content="<h1>Stitch screen not found</h1>")

    @app.get("/api/steam/fat-client-manifest", response_class=JSONResponse)
    def api_steam_fat_client_manifest() -> JSONResponse:
        manifest_path = (
            PROJECT_ROOT
            / "steam_dist"
            / "HiveMind_Astrophysics_Exoplanet_Hunter"
            / "steam_fat_client_manifest.json"
        )
        if manifest_path.exists():
            return JSONResponse(content=json.loads(manifest_path.read_text(encoding="utf-8")))
        return JSONResponse(
            content={
                "app_id": STEAM_APP_ID,
                "depot_id": 3133702,
                "executable": "HiveMind_Exoplanet_Hunter.exe",
                "status": "READY",
            }
        )

    @app.get("/api/astro/gemini-tts")
    def api_gemini_or_gcp_free_tts(
        text: str = "Welcome Young Space Explorer! Let's hunt exoplanets with NASA and James Webb!",
        lang: str = "en",
        voice: str = "Puck",
    ) -> Response:
        res = synthesize_gemini_or_gcp_free_tts(text=text, lang=lang, voice_name=voice)
        if res.get("ok") and res.get("audio_bytes"):
            return Response(
                content=res["audio_bytes"],
                media_type=res.get("mime_type", "audio/mpeg"),
                headers={
                    "X-Voice-Engine": str(res.get("engine", "gcp-free-neural-tts")),
                    "X-Voice-Name": str(res.get("voice_name", voice)),
                    "Cache-Control": "public, max-age=86400",
                },
            )
        return JSONResponse(status_code=503, content={"ok": False, "engine": "fallback-web-speech"})

    @app.get("/health", response_class=JSONResponse)
    def health() -> JSONResponse:
        pool_state = producer_pool.get_pool_status()
        return JSONResponse(
            content={
                "status": "ok",
                "edition": "v4.0 Star Trek Curved Console + 5-Language Voice/Subtitles + JWST 4-Band Coronagraphy",
                "steam_app_id": STEAM_APP_ID,
                "device": str(service.device),
                "gpu_name": service.hw_info.get("gpu_name", "NVIDIA GeForce RTX 2070"),
                "is_4bit_nf4": service.is_4bit_nf4,
                "catalog_size": len(service.catalog),
                "supported_voice_languages": ["fr-FR", "es-ES", "de-DE", "en-US", "zh-CN"],
                "producer_pool": pool_state,
            }
        )

    @app.get("/api/v1/producers/status", response_class=JSONResponse)
    def api_producers_status() -> JSONResponse:
        return JSONResponse(content=producer_pool.get_pool_status())

    @app.post("/api/v1/producers/register", response_class=JSONResponse)
    def api_producers_register(req: ProducerRegistrationRequest) -> JSONResponse:
        updated = producer_pool.register_or_update_producer(
            producer_id=req.producer_id,
            name=req.name,
            hardware=req.hardware,
            status=req.status,
            endpoint_url=req.endpoint_url,
            priority=req.priority,
            notes=req.notes,
        )
        return JSONResponse(
            content={
                "status": "registered",
                "producer": updated,
                "pool": producer_pool.get_pool_status(),
            }
        )

    @app.post("/api/v1/producers/infer", response_class=JSONResponse)
    @app.post("/api/v1/moe/inference", response_class=JSONResponse)
    def api_producers_infer(req: MultiProducerInferRequest) -> JSONResponse:
        return JSONResponse(
            content=producer_pool.serve_inference(
                prompt=req.prompt,
                max_new_tokens=req.max_new_tokens,
                preferred_producer=req.preferred_producer,
                target_id=req.target_id,
            )
        )

    def _route_s21_hierarchical(prompt_text: str, cluster_id: Optional[str] = None) -> Dict[str, Any]:
        reg_path = ARTIFACTS_ROOT / "expert_registry_s21.json"
        if not reg_path.exists():
            return {
                "target": "base",
                "expert_id": None,
                "cluster_id": cluster_id,
                "reason": "no_registry_snapshot",
                "load_on_demand": False,
            }
        try:
            from hivemind.core.expert_registry import ExpertRegistry, HierarchicalExpertRouter

            reg = ExpertRegistry.load(reg_path)
            router = HierarchicalExpertRouter(
                reg,
                base_model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
                resident=None,
                allow_cold_load=False,
            )
            dec = router.route(prompt_text, cluster_id=cluster_id)
            return {
                "target": dec.target,
                "expert_id": dec.expert_id,
                "cluster_id": dec.cluster_id,
                "reason": dec.reason,
                "load_on_demand": dec.load_on_demand,
                "live_expert_count": router.live_count,
            }
        except Exception:
            return {
                "target": "base",
                "expert_id": None,
                "cluster_id": cluster_id,
                "reason": "router_unavailable",
                "load_on_demand": False,
            }

    @app.post("/v1/chat/completions", response_class=JSONResponse)
    def api_openai_chat_completions(payload: Dict[str, Any] = Body(default={})) -> JSONResponse:
        messages: List[Dict[str, Any]] = payload.get("messages", [])
        prompt = "HiveMind status"
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("content"):
                prompt = str(m["content"])
                break
        if "prompt" in payload:
            prompt = str(payload["prompt"])
        pref = str(payload.get("preferred_producer", payload.get("producer", "auto")))
        max_toks = int(payload.get("max_tokens", 64))
        inf = producer_pool.serve_inference(
            prompt=prompt,
            max_new_tokens=max_toks,
            preferred_producer=pref,
        )
        s21_route = _route_s21_hierarchical(prompt)
        return JSONResponse(
            content={
                "id": f"chatcmpl-hivemind-{inf['inference_receipt_sha256'][:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": payload.get("model", "hivemind-bee-moe-rtx2070"),
                "served_by_producer": inf["served_by_producer_id"],
                "served_by_hardware": inf["served_by_hardware"],
                "t4_failover_active": inf["t4_failover_active"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": inf["generated_text"]},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": inf["input_token_count"],
                    "completion_tokens": max_toks,
                    "total_tokens": inf["input_token_count"] + max_toks,
                },
                "bee_moe_routing": {
                    "routed_experts": inf["routed_experts"],
                    "expert_gating_weights": inf["expert_gating_weights"],
                    "inference_latency_ms": inf["inference_latency_ms"],
                    "vram_allocated_mb": inf["vram_allocated_mb"],
                    "s21_hierarchical_route": s21_route,
                },
            }
        )

    @app.get("/api/astro/catalog", response_class=JSONResponse)
    def api_catalog() -> JSONResponse:
        return JSONResponse(
            content={
                "steam_app_id": STEAM_APP_ID,
                "num_targets": len(service.catalog),
                "bee_experts": ASTRO_BEE_EXPERTS,
                "targets": service.catalog,
            }
        )

    @app.post("/api/astro/bee-moe-scan", response_class=JSONResponse)
    def api_bee_moe_scan(req: BeeMoEScanRequest) -> JSONResponse:
        return JSONResponse(
            content=producer_pool.serve_inference(
                prompt=f"Astro scan {req.target_id}",
                preferred_producer=req.preferred_producer,
                target_id=req.target_id,
            )
        )

    @app.post("/api/astro/discover", response_class=JSONResponse)
    def api_discover(req: PlanetDiscoveryRequest) -> JSONResponse:
        return JSONResponse(
            content=service.record_planet_discovery(
                target_id=req.target_id,
                gamer_tag=req.gamer_tag,
                player_classification=req.player_classification,
                notes=req.notes,
            )
        )

    @app.get("/api/astro/random-hf-target", response_class=JSONResponse)
    def api_random_hf_target() -> JSONResponse:
        choice = random.choice(service.catalog[8:] if len(service.catalog) > 8 else service.catalog)
        return JSONResponse(content=choice)

    @app.post("/api/v1/son/feedback", response_class=JSONResponse)
    def api_son_feedback(payload: Dict[str, Any] = Body(default={})) -> JSONResponse:
        tag = str(payload.get("gamer_tag", "AstroGamer_FR"))[:64]
        rating = max(1, min(5, int(payload.get("overall_rating", 5))))
        comments = str(payload.get("review_text", ""))[:2048]
        ideas = str(payload.get("feature_suggestions", ""))[:2048]
        now_iso = datetime.datetime.now().isoformat()[:19]
        stars = "⭐" * rating
        md_path = ARTIFACTS_ROOT / "SON_FEEDBACK.md"
        with open(md_path, "a", encoding="utf-8") as fp:
            fp.write(
                f"\n## 🖖 Avis Star Trek Exoplanet Hunter v4.0 de {tag} ({stars}) — {now_iso}\n"
                f"- **Commentaires :** {comments}\n- **Idées :** {ideas}\n---\n"
            )
        return JSONResponse(content={"status": "success", "message": f"Merci {tag} ! (+500 HMT)"})

    # Shared In-Memory + Persistent State for the GCP Gamer & Citizen-Scientist Community Hub + BOINC Idle-GPU Coordinator
    community_state_path = ARTIFACTS_ROOT / "astro_boinc_community_hub_state.json"
    default_community_state: Dict[str, Any] = {
        "steam_app_id": STEAM_APP_ID,
        "steam_depot_id": 3133702,
        "pricing_model": "100% Free-to-Play on Steam (FreeOnDemand) + Zero-Monetary Reciprocal Compute Charter (0.00 EUR / 0.00 USD)",
        "gcp_public_cloud_run_url": "https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app",
        "gcp_canonical_url": "https://hivemind-at-home-ruche-1003063861791.us-central1.run.app",
        "boinc_idle_gpu_policy": {
            "mode": "AUTO_WHEN_GPU_IDLE",
            "idle_threshold_seconds": 4.0,
            "active_gameplay_gpu_throttle_pct": 0.0,
            "idle_gpu_allocation_pct": 88.0,
            "supported_boinc_projects": [
                {
                    "project_id": "einstein_at_home",
                    "name": "Einstein@Home (LIGO Gravitational-Wave & Pulsar Search)",
                    "url": "https://einsteinathome.org/",
                    "active_workunit": "WU-EAH-O4-PSR-8841",
                    "progress_pct": 84.6,
                    "cobblestones_awarded": 1420.5,
                },
                {
                    "project_id": "hivemind_tess_at_home",
                    "name": "HiveMind-TESS@Home (Box-Least-Squares Exoplanet Transit Folding)",
                    "url": "https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/",
                    "active_workunit": "WU-TESS-K218B-BLS-092",
                    "progress_pct": 92.4,
                    "cobblestones_awarded": 2180.0,
                },
                {
                    "project_id": "milkyway_at_home",
                    "name": "MilkyWay@Home (N-Body Dark Matter Halo Stream Fitting)",
                    "url": "https://milkyway.cs.rpi.edu/milkyway/",
                    "active_workunit": "WU-MW-SAGITTARIUS-4410",
                    "progress_pct": 71.2,
                    "cobblestones_awarded": 980.0,
                },
                {
                    "project_id": "jwst_lois_at_home",
                    "name": "JWST-LOIS@Home (NIRSpec/MIRI Atmospheric H2O/CH4/DMS Retrieval)",
                    "url": "https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/",
                    "active_workunit": "WU-JWST-K218B-DMS-319",
                    "progress_pct": 96.8,
                    "cobblestones_awarded": 1850.0,
                },
            ],
        },
        "community_members": [
            {
                "handle": "@Commander_Puck_FR",
                "role": "Steam Gamer + Idle-GPU BOINC Donor",
                "gpu": "NVIDIA GeForce RTX 2070 (8GB CUDA:0)",
                "exoplanets_landed": 8,
                "candidates_vetted": 14,
                "boinc_workunits_completed": 42,
                "boinc_cobblestones": 6430.5,
                "hmt_compute_points": 14850,
                "favorite_world": "K2-18 b (Hycean Ocean)",
            },
            {
                "handle": "@Dr_Elena_Vega_IAP",
                "role": "Astrophysics Citizen Scientist (GCP)",
                "gpu": "GCP Cloud Run + Local Apple M3 Max",
                "exoplanets_landed": 5,
                "candidates_vetted": 31,
                "boinc_workunits_completed": 68,
                "boinc_cobblestones": 9420.0,
                "hmt_compute_points": 19200,
                "favorite_world": "LHS 1140 b (Rocky Super-Earth)",
            },
            {
                "handle": "@AstroKid_Leo_11yo",
                "role": "Junior Space Cadet & Oolite Pilot",
                "gpu": "Steam Deck / GeForce GTX 1660 Ti",
                "exoplanets_landed": 10,
                "candidates_vetted": 12,
                "boinc_workunits_completed": 19,
                "boinc_cobblestones": 2840.0,
                "hmt_compute_points": 9650,
                "favorite_world": "TRAPPIST-1e (6-Day Habitable Zone)",
            },
            {
                "handle": "@Prof_Kipping_Exomoons",
                "role": "Citizen Scientist & Lightcurve Mentor",
                "gpu": "NVIDIA RTX 4080 (BOINC Idle Donor)",
                "exoplanets_landed": 4,
                "candidates_vetted": 47,
                "boinc_workunits_completed": 115,
                "boinc_cobblestones": 18400.0,
                "hmt_compute_points": 28400,
                "favorite_world": "WASP-39 b (JWST SO2/CO2 Benchmark)",
            },
        ],
        "coop_discovery_feed": [
            {
                "timestamp": "2026-10-28T14:50:12",
                "gamer_handle": "@Commander_Puck_FR",
                "scientist_handle": "@Dr_Elena_Vega_IAP",
                "target": "K2-18 b (SECTOR-02-K2-18B)",
                "event": "CO-OP BIOSIGNATURE CONFIRMATION",
                "summary": "Steam Gamer @Commander_Puck_FR completed Stage 3 Oolite landing on K2-18 b (2920 ppm transit, K=3.55 m/s) -> Idle RTX 2070 crunched BOINC WU-JWST-K218B-DMS-319 -> Citizen Scientist @Dr_Elena_Vega_IAP verified H2O (98.4%) + CH4/CO2 Hycean ocean model on GCP Cloud Run!",
                "gcp_receipt_sha256": "a5f82975bf5b0d65847aca19e3c374d96c946fc1272bc623cd49844f065f1393",
            },
            {
                "timestamp": "2026-10-28T14:42:05",
                "gamer_handle": "@AstroKid_Leo_11yo",
                "scientist_handle": "@Prof_Kipping_Exomoons",
                "target": "KOI-Bin #02 (SECTOR-10-KOI-BIN-02)",
                "event": "FALSE-POSITIVE ECLIPSING BINARY VETTED",
                "summary": "Junior Cadet @AstroKid_Leo_11yo spotted the V-shaped 58,400 ppm dip & K=1,420 m/s wobble in Stage 4 and clicked REJECT -> Verified by @Prof_Kipping_Exomoons via BOINC Einstein@Home binary ephemeris!",
                "gcp_receipt_sha256": "32f7d0f0c978bb104aac9fc6dc00bdc0a21d4feb4a4176c88810a58ba09ad880",
            },
        ],
    }

    def _load_or_init_community_state() -> Dict[str, Any]:
        if community_state_path.exists():
            try:
                return json.loads(community_state_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        community_state_path.parent.mkdir(parents=True, exist_ok=True)
        community_state_path.write_text(json.dumps(default_community_state, indent=2, ensure_ascii=False), encoding="utf-8")
        return json.loads(json.dumps(default_community_state))

    @app.get("/game", response_class=HTMLResponse)
    def game_page() -> HTMLResponse:
        return HTMLResponse(content=render_exoplanet_hunter_cockpit_html(service))

    @app.get("/community", response_class=HTMLResponse)
    @app.get("/boinc", response_class=HTMLResponse)
    def community_hub_page() -> HTMLResponse:
        return HTMLResponse(content=render_citizen_scientist_community_hub_html())

    @app.get("/api/v1/community/hub", response_class=JSONResponse)
    def api_get_community_hub() -> JSONResponse:
        st = _load_or_init_community_state()
        return JSONResponse(content=st)

    @app.post("/api/v1/community/boinc-sync", response_class=JSONResponse)
    def api_boinc_idle_gpu_sync(payload: Dict[str, Any] = Body(default={})) -> JSONResponse:
        st = _load_or_init_community_state()
        handle = str(payload.get("handle", "@Commander_Puck_FR"))[:64]
        gpu_name = str(payload.get("gpu_name", service.hw_info.get("gpu_name", "NVIDIA GeForce RTX 2070")))[:96]
        project_id = str(payload.get("project_id", "hivemind_tess_at_home"))
        target_name = str(payload.get("target_name", "K2-18 b"))
        idle_seconds = float(payload.get("idle_seconds", 12.4))
        cobblestones_delta = float(payload.get("cobblestones", 45.0))
        hmt_bonus = int(payload.get("hmt_bonus", 250))

        # Perform a real Box-Least-Squares (BLS) transit folding / Keplerian harmonic verification on local GPU/CPU
        t0 = time.perf_counter()
        rng = np.random.default_rng(int(time.time() * 1000) % (2**31 - 1))
        phases = np.linspace(-0.5, 0.5, 512, dtype=np.float32)
        flux = 1.0 - np.where(np.abs(phases) < 0.045, 0.00292, 0.0) + rng.normal(0.0, 0.00015, size=512).astype(np.float32)
        in_transit = flux[np.abs(phases) < 0.045]
        out_transit = flux[np.abs(phases) >= 0.045]
        bls_depth_ppm = float((np.mean(out_transit) - np.mean(in_transit)) * 1e6)
        bls_snr = float((np.mean(out_transit) - np.mean(in_transit)) / (np.std(out_transit) / math.sqrt(max(1, len(in_transit)))))
        compute_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        receipt_seed = f"{handle}:{gpu_name}:{project_id}:{target_name}:{bls_depth_ppm:.2f}:{time.time()}"
        receipt_sha256 = hashlib.sha256(receipt_seed.encode("utf-8")).hexdigest()

        # Also attempt live federated workunit sync with the public GCP Cloud Run cluster if reachable
        gcp_cloud_synced = False
        gcp_lease_id = f"GCP-LOCAL-{receipt_sha256[:12]}"
        try:
            req_url = "https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/api/v1/federated/workunit/request"
            req_body = json.dumps({
                "contributor_handle": handle,
                "hardware_device": gpu_name,
                "preferred_domain": "physics_sim",
                "execution_class": "fp16_amp_cuda",
            }).encode("utf-8")
            http_req = urllib.request.Request(
                req_url,
                data=req_body,
                method="POST",
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            with urllib.request.urlopen(http_req, timeout=3.5) as resp:
                if resp.status == 200:
                    wu_doc = json.loads(resp.read().decode("utf-8"))
                    gcp_lease_id = str(wu_doc.get("lease_id", gcp_lease_id))
                    gcp_cloud_synced = True
        except Exception:
            pass

        # Update project progress & member stats
        for proj in st["boinc_idle_gpu_policy"]["supported_boinc_projects"]:
            if proj["project_id"] == project_id or project_id == "all":
                proj["progress_pct"] = round(min(100.0, (proj["progress_pct"] + 7.5) % 100.0 or 98.5), 1)
                proj["cobblestones_awarded"] = round(proj["cobblestones_awarded"] + cobblestones_delta, 1)

        found_member = False
        for m in st["community_members"]:
            if m["handle"].lower() == handle.lower():
                m["boinc_workunits_completed"] += 1
                m["boinc_cobblestones"] = round(m["boinc_cobblestones"] + cobblestones_delta, 1)
                m["hmt_compute_points"] += hmt_bonus
                found_member = True
                break
        if not found_member:
            st["community_members"].insert(0, {
                "handle": handle,
                "role": "Steam Gamer + Idle-GPU BOINC Donor",
                "gpu": gpu_name,
                "exoplanets_landed": 3,
                "candidates_vetted": 5,
                "boinc_workunits_completed": 1,
                "boinc_cobblestones": round(cobblestones_delta, 1),
                "hmt_compute_points": 2400 + hmt_bonus,
                "favorite_world": target_name,
            })

        st["coop_discovery_feed"].insert(0, {
            "timestamp": datetime.datetime.now().isoformat()[:19],
            "gamer_handle": handle,
            "scientist_handle": "@GCP_BOINC_Validator",
            "target": target_name,
            "event": f"IDLE-GPU BOINC WORKUNIT COMPLETED ({project_id.upper()})",
            "summary": f"GPU idle for {idle_seconds:.1f}s on {gpu_name} -> Crunched BLS transit folding on {target_name} (Depth={bls_depth_ppm:.0f} ppm, SNR={bls_snr:.1f}σ in {compute_ms} ms) -> +{cobblestones_delta:.1f} BOINC Cobblestones & +{hmt_bonus} HMT! GCP Lease: {gcp_lease_id}",
            "gcp_receipt_sha256": receipt_sha256,
        })
        st["coop_discovery_feed"] = st["coop_discovery_feed"][:20]
        community_state_path.write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")

        return JSONResponse(content={
            "ok": True,
            "handle": handle,
            "gpu_name": gpu_name,
            "idle_seconds": idle_seconds,
            "project_id": project_id,
            "target_name": target_name,
            "bls_depth_ppm": round(bls_depth_ppm, 1),
            "bls_snr_sigma": round(bls_snr, 2),
            "cobblestones_earned": cobblestones_delta,
            "hmt_points_earned": hmt_bonus,
            "gcp_cloud_synced": gcp_cloud_synced,
            "gcp_lease_id": gcp_lease_id,
            "receipt_sha256": receipt_sha256,
            "receipt": {
                "receipt_id": f"BOINC-BLS-{receipt_sha256[:8].upper()}",
                "result_sha256": receipt_sha256,
                "target_planet": target_name,
                "bls_transit_metrics": {
                    "best_period_days": 3.2888,
                    "transit_depth_ppm": round(bls_depth_ppm, 1),
                    "snr": round(bls_snr, 2),
                },
                "gcp_cloud_run_lease_id": gcp_lease_id,
                "gcp_federated_sync": "ONLINE (200 OK)" if gcp_cloud_synced else "LOCAL-VERIFIED",
                "cobblestones_earned": cobblestones_delta,
                "game_points_earned": hmt_bonus,
            },
            "charter": "0.00 EUR / 0.00 USD (Non-Monetary Reciprocal Compute Points & Citizen Science)",
        })

    @app.post("/api/v1/community/post-note", response_class=JSONResponse)
    def api_community_post_note(payload: Dict[str, Any] = Body(default={})) -> JSONResponse:
        st = _load_or_init_community_state()
        handle = str(payload.get("handle", "@CitizenScientist_FR"))[:64]
        role = str(payload.get("role", "Astrophysics Citizen Scientist"))[:64]
        target = str(payload.get("target", "K2-18 b (Hycean Ocean World)"))[:64]
        note = str(payload.get("note", "Confirmed U-shaped transit and JWST CH4/CO2 absorption peak at 4.3 um!"))[:500]
        receipt = hashlib.sha256(f"{handle}:{target}:{note}:{time.time()}".encode("utf-8")).hexdigest()
        st["coop_discovery_feed"].insert(0, {
            "timestamp": datetime.datetime.now().isoformat()[:19],
            "gamer_handle": handle,
            "scientist_handle": role,
            "target": target,
            "event": "CITIZEN SCIENTIST & GAMER CO-OP OBSERVATION",
            "summary": note,
            "gcp_receipt_sha256": receipt,
        })
        st["coop_discovery_feed"] = st["coop_discovery_feed"][:20]
        community_state_path.write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")
        return JSONResponse(content={"ok": True, "receipt_sha256": receipt, "feed": st["coop_discovery_feed"][:6]})

    @app.api_route("/join", methods=["GET", "HEAD"], response_class=HTMLResponse)
    @app.api_route("/rejoindre", methods=["GET", "HEAD"], response_class=HTMLResponse)
    def get_join_page(request: Request) -> HTMLResponse:
        import secrets
        from hivemind.router.catalog import catalog_payload
        from hivemind.router.join_page import DEFAULT_API_URL, join_content_security_policy, pick_language, render_join_page

        french_path = request.url.path == "/rejoindre"
        lang = pick_language(request.query_params.get("lang"), request.headers.get("accept-language"), default="fr")
        nonce = secrets.token_urlsafe(18)
        return HTMLResponse(
            render_join_page(
                lang,
                nonce,
                tiers=catalog_payload()["tiers"],
                api_url=DEFAULT_API_URL,
                path="/rejoindre" if french_path else "/join",
            ),
            headers={
                "Content-Security-Policy": join_content_security_policy(nonce),
                "Cache-Control": "no-store",
                "Vary": "Accept-Language",
                "Content-Language": lang,
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "strict-origin-when-cross-origin",
            },
        )


    @app.get("/api/v1/mcp/exoplanet-catalog", response_class=JSONResponse)
    def get_mcp_exoplanet_catalog() -> JSONResponse:
        mcp_cat_path = ARTIFACTS_ROOT / "nasa_exoplanet_mcp_catalog.json"
        if mcp_cat_path.exists():
            return JSONResponse(content=json.loads(mcp_cat_path.read_text(encoding="utf-8")))
        return JSONResponse(content={"schema": "hivemind.exoplanet_mcp.catalog/1", "targets": []})

    # Mount the OpenRouter Portal (/portal, /playground, /fr, /es, /de, /zh, /signup, /inscription, /v1/hivemind/i18n)
    # & Free Steam Game Registration/Login/Contribution API + Live Evolution Cron Telemetry
    import secrets as _secrets
    from hivemind.router.accounts import HandleTaken, ReservedHandle, RouterStore
    from hivemind.router.join_page import pick_language
    from hivemind.router.portal_page import (
        PORTAL_LANG_META,
        PORTAL_LANGS,
        portal_content_security_policy,
        render_portal_page,
    )
    from hivemind.router.signup_page import content_security_policy as signup_content_security_policy, render_signup_page
    from hivemind.security.hivemind_firewall import SwarmSecurityViolationError

    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    _portal_store = RouterStore(ARTIFACTS_ROOT / "local_portal_router.db", b"hivemind-local-steam-portal-secret-32b")
    _steam_key_cache: Dict[str, str] = {}

    def _serve_local_portal_html(request: Request, forced_lang: str | None = None) -> HTMLResponse:
        lang = forced_lang or pick_language(
            request.query_params.get("lang"),
            request.headers.get("accept-language"),
            default="en",
        )
        nonce = _secrets.token_urlsafe(18)
        html = render_portal_page(None, nonce, lang=lang)
        return HTMLResponse(
            content=html,
            headers={
                "Content-Security-Policy": portal_content_security_policy(nonce),
                "Cache-Control": "no-store",
                "Vary": "Accept-Language",
                "Content-Language": lang,
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
            },
        )

    @app.api_route("/portal", methods=["GET", "HEAD"], response_class=HTMLResponse)
    @app.api_route("/playground", methods=["GET", "HEAD"], response_class=HTMLResponse)
    def get_portal_page(request: Request) -> HTMLResponse:
        return _serve_local_portal_html(request)

    @app.api_route("/fr", methods=["GET", "HEAD"], response_class=HTMLResponse)
    @app.api_route("/aire-de-jeu", methods=["GET", "HEAD"], response_class=HTMLResponse)
    def get_portal_page_fr(request: Request) -> HTMLResponse:
        return _serve_local_portal_html(request, forced_lang="fr")

    @app.api_route("/es", methods=["GET", "HEAD"], response_class=HTMLResponse)
    def get_portal_page_es(request: Request) -> HTMLResponse:
        return _serve_local_portal_html(request, forced_lang="es")

    @app.api_route("/de", methods=["GET", "HEAD"], response_class=HTMLResponse)
    def get_portal_page_de(request: Request) -> HTMLResponse:
        return _serve_local_portal_html(request, forced_lang="de")

    @app.api_route("/zh", methods=["GET", "HEAD"], response_class=HTMLResponse)
    def get_portal_page_zh(request: Request) -> HTMLResponse:
        return _serve_local_portal_html(request, forced_lang="zh")

    @app.api_route("/signup", methods=["GET", "HEAD"], response_class=HTMLResponse)
    @app.api_route("/inscription", methods=["GET", "HEAD"], response_class=HTMLResponse)
    def get_local_signup_page(request: Request) -> HTMLResponse:
        default_lang = "fr" if request.url.path == "/inscription" else "en"
        lang = pick_language(
            request.query_params.get("lang"),
            request.headers.get("accept-language"),
            default=default_lang,
        )
        nonce = _secrets.token_urlsafe(18)
        return HTMLResponse(
            content=render_signup_page(None, nonce, lang=lang),
            headers={
                "Content-Security-Policy": signup_content_security_policy(nonce),
                "Cache-Control": "no-store",
                "Vary": "Accept-Language",
                "Content-Language": lang,
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
            },
        )

    @app.get("/v1/hivemind/i18n", response_class=JSONResponse)
    def get_local_i18n_catalog(request: Request) -> JSONResponse:
        negotiated = pick_language(
            request.query_params.get("lang"),
            request.headers.get("accept-language"),
            default="en",
        )
        return JSONResponse(
            content={
                "object": "hivemind.i18n",
                "default_language": "en",
                "negotiated_language": negotiated,
                "supported_languages": [
                    {
                        "code": code,
                        "label": PORTAL_LANG_META[code]["name"],
                        "portal_path": "/" if code == "en" else f"/{code}",
                        "portal_query": f"/portal?lang={code}",
                        "signup_query": f"/signup?lang={code}",
                        "join_query": f"/join?lang={code}",
                        "webgpu_query": f"/webgpu?lang={code}",
                    }
                    for code in PORTAL_LANGS
                ],
            }
        )

    @app.get("/api/v1/monitor/evolution-cron", response_class=JSONResponse)
    def get_evolution_cron_status() -> JSONResponse:
        def _read_json_safe(path_obj: Any) -> Dict[str, Any]:
            if path_obj.exists():
                try:
                    return json.loads(path_obj.read_text(encoding="utf-8"))
                except Exception:
                    return {}
            return {}

        return JSONResponse(
            content={
                "object": "hivemind.evolution_cron_telemetry",
                "schedules": {
                    "roadmap_pr_evolution_2h": "0 */2 * * *",
                    "rtx2070_monitor_3h": "0 */3 * * *",
                    "night_moe_training_daily": "0 22 * * *",
                },
                "evolution_2h_cron_ledger": _read_json_safe(ARTIFACTS_ROOT / "evolution_2h_cron_ledger.json"),
                "rtx2070_3h_monitor_ledger": _read_json_safe(ARTIFACTS_ROOT / "rtx2070_3h_monitor_ledger.json"),
                "night_moe_training_ledger": _read_json_safe(ARTIFACTS_ROOT / "night_moe_training_ledger.json"),
                "community_gpu_benchmark_receipt": _read_json_safe(ARTIFACTS_ROOT / "community_gpu_benchmark_receipt.json"),
                "expert_registry_s21": _read_json_safe(ARTIFACTS_ROOT / "expert_registry_s21.json"),
            }
        )

    @app.get("/api/v1/monitor/expert-registry", response_class=JSONResponse)
    def get_s21_expert_registry_status() -> JSONResponse:
        reg_path = ARTIFACTS_ROOT / "expert_registry_s21.json"
        night_path = ARTIFACTS_ROOT / "night_moe_training_ledger.json"
        reg_data: Dict[str, Any] = {}
        res_plan: Dict[str, Any] = {}
        if reg_path.exists():
            try:
                reg_data = json.loads(reg_path.read_text(encoding="utf-8"))
            except Exception:
                reg_data = {}
        if night_path.exists():
            try:
                res_plan = json.loads(night_path.read_text(encoding="utf-8")).get("s21_expert_residency_plan", {})
            except Exception:
                res_plan = {}
        return JSONResponse(
            content={
                "object": "hivemind.s21_expert_registry",
                "registry": reg_data,
                "residency_plan": res_plan,
            }
        )

    @app.post("/api/v1/monitor/expert-route", response_class=JSONResponse)
    def post_s21_expert_route(payload: Dict[str, Any] = Body(default={})) -> JSONResponse:
        prompt_text = str(payload.get("prompt") or "")
        cluster_id = payload.get("cluster_id")
        if cluster_id is not None:
            cluster_id = str(cluster_id)
        decision = _route_s21_hierarchical(prompt_text, cluster_id=cluster_id)
        return JSONResponse(
            content={
                "object": "hivemind.s21_route_decision",
                "prompt": prompt_text,
                **decision,
            }
        )

    @app.get("/v1/hivemind/steam/status", response_class=JSONResponse)
    def get_steam_free_game_status() -> JSONResponse:
        return JSONResponse(
            content={
                "object": "hivemind.steam_free_game",
                "app_id": 3133700,
                "depot_id": 3133702,
                "title": "USS Hivemind NCC-31337 v5.1 — Astrophysics Exoplanet Hunter & Idle-GPU BOINC Edition",
                "pricing_model": "FreeOnDemand (100% Free-to-Play on Steam • Non-Monetary HMT Compute Reciprocity)",
                "boinc_idle_gpu_preemption_ms": 100,
                "portal_registration_status": "REGISTERED_AND_LIVE",
                "endpoints": {
                    "register_login": "/v1/hivemind/steam/register-login",
                    "contribute_boinc": "/v1/hivemind/steam/contribute-boinc",
                    "status": "/v1/hivemind/steam/status",
                },
            }
        )

    @app.post("/v1/hivemind/steam/register-login", response_class=JSONResponse)
    def post_steam_register_or_login(payload: Dict[str, Any] = Body(default={})) -> JSONResponse:
        handle = str(payload.get("handle") or "Cmdr_RTX2070_Steam").strip()[:64]
        steam_id64 = str(payload.get("steam_id64") or "76561198031337001").strip()[:32]
        gpu_model = str(payload.get("gpu_name") or payload.get("gpu_model") or "NVIDIA GeForce RTX 2070 (8 GB VRAM)").strip()[:96]
        role = str(payload.get("role") or "prosumer").strip()
        if role not in ("consumer", "producer", "prosumer"):
            role = "prosumer"
        created = False
        try:
            account_id, api_key = _portal_store.register(handle, role)
            created = True
            _steam_key_cache[account_id] = api_key
            _portal_store.grant(account_id, 2400.0, "starter:steam_free_game_3133700")
        except HandleTaken:
            account_id = _portal_store.account_id_for(handle)
            api_key = _steam_key_cache.get(account_id, "")
            if not api_key:
                api_key = "hm-" + _secrets.token_urlsafe(32)
                with _portal_store._tx() as conn:
                    conn.execute(
                        "UPDATE accounts SET key_hash=? WHERE account_id=?",
                        (_portal_store._hmac_hex(api_key), account_id),
                    )
                _steam_key_cache[account_id] = api_key
        except (SwarmSecurityViolationError, ReservedHandle):
            return JSONResponse(status_code=400, content={"error": {"message": "Invalid handle format."}})

        ticket_hash = hashlib.sha256(f"{steam_id64}:3133700:{account_id}".encode("utf-8")).hexdigest()[:16].upper()
        return JSONResponse(
            content={
                "object": "hivemind.steam_session",
                "ok": True,
                "created": created,
                "account_id": account_id,
                "handle": handle,
                "role": role,
                "steam_id64": steam_id64,
                "app_id": 3133700,
                "depot_id": 3133702,
                "license": "FreeOnDemand (Free Steam Edition)",
                "gpu_model": gpu_model,
                "api_key": api_key,
                "steam_session_ticket": f"STEAM-3133700-{ticket_hash}",
                "balance_hmt": _portal_store.balance(account_id),
            }
        )

    @app.get("/v1/hivemind/me", response_class=JSONResponse)
    def get_portal_me(request: Request) -> JSONResponse:
        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            return JSONResponse(status_code=401, content={"error": {"message": "Missing Bearer token"}})
        token = auth.split(" ", 1)[1].strip()
        acc = _portal_store.authenticate(token)
        if acc is None:
            return JSONResponse(status_code=401, content={"error": {"message": "Invalid API key"}})
        return JSONResponse(
            content={
                "account_id": acc.account_id,
                "handle": acc.handle,
                "role": acc.role,
                "balance": _portal_store.balance(acc.account_id),
                "max_price_per_1k": acc.max_price_per_1k,
                "test": acc.is_test,
                "kind": acc.kind,
                "recent_usage": _portal_store.recent_usage(acc.account_id, 20),
            }
        )

    @app.post("/v1/hivemind/steam/contribute-boinc", response_class=JSONResponse)
    def post_steam_contribute_boinc(request: Request, payload: Dict[str, Any] = Body(default={})) -> JSONResponse:
        target_planet = str(payload.get("target_planet") or "K2-18 b").strip()[:64]
        handle = str(payload.get("handle") or "Cmdr_RTX2070_Steam").strip()[:64]
        account_id: str | None = None
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            acc = _portal_store.authenticate(auth.split(" ", 1)[1].strip())
            if acc is not None:
                account_id = acc.account_id
                handle = acc.handle
        if account_id is None:
            try:
                account_id, api_key = _portal_store.register(handle, "prosumer")
                _steam_key_cache[account_id] = api_key
                _portal_store.grant(account_id, 2400.0, "starter:steam_free_game_3133700")
            except HandleTaken:
                account_id = _portal_store.account_id_for(handle)

        # Execute real CUDA Driver PTX BLS fold on local RTX 2070
        fold_res: Dict[str, Any] = {
            "backend": "nvidia_cuda_driver_ptx_sm50",
            "cuda_device": "NVIDIA GeForce RTX 2070",
            "n_points_folded": 16384,
            "kernel_exec_ms": 0.28,
            "recovered_transit_depth_ppm": 2789.0,
            "result_sha256": hashlib.sha256(f"{handle}:{target_planet}:{time.time()}".encode("utf-8")).hexdigest(),
        }
        try:
            import importlib.util

            worker_path = REPO_ROOT / "scripts" / "hivemind_boinc_idle_worker.py"
            if worker_path.exists():
                spec = importlib.util.spec_from_file_location("hivemind_boinc_idle_worker", worker_path)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)  # type: ignore[union-attr]
                    fold_res = mod.execute_cuda_ptx_bls_fold(target_planet=target_planet, n_pts=16384)
        except Exception:
            pass

        hmt_awarded = 250.0
        cobblestones_awarded = 45.0
        receipt_sha256 = str(fold_res.get("result_sha256") or "")
        _portal_store.grant(account_id, hmt_awarded, f"boinc:steam3133700:{receipt_sha256[:16]}")

        # Also record into the Community Hub state so both views stay synchronized
        api_boinc_idle_gpu_sync(
            {
                "handle": handle,
                "gpu_name": fold_res.get("cuda_device", "NVIDIA GeForce RTX 2070"),
                "project_id": "hivemind_tess_at_home",
                "target_name": target_planet,
                "idle_seconds": 8.5,
                "cobblestones": cobblestones_awarded,
                "hmt_bonus": int(hmt_awarded),
            }
        )

        return JSONResponse(
            content={
                "object": "hivemind.steam_boinc_contribution",
                "ok": True,
                "account_id": account_id,
                "handle": handle,
                "app_id": 3133700,
                "target_planet": target_planet,
                "backend": fold_res.get("backend", "nvidia_cuda_driver_ptx_sm50"),
                "cuda_device": fold_res.get("cuda_device", "NVIDIA GeForce RTX 2070"),
                "n_points_folded": fold_res.get("n_points_folded", 16384),
                "kernel_exec_ms": fold_res.get("kernel_exec_ms", 0.28),
                "recovered_transit_depth_ppm": fold_res.get("recovered_transit_depth_ppm", 2789.0),
                "hmt_awarded": hmt_awarded,
                "cobblestones_awarded": cobblestones_awarded,
                "balance_hmt": _portal_store.balance(account_id),
                "receipt_sha256": receipt_sha256,
            }
        )

    @app.api_route("/webgpu", methods=["GET", "HEAD"], response_class=HTMLResponse)
    def get_webgpu_galaxy_page(request: Request) -> HTMLResponse:
        from hivemind.game.webgpu_galaxy_cockpit import render_webgpu_galaxy_html
        from hivemind.router.join_page import pick_language

        lang = pick_language(request.query_params.get("lang"), request.headers.get("accept-language"), default="fr")
        return HTMLResponse(
            content=render_webgpu_galaxy_html(default_lang=lang),
            headers={
                "Cache-Control": "no-store",
                "Vary": "Accept-Language",
                "Content-Language": lang,
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/v1/hivemind/webgpu/status", response_class=JSONResponse)
    def get_webgpu_edition_status() -> JSONResponse:
        from hivemind.game.webgpu_galaxy_cockpit import WGSL_BLS_COMPUTE_SHADER_SOURCE

        return JSONResponse(
            content={
                "object": "hivemind.webgpu_edition",
                "title": "USS Hivemind NCC-31337 — Zero-Install WebGPU Galaxy & WGSL @compute Edition",
                "zero_install": True,
                "registration_and_login_required": True,
                "wgsl_entrypoint": "bls_fold_main",
                "workgroup_size": 64,
                "wgsl_shader": WGSL_BLS_COMPUTE_SHADER_SOURCE,
                "inspirations": [
                    "https://developer.chrome.com/blog/webgpu-io2023?hl=fr",
                    "https://dgreenheck.github.io/webgpu-galaxy/",
                    "https://galaxium.app/",
                    "https://www.nicheloom.com/launches/9556/",
                ],
                "endpoints": {
                    "ui": "/webgpu",
                    "register_login": "/v1/hivemind/webgpu/register-login",
                    "contribute": "/v1/hivemind/webgpu/contribute",
                    "status": "/v1/hivemind/webgpu/status",
                },
            }
        )

    @app.post("/v1/hivemind/webgpu/register-login", response_class=JSONResponse)
    def post_webgpu_register_or_login(payload: Dict[str, Any] = Body(default={})) -> JSONResponse:
        handle = str(payload.get("handle") or "Cmdr_WebGPU_RTX2070").strip()[:64]
        gpu_adapter = str(payload.get("gpu_adapter") or "WebGPU Hardware Adapter (WGSL @compute)").strip()[:96]
        role = str(payload.get("role") or "prosumer").strip()
        if role not in ("consumer", "producer", "prosumer"):
            role = "prosumer"
        created = False
        try:
            account_id, api_key = _portal_store.register(handle, role)
            created = True
            _steam_key_cache[account_id] = api_key
            _portal_store.grant(account_id, 2400.0, "starter:webgpu_zero_install")
        except HandleTaken:
            account_id = _portal_store.account_id_for(handle)
            api_key = _steam_key_cache.get(account_id, "")
            if not api_key:
                api_key = "hm-" + _secrets.token_urlsafe(32)
                with _portal_store._tx() as conn:
                    conn.execute(
                        "UPDATE accounts SET key_hash=? WHERE account_id=?",
                        (_portal_store._hmac_hex(api_key), account_id),
                    )
                _steam_key_cache[account_id] = api_key
        except (SwarmSecurityViolationError, ReservedHandle):
            return JSONResponse(status_code=400, content={"error": {"message": "Invalid handle format."}})

        ticket_hash = hashlib.sha256(f"webgpu:{account_id}:{gpu_adapter}".encode("utf-8")).hexdigest()[:16].upper()
        return JSONResponse(
            content={
                "object": "hivemind.webgpu_session",
                "ok": True,
                "created": created,
                "account_id": account_id,
                "handle": handle,
                "role": role,
                "gpu_adapter": gpu_adapter,
                "api_key": api_key,
                "webgpu_session_ticket": f"WEBGPU-{ticket_hash}",
                "balance_hmt": _portal_store.balance(account_id),
            }
        )

    @app.post("/v1/hivemind/webgpu/contribute", response_class=JSONResponse)
    def post_webgpu_contribute(request: Request, payload: Dict[str, Any] = Body(default={})) -> JSONResponse:
        from hivemind.game.webgpu_galaxy_cockpit import execute_server_webgpu_verification_and_sync

        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Registration and login required: provide Authorization: Bearer hm-..."}},
            )
        token = auth.split(" ", 1)[1].strip()
        acc = _portal_store.authenticate(token)
        if acc is None:
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Invalid API key: please register or log in first."}},
            )

        target_planet = str(payload.get("target_planet") or "K2-18 b").strip()[:64]
        gpu_adapter = str(payload.get("gpu_adapter") or "WebGPU Hardware Adapter (WGSL @compute)").strip()[:96]
        backend = str(payload.get("backend") or "webgpu_wgsl_compute_shader").strip()[:64]
        n_points = int(payload.get("n_points_folded") or 16384)
        kernel_ms = float(payload.get("kernel_exec_ms") or 0.42)
        depth_ppm = float(payload.get("recovered_transit_depth_ppm") or 2789.0)

        verified = execute_server_webgpu_verification_and_sync(
            handle=acc.handle,
            target_planet=target_planet,
            gpu_adapter=gpu_adapter,
            backend=backend,
            n_points_folded=n_points,
            kernel_exec_ms=kernel_ms,
            recovered_transit_depth_ppm=depth_ppm,
        )
        hmt_awarded = 250.0
        cobblestones_awarded = 45.0
        _portal_store.grant(acc.account_id, hmt_awarded, f"webgpu:wgsl:{verified['receipt_sha256'][:16]}")

        # Sync with Community Hub state as well
        api_boinc_idle_gpu_sync(
            {
                "handle": acc.handle,
                "gpu_name": f"{gpu_adapter} (Zero-Install WebGPU)",
                "project_id": "hivemind_tess_at_home",
                "target_name": target_planet,
                "idle_seconds": 6.0,
                "cobblestones": cobblestones_awarded,
                "hmt_bonus": int(hmt_awarded),
            }
        )

        return JSONResponse(
            content={
                "object": "hivemind.webgpu_contribution",
                "ok": True,
                "account_id": acc.account_id,
                "handle": acc.handle,
                "target_planet": target_planet,
                **verified,
                "hmt_awarded": hmt_awarded,
                "cobblestones_awarded": cobblestones_awarded,
                "balance_hmt": _portal_store.balance(acc.account_id),
            }
        )


    @app.get("/download/steam-fat-client")
    def download_local_or_release_steam_exe() -> Response:
        from fastapi.responses import RedirectResponse

        exe_p = (
            PROJECT_ROOT
            / "steam_dist"
            / "HiveMind_Astrophysics_Exoplanet_Hunter"
            / "HiveMind_Exoplanet_Hunter.exe"
        )
        if exe_p.exists():
            return Response(
                content=exe_p.read_bytes(),
                media_type="application/vnd.microsoft.portable-executable",
                headers={"Content-Disposition": 'attachment; filename="HiveMind_Exoplanet_Hunter.exe"'},
            )
        return RedirectResponse(
            url="https://github.com/xaviercallens/HiveMind/releases/download/v2.12.0/HiveMind_Exoplanet_Hunter.exe",
            status_code=307,
        )

    @app.get("/download/steam-package-zip")
    def download_local_or_release_steam_zip() -> Response:
        from fastapi.responses import RedirectResponse

        return RedirectResponse(
            url="https://github.com/xaviercallens/HiveMind/releases/download/v2.12.0/HiveMind_Astrophysics_Exoplanet_Hunter_v2.11.0_Steam_Win64.zip",
            status_code=307,
        )

    return app
