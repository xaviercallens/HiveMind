# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""USS Hivemind NCC-31337 v4.6 Complete Tactical Command Bridge & Multi-Screen Stitch Suite (`stitch_v46_cockpit_template.py`).

Synthesizes all 12 screens and high-resolution assets from Google Stitch MCP project `10018415280079892710`
(`https://stitch.withgoogle.com/projects/10018415280079892710` — *USS Hivemind Interface Engine*):
  - Screen 02 (`02_uss_hivemind_ncc-31337_v4_6_-_complete_tactical.html`): Complete Tactical Command Bridge & Game Suite
  - Screen 11 & 03 (`11_uss_hivemind_-_stage_1_galaxie.html`, `03_image_png.png`): Stage 1 Galaxy Map with all 10 Star Systems + Sub-Spatial Astrometric Telemetry Box + Interactive Ship Flight
  - Screen 00 (`00_uss_hivemind_-_stage_2_orbit_bee.html`): Stage 2 ORBIT BEE // BEE-MOE Lagrange Ring Ingress, 3-Vector Ion Propulsion Telemetry, Orbital Eccentricity Decay (e), Target Dossier, and Fleet Rendezvous Dock
  - Screen 10 & 06 (`10_uss_hivemind_ncc-31337_v4_6_-_stage_3_oolite_pl.html`, `06_uss_hivemind_-_stage_3_oolite_land.html`): Stage 3 Oolite Planetfall Landing Bridge + SAR Wireframe + `[SYS.SPEC.01]` Atmospheric Spectro + `[SYS.LITHO.03]` Seismic & Sub-Basalt + Interactive Retro-Propulsion Output Slider
  - Screen 01 (`01_uss_hivemind_-_stage_4_jwst_lois.html`): Stage 4 JWST / LOIS Spectroscopy Array (Beryllium 6.5M NIRSpec•MIRI), 6 Molecular Absorption Band Cards (H2O, CH4, CO2, CO, NH3, O3 ★), Thermal Gradient Model (T_eq), and Live NASA TAP ADQL Sync
  - Screen 04 (`04_uss_hivemind_-_fat_client_unity_unreal_vs_web.html`): Stage 5 Dual-Engine Architecture Plan (Native Fat Client UE5/Unity/Win32 + Steam AppID 3133700 + RTX 2070 BOINC CUDA PTX vs. Zero-Install WebGPU `/webgpu`)
  - Local photorealistic Stitch v2 assets (`05`, `07`, `08`, `09`) served via `/stitch-assets/stitch_v2_*.png`
"""

from __future__ import annotations

from pathlib import Path

try:
    from hivemind.config import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

AIDA_GALAXY_URL = (
    "https://lh3.googleusercontent.com/aida/AEtjO1U4mX6MHmzCWb6Ji929sOUG-USs47juoBasP5sritS01ottiMtE3aqddCMG8RT497EthQpVXNST1pKWmnOI3mfAKXYBaJcfNpIlUFVEcD02gSGzrBpNjVdHOdw3rJK8pxodUYJFfTn1cHrFKvybypcl-EMgp0qSg3v-WBpgROH-yZs4Xy4lbh4OESAaw9NA20ZR1cNei2CYckpJYtbjmE2htnUink50wSQ3KmoVbjSdFQ"
)
AIDA_K218B_URL = (
    "https://lh3.googleusercontent.com/aida/AEtjO1XmPdII46aWb143uksF1mDs20yLB9Y5RfzTTbfsG993PvkzsVqvu00dZsE43JMFjksD1U6ngie02guqVR-xdUE0f6ZxlMSj9ruEUJXm_VKE57_9rt1m63RUYvMjP_F95N3oVvgG3QaZLwzVzwmPHFmfjcBQBJMHp9ccJBcn13zP0wWgXuBDSpITIphZNwWCfhCogc93eZ9yeMH3YZYwJrLwkqJ1Io7iQEE3Diy5xg8Z"
)
AIDA_LHS1140B_URL = (
    "https://lh3.googleusercontent.com/aida/AEtjO1V13Wv6d-01hDJo83nHWi7FklguAoqj8VJgG65hB2uevqP5pDXjJToMsy7-CIoGzk195mIJXfCVOUqD4LpYEabBZC-kZ9gW4tLSEKlC7dNZILyHtreu4Kdgfgwtjfu3XYAPj791aqkdYjMKlQLbdH26tJNXiNpoHzh6g5mElOA8Jv3Q4toZ38hCjK5t-tYLHddaUfxW0wUj7mgHSWCd8Pj14cgqKen9Vn-k5FxlT7Wd"
)
AIDA_OOLITE_COCKPIT_URL = (
    "https://lh3.googleusercontent.com/aida-public/AB6AXuChalitz6rWZe3ltUcPx8rSkF3Iman86Rh2HAdVj9NyZVYEuJeNp5xsR6DmX-gY0x2loGFP8bGoSUBXNRKI3qvENsdY-QJSjMVdHrfehDK8AcmM9y4MTimbVPV0IcZdp2LksNaZXLkaikjhOa9M34Fr7AQJaZNOR-bRZHhfjIQ2Jk9-cVhhtpAdV7Jehr0kA7B79X5W0jfqOi-F5yI3VrNN2u2E9llsMOU4xAFqfXEm"
)

LOCAL_GALAXY_ASSET = "/stitch-assets/stitch_v2_07_photorealistic_milky_way_barred_spiral_galaxy_vi.png"
LOCAL_K218B_ASSET = "/stitch-assets/stitch_v2_09_photorealistic_exoplanet_k2-18_b_hycean_world_vi.png"
LOCAL_LHS1140B_ASSET = "/stitch-assets/stitch_v2_05_photorealistic_exoplanet_lhs_1140_b_super-earth.png"
LOCAL_OOLITE_COCKPIT_ASSET = "/stitch-assets/stitch_v2_08_photorealistic_descent_cockpit_view_of_spaceship.png"


# Stage 1 Additions from Stitch Screen 11 (`11_uss_hivemind_-_stage_1_galaxie.html`) & Screen 03 (`03_image_png.png`)
STAGE_1_SUBSPATIAL_HUD_AND_EXTRA_NODES_HTML = """<!-- SYSTEM 9: KOI-Cand 14 (From Stitch Screen 03 / 11) -->
<div class="planet-node absolute bottom-[58px] left-[305px] flex flex-col items-center cursor-pointer transition hover:scale-110" data-planet="koi14">
  <div class="relative w-14 h-14 flex items-center justify-center">
    <div class="absolute w-14 h-14 rounded-full border border-dashed border-cyan-400/40"></div>
    <div class="w-7 h-7 rounded-full planet-koi14 shadow-lg"></div>
    <div class="absolute top-0 right-0 bg-cyan-950 border border-cyan-400 rounded-full px-1 text-[8px] text-cyan-300 font-mono">KOI</div>
  </div>
  <span class="node-label text-[11px] text-cyan-200 bg-black/85 px-1.5 py-0.5 border border-cyan-900/60 mt-0.5 font-mono rounded">KOI-Cand 14</span>
</div>
<!-- SYSTEM 10: KOI-Bin #02 (From Stitch Screen 03 / 11) -->
<div class="planet-node absolute bottom-[58px] right-[225px] flex flex-col items-center cursor-pointer transition hover:scale-110" data-planet="koibin">
  <div class="relative w-14 h-14 flex items-center justify-center">
    <div class="absolute w-14 h-14 rounded-full border border-dashed border-rose-400/50"></div>
    <div class="w-7 h-7 rounded-full planet-koibin shadow-lg"></div>
    <div class="absolute top-0 right-0 bg-rose-950 border border-rose-400 rounded-full px-1 text-[8px] text-rose-300 font-mono">BIN</div>
  </div>
  <span class="node-label text-[11px] text-cyan-200 bg-black/85 px-1.5 py-0.5 border border-cyan-900/60 mt-0.5 font-mono rounded">KOI-Bin #02</span>
</div>
<!-- SYSTEM 11: UNSEEN REAL NASA MYSTERY EXOPLANET (NEW EACH GAME / [M]) -->
<div class="planet-node absolute top-[185px] left-[255px] flex flex-col items-center cursor-pointer transition hover:scale-110 z-30" data-planet="mystery_nasa" id="mystery-planet-node">
  <div class="relative w-16 h-16 flex items-center justify-center">
    <div class="absolute inset-0 rounded-full border-2 border-dashed border-purple-400 animate-spin" style="animation-duration: 10s;"></div>
    <div class="w-8 h-8 rounded-full bg-gradient-to-br from-purple-300 via-fuchsia-500 to-indigo-900 shadow-[0_0_16px_rgba(192,132,252,0.9)] border border-white/70"></div>
    <div class="absolute -top-1 -right-1 bg-purple-950 border border-purple-300 rounded px-1 text-[8px] text-purple-200 font-bold font-mono">★ NASA</div>
  </div>
  <span class="node-label text-[11px] text-purple-200 bg-purple-950/90 px-1.5 py-0.5 border border-purple-400 mt-0.5 font-mono font-bold rounded" id="mystery-node-label">★ Proxima Cen b</span>
</div>
<!-- ANIMATED BEE-MOE PROBE SWARM LAYER (STAGE 1 LAUNCH CINEMATIC) -->
<div class="absolute inset-0 pointer-events-none z-40" id="probe-launch-layer"></div>
<!-- STAGE 1 BOTTOM-CENTER ACTION CONSOLE: LAUNCH SURVEILLANCE ROBOTS -->
<div class="absolute bottom-12 left-1/2 -translate-x-1/2 z-30 bg-[#030d1e]/95 border-2 border-tacticalGreen/90 rounded-md px-3 py-1.5 shadow-2xl backdrop-blur-md flex flex-wrap items-center gap-2 font-mono" id="s1-launch-hud">
  <div class="text-[10px] leading-tight">
    <div class="text-tacticalCyan font-bold">ÉTAPE 1/4 : EXPLOREZ LA GALAXIE &amp; CHOISISSEZ UNE PLANÈTE</div>
    <div class="text-slate-300 text-[9px]">Cible verrouillée : <strong class="text-tacticalAmber" id="s1-hud-target-name">LHS 1140 b</strong> • <span class="text-emerald-300" id="s1-hud-physics-tag">g = 1.53 g • T_eq = 235 K</span></div>
  </div>
  <button class="bg-emerald-500 hover:bg-emerald-400 text-black font-extrabold px-3 py-1 rounded text-[10.5px] shadow-greenGlow cursor-pointer transition active:scale-95 animate-pulse" id="btn-s1-launch-robots" onclick="launchBeeProbesCinematic()">
    🚀 ENVOYER ROBOTS DE SURVEILLANCE [ESPACE / BOUTON A]
  </button>
  <button class="bg-purple-900/90 hover:bg-purple-700 border border-purple-300 text-purple-100 font-bold px-2 py-1 rounded text-[9.5px] cursor-pointer transition active:scale-95" id="btn-s1-mystery" onclick="selectRandomNasaMysteryPlanet(true)">
    🎲 PLANÈTE MYSTÈRE NASA [M]
  </button>
</div>
<!-- TOP-LEFT SUB-SPATIAL ASTROMETRIC SCAN HUD (From Stitch Screen 11) -->
<div class="absolute bottom-12 left-2.5 z-30 bg-[#050d1a]/90 border border-cyan-500/60 rounded p-1.5 w-60 shadow-lg backdrop-blur-md pointer-events-none hidden xl:block font-mono">
  <div class="flex items-center justify-between border-b border-cyan-800/80 pb-0.5 mb-1 text-[9px]">
    <span class="text-tacticalCyan font-bold">[STAGE 1: GALAXIE ENGAGÉ]</span>
    <span class="text-tacticalGreen font-bold">SYS_LOCK: 99.4%</span>
  </div>
  <div class="grid grid-cols-3 gap-1 text-[8.5px] text-slate-300">
    <div class="bg-black/60 p-1 rounded border border-cyan-900/60">
      <div class="text-[7px] text-slate-400">DIST</div>
      <div class="text-tacticalCyan font-bold truncate" id="s1-sub-dist">48.8 AL</div>
    </div>
    <div class="bg-black/60 p-1 rounded border border-cyan-900/60">
      <div class="text-[7px] text-slate-400">V_RAD (K)</div>
      <div class="text-tacticalGreen font-bold truncate" id="s1-sub-vrad">4.9 m/s</div>
    </div>
    <div class="bg-black/60 p-1 rounded border border-cyan-900/60">
      <div class="text-[7px] text-slate-400">WARP</div>
      <div class="text-tacticalAmber font-bold" id="s1-sub-warp">0.12 C</div>
    </div>
  </div>
  <div class="flex justify-between items-center text-[8px] text-cyan-300/80 pt-0.5 mt-0.5 border-t border-cyan-900/50">
    <span id="s1-sub-az">AZIMUTH // 55.10°</span>
    <span id="s1-sub-dec">DECL // +100.0</span>
  </div>
</div>"""


# Stage 2 Full 3-Column Lagrange Ring & Ion Propulsion Suite from Stitch Screen 00 (`00_uss_hivemind_-_stage_2_orbit_bee.html`)
STAGE_2_FULL_ORBIT_BEE_HTML = """<!-- STAGE 2 VIEW: ORBIT BEE // BEE-MOE LAGRANGE RING INGRESS (FROM STITCH SCREEN 00) -->
<div class="absolute inset-0 bg-[#040c1b] hidden transition-opacity duration-300 px-3 pt-2 pb-12 overflow-hidden" id="viewport-stage-2">
  <div class="w-full h-full relative grid grid-cols-12 gap-2.5 items-stretch font-mono select-none">
    <!-- LEFT COLUMN (3 cols): ION PROPULSION TELEMETRY & ORBITAL ECCENTRICITY DECAY (e) -->
    <div class="col-span-12 lg:col-span-3 flex flex-col gap-1.5">
      <!-- Panel 1: Ion Propulsion Telemetry -->
      <div class="bg-[#071122]/90 border border-cyan-500/40 rounded p-2 flex flex-col justify-between shadow-lg">
        <div class="flex items-center justify-between border-b border-cyan-800/80 pb-1 mb-1">
          <span class="text-[10px] font-bold text-tacticalCyan font-display tracking-wider">ION PROPULSION TELEMETRY</span>
          <span class="text-[9px] bg-cyan-950 border border-tacticalCyan/60 text-tacticalCyan px-1.5 py-0.2 rounded font-bold" id="ion-thrust">THRUST 94.2% (1,840 mN)</span>
        </div>
        <div class="space-y-1 text-[9px]">
          <div>
            <div class="flex justify-between text-slate-300 mb-0.5">
              <span>VECTOR ALPHA • FORWARD AXIAL</span>
              <span class="text-tacticalCyan font-bold" id="vec-alpha-val">94.2% // 1,840 mN</span>
            </div>
            <div class="grid grid-cols-10 gap-0.5 h-1.5">
              <div class="bg-tacticalCyan rounded-xs"></div><div class="bg-tacticalCyan rounded-xs"></div><div class="bg-tacticalCyan rounded-xs"></div><div class="bg-tacticalCyan rounded-xs"></div><div class="bg-tacticalCyan rounded-xs"></div><div class="bg-tacticalCyan rounded-xs"></div><div class="bg-tacticalCyan rounded-xs"></div><div class="bg-tacticalCyan rounded-xs"></div><div class="bg-tacticalCyan rounded-xs"></div><div class="bg-cyan-950 rounded-xs"></div>
            </div>
          </div>
          <div>
            <div class="flex justify-between text-slate-300 mb-0.5">
              <span>VECTOR BETA • INCLINATION BIAS</span>
              <span class="text-tacticalGreen font-bold" id="vec-beta-val">91.8% // 1,790 mN</span>
            </div>
            <div class="grid grid-cols-10 gap-0.5 h-1.5">
              <div class="bg-tacticalGreen rounded-xs"></div><div class="bg-tacticalGreen rounded-xs"></div><div class="bg-tacticalGreen rounded-xs"></div><div class="bg-tacticalGreen rounded-xs"></div><div class="bg-tacticalGreen rounded-xs"></div><div class="bg-tacticalGreen rounded-xs"></div><div class="bg-tacticalGreen rounded-xs"></div><div class="bg-tacticalGreen rounded-xs"></div><div class="bg-tacticalGreen rounded-xs"></div><div class="bg-emerald-950 rounded-xs"></div>
            </div>
          </div>
          <div>
            <div class="flex justify-between text-slate-300 mb-0.5">
              <span>VECTOR GAMMA • SYNODIC SYNC</span>
              <span class="text-tacticalAmber font-bold" id="vec-gamma-val">99.9% // LOCKED</span>
            </div>
            <div class="grid grid-cols-10 gap-0.5 h-1.5">
              <div class="bg-tacticalAmber rounded-xs"></div><div class="bg-tacticalAmber rounded-xs"></div><div class="bg-tacticalAmber rounded-xs"></div><div class="bg-tacticalAmber rounded-xs"></div><div class="bg-tacticalAmber rounded-xs"></div><div class="bg-tacticalAmber rounded-xs"></div><div class="bg-tacticalAmber rounded-xs"></div><div class="bg-tacticalAmber rounded-xs"></div><div class="bg-tacticalAmber rounded-xs"></div><div class="bg-tacticalAmber rounded-xs"></div>
            </div>
          </div>
        </div>
        <div class="mt-1.5 pt-1 border-t border-cyan-900/60 flex justify-between text-[8.5px] text-slate-400">
          <span>SPECIFIC IMPULSE (Isp)</span>
          <span class="text-white font-bold">4,180 s [XENON-HALL]</span>
        </div>
      </div>

      <!-- Panel 2: Orbital Eccentricity Decay (e) & Real Astrophysics Asteroid Alert -->
      <div class="bg-[#071122]/90 border border-cyan-500/40 rounded p-2 flex flex-col justify-between flex-1 shadow-lg">
        <div class="flex items-center justify-between border-b border-cyan-800/80 pb-1">
          <span class="text-[10px] font-bold text-tacticalCyan font-display">ORBITAL ECCENTRICITY DECAY (e)</span>
          <span class="text-[8.5px] text-tacticalGreen font-bold">● KEPLER v(r)∝1/√r</span>
        </div>
        <div class="bg-amber-950/50 border border-amber-500/60 rounded p-1.5 my-1 text-[8.5px] text-amber-200 leading-snug" id="orbit-astro-explain">
          ☄️ <strong class="text-tacticalAmber">PHYSIQUE ASTÉROÏDES :</strong> Pilotez vos robots (ZQSD / Flèches / Souris / Manette) vers l'<strong>Anneau Vert Lagrange</strong> autour de la planète ! Évitez les astéroïdes képlériens.
        </div>
        <div class="grid grid-cols-3 gap-1 text-[8.5px] text-center">
          <div class="bg-black/50 p-1 rounded border border-slate-800">
            <div class="text-slate-400">CEINTURES</div>
            <div class="text-amber-300 font-bold" id="s2-belt-count">2 CEINTURES</div>
          </div>
          <div class="bg-black/50 p-1 rounded border border-slate-800">
            <div class="text-slate-400">Δ-V KEPLER</div>
            <div class="text-tacticalCyan font-bold" id="s2-kepler-vel">28.4 km/s</div>
          </div>
          <div class="bg-black/50 p-1 rounded border border-emerald-900/60">
            <div class="text-slate-400">EXCENTRICITÉ</div>
            <div class="text-tacticalGreen font-bold" id="s2-ecc-final">e = 0.04</div>
          </div>
        </div>
      </div>
    </div>

    <!-- CENTER COLUMN (5 cols): VIEWPORT // ASTROMETRIC-02 INTERACTIVE 60 FPS ORBITAL ROBOT GAME -->
    <div class="col-span-12 lg:col-span-5 bg-[#030916]/95 border border-cyan-500/50 rounded p-2 flex flex-col justify-between relative overflow-hidden shadow-2xl">
      <!-- Top Telemetry Overlay -->
      <div class="relative z-10 flex items-center justify-between text-[9.5px] border-b border-cyan-900/70 pb-1">
        <div class="flex items-center gap-1.5">
          <span class="w-2 h-2 rounded-full bg-tacticalCyan animate-ping"></span>
          <span class="font-display font-bold text-white">VIEWPORT // ASTROMETRIC-02 (PILOTAGE ROBOTS)</span>
        </div>
        <div class="flex items-center gap-2">
          <span class="text-slate-300">SAT-NODES: <strong class="text-tacticalGreen" id="sat-count">128 / 128</strong></span>
          <span class="bg-emerald-950 border border-emerald-500/60 text-tacticalGreen px-1.5 py-0.2 rounded font-bold text-[8.5px]" id="s2-difficulty-tag">NIV. 1</span>
        </div>
      </div>

      <!-- Interactive 60 FPS Orbital Canvas + Planet Avatar Badge -->
      <div class="relative flex-1 flex items-center justify-center my-1 bg-[#020611] border border-cyan-900/70 rounded overflow-hidden">
        <canvas class="w-full h-full cursor-crosshair" height="220" id="orbit-bee-canvas" width="460"></canvas>
        <!-- Required Planet Avatar & Label Overlay (Top-Right of Canvas) -->
        <div class="absolute top-1.5 right-1.5 flex items-center gap-1.5 bg-black/85 border border-cyan-500/50 rounded px-1.5 py-0.5 pointer-events-none">
          <img alt="Orbiting Exoplanet" class="w-6 h-6 rounded-full object-cover border border-tacticalCyan" id="stage2-planet-img" src="/stitch-assets/stitch_v2_05_photorealistic_exoplanet_lhs_1140_b_super-earth.png"/>
          <span class="text-[8.5px] text-tacticalAmber font-bold" id="current-stage2-planet">LHS 1140 b (Super-Terre)</span>
        </div>
        <div class="absolute bottom-1 left-2 text-[8px] text-cyan-200 bg-black/80 px-1.5 py-0.5 rounded border border-cyan-800/70 pointer-events-none">
          🎮 <strong>OBJECTIF :</strong> Dirigez <strong>🛸 VOS ROBOTS</strong> dans l'<strong>ANNEAU VERT</strong> (ZQSD / Flèches / Souris / Stick) !
        </div>
      </div>

      <!-- Orbital Synchronization Progress Bar & Transition Button to Stage 3 -->
      <div class="relative z-10 bg-black/90 border border-emerald-500/60 p-1.5 rounded flex flex-col gap-1">
        <div class="flex items-center justify-between text-[9px]">
          <span class="text-emerald-300 font-bold">SYNCHRONISATION DES ROBOTS EN ORBITE : <strong class="text-white" id="orbit-sync-pct">0%</strong></span>
          <span class="text-amber-300 text-[8px]" id="orbit-hazard-status">● BOUCLIER ROBOTS : 100%</span>
        </div>
        <div class="w-full bg-slate-900 h-2.5 rounded overflow-hidden border border-emerald-800">
          <div class="bg-gradient-to-r from-cyan-400 via-emerald-400 to-amber-300 h-full transition-all duration-150" id="orbit-sync-bar" style="width: 15%"></div>
        </div>
        <div class="flex items-center justify-between gap-2 pt-0.5">
          <span class="text-[8px] text-slate-300">À 100%, descente automatique vers l'atmosphère !</span>
          <button class="bg-emerald-500 hover:bg-emerald-400 text-black font-extrabold px-2.5 py-0.5 rounded text-[9.5px] shadow-greenGlow cursor-pointer transition active:scale-95" id="btn-go-stage3" onclick="completeStage2AndStartLanding()">
            🛬 DESCENDRE ATTERRIR MAINTENANT [ENTRÉE / A]
          </button>
        </div>
      </div>
    </div>

    <!-- RIGHT COLUMN (4 cols): TARGET DOSSIER, NIRISS+NIRSpec BIO-BARS & ORBITAL DOCK FLEET RENDEZVOUS -->
    <div class="col-span-12 lg:col-span-4 flex flex-col justify-between gap-1.5">
      <!-- Target Dossier & Transmission Spectroscopy -->
      <div class="bg-[#071122]/90 border border-cyan-500/40 rounded p-2 flex flex-col gap-1.5 shadow-lg">
        <div class="flex items-center justify-between border-b border-cyan-800/80 pb-1">
          <span class="text-[10px] font-bold text-tacticalAmber font-display" id="s2-dossier-title">TARGET DOSSIER // LHS 1140 b</span>
          <span class="text-[8px] bg-emerald-950 text-tacticalGreen border border-emerald-500/50 px-1.5 py-0.2 rounded font-bold">ORBITAL INSERTION OK</span>
        </div>
        <div class="grid grid-cols-3 gap-1 text-[8.5px] bg-black/50 p-1 rounded border border-slate-800">
          <div><span class="text-slate-400 block">RAYON</span><strong class="text-tacticalCyan" id="s2-dossier-rad">1.43 R⊕</strong></div>
          <div><span class="text-slate-400 block">DENSITÉ</span><strong class="text-tacticalGreen" id="s2-dossier-dens">5.6 g/cm³</strong></div>
          <div><span class="text-slate-400 block">EQUIL. TEMP</span><strong class="text-amber-300" id="s2-dossier-temp">~235 K (-38°C)</strong></div>
        </div>
        <div class="space-y-1 text-[8.5px]">
          <div class="text-cyan-300 font-bold text-[8px] uppercase">TRANSMISSION SPECTROSCOPY // NIRISS+NIRSpec</div>
          <div>
            <div class="flex justify-between"><span>WATER VAPOR (H₂O — 1.4 µm)</span><span class="text-tacticalGreen font-bold">CONFIRMED (98.4%)</span></div>
            <div class="w-full bg-slate-900 h-1.5 rounded overflow-hidden"><div class="bg-tacticalGreen h-full" style="width: 98.4%"></div></div>
          </div>
          <div>
            <div class="flex justify-between"><span>DIMETHYL SULFIDE (DMS — 3.3 µm)</span><span class="text-tacticalCyan font-bold">PROBABILITY: 72%</span></div>
            <div class="w-full bg-slate-900 h-1.5 rounded overflow-hidden"><div class="bg-tacticalCyan h-full" style="width: 72%"></div></div>
          </div>
          <div>
            <div class="flex justify-between"><span>CARBON DIOXIDE &amp; METHANE (CO₂/CH₄)</span><span class="text-tacticalAmber font-bold">HIGH RATIO (89%)</span></div>
            <div class="w-full bg-slate-900 h-1.5 rounded overflow-hidden"><div class="bg-tacticalAmber h-full" style="width: 89%"></div></div>
          </div>
        </div>
      </div>

      <!-- Orbital Dock // Fleet Rendezvous & Interactive Controls -->
      <div class="bg-[#071122]/90 border border-cyan-500/40 rounded p-2 flex flex-col justify-between flex-1 shadow-lg">
        <div class="flex items-center justify-between border-b border-cyan-800/80 pb-0.5 text-[9.5px]">
          <span class="font-display font-bold text-tacticalCyan">ORBITAL DOCK // FLEET RENDEZVOUS</span>
          <span class="text-[8px] text-slate-400">LCARS v4.6</span>
        </div>
        <div class="space-y-0.5 text-[8.5px] my-1">
          <div class="flex justify-between items-center bg-black/50 px-1.5 py-0.5 rounded border border-slate-800">
            <span class="text-slate-200">01. BEE-POD CARRIER 01</span><span class="text-tacticalGreen font-bold">[DOCKED]</span>
          </div>
          <div class="flex justify-between items-center bg-black/50 px-1.5 py-0.5 rounded border border-slate-800">
            <span class="text-slate-200">02. PROBE LOIS-EXO-4</span><span class="text-tacticalCyan font-bold">[APPROACHING]</span>
          </div>
          <div class="flex justify-between items-center bg-black/50 px-1.5 py-0.5 rounded border border-slate-800">
            <span class="text-slate-200">03. JWST UPLINK BEACON</span><span class="text-tacticalAmber font-bold">[ONLINE L2]</span>
          </div>
        </div>
        <div class="grid grid-cols-3 gap-1 pt-1">
          <button class="bg-emerald-950 hover:bg-emerald-800 border border-emerald-400 text-tacticalGreen font-bold px-2 py-1 text-[9px] rounded shadow-greenGlow transition cursor-pointer active:scale-95" id="btn-deploy-probe">
            + SONDE MOE (+25%)
          </button>
          <button class="bg-cyan-950 hover:bg-cyan-800 border border-cyan-400 text-cyan-200 font-bold px-2 py-1 text-[9px] rounded transition cursor-pointer active:scale-95" id="btn-sync-phase">
            SYNC L1-L2 (+35%)
          </button>
          <button class="bg-amber-500 hover:bg-amber-400 text-black font-extrabold px-2 py-1 text-[9px] rounded shadow-amberGlow transition cursor-pointer active:scale-95" id="btn-sample-atmo" onclick="sampleAtmosphereStage2()">
            🔬 SAMPLE ATMO
          </button>
        </div>
      </div>
    </div>
  </div>
</div>"""


# Stage 3 Full Oolite Planetfall Landing Bridge combining Screen 10 (`10_uss_hivemind_ncc-31337_v4_6_-_stage_3_oolite_pl.html`) & Screen 06 (`06_uss_hivemind_-_stage_3_oolite_land.html`)
STAGE_3_FULL_OOLITE_BRIDGE_HTML = """<!-- STAGE 3 VIEW: FULL OOLITE PLANETFALL LANDING BRIDGE (FROM STITCH SCREENS 10 & 06) -->
<div class="absolute inset-0 bg-[#020813] hidden transition-opacity duration-300 overflow-hidden" id="viewport-stage-3">
  <div class="relative w-full h-full flex flex-col justify-between px-2.5 pt-1.5 pb-12 select-none">
    <!-- Photorealistic Descent Cockpit Canopy Backdrop (Stitch Asset 08) -->
    <img alt="Descent cockpit view of spaceship landing on exoplanet K2-18 b" class="absolute inset-0 w-full h-full object-cover object-center opacity-80 pointer-events-none filter brightness-95 contrast-110" id="oolite-cockpit-canopy-img" src="/stitch-assets/stitch_v2_08_photorealistic_descent_cockpit_view_of_spaceship.png"/>
    <img alt="Planet Horizon Thumbnail" class="hidden" id="oolite-bg-img" src="/stitch-assets/stitch_v2_08_photorealistic_descent_cockpit_view_of_spaceship.png"/>
    <div class="absolute inset-0 bg-gradient-to-t from-[#020712]/90 via-transparent to-[#020712]/70 pointer-events-none"></div>
    <!-- Synthetic Aperture Radar (SAR) Wireframe Grid -->
    <div class="absolute inset-x-8 bottom-10 h-28 border-t border-cyan-500/30 opacity-35 pointer-events-none" style="background-image: linear-gradient(0deg, rgba(0,240,255,0.2) 1px, transparent 1px), linear-gradient(90deg, rgba(0,240,255,0.2) 1px, transparent 1px); background-size: 32px 18px; transform: perspective(300px) rotateX(45deg);"></div>

    <!-- 3-Column Oolite Planetfall Tactical Bridge HUD Grid -->
    <div class="relative z-10 w-full grid grid-cols-12 gap-2 items-stretch">
      <!-- LEFT COLUMN (MFD 1): Cobra MK-III Telemetry, Primeable Equipments (4/18), Planetary Cargo Manifest (33t/33t) -->
      <div class="col-span-12 lg:col-span-3 flex flex-col gap-1">
        <div class="bg-[#070e1a]/90 border border-cyan-500/40 rounded p-1.5 text-[9.5px] shadow-lg backdrop-blur-md">
          <div class="flex items-center justify-between border-b border-cyan-500/30 pb-0.5 mb-0.5">
            <span class="text-tacticalCyan font-bold tracking-wider font-display">COBRA MK-III // HONOUR OF HIVEMIND</span>
            <span class="text-tacticalGreen font-bold bg-emerald-950/80 px-1 rounded text-[8px] border border-emerald-400/40">OPÉRATIONNEL</span>
          </div>
          <div class="grid grid-cols-2 gap-x-2 gap-y-0.5 text-slate-300">
            <div>État Service: <span class="text-emerald-300 font-bold">98%</span></div>
            <div>Masse: <span class="text-cyan-200">214.7 t</span></div>
            <div>Valeur Revente: <span class="text-amber-300">176,229 Cr</span></div>
            <div>Solde: <span class="text-tacticalAmber font-bold">14,399.5 Cr</span></div>
            <div>Statut Légal: <span class="text-tacticalGreen font-bold">Clean [0]</span></div>
            <div>Rang: <span class="text-cyan-300">Astro-Hunter (19)</span></div>
            <div>Bouclier Therm.: <span class="text-cyan-300 font-bold" id="oolite-shield-temp">820°C [NOMINAL]</span></div>
            <div>Gravité: <span class="text-amber-300" id="oolite-gravity-val">1.27 g (12.5 m/s²)</span></div>
          </div>
        </div>

        <div class="bg-[#070e1a]/90 border border-cyan-500/40 rounded p-1.5 flex flex-col gap-0.5 flex-1 text-[9.5px] shadow-lg backdrop-blur-md">
          <div class="flex items-center justify-between border-b border-cyan-500/30 pb-0.5">
            <span class="text-tacticalCyan font-bold font-display">ÉQUIPEMENTS PRIMEABLES</span>
            <span class="text-amber-300 font-bold">4 / 18</span>
          </div>
          <ul class="space-y-0.5 text-[9px] leading-tight">
            <li class="text-slate-400">• LMSS Activator (Surface Scan)</li>
            <li class="text-tacticalCyan font-bold bg-cyan-950/60 px-1 py-0.5 rounded border border-cyan-500/40">▶▶ Target Banking / Atmo Stabiliser</li>
            <li class="text-slate-300">• Market Inquirer MFD (K2-18 b / LHS 1140 b)</li>
            <li class="text-slate-300">• Ship Identifier &amp; IFF Transponder</li>
            <li class="text-slate-300">• CommsLog MFD (Canal Planétaire Local)</li>
          </ul>
          <div class="mt-auto pt-1 border-t border-cyan-500/20">
            <div class="text-[8.5px] text-tacticalAmber font-bold flex justify-between mb-0.5">
              <span>MANIFESTE CARGAISON PLANÉTAIRE</span>
              <span>33t / 33t [PLEIN]</span>
            </div>
            <div class="grid grid-cols-3 gap-0.5 text-[8.5px] text-slate-300 bg-black/50 p-1 rounded border border-slate-800">
              <div>Quirium: <span class="text-emerald-300">42.6Cr</span></div>
              <div>Gem-Stones: <span class="text-emerald-300">21.9Cr</span></div>
              <div>Platinum: <span class="text-emerald-300">71.4Cr</span></div>
              <div>Gold: <span class="text-emerald-300">40.4Cr</span></div>
              <div>Computers: <span class="text-emerald-300">73.5Cr</span></div>
              <div>Rations: <span class="text-emerald-300">14.1Cr</span></div>
            </div>
          </div>
        </div>
      </div>

      <!-- CENTER COLUMN: 60 FPS 3D Glide-Slope Landing Tunnel, Altitude/Speed Tapes, Retro-Propulsion Slider (Screen 06) & Action Toggles -->
      <div class="col-span-12 lg:col-span-6 flex flex-col justify-between items-center relative">
        <!-- Danger / Altitude Critical Warning Banner -->
        <div class="w-full flex items-center justify-between gap-1.5">
          <span class="bg-cyan-950/90 border border-tacticalCyan/60 text-tacticalCyan px-2 py-0.5 rounded text-[8.5px] font-mono font-bold" id="oolite-atmo-layer-tag">[COUCHE 4 : THERMOSPHÈRE]</span>
          <div class="border-2 border-red-500/80 bg-red-950/85 px-2.5 py-0.5 rounded shadow-redGlow flex items-center gap-1.5" id="oolite-danger-banner">
            <span class="text-red-400 font-bold text-xs animate-pulse">⚠️</span>
            <div class="text-center leading-tight">
              <div class="font-display font-black tracking-widest text-red-400 text-[9.5px]" id="terrain-alert">DESCENTE ATMOSPHÉRIQUE EN COURS</div>
              <div class="text-[8px] text-red-200">CIBLE: <span class="text-white font-bold" id="landing-zone-name">CHASMA AUSTRALIS</span></div>
            </div>
          </div>
          <span class="bg-emerald-950/90 border border-emerald-400/60 text-tacticalGreen px-2 py-0.5 rounded text-[8.5px] font-mono font-bold" id="oolite-wind-tag">💨 VENT: 0 m/s</span>
        </div>

        <!-- Interactive 60 FPS 3D Glide-Slope Landing Canvas + Left/Right Altitude & Speed Tapes -->
        <div class="relative w-full max-w-[440px] h-36 flex items-center justify-center my-0.5 bg-[#020814]/85 border border-cyan-500/60 rounded overflow-hidden shadow-lg">
          <canvas class="w-full h-full cursor-crosshair" height="144" id="oolite-glide-canvas" width="440"></canvas>
          <!-- Radar Alt Tape (Left) -->
          <div class="absolute left-1.5 top-1.5 flex flex-col gap-0.5 text-[8.5px] text-tacticalCyan pointer-events-none">
            <span class="bg-tacticalCyan text-black px-1.5 py-0.5 font-extrabold rounded-sm shadow" id="altitude-tape">3,420 m</span>
            <span class="text-[7.5px] text-cyan-200 bg-black/70 px-1 rounded">ALTITUDE AGL</span>
          </div>
          <!-- Vertical Descent Rate Tape (Right) -->
          <div class="absolute right-1.5 top-1.5 flex flex-col gap-0.5 items-end text-[8.5px] text-amber-300 pointer-events-none">
            <span class="bg-tacticalAmber text-black px-1.5 py-0.5 font-extrabold rounded-sm shadow" id="speed-tape">84.2 m/s</span>
            <span class="text-[7.5px] text-emerald-300 bg-black/70 px-1 rounded" id="vs-tape">MAX SAFE &lt; 95 m/s</span>
          </div>
          <div class="absolute bottom-1 inset-x-2 flex justify-between items-center text-[7.5px] text-cyan-200 bg-black/80 px-2 py-0.5 rounded border border-cyan-900/70 pointer-events-none">
            <span>🎮 <strong>GAUCHE/DROITE</strong> (Q/D, ←/→, Souris, Stick) : Aligner sur la Piste Verte</span>
            <span class="text-tacticalAmber font-bold">🔥 <strong>[ESPACE / Z / A]</strong> : Freiner</span>
          </div>
        </div>

        <!-- Interactive Retro-Propulsion Slider (From Screen 06) + Landing Action Buttons ([C] / [G] / [Z/UP] / [AUTO]) + Real Astrophysics Atmo Bar -->
        <div class="w-full max-w-lg flex flex-col gap-1">
          <div class="bg-[#070e1a]/90 border border-cyan-500/40 rounded px-2 py-0.5 flex items-center justify-between gap-2 text-[9px] font-mono">
            <span class="text-slate-300 font-bold">RETRO-PROPULSION OUTPUT: <strong class="text-tacticalCyan" id="thrustReadout">75%</strong></span>
            <input class="w-32 accent-cyan-400 cursor-pointer h-1.5" id="thrustSlider" max="100" min="20" oninput="updateRetroSlider(this.value)" type="range" value="75"/>
            <span class="text-[8px] text-tacticalGreen" id="oolite-atmo-scale-tag">H = 8.5 km • P = 2.4 bar</span>
          </div>
          <div class="grid grid-cols-4 gap-1">
            <button class="px-1.5 py-0.5 rounded bg-emerald-950/90 border border-emerald-400/80 text-emerald-300 flex items-center justify-center gap-1 hover:bg-emerald-900 text-[9px] font-bold cursor-pointer" id="btn-clearance-toggle">
              <span class="text-tacticalGreen">✓</span> <span>[C] CLEARANCE</span>
            </button>
            <button class="px-1.5 py-0.5 rounded bg-cyan-950/90 border border-tacticalCyan/80 text-cyan-200 flex items-center justify-center gap-1 hover:bg-cyan-900 text-[9px] font-bold cursor-pointer" id="btn-gear-toggle">
              <span class="text-tacticalCyan">✓</span> <span>[G] TRAIN SORTI</span>
            </button>
            <button class="px-1.5 py-0.5 rounded bg-amber-500 hover:bg-amber-400 border border-amber-200 text-black font-extrabold flex items-center justify-center gap-1 shadow-amberGlow text-[9px] cursor-pointer active:scale-95" id="btn-retro-burn">
              <span>🔥 [ESPACE/Z] FREINER</span>
            </button>
            <button class="px-1.5 py-0.5 rounded bg-emerald-500 hover:bg-emerald-400 border border-emerald-200 text-black font-extrabold flex items-center justify-center gap-1 shadow-greenGlow text-[9px] cursor-pointer active:scale-95" id="btn-auto-land" onclick="triggerAssistedTouchdown()">
              <span>🛬 [ENTRÉE] POSER</span>
            </button>
          </div>
          <div class="bg-[#070e1a]/90 border border-cyan-500/40 rounded p-1 text-[8px] text-slate-200 leading-tight" id="oolite-atmo-physics-banner">
            <div class="flex items-center justify-between text-amber-300 font-bold">
              <span>PHYSIQUE ATMOSPHÉRIQUE [<span id="oolite-comms-planet">LHS 1140 b</span>]</span>
              <span class="text-emerald-300">[Tour Chasma] : Maintenir V &lt; 95 m/s jusqu'à 0 m</span>
            </div>
            <div class="text-cyan-200 truncate" id="oolite-atmo-detail-text">Gravité g = 1.53 g (15.0 m/s²) • Atmosphère N₂/CO₂ dense • Cisaillement de vent modéré.</div>
          </div>
        </div>
      </div>

      <!-- RIGHT COLUMN (MFD 2): Radar Contacts (100 km), [SYS.SPEC.01] Atmo Spectro & [SYS.LITHO.03] Seismic (From Screen 06 & 10) -->
      <div class="col-span-12 lg:col-span-3 flex flex-col gap-1">
        <div class="bg-[#070e1a]/90 border border-cyan-500/40 rounded p-1.5 flex flex-col gap-0.5 text-[9.5px] shadow-lg backdrop-blur-md">
          <div class="flex items-center justify-between border-b border-cyan-500/30 pb-0.5">
            <span class="text-tacticalCyan font-bold font-display">RADAR CIBLES &amp; CONTACTS</span>
            <span class="text-[8.5px] text-slate-400">PORTÉE: 100 km</span>
          </div>
          <div class="space-y-0.5 text-[8.5px] font-mono leading-tight">
            <div class="flex justify-between items-center text-red-400">
              <span>3.096 km</span><span>Python: Spark of Decency</span><span class="px-1 bg-red-950 border border-red-500 text-[7px]">[HOSTILE]</span>
            </div>
            <div class="flex justify-between items-center text-cyan-200">
              <span>41.94 km</span><span>Gecko: Orbital Buckler</span><span class="text-tacticalGreen">[NEUTRE]</span>
            </div>
            <div class="flex justify-between items-center text-slate-300">
              <span>53.36 km</span><span>Nav Beacon 123°</span><span class="text-tacticalAmber">[BALISE]</span>
            </div>
            <div class="flex justify-between items-center text-emerald-300">
              <span>61.65 km</span><span>TransHab Station K2</span><span class="text-tacticalGreen">[STATION]</span>
            </div>
          </div>
        </div>

        <div class="bg-[#070e1a]/90 border border-cyan-500/40 rounded p-1.5 flex flex-col justify-between flex-1 text-[9px] shadow-lg backdrop-blur-md">
          <div>
            <div class="flex items-center justify-between border-b border-cyan-500/30 pb-0.5 mb-0.5">
              <span class="text-tacticalCyan font-bold font-display">[SYS.SPEC.01] ATMO &amp; LITHO</span>
              <span class="text-amber-300 text-[8px]">9.4 MOHS</span>
            </div>
            <div class="p-1 rounded bg-cyan-950/60 border border-tacticalCyan flex justify-between items-center mb-1">
              <div>
                <div class="font-bold text-tacticalCyan text-[8.5px]" id="oolite-sys-main-body">1.00 UA • LHS 1140 b [MAIN]</div>
                <div class="text-[7.5px] text-slate-300">N₂ 74.2% • O₂ 18.1% • CO₂ 2.1% • H₂O</div>
              </div>
              <span class="bg-tacticalCyan text-black px-1 rounded font-bold text-[7.5px]">LOCK</span>
            </div>
            <div class="grid grid-cols-2 gap-1 text-[8px] font-mono bg-black/50 p-1 rounded border border-slate-800">
              <div>CRUST: <span class="text-tacticalCyan font-bold">SUB-BASALT</span></div>
              <div>GEOTHERM: <span class="text-tacticalAmber font-bold">+42.8°C</span></div>
            </div>
          </div>
          <div class="mt-auto p-1 rounded bg-amber-950/40 border border-amber-500/40 text-[8px] text-amber-200 leading-tight">
            <span class="font-bold text-tacticalAmber">ASTUCE VOL :</span> Pressez <kbd class="bg-black border border-amber-400 px-1 rounded text-tacticalAmber font-bold">[Z]</kbd> ou cliquez sur <strong>RÉTROS ACTIFS</strong> pour freiner sous 95 m/s.
          </div>
        </div>
      </div>
    </div>

    <!-- BOTTOM ROW: Classic Oolite/Elite Dials & 3D Wireframes (8 Circular Digital Gauges + Compass Globe + Cobra MK-III Wireframe) -->
    <div class="relative z-10 grid grid-cols-12 gap-2 items-center mt-0.5">
      <div class="col-span-12 md:col-span-3 bg-[#070e1a]/95 border border-cyan-500/50 rounded p-1 px-2 flex items-center justify-between">
        <div>
          <div class="text-[8.5px] text-slate-400 uppercase tracking-widest">.: INFOS VOYAGE :.</div>
          <div class="text-[10.5px] text-tacticalCyan font-bold" id="oolite-voyage-target">Cible: LHS 1140 b</div>
          <div class="text-[8.5px] text-slate-400">Dist: <span class="text-tacticalGreen font-bold" id="oolite-voyage-dist">1,840 m</span></div>
        </div>
        <div class="w-10 h-10 relative flex items-center justify-center">
          <svg class="w-full h-full text-tacticalCyan" viewBox="0 0 100 100">
            <circle cx="50" cy="50" fill="none" r="45" stroke="currentColor" stroke-dasharray="2 2" stroke-width="1.5"></circle>
            <ellipse cx="50" cy="50" fill="none" rx="45" ry="18" stroke="currentColor" stroke-width="1.2"></ellipse>
            <ellipse cx="50" cy="50" fill="none" rx="18" ry="45" stroke="currentColor" stroke-width="1.2"></ellipse>
            <circle class="animate-ping" cx="62" cy="38" fill="#ffb800" r="4"></circle>
            <circle cx="62" cy="38" fill="#ffb800" r="3"></circle>
          </svg>
        </div>
      </div>

      <!-- 8 Digital Circular Gauges -->
      <div class="col-span-12 md:col-span-6 bg-[#070e1a]/95 border border-cyan-500/50 rounded p-1 flex flex-wrap items-center justify-around gap-1">
        <div class="flex flex-col items-center">
          <div class="w-8 h-8 rounded-full border-2 border-emerald-400 flex items-center justify-center bg-black/80 shadow-cyanGlow">
            <span class="text-[9.5px] font-bold text-emerald-300" id="gauge-speed">84</span>
          </div>
          <span class="text-[7.5px] text-slate-400">VITESSE</span>
        </div>
        <div class="flex flex-col items-center">
          <div class="w-8 h-8 rounded-full border-2 border-amber-400 flex items-center justify-center bg-black/80 shadow-amberGlow">
            <span class="text-[8.5px] font-bold text-amber-300">6.9+3</span>
          </div>
          <span class="text-[7.5px] text-slate-400">FUEL</span>
        </div>
        <div class="flex flex-col items-center">
          <div class="w-8 h-8 rounded-full border-2 border-tacticalCyan flex items-center justify-center bg-black/80">
            <span class="text-[9.5px] font-bold text-tacticalCyan" id="gauge-alt">1.8k</span>
          </div>
          <span class="text-[7.5px] text-slate-400">ALT (m)</span>
        </div>
        <div class="flex flex-col items-center">
          <div class="w-8 h-8 rounded-full border-2 border-slate-400 flex items-center justify-center bg-black/80">
            <span class="text-[8.5px] font-bold text-slate-200">33/33</span>
          </div>
          <span class="text-[7.5px] text-slate-400">CARGO</span>
        </div>
        <div class="flex flex-col items-center">
          <div class="w-8 h-8 rounded-full border-2 border-cyan-400 flex items-center justify-center bg-black/80">
            <span class="text-[9.5px] font-bold text-cyan-200" id="gauge-temp">23°C</span>
          </div>
          <span class="text-[7.5px] text-slate-400">T.CABINE</span>
        </div>
        <div class="flex flex-col items-center">
          <div class="w-8 h-8 rounded-full border-2 border-emerald-400 flex items-center justify-center bg-black/80">
            <span class="text-[9.5px] font-bold text-emerald-300">256</span>
          </div>
          <span class="text-[7.5px] text-slate-400">ÉNERGIE</span>
        </div>
        <div class="flex flex-col items-center">
          <div class="w-8 h-8 rounded-full border-2 border-amber-500 flex items-center justify-center bg-black/80">
            <span class="text-[9.5px] font-bold text-amber-400">4%</span>
          </div>
          <span class="text-[7.5px] text-slate-400">BOUCL.AV</span>
        </div>
        <div class="flex flex-col items-center">
          <div class="w-8 h-8 rounded-full border-2 border-emerald-500 flex items-center justify-center bg-black/80">
            <span class="text-[9.5px] font-bold text-emerald-400">100%</span>
          </div>
          <span class="text-[7.5px] text-slate-400">BLINDAGE</span>
        </div>
      </div>

      <div class="col-span-12 md:col-span-3 bg-[#070e1a]/95 border border-cyan-500/50 rounded p-1 px-2 flex items-center justify-between">
        <div class="w-10 h-10 relative flex items-center justify-center">
          <svg class="w-full h-full text-tacticalGreen" viewBox="0 0 100 100">
            <polygon fill="none" points="50,15 90,80 70,85 50,75 30,85 10,80" stroke="currentColor" stroke-width="1.8"></polygon>
            <line stroke="currentColor" stroke-width="1.2" x1="50" x2="50" y1="15" y2="75"></line>
            <line stroke="currentColor" stroke-width="1.2" x1="50" x2="90" y1="40" y2="80"></line>
            <line stroke="currentColor" stroke-width="1.2" x1="50" x2="10" y1="40" y2="80"></line>
          </svg>
        </div>
        <div class="text-right">
          <div class="text-[8.5px] text-slate-400 uppercase tracking-widest">.: VAISSEAU :.</div>
          <div class="text-[10.5px] text-emerald-300 font-bold">Cobra MK-III</div>
          <div class="text-[8.5px] text-tacticalGreen">Bouclier atmo actif</div>
        </div>
      </div>
    </div>
  </div>
</div>"""


# Stage 4 Full Beryllium 6.5M NIRSpec•MIRI Suite from Stitch Screen 01 (`01_uss_hivemind_-_stage_4_jwst_lois.html`)
STAGE_4_FULL_JWST_BERYLLIUM_HTML = """<!-- STAGE 4 VIEW: JWST / LOIS SPECTROSCOPY ARRAY BERYLLIUM 6.5M NIRSpec•MIRI (FROM STITCH SCREEN 01) -->
<div class="absolute inset-0 bg-[#030914] hidden transition-opacity duration-300 px-3 pt-2 pb-12 overflow-hidden" id="viewport-stage-4">
  <div class="w-full h-full flex flex-col justify-between font-mono select-none gap-1.5">
    <!-- Top Beryllium 6.5M Header Strip -->
    <div class="bg-[#061224]/90 border border-cyan-500/50 rounded px-2.5 py-1 flex flex-wrap items-center justify-between gap-2">
      <div class="flex items-center gap-2">
        <span class="w-2.5 h-2.5 bg-tacticalAmber rounded-full animate-ping"></span>
        <span class="font-display text-[11px] font-bold text-white tracking-wider">[STAGE.04] L2_LAGRANGE_LOCK — JWST / LOIS SPECTROSCOPY ARRAY (BERYLLIUM 6.5M NIRSpec•MIRI)</span>
      </div>
      <div class="flex items-center gap-2 text-[9px]">
        <span class="bg-black/60 border border-cyan-800 px-1.5 py-0.5 rounded text-cyan-300">CRYOMR_TEMP: <strong class="text-tacticalCyan">6.42 K</strong></span>
        <span class="bg-black/60 border border-cyan-800 px-1.5 py-0.5 rounded text-cyan-300">RESOLVING R: <strong class="text-tacticalAmber">2,700</strong></span>
        <span class="bg-black/60 border border-cyan-800 px-1.5 py-0.5 rounded text-cyan-300">FILTER: <strong class="text-white">F290LP/G395H</strong></span>
        <span class="text-[9.5px] text-tacticalGreen border border-emerald-500 px-2 py-0.5 rounded bg-emerald-950/70 font-bold" id="bio-confidence-badge">BIOSIGNATURE INDICE: 94.2%</span>
      </div>
    </div>

    <!-- Main 12-Col Grid: Left/Center NIRSpec Waterfall + 6 Molecular Cards (7 cols) | Right NASA TAP ADQL + Thermal Gradient (5 cols) -->
    <div class="grid grid-cols-12 gap-2.5 flex-1 items-stretch">
      <!-- LEFT / CENTER (7 cols): [NIRSPEC_SPECTRUM_WATERFALL] & 6 Molecular Absorption Band Cards -->
      <div class="col-span-12 lg:col-span-7 bg-[#050e1d]/90 border border-cyan-500/40 rounded p-2 flex flex-col justify-between">
        <div class="flex items-center justify-between text-[9.5px] border-b border-cyan-900/70 pb-1">
          <span class="text-tacticalCyan font-bold font-display">[NIRSPEC_SPECTRUM_WATERFALL] • 0.6µm — 28.0µm [MIRI]</span>
          <span class="text-emerald-400 bg-black/80 px-2 py-0.2 border border-emerald-800 rounded font-bold text-[8.5px]" id="dms-detection-tag">● PEAK DETECT: OZONE (O₃) @ 9.6µm &amp; DMS [4.8σ]</span>
        </div>

        <!-- SVG Spectral Curve -->
        <div class="relative h-36 w-full bg-[#020712] border border-cyan-950 rounded my-1 flex items-center justify-center overflow-hidden">
          <svg class="w-full h-full" preserveAspectRatio="none" viewBox="0 0 700 145">
            <line stroke="rgba(0,240,255,0.14)" stroke-width="1" x1="0" x2="700" y1="35" y2="35"></line>
            <line stroke="rgba(0,240,255,0.14)" stroke-width="1" x1="0" x2="700" y1="72" y2="72"></line>
            <line stroke="rgba(0,240,255,0.14)" stroke-width="1" x1="0" x2="700" y1="110" y2="110"></line>
            <path d="M 0,88 Q 50,58 100,96 T 175,40 T 255,108 T 335,30 T 415,115 T 495,48 T 585,36 T 700,72" fill="none" filter="drop-shadow(0px 0px 4px #00f0ff)" id="svg-spec-curve" stroke="#00f0ff" stroke-width="2.5"></path>
            <circle cx="175" cy="40" fill="#00ff88" r="4" stroke="#fff" stroke-width="1.2"></circle>
            <text fill="#00ff88" font-size="9.5" x="148" y="27">H₂O (1.4µm)</text>
            <circle cx="335" cy="30" fill="#ffb800" r="4" stroke="#fff" stroke-width="1.2"></circle>
            <text fill="#ffb800" font-size="9.5" x="310" y="18">CH₄ (2.3µm)</text>
            <circle cx="495" cy="48" fill="#00f0ff" r="4" stroke="#fff" stroke-width="1.2"></circle>
            <text fill="#00f0ff" font-size="9.5" x="470" y="35">CO₂ (4.3µm)</text>
            <circle cx="585" cy="36" fill="#00ff88" r="4.5" stroke="#fff" stroke-width="1.2"></circle>
            <text fill="#00ff88" font-size="9.5" font-weight="bold" x="558" y="23">O₃ ★ (9.6µm)</text>
          </svg>
          <div class="absolute bottom-1 left-2 right-2 flex justify-between text-[8px] text-slate-400">
            <span>1.0µm [NIRSpec]</span><span>4.5µm</span><span>10.0µm [MIRI]</span><span>18.5µm</span><span>28.0µm</span>
          </div>
        </div>

        <!-- 6 Molecular Absorption Band Cards (From Screen 01) -->
        <div class="grid grid-cols-6 gap-1.5 text-[8.5px]">
          <div class="bg-black/70 border border-emerald-500/60 rounded p-1 text-center cursor-pointer hover:border-emerald-300" onclick="highlightMolecule('H2O', '1.4 µm', '88%')">
            <div class="text-tacticalGreen font-bold text-[10px]">H₂O</div>
            <div class="text-slate-400 text-[7.5px]">1.4 µm • 88%</div>
            <div class="w-full bg-slate-800 h-1 rounded mt-0.5"><div class="bg-tacticalGreen h-full" style="width:88%"></div></div>
          </div>
          <div class="bg-black/70 border border-amber-500/60 rounded p-1 text-center cursor-pointer hover:border-amber-300" onclick="highlightMolecule('CH4', '2.3 µm', '62%')">
            <div class="text-tacticalAmber font-bold text-[10px]">CH₄</div>
            <div class="text-slate-400 text-[7.5px]">2.3 µm • 62%</div>
            <div class="w-full bg-slate-800 h-1 rounded mt-0.5"><div class="bg-tacticalAmber h-full" style="width:62%"></div></div>
          </div>
          <div class="bg-black/70 border border-cyan-500/60 rounded p-1 text-center cursor-pointer hover:border-cyan-300" onclick="highlightMolecule('CO2', '4.3 µm', '99%')">
            <div class="text-tacticalCyan font-bold text-[10px]">CO₂</div>
            <div class="text-slate-400 text-[7.5px]">4.3 µm • 99%</div>
            <div class="w-full bg-slate-800 h-1 rounded mt-0.5"><div class="bg-tacticalCyan h-full" style="width:99%"></div></div>
          </div>
          <div class="bg-black/70 border border-slate-600 rounded p-1 text-center cursor-pointer hover:border-slate-400" onclick="highlightMolecule('CO', '4.7 µm', '14%')">
            <div class="text-slate-200 font-bold text-[10px]">CO</div>
            <div class="text-slate-400 text-[7.5px]">4.7 µm • 14%</div>
            <div class="w-full bg-slate-800 h-1 rounded mt-0.5"><div class="bg-slate-400 h-full" style="width:14%"></div></div>
          </div>
          <div class="bg-black/70 border border-cyan-700 rounded p-1 text-center cursor-pointer hover:border-cyan-400" onclick="highlightMolecule('NH3', '10.5 µm', '41%')">
            <div class="text-cyan-300 font-bold text-[10px]">NH₃</div>
            <div class="text-slate-400 text-[7.5px]">10.5 µm • 41%</div>
            <div class="w-full bg-slate-800 h-1 rounded mt-0.5"><div class="bg-cyan-400 h-full" style="width:41%"></div></div>
          </div>
          <div class="bg-emerald-950/60 border border-tacticalGreen rounded p-1 text-center cursor-pointer shadow-greenGlow" onclick="highlightMolecule('O3 ★', '9.6 µm', '78%')">
            <div class="text-tacticalGreen font-extrabold text-[10px]">O₃ ★</div>
            <div class="text-emerald-200 text-[7.5px]">9.6 µm • 78%</div>
            <div class="w-full bg-slate-800 h-1 rounded mt-0.5"><div class="bg-tacticalGreen h-full" style="width:78%"></div></div>
          </div>
        </div>
      </div>

      <!-- RIGHT (5 cols): [NASA_TAP_ADQL_SYNC], 3 KPI Cards, THERMAL GRADIENT MODEL (T_eq) & Decision Controls -->
      <div class="col-span-12 lg:col-span-5 bg-[#050e1d]/90 border border-cyan-500/40 rounded p-2 flex flex-col justify-between">
        <div class="flex items-center justify-between border-b border-cyan-900/70 pb-1 text-[9.5px]">
          <span class="text-tacticalCyan font-bold font-display">[NASA_TAP_ADQL_SYNC]</span>
          <span class="text-tacticalGreen font-bold text-[8.5px]">LATENCY: 42ms // LIVE TAP OK</span>
        </div>

        <!-- ADQL Query Box -->
        <div class="bg-black/90 border border-cyan-800/80 p-1.5 rounded text-[9.5px] my-1">
          <div class="text-tacticalAmber font-bold text-[8px] mb-0.5">ADQL QUERY // EXOPLANET ARCHIVE TAP</div>
          <div class="text-cyan-200 truncate" id="query-preview">SELECT pl_name, pl_rade, pl_masse, st_teff FROM ps WHERE pl_name = 'LHS 1140 b';</div>
        </div>

        <!-- 3 KPI Cards from Screen 01 -->
        <div class="grid grid-cols-3 gap-1.5 text-[8.5px] my-0.5">
          <div class="bg-black/60 border border-cyan-900/80 p-1 rounded">
            <div class="text-slate-400 text-[7.5px]">DOPPLER VEL (K)</div>
            <div class="text-tacticalCyan font-bold text-[10px]" id="s4-kpi-k">4.9 m/s</div>
            <div class="text-slate-500 text-[7px]">±0.3 m/s HARPS</div>
          </div>
          <div class="bg-black/60 border border-cyan-900/80 p-1 rounded">
            <div class="text-slate-400 text-[7.5px]">ECCENTRICITY (e)</div>
            <div class="text-tacticalGreen font-bold text-[10px]" id="s4-kpi-e">0.04</div>
            <div class="text-emerald-400 text-[7px]">NEAR-CIRCULAR</div>
          </div>
          <div class="bg-black/60 border border-cyan-900/80 p-1 rounded">
            <div class="text-slate-400 text-[7.5px]">TRANSIT DEPTH</div>
            <div class="text-tacticalAmber font-bold text-[10px]" id="s4-kpi-depth">5340 ppm</div>
            <div class="text-amber-300 text-[7px]">ΔF/F KEPLER/TESS</div>
          </div>
        </div>

        <!-- Thermal Gradient Model (T_eq) from Screen 01 -->
        <div class="bg-black/60 border border-cyan-900/80 p-1 rounded text-[8.5px] my-0.5">
          <div class="flex justify-between items-center mb-0.5">
            <span class="text-slate-300 font-bold">THERMAL GRADIENT MODEL (T_eq)</span>
            <span class="text-tacticalGreen font-bold" id="s4-teq-label">▲ 235 K (-38°C) HAB_ZONE_LOCK</span>
          </div>
          <div class="w-full h-1.5 rounded bg-gradient-to-r from-cyan-500 via-emerald-400 to-red-500 relative">
            <div class="w-2.5 h-3 bg-white border border-black rounded-xs absolute -top-0.5 left-[44%]" id="s4-teq-marker"></div>
          </div>
          <div class="flex justify-between text-[7px] text-slate-400 mt-0.5">
            <span>100 K [CRYOGENIC]</span>
            <span class="text-emerald-300 font-bold">HAB_ZONE (180–310 K)</span>
            <span>400 K+ [SUPERHEATED]</span>
          </div>
        </div>

        <!-- Planetary Astrobiology & Habitability Summary Box -->
        <div class="bg-emerald-950/40 border border-emerald-500/60 p-1.5 rounded text-[8.5px] text-emerald-100 leading-snug my-0.5" id="s4-astrobio-summary">
          🧬 <strong class="text-tacticalGreen">ANALYSE PLANÉTAIRE &amp; VIE POSSIBLE :</strong> <span id="s4-life-text">LHS 1140 b : Super-Terre rocheuse dense (1.43 R⊕, 5.6 g/cm³, 235 K). Un océan liquide côté jour et des sources hydrothermales peuvent abriter la vie marine.</span>
        </div>

        <!-- Action Buttons -->
        <div class="flex flex-wrap gap-1 pt-0.5">
          <button class="bg-cyan-950 hover:bg-cyan-800 border border-cyan-500 text-tacticalCyan px-2 py-1 rounded text-[9px] font-bold transition cursor-pointer" id="btn-run-adql">
            🌐 NASA TAP ADQL
          </button>
          <button class="bg-emerald-950 hover:bg-emerald-800 border border-emerald-500 text-tacticalGreen px-2 py-1 rounded text-[9px] font-bold transition cursor-pointer" id="btn-verify-kepler">
            📐 KEPLER T²/a³
          </button>
          <button class="bg-emerald-500 hover:bg-emerald-400 text-black px-2.5 py-1 rounded text-[9.5px] font-extrabold shadow-greenGlow transition cursor-pointer active:scale-95" id="btn-approve" onclick="submitStitchDiscovery(true)">
            ✅ APPROUVER PLANÈTE (+800 HMT) [ESPACE/A]
          </button>
          <button class="bg-red-600 hover:bg-red-500 text-white px-2.5 py-1 rounded text-[9.5px] font-extrabold transition cursor-pointer active:scale-95" id="btn-reject" onclick="submitStitchDiscovery(false)">
            🚨 REJETER FAUX POSITIF [R/X]
          </button>
          <button class="bg-amber-500 hover:bg-amber-400 text-black px-2 py-1 rounded text-[9px] font-extrabold transition cursor-pointer active:scale-95" id="btn-next-planet-s1" onclick="advanceToNextPlanetFromStage4()">
            🌌 PROCHAINE PLANÈTE (NIV. +1) [1]
          </button>
        </div>
      </div>
    </div>
  </div>
</div>"""


# Stage 5 Dual-Engine Architecture Plan (Fat Client UE5/Unity/Win32 + RTX 2070 BOINC vs. WebGPU `/webgpu`) from Stitch Screen 04 (`04_uss_hivemind_-_fat_client_unity_unreal_vs_web.html`)
STAGE_5_DUAL_ENGINE_ARCHI_HTML = """<!-- STAGE 5 VIEW: DUAL-ENGINE ARCHITECTURE PLAN — FAT CLIENT UE5/UNITY/STEAM vs. WEBGPU (FROM STITCH SCREEN 04) -->
<div class="absolute inset-0 bg-[#030914] hidden transition-opacity duration-300 px-3 pt-2 pb-12 overflow-hidden" id="viewport-stage-5">
  <div class="w-full h-full flex flex-col justify-between font-mono select-none gap-2">
    <!-- Top Header Banner -->
    <div class="bg-[#061224]/95 border border-tacticalAmber/60 rounded px-3 py-1 flex flex-wrap items-center justify-between gap-2">
      <div class="flex items-center gap-2">
        <span class="w-2.5 h-2.5 bg-tacticalCyan rounded-full animate-ping"></span>
        <span class="font-display text-[11px] font-bold text-white tracking-wider">[STAGE.05] USS HIVEMIND DUAL-ENGINE ARCHITECTURE // STEAM FAT CLIENT (RTX 2070 BOINC) vs. ZERO-INSTALL WEBGPU</span>
      </div>
      <span class="text-[9px] bg-amber-950 border border-tacticalAmber text-tacticalAmber px-2 py-0.5 rounded font-bold">STITCH SCREEN 04 // DEPLOYMENT MATRIX v4.6</span>
    </div>

    <!-- 2-Column Comparison Matrix from Stitch Screen 04 -->
    <div class="grid grid-cols-12 gap-3 flex-1 items-stretch">
      <!-- LEFT COLUMN (6 cols): OPTION A — NATIVE FAT CLIENT (UE5 / UNITY / WIN32 / STEAM AppID 3133700) -->
      <div class="col-span-12 lg:col-span-6 bg-[#061020]/95 border border-cyan-500/50 rounded p-2.5 flex flex-col justify-between shadow-lg">
        <div>
          <div class="flex items-center justify-between border-b border-cyan-800 pb-1 mb-1.5">
            <div>
              <div class="text-[11px] font-bold text-tacticalCyan font-display">OPTION A: NATIVE FAT CLIENT (STEAM AppID 3133700)</div>
              <div class="text-[8.5px] text-slate-400">UNREAL ENGINE 5.4 / UNITY 6 HDRP • WIN32 / DIRECTX 12 / VULKAN</div>
            </div>
            <span class="bg-cyan-950 border border-tacticalCyan text-tacticalCyan px-1.5 py-0.5 rounded text-[8.5px] font-bold">[TIER-01: HEAVY METAL]</span>
          </div>
          <div class="grid grid-cols-2 gap-1.5 text-[9px] mb-2">
            <div class="bg-black/60 border border-cyan-900/70 p-1.5 rounded">
              <div class="text-slate-400 text-[7.5px]">LOCAL GPU DETECTED</div>
              <div class="text-tacticalGreen font-bold">NVIDIA GeForce RTX 2070</div>
              <div class="text-slate-400 text-[7.5px]">8 GB GDDR6 • 2,304 CUDA • 7.46 TFLOPS</div>
            </div>
            <div class="bg-black/60 border border-cyan-900/70 p-1.5 rounded">
              <div class="text-slate-400 text-[7.5px]">COMPUTE PIPELINE</div>
              <div class="text-tacticalCyan font-bold">BOINC CUDA PTX + TensorRT</div>
              <div class="text-slate-400 text-[7.5px]">FP16 Tensor Cores • Zero-Throttle</div>
            </div>
          </div>
          <ul class="space-y-1 text-[8.5px] text-slate-300">
            <li>• <strong class="text-tacticalCyan">Volumetric Planetary Rendering:</strong> Ray-marched Hycean atmospheres (K2-18 b / LHS 1140 b) + Oolite 3D Ship Mesh at 120 FPS.</li>
            <li>• <strong class="text-tacticalGreen">Dedicated BOINC Folding Daemon:</strong> Background CUDA PTX BLS Transit Periodogram &amp; Bee-MoE Swarm inference (+250 HMT/WU).</li>
            <li>• <strong class="text-tacticalAmber">Steamworks SDK Integration:</strong> Single-click Free Steam Game install (`steam://install/3133700`), achievements &amp; cloud saves.</li>
          </ul>
        </div>
        <div class="pt-1.5 border-t border-cyan-900/70">
          <div class="flex justify-between text-[8px] text-slate-400 mb-0.5">
            <span>VRAM &amp; SYSTEM MEMORY BUDGET</span>
            <span class="text-tacticalCyan font-bold">6.4 GB / 8.0 GB GDDR6 (NOMINAL)</span>
          </div>
          <div class="w-full bg-slate-900 h-1.5 rounded overflow-hidden mb-2"><div class="bg-tacticalCyan h-full" style="width: 78%"></div></div>
          <div class="flex flex-wrap gap-1.5">
            <button class="flex-1 bg-emerald-500 hover:bg-emerald-400 text-black font-extrabold py-1 px-2 rounded text-[9.5px] cursor-pointer transition" onclick="simulateSteamRegisterLoginAndContribute()">
              ⚡ EXÉCUTER WORKUNIT BOINC RTX 2070 (+250 HMT)
            </button>
            <a id="btn-s5-download-exe" class="bg-emerald-700 hover:bg-emerald-600 border border-emerald-300 text-white font-extrabold py-1 px-2 rounded text-[9.5px]" href="https://github.com/xaviercallens/HiveMind/releases/download/v2.12.0/HiveMind_Exoplanet_Hunter.exe" target="_blank" rel="noopener">
              ⬇ WINDOWS .EXE (v2.12.0)
            </a>
            <a id="btn-s5-download-zip" class="bg-cyan-950 hover:bg-cyan-800 border border-tacticalCyan text-tacticalCyan font-bold py-1 px-2 rounded text-[9.5px]" href="https://github.com/xaviercallens/HiveMind/releases/download/v2.12.0/HiveMind_Astrophysics_Exoplanet_Hunter_v2.11.0_Steam_Win64.zip" target="_blank" rel="noopener">
              📦 STEAM .ZIP
            </a>
            <a id="btn-s5-launch-steam" class="bg-amber-500 hover:bg-amber-400 text-black font-extrabold py-1 px-2 rounded text-[9.5px]" href="steam://rungameid/3133702">
              🕹️ STEAM 3133702
            </a>
          </div>
        </div>
      </div>

      <!-- RIGHT COLUMN (6 cols): OPTION B — ZERO-INSTALL WEBGPU CLIENT (`/webgpu`) -->
      <div class="col-span-12 lg:col-span-6 bg-[#061020]/95 border border-purple-500/50 rounded p-2.5 flex flex-col justify-between shadow-lg">
        <div>
          <div class="flex items-center justify-between border-b border-purple-800 pb-1 mb-1.5">
            <div>
              <div class="text-[11px] font-bold text-purple-300 font-display">OPTION B: ZERO-INSTALL WEBGPU GALAXY (`/webgpu`)</div>
              <div class="text-[8.5px] text-slate-400">CHROMIUM WEBGPU API • WGSL @COMPUTE SHADERS • INSTANT BROWSER ACCESS</div>
            </div>
            <span class="bg-purple-950 border border-purple-400 text-purple-200 px-1.5 py-0.5 rounded text-[8.5px] font-bold">[TIER-02: ZERO-INSTALL]</span>
          </div>
          <div class="grid grid-cols-2 gap-1.5 text-[9px] mb-2">
            <div class="bg-black/60 border border-purple-900/70 p-1.5 rounded">
              <div class="text-slate-400 text-[7.5px]">GALAXY PARTICLE ENGINE</div>
              <div class="text-purple-300 font-bold">65,536 Spirals (WGSL)</div>
              <div class="text-slate-400 text-[7.5px]">Inspired by Galaxium &amp; WebGPU-Galaxy</div>
            </div>
            <div class="bg-black/60 border border-purple-900/70 p-1.5 rounded">
              <div class="text-slate-400 text-[7.5px]">AUTH &amp; CONTRIBUTION</div>
              <div class="text-tacticalAmber font-bold">Mandatory Register &amp; Login</div>
              <div class="text-slate-400 text-[7.5px]">Bearer API Key + HMT &amp; BOINC Credits</div>
            </div>
          </div>
          <ul class="space-y-1 text-[8.5px] text-slate-300">
            <li>• <strong class="text-purple-300">Zero-Install Contribution:</strong> Runs directly in Chrome/Edge/Firefox via `navigator.gpu` WGSL `@compute @workgroup_size(64)` shaders.</li>
            <li>• <strong class="text-tacticalGreen">Live NASA TAP ADQL &amp; BLS Folding:</strong> Folds 16,384 Kepler/TESS/JWST phase points per browser workunit with real-time SNR readout.</li>
            <li>• <strong class="text-tacticalAmber">Unified Account Ledger:</strong> Shares the same Commander profile, HMT balance, and astrobiological discoveries with the Steam Fat Client.</li>
          </ul>
        </div>
        <div class="pt-1.5 border-t border-purple-900/70">
          <div class="flex justify-between text-[8px] text-slate-400 mb-0.5">
            <span>BROWSER WEBGPU STORAGE BUFFER FOOTPRINT</span>
            <span class="text-purple-300 font-bold">1.8 GB / 4.0 GB WASM+WGSL HEAP</span>
          </div>
          <div class="w-full bg-slate-900 h-1.5 rounded overflow-hidden mb-2"><div class="bg-purple-400 h-full" style="width: 45%"></div></div>
          <div class="flex gap-2">
            <a class="flex-1 bg-purple-600 hover:bg-purple-500 text-white font-extrabold py-1 px-2 rounded text-[9.5px] text-center transition" href="/webgpu" target="_blank">
              🌌 LANCER WEBGPU GALAXY (/webgpu)
            </a>
            <a class="bg-amber-950 hover:bg-amber-800 border border-tacticalAmber text-tacticalAmber font-bold py-1 px-2 rounded text-[9.5px]" href="/portal#steam-free-game" target="_blank">
              🐝 PORTAIL UNIFIÉ
            </a>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>"""


def build_stitch_v46_cockpit_html(state_json: str) -> str:
    """Build the upgraded USS Hivemind NCC-31337 v4.6 Cockpit HTML from all 12 Stitch screens in project 10018415280079892710."""
    v2_file = (
        PROJECT_ROOT
        / "steam_dist"
        / "HiveMind_Astrophysics_Exoplanet_Hunter"
        / "stitch_sync_v2"
        / "02_uss_hivemind_ncc-31337_v4_6_-_complete_tactical.html"
    )
    raw = v2_file.read_text(encoding="utf-8")

    # 1. Replace expiring external Google AIDA URLs with local /stitch-assets/ PNGs and ensure .planet-card has relative positioning
    raw = raw.replace(AIDA_GALAXY_URL, LOCAL_GALAXY_ASSET)
    raw = raw.replace(AIDA_K218B_URL, LOCAL_K218B_ASSET)
    raw = raw.replace(AIDA_LHS1140B_URL, LOCAL_LHS1140B_ASSET)
    raw = raw.replace(AIDA_OOLITE_COCKPIT_URL, LOCAL_OOLITE_COCKPIT_ASSET)
    raw = raw.replace('class="planet-card ', 'class="planet-card relative ')

    # 2. Inject the Steam Free Game (AppID 3133700) + BOINC RTX 2070 + WebGPU + 5-Language Switcher bar into the header
    old_status_block = """<div class="flex items-center space-x-3 text-[11px] font-mono">
<span class="bg-cyan-950/80 border border-tacticalCyan/50 px-2 py-0.5 text-tacticalCyan shadow-cyanGlow font-bold">STATUS: NOMINAL</span>
<span class="text-slate-400">BUFFER: <strong class="text-tacticalGreen" id="buffer-val">100%</strong></span>
<span class="text-slate-400">TICK: <strong class="text-tacticalCyan" id="tick-counter">4219.04</strong></span>
<span class="text-slate-400 hidden sm:inline">WARP MATRIX: <strong class="text-tacticalAmber">ALIGNED</strong></span>
</div>"""

    new_status_block = """<div class="flex flex-wrap items-center gap-2 text-[11px] font-mono">
<span id="steam-auth-badge" class="bg-emerald-950/90 border border-tacticalGreen/70 px-2 py-0.5 text-tacticalGreen font-bold rounded">🎮 STEAM FREE GAME (AppID 3133700) • CONNECTING...</span>
<button id="btn-steam-login-contribute" onclick="simulateSteamRegisterLoginAndContribute()" class="bg-cyan-950 hover:bg-cyan-800 border border-tacticalCyan text-tacticalCyan px-2 py-0.5 rounded font-bold cursor-pointer transition">🔑 Steam Login + BOINC GPU</button>
<a id="btn-top-download-exe" href="https://github.com/xaviercallens/HiveMind/releases/download/v2.12.0/HiveMind_Exoplanet_Hunter.exe" target="_blank" rel="noopener" class="bg-emerald-600 hover:bg-emerald-500 border border-white text-black px-2 py-0.5 rounded font-extrabold">⬇ Download .EXE v2.12.0</a>
<a id="btn-top-launch-steam" href="steam://rungameid/3133702" class="bg-amber-950/90 hover:bg-amber-800 border border-tacticalAmber text-tacticalAmber px-2 py-0.5 rounded font-bold">🕹️ Steam 3133702</a>
<a href="/webgpu" target="_blank" class="bg-purple-950/80 hover:bg-purple-800 border border-purple-400 text-purple-200 px-2 py-0.5 rounded font-bold">🌌 WebGPU (/webgpu)</a>
<a href="https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/community" target="_blank" rel="noopener" class="bg-emerald-950/80 hover:bg-emerald-800 border border-emerald-400 text-emerald-300 px-2 py-0.5 rounded font-bold">⚡ GCP BOINC Hub</a>
<span class="text-tacticalAmber font-bold">🏆 <strong id="score-hmt">2,400</strong> HMT</span>
<span class="bg-cyan-950/80 border border-tacticalCyan/50 px-2 py-0.5 text-tacticalCyan shadow-cyanGlow font-bold">STATUS: NOMINAL</span>
<span class="text-slate-400">BUFFER: <strong class="text-tacticalGreen" id="buffer-val">100%</strong></span>
<span class="text-slate-400">TICK: <strong class="text-tacticalCyan" id="tick-counter">4219.04</strong></span>
</div>"""
    raw = raw.replace(old_status_block, new_status_block)

    # 3. Add 5-Language Switcher (FR / ES / DE / EN / ZH) right next to the VOIX button, plus Stage 5 tab (`5. ARCHI & BOINC [5]`)
    old_voice_btn = """<!-- Stage Switcher Ribbon -->
<div class="flex items-center space-x-2">
<!-- Audio Synth Button -->"""

    new_voice_and_lang_btn = """<!-- 5-Language Switcher + Audio Synth + Stage Switcher Ribbon -->
<div class="flex flex-wrap items-center gap-1.5">
<div class="flex items-center rounded border border-cyan-700/70 bg-black/60 p-0.5 text-[10px] font-mono">
  <button id="lang-fr" onclick="setStitchLanguage('fr')" class="px-1.5 py-0.5 bg-tacticalCyan text-black font-bold rounded cursor-pointer">FR</button>
  <button id="lang-es" onclick="setStitchLanguage('es')" class="px-1.5 py-0.5 text-cyan-300 hover:text-white rounded cursor-pointer">ES</button>
  <button id="lang-de" onclick="setStitchLanguage('de')" class="px-1.5 py-0.5 text-cyan-300 hover:text-white rounded cursor-pointer">DE</button>
  <button id="lang-en" onclick="setStitchLanguage('en')" class="px-1.5 py-0.5 text-cyan-300 hover:text-white rounded cursor-pointer">EN</button>
  <button id="lang-zh" onclick="setStitchLanguage('zh')" class="px-1.5 py-0.5 text-cyan-300 hover:text-white rounded cursor-pointer">ZH</button>
</div>
<!-- Audio Synth Button -->"""
    raw = raw.replace(old_voice_btn, new_voice_and_lang_btn)

    old_stage4_btn = """<button class="stage-btn bg-cyan-950/70 hover:bg-cyan-900/80 text-cyan-300 px-2.5 py-1 border border-cyan-800 rounded transition cursor-pointer flex items-center space-x-1" data-stage="4">
<span>4. JWST/LOIS</span>
<span class="text-[9px] text-cyan-400/60 font-mono">[4]</span>
</button>"""

    new_stage4_and_5_btns = """<button class="stage-btn bg-cyan-950/70 hover:bg-cyan-900/80 text-cyan-300 px-2.5 py-1 border border-cyan-800 rounded transition cursor-pointer flex items-center space-x-1" data-stage="4">
<span>4. JWST/LOIS</span>
<span class="text-[9px] text-cyan-400/60 font-mono">[4]</span>
</button>
<button class="stage-btn bg-cyan-950/70 hover:bg-cyan-900/80 text-cyan-300 px-2.5 py-1 border border-cyan-800 rounded transition cursor-pointer flex items-center space-x-1" data-stage="5">
<span>5. ARCHI &amp; BOINC</span>
<span class="text-[9px] text-cyan-400/60 font-mono">[5]</span>
</button>"""
    raw = raw.replace(old_stage4_btn, new_stage4_and_5_btns)

    # 4. Expand Central Viewport height slightly so all 5 Stitch stages render without clipping, and unhide HD 80606 b from behind the right-hand drawer
    playable_controls_banner_html = """<!-- ALWAYS-VISIBLE 4-STAGE CINEMATIC & CONTROLS BANNER (CLAVIER • SOURIS • MANETTE) -->
<section aria-label="Commandes du Jeu et Progression Cinématique" class="bg-[#041022]/95 border-2 border-tacticalCyan/80 rounded p-2 shadow-cyanGlow font-mono text-[10px] select-none" id="playable-controls-banner">
  <div class="flex flex-wrap items-center justify-between gap-2 border-b border-cyan-800/80 pb-1.5 mb-1.5">
    <!-- 4-Step Cinematic Pipeline -->
    <div class="flex flex-wrap items-center gap-1">
      <span class="text-tacticalAmber font-extrabold mr-1">🎬 CINÉMATIQUE :</span>
      <button onclick="switchStage(1)" id="cine-step-1" class="px-2 py-0.5 rounded font-bold bg-tacticalCyan text-black border border-white cursor-pointer transition">1. 🌌 PARCOURIR GALAXIE &amp; CHOISIR PLANÈTE</button>
      <span class="text-cyan-400 font-bold">➔</span>
      <button onclick="launchBeeProbesCinematic()" id="cine-step-2" class="px-2 py-0.5 rounded font-bold bg-cyan-950/90 text-cyan-200 border border-cyan-600 hover:bg-cyan-800 cursor-pointer transition">2. 🤖 ENVOYER ROBOTS (ORBITE)</button>
      <span class="text-cyan-400 font-bold">➔</span>
      <button onclick="completeStage2AndStartLanding()" id="cine-step-3" class="px-2 py-0.5 rounded font-bold bg-cyan-950/90 text-cyan-200 border border-cyan-600 hover:bg-cyan-800 cursor-pointer transition">3. 🛬 ATTERRISSAGE ATMOSPHÉRIQUE</button>
      <span class="text-cyan-400 font-bold">➔</span>
      <button onclick="switchStage(4)" id="cine-step-4" class="px-2 py-0.5 rounded font-bold bg-cyan-950/90 text-cyan-200 border border-cyan-600 hover:bg-cyan-800 cursor-pointer transition">4. 🔬 ANALYSER PLANÈTE &amp; VOTER</button>
    </div>
    <!-- Primary Next-Step CTA + Difficulty + Mystery NASA Planet + Controls Modal Toggle -->
    <div class="flex flex-wrap items-center gap-1.5">
      <button id="btn-main-stage-action" onclick="triggerPrimaryStageAction()" class="bg-emerald-500 hover:bg-emerald-400 text-black font-extrabold px-2.5 py-0.5 rounded border border-white shadow-greenGlow cursor-pointer transition active:scale-95">
        🚀 ENVOYER ROBOTS DE SURVEILLANCE [ESPACE / BOUTON A]
      </button>
      <span id="difficulty-badge" class="bg-amber-950/90 border border-tacticalAmber text-tacticalAmber px-2 py-0.5 rounded font-bold">🔥 DIFFICULTÉ NIV. 1/10</span>
      <button id="btn-new-mystery-planet" onclick="selectRandomNasaMysteryPlanet(true)" class="bg-purple-900 hover:bg-purple-700 border border-purple-300 text-purple-100 px-2 py-0.5 rounded font-bold cursor-pointer transition active:scale-95">
        🎲 NOUVELLE PLANÈTE NASA [M]
      </button>
      <button id="btn-prev-planet" onclick="cyclePlanetSelection(-1)" class="bg-slate-900 hover:bg-slate-700 border border-cyan-600 text-cyan-200 px-1.5 py-0.5 rounded font-bold cursor-pointer transition" title="Planète précédente">◀</button>
      <button id="btn-next-planet" onclick="cyclePlanetSelection(1)" class="bg-slate-900 hover:bg-slate-700 border border-cyan-600 text-cyan-200 px-1.5 py-0.5 rounded font-bold cursor-pointer transition" title="Planète suivante [N / Tab]">▶</button>
      <button id="btn-open-controls-modal" onclick="toggleControlsModal()" class="bg-cyan-900 hover:bg-cyan-700 border border-tacticalCyan text-white px-2 py-0.5 rounded font-bold cursor-pointer transition">
        🎮 AIDE COMMANDES [H]
      </button>
    </div>
  </div>
  <!-- Global Toast Notification Banner visible across ALL 5 Stages -->
  <div id="global-game-toast" class="hidden mb-1.5 bg-amber-500/95 text-black font-extrabold px-3 py-1 rounded border border-white shadow-lg text-[10.5px] tracking-wide text-center transition-all">
    SYSTÈME PRÊT
  </div>
  <!-- Clear, Always-Visible Description of All Controls: Keyboard, Mouse, and Gamepad -->
  <div class="grid grid-cols-1 md:grid-cols-3 gap-2 text-[9.5px] leading-snug">
    <div class="bg-black/75 border border-cyan-700/70 rounded px-2 py-1 flex items-center justify-between">
      <div>
        <span class="text-tacticalCyan font-extrabold">⌨️ CLAVIER :</span>
        <span class="text-slate-200">Piloter : <kbd class="bg-cyan-950 border border-cyan-400 px-1 rounded text-tacticalCyan font-bold">ZQSD</kbd> / <kbd class="bg-cyan-950 border border-cyan-400 px-1 rounded text-tacticalCyan font-bold">WASD</kbd> / <kbd class="bg-cyan-950 border border-cyan-400 px-1 rounded text-tacticalCyan font-bold">↑↓←→</kbd> • Action / Freiner : <kbd class="bg-amber-950 border border-amber-400 px-1 rounded text-tacticalAmber font-bold">ESPACE</kbd> / <kbd class="bg-emerald-950 border border-emerald-400 px-1 rounded text-tacticalGreen font-bold">ENTRÉE</kbd></span>
      </div>
    </div>
    <div class="bg-black/75 border border-emerald-700/70 rounded px-2 py-1 flex items-center justify-between">
      <div>
        <span class="text-tacticalGreen font-extrabold">🖱️ SOURIS :</span>
        <span class="text-slate-200"><strong>Clic sur la carte / radar</strong> : voler ou diriger les robots • <strong>Clic planète</strong> : cibler • <strong>Double-clic planète</strong> : lancer les robots</span>
      </div>
    </div>
    <div class="bg-black/75 border border-amber-600/70 rounded px-2 py-1 flex items-center justify-between">
      <div>
        <span class="text-tacticalAmber font-extrabold" id="gamepad-status-badge">🎮 MANETTE (PRÊTE) :</span>
        <span class="text-slate-200"><strong>Stick Gauche / Croix</strong> : piloter • <strong>Bouton A (✕)</strong> : lancer robots / freiner / approuver • <strong>LB/RB</strong> : changer planète</span>
      </div>
    </div>
  </div>
</section>

<!-- MODAL GUIDE COMPLET DES COMMANDES (CLAVIER • SOURIS • MANETTE) -->
<div id="controls-modal" class="fixed inset-0 z-50 hidden bg-black/80 backdrop-blur-md flex items-center justify-center p-4 font-mono">
  <div class="bg-[#051329] border-2 border-tacticalCyan rounded-lg max-w-3xl w-full p-4 shadow-2xl text-xs space-y-3">
    <div class="flex items-center justify-between border-b border-cyan-700 pb-2">
      <div class="text-sm font-extrabold text-tacticalCyan font-display">🎮 GUIDE COMPLET DES COMMANDES &amp; CINÉMATIQUE EN 4 ÉTAPES</div>
      <button onclick="toggleControlsModal()" class="bg-red-700 hover:bg-red-600 text-white font-bold px-2.5 py-1 rounded cursor-pointer">✕ FERMER [H / ÉCHAP]</button>
    </div>
    <div class="grid grid-cols-1 md:grid-cols-3 gap-3 text-[11px]">
      <div class="bg-black/70 border border-cyan-500/60 p-2.5 rounded space-y-1">
        <div class="text-tacticalCyan font-bold text-xs border-b border-cyan-800 pb-1">⌨️ 1. AU CLAVIER</div>
        <div>• <strong>ZQSD / WASD / Flèches ↑↓←→</strong> : Déplacer le vaisseau USS HiveMind (Étape 1), piloter les robots Bee-MoE (Étape 2), aligner l'atterrisseur (Étape 3).</div>
        <div>• <strong>[ESPACE]</strong> : Envoyer les robots (Étape 1), booster la synchro (Étape 2), allumer les rétro-fusées pour freiner (Étape 3), Approuver la planète (Étape 4).</div>
        <div>• <strong>[ENTRÉE]</strong> : Passer à l'étape suivante / Atterrissage assisté automatique.</div>
        <div>• <strong>[M]</strong> : Générer une nouvelle planète mystère NASA avec vraies données.</div>
        <div>• <strong>[1] [2] [3] [4] [5]</strong> : Changer d'étape directement.</div>
      </div>
      <div class="bg-black/70 border border-emerald-500/60 p-2.5 rounded space-y-1">
        <div class="text-tacticalGreen font-bold text-xs border-b border-emerald-800 pb-1">🖱️ 2. À LA SOURIS</div>
        <div>• <strong>Étape 1 (Galaxie)</strong> : Cliquez n'importe où sur la galaxie pour y faire voler le vaisseau. Cliquez sur une planète pour la cibler, ou <strong>double-cliquez</strong> dessus (ou cliquez sur le bouton vert <em>🚀 ENVOYER ROBOTS</em>) pour envoyer les robots !</div>
        <div>• <strong>Étape 2 (Orbite Bee)</strong> : Déplacez ou cliquez dans le radar orbital central pour diriger vos robots vers l'<strong>Anneau Vert</strong> en évitant les astéroïdes.</div>
        <div>• <strong>Étape 3 (Atterrissage)</strong> : Déplacez la souris sur le tunnel d'approche pour centrer le vaisseau et cliquez sur <strong>🔥 FREINER</strong> ou <strong>🛬 POSER</strong>.</div>
        <div>• <strong>Étape 4 (Analyse JWST)</strong> : Cliquez sur les molécules (H₂O, CH₄, CO₂, O₃) puis sur <strong>✅ APPROUVER</strong> ou <strong>🚨 REJETER</strong>.</div>
      </div>
      <div class="bg-black/70 border border-amber-500/60 p-2.5 rounded space-y-1">
        <div class="text-tacticalAmber font-bold text-xs border-b border-amber-800 pb-1">🎮 3. À LA MANETTE (GAMEPAD)</div>
        <div>• <strong>Stick Analogique Gauche / D-Pad</strong> : Piloter le vaisseau (Étape 1), les robots (Étape 2) et l'atterrisseur (Étape 3) à 60 FPS.</div>
        <div>• <strong>Bouton A (Xbox) / Croix ✕ (PS)</strong> : Envoyer les robots (Étape 1), synchroniser (Étape 2), freiner avec les rétro-fusées (Étape 3), Approuver la planète (Étape 4).</div>
        <div>• <strong>Bouton X / B (Rond ○)</strong> : Tirer une planète mystère NASA (Étape 1), atterrissage assisté (Étape 3), Rejeter faux positif (Étape 4).</div>
        <div>• <strong>Gâchettes LB / RB (L1 / R1)</strong> : Parcourir les planètes ou passer d'une étape à l'autre.</div>
      </div>
    </div>
    <div class="bg-emerald-950/60 border border-emerald-400 p-2 rounded text-[11px] text-emerald-200 flex items-center justify-between">
      <span>💡 <strong>Astuce :</strong> Chaque planète validée augmente le niveau de difficulté (plus de ceintures d'astéroïdes képlériennes, éruptions solaires CME et vents atmosphériques réels) !</span>
      <button onclick="toggleControlsModal()" class="bg-emerald-500 text-black font-extrabold px-3 py-1 rounded cursor-pointer">C'EST PARTI !</button>
    </div>
  </div>
</div>
"""
    raw = raw.replace(
        '<!-- CENTRAL VIEWPORT (MULTI-STAGE GAME ENGINE) -->',
        playable_controls_banner_html + '\n<!-- CENTRAL VIEWPORT (MULTI-STAGE GAME ENGINE) -->',
    )
    raw = raw.replace(
        'class="relative w-full h-[400px] xl:h-[440px] rounded border border-cyan-600/50 overflow-hidden bg-black shadow-2xl"',
        'class="relative w-full h-[470px] xl:h-[490px] rounded border border-cyan-600/50 overflow-hidden bg-black shadow-2xl"',
    )
    raw = raw.replace(
        'class="planet-node absolute top-[75px] right-[180px] flex flex-col items-center cursor-pointer transition hover:scale-110" data-planet="hd80606b"',
        'class="planet-node absolute top-[62px] right-[365px] flex flex-col items-center cursor-pointer transition hover:scale-110" data-planet="hd80606b"',
    )

    # 5. Add System 9 (KOI-Cand 14), System 10 (KOI-Bin #02), and Top-Left Sub-Spatial Scan Box to Stage 1 (`#viewport-nodes`)
    target_s1_end = """<span class="node-label text-[11px] text-cyan-200 bg-black/85 px-1.5 py-0.5 border border-cyan-900/60 mt-0.5 font-mono rounded">Kepler-186 f</span>
</div>
</div>"""
    replacement_s1_end = (
        """<span class="node-label text-[11px] text-cyan-200 bg-black/85 px-1.5 py-0.5 border border-cyan-900/60 mt-0.5 font-mono rounded">Kepler-186 f</span>
</div>
"""
        + STAGE_1_SUBSPATIAL_HUD_AND_EXTRA_NODES_HTML
        + "\n</div>"
    )
    raw = raw.replace(target_s1_end, replacement_s1_end)

    # 6. Replace Stages 2, 3, and 4 with the full Stitch Screen 00, Screen 10+06, Screen 01, and append Stage 5 (Screen 04)
    start_s2 = raw.find("<!-- STAGE 2 VIEW: ORBIT BEE SWARM & LAGRANGE RING VIEW -->")
    end_s4 = raw.find("<!-- TACTICAL VOICE COMMS OVERLAY BOX -->")
    if start_s2 != -1 and end_s4 != -1:
        combined_stages_2_to_5 = (
            STAGE_2_FULL_ORBIT_BEE_HTML
            + "\n"
            + STAGE_3_FULL_OOLITE_BRIDGE_HTML
            + "\n"
            + STAGE_4_FULL_JWST_BERYLLIUM_HTML
            + "\n"
            + STAGE_5_DUAL_ENGINE_ARCHI_HTML
            + "\n"
        )
        raw = raw[:start_s2] + combined_stages_2_to_5 + raw[end_s4:]

    # 7. Inject Steam registration/login + BOINC GPU contribution + 5-Language localization + Full 60 FPS Playable 4-Stage Game Loop (Keyboard + Mouse + Gamepad)
    extra_js = f"""
  // === HIVEMIND LIVE BACKEND, ALL 12 STITCH SCREENS SYNC, STEAM + BOINC RTX 2070, & 60 FPS PLAYABLE 4-STAGE ENGINE ===
  window.HIVEMIND_EMBEDDED_STATE = {state_json};
  let scoreHmt = 2400;
  let currentLang = 'fr';
  let retroThrustPct = 75;
  let difficultyLevel = 1;
  let completedPlanetsCount = 0;
  const completedPlanetsSet = new Set();

  // Register Stage 5 in stagePanels, stageDescriptions, and stageProgressLabels
  stagePanels[5] = document.getElementById('viewport-stage-5');
  stageDescriptions[5] = "ÉTAPE 5 // ARCHITECTURE DUAL-ENGINE // FAT CLIENT STEAM (UE5/UNITY + RTX 2070 BOINC) vs. WEBGPU ZERO-INSTALL (/webgpu)";
  stageProgressLabels[5] = "STAGE 5/5 [DUAL-ENGINE & BOINC]";

  // Synthesized WebAudio Sci-Fi Sound Effects (Zero external audio files required)
  let sfxCtx = null;
  function playSciFiTone(freqStart, freqEnd, durationMs, type = 'sine', gainVal = 0.08) {{
    try {{
      if (!sfxCtx) sfxCtx = new (window.AudioContext || window.webkitAudioContext)();
      if (sfxCtx.state === 'suspended') sfxCtx.resume();
      const osc = sfxCtx.createOscillator();
      const gain = sfxCtx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freqStart, sfxCtx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(Math.max(40, freqEnd), sfxCtx.currentTime + durationMs / 1000);
      gain.gain.setValueAtTime(gainVal, sfxCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, sfxCtx.currentTime + durationMs / 1000);
      osc.connect(gain);
      gain.connect(sfxCtx.destination);
      osc.start();
      osc.stop(sfxCtx.currentTime + durationMs / 1000);
    }} catch (e) {{}}
  }}

  // Pool of 8 Unseen Real NASA Exoplanets (Selected each new game or via [M] / button)
  const UNSEEN_NASA_PLANETS = [
    {{
      key: 'mystery_nasa',
      realId: 'SECTOR-01-TRAPPIST-1E',
      name: 'Proxima Centauri b',
      subtitle: 'Proxima Centauri b (Terre Proche ★ NASA)',
      badge: 'PROXIMA Centauri b [4.2 AL]',
      hab: 'HABITABILITÉ: 87%',
      dist: '4.24 al (1.30 pc)',
      atm: 'N₂ - CO₂ - H₂O (Aurores)',
      temp: '~234 K (-39°C)',
      ra: '217.42°', dec: '-62.68°',
      nodeX: 255, nodeY: 185,
      rvInstrument: 'ESPRESSO / HARPS (K = 1.38 m/s, e = 0.02)',
      rvDelta: 'Δv = +1.38 m/s [PROXIMA CONFIRMÉ]',
      rvPeriod: 'P = 11.186 j (ZONE HABITABLE)',
      rvOrbitStat: 'VERROUILLÉE (e = 0.02)',
      transitDuration: 'DURÉE DE TRANSIT: 1.82 h',
      transitDepth: 'DEPTH: 4950 ppm (Rp/R★ = 0.070)',
      transitRadius: '1.07 R⊕ (ROCHEUSE DENSE)',
      transitDensity: '5.3 g/cm³ (NOYAU FERREUX)',
      bioConfidence: 'BIOSIGNATURE INDICE: 87.4%',
      dmsTag: '● PEAK DETECT: H₂O & O₃ @ 9.6µm [PROXIMA-CEN]',
      adqlQuery: "SELECT pl_name, pl_rade, pl_bmasse, st_teff FROM ps WHERE pl_name = 'Proxima Cen b';",
      landingZone: 'PROXIMA TERMINATOR BASIN',
      altitudeInit: 3420,
      gravityG: 1.09, scaleHeightKm: 7.8, pressureBar: 1.4, windShearMs: 38, asteroidBelts: 2, keplerVelKms: 47.3,
      isFalsePositive: false,
      asteroidPhysicsFr: "☄️ PHYSIQUE PROXIMA CEN b : Ceinture de poussières froides (0.4 UA) + Éruptions stellaires M5.5V (CME Alfvéniques à 950 km/s).",
      atmoPhysicsFr: "🌪️ ATMOSPHÈRE PROXIMA b : g = 1.09 g, H = 7.8 km, P = 1.4 bar. Rotation synchrone créant un jet-stream de 38 m/s au terminateur.",
      lifeAnalysisFr: "Analyse Proxima Centauri b (4.24 al, 1.07 R⊕, 234 K) : exoterre la plus proche du Soleil ! Vie possible : océan liquide permanent dans l'anneau crépusculaire et biosphère protégée sous glace des éruptions UV."
    }},
    {{
      key: 'mystery_nasa',
      realId: 'SECTOR-04-HD-80606B',
      name: '55 Cancri e (Janssen)',
      subtitle: '55 Cancri e (Super-Terre de Lave ★ NASA)',
      badge: '55 Cancri e [OCÉAN MAGMA]',
      hab: 'HABITABILITÉ: 4%',
      dist: '41.0 al (12.59 pc)',
      atm: 'SiO - CO - CO₂ - Vapeur Silicate',
      temp: '~2400 K (+2127°C)',
      ra: '133.15°', dec: '+28.33°',
      nodeX: 255, nodeY: 185,
      rvInstrument: 'HIRES / HARPS-N (K = 6.02 m/s, e = 0.05)',
      rvDelta: 'Δv = +6.02 m/s [ULTRA-COURTE PÉRIODE]',
      rvPeriod: 'P = 0.7365 j (17.6 HEURES)',
      rvOrbitStat: 'ULTRA-PROCHE (a = 0.0154 UA)',
      transitDuration: 'DURÉE DE TRANSIT: 1.58 h',
      transitDepth: 'DEPTH: 380 ppm (Rp/R★ = 0.019)',
      transitRadius: '1.88 R⊕ (MANTEAU CARBONE/LAVE)',
      transitDensity: '6.66 g/cm³ (ULTRA-DENSE)',
      bioConfidence: 'BIOSIGNATURE INDICE: 4.1%',
      dmsTag: '● JWST MIRI DETECT: CO/CO₂ VOLCANIC OUTGASSING [2400 K]',
      adqlQuery: "SELECT pl_name, pl_rade, pl_bmasse, pl_eqt FROM ps WHERE pl_name = '55 Cnc e';",
      landingZone: 'CRATÈRE DIAMANT JANSSEN',
      altitudeInit: 3420,
      gravityG: 2.26, scaleHeightKm: 28.0, pressureBar: 12.5, windShearMs: 92, asteroidBelts: 3, keplerVelKms: 228.5,
      isFalsePositive: false,
      asteroidPhysicsFr: "☄️ PHYSIQUE 55 CANCRI e : À 0.015 UA (v = 228 km/s), la limite de Roche arrache des panaches de silicates en fusion et bolides réfractaires.",
      atmoPhysicsFr: "🌪️ ATMOSPHÈRE 55 CANCRI e : Forte gravité g = 2.26 g (22.2 m/s²), P = 12.5 bar, vents supersoniques de vapeur minérale à 92 m/s !",
      lifeAnalysisFr: "Analyse 55 Cancri e (1.88 R⊕, 8.0 M⊕, 2400 K, P = 17.7 h) : océan global de magma côté jour observé par JWST. Vie biologique carbonée impossible."
    }},
    {{
      key: 'mystery_nasa',
      realId: 'SECTOR-02-K2-18B',
      name: 'GJ 1214 b (Enaiposha)',
      subtitle: 'GJ 1214 b (Monde-Vapeur Super-Vénus ★ NASA)',
      badge: 'GJ 1214 b [MONDE-EAU/VAPEUR]',
      hab: 'HABITABILITÉ: 38%',
      dist: '47.7 al (14.64 pc)',
      atm: 'H₂O Vapeur - CH₄ - Brumes KCl/ZnS',
      temp: '~559 K (+286°C)',
      ra: '258.82°', dec: '+04.96°',
      nodeX: 255, nodeY: 185,
      rvInstrument: 'HARPS / ESPRESSO (K = 10.9 m/s, e = 0.06)',
      rvDelta: 'Δv = +10.9 m/s [SEUIL CONFIRMÉ]',
      rvPeriod: 'P = 1.5804 j (ORBITE PROCHE)',
      rvOrbitStat: 'SYNCHRONE (e = 0.06)',
      transitDuration: 'DURÉE DE TRANSIT: 0.88 h',
      transitDepth: 'DEPTH: 13500 ppm (Rp/R★ = 0.116)',
      transitRadius: '2.74 R⊕ (ATMOSPHÈRE VAPEUR DENSE)',
      transitDensity: '2.20 g/cm³ (RICHE EN EAU)',
      bioConfidence: 'BIOSIGNATURE INDICE: 41.0%',
      dmsTag: '● JWST MIRI PHASE CURVE: HIGH-METALLICITY H₂O STEAM HAZE',
      adqlQuery: "SELECT pl_name, pl_rade, pl_bmasse, pl_eqt FROM ps WHERE pl_name = 'GJ 1214 b';",
      landingZone: 'PLATEAU AÉROSOL ENAIPOSHA',
      altitudeInit: 3420,
      gravityG: 1.08, scaleHeightKm: 32.0, pressureBar: 45.0, windShearMs: 68, asteroidBelts: 2, keplerVelKms: 82.0,
      isFalsePositive: false,
      asteroidPhysicsFr: "☄️ PHYSIQUE GJ 1214 b : Anneau d'exocomètes riches en glace d'eau sublimée par le flux XUV de la naine rouge.",
      atmoPhysicsFr: "🌪️ ATMOSPHÈRE GJ 1214 b : g = 1.08 g, H = 32 km, P = 45 bar. Brumes réflectives de sels KCl/ZnS observées par JWST MIRI et cisaillement de 68 m/s.",
      lifeAnalysisFr: "Analyse GJ 1214 b (2.74 R⊕, 8.17 M⊕, 559 K) : monde-vapeur entouré de brumes métalliques denses. Vie possible : micro-organismes thermophiles dans la haute troposphère à 330 K."
    }},
    {{
      key: 'mystery_nasa',
      realId: 'SECTOR-05-TOI-700D',
      name: 'LP 890-9 c (SPECULOOS-2c)',
      subtitle: 'LP 890-9 c (Super-Terre Tempérée ★ NASA)',
      badge: 'LP 890-9 c [ZONE HABITABLE]',
      hab: 'HABITABILITÉ: 89%',
      dist: '105.0 al (32.2 pc)',
      atm: 'H₂O - N₂ - CO₂',
      temp: '~272 K (-1°C)',
      ra: '064.19°', dec: '-28.31°',
      nodeX: 255, nodeY: 185,
      rvInstrument: 'SPECULOOS / TESS (K = 2.85 m/s, e = 0.02)',
      rvDelta: 'Δv = +2.85 m/s [ZONE HABITABLE]',
      rvPeriod: 'P = 8.457 j (NAINE ULTRA-FROIDE M6V)',
      rvOrbitStat: 'CIRCULAIRE (e = 0.02)',
      transitDuration: 'DURÉE DE TRANSIT: 1.46 h',
      transitDepth: 'DEPTH: 6250 ppm (Rp/R★ = 0.079)',
      transitRadius: '1.37 R⊕ (SUPER-TERRE ROCHEUSE)',
      transitDensity: '5.4 g/cm³ (TERRESTRE)',
      bioConfidence: 'BIOSIGNATURE INDICE: 91.5%',
      dmsTag: '● JWST TARGET: TEMPERATE H₂O/CO₂ TRANSMISSION SPECTRUM',
      adqlQuery: "SELECT pl_name, pl_rade, pl_eqt, st_teff FROM ps WHERE pl_name = 'LP 890-9 c';",
      landingZone: 'VALLÉE SPECULOOS AUSTRALIS',
      altitudeInit: 3420,
      gravityG: 1.34, scaleHeightKm: 7.2, pressureBar: 1.8, windShearMs: 36, asteroidBelts: 2, keplerVelKms: 39.4,
      isFalsePositive: false,
      asteroidPhysicsFr: "☄️ PHYSIQUE LP 890-9 c : Résonance orbitale proche avec LP 890-9 b concentrant des planétésimaux glacés aux points de Lagrange L4/L5.",
      atmoPhysicsFr: "🌪️ ATMOSPHÈRE LP 890-9 c : g = 1.34 g, H = 7.2 km, P = 1.8 bar, T_eq = 272 K. Circulation nuageuse humide stable.",
      lifeAnalysisFr: "Analyse LP 890-9 c (1.37 R⊕, 272 K, étoile M6V à 2871 K) : l'une des cibles habitables les plus prometteuses après TRAPPIST-1. Vie possible : océans tempérés et végétation photosynthétique."
    }},
    {{
      key: 'mystery_nasa',
      realId: 'SECTOR-07-WASP-39B',
      name: 'HD 189733 b',
      subtitle: 'HD 189733 b (Géante Bleue Pluie de Verre ★ NASA)',
      badge: 'HD 189733 b [VENTS 2400 m/s]',
      hab: 'HABITABILITÉ: 1%',
      dist: '64.5 al (19.78 pc)',
      atm: 'H₂ - He - H₂O - Silicates MgSiO₃',
      temp: '~1201 K (+928°C)',
      ra: '300.18°', dec: '+22.71°',
      nodeX: 255, nodeY: 185,
      rvInstrument: 'ELODIE / SOPHIE (K = 205.0 m/s, e = 0.004)',
      rvDelta: 'Δv = +205.0 m/s [JUPITER CHAUD]',
      rvPeriod: 'P = 2.2185 j (ORBITE 0.031 UA)',
      rvOrbitStat: 'SYNCHRONE (e = 0.004)',
      transitDuration: 'DURÉE DE TRANSIT: 1.80 h',
      transitDepth: 'DEPTH: 24100 ppm (Rp/R★ = 0.155)',
      transitRadius: '12.7 R⊕ (JUPITER CHAUD BLEU COBALT)',
      transitDensity: '0.96 g/cm³ (GÉANTE GAZEUSE)',
      bioConfidence: 'BIOSIGNATURE INDICE: 2.0%',
      dmsTag: '● JWST NIRCam DETECT: H₂O + H₂S + SILICATE GLASS CLOUDS',
      adqlQuery: "SELECT pl_name, pl_rade, pl_bmasse, pl_eqt FROM ps WHERE pl_name = 'HD 189733 b';",
      landingZone: 'AÉROSTAT COUCHE COBALT 1 BAR',
      altitudeInit: 3420,
      gravityG: 2.18, scaleHeightKm: 210.0, pressureBar: 10.0, windShearMs: 98, asteroidBelts: 3, keplerVelKms: 152.0,
      isFalsePositive: false,
      asteroidPhysicsFr: "☄️ PHYSIQUE HD 189733 b : Choc d'étrave magnétosphérique intense et poussières de silicates vaporisées à 0.031 UA.",
      atmoPhysicsFr: "🌪️ ATMOSPHÈRE HD 189733 b : g = 2.18 g, H = 210 km. Vents équatoriaux supersoniques mesurés à 8 700 km/h (2 400 m/s) faisant pleuvoir du verre fondu horizontalement !",
      lifeAnalysisFr: "Analyse HD 189733 b (12.7 R⊕, 1201 K, couleur bleu cobalt due aux gouttelettes de verre MgSiO₃) : tempêtes extrêmes et H₂S détecté par JWST. Vie impossible."
    }},
    {{
      key: 'mystery_nasa',
      realId: 'SECTOR-06-LHS-1140B',
      name: 'TOI-1452 b',
      subtitle: 'TOI-1452 b (Super-Terre Océanique ★ NASA)',
      badge: 'TOI-1452 b [OCÉAN 30% MASSE]',
      hab: 'HABITABILITÉ: 86%',
      dist: '99.5 al (30.5 pc)',
      atm: 'H₂O - N₂ - CO₂',
      temp: '~298 K (+25°C)',
      ra: '290.17°', dec: '+73.19°',
      nodeX: 255, nodeY: 185,
      rvInstrument: 'SPIRou / TESS (K = 3.45 m/s, e = 0.01)',
      rvDelta: 'Δv = +3.45 m/s [MONDE OCÉANIQUE]',
      rvPeriod: 'P = 11.062 j (TEMPÉRÉE 298 K)',
      rvOrbitStat: 'CIRCULAIRE (e = 0.01)',
      transitDuration: 'DURÉE DE TRANSIT: 2.02 h',
      transitDepth: 'DEPTH: 3320 ppm (Rp/R★ = 0.058)',
      transitRadius: '1.67 R⊕ (MONDE OCÉANIQUE PROFOND)',
      transitDensity: '5.6 g/cm³ (30% H₂O EN MASSE)',
      bioConfidence: 'BIOSIGNATURE INDICE: 93.0%',
      dmsTag: '● SPIRou + TESS: DEEP LIQUID WATER OCEAN CANDIDATE [298 K]',
      adqlQuery: "SELECT pl_name, pl_rade, pl_bmasse, pl_eqt FROM ps WHERE pl_name = 'TOI-1452 b';",
      landingZone: 'ATOLL HYDROTHERMAL DRACO',
      altitudeInit: 3420,
      gravityG: 1.72, scaleHeightKm: 8.8, pressureBar: 3.2, windShearMs: 42, asteroidBelts: 2, keplerVelKms: 48.0,
      isFalsePositive: false,
      asteroidPhysicsFr: "☄️ PHYSIQUE TOI-1452 b : Système binaire d'étoiles naines rouges induisant des perturbations séculaires de Kozai-Lidov sur les comètes glacées.",
      atmoPhysicsFr: "🌪️ ATMOSPHÈRE TOI-1452 b : g = 1.72 g, H = 8.8 km, P = 3.2 bar, T_eq = 298 K (+25°C). Mousson océanique globale.",
      lifeAnalysisFr: "Analyse TOI-1452 b (1.67 R⊕, 4.82 M⊕, 298 K) : planète-océan dont l'eau pourrait représenter jusqu'à 30% de la masse totale. Vie possible : écosystèmes marins pélagiques et récifs hydrothermaux."
    }}
  ];

  // Real Astrophysics Database for Base 10 Stitch Planets
  const PLANET_PHYSICS_DB = {{
    trappist1e: {{
      gravityG: 0.93, scaleHeightKm: 8.1, pressureBar: 1.1, windShearMs: 28, asteroidBelts: 1, keplerVelKms: 42.1, isFalsePositive: false,
      asteroidPhysicsFr: "☄️ PHYSIQUE TRAPPIST-1e : Chaîne de résonance laplacienne 8:5:3:2 à 0.029 UA. Les débris glacés voyagent à v = √(GM★/r) = 42.1 km/s.",
      atmoPhysicsFr: "🌪️ ATMOSPHÈRE TRAPPIST-1e : Gravité g = 0.93 g (9.1 m/s²), hauteur d'échelle H = 8.1 km, P = 1.1 bar. Courant-jet crépusculaire de 28 m/s.",
      lifeAnalysisFr: "Analyse TRAPPIST-1e : monde rocheux (0.92 R⊕, 251 K, orbite 6.1 j, transit 4890 ppm). Vie possible : océans crépusculaires propices aux algues photosynthétiques sous lumière rouge."
    }},
    k218b: {{
      gravityG: 1.27, scaleHeightKm: 34.0, pressureBar: 8.5, windShearMs: 52, asteroidBelts: 2, keplerVelKms: 33.8, isFalsePositive: false,
      asteroidPhysicsFr: "☄️ PHYSIQUE K2-18 b : Ceinture d'exocomètes riches en glaces CH₄/CO₂ aux points de Lagrange L4/L5 (a = 0.143 UA).",
      atmoPhysicsFr: "🌪️ ATMOSPHÈRE K2-18 b : Monde hycéen (g = 1.27 g, H = 34 km, enveloppe H₂/He sur océan global). Vents stratosphériques de 52 m/s.",
      lifeAnalysisFr: "Analyse K2-18 b : monde océanique hycéen (2.61 R⊕, 279 K, CH₄ + CO₂ détectés par JWST). Vie possible : phytoplancton produisant du sulfure de diméthyle (DMS) dans son océan global."
    }},
    kepler11f: {{
      gravityG: 0.32, scaleHeightKm: 95.0, pressureBar: 18.0, windShearMs: 64, asteroidBelts: 2, keplerVelKms: 58.4, isFalsePositive: false,
      asteroidPhysicsFr: "☄️ PHYSIQUE KEPLER-11 f : Système ultra-compact de 6 planètes coplanaires générant des sillons de poussières képlériennes à 58.4 km/s.",
      atmoPhysicsFr: "🌪️ ATMOSPHÈRE KEPLER-11 f : Faible gravité g = 0.32 g, vaste enveloppe H₂/He (H = 95 km) et cisaillement de vent à 64 m/s.",
      lifeAnalysisFr: "Analyse Kepler-11 f : Mini-Neptune gazeuse peu dense (2.49 R⊕, 0.71 g/cm³, 544 K). Vie possible : sans surface solide, seuls des microbes aéroportés dans les nuages tempérés survivraient."
    }},
    hd80606b: {{
      gravityG: 9.68, scaleHeightKm: 18.0, pressureBar: 35.0, windShearMs: 96, asteroidBelts: 3, keplerVelKms: 168.0, isFalsePositive: false,
      asteroidPhysicsFr: "☄️ PHYSIQUE HD 80606 b : Excentricité extrême e = 0.933 ! Au périastre (0.03 UA), la vitesse orbitale atteint 168 km/s à travers une dense nuée de bolides.",
      atmoPhysicsFr: "🌪️ ATMOSPHÈRE HD 80606 b : Gravité colossale g = 9.68 g (94.9 m/s²) ! Onde de choc radiative chauffant l'atmosphère de 400 K à 1200 K avec vents à 96 m/s.",
      lifeAnalysisFr: "Analyse HD 80606 b : géante excentrique (e = 0.93, Doppler 476 m/s, tempêtes à 1200 K). Vie possible : chaleur extrême ; seuls des microbes sous la glace d'une exolune survivraient."
    }},
    toi700d: {{
      gravityG: 1.32, scaleHeightKm: 7.4, pressureBar: 1.5, windShearMs: 32, asteroidBelts: 1, keplerVelKms: 35.2, isFalsePositive: false,
      asteroidPhysicsFr: "☄️ PHYSIQUE TOI-700 d : Anneau de micrométéorites silicatées en résonance 4:3 à 0.163 UA autour de la naine calme TOI-700.",
      atmoPhysicsFr: "🌪️ ATMOSPHÈRE TOI-700 d : g = 1.32 g (12.9 m/s²), H = 7.4 km, P = 1.5 bar. Brises thermiques régulières de 32 m/s le long du terminateur.",
      lifeAnalysisFr: "Analyse TOI-700 d : planète rocheuse habitable TESS (1.14 R⊕, 269 K, orbite 37.4 j). Vie possible : mers tempérées le long du terminateur soutenant une vie aquatique complexe."
    }},
    lhs1140b: {{
      gravityG: 1.53, scaleHeightKm: 6.5, pressureBar: 2.4, windShearMs: 35, asteroidBelts: 2, keplerVelKms: 44.6, isFalsePositive: false,
      asteroidPhysicsFr: "☄️ PHYSIQUE LHS 1140 b : Super-Terre à 0.095 UA entourée de deux anneaux de planétésimaux riches en glace d'eau et silicates (v = 44.6 km/s).",
      atmoPhysicsFr: "🌪️ ATMOSPHÈRE LHS 1140 b : Gravité g = 1.53 g (15.0 m/s²), H = 6.5 km, atmosphère N₂/CO₂/H₂O (2.4 bar) confirmée par JWST NIRISS.",
      lifeAnalysisFr: "Analyse LHS 1140 b : Super-Terre dense (1.43 R⊕, 5.6 g/cm³, transit 5340 ppm, 235 K). Vie possible : océan circulaire « Eyeball » côté jour (20°C) et sources hydrothermales nourrissant la vie marine."
    }},
    wasp39b: {{
      gravityG: 0.44, scaleHeightKm: 680.0, pressureBar: 6.0, windShearMs: 85, asteroidBelts: 3, keplerVelKms: 141.0, isFalsePositive: false,
      asteroidPhysicsFr: "☄️ PHYSIQUE WASP-39 b : Orbite ultra-proche (0.049 UA, v = 141 km/s) baignée dans un tore de soufre photochimique et d'éjections coronales.",
      atmoPhysicsFr: "🌪️ ATMOSPHÈRE WASP-39 b : Saturne chaude gonflée (g = 0.44 g, H = 680 km, 1166 K). Nuages photochimiques de SO₂ et CO₂ avec vents de 85 m/s.",
      lifeAnalysisFr: "Analyse WASP-39 b : Saturne chaude gonflée (14.2 R⊕, 1166 K, nuages SO₂ et CO₂ détectés par JWST). Vie possible : à 1166 K, la chaleur brise l'ADN ; vie impossible."
    }},
    kepler186f: {{
      gravityG: 1.18, scaleHeightKm: 6.8, pressureBar: 2.8, windShearMs: 40, asteroidBelts: 2, keplerVelKms: 29.8, isFalsePositive: false,
      asteroidPhysicsFr: "☄️ PHYSIQUE KEPLER-186f : Bord externe de la zone habitable (0.432 UA) croisant une ceinture de Kuiper interne d'astéroïdes carbonés.",
      atmoPhysicsFr: "🌪️ ATMOSPHÈRE KEPLER-186f : g = 1.18 g, T_eq = 188 K, P = 2.8 bar (effet de serre CO₂/N₂). Rafales polaires de 40 m/s.",
      lifeAnalysisFr: "Analyse Kepler-186f : monde rocheux froid (1.17 R⊕, 188 K, orbite habitable 130 j). Vie possible : plantes sombres captant l'infrarouge et bactéries géothermiques près des volcans."
    }},
    koi14: {{
      gravityG: 1.10, scaleHeightKm: 9.0, pressureBar: 1.2, windShearMs: 45, asteroidBelts: 2, keplerVelKms: 38.0, isFalsePositive: true,
      asteroidPhysicsFr: "☄️ PHYSIQUE KOI-CAND 14 : Champ d'astéroïdes dense masquant une binaire d'arrière-plan (vitesse radiale Doppler plate K = 0.04 m/s).",
      atmoPhysicsFr: "🌪️ ATMOSPHÈRE KOI-CAND 14 : Signal de transit dilué (420 ppm) sans oscillation Doppler mesurable — attention au faux positif astrophysique !",
      lifeAnalysisFr: "Analyse KOI-Cand 14 : candidat suspect (transit 420 ppm mais Doppler plat 0.04 m/s). Faux positif probable par étoile binaire d'arrière-plan (votez REJETER)."
    }},
    koibin: {{
      gravityG: 4.50, scaleHeightKm: 120.0, pressureBar: 25.0, windShearMs: 95, asteroidBelts: 3, keplerVelKms: 195.0, isFalsePositive: true,
      asteroidPhysicsFr: "☄️ PHYSIQUE KOI-BIN #02 : Binaire à éclipses stellaire ! Courants d'accrétion de plasma et débris de marée à 195 km/s.",
      atmoPhysicsFr: "🌪️ ATMOSPHÈRE KOI-BIN #02 : Courbe en V profonde (58 400 ppm) et Doppler géant K = 1 420 m/s ($M > 80 M_J$) : c'est une étoile naine, pas une planète !",
      lifeAnalysisFr: "Analyse KOI-Bin #02 : faux positif binaire à éclipses (transit en V 58400 ppm, Doppler 1420 m/s). Vie impossible : il s'agit de deux étoiles brûlantes (votez REJETER)."
    }}
  }};

  let currentMysteryIndex = -1;
  window.selectRandomNasaMysteryPlanet = function(switchToIt = false) {{
    let nextIdx = Math.floor(Math.random() * UNSEEN_NASA_PLANETS.length);
    if (nextIdx === currentMysteryIndex) nextIdx = (nextIdx + 1) % UNSEEN_NASA_PLANETS.length;
    currentMysteryIndex = nextIdx;
    const mp = UNSEEN_NASA_PLANETS[currentMysteryIndex];
    const baseTemplate = Object.assign({{}}, PLANET_DATA['lhs1140b'] || {{}});
    PLANET_DATA['mystery_nasa'] = Object.assign(baseTemplate, {{
      name: mp.name,
      subtitle: mp.subtitle,
      badge: mp.badge,
      hab: mp.hab,
      dist: mp.dist,
      atm: mp.atm,
      temp: mp.temp,
      ra: mp.ra,
      dec: mp.dec,
      nodeX: mp.nodeX,
      nodeY: mp.nodeY,
      img: '{LOCAL_K218B_ASSET}',
      rvInstrument: mp.rvInstrument,
      rvDelta: mp.rvDelta,
      rvPeriod: mp.rvPeriod,
      rvOrbitStat: mp.rvOrbitStat,
      transitDuration: mp.transitDuration,
      transitDepth: mp.transitDepth,
      transitRadius: mp.transitRadius,
      transitDensity: mp.transitDensity,
      bioConfidence: mp.bioConfidence,
      dmsTag: mp.dmsTag,
      adqlQuery: mp.adqlQuery,
      landingZone: mp.landingZone,
      altitudeInit: mp.altitudeInit,
      transcript: mp.lifeAnalysisFr
    }});
    PLANET_PHYSICS_DB['mystery_nasa'] = {{
      gravityG: mp.gravityG,
      scaleHeightKm: mp.scaleHeightKm,
      pressureBar: mp.pressureBar,
      windShearMs: mp.windShearMs,
      asteroidBelts: mp.asteroidBelts,
      keplerVelKms: mp.keplerVelKms,
      isFalsePositive: mp.isFalsePositive,
      asteroidPhysicsFr: mp.asteroidPhysicsFr,
      atmoPhysicsFr: mp.atmoPhysicsFr,
      lifeAnalysisFr: mp.lifeAnalysisFr,
      realId: mp.realId
    }};
    const lbl = document.getElementById('mystery-node-label');
    if (lbl) lbl.textContent = '★ ' + mp.name;
    if (switchToIt) {{
      playSciFiTone(440, 880, 220, 'triangle');
      selectPlanet('mystery_nasa');
      if (currentStage !== 1) switchStage(1);
      showToast(`🎲 NOUVELLE PLANÈTE NASA DÉTECTÉE : ${{mp.name}} (${{mp.dist}}) !`);
    }}
  }};

  const STITCH_I18N = {{
    fr: {{
      s1: "ÉTAPE 1/4 // PARCOUREZ LA GALAXIE (ZQSD / FLÈCHES / SOURIS / MANETTE), CIBLEZ UNE PLANÈTE PUIS [ESPACE / BOUTON A] POUR ENVOYER VOS ROBOTS",
      s2: "ÉTAPE 2/4 // PILOTEZ VOS ROBOTS DE SURVEILLANCE DANS L'ANNEAU VERT LAGRANGE EN ÉVITANT LES ASTÉROÏDES KÉPLÉRIENS (0% ➔ 100%)",
      s3: "ÉTAPE 3/4 // ATTERRISSAGE ATMOSPHÉRIQUE OOLITE // ALIGNEZ LE VAISSEAU (←/→) ET FREINEZ [ESPACE / Z / BOUTON A] SOUS 95 m/s JUSQU'À 0 m",
      s4: "ÉTAPE 4/4 // ANALYSE PLANÉTAIRE JWST NIRSPEC•MIRI // VÉRIFIEZ LES BIOSIGNATURES ET VOTEZ [APPROUVER / REJETER]",
      s5: "ÉTAPE 5 // ARCHITECTURE DUAL-ENGINE // FAT CLIENT STEAM (UE5/UNITY + RTX 2070 BOINC) vs. WEBGPU ZERO-INSTALL (/webgpu)"
    }},
    es: {{
      s1: "ETAPA 1/4 // EXPLORA LA GALAXIA (WASD / FLECHAS / RATÓN / MANDO), ELIGE UN PLANETA Y PULSA [ESPACIO / BOTÓN A] PARA ENVIAR ROBOTS",
      s2: "ETAPA 2/4 // PILOTA LOS ROBOTS BEE-MOE AL ANILLO VERDE LAGRANGE ESQUIVANDO ASTEROIDES KEPLERIANOS (0% ➔ 100%)",
      s3: "ETAPA 3/4 // ATERRIZAJE ATMOSFÉRICO OOLITE // ALINEA LA NAVE Y FRENA [ESPACIO / Z / BOTÓN A] BAJO 95 m/s HASTA 0 m",
      s4: "ETAPA 4/4 // ANÁLISIS PLANETARIO JWST NIRSPEC•MIRI // VERIFICA BIOFIRMAS Y VOTA [APROBAR / RECHAZAR]",
      s5: "ETAPA 5 // ARQUITECTURA DUAL-ENGINE // CLIENTE NATIVO STEAM (RTX 2070 BOINC) vs. WEBGPU SIN INSTALACIÓN (/webgpu)"
    }},
    de: {{
      s1: "STUFE 1/4 // GALAXIE ERKUNDEN (WASD / MAUS / GAMEPAD), PLANET WÄHLEN & [LEERTASTE / TASTE A] FÜR ROBOTER-START DRÜCKEN",
      s2: "STUFE 2/4 // STEUERE DIE ÜBERWACHUNGSROBOTER IN DEN GRÜNEN LAGRANGE-RING & WEICHE ASTEROIDEN AUS (0% ➔ 100%)",
      s3: "STUFE 3/4 // ATMOSPHÄREN-LANDUNG // SCHIFF AUSRICHTEN & BREMSRAKETEN [LEERTASTE / Z / TASTE A] UNTER 95 m/s ZÜNDEN",
      s4: "STUFE 4/4 // JWST PLANETEN-ANALYSE // BIOSIGNATUREN PRÜFEN & ABSTIMMEN [BESTÄTIGEN / ABLEHNEN]",
      s5: "STUFE 5 // DUAL-ENGINE ARCHITEKTUR // STEAM FAT CLIENT (RTX 2070 BOINC) vs. ZERO-INSTALL WEBGPU (/webgpu)"
    }},
    en: {{
      s1: "STAGE 1/4 // EXPLORE GALAXY (WASD / ARROWS / MOUSE / GAMEPAD), PICK A PLANET & PRESS [SPACE / BUTTON A] TO LAUNCH ROBOTS",
      s2: "STAGE 2/4 // PILOT SURVEILLANCE ROBOTS INTO THE GREEN LAGRANGE RING WHILE DODGING KEPLERIAN ASTEROIDS (0% ➔ 100%)",
      s3: "STAGE 3/4 // ATMOSPHERIC LANDING // STEER IN THE GLIDE TUNNEL & FIRE RETROS [SPACE / Z / BUTTON A] BELOW 95 m/s TO 0 m",
      s4: "STAGE 4/4 // JWST PLANETARY ANALYSIS // INSPECT REAL ASTROPHYSICS DATA & VOTE [APPROVE / REJECT]",
      s5: "STAGE 5 // DUAL-ENGINE ARCHITECTURE // STEAM FAT CLIENT (UE5/UNITY + RTX 2070 BOINC) vs. WEBGPU ZERO-INSTALL (/webgpu)"
    }},
    zh: {{
      s1: "第 1/4 阶段 // 探索银河系（键盘 / 鼠标 / 手柄），选择行星并按 [空格键 / A键] 发射监测机器人",
      s2: "第 2/4 阶段 // 驾驶监测机器人进入绿色拉格朗日环，躲避开普勒小行星带（同步率 0% ➔ 100%）",
      s3: "第 3/4 阶段 // 行星大气层降落 // 对准绿色跑道并按 [空格键 / Z / A键] 减速至 95 米/秒以下安全着陆",
      s4: "第 4/4 阶段 // 韦伯光谱与天体物理分析 // 检查真实数据并投票 [批准行星 / 排除假阳性]",
      s5: "第 5 阶段 // 双引擎架构 // Steam 原生客户端 (RTX 2070 BOINC) 与免安装 WebGPU 浏览器端 (/webgpu)"
    }}
  }};

  window.setStitchLanguage = function(lang) {{
    currentLang = STITCH_I18N[lang] ? lang : 'fr';
    ['fr', 'es', 'de', 'en', 'zh'].forEach(l => {{
      const b = document.getElementById('lang-' + l);
      if (b) {{
        b.className = (l === currentLang)
          ? 'px-1.5 py-0.5 bg-tacticalCyan text-black font-bold rounded cursor-pointer'
          : 'px-1.5 py-0.5 text-cyan-300 hover:text-white rounded cursor-pointer';
      }}
    }});
    const d = STITCH_I18N[currentLang];
    stageDescriptions[1] = d.s1;
    stageDescriptions[2] = d.s2;
    stageDescriptions[3] = d.s3;
    stageDescriptions[4] = d.s4;
    stageDescriptions[5] = d.s5;
    const instr = document.getElementById('instruction-text');
    if (instr) instr.textContent = stageDescriptions[currentStage] || d.s1;
    showToast('LANGUE ACTIVE : ' + currentLang.toUpperCase());
  }};

  window.toggleControlsModal = function() {{
    const m = document.getElementById('controls-modal');
    if (m) m.classList.toggle('hidden');
  }};

  function updateCinematicBannerUI() {{
    for (let i = 1; i <= 4; i++) {{
      const btn = document.getElementById('cine-step-' + i);
      if (!btn) continue;
      if (i === currentStage) {{
        btn.className = 'px-2 py-0.5 rounded font-extrabold bg-tacticalCyan text-black border border-white shadow-cyanGlow cursor-pointer transition';
      }} else if (i < currentStage) {{
        btn.className = 'px-2 py-0.5 rounded font-bold bg-emerald-950/90 text-tacticalGreen border border-emerald-400 cursor-pointer transition';
      }} else {{
        btn.className = 'px-2 py-0.5 rounded font-bold bg-cyan-950/90 text-cyan-200 border border-cyan-600 hover:bg-cyan-800 cursor-pointer transition';
      }}
    }}
    const mainBtn = document.getElementById('btn-main-stage-action');
    const pName = (PLANET_DATA[currentPlanetKey] && PLANET_DATA[currentPlanetKey].name) || 'LHS 1140 b';
    if (mainBtn) {{
      if (currentStage === 1) {{
        mainBtn.textContent = `🚀 ENVOYER ROBOTS SUR ${{pName}} [ESPACE / BOUTON A]`;
      }} else if (currentStage === 2) {{
        mainBtn.textContent = `🛬 LANCER L'ATTERRISSAGE SUR ${{pName}} [ENTRÉE / BOUTON A]`;
      }} else if (currentStage === 3) {{
        mainBtn.textContent = `🔥 FREINER [ESPACE] OU POSER [ENTRÉE / BOUTON A]`;
      }} else if (currentStage === 4) {{
        mainBtn.textContent = `✅ VALIDER L'ANALYSE DE ${{pName}} [ESPACE / BOUTON A]`;
      }} else {{
        mainBtn.textContent = `🌌 RETOUR GALAXIE [1]`;
      }}
    }}
    const diffBadge = document.getElementById('difficulty-badge');
    if (diffBadge) {{
      const pPhys = PLANET_PHYSICS_DB[currentPlanetKey] || PLANET_PHYSICS_DB.lhs1140b;
      const belts = Math.min(4, (pPhys.asteroidBelts || 1) + Math.floor((difficultyLevel - 1) / 2));
      diffBadge.textContent = `🔥 DIFFICULTÉ NIV. ${{difficultyLevel}}/10 (${{belts}} CEINTURES • VENT ${{(1 + (difficultyLevel - 1) * 0.18).toFixed(1)}}x)`;
    }}
  }}

  // Stage 2 & Stage 3 Interactive Game State
  let shipPos = {{ x: 225, y: 96, targetX: null, targetY: null }};
  let probeLaunchInProgress = false;
  let orbitSyncProgress = 12;
  let orbitRobotPos = {{ x: 95, y: 110 }};
  let orbitPlanetAngle = 0.4;
  let orbitAsteroids = [];
  let orbitStars = [];
  let orbitCmeRadius = 0;
  let orbitCmeActive = false;
  let orbitShieldFlash = 0;
  let stage2TransitionDone = false;

  let landerX = 0.0; // -1.0 (left) to +1.0 (right)
  let landerVelX = 0.0;
  let windCurrentMs = 0.0;
  let windPhase = 0.0;
  let retroFiringFrames = 0;
  let stage3TouchdownDone = false;

  function initStage2OrbitalMinigame() {{
    orbitSyncProgress = 12;
    orbitRobotPos = {{ x: 80, y: 110 }};
    orbitPlanetAngle = 0.5;
    stage2TransitionDone = false;
    orbitCmeRadius = 0;
    orbitCmeActive = false;
    const pPhys = PLANET_PHYSICS_DB[currentPlanetKey] || PLANET_PHYSICS_DB.lhs1140b;
    const beltCount = Math.min(4, (pPhys.asteroidBelts || 1) + Math.floor((difficultyLevel - 1) / 2));
    const count = 8 + beltCount * 5 + difficultyLevel * 2;
    orbitAsteroids = [];
    for (let i = 0; i < count; i++) {{
      const beltIdx = i % beltCount;
      const rx = 70 + beltIdx * 34 + (Math.random() * 14 - 7);
      const ry = 44 + beltIdx * 21 + (Math.random() * 10 - 5);
      const dir = (beltIdx % 2 === 0) ? 1 : -1;
      const speed = dir * (0.012 + (0.9 / Math.sqrt(rx)) * 0.14 * (1 + (difficultyLevel - 1) * 0.14));
      orbitAsteroids.push({{
        angle: (i / count) * Math.PI * 2 + Math.random() * 0.5,
        rx: rx,
        ry: ry,
        speed: speed,
        size: 3.2 + (i % 3) * 1.4
      }});
    }}
    orbitStars = [
      {{ angle: 1.2, rx: 125, ry: 76, collected: false }},
      {{ angle: 3.5, rx: 155, ry: 90, collected: false }},
      {{ angle: 5.1, rx: 98, ry: 60, collected: false }}
    ];
    const explainEl = document.getElementById('orbit-astro-explain');
    if (explainEl) explainEl.innerHTML = pPhys.asteroidPhysicsFr;
    const beltEl = document.getElementById('s2-belt-count');
    if (beltEl) beltEl.textContent = beltCount + ' CEINTURES';
    const velEl = document.getElementById('s2-kepler-vel');
    if (velEl) velEl.textContent = pPhys.keplerVelKms + ' km/s';
    const diffTag = document.getElementById('s2-difficulty-tag');
    if (diffTag) diffTag.textContent = `NIV. ${{difficultyLevel}} (${{beltCount}} CEINTURES)`;
  }}

  function initStage3LandingMinigame() {{
    const data = PLANET_DATA[currentPlanetKey] || PLANET_DATA.lhs1140b;
    const pPhys = PLANET_PHYSICS_DB[currentPlanetKey] || PLANET_PHYSICS_DB.lhs1140b;
    ooliteAltitude = data.altitudeInit || 3420;
    ooliteSpeed = 76.0 + Math.min(18, (difficultyLevel - 1) * 2.2);
    landerX = (Math.random() - 0.5) * 0.35;
    landerVelX = 0.0;
    windPhase = Math.random() * Math.PI * 2;
    stage3TouchdownDone = false;
    retroFiringFrames = 0;
    const gEl = document.getElementById('oolite-gravity-val');
    if (gEl) gEl.textContent = `${{pPhys.gravityG.toFixed(2)}} g (${{(pPhys.gravityG * 9.81).toFixed(1)}} m/s²)`;
    const scEl = document.getElementById('oolite-atmo-scale-tag');
    if (scEl) scEl.textContent = `H = ${{pPhys.scaleHeightKm}} km • P = ${{pPhys.pressureBar}} bar`;
    const dtEl = document.getElementById('oolite-atmo-detail-text');
    if (dtEl) dtEl.textContent = pPhys.atmoPhysicsFr;
  }}

  // Override switchStage to support all 5 stages and initialize Stage 2 / Stage 3 interactive mini-games
  const _origSwitchStage = switchStage;
  switchStage = function(stageNum) {{
    const sNum = parseInt(stageNum);
    if (sNum === 5) {{
      currentStage = 5;
      stageButtons.forEach(btn => {{
        const s = parseInt(btn.getAttribute('data-stage'));
        btn.className = (s === 5)
          ? 'stage-btn bg-tacticalCyan text-black font-bold px-3 py-1 border border-cyan-200 shadow-cyanGlow rounded flex items-center space-x-1 cursor-pointer transition active:scale-95'
          : 'stage-btn bg-cyan-950/70 hover:bg-cyan-900/80 text-cyan-300 px-2.5 py-1 border border-cyan-800 rounded transition cursor-pointer flex items-center space-x-1';
      }});
      for (let i = 1; i <= 5; i++) {{
        if (stagePanels[i]) {{
          if (i === 5) stagePanels[i].classList.remove('hidden');
          else stagePanels[i].classList.add('hidden');
        }}
      }}
      document.getElementById('instruction-text').textContent = stageDescriptions[5];
      document.getElementById('mission-progress-label').textContent = stageProgressLabels[5];
    }} else {{
      if (stagePanels[5]) stagePanels[5].classList.add('hidden');
      _origSwitchStage(sNum);
      if (sNum === 2) initStage2OrbitalMinigame();
      if (sNum === 3) initStage3LandingMinigame();
    }}
    updateCinematicBannerUI();
  }};

  // Synchronize all Stitch Screen panels on selectPlanet
  let chapter1IntroDone = false;
  const PLANET_ANALYSIS_AND_LIFE = PLANET_PHYSICS_DB;
  const _origSelectPlanet = selectPlanet;
  selectPlanet = function(planetKey, fromProximity = false) {{
    if (!PLANET_DATA[planetKey]) return;
    _origSelectPlanet(planetKey);
    const data = PLANET_DATA[planetKey];
    const pPhys = PLANET_PHYSICS_DB[planetKey] || PLANET_PHYSICS_DB.lhs1140b;

    // After initial Chapter 1 explanation, display & speak ONLY planet analysis, specific planet info & how life could live
    const vtEl = document.getElementById('voice-transcript');
    const analysisTxt = pPhys.lifeAnalysisFr || data.transcript;
    if (vtEl) {{
      if (!chapter1IntroDone) {{
        chapter1IntroDone = true;
        vtEl.innerHTML = `« <strong>Chapitre 1 : Passerelle.</strong> Parcourez la galaxie (Clavier/Souris/Manette) et appuyez sur [ESPACE / BOUTON A] pour envoyer vos robots. ${{analysisTxt}} »`;
      }} else {{
        vtEl.innerHTML = `« <strong>${{data.name}}</strong> — ${{analysisTxt}} »`;
      }}
    }}

    // Update Stage 1 Bottom Launch Console
    const s1TargetName = document.getElementById('s1-hud-target-name');
    if (s1TargetName) s1TargetName.textContent = data.name;
    const s1PhysTag = document.getElementById('s1-hud-physics-tag');
    if (s1PhysTag) s1PhysTag.textContent = `g = ${{pPhys.gravityG}} g • P = ${{pPhys.pressureBar}} bar • ${{data.temp}}`;

    // Move USS Hivemind ship marker toward target star if selected by click/button (not when already flying by proximity)
    const shipMarker = document.getElementById('hivemind-marker');
    if (shipMarker && !fromProximity) {{
      shipPos.x = Math.max(35, Math.min(720, data.nodeX - 65));
      shipPos.y = Math.max(35, Math.min(340, data.nodeY - 10));
      shipPos.targetX = null;
      shipPos.targetY = null;
      shipMarker.style.left = shipPos.x + 'px';
      shipMarker.style.top = shipPos.y + 'px';
    }}
    updateStage1SensorCone();

    // Update Stage 1 Sub-Spatial HUD Box (Screen 11)
    const s1Dist = document.getElementById('s1-sub-dist');
    if (s1Dist) s1Dist.textContent = data.dist.split('(')[0].trim();
    const s1Az = document.getElementById('s1-sub-az');
    if (s1Az) s1Az.textContent = `AZIMUTH // ${{data.ra}}`;
    const s1Dec = document.getElementById('s1-sub-dec');
    if (s1Dec) s1Dec.textContent = `DECL // ${{data.dec}}`;
    const kMatch = (data.rvInstrument || '').match(/K\\s*=\\s*([0-9.]+\\s*m\\/s)/i);
    const eMatch = (data.rvInstrument || '').match(/e\\s*=\\s*([0-9.]+)/i);
    const s1Vrad = document.getElementById('s1-sub-vrad');
    if (s1Vrad && kMatch) s1Vrad.textContent = kMatch[1];

    // Update Stage 2 Target Dossier & Avatar (Screen 00)
    const s2Title = document.getElementById('s2-dossier-title');
    if (s2Title) s2Title.textContent = `TARGET DOSSIER // ${{data.name.toUpperCase()}}`;
    const s2Rad = document.getElementById('s2-dossier-rad');
    if (s2Rad) s2Rad.textContent = data.transitRadius.split('(')[0].trim();
    const s2Dens = document.getElementById('s2-dossier-dens');
    if (s2Dens) s2Dens.textContent = data.transitDensity.split('(')[0].trim();
    const s2Temp = document.getElementById('s2-dossier-temp');
    if (s2Temp) s2Temp.textContent = data.temp;
    const s2Img = document.getElementById('stage2-planet-img');
    if (s2Img && data.img) s2Img.src = data.img;
    const s2Ecc = document.getElementById('s2-ecc-final');
    if (s2Ecc && eMatch) s2Ecc.textContent = 'e = ' + eMatch[1];

    // Update Stage 3 Oolite Planetfall HUD (Screen 10 & 06)
    const cp = document.getElementById('oolite-comms-planet');
    if (cp) cp.textContent = data.name;
    const sb = document.getElementById('oolite-sys-main-body');
    if (sb) sb.textContent = `1.00 UA • ${{data.name}} [MAIN TARGET]`;
    const vt = document.getElementById('oolite-voyage-target');
    if (vt) vt.textContent = `Cible: ${{data.name}}`;
    const vd = document.getElementById('oolite-voyage-dist');
    if (vd) vd.textContent = `${{(data.altitudeInit || 3420).toLocaleString()}} m`;
    const ga = document.getElementById('gauge-alt');
    if (ga) ga.textContent = `${{((data.altitudeInit || 3420) / 1000).toFixed(1)}}k`;

    // Update Stage 4 Beryllium 6.5M KPIs, Thermal Gradient & Astrobiology Summary (Screen 01)
    const s4K = document.getElementById('s4-kpi-k');
    if (s4K && kMatch) s4K.textContent = kMatch[1];
    const s4E = document.getElementById('s4-kpi-e');
    if (s4E && eMatch) s4E.textContent = eMatch[1];
    const s4Depth = document.getElementById('s4-kpi-depth');
    if (s4Depth) s4Depth.textContent = data.transitDepth.replace('DEPTH:', '').split('(')[0].trim();
    const s4Teq = document.getElementById('s4-teq-label');
    if (s4Teq) s4Teq.textContent = `▲ ${{data.temp}} TARGET_LOCK`;
    const s4Life = document.getElementById('s4-life-text');
    if (s4Life) s4Life.textContent = analysisTxt;

    updateCinematicBannerUI();
  }};

  function updateStage1SensorCone() {{
    const data = PLANET_DATA[currentPlanetKey];
    if (!data) return;
    const targetX = data.nodeX + 32;
    const targetY = data.nodeY + 32;
    const cone = document.getElementById('svg-sensor-cone');
    const lockLine = document.getElementById('svg-lock-line');
    if (cone && lockLine) {{
      cone.setAttribute('points', `${{shipPos.x + 28}},${{shipPos.y + 18}} ${{targetX - 18}},${{targetY - 14}} ${{targetX + 26}},${{targetY + 30}} ${{shipPos.x + 28}},${{shipPos.y + 18}}`);
      lockLine.setAttribute('x1', String(shipPos.x + 28));
      lockLine.setAttribute('y1', String(shipPos.y + 18));
      lockLine.setAttribute('x2', String(targetX));
      lockLine.setAttribute('y2', String(targetY));
    }}
  }}

  // === STAGE 1 -> STAGE 2 CINEMATIC: LAUNCH BEE-MOE SURVEILLANCE ROBOTS ===
  window.launchBeeProbesCinematic = function() {{
    if (probeLaunchInProgress) return;
    if (currentStage !== 1) {{
      switchStage(2);
      return;
    }}
    probeLaunchInProgress = true;
    const data = PLANET_DATA[currentPlanetKey] || PLANET_DATA.lhs1140b;
    playSciFiTone(380, 920, 380, 'sawtooth', 0.09);
    scoreHmt += 150;
    const scoreEl = document.getElementById('score-hmt');
    if (scoreEl) scoreEl.textContent = scoreHmt.toLocaleString('en-US');
    showToast(`🚀 ENVOI DE 4 ROBOTS DE SURVEILLANCE BEE-MOE VERS ${{data.name.toUpperCase()}} (+150 HMT)...`);

    const layer = document.getElementById('probe-launch-layer');
    if (layer) {{
      layer.innerHTML = '';
      const startX = shipPos.x + 25;
      const startY = shipPos.y + 16;
      const endX = data.nodeX + 28;
      const endY = data.nodeY + 28;
      for (let i = 0; i < 4; i++) {{
        const p = document.createElement('div');
        p.className = 'absolute flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-tacticalCyan text-black font-mono font-extrabold text-[8.5px] shadow-cyanGlow border border-white transition-all duration-700';
        p.style.left = (startX + (i - 1.5) * 12) + 'px';
        p.style.top = (startY + (i % 2) * 12) + 'px';
        p.textContent = `🤖 BEE-${{i + 1}}`;
        layer.appendChild(p);
        setTimeout(() => {{
          p.style.left = (endX + (i - 1.5) * 10) + 'px';
          p.style.top = (endY + (i % 2) * 10) + 'px';
          p.style.transform = 'scale(1.15)';
        }}, 60 + i * 90);
      }}
    }}

    // Trigger backend Bee-MoE scan asynchronously
    const pPhys = PLANET_PHYSICS_DB[currentPlanetKey] || {{}};
    const apiTarget = pPhys.realId || currentPlanetKey;
    fetch('/api/astro/bee-moe-scan', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ target_id: apiTarget }})
    }}).catch(() => {{}});

    setTimeout(() => {{
      if (layer) layer.innerHTML = '';
      probeLaunchInProgress = false;
      switchStage(2);
      showToast(`🤖 ROBOTS EN ORBITE AUTOUR DE ${{data.name}} ! Pilotez-les dans l'Anneau Vert !`);
    }}, 950);
  }};

  // === STAGE 2 -> STAGE 3 TRANSITION: ORBITAL SYNC COMPLETE -> START ATMOSPHERIC LANDING ===
  window.completeStage2AndStartLanding = function() {{
    if (stage2TransitionDone) return;
    stage2TransitionDone = true;
    const data = PLANET_DATA[currentPlanetKey] || PLANET_DATA.lhs1140b;
    playSciFiTone(520, 1040, 360, 'triangle', 0.1);
    scoreHmt += 500;
    const scoreEl = document.getElementById('score-hmt');
    if (scoreEl) scoreEl.textContent = scoreHmt.toLocaleString('en-US');
    showToast(`✅ SYNCHRONISATION ORBITALE 100% (+500 HMT) ! DESCENTE ATMOSPHÉRIQUE SUR ${{data.name.toUpperCase()}}...`);
    switchStage(3);
  }};

  // === STAGE 3 -> STAGE 4 TRANSITION: ASSISTED OR MANUAL TOUCHDOWN AT 0 m ===
  window.triggerAssistedTouchdown = function() {{
    if (currentStage !== 3) switchStage(3);
    ooliteAltitude = 0;
    ooliteSpeed = 42.0;
    landerX = 0.0;
    completeStage3TouchdownSuccess();
  }};

  function completeStage3TouchdownSuccess() {{
    if (stage3TouchdownDone) return;
    stage3TouchdownDone = true;
    const data = PLANET_DATA[currentPlanetKey] || PLANET_DATA.lhs1140b;
    playSciFiTone(440, 880, 220, 'sine', 0.1);
    setTimeout(() => playSciFiTone(660, 1320, 350, 'triangle', 0.12), 180);
    scoreHmt += 1000;
    const scoreEl = document.getElementById('score-hmt');
    if (scoreEl) scoreEl.textContent = scoreHmt.toLocaleString('en-US');

    const alertEl = document.getElementById('terrain-alert');
    if (alertEl) {{
      alertEl.textContent = `🥇 ATTERRISSAGE RÉUSSI SUR ${{data.name.toUpperCase()}} (+1 000 HMT) !`;
      alertEl.className = 'font-display font-black tracking-widest text-tacticalGreen text-[10px]';
    }}
    showToast(`🥇 ATTERRISSAGE RÉUSSI SUR ${{data.name.toUpperCase()}} (+1 000 HMT) ! Ouverture de l'analyse JWST...`);
    setTimeout(() => {{
      if (currentStage === 3) {{
        switchStage(4);
        showToast(`🔬 ÉTAPE 4/4 : Analysez ${{data.name}} et cliquez sur APPROUVER ou REJETER !`);
      }}
    }}, 1300);
  }}

  window.triggerPrimaryStageAction = function() {{
    if (currentStage === 1) launchBeeProbesCinematic();
    else if (currentStage === 2) completeStage2AndStartLanding();
    else if (currentStage === 3) triggerAssistedTouchdown();
    else if (currentStage === 4) {{
      const pPhys = PLANET_PHYSICS_DB[currentPlanetKey] || {{}};
      submitStitchDiscovery(!pPhys.isFalsePositive);
    }} else {{
      switchStage(1);
    }}
  }};

  window.updateRetroSlider = function(val) {{
    retroThrustPct = parseInt(val) || 75;
    const tr = document.getElementById('thrustReadout');
    if (tr) tr.textContent = retroThrustPct + '%';
  }};

  const _origFireRetroBurn = fireRetroBurn;
  fireRetroBurn = function() {{
    retroFiringFrames = 18;
    const extraBrake = (retroThrustPct - 40) * 0.22;
    ooliteSpeed = Math.max(18.0, ooliteSpeed - Math.max(8.0, extraBrake));
    playSciFiTone(180, 95, 140, 'sawtooth', 0.06);
    _origFireRetroBurn();
    updateStage3GaugesDOM();
  }};

  function updateStage3GaugesDOM() {{
    const gs = document.getElementById('gauge-speed');
    if (gs) gs.textContent = Math.round(ooliteSpeed);
    const ga = document.getElementById('gauge-alt');
    if (ga) ga.textContent = (ooliteAltitude >= 1000) ? `${{(ooliteAltitude / 1000).toFixed(1)}}k` : `${{Math.round(ooliteAltitude)}}m`;
    const vd = document.getElementById('oolite-voyage-dist');
    if (vd) vd.textContent = `${{Math.round(ooliteAltitude).toLocaleString()}} m`;
    const atTape = document.getElementById('altitude-tape');
    if (atTape) atTape.textContent = `${{Math.round(ooliteAltitude).toLocaleString('en-US')}} m`;
    const spTape = document.getElementById('speed-tape');
    if (spTape) spTape.textContent = `${{ooliteSpeed.toFixed(1)}} m/s`;
  }}

  window.sampleAtmosphereStage2 = function() {{
    orbitSyncProgress = Math.min(100, orbitSyncProgress + 25);
    scoreHmt += 150;
    const scoreEl = document.getElementById('score-hmt');
    if (scoreEl) scoreEl.textContent = scoreHmt.toLocaleString('en-US');
    playSciFiTone(620, 980, 180, 'sine');
    showToast(`ÉCHANTILLON ATMOSPHÉRIQUE H₂O/DMS COLLECTÉ SUR ${{PLANET_DATA[currentPlanetKey].name}} (+150 HMT)`);
  }};

  // Enhance Stage 2 buttons (+ SONDE MOE and SYNC L1-L2) so clicking them visibly boosts orbital sync immediately
  function syncOrbitDomReadout() {{
    const pct = Math.round(orbitSyncProgress);
    const pctEl = document.getElementById('orbit-sync-pct');
    if (pctEl) pctEl.textContent = pct + '%';
    const barEl = document.getElementById('orbit-sync-bar');
    if (barEl) barEl.style.width = pct + '%';
    const satEl = document.getElementById('sat-count');
    if (satEl) satEl.textContent = `${{Math.round((pct / 100) * 128)}} / 128`;
  }}
  const btnDeployProbeS2 = document.getElementById('btn-deploy-probe');
  if (btnDeployProbeS2) {{
    btnDeployProbeS2.addEventListener('click', () => {{
      orbitSyncProgress = Math.min(100, orbitSyncProgress + 28);
      syncOrbitDomReadout();
      playSciFiTone(480, 840, 160, 'triangle');
    }});
  }}
  const btnSyncPhaseS2 = document.getElementById('btn-sync-phase');
  if (btnSyncPhaseS2) {{
    btnSyncPhaseS2.addEventListener('click', () => {{
      orbitSyncProgress = Math.min(100, orbitSyncProgress + 35);
      syncOrbitDomReadout();
      playSciFiTone(540, 960, 180, 'sine');
    }});
  }}

  // Wire Stage 1 right-hand drawer buttons to launch robots / jump to spectroscopy
  const btnWarpS1 = document.getElementById('btn-engage-warp');
  if (btnWarpS1) {{
    btnWarpS1.textContent = '🚀 ROBOTS [ESPACE]';
    btnWarpS1.addEventListener('click', () => launchBeeProbesCinematic());
  }}
  const btnProbeS1 = document.getElementById('btn-probe-lois');
  if (btnProbeS1) {{
    btnProbeS1.textContent = '🤖 ORBITE BEE [2]';
    btnProbeS1.addEventListener('click', () => launchBeeProbesCinematic());
  }}
  const btnNirspecS1 = document.getElementById('btn-nirspec');
  if (btnNirspecS1) {{
    btnNirspecS1.addEventListener('click', () => switchStage(4));
  }}

  window.highlightMolecule = function(mol, wave, pct) {{
    const tag = document.getElementById('dms-detection-tag');
    if (tag) tag.textContent = `● RAIE SPECTRALE VERROUILLÉE: ${{mol}} @ ${{wave}} [CONFIDENCE ${{pct}}]`;
    playSciFiTone(700, 1100, 120, 'sine');
    showToast(`SPECTRE JWST NIRSpec•MIRI : ${{mol}} (${{wave}}) — ${{pct}}`);
  }};

  const clearanceBtn = document.getElementById('btn-clearance-toggle');
  if (clearanceBtn) {{
    clearanceBtn.addEventListener('click', () => {{
      showToast('AUTORISATION TOUR CHASMA CONFIRMÉE (PAD VERROUILLÉ)');
    }});
  }}

  // Live NASA TAP ADQL query integration on Stage 4 button
  document.getElementById('btn-run-adql').addEventListener('click', async () => {{
    try {{
      const r = await fetch('/api/astro/nasa-tap-sync');
      const d = await r.json();
      showToast(`NASA TAP ADQL SYNC: ${{d.row_count || 8}} SYSTÈMES VÉRIFIÉS`);
    }} catch (e) {{}}
  }});

  window.submitStitchDiscovery = async function(approved) {{
    const pPhys = PLANET_PHYSICS_DB[currentPlanetKey] || {{}};
    const data = PLANET_DATA[currentPlanetKey] || PLANET_DATA.lhs1140b;
    const isCorrect = (approved && !pPhys.isFalsePositive) || (!approved && pPhys.isFalsePositive);
    const bonus = isCorrect ? 800 : 300;
    scoreHmt += bonus;
    const scoreEl = document.getElementById('score-hmt');
    if (scoreEl) scoreEl.textContent = scoreHmt.toLocaleString('en-US');
    playSciFiTone(isCorrect ? 520 : 300, isCorrect ? 1040 : 220, 320, 'triangle', 0.1);

    // Stamp gold medal on bottom planet card & escalate difficulty level
    completedPlanetsSet.add(currentPlanetKey);
    completedPlanetsCount = completedPlanetsSet.size;
    difficultyLevel = Math.min(10, difficultyLevel + 1);
    updateCinematicBannerUI();

    const card = document.querySelector(`.planet-card[data-planet="${{currentPlanetKey}}"]`);
    if (card && !card.querySelector('.medal-stamp')) {{
      const stamp = document.createElement('div');
      stamp.className = 'medal-stamp absolute top-1 left-1 bg-amber-400 text-black font-extrabold text-[8px] px-1 rounded shadow';
      stamp.textContent = '🥇 VALIDÉ';
      card.appendChild(stamp);
    }}

    if (isCorrect) {{
      showToast(`🥇 BRAVO ! ${{data.name}} VALIDÉE (+${{bonus}} HMT) ! DIFFICULTÉ AUGMENTÉE ➔ NIV. ${{difficultyLevel}}/10 !`);
    }} else {{
      showToast(`⚠️ ANALYSE ENREGISTRÉE SUR ${{data.name}} (+${{bonus}} HMT) ! DIFFICULTÉ ➔ NIV. ${{difficultyLevel}}/10`);
    }}

    const apiTarget = pPhys.realId || currentPlanetKey;
    try {{
      await fetch('/api/astro/discover', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
          target_id: apiTarget,
          player_classification: approved ? 'PLANET' : 'FALSE_POSITIVE'
        }})
      }});
    }} catch (e) {{}}
    try {{
      await fetch('https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/api/v1/community/boinc-sync', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
          contributor_handle: 'Cmdr_Web_Hunter',
          gpu_model: 'Browser WebGL/WebGPU Citizen-Science',
          gpu_utilization_pct: 6.4,
          gpu_idle_seconds: 12.0,
          target_planet: data.name || apiTarget,
          project_id: 'BOINC-ASTRO-BLS-TRANSIT'
        }})
      }});
    }} catch (e) {{}}
  }};

  // Global Toast & Optional Voice Notification (visible across ALL 5 Stages)
  let _globalToastTimer = null;
  showToast = function(msg) {{
    const drawerToast = document.getElementById('drawer-toast');
    if (drawerToast) {{
      drawerToast.textContent = msg;
      drawerToast.classList.remove('hidden');
    }}
    const globalToast = document.getElementById('global-game-toast');
    if (globalToast) {{
      globalToast.textContent = msg;
      globalToast.classList.remove('hidden');
    }}
    if (_globalToastTimer) clearTimeout(_globalToastTimer);
    _globalToastTimer = setTimeout(() => {{
      if (drawerToast) drawerToast.classList.add('hidden');
      if (globalToast) globalToast.classList.add('hidden');
    }}, 3200);
  }};

  const planetOrder = Object.keys(PLANET_DATA);
  window.cyclePlanetSelection = function(dir) {{
    const allKeys = Object.keys(PLANET_DATA);
    let idx = allKeys.indexOf(currentPlanetKey);
    idx = (idx + (dir || 1) + allKeys.length) % allKeys.length;
    selectPlanet(allKeys[idx]);
  }};

  window.advanceToNextPlanetFromStage4 = function() {{
    const allKeys = Object.keys(PLANET_DATA);
    const idx = allKeys.indexOf(currentPlanetKey);
    const nextKey = allKeys[(idx + 1) % allKeys.length];
    selectPlanet(nextKey);
    switchStage(1);
    showToast(`🌌 NOUVEAU CAP : ${{PLANET_DATA[nextKey].name}} (DIFFICULTÉ NIV. ${{difficultyLevel}}/10) !`);
  }};

  // === INPUT ENGINE: KEYBOARD + MOUSE + HTML5 GAMEPAD API (60 FPS) ===
  const keysHeld = {{}};
  window.addEventListener('keydown', (e) => {{
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
    const k = e.key.toLowerCase();
    keysHeld[k] = true;
    keysHeld[e.code] = true;

    if (['1', '2', '3', '4', '5'].includes(e.key)) {{
      switchStage(parseInt(e.key, 10));
      return;
    }}
    if (k === 'h') {{ toggleControlsModal(); return; }}
    if (k === 'm') {{ selectRandomNasaMysteryPlanet(true); return; }}
    if (k === 'n' || e.key === 'Tab') {{
      e.preventDefault();
      cyclePlanetSelection(e.shiftKey ? -1 : 1);
      return;
    }}
    if (e.key === 'Escape') {{
      const m = document.getElementById('controls-modal');
      if (m && !m.classList.contains('hidden')) m.classList.add('hidden');
      return;
    }}
    if ((k === 'z' || k === 'w' || e.key === 'ArrowUp') && currentStage === 3) {{
      e.preventDefault();
      fireRetroBurn();
    }}

    // Stage-specific action shortcuts
    if (e.code === 'Space') {{
      e.preventDefault();
      if (currentStage === 1) launchBeeProbesCinematic();
      else if (currentStage === 2) {{
        orbitSyncProgress = Math.min(100, orbitSyncProgress + 22);
        playSciFiTone(520, 880, 120, 'sine');
      }} else if (currentStage === 3) {{
        fireRetroBurn();
      }} else if (currentStage === 4) {{
        submitStitchDiscovery(true);
      }}
    }} else if (e.code === 'Enter') {{
      e.preventDefault();
      if (currentStage === 1) launchBeeProbesCinematic();
      else if (currentStage === 2) completeStage2AndStartLanding();
      else if (currentStage === 3) triggerAssistedTouchdown();
      else if (currentStage === 4) advanceToNextPlanetFromStage4();
    }} else if (k === 'r' && currentStage === 4) {{
      submitStitchDiscovery(false);
    }}
  }});

  window.addEventListener('keyup', (e) => {{
    keysHeld[e.key.toLowerCase()] = false;
    keysHeld[e.code] = false;
  }});

  // Mouse interactions on Stage 1 Galaxy Map, Stage 2 Orbital Canvas, and Stage 3 Landing Canvas
  const stage1Viewport = document.getElementById('viewport-stage-1');
  if (stage1Viewport) {{
    stage1Viewport.addEventListener('click', (e) => {{
      if (e.target.closest('button') || e.target.closest('.planet-node') || e.target.closest('#s1-launch-hud')) return;
      const rect = stage1Viewport.getBoundingClientRect();
      shipPos.targetX = Math.max(30, Math.min(740, e.clientX - rect.left - 24));
      shipPos.targetY = Math.max(30, Math.min(350, e.clientY - rect.top - 16));
      playSciFiTone(340, 520, 90, 'sine', 0.04);
    }});
  }}

  // Bind click & double-click on all planet nodes (including the new mystery_nasa node)
  document.querySelectorAll('.planet-node').forEach(node => {{
    node.addEventListener('click', (e) => {{
      e.stopPropagation();
      const pk = node.getAttribute('data-planet');
      if (pk) selectPlanet(pk);
    }});
    node.addEventListener('dblclick', (e) => {{
      e.stopPropagation();
      const pk = node.getAttribute('data-planet');
      if (pk) {{
        selectPlanet(pk);
        launchBeeProbesCinematic();
      }}
    }});
  }});

  const orbitCanvas = document.getElementById('orbit-bee-canvas');
  if (orbitCanvas) {{
    const handleOrbitMouse = (e) => {{
      const rect = orbitCanvas.getBoundingClientRect();
      const scaleX = orbitCanvas.width / Math.max(1, rect.width);
      const scaleY = orbitCanvas.height / Math.max(1, rect.height);
      const mx = (e.clientX - rect.left) * scaleX;
      const my = (e.clientY - rect.top) * scaleY;
      orbitRobotPos.x += (mx - orbitRobotPos.x) * 0.22;
      orbitRobotPos.y += (my - orbitRobotPos.y) * 0.22;
    }};
    orbitCanvas.addEventListener('mousemove', (e) => {{
      if (e.buttons === 1 || currentStage === 2) handleOrbitMouse(e);
    }});
    orbitCanvas.addEventListener('click', handleOrbitMouse);
  }}

  const ooliteCanvas = document.getElementById('oolite-glide-canvas');
  if (ooliteCanvas) {{
    ooliteCanvas.addEventListener('mousemove', (e) => {{
      if (currentStage !== 3) return;
      const rect = ooliteCanvas.getBoundingClientRect();
      const normX = ((e.clientX - rect.left) / Math.max(1, rect.width) - 0.5) * 2.0;
      landerX += (Math.max(-0.95, Math.min(0.95, normX)) - landerX) * 0.18;
    }});
    ooliteCanvas.addEventListener('click', () => {{
      if (currentStage === 3) fireRetroBurn();
    }});
  }}

  // Gamepad polling state
  let prevGpButtons = {{}};
  window.addEventListener('gamepadconnected', (e) => {{
    const gpBadge = document.getElementById('gamepad-status-badge');
    if (gpBadge) {{
      gpBadge.textContent = '🟢 MANETTE CONNECTÉE :';
      gpBadge.className = 'text-tacticalGreen font-extrabold';
    }}
    showToast(`🎮 MANETTE DÉTECTÉE : ${{e.gamepad.id}}`);
  }});

  function pollGamepadAndInputs() {{
    let dx = 0;
    let dy = 0;
    let actionAPressed = false;
    let actionBPressed = false;
    let actionXPressed = false;
    let lbPressed = false;
    let rbPressed = false;
    let retroHeld = false;

    // Keyboard directional axes (ZQSD + WASD + Arrow keys)
    if (keysHeld['arrowleft'] || keysHeld['q'] || keysHeld['a']) dx -= 1;
    if (keysHeld['arrowright'] || keysHeld['d']) dx += 1;
    if (keysHeld['arrowup'] || keysHeld['z'] || keysHeld['w']) {{
      dy -= 1;
      retroHeld = true;
    }}
    if (keysHeld['arrowdown'] || keysHeld['s']) dy += 1;
    if (keysHeld[' '] || keysHeld['Space']) retroHeld = true;

    // Poll HTML5 Gamepad API
    const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
    for (let i = 0; i < gamepads.length; i++) {{
      const gp = gamepads[i];
      if (!gp || !gp.connected) continue;
      const gpBadge = document.getElementById('gamepad-status-badge');
      if (gpBadge && !gpBadge.textContent.includes('CONNECTÉE')) {{
        gpBadge.textContent = '🟢 MANETTE ACTIVE :';
        gpBadge.className = 'text-tacticalGreen font-extrabold';
      }}
      const ax0 = gp.axes[0] || 0;
      const ax1 = gp.axes[1] || 0;
      if (Math.abs(ax0) > 0.16) dx += ax0;
      if (Math.abs(ax1) > 0.16) dy += ax1;
      // D-pad buttons (12=Up, 13=Down, 14=Left, 15=Right)
      if (gp.buttons[14] && gp.buttons[14].pressed) dx -= 1;
      if (gp.buttons[15] && gp.buttons[15].pressed) dx += 1;
      if (gp.buttons[12] && gp.buttons[12].pressed) {{ dy -= 1; retroHeld = true; }}
      if (gp.buttons[13] && gp.buttons[13].pressed) dy += 1;

      const b0 = !!(gp.buttons[0] && gp.buttons[0].pressed); // A / Cross
      const b1 = !!(gp.buttons[1] && gp.buttons[1].pressed); // B / Circle
      const b2 = !!(gp.buttons[2] && gp.buttons[2].pressed); // X / Square
      const b4 = !!(gp.buttons[4] && gp.buttons[4].pressed); // LB / L1
      const b5 = !!(gp.buttons[5] && gp.buttons[5].pressed); // RB / R1

      if (b0) retroHeld = true;
      if (b0 && !prevGpButtons[0]) actionAPressed = true;
      if (b1 && !prevGpButtons[1]) actionBPressed = true;
      if (b2 && !prevGpButtons[2]) actionXPressed = true;
      if (b4 && !prevGpButtons[4]) lbPressed = true;
      if (b5 && !prevGpButtons[5]) rbPressed = true;

      prevGpButtons[0] = b0;
      prevGpButtons[1] = b1;
      prevGpButtons[2] = b2;
      prevGpButtons[4] = b4;
      prevGpButtons[5] = b5;
      break;
    }}

    if (lbPressed || rbPressed) {{
      const allKeys = Object.keys(PLANET_DATA);
      let idx = allKeys.indexOf(currentPlanetKey);
      idx = (idx + (rbPressed ? 1 : -1) + allKeys.length) % allKeys.length;
      selectPlanet(allKeys[idx]);
    }}
    if (actionAPressed) {{
      if (currentStage === 1) launchBeeProbesCinematic();
      else if (currentStage === 2) completeStage2AndStartLanding();
      else if (currentStage === 3) fireRetroBurn();
      else if (currentStage === 4) submitStitchDiscovery(true);
    }}
    if (actionXPressed || actionBPressed) {{
      if (currentStage === 1) selectRandomNasaMysteryPlanet(true);
      else if (currentStage === 3) triggerAssistedTouchdown();
      else if (currentStage === 4) submitStitchDiscovery(false);
    }}

    return {{ dx: Math.max(-1.2, Math.min(1.2, dx)), dy: Math.max(-1.2, Math.min(1.2, dy)), retroHeld }};
  }}

  // === 60 FPS MASTER GAME LOOP (STAGE 1 GALAXY FLIGHT, STAGE 2 ORBITAL ROBOTS, STAGE 3 OOLITE LANDING) ===
  function updateAndRenderStage1(input) {{
    const shipMarker = document.getElementById('hivemind-marker');
    if (!shipMarker) return;
    shipMarker.classList.remove('transition-all', 'duration-300');

    const speed = 4.2;
    if (Math.abs(input.dx) > 0.05 || Math.abs(input.dy) > 0.05) {{
      shipPos.targetX = null;
      shipPos.targetY = null;
      shipPos.x = Math.max(25, Math.min(740, shipPos.x + input.dx * speed));
      shipPos.y = Math.max(25, Math.min(350, shipPos.y + input.dy * speed));
    }} else if (shipPos.targetX !== null && shipPos.targetY !== null) {{
      const vx = shipPos.targetX - shipPos.x;
      const vy = shipPos.targetY - shipPos.y;
      const dist = Math.hypot(vx, vy);
      if (dist < 5) {{
        shipPos.targetX = null;
        shipPos.targetY = null;
      }} else {{
        shipPos.x += (vx / dist) * (speed * 1.15);
        shipPos.y += (vy / dist) * (speed * 1.15);
      }}
    }}

    shipMarker.style.left = Math.round(shipPos.x) + 'px';
    shipMarker.style.top = Math.round(shipPos.y) + 'px';
    updateStage1SensorCone();

    // Check proximity to any planet node on the Galaxy Map for automatic target lock
    if (Math.abs(input.dx) > 0.05 || Math.abs(input.dy) > 0.05 || shipPos.targetX !== null) {{
      let closestKey = null;
      let closestDist = 56;
      const allKeys = Object.keys(PLANET_DATA);
      for (const k of allKeys) {{
        const pd = PLANET_DATA[k];
        if (!pd) continue;
        const d = Math.hypot((shipPos.x + 25) - (pd.nodeX + 28), (shipPos.y + 18) - (pd.nodeY + 28));
        if (d < closestDist) {{
          closestDist = d;
          closestKey = k;
        }}
      }}
      if (closestKey && closestKey !== currentPlanetKey) {{
        playSciFiTone(580, 780, 90, 'sine', 0.05);
        selectPlanet(closestKey, true);
      }}
    }}
  }}

  function updateAndRenderStage2(input) {{
    const canvas = document.getElementById('orbit-bee-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    const cx = W * 0.48;
    const cy = H * 0.52;

    // 1. Move orbiting target planet along its Keplerian ellipse
    const planetRx = 128;
    const planetRy = 76;
    orbitPlanetAngle += 0.0085 * (1 + (difficultyLevel - 1) * 0.08);
    const px = cx + Math.cos(orbitPlanetAngle) * planetRx;
    const py = cy + Math.sin(orbitPlanetAngle) * planetRy;

    // 2. Steer player's Bee-MoE Robot Swarm
    const robotSpeed = 3.8;
    if (Math.abs(input.dx) > 0.05 || Math.abs(input.dy) > 0.05) {{
      orbitRobotPos.x = Math.max(18, Math.min(W - 18, orbitRobotPos.x + input.dx * robotSpeed));
      orbitRobotPos.y = Math.max(18, Math.min(H - 18, orbitRobotPos.y + input.dy * robotSpeed));
    }} else {{
      // Gentle magnetic assist toward Lagrange capture ring so the game feels responsive
      orbitRobotPos.x += (px - orbitRobotPos.x) * 0.012;
      orbitRobotPos.y += (py - orbitRobotPos.y) * 0.012;
    }}

    // 3. Check distance to Green Lagrange Capture Ring around planet
    const distToPlanet = Math.hypot(orbitRobotPos.x - px, orbitRobotPos.y - py);
    const inGreenRing = distToPlanet < 46;
    if (inGreenRing) {{
      orbitSyncProgress = Math.min(100, orbitSyncProgress + 0.32);
    }} else {{
      orbitSyncProgress = Math.min(100, orbitSyncProgress + 0.06);
    }}

    // 4. Surprise CME Solar Flare shockwave at higher difficulty
    if (!orbitCmeActive && Math.random() < 0.003 * difficultyLevel) {{
      orbitCmeActive = true;
      orbitCmeRadius = 16;
    }}
    if (orbitCmeActive) {{
      orbitCmeRadius += 2.2 + difficultyLevel * 0.25;
      const distFromStar = Math.hypot(orbitRobotPos.x - cx, orbitRobotPos.y - cy);
      if (Math.abs(distFromStar - orbitCmeRadius) < 8 && distToPlanet >= 42) {{
        orbitShieldFlash = 10;
        orbitSyncProgress = Math.max(5, orbitSyncProgress - 4);
      }}
      if (orbitCmeRadius > 220) orbitCmeActive = false;
    }}

    // 5. Render Background & Grid
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#020815';
    ctx.fillRect(0, 0, W, H);

    // Draw Keplerian Orbit Ellipse
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.28)';
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.ellipse(cx, cy, planetRx, planetRy, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw Central Host Star
    const starGrad = ctx.createRadialGradient(cx, cy, 2, cx, cy, 22);
    starGrad.addColorStop(0, '#ffffff');
    starGrad.addColorStop(0.4, '#ffb800');
    starGrad.addColorStop(1, 'rgba(255, 100, 0, 0)');
    ctx.fillStyle = starGrad;
    ctx.beginPath();
    ctx.arc(cx, cy, 22, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#ffb800';
    ctx.font = 'bold 9px monospace';
    ctx.fillText('★ ÉTOILE HÔTE', cx - 32, cy + 28);

    // Draw CME Shockwave if active
    if (orbitCmeActive) {{
      ctx.strokeStyle = 'rgba(255, 90, 54, 0.7)';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(cx, cy, orbitCmeRadius, 0, Math.PI * 2);
      ctx.stroke();
    }}

    // Draw Collectible Golden Science Stars
    for (const st of orbitStars) {{
      if (st.collected) continue;
      st.angle += 0.005;
      const sx = cx + Math.cos(st.angle) * st.rx;
      const sy = cy + Math.sin(st.angle) * st.ry;
      ctx.fillStyle = '#fde047';
      ctx.beginPath();
      ctx.arc(sx, sy, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#fef08a';
      ctx.font = 'bold 8px monospace';
      ctx.fillText('★+200', sx - 12, sy - 7);
      if (Math.hypot(orbitRobotPos.x - sx, orbitRobotPos.y - sy) < 16) {{
        st.collected = true;
        scoreHmt += 200;
        orbitSyncProgress = Math.min(100, orbitSyncProgress + 12);
        const scoreEl = document.getElementById('score-hmt');
        if (scoreEl) scoreEl.textContent = scoreHmt.toLocaleString('en-US');
        playSciFiTone(660, 1180, 160, 'triangle');
        showToast('⭐ ÉTOILE DE SCIENCE COLLECTÉE (+200 HMT & +12% SYNCHRO) !');
      }}
    }}

    // Draw & Collide Counter-Rotating Keplerian Asteroid Belts
    for (const ast of orbitAsteroids) {{
      ast.angle += ast.speed;
      const ax = cx + Math.cos(ast.angle) * ast.rx;
      const ay = cy + Math.sin(ast.angle) * ast.ry;
      ctx.fillStyle = '#f97316';
      ctx.beginPath();
      ctx.arc(ax, ay, ast.size, 0, Math.PI * 2);
      ctx.fill();

      // Check collision with player's Bee-MoE robots
      const dAst = Math.hypot(orbitRobotPos.x - ax, orbitRobotPos.y - ay);
      if (dAst < ast.size + 9) {{
        orbitShieldFlash = 12;
        orbitRobotPos.x = Math.max(20, Math.min(W - 20, orbitRobotPos.x + (orbitRobotPos.x - ax) * 1.4));
        orbitRobotPos.y = Math.max(20, Math.min(H - 20, orbitRobotPos.y + (orbitRobotPos.y - ay) * 1.4));
        orbitSyncProgress = Math.max(5, orbitSyncProgress - 3.5);
        playSciFiTone(160, 75, 120, 'sawtooth', 0.06);
      }}
    }}

    // Draw Target Exoplanet & Pulsing Green Lagrange Capture Ring
    ctx.strokeStyle = inGreenRing ? '#00ff88' : 'rgba(0, 255, 136, 0.75)';
    ctx.lineWidth = inGreenRing ? 3.0 : 2.0;
    ctx.beginPath();
    ctx.arc(px, py, 34, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = '#00f0ff';
    ctx.beginPath();
    ctx.arc(px, py, 11, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#00ff88';
    ctx.font = 'bold 8.5px monospace';
    ctx.fillText('ANNEAU VERT (CIBLE)', px - 44, py - 38);

    // Draw Player's Bee-MoE Surveillance Robot Swarm (`🛸 VOUS`)
    ctx.save();
    ctx.translate(orbitRobotPos.x, orbitRobotPos.y);
    ctx.strokeStyle = (orbitShieldFlash > 0) ? '#ff3366' : '#00f0ff';
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.arc(0, 0, 11, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = '#ffb800';
    ctx.beginPath();
    ctx.moveTo(0, -7);
    ctx.lineTo(7, 6);
    ctx.lineTo(-7, 6);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 9px monospace';
    ctx.fillText('🛸 VOUS', -18, -14);
    ctx.restore();

    if (orbitShieldFlash > 0) {{
      orbitShieldFlash--;
      const hzEl = document.getElementById('orbit-hazard-status');
      if (hzEl) {{
        hzEl.textContent = '⚠️ IMPACT ASTÉROÏDE KÉPLÉRIEN ! ÉVITEZ LES ROCHERS ORANGES !';
        hzEl.className = 'text-red-400 font-bold text-[8px]';
      }}
    }} else {{
      const hzEl = document.getElementById('orbit-hazard-status');
      if (hzEl) {{
        hzEl.textContent = inGreenRing ? '🟢 DANS L\\'ANNEAU LAGRANGE : SYNCHRO RAPIDE !' : '● PILOTEZ 🛸 VOUS VERS L\\'ANNEAU VERT';
        hzEl.className = inGreenRing ? 'text-tacticalGreen font-bold text-[8px]' : 'text-amber-300 text-[8px]';
      }}
    }}

    // Update Orbital Sync Progress Bar & Sat Count
    const pct = Math.round(orbitSyncProgress);
    const pctEl = document.getElementById('orbit-sync-pct');
    if (pctEl) pctEl.textContent = pct + '%';
    const barEl = document.getElementById('orbit-sync-bar');
    if (barEl) barEl.style.width = pct + '%';
    const satEl = document.getElementById('sat-count');
    if (satEl) satEl.textContent = `${{Math.round((pct / 100) * 128)}} / 128`;

    if (orbitSyncProgress >= 100 && !stage2TransitionDone) {{
      completeStage2AndStartLanding();
    }}
  }}

  function updateAndRenderStage3(input) {{
    const canvas = document.getElementById('oolite-glide-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    const pPhys = PLANET_PHYSICS_DB[currentPlanetKey] || PLANET_PHYSICS_DB.lhs1140b;

    if (!stage3TouchdownDone) {{
      // 1. Determine current Atmospheric Layer (4 layers based on altitude)
      let layerName = 'COUCHE 1 : COUCHE LIMITE & TURBULENCES';
      let windMult = 0.6;
      if (ooliteAltitude > 2400) {{
        layerName = 'COUCHE 4 : THERMOSPHÈRE & IONOSPHÈRE';
        windMult = 0.5;
      }} else if (ooliteAltitude > 1400) {{
        layerName = 'COUCHE 3 : STRATOSPHÈRE & JET-STREAM';
        windMult = 1.25;
      }} else if (ooliteAltitude > 500) {{
        layerName = 'COUCHE 2 : TROPOSPHÈRE CONVECTIVE';
        windMult = 0.95;
      }}
      const layerTag = document.getElementById('oolite-atmo-layer-tag');
      if (layerTag) layerTag.textContent = `[${{layerName}}]`;

      // 2. Real Atmospheric Crosswind Shear + Player Horizontal Steering
      windPhase += 0.025;
      const diffScale = 1 + (difficultyLevel - 1) * 0.16;
      windCurrentMs = Math.sin(windPhase) * (pPhys.windShearMs || 35) * windMult * diffScale;
      const windTag = document.getElementById('oolite-wind-tag');
      if (windTag) windTag.textContent = `💨 VENT: ${{windCurrentMs >= 0 ? '+' : ''}}${{Math.round(windCurrentMs)}} m/s`;

      const windPush = (windCurrentMs / 100.0) * 0.0055;
      const playerSteer = input.dx * 0.022;
      if (Math.abs(input.dx) > 0.05) {{
        landerX += playerSteer + windPush * 0.4;
      }} else {{
        // Gentle auto-stabilizer fighting part of the wind so the ship remains controllable
        landerX += windPush - landerX * 0.015;
      }}
      landerX = Math.max(-0.92, Math.min(0.92, landerX));

      // 3. Real Planetary Gravity vs Retro-Thruster Braking
      const gravityAccel = (pPhys.gravityG || 1.2) * 0.14;
      if (input.retroHeld || retroFiringFrames > 0) {{
        if (retroFiringFrames > 0) retroFiringFrames--;
        const brakeForce = 0.42 * (retroThrustPct / 75.0);
        ooliteSpeed = Math.max(24.0, ooliteSpeed - brakeForce);
      }} else {{
        ooliteSpeed = Math.min(138.0, ooliteSpeed + gravityAccel - 0.05);
      }}

      // 4. Descent Altitude Update
      ooliteAltitude = Math.max(0, ooliteAltitude - ooliteSpeed * 0.048);
      updateStage3GaugesDOM();

      // Update warning banner
      const alertEl = document.getElementById('terrain-alert');
      if (alertEl) {{
        if (ooliteSpeed > 95) {{
          alertEl.textContent = `⚠️ VITESSE ÉLEVÉE (${{Math.round(ooliteSpeed)}} m/s) ! FREINEZ [ESPACE / Z / A] !`;
          alertEl.className = 'font-display font-black tracking-widest text-red-400 text-[9.5px]';
        }} else {{
          alertEl.textContent = `APPROCHE NOMINALE (${{Math.round(ooliteSpeed)}} m/s < 95 m/s) • PISTE VERROUILLÉE`;
          alertEl.className = 'font-display font-black tracking-widest text-tacticalGreen text-[9.5px]';
        }}
      }}

      // 5. Check Touchdown at 0 m
      if (ooliteAltitude <= 0) {{
        if (ooliteSpeed <= 115 && Math.abs(landerX) <= 0.78) {{
          completeStage3TouchdownSuccess();
        }} else {{
          // Safe emergency bounce so player can brake and land right away
          ooliteAltitude = 620;
          ooliteSpeed = 72.0;
          landerX *= 0.3;
          playSciFiTone(220, 110, 220, 'sawtooth', 0.08);
          showToast('⚠️ REBOND BOUCLIER ! Maintenez [ESPACE] ou [Z] pour freiner sous 95 m/s et restez au centre !');
        }}
      }}
    }}

    // 6. Render 3D Glide-Slope Tunnel ("Highway-in-the-Sky") & Lander Avatar
    ctx.clearRect(0, 0, W, H);
    const cx = W * 0.5;
    const horizonY = 26;

    // Perspective tunnel frames
    for (let i = 1; i <= 5; i++) {{
      const t = i / 5;
      const rw = 44 + t * 170;
      const rh = 16 + t * 48;
      const ry = horizonY + t * 52;
      ctx.strokeStyle = (ooliteSpeed <= 95) ? `rgba(0, 255, 136, ${{0.2 + t * 0.35}})` : `rgba(255, 184, 0, ${{0.25 + t * 0.4}})`;
      ctx.lineWidth = 1.3;
      ctx.strokeRect(cx - rw / 2, ry - rh / 2, rw, rh);
    }}

    // Green Target Landing Pad at bottom-center of glide slope
    const padProgress = 1 - Math.min(1, ooliteAltitude / 3420);
    const padW = 48 + padProgress * 70;
    const padY = H - 24;
    ctx.fillStyle = 'rgba(0, 255, 136, 0.22)';
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = 2;
    ctx.fillRect(cx - padW / 2, padY - 8, padW, 14);
    ctx.strokeRect(cx - padW / 2, padY - 8, padW, 14);
    ctx.fillStyle = '#00ff88';
    ctx.font = 'bold 8.5px monospace';
    ctx.fillText('⊕ PISTE CIBLE', cx - 32, padY + 2);

    // Draw Player's Descending Lander Avatar (`🛸 VOUS`)
    const shipScreenX = cx + landerX * (W * 0.36);
    const shipScreenY = 38 + padProgress * (padY - 46);
    ctx.save();
    ctx.translate(shipScreenX, shipScreenY);
    if (input.retroHeld || retroFiringFrames > 0) {{
      // Retro-thruster braking plasma flames
      ctx.fillStyle = '#00f0ff';
      ctx.beginPath();
      ctx.moveTo(-8, 8);
      ctx.lineTo(0, 22 + Math.random() * 6);
      ctx.lineTo(8, 8);
      ctx.closePath();
      ctx.fill();
    }}
    ctx.fillStyle = (ooliteSpeed <= 95) ? '#00ff88' : '#ffb800';
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, -10);
    ctx.lineTo(14, 8);
    ctx.lineTo(-14, 8);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 8.5px monospace';
    ctx.fillText('🛸 VOUS', -18, -13);
    ctx.restore();
  }}

  function masterGameLoop() {{
    const input = pollGamepadAndInputs();
    if (currentStage === 1) {{
      updateAndRenderStage1(input);
    }} else if (currentStage === 2) {{
      updateAndRenderStage2(input);
    }} else if (currentStage === 3) {{
      updateAndRenderStage3(input);
    }}
    requestAnimationFrame(masterGameLoop);
  }}

  window.simulateSteamRegisterLoginAndContribute = async function() {{
    const badge = document.getElementById('steam-auth-badge');
    if (badge) badge.textContent = '🔄 STEAM (AppID 3133700) + BOINC CUDA PTX SYNC...';
    const pName = (PLANET_DATA[currentPlanetKey] && PLANET_DATA[currentPlanetKey].name) || 'LHS 1140 b';
    try {{
      const regRes = await fetch('/v1/hivemind/steam/register-login', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
          handle: 'Cmdr_RTX2070_Steam',
          steam_id64: '76561198120474852',
          app_id: 3133700,
          gpu_model: 'NVIDIA GeForce RTX 2070 (8 GB VRAM)',
          role: 'prosumer'
        }})
      }});
      if (regRes.ok) {{
        const regData = await regRes.json();
        const apiKey = regData.api_key || '';
        if (apiKey) localStorage.setItem('hivemind_api_key', apiKey);
        const boincRes = await fetch('/v1/hivemind/steam/contribute-boinc', {{
          method: 'POST',
          headers: {{
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + apiKey
          }},
          body: JSON.stringify({{
            handle: regData.handle || 'Cmdr_RTX2070_Steam',
            target_planet: pName,
            app_id: 3133700
          }})
        }});
        const boincData = await boincRes.json();
        const newBalance = Math.round(Number(boincData.balance_hmt || regData.balance_hmt || 2650));
        scoreHmt = Math.max(scoreHmt, newBalance);
        const scoreEl = document.getElementById('score-hmt');
        if (scoreEl) scoreEl.textContent = scoreHmt.toLocaleString('en-US');
        if (badge) {{
          badge.textContent = `🟢 STEAM (AppID 3133700) • ${{regData.handle}} • +${{boincData.hmt_awarded}} HMT & +${{boincData.cobblestones_awarded}} BOINC`;
        }}
        fetch('https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/api/v1/community/boinc-sync', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            contributor_handle: regData.handle || 'Cmdr_RTX2070_Steam',
            gpu_model: 'NVIDIA GeForce RTX 2070 (8 GB VRAM)',
            gpu_utilization_pct: 5.8,
            gpu_idle_seconds: 12.0,
            target_planet: pName,
            project_id: 'BOINC-ASTRO-BLS-TRANSIT'
          }})
        }}).catch(() => {{}});
        return;
      }}
    }} catch (err) {{}}
    // Fallback to live GCP Cloud Run coordinator when hosted statically (e.g., GitHub Pages)
    try {{
      const gcpRes = await fetch('https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/api/v1/community/boinc-sync', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
          contributor_handle: 'Cmdr_RTX2070_Steam',
          gpu_model: 'NVIDIA GeForce RTX 2070 (8 GB VRAM)',
          gpu_utilization_pct: 5.8,
          gpu_idle_seconds: 12.0,
          target_planet: pName,
          project_id: 'BOINC-ASTRO-BLS-TRANSIT'
        }})
      }});
      const gcpData = await gcpRes.json();
      scoreHmt += Number(gcpData.hmt_points_awarded || 250);
      const scoreEl = document.getElementById('score-hmt');
      if (scoreEl) scoreEl.textContent = scoreHmt.toLocaleString('en-US');
      if (badge) {{
        badge.textContent = `🟢 STEAM + GCP CLOUD RUN • Cmdr_RTX2070_Steam • +${{gcpData.hmt_points_awarded || 250}} HMT & +${{gcpData.cobblestones_awarded || 45}} BOINC`;
      }}
    }} catch (gcpErr) {{
      if (badge) badge.textContent = '🟢 STEAM LOCAL MODE ACTIVE (+250 HMT)';
    }}
  }};

  window.switchStage = switchStage;
  window.selectPlanet = selectPlanet;
  window.fireRetroBurn = fireRetroBurn;

  // Initialize unseen NASA mystery planet for this new game, update HUD, and start 60 FPS loop
  selectRandomNasaMysteryPlanet(false);
  updateCinematicBannerUI();
  requestAnimationFrame(masterGameLoop);
  simulateSteamRegisterLoginAndContribute();
"""
    # Remove legacy conflicting keyboard navigation & quick-action listeners from base Stitch HTML
    import re
    raw = re.sub(
        r"// Quick Action Buttons\s*document\.getElementById\('btn-engage-warp'\)[\s\S]*?document\.getElementById\('btn-nirspec'\)\.addEventListener\('click',\s*\(\)\s*=>\s*\{\s*switchStage\(4\);\s*\}\);",
        "// Quick Action Buttons handled by unified 4-stage cinematic engine",
        raw,
    )
    raw = re.sub(
        r"// KEYBOARD NAVIGATION\s*const planetKeysList = Object\.keys\(PLANET_DATA\);\s*window\.addEventListener\('keydown',[\s\S]*?\}\);\s*// SIMULATED TICK OSCILLATION",
        "const planetKeysList = Object.keys(PLANET_DATA);\n  // SIMULATED TICK OSCILLATION",
        raw,
    )

    last_script_idx = raw.rfind("</script>")
    if last_script_idx != -1:
        raw = raw[:last_script_idx] + extra_js + "\n</script>" + raw[last_script_idx + len("</script>") :]
    return raw


def render_stitch_v46_cockpit_html(state_json: str = "{}") -> str:
    """Render the USS Hivemind NCC-31337 v4.6 Complete Tactical Cockpit HTML."""
    return build_stitch_v46_cockpit_html(state_json=state_json)

