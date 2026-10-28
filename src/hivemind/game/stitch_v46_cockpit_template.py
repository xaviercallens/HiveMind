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

from hivemind.config import PROJECT_ROOT

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
<!-- TOP-LEFT SUB-SPATIAL ASTROMETRIC SCAN HUD (From Stitch Screen 11) -->
<div class="absolute bottom-12 left-2.5 z-30 bg-[#050d1a]/90 border border-cyan-500/60 rounded p-1.5 w-64 shadow-lg backdrop-blur-md pointer-events-none hidden lg:block font-mono">
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
    <div class="col-span-12 lg:col-span-3 flex flex-col gap-2">
      <!-- Panel 1: Ion Propulsion Telemetry -->
      <div class="bg-[#071122]/90 border border-cyan-500/40 rounded p-2 flex flex-col justify-between shadow-lg">
        <div class="flex items-center justify-between border-b border-cyan-800/80 pb-1 mb-1.5">
          <span class="text-[10px] font-bold text-tacticalCyan font-display tracking-wider">ION PROPULSION TELEMETRY</span>
          <span class="text-[9px] bg-cyan-950 border border-tacticalCyan/60 text-tacticalCyan px-1.5 py-0.2 rounded font-bold" id="ion-thrust">THRUST 94.2% (1,840 mN)</span>
        </div>
        <div class="space-y-1.5 text-[9px]">
          <div>
            <div class="flex justify-between text-slate-300 mb-0.5">
              <span>VECTOR ALPHA • FORWARD AXIAL</span>
              <span class="text-tacticalCyan font-bold" id="vec-alpha-val">94.2% // 1,840 mN</span>
            </div>
            <div class="grid grid-cols-10 gap-0.5 h-2">
              <div class="bg-tacticalCyan rounded-xs"></div><div class="bg-tacticalCyan rounded-xs"></div><div class="bg-tacticalCyan rounded-xs"></div><div class="bg-tacticalCyan rounded-xs"></div><div class="bg-tacticalCyan rounded-xs"></div><div class="bg-tacticalCyan rounded-xs"></div><div class="bg-tacticalCyan rounded-xs"></div><div class="bg-tacticalCyan rounded-xs"></div><div class="bg-tacticalCyan rounded-xs"></div><div class="bg-cyan-950 rounded-xs"></div>
            </div>
          </div>
          <div>
            <div class="flex justify-between text-slate-300 mb-0.5">
              <span>VECTOR BETA • INCLINATION BIAS</span>
              <span class="text-tacticalGreen font-bold" id="vec-beta-val">91.8% // 1,790 mN</span>
            </div>
            <div class="grid grid-cols-10 gap-0.5 h-2">
              <div class="bg-tacticalGreen rounded-xs"></div><div class="bg-tacticalGreen rounded-xs"></div><div class="bg-tacticalGreen rounded-xs"></div><div class="bg-tacticalGreen rounded-xs"></div><div class="bg-tacticalGreen rounded-xs"></div><div class="bg-tacticalGreen rounded-xs"></div><div class="bg-tacticalGreen rounded-xs"></div><div class="bg-tacticalGreen rounded-xs"></div><div class="bg-tacticalGreen rounded-xs"></div><div class="bg-emerald-950 rounded-xs"></div>
            </div>
          </div>
          <div>
            <div class="flex justify-between text-slate-300 mb-0.5">
              <span>VECTOR GAMMA • SYNODIC SYNC</span>
              <span class="text-tacticalAmber font-bold" id="vec-gamma-val">99.9% // LOCKED</span>
            </div>
            <div class="grid grid-cols-10 gap-0.5 h-2">
              <div class="bg-tacticalAmber rounded-xs"></div><div class="bg-tacticalAmber rounded-xs"></div><div class="bg-tacticalAmber rounded-xs"></div><div class="bg-tacticalAmber rounded-xs"></div><div class="bg-tacticalAmber rounded-xs"></div><div class="bg-tacticalAmber rounded-xs"></div><div class="bg-tacticalAmber rounded-xs"></div><div class="bg-tacticalAmber rounded-xs"></div><div class="bg-tacticalAmber rounded-xs"></div><div class="bg-tacticalAmber rounded-xs"></div>
            </div>
          </div>
        </div>
        <div class="mt-2 pt-1 border-t border-cyan-900/60 flex justify-between text-[9px] text-slate-400">
          <span>SPECIFIC IMPULSE (Isp)</span>
          <span class="text-white font-bold">4,180 s [XENON-HALL]</span>
        </div>
      </div>

      <!-- Panel 2: Orbital Eccentricity Decay (e) -->
      <div class="bg-[#071122]/90 border border-cyan-500/40 rounded p-2 flex flex-col justify-between flex-1 shadow-lg">
        <div class="flex items-center justify-between border-b border-cyan-800/80 pb-1">
          <span class="text-[10px] font-bold text-tacticalCyan font-display">ORBITAL ECCENTRICITY DECAY (e)</span>
          <span class="text-[8.5px] text-tacticalGreen font-bold">● CIRCULARIZATION STABLE</span>
        </div>
        <div class="relative h-24 w-full bg-[#030812] border border-cyan-950 rounded my-1 p-1 flex items-center justify-center">
          <svg class="w-full h-full" viewBox="0 0 240 85">
            <line stroke="rgba(0,240,255,0.15)" stroke-dasharray="2,2" stroke-width="1" x1="0" x2="240" y1="25" y2="25"></line>
            <line stroke="rgba(0,255,136,0.35)" stroke-dasharray="3,2" stroke-width="1" x1="0" x2="240" y1="68" y2="68"></line>
            <text fill="#00ff88" font-size="7.5" x="172" y="64">e_crit = 0.012</text>
            <path d="M 10,14 Q 65,18 110,46 T 230,66" fill="none" stroke="#00f0ff" stroke-width="2.2"></path>
            <circle cx="15" cy="15" fill="#ffb800" r="3"></circle>
            <circle cx="115" cy="48" fill="#00f0ff" r="3"></circle>
            <circle cx="225" cy="66" fill="#00ff88" r="3.5"></circle>
          </svg>
        </div>
        <div class="grid grid-cols-3 gap-1 text-[8.5px] text-center">
          <div class="bg-black/50 p-1 rounded border border-slate-800">
            <div class="text-slate-400">APOAPSIS</div>
            <div class="text-amber-300 font-bold">e = 0.090</div>
          </div>
          <div class="bg-black/50 p-1 rounded border border-slate-800">
            <div class="text-slate-400">Δ-V STABILIZED</div>
            <div class="text-tacticalCyan font-bold">-1.42 km/s</div>
          </div>
          <div class="bg-black/50 p-1 rounded border border-emerald-900/60">
            <div class="text-slate-400">FINAL LOCK</div>
            <div class="text-tacticalGreen font-bold" id="s2-ecc-final">e = 0.012</div>
          </div>
        </div>
      </div>
    </div>

    <!-- CENTER COLUMN (5 cols): VIEWPORT // ASTROMETRIC-02 LAGRANGE RING L1-L5 -->
    <div class="col-span-12 lg:col-span-5 bg-[#030916]/95 border border-cyan-500/50 rounded p-2 flex flex-col justify-between relative overflow-hidden shadow-2xl">
      <div class="absolute inset-0 opacity-15 pointer-events-none" style="background-image: radial-gradient(circle, #00f0ff 1px, transparent 1px); background-size: 26px 26px;"></div>
      <!-- Top Telemetry Overlay -->
      <div class="relative z-10 flex items-center justify-between text-[9.5px] border-b border-cyan-900/70 pb-1">
        <div class="flex items-center gap-1.5">
          <span class="w-2 h-2 rounded-full bg-tacticalCyan animate-ping"></span>
          <span class="font-display font-bold text-white">VIEWPORT // ASTROMETRIC-02</span>
        </div>
        <div class="flex items-center gap-2">
          <span class="text-slate-300">SAT-NODES: <strong class="text-tacticalGreen" id="sat-count">128 / 128</strong></span>
          <span class="bg-emerald-950 border border-emerald-500/60 text-tacticalGreen px-1.5 py-0.2 rounded font-bold text-[8.5px]">RADAR LOCK 99.8%</span>
        </div>
      </div>

      <!-- Interactive SVG Lagrange Ring L1-L5 & Orbiting Exoplanet -->
      <div class="relative flex-1 flex items-center justify-center my-1">
        <svg class="w-full h-full max-h-[260px]" viewBox="0 0 360 240">
          <!-- Outer L2 / L3 Trojan Ring -->
          <ellipse cx="180" cy="120" fill="none" rx="150" ry="96" stroke="rgba(0,240,255,0.22)" stroke-dasharray="4 3" stroke-width="1.2"></ellipse>
          <!-- Inner BEE-MOE Resonance Ring -->
          <ellipse cx="180" cy="120" fill="none" rx="102" ry="66" stroke="rgba(0,255,136,0.45)" stroke-width="1.5"></ellipse>
          <!-- Synodic Lock Projection Arc -->
          <path d="M 78,120 A 102,66 0 0,1 282,120" fill="none" stroke="rgba(255,184,0,0.55)" stroke-dasharray="5 3" stroke-width="2"></path>
          <!-- Host Star M-Dwarf (Left Focal Point) -->
          <circle cx="115" cy="120" fill="#ffb800" r="10"></circle>
          <circle cx="115" cy="120" fill="none" r="15" stroke="rgba(255,184,0,0.45)" stroke-width="1.5"></circle>
          <text fill="#ffb800" font-size="8" font-weight="bold" x="98" y="145">HOST ★</text>
          <!-- Lagrange Points L1, L2, L3, L4, L5 -->
          <circle cx="155" cy="120" fill="#00f0ff" r="4"></circle>
          <text fill="#00f0ff" font-size="8" font-weight="bold" x="142" y="110">L1 [GATE]</text>
          <circle cx="330" cy="120" fill="#00f0ff" r="4"></circle>
          <text fill="#00f0ff" font-size="8" x="285" y="112">L2 [JWST]</text>
          <circle cx="30" cy="120" fill="#64748b" r="3.5"></circle>
          <text fill="#94a3b8" font-size="7.5" x="18" y="110">L3</text>
          <!-- L4 Station & USS HIVEMIND -->
          <circle cx="215" cy="56" fill="#00ff88" r="5"></circle>
          <text fill="#00ff88" font-size="8" font-weight="bold" x="145" y="45">L4 STATION // USS HIVEMIND</text>
          <!-- L5 Trojan Relay -->
          <circle cx="215" cy="184" fill="#ffb800" r="4.5"></circle>
          <text fill="#ffb800" font-size="8" x="170" y="200">L5 [BEE-RELAY]</text>
        </svg>
        <!-- Center-Right Photorealistic Planet Avatar inside Ring -->
        <div class="absolute right-[24%] top-1/2 -translate-y-1/2 flex flex-col items-center pointer-events-none">
          <img alt="Orbiting Exoplanet" class="w-12 h-12 rounded-full object-cover border-2 border-tacticalCyan shadow-cyanGlow" id="stage2-planet-img" src="/stitch-assets/stitch_v2_05_photorealistic_exoplanet_lhs_1140_b_super-earth.png"/>
          <span class="text-[8.5px] bg-black/90 text-tacticalAmber border border-amber-500/60 px-1.5 py-0.2 rounded mt-1 font-bold" id="current-stage2-planet">LHS 1140 b (Super-Terre)</span>
        </div>
      </div>

      <!-- Bottom Orbital Metrics Bar -->
      <div class="relative z-10 grid grid-cols-4 gap-1 bg-black/80 border border-cyan-900/80 p-1.5 rounded text-[8.5px]">
        <div><span class="text-slate-400">ALT:</span> <strong class="text-tacticalCyan">42,160 KM</strong></div>
        <div><span class="text-slate-400">INCL:</span> <strong class="text-tacticalGreen">0.04°</strong></div>
        <div><span class="text-slate-400">SYNODIC:</span> <strong class="text-tacticalAmber">ENGAGED</strong></div>
        <div><span class="text-slate-400">FLUX:</span> <strong class="text-emerald-300">NOMINAL</strong></div>
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
            + SONDE MOE
          </button>
          <button class="bg-cyan-950 hover:bg-cyan-800 border border-cyan-400 text-cyan-200 font-bold px-2 py-1 text-[9px] rounded transition cursor-pointer active:scale-95" id="btn-sync-phase">
            SYNC L1-L2
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

      <!-- CENTER COLUMN: Flight Crosshair, Pitch Ladder, Altitude/Speed Tapes, Retro-Propulsion Slider (Screen 06) & Action Toggles -->
      <div class="col-span-12 lg:col-span-6 flex flex-col justify-between items-center relative">
        <!-- Danger / Altitude Critical Warning Banner -->
        <div class="w-full flex items-center justify-between gap-1.5">
          <span class="bg-cyan-950/90 border border-tacticalCyan/60 text-tacticalCyan px-2 py-0.5 rounded text-[8.5px] font-mono font-bold">[RADAR-SAR: ACTIVE] • G-LOAD: 1.84 G</span>
          <div class="border-2 border-red-500/80 bg-red-950/85 px-2.5 py-0.5 rounded shadow-redGlow flex items-center gap-1.5" id="oolite-danger-banner">
            <span class="text-red-400 font-bold text-xs animate-pulse">⚠️</span>
            <div class="text-center leading-tight">
              <div class="font-display font-black tracking-widest text-red-400 text-[10px]" id="terrain-alert">DANGER ! ALTITUDE TOO LOW</div>
              <div class="text-[8px] text-red-200">CIBLE: <span class="text-white font-bold" id="landing-zone-name">CHASMA AUSTRALIS</span></div>
            </div>
          </div>
          <span class="bg-emerald-950/90 border border-emerald-400/60 text-tacticalGreen px-2 py-0.5 rounded text-[8.5px] font-mono font-bold">📍 BEACON LOCKED</span>
        </div>

        <!-- Central Circular Reticle & Pitch Ladder (+10° to -10°) -->
        <div class="relative w-48 h-32 flex items-center justify-center my-0.5">
          <div class="absolute w-32 h-32 rounded-full border border-tacticalCyan/40 border-dashed animate-spin" style="animation-duration: 50s;"></div>
          <div class="absolute w-24 h-24 rounded-full border border-emerald-400/30"></div>
          <div class="absolute inset-0 flex flex-col justify-between items-center py-1.5 text-[8px] text-tacticalCyan font-mono opacity-90">
            <div class="flex items-center gap-2"><span>-- +10° --</span><div class="w-6 h-[1px] bg-tacticalCyan"></div><span>-- +10° --</span></div>
            <div class="flex items-center gap-3"><span>-- +05° --</span><div class="w-10 h-[1px] bg-tacticalGreen"></div><span>-- +05° --</span></div>
            <div class="w-full flex items-center justify-center gap-1.5">
              <div class="w-10 h-[2px] bg-tacticalCyan shadow-cyanGlow"></div>
              <div class="w-3 h-3 border-2 border-tacticalCyan rotate-45 flex items-center justify-center">
                <div class="w-1 h-1 bg-tacticalAmber rounded-full"></div>
              </div>
              <div class="w-10 h-[2px] bg-tacticalCyan shadow-cyanGlow"></div>
            </div>
            <div class="flex items-center gap-3"><span>-- -05° --</span><div class="w-10 h-[1px] bg-tacticalGreen"></div><span>-- -05° --</span></div>
            <div class="flex items-center gap-2"><span>-- -10° --</span><div class="w-6 h-[1px] bg-tacticalCyan"></div><span>-- -10° --</span></div>
          </div>
          <!-- Radar Alt Tape (Left) -->
          <div class="absolute -left-14 top-1/2 -translate-y-1/2 flex flex-col gap-0.5 text-[8.5px] text-tacticalCyan">
            <span class="bg-tacticalCyan text-black px-1.5 py-0.5 font-bold rounded-sm" id="altitude-tape">1,840 m</span>
            <span class="text-[7.5px] text-cyan-300">ALT AGL</span>
          </div>
          <!-- Vertical Descent Rate Tape (Right) -->
          <div class="absolute -right-14 top-1/2 -translate-y-1/2 flex flex-col gap-0.5 items-end text-[8.5px] text-amber-300">
            <span class="bg-tacticalAmber text-black px-1.5 py-0.5 font-bold rounded-sm" id="speed-tape">84.2 m/s</span>
            <span class="text-[7.5px] text-emerald-300" id="vs-tape">VS -12.4 m/s</span>
          </div>
        </div>

        <!-- Interactive Retro-Propulsion Slider (From Screen 06) + Landing Action Buttons ([C] / [G] / [Z/UP]) + Chasma Tower Comms -->
        <div class="w-full max-w-lg flex flex-col gap-1">
          <div class="bg-[#070e1a]/90 border border-cyan-500/40 rounded px-2 py-0.5 flex items-center justify-between gap-2 text-[9px] font-mono">
            <span class="text-slate-300 font-bold">RETRO-PROPULSION OUTPUT: <strong class="text-tacticalCyan" id="thrustReadout">75%</strong></span>
            <input class="w-36 accent-cyan-400 cursor-pointer h-1.5" id="thrustSlider" max="100" min="20" oninput="updateRetroSlider(this.value)" type="range" value="75"/>
            <span class="text-[8px] text-tacticalGreen">THERMAL NOMINAL</span>
          </div>
          <div class="grid grid-cols-3 gap-1.5">
            <button class="px-2 py-0.5 rounded bg-emerald-950/90 border border-emerald-400/80 text-emerald-300 flex items-center justify-center gap-1 hover:bg-emerald-900 text-[9.5px] font-bold cursor-pointer" id="btn-clearance-toggle">
              <span class="text-tacticalGreen">✓</span> <span>[C] CLEARANCE</span>
            </button>
            <button class="px-2 py-0.5 rounded bg-cyan-950/90 border border-tacticalCyan/80 text-cyan-200 flex items-center justify-center gap-1 hover:bg-cyan-900 text-[9.5px] font-bold cursor-pointer" id="btn-gear-toggle">
              <span class="text-tacticalCyan">✓</span> <span>[G] TRAIN SORTI</span>
            </button>
            <button class="px-2 py-0.5 rounded bg-amber-500 hover:bg-amber-400 border border-amber-200 text-black font-extrabold flex items-center justify-center gap-1 shadow-amberGlow text-[9.5px] cursor-pointer active:scale-95" id="btn-retro-burn">
              <span>🔥 [Z/UP] RÉTROS ACTIFS</span>
            </button>
          </div>
          <div class="bg-[#070e1a]/90 border border-cyan-500/40 rounded p-1 text-[8.5px] text-slate-300 leading-tight">
            <div class="flex items-center justify-between text-amber-300 font-bold">
              <span>CANAL CONTRÔLE PLANÉTAIRE [<span id="oolite-comms-planet">LHS 1140 b</span>] • 124.85 MHz</span>
              <span class="text-emerald-300">[Tour Chasma] : Maintenir V &lt; 95 m/s</span>
            </div>
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
        <div class="bg-black/60 border border-cyan-900/80 p-1.5 rounded text-[8.5px] my-0.5">
          <div class="flex justify-between items-center mb-1">
            <span class="text-slate-300 font-bold">THERMAL GRADIENT MODEL (T_eq)</span>
            <span class="text-tacticalGreen font-bold" id="s4-teq-label">▲ 235 K (-38°C) HAB_ZONE_LOCK</span>
          </div>
          <div class="w-full h-2 rounded bg-gradient-to-r from-cyan-500 via-emerald-400 to-red-500 relative">
            <div class="w-2.5 h-3.5 bg-white border border-black rounded-xs absolute -top-0.5 left-[44%]" id="s4-teq-marker"></div>
          </div>
          <div class="flex justify-between text-[7.5px] text-slate-400 mt-0.5">
            <span>100 K [CRYOGENIC]</span>
            <span class="text-emerald-300 font-bold">HAB_ZONE (180–310 K)</span>
            <span>400 K+ [SUPERHEATED]</span>
          </div>
        </div>

        <!-- Action Buttons -->
        <div class="flex flex-wrap gap-1 pt-1">
          <button class="bg-cyan-950 hover:bg-cyan-800 border border-cyan-500 text-tacticalCyan px-2 py-1 rounded text-[9.5px] font-bold transition cursor-pointer" id="btn-run-adql">
            🌐 NASA TAP ADQL
          </button>
          <button class="bg-emerald-950 hover:bg-emerald-800 border border-emerald-500 text-tacticalGreen px-2 py-1 rounded text-[9.5px] font-bold transition cursor-pointer" id="btn-verify-kepler">
            📐 KEPLER T²/a³
          </button>
          <button class="bg-emerald-500 hover:bg-emerald-400 text-black px-2 py-1 rounded text-[9.5px] font-extrabold transition cursor-pointer" id="btn-approve" onclick="submitStitchDiscovery(true)">
            ✅ APPROUVER (+600 HMT)
          </button>
          <button class="bg-red-600 hover:bg-red-500 text-white px-2 py-1 rounded text-[9.5px] font-extrabold transition cursor-pointer" id="btn-reject" onclick="submitStitchDiscovery(false)">
            🚨 REJETER
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
          <div class="flex gap-2">
            <button class="flex-1 bg-emerald-500 hover:bg-emerald-400 text-black font-extrabold py-1 px-2 rounded text-[9.5px] cursor-pointer transition" onclick="simulateSteamRegisterLoginAndContribute()">
              ⚡ EXÉCUTER WORKUNIT BOINC RTX 2070 (+250 HMT)
            </button>
            <a class="bg-cyan-950 hover:bg-cyan-800 border border-tacticalCyan text-tacticalCyan font-bold py-1 px-2 rounded text-[9.5px]" href="/api/steam/fat-client-manifest" target="_blank">
              📦 STEAM MANIFEST JSON
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
<a href="/webgpu" target="_blank" class="bg-purple-950/80 hover:bg-purple-800 border border-purple-400 text-purple-200 px-2 py-0.5 rounded font-bold">🌌 WebGPU (/webgpu)</a>
<a href="/portal#steam-free-game" target="_blank" class="bg-amber-950/80 hover:bg-amber-800 border border-tacticalAmber text-tacticalAmber px-2 py-0.5 rounded font-bold">🐝 Portal</a>
<a href="/community" target="_blank" class="bg-emerald-950/80 hover:bg-emerald-800 border border-emerald-400 text-emerald-300 px-2 py-0.5 rounded font-bold">⚡ BOINC Hub</a>
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

    # 4. Expand Central Viewport height slightly so all 5 Stitch stages render without clipping
    raw = raw.replace(
        'class="relative w-full h-[400px] xl:h-[440px] rounded border border-cyan-600/50 overflow-hidden bg-black shadow-2xl"',
        'class="relative w-full h-[470px] xl:h-[490px] rounded border border-cyan-600/50 overflow-hidden bg-black shadow-2xl"',
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

    # 7. Inject Steam registration/login + BOINC GPU contribution + 5-Language localization + 5-Stage & All-Screen sync before last </script>
    extra_js = f"""
  // === HIVEMIND LIVE BACKEND, ALL 12 STITCH SCREENS SYNC, STEAM + BOINC RTX 2070, & 5-LANGUAGE INTEGRATION ===
  window.HIVEMIND_EMBEDDED_STATE = {state_json};
  let scoreHmt = 2400;
  let currentLang = 'fr';
  let retroThrustPct = 75;

  // Register Stage 5 in stagePanels, stageDescriptions, and stageProgressLabels
  stagePanels[5] = document.getElementById('viewport-stage-5');
  stageDescriptions[5] = "ÉTAPE 5 // ARCHITECTURE DUAL-ENGINE // FAT CLIENT STEAM (UE5/UNITY + RTX 2070 BOINC) vs. WEBGPU ZERO-INSTALL (/webgpu)";
  stageProgressLabels[5] = "STAGE 5/5 [DUAL-ENGINE & BOINC]";

  const STITCH_I18N = {{
    fr: {{
      s1: "ÉTAPE 1 // PILOTEZ L'USS HIVEMIND (ZQSD / FLÈCHES / CLIC) VERS UNE ÉTOILE PUIS [ESPACE / CLIC] POUR DÉPLOYER LE SWARM",
      s2: "ÉTAPE 2 // DÉPLOIEMENT DU SWARM BEE EN COURS // SYNCHRONISATION LAGRANGE L1-L5 & ÉJECTION DE 128 MICROSATELLITES",
      s3: "ÉTAPE 3 // ATTERRISSAGE OOLITE PLANETFALL // CONTRÔLE DES RÉTRO-FUSÉES [Z/HAUT] POUR FREINER SOUS 95 m/s",
      s4: "ÉTAPE 4 // SPECTROGRAMME NIRSPEC•MIRI JWST (BERYLLIUM 6.5M) // 3ÈME LOI DE KEPLER (T²/a³ = k) & BIOSIGNATURES",
      s5: "ÉTAPE 5 // ARCHITECTURE DUAL-ENGINE // FAT CLIENT STEAM (UE5/UNITY + RTX 2070 BOINC) vs. WEBGPU ZERO-INSTALL (/webgpu)"
    }},
    es: {{
      s1: "ETAPA 1 // PILOTA EL USS HIVEMIND (WASD / FLECHAS / CLIC) HACIA UNA ESTRELLA Y PULSA [ESPACIO] PARA DESPLEGAR EL ENJAMBRE BEE-MOE",
      s2: "ETAPA 2 // DESPLIEGUE DEL ENJAMBRE BEE-MOE // SINCRONIZACIÓN LAGRANGE L1-L5 Y 128 MICROSATÉLITES ACTIVOS",
      s3: "ETAPA 3 // ATERRIZAJE OOLITE PLANETFALL // ACTIVA RETROCOHETES [Z/ARRIBA] PARA FRENAR BAJO 95 m/s",
      s4: "ETAPA 4 // ESPECTRÓGRAFO JWST NIRSPEC•MIRI // VERIFICACIÓN DE LA 3ª LEY DE KEPLER (T²/a³ = k) Y BIOFIRMAS",
      s5: "ETAPA 5 // ARQUITECTURA DUAL-ENGINE // CLIENTE NATIVO STEAM (RTX 2070 BOINC) vs. WEBGPU SIN INSTALACIÓN (/webgpu)"
    }},
    de: {{
      s1: "STUFE 1 // STEUERE DIE USS HIVEMIND (WASD / KLICK) ZU EINEM STERN & DRÜCKE [LEERTASTE] FÜR BEE-MOE SCHWARM-START",
      s2: "STUFE 2 // BEE-MOE SCHWARM IM EINSATZ // LAGRANGE-SYNCHRONISATION L1-L5 & 128 AKTIVE MIKROSATELLITEN",
      s3: "STUFE 3 // OOLITE PLANETFALL LANDUNG // BREMSRAKETEN [Z/OBEN] ZÜNDEN UNTER 95 m/s",
      s4: "STUFE 4 // JWST NIRSPEC•MIRI SPEKTROSKOPIE // 3. KEPLERSCHES GESETZ (T²/a³ = k) & BIOSIGNATUR-PRÜFUNG",
      s5: "STUFE 5 // DUAL-ENGINE ARCHITEKTUR // STEAM FAT CLIENT (RTX 2070 BOINC) vs. ZERO-INSTALL WEBGPU (/webgpu)"
    }},
    en: {{
      s1: "STAGE 1 // PILOT USS HIVEMIND (WASD / ARROWS / CLICK) TO A STAR THEN PRESS [SPACE / CLICK] TO DEPLOY THE BEE-MOE SWARM",
      s2: "STAGE 2 // BEE-MOE SWARM DEPLOYMENT // LAGRANGE L1-L5 PHASE LOCK & 128 ACTIVE MICROSATELLITES",
      s3: "STAGE 3 // OOLITE PLANETFALL DESCENT // FIRE RETRO-THRUSTERS [Z/UP] TO BRAKE BELOW 95 m/s",
      s4: "STAGE 4 // JWST NIRSPEC•MIRI SPECTROSCOPY (BERYLLIUM 6.5M) // KEPLER'S 3RD LAW (T²/a³ = k) & BIOSIGNATURES",
      s5: "STAGE 5 // DUAL-ENGINE ARCHITECTURE // STEAM FAT CLIENT (UE5/UNITY + RTX 2070 BOINC) vs. WEBGPU ZERO-INSTALL (/webgpu)"
    }},
    zh: {{
      s1: "第 1 阶段 // 驾驶蜂群号星舰（WASD / 方向键 / 点击）前往恒星，按 [空格键] 部署 Bee-MoE 探测器蜂群",
      s2: "第 2 阶段 // Bee-MoE 轨道蜂群部署中 // 拉格朗日 L1-L5 同步锁定与 128 颗微卫星遥测",
      s3: "第 3 阶段 // Oolite 行星大气层降落 // 点燃反推火箭 [Z/上键] 将速度控制在 95 米/秒以下",
      s4: "第 4 阶段 // 韦伯 NIRSpec•MIRI 透射光谱 // 验证开普勒第三定律 (T²/a³ = k) 与生物特征分子",
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

  // Override switchStage to support all 5 stages cleanly
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
    }}
  }};

  // Synchronize all Stitch Screen panels (Stage 1 Sub-Spatial HUD, Stage 2 Dossier, Stage 3 Oolite Gauges, Stage 4 Beryllium KPIs) on selectPlanet
  const _origSelectPlanet = selectPlanet;
  selectPlanet = function(planetKey) {{
    _origSelectPlanet(planetKey);
    const data = PLANET_DATA[planetKey];
    if (!data) return;

    // Move USS Hivemind ship marker toward target star on Stage 1 Galaxy Map
    const shipMarker = document.getElementById('hivemind-marker');
    if (shipMarker) {{
      const shipX = Math.max(40, Math.min(720, data.nodeX - 75));
      const shipY = Math.max(35, Math.min(330, data.nodeY - 12));
      shipMarker.style.left = shipX + 'px';
      shipMarker.style.top = shipY + 'px';
      const targetX = data.nodeX + 35;
      const targetY = data.nodeY + 35;
      const cone = document.getElementById('svg-sensor-cone');
      const lockLine = document.getElementById('svg-lock-line');
      if (cone && lockLine) {{
        cone.setAttribute('points', `${{shipX + 30}},${{shipY + 18}} ${{targetX - 20}},${{targetY - 15}} ${{targetX + 30}},${{targetY + 35}} ${{shipX + 30}},${{shipY + 18}}`);
        lockLine.setAttribute('x1', String(shipX + 30));
        lockLine.setAttribute('y1', String(shipY + 18));
        lockLine.setAttribute('x2', String(targetX));
        lockLine.setAttribute('y2', String(targetY));
      }}
    }}

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

    // Update Stage 4 Beryllium 6.5M KPIs & Thermal Gradient (Screen 01)
    const s4K = document.getElementById('s4-kpi-k');
    if (s4K && kMatch) s4K.textContent = kMatch[1];
    const s4E = document.getElementById('s4-kpi-e');
    if (s4E && eMatch) s4E.textContent = eMatch[1];
    const s4Depth = document.getElementById('s4-kpi-depth');
    if (s4Depth) s4Depth.textContent = data.transitDepth.replace('DEPTH:', '').split('(')[0].trim();
    const s4Teq = document.getElementById('s4-teq-label');
    if (s4Teq) s4Teq.textContent = `▲ ${{data.temp}} TARGET_LOCK`;
  }};

  window.updateRetroSlider = function(val) {{
    retroThrustPct = parseInt(val) || 75;
    const tr = document.getElementById('thrustReadout');
    if (tr) tr.textContent = retroThrustPct + '%';
  }};

  const _origFireRetroBurn = fireRetroBurn;
  fireRetroBurn = function() {{
    const extraBrake = (retroThrustPct - 50) * 0.12;
    ooliteSpeed = Math.max(12.0, ooliteSpeed - Math.max(0, extraBrake));
    _origFireRetroBurn();
    const gs = document.getElementById('gauge-speed');
    if (gs) gs.textContent = Math.round(ooliteSpeed);
    const ga = document.getElementById('gauge-alt');
    if (ga) ga.textContent = (ooliteAltitude >= 1000) ? `${{(ooliteAltitude / 1000).toFixed(1)}}k` : `${{Math.round(ooliteAltitude)}}m`;
    const vd = document.getElementById('oolite-voyage-dist');
    if (vd) vd.textContent = `${{Math.round(ooliteAltitude).toLocaleString()}} m`;
  }};

  window.sampleAtmosphereStage2 = function() {{
    scoreHmt += 150;
    const scoreEl = document.getElementById('score-hmt');
    if (scoreEl) scoreEl.textContent = scoreHmt.toLocaleString('en-US');
    showToast(`ÉCHANTILLON ATMOSPHÉRIQUE H₂O/DMS COLLECTÉ SUR ${{PLANET_DATA[currentPlanetKey].name}} (+150 HMT)`);
  }};

  window.highlightMolecule = function(mol, wave, pct) {{
    const tag = document.getElementById('dms-detection-tag');
    if (tag) tag.textContent = `● RAIE SPECTRALE VERROUILLÉE: ${{mol}} @ ${{wave}} [CONFIDENCE ${{pct}}]`;
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
    scoreHmt += approved ? 600 : 300;
    const scoreEl = document.getElementById('score-hmt');
    if (scoreEl) scoreEl.textContent = scoreHmt.toLocaleString('en-US');
    showToast(approved ? `DÉCOUVERTE APPROUVÉE SUR ${{PLANET_DATA[currentPlanetKey].name}} (+600 HMT)` : `FAUX POSITIF REJETÉ (+300 HMT)`);
    try {{
      await fetch('/api/astro/discover', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
          target_id: currentPlanetKey,
          player_classification: approved ? 'PLANET' : 'FALSE_POSITIVE'
        }})
      }});
    }} catch (e) {{}}
  }};

  window.simulateSteamRegisterLoginAndContribute = async function() {{
    const badge = document.getElementById('steam-auth-badge');
    if (badge) badge.textContent = '🔄 STEAM (AppID 3133700) + BOINC CUDA PTX SYNC...';
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
      const pName = (PLANET_DATA[currentPlanetKey] && PLANET_DATA[currentPlanetKey].name) || 'LHS 1140 b';
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
    }} catch (err) {{
      if (badge) badge.textContent = '⚠️ STEAM PORTAL OFFLINE';
    }}
  }};

  window.addEventListener('keydown', (e) => {{
    if (e.key === '5') switchStage(5);
  }});

  window.switchStage = switchStage;
  window.selectPlanet = selectPlanet;
  window.fireRetroBurn = fireRetroBurn;
  simulateSteamRegisterLoginAndContribute();
"""
    last_script_idx = raw.rfind("</script>")
    if last_script_idx != -1:
        raw = raw[:last_script_idx] + extra_js + "\n</script>" + raw[last_script_idx + len("</script>") :]
    return raw
