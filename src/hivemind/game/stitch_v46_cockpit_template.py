# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""USS Hivemind NCC-31337 v4.6 Complete Tactical Command Bridge & Oolite Planetfall Suite (`stitch_v46_cockpit_template.py`).

Synthesizes the latest Google Stitch MCP screens from project `10018415280079892710` (`https://stitch.withgoogle.com/projects/10018415280079892710`):
  - Screen 02 (`02_uss_hivemind_ncc-31337_v4_6_-_complete_tactical.html`): Complete Tactical Command Bridge & Game Suite (Stages 1, 2, 3, 4 + Dual Telemetry Oscilloscopes + 10-Slot Astrobiological Collection Dock)
  - Screen 10 (`10_uss_hivemind_ncc-31337_v4_6_-_stage_3_oolite_pl.html`): Full Stage 3 Oolite Planetfall Landing Bridge (Left Cobra MK-III MFD + Primeable Equipments + Cargo Manifest, Center Flight Crosshair & Pitch Ladder + Clearance/Gear/Retro Controls + Chasma Tower Comms, Right Radar Contacts + K2-18 System Bodies, and Bottom 8 Circular Digital Gauges + 3D Wireframe Compass & Cobra MK-III)
  - Local photorealistic Stitch v2 assets served via `/stitch-assets/stitch_v2_*.png`
  - Integrated 5-Language Switcher (`FR`, `ES`, `DE`, `EN`, `ZH`), Neural Voice-Over, NASA TAP ADQL live sync, Free Steam Game (`AppID 3133700`) + RTX 2070 BOINC CUDA PTX folding, and Zero-Install WebGPU (`/webgpu`).
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


STAGE_3_FULL_OOLITE_BRIDGE_HTML = """<!-- STAGE 3 VIEW: FULL OOLITE PLANETFALL LANDING BRIDGE (FROM STITCH SCREEN 10) -->
<div class="absolute inset-0 bg-[#020813] hidden transition-opacity duration-300 overflow-hidden" id="viewport-stage-3">
  <div class="relative w-full h-full flex flex-col justify-between px-2.5 pt-2 pb-12 select-none">
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
            <div>Rép. Colis: <span class="text-cyan-300 font-bold">0.70</span></div>
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
            <li class="text-slate-400">• Adv. Nav Computer (Planetfall OXP)</li>
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

      <!-- CENTER COLUMN: Flight Crosshair, Pitch Ladder, Altitude/Speed Tapes, Action Toggles (C / G / Z) & Tower Comms -->
      <div class="col-span-12 lg:col-span-6 flex flex-col justify-between items-center relative">
        <!-- Danger / Altitude Critical Warning Banner -->
        <div class="border-2 border-red-500/80 bg-red-950/85 px-3 py-0.5 rounded shadow-redGlow flex items-center gap-2" id="oolite-danger-banner">
          <span class="text-red-400 font-bold text-xs animate-pulse">⚠️</span>
          <div class="text-center leading-tight">
            <div class="font-display font-black tracking-widest text-red-400 text-[10.5px]" id="terrain-alert">DANGER ! ALTITUDE TOO LOW</div>
            <div class="text-[8.5px] text-red-200">SEUIL CRITIQUE D'IMPACT PLANÉTAIRE &lt; 2000m • CIBLE: <span class="text-white font-bold" id="landing-zone-name">CHASMA AUSTRALIS</span></div>
          </div>
        </div>

        <!-- Central Circular Reticle & Pitch Ladder (+10° to -10°) -->
        <div class="relative w-48 h-36 flex items-center justify-center my-0.5">
          <div class="absolute w-36 h-36 rounded-full border border-tacticalCyan/40 border-dashed animate-spin" style="animation-duration: 50s;"></div>
          <div class="absolute w-28 h-28 rounded-full border border-emerald-400/30"></div>
          <div class="absolute inset-0 flex flex-col justify-between items-center py-2 text-[8px] text-tacticalCyan font-mono opacity-90">
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
          <div class="absolute -left-12 top-1/2 -translate-y-1/2 flex flex-col gap-0.5 text-[8.5px] text-tacticalCyan">
            <span class="bg-tacticalCyan text-black px-1.5 py-0.5 font-bold rounded-sm" id="altitude-tape">1,840 m</span>
            <span class="text-[7.5px] text-cyan-300">ALT AGL</span>
          </div>
          <!-- Vertical Descent Rate Tape (Right) -->
          <div class="absolute -right-12 top-1/2 -translate-y-1/2 flex flex-col gap-0.5 items-end text-[8.5px] text-amber-300">
            <span class="bg-tacticalAmber text-black px-1.5 py-0.5 font-bold rounded-sm" id="speed-tape">84.2 m/s</span>
            <span class="text-[7.5px] text-emerald-300" id="vs-tape">VS -12.4 m/s</span>
          </div>
          <div class="absolute -bottom-1 text-[8px] text-red-400 tracking-wider font-bold bg-black/70 px-1.5 rounded">
            WEAPONS SYSTEMS OFFLINE (ATMOSPHERIC MODE)
          </div>
        </div>

        <!-- Interactive Landing Action Buttons ([C] CLEARANCE / [G] TRAIN SORTI / [Z/UP] RÉTROS ACTIFS) & Chasma Tower Comms -->
        <div class="w-full max-w-lg flex flex-col gap-1">
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
          <div class="bg-[#070e1a]/90 border border-cyan-500/40 rounded p-1 text-[9px] text-slate-300 leading-tight">
            <div class="flex items-center justify-between text-amber-300 font-bold border-b border-slate-700/60 pb-0.5 mb-0.5">
              <span>CANAL CONTRÔLE PLANÉTAIRE [<span id="oolite-comms-planet">LHS 1140 b / K2-18 b</span>]</span>
              <span class="text-tacticalCyan font-mono">124.85 MHz</span>
            </div>
            <div class="text-emerald-200 truncate">
              <strong class="text-tacticalCyan">[Tour Chasma] :</strong> Autorisation d'atterrissage confirmée. Maintenir vitesse &lt; 95 m/s et T° verrière &lt; 180°C.
            </div>
          </div>
        </div>
      </div>

      <!-- RIGHT COLUMN (MFD 2): Radar Contacts (100 km) & Planetary Bodies in System -->
      <div class="col-span-12 lg:col-span-3 flex flex-col gap-1">
        <div class="bg-[#070e1a]/90 border border-cyan-500/40 rounded p-1.5 flex flex-col gap-0.5 text-[9.5px] shadow-lg backdrop-blur-md">
          <div class="flex items-center justify-between border-b border-cyan-500/30 pb-0.5">
            <span class="text-tacticalCyan font-bold font-display">RADAR CIBLES &amp; CONTACTS</span>
            <span class="text-[8.5px] text-slate-400">PORTÉE: 100 km</span>
          </div>
          <div class="space-y-0.5 text-[9px] font-mono leading-tight">
            <div class="flex justify-between items-center text-red-400">
              <span>3.096 km</span><span>Python: Spark of Decency</span><span class="px-1 bg-red-950 border border-red-500 text-[7.5px]">[HOSTILE]</span>
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
            <div class="flex justify-between items-center text-slate-400">
              <span>828.6 km</span><span>Witchpoint Beacon 110°</span><span>[SAUT]</span>
            </div>
          </div>
        </div>

        <div class="bg-[#070e1a]/90 border border-cyan-500/40 rounded p-1.5 flex flex-col gap-0.5 flex-1 text-[9.5px] shadow-lg backdrop-blur-md">
          <div class="flex items-center justify-between border-b border-cyan-500/30 pb-0.5">
            <span class="text-tacticalCyan font-bold font-display">CORPS DU SYSTÈME STELLAIRE</span>
            <span class="text-amber-300 text-[8.5px]">SPECTRE: M2.5V / M4.5V</span>
          </div>
          <div class="space-y-0.5 text-[9px] text-slate-300">
            <div class="p-1 rounded bg-cyan-950/60 border border-tacticalCyan flex justify-between items-center">
              <div>
                <div class="font-bold text-tacticalCyan" id="oolite-sys-main-body">1.00 UA • LHS 1140 b [MAIN]</div>
                <div class="text-[8px] text-slate-300">Monde Habitable • Bio: H₂O/CH₄/DMS</div>
              </div>
              <span class="bg-tacticalCyan text-black px-1 rounded font-bold text-[8px]">LOCK</span>
            </div>
            <div class="p-0.5 px-1 rounded bg-black/40 border border-slate-800 flex justify-between items-center">
              <span class="text-slate-200">0.63 UA • Miniterra Inner</span>
              <span class="text-slate-500 text-[8px]">ORBITE INT.</span>
            </div>
            <div class="p-0.5 px-1 rounded bg-black/40 border border-slate-800 flex justify-between items-center">
              <span class="text-slate-200">4.31 UA • Sous-Neptune Externe</span>
              <span class="text-slate-500 text-[8px]">PÉRIPHÉRIE</span>
            </div>
          </div>
          <div class="mt-auto p-1 rounded bg-amber-950/40 border border-amber-500/40 text-[8px] text-amber-200 leading-tight">
            <span class="font-bold text-tacticalAmber">ASTUCE VOL :</span> Pressez <kbd class="bg-black border border-amber-400 px-1 rounded text-tacticalAmber font-bold">[Z]</kbd> ou cliquez sur <strong>RÉTROS ACTIFS</strong> pour freiner sous 95 m/s.
          </div>
        </div>
      </div>
    </div>

    <!-- BOTTOM ROW: Classic Oolite/Elite Dials & 3D Wireframes (8 Circular Digital Gauges + Compass Globe + Cobra MK-III Wireframe) -->
    <div class="relative z-10 grid grid-cols-12 gap-2 items-center mt-1">
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


def build_stitch_v46_cockpit_html(state_json: str) -> str:
    """Build the upgraded USS Hivemind NCC-31337 v4.6 Cockpit HTML from Stitch project 10018415280079892710."""
    v2_file = (
        PROJECT_ROOT
        / "steam_dist"
        / "HiveMind_Astrophysics_Exoplanet_Hunter"
        / "stitch_sync_v2"
        / "02_uss_hivemind_ncc-31337_v4_6_-_complete_tactical.html"
    )
    raw = v2_file.read_text(encoding="utf-8")

    # 1. Replace expiring external Google AIDA URLs with local /stitch-assets/ PNGs
    raw = raw.replace(AIDA_GALAXY_URL, LOCAL_GALAXY_ASSET)
    raw = raw.replace(AIDA_K218B_URL, LOCAL_K218B_ASSET)
    raw = raw.replace(AIDA_LHS1140B_URL, LOCAL_LHS1140B_ASSET)
    raw = raw.replace(AIDA_OOLITE_COCKPIT_URL, LOCAL_OOLITE_COCKPIT_ASSET)

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

    # 3. Add 5-Language Switcher (FR / ES / DE / EN / ZH) right next to the VOIX button
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

    # 4. Upgrade Stage 3 inside the central viewport with the Full Oolite Planetfall Landing Bridge from Screen 10
    raw = raw.replace(
        'class="relative w-full h-[400px] xl:h-[440px] rounded border border-cyan-600/50 overflow-hidden bg-black shadow-2xl"',
        'class="relative w-full h-[465px] xl:h-[485px] rounded border border-cyan-600/50 overflow-hidden bg-black shadow-2xl"',
    )
    raw = raw.replace(
        'class="absolute inset-0 bg-[#040c1b] hidden transition-opacity duration-300 p-4" id="viewport-stage-2"',
        'class="absolute inset-0 bg-[#040c1b] hidden transition-opacity duration-300 px-4 pt-3 pb-12" id="viewport-stage-2"',
    )
    raw = raw.replace(
        'class="absolute inset-0 bg-[#030914] hidden transition-opacity duration-300 p-3" id="viewport-stage-4"',
        'class="absolute inset-0 bg-[#030914] hidden transition-opacity duration-300 px-3 pt-2.5 pb-12" id="viewport-stage-4"',
    )
    start_s3 = raw.find("<!-- STAGE 3 VIEW: OOLITE PLANETFALL DESCENT COCKPIT HUD -->")
    end_s3 = raw.find("<!-- STAGE 4 VIEW: JWST NIRSPEC SPECTROSCOPY & TAP ADQL QUERY -->")
    if start_s3 != -1 and end_s3 != -1:
        raw = raw[:start_s3] + STAGE_3_FULL_OOLITE_BRIDGE_HTML + "\n" + raw[end_s3:]

    # 5. Add Approve / Reject discovery buttons & Live NASA TAP button to Stage 4 ADQL terminal
    old_adql_btns = """<div class="flex space-x-2">
<button class="bg-cyan-950 hover:bg-cyan-800 border border-cyan-500 text-tacticalCyan px-2.5 py-1 rounded text-[11px] font-bold transition cursor-pointer" id="btn-run-adql">
              EXÉCUTER ADQL
            </button>
<button class="bg-emerald-950 hover:bg-emerald-800 border border-emerald-500 text-tacticalGreen px-2.5 py-1 rounded text-[11px] font-bold transition cursor-pointer" id="btn-verify-kepler">
              LOI DE KEPLER T²/a³
            </button>
</div>"""

    new_adql_btns = """<div class="flex flex-wrap gap-1.5">
<button class="bg-cyan-950 hover:bg-cyan-800 border border-cyan-500 text-tacticalCyan px-2.5 py-1 rounded text-[11px] font-bold transition cursor-pointer" id="btn-run-adql">
              🌐 EXÉCUTER NASA TAP ADQL
            </button>
<button class="bg-emerald-950 hover:bg-emerald-800 border border-emerald-500 text-tacticalGreen px-2.5 py-1 rounded text-[11px] font-bold transition cursor-pointer" id="btn-verify-kepler">
              📐 LOI DE KEPLER T²/a³
            </button>
<button onclick="submitStitchDiscovery(true)" class="bg-emerald-500 hover:bg-emerald-400 text-black px-2.5 py-1 rounded text-[11px] font-extrabold transition cursor-pointer" id="btn-approve">
              ✅ APPROUVER (+600 HMT)
            </button>
<button onclick="submitStitchDiscovery(false)" class="bg-red-600 hover:bg-red-500 text-white px-2.5 py-1 rounded text-[11px] font-extrabold transition cursor-pointer" id="btn-reject">
              🚨 REJETER FAUX POSITIF
            </button>
</div>"""
    raw = raw.replace(old_adql_btns, new_adql_btns)

    # 6. Inject Steam registration/login + BOINC GPU contribution + 5-Language localization + Oolite gauges sync before </script>
    extra_js = f"""
  // === HIVEMIND LIVE BACKEND, STEAM + BOINC RTX 2070, & 5-LANGUAGE INTEGRATION ===
  window.HIVEMIND_EMBEDDED_STATE = {state_json};
  let scoreHmt = 2400;
  let currentLang = 'fr';

  const STITCH_I18N = {{
    fr: {{
      s1: "ÉTAPE 1 // PILOTEZ L'USS HIVEMIND (ZQSD / FLÈCHES / CLIC) VERS UNE ÉTOILE PUIS [ESPACE / CLIC] POUR DÉPLOYER LE SWARM",
      s2: "ÉTAPE 2 // DÉPLOIEMENT DU SWARM BEE EN COURS // SYNCHRONISATION LAGRANGE L1-L5 & ÉJECTION DE 128 MICROSATELLITES",
      s3: "ÉTAPE 3 // ATTERRISSAGE OOLITE PLANETFALL // CONTRÔLE DES RÉTRO-FUSÉES [Z/HAUT] POUR FREINER SOUS 95 m/s",
      s4: "ÉTAPE 4 // SPECTROGRAMME NIRSPEC JWST // VÉRIFICATION DE LA 3ÈME LOI DE KEPLER (T²/a³ = k) & DÉTECTION BIOSIGNATURE",
      dock: "COLLECTION ASTROBIOLOGIQUE & EXOPLANÈTES APPROUVÉES (CLIQUEZ POUR INSPECTION IMMÉDIATE) :"
    }},
    es: {{
      s1: "ETAPA 1 // PILOTA EL USS HIVEMIND (WASD / FLECHAS / CLIC) HACIA UNA ESTRELLA Y PULSA [ESPACIO] PARA DESPLEGAR EL ENJAMBRE BEE-MOE",
      s2: "ETAPA 2 // DESPLIEGUE DEL ENJAMBRE BEE-MOE // SINCRONIZACIÓN LAGRANGE L1-L5 Y 128 MICROSATÉLITES ACTIVOS",
      s3: "ETAPA 3 // ATERRIZAJE OOLITE PLANETFALL // ACTIVA RETROCOHETES [Z/ARRIBA] PARA FRENAR BAJO 95 m/s",
      s4: "ETAPA 4 // ESPECTRÓGRAFO JWST NIRSPEC // VERIFICACIÓN DE LA 3ª LEY DE KEPLER (T²/a³ = k) Y BIOFIRMAS",
      dock: "COLECCIÓN ASTROBIOLÓGICA Y EXOPLANETAS APROBADOS (HAZ CLIC PARA INSPECCIÓN INMEDIATA) :"
    }},
    de: {{
      s1: "STUFE 1 // STEUERE DIE USS HIVEMIND (WASD / KLICK) ZU EINEM STERN & DRÜCKE [LEERTASTE] FÜR BEE-MOE SCHWARM-START",
      s2: "STUFE 2 // BEE-MOE SCHWARM IM EINSATZ // LAGRANGE-SYNCHRONISATION L1-L5 & 128 AKTIVE MIKROSATELLITEN",
      s3: "STUFE 3 // OOLITE PLANETFALL LANDUNG // BREMSRAKETEN [Z/OBEN] ZÜNDEN UNTER 95 m/s",
      s4: "STUFE 4 // JWST NIRSPEC SPEKTROSKOPIE // 3. KEPLERSCHES GESETZ (T²/a³ = k) & BIOSIGNATUR-PRÜFUNG",
      dock: "ASTROBIOLOGISCHE SAMMLUNG & BESTÄTIGTE EXOPLANETEN (KLICKEN ZUR SOFORT-INSPEKTION) :"
    }},
    en: {{
      s1: "STAGE 1 // PILOT USS HIVEMIND (WASD / ARROWS / CLICK) TO A STAR THEN PRESS [SPACE / CLICK] TO DEPLOY THE BEE-MOE SWARM",
      s2: "STAGE 2 // BEE-MOE SWARM DEPLOYMENT // LAGRANGE L1-L5 PHASE LOCK & 128 ACTIVE MICROSATELLITES",
      s3: "STAGE 3 // OOLITE PLANETFALL DESCENT // FIRE RETRO-THRUSTERS [Z/UP] TO BRAKE BELOW 95 m/s",
      s4: "STAGE 4 // JWST NIRSPEC SPECTROSCOPY // KEPLER'S 3RD LAW (T²/a³ = k) & BIOSIGNATURE VERIFICATION",
      dock: "ASTROBIOLOGICAL COLLECTION & APPROVED EXOPLANETS (CLICK ANY WORLD FOR IMMEDIATE INSPECTION) :"
    }},
    zh: {{
      s1: "第 1 阶段 // 驾驶蜂群号星舰（WASD / 方向键 / 点击）前往恒星，按 [空格键] 部署 Bee-MoE 探测器蜂群",
      s2: "第 2 阶段 // Bee-MoE 轨道蜂群部署中 // 拉格朗日 L1-L5 同步锁定与 128 颗微卫星遥测",
      s3: "第 3 阶段 // Oolite 行星大气层降落 // 点燃反推火箭 [Z/上键] 将速度控制在 95 米/秒以下",
      s4: "第 4 阶段 // 韦伯 NIRSpec 透射光谱 // 验证开普勒第三定律 (T²/a³ = k) 与生物特征分子",
      dock: "天体生物学馆藏与已验证系外行星（点击行星卡片立即巡检）："
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
    const instr = document.getElementById('instruction-text');
    if (instr) instr.textContent = stageDescriptions[currentStage] || d.s1;
    showToast('LANGUE ACTIVE : ' + currentLang.toUpperCase());
  }};

  // Synchronize Oolite Stage 3 gauges and labels whenever planet or retro-burn changes
  const _origSelectPlanet = selectPlanet;
  selectPlanet = function(planetKey) {{
    _origSelectPlanet(planetKey);
    const data = PLANET_DATA[planetKey];
    if (!data) return;
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
  }};

  const _origFireRetroBurn = fireRetroBurn;
  fireRetroBurn = function() {{
    _origFireRetroBurn();
    const gs = document.getElementById('gauge-speed');
    if (gs) gs.textContent = Math.round(ooliteSpeed);
    const ga = document.getElementById('gauge-alt');
    if (ga) ga.textContent = (ooliteAltitude >= 1000) ? `${{(ooliteAltitude / 1000).toFixed(1)}}k` : `${{Math.round(ooliteAltitude)}}m`;
    const vd = document.getElementById('oolite-voyage-dist');
    if (vd) vd.textContent = `${{Math.round(ooliteAltitude).toLocaleString()}} m`;
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

  window.switchStage = switchStage;
  window.selectPlanet = selectPlanet;
  window.fireRetroBurn = fireRetroBurn;
  simulateSteamRegisterLoginAndContribute();
"""
    last_script_idx = raw.rfind("</script>")
    if last_script_idx != -1:
        raw = raw[:last_script_idx] + extra_js + "\n</script>" + raw[last_script_idx + len("</script>") :]
    return raw
