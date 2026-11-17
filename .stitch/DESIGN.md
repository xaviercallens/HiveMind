# USS HIVEMIND NCC-31337 v5.1 — Google Stitch Design System (`DESIGN.md`)

- **Stitch Project ID:** [`10018415280079892710`](https://stitch.withgoogle.com/projects/10018415280079892710)
- **Vibe & Purpose:** High-fidelity *Star Trek LCARS* meets *Elite Dangerous* tactical starship bridge console (`USS HIVEMIND NCC-31337`) for real-time astrophysical exoplanet discovery, Oolite planetary landing, JWST spectroscopy, and BOINC/GCP citizen-science GPU sharing.
- **Platform:** Web & Native Desktop Widescreen Cockpit Layout (`16:9` viewport lock, zero page scroll, `pointer-events-none` HUD overlay containers with `pointer-events-auto` tactical controls over full-bleed `<SimulationViewport/>`).

---

## 1. Calibrated Color Tokens

| Token | Hex / RGBA | Role in USS HiveMind HUD |
| :--- | :--- | :--- |
| `--hm-bg-obsidian` | `#030914` | Deep Space Obsidian primary viewport background |
| `--hm-mfd-glass` | `rgba(4, 14, 28, 0.82)` | Dark glassmorphism MFD backing (`backdrop-filter: blur(10px)`) preventing planetary glare bleed |
| `--hm-cyan-tactical` | `#00F0FF` | Primary telemetry text, active vectors, chamfered MFD 1px glowing borders, Kepler/TESS light curves |
| `--hm-emerald-bio` | `#00FF88` | Habitable zone indicators, approved exoplanet badges, UCSC Doppler radial velocity sine/Keplerian wave, glide-slope lock |
| `--hm-amber-solar` | `#FFB800` | Active target selection, `VOIX` speech synthesis controls, lateral wind-shear drift vectors |
| `--hm-red-plasma` | `#FF2A2A` | CME debris alerts, `DANGER! ALTITUDE TOO LOW`, `[Z/UP] FIRE RETRO-THRUSTERS`, re-entry plasma vignette |
| `--hm-steel-passive` | `#5C88A6` | Passive logs (`COMMS LOG`, historical radar traces, inactive tabs) |

---

## 2. Sci-Fi Typography & Surface Geometry

- **Display / Stage Headers (`--font-display`):** `'Orbitron', 'Space Grotesk', sans-serif` — uppercase, `letter-spacing: 0.08em`, `font-weight: 700`.
- **Telemetry & Tabular Numerals (`--font-mono`):** `'Share Tech Mono', 'JetBrains Mono', monospace` — `font-variant-numeric: tabular-nums` for Radial Velocity $K$ ($\text{m/s}$), Transit Depth ($\text{ppm}$), Altitude ($\text{km}$), and BOINC Cobblestones.
- **Chamfered MFD Geometry:**
  ```css
  .mfd-panel {
    background: linear-gradient(135deg, rgba(4, 14, 28, 0.86) 0%, rgba(2, 8, 18, 0.78) 100%);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(0, 240, 255, 0.38);
    box-shadow: inset 0 0 15px rgba(0, 240, 255, 0.12), 0 4px 20px rgba(0, 0, 0, 0.65);
    clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 12px 100%, 0 calc(100% - 12px));
    color: #d8f8ff;
    font-family: 'Share Tech Mono', 'JetBrains Mono', monospace;
  }
  ```

---

## 3. Decoupled Layout Zones (`16:9` Viewport Lock)

1. **Top Telemetry & Stage Switcher Bar (`6%` height):**
   - Compact horizontal LCARS header displaying live system metrics (`FPS | HMT | BOINC IDLE`), 5-language toggles (`FR` / `ES` / `DE` / `EN` / `ZH`), voice synthesis toggle (`VOIX`), and the stage selector pills (`1.GALAXIE`, `2.ORBIT BEE`, `3.OOLITE LAND`, `4.JWST/LOIS`, `5.BOINC/GCP`).
2. **Central Interactive `<SimulationViewport/>` (`54%` height):**
   - Full-bleed WebGL/Canvas simulation container overlaid with floating glassmorphic HUD reticles (`pointer-events: none`), a bottom-pinned narrative comm-link bar, and contextual flight/astrobiology alerts:
     - **Stage 1 (`1.GALAXIE`):** 3D Milky Way sector scan, relativistic warp tunnel, and 5-expert Astro-Bee MoE routing (`BEE-STARS`, `BEE-PLANETS`, `BEE-GAIA`, `BEE-JWST`, `BEE-LOIS`).
     - **Stage 2 (`2.ORBIT BEE`):** Lagrange L2 Honeycomb Orbital Ring (`BEE-MOE`) with CME solar flare shielding.
     - **Stage 3 (`3.OOLITE LAND`):** Curved Hycean horizon shader (Rayleigh/Mie cyan-blue scattering), dynamic re-entry plasma vignette tied to `C.TEMP` & `SPEED`, 3D perspective glide-slope tunnel, pitch ladder with live wind-shear & pad-offset drift arrows, and collapsible `[T] TRADE & EQUIP` dark-glassmorphic tabs.
     - **Stage 4 (`4.JWST/LOIS`):** JWST NIRSpec/MIRI transmission spectroscopy ($\text{CH}_4$, $\text{CO}_2$, $\text{H}_2\text{O}$, DMS) and Kepler/Newton/Einstein orbital laws.
     - **Stage 5 (`5.BOINC/GCP`):** Idle-GPU BOINC Box-Least-Squares (BLS) transit periodogram folding (`0%` GPU overhead while gaming, automatic resume after `4.0s` idle) synced to GCP Cloud Run.
3. **Dual Scientific Telemetry Console (`22%` height):**
   - Split 50/50 horizontal oscilloscope dock bound to `exoplanetCatalog.ts`:
     - **Left (`CONSOLE SYSTÉMIQUE UCSC // VITESSE RADIALE DOPPLER`):** Glowing emerald (`#00FF88`) Keplerian radial velocity curve ($K$ in $\text{m/s}$, eccentricity $e$) over a sub-pixel coordinate grid.
     - **Right (`PHOTOMÉTRIE DE TRANSIT KEPLER/TESS`):** Glowing cyan (`#00F0FF`) Mandel-Agol limb-darkened transit light curve dip in $\text{ppm}$.
4. **Astrobiological Collection Dock (`18%` height):**
   - 10-slot horizontal grid (`TRAPPIST-1 e`, `K2-18 b`, `Kepler-11 f`, `HD 80606 b`, `TOI-700 d`, `LHS 1140 b`, `WASP-39 b`, `Kepler-186 f`, `KOI-Bin #01`, `KOI-Bin #02`) with 3D holographic planetary spheres, orbital projection rings, and approval status borders.
