// Decoupled Layered HUD Architecture for USS HIVEMIND NCC-31337 v5.1
// Follows .stitch/DESIGN.md (Stitch Project 10018415280079892710) & exoplanetCatalog.ts (NASA Exoplanet Archive MCP)

import React, { useEffect, useRef, useState } from "react";
import { EXOPLANET_CATALOG, ExoplanetTelemetry } from "./exoplanetCatalog";

export type OperationalStage = 0 | 1 | 2 | 3 | 4; // 1.GALAXIE, 2.ORBIT BEE, 3.OOLITE LAND, 4.JWST/LOIS, 5.BOINC/GCP
export type BridgeLanguage = "FR" | "ES" | "DE" | "EN" | "ZH";

export interface SimulationViewportProps {
  activeStage: OperationalStage;
  activePlanet: ExoplanetTelemetry;
  cabinTempC: number;
  descentSpeedMs: number;
  padOffsetMeters: number;
  windShearMs: number;
}

/**
 * Full-screen 3D WebGL / Canvas Simulation Layer (<SimulationViewport/>)
 * Decoupled from the React/Tailwind Glassmorphic HUD overlays.
 */
export const SimulationViewport: React.FC<SimulationViewportProps> = ({
  activeStage,
  activePlanet,
  cabinTempC,
  descentSpeedMs,
  padOffsetMeters,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId = 0;
    const render = (tMs: number) => {
      const t = tMs * 0.001;
      const W = canvas.width;
      const H = canvas.height;

      // Deep Space Obsidian (#030914) base from .stitch/DESIGN.md
      ctx.fillStyle = "#030914";
      ctx.fillRect(0, 0, W, H);

      if (activeStage === 2) {
        // Stage 3 (3.OOLITE LAND): Curved Hycean Horizon + Rayleigh/Mie Scattering + 3D Glide-Slope Tunnel
        const horizY = H * 0.38;
        const mieGrad = ctx.createLinearGradient(0, horizY - 48, 0, horizY + 24);
        mieGrad.addColorStop(0, "rgba(0, 240, 255, 0)");
        mieGrad.addColorStop(0.6, "rgba(0, 240, 255, 0.55)");
        mieGrad.addColorStop(1, "rgba(2, 132, 199, 0.92)");
        ctx.fillStyle = mieGrad;
        ctx.beginPath();
        ctx.moveTo(0, horizY + 28);
        ctx.quadraticCurveTo(W * 0.5, horizY - 36, W, horizY + 28);
        ctx.lineTo(W, H);
        ctx.lineTo(0, H);
        ctx.closePath();
        ctx.fill();

        // Dynamic Re-Entry Plasma Vignette when SPEED > 95 m/s or C.TEMP approaches 100°C
        if (descentSpeedMs > 95 || cabinTempC > 85) {
          const pulse = 0.35 + 0.25 * Math.sin(t * 6.0);
          ctx.strokeStyle = `rgba(255, 42, 42, ${pulse.toFixed(2)})`;
          ctx.lineWidth = 18;
          ctx.strokeRect(0, 0, W, H);
        }

        // 3D Perspective Glide-Slope Landing Tunnel
        const cx = W * 0.5;
        const cy = H * 0.48;
        const padX = cx - padOffsetMeters * 1.2;
        const padY = horizY + 24;
        for (let i = 0; i < 5; i++) {
          const frac = ((i / 5) + (t * 0.35) % 0.2);
          const rx = cx + (padX - cx) * frac;
          const ry = cy + (padY - cy) * frac;
          const rw = W * 0.28 * (1 - frac * 0.75);
          const rh = H * 0.24 * (1 - frac * 0.75);
          ctx.strokeStyle = Math.abs(padOffsetMeters) < 35 ? "#00FF88" : "#FFB800";
          ctx.lineWidth = i === 0 ? 2 : 1;
          ctx.strokeRect(rx - rw * 0.5, ry - rh * 0.5, rw, rh);
        }
      }

      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animId);
  }, [activeStage, activePlanet, cabinTempC, descentSpeedMs, padOffsetMeters]);

  return (
    <canvas
      ref={canvasRef}
      width={1920}
      height={1080}
      className="fixed inset-0 w-full h-full object-cover z-0"
    />
  );
};

/**
 * Decoupled Tactical Bridge HUD Overlay following .stitch/DESIGN.md:
 * - pointer-events-none on layout containers so 3D canvas drag/orbit passes through
 * - pointer-events-auto on interactive LCARS buttons, stage pills, MFD tabs, and planet cards
 */
export const USSHivemindTacticalBridgeHUD: React.FC = () => {
  const [activeStage, setActiveStage] = useState<OperationalStage>(2);
  const [selectedPlanet, setSelectedPlanet] = useState<ExoplanetTelemetry>(
    EXOPLANET_CATALOG.find((p) => p.planet === "K2-18 b") || EXOPLANET_CATALOG[0]
  );
  const [lang, setLang] = useState<BridgeLanguage>("FR");
  const [voiceEnabled, setVoiceEnabled] = useState<boolean>(true);
  const [tradeTabExpanded, setTradeTabExpanded] = useState<boolean>(false);

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-[#030914] text-[#d8f8ff] font-mono select-none">
      {/* 1. Background 3D/Canvas Simulation Viewport */}
      <SimulationViewport
        activeStage={activeStage}
        activePlanet={selectedPlanet}
        cabinTempC={96}
        descentSpeedMs={137}
        padOffsetMeters={29.1}
        windShearMs={21.4}
      />

      {/* 2. Decoupled HUD Layer (pointer-events-none container, pointer-events-auto controls) */}
      <div className="relative z-10 flex flex-col justify-between w-full h-full pointer-events-none p-2 gap-2">
        {/* ZONE 1: Top Telemetry & Stage Switcher Bar */}
        <header className="pointer-events-auto flex items-center justify-between px-4 py-2 bg-[#040e1c]/85 backdrop-blur-md border border-[#00F0FF]/40">
          <div className="flex items-center gap-3">
            <span className="font-bold tracking-widest text-[#00F0FF]">
              USS HIVEMIND NCC-31337 // v5.1 STITCH-MCP
            </span>
            <span className="text-xs text-[#00FF88]">
              TARGET: {selectedPlanet.planet} ({selectedPlanet.distance_pc} pc)
            </span>
          </div>

          <div className="flex items-center gap-1">
            {(["1.GALAXIE", "2.ORBIT BEE", "3.OOLITE LAND", "4.JWST/LOIS", "5.BOINC/GCP"] as const).map(
              (label, idx) => (
                <button
                  key={label}
                  onClick={() => setActiveStage(idx as OperationalStage)}
                  className={`px-3 py-1 text-xs font-bold border transition ${
                    activeStage === idx
                      ? "bg-[#00FF88]/20 border-[#00FF88] text-[#00FF88]"
                      : "bg-[#040e1c]/80 border-[#00F0FF]/35 text-[#00F0FF]"
                  }`}
                >
                  {label}
                </button>
              )
            )}
          </div>

          <div className="flex items-center gap-1">
            {(["FR", "ES", "DE", "EN", "ZH"] as const).map((code) => (
              <button
                key={code}
                onClick={() => setLang(code)}
                className={`px-2 py-0.5 text-xs border ${
                  lang === code
                    ? "border-[#FFB800] text-[#FFB800] font-bold"
                    : "border-[#5C88A6]/40 text-[#5C88A6]"
                }`}
              >
                {code}
              </button>
            ))}
            <button
              onClick={() => setVoiceEnabled(!voiceEnabled)}
              className="ml-2 px-2.5 py-0.5 text-xs border border-[#FFB800] text-[#FFB800]"
            >
              VOIX: {voiceEnabled ? "ON" : "MUTE"}
            </button>
          </div>
        </header>

        {/* ZONE 2: Central Interactive Viewport HUD Overlays (60% height) */}
        <main className="flex-1 flex justify-between items-start px-2 pointer-events-none">
          {/* Collapsible Dark-Glassmorphic Trade & Equipment MFD Tab in Stage 3 */}
          {activeStage === 2 && (
            <aside className="pointer-events-auto max-w-xs p-3 bg-[#040e1c]/85 backdrop-blur-md border border-[#00F0FF]/40 shadow-[inset_0_0_15px_rgba(0,240,255,0.12)]">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold tracking-wider text-[#00F0FF]">
                  STAGE 3 // FLIGHT MODE HUD
                </span>
                <button
                  onClick={() => setTradeTabExpanded(!tradeTabExpanded)}
                  className="px-2 py-0.5 text-[11px] border border-[#FFB800] text-[#FFB800]"
                >
                  {tradeTabExpanded ? "HIDE TRADE [T]" : "EXPAND TRADE [T]"}
                </button>
              </div>
              <p className="text-xs text-[#00FF88]">
                Glide-Slope Lock: +29.1 m (&lt; 35 m) | Wind Shear: 21.4 m/s
              </p>
              {tradeTabExpanded && (
                <div className="mt-2 pt-2 border-t border-[#00F0FF]/25 text-xs text-[#5C88A6]">
                  <div>Quirium Crystal: 26.8 HMT/kg</div>
                  <div>Gem-stones: 16.0 HMT/g</div>
                  <div>Platinum: 65.2 HMT/g</div>
                </div>
              )}
            </aside>
          )}
        </main>

        {/* ZONE 3: Dual Scientific Telemetry Console (22% height) */}
        <section className="pointer-events-auto grid grid-cols-2 gap-2 h-[22%] min-h-[150px]">
          <div className="p-3 bg-[#040e1c]/85 backdrop-blur-md border border-[#00F0FF]/40 flex flex-col justify-between">
            <div className="flex justify-between text-xs">
              <span className="font-bold text-[#00FF88]">
                CONSOLE SYSTÉMIQUE UCSC // VITESSE RADIALE DOPPLER
              </span>
              <span className="text-[#00F0FF]">
                K = {selectedPlanet.rv_semi_amplitude_k_ms} m/s | e = {selectedPlanet.eccentricity} | P ={" "}
                {selectedPlanet.period_days} d
              </span>
            </div>
            <div className="text-xs text-[#5C88A6]">
              Keplerian RV Curve bound via NASA Exoplanet Archive MCP ({selectedPlanet.source})
            </div>
          </div>

          <div className="p-3 bg-[#040e1c]/85 backdrop-blur-md border border-[#00F0FF]/40 flex flex-col justify-between">
            <div className="flex justify-between text-xs">
              <span className="font-bold text-[#00F0FF]">
                PHOTOMÉTRIE DE TRANSIT KEPLER/TESS (MANDEL-AGOL)
              </span>
              <span className="text-[#00FF88]">
                Depth = {selectedPlanet.transit_depth_ppm} ppm | R_p = {selectedPlanet.radius_earth} R_Earth | T_eq ={" "}
                {selectedPlanet.eq_temp_k} K
              </span>
            </div>
            <div className="text-xs text-[#5C88A6]">
              Sub-pixel limb-darkened transit dip (1 - {selectedPlanet.transit_depth_ppm} ppm)
            </div>
          </div>
        </section>

        {/* ZONE 4: Astrobiological Collection Dock (18% height) */}
        <footer className="pointer-events-auto grid grid-cols-8 gap-2 h-[16%] min-h-[105px]">
          {EXOPLANET_CATALOG.map((item) => {
            const isSelected = item.planet === selectedPlanet.planet;
            return (
              <button
                key={item.planet}
                onClick={() => setSelectedPlanet(item)}
                className={`p-2 text-left bg-[#040e1c]/85 backdrop-blur-md border transition flex flex-col justify-between ${
                  isSelected
                    ? "border-[#FFB800] shadow-[inset_0_0_12px_rgba(255,184,0,0.25)]"
                    : item.habitable_zone
                    ? "border-[#00FF88]/50"
                    : "border-[#00F0FF]/30"
                }`}
              >
                <div className="text-xs font-bold text-[#00F0FF] truncate">{item.planet}</div>
                <div className="text-[11px] text-[#00FF88]">
                  K={item.rv_semi_amplitude_k_ms} m/s · {item.transit_depth_ppm} ppm
                </div>
                <div className="text-[10px] text-[#5C88A6]">
                  P={item.period_days}d · e={item.eccentricity}
                </div>
              </button>
            );
          })}
        </footer>
      </div>
    </div>
  );
};
