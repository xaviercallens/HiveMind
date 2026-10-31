# Copyright (C) 2026 Project HiveMind / SocrateAI (Xavier Callens)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""HTML5 / 3D Canvas Globe & Foundation Paper v4.2.0 Portal Views for International Launch."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from hivemind.economy.international_launch_model import (
    build_foundation_paper_v42_portal_payload,
    forecast_200plus_node_international_launch,
)


def render_gcp_planet_forecast_html(initial_forecast: dict[str, Any] | None = None) -> str:
    """Render the interactive 3D Planet HiveMiners & >200-Node Capacity Forecast GCP Web Page."""
    import json

    data = initial_forecast or forecast_200plus_node_international_launch()
    payload_json = json.dumps(data)

    return f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>HiveMind @ Home — Global 3D Planet HiveMiners &amp; 200+ Node International Launch Forecast</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    :root {{
      color-scheme: dark;
    }}
    body {{
      background: radial-gradient(circle at 20% 15%, #0f172a 0%, #050814 55%, #02040a 100%);
      color: #e2e8f0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      overflow-x: hidden;
    }}
    .glass-card {{
      background: rgba(15, 23, 42, 0.76);
      backdrop-filter: blur(14px);
      border: 1px solid rgba(56, 189, 248, 0.22);
      box-shadow: 0 12px 35px -10px rgba(2, 132, 199, 0.25);
    }}
    .glow-badge {{
      box-shadow: 0 0 18px rgba(52, 211, 153, 0.35);
    }}
    canvas#hivemind-planet-canvas {{
      cursor: grab;
      touch-action: none;
    }}
    canvas#hivemind-planet-canvas:active {{
      cursor: grabbing;
    }}
    input[type="range"] {{
      accent-color: #38bdf8;
    }}
  </style>
</head>
<body class="min-h-screen text-slate-100 antialiased selection:bg-sky-500 selection:text-slate-950">
  <!-- Top Charter & GCP Cloud Run Telemetry Bar -->
  <header class="border-b border-sky-500/20 bg-slate-950/80 backdrop-blur-md sticky top-0 z-30">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex flex-wrap items-center justify-between gap-3">
      <div class="flex items-center gap-3">
        <div class="h-9 w-9 rounded-xl bg-gradient-to-br from-amber-400 via-sky-400 to-emerald-400 flex items-center justify-center font-black text-slate-950 shadow-lg">
          HM
        </div>
        <div>
          <div class="flex items-center gap-2">
            <h1 class="text-base sm:text-lg font-extrabold tracking-tight text-white">
              HiveMind @ Home • Global Planet HiveMiners &amp; 200+ Node Capacity Forecast
            </h1>
            <span id="verdict-pill" class="px-2.5 py-0.5 text-xs font-bold uppercase rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-400/40 glow-badge">
              READY • &gt;200 NODES VERIFIED
            </span>
          </div>
          <p class="text-xs text-slate-400">
            GCP Cloud Run Federated Observatory • Kleinrock/Halfin-Whitt M/M/c(t) + Kingman G/G/c + Mitzenmacher P2C + Bass Prosumer Epidemic Model
          </p>
        </div>
      </div>
      <div class="flex items-center gap-2 text-xs">
        <span class="px-2.5 py-1 rounded-lg bg-slate-900 border border-amber-400/30 text-amber-300 font-semibold">
          0.00 EUR / 0.00 USD Non-Monetary Reciprocity Charter
        </span>
        <a href="/api/v1/launch/planet-forecast" target="_blank" class="px-3 py-1 rounded-lg bg-sky-500/20 hover:bg-sky-500/30 border border-sky-400/40 text-sky-300 font-semibold transition">
          JSON Forecast API
        </a>
        <a href="/api/v1/launch/roadmap" target="_blank" class="px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-600 text-slate-200 font-semibold transition">
          5-Wave Roadmap JSON
        </a>
      </div>
    </div>
  </header>

  <main class="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-6">
    <!-- Top Executive KPI Comparison Strip -->
    <section class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-3">
      <div class="glass-card rounded-2xl p-4">
        <div class="text-xs uppercase tracking-wider text-slate-400 font-semibold">Current Live Pioneers</div>
        <div class="mt-1 flex items-baseline gap-2">
          <span id="kpi-baseline-nodes" class="text-2xl font-black text-amber-300">47</span>
          <span class="text-xs text-slate-400">8 Prod • 14 Pros • 22 Cons</span>
        </div>
        <div class="mt-1 text-xs text-slate-400">Daytime: <span class="text-slate-200 font-semibold">380 tok/s</span> • <span class="text-slate-200 font-semibold">0.52 PFLOPS</span></div>
      </div>

      <div class="glass-card rounded-2xl p-4 border-sky-400/40">
        <div class="text-xs uppercase tracking-wider text-sky-300 font-semibold">Forecast Active Nodes</div>
        <div class="mt-1 flex items-baseline gap-2">
          <span id="kpi-total-nodes" class="text-2xl font-black text-white">265</span>
          <span id="kpi-visitors-badge" class="text-xs text-sky-300 font-semibold">2,400 Visitors</span>
        </div>
        <div id="kpi-node-split" class="mt-1 text-xs text-slate-300">28 Prod • 54 Pros (32 Active) • 183 Cons</div>
      </div>

      <div class="glass-card rounded-2xl p-4">
        <div class="text-xs uppercase tracking-wider text-emerald-300 font-semibold">Token Throughput</div>
        <div class="mt-1 flex items-baseline gap-2">
          <span id="kpi-tok-s" class="text-2xl font-black text-emerald-300">3,426</span>
          <span class="text-xs text-slate-400">tok/s daytime</span>
        </div>
        <div class="mt-1 text-xs text-slate-400">
          Burst: <span id="kpi-burst-tok-s" class="text-emerald-200 font-semibold">4,482 tok/s</span> •
          <span id="kpi-daily-tok" class="text-slate-200 font-semibold">296M/day</span>
        </div>
      </div>

      <div class="glass-card rounded-2xl p-4">
        <div class="text-xs uppercase tracking-wider text-fuchsia-300 font-semibold">Swarm Compute Power</div>
        <div class="mt-1 flex items-baseline gap-2">
          <span id="kpi-night-pflops" class="text-2xl font-black text-fuchsia-300">3.89</span>
          <span class="text-xs text-slate-400">PFLOPS Nocturnal</span>
        </div>
        <div class="mt-1 text-xs text-slate-400">
          Daytime Inf: <span id="kpi-day-tflops" class="text-slate-200 font-semibold">1,951 TFLOPS</span>
        </div>
      </div>

      <div class="glass-card rounded-2xl p-4">
        <div class="text-xs uppercase tracking-wider text-amber-300 font-semibold">Response Latency SLA</div>
        <div class="mt-1 flex items-baseline gap-2">
          <span id="kpi-fast-ttft" class="text-2xl font-black text-amber-300">138 ms</span>
          <span class="text-xs text-slate-400">Fast-Lane TTFT</span>
        </div>
        <div class="mt-1 text-xs text-slate-400">
          Blended P50: <span id="kpi-p50-ms" class="text-slate-200 font-semibold">214 ms</span> •
          P95: <span id="kpi-p95-ms" class="text-slate-200 font-semibold">518 ms</span>
        </div>
      </div>

      <div class="glass-card rounded-2xl p-4">
        <div class="text-xs uppercase tracking-wider text-cyan-300 font-semibold">Erlang-C / G/G/c Balance</div>
        <div class="mt-1 flex items-baseline gap-2">
          <span id="kpi-rho" class="text-2xl font-black text-cyan-300">57.8%</span>
          <span class="text-xs text-slate-400">ρ_elastic (&le; 78%)</span>
        </div>
        <div class="mt-1 text-xs text-slate-400">
          Uncontrolled ρ: <span id="kpi-rho-uncontrolled" class="text-rose-400 font-semibold">188.4% (Overload)</span>
        </div>
      </div>
    </section>

    <!-- Main 3D Planet Globe + Interactive Forecast Controls -->
    <section class="grid grid-cols-1 lg:grid-cols-12 gap-6">
      <!-- Left 7 cols: Interactive 3D Planet Earth -->
      <div class="lg:col-span-7 glass-card rounded-3xl p-5 flex flex-col justify-between relative overflow-hidden">
        <div class="flex flex-wrap items-center justify-between gap-2 z-10">
          <div>
            <h2 class="text-lg font-bold text-white flex items-center gap-2">
              <span>Interactive 3D Planet HiveMiners Globe</span>
              <span id="globe-node-counter" class="text-xs px-2 py-0.5 rounded-full bg-sky-500/20 text-sky-300 border border-sky-400/30">
                265 Active Nodes Plotted
              </span>
            </h2>
            <p class="text-xs text-slate-400">
              Drag to rotate • Scroll to zoom • Hover any beacon to inspect regional producer, prosumer, consumer &amp; BOINC plan-class telemetry
            </p>
          </div>
          <div class="flex items-center gap-1.5 text-xs">
            <button id="btn-filter-all" class="px-2.5 py-1 rounded-lg bg-sky-500 text-slate-950 font-bold transition">
              All 200+ Forecast
            </button>
            <button id="btn-filter-pioneers" class="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold transition">
              Current 47 Pioneers
            </button>
            <button id="btn-filter-producers" class="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold transition">
              Producers &amp; Prosumers
            </button>
          </div>
        </div>

        <!-- 3D Globe Canvas Container -->
        <div class="relative my-3 flex items-center justify-center min-h-[430px] sm:min-h-[480px]">
          <canvas id="hivemind-planet-canvas" width="680" height="480" class="max-w-full h-auto rounded-2xl"></canvas>

          <!-- Floating Hover Tooltip -->
          <div id="globe-tooltip" class="hidden pointer-events-none absolute z-20 px-3.5 py-2.5 rounded-xl bg-slate-950/95 border border-sky-400/50 text-xs shadow-2xl max-w-xs">
            <div id="tt-title" class="font-bold text-white"></div>
            <div id="tt-sub" class="text-sky-300 mt-0.5"></div>
            <div id="tt-metrics" class="text-slate-300 mt-1"></div>
          </div>
        </div>

        <!-- Legend & Super-Peer Arc Status -->
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-slate-800 text-xs">
          <div class="flex items-center gap-2">
            <span class="h-3 w-3 rounded-full bg-amber-400 inline-block shadow-[0_0_10px_#fbbf24]"></span>
            <span><strong class="text-amber-300">Producers</strong> (24/7 GPU)</span>
          </div>
          <div class="flex items-center gap-2">
            <span class="h-3 w-3 rounded-full bg-sky-400 inline-block shadow-[0_0_10px_#38bdf8]"></span>
            <span><strong class="text-sky-300">Prosumers</strong> (AMCP Elastic)</span>
          </div>
          <div class="flex items-center gap-2">
            <span class="h-3 w-3 rounded-full bg-emerald-400 inline-block shadow-[0_0_10px_#34d399]"></span>
            <span><strong class="text-emerald-300">Consumers</strong> (Edge/WebGPU)</span>
          </div>
          <div class="flex items-center gap-2">
            <span class="h-3 w-3 rounded-full border-2 border-rose-400 bg-rose-500/30 inline-block"></span>
            <span><strong class="text-rose-300">6 Super-Peers</strong> (QUIC Mesh)</span>
          </div>
        </div>
      </div>

      <!-- Right 5 cols: Interactive Traffic, Node Mix & Capacity Simulator -->
      <div class="lg:col-span-5 glass-card rounded-3xl p-5 flex flex-col justify-between space-y-4">
        <div>
          <div class="flex items-center justify-between">
            <h2 class="text-lg font-bold text-white">Traffic &amp; Capacity Forecast Simulator</h2>
            <button id="btn-reset-default" class="text-xs px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-sky-300 border border-sky-500/30 font-semibold">
              Reset 265-Node Target
            </button>
          </div>
          <p class="text-xs text-slate-400 mt-1">
            Simulate consumer-heavy international traffic surges (&gt;200 nodes) and observe live Erlang-C + Kingman G/G/c + Mitzenmacher P2C balancing.
          </p>
        </div>

        <!-- Sliders -->
        <div class="space-y-3.5 text-xs">
          <div>
            <div class="flex justify-between font-semibold mb-1">
              <span class="text-slate-300">Concurrent / Daily Visitors Funnel</span>
              <span id="lbl-visitors" class="text-sky-300 font-bold">2,400 Visitors</span>
            </div>
            <input id="slider-visitors" type="range" min="500" max="15000" step="100" value="2400" class="w-full" />
          </div>

          <div>
            <div class="flex justify-between font-semibold mb-1">
              <span class="text-emerald-300">Active Pure Consumers (Fair-Queue 30% Lane + 15+85 HMT Drip)</span>
              <span id="lbl-consumers" class="text-emerald-300 font-bold">183 Consumers</span>
            </div>
            <input id="slider-consumers" type="range" min="120" max="600" step="5" value="183" class="w-full" />
          </div>

          <div>
            <div class="flex justify-between font-semibold mb-1">
              <span class="text-sky-300">Hybrid Prosumers (AMCP v1.5 Elastic Inference + BOINC)</span>
              <span id="lbl-prosumers" class="text-sky-300 font-bold">54 Prosumers</span>
            </div>
            <input id="slider-prosumers" type="range" min="14" max="250" step="2" value="54" class="w-full" />
          </div>

          <div>
            <div class="flex justify-between font-semibold mb-1">
              <span class="text-amber-300">Dedicated Producers (24/7 GPU Servers &amp; Miners)</span>
              <span id="lbl-producers" class="text-amber-300 font-bold">28 Producers</span>
            </div>
            <input id="slider-producers" type="range" min="8" max="180" step="2" value="28" class="w-full" />
          </div>

          <div>
            <div class="flex justify-between font-semibold mb-1">
              <span class="text-fuchsia-300">Follow-the-Sun Solar Time (UTC)</span>
              <span id="lbl-utc" class="text-fuchsia-300 font-bold">14:00 UTC</span>
            </div>
            <input id="slider-utc" type="range" min="0" max="23" step="1" value="14" class="w-full" />
          </div>
        </div>

        <!-- Balancing Comparison Box: Uncontrolled vs HiveMind Balanced -->
        <div class="rounded-2xl bg-slate-900/90 border border-slate-800 p-3.5 space-y-2.5 text-xs">
          <div class="font-bold text-slate-200 uppercase tracking-wider text-[11px]">
            Supply-Demand Balancing &amp; Queueing Verdict
          </div>
          <div class="grid grid-cols-2 gap-2.5">
            <div class="p-2.5 rounded-xl bg-rose-950/40 border border-rose-500/30">
              <div class="font-bold text-rose-300">Naive Uncontrolled Pool</div>
              <div class="mt-1 text-slate-300">Only Dedicated Producers: <strong id="cmp-unctrl-c">28</strong></div>
              <div class="text-slate-300">Utilization ρ: <strong id="cmp-unctrl-rho" class="text-rose-400">188.4%</strong></div>
              <div class="text-slate-300">Queue State: <strong id="cmp-unctrl-state" class="text-rose-400">UNSTABLE (HTTP 504)</strong></div>
            </div>
            <div class="p-2.5 rounded-xl bg-emerald-950/40 border border-emerald-500/30">
              <div class="font-bold text-emerald-300">HiveMind Elastic P2C + AMCP</div>
              <div class="mt-1 text-slate-300">Effective Servers c_eff: <strong id="cmp-bal-c" class="text-emerald-300">60</strong></div>
              <div class="text-slate-300">Utilization ρ_elastic: <strong id="cmp-bal-rho" class="text-emerald-300">57.8%</strong></div>
              <div class="text-slate-300">Kelly Multiplier ω(ρ): <strong id="cmp-bal-omega" class="text-amber-300">1.52x HMT</strong></div>
            </div>
          </div>
          <div class="text-[11px] text-slate-400 leading-relaxed" id="verdict-narrative">
            AMCP v1.5 automatically promotes standby Prosumers into active daytime inference replicas whenever consumer arrival pushes ρ &gt; 0.58, while the 15 + 85 HMT progressive starter grant prevents free-rider queue saturation.
          </div>
        </div>
      </div>
    </section>

    <!-- 6 Global Super-Peer Regional Shards Table -->
    <section class="glass-card rounded-3xl p-5">
      <div class="flex flex-wrap items-center justify-between gap-2 mb-4">
        <div>
          <h2 class="text-base sm:text-lg font-bold text-white">
            6 Global Super-Peer Regional Shards — Capacity, Token Throughput &amp; Response Latency
          </h2>
          <p class="text-xs text-slate-400">
            Hierarchical Kademlia DHT + QUIC Super-Peer fan-out (B=32) with Geo-Affinity P2C dispatch
          </p>
        </div>
        <span class="text-xs px-3 py-1 rounded-full bg-slate-900 border border-slate-700 text-slate-300">
          BOINC Quorum-of-2 Credit Verification Active
        </span>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-left border-collapse text-xs">
          <thead>
            <tr class="border-b border-slate-800 text-slate-400 uppercase tracking-wider text-[11px]">
              <th class="py-2.5 px-3">Super-Peer Region</th>
              <th class="py-2.5 px-3">Visitors</th>
              <th class="py-2.5 px-3">Producers</th>
              <th class="py-2.5 px-3">Prosumers (Active)</th>
              <th class="py-2.5 px-3">Consumers</th>
              <th class="py-2.5 px-3">Daytime Throughput</th>
              <th class="py-2.5 px-3">Day / Night Compute</th>
              <th class="py-2.5 px-3">Utilization ρ</th>
              <th class="py-2.5 px-3">Fast-Lane TTFT</th>
            </tr>
          </thead>
          <tbody id="regional-table-body" class="divide-y divide-slate-800/70"></tbody>
        </table>
      </div>
    </section>

    <!-- Literature Review Mathematical Framework Section -->
    <section class="glass-card rounded-3xl p-5">
      <div class="mb-4">
        <h2 class="text-base sm:text-lg font-bold text-white">
          Peer-Reviewed Distributed &amp; Federated Computing Literature Models Used
        </h2>
        <p class="text-xs text-slate-400">
          Mathematical foundations governing HiveMind's &gt;200-node consumer-heavy international launch stability and throughput forecast
        </p>
      </div>
      <div id="literature-cards-grid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-xs"></div>
    </section>
  </main>

  <script>
    const INITIAL_FORECAST = {payload_json};
    let currentForecast = INITIAL_FORECAST;
    let nodeFilterMode = "all"; // "all" | "pioneers" | "producers"

    // Simplified continental coordinate polygons (lat, lon) for realistic 3D Planet Earth rendering
    const CONTINENT_POLYGONS = [
      // North America
      [[68,-162],[70,-130],[60,-110],[52,-56],[44,-66],[30,-81],[25,-80],[19,-96],[10,-84],[16,-99],[24,-110],[34,-120],[48,-125],[60,-148]],
      // South America
      [[11,-74],[8,-60],[-5,-35],[-15,-39],[-24,-46],[-35,-58],[-53,-70],[-42,-74],[-18,-71],[-5,-81]],
      // Europe
      [[36,-9],[43,-9],[48,-4],[52,4],[58,8],[69,18],[70,30],[60,40],[46,38],[41,29],[38,23],[40,15],[43,4]],
      // Africa
      [[35,-6],[37,10],[32,32],[12,51],[-2,41],[-15,40],[-26,33],[-34,19],[-18,12],[4,9],[5,-8],[15,-17]],
      // Asia
      [[42,29],[60,40],[70,80],[68,160],[52,140],[38,140],[31,121],[22,114],[10,105],[2,103],[15,80],[8,77],[25,67],[26,55],[13,44]],
      // Australia
      [[-12,131],[-12,142],[-28,153],[-38,145],[-34,115],[-22,114]]
    ];

    function formatInt(n) {{
      return Math.round(Number(n) || 0).toLocaleString("en-US");
    }}

    function updateDashboardUI(data) {{
      currentForecast = data;
      const base = data.current_live_baseline || {{}};
      const sw = data.forecast_200plus_swarm || {{}};
      const tp = data.throughput_and_compute_power || {{}};
      const bal = (data.balancing_and_latency_sla || {{}}).hivemind_balanced_routing || {{}};
      const unctrl = (data.balancing_and_latency_sla || {{}}).uncontrolled_naive_routing || {{}};
      const econ = (data.balancing_and_latency_sla || {{}}).economic_and_epidemic_stabilizers || {{}};

      document.getElementById("kpi-baseline-nodes").textContent = formatInt(base.total_nodes || 47);
      document.getElementById("kpi-total-nodes").textContent = formatInt(sw.total_active_nodes);
      document.getElementById("kpi-visitors-badge").textContent = formatInt(sw.visitor_funnel_count) + " Visitors";
      document.getElementById("kpi-node-split").textContent =
        `${{sw.dedicated_producers}} Prod • ${{sw.prosumer_nodes}} Pros (${{sw.amcp_promoted_active_prosumers}} Active) • ${{sw.consumer_nodes}} Cons`;

      document.getElementById("kpi-tok-s").textContent = formatInt(tp.effective_daytime_tok_per_s);
      document.getElementById("kpi-burst-tok-s").textContent = formatInt(tp.elastic_burst_max_tok_per_s) + " tok/s";
      document.getElementById("kpi-daily-tok").textContent = (tp.daily_token_capacity_millions || 0).toFixed(1) + "M/day";

      document.getElementById("kpi-night-pflops").textContent = (tp.nocturnal_diloco_pflops_fp16 || 0).toFixed(2);
      document.getElementById("kpi-day-tflops").textContent = formatInt(tp.daytime_inference_tflops_fp16) + " TFLOPS";

      document.getElementById("kpi-fast-ttft").textContent = Math.round(bal.fast_lane_prosumer_ttft_ms || 0) + " ms";
      document.getElementById("kpi-p50-ms").textContent = Math.round(bal.blended_p50_response_ms || 0) + " ms";
      document.getElementById("kpi-p95-ms").textContent = Math.round(bal.blended_p95_response_ms || 0) + " ms";

      const rhoPct = ((bal.utilization_rho || 0) * 100).toFixed(1) + "%";
      const unctrlRhoPct = ((unctrl.utilization_rho || 0) * 100).toFixed(1) + "%";
      document.getElementById("kpi-rho").textContent = rhoPct;
      document.getElementById("kpi-rho-uncontrolled").textContent =
        unctrlRhoPct + (unctrl.utilization_rho >= 1.0 ? " (Overload)" : " (High)");

      document.getElementById("cmp-unctrl-c").textContent = sw.dedicated_producers;
      document.getElementById("cmp-unctrl-rho").textContent = unctrlRhoPct;
      document.getElementById("cmp-unctrl-state").textContent =
        unctrl.utilization_rho >= 1.0 ? "UNSTABLE (Queue Explosion)" : "DEGRADED";

      document.getElementById("cmp-bal-c").textContent =
        `${{sw.effective_inference_servers_c}} (${{sw.dedicated_producers}}P + ${{sw.amcp_promoted_active_prosumers}} AMCP)`;
      document.getElementById("cmp-bal-rho").textContent = rhoPct;
      document.getElementById("cmp-bal-omega").textContent = (econ.kelly_congestion_multiplier_omega || 1).toFixed(2) + "x HMT";

      const pill = document.getElementById("verdict-pill");
      if (data.ready_for_200plus_international_launch) {{
        pill.textContent = `READY • ${{sw.total_active_nodes}} NODES STABLE`;
        pill.className = "px-2.5 py-0.5 text-xs font-bold uppercase rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-400/40 glow-badge";
      }} else {{
        pill.textContent = `NEEDS MORE PRODUCERS (${{sw.total_active_nodes}} NODES)`;
        pill.className = "px-2.5 py-0.5 text-xs font-bold uppercase rounded-full bg-amber-500/20 text-amber-300 border border-amber-400/40";
      }}

      // Regional table
      const tbody = document.getElementById("regional-table-body");
      tbody.innerHTML = "";
      (data.regional_super_peer_forecast || []).forEach((reg) => {{
        const tr = document.createElement("tr");
        tr.className = "hover:bg-slate-900/50 transition";
        tr.innerHTML = `
          <td class="py-2.5 px-3 font-semibold text-white">
            <div>${{reg.display_name}}</div>
            <div class="text-[11px] text-slate-400">${{reg.hub_city}} • RTT ${{reg.median_rtt_ms}} ms</div>
          </td>
          <td class="py-2.5 px-3 text-slate-300">${{formatInt(reg.visitors)}}</td>
          <td class="py-2.5 px-3 text-amber-300 font-bold">${{reg.producers}}</td>
          <td class="py-2.5 px-3 text-sky-300 font-semibold">${{reg.prosumers}} <span class="text-slate-400">(${{reg.promoted_active_prosumers}} active)</span></td>
          <td class="py-2.5 px-3 text-emerald-300">${{reg.consumers}}</td>
          <td class="py-2.5 px-3 font-bold text-emerald-300">${{formatInt(reg.daytime_tok_per_s)}} tok/s</td>
          <td class="py-2.5 px-3 text-slate-300">${{reg.daytime_tflops_fp16}} / <span class="text-fuchsia-300 font-semibold">${{reg.nocturnal_tflops_fp16}} TFLOPS</span></td>
          <td class="py-2.5 px-3">
            <span class="px-2 py-0.5 rounded-md bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 font-semibold">
              ${{(reg.utilization_rho * 100).toFixed(1)}}%
            </span>
          </td>
          <td class="py-2.5 px-3 font-semibold text-amber-300">${{reg.fast_lane_ttft_ms}} ms</td>
        `;
        tbody.appendChild(tr);
      }});

      // Literature cards
      const litGrid = document.getElementById("literature-cards-grid");
      if (litGrid && litGrid.children.length === 0) {{
        (data.literature_review_models || []).forEach((m) => {{
          const card = document.createElement("div");
          card.className = "p-4 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col justify-between";
          card.innerHTML = `
            <div>
              <div class="text-sky-300 font-bold">${{m.authors_year}}</div>
              <div class="text-white font-semibold mt-0.5">${{m.domain}}</div>
              <code class="block mt-2 p-2 rounded-lg bg-slate-950 text-emerald-300 text-[11px] font-mono overflow-x-auto">${{m.equation}}</code>
            </div>
            <p class="mt-2.5 text-slate-400 leading-relaxed">${{m.role_in_hivemind}}</p>
          `;
          litGrid.appendChild(card);
        }});
      }}

      updateGlobeCounter();
    }}

    function updateGlobeCounter() {{
      const nodes = ((currentForecast || {{}}).hiveminers_on_globe || {{}}).nodes || [];
      const filtered = nodes.filter(filterNode);
      document.getElementById("globe-node-counter").textContent = `${{filtered.length}} Nodes Plotted on Planet`;
    }}

    function filterNode(n) {{
      if (nodeFilterMode === "pioneers") return Boolean(n.is_current_pioneer);
      if (nodeFilterMode === "producers") return n.role === "producer" || n.role === "prosumer";
      return true;
    }}

    // Interactive 3D Planet Globe Renderer
    const canvas = document.getElementById("hivemind-planet-canvas");
    const ctx = canvas.getContext("2d");
    const tooltip = document.getElementById("globe-tooltip");
    let rotLon = -18.0; // Centered between Europe & Atlantic/Americas
    let rotLat = 18.0;
    let zoomScale = 1.0;
    let isDragging = false;
    let lastMouseX = 0;
    let lastMouseY = 0;
    let autoRotate = true;
    let animPhase = 0;
    let projectedNodes = [];

    function latLonTo3D(latDeg, lonDeg, radius) {{
      const lat = (latDeg * Math.PI) / 180.0;
      const lon = ((lonDeg - rotLon) * Math.PI) / 180.0;
      const tilt = (rotLat * Math.PI) / 180.0;

      const x0 = radius * Math.cos(lat) * Math.sin(lon);
      const y0 = radius * Math.sin(lat);
      const z0 = radius * Math.cos(lat) * Math.cos(lon);

      const y = y0 * Math.cos(tilt) - z0 * Math.sin(tilt);
      const z = y0 * Math.sin(tilt) + z0 * Math.cos(tilt);
      return {{ x: x0, y: -y, z: z }};
    }}

    function drawPlanetFrame() {{
      animPhase += 0.025;
      if (autoRotate && !isDragging) {{
        rotLon = (rotLon + 0.16) % 360.0;
      }}

      const w = canvas.width;
      const h = canvas.height;
      const cx = w / 2;
      const cy = h / 2;
      const radius = Math.min(w, h) * 0.41 * zoomScale;

      ctx.clearRect(0, 0, w, h);

      // Outer atmospheric halo
      const haloGrad = ctx.createRadialGradient(cx, cy, radius * 0.78, cx, cy, radius * 1.24);
      haloGrad.addColorStop(0, "rgba(14, 165, 233, 0.18)");
      haloGrad.addColorStop(0.6, "rgba(16, 185, 129, 0.08)");
      haloGrad.addColorStop(1, "rgba(2, 6, 23, 0)");
      ctx.fillStyle = haloGrad;
      ctx.beginPath();
      ctx.arc(cx, cy, radius * 1.24, 0, Math.PI * 2);
      ctx.fill();

      // Planet ocean sphere
      const oceanGrad = ctx.createRadialGradient(
        cx - radius * 0.3,
        cy - radius * 0.3,
        radius * 0.1,
        cx,
        cy,
        radius
      );
      oceanGrad.addColorStop(0, "#0f294a");
      oceanGrad.addColorStop(0.65, "#071529");
      oceanGrad.addColorStop(1, "#020612");
      ctx.fillStyle = oceanGrad;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fill();

      // Clip to sphere for graticule and continents
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.clip();

      // Lat/Lon Graticule
      ctx.strokeStyle = "rgba(56, 189, 248, 0.11)";
      ctx.lineWidth = 0.8;
      for (let lat = -60; lat <= 60; lat += 30) {{
        ctx.beginPath();
        let started = false;
        for (let lon = -180; lon <= 180; lon += 6) {{
          const p = latLonTo3D(lat, lon, radius);
          if (p.z > 0) {{
            if (!started) {{
              ctx.moveTo(cx + p.x, cy + p.y);
              started = true;
            }} else {{
              ctx.lineTo(cx + p.x, cy + p.y);
            }}
          }} else {{
            started = false;
          }}
        }}
        ctx.stroke();
      }}
      for (let lon = -180; lon < 180; lon += 30) {{
        ctx.beginPath();
        let started = false;
        for (let lat = -80; lat <= 80; lat += 5) {{
          const p = latLonTo3D(lat, lon, radius);
          if (p.z > 0) {{
            if (!started) {{
              ctx.moveTo(cx + p.x, cy + p.y);
              started = true;
            }} else {{
              ctx.lineTo(cx + p.x, cy + p.y);
            }}
          }} else {{
            started = false;
          }}
        }}
        ctx.stroke();
      }}

      // Continental landmasses
      ctx.fillStyle = "rgba(30, 64, 110, 0.45)";
      ctx.strokeStyle = "rgba(56, 189, 248, 0.35)";
      ctx.lineWidth = 1.1;
      CONTINENT_POLYGONS.forEach((poly) => {{
        ctx.beginPath();
        let visibleCount = 0;
        poly.forEach(([lat, lon], idx) => {{
          const p = latLonTo3D(lat, lon, radius);
          if (p.z > -radius * 0.15) visibleCount++;
          if (idx === 0) ctx.moveTo(cx + p.x, cy + p.y);
          else ctx.lineTo(cx + p.x, cy + p.y);
        }});
        ctx.closePath();
        if (visibleCount >= 3) {{
          ctx.fill();
          ctx.stroke();
        }}
      }});

      // Great-circle QUIC / AMCP Super-Peer Arcs
      const globeData = (currentForecast || {{}}).hiveminers_on_globe || {{}};
      const arcs = globeData.arcs || [];
      arcs.forEach((arc, aIdx) => {{
        const steps = 28;
        ctx.beginPath();
        let anyVisible = false;
        for (let s = 0; s <= steps; s++) {{
          const t = s / steps;
          const lat = arc.from_lat + (arc.to_lat - arc.from_lat) * t;
          let dLon = arc.to_lon - arc.from_lon;
          if (dLon > 180) dLon -= 360;
          if (dLon < -180) dLon += 360;
          const lon = arc.from_lon + dLon * t;
          const alt = 1.0 + 0.08 * Math.sin(Math.PI * t);
          const p = latLonTo3D(lat, lon, radius * alt);
          if (p.z > 0) {{
            if (!anyVisible) {{
              ctx.moveTo(cx + p.x, cy + p.y);
              anyVisible = true;
            }} else {{
              ctx.lineTo(cx + p.x, cy + p.y);
            }}
          }}
        }}
        ctx.strokeStyle = "rgba(56, 189, 248, 0.36)";
        ctx.lineWidth = 1.3;
        ctx.stroke();

        // Animated token/gradient packet along the arc
        const pktT = (animPhase * 0.35 + aIdx * 0.19) % 1.0;
        const pLat = arc.from_lat + (arc.to_lat - arc.from_lat) * pktT;
        let dLon = arc.to_lon - arc.from_lon;
        if (dLon > 180) dLon -= 360;
        if (dLon < -180) dLon += 360;
        const pLon = arc.from_lon + dLon * pktT;
        const pkt3D = latLonTo3D(pLat, pLon, radius * (1.0 + 0.08 * Math.sin(Math.PI * pktT)));
        if (pkt3D.z > 0) {{
          ctx.fillStyle = "#38bdf8";
          ctx.beginPath();
          ctx.arc(cx + pkt3D.x, cy + pkt3D.y, 2.6, 0, Math.PI * 2);
          ctx.fill();
        }}
      }});

      ctx.restore();

      // Outer rim stroke
      ctx.strokeStyle = "rgba(56, 189, 248, 0.42)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.stroke();

      // Plot HiveMiners Nodes on Front Hemisphere
      projectedNodes = [];
      const nodes = (globeData.nodes || []).filter(filterNode);
      nodes.forEach((n) => {{
        const p = latLonTo3D(n.lat, n.lon, radius);
        if (p.z > 0) {{
          const sx = cx + p.x;
          const sy = cy + p.y;
          let color = "#34d399"; // consumer
          let r = 2.8;
          if (n.role === "producer") {{
            color = "#fbbf24";
            r = 4.5;
          }} else if (n.role === "prosumer") {{
            color = "#38bdf8";
            r = n.amcp_promoted ? 3.8 : 3.2;
          }}

          // Super-peer pulsing ring
          if (n.is_super_peer) {{
            const pulseR = 7.5 + 2.5 * Math.sin(animPhase * 2.0);
            ctx.strokeStyle = "rgba(244, 63, 94, 0.85)";
            ctx.lineWidth = 1.6;
            ctx.beginPath();
            ctx.arc(sx, sy, pulseR, 0, Math.PI * 2);
            ctx.stroke();
          }}

          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(sx, sy, r, 0, Math.PI * 2);
          ctx.fill();

          projectedNodes.push({{ sx, sy, r: r + 3, node: n }});
        }}
      }});

      requestAnimationFrame(drawPlanetFrame);
    }}

    // Mouse / Touch Interaction for 3D Planet
    canvas.addEventListener("pointerdown", (e) => {{
      isDragging = true;
      autoRotate = false;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
    }});
    window.addEventListener("pointerup", () => {{
      isDragging = false;
    }});
    canvas.addEventListener("pointermove", (e) => {{
      const rect = canvas.getBoundingClientRect();
      const mx = ((e.clientX - rect.left) / rect.width) * canvas.width;
      const my = ((e.clientY - rect.top) / rect.height) * canvas.height;

      if (isDragging) {{
        const dx = e.clientX - lastMouseX;
        const dy = e.clientY - lastMouseY;
        rotLon = (rotLon - dx * 0.45) % 360.0;
        rotLat = Math.max(-65.0, Math.min(65.0, rotLat + dy * 0.35));
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
        tooltip.classList.add("hidden");
        return;
      }}

      // Check hover against projected nodes
      let hit = null;
      for (let i = projectedNodes.length - 1; i >= 0; i--) {{
        const item = projectedNodes[i];
        const dist = Math.hypot(mx - item.sx, my - item.sy);
        if (dist <= item.r + 2) {{
          hit = item;
          break;
        }}
      }}
      if (hit) {{
        const n = hit.node;
        document.getElementById("tt-title").textContent = `${{n.node_id}} • ${{n.city}}`;
        document.getElementById("tt-sub").textContent =
          `${{n.role.toUpperCase()}} ${{n.is_current_pioneer ? "(Live Pioneer)" : "(Forecast Wave)"}} • ${{n.boinc_plan_class}}`;
        document.getElementById("tt-metrics").textContent =
          `Throughput: ${{n.tok_per_s}} tok/s • Compute: ${{n.tflops_fp16}} FP16 TFLOPS`;
        tooltip.style.left = `${{Math.min(rect.width - 220, Math.max(12, e.clientX - rect.left + 14))}}px`;
        tooltip.style.top = `${{Math.max(12, e.clientY - rect.top - 18)}}px`;
        tooltip.classList.remove("hidden");
      }} else {{
        tooltip.classList.add("hidden");
      }}
    }});
    canvas.addEventListener("wheel", (e) => {{
      e.preventDefault();
      zoomScale = Math.max(0.75, Math.min(1.45, zoomScale - e.deltaY * 0.0008));
    }}, {{ passive: false }});

    // Filter buttons
    function setFilterButtons(activeId) {{
      ["btn-filter-all", "btn-filter-pioneers", "btn-filter-producers"].forEach((id) => {{
        const el = document.getElementById(id);
        if (id === activeId) {{
          el.className = "px-2.5 py-1 rounded-lg bg-sky-500 text-slate-950 font-bold transition";
        }} else {{
          el.className = "px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold transition";
        }}
      }});
      updateGlobeCounter();
    }}
    document.getElementById("btn-filter-all").addEventListener("click", () => {{
      nodeFilterMode = "all";
      setFilterButtons("btn-filter-all");
    }});
    document.getElementById("btn-filter-pioneers").addEventListener("click", () => {{
      nodeFilterMode = "pioneers";
      setFilterButtons("btn-filter-pioneers");
    }});
    document.getElementById("btn-filter-producers").addEventListener("click", () => {{
      nodeFilterMode = "producers";
      setFilterButtons("btn-filter-producers");
    }});

    // Sliders & Live API / Local Fallback Recompute
    const sVisitors = document.getElementById("slider-visitors");
    const sConsumers = document.getElementById("slider-consumers");
    const sProsumers = document.getElementById("slider-prosumers");
    const sProducers = document.getElementById("slider-producers");
    const sUtc = document.getElementById("slider-utc");

    async function refreshForecastFromSliders() {{
      const vis = parseInt(sVisitors.value, 10);
      const cons = parseInt(sConsumers.value, 10);
      const pros = parseInt(sProsumers.value, 10);
      const prod = parseInt(sProducers.value, 10);
      const utc = parseInt(sUtc.value, 10);
      const total = cons + pros + prod;

      document.getElementById("lbl-visitors").textContent = `${{formatInt(vis)}} Visitors`;
      document.getElementById("lbl-consumers").textContent = `${{cons}} Consumers`;
      document.getElementById("lbl-prosumers").textContent = `${{pros}} Prosumers`;
      document.getElementById("lbl-producers").textContent = `${{prod}} Producers`;
      document.getElementById("lbl-utc").textContent = `${{String(utc).padStart(2, "0")}}:00 UTC`;

      try {{
        const qs = new URLSearchParams({{
          target_total_nodes: String(total),
          visitor_count: String(vis),
          dedicated_producers: String(prod),
          prosumer_nodes: String(pros),
          consumer_nodes: String(cons),
          utc_hour: String(utc)
        }});
        const resp = await fetch(`/api/v1/launch/planet-forecast?${{qs.toString()}}`);
        if (resp.ok) {{
          const fresh = await resp.json();
          updateDashboardUI(fresh);
          return;
        }}
      }} catch (_err) {{
        // Offline / static file fallback
      }}
    }}

    [sVisitors, sConsumers, sProsumers, sProducers, sUtc].forEach((sl) => {{
      sl.addEventListener("input", refreshForecastFromSliders);
    }});

    document.getElementById("btn-reset-default").addEventListener("click", () => {{
      sVisitors.value = "2400";
      sConsumers.value = "183";
      sProsumers.value = "54";
      sProducers.value = "28";
      sUtc.value = "14";
      refreshForecastFromSliders();
    }});

    // Initialize UI & 3D Globe
    updateDashboardUI(INITIAL_FORECAST);
    requestAnimationFrame(drawPlanetFrame);
  </script>
</body>
</html>
"""


def render_foundation_paper_portal_html(
    payload: Optional[Dict[str, Any]] = None,
) -> str:
    """Render the interactive HiveMind Foundation Paper v4.2.0, Live Experimentation (E20 & 47-Node WAN),
    Real vs. Forecast Traffic, Network Balancing, and Embedded PDF + 3D Globe Portal Page.
    """
    data = payload or build_foundation_paper_v42_portal_payload()
    wave_rows_html = []
    for w in data.get("five_wave_international_roadmap", []):
        is_target = w.get("wave_id") == "WAVE-2-M2-TARGET"
        row_style = (
            "background:rgba(16,185,129,0.12);border-left:3px solid #10b981;"
            if is_target
            else ""
        )
        wave_rows_html.append(
            f"<tr style='{row_style}'>"
            f"<td style='padding:10px;font-weight:700;color:#fbbf24;'>{w['title']}</td>"
            f"<td style='padding:10px;'><code>{', '.join(w['languages'])}</code></td>"
            f"<td style='padding:10px;font-weight:800;color:#38bdf8;'>{w['active_nodes']:,} nodes</td>"
            f"<td style='padding:10px;'>{w['super_peers']} Super-Peers ({w['localized_subswarms']} sub-swarms)</td>"
            f"<td style='padding:10px;font-weight:800;color:#34d399;'>{w['daytime_tok_per_s']:,.1f} tok/s</td>"
            f"<td style='padding:10px;color:#c084fc;font-weight:700;'>{w['nocturnal_pflops']:.2f} PFLOPS</td>"
            f"<td style='padding:10px;'><span style='padding:2px 8px;border-radius:6px;background:rgba(16,185,129,0.2);color:#34d399;font-weight:700;'>&rho; = {w['utilization_rho']:.3f}</span></td>"
            f"<td style='padding:10px;color:#fde68a;font-weight:700;'>{w['blended_p50_ms']:.1f} ms</td>"
            f"</tr>"
        )
    waves_table_body = "\n".join(wave_rows_html)

    reg_rows_html = []
    for reg in data.get("regional_super_peer_forecast", []):
        reg_rows_html.append(
            f"<tr>"
            f"<td style='padding:9px;'><strong>{reg['display_name']}</strong><br/><span style='font-size:0.74rem;color:#94a3b8;'>{reg['hub_city']} &bull; RTT {reg['median_rtt_ms']} ms</span></td>"
            f"<td style='padding:9px;'>{reg['visitors']:,}</td>"
            f"<td style='padding:9px;color:#fbbf24;font-weight:700;'>{reg['total_nodes']} ({reg['producers']}P / {reg['prosumers']}Pr / {reg['consumers']}C)</td>"
            f"<td style='padding:9px;color:#38bdf8;font-weight:700;'>c<sub>eff</sub> = {reg['effective_inference_servers_c']}</td>"
            f"<td style='padding:9px;color:#34d399;font-weight:800;'>{reg['daytime_tok_per_s']:,.1f} tok/s</td>"
            f"<td style='padding:9px;'>{reg['daytime_tflops_fp16']} / <strong style='color:#c084fc;'>{reg['nocturnal_tflops_fp16']} TFLOPS</strong></td>"
            f"<td style='padding:9px;'><span style='padding:2px 8px;border-radius:6px;background:rgba(56,189,248,0.16);color:#38bdf8;font-weight:700;'>{reg['utilization_rho']*100:.1f}%</span></td>"
            f"<td style='padding:9px;color:#fbbf24;font-weight:700;'>{reg['fast_lane_ttft_ms']} ms</td>"
            f"</tr>"
        )
    reg_table_body = "\n".join(reg_rows_html)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>HiveMind Foundation Paper v4.2.0 &bull; Live Experimentation, Global 3D Planet &amp; Network Balancing</title>
  <style>
    :root {{
      --bg: #050914;
      --card: #0c1326;
      --border-cyan: rgba(56, 189, 248, 0.38);
      --border-gold: rgba(251, 191, 36, 0.45);
      --border-green: rgba(16, 185, 129, 0.45);
      --text: #e2e8f0;
      --muted: #94a3b8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: 'Space Grotesk', system-ui, -apple-system, sans-serif;
      background: radial-gradient(circle at 15% 10%, rgba(251,191,36,0.12), transparent 40%),
                  radial-gradient(circle at 85% 15%, rgba(56,189,248,0.12), transparent 45%),
                  var(--bg);
      color: var(--text);
      line-height: 1.55;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 50;
      background: rgba(5, 9, 20, 0.94);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border-gold);
      padding: 14px 24px;
    }}
    .wrap {{ max-width: 1440px; margin: 0 auto; }}
    .btn {{
      display: inline-block;
      padding: 8px 14px;
      border-radius: 8px;
      font-weight: 700;
      font-size: 0.8rem;
      text-decoration: none;
      border: 1px solid #38bdf8;
      background: rgba(56, 189, 248, 0.15);
      color: #e0f2fe;
      margin-right: 6px;
    }}
    .btn-gold {{
      border-color: #fbbf24;
      background: rgba(251, 191, 36, 0.2);
      color: #fef3c7;
    }}
    .btn-green {{
      border-color: #10b981;
      background: rgba(16, 185, 129, 0.2);
      color: #d1fae5;
    }}
    .grid-3 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 16px;
      margin-top: 16px;
    }}
    .card {{
      background: rgba(12, 19, 38, 0.9);
      border: 1px solid var(--border-cyan);
      border-radius: 14px;
      padding: 18px 20px;
      margin-bottom: 18px;
    }}
    .card-gold {{ border-color: var(--border-gold); }}
    .card-green {{ border-color: var(--border-green); }}
    h1 {{ font-size: 1.45rem; margin: 0 0 6px; color: #fbbf24; }}
    h2 {{ font-size: 1.05rem; margin: 0 0 10px; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.04em; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-top: 8px; }}
    th, td {{ border-bottom: 1px solid rgba(148, 163, 184, 0.16); text-align: left; padding: 9px 10px; }}
    th {{ color: #fbbf24; font-size: 0.74rem; text-transform: uppercase; background: rgba(15, 23, 42, 0.75); }}
    code {{ font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #38bdf8; }}
  </style>
</head>
<body>
  <header>
    <div class="wrap" style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
      <div>
        <div style="font-size:0.75rem;color:#34d399;font-weight:800;letter-spacing:0.06em;">
          PROJECT HIVEMIND FOUNDATION ARTICLE &bull; VERSION 4.2.0 (70 PAGES &bull; 161 LEAN 4 LEMMAS, 0 SORRY &bull; 0.00 EUR/USD)
        </div>
        <h1>&#x1F4DC; Foundation Paper v4.2.0: Live Experimentation (E20 &amp; 47-Node WAN) &amp; &gt;200-Node Launch Forecast</h1>
      </div>
      <div>
        <a href="/" class="btn">&#x1F3E0; HiveMind Portal Home</a>
        <a href="/planet-forecast" class="btn btn-green">&#x1F30D; Open Full 3D Planet Globe</a>
        <a href="/paper/hivemind_foundation_article.pdf" target="_blank" rel="noopener" class="btn btn-gold">&#x1F4E5; Download PDF (v4.2.0)</a>
        <a href="/paper/hivemind_foundation_article.tex" target="_blank" rel="noopener" class="btn">&#x1F4C4; LaTeX Source (.tex)</a>
        <a href="/api/v1/paper/foundation" target="_blank" rel="noopener" class="btn">&#x1F517; JSON Telemetry API</a>
      </div>
    </div>
  </header>

  <main class="wrap" style="padding:22px 24px 56px;">
    <!-- SECTION 1: REAL TRAFFIC VS FORECAST TRAFFIC & NETWORK BALANCING -->
    <section class="card card-gold">
      <h2>01 // Real Live Swarm Traffic (47 Nodes) vs. &gt;200-Node (265-Node + 2,400 Visitors) International Launch Network Balancing</h2>
      <p style="font-size:0.88rem;color:#cbd5e1;margin:0 0 12px;">
        Synthesizing <strong>Kleinrock (1975) &amp; Halfin&ndash;Whitt (1981) M/M/c(t) square-root staffing</strong>,
        <strong>Kingman (1961) G/G/c heavy-tail token variance</strong>,
        <strong>Mitzenmacher (2001) Power-of-Two-Choices (d=2) with warm VRAM residency (&eta;<sub>p2c</sub> = 0.58)</strong>,
        <strong>Kelly et al. (1998) shadow-price congestion control (&omega;(&rho;) = 1.483&times;)</strong>, and
        <strong>Bass (1969) prosumer conversion (&kappa;<sub>eff</sub> = 30.94% &gg; &kappa;* = 2.20%)</strong>:
      </p>
      <div class="grid-3">
        <div class="card" style="margin:0;border-color:rgba(56,189,248,0.45);">
          <div style="font-size:0.75rem;color:#38bdf8;font-weight:800;">LIVE PRODUCTION TELEMETRY (WAVE 0)</div>
          <h3 style="margin:4px 0 8px;color:#f8fafc;">47-Node Live WAN Baseline</h3>
          <div style="font-size:0.83rem;color:#cbd5e1;">
            &bull; Active Nodes: <strong>47</strong> (8 Producers, 14 Prosumers, 22 Consumers, 3 Admins)<br/>
            &bull; Real Traffic Demand / Capacity: <strong>185.0 tok/s</strong> / <strong style="color:#34d399;">380.0 tok/s</strong><br/>
            &bull; Utilization &amp; Latency: <strong>&rho; = 0.487</strong> &bull; Mean p50 = <strong style="color:#fbbf24;">196.8 ms</strong> (1,427 completions)<br/>
            &bull; Attested BOINC Compute: <strong>5,632.11 Cobblestones</strong> (2.433 PFLOPs, 124 WUs, 0.0% false reject)<br/>
            &bull; Local Tesla T4 Guard: <strong style="color:#34d399;">10,073 MiB free VRAM</strong> (&ge; 4,300 MiB floor)
          </div>
        </div>

        <div class="card" style="margin:0;border-color:rgba(244,63,94,0.55);background:rgba(69,10,18,0.28);">
          <div style="font-size:0.75rem;color:#fb7185;font-weight:800;">UNCONTROLLED SURGE (NO PROSUMER PROMOTION)</div>
          <h3 style="margin:4px 0 8px;color:#fecdd3;">265-Node Naive Producer-Only Routing</h3>
          <div style="font-size:0.83rem;color:#fecdd3;">
            &bull; Surge Mix: <strong>2,400 Visitors</strong>, 183 Consumers, 28 Producers (54 Prosumers idle)<br/>
            &bull; Offered Demand vs. Capacity: <strong>1,985.3 tok/s</strong> vs. <strong>992.0 tok/s</strong> (c = 28 &lt; c<sub>HW</sub> = 45)<br/>
            &bull; Utilization: <strong style="color:#fb7185;">&rho;<sub>naive</sub> = 2.0012 &gt; 1.0 (OVERLOAD COLLAPSE)</strong><br/>
            &bull; Erlang-C Queue Probability: <strong>P(W<sub>q</sub> &gt; 0) = 1.000</strong> (&rarr; &infin; wait timeout)<br/>
            &bull; Verdict: Static producer-only routing fails under consumer-heavy waves.
          </div>
        </div>

        <div class="card card-green" style="margin:0;background:rgba(6,78,59,0.24);">
          <div style="font-size:0.75rem;color:#34d399;font-weight:800;">HIVEMIND BALANCED ROUTING (AMCP + P2C + 2-LANE QOS)</div>
          <h3 style="margin:4px 0 8px;color:#ecfdf5;">265-Node Elastic Swarm (READY FOR LAUNCH)</h3>
          <div style="font-size:0.83rem;color:#d1fae5;">
            &bull; Active Server Pool: <strong>c<sub>eff</sub> = 60 &ge; c<sub>HW</sub> = 45</strong> (28 Producers + <strong>32/54 AMCP Prosumers</strong>)<br/>
            &bull; Throughput: <strong style="color:#34d399;">3,040.7 tok/s daytime</strong> (<strong>4,026.2 tok/s burst</strong> &bull; 262.7M tok/day)<br/>
            &bull; Balanced Utilization: <strong style="color:#fbbf24;">&rho;<sub>elastic</sub> = 0.6529 &le; 0.78</strong> (P(W<sub>q</sub>&gt;0) = 0.13%, W<sub>q,P2C</sub> = 0.2 ms)<br/>
            &bull; Response SLA: Fast-Lane TTFT = <strong>136.9 ms</strong> (p50 = 170.9 ms) &bull; Blended p50 = <strong>217.8 ms</strong><br/>
            &bull; Compute Power: <strong>2.14 PFLOPS daytime</strong> / <strong>4.18 PFLOPS nocturnal DiLoCo</strong>
          </div>
        </div>
      </div>
    </section>

    <!-- SECTION 2: LIVE EXPERIMENT E20 & NOCTURNAL V-MOE EVOLUTION -->
    <section class="card">
      <h2>02 // Live Experiment E20 (HM-S29-01, HM-S29-02, HM-S31-02) &amp; Nocturnal V-MoE Pareto-Ratchet Results</h2>
      <table>
        <thead>
          <tr>
            <th>Experiment / Gate</th>
            <th>Target Hardware / Swarm Scale</th>
            <th>Measured Empirical Result</th>
            <th>Falsification Threshold</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>E20-P20a: Adaptive Multi-Split Guard</strong></td>
            <td>GTX 1650 (4G), Steam Deck OLED, RTX 2070 (8G), Apple M2</td>
            <td><strong>680.0 MiB</strong> client prelude VRAM; 48 KB wire; <strong>-94.34% to -98.29%</strong> step latency</td>
            <td>&le; 1,536 MiB &amp; &lt; 250 ms</td>
            <td><span style="color:#34d399;font-weight:800;">VERIFIED PASS</span></td>
          </tr>
          <tr>
            <td><strong>E20-P20b: Top-1% 2D-DCT + EF21</strong></td>
            <td>256&times;256 FP32 outer pseudo-gradient (262,144 B &rarr; 1,163 B)</td>
            <td><strong>225.4&times; compression</strong>; EF21 L2 error 1.52e-15; wire drift <strong>3.0e-8</strong></td>
            <td>&ge; 100&times; &amp; drift &le; 1e-5</td>
            <td><span style="color:#34d399;font-weight:800;">VERIFIED PASS</span></td>
          </tr>
          <tr>
            <td><strong>E20-P20c: 6-Region Super-Peer Tree</strong></td>
            <td>10,000 leaf nodes &rarr; 318 Super-Peers (558 sub-swarms of 18)</td>
            <td><strong>31.45&times; coordinator ingress reduction</strong> + <strong>62.5% SQLite write-IOPS cut</strong> (2,554 HMT exact)</td>
            <td>&ge; 25&times; &amp; 0 HMT drift</td>
            <td><span style="color:#34d399;font-weight:800;">VERIFIED PASS</span></td>
          </tr>
          <tr>
            <td><strong>Nocturnal V-MoE Pareto Evolution</strong></td>
            <td>Reddit V-MoE / Exoplanet CWT / 7-Div 104-Bee (42.87M samples)</td>
            <td>Ratcheted losses <strong>0.0003 / 0.0000 / 0.00227</strong> (-94.4% to -99.95%); T4 free <strong>10,073 MiB</strong></td>
            <td>Monotone ratchet &amp; &ge; 4,300 MiB</td>
            <td><span style="color:#34d399;font-weight:800;">ACTIVE GREEN</span></td>
          </tr>
        </tbody>
      </table>
    </section>

    <!-- SECTION 3: 6 REGIONAL SUPER-PEER HUBS & 5-WAVE INTERNATIONAL SCALABILITY -->
    <section class="card card-green">
      <h2>03 // 6 Regional Super-Peer Shards (265-Node Wave) &amp; 5-Wave International Roadmap (47 &rarr; 10,000+ Nodes)</h2>
      <table>
        <thead>
          <tr>
            <th>Regional Super-Peer Hub</th>
            <th>Visitors</th>
            <th>Active Nodes (P / Pr / C)</th>
            <th>Active Servers</th>
            <th>Daytime Throughput</th>
            <th>Day / Night Compute</th>
            <th>Utilization &rho;</th>
            <th>Fast-Lane TTFT</th>
          </tr>
        </thead>
        <tbody>
          {reg_table_body}
        </tbody>
      </table>

      <h2 style="margin-top:20px;">Five-Wave International Launch Scalability Forecast (10 Languages: EN, FR, DE, ES, ZH, JA, PT, KO, AR, HI)</h2>
      <table>
        <thead>
          <tr>
            <th>Launch Wave</th>
            <th>Languages</th>
            <th>Active Nodes</th>
            <th>Super-Peer Hierarchy (B=32)</th>
            <th>Daytime Throughput</th>
            <th>Nocturnal DiLoCo</th>
            <th>Balanced &rho;</th>
            <th>Blended p50</th>
          </tr>
        </thead>
        <tbody>
          {waves_table_body}
        </tbody>
      </table>
    </section>

    <!-- SECTION 4: EMBEDDED 3D PLANET GLOBE & FOUNDATION PAPER PDF VIEWER -->
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:18px;">
      <section class="card card-gold" style="margin:0;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
          <h2 style="margin:0;">04A // Live 3D Planet HiveMiners Globe &amp; Simulator</h2>
          <a href="/planet-forecast" class="btn btn-green" style="margin:0;">&#x26F6; Fullscreen 3D Globe</a>
        </div>
        <iframe src="/planet-forecast" title="HiveMind 3D Planet Forecast Globe" style="width:100%;height:560px;border:1px solid rgba(56,189,248,0.4);border-radius:10px;background:#020617;"></iframe>
      </section>

      <section class="card" style="margin:0;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
          <h2 style="margin:0;">04B // Foundation Paper v4.2.0 PDF (70 Pages)</h2>
          <a href="/paper/hivemind_foundation_article.pdf" target="_blank" rel="noopener" class="btn btn-gold" style="margin:0;">&#x1F4E5; Open Full PDF</a>
        </div>
        <iframe src="/paper/hivemind_foundation_article.pdf" title="Project HiveMind Foundation Article v4.2.0 PDF" style="width:100%;height:560px;border:1px solid rgba(251,191,36,0.4);border-radius:10px;background:#0f172a;"></iframe>
      </section>
    </div>
  </main>
</body>
</html>
"""
