#!/usr/bin/env python3
"""Consumer-GPU economics for a volunteer swarm, and the project's exposure to an AI-capex downturn.

Every parameter carries its source and date. Outputs are arithmetic on those parameters, not
measurements; the receipt says so. Nothing here is a forecast of prices. Run:

    python3 scripts/consumer_gpu_economics.py [--out artifacts/experiments/CONSUMER_GPU_ECONOMICS_2026-11-07.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

# ---- parameters, each with provenance -------------------------------------------------------
P = {
    # Supply side: how many consumer GPUs exist and how many are idle
    "steam_peak_concurrent_users": {"value": 43_505_724, "unit": "accounts logged in", "source": "SteamDB record, 11 October 2026 (press reports)", "note": "logged-in accounts, not GPUs; 14.5M in-game at the time"},
    "steam_mau_stale": {"value": 132_000_000, "unit": "monthly active accounts", "source": "Valve, 2021 (last published MAU)", "note": "stale by five years; upper bound only"},
    "steam_vram_share_16gb": {"value": 0.2721, "unit": "share of surveyed GPUs", "source": "Steam Hardware Survey, September 2026 (press reports)"},
    "steam_vram_share_8gb": {"value": 0.2671, "unit": "share", "source": "Steam Hardware Survey, September 2026 (press reports)"},
    "steam_vram_share_12gb": {"value": 0.1301, "unit": "share", "source": "Steam Hardware Survey, June 2026 (press report; later months not found)"},
    "jpr_q1_2026_discrete_shipments": {"value": 11_800_000, "unit": "add-in boards per quarter", "source": "Jon Peddie Research, Q1 2026", "note": "sell-in to channel, not sell-through"},
    "jpr_discrete_installed_base_forecast": {"value": 183_000_000, "unit": "discrete GPUs", "source": "JPR long-term forecast, cited by secondary aggregators; verify against JPR", "note": "end-of-forecast-period figure, not today's"},
    "idle_hours_per_day": {"value": 18.0, "unit": "h/day", "source": "project's own claim in the r/SelfHostedAI post (0% nearly 18 hours a day)", "note": "unmeasured; treated as a parameter"},
    # Cost side: what a volunteer actually pays
    "eu_household_electricity_eur_kwh": {"value": 0.28, "unit": "EUR/kWh", "source": "Eurostat, first half 2026 (released 30 October 2026), EU average"},
    "de_household_electricity_eur_kwh": {"value": 0.36, "unit": "EUR/kWh", "source": "Eurostat H1 2026, Germany"},
    "hu_household_electricity_eur_kwh": {"value": 0.11, "unit": "EUR/kWh", "source": "Eurostat H1 2026, Hungary (EU minimum)"},
    "rtx2070_board_power_w": {"value": 175.0, "unit": "W", "source": "NVIDIA reference specification (TGP); sustained draw under QLoRA is at or below this"},
    "rtx2070_tokens_per_s_inference": {"value": 72.0, "unit": "tok/s", "source": "project's own figure posted on r/TechnologieFrance (7B-class bee, 'around 72 tokens/s'); unmeasured here"},
    "t4_board_power_w": {"value": 70.0, "unit": "W", "source": "NVIDIA reference specification"},
    # Cloud comparison (indicative; the trackers disagree)
    "h100_on_demand_usd_h_oct2026": {"value": 3.81, "unit": "USD/GPU-h", "source": "median on-demand price, tracker, mid-October 2026"},
    "h100_q2_2026_low_usd_h": {"value": 1.80, "unit": "USD/GPU-h", "source": "Q2 2026 low end, market summary (64-75% below the 2024 peak of 8-10 USD/h)"},
    "h100_1y_contract_mar2026_usd_h": {"value": 2.35, "unit": "USD/GPU-h", "source": "SemiAnalysis-based report, March 2026 (up from 1.70 in October 2025)"},
    "t4_gcp_on_demand_usd_h": {"value": 0.35, "unit": "USD/GPU-h", "source": "GCP list price as known to the project (us-central1); verify on the pricing page"},
    "rtx2070_vs_h100_throughput_ratio": {"value": 0.02, "unit": "fraction", "source": "rough order of magnitude for mixed-precision training throughput (not a benchmark)", "note": "sensitivity shown at 0.01 and 0.04"},
    # Macro exposure
    "hyperscaler_capex_2026_guidance_usd_b": {"value": 732.5, "unit": "USD billion", "source": "combined guidance of the four largest hyperscalers, mid-October 2026 report (720-745)"},
    "severe_unwind_probability_one_analyst": {"value": 0.20, "unit": "probability", "source": "one industry analysis's own estimate; an opinion, not data"},
    "project_gcp_budget_usd_month": {"value": 150.0, "unit": "USD/month", "source": "AGENTS.md owner decision, 2026-11-04"},
    "project_gpu_capex_usd": {"value": 0.0, "unit": "USD", "source": "no GPUs purchased; the T4 is a Spot VM, the RTX 2070 pre-exists"},
    "usd_per_eur": {"value": 1.0, "unit": "USD/EUR", "source": "not converted; costs are compared in their own currency, differences of order 10% are below the model's precision"},
}


def v(k: str) -> float:
    return float(P[k]["value"])


def supply_scenarios() -> dict:
    """Addressable idle GPU-hours and an H100-equivalent, for participation counts the project could reach."""
    out = {}
    for participants in (55, 1_000, 10_000, 100_000, 1_000_000):
        idle_h_day = participants * v("idle_hours_per_day")
        rows = {}
        for ratio in (0.01, v("rtx2070_vs_h100_throughput_ratio"), 0.04):
            rows[f"ratio_{ratio}"] = {"h100_equivalent_gpus_24h": round(idle_h_day * ratio / 24.0, 2)}
        out[str(participants)] = {"idle_gpu_hours_per_day": idle_h_day, "h100_equivalent": rows,
                                  "note": "55 = active accounts claimed in the r/TechnologieFrance post (accounts, not GPUs)"}
    share_ge8 = v("steam_vram_share_16gb") + v("steam_vram_share_8gb") + v("steam_vram_share_12gb")
    out["steam_share_8_12_16gb"] = round(share_ge8, 4)
    out["steam_peak_concurrent_times_share"] = int(v("steam_peak_concurrent_users") * share_ge8)
    out["reading"] = ("'Millions of consumer GPUs' is right as an installed base (JPR ships ~12M discrete boards a quarter) and wrong as a volunteer base: "
                      "the project has 55 claimed accounts. The gap is recruitment and trust, not silicon. Even 100,000 volunteers at an 18 h/day idle "
                      "assumption are ~1,500 H100-equivalents at the 0.02 ratio, before validation overhead and churn.")
    return out


def cost_per_hour_and_token() -> dict:
    """Marginal electricity cost of a volunteer GPU-hour and of a million generated tokens, against cloud list prices."""
    rows = {}
    for label, price in (("eu_avg", v("eu_household_electricity_eur_kwh")), ("de", v("de_household_electricity_eur_kwh")), ("hu", v("hu_household_electricity_eur_kwh"))):
        kwh_h = v("rtx2070_board_power_w") / 1000.0
        eur_h = kwh_h * price
        hours_per_m_tokens = 1e6 / v("rtx2070_tokens_per_s_inference") / 3600.0
        rows[label] = {"electricity_eur_per_gpu_hour": round(eur_h, 4),
                       "electricity_eur_per_million_tokens_inference": round(eur_h * hours_per_m_tokens, 3),
                       "electricity_eur_per_8h_night_training": round(eur_h * 8.0, 3)}
    rows["cloud"] = {
        "t4_gcp_on_demand_usd_per_gpu_hour": v("t4_gcp_on_demand_usd_h"),
        "h100_on_demand_usd_per_gpu_hour_oct2026": v("h100_on_demand_usd_h_oct2026"),
        "h100_q2_2026_low_usd_per_gpu_hour": v("h100_q2_2026_low_usd_h"),
    }
    rows["hours_per_million_tokens_at_72_tok_s"] = round(1e6 / 72.0 / 3600.0, 2)
    rows["reading"] = ("A volunteer pays about 5 euro cents of electricity per RTX 2070 hour at the EU average (3 in Hungary, 6 in Germany); an 8-hour night costs "
                       "them roughly 0.4 EUR. Hardware depreciation, the volunteer's time, and the project's validation cost are not in this number. "
                       "The non-monetary reciprocity charter means the project never pays this; it must be worth 0.4 EUR a night to the volunteer in API access or purpose.")
    return rows


def bubble_scenarios() -> dict:
    """What an AI-capex downturn would do to a non-monetary volunteer swarm. Qualitative; the numbers are the parameters above."""
    return {
        "exposure_of_the_project": {
            "gpu_capex_usd": v("project_gpu_capex_usd"),
            "monthly_cloud_budget_usd": v("project_gcp_budget_usd_month"),
            "debt": 0.0,
            "reading": "The project owns no GPUs, carries no debt, and caps cloud spend at 150 USD a month. Its financial exposure to a downturn is close to nil. Its real exposures are elsewhere: volunteer attention, the supply of open-weight models, and dependence on Ollama and Hugging Face as distribution channels.",
        },
        "S1_cloud_price_collapse": {
            "assumption": "H100 rents fall toward or below 1 USD/h as capacity outruns demand",
            "effect_on_volunteer_compute": "For inference, cheap cloud narrows the gap; electricity-only cost at home (~0.05 EUR/h for a 2070) still undercuts any rented H100 per hour, but per token the H100 wins on throughput. For training small adapters, the home GPU stays viable only where the data must stay local (BYOM) or where purpose, not price, drives participation.",
            "what_to_do": "Keep the pitch on sovereignty, locality and reciprocity, not on cost; stop citing cloud prices as the advantage.",
        },
        "S2_capex_retrenchment": {
            "assumption": "Hyperscalers cut the 2026 guidance of ~730 USD billion; free tiers and subsidised APIs shrink; some labs stop releasing open weights",
            "effect_on_volunteer_compute": "Demand for non-commercial compute rises; second-hand datacenter GPUs (A100, H100) reach the secondary market at low prices, and a 'consumer' swarm becomes a prosumer swarm. The binding risk is model supply: a QLoRA swarm needs open base weights.",
            "what_to_do": "Pin and mirror the base weights the chain depends on (Granite, Qwen, Gemma) under their licences; support 24-40 GB prosumer cards in the trainer; keep the catalog honest so the project is credible when attention arrives.",
        },
        "S3_no_burst": {
            "assumption": "Capex and rents stay roughly where they are into 2027",
            "effect_on_volunteer_compute": "Status quo: volunteer compute competes on locality and purpose; the project's bottleneck remains the chain's quality (currently regressing) and recruitment, not the macro cycle.",
            "what_to_do": "Fix the chain (HM-S33-06) before scaling recruitment.",
        },
        "probability_note": "One analyst puts a severe credit unwind at 20%. That is an opinion; the project should plan for all three scenarios rather than bet on one.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "artifacts/experiments/CONSUMER_GPU_ECONOMICS_2026-11-07.json"))
    a = ap.parse_args()
    receipt = {
        "receipt": "CONSUMER_GPU_ECONOMICS",
        "date": "2026-11-07",
        "nature": "arithmetic on sourced parameters; not a measurement, not a forecast",
        "parameters": P,
        "supply": supply_scenarios(),
        "cost": cost_per_hour_and_token(),
        "bubble": bubble_scenarios(),
    }
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    c = receipt["cost"]["eu_avg"]
    s = receipt["supply"]
    print(f"electricity EUR/GPU-h (2070, EU avg) {c['electricity_eur_per_gpu_hour']}; EUR per 8h night {c['electricity_eur_per_8h_night_training']}; EUR per M tokens {c['electricity_eur_per_million_tokens_inference']}")
    print(f"100k volunteers @18h idle -> H100-equivalents (ratio 0.02): {s['100000']['h100_equivalent']['ratio_0.02']['h100_equivalent_gpus_24h']}")
    print(f"Steam share 8/12/16 GB: {s['steam_share_8_12_16gb']}; peak concurrent x share: {s['steam_peak_concurrent_times_share']:,}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
