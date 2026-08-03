"""Derivatives Time Machine: airline jet-fuel hedge audit.

This reproducible teaching model freezes a hedge recommendation on 2023-07-31,
audits it from 2023-08 through 2026-06, and produces a current 2026-07-27
one-month hedge ticket. Prices come from EIA, volatility from Cboe OVX, and the
one-month Treasury yield from FRED. Option values use Black-76 on a synthetic
one-month Brent future. Results are illustrative, before airline-specific
credit, tax, liquidity, and margin effects.

Run:
    python derivatives_time_machine.py --output-dir analysis_outputs

Dependencies: pandas, numpy, matplotlib
"""

from __future__ import annotations

import argparse
import io
import json
import math
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # Core calculations remain available without charts.
    plt = None


AS_OF_LOCK = pd.Timestamp("2023-07-31")
AUDIT_END = pd.Timestamp("2026-06-30")
CURRENT_AS_OF = pd.Timestamp("2026-07-27")
TRAINING_START = pd.Timestamp("2018-01-31")

MONTHLY_GALLONS = 10_000_000
CALL_OTM = 0.05
MIN_UNHEDGED = 0.20
MAX_FUTURES = 0.40
MIN_OPTION_SHARE = 0.30
GRID_STEP = 0.10
RISK_AVERSION = 0.75
FUTURES_FRICTION_PER_HEDGED_GALLON = 0.0010
OPTION_FRICTION_PER_HEDGED_GALLON = 0.0020

SOURCES = {
    "brent": "https://www.eia.gov/dnav/pet/hist/RBRTED.htm",
    "jet": "https://www.eia.gov/dnav/pet/hist/EER_EPJK_PF4_RGC_DPGD.htm",
    "ovx": "https://www.cboe.com/tradable_products/vix/vix_historical_data/",
    "rate": "https://fred.stlouisfed.org/series/DGS1MO",
    "cme": "https://www.cmegroup.com/markets/energy/crude-oil/brent-crude-oil-last-day.contractSpecs.html",
}


@dataclass(frozen=True)
class HedgeWeights:
    unhedged: float
    futures: float
    call: float
    collar: float


def _get_bytes(url: str, timeout: int = 90) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _eia_url(series: str, start: str = "2018-01-01", end: str = "2026-07-27") -> str:
    params = [
        ("api_key", "DEMO_KEY"),
        ("frequency", "daily"),
        ("data[0]", "value"),
        ("facets[series][]", series),
        ("start", start),
        ("end", end),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "asc"),
        ("offset", "0"),
        ("length", "5000"),
    ]
    return "https://api.eia.gov/v2/petroleum/pri/spt/data/?" + urllib.parse.urlencode(params)


def fetch_sources(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    downloads = {
        "eia_brent.json": _eia_url("RBRTE"),
        "eia_jet.json": _eia_url("EER_EPJK_PF4_RGC_DPG"),
        "ovx_daily.csv": "https://cdn.cboe.com/api/global/us_indices/daily_prices/OVX_History.csv",
        "dgs1mo.csv": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS1MO",
    }
    for name, url in downloads.items():
        (data_dir / name).write_bytes(_get_bytes(url))


def _read_eia(path: Path, value_name: str) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = payload["response"]["data"]
    frame = pd.DataFrame(rows)[["period", "value"]]
    frame["period"] = pd.to_datetime(frame["period"])
    frame[value_name] = pd.to_numeric(frame["value"], errors="coerce")
    return frame[["period", value_name]].dropna().set_index("period").sort_index()


def load_daily(data_dir: Path) -> pd.DataFrame:
    brent = _read_eia(data_dir / "eia_brent.json", "brent_bbl")
    jet = _read_eia(data_dir / "eia_jet.json", "jet_gal")

    ovx = pd.read_csv(data_dir / "ovx_daily.csv")
    ovx["DATE"] = pd.to_datetime(ovx["DATE"], format="%m/%d/%Y")
    ovx["ovx"] = pd.to_numeric(ovx["OVX"], errors="coerce")
    ovx = ovx.set_index("DATE")[["ovx"]].sort_index()

    rate = pd.read_csv(data_dir / "dgs1mo.csv")
    rate["observation_date"] = pd.to_datetime(rate["observation_date"])
    rate["rate_pct"] = pd.to_numeric(rate["DGS1MO"], errors="coerce")
    rate = rate.set_index("observation_date")[["rate_pct"]].sort_index()

    # EIA dates define executable observation days; market indicators are
    # carried forward only from information already published by that date.
    daily = brent.join(jet, how="inner").join(ovx, how="left").join(rate, how="left")
    daily[["ovx", "rate_pct"]] = daily[["ovx", "rate_pct"]].ffill()
    return daily.loc[:CURRENT_AS_OF].dropna()


def month_end_snapshots(daily: pd.DataFrame) -> pd.DataFrame:
    monthly = daily.resample("ME").last().copy()
    monthly["obs_date"] = daily.groupby(daily.index.to_period("M")).apply(lambda x: x.index.max()).values
    return monthly.dropna()


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black76(kind: str, future: float, strike: float, rate: float, years: float, sigma: float) -> float:
    if years <= 0 or sigma <= 0 or future <= 0 or strike <= 0:
        intrinsic = max(future - strike, 0.0) if kind == "call" else max(strike - future, 0.0)
        return math.exp(-rate * max(years, 0.0)) * intrinsic
    root_t = math.sqrt(years)
    d1 = (math.log(future / strike) + 0.5 * sigma * sigma * years) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    discount = math.exp(-rate * years)
    if kind == "call":
        return discount * (future * normal_cdf(d1) - strike * normal_cdf(d2))
    return discount * (strike * normal_cdf(-d2) - future * normal_cdf(-d1))


def zero_cost_put_strike(future: float, call_strike: float, rate: float, years: float, sigma: float) -> float:
    target = black76("call", future, call_strike, rate, years, sigma)
    low, high = 0.25 * future, future
    for _ in range(90):
        mid = 0.5 * (low + high)
        put = black76("put", future, mid, rate, years, sigma)
        if put < target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def estimate_beta(monthly: pd.DataFrame, end_date: pd.Timestamp, window: int | None = None) -> float:
    sample = monthly.loc[:end_date, ["brent_bbl", "jet_gal"]].copy()
    if window:
        sample = sample.tail(window + 1)
    changes = sample.diff().dropna()
    x = changes["brent_bbl"] / 42.0
    y = changes["jet_gal"]
    beta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    return float(np.clip(beta, 0.0, 1.5))


def build_periods(monthly: pd.DataFrame, beta: float) -> pd.DataFrame:
    rows: list[dict[str, float | pd.Timestamp]] = []
    for i in range(len(monthly) - 1):
        start = monthly.iloc[i]
        end = monthly.iloc[i + 1]
        start_date = monthly.index[i]
        end_date = monthly.index[i + 1]
        years = max((end["obs_date"] - start["obs_date"]).days / 365.0, 1.0 / 365.0)
        rate = max(float(start["rate_pct"]) / 100.0, 0.0)
        sigma = float(np.clip(float(start["ovx"]) / 100.0, 0.10, 1.25))
        spot = float(start["brent_bbl"])
        future = spot * math.exp(rate * years)
        call_strike = future * (1.0 + CALL_OTM)
        put_strike = zero_cost_put_strike(future, call_strike, rate, years, sigma)
        call_premium = black76("call", future, call_strike, rate, years, sigma)
        put_premium = black76("put", future, put_strike, rate, years, sigma)
        settle = float(end["brent_bbl"])
        call_payoff = max(settle - call_strike, 0.0)
        put_payoff = max(put_strike - settle, 0.0)
        unhedged = float(end["jet_gal"])
        futures = unhedged - beta * (settle - future) / 42.0 + beta * FUTURES_FRICTION_PER_HEDGED_GALLON
        call = unhedged - beta * call_payoff / 42.0 + beta * call_premium / 42.0 + beta * OPTION_FRICTION_PER_HEDGED_GALLON
        collar = unhedged - beta * (call_payoff - put_payoff) / 42.0 + beta * OPTION_FRICTION_PER_HEDGED_GALLON
        rows.append(
            {
                "start_date": start_date,
                "end_date": end_date,
                "start_obs": start["obs_date"],
                "end_obs": end["obs_date"],
                "brent_start": spot,
                "brent_end": settle,
                "jet_start": float(start["jet_gal"]),
                "jet_end": unhedged,
                "ovx": float(start["ovx"]),
                "rate_pct": float(start["rate_pct"]),
                "future": future,
                "call_strike": call_strike,
                "put_strike": put_strike,
                "call_premium": call_premium,
                "put_premium": put_premium,
                "unhedged": unhedged,
                "futures": futures,
                "call": call,
                "collar": collar,
            }
        )
    return pd.DataFrame(rows).set_index("end_date")


def cvar95(costs: pd.Series) -> float:
    threshold = costs.quantile(0.95)
    return float(costs[costs >= threshold].mean())


def optimize_weights(training: pd.DataFrame) -> HedgeWeights:
    best: tuple[float, HedgeWeights] | None = None
    steps = int(round(1.0 / GRID_STEP))
    for iu in range(int(round(MIN_UNHEDGED / GRID_STEP)), steps + 1):
        for i_f in range(steps - iu + 1):
            for i_c in range(steps - iu - i_f + 1):
                i_col = steps - iu - i_f - i_c
                w = HedgeWeights(iu / steps, i_f / steps, i_c / steps, i_col / steps)
                # Airline risk-policy guardrails prevent the optimizer from
                # collapsing into a linear all-futures hedge. At least 30% of
                # fuel volume retains option convexity and futures are capped
                # at 40%; these constraints are fixed before the audit.
                if w.futures > MAX_FUTURES + 1e-12:
                    continue
                if w.call + w.collar < MIN_OPTION_SHARE - 1e-12:
                    continue
                costs = (
                    w.unhedged * training["unhedged"]
                    + w.futures * training["futures"]
                    + w.call * training["call"]
                    + w.collar * training["collar"]
                )
                mean = float(costs.mean())
                objective = mean + RISK_AVERSION * (cvar95(costs) - mean)
                if best is None or objective < best[0] - 1e-12:
                    best = (objective, w)
    assert best is not None
    return best[1]


def apply_hybrid(periods: pd.DataFrame, weights: HedgeWeights) -> pd.DataFrame:
    result = periods.copy()
    result["hybrid"] = (
        weights.unhedged * result["unhedged"]
        + weights.futures * result["futures"]
        + weights.call * result["call"]
        + weights.collar * result["collar"]
    )
    return result


def metrics_for(costs: pd.Series, unhedged: pd.Series) -> dict[str, float]:
    total_cost = float(costs.sum() * MONTHLY_GALLONS)
    unhedged_total = float(unhedged.sum() * MONTHLY_GALLONS)
    variance_base = float(unhedged.var(ddof=1))
    variance = float(costs.var(ddof=1))
    return {
        "total_cost_usd": total_cost,
        "savings_vs_unhedged_usd": unhedged_total - total_cost,
        "mean_cost_per_gal": float(costs.mean()),
        "worst_month_cost_per_gal": float(costs.max()),
        "cvar95_cost_per_gal": cvar95(costs),
        "cost_volatility_per_gal": float(costs.std(ddof=1)),
        "hedge_efficiency": 1.0 - variance / variance_base if variance_base > 0 else float("nan"),
    }


def current_ticket(monthly: pd.DataFrame, current_beta: float, current_weights: HedgeWeights) -> dict[str, float | str]:
    current = monthly.loc[monthly["obs_date"] <= CURRENT_AS_OF].iloc[-1]
    obs_date = pd.Timestamp(current["obs_date"])
    years = 31.0 / 365.0
    rate = float(current["rate_pct"]) / 100.0
    sigma = float(current["ovx"]) / 100.0
    spot = float(current["brent_bbl"])
    future = spot * math.exp(rate * years)
    call_strike = future * (1.0 + CALL_OTM)
    put_strike = zero_cost_put_strike(future, call_strike, rate, years, sigma)
    call_premium = black76("call", future, call_strike, rate, years, sigma)
    hedge_fraction = 1.0 - current_weights.unhedged
    notional_gallons = MONTHLY_GALLONS * current_beta * hedge_fraction
    futures_gallons = MONTHLY_GALLONS * current_beta * current_weights.futures
    call_gallons = MONTHLY_GALLONS * current_beta * current_weights.call
    collar_gallons = MONTHLY_GALLONS * current_beta * current_weights.collar
    return {
        "as_of": obs_date.strftime("%Y-%m-%d"),
        "brent_spot_usd_bbl": spot,
        "jet_spot_usd_gal": float(current["jet_gal"]),
        "ovx_pct": float(current["ovx"]),
        "rate_pct": float(current["rate_pct"]),
        "synthetic_one_month_future": future,
        "call_strike": call_strike,
        "zero_cost_put_strike": put_strike,
        "call_premium_usd_bbl": call_premium,
        "cross_hedge_beta": current_beta,
        "economic_hedge_fraction": hedge_fraction,
        "effective_hedged_gallons": notional_gallons,
        "futures_gallons": futures_gallons,
        "call_gallons": call_gallons,
        "collar_gallons": collar_gallons,
        "futures_contracts_1000_bbl": round(futures_gallons / 42.0 / 1_000.0),
        "long_call_contracts_1000_bbl": round(call_gallons / 42.0 / 1_000.0),
        "collar_contracts_1000_bbl": round(collar_gallons / 42.0 / 1_000.0),
        "recommendation": "HEDGE" if hedge_fraction >= 0.50 else "PARTIAL HEDGE",
    }


def create_charts(audit: pd.DataFrame, output_dir: Path) -> None:
    if plt is None:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"unhedged": "#A6A6A6", "futures": "#2F5597", "call": "#70AD47", "collar": "#ED7D31", "hybrid": "#00A6A6"}

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    for name in ["unhedged", "hybrid"]:
        ax.plot(audit.index, audit[name], label=name.title(), linewidth=2.4, color=colors[name])
    ax.set_title("Monthly jet-fuel cost: locked 2023 hedge vs. unhedged")
    ax.set_ylabel("Net cost ($/gallon)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "monthly_cost_path.png", bbox_inches="tight")
    plt.close(fig)

    strategies = ["unhedged", "futures", "call", "collar", "hybrid"]
    savings = [(audit["unhedged"].sum() - audit[s].sum()) * MONTHLY_GALLONS / 1_000_000 for s in strategies]
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    bars = ax.bar([s.title() for s in strategies], savings, color=[colors[s] for s in strategies])
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_title("Cumulative savings versus unhedged")
    ax.set_ylabel("Savings ($ millions)")
    ax.bar_label(bars, fmt="%.1f", padding=3)
    fig.tight_layout()
    fig.savefig(output_dir / "strategy_savings.png", bbox_inches="tight")
    plt.close(fig)


def run(data_dir: Path, output_dir: Path, fetch: bool) -> dict:
    if fetch:
        fetch_sources(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily = load_daily(data_dir)
    monthly = month_end_snapshots(daily)

    locked_beta = estimate_beta(monthly, AS_OF_LOCK)
    locked_periods = build_periods(monthly, locked_beta)
    training = locked_periods.loc[(locked_periods.index >= TRAINING_START) & (locked_periods.index <= AS_OF_LOCK)]
    locked_weights = optimize_weights(training)
    locked_periods = apply_hybrid(locked_periods, locked_weights)
    audit = locked_periods.loc[(locked_periods.index > AS_OF_LOCK) & (locked_periods.index <= AUDIT_END)].copy()

    names = ["unhedged", "futures", "call", "collar", "hybrid"]
    scorecard = {name: metrics_for(audit[name], audit["unhedged"]) for name in names}
    shock = audit.loc[audit.index >= pd.Timestamp("2026-01-31")]
    shock_scorecard = {name: metrics_for(shock[name], shock["unhedged"]) for name in names}
    monthly_savings = (audit["unhedged"] - audit["hybrid"]) * MONTHLY_GALLONS
    peak_month = monthly_savings.idxmax()
    worst_tradeoff_month = monthly_savings.idxmin()
    full_savings_positive = scorecard["hybrid"]["savings_vs_unhedged_usd"] > 0
    risk_reduced = scorecard["hybrid"]["hedge_efficiency"] > 0
    shock_savings_positive = shock_scorecard["hybrid"]["savings_vs_unhedged_usd"] > 0
    if full_savings_positive and risk_reduced:
        decision = "CORRECT"
    elif risk_reduced and shock_savings_positive:
        decision = "PARTIALLY CORRECT"
    else:
        decision = "NOT CORRECT"

    # The current refresh uses only data known by the current snapshot date.
    current_beta = estimate_beta(monthly, monthly.index[-1], window=60)
    refreshed_periods = build_periods(monthly, current_beta)
    refreshed_training = refreshed_periods.loc[refreshed_periods.index <= AUDIT_END]
    current_weights = optimize_weights(refreshed_training.tail(60))
    ticket = current_ticket(monthly, current_beta, current_weights)

    summary = {
        "project": "The Derivatives Time Machine",
        "question": "Could a Gulf airline's 2023 Brent hedge have protected it from the 2026 oil shock?",
        "model_scope": {
            "lock_date": AS_OF_LOCK.strftime("%Y-%m-%d"),
            "audit_start": audit.index.min().strftime("%Y-%m-%d"),
            "audit_end": audit.index.max().strftime("%Y-%m-%d"),
            "audit_months": int(len(audit)),
            "monthly_gallons": MONTHLY_GALLONS,
            "training_start": TRAINING_START.strftime("%Y-%m-%d"),
        },
        "locked_2023": {
            "cross_hedge_beta": locked_beta,
            "weights": asdict(locked_weights),
            "decision": decision,
            "decision_rule": "Correct if full-window savings and variance reduction are positive; partially correct if it reduced variance and saved money during the 2026 shock but not over the full window.",
        },
        "scorecard": scorecard,
        "shock_2026_scorecard": shock_scorecard,
        "headline_findings": {
            "full_window_savings_usd": scorecard["hybrid"]["savings_vs_unhedged_usd"],
            "shock_2026_savings_usd": shock_scorecard["hybrid"]["savings_vs_unhedged_usd"],
            "variance_reduction_pct": scorecard["hybrid"]["hedge_efficiency"] * 100.0,
            "cvar95_reduction_usd_per_gallon": scorecard["unhedged"]["cvar95_cost_per_gal"] - scorecard["hybrid"]["cvar95_cost_per_gal"],
            "peak_protection_month": peak_month.strftime("%Y-%m-%d"),
            "peak_protection_usd": float(monthly_savings.loc[peak_month]),
            "largest_tradeoff_month": worst_tradeoff_month.strftime("%Y-%m-%d"),
            "largest_tradeoff_usd": float(monthly_savings.loc[worst_tradeoff_month]),
        },
        "current_ticket": ticket,
        "current_weights": asdict(current_weights),
        "assumptions": {
            "instrument": "Synthetic one-month Brent future; Black-76 options",
            "volatility_proxy": "Cboe OVX, clipped only to 10%-125% for numerical stability",
            "call_moneyness": CALL_OTM,
            "minimum_unhedged_share": MIN_UNHEDGED,
            "maximum_futures_share": MAX_FUTURES,
            "minimum_option_share": MIN_OPTION_SHARE,
            "weight_grid_step": GRID_STEP,
            "risk_aversion": RISK_AVERSION,
            "futures_friction_usd_per_hedged_gallon": FUTURES_FRICTION_PER_HEDGED_GALLON,
            "option_friction_usd_per_hedged_gallon": OPTION_FRICTION_PER_HEDGED_GALLON,
        },
        "sources": SOURCES,
        "limitations": [
            "Public EIA spot prices are used to settle a synthetic one-month Brent future; historical exchange settlement data are not claimed.",
            "OVX is an options-implied volatility proxy on USO, not a Brent volatility surface.",
            "The hypothetical airline consumes a constant 10 million gallons per month.",
            "Results are illustrative and exclude credit, margin, tax, liquidity, and airline-specific procurement effects.",
        ],
    }

    audit_export = audit.reset_index().copy()
    for col in ["start_date", "end_date", "start_obs", "end_obs"]:
        if col in audit_export:
            audit_export[col] = pd.to_datetime(audit_export[col]).dt.strftime("%Y-%m-%d")
    audit_export.to_csv(output_dir / "audit_monthly.csv", index=False)
    monthly_export = monthly.reset_index(names="month_end")
    monthly_export["month_end"] = monthly_export["month_end"].dt.strftime("%Y-%m-%d")
    monthly_export["obs_date"] = pd.to_datetime(monthly_export["obs_date"]).dt.strftime("%Y-%m-%d")
    monthly_export.to_csv(output_dir / "monthly_market_data.csv", index=False)
    (output_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    create_charts(audit, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs"))
    parser.add_argument("--no-fetch", action="store_true", help="Use files already present in --data-dir.")
    args = parser.parse_args()
    summary = run(args.data_dir, args.output_dir, fetch=not args.no_fetch)
    hybrid = summary["scorecard"]["hybrid"]
    print(json.dumps({
        "locked_decision": summary["locked_2023"]["decision"],
        "hybrid_savings_usd": round(hybrid["savings_vs_unhedged_usd"], 2),
        "hybrid_hedge_efficiency": round(hybrid["hedge_efficiency"], 4),
        "current_recommendation": summary["current_ticket"]["recommendation"],
    }, indent=2))


if __name__ == "__main__":
    main()
