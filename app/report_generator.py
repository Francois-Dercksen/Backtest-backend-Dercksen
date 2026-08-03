"""
report_generator.py

Builds the HTML dashboard by injecting JSON data into the placeholder
tokens inside the hedge report template. Field names match exactly
what backtest_report_hedge_template.html's JS expects (see STATS
lookups via getStat(section, metric) and RESULTS.map(r => r.<field>)).
"""

import json
import math
import os

import numpy as np
import pandas as pd

TEMPLATE_PATH = os.path.join("Input", "backtest_report_hedge_template.html")


def _clean_val(v):
    if hasattr(v, "isoformat"):
        return v.isoformat()[:10]
    if hasattr(v, "strftime") and not hasattr(v, "isoformat"):
        return str(v)
    if type(v).__module__ == "numpy":
        item = v.item()
        return None if isinstance(item, float) and math.isnan(item) else item
    if isinstance(v, bool):
        return v
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def sanitise_for_json(records):
    return [{k: _clean_val(v) for k, v in row.items()} for row in records]


def build_period_records(results_df):
    """
    Field names must match the template's RESULTS.map(r => r.<field>)
    usage exactly: period, portfolio_return_adjusted, spx_return,
    exercised, premium_paid_pct_initial, payoff_pct_initial,
    net_hedge_pnl_pct_port, hedged_return, cumulative_return,
    hedged_cumulative_return, n_holdings.
    """
    cols = [
        "period", "date", "portfolio_return_adjusted", "spx_return",
        "exercised", "premium_paid_pct_initial", "payoff_pct_initial",
        "net_hedge_pnl_pct_port", "hedged_return", "cumulative_return",
        "hedged_cumulative_return", "n_holdings",
    ]
    records = results_df[[c for c in cols if c in results_df.columns]].to_dict(orient="records")
    return records


def build_stats_records(metrics, frequency, risk_frequency, benchmark_ticker):
    """
    metrics is expected to be {"unhedged": {...}, "hedged": {...}, "put_stats": {...}}.
    Rows must match template's getStat(section, metric) lookups:
    section in {"Info", "Unhedged", "Hedged", "Put/Hedge Stats"}.
    """
    rows = [
        {"section": "Info", "metric": "frequency", "value": frequency, "description": ""},
        {"section": "Info", "metric": "risk_frequency", "value": risk_frequency, "description": ""},
        {"section": "Info", "metric": "OTM", "value": 0.0, "description": "Long-only, no hedge overlay"},
        {"section": "Info", "metric": "Beta", "value": None, "description": ""},
    ]

    metric_keys = [
        "annualised_return", "total_cumulative", "sharpe", "sortino",
        "max_drawdown", "calmar", "var_95", "hit_rate", "std_period_return",
        "best_rolling_cagr", "worst_rolling_cagr",
        "best_period", "best_period_return", "worst_period", "worst_period_return",
    ]

    for section_label, key in [("Unhedged", "unhedged"), ("Hedged", "hedged")]:
        m = metrics.get(key, {}) or {}
        for k in metric_keys:
            rows.append({"section": section_label, "metric": k, "value": m.get(k), "description": ""})
        bm = m.get("benchmark")
        if bm:
            for k, v in bm.items():
                rows.append({"section": f"{section_label} vs {benchmark_ticker}", "metric": k, "value": v, "description": ""})

    put_stats = metrics.get("put_stats", {}) or {}
    for k, v in put_stats.items():
        rows.append({"section": "Put/Hedge Stats", "metric": k, "value": v, "description": ""})

    return rows


def generate_report_html(
    results_df: pd.DataFrame,
    metrics: dict,
    frequency: str,
    risk_frequency: str,
    benchmark_ticker: str = "SPX",
    template_path: str = TEMPLATE_PATH,
):
    results_df = results_df.copy()
    if "period" in results_df.columns:
        results_df["period"] = results_df["period"].astype(str)
        results_df = results_df.sort_values("period").reset_index(drop=True)

    stats_records = sanitise_for_json(build_stats_records(metrics, frequency, risk_frequency, benchmark_ticker))
    period_records = sanitise_for_json(build_period_records(results_df))
    missing_records = []

    stats_js = json.dumps(stats_records, ensure_ascii=False)
    periods_js = json.dumps(period_records, ensure_ascii=False)
    missing_js = json.dumps(missing_records, ensure_ascii=False)

    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()

    html = html.replace("__STATS_JSON__", stats_js)
    html = html.replace("__PERIODS_JSON__", periods_js)
    html = html.replace("__MISSING_JSON__", missing_js)
    html = html.replace("__BENCH__", benchmark_ticker)
    html = html.replace("__FREQUENCY__", frequency)
    html = html.replace("__RISK_FREQUENCY__", risk_frequency.capitalize())

    return html
