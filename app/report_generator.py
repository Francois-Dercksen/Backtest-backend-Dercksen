"""
report_generator.py

Ported from H2_report_hedge-2.py. Builds an HTML dashboard by injecting
JSON data into the placeholder tokens inside the report template.

For long-only backtests, period records are mapped onto the same
field names the hedge template's JS expects (port_return, spx_return,
exercised, premium_paid, payoff, net_hedge_pnl, hedged_return,
unhedged_cumulative_return, hedged_cumulative_return, holdings) with
hedge-specific fields neutralised to zero/False since there is no
hedge overlay in long-only mode.
"""

import json
import math
import os

import numpy as np
import pandas as pd

TEMPLATE_PATH = os.path.join("Input", "backtest_report_hedge_template.html")


def sanitise_for_json(records):
    clean = []
    for row in records:
        new_row = {}
        for k, v in row.items():
            if hasattr(v, "isoformat"):
                new_row[k] = v.isoformat()[:10]
            elif hasattr(v, "strftime") and not hasattr(v, "isoformat"):
                new_row[k] = str(v)
            elif type(v).__module__ == "numpy":
                item = v.item()
                new_row[k] = None if isinstance(item, float) and math.isnan(item) else item
            elif isinstance(v, bool):
                new_row[k] = v
            elif isinstance(v, float) and math.isnan(v):
                new_row[k] = None
            else:
                new_row[k] = v
        clean.append(new_row)
    return clean


def build_period_records_for_template(results_df):
    """
    Maps long-only results_df columns onto the exact field names the
    hedge report template's JS expects. Hedge-specific fields are
    neutralised (0 / False) since long-only has no option overlay.
    """
    mapped = []
    for _, row in results_df.iterrows():
        port_ret = row.get("portfolio_return_adjusted", None)
        cum_ret = row.get("cumulative_return", None)
        mapped.append({
            "period": str(row.get("period", "")),
            "date": row.get("date", ""),
            "portfolio_return_adjusted": port_ret,
            "spx_return": None,
            "exercised": False,
            "premium_paid": 0.0,
            "payoff": 0.0,
            "net_hedge_pnl": 0.0,
            "hedged_return": port_ret,
            "unhedged_cumulative_return": cum_ret,
            "hedged_cumulative_return": cum_ret,
            "holdings": row.get("n_holdings", None),
        })
    return mapped


def build_stats_records(metrics, frequency, risk_frequency, benchmark_ticker):
    rows = [
        {"section": "Info", "metric": "frequency", "value": frequency, "description": ""},
        {"section": "Info", "metric": "risk_frequency", "value": risk_frequency, "description": ""},
        {"section": "Info", "metric": "OTM", "value": 0.0, "description": "Long-only, no hedge overlay"},
        {"section": "Info", "metric": "Beta", "value": None, "description": ""},
    ]

    for section_label in ["Unhedged", "Hedged"]:
        for k in ["annualised_return", "total_cumulative", "sharpe", "sortino",
                  "max_drawdown", "calmar", "var95", "hit_rate",
                  "best_rolling_cagr", "worst_rolling_cagr"]:
            rows.append({"section": section_label, "metric": k, "value": metrics.get(k), "description": ""})

        bm = metrics.get("benchmark")
        if bm:
            for k, v in bm.items():
                rows.append({"section": f"{section_label} vs {benchmark_ticker}", "metric": k, "value": v, "description": ""})

    # Put/Hedge stats placeholder — zeroed for long-only
    for k in ["total_premium_paid", "total_payoff", "net_hedge_pnl", "n_exercised"]:
        rows.append({"section": "Put/Hedge Stats", "metric": k, "value": 0.0, "description": "No hedge overlay (long-only)"})

    return rows


def generate_report_html(
    results_df: pd.DataFrame,
    metrics: dict,
    frequency: str,
    risk_frequency: str,
    benchmark_ticker: str = "SPX",
    template_path: str = TEMPLATE_PATH,
):
    """
    Returns the fully rendered HTML report as a string.
    """
    results_df = results_df.copy()
    if "period" in results_df.columns:
        results_df["period"] = results_df["period"].astype(str)
        results_df = results_df.sort_values("period").reset_index(drop=True)

    stats_records = sanitise_for_json(build_stats_records(metrics, frequency, risk_frequency, benchmark_ticker))
    period_records = sanitise_for_json(build_period_records_for_template(results_df))
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
