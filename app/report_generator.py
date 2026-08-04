"""
report_generator.py

Builds the self-contained HTML dashboard for a backtest run.
Consumes the modular {"net": {...}, "legs": {...}} structure produced by
backtest_engine.run_backtest(), so any combination of legs renders correctly
without special-casing per strategy.
"""

import json
import math
import os

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "report_template.html")


def sanitise_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitise_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitise_for_json(v) for v in obj]
    if hasattr(obj, "isoformat"):
        return str(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if hasattr(obj, "item"):  # numpy scalar
        item = obj.item()
        if isinstance(item, float) and (math.isnan(item) or math.isinf(item)):
            return None
        return item
    return obj


def generate_report_html(net_results_df, net_metrics, legs, frequency="annual",
                          risk_frequency="monthly", benchmark_ticker="SPX"):
    net_df = net_results_df.copy()
    net_df["period"] = net_df["period"].astype(str)
    net_records = sanitise_for_json(net_df.to_dict(orient="records"))
    net_metrics_clean = sanitise_for_json(net_metrics)

    legs_payload = {}
    for leg_key, leg_data in legs.items():
        leg_df = leg_data["results_df"].copy()
        leg_df["period"] = leg_df["period"].astype(str)
        legs_payload[leg_key] = {
            "label": leg_data["label"],
            "type": leg_data["type"],
            "metrics": sanitise_for_json(leg_data["metrics"]),
            "periods": sanitise_for_json(leg_df.to_dict(orient="records")),
        }

    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    html = html.replace("__NET_PERIODS_JSON__", json.dumps(net_records, ensure_ascii=False))
    html = html.replace("__NET_METRICS_JSON__", json.dumps(net_metrics_clean, ensure_ascii=False))
    html = html.replace("__LEGS_JSON__", json.dumps(legs_payload, ensure_ascii=False))
    html = html.replace("__FREQUENCY__", frequency)
    html = html.replace("__RISK_FREQUENCY__", risk_frequency)
    html = html.replace("__BENCHMARK_TICKER__", benchmark_ticker)

    return html
