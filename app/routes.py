from flask import Flask, jsonify, request
from flask_cors import CORS
import math

from app.backtest_engine import run_backtest
from app.report_generator import generate_report_html


def sanitise(d):
    """Recursively replace NaN/Inf with None so the payload is valid JSON."""
    if d is None:
        return None
    if isinstance(d, dict):
        return {k: sanitise(v) for k, v in d.items()}
    if isinstance(d, list):
        return [sanitise(v) for v in d]
    if isinstance(d, float) and (math.isnan(d) or math.isinf(d)):
        return None
    return d


def parse_pct(val, default=0.0):
    if val is None or val == "":
        return default
    try:
        return float(str(val).replace("%", "").strip()) / 100.0
    except ValueError:
        return default


def parse_float(val, default=0.0):
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# Each parser turns the raw, frontend-friendly params for a leg (typically
# strings like "90%") into the floats run_backtest()'s leg builders expect.
LEG_PARAM_PARSERS = {
    "long_portfolio": lambda p: {
        "weight_pct": parse_pct(p.get("weight"), default=1.0),
    },
    "short_portfolio": lambda p: {
        "weight_pct": parse_pct(p.get("weight"), default=1.0),
    },
    "long_put": lambda p: {
        "otm_pct": 1 - parse_pct(p.get("strike"), default=0.90),
        "premium_pct": parse_pct(p.get("premium"), default=0.0419),
        "notional_method": p.get("notional_method", "real-beta"),
        "custom_beta": (
            parse_float(p.get("custom_beta"), default=1.0)
            if p.get("notional_method") == "custom" else None
        ),
    },
    "long_call": lambda p: {
        "strike_pct": parse_pct(p.get("strike"), default=1.00),
        "premium_pct": parse_pct(p.get("premium"), default=0.0419),
        "sizing_mode": p.get("sizing_mode", "fixed"),
        "fixed_pct": (
            parse_pct(p.get("fixed_pct"), default=0.10)
            if p.get("sizing_mode", "fixed") == "fixed" else None
        ),
        "weight_pct": (
            parse_pct(p.get("weight_pct"), default=0.10)
            if p.get("sizing_mode") == "weight" else None
        ),
    },
    "short_call": lambda p: {
        "strike_pct": parse_pct(p.get("strike"), default=1.00),
        "premium_pct": parse_pct(p.get("premium"), default=0.04),
        "notional_pct": parse_pct(p.get("notional"), default=1.00),
    },
    "short_put": lambda p: {
        "strike_pct": parse_pct(p.get("strike"), default=0.90),
        "premium_pct": parse_pct(p.get("premium"), default=0.0419),
        "notional_pct": parse_pct(p.get("notional"), default=1.00),
    },
}

LEG_LABELS = {
    "long_portfolio": "Long Portfolio",
    "short_portfolio": "Short Portfolio",
    "long_put": "Long Put",
    "long_call": "Long Call",
    "short_call": "Short Call",
    "short_put": "Short Put",
}


def build_legs_from_payload(raw_legs):
    """
    raw_legs: list of {"key": str, "type": str, "enabled": bool, "label": str, "params": {...}}
    Only enabled legs of a recognised type are included. Any strategy
    combination is valid — the engine has no special-casing per combination.
    """
    legs = []
    for leg in raw_legs or []:
        if not leg.get("enabled"):
            continue
        leg_type = leg.get("type")
        parser = LEG_PARAM_PARSERS.get(leg_type)
        if parser is None:
            continue
        raw_params = leg.get("params", {}) or {}
        parsed_params = {k: v for k, v in parser(raw_params).items() if v is not None}
        legs.append({
            "key": leg.get("key", leg_type),
            "type": leg_type,
            "label": leg.get("label", LEG_LABELS.get(leg_type, leg_type)),
            "params": parsed_params,
        })
    return legs


def create_app():
    app = Flask(__name__)
    CORS(app)  # allow the Cloudflare Pages frontend (different origin) to call this API

    @app.route("/")
    def health_check():
        return jsonify({"status": "ok", "service": "dercksen-backtest-backend"})

    @app.route("/api/backtest", methods=["POST"])
    def run_backtest_route():
        payload = request.get_json(force=True) or {}

        try:
            start_date = payload.get("start_date", "2005-01")
            end_date = payload.get("end_date", "2025-12")
            start_year, start_month = [int(x) for x in start_date.split("-")]
            end_year, end_month = [int(x) for x in end_date.split("-")]

            risk_free_rate = parse_pct(payload.get("risk_free_rate"), default=0.045)

            legs = build_legs_from_payload(payload.get("legs"))
            if not legs:
                return jsonify({"error": "Select at least one strategy leg to run a backtest."}), 400

            result = run_backtest(
                start_year=start_year,
                start_month=start_month,
                end_year=end_year,
                end_month=end_month,
                legs=legs,
                risk_free_rate=risk_free_rate,
            )

            net_results_df = result["net"]["results_df"]
            net_metrics = sanitise(result["net"]["metrics"])

            legs_payload = {}
            for leg_key, leg_data in result["legs"].items():
                leg_df = leg_data["results_df"]
                legs_payload[leg_key] = {
                    "label": leg_data["label"],
                    "type": leg_data["type"],
                    "metrics": sanitise(leg_data["metrics"]),
                    "periods": sanitise(
                        leg_df.assign(period=leg_df["period"].astype(str)).to_dict(orient="records")
                    ),
                }

            net_periods = sanitise(
                net_results_df.assign(period=net_results_df["period"].astype(str)).to_dict(orient="records")
            )

            html = None
            try:
                html = generate_report_html(
                    net_results_df=net_results_df,
                    net_metrics=result["net"]["metrics"],
                    legs=result["legs"],
                    frequency="annual",
                    risk_frequency="monthly",
                    benchmark_ticker="SPX",
                )
            except Exception:
                # report_generator.py has not yet been updated for the legs
                # structure -- fall back to raw JSON only, don't fail the request.
                html = None

            return jsonify({
                "name": payload.get("name", "Untitled Backtest"),
                "net": {
                    "metrics": net_metrics,
                    "periods": net_periods,
                },
                "legs": legs_payload,
                "dashboard_html": html,
            })

        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/portfolios", methods=["GET"])
    def list_portfolios():
        # Placeholder: hardcoded until Data tab uploads are wired to a real store
        return jsonify({"portfolios": ["BAM_f7_default"]})

    return app
