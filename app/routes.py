from flask import Flask, jsonify, request
from flask_cors import CORS
import math

from app.backtest_engine import run_long_only_backtest
from app.report_generator import generate_report_html


def sanitise_metrics(d):
    if d is None:
        return None
    if isinstance(d, dict):
        return {k: sanitise_metrics(v) for k, v in d.items()}
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


def create_app():
    app = Flask(__name__)
    CORS(app)  # allow the Cloudflare Pages frontend (different origin) to call this API

    @app.route("/")
    def health_check():
        return jsonify({"status": "ok", "service": "dercksen-backtest-backend"})

    @app.route("/api/backtest", methods=["POST"])
    def run_backtest():
        payload = request.get_json(force=True) or {}

        try:
            start_date = payload.get("start_date", "2005-01")
            end_date = payload.get("end_date", "2025-12")
            start_year, start_month = [int(x) for x in start_date.split("-")]
            end_year, end_month = [int(x) for x in end_date.split("-")]

            weight_str = payload.get("weight", "100%")
            weight_pct = parse_pct(weight_str, default=1.0)

            long_put_payload = payload.get("long_put")
            long_put = None
            if long_put_payload:
                strike_pct = parse_pct(long_put_payload.get("strike"), default=0.90)
                otm_pct = 1 - strike_pct
                premium_pct = parse_pct(long_put_payload.get("premium"), default=0.0419)
                notional_method = long_put_payload.get("notional_method", "real-beta")

                custom_beta = None
                if notional_method == "custom":
                    try:
                        custom_beta = float(long_put_payload.get("custom_beta"))
                    except (TypeError, ValueError):
                        custom_beta = 1.0

                long_put = {
                    "otm_pct": otm_pct,
                    "premium_pct": premium_pct,
                    "notional_method": notional_method,
                    "custom_beta": custom_beta,
                }

            results_df, metrics = run_long_only_backtest(
                start_year=start_year,
                start_month=start_month,
                end_year=end_year,
                end_month=end_month,
                weight_pct=weight_pct,
                long_put=long_put,
            )

            html = generate_report_html(
                results_df=results_df,
                metrics=metrics,
                frequency="annual",
                risk_frequency="monthly",
                benchmark_ticker="SPX",
            )

            clean_metrics = sanitise_metrics(metrics)
            primary_key = "hedged" if long_put else "unhedged"
            primary_metrics = clean_metrics.get(primary_key) or {}

            return jsonify({
                "name": payload.get("name", "Untitled Backtest"),
                "metrics": primary_metrics,
                "benchmark": primary_metrics.get("benchmark"),
                "put_stats": clean_metrics.get("put_stats"),
                "dashboard_html": html,
            })

        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/portfolios", methods=["GET"])
    def list_portfolios():
        # Placeholder: hardcoded until Data tab uploads are wired to a real store
        return jsonify({"portfolios": ["BAM_f7_default"]})

    return app
