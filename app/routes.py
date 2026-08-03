from flask import Flask, jsonify, request
from flask_cors import CORS
import math

from app.backtest_engine import run_long_only_backtest
from app.report_generator import generate_report_html


def sanitise_metrics(d):
    if d is None:
        return None
    clean = {}
    for k, v in d.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            clean[k] = None
        elif isinstance(v, dict):
            clean[k] = sanitise_metrics(v)
        else:
            clean[k] = v
    return clean


def create_app():
    app = Flask(__name__)
    CORS(app)

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
            weight_pct = float(str(weight_str).replace("%", "").strip()) / 100.0

            results_df, metrics = run_long_only_backtest(
                start_year=start_year,
                start_month=start_month,
                end_year=end_year,
                end_month=end_month,
                weight_pct=weight_pct,
            )

            html = generate_report_html(
                results_df=results_df,
                metrics=metrics,
                frequency="annual",
                risk_frequency="monthly",
                benchmark_ticker="SPX",
            )

            clean_metrics = sanitise_metrics({k: v for k, v in metrics.items() if k != "benchmark"})
            clean_benchmark = sanitise_metrics(metrics.get("benchmark"))

            return jsonify({
                "name": payload.get("name", "Untitled Backtest"),
                "metrics": clean_metrics,
                "benchmark": clean_benchmark,
                "dashboard_html": html,
            })

        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/portfolios", methods=["GET"])
    def list_portfolios():
        return jsonify({"portfolios": ["BAM_f7_default"]})

    return app
