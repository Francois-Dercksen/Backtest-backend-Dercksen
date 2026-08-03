from flask import Flask, jsonify

def create_app():
    app = Flask(__name__)

    @app.route("/")
    def health_check():
        return jsonify({"status": "ok", "service": "dercksen-backtest-backend"})

    @app.route("/api/backtest", methods=["POST"])
    def run_backtest():
        # Placeholder: actual backtest_engine.py logic will be wired in here
        return jsonify({"message": "Backtest endpoint placeholder — not yet implemented"}), 501

    return app
