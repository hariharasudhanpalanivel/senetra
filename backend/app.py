from flask import Flask, jsonify

from mlops.extension import init_mlops
from routes.country import country_bp
from routes.ml import ml_bp
from routes.phc import phc_bp


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(config or {})

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "healthy"}), 200

    app.register_blueprint(country_bp, url_prefix="/api/countries")

    app.register_blueprint(phc_bp, url_prefix="/api/phcs")

    app.register_blueprint(ml_bp, url_prefix="/api/ml")

    init_mlops(app)
    return app


app = create_app()


if __name__ == "__main__":
    # The reloader would start a second process (and a second scheduler candidate).
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)
