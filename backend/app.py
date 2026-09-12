from flask import Flask, jsonify

from routes.country import country_bp
from routes.phc import phc_bp

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


app.register_blueprint(country_bp, url_prefix="/api/countries")

app.register_blueprint(phc_bp, url_prefix="/api/phcs")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
