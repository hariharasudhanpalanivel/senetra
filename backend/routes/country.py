from flask import Blueprint, jsonify
from services.country import get_countries, get_country_by_id

country_bp = Blueprint("country", __name__)


@country_bp.route("/", methods=["GET"])
def countries():
    return jsonify(get_countries())


@country_bp.route("/<int:country_id>", methods=["GET"])
def country(country_id):
    return jsonify(get_country_by_id(country_id))
