from flask import Blueprint, jsonify
from services.phc import get_phcs, get_phcs_by_district, get_phc_by_id

phc_bp = Blueprint("phc", __name__)


@phc_bp.route("/", methods=["GET"])
def phcs():
    return jsonify(get_phcs())


@phc_bp.route("/<int:phc_id>", methods=["GET"])
def phc(phc_id):
    return jsonify(get_phc_by_id(phc_id))


@phc_bp.route("/district/<int:district_id>", methods=["GET"])
def district_phcs(district_id):
    return jsonify(get_phcs_by_district(district_id))
