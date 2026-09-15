"""SENETRA ML API: forecasts with canary routing, continuous training, deployments and monitoring."""

from __future__ import annotations

import hmac
import json
from functools import wraps

from flask import Blueprint, current_app, request

from senetra_ml.inference import Scenario
from senetra_ml.tracking import ModelUnavailableError

from mlops.extension import get_mlops

ml_bp = Blueprint("ml", __name__)


def _default(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def respond(payload, status: int = 200):
    return current_app.response_class(json.dumps(payload, default=_default), status=status,
                                      mimetype="application/json")


def admin_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        token = get_mlops().settings.admin_token
        supplied = request.headers.get("Authorization", "")
        if token and not hmac.compare_digest(supplied, f"Bearer {token}"):
            return respond({"error": "unauthorized", "detail": "Authorization: Bearer <ML_ADMIN_TOKEN> required"}, 401)
        return view(*args, **kwargs)
    return wrapper


@ml_bp.errorhandler(ModelUnavailableError)
def _unavailable(exc):
    return respond({"error": "model_unavailable", "detail": str(exc)}, 503)


@ml_bp.errorhandler(LookupError)
def _not_found(exc):
    return respond({"error": "not_found", "detail": str(exc).strip("'\"")}, 404)


@ml_bp.errorhandler(ValueError)
def _invalid(exc):
    return respond({"error": "invalid_request", "detail": str(exc)}, 422)


def _forecast_request() -> dict:
    districts = request.args.getlist("districts", type=int)
    level = request.args.get("level", "district")
    entity_id = request.args.get("entity_id", type=int)
    override = (request.headers.get("X-ML-Variant") or "").lower() or None
    if override not in (None, "stable", "canary"):
        raise ValueError("X-ML-Variant must be 'stable' or 'canary'")
    return {
        "level": level,
        "entity_id": entity_id,
        "scenario": Scenario(request.args.get("scenario", "observed"), request.args.get("severity", type=float),
                             tuple(districts) if districts else None),
        "origin": request.args.get("origin"),
        "routing_key": request.headers.get("X-Client-Id") or f"{level}:{entity_id}",
        "override": override,
    }


# ---------------------------------------------------------------------------- forecasts

@ml_bp.get("/forecast/<target>")
def forecast(target: str):
    get_mlops().cfg.target(target)
    return respond(get_mlops().predictor.forecast(target, **_forecast_request()))


@ml_bp.get("/forecast")
def forecast_all():
    return respond(get_mlops().predictor.forecast_all(**_forecast_request()))


@ml_bp.get("/targets")
def targets():
    return respond([{"target": name, "source": t.source, "medicine": t.medicine, "family": t.family,
                     "aggregation": t.aggregation, "unit": t.unit} for name, t in get_mlops().cfg.targets.items()])


@ml_bp.get("/hierarchy")
def hierarchy():
    frame = get_mlops().predictor.hierarchy
    return respond([
        {"country_id": int(cid), "country": str(cname), "districts": [
            {"district_id": int(did), "district": str(dname), "phcs": int(len(rows))}
            for (did, dname), rows in group.groupby(["district_id", "district_name"])]}
        for (cid, cname), group in frame.groupby(["country_id", "country_name"])
    ])


# ---------------------------------------------------------------------------- status

@ml_bp.get("/health")
def health():
    mlops = get_mlops()
    deployments = [d.to_dict() for d in mlops.deployments.all_status()]
    return respond({
        "status": "ok" if all(d["stable_version"] for d in deployments) else "degraded",
        "scheduler": mlops.scheduler_info(),
        "active_job": mlops.state.active_job("retrain"),
        "deployments": deployments,
    })


# ---------------------------------------------------------------------------- continuous training

@ml_bp.get("/data/status")
def data_status():
    return respond(get_mlops().orchestrator.data_status())


@ml_bp.post("/data/arrived")
@admin_required
def data_arrived():
    """Webhook for ingestion: call after loading new data; retrains if the policy says so."""
    return respond(get_mlops().orchestrator.check_for_new_data(trigger="data_arrival"), 202)


@ml_bp.post("/retrain")
@admin_required
def retrain():
    body = request.get_json(silent=True) or {}
    targets = body.get("targets")
    for target in targets or []:
        get_mlops().cfg.target(target)
    result = get_mlops().orchestrator.check_for_new_data(trigger="manual", force=bool(body.get("force", True)),
                                                         targets=targets)
    return respond(result, 202)


@ml_bp.get("/jobs")
def jobs():
    return respond(get_mlops().state.list_jobs(limit=request.args.get("limit", 50, type=int)))


@ml_bp.get("/jobs/<int:job_id>")
def job(job_id: int):
    found = get_mlops().state.get_job(job_id)
    if not found:
        raise LookupError(f"job {job_id} not found")
    return respond(found)


# ---------------------------------------------------------------------------- deployments

@ml_bp.get("/deployments")
def deployments():
    return respond([d.to_dict() for d in get_mlops().deployments.all_status(refresh=True)])


@ml_bp.get("/deployments/<target>")
def deployment(target: str):
    mlops = get_mlops()
    status = mlops.deployments.status(target, refresh=True)
    since = status.canary_stage_started_at
    traffic = {}
    if status.canary_version:
        traffic = {
            "canary": mlops.state.prediction_stats(target, "canary", status.canary_version, since),
            "stable": mlops.state.prediction_stats(target, "stable", status.stable_version, since),
        }
    return respond({**status.to_dict(), "stage_traffic": traffic, "events": mlops.state.list_events(target, 20)})


@ml_bp.post("/deployments/<target>/evaluate")
@admin_required
def evaluate_canary(target: str):
    return respond(get_mlops().orchestrator.evaluate_canary(target))


def _reason() -> str:
    return (request.get_json(silent=True) or {}).get("reason") or "manual request"


@ml_bp.post("/deployments/<target>/promote")
@admin_required
def promote(target: str):
    return respond(get_mlops().deployments.promote(target, f"manual: {_reason()}"))


@ml_bp.post("/deployments/<target>/rollback")
@admin_required
def rollback(target: str):
    return respond(get_mlops().deployments.rollback(target, f"manual: {_reason()}"))


@ml_bp.post("/deployments/<target>/revert")
@admin_required
def revert(target: str):
    return respond(get_mlops().deployments.revert(target, f"manual: {_reason()}"))


# ---------------------------------------------------------------------------- monitoring

@ml_bp.post("/monitoring/run")
@admin_required
def run_monitoring():
    return respond(get_mlops().orchestrator.run_monitoring())


@ml_bp.get("/monitoring/latest")
def latest_monitoring():
    path = get_mlops().cfg.reports_dir / "monitoring" / "latest.json"
    if not path.is_file():
        raise LookupError("no monitoring report yet; POST /api/ml/monitoring/run")
    return respond(json.loads(path.read_text(encoding="utf-8")))
