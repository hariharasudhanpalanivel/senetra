"""FastAPI service exposing hierarchical, scenario-conditioned forecasts from registry champions."""

from __future__ import annotations

from datetime import date
from typing import Literal

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse

from senetra_ml import __version__
from senetra_ml.config import PipelineConfig, load_config
from senetra_ml.inference import ForecastService, Scenario
from senetra_ml.tracking import ModelUnavailableError, setup_mlflow

Level = Literal["phc", "district", "country", "world"]
ScenarioName = Literal["observed", "normal", "outbreak", "simulated"]


def create_app(cfg: PipelineConfig | None = None, service: ForecastService | None = None) -> FastAPI:
    cfg = cfg or load_config()
    app = FastAPI(
        title="SENETRA Forecast API", version=__version__,
        description="Patient footfall, bed occupancy, staff availability and medicine demand forecasts "
                    "for PHC, district, country and world levels, under normal, observed, simulated or "
                    "what-if outbreak scenarios.",
    )
    state: dict[str, ForecastService | None] = {"service": service}

    def svc() -> ForecastService:
        if state["service"] is None:
            setup_mlflow(cfg)
            state["service"] = ForecastService(cfg)
        return state["service"]

    @app.exception_handler(ModelUnavailableError)
    async def _unavailable(_: Request, exc: ModelUnavailableError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(LookupError)
    async def _not_found(_: Request, exc: LookupError):
        return JSONResponse(status_code=404, content={"detail": str(exc).strip("'\"")})

    @app.exception_handler(ValueError)
    async def _invalid(_: Request, exc: ValueError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict:
        service_ = svc()
        lo, hi = service_.repo.date_bounds()
        models = {}
        for target in cfg.targets:
            try:
                models[target] = service_.model(target).label
            except ModelUnavailableError:
                models[target] = None
        return {"status": "healthy" if all(models.values()) else "degraded", "service": "senetra-ml",
                "version": __version__, "data_window": {"start": str(lo.date()), "end": str(hi.date())},
                "models": models}

    @app.get("/v1/targets")
    def targets() -> list[dict]:
        return [{"target": name, "source": t.source, "medicine": t.medicine, "family": t.family,
                 "aggregation": t.aggregation, "unit": t.unit, "horizon_days": cfg.forecast.horizon_days}
                for name, t in cfg.targets.items()]

    @app.get("/v1/hierarchy")
    def hierarchy() -> list[dict]:
        frame = svc().hierarchy
        countries = []
        for (country_id, country_name), c_rows in frame.groupby(["country_id", "country_name"]):
            districts = [{"district_id": int(d_id), "district": str(d_name), "phcs": int(len(d_rows))}
                         for (d_id, d_name), d_rows in c_rows.groupby(["district_id", "district_name"])]
            countries.append({"country_id": int(country_id), "country": str(country_name),
                              "phcs": int(len(c_rows)), "districts": districts})
        return countries

    def _scenario(name: str, severity: float | None, districts: list[int] | None) -> Scenario:
        return Scenario(name=name, severity=severity, district_ids=tuple(districts) if districts else None)

    @app.get("/v1/forecast/{target}")
    def forecast(target: str, level: Level = "district", entity_id: int | None = None,
                 scenario: ScenarioName = "observed", severity: float | None = Query(None, gt=0, le=10),
                 districts: list[int] | None = Query(None), origin: date | None = None) -> dict:
        return svc().forecast(target, level, entity_id, _scenario(scenario, severity, districts), origin)

    @app.get("/v1/forecast")
    def forecast_all(level: Level = "district", entity_id: int | None = None,
                     scenario: ScenarioName = "observed", severity: float | None = Query(None, gt=0, le=10),
                     districts: list[int] | None = Query(None), origin: date | None = None) -> dict:
        return svc().forecast_all(level, entity_id, _scenario(scenario, severity, districts), origin)

    @app.get("/v1/federation/{target}")
    def federation(target: str) -> dict:
        loaded = svc().model(target)
        meta = loaded.bundle.metadata
        return {
            "target": target, "model": loaded.label, "parameter_level": loaded.bundle.selected_level,
            "trained_at": meta.get("trained_at"), "training_window": [meta.get("training_start"), meta.get("training_end")],
            "hierarchy": meta.get("hierarchy"), "rounds": meta.get("federated_history"),
            "config": meta.get("federated_config"), "phc_mae_by_level": meta.get("level_phc_mae"),
            "reference_metrics": meta.get("reference_metrics"), "outbreak_effect": meta.get("outbreak_effect"),
            "key_drivers": meta.get("world_coefficients"), "privacy_contract": meta.get("privacy_contract"),
        }

    @app.post("/v1/models/reload")
    def reload() -> dict:
        svc().reload()
        return {"status": "reloaded"}

    return app
