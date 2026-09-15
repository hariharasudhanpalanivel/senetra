---
name: senetra-prediction-serving
description: Produce SENETRA forecasts — patient counts, bed occupancy, staff availability and medicine/tablet demand — for a PHC, district, country or the world under normal, observed, simulated or what-if outbreak (pandemic) scenarios, with intervals and stock-out risk, via CLI, FastAPI or the predictions table. Use when someone needs a prediction, integrates the backend/frontend with the model, or runs batch forecasts.
---

# SENETRA prediction and serving

Models are loaded from the MLflow registry alias `@champion` (`senetra-forecast-<target>`).
Run `senetra-ml evaluate` first; `--allow-unpromoted` uses the latest version for experiments only.

## Scenarios — pick deliberately

| Scenario | Target-day outbreak input | Use for |
|---|---|---|
| `observed` (default) | current district prevalence persists | Operational forecasts, alerts |
| `normal` | no outbreak | "Normal day" demand, baseline planning |
| `outbreak --severity S [--districts ...]` | intensity S / `severity_reference` (8 = historical outbreak level), capped at `max_intensity` | Pandemic/what-if planning, surge stock |
| `simulated` | observed, raised to active `simulation_events` severity | Emergency simulator demo |

Intervals: `observed` uses persistence residuals (wider at regime changes); other scenarios use
scenario-correct residuals. The outbreak response is learned from the dengue-like episode in history;
FLOOD/HEATWAVE events reuse that response and should be presented as assumptions.

## CLI

```bash
cd ml
# Patient count next 7 days for district 5, normal vs pandemic
uv run senetra-ml predict --targets patient_footfall --level district --entity-id 5 --scenario normal
uv run senetra-ml predict --targets patient_footfall --level district --entity-id 5 --scenario outbreak --severity 9
# All drugs for one PHC during a simulated emergency
uv run senetra-ml predict --targets ors paracetamol iv_fluids antibiotics insulin --level phc --entity-id 42 --scenario simulated
# Everything, national and world level
uv run senetra-ml predict --level country --entity-id 3
uv run senetra-ml predict --level world
# Batch-write PHC medicine forecasts into the predictions table (idempotent per model version)
uv run senetra-ml predict --level world --targets ors paracetamol iv_fluids antibiotics insulin --write-db
```

Outputs JSON to stdout and `reports/predictions/*.json`, and logs a `senetra-predictions` MLflow run.

## API

`uv run senetra-ml serve --port 8080` (Docker: `http://localhost:8001`). OpenAPI docs at `/docs`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | data window, champion version per target (`degraded` if any missing) |
| `GET /v1/targets`, `GET /v1/hierarchy` | what can be forecast, valid entity ids |
| `GET /v1/forecast/{target}?level=&entity_id=&scenario=&severity=&districts=&origin=` | one target |
| `GET /v1/forecast?level=&entity_id=&scenario=...` | all targets at once (`errors` lists unavailable ones) |
| `GET /v1/federation/{target}` | rounds, hierarchy, per-level accuracy, key drivers, privacy contract |
| `POST /v1/models/reload` | pick up newly promoted champions |

Response: `forecast[]` with `date, horizon, yhat, lower_80, upper_80, lower_95, upper_95, confidence, imputed_share`;
`horizon_total` for count targets; for medicines `stock` with `current_stock`, `days_of_supply`,
`days_of_supply_pessimistic` (upper 80% demand), `risk_level` (CRITICAL <= 3 days, HIGH <= 7, WATCH <= 14, else HEALTHY).
Status codes: 404 unknown target/entity, 422 invalid scenario (e.g. outbreak without severity), 503 no champion.

## Semantics to keep in mind

- District/country/world values are bottom-up sums (counts) or means (percentages) of PHC forecasts, so
  levels are coherent: world total == sum of countries.
- `model.parameter_level` says which federation tier's parameters served the PHCs.
- Forecast origin defaults to the latest date in the DB; `origin=YYYY-MM-DD` replays history.
- Gemini/copilot integrations should pass these structured fields verbatim (forecast, stock, scenario, model version) and never invent numbers.
