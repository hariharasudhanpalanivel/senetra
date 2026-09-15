# SENETRA backend

Flask API. Besides the country/PHC endpoints it runs the SENETRA ML pipeline (`../ml`, package
`senetra_ml`) in production mode:

- **Forecast serving** for patient footfall, bed occupancy, staff availability and 5 medicines, at PHC,
  district, country or world level, under normal / observed / outbreak / simulated scenarios.
- **Continuous training**: new data in `senetra.db` is detected automatically (or via a webhook) and the
  models are retrained in a separate process.
- **Canary deployment**: a retrained model that passes the evaluation gates receives a growing share of
  traffic, is checked for errors, latency and accuracy on fresh data, and is then promoted or rolled back.

## Architecture

```
                 ingestion writes new rows
                           │
                     senetra.db (SQLite, read-only for ML)
                           │
┌──────────────────────────┼───────────────────────────────────────────────────────────┐
│ Flask backend            │                                                            │
│                          ▼                                                            │
│  MLScheduler (one leader across processes, lease in instance/mlops.db)                │
│   ├─ every 5 min  data check ──► retraining policy ──► JobRunner ──► retrain process  │
│   ├─ every 10 min canary evaluation ──► advance / promote / rollback                  │
│   └─ daily        monitoring (drift, data quality, live performance)                  │
│                                                                                       │
│  retrain process: validate → train → evaluate (gates) → @canary or @challenger        │
│                                                                                       │
│  /api/ml/forecast ──► CanaryPredictor ──► route by sticky hash ──► @champion / @canary │
│                          └─► prediction log (status, latency) ──► canary decisions     │
└───────────────────────────────────────────────────────────────────────────────────────┘
                           │
                  MLflow tracking + model registry
     aliases per model: champion · canary · previous_champion · challenger
```

| Module | Responsibility |
|---|---|
| `mlops/settings.py` | All MLOps settings (`ML_*` environment variables) |
| `mlops/state.py` | Shared SQLite state: jobs, data watermarks, prediction log, deployment events, scheduler lease |
| `mlops/data_watch.py` | Data signature of the training tables and the retraining policy |
| `mlops/jobs.py` | Retraining job (runs in its own process) and the job runner |
| `mlops/deployment.py` | Registry aliases, canary stages in model-version tags, sticky traffic routing |
| `mlops/canary.py` | Canary decision rules (hold / advance / promote / rollback) |
| `mlops/orchestrator.py` | Data checks, canary evaluation, monitoring, scheduler thread |
| `mlops/predictor.py` | Forecasts through stable or canary model, prediction logging |
| `mlops/extension.py` | Flask wiring (`app.extensions["senetra_mlops"]`) |
| `routes/ml.py` | `/api/ml/*` endpoints |

## Setup and run

```bash
cd backend
uv venv .venv --python 3.12              # or: python -m venv .venv (Python 3.11-3.13)
uv pip install --python .venv -r requirements-dev.txt   # or: .venv/Scripts/pip install -r requirements-dev.txt
.venv/Scripts/python app.py              # Windows; Linux/macOS: .venv/bin/python app.py
```

The API listens on http://localhost:8000. On Linux, production-style: `gunicorn -w 4 -b 0.0.0.0:8000 app:app`
(any number of workers: only the lease holder runs the scheduler).

First start with no models: the first data check (within `ML_DATA_CHECK_INTERVAL_SECONDS`, or immediately with
`POST /api/ml/data/arrived`) trains all targets and, because there is no champion yet, promotes them directly.
If champions already exist (for example trained with `senetra-ml pipeline`), the first check only records the
current data as the baseline.

## Configuration

ML pipeline settings come from `ml/configs/pipeline.yaml`; paths can be overridden with
`SENETRA_DB_PATH`, `MLFLOW_TRACKING_URI` and `SENETRA_CONFIG`. Defaults use `../senetra.db` and the local MLflow
store `ml/mlruns`, shared with the `senetra-ml` CLI.

| Variable | Default | Meaning |
|---|---|---|
| `ML_ADMIN_TOKEN` | unset (open) | Bearer token required by every state-changing endpoint. **Set it outside local development.** |
| `ML_STATE_DB` | `backend/instance/mlops.db` | Shared MLOps state |
| `ML_SCHEDULER_ENABLED` | `true` | Run periodic data checks, canary evaluation and monitoring |
| `ML_SCHEDULER_POLL_SECONDS` | `15` | Scheduler loop interval |
| `ML_DATA_CHECK_INTERVAL_SECONDS` | `300` | How often to look for new data |
| `ML_CANARY_CHECK_INTERVAL_SECONDS` | `600` | How often to evaluate active canaries |
| `ML_MONITOR_INTERVAL_SECONDS` | `86400` | Monitoring cadence |
| `ML_RETRAIN_TARGETS` | all | Comma-separated targets to retrain |
| `ML_RETRAIN_MIN_NEW_DAYS` | `1` | New days required before retraining |
| `ML_RETRAIN_COOLDOWN_MINUTES` | `60` | Minimum gap between retraining jobs |
| `ML_JOB_RUNNER` | `process` | `process` (separate OS process) or `inline` (tests/debugging) |
| `ML_JOB_STALE_MINUTES` | `10` | A running job without heartbeat for this long is marked failed |
| `ML_CANARY_STAGES` | `10,25,50,100` | Traffic percentages per stage (must end at 100) |
| `ML_CANARY_MIN_STAGE_MINUTES` | `60` | Minimum time per stage |
| `ML_CANARY_MIN_REQUESTS` | `20` | Minimum canary requests per stage |
| `ML_CANARY_MAX_ERROR_RATE_DELTA` | `0.02` | Allowed error-rate increase over stable |
| `ML_CANARY_MAX_LATENCY_RATIO` | `1.5` | Allowed p95 latency ratio over stable … |
| `ML_CANARY_LATENCY_FLOOR_MS` | `250` | … ignored while canary p95 stays below this |
| `ML_CANARY_MAX_WAPE_REGRESSION` | `0.05` | Allowed district WAPE regression on fresh actuals |
| `ML_CANARY_MAX_PERCENT_WITHOUT_ACTUALS` | `25` | Traffic cap until accuracy on fresh data is known; set `100` to allow promotion without it |

## Continuous training

1. **Detect.** The signature of `daily_metrics`, `medicine_consumption` and `phcs` (row count, max id, latest date)
   is compared with the signature recorded by the last successful retraining. Inventory updates do not count.
2. **Decide** (`data_watch.decide_retrain`): retrain on new days (≥ `ML_RETRAIN_MIN_NEW_DAYS`) or on changed
   rows (backfill/corrections); skip while a canary is active (its accuracy check needs the new data), during
   the cooldown, or when a job is already running. `POST /api/ml/retrain` forces a run.
3. **Run** in a separate process: data validation → `run_training` → `run_evaluation` (gates, no automatic
   promotion). Passing models start a canary; failing ones get the `challenger` alias. The trained data
   signature is recorded, so the same data never triggers a second run.

## Canary deployment

```
retrained model passes gates ──► @canary at 10% ──► 25% ──► 50% ──► 100% ──► promote: @champion
                                     │ each stage: min time + min requests      (old champion → @previous_champion)
                                     └─ any stage: error rate / p95 latency / fresh-data WAPE worse ──► rollback
```

- **Routing** is sticky: `sha256(target | canary version | X-Client-Id)` (or `level:entity_id` without the
  header) picks a 0–99 bucket; buckets below the stage percentage get the canary. A caller keeps its variant
  during a stage, and buckets reshuffle for each new canary version.
- **Evidence** comes from the prediction log (forced requests and client errors are excluded) and from
  replaying both models on actuals that arrived after both were trained. Without such actuals the rollout
  stops at `ML_CANARY_MAX_PERCENT_WITHOUT_ACTUALS`.
- **State** (stage, percent, timestamps) is stored in MLflow model-version tags (`canary.*`), so all backend
  instances agree and the rollout is visible in the MLflow UI. Every transition is also written to
  `ml_deployment_events`.
- **Manual control**: promote, roll back, or revert production to the previous champion at any time.
- `X-ML-Variant: stable|canary` forces a variant for QA; those requests are not counted as canary evidence.

## API (`/api/ml`)

State-changing endpoints (★) need `Authorization: Bearer $ML_ADMIN_TOKEN` when the token is set.

| Method | Path | Purpose |
|---|---|---|
| GET | `/forecast/<target>?level=&entity_id=&scenario=&severity=&districts=&origin=` | One target; response includes `deployment` (variant, version, canary %) |
| GET | `/forecast?level=&entity_id=&scenario=…` | All targets |
| GET | `/targets`, `/hierarchy` | Forecastable targets; valid entity ids |
| GET | `/health` | Deployments per target, scheduler/lease, active job |
| GET | `/data/status` | Current vs last-trained data signature and what the policy would do |
| POST ★ | `/data/arrived` | Webhook for ingestion: check now and retrain if needed |
| POST ★ | `/retrain` `{"targets": [...], "force": true}` | Manual retraining |
| GET | `/jobs`, `/jobs/<id>` | Job history, status, results, errors |
| GET | `/deployments`, `/deployments/<target>` | Versions, canary stage, stage traffic stats, recent events |
| POST ★ | `/deployments/<target>/evaluate` | Run the canary decision now |
| POST ★ | `/deployments/<target>/promote` \| `/rollback` \| `/revert` | Manual rollout control (`{"reason": "..."}`) |
| POST ★ | `/monitoring/run` · GET `/monitoring/latest` | Monitoring run / latest report |

Status codes: 401 missing token, 404 unknown target/entity/job, 422 invalid request (e.g. outbreak without
severity), 503 no deployed model.

```bash
curl "http://localhost:8000/api/ml/forecast/ors?level=district&entity_id=4&scenario=outbreak&severity=8" -H "X-Client-Id: officer-17"
curl -X POST http://localhost:8000/api/ml/data/arrived -H "Authorization: Bearer $ML_ADMIN_TOKEN"
curl http://localhost:8000/api/ml/deployments/ors
curl -X POST http://localhost:8000/api/ml/deployments/ors/rollback -H "Authorization: Bearer $ML_ADMIN_TOKEN" -H "Content-Type: application/json" -d '{"reason":"bad forecasts reported"}'
```

## Tests

```bash
cd backend
.venv/Scripts/python -m pytest          # ~2-3 minutes
```

- `tests/test_mlops_units.py`: canary decision table, retraining policy, data signature, sticky traffic split,
  settings parsing, scheduler lease, prediction statistics.
- `tests/test_mlops_flow.py`: a synthetic database whose last 3 days arrive later, driven through the HTTP API:
  bootstrap champion → no retrain without data → new data retrains into a canary → sticky 50/50 split →
  advance → promote → injected errors roll the next canary back → revert to the previous champion.

## Operations

| Situation | Action |
|---|---|
| Stop a bad rollout now | `POST /deployments/<target>/rollback` |
| Production model misbehaves | `POST /deployments/<target>/revert` (previous champion) |
| Job stuck in `running` | It is failed automatically after `ML_JOB_STALE_MINUTES` without heartbeat; then `POST /retrain` |
| Rollout waiting for actuals | Expected until data after the canary's training end arrives; promote manually if justified |
| Retraining never starts | `GET /data/status` shows the reason (canary active, cooldown, no new data) |
| Several instances | Use one shared `ML_STATE_DB` and one MLflow tracking server; only the lease holder schedules |

## Limitations

- The existing `/api/countries` and `/api/phcs` routes still use PostgreSQL (`clients/db.py`), while the ML
  pipeline reads the SQLite `senetra.db`. Point both at the same data before production.
- `ML_STATE_DB` is SQLite: fine for one host with several workers; use a network database for multi-host setups.
- A retraining process needs ~1.5 GB of memory for the full dataset.
