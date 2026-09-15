# SENETRA — Getting started, running and testing

This guide explains what is in the repository, how to set it up, how to run every part of the
machine-learning system, and how to test it (automated tests, a real-data smoke test, the API and Docker).

---

## 1. What is in the repository

```
senetra/
├── senetra.db                 SQLite operational database (built from seeds/, ~65 MB, not committed)
├── seeds/
│   ├── DDL.sql                PostgreSQL schema (countries → states → districts → PHCs, medicines, metrics, …)
│   ├── DML.sql                PostgreSQL seed data (random, relative to the day it is generated)
│   └── build_sqlite.py        Converts DDL/DML to SQLite and builds a database file
├── backend/                   Flask API: country/PHC routes (Postgres) + /api/ml forecasts, continuous training, canary rollout
├── documents/                 Reference PDFs
├── .claude/skills/            Claude Code skills: how to operate each ML stage (see §9)
└── ml/                        Forecasting system (Python package `senetra_ml`)
    ├── configs/pipeline.yaml  All settings: targets, features, federation, backtest, gates, monitoring
    ├── src/senetra_ml/
    │   ├── data/              Read-only SQL access, in-memory panel, data-contract validation
    │   ├── eda/               Exploratory analysis and report generation
    │   ├── features/          Leakage-safe feature engineering, federated standardization
    │   ├── federated/         GLM solver and hierarchical federated trainer (PHC → district → country → world)
    │   ├── models/            Deployable forecast bundle, MLflow pyfunc wrapper, XGBoost benchmark
    │   ├── training/          Rolling-origin backtesting
    │   ├── evaluation/        Metrics, level selection, conformal intervals
    │   ├── monitoring/        Regime-matched drift (PSI)
    │   ├── pipelines/         eda · train · evaluate · predict · monitor
    │   ├── inference.py       Forecast service (scenarios, reconciliation, stock-out risk)
    │   ├── serving/api.py     FastAPI application
    │   └── cli.py             `senetra-ml` command line
    ├── tests/                 39 tests on a synthetic database (never touch senetra.db)
    ├── reports/               eda/ and evaluation/ reports (monitoring/ and predictions/ are git-ignored)
    ├── mlruns/                Local MLflow tracking database and artifacts (git-ignored)
    ├── Dockerfile, docker-compose.yml
    └── README.md              Design, results and limitations
```

### What the system does

It forecasts the next **7 days** of:

| Target | Kind |
|---|---|
| `patient_footfall` | patients per day (count) |
| `bed_occupancy`, `staff_availability` | percent |
| `ors`, `paracetamol`, `iv_fluids`, `antibiotics`, `insulin` | medicine units consumed per day (count) |

…for a **PHC, district, country or the whole world**, under one of four **scenarios**:

| Scenario | Meaning |
|---|---|
| `observed` (default) | the current outbreak state continues |
| `normal` | normal day, no outbreak |
| `outbreak` + `severity` (1–10) | pandemic / what-if; severity 8 = the outbreak seen in history |
| `simulated` | applies the active rows in `simulation_events` |

Each forecast has 80%/95% intervals, and medicine forecasts include current stock, days of supply and a
risk level (CRITICAL ≤ 3 days, HIGH ≤ 7, WATCH ≤ 14, else HEALTHY).

### How it works in one paragraph

Every PHC is a federated client: its raw rows are only read by queries scoped to that PHC. Clients send
summed loss/gradient/Hessian statistics up through district → country → world, so the world model equals
a centralized fit without pooling data; district, country and PHC models then refine their parent model.
Models are generalized linear models (Poisson for counts, Gaussian for percentages) with a target-day
outbreak input that the scenario controls. Training backtests every model against centralized XGBoost,
local-only models and naive baselines, registers the result in MLflow, and evaluation promotes models
that pass quality gates to the `@champion` alias that prediction, the API and monitoring use.

---

## 2. Prerequisites

| Tool | Version | Needed for |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | 0.8+ | Python environment (installs Python 3.12 automatically) |
| Git Bash or PowerShell | — | running commands (examples below work in both unless noted) |
| Docker Desktop | 29+ with Compose v2 | optional: containerized stack |
| RAM | 16 GB recommended | a full training run peaks at ~1.5 GB; see §8 |

PowerShell note: use `curl.exe` (not `curl`, which is an alias for `Invoke-WebRequest`).

---

## 3. One-time setup

### 3.1 Database

The pipeline reads `senetra.db` at the repository root. If it is missing, build it from the seeds:

```bash
# from the repository root
python seeds/build_sqlite.py senetra.db
```

Notes:
- The seed SQL uses random values and dates relative to **today**, so each rebuild gives slightly
  different numbers and a fresh date window.
- Any existing file at that path is replaced.
- Use another database with `SENETRA_DB_PATH=/path/to/file.db`.

### 3.2 Python environment

```bash
cd ml
uv sync                      # creates ml/.venv with all dependencies from uv.lock
uv run senetra-ml --help     # confirms the CLI is installed
```

All commands below are run from `ml/`.

---

## 4. Test level 1 — automated tests (no real data needed)

```bash
uv run pytest                                  # all 39 tests, ~5 minutes
uv run pytest tests/test_features.py tests/test_federated.py tests/test_metrics_risk_drift.py   # 31 unit tests, ~1 minute
uv run pytest tests/test_pipeline_e2e.py       # 8 end-to-end tests, ~3-4 minutes
uv run pytest -k federated -v                  # a subset, verbose
```

The tests build their own **synthetic SQLite database** from `seeds/DDL.sql` (2 countries, 3 districts,
11 PHCs, 70 days, one outbreak) and a temporary MLflow store, so they never modify `senetra.db` or `ml/mlruns`.

| File | What it proves |
|---|---|
| `test_features.py` | features never use future data (tampering test), rolling statistics are correct, scenario/regime flags are right |
| `test_federated.py` | a client's update depends only on its own rows; federated Newton equals the centralized fit; hierarchy levels shrink to their parent; FedAvg maths; Gaussian and Poisson recovery |
| `test_metrics_risk_drift.py` | WAPE/MAE/bias formulas, level aggregation, conformal interval coverage, days of supply, risk levels, PSI |
| `test_pipeline_e2e.py` | validate → EDA → train → evaluate → forecasts at every level and scenario → DB write → monitoring → every API endpoint; checks that backtest artifacts contain no actuals, outbreak forecasts exceed normal ones, world totals equal the sum of countries |

Expected result: `39 passed`.

---

## 5. Test level 2 — run the full pipeline on the real database

Run the stages one at a time the first time, so you can inspect each output.

| # | Command | Time | What to check |
|---|---|---|---|
| 1 | `uv run senetra-ml validate` | ~5 s | JSON with `"errors": 0`. Warnings about the stock ledger, outbreak labels and coordinates are known data issues. |
| 2 | `uv run senetra-ml eda` | ~15 s | `reports/eda/eda_report.md` (open it: implications, tables, figures) |
| 3 | `uv run senetra-ml train` | ~6 min | log line per target: `done in …s: level=… version=N district WAPE (observed)=…` |
| 4 | `uv run senetra-ml evaluate` | ~1 min | `reports/evaluation/summary.md`; each target `promoted to @champion` or `kept as @challenger` |
| 5 | `uv run senetra-ml predict --targets ors --level district --entity-id 4` | ~10 s | 7-day ORS forecast for Coimbatore with stock risk |
| 6 | `uv run senetra-ml monitor` | ~20 s | `reports/monitoring/latest.json`; status `ok` or `warn` means healthy |

Steps 2–4 in one command: `uv run senetra-ml pipeline` (add `--skip-eda` to skip step 2).

### Expected evaluation results

District-level error (WAPE) on the backtest; your numbers will be close if the database was built the same way:

| target | federated (observed) | federated (correct scenario) | centralized XGBoost | 7-day average |
|---|---:|---:|---:|---:|
| patient_footfall | 3.8% | 2.5% | 4.0% | 8.8% |
| bed_occupancy | 2.5% | 1.8% | 2.7% | 5.4% |
| ors | 4.7% | 3.2% | 6.7% | 10.6% |
| paracetamol | 4.1% | 2.9% | 5.6% | 9.1% |

If a newly trained target shows district WAPE above ~10% while the 7-day average stays lower, treat it as a
regression and read the evaluation skill (`.claude/skills/senetra-model-evaluation`).

### Useful IDs for testing (current database)

| Level | ID | Name |
|---|---|---|
| country | 3 | India (the only country with PHCs) |
| district | 4, 5, 6, 7, 8 | Coimbatore, Cuddalore, Dharmapuri, Dindigul, Erode |
| PHC | 171, 172, 173 | KARAMADAI … (Coimbatore) |

List all IDs with the API endpoint `GET /v1/hierarchy` (§6) or:

```bash
uv run python -c "from senetra_ml.config import load_config; from senetra_ml.data.repository import SenetraRepository; print(SenetraRepository(load_config().data.db_path).hierarchy()[['phc_id','phc_name','district_id','district_name','country_id']].to_string())"
```

### Scenario acceptance checks (manual)

Run these after training and evaluation. Each line states the expected behaviour.

```bash
# Patient count: normal vs pandemic for Coimbatore (district 4) — outbreak should be ~1.78x normal
uv run senetra-ml predict --targets patient_footfall --level district --entity-id 4 --scenario normal
uv run senetra-ml predict --targets patient_footfall --level district --entity-id 4 --scenario outbreak --severity 8

# Drugs: ORS ~2x, Paracetamol ~1.8x, IV Fluids ~1.5x; Antibiotics and Insulin unchanged
uv run senetra-ml predict --targets ors paracetamol iv_fluids antibiotics insulin --level district --entity-id 4 --scenario outbreak --severity 8

# One PHC, all 8 targets
uv run senetra-ml predict --level phc --entity-id 171 --scenario normal

# National and world level (identical today because only India has PHCs)
uv run senetra-ml predict --level country --entity-id 3 --scenario outbreak --severity 9
uv run senetra-ml predict --level world --scenario simulated

# Outbreak limited to Coimbatore: district 5 (Cuddalore) should stay at normal levels
uv run senetra-ml predict --targets ors --level district --entity-id 5 --scenario outbreak --severity 8 --districts 4
```

Reference values on the current database (day 1, Coimbatore): ORS ≈ 2,686 normal vs ≈ 5,378 outbreak;
risk HEALTHY (25 days of supply) vs WATCH (13 days); bed occupancy ≈ 57% vs ≈ 82%.

Each run prints JSON and saves it to `reports/predictions/`. `--write-db` also stores PHC-level medicine
forecasts in the `predictions` table (rerunning the same model version replaces its rows).

---

## 6. Test level 3 — the forecast API

```bash
uv run senetra-ml serve --port 8080
```

Open **http://localhost:8080/docs** for interactive Swagger UI (try every endpoint in the browser), or use curl
from a second terminal:

```bash
curl.exe http://localhost:8080/health
curl.exe http://localhost:8080/v1/targets
curl.exe http://localhost:8080/v1/hierarchy
curl.exe "http://localhost:8080/v1/forecast/patient_footfall?level=district&entity_id=4&scenario=normal"
curl.exe "http://localhost:8080/v1/forecast/ors?level=phc&entity_id=171&scenario=outbreak&severity=9"
curl.exe "http://localhost:8080/v1/forecast?level=world&scenario=simulated"
curl.exe "http://localhost:8080/v1/forecast/ors?level=district&entity_id=5&scenario=outbreak&severity=8&districts=4"
curl.exe "http://localhost:8080/v1/forecast/ors?level=district&entity_id=4&origin=2026-08-01"
curl.exe http://localhost:8080/v1/federation/ors
curl.exe -X POST http://localhost:8080/v1/models/reload
```

| Endpoint | Expected |
|---|---|
| `/health` | `"status": "healthy"` and a model version for all 8 targets (`degraded` if any target has no champion) |
| `/v1/forecast/{target}` | 7 `forecast` points with `yhat`, `lower_80/upper_80`, `lower_95/upper_95`, `confidence`; `stock` block for medicines |
| `/v1/forecast` | `forecasts` for every target, `errors` for any without a champion |
| `/v1/federation/{target}` | training rounds, hierarchy sizes, error by federation level, key drivers, privacy contract |

Negative tests (the API should reject these):

| Request | Expected status |
|---|---|
| `/v1/forecast/ors?level=district&entity_id=999` | 404 unknown entity |
| `/v1/forecast/not_a_target` | 404 unknown target |
| `/v1/forecast/ors?level=district&entity_id=4&scenario=outbreak` (no severity) | 422 |
| `/v1/forecast/ors?level=district` (no entity_id) | 422 |
| any forecast before training/evaluation | 503 no champion model |

After training and promoting new models, call `POST /v1/models/reload` (or restart the server).

---

## 7. Test level 4 — MLflow and Docker

### 7.1 MLflow UI (local runs)

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5001
```

Open http://localhost:5001:
- **Experiments**: `senetra-eda`, `senetra-training` (one parent run with a child per target), `senetra-evaluation`,
  `senetra-predictions`, `senetra-monitoring`.
- **Models**: `senetra-forecast-<target>` with aliases `champion` / `challenger` and `eval.*` tags.
- In a training child run, the metrics `final_world_objective` / `fold*_world_objective` show federated convergence per round.

### 7.2 Full stack in Docker

Stop any local `serve` first (port clash is not an issue, but memory is — see §8).

```bash
cd ml
docker compose build                             # ~2 min first time, image ~290 MB
docker compose up -d mlflow api                  # MLflow http://localhost:5000, API http://localhost:8001/docs
docker compose ps                                # both should be "running (healthy)"
curl.exe http://localhost:8001/health            # "degraded" until models exist in the container registry
docker compose --profile jobs run --rm pipeline pipeline --skip-eda   # train + evaluate inside Docker, ~20 min
curl.exe -X POST http://localhost:8001/v1/models/reload
curl.exe http://localhost:8001/health            # now "healthy"
curl.exe "http://localhost:8001/v1/forecast/ors?level=district&entity_id=4&scenario=outbreak&severity=8"
```

Other container jobs:

```bash
docker compose --profile jobs run --rm pipeline monitor --fail-on-alert
docker compose --profile jobs run --rm predict-writer            # writes PHC medicine forecasts into senetra.db
docker compose --profile monitoring up -d monitor                # daily monitoring loop
docker compose logs -f api
docker compose down                                              # stop everything (volume mlflow-data keeps the registry)
```

The Docker MLflow registry (volume `mlflow-data`) is separate from the local one in `ml/mlruns`; models must
be trained in the registry that the API reads. The database is mounted read-only except in `predict-writer`.

### 7.3 Backend integration: continuous training and canary deployment

The Flask backend runs the pipeline for the application (details: `backend/README.md`).

```bash
cd backend
uv venv .venv --python 3.12
uv pip install --python .venv -r requirements-dev.txt
.venv/Scripts/python -m pytest                      # unit tests + full rollout timeline, ~2-3 minutes
set ML_ADMIN_TOKEN=dev-token                         # PowerShell: $env:ML_ADMIN_TOKEN="dev-token"; bash: export ML_ADMIN_TOKEN=dev-token
.venv/Scripts/python app.py                          # http://localhost:8000
```

Manual walk-through against the running backend:

| Step | Request | Expected |
|---|---|---|
| 1 | `GET /api/ml/health` | deployments per target; if you already ran `senetra-ml pipeline`, champions are listed |
| 2 | `POST /api/ml/data/arrived` (with `Authorization: Bearer dev-token`) | `baseline_recorded` (champions exist) or `retrain` (first run, several minutes) |
| 3 | `GET /api/ml/forecast/ors?level=district&entity_id=4` | forecast plus `deployment.variant = "stable"` |
| 4 | Rebuild or append data, then `POST /api/ml/data/arrived` | `retrain`; after the job, `GET /api/ml/deployments/ors` shows a `canary_version` at 10% |
| 5 | Call step 3 with different `X-Client-Id` headers | ~10% of callers get `"variant": "canary"`, each caller always the same |
| 6 | `POST /api/ml/deployments/ors/evaluate` | `hold` / `advance` / `promote` / `rollback` with the reason and checks |
| 7 | `POST /api/ml/deployments/ors/rollback` or `/promote` or `/revert` | immediate manual control |
| 8 | `GET /api/ml/jobs`, `GET /api/ml/data/status` | job history; why the next retraining will or will not start |

Without new data, the scheduler checks every 5 minutes and evaluates canaries every 10 minutes by itself.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `SENETRA database not found` | Build it (§3.1) or set `SENETRA_DB_PATH`. |
| API `503` / `has no 'champion' version` | Run `train` then `evaluate`; call `/v1/models/reload`. `predict --allow-unpromoted` uses the latest version for experiments. |
| A target is `kept as @challenger` | It failed a gate; see `reports/evaluation/summary.md` → Gates. Do not loosen gates to force it. |
| Monitoring `freshness` warning | The database's latest date is more than 2 days old. Rebuild the database from seeds or load new data. |
| Monitoring `drift:recovery` / `regime_shift` warnings | Expected after an outbreak ends; only `alert` entries need action (see `.claude/skills/senetra-model-monitoring`). |
| Background job killed / machine slows down | Memory. Don't run training, `docker compose build` and the containers at the same time on 16 GB; Docker Desktop's WSL VM keeps several GB. Train fewer targets with `--targets`. |
| Killed run still shows `RUNNING` in MLflow | Harmless; evaluation only uses finished runs. |
| Docker build fails downloading packages | Transient network/DNS; rerun `docker compose build` (downloads are cached). |
| MLflow `Invalid Host header` | Add the hostname to `--allowed-hosts` in `docker-compose.yml`. |
| `curl` in PowerShell behaves oddly | Use `curl.exe`. |
| `seeds/__pycache__/` appears | Created by the tests importing `build_sqlite.py`; safe to delete or git-ignore. |

---

## 9. Where to read more

| Document | Contents |
|---|---|
| `ml/README.md` | design decisions, full results table, limitations |
| `ml/reports/eda/eda_report.md` | what the data contains and why the model looks the way it does |
| `ml/reports/evaluation/summary.md` | latest accuracy, benchmarks, gates and promotion decisions |
| `ml/configs/pipeline.yaml` | every tunable setting, with comments |
| `.claude/skills/senetra-ml-architecture` | architecture, invariants, how to extend |
| `.claude/skills/senetra-eda` | running and interpreting EDA |
| `.claude/skills/senetra-feature-engineering` | feature catalog and leakage rules |
| `.claude/skills/senetra-federated-training` | federation algorithms, tuning, privacy contract |
| `.claude/skills/senetra-model-evaluation` | backtest design, metrics, gates |
| `.claude/skills/senetra-prediction-serving` | scenarios, CLI and API usage |
| `.claude/skills/senetra-model-monitoring` | drift/performance checks and alert triage |
| `.claude/skills/senetra-mlops-docker` | Docker/MLflow operations |

In Claude Code, these skills load automatically when you ask about the matching task (for example
"forecast ORS for Coimbatore during an outbreak" or "why did monitoring alert?").

---

## 10. Quick reference

```bash
cd ml
uv sync                                   # setup
uv run pytest                             # all tests
uv run senetra-ml pipeline                # eda → train → evaluate
uv run senetra-ml predict --level district --entity-id 4 --scenario outbreak --severity 8
uv run senetra-ml monitor
uv run senetra-ml serve --port 8080       # http://localhost:8080/docs
docker compose up -d mlflow api           # containerized stack
```

Known limitations: the seeded data is synthetic with a single outbreak episode; only India has PHC data;
federation runs in one process (no network transport yet); no formal differential-privacy accounting; the
Flask `backend/` does not call the forecast API yet.
