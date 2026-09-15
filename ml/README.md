# SENETRA ML — hierarchical federated forecasting

Forecasts the next 7 days of **patient footfall, bed occupancy, staff availability and consumption of
ORS, Paracetamol, IV Fluids, Antibiotics and Insulin** for any **PHC, district, country or the world**,
under **normal, observed, simulated or what-if outbreak (pandemic)** scenarios — with prediction
intervals and stock-out risk. Models are trained federatedly (PHC -> district -> country -> world),
tracked and registered in MLflow, and served by a FastAPI service packaged with Docker.

## Principles

- **The database is the only copy of the data.** Pipelines read `../senetra.db` in place (read-only).
  No CSV/parquet extracts; MLflow stores aggregates, predictions, parameters and a DB fingerprint.
- **Raw PHC rows never leave the PHC.** Row-level reads are PHC-scoped; PHCs share summed
  loss/gradient/Hessian statistics, row counts and feature moments; districts additionally see daily
  outbreak-flag counts.
- **Findings drive design, and evaluation is honest.** Every model is compared with naive baselines,
  local-only models and a centralized XGBoost on identical rows; cold-start outbreaks are reported, not hidden.

## What the data says (`reports/eda/eda_report.md`)

- 1,339 PHCs in 26 districts, all in Tamil Nadu (India); other BRICS countries have no PHC data yet; no coordinates.
- Within a regime, daily values are ~independent noise (|ACF| < 0.05, no weekday effect).
- One outbreak (2026-07-30..08-14) multiplies demand as a step: ORS x2.0, Paracetamol x1.8, IV Fluids x1.5,
  footfall x1.78, bed occupancy x1.44. Antibiotics, Insulin and staff availability do not react.
- District aggregation cuts relative noise to ~15% of PHC level.
- The stock ledger does not chain (0.6%); simulation events show no effect in history.

## Modelling approach

| Stage | Design |
|---|---|
| Features (`features/`) | Row = (PHC, origin t, horizon h). History windows (<= t) of the target, footfall, beds, own outbreak flags and district prevalence; calendar; **`scenario_outbreak`** = district outbreak regime on the target day (realized 0/1 in training, scenario intensity at inference). |
| Model (`federated/glm.py`) | GLM: Poisson/log link for counts (multiplicative outbreak effect), Gaussian/identity for percentages. |
| Federation (`federated/trainer.py`) | **Exact hierarchical federated Newton**: summed statistics flow PHC -> district -> country -> world, so the world model equals a centralized fit. Country, district and PHC models refine their parent with a prior worth N rows. FedAvg/FedProx (with clipping and DP noise) is available but converges to a biased point on this data. |
| Backtest (`training/backtest.py`) | 4 rolling-origin folds covering cold start (no outbreak seen), active outbreak, recovery and the latest period; oracle and observed scenarios. |
| Selection & calibration (`evaluation/calibration.py`) | Serving federation level chosen by backtest PHC MAE; split-conformal intervals per level x horizon. |
| Evaluation (`pipelines/evaluate.py`) | WAPE/MAE/bias per level, regime, transition, horizon, cold start; noise floor; cross-fitted coverage; promotion gates -> `@champion` / `@challenger`. |
| Serving (`inference.py`, `serving/api.py`) | Features rebuilt at the origin from the DB, scenario applied, bottom-up reconciliation, days-of-supply risk. |
| Monitoring (`pipelines/monitor.py`) | Data contract, reporting ratio, regime-matched PSI drift, performance replay vs backtest reference. |

## Results (backtest on `senetra.db`, see `reports/evaluation/summary.md`)

District-level WAPE, warm folds (training had seen an outbreak), same rows for every model.
*observed* = current outbreak state persists (operational); *scenario* = correct scenario supplied.

| target | federated (observed) | federated (scenario) | centralized XGBoost | local-only | 7-day mean | PHC MAE / noise floor |
|---|---:|---:|---:|---:|---:|---:|
| patient_footfall | 3.8% | 2.5% | 4.0% | 7.7% | 8.8% | 24.4 / 23.5 |
| bed_occupancy | 2.5% | 1.8% | 2.7% | 4.2% | 5.4% | 8.3 / 8.0 |
| staff_availability | 0.8% | 0.8% | 0.8% | 1.1% | 0.8% | 5.0 / 5.0 |
| ors | 4.7% | 3.2% | 6.7% | 11.7% | 10.6% | 13.7 / 13.2 |
| paracetamol | 4.1% | 2.9% | 5.6% | 7.8% | 9.1% | 19.6 / 18.9 |
| iv_fluids | 4.1% | 3.3% | 4.7% | 7.7% | 7.2% | 2.9 / 2.9 |
| antibiotics | 3.1% | 3.1% | 3.1% | 6.2% | 3.3% | 6.3 / 6.3 |
| insulin | 4.1% | 4.1% | 3.6% | 7.1% | 3.8% | 1.0 / 1.0 |

- The federated model matches or beats the centralized benchmark without pooling rows, and PHC-level errors
  sit within ~5% of the non-deployable noise floor: the remaining error is irreducible day-to-day noise.
- During outbreaks with the right scenario, district WAPE stays at 1-4%. With the observed scenario, errors
  concentrate on the days an outbreak starts or ends.
- **Cold start** (no outbreak ever seen in training): 30-50% district WAPE on outbreak days for every model,
  including XGBoost — no model can learn an effect it has never observed. This is why the scenario API exists.
- Cross-fitted 80% intervals cover 77-79% at district level. All 8 targets passed the promotion gates.
- Replaced designs, kept for the record: FedAvg reached 12-15% district WAPE on the same folds (biased fixed
  point); a continuous prevalence covariate under-estimated the outbreak effect (x1.54 vs x1.92).

## Quickstart (local)

```bash
cd ml
uv sync
uv run senetra-ml validate
uv run senetra-ml pipeline                 # eda -> train -> evaluate (promotes champions that pass gates)
uv run senetra-ml predict --targets patient_footfall ors --level district --entity-id 1 --scenario normal
uv run senetra-ml predict --targets patient_footfall ors --level district --entity-id 1 --scenario outbreak --severity 9
uv run senetra-ml monitor
uv run senetra-ml serve --port 8080        # http://localhost:8080/docs
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
uv run pytest
```

## Quickstart (Docker)

```bash
cd ml
docker compose build
docker compose up -d mlflow api                  # MLflow http://localhost:5000, API http://localhost:8001/docs
docker compose --profile jobs run --rm pipeline  # train + evaluate into the MLflow server
curl -X POST http://localhost:8001/v1/models/reload
docker compose --profile monitoring up -d monitor
```

## API examples

```bash
curl "http://localhost:8001/v1/forecast/patient_footfall?level=district&entity_id=1&scenario=normal"
curl "http://localhost:8001/v1/forecast/ors?level=phc&entity_id=42&scenario=outbreak&severity=9"
curl "http://localhost:8001/v1/forecast?level=world&scenario=simulated"        # every target
curl "http://localhost:8001/v1/federation/ors"                                   # rounds, levels, drivers, privacy contract
```

Medicine responses include `stock.current_stock`, `days_of_supply`, `days_of_supply_pessimistic` and
`risk_level` (CRITICAL <= 3 days, HIGH <= 7, WATCH <= 14).

## Layout

```
configs/pipeline.yaml      targets, features, federation, backtest, gates, monitoring, MLflow
src/senetra_ml/            data · eda · features · federated · models · training · evaluation · monitoring · pipelines · serving
tests/                     synthetic DB fixture from seeds/DDL.sql; leakage, federated maths, metrics, end-to-end
reports/                   eda/ and evaluation/ (aggregates only); monitoring/ and predictions/ are git-ignored
Dockerfile, docker-compose.yml
```

Operational playbooks for each stage live in `.claude/skills/senetra-*/SKILL.md`.

## Limitations

- The seeded data is synthetic: within-regime noise is i.i.d. and there is a single outbreak episode, so
  outbreak effects come from one episode and severities other than the reference (8) are extrapolations.
- Only one country has PHCs; the country and world tiers currently have a single member.
- Federation is simulated in one process with the exact messages each tier would exchange; a network
  transport (e.g. Flower) is not wired in. No formal differential-privacy accounting is performed.
- No coordinates, so no distance features or transfer-cost modelling.
