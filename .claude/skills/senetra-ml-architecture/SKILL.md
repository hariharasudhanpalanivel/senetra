---
name: senetra-ml-architecture
description: Architecture and invariants of the SENETRA hierarchical federated forecasting system (PHC -> district -> country -> world). Use before changing anything under ml/, when deciding where a new capability belongs, when adding a target, feature, country or model, or when someone asks how SENETRA predicts patient counts and medicine demand in normal and outbreak conditions.
---

# SENETRA ML architecture

SENETRA forecasts, 1-7 days ahead and at PHC, district, country and world level:
patient footfall, bed occupancy, staff availability and consumption of ORS, Paracetamol,
IV Fluids, Antibiotics and Insulin — under normal, observed, simulated or what-if outbreak scenarios.
Code lives in `ml/` (package `senetra_ml`), configuration in `ml/configs/pipeline.yaml`.

## Non-negotiable invariants

1. **The database is the single copy of the data.** Pipelines read `senetra.db` in place
   (`SenetraRepository`, read-only URI). Never export rows to CSV/parquet, never log actuals or
   feature matrices to MLflow. Artifacts hold aggregates, predictions, parameters and metadata only.
   Lineage is recorded as a DB fingerprint tag, not a data copy.
2. **Federated data boundary.** Row-level PHC records are read only through PHC-scoped queries
   (`phc_metric_rows`, `phc_consumption_rows`). Cross-PHC reads must be aggregates
   (`district_surveillance`) or be reduced immediately to additive statistics
   (`target_actuals` in evaluation/monitoring). Clients share summed loss/gradient/Hessian statistics
   (or parameter deltas under FedAvg), row counts and feature moments — nothing else.
3. **No leakage.** A row is (PHC, origin t, horizon h) predicting day t+h. History features use
   data <= t. The only target-day inputs are the calendar and the scenario covariate. Training rows
   for a cutoff c need target day <= c. `tests/test_features.py` enforces this — keep it green.
4. **Honest evaluation.** Every model is compared with naive baselines, a centralized XGBoost and
   local-only models on identical rows; cold-start folds are reported, never hidden; gates are never
   loosened to force a promotion.

## Why the design looks like this (from EDA — rerun `senetra-ml eda` if data changes)

| Finding in `ml/reports/eda/eda_report.md` | Design consequence |
|---|---|
| Within a regime, daily values are ~i.i.d. (max within-regime ACF < 0.05, no weekday effect) | Accuracy is bounded by noise; report skill vs baselines and a noise floor; lag features kept for real data but carry little weight |
| Outbreaks multiply demand as a step (ORS x2.0, Paracetamol x1.8, IV x1.5, footfall x1.78); Antibiotics/Insulin/staff do not react | Log-link Poisson GLM with a binary target-day outbreak covariate, scaled by severity at inference |
| FedAvg on this data converges to a biased point (clients have near-collinear designs) | Exact federated Newton on summed gradients/Hessians is the default trainer |
| No leading indicator: demand jumps the same day surveillance flags rise | Forecasts are scenario-conditioned; "observed" persistence is the operational default |
| District aggregation cuts noise to ~15% of PHC level | Bottom-up hierarchical forecasts; district/country/world are far more precise than PHC |
| Footfall correlates with medicines only through the regime | Footfall is a regime signal, not a per-PHC causal driver |
| Stock ledger does not chain; `inventory` snapshot is coherent | Risk = inventory snapshot / forecast demand (days of supply) |
| `simulation_events` show no effect in history | Events are scenario inputs (`simulated`), never labels |
| Only India (Tamil Nadu) has PHCs; no coordinates | Country/world tiers run with one member today; no distance features |
| One outbreak episode | Outbreak response is estimated from one episode; cold-start fold quantifies the risk |

## Components

```
data/        repository (SQL, scoped reads) · panel (in-memory C x T x K array) · validation (data contract)
eda/         analysis (data-derived findings) · report (markdown + figures)
features/    engineering (as-of-origin features, baselines) · transform (intensity scaling, federated standardization)
federated/   glm (batched Newton, Poisson/Gaussian) · strategy (weighted means, clipping, DP noise) · trainer (hierarchical FedAvg)
models/      centralized (XGBoost benchmark) · bundle (deployable params + calibration + reconciliation) · pyfunc (MLflow model)
training/    backtest (rolling-origin folds, all variants, both scenarios)
evaluation/  metrics (additive stats, level aggregation) · calibration (level selection, conformal, coverage)
monitoring/  drift (regime-matched PSI)
pipelines/   eda · train · evaluate · predict · monitor
inference.py ForecastService (features at origin + scenario + model + risk) · serving/api.py FastAPI · cli.py
```

## Stage map

`validate -> eda -> train (backtest + final fit + register) -> evaluate (gates -> @champion/@challenger) -> predict/serve -> monitor -> retrain`

Detailed skills: `senetra-eda`, `senetra-feature-engineering`, `senetra-federated-training`,
`senetra-model-evaluation`, `senetra-prediction-serving`, `senetra-model-monitoring`, `senetra-mlops-docker`.

## Extending safely

- **New target** (e.g. another medicine): add it under `targets:` in `pipeline.yaml`
  (`source`, `medicine`/`column`, `family`, `aggregation`), run `senetra-ml validate`, `train --targets <name>`,
  `evaluate`. Counts -> `poisson` + `sum`; percentages -> `gaussian` + `mean` + `bounds`.
- **New country**: load its PHCs into the same schema. The hierarchy is read from the DB, so the country
  tier gains a member automatically; retrain and compare `backtest_phc_mae_fl_*` per level.
- **New model family**: add it as a benchmark column in `training/backtest.py::predict_variants` first.
  It may only replace the federated GLM for serving if it can be trained without pooling rows.
- **New scenario type**: extend `inference.Scenario` and `ForecastService.scenario_prevalence`;
  document that effects outside the observed outbreak range are extrapolated (intensity is capped by
  `scenarios.max_intensity`).

Always finish with `cd ml && uv run pytest`.
