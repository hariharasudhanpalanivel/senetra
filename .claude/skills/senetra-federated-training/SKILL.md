---
name: senetra-federated-training
description: Train, tune and debug SENETRA's hierarchical federated models (PHC clients -> district -> country -> world; exact federated Newton or FedAvg/FedProx on a Poisson/Gaussian GLM) with MLflow tracking and registry. Use when running or changing training, tuning federation hyperparameters, adding countries, reading federation history, or explaining what data leaves a PHC.
---

# SENETRA federated training

## Run

```bash
cd ml
uv run senetra-ml train                              # all targets, registers senetra-forecast-<target>
uv run senetra-ml train --targets ors paracetamol    # subset
uv run senetra-ml train --no-register                # experiment only
uv run senetra-ml pipeline                           # validate -> eda -> train -> evaluate
```

Local tracking: `sqlite:///ml/mlruns/mlflow.db` (UI: `uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`).
Docker/remote: set `MLFLOW_TRACKING_URI`.

## What one training run does (per target)

1. `build_features` on the in-memory panel (PHC-scoped reads).
2. `run_backtest`: for each rolling-origin fold fit **fl_{world,country,district,phc}** (one federated run
   yields all four levels), **local** (PHC alone), **xgb** (centralized benchmark), plus baselines;
   predict under **oracle** (true scenario) and **observed** (persistence) scenarios.
3. `select_level`: federation level with the lowest PHC MAE (observed, non-cold-start folds) becomes the
   serving `parameter_level`; unseen PHCs fall back to district -> country -> world.
4. Final federated fit on all history, conformal tables per level x horizon, regime-matched drift profile.
5. Log to MLflow `senetra-training` (parent run + one child per target) and register the pyfunc model.

## Algorithms (`federated/trainer.py`, `federated/glm.py`)

**`newton` (default) — exact hierarchical federated Newton (GLORE-style).**
Each round every PHC evaluates the summed loss, gradient and Hessian of its own rows at the current world
model; districts add their PHCs' sums, countries add district sums, the world adds country sums and takes a
damped Newton step with backtracking on the summed loss. The world model therefore equals a centralized
fit (`test_federated_newton_world_model_equals_centralized_fit`) without moving rows.
Then `group_newton` refines downward: country models on country sums, district models on district sums,
PHC models on their own rows — each shrunk toward its parent with a prior worth `*_prior_rows` rows, so
small groups stay close to the parent and data-rich groups can diverge.

**`fedavg` — FedAvg/FedProx.** Per round, PHCs take `local_newton_steps` proximal Newton steps and districts
average parameter deltas (`district_rounds` times), then countries and the world average. Supports
`update_clip_norm` + `dp_noise_multiplier`. On SENETRA data it stalls at a biased fixed point (objective
-142.885 vs -143.204 centralized on the synthetic test data, independent of rounds), so use it only for
privacy-mechanism experiments.

Batching clients in one array is a simulation speed-up; every reduction is within-client
(`test_client_update_depends_only_on_its_own_rows`).

## Knobs (`federated:` in pipeline.yaml)

| Knob | Algorithm | Effect | Tune when |
|---|---|---|---|
| `rounds` | both | World Newton iterations / global rounds | `final_world_objective` still falling at the end -> increase |
| `l2` | both | Ridge on slopes | Unstable coefficients -> increase |
| `level_newton_steps` | newton | Refinement steps at country and district level | 0 disables refinement (levels = world) |
| `country_prior_rows`, `district_prior_rows` | newton | Shrinkage toward the parent level | `fl_district` worse than `fl_country` in backtest -> increase |
| `phc_prior_rows`, `personalization_steps` | both | PHC personalization strength | `fl_phc` worse than `fl_district` -> increase prior rows or set steps 0 |
| `district_rounds`, `local_newton_steps`, `proximal_mu`, `client_fraction` | fedavg | Local work, drift control, participation | Divergence -> raise `proximal_mu` |
| `update_clip_norm`, `dp_noise_multiplier` | fedavg | Clipping / Gaussian mechanism | Privacy experiments. No epsilon accounting — never claim a formal DP guarantee |

## Reading results in MLflow

- Child run metrics: `backtest_phc_mae_fl_<level>` (which tier generalizes best), `reference_phc_mae`,
  `reference_district_wape`, `fold<k>_world_objective` / `final_world_objective` per round (convergence).
- Tags: `senetra.selected_level`, `senetra.model_version`.
- Artifacts: `backtest/predictions.parquet` (predictions only), `backtest/summary.json`, `model/` (pyfunc with bundle).
- Bundle metadata: `outbreak_effect` (x demand at intensity 1 holding inputs fixed), `world_coefficients`
  (standardized "key drivers"), `hierarchy`, `privacy_contract`.

## Privacy contract (keep code and docs in sync with `PRIVACY_CONTRACT` in pipelines/train.py)

PHC rows stay at the PHC. PHCs send summed loss/gradient/Hessian statistics (newton) or parameter deltas
(fedavg), row counts and feature moments.
Districts additionally see daily outbreak-flag counts. Country and world tiers see parameters and counts.
Centralized XGBoost pools rows and is a backtest benchmark only — never register or serve it.

## Troubleshooting

- `No training rows` -> not enough history before the cutoff; check `data.min_history_days` and data range.
- NaN parameters -> check for constant/NaN features in validation; raise `l2`/`proximal_mu`.
- Memory: arrays are (PHCs x origins x horizons x features) float64; train fewer targets per run if needed.
- Adding a country: nothing to code; retrain. With one country, country == world by construction.
