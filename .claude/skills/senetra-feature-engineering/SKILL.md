---
name: senetra-feature-engineering
description: Leakage-safe feature engineering for SENETRA forecasts — feature catalog, as-of-origin semantics, the outbreak scenario covariate, federated standardization and how to add or change features. Use when adding, removing or debugging features, when a backtest looks too good, or when training and serving features disagree.
---

# SENETRA feature engineering

Code: `ml/src/senetra_ml/features/engineering.py` (construction) and `features/transform.py` (fitted transform).
The same `build_features` runs in training, backtesting, monitoring and online inference — there is no second implementation.

## Row definition

A row is **(PHC c, origin day t, horizon h in 1..7)** and predicts the target on day **d = t + h**.
Arrays are `(C, N, F)` with N = origins x horizons; rows share the origin/horizon grid across PHCs.

## Catalog

| Feature | Uses data from | Meaning |
|---|---|---|
| `y_last`, `y_mean{3,7,14,28}` | <= t | Target level (log1p for Poisson targets) |
| `y_disp7`, `y_trend` | <= t | 7-day dispersion; 7-day vs 28-day trend |
| `ff_mean7`, `ff_trend` | <= t | Footfall level/trend (omitted when footfall is the target) |
| `bed_mean7` | <= t | Bed occupancy level (omitted when it is the target) |
| `flag_share7` | <= t | Share of days this PHC flagged an outbreak |
| `dprev_origin`, `dprev_mean7` | <= t | District outbreak prevalence (aggregate surveillance) |
| `scenario_outbreak` | day d (scenario) | District outbreak regime on the target day: realized 0/1 in training, scenario intensity at inference |
| `dow_sin`, `dow_cos`, `horizon` | calendar | Known in advance |

Baselines built alongside: `bl_naive` (y_t), `bl_seasonal7` (same weekday, past only), `bl_mean7`.

## Scenario covariate

Training value: 1.0 when the PHC's district has outbreak-flag prevalence above
`data.outbreak_prevalence_threshold` on day d, else 0.0 — a step, matching how outbreaks act on demand in EDA.
A continuous prevalence was tried first and rejected: its day-to-day sampling noise attenuated the learned
effect (x1.54 vs x1.92 empirical onset uplift) and pushed outbreak signal into origin-day features.

Inference value (`ForecastService.scenario_intensity`), capped by `FeatureTransform.intensity` at
`scenarios.max_intensity`: `normal` = 0, `observed` = regime at origin, `outbreak` = severity /
`scenarios.severity_reference` (so severity 8 = the historical outbreak, 10 = 1.25x its log-effect),
`simulated` = max(observed, active event severity / reference). Severities other than the reference
extrapolate the single learned episode and must be presented as assumptions.

## Standardization is federated

`fit_transform` uses per-client sufficient statistics (n, sum x, sum x^2) summed across PHCs — never a
pooled matrix. Missing inputs at inference become 0 after standardization (training mean) and are
reported as `imputed_share`.

## Leakage rules

- Only `scenario_prevalence` and calendar columns may read day d. Everything else reads <= t.
- Training rows need `target_idx <= cutoff` (`FeatureSet.train_mask`).
- Seasonal baselines must index `t + h - 7*ceil(h/7)` (<= t for any horizon).
- Fitted quantities (the scaler) are refit per backtest fold from data <= cutoff.

## Adding a feature

1. Compute a `(C, T)` array from the panel using only past-inclusive windows (`trailing_stats`), add it to `history` (or `target_day` only if genuinely known in advance).
2. If it is a regime signal, add its name to `REGIME_FEATURES` in `monitoring/drift.py`; calendar-like features go in `SKIP_FEATURES`.
3. `uv run pytest tests/test_features.py` — the future-tampering test must pass unchanged.
4. Retrain and compare `backtest_phc_mae_fl_*` and district WAPE with the previous run in MLflow; keep the feature only if it improves out-of-sample error.

## Red flags

- PHC-level MAE below the evaluation noise floor -> leakage.
- Feature importance dominated by something that is only known after day t.
- Training/serving skew: never recompute a feature in the API differently from `build_features`.
