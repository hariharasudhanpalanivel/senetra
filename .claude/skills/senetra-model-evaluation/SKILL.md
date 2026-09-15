---
name: senetra-model-evaluation
description: Evaluate SENETRA forecasting runs — rolling-origin backtests, metrics per hierarchy level/regime/horizon, cold-start outbreak analysis, noise floor, conformal coverage, promotion gates and MLflow champion/challenger aliases. Use after training, before promoting or serving a model, when comparing runs, or when asked how accurate predictions are in normal versus pandemic conditions.
---

# SENETRA model evaluation

## Run

```bash
cd ml
uv run senetra-ml evaluate                        # latest finished training run
uv run senetra-ml evaluate --train-run-id <id>    # a specific run
uv run senetra-ml evaluate --no-promote           # metrics and gates only
```

Outputs: `reports/evaluation/summary.md` + `summary.json`; MLflow `senetra-evaluation` runs with
`evaluation/metrics.csv`, `gates.json`, `coverage.json`, `world_h1_backtest.png`, `district_wape_by_horizon.png`;
registry tags `eval.*` on the model version and alias `@champion` (passed) or `@challenger` (failed).

## Backtest design (`training/backtest.py`)

Cutoff c_k = end - horizon - test_origin_days - k * step_days. Training sees targets <= c_k; testing uses
origins c_k .. c_k + 6 at every horizon. A fold whose training window contains no outbreak is a
**cold start**: it shows what happens when an outbreak has never been seen. Cold-start folds are excluded
from headline metrics, gates, level selection and conformal calibration, and reported separately.

Actuals are re-read from the DB at evaluation time (the training artifact has no actuals).

## Metrics

All from additive per-PHC statistics, computed after bottom-up aggregation to each level.

| Metric | Definition | Use |
|---|---|---|
| WAPE | sum abs error / sum actual | Headline at district/country/world |
| MAE, RMSE, bias | usual; bias = sum error / sum actual | PHC level, systematic over/under-forecast |
| Skill vs baseline | 1 - MAE_model / MAE_bl_mean7 | Must be >= 0 at PHC level |
| Noise floor | leave-one-out same-district same-day mean of actuals (not deployable) | PHC MAE near the floor = little left to learn; below it = leakage |
| Coverage80/95 | share of actuals inside conformal intervals, cross-fitted by fold | Interval honesty |

Slices: `all`, `normal`, `outbreak`, `transition` (regime at origin != regime at target), `h1..h7`,
`cold_start_normal`, `cold_start_outbreak`. Model columns: `fl_<level>__observed|oracle`,
`local__*`, `xgb__*`, `bl_*`, `noise_floor_oracle`.

Interpretation:
- **observed** = operational forecast (outbreak state persists). Errors concentrate at regime transitions.
- **oracle/scenario** = accuracy when the scenario is right — what the what-if/simulator forecasts deliver.
- `fl_*` vs `xgb__*` = cost of federation; `fl_*` vs `local__*` = value of federation.

## Gates (`evaluation.gates` in pipeline.yaml)

district WAPE (observed) <= `max_district_wape`; PHC skill vs `baseline` >= `min_phc_skill_vs_baseline`;
district coverage80 within `district_coverage80`; no regression vs current champion beyond
`max_regression_vs_champion`. A gate that cannot be computed is `skip`, never `pass`.

The regression gate only runs when the champion was evaluated on the same data (`eval.data_fingerprint` tag).
After new data arrives the backtest folds shift, so backtest errors of old and new models are not comparable
(seen in testing: 0.289 vs 0.362 on different windows for equally good models). The gate is then `skip`, and
the regression check happens online: the backend canary replays both models on the same fresh actuals
(`senetra-backend-mlops`).

Rules: change thresholds only with a written rationale in the PR, never to force a promotion. Compare
candidates on identical folds (same data fingerprint tag `data.fingerprint`).

## Reference numbers (champions v3, seeded `senetra.db`)

District WAPE observed / scenario: footfall 3.8/2.5%, beds 2.5/1.8%, staff 0.8/0.8%, ORS 4.7/3.2%,
Paracetamol 4.1/2.9%, IV Fluids 4.1/3.3%, Antibiotics 3.1/3.1%, Insulin 4.1/4.1%. Centralized XGBoost is equal
or worse except Insulin (3.6%); local-only is 1.5-2.5x worse; PHC MAE is within ~5% of the noise floor;
coverage80 77-79%; cold-start outbreak WAPE 30-50% for all models. A new candidate that is clearly worse
than these on the same data fingerprint is a regression, not noise.

Diagnosing a bad run: break district bias down by fold x regime x model (oracle scenario). Bias in every
model -> data/feature problem; bias only in `fl_*` -> federation/solver problem (this is how the FedAvg bias
and a Gaussian clipping bug were found); bias only in `xgb__*` -> benchmark setup.

## Quick comparisons in MLflow

Filter `senetra-evaluation` child runs by `tags.senetra.target`, chart `district.fl_<level>__observed.all.wape`
against `district.xgb__observed.all.wape` and `district.bl_mean7.all.wape`.
