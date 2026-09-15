---
name: senetra-model-monitoring
description: Monitor SENETRA data and models in production — data freshness/reporting, regime-matched feature and target drift (PSI), live performance replay against backtest references, alert triage and retraining decisions. Use when scheduling or running monitoring, when an alert fires, after new data arrives, or when deciding whether to retrain.
---

# SENETRA model monitoring

## Run

```bash
cd ml
uv run senetra-ml monitor                          # all targets, window = monitoring.window_days
uv run senetra-ml monitor --targets ors --window-days 21
uv run senetra-ml monitor --fail-on-alert          # exit 2 on alert (CI/cron)
uv run senetra-ml monitor --interval-seconds 86400 # loop (Docker `monitor` service)
```

Outputs `reports/monitoring/latest.json` (+ timestamped copy) and a `senetra-monitoring` MLflow run with
`alerts`, `warnings`, `reporting_ratio`, `<target>.max_psi`, `<target>.phc_mae_ratio`, `<target>.district_wape_ratio`.
Status is `ok`, `warn` or `alert`.

## Checks

| Check | How | Severity |
|---|---|---|
| Data contract | `validate_database` (ranges, nulls, completeness, freshness, label consistency, ledger) | error -> alert, warn -> warn |
| Reporting ratio | PHCs reporting on the latest day / all PHCs < `min_reporting_ratio` | alert |
| Regime shift | outbreak share of recent rows vs training differs by > 25 points | warn |
| Drift | PSI per feature and target, recent rows vs the **same-regime** reference profile stored in the model | >= `psi_alert`: alert (warn for regime features); >= `psi_warn`: warn |
| Performance | replay champion over the window with the observed scenario; PHC MAE and district WAPE / backtest reference | ratio > `degradation_ratio_alert`: alert |
| In-sample window | share of window dates that were in training | > 50%: warn (errors optimistic) |

Regime-matched PSI matters: an outbreak shifts footfall and demand by design, and rolling windows carry that
shift for up to 28 days. Rows are split into **normal** (no outbreak in the long window), **recovery** (an
outbreak inside the window) and **outbreak** (outbreak on the origin or target day), and each is compared with
the same regime in the model's reference profile (warm-up rows excluded). Regime features
(`scenario_outbreak`, `dprev_*`, `flag_share7`) only warn; in recovery, rolling-window features also only warn
because their values depend on days since the outbreak ended. Target PSI keeps full severity in every regime.
Seen on the seeded DB: before this split, 39 false alerts on `y_mean14/28` after the July-August outbreak.

## Triage

| Alert | Likely cause | Action |
|---|---|---|
| `freshness`, `reporting_ratio`, `completeness` | Upstream ingestion stopped | Fix data feed; do not retrain on partial data |
| `*_range`, `metric_nulls` errors | Bad data | Quarantine/correct rows; rerun validate |
| `drift:normal:<feature>` alert | Real change in normal operations (new PHCs, reporting practice) | Run `senetra-ml eda`, confirm, retrain + evaluate |
| `drift:outbreak:*` or `regime_shift` | Outbreak started/ended | Use `observed`/`simulated` scenarios; check intervals; retrain after the episode to learn its response |
| `performance:*_ratio` | Model no longer matches reality | `senetra-ml train` -> `evaluate`; the gate protects the champion |
| `model_available` | No champion for a target | Train and evaluate that target |

## Retraining policy

Retrain when: a performance alert persists for two runs, an outbreak episode ends (new response data),
new countries/districts onboard, or weekly by schedule. Always `evaluate` before serving; promotion is gated.
