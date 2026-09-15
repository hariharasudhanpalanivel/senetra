---
name: senetra-eda
description: Run and extend exploratory data analysis for SENETRA directly on senetra.db (regime detection, noise structure, driver correlations, aggregation effects, data quality). Use when the database or seeds change, before any modelling decision, when a model behaves unexpectedly, or when asked to find patterns in PHC, patient or medicine data.
---

# SENETRA EDA

EDA runs in place on the database and persists only aggregates. Findings are computed, not typed in.

## Run

```bash
cd ml
uv run senetra-ml validate          # data contract; exit 1 on errors
uv run senetra-ml eda               # writes reports/eda/{eda_report.md, eda_summary.json, *.png}, logs to MLflow senetra-eda
uv run senetra-ml eda --no-track    # without MLflow
```

Read `reports/eda/eda_report.md` top-down: **Modelling implications** first, then the evidence tables.

## How to read each section

| Section | Question it answers | Act when |
|---|---|---|
| Geography and federation coverage | Which countries/districts/PHCs actually have data? Constant PHC attributes? | A tier has one member -> federation there is structural only; constant attributes -> never use as features |
| Outbreak regimes | Are outbreak periods detectable, and do surveillance and demand agree? | Agreement < 90% -> revisit `data.outbreak_prevalence_threshold` |
| Targets by regime | Level, CV and outbreak uplift per target | Uplift ~1 -> no outbreak response; uplift >> 1 -> scenario covariate matters |
| Max within-regime ACF / lag-1 pooled | Is there temporal memory beyond the regime? | Pooled lag-1 high but within-regime ~0 -> autocorrelation is the regime, not persistence |
| Noise versus aggregation level | How much noise averages out at district/network level | Guides which level to promise accuracy at |
| Driver correlations pooled / within | Is a driver causal per PHC or just a regime proxy? | Pooled high + within ~0 -> regime proxy only |
| Data quality signals | Label consistency, stock ledger, simulation-event alignment | Ledger < 95% -> risk must use `inventory`; events without effect -> scenario inputs only |
| Inventory days of supply | Current stock-out exposure | Share <= 3 days drives CRITICAL alerts |

## Rules

- Query through `SenetraRepository`; for PHC-level series use `load_panel` (PHC-scoped reads).
  Never `SELECT *` rows into files, notebooks outputs or MLflow artifacts.
- Separate pooled from within-regime statistics — pooled correlations on this data are dominated by the outbreak.
- Derive regimes from data (surveillance prevalence, robust demand shifts), never from knowledge of how seeds were generated.
- Report counts and shares with their denominators.

## Extending

1. Add a function in `ml/src/senetra_ml/eda/analysis.py` returning JSON-safe aggregates.
2. Register it in `run_analysis` and, if it implies a modelling decision, add a threshold-based rule in `implications`.
3. Add a table/figure in `eda/report.py`.
4. Assert the new key in `tests/test_pipeline_e2e.py::test_eda_report`, then `uv run pytest`.
