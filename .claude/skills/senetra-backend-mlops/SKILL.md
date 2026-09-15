---
name: senetra-backend-mlops
description: Operate and extend the SENETRA backend's production ML loop — forecast serving through stable/canary models, automatic retraining when new data arrives, canary rollout (advance, promote, rollback, revert), scheduler, jobs and the /api/ml endpoints. Use when integrating forecasts into the app, when new data should trigger retraining, when a rollout is stuck or misbehaving, or when changing deployment policy.
---

# SENETRA backend MLOps

Code: `backend/mlops/*`, endpoints in `backend/routes/ml.py`, docs in `backend/README.md`.
The ML library (`ml/`, package `senetra_ml`) is imported in-process; retraining runs in a spawned process.

## Lifecycle

```
new rows in senetra.db ─► data signature changed ─► decide_retrain ─► retrain process
retrain process: validate → train → evaluate (gates, promote=False)
   pass ─► @canary (stage 0)          fail ─► @challenger
   no champion yet ─► @champion directly (bootstrap)
canary: every ML_CANARY_CHECK_INTERVAL_SECONDS ─► decide(): hold | advance | promote | rollback
promote: @champion ← canary, @previous_champion ← old champion
```

## Invariants

- Serving only ever uses registry aliases `champion` and `canary`; never load "latest".
- Retraining never promotes directly while a champion exists — every new model goes through a canary.
- Canary state lives in model-version tags `canary.*` (shared by all instances); every transition is also an
  event in `ml_deployment_events`. Change state only through `DeploymentManager`.
- Forced requests (`X-ML-Variant`) and client errors (4xx) are excluded from canary evidence.
- Latency excludes one-time model loading; stable and canary share the feature cache.
- Accuracy is compared only on target days after both models' training data (`evaluation/online.py`).
  Without such days the rollout caps at `ML_CANARY_MAX_PERCENT_WITHOUT_ACTUALS`.
- The scheduler runs only in the lease holder; spawned job processes and tests never start it.
- The retraining signature ignores `inventory`; stock updates are not training data.

## Common tasks

| Task | How |
|---|---|
| Tell the backend data was loaded | `POST /api/ml/data/arrived` (★) — returns the decision and the job |
| Why no retraining? | `GET /api/ml/data/status` → `reason` |
| Force retraining | `POST /api/ml/retrain {"force": true, "targets": ["ors"]}` (★) |
| Watch a job | `GET /api/ml/jobs/<id>` (`status`, `result.deployments`, `error`) |
| Inspect a rollout | `GET /api/ml/deployments/<target>` (percent, stage traffic, events) |
| Decide now instead of waiting | `POST /api/ml/deployments/<target>/evaluate` (★) |
| Stop a rollout / restore production | `POST …/rollback` / `POST …/revert` (★) |
| QA the canary | send `X-ML-Variant: canary` |

★ = requires `Authorization: Bearer $ML_ADMIN_TOKEN` when set.

## Changing policy

- Retraining rules: `mlops/data_watch.py::decide_retrain` + `ML_RETRAIN_*` settings. Add a case to
  `tests/test_mlops_units.py::test_retraining_policy` for every new rule.
- Rollout rules: `mlops/canary.py::decide` + `ML_CANARY_*` settings. Keep the order safety → accuracy →
  evidence; add a row to `test_canary_decisions`.
- New endpoint: `routes/ml.py`; decorate state-changing ones with `@admin_required`.
- Always run `cd backend && .venv/Scripts/python -m pytest` (unit + end-to-end rollout timeline).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Forecast 503 | No champion yet: `POST /data/arrived` (bootstraps) or train with the CLI |
| Canary stuck on `hold` "waiting for fresh actuals" | Normal until data newer than the canary's training arrives; or promote manually |
| Canary rolled back on latency right after start | Check that model loading is still excluded from `latency_ms` in `predictor.py` |
| Two schedulers acting | Instances use different `ML_STATE_DB` files; point them at one shared state store |
| Job `failed` with "no heartbeat" | Worker process died (often memory); retrain fewer targets via `ML_RETRAIN_TARGETS` |
| Country/PHC endpoints fail | They still use PostgreSQL (`clients/db.py`); ML uses SQLite `senetra.db` |
