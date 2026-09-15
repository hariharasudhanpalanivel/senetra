---
name: senetra-mlops-docker
description: Operate the SENETRA ML stack with Docker Compose and MLflow — build images, run the tracking/registry server, forecast API, training/evaluation jobs, prediction writer and scheduled monitoring; manage environments, volumes, aliases and troubleshooting. Use when deploying, running pipelines in containers, wiring the backend to the ML API, or debugging MLflow/Docker issues.
---

# SENETRA MLOps with Docker and MLflow

All commands from `ml/`. The image is built with uv from `uv.lock` (reproducible), runs as non-root,
and exposes the `senetra-ml` CLI as entrypoint.

## Services (`docker-compose.yml`)

| Service | Profile | What | Ports / volumes |
|---|---|---|---|
| `mlflow` | default | Tracking server + model registry (SQLite backend, proxied artifacts) | `5000`, volume `mlflow-data` |
| `api` | default | FastAPI forecasts from `@champion` models | `8001 -> 8080`, DB read-only |
| `pipeline` | `jobs` | `validate -> eda -> train -> evaluate` (override command for any stage) | DB read-only, `./reports` |
| `predict-writer` | `jobs` | Writes PHC medicine forecasts to `predictions` | DB **read-write** |
| `monitor` | `monitoring` | Daily monitoring loop | DB read-only, `./reports` |

## Commands

```bash
docker compose build
docker compose up -d mlflow api                         # UI http://localhost:5000, API http://localhost:8001/docs
docker compose run --rm pipeline                        # full training cycle into the server registry
docker compose run --rm pipeline train --targets ors    # any CLI stage
docker compose run --rm pipeline evaluate
docker compose run --rm pipeline monitor --fail-on-alert
curl -X POST http://localhost:8001/v1/models/reload     # after a new champion
docker compose --profile jobs run --rm predict-writer
docker compose --profile monitoring up -d monitor
docker compose logs -f api
```

`SENETRA_DB_FILE=/abs/path/senetra.db docker compose ...` points at another database (default `../senetra.db`).

## Environment

| Variable | Meaning |
|---|---|
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` in compose; unset locally -> `sqlite:///ml/mlruns/mlflow.db` |
| `SENETRA_DB_PATH` | DB path inside the process (compose: `/data/senetra.db`) |
| `SENETRA_CONFIG` | alternative pipeline YAML |
| `SENETRA_ML_HOME` | directory with `configs/` and `reports/` (image: `/app`) |

Local and Docker registries are separate stores; promote in the one the API reads.

## Release flow

1. `uv run pytest` (must be green) -> `docker compose build`.
2. `docker compose run --rm pipeline` -> check `reports/evaluation/summary.md` and gates.
3. `POST /v1/models/reload`; verify `GET /health` shows the new versions.
4. Rollback: point `@champion` to the previous version in the MLflow UI (or `MlflowClient().set_registered_model_alias`) and reload.

## Troubleshooting

- **`Invalid Host header` from MLflow**: add the hostname to `--allowed-hosts` in the `mlflow` command.
- **API `503`**: no `@champion` for that target -> run `pipeline` / `evaluate`.
- **`attempt to write a readonly database`**: only `predict-writer` mounts the DB read-write, by design.
- **Bind-mount permission denied on Linux**: the container user is uid 10001; `chown 10001 reports` or set `user:` in an override file.
- **Jobs killed / out of memory**: a full training run holds the whole panel (1,339 PHCs x 588 rows x ~17
  features, float64) plus XGBoost matrices, ~1.5 GB peak. Do not run training, a `docker compose build` and the
  MLflow/API containers at the same time on a 16 GB laptop; Docker Desktop's WSL VM keeps several GB after
  `docker compose down`. Run stages sequentially, or train fewer targets per run (`--targets`).
  A killed run stays `RUNNING` in MLflow; mark it with `MlflowClient().set_terminated(run_id, "KILLED")`.
- **Stale forecasts**: panels are cached 60 s per origin/scope; `POST /v1/models/reload` clears caches.
- **Backups**: the registry and artifacts live in volume `mlflow-data`; back it up with `docker run --rm -v senetra-ml_mlflow-data:/m -v $PWD:/b busybox tar czf /b/mlflow-backup.tgz /m`.
