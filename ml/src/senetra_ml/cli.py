"""senetra-ml command line interface."""

from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

from senetra_ml.config import LEVELS, load_config  # noqa: E402

log = logging.getLogger("senetra_ml")


def _print(data) -> None:
    print(json.dumps(data, indent=2, default=str))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="senetra-ml", description="SENETRA hierarchical federated forecasting")
    parser.add_argument("--config", help="pipeline YAML (default: configs/pipeline.yaml)")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate", help="run data contract checks on the database")

    eda = sub.add_parser("eda", help="exploratory analysis report")
    eda.add_argument("--output-dir")
    eda.add_argument("--no-track", action="store_true", help="skip MLflow logging")

    train = sub.add_parser("train", help="backtest, fit and register models")
    train.add_argument("--targets", nargs="*")
    train.add_argument("--no-register", action="store_true")

    evaluate = sub.add_parser("evaluate", help="evaluate a training run and manage aliases")
    evaluate.add_argument("--train-run-id")
    evaluate.add_argument("--targets", nargs="*")
    evaluate.add_argument("--no-promote", action="store_true")

    predict = sub.add_parser("predict", help="forecast with champion models")
    predict.add_argument("--targets", nargs="*")
    predict.add_argument("--level", choices=LEVELS, default="district")
    predict.add_argument("--entity-id", type=int)
    predict.add_argument("--scenario", choices=["observed", "normal", "outbreak", "simulated"], default="observed")
    predict.add_argument("--severity", type=float)
    predict.add_argument("--districts", nargs="*", type=int, help="limit an outbreak scenario to these districts")
    predict.add_argument("--origin", help="forecast origin date (default: latest data)")
    predict.add_argument("--write-db", action="store_true", help="store PHC medicine forecasts in `predictions`")
    predict.add_argument("--output")
    predict.add_argument("--allow-unpromoted", action="store_true", help="use the latest version if no champion")

    monitor = sub.add_parser("monitor", help="data quality, drift and performance monitoring")
    monitor.add_argument("--targets", nargs="*")
    monitor.add_argument("--window-days", type=int)
    monitor.add_argument("--fail-on-alert", action="store_true")
    monitor.add_argument("--interval-seconds", type=int, help="repeat forever with this interval")

    serve = sub.add_parser("serve", help="run the forecast API")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8080)

    pipeline = sub.add_parser("pipeline", help="validate -> eda -> train -> evaluate")
    pipeline.add_argument("--targets", nargs="*")
    pipeline.add_argument("--skip-eda", action="store_true")
    pipeline.add_argument("--no-promote", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("alembic", "urllib3", "mlflow.store.db.utils"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load_config(args.config)

    if args.command == "validate":
        from senetra_ml.data.repository import SenetraRepository
        from senetra_ml.data.validation import validate_database

        with SenetraRepository(cfg.data.db_path) as repo:
            report = validate_database(repo, cfg)
        _print(report.to_dict())
        return 1 if report.errors else 0

    if args.command == "eda":
        from senetra_ml.pipelines.eda import run_eda

        summary = run_eda(cfg, args.output_dir, track=not args.no_track)
        _print({"implications": summary["implications"], "report": str(cfg.reports_dir / "eda" / "eda_report.md")})
        return 0

    if args.command == "train":
        from senetra_ml.pipelines.train import run_training

        _print(run_training(cfg, args.targets, register=not args.no_register))
        return 0

    if args.command == "evaluate":
        from senetra_ml.pipelines.evaluate import run_evaluation

        result = run_evaluation(cfg, args.train_run_id, args.targets, promote=not args.no_promote)
        _print({"run_id": result["run_id"], "summary": result["summary_path"],
                "decisions": {r["target"]: r["decision"] for r in result["results"]}})
        return 0

    if args.command == "predict":
        from senetra_ml.inference import Scenario
        from senetra_ml.pipelines.predict import run_prediction

        scenario = Scenario(args.scenario, args.severity, tuple(args.districts) if args.districts else None)
        result = run_prediction(cfg, args.targets, args.level, args.entity_id, scenario, args.origin,
                                args.write_db, args.output, args.allow_unpromoted)
        _print(result)
        return 1 if result["errors"] and not result["forecasts"] else 0

    if args.command == "monitor":
        from senetra_ml.pipelines.monitor import run_monitoring

        while True:
            report = run_monitoring(cfg, args.targets, args.window_days)
            _print({"status": report["status"], "alerts": report["alerts"]})
            if not args.interval_seconds:
                return 2 if args.fail_on_alert and report["status"] == "alert" else 0
            time.sleep(args.interval_seconds)

    if args.command == "serve":
        import uvicorn

        from senetra_ml.serving.api import create_app

        uvicorn.run(create_app(cfg), host=args.host, port=args.port)
        return 0

    if args.command == "pipeline":
        from senetra_ml.pipelines.evaluate import run_evaluation
        from senetra_ml.pipelines.train import run_training

        if not args.skip_eda:
            from senetra_ml.pipelines.eda import run_eda

            run_eda(cfg)
        trained = run_training(cfg, args.targets)
        result = run_evaluation(cfg, trained["run_id"], promote=not args.no_promote)
        _print({"train_run_id": trained["run_id"], "summary": result["summary_path"],
                "decisions": {r["target"]: r["decision"] for r in result["results"]}})
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
