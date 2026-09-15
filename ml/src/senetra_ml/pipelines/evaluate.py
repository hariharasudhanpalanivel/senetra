"""Evaluation pipeline.

Joins logged backtest predictions with actuals re-read from the database, reports errors per
hierarchy level, regime, transition, horizon and cold start, measures cross-fitted interval
coverage, applies promotion gates and sets registry aliases (champion / challenger).
"""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402

from senetra_ml.config import LEVELS, PipelineConfig  # noqa: E402
from senetra_ml.data.repository import SenetraRepository  # noqa: E402
from senetra_ml.evaluation.calibration import cross_fitted_coverage  # noqa: E402
from senetra_ml.evaluation.metrics import aggregate_to_level, skill, summarize_errors  # noqa: E402
from senetra_ml.tracking import experiment_id, setup_mlflow  # noqa: E402

log = logging.getLogger(__name__)
NOISE_FLOOR = "noise_floor_oracle"


def latest_training_run(client, cfg: PipelineConfig):
    experiment = client.get_experiment_by_name(cfg.experiment_name("training"))
    if experiment is None:
        raise LookupError("No training runs found; run `senetra-ml train` first")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="tags.`senetra.role` = 'parent' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"], max_results=1,
    )
    if not runs:
        raise LookupError("No finished training run found; run `senetra-ml train` first")
    return runs[0]


def _lookup(metrics: pd.DataFrame, level: str, model: str, slice_: str, column: str) -> float:
    rows = metrics[(metrics["level"] == level) & (metrics["model"] == model) & (metrics["slice"] == slice_)]
    return float(rows[column].iloc[0]) if len(rows) else float("nan")


def _with_noise_floor(frame: pd.DataFrame) -> pd.DataFrame:
    """Non-deployable reference: leave-one-out mean of other PHCs in the district on the same day."""
    keys = ["fold", "district_id", "origin_date", "horizon", "target_date"]
    grouped = frame.groupby(keys)["actual"]
    total, count = grouped.transform("sum"), grouped.transform("size")
    frame[NOISE_FLOOR] = ((total - frame["actual"]) / (count - 1)).where(count > 1)
    return frame


def metrics_table(frame: pd.DataFrame, models: list[str], aggregation: str) -> pd.DataFrame:
    tables = []
    for level in LEVELS:
        columns = models + ([NOISE_FLOOR] if level == "phc" else [])
        data = aggregate_to_level(frame, level, ["actual", *columns], aggregation)
        warm = data[~data["cold_start"]] if (~data["cold_start"]).any() else data
        parts = [summarize_errors(warm, columns).assign(slice="all")]
        by_regime = summarize_errors(warm.assign(regime=np.where(warm["outbreak"], "outbreak", "normal")),
                                     columns, by=["regime"])
        parts.append(by_regime.rename(columns={"regime": "slice"}))
        if warm["transition"].any():
            parts.append(summarize_errors(warm[warm["transition"]], columns).assign(slice="transition"))
        horizons = summarize_errors(warm, columns, by=["horizon"])
        horizons["slice"] = "h" + horizons.pop("horizon").astype(str)
        parts.append(horizons)
        cold = data[data["cold_start"]]
        if len(cold) and len(cold) < len(data):
            cold_regime = summarize_errors(cold.assign(regime=np.where(cold["outbreak"], "outbreak", "normal")),
                                           columns, by=["regime"])
            cold_regime["slice"] = "cold_start_" + cold_regime.pop("regime")
            parts.append(cold_regime)
        table = pd.concat(parts, ignore_index=True)
        table.insert(0, "level", level)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def _plots(frame: pd.DataFrame, metrics: pd.DataFrame, variant: str, aggregation: str,
           target: str, out_dir: Path) -> list[Path]:
    paths = []
    candidates = [f"{variant}__observed", f"{variant}__oracle", "xgb__observed", "bl_mean7"]
    models = [m for m in candidates if m in frame.columns]
    world = aggregate_to_level(frame[frame["horizon"] == 1], "world", ["actual", *models], aggregation)
    world = world.sort_values("target_date")
    fig, ax = plt.subplots(figsize=(11, 4))
    for row in world[world["outbreak"]].itertuples():
        ax.axvspan(row.target_date, row.target_date + pd.Timedelta(days=1), color="#f4a261", alpha=0.15, lw=0)
    ax.plot(world["target_date"], world["actual"], "k.-", label="actual", lw=1.5)
    for model in models:
        ax.plot(world["target_date"], world[model], lw=1.2, label=model)
    ax.set(title=f"{target}: world level, 1-day-ahead backtest (shaded: outbreak)", ylabel=target)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    paths.append(out_dir / "world_h1_backtest.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    district = metrics[(metrics["level"] == "district") & metrics["slice"].str.match(r"^h\d+$")]
    for model in models:
        rows = district[district["model"] == model].assign(h=lambda d: d["slice"].str[1:].astype(int)).sort_values("h")
        ax.plot(rows["h"], rows["wape"], marker="o", label=model)
    ax.set(title=f"{target}: district WAPE by horizon", xlabel="horizon (days)", ylabel="WAPE")
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths.append(out_dir / "district_wape_by_horizon.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)
    return paths


def evaluate_target(cfg: PipelineConfig, repo: SenetraRepository, client, child, exp_id: str,
                    promote: bool, data_fingerprint: str | None = None) -> dict:
    tags = child.data.tags
    target = tags["senetra.target"]
    target_cfg = cfg.target(target)
    selected, version = tags["senetra.selected_level"], tags.get("senetra.model_version")
    variant = f"fl_{selected}"
    name = cfg.registered_model_name(target)
    gates_cfg = cfg.evaluation.gates

    with mlflow.start_run(experiment_id=exp_id, run_name=target, nested=True) as run:
        mlflow.set_tags({"senetra.stage": "evaluation", "senetra.target": target,
                         "senetra.train_run_id": child.info.run_id, "senetra.model_version": str(version)})
        with tempfile.TemporaryDirectory() as tmp:
            path = mlflow.artifacts.download_artifacts(run_id=child.info.run_id,
                                                       artifact_path="backtest/predictions.parquet", dst_path=tmp)
            predictions = pd.read_parquet(path)
        for column in ("origin_date", "target_date"):
            predictions[column] = pd.to_datetime(predictions[column]).astype("datetime64[ns]")
        actuals = repo.target_actuals(target_cfg, predictions["target_date"].min(), predictions["target_date"].max())
        actuals = actuals.rename(columns={"date": "target_date"})
        actuals["target_date"] = actuals["target_date"].astype("datetime64[ns]")
        frame = predictions.merge(actuals, on=["phc_id", "target_date"], how="inner", validate="many_to_one")
        if len(frame) < len(predictions):
            log.warning("[%s] %d backtest rows have no actual in the database", target, len(predictions) - len(frame))
        frame = _with_noise_floor(frame)

        models = [c for c in predictions.columns if "__" in c or c.startswith("bl_")]
        metrics = metrics_table(frame, models, target_cfg.aggregation)
        coverage = {level: {kind: cross_fitted_coverage(frame, variant, target_cfg.aggregation, level, kind)
                            for kind in ("observed", "scenario")} for level in LEVELS}

        observed = f"{variant}__observed"
        district_wape = _lookup(metrics, "district", observed, "all", "wape")
        phc_mae = _lookup(metrics, "phc", observed, "all", "mae")
        phc_skill = skill(phc_mae, _lookup(metrics, "phc", gates_cfg.baseline, "all", "mae"))
        coverage80 = coverage["district"]["observed"]["coverage80"]

        # Backtest errors are only comparable when both models were backtested on the same data. After new
        # data arrives the folds shift, so the comparison is skipped and left to the online canary check.
        champion_version, champion_wape, other_data = None, None, False
        try:
            champion = client.get_model_version_by_alias(name, cfg.mlflow.champion_alias)
            champion_version = str(champion.version)
            if champion_version != str(version) and "eval.district_wape" in champion.tags:
                if data_fingerprint and champion.tags.get("eval.data_fingerprint") == data_fingerprint:
                    champion_wape = float(champion.tags["eval.district_wape"])
                else:
                    other_data = True
        except MlflowException:
            pass
        if champion_wape is not None:
            regression_threshold = (f"<= {champion_wape * (1 + gates_cfg.max_regression_vs_champion):.4f} "
                                    f"(champion v{champion_version})")
        elif other_data:
            regression_threshold = (f"champion v{champion_version} was backtested on different data; "
                                    "compare online (canary)")
        else:
            regression_threshold = "no other champion"

        low, high = gates_cfg.district_coverage80
        gates = [
            {"name": "district_wape", "value": district_wape, "threshold": f"<= {gates_cfg.max_district_wape}",
             "passed": bool(district_wape <= gates_cfg.max_district_wape)},
            {"name": f"phc_skill_vs_{gates_cfg.baseline}", "value": phc_skill,
             "threshold": f">= {gates_cfg.min_phc_skill_vs_baseline}",
             "passed": bool(phc_skill >= gates_cfg.min_phc_skill_vs_baseline)},
            {"name": "district_coverage80", "value": coverage80, "threshold": f"in [{low}, {high}]",
             "passed": None if not np.isfinite(coverage80) else bool(low <= coverage80 <= high)},
            {"name": "no_regression_vs_champion", "value": district_wape, "threshold": regression_threshold,
             "passed": None if champion_wape is None
             else bool(district_wape <= champion_wape * (1 + gates_cfg.max_regression_vs_champion))},
        ]
        passed = all(g["passed"] is not False for g in gates)

        headline = {}
        for level in LEVELS:
            for model in [observed, f"{variant}__oracle", "xgb__observed", "local__observed", gates_cfg.baseline,
                          "bl_seasonal7", NOISE_FLOOR]:
                for slice_ in ("all", "outbreak", "cold_start_outbreak"):
                    for column in ("wape", "mae"):
                        value = _lookup(metrics, level, model, slice_, column)
                        if np.isfinite(value):
                            headline[f"{level}.{model}.{slice_}.{column}"] = value
            for kind in ("observed", "scenario"):
                value = coverage[level][kind]["coverage80"]
                if np.isfinite(value):
                    headline[f"{level}.coverage80.{kind}"] = value
        mlflow.log_metrics(headline)
        mlflow.log_metrics({"gates_passed": float(passed), "phc_skill": phc_skill, "district_wape": district_wape})

        decision = "unregistered"
        if version and version != "unregistered":
            for key, value in {"eval.district_wape": district_wape, "eval.phc_mae": phc_mae,
                               "eval.phc_skill": phc_skill, "eval.district_coverage80": coverage80,
                               "eval.gates_passed": passed, "eval.run_id": run.info.run_id,
                               "eval.data_fingerprint": data_fingerprint or "unknown"}.items():
                client.set_model_version_tag(name, version, key, str(value))
            if not promote:
                decision = "not promoted (--no-promote)"
            elif passed:
                client.set_registered_model_alias(name, cfg.mlflow.champion_alias, version)
                decision = f"promoted to @{cfg.mlflow.champion_alias}"
            else:
                client.set_registered_model_alias(name, cfg.mlflow.challenger_alias, version)
                decision = f"kept as @{cfg.mlflow.challenger_alias} (gates failed)"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            metrics.to_csv(tmp_dir / "metrics.csv", index=False)
            (tmp_dir / "gates.json").write_text(json.dumps({"passed": passed, "gates": gates, "decision": decision},
                                                           indent=2), encoding="utf-8")
            (tmp_dir / "coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
            _plots(frame, metrics, variant, target_cfg.aggregation, target, tmp_dir)
            mlflow.log_artifacts(str(tmp_dir), artifact_path="evaluation")

        log.info("[%s] district WAPE=%.4f PHC skill=%.3f coverage80=%.3f -> %s",
                 target, district_wape, phc_skill, coverage80, decision)
        return {
            "target": target, "model_version": version, "selected_level": selected, "decision": decision,
            "gates_passed": passed, "gates": gates,
            "district_wape_observed": district_wape,
            "district_wape_scenario": _lookup(metrics, "district", f"{variant}__oracle", "all", "wape"),
            "district_wape_xgb": _lookup(metrics, "district", "xgb__observed", "all", "wape"),
            "district_wape_local": _lookup(metrics, "district", "local__observed", "all", "wape"),
            "district_wape_baseline": _lookup(metrics, "district", gates_cfg.baseline, "all", "wape"),
            "district_wape_outbreak_scenario": _lookup(metrics, "district", f"{variant}__oracle", "outbreak", "wape"),
            "district_wape_cold_start_outbreak": _lookup(metrics, "district", f"{variant}__oracle",
                                                         "cold_start_outbreak", "wape"),
            "world_wape_observed": _lookup(metrics, "world", observed, "all", "wape"),
            "phc_mae": phc_mae, "phc_skill_vs_baseline": phc_skill,
            "phc_noise_floor_mae": _lookup(metrics, "phc", NOISE_FLOOR, "all", "mae"),
            "district_coverage80": coverage80,
            "evaluation_run_id": run.info.run_id,
        }


def _fmt(value, pct: bool = False) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    return f"{value:.1%}" if pct else f"{value:.3f}"


def write_summary(results: list[dict], train_run_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SENETRA model evaluation",
        "",
        f"Training run `{train_run_id}`, evaluated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}. "
        "Backtest folds whose training window had no outbreak (cold start) are excluded from the headline "
        "numbers and reported in their own column.",
        "",
        "District WAPE compares models on the same rows. *observed* = current outbreak state assumed to persist "
        "(operational default); *scenario* = correct outbreak scenario supplied (what-if / simulator).",
        "",
        "| target | level | district WAPE observed | scenario | outbreak (scenario) | cold-start outbreak | "
        "XGBoost (central) | local-only | 7-day mean | PHC MAE | PHC noise floor | PHC skill | coverage80 | decision |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['target']} | {r['selected_level']} | {_fmt(r['district_wape_observed'], True)} | "
            f"{_fmt(r['district_wape_scenario'], True)} | {_fmt(r['district_wape_outbreak_scenario'], True)} | "
            f"{_fmt(r['district_wape_cold_start_outbreak'], True)} | {_fmt(r['district_wape_xgb'], True)} | "
            f"{_fmt(r['district_wape_local'], True)} | {_fmt(r['district_wape_baseline'], True)} | "
            f"{_fmt(r['phc_mae'])} | {_fmt(r['phc_noise_floor_mae'])} | {_fmt(r['phc_skill_vs_baseline'])} | "
            f"{_fmt(r['district_coverage80'], True)} | {r['decision']} |"
        )
    lines += ["", "## Gates", ""]
    for r in results:
        gate_text = "; ".join(
            f"{g['name']} {_fmt(g['value'])} {g['threshold']} "
            f"{'PASS' if g['passed'] else 'skip' if g['passed'] is None else 'FAIL'}" for g in r["gates"])
        lines.append(f"- **{r['target']}** v{r['model_version']}: {gate_text}")
    path = out_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return path


def run_evaluation(cfg: PipelineConfig, train_run_id: str | None = None, targets: list[str] | None = None,
                   promote: bool = True) -> dict:
    client = setup_mlflow(cfg)
    parent = client.get_run(train_run_id) if train_run_id else latest_training_run(client, cfg)
    children = client.search_runs(
        [parent.info.experiment_id],
        filter_string=f"tags.`mlflow.parentRunId` = '{parent.info.run_id}' and tags.`senetra.role` = 'target'",
    )
    if targets:
        children = [c for c in children if c.data.tags.get("senetra.target") in set(targets)]
    if not children:
        raise LookupError(f"Training run {parent.info.run_id} has no target runs to evaluate")

    exp_id = experiment_id(cfg, "evaluation")
    results = []
    with SenetraRepository(cfg.data.db_path) as repo:
        with mlflow.start_run(experiment_id=exp_id, run_name=f"evaluate-{parent.info.run_id[:8]}") as run:
            mlflow.set_tags({"senetra.stage": "evaluation", "senetra.role": "parent",
                             "senetra.train_run_id": parent.info.run_id})
            order = {name: i for i, name in enumerate(cfg.targets)}
            for child in sorted(children, key=lambda c: order.get(c.data.tags.get("senetra.target"), 99)):
                results.append(evaluate_target(cfg, repo, client, child, exp_id, promote,
                                               parent.data.tags.get("data.fingerprint")))
            summary_path = write_summary(results, parent.info.run_id, cfg.reports_dir / "evaluation")
            mlflow.log_artifact(str(summary_path), artifact_path="evaluation")
    return {"run_id": run.info.run_id, "train_run_id": parent.info.run_id, "results": results,
            "summary_path": str(summary_path)}
