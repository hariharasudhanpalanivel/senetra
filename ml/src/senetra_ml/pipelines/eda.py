"""EDA pipeline: analysis on the live database, report to reports/eda and MLflow."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from senetra_ml.config import PipelineConfig
from senetra_ml.data.panel import load_panel
from senetra_ml.data.repository import SenetraRepository
from senetra_ml.eda.analysis import run_analysis
from senetra_ml.eda.report import write_figures, write_markdown
from senetra_ml.tracking import experiment_id, lineage_tags, setup_mlflow

log = logging.getLogger(__name__)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and value != value:
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def run_eda(cfg: PipelineConfig, output_dir: Path | None = None, track: bool = True) -> dict:
    out_dir = Path(output_dir or cfg.reports_dir / "eda")
    with SenetraRepository(cfg.data.db_path) as repo:
        log.info("Loading panel for EDA from %s", cfg.data.db_path)
        panel = load_panel(repo)
        summary, plot_data = run_analysis(repo, panel, cfg)
        summary = _json_safe(summary)
        figures = write_figures(summary, plot_data, out_dir)
        (out_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        report = write_markdown(summary, figures, out_dir / "eda_report.md")
        log.info("EDA report written to %s", report)

        if track:
            setup_mlflow(cfg)
            with mlflow.start_run(experiment_id=experiment_id(cfg, "eda"),
                                  run_name=f"eda-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}") as run:
                mlflow.set_tags({**lineage_tags(cfg, repo), "senetra.stage": "eda"})
                mlflow.log_artifacts(str(out_dir), artifact_path="eda")
                mlflow.log_metrics({
                    "phcs": summary["geography"]["phcs"],
                    "districts_with_phcs": summary["geography"]["districts_with_phcs"],
                    "outbreak_days": summary["regimes"]["outbreak_days"],
                    "ledger_coherence": summary["stock"]["ledger_coherence"],
                })
                summary["mlflow_run_id"] = run.info.run_id
    return summary
