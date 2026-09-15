"""Pipeline configuration: one YAML file validated with pydantic, with environment overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

METRIC_COLUMNS = ("patient_footfall", "bed_occupancy", "staff_availability")
LEVELS = ("phc", "district", "country", "world")


def ml_home() -> Path:
    """Directory holding configs/ and reports/ (the ml/ folder in the repo, /app in Docker)."""
    env = os.environ.get("SENETRA_ML_HOME")
    if env:
        return Path(env).resolve()
    source_home = Path(__file__).resolve().parents[2]
    if (source_home / "configs" / "pipeline.yaml").is_file():
        return source_home
    return Path.cwd().resolve()


class DataConfig(BaseModel):
    db_path: Path
    min_history_days: int = Field(7, ge=7)
    outbreak_prevalence_threshold: float = Field(0.02, gt=0, lt=1)


class ForecastConfig(BaseModel):
    horizon_days: int = Field(7, ge=1, le=14)
    rolling_windows: list[int] = Field(default_factory=lambda: [3, 7, 14, 28])

    @model_validator(mode="after")
    def _windows(self) -> ForecastConfig:
        if 7 not in self.rolling_windows:
            raise ValueError("forecast.rolling_windows must include 7 (used by dispersion and trend features)")
        self.rolling_windows = sorted(set(self.rolling_windows))
        return self

    @property
    def long_window(self) -> int:
        return max(self.rolling_windows)


class ScenarioConfig(BaseModel):
    severity_reference: float = Field(8, gt=0)
    max_intensity: float = Field(1.5, ge=1)


class TargetConfig(BaseModel):
    source: Literal["daily_metrics", "medicine_consumption"]
    column: str | None = None
    medicine: str | None = None
    family: Literal["poisson", "gaussian"]
    aggregation: Literal["sum", "mean"]
    bounds: tuple[float, float] | None = None
    unit: str | None = None

    @model_validator(mode="after")
    def _source_fields(self) -> TargetConfig:
        if self.source == "daily_metrics" and self.column not in METRIC_COLUMNS:
            raise ValueError(f"daily_metrics targets need column in {METRIC_COLUMNS}")
        if self.source == "medicine_consumption" and not self.medicine:
            raise ValueError("medicine_consumption targets need a medicine name")
        return self

    @property
    def series(self) -> str:
        """Panel series key holding this target's history."""
        return self.column if self.source == "daily_metrics" else f"medicine:{self.medicine}"

    @property
    def is_medicine(self) -> bool:
        return self.source == "medicine_consumption"


class FederatedConfig(BaseModel):
    algorithm: Literal["newton", "fedavg"] = "newton"
    rounds: int = Field(10, ge=1)
    l2: float = Field(1e-3, ge=0)
    # Hierarchical refinement: each level is shrunk toward its parent with a prior worth this many rows.
    level_newton_steps: int = Field(3, ge=0)
    country_prior_rows: float = Field(2000, ge=0)
    district_prior_rows: float = Field(2000, ge=0)
    phc_prior_rows: float = Field(500, ge=0)
    personalization_steps: int = Field(2, ge=0)
    local_only_steps: int = Field(8, ge=1)
    # FedAvg / FedProx only.
    district_rounds: int = Field(2, ge=1)
    local_newton_steps: int = Field(1, ge=1)
    proximal_mu: float = Field(0.01, ge=0)
    client_fraction: float = Field(1.0, gt=0, le=1)
    update_clip_norm: float | None = Field(None, gt=0)
    dp_noise_multiplier: float = Field(0.0, ge=0)
    seed: int = 42


class XGBoostConfig(BaseModel):
    enabled: bool = True
    num_boost_round: int = 400
    early_stopping_rounds: int = 30
    params: dict[str, Any] = Field(default_factory=dict)


class BenchmarksConfig(BaseModel):
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)


class BacktestConfig(BaseModel):
    n_folds: int = Field(4, ge=1)
    step_days: int = Field(14, ge=1)
    test_origin_days: int = Field(7, ge=1)


class GateConfig(BaseModel):
    max_district_wape: float = 0.15
    min_phc_skill_vs_baseline: float = 0.0
    baseline: str = "bl_mean7"
    district_coverage80: tuple[float, float] = (0.7, 0.95)
    max_regression_vs_champion: float = 0.05


class EvaluationConfig(BaseModel):
    gates: GateConfig = Field(default_factory=GateConfig)


class RiskConfig(BaseModel):
    critical_days: float = 3
    high_days: float = 7
    watch_days: float = 14


class MonitoringConfig(BaseModel):
    window_days: int = Field(14, ge=1)
    psi_warn: float = 0.1
    psi_alert: float = 0.25
    degradation_ratio_alert: float = 1.3
    max_staleness_days: int = 2
    min_reporting_ratio: float = 0.95


class MlflowConfig(BaseModel):
    tracking_uri: str | None = None
    experiment_prefix: str = "senetra"
    registered_model_prefix: str = "senetra-forecast"
    champion_alias: str = "champion"
    challenger_alias: str = "challenger"


class PipelineConfig(BaseModel):
    home: Path
    data: DataConfig
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    scenarios: ScenarioConfig = Field(default_factory=ScenarioConfig)
    targets: dict[str, TargetConfig]
    federated: FederatedConfig = Field(default_factory=FederatedConfig)
    benchmarks: BenchmarksConfig = Field(default_factory=BenchmarksConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    mlflow: MlflowConfig = Field(default_factory=MlflowConfig)

    @property
    def reports_dir(self) -> Path:
        return self.home / "reports"

    def target(self, name: str) -> TargetConfig:
        try:
            return self.targets[name]
        except KeyError:
            raise KeyError(f"Unknown target {name!r}; configured: {sorted(self.targets)}") from None

    def resolve_targets(self, names: list[str] | None) -> list[str]:
        if not names:
            return list(self.targets)
        for name in names:
            self.target(name)
        return list(names)

    def tracking_uri(self) -> str:
        uri = os.environ.get("MLFLOW_TRACKING_URI") or self.mlflow.tracking_uri
        if uri:
            return uri
        return f"sqlite:///{(self.home / 'mlruns' / 'mlflow.db').as_posix()}"

    def artifact_root(self) -> str | None:
        """Artifact location for new experiments; None lets a remote tracking server decide."""
        if self.tracking_uri().startswith(("http://", "https://", "databricks")):
            return None
        return (self.home / "mlruns" / "artifacts").as_uri()

    def experiment_name(self, stage: str) -> str:
        return f"{self.mlflow.experiment_prefix}-{stage}"

    def registered_model_name(self, target: str) -> str:
        return f"{self.mlflow.registered_model_prefix}-{target.replace('_', '-')}"


def load_config(path: str | Path | None = None) -> PipelineConfig:
    home = ml_home()
    config_path = Path(path or os.environ.get("SENETRA_CONFIG") or home / "configs" / "pipeline.yaml")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw["home"] = home

    data = raw.setdefault("data", {})
    db_path = Path(os.environ.get("SENETRA_DB_PATH") or data.get("db_path") or "../senetra.db")
    data["db_path"] = db_path if db_path.is_absolute() else (home / db_path).resolve()
    return PipelineConfig.model_validate(raw)
