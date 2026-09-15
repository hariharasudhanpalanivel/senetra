"""Backend MLOps settings, read from environment variables (see backend/README.md)."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def _strings(value: str) -> tuple[str, ...] | None:
    items = tuple(part.strip() for part in value.split(",") if part.strip())
    return items or None


@dataclass(frozen=True)
class MLOpsSettings:
    # Shared state (jobs, data watermarks, prediction log, deployment events, scheduler lease).
    state_db: Path = BACKEND_ROOT / "instance" / "mlops.db"
    # Scheduler: one leader across all backend processes runs the periodic tasks.
    scheduler_enabled: bool = True
    scheduler_poll_seconds: float = 15.0
    data_check_interval_seconds: int = 300
    canary_check_interval_seconds: int = 600
    monitor_interval_seconds: int = 86400
    # Continuous training.
    retrain_targets: tuple[str, ...] | None = None
    retrain_min_new_days: int = 1
    retrain_cooldown_minutes: float = 60.0
    job_runner: str = "process"  # "process" (separate OS process) or "inline" (tests, debugging)
    job_stale_minutes: float = 10.0
    # Canary deployment.
    canary_stages: tuple[int, ...] = (10, 25, 50, 100)
    canary_min_stage_minutes: float = 60.0
    canary_min_requests: int = 20
    canary_max_error_rate_delta: float = 0.02
    canary_max_latency_ratio: float = 1.5
    canary_latency_floor_ms: float = 250.0
    canary_max_wape_regression: float = 0.05
    canary_max_percent_without_actuals: int = 25
    # Protects endpoints that change state. Unset = open (local development only).
    admin_token: str | None = None

    def __post_init__(self) -> None:
        stages = self.canary_stages
        if not stages or stages[-1] != 100 or any(not 0 < s <= 100 for s in stages) or list(stages) != sorted(set(stages)):
            raise ValueError("canary_stages must be strictly increasing percentages ending at 100")
        if self.job_runner not in {"process", "inline"}:
            raise ValueError("job_runner must be 'process' or 'inline'")

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> MLOpsSettings:
        env = os.environ if env is None else env
        parsers = {
            "state_db": Path, "scheduler_enabled": _bool, "scheduler_poll_seconds": float,
            "data_check_interval_seconds": int, "canary_check_interval_seconds": int,
            "monitor_interval_seconds": int, "retrain_targets": _strings, "retrain_min_new_days": int,
            "retrain_cooldown_minutes": float, "job_runner": str, "job_stale_minutes": float,
            "canary_stages": _ints, "canary_min_stage_minutes": float, "canary_min_requests": int,
            "canary_max_error_rate_delta": float, "canary_max_latency_ratio": float,
            "canary_latency_floor_ms": float, "canary_max_wape_regression": float,
            "canary_max_percent_without_actuals": int, "admin_token": lambda v: v or None,
        }
        values = {}
        for field in fields(cls):
            raw = env.get(f"ML_{field.name.upper()}")
            if raw is not None:
                values[field.name] = parsers[field.name](raw)
        return cls(**values)
