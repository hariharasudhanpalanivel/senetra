"""Canary rollout decisions: hold, advance, promote or roll back.

Checks, in order:
1. Safety (once the canary has served `canary_min_requests`): error rate and p95 latency versus stable.
2. Accuracy on fresh actuals (days after both models' training data): district WAPE versus stable.
3. Evidence: minimum time and requests per stage. Without fresh actuals the rollout never exceeds
   `canary_max_percent_without_actuals` and is not promoted (unless that limit is 100).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime

from mlops.deployment import TargetDeployment
from mlops.settings import MLOpsSettings


@dataclass
class VariantStats:
    requests: int = 0
    errors: int = 0
    p95_latency_ms: float | None = None

    @property
    def error_rate(self) -> float:
        return self.errors / self.requests if self.requests else 0.0

    @classmethod
    def from_dict(cls, data: dict) -> VariantStats:
        return cls(requests=data["requests"], errors=data["errors"], p95_latency_ms=data["p95_latency_ms"])


@dataclass
class CanaryDecision:
    action: str  # hold | advance | promote | rollback
    reason: str
    checks: dict = field(default_factory=dict)


def decide(deployment: TargetDeployment, canary: VariantStats, stable: VariantStats, accuracy: dict,
           settings: MLOpsSettings, now: datetime) -> CanaryDecision:
    checks = {
        "percent": deployment.canary_percent,
        "canary": {**asdict(canary), "error_rate": canary.error_rate},
        "stable": {**asdict(stable), "error_rate": stable.error_rate},
        "accuracy": accuracy,
    }
    enough_stable = stable.requests >= settings.canary_min_requests

    if canary.requests >= settings.canary_min_requests:
        stable_rate = stable.error_rate if enough_stable else 0.0
        if canary.error_rate > stable_rate + settings.canary_max_error_rate_delta:
            return CanaryDecision("rollback", f"canary error rate {canary.error_rate:.1%} exceeds stable "
                                              f"{stable_rate:.1%} + {settings.canary_max_error_rate_delta:.1%}", checks)
        if (enough_stable and canary.p95_latency_ms and stable.p95_latency_ms
                and canary.p95_latency_ms > max(settings.canary_latency_floor_ms,
                                                settings.canary_max_latency_ratio * stable.p95_latency_ms)):
            return CanaryDecision("rollback", f"canary p95 latency {canary.p95_latency_ms:.0f} ms exceeds "
                                              f"{settings.canary_max_latency_ratio}x stable "
                                              f"({stable.p95_latency_ms:.0f} ms)", checks)

    compared = accuracy.get("status") == "compared"
    if compared:
        stable_wape, canary_wape = accuracy["stable_district_wape"], accuracy["canary_district_wape"]
        if canary_wape > stable_wape * (1 + settings.canary_max_wape_regression):
            return CanaryDecision("rollback", f"canary district WAPE {canary_wape:.2%} is worse than stable "
                                              f"{stable_wape:.2%} by more than "
                                              f"{settings.canary_max_wape_regression:.0%}", checks)

    stage_started = datetime.fromisoformat(deployment.canary_stage_started_at) if deployment.canary_stage_started_at else now
    minutes = (now - stage_started).total_seconds() / 60
    if minutes < settings.canary_min_stage_minutes:
        return CanaryDecision("hold", f"stage running {minutes:.0f} of {settings.canary_min_stage_minutes:.0f} minutes", checks)
    if canary.requests < settings.canary_min_requests:
        return CanaryDecision("hold", f"canary served {canary.requests} of {settings.canary_min_requests} "
                                      "required requests", checks)

    next_index = deployment.canary_stage_index + 1
    if next_index >= len(settings.canary_stages):
        if compared or settings.canary_max_percent_without_actuals >= 100:
            return CanaryDecision("promote", "all stages passed" + (" including fresh-data accuracy" if compared else ""),
                                  checks)
        return CanaryDecision("hold", "waiting for fresh actuals before promotion", checks)
    next_percent = settings.canary_stages[next_index]
    if not compared and next_percent > settings.canary_max_percent_without_actuals:
        return CanaryDecision("hold", f"waiting for fresh actuals before exceeding "
                                      f"{settings.canary_max_percent_without_actuals}% traffic", checks)
    return CanaryDecision("advance", f"healthy at {deployment.canary_percent}%; moving to {next_percent}%", checks)
