"""Data contract checks run before training and during monitoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import pandas as pd

from senetra_ml.config import PipelineConfig
from senetra_ml.data.repository import SenetraRepository

Status = Literal["pass", "info", "warn", "error"]

REQUIRED_TABLES = {
    "countries", "states", "districts", "phcs", "medicines",
    "daily_metrics", "medicine_consumption", "inventory", "simulation_events",
}


@dataclass
class Check:
    name: str
    status: Status
    detail: str
    value: float | None = None


@dataclass
class ValidationReport:
    checks: list[Check]

    @property
    def errors(self) -> list[Check]:
        return [c for c in self.checks if c.status == "error"]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status == "warn"]

    def raise_for_errors(self) -> None:
        if self.errors:
            details = "; ".join(f"{c.name}: {c.detail}" for c in self.errors)
            raise ValueError(f"Data validation failed: {details}")

    def to_dict(self) -> dict:
        return {"checks": [asdict(c) for c in self.checks],
                "errors": len(self.errors), "warnings": len(self.warnings)}


def _share_status(share: float, warn_above: float = 0.0, error_above: float = 0.01) -> Status:
    if share > error_above:
        return "error"
    return "warn" if share > warn_above else "pass"


def validate_database(repo: SenetraRepository, cfg: PipelineConfig,
                      today: pd.Timestamp | None = None) -> ValidationReport:
    checks: list[Check] = []
    missing = REQUIRED_TABLES - repo.table_names()
    if missing:
        checks.append(Check("required_tables", "error", f"missing tables: {sorted(missing)}"))
        return ValidationReport(checks)
    checks.append(Check("required_tables", "pass", "all required tables present"))

    (metric_rows, consumption_rows, phcs), = repo.rows(
        "SELECT (SELECT COUNT(*) FROM daily_metrics), (SELECT COUNT(*) FROM medicine_consumption),"
        " (SELECT COUNT(*) FROM phcs)"
    )
    if not metric_rows or not phcs:
        checks.append(Check("volume", "error", "daily_metrics or phcs is empty"))
        return ValidationReport(checks)
    checks.append(Check("volume", "pass",
                        f"{metric_rows} metric rows, {consumption_rows} consumption rows, {phcs} PHCs",
                        float(metric_rows)))

    lo, hi = repo.date_bounds()
    days = (hi - lo).days + 1
    (reporting_phcs,), = repo.rows("SELECT COUNT(DISTINCT phc_id) FROM daily_metrics")
    completeness = metric_rows / max(reporting_phcs * days, 1)
    checks.append(Check("completeness", "pass" if completeness >= 0.95 else "warn",
                        f"{completeness:.1%} of PHC-days reported over {days} days ({lo.date()}..{hi.date()})",
                        completeness))
    if reporting_phcs < phcs:
        checks.append(Check("silent_phcs", "warn", f"{phcs - reporting_phcs} PHCs have no daily metrics",
                            float(phcs - reporting_phcs)))

    today = today or pd.Timestamp.today().normalize()
    staleness = (today - hi).days
    checks.append(Check("freshness", "pass" if staleness <= cfg.monitoring.max_staleness_days else "warn",
                        f"latest data {hi.date()} is {staleness} day(s) old", float(staleness)))

    (bad_footfall, bad_bed, bad_staff, null_metrics), = repo.rows(
        """SELECT SUM(patient_footfall < 0), SUM(bed_occupancy < 0 OR bed_occupancy > 100),
                  SUM(staff_availability < 0 OR staff_availability > 100),
                  SUM(patient_footfall IS NULL OR bed_occupancy IS NULL OR staff_availability IS NULL)
           FROM daily_metrics"""
    )
    for name, count in [("footfall_range", bad_footfall), ("bed_occupancy_range", bad_bed),
                        ("staff_availability_range", bad_staff), ("metric_nulls", null_metrics)]:
        share = (count or 0) / metric_rows
        checks.append(Check(name, _share_status(share), f"{count or 0} violating rows ({share:.3%})", share))

    (bad_consumption,), = repo.rows(
        "SELECT SUM(consumption < 0 OR consumption IS NULL) FROM medicine_consumption")
    share = (bad_consumption or 0) / max(consumption_rows, 1)
    checks.append(Check("consumption_range", _share_status(share),
                        f"{bad_consumption or 0} negative/null consumption rows", share))

    medicines = set(repo.medicines()["name"])
    for name, target in cfg.targets.items():
        if target.is_medicine and target.medicine not in medicines:
            checks.append(Check(f"target_{name}", "error", f"medicine {target.medicine!r} not in medicines table"))

    (label_mismatch,), = repo.rows(
        "SELECT SUM((outbreak_flag = 1) <> (outbreak_type IS NOT NULL)) FROM daily_metrics")
    share = (label_mismatch or 0) / metric_rows
    checks.append(Check("outbreak_label_consistency", "warn" if share > 0.001 else "pass",
                        f"{share:.2%} of rows disagree between outbreak_flag and outbreak_type; "
                        "features use outbreak_flag only", share))

    (chained, compared), = repo.rows(
        """SELECT SUM(opening_stock = prev_close), COUNT(prev_close) FROM (
             SELECT opening_stock, LAG(closing_stock) OVER (
               PARTITION BY phc_id, medicine_id ORDER BY date) AS prev_close
             FROM medicine_consumption)"""
    )
    coherence = (chained or 0) / max(compared or 0, 1)
    checks.append(Check("stock_ledger_coherence", "pass" if coherence >= 0.95 else "warn",
                        f"{coherence:.1%} of opening stocks equal the previous closing stock; "
                        "stock-out risk uses the inventory snapshot, not the ledger", coherence))

    (no_coords,), = repo.rows("SELECT SUM(latitude IS NULL OR longitude IS NULL) FROM districts")
    if no_coords:
        checks.append(Check("district_coordinates", "warn",
                            f"{no_coords} districts lack coordinates; distance features are unavailable",
                            float(no_coords)))

    coverage = repo.query(
        """SELECT c.name, COUNT(p.id) AS phcs FROM countries c
           LEFT JOIN states s ON s.country_id = c.id LEFT JOIN districts d ON d.state_id = s.id
           LEFT JOIN phcs p ON p.district_id = d.id GROUP BY c.id"""
    )
    empty = coverage.loc[coverage["phcs"] == 0, "name"].tolist()
    checks.append(Check("federation_coverage", "info" if empty else "pass",
                        f"{(coverage['phcs'] > 0).sum()} of {len(coverage)} countries have PHC data"
                        + (f"; no data for {', '.join(empty)}" if empty else ""),
                        float((coverage["phcs"] > 0).sum())))

    (events_active, events_outside), = repo.rows(
        "SELECT SUM(active = 1), SUM(started_at < ?) FROM simulation_events", (lo.strftime("%Y-%m-%d"),))
    checks.append(Check("simulation_events", "info",
                        f"{events_active or 0} active events, {events_outside or 0} started before the data "
                        "window; events are scenario inputs, not training labels"))
    return ValidationReport(checks)
