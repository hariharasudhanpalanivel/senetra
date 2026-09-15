"""Exploratory data analysis computed in place on the database.

Findings are derived from the data, not hard-coded: regimes are detected from surveillance and
demand signals, and modelling implications are generated from measured thresholds.
Only aggregate statistics leave this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from senetra_ml.config import PipelineConfig
from senetra_ml.data.panel import Panel
from senetra_ml.data.repository import SenetraRepository

MAX_LAG = 14


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _describe(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0}
    mean = float(values.mean())
    return {
        "n": int(values.size), "mean": mean, "std": float(values.std()),
        "cv": float(values.std() / mean) if mean else float("nan"),
        "min": float(values.min()), "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def _segments(mask: np.ndarray, dates: pd.DatetimeIndex) -> list[dict]:
    segments, start = [], None
    for i, flag in enumerate(list(mask) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            segments.append({"start": dates[start].strftime("%Y-%m-%d"),
                             "end": dates[i - 1].strftime("%Y-%m-%d"), "days": i - start})
            start = None
    return segments


def _district_daily_mean(panel: Panel, x: np.ndarray) -> np.ndarray:
    n_districts = len(panel.district_ids)
    sums = np.zeros((n_districts, panel.n_dates))
    counts = np.zeros((n_districts, panel.n_dates))
    np.add.at(sums, panel.district_index, np.nan_to_num(x))
    np.add.at(counts, panel.district_index, np.isfinite(x).astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums / counts


def data_inventory(repo: SenetraRepository) -> dict:
    tables = ["countries", "states", "districts", "phcs", "medicines", "daily_metrics",
              "medicine_consumption", "inventory", "predictions", "alerts", "simulation_events"]
    counts = {t: int(repo.rows(f"SELECT COUNT(*) FROM {t}")[0][0]) for t in tables if t in repo.table_names()}
    lo, hi = repo.date_bounds()
    return {"row_counts": counts, "start": lo.strftime("%Y-%m-%d"), "end": hi.strftime("%Y-%m-%d"),
            "days": int((hi - lo).days + 1)}


def geography(repo: SenetraRepository, panel: Panel) -> dict:
    coverage = repo.query(
        """SELECT c.name AS country, COUNT(DISTINCT s.id) AS states, COUNT(DISTINCT d.id) AS districts,
                  COUNT(p.id) AS phcs
           FROM countries c LEFT JOIN states s ON s.country_id = c.id
           LEFT JOIN districts d ON d.state_id = s.id LEFT JOIN phcs p ON p.district_id = d.id
           GROUP BY c.id ORDER BY phcs DESC"""
    )
    per_district = panel.hierarchy.groupby("district_id").size()
    (no_coords, districts), = repo.rows("SELECT SUM(latitude IS NULL OR longitude IS NULL), COUNT(*) FROM districts")
    (phc_no_coords,), = repo.rows("SELECT SUM(latitude IS NULL OR longitude IS NULL) FROM phcs")
    return {
        "countries": coverage.to_dict("records"),
        "countries_with_phcs": int((coverage["phcs"] > 0).sum()),
        "districts_with_phcs": int(per_district.size),
        "states_with_phcs": int(panel.hierarchy["state_id"].nunique()),
        "phcs": int(panel.n_phcs),
        "phcs_per_district": {"min": int(per_district.min()), "median": float(per_district.median()),
                              "max": int(per_district.max())},
        "districts_missing_coordinates": int(no_coords or 0), "districts_total": int(districts),
        "phcs_missing_coordinates": int(phc_no_coords or 0),
    }


def static_attributes(repo: SenetraRepository) -> dict:
    columns = [row[1] for row in repo.rows("PRAGMA table_info(phcs)")]
    numeric = [c for c in columns if c not in ("id", "district_id", "name", "latitude", "longitude")]
    out = {}
    for column in numeric:
        (distinct, lo, hi), = repo.rows(f"SELECT COUNT(DISTINCT {column}), MIN({column}), MAX({column}) FROM phcs")
        out[column] = {"distinct": int(distinct), "min": lo, "max": hi}
    return out


def regimes(panel: Panel, cfg: PipelineConfig) -> tuple[dict, np.ndarray, np.ndarray]:
    threshold = cfg.data.outbreak_prevalence_threshold
    phc_outbreak = np.nan_to_num(panel.client_prevalence()) > threshold
    surveillance_days = phc_outbreak.mean(axis=0) > 0.5

    footfall = np.nanmean(panel.series("patient_footfall"), axis=0)
    median = np.nanmedian(footfall)
    mad = 1.4826 * np.nanmedian(np.abs(footfall - median)) or 1e-9
    demand_days = footfall > median + 5 * mad
    district_prev = panel.district_prevalence
    active = district_prev[np.isfinite(district_prev) & (district_prev > threshold)]
    quiet = district_prev[np.isfinite(district_prev) & (district_prev <= threshold)]
    summary = {
        "threshold": threshold,
        "outbreak_days": int(surveillance_days.sum()),
        "normal_days": int((~surveillance_days).sum()),
        "surveillance_segments": _segments(surveillance_days, panel.dates),
        "demand_shift_segments": _segments(demand_days, panel.dates),
        "surveillance_demand_agreement": float((surveillance_days == demand_days).mean()),
        "district_prevalence_outbreak": _describe(active),
        "district_prevalence_normal_max": float(quiet.max()) if quiet.size else float("nan"),
    }
    return summary, phc_outbreak, surveillance_days


def target_profiles(panel: Panel, cfg: PipelineConfig, phc_outbreak: np.ndarray) -> dict:
    out = {}
    for name, target in cfg.targets.items():
        x = panel.series(target.series)
        normal, outbreak = _describe(x[~phc_outbreak]), _describe(x[phc_outbreak])
        uplift = outbreak.get("mean", np.nan) / normal["mean"] if normal.get("mean") else float("nan")
        out[name] = {"normal": normal, "outbreak": outbreak, "outbreak_uplift": float(uplift)}
    return out


def _regime_deviation(x: np.ndarray, phc_outbreak: np.ndarray) -> np.ndarray:
    normal_mean = np.nanmean(np.where(phc_outbreak, np.nan, x))
    outbreak_mean = np.nanmean(np.where(phc_outbreak, x, np.nan)) if phc_outbreak.any() else normal_mean
    return x - np.where(phc_outbreak, outbreak_mean, normal_mean)


def temporal_structure(panel: Panel, cfg: PipelineConfig, phc_outbreak: np.ndarray) -> dict:
    weekday = panel.dates.dayofweek.to_numpy()
    out = {}
    for name, target in cfg.targets.items():
        x = panel.series(target.series)
        deviation = _regime_deviation(x, phc_outbreak)
        acf = []
        for k in range(1, MAX_LAG + 1):
            same = phc_outbreak[:, k:] == phc_outbreak[:, :-k]
            acf.append(_corr(deviation[:, k:][same], deviation[:, :-k][same]))
        normal_values = np.where(phc_outbreak, np.nan, x)
        dow_means = [float(np.nanmean(normal_values[:, weekday == d])) for d in range(7)]
        overall = float(np.nanmean(normal_values))
        out[name] = {
            "acf_within_regime": acf,
            "max_abs_acf_within_regime": float(np.nanmax(np.abs(acf))),
            "lag1_pooled": _corr(x[:, 1:], x[:, :-1]),
            "dow_means_normal": dow_means,
            "dow_relative_range": float((max(dow_means) - min(dow_means)) / overall) if overall else float("nan"),
        }
    return out


def aggregation_noise(panel: Panel, cfg: PipelineConfig, phc_outbreak: np.ndarray,
                      surveillance_days: np.ndarray) -> dict:
    district_outbreak = np.nan_to_num(panel.district_prevalence) > cfg.data.outbreak_prevalence_threshold
    out = {}
    for name, target in cfg.targets.items():
        x = panel.series(target.series)
        phc = _describe(x[~phc_outbreak])
        district_mean = _district_daily_mean(panel, x)
        district = _describe(district_mean[~district_outbreak])
        network = _describe(np.nanmean(x, axis=0)[~surveillance_days])
        out[name] = {"phc_cv": phc.get("cv"), "district_cv": district.get("cv"), "network_cv": network.get("cv")}
    return out


def driver_correlations(panel: Panel, cfg: PipelineConfig, phc_outbreak: np.ndarray) -> dict:
    drivers = {name: panel.series(name) for name in ("patient_footfall", "bed_occupancy", "staff_availability", "outbreak_flag")}
    driver_dev = {name: _regime_deviation(x, phc_outbreak) for name, x in drivers.items()}
    out = {}
    for name, target in cfg.targets.items():
        x = panel.series(target.series)
        deviation = _regime_deviation(x, phc_outbreak)
        out[name] = {
            driver: {"pooled": _corr(x, dx), "within_regime": _corr(deviation, driver_dev[driver])}
            for driver, dx in drivers.items() if driver != target.series
        }
    return out


def outbreak_signal_quality(repo: SenetraRepository, panel: Panel, phc_outbreak: np.ndarray) -> dict:
    crosstab = repo.query(
        """SELECT outbreak_flag AS flag, (outbreak_type IS NOT NULL) AS has_type, COUNT(*) AS rows
           FROM daily_metrics GROUP BY 1, 2"""
    )
    total = crosstab["rows"].sum()
    mismatch = crosstab.loc[crosstab["flag"] != crosstab["has_type"], "rows"].sum()
    flag = panel.series("outbreak_flag")
    footfall = panel.series("patient_footfall")
    in_outbreak = phc_outbreak & np.isfinite(flag)
    flagged = footfall[in_outbreak & (flag == 1)]
    unflagged = footfall[in_outbreak & (flag == 0)]
    return {
        "flag_type_mismatch_share": float(mismatch / total) if total else float("nan"),
        "flag_share_during_outbreak": float(np.nanmean(flag[phc_outbreak])) if phc_outbreak.any() else float("nan"),
        "flag_share_during_normal": float(np.nanmean(flag[~phc_outbreak])),
        "footfall_flagged_vs_unflagged_during_outbreak": (
            float(np.nanmean(flagged) / np.nanmean(unflagged)) if flagged.size and unflagged.size else float("nan")),
    }


def stock_analysis(repo: SenetraRepository, panel: Panel) -> dict:
    (chained, compared), = repo.rows(
        """SELECT SUM(opening_stock = prev_close), COUNT(prev_close) FROM (
             SELECT opening_stock, LAG(closing_stock) OVER (PARTITION BY phc_id, medicine_id ORDER BY date) AS prev_close
             FROM medicine_consumption)"""
    )
    inventory = repo.inventory()
    position = pd.Series(np.arange(panel.n_phcs), index=panel.phc_ids)
    days = {}
    for medicine, rows in inventory.groupby("medicine"):
        series = f"medicine:{medicine}"
        if series not in panel.series_names:
            continue
        recent = np.nanmean(panel.series(series)[:, -7:], axis=1)
        idx = position.reindex(rows["phc_id"]).to_numpy()
        ok = np.isfinite(idx)
        consumption = recent[idx[ok].astype(int)]
        with np.errstate(divide="ignore", invalid="ignore"):
            dos = rows["current_stock"].to_numpy(dtype=float)[ok] / consumption
        dos = dos[np.isfinite(dos)]
        days[medicine] = {**_describe(dos), "share_le_3_days": float((dos <= 3).mean()),
                          "share_le_7_days": float((dos <= 7).mean())}
    return {"ledger_coherence": float((chained or 0) / max(compared or 0, 1)), "days_of_supply": days,
            "inventory_values": inventory["current_stock"].to_numpy(dtype=float)}


def event_alignment(repo: SenetraRepository, panel: Panel) -> dict:
    events = repo.query("SELECT district_id, event_type, severity, active, started_at FROM simulation_events")
    lo, hi = panel.dates[0], panel.dates[-1]
    events["started"] = pd.to_datetime(events["started_at"].str[:10])
    active = events[(events["active"] == 1) & (events["started"] <= hi)]
    footfall = panel.series("patient_footfall")
    event_districts = set(active["district_id"])
    district_of_phc = panel.hierarchy["district_id"].to_numpy()
    ratios = []
    for event in active.itertuples(index=False):
        start = panel.date_index(max(event.started, lo))
        inside = district_of_phc == event.district_id
        control = ~np.isin(district_of_phc, list(event_districts))
        if inside.any() and control.any():
            ratios.append(float(np.nanmean(footfall[inside, start:]) / np.nanmean(footfall[control, start:])))
    return {
        "events_total": int(len(events)),
        "events_active": int((events["active"] == 1).sum()),
        "events_started_before_data": int((events["started"] < lo).sum()),
        "active_event_types": active["event_type"].value_counts().to_dict(),
        "footfall_ratio_event_vs_control": {
            "mean": float(np.mean(ratios)) if ratios else float("nan"),
            "min": float(np.min(ratios)) if ratios else float("nan"),
            "max": float(np.max(ratios)) if ratios else float("nan"),
        },
    }


def implications(summary: dict, cfg: PipelineConfig) -> list[str]:
    notes = []
    temporal, profiles, noise = summary["temporal"], summary["targets"], summary["aggregation_noise"]
    flat_acf = [t for t, v in temporal.items() if v["max_abs_acf_within_regime"] < 0.05]
    if flat_acf:
        notes.append(
            f"Within a regime, daily values of {', '.join(flat_acf)} behave like independent noise "
            "(max |ACF| < 0.05 up to lag 14). Lag features add little, accuracy is bounded by irreducible noise, "
            "so evaluation reports skill against naive baselines and a non-deployable noise floor."
        )
    responsive = {t: v["outbreak_uplift"] for t, v in profiles.items() if np.isfinite(v["outbreak_uplift"])
                  and abs(v["outbreak_uplift"] - 1) >= 0.1}
    flat = [t for t, v in profiles.items() if np.isfinite(v["outbreak_uplift"]) and abs(v["outbreak_uplift"] - 1) < 0.05]
    if responsive:
        notes.append("Outbreak regime multiplies demand: " + ", ".join(f"{t} x{u:.2f}" for t, u in responsive.items())
                     + ". A log-link (multiplicative) model with a scenario covariate for outbreak intensity fits this shape.")
    if flat:
        notes.append(f"{', '.join(flat)} show no outbreak response (<5%); forecasts reduce to a stable mean and "
                     "threshold-based risk rules are sufficient.")
    ratios = [v["district_cv"] / v["phc_cv"] for v in noise.values() if v.get("phc_cv") and v.get("district_cv")]
    if ratios:
        notes.append(f"Aggregating PHCs to districts cuts relative noise to {np.median(ratios):.0%} of PHC level "
                     "(median across targets), so district and higher forecasts are far more precise than single-PHC ones.")
    dow = [v["dow_relative_range"] for v in temporal.values() if np.isfinite(v["dow_relative_range"])]
    if dow and max(dow) < 0.02:
        notes.append("No day-of-week pattern (<2% range); calendar features stay in the model but should carry ~zero weight.")
    drivers = summary["drivers"]
    confounded = [t for t, d in drivers.items() if "patient_footfall" in d
                  and abs(d["patient_footfall"]["pooled"]) > 0.3 and abs(d["patient_footfall"]["within_regime"]) < 0.05]
    if confounded:
        notes.append(f"Footfall correlates with {', '.join(confounded)} only through the shared outbreak regime "
                     "(high pooled, ~0 within-regime correlation): footfall is a regime indicator, not a per-PHC driver.")
    if summary["stock"]["ledger_coherence"] < 0.95:
        notes.append(f"Only {summary['stock']['ledger_coherence']:.1%} of opening stocks match the previous closing stock; "
                     "stock-out risk must use the inventory snapshot plus forecasts, not the historical ledger.")
    ratio = summary["events"]["footfall_ratio_event_vs_control"]["mean"]
    if np.isfinite(ratio) and abs(ratio - 1) < 0.05:
        notes.append("Active simulation_events show no footfall effect in their districts (ratio "
                     f"{ratio:.2f}); they are used as scenario inputs, never as training labels.")
    geo = summary["geography"]
    if geo["countries_with_phcs"] <= 1:
        notes.append(f"Only {geo['countries_with_phcs']} country has PHC data; the country and world federation tiers "
                     "are exercised with a single member until more countries onboard.")
    if geo["districts_missing_coordinates"]:
        notes.append("District/PHC coordinates are missing, so distance-based features and transfer costs are unavailable.")
    episodes = len(summary["regimes"]["surveillance_segments"])
    if episodes <= 1:
        notes.append(f"History holds {episodes} outbreak episode(s): the outbreak effect is estimated from one episode, "
                     "and the cold-start backtest fold measures performance before any outbreak is seen.")
    label = summary["outbreak_signal"]["flag_type_mismatch_share"]
    if np.isfinite(label) and label > 0.001:
        notes.append(f"{label:.1%} of rows disagree between outbreak_flag and outbreak_type; features use the flag only.")
    return notes


def run_analysis(repo: SenetraRepository, panel: Panel, cfg: PipelineConfig) -> tuple[dict, dict]:
    regime_summary, phc_outbreak, surveillance_days = regimes(panel, cfg)
    stock = stock_analysis(repo, panel)
    inventory_values = stock.pop("inventory_values")
    summary = {
        "inventory": data_inventory(repo),
        "geography": geography(repo, panel),
        "static_attributes": static_attributes(repo),
        "regimes": regime_summary,
        "targets": target_profiles(panel, cfg, phc_outbreak),
        "temporal": temporal_structure(panel, cfg, phc_outbreak),
        "aggregation_noise": aggregation_noise(panel, cfg, phc_outbreak, surveillance_days),
        "drivers": driver_correlations(panel, cfg, phc_outbreak),
        "outbreak_signal": outbreak_signal_quality(repo, panel, phc_outbreak),
        "stock": stock,
        "events": event_alignment(repo, panel),
    }
    summary["implications"] = implications(summary, cfg)
    network = {name: np.nanmean(panel.series(t.series), axis=0) for name, t in cfg.targets.items()}
    plot_data = {"dates": panel.dates, "network_means": network, "outbreak_days": surveillance_days,
                 "inventory_values": inventory_values}
    return summary, plot_data
