"""Markdown report and figures for the EDA summary."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, float):
        return "-" if not np.isfinite(value) else f"{value:,.{digits}f}"
    return str(value)


def _shade(ax, dates, mask) -> None:
    for i, flag in enumerate(mask):
        if flag:
            ax.axvspan(dates[i], dates[i] + np.timedelta64(1, "D"), color="#f4a261", alpha=0.18, lw=0)


def write_figures(summary: dict, plot_data: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    dates, outbreak = plot_data["dates"], plot_data["outbreak_days"]

    targets = list(plot_data["network_means"])
    fig, axes = plt.subplots(len(targets), 1, figsize=(11, 1.9 * len(targets)), sharex=True)
    for ax, name in zip(np.atleast_1d(axes), targets):
        series = plot_data["network_means"][name]
        _shade(ax, dates, outbreak)
        ax.plot(dates, series, color="#264653", lw=1.4)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("Network daily mean per PHC (shaded: outbreak days from surveillance)")
    fig.tight_layout()
    paths.append(out_dir / "network_daily_means.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    lags = np.arange(1, len(next(iter(summary["temporal"].values()))["acf_within_regime"]) + 1)
    width = 0.8 / len(targets)
    for i, name in enumerate(targets):
        ax.bar(lags + i * width, summary["temporal"][name]["acf_within_regime"], width=width, label=name)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xlabel="lag (days)", ylabel="autocorrelation", title="Within-regime autocorrelation of PHC daily values")
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    paths.append(out_dir / "acf_within_regime.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    noise = summary["aggregation_noise"]
    x = np.arange(len(targets))
    for i, (key, label) in enumerate([("phc_cv", "PHC"), ("district_cv", "District"), ("network_cv", "Network")]):
        ax.bar(x + (i - 1) * 0.27, [noise[t][key] or 0 for t in targets], width=0.27, label=label)
    ax.set_xticks(x, targets, rotation=30, ha="right", fontsize=8)
    ax.set(ylabel="coefficient of variation (normal regime)", title="Noise shrinks with aggregation")
    ax.legend()
    fig.tight_layout()
    paths.append(out_dir / "aggregation_noise.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    drivers = {t: d["patient_footfall"] for t, d in summary["drivers"].items() if "patient_footfall" in d}
    if drivers:
        fig, ax = plt.subplots(figsize=(10, 4))
        names = list(drivers)
        x = np.arange(len(names))
        ax.bar(x - 0.2, [drivers[t]["pooled"] for t in names], width=0.4, label="pooled")
        ax.bar(x + 0.2, [drivers[t]["within_regime"] for t in names], width=0.4, label="within regime")
        ax.set_xticks(x, names, rotation=30, ha="right", fontsize=8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set(ylabel="correlation with patient footfall", title="Footfall correlation is regime-driven")
        ax.legend()
        fig.tight_layout()
        paths.append(out_dir / "footfall_correlation.png")
        fig.savefig(paths[-1], dpi=120)
        plt.close(fig)

    days = summary["stock"]["days_of_supply"]
    if days:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(list(days), [v.get("p50", 0) for v in days.values()], color="#2a9d8f", label="median")
        ax.bar(list(days), [v.get("p05", 0) for v in days.values()], color="#e76f51", label="5th percentile")
        ax.set(ylabel="days of supply (snapshot / last-7-day use)", title="Inventory days of supply")
        ax.legend()
        fig.tight_layout()
        paths.append(out_dir / "days_of_supply.png")
        fig.savefig(paths[-1], dpi=120)
        plt.close(fig)
    return paths


def write_markdown(summary: dict, figures: list[Path], out_path: Path) -> Path:
    inv, geo, reg = summary["inventory"], summary["geography"], summary["regimes"]
    lines = [
        "# SENETRA exploratory data analysis",
        "",
        f"Generated from the operational database ({inv['start']} to {inv['end']}, {inv['days']} days). "
        "Only aggregate statistics appear here; no rows were exported.",
        "",
        "## Modelling implications",
        "",
        *[f"- {note}" for note in summary["implications"]],
        "",
        "## Data inventory",
        "",
        "| table | rows |",
        "|---|---:|",
        *[f"| {t} | {_fmt(n)} |" for t, n in inv["row_counts"].items()],
        "",
        "## Geography and federation coverage",
        "",
        f"{geo['phcs']:,} PHCs in {geo['districts_with_phcs']} districts across {geo['states_with_phcs']} state(s) "
        f"and {geo['countries_with_phcs']} country(ies). PHCs per district: min {geo['phcs_per_district']['min']}, "
        f"median {geo['phcs_per_district']['median']:.0f}, max {geo['phcs_per_district']['max']}. "
        f"{geo['districts_missing_coordinates']} of {geo['districts_total']} districts and "
        f"{geo['phcs_missing_coordinates']:,} PHCs lack coordinates.",
        "",
        "| country | states | districts | PHCs |",
        "|---|---:|---:|---:|",
        *[f"| {c['country']} | {c['states']} | {c['districts']} | {c['phcs']} |" for c in geo["countries"]],
        "",
        "Static PHC attributes (distinct values): "
        + ", ".join(f"{k}={v['distinct']}" for k, v in summary["static_attributes"].items()) + ".",
        "",
        "## Outbreak regimes",
        "",
        f"District-day outbreak threshold: prevalence > {reg['threshold']}. Outbreak days: {reg['outbreak_days']}, "
        f"normal days: {reg['normal_days']}. Surveillance-based and demand-based regime detection agree on "
        f"{reg['surveillance_demand_agreement']:.1%} of days.",
        "",
        "Surveillance segments: " + (", ".join(f"{s['start']}..{s['end']} ({s['days']}d)"
                                            for s in reg["surveillance_segments"]) or "none") + ".  ",
        "Demand-shift segments: " + (", ".join(f"{s['start']}..{s['end']} ({s['days']}d)"
                                             for s in reg["demand_shift_segments"]) or "none") + ".",
        "",
        "## Targets by regime",
        "",
        "| target | normal mean | normal CV | outbreak mean | outbreak CV | uplift | max within-regime ACF | lag-1 pooled | DOW range |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, profile in summary["targets"].items():
        temporal = summary["temporal"][name]
        lines.append(
            f"| {name} | {_fmt(profile['normal'].get('mean'))} | {_fmt(profile['normal'].get('cv'))} | "
            f"{_fmt(profile['outbreak'].get('mean'))} | {_fmt(profile['outbreak'].get('cv'))} | "
            f"{_fmt(profile['outbreak_uplift'])} | {_fmt(temporal['max_abs_acf_within_regime'], 3)} | "
            f"{_fmt(temporal['lag1_pooled'], 3)} | {temporal['dow_relative_range']:.1%} |"
        )
    lines += [
        "",
        "## Noise versus aggregation level (normal regime CV)",
        "",
        "| target | PHC | district | network |",
        "|---|---:|---:|---:|",
        *[f"| {t} | {_fmt(v['phc_cv'], 3)} | {_fmt(v['district_cv'], 3)} | {_fmt(v['network_cv'], 3)} |"
          for t, v in summary["aggregation_noise"].items()],
        "",
        "## Driver correlations (pooled / within regime)",
        "",
        "| target | footfall | bed occupancy | staff availability | outbreak flag |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, drivers in summary["drivers"].items():
        cells = []
        for driver in ("patient_footfall", "bed_occupancy", "staff_availability", "outbreak_flag"):
            d = drivers.get(driver)
            cells.append(f"{_fmt(d['pooled'])} / {_fmt(d['within_regime'])}" if d else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    sig, stock, ev = summary["outbreak_signal"], summary["stock"], summary["events"]
    lines += [
        "",
        "## Data quality signals",
        "",
        f"- outbreak_flag vs outbreak_type disagreement: {sig['flag_type_mismatch_share']:.2%} of rows.",
        f"- Outbreak flag share: {_fmt(sig['flag_share_during_outbreak'], 3)} during outbreaks, "
        f"{_fmt(sig['flag_share_during_normal'], 3)} otherwise; flagged/unflagged footfall ratio during outbreaks "
        f"{_fmt(sig['footfall_flagged_vs_unflagged_during_outbreak'], 3)}.",
        f"- Stock ledger coherence (opening = previous closing): {stock['ledger_coherence']:.1%}.",
        f"- Simulation events: {ev['events_active']} active of {ev['events_total']}, "
        f"{ev['events_started_before_data']} started before the data window; footfall ratio in active-event districts "
        f"vs others: {_fmt(ev['footfall_ratio_event_vs_control']['mean'], 3)}.",
        "",
        "## Inventory days of supply",
        "",
        "| medicine | p05 | median | p95 | share <= 3 days | share <= 7 days |",
        "|---|---:|---:|---:|---:|---:|",
        *[f"| {m} | {_fmt(v.get('p05'), 1)} | {_fmt(v.get('p50'), 1)} | {_fmt(v.get('p95'), 1)} | "
          f"{v['share_le_3_days']:.1%} | {v['share_le_7_days']:.1%} |" for m, v in stock["days_of_supply"].items()],
        "",
        "## Figures",
        "",
        *[f"![{p.stem}]({p.name})" for p in figures],
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
