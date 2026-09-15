"""In-memory PHC x date x series panel assembled from PHC-scoped reads.

The panel lives only in process memory for the duration of a pipeline run; it is never
written to disk or logged, so the operational database stays the single copy of the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd

from senetra_ml.data.repository import SenetraRepository, day

METRIC_SERIES = ("patient_footfall", "bed_occupancy", "staff_availability", "outbreak_flag")


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    hierarchy: pd.DataFrame  # one row per PHC, in panel order
    series_names: tuple[str, ...]
    values: np.ndarray  # (C, T, K) float64, NaN where a PHC did not report
    district_ids: np.ndarray  # (D,)
    district_prevalence: np.ndarray  # (D, T) share of PHCs flagging an outbreak
    medicine_units: dict[str, str] = field(default_factory=dict)

    @property
    def n_phcs(self) -> int:
        return self.values.shape[0]

    @property
    def n_dates(self) -> int:
        return self.values.shape[1]

    @property
    def phc_ids(self) -> np.ndarray:
        return self.hierarchy["phc_id"].to_numpy()

    @cached_property
    def district_index(self) -> np.ndarray:
        """(C,) position of each PHC's district in `district_ids`."""
        return np.searchsorted(self.district_ids, self.hierarchy["district_id"].to_numpy())

    @cached_property
    def country_ids(self) -> np.ndarray:
        return np.unique(self.hierarchy["country_id"].to_numpy())

    @cached_property
    def district_country_index(self) -> np.ndarray:
        """(D,) position of each district's country in `country_ids`."""
        first = self.hierarchy.drop_duplicates("district_id").set_index("district_id")["country_id"]
        return np.searchsorted(self.country_ids, first.loc[self.district_ids].to_numpy())

    def series(self, name: str) -> np.ndarray:
        try:
            return self.values[:, :, self.series_names.index(name)]
        except ValueError:
            raise KeyError(f"Series {name!r} not in panel; available: {self.series_names}") from None

    def client_prevalence(self) -> np.ndarray:
        """(C, T) district outbreak prevalence broadcast to each PHC in the district."""
        return self.district_prevalence[self.district_index]

    def date_index(self, date: str | pd.Timestamp) -> int:
        position = self.dates.get_indexer([pd.Timestamp(date)])[0]
        if position < 0:
            raise KeyError(f"{day(date)} is outside the panel ({day(self.dates[0])}..{day(self.dates[-1])})")
        return int(position)


def load_panel(
    repo: SenetraRepository,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    phc_ids: list[int] | np.ndarray | None = None,
) -> Panel:
    lo, hi = repo.date_bounds()
    start_ts = max(pd.Timestamp(start), lo) if start is not None else lo
    end_ts = min(pd.Timestamp(end), hi) if end is not None else hi
    if start_ts > end_ts:
        raise ValueError(f"Empty date window {day(start_ts)}..{day(end_ts)}")
    dates = pd.date_range(start_ts, end_ts, freq="D")

    hierarchy = repo.hierarchy()
    if phc_ids is not None:
        wanted = pd.Index(np.asarray(phc_ids, dtype=int))
        hierarchy = hierarchy[hierarchy["phc_id"].isin(wanted)]
    hierarchy = hierarchy.reset_index(drop=True)
    if hierarchy.empty:
        raise ValueError("No PHCs match the requested scope")

    medicines = repo.medicines()
    series_names = METRIC_SERIES + tuple(f"medicine:{name}" for name in medicines["name"])
    medicine_column = {int(mid): len(METRIC_SERIES) + i for i, mid in enumerate(medicines["medicine_id"])}
    position = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(dates)}

    values = np.full((len(hierarchy), len(dates), len(series_names)), np.nan)
    for ci, phc_id in enumerate(hierarchy["phc_id"].tolist()):
        # Client-local reads: each query touches only this PHC's rows.
        for date, *metrics in repo.phc_metric_rows(phc_id, start_ts, end_ts):
            t = position.get(date[:10])
            if t is not None:
                values[ci, t, :4] = [np.nan if v is None else v for v in metrics]
        for date, medicine_id, consumption in repo.phc_consumption_rows(phc_id, start_ts, end_ts):
            t = position.get(date[:10])
            k = medicine_column.get(int(medicine_id))
            if t is not None and k is not None and consumption is not None:
                values[ci, t, k] = consumption

    district_ids = np.sort(hierarchy["district_id"].unique())
    prevalence = np.full((len(district_ids), len(dates)), np.nan)
    surveillance = repo.district_surveillance(start_ts, end_ts)
    surveillance = surveillance[surveillance["district_id"].isin(district_ids)]
    if not surveillance.empty:
        d_idx = np.searchsorted(district_ids, surveillance["district_id"].to_numpy())
        t_idx = dates.get_indexer(surveillance["date"])
        keep = t_idx >= 0
        prevalence[d_idx[keep], t_idx[keep]] = surveillance["prevalence"].to_numpy()[keep]

    return Panel(
        dates=dates,
        hierarchy=hierarchy,
        series_names=series_names,
        values=values,
        district_ids=district_ids,
        district_prevalence=prevalence,
        medicine_units=dict(zip(medicines["name"], medicines["unit"].fillna(""))),
    )
