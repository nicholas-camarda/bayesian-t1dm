"""Shared definitions every analysis reads from, so one rule is stated once.

- CGM validity: a Tandem CGM reading counts when it is a number from ``CGM_MIN_MGDL`` to ``CGM_MAX_MGDL`` (the sensor's
  reporting range); readings of 0 or below 40 mg/dL are sensor artifacts, not glucose. Only Tandem CGM readings are used for
  glucose metrics (Apple Health blood glucose is not a sensor stream).
- Exercise: a workout of at least ``EXERCISE_MIN_MINUTES`` (walks included) from its start until ``EXERCISE_AFTER_HOURS``
  after its end, and any exercise mode on the pump from its start until ``EXERCISE_AFTER_HOURS`` after it ends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CGM_MIN_MGDL = 40.0
CGM_MAX_MGDL = 400.0
EXERCISE_MIN_MINUTES = 10
EXERCISE_AFTER_HOURS = 4.0
EXERCISE_MODE = 2


def valid_cgm(cgm: pd.DataFrame) -> pd.DataFrame:
    """The CGM readings inside the sensor's reporting range, with glucose as a number."""
    out = cgm.copy()
    out["glucose"] = pd.to_numeric(out["glucose"], errors="coerce")
    return out[out["glucose"].between(CGM_MIN_MGDL, CGM_MAX_MGDL)].reset_index(drop=True)


def exercise_intervals(workouts: pd.DataFrame | None, modes: pd.DataFrame | None) -> pd.DataFrame:
    """Start and end of every exercise interval (``start``, ``end``), sorted, from workouts and exercise-mode periods."""
    rows: list[pd.DataFrame] = []
    after = pd.Timedelta(hours=EXERCISE_AFTER_HOURS)
    if workouts is not None and not workouts.empty:
        long = workouts[pd.to_numeric(workouts["duration_seconds"], errors="coerce") >= EXERCISE_MIN_MINUTES * 60]
        rows.append(pd.DataFrame({"start": pd.to_datetime(long["start_timestamp"]), "end": pd.to_datetime(long["end_timestamp"]) + after}))
    if modes is not None and not modes.empty:
        for _, group in modes.sort_values("timestamp").groupby("pump_serial"):
            stamps = group["timestamp"].reset_index(drop=True)
            into = group["mode"].reset_index(drop=True).eq(EXERCISE_MODE)
            ends = stamps.shift(-1)  # the mode lasts until the pump's next mode change
            rows.append(pd.DataFrame({"start": stamps[into], "end": ends[into] + after}).dropna())
    if not rows:
        return pd.DataFrame({"start": pd.Series(dtype="datetime64[ns]"), "end": pd.Series(dtype="datetime64[ns]")})
    out = pd.concat(rows, ignore_index=True).sort_values("start").reset_index(drop=True)
    return out.astype({"start": "datetime64[ns]", "end": "datetime64[ns]"})


def inside_intervals(times: pd.Series, intervals: pd.DataFrame) -> np.ndarray:
    """True for each time that falls inside any ``start``..``end`` interval."""
    values = pd.to_datetime(times).to_numpy(dtype="datetime64[ns]")
    if intervals.empty:
        return np.zeros(len(values), dtype=bool)
    starts = intervals["start"].to_numpy(dtype="datetime64[ns]")
    ends = intervals["end"].to_numpy(dtype="datetime64[ns]")
    order = np.argsort(starts)
    starts, ends = starts[order], ends[order]
    reach = np.maximum.accumulate(ends)  # the furthest end among intervals that started at or before each one
    position = np.searchsorted(starts, values, side="right") - 1
    return (position >= 0) & (values <= reach[np.maximum(position, 0)])
