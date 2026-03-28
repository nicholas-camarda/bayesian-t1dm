from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def _curve_parameters(duration_minutes: int, peak_minutes: int = 75) -> tuple[float, float, float]:
    if duration_minutes <= 2 * peak_minutes:
        raise ValueError("duration_minutes must be greater than twice peak_minutes")
    td = float(duration_minutes)
    tp = float(peak_minutes)
    tau = tp * (1 - tp / td) / (1 - (2 * tp / td))
    a = 2 * tau / td
    scale = 1 / (1 - a + (1 + a) * np.exp(-td / tau))
    return tau, a, scale


def insulin_action_curve(
    dose_units: float,
    duration_minutes: int = 300,
    step_minutes: int = 5,
    peak_minutes: int = 75,
) -> pd.DataFrame:
    if dose_units < 0:
        raise ValueError("dose_units must be non-negative")
    tau, a, scale = _curve_parameters(duration_minutes=duration_minutes, peak_minutes=peak_minutes)
    td = float(duration_minutes)
    time_grid = np.arange(0, duration_minutes + step_minutes, step_minutes, dtype=float)
    activity = dose_units * (scale / tau**2) * time_grid * (1 - time_grid / td) * np.exp(-time_grid / tau)
    iob = dose_units * (1 - scale * (1 - a) * (((time_grid**2 / (tau * td * (1 - a))) - time_grid / tau - 1) * np.exp(-time_grid / tau) + 1))
    curve = pd.DataFrame({
        "minutes": time_grid.astype(int),
        "ia_units": activity,
        "iob_units": iob,
    })
    curve["dose_units"] = dose_units
    curve["ia_fraction"] = np.where(dose_units > 0, curve["ia_units"] / dose_units, 0.0)
    curve["iob_fraction"] = np.where(dose_units > 0, curve["iob_units"] / dose_units, 0.0)
    return curve


def expand_bolus_to_grid(
    bolus: pd.DataFrame,
    time_grid: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    dose_col: str = "bolus_units",
    duration_minutes: int = 300,
    step_minutes: int = 5,
    peak_minutes: int = 75,
) -> pd.DataFrame:
    if bolus.empty:
        out = time_grid.copy()
        out["bolus_units"] = 0.0
        out["insulin_activity_units"] = 0.0
        out["iob_units"] = 0.0
        return out

    grid = time_grid.copy()
    grid[timestamp_col] = pd.to_datetime(grid[timestamp_col])
    contributions: list[pd.DataFrame] = []
    for row in bolus[[timestamp_col, dose_col]].dropna().itertuples(index=False):
        event_ts = pd.Timestamp(getattr(row, timestamp_col))
        dose = float(getattr(row, dose_col))
        if dose <= 0:
            continue
        curve = insulin_action_curve(
            dose_units=dose,
            duration_minutes=duration_minutes,
            step_minutes=step_minutes,
            peak_minutes=peak_minutes,
        )
        curve["timestamp"] = event_ts + pd.to_timedelta(curve["minutes"], unit="m")
        contributions.append(curve[["timestamp", "ia_units", "iob_units"]])

    if contributions:
        exposure = pd.concat(contributions, ignore_index=True)
        exposure["timestamp"] = exposure["timestamp"].dt.floor(f"{step_minutes}min")
        exposure = exposure.groupby("timestamp", as_index=False).sum(numeric_only=True)
    else:
        exposure = pd.DataFrame(columns=["timestamp", "ia_units", "iob_units"])

    merged = grid.merge(exposure, on="timestamp", how="left")
    merged["ia_units"] = merged["ia_units"].fillna(0.0)
    merged["iob_units"] = merged["iob_units"].fillna(0.0)
    merged["bolus_units"] = 0.0
    if dose_col in bolus.columns:
        bolus_by_time = bolus.groupby(pd.to_datetime(bolus[timestamp_col]).dt.floor(f"{step_minutes}min"))[dose_col].sum()
        merged["bolus_units"] = merged[timestamp_col].map(bolus_by_time).fillna(0.0)
    merged = merged.rename(columns={"ia_units": "insulin_activity_units"})
    return merged
