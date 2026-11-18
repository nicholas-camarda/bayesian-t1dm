"""Consensus glucose outcome metrics: the numbers a setting change is meant to move.

Metrics follow the international consensus on CGM use (Battelino et al. 2019): time in range 70-180 mg/dL,
time below 70 and below 54, time above 180 and above 250, mean glucose, coefficient of variation and the glucose
management indicator (GMI = 3.31 + 0.02392 x mean glucose in mg/dL). A day counts only when it holds at least 70%
of its readings (the consensus minimum for a 14-day report), so a day with a sensor gap does not shift a rate.

Time-of-day bins are the pump's own profile segments, so a change to one setting is judged over the hours it
governs.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

READING_MINUTES = 5
READINGS_PER_DAY = 24 * 60 // READING_MINUTES
MIN_DAY_COVERAGE = 0.70
TARGET_LOW = 70.0
TARGET_HIGH = 180.0
VERY_LOW = 54.0
VERY_HIGH = 250.0
# Consensus targets: time in range above 70%, below 70 under 4%, below 54 under 1%.
LOW_LIMIT_PERCENT = 4.0
VERY_LOW_LIMIT_PERCENT = 1.0

METRIC_COLUMNS = [
    "readings",
    "coverage",
    "time_in_range_pct",
    "time_below_70_pct",
    "time_below_54_pct",
    "time_above_180_pct",
    "time_above_250_pct",
    "mean_glucose_mgdl",
    "cv_pct",
    "gmi_pct",
]


@dataclass(frozen=True)
class DayPart:
    """One time-of-day bin, ``start_minutes`` inclusive to ``end_minutes`` exclusive (minutes after midnight)."""

    name: str
    start_minutes: int
    end_minutes: int

    def contains(self, minutes: pd.Series | np.ndarray) -> np.ndarray:
        minutes = np.asarray(minutes)
        if self.start_minutes < self.end_minutes:
            return (minutes >= self.start_minutes) & (minutes < self.end_minutes)
        return (minutes >= self.start_minutes) | (minutes < self.end_minutes)  # wraps midnight

    @property
    def hours(self) -> float:
        return ((self.end_minutes - self.start_minutes) % (24 * 60) or 24 * 60) / 60.0


def _label(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def day_parts_from_starts(starts: Sequence[int]) -> list[DayPart]:
    """Bins that run from each segment start to the next, the last wrapping to the first."""
    ordered = sorted({int(s) % (24 * 60) for s in starts})
    if not ordered:
        raise ValueError("At least one segment start is needed to build time-of-day bins.")
    parts = []
    for index, start in enumerate(ordered):
        end = ordered[(index + 1) % len(ordered)]
        parts.append(DayPart(f"{_label(start)}-{_label(end)}", start, end))
    return parts


def segment_starts_in_force(history, pump_serial: str, moment: dt.datetime, *, step_minutes: int = 15) -> list[int]:
    """Start minutes of the segments of the profile active at ``moment``, read off the pump's profile history."""
    profile_id = history.active_profile(pump_serial, moment)
    if profile_id is None:
        return []
    starts: set[int] = set()
    for minute in range(0, 24 * 60, step_minutes):
        segment = history.segment_in_force(pump_serial, profile_id, moment, minute_of_day=minute)
        if segment is not None:
            starts.add(int(segment["start_minutes"]))
    return sorted(starts)


def _summarize(glucose: pd.Series, *, expected_readings: float) -> dict[str, float]:
    values = pd.to_numeric(glucose, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(values)
    row: dict[str, float] = {name: np.nan for name in METRIC_COLUMNS}
    row["readings"] = float(n)
    row["coverage"] = n / expected_readings if expected_readings > 0 else np.nan
    if n == 0:
        return row
    mean = float(values.mean())
    row.update(
        time_in_range_pct=100.0 * float(((values >= TARGET_LOW) & (values <= TARGET_HIGH)).mean()),
        time_below_70_pct=100.0 * float((values < TARGET_LOW).mean()),
        time_below_54_pct=100.0 * float((values < VERY_LOW).mean()),
        time_above_180_pct=100.0 * float((values > TARGET_HIGH).mean()),
        time_above_250_pct=100.0 * float((values > VERY_HIGH).mean()),
        mean_glucose_mgdl=mean,
        cv_pct=100.0 * float(values.std(ddof=0)) / mean if mean > 0 else np.nan,
        gmi_pct=3.31 + 0.02392 * mean,
    )
    return row


def _prepare(cgm: pd.DataFrame) -> pd.DataFrame:
    frame = cgm[["timestamp", "glucose"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["glucose"] = pd.to_numeric(frame["glucose"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "glucose"])
    # One reading per 5-minute slot: two pumps or a re-download can report the same reading twice.
    frame["slot"] = frame["timestamp"].dt.floor(f"{READING_MINUTES}min")
    frame = frame.drop_duplicates("slot", keep="first")
    frame["date"] = frame["timestamp"].dt.normalize()
    frame["minute"] = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
    return frame


def daily_metrics(cgm: pd.DataFrame, *, part: DayPart | None = None) -> pd.DataFrame:
    """Per-day consensus metrics, over all 24 hours or over one day part; one row per day, ``date`` as index.

    ``counted`` marks a day with at least 70% of the readings expected for the hours covered.
    """
    frame = _prepare(cgm)
    expected = READINGS_PER_DAY if part is None else part.hours * 60 / READING_MINUTES
    if part is not None:
        frame = frame[part.contains(frame["minute"])]
    rows = []
    for date, group in frame.groupby("date", sort=True):
        row = _summarize(group["glucose"], expected_readings=expected)
        row["date"] = date
        rows.append(row)
    out = pd.DataFrame(rows, columns=["date", *METRIC_COLUMNS])
    out["counted"] = out["coverage"] >= MIN_DAY_COVERAGE
    return out.set_index("date")


def pooled_metrics(cgm: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp, part: DayPart | None = None) -> dict[str, float]:
    """Consensus metrics over readings from ``start`` (inclusive) to ``end`` (exclusive), counting only days with
    enough coverage. ``days`` is the number of days that count out of ``days_in_window``."""
    frame = _prepare(cgm)
    frame = frame[(frame["timestamp"] >= start) & (frame["timestamp"] < end)]
    expected = READINGS_PER_DAY if part is None else part.hours * 60 / READING_MINUTES
    if part is not None:
        frame = frame[part.contains(frame["minute"])]
    counts = frame.groupby("date")["glucose"].count()
    counted = counts[counts / expected >= MIN_DAY_COVERAGE].index
    kept = frame[frame["date"].isin(counted)]
    days_in_window = int(np.ceil((end - start) / pd.Timedelta(days=1)))
    row = _summarize(kept["glucose"], expected_readings=expected * max(len(counted), 1))
    row["days"] = float(len(counted))
    row["days_in_window"] = float(days_in_window)
    return row


def baseline_table(cgm: pd.DataFrame, *, as_of: pd.Timestamp, windows_days: Sequence[int] = (14, 90)) -> pd.DataFrame:
    """Whole-day metrics for the last ``windows_days`` days before ``as_of`` (exclusive)."""
    rows = []
    for days in windows_days:
        row = pooled_metrics(cgm, start=as_of - pd.Timedelta(days=days), end=as_of)
        rows.append({"window": f"last {days} days", **row})
    return pd.DataFrame(rows)


def day_part_table(cgm: pd.DataFrame, parts: Sequence[DayPart], *, as_of: pd.Timestamp, days: int = 90) -> pd.DataFrame:
    """Metrics per time-of-day bin over the last ``days`` days, to show where lows and highs concentrate."""
    rows = []
    for part in parts:
        row = pooled_metrics(cgm, start=as_of - pd.Timedelta(days=days), end=as_of, part=part)
        rows.append({"day_part": part.name, "hours": part.hours, **row})
    return pd.DataFrame(rows)


def passes_low_limits(row: dict[str, float] | pd.Series) -> bool:
    """True when time below 70 is under 4% and time below 54 under 1%."""
    below_70 = float(row["time_below_70_pct"])
    below_54 = float(row["time_below_54_pct"])
    return bool(np.isfinite(below_70) and np.isfinite(below_54) and below_70 < LOW_LIMIT_PERCENT and below_54 < VERY_LOW_LIMIT_PERCENT)
