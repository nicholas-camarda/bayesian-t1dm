"""What Control-IQ does relative to the programmed basal, by time of day.

The pump logs, about every 5 minutes (event 279), the rate it commanded and the rate the profile calls for. The
controller adds insulin above the profile when it predicts glucose will run high and withholds it when it predicts a
low, so a controller that adds the same extra on most days at the same hours, while glucose still averages above
range, is telling us the programmed basal is too low there (and the reverse for withholding).

Per day and pump segment this module measures the net insulin the controller delivered above or below the profile,
as a fraction of the profile's own insulin, over the time that is not explained by something else: not the 4 hours
after a bolus or logged carbohydrate, not exercise mode or the hour after it, not near unlogged food, and not on a
day listed in ``context_events.csv``. Sleep mode is kept: it governs the overnight hours this analysis is mostly about,
and the sleep-mode glucose target (higher than the daytime target) is reported as its own mode column.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .outcomes import MIN_DAY_COVERAGE, DayPart

BASAL_EVENT = 279
MODE_EVENT = 229
MODE_NAMES = {0: "standard", 1: "sleep", 2: "exercise"}
MAX_INTERVAL_MINUTES = 10  # delivery records come every ~5 minutes; longer gaps are not bridged
AFTER_MEAL_HOURS = 4
AFTER_EXERCISE_HOURS = 1
MIN_MEAL_BOLUS = 0.5
TARGET_LOW, TARGET_HIGH = 70.0, 180.0


def load_basal_delivery(tandem_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Commanded and profile basal rates (U/h) and user-mode changes from the archived pump-logs JSON.

    Each rate record is kept to the second; the same record in overlapping windows is kept once.
    """
    rates: list[dict] = []
    modes: list[dict] = []
    for path in sorted(Path(tandem_root).glob("*/raw/pump_logs_*.json")):
        if path.name.endswith(".meta.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        pump = str(payload["pump_serial"])
        for chunk in payload["chunks"]:
            for event in chunk["response"].get("events") or []:
                props = event["eventProperties"]
                if event["eventCode"] == BASAL_EVENT:
                    rates.append(
                        {"pump_serial": pump, "sequence": int(event["sequenceNumber"]), "timestamp": pd.Timestamp(event["pumpDateTime"]),
                         "commanded_u_per_h": props["commandedRate"] / 1000.0, "profile_u_per_h": props["profileBasalRate"] / 1000.0}
                    )
                elif event["eventCode"] == MODE_EVENT:
                    modes.append({"pump_serial": pump, "sequence": int(event["sequenceNumber"]), "timestamp": pd.Timestamp(event["pumpDateTime"]), "mode": int(props["currentUserMode"])})
    rate_frame = pd.DataFrame(rates, columns=["pump_serial", "sequence", "timestamp", "commanded_u_per_h", "profile_u_per_h"])
    mode_frame = pd.DataFrame(modes, columns=["pump_serial", "sequence", "timestamp", "mode"])
    rate_frame = rate_frame.drop_duplicates(["pump_serial", "sequence"]).sort_values("timestamp").reset_index(drop=True)
    mode_frame = mode_frame.drop_duplicates(["pump_serial", "sequence"]).sort_values("timestamp").reset_index(drop=True)
    return rate_frame, mode_frame


def _mode_at(times: pd.Series, modes: pd.DataFrame) -> np.ndarray:
    if modes.empty:
        return np.zeros(len(times), dtype=int)
    position = np.searchsorted(modes["timestamp"].to_numpy(dtype="datetime64[ns]"), times.to_numpy(dtype="datetime64[ns]"), side="right") - 1
    values = modes["mode"].to_numpy()
    return np.where(position >= 0, values[np.maximum(position, 0)], 0)


def _near_after(times: pd.Series, anchors: np.ndarray, hours: float) -> np.ndarray:
    """True when an anchor lies within ``hours`` before the time."""
    if len(anchors) == 0:
        return np.zeros(len(times), dtype=bool)
    anchors = np.sort(anchors)
    values = times.to_numpy(dtype="datetime64[ns]")
    position = np.searchsorted(anchors, values, side="right") - 1
    last = anchors[np.maximum(position, 0)]
    return (position >= 0) & ((values - last) <= np.timedelta64(int(hours * 60), "m"))


def delivery_intervals(rates: pd.DataFrame) -> pd.DataFrame:
    """Each rate record held until the next, with the extra insulin (commanded minus profile) over that interval."""
    frame = rates.sort_values("timestamp").reset_index(drop=True).copy()
    gap = frame["timestamp"].shift(-1) - frame["timestamp"]
    frame["hours"] = gap.dt.total_seconds() / 3600.0
    frame = frame[(frame["hours"] > 0) & (frame["hours"] <= MAX_INTERVAL_MINUTES / 60.0)].copy()
    frame["excess_units"] = (frame["commanded_u_per_h"] - frame["profile_u_per_h"]) * frame["hours"]
    frame["profile_units"] = frame["profile_u_per_h"] * frame["hours"]
    return frame


def daily_controller_pattern(
    rates: pd.DataFrame,
    modes: pd.DataFrame,
    cgm: pd.DataFrame,
    meal_times: np.ndarray,
    parts: list[DayPart],
    *,
    unlogged: pd.DataFrame | None = None,
    excluded_days=(),
) -> pd.DataFrame:
    """One row per day and pump segment: extra insulin as a share of the profile's, and mean glucose, over the clean time."""
    intervals = delivery_intervals(rates)
    if intervals.empty:
        return pd.DataFrame()
    stamp = intervals["timestamp"]
    mode = _mode_at(stamp, modes)
    # "within an hour after exercise mode began or ended": any mode change away from exercise counts as an end.
    exercise_ends = modes.loc[modes["mode"].shift(1).eq(2) & modes["mode"].ne(2), "timestamp"].to_numpy(dtype="datetime64[ns]") if not modes.empty else np.array([], dtype="datetime64[ns]")
    keep = ~_near_after(stamp, meal_times, AFTER_MEAL_HOURS)
    keep &= mode != 2
    keep &= ~_near_after(stamp, exercise_ends, AFTER_EXERCISE_HOURS)
    if unlogged is not None and not unlogged.empty:
        flagged = np.zeros(len(stamp), dtype=bool)
        for row in unlogged.itertuples(index=False):
            flagged |= ((stamp >= row.start) & (stamp <= row.end)).to_numpy()
        keep &= ~flagged
    excluded = {pd.Timestamp(day).normalize() for day in excluded_days}
    intervals = intervals.assign(mode=mode)[keep].copy()
    intervals["date"] = intervals["timestamp"].dt.normalize()
    intervals = intervals[~intervals["date"].isin(excluded)]
    intervals["minute"] = intervals["timestamp"].dt.hour * 60 + intervals["timestamp"].dt.minute
    glucose = cgm[["timestamp", "glucose"]].copy()
    glucose["timestamp"] = pd.to_datetime(glucose["timestamp"])
    glucose = glucose.dropna().sort_values("timestamp")
    rows = []
    for part in parts:
        inside = intervals[part.contains(intervals["minute"])]
        for date, group in inside.groupby("date"):
            hours = float(group["hours"].sum())
            profile_units = float(group["profile_units"].sum())
            if hours < part.hours * MIN_DAY_COVERAGE or profile_units <= 0:
                continue
            window = glucose[(glucose["timestamp"] >= group["timestamp"].min()) & (glucose["timestamp"] <= group["timestamp"].max() + pd.Timedelta(minutes=5))]
            clean = window[window["timestamp"].dt.floor("5min").isin(group["timestamp"].dt.floor("5min"))]["glucose"]
            rows.append(
                {
                    "date": date, "part": part.name, "clean_hours": hours, "profile_units": profile_units,
                    "excess_units": float(group["excess_units"].sum()), "excess_fraction": float(group["excess_units"].sum() / profile_units),
                    "mean_glucose": float(clean.mean()) if len(clean) else np.nan, "time_in_range_pct": float(100 * ((clean >= TARGET_LOW) & (clean <= TARGET_HIGH)).mean()) if len(clean) else np.nan,
                    "below_70_pct": float(100 * (clean < TARGET_LOW).mean()) if len(clean) else np.nan,
                    "below_54_pct": float(100 * (clean < 54.0).mean()) if len(clean) else np.nan,
                    "sleep_share": float((group["mode"] == 1).mean()),
                }
            )
    return pd.DataFrame(rows)


def controller_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Per segment: how many clean days, and the distribution of the controller's extra insulin across them."""
    rows = []
    for part, group in daily.groupby("part", sort=False):
        fraction = group["excess_fraction"]
        rows.append(
            {
                "part": part, "days": int(len(group)), "median_excess_fraction": float(fraction.median()),
                "p25": float(fraction.quantile(0.25)), "p75": float(fraction.quantile(0.75)), "sd_between_days": float(fraction.std(ddof=1)) if len(group) > 1 else np.nan,
                "mean_glucose": float(group["mean_glucose"].mean()), "time_in_range_pct": float(group["time_in_range_pct"].mean()),
                "sleep_share": float(group["sleep_share"].mean()),
            }
        )
    return pd.DataFrame(rows)


# The written rule for turning the pattern into a basal signal (chosen with the person, 2026-11-18).
SIGNAL_EXTRA_FRACTION = 0.25  # X: the controller runs this far above (or below) the profile
SIGNAL_SHARE_OF_DAYS = 0.50   # Y: on at least this share of the recent clean days
SIGNAL_RECENT_DAYS = 28
SIGNAL_MIN_DAYS = 14
HIGH_GLUCOSE_MGDL = 120.0     # the top of the sleep-mode target: glucose still above it despite the extra insulin


def basal_signal(daily: pd.DataFrame, *, recent_days: int = SIGNAL_RECENT_DAYS) -> pd.DataFrame:
    """Per pump segment, over its most recent clean days: does the controller consistently work against the programmed basal?

    "Basal likely too low": the controller delivers more than ``SIGNAL_EXTRA_FRACTION`` above the profile on at least
    ``SIGNAL_SHARE_OF_DAYS`` of the days, mean glucose over those days is still above ``HIGH_GLUCOSE_MGDL``, and time
    below 70 and below 54 stay within their limits. "Basal likely too high": it withholds more than that on at least
    that share of days. Fewer than ``SIGNAL_MIN_DAYS`` clean days gives no signal.
    """
    from .outcomes import LOW_LIMIT_PERCENT, VERY_LOW_LIMIT_PERCENT

    rows = []
    for part, group in daily.groupby("part", sort=False):
        recent = group.sort_values("date").tail(recent_days)
        n = len(recent)
        over = float((recent["excess_fraction"] > SIGNAL_EXTRA_FRACTION).mean()) if n else np.nan
        under = float((recent["excess_fraction"] < -SIGNAL_EXTRA_FRACTION).mean()) if n else np.nan
        glucose = float(recent["mean_glucose"].mean()) if n else np.nan
        below_70, below_54 = (float(recent["below_70_pct"].mean()), float(recent["below_54_pct"].mean())) if n else (np.nan, np.nan)
        lows_ok = below_70 < LOW_LIMIT_PERCENT and below_54 < VERY_LOW_LIMIT_PERCENT
        if n < SIGNAL_MIN_DAYS:
            signal = f"not enough clean days ({n} of {SIGNAL_MIN_DAYS})"
        elif over >= SIGNAL_SHARE_OF_DAYS and glucose > HIGH_GLUCOSE_MGDL and lows_ok:
            signal = "basal likely too low"
        elif over >= SIGNAL_SHARE_OF_DAYS and not lows_ok:
            signal = "controller adds insulin but lows are over a limit: no added basal"
        elif over >= SIGNAL_SHARE_OF_DAYS:
            signal = "controller adds insulin and glucose is within the target: basal is being covered by the controller"
        elif under >= SIGNAL_SHARE_OF_DAYS:
            signal = "basal likely too high"
        else:
            signal = "no consistent pattern"
        rows.append(
            {"part": part, "days": n, "share_over_x": over, "share_under_minus_x": under, "mean_glucose": glucose, "below_70_pct": below_70,
             "below_54_pct": below_54, "median_excess_fraction": float(recent["excess_fraction"].median()) if n else np.nan, "signal": signal}
        )
    return pd.DataFrame(rows)
