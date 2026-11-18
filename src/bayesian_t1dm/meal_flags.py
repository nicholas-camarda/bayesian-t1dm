"""Food and glucose rises the pump did not log, found by comparing the pump, Bevel (logged food) and the CGM.

Three flags, each read in one direction only:

- ``food_without_pump_carbs``: a Bevel entry with at least ``MIN_FOOD_CARBS`` g of carbohydrate and no pump carb entry
  or bolus within 30 minutes. The glucose after it is not explained by a logged dose.
- ``bevel_missing``: a pump carb entry with no Bevel entry within 30 minutes. This is normal (not every meal is
  logged in Bevel); it is reported, never read as "no meal", and analyses that need Bevel use only the pump carbs for
  that meal.
- ``possible_unlogged_food``: a glucose rise of at least ``RISE_MGDL`` within an hour with no pump carbs, no bolus
  and no Bevel entry from 2.5 hours before it (a logged meal explains the rise for its absorption time) to an hour after.

The first and third mark intervals of ``AFTER_FLAG_MINUTES`` after (and ``BEFORE_FLAG_MINUTES`` before) the event.
A meal window or a basal-evidence window that overlaps one is excluded from I:C and basal evidence and counted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MATCH_MINUTES = 30
MIN_FOOD_CARBS = 5.0
RISE_MGDL = 40.0
RISE_WINDOW_MINUTES = 60
QUIET_BEFORE_MINUTES = 150  # a logged meal explains a rise for its absorption time, not just the hour after
QUIET_AFTER_MINUTES = 60
BEFORE_FLAG_MINUTES = 30
AFTER_FLAG_MINUTES = 180
MIN_BOLUS_UNITS = 0.5
INTERVAL_COLUMNS = ["kind", "event_time", "start", "end"]


def _near(times: pd.Series, anchors: np.ndarray, minutes: int) -> np.ndarray:
    """True for each time that has an anchor within ``minutes``."""
    if len(anchors) == 0:
        return np.zeros(len(times), dtype=bool)
    anchors = np.sort(anchors)
    values = times.to_numpy(dtype="datetime64[ns]")
    position = np.searchsorted(anchors, values)
    gap = np.timedelta64(minutes, "m")
    before = np.where(position > 0, np.abs(values - anchors[np.maximum(position - 1, 0)]), np.timedelta64(10**4, "D"))
    after = np.where(position < len(anchors), np.abs(anchors[np.minimum(position, len(anchors) - 1)] - values), np.timedelta64(10**4, "D"))
    return (before <= gap) | (after <= gap)


def _within(times: pd.Series, anchors: np.ndarray, before_minutes: int, after_minutes: int) -> np.ndarray:
    """True for each time with an anchor from ``before_minutes`` earlier to ``after_minutes`` later."""
    if len(anchors) == 0:
        return np.zeros(len(times), dtype=bool)
    anchors = np.sort(anchors)
    values = times.to_numpy(dtype="datetime64[ns]")
    first = np.searchsorted(anchors, values - np.timedelta64(before_minutes, "m"))
    last = np.searchsorted(anchors, values + np.timedelta64(after_minutes, "m"), side="right")
    return last > first


def _pump_event_times(research_frame: pd.DataFrame) -> np.ndarray:
    carbs = pd.to_numeric(research_frame.get("explicit_carb_grams"), errors="coerce").fillna(0.0).gt(0)
    bolus = pd.to_numeric(research_frame.get("bolus_units"), errors="coerce").fillna(0.0).ge(MIN_BOLUS_UNITS)
    return pd.to_datetime(research_frame.loc[carbs | bolus, "timestamp"]).to_numpy(dtype="datetime64[ns]")


def _interval(kind: str, when: pd.Timestamp) -> dict[str, object]:
    return {"kind": kind, "event_time": when, "start": when - pd.Timedelta(minutes=BEFORE_FLAG_MINUTES), "end": when + pd.Timedelta(minutes=AFTER_FLAG_MINUTES)}


def food_without_pump_carbs(research_frame: pd.DataFrame, nutrition: pd.DataFrame | None) -> pd.DataFrame:
    if nutrition is None or nutrition.empty:
        return pd.DataFrame(columns=INTERVAL_COLUMNS)
    food = nutrition.copy()
    food["timestamp"] = pd.to_datetime(food["timestamp"], errors="coerce")
    food = food[pd.to_numeric(food["carb_grams"], errors="coerce").ge(MIN_FOOD_CARBS) & food["timestamp"].notna()]
    covered = _near(food["timestamp"], _pump_event_times(research_frame), MATCH_MINUTES)
    return pd.DataFrame([_interval("food_without_pump_carbs", t) for t in food.loc[~covered, "timestamp"]], columns=INTERVAL_COLUMNS)


def bevel_missing(research_frame: pd.DataFrame, nutrition: pd.DataFrame | None) -> pd.DataFrame:
    """Pump carb entries with no Bevel entry nearby (``event_time`` only; these are not exclusions)."""
    carbs = research_frame[pd.to_numeric(research_frame.get("explicit_carb_grams"), errors="coerce").fillna(0.0).gt(0)]
    times = pd.to_datetime(carbs["timestamp"])
    anchors = pd.to_datetime(nutrition["timestamp"], errors="coerce").dropna().to_numpy(dtype="datetime64[ns]") if nutrition is not None and not nutrition.empty else np.array([], dtype="datetime64[ns]")
    missing = times[~_near(times, anchors, MATCH_MINUTES)]
    return pd.DataFrame({"kind": "bevel_missing", "event_time": missing, "start": pd.NaT, "end": pd.NaT}, columns=INTERVAL_COLUMNS)


def possible_unlogged_food(research_frame: pd.DataFrame, nutrition: pd.DataFrame | None) -> pd.DataFrame:
    frame = research_frame[["timestamp", "glucose"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").set_index("timestamp")
    glucose = pd.to_numeric(frame["glucose"], errors="coerce")
    steps = RISE_WINDOW_MINUTES // 5
    rise = glucose.shift(-steps) - glucose
    starts = rise[rise.ge(RISE_MGDL)].index
    if len(starts) == 0:
        return pd.DataFrame(columns=INTERVAL_COLUMNS)
    anchors = [_pump_event_times(research_frame)]
    if nutrition is not None and not nutrition.empty:
        anchors.append(pd.to_datetime(nutrition["timestamp"], errors="coerce").dropna().to_numpy(dtype="datetime64[ns]"))
    all_anchors = np.concatenate(anchors)
    quiet = ~_within(pd.Series(starts), all_anchors, QUIET_BEFORE_MINUTES, QUIET_AFTER_MINUTES)
    rows, last = [], None
    for start in starts[quiet]:
        if last is not None and start - last < pd.Timedelta(minutes=AFTER_FLAG_MINUTES):
            continue  # one rise, one flag
        rows.append(_interval("possible_unlogged_food", start))
        last = start
    return pd.DataFrame(rows, columns=INTERVAL_COLUMNS)


def unlogged_intervals(research_frame: pd.DataFrame, nutrition: pd.DataFrame | None) -> pd.DataFrame:
    """The intervals that exclude a window: food without pump carbs and possible unlogged food."""
    parts = [food_without_pump_carbs(research_frame, nutrition), possible_unlogged_food(research_frame, nutrition)]
    return pd.concat([part for part in parts if not part.empty], ignore_index=True) if any(not part.empty for part in parts) else pd.DataFrame(columns=INTERVAL_COLUMNS)


def overlaps(intervals: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, *, ignore_event_within: tuple[pd.Timestamp, int] | None = None) -> list[str]:
    """Kinds of flagged interval that overlap ``start``..``end``.

    ``ignore_event_within`` is (the meal's time, minutes): a flagged event at the meal itself is the meal, not an extra.
    """
    if intervals.empty:
        return []
    hit = intervals[(intervals["start"] <= end) & (intervals["end"] >= start)]
    if ignore_event_within is not None:
        meal, minutes = ignore_event_within
        hit = hit[(hit["event_time"] - meal).abs() > pd.Timedelta(minutes=minutes)]
    return sorted(set(hit["kind"]))
