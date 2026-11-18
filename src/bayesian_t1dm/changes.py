"""Setting changes detected from the pump's own profile history, and the before/after evaluation of one change.

A change is a segment value (I:C, CF, basal, target or start time) that differs from what the same segment held
before; re-logged unchanged segments are not changes. The evaluation compares the 14 days before a change with the 14
days after it, over the hours the setting governs plus 4 hours, as written into the proposal when it was made.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .outcomes import DayPart, passes_low_limits, pooled_metrics
from .pump_profiles import PROFILE_COPIED, PROFILE_CREATED, PROFILE_DELETED, PROFILE_EVENT, SEGMENT_DELETE, SEGMENT_EVENT

SETTING_FIELDS = {
    "carb_ratio_g_per_unit": "I:C (g/U)",
    "isf_mgdl_per_unit": "correction factor (mg/dL/U)",
    "basal_units_per_hour": "basal (U/h)",
    "target_bg_mgdl": "target (mg/dL)",
    "start_minutes": "segment start (min)",
}
EVALUATION_DAYS = 14
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 20261118


def detect_setting_changes(events: pd.DataFrame) -> pd.DataFrame:
    """One row per changed value: when, pump, profile, segment, setting, old and new value.

    Replays the segment edits per profile; a record that re-logs the values a segment already holds changes nothing.
    A change to a segment the replay has not seen (a profile that predates the archive) is reported without an old
    value only when the segment slot was seen before, so the first sight of a profile is never a change.
    """
    columns = ["timestamp", "pump_serial", "profile_id", "segment_index", "segment_start_minutes", "setting", "old_value", "new_value"]
    tables: dict[tuple[str, int], list[dict[str, float] | None]] = {}
    changes: list[dict] = []
    actions = events[events["event"].isin((SEGMENT_EVENT, PROFILE_EVENT))].sort_values(["pump_serial", "sequence"])
    for row in actions.itertuples(index=False):
        key = (row.pump_serial, int(row.profile_id))
        if int(row.event) == PROFILE_EVENT:
            status = int(row.status)
            if status == PROFILE_CREATED:
                tables[key] = []
            elif status == PROFILE_COPIED:
                tables[key] = [None if seg is None else dict(seg) for seg in tables.get((row.pump_serial, int(row.source_profile_id)), [])]
            elif status == PROFILE_DELETED:
                tables.pop(key, None)
            continue
        table = tables.setdefault(key, [])
        index = int(row.segment_index)
        if int(row.modification) == SEGMENT_DELETE:
            if index < len(table):
                table.pop(index)
            continue
        values = {field: float(getattr(row, field)) for field in SETTING_FIELDS}
        table.extend([None] * (index + 1 - len(table)))
        previous = table[index]
        if previous is not None and int(row.status) == 15:
            for field in SETTING_FIELDS:
                if field in values and field in previous and not np.isclose(values[field], previous[field]):
                    changes.append(
                        {"timestamp": row.timestamp, "pump_serial": row.pump_serial, "profile_id": int(row.profile_id), "segment_index": index,
                         "segment_start_minutes": values["start_minutes"], "setting": field, "old_value": previous[field], "new_value": values[field]}
                    )
        table[index] = values if int(row.status) == 15 else None
    return pd.DataFrame(changes, columns=columns)


def _bootstrap_difference(before: np.ndarray, after: np.ndarray, *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    """Mean difference (after minus before) of daily values with a day-level percentile 90% interval."""
    rng = np.random.default_rng(seed)
    differences = np.array(
        [rng.choice(after, size=len(after)).mean() - rng.choice(before, size=len(before)).mean() for _ in range(draws)]
    )
    return float(after.mean() - before.mean()), float(np.percentile(differences, 5)), float(np.percentile(differences, 95))


def evaluate_change(
    cgm: pd.DataFrame,
    changed_at: pd.Timestamp,
    part: DayPart,
    *,
    now: pd.Timestamp,
    excluded_days: Iterable[pd.Timestamp] = (),
    days: int = EVALUATION_DAYS,
) -> dict[str, object]:
    """Compare the ``days`` days before the change's day with the ``days`` days after it, over ``part``.

    Days from the change's own day are left out of both sides (the change happened inside that day). The result says
    ``pending`` until the full after-window has been observed. Associational: one person, before versus after.
    """
    excluded = {pd.Timestamp(day).normalize() for day in excluded_days}
    change_day = pd.Timestamp(changed_at).normalize()
    before_start, after_start = change_day - pd.Timedelta(days=days), change_day + pd.Timedelta(days=1)
    after_end = after_start + pd.Timedelta(days=days)
    available_after = int(max(0, min(days, (pd.Timestamp(now).normalize() - after_start) / pd.Timedelta(days=1))))
    result: dict[str, object] = {"change_day": change_day.date(), "hours": part.name, "after_days_observed": available_after, "excluded_days": len(excluded)}
    if available_after < days:
        result["status"] = "pending"
        result["reason"] = f"{available_after} of {days} days after the change observed"
        return result

    from .outcomes import READING_MINUTES, MIN_DAY_COVERAGE, _prepare

    frame = _prepare(cgm)
    frame = frame[part.contains(frame["minute"])]
    expected = part.hours * 60 / READING_MINUTES

    def daily(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        window = frame[(frame["date"] >= start) & (frame["date"] < end) & ~frame["date"].isin(excluded)]
        counts = window.groupby("date")["glucose"].count()
        kept = window[window["date"].isin(counts[counts / expected >= MIN_DAY_COVERAGE].index)]
        grouped = kept.groupby("date")["glucose"]
        return pd.DataFrame({"tir": grouped.apply(lambda g: 100 * ((g >= 70) & (g <= 180)).mean()), "below70": grouped.apply(lambda g: 100 * (g < 70).mean())})

    before, after = daily(before_start, change_day), daily(after_start, after_end)
    if len(before) < days // 2 or len(after) < days // 2:
        result["status"] = "insufficient"
        result["reason"] = f"{len(before)} days before and {len(after)} days after have enough readings; at least {days // 2} each are needed"
        return result
    difference, lower, upper = _bootstrap_difference(before["tir"].to_numpy(), after["tir"].to_numpy())
    after_pooled = pooled_metrics(cgm, start=after_start, end=after_end, part=part)
    result.update(
        status="evaluated",
        days_before=len(before), days_after=len(after),
        tir_before=float(before["tir"].mean()), tir_after=float(after["tir"].mean()),
        tir_difference=difference, tir_difference_lower_90=lower, tir_difference_upper_90=upper,
        below_70_after=float(after_pooled["time_below_70_pct"]), below_54_after=float(after_pooled["time_below_54_pct"]),
        low_limits_after="within limits" if passes_low_limits(after_pooled) else "over a limit",
        verdict=("time in range rose" if lower > 0 else "time in range fell" if upper < 0 else "no clear change in time in range"),
    )
    return result
