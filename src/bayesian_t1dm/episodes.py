"""Problem episodes: long highs and lows, what surrounded each, the most likely pattern, and whether it repeats.

An episode is a stretch of CGM readings above 250 mg/dL for at least 90 minutes (high) or below 70 mg/dL for at least
15 minutes (low). Each is described by what the pump, the food log, the workout log and the controller recorded around
it, then given the most likely *pattern* by transparent rules, first match wins, with the evidence that triggered the
rule. A pattern is not a diagnosis: it says which explanation the recorded data support, and where a pattern repeats at
the same time of day it points at the setting that governs that time (an I:C for a meal pattern, basal otherwise).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HIGH_MGDL = 250.0
HIGH_MINUTES = 90
LOW_MGDL = 70.0
LOW_MINUTES = 15
MEAL_LOOKBACK_MINUTES = 120
UNDER_DOSE_RATIO = 1.15       # carbs / I:C needed against the units given: the dose was this much short
OVER_DOSE_RATIO = 0.85        # ... or this much over
SITE_WINDOW_BEFORE_MINUTES = 30
SITE_WINDOW_AFTER_MINUTES = 60
MIN_SITE_EPISODE_HOURS = 1.5
LATE_DROP_MGDL = 120.0
LATE_DROP_MINUTES = 90
STACKED_IOB_UNITS = 2.0
EXERCISE_WINDOW_MINUTES = 12 * 60
MIN_WORKOUT_MINUTES = 10      # a workout counts as exercise from this length, walks included
BOLUS_LOOKBACK_MINUTES = 240
CORRECTION_LOOKBACK_MINUTES = 180
REPEAT_MIN_EPISODES = 4
REPEAT_MIN_SHARE = 0.5
SITE_EVENT_CODES = {33: "cartridge filled", 61: "cannula filled", 63: "tubing filled"}

HIGH_PATTERNS = {
    "site_or_insulin": "Possible infusion-site or insulin problem",
    "meal_underdosed": "Meal dosed short of the pump's own ratio",
    "meal_dosed_high": "Meal dosed as calculated, glucose still rose",
    "unlogged_food": "Rise with no logged cause",
    "late_insulin": "Stacked or late insulin action",
    "persistent": "Persistent high with no meal before it",
    "unexplained": "No pattern the data can name",
}
LOW_PATTERNS = {
    "after_exercise": "Low after a recent workout",
    "over_correction": "Low after corrections without carbohydrate",
    "meal_overdosed": "Meal dosed above the pump's own ratio",
    "overnight": "Overnight low with no bolus before it",
    "unexplained": "No pattern the data can name",
}
# What each pattern points at when it repeats at one time of day, and what to check meanwhile.
PLAYBOOK = {
    "site_or_insulin": ("a site or insulin problem, not a setting", "Look at the site and the insulin. If sugar stays high despite corrections, change the site; check ketones per your care plan."),
    "meal_underdosed": ("the I:C of that segment, or carb counting", "If it repeats at this time of day, the segment's I:C is probably too weak; if it is one-off, check the carb count."),
    "meal_dosed_high": ("the I:C of that segment, or meal timing and composition", "Look at pre-bolus timing and fat/protein; if it repeats at one time of day, the I:C is probably too weak there."),
    "unlogged_food": ("logging, not a setting", "Log the food or treatment (including treating a low); unlogged carbohydrate misleads every setting estimate."),
    "late_insulin": ("correction stacking", "Avoid stacking corrections while insulin is still active; wait for the pump's insulin on board to clear before correcting again."),
    "persistent": ("basal for that time of day", "If it repeats at one time of day with the controller adding insulin, the programmed basal there is probably too low."),
    "after_exercise": ("exercise sensitivity", "Your data show how often lows follow meals after a workout (below); consider a smaller bolus for meals within hours of exercise."),
    "over_correction": ("correction factor or stacking", "Corrections without carbohydrate preceded this low; check the correction factor and avoid repeated corrections."),
    "meal_overdosed": ("the I:C of that segment, or carb counting", "If it repeats at this time of day, the segment's I:C may be too strong; if it is one-off, check the carb count."),
    "overnight": ("basal for the overnight hours", "A low with no bolus before it points at basal or an earlier evening dose."),
    "unexplained": ("nothing specific", "No pattern in the recorded data."),
}


def load_site_events(tandem_root: str | Path) -> pd.DataFrame:
    """Cartridge, cannula and tubing fills from the archived pump-logs JSON (available from the 3.x downloads on)."""
    rows = []
    for path in sorted(Path(tandem_root).glob("*/raw/pump_logs_*.json")):
        if path.name.endswith(".meta.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for chunk in payload["chunks"]:
            for event in chunk["response"].get("events") or []:
                if event["eventCode"] in SITE_EVENT_CODES:
                    rows.append({"timestamp": pd.Timestamp(event["pumpDateTime"]), "kind": SITE_EVENT_CODES[event["eventCode"]]})
    frame = pd.DataFrame(rows, columns=["timestamp", "kind"]).drop_duplicates().sort_values("timestamp").reset_index(drop=True)
    return frame


def minutes_since_workout(times: pd.Series, workouts: pd.DataFrame | None, *, minimum_minutes: int = MIN_WORKOUT_MINUTES) -> np.ndarray:
    """Minutes from the end of the last qualifying workout to each time (NaN when none came before)."""
    if workouts is None or workouts.empty:
        return np.full(len(times), np.nan)
    kept = workouts[pd.to_numeric(workouts["duration_seconds"], errors="coerce") >= minimum_minutes * 60]
    ends = np.sort(pd.to_datetime(kept["end_timestamp"]).to_numpy(dtype="datetime64[ns]"))
    if len(ends) == 0:
        return np.full(len(times), np.nan)
    values = pd.to_datetime(times).to_numpy(dtype="datetime64[ns]")
    position = np.searchsorted(ends, values, side="right") - 1
    gap = (values - ends[np.maximum(position, 0)]) / np.timedelta64(1, "m")
    return np.where(position >= 0, gap, np.nan)


def _runs(glucose: pd.Series, mask: pd.Series, minutes: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    mask = mask.fillna(False)
    group = (mask != mask.shift()).cumsum()
    return [(g.index[0], g.index[-1]) for _, g in mask[mask].groupby(group[mask]) if (len(g) - 1) * 5 + 5 >= minutes]


def find_episodes(cgm: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """High and low episodes with their peak or nadir, one row each, between ``start`` and ``end`` (exclusive)."""
    frame = cgm[["timestamp", "glucose"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.floor("5min")
    series = frame.dropna().drop_duplicates("timestamp").set_index("timestamp")["glucose"].sort_index().asfreq("5min")
    series = series[(series.index >= start) & (series.index < end)]
    rows = []
    for kind, mask, minutes in (("high", series > HIGH_MGDL, HIGH_MINUTES), ("low", series < LOW_MGDL, LOW_MINUTES)):
        for first, last in _runs(series, mask, minutes):
            span = series[first:last]
            rows.append(
                {"kind": kind, "start": first, "end": last, "hours": ((last - first).total_seconds() + 300) / 3600.0,
                 "extreme": float(span.max() if kind == "high" else span.min())}
            )
    return pd.DataFrame(rows, columns=["kind", "start", "end", "hours", "extreme"]).sort_values("start").reset_index(drop=True)


def _before(frame: pd.DataFrame, when: pd.Timestamp, minutes: int, column: str) -> pd.Series:
    window = frame[(frame["timestamp"] >= when - pd.Timedelta(minutes=minutes)) & (frame["timestamp"] <= when)]
    return pd.to_numeric(window[column], errors="coerce")


def describe_episodes(
    episodes: pd.DataFrame,
    grid: pd.DataFrame,
    site_events: pd.DataFrame,
    unlogged: pd.DataFrame | None = None,
    delivery: pd.DataFrame | None = None,
    workouts: pd.DataFrame | None = None,
    site_coverage_from: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Add what surrounded each episode. ``site_coverage_from`` is when site-change events begin to be recorded
    (episodes before it are not judged on sites); by default the first event.

    ``grid`` is the 5-minute dataset (``timestamp``, ``glucose``, ``iob_units``, ``bolus_units``, ``carb_grams``,
    ``carb_ratio_g_per_unit``, ``minutes_since_last_workout``). ``delivery`` is the controller's delivery intervals
    (``timestamp``, ``excess_units``) for the extra insulin it added during a high.
    """
    grid = grid.copy()
    grid["timestamp"] = pd.to_datetime(grid["timestamp"])
    grid = grid.sort_values("timestamp")
    sites = site_events["timestamp"].to_numpy(dtype="datetime64[ns]") if not site_events.empty else np.array([], dtype="datetime64[ns]")
    site_from = site_coverage_from if site_coverage_from is not None else (site_events["timestamp"].min() if not site_events.empty else None)
    rows = []
    for ep in episodes.itertuples(index=False):
        s, e = ep.start, ep.end
        carbs = _before(grid, s + pd.Timedelta(minutes=30), MEAL_LOOKBACK_MINUTES + 30, "carb_grams").fillna(0.0)
        meal_carbs = float(carbs.sum())
        window_bolus = _before(grid, s + pd.Timedelta(minutes=30), MEAL_LOOKBACK_MINUTES + 30, "bolus_units").fillna(0.0)
        meal_units = float(window_bolus.sum())
        ratio_series = _before(grid, s + pd.Timedelta(minutes=30), MEAL_LOOKBACK_MINUTES + 30, "carb_ratio_g_per_unit")
        icr = float(ratio_series.dropna().iloc[-1]) if ratio_series.notna().any() else float("nan")
        # How the dose compares with what the pump's own ratio calls for: needed units / given units.
        needed = meal_carbs / icr if meal_carbs > 0 and np.isfinite(icr) and icr > 0 else float("nan")
        dose_ratio = needed / meal_units if np.isfinite(needed) and meal_units > 0 else (float("inf") if np.isfinite(needed) else float("nan"))
        during = grid[(grid["timestamp"] >= s) & (grid["timestamp"] <= e)]
        corrections = pd.to_numeric(during["bolus_units"], errors="coerce").fillna(0.0)
        correction_units = float(corrections.sum())
        extra_basal = float("nan")
        if delivery is not None and not delivery.empty:
            inside = delivery[(delivery["timestamp"] >= s) & (delivery["timestamp"] <= e)]
            extra_basal = float(inside["excess_units"].sum()) if len(inside) else float("nan")
        after = grid[(grid["timestamp"] > e) & (grid["timestamp"] <= e + pd.Timedelta(minutes=LATE_DROP_MINUTES))]
        after_glucose = pd.to_numeric(after["glucose"], errors="coerce")
        drop = float(ep.extreme - after_glucose.min()) if ep.kind == "high" and len(after_glucose.dropna()) else float("nan")
        iob_end = _before(grid, e, 5, "iob_units")
        near_site = None
        if len(sites) and site_from is not None and s >= site_from:
            gaps = (sites - np.datetime64(e)) / np.timedelta64(1, "m")
            starts_gap = (sites - np.datetime64(s)) / np.timedelta64(1, "m")
            hit = (starts_gap >= SITE_WINDOW_BEFORE_MINUTES) & (gaps <= SITE_WINDOW_AFTER_MINUTES)
            near_site = pd.Timestamp(sites[hit][0]) if hit.any() else None
        since = minutes_since_workout(pd.Series([s]), workouts)[0] if workouts is not None else float("nan")
        prior = grid[(grid["timestamp"] >= s - pd.Timedelta(minutes=BOLUS_LOOKBACK_MINUTES)) & (grid["timestamp"] < s)]
        prior_bolus = pd.to_numeric(prior["bolus_units"], errors="coerce").fillna(0.0)
        prior_carbs = pd.to_numeric(prior["carb_grams"], errors="coerce").fillna(0.0)
        recent = grid[(grid["timestamp"] >= s - pd.Timedelta(minutes=CORRECTION_LOOKBACK_MINUTES)) & (grid["timestamp"] < s)]
        recent_bolus = pd.to_numeric(recent["bolus_units"], errors="coerce").fillna(0.0)
        recent_carbs = pd.to_numeric(recent["carb_grams"], errors="coerce").fillna(0.0)
        corrections_no_carbs = float(recent_bolus[recent_bolus > 0].sum()) if recent_carbs.sum() == 0 else 0.0
        flagged = bool(unlogged is not None and not unlogged.empty and ((unlogged["start"] <= s) & (unlogged["end"] >= s - pd.Timedelta(minutes=60))).any())
        rows.append(
            {
                "meal_carbs": meal_carbs, "meal_units": meal_units, "icr": icr, "dose_ratio": dose_ratio, "correction_units": correction_units,
                "corrections": int((corrections > 0).sum()), "extra_basal_units": extra_basal, "drop_after": drop,
                "iob_at_end": float(iob_end.dropna().iloc[-1]) if iob_end.notna().any() else float("nan"),
                "site_change": near_site, "minutes_since_workout": float(since),
                "bolus_prior_4h": float(prior_bolus.sum()), "carbs_prior_4h": float(prior_carbs.sum()), "corrections_no_carbs_3h": corrections_no_carbs,
                "unlogged_flag": flagged,
            }
        )
    return pd.concat([episodes.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def classify(row: pd.Series) -> tuple[str, str]:
    """(pattern, the evidence that triggered it) for one described episode. The first rule that matches wins."""
    if row["kind"] == "high":
        meal = row["meal_carbs"] > 0
        if pd.notna(row["site_change"]) and row["hours"] >= MIN_SITE_EPISODE_HOURS and (row["corrections"] > 0 or (pd.notna(row["extra_basal_units"]) and row["extra_basal_units"] > 0)):
            gap = (pd.Timestamp(row["site_change"]) - row["end"]).total_seconds() / 60
            return "site_or_insulin", f"high for {row['hours']:.1f} h despite {row['correction_units']:.1f} U of corrections" + (f" and {row['extra_basal_units']:.1f} U of extra basal" if pd.notna(row["extra_basal_units"]) else "") + f"; the site was changed {abs(gap):.0f} min {'after' if gap >= 0 else 'before'} it ended"
        if meal and (row["meal_units"] == 0 or (pd.notna(row["dose_ratio"]) and row["dose_ratio"] >= UNDER_DOSE_RATIO)):
            return "meal_underdosed", f"{row['meal_carbs']:.0f} g logged, {row['meal_units']:.1f} U given; the pump's own ratio ({row['icr']:.1f} g/U) calls for {row['meal_carbs'] / row['icr']:.1f} U" if pd.notna(row["icr"]) and row["icr"] > 0 else f"{row['meal_carbs']:.0f} g logged and {row['meal_units']:.1f} U given"
        if meal:
            return "meal_dosed_high", f"{row['meal_carbs']:.0f} g logged and {row['meal_units']:.1f} U given, in line with the pump's ratio, yet glucose stayed above 250 for {row['hours']:.1f} h"
        if row["unlogged_flag"]:
            return "unlogged_food", "a rise with no pump carbs, bolus or food-log entry before it"
        if pd.notna(row["drop_after"]) and row["drop_after"] >= LATE_DROP_MGDL and pd.notna(row["iob_at_end"]) and row["iob_at_end"] >= STACKED_IOB_UNITS:
            return "late_insulin", f"glucose fell {row['drop_after']:.0f} mg/dL within {LATE_DROP_MINUTES} min of the end with {row['iob_at_end']:.1f} U still active"
        if row["hours"] >= 3:
            return "persistent", f"high for {row['hours']:.1f} h with no meal in the 2 hours before it" + (
                f"; the controller added {row['extra_basal_units']:.1f} U above the profile and {row['correction_units']:.1f} U of corrections were given" if pd.notna(row["extra_basal_units"]) else f"; {row['correction_units']:.1f} U of corrections were given")
        return "unexplained", "nothing in the record before it explains it"
    recent_workout = pd.notna(row["minutes_since_workout"]) and row["minutes_since_workout"] <= EXERCISE_WINDOW_MINUTES
    if recent_workout and row["bolus_prior_4h"] > 0:
        return "after_exercise", f"a workout ended {row['minutes_since_workout'] / 60:.1f} h earlier and {row['bolus_prior_4h']:.1f} U was given in the 4 hours before"
    if row["corrections_no_carbs_3h"] >= 1.0:
        return "over_correction", f"{row['corrections_no_carbs_3h']:.1f} U was given as corrections, with no carbohydrate, in the 3 hours before"
    if row["carbs_prior_4h"] > 0 and pd.notna(row["icr"]) and row["icr"] > 0 and row["bolus_prior_4h"] > 0 and (row["carbs_prior_4h"] / row["icr"]) / row["bolus_prior_4h"] <= OVER_DOSE_RATIO:
        return "meal_overdosed", f"{row['bolus_prior_4h']:.1f} U was given for {row['carbs_prior_4h']:.0f} g; the pump's own ratio ({row['icr']:.1f} g/U) calls for {row['carbs_prior_4h'] / row['icr']:.1f} U"
    if row["bolus_prior_4h"] == 0 and 0 <= row["start"].hour < 6:
        return "overnight", "no bolus in the 4 hours before it, between midnight and 06:00"
    return "unexplained", "nothing in the record before it explains it"


def classify_all(described: pd.DataFrame) -> pd.DataFrame:
    result = described.copy()
    if result.empty:
        return result.assign(pattern=pd.Series(dtype=str), evidence=pd.Series(dtype=str))
    labelled = result.apply(classify, axis=1, result_type="expand")
    result["pattern"], result["evidence"] = labelled[0], labelled[1]
    return result


def repeats(classified: pd.DataFrame, parts, *, minimum: int = REPEAT_MIN_EPISODES, share: float = REPEAT_MIN_SHARE) -> list[dict]:
    """Patterns that recur in one pump segment: at least ``minimum`` episodes of one kind there, and that pattern at least
    ``share`` of them (excluding ones the data cannot name). Returns the segment, the pattern, counts and what it points at."""
    out = []
    if classified.empty:
        return out
    minute = classified["start"].dt.hour * 60 + classified["start"].dt.minute
    for part in parts:
        inside = classified[part.contains(minute)]
        for kind, names in (("high", HIGH_PATTERNS), ("low", LOW_PATTERNS)):
            subset = inside[inside["kind"].eq(kind)]
            if len(subset) < minimum:
                continue
            counts = subset["pattern"].value_counts()
            counts = counts[counts.index != "unexplained"]
            if counts.empty:
                continue
            pattern, n = counts.index[0], int(counts.iloc[0])
            if n / len(subset) >= share and n >= minimum:
                points_at, advice = PLAYBOOK[pattern]
                out.append({"segment": part.name, "kind": kind, "pattern": pattern, "label": names[pattern], "n": n, "of": int(len(subset)), "points_at": points_at, "advice": advice})
    return out


def wilson(successes: int, n: int, z: float = 1.645) -> tuple[float, float]:
    """Wilson 90% interval for a proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def exercise_effect(grid: pd.DataFrame, workouts: pd.DataFrame, *, min_bolus: float = 1.5) -> dict:
    """How often a meal bolus is followed by a low within 4 hours, for meals within 12 hours after a workout of at least
    10 minutes and for the others."""
    frame = grid.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    meals = frame[(pd.to_numeric(frame["bolus_units"], errors="coerce").fillna(0) >= min_bolus) & (pd.to_numeric(frame["carb_grams"], errors="coerce").fillna(0) > 0)]
    if meals.empty:
        return {}
    since = minutes_since_workout(meals["timestamp"], workouts)
    glucose = frame.set_index("timestamp")["glucose"]
    after_workout, other = [0, 0], [0, 0]  # [meals followed by a low, meals]
    for time, gap in zip(meals["timestamp"], since):
        window = glucose[time : time + pd.Timedelta(hours=4)]
        if len(window.dropna()) < 36:  # under 3 of 4 hours of readings
            continue
        group = after_workout if np.isfinite(gap) and gap <= EXERCISE_WINDOW_MINUTES else other
        group[0] += int((window < LOW_MGDL).any())
        group[1] += 1
    return {
        "after_workout_meals": after_workout[1], "after_workout_lows": after_workout[0], "after_workout_share": after_workout[0] / after_workout[1] if after_workout[1] else float("nan"),
        "after_workout_interval": wilson(after_workout[0], after_workout[1]),
        "other_meals": other[1], "other_lows": other[0], "other_share": other[0] / other[1] if other[1] else float("nan"), "other_interval": wilson(other[0], other[1]),
    }


def site_change_link(classified: pd.DataFrame, site_events: pd.DataFrame, *, minimum_hours: float = 3.0) -> dict:
    """Of the long highs, how many ended within an hour after a site change, against how often that would happen by chance."""
    if site_events.empty or classified.empty:
        return {}
    first = site_events["timestamp"].min()
    highs = classified[classified["kind"].eq("high") & (classified["start"] >= first) & (classified["hours"] >= minimum_hours)]
    linked = int(highs["site_change"].notna().sum())
    span_hours = max(1.0, (classified["end"].max() - first).total_seconds() / 3600)
    changes = int(site_events[site_events["kind"].eq("cannula filled")].shape[0])
    chance = min(1.0, changes * (SITE_WINDOW_AFTER_MINUTES / 60.0 + minimum_hours) / span_hours)
    return {"long_highs": int(len(highs)), "linked": linked, "site_changes": changes, "chance_share": float(chance)}
