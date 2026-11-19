"""What Control-IQ does relative to the programmed basal, by time of day.

The pump logs, about every 5 minutes (event 279), the rate it commanded and the rate the profile calls for. The
controller adds insulin above the profile when it predicts glucose will run high and withholds it when it predicts a
low, so a controller that adds the same extra on most days at the same hours, while glucose still averages above
range, is telling us the programmed basal is too low there (and the reverse for withholding).

Per day and pump segment this module measures the net insulin the controller delivered above or below the profile,
as a fraction of the profile's own insulin, over the time that is not explained by something else: not the 4 hours
after a bolus or logged carbohydrate, not exercise (a workout of 10 minutes or more, or exercise mode) or the hours after it, not near unlogged food, and not on a
day listed in ``context_events.csv``. Sleep mode is kept: it governs the overnight hours this analysis is mostly about,
and sleep mode (Tandem's treatment range 112.5-120 mg/dL, against 112.5-160 without Sleep or Exercise, and no automatic correction boluses) is reported as its own mode column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .context_rules import exercise_intervals, inside_intervals
from .outcomes import MIN_DAY_COVERAGE, TARGET_HIGH, TARGET_LOW, VERY_LOW, DayPart
from .pump_profiles import pump_key

MODE_NAMES = {0: "standard", 1: "sleep", 2: "exercise"}
MAX_INTERVAL_MINUTES = 10  # delivery records come every ~5 minutes; longer gaps are not bridged
AFTER_MEAL_HOURS = 4          # the strict rule: this long after any carbohydrate or bolus of 0.5 U or more
MIN_MEAL_BOLUS = 0.5


def mode_at(times: pd.Series, modes: pd.DataFrame) -> np.ndarray:
    if modes.empty:
        return np.zeros(len(times), dtype=int)
    position = np.searchsorted(modes["timestamp"].to_numpy(dtype="datetime64[ns]"), times.to_numpy(dtype="datetime64[ns]"), side="right") - 1
    values = modes["mode"].to_numpy()
    return np.where(position >= 0, values[np.maximum(position, 0)], 0)


MIN_MEAL_CARBS = 15.0         # smaller entries (snacks, low treatments) do not set a meal window
MIN_MEAL_UNITS = 1.0          # smaller boluses are mostly Control-IQ's own corrections, the behaviour being measured


def meal_windows(bolus: pd.DataFrame, carbs: pd.DataFrame, *, strict: bool = False) -> pd.DataFrame:
    """When a meal still explains the controller's behaviour: each meal's time and how many hours after it to set aside.

    The default scales the window with the meal: carbohydrate of 15 g or more sets 2 hours below 20 g, 3 hours below 40 g
    and 4 hours above; a bolus of 1 U or more (with no carbohydrate entered) sets 2 hours below 2 U, 3 below 4 U and 4
    above. Smaller carbohydrate entries and the pump's automatic corrections do not set a window. ``strict`` is the
    earlier rule, kept as a sensitivity: any carbohydrate or a bolus of 0.5 U or more, 4 hours each.
    """
    carb_rows = carbs[pd.to_numeric(carbs["carb_grams"], errors="coerce") > 0][["timestamp", "carb_grams"]].copy()
    bolus_rows = bolus[pd.to_numeric(bolus["bolus_units"], errors="coerce") >= (MIN_MEAL_BOLUS if strict else MIN_MEAL_UNITS)][["timestamp", "bolus_units"]].copy()
    if strict:
        out = pd.concat([pd.DataFrame({"timestamp": pd.to_datetime(carb_rows["timestamp"]), "window_hours": float(AFTER_MEAL_HOURS)}),
                         pd.DataFrame({"timestamp": pd.to_datetime(bolus_rows["timestamp"]), "window_hours": float(AFTER_MEAL_HOURS)})], ignore_index=True)
        return out.sort_values("timestamp").reset_index(drop=True)
    carb_rows = carb_rows[carb_rows["carb_grams"] >= MIN_MEAL_CARBS]
    by_carbs = np.where(carb_rows["carb_grams"] < 20, 2.0, np.where(carb_rows["carb_grams"] < 40, 3.0, 4.0))
    by_units = np.where(bolus_rows["bolus_units"] < 2, 2.0, np.where(bolus_rows["bolus_units"] < 4, 3.0, 4.0))
    out = pd.concat([pd.DataFrame({"timestamp": pd.to_datetime(carb_rows["timestamp"]), "window_hours": by_carbs}),
                     pd.DataFrame({"timestamp": pd.to_datetime(bolus_rows["timestamp"]), "window_hours": by_units})], ignore_index=True)
    return out.sort_values("timestamp").reset_index(drop=True)


def _inside_windows(times: pd.Series, meals: pd.DataFrame) -> np.ndarray:
    """True for each time that falls inside any meal's window (a long window is not cut short by a later short one)."""
    if meals.empty:
        return np.zeros(len(times), dtype=bool)
    order = meals.sort_values("timestamp")
    starts = order["timestamp"].to_numpy(dtype="datetime64[ns]")
    ends = starts + (order["window_hours"].to_numpy() * 60).astype("timedelta64[m]")
    latest_end = np.maximum.accumulate(ends)
    values = times.to_numpy(dtype="datetime64[ns]")
    position = np.searchsorted(starts, values, side="right") - 1
    return (position >= 0) & (latest_end[np.maximum(position, 0)] >= values)


def profile_rate_at(history, pump_serials: pd.Series, times: pd.Series) -> np.ndarray:
    """Programmed basal in U/h at the exact moment of each pump record."""
    return history.values_at(pump_serials, times, "basal_units_per_hour")


def programmed_units_by_slot(analysis):
    """Integrate in-force programmed U/h over exactly the logged delivery exposure in each five-minute bin."""
    basal = analysis.merged.tandem_data.basal
    totals = []
    for pump, group in basal.groupby(basal.pump_serial.map(pump_key)):
        starts = pd.DatetimeIndex(group.start_timestamp)
        ends = pd.DatetimeIndex(group.end_timestamp)
        usable = starts.notna() & ends.notna() & (ends > starts)
        starts, ends = starts[usable], ends[usable]
        if not len(starts):
            continue
        slots = pd.date_range(starts.min().floor("5min"), ends.max().ceil("5min"), freq="5min")
        edits = analysis.history.events.loc[analysis.history.events.pump_serial.eq(pump)]
        offsets = edits.start_minutes.dropna().astype(int).unique()
        days = pd.date_range(slots.min().normalize(), slots.max().normalize())
        edges = slots.union(starts.unique()).union(ends.unique()).union(pd.DatetimeIndex(edits.timestamp.unique()))
        for minute in offsets[offsets % 5 != 0]:
            edges = edges.union(days + pd.Timedelta(minutes=int(minute)))
        edges = edges.sort_values()
        left = edges[:-1]
        multiplicity = starts.sort_values().searchsorted(left, side="right") - ends.sort_values().searchsorted(left, side="right")
        observed = multiplicity > 0
        times = pd.Series(left[observed])
        hours = (edges[1:] - left).total_seconds().to_numpy()[observed] / 3600
        rate = analysis.history.values_at(pd.Series(pump, index=times.index), times, "basal_units_per_hour")
        totals.append(pd.DataFrame({"slot": times.dt.floor("5min"), "units": rate * hours * multiplicity[observed],
                                    "unknown": ~np.isfinite(rate), "pump": pump}))
    if not totals:
        return pd.Series(dtype=float)
    groups = pd.concat(totals).groupby("slot")
    units = groups.units.sum()
    return units.mask(groups.unknown.any() | groups.pump.nunique().gt(1))


def basal_records(merged, history) -> pd.DataFrame:
    """The controller's commanded rate for the whole record, with the programmed rate of the segment in force.

    The commanded rate is the pump's basal delivery log (every ~5 minutes; the merged ``basal`` table holds it for the
    binary archive and the newer downloads alike: it equals event 279's ``commandedRate`` exactly where both exist).
    The programmed rate comes from the profile history, which covers the same years.
    """
    basal = merged.tandem_data.basal
    frame = pd.DataFrame(
        {"pump_serial": basal["pump_serial"].map(pump_key), "timestamp": pd.to_datetime(basal["start_timestamp"]).astype("datetime64[ns]"),
         "commanded_u_per_h": pd.to_numeric(basal["basal_units_per_hour"], errors="coerce")}
    ).dropna().sort_values("timestamp").reset_index(drop=True)
    frame["profile_u_per_h"] = profile_rate_at(history, frame["pump_serial"], frame["timestamp"])
    return frame


def delivery_intervals(rates: pd.DataFrame) -> pd.DataFrame:
    """Each rate record held until the next, with the extra insulin (commanded minus profile) over that interval."""
    frame = rates.sort_values("timestamp").reset_index(drop=True).copy()
    gap = frame.groupby("pump_serial")["timestamp"].shift(-1) - frame["timestamp"]
    frame["hours"] = gap.dt.total_seconds() / 3600.0
    frame = frame[(frame["hours"] > 0) & (frame["hours"] <= MAX_INTERVAL_MINUTES / 60.0)].copy()
    frame["commanded_units"] = frame["commanded_u_per_h"] * frame["hours"]
    frame["excess_units"] = (frame["commanded_u_per_h"] - frame["profile_u_per_h"]) * frame["hours"]  # NaN where the programmed rate is unknown
    frame["profile_units"] = frame["profile_u_per_h"] * frame["hours"]
    return frame


def daily_controller_pattern(
    rates: pd.DataFrame,
    modes: pd.DataFrame,
    cgm: pd.DataFrame,
    meals: pd.DataFrame,
    parts: list[DayPart],
    *,
    unlogged: pd.DataFrame | None = None,
    excluded_days=(),
    min_clean_fraction: float | None = MIN_DAY_COVERAGE,
    min_clean_hours: float = 0.0,
    workouts: pd.DataFrame | None = None,
    flow: list[dict] | None = None,
    post_meal: bool = False,
) -> pd.DataFrame:
    """One row per day and pump segment: extra insulin as a share of the profile's, and mean glucose, over the clean time.

    A day-segment counts when its clean time is at least ``min_clean_fraction`` of the segment's hours (None for no
    fraction rule) and at least ``min_clean_hours``. ``clean_hours`` lets a later pooling weight each day by its clean time.
    """
    intervals = delivery_intervals(rates).dropna(subset=["profile_units", "commanded_units"])
    if intervals.empty:
        return pd.DataFrame()
    stamp = intervals["timestamp"]
    mode = mode_at(stamp, modes)
    intervals["minute"] = stamp.dt.hour * 60 + stamp.dt.minute
    excluded = {pd.Timestamp(day).normalize() for day in excluded_days}
    keep = np.ones(len(intervals), dtype=bool)

    def audit(stage):
        if flow is None:
            return
        for part in parts:
            group = intervals.loc[keep & part.contains(intervals["minute"])]
            flow.append({"stage": stage, "part": part.name, "days": part.analysis_date(group.timestamp).nunique(),
                         "hours": group.hours.sum(), "first": group.timestamp.min(), "last": group.timestamp.max()})

    audit("Paired programmed and commanded rates")
    in_meal_window = _inside_windows(stamp, meals)
    keep &= in_meal_window if post_meal else ~in_meal_window
    audit("After meal exclusions")
    keep &= ~inside_intervals(stamp, exercise_intervals(workouts, modes))
    audit("After exercise exclusions")
    if unlogged is not None and not unlogged.empty:
        keep &= ~inside_intervals(stamp, unlogged[["start", "end"]])
    audit("After unlogged-food exclusions")
    intervals = intervals.assign(mode=mode)[keep].copy()
    glucose = cgm[["timestamp", "glucose"]].copy()
    glucose["timestamp"] = pd.to_datetime(glucose["timestamp"])
    glucose = glucose.dropna().sort_values("timestamp")
    rows = []
    for part in parts:
        inside = intervals[part.contains(intervals["minute"])].copy()
        inside["date"] = part.analysis_date(inside["timestamp"])
        inside = inside[~inside["date"].isin(excluded)]
        if flow is not None:
            flow.append({"stage": "After context-day exclusions", "part": part.name, "days": inside.date.nunique(),
                         "hours": inside.hours.sum(), "first": inside.timestamp.min(), "last": inside.timestamp.max()})
        for date, group in inside.groupby("date"):
            hours = float(group["hours"].sum())
            profile_units = float(group["profile_units"].sum())
            needed = max(part.hours * min_clean_fraction if min_clean_fraction is not None else 0.0, min_clean_hours)
            if hours < needed or profile_units <= 0:
                continue
            window = glucose[(glucose["timestamp"] >= group["timestamp"].min()) & (glucose["timestamp"] <= group["timestamp"].max() + pd.Timedelta(minutes=5))]
            clean = window[window["timestamp"].dt.floor("5min").isin(group["timestamp"].dt.floor("5min"))]["glucose"]
            rows.append(
                {
                    "date": date, "part": part.name, "clean_hours": hours, "commanded_units": float(group["commanded_units"].sum()),
                    "profile_units": profile_units, "profile_rate": round(float(profile_units / hours), 3) if profile_units > 0 else np.nan,
                    "excess_units": float(group["excess_units"].sum()), "excess_fraction": float(group["excess_units"].sum() / profile_units) if profile_units > 0 else np.nan,
                    "mean_glucose": float(clean.mean()) if len(clean) else np.nan, "time_in_range_pct": float(100 * ((clean >= TARGET_LOW) & (clean <= TARGET_HIGH)).mean()) if len(clean) else np.nan,
                    "below_70_pct": float(100 * (clean < TARGET_LOW).mean()) if len(clean) else np.nan,
                    "below_54_pct": float(100 * (clean < VERY_LOW).mean()) if len(clean) else np.nan,
                    "sleep_share": float((group["mode"] == 1).mean()),
                }
            )
    result = pd.DataFrame(rows)
    if flow is not None:
        for part in parts:
            group = result[result.part.eq(part.name)] if len(result) else result
            flow.append({"stage": "Meets daily clean-time and profile support", "part": part.name, "days": len(group),
                         "hours": group.clean_hours.sum() if len(group) else 0,
                         "first": group.date.min() if len(group) else pd.NaT, "last": group.date.max() if len(group) else pd.NaT})
    return result


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
SIGNAL_SHARE_OF_TIME = 0.50   # Y: for at least this share of the recent clean time
SIGNAL_RECENT_DAYS = 28
SIGNAL_WINDOWS = (28, 56, 90)  # a segment with too little clean time in 28 days widens to 56, then 90, and says so
SIGNAL_MIN_DAYS = 10
SIGNAL_MIN_HOURS = 14.0
MIN_CLEAN_HOURS = 1.0         # a day-segment counts toward the pooled estimate with an hour of clean time
HIGH_GLUCOSE_MGDL = 120.0     # the top of the sleep-mode target: glucose still above it despite the extra insulin
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261118


def basal_signal(daily: pd.DataFrame, current_rates: dict[str, float] | None = None, *, windows: tuple[int, ...] = SIGNAL_WINDOWS, draws: int = BOOTSTRAP_DRAWS) -> pd.DataFrame:
    """Per pump segment: does the controller consistently work against the programmed basal at its current rate?

    Pooled by hours, over the last 28 calendar days (or the shortest of ``windows`` that holds enough clean time, reported
    as ``window_days``), whatever basal rate the segment held then. What the
    controller commanded is an absolute rate (U/h) that does not depend on which profile or basal edit was running, so
    each day's commanded rate is compared with the segment's *current* programmed rate: a change of profile or an edit
    does not throw away the earlier days. Each day is weighted by its clean hours, so a day with two clean hours counts in
    proportion and daytime segments, where meals leave little clean time, can still be assessed. The share of clean time
    with the controller commanding more than ``SIGNAL_EXTRA_FRACTION`` above (or below) the current rate, the pooled
    extra insulin and its interval come from resampling days. ``prob_rule_met`` is the share of resamples in which the
    controller ran that far above the current rate for at least ``SIGNAL_SHARE_OF_TIME`` of the clean time.

    "Basal likely too low": that share is at least ``SIGNAL_SHARE_OF_TIME``, mean glucose over the clean time is above
    ``HIGH_GLUCOSE_MGDL``, and time below 70 and below 54 are within their limits. "Basal likely too high": the mirror
    case. Fewer than ``SIGNAL_MIN_DAYS`` days or ``SIGNAL_MIN_HOURS`` clean hours gives no signal.
    """
    from .outcomes import LOW_LIMIT_PERCENT, VERY_LOW_LIMIT_PERCENT

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for part, group in daily.groupby("part", sort=False):
        group = group.sort_values("date")
        rate = round(float((current_rates or {}).get(part, group["profile_rate"].dropna().iloc[-1] if group["profile_rate"].notna().any() else np.nan)), 3)
        if not np.isfinite(rate) or rate <= 0:
            rows.append({"part": part, "signal": "the programmed rate for this segment is not known"})
            continue
        for window in windows:
            recent = group[group["date"] > group["date"].max() - pd.Timedelta(days=window)]
            n, hours = len(recent), float(recent["clean_hours"].sum())
            if n >= SIGNAL_MIN_DAYS and hours >= SIGNAL_MIN_HOURS:
                break
        row = {"part": part, "profile_rate": rate, "window_days": window, "days": n, "clean_hours": hours}
        if n < SIGNAL_MIN_DAYS or hours < SIGNAL_MIN_HOURS:
            row.update({"signal": f"not enough clean time ({n} days, {hours:.0f} h in the last {window} days; need {SIGNAL_MIN_DAYS} days and {SIGNAL_MIN_HOURS:.0f} h)"})
            rows.append(row)
            continue
        w = recent["clean_hours"].to_numpy()
        commanded = recent["commanded_units"].to_numpy()
        frac = commanded / (rate * w) - 1.0  # each day's commanded rate against the current programmed rate
        over = float(w[frac > SIGNAL_EXTRA_FRACTION].sum() / w.sum())
        under = float(w[frac < -SIGNAL_EXTRA_FRACTION].sum() / w.sum())
        measured = recent["mean_glucose"].notna().to_numpy()  # a day with no clean CGM says nothing about glucose or lows
        glucose = float(np.average(recent["mean_glucose"].to_numpy()[measured], weights=w[measured])) if measured.any() else float("nan")
        below_70 = float(np.average(recent["below_70_pct"].to_numpy()[measured], weights=w[measured])) if measured.any() else float("nan")
        below_54 = float(np.average(recent["below_54_pct"].to_numpy()[measured], weights=w[measured])) if measured.any() else float("nan")
        pick = rng.integers(0, n, size=(draws, n))
        boot_over = np.array([w[i][frac[i] > SIGNAL_EXTRA_FRACTION].sum() / w[i].sum() for i in pick])
        boot_excess = np.array([commanded[i].sum() / (rate * w[i].sum()) - 1.0 for i in pick])
        lows_ok = below_70 < LOW_LIMIT_PERCENT and below_54 < VERY_LOW_LIMIT_PERCENT
        if not measured.any():
            signal = "no clean glucose in the window: glucose and lows are unknown"
        elif over >= SIGNAL_SHARE_OF_TIME and glucose > HIGH_GLUCOSE_MGDL and lows_ok:
            signal = "basal likely too low"
        elif over >= SIGNAL_SHARE_OF_TIME and not lows_ok:
            signal = "controller adds insulin but lows are over a limit: no added basal"
        elif over >= SIGNAL_SHARE_OF_TIME:
            signal = "controller adds insulin and glucose is within the target: basal is being covered by the controller"
        elif under >= SIGNAL_SHARE_OF_TIME:
            signal = "basal likely too high"
        else:
            signal = "no consistent pattern"
        row.update(
            {"share_over_x": over, "share_over_lo": float(np.percentile(boot_over, 5)), "share_over_hi": float(np.percentile(boot_over, 95)), "share_under_minus_x": under,
             "prob_rule_met": float((boot_over >= SIGNAL_SHARE_OF_TIME).mean()), "pooled_excess": float(commanded.sum() / (rate * w.sum()) - 1.0),
             "excess_lo": float(np.percentile(boot_excess, 5)), "excess_hi": float(np.percentile(boot_excess, 95)), "mean_glucose": glucose,
             "below_70_pct": below_70, "below_54_pct": below_54, "days_with_glucose": int(measured.sum()), "signal": signal}
        )
        rows.append(row)
    return pd.DataFrame(rows)


def controller_inputs(analysis, parts: list[DayPart], *, strict: bool = False, rates: pd.DataFrame | None = None, flow: list[dict] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The daily controller pattern with its meal, unlogged-food and exercise exclusions, from the shared ``AnalysisData``.

    Returns the daily table, the rate records it was built from (empty when the archive holds no rate records) and the
    unlogged-food intervals found along the way.
    """
    merged = analysis.merged
    if rates is None:
        rates = basal_records(merged, analysis.history)
    if rates.empty:
        return pd.DataFrame(), rates, pd.DataFrame()
    cgm, bolus, carbs = analysis.cgm, merged.tandem_data.bolus, merged.tandem_data.carbs
    daily = daily_controller_pattern(rates, analysis.mode_events, cgm, meal_windows(bolus, carbs, strict=strict), parts, unlogged=analysis.unlogged, excluded_days=analysis.excluded_days,
                                     min_clean_fraction=None, min_clean_hours=MIN_CLEAN_HOURS, workouts=merged.data.workouts, flow=flow)
    return daily, rates, analysis.unlogged
