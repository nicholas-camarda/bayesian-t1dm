"""The personal readout's data: one row per day of glucose, insulin, food, sleep, recovery, exercise and site context.

``daily_table`` builds the day table from :class:`AnalysisData`. Every context variable is measured before the glucose
day it is paired with (the night before, or the day before), so a driver can be read as "what came before this day".
``drivers`` turns the day table into the prespecified yes/no drivers the readout model estimates. Personal baselines are
the median and spread of the previous 28 days on the same device, so a switch of watch or ring does not look like a
change in the person.

Glucose metrics follow the consensus definitions in :mod:`outcomes`; a day counts only with at least 70% of its readings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .outcomes import MIN_DAY_COVERAGE, daily_metrics

# Production values for the readout, defined once.
BASELINE_DAYS = 28
BASELINE_MIN_DAYS = 10
SHORT_SLEEP_HOURS = 6.0
WORKOUT_KCAL = 200.0  # a day counts as a workout day with at least this much workout active energy
LATE_MEAL_MINUTE = 21 * 60  # a carb entry of LATE_MEAL_GRAMS or more at or after 21:00
LATE_MEAL_GRAMS = 10.0
OLD_SITE_DAYS = 3.0  # site age at the start of the day, from the last cannula fill
RECOVERY_Z = 1.0  # resting HR this many spreads above baseline, or HRV this many below
SITE_EVENT = "cannula filled"
SLEEP_DEVICE_PRIORITY = ("Oura", "Apple Watch")  # one sleep record per night: the first device in this order that recorded it

# The prespecified drivers: (column, label shown on the page, plain definition).
DRIVERS: tuple[tuple[str, str, str], ...] = (
    ("short_sleep", "Sleep under 6 h", f"total sleep the night before was under {SHORT_SLEEP_HOURS:.0f} hours"),
    ("low_recovery", "Low recovery", f"the day before, resting heart rate was more than {RECOVERY_Z:.0f} spread above, or HRV more than {RECOVERY_Z:.0f} spread below, the previous 28 days on the same device"),
    ("workout_1d", "Workout the day before", f"at least {WORKOUT_KCAL:.0f} kcal of workout active energy the day before"),
    ("workout_2d", "Workout two days before", f"at least {WORKOUT_KCAL:.0f} kcal of workout active energy two days before"),
    ("late_meal", "Late meal", f"{LATE_MEAL_GRAMS:.0f} g of carbohydrate or more entered at or after 21:00 the evening before"),
    ("old_site", "Infusion site day 4+", f"the infusion site was {OLD_SITE_DAYS:.0f} or more days old at the start of the day (its fourth day or later)"),
)
DRIVER_COLUMNS = tuple(column for column, _, _ in DRIVERS)
DRIVER_LABELS = {column: label for column, label, _ in DRIVERS}


@dataclass(frozen=True)
class DayTable:
    days: pd.DataFrame  # one row per date, index ``date``
    site_observed_from: pd.Timestamp


def _per_day_device(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    """Daily mean of ``value`` and the device that recorded most of the day's readings."""
    frame = frame.assign(date=frame["timestamp"].dt.normalize(), device=frame["source_device"].map(_device_family))
    means = frame.groupby("date")[value].mean()
    device = frame.groupby("date")["device"].agg(lambda s: s.mode().iloc[0])
    return pd.DataFrame({value: means, "device": device})


def _device_family(name: object) -> str:
    text = str(name)
    if "Oura" in text:
        return "Oura"
    if "Watch" in text:
        return "Apple Watch"
    return "other"


def baseline_z(values: pd.Series, devices: pd.Series, *, days: int = BASELINE_DAYS, minimum: int = BASELINE_MIN_DAYS) -> pd.Series:
    """How far each day's value sits from the previous ``days`` days on the same device, in robust spreads.

    The spread is the median absolute deviation scaled to a standard deviation (x 1.4826). A day with fewer than
    ``minimum`` earlier days on its device, or a zero spread, has no z (NaN).
    """
    out = pd.Series(np.nan, index=values.index, dtype=float)
    frame = pd.DataFrame({"value": values, "device": devices}).dropna()
    for device, group in frame.groupby("device"):
        series = group["value"].asfreq("D")
        window = series.shift(1).rolling(f"{days}D", min_periods=minimum)
        median = window.median()
        spread = window.apply(lambda w: 1.4826 * np.nanmedian(np.abs(w - np.nanmedian(w))), raw=True)
        z = (series - median) / spread.replace(0.0, np.nan)
        out.loc[group.index] = z.reindex(group.index).to_numpy()
    return out


def sleep_by_night(sleep: pd.DataFrame) -> pd.DataFrame:
    """One sleep record per wake-up date, from the first device in ``SLEEP_DEVICE_PRIORITY`` that recorded it."""
    frame = sleep.assign(device=sleep["source_device"].map(_device_family), date=pd.to_datetime(sleep["date"]).dt.normalize())
    frame = frame[frame["device"].isin(SLEEP_DEVICE_PRIORITY) & frame["total_sleep_hours"].gt(0)]
    frame = frame.assign(rank=frame["device"].map({device: i for i, device in enumerate(SLEEP_DEVICE_PRIORITY)}))
    frame = frame.sort_values(["date", "rank", "total_sleep_hours"], ascending=[True, True, False]).drop_duplicates("date")
    return frame.set_index("date")[["total_sleep_hours", "sleep_start", "sleep_end", "device"]].rename(columns={"total_sleep_hours": "sleep_hours", "device": "sleep_device"})


def _basal_units_by_day(basal: pd.DataFrame) -> pd.Series:
    """Delivered basal per day: each interval's rate times its length, booked to the day it started."""
    hours = (basal["end_timestamp"] - basal["start_timestamp"]).dt.total_seconds() / 3600.0
    units = basal["basal_units_per_hour"] * hours.clip(lower=0, upper=1)  # intervals are about 5 minutes; longer gaps are not bridged
    return units.groupby(basal["start_timestamp"].dt.normalize()).sum()


def site_age_at_midnight(dates: pd.DatetimeIndex, site_events: pd.DataFrame, observed_from: pd.Timestamp) -> pd.Series:
    """Days since the last cannula fill at the start of each day; NaN before the first fill seen after ``observed_from``."""
    fills = np.sort(pd.to_datetime(site_events.loc[site_events["kind"].eq(SITE_EVENT), "timestamp"]).to_numpy())
    fills = fills[fills >= np.datetime64(observed_from)]
    out = pd.Series(np.nan, index=dates, dtype=float)
    if fills.size == 0:
        return out
    position = np.searchsorted(fills, dates.to_numpy(), side="right") - 1
    has = position >= 0
    out[has] = (dates.to_numpy()[has] - fills[position[has]]) / np.timedelta64(1, "D")
    return out


def daily_table(analysis) -> DayTable:
    """The day table for the whole record: glucose outcomes, insulin and carbs, and the context before each day."""
    merged = analysis.merged
    glucose = daily_metrics(analysis.cgm)
    days = pd.DataFrame(index=pd.DatetimeIndex(glucose.index, name="date"))
    days["readings"] = glucose["readings"].astype(int)
    days["counted"] = glucose["coverage"] >= MIN_DAY_COVERAGE
    for column in ("time_in_range_pct", "time_below_70_pct", "time_below_54_pct", "time_above_180_pct", "mean_glucose_mgdl", "cv_pct"):
        days[column] = glucose[column]
    days["in_range_readings"] = (days["time_in_range_pct"] / 100.0 * days["readings"]).round().astype(int)
    days["below_70_readings"] = (days["time_below_70_pct"] / 100.0 * days["readings"]).round().astype(int)

    tandem = merged.tandem_data
    bolus = tandem.bolus.groupby(tandem.bolus["timestamp"].dt.normalize())["bolus_units"].sum()
    carbs = tandem.carbs.groupby(tandem.carbs["timestamp"].dt.normalize())["carb_grams"].sum()
    days["bolus_units"] = bolus.reindex(days.index).fillna(0.0)
    days["basal_units"] = _basal_units_by_day(tandem.basal).reindex(days.index)
    days["insulin_units"] = days["bolus_units"] + days["basal_units"]
    days["carb_grams"] = carbs.reindex(days.index).fillna(0.0)
    late = tandem.carbs[(tandem.carbs["timestamp"].dt.hour * 60 + tandem.carbs["timestamp"].dt.minute >= LATE_MEAL_MINUTE) & tandem.carbs["carb_grams"].ge(LATE_MEAL_GRAMS)]
    late_dates = set(late["timestamp"].dt.normalize())
    days["late_meal"] = [(day - pd.Timedelta(days=1)) in late_dates for day in days.index]

    days = days.join(sleep_by_night(merged.data.sleep))
    days["short_sleep"] = days["sleep_hours"].lt(SHORT_SLEEP_HOURS).where(days["sleep_hours"].notna())

    measurements = merged.data.health_measurements
    rhr = _per_day_device(measurements[measurements["metric"].eq("resting_heart_rate")].rename(columns={"value": "resting_hr"}), "resting_hr")
    hrv = _per_day_device(measurements[measurements["metric"].eq("heart_rate_variability")].rename(columns={"value": "hrv_ms"}), "hrv_ms")
    full = pd.date_range(days.index.min(), days.index.max(), freq="D", name="date")
    rhr, hrv = rhr.reindex(full), hrv.reindex(full)
    rhr["resting_hr_z"] = baseline_z(rhr["resting_hr"], rhr["device"])
    hrv["hrv_z"] = baseline_z(hrv["hrv_ms"], hrv["device"])
    recovery = pd.DataFrame({"resting_hr": rhr["resting_hr"], "resting_hr_z": rhr["resting_hr_z"], "hrv_ms": hrv["hrv_ms"], "hrv_z": hrv["hrv_z"]})
    days = days.join(recovery)
    yesterday = recovery.shift(1, freq="D").reindex(days.index)
    observed = yesterday["resting_hr_z"].notna() | yesterday["hrv_z"].notna()
    low = yesterday["resting_hr_z"].gt(RECOVERY_Z) | yesterday["hrv_z"].lt(-RECOVERY_Z)
    days["low_recovery"] = low.where(observed)
    days["recovery_score"] = recovery_score(days["resting_hr_z"], days["hrv_z"])

    workouts = merged.data.workouts
    minutes = workouts.groupby(workouts["start_timestamp"].dt.normalize())["duration_seconds"].sum() / 60.0
    energy = workout_energy_by_day(workouts, measurements, full)
    days["workout_kcal"] = energy.reindex(days.index)
    days["workout_minutes"] = minutes.reindex(days.index).fillna(0.0).where(days["workout_kcal"].notna())
    for column, lag in (("workout_1d", 1), ("workout_2d", 2)):
        prior = energy.shift(lag).reindex(days.index)
        days[column] = prior.ge(WORKOUT_KCAL).where(prior.notna())

    observed_from = analysis.site_observed_from()
    days["site_age_days"] = site_age_at_midnight(days.index, analysis.site_events, observed_from)
    days["old_site"] = days["site_age_days"].ge(OLD_SITE_DAYS).where(days["site_age_days"].notna())

    days["excluded_day"] = days.index.isin(pd.DatetimeIndex(analysis.excluded_days))
    days["weekend"] = days.index.dayofweek >= 5
    days["insulin_vs_usual_pct"] = versus_usual(days["insulin_units"])
    days["carbs_vs_usual_pct"] = versus_usual(days["carb_grams"])
    return DayTable(days=days, site_observed_from=observed_from)


def workout_energy_by_day(workouts: pd.DataFrame, measurements: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """Workout active energy per day; 0 on a day a wearable recorded no workout, NaN when nothing was recording.

    A day counts as measured when it has a workout or any heart-rate reading (a ring or watch was worn), from the first
    workout in the record on. Without this, days the wearable was off would count as rest days.
    """
    energy = workouts.groupby(workouts["start_timestamp"].dt.normalize())["active_energy_value"].sum().reindex(dates)
    worn = dates.isin(pd.DatetimeIndex(measurements.loc[measurements["metric"].eq("heart_rate"), "timestamp"].dt.normalize().unique()))
    started = dates >= workouts["start_timestamp"].min().normalize()
    return energy.fillna(0.0).where(started & (worn | energy.notna()))


def versus_usual(values: pd.Series) -> pd.Series:
    """Percent above or below the median of the previous 28 days; NaN when that median is missing or zero."""
    usual = values.shift(1).rolling(BASELINE_DAYS, min_periods=BASELINE_MIN_DAYS).median().replace(0.0, np.nan)
    return 100.0 * (values / usual - 1.0)


def recovery_score(resting_hr_z: pd.Series, hrv_z: pd.Series) -> pd.Series:
    """A 0-100 recovery score: the normal percentile of (HRV z minus resting-HR z) / sqrt(2); 50 is a usual day.

    With only one of the two measured, that one's z is used alone. Higher HRV and lower resting heart rate than the
    person's own baseline raise it.
    """
    from scipy.stats import norm

    both = (hrv_z - resting_hr_z) / np.sqrt(2.0)
    combined = both.where(hrv_z.notna() & resting_hr_z.notna(), hrv_z.where(hrv_z.notna(), -resting_hr_z))
    return pd.Series(100.0 * norm.cdf(combined), index=combined.index).where(combined.notna())


def model_days(days: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """The days the drivers model uses, with the flow of days left out at each step (for the diagnostics workbook)."""
    flow = [{"step": "days with any CGM reading", "days": int(len(days)), "left_out": 0, "reason": ""}]
    keep = days
    for reason, mask in (
        ("fewer than 70% of the day's readings", keep["counted"]),
        ("a context-event day (listed by the person, for example illness or a steroid injection)", ~keep["excluded_day"]),
        ("before infusion-site changes are observed for the current pump", keep["site_age_days"].notna()),
        ("a driver was not measured (no sleep record, no recovery data, or before workouts were exported)", keep[list(DRIVER_COLUMNS)].notna().all(axis=1)),
    ):
        before = len(keep)
        keep = keep[mask.reindex(keep.index).fillna(False).astype(bool)]
        flow.append({"step": f"after leaving out days with {reason}", "days": int(len(keep)), "left_out": before - len(keep), "reason": reason})
    keep = keep.copy()
    for column in DRIVER_COLUMNS:
        keep[column] = keep[column].astype(bool)
    return keep, flow
