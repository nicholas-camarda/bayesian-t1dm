"""The hour-day table: one row per clock hour of every day, with the glucose counts, the settings in force and the context.

Why this grain. A time-of-day segment is a run of clock hours, so a model of one row per hour can later be rolled up to
any segment layout (including layouts that do not exist yet); the segment-day is a reporting roll-up of these rows
(``rollup_segment_days``), not the unit of analysis.

Columns, one row per (date, hour), timestamps in the local clock time the pump records:

- glucose: ``n_valid`` readings (one per 5-minute slot, at most 12), ``n_in_range`` (70-180 mg/dL), ``n_below_70``,
  ``n_below_54``, ``n_above_180`` and ``glucose_sum``;
- settings in force: ``icr``, ``cf``, ``basal`` and ``target``, each the mean over the hour's twelve 5-minute moments of the
  value the pump's active profile held then (an edit at 09:07 gives hour 9 two moments of the old value and ten of the new),
  ``segment_start`` (the segment start covering most of the hour), ``settings_known_slots`` (0-12; a moment the archive
  cannot show is not counted) and ``settings_mixed`` (the values changed inside the hour);
- context: ``carbs_g`` and ``bolus_u`` in the hour, ``carbs_prev3h_g`` (this hour and the three before it),
  ``exercise_share`` (share of moments inside an exercise interval), ``sleep_mode_share`` (share of moments the pump is in
  sleep mode, NaN where the pump's mode changes are not observed), ``apple_asleep_share`` (share of moments inside an Apple
  Health sleep period; NaN outside the exports' coverage), ``site_age_days`` (days since the last cannula fill; NaN before
  site changes are observed), ``excluded_day`` (a day marked in ``context_events.csv``) and ``dst_day``.

The settings are read with ``ProfileHistory.values_at``, the same replay the controller analysis uses, so an hour's
settings agree with every other analysis at the same moment.
"""

from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .analysis_data import AnalysisData
from .context_rules import inside_intervals
from .controller import mode_at
from .outcomes import DayPart, READING_MINUTES, _prepare
from .pump_profiles import pump_key

SLOTS_PER_HOUR = 60 // READING_MINUTES
SLEEP_MODE = 1
LOCAL_ZONE = ZoneInfo("America/New_York")
SETTING_FIELDS = {
    "icr": "carb_ratio_g_per_unit",
    "cf": "isf_mgdl_per_unit",
    "basal": "basal_units_per_hour",
    "target": "target_bg_mgdl",
}
GLUCOSE_COLUMNS = ["n_valid", "n_in_range", "n_below_70", "n_below_54", "n_above_180", "glucose_sum"]


@dataclass(frozen=True)
class HourDayFlow:
    """Row counts of a built table, for the diagnostics workbook."""

    hours: int
    hours_with_readings: int
    hours_with_all_settings: int
    hours_with_mixed_settings: int
    hours_on_excluded_days: int


def _hour_counts(cgm: pd.DataFrame) -> pd.DataFrame:
    frame = _prepare(cgm)
    frame["hour_start"] = frame["timestamp"].dt.floor("h")
    glucose = frame["glucose"]
    frame["n_in_range"] = ((glucose >= 70) & (glucose <= 180)).astype(int)
    frame["n_below_70"] = (glucose < 70).astype(int)
    frame["n_below_54"] = (glucose < 54).astype(int)
    frame["n_above_180"] = (glucose > 180).astype(int)
    frame["n_valid"] = 1
    frame["glucose_sum"] = glucose
    return frame.groupby("hour_start")[GLUCOSE_COLUMNS].sum()


def _hour_pumps(cgm: pd.DataFrame, hours: pd.DatetimeIndex) -> pd.Series:
    """The pump whose readings fill each hour (the one with most readings; the earlier serial on a tie), carried forward into empty hours."""
    frame = pd.DataFrame({"hour_start": pd.to_datetime(cgm["timestamp"]).dt.floor("h"), "pump": cgm["pump_serial"].map(pump_key)})
    frame = frame[frame["pump"].ne("")]
    counts = frame.groupby(["hour_start", "pump"]).size().rename("n").reset_index().sort_values(["hour_start", "n", "pump"], ascending=[True, False, True])
    majority = counts.drop_duplicates("hour_start").set_index("hour_start")["pump"]
    return majority.reindex(hours).ffill().bfill()


def _slot_moments(hours: pd.DatetimeIndex) -> pd.DatetimeIndex:
    offsets = pd.to_timedelta(np.arange(SLOTS_PER_HOUR) * READING_MINUTES, unit="min")
    return pd.DatetimeIndex((hours.to_numpy()[:, None] + offsets.to_numpy()[None, :]).ravel())


def _share_inside(moments: pd.DatetimeIndex, intervals: pd.DataFrame, n_hours: int) -> np.ndarray:
    inside = inside_intervals(pd.Series(moments), intervals)
    return inside.reshape(n_hours, SLOTS_PER_HOUR).mean(axis=1)


def _settings(analysis: AnalysisData, hours: pd.DatetimeIndex, pumps: pd.Series) -> pd.DataFrame:
    moments = _slot_moments(hours)
    moment_pumps = pd.Series(np.repeat(pumps.to_numpy(), SLOTS_PER_HOUR))
    out: dict[str, np.ndarray] = {}
    known = np.zeros((len(hours), SLOTS_PER_HOUR), dtype=bool)
    mixed = np.zeros(len(hours), dtype=bool)
    for name, field in {**SETTING_FIELDS, "segment_start": "start_minutes"}.items():
        values = analysis.history.values_at(moment_pumps, pd.Series(moments), field).reshape(len(hours), SLOTS_PER_HOUR)
        finite = np.isfinite(values)
        if name == "basal":
            known = finite  # a segment whose rate is unknown is unknown for every value of that segment
        counts = finite.sum(axis=1)
        with np.errstate(invalid="ignore"):
            if name == "segment_start":
                # The segment start covering most of the hour's moments.
                out[name] = np.array([np.nan if not row.any() else pd.Series(v[row]).mode().iloc[0] for v, row in zip(values, finite)])
            else:
                total = np.where(finite, values, 0.0).sum(axis=1)
                out[name] = np.where(counts > 0, total / np.maximum(counts, 1), np.nan)
                low = np.where(finite, values, np.inf).min(axis=1)
                high = np.where(finite, values, -np.inf).max(axis=1)
                mixed |= (counts > 0) & (high - low > 1e-9)
    frame = pd.DataFrame(out, index=hours)
    frame["settings_known_slots"] = known.sum(axis=1)
    frame["settings_mixed"] = mixed
    return frame


def _site_age_days(site_events: pd.DataFrame, hours: pd.DatetimeIndex) -> np.ndarray:
    fills = pd.DatetimeIndex(site_events.loc[site_events["kind"].eq("cannula filled"), "timestamp"].dropna()).sort_values()
    if fills.empty:
        return np.full(len(hours), np.nan)
    position = fills.searchsorted(hours, side="right") - 1
    age = (hours - fills[np.maximum(position, 0)]).total_seconds() / 86400
    return np.where(position >= 0, age, np.nan)


def _sleep_mode_share(analysis: AnalysisData, hours: pd.DatetimeIndex, pumps: pd.Series) -> np.ndarray:
    moments = _slot_moments(hours)
    moment_pumps = np.repeat(pumps.to_numpy(), SLOTS_PER_HOUR)
    share = np.full(len(moments), np.nan)
    for pump, (first, last) in analysis.event_span.items():
        selected = (moment_pumps == pump) & (moments >= pd.Timestamp(first)) & (moments < pd.Timestamp(last) + pd.Timedelta(days=1))
        if not selected.any():
            continue
        modes = analysis.mode_events[analysis.mode_events["pump_serial"].map(pump_key).eq(pump)].sort_values("timestamp")
        share[selected] = mode_at(pd.Series(moments[selected]), modes) == SLEEP_MODE
    return _row_mean(share.reshape(len(hours), SLOTS_PER_HOUR))


def _row_mean(values: np.ndarray) -> np.ndarray:
    """Mean of each row, NaN when the whole row is NaN (without a warning)."""
    finite = np.isfinite(values)
    counts = finite.sum(axis=1)
    return np.where(counts > 0, np.where(finite, values, 0.0).sum(axis=1) / np.maximum(counts, 1), np.nan)


def _dst_days(dates: pd.DatetimeIndex) -> np.ndarray:
    """True for the days the clock changes: the UTC offset at the start of the day differs from the end of the day."""
    return np.array([date.to_pydatetime().replace(hour=0, minute=30, tzinfo=LOCAL_ZONE).utcoffset() != date.to_pydatetime().replace(hour=23, minute=30, tzinfo=LOCAL_ZONE).utcoffset() for date in dates])


def build_hour_day_table(analysis: AnalysisData) -> pd.DataFrame:
    """One row per clock hour from the first to the last CGM day, with glucose counts, settings in force and context."""
    first = pd.to_datetime(analysis.cgm["timestamp"]).min().normalize()
    last = pd.to_datetime(analysis.cgm["timestamp"]).max().normalize()
    hours = pd.date_range(first, last + pd.Timedelta(hours=23), freq="h")
    table = pd.DataFrame(index=hours)
    table.index.name = "hour_start"
    counts = _hour_counts(analysis.cgm).reindex(hours).fillna(0).astype(int)
    for column in GLUCOSE_COLUMNS:
        table[column] = counts[column].to_numpy()
    table["date"] = hours.normalize()
    table["hour"] = hours.hour
    pumps = _hour_pumps(analysis.cgm, hours)
    table["pump_serial"] = pumps.to_numpy()
    table = table.join(_settings(analysis, hours, pumps))

    grid = analysis.grid.assign(hour_start=pd.to_datetime(analysis.grid["timestamp"]).dt.floor("h"))
    delivered = grid.groupby("hour_start")[["carb_grams", "bolus_units"]].sum(min_count=1).reindex(hours).fillna(0.0)
    table["carbs_g"] = delivered["carb_grams"].to_numpy()
    table["bolus_u"] = delivered["bolus_units"].to_numpy()
    table["carbs_prev3h_g"] = table["carbs_g"].rolling(4, min_periods=1).sum()

    table["exercise_share"] = _share_inside(_slot_moments(hours), analysis.exercise, len(hours))
    table["sleep_mode_share"] = _sleep_mode_share(analysis, hours, pumps)
    sleep = analysis.merged.data.sleep.dropna(subset=["sleep_start", "sleep_end"]).rename(columns={"sleep_start": "start", "sleep_end": "end"})[["start", "end"]]
    exports = analysis.merged.data.sleep
    asleep = _share_inside(_slot_moments(hours), sleep, len(hours)) if not sleep.empty else np.full(len(hours), np.nan)
    if not exports.empty:
        covered = (hours >= exports["covered_start_date"].min()) & (hours < exports["covered_end_date"].max() + pd.Timedelta(days=1))
        asleep = np.where(covered, asleep, np.nan)
    table["apple_asleep_share"] = asleep
    table["site_age_days"] = _site_age_days(analysis.site_events, hours)
    table["excluded_day"] = table["date"].isin(set(pd.Timestamp(day).normalize() for day in analysis.excluded_days)).to_numpy()
    dates = pd.DatetimeIndex(table["date"].unique())
    table["dst_day"] = table["date"].map(dict(zip(dates, _dst_days(dates)))).to_numpy()
    return table.reset_index()


def hour_day_flow(table: pd.DataFrame) -> HourDayFlow:
    return HourDayFlow(
        hours=int(len(table)),
        hours_with_readings=int((table["n_valid"] > 0).sum()),
        hours_with_all_settings=int((table["settings_known_slots"] == SLOTS_PER_HOUR).sum()),
        hours_with_mixed_settings=int(table["settings_mixed"].sum()),
        hours_on_excluded_days=int(table["excluded_day"].sum()),
    )


def rollup_segment_days(table: pd.DataFrame, parts: list[DayPart]) -> tuple[pd.DataFrame, int]:
    """Roll the hour rows up to one row per (part, day): the counts summed, the day being the date the part started.

    An hour belongs to the part that holds its half-past minute; a part that wraps midnight takes its after-midnight hours
    from the previous date (21:00-05:00 on the 5th includes 01:00 on the 6th). Also returns the number of hours that
    straddle a part boundary (parts starting at :30, for example), which the roll-up assigns whole.
    """
    stamps = pd.to_datetime(table["hour_start"])
    middle = stamps + pd.Timedelta(minutes=30)
    minute = (middle.dt.hour * 60 + middle.dt.minute).to_numpy()
    start_minute = (stamps.dt.hour * 60).to_numpy()
    straddles = np.zeros(len(table), dtype=bool)
    pieces: list[pd.DataFrame] = []
    for part in parts:
        inside = part.contains(minute)
        # An hour straddles when its first or last minute is in a different part than its middle.
        straddles |= inside & ~(part.contains(start_minute) & part.contains(start_minute + 59))
        selected = table.loc[inside].copy()
        selected["part"] = part.name
        selected["analysis_date"] = part.analysis_date(pd.to_datetime(selected["hour_start"]) + pd.Timedelta(minutes=30))
        pieces.append(selected)
    rolled = pd.concat(pieces, ignore_index=True).groupby(["part", "analysis_date"], as_index=False)[GLUCOSE_COLUMNS].sum()
    return rolled, int(straddles.sum())
