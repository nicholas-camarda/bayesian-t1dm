"""The one place the analyses get their data from the merged dataset.

``load_analysis_data`` reads the merged dataset once and returns an :class:`AnalysisData` holding:

- the valid Tandem CGM readings (``cgm``; the single validity rule is in ``context_rules``);
- the pump's profile history, replayed once (``history``);
- the latest pump, its active profile and the one moment (``moment``) its settings and time-of-day bins are read at;
- the 5-minute analysis grid (``grid``): every slot of the record with observed glucose, boluses, carbohydrate and IOB;
- the unlogged-food intervals (``unlogged``), found once from the grid by time, and the exercise intervals (``exercise``);
- Control-IQ mode changes and site events for the whole record, and the dates they are observed over;
- the days the person marked as context events (``excluded_days``).

Every analysis takes these instead of re-reading the files, so two analyses cannot read different moments, different
CGM rules or different spans.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import meal_flags
from .context_rules import exercise_intervals, valid_cgm
from .dataset import MergedDataset, load_merged_dataset
from .outcomes import DayPart, day_parts_from_starts, segment_starts_in_force
from .pump_profiles import ProfileHistory, pump_key
from .therapy_research import load_excluded_periods


@dataclass
class AnalysisData:
    merged: MergedDataset
    cgm: pd.DataFrame
    history: ProfileHistory
    pump: str
    profile_id: int
    moment: dt.datetime
    mode_events: pd.DataFrame
    site_events: pd.DataFrame
    grid: pd.DataFrame
    unlogged: pd.DataFrame
    exercise: pd.DataFrame
    event_span: dict[str, tuple[dt.date, dt.date]]
    excluded_days: list[pd.Timestamp]

    def day_parts(self) -> tuple[list[DayPart], str]:
        """Time-of-day bins from the segments of the profile the latest pump is running, read at ``moment``."""
        starts = segment_starts_in_force(self.history, self.pump, self.moment)
        if not starts:
            raise ValueError(f"No profile segments are visible for pump {self.pump}.")
        source = f"pump {self.pump}, profile {self.profile_id}, segments as last seen on {self.moment.date()}"
        return day_parts_from_starts(starts), source

    def segment_settings(self) -> dict[float, dict[str, float]]:
        """The I:C, CF, basal and target each pump segment holds at ``moment``, keyed by the segment's start minute."""
        settings: dict[float, dict[str, float]] = {}
        for minute in range(0, 24 * 60, 15):
            segment = self.history.segment_in_force(self.pump, self.profile_id, self.moment, minute_of_day=minute)
            if segment is not None:
                settings[float(segment["start_minutes"])] = {
                    "icr": segment["carb_ratio_g_per_unit"], "cf": segment["isf_mgdl_per_unit"],
                    "basal": segment.get("basal_units_per_hour", float("nan")), "target": segment["target_bg_mgdl"],
                }
        return settings

    def site_observed_from(self) -> pd.Timestamp:
        """The first moment site changes are observed for the latest pump (earlier infusion-site changes are unknown)."""
        return pd.Timestamp(self.event_span[self.pump][0]) if self.pump in self.event_span else pd.Timestamp.max


def unlogged_for(merged: MergedDataset) -> pd.DataFrame:
    """Unlogged-food intervals, found once from the analysis grid (observed glucose only) by time."""
    frame = merged.grid.rename(columns={"carb_grams": "explicit_carb_grams"})[["timestamp", "glucose", "explicit_carb_grams", "bolus_units"]]
    return meal_flags.unlogged_intervals(frame, merged.data.nutrition)


def exercise_for(merged: MergedDataset) -> pd.DataFrame:
    """The shared exercise intervals from the workout log and the exercise-mode periods."""
    return exercise_intervals(merged.data.workouts, merged.mode_events)


def load_analysis_data(raw_root: str | Path, merged: MergedDataset | None = None) -> AnalysisData:
    merged = merged if merged is not None else load_merged_dataset(raw_root)
    visible = {
        pump: (dt.date.fromisoformat(item["edits_visible"][0][:10]), dt.date.fromisoformat(item["edits_visible"][1][:10]))
        for pump, item in merged.manifest["pump_profile_history"].items()
    }
    cgm = valid_cgm(merged.tandem_data.cgm)
    pump = pump_key(merged.tandem_data.cgm.sort_values("timestamp")["pump_serial"].dropna().iloc[-1])
    if pump not in visible:
        raise ValueError(f"The pump profile history has no visible segments for pump {pump}; time-of-day bins cannot be built.")
    history = ProfileHistory(merged.profile_events, visible)
    moment = dt.datetime.combine(visible[pump][1], dt.time(23, 59))  # one moment for the bins and the settings
    profile_id = history.active_profile(pump, moment)
    if profile_id is None:
        raise ValueError(f"No profile is active for pump {pump} on {moment.date()}.")
    span = {pump_name: (dt.date.fromisoformat(a[:10]), dt.date.fromisoformat(b[:10])) for pump_name, (a, b) in merged.manifest["device_event_span"].items()}
    excluded = [pd.Timestamp(day) for period in load_excluded_periods(raw_root) for day in pd.date_range(period.start, period.end, freq="D")]
    return AnalysisData(merged=merged, cgm=cgm, history=history, pump=pump, profile_id=int(profile_id), moment=moment,
                        mode_events=merged.mode_events, site_events=merged.site_events, grid=merged.grid, unlogged=unlogged_for(merged), exercise=exercise_for(merged), event_span=span, excluded_days=excluded)
