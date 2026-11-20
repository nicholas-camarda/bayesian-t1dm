from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .features import MEAL_RECORD_COLUMNS
from .health_auto_export import AnalysisReadyHealthDataset
from . import meal_flags
from .context_rules import inside_intervals
from .pump_profiles import pump_key
from .latent_meal_trajectory import (
    MIN_LOGGED_MEALS_FOR_ICR,
    CarbSensitivityObservation,
    MealWindowTrajectory,
    absorbed_units_from_deliveries,
    estimate_carbs_from_trajectory,
    observe_carb_sensitivity,
    pool_effective_icr,
)
from .report_page import icr_verdict, plain


DEFAULT_SEGMENT_SPEC = "overnight=00:00-06:00,morning=06:00-11:00,afternoon=11:00-17:00,evening=17:00-24:00"
MEANINGFUL_BASAL_RATE_CHANGE_THRESHOLD = 0.4
MAX_PLAUSIBLE_WORKOUT_COUNT_24H = 12
MODERATE_ACTIVITY_STEPS_60M = 500.0
HIGH_ACTIVITY_STEPS_60M = 1000.0


def _column_or_default(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index, dtype=float)


def _workout_summary_trustworthy(frame: pd.DataFrame) -> pd.Series:
    plausible = pd.to_numeric(_column_or_default(frame, "workout_summary_plausible", default=np.nan), errors="coerce")
    count_24h = pd.to_numeric(_column_or_default(frame, "workout_count_24h", default=np.nan), errors="coerce")
    duration_24h = pd.to_numeric(_column_or_default(frame, "workout_duration_sum_24h", default=np.nan), errors="coerce")
    trust = pd.Series(True, index=frame.index, dtype=bool)
    if plausible.notna().any():
        trust &= plausible.fillna(0.0).eq(1.0)
    if count_24h.notna().any():
        trust &= count_24h.fillna(0.0).le(MAX_PLAUSIBLE_WORKOUT_COUNT_24H)
    if duration_24h.notna().any():
        trust &= duration_24h.fillna(0.0).le(24 * 60 * 60)
    return trust.astype(int)


@dataclass(frozen=True)
class TherapySegment:
    name: str
    start_minute: int
    end_minute: int

    def contains(self, minute_of_day: pd.Series) -> pd.Series:
        if self.start_minute < self.end_minute:
            return minute_of_day.ge(self.start_minute) & minute_of_day.lt(self.end_minute)
        return minute_of_day.ge(self.start_minute) | minute_of_day.lt(self.end_minute)


@dataclass(frozen=True)
class LatentMealTherapySettings:
    icr_grams_per_unit: dict[str, float]
    correction_factor_mgdl_per_unit: dict[str, float]
    target_glucose_mgdl: float = 110.0
    insulin_action_minutes: int = 300
    carb_absorption_minutes_prior: int = 180
    max_plausible_meal_carbs: float = 180.0

    @classmethod
    def default(cls) -> "LatentMealTherapySettings":
        return cls(
            icr_grams_per_unit={"overnight": 10.0, "morning": 10.0, "afternoon": 10.0, "evening": 10.0},
            correction_factor_mgdl_per_unit={"overnight": 40.0, "morning": 40.0, "afternoon": 40.0, "evening": 40.0},
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "LatentMealTherapySettings":
        default = cls.default()
        return cls(
            icr_grams_per_unit=_parse_segment_numeric_mapping(
                payload.get("icr_grams_per_unit"),
                default=default.icr_grams_per_unit,
                name="icr_grams_per_unit",
            ),
            correction_factor_mgdl_per_unit=_parse_segment_numeric_mapping(
                payload.get("correction_factor_mgdl_per_unit"),
                default=default.correction_factor_mgdl_per_unit,
                name="correction_factor_mgdl_per_unit",
            ),
            target_glucose_mgdl=_positive_float(payload.get("target_glucose_mgdl", default.target_glucose_mgdl), "target_glucose_mgdl"),
            insulin_action_minutes=int(_positive_float(payload.get("insulin_action_minutes", default.insulin_action_minutes), "insulin_action_minutes")),
            carb_absorption_minutes_prior=int(
                _positive_float(payload.get("carb_absorption_minutes_prior", default.carb_absorption_minutes_prior), "carb_absorption_minutes_prior")
            ),
            max_plausible_meal_carbs=_positive_float(payload.get("max_plausible_meal_carbs", default.max_plausible_meal_carbs), "max_plausible_meal_carbs"),
        )

    def icr_for_segment(self, segment: str) -> float:
        return _segment_value(self.icr_grams_per_unit, segment, default=10.0)

    def correction_factor_for_segment(self, segment: str) -> float:
        return _segment_value(self.correction_factor_mgdl_per_unit, segment, default=40.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "icr_grams_per_unit": dict(self.icr_grams_per_unit),
            "correction_factor_mgdl_per_unit": dict(self.correction_factor_mgdl_per_unit),
            "target_glucose_mgdl": float(self.target_glucose_mgdl),
            "insulin_action_minutes": int(self.insulin_action_minutes),
            "carb_absorption_minutes_prior": int(self.carb_absorption_minutes_prior),
            "max_plausible_meal_carbs": float(self.max_plausible_meal_carbs),
        }


@dataclass(frozen=True)
class LatentMealResearchResult:
    prepared_dataset: AnalysisReadyHealthDataset
    research_frame: pd.DataFrame
    research_gate: pd.DataFrame
    meal_event_registry: pd.DataFrame
    meal_windows: pd.DataFrame
    first_meal_exclusion_summary: pd.DataFrame
    posterior_meals: pd.DataFrame
    model_comparison: pd.DataFrame
    research_gate_markdown: str
    meal_truth_semantics_report_markdown: str
    meal_window_audit_markdown: str
    fit_summary_markdown: str
    confidence_report_markdown: str
    model_comparison_markdown: str
    segments: tuple[TherapySegment, ...]
    meal_proxy_mode: str
    research_scope: str = "foundation"
    therapy_settings: LatentMealTherapySettings | None = None
    latent_carb_estimates: pd.DataFrame | None = None
    posterior_samples: pd.DataFrame | None = None
    latent_carb_estimator_report_markdown: str = ""
    tier_summary: pd.DataFrame | None = None
    tier_summary_markdown: str = ""
    effective_icr_evidence: pd.DataFrame | None = None
    icr_support: pd.DataFrame | None = None


@dataclass(frozen=True)
class LatentMealFixtureResult:
    source_result: LatentMealResearchResult
    fixture_result: LatentMealResearchResult
    fixture_dataset: AnalysisReadyHealthDataset
    selected_day_manifest: pd.DataFrame
    summary_markdown: str
    background_days_requested: int
    seed: int


def _parse_clock(text: str) -> int:
    if text == "24:00":
        return 24 * 60
    hour_text, minute_text = text.split(":", 1)
    hour = int(hour_text)
    minute = int(minute_text)
    return hour * 60 + minute


def parse_therapy_segments(spec: str | None = None) -> tuple[TherapySegment, ...]:
    raw = (spec or DEFAULT_SEGMENT_SPEC).strip()
    if not raw:
        raise ValueError("Segment specification cannot be empty")
    segments: list[TherapySegment] = []
    for chunk in raw.split(","):
        name_text, range_text = chunk.split("=", 1)
        start_text, end_text = range_text.split("-", 1)
        segments.append(
            TherapySegment(
                name=name_text.strip(),
                start_minute=_parse_clock(start_text.strip()),
                end_minute=_parse_clock(end_text.strip()),
            )
        )
    if not segments:
        raise ValueError("At least one therapy segment is required")
    return tuple(segments)


def _assign_segments(frame: pd.DataFrame, segments: tuple[TherapySegment, ...]) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)
    minute_of_day = out["timestamp"].dt.hour * 60 + out["timestamp"].dt.minute
    out["therapy_segment"] = ""
    out["therapy_segment_order"] = -1
    for index, segment in enumerate(segments):
        mask = segment.contains(minute_of_day)
        out.loc[mask, "therapy_segment"] = segment.name
        out.loc[mask, "therapy_segment_order"] = index
        out[f"segment__{segment.name}"] = mask.astype(int)
    if out["therapy_segment"].eq("").any():
        out.loc[out["therapy_segment"].eq(""), "therapy_segment"] = segments[0].name
        out.loc[out["therapy_segment_order"].lt(0), "therapy_segment_order"] = 0
    return out


def _therapy_epoch_ids(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=int)
    timestamp = pd.to_datetime(frame["timestamp"], errors="coerce")
    gap_break = timestamp.diff().dt.total_seconds().div(60.0).fillna(0.0).gt(180.0)
    basal_change = _column_or_default(frame, "basal_schedule_change").astype(float).gt(0)
    epoch_start = (gap_break | basal_change).astype(int)
    return epoch_start.cumsum().astype(int)


def _classify_bolus_proxy(
    frame: pd.DataFrame,
    *,
    direct_meal_signal: pd.Series,
    mode: str,
) -> pd.DataFrame:
    out = frame.copy()
    bolus_units = pd.to_numeric(_column_or_default(out, "bolus_units"), errors="coerce").fillna(0.0)
    glucose = pd.to_numeric(_column_or_default(out, "glucose"), errors="coerce").fillna(0.0)
    glucose_lag_30m = pd.to_numeric(_column_or_default(out, "glucose_lag_30m", default=np.nan), errors="coerce")
    glucose_roll_mean_60m = pd.to_numeric(_column_or_default(out, "glucose_roll_mean_60m", default=np.nan), errors="coerce")
    timestamps = pd.to_datetime(out["timestamp"], errors="coerce")
    hour = timestamps.dt.hour.fillna(0).astype(int)
    minute = timestamps.dt.minute.fillna(0).astype(int)
    minute_of_day = hour * 60 + minute
    meal_window = (
        minute_of_day.between(6 * 60, 10 * 60 - 1)
        | minute_of_day.between(11 * 60, 14 * 60 - 1)
        | minute_of_day.between(17 * 60, 21 * 60 - 1)
    )
    glucose_uptrend = (glucose - glucose_lag_30m.fillna(glucose)).gt(12.0)
    recent_bolus_count = bolus_units.gt(0.2).rolling(12, min_periods=1).sum()
    stacked_bolus = recent_bolus_count.gt(1.0)
    meaningful_bolus = bolus_units.ge(0.5)

    correction_like = meaningful_bolus & (
        glucose.ge(165.0)
        & (glucose_uptrend | glucose_roll_mean_60m.fillna(glucose).ge(155.0) | ~meal_window)
        & ~direct_meal_signal
    )
    meal_like = meaningful_bolus & ~correction_like & (
        direct_meal_signal
        | (meal_window & glucose.lt(180.0) & ~stacked_bolus)
    )
    ambiguous = meaningful_bolus & ~(meal_like | correction_like)

    if mode == "off":
        meal_like = pd.Series(False, index=out.index, dtype=bool)
        correction_like = pd.Series(False, index=out.index, dtype=bool)
        ambiguous = meaningful_bolus
    elif mode == "broad":
        meal_like = meaningful_bolus & ~correction_like
        ambiguous = meaningful_bolus & ~(meal_like | correction_like)

    confidence = pd.Series(0.0, index=out.index, dtype=float)
    confidence.loc[meal_like & direct_meal_signal] = 0.98
    confidence.loc[meal_like & ~direct_meal_signal] = 0.80
    confidence.loc[correction_like] = 0.85
    confidence.loc[ambiguous] = 0.45

    out["bolus_proxy_class"] = np.where(
        meal_like,
        "meal_like",
        np.where(correction_like, "correction_like", np.where(ambiguous, "ambiguous", "none")),
    )
    out["meal_proxy_event"] = (meal_like & confidence.ge(0.8)).astype(int)
    out["meal_proxy_confidence"] = np.where(meal_like, confidence, 0.0)
    out["meal_proxy_units"] = np.where(out["meal_proxy_event"].astype(bool), bolus_units, 0.0)
    out["meal_proxy_size_class"] = np.where(
        out["meal_proxy_units"].ge(6.0),
        "large",
        np.where(out["meal_proxy_units"].ge(3.0), "medium", np.where(out["meal_proxy_units"].gt(0), "small", "none")),
    )
    out["correction_proxy_event"] = (correction_like & confidence.ge(0.8)).astype(int)
    out["correction_proxy_confidence"] = np.where(correction_like, confidence, 0.0)
    return out


@dataclass(frozen=True)
class ExcludedPeriod:
    """A stretch of days left out of the meal and basal analyses (for example after a steroid injection)."""

    start: pd.Timestamp  # inclusive, start of day
    end: pd.Timestamp  # inclusive, end of day
    reason: str


CONTEXT_EVENTS_FILENAME = "context_events.csv"


def load_excluded_periods(raw_root: str | Path) -> tuple[ExcludedPeriod, ...]:
    """Read ``<raw>/context_events.csv`` (columns start_date, end_date, reason). No file means no exclusions."""
    path = Path(raw_root) / CONTEXT_EVENTS_FILENAME
    if not path.exists():
        return ()
    frame = pd.read_csv(path, dtype=str).fillna("")
    missing = {"start_date", "end_date", "reason"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    periods = []
    for row in frame.itertuples(index=False):
        start = pd.to_datetime(row.start_date, errors="coerce")
        end = pd.to_datetime(row.end_date, errors="coerce")
        if pd.isna(start) or pd.isna(end) or end < start or not row.reason.strip():
            raise ValueError(f"{path}: invalid row {row.start_date!r}, {row.end_date!r}, {row.reason!r}")
        periods.append(ExcludedPeriod(start=start.normalize(), end=end.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1), reason=row.reason.strip()))
    return tuple(periods)


def _in_excluded_period(timestamps: pd.Series, periods: tuple[ExcludedPeriod, ...]) -> pd.Series:
    stamps = pd.to_datetime(timestamps, errors="coerce")
    flagged = pd.Series(False, index=timestamps.index)
    for period in periods:
        flagged |= stamps.between(period.start, period.end)
    return flagged


def build_therapy_research_frame(
    dataset: AnalysisReadyHealthDataset,
    *,
    segments: tuple[TherapySegment, ...] | None = None,
    meal_proxy_mode: str = "strict",
    excluded_periods: tuple[ExcludedPeriod, ...] = (),
) -> pd.DataFrame:
    frame = dataset.frame.copy()
    if frame.empty:
        return frame
    segments = segments or parse_therapy_segments()
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame = _assign_segments(frame, segments)

    explicit_source_available = bool(getattr(dataset, "explicit_carb_source_available", False))
    if "explicit_carb_grams" not in frame.columns:
        if explicit_source_available:
            legacy_carb = pd.to_numeric(_column_or_default(frame, "carb_grams"), errors="coerce")
            frame["explicit_carb_grams"] = legacy_carb.where(legacy_carb.gt(0), np.nan)
        else:
            frame["explicit_carb_grams"] = np.nan
    if "explicit_meal_event" not in frame.columns:
        if explicit_source_available:
            frame["explicit_meal_event"] = pd.to_numeric(frame["explicit_carb_grams"], errors="coerce").fillna(0.0).gt(0).astype(float)
        else:
            frame["explicit_meal_event"] = np.nan
    if "minutes_since_last_explicit_meal" not in frame.columns:
        if explicit_source_available:
            explicit_meal_event_series = pd.to_numeric(frame["explicit_meal_event"], errors="coerce").fillna(0.0)
            last_explicit_meal_ts = pd.to_datetime(frame["timestamp"], errors="coerce").where(explicit_meal_event_series.gt(0)).ffill()
            frame["minutes_since_last_explicit_meal"] = (
                pd.to_datetime(frame["timestamp"], errors="coerce") - pd.to_datetime(last_explicit_meal_ts, errors="coerce")
            ).dt.total_seconds().div(60.0)
        else:
            frame["minutes_since_last_explicit_meal"] = np.nan
    frame["explicit_carb_source_available"] = int(explicit_source_available)
    if "meal_truth_status" not in frame.columns:
        explicit_event_series = pd.to_numeric(frame["explicit_meal_event"], errors="coerce")
        frame["meal_truth_status"] = np.where(
            explicit_event_series.fillna(0.0).gt(0),
            "observed_explicit",
            np.where(frame["explicit_carb_source_available"].astype(int).eq(1), "missing_from_source", "unavailable"),
        )

    for column in ["carb_roll_sum_120m", "iob_roll_sum_120m", "iob_roll_sum_60m", "minutes_since_last_meal", "bolus_units", "glucose"]:
        if column not in frame.columns:
            frame[column] = 0.0
    explicit_meal_event = pd.to_numeric(_column_or_default(frame, "explicit_meal_event", default=np.nan), errors="coerce")
    explicit_carb_grams = pd.to_numeric(_column_or_default(frame, "explicit_carb_grams", default=np.nan), errors="coerce")
    minutes_since_last_explicit_meal = pd.to_numeric(_column_or_default(frame, "minutes_since_last_explicit_meal", default=np.nan), errors="coerce")
    bolus_units = pd.to_numeric(frame["bolus_units"], errors="coerce").fillna(0.0)
    glucose = pd.to_numeric(frame["glucose"], errors="coerce").fillna(np.nan)
    iob_roll_sum_120m = pd.to_numeric(frame["iob_roll_sum_120m"], errors="coerce").fillna(0.0)

    meal_signal_available = bool(
        frame["explicit_carb_source_available"].astype(int).eq(1).any()
        and (explicit_meal_event.fillna(0.0).gt(0).any() or explicit_carb_grams.fillna(0.0).gt(0).any())
    )
    direct_meal_signal = (
        explicit_meal_event.fillna(0.0).gt(0)
        | explicit_carb_grams.fillna(0.0).gt(0)
        | (minutes_since_last_explicit_meal.le(30).fillna(False) if meal_signal_available else pd.Series(False, index=frame.index, dtype=bool))
    )
    frame = _classify_bolus_proxy(frame, direct_meal_signal=direct_meal_signal, mode=meal_proxy_mode)
    if meal_signal_available:
        recent_meal = (
            direct_meal_signal
            | frame["meal_proxy_event"].astype(int).eq(1)
            | minutes_since_last_explicit_meal.le(120).fillna(False)
        )
        fasting_like = (
            minutes_since_last_explicit_meal.ge(240).fillna(False)
            & frame["meal_proxy_event"].astype(int).eq(0)
        )
    else:
        # No direct meal signal is available, so missing carb records should not imply
        # that every row is meal-adjacent. We fall back to "meal timing unknown" and let
        # insulin context drive exclusion for basal-like analyses.
        recent_meal = frame["meal_proxy_event"].astype(int).eq(1)
        fasting_like = frame["meal_proxy_event"].astype(int).eq(0)

    frame["recent_meal_120m"] = recent_meal.astype(int)
    frame["recent_bolus_120m"] = (
        bolus_units.gt(0)
        | iob_roll_sum_120m.gt(1.5)
    ).astype(int)
    frame["workout_summary_trustworthy"] = _workout_summary_trustworthy(frame)
    workout_trustworthy = frame["workout_summary_trustworthy"].eq(1)
    recent_workout_6h = pd.to_numeric(_column_or_default(frame, "recent_workout_6h"), errors="coerce").fillna(0.0).gt(0) & workout_trustworthy
    recent_workout_12h = pd.to_numeric(_column_or_default(frame, "recent_workout_12h"), errors="coerce").fillna(0.0).gt(0) & workout_trustworthy
    activity_60m = pd.to_numeric(_column_or_default(frame, "health_activity_roll_sum_60m"), errors="coerce").fillna(0.0)
    frame["recent_workout_context"] = recent_workout_6h.astype(int)
    frame["recent_workout_12h_context"] = recent_workout_12h.astype(int)
    frame["light_activity_context"] = (activity_60m.gt(0.0) & activity_60m.lt(MODERATE_ACTIVITY_STEPS_60M)).astype(int)
    frame["moderate_activity_context"] = (activity_60m.ge(MODERATE_ACTIVITY_STEPS_60M) & activity_60m.lt(HIGH_ACTIVITY_STEPS_60M)).astype(int)
    frame["high_activity_context"] = activity_60m.ge(HIGH_ACTIVITY_STEPS_60M).astype(int)
    frame["recent_exercise_context"] = (recent_workout_6h | frame["high_activity_context"].astype(bool)).astype(int)
    frame["basal_context"] = (
        fasting_like
        & frame["recent_bolus_120m"].eq(0)
        & iob_roll_sum_120m.le(1.5)
        & _column_or_default(frame, "missing_cgm").astype(float).eq(0)
        & frame["recent_exercise_context"].fillna(0).eq(0)
    ).astype(int)
    frame["meal_context"] = frame["recent_meal_120m"].astype(int)
    frame["correction_context"] = (
        frame["correction_proxy_event"].astype(int).eq(1)
        & frame["recent_meal_120m"].eq(0)
        & bolus_units.gt(0)
        & glucose.ge(140.0)
    ).astype(int)
    frame["sleep_deficit_flag"] = (
        pd.to_numeric(_column_or_default(frame, "prior_night_total_sleep_hours"), errors="coerce").lt(7.0)
        & pd.to_numeric(_column_or_default(frame, "prior_night_sleep_known"), errors="coerce").eq(1)  # unknown sleep is not short sleep
    ).astype(int)
    frame["post_workout_meal_context"] = frame["meal_context"] * frame["recent_exercise_context"]
    if "hrv_latest" in frame.columns:
        frame["overnight_hrv_interaction"] = _column_or_default(frame, "segment__overnight").astype(float) * pd.to_numeric(frame["hrv_latest"], errors="coerce").fillna(0.0)
    else:
        frame["overnight_hrv_interaction"] = 0.0
    if "heart_rate_avg_latest" in frame.columns:
        frame["in_sleep_heart_rate_interaction"] = _column_or_default(frame, "in_sleep").astype(float) * pd.to_numeric(frame["heart_rate_avg_latest"], errors="coerce").fillna(0.0)
    else:
        frame["in_sleep_heart_rate_interaction"] = 0.0
    frame["therapy_stable_epoch"] = _therapy_epoch_ids(frame)
    frame["high_iob_state"] = iob_roll_sum_120m.gt(1.5).astype(int)
    frame["stacked_bolus_state"] = bolus_units.gt(0.2).rolling(12, min_periods=1).sum().gt(1.0).astype(int)
    basal_rate = pd.to_numeric(_column_or_default(frame, "basal_units_per_hour"), errors="coerce")
    basal_rate_step_change = basal_rate.diff().abs().fillna(0.0)
    frame["meaningful_basal_rate_change"] = basal_rate_step_change.ge(MEANINGFUL_BASAL_RATE_CHANGE_THRESHOLD).astype(int)
    frame["recent_therapy_transition"] = frame["meaningful_basal_rate_change"].astype(int)
    frame["closed_loop_confounding_flag"] = (
        frame["high_iob_state"].eq(1)
        | frame["stacked_bolus_state"].eq(1)
        | frame["recent_therapy_transition"].eq(1)
    ).astype(int)
    frame["context_event_excluded"] = _in_excluded_period(frame["timestamp"], excluded_periods).astype(int)
    frame["basal_context"] = (
        frame["basal_context"].astype(int).eq(1)
        & frame["closed_loop_confounding_flag"].eq(0)
        & frame["context_event_excluded"].eq(0)
    ).astype(int)
    return frame


def build_meal_event_registry(
    research_frame: pd.DataFrame,
    *,
    cluster_gap_minutes: int = 90,
) -> pd.DataFrame:
    columns = [
        "meal_id",
        "timestamp",
        "segment",
        "event_row_index",
        "evidence_source",
        "explicit_carb_rows",
        "proxy_rows",
        "stated_carbs",
        "bolus_units",
        "meal_proxy_confidence",
        "glucose",
    ]
    if research_frame.empty:
        return pd.DataFrame(columns=columns)

    ordered = research_frame.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="coerce")
    ordered = ordered.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=False)
    explicit_signal = (
        pd.to_numeric(_column_or_default(ordered, "explicit_meal_event", default=np.nan), errors="coerce").fillna(0.0).gt(0)
        | pd.to_numeric(_column_or_default(ordered, "explicit_carb_grams", default=np.nan), errors="coerce").fillna(0.0).gt(0)
    )
    proxy_signal = pd.to_numeric(_column_or_default(ordered, "meal_proxy_event"), errors="coerce").fillna(0.0).gt(0)
    candidates = ordered.loc[explicit_signal | proxy_signal].copy()
    if candidates.empty:
        return pd.DataFrame(columns=columns)

    gap = candidates["timestamp"].diff().dt.total_seconds().div(60.0).fillna(float(cluster_gap_minutes + 1))
    candidates["meal_cluster_id"] = gap.gt(cluster_gap_minutes).cumsum().astype(int)
    rows: list[dict[str, Any]] = []
    for cluster_id, cluster in candidates.groupby("meal_cluster_id", dropna=False):
        cluster = cluster.copy()
        cluster["explicit_signal"] = (
            pd.to_numeric(_column_or_default(cluster, "explicit_meal_event", default=np.nan), errors="coerce").fillna(0.0).gt(0)
            | pd.to_numeric(_column_or_default(cluster, "explicit_carb_grams", default=np.nan), errors="coerce").fillna(0.0).gt(0)
        ).astype(int)
        cluster["proxy_signal"] = pd.to_numeric(_column_or_default(cluster, "meal_proxy_event"), errors="coerce").fillna(0.0).gt(0).astype(int)
        cluster["ranking_stated_carbs"] = pd.to_numeric(_column_or_default(cluster, "explicit_carb_grams", default=np.nan), errors="coerce").fillna(0.0)
        cluster["ranking_bolus"] = pd.to_numeric(_column_or_default(cluster, "bolus_units"), errors="coerce").fillna(0.0)
        cluster["ranking_proxy_confidence"] = pd.to_numeric(_column_or_default(cluster, "meal_proxy_confidence"), errors="coerce").fillna(0.0)
        selected = cluster.sort_values(
            ["explicit_signal", "ranking_stated_carbs", "ranking_proxy_confidence", "ranking_bolus", "timestamp"],
            ascending=[False, False, False, False, True],
        ).iloc[0]
        explicit_rows = int(cluster["explicit_signal"].sum())
        proxy_rows = int(cluster["proxy_signal"].sum())
        evidence_source = "explicit_and_proxy" if explicit_rows and proxy_rows else "explicit_logged" if explicit_rows else "proxy_only"
        stated_carbs = pd.to_numeric(pd.Series([selected.get("explicit_carb_grams")]), errors="coerce").iloc[0]
        rows.append(
            {
                "meal_id": f"meal_{int(cluster_id):04d}",
                "timestamp": pd.Timestamp(selected["timestamp"]),
                "segment": str(selected.get("therapy_segment") or ""),
                "event_row_index": int(selected["index"]),
                "evidence_source": evidence_source,
                "explicit_carb_rows": explicit_rows,
                "proxy_rows": proxy_rows,
                "stated_carbs": None if pd.isna(stated_carbs) or float(stated_carbs) <= 0 else float(stated_carbs),
                "bolus_units": float(pd.to_numeric(pd.Series([selected.get("bolus_units")]), errors="coerce").fillna(0.0).iloc[0]),
                "meal_proxy_confidence": float(pd.to_numeric(pd.Series([selected.get("meal_proxy_confidence")]), errors="coerce").fillna(0.0).iloc[0]),
                "glucose": float(pd.to_numeric(pd.Series([selected.get("glucose")]), errors="coerce").fillna(np.nan).iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def build_first_meal_clean_window_registry(
    research_frame: pd.DataFrame,
    *,
    premeal_minutes: int = 30,
    postmeal_minutes: int = 180,
    unlogged: pd.DataFrame | None = None,
    exercise: pd.DataFrame | None = None,
    meal_segments: tuple[str, ...] = ("morning",),
) -> pd.DataFrame:
    """One row per first-meal candidate with its exclusion reasons. ``exercise`` is the shared exercise rule
    (``context_rules.exercise_intervals``): a meal that starts inside it is excluded as ``exercise``."""
    columns = [
        "candidate_id",
        "meal_id",
        "date",
        "timestamp",
        "segment",
        "candidate_status",
        "included",
        "exclusion_reasons",
        "explicit_carb_source_available",
        "meal_truth_status",
        "stated_carbs",
        *MEAL_RECORD_COLUMNS,
        "bolus_units",
        "premeal_glucose",
        "premeal_delta_30m",
        "premeal_iob",
        "in_exercise",
        "closed_loop_confounding_premeal",
        "prior_bolus_4h",
        "postmeal_additional_bolus_120m",
        "cgm_coverage_fraction",
        "missing_cgm_first_90m",
        "window_row_count",
        "source_quality_grade",
    ]
    if research_frame.empty:
        return pd.DataFrame(columns=columns)

    ordered = research_frame.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="coerce")
    ordered = ordered.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ordered["date"] = ordered["timestamp"].dt.normalize()
    step_seconds = ordered["timestamp"].diff().dt.total_seconds().dropna()
    step_minutes = float(step_seconds[step_seconds > 0].median() / 60.0) if (step_seconds > 0).any() else 5.0
    morning_candidates = ordered.loc[
        ordered["therapy_segment"].astype(str).isin(meal_segments)
        & pd.to_numeric(_column_or_default(ordered, "bolus_units"), errors="coerce").fillna(0.0).ge(1.5)
    ].copy()
    if morning_candidates.empty:
        return pd.DataFrame(columns=columns)

    # The first bolus of 1.5 U or more in each meal segment of each day.
    candidate_rows = morning_candidates.groupby(["date", morning_candidates["therapy_segment"].astype(str)], as_index=False).head(1).copy().reset_index(drop=True)
    stamps = pd.DatetimeIndex(ordered["timestamp"])

    def window(start: pd.Timestamp, end: pd.Timestamp, *, include_start: bool = True, include_end: bool = True) -> pd.DataFrame:
        first = stamps.searchsorted(start, side="left" if include_start else "right")
        last = stamps.searchsorted(end, side="right" if include_end else "left")
        return ordered.iloc[first:last]

    rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidate_rows.itertuples(index=False), start=1):
        timestamp = pd.Timestamp(candidate.timestamp)
        beginning = timestamp - pd.Timedelta(minutes=premeal_minutes)
        end = timestamp + pd.Timedelta(minutes=postmeal_minutes)
        pre = window(beginning, timestamp, include_end=False)
        evaluation_window = window(beginning, end)
        post_early = window(timestamp, timestamp + pd.Timedelta(minutes=90))
        prior_bolus_window = window(timestamp - pd.Timedelta(hours=4), timestamp, include_end=False)
        post_bolus_window = window(timestamp, timestamp + pd.Timedelta(minutes=120), include_start=False)

        pre_glucose_series = pd.to_numeric(pre.get("glucose"), errors="coerce").dropna()
        anchor_glucose = float(pd.to_numeric(pd.Series([getattr(candidate, "glucose", np.nan)]), errors="coerce").iloc[0])
        premeal_glucose = float(pre_glucose_series.iloc[-1]) if not pre_glucose_series.empty else anchor_glucose
        premeal_start_glucose = float(pre_glucose_series.iloc[0]) if not pre_glucose_series.empty else float("nan")
        premeal_delta_30m = float(premeal_glucose - premeal_start_glucose) if np.isfinite(premeal_glucose) and np.isfinite(premeal_start_glucose) else float("nan")

        premeal_iob_series = pd.to_numeric(pre.get("iob_units"), errors="coerce").dropna()
        premeal_iob = float(premeal_iob_series.iloc[-1]) if not premeal_iob_series.empty else float(pd.to_numeric(pd.Series([getattr(candidate, "iob_units", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        in_exercise = bool(exercise is not None and inside_intervals(pd.Series([timestamp]), exercise)[0])
        meaningful_basal_transition_premeal = bool(pd.to_numeric(pre.get("meaningful_basal_rate_change"), errors="coerce").fillna(0.0).max() > 0)
        stacked_bolus_premeal = bool(pd.to_numeric(pre.get("stacked_bolus_state"), errors="coerce").fillna(0.0).max() > 0)
        closed_loop_confounding_premeal = meaningful_basal_transition_premeal or stacked_bolus_premeal
        prior_bolus_4h = bool(pd.to_numeric(prior_bolus_window.get("bolus_units"), errors="coerce").fillna(0.0).ge(0.5).any())
        postmeal_additional_bolus_120m = bool(pd.to_numeric(post_bolus_window.get("bolus_units"), errors="coerce").fillna(0.0).ge(0.5).any())
        # Coverage is measured against the rows a complete window would have, not the rows
        # that happen to be present: the prepared frame drops rows whose forecast target is
        # missing, so a CGM gap removes rows as well as flagging them.
        expected_window_rows = int(round((premeal_minutes + postmeal_minutes) / step_minutes)) + 1
        expected_early_rows = int(round(90.0 / step_minutes)) + 1
        observed_window_rows = int(pd.to_numeric(evaluation_window.get("missing_cgm"), errors="coerce").fillna(1.0).eq(0).sum()) if not evaluation_window.empty else 0
        observed_early_rows = int(pd.to_numeric(post_early.get("missing_cgm"), errors="coerce").fillna(1.0).eq(0).sum()) if not post_early.empty else 0
        cgm_coverage_fraction = float(min(1.0, observed_window_rows / expected_window_rows))
        missing_cgm_first_90m = bool(observed_early_rows < expected_early_rows)
        source_quality_grade = "good" if cgm_coverage_fraction >= 0.9 and not missing_cgm_first_90m else "degraded"
        exclusion_reasons: list[str] = []
        if prior_bolus_4h:
            exclusion_reasons.append("recent_bolus_4h")
        if not np.isfinite(premeal_iob) or premeal_iob >= 1.0:
            exclusion_reasons.append("high_premeal_iob")
        if not np.isfinite(premeal_glucose) or premeal_glucose < 80.0 or premeal_glucose > 130.0:
            exclusion_reasons.append("premeal_bg_out_of_range")
        if not np.isfinite(premeal_delta_30m) or abs(premeal_delta_30m) > 20.0:
            exclusion_reasons.append("steep_premeal_slope")
        if in_exercise:
            exclusion_reasons.append("exercise")
        if closed_loop_confounding_premeal:
            exclusion_reasons.append("closed_loop_confounding_premeal")
        if postmeal_additional_bolus_120m:
            exclusion_reasons.append("postmeal_additional_bolus_120m")
        if cgm_coverage_fraction < 0.9:
            exclusion_reasons.append("poor_cgm_coverage")
        if missing_cgm_first_90m:
            exclusion_reasons.append("missing_cgm_first_90m")
        if int(pd.to_numeric(pd.Series([getattr(candidate, "context_event_excluded", 0)]), errors="coerce").fillna(0).iloc[0]):
            exclusion_reasons.append("context_event_excluded")
        if unlogged is not None and meal_flags.overlaps(unlogged, timestamp - pd.Timedelta(minutes=premeal_minutes), timestamp + pd.Timedelta(minutes=postmeal_minutes)):
            exclusion_reasons.append("unlogged_food_in_window")

        # A carb entry rarely lands in the same 5-minute bin as the bolus, so take every
        # explicit entry from 20 minutes before to 30 minutes after the bolus.
        stated_carbs = float("nan")
        recorded_settings = {column: float("nan") for column in MEAL_RECORD_COLUMNS}
        if "explicit_carb_grams" in ordered.columns:
            nearby = window(timestamp - pd.Timedelta(minutes=20), timestamp + pd.Timedelta(minutes=30))
            carb = pd.to_numeric(nearby["explicit_carb_grams"], errors="coerce")
            positive = carb[carb.gt(0)]
            if not positive.empty:
                stated_carbs = float(positive.sum())
                largest = nearby.loc[positive.idxmax()]
                recorded_settings = {column: _as_float(largest.get(column)) for column in MEAL_RECORD_COLUMNS}
        rows.append(
            {
                "candidate_id": f"first_meal_{candidate_index:04d}",
                "meal_id": f"first_meal_{candidate_index:04d}",
                "date": pd.Timestamp(candidate.date),
                "timestamp": timestamp,
                "segment": str(getattr(candidate, "therapy_segment", "")),
                "candidate_status": "accepted" if not exclusion_reasons else "rejected",
                "included": int(not exclusion_reasons),
                "exclusion_reasons": "|".join(exclusion_reasons),
                "explicit_carb_source_available": int(pd.to_numeric(pd.Series([getattr(candidate, "explicit_carb_source_available", 0)]), errors="coerce").fillna(0.0).iloc[0] > 0),
                "meal_truth_status": str(getattr(candidate, "meal_truth_status", "unavailable")),
                "stated_carbs": None if pd.isna(stated_carbs) else float(stated_carbs),
                **recorded_settings,
                "bolus_units": float(pd.to_numeric(pd.Series([getattr(candidate, "bolus_units", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
                "premeal_glucose": float(premeal_glucose) if np.isfinite(premeal_glucose) else float("nan"),
                "premeal_delta_30m": premeal_delta_30m,
                "premeal_iob": premeal_iob,
                "in_exercise": int(in_exercise),
                "closed_loop_confounding_premeal": int(closed_loop_confounding_premeal),
                "prior_bolus_4h": int(prior_bolus_4h),
                "postmeal_additional_bolus_120m": int(postmeal_additional_bolus_120m),
                "cgm_coverage_fraction": cgm_coverage_fraction,
                "missing_cgm_first_90m": int(missing_cgm_first_90m),
                "window_row_count": int(len(evaluation_window)),
                "source_quality_grade": source_quality_grade,
            }
        )
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def build_first_meal_exclusion_summary(first_meal_windows: pd.DataFrame) -> pd.DataFrame:
    if first_meal_windows.empty:
        return pd.DataFrame([{"reason": "no_candidates", "count": 0}])
    counts: dict[str, int] = {}
    for reason_string in first_meal_windows["exclusion_reasons"].astype(str):
        if not reason_string:
            continue
        for reason in [item for item in reason_string.split("|") if item]:
            counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return pd.DataFrame([{"reason": "accepted_without_exclusions", "count": int(first_meal_windows["included"].sum())}])
    return pd.DataFrame([{"reason": reason, "count": count} for reason, count in sorted(counts.items())])


# Reasons no relaxed tier can override: missing CGM, or a day inside an excluded period.
LATENT_MEAL_CGM_REQUIRED_REASONS = frozenset({"poor_cgm_coverage", "missing_cgm_first_90m", "context_event_excluded"})
LATENT_MEAL_ANALYSIS_TIERS: tuple[dict[str, Any], ...] = (
    {
        "analysis_tier": "clean_reference",
        "analysis_tier_rank": 0,
        "analysis_tier_label": "Clean reference",
        "allowed_reasons": frozenset(),
        "confidence_multiplier": 1.00,
        "uncertainty_multiplier": 1.00,
    },
    {
        "analysis_tier": "workout_context_allowed",
        "analysis_tier_rank": 1,
        "analysis_tier_label": "Workout context allowed",
        "allowed_reasons": frozenset({"exercise"}),
        "confidence_multiplier": 0.88,
        "uncertainty_multiplier": 1.12,
    },
    {
        "analysis_tier": "controller_context_allowed",
        "analysis_tier_rank": 2,
        "analysis_tier_label": "Controller context allowed",
        "allowed_reasons": frozenset({"exercise", "closed_loop_confounding_premeal"}),
        "confidence_multiplier": 0.78,
        "uncertainty_multiplier": 1.22,
    },
    {
        "analysis_tier": "baseline_drift_allowed",
        "analysis_tier_rank": 3,
        "analysis_tier_label": "Baseline drift allowed",
        "allowed_reasons": frozenset({"exercise", "closed_loop_confounding_premeal", "premeal_bg_out_of_range", "steep_premeal_slope"}),
        "confidence_multiplier": 0.70,
        "uncertainty_multiplier": 1.38,
    },
    {
        "analysis_tier": "exploratory_cgm_quality_only",
        "analysis_tier_rank": 4,
        "analysis_tier_label": "Exploratory CGM quality only",
        "allowed_reasons": frozenset(
            {
                "exercise",
                "closed_loop_confounding_premeal",
                "premeal_bg_out_of_range",
                "steep_premeal_slope",
                "postmeal_additional_bolus_120m",
                "recent_bolus_4h",
                "high_premeal_iob",
            }
        ),
        "confidence_multiplier": 0.52,
        "uncertainty_multiplier": 1.70,
    },
)


def _exclusion_reason_set(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    return {item for item in str(value).split("|") if item and item.lower() != "nan"}


def _latent_meal_tier_for_reasons(reasons: set[str]) -> dict[str, Any] | None:
    if reasons & LATENT_MEAL_CGM_REQUIRED_REASONS:
        return None
    for tier in LATENT_MEAL_ANALYSIS_TIERS:
        if reasons.issubset(tier["allowed_reasons"]):
            return tier
    return None


def build_latent_meal_tiered_window_registry(first_meal_windows: pd.DataFrame) -> pd.DataFrame:
    if first_meal_windows.empty:
        out = first_meal_windows.copy()
        for column in [
            "clean_included",
            "tier_included",
            "analysis_tier",
            "analysis_tier_rank",
            "analysis_tier_label",
            "relaxed_exclusion_reasons",
            "hard_exclusion_reasons",
        ]:
            if column not in out.columns:
                out[column] = pd.Series(dtype=object)
        return out
    out = first_meal_windows.copy()
    out["clean_included"] = pd.to_numeric(out.get("included"), errors="coerce").fillna(0).astype(int)
    tier_rows: list[dict[str, Any]] = []
    for row in out.itertuples(index=False):
        reasons = _exclusion_reason_set(getattr(row, "exclusion_reasons", ""))
        tier = _latent_meal_tier_for_reasons(reasons)
        if tier is None:
            tier_rows.append(
                {
                    "tier_included": 0,
                    "analysis_tier": "rejected",
                    "analysis_tier_rank": 99,
                    "analysis_tier_label": "Rejected",
                    "relaxed_exclusion_reasons": "",
                    "hard_exclusion_reasons": "|".join(sorted(reasons)),
                    "candidate_status": "rejected",
                }
            )
            continue
        tier_rank = int(tier["analysis_tier_rank"])
        tier_rows.append(
            {
                "tier_included": 1,
                "analysis_tier": str(tier["analysis_tier"]),
                "analysis_tier_rank": tier_rank,
                "analysis_tier_label": str(tier["analysis_tier_label"]),
                "relaxed_exclusion_reasons": "" if tier_rank == 0 else "|".join(sorted(reasons)),
                "hard_exclusion_reasons": "",
                "candidate_status": "accepted" if tier_rank == 0 else "accepted_relaxed",
            }
        )
    tier_frame = pd.DataFrame(tier_rows, index=out.index)
    for column in tier_frame.columns:
        out[column] = tier_frame[column]
    out["included"] = out["tier_included"].astype(int)
    return out


def build_latent_meal_tier_summary(meal_windows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "analysis_tier",
        "analysis_tier_rank",
        "analysis_tier_label",
        "accepted_windows",
        "clean_windows",
        "relaxed_windows",
        "median_bolus_units",
        "median_premeal_glucose",
        "median_premeal_iob",
        "median_cgm_coverage_fraction",
    ]
    if meal_windows.empty or "analysis_tier" not in meal_windows.columns:
        return pd.DataFrame(columns=columns)
    included = meal_windows.loc[pd.to_numeric(meal_windows.get("included"), errors="coerce").fillna(0).gt(0)].copy()
    if included.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for tier in LATENT_MEAL_ANALYSIS_TIERS:
        tier_name = str(tier["analysis_tier"])
        subset = included.loc[included["analysis_tier"].astype(str).eq(tier_name)].copy()
        rows.append(
            {
                "analysis_tier": tier_name,
                "analysis_tier_rank": int(tier["analysis_tier_rank"]),
                "analysis_tier_label": str(tier["analysis_tier_label"]),
                "accepted_windows": int(len(subset)),
                "clean_windows": int(pd.to_numeric(subset.get("clean_included", pd.Series(dtype=float)), errors="coerce").fillna(0).gt(0).sum()) if not subset.empty else 0,
                "relaxed_windows": int(pd.to_numeric(subset.get("clean_included", pd.Series(dtype=float)), errors="coerce").fillna(0).eq(0).sum()) if not subset.empty else 0,
                "median_bolus_units": _safe_median(subset.get("bolus_units", pd.Series(dtype=float)), default=float("nan")) if not subset.empty else float("nan"),
                "median_premeal_glucose": _safe_median(subset.get("premeal_glucose", pd.Series(dtype=float)), default=float("nan")) if not subset.empty else float("nan"),
                "median_premeal_iob": _safe_median(subset.get("premeal_iob", pd.Series(dtype=float)), default=float("nan")) if not subset.empty else float("nan"),
                "median_cgm_coverage_fraction": _safe_median(subset.get("cgm_coverage_fraction", pd.Series(dtype=float)), default=float("nan")) if not subset.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _render_latent_meal_tier_summary(summary: pd.DataFrame) -> str:
    lines = [
        "# Latent Meal Tier Summary",
        "",
        "| analysis_tier | accepted_windows | relaxed_windows | median_bolus_units | median_premeal_glucose | median_premeal_iob | median_cgm_coverage_fraction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if summary.empty:
        lines.append("| none | 0 | 0 | NA | NA | NA | NA |")
    else:
        for row in summary.sort_values("analysis_tier_rank").itertuples(index=False):
            lines.append(
                f"| {row.analysis_tier} | {int(row.accepted_windows)} | {int(row.relaxed_windows)} | "
                f"{'NA' if pd.isna(row.median_bolus_units) else f'{float(row.median_bolus_units):.2f}'} | "
                f"{'NA' if pd.isna(row.median_premeal_glucose) else f'{float(row.median_premeal_glucose):.1f}'} | "
                f"{'NA' if pd.isna(row.median_premeal_iob) else f'{float(row.median_premeal_iob):.2f}'} | "
                f"{'NA' if pd.isna(row.median_cgm_coverage_fraction) else f'{float(row.median_cgm_coverage_fraction):.3f}'} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `clean_reference` rows satisfy every original clean-window gate.",
            "- Relaxed tiers are research-only sensitivity cohorts and do not create observed carb truth.",
            "- `poor_cgm_coverage` and `missing_cgm_first_90m` remain hard exclusions in every tier.",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_meal_truth_semantics_report(
    dataset: AnalysisReadyHealthDataset,
    research_frame: pd.DataFrame,
) -> str:
    status_counts = research_frame.get("meal_truth_status", pd.Series(dtype=str)).astype(str).value_counts(dropna=False).to_dict() if not research_frame.empty else {}
    lines = [
        "# Meal Truth Semantics Report",
        "",
        f"- explicit_carb_source_available: {bool(getattr(dataset, 'explicit_carb_source_available', False))}",
        f"- observed_explicit_rows: {int(status_counts.get('observed_explicit', 0))}",
        f"- missing_from_source_rows: {int(status_counts.get('missing_from_source', 0))}",
        f"- unavailable_rows: {int(status_counts.get('unavailable', 0))}",
        "",
        "## Interpretation",
        "",
        "- `explicit_carb_grams` and `explicit_meal_event` represent observed explicit carb truth when it exists.",
        "- Legacy `carb_grams`, `meal_event`, and `minutes_since_last_meal` are retained for compatibility but must not be interpreted as observed truth when explicit carb source data is unavailable.",
        "- When Tandem/tconnectsync does not provide explicit carb records, meal truth is marked as `unavailable` and downstream meal research should rely on proxy logic only for cohorting, not for observed truth claims.",
        "",
        "## Row Counts",
        "",
        "| meal_truth_status | count |",
        "| --- | ---: |",
    ]
    if status_counts:
        for status, count in sorted(status_counts.items()):
            lines.append(f"| {status} | {int(count)} |")
    else:
        lines.append("| none | 0 |")
    return "\n".join(lines) + "\n"


def _latent_meal_gate(
    dataset: AnalysisReadyHealthDataset,
    first_meal_windows: pd.DataFrame,
    *,
    research_scope: str,
) -> tuple[pd.DataFrame, str]:
    evaluated_candidates = int(len(first_meal_windows))
    accepted_windows = int(pd.to_numeric(first_meal_windows.get("included"), errors="coerce").fillna(0.0).sum()) if not first_meal_windows.empty else 0
    accepted_fraction = float(accepted_windows / evaluated_candidates) if evaluated_candidates else 0.0
    dataset_source_quality = str(getattr(dataset, "source_quality_status", "unknown") or "unknown")
    source_quality_status = dataset_source_quality if dataset_source_quality not in {"unknown", "good"} else "good" if accepted_windows > 0 else "limited" if evaluated_candidates > 0 else "failed"
    cohort_status = "ready_for_latent_fit" if accepted_windows > 0 else "no_clean_windows_identified"
    gate = pd.DataFrame(
        [
            {
                "research_scope": research_scope,
                "explicit_carb_source_available": bool(getattr(dataset, "explicit_carb_source_available", False)),
                "latent_fit_status": "intentionally_skipped_foundation_mode",
                "evaluated_candidates": evaluated_candidates,
                "accepted_windows": accepted_windows,
                "accepted_fraction": accepted_fraction,
                "source_quality_status": source_quality_status,
                "cohort_status": cohort_status,
            }
        ]
    )
    lines = [
        "# Latent Meal Research Gate",
        "",
        f"- research_scope: {research_scope}",
        f"- explicit_carb_source_available: {bool(getattr(dataset, 'explicit_carb_source_available', False))}",
        "- latent_fit_status: intentionally_skipped_foundation_mode",
        f"- evaluated_candidates: {evaluated_candidates}",
        f"- accepted_windows: {accepted_windows}",
        f"- accepted_fraction: {accepted_fraction:.3f}",
        f"- source_quality_status: {source_quality_status}",
        f"- cohort_status: {cohort_status}",
        "",
        "## Foundation Gate",
        "",
        "| research_scope | explicit_carb_source_available | latent_fit_status | evaluated_candidates | accepted_windows | source_quality_status | cohort_status |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
        f"| {research_scope} | {str(bool(getattr(dataset, 'explicit_carb_source_available', False))).lower()} | intentionally_skipped_foundation_mode | {evaluated_candidates} | {accepted_windows} | {source_quality_status} | {cohort_status} |",
    ]
    return gate, "\n".join(lines) + "\n"


def _render_latent_meal_window_audit(
    first_meal_windows: pd.DataFrame,
    exclusion_summary: pd.DataFrame,
) -> str:
    accepted = first_meal_windows.loc[pd.to_numeric(first_meal_windows.get("included"), errors="coerce").fillna(0.0).gt(0)].copy() if not first_meal_windows.empty else pd.DataFrame()
    lines = [
        "# Which Breakfasts Were Used",
        "",
        f"{int(len(accepted))} of {int(len(first_meal_windows))} first meals of the day passed the quality checks"
        + (
            f"; {int(pd.to_numeric(accepted.get('clean_included', pd.Series(0, index=accepted.index)), errors='coerce').fillna(0).gt(0).sum())} passed every check and the rest passed looser versions."
            if not accepted.empty
            else "."
        ),
        "",
    ]
    if "analysis_tier" in first_meal_windows.columns:
        tier_counts = accepted["analysis_tier"].astype(str).value_counts().to_dict() if not accepted.empty else {}
        lines.extend(["## Quality of the breakfasts used", "", "| quality | breakfasts |", "| --- | ---: |"])
        if tier_counts:
            for tier in LATENT_MEAL_ANALYSIS_TIERS:
                tier_name = str(tier["analysis_tier"])
                lines.append(f"| {plain(tier_name)} | {int(tier_counts.get(tier_name, 0))} |")
        else:
            lines.append("| none | 0 |")
        lines.append("")
    lines.extend(
        [
            "## Why breakfasts were excluded",
            "",
            "One breakfast can have several reasons, so the counts add up to more than the number excluded.",
            "",
            "| reason | breakfasts |",
            "| --- | ---: |",
        ]
    )
    if exclusion_summary.empty:
        lines.append("| none | 0 |")
    else:
        for row in exclusion_summary.sort_values("count", ascending=False).itertuples(index=False):
            lines.append(f"| {plain(row.reason)} | {int(row.count)} |")
    lines.extend(
        [
            "",
            "## First 10 breakfasts used",
            "",
            "| time | quality | glucose before (mg/dL) | insulin on board (U) | CGM coverage | looser checks it needed |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    if accepted.empty:
        lines.append("| none | none | NA | NA | NA | none |")
    else:
        for row in accepted.head(10).itertuples(index=False):
            relaxed = [plain(reason) for reason in str(getattr(row, "relaxed_exclusion_reasons", "") or "").split("|") if reason and reason != "none"]
            lines.append(
                f"| {row.timestamp} | {plain(getattr(row, 'analysis_tier', 'clean_reference'))} | "
                f"{'NA' if pd.isna(row.premeal_glucose) else f'{float(row.premeal_glucose):.0f}'} | "
                f"{'NA' if pd.isna(row.premeal_iob) else f'{float(row.premeal_iob):.2f}'} | "
                f"{float(row.cgm_coverage_fraction) * 100:.0f}% | {'; '.join(relaxed) or 'none'} |"
            )
    return "\n".join(lines) + "\n"


def _safe_median(series: pd.Series, *, default: float) -> float:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return float(default)
    return float(numeric.median())


def _safe_correlation(left: pd.Series, right: pd.Series) -> float:
    frame = pd.DataFrame({"left": pd.to_numeric(left, errors="coerce"), "right": pd.to_numeric(right, errors="coerce")}).dropna()
    if len(frame) < 2:
        return float("nan")
    if frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return float("nan")
    return float(frame["left"].corr(frame["right"]))


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(min(max(value, lower), upper))


def _positive_float(value: Any, name: str) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(numeric) or float(numeric) <= 0:
        raise ValueError(f"{name} must be a positive number")
    return float(numeric)


def _parse_segment_numeric_mapping(value: Any, *, default: Mapping[str, float], name: str) -> dict[str, float]:
    if value is None:
        return {str(key): float(val) for key, val in default.items()}
    if isinstance(value, Mapping):
        out = {str(key): _positive_float(val, f"{name}.{key}") for key, val in value.items()}
        if "default" not in out:
            default_values = [float(val) for val in out.values()]
            out["default"] = float(np.median(default_values)) if default_values else float(next(iter(default.values())))
        return out
    scalar = _positive_float(value, name)
    out = {str(key): scalar for key in default.keys()}
    out["default"] = scalar
    return out


def _segment_value(mapping: Mapping[str, float], segment: str, *, default: float) -> float:
    if segment in mapping:
        return float(mapping[segment])
    if "default" in mapping:
        return float(mapping["default"])
    if mapping:
        return float(np.median([float(value) for value in mapping.values()]))
    return float(default)


def load_latent_meal_therapy_settings(path: str | Path) -> LatentMealTherapySettings:
    settings_path = Path(path)
    if not settings_path.exists():
        raise FileNotFoundError(f"Latent meal therapy settings file not found: {settings_path}")
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Latent meal therapy settings must be a JSON object")
    return LatentMealTherapySettings.from_mapping(payload)


def _safe_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _tandem_scaled_ratio(value: Any) -> float:
    numeric = _positive_float(value, "tandem profile value")
    return numeric / 1000.0 if numeric >= 1000.0 else numeric


def _active_tandem_profile(settings_payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    profiles_payload = settings_payload.get("profiles")
    if not isinstance(profiles_payload, Mapping):
        return None
    profiles = profiles_payload.get("profile")
    if not isinstance(profiles, list):
        return None
    active_idp = profiles_payload.get("activeIdp")
    for profile in profiles:
        if isinstance(profile, Mapping) and profile.get("idp") == active_idp:
            return profile
    for profile in profiles:
        if isinstance(profile, Mapping):
            segments = profile.get("tDependentSegs")
            if isinstance(segments, list) and any(isinstance(seg, Mapping) and float(seg.get("carbRatio") or 0) > 0 for seg in segments):
                return profile
    return None


def _valid_tandem_profile_segments(profile: Mapping[str, Any]) -> list[dict[str, float]]:
    raw_segments = profile.get("tDependentSegs")
    if not isinstance(raw_segments, list):
        return []
    segments: list[dict[str, float]] = []
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, Mapping):
            continue
        carb_ratio = pd.to_numeric(pd.Series([raw_segment.get("carbRatio")]), errors="coerce").iloc[0]
        isf = pd.to_numeric(pd.Series([raw_segment.get("isf")]), errors="coerce").iloc[0]
        target = pd.to_numeric(pd.Series([raw_segment.get("targetBg")]), errors="coerce").iloc[0]
        start = pd.to_numeric(pd.Series([raw_segment.get("startTime")]), errors="coerce").iloc[0]
        if not all(np.isfinite(value) and float(value) > 0 for value in [carb_ratio, isf, target]):
            continue
        if not np.isfinite(start) or float(start) < 0 or float(start) >= 1440:
            continue
        segments.append(
            {
                "start_minute": float(start),
                "carb_ratio": _tandem_scaled_ratio(carb_ratio),
                "isf": float(isf),
                "target_bg": float(target),
            }
        )
    return sorted(segments, key=lambda item: item["start_minute"])


def _profile_value_at_minute(profile_segments: list[dict[str, float]], minute: int, value_key: str) -> float:
    if not profile_segments:
        return float("nan")
    selected = profile_segments[-1]
    for segment in profile_segments:
        if minute >= int(segment["start_minute"]):
            selected = segment
        else:
            break
    return float(selected[value_key])


def _therapy_segment_minutes(segment: TherapySegment) -> list[int]:
    if segment.start_minute < segment.end_minute:
        return list(range(segment.start_minute, segment.end_minute, 5))
    return list(range(segment.start_minute, 1440, 5)) + list(range(0, segment.end_minute, 5))


def _profile_value_for_therapy_segment(profile_segments: list[dict[str, float]], segment: TherapySegment, value_key: str) -> float:
    minutes = _therapy_segment_minutes(segment)
    values = [_profile_value_at_minute(profile_segments, minute, value_key) for minute in minutes]
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.mean())


def _archived_pump_settings(file_name: str, payload: Mapping[str, Any]) -> list[tuple[pd.Timestamp, dict[str, Any]]]:
    """Pump settings from an archived Tandem window, as (time, {"profiles", "maxBolus"}).

    Two archive generations exist: ``pumper.json`` (tconnectsync 3.x, the current Tandem Source API)
    and ``pump_event_metadata.json`` (tconnectsync 2.x, windows collected before mid-2026). Both are
    read into the same shape; profile time segments are exposed as ``tDependentSegs``. Each snapshot is dated by
    when the pump uploaded it, not by the window it was archived with: every window collected on the same day
    holds the same, current snapshot.
    """
    found: list[tuple[pd.Timestamp, dict[str, Any]]] = []
    if file_name == "pumper.json":
        selected = set(payload.get("selected_pump_serials") or [])
        for pump in (payload.get("pumper") or {}).get("pumps") or []:
            if selected and str(pump.get("serialNumber")) not in selected:
                continue
            settings = pump.get("settings") or {}
            details = settings.get("details") or {}
            profiles = details.get("profiles")
            if not isinstance(profiles, Mapping):
                continue
            profiles = {
                **profiles,
                "profile": [
                    {**profile, "tDependentSegs": profile.get("timeDependentSegments")}
                    for profile in profiles.get("profile") or []
                    if isinstance(profile, Mapping)
                ],
            }
            uploaded = pd.to_datetime(settings.get("uploadedTimeStamp"), errors="coerce", utc=True)
            when = uploaded.tz_localize(None) if pd.notna(uploaded) else pd.Timestamp.min
            max_bolus = (details.get("globalMaxBolusSettings") or {}).get("maxBolus", 25000)
            found.append((pd.Timestamp(when), {"profiles": profiles, "maxBolus": max_bolus}))
    else:
        last_upload = (payload.get("selected_pump") or {}).get("lastUpload")
        settings = last_upload.get("settings") if isinstance(last_upload, Mapping) else None
        if isinstance(settings, Mapping):
            uploaded = pd.to_datetime(last_upload.get("lastUploadedAt"), errors="coerce", utc=True)
            when = uploaded.tz_localize(None) if pd.notna(uploaded) else pd.Timestamp.min
            max_bolus = (settings.get("pumpSettings") or {}).get("maxBolus", 25000)
            found.append((pd.Timestamp(when), {"profiles": settings.get("profiles"), "maxBolus": max_bolus}))
    return found


def extract_latent_meal_therapy_settings_from_tandem_raw(
    raw_root: str | Path,
    *,
    segments: tuple[TherapySegment, ...] | None = None,
) -> LatentMealTherapySettings:
    root = Path(raw_root).expanduser()
    search_roots = [root]
    if (root / "tconnectsync").exists():
        search_roots.insert(0, root / "tconnectsync")
    candidates: list[tuple[pd.Timestamp, Path, Mapping[str, Any], Mapping[str, Any]]] = []
    for search_root in search_roots:
        for path in [*search_root.rglob("pumper.json"), *search_root.rglob("pump_event_metadata.json")]:
            payload = _safe_json_object(path)
            if payload is None:
                continue
            for sort_time, settings_payload in _archived_pump_settings(path.name, payload):
                active_profile = _active_tandem_profile(settings_payload)
                if active_profile is not None:
                    candidates.append((sort_time, path, settings_payload, active_profile))
    if not candidates:
        raise FileNotFoundError(
        f"No Tandem pump profile settings found under {root}. The full latent-meal scope needs the programmed I:C ratio "
        "and correction factor: either archive a Tandem window (its raw/pumper.json holds the settings), or pass "
        "--therapy-settings path/to/therapy_settings.json (format in the README)."
    )
    _, _, settings_payload, active_profile = sorted(candidates, key=lambda item: (item[0], str(item[1])))[-1]
    profile_segments = _valid_tandem_profile_segments(active_profile)
    if not profile_segments:
        raise ValueError("Tandem pump profile did not contain any valid time-dependent therapy segments")
    therapy_segments = segments or parse_therapy_segments()
    icr = {
        segment.name: _profile_value_for_therapy_segment(profile_segments, segment, "carb_ratio")
        for segment in therapy_segments
    }
    correction_factor = {
        segment.name: _profile_value_for_therapy_segment(profile_segments, segment, "isf")
        for segment in therapy_segments
    }
    targets = [
        _profile_value_for_therapy_segment(profile_segments, segment, "target_bg")
        for segment in therapy_segments
    ]
    insulin_duration = int(_positive_float(active_profile.get("insulinDuration", 300), "insulinDuration"))
    max_bolus_raw = _positive_float(active_profile.get("maxBolus", settings_payload.get("maxBolus", 25000)), "maxBolus")
    max_bolus_units = max_bolus_raw / 1000.0 if max_bolus_raw >= 1000.0 else max_bolus_raw
    max_plausible_meal_carbs = max(30.0, max_bolus_units * max(icr.values()))
    return LatentMealTherapySettings(
        icr_grams_per_unit=icr,
        correction_factor_mgdl_per_unit=correction_factor,
        target_glucose_mgdl=float(np.nanmedian(targets)),
        insulin_action_minutes=insulin_duration,
        carb_absorption_minutes_prior=180,
        max_plausible_meal_carbs=float(max_plausible_meal_carbs),
    )


def _settings_for_estimation(settings: LatentMealTherapySettings | None) -> LatentMealTherapySettings:
    return settings if settings is not None else LatentMealTherapySettings.default()


def _attach_resolved_settings(window_metrics: pd.DataFrame, settings: LatentMealTherapySettings) -> pd.DataFrame:
    """Add each meal's own I:C and correction factor (``resolved_icr``, ``resolved_cf``).

    The values come from ``_resolve_meal_settings`` (bolus calculator, then pump profile history), so a meal is
    judged against the settings its pump held at the time and not against the latest profile snapshot.
    """
    frame = window_metrics.copy()
    frame["resolved_icr"] = np.nan
    frame["resolved_cf"] = np.nan
    if frame.empty or "segment" not in frame:
        return frame
    for segment, group in frame.groupby(frame["segment"].astype(str), sort=True):
        ordered = group.sort_values("timestamp")
        resolved = _resolve_meal_settings(list(ordered.itertuples(index=False)), str(segment), settings)
        frame.loc[ordered.index, "resolved_icr"] = [icr for icr, _, _ in resolved]
        frame.loc[ordered.index, "resolved_cf"] = [cf for _, cf, _ in resolved]
    return frame


def _settings_series(
    window_metrics: pd.DataFrame,
    settings: LatentMealTherapySettings,
    *,
    field: str,
) -> pd.Series:
    resolved_column = {"icr": "resolved_icr", "correction_factor": "resolved_cf"}.get(field)
    if resolved_column is not None and resolved_column in window_metrics:
        return pd.to_numeric(window_metrics[resolved_column], errors="raise").astype(float)
    segments = window_metrics.get("segment", pd.Series("default", index=window_metrics.index, dtype=str)).astype(str)
    if field == "icr":
        values = [settings.icr_for_segment(segment) for segment in segments]
    elif field == "correction_factor":
        values = [settings.correction_factor_for_segment(segment) for segment in segments]
    else:
        raise ValueError(f"Unsupported settings field: {field}")
    return pd.Series(values, index=window_metrics.index, dtype=float)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return float("nan")
    ordered = np.argsort(values[valid])
    sorted_values = values[valid][ordered]
    sorted_weights = weights[valid][ordered]
    cumulative = np.cumsum(sorted_weights)
    cumulative = cumulative / cumulative[-1]
    return float(np.interp(float(quantile), cumulative, sorted_values))


def _accepted_first_meal_window_metrics(
    research_frame: pd.DataFrame,
    meal_event_registry: pd.DataFrame,
    meal_windows: pd.DataFrame,
    *,
    premeal_minutes: int = 30,
    postmeal_minutes: int = 180,
) -> pd.DataFrame:
    columns = [
        "meal_id",
        "candidate_id",
        "timestamp",
        "segment",
        "evidence_source",
        "explicit_carb_source_available",
        "meal_truth_status",
        "stated_carbs",
        *MEAL_RECORD_COLUMNS,
        "bolus_units",
        "meal_proxy_confidence",
        "premeal_glucose",
        "premeal_iob",
        "observed_peak_glucose",
        "observed_peak_delta",
        "observed_auc_above_baseline",
        "observed_mean_glucose_180m",
        "observed_cgm_points",
        "cgm_coverage_fraction",
        "source_quality_grade",
        "sleep_deficit_flag",
        "recent_exercise_context",
        "recent_workout_context",
        "recent_workout_12h_context",
        "light_activity_context",
        "moderate_activity_context",
        "high_activity_context",
        "health_activity_roll_sum_60m",
        "post_workout_meal_context",
        "overnight_hrv_interaction",
        "in_sleep_heart_rate_interaction",
        "analysis_tier",
        "analysis_tier_rank",
        "analysis_tier_label",
        "relaxed_exclusion_reasons",
        "original_exclusion_reasons",
    ]
    if research_frame.empty or meal_windows.empty:
        return pd.DataFrame(columns=columns)
    accepted = meal_windows.loc[pd.to_numeric(meal_windows.get("included"), errors="coerce").fillna(0.0).gt(0)].copy()
    if accepted.empty:
        return pd.DataFrame(columns=columns)

    ordered = research_frame.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="coerce")
    ordered = ordered.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    events = meal_event_registry.copy()
    if not events.empty:
        events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce")
        events = events.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for row in accepted.itertuples(index=False):
        anchor_time = pd.Timestamp(row.timestamp)
        pre = ordered.loc[
            ordered["timestamp"].ge(anchor_time - pd.Timedelta(minutes=premeal_minutes))
            & ordered["timestamp"].lt(anchor_time)
        ].copy()
        post = ordered.loc[
            ordered["timestamp"].ge(anchor_time)
            & ordered["timestamp"].le(anchor_time + pd.Timedelta(minutes=postmeal_minutes))
        ].copy()
        if post.empty:
            continue
        anchor_candidates = ordered.loc[
            ordered["timestamp"].ge(anchor_time - pd.Timedelta(minutes=5))
            & ordered["timestamp"].le(anchor_time + pd.Timedelta(minutes=5))
        ].copy()
        anchor_row = anchor_candidates.iloc[0] if not anchor_candidates.empty else post.iloc[0]

        pre_glucose_series = pd.to_numeric(pre.get("glucose"), errors="coerce").dropna()
        baseline_glucose = (
            float(row.premeal_glucose)
            if np.isfinite(pd.to_numeric(pd.Series([row.premeal_glucose]), errors="coerce").iloc[0])
            else float(pre_glucose_series.iloc[-1]) if not pre_glucose_series.empty else float(pd.to_numeric(pd.Series([anchor_row.get("glucose")]), errors="coerce").fillna(np.nan).iloc[0])
        )
        post_glucose = pd.to_numeric(post.get("glucose"), errors="coerce")
        if post_glucose.notna().any():
            peak_idx = int(post_glucose.idxmax())
            peak_glucose = float(post_glucose.loc[peak_idx])
            peak_delta = float(peak_glucose - baseline_glucose) if np.isfinite(baseline_glucose) else float("nan")
            auc_above_baseline = float((post_glucose.sub(baseline_glucose).clip(lower=0.0).fillna(0.0) * (5.0 / 60.0)).sum())
            mean_glucose_180m = float(post_glucose.mean())
        else:
            peak_glucose = float("nan")
            peak_delta = float("nan")
            auc_above_baseline = float("nan")
            mean_glucose_180m = float("nan")

        evidence_source = "proxy_only"
        if not events.empty:
            same_day = events.loc[events["timestamp"].dt.normalize().eq(anchor_time.normalize())].copy()
            if not same_day.empty:
                nearest_idx = (same_day["timestamp"] - anchor_time).abs().idxmin()
                if abs(pd.Timedelta(same_day.loc[nearest_idx, "timestamp"] - anchor_time)) <= pd.Timedelta(minutes=90):
                    evidence_source = str(same_day.loc[nearest_idx, "evidence_source"])
        if evidence_source == "proxy_only" and pd.notna(row.stated_carbs) and float(row.stated_carbs) > 0:
            evidence_source = "explicit_logged"

        rows.append(
            {
                "meal_id": str(row.meal_id),
                "candidate_id": str(row.candidate_id),
                "timestamp": anchor_time,
                "segment": str(row.segment),
                "evidence_source": evidence_source,
                "explicit_carb_source_available": int(row.explicit_carb_source_available),
                "meal_truth_status": str(row.meal_truth_status),
                "stated_carbs": None if pd.isna(row.stated_carbs) else float(row.stated_carbs),
                **{column: float(pd.to_numeric(pd.Series([getattr(row, column, float("nan"))]), errors="coerce").iloc[0]) for column in MEAL_RECORD_COLUMNS},
                "bolus_units": float(row.bolus_units),
                "meal_proxy_confidence": float(pd.to_numeric(pd.Series([anchor_row.get("meal_proxy_confidence")]), errors="coerce").fillna(0.0).iloc[0]),
                "premeal_glucose": baseline_glucose,
                "premeal_iob": float(row.premeal_iob),
                "observed_peak_glucose": peak_glucose,
                "observed_peak_delta": peak_delta,
                "observed_auc_above_baseline": auc_above_baseline,
                "observed_mean_glucose_180m": mean_glucose_180m,
                "observed_cgm_points": int(post_glucose.notna().sum()),
                "cgm_coverage_fraction": float(row.cgm_coverage_fraction),
                "source_quality_grade": str(row.source_quality_grade),
                "sleep_deficit_flag": int(pd.to_numeric(pd.Series([anchor_row.get("sleep_deficit_flag")]), errors="coerce").fillna(0.0).iloc[0] > 0),
                "recent_exercise_context": int(pd.to_numeric(pd.Series([anchor_row.get("recent_exercise_context")]), errors="coerce").fillna(0.0).iloc[0] > 0),
                "recent_workout_context": int(pd.to_numeric(pd.Series([anchor_row.get("recent_workout_context")]), errors="coerce").fillna(0.0).iloc[0] > 0),
                "recent_workout_12h_context": int(pd.to_numeric(pd.Series([anchor_row.get("recent_workout_12h_context")]), errors="coerce").fillna(0.0).iloc[0] > 0),
                "light_activity_context": int(pd.to_numeric(pd.Series([anchor_row.get("light_activity_context")]), errors="coerce").fillna(0.0).iloc[0] > 0),
                "moderate_activity_context": int(pd.to_numeric(pd.Series([anchor_row.get("moderate_activity_context")]), errors="coerce").fillna(0.0).iloc[0] > 0),
                "high_activity_context": int(pd.to_numeric(pd.Series([anchor_row.get("high_activity_context")]), errors="coerce").fillna(0.0).iloc[0] > 0),
                "health_activity_roll_sum_60m": float(pd.to_numeric(pd.Series([anchor_row.get("health_activity_roll_sum_60m")]), errors="coerce").fillna(0.0).iloc[0]),
                "post_workout_meal_context": int(pd.to_numeric(pd.Series([anchor_row.get("post_workout_meal_context")]), errors="coerce").fillna(0.0).iloc[0] > 0),
                "overnight_hrv_interaction": float(pd.to_numeric(pd.Series([anchor_row.get("overnight_hrv_interaction")]), errors="coerce").fillna(0.0).iloc[0]),
                "in_sleep_heart_rate_interaction": float(pd.to_numeric(pd.Series([anchor_row.get("in_sleep_heart_rate_interaction")]), errors="coerce").fillna(0.0).iloc[0]),
                "analysis_tier": str(getattr(row, "analysis_tier", "clean_reference")),
                "analysis_tier_rank": int(pd.to_numeric(pd.Series([getattr(row, "analysis_tier_rank", 0)]), errors="coerce").fillna(0).iloc[0]),
                "analysis_tier_label": str(getattr(row, "analysis_tier_label", "Clean reference")),
                "relaxed_exclusion_reasons": str(getattr(row, "relaxed_exclusion_reasons", "") or ""),
                "original_exclusion_reasons": str(getattr(row, "exclusion_reasons", "") or ""),
            }
        )
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def _scheduled_basal_units_per_row(
    ordered: pd.DataFrame,
    *,
    slot_minutes: int = 30,
    window_days: int = 43,
    min_days: int = 5,
) -> pd.Series:
    """Estimate the scheduled (un-modulated) basal delivery for every row.

    The frame only records what was delivered. The pump's schedule repeats daily while
    closed-loop adjustments differ from day to day, so the median delivery at the same
    time of day across nearby days is a robust estimate of the schedule. Rows without at
    least ``min_days`` comparable days in the centred ``window_days`` window are NaN.
    """
    delivered = pd.to_numeric(ordered["basal_units_delivered"], errors="coerce")
    timestamps = ordered["timestamp"]
    slot = (timestamps.dt.hour * 60 + timestamps.dt.minute) // int(slot_minutes)
    date = timestamps.dt.normalize()
    per_day = delivered.groupby([date, slot]).median().unstack(level=1).sort_index()
    if per_day.empty:
        return pd.Series(np.nan, index=ordered.index, dtype=float)
    per_day = per_day.reindex(pd.date_range(per_day.index.min(), per_day.index.max(), freq="D"))
    schedule = per_day.rolling(window=int(window_days), center=True, min_periods=int(min_days)).median()
    lookup = schedule.stack(future_stack=True)
    keys = pd.MultiIndex.from_arrays([date, slot])
    return pd.Series(lookup.reindex(keys).to_numpy(dtype=float), index=ordered.index, dtype=float)


def _first_meal_window_trajectories(
    research_frame: pd.DataFrame,
    window_metrics: pd.DataFrame,
    *,
    programmed_units: pd.Series | None = None,
    missing_basal_windows: set[str] | None = None,
    postmeal_minutes: int = 180,
    basal_reference_minutes: int = 180,
    insulin_action_minutes: int = 300,
) -> dict[str, MealWindowTrajectory]:
    """Extract the observed post-meal CGM trace and absorbed insulin for each accepted window.

    Bolus insulin absorbed since the anchor is reconstructed from the frame's own IOB
    bookkeeping: ``iob(anchor) + boluses delivered after the anchor - iob(t)``. Windows
    whose insulin columns are missing or not physically consistent (absorbed bolus insulin
    that goes down) are left out, and the caller falls back to the summary-statistic
    estimator for them.

    Closed-loop basal modulation is added on top when ``basal_units_delivered`` is present:
    delivery above or below the median rate of the ``basal_reference_minutes`` before the
    meal is treated as extra (or withheld) insulin and pushed through the action curve.
    """
    trajectories: dict[str, MealWindowTrajectory] = {}
    required = {"timestamp", "glucose", "iob_units", "bolus_units"}
    if research_frame.empty or window_metrics.empty or not required.issubset(research_frame.columns):
        return trajectories
    ordered = research_frame.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="coerce")
    ordered = ordered.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    scheduled_basal = _scheduled_basal_units_per_row(ordered) if "basal_units_delivered" in ordered.columns else None
    if programmed_units is not None:
        scheduled_basal = ordered["timestamp"].map(programmed_units)
    for row in window_metrics.itertuples(index=False):
        anchor_time = pd.Timestamp(row.timestamp)
        post = ordered.loc[
            ordered["timestamp"].ge(anchor_time)
            & ordered["timestamp"].le(anchor_time + pd.Timedelta(minutes=postmeal_minutes))
        ]
        if post.empty:
            continue
        if programmed_units is not None and scheduled_basal.loc[post.index].isna().any():
            if missing_basal_windows is not None:
                missing_basal_windows.add(str(row.meal_id))
            continue
        iob = pd.to_numeric(post["iob_units"], errors="coerce")
        bolus = pd.to_numeric(post["bolus_units"], errors="coerce").fillna(0.0)
        if iob.isna().any():
            continue
        later_bolus = bolus.cumsum() - float(bolus.iloc[0])
        absorbed = float(iob.iloc[0]) + later_bolus - iob
        if (absorbed.diff().dropna() < -0.05).any() or absorbed.min() < -0.05:
            continue
        if scheduled_basal is not None:
            delivered = pd.to_numeric(post["basal_units_delivered"], errors="coerce")
            reference = scheduled_basal.loc[post.index]
            if programmed_units is not None and (reference.isna().any() or delivered.isna().any()):
                continue
            if reference.isna().any():
                # Too few comparable days to infer the schedule: fall back to the pre-meal rate.
                premeal = pd.to_numeric(
                    ordered.loc[
                        ordered["timestamp"].ge(anchor_time - pd.Timedelta(minutes=basal_reference_minutes))
                        & ordered["timestamp"].lt(anchor_time),
                        "basal_units_delivered",
                    ],
                    errors="coerce",
                ).dropna()
                reference = pd.Series(float(premeal.median()) if len(premeal) >= 12 else np.nan, index=post.index)
            if delivered.notna().all() and reference.notna().all():
                post_minutes = (post["timestamp"] - anchor_time).dt.total_seconds().div(60.0).to_numpy(dtype=float)
                excess = delivered.to_numpy(dtype=float) - reference.to_numpy(dtype=float)
                absorbed = absorbed + absorbed_units_from_deliveries(
                    post_minutes,
                    excess,
                    post_minutes,
                    insulin_action_minutes=insulin_action_minutes,
                )
        glucose = pd.to_numeric(post["glucose"], errors="coerce")
        observed = glucose.notna()
        if "missing_cgm" in post.columns:
            observed &= pd.to_numeric(post["missing_cgm"], errors="coerce").fillna(1.0).eq(0)
        if not observed.any():
            continue
        minutes = (post.loc[observed, "timestamp"] - anchor_time).dt.total_seconds().div(60.0)
        trajectory = MealWindowTrajectory(
            minutes=minutes.to_numpy(dtype=float),
            glucose=glucose.loc[observed].to_numpy(dtype=float),
            insulin_absorbed_units=absorbed.loc[observed].to_numpy(dtype=float),
        )
        if trajectory.is_usable():
            trajectories[str(row.meal_id)] = trajectory
    return trajectories


EFFECTIVE_ICR_EVIDENCE_COLUMNS = [
    "analysis",
    "logged_food_scale",
    "logged_food_pairs",
    "segment",
    "programmed_icr",
    "settings_source",
    "period_start",
    "period_end",
    "is_current_setting",
    "pump_serial",
    "profile_id",
    "pump_segment_start_minutes",
    "logged_meal_windows",
    "n_meals",
    "sensitivity_ratio",
    "sensitivity_ratio_lower_90",
    "sensitivity_ratio_upper_90",
    "effective_icr",
    "effective_icr_lower_90",
    "effective_icr_upper_90",
    "between_meal_log_sd",
    "status",
]


# Logged food entries (for example Bevel) are a second, noisy estimate of a meal's carbs. They are
# never treated as truth: the headline I:C stays anchored to the pump entries the insulin was dosed on.
LOGGED_FOOD_MATCH_MINUTES = 30
HIGH_FAT_GRAMS = 20.0
HIGH_PROTEIN_GRAMS = 25.0
LOGGED_FOOD_COLUMNS = ("logged_food_carb_grams", "logged_food_protein_grams", "logged_food_fat_grams", "high_fat_protein")


def _logged_food_near(nutrition: pd.DataFrame, when: pd.Timestamp) -> pd.DataFrame:
    stamps = pd.to_datetime(nutrition["timestamp"], errors="coerce")
    window = pd.Timedelta(minutes=LOGGED_FOOD_MATCH_MINUTES)
    return nutrition.loc[stamps.between(when - window, when + window)]


def _attach_logged_food(window_metrics: pd.DataFrame, nutrition: pd.DataFrame | None) -> pd.DataFrame:
    """Add the logged food (carbs, protein, fat) within 30 minutes of each meal, and the high fat/protein flag."""
    out = window_metrics.copy()
    for column in LOGGED_FOOD_COLUMNS:
        out[column] = np.nan if column != "high_fat_protein" else False
    if nutrition is None or nutrition.empty or out.empty:
        return out
    for index, row in out.iterrows():
        nearby = _logged_food_near(nutrition, pd.Timestamp(row["timestamp"]))
        if nearby.empty:
            continue
        carbs, protein, fat = (float(pd.to_numeric(nearby[column], errors="coerce").sum()) for column in ("carb_grams", "protein_grams", "fat_grams"))
        out.at[index, "logged_food_carb_grams"] = carbs if carbs > 0 else np.nan
        out.at[index, "logged_food_protein_grams"] = protein
        out.at[index, "logged_food_fat_grams"] = fat
        out.at[index, "high_fat_protein"] = bool(fat >= HIGH_FAT_GRAMS or protein >= HIGH_PROTEIN_GRAMS)
    return out


def _logged_food_scale(research_frame: pd.DataFrame, nutrition: pd.DataFrame | None) -> tuple[float, int]:
    """Median ratio of logged-food carbs to pump-entered carbs over every meal that has both (any time of day).

    Dividing logged-food carbs by this puts them on the pump-entry scale, so combining the two can reduce
    noise without shifting the overall level.
    """
    if nutrition is None or nutrition.empty or "explicit_carb_grams" not in research_frame:
        return float("nan"), 0
    pump = research_frame.loc[pd.to_numeric(research_frame["explicit_carb_grams"], errors="coerce").gt(0), ["timestamp", "explicit_carb_grams"]].copy()
    pump["timestamp"] = pd.to_datetime(pump["timestamp"], errors="coerce")
    ratios = []
    for row in nutrition.itertuples(index=False):
        food = float(pd.to_numeric(pd.Series([row.carb_grams]), errors="coerce").iloc[0])
        if not np.isfinite(food) or food <= 0:
            continue
        when = pd.Timestamp(row.timestamp)
        near = pump.loc[pump["timestamp"].between(when - pd.Timedelta(minutes=LOGGED_FOOD_MATCH_MINUTES), when + pd.Timedelta(minutes=LOGGED_FOOD_MATCH_MINUTES))]
        entered = float(pd.to_numeric(near["explicit_carb_grams"], errors="coerce").sum())
        if entered > 0:
            ratios.append(food / entered)
    if len(ratios) < MIN_LOGGED_MEALS_FOR_ICR:
        return float("nan"), len(ratios)
    return float(np.exp(np.median(np.log(ratios)))), len(ratios)


# A calculator correction smaller than this is too coarsely rounded (0.01 U) to recover the correction factor from.
RECOVERED_CF_MIN_CORRECTION_UNITS = 0.3
RECOVERED_CF_MIN_MEALS = 3


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


SETTINGS_SOURCES = (
    "bolus calculator",
    "pump profile history",
    "recovered from the bolus calculator",
    "carried from nearby meals",
    "pump profile",
)


def _resolve_meal_settings(rows: list[Any], segment: str, settings: LatentMealTherapySettings) -> list[tuple[float, float, str]]:
    """(I:C, correction factor, source) in force for each meal of one time-of-day segment, in time order.

    Each value comes from the first of these that has it; the source names the weakest one used.

    1. **Bolus calculator.** The I:C the calculator recorded (t:slim), or recovered from the food portion of the
       bolus (Mobi, ``acquisition._calculator_carb_ratio``); the correction factor the t:slim recorded.
    2. **Pump profile history.** The segment in force in the profile active at the meal, replayed from the
       pump's profile edits (``pump_profiles``), while the archive shows those edits.
    3. **Recovered from the bolus calculator.** The Mobi records the correction factor and target as 0. The
       calculator's correction is (glucose - target) / CF - insulin on board, so within a run of meals on one
       pump and profile with one I:C, correction + insulin on board is a straight line in glucose with slope
       1 / CF. Over meals with a correction of at least 0.3 U, a Theil-Sen fit finds the majority setting and a
       least-squares refit of the meals within 5% of it gives the run's CF and target; a meal whose own
       (glucose - target) / (correction + insulin on board) is more than 5% from the run's CF keeps its own
       value (a setting changed inside the run).
    4. **Carried from nearby meals.** A meal with no value takes that of the nearest earlier meal on the same
       pump and profile, or the nearest later one; a profile switch ends the carry, and so does the end of the
       archive's view of profile edits.
    5. **Pump profile.** Only a value that no meal on that pump and profile has falls back to the programmed
       profile, and only that value: a meal's own I:C is kept when its correction factor is missing.
    """
    from scipy.stats import theilslopes

    def column(name: str) -> list[float]:
        return [_as_float(getattr(row, name, np.nan)) for row in rows]

    frame = pd.DataFrame(
        {
            "pump": column("pump_serial"),
            "profile": column("profile_id"),
            "visible": column("profile_edits_visible"),
            "icr": column("carb_ratio_g_per_unit"),
            "cf": column("isf_mgdl_per_unit"),
            "profile_icr": column("profile_carb_ratio_g_per_unit"),
            "profile_cf": column("profile_isf_mgdl_per_unit"),
            "glucose": column("calculator_glucose_mgdl"),
            "iob": column("calculator_iob_units"),
            "correction": column("correction_bolus_units"),
        }
    )
    for name in ("icr", "cf", "profile_icr", "profile_cf"):
        frame.loc[~frame[name].gt(0), name] = np.nan
    frame["pump"] = frame["pump"].ffill().bfill()
    frame["profile"] = frame["profile"].fillna(-1.0)  # no profile switch seen yet: one unknown profile
    frame["visible"] = frame["visible"].fillna(-1.0)
    rank = {source: k for k, source in enumerate(SETTINGS_SOURCES)}
    icr_source = pd.Series(np.where(frame["icr"].notna(), rank["bolus calculator"], np.nan), index=frame.index)
    cf_source = pd.Series(np.where(frame["cf"].notna(), rank["bolus calculator"], np.nan), index=frame.index)
    from_history = frame["icr"].isna() & frame["profile_icr"].notna()
    frame.loc[from_history, "icr"] = frame.loc[from_history, "profile_icr"]
    icr_source[from_history] = rank["pump profile history"]
    from_history = frame["cf"].isna() & frame["profile_cf"].notna()
    frame.loc[from_history, "cf"] = frame.loc[from_history, "profile_cf"]
    cf_source[from_history] = rank["pump profile history"]

    # A value is carried only within one pump and profile, and not from a time the archive shows the profile's edits
    # into a time it does not (the profile may have been edited unseen in between).
    keys = ["pump", "profile", "visible"]
    # A carry stays inside one unbroken stretch of meals on the same pump, profile and edit visibility: a switch
    # to another profile and back starts a new stretch, so values never cross it.
    stretch = frame[keys].ne(frame[keys].shift()).any(axis=1).cumsum()
    carried_icr = frame.groupby(stretch)["icr"].transform(lambda s: s.ffill().bfill())
    icr_source[frame["icr"].isna() & carried_icr.notna()] = rank["carried from nearby meals"]
    frame["icr"] = carried_icr
    frame["run"] = (frame[[*keys, "icr"]].ne(frame[[*keys, "icr"]].shift())).any(axis=1).cumsum()

    frame["cf_recovered"] = np.nan
    for _, run in frame.loc[frame["cf"].isna()].groupby("run"):
        usable = run.loc[run["correction"].ge(RECOVERED_CF_MIN_CORRECTION_UNITS) & run["glucose"].notna() & run["iob"].notna()]
        if len(usable) < RECOVERED_CF_MIN_MEALS or usable["glucose"].nunique() < 2:
            continue
        units = usable["correction"] + usable["iob"]
        slope, intercept, _, _ = theilslopes(units, usable["glucose"])
        if not slope > 0:
            continue
        # The robust fit finds the majority setting; refitting only the breakfasts close to it removes the pull of
        # a minority with another CF, and least squares is exact for a calculator that rounds to 0.01 U.
        own_cf = (usable["glucose"] + intercept / slope) / units
        core = (own_cf / own_cf.median() - 1.0).abs() <= 0.05
        if core.sum() >= RECOVERED_CF_MIN_MEALS and usable.loc[core, "glucose"].nunique() >= 2:
            slope, intercept = np.polyfit(usable.loc[core, "glucose"], units[core], 1)
        run_cf, target = 1.0 / slope, -intercept / slope
        own_cf = (usable["glucose"] - target) / units
        frame.loc[usable.index, "cf_recovered"] = np.where((own_cf / run_cf - 1.0).abs() <= 0.05, round(run_cf), own_cf.round())
    recovered = frame["cf"].isna() & frame["cf_recovered"].notna()
    frame.loc[recovered, "cf"] = frame.loc[recovered, "cf_recovered"]
    cf_source[recovered] = rank["recovered from the bolus calculator"]
    carried_cf = frame.groupby(stretch)["cf"].transform(lambda s: s.ffill().bfill())
    cf_source[frame["cf"].isna() & carried_cf.notna()] = rank["carried from nearby meals"]
    frame["cf"] = carried_cf

    if frame["icr"].isna().any():
        icr_source[frame["icr"].isna()] = rank["pump profile"]
        frame["icr"] = frame["icr"].fillna(float(settings.icr_for_segment(segment)))
    if frame["cf"].isna().any():
        cf_source[frame["cf"].isna()] = rank["pump profile"]
        frame["cf"] = frame["cf"].fillna(float(settings.correction_factor_for_segment(segment)))
    return [
        (float(icr), float(cf), SETTINGS_SOURCES[int(max(a, b))])
        for icr, cf, a, b in zip(frame["icr"], frame["cf"], icr_source, cf_source)
    ]


def _had_correction_bolus(meal: dict[str, Any]) -> bool:
    """True when the bolus calculator recorded a correction portion (an unrecorded one counts as none)."""
    value = _as_float(getattr(meal["row"], "correction_bolus_units", np.nan))
    return bool(np.isfinite(value) and value > 0.05)


# Re-pools of a period with fewer or different meals. A proposal needs these to agree with the primary on direction.
AGREEMENT_CHECKS: tuple[tuple[str, Any], ...] = (
    ("without looser tiers (clean, workout and controller context only)", lambda rank, meal: rank <= 2),
    ("clean reference breakfasts only", lambda rank, meal: rank == 0),
    ("without correction boluses", lambda rank, meal: not _had_correction_bolus(meal)),
)


def _pool_label(window_metrics: pd.DataFrame) -> pd.Series:
    """The group a meal's evidence is pooled in: the pump segment covering it (a setting governs every meal in its
    segment, whichever meal of the day it is), or the therapy segment when the archive does not show the pump's."""
    start = pd.to_numeric(window_metrics.get("profile_segment_start_minutes", pd.Series(np.nan, index=window_metrics.index)), errors="coerce")
    therapy = window_metrics["segment"].astype(str)
    label = start.map(lambda minute: "" if not np.isfinite(minute) else f"{int(minute) // 60:02d}:{int(minute) % 60:02d} segment")
    return label.where(label.ne(""), therapy)


def _unknown_as(value: float, unknown: float = -1.0) -> float:
    """A group key for a value the archive does not show (NaN never equals itself, so it cannot be a key)."""
    return unknown if not np.isfinite(value) else float(value)


def _effective_icr_evidence(
    window_metrics: pd.DataFrame,
    trajectories: Mapping[str, MealWindowTrajectory],
    *,
    settings: LatentMealTherapySettings,
    bevel_scale: float = float("nan"),
    bevel_pairs: int = 0,
    support: list[dict[str, Any]] | None = None,
    missing_basal_windows: set[str] | None = None,
) -> pd.DataFrame:
    """Effective I:C from accepted windows with a logged carb count, per segment and settings period.

    Each meal is compared with the I:C and correction factor in force for that meal, so a change of
    pump settings does not bias the comparison. Meals are then pooled within each settings period (a
    run of meals with the same programmed I:C); the most recent period is the current setting.

    The primary analysis uses the pump-entered carbs. Two sensitivity analyses rerun it: without
    breakfasts that logged food marks as high in fat or protein (late absorption the estimator cannot
    see), and with each meal's carbs replaced by the geometric mean of the pump entry and the logged-food
    estimate put on the pump scale (``bevel_scale``).
    """
    if window_metrics.empty:
        return pd.DataFrame(columns=EFFECTIVE_ICR_EVIDENCE_COLUMNS)

    def pump_carbs(row: Any) -> float:
        return float(pd.to_numeric(pd.Series([row.stated_carbs]), errors="coerce").fillna(np.nan).iloc[0])

    def combined_carbs(row: Any) -> float:
        entered = pump_carbs(row)
        food = float(pd.to_numeric(pd.Series([getattr(row, "logged_food_carb_grams", np.nan)]), errors="coerce").iloc[0])
        if not np.isfinite(entered) or entered <= 0 or not np.isfinite(food) or food <= 0:
            return entered
        return float(np.sqrt(entered * food / bevel_scale))

    analyses: list[tuple[str, Any, Any]] = [("primary", lambda row: True, pump_carbs)]
    has_food = "high_fat_protein" in window_metrics and window_metrics["logged_food_carb_grams"].notna().any()
    if has_food:
        analyses.append(("without high fat or protein breakfasts", lambda row: not bool(getattr(row, "high_fat_protein", False)), pump_carbs))
        if np.isfinite(bevel_scale):
            analyses.append(("pump and logged-food carbs combined", lambda row: True, combined_carbs))

    rows: list[dict[str, Any]] = []
    for analysis, include, carbs_for in analyses:
        for segment, group in window_metrics.groupby(_pool_label(window_metrics), sort=True):
            ordered_rows = list(group.sort_values("timestamp").itertuples(index=False))
            in_force = _resolve_meal_settings(ordered_rows, str(group["segment"].astype(str).mode().iloc[0]), settings)
            meals: list[dict[str, Any]] = []
            for metric_row, (icr, cf, source) in zip(ordered_rows, in_force):
                stated = carbs_for(metric_row)
                if not np.isfinite(stated) or stated <= 0 or not include(metric_row):
                    continue
                meals.append({
                    "row": metric_row, "stated": stated, "icr": round(icr, 2), "cf": round(cf), "source": source, "timestamp": pd.Timestamp(metric_row.timestamp),
                    "pump": pump_key(getattr(metric_row, "pump_serial", "")), "profile": _unknown_as(_as_float(getattr(metric_row, "profile_id", np.nan))),
                    "segment_start": _unknown_as(_as_float(getattr(metric_row, "profile_segment_start_minutes", np.nan))),
                })
            base = {"analysis": analysis, "logged_food_scale": bevel_scale, "logged_food_pairs": bevel_pairs}
            if not meals:
                rows.append({**base, "segment": segment, "programmed_icr": float(settings.icr_for_segment(segment)), "settings_source": "pump profile",
                             "period_start": None, "period_end": None, "is_current_setting": True, "logged_meal_windows": 0,
                             **pool_effective_icr([], programmed_icr=float(settings.icr_for_segment(segment))).to_dict()})
                continue
            # A settings period is a run of breakfasts with the same I:C and CF among those of one pump, profile and
            # pump segment (the pump segment is the time-of-day slice the pump's own settings are set for). Runs
            # are formed inside that subsequence, so a stay on another profile in between does not split one, while
            # an edit to this profile's own setting does.
            by_slice: dict[tuple[str, float, float], list[list[dict[str, Any]]]] = {}
            for meal in meals:
                slice_runs = by_slice.setdefault((meal["pump"], meal["profile"], meal["segment_start"]), [])
                if slice_runs and (slice_runs[-1][-1]["icr"], slice_runs[-1][-1]["cf"]) == (meal["icr"], meal["cf"]):
                    slice_runs[-1].append(meal)
                else:
                    slice_runs.append([meal])
            latest = max(meals, key=lambda meal: meal["timestamp"])
            periods = sorted((run for runs in by_slice.values() for run in runs), key=lambda run: run[0]["timestamp"])
            current_slice_runs = {id(runs[-1]) for key, runs in by_slice.items() if key[:2] == (latest["pump"], latest["profile"])}
            for index, period in enumerate(periods):
                observations: list[CarbSensitivityObservation] = []
                observed: list[tuple[dict[str, Any], int, CarbSensitivityObservation]] = []
                for meal in period:
                    trajectory = trajectories.get(str(meal["row"].meal_id))
                    missing = trajectory is None
                    snapshot = support is not None and meal["source"] == "pump profile"
                    if support is not None and analysis == "primary":
                        support.append({"date": meal["timestamp"].date(), "segment": segment,
                                        "historical_basal_supported": str(meal["row"].meal_id) not in (missing_basal_windows or set()),
                                        "settings_supported": not snapshot,
                                        "in_estimate": False})
                    if missing or snapshot:
                        continue
                    tier = next(
                        (item for item in LATENT_MEAL_ANALYSIS_TIERS if str(item["analysis_tier"]) == str(getattr(meal["row"], "analysis_tier", "clean_reference"))),
                        LATENT_MEAL_ANALYSIS_TIERS[-1],
                    )
                    observation = observe_carb_sensitivity(
                        trajectory,
                        stated_carbs=meal["stated"],
                        icr_grams_per_unit=meal["icr"],
                        correction_factor_mgdl_per_unit=meal["cf"],
                        baseline_glucose=float(meal["row"].premeal_glucose),
                        uncertainty_multiplier=float(tier["uncertainty_multiplier"]),
                    )
                    if observation is not None:
                        if support is not None and analysis == "primary":
                            support[-1]["in_estimate"] = True
                        observations.append(observation)
                        observed.append((meal, int(tier["analysis_tier_rank"]), observation))
                period_fields = {
                    **base,
                    "segment": segment,
                    "settings_source": " and ".join(sorted({meal["source"] for meal in period})),
                    "period_start": period[0]["timestamp"],
                    "period_end": period[-1]["timestamp"],
                    "is_current_setting": id(period) in current_slice_runs,
                    "pump_serial": period[0]["pump"],
                    "profile_id": period[0]["profile"],
                    "pump_segment_start_minutes": period[0]["segment_start"],
                }
                estimate = pool_effective_icr(observations, programmed_icr=period[0]["icr"])
                rows.append({**period_fields, "logged_meal_windows": len(period), **estimate.to_dict()})
                if analysis == "primary":
                    # Checks on the same period that a proposal must agree with.
                    for check, keep in AGREEMENT_CHECKS:
                        subset = [item for meal, rank, item in observed if keep(rank, meal)]
                        check_estimate = pool_effective_icr(subset, programmed_icr=period[0]["icr"])
                        rows.append({**period_fields, "analysis": check, "logged_meal_windows": len(subset), **check_estimate.to_dict()})
    return pd.DataFrame(rows, columns=EFFECTIVE_ICR_EVIDENCE_COLUMNS)


def _period_text(start: Any, end: Any) -> str:
    start_ts, end_ts = pd.to_datetime(start, errors="coerce"), pd.to_datetime(end, errors="coerce")
    if pd.isna(start_ts) or pd.isna(end_ts):
        return "none"
    return f"{start_ts:%b %Y} to {end_ts:%b %Y}" if (start_ts.year, start_ts.month) != (end_ts.year, end_ts.month) else f"{start_ts:%b %Y}"


def _render_effective_icr_section(evidence: pd.DataFrame | None) -> list[str]:
    lines = [
        "## Effective I:C From Logged Meals",
        "",
        "Uses accepted windows with a logged carb count. Each meal is compared with the I:C ratio and correction factor in force "
        "for it (from the bolus calculator, the pump's profile history, recovered from the calculator's corrections, or carried from the nearest "
        "meal on the same pump and profile), then meals are pooled within each "
        "settings period. It assumes that correction factor was right.",
        "",
    ]
    if evidence is None or evidence.empty or int(pd.to_numeric(evidence["logged_meal_windows"], errors="coerce").fillna(0).sum()) == 0:
        lines.extend(
            [
                "No accepted window has a logged carb count, so the I:C setting itself cannot be assessed: without carb truth,",
                "an undercounted meal and a too-weak ratio produce the same glucose trace.",
                "",
                f"To unlock this, log carbs for at least {MIN_LOGGED_MEALS_FOR_ICR} clean first meals (20 is better) on the pump or in a `carbs.csv`",
                "with `timestamp,carb_grams` columns placed in the raw data tree, then run `bayesian-t1dm build-dataset`.",
                "",
            ]
        )
        return lines
    sensitivity = evidence.loc[evidence["analysis"].ne("primary")]
    evidence = evidence.loc[evidence["analysis"].eq("primary")]
    for segment, group in evidence.groupby("segment", sort=True):
        current = group.loc[group["is_current_setting"].astype(bool)]
        row = current.iloc[0] if not current.empty else group.iloc[-1]
        if np.isfinite(row.effective_icr):
            verdict = icr_verdict(str(segment), float(row.effective_icr), float(row.effective_icr_lower_90), float(row.effective_icr_upper_90), float(row.programmed_icr), str(row.status))
            lines.extend([f"**Current setting ({_period_text(row.period_start, row.period_end)}): {verdict.label}.** {verdict.headline} {verdict.explanation}", ""])
        else:
            lines.extend(
                [
                    f"**Current {segment} setting ({float(row.programmed_icr):.1f} g/U since {_period_text(row.period_start, row.period_start)}): "
                    f"{plain(row.status)}** ({int(row.n_meals)} usable meals; at least {MIN_LOGGED_MEALS_FOR_ICR} are needed).",
                    "",
                ]
            )
    lines.extend(
        [
            "| time of day | settings period | pump I:C (g/U) | settings from | breakfasts with logged carbs | used | effective I:C (g/U) | 90% range | result |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in evidence.itertuples(index=False):
        estimated = np.isfinite(row.effective_icr)
        lines.append(
            f"| {row.segment}{' (current)' if bool(row.is_current_setting) else ''} | {_period_text(row.period_start, row.period_end)} | {float(row.programmed_icr):.1f} | "
            f"{row.settings_source} | {int(row.logged_meal_windows)} | {int(row.n_meals)} | "
            + (f"{float(row.effective_icr):.1f} | {float(row.effective_icr_lower_90):.1f} to {float(row.effective_icr_upper_90):.1f}" if estimated else "NA | NA")
            + f" | {plain(row.status)} |"
        )
    if not sensitivity.empty:
        scale = float(sensitivity["logged_food_scale"].iloc[0])
        pairs = int(sensitivity["logged_food_pairs"].iloc[0])
        lines.extend(
            [
                "",
                "### Sensitivity analyses with logged food (for example Bevel)",
                "",
                (f"Logged-food carbs ran {scale:.2f} times the pump entry for the same meal (median over {pairs} meals with both). " if np.isfinite(scale) else "")
                + f"High fat or protein means at least {HIGH_FAT_GRAMS:.0f} g fat or {HIGH_PROTEIN_GRAMS:.0f} g protein logged within {LOGGED_FOOD_MATCH_MINUTES} minutes; "
                "breakfasts without logged food are kept. The combined count is the geometric mean of the pump entry and the logged-food "
                "carbs divided by that ratio, so it can reduce noise but not shift the overall level.",
                "",
                "| analysis | time of day | settings period | pump I:C (g/U) | used | effective I:C (g/U) | 90% range | result |",
                "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for row in sensitivity.loc[sensitivity["is_current_setting"].astype(bool)].itertuples(index=False):
            estimated = np.isfinite(row.effective_icr)
            lines.append(
                f"| {row.analysis} | {row.segment} | {_period_text(row.period_start, row.period_end)} | {float(row.programmed_icr):.1f} | {int(row.n_meals)} | "
                + (f"{float(row.effective_icr):.1f} | {float(row.effective_icr_lower_90):.1f} to {float(row.effective_icr_upper_90):.1f}" if estimated else "NA | NA")
                + f" | {plain(row.status)} |"
            )
    lines.extend(
        [
            "",
            "A lower effective I:C means each gram raised glucose more than the pump setting assumes, so more insulin per gram would have been needed.",
            "This is research evidence for a conversation with a clinician, not an instruction to change a pump setting.",
            "",
        ]
    )
    return lines

def _latent_meal_reference_parameters(
    window_metrics: pd.DataFrame,
    *,
    therapy_settings: LatentMealTherapySettings | None = None,
) -> dict[str, float]:
    settings = _settings_for_estimation(therapy_settings)
    settings_icr_values = [settings.icr_for_segment(segment) for segment in settings.icr_grams_per_unit.keys()]
    settings_cf_values = [settings.correction_factor_for_segment(segment) for segment in settings.correction_factor_mgdl_per_unit.keys()]
    settings_icr_prior = float(np.median(settings_icr_values)) if settings_icr_values else 10.0
    settings_cf_prior = float(np.median(settings_cf_values)) if settings_cf_values else 40.0
    if window_metrics.empty:
        return {
            "icr_prior": settings_icr_prior,
            "correction_factor_prior": settings_cf_prior,
            "target_glucose_mgdl": float(settings.target_glucose_mgdl),
            "insulin_action_minutes": float(settings.insulin_action_minutes),
            "carb_absorption_minutes_prior": float(settings.carb_absorption_minutes_prior),
            "max_plausible_meal_carbs": float(settings.max_plausible_meal_carbs),
            "peak_per_carb": 1.0,
            "auc_per_carb": 2.0,
        }
    bolus_units = pd.to_numeric(window_metrics.get("bolus_units"), errors="coerce").replace(0.0, np.nan)
    stated_carbs = pd.to_numeric(window_metrics.get("stated_carbs"), errors="coerce")
    stated_icr = stated_carbs.div(bolus_units)
    icr_prior = _safe_median(stated_icr.loc[stated_icr.gt(3.0) & stated_icr.lt(30.0)], default=settings_icr_prior)
    icr_by_row = _settings_series(window_metrics, settings, field="icr")
    bolus_reference = bolus_units.fillna(0.0) * icr_by_row
    peak_per_carb = _safe_median(
        pd.to_numeric(window_metrics.get("observed_peak_delta"), errors="coerce").div(bolus_reference.replace(0.0, np.nan)),
        default=4.0,
    )
    auc_per_carb = _safe_median(
        pd.to_numeric(window_metrics.get("observed_auc_above_baseline"), errors="coerce").div(bolus_reference.replace(0.0, np.nan)),
        default=8.0,
    )
    return {
        "icr_prior": _clamp(icr_prior, 4.0, 20.0),
        "correction_factor_prior": _clamp(settings_cf_prior, 10.0, 150.0),
        "target_glucose_mgdl": float(settings.target_glucose_mgdl),
        "insulin_action_minutes": float(settings.insulin_action_minutes),
        "carb_absorption_minutes_prior": float(settings.carb_absorption_minutes_prior),
        "max_plausible_meal_carbs": float(settings.max_plausible_meal_carbs),
        "peak_per_carb": _clamp(peak_per_carb, 0.5, 12.0),
        "auc_per_carb": _clamp(auc_per_carb, 0.5, 30.0),
    }


def _estimate_latent_carbs_for_model(
    window_metrics: pd.DataFrame,
    *,
    model_name: str,
    params: dict[str, float],
    therapy_settings: LatentMealTherapySettings | None = None,
) -> pd.Series:
    settings = _settings_for_estimation(therapy_settings)
    bolus_units = pd.to_numeric(window_metrics.get("bolus_units"), errors="coerce").fillna(0.0)
    stated_carbs = pd.to_numeric(window_metrics.get("stated_carbs"), errors="coerce")
    peak_delta = pd.to_numeric(window_metrics.get("observed_peak_delta"), errors="coerce")
    auc_above_baseline = pd.to_numeric(window_metrics.get("observed_auc_above_baseline"), errors="coerce")
    bolus_reference = bolus_units * _settings_series(window_metrics, settings, field="icr")
    stated_prior_blend = np.where(stated_carbs.notna() & stated_carbs.gt(0), 0.7 * stated_carbs + 0.3 * bolus_reference, bolus_reference)
    response_signal = auc_above_baseline.div(float(params["auc_per_carb"])).replace([np.inf, -np.inf], np.nan)
    peak_signal = peak_delta.div(float(params["peak_per_carb"])).replace([np.inf, -np.inf], np.nan)
    response_adjusted = 0.55 * response_signal.fillna(bolus_reference) + 0.25 * peak_signal.fillna(bolus_reference) + 0.20 * bolus_reference
    if stated_carbs.notna().any():
        response_adjusted = np.where(stated_carbs.notna() & stated_carbs.gt(0), 0.2 * stated_carbs + 0.8 * response_adjusted, response_adjusted)
    if model_name == "bolus_reference":
        latent = bolus_reference
    elif model_name == "stated_prior_blend":
        latent = stated_prior_blend
    elif model_name == "response_adjusted":
        latent = response_adjusted
    else:
        raise ValueError(f"Unknown latent meal model: {model_name}")
    return pd.Series(latent, index=window_metrics.index, dtype=float).clip(lower=1.0)


def _latent_meal_model_comparison(
    window_metrics: pd.DataFrame,
    *,
    therapy_settings: LatentMealTherapySettings | None = None,
) -> tuple[pd.DataFrame, str, dict[str, float]]:
    columns = ["model", "meal_count", "stated_carb_mae", "peak_delta_mae", "peak_delta_correlation", "selected"]
    if window_metrics.empty:
        return pd.DataFrame(columns=columns), "bolus_reference", _latent_meal_reference_parameters(window_metrics, therapy_settings=therapy_settings)
    models = ["bolus_reference", "stated_prior_blend", "response_adjusted"]
    rows: list[dict[str, Any]] = []
    for model_name in models:
        peak_errors: list[float] = []
        stated_errors: list[float] = []
        correlations: list[float] = []
        for holdout_index in window_metrics.index:
            train = window_metrics.loc[window_metrics.index != holdout_index].copy()
            if train.empty:
                train = window_metrics.copy()
            params = _latent_meal_reference_parameters(train, therapy_settings=therapy_settings)
            latent_train = _estimate_latent_carbs_for_model(train, model_name=model_name, params=params, therapy_settings=therapy_settings)
            peak_scale = _safe_median(
                pd.to_numeric(train.get("observed_peak_delta"), errors="coerce").div(latent_train.replace(0.0, np.nan)),
                default=float(params["peak_per_carb"]),
            )
            holdout = window_metrics.loc[[holdout_index]].copy()
            latent_holdout = _estimate_latent_carbs_for_model(holdout, model_name=model_name, params=params, therapy_settings=therapy_settings)
            predicted_peak = float(latent_holdout.iloc[0] * peak_scale)
            observed_peak = float(pd.to_numeric(holdout["observed_peak_delta"], errors="coerce").iloc[0])
            if np.isfinite(observed_peak):
                peak_errors.append(abs(predicted_peak - observed_peak))
            stated_carbs = pd.to_numeric(holdout["stated_carbs"], errors="coerce").iloc[0]
            if np.isfinite(stated_carbs):
                stated_errors.append(abs(float(latent_holdout.iloc[0]) - float(stated_carbs)))
            correlations.append(
                _safe_correlation(
                    pd.Series([float(latent_holdout.iloc[0])]),
                    pd.Series([observed_peak]),
                )
            )
        full_params = _latent_meal_reference_parameters(window_metrics, therapy_settings=therapy_settings)
        latent_full = _estimate_latent_carbs_for_model(window_metrics, model_name=model_name, params=full_params, therapy_settings=therapy_settings)
        rows.append(
            {
                "model": model_name,
                "meal_count": int(len(window_metrics)),
                "stated_carb_mae": float(np.mean(stated_errors)) if stated_errors else float("nan"),
                "peak_delta_mae": float(np.mean(peak_errors)) if peak_errors else float("nan"),
                "peak_delta_correlation": _safe_correlation(latent_full, pd.to_numeric(window_metrics.get("observed_peak_delta"), errors="coerce")),
                "selected": False,
            }
        )
    comparison = pd.DataFrame(rows, columns=columns)
    if comparison["stated_carb_mae"].notna().any():
        selected_index = (
            comparison.assign(
                _sort_peak=comparison["peak_delta_mae"].fillna(np.inf),
                _sort_stated=comparison["stated_carb_mae"].fillna(np.inf),
            )
            .sort_values(["_sort_peak", "_sort_stated", "model"])
            .index[0]
        )
    else:
        selected_index = (
            comparison.assign(_sort_peak=comparison["peak_delta_mae"].fillna(np.inf))
            .sort_values(["_sort_peak", "model"])
            .index[0]
        )
    selected_model = str(comparison.loc[selected_index, "model"])
    comparison.loc[selected_index, "selected"] = True
    selected_params = _latent_meal_reference_parameters(window_metrics, therapy_settings=therapy_settings)
    return comparison, selected_model, selected_params


def _latent_carb_posterior_for_window(
    row: pd.Series,
    *,
    settings: LatentMealTherapySettings,
    params: dict[str, float],
    trajectory: MealWindowTrajectory | None = None,
) -> tuple[dict[str, float | str | bool], pd.DataFrame]:
    segment = str(row.get("segment") or "default")
    if "resolved_icr" in row.index:  # the meal's own settings, resolved by ``_attach_resolved_settings``
        icr = float(row["resolved_icr"])
        correction_factor = float(row["resolved_cf"])
    else:  # a caller with no per-meal history (synthetic validation) uses the programmed segment
        icr = settings.icr_for_segment(segment)
        correction_factor = settings.correction_factor_for_segment(segment)
    bolus_units = _positive_float(max(float(pd.to_numeric(pd.Series([row.get("bolus_units")]), errors="coerce").fillna(0.0).iloc[0]), 1e-6), "bolus_units")
    premeal_iob = float(pd.to_numeric(pd.Series([row.get("premeal_iob")]), errors="coerce").fillna(0.0).iloc[0])
    observed_peak_delta = float(pd.to_numeric(pd.Series([row.get("observed_peak_delta")]), errors="coerce").fillna(np.nan).iloc[0])
    observed_auc = float(pd.to_numeric(pd.Series([row.get("observed_auc_above_baseline")]), errors="coerce").fillna(np.nan).iloc[0])
    observed_peak_glucose = float(pd.to_numeric(pd.Series([row.get("observed_peak_glucose")]), errors="coerce").fillna(np.nan).iloc[0])
    observed_mean_glucose = float(pd.to_numeric(pd.Series([row.get("observed_mean_glucose_180m")]), errors="coerce").fillna(np.nan).iloc[0])
    cgm_coverage = float(pd.to_numeric(pd.Series([row.get("cgm_coverage_fraction")]), errors="coerce").fillna(0.0).iloc[0])
    explicit_available = bool(int(pd.to_numeric(pd.Series([row.get("explicit_carb_source_available")]), errors="coerce").fillna(0.0).iloc[0]) > 0)
    stated_carbs = float(pd.to_numeric(pd.Series([row.get("stated_carbs")]), errors="coerce").fillna(np.nan).iloc[0])
    analysis_tier = str(row.get("analysis_tier") or "clean_reference")
    tier = next((item for item in LATENT_MEAL_ANALYSIS_TIERS if str(item["analysis_tier"]) == analysis_tier), LATENT_MEAL_ANALYSIS_TIERS[-1])
    analysis_tier_rank = int(tier["analysis_tier_rank"])
    tier_confidence_multiplier = float(tier["confidence_multiplier"])
    tier_uncertainty_multiplier = float(tier["uncertainty_multiplier"])
    relaxed_exclusion_reasons = str(row.get("relaxed_exclusion_reasons") or "")

    bolus_implied = float(bolus_units * icr)
    carb_prior_mean = float(0.7 * stated_carbs + 0.3 * bolus_implied) if np.isfinite(stated_carbs) and stated_carbs > 0 else bolus_implied
    carb_prior_sd = max(10.0 if explicit_available else 15.0, carb_prior_mean * (0.22 if explicit_available else 0.38))
    carb_prior_sd *= tier_uncertainty_multiplier
    max_carbs = max(float(settings.max_plausible_meal_carbs), carb_prior_mean * 1.6, carb_prior_mean + 40.0, 30.0)
    max_carbs = min(max_carbs, float(settings.max_plausible_meal_carbs))
    carb_grid = np.linspace(1.0, max_carbs, 161)

    absorption_prior = float(settings.carb_absorption_minutes_prior)
    absorption_low = max(45.0, absorption_prior * 0.45)
    absorption_high = min(420.0, max(absorption_prior * 1.9, absorption_prior + 90.0))
    absorption_grid = np.linspace(absorption_low, absorption_high, 49)
    carbs, absorption = np.meshgrid(carb_grid, absorption_grid, indexing="ij")

    absorption_peak_modifier = np.clip((120.0 / absorption) ** 0.35, 0.55, 1.45)
    carb_peak_per_gram = np.clip((correction_factor / max(icr, 1e-6)) * absorption_peak_modifier * 0.65, 0.25, 12.0)
    insulin_peak_fraction = np.clip(150.0 / max(float(settings.insulin_action_minutes), 1.0), 0.25, 0.65)
    raw_peak_delta = carbs * carb_peak_per_gram - bolus_units * correction_factor * insulin_peak_fraction - max(premeal_iob, 0.0) * correction_factor * 0.18

    carb_auc_per_gram = np.clip((correction_factor / max(icr, 1e-6)) * (absorption / 60.0) * 0.22, 0.15, 25.0)
    insulin_auc_offset = bolus_units * correction_factor * (float(settings.insulin_action_minutes) / 60.0) * 0.08
    raw_auc = np.maximum(0.0, carbs * carb_auc_per_gram - insulin_auc_offset - max(premeal_iob, 0.0) * correction_factor * 0.18)

    peak_target = observed_peak_delta if np.isfinite(observed_peak_delta) else float(pd.to_numeric(pd.Series([row.get("observed_mean_glucose_180m")]), errors="coerce").fillna(0.0).iloc[0])
    drift_bound = 30.0 * tier_uncertainty_multiplier
    drift = np.clip(peak_target - raw_peak_delta, -drift_bound, drift_bound)
    peak_residual = peak_target - (raw_peak_delta + drift)
    auc_target = observed_auc if np.isfinite(observed_auc) else float("nan")
    auc_residual = np.zeros_like(raw_auc)
    if np.isfinite(auc_target):
        auc_residual = auc_target - (raw_auc + np.maximum(drift, 0.0) * 1.2)

    glucose_sd = max(12.0, abs(peak_target) * 0.25, 18.0 if cgm_coverage < 0.9 else 0.0) * tier_uncertainty_multiplier
    auc_sd = max(15.0, abs(auc_target) * 0.35 if np.isfinite(auc_target) else 0.0) * tier_uncertainty_multiplier
    drift_sd = 14.0 * tier_uncertainty_multiplier
    absorption_sd = max(30.0, absorption_prior * 0.35) * tier_uncertainty_multiplier

    log_weight = (
        -0.5 * ((carbs - carb_prior_mean) / carb_prior_sd) ** 2
        -0.5 * ((absorption - absorption_prior) / absorption_sd) ** 2
        -0.5 * (peak_residual / glucose_sd) ** 2
        -0.5 * (drift / drift_sd) ** 2
    )
    if np.isfinite(auc_target):
        log_weight = log_weight - 0.5 * (auc_residual / auc_sd) ** 2
    log_weight = log_weight - np.nanmax(log_weight)
    weights = np.exp(log_weight)
    weights = weights / np.nansum(weights)

    flat_carbs = carbs.ravel()
    flat_absorption = absorption.ravel()
    flat_weights = weights.ravel()
    carb_mean = float(np.nansum(flat_carbs * flat_weights))
    absorption_mean = float(np.nansum(flat_absorption * flat_weights))
    carb_lower_50 = _weighted_quantile(flat_carbs, flat_weights, 0.25)
    carb_upper_50 = _weighted_quantile(flat_carbs, flat_weights, 0.75)
    carb_lower_90 = _weighted_quantile(flat_carbs, flat_weights, 0.05)
    carb_median = _weighted_quantile(flat_carbs, flat_weights, 0.50)
    carb_upper_90 = _weighted_quantile(flat_carbs, flat_weights, 0.95)
    absorption_lower_90 = _weighted_quantile(flat_absorption, flat_weights, 0.05)
    absorption_upper_90 = _weighted_quantile(flat_absorption, flat_weights, 0.95)

    # Preferred path: fit the whole post-meal trace. The summary-statistic posterior above
    # is kept only as a fallback for windows whose trace or insulin bookkeeping is unusable.
    estimator = "summary_heuristic"
    prior_contraction = float("nan")
    trajectory_fit_rmse = float("nan")
    trajectory_poor_fit = False
    if trajectory is not None and trajectory.is_usable():
        premeal_glucose = float(pd.to_numeric(pd.Series([row.get("premeal_glucose")]), errors="coerce").fillna(np.nan).iloc[0])
        trajectory_posterior = estimate_carbs_from_trajectory(
            trajectory,
            icr_grams_per_unit=icr,
            correction_factor_mgdl_per_unit=correction_factor,
            bolus_implied_carbs=bolus_implied,
            baseline_glucose=premeal_glucose,
            max_carbs=max_carbs,
            # The settings carry an absorption *duration*; a gamma meal has absorbed ~95% of
            # its carbohydrate at roughly 2.2x its mean absorption time.
            absorption_mean_minutes_prior=absorption_prior / 2.2,
            carb_prior_mean=carb_prior_mean,
            carb_prior_sd=None if not explicit_available else carb_prior_sd,
            uncertainty_multiplier=tier_uncertainty_multiplier,
        )
        estimator = "trajectory"
        carb_mean = trajectory_posterior.carb_mean
        carb_median = trajectory_posterior.carb_median
        carb_lower_50 = trajectory_posterior.carb_lower_50
        carb_upper_50 = trajectory_posterior.carb_upper_50
        carb_lower_90 = trajectory_posterior.carb_lower_90
        carb_upper_90 = trajectory_posterior.carb_upper_90
        absorption_mean = trajectory_posterior.absorption_minutes_mean
        absorption_lower_90 = trajectory_posterior.absorption_minutes_lower_90
        absorption_upper_90 = trajectory_posterior.absorption_minutes_upper_90
        flat_carbs = trajectory_posterior.grid_carbs
        flat_absorption = trajectory_posterior.grid_absorption_minutes
        flat_weights = trajectory_posterior.grid_weights
        prior_contraction = trajectory_posterior.prior_contraction
        trajectory_fit_rmse = trajectory_posterior.fit_rmse_mgdl
        trajectory_poor_fit = trajectory_posterior.poor_fit

    prior_gap = abs(carb_median - bolus_implied)
    interval_width_90 = carb_upper_90 - carb_lower_90
    interval_penalty = min(1.0, interval_width_90 / max(carb_median, 1.0))
    recent_exercise = bool(int(pd.to_numeric(pd.Series([row.get("recent_exercise_context")]), errors="coerce").fillna(0.0).iloc[0]) > 0)
    recent_workout_context = bool(int(pd.to_numeric(pd.Series([row.get("recent_workout_context")]), errors="coerce").fillna(0.0).iloc[0]) > 0)
    recent_workout_12h_context = bool(int(pd.to_numeric(pd.Series([row.get("recent_workout_12h_context")]), errors="coerce").fillna(0.0).iloc[0]) > 0)
    light_activity_context = bool(int(pd.to_numeric(pd.Series([row.get("light_activity_context")]), errors="coerce").fillna(0.0).iloc[0]) > 0)
    moderate_activity_context = bool(int(pd.to_numeric(pd.Series([row.get("moderate_activity_context")]), errors="coerce").fillna(0.0).iloc[0]) > 0)
    high_activity_context = bool(int(pd.to_numeric(pd.Series([row.get("high_activity_context")]), errors="coerce").fillna(0.0).iloc[0]) > 0)
    health_activity_roll_sum_60m = float(pd.to_numeric(pd.Series([row.get("health_activity_roll_sum_60m")]), errors="coerce").fillna(0.0).iloc[0])
    soft_activity_context = (recent_workout_12h_context and not recent_workout_context) or moderate_activity_context
    source_quality = str(row.get("source_quality_grade") or "unknown")
    post_meal_control_success = bool(
        cgm_coverage >= 0.9
        and source_quality == "good"
        and not recent_exercise
        and np.isfinite(observed_peak_glucose)
        and observed_peak_glucose <= max(180.0, settings.target_glucose_mgdl + 65.0)
        and (not np.isfinite(observed_mean_glucose) or observed_mean_glucose <= max(165.0, settings.target_glucose_mgdl + 45.0))
    )

    confidence = math.exp(-prior_gap / max(15.0, bolus_implied * 0.35))
    confidence *= max(0.25, min(1.0, cgm_coverage))
    confidence *= max(0.30, 1.0 - 0.45 * interval_penalty)
    if post_meal_control_success:
        confidence += 0.08
    if recent_exercise:
        confidence *= 0.75
    elif soft_activity_context:
        confidence *= 0.90
    if source_quality != "good":
        confidence *= 0.70
    confidence *= tier_confidence_multiplier
    confidence_cap = 0.95 if explicit_available else 0.75
    confidence = float(np.clip(confidence, 0.0, confidence_cap))

    flags: list[str] = []
    if not explicit_available:
        flags.append("proxy_only")
    if analysis_tier != "clean_reference":
        flags.append(analysis_tier)
    if post_meal_control_success:
        flags.append("post_meal_control_success")
    if interval_width_90 / max(carb_median, 1.0) > 0.85:
        flags.append("wide_credible_interval")
    if recent_exercise:
        flags.append("exercise_context")
    elif soft_activity_context:
        flags.append("soft_activity_context")
    if light_activity_context:
        flags.append("light_activity_context")
    if source_quality != "good":
        flags.append("degraded_source_quality")
    relaxed_reason_set = _exclusion_reason_set(relaxed_exclusion_reasons)
    if "closed_loop_confounding_premeal" in relaxed_reason_set:
        flags.append("controller_context")
    if {"premeal_bg_out_of_range", "steep_premeal_slope"} & relaxed_reason_set:
        flags.append("baseline_drift_context")
    if {"recent_bolus_4h", "postmeal_additional_bolus_120m", "high_premeal_iob"} & relaxed_reason_set:
        flags.append("insulin_context")
    if peak_target < 20.0:
        flags.append("weak_postmeal_signal")
    if estimator != "trajectory":
        flags.append("summary_heuristic_estimator")
    if trajectory_poor_fit:
        flags.append("trajectory_poor_fit")

    sample_quantiles = np.linspace(0.005, 0.995, 100)
    cumulative = np.cumsum(flat_weights)
    sample_rows: list[dict[str, Any]] = []
    for sample_index, quantile in enumerate(sample_quantiles):
        sample_pos = int(np.searchsorted(cumulative, quantile, side="left"))
        grid_index = min(sample_pos, len(flat_weights) - 1)
        sample_rows.append(
            {
                "meal_id": str(row.get("meal_id")),
                "sample_index": sample_index,
                "analysis_tier": analysis_tier,
                "latent_carb_grams": float(flat_carbs[grid_index]),
                "absorption_minutes": float(flat_absorption[grid_index]),
                "posterior_weight": float(flat_weights[grid_index]),
            }
        )

    summary: dict[str, float | str | bool] = {
        "icr_prior": float(icr),
        "correction_factor_prior": float(correction_factor),
        "bolus_implied_carbs": float(bolus_implied),
        "latent_carb_mean": carb_mean,
        "latent_carb_median": carb_median,
        "latent_carb_lower_50": carb_lower_50,
        "latent_carb_upper_50": carb_upper_50,
        "latent_carb_lower_90": carb_lower_90,
        "latent_carb_upper_90": carb_upper_90,
        "absorption_minutes_mean": absorption_mean,
        "absorption_minutes_lower_90": absorption_lower_90,
        "absorption_minutes_upper_90": absorption_upper_90,
        "post_meal_control_success": post_meal_control_success,
        "posterior_interval_width_90": interval_width_90,
        "confidence": confidence,
        "recent_workout_context": recent_workout_context,
        "recent_workout_12h_context": recent_workout_12h_context,
        "light_activity_context": light_activity_context,
        "moderate_activity_context": moderate_activity_context,
        "high_activity_context": high_activity_context,
        "health_activity_roll_sum_60m": health_activity_roll_sum_60m,
        "analysis_tier": analysis_tier,
        "analysis_tier_rank": analysis_tier_rank,
        "relaxed_exclusion_reasons": relaxed_exclusion_reasons,
        "flags": "|".join(flags),
        "estimator": estimator,
        "prior_contraction": prior_contraction,
        "trajectory_fit_rmse_mgdl": trajectory_fit_rmse,
    }
    return summary, pd.DataFrame(sample_rows)


def _build_latent_meal_posterior_meals(
    window_metrics: pd.DataFrame,
    *,
    model_name: str,
    params: dict[str, float],
    therapy_settings: LatentMealTherapySettings | None = None,
    trajectories: Mapping[str, MealWindowTrajectory] | None = None,
) -> pd.DataFrame:
    columns = [
        "meal_id",
        "timestamp",
        "meal_start",
        "segment",
        "therapy_segment",
        "evidence_source",
        "stated_carbs",
        "bolus_units",
        "icr_prior",
        "bolus_implied_carbs",
        "bolus_reference_carbs",
        "stated_prior_blend_carbs",
        "latent_response_adjusted_carbs",
        "latent_carb_mean",
        "latent_carb_median",
        "latent_carb_lower_50",
        "latent_carb_upper_50",
        "latent_carb_lower_90",
        "latent_carb_upper_90",
        "absorption_minutes_mean",
        "absorption_minutes_lower_90",
        "absorption_minutes_upper_90",
        "post_meal_control_success",
        "posterior_interval_width_90",
        "confidence",
        "flags",
        "recent_workout_context",
        "recent_workout_12h_context",
        "light_activity_context",
        "moderate_activity_context",
        "high_activity_context",
        "health_activity_roll_sum_60m",
        "analysis_tier",
        "analysis_tier_rank",
        "relaxed_exclusion_reasons",
        "latent_carbs_posterior_mean",
        "latent_carbs_lower",
        "latent_carbs_upper",
        "carb_accuracy_score",
        "ic_posterior_mean",
        "ic_lower",
        "ic_upper",
        "confounding_grade",
        "source_quality_grade",
        "identifiability_status",
        "suppression_reason",
        "observed_peak_delta",
        "observed_auc_above_baseline",
        "selected_model",
        "estimator",
        "prior_contraction",
        "trajectory_fit_rmse_mgdl",
    ]
    if window_metrics.empty:
        return pd.DataFrame(columns=columns)
    settings = _settings_for_estimation(therapy_settings)
    window_metrics = window_metrics.copy()
    bolus_units = pd.to_numeric(window_metrics.get("bolus_units"), errors="coerce").fillna(0.0)
    stated_carbs = pd.to_numeric(window_metrics.get("stated_carbs"), errors="coerce")
    peak_delta = pd.to_numeric(window_metrics.get("observed_peak_delta"), errors="coerce")
    icr_by_row = _settings_series(window_metrics, settings, field="icr")
    bolus_reference = bolus_units * icr_by_row
    explicit_available = window_metrics["explicit_carb_source_available"].astype(int).eq(1)
    stated_prior_blend = _estimate_latent_carbs_for_model(window_metrics, model_name="stated_prior_blend", params=params, therapy_settings=settings)
    response_adjusted = _estimate_latent_carbs_for_model(window_metrics, model_name="response_adjusted", params=params, therapy_settings=settings)

    posterior_summaries: list[dict[str, Any]] = []
    for _, metric_row in window_metrics.iterrows():
        posterior_summary, _ = _latent_carb_posterior_for_window(
            metric_row,
            settings=settings,
            params=params,
            trajectory=(trajectories or {}).get(str(metric_row.get("meal_id"))),
        )
        posterior_summaries.append(posterior_summary)
    posterior_summary_frame = pd.DataFrame(posterior_summaries, index=window_metrics.index)
    latent_mean = pd.to_numeric(posterior_summary_frame["latent_carb_mean"], errors="coerce")
    latent_lower = pd.to_numeric(posterior_summary_frame["latent_carb_lower_90"], errors="coerce")
    latent_upper = pd.to_numeric(posterior_summary_frame["latent_carb_upper_90"], errors="coerce")

    stated_gap = (latent_mean - stated_carbs).abs()
    prior_gap = (latent_mean - bolus_reference).abs()
    carb_accuracy_score = np.where(
        stated_carbs.notna() & stated_carbs.gt(0),
        np.exp(-stated_gap / np.maximum(15.0, stated_carbs.fillna(0.0) * 0.35)),
        0.75 * np.exp(-prior_gap / np.maximum(12.0, bolus_reference * 0.30)),
    )
    carb_accuracy_score = pd.Series(carb_accuracy_score, index=window_metrics.index, dtype=float).clip(lower=0.0, upper=1.0)

    relative_interval = (latent_upper - latent_lower) / latent_mean.replace(0.0, np.nan)
    identifiability_status = []
    suppression_reason = []
    ic_mean = []
    ic_lower = []
    ic_upper = []
    confounding_grade = []
    for idx in window_metrics.index:
        reasons: list[str] = []
        if str(window_metrics.loc[idx, "source_quality_grade"]) != "good":
            reasons.append("degraded_source_quality")
        if bool(pd.to_numeric(pd.Series([window_metrics.loc[idx, "recent_exercise_context"]]), errors="coerce").fillna(0.0).iloc[0] > 0):
            reasons.append("exercise_context")
        if not np.isfinite(float(peak_delta.loc[idx])) or float(peak_delta.loc[idx]) < 20.0:
            reasons.append("weak_postmeal_signal")
        if float(relative_interval.loc[idx]) > 0.70:
            reasons.append("wide_posterior_interval")
        if float(bolus_units.loc[idx]) < 2.0:
            reasons.append("small_bolus_anchor")
        if reasons:
            if "degraded_source_quality" in reasons or "weak_postmeal_signal" in reasons:
                identifiability_status.append("not_identified")
            else:
                identifiability_status.append("weakly_identified")
            suppression_reason.append("|".join(reasons))
            ic_mean.append(float("nan"))
            ic_lower.append(float("nan"))
            ic_upper.append(float("nan"))
        else:
            identifiability_status.append("proxy_supported" if not bool(explicit_available.loc[idx]) else "directly_observed")
            suppression_reason.append("")
            ic_mean.append(float(latent_mean.loc[idx] / max(float(bolus_units.loc[idx]), 1e-6)))
            ic_lower.append(float(latent_lower.loc[idx] / max(float(bolus_units.loc[idx]), 1e-6)))
            ic_upper.append(float(latent_upper.loc[idx] / max(float(bolus_units.loc[idx]), 1e-6)))
        if "exercise_context" in reasons:
            confounding_grade.append("moderate")
        elif reasons:
            confounding_grade.append("low_to_moderate")
        else:
            confounding_grade.append("low")

    posterior = pd.DataFrame(
        {
            "meal_id": window_metrics["meal_id"].astype(str),
            "timestamp": window_metrics["timestamp"],
            "meal_start": window_metrics["timestamp"],
            "segment": window_metrics["segment"].astype(str),
            "therapy_segment": window_metrics["segment"].astype(str),
            "evidence_source": window_metrics["evidence_source"].astype(str),
            "stated_carbs": stated_carbs,
            "bolus_units": bolus_units,
            "icr_prior": posterior_summary_frame["icr_prior"],
            "bolus_implied_carbs": posterior_summary_frame["bolus_implied_carbs"],
            "bolus_reference_carbs": bolus_reference,
            "stated_prior_blend_carbs": stated_prior_blend,
            "latent_response_adjusted_carbs": response_adjusted,
            "latent_carb_mean": posterior_summary_frame["latent_carb_mean"],
            "latent_carb_median": posterior_summary_frame["latent_carb_median"],
            "latent_carb_lower_50": posterior_summary_frame["latent_carb_lower_50"],
            "latent_carb_upper_50": posterior_summary_frame["latent_carb_upper_50"],
            "latent_carb_lower_90": posterior_summary_frame["latent_carb_lower_90"],
            "latent_carb_upper_90": posterior_summary_frame["latent_carb_upper_90"],
            "absorption_minutes_mean": posterior_summary_frame["absorption_minutes_mean"],
            "absorption_minutes_lower_90": posterior_summary_frame["absorption_minutes_lower_90"],
            "absorption_minutes_upper_90": posterior_summary_frame["absorption_minutes_upper_90"],
            "post_meal_control_success": posterior_summary_frame["post_meal_control_success"],
            "posterior_interval_width_90": posterior_summary_frame["posterior_interval_width_90"],
            "confidence": posterior_summary_frame["confidence"],
            "flags": posterior_summary_frame["flags"],
            "recent_workout_context": posterior_summary_frame["recent_workout_context"],
            "recent_workout_12h_context": posterior_summary_frame["recent_workout_12h_context"],
            "light_activity_context": posterior_summary_frame["light_activity_context"],
            "moderate_activity_context": posterior_summary_frame["moderate_activity_context"],
            "high_activity_context": posterior_summary_frame["high_activity_context"],
            "health_activity_roll_sum_60m": posterior_summary_frame["health_activity_roll_sum_60m"],
            "analysis_tier": posterior_summary_frame["analysis_tier"],
            "analysis_tier_rank": posterior_summary_frame["analysis_tier_rank"],
            "relaxed_exclusion_reasons": posterior_summary_frame["relaxed_exclusion_reasons"],
            "latent_carbs_posterior_mean": latent_mean,
            "latent_carbs_lower": latent_lower,
            "latent_carbs_upper": latent_upper,
            "carb_accuracy_score": carb_accuracy_score,
            "ic_posterior_mean": pd.Series(ic_mean, index=window_metrics.index, dtype=float),
            "ic_lower": pd.Series(ic_lower, index=window_metrics.index, dtype=float),
            "ic_upper": pd.Series(ic_upper, index=window_metrics.index, dtype=float),
            "confounding_grade": pd.Series(confounding_grade, index=window_metrics.index, dtype=str),
            "source_quality_grade": window_metrics["source_quality_grade"].astype(str),
            "identifiability_status": pd.Series(identifiability_status, index=window_metrics.index, dtype=str),
            "suppression_reason": pd.Series(suppression_reason, index=window_metrics.index, dtype=str),
            "observed_peak_delta": peak_delta,
            "observed_auc_above_baseline": pd.to_numeric(window_metrics.get("observed_auc_above_baseline"), errors="coerce"),
            "selected_model": model_name,
            "estimator": posterior_summary_frame["estimator"],
            "prior_contraction": posterior_summary_frame["prior_contraction"],
            "trajectory_fit_rmse_mgdl": posterior_summary_frame["trajectory_fit_rmse_mgdl"],
        }
    ).sort_values("timestamp").reset_index(drop=True)
    return posterior


LATENT_CARB_ESTIMATE_COLUMNS = [
    "meal_id",
    "meal_start",
    "therapy_segment",
    "evidence_source",
    "bolus_units",
    "icr_prior",
    "bolus_implied_carbs",
    "latent_carb_mean",
    "latent_carb_median",
    "latent_carb_lower_50",
    "latent_carb_upper_50",
    "latent_carb_lower_90",
    "latent_carb_upper_90",
    "absorption_minutes_mean",
    "absorption_minutes_lower_90",
    "absorption_minutes_upper_90",
    "post_meal_control_success",
    "posterior_interval_width_90",
    "confidence",
    "recent_workout_context",
    "recent_workout_12h_context",
    "light_activity_context",
    "moderate_activity_context",
    "high_activity_context",
    "health_activity_roll_sum_60m",
    "analysis_tier",
    "analysis_tier_rank",
    "relaxed_exclusion_reasons",
    "estimator",
    "prior_contraction",
    "trajectory_fit_rmse_mgdl",
    "flags",
    "exclusion_reasons",
]


def _build_latent_carb_estimates(posterior_meals: pd.DataFrame) -> pd.DataFrame:
    if posterior_meals.empty:
        return pd.DataFrame(columns=LATENT_CARB_ESTIMATE_COLUMNS)
    estimates = posterior_meals.copy()
    if "exclusion_reasons" not in estimates.columns:
        estimates["exclusion_reasons"] = estimates.get("suppression_reason", "")
    for column in LATENT_CARB_ESTIMATE_COLUMNS:
        if column not in estimates.columns:
            estimates[column] = np.nan
    return estimates[LATENT_CARB_ESTIMATE_COLUMNS].sort_values("meal_start").reset_index(drop=True)


def _build_latent_carb_posterior_samples(
    window_metrics: pd.DataFrame,
    *,
    settings: LatentMealTherapySettings,
    params: dict[str, float],
    trajectories: Mapping[str, MealWindowTrajectory] | None = None,
) -> pd.DataFrame:
    columns = ["meal_id", "sample_index", "analysis_tier", "latent_carb_grams", "absorption_minutes", "posterior_weight"]
    if window_metrics.empty:
        return pd.DataFrame(columns=columns)
    frames: list[pd.DataFrame] = []
    for _, metric_row in window_metrics.iterrows():
        _, samples = _latent_carb_posterior_for_window(
            metric_row,
            settings=settings,
            params=params,
            trajectory=(trajectories or {}).get(str(metric_row.get("meal_id"))),
        )
        frames.append(samples)
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


def _render_full_latent_meal_gate(
    dataset: AnalysisReadyHealthDataset,
    meal_windows: pd.DataFrame,
    posterior_meals: pd.DataFrame,
    *,
    research_scope: str,
) -> tuple[pd.DataFrame, str]:
    accepted_windows = int(pd.to_numeric(meal_windows.get("included"), errors="coerce").fillna(0.0).sum()) if not meal_windows.empty else 0
    reportable_ic = int(posterior_meals["ic_posterior_mean"].notna().sum()) if not posterior_meals.empty else 0
    dataset_source_quality = str(getattr(dataset, "source_quality_status", "unknown") or "unknown")
    source_quality_status = dataset_source_quality if dataset_source_quality not in {"unknown", "good"} else "good" if accepted_windows > 0 else "limited"
    rows = [
        {
            "parameter": "latent_carbs",
            "estimand": "first-meal latent carb exposure",
            "identifiability": "proxy_supported" if accepted_windows > 0 else "not_identified",
            "gate_status": "research_enabled" if accepted_windows > 0 else "diagnostics_only",
            "source_quality_status": source_quality_status,
            "closed_loop_confounding_risk": "controlled_or_flagged_by_tiered_gate",
        },
        {
            "parameter": "carb_entry_confidence",
            "estimand": "carb-entry consistency score",
            "identifiability": "proxy_supported" if accepted_windows > 0 else "not_identified",
            "gate_status": "research_enabled" if accepted_windows > 0 else "diagnostics_only",
            "source_quality_status": source_quality_status,
            "closed_loop_confounding_risk": "controlled_or_flagged_by_tiered_gate",
        },
        {
            "parameter": "morning_icr",
            "estimand": "effective morning I/C on tiered first-meal windows",
            "identifiability": "weakly_identified" if reportable_ic > 0 else "not_identified",
            "gate_status": "research_enabled" if reportable_ic > 0 else "diagnostics_only",
            "source_quality_status": source_quality_status,
            "closed_loop_confounding_risk": "controlled_or_flagged_by_tiered_gate",
        },
    ]
    gate = pd.DataFrame(rows)
    lines = [
        "# Latent Meal Research Gate",
        "",
        f"- research_scope: {research_scope}",
        f"- explicit_carb_source_available: {bool(getattr(dataset, 'explicit_carb_source_available', False))}",
        f"- accepted_windows: {accepted_windows}",
        f"- reportable_morning_ic_windows: {reportable_ic}",
        f"- source_quality_status: {source_quality_status}",
        "",
        "## Parameter Gate",
        "",
        "| parameter | estimand | identifiability | gate_status | source_quality_status | closed_loop_confounding_risk |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in gate.itertuples(index=False):
        lines.append(
            f"| {row.parameter} | {row.estimand} | {row.identifiability} | {row.gate_status} | {row.source_quality_status} | {row.closed_loop_confounding_risk} |"
        )
    return gate, "\n".join(lines) + "\n"


def _render_latent_meal_fit_summary(
    window_metrics: pd.DataFrame,
    posterior_meals: pd.DataFrame,
    model_comparison: pd.DataFrame,
    *,
    selected_model: str,
    params: dict[str, float],
) -> str:
    selected_row = model_comparison.loc[model_comparison["selected"].astype(bool)].head(1)
    peak_mae = float(selected_row["peak_delta_mae"].iloc[0]) if not selected_row.empty and pd.notna(selected_row["peak_delta_mae"].iloc[0]) else float("nan")
    reportable_ic = int(posterior_meals["ic_posterior_mean"].notna().sum()) if not posterior_meals.empty else 0
    lines = [
        "# Latent Meal Fit Summary",
        "",
        "- fit_family: constrained_latent_response_approximation",
        f"- selected_model: {selected_model}",
        f"- meal_count: {int(len(window_metrics))}",
        f"- reportable_morning_ic_windows: {reportable_ic}",
        f"- cohort_icr_prior: {float(params['icr_prior']):.3f}",
        f"- cohort_peak_per_carb: {float(params['peak_per_carb']):.3f}",
        f"- cohort_auc_per_carb: {float(params['auc_per_carb']):.3f}",
        f"- leave_one_window_out_peak_delta_mae: {'NA' if not np.isfinite(peak_mae) else f'{peak_mae:.3f}'}",
        "",
        "## Interpretation",
        "",
        "- This is a research-only per-window posterior estimator on accepted first-meal windows.",
        "- Latent carb estimates are anchored by configured therapy settings, bolus, and post-meal glucose response; exact carb truth remains unobserved when explicit carb source data is unavailable.",
        "- Morning effective I/C is only reported for windows with sufficiently narrow posterior intervals and low residual confounding.",
        "",
        "## Accepted Window Preview",
        "",
        "| meal_id | timestamp | evidence_source | latent_carbs | ic_posterior_mean | identifiability_status |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    if posterior_meals.empty:
        lines.append("| none | none | none | NA | NA | none |")
    else:
        for row in posterior_meals.head(10).itertuples(index=False):
            lines.append(
                f"| {row.meal_id} | {row.timestamp} | {row.evidence_source} | {float(row.latent_carbs_posterior_mean):.2f} | {'NA' if pd.isna(row.ic_posterior_mean) else f'{float(row.ic_posterior_mean):.2f}'} | {row.identifiability_status} |"
            )
    return "\n".join(lines) + "\n"


def _render_latent_carb_estimator_report(
    latent_carb_estimates: pd.DataFrame,
    *,
    settings: LatentMealTherapySettings,
    selected_model: str,
    params: dict[str, float],
    effective_icr_evidence: pd.DataFrame | None = None,
) -> str:
    lines = ["# Breakfast Carb Estimates and Effective I:C", ""]
    lines.extend(_render_effective_icr_section(effective_icr_evidence))
    lines.extend(
        [
            "## How to read the carb estimates",
            "",
            "- Each breakfast gets a glucose-implied carb amount: the amount that best explains the post-meal CGM curve, given the bolus and Control-IQ delivery.",
            "- These are estimates under the programmed I:C and correction factor, not verified grams eaten.",
            "- The 90% ranges assume the response model, CGM quality and breakfast-selection rules hold.",
            "",
            "## Technical record",
            "",
            f"- selected_model: {selected_model}",
            f"- estimate_rows: {int(len(latent_carb_estimates))}",
            f"- target_glucose_mgdl: {float(settings.target_glucose_mgdl):.1f}",
            f"- insulin_action_minutes: {int(settings.insulin_action_minutes)}",
            f"- carb_absorption_minutes_prior: {int(settings.carb_absorption_minutes_prior)}",
            f"- max_plausible_meal_carbs: {float(settings.max_plausible_meal_carbs):.1f}",
            f"- cohort_peak_per_carb: {float(params['peak_per_carb']):.3f}",
            f"- cohort_auc_per_carb: {float(params['auc_per_carb']):.3f}",
            "",
        ]
    )
    if latent_carb_estimates.empty:
        lines.extend(["No latent carb estimates were generated.", ""])
        return "\n".join(lines)

    interval_width = pd.to_numeric(latent_carb_estimates["posterior_interval_width_90"], errors="coerce")
    confidence = pd.to_numeric(latent_carb_estimates["confidence"], errors="coerce")
    success_count = int(latent_carb_estimates["post_meal_control_success"].astype(bool).sum())
    evidence_counts = latent_carb_estimates["evidence_source"].astype(str).value_counts().to_dict()
    tier_counts = latent_carb_estimates.get("analysis_tier", pd.Series("clean_reference", index=latent_carb_estimates.index)).astype(str).value_counts().to_dict()
    activity_columns = [
        "recent_workout_context",
        "recent_workout_12h_context",
        "light_activity_context",
        "moderate_activity_context",
        "high_activity_context",
    ]
    activity_counts = {
        column: int(pd.to_numeric(latent_carb_estimates.get(column, pd.Series(0, index=latent_carb_estimates.index)), errors="coerce").fillna(0).gt(0).sum())
        for column in activity_columns
    }
    exercise_flag_count = int(latent_carb_estimates["flags"].astype(str).str.contains("exercise_context", regex=False).sum())
    soft_activity_flag_count = int(latent_carb_estimates["flags"].astype(str).str.contains("soft_activity_context", regex=False).sum())
    activity_roll = pd.to_numeric(
        latent_carb_estimates.get("health_activity_roll_sum_60m", pd.Series(np.nan, index=latent_carb_estimates.index)),
        errors="coerce",
    )
    estimator_series = latent_carb_estimates.get("estimator", pd.Series("summary_heuristic", index=latent_carb_estimates.index)).astype(str)
    estimator_counts = estimator_series.value_counts().to_dict()
    contraction = pd.to_numeric(latent_carb_estimates.get("prior_contraction", pd.Series(np.nan, index=latent_carb_estimates.index)), errors="coerce")
    fit_rmse = pd.to_numeric(latent_carb_estimates.get("trajectory_fit_rmse_mgdl", pd.Series(np.nan, index=latent_carb_estimates.index)), errors="coerce")
    poor_fit_count = int(latent_carb_estimates["flags"].astype(str).str.contains("trajectory_poor_fit", regex=False).sum())
    lines.extend(
        [
            "### Estimator",
            "",
            "- `trajectory` rows fit the whole post-meal CGM trace against absorbed insulin (see `latent_meal_trajectory.py`).",
            "- `summary_heuristic` rows fall back to the older peak/AUC posterior because the trace or insulin bookkeeping was unusable.",
            "  In synthetic truth recovery (`bayesian-t1dm validate-latent-meal`) that fallback is biased high with under-covering intervals.",
            f"- trajectory_rows: {int(estimator_counts.get('trajectory', 0))}",
            f"- summary_heuristic_rows: {int(estimator_counts.get('summary_heuristic', 0))}",
            f"- median_prior_contraction: {'NA' if contraction.dropna().empty else f'{float(contraction.median()):.2f}'}"
            " (0 = the trace added nothing beyond bolus x I:C, 1 = the trace dominates)",
            f"- median_trajectory_fit_rmse_mgdl: {'NA' if fit_rmse.dropna().empty else f'{float(fit_rmse.median()):.1f}'}",
            f"- trajectory_poor_fit_rows: {poor_fit_count}",
            "",
            "### Cohort Summary",
            "",
            f"- post_meal_control_success_rows: {success_count}",
            f"- median_90pct_interval_width: {float(interval_width.median()):.2f}",
            f"- median_confidence: {float(confidence.median()):.3f}",
            "",
            "### Analysis Tiers",
            "",
            "| analysis_tier | count | median_90pct_interval_width | median_confidence |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for tier in LATENT_MEAL_ANALYSIS_TIERS:
        tier_name = str(tier["analysis_tier"])
        subset = latent_carb_estimates.loc[latent_carb_estimates.get("analysis_tier", pd.Series("clean_reference", index=latent_carb_estimates.index)).astype(str).eq(tier_name)]
        if subset.empty:
            lines.append(f"| {tier_name} | 0 | NA | NA |")
        else:
            subset_width = pd.to_numeric(subset["posterior_interval_width_90"], errors="coerce")
            subset_confidence = pd.to_numeric(subset["confidence"], errors="coerce")
            lines.append(
                f"| {tier_name} | {int(tier_counts.get(tier_name, 0))} | {float(subset_width.median()):.2f} | {float(subset_confidence.median()):.3f} |"
            )
    lines.extend(
        [
            "",
            "### Activity Context",
            "",
            f"- recent_workout_context_rows: {activity_counts['recent_workout_context']}",
            f"- recent_workout_12h_context_rows: {activity_counts['recent_workout_12h_context']}",
            f"- light_activity_context_rows: {activity_counts['light_activity_context']}",
            f"- moderate_activity_context_rows: {activity_counts['moderate_activity_context']}",
            f"- high_activity_context_rows: {activity_counts['high_activity_context']}",
            f"- hard_exercise_flag_rows: {exercise_flag_count}",
            f"- soft_activity_flag_rows: {soft_activity_flag_count}",
            f"- median_health_activity_roll_sum_60m: {'NA' if activity_roll.dropna().empty else f'{float(activity_roll.median()):.1f}'}",
            "",
            "### Evidence Sources",
            "",
            "| evidence_source | count |",
            "| --- | ---: |",
        ]
    )
    for evidence_source, count in sorted(evidence_counts.items()):
        lines.append(f"| {evidence_source} | {int(count)} |")
    lines.extend(
        [
            "",
            "### Estimate Preview",
            "",
            "| meal_id | meal_start | bolus_implied_carbs | latent_carb_median | latent_carb_90pct_interval | confidence | flags |",
            "| --- | --- | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for row in latent_carb_estimates.head(12).itertuples(index=False):
        lines.append(
            f"| {row.meal_id} | {row.meal_start} | {float(row.bolus_implied_carbs):.1f} | {float(row.latent_carb_median):.1f} | "
            f"[{float(row.latent_carb_lower_90):.1f}, {float(row.latent_carb_upper_90):.1f}] | {float(row.confidence):.3f} | {row.flags or 'none'} |"
        )
    return "\n".join(lines) + "\n"


def _render_latent_meal_confidence_report(posterior_meals: pd.DataFrame) -> str:
    lines = [
        "# Latent Meal Confidence Report",
        "",
    ]
    if posterior_meals.empty:
        lines.extend(
            [
                "- meal_count: 0",
                "",
                "No posterior meals were generated.",
            ]
        )
        return "\n".join(lines) + "\n"
    confidence_column = "confidence" if "confidence" in posterior_meals.columns else "carb_accuracy_score"
    confidence = pd.to_numeric(posterior_meals[confidence_column], errors="coerce").fillna(0.0)
    interval_width = pd.to_numeric(posterior_meals.get("posterior_interval_width_90", pd.Series(np.nan, index=posterior_meals.index)), errors="coerce")
    tiers = pd.Series(
        np.where(confidence >= 0.8, "high", np.where(confidence >= 0.6, "medium", "low")),
        index=posterior_meals.index,
        dtype=str,
    )
    tier_counts = tiers.value_counts().to_dict()
    evidence_counts = posterior_meals["evidence_source"].astype(str).value_counts().to_dict()
    lines.extend(
        [
            f"- meal_count: {int(len(posterior_meals))}",
            f"- high_confidence_windows: {int(tier_counts.get('high', 0))}",
            f"- medium_confidence_windows: {int(tier_counts.get('medium', 0))}",
            f"- low_confidence_windows: {int(tier_counts.get('low', 0))}",
            f"- median_90pct_interval_width: {'NA' if interval_width.dropna().empty else f'{float(interval_width.median()):.2f}'}",
            "",
            "## Evidence Sources",
            "",
            "| evidence_source | count |",
            "| --- | ---: |",
        ]
    )
    for evidence_source, count in sorted(evidence_counts.items()):
        lines.append(f"| {evidence_source} | {int(count)} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Accuracy scores compare latent carb exposure to the best available auxiliary anchor.",
            "- When explicit carb source data is unavailable, scores are capped and should be interpreted as consistency-with-prior rather than correctness against ground truth.",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_latent_meal_model_comparison(model_comparison: pd.DataFrame) -> str:
    lines = [
        "# Latent Meal Model Comparison",
        "",
        "| model | meal_count | stated_carb_mae | peak_delta_mae | peak_delta_correlation | selected |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    if model_comparison.empty:
        lines.append("| none | 0 | NA | NA | NA | no |")
        return "\n".join(lines) + "\n"
    for row in model_comparison.itertuples(index=False):
        lines.append(
            f"| {row.model} | {int(row.meal_count)} | {'NA' if pd.isna(row.stated_carb_mae) else f'{float(row.stated_carb_mae):.3f}'} | {'NA' if pd.isna(row.peak_delta_mae) else f'{float(row.peak_delta_mae):.3f}'} | {'NA' if pd.isna(row.peak_delta_correlation) else f'{float(row.peak_delta_correlation):.3f}'} | {'yes' if bool(row.selected) else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def run_latent_meal_icr_research(
    dataset: AnalysisReadyHealthDataset,
    *,
    segments: tuple[TherapySegment, ...] | None = None,
    meal_proxy_mode: str = "strict",
    research_scope: str = "foundation",
    therapy_settings: LatentMealTherapySettings | None = None,
    excluded_periods: tuple[ExcludedPeriod, ...] = (),
    nutrition: pd.DataFrame | None = None,
    unlogged: pd.DataFrame | None = None,
    exercise: pd.DataFrame | None = None,
    meal_segments: tuple[str, ...] = ("morning",),
    programmed_units: pd.Series | None = None,
) -> LatentMealResearchResult:
    if research_scope not in {"foundation", "full"}:
        raise ValueError(f"Unsupported research_scope: {research_scope}")
    segments = segments or parse_therapy_segments()
    research_frame = build_therapy_research_frame(dataset, segments=segments, meal_proxy_mode=meal_proxy_mode, excluded_periods=excluded_periods)
    meal_event_registry = build_meal_event_registry(research_frame)
    if unlogged is None:
        unlogged = meal_flags.unlogged_intervals(research_frame, nutrition)
    clean_meal_windows = build_first_meal_clean_window_registry(research_frame, unlogged=unlogged, exercise=exercise, meal_segments=meal_segments)
    first_meal_exclusion_summary = build_first_meal_exclusion_summary(clean_meal_windows)
    if research_scope == "foundation":
        gate, gate_markdown = _latent_meal_gate(dataset, clean_meal_windows, research_scope=research_scope)

        return LatentMealResearchResult(
            prepared_dataset=dataset,
            research_frame=research_frame,
            research_gate=gate,
            meal_event_registry=meal_event_registry,
            meal_windows=clean_meal_windows,
            first_meal_exclusion_summary=first_meal_exclusion_summary,
            posterior_meals=pd.DataFrame(columns=["meal_id", "timestamp", "segment", "stated_carbs", "latent_carbs_posterior_mean", "carb_accuracy_score"]),
            model_comparison=pd.DataFrame(columns=["model", "meal_count", "stated_carb_mae", "peak_delta_mae", "peak_delta_correlation", "selected"]),
            research_gate_markdown=gate_markdown,
            meal_truth_semantics_report_markdown=_render_meal_truth_semantics_report(dataset, research_frame),
            meal_window_audit_markdown=_render_latent_meal_window_audit(clean_meal_windows, first_meal_exclusion_summary),
            fit_summary_markdown="",
            confidence_report_markdown="",
            model_comparison_markdown="",
            segments=segments,
            meal_proxy_mode=meal_proxy_mode,
            research_scope=research_scope,
            therapy_settings=therapy_settings,
            latent_carb_estimates=pd.DataFrame(columns=LATENT_CARB_ESTIMATE_COLUMNS),
            posterior_samples=pd.DataFrame(columns=["meal_id", "sample_index", "analysis_tier", "latent_carb_grams", "absorption_minutes", "posterior_weight"]),
            tier_summary=pd.DataFrame(columns=["analysis_tier", "analysis_tier_rank", "analysis_tier_label", "accepted_windows"]),
            tier_summary_markdown="",
        )

    meal_windows = build_latent_meal_tiered_window_registry(clean_meal_windows)
    tier_summary = build_latent_meal_tier_summary(meal_windows)
    tier_summary_markdown = _render_latent_meal_tier_summary(tier_summary)
    window_metrics = _accepted_first_meal_window_metrics(research_frame, meal_event_registry, meal_windows)
    settings = _settings_for_estimation(therapy_settings)
    window_metrics = _attach_resolved_settings(window_metrics, settings)
    model_comparison, selected_model, params = _latent_meal_model_comparison(window_metrics, therapy_settings=settings)
    missing_basal_windows: set[str] = set()
    trajectories = _first_meal_window_trajectories(research_frame, window_metrics, programmed_units=programmed_units,
                                                 missing_basal_windows=missing_basal_windows)
    posterior_meals = _build_latent_meal_posterior_meals(
        window_metrics,
        model_name=selected_model,
        params=params,
        therapy_settings=settings,
        trajectories=trajectories,
    )
    latent_carb_estimates = _build_latent_carb_estimates(posterior_meals)
    posterior_samples = _build_latent_carb_posterior_samples(window_metrics, settings=settings, params=params, trajectories=trajectories)
    window_metrics = _attach_logged_food(window_metrics, nutrition)
    bevel_scale, bevel_pairs = _logged_food_scale(research_frame, nutrition)
    support = [] if programmed_units is not None else None
    effective_icr_evidence = _effective_icr_evidence(window_metrics, trajectories, settings=settings, bevel_scale=bevel_scale,
                                                    bevel_pairs=bevel_pairs, support=support, missing_basal_windows=missing_basal_windows)
    gate, gate_markdown = _render_full_latent_meal_gate(dataset, meal_windows, posterior_meals, research_scope=research_scope)
    return LatentMealResearchResult(
        prepared_dataset=dataset,
        research_frame=research_frame,
        research_gate=gate,
        meal_event_registry=meal_event_registry,
        meal_windows=meal_windows,
        first_meal_exclusion_summary=first_meal_exclusion_summary,
        posterior_meals=posterior_meals,
        model_comparison=model_comparison,
        research_gate_markdown=gate_markdown,
        meal_truth_semantics_report_markdown=_render_meal_truth_semantics_report(dataset, research_frame),
        meal_window_audit_markdown=_render_latent_meal_window_audit(meal_windows, first_meal_exclusion_summary),
        fit_summary_markdown=_render_latent_meal_fit_summary(
            window_metrics,
            posterior_meals,
            model_comparison,
            selected_model=selected_model,
            params=params,
        ),
        confidence_report_markdown=_render_latent_meal_confidence_report(posterior_meals),
        model_comparison_markdown=_render_latent_meal_model_comparison(model_comparison),
        segments=segments,
        meal_proxy_mode=meal_proxy_mode,
        research_scope=research_scope,
        therapy_settings=settings,
        latent_carb_estimates=latent_carb_estimates,
        posterior_samples=posterior_samples,
        tier_summary=tier_summary,
        tier_summary_markdown=tier_summary_markdown,
        latent_carb_estimator_report_markdown=_render_latent_carb_estimator_report(
            latent_carb_estimates,
            settings=settings,
            selected_model=selected_model,
            params=params,
            effective_icr_evidence=effective_icr_evidence,
        ),
        effective_icr_evidence=effective_icr_evidence,
        icr_support=pd.DataFrame(support, columns=["date", "segment", "historical_basal_supported", "settings_supported", "in_estimate"]) if support is not None else None,
    )


def write_latent_meal_research_artifacts(
    result: LatentMealResearchResult,
    report_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(report_dir)
    root.mkdir(parents=True, exist_ok=True)
    if result.research_scope == "foundation":
        paths = {
            "research_gate": root / "latent_meal_research_gate.md",
            "meal_truth_semantics_report": root / "meal_truth_semantics_report.md",
            "meal_event_registry": root / "meal_event_registry.csv",
            "first_meal_clean_window_registry": root / "first_meal_clean_window_registry.csv",
            "first_meal_clean_window_audit": root / "first_meal_clean_window_audit.md",
            "first_meal_exclusion_summary": root / "first_meal_exclusion_summary.csv",
        }
        paths["research_gate"].write_text(result.research_gate_markdown, encoding="utf-8")
        paths["meal_truth_semantics_report"].write_text(result.meal_truth_semantics_report_markdown, encoding="utf-8")
        result.meal_event_registry.to_csv(paths["meal_event_registry"], index=False)
        result.meal_windows.to_csv(paths["first_meal_clean_window_registry"], index=False)
        paths["first_meal_clean_window_audit"].write_text(result.meal_window_audit_markdown, encoding="utf-8")
        result.first_meal_exclusion_summary.to_csv(paths["first_meal_exclusion_summary"], index=False)
        return paths
    paths = {
        "research_gate": root / "latent_meal_research_gate.md",
        "meal_event_registry": root / "meal_event_registry.csv",
        "meal_window_audit": root / "meal_window_audit.md",
        "latent_meal_window_audit": root / "latent_meal_window_audit.csv",
        "latent_carb_estimator_report": root / "latent_carb_estimator_report.md",
        "latent_carb_estimates_report_copy": root / "latent_carb_estimates.csv",
        "fit_summary": root / "latent_meal_fit_summary.md",
        "posterior_meals": root / "latent_meal_posterior_meals.csv",
        "confidence_report": root / "latent_meal_confidence_report.md",
        "model_comparison": root / "latent_meal_model_comparison.md",
        "tier_summary": root / "latent_meal_tier_summary.csv",
        "tier_summary_report": root / "latent_meal_tier_summary.md",
    }
    paths["research_gate"].write_text(result.research_gate_markdown, encoding="utf-8")
    result.meal_event_registry.to_csv(paths["meal_event_registry"], index=False)
    paths["meal_window_audit"].write_text(result.meal_window_audit_markdown, encoding="utf-8")
    result.meal_windows.to_csv(paths["latent_meal_window_audit"], index=False)
    paths["latent_carb_estimator_report"].write_text(result.latent_carb_estimator_report_markdown, encoding="utf-8")
    paths["fit_summary"].write_text(result.fit_summary_markdown, encoding="utf-8")
    result.posterior_meals.to_csv(paths["posterior_meals"], index=False)
    paths["confidence_report"].write_text(result.confidence_report_markdown, encoding="utf-8")
    paths["model_comparison"].write_text(result.model_comparison_markdown, encoding="utf-8")
    tier_summary = result.tier_summary if result.tier_summary is not None else pd.DataFrame()
    tier_summary.to_csv(paths["tier_summary"], index=False)
    paths["tier_summary_report"].write_text(result.tier_summary_markdown or _render_latent_meal_tier_summary(tier_summary), encoding="utf-8")
    cache_root = Path(cache_dir) if cache_dir is not None else root
    cache_root.mkdir(parents=True, exist_ok=True)
    latent_estimates = result.latent_carb_estimates if result.latent_carb_estimates is not None else pd.DataFrame(columns=LATENT_CARB_ESTIMATE_COLUMNS)
    posterior_samples = result.posterior_samples if result.posterior_samples is not None else pd.DataFrame(columns=["meal_id", "sample_index", "analysis_tier", "latent_carb_grams", "absorption_minutes", "posterior_weight"])
    paths["latent_carb_estimates"] = cache_root / "latent_carb_estimates.csv"
    paths["therapy_settings_used"] = cache_root / "therapy_settings_used.json"
    paths["posterior_samples"] = cache_root / "latent_carb_posterior_samples.parquet"
    latent_estimates.to_csv(paths["latent_carb_estimates"], index=False)
    latent_estimates.to_csv(paths["latent_carb_estimates_report_copy"], index=False)
    settings = result.therapy_settings if result.therapy_settings is not None else LatentMealTherapySettings.default()
    paths["therapy_settings_used"].write_text(json.dumps(settings.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    posterior_samples.to_parquet(paths["posterior_samples"], index=False)
    paths["effective_icr_evidence"] = root / "effective_icr_evidence.csv"
    effective_icr = result.effective_icr_evidence if result.effective_icr_evidence is not None else pd.DataFrame(columns=EFFECTIVE_ICR_EVIDENCE_COLUMNS)
    effective_icr.to_csv(paths["effective_icr_evidence"], index=False)
    return paths


def _evenly_spaced_background_days(
    day_summary: pd.DataFrame,
    *,
    excluded_dates: set[pd.Timestamp],
    background_days: int,
) -> pd.DataFrame:
    if background_days <= 0 or day_summary.empty:
        return pd.DataFrame(columns=["date", "row_count", "bolus_rows", "selection_type", "selection_reason"])
    eligible = day_summary.loc[~day_summary["date"].isin(excluded_dates)].copy()
    if eligible.empty:
        return pd.DataFrame(columns=["date", "row_count", "bolus_rows", "selection_type", "selection_reason"])
    eligible = eligible.sort_values("date").reset_index(drop=True)
    picks = min(int(background_days), len(eligible))
    indices = np.linspace(0, len(eligible) - 1, num=picks).round().astype(int)
    indices = sorted(set(int(index) for index in indices))
    background = eligible.iloc[indices].copy().reset_index(drop=True)
    background["selection_type"] = "background"
    background["selection_reason"] = "evenly_spaced_non_candidate_day"
    return background


def _render_latent_meal_fixture_summary(
    *,
    source_result: LatentMealResearchResult,
    fixture_result: LatentMealResearchResult,
    selected_day_manifest: pd.DataFrame,
    background_days_requested: int,
    seed: int,
) -> str:
    source_frame = source_result.research_frame.copy()
    fixture_frame = fixture_result.research_frame.copy()
    source_day_count = int(pd.to_datetime(source_frame.get("timestamp"), errors="coerce").dt.normalize().nunique()) if not source_frame.empty else 0
    fixture_day_count = int(pd.to_datetime(fixture_frame.get("timestamp"), errors="coerce").dt.normalize().nunique()) if not fixture_frame.empty else 0
    source_candidate_days = int(pd.to_datetime(source_result.meal_windows.get("date"), errors="coerce").nunique()) if not source_result.meal_windows.empty else 0
    fixture_candidate_days = int(pd.to_datetime(fixture_result.meal_windows.get("date"), errors="coerce").nunique()) if not fixture_result.meal_windows.empty else 0
    background_selected = int(selected_day_manifest["selection_type"].astype(str).eq("background").sum()) if not selected_day_manifest.empty else 0
    source_exclusions = (
        source_result.first_meal_exclusion_summary.set_index("reason")["count"].to_dict()
        if not source_result.first_meal_exclusion_summary.empty
        else {}
    )
    fixture_exclusions = (
        fixture_result.first_meal_exclusion_summary.set_index("reason")["count"].to_dict()
        if not fixture_result.first_meal_exclusion_summary.empty
        else {}
    )
    all_reasons = sorted(set(source_exclusions) | set(fixture_exclusions))
    lines = [
        "# Latent Meal Representative Fixture",
        "",
        f"- source_rows: {int(len(source_result.prepared_dataset.frame))}",
        f"- fixture_rows: {int(len(fixture_result.prepared_dataset.frame))}",
        f"- source_days: {source_day_count}",
        f"- fixture_days: {fixture_day_count}",
        f"- source_candidate_days: {source_candidate_days}",
        f"- fixture_candidate_days: {fixture_candidate_days}",
        f"- candidate_day_retention: {float(fixture_candidate_days / source_candidate_days):.3f}" if source_candidate_days else "- candidate_day_retention: NA",
        f"- background_days_requested: {int(background_days_requested)}",
        f"- background_days_selected: {background_selected}",
        f"- explicit_carb_source_available: {bool(getattr(source_result.prepared_dataset, 'explicit_carb_source_available', False))}",
        f"- seed: {int(seed)}",
        "",
        "## Exclusion Coverage",
        "",
        "| reason | source_count | fixture_count | retained |",
        "| --- | ---: | ---: | --- |",
    ]
    if not all_reasons:
        lines.append("| none | 0 | 0 | NA |")
    else:
        for reason in all_reasons:
            source_count = int(source_exclusions.get(reason, 0))
            fixture_count = int(fixture_exclusions.get(reason, 0))
            retained = "yes" if fixture_count > 0 else "no"
            lines.append(f"| {reason} | {source_count} | {fixture_count} | {retained} |")
    lines.extend(
        [
            "",
            "## Selected Days",
            "",
            "| date | selection_type | selection_reason | row_count | bolus_rows | candidate_id | source_quality_grade | exclusion_reasons |",
            "| --- | --- | --- | ---: | ---: | --- | --- | --- |",
        ]
    )
    if selected_day_manifest.empty:
        lines.append("| none | none | none | 0 | 0 | none | none | none |")
    else:
        for row in selected_day_manifest.sort_values("date").itertuples(index=False):
            lines.append(
                f"| {pd.Timestamp(row.date).date()} | {row.selection_type} | {row.selection_reason} | {int(row.row_count)} | {int(row.bolus_rows)} | {getattr(row, 'candidate_id', '') or ''} | {getattr(row, 'source_quality_grade', '') or ''} | {getattr(row, 'exclusion_reasons', '') or ''} |"
            )
    return "\n".join(lines) + "\n"


def build_representative_latent_meal_fixture(
    dataset: AnalysisReadyHealthDataset,
    *,
    segments: tuple[TherapySegment, ...] | None = None,
    meal_proxy_mode: str = "strict",
    background_days: int = 7,
    seed: int = 17,
) -> LatentMealFixtureResult:
    if background_days < 0:
        raise ValueError("background_days must be >= 0")
    segments = segments or parse_therapy_segments()
    source_result = run_latent_meal_icr_research(
        dataset,
        segments=segments,
        meal_proxy_mode=meal_proxy_mode,
        research_scope="foundation",
    )
    research_frame = source_result.research_frame.copy()
    if research_frame.empty:
        selected_day_manifest = pd.DataFrame(
            columns=[
                "date",
                "selection_type",
                "selection_reason",
                "row_count",
                "bolus_rows",
                "candidate_id",
                "source_quality_grade",
                "exclusion_reasons",
            ]
        )
        fixture_dataset = replace(dataset, frame=dataset.frame.iloc[0:0].copy())
        fixture_result = run_latent_meal_icr_research(
            fixture_dataset,
            segments=segments,
            meal_proxy_mode=meal_proxy_mode,
            research_scope="foundation",
        )
        summary_markdown = _render_latent_meal_fixture_summary(
            source_result=source_result,
            fixture_result=fixture_result,
            selected_day_manifest=selected_day_manifest,
            background_days_requested=background_days,
            seed=seed,
        )
        return LatentMealFixtureResult(
            source_result=source_result,
            fixture_result=fixture_result,
            fixture_dataset=fixture_dataset,
            selected_day_manifest=selected_day_manifest,
            summary_markdown=summary_markdown,
            background_days_requested=background_days,
            seed=seed,
        )

    ordered = research_frame.copy()
    ordered["date"] = pd.to_datetime(ordered["timestamp"], errors="coerce").dt.normalize()
    ordered = ordered.dropna(subset=["date"]).reset_index(drop=True)
    day_summary = (
        ordered.groupby("date", as_index=False)
        .agg(
            row_count=("timestamp", "size"),
            bolus_rows=("bolus_units", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0.0).gt(0).sum())),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    candidate_manifest = source_result.meal_windows.copy()
    if candidate_manifest.empty:
        candidate_manifest = pd.DataFrame(columns=["date", "candidate_id", "source_quality_grade", "exclusion_reasons"])
    else:
        candidate_manifest = candidate_manifest.loc[:, [column for column in ["date", "candidate_id", "source_quality_grade", "exclusion_reasons"] if column in candidate_manifest.columns]].copy()
        candidate_manifest["date"] = pd.to_datetime(candidate_manifest["date"], errors="coerce").dt.normalize()
        candidate_manifest = candidate_manifest.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="first")
    candidate_manifest = candidate_manifest.merge(day_summary, on="date", how="left")
    candidate_manifest["selection_type"] = "candidate"
    candidate_manifest["selection_reason"] = "first_meal_candidate_day"
    excluded_dates = set(candidate_manifest["date"].dropna().tolist())
    background_manifest = _evenly_spaced_background_days(
        day_summary,
        excluded_dates=excluded_dates,
        background_days=background_days,
    )
    if not background_manifest.empty:
        background_manifest["candidate_id"] = ""
        background_manifest["source_quality_grade"] = ""
        background_manifest["exclusion_reasons"] = ""
    selected_day_manifest = pd.concat([candidate_manifest, background_manifest], ignore_index=True, sort=False)
    selected_day_manifest = selected_day_manifest.loc[:, [
        "date",
        "selection_type",
        "selection_reason",
        "row_count",
        "bolus_rows",
        "candidate_id",
        "source_quality_grade",
        "exclusion_reasons",
    ]].sort_values(["date", "selection_type"]).reset_index(drop=True)

    selected_dates = set(pd.to_datetime(selected_day_manifest["date"], errors="coerce").dropna().tolist())
    subset_frame = dataset.frame.copy()
    subset_frame = subset_frame.loc[
        pd.to_datetime(subset_frame["timestamp"], errors="coerce").dt.normalize().isin(selected_dates)
    ].copy().reset_index(drop=True)
    fixture_dataset = replace(dataset, frame=subset_frame)
    fixture_result = run_latent_meal_icr_research(
        fixture_dataset,
        segments=segments,
        meal_proxy_mode=meal_proxy_mode,
        research_scope="foundation",
    )
    summary_markdown = _render_latent_meal_fixture_summary(
        source_result=source_result,
        fixture_result=fixture_result,
        selected_day_manifest=selected_day_manifest,
        background_days_requested=background_days,
        seed=seed,
    )
    return LatentMealFixtureResult(
        source_result=source_result,
        fixture_result=fixture_result,
        fixture_dataset=fixture_dataset,
        selected_day_manifest=selected_day_manifest,
        summary_markdown=summary_markdown,
        background_days_requested=background_days,
        seed=seed,
    )


def write_representative_latent_meal_fixture_artifacts(
    result: LatentMealFixtureResult,
    report_dir: str | Path,
    *,
    prepared_csv_path: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(report_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "fixture_summary": root / "latent_meal_fixture_summary.md",
        "selected_day_manifest": root / "latent_meal_fixture_day_manifest.csv",
    }
    if prepared_csv_path is not None:
        paths["prepared_model_data"] = Path(prepared_csv_path)
        paths["prepared_model_data"].parent.mkdir(parents=True, exist_ok=True)
        result.fixture_dataset.frame.to_csv(paths["prepared_model_data"], index=False)
    else:
        paths["prepared_model_data"] = root / "prepared_model_data_5min.csv"
        result.fixture_dataset.frame.to_csv(paths["prepared_model_data"], index=False)
    paths["fixture_summary"].write_text(result.summary_markdown, encoding="utf-8")
    result.selected_day_manifest.to_csv(paths["selected_day_manifest"], index=False)
    paths.update(write_latent_meal_research_artifacts(result.fixture_result, root))
    return paths
