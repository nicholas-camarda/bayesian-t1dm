from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .evaluate import CalibrationSummary, FoldResult, WalkForwardReport, walk_forward_splits
from .features import FeatureConfig, FeatureFrame, recompute_scenario_features
from .health_auto_export import AnalysisReadyHealthDataset
from .model import BayesianGlucoseModel, FitDiagnostics


TARGET_GLUCOSE_COLUMN = "target_glucose"
TARGET_DELTA_COLUMN = "target_delta"
DEFAULT_SEGMENT_SPEC = "overnight=00:00-06:00,morning=06:00-11:00,afternoon=11:00-17:00,evening=17:00-24:00"
RESEARCH_FEATURE_EXCLUDE = {
    "timestamp",
    "glucose_observed",
    TARGET_GLUCOSE_COLUMN,
    TARGET_DELTA_COLUMN,
    "therapy_segment",
    "therapy_segment_order",
}
TASK_PARAMETER_LABELS = {
    "basal": "basal",
    "icr": "I/C ratio",
    "sensitivity_factor": "sensitivity factor",
}


def _column_or_default(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index, dtype=float)


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
class TherapyTask:
    name: str
    parameter: str
    context_column: str
    staged: bool = False


@dataclass(frozen=True)
class TherapyResearchResult:
    prepared_dataset: AnalysisReadyHealthDataset
    research_frame: pd.DataFrame
    research_gate: pd.DataFrame
    feature_registry: pd.DataFrame
    meal_proxy_audit: pd.DataFrame
    model_comparison: pd.DataFrame
    segment_evidence: pd.DataFrame
    recommendations: pd.DataFrame
    research_gate_markdown: str
    feature_audit_markdown: str
    meal_proxy_audit_markdown: str
    model_comparison_markdown: str
    recommendation_markdown: str
    tandem_source_report_markdown: str
    apple_source_report_markdown: str
    source_numeric_summary: pd.DataFrame
    source_missingness_summary: pd.DataFrame
    segments: tuple[TherapySegment, ...]
    include_models: tuple[str, ...]
    meal_proxy_mode: str
    ic_policy: str


@dataclass(frozen=True)
class TherapyInfraValidationResult:
    scenario_results: pd.DataFrame
    report_markdown: str
    recommendation_audit_markdown: str


@dataclass(frozen=True)
class LatentMealResearchResult:
    prepared_dataset: AnalysisReadyHealthDataset
    research_frame: pd.DataFrame
    research_gate: pd.DataFrame
    meal_event_registry: pd.DataFrame
    meal_windows: pd.DataFrame
    posterior_meals: pd.DataFrame
    model_comparison: pd.DataFrame
    research_gate_markdown: str
    meal_window_audit_markdown: str
    fit_summary_markdown: str
    confidence_report_markdown: str
    model_comparison_markdown: str
    segments: tuple[TherapySegment, ...]
    meal_proxy_mode: str


@dataclass(frozen=True)
class _ModelPrediction:
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class _ModelFitBundle:
    feature_names: list[str]
    prediction: callable
    diagnostics: FitDiagnostics | None = None
    feature_importance: dict[str, float] | None = None
    feature_importance_by_segment: dict[str, dict[str, float]] | None = None


TASKS: tuple[TherapyTask, ...] = (
    TherapyTask(name="basal", parameter="basal", context_column="basal_context"),
    TherapyTask(name="icr", parameter="I/C ratio", context_column="meal_context"),
    TherapyTask(name="sensitivity_factor", parameter="sensitivity factor", context_column="correction_context", staged=True),
)

IDENTIFIABILITY_LEVELS = {
    "directly_observed",
    "proxy_supported",
    "weakly_identified",
    "not_identified",
}


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


def parse_model_list(value: str | None) -> tuple[str, ...]:
    raw = value or "bayesian,ridge,elastic_net,segmented_ridge,tree_boost,ensemble"
    names = tuple(dict.fromkeys(item.strip() for item in raw.split(",") if item.strip()))
    if not names:
        raise ValueError("At least one model family must be requested")
    return names


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


def _summarize_source_report_card(
    frame: pd.DataFrame,
    *,
    family: str,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    present_columns = [column for column in columns if column in frame.columns]
    numeric_rows: list[dict[str, Any]] = []
    missingness_rows: list[dict[str, Any]] = []
    impossible_rows: list[str] = []
    timestamp = pd.to_datetime(frame.get("timestamp", pd.Series(dtype="datetime64[ns]")), errors="coerce")
    for column in present_columns:
        series = frame[column]
        missingness_rows.append(
            {
                "family": family,
                "feature": column,
                "missing_fraction": float(series.isna().mean()),
                "non_missing_count": int(series.notna().sum()),
            }
        )
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            q1 = float(numeric.quantile(0.25))
            q3 = float(numeric.quantile(0.75))
            numeric_rows.append(
                {
                    "family": family,
                    "feature": column,
                    "mean": float(numeric.mean()),
                    "median": float(numeric.median()),
                    "iqr": float(q3 - q1),
                    "min": float(numeric.min()),
                    "max": float(numeric.max()),
                }
            )
            if column == "glucose":
                invalid = int(numeric.lt(40).sum() + numeric.gt(450).sum())
            elif column == "basal_units_per_hour":
                invalid = int(numeric.lt(0).sum() + numeric.gt(10).sum())
            elif column == "bolus_units":
                invalid = int(numeric.lt(0).sum() + numeric.gt(40).sum())
            elif "heart_rate" in column:
                invalid = int(numeric.lt(25).sum() + numeric.gt(220).sum())
            elif "hrv" in column:
                invalid = int(numeric.lt(0).sum() + numeric.gt(250).sum())
            elif "respiratory" in column:
                invalid = int(numeric.lt(5).sum() + numeric.gt(40).sum())
            elif "weight" in column:
                invalid = int(numeric.lt(50).sum() + numeric.gt(500).sum())
            elif column == "workout_count_24h":
                invalid = int(numeric.lt(0).sum() + numeric.gt(12).sum())
            elif column == "workout_duration_sum_24h":
                invalid = int(numeric.lt(0).sum() + numeric.gt(24 * 60 * 60).sum())
            elif column == "minutes_since_last_workout":
                invalid = int(numeric.lt(0).sum())
            else:
                invalid = 0
            if invalid > 0:
                impossible_rows.append(f"- {column}: {invalid} impossible-value rows")
    lines = [
        f"# {family.title()} Source Report Card",
        "",
        f"- row_count: {len(frame)}",
        f"- timestamp_start: {'' if timestamp.empty or pd.isna(timestamp.min()) else pd.Timestamp(timestamp.min())}",
        f"- timestamp_end: {'' if timestamp.empty or pd.isna(timestamp.max()) else pd.Timestamp(timestamp.max())}",
        f"- feature_count: {len(present_columns)}",
    ]
    if impossible_rows:
        lines.extend(["", "## Impossible-Value Checks", ""])
        lines.extend(impossible_rows)
    lines.extend(["", "## Feature Missingness", "", "| feature | missing_fraction | non_missing_count |", "| --- | ---: | ---: |"])
    if missingness_rows:
        for row in missingness_rows:
            lines.append(f"| {row['feature']} | {row['missing_fraction']:.3f} | {row['non_missing_count']} |")
    else:
        lines.append("| none | 1.000 | 0 |")
    lines.extend(["", "## Numeric Summary", "", "| feature | mean | median | iqr | min | max |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    if numeric_rows:
        for row in numeric_rows:
            lines.append(
                f"| {row['feature']} | {row['mean']:.3f} | {row['median']:.3f} | {row['iqr']:.3f} | {row['min']:.3f} | {row['max']:.3f} |"
            )
    else:
        lines.append("| none | NA | NA | NA | NA | NA |")
    return pd.DataFrame(numeric_rows), pd.DataFrame(missingness_rows), "\n".join(lines) + "\n"


def build_source_report_cards(dataset: AnalysisReadyHealthDataset) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    tandem_numeric, tandem_missing, tandem_markdown = _summarize_source_report_card(
        dataset.frame,
        family="tandem",
        columns=list(dict.fromkeys(dataset.tandem_feature_columns)),
    )
    apple_numeric, apple_missing, apple_markdown = _summarize_source_report_card(
        dataset.frame,
        family="apple",
        columns=list(dict.fromkeys(dataset.health_feature_columns)),
    )
    source_numeric = pd.concat([tandem_numeric, apple_numeric], ignore_index=True)
    source_missingness = pd.concat([tandem_missing, apple_missing], ignore_index=True)
    return source_numeric, source_missingness, tandem_markdown, apple_markdown


def _parameter_identifiability(
    *,
    parameter: str,
    direct_meal_rows: int,
    proxy_meal_rows: int,
    basal_rows: int,
    correction_rows: int,
    apple_available: bool,
    source_quality_issue_count: int,
) -> tuple[str, str]:
    if parameter == "basal":
        if basal_rows >= 96 and source_quality_issue_count == 0:
            return "directly_observed", "research_enabled"
        if basal_rows >= 48:
            return "weakly_identified", "diagnostics_only"
        return "not_identified", "diagnostics_only"
    if parameter == "I/C ratio":
        if direct_meal_rows >= 48:
            return "directly_observed", "research_enabled"
        if proxy_meal_rows >= 48:
            return "proxy_supported", "diagnostics_only"
        return "not_identified", "diagnostics_only"
    if parameter == "sensitivity factor":
        if correction_rows >= 48 and apple_available:
            return "weakly_identified", "diagnostics_only"
        return "not_identified", "diagnostics_only"
    return "not_identified", "diagnostics_only"


def build_research_gate(
    dataset: AnalysisReadyHealthDataset,
    research_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    timestamp = pd.to_datetime(research_frame.get("timestamp", pd.Series(dtype="datetime64[ns]")), errors="coerce")
    invalid_timestamp_rows = int(timestamp.isna().sum())
    suspicious_timestamp_rows = int(timestamp.lt(pd.Timestamp("2024-01-01")).sum()) if not timestamp.empty else 0
    direct_meal_rows = int(pd.to_numeric(_column_or_default(research_frame, "meal_event"), errors="coerce").fillna(0.0).gt(0).sum())
    proxy_meal_rows = int(_column_or_default(research_frame, "meal_proxy_event").astype(float).gt(0).sum())
    basal_rows = int(_column_or_default(research_frame, "basal_context").astype(float).gt(0).sum())
    correction_rows = int(_column_or_default(research_frame, "correction_context").astype(float).gt(0).sum())
    ambiguous_proxy_rows = int(research_frame.get("bolus_proxy_class", pd.Series(dtype=str)).astype(str).eq("ambiguous").sum())
    source_quality_issue_count = invalid_timestamp_rows + suspicious_timestamp_rows
    if source_quality_issue_count > 0:
        source_quality_status = "degraded"
    elif dataset.frame.empty:
        source_quality_status = "failed"
    else:
        source_quality_status = "good"
    if direct_meal_rows > 0:
        closed_loop_confounding = "moderate"
    elif proxy_meal_rows > 0 or ambiguous_proxy_rows > 0:
        closed_loop_confounding = "high"
    else:
        closed_loop_confounding = "moderate"
    apple_availability = 0.0
    if dataset.health_feature_columns:
        apple_availability = float(dataset.frame.loc[:, dataset.health_feature_columns].notna().mean().mean())
    apple_alignment_status = "credible" if dataset.apple_available and apple_availability >= 0.1 else "limited" if dataset.apple_available else "absent"
    rows: list[dict[str, Any]] = []
    for parameter in ["basal", "I/C ratio", "sensitivity factor"]:
        identifiability, gate_status = _parameter_identifiability(
            parameter=parameter,
            direct_meal_rows=direct_meal_rows,
            proxy_meal_rows=proxy_meal_rows,
            basal_rows=basal_rows,
            correction_rows=correction_rows,
            apple_available=dataset.apple_available,
            source_quality_issue_count=source_quality_issue_count,
        )
        rows.append(
            {
                "parameter": parameter,
                "identifiability": identifiability,
                "gate_status": gate_status,
                "source_quality_status": source_quality_status,
                "direct_meal_rows": direct_meal_rows,
                "proxy_meal_rows": proxy_meal_rows,
                "basal_context_rows": basal_rows,
                "correction_context_rows": correction_rows,
                "closed_loop_confounding_risk": closed_loop_confounding,
                "apple_alignment_status": apple_alignment_status,
            }
        )
    gate = pd.DataFrame(rows)
    lines = [
        "# Therapy Research Gate",
        "",
        f"- source_quality_status: {source_quality_status}",
        f"- invalid_timestamp_rows: {invalid_timestamp_rows}",
        f"- suspicious_timestamp_rows: {suspicious_timestamp_rows}",
        f"- direct_meal_rows: {direct_meal_rows}",
        f"- proxy_meal_rows: {proxy_meal_rows}",
        f"- ambiguous_proxy_rows: {ambiguous_proxy_rows}",
        f"- basal_context_rows: {basal_rows}",
        f"- correction_context_rows: {correction_rows}",
        f"- closed_loop_confounding_risk: {closed_loop_confounding}",
        f"- apple_alignment_status: {apple_alignment_status}",
        "",
        "## Parameter Gate",
        "",
        "| parameter | identifiability | gate_status | source_quality_status | closed_loop_confounding_risk | apple_alignment_status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in gate.itertuples(index=False):
        lines.append(
            f"| {row.parameter} | {row.identifiability} | {row.gate_status} | {row.source_quality_status} | {row.closed_loop_confounding_risk} | {row.apple_alignment_status} |"
        )
    return gate, "\n".join(lines) + "\n"


def build_meal_proxy_audit(research_frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if research_frame.empty or "bolus_proxy_class" not in research_frame.columns:
        audit = pd.DataFrame(columns=["bolus_proxy_class", "count", "mean_confidence", "high_glucose_fraction"])
    else:
        confidence = pd.to_numeric(research_frame.get("meal_proxy_confidence"), errors="coerce").fillna(
            pd.to_numeric(research_frame.get("correction_proxy_confidence"), errors="coerce").fillna(0.0)
        )
        audit = (
            pd.DataFrame(
                {
                    "bolus_proxy_class": research_frame["bolus_proxy_class"].astype(str),
                    "confidence": confidence,
                    "high_glucose": pd.to_numeric(_column_or_default(research_frame, "glucose"), errors="coerce").fillna(0.0).ge(160.0).astype(int),
                }
            )
            .groupby("bolus_proxy_class", as_index=False)
            .agg(
                count=("bolus_proxy_class", "size"),
                mean_confidence=("confidence", "mean"),
                high_glucose_fraction=("high_glucose", "mean"),
            )
        )
    lines = [
        "# Meal Proxy Audit",
        "",
        "- Proxy units remain insulin-based meal proxies and must not be interpreted as observed carb grams.",
        "- Only high-confidence `meal_like` boluses are allowed into proxy meal contexts in strict mode.",
        "",
        "| bolus_proxy_class | count | mean_confidence | high_glucose_fraction |",
        "| --- | ---: | ---: | ---: |",
    ]
    if audit.empty:
        lines.append("| none | 0 | 0.000 | 0.000 |")
    else:
        for row in audit.itertuples(index=False):
            lines.append(
                f"| {row.bolus_proxy_class} | {int(row.count)} | {float(row.mean_confidence):.3f} | {float(row.high_glucose_fraction):.3f} |"
            )
    return audit, "\n".join(lines) + "\n"


def build_therapy_research_frame(
    dataset: AnalysisReadyHealthDataset,
    *,
    segments: tuple[TherapySegment, ...] | None = None,
    meal_proxy_mode: str = "strict",
) -> pd.DataFrame:
    frame = dataset.frame.copy()
    if frame.empty:
        return frame
    segments = segments or parse_therapy_segments()
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame = _assign_segments(frame, segments)

    for column in ["carb_roll_sum_120m", "iob_roll_sum_120m", "iob_roll_sum_60m", "minutes_since_last_meal", "bolus_units", "glucose"]:
        if column not in frame.columns:
            frame[column] = 0.0
    meal_event = pd.to_numeric(_column_or_default(frame, "meal_event"), errors="coerce").fillna(0.0)
    carb_roll_sum_120m = pd.to_numeric(frame["carb_roll_sum_120m"], errors="coerce").fillna(0.0)
    minutes_since_last_meal = pd.to_numeric(frame["minutes_since_last_meal"], errors="coerce")
    bolus_units = pd.to_numeric(frame["bolus_units"], errors="coerce").fillna(0.0)
    glucose = pd.to_numeric(frame["glucose"], errors="coerce").fillna(np.nan)
    iob_roll_sum_120m = pd.to_numeric(frame["iob_roll_sum_120m"], errors="coerce").fillna(0.0)

    meal_signal_available = bool(meal_event.gt(0).any() or carb_roll_sum_120m.gt(0).any())
    direct_meal_signal = (
        meal_event.gt(0)
        | carb_roll_sum_120m.gt(10.0)
        | (minutes_since_last_meal.le(30).fillna(False) if meal_signal_available else pd.Series(False, index=frame.index, dtype=bool))
    )
    frame = _classify_bolus_proxy(frame, direct_meal_signal=direct_meal_signal, mode=meal_proxy_mode)
    if meal_signal_available:
        recent_meal = (
            direct_meal_signal
            | frame["meal_proxy_event"].astype(int).eq(1)
            | minutes_since_last_meal.le(120).fillna(False)
            | carb_roll_sum_120m.gt(10.0)
        )
        fasting_like = (
            minutes_since_last_meal.ge(240).fillna(False)
            & carb_roll_sum_120m.le(10.0)
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
    frame["recent_exercise_context"] = (
        _column_or_default(frame, "recent_workout_12h").astype(float).gt(0)
        | pd.to_numeric(_column_or_default(frame, "health_activity_roll_sum_60m"), errors="coerce").fillna(0.0).gt(0.0)
    ).astype(int)
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
        pd.to_numeric(_column_or_default(frame, "prior_night_total_sleep_hours"), errors="coerce").fillna(0.0).lt(7.0)
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
    frame["recent_therapy_transition"] = _column_or_default(frame, "basal_schedule_change").astype(float).rolling(12, min_periods=1).max().fillna(0.0).gt(0).astype(int)
    frame["closed_loop_confounding_flag"] = (
        frame["high_iob_state"].eq(1)
        | frame["stacked_bolus_state"].eq(1)
        | frame["recent_therapy_transition"].eq(1)
    ).astype(int)
    frame["basal_context"] = (
        frame["basal_context"].astype(int).eq(1)
        & frame["closed_loop_confounding_flag"].eq(0)
    ).astype(int)
    return frame


def _source_family(column: str, dataset: AnalysisReadyHealthDataset) -> str:
    if column.startswith("meal_proxy_") or column.startswith("correction_proxy_") or column == "bolus_proxy_class":
        return "meal_bolus_proxy"
    if column.startswith("segment__") or column.endswith("_context") or column in {"sleep_deficit_flag", "post_workout_meal_context", "overnight_hrv_interaction", "in_sleep_heart_rate_interaction"}:
        return "therapy_context"
    if column in dataset.health_feature_columns:
        if "sleep" in column:
            return "apple_sleep"
        if "workout" in column:
            return "apple_workout"
        if any(token in column for token in ["heart_rate", "hrv", "respiratory", "weight"]):
            return "apple_measurements"
        return "apple_activity"
    if column.startswith("glucose") or column.startswith("cgm") or column in {"glucose", "missing_cgm", "minutes_since_last_cgm"}:
        return "tandem_cgm"
    if "basal" in column:
        return "tandem_basal"
    if any(token in column for token in ["bolus", "iob", "insulin"]):
        return "tandem_bolus"
    if "carb" in column or "meal" in column:
        return "tandem_carbs"
    if "activity" in column:
        return "activity"
    if any(token in column for token in ["hour_", "dow_", "weekend"]):
        return "calendar"
    return "derived"


def _alignment_rule(column: str) -> str:
    if column.startswith("meal_proxy_") or column.startswith("correction_proxy_"):
        return "event_proxy"
    if column == "bolus_proxy_class":
        return "proxy_label"
    if column.startswith("glucose_lag_"):
        return "lagged_observation"
    if "_roll_" in column:
        return "rolling_summary"
    if column.startswith("prior_night_"):
        return "projected_next_day"
    if column == "in_sleep":
        return "interval_overlay"
    if column.endswith("_latest") or column in {"weight_latest", "hrv_latest", "respiratory_rate_latest", "resting_heart_rate_latest"}:
        return "carry_forward"
    if column.startswith("segment__") or column.endswith("_context") or column.endswith("_flag"):
        return "context_mask"
    if column.endswith("_interaction"):
        return "interaction_term"
    return "current_bin_or_direct_state"


def _allowed_use_cases(column: str) -> str:
    if column.startswith("meal_proxy_") or column.startswith("correction_proxy_") or column == "bolus_proxy_class":
        return "icr,sensitivity_factor"
    if column.startswith("segment__") or column.endswith("_context"):
        return "basal,icr,sensitivity_factor"
    if "meal" in column or "carb" in column:
        return "icr"
    if "basal" in column:
        return "basal"
    if "correction" in column or "glucose" in column:
        return "basal,icr,sensitivity_factor"
    if any(token in column for token in ["sleep", "workout", "heart_rate", "hrv", "respiratory"]):
        return "basal,icr"
    return "basal,icr,sensitivity_factor"


def _leakage_risk(column: str) -> str:
    if column.startswith("meal_proxy_") or column.startswith("correction_proxy_") or column == "bolus_proxy_class":
        return "moderate"
    if column in {TARGET_GLUCOSE_COLUMN, TARGET_DELTA_COLUMN}:
        return "high"
    if column.endswith("_latest") and "weight" not in column:
        return "low"
    if column.startswith("prior_night_") or column == "in_sleep":
        return "low"
    return "low"


def build_therapy_feature_registry(
    dataset: AnalysisReadyHealthDataset,
    research_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in research_frame.columns:
        if column in {TARGET_GLUCOSE_COLUMN, TARGET_DELTA_COLUMN, "timestamp"}:
            continue
        series = research_frame[column]
        numeric = pd.to_numeric(series, errors="coerce") if series.dtype != object else pd.Series(dtype=float)
        rows.append(
            {
                "feature": column,
                "source_family": _source_family(column, dataset),
                "alignment_rule": _alignment_rule(column),
                "allowed_use_cases": _allowed_use_cases(column),
                "leakage_risk": _leakage_risk(column),
                "availability": float(series.notna().mean()),
                "nonzero_fraction": float((numeric.fillna(0.0) != 0.0).mean()) if not numeric.empty else float(series.astype(str).ne("").mean()),
                "intended_role": _source_family(column, dataset).replace("_", " "),
            }
        )
    return pd.DataFrame(rows).sort_values(["source_family", "feature"]).reset_index(drop=True)


def render_therapy_feature_audit(
    dataset: AnalysisReadyHealthDataset,
    research_frame: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    segments: tuple[TherapySegment, ...],
) -> str:
    lines = [
        "# Therapy Feature Audit",
        "",
        f"- prepared_mode: {dataset.mode}",
        f"- apple_available: {dataset.apple_available}",
        f"- row_count: {len(research_frame)}",
        f"- feature_count: {len(registry)}",
        f"- segment_schedule: {', '.join(segment.name for segment in segments)}",
        "",
        "## Context Masks",
        "",
        f"- basal_context_rows: {int(research_frame.get('basal_context', pd.Series(dtype=int)).sum())}",
        f"- meal_context_rows: {int(research_frame.get('meal_context', pd.Series(dtype=int)).sum())}",
        f"- correction_context_rows: {int(research_frame.get('correction_context', pd.Series(dtype=int)).sum())}",
        f"- meal_proxy_rows: {int(research_frame.get('meal_proxy_event', pd.Series(dtype=int)).sum())}",
        f"- correction_proxy_rows: {int(research_frame.get('correction_proxy_event', pd.Series(dtype=int)).sum())}",
        f"- therapy_stable_epochs: {int(pd.to_numeric(_column_or_default(research_frame, 'therapy_stable_epoch'), errors='coerce').nunique()) if not research_frame.empty else 0}",
        "",
        "## Alignment Findings",
        "",
        "- Sleep summaries are projected onto the following day and `in_sleep` is overlaid on active intervals.",
        "- Sparse Apple measurement features use latest-known carry-forward alignment in the current implementation.",
        "- Workout features are anchored on workout end time with recent-workout summaries and time-since-workout proxies.",
        "- Research mode adds explicit therapy-context masks, meal-bolus proxy features, and selected interaction terms so health variables can be tested in physiologically relevant contexts.",
        "",
        "## Potential Risks",
        "",
        "- Carry-forward sparse physiology features can become stale in long gaps and should be interpreted cautiously.",
        "- Current workout and sleep engineering is deterministic and interpretable, but still coarse relative to richer recovery-state modeling.",
        "- Bolus-based meal proxies are inferential and remain vulnerable to correction/automation confounding.",
        "- This workflow is associational and predictive decision support, not causal identification of setting effects.",
        "",
        "## Feature Families",
        "",
        "| source_family | feature_count | mean_availability | mean_nonzero_fraction |",
        "| --- | ---: | ---: | ---: |",
    ]
    if registry.empty:
        lines.append("| none | 0 | 0.000 | 0.000 |")
    else:
        grouped = (
            registry.groupby("source_family", as_index=False)
            .agg(
                feature_count=("feature", "size"),
                mean_availability=("availability", "mean"),
                mean_nonzero_fraction=("nonzero_fraction", "mean"),
            )
        )
        for row in grouped.itertuples(index=False):
            lines.append(
                f"| {row.source_family} | {int(row.feature_count)} | {float(row.mean_availability):.3f} | {float(row.mean_nonzero_fraction):.3f} |"
            )
    lines.extend(["", "## Registry Preview", "", "| feature | source_family | alignment_rule | allowed_use_cases | leakage_risk | availability |", "| --- | --- | --- | --- | --- | ---: |"])
    preview = registry.head(15)
    for row in preview.itertuples(index=False):
        lines.append(
            f"| {row.feature} | {row.source_family} | {row.alignment_rule} | {row.allowed_use_cases} | {row.leakage_risk} | {float(row.availability):.3f} |"
        )
    return "\n".join(lines) + "\n"


def _as_feature_matrix(frame: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    if not feature_names:
        return np.zeros((len(frame), 0), dtype=float)
    return frame.loc[:, feature_names].astype(float).to_numpy(dtype=float)


def _soft_threshold(value: float, penalty: float) -> float:
    if value > penalty:
        return value - penalty
    if value < -penalty:
        return value + penalty
    return 0.0


def _fit_ridge(train: pd.DataFrame, feature_names: list[str], *, target_col: str, alpha: float = 1.0) -> _ModelFitBundle:
    X = _as_feature_matrix(train, feature_names)
    y = train[target_col].to_numpy(dtype=float)
    means = X.mean(axis=0) if X.shape[1] else np.zeros(0, dtype=float)
    scales = X.std(axis=0, ddof=0) if X.shape[1] else np.zeros(0, dtype=float)
    if X.shape[1]:
        scales = np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)
        Xs = (X - means) / scales
        y_mean = float(np.mean(y))
        centered = y - y_mean
        gram = Xs.T @ Xs + alpha * np.eye(Xs.shape[1], dtype=float)
        coef = np.linalg.solve(gram, Xs.T @ centered)
    else:
        Xs = X
        y_mean = float(np.mean(y)) if len(y) else 0.0
        coef = np.zeros(0, dtype=float)
    train_pred = y_mean + (Xs @ coef if Xs.shape[1] else np.zeros(len(train), dtype=float))
    residual = y - train_pred
    lower_q = float(np.quantile(residual, 0.1)) if len(residual) else 0.0
    upper_q = float(np.quantile(residual, 0.9)) if len(residual) else 0.0
    importance = {name: float(abs(value)) for name, value in zip(feature_names, coef, strict=False)}

    def predict(frame: pd.DataFrame) -> _ModelPrediction:
        Xt = _as_feature_matrix(frame, feature_names)
        if Xt.shape[1]:
            Xt = (Xt - means) / scales
            mean = y_mean + Xt @ coef
        else:
            mean = np.repeat(y_mean, len(frame))
        return _ModelPrediction(mean=mean, lower=mean + lower_q, upper=mean + upper_q)

    return _ModelFitBundle(feature_names=feature_names, prediction=predict, feature_importance=importance)


def _fit_elastic_net(
    train: pd.DataFrame,
    feature_names: list[str],
    *,
    target_col: str,
    alpha: float = 0.05,
    l1_ratio: float = 0.5,
    max_iter: int = 300,
    tol: float = 1e-5,
) -> _ModelFitBundle:
    X = _as_feature_matrix(train, feature_names)
    y = train[target_col].to_numpy(dtype=float)
    means = X.mean(axis=0) if X.shape[1] else np.zeros(0, dtype=float)
    scales = X.std(axis=0, ddof=0) if X.shape[1] else np.zeros(0, dtype=float)
    if X.shape[1]:
        scales = np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)
        Xs = (X - means) / scales
        y_mean = float(np.mean(y))
        centered = y - y_mean
        coef = np.zeros(Xs.shape[1], dtype=float)
        residual = centered.copy()
        denom = np.mean(Xs * Xs, axis=0) + alpha * (1.0 - l1_ratio)
        denom = np.where(np.isfinite(denom) & (denom > 0), denom, 1.0)
        for _ in range(max_iter):
            max_change = 0.0
            for index in range(Xs.shape[1]):
                partial = residual + Xs[:, index] * coef[index]
                rho = float(np.mean(Xs[:, index] * partial))
                updated = _soft_threshold(rho, alpha * l1_ratio) / denom[index]
                change = updated - coef[index]
                if change != 0.0:
                    residual = partial - Xs[:, index] * updated
                    coef[index] = updated
                    max_change = max(max_change, abs(change))
            if max_change < tol:
                break
    else:
        Xs = X
        y_mean = float(np.mean(y)) if len(y) else 0.0
        coef = np.zeros(0, dtype=float)
    train_pred = y_mean + (Xs @ coef if Xs.shape[1] else np.zeros(len(train), dtype=float))
    residual = y - train_pred
    lower_q = float(np.quantile(residual, 0.1)) if len(residual) else 0.0
    upper_q = float(np.quantile(residual, 0.9)) if len(residual) else 0.0
    importance = {name: float(abs(value)) for name, value in zip(feature_names, coef, strict=False)}

    def predict(frame: pd.DataFrame) -> _ModelPrediction:
        Xt = _as_feature_matrix(frame, feature_names)
        if Xt.shape[1]:
            Xt = (Xt - means) / scales
            mean = y_mean + Xt @ coef
        else:
            mean = np.repeat(y_mean, len(frame))
        return _ModelPrediction(mean=mean, lower=mean + lower_q, upper=mean + upper_q)

    return _ModelFitBundle(feature_names=feature_names, prediction=predict, feature_importance=importance)


def _fit_segmented_ridge(
    train: pd.DataFrame,
    feature_names: list[str],
    *,
    target_col: str,
    group_col: str = "therapy_segment",
    alpha: float = 1.0,
) -> _ModelFitBundle:
    global_fit = _fit_ridge(train, feature_names, target_col=target_col, alpha=alpha)
    fits: dict[str, _ModelFitBundle] = {}
    importance_by_segment: dict[str, dict[str, float]] = {}
    for segment, frame in train.groupby(group_col):
        if len(frame) < max(len(feature_names) + 5, 20):
            continue
        seg_fit = _fit_ridge(frame, feature_names, target_col=target_col, alpha=alpha)
        fits[str(segment)] = seg_fit
        importance_by_segment[str(segment)] = seg_fit.feature_importance or {}

    def predict(frame: pd.DataFrame) -> _ModelPrediction:
        means = np.zeros(len(frame), dtype=float)
        lowers = np.zeros(len(frame), dtype=float)
        uppers = np.zeros(len(frame), dtype=float)
        for idx, row in frame.iterrows():
            bundle = fits.get(str(row[group_col]), global_fit)
            pred = bundle.prediction(frame.loc[[idx]])
            position = int(frame.index.get_loc(idx))
            means[position] = float(pred.mean[0])
            lowers[position] = float(pred.lower[0])
            uppers[position] = float(pred.upper[0])
        return _ModelPrediction(mean=means, lower=lowers, upper=uppers)

    return _ModelFitBundle(
        feature_names=feature_names,
        prediction=predict,
        feature_importance=global_fit.feature_importance,
        feature_importance_by_segment=importance_by_segment,
    )


def _fit_tree_boost(
    train: pd.DataFrame,
    feature_names: list[str],
    *,
    target_col: str,
    n_estimators: int = 20,
    learning_rate: float = 0.1,
) -> _ModelFitBundle:
    X = _as_feature_matrix(train, feature_names)
    y = train[target_col].to_numpy(dtype=float)
    baseline = float(np.mean(y)) if len(y) else 0.0
    pred = np.repeat(baseline, len(y))
    stumps: list[dict[str, Any]] = []
    importance: dict[str, float] = {name: 0.0 for name in feature_names}
    if X.shape[1]:
        for _ in range(n_estimators):
            residual = y - pred
            best_feature = -1
            best_threshold = 0.0
            best_left = 0.0
            best_right = 0.0
            best_loss = math.inf
            for feature_index, name in enumerate(feature_names):
                values = X[:, feature_index]
                unique = np.unique(values)
                if len(unique) <= 2:
                    candidates = unique.tolist()
                else:
                    quantiles = np.linspace(0.1, 0.9, num=min(9, len(unique) - 1))
                    candidates = np.unique(np.quantile(values, quantiles)).tolist()
                for threshold in candidates:
                    left_mask = values <= threshold
                    right_mask = ~left_mask
                    if not np.any(left_mask) or not np.any(right_mask):
                        continue
                    left_value = float(np.mean(residual[left_mask]))
                    right_value = float(np.mean(residual[right_mask]))
                    updated = pred + learning_rate * np.where(left_mask, left_value, right_value)
                    loss = float(np.mean((y - updated) ** 2))
                    if loss < best_loss:
                        best_feature = feature_index
                        best_threshold = float(threshold)
                        best_left = left_value
                        best_right = right_value
                        best_loss = loss
            if best_feature < 0:
                break
            values = X[:, best_feature]
            increment = learning_rate * np.where(values <= best_threshold, best_left, best_right)
            pred = pred + increment
            importance[feature_names[best_feature]] += float(np.var(increment))
            stumps.append(
                {
                    "feature_index": best_feature,
                    "threshold": best_threshold,
                    "left": best_left,
                    "right": best_right,
                }
            )
    residual = y - pred
    lower_q = float(np.quantile(residual, 0.1)) if len(residual) else 0.0
    upper_q = float(np.quantile(residual, 0.9)) if len(residual) else 0.0

    def predict(frame: pd.DataFrame) -> _ModelPrediction:
        Xt = _as_feature_matrix(frame, feature_names)
        mean = np.repeat(baseline, len(frame))
        for stump in stumps:
            values = Xt[:, stump["feature_index"]] if Xt.shape[1] else np.zeros(len(frame))
            mean = mean + learning_rate * np.where(values <= stump["threshold"], stump["left"], stump["right"])
        return _ModelPrediction(mean=mean, lower=mean + lower_q, upper=mean + upper_q)

    return _ModelFitBundle(feature_names=feature_names, prediction=predict, feature_importance=importance)


def _fit_bayesian(
    train: pd.DataFrame,
    feature_names: list[str],
    *,
    target_col: str,
    horizon_minutes: int,
    draws: int = 150,
    tune: int = 150,
    chains: int = 2,
) -> _ModelFitBundle:
    frame = FeatureFrame(
        frame=train.loc[:, ["timestamp", *feature_names, target_col]].copy(),
        feature_columns=feature_names,
        target_column=target_col,
        horizon_minutes=horizon_minutes,
        config=FeatureConfig(horizon_minutes=horizon_minutes),
    )
    model = BayesianGlucoseModel(draws=draws, tune=tune, chains=chains)
    fit = model.fit(frame)
    importance: dict[str, float] | None = None
    try:
        beta = fit.posterior.posterior["beta"].stack(sample=("chain", "draw")).values
        if beta.ndim == 2:
            importance = {
                name: float(np.mean(np.abs(beta[index])))
                for index, name in enumerate(feature_names)
            }
    except Exception:
        importance = None

    def predict(frame: pd.DataFrame) -> _ModelPrediction:
        preds = model.predict(fit, frame.loc[:, ["timestamp", *feature_names, target_col]])
        return _ModelPrediction(
            mean=preds["mean"].to_numpy(dtype=float),
            lower=preds["lower"].to_numpy(dtype=float),
            upper=preds["upper"].to_numpy(dtype=float),
        )

    return _ModelFitBundle(feature_names=feature_names, prediction=predict, diagnostics=fit.diagnostics, feature_importance=importance)


def _fit_ensemble(
    train: pd.DataFrame,
    feature_names: list[str],
    *,
    target_col: str,
) -> _ModelFitBundle:
    components = [
        _fit_ridge(train, feature_names, target_col=target_col, alpha=1.0),
        _fit_segmented_ridge(train, feature_names, target_col=target_col, alpha=1.0),
        _fit_tree_boost(train, feature_names, target_col=target_col, n_estimators=12, learning_rate=0.1),
    ]

    def predict(frame: pd.DataFrame) -> _ModelPrediction:
        preds = [component.prediction(frame) for component in components]
        mean = np.mean([pred.mean for pred in preds], axis=0)
        lower = np.mean([pred.lower for pred in preds], axis=0)
        upper = np.mean([pred.upper for pred in preds], axis=0)
        return _ModelPrediction(mean=mean, lower=lower, upper=upper)

    importance: dict[str, float] = {}
    for component in components:
        for name, value in (component.feature_importance or {}).items():
            importance[name] = importance.get(name, 0.0) + float(value)
    return _ModelFitBundle(feature_names=feature_names, prediction=predict, feature_importance=importance)


def _fit_model_family(
    model_name: str,
    train: pd.DataFrame,
    feature_names: list[str],
    *,
    target_col: str,
    horizon_minutes: int,
) -> _ModelFitBundle:
    if model_name == "ridge":
        return _fit_ridge(train, feature_names, target_col=target_col, alpha=1.0)
    if model_name == "elastic_net":
        return _fit_elastic_net(train, feature_names, target_col=target_col, alpha=0.05, l1_ratio=0.5)
    if model_name == "segmented_ridge":
        return _fit_segmented_ridge(train, feature_names, target_col=target_col, alpha=1.0)
    if model_name == "tree_boost":
        return _fit_tree_boost(train, feature_names, target_col=target_col, n_estimators=18, learning_rate=0.1)
    if model_name == "ensemble":
        return _fit_ensemble(train, feature_names, target_col=target_col)
    if model_name == "bayesian":
        return _fit_bayesian(train, feature_names, target_col=target_col, horizon_minutes=horizon_minutes)
    raise ValueError(f"Unsupported model family: {model_name}")


def _prediction_interval_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    if len(actual) == 0:
        return float("nan")
    return float(np.mean((actual >= lower) & (actual <= upper)))


def _safe_rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    if len(actual) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def _safe_mae(actual: np.ndarray, pred: np.ndarray) -> float:
    if len(actual) == 0:
        return float("nan")
    return float(np.mean(np.abs(actual - pred)))


def _safety_weighted_mae(actual: np.ndarray, pred: np.ndarray) -> float:
    if len(actual) == 0:
        return float("nan")
    weights = np.where(actual < 70.0, 3.0, np.where(actual < 90.0, 2.0, 1.0))
    return float(np.average(np.abs(actual - pred), weights=weights))


def _low_glucose_mae(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = actual < 80.0
    if not np.any(mask):
        return float(np.mean(np.abs(actual - pred))) if len(actual) else float("nan")
    return float(np.mean(np.abs(actual[mask] - pred[mask])))


def _score_loss(glucose: np.ndarray) -> np.ndarray:
    return (
        np.abs(glucose - 110.0)
        + np.clip(80.0 - glucose, a_min=0.0, a_max=None) * 2.0
        + np.clip(70.0 - glucose, a_min=0.0, a_max=None) * 4.0
        + np.clip(glucose - 180.0, a_min=0.0, a_max=None) * 0.5
    )


def _task_feature_names(frame: pd.DataFrame) -> list[str]:
    names: list[str] = []
    for column in frame.columns:
        if column in RESEARCH_FEATURE_EXCLUDE:
            continue
        if column == "glucose":
            names.append(column)
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        names.append(column)
    return [name for name in dict.fromkeys(names) if name not in {TARGET_GLUCOSE_COLUMN, TARGET_DELTA_COLUMN}]


def _build_task_frame(research_frame: pd.DataFrame, task: TherapyTask) -> pd.DataFrame:
    if research_frame.empty or task.context_column not in research_frame.columns:
        return pd.DataFrame()
    out = research_frame.loc[research_frame[task.context_column].fillna(0).astype(int).eq(1)].copy()
    return out.sort_values("timestamp").reset_index(drop=True)


def _evaluation_splits(frame: pd.DataFrame) -> list[Any]:
    n_rows = len(frame)
    if n_rows < 40:
        return []
    initial_train_size = max(int(n_rows * 0.5), 24)
    test_size = max(int(n_rows * 0.2), 12)
    if initial_train_size + test_size > n_rows:
        initial_train_size = max(int(n_rows * 0.6), 12)
        test_size = max(n_rows - initial_train_size, 8)
    return list(walk_forward_splits(n_rows, initial_train_size=initial_train_size, test_size=test_size))


def _top_features(importance: dict[str, float] | None, *, limit: int = 5) -> str:
    if not importance:
        return ""
    items = sorted(importance.items(), key=lambda item: (-float(item[1]), item[0]))
    return ", ".join(f"{name}:{value:.3f}" for name, value in items[:limit])


def run_model_comparison(
    research_frame: pd.DataFrame,
    *,
    include_models: tuple[str, ...],
    horizon_minutes: int,
) -> tuple[pd.DataFrame, dict[str, str]]:
    rows: list[dict[str, Any]] = []
    notes: dict[str, str] = {}
    for task in TASKS:
        task_frame = _build_task_frame(research_frame, task)
        feature_names = _task_feature_names(task_frame)
        if task_frame.empty or len(feature_names) == 0:
            for model_name in include_models:
                rows.append(
                    {
                        "task": task.name,
                        "parameter": task.parameter,
                        "model_name": model_name,
                        "status": "insufficient_data",
                        "n_rows": int(len(task_frame)),
                        "n_features": int(len(feature_names)),
                    }
                )
            continue
        splits = _evaluation_splits(task_frame)
        for model_name in include_models:
            if task.staged and task.name == "sensitivity_factor":
                rows.append(
                    {
                        "task": task.name,
                        "parameter": task.parameter,
                        "model_name": model_name,
                        "status": "staged",
                        "n_rows": int(len(task_frame)),
                        "n_features": int(len(feature_names)),
                    }
                )
                continue
            if not splits:
                rows.append(
                    {
                        "task": task.name,
                        "parameter": task.parameter,
                        "model_name": model_name,
                        "status": "insufficient_data",
                        "n_rows": int(len(task_frame)),
                        "n_features": int(len(feature_names)),
                    }
                )
                continue
            fold_rows: list[dict[str, Any]] = []
            fit_example: _ModelFitBundle | None = None
            failed_note: str | None = None
            for split in splits:
                train = task_frame.iloc[split.train_start:split.train_end].copy()
                test = task_frame.iloc[split.test_start:split.test_end].copy()
                if len(train) < max(len(feature_names) + 5, 20) or len(test) < 8:
                    continue
                try:
                    fit_bundle = _fit_model_family(
                        model_name,
                        train,
                        feature_names,
                        target_col=TARGET_DELTA_COLUMN,
                        horizon_minutes=horizon_minutes,
                    )
                except Exception as exc:
                    failed_note = str(exc)
                    break
                fit_example = fit_bundle
                prediction = fit_bundle.prediction(test)
                actual_delta = test[TARGET_DELTA_COLUMN].to_numpy(dtype=float)
                actual_future = test[TARGET_GLUCOSE_COLUMN].to_numpy(dtype=float)
                pred_future = test["glucose"].to_numpy(dtype=float) + prediction.mean
                pred_lower = test["glucose"].to_numpy(dtype=float) + prediction.lower
                pred_upper = test["glucose"].to_numpy(dtype=float) + prediction.upper
                fold_rows.append(
                    {
                        "mae": _safe_mae(actual_future, pred_future),
                        "rmse": _safe_rmse(actual_future, pred_future),
                        "coverage": _prediction_interval_coverage(actual_future, pred_lower, pred_upper),
                        "interval_width": float(np.mean(pred_upper - pred_lower)) if len(pred_upper) else float("nan"),
                        "low_bg_mae": _low_glucose_mae(actual_future, pred_future),
                        "safety_weighted_mae": _safety_weighted_mae(actual_future, pred_future),
                        "n_test": int(len(test)),
                        "fit_diagnostics": fit_bundle.diagnostics,
                        "actual_delta_mean": float(np.mean(actual_delta)) if len(actual_delta) else float("nan"),
                    }
                )
            if failed_note is not None:
                rows.append(
                    {
                        "task": task.name,
                        "parameter": task.parameter,
                        "model_name": model_name,
                        "status": "failed",
                        "n_rows": int(len(task_frame)),
                        "n_features": int(len(feature_names)),
                        "note": failed_note,
                    }
                )
                notes[f"{task.name}:{model_name}"] = failed_note
                continue
            if not fold_rows:
                rows.append(
                    {
                        "task": task.name,
                        "parameter": task.parameter,
                        "model_name": model_name,
                        "status": "insufficient_data",
                        "n_rows": int(len(task_frame)),
                        "n_features": int(len(feature_names)),
                    }
                )
                continue
            fold_frame = pd.DataFrame(fold_rows)
            rows.append(
                {
                    "task": task.name,
                    "parameter": task.parameter,
                    "model_name": model_name,
                    "status": "completed",
                    "n_rows": int(len(task_frame)),
                    "n_features": int(len(feature_names)),
                    "fold_count": int(len(fold_frame)),
                    "mae": float(fold_frame["mae"].mean()),
                    "rmse": float(fold_frame["rmse"].mean()),
                    "coverage": float(fold_frame["coverage"].mean()),
                    "interval_width": float(fold_frame["interval_width"].mean()),
                    "low_bg_mae": float(fold_frame["low_bg_mae"].mean()),
                    "safety_weighted_mae": float(fold_frame["safety_weighted_mae"].mean()),
                    "fold_mae_std": float(fold_frame["mae"].std(ddof=0)) if len(fold_frame) > 1 else 0.0,
                    "fold_safety_std": float(fold_frame["safety_weighted_mae"].std(ddof=0)) if len(fold_frame) > 1 else 0.0,
                    "top_features": _top_features(fit_example.feature_importance if fit_example is not None else None),
                    "diagnostics_status": "available" if fit_example is not None and fit_example.diagnostics is not None else "not_available",
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result, notes
    for task_name in sorted(result["task"].dropna().unique()):
        task_mask = result["task"].eq(task_name) & result["status"].eq("completed")
        if not task_mask.any():
            continue
        completed = result.loc[task_mask].copy()
        completed["rank_key"] = list(
            zip(
                completed["safety_weighted_mae"].round(6),
                completed["low_bg_mae"].round(6),
                completed["fold_safety_std"].round(6),
                completed["mae"].round(6),
            )
        )
        ordered = completed.sort_values(["safety_weighted_mae", "low_bg_mae", "fold_safety_std", "mae", "model_name"])
        best_single = ordered.loc[ordered["model_name"] != "ensemble"].head(1)
        if not best_single.empty and (ordered["model_name"] == "ensemble").any():
            ensemble_index = ordered.loc[ordered["model_name"] == "ensemble"].index[0]
            best_row = best_single.iloc[0]
            ensemble_row = ordered.loc[ensemble_index]
            if not (
                float(ensemble_row["safety_weighted_mae"]) < float(best_row["safety_weighted_mae"]) * 0.995
                and float(ensemble_row["fold_safety_std"]) <= float(best_row["fold_safety_std"])
            ):
                result.loc[ensemble_index, "status"] = "rejected"
                result.loc[ensemble_index, "note"] = "ensemble did not materially improve both safety-weighted error and stability"
        completed_mask = result["task"].eq(task_name) & result["status"].eq("completed")
        if completed_mask.any():
            best_index = result.loc[completed_mask].sort_values(
                ["safety_weighted_mae", "low_bg_mae", "fold_safety_std", "mae", "model_name"]
            ).index[0]
            result.loc[result["task"].eq(task_name), "selected"] = False
            result.loc[best_index, "selected"] = True
    return result, notes


def _selected_model_name(model_comparison: pd.DataFrame, task_name: str) -> str | None:
    if model_comparison.empty:
        return None
    selected = model_comparison.loc[
        model_comparison["task"].eq(task_name)
        & model_comparison["status"].eq("completed")
        & model_comparison.get("selected", pd.Series(dtype=bool)).fillna(False)
    ]
    if selected.empty:
        return None
    return str(selected.iloc[0]["model_name"])


def _apply_segment_scenario(frame: pd.DataFrame, *, task: TherapyTask, segment: str, change_percent: float) -> pd.DataFrame:
    out = frame.copy()
    mask = out["therapy_segment"].astype(str).eq(segment)
    multiplier = 1.0 + change_percent / 100.0
    if task.name == "basal":
        for column in ["basal_units_per_hour", "basal_units_delivered"]:
            if column in out.columns:
                out.loc[mask, column] = pd.to_numeric(out.loc[mask, column], errors="coerce").fillna(0.0) * multiplier
        return out
    if task.name == "icr":
        if "bolus_units" in out.columns:
            out.loc[mask, "bolus_units"] = pd.to_numeric(out.loc[mask, "bolus_units"], errors="coerce").fillna(0.0) * multiplier
        return recompute_scenario_features(out)
    return out


def build_segment_evidence_and_recommendations(
    research_frame: pd.DataFrame,
    research_gate: pd.DataFrame,
    model_comparison: pd.DataFrame,
    *,
    include_models: tuple[str, ...],
    segments: tuple[TherapySegment, ...],
    horizon_minutes: int,
    ic_policy: str = "exploratory_only",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evidence_rows: list[dict[str, Any]] = []
    recommendation_rows: list[dict[str, Any]] = []
    for task in TASKS:
        gate_row = research_gate.loc[research_gate["parameter"].eq(task.parameter)].head(1)
        gate_status = str(gate_row.iloc[0]["gate_status"]) if not gate_row.empty else "research_enabled"
        identifiability = str(gate_row.iloc[0]["identifiability"]) if not gate_row.empty else "weakly_identified"
        task_frame = _build_task_frame(research_frame, task)
        feature_names = _task_feature_names(task_frame)
        model_name = _selected_model_name(model_comparison, task.name)
        task_model_row = model_comparison.loc[
            model_comparison["task"].eq(task.name) & model_comparison["model_name"].eq(model_name)
        ].head(1)
        if task.staged:
            for segment in segments:
                recommendation_rows.append(
                    {
                        "parameter": task.parameter,
                        "segment": segment.name,
                        "status": "suppressed",
                        "proposed_change_percent": np.nan,
                        "expected_direction": "",
                        "mean_expected_gain": np.nan,
                        "fold_better_fraction": np.nan,
                        "confidence": "staged",
                        "reasons_for": "",
                        "reasons_against": "staged_parameter",
                        "identifiability": identifiability,
                    }
                )
            continue
        if task_frame.empty or not feature_names or model_name is None or task_model_row.empty:
            for segment in segments:
                recommendation_rows.append(
                    {
                        "parameter": task.parameter,
                        "segment": segment.name,
                        "status": "suppressed",
                        "proposed_change_percent": np.nan,
                        "expected_direction": "",
                        "mean_expected_gain": np.nan,
                        "fold_better_fraction": np.nan,
                        "confidence": "low",
                        "reasons_for": "",
                        "reasons_against": "insufficient_data_or_model",
                        "identifiability": identifiability,
                    }
                )
            continue
        splits = _evaluation_splits(task_frame)
        if not splits:
            continue
        scenario_changes = (-10.0, -5.0, 5.0, 10.0)
        scenario_rows: list[dict[str, Any]] = []
        for split in splits:
            train = task_frame.iloc[split.train_start:split.train_end].copy()
            test = task_frame.iloc[split.test_start:split.test_end].copy()
            if len(train) < max(len(feature_names) + 5, 20) or len(test) < 8:
                continue
            fit_bundle = _fit_model_family(
                model_name,
                train,
                feature_names,
                target_col=TARGET_DELTA_COLUMN,
                horizon_minutes=horizon_minutes,
            )
            base_pred = fit_bundle.prediction(test).mean + test["glucose"].to_numpy(dtype=float)
            base_loss = _score_loss(base_pred)
            for segment in segments:
                segment_mask = test["therapy_segment"].astype(str).eq(segment.name)
                if int(segment_mask.sum()) < 3:
                    continue
                segment_frame = test.loc[segment_mask].copy()
                importance_map = fit_bundle.feature_importance_by_segment or {}
                top_features = _top_features(
                    importance_map.get(segment.name) or fit_bundle.feature_importance,
                    limit=3,
                )
                evidence_rows.append(
                    {
                        "parameter": task.parameter,
                        "task": task.name,
                        "segment": segment.name,
                        "split_rows": int(len(segment_frame)),
                        "selected_model": model_name,
                        "gate_status": gate_status,
                        "identifiability": identifiability,
                        "top_features": top_features,
                        "actual_target_mean": float(segment_frame[TARGET_GLUCOSE_COLUMN].mean()),
                        "actual_target_delta_mean": float(segment_frame[TARGET_DELTA_COLUMN].mean()),
                        "low_target_fraction": float(segment_frame[TARGET_GLUCOSE_COLUMN].lt(80).mean()),
                    }
                )
                for change_percent in scenario_changes:
                    scenario_test = _apply_segment_scenario(test, task=task, segment=segment.name, change_percent=change_percent)
                    scenario_pred = fit_bundle.prediction(scenario_test).mean + scenario_test["glucose"].to_numpy(dtype=float)
                    scenario_loss = _score_loss(scenario_pred)
                    gain = float(np.mean(base_loss[segment_mask] - scenario_loss[segment_mask]))
                    scenario_rows.append(
                        {
                            "parameter": task.parameter,
                            "task": task.name,
                            "segment": segment.name,
                            "change_percent": change_percent,
                            "gain": gain,
                            "fold_id": f"{split.train_end}:{split.test_end}",
                            "segment_rows": int(segment_mask.sum()),
                        }
                    )
        scenario_frame = pd.DataFrame(scenario_rows)
        for segment in segments:
            segment_scenarios = scenario_frame.loc[scenario_frame["segment"].eq(segment.name)] if not scenario_frame.empty else pd.DataFrame()
            if segment_scenarios.empty:
                recommendation_rows.append(
                    {
                        "parameter": task.parameter,
                        "segment": segment.name,
                        "status": "suppressed",
                        "proposed_change_percent": np.nan,
                        "expected_direction": "",
                        "mean_expected_gain": np.nan,
                        "fold_better_fraction": np.nan,
                        "confidence": "low",
                        "reasons_for": "",
                        "reasons_against": "insufficient_segment_evidence",
                        "identifiability": identifiability,
                    }
                )
                continue
            grouped = (
                segment_scenarios.groupby("change_percent", as_index=False)
                .agg(
                    mean_expected_gain=("gain", "mean"),
                    fold_better_fraction=("gain", lambda s: float(np.mean(pd.Series(s).gt(0)))),
                    gain_std=("gain", "std"),
                    fold_count=("gain", "size"),
                )
                .sort_values(["mean_expected_gain", "fold_better_fraction", "change_percent"], ascending=[False, False, True])
            )
            best = grouped.iloc[0]
            reasons_against: list[str] = []
            status = "candidate"
            if float(task_model_row["safety_weighted_mae"].iloc[0]) > 25.0:
                reasons_against.append("model_safety_error_too_high")
            if float(task_model_row["coverage"].iloc[0]) < 0.65:
                reasons_against.append("coverage_too_low")
            if gate_status != "research_enabled":
                reasons_against.append(f"gate_{gate_status}")
            if int(best["fold_count"]) < 2:
                reasons_against.append("too_few_folds")
            if float(best["mean_expected_gain"]) < 1.0:
                reasons_against.append("expected_gain_too_small")
            if float(best["fold_better_fraction"]) < 0.6:
                reasons_against.append("inconsistent_fold_direction")
            if reasons_against:
                status = "suppressed"
            if task.name == "icr" and ic_policy == "exploratory_only":
                status = "exploratory" if status == "candidate" else status
                reasons_against.append("exploratory_only_policy")
            confidence = "high" if status == "candidate" and float(best["fold_better_fraction"]) >= 0.8 and float(best["mean_expected_gain"]) >= 5.0 else "moderate" if status == "candidate" else "low"
            recommendation_rows.append(
                {
                    "parameter": task.parameter,
                    "segment": segment.name,
                    "status": status,
                    "proposed_change_percent": float(best["change_percent"]) if status == "candidate" else np.nan,
                    "expected_direction": "increase" if float(best["change_percent"]) > 0 else "decrease" if float(best["change_percent"]) < 0 else "hold",
                    "mean_expected_gain": float(best["mean_expected_gain"]),
                    "fold_better_fraction": float(best["fold_better_fraction"]),
                    "confidence": confidence,
                    "reasons_for": "predicted safety-weighted loss improved on held-out folds",
                    "reasons_against": ",".join(reasons_against),
                    "identifiability": identifiability,
                }
            )
    return pd.DataFrame(evidence_rows), pd.DataFrame(recommendation_rows)


def render_model_comparison_markdown(model_comparison: pd.DataFrame, notes: dict[str, str]) -> str:
    lines = [
        "# Therapy Model Comparison",
        "",
        "- Research outputs are predictive and associational, not causal.",
        "- Lower safety-weighted MAE and lower low-glucose MAE are prioritized over raw RMSE.",
        "",
    ]
    if model_comparison.empty:
        lines.append("No model comparison results were available.")
        return "\n".join(lines) + "\n"
    for task in TASKS:
        lines.extend([f"## {TASK_PARAMETER_LABELS[task.name].title()} Task", "", "| model | status | rows | features | mae | rmse | safety_weighted_mae | low_bg_mae | coverage | selected |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
        subset = model_comparison.loc[model_comparison["task"].eq(task.name)].copy()
        if subset.empty:
            lines.append("| none | missing | 0 | 0 | NA | NA | NA | NA | NA | no |")
            lines.append("")
            continue
        for row in subset.itertuples(index=False):
            lines.append(
                "| {model} | {status} | {rows} | {features} | {mae} | {rmse} | {safety} | {low_bg} | {coverage} | {selected} |".format(
                    model=row.model_name,
                    status=row.status,
                    rows=int(getattr(row, "n_rows", 0) or 0),
                    features=int(getattr(row, "n_features", 0) or 0),
                    mae="NA" if pd.isna(getattr(row, "mae", np.nan)) else f"{float(row.mae):.3f}",
                    rmse="NA" if pd.isna(getattr(row, "rmse", np.nan)) else f"{float(row.rmse):.3f}",
                    safety="NA" if pd.isna(getattr(row, "safety_weighted_mae", np.nan)) else f"{float(row.safety_weighted_mae):.3f}",
                    low_bg="NA" if pd.isna(getattr(row, "low_bg_mae", np.nan)) else f"{float(row.low_bg_mae):.3f}",
                    coverage="NA" if pd.isna(getattr(row, "coverage", np.nan)) else f"{float(row.coverage):.3f}",
                    selected="yes" if bool(getattr(row, "selected", False)) else "no",
                )
            )
        lines.append("")
    if notes:
        lines.extend(["## Notes", ""])
        for key, value in sorted(notes.items()):
            lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def render_recommendation_markdown(recommendations: pd.DataFrame, segment_evidence: pd.DataFrame) -> str:
    lines = [
        "# Therapy Recommendation Research",
        "",
        "- These recommendations are decision support candidates generated from predictive and associational evidence.",
        "- They are not causal estimates and they are not automatic pump-setting changes.",
        "",
        "## Segment Recommendations",
        "",
    ]
    if recommendations.empty:
        lines.append("No segment-level recommendations were available.")
    else:
        lines.extend(["| parameter | segment | status | identifiability | proposed_change_percent | mean_expected_gain | fold_better_fraction | confidence | reasons_against |", "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |"])
        for row in recommendations.itertuples(index=False):
            lines.append(
                "| {parameter} | {segment} | {status} | {identifiability} | {change} | {gain} | {fraction} | {confidence} | {reasons} |".format(
                    parameter=row.parameter,
                    segment=row.segment,
                    status=row.status,
                    identifiability=getattr(row, "identifiability", ""),
                    change="NA" if pd.isna(row.proposed_change_percent) else f"{float(row.proposed_change_percent):.1f}",
                    gain="NA" if pd.isna(row.mean_expected_gain) else f"{float(row.mean_expected_gain):.3f}",
                    fraction="NA" if pd.isna(row.fold_better_fraction) else f"{float(row.fold_better_fraction):.3f}",
                    confidence=row.confidence,
                    reasons=row.reasons_against or "",
                )
            )
    if not segment_evidence.empty:
        lines.extend(["", "## Segment Evidence Snapshot", "", "| parameter | segment | selected_model | split_rows | top_features | low_target_fraction |", "| --- | --- | --- | ---: | --- | ---: |"])
        summary = (
            segment_evidence.groupby(["parameter", "segment", "selected_model"], as_index=False)
            .agg(
                split_rows=("split_rows", "sum"),
                top_features=("top_features", "first"),
                low_target_fraction=("low_target_fraction", "mean"),
            )
        )
        for row in summary.itertuples(index=False):
            lines.append(
                f"| {row.parameter} | {row.segment} | {row.selected_model} | {int(row.split_rows)} | {row.top_features or ''} | {float(row.low_target_fraction):.3f} |"
            )
    return "\n".join(lines) + "\n"


def run_therapy_research(
    dataset: AnalysisReadyHealthDataset,
    *,
    segments: tuple[TherapySegment, ...] | None = None,
    include_models: tuple[str, ...] | None = None,
    meal_proxy_mode: str = "strict",
    ic_policy: str = "exploratory_only",
) -> TherapyResearchResult:
    segments = segments or parse_therapy_segments()
    include_models = include_models or parse_model_list(None)
    research_frame = build_therapy_research_frame(dataset, segments=segments, meal_proxy_mode=meal_proxy_mode)
    research_gate, research_gate_markdown = build_research_gate(dataset, research_frame)
    feature_registry = build_therapy_feature_registry(dataset, research_frame)
    meal_proxy_audit, meal_proxy_audit_markdown = build_meal_proxy_audit(research_frame)
    source_numeric_summary, source_missingness_summary, tandem_source_report_markdown, apple_source_report_markdown = build_source_report_cards(dataset)
    feature_audit_markdown = render_therapy_feature_audit(dataset, research_frame, feature_registry, segments=segments)
    model_comparison, notes = run_model_comparison(
        research_frame,
        include_models=include_models,
        horizon_minutes=dataset.horizon_minutes,
    )
    segment_evidence, recommendations = build_segment_evidence_and_recommendations(
        research_frame,
        research_gate,
        model_comparison,
        include_models=include_models,
        segments=segments,
        horizon_minutes=dataset.horizon_minutes,
        ic_policy=ic_policy,
    )
    model_comparison_markdown = render_model_comparison_markdown(model_comparison, notes)
    recommendation_markdown = render_recommendation_markdown(recommendations, segment_evidence)
    return TherapyResearchResult(
        prepared_dataset=dataset,
        research_frame=research_frame,
        research_gate=research_gate,
        feature_registry=feature_registry,
        meal_proxy_audit=meal_proxy_audit,
        model_comparison=model_comparison,
        segment_evidence=segment_evidence,
        recommendations=recommendations,
        research_gate_markdown=research_gate_markdown,
        feature_audit_markdown=feature_audit_markdown,
        meal_proxy_audit_markdown=meal_proxy_audit_markdown,
        model_comparison_markdown=model_comparison_markdown,
        recommendation_markdown=recommendation_markdown,
        tandem_source_report_markdown=tandem_source_report_markdown,
        apple_source_report_markdown=apple_source_report_markdown,
        source_numeric_summary=source_numeric_summary,
        source_missingness_summary=source_missingness_summary,
        segments=segments,
        include_models=include_models,
        meal_proxy_mode=meal_proxy_mode,
        ic_policy=ic_policy,
    )


def write_therapy_research_artifacts(
    result: TherapyResearchResult,
    report_dir: str | Path,
    *,
    write_source_report_cards: bool = True,
    write_research_gate: bool = True,
) -> dict[str, Path]:
    root = Path(report_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "research_gate": root / "therapy_research_gate.md",
        "feature_audit": root / "therapy_feature_audit.md",
        "feature_registry": root / "therapy_feature_registry.csv",
        "meal_proxy_audit": root / "meal_proxy_audit.md",
        "model_comparison": root / "therapy_model_comparison.md",
        "segment_evidence": root / "therapy_segment_evidence.csv",
        "recommendation_research": root / "therapy_recommendation_research.md",
        "tandem_source_report": root / "tandem_source_report_card.md",
        "apple_source_report": root / "apple_source_report_card.md",
        "source_numeric_summary": root / "source_numeric_summary.csv",
        "source_missingness_summary": root / "source_missingness_summary.csv",
    }
    if write_research_gate:
        paths["research_gate"].write_text(result.research_gate_markdown, encoding="utf-8")
    paths["feature_audit"].write_text(result.feature_audit_markdown, encoding="utf-8")
    result.feature_registry.to_csv(paths["feature_registry"], index=False)
    paths["meal_proxy_audit"].write_text(result.meal_proxy_audit_markdown, encoding="utf-8")
    paths["model_comparison"].write_text(result.model_comparison_markdown, encoding="utf-8")
    result.segment_evidence.to_csv(paths["segment_evidence"], index=False)
    paths["recommendation_research"].write_text(result.recommendation_markdown, encoding="utf-8")
    if write_source_report_cards:
        paths["tandem_source_report"].write_text(result.tandem_source_report_markdown, encoding="utf-8")
        paths["apple_source_report"].write_text(result.apple_source_report_markdown, encoding="utf-8")
        result.source_numeric_summary.to_csv(paths["source_numeric_summary"], index=False)
        result.source_missingness_summary.to_csv(paths["source_missingness_summary"], index=False)
    return paths


def _fit_ridge_regression(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    alpha: float = 4.0,
) -> dict[str, Any] | None:
    if frame.empty or target_column not in frame.columns:
        return None
    X = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y)
    if not np.any(valid):
        return None
    X = X[valid]
    y = y[valid]
    if X.shape[0] < 3:
        return None
    means = X.mean(axis=0)
    scales = X.std(axis=0, ddof=0)
    scales = np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)
    standardized = (X - means) / scales
    y_mean = float(np.mean(y))
    centered = y - y_mean
    gram = standardized.T @ standardized + alpha * np.eye(standardized.shape[1], dtype=float)
    coef = np.linalg.solve(gram, standardized.T @ centered)
    fitted = standardized @ coef + y_mean
    residual_rmse = float(np.sqrt(np.mean((fitted - y) ** 2))) if len(y) else float("nan")
    return {
        "feature_columns": list(feature_columns),
        "coef": coef,
        "means": means,
        "scales": scales,
        "intercept": y_mean,
        "residual_rmse": residual_rmse,
        "training_rows": int(len(y)),
    }


def _predict_ridge_regression(model: dict[str, Any] | None, frame: pd.DataFrame) -> np.ndarray:
    if model is None or frame.empty:
        return np.array([], dtype=float)
    X = frame.loc[:, model["feature_columns"]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    standardized = (X - model["means"]) / model["scales"]
    return standardized @ model["coef"] + float(model["intercept"])


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
        pd.to_numeric(_column_or_default(ordered, "meal_event"), errors="coerce").fillna(0.0).gt(0)
        | pd.to_numeric(_column_or_default(ordered, "carb_grams"), errors="coerce").fillna(0.0).gt(0)
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
            pd.to_numeric(_column_or_default(cluster, "meal_event"), errors="coerce").fillna(0.0).gt(0)
            | pd.to_numeric(_column_or_default(cluster, "carb_grams"), errors="coerce").fillna(0.0).gt(0)
        ).astype(int)
        cluster["proxy_signal"] = pd.to_numeric(_column_or_default(cluster, "meal_proxy_event"), errors="coerce").fillna(0.0).gt(0).astype(int)
        cluster["ranking_stated_carbs"] = pd.to_numeric(_column_or_default(cluster, "carb_grams"), errors="coerce").fillna(0.0)
        cluster["ranking_bolus"] = pd.to_numeric(_column_or_default(cluster, "bolus_units"), errors="coerce").fillna(0.0)
        cluster["ranking_proxy_confidence"] = pd.to_numeric(_column_or_default(cluster, "meal_proxy_confidence"), errors="coerce").fillna(0.0)
        selected = cluster.sort_values(
            ["explicit_signal", "ranking_stated_carbs", "ranking_proxy_confidence", "ranking_bolus", "timestamp"],
            ascending=[False, False, False, False, True],
        ).iloc[0]
        explicit_rows = int(cluster["explicit_signal"].sum())
        proxy_rows = int(cluster["proxy_signal"].sum())
        evidence_source = "explicit_and_proxy" if explicit_rows and proxy_rows else "explicit_logged" if explicit_rows else "proxy_only"
        stated_carbs = pd.to_numeric(pd.Series([selected.get("carb_grams")]), errors="coerce").iloc[0]
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


def build_meal_window_dataset(
    research_frame: pd.DataFrame,
    meal_event_registry: pd.DataFrame,
    *,
    premeal_minutes: int = 30,
    postmeal_minutes: int = 180,
) -> pd.DataFrame:
    columns = [
        "meal_id",
        "timestamp",
        "segment",
        "evidence_source",
        "stated_carbs",
        "bolus_units_window",
        "iob_at_start",
        "premeal_glucose",
        "peak_glucose_180m",
        "peak_delta_180m",
        "positive_auc_180m",
        "window_complete",
        "confounding_grade",
        "source_quality_grade",
        "recent_exercise_context",
        "sleep_deficit_flag",
        "hrv_latest_window",
        "heart_rate_avg_latest_window",
        "hrv_minutes_since_last",
        "resting_heart_rate_minutes_since_last",
        "missing_cgm_fraction",
        "suppression_reason",
    ]
    if research_frame.empty or meal_event_registry.empty:
        return pd.DataFrame(columns=columns)

    ordered = research_frame.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="coerce")
    ordered = ordered.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    step_minutes = 5.0
    if len(ordered) > 1:
        deltas = ordered["timestamp"].diff().dt.total_seconds().div(60.0).dropna()
        if not deltas.empty:
            step_minutes = float(max(deltas.median(), 1.0))

    rows: list[dict[str, Any]] = []
    for meal in meal_event_registry.itertuples(index=False):
        timestamp = pd.Timestamp(meal.timestamp)
        pre = ordered.loc[
            ordered["timestamp"].ge(timestamp - pd.Timedelta(minutes=premeal_minutes))
            & ordered["timestamp"].lt(timestamp)
        ].copy()
        post = ordered.loc[
            ordered["timestamp"].ge(timestamp)
            & ordered["timestamp"].le(timestamp + pd.Timedelta(minutes=postmeal_minutes))
        ].copy()
        if post.empty:
            continue
        meal_row = ordered.loc[ordered["timestamp"].eq(timestamp)].head(1)
        meal_row = post.head(1) if meal_row.empty else meal_row
        def _meal_row_value(column: str, default: float) -> float:
            if column not in meal_row.columns:
                return float(default)
            return float(pd.to_numeric(meal_row[column], errors="coerce").fillna(default).iloc[0])
        premeal_glucose = pd.to_numeric(pre.get("glucose"), errors="coerce").median()
        if pd.isna(premeal_glucose):
            premeal_glucose = _meal_row_value("glucose", float("nan"))
        post_glucose = pd.to_numeric(post.get("glucose"), errors="coerce")
        peak_glucose = float(post_glucose.max()) if post_glucose.notna().any() else float("nan")
        peak_delta = float(max(peak_glucose - premeal_glucose, 0.0)) if np.isfinite(premeal_glucose) and np.isfinite(peak_glucose) else float("nan")
        positive_auc = float(np.maximum(post_glucose.fillna(premeal_glucose) - premeal_glucose, 0.0).sum() * step_minutes / 60.0)
        bolus_units_window = float(
            pd.to_numeric(
                ordered.loc[
                    ordered["timestamp"].ge(timestamp - pd.Timedelta(minutes=15))
                    & ordered["timestamp"].le(timestamp + pd.Timedelta(minutes=30)),
                    "bolus_units",
                ],
                errors="coerce",
            ).fillna(0.0).sum()
        )
        iob_at_start = _meal_row_value("iob_units", 0.0)
        missing_cgm_fraction = float(pd.to_numeric(post.get("missing_cgm"), errors="coerce").fillna(0.0).mean())
        workout_plausible = bool(_meal_row_value("workout_summary_plausible", 1.0) > 0)
        hrv_freshness = _meal_row_value("hrv_minutes_since_last", 1e6)
        resting_hr_freshness = _meal_row_value("resting_heart_rate_minutes_since_last", 1e6)
        closed_loop_flag = bool(pd.to_numeric(post.get("closed_loop_confounding_flag"), errors="coerce").fillna(0.0).max() > 0)
        recent_exercise = bool(_meal_row_value("recent_exercise_context", 0.0) > 0)
        if closed_loop_flag or iob_at_start > 3.0:
            confounding_grade = "high"
        elif recent_exercise or bolus_units_window > 10.0:
            confounding_grade = "moderate"
        else:
            confounding_grade = "low"
        if missing_cgm_fraction > 0 or not workout_plausible:
            source_quality_grade = "degraded"
        elif hrv_freshness > 24 * 60 and resting_hr_freshness > 7 * 24 * 60:
            source_quality_grade = "limited"
        else:
            source_quality_grade = "good"
        window_complete = bool(len(pre) >= max(int(premeal_minutes / step_minutes / 2), 3) and len(post) >= max(int(postmeal_minutes / step_minutes * 0.7), 12))
        suppression_reason = ""
        if bolus_units_window < 0.5:
            suppression_reason = "no_meaningful_bolus"
        elif not window_complete:
            suppression_reason = "incomplete_window"
        elif confounding_grade == "high":
            suppression_reason = "high_closed_loop_confounding"
        elif source_quality_grade == "degraded":
            suppression_reason = "source_quality_degraded"
        rows.append(
            {
                "meal_id": meal.meal_id,
                "timestamp": timestamp,
                "segment": meal.segment,
                "evidence_source": meal.evidence_source,
                "stated_carbs": meal.stated_carbs,
                "bolus_units_window": bolus_units_window,
                "iob_at_start": iob_at_start,
                "premeal_glucose": float(premeal_glucose) if np.isfinite(premeal_glucose) else float("nan"),
                "peak_glucose_180m": peak_glucose,
                "peak_delta_180m": peak_delta,
                "positive_auc_180m": positive_auc,
                "window_complete": int(window_complete),
                "confounding_grade": confounding_grade,
                "source_quality_grade": source_quality_grade,
                "recent_exercise_context": int(recent_exercise),
                "sleep_deficit_flag": int(_meal_row_value("sleep_deficit_flag", 0.0)),
                "hrv_latest_window": _meal_row_value("hrv_latest", 0.0) if hrv_freshness <= 24 * 60 else 0.0,
                "heart_rate_avg_latest_window": _meal_row_value("heart_rate_avg_latest", 0.0),
                "hrv_minutes_since_last": hrv_freshness,
                "resting_heart_rate_minutes_since_last": resting_hr_freshness,
                "missing_cgm_fraction": missing_cgm_fraction,
                "suppression_reason": suppression_reason,
            }
        )
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def _latent_meal_gate(meal_windows: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    logged_meals = int(pd.to_numeric(meal_windows.get("stated_carbs"), errors="coerce").notna().sum()) if not meal_windows.empty else 0
    proxy_only_meals = int(meal_windows.get("evidence_source", pd.Series(dtype=str)).astype(str).eq("proxy_only").sum()) if not meal_windows.empty else 0
    complete_windows = int(pd.to_numeric(meal_windows.get("window_complete"), errors="coerce").fillna(0.0).sum()) if not meal_windows.empty else 0
    trainable_windows = int(
        (
            pd.to_numeric(meal_windows.get("window_complete"), errors="coerce").fillna(0.0).gt(0)
            & pd.to_numeric(meal_windows.get("bolus_units_window"), errors="coerce").fillna(0.0).ge(0.5)
            & meal_windows.get("confounding_grade", pd.Series(dtype=str)).astype(str).ne("high")
        ).sum()
    ) if not meal_windows.empty else 0
    degraded_fraction = float(meal_windows.get("source_quality_grade", pd.Series(dtype=str)).astype(str).eq("degraded").mean()) if not meal_windows.empty else 1.0
    if degraded_fraction > 0.2:
        source_quality_status = "degraded"
    elif meal_windows.empty:
        source_quality_status = "failed"
    else:
        source_quality_status = "good"
    high_confounding_fraction = float(meal_windows.get("confounding_grade", pd.Series(dtype=str)).astype(str).eq("high").mean()) if not meal_windows.empty else 1.0
    if high_confounding_fraction >= 0.5:
        confounding_risk = "high"
    elif high_confounding_fraction >= 0.2:
        confounding_risk = "moderate"
    else:
        confounding_risk = "low"
    if trainable_windows >= 12 and logged_meals >= 12:
        identifiability = "prior_informed"
        gate_status = "research_only"
    elif trainable_windows >= 18 and proxy_only_meals >= 12:
        identifiability = "response_inferred"
        gate_status = "diagnostics_only"
    elif complete_windows >= 8:
        identifiability = "weakly_identified"
        gate_status = "diagnostics_only"
    else:
        identifiability = "not_identified"
        gate_status = "diagnostics_only"
    gate = pd.DataFrame(
        [
            {
                "parameter": "I/C ratio",
                "estimand": "latent meal carbs + effective I/C",
                "identifiability": identifiability,
                "gate_status": gate_status,
                "source_quality_status": source_quality_status,
                "closed_loop_confounding_risk": confounding_risk,
                "meal_windows": int(len(meal_windows)),
                "logged_meals": logged_meals,
                "proxy_only_meals": proxy_only_meals,
                "complete_windows": complete_windows,
                "trainable_windows": trainable_windows,
            }
        ]
    )
    lines = [
        "# Latent Meal Research Gate",
        "",
        f"- source_quality_status: {source_quality_status}",
        f"- closed_loop_confounding_risk: {confounding_risk}",
        f"- meal_windows: {int(len(meal_windows))}",
        f"- logged_meals: {logged_meals}",
        f"- proxy_only_meals: {proxy_only_meals}",
        f"- complete_windows: {complete_windows}",
        f"- trainable_windows: {trainable_windows}",
        "",
        "## Parameter Gate",
        "",
        "| parameter | estimand | identifiability | gate_status | source_quality_status | closed_loop_confounding_risk |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    row = gate.iloc[0]
    lines.append(
        f"| {row['parameter']} | {row['estimand']} | {row['identifiability']} | {row['gate_status']} | {row['source_quality_status']} | {row['closed_loop_confounding_risk']} |"
    )
    return gate, "\n".join(lines) + "\n"


def _response_feature_frame(meal_windows: pd.DataFrame) -> pd.DataFrame:
    out = meal_windows.copy()
    out["confounding_is_low"] = out.get("confounding_grade", pd.Series(dtype=str)).astype(str).eq("low").astype(float)
    out["source_quality_is_good"] = out.get("source_quality_grade", pd.Series(dtype=str)).astype(str).eq("good").astype(float)
    return out


def _render_latent_meal_window_audit(meal_windows: pd.DataFrame) -> str:
    lines = [
        "# Meal Window Audit",
        "",
        f"- meal_window_count: {len(meal_windows)}",
        f"- complete_windows: {int(pd.to_numeric(meal_windows.get('window_complete'), errors='coerce').fillna(0.0).sum()) if not meal_windows.empty else 0}",
        "",
        "| confounding_grade | count | mean_peak_delta_180m | mean_positive_auc_180m |",
        "| --- | ---: | ---: | ---: |",
    ]
    if meal_windows.empty:
        lines.append("| none | 0 | 0.000 | 0.000 |")
    else:
        grouped = (
            meal_windows.groupby("confounding_grade", as_index=False)
            .agg(
                count=("meal_id", "size"),
                mean_peak_delta_180m=("peak_delta_180m", "mean"),
                mean_positive_auc_180m=("positive_auc_180m", "mean"),
            )
            .sort_values("confounding_grade")
        )
        for row in grouped.itertuples(index=False):
            lines.append(
                f"| {row.confounding_grade} | {int(row.count)} | {float(row.mean_peak_delta_180m):.3f} | {float(row.mean_positive_auc_180m):.3f} |"
            )
    return "\n".join(lines) + "\n"


def _render_latent_meal_fit_summary(
    *,
    gate: pd.DataFrame,
    reference_ic: float,
    carb_model: dict[str, Any] | None,
    selected_model: str,
    selected_peak_delta_mae: float | None,
) -> str:
    training_rows = 0 if carb_model is None else int(carb_model.get("training_rows", 0))
    residual_rmse = None if carb_model is None else carb_model.get("residual_rmse")
    lines = [
        "# Latent Meal Fit Summary",
        "",
        "- This workflow is a latent meal deconvolution approximation intended for research and diagnostics.",
        "- Logged carbs are treated as noisy priors when present, not exact truth.",
        f"- selected_model: {selected_model}",
        f"- reference_ic_ratio: {reference_ic:.3f}",
        f"- training_meals: {training_rows}",
        f"- carb_model_residual_rmse: {'NA' if residual_rmse is None or pd.isna(residual_rmse) else f'{float(residual_rmse):.3f}'}",
        f"- selected_peak_delta_mae: {'NA' if selected_peak_delta_mae is None or pd.isna(selected_peak_delta_mae) else f'{float(selected_peak_delta_mae):.3f}'}",
        "",
        "## Gate Snapshot",
        "",
        "| parameter | identifiability | gate_status | closed_loop_confounding_risk |",
        "| --- | --- | --- | --- |",
    ]
    if gate.empty:
        lines.append("| I/C ratio | not_identified | diagnostics_only | high |")
    else:
        row = gate.iloc[0]
        lines.append(
            f"| {row['parameter']} | {row['identifiability']} | {row['gate_status']} | {row['closed_loop_confounding_risk']} |"
        )
    return "\n".join(lines) + "\n"


def _render_latent_meal_confidence_report(posterior_meals: pd.DataFrame) -> str:
    lines = [
        "# Latent Meal Confidence Report",
        "",
        "| evidence_source | count | mean_accuracy_score | high_confidence_fraction |",
        "| --- | ---: | ---: | ---: |",
    ]
    if posterior_meals.empty:
        lines.append("| none | 0 | 0.000 | 0.000 |")
        return "\n".join(lines) + "\n"
    grouped = (
        posterior_meals.groupby("evidence_source", as_index=False)
        .agg(
            count=("meal_id", "size"),
            mean_accuracy_score=("carb_accuracy_score", "mean"),
            high_confidence_fraction=("carb_accuracy_score", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0.0).ge(0.75).mean())),
        )
        .sort_values("evidence_source")
    )
    for row in grouped.itertuples(index=False):
        lines.append(
            f"| {row.evidence_source} | {int(row.count)} | {float(row.mean_accuracy_score):.3f} | {float(row.high_confidence_fraction):.3f} |"
        )
    return "\n".join(lines) + "\n"


def _render_latent_meal_model_comparison(model_comparison: pd.DataFrame) -> str:
    lines = [
        "# Latent Meal Model Comparison",
        "",
        "- Retrospective comparison is anchored on logged meals and post-meal peak-delta explanation.",
        "",
        "| model | meal_count | stated_carb_mae | peak_delta_mae | peak_delta_correlation | selected |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    if model_comparison.empty:
        lines.append("| none | 0 | NA | NA | NA | no |")
        return "\n".join(lines) + "\n"
    for row in model_comparison.itertuples(index=False):
        lines.append(
            "| {model} | {meal_count} | {carb_mae} | {peak_mae} | {corr} | {selected} |".format(
                model=row.model,
                meal_count=int(row.meal_count),
                carb_mae="NA" if pd.isna(row.stated_carb_mae) else f"{float(row.stated_carb_mae):.3f}",
                peak_mae="NA" if pd.isna(row.peak_delta_mae) else f"{float(row.peak_delta_mae):.3f}",
                corr="NA" if pd.isna(row.peak_delta_correlation) else f"{float(row.peak_delta_correlation):.3f}",
                selected="yes" if bool(row.selected) else "no",
            )
        )
    return "\n".join(lines) + "\n"


def run_latent_meal_icr_research(
    dataset: AnalysisReadyHealthDataset,
    *,
    segments: tuple[TherapySegment, ...] | None = None,
    meal_proxy_mode: str = "strict",
) -> LatentMealResearchResult:
    segments = segments or parse_therapy_segments()
    research_frame = build_therapy_research_frame(dataset, segments=segments, meal_proxy_mode=meal_proxy_mode)
    meal_event_registry = build_meal_event_registry(research_frame)
    meal_windows = build_meal_window_dataset(research_frame, meal_event_registry)
    gate, gate_markdown = _latent_meal_gate(meal_windows)

    feature_frame = _response_feature_frame(meal_windows)
    training_mask = (
        pd.to_numeric(feature_frame.get("stated_carbs"), errors="coerce").notna()
        & pd.to_numeric(feature_frame.get("window_complete"), errors="coerce").fillna(0.0).gt(0)
        & pd.to_numeric(feature_frame.get("bolus_units_window"), errors="coerce").fillna(0.0).ge(0.5)
        & feature_frame.get("confounding_grade", pd.Series(dtype=str)).astype(str).ne("high")
    )
    training = feature_frame.loc[training_mask].copy()
    ic_series = (
        pd.to_numeric(training.get("stated_carbs"), errors="coerce")
        / pd.to_numeric(training.get("bolus_units_window"), errors="coerce").replace(0.0, np.nan)
    )
    reference_ic = float(np.nanmedian(ic_series.to_numpy(dtype=float))) if np.isfinite(np.nanmedian(ic_series.to_numpy(dtype=float))) else 12.0
    if not np.isfinite(reference_ic):
        reference_ic = 12.0
    reference_ic = float(np.clip(reference_ic, 4.0, 40.0))

    carb_model_features = [
        "bolus_units_window",
        "iob_at_start",
        "premeal_glucose",
        "peak_delta_180m",
        "positive_auc_180m",
        "recent_exercise_context",
        "sleep_deficit_flag",
        "hrv_latest_window",
        "heart_rate_avg_latest_window",
        "confounding_is_low",
        "source_quality_is_good",
    ]
    carb_model = _fit_ridge_regression(
        training,
        feature_columns=carb_model_features,
        target_column="stated_carbs",
        alpha=4.0,
    )
    response_model = _fit_ridge_regression(
        training.assign(candidate_carbs=pd.to_numeric(training.get("stated_carbs"), errors="coerce").fillna(0.0)),
        feature_columns=["candidate_carbs", "bolus_units_window", "premeal_glucose"],
        target_column="peak_delta_180m",
        alpha=2.0,
    )

    windows = feature_frame.copy()
    windows["bolus_reference_carbs"] = np.clip(
        pd.to_numeric(windows.get("bolus_units_window"), errors="coerce").fillna(0.0) * reference_ic,
        0.0,
        250.0,
    )
    stated_carbs = pd.to_numeric(windows.get("stated_carbs"), errors="coerce")
    windows["stated_prior_blend_carbs"] = stated_carbs.fillna(windows["bolus_reference_carbs"])
    response_adjusted_raw = _predict_ridge_regression(carb_model, windows.loc[:, carb_model_features]) if carb_model is not None else np.array([], dtype=float)
    if len(response_adjusted_raw) != len(windows):
        response_adjusted_raw = windows["bolus_reference_carbs"].to_numpy(dtype=float)
    raw_response_series = pd.Series(np.clip(response_adjusted_raw, 0.0, 250.0), index=windows.index, dtype=float)
    prior_weight = np.where(
        stated_carbs.notna(),
        np.where(
            windows.get("confounding_grade", pd.Series(dtype=str)).astype(str).eq("low"),
            0.65,
            np.where(windows.get("confounding_grade", pd.Series(dtype=str)).astype(str).eq("moderate"), 0.45, 0.25),
        ),
        0.0,
    )
    windows["latent_response_adjusted_carbs"] = np.clip(
        np.where(stated_carbs.notna(), prior_weight * stated_carbs.fillna(0.0) + (1.0 - prior_weight) * raw_response_series, raw_response_series),
        0.0,
        250.0,
    )

    eval_logged = windows.loc[stated_carbs.notna()].copy()
    comparison_rows: list[dict[str, Any]] = []
    selected_model = "latent_response_adjusted"
    selected_peak_delta_mae = None
    for model_name, carb_column in [
        ("bolus_reference", "bolus_reference_carbs"),
        ("stated_prior_blend", "stated_prior_blend_carbs"),
        ("latent_response_adjusted", "latent_response_adjusted_carbs"),
    ]:
        peak_delta_predictions = np.full(len(eval_logged), np.nan, dtype=float)
        if response_model is not None and not eval_logged.empty:
            prediction_input = eval_logged.assign(candidate_carbs=pd.to_numeric(eval_logged[carb_column], errors="coerce").fillna(0.0))
            peak_delta_predictions = _predict_ridge_regression(response_model, prediction_input)
        actual_peak_delta = pd.to_numeric(eval_logged.get("peak_delta_180m"), errors="coerce")
        carb_predictions = pd.to_numeric(eval_logged.get(carb_column), errors="coerce")
        peak_delta_mae = float(np.nanmean(np.abs(peak_delta_predictions - actual_peak_delta.to_numpy(dtype=float)))) if len(eval_logged) else float("nan")
        if len(eval_logged) > 1 and np.isfinite(actual_peak_delta).sum() > 1 and np.isfinite(peak_delta_predictions).sum() > 1:
            peak_delta_correlation = float(np.corrcoef(actual_peak_delta.to_numpy(dtype=float), peak_delta_predictions)[0, 1])
        else:
            peak_delta_correlation = float("nan")
        stated_carb_mae = float(np.nanmean(np.abs(carb_predictions.to_numpy(dtype=float) - stated_carbs.loc[eval_logged.index].to_numpy(dtype=float)))) if len(eval_logged) else float("nan")
        comparison_rows.append(
            {
                "model": model_name,
                "meal_count": int(len(eval_logged)),
                "stated_carb_mae": stated_carb_mae,
                "peak_delta_mae": peak_delta_mae,
                "peak_delta_correlation": peak_delta_correlation,
                "selected": False,
            }
        )
    model_comparison = pd.DataFrame(comparison_rows)
    if not model_comparison.empty:
        model_comparison = model_comparison.sort_values(["peak_delta_mae", "stated_carb_mae"], na_position="last").reset_index(drop=True)
        model_comparison.loc[0, "selected"] = True
        selected_model = str(model_comparison.iloc[0]["model"])
        selected_peak_delta_mae = None if pd.isna(model_comparison.iloc[0]["peak_delta_mae"]) else float(model_comparison.iloc[0]["peak_delta_mae"])

    selected_carb_column = {
        "bolus_reference": "bolus_reference_carbs",
        "stated_prior_blend": "stated_prior_blend_carbs",
        "latent_response_adjusted": "latent_response_adjusted_carbs",
    }.get(selected_model, "latent_response_adjusted_carbs")
    residual_scale = float(carb_model.get("residual_rmse", 18.0)) if carb_model is not None else 18.0
    residual_scale = float(max(residual_scale, 12.0))

    posterior_rows: list[dict[str, Any]] = []
    for row in windows.itertuples(index=False):
        latent_carbs = float(pd.to_numeric(pd.Series([getattr(row, selected_carb_column)]), errors="coerce").fillna(0.0).iloc[0])
        stated_value = pd.to_numeric(pd.Series([getattr(row, "stated_carbs")]), errors="coerce").iloc[0]
        raw_discrepancy = abs(latent_carbs - float(stated_value)) if pd.notna(stated_value) else residual_scale
        quality_multiplier = 1.0 if getattr(row, "source_quality_grade") == "good" else 0.75 if getattr(row, "source_quality_grade") == "limited" else 0.45
        confounding_multiplier = 1.0 if getattr(row, "confounding_grade") == "low" else 0.7 if getattr(row, "confounding_grade") == "moderate" else 0.35
        if pd.notna(stated_value):
            carb_accuracy_score = float(np.clip(np.exp(-raw_discrepancy / residual_scale) * quality_multiplier * confounding_multiplier, 0.0, 1.0))
        else:
            carb_accuracy_score = float(np.clip(0.45 * quality_multiplier * confounding_multiplier, 0.0, 0.5))
        latent_lower = max(latent_carbs - 1.64 * residual_scale, 0.0)
        latent_upper = latent_carbs + 1.64 * residual_scale
        bolus_units = float(pd.to_numeric(pd.Series([getattr(row, "bolus_units_window")]), errors="coerce").fillna(0.0).iloc[0])
        if bolus_units >= 0.5:
            ic_mean = latent_carbs / bolus_units
            ic_lower = latent_lower / bolus_units
            ic_upper = latent_upper / bolus_units
        else:
            ic_mean = float("nan")
            ic_lower = float("nan")
            ic_upper = float("nan")
        if getattr(row, "suppression_reason"):
            identifiability_status = "not_identified" if getattr(row, "suppression_reason") in {"no_meaningful_bolus", "incomplete_window"} else "weakly_identified"
        elif pd.notna(stated_value):
            identifiability_status = "prior_informed"
        elif getattr(row, "confounding_grade") == "low":
            identifiability_status = "response_inferred"
        else:
            identifiability_status = "weakly_identified"
        posterior_rows.append(
            {
                "meal_id": getattr(row, "meal_id"),
                "timestamp": getattr(row, "timestamp"),
                "segment": getattr(row, "segment"),
                "evidence_source": getattr(row, "evidence_source"),
                "stated_carbs": None if pd.isna(stated_value) else float(stated_value),
                "latent_carbs_posterior_mean": latent_carbs,
                "latent_carbs_lower": latent_lower,
                "latent_carbs_upper": latent_upper,
                "carb_accuracy_score": carb_accuracy_score,
                "ic_posterior_mean": ic_mean,
                "ic_lower": ic_lower,
                "ic_upper": ic_upper,
                "confounding_grade": getattr(row, "confounding_grade"),
                "source_quality_grade": getattr(row, "source_quality_grade"),
                "identifiability_status": identifiability_status,
                "suppression_reason": getattr(row, "suppression_reason") or "",
            }
        )
    posterior_meals = pd.DataFrame(posterior_rows).sort_values("timestamp").reset_index(drop=True)

    return LatentMealResearchResult(
        prepared_dataset=dataset,
        research_frame=research_frame,
        research_gate=gate,
        meal_event_registry=meal_event_registry,
        meal_windows=meal_windows,
        posterior_meals=posterior_meals,
        model_comparison=model_comparison,
        research_gate_markdown=gate_markdown,
        meal_window_audit_markdown=_render_latent_meal_window_audit(meal_windows),
        fit_summary_markdown=_render_latent_meal_fit_summary(
            gate=gate,
            reference_ic=reference_ic,
            carb_model=carb_model,
            selected_model=selected_model,
            selected_peak_delta_mae=selected_peak_delta_mae,
        ),
        confidence_report_markdown=_render_latent_meal_confidence_report(posterior_meals),
        model_comparison_markdown=_render_latent_meal_model_comparison(model_comparison),
        segments=segments,
        meal_proxy_mode=meal_proxy_mode,
    )


def write_latent_meal_research_artifacts(
    result: LatentMealResearchResult,
    report_dir: str | Path,
) -> dict[str, Path]:
    root = Path(report_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "research_gate": root / "latent_meal_research_gate.md",
        "meal_event_registry": root / "meal_event_registry.csv",
        "meal_window_audit": root / "meal_window_audit.md",
        "fit_summary": root / "latent_meal_fit_summary.md",
        "posterior_meals": root / "latent_meal_posterior_meals.csv",
        "confidence_report": root / "latent_meal_confidence_report.md",
        "model_comparison": root / "latent_meal_model_comparison.md",
    }
    paths["research_gate"].write_text(result.research_gate_markdown, encoding="utf-8")
    result.meal_event_registry.to_csv(paths["meal_event_registry"], index=False)
    paths["meal_window_audit"].write_text(result.meal_window_audit_markdown, encoding="utf-8")
    paths["fit_summary"].write_text(result.fit_summary_markdown, encoding="utf-8")
    result.posterior_meals.to_csv(paths["posterior_meals"], index=False)
    paths["confidence_report"].write_text(result.confidence_report_markdown, encoding="utf-8")
    paths["model_comparison"].write_text(result.model_comparison_markdown, encoding="utf-8")
    return paths


def _synthetic_base_dataset(
    *,
    apple: bool,
    explicit_carbs: bool,
    proxy_only: bool = False,
    helpful_apple: bool = False,
    null_apple: bool = False,
    corrupted: bool = False,
    low_identifiability: bool = False,
) -> AnalysisReadyHealthDataset:
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=7 * 288, freq="5min")
    minute_of_day = timestamps.hour * 60 + timestamps.minute
    overnight = ((minute_of_day < 360)).astype(float)
    morning = ((minute_of_day >= 360) & (minute_of_day < 660)).astype(float)
    afternoon = ((minute_of_day >= 660) & (minute_of_day < 1020)).astype(float)
    evening = ((minute_of_day >= 1020)).astype(float)
    breakfast = ((minute_of_day >= 7 * 60) & (minute_of_day < 9 * 60)).astype(float)
    lunch = ((minute_of_day >= 12 * 60) & (minute_of_day < 13 * 60 + 30)).astype(float)
    dinner = ((minute_of_day >= 18 * 60) & (minute_of_day < 19 * 60 + 30)).astype(float)
    meal_event = (((breakfast + lunch + dinner) > 0) & (pd.Series(timestamps).dt.minute.eq(0))).astype(float)

    basal = 0.85 + 0.05 * morning + 0.15 * afternoon - 0.18 * overnight
    carbs = meal_event * np.where(morning > 0, 50.0, np.where(afternoon > 0, 60.0, 70.0))
    bolus_units = meal_event * np.where(morning > 0, 3.8, np.where(afternoon > 0, 4.4, 4.8))
    if proxy_only:
        carbs = carbs * 0.0
    activity_value = np.where((minute_of_day >= 17 * 60) & (minute_of_day < 18 * 60), 40.0, 0.0)

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "glucose": 112.0 + 14.0 * overnight + 7.0 * evening + 6.0 * np.sin(np.arange(len(timestamps)) / 18.0),
            "missing_cgm": 0,
            "minutes_since_last_cgm": 0.0,
            "basal_units_per_hour": basal,
            "basal_units_delivered": basal * (5.0 / 60.0),
            "basal_schedule_change": (pd.Series(basal).diff().abs().fillna(0.0) > 0.05).astype(int),
            "minutes_since_basal_change": 0.0,
            "bolus_units": bolus_units,
            "activity_value": activity_value,
            "hour_sin": np.sin(2 * np.pi * (timestamps.hour + timestamps.minute / 60.0) / 24.0),
            "hour_cos": np.cos(2 * np.pi * (timestamps.hour + timestamps.minute / 60.0) / 24.0),
            "dow_sin": np.sin(2 * np.pi * pd.Series(timestamps).dt.dayofweek / 7.0),
            "dow_cos": np.cos(2 * np.pi * pd.Series(timestamps).dt.dayofweek / 7.0),
            "is_weekend": (pd.Series(timestamps).dt.dayofweek >= 5).astype(int),
            "meal_event": meal_event if explicit_carbs and not proxy_only else 0.0,
            "carb_grams": carbs if explicit_carbs and not proxy_only else 0.0,
        }
    )
    frame["carb_roll_sum_60m"] = frame["carb_grams"].rolling(12, min_periods=1).sum()
    frame["carb_roll_sum_120m"] = frame["carb_grams"].rolling(24, min_periods=1).sum()
    frame["minutes_since_last_meal"] = (
        frame["timestamp"] - frame["timestamp"].where(frame["meal_event"].gt(0)).ffill()
    ).dt.total_seconds().div(60.0).fillna(1e6)
    frame["insulin_activity_units"] = frame["bolus_units"].rolling(12, min_periods=1).sum() / 6.0
    frame["iob_units"] = frame["bolus_units"].rolling(24, min_periods=1).sum()
    frame["iob_roll_sum_60m"] = frame["iob_units"].rolling(12, min_periods=1).sum()
    frame["iob_roll_sum_120m"] = frame["iob_units"].rolling(24, min_periods=1).sum()
    frame["glucose_roll_mean_30m"] = frame["glucose"].rolling(6, min_periods=1).mean()
    frame["glucose_roll_mean_60m"] = frame["glucose"].rolling(12, min_periods=1).mean()
    frame["glucose_lag_5m"] = frame["glucose"].shift(1).fillna(frame["glucose"])
    frame["glucose_lag_30m"] = frame["glucose"].shift(6).fillna(frame["glucose"])
    frame["activity_roll_sum_60m"] = frame["activity_value"].rolling(12, min_periods=1).sum()
    frame["carb_bolus_interaction_60m"] = frame["carb_roll_sum_60m"] * frame["bolus_units"]
    frame["carb_iob_interaction_60m"] = frame["carb_roll_sum_60m"] * frame["iob_units"]

    if apple:
        poor_sleep = np.where(pd.Series(timestamps).dt.dayofweek.isin([0, 1]), 1.0, 0.0)
        frame["prior_night_total_sleep_hours"] = 7.8 - 1.6 * poor_sleep
        frame["in_sleep"] = (overnight > 0).astype(int)
        frame["recent_workout_12h"] = np.where((minute_of_day >= 17 * 60) | (minute_of_day < 6 * 60), 1, 0)
        frame["health_activity_roll_sum_60m"] = frame["activity_roll_sum_60m"]
        if helpful_apple:
            rng = np.random.default_rng(17)
            latent_recovery = rng.normal(0.0, 1.0, size=len(frame))
            frame["hrv_latest"] = 44.0 + 9.0 * latent_recovery
            frame["heart_rate_avg_latest"] = 72.0 - 6.0 * latent_recovery + 6.0 * frame["activity_value"].gt(0).astype(float)
            frame["respiratory_rate_latest"] = 14.0 - 0.8 * latent_recovery
            frame["_latent_recovery"] = latent_recovery
        elif null_apple:
            rng = np.random.default_rng(7)
            frame["hrv_latest"] = rng.normal(45.0, 3.0, size=len(frame))
            frame["heart_rate_avg_latest"] = rng.normal(72.0, 4.0, size=len(frame))
            frame["respiratory_rate_latest"] = rng.normal(14.0, 1.0, size=len(frame))
        else:
            frame["hrv_latest"] = 46.0 + 7.0 * overnight - 4.0 * poor_sleep
            frame["heart_rate_avg_latest"] = 68.0 + 8.0 * frame["activity_value"].gt(0).astype(float) + 5.0 * poor_sleep
            frame["respiratory_rate_latest"] = 14.0 + 0.8 * frame["activity_value"].gt(0).astype(float)
    target_delta = (
        0.18 * (110.0 - frame["glucose"])
        - 8.0 * frame["basal_units_per_hour"]
        - 2.4 * frame["bolus_units"]
        + 0.14 * frame["carb_roll_sum_60m"]
        - 0.03 * frame["activity_value"]
        + 7.5 * overnight
    )
    if helpful_apple and apple:
        target_delta = target_delta - 0.65 * frame["hrv_latest"] + 0.35 * frame["heart_rate_avg_latest"] + 4.0 * pd.Series(frame["_latent_recovery"]).lt(-0.75).astype(float)
    if low_identifiability:
        target_delta = pd.Series(0.0, index=frame.index)
    frame["target_delta"] = target_delta
    frame["target_glucose"] = (frame["glucose"] + frame["target_delta"]).clip(lower=65.0, upper=260.0)
    if corrupted:
        frame.loc[0, "timestamp"] = "2020-01-15 10:59:35"
        frame.loc[1, "glucose"] = 999.0

    tandem_features = [
        "glucose",
        "missing_cgm",
        "minutes_since_last_cgm",
        "basal_units_per_hour",
        "basal_units_delivered",
        "basal_schedule_change",
        "minutes_since_basal_change",
        "bolus_units",
        "insulin_activity_units",
        "iob_units",
        "iob_roll_sum_60m",
        "iob_roll_sum_120m",
        "carb_grams",
        "meal_event",
        "minutes_since_last_meal",
        "carb_roll_sum_60m",
        "carb_roll_sum_120m",
        "activity_value",
        "activity_roll_sum_60m",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "glucose_lag_5m",
        "glucose_lag_30m",
        "glucose_roll_mean_30m",
        "glucose_roll_mean_60m",
        "carb_bolus_interaction_60m",
        "carb_iob_interaction_60m",
    ]
    health_features = []
    if apple:
        health_features = [
            "prior_night_total_sleep_hours",
            "in_sleep",
            "recent_workout_12h",
            "health_activity_roll_sum_60m",
            "hrv_latest",
            "heart_rate_avg_latest",
            "respiratory_rate_latest",
        ]
    frame = frame.drop(columns=["_latent_recovery"], errors="ignore")
    return AnalysisReadyHealthDataset(
        frame=frame,
        feature_columns=tandem_features + health_features,
        tandem_feature_columns=tandem_features,
        health_feature_columns=health_features,
        target_column="target_glucose",
        horizon_minutes=30,
        config=FeatureConfig(horizon_minutes=30),
        mode="apple_enriched" if apple else "tandem_only",
        apple_available=apple,
    )


def _best_completed_metric(model_comparison: pd.DataFrame, *, task: str, metric: str) -> float | None:
    subset = model_comparison.loc[model_comparison["task"].eq(task) & model_comparison["status"].eq("completed")]
    if subset.empty or metric not in subset.columns:
        return None
    ordered = subset.sort_values(["selected", metric], ascending=[False, True])
    value = ordered.iloc[0][metric]
    return None if pd.isna(value) else float(value)


def validate_therapy_infra(
    *,
    meal_proxy_mode: str = "strict",
    ic_policy: str = "exploratory_only",
    include_models: tuple[str, ...] | None = None,
) -> TherapyInfraValidationResult:
    include_models = include_models or parse_model_list("ridge,segmented_ridge,tree_boost,ensemble")
    scenarios = [
        ("basal_direction", _synthetic_base_dataset(apple=False, explicit_carbs=True), {"parameter": "basal", "segment": "overnight", "expected_direction": "decrease"}),
        ("explicit_icr", _synthetic_base_dataset(apple=False, explicit_carbs=True), {"parameter": "I/C ratio", "segment": "morning", "expected_status": "exploratory"}),
        ("proxy_only_icr", _synthetic_base_dataset(apple=False, explicit_carbs=False, proxy_only=True), {"parameter": "I/C ratio", "segment": "morning", "max_status": "exploratory"}),
        ("apple_helpful", _synthetic_base_dataset(apple=True, explicit_carbs=True, helpful_apple=True), {"apple_compare": True, "expect_apple_help": True}),
        ("apple_null", _synthetic_base_dataset(apple=True, explicit_carbs=True, null_apple=True), {"apple_compare": True, "expect_apple_help": False}),
        ("corrupted", _synthetic_base_dataset(apple=True, explicit_carbs=False, proxy_only=True, corrupted=True), {"expect_source_degraded": True}),
        ("low_identifiability", _synthetic_base_dataset(apple=False, explicit_carbs=False, low_identifiability=True), {"expect_suppressed": True}),
    ]
    rows: list[dict[str, Any]] = []
    audit_lines = [
        "# Therapy Synthetic Recommendation Audit",
        "",
        "| scenario | parameter | segment | status | proposed_change_percent | reasons_against |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for name, dataset, expectation in scenarios:
        result = run_therapy_research(
            dataset,
            segments=parse_therapy_segments(),
            include_models=include_models,
            meal_proxy_mode=meal_proxy_mode,
            ic_policy=ic_policy,
        )
        passed = True
        note = ""
        if expectation.get("apple_compare"):
            no_apple_dataset = AnalysisReadyHealthDataset(
                frame=dataset.frame.drop(columns=dataset.health_feature_columns, errors="ignore"),
                feature_columns=list(dataset.tandem_feature_columns),
                tandem_feature_columns=list(dataset.tandem_feature_columns),
                health_feature_columns=[],
                target_column=dataset.target_column,
                horizon_minutes=dataset.horizon_minutes,
                config=dataset.config,
                mode="tandem_only",
                apple_available=False,
            )
            ablated = run_therapy_research(
                no_apple_dataset,
                segments=parse_therapy_segments(),
                include_models=include_models,
                meal_proxy_mode=meal_proxy_mode,
                ic_policy=ic_policy,
            )
            with_apple = _best_completed_metric(result.model_comparison, task="icr", metric="safety_weighted_mae")
            without_apple = _best_completed_metric(ablated.model_comparison, task="icr", metric="safety_weighted_mae")
            if with_apple is None or without_apple is None:
                passed = False
                note = "apple_comparison_missing"
            elif expectation["expect_apple_help"]:
                passed = with_apple < without_apple
                note = f"with_apple={with_apple:.3f},without_apple={without_apple:.3f}"
            else:
                passed = with_apple >= without_apple * 0.98
                note = f"with_apple={with_apple:.3f},without_apple={without_apple:.3f}"
        elif expectation.get("expect_source_degraded"):
            statuses = set(result.research_gate["source_quality_status"])
            passed = "degraded" in statuses
            note = ",".join(sorted(statuses))
        elif expectation.get("expect_suppressed"):
            basal_rows = result.recommendations.loc[result.recommendations["parameter"].eq("basal")]
            passed = not basal_rows.empty and set(basal_rows["status"]).issubset({"suppressed"})
            note = ",".join(sorted(set(basal_rows["status"]))) if not basal_rows.empty else "missing"
        else:
            rec = result.recommendations.loc[
                result.recommendations["parameter"].eq(expectation["parameter"])
                & result.recommendations["segment"].eq(expectation["segment"])
            ].head(1)
            if rec.empty:
                passed = False
                note = "missing_recommendation"
            else:
                status = str(rec.iloc[0]["status"])
                direction = str(rec.iloc[0]["expected_direction"])
                if "expected_direction" in expectation:
                    gain = float(rec.iloc[0]["mean_expected_gain"]) if not pd.isna(rec.iloc[0]["mean_expected_gain"]) else float("-inf")
                    passed = direction == expectation["expected_direction"] and gain > 0
                    note = f"status={status},direction={direction},gain={gain:.3f}"
                else:
                    allowed = expectation.get("max_status", expectation.get("expected_status"))
                    if allowed == "exploratory":
                        passed = status in {"exploratory", "suppressed"}
                    else:
                        passed = status == expectation["expected_status"]
                    note = f"status={status}"
            for rec_row in result.recommendations.itertuples(index=False):
                audit_lines.append(
                    "| {scenario} | {parameter} | {segment} | {status} | {change} | {reasons} |".format(
                        scenario=name,
                        parameter=rec_row.parameter,
                        segment=rec_row.segment,
                        status=rec_row.status,
                        change="NA" if pd.isna(rec_row.proposed_change_percent) else f"{float(rec_row.proposed_change_percent):.1f}",
                        reasons=getattr(rec_row, "reasons_against", "") or "",
                    )
                )
        rows.append({"scenario": name, "passed": bool(passed), "note": note})
    scenario_results = pd.DataFrame(rows)
    report_lines = [
        "# Therapy Infrastructure Validation",
        "",
        f"- scenario_count: {len(scenario_results)}",
        f"- passed_count: {int(scenario_results['passed'].sum()) if not scenario_results.empty else 0}",
        "",
        "| scenario | passed | note |",
        "| --- | --- | --- |",
    ]
    for row in scenario_results.itertuples(index=False):
        report_lines.append(f"| {row.scenario} | {'yes' if bool(row.passed) else 'no'} | {row.note} |")
    return TherapyInfraValidationResult(
        scenario_results=scenario_results,
        report_markdown="\n".join(report_lines) + "\n",
        recommendation_audit_markdown="\n".join(audit_lines) + "\n",
    )


def write_therapy_infra_validation_artifacts(result: TherapyInfraValidationResult, report_dir: str | Path) -> dict[str, Path]:
    root = Path(report_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "validation_report": root / "therapy_infra_validation.md",
        "scenario_results": root / "therapy_synthetic_results.csv",
        "recommendation_audit": root / "therapy_synthetic_recommendation_audit.md",
    }
    paths["validation_report"].write_text(result.report_markdown, encoding="utf-8")
    result.scenario_results.to_csv(paths["scenario_results"], index=False)
    paths["recommendation_audit"].write_text(result.recommendation_audit_markdown, encoding="utf-8")
    return paths


def summarize_overnight_basal_evidence(result: TherapyResearchResult) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = result.research_frame.copy()
    if frame.empty or "therapy_segment" not in frame.columns:
        summary = {
            "status": "blocked",
            "reason": "no_research_frame",
            "overnight_rows": 0,
            "clean_rows": 0,
            "overnight_nights": 0,
            "clean_nights": 0,
            "stable_epochs": 0,
            "mean_glucose": np.nan,
            "mean_target_delta": np.nan,
            "expected_direction": "",
        }
        return summary, pd.DataFrame(columns=["reason", "count"])

    overnight = frame.loc[frame["therapy_segment"].astype(str).eq("overnight")].copy()
    if overnight.empty:
        summary = {
            "status": "blocked",
            "reason": "no_overnight_rows",
            "overnight_rows": 0,
            "clean_rows": 0,
            "overnight_nights": 0,
            "clean_nights": 0,
            "stable_epochs": 0,
            "mean_glucose": np.nan,
            "mean_target_delta": np.nan,
            "expected_direction": "",
        }
        return summary, pd.DataFrame(columns=["reason", "count"])

    overnight["date"] = pd.to_datetime(overnight["timestamp"], errors="coerce").dt.normalize()
    clean_mask = overnight.get("basal_context", pd.Series(0, index=overnight.index)).fillna(0).astype(int).eq(1)
    clean = overnight.loc[clean_mask].copy()
    exclusion_counts = pd.DataFrame(
        [
            {"reason": "recent_meal_or_proxy", "count": int(overnight.get("recent_meal_120m", pd.Series(0, index=overnight.index)).fillna(0).astype(int).sum())},
            {"reason": "recent_bolus_or_iob", "count": int(overnight.get("recent_bolus_120m", pd.Series(0, index=overnight.index)).fillna(0).astype(int).sum())},
            {"reason": "recent_exercise", "count": int(overnight.get("recent_exercise_context", pd.Series(0, index=overnight.index)).fillna(0).astype(int).sum())},
            {"reason": "closed_loop_or_transition", "count": int(overnight.get("closed_loop_confounding_flag", pd.Series(0, index=overnight.index)).fillna(0).astype(int).sum())},
            {"reason": "missing_cgm", "count": int(pd.to_numeric(overnight.get("missing_cgm", pd.Series(0, index=overnight.index)), errors="coerce").fillna(0.0).gt(0).sum())},
        ]
    )
    rec = result.recommendations.loc[
        result.recommendations["parameter"].eq("basal")
        & result.recommendations["segment"].eq("overnight")
    ].head(1)
    expected_direction = "" if rec.empty else str(rec.iloc[0].get("expected_direction") or "")
    reason = ""
    status = "blocked"
    if len(clean) >= 72 and int(clean["date"].nunique()) >= 3 and int(pd.to_numeric(clean.get("therapy_stable_epoch"), errors="coerce").nunique()) >= 2:
        status = "identifiable"
    elif len(clean) >= 36 and int(clean["date"].nunique()) >= 2:
        status = "weakly_identifiable"
        reason = "limited_clean_overnight_windows"
    else:
        reason = "insufficient_clean_overnight_windows"
    summary = {
        "status": status,
        "reason": reason,
        "overnight_rows": int(len(overnight)),
        "clean_rows": int(len(clean)),
        "overnight_nights": int(overnight["date"].nunique()),
        "clean_nights": int(clean["date"].nunique()) if not clean.empty else 0,
        "stable_epochs": int(pd.to_numeric(clean.get("therapy_stable_epoch"), errors="coerce").nunique()) if not clean.empty else 0,
        "mean_glucose": float(pd.to_numeric(clean.get("glucose"), errors="coerce").mean()) if not clean.empty else float("nan"),
        "mean_target_delta": float(pd.to_numeric(clean.get(TARGET_DELTA_COLUMN), errors="coerce").mean()) if not clean.empty else float("nan"),
        "expected_direction": expected_direction,
    }
    return summary, exclusion_counts


__all__ = [
    "DEFAULT_SEGMENT_SPEC",
    "LatentMealResearchResult",
    "TherapyInfraValidationResult",
    "TherapyResearchResult",
    "TherapySegment",
    "build_meal_event_registry",
    "build_therapy_feature_registry",
    "build_therapy_research_frame",
    "build_source_report_cards",
    "parse_model_list",
    "parse_therapy_segments",
    "run_latent_meal_icr_research",
    "run_therapy_research",
    "summarize_overnight_basal_evidence",
    "validate_therapy_infra",
    "write_latent_meal_research_artifacts",
    "write_therapy_infra_validation_artifacts",
    "write_therapy_research_artifacts",
]
