from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import bayesian_t1dm.cli as cli
from bayesian_t1dm.cli import main
from bayesian_t1dm.health_auto_export import AnalysisReadyHealthDataset
from bayesian_t1dm.features import FeatureConfig
from bayesian_t1dm.therapy_research import (
    build_representative_latent_meal_fixture,
    build_first_meal_clean_window_registry,
    build_therapy_research_frame,
    parse_model_list,
    parse_therapy_segments,
    run_latent_meal_icr_research,
    run_therapy_research,
    write_latent_meal_research_artifacts,
    write_representative_latent_meal_fixture_artifacts,
    validate_therapy_infra,
    _synthetic_base_dataset,
)


def _build_prepared_dataset(*, apple: bool) -> AnalysisReadyHealthDataset:
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=576, freq="5min")
    minute_of_day = timestamps.hour * 60 + timestamps.minute
    overnight = ((minute_of_day < 360)).astype(float)
    morning = ((minute_of_day >= 360) & (minute_of_day < 660)).astype(float)
    afternoon = ((minute_of_day >= 660) & (minute_of_day < 1020)).astype(float)
    evening = ((minute_of_day >= 1020)).astype(float)

    basal = 0.8 + 0.2 * afternoon + 0.05 * morning
    glucose = 105.0 + 18.0 * overnight + 8.0 * evening + 4.0 * np.sin(np.arange(len(timestamps)) / 12.0)
    meal_times = {
        pd.Timestamp("2025-01-01 08:00:00"),
        pd.Timestamp("2025-01-01 12:00:00"),
        pd.Timestamp("2025-01-01 18:00:00"),
        pd.Timestamp("2025-01-02 08:00:00"),
        pd.Timestamp("2025-01-02 12:00:00"),
        pd.Timestamp("2025-01-02 18:00:00"),
    }
    meal_event = pd.Series(timestamps).isin(meal_times).astype(float)
    carb_grams = meal_event * np.where(morning > 0, 45.0, np.where(afternoon > 0, 55.0, 60.0))
    bolus_units = meal_event * np.where(morning > 0, 4.5, np.where(afternoon > 0, 5.0, 5.5))
    activity_value = np.where((minute_of_day >= 720) & (minute_of_day < 780), 30.0, 0.0)

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "glucose": glucose,
            "missing_cgm": 0,
            "minutes_since_last_cgm": 0.0,
            "basal_units_per_hour": basal,
            "basal_units_delivered": basal * (5.0 / 60.0),
            "basal_schedule_change": (pd.Series(basal).diff().abs().fillna(0.0) > 0).astype(int),
            "minutes_since_basal_change": 0.0,
            "bolus_units": bolus_units,
            "activity_value": activity_value,
            "hour_sin": np.sin(2 * np.pi * (timestamps.hour + timestamps.minute / 60.0) / 24.0),
            "hour_cos": np.cos(2 * np.pi * (timestamps.hour + timestamps.minute / 60.0) / 24.0),
            "dow_sin": np.sin(2 * np.pi * pd.Series(timestamps).dt.dayofweek / 7.0),
            "dow_cos": np.cos(2 * np.pi * pd.Series(timestamps).dt.dayofweek / 7.0),
            "is_weekend": (pd.Series(timestamps).dt.dayofweek >= 5).astype(int),
            "meal_event": meal_event,
            "carb_grams": carb_grams,
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
        frame["prior_night_total_sleep_hours"] = np.where(pd.Series(timestamps).dt.date == pd.Timestamp("2025-01-01").date(), 6.5, 7.8)
        frame["in_sleep"] = ((minute_of_day < 360)).astype(int)
        frame["recent_workout_12h"] = np.where((minute_of_day >= 1020) | (minute_of_day < 300), 1, 0)
        frame["health_activity_roll_sum_60m"] = frame["activity_roll_sum_60m"]
        frame["hrv_latest"] = 48.0 + 5.0 * overnight - 3.0 * evening
        frame["heart_rate_avg_latest"] = 68.0 + 6.0 * frame["activity_value"].gt(0).astype(float)
    target_delta = (
        0.18 * (110.0 - frame["glucose"])
        + 0.12 * frame["carb_roll_sum_60m"]
        - 6.0 * frame["basal_units_per_hour"]
        - 1.5 * frame["bolus_units"]
        - 0.02 * frame["activity_value"]
        - 0.05 * frame.get("hrv_latest", pd.Series(0.0, index=frame.index)).fillna(0.0)
        + 4.0 * overnight
    )
    frame["target_delta"] = target_delta
    frame["target_glucose"] = (frame["glucose"] + frame["target_delta"]).clip(lower=65.0, upper=240.0)

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
    health_features = [
        "prior_night_total_sleep_hours",
        "in_sleep",
        "recent_workout_12h",
        "health_activity_roll_sum_60m",
        "hrv_latest",
        "heart_rate_avg_latest",
    ] if apple else []
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
        explicit_carb_source_available=True,
    )


def test_build_therapy_research_frame_assigns_segments_and_contexts():
    dataset = _build_prepared_dataset(apple=True)
    segments = parse_therapy_segments()
    research_frame = build_therapy_research_frame(dataset, segments=segments)

    assert set(research_frame["therapy_segment"]) == {"overnight", "morning", "afternoon", "evening"}
    assert research_frame["basal_context"].sum() > 0
    assert research_frame["meal_context"].sum() > 0
    assert "sleep_deficit_flag" in research_frame.columns
    assert "post_workout_meal_context" in research_frame.columns
    assert "overnight_hrv_interaction" in research_frame.columns
    assert "meal_proxy_event" in research_frame.columns
    assert "therapy_stable_epoch" in research_frame.columns
    assert "explicit_carb_grams" in research_frame.columns
    assert "explicit_meal_event" in research_frame.columns
    assert "meal_truth_status" in research_frame.columns
    assert set(research_frame["meal_truth_status"]) >= {"observed_explicit", "missing_from_source"}


def test_build_therapy_research_frame_handles_missing_meal_signal_without_marking_all_rows_as_meals():
    dataset = _build_prepared_dataset(apple=True)
    frame = dataset.frame.copy()
    frame["timestamp"] = frame["timestamp"].astype(str)
    frame["meal_event"] = 0.0
    frame["carb_grams"] = 0.0
    frame["carb_roll_sum_60m"] = 0.0
    frame["carb_roll_sum_120m"] = 0.0
    frame["minutes_since_last_meal"] = 0.0
    frame["bolus_units"] = 0.0
    frame["iob_units"] = 0.0
    frame["iob_roll_sum_60m"] = 0.0
    frame["iob_roll_sum_120m"] = 0.0
    dataset = AnalysisReadyHealthDataset(
        frame=frame,
        feature_columns=dataset.feature_columns,
        tandem_feature_columns=dataset.tandem_feature_columns,
        health_feature_columns=dataset.health_feature_columns,
        target_column=dataset.target_column,
        horizon_minutes=dataset.horizon_minutes,
        config=dataset.config,
        mode=dataset.mode,
        apple_available=dataset.apple_available,
        explicit_carb_source_available=dataset.explicit_carb_source_available,
    )

    research_frame = build_therapy_research_frame(dataset, segments=parse_therapy_segments())

    assert pd.api.types.is_datetime64_any_dtype(research_frame["timestamp"])
    assert research_frame["meal_context"].sum() == 0
    assert research_frame["basal_context"].sum() > 0
    assert research_frame["correction_context"].sum() == 0


def test_run_therapy_research_builds_outputs_and_stages_sensitivity_factor():
    dataset = _build_prepared_dataset(apple=True)
    result = run_therapy_research(
        dataset,
        segments=parse_therapy_segments(),
        include_models=parse_model_list("ridge,elastic_net,segmented_ridge,tree_boost,ensemble"),
    )

    assert not result.feature_registry.empty
    assert {"therapy_context", "apple_measurements"}.issubset(set(result.feature_registry["source_family"]))
    assert not result.research_gate.empty
    assert not result.meal_proxy_audit.empty
    assert not result.model_comparison.empty
    assert {"basal", "icr", "sensitivity_factor"}.issubset(set(result.model_comparison["task"]))
    assert not result.recommendations.empty
    sensitivity_rows = result.recommendations.loc[result.recommendations["parameter"] == "sensitivity factor"]
    assert not sensitivity_rows.empty
    assert set(sensitivity_rows["status"]) == {"suppressed"}
    assert any("predictive and associational" in line for line in result.recommendation_markdown.splitlines())
    assert "Therapy Research Gate" in result.research_gate_markdown
    assert "Meal Proxy Audit" in result.meal_proxy_audit_markdown


def test_run_therapy_research_without_apple_degrades_cleanly():
    dataset = _build_prepared_dataset(apple=False)
    result = run_therapy_research(
        dataset,
        segments=parse_therapy_segments(),
        include_models=parse_model_list("ridge,segmented_ridge,tree_boost"),
    )

    assert result.prepared_dataset.apple_available is False
    assert not result.model_comparison.empty
    assert "apple_measurements" not in set(result.feature_registry["source_family"])
    assert "therapy_feature_audit" not in result.feature_audit_markdown.lower() or "apple_available: False" in result.feature_audit_markdown


def test_run_latent_meal_icr_research_builds_explicit_meal_outputs(tmp_path):
    dataset = _synthetic_base_dataset(apple=True, explicit_carbs=True)

    result = run_latent_meal_icr_research(
        dataset,
        segments=parse_therapy_segments(),
        meal_proxy_mode="strict",
    )

    assert not result.research_gate.empty
    assert not result.meal_event_registry.empty
    assert not result.meal_windows.empty
    assert result.posterior_meals.empty
    assert result.model_comparison.empty
    assert not result.first_meal_exclusion_summary.empty
    assert "Latent Meal Research Gate" in result.research_gate_markdown
    assert "Meal Truth Semantics Report" in result.meal_truth_semantics_report_markdown
    assert "First Meal Clean Window Audit" in result.meal_window_audit_markdown

    paths = write_latent_meal_research_artifacts(result, tmp_path / "latent")
    assert paths["research_gate"].exists()
    assert paths["meal_truth_semantics_report"].exists()
    assert paths["meal_event_registry"].exists()
    assert paths["first_meal_clean_window_registry"].exists()
    assert paths["first_meal_exclusion_summary"].exists()


def test_run_latent_meal_icr_research_returns_honest_empty_or_strict_results():
    dataset = _synthetic_base_dataset(apple=False, explicit_carbs=False, proxy_only=True)

    result = run_latent_meal_icr_research(
        dataset,
        segments=parse_therapy_segments(),
        meal_proxy_mode="strict",
    )

    assert result.research_scope == "foundation"
    assert "explicit_carb_source_available: False" in result.meal_truth_semantics_report_markdown
    assert "latent_fit_status: intentionally_skipped_foundation_mode" in result.research_gate_markdown
    assert result.posterior_meals.empty


def test_run_latent_meal_icr_research_full_scope_not_implemented():
    dataset = _synthetic_base_dataset(apple=True, explicit_carbs=True)

    try:
        run_latent_meal_icr_research(
            dataset,
            segments=parse_therapy_segments(),
            meal_proxy_mode="strict",
            research_scope="full",
        )
    except NotImplementedError as exc:
        assert "not_yet_implemented" in str(exc)
    else:
        raise AssertionError("Expected NotImplementedError for full scope")


def test_build_representative_latent_meal_fixture_retains_candidate_days_and_selects_background_days(tmp_path):
    dataset = _synthetic_base_dataset(apple=True, explicit_carbs=True)
    frame = dataset.frame.copy()
    all_dates = sorted(pd.to_datetime(frame["timestamp"], errors="coerce").dt.normalize().dropna().unique())
    suppressed_dates = set(all_dates[-2:])
    timestamp_series = pd.to_datetime(frame["timestamp"], errors="coerce")
    morning_mask = (
        timestamp_series.dt.normalize().isin(suppressed_dates)
        & (((timestamp_series.dt.hour * 60) + timestamp_series.dt.minute) >= 360)
        & (((timestamp_series.dt.hour * 60) + timestamp_series.dt.minute) < 660)
    )
    frame.loc[morning_mask, "meal_event"] = 0.0
    frame.loc[morning_mask, "carb_grams"] = 0.0
    frame.loc[morning_mask, "bolus_units"] = 0.0
    frame["carb_roll_sum_60m"] = frame["carb_grams"].rolling(12, min_periods=1).sum()
    frame["carb_roll_sum_120m"] = frame["carb_grams"].rolling(24, min_periods=1).sum()
    frame["minutes_since_last_meal"] = (
        frame["timestamp"] - frame["timestamp"].where(frame["meal_event"].gt(0)).ffill()
    ).dt.total_seconds().div(60.0).fillna(1e6)
    frame["insulin_activity_units"] = frame["bolus_units"].rolling(12, min_periods=1).sum() / 6.0
    frame["iob_units"] = frame["bolus_units"].rolling(24, min_periods=1).sum()
    frame["iob_roll_sum_60m"] = frame["iob_units"].rolling(12, min_periods=1).sum()
    frame["iob_roll_sum_120m"] = frame["iob_units"].rolling(24, min_periods=1).sum()
    frame["carb_bolus_interaction_60m"] = frame["carb_roll_sum_60m"] * frame["bolus_units"]
    frame["carb_iob_interaction_60m"] = frame["carb_roll_sum_60m"] * frame["iob_units"]
    dataset = AnalysisReadyHealthDataset(
        frame=frame,
        feature_columns=dataset.feature_columns,
        tandem_feature_columns=dataset.tandem_feature_columns,
        health_feature_columns=dataset.health_feature_columns,
        target_column=dataset.target_column,
        horizon_minutes=dataset.horizon_minutes,
        config=dataset.config,
        mode=dataset.mode,
        apple_available=dataset.apple_available,
        explicit_carb_source_available=dataset.explicit_carb_source_available,
    )

    fixture = build_representative_latent_meal_fixture(
        dataset,
        segments=parse_therapy_segments(),
        meal_proxy_mode="strict",
        background_days=2,
        seed=11,
    )

    source_candidate_days = int(pd.to_datetime(fixture.source_result.meal_windows["date"], errors="coerce").nunique())
    fixture_candidate_days = int(pd.to_datetime(fixture.fixture_result.meal_windows["date"], errors="coerce").nunique())
    background_selected = int(fixture.selected_day_manifest["selection_type"].astype(str).eq("background").sum())

    assert source_candidate_days == fixture_candidate_days
    assert background_selected == 2
    assert "candidate_day_retention: 1.000" in fixture.summary_markdown

    paths = write_representative_latent_meal_fixture_artifacts(fixture, tmp_path / "fixture")
    assert paths["prepared_model_data"].exists()
    assert paths["fixture_summary"].exists()
    assert paths["selected_day_manifest"].exists()
    assert paths["research_gate"].exists()


def test_first_meal_registry_ignores_implausible_workout_flags_and_schedule_change_only():
    dataset = _build_prepared_dataset(apple=True)
    frame = dataset.frame.copy()
    frame["recent_workout_6h"] = 1
    frame["recent_workout_12h"] = 1
    frame["workout_count_24h"] = 1000.0
    frame["workout_duration_sum_24h"] = 1000.0 * 3600.0
    frame["workout_summary_plausible"] = 0
    frame["health_activity_roll_sum_60m"] = 0.0
    frame["basal_schedule_change"] = 1
    frame["minutes_since_basal_change"] = 0.0
    # Keep basal stable so schedule-change proxy alone should not trigger premeal closed-loop confounding.
    morning_mask = pd.to_datetime(frame["timestamp"], errors="coerce").dt.hour.lt(11)
    frame.loc[morning_mask, "basal_units_per_hour"] = 0.8
    dataset = AnalysisReadyHealthDataset(
        frame=frame,
        feature_columns=dataset.feature_columns,
        tandem_feature_columns=dataset.tandem_feature_columns,
        health_feature_columns=dataset.health_feature_columns,
        target_column=dataset.target_column,
        horizon_minutes=dataset.horizon_minutes,
        config=dataset.config,
        mode=dataset.mode,
        apple_available=dataset.apple_available,
        explicit_carb_source_available=dataset.explicit_carb_source_available,
    )

    research_frame = build_therapy_research_frame(dataset, segments=parse_therapy_segments(), meal_proxy_mode="strict")
    windows = build_first_meal_clean_window_registry(research_frame)

    assert len(windows) == 2
    assert windows["recent_workout_6h"].sum() == 0
    assert windows["closed_loop_confounding_premeal"].sum() == 0
    assert windows["included"].sum() == 2


def test_strict_meal_proxy_only_creates_high_confidence_meal_contexts():
    dataset = _build_prepared_dataset(apple=False)
    frame = dataset.frame.copy()
    frame["meal_event"] = 0.0
    frame["carb_grams"] = 0.0
    frame["carb_roll_sum_60m"] = 0.0
    frame["carb_roll_sum_120m"] = 0.0
    frame["minutes_since_last_meal"] = 0.0
    frame.loc[frame["timestamp"].isin([pd.Timestamp("2025-01-01 08:00:00"), pd.Timestamp("2025-01-01 12:00:00")]), "glucose"] = 120.0
    dataset = AnalysisReadyHealthDataset(
        frame=frame,
        feature_columns=dataset.feature_columns,
        tandem_feature_columns=dataset.tandem_feature_columns,
        health_feature_columns=dataset.health_feature_columns,
        target_column=dataset.target_column,
        horizon_minutes=dataset.horizon_minutes,
        config=dataset.config,
        mode=dataset.mode,
        apple_available=dataset.apple_available,
        explicit_carb_source_available=dataset.explicit_carb_source_available,
    )
    research_frame = build_therapy_research_frame(dataset, segments=parse_therapy_segments(), meal_proxy_mode="strict")

    assert research_frame["meal_proxy_event"].sum() > 0
    assert set(research_frame.loc[research_frame["meal_proxy_event"].eq(1), "bolus_proxy_class"]) == {"meal_like"}


def test_research_therapy_settings_cli_writes_artifacts(tmp_path, monkeypatch):
    dataset = _build_prepared_dataset(apple=True)
    preparation = SimpleNamespace(dataset=dataset)
    monkeypatch.setattr(cli, "_prepare_model_data", lambda args, paths, session=None: preparation)

    report_dir = tmp_path / "reports"
    exit_code = main(
        [
            "--root",
            str(tmp_path),
            "research-therapy-settings",
            "--report-dir",
            str(report_dir),
            "--include-models",
            "ridge,elastic_net,segmented_ridge,tree_boost,ensemble",
        ]
    )

    assert exit_code == 0
    assert (report_dir / "therapy_research_gate.md").exists()
    assert (report_dir / "therapy_feature_audit.md").exists()
    assert (report_dir / "therapy_feature_registry.csv").exists()
    assert (report_dir / "meal_proxy_audit.md").exists()
    assert (report_dir / "therapy_model_comparison.md").exists()
    assert (report_dir / "therapy_segment_evidence.csv").exists()
    assert (report_dir / "therapy_recommendation_research.md").exists()
    assert (report_dir / "tandem_source_report_card.md").exists()
    assert (report_dir / "apple_source_report_card.md").exists()
    assert (report_dir / "source_numeric_summary.csv").exists()
    assert (report_dir / "source_missingness_summary.csv").exists()

    comparison = pd.read_csv(report_dir / "therapy_feature_registry.csv")
    assert "feature" in comparison.columns


def test_validate_therapy_infra_reports_scenarios():
    result = validate_therapy_infra(include_models=parse_model_list("ridge,segmented_ridge,tree_boost"))

    assert not result.scenario_results.empty
    assert {"basal_direction", "explicit_icr", "proxy_only_icr", "apple_helpful", "apple_null", "corrupted", "low_identifiability"}.issubset(set(result.scenario_results["scenario"]))
    assert "Therapy Infrastructure Validation" in result.report_markdown
