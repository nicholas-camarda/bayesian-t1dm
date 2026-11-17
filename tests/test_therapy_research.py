from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import bayesian_t1dm.cli as cli
from bayesian_t1dm.cli import main
from bayesian_t1dm.health_auto_export import AnalysisReadyHealthDataset
from bayesian_t1dm.features import FeatureConfig
from bayesian_t1dm.therapy_research import (
    LatentMealTherapySettings,
    build_representative_latent_meal_fixture,
    build_first_meal_clean_window_registry,
    build_latent_meal_tiered_window_registry,
    build_therapy_research_frame,
    extract_latent_meal_therapy_settings_from_tandem_raw,
    parse_model_list,
    parse_therapy_segments,
    run_latent_meal_icr_research,
    run_therapy_research,
    write_latent_meal_research_artifacts,
    write_representative_latent_meal_fixture_artifacts,
    validate_therapy_infra,
    _synthetic_base_dataset,
)


def _latent_settings() -> LatentMealTherapySettings:
    return LatentMealTherapySettings(
        icr_grams_per_unit={"overnight": 12.0, "morning": 10.0, "afternoon": 11.0, "evening": 9.0},
        correction_factor_mgdl_per_unit={"overnight": 45.0, "morning": 40.0, "afternoon": 45.0, "evening": 40.0},
        target_glucose_mgdl=110.0,
        insulin_action_minutes=300,
        carb_absorption_minutes_prior=180,
        max_plausible_meal_carbs=180.0,
    )


def _pump_event_metadata_payload() -> dict:
    return {
        "requested_end": "2026-03-31",
        "selected_pump": {
            "lastUpload": {
                "lastUploadedAt": "2026-04-01T02:15:16.6",
                "settings": {
                    "profiles": {
                        "activeIdp": 0,
                        "profile": [
                            {
                                "idp": 0,
                                "name": "Main",
                                "insulinDuration": 300,
                                "maxBolus": 25000,
                                "tDependentSegs": [
                                    {"startTime": 0, "carbRatio": 7000, "isf": 60, "targetBg": 110},
                                    {"startTime": 300, "carbRatio": 7000, "isf": 65, "targetBg": 110},
                                    {"startTime": 540, "carbRatio": 7000, "isf": 65, "targetBg": 110},
                                    {"startTime": 900, "carbRatio": 6800, "isf": 60, "targetBg": 110},
                                    {"startTime": 1140, "carbRatio": 6800, "isf": 60, "targetBg": 110},
                                    {"startTime": 1260, "carbRatio": 7000, "isf": 58, "targetBg": 110},
                                ],
                            },
                            {
                                "idp": 1,
                                "name": "Unused",
                                "insulinDuration": 300,
                                "maxBolus": 25000,
                                "tDependentSegs": [{"startTime": 0, "carbRatio": 12000, "isf": 80, "targetBg": 120}],
                            },
                        ],
                    },
                    "pumpSettings": {"maxBolus": 25000},
                },
            }
        },
    }


def _first_latent_carb_median(dataset: AnalysisReadyHealthDataset) -> float:
    result = run_latent_meal_icr_research(
        dataset,
        segments=parse_therapy_segments(),
        meal_proxy_mode="strict",
        research_scope="full",
        therapy_settings=_latent_settings(),
    )
    assert result.latent_carb_estimates is not None
    assert not result.latent_carb_estimates.empty
    return float(result.latent_carb_estimates.sort_values("meal_start")["latent_carb_median"].iloc[0])


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


def _build_clean_first_meal_dataset(*, apple: bool, explicit_carbs: bool) -> AnalysisReadyHealthDataset:
    dataset = _build_prepared_dataset(apple=apple)
    frame = dataset.frame.copy()
    frame["recent_workout_6h"] = 0
    frame["recent_workout_12h"] = 0
    frame["workout_count_24h"] = 0.0
    frame["workout_duration_sum_24h"] = 0.0
    frame["workout_summary_plausible"] = 1
    frame["health_activity_roll_sum_60m"] = 0.0
    frame["basal_schedule_change"] = 0
    frame["minutes_since_basal_change"] = 1e6
    morning_mask = pd.to_datetime(frame["timestamp"], errors="coerce").dt.hour.lt(11)
    frame.loc[morning_mask, "basal_units_per_hour"] = 0.8
    if not explicit_carbs:
        frame["meal_event"] = 0.0
        frame["carb_grams"] = 0.0
        frame["carb_roll_sum_60m"] = 0.0
        frame["carb_roll_sum_120m"] = 0.0
        frame["minutes_since_last_meal"] = 0.0
        explicit_carb_source_available = False
    else:
        explicit_carb_source_available = True
    return AnalysisReadyHealthDataset(
        frame=frame,
        feature_columns=dataset.feature_columns,
        tandem_feature_columns=dataset.tandem_feature_columns,
        health_feature_columns=dataset.health_feature_columns,
        target_column=dataset.target_column,
        horizon_minutes=dataset.horizon_minutes,
        config=dataset.config,
        mode=dataset.mode,
        apple_available=dataset.apple_available,
        explicit_carb_source_available=explicit_carb_source_available,
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


def test_build_therapy_research_frame_splits_step_activity_from_hard_exercise():
    dataset = _build_prepared_dataset(apple=True)
    frame = dataset.frame.copy()
    frame["recent_workout_6h"] = 0
    frame["recent_workout_12h"] = 0
    frame["workout_summary_plausible"] = 1
    frame["health_activity_roll_sum_60m"] = 0.0
    frame.loc[[10, 11, 12], "health_activity_roll_sum_60m"] = [1.0, 600.0, 1200.0]
    dataset = replace(dataset, frame=frame)

    research_frame = build_therapy_research_frame(dataset, segments=parse_therapy_segments())

    light = research_frame.loc[research_frame["timestamp"].eq(frame.loc[10, "timestamp"])].iloc[0]
    moderate = research_frame.loc[research_frame["timestamp"].eq(frame.loc[11, "timestamp"])].iloc[0]
    high = research_frame.loc[research_frame["timestamp"].eq(frame.loc[12, "timestamp"])].iloc[0]
    assert int(light["light_activity_context"]) == 1
    assert int(light["moderate_activity_context"]) == 0
    assert int(light["high_activity_context"]) == 0
    assert int(light["recent_exercise_context"]) == 0
    assert int(moderate["light_activity_context"]) == 0
    assert int(moderate["moderate_activity_context"]) == 1
    assert int(moderate["high_activity_context"]) == 0
    assert int(moderate["recent_exercise_context"]) == 0
    assert int(high["light_activity_context"]) == 0
    assert int(high["moderate_activity_context"]) == 0
    assert int(high["high_activity_context"]) == 1
    assert int(high["recent_exercise_context"]) == 1


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
    assert "Which Breakfasts Were Used" in result.meal_window_audit_markdown

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


def test_run_latent_meal_icr_research_full_scope_builds_posterior_outputs():
    dataset = _build_clean_first_meal_dataset(apple=True, explicit_carbs=True)

    result = run_latent_meal_icr_research(
        dataset,
        segments=parse_therapy_segments(),
        meal_proxy_mode="strict",
        research_scope="full",
        therapy_settings=_latent_settings(),
    )

    assert result.research_scope == "full"
    assert not result.posterior_meals.empty
    assert result.latent_carb_estimates is not None
    assert not result.latent_carb_estimates.empty
    assert {
        "latent_carb_median",
        "latent_carb_lower_50",
        "latent_carb_upper_50",
        "latent_carb_lower_90",
        "latent_carb_upper_90",
        "absorption_minutes_mean",
        "post_meal_control_success",
        "confidence",
    }.issubset(result.posterior_meals.columns)
    assert not result.model_comparison.empty
    assert int(result.model_comparison["selected"].sum()) == 1
    assert "Latent Meal Fit Summary" in result.fit_summary_markdown
    assert "Latent Meal Confidence Report" in result.confidence_report_markdown
    assert "Latent Meal Model Comparison" in result.model_comparison_markdown
    assert {"latent_carbs", "carb_entry_confidence", "morning_icr"}.issubset(set(result.research_gate["parameter"]))


def test_run_latent_meal_icr_research_full_scope_caps_proxy_only_confidence():
    dataset = _build_clean_first_meal_dataset(apple=False, explicit_carbs=False)

    result = run_latent_meal_icr_research(
        dataset,
        segments=parse_therapy_segments(),
        meal_proxy_mode="strict",
        research_scope="full",
        therapy_settings=_latent_settings(),
    )

    assert not result.posterior_meals.empty
    assert float(result.posterior_meals["carb_accuracy_score"].max()) <= 0.75 + 1e-9
    assert float(result.posterior_meals["confidence"].max()) <= 0.75 + 1e-9
    assert set(result.posterior_meals["evidence_source"]) == {"proxy_only"}
    selected = result.model_comparison.loc[result.model_comparison["selected"].astype(bool)].iloc[0]
    assert float(selected["peak_delta_mae"]) == float(result.model_comparison["peak_delta_mae"].min())


def test_latent_carb_light_activity_does_not_set_exercise_context():
    dataset = _build_clean_first_meal_dataset(apple=True, explicit_carbs=False)
    frame = dataset.frame.copy()
    frame["health_activity_roll_sum_60m"] = 100.0
    dataset = replace(dataset, frame=frame)

    result = run_latent_meal_icr_research(
        dataset,
        segments=parse_therapy_segments(),
        meal_proxy_mode="strict",
        research_scope="full",
        therapy_settings=_latent_settings(),
    )

    assert not result.posterior_meals.empty
    flags = result.posterior_meals["flags"].astype(str)
    assert not flags.str.contains("exercise_context", regex=False).any()
    assert flags.str.contains("light_activity_context", regex=False).any()
    assert int(pd.to_numeric(result.posterior_meals["recent_workout_context"], errors="coerce").fillna(0).sum()) == 0
    assert int(pd.to_numeric(result.posterior_meals["high_activity_context"], errors="coerce").fillna(0).sum()) == 0


def test_extract_latent_meal_therapy_settings_from_tandem_pump_metadata(tmp_path):
    raw_dir = tmp_path / "raw" / "tconnectsync" / "2026-03-02__2026-03-31" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "pump_event_metadata.json").write_text(json.dumps(_pump_event_metadata_payload()), encoding="utf-8")

    settings = extract_latent_meal_therapy_settings_from_tandem_raw(tmp_path / "raw", segments=parse_therapy_segments())

    assert settings.insulin_action_minutes == 300
    assert settings.target_glucose_mgdl == pytest.approx(110.0)
    assert settings.max_plausible_meal_carbs == pytest.approx(175.0)
    assert settings.icr_grams_per_unit["morning"] == pytest.approx(7.0)
    assert settings.icr_grams_per_unit["afternoon"] == pytest.approx((7.0 * 48 + 6.8 * 24) / 72)
    assert settings.correction_factor_mgdl_per_unit["evening"] == pytest.approx((60.0 * 48 + 58.0 * 36) / 84)


def _with_physical_iob(dataset: AnalysisReadyHealthDataset) -> AnalysisReadyHealthDataset:
    """Recompute IOB from the bolus column so insulin bookkeeping is self-consistent."""
    from bayesian_t1dm.insulin import expand_bolus_series_to_grid

    frame = dataset.frame.copy()
    exposure = expand_bolus_series_to_grid(frame[["timestamp", "bolus_units"]])
    frame["iob_units"] = exposure["iob_units"].to_numpy()
    frame["insulin_activity_units"] = exposure["insulin_activity_units"].to_numpy()
    return replace(dataset, frame=frame)


def test_latent_carb_estimator_directionality_with_insulin_and_glucose_response():
    base = _with_physical_iob(_build_clean_first_meal_dataset(apple=False, explicit_carbs=False))
    base_median = _first_latent_carb_median(base)

    more_insulin_frame = base.frame.copy()
    bolus_mask = pd.to_numeric(more_insulin_frame["bolus_units"], errors="coerce").fillna(0.0).gt(0)
    more_insulin_frame.loc[bolus_mask, "bolus_units"] *= 1.5
    more_insulin_median = _first_latent_carb_median(_with_physical_iob(replace(base, frame=more_insulin_frame)))

    larger_response_frame = base.frame.copy()
    timestamps = pd.to_datetime(larger_response_frame["timestamp"], errors="coerce")
    for meal_time in timestamps.loc[bolus_mask]:
        response_mask = timestamps.gt(meal_time) & timestamps.le(meal_time + pd.Timedelta(minutes=180))
        minutes_after = (timestamps.loc[response_mask] - meal_time).dt.total_seconds() / 60.0
        larger_response_frame.loc[response_mask, "glucose"] += 40.0 * np.clip(minutes_after / 90.0, 0.0, 1.0)
    larger_response_median = _first_latent_carb_median(replace(base, frame=larger_response_frame))

    assert more_insulin_median > base_median
    assert larger_response_median > base_median


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


def test_latent_meal_tiered_window_registry_preserves_clean_gate_and_marks_relaxed_rows():
    windows = pd.DataFrame(
        {
            "candidate_id": ["clean", "workout", "controller", "baseline", "exploratory", "bad_cgm"],
            "included": [1, 0, 0, 0, 0, 0],
            "exclusion_reasons": [
                "",
                "recent_workout_6h",
                "recent_workout_6h|closed_loop_confounding_premeal",
                "premeal_bg_out_of_range|steep_premeal_slope",
                "postmeal_additional_bolus_120m|recent_bolus_4h",
                "poor_cgm_coverage",
            ],
        }
    )

    tiered = build_latent_meal_tiered_window_registry(windows)

    assert tiered["included"].tolist() == [1, 1, 1, 1, 1, 0]
    assert tiered["clean_included"].tolist() == [1, 0, 0, 0, 0, 0]
    assert tiered["analysis_tier"].tolist() == [
        "clean_reference",
        "workout_context_allowed",
        "controller_context_allowed",
        "baseline_drift_allowed",
        "exploratory_cgm_quality_only",
        "rejected",
    ]
    assert tiered.loc[tiered["candidate_id"].eq("bad_cgm"), "hard_exclusion_reasons"].iloc[0] == "poor_cgm_coverage"


def test_research_therapy_settings_cli_writes_artifacts(tmp_path, monkeypatch):
    dataset = _build_prepared_dataset(apple=True)
    preparation = SimpleNamespace(dataset=dataset)
    monkeypatch.setattr(cli, "_load_dataset", lambda args, session=None: SimpleNamespace(preparation=preparation))

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


def test_first_meal_window_coverage_counts_rows_missing_from_the_frame():
    from bayesian_t1dm.therapy_research import build_first_meal_clean_window_registry, build_therapy_research_frame

    dataset = _build_clean_first_meal_dataset(apple=False, explicit_carbs=False)
    complete = build_first_meal_clean_window_registry(build_therapy_research_frame(dataset, segments=parse_therapy_segments()))
    assert not complete.empty
    first = complete.sort_values("timestamp").iloc[0]
    assert first["cgm_coverage_fraction"] == pytest.approx(1.0)

    # Remove 40 minutes of rows right after the first meal, as the prepared frame does when
    # a CGM gap leaves those rows without a forecast target. No remaining row is flagged
    # missing_cgm, so only a row count can reveal the hole.
    timestamps = pd.to_datetime(dataset.frame["timestamp"])
    meal_time = pd.Timestamp(first["timestamp"])
    hole = timestamps.gt(meal_time + pd.Timedelta(minutes=20)) & timestamps.le(meal_time + pd.Timedelta(minutes=60))
    gappy = replace(dataset, frame=dataset.frame.loc[~hole].reset_index(drop=True))
    windows = build_first_meal_clean_window_registry(build_therapy_research_frame(gappy, segments=parse_therapy_segments()))
    affected = windows.loc[pd.to_datetime(windows["timestamp"]).eq(meal_time)].iloc[0]

    assert affected["cgm_coverage_fraction"] < 0.9
    assert int(affected["missing_cgm_first_90m"]) == 1
    assert {"poor_cgm_coverage", "missing_cgm_first_90m"}.issubset(set(str(affected["exclusion_reasons"]).split("|")))
    assert int(affected["included"]) == 0


def _basal_modulation_frame(days: int = 10) -> pd.DataFrame:
    timestamps = pd.date_range("2025-03-01 00:00:00", periods=days * 288, freq="5min")
    hour = timestamps.hour
    scheduled_rate = np.where(hour < 6, 0.6, np.where(hour < 12, 0.9, 0.8))
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "glucose": 110.0,
            "missing_cgm": 0,
            "bolus_units": 0.0,
            "iob_units": 0.0,
            "basal_units_delivered": scheduled_rate * 5.0 / 60.0,
        }
    )
    return frame


def test_scheduled_basal_is_recovered_despite_one_day_of_modulation():
    from bayesian_t1dm.therapy_research import _scheduled_basal_units_per_row

    frame = _basal_modulation_frame()
    timestamps = frame["timestamp"]
    truth = frame["basal_units_delivered"].copy()
    modulated = timestamps.ge("2025-03-05 08:00") & timestamps.lt("2025-03-05 10:00")
    frame.loc[modulated, "basal_units_delivered"] = 3.0 * 5.0 / 60.0

    schedule = _scheduled_basal_units_per_row(frame)

    np.testing.assert_allclose(schedule.to_numpy(), truth.to_numpy(), atol=1e-9)


def test_scheduled_basal_is_nan_without_enough_comparable_days():
    from bayesian_t1dm.therapy_research import _scheduled_basal_units_per_row

    assert _scheduled_basal_units_per_row(_basal_modulation_frame(days=3)).isna().all()


def test_window_trajectory_counts_bolus_decay_and_basal_above_schedule():
    from bayesian_t1dm.insulin import expand_bolus_series_to_grid
    from bayesian_t1dm.latent_meal_trajectory import absorbed_units_from_deliveries
    from bayesian_t1dm.therapy_research import _first_meal_window_trajectories

    frame = _basal_modulation_frame()
    meal_time = pd.Timestamp("2025-03-05 08:00")
    timestamps = frame["timestamp"]
    frame.loc[timestamps.eq(meal_time), "bolus_units"] = 4.0
    frame["iob_units"] = expand_bolus_series_to_grid(frame[["timestamp", "bolus_units"]])["iob_units"].to_numpy()
    window_metrics = pd.DataFrame({"meal_id": ["m1"], "timestamp": [meal_time]})

    bolus_only = _first_meal_window_trajectories(frame, window_metrics)["m1"]
    assert bolus_only.insulin_absorbed_units[0] == pytest.approx(0.0, abs=1e-9)
    assert 3.0 < bolus_only.insulin_absorbed_units[-1] < 4.0  # most, not all, of 4 U by 180 min

    # The controller doubles basal for the first hour after the meal on this day only.
    boosted = timestamps.ge(meal_time) & timestamps.lt(meal_time + pd.Timedelta(minutes=60))
    frame.loc[boosted, "basal_units_delivered"] *= 2.0
    with_basal = _first_meal_window_trajectories(frame, window_metrics)["m1"]

    extra_units = np.full(12, 0.9 * 5.0 / 60.0)
    expected_extra = absorbed_units_from_deliveries(np.arange(0.0, 60.0, 5.0), extra_units, with_basal.minutes)
    np.testing.assert_allclose(with_basal.insulin_absorbed_units - bolus_only.insulin_absorbed_units, expected_extra, atol=1e-6)
    assert expected_extra[-1] > 0.5


def test_window_trajectory_is_skipped_when_iob_bookkeeping_is_inconsistent():
    from bayesian_t1dm.therapy_research import _first_meal_window_trajectories

    frame = _basal_modulation_frame()
    meal_time = pd.Timestamp("2025-03-05 08:00")
    frame.loc[frame["timestamp"].eq(meal_time), "bolus_units"] = 4.0
    # IOB that rises with no bolus behind it: absorbed insulin would have to go negative.
    frame.loc[frame["timestamp"].gt(meal_time), "iob_units"] = 2.0

    assert _first_meal_window_trajectories(frame, pd.DataFrame({"meal_id": ["m1"], "timestamp": [meal_time]})) == {}


def test_settings_come_from_the_newest_window_including_tconnectsync_3_pumper_json(tmp_path):
    """Windows from tconnectsync 3.x store settings in pumper.json; the newest window wins over older 2.x snapshots."""
    old_dir = tmp_path / "raw" / "tconnectsync" / "2026-03-02__2026-03-31" / "raw"
    old_dir.mkdir(parents=True)
    (old_dir / "pump_event_metadata.json").write_text(json.dumps(_pump_event_metadata_payload()), encoding="utf-8")
    new_dir = tmp_path / "raw" / "tconnectsync" / "2026-10-19__2026-11-17" / "raw"
    new_dir.mkdir(parents=True)
    segments = [
        {"startTime": 0, "basalRate": 700, "carbRatio": 7000, "targetBg": 110, "isf": 60, "status": []},
        {"startTime": 300, "basalRate": 710, "carbRatio": 6500, "targetBg": 110, "isf": 65, "status": []},
        {"startTime": 900, "basalRate": 710, "carbRatio": 6800, "targetBg": 110, "isf": 60, "status": []},
    ]
    pumper = {
        "requested_end": "2026-11-17",
        "selected_pump_serials": ["1453346"],
        "pumper": {
            "pumps": [
                {
                    "serialNumber": "1453346",
                    "settings": {
                        "uploadedTimeStamp": "2026-11-17T14:29:57Z",
                        "details": {
                            "profiles": {
                                "activeIdp": 0,
                                "profile": [{"idp": 0, "name": "Main", "insulinDuration": 300, "maxBolus": 25000, "timeDependentSegments": segments}],
                            },
                            "globalMaxBolusSettings": {"maxBolus": 25000},
                        },
                    },
                },
                # A pump not selected for this window is ignored even if it has settings.
                {"serialNumber": "849064", "settings": {"details": {"profiles": {"activeIdp": 0, "profile": [{"idp": 0, "timeDependentSegments": [{"startTime": 0, "carbRatio": 20000, "isf": 90, "targetBg": 120}]}]}}}},
            ]
        },
    }
    (new_dir / "pumper.json").write_text(json.dumps(pumper), encoding="utf-8")

    settings = extract_latent_meal_therapy_settings_from_tandem_raw(tmp_path / "raw", segments=parse_therapy_segments())

    # Morning (06:00-11:00) falls in the 05:00 segment of the newest window: 6.5 g/U, not the old 7.0.
    assert settings.icr_grams_per_unit["morning"] == pytest.approx(6.5)
    assert settings.correction_factor_mgdl_per_unit["morning"] == pytest.approx(65.0)


def test_rows_in_an_excluded_period_never_count_as_clean_basal_time():
    from bayesian_t1dm.therapy_research import ExcludedPeriod, build_therapy_research_frame

    dataset = _synthetic_base_dataset(apple=False, explicit_carbs=True)
    stamps = pd.to_datetime(dataset.frame["timestamp"])
    day = stamps.dt.normalize().iloc[len(stamps) // 2]
    period = ExcludedPeriod(start=day, end=day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1), reason="steroid injection")

    without = build_therapy_research_frame(dataset)
    with_exclusion = build_therapy_research_frame(dataset, excluded_periods=(period,))
    inside = pd.to_datetime(with_exclusion["timestamp"]).dt.normalize().eq(day)

    assert without.loc[inside, "basal_context"].sum() > 0, "the test day must have clean basal time to exclude"
    assert with_exclusion.loc[inside, "basal_context"].sum() == 0
    assert with_exclusion.loc[inside, "context_event_excluded"].eq(1).all()
    pd.testing.assert_series_equal(with_exclusion.loc[~inside, "basal_context"], without.loc[~inside, "basal_context"])
