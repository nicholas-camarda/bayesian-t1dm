from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bayesian_t1dm.evaluate import CalibrationSummary, FoldResult, WalkForwardReport
from bayesian_t1dm.features import FeatureConfig, build_feature_frame
from bayesian_t1dm.ingest import load_tandem_exports
from bayesian_t1dm.model import FitDiagnostics, ScenarioForecast
from bayesian_t1dm.quality import DataQualitySummary
from bayesian_t1dm.recommend import RecommendationPolicy, Scenario, build_recommendation_policy, recommend_setting_changes


class DummyModel:
    def scenario_forecasts(self, fit, scenarios):
        outputs = []
        for name, frame in scenarios:
            if name == "current":
                loss = 25.0
            elif name == "basal_minus_10":
                loss = 16.0
            elif name == "bolus_plus_10":
                loss = 23.0
            else:
                loss = 28.0
            outputs.append(
                ScenarioForecast(
                    scenario_name=name,
                    mean=np.array([110.0, 112.0]),
                    lower=np.array([100.0, 102.0]),
                    upper=np.array([120.0, 122.0]),
                    expected_loss=loss,
                )
            )
        return outputs


def _healthy_walk_forward(*, coverage: float = 0.8, model_mae: float = 10.0, persistence_mae: float = 15.0) -> WalkForwardReport:
    fold = FoldResult(
        fold=1,
        n_train=500,
        n_test=250,
        model_mae=model_mae,
        model_rmse=model_mae + 1.0,
        model_coverage=coverage,
        persistence_mae=persistence_mae,
        fit_diagnostics=None,
    )
    fold2 = FoldResult(
        fold=2,
        n_train=750,
        n_test=250,
        model_mae=model_mae,
        model_rmse=model_mae + 1.0,
        model_coverage=coverage,
        persistence_mae=persistence_mae,
        fit_diagnostics=None,
    )
    return WalkForwardReport(
        folds=[fold, fold2],
        aggregate=CalibrationSummary(
            mae=model_mae,
            rmse=model_mae + 1.0,
            coverage=coverage,
            interval_width=20.0,
        ),
        aggregate_persistence_mae=persistence_mae,
    )


def _healthy_fit_diagnostics(**overrides) -> FitDiagnostics:
    base = dict(
        draws=100,
        tune=100,
        chains=2,
        target_accept=0.9,
        max_treedepth=10,
        wall_time_seconds=1.0,
        divergences=0,
        max_tree_depth_observed=7,
        max_tree_depth_hits=0,
        rhat_max=1.0,
        ess_bulk_min=300.0,
        ess_tail_min=300.0,
    )
    base.update(overrides)
    return FitDiagnostics(**base)


def _good_data_quality(**overrides) -> DataQualitySummary:
    base = dict(
        status="good",
        contributing_window_ids=["window-a"],
        incomplete_window_count=0,
        reason_counts={},
        evaluation_touches_incomplete_windows=False,
    )
    base.update(overrides)
    return DataQualitySummary(**base)


def test_recommend_setting_changes_ranks_only_useful_actions(tandem_fixture_dir):
    data = load_tandem_exports(tandem_fixture_dir)
    feature_frame = build_feature_frame(data, FeatureConfig(horizon_minutes=30))
    model = DummyModel()
    fit = SimpleNamespace(diagnostics=_healthy_fit_diagnostics())

    recommendations, forecasts, policy = recommend_setting_changes(
        model,
        fit,
        feature_frame.frame,
        walk_forward=_healthy_walk_forward(),
        data_quality=_good_data_quality(),
        min_expected_gain_mgdl=5.0,
    )

    assert policy.status == "generated"
    assert forecasts[0].scenario_name == "current"
    assert recommendations[0].setting == "basal"
    assert recommendations[0].change_percent == 10.0
    assert recommendations[0].expected_gain_mgdl > 5.0


class RecordingModel:
    def __init__(self) -> None:
        self.frames_by_name: dict[str, pd.DataFrame] = {}

    def scenario_forecasts(self, fit, scenarios):
        outputs = []
        for name, frame in scenarios:
            self.frames_by_name[name] = frame.copy()
            outputs.append(
                ScenarioForecast(
                    scenario_name=name,
                    mean=np.array([110.0, 112.0]),
                    lower=np.array([100.0, 102.0]),
                    upper=np.array([120.0, 122.0]),
                    expected_loss=10.0 if name != "current" else 20.0,
                )
            )
        return outputs


def test_recommend_setting_changes_recomputes_derived_scenario_features():
    feature_frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-06-01 00:00:00", "2023-06-01 00:05:00"]),
            "bolus_units": [2.0, 0.0],
            "insulin_activity_units": [0.8, 0.4],
            "iob_units": [1.5, 1.0],
            "basal_units_per_hour": [1.0, 1.0],
            "basal_units_delivered": [0.0833333333, 0.0833333333],
            "carb_grams": [20.0, 25.0],
            "carb_roll_sum_60m": [20.0, 45.0],
            "carb_bolus_interaction_60m": [40.0, 0.0],
            "carb_iob_interaction_60m": [30.0, 45.0],
            "target_glucose": [110.0, 112.0],
        }
    )
    model = RecordingModel()
    fit = SimpleNamespace(diagnostics=_healthy_fit_diagnostics())

    recommendations, _, policy = recommend_setting_changes(
        model,
        fit,
        feature_frame,
        scenarios=[Scenario("bolus_plus_10", bolus_multiplier=1.1)],
        walk_forward=_healthy_walk_forward(),
        data_quality=_good_data_quality(),
        min_expected_gain_mgdl=5.0,
    )

    assert policy.status == "generated"
    scenario_frame = model.frames_by_name["bolus_plus_10"]
    assert scenario_frame["bolus_units"].iloc[0] == pytest.approx(2.2)
    assert scenario_frame["carb_bolus_interaction_60m"].iloc[0] == pytest.approx(
        scenario_frame["carb_roll_sum_60m"].iloc[0] * scenario_frame["bolus_units"].iloc[0]
    )
    assert scenario_frame["carb_iob_interaction_60m"].iloc[0] == pytest.approx(
        scenario_frame["carb_roll_sum_60m"].iloc[0] * scenario_frame["iob_units"].iloc[0]
    )
    assert recommendations


def test_build_recommendation_policy_suppresses_on_poor_coverage():
    policy, allowed = build_recommendation_policy(
        walk_forward=_healthy_walk_forward(coverage=0.5),
        fit_diagnostics=_healthy_fit_diagnostics(),
        data_quality=_good_data_quality(),
        carbs_present=True,
        activity_present=True,
    )

    assert policy.status == "suppressed"
    assert "coverage_out_of_range" in policy.reasons
    assert policy.validation_passed is False
    assert allowed == {"basal", "bolus", "I/C ratio"}


def test_build_recommendation_policy_suppresses_when_mae_does_not_clear_persistence_margin():
    policy, _ = build_recommendation_policy(
        walk_forward=_healthy_walk_forward(model_mae=14.0, persistence_mae=15.0),
        fit_diagnostics=_healthy_fit_diagnostics(),
        data_quality=_good_data_quality(),
        carbs_present=True,
        activity_present=True,
    )

    assert policy.status == "suppressed"
    assert "mae_not_meaningfully_better_than_persistence" in policy.reasons


def test_build_recommendation_policy_suppresses_on_bad_sampler_health():
    policy, _ = build_recommendation_policy(
        walk_forward=_healthy_walk_forward(),
        fit_diagnostics=_healthy_fit_diagnostics(chains=1, divergences=3, max_tree_depth_hits=1),
        data_quality=_good_data_quality(),
        carbs_present=True,
        activity_present=True,
    )

    assert policy.status == "suppressed"
    assert "sampler_chains_lt_2" in policy.reasons
    assert "sampler_divergences" in policy.reasons
    assert "sampler_max_treedepth_hits" in policy.reasons


def test_build_recommendation_policy_suppresses_bolus_and_icr_when_carbs_missing():
    policy, allowed = build_recommendation_policy(
        walk_forward=_healthy_walk_forward(),
        fit_diagnostics=_healthy_fit_diagnostics(),
        data_quality=_good_data_quality(),
        carbs_present=False,
        activity_present=True,
    )

    assert policy.status == "generated"
    assert "missing_carbs" in policy.reasons
    assert allowed == {"basal"}


def test_recommend_setting_changes_marks_basal_low_confidence_when_signal_is_low():
    feature_frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-06-01 00:00:00", "2023-06-01 00:05:00"]),
            "bolus_units": [2.0, 0.0],
            "insulin_activity_units": [0.8, 0.4],
            "iob_units": [1.5, 1.0],
            "basal_units_per_hour": [1.0, 1.0],
            "basal_units_delivered": [0.0833333333, 0.0833333333],
            "target_glucose": [110.0, 112.0],
        }
    )
    model = RecordingModel()
    fit = SimpleNamespace(diagnostics=_healthy_fit_diagnostics())

    recommendations, _, policy = recommend_setting_changes(
        model,
        fit,
        feature_frame,
        scenarios=[Scenario("basal_minus_10", basal_multiplier=0.9), Scenario("bolus_plus_10", bolus_multiplier=1.1)],
        walk_forward=_healthy_walk_forward(),
        data_quality=_good_data_quality(),
        carbs_present=False,
        activity_present=False,
        min_expected_gain_mgdl=5.0,
    )

    assert policy.status == "generated"
    assert all(rec.setting == "basal" for rec in recommendations)
    assert all(rec.confidence == "low" for rec in recommendations)
    assert all("low_signal" in rec.flags for rec in recommendations)


def test_build_recommendation_policy_suppresses_on_incomplete_data():
    policy, _ = build_recommendation_policy(
        walk_forward=_healthy_walk_forward(),
        fit_diagnostics=_healthy_fit_diagnostics(),
        data_quality=_good_data_quality(status="degraded", incomplete_window_count=1, reason_counts={"ends_early": 1}),
        carbs_present=True,
        activity_present=True,
    )

    assert policy.status == "suppressed"
    assert "data_incomplete" in policy.reasons
