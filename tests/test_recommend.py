from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bayesian_t1dm.features import FeatureConfig, build_feature_frame
from bayesian_t1dm.ingest import load_tandem_exports
from bayesian_t1dm.model import ScenarioForecast
from bayesian_t1dm.recommend import Scenario, recommend_setting_changes


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


def test_recommend_setting_changes_ranks_only_useful_actions(tandem_fixture_dir):
    data = load_tandem_exports(tandem_fixture_dir)
    feature_frame = build_feature_frame(data, FeatureConfig(horizon_minutes=30))
    model = DummyModel()
    fit = SimpleNamespace()

    recommendations, forecasts = recommend_setting_changes(model, fit, feature_frame.frame, min_expected_gain_mgdl=5.0)

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
    fit = SimpleNamespace()

    recommendations, _ = recommend_setting_changes(model, fit, feature_frame, scenarios=[Scenario("bolus_plus_10", bolus_multiplier=1.1)], min_expected_gain_mgdl=5.0)

    scenario_frame = model.frames_by_name["bolus_plus_10"]
    assert scenario_frame["bolus_units"].iloc[0] == pytest.approx(2.2)
    assert scenario_frame["carb_bolus_interaction_60m"].iloc[0] == pytest.approx(
        scenario_frame["carb_roll_sum_60m"].iloc[0] * scenario_frame["bolus_units"].iloc[0]
    )
    assert scenario_frame["carb_iob_interaction_60m"].iloc[0] == pytest.approx(
        scenario_frame["carb_roll_sum_60m"].iloc[0] * scenario_frame["iob_units"].iloc[0]
    )
    assert recommendations
