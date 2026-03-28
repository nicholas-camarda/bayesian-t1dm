from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from bayesian_t1dm.features import FeatureConfig, build_feature_frame
from bayesian_t1dm.ingest import load_tandem_exports
from bayesian_t1dm.model import ScenarioForecast
from bayesian_t1dm.recommend import recommend_setting_changes


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
