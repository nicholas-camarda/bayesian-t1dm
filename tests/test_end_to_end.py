from __future__ import annotations

import pytest

pm = pytest.importorskip("pymc")

from bayesian_t1dm.evaluate import CalibrationSummary, WalkForwardReport
from bayesian_t1dm.features import FeatureConfig, build_feature_frame
from bayesian_t1dm.ingest import load_tandem_exports, summarize_coverage
from bayesian_t1dm.model import BayesianGlucoseModel
from bayesian_t1dm.recommend import recommend_setting_changes


def test_end_to_end_pipeline_smoke_runs(tandem_fixture_dir):
    data = load_tandem_exports(tandem_fixture_dir)
    coverage = summarize_coverage(data)
    frame = build_feature_frame(data, FeatureConfig(horizon_minutes=30))
    model = BayesianGlucoseModel(draws=20, tune=20, chains=1, random_seed=3)
    fit = model.fit(frame)
    predictions = model.predict(fit, frame.frame)
    walk_forward = WalkForwardReport(
        folds=[],
        aggregate=CalibrationSummary(mae=1.0, rmse=1.0, coverage=0.8, interval_width=20.0),
        aggregate_persistence_mae=2.0,
    )
    recommendations, forecasts, policy = recommend_setting_changes(
        model,
        fit,
        frame.frame,
        walk_forward=walk_forward,
        min_expected_gain_mgdl=0.0,
    )

    assert coverage.cgm_rows == 7
    assert coverage.manifest_rows >= 4
    assert coverage.is_complete
    assert not predictions.empty
    assert policy.status in {"generated", "suppressed"}
    assert isinstance(forecasts, list)
    assert isinstance(recommendations, list)
