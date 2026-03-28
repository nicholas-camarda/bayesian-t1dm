from __future__ import annotations

from bayesian_t1dm.features import FeatureConfig, build_feature_frame
from bayesian_t1dm.ingest import load_tandem_exports


def test_build_feature_frame_creates_lags_and_target(tandem_fixture_dir):
    data = load_tandem_exports(tandem_fixture_dir)
    frame = build_feature_frame(data, FeatureConfig(horizon_minutes=30))

    assert frame.target_column == "target_glucose"
    assert "glucose_lag_30m" in frame.feature_columns
    assert "iob_roll_sum_60m" in frame.feature_columns
    assert frame.frame["timestamp"].is_monotonic_increasing
    assert frame.frame["target_glucose"].notna().all()
    assert frame.frame["target_delta"].notna().all()
