from __future__ import annotations

from bayesian_t1dm.features import FeatureConfig, build_feature_frame
from bayesian_t1dm.ingest import load_tandem_exports, summarize_coverage


def test_fixture_exports_load_and_build_features(tandem_fixture_dir):
    data = load_tandem_exports(tandem_fixture_dir)
    coverage = summarize_coverage(data)
    frame = build_feature_frame(data, FeatureConfig(horizon_minutes=30))

    assert coverage.cgm_rows == 7
    assert coverage.manifest_rows >= 4
    assert coverage.is_complete
    assert not frame.frame.empty
