from __future__ import annotations

import pytest

from bayesian_t1dm.features import FeatureConfig, build_feature_frame
from bayesian_t1dm.ingest import build_export_manifest, load_tandem_exports, summarize_coverage
from bayesian_t1dm.quality import assess_data_quality


def test_realistic_14d_fixture_supports_integration_pipeline(monkeypatch, tandem_realistic_fixture_dir):
    monkeypatch.setenv("TIMEZONE_NAME", "America/New_York")

    data = load_tandem_exports(tandem_realistic_fixture_dir)

    coverage = summarize_coverage(data)
    assert coverage.cgm_rows > 3000
    assert coverage.bolus_rows > 100
    assert coverage.basal_rows > 4000
    assert coverage.activity_rows == 0
    assert str(data.cgm["timestamp"].dtype) == "datetime64[ns]"

    with pytest.warns(UserWarning, match="Carb data is empty"):
        feature_frame = build_feature_frame(data, FeatureConfig(horizon_minutes=30))

    assert not feature_frame.frame.empty
    assert len(feature_frame.feature_columns) > 5

    data_quality, _ = assess_data_quality(tandem_realistic_fixture_dir, export_manifest=build_export_manifest(data))

    assert data_quality.status in {"good", "degraded"}
