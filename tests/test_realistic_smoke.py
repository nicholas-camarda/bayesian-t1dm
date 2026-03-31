from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bayesian_t1dm.cli import main
from bayesian_t1dm.evaluate import run_walk_forward
from bayesian_t1dm.features import FeatureConfig, FeatureFrame, build_feature_frame
from bayesian_t1dm.ingest import build_export_manifest, load_tandem_exports, summarize_coverage
from bayesian_t1dm.quality import assess_data_quality
from bayesian_t1dm.recommend import RecommendationPolicy
from bayesian_t1dm.report import build_run_summary, write_json_report
from bayesian_t1dm.review import write_run_review_html


class _PerfectFutureModel:
    def fit(self, feature_frame: FeatureFrame) -> object:
        return object()

    def predict(self, fit: object, frame: pd.DataFrame) -> pd.DataFrame:
        target = frame["target_glucose"].to_numpy(dtype=float)
        return pd.DataFrame(
            {
                "mean": target,
                "lower": target - 15.0,
                "upper": target + 15.0,
            },
            index=frame.index,
        )


def test_realistic_14d_fixture_supports_integration_pipeline(monkeypatch, tandem_realistic_fixture_dir, tmp_path):
    monkeypatch.setenv("TIMEZONE_NAME", "America/New_York")

    with pytest.warns(UserWarning, match="Mixed timezone-aware"):
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

    walk_forward = run_walk_forward(feature_frame, _PerfectFutureModel(), n_folds=2, min_test_rows=200)
    assert walk_forward is not None
    assert walk_forward.n_folds == 2
    data_quality, _ = assess_data_quality(tandem_realistic_fixture_dir, export_manifest=build_export_manifest(data))

    summary = build_run_summary(
        coverage=coverage,
        walk_forward=walk_forward,
        recommendations=[],
        fit_diagnostics=None,
        data_quality=data_quality,
        recommendation_policy=RecommendationPolicy(
            status="skipped",
            reasons=["skipped_by_flag"],
            validation_passed=False,
            sampler_passed=False,
            signal_passed=False,
        ),
        review_artifacts={"run_review_html": str(tmp_path / "run_review.html")},
    )
    out_path = write_json_report(summary, tmp_path / "run_summary.json")
    write_run_review_html(summary, tmp_path / "run_review.html")
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert payload["coverage"]["cgm_rows"] > 3000
    assert len(payload["walk_forward"]["folds"]) == 2
    assert payload["walk_forward"]["aggregate"]["coverage"] >= 0.9
    assert isinstance(payload["walk_forward"]["aggregate_persistence_mae"], float)
    assert payload["data_quality"]["status"] in {"good", "degraded"}
    assert "fit_diagnostics" in payload
    assert payload["recommendation_policy"]["status"] == "skipped"
    assert payload["recommendations"] == []
    assert (tmp_path / "run_review.html").exists()


def test_real_data_smoke_cli_run_when_raw_path_is_provided(tmp_path):
    raw_dir = os.getenv("BAYESIAN_T1DM_REAL_SMOKE_RAW")
    if not raw_dir:
        pytest.skip("Set BAYESIAN_T1DM_REAL_SMOKE_RAW to run the local real-data smoke test.")
    pytest.importorskip("pymc")

    report_path = tmp_path / "real_smoke_run_summary.md"
    manifest_path = tmp_path / "real_smoke_manifest.csv"

    exit_code = main(
        [
            "--root",
            str(tmp_path),
            "run",
            "--raw",
            raw_dir,
            "--report",
            str(report_path),
            "--manifest",
            str(manifest_path),
            "--eval-folds",
            "1",
            "--draws",
            "10",
            "--tune",
            "10",
            "--chains",
            "1",
            "--skip-recommendations",
        ]
    )

    assert exit_code == 0
    payload = json.loads(report_path.with_suffix(".json").read_text(encoding="utf-8"))
    aggregate = payload["walk_forward"]["aggregate"]

    assert np.isfinite(float(aggregate["mae"]))
    assert np.isfinite(float(aggregate["coverage"]))
    assert np.isfinite(float(payload["walk_forward"]["aggregate_persistence_mae"]))
    assert payload["fit_diagnostics"] is None
    assert payload["data_quality"]["status"] in {"good", "degraded", "broken"}
    assert payload["recommendation_policy"]["status"] == "skipped"
