from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bayesian_t1dm.evaluate import calibration_summary, run_walk_forward, walk_forward_splits
from bayesian_t1dm.features import FeatureConfig, FeatureFrame


def test_walk_forward_splits_progresses_forward():
    splits = list(walk_forward_splits(20, initial_train_size=8, test_size=4, step_size=4))

    assert splits[0].train_end == 8
    assert splits[0].test_start == 8
    assert splits[1].train_end == 12
    assert splits[1].test_end == 16


def test_calibration_summary_reports_coverage():
    summary = calibration_summary(
        np.array([100.0, 110.0, 120.0]),
        np.array([102.0, 108.0, 121.0]),
        np.array([95.0, 100.0, 115.0]),
        np.array([105.0, 115.0, 125.0]),
    )

    assert summary.coverage == 1.0
    assert summary.mae > 0
    assert summary.interval_width > 0


class _PerfectModel:
    def fit(self, feature_frame):
        return object()

    def predict(self, fit, frame: pd.DataFrame) -> pd.DataFrame:
        y = frame["target_glucose"].to_numpy(dtype=float)
        return pd.DataFrame({"mean": y, "lower": y - 1.0, "upper": y + 1.0}, index=frame.index)


def test_run_walk_forward_reports_folds_and_persistence():
    n_rows = 50
    horizon_steps = 6
    glucose = np.arange(n_rows, dtype=float)
    frame = pd.DataFrame(
        {
            "glucose": glucose,
            "target_glucose": glucose + horizon_steps,
            "x": np.ones(n_rows, dtype=float),
        }
    )
    ff = FeatureFrame(
        frame=frame,
        feature_columns=["x"],
        target_column="target_glucose",
        horizon_minutes=30,
        config=FeatureConfig(freq="5min", horizon_minutes=30),
    )
    report = run_walk_forward(ff, _PerfectModel(), n_folds=4, min_test_rows=1)

    assert report is not None
    assert report.n_folds == 4
    assert report.aggregate.mae == 0.0
    assert report.aggregate.coverage == 1.0
    assert report.aggregate_persistence_mae == pytest.approx(float(horizon_steps))


def test_run_walk_forward_returns_none_when_too_small():
    n_rows = 4
    frame = pd.DataFrame({"glucose": np.arange(n_rows, dtype=float), "target_glucose": np.arange(n_rows, dtype=float) + 6, "x": 1.0})
    ff = FeatureFrame(
        frame=frame,
        feature_columns=["x"],
        target_column="target_glucose",
        horizon_minutes=30,
        config=FeatureConfig(freq="5min", horizon_minutes=30),
    )
    with pytest.warns(UserWarning, match="Insufficient data"):
        report = run_walk_forward(ff, _PerfectModel(), n_folds=4, min_test_rows=1)
    assert report is None
