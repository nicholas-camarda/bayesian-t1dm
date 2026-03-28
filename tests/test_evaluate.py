from __future__ import annotations

import numpy as np

from bayesian_t1dm.evaluate import calibration_summary, walk_forward_splits


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
