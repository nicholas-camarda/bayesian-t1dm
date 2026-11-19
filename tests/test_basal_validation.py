import pandas as pd

from bayesian_t1dm.basal_validation import baseline_distribution, score_distribution


def test_baseline_excludes_origin_and_future_and_scores_zero():
    dates = pd.date_range("2026-01-01", periods=5)
    rows = pd.DataFrame({"date": dates, "clean_hours": 1.0, "command_ratio": [0.0, 0.8, 1.2, 8, 8]})
    forecast = baseline_distribution(rows, dates[3], decay=True)
    assert forecast == baseline_distribution(rows.iloc[:3], dates[3], decay=True)
    assert forecast["training_days"] == 3
    assert baseline_distribution(rows, dates[3], decay=False)["training_days"] == 3
    scored = score_distribution(forecast, 0)
    assert scored["lower_90"] == 0
    assert scored["hit_below"] and not scored["hit_above"]


def test_bayesian_scoring_keeps_the_point_mass_and_thresholds():
    import numpy as np
    import pytest

    from bayesian_t1dm.basal_model import score_prediction

    rng = np.random.default_rng(9)
    p_zero = np.full(2000, 0.2)
    score = score_prediction(np.zeros(2000), np.full(2000, 0.01), p_zero, 0.0, rng=rng)
    assert score["lower_90"] == 0
    assert score["p_below"] == pytest.approx(0.2)
    assert score["log_score"] == pytest.approx(np.log(0.2))
