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


def _fake_analysis():
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    day = pd.Timestamp("2026-03-10")
    cgm = pd.DataFrame({"timestamp": day + pd.to_timedelta([-20, -10, -5, -1], unit="h"),
                        "glucose": [100.0, np.nan, 120.0, 140.0]})
    data = SimpleNamespace(
        sleep=pd.DataFrame({"sleep_end": [day - pd.Timedelta(hours=17)], "total_sleep_hours": [7.5]}),
        workouts=pd.DataFrame({"duration_seconds": [3600.0], "end_timestamp": [day - pd.Timedelta(hours=30)],
                               "active_energy_value": [300.0]}),
        nutrition=pd.DataFrame(),
    )
    site = pd.DataFrame({"timestamp": [day - pd.Timedelta(days=3)]})
    return SimpleNamespace(merged=SimpleNamespace(data=data), cgm=cgm, site_events=site), day


def test_origin_features_ignore_row_order_and_missing_glucose():
    import numpy as np
    import pandas as pd
    import pytest

    from bayesian_t1dm.basal_model import attach_origin_features

    analysis, day = _fake_analysis()
    # Rows grouped by context, so dates are not globally sorted, as in the real training frame.
    rows = pd.DataFrame({"date": [day + pd.Timedelta(days=2), day, day + pd.Timedelta(days=1)],
                         "part": "19:00-21:00", "context": ["a", "a", "b"],
                         "sleep_share": [0.1, 0.2, 0.3], "context_event_day": [False, True, False]})
    out = attach_origin_features(rows, analysis)
    assert out.date.tolist() == rows.date.tolist()
    # The 24 hours before midnight hold readings 100 (-20h), NaN (-10h), 120 (-5h) and 140 (-1h).
    assert out.loc[1, "prior_glucose"] == pytest.approx(np.mean([100.0, 120.0, 140.0]))
    assert out.loc[1, "days_since_site"] == pytest.approx(3.0)
    assert out.loc[1, "prior_sleep_hours"] == pytest.approx(7.5)
    assert np.isnan(out.loc[0, "prior_sleep_hours"])
    assert out.loc[1, "hours_since_workout"] == pytest.approx(30.0)


def test_design_and_model_build_on_synthetic_rows():
    import numpy as np
    import pandas as pd

    from bayesian_t1dm.basal_model import build_model, design_for_training, sampler_gate
    from bayesian_t1dm.model import FitDiagnostics

    day = pd.Timestamp("2026-01-01")
    train = pd.DataFrame({"date": list(pd.date_range(day, periods=30)) * 2, "part": "p",
                          "context": ["a"] * 30 + ["b"] * 30, "command_ratio": np.r_[np.linspace(0.9, 1.1, 59), 0.0],
                          "clean_hours": 3.0, "x": np.r_[np.arange(59.0), np.nan]})
    data, center, scale, groups = design_for_training(train, ["x"])
    assert data.n_groups == 2 and data.n_weeks == 5
    assert data.design.shape == (60, 1)
    assert build_model(data).free_RVs
    good = FitDiagnostics(600, 600, 4, 0.95, 10, 22.0, 0, 5, 0, 1.002, 900.0, 800.0)
    assert sampler_gate(good)
    assert not sampler_gate(good.__class__(**{**good.__dict__, "rhat_max": 1.02}))
    assert not sampler_gate(good.__class__(**{**good.__dict__, "rhat_max": None}))
