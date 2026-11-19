"""Tests for the hour-day outcomes model on synthetic hour-day tables (no clinical data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bayesian_t1dm import outcomes_model as om


def synthetic_table(days: int = 60, *, seed: int = 3, beta_icr: float = 0.0) -> pd.DataFrame:
    """A complete hour-day table; the in-range share depends on clock hour and (when ``beta_icr`` is set) on I:C."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    rows = []
    for date in dates:
        icr_day = 10.0 * np.exp(rng.normal(0, 0.15))
        day_shift = int(rng.integers(0, 3))
        for hour in range(24):
            icr = icr_day * (1.0 + 0.1 * (hour % 3))
            eta = 0.6 + 0.3 * np.sin(2 * np.pi * hour / 24) + beta_icr * np.log(icr / 10.0) / 0.15
            p = 1 / (1 + np.exp(-eta))
            n = 12
            in_range = int(rng.binomial(n, p))
            rows.append({
                "date": date, "hour": hour, "n_valid": n, "n_in_range": in_range,
                "n_below_70": int(rng.binomial(n - in_range, 0.3)), "n_below_54": 0,
                "icr": icr, "cf": 45.0 + (hour % 4) + 3 * day_shift, "basal": 0.8 + 0.1 * (hour % 5) + 0.02 * rng.normal(), "target": 110.0 + 5 * (hour % 2) + 5 * day_shift,
                "settings_known_slots": 12, "excluded_day": False,
                "carbs_prev3h_g": float(rng.integers(0, 60)), "exercise_share": float(rng.random() < 0.1),
                "sleep_mode_share": float(hour < 6), "site_age_days": float(rng.integers(0, 4)),
            })
    return pd.DataFrame(rows)


def test_usable_rows_drop_unknown_settings_excluded_days_and_empty_hours() -> None:
    table = synthetic_table(days=4)
    table.loc[0, "settings_known_slots"] = 7
    table.loc[1, "n_valid"] = 0
    table.loc[table["date"] == table["date"].iloc[-1], "excluded_day"] = True
    flow = om.design_flow(table)
    assert flow["rows"].is_monotonic_decreasing
    assert flow["rows"].iloc[0] == len(table)
    usable = om.usable_rows(table)
    assert flow["rows"].iloc[-1] == int(usable.sum()) == len(table) - 2 - 24
    assert not usable.iloc[0] and not usable.iloc[1]


def test_design_standardizes_on_training_rows_and_reuses_them_for_new_rows() -> None:
    table = synthetic_table(days=30)
    train = om.build_design(table, outcome="tir")
    assert np.allclose(train.x.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(train.x.std(axis=0), 1, atol=1e-9)
    later = table.assign(icr=table["icr"] * 1.5)
    held_out = om.build_design(later, outcome="tir", center=train.center, scale=train.scale)
    assert held_out.x[:, 0].mean() > 1.0
    assert train.center.equals(held_out.center)


def test_design_indexes_hours_and_days() -> None:
    design = om.build_design(synthetic_table(days=15), outcome="tir")
    assert design.rows == 15 * 24
    assert set(design.hour) == set(range(24))
    assert design.day.max() == 14 and (design.y <= design.n).all()
    assert om.build_design(synthetic_table(days=15), outcome="below70").y.sum() <= design.n.sum()


def test_unknown_outcome_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown outcome"):
        om.build_design(synthetic_table(days=3), outcome="mean")


def test_collinearity_flags_a_duplicated_covariate() -> None:
    table = synthetic_table(days=30)
    table["cf"] = table["icr"] * 4.0
    vif = om.collinearity(om.build_design(table, outcome="tir")).set_index("covariate")
    assert vif.loc["log_icr", "vif_raw"] > om.VIF_STOP
    assert vif.loc["log_icr", "status"] == "concern"
    assert vif.loc["exercise_share", "status"] == "no concern"
    assert vif.loc["sleep_mode_share", "status"] == "not assessed"


def test_collinearity_marks_a_covariate_constant_within_hours_as_not_assessed() -> None:
    table = synthetic_table(days=30).assign(cf=lambda frame: 45.0 + frame["hour"] % 4)
    vif = om.collinearity(om.build_design(table, outcome="tir")).set_index("covariate")
    assert vif.loc["log_cf", "status"] == "not assessed"


def test_thin_days_keeps_whole_days() -> None:
    table = synthetic_table(days=20)
    thinned = om.thin_days(table, 5)
    assert sorted(thinned["date"].unique()) == list(pd.date_range("2025-01-01", periods=20, freq="D")[::5])
    assert len(thinned) == 4 * 24
    with pytest.raises(ValueError):
        om.thin_days(table, 0)


def test_model_logp_is_finite_and_prior_predictive_respects_counts() -> None:
    pytest.importorskip("pymc")
    design = om.build_design(synthetic_table(days=10), outcome="tir")
    model = om.build_model(design)
    assert np.isfinite(model.compile_logp()(model.initial_point()))
    check = om.prior_predictive(design, samples=20)
    assert 0 <= check["overall_share_5th"] <= check["overall_share_95th"] <= 1


def test_fit_recovers_an_icr_coefficient() -> None:
    pytest.importorskip("pymc")
    from bayesian_t1dm.sampler_health import stack_draws

    design = om.build_design(synthetic_table(days=60, beta_icr=-0.5), outcome="tir")
    idata, diagnostics, timing = om.fit(design, draws=400, tune=400, chains=2)
    beta = stack_draws(idata, "beta")[:, 0]
    assert -0.8 < beta.mean() < -0.3
    assert timing["sample_seconds"] > 0
    assert diagnostics.chains == 2
