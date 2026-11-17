from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

arviz = pytest.importorskip("arviz")
import xarray as xr

from bayesian_t1dm.features import FeatureConfig, build_feature_frame
from bayesian_t1dm.ingest import load_tandem_exports
from bayesian_t1dm.model import BayesianGlucoseModel, ModelFit, extract_fit_diagnostics


def test_predict_uses_posterior_samples(tandem_fixture_dir):
    data = load_tandem_exports(tandem_fixture_dir)
    frame = build_feature_frame(data, FeatureConfig(horizon_minutes=30))
    features = frame.frame.iloc[:3].copy()
    feature_columns = list(frame.feature_columns[:2])
    feature_means = pd.Series({column: 0.0 for column in feature_columns})
    feature_scales = pd.Series({column: 1.0 for column in feature_columns})

    posterior = SimpleNamespace(
        posterior=xr.Dataset(
            {
                "intercept": (("chain", "draw"), np.array([[0.0, 0.2]])),
                "beta": (("chain", "draw", "beta_dim_0"), np.array([[[0.1, -0.1], [0.2, -0.2]]])),
                "latent": (("chain", "draw", "latent_dim_0"), np.array([[[0.0, 0.1, 0.2], [0.1, 0.0, -0.1]]])),
                "rho": (("chain", "draw"), np.array([[0.9, 0.8]])),
                "sigma_state": (("chain", "draw"), np.array([[0.1, 0.1]])),
                "sigma_obs": (("chain", "draw"), np.array([[1.0, 1.0]])),
            }
        )
    )
    fit = ModelFit(
        posterior=posterior,
        feature_columns=feature_columns,
        feature_means=feature_means,
        feature_scales=feature_scales,
        target_mean=100.0,
        target_scale=10.0,
        horizon_minutes=30,
        model=None,
    )

    model = BayesianGlucoseModel(draws=10, tune=10, chains=1)
    prediction = model.predict(fit, features)

    assert list(prediction.columns) == ["mean", "lower", "upper"]
    assert len(prediction) == len(features)
    assert prediction["upper"].ge(prediction["lower"]).all()


def _make_minimal_fit(*, sigma_obs: float, sigma_state: float = 0.0) -> ModelFit:
    feature_columns = ["x"]
    feature_means = pd.Series({"x": 0.0})
    feature_scales = pd.Series({"x": 1.0})
    posterior = SimpleNamespace(
        posterior=xr.Dataset(
            {
                "intercept": (("chain", "draw"), np.array([[0.0, 0.0, 0.0]])),
                "beta": (("chain", "draw", "beta_dim_0"), np.array([[[0.0], [0.0], [0.0]]])),
                "latent": (("chain", "draw", "latent_dim_0"), np.array([[[0.0], [0.0], [0.0]]])),
                "rho": (("chain", "draw"), np.array([[0.0, 0.0, 0.0]])),
                "sigma_state": (("chain", "draw"), np.array([[sigma_state, sigma_state, sigma_state]])),
                "sigma_obs": (("chain", "draw"), np.array([[sigma_obs, sigma_obs, sigma_obs]])),
            }
        )
    )
    return ModelFit(
        posterior=posterior,
        feature_columns=feature_columns,
        feature_means=feature_means,
        feature_scales=feature_scales,
        target_mean=100.0,
        target_scale=10.0,
        horizon_minutes=30,
        model=None,
    )


def test_predict_intervals_widen_with_sigma_obs():
    features = pd.DataFrame({"x": np.zeros(5, dtype=float)})
    model = BayesianGlucoseModel(draws=10, tune=10, chains=1, random_seed=123)

    narrow = model.predict(_make_minimal_fit(sigma_obs=0.0), features)
    wide = model.predict(_make_minimal_fit(sigma_obs=1.0), features)

    assert float((narrow["upper"] - narrow["lower"]).mean()) == 0.0
    assert float((wide["upper"] - wide["lower"]).mean()) > 0.0


def test_predict_intervals_widen_with_state_innovation():
    features = pd.DataFrame({"x": np.zeros(5, dtype=float)})
    model = BayesianGlucoseModel(draws=10, tune=10, chains=1, random_seed=123)

    narrow = model.predict(_make_minimal_fit(sigma_obs=0.0, sigma_state=0.0), features)
    wide = model.predict(_make_minimal_fit(sigma_obs=0.0, sigma_state=1.0), features)

    assert float((narrow["upper"] - narrow["lower"]).mean()) == 0.0
    assert float((wide["upper"] - wide["lower"]).mean()) > 0.0


def test_scenario_forecasts_requires_target_glucose_column():
    model = BayesianGlucoseModel(draws=10, tune=10, chains=1)

    def _predict(_fit, frame: pd.DataFrame) -> pd.DataFrame:
        n = len(frame)
        return pd.DataFrame({"mean": np.zeros(n), "lower": np.zeros(n), "upper": np.zeros(n)}, index=frame.index)

    model.predict = _predict  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="missing 'target_glucose'"):
        model.scenario_forecasts(None, [("baseline", pd.DataFrame({"x": [1.0]}))])


def test_extract_fit_diagnostics_reads_sampler_health():
    posterior = SimpleNamespace(
        posterior=xr.Dataset(
            {
                "intercept": (("chain", "draw"), np.array([[0.0, 0.1], [0.2, 0.3]])),
                "beta": (("chain", "draw", "beta_dim_0"), np.array([[[0.1], [0.1]], [[0.1], [0.1]]])),
            }
        ),
        sample_stats=xr.Dataset(
            {
                "diverging": (("chain", "draw"), np.array([[0, 1], [0, 0]], dtype=int)),
                "tree_depth": (("chain", "draw"), np.array([[4, 10], [6, 8]], dtype=int)),
                "reached_max_treedepth": (("chain", "draw"), np.array([[0, 1], [0, 0]], dtype=int)),
            }
        ),
    )

    diagnostics = extract_fit_diagnostics(
        posterior,
        draws=20,
        tune=30,
        chains=2,
        target_accept=0.9,
        max_treedepth=10,
        wall_time_seconds=1.25,
    )

    assert diagnostics.draws == 20
    assert diagnostics.tune == 30
    assert diagnostics.chains == 2
    assert diagnostics.divergences == 1
    assert diagnostics.max_tree_depth_observed == 10
    assert diagnostics.max_tree_depth_hits == 1


def _slope_uncertain_fit() -> ModelFit:
    # Four posterior draws that disagree about the sign of the effect of "x":
    # three say raising x lowers glucose, one says it raises glucose.
    betas = np.array([[[-1.0], [-1.0], [-1.0], [1.0]]])
    n = betas.shape[1]
    posterior = SimpleNamespace(
        posterior=xr.Dataset(
            {
                "intercept": (("chain", "draw"), np.full((1, n), 2.0)),
                "beta": (("chain", "draw", "beta_dim_0"), betas),
                "latent": (("chain", "draw", "latent_dim_0"), np.zeros((1, n, 1))),
                "rho": (("chain", "draw"), np.zeros((1, n))),
                "sigma_state": (("chain", "draw"), np.zeros((1, n))),
                "sigma_obs": (("chain", "draw"), np.zeros((1, n))),
            }
        )
    )
    return ModelFit(
        posterior=posterior,
        feature_columns=["x"],
        feature_means=pd.Series({"x": 0.0}),
        feature_scales=pd.Series({"x": 1.0}),
        target_mean=100.0,
        target_scale=10.0,
        horizon_minutes=30,
        model=None,
    )


def test_scenario_forecasts_expose_paired_target_loss_draws():
    model = BayesianGlucoseModel(draws=10, tune=10, chains=1, random_seed=3)
    fit = _slope_uncertain_fit()
    baseline = pd.DataFrame({"x": np.zeros(6), "target_glucose": np.full(6, 110.0)})
    raised = baseline.assign(x=1.0)

    current, candidate = model.scenario_forecasts(fit, [("current", baseline), ("raised", raised)], target_bg=110.0)

    # Baseline predicts 120 in every draw: 10 mg/dL from target.
    assert current.target_loss_draws == pytest.approx([10.0] * 4)
    # Raising x gives 110 in three draws (on target) and 130 in one (worse).
    assert candidate.target_loss_draws == pytest.approx([0.0, 0.0, 0.0, 20.0])


def test_scenario_forecasts_omit_draw_losses_without_target():
    model = BayesianGlucoseModel(draws=10, tune=10, chains=1)
    frame = pd.DataFrame({"x": np.zeros(3), "target_glucose": np.full(3, 110.0)})

    (forecast,) = model.scenario_forecasts(_slope_uncertain_fit(), [("current", frame)])

    assert forecast.target_loss_draws is None
