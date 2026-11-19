from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

az = pytest.importorskip("arviz")
import xarray as xr

from bayesian_t1dm.features import FeatureConfig, FeatureFrame, build_feature_frame
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


def _make_minimal_fit(*, sigma_obs: float) -> ModelFit:
    feature_columns = ["x"]
    feature_means = pd.Series({"x": 0.0})
    feature_scales = pd.Series({"x": 1.0})
    posterior = SimpleNamespace(
        posterior=xr.Dataset(
            {
                "intercept": (("chain", "draw"), np.array([[0.0, 0.0, 0.0]])),
                "beta": (("chain", "draw", "beta_dim_0"), np.array([[[0.0], [0.0], [0.0]]])),
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


def test_fit_recovers_a_known_linear_relation_on_all_cores():
    rng = np.random.default_rng(0)
    n = 600
    glucose = rng.normal(140.0, 40.0, n)
    iob = rng.gamma(2.0, 1.0, n)
    target = 15.0 + 0.9 * glucose - 6.0 * iob + rng.normal(0.0, 5.0, n)
    frame = FeatureFrame(
        frame=pd.DataFrame({"glucose": glucose, "iob_units": iob, "target_glucose": target}),
        feature_columns=["glucose", "iob_units"],
        target_column="target_glucose",
        horizon_minutes=30,
        config=FeatureConfig(horizon_minutes=30),
    )
    model = BayesianGlucoseModel(draws=300, tune=300, chains=2, random_seed=1)
    fit = model.fit(frame)

    assert fit.diagnostics is not None and fit.diagnostics.rhat_max < 1.01 and fit.diagnostics.divergences == 0
    new_rows = pd.DataFrame({"glucose": [100.0, 200.0], "iob_units": [0.0, 0.0]})
    predicted = model.predict(fit, new_rows)["mean"].to_numpy()
    assert predicted == pytest.approx([105.0, 195.0], abs=4.0)
    # Raising insulin on board by 1 U lowers the forecast by about 6 mg/dL.
    lower = model.predict(fit, new_rows.assign(iob_units=1.0))["mean"].to_numpy()
    assert (predicted - lower) == pytest.approx([6.0, 6.0], abs=1.5)


def test_scenario_forecasts_requires_target_glucose_column():
    model = BayesianGlucoseModel(draws=10, tune=10, chains=1)

    def _predict(_fit, frame: pd.DataFrame) -> pd.DataFrame:
        n = len(frame)
        return pd.DataFrame({"mean": np.zeros(n), "lower": np.zeros(n), "upper": np.zeros(n)}, index=frame.index)

    model.predict = _predict  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="missing 'target_glucose'"):
        model.scenario_forecasts(None, [("baseline", pd.DataFrame({"x": [1.0]}))])


def test_extract_fit_diagnostics_reads_sampler_health():
    rng = np.random.default_rng(11)
    diverging = np.zeros((2, 50), dtype=int)
    diverging[0, 1] = 1
    tree_depth = np.full((2, 50), 5, dtype=int)
    tree_depth[0, 1] = 10
    posterior = SimpleNamespace(
        posterior=xr.Dataset(
            {
                "intercept": (("chain", "draw"), rng.normal(size=(2, 50))),
                "beta": (("chain", "draw", "beta_dim_0"), rng.normal(size=(2, 50, 1))),
            }
        ),
        sample_stats=xr.Dataset(
            {
                "diverging": (("chain", "draw"), diverging),
                "tree_depth": (("chain", "draw"), tree_depth),
                "reached_max_treedepth": (("chain", "draw"), (tree_depth >= 10).astype(int)),
            }
        ),
    )
    posterior = az.InferenceData(posterior=posterior.posterior, sample_stats=posterior.sample_stats)

    diagnostics = extract_fit_diagnostics(
        posterior,
        draws=50,
        tune=30,
        chains=2,
        target_accept=0.9,
        max_treedepth=10,
        wall_time_seconds=1.25,
    )

    assert diagnostics.draws == 50
    assert diagnostics.tune == 30
    assert diagnostics.chains == 2
    assert diagnostics.divergences == 1
    assert diagnostics.max_tree_depth_observed == 10
    assert diagnostics.max_tree_depth_hits == 1
    assert diagnostics.rhat_max is not None and np.isfinite(diagnostics.rhat_max)


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


def test_predictive_quantiles_match_fitted_student_likelihood():
    from scipy.stats import t

    model = BayesianGlucoseModel(target_quantiles=(0.05, 0.95))
    n = 100_000
    fit = _make_minimal_fit(sigma_obs=1.0)
    summary = model._summarize_draws(fit, np.full((n, 1), 110.0), np.ones(n), np.random.default_rng(14), [0])
    expected = 110 + fit.target_scale * t.ppf([0.05, 0.95], df=5)
    np.testing.assert_allclose(summary.loc[0, ["lower", "upper"]], expected, atol=0.4)
    assert summary.loc[0, "mean"] == 110


def _posterior(parameters: dict[str, np.ndarray]):
    return az.from_dict(posterior=parameters)


def test_diagnostics_reduce_independent_dimensions_separately():
    from bayesian_t1dm.model import extract_fit_diagnostics

    rng = np.random.default_rng(3)
    posterior = _posterior({"level": rng.normal(size=(4, 500, 3)), "beta": rng.normal(size=(4, 500, 5)),
                            "sigma": np.abs(rng.normal(size=(4, 500)))})
    diagnostics = extract_fit_diagnostics(posterior, draws=500, tune=500, chains=4, target_accept=0.95,
                                          max_treedepth=10, wall_time_seconds=1.0)
    assert 0.99 < diagnostics.rhat_max < 1.02
    assert 500 < diagnostics.ess_bulk_min < 3000 and 500 < diagnostics.ess_tail_min < 3000


def test_diagnostics_fail_when_convergence_cannot_be_assessed():
    import pytest

    from bayesian_t1dm.model import extract_fit_diagnostics

    rng = np.random.default_rng(4)
    posterior = _posterior({"beta": rng.normal(size=(4, 200, 2)), "constant": np.ones((4, 200))})
    with pytest.raises(ValueError, match="constant"):
        extract_fit_diagnostics(posterior, draws=200, tune=200, chains=4, target_accept=0.95,
                                max_treedepth=10, wall_time_seconds=1.0)
