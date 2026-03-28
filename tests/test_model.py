from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

arviz = pytest.importorskip("arviz")
import xarray as xr

from bayesian_t1dm.features import FeatureConfig, build_feature_frame
from bayesian_t1dm.ingest import load_tandem_exports
from bayesian_t1dm.model import BayesianGlucoseModel, ModelFit


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
