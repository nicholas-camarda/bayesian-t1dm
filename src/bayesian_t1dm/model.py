from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .features import FeatureFrame


os.environ.setdefault("PYTENSOR_FLAGS", f"compiledir={tempfile.gettempdir()}/bayesian_t1dm_pytensor")


@dataclass(frozen=True)
class ModelFit:
    posterior: object
    feature_columns: list[str]
    feature_means: pd.Series
    feature_scales: pd.Series
    target_mean: float
    target_scale: float
    horizon_minutes: int
    model: object | None = None


@dataclass(frozen=True)
class ScenarioForecast:
    scenario_name: str
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    expected_loss: float


class BayesianGlucoseModel:
    def __init__(
        self,
        *,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 2,
        target_quantiles: tuple[float, float] = (0.1, 0.9),
        random_seed: int = 7,
    ) -> None:
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_quantiles = target_quantiles
        self.random_seed = random_seed

    @staticmethod
    def _standardize(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        features = frame.loc[:, list(columns)].astype(float)
        means = features.mean(axis=0)
        scales = features.std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0)
        standardized = (features - means) / scales
        return standardized, means, scales

    def fit(self, frame: FeatureFrame) -> ModelFit:
        try:
            import arviz as az
            import pymc as pm
        except Exception as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("PyMC and ArviZ are required to fit the Bayesian model") from exc

        data = frame.frame.copy()
        standardized, means, scales = self._standardize(data, frame.feature_columns)
        y = data[frame.target_column].astype(float)
        target_mean = float(y.mean())
        target_scale = float(y.std(ddof=0))
        if not np.isfinite(target_scale) or target_scale == 0.0:
            target_scale = 1.0
        y_std = (y - target_mean) / target_scale

        X = standardized.to_numpy(dtype=float)
        y_values = y_std.to_numpy(dtype=float)

        with pm.Model() as model:
            intercept = pm.Normal("intercept", mu=0.0, sigma=1.0)
            beta = pm.Normal("beta", mu=0.0, sigma=0.5, shape=X.shape[1])
            rho = pm.Beta("rho", alpha=2.0, beta=2.0)
            sigma_state = pm.Exponential("sigma_state", 1.0)
            sigma_obs = pm.Exponential("sigma_obs", 1.0)
            latent = pm.AR("latent", rho=rho, sigma=sigma_state, constant=False, ar_order=1, shape=len(y_values))
            mu = intercept + pm.math.dot(X, beta) + latent
            pm.StudentT("obs", nu=5.0, mu=mu, sigma=sigma_obs, observed=y_values)
            posterior = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                cores=1,
                target_accept=0.9,
                random_seed=self.random_seed,
                progressbar=False,
                return_inferencedata=True,
            )

        return ModelFit(
            posterior=posterior,
            feature_columns=list(frame.feature_columns),
            feature_means=means,
            feature_scales=scales,
            target_mean=target_mean,
            target_scale=target_scale,
            horizon_minutes=frame.horizon_minutes,
            model=model,
        )

    def _prepare_matrix(self, fit: ModelFit, frame: pd.DataFrame) -> np.ndarray:
        X = frame.loc[:, fit.feature_columns].astype(float)
        X = (X - fit.feature_means) / fit.feature_scales.replace(0, 1.0)
        return X.to_numpy(dtype=float)

    @staticmethod
    def _stack_draws(posterior: object, name: str) -> np.ndarray:
        variable = posterior.posterior[name].stack(sample=("chain", "draw"))
        dims = ["sample"] + [dim for dim in variable.dims if dim != "sample"]
        return variable.transpose(*dims).values

    def predict(self, fit: ModelFit, frame: pd.DataFrame) -> pd.DataFrame:
        posterior = fit.posterior
        X = self._prepare_matrix(fit, frame)
        intercept = self._stack_draws(posterior, "intercept")
        beta = self._stack_draws(posterior, "beta")
        latent = self._stack_draws(posterior, "latent")
        rho = self._stack_draws(posterior, "rho")

        if beta.ndim != 2:
            beta = np.reshape(beta, (beta.shape[0], -1))
        if X.shape[1] != beta.shape[1]:
            raise ValueError(
                f"Prediction matrix has {X.shape[1]} features but posterior beta expects {beta.shape[1]}"
            )

        n_samples = intercept.shape[0]
        n_obs = X.shape[0]
        mean = np.zeros((n_samples, n_obs), dtype=float)
        last_latent = latent[:, -1]
        state = last_latent.copy()
        for t in range(n_obs):
            linear = intercept + (X[t] @ beta.T)
            state = rho * state
            mean[:, t] = (linear + state) * fit.target_scale + fit.target_mean
        lower_q, upper_q = self.target_quantiles
        lower = np.quantile(mean, lower_q, axis=0)
        upper = np.quantile(mean, upper_q, axis=0)
        central = np.mean(mean, axis=0)
        return pd.DataFrame({
            "mean": central,
            "lower": lower,
            "upper": upper,
        }, index=frame.index)

    def scenario_forecasts(
        self,
        fit: ModelFit,
        scenarios: Iterable[tuple[str, pd.DataFrame]],
    ) -> list[ScenarioForecast]:
        results: list[ScenarioForecast] = []
        for name, scenario_frame in scenarios:
            prediction = self.predict(fit, scenario_frame)
            mean = prediction["mean"].to_numpy()
            lower = prediction["lower"].to_numpy()
            upper = prediction["upper"].to_numpy()
            expected_loss = float(np.mean(np.abs(mean - scenario_frame.get("target_glucose", mean))))
            results.append(
                ScenarioForecast(
                    scenario_name=name,
                    mean=mean,
                    lower=lower,
                    upper=upper,
                    expected_loss=expected_loss,
                )
            )
        return results
