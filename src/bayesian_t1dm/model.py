from __future__ import annotations

import os
import time
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
    diagnostics: "FitDiagnostics | None" = None
    model: object | None = None


@dataclass(frozen=True)
class ScenarioForecast:
    scenario_name: str
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    expected_loss: float
    # Per-posterior-draw mean absolute deviation from the glucose target, when the caller
    # supplied one. Draws are paired across scenarios (same parameters, same innovations),
    # so comparing two scenarios draw-by-draw gives a genuine posterior probability.
    target_loss_draws: np.ndarray | None = None


@dataclass(frozen=True)
class FitDiagnostics:
    draws: int
    tune: int
    chains: int
    target_accept: float
    max_treedepth: int
    wall_time_seconds: float
    divergences: int
    max_tree_depth_observed: int | None
    max_tree_depth_hits: int
    rhat_max: float | None
    ess_bulk_min: float | None
    ess_tail_min: float | None


def _finite_stat(values: np.ndarray, reducer) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(reducer(finite))


def extract_fit_diagnostics(
    posterior: object,
    *,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    max_treedepth: int,
    wall_time_seconds: float,
) -> FitDiagnostics:
    try:
        import arviz as az
    except Exception:  # pragma: no cover - dependency guard
        az = None

    sample_stats = getattr(posterior, "sample_stats", None)
    divergences = 0
    max_tree_depth_observed: int | None = None
    max_tree_depth_hits = 0

    if sample_stats is not None:
        if "diverging" in sample_stats:
            divergences = int(np.asarray(sample_stats["diverging"]).sum())
        elif "divergences" in sample_stats:
            divergences = int(np.asarray(sample_stats["divergences"]).sum())
        if "tree_depth" in sample_stats:
            tree_depth = np.asarray(sample_stats["tree_depth"], dtype=float)
            max_depth = _finite_stat(tree_depth, np.max)
            max_tree_depth_observed = None if max_depth is None else int(max_depth)
            if "reached_max_treedepth" in sample_stats:
                max_tree_depth_hits = int(np.asarray(sample_stats["reached_max_treedepth"]).sum())
            elif max_tree_depth_observed is not None:
                max_tree_depth_hits = int(np.sum(tree_depth >= max_treedepth))
        elif "reached_max_treedepth" in sample_stats:
            max_tree_depth_hits = int(np.asarray(sample_stats["reached_max_treedepth"]).sum())

    rhat_max: float | None = None
    ess_bulk_min: float | None = None
    ess_tail_min: float | None = None
    if az is not None and chains >= 2:
        try:
            rhat_dataset = az.rhat(posterior)
            rhat_max = _finite_stat(np.asarray(rhat_dataset.to_array()), np.max)
        except Exception:
            rhat_max = None
        try:
            ess_bulk_dataset = az.ess(posterior, method="bulk")
            ess_bulk_min = _finite_stat(np.asarray(ess_bulk_dataset.to_array()), np.min)
        except Exception:
            ess_bulk_min = None
        try:
            ess_tail_dataset = az.ess(posterior, method="tail")
            ess_tail_min = _finite_stat(np.asarray(ess_tail_dataset.to_array()), np.min)
        except Exception:
            ess_tail_min = None

    return FitDiagnostics(
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        max_treedepth=max_treedepth,
        wall_time_seconds=float(wall_time_seconds),
        divergences=divergences,
        max_tree_depth_observed=max_tree_depth_observed,
        max_tree_depth_hits=max_tree_depth_hits,
        rhat_max=rhat_max,
        ess_bulk_min=ess_bulk_min,
        ess_tail_min=ess_tail_min,
    )


class BayesianGlucoseModel:
    def __init__(
        self,
        *,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        target_quantiles: tuple[float, float] = (0.1, 0.9),
        target_accept: float = 0.95,
        max_treedepth: int = 12,
        random_seed: int = 7,
    ) -> None:
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_quantiles = target_quantiles
        self.target_accept = target_accept
        self.max_treedepth = max_treedepth
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

        # Direct regression of glucose `horizon` minutes ahead on the current features (current and lagged
        # glucose, insulin, carbs, basal, time of day). Current glucose is a feature, so "glucose stays the
        # same" is a special case of the model. There is no per-row latent state: rows are not contiguous
        # in time (CGM gaps are dropped), and a latent state per row would make the posterior 100k+ dimensional.
        with pm.Model() as model:
            # Data containers keep the design matrix out of the compiled graph; as literal constants a
            # 150k x 80 matrix makes the compiled sampler large enough to crash the worker processes.
            X_data = pm.Data("X", X)
            y_data = pm.Data("y", y_values)
            intercept = pm.Normal("intercept", mu=0.0, sigma=1.0)
            beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=X.shape[1])
            sigma_obs = pm.Exponential("sigma_obs", 1.0)
            mu = intercept + pm.math.dot(X_data, beta)
            pm.StudentT("obs", nu=5.0, mu=mu, sigma=sigma_obs, observed=y_data)
            sample_started = time.perf_counter()
            posterior = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                cores=min(self.chains, os.cpu_count() or 1),
                target_accept=self.target_accept,
                random_seed=self.random_seed,
                progressbar=False,
                return_inferencedata=True,
                nuts={"max_treedepth": self.max_treedepth},
            )
            wall_time_seconds = time.perf_counter() - sample_started

        diagnostics = extract_fit_diagnostics(
            posterior,
            draws=self.draws,
            tune=self.tune,
            chains=self.chains,
            target_accept=self.target_accept,
            max_treedepth=self.max_treedepth,
            wall_time_seconds=wall_time_seconds,
        )

        return ModelFit(
            posterior=posterior,
            feature_columns=list(frame.feature_columns),
            feature_means=means,
            feature_scales=scales,
            target_mean=target_mean,
            target_scale=target_scale,
            horizon_minutes=frame.horizon_minutes,
            diagnostics=diagnostics,
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

    def _simulate_mean_draws(self, fit: ModelFit, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.random.Generator]:
        """Return (mean_draws, sigma_obs, rng): the predicted mean for every posterior draw.

        ``mean_draws`` has shape (n_posterior_samples, n_rows) in glucose units. The random
        generator is seeded from ``random_seed`` on every call, so two calls on scenario
        variants of the same frame use identical observation noise (common random numbers).
        """
        posterior = fit.posterior
        X = self._prepare_matrix(fit, frame)
        intercept = self._stack_draws(posterior, "intercept")
        beta = self._stack_draws(posterior, "beta")
        if beta.ndim != 2:
            beta = np.reshape(beta, (beta.shape[0], -1))
        if X.shape[1] != beta.shape[1]:
            raise ValueError(
                f"Prediction matrix has {X.shape[1]} features but posterior beta expects {beta.shape[1]}"
            )
        sigma_obs = self._stack_draws(posterior, "sigma_obs")  # shape (n_samples,)
        rng = np.random.default_rng(self.random_seed)
        mean = (intercept[:, np.newaxis] + beta @ X.T) * fit.target_scale + fit.target_mean
        return mean, sigma_obs, rng

    def _summarize_draws(self, fit: ModelFit, mean: np.ndarray, sigma_obs: np.ndarray, rng: np.random.Generator, index) -> pd.DataFrame:
        central = np.mean(mean, axis=0)
        # Add observation noise to each posterior draw before computing quantiles,
        # so that predictive intervals reflect full posterior predictive uncertainty.
        predictive = rng.normal(0.0, sigma_obs[:, np.newaxis] * fit.target_scale, size=mean.shape)
        predictive += mean
        lower_q, upper_q = self.target_quantiles
        lower, upper = np.quantile(predictive, [lower_q, upper_q], axis=0)
        return pd.DataFrame({"mean": central, "lower": lower, "upper": upper}, index=index)

    def predict(self, fit: ModelFit, frame: pd.DataFrame) -> pd.DataFrame:
        mean, sigma_obs, rng = self._simulate_mean_draws(fit, frame)
        return self._summarize_draws(fit, mean, sigma_obs, rng, frame.index)

    def scenario_forecasts(
        self,
        fit: ModelFit,
        scenarios: Iterable[tuple[str, pd.DataFrame]],
        *,
        target_bg: float | None = None,
    ) -> list[ScenarioForecast]:
        results: list[ScenarioForecast] = []
        for name, scenario_frame in scenarios:
            if "target_glucose" not in scenario_frame.columns:
                raise ValueError(
                    f"Scenario '{name}' frame is missing 'target_glucose' column. "
                    "Pass the full feature frame (including target) to scenario_forecasts."
                )
            mean_draws, sigma_obs, rng = self._simulate_mean_draws(fit, scenario_frame)
            target_loss_draws = None
            if target_bg is not None:
                target_loss_draws = np.mean(np.abs(mean_draws - float(target_bg)), axis=1)
            prediction = self._summarize_draws(fit, mean_draws, sigma_obs, rng, scenario_frame.index)
            del mean_draws
            mean = prediction["mean"].to_numpy()
            expected_loss = float(np.mean(np.abs(mean - scenario_frame["target_glucose"].to_numpy())))
            results.append(
                ScenarioForecast(
                    scenario_name=name,
                    mean=mean,
                    lower=prediction["lower"].to_numpy(),
                    upper=prediction["upper"].to_numpy(),
                    expected_loss=expected_loss,
                    target_loss_draws=target_loss_draws,
                )
            )
        return results
