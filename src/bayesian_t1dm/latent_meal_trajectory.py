"""Trajectory-based latent carbohydrate estimation for a single meal window.

The estimator answers one question: given the CGM trace after a bolus, the insulin that
was actually absorbed over that trace, and the programmed therapy settings, how many
grams of carbohydrate are needed to explain the glucose that was observed?

Forward model, for minutes ``t`` after the meal anchor::

    glucose(t) = baseline + b0 + b1 * t
                 + (CF / ICR) * carbs * A(t - delay; shape, mean_time)
                 - CF * (1 + d) * insulin_absorbed(t)
                 + noise(t)

``A`` is a gamma CDF describing the fraction of the meal absorbed so far. ``b0`` (baseline
offset), ``b1`` (linear drift) and ``d`` (relative error in the programmed correction
factor) are nuisance terms with zero-mean Gaussian priors; they are integrated out in
closed form. ``noise`` is AR(1) so that 37 correlated CGM samples are not counted as 37
independent observations. The posterior over (carbs, shape, mean_time, delay) is evaluated
on a grid.

What this can and cannot identify
---------------------------------
Glucose only identifies the product ``carbs * CF / ICR``. The carb estimate is therefore a
*carb-equivalent under the programmed settings*: if the programmed I:C ratio or correction
factor is wrong, the estimate absorbs that error. Carbohydrate that has not been absorbed
by the end of the window (slow, high-fat meals) is invisible and biases the estimate low.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.stats import gamma as _gamma


@dataclass(frozen=True)
class MealWindowTrajectory:
    """Observed post-meal trace for one window.

    ``minutes`` are minutes since the meal anchor for rows with a real CGM reading,
    ``glucose`` the readings, and ``insulin_absorbed_units`` the cumulative units of
    bolus insulin absorbed since the anchor (including decay of insulin already on board).
    """

    minutes: np.ndarray
    glucose: np.ndarray
    insulin_absorbed_units: np.ndarray

    def __post_init__(self) -> None:
        minutes = np.asarray(self.minutes, dtype=float)
        glucose = np.asarray(self.glucose, dtype=float)
        absorbed = np.asarray(self.insulin_absorbed_units, dtype=float)
        if not (minutes.shape == glucose.shape == absorbed.shape) or minutes.ndim != 1:
            raise ValueError("minutes, glucose and insulin_absorbed_units must be 1-D arrays of equal length")
        object.__setattr__(self, "minutes", minutes)
        object.__setattr__(self, "glucose", glucose)
        object.__setattr__(self, "insulin_absorbed_units", absorbed)

    @property
    def n_points(self) -> int:
        return int(self.minutes.size)

    def is_usable(self, *, min_points: int = 18, min_span_minutes: float = 150.0, max_start_minutes: float = 10.0) -> bool:
        if self.n_points < min_points:
            return False
        if not (np.all(np.isfinite(self.minutes)) and np.all(np.isfinite(self.glucose)) and np.all(np.isfinite(self.insulin_absorbed_units))):
            return False
        if np.any(np.diff(self.minutes) <= 0):
            return False
        return bool(self.minutes[0] <= max_start_minutes and self.minutes[-1] - self.minutes[0] >= min_span_minutes)


@dataclass(frozen=True)
class TrajectoryEstimatorConfig:
    # CGM + model-misfit noise. AR(1) correlation is per 5 minutes.
    noise_sd_mgdl: float = 8.0
    noise_ar1_per_5min: float = 0.7
    # Nuisance priors (zero mean).
    baseline_sd_mgdl: float = 5.0
    drift_sd_mgdl_per_hour: float = 8.0
    correction_factor_relative_sd: float = 0.15
    # Carb prior: centred on the bolus-implied carbs, deliberately wide so the trace decides.
    carb_prior_relative_sd: float = 0.5
    carb_prior_min_sd_grams: float = 20.0
    carb_grid_points: int = 181
    # Absorption family: gamma CDFs. ``shape`` 1.5 is front-loaded with a long tail, 7 is
    # close to a symmetric S-curve. ``mean_time`` is the mean absorption time in minutes.
    absorption_shapes: tuple[float, ...] = (1.5, 2.5, 4.0, 7.0)
    absorption_mean_minutes_grid: tuple[float, ...] = tuple(float(value) for value in np.linspace(35.0, 170.0, 19))
    absorption_mean_minutes_prior_log_sd: float = 0.45
    # Minutes between the bolus and the start of carbohydrate appearance.
    meal_delay_minutes: tuple[float, ...] = (0.0, 10.0, 20.0)
    # Posterior-mean fit worse than this is flagged as a poor fit.
    poor_fit_rmse_mgdl: float = 15.0


@dataclass(frozen=True)
class TrajectoryCarbPosterior:
    carb_mean: float
    carb_median: float
    carb_lower_50: float
    carb_upper_50: float
    carb_lower_90: float
    carb_upper_90: float
    absorption_minutes_mean: float
    absorption_minutes_lower_90: float
    absorption_minutes_upper_90: float
    carb_prior_mean: float
    carb_prior_sd: float
    # 1 - posterior_var / prior_var for carbs: 0 means the trace taught us nothing beyond
    # the bolus-implied prior, values near 1 mean the trace dominates.
    prior_contraction: float
    fit_rmse_mgdl: float
    poor_fit: bool
    n_points: int
    # Flattened grid for sampling: carbs, time-to-95%-absorbed, weight. Sorted by carbs.
    grid_carbs: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    grid_absorption_minutes: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    grid_weights: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))

    def summary(self) -> dict[str, Any]:
        return {
            "latent_carb_mean": self.carb_mean,
            "latent_carb_median": self.carb_median,
            "latent_carb_lower_50": self.carb_lower_50,
            "latent_carb_upper_50": self.carb_upper_50,
            "latent_carb_lower_90": self.carb_lower_90,
            "latent_carb_upper_90": self.carb_upper_90,
            "absorption_minutes_mean": self.absorption_minutes_mean,
            "absorption_minutes_lower_90": self.absorption_minutes_lower_90,
            "absorption_minutes_upper_90": self.absorption_minutes_upper_90,
            "prior_contraction": self.prior_contraction,
            "trajectory_fit_rmse_mgdl": self.fit_rmse_mgdl,
        }


def absorbed_fraction(minutes: np.ndarray, *, shape: float, mean_minutes: float, delay_minutes: float = 0.0) -> np.ndarray:
    """Fraction of a meal absorbed ``minutes`` after the anchor (gamma CDF)."""
    elapsed = np.clip(np.asarray(minutes, dtype=float) - float(delay_minutes), 0.0, None)
    return _gamma.cdf(elapsed, a=float(shape), scale=float(mean_minutes) / float(shape))


@lru_cache(maxsize=64)
def _absorption_basis(
    minutes_bytes: bytes,
    shapes: tuple[float, ...],
    mean_grid: tuple[float, ...],
    delays: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Absorbed-fraction curves [shape, mean, delay, time] and time-to-95% [shape, mean].

    Cached on the time grid because nearly every window shares the same 5-minute grid.
    """
    minutes = np.frombuffer(minutes_bytes, dtype=float)
    absorbed = np.stack(
        [
            [[absorbed_fraction(minutes, shape=shape, mean_minutes=mean, delay_minutes=delay) for delay in delays] for mean in mean_grid]
            for shape in shapes
        ]
    )
    t95 = np.stack([[_gamma.ppf(0.95, a=shape, scale=mean / shape) for mean in mean_grid] for shape in shapes])
    absorbed.setflags(write=False)
    t95.setflags(write=False)
    return absorbed, t95


def _marginal_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Quantile of a discrete distribution whose support is sorted ascending."""
    cumulative = np.cumsum(weights)
    cumulative = cumulative / cumulative[-1]
    return float(np.interp(float(quantile), cumulative, values))


def estimate_carbs_from_trajectory(
    trajectory: MealWindowTrajectory,
    *,
    icr_grams_per_unit: float,
    correction_factor_mgdl_per_unit: float,
    bolus_implied_carbs: float,
    baseline_glucose: float | None = None,
    max_carbs: float = 180.0,
    absorption_mean_minutes_prior: float = 80.0,
    carb_prior_mean: float | None = None,
    carb_prior_sd: float | None = None,
    uncertainty_multiplier: float = 1.0,
    config: TrajectoryEstimatorConfig | None = None,
) -> TrajectoryCarbPosterior:
    """Grid posterior over meal carbohydrate given an observed post-meal trace."""
    config = config or TrajectoryEstimatorConfig()
    if icr_grams_per_unit <= 0 or correction_factor_mgdl_per_unit <= 0:
        raise ValueError("icr_grams_per_unit and correction_factor_mgdl_per_unit must be positive")
    if max_carbs <= 0:
        raise ValueError("max_carbs must be positive")
    if trajectory.n_points < 4:
        raise ValueError("trajectory needs at least four CGM points")

    t = trajectory.minutes
    y = trajectory.glucose
    absorbed_units = trajectory.insulin_absorbed_units
    n = t.size
    cf = float(correction_factor_mgdl_per_unit)
    carb_sensitivity = cf / float(icr_grams_per_unit)
    scale = float(max(uncertainty_multiplier, 1e-6))
    baseline = float(y[0]) if baseline_glucose is None or not np.isfinite(baseline_glucose) else float(baseline_glucose)

    # Observation covariance: AR(1) noise plus the integrated-out nuisance terms.
    lag_steps = np.abs(t[:, None] - t[None, :]) / 5.0
    noise_cov = (config.noise_sd_mgdl * scale) ** 2 * config.noise_ar1_per_5min**lag_steps
    nuisance_design = np.column_stack([np.ones(n), t / 60.0, cf * absorbed_units])
    nuisance_prior_cov = np.diag(
        [
            (config.baseline_sd_mgdl * scale) ** 2,
            (config.drift_sd_mgdl_per_hour * scale) ** 2,
            (config.correction_factor_relative_sd * scale) ** 2,
        ]
    )
    precision = np.linalg.inv(noise_cov + nuisance_design @ nuisance_prior_cov @ nuisance_design.T)

    shapes = np.asarray(config.absorption_shapes, dtype=float)
    mean_grid = np.asarray(config.absorption_mean_minutes_grid, dtype=float)
    delays = np.asarray(config.meal_delay_minutes, dtype=float)
    carb_grid = np.linspace(0.0, float(max_carbs), int(config.carb_grid_points))

    # absorbed[a, m, d, n], t95[a, m]
    absorbed, t95 = _absorption_basis(
        np.ascontiguousarray(t, dtype=float).tobytes(),
        tuple(float(value) for value in shapes),
        tuple(float(value) for value in mean_grid),
        tuple(float(value) for value in delays),
    )
    # Glucose left to be explained by carbohydrate once insulin is accounted for.
    carb_signal = y - baseline + cf * absorbed_units
    quad_const = float(carb_signal @ precision @ carb_signal)
    quad_linear = np.einsum("amdn,nk,k->amd", absorbed, precision, carb_signal)
    quad_square = np.einsum("amdn,nk,amdk->amd", absorbed, precision, absorbed)
    effect = carb_sensitivity * carb_grid[:, None, None, None]
    quad = quad_const - 2.0 * effect * quad_linear[None] + effect**2 * quad_square[None]

    prior_mean = float(bolus_implied_carbs if carb_prior_mean is None else carb_prior_mean)
    prior_sd = float(
        max(config.carb_prior_min_sd_grams, config.carb_prior_relative_sd * prior_mean) * scale
        if carb_prior_sd is None
        else carb_prior_sd
    )
    log_weight = (
        -0.5 * quad
        - 0.5 * ((carb_grid[:, None, None, None] - prior_mean) / prior_sd) ** 2
        - 0.5
        * (
            (np.log(mean_grid)[None, None, :, None] - np.log(float(absorption_mean_minutes_prior)))
            / config.absorption_mean_minutes_prior_log_sd
        )
        ** 2
    )
    log_weight = log_weight - np.max(log_weight)
    weights = np.exp(log_weight)
    weights = weights / weights.sum()

    carb_marginal = weights.sum(axis=(1, 2, 3))
    carb_mean = float(np.sum(carb_grid * carb_marginal))
    carb_var = float(np.sum((carb_grid - carb_mean) ** 2 * carb_marginal))

    # Duration-like absorption summary: minutes until 95% of the meal has been absorbed.
    t95_full = np.broadcast_to(t95[None, :, :, None] + delays[None, None, None, :], weights.shape)
    absorption_weights = weights.sum(axis=0)  # [a, m, d]
    t95_cells = (t95[:, :, None] + delays[None, None, :]).ravel()
    order = np.argsort(t95_cells)
    t95_sorted = t95_cells[order]
    t95_weights = absorption_weights.ravel()[order]

    # Fit quality at the posterior mode, with nuisance terms set to their conditional mean.
    mode_index = np.unravel_index(int(np.argmax(weights)), weights.shape)
    mode_residual = carb_signal - carb_sensitivity * carb_grid[mode_index[0]] * absorbed[mode_index[1], mode_index[2], mode_index[3]]
    nuisance_mean = nuisance_prior_cov @ nuisance_design.T @ precision @ mode_residual
    fit_residual = mode_residual - nuisance_design @ nuisance_mean
    fit_rmse = float(np.sqrt(np.mean(fit_residual**2)))

    flat_carbs = np.broadcast_to(carb_grid[:, None, None, None], weights.shape).ravel()
    flat_absorption = t95_full.ravel()
    flat_weights = weights.ravel()
    flat_order = np.argsort(flat_carbs, kind="stable")

    return TrajectoryCarbPosterior(
        carb_mean=carb_mean,
        carb_median=_marginal_quantile(carb_grid, carb_marginal, 0.50),
        carb_lower_50=_marginal_quantile(carb_grid, carb_marginal, 0.25),
        carb_upper_50=_marginal_quantile(carb_grid, carb_marginal, 0.75),
        carb_lower_90=_marginal_quantile(carb_grid, carb_marginal, 0.05),
        carb_upper_90=_marginal_quantile(carb_grid, carb_marginal, 0.95),
        absorption_minutes_mean=float(np.sum(t95_sorted * t95_weights)),
        absorption_minutes_lower_90=_marginal_quantile(t95_sorted, t95_weights, 0.05),
        absorption_minutes_upper_90=_marginal_quantile(t95_sorted, t95_weights, 0.95),
        carb_prior_mean=prior_mean,
        carb_prior_sd=prior_sd,
        prior_contraction=float(np.clip(1.0 - carb_var / prior_sd**2, 0.0, 1.0)),
        fit_rmse_mgdl=fit_rmse,
        poor_fit=bool(fit_rmse > config.poor_fit_rmse_mgdl * scale),
        n_points=int(n),
        grid_carbs=flat_carbs[flat_order],
        grid_absorption_minutes=flat_absorption[flat_order],
        grid_weights=flat_weights[flat_order],
    )
