"""Held-out retrospective check of meal trajectory coverage and residual dependence."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import therapy_research as meals
from .analysis_data import load_analysis_data
from .controller import programmed_units_by_slot
from .latent_meal_trajectory import (
    MIN_LOGGED_MEALS_FOR_ICR,
    TrajectoryEstimatorConfig,
    _absorption_basis,
    observe_carb_sensitivity,
    pool_effective_icr,
)

VALIDATION_DAYS = 120
DRAWS = 500
SEED = 20261118
ABSORPTION_MEAN_MINUTES = 80.0


def _known_setting(row, calculator: str, profile: str) -> float:
    for name in (calculator, profile):
        result = meals._as_float(getattr(row, name, np.nan))
        if np.isfinite(result) and result > 0:
            return result
    return np.nan


def _predict(trajectory, row, scale, pool, rng, config):
    """Generate CGM before viewing held-out CGM, conditional on the recorded insulin trace."""
    t = trajectory.minutes
    cf = row.cf
    sensitivity = pool.sensitivity_ratio * np.exp(rng.normal(0, pool.between_meal_log_sd, DRAWS))
    basis, _ = _absorption_basis(t.tobytes(), config.absorption_shapes, config.absorption_mean_minutes_grid, config.meal_delay_minutes)
    log_mean = np.log(config.absorption_mean_minutes_grid)
    weights = np.exp(-0.5 * ((log_mean - np.log(ABSORPTION_MEAN_MINUTES)) / config.absorption_mean_minutes_prior_log_sd)**2)
    absorption = basis[rng.integers(len(config.absorption_shapes), size=DRAWS),
                       rng.choice(len(log_mean), p=weights / weights.sum(), size=DRAWS),
                       rng.integers(len(config.meal_delay_minutes), size=DRAWS)]
    insulin = cf * trajectory.insulin_absorbed_units
    baseline = row.premeal_glucose
    center = baseline + (row.stated_carbs * sensitivity * cf / row.icr)[:, None] * absorption - insulin
    lag = np.abs(t[:, None] - t[None, :]) / 5
    design = np.column_stack([np.ones(len(t)), t / 60, insulin])
    noise = config.noise_sd_mgdl**2 * config.noise_ar1_per_5min**lag
    nuisance = np.array([config.baseline_sd_mgdl, config.drift_sd_mgdl_per_hour, config.correction_factor_relative_sd])
    cov = (noise + (design * nuisance**2) @ design.T) * scale**2
    draws = center + rng.normal(size=center.shape) @ np.linalg.cholesky(cov).T
    return draws


def validate(raw: Path, output: Path):
    analysis = load_analysis_data(raw)
    d = analysis.merged.preparation.dataset
    frame = meals.build_therapy_research_frame(d, excluded_periods=meals.load_excluded_periods(raw))
    registry = meals.build_meal_event_registry(frame)
    clean = meals.build_first_meal_clean_window_registry(frame, unlogged=analysis.unlogged, exercise=analysis.exercise,
                                                         meal_segments=("morning", "afternoon", "evening"))
    windows = meals.build_latent_meal_tiered_window_registry(clean)
    metrics = meals._accepted_first_meal_window_metrics(frame, registry, windows)
    trajectories = meals._first_meal_window_trajectories(frame, metrics, programmed_units=programmed_units_by_slot(analysis))
    metrics = metrics.loc[metrics.meal_id.astype(str).isin(trajectories) & metrics.stated_carbs.gt(0)].copy()
    rows = list(metrics.itertuples(index=False))
    metrics["icr"] = [_known_setting(r, "carb_ratio_g_per_unit", "profile_carb_ratio_g_per_unit") for r in rows]
    metrics["cf"] = [_known_setting(r, "isf_mgdl_per_unit", "profile_isf_mgdl_per_unit") for r in rows]
    metrics = metrics.dropna(subset=["icr", "cf", "premeal_glucose"])
    holdout = analysis.cgm.timestamp.max().normalize() - pd.Timedelta(days=VALIDATION_DAYS)
    config = TrajectoryEstimatorConfig()
    scales = {str(t["analysis_tier"]): t["uncertainty_multiplier"] for t in meals.LATENT_MEAL_ANALYSIS_TIERS}
    rng = np.random.default_rng(SEED)
    results = []
    training = []
    for segment, group in metrics.groupby("segment"):
        observations = []
        for row in group.loc[group.timestamp.lt(holdout)].itertuples():
            observation = observe_carb_sensitivity(
                trajectories[str(row.meal_id)], stated_carbs=row.stated_carbs, icr_grams_per_unit=row.icr,
                correction_factor_mgdl_per_unit=row.cf, baseline_glucose=row.premeal_glucose,
                uncertainty_multiplier=scales[row.analysis_tier], config=config,
            )
            if observation is not None:
                observations.append(observation)
        pool = pool_effective_icr(observations, programmed_icr=1)
        future = group.loc[group.timestamp.ge(holdout)]
        training.append({"segment": segment, "training_meals": len(observations), "held_out_meals": len(future), "holdout_start": holdout.date()})
        if len(observations) < MIN_LOGGED_MEALS_FOR_ICR:
            continue
        for row in future.itertuples():
            trajectory = trajectories[str(row.meal_id)]
            y, t = trajectory.glucose, trajectory.minutes
            draws = _predict(trajectory, row, scales[row.analysis_tier], pool, rng, config)
            lo, median, hi = np.quantile(draws, [0.05, 0.5, 0.95], axis=0)
            residual = y - median
            simulated = draws - median
            adjacent = np.flatnonzero(np.diff(t, prepend=-np.inf) == 5)
            pairs = set(adjacent)
            for i, (err, minute) in enumerate(zip(residual, t)):
                results.append({"meal": row.meal_id, "segment": segment, "minutes": minute, "covered": lo[i] <= y[i] <= hi[i],
                                "width": hi[i] - lo[i], "residual": err, "prior_residual": residual[i - 1] if i in pairs else np.nan,
                                "simulated_covariance": np.mean(simulated[:, i] * simulated[:, i - 1]) if i in pairs else np.nan,
                                "simulated_variance": np.mean(simulated[:, i]**2), "simulated_prior_variance": np.mean(simulated[:, i - 1]**2) if i in pairs else np.nan})
    diagnostics = pd.DataFrame(results)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(training).to_csv(output / "denominators.csv", index=False)
    if diagnostics.empty:
        return
    diagnostics["time"] = pd.cut(diagnostics.minutes, [-1, 30, 90, 180], labels=["0-30", "35-90", "95-180"])
    summary = []
    for (segment, window), group in diagnostics.groupby(["segment", "time"], observed=True):
        adjacent = group.dropna(subset=["prior_residual"])
        observed = adjacent.residual.corr(adjacent.prior_residual)
        expected = adjacent.simulated_covariance.mean() / np.sqrt(adjacent.simulated_variance.mean() * adjacent.simulated_prior_variance.mean())
        summary.append(dict(segment=segment, window=str(window), meals=group.meal.nunique(), readings=len(group),
                            coverage=group.covered.mean(), mean_width_mgdl=group.width.mean(),
                            mean_residual_mgdl=group.residual.mean(), five_minute_pairs=len(adjacent),
                            observed_residual_correlation=observed, simulated_residual_correlation=expected))
    pd.DataFrame(summary).to_csv(output / "coverage_and_dependence.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate(args.raw, args.output)
