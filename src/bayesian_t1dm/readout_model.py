"""The readout's drivers model: how next-day time in range and time below 70 went with what came before the day.

For each outcome, one beta-binomial regression over the days in :func:`readout.model_days`:

    readings in range on day d ~ BetaBinomial(n_d, mu_d * phi, (1 - mu_d) * phi)
    logit(mu_d) = alpha + month effect + weekend + sum_k beta_k * driver_k(d)

The month effect (partially pooled) absorbs slow change in settings, season and habits, so a driver is compared with
days of the same month. Driver coefficients have a Normal(0, 0.5) prior on the logit scale, which pulls effects with
little support toward zero. The beta-binomial allows readings within a day to be more alike than independent readings.

Each driver's effect is reported in percentage points of the day as the average, over the modelled days, of the
difference in predicted rate with the driver set to yes and to no (an average marginal effect), with a 90% interval and
the probability that it is above zero. These are within-person associations: a driver "went with" a change, it is not
shown to cause it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .readout import DRIVER_COLUMNS
from .sampler_health import FitDiagnostics, extract_fit_diagnostics, sampler_gate, stack_draws

OUTCOMES = {"time_in_range": ("in_range_readings", "Time in range 70-180 mg/dL"), "time_below_70": ("below_70_readings", "Time below 70 mg/dL")}
DRAWS, TUNE, CHAINS, TARGET_ACCEPT, MAX_TREEDEPTH = 1000, 1000, 4, 0.95, 10
SEED = 20261120
MIN_DAYS_EACH_SIDE = 15  # a driver is estimated only with this many modelled days with it and without it
INTERVAL = (5.0, 95.0)


@dataclass(frozen=True)
class DriverFit:
    outcome: str
    effects: pd.DataFrame  # one row per driver: days_with, days_without, estimate_pp, lower_pp, upper_pp, prob_above_zero
    diagnostics: FitDiagnostics
    usable: bool
    days: int


def estimable_drivers(days: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """The drivers with enough modelled days on each side, and the count table for every driver."""
    counts = pd.DataFrame({"driver": list(DRIVER_COLUMNS), "days_with": [int(days[c].sum()) for c in DRIVER_COLUMNS]})
    counts["days_without"] = len(days) - counts["days_with"]
    counts["estimated"] = counts["days_with"].ge(MIN_DAYS_EACH_SIDE) & counts["days_without"].ge(MIN_DAYS_EACH_SIDE)
    return counts.loc[counts["estimated"], "driver"].tolist(), counts


def fit_outcome(days: pd.DataFrame, outcome: str, *, draws: int = DRAWS, tune: int = TUNE, chains: int = CHAINS) -> DriverFit:
    import pymc as pm

    count_column, _ = OUTCOMES[outcome]
    drivers, counts = estimable_drivers(days)
    x = days[drivers].astype(float).to_numpy()
    weekend = days["weekend"].astype(float).to_numpy()
    month_codes, months = pd.factorize(days.index.to_period("M"), sort=True)
    y = days[count_column].to_numpy(dtype=int)
    n = days["readings"].to_numpy(dtype=int)
    base = np.clip(y.sum() / n.sum(), 1e-3, 1 - 1e-3)
    with pm.Model():
        alpha = pm.Normal("alpha", mu=float(np.log(base / (1 - base))), sigma=1.0)
        beta = pm.Normal("beta", mu=0.0, sigma=0.5, shape=len(drivers))
        gamma = pm.Normal("weekend", mu=0.0, sigma=0.5)
        month_scale = pm.HalfNormal("month_scale", sigma=0.5)
        z_month = pm.Normal("z_month", 0.0, 1.0, shape=len(months))
        log_phi = pm.Normal("log_phi", 2.0, 1.5)
        eta = alpha + month_scale * z_month[month_codes] + gamma * weekend + pm.math.dot(x, beta)
        mu = pm.math.invlogit(eta)
        phi = pm.math.exp(log_phi)
        pm.BetaBinomial("y", n=n, alpha=mu * phi, beta=(1 - mu) * phi, observed=y)
        started = time.perf_counter()
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=chains, random_seed=SEED, target_accept=TARGET_ACCEPT, progressbar=False, compute_convergence_checks=False)
    diagnostics = extract_fit_diagnostics(idata, draws=draws, tune=tune, chains=chains, target_accept=TARGET_ACCEPT, max_treedepth=MAX_TREEDEPTH, wall_time_seconds=time.perf_counter() - started)

    a = stack_draws(idata, "alpha")
    b = stack_draws(idata, "beta").reshape(len(a), len(drivers))
    g = stack_draws(idata, "weekend")
    m = stack_draws(idata, "month_scale")[:, None] * stack_draws(idata, "z_month").reshape(len(a), len(months))
    eta_draws = a[:, None] + m[:, month_codes] + g[:, None] * weekend[None, :] + b @ x.T
    rows = []
    for k, driver in enumerate(drivers):
        without = eta_draws - b[:, [k]] * x[None, :, k]
        effect = 100.0 * (_invlogit(without + b[:, [k]]) - _invlogit(without)).mean(axis=1)
        lower, upper = np.percentile(effect, INTERVAL)
        rows.append({"driver": driver, "estimate_pp": float(effect.mean()), "lower_pp": float(lower), "upper_pp": float(upper), "prob_above_zero": float((effect > 0).mean())})
    effects = counts.merge(pd.DataFrame(rows, columns=["driver", "estimate_pp", "lower_pp", "upper_pp", "prob_above_zero"]), on="driver", how="left")
    return DriverFit(outcome=outcome, effects=effects, diagnostics=diagnostics, usable=sampler_gate(diagnostics), days=len(days))


def _invlogit(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def fit_all(days: pd.DataFrame, *, draws: int = DRAWS, tune: int = TUNE, chains: int = CHAINS) -> dict[str, DriverFit]:
    return {outcome: fit_outcome(days, outcome, draws=draws, tune=tune, chains=chains) for outcome in OUTCOMES}


def reading(row: pd.Series) -> str:
    """One plain word for a driver effect on time in range: helped, cost, or no clear link (90% interval spans zero)."""
    if not row.get("estimated", False) or pd.isna(row.get("estimate_pp")):
        return "not enough data yet"
    if row["lower_pp"] > 0:
        return "went with more time in range"
    if row["upper_pp"] < 0:
        return "went with less time in range"
    return "no clear link"
