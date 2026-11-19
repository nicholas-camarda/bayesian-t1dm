"""Sampler-health summaries shared by the Bayesian models: divergences, tree depth, R-hat and effective sample size."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("PYTENSOR_FLAGS", f"compiledir={tempfile.gettempdir()}/bayesian_t1dm_pytensor")


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


def _dataset_extremum(dataset, reducer, name: str) -> float:
    """Reduce each parameter on its own so independent dimensions never broadcast against one another.

    A non-finite value means the diagnostic is undefined for that parameter, so convergence cannot be assessed.
    """
    extrema = []
    for variable, item in dataset.data_vars.items():
        values = np.asarray(item)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} is not finite for parameter '{variable}'; convergence cannot be assessed")
        extrema.append(reducer(values))
    if not extrema:
        raise ValueError(f"{name} was computed for no parameters")
    return float(reducer(extrema))


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
        rhat_max = _dataset_extremum(az.rhat(posterior), np.max, "R-hat")
        ess_bulk_min = _dataset_extremum(az.ess(posterior, method="bulk"), np.min, "Bulk ESS")
        ess_tail_min = _dataset_extremum(az.ess(posterior, method="tail"), np.min, "Tail ESS")

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


def stack_draws(posterior: object, name: str) -> np.ndarray:
    """Posterior draws of one variable as an array with samples (chains x draws) first."""
    variable = posterior.posterior[name].stack(sample=("chain", "draw"))
    dims = ["sample"] + [dim for dim in variable.dims if dim != "sample"]
    return variable.transpose(*dims).values
