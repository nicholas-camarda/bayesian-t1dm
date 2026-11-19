from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

az = pytest.importorskip("arviz")
import xarray as xr

from bayesian_t1dm.sampler_health import extract_fit_diagnostics


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


def _posterior(parameters: dict[str, np.ndarray]):
    return az.from_dict(posterior=parameters)


def test_diagnostics_reduce_independent_dimensions_separately():
    rng = np.random.default_rng(3)
    posterior = _posterior({"level": rng.normal(size=(4, 500, 3)), "beta": rng.normal(size=(4, 500, 5)),
                            "sigma": np.abs(rng.normal(size=(4, 500)))})
    diagnostics = extract_fit_diagnostics(posterior, draws=500, tune=500, chains=4, target_accept=0.95,
                                          max_treedepth=10, wall_time_seconds=1.0)
    assert 0.99 < diagnostics.rhat_max < 1.02
    assert 500 < diagnostics.ess_bulk_min < 3000 and 500 < diagnostics.ess_tail_min < 3000


def test_diagnostics_fail_when_convergence_cannot_be_assessed():
    rng = np.random.default_rng(4)
    posterior = _posterior({"beta": rng.normal(size=(4, 200, 2)), "constant": np.ones((4, 200))})
    with pytest.raises(ValueError, match="constant"):
        extract_fit_diagnostics(posterior, draws=200, tune=200, chains=4, target_accept=0.95,
                                max_treedepth=10, wall_time_seconds=1.0)
