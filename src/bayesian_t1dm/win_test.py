"""The win test: a paired CRPS difference with a circular block bootstrap interval, and the verdict read from it.

The difference ``d_k = CRPS(candidate, k) - CRPS(reference, k)`` is taken at each scored forecast origin ``k`` (negative means
the candidate is better). The statistic is the mean of ``d``. Origins overlap in time, so ``d`` is autocorrelated; the
circular block bootstrap resamples blocks of consecutive origins (wrapping around the end of the sequence) and so keeps the
dependence inside a block. The candidate wins only when the whole 90% interval lies below zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261119
INTERVAL_PERCENT = (5.0, 95.0)
VERDICTS = ("win", "tie", "loss")


@dataclass(frozen=True)
class PairedInterval:
    """Mean paired difference with its bootstrap interval and the verdict the interval gives."""

    mean: float
    lower: float
    upper: float
    origins: int
    block: int
    verdict: str

    @property
    def half_width(self) -> float:
        return (self.upper - self.lower) / 2.0


def crps_points(draws: np.ndarray, observed: float) -> float:
    """CRPS of predictive draws of a share (0 to 1) against the observed share, in percentage points."""
    from .replay import crps

    return 100.0 * crps(np.asarray(draws, dtype=float), float(observed))


def circular_block_indices(n: int, block: int, resamples: int, seed: int) -> np.ndarray:
    """Index matrix (resamples x n) of a circular block bootstrap: blocks of ``block`` consecutive positions start anywhere and wrap around the end."""
    if n < 1:
        raise ValueError("The bootstrap needs at least one origin")
    if not 1 <= block <= n:
        raise ValueError(f"Block length {block} must be between 1 and the number of origins ({n})")
    rng = np.random.default_rng(seed)
    blocks_per_sample = -(-n // block)
    starts = rng.integers(0, n, size=(resamples, blocks_per_sample))
    positions = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return positions.reshape(resamples, blocks_per_sample * block)[:, :n]


def verdict_from_interval(lower: float, upper: float) -> str:
    """"win" when the interval lies entirely below zero, "loss" when entirely above, otherwise "tie" (the interval holds zero)."""
    if upper < 0:
        return "win"
    if lower > 0:
        return "loss"
    return "tie"


def paired_interval(difference: np.ndarray, *, block: int, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> PairedInterval:
    """Mean of ``difference`` (in time order) with the 90% percentile interval of its circular block bootstrap and the verdict."""
    values = np.asarray(difference, dtype=float)
    indices = circular_block_indices(len(values), block, resamples, seed)
    means = values[indices].mean(axis=1)
    lower, upper = np.percentile(means, INTERVAL_PERCENT)
    return PairedInterval(mean=float(values.mean()), lower=float(lower), upper=float(upper), origins=len(values), block=block, verdict=verdict_from_interval(lower, upper))


def verdict_text(result: PairedInterval, *, unit: str = "points") -> str:
    """One sentence for a table: the mean difference with its interval, the block length and what the interval says."""
    reading = {"win": "the interval is entirely below zero: a win", "tie": "the interval holds zero: a tie", "loss": "the interval is entirely above zero: a loss"}[result.verdict]
    return f"mean difference {result.mean:+.2f} {unit} (90% interval {result.lower:+.2f} to {result.upper:+.2f}; {result.origins} origins, block {result.block}); {reading}"
