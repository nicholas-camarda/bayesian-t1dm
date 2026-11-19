from types import SimpleNamespace

import numpy as np

from bayesian_t1dm.latent_meal_trajectory import MealWindowTrajectory, TrajectoryEstimatorConfig
from bayesian_t1dm.meal_validation import _predict


def test_held_out_predictions_are_independent_of_held_out_glucose():
    times = np.arange(0, 185, 5)
    row = SimpleNamespace(cf=50, icr=7, stated_carbs=30, premeal_glucose=110)
    pool = SimpleNamespace(sensitivity_ratio=1.1, between_meal_log_sd=0.1)
    first = MealWindowTrajectory(times, np.full(len(times), 110), times / 120)
    second = MealWindowTrajectory(times, np.full(len(times), 200), times / 120)
    config = TrajectoryEstimatorConfig()
    left = _predict(first, row, 1, pool, np.random.default_rng(4), config)
    right = _predict(second, row, 1, pool, np.random.default_rng(4), config)
    np.testing.assert_array_equal(left, right)
    assert left.shape[1] == len(times)
