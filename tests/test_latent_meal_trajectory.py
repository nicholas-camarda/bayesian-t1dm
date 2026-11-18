from __future__ import annotations

import numpy as np
import pytest

from bayesian_t1dm.latent_meal_trajectory import (
    MealWindowTrajectory,
    TrajectoryEstimatorConfig,
    absorbed_fraction,
    estimate_carbs_from_trajectory,
)
from bayesian_t1dm.latent_meal_validation import (
    DEFAULT_SCENARIOS,
    MealSimulationScenario,
    _insulin_absorbed_fraction,
    render_latent_meal_recovery_markdown,
    run_latent_meal_truth_recovery,
    simulate_meal,
    write_latent_meal_recovery_artifacts,
)

ICR = 10.0
CF = 40.0


def _noiseless_meal(carbs: float, bolus_units: float, *, mean_minutes: float = 70.0, shape: float = 2.5) -> MealWindowTrajectory:
    minutes = np.arange(0.0, 185.0, 5.0)
    insulin = bolus_units * _insulin_absorbed_fraction(minutes)
    glucose = 110.0 + (CF / ICR) * carbs * absorbed_fraction(minutes, shape=shape, mean_minutes=mean_minutes) - CF * insulin
    return MealWindowTrajectory(minutes=minutes, glucose=glucose, insulin_absorbed_units=insulin)


def _estimate(trajectory: MealWindowTrajectory, *, bolus_units: float, **kwargs):
    return estimate_carbs_from_trajectory(
        trajectory,
        icr_grams_per_unit=ICR,
        correction_factor_mgdl_per_unit=CF,
        bolus_implied_carbs=bolus_units * ICR,
        baseline_glucose=110.0,
        **kwargs,
    )


def test_absorbed_fraction_is_a_monotone_cdf_with_delay():
    minutes = np.arange(0.0, 600.0, 5.0)
    curve = absorbed_fraction(minutes, shape=2.5, mean_minutes=80.0, delay_minutes=15.0)

    assert curve[0] == 0.0
    assert np.all(curve[minutes <= 15.0] == 0.0)
    assert np.all(np.diff(curve) >= 0.0)
    assert curve[-1] == pytest.approx(1.0, abs=1e-3)


def test_trajectory_rejects_mismatched_arrays():
    with pytest.raises(ValueError, match="equal length"):
        MealWindowTrajectory(minutes=np.arange(3.0), glucose=np.arange(4.0), insulin_absorbed_units=np.arange(3.0))


@pytest.mark.parametrize(
    "minutes",
    [
        np.arange(0.0, 60.0, 5.0),  # too short
        np.arange(30.0, 215.0, 5.0),  # starts too late
        np.array([0.0, 5.0, 5.0] + list(np.arange(10.0, 185.0, 5.0))),  # duplicate timestamp
    ],
)
def test_trajectory_usability_guards(minutes):
    trajectory = MealWindowTrajectory(minutes=minutes, glucose=np.full(minutes.size, 110.0), insulin_absorbed_units=np.zeros(minutes.size))
    assert not trajectory.is_usable()


def test_recovers_undercounted_meal_from_noiseless_trace():
    # 70 g eaten but only bolused for 40 g: the trace has to pull the estimate well above the prior.
    posterior = _estimate(_noiseless_meal(70.0, 4.0), bolus_units=4.0)

    assert posterior.carb_median == pytest.approx(70.0, abs=8.0)
    assert posterior.carb_lower_90 <= 70.0 <= posterior.carb_upper_90
    assert posterior.carb_lower_90 > 40.0
    assert posterior.prior_contraction > 0.5
    assert not posterior.poor_fit


def test_recovers_overcounted_meal_from_noiseless_trace():
    # 30 g eaten but bolused for 60 g.
    posterior = _estimate(_noiseless_meal(30.0, 6.0), bolus_units=6.0)

    assert posterior.carb_median == pytest.approx(30.0, abs=8.0)
    assert posterior.carb_upper_90 < 60.0


def test_more_absorbed_insulin_with_same_glucose_implies_more_carbs():
    base = _noiseless_meal(50.0, 5.0)
    more_insulin = MealWindowTrajectory(
        minutes=base.minutes,
        glucose=base.glucose,
        insulin_absorbed_units=base.insulin_absorbed_units * 1.4,
    )

    assert _estimate(more_insulin, bolus_units=5.0).carb_median > _estimate(base, bolus_units=5.0).carb_median


def test_flat_trace_with_no_insulin_estimates_near_zero_carbs():
    minutes = np.arange(0.0, 185.0, 5.0)
    flat = MealWindowTrajectory(minutes=minutes, glucose=np.full(minutes.size, 110.0), insulin_absorbed_units=np.zeros(minutes.size))

    posterior = _estimate(flat, bolus_units=0.0)

    assert posterior.carb_median < 5.0


def test_uncertainty_multiplier_widens_the_interval():
    trajectory = _noiseless_meal(50.0, 5.0)

    tight = _estimate(trajectory, bolus_units=5.0)
    loose = _estimate(trajectory, bolus_units=5.0, uncertainty_multiplier=1.7)

    assert (loose.carb_upper_90 - loose.carb_lower_90) > (tight.carb_upper_90 - tight.carb_lower_90)


def test_sampling_grid_is_sorted_and_normalized():
    posterior = _estimate(_noiseless_meal(50.0, 5.0), bolus_units=5.0)

    assert np.all(np.diff(posterior.grid_carbs) >= 0.0)
    assert posterior.grid_weights.sum() == pytest.approx(1.0)
    assert posterior.grid_carbs.shape == posterior.grid_absorption_minutes.shape == posterior.grid_weights.shape


def test_irregular_time_grid_is_supported():
    full = _noiseless_meal(60.0, 5.0)
    keep = np.ones(full.n_points, dtype=bool)
    keep[[7, 8, 20]] = False  # CGM dropouts
    gappy = MealWindowTrajectory(minutes=full.minutes[keep], glucose=full.glucose[keep], insulin_absorbed_units=full.insulin_absorbed_units[keep])

    assert gappy.is_usable()
    assert _estimate(gappy, bolus_units=5.0).carb_median == pytest.approx(_estimate(full, bolus_units=5.0).carb_median, abs=3.0)


def test_invalid_settings_are_rejected():
    trajectory = _noiseless_meal(50.0, 5.0)
    with pytest.raises(ValueError):
        estimate_carbs_from_trajectory(trajectory, icr_grams_per_unit=0.0, correction_factor_mgdl_per_unit=CF, bolus_implied_carbs=50.0)
    with pytest.raises(ValueError):
        estimate_carbs_from_trajectory(trajectory, icr_grams_per_unit=ICR, correction_factor_mgdl_per_unit=CF, bolus_implied_carbs=50.0, max_carbs=0.0)


def test_poor_fit_is_flagged_when_trace_is_not_meal_shaped():
    minutes = np.arange(0.0, 185.0, 5.0)
    sawtooth = 110.0 + 60.0 * ((minutes // 15) % 2)
    trajectory = MealWindowTrajectory(minutes=minutes, glucose=sawtooth, insulin_absorbed_units=np.zeros(minutes.size))

    assert _estimate(trajectory, bolus_units=0.0, config=TrajectoryEstimatorConfig()).poor_fit


def test_simulated_meal_is_reproducible_and_physical():
    scenario = DEFAULT_SCENARIOS[0]
    first = simulate_meal(np.random.default_rng(5), scenario)
    second = simulate_meal(np.random.default_rng(5), scenario)

    assert first.true_carbs == second.true_carbs
    np.testing.assert_allclose(first.trajectory.glucose, second.trajectory.glucose)
    assert first.trajectory.is_usable()
    assert np.all(np.diff(first.trajectory.insulin_absorbed_units) >= 0.0)
    assert first.trajectory.insulin_absorbed_units[-1] < first.bolus_units


def test_truth_recovery_trajectory_beats_bolus_implied_and_heuristic():
    scenarios = tuple(scenario for scenario in DEFAULT_SCENARIOS if scenario.name in {"matched_assumptions", "everything_at_once"})
    result = run_latent_meal_truth_recovery(n_meals_per_scenario=60, seed=11, scenarios=scenarios)

    assert result.passed()
    for scenario_name in ["matched_assumptions", "everything_at_once"]:
        scores = result.summary.loc[result.summary["scenario"].eq(scenario_name)].set_index("estimator")
        assert scores.loc["trajectory", "mae_grams"] < scores.loc["bolus_implied", "mae_grams"]
        assert scores.loc["trajectory", "mae_grams"] < scores.loc["summary_heuristic", "mae_grams"]
        assert scores.loc["trajectory", "coverage_90"] >= 0.8
        # The estimator has to recover most of the carb-counting error, not echo the bolus.
        assert scores.loc["trajectory", "counting_error_recovery_slope"] > 0.6


def test_truth_recovery_documents_the_late_absorption_blind_spot():
    # Carbohydrate that lands after the window cannot be seen. This test pins the known
    # limitation so that a change which hides it (or fixes it) is noticed.
    scenarios = tuple(scenario for scenario in DEFAULT_SCENARIOS if scenario.name == "biphasic_absorption")
    result = run_latent_meal_truth_recovery(n_meals_per_scenario=60, seed=11, scenarios=scenarios)

    trajectory = result.summary.set_index("estimator").loc["trajectory"]
    assert trajectory["bias_grams"] < -5.0


def test_truth_recovery_is_deterministic_and_writes_artifacts(tmp_path):
    scenarios = (MealSimulationScenario("matched_assumptions", "easy"),)
    first = run_latent_meal_truth_recovery(n_meals_per_scenario=8, seed=3, scenarios=scenarios)
    second = run_latent_meal_truth_recovery(n_meals_per_scenario=8, seed=3, scenarios=scenarios)

    assert first.summary.equals(second.summary)
    assert not first.passed()  # the composite scenario was not run

    paths = write_latent_meal_recovery_artifacts(first, tmp_path / "validation")
    assert all(path.exists() for path in paths.values())
    assert "matched_assumptions" in render_latent_meal_recovery_markdown(first, scenarios)


def test_truth_recovery_rejects_empty_runs():
    with pytest.raises(ValueError):
        run_latent_meal_truth_recovery(n_meals_per_scenario=0)


def test_absorbed_units_from_deliveries_matches_bolus_curve_and_handles_negatives():
    from bayesian_t1dm.latent_meal_trajectory import absorbed_units_from_deliveries

    at = np.arange(0.0, 305.0, 5.0)
    single = absorbed_units_from_deliveries(np.array([0.0]), np.array([2.0]), at)
    np.testing.assert_allclose(single, 2.0 * _insulin_absorbed_fraction(at), atol=1e-9)
    assert single[-1] == pytest.approx(2.0, abs=1e-6)

    # Nothing is absorbed before it is delivered, and withheld insulin counts negative.
    later = absorbed_units_from_deliveries(np.array([60.0]), np.array([-0.5]), at)
    assert np.all(later[at <= 60.0] == 0.0)
    assert later[-1] < 0.0

    assert np.all(absorbed_units_from_deliveries(np.array([]), np.array([]), at) == 0.0)


def test_closed_loop_meals_are_estimated_better_when_basal_deviation_is_counted():
    scenario = next(item for item in DEFAULT_SCENARIOS if item.name == "closed_loop_basal")
    rng = np.random.default_rng(4)
    with_basal: list[float] = []
    bolus_only: list[float] = []
    modulated = 0
    for _ in range(80):
        meal = simulate_meal(rng, scenario)
        extra = meal.trajectory.insulin_absorbed_units - meal.bolus_only_absorbed_units
        if np.abs(extra).max() < 0.3:
            continue
        modulated += 1
        kwargs = dict(bolus_units=meal.bolus_units)
        with_basal.append(abs(_estimate(meal.trajectory, **kwargs).carb_median - meal.true_carbs))
        ignoring = MealWindowTrajectory(
            minutes=meal.trajectory.minutes,
            glucose=meal.trajectory.glucose,
            insulin_absorbed_units=meal.bolus_only_absorbed_units,
        )
        bolus_only.append(abs(_estimate(ignoring, **kwargs).carb_median - meal.true_carbs))

    assert modulated >= 10
    assert np.mean(with_basal) < np.mean(bolus_only)
