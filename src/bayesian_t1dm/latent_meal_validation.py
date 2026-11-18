"""Synthetic truth-recovery validation for the latent carbohydrate estimators.

Real Tandem exports carry no carbohydrate truth, so the only way to know whether a latent
carb estimator works is to simulate meals where the truth is known and check what comes
back. This module simulates post-meal CGM traces under a simple, standard physiological
model (linear carb and insulin effects, the repo's own insulin action curve) and scores
three estimators against the simulated truth:

- ``bolus_implied``: ``bolus_units * ICR``. This is what you get for free and is the bar
  any estimator has to clear.
- ``summary_heuristic``: the original peak/AUC grid posterior.
- ``trajectory``: the full-trace posterior in :mod:`bayesian_t1dm.latent_meal_trajectory`.

Scenarios deliberately break the estimator's assumptions one at a time (absorption shape,
pre-bolus timing, wrong therapy settings, drift) and all together. A scenario where the
simulator and the estimator share every assumption proves little; the misspecified ones
are the evidence.

This is a statement about the estimator under simulation, not about real meals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .insulin import insulin_action_curve
from .latent_meal_trajectory import (
    MealWindowTrajectory,
    absorbed_fraction,
    absorbed_units_from_deliveries,
    observe_carb_sensitivity,
    pool_effective_icr,
)


@dataclass(frozen=True)
class MealSimulationScenario:
    name: str
    description: str
    # "s_curve": symmetric absorption finishing at ``duration``; "gamma": front-loaded with
    # a long tail; "biphasic": 60% fast plus 40% slow and delayed (fat/protein-like).
    absorption: str = "s_curve"
    prebolus_max_minutes: float = 0.0
    settings_error_log_sd: float = 0.0
    drift_sd_mgdl_per_hour: float = 0.0
    noise_sd_mgdl: float = 4.0
    carb_count_error_log_sd: float = 0.35
    # Simulate an automated insulin delivery controller that raises basal when glucose
    # runs high and suspends it when glucose runs low.
    closed_loop_basal: bool = False


DEFAULT_SCENARIOS: tuple[MealSimulationScenario, ...] = (
    MealSimulationScenario("matched_assumptions", "Symmetric absorption, exact settings, no drift. Easiest case."),
    MealSimulationScenario("long_tail_absorption", "Front-loaded absorption with a long tail.", absorption="gamma"),
    MealSimulationScenario("biphasic_absorption", "60% fast, 40% slow and delayed; some carbohydrate lands after the window.", absorption="biphasic"),
    MealSimulationScenario("prebolus_timing", "Meal starts 0-20 minutes after the bolus.", prebolus_max_minutes=20.0),
    MealSimulationScenario("wrong_settings", "True I:C and correction factor each differ from programmed by ~15%.", settings_error_log_sd=0.15),
    MealSimulationScenario("baseline_drift", "Unmodelled linear drift, SD 10 mg/dL per hour.", drift_sd_mgdl_per_hour=10.0),
    MealSimulationScenario(
        "closed_loop_basal",
        "A controller raises basal above 160 mg/dL and suspends it below 90 mg/dL; the estimator is given the basal deviation.",
        closed_loop_basal=True,
    ),
    MealSimulationScenario(
        "everything_at_once",
        "Long-tail absorption, pre-bolus, wrong settings, drift and noisier CGM together.",
        absorption="gamma",
        prebolus_max_minutes=20.0,
        settings_error_log_sd=0.15,
        drift_sd_mgdl_per_hour=10.0,
        noise_sd_mgdl=6.0,
    ),
)


def _s_curve(minutes: np.ndarray, duration: float) -> np.ndarray:
    x = np.clip(np.asarray(minutes, dtype=float) / float(duration), 0.0, 1.0)
    return np.where(x < 0.5, 2.0 * x**2, 1.0 - 2.0 * (1.0 - x) ** 2)


def _true_absorbed_fraction(minutes: np.ndarray, *, kind: str, duration: float) -> np.ndarray:
    if kind == "s_curve":
        return _s_curve(minutes, duration)
    if kind == "gamma":
        return absorbed_fraction(minutes, shape=2.2, mean_minutes=0.44 * duration)
    if kind == "biphasic":
        return 0.6 * _s_curve(minutes, 0.5 * duration) + 0.4 * _s_curve(np.asarray(minutes, dtype=float) - 30.0, 1.3 * duration)
    raise ValueError(f"Unknown absorption kind: {kind}")


def _insulin_absorbed_fraction(minutes: np.ndarray, *, duration_minutes: int = 300) -> np.ndarray:
    curve = insulin_action_curve(1.0, duration_minutes=duration_minutes)
    remaining = np.interp(np.clip(minutes, 0.0, duration_minutes), curve["minutes"], curve["iob_fraction"])
    return np.where(np.asarray(minutes) < 0, 0.0, 1.0 - remaining)


@dataclass(frozen=True)
class SimulatedMeal:
    trajectory: MealWindowTrajectory
    true_carbs: float
    bolus_units: float
    programmed_icr: float
    programmed_correction_factor: float
    premeal_glucose: float
    # Insulin absorbed from the meal bolus alone, ignoring any basal modulation.
    bolus_only_absorbed_units: np.ndarray | None = None

    @property
    def bolus_implied_carbs(self) -> float:
        return self.bolus_units * self.programmed_icr


def simulate_meal(
    rng: np.random.Generator,
    scenario: MealSimulationScenario,
    *,
    programmed_icr: float = 10.0,
    programmed_correction_factor: float = 40.0,
    window_minutes: int = 180,
    premeal_glucose: float = 110.0,
    true_icr_multiplier: float = 1.0,
) -> SimulatedMeal:
    minutes = np.arange(0.0, window_minutes + 5.0, 5.0)
    true_icr = programmed_icr * float(true_icr_multiplier) * float(np.exp(rng.normal(0.0, scenario.settings_error_log_sd)))
    true_cf = programmed_correction_factor * float(np.exp(rng.normal(0.0, scenario.settings_error_log_sd)))
    true_carbs = float(rng.uniform(15.0, 100.0))
    # The person boluses for what they *think* they are eating.
    bolus_units = true_carbs * float(rng.lognormal(0.0, scenario.carb_count_error_log_sd)) / programmed_icr
    duration = float(rng.uniform(120.0, 240.0))
    delay = float(rng.uniform(0.0, scenario.prebolus_max_minutes)) if scenario.prebolus_max_minutes > 0 else 0.0
    drift = float(rng.normal(0.0, scenario.drift_sd_mgdl_per_hour)) if scenario.drift_sd_mgdl_per_hour > 0 else 0.0

    noise = np.zeros(minutes.size)
    innovation_sd = scenario.noise_sd_mgdl * np.sqrt(1.0 - 0.7**2)
    for index in range(1, minutes.size):
        noise[index] = 0.7 * noise[index - 1] + rng.normal(0.0, innovation_sd)

    insulin_fraction = _insulin_absorbed_fraction(minutes)
    bolus_absorbed = bolus_units * insulin_fraction
    carb_effect = (true_cf / true_icr) * true_carbs * _true_absorbed_fraction(minutes - delay, kind=scenario.absorption, duration=duration)
    passive = premeal_glucose + carb_effect - true_cf * bolus_absorbed + drift * minutes / 60.0 + noise
    basal_absorbed = np.zeros(minutes.size)
    if scenario.closed_loop_basal:
        scheduled_units_per_hour = 0.8
        excess_units = np.zeros(minutes.size)
        glucose = passive.copy()
        for index in range(1, minutes.size):
            previous = glucose[index - 1]
            if previous > 160.0:
                rate = min(0.03 * (previous - 160.0), 3.0)
            elif previous < 90.0:
                rate = -scheduled_units_per_hour
            else:
                rate = 0.0
            excess_units[index - 1] = rate * 5.0 / 60.0
            basal_absorbed[index] = absorbed_units_from_deliveries(minutes[:index], excess_units[:index], minutes[index : index + 1])[0]
            glucose[index] = passive[index] - true_cf * basal_absorbed[index]
    else:
        glucose = passive
    return SimulatedMeal(
        trajectory=MealWindowTrajectory(minutes=minutes, glucose=glucose, insulin_absorbed_units=bolus_absorbed + basal_absorbed),
        true_carbs=true_carbs,
        bolus_units=bolus_units,
        programmed_icr=programmed_icr,
        programmed_correction_factor=programmed_correction_factor,
        premeal_glucose=premeal_glucose,
        bolus_only_absorbed_units=bolus_absorbed,
    )


def _window_metric_row(meal: SimulatedMeal, meal_id: str) -> pd.Series:
    glucose = meal.trajectory.glucose
    baseline = meal.premeal_glucose
    return pd.Series(
        {
            "meal_id": meal_id,
            "segment": "morning",
            "bolus_units": meal.bolus_units,
            "premeal_iob": 0.0,
            "premeal_glucose": baseline,
            "observed_peak_delta": float(glucose.max() - baseline),
            "observed_auc_above_baseline": float(np.clip(glucose - baseline, 0.0, None).sum() * 5.0 / 60.0),
            "observed_peak_glucose": float(glucose.max()),
            "observed_mean_glucose_180m": float(glucose.mean()),
            "cgm_coverage_fraction": 1.0,
            "explicit_carb_source_available": 0,
            "stated_carbs": np.nan,
            "analysis_tier": "clean_reference",
            "source_quality_grade": "good",
        }
    )


@dataclass(frozen=True)
class LatentMealRecoveryResult:
    summary: pd.DataFrame
    meals: pd.DataFrame
    n_meals_per_scenario: int
    seed: int

    def passed(self, *, min_coverage: float = 0.75) -> bool:
        """The trajectory estimator beats the free baseline and keeps honest intervals in the composite scenario."""
        composite = self.summary.loc[self.summary["scenario"].eq("everything_at_once")].set_index("estimator")
        if not {"trajectory", "bolus_implied"}.issubset(composite.index):
            return False
        return bool(
            composite.loc["trajectory", "mae_grams"] < composite.loc["bolus_implied", "mae_grams"]
            and composite.loc["trajectory", "coverage_90"] >= min_coverage
        )


def run_latent_meal_truth_recovery(
    *,
    n_meals_per_scenario: int = 200,
    seed: int = 20260403,
    scenarios: tuple[MealSimulationScenario, ...] = DEFAULT_SCENARIOS,
) -> LatentMealRecoveryResult:
    # Imported here: therapy_research imports the trajectory module, and this harness sits on top of both.
    from .therapy_research import LatentMealTherapySettings, _latent_carb_posterior_for_window

    if n_meals_per_scenario <= 0:
        raise ValueError("n_meals_per_scenario must be positive")
    meal_rows: list[dict[str, object]] = []
    for scenario in scenarios:
        # Every scenario sees the same meals (same seed); only the scenario's own
        # perturbations differ, so rows are comparable across scenarios.
        rng = np.random.default_rng(seed)
        for index in range(n_meals_per_scenario):
            meal = simulate_meal(rng, scenario)
            settings = LatentMealTherapySettings(
                icr_grams_per_unit={"morning": meal.programmed_icr},
                correction_factor_mgdl_per_unit={"morning": meal.programmed_correction_factor},
                target_glucose_mgdl=110.0,
                insulin_action_minutes=300,
                carb_absorption_minutes_prior=180,
                max_plausible_meal_carbs=180.0,
            )
            row = _window_metric_row(meal, f"{scenario.name}_{index:04d}")
            heuristic, _ = _latent_carb_posterior_for_window(row, settings=settings, params={})
            trajectory, _ = _latent_carb_posterior_for_window(row, settings=settings, params={}, trajectory=meal.trajectory)
            estimates = {
                "bolus_implied": (meal.bolus_implied_carbs, np.nan, np.nan),
                "summary_heuristic": (heuristic["latent_carb_median"], heuristic["latent_carb_lower_90"], heuristic["latent_carb_upper_90"]),
                "trajectory": (trajectory["latent_carb_median"], trajectory["latent_carb_lower_90"], trajectory["latent_carb_upper_90"]),
            }
            for estimator, (estimate, lower, upper) in estimates.items():
                meal_rows.append(
                    {
                        "scenario": scenario.name,
                        "meal_index": index,
                        "estimator": estimator,
                        "true_carbs": meal.true_carbs,
                        "bolus_implied_carbs": meal.bolus_implied_carbs,
                        "estimate": float(estimate),
                        "lower_90": float(lower),
                        "upper_90": float(upper),
                    }
                )
    meals = pd.DataFrame(meal_rows)
    meals["error"] = meals["estimate"] - meals["true_carbs"]
    meals["covered"] = np.where(
        meals["lower_90"].notna(),
        (meals["true_carbs"].ge(meals["lower_90"]) & meals["true_carbs"].le(meals["upper_90"])).astype(float),
        np.nan,
    )
    # How much of the carb-counting error (truth minus what was bolused for) the estimator recovers.
    meals["counting_error"] = meals["true_carbs"] - meals["bolus_implied_carbs"]
    meals["estimated_counting_error"] = meals["estimate"] - meals["bolus_implied_carbs"]

    summary_rows: list[dict[str, object]] = []
    for (scenario_name, estimator), group in meals.groupby(["scenario", "estimator"], sort=False):
        if estimator == "bolus_implied":
            recovery_slope = 0.0
        else:
            recovery_slope = float(np.polyfit(group["counting_error"], group["estimated_counting_error"], 1)[0])
        summary_rows.append(
            {
                "scenario": scenario_name,
                "estimator": estimator,
                "n_meals": int(len(group)),
                "mae_grams": float(group["error"].abs().mean()),
                "bias_grams": float(group["error"].mean()),
                "coverage_90": float(group["covered"].mean()) if group["covered"].notna().any() else float("nan"),
                "mean_interval_width_90": float((group["upper_90"] - group["lower_90"]).mean()),
                "counting_error_recovery_slope": recovery_slope,
            }
        )
    return LatentMealRecoveryResult(
        summary=pd.DataFrame(summary_rows),
        meals=meals,
        n_meals_per_scenario=n_meals_per_scenario,
        seed=seed,
    )


LOGGED_MEAL_SCENARIO = MealSimulationScenario(
    "logged_meals",
    "Meals with logged carbs (10% logging error), long-tail absorption, pre-bolus, drift and noisy CGM. Correction factor exact.",
    absorption="gamma",
    prebolus_max_minutes=20.0,
    drift_sd_mgdl_per_hour=10.0,
    noise_sd_mgdl=6.0,
    carb_count_error_log_sd=0.10,
)


def run_effective_icr_recovery(
    *,
    n_cohorts: int = 40,
    meals_per_cohort: int = 20,
    true_icr_multipliers: tuple[float, ...] = (0.8, 1.0, 1.25),
    programmed_icr: float = 10.0,
    programmed_correction_factor: float = 40.0,
    seed: int = 20260403,
    scenario: MealSimulationScenario = LOGGED_MEAL_SCENARIO,
) -> pd.DataFrame:
    """Can a cohort of logged meals recover an I:C ratio that differs from the programmed one?

    Each cohort is ``meals_per_cohort`` simulated meals whose carbs were logged (and bolused
    for with the programmed ratio) while the body's true ratio is
    ``programmed_icr * multiplier``. Reports how close the pooled estimate lands, how often
    its 90% interval contains the truth, and how often it calls a difference.
    """
    if n_cohorts <= 0 or meals_per_cohort <= 0:
        raise ValueError("n_cohorts and meals_per_cohort must be positive")
    rows: list[dict[str, object]] = []
    for multiplier in true_icr_multipliers:
        rng = np.random.default_rng(seed)
        true_icr = programmed_icr * float(multiplier)
        estimates: list[float] = []
        covered = 0
        flagged = 0
        usable_meals = 0
        for _ in range(n_cohorts):
            observations = []
            for _ in range(meals_per_cohort):
                meal = simulate_meal(
                    rng,
                    scenario,
                    programmed_icr=programmed_icr,
                    programmed_correction_factor=programmed_correction_factor,
                    true_icr_multiplier=float(multiplier),
                )
                observation = observe_carb_sensitivity(
                    meal.trajectory,
                    # What was entered into the pump: the bolus calculator divided it by the programmed ratio.
                    stated_carbs=meal.bolus_implied_carbs,
                    icr_grams_per_unit=programmed_icr,
                    correction_factor_mgdl_per_unit=programmed_correction_factor,
                    baseline_glucose=meal.premeal_glucose,
                )
                if observation is not None:
                    observations.append(observation)
            usable_meals += len(observations)
            estimate = pool_effective_icr(observations, programmed_icr=programmed_icr)
            if not np.isfinite(estimate.effective_icr):
                continue
            estimates.append(estimate.effective_icr)
            covered += int(estimate.effective_icr_lower_90 <= true_icr <= estimate.effective_icr_upper_90)
            flagged += int(estimate.status != "consistent_with_programmed")
        completed = len(estimates)
        rows.append(
            {
                "true_icr": true_icr,
                "programmed_icr": programmed_icr,
                "n_cohorts": completed,
                "meals_per_cohort": meals_per_cohort,
                "mean_usable_meals": usable_meals / n_cohorts,
                "mean_estimated_icr": float(np.mean(estimates)) if estimates else float("nan"),
                "sd_estimated_icr": float(np.std(estimates)) if estimates else float("nan"),
                "coverage_90": covered / completed if completed else float("nan"),
                "flagged_as_different": flagged / completed if completed else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def render_effective_icr_recovery_markdown(recovery: pd.DataFrame) -> str:
    lines = [
        "# Effective I:C From Logged Meals: Synthetic Recovery",
        "",
        "Each cohort is a set of simulated meals whose carbs were logged and bolused for with the programmed I:C,",
        "while the simulated body responds with a different true I:C. The pooled estimate should land on the truth.",
        "",
        "| true I:C | programmed | cohorts | meals / cohort | mean estimate | SD | 90% coverage | flagged as different |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in recovery.itertuples(index=False):
        lines.append(
            f"| {row.true_icr:.1f} | {row.programmed_icr:.1f} | {int(row.n_cohorts)} | {int(row.meals_per_cohort)} | "
            f"{row.mean_estimated_icr:.2f} | {row.sd_estimated_icr:.2f} | {row.coverage_90:.2f} | {row.flagged_as_different:.2f} |"
        )
    lines.extend(
        [
            "",
            "- `flagged as different` in the row where true equals programmed is the false-alarm rate (nominal 0.10).",
            "- The estimate assumes the programmed correction factor is right. If it is off by x%, the I:C estimate is off by about x%.",
            "- Simulation only: see the caveats in the latent carb recovery report.",
            "",
        ]
    )
    return "\n".join(lines)


def render_latent_meal_recovery_markdown(
    result: LatentMealRecoveryResult,
    scenarios: tuple[MealSimulationScenario, ...] = DEFAULT_SCENARIOS,
) -> str:
    descriptions = {scenario.name: scenario.description for scenario in scenarios}
    lines = [
        "# Latent Carb Estimator: Synthetic Truth Recovery",
        "",
        f"- meals per scenario: `{result.n_meals_per_scenario}`",
        f"- seed: `{result.seed}`",
        f"- composite check (trajectory beats bolus-implied and keeps >=75% coverage when everything is wrong at once): `{'pass' if result.passed() else 'FAIL'}`",
        "",
        "Simulated meals have known carbohydrate. Each is bolused for a mis-counted amount (log-SD 0.35), so",
        "`bolus_implied` (bolus x programmed I:C) is wrong by construction; the question is how much of that",
        "error each estimator recovers from the glucose trace.",
        "",
        "- `mae_grams` / `bias_grams`: error of the posterior median against simulated truth.",
        "- `coverage_90`: share of meals whose truth falls inside the 90% interval. Nominal is 0.90.",
        "- `counting_error_recovery_slope`: regression of (estimate - bolus-implied) on (truth - bolus-implied).",
        "  0 means the estimator just echoes the bolus; 1 means it fully recovers the counting error.",
        "",
    ]
    for scenario_name, group in result.summary.groupby("scenario", sort=False):
        lines.extend(
            [
                f"## {scenario_name}",
                "",
                descriptions.get(str(scenario_name), ""),
                "",
                "| estimator | MAE (g) | bias (g) | 90% coverage | mean 90% width (g) | recovery slope |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in group.itertuples(index=False):
            coverage = "n/a" if not np.isfinite(row.coverage_90) else f"{row.coverage_90:.2f}"
            width = "n/a" if not np.isfinite(row.mean_interval_width_90) else f"{row.mean_interval_width_90:.1f}"
            lines.append(
                f"| {row.estimator} | {row.mae_grams:.1f} | {row.bias_grams:+.1f} | {coverage} | {width} | {row.counting_error_recovery_slope:.2f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Reading this",
            "",
            "- This is evidence about the estimator under a simulator, not about real meals. The simulator is linear,",
            "  has no exercise and no sensor lag, and its closed-loop controller is a threshold rule, not Control-IQ.",
            "- Glucose identifies `carbs x CF / ICR`, not carbs. The `wrong_settings` scenario shows the cost: the",
            "  estimate is a carb-equivalent under the programmed settings.",
            "- Carbohydrate absorbed after the 180-minute window is invisible; `biphasic_absorption` shows the",
            "  resulting low bias.",
            "",
        ]
    )
    return "\n".join(lines)


def write_latent_meal_recovery_artifacts(
    result: LatentMealRecoveryResult,
    report_dir: str | Path,
    *,
    icr_recovery: pd.DataFrame | None = None,
) -> dict[str, Path]:
    root = Path(report_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": root / "latent_meal_truth_recovery.md",
        "summary": root / "latent_meal_truth_recovery_summary.csv",
        "meals": root / "latent_meal_truth_recovery_meals.csv",
    }
    paths["report"].write_text(render_latent_meal_recovery_markdown(result), encoding="utf-8")
    result.summary.to_csv(paths["summary"], index=False)
    result.meals.to_csv(paths["meals"], index=False)
    if icr_recovery is not None:
        paths["icr_report"] = root / "effective_icr_recovery.md"
        paths["icr_summary"] = root / "effective_icr_recovery.csv"
        paths["icr_report"].write_text(render_effective_icr_recovery_markdown(icr_recovery), encoding="utf-8")
        icr_recovery.to_csv(paths["icr_summary"], index=False)
    return paths
