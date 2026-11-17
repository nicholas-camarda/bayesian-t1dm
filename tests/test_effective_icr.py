"""End-to-end and unit tests for estimating an effective I:C ratio from logged meals."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from bayesian_t1dm.features import FeatureConfig
from bayesian_t1dm.health_auto_export import build_analysis_ready_health_dataset
from bayesian_t1dm.ingest import IngestedData
from bayesian_t1dm.latent_meal_trajectory import (
    CarbSensitivityObservation,
    MealWindowTrajectory,
    absorbed_fraction,
    observe_carb_sensitivity,
    pool_effective_icr,
)
from bayesian_t1dm.latent_meal_validation import (
    _insulin_absorbed_fraction,
    render_effective_icr_recovery_markdown,
    run_effective_icr_recovery,
)
from bayesian_t1dm.review import write_latent_meal_review_html
from bayesian_t1dm.therapy_research import (
    LatentMealTherapySettings,
    parse_therapy_segments,
    run_latent_meal_icr_research,
    write_latent_meal_research_artifacts,
)

PROGRAMMED_ICR = 10.0
CORRECTION_FACTOR = 40.0


def _observation(log_ratio: float, sd: float = 0.1) -> CarbSensitivityObservation:
    return CarbSensitivityObservation(log_ratio=log_ratio, log_ratio_sd=sd, carb_equivalent=50.0, stated_carbs=50.0, fit_rmse_mgdl=3.0)


def test_pool_reports_insufficient_meals_below_minimum():
    estimate = pool_effective_icr([_observation(0.2)] * 4, programmed_icr=PROGRAMMED_ICR)

    assert estimate.status == "insufficient_logged_meals"
    assert estimate.n_meals == 4
    assert np.isnan(estimate.effective_icr)


def test_pool_recovers_a_consistent_ratio_shift():
    rng = np.random.default_rng(0)
    true_ratio = 1.25  # carbs hit 25% harder than programmed
    observations = [_observation(float(np.log(true_ratio) + rng.normal(0.0, 0.08)), sd=0.08) for _ in range(20)]

    estimate = pool_effective_icr(observations, programmed_icr=PROGRAMMED_ICR)

    assert estimate.status == "carbs_hit_harder_than_programmed"
    assert estimate.sensitivity_ratio == pytest.approx(true_ratio, rel=0.05)
    assert estimate.effective_icr == pytest.approx(PROGRAMMED_ICR / true_ratio, rel=0.05)
    assert estimate.effective_icr_lower_90 < PROGRAMMED_ICR / true_ratio < estimate.effective_icr_upper_90
    assert estimate.effective_icr_upper_90 < PROGRAMMED_ICR


def test_pool_is_consistent_when_ratio_is_one_and_widens_with_meal_to_meal_spread():
    rng = np.random.default_rng(1)
    calm = pool_effective_icr([_observation(float(rng.normal(0.0, 0.05)), sd=0.05) for _ in range(15)], programmed_icr=PROGRAMMED_ICR)
    rng = np.random.default_rng(1)
    noisy = pool_effective_icr([_observation(float(rng.normal(0.0, 0.40)), sd=0.05) for _ in range(15)], programmed_icr=PROGRAMMED_ICR)

    assert calm.status == "consistent_with_programmed"
    assert noisy.between_meal_log_sd > calm.between_meal_log_sd
    assert (noisy.effective_icr_upper_90 - noisy.effective_icr_lower_90) > (calm.effective_icr_upper_90 - calm.effective_icr_lower_90)


def test_pool_flags_the_softer_direction():
    estimate = pool_effective_icr([_observation(-0.3 + 0.01 * index, sd=0.05) for index in range(8)], programmed_icr=PROGRAMMED_ICR)

    assert estimate.status == "carbs_hit_softer_than_programmed"
    assert estimate.effective_icr > PROGRAMMED_ICR


def _trace(carbs: float, bolus_units: float, *, true_icr: float) -> MealWindowTrajectory:
    minutes = np.arange(0.0, 185.0, 5.0)
    insulin = bolus_units * _insulin_absorbed_fraction(minutes)
    glucose = 110.0 + (CORRECTION_FACTOR / true_icr) * carbs * absorbed_fraction(minutes, shape=2.5, mean_minutes=70.0) - CORRECTION_FACTOR * insulin
    return MealWindowTrajectory(minutes=minutes, glucose=glucose, insulin_absorbed_units=insulin)


def test_observation_ratio_tracks_true_sensitivity_not_the_logged_amount():
    kwargs = dict(stated_carbs=50.0, icr_grams_per_unit=PROGRAMMED_ICR, correction_factor_mgdl_per_unit=CORRECTION_FACTOR, baseline_glucose=110.0)
    matched = observe_carb_sensitivity(_trace(50.0, 5.0, true_icr=10.0), **kwargs)
    harder = observe_carb_sensitivity(_trace(50.0, 5.0, true_icr=8.0), **kwargs)

    assert matched is not None and harder is not None
    assert matched.log_ratio == pytest.approx(0.0, abs=0.12)
    assert harder.log_ratio == pytest.approx(np.log(10.0 / 8.0), abs=0.12)


def test_observation_requires_positive_logged_carbs():
    trace = _trace(50.0, 5.0, true_icr=10.0)
    for stated in (0.0, -5.0, float("nan")):
        assert observe_carb_sensitivity(trace, stated_carbs=stated, icr_grams_per_unit=PROGRAMMED_ICR, correction_factor_mgdl_per_unit=CORRECTION_FACTOR) is None


def test_effective_icr_recovery_harness_lands_on_truth():
    recovery = run_effective_icr_recovery(n_cohorts=12, meals_per_cohort=20, seed=5)

    by_truth = recovery.set_index("true_icr")
    for true_icr in (8.0, 10.0, 12.5):
        assert by_truth.loc[true_icr, "mean_estimated_icr"] == pytest.approx(true_icr, rel=0.06)
        assert by_truth.loc[true_icr, "coverage_90"] >= 0.7
    assert by_truth.loc[8.0, "flagged_as_different"] >= 0.9
    assert by_truth.loc[12.5, "flagged_as_different"] >= 0.9
    assert by_truth.loc[10.0, "flagged_as_different"] <= 0.25
    assert "true I:C" in render_effective_icr_recovery_markdown(recovery)


def test_effective_icr_recovery_rejects_empty_runs():
    with pytest.raises(ValueError):
        run_effective_icr_recovery(n_cohorts=0)


def _simulated_breakfast_data(*, days: int, true_icr: float, log_carbs: bool, seed: int = 0) -> IngestedData:
    """Raw-style tables for ``days`` days with one physically simulated 08:00 breakfast each."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-03-01 00:00:00", periods=days * 288, freq="5min")
    glucose = np.full(len(timestamps), 110.0)
    bolus_rows = []
    carb_rows = []
    window = np.arange(0.0, 305.0, 5.0)
    for day in range(days):
        start = day * 288 + 8 * 12
        carbs = float(rng.uniform(35.0, 75.0))
        bolus_units = carbs / PROGRAMMED_ICR
        response = (CORRECTION_FACTOR / true_icr) * carbs * absorbed_fraction(window, shape=2.5, mean_minutes=70.0) - CORRECTION_FACTOR * bolus_units * _insulin_absorbed_fraction(window)
        glucose[start : start + len(window)] += response
        # Whatever the meal left behind persists until the pre-dawn reset, as it would without a correction.
        glucose[start + len(window) : (day + 1) * 288] += response[-1]
        bolus_rows.append({"timestamp": timestamps[start], "bolus_units": bolus_units})
        # Logged five minutes after the bolus: not the same 5-minute bin.
        carb_rows.append({"timestamp": timestamps[start + 1], "carb_grams": carbs})
    glucose = glucose + rng.normal(0.0, 1.5, size=len(glucose))
    cgm = pd.DataFrame({"timestamp": timestamps, "glucose": np.clip(glucose, 45.0, 395.0), "source_file": "sim.csv"})
    basal = pd.DataFrame(
        {
            "start_timestamp": [timestamps[0]],
            "end_timestamp": [timestamps[-1] + pd.Timedelta(minutes=5)],
            "basal_units_per_hour": [0.8],
            "source_file": ["sim.csv"],
        }
    )
    return IngestedData(
        cgm=cgm,
        bolus=pd.DataFrame(bolus_rows).assign(source_file="sim.csv"),
        basal=basal,
        carbs=pd.DataFrame(carb_rows).assign(source_file="sim.csv") if log_carbs else pd.DataFrame(),
    )


def _run_full(data: IngestedData):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = build_analysis_ready_health_dataset(tandem_data=data, health_data=IngestedData(), config=FeatureConfig(horizon_minutes=30))
        return run_latent_meal_icr_research(
            dataset,
            segments=parse_therapy_segments(),
            research_scope="full",
            therapy_settings=LatentMealTherapySettings(
                icr_grams_per_unit={"morning": PROGRAMMED_ICR},
                correction_factor_mgdl_per_unit={"morning": CORRECTION_FACTOR},
                target_glucose_mgdl=110.0,
                insulin_action_minutes=300,
                carb_absorption_minutes_prior=180,
                max_plausible_meal_carbs=180.0,
            ),
        )


def test_pipeline_recovers_effective_icr_from_logged_breakfasts(tmp_path):
    true_icr = 8.0
    result = _run_full(_simulated_breakfast_data(days=14, true_icr=true_icr, log_carbs=True))

    assert result.latent_carb_estimates is not None and not result.latent_carb_estimates.empty
    assert result.latent_carb_estimates["estimator"].eq("trajectory").all()

    evidence = result.effective_icr_evidence
    assert evidence is not None
    morning = evidence.set_index("segment").loc["morning"]
    assert int(morning["n_meals"]) >= 8
    assert morning["status"] == "carbs_hit_harder_than_programmed"
    assert morning["effective_icr"] == pytest.approx(true_icr, rel=0.08)
    assert morning["effective_icr_lower_90"] <= true_icr <= morning["effective_icr_upper_90"]

    paths = write_latent_meal_research_artifacts(result, tmp_path / "latent")
    assert paths["effective_icr_evidence"].exists()
    report = paths["latent_carb_estimator_report"].read_text(encoding="utf-8")
    assert "## Effective I:C From Logged Meals" in report
    assert "carbs_hit_harder_than_programmed" in report

    review = write_latent_meal_review_html(result, tmp_path / "latent" / "latent_meal_review.html").read_text(encoding="utf-8")
    assert "How These Estimates Were Made" in review
    assert "Effective I:C From Logged Meals" in review
    assert "carbs_hit_harder_than_programmed" in review


def test_pipeline_says_icr_is_unassessable_without_logged_carbs(tmp_path):
    result = _run_full(_simulated_breakfast_data(days=8, true_icr=8.0, log_carbs=False))

    evidence = result.effective_icr_evidence
    assert evidence is not None
    assert int(pd.to_numeric(evidence["logged_meal_windows"]).sum()) == 0
    assert evidence["status"].eq("insufficient_logged_meals").all()

    # Without carb truth the trace is explained by "more carbs", which is the documented ambiguity.
    estimates = result.latent_carb_estimates
    assert (estimates["latent_carb_median"] > estimates["bolus_implied_carbs"]).mean() > 0.8

    paths = write_latent_meal_research_artifacts(result, tmp_path / "latent")
    report = paths["latent_carb_estimator_report"].read_text(encoding="utf-8")
    assert "No accepted window has a logged carb count" in report
    assert "carbs.csv" in report

    review = write_latent_meal_review_html(result, tmp_path / "latent" / "latent_meal_review.html").read_text(encoding="utf-8")
    assert "No accepted window has a logged carb count" in review


def test_cli_workflow_from_raw_tree_with_hand_kept_carb_log(tmp_path, monkeypatch):
    """prepare-model-data -> research-latent-meal-icr on a raw tree laid out like the real archive."""
    import json

    from bayesian_t1dm.cli import main

    cloud = tmp_path / "cloud"
    runtime = tmp_path / "runtime"
    repo = tmp_path / "repo"
    repo.mkdir()
    normalized = cloud / "data" / "raw" / "tconnectsync" / "2025-03-01__2025-03-14" / "normalized"
    normalized.mkdir(parents=True)
    data = _simulated_breakfast_data(days=14, true_icr=12.5, log_carbs=True, seed=2)
    data.cgm[["timestamp", "glucose"]].to_csv(normalized / "cgm.csv", index=False)
    data.bolus[["timestamp", "bolus_units"]].to_csv(normalized / "bolus.csv", index=False)
    data.basal[["start_timestamp", "end_timestamp", "basal_units_per_hour"]].to_csv(normalized / "basal.csv", index=False)
    # The hand-kept log sits at the top of the raw tree, not inside a normalized window.
    data.carbs[["timestamp", "carb_grams"]].to_csv(cloud / "data" / "raw" / "carbs.csv", index=False)
    settings_path = tmp_path / "therapy_settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "icr_grams_per_unit": {segment: PROGRAMMED_ICR for segment in ["overnight", "morning", "afternoon", "evening"]},
                "correction_factor_mgdl_per_unit": {segment: CORRECTION_FACTOR for segment in ["overnight", "morning", "afternoon", "evening"]},
                "target_glucose_mgdl": 110.0,
                "insulin_action_minutes": 300,
                "carb_absorption_minutes_prior": 180,
                "max_plausible_meal_carbs": 180,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(cloud))
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(runtime))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert main(["--root", str(repo), "prepare-model-data", "--skip-backfill"]) == 0
        assert (
            main(
                [
                    "--root",
                    str(repo),
                    "research-latent-meal-icr",
                    "--research-scope",
                    "full",
                    "--skip-backfill",
                    "--therapy-settings",
                    str(settings_path),
                ]
            )
            == 0
        )

    output = runtime / "output" / "latent_meal"
    evidence = pd.read_csv(output / "effective_icr_evidence.csv").set_index("segment").loc["morning"]
    assert int(evidence["n_meals"]) >= 8
    assert evidence["status"] == "carbs_hit_softer_than_programmed"
    assert evidence["effective_icr"] == pytest.approx(12.5, rel=0.08)
    assert (output / "latent_meal_review.html").exists()
    estimates = pd.read_csv(output / "latent_carb_estimates.csv")
    assert estimates["estimator"].eq("trajectory").all()


def test_full_scope_without_pump_profile_explains_how_to_supply_settings(tmp_path):
    from bayesian_t1dm.therapy_research import extract_latent_meal_therapy_settings_from_tandem_raw

    with pytest.raises(FileNotFoundError, match="--therapy-settings"):
        extract_latent_meal_therapy_settings_from_tandem_raw(tmp_path)
