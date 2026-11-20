"""Tests for the replay-as-predictor module on synthetic data (no clinical data, no sampler)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from bayesian_t1dm import forecast_harness as fh
from bayesian_t1dm import outcomes_model as om
from bayesian_t1dm import replay
from bayesian_t1dm import replay_predictor as rp
from bayesian_t1dm import replay_predictor_report as report
from bayesian_t1dm.diagnostics_workbook import STATUSES, read_workbook
from bayesian_t1dm.sampler_health import FitDiagnostics

START = pd.Timestamp("2024-01-01")
CHANGE_DAYS = list(range(60, 400, 35))  # ten changes, 35 days apart: each is isolated
CHANGED_HOURS = range(8, 13)


def synthetic_table(*, days: int = 460, seed: int = 3, shifts=None, steps=None, change_days=CHANGE_DAYS) -> pd.DataFrame:
    """A complete hour-day table: about 70% of readings in range, I:C lowered by ``steps[k]`` g/U (default 0.5) at hours 8-12 on change ``k``, and ``shifts[k]`` (a share, default 0) added to the in-range share on the 14 days after it."""
    rng = np.random.default_rng(seed)
    steps = np.full(len(change_days), 0.5) if steps is None else np.asarray(steps, dtype=float)
    shifts = np.zeros(len(change_days)) if shifts is None else np.asarray(shifts, dtype=float)
    level = np.clip(0.70 + rng.normal(0, 0.04, days), 0.4, 0.95)
    for day, shift in zip(change_days, shifts):
        level[day + 1: day + 1 + replay.AFTER_DAYS] += shift
    hour_start = pd.date_range(START, periods=days * 24, freq="h")
    share = np.repeat(level, 24)
    counts = np.array([rng.multinomial(12, [p, 0.03, max(1 - p - 0.03, 0)]) for p in share])
    table = pd.DataFrame({"hour_start": hour_start, "date": hour_start.normalize(), "hour": hour_start.hour, "n_valid": 12, "n_in_range": counts[:, 0], "n_below_70": counts[:, 1], "n_below_54": 0,
                          "n_above_180": counts[:, 2], "glucose_sum": 12 * 130.0, "settings_known_slots": 12, "settings_mixed": False, "excluded_day": False, "pump_serial": "A",
                          "icr": 10.0, "cf": 45.0, "basal": 0.8, "target": 110.0, "carbs_prev3h_g": (np.arange(days * 24) // 24).astype(float), "exercise_share": 0.0,
                          "sleep_mode_share": (hour_start.hour < 6).astype(float), "site_age_days": (np.arange(days * 24) // 24 % 4).astype(float)})
    table["settings_changed"] = False
    for name in ("icr", "cf", "basal", "target"):
        table["d_" + name] = 0.0
    day_index = (table["date"] - START).dt.days.to_numpy()
    for count, day in enumerate(change_days, start=1):
        rows = (day_index >= day) & table["hour"].between(CHANGED_HOURS.start, CHANGED_HOURS.stop - 1).to_numpy()
        table.loc[rows, "icr"] = 10.0 - steps[:count].sum()
        flagged = (day_index == day) & table["hour"].between(CHANGED_HOURS.start, CHANGED_HOURS.stop - 1).to_numpy()
        table.loc[flagged, "settings_changed"] = True
        table.loc[flagged, "d_icr"] = -steps[count - 1]
    return table


def profiles_from(table: pd.DataFrame) -> fh.SettingsProfiles:
    """The profile frozen at the last moment of the day before each date, read off the table's settings."""
    dates = pd.DatetimeIndex(sorted(table["date"].unique()))
    wide = table.sort_values(["date", "hour"])[["icr", "cf", "basal", "target"]].to_numpy().reshape(len(dates), 24, 4)
    values = np.full((len(dates), 24, 4), np.nan)
    values[1:] = wide[:-1]
    return fh.SettingsProfiles(dates, values)


def plan(table: pd.DataFrame, *, min_train_days: int = 30):
    results = replay.run_replay(table)
    grid = replay.day_grid(table)
    events = replay.find_events(grid)
    clusters, isolated = replay.cluster_events(events)
    daily = fh.daily_counts(table, "tir")
    tdd = pd.Series(40.0, index=pd.DatetimeIndex(sorted(table["date"].unique())))
    targets, flow, issues = rp.plan_targets(table, grid, events, clusters, isolated, results["scores"], results["issues"], profiles_from(table), daily, tdd, min_train_days=min_train_days)
    return results, targets, flow, issues


def fake_posterior(origin: pd.Timestamp, *, draws: int = 300, beta_icr: float = 0.3, seed: int = 5, hour_scale: float = 0.3, day_scale: float = 0.2) -> fh.ModelPosterior:
    rng = np.random.default_rng(seed)
    diagnostics = FitDiagnostics(draws=1000, tune=1000, chains=4, target_accept=0.95, max_treedepth=10, wall_time_seconds=1.0, divergences=0, max_tree_depth_observed=5, max_tree_depth_hits=0,
                                 rhat_max=1.002, ess_bulk_min=900.0, ess_tail_min=950.0)
    beta = np.zeros((draws, 8))
    beta[:, 0] = rng.normal(beta_icr, 0.05, draws)
    return fh.ModelPosterior(origin=origin, alpha=rng.normal(1.0, 0.05, draws), beta=beta, z_hour=rng.normal(0, 0.1, (draws, 24)), hour_scale=np.full(draws, hour_scale), day_scale=np.full(draws, day_scale),
                             log_phi=np.full(draws, 2.0), center=np.array([2.3, 3.8, -0.2, 11.0, 2.0, 0.0, 0.2, 1.5]), scale=np.array([0.1, 0.1, 0.1, 0.5, 1.0, 1.0, 0.4, 1.0]), diagnostics=diagnostics,
                             timing={"sample_seconds": 1.0, "compile_seconds": 1.0, "build_seconds": 0.1}, rows=1000, days=100, fingerprint="synthetic", model_hash="synthetic-model")


def manual_target(old_icr: float, new_icr: float, *, cut: pd.Timestamp = START) -> rp.EventTarget:
    """A target whose settings differ only in I:C, with readings in every cell, a plain exposure pool and the same cut as its date."""
    old = np.tile(np.array([old_icr, 45.0, 0.8, 110.0]), (24, 1))
    new = np.tile(old, (14, 1, 1))
    new[:, :, 0] = new_icr
    pool = np.random.default_rng(1).normal(size=(5, 24, 4))
    return rp.EventTarget(0, cut, cut + pd.Timedelta(days=2), 0, True, "more insulin", "I:C", "", 14, 200, np.full((14, 24), 12.0), old, new, pool, {"tir": 70.0, "below70": 3.0, "below54": 0.5}, cut)


# ---- the prediction contrast ---------------------------------------------------

def test_predicted_change_has_the_sign_of_the_coefficient_times_the_setting_difference() -> None:
    posterior = fake_posterior(START, beta_icr=0.3)
    lowered = manual_target(10.0, 9.0)  # I:C down: more insulin
    draws = rp.predict_event(posterior, lowered, seed=1)
    change = draws["expected_new"] - draws["expected_old"]
    assert (change < 0).all()  # a positive coefficient on log I:C and a lower I:C: fewer readings in range
    assert (rp.predict_event(fake_posterior(START, beta_icr=-0.3), lowered, seed=1)["expected_new"] > rp.predict_event(fake_posterior(START, beta_icr=-0.3), lowered, seed=1)["expected_old"]).all()


def test_swapping_old_and_new_settings_flips_the_predicted_change_exactly_and_equal_settings_give_zero() -> None:
    posterior = fake_posterior(START)
    forward = rp.predict_event(posterior, manual_target(10.0, 9.0), seed=2)
    backward = rp.predict_event(posterior, manual_target(9.0, 10.0), seed=2)
    np.testing.assert_allclose(forward["expected_new"] - forward["expected_old"], -(backward["expected_new"] - backward["expected_old"]), atol=1e-12)
    np.testing.assert_allclose(forward["expected_new"], backward["expected_old"], atol=1e-12)
    same = rp.predict_event(posterior, manual_target(10.0, 10.0), seed=2)
    np.testing.assert_array_equal(same["expected_new"], same["expected_old"])
    np.testing.assert_array_equal(same["simulated_new"], same["simulated_old"])


def test_expected_change_matches_the_closed_form_without_hour_day_or_exposure_effects() -> None:
    posterior = fake_posterior(START, hour_scale=0.0, day_scale=0.0)
    posterior = replace(posterior, z_hour=np.zeros_like(posterior.z_hour))
    target = manual_target(10.0, 8.0)
    draws = rp.predict_event(posterior, target, seed=3)
    centred = lambda icr: (np.log(icr) - posterior.center[0]) / posterior.scale[0]  # noqa: E731
    expected = lambda icr: expit(posterior.alpha + posterior.beta[:, 0] * centred(icr))  # noqa: E731
    np.testing.assert_allclose(draws["expected_new"], expected(8.0), atol=1e-10)
    np.testing.assert_allclose(draws["expected_old"], expected(10.0), atol=1e-10)


def test_cells_without_readings_take_no_part_in_the_prediction() -> None:
    posterior = fake_posterior(START)
    target = manual_target(10.0, 9.0)
    counts = np.zeros((14, 24))
    counts[:, 8:13] = 12.0  # only the affected hours were counted
    new = target.new.copy()
    new[:, :8, :] = np.nan  # settings elsewhere are unknown: they must not matter
    narrowed = replace(target, counts=counts, new=new)
    draws = rp.predict_event(posterior, narrowed, seed=4)
    assert np.isfinite(draws["expected_new"]).all() and np.isfinite(draws["simulated_new"]).all()


# ---- inputs known at the change date ---------------------------------------------

def test_cuts_share_a_fit_only_within_fourteen_days_after_the_first_change_of_the_group() -> None:
    dates = [START + pd.Timedelta(days=day) for day in (0, 10, 14, 15, 40)]
    targets = [replace(manual_target(10.0, 9.0), event_id=index, event_date=date, cut=date) for index, date in enumerate(dates)]
    cuts = [target.cut for target in rp.assign_cuts(targets)]
    assert cuts == [dates[0], dates[0], dates[0], dates[3], dates[4]]
    assert all(target.cut <= target.event_date for target in rp.assign_cuts(targets))


def test_nothing_on_or_after_the_change_reaches_its_inputs() -> None:
    table = synthetic_table()
    _, targets, _, _ = plan(table)
    victim = targets[3]
    garbage = table.copy()
    future = garbage["date"] >= victim.event_date
    garbage.loc[future, ["n_in_range", "n_below_70", "carbs_prev3h_g", "exercise_share", "sleep_mode_share", "site_age_days"]] = [3, 9, 777.0, 0.9, 0.9, 3.0]
    garbage.loc[future, "icr"] = 99.0
    _, changed_targets, _, _ = plan(garbage)
    same = next(target for target in changed_targets if target.event_id == victim.event_id)
    np.testing.assert_array_equal(same.pool, victim.pool)
    np.testing.assert_array_equal(same.old, victim.old)
    assert (same.cut, same.training_days) == (victim.cut, victim.training_days)
    before = rp.check_no_leakage(table, victim)
    after = rp.check_no_leakage(garbage, same)
    assert before["date"].max() < victim.event_date
    assert fh.design_fingerprint(om.build_design(before, outcome="tir")) == fh.design_fingerprint(om.build_design(after, outcome="tir"))


def test_a_fit_cut_after_the_change_date_is_refused() -> None:
    table = synthetic_table()
    _, targets, _, _ = plan(table)
    with pytest.raises(AssertionError, match="would reach its fit"):
        rp.check_no_leakage(table, replace(targets[0], cut=targets[0].event_date + pd.Timedelta(days=1)))


def test_a_change_is_never_given_another_cuts_fit() -> None:
    _, targets, _, _ = plan(synthetic_table())
    with pytest.raises(AssertionError, match="needs the fit"):
        rp.check_posterior_precedes(fake_posterior(targets[0].cut - pd.Timedelta(days=1)), targets[0])


def test_changes_are_left_out_with_a_reason_and_the_flow_reconciles() -> None:
    table = synthetic_table()
    _, targets, flow, issues = plan(table, min_train_days=100)
    stage = flow.set_index("stage")["changes"]
    assert stage.iloc[0] == len(targets) + len(issues)
    assert (issues["step"] == "enough training days").any()  # the first change comes 60 days in
    assert set(issues["step"]) <= {"scored by the replay", "enough training days", "settings known and exposures available", "different setting values"}


def test_a_change_whose_setting_values_do_not_differ_is_left_out_not_scored_as_zero() -> None:
    table = synthetic_table()
    table["icr"] = 10.0  # the flags say a change, the values do not
    _, targets, _, issues = plan(table)
    assert not targets and (issues["step"] == "different setting values").sum() > 0


def test_the_description_uses_the_pump_units_and_names_only_settings_that_differ() -> None:
    old = np.tile(np.array([10.0, 45.0, 0.8, 110.0]), (24, 1))
    new = old.copy()
    new[8:13, 0] = 9.5
    new[8:13, 2] = 0.85
    hours = np.zeros(24, dtype=bool)
    hours[8:17] = True
    text = rp.describe_change(old, new, hours)
    assert text == "I:C 10.00 to 9.50 g/U; basal 0.800 to 0.850 U/h"


# ---- the cluster bootstrap and the statistics ----------------------------------

def test_cluster_bootstrap_is_reproducible_and_a_known_slope_is_recovered() -> None:
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(15), 2)
    x = rng.normal(size=30)
    y = 2.0 * x + rng.normal(0, 0.3, 30)
    values = np.column_stack([x, y])
    first = rp.cluster_bootstrap(values, clusters, rp.slope, draws=2000, seed=7)
    second = rp.cluster_bootstrap(values, clusters, rp.slope, draws=2000, seed=7)
    assert first == second
    assert first.estimate == pytest.approx(2.0, abs=0.3)
    assert first.lower < 2.0 < first.upper and first.lower > 0
    assert first.clusters == 15 and first.undefined == 0


def test_cluster_bootstrap_with_no_relationship_usually_holds_zero() -> None:
    covered = 0
    for seed in range(20):  # a 90% interval misses zero by chance about one time in ten (a little more with 12 clusters), so judge several samples
        rng = np.random.default_rng(seed)
        values = np.column_stack([rng.normal(size=36), rng.normal(size=36)])
        result = rp.cluster_bootstrap(values, np.repeat(np.arange(12), 3), rp.slope, draws=1000, seed=3)
        covered += int(result.lower < 0 < result.upper)
    assert covered >= 14


def test_cluster_bootstrap_keeps_a_cluster_together() -> None:
    values = np.arange(5.0)[:, None]
    clusters = np.array([0, 0, 1, 1, 1])  # sizes 2 and 3: whole clusters give totals of 4, 5 or 6 rows, never a mix
    result = rp.cluster_bootstrap(values, clusters, lambda v: float(v.shape[0]), draws=2000, seed=5)
    assert (result.estimate, result.lower, result.upper) == (5.0, 4.0, 6.0)


def test_a_resample_without_variation_in_the_predicted_change_is_counted_undefined() -> None:
    values = np.column_stack([np.zeros(9), np.arange(9.0)])
    result = rp.cluster_bootstrap(values, np.repeat(np.arange(3), 3), rp.slope, draws=100, seed=2)
    assert np.isnan(result.estimate) and result.undefined == 100


def test_direction_agreement_counts_matching_signs() -> None:
    assert rp.agreement(np.array([[1.0, 2.0], [-1.0, -3.0], [1.0, -1.0], [2.0, 0.5]])) == 0.75


def test_readings_say_what_the_interval_says() -> None:
    assert "does not show" in rp.tracking_reading({"slope_lower_5th": -0.5, "slope_upper_95th": 0.9})
    assert "rise with" in rp.tracking_reading({"slope_lower_5th": 0.1, "slope_upper_95th": 0.9})
    assert "against" in rp.tracking_reading({"slope_lower_5th": -0.9, "slope_upper_95th": -0.1})
    assert "better than the old" in rp.information_reading(rp.Interval(-0.4, -0.7, -0.1, 0, 5))
    assert "does not show" in rp.information_reading(rp.Interval(-0.4, -0.9, 0.2, 0, 5))
    assert "old settings predict" in rp.information_reading(rp.Interval(0.4, 0.1, 0.8, 0, 5))


# ---- end to end on a synthetic record -------------------------------------------

def test_the_report_reconciles_and_follows_the_workbook_rules(tmp_path) -> None:
    table = synthetic_table()
    results, targets, flow, issues = plan(table)
    cuts = sorted({target.cut for target in targets})
    posteriors = {"tir": {cut: fake_posterior(cut, seed=index) for index, cut in enumerate(cuts)}}
    rows = rp.event_rows(posteriors["tir"], targets, "tir")
    joined = rp.join_observed(rows, results["scores"], "tir")
    assert len(joined) == len(targets)
    assert (joined["crps_new_minus_old_points"] == joined["crps_new_points"] - joined["crps_old_points"]).all()
    report.write_outputs(folder=tmp_path, run_id="test", raw="synthetic", table=table, flow=flow, issues=issues, joined=joined, targets=targets, posteriors=posteriors, replay_results=results,
                         draws=1000, tune=1000, chains=4, min_train_days=30, seconds=1.0, command="test", note="")
    workbook = read_workbook(tmp_path / "diagnostics.xlsx")
    names = list(workbook)
    assert names[0] == "Specification_Coding" and names[1] == "Denominators_Flow" and names[-2:] == ["Warnings_Limitations", "Artifacts_Provenance"]
    for sheet in ("Checks", "Model_Fits"):
        assert set(workbook[sheet]["status"]) <= set(STATUSES)
    assert (tmp_path / "replay_predictor_figure.png").exists() and (tmp_path / "replay_predictor_figure.svg").exists()
    summary = (tmp_path / "replay_predictor_summary.md").read_text(encoding="utf-8")
    assert "## Verdict" in summary and "percentage points" in summary
    assert summary.isascii()


def test_an_effect_planted_in_proportion_to_the_prediction_gives_a_slope_near_one_and_none_gives_zero() -> None:
    """The model's own I:C coefficient predicts about -7.5 points of time in range per g/U lowered; plant that effect and the slope of observed on predicted is near 1; plant none and the interval holds 0."""
    steps = np.linspace(0.3, 1.5, len(CHANGE_DAYS))
    for planted, expect_tracking in ((-0.075 * steps, True), (np.zeros(len(steps)), False)):
        table = synthetic_table(steps=steps, shifts=planted)
        results, targets, _, _ = plan(table)
        posteriors = {target.cut: fake_posterior(target.cut) for target in targets}
        joined = rp.join_observed(rp.event_rows(posteriors, targets, "tir"), results["scores"], "tir")
        row = rp.tracking_table(joined, draws=1000).query("group == 'all changes'").iloc[0]
        if expect_tracking:
            assert row["slope_lower_5th"] > 0 and 0.4 < row["slope"] < 2.0
        else:
            assert row["slope_lower_5th"] < 0 < row["slope_upper_95th"]
