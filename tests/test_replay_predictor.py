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


def synthetic_table(*, days: int = 460, seed: int = 3, shifts=None, steps=None, change_days=CHANGE_DAYS, day_sd: float = 0.04) -> pd.DataFrame:
    """A complete hour-day table: about 70% of readings in range, I:C lowered by ``steps[k]`` g/U (default 0.5) at hours 8-12 on change ``k``, and ``shifts[k]`` (a share, default 0) added to the in-range share on the 14 days after it."""
    rng = np.random.default_rng(seed)
    steps = np.full(len(change_days), 0.5) if steps is None else np.asarray(steps, dtype=float)
    shifts = np.zeros(len(change_days)) if shifts is None else np.asarray(shifts, dtype=float)
    level = np.clip(0.70 + rng.normal(0, day_sd, days), 0.2, 0.95)
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


def manual_target(old_icr: float, new_icr: float, *, cut: pd.Timestamp = START, baseline_icr=None, baseline_weights=None) -> rp.EventTarget:
    """A target whose settings differ only in I:C, with readings in every cell, a plain exposure pool and the same cut as its date; the baseline days default to the old settings."""
    old = np.tile(np.array([old_icr, 45.0, 0.8, 110.0]), (24, 1))
    new = np.tile(old, (14, 1, 1))
    new[:, :, 0] = new_icr
    baseline_icr = [old_icr] * 3 if baseline_icr is None else baseline_icr
    baseline = np.tile(old, (len(baseline_icr), 1, 1))
    baseline[:, :, 0] = np.asarray(baseline_icr)[:, None]
    weights = np.ones(len(baseline_icr)) if baseline_weights is None else np.asarray(baseline_weights, dtype=float)
    pool = np.random.default_rng(1).normal(size=(5, 24, 4))
    return rp.EventTarget(0, cut, cut + pd.Timedelta(days=2), 0, True, "more insulin", "I:C", "", 14, 200, np.full((14, 24), 12.0), old, new, baseline, weights, pool, np.array([True, False, False, False]),
                          {"tir": 70.0, "below70": 3.0, "below54": 0.5}, cut)


# ---- the prediction contrast ---------------------------------------------------

def test_predicted_change_has_the_sign_of_the_coefficient_times_the_setting_difference() -> None:
    lowered = manual_target(10.0, 9.0)  # I:C down: more insulin
    draws = rp.predict_event(fake_posterior(START, beta_icr=0.3), lowered, seed=1)
    assert (draws["expected_new"] - draws["expected_baseline"] < 0).all()  # a positive coefficient on log I:C and a lower I:C: fewer readings in range
    negative = rp.predict_event(fake_posterior(START, beta_icr=-0.3), lowered, seed=1)
    assert (negative["expected_new"] > negative["expected_baseline"]).all()


def test_swapping_old_and_new_settings_flips_the_predicted_change_exactly_and_equal_settings_give_zero() -> None:
    posterior = fake_posterior(START)
    forward = rp.predict_event(posterior, manual_target(10.0, 9.0), seed=2)
    backward = rp.predict_event(posterior, manual_target(9.0, 10.0), seed=2)
    np.testing.assert_allclose(forward["expected_new"] - forward["expected_baseline"], -(backward["expected_new"] - backward["expected_baseline"]), atol=1e-12)
    np.testing.assert_allclose(forward["expected_new"], backward["expected_old"], atol=1e-12)
    same = rp.predict_event(posterior, manual_target(10.0, 10.0), seed=2)
    np.testing.assert_allclose(same["expected_new"], same["expected_baseline"], atol=1e-12)
    np.testing.assert_array_equal(same["simulated_new"], same["simulated_old"])


def test_expected_change_matches_the_closed_form_without_hour_day_or_exposure_effects() -> None:
    posterior = fake_posterior(START, hour_scale=0.0, day_scale=0.0)
    posterior = replace(posterior, z_hour=np.zeros_like(posterior.z_hour))
    target = manual_target(10.0, 8.0, baseline_icr=[10.0, 7.0], baseline_weights=[1.0, 3.0])
    draws = rp.predict_event(posterior, target, seed=3)
    expected = lambda icr: expit(posterior.alpha + posterior.beta[:, 0] * (np.log(icr) - posterior.center[0]) / posterior.scale[0])  # noqa: E731
    np.testing.assert_allclose(draws["expected_new"], expected(8.0), atol=1e-10)
    np.testing.assert_allclose(draws["expected_old"], expected(10.0), atol=1e-10)
    np.testing.assert_allclose(draws["expected_baseline"], 0.25 * expected(10.0) + 0.75 * expected(7.0), atol=1e-10)


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
    assert not issues["reason"].str.contains("segment start time").any()


def test_a_change_whose_setting_values_do_not_differ_is_left_out_not_scored_as_zero() -> None:
    table = synthetic_table()
    table["icr"] = 10.0  # the flags say a change, the values do not
    _, targets, _, issues = plan(table)
    assert "equal before and after" in issues["reason"].iloc[0]
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
    unknown = new.copy()
    unknown[9, 1] = np.nan
    with pytest.raises(ValueError, match="unknown"):
        rp.describe_change(old, unknown, hours)


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

def test_the_report_reconciles_and_follows_the_workbook_rules(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rp, "POWER_SIMULATIONS", 10)
    monkeypatch.setattr(rp, "POWER_BOOTSTRAP_DRAWS", 100)
    table = synthetic_table()
    results, targets, flow, issues = plan(table)
    cuts = sorted({target.cut for target in targets})
    posteriors = {"tir": {cut: fake_posterior(cut, seed=index) for index, cut in enumerate(cuts)}}
    rows = rp.event_rows(posteriors["tir"], targets, "tir")
    joined = rp.join_observed(rows, results["scores"], "tir")
    assert len(joined) == len(targets)
    assert (joined["crps_new_minus_old_points"] == joined["crps_new_points"] - joined["crps_old_points"]).all()
    report.write_outputs(folder=tmp_path, run_id="test", raw="synthetic", table=table, flow=flow, issues=issues, joined=joined, targets=targets, posteriors=posteriors, replay_results=results,
                         draws=1000, tune=1000, chains=4, target_accept=0.95, min_train_days=30, seconds=1.0, command="test", note="")
    workbook = read_workbook(tmp_path / "diagnostics.xlsx")
    names = list(workbook)
    assert names[0] == "Specification_Coding" and names[1] == "Denominators_Flow" and names[-2:] == ["Warnings_Limitations", "Artifacts_Provenance"]
    assert {"Sensitivity", "Power"} <= set(names)
    for sheet in ("Checks", "Model_Fits"):
        assert set(workbook[sheet]["status"]) <= set(STATUSES)
    assert (tmp_path / "replay_predictor_figure.png").exists() and (tmp_path / "replay_predictor_figure.svg").exists()
    summary = (tmp_path / "replay_predictor_summary.md").read_text(encoding="utf-8")
    assert "## Verdict" in summary and "percentage points" in summary and "Slope intervals examined" in summary
    assert summary.isascii()


# ---- baseline, later changes, spans, agreement of the two readings of the settings ------

def test_the_old_arm_is_the_settings_in_force_on_the_baseline_days_not_the_profile_of_the_day_before() -> None:
    """A change that reverses the one 10 days before it: the 28 days the replay measures against hold both settings, so the predicted change is smaller than against the profile of the day before."""
    table = synthetic_table(change_days=[100, 110, 250], steps=[1.0, -1.0, 0.3])
    _, targets, _, _ = plan(table)
    reversal = next(target for target in targets if target.event_id == 1)
    icr = reversal.baseline[:, 8, 0]
    assert set(np.round(icr, 6)) == {10.0, 9.0} and (icr == 10.0).sum() == 18 and (icr == 9.0).sum() == 10  # days 82..99 under the old setting, 100..109 under the first change
    posterior = fake_posterior(reversal.cut, hour_scale=0.0, day_scale=0.0)
    draws = rp.predict_event(posterior, reversal, seed=5)
    against_baseline = (draws["expected_new"] - draws["expected_baseline"]).mean()
    against_day_before = (draws["expected_new"] - draws["expected_old"]).mean()
    assert 0 < against_baseline < against_day_before
    assert (draws["expected_baseline"] > draws["expected_old"]).all() and (draws["expected_baseline"] < draws["expected_new"]).all()


def test_a_later_change_inside_the_window_is_not_in_the_prediction_of_the_earlier_one() -> None:
    table = synthetic_table(change_days=[100, 105, 250], steps=[0.5, 1.0, 0.3])
    _, targets, _, _ = plan(table)
    first = next(target for target in targets if target.event_id == 0)
    np.testing.assert_allclose(first.new[:, 8:13, 0], 9.5)  # the profile after the first change only, held for the window
    np.testing.assert_allclose(first.old[8:13, 0], 10.0)
    assert not first.isolated and first.cluster_id == next(target for target in targets if target.event_id == 1).cluster_id


def test_a_change_over_two_days_has_one_old_and_one_new_profile() -> None:
    table = synthetic_table(change_days=[100, 250], steps=[0.5, 0.3])
    day = (table["date"] - START).dt.days.to_numpy()
    later = (day >= 101) & table["hour"].between(14, 16).to_numpy()  # the edit reached the hours already passed the next day
    table.loc[later, "icr"] = 9.0
    table.loc[(day == 101) & table["hour"].between(14, 16).to_numpy(), ["settings_changed", "d_icr"]] = [True, -1.0]
    _, targets, _, _ = plan(table)
    target = targets[0]
    assert target.counts[:, 8:13].sum() > 0 and target.counts[:, 14:17].sum() > 0
    np.testing.assert_allclose(target.new[0, 8:13, 0], 9.5)
    np.testing.assert_allclose(target.new[0, 14:17, 0], 9.0)
    np.testing.assert_allclose(target.old[[8, 9, 14, 15], 0], 10.0)
    assert "I:C" in target.description


def test_the_frozen_profile_and_the_hour_day_table_must_agree_on_days_without_a_change() -> None:
    table = synthetic_table()
    grid = replay.day_grid(table)
    profiles = profiles_from(table)
    assert rp.check_profiles_agree(table, grid, profiles) > 0
    broken = fh.SettingsProfiles(profiles.dates, profiles.values.copy())
    broken.values[30, 3, 0] += 0.5  # a day with no change
    with pytest.raises(AssertionError, match="disagree"):
        rp.check_profiles_agree(table, grid, broken)


def test_a_day_whose_difference_is_unknown_is_not_compared() -> None:
    table = synthetic_table()
    grid = replay.day_grid(table)
    profiles = profiles_from(table)
    day = (table["date"] - START).dt.days.to_numpy()
    table.loc[(day == 30) & (table["hour"] == 3).to_numpy(), "d_icr"] = np.nan
    broken = fh.SettingsProfiles(profiles.dates, profiles.values.copy())
    broken.values[30, 3, 0] += 0.5  # an edit during the day that the replay cannot see
    assert rp.check_profiles_agree(table, grid, broken) > 0


def test_unknown_settings_on_a_baseline_day_leave_the_change_out_with_a_reason() -> None:
    table = synthetic_table()
    day = (table["date"] - START).dt.days.to_numpy()
    table.loc[(day == 90) & (table["hour"] == 9).to_numpy(), "settings_known_slots"] = 6  # a baseline day of the second change (change on day 95 is the first)
    _, targets, _, issues = plan(table)
    left_out = issues[issues["reason"].str.contains("baseline day")]
    assert len(left_out) >= 1 and left_out["step"].eq("settings known and exposures available").all()


def test_the_prior_sd_used_for_the_prior_check_is_the_one_in_the_model() -> None:
    import inspect
    import re

    match = re.search(r'beta = pm\.Normal\("beta", mu=0\.0, sigma=([0-9.]+)', inspect.getsource(om.build_model))
    assert match and float(match.group(1)) == rp.SETTING_PRIOR_SD


def test_a_change_that_moves_a_setting_the_fit_has_not_learnt_is_marked_prior_only() -> None:
    posterior = fake_posterior(START)
    beta = posterior.beta.copy()
    beta[:, 3] = np.random.default_rng(0).normal(0, rp.SETTING_PRIOR_SD, len(beta))  # target: as wide as its prior
    beta[:, 0] = np.random.default_rng(1).normal(0, 0.02, len(beta))  # I:C: learnt
    posterior = replace(posterior, beta=beta)
    assert rp.moved_prior_ratio(posterior, np.array([True, False, False, False])) < 0.2
    assert rp.moved_prior_ratio(posterior, np.array([True, False, False, True])) > rp.PRIOR_DOMINATED_RATIO


def test_the_no_change_noise_range_is_centred_and_widened_for_the_mean_of_five_placebos() -> None:
    rng = np.random.default_rng(0)
    placebo = rng.normal(3.0, 5.0, 5000)  # a biased baseline: the misses are not centred on zero
    rows = []
    for event in range(1000):
        rows.append({"event_id": event, "kind": "change", "predictor": replay.PRIMARY, "tir_residual": 0.0, "cluster_id": event, "isolated": True, "direction": "mixed"})
        for placebo_value in placebo[event * 5: event * 5 + 5]:
            rows.append({"event_id": event, "kind": "matched placebo", "predictor": replay.PRIMARY, "tir_residual": placebo_value, "cluster_id": event, "isolated": True, "direction": "mixed"})
    out = rp.observed_changes(pd.DataFrame(rows), "tir")
    lower, upper = out["no_change_noise_lower_5th"].iloc[0], out["no_change_noise_upper_95th"].iloc[0]
    assert lower == pytest.approx(-upper, rel=0.1)  # centred
    assert upper == pytest.approx(1.645 * 5.0 * np.sqrt(1.2), rel=0.05)
    assert out["observed_change_points"].mean() == pytest.approx(-3.0, abs=0.3)  # the change's miss minus the mean placebo miss


# ---- the verdict is written from the numbers ------------------------------------

def fake_joined(*, slope: float, crps_gap: float = 0.0, noise: float = 0.5, clusters: int = 12, seed: int = 0, outcomes=("tir",)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for outcome in outcomes:
        x = rng.normal(0, 4, clusters)
        frames.append(pd.DataFrame({"outcome": outcome, "event_id": np.arange(clusters), "cluster_id": np.arange(clusters), "isolated": True, "change_date": [f"2025-01-{index + 1:02d}" for index in range(clusters)],
                                    "counted_days": 14, "predicted_change_points": x, "predicted_change_day_before_points": x, "observed_change_points": slope * x + rng.normal(0, noise, clusters),
                                    "inside_credible_interval": True, "inside_noise_range": True, "prior_dominated": False,
                                    "crps_new_points": 3.0 + crps_gap + rng.normal(0, 0.2, clusters), "crps_old_points": 3.0 + rng.normal(0, 0.2, clusters)}).assign(crps_new_minus_old_points=lambda f: f["crps_new_points"] - f["crps_old_points"]))
    return pd.concat(frames, ignore_index=True)


def verdict_for(joined: pd.DataFrame) -> tuple[list[str], str]:
    tracking = rp.tracking_table(joined, draws=500)
    return report.verdict_lines(joined, tracking, rp.information_table(joined, draws=500), rp.power_table(joined, simulations=20, draws=200), rp.sensitivity_table(joined, draws=200))


def test_verdict_says_the_old_settings_predicted_better_when_the_interval_is_above_zero() -> None:
    bullets, reading = verdict_for(fake_joined(slope=0.0, crps_gap=1.0))
    text = "\n".join(bullets)
    assert "old settings predict the window better" in text.lower().replace("the old settings predict the window better than the settings in force", "old settings predict the window better")
    assert "the old settings predicted the 14-day window better than the settings in force" in reading
    assert "better than the old settings" not in reading.replace("the old settings predicted the 14-day window better than the settings in force", "")


def test_verdict_reports_a_slope_interval_below_zero_as_moving_against_the_prediction() -> None:
    bullets, reading = verdict_for(fake_joined(slope=-1.5))
    assert any(line.strip().startswith("- Below zero: time in range, isolated changes") or "Below zero: time in range, isolated changes" in line for line in bullets)
    assert "moved against the predicted direction" in reading and "Predicted changes line up" not in reading


def test_verdict_lists_every_interval_examined_and_the_groups_that_could_not_be_estimated() -> None:
    joined = fake_joined(slope=0.0, outcomes=("tir", "below70", "below54"))
    joined.loc[joined["outcome"].eq("below54"), "isolated"] = False  # no isolated rows for one outcome
    bullets, _ = verdict_for(joined)
    text = "\n".join(bullets)
    assert "Slope intervals examined: 6" in text
    assert "Not estimated: time below 54, isolated changes" in text


def test_a_zero_slope_with_low_power_is_called_uninformative_not_a_failure_of_the_model() -> None:
    _, reading = verdict_for(fake_joined(slope=0.0, noise=30.0))
    assert "uninformative" in reading and "not evidence that they carry none" in reading
    _, tracked = verdict_for(fake_joined(slope=1.0, noise=0.3))
    assert "Predicted changes line up" in tracked and "uninformative" not in tracked


def test_the_summary_tables_name_the_dropped_groups(tmp_path) -> None:
    tracking = rp.tracking_table(fake_joined(slope=0.0, clusters=2), draws=100)
    assert tracking["reading"].str.startswith("not estimated").all()


# ---- what the test can detect ----------------------------------------------------

def test_a_planted_slope_is_recovered_with_realistic_day_noise_and_the_power_table_says_what_is_detectable() -> None:
    """Daily SD of the in-range share of 8 points (the real record's is of that order), effects planted in proportion to the model's own prediction (about -7.5 points per g/U): the slope interval
    is above zero and near 1 for a planted slope of 1, holds zero when nothing is planted, and the power rises with the true slope."""
    steps = np.linspace(0.2, 1.0, len(CHANGE_DAYS))
    results = {}
    for name, planted in (("planted", -0.075 * steps), ("none", np.zeros(len(steps)))):
        table = synthetic_table(steps=steps, shifts=planted, day_sd=0.08, seed=11)
        replay_results, targets, _, _ = plan(table)
        posteriors = {target.cut: fake_posterior(target.cut) for target in targets}
        joined = rp.join_observed(rp.event_rows(posteriors, targets, "tir"), replay_results["scores"], "tir")
        results[name] = (rp.tracking_table(joined, draws=1000).query("group == 'all changes'").iloc[0], rp.power_table(joined, simulations=60, draws=300).query("group == 'all changes'").iloc[0])
    planted_row, planted_power = results["planted"]
    assert planted_row["slope_lower_5th"] > 0 and 0.4 < planted_row["slope"] < 2.0
    none_row, _ = results["none"]
    assert none_row["slope_lower_5th"] < 0 < none_row["slope_upper_95th"]
    assert planted_power["power_true_slope_4"] >= planted_power["power_true_slope_1"] >= planted_power["power_true_slope_0.5"]
    assert planted_power["power_true_slope_4"] > 0.8
