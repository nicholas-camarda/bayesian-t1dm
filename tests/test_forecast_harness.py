"""Tests for the forecast harness and the win test on synthetic data (no clinical data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from bayesian_t1dm import forecast_harness as fh
from bayesian_t1dm import forecast_harness_report as report
from bayesian_t1dm import outcomes_model_report as omr
from bayesian_t1dm import win_test as wt
from bayesian_t1dm.diagnostics_workbook import STATUSES, read_workbook
from bayesian_t1dm.sampler_health import FitDiagnostics

FIRST_DAY = pd.Timestamp("2024-01-01")


def synthetic_table(days: int = 700, *, seed: int = 11, first: pd.Timestamp = FIRST_DAY) -> pd.DataFrame:
    """A complete hour-day table: a slow level plus day noise in the in-range share, constant settings, a unique carbohydrate value per day."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(first, periods=days, freq="D")
    level = 0.72 + 0.05 * np.sin(2 * np.pi * np.arange(days) / 120)
    rows = []
    for index, date in enumerate(dates):
        share = float(np.clip(level[index] + rng.normal(0, 0.04), 0.3, 0.95))
        for hour in range(24):
            hits = int(rng.binomial(12, share))
            rows.append({"date": date, "hour": hour, "n_valid": 12, "n_in_range": hits, "n_below_70": 12 - hits, "n_below_54": 0, "pump_serial": "A",
                         "icr": 10.0, "cf": 45.0, "basal": 0.8, "target": 110.0, "settings_known_slots": 12, "excluded_day": False,
                         "carbs_prev3h_g": float(index), "exercise_share": 0.0, "sleep_mode_share": float(hour < 6), "site_age_days": float(index % 4)})
    return pd.DataFrame(rows)


def constant_profiles(table: pd.DataFrame) -> fh.SettingsProfiles:
    dates = pd.DatetimeIndex(sorted(table["date"].unique()))
    return fh.SettingsProfiles(dates, np.tile(np.array([10.0, 45.0, 0.8, 110.0]), (len(dates), 24, 1)))


def constant_tdd(table: pd.DataFrame, value: float = 40.0) -> pd.Series:
    return pd.Series(value, index=pd.DatetimeIndex(sorted(table["date"].unique())), name="tdd")


class FakeHistory:
    """A pump profile that changes its I:C from 10 to 8 at 2025-02-10 12:00 (the profile as it stood at the queried moment)."""

    def active_profile(self, pump: str, moment) -> int:
        return 1

    def segment_in_force(self, pump: str, profile: int, moment, *, minute_of_day: int | None = None) -> dict[str, float]:
        icr = 8.0 if moment >= pd.Timestamp("2025-02-10 12:00").to_pydatetime() else 10.0
        return {"carb_ratio_g_per_unit": icr, "isf_mgdl_per_unit": 45.0, "basal_units_per_hour": 0.8, "target_bg_mgdl": 110.0}


def fake_posterior(origin: pd.Timestamp, draws: int = 200, seed: int = 5) -> fh.ModelPosterior:
    rng = np.random.default_rng(seed)
    diagnostics = FitDiagnostics(draws=1000, tune=1000, chains=4, target_accept=0.95, max_treedepth=10, wall_time_seconds=1.0, divergences=0, max_tree_depth_observed=5, max_tree_depth_hits=0,
                                 rhat_max=1.002, ess_bulk_min=900.0, ess_tail_min=950.0)
    return fh.ModelPosterior(origin=origin, alpha=rng.normal(1.0, 0.05, draws), beta=rng.normal(0, 0.05, (draws, 8)), z_hour=rng.normal(0, 0.1, (draws, 24)), hour_scale=np.full(draws, 0.3), day_scale=np.full(draws, 0.2),
                             log_phi=np.full(draws, 2.0), center=np.array([2.3, 3.8, -0.2, 11.0, 2.0, 0.0, 0.2, 1.5]), scale=np.array([0.1, 0.1, 0.1, 0.5, 1.0, 1.0, 0.4, 1.0]), diagnostics=diagnostics,
                             timing={"sample_seconds": 1.0, "compile_seconds": 1.0, "build_seconds": 0.1}, rows=1000, days=100, fingerprint="synthetic")


# ---- origins, target, windows ------------------------------------------------

def test_evaluation_origins_are_52_weekly_and_the_last_window_ends_on_the_last_day() -> None:
    last = pd.Timestamp("2026-11-17")
    origins = fh.evaluation_origins(last)
    assert len(origins) == 52 and origins == sorted(origins)
    assert {(b - a).days for a, b in zip(origins, origins[1:])} == {7}
    assert origins[-1] + pd.Timedelta(days=13) == last
    fh.check_origins_in_record(origins, origins[0], last)
    with pytest.raises(ValueError, match="do not all fit"):
        fh.check_origins_in_record(origins, origins[1], last)


def test_development_origins_are_every_second_week_and_end_before_the_evaluation_windows() -> None:
    development = fh.development_origins()
    assert development[0] == pd.Timestamp("2025-01-07") and {(b - a).days for a, b in zip(development, development[1:])} == {14}
    assert development[-1] <= pd.Timestamp("2025-11-04")
    assert development[-1] + pd.Timedelta(days=13) < fh.evaluation_origins(pd.Timestamp("2026-11-17"))[0]


def test_target_includes_excluded_days_and_unknown_settings_and_weights_by_readings() -> None:
    table = synthetic_table(days=20, first=pd.Timestamp("2025-03-01"))
    origin = pd.Timestamp("2025-03-03")
    window = (table["date"] >= origin) & (table["date"] < origin + pd.Timedelta(days=14))
    table.loc[window & (table["date"] == origin), "excluded_day"] = True
    table.loc[window & (table["date"] == origin + pd.Timedelta(days=1)) & (table["hour"] < 12), "settings_known_slots"] = 0
    table.loc[window & (table["date"] == origin + pd.Timedelta(days=2)), "n_valid"] = 6
    table.loc[window & (table["date"] == origin + pd.Timedelta(days=2)), "n_in_range"] = 6
    share, coverage = fh.window_target(table, origin, "tir")
    rows = table[window]
    assert share == pytest.approx(rows["n_in_range"].sum() / rows["n_valid"].sum())
    assert coverage == pytest.approx(rows["n_valid"].sum() / 4032)
    unweighted = rows.groupby("date").apply(lambda day: day["n_in_range"].sum() / day["n_valid"].sum(), include_groups=False).mean()
    assert share != pytest.approx(unweighted)


def test_a_window_with_too_few_readings_is_not_scored_and_says_why() -> None:
    table = synthetic_table(days=420, first=pd.Timestamp("2024-06-01"))
    origin = fh.evaluation_origins(table["date"].max(), count=1)[0]
    table.loc[(table["date"] >= origin) & (table["date"] < origin + pd.Timedelta(days=14)) & (table["date"].dt.dayofweek < 5), "n_valid"] = 0
    scores, origins, _ = fh.score_origins(table, [origin], outcome="tir", daily=fh.daily_counts(table, "tir"), tdd=constant_tdd(table), profiles=constant_profiles(table), set_name="evaluation",
                                          posteriors=None, predictors=fh.BASELINES)
    low = origins.iloc[0]
    assert not low["scored"] and "of its possible readings" in low["reason"]
    assert scores.empty


# ---- leakage -----------------------------------------------------------------

def test_known_holds_only_rows_before_the_origin_and_future_values_change_nothing() -> None:
    table = synthetic_table(days=520)
    origin = pd.Timestamp("2025-03-10")
    tdd, profiles = constant_tdd(table), constant_profiles(table)
    known = fh.known_at(origin, "tir", table, fh.daily_counts(table, "tir"), tdd, profiles)
    assert known.table["date"].max() == origin - pd.Timedelta(days=1)
    assert known.daily.index.max() == origin - pd.Timedelta(days=1) and known.tdd.index.max() == origin - pd.Timedelta(days=1)
    assert known.profiles.dates.max() == origin
    sentinel = table.copy()
    future = sentinel["date"] >= origin
    sentinel.loc[future, ["n_in_range", "n_below_70"]] = [0, 12]
    sentinel.loc[future, "carbs_prev3h_g"] = 1e6
    changed_tdd = tdd.copy()
    changed_tdd[changed_tdd.index >= origin] = 1e6
    other = fh.known_at(origin, "tir", sentinel, fh.daily_counts(sentinel, "tir"), changed_tdd, profiles)
    for name in ("last 28 days", "last 14 days", "recency-weighted"):
        assert np.array_equal(fh.baseline_draws(known, name, np.random.default_rng(1)), fh.baseline_draws(other, name, np.random.default_rng(1)))
    first, _ = fh.ridge_draws(known, np.random.default_rng(1))
    second, _ = fh.ridge_draws(other, np.random.default_rng(1))
    assert np.array_equal(first, second)
    assert np.array_equal(fh.exposure_pool(known), fh.exposure_pool(other))


def test_profiles_cut_at_the_origin_hide_later_dates() -> None:
    table = synthetic_table(days=40)
    profiles = constant_profiles(table)
    assert profiles.upto(pd.Timestamp("2024-01-10")).dates.max() == pd.Timestamp("2024-01-10")


# ---- frozen settings ---------------------------------------------------------

def test_frozen_profile_is_the_profile_at_the_last_moment_before_the_origin() -> None:
    table = synthetic_table(days=60, first=pd.Timestamp("2025-01-20"))
    profiles = fh.frozen_profiles(FakeHistory(), table)
    assert profiles.at(pd.Timestamp("2025-02-10"))[:, 0].tolist() == [10.0] * 24  # the change comes at 12:00 on the 10th, after 23:55 of the 9th
    assert profiles.at(pd.Timestamp("2025-02-11"))[:, 0].tolist() == [8.0] * 24
    assert np.isnan(profiles.at(pd.Timestamp("2025-01-20"))).all()  # nothing precedes the first day


def test_frozen_settings_ignore_a_change_inside_the_window_and_the_planned_variant_follows_it() -> None:
    table = synthetic_table(days=80, first=pd.Timestamp("2025-01-20"))
    origin = pd.Timestamp("2025-02-05")
    changed = table["date"] >= pd.Timestamp("2025-02-10")
    table.loc[changed, "icr"] = 8.0
    profiles = fh.frozen_profiles(FakeHistory(), table)
    known = fh.known_at(origin, "tir", table, fh.daily_counts(table, "tir"), constant_tdd(table), profiles)
    window = fh.window_for(table, origin)
    assert fh.settings_changed_in_window(window, profiles.at(origin))
    posterior = fake_posterior(origin)
    frozen = fh.model_draws(posterior, known, window, np.random.default_rng(3), planned=False)
    unchanged = table.copy()
    unchanged["icr"] = 10.0
    again = fh.model_draws(posterior, known, fh.window_for(unchanged, origin), np.random.default_rng(3), planned=False)
    assert np.array_equal(frozen, again)
    planned = fh.model_draws(posterior, known, window, np.random.default_rng(3), planned=True)
    posterior_strong = fake_posterior(origin)
    posterior_strong = fh.ModelPosterior(**{**posterior_strong.__dict__, "beta": np.tile(np.array([-2.0, 0, 0, 0, 0, 0, 0, 0]), (200, 1))})
    differ = fh.model_draws(posterior_strong, known, window, np.random.default_rng(3), planned=True).mean() != fh.model_draws(posterior_strong, known, window, np.random.default_rng(3), planned=False).mean()
    assert differ and planned.shape == frozen.shape


def test_a_window_without_a_settings_change_is_flagged_false() -> None:
    table = synthetic_table(days=40)
    origin = pd.Timestamp("2024-01-20")
    assert not fh.settings_changed_in_window(fh.window_for(table, origin), constant_profiles(table).at(origin))


def test_the_model_is_not_scored_where_the_frozen_profile_is_unknown() -> None:
    table = synthetic_table(days=60)
    origin = pd.Timestamp("2024-02-10")
    profiles = constant_profiles(table)
    values = profiles.values.copy()
    values[profiles.dates.get_loc(origin), 5] = np.nan
    known = fh.known_at(origin, "tir", table, fh.daily_counts(table, "tir"), constant_tdd(table), fh.SettingsProfiles(profiles.dates, values))
    assert "unknown" in fh.model_unscorable_reason(known, fh.window_for(table, origin), planned=False)


# ---- baselines ---------------------------------------------------------------

def test_baseline_point_and_residuals_use_only_days_and_windows_before_the_origin() -> None:
    rng = np.random.default_rng(2)
    y, n = rng.integers(60, 200, 400).astype(float), np.full(400, 288.0)
    points = fh.lookback_points(y, n, "last 28 days")
    changed = y.copy()
    changed[300:] = 0.0
    assert points[300] == fh.lookback_points(changed, n, "last 28 days")[300]  # the point at position 300 never sees day 300 or later
    assert points[300] == pytest.approx(y[272:300].sum() / n[272:300].sum())
    recency = fh.lookback_points(y, n, "recency-weighted")
    assert recency[300] == fh.lookback_points(changed, n, "recency-weighted")[300] and np.isnan(recency[89]) and np.isfinite(recency[90])
    pool = fh.residual_pool(y[:300], n[:300], "last 28 days")
    shares = np.array([y[s:s + 14].sum() / n[s:s + 14].sum() for s in range(28, 300 - 13)])
    expected = fh._clipped_logit(shares) - fh._clipped_logit(np.array([points[s] for s in range(28, 300 - 13)]))
    assert np.allclose(np.sort(pool), np.sort(expected))
    assert len(pool) == 300 - 13 - 28  # a window that would end on or after day 300 is not in the pool


def test_a_constant_series_gives_a_degenerate_forecast_with_zero_crps() -> None:
    table = synthetic_table(days=200)
    table["n_in_range"] = 9
    table["n_below_70"] = 3
    daily = fh.daily_counts(table, "tir")
    origin = pd.Timestamp("2024-06-01")
    known = fh.known_at(origin, "tir", table, daily, constant_tdd(table), constant_profiles(table))
    for name in fh.BASELINES:
        draws = fh.baseline_draws(known, name, np.random.default_rng(0))
        assert np.allclose(draws, 0.75)
        assert wt.crps_points(draws, 0.75) == pytest.approx(0.0, abs=1e-9)


def test_baseline_spread_is_its_own_past_error() -> None:
    table = synthetic_table(days=300, seed=4)
    known = fh.known_at(pd.Timestamp("2024-10-01"), "tir", table, fh.daily_counts(table, "tir"), constant_tdd(table), constant_profiles(table))
    draws = fh.baseline_draws(known, "last 28 days", np.random.default_rng(0))
    y, n = known.daily["y"].to_numpy(), known.daily["n"].to_numpy()
    pool = fh.residual_pool(y, n, "last 28 days")
    point = fh.lookback_points(y, n, "last 28 days")[len(y)]
    assert set(np.round(draws, 9)) <= set(np.round(1 / (1 + np.exp(-(np.log(point / (1 - point)) + pool))), 9))
    assert draws.std() > 0


def test_a_baseline_without_enough_history_is_not_scored() -> None:
    table = synthetic_table(days=60)
    known = fh.known_at(pd.Timestamp("2024-02-15"), "tir", table, fh.daily_counts(table, "tir"), constant_tdd(table), constant_profiles(table))
    with pytest.raises(fh.NotScored, match="earlier windows"):
        fh.baseline_draws(known, "last 28 days", np.random.default_rng(0))


# ---- doses ---------------------------------------------------------------------

def test_trailing_dose_uses_only_days_before_each_day() -> None:
    dose = pd.Series(np.arange(1.0, 61.0), index=pd.date_range("2025-01-01", periods=60, freq="D"))
    trailing = fh.trailing_tdd(dose)
    day = pd.Timestamp("2025-02-20")
    assert trailing[day] == pytest.approx(dose[day - pd.Timedelta(days=28):day - pd.Timedelta(days=1)].mean())
    changed = dose.copy()
    changed[changed.index >= day] = 1e6
    assert fh.trailing_tdd(changed)[day] == trailing[day]
    assert np.isnan(trailing.iloc[10])  # fewer than 14 earlier days


def test_daily_insulin_adds_bolus_and_delivered_basal_and_needs_70_percent_basal_coverage() -> None:
    start = pd.date_range("2025-01-01", "2025-01-03 23:55", freq="5min")
    grid = pd.DataFrame({"timestamp": start, "bolus_units": 0.0})
    grid.loc[grid["timestamp"] == pd.Timestamp("2025-01-01 08:00"), "bolus_units"] = 5.0
    basal = pd.DataFrame({"start_timestamp": start, "end_timestamp": start + pd.Timedelta(minutes=5), "basal_units_per_hour": 1.2})
    basal = basal[~((basal["start_timestamp"] >= "2025-01-03") & (basal["start_timestamp"] < "2025-01-03 12:00"))]  # half of the third day is missing
    dose = fh.daily_insulin(grid, basal)
    assert dose[pd.Timestamp("2025-01-01")] == pytest.approx(5.0 + 1.2 * 24)
    assert dose[pd.Timestamp("2025-01-02")] == pytest.approx(1.2 * 24)
    assert np.isnan(dose[pd.Timestamp("2025-01-03")])


# ---- exposure resampling ------------------------------------------------------

def test_exposure_resampling_draws_only_complete_days_from_the_28_days_before_the_origin() -> None:
    table = synthetic_table(days=120)
    origin = pd.Timestamp("2024-04-10")
    day_before = table["date"] == origin - pd.Timedelta(days=3)
    table.loc[day_before & (table["hour"] == 4), "sleep_mode_share"] = np.nan  # an incomplete day cannot be resampled
    known = fh.known_at(origin, "tir", table, fh.daily_counts(table, "tir"), constant_tdd(table), constant_profiles(table))
    pool = fh.exposure_pool(known)
    index = (origin - FIRST_DAY).days
    first = np.log1p(float(index - 28))
    carbs_days = {round(float(day[0, 0]), 6) for day in pool}
    allowed = {round(np.log1p(float(k)), 6) for k in range(index - 28, index)} - {round(np.log1p(float(index - 3)), 6)}
    assert carbs_days == allowed and min(carbs_days) >= round(first, 6)
    assert pool.shape == (27, 24, 4)
    assert fh.model_draws(fake_posterior(origin), known, fh.window_for(table, origin), np.random.default_rng(0), planned=False).shape == (200,)


def test_model_draws_are_a_share_between_zero_and_one_from_the_realised_reading_counts() -> None:
    table = synthetic_table(days=120)
    origin = pd.Timestamp("2024-04-10")
    window = fh.window_for(table, origin)
    known = fh.known_at(origin, "tir", table, fh.daily_counts(table, "tir"), constant_tdd(table), constant_profiles(table))
    draws = fh.model_draws(fake_posterior(origin), known, window, np.random.default_rng(0), planned=False)
    assert ((draws >= 0) & (draws <= 1)).all() and 0.5 < draws.mean() < 0.95


# ---- CRPS and the win test -------------------------------------------------------

def test_crps_matches_the_closed_form_for_a_normal_and_a_hand_computed_two_point_case() -> None:
    n = 40000
    draws = norm.ppf((np.arange(n) + 0.5) / n)
    z = 0.7
    closed = z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)
    assert wt.crps_points(draws, z) / 100 == pytest.approx(closed, abs=2e-3)
    assert wt.crps_points(np.array([0.0, 1.0]), 0.25) == pytest.approx(25.0)
    assert wt.crps_points(np.array([0.4, 0.4, 0.4]), 0.4) == pytest.approx(0.0)


def test_circular_block_indices_wrap_around_and_are_reproducible() -> None:
    indices = wt.circular_block_indices(10, 4, 200, seed=7)
    assert indices.shape == (200, 10) and indices.min() >= 0 and indices.max() <= 9
    assert np.array_equal(indices, wt.circular_block_indices(10, 4, 200, seed=7))
    assert not np.array_equal(indices, wt.circular_block_indices(10, 4, 200, seed=8))
    for row in indices[:50]:
        for start in range(0, 8, 4):  # inside a block the positions step by one, modulo the length
            assert (np.diff(row[start:start + 4]) % 10 == 1).all()
    assert (indices == 0).any(axis=1).any() and ((indices.max(axis=1) == 9) & (indices.min(axis=1) == 0)).any()  # some blocks run past the end back to the start
    with pytest.raises(ValueError):
        wt.circular_block_indices(5, 6, 10, seed=1)


def test_bootstrap_interval_is_reproducible_and_recovers_a_known_shift() -> None:
    rng = np.random.default_rng(9)
    shifted = -0.5 + rng.normal(0, 0.3, 52)
    first = wt.paired_interval(shifted, block=4)
    assert first == wt.paired_interval(shifted, block=4)
    assert first.lower < first.mean < first.upper and first.upper < 0 and first.verdict == "win"
    assert first.mean == pytest.approx(shifted.mean())
    assert wt.paired_interval(rng.normal(0, 0.3, 52), block=4).verdict == "tie"
    assert wt.paired_interval(0.5 + rng.normal(0, 0.3, 52), block=4).verdict == "loss"


def test_verdict_needs_the_whole_interval_below_zero() -> None:
    assert wt.verdict_from_interval(-0.9, -0.01) == "win"
    assert wt.verdict_from_interval(-0.9, 0.0) == "tie"
    assert wt.verdict_from_interval(-0.2, 0.4) == "tie"
    assert wt.verdict_from_interval(0.1, 0.4) == "loss"


def test_a_lower_mean_crps_with_an_interval_spanning_zero_is_a_tie_and_a_concern() -> None:
    rng = np.random.default_rng(1)
    base = rng.uniform(2, 4, 52)
    noise = rng.normal(0, 0.6, 52)
    candidate = base + (noise - noise.mean() - 0.02)  # a mean difference of exactly -0.02 points inside a wide interval
    table = pd.DataFrame({fh.REFERENCE: base, "hour-day model": candidate}, index=pd.date_range("2025-11-12", periods=52, freq="7D"))
    comparisons = report.build_comparisons(table)
    primary = comparisons[comparisons["role"] == "win test"].iloc[0]
    assert primary["mean_difference_points"] < 0 and primary["verdict"] == "tie" and primary["status"] == "concern"
    sensitivity = comparisons[(comparisons["candidate"] == "hour-day model") & (comparisons["role"] == "sensitivity (block length)")]
    assert sorted(sensitivity["block_length_origins"]) == [2, 8]


def rolling_scores(model_difference: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    base = rng.uniform(0.02, 0.05, len(model_difference))
    rows = []
    for index, (b, d) in enumerate(zip(base, model_difference)):
        date = (pd.Timestamp("2026-09-01") + pd.Timedelta(days=index)).date()
        for name, value in (("last 28 days", b), ("hour-day model", b + d)):
            rows.append({"origin": date, "date": date, "predictor": name, "observed": 0.7, "predicted_mean": 0.7, "lower_90": 0.6, "upper_90": 0.8, "crps": value, "inside_90": True})
    return pd.DataFrame(rows)


def test_workbook_rule_no_longer_calls_a_lower_mean_crps_no_concern() -> None:
    """Regression: the rolling-origin check used to be "no concern" whenever the model's mean CRPS was lower."""
    noisy = np.random.default_rng(4).normal(0, 0.01, 40)
    noisy = noisy - noisy.mean() - 0.0005  # a mean difference of -0.05 points inside a wide interval
    scores = rolling_scores(noisy)
    summary = omr.origin_summary(scores)
    assert summary.loc["hour-day model", "crps_points"] < summary.loc["last 28 days", "crps_points"]
    text, status = omr.rolling_origin_check(scores)
    assert status == "concern" and "90% interval" in text and "a tie" in text
    clear, clear_status = omr.rolling_origin_check(rolling_scores(np.full(40, -0.01) + np.random.default_rng(5).normal(0, 0.0005, 40)))
    assert clear_status == "no concern" and "a win" in clear
    assert omr.rolling_origin_check(rolling_scores(np.full(40, 0.01)))[1] == "concern"


# ---- the whole run on synthetic data ------------------------------------------------

def test_scoring_and_reporting_a_synthetic_record_end_to_end(tmp_path) -> None:
    table = synthetic_table(days=700)
    daily = fh.daily_counts(table, "tir")
    profiles, tdd = constant_profiles(table), constant_tdd(table)
    evaluation, development = fh.evaluation_origins(daily.index.max()), fh.development_origins()
    fh.check_origins_in_record(evaluation + development, daily.index.min(), daily.index.max())
    posteriors = {origin: fake_posterior(origin) for origin in fh.scorable_for_model(table, evaluation, profiles, outcome="tir", daily=daily, tdd=tdd)}
    assert len(posteriors) == 52
    results = {}
    for name, origins in (("evaluation", evaluation), ("development", development)):
        results[name] = fh.score_origins(table, origins, outcome="tir", daily=daily, tdd=tdd, profiles=profiles, set_name=name, posteriors=posteriors if name == "evaluation" else None,
                                         predictors=fh.PREDICTORS if name == "evaluation" else fh.PREDICTORS[:4])
    scores = results["evaluation"][0]
    assert scores[scores["reason"] == ""].groupby("predictor").size().to_dict() == {name: 52 for name in fh.PREDICTORS}
    report.write_outputs(folder=tmp_path, run_id="synthetic", raw="synthetic", outcome="tir", table=table, results=results, posteriors=posteriors, draws=1000, tune=1000, chains=4, seconds=1.0, score_model=True)
    workbook = read_workbook(tmp_path / "diagnostics.xlsx")
    names = list(workbook)
    assert names[:2] == ["Specification_Coding", "Denominators_Flow"] and names[-2:] == ["Warnings_Limitations", "Artifacts_Provenance"] and len(names) <= 13
    for sheet in workbook.values():
        if "status" in sheet:
            assert set(sheet["status"]) <= set(STATUSES)
    assert (tmp_path / "forecast_harness_figure.png").stat().st_size > 10_000 and (tmp_path / "forecast_harness_summary.md").read_text().count("Model minus last 28 days") == 1
    win_test = workbook["Win_Test"]
    primary = win_test[win_test["role"] == "win test"].iloc[0]
    assert primary["status"] == ("no concern" if primary["verdict"] == "win" else "concern")
    assert (workbook["Model_Fits"]["status"] == "no concern").all()


def test_the_run_without_the_model_says_it_was_not_assessed(tmp_path) -> None:
    table = synthetic_table(days=700)
    daily = fh.daily_counts(table, "tir")
    profiles, tdd = constant_profiles(table), constant_tdd(table)
    results = {name: fh.score_origins(table, origins, outcome="tir", daily=daily, tdd=tdd, profiles=profiles, set_name=name, posteriors=None, predictors=fh.PREDICTORS[:4])
               for name, origins in (("evaluation", fh.evaluation_origins(daily.index.max())), ("development", fh.development_origins()))}
    report.write_outputs(folder=tmp_path, run_id="synthetic", raw="synthetic", outcome="tir", table=table, results=results, posteriors=None, draws=1000, tune=1000, chains=4, seconds=1.0, score_model=False)
    workbook = read_workbook(tmp_path / "diagnostics.xlsx")
    checks = workbook["Checks"].set_index("check")["status"]
    assert checks["Win test"] == "not assessed" and checks["Sampler gate on every refit"] == "not assessed" and "Model_Fits" not in workbook
