from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bayesian_t1dm import controller
from bayesian_t1dm.outcomes import DayPart

NIGHT = DayPart("00:00-05:00", 0, 300)


def _rates(days=3, commanded=1.5, profile=1.0):
    index = pd.date_range("2026-06-01", periods=days * 288, freq="5min")
    return pd.DataFrame({"pump_serial": "1", "sequence": np.arange(len(index)), "timestamp": index, "commanded_u_per_h": commanded, "profile_u_per_h": profile})


def _cgm(days=3, glucose=130.0):
    index = pd.date_range("2026-06-01", periods=days * 288, freq="5min")
    return pd.DataFrame({"timestamp": index, "glucose": glucose})


NO_MODES = pd.DataFrame(columns=["pump_serial", "sequence", "timestamp", "mode"])
NO_MEALS = pd.DataFrame({"timestamp": pd.to_datetime([]), "window_hours": []})


def _meals(*times, hours=4.0):
    return pd.DataFrame({"timestamp": pd.to_datetime(list(times)), "window_hours": hours})


def test_extra_insulin_is_the_commanded_rate_over_the_profile_rate_as_a_share_of_the_profile():
    daily = controller.daily_controller_pattern(_rates(), NO_MODES, _cgm(), NO_MEALS, [NIGHT])
    assert len(daily) == 3
    assert daily["excess_fraction"].round(3).tolist() == [0.5, 0.5, 0.5]
    assert daily.loc[0, "excess_units"] == pytest.approx(0.5 * 5, rel=0.02)  # 0.5 U/h extra over 5 hours
    assert daily.loc[0, "mean_glucose"] == pytest.approx(130.0)


def test_time_after_a_meal_or_bolus_is_left_out_and_a_day_with_too_little_left_does_not_count():
    meals = _meals("2026-06-01 00:00", "2026-06-02 02:00")
    daily = controller.daily_controller_pattern(_rates(), NO_MODES, _cgm(), meals, [NIGHT])
    # Day 1: the first 4 of 5 hours are after the meal, leaving 1 hour (under 70%): the day does not count.
    # Day 2: a meal at 02:00 leaves 00:00-02:00 only (2 of 5 hours): not counted either. Day 3 is clean.
    assert daily["date"].dt.day.tolist() == [3]


def test_exercise_mode_and_the_hour_after_it_are_left_out_but_sleep_mode_is_kept():
    modes = pd.DataFrame(
        {"pump_serial": "1", "sequence": [1, 2, 3], "timestamp": pd.to_datetime(["2026-06-01 00:00", "2026-06-02 00:00", "2026-06-02 01:00"]), "mode": [1, 2, 0]}
    )
    daily = controller.daily_controller_pattern(_rates(), modes, _cgm(), NO_MEALS, [NIGHT])
    # Day 2: exercise 00:00-01:00 plus the hour after leaves 3 of 5 hours, under 70%, so it does not count.
    assert daily["date"].dt.day.tolist() == [1, 3]
    first = daily[daily["date"].dt.day == 1].iloc[0]
    assert first["sleep_share"] == pytest.approx(1.0)


def test_withholding_gives_a_negative_share_and_context_days_are_excluded():
    daily = controller.daily_controller_pattern(_rates(commanded=0.5), NO_MODES, _cgm(), NO_MEALS, [NIGHT], excluded_days=[pd.Timestamp("2026-06-02")])
    assert daily["date"].dt.day.tolist() == [1, 3] and daily["excess_fraction"].round(3).tolist() == [-0.5, -0.5]


def test_summary_reports_the_spread_across_days():
    daily = pd.DataFrame({"part": "00:00-05:00", "excess_fraction": [0.0, 0.2, 0.4, 0.6], "mean_glucose": 140.0, "time_in_range_pct": 80.0, "sleep_share": 1.0})
    summary = controller.controller_summary(daily).iloc[0]
    assert summary["days"] == 4 and summary["median_excess_fraction"] == pytest.approx(0.3) and summary["sd_between_days"] > 0


def _days(fractions, hours=5.0, rate=0.7, glucose=150.0, below_70=0.5, part="00:00-05:00", start="2026-06-01"):
    fractions = np.asarray(fractions, dtype=float)
    hours = np.broadcast_to(np.asarray(hours, dtype=float), fractions.shape)
    units = rate * hours
    return pd.DataFrame({"date": pd.date_range(start, periods=len(fractions)), "part": part, "clean_hours": hours, "commanded_units": units + fractions * units, "profile_units": units, "excess_units": fractions * units,
                         "excess_fraction": fractions, "profile_rate": rate, "mean_glucose": glucose, "below_70_pct": below_70, "below_54_pct": 0.0})


def _signal(frame):
    return controller.basal_signal(frame).iloc[0]


def test_basal_signal_needs_extra_insulin_for_half_the_clean_time_glucose_still_high_and_lows_within_limits():
    assert _signal(_days([0.4] * 20))["signal"] == "basal likely too low"
    assert _signal(_days([0.4] * 10 + [0.0] * 10))["signal"] == "basal likely too low"  # exactly half the clean time
    assert _signal(_days([0.4] * 9 + [0.0] * 11))["signal"] == "no consistent pattern"
    assert _signal(_days([0.4] * 20, below_70=5.0))["signal"].startswith("controller adds insulin but lows are over a limit")
    assert _signal(_days([0.4] * 20, glucose=110.0))["signal"].startswith("controller adds insulin and glucose is within the target")
    assert _signal(_days([-0.4] * 20))["signal"] == "basal likely too high"


def test_basal_signal_gives_a_range_and_the_chance_the_rule_is_met():
    row = _signal(_days([0.4] * 20))
    assert row["share_over_x"] == pytest.approx(1.0) and row["prob_rule_met"] == pytest.approx(1.0) and row["pooled_excess"] == pytest.approx(0.4)
    mixed = _signal(_days([0.4] * 10 + [0.0] * 10))
    assert 0.2 < mixed["prob_rule_met"] < 0.8 and mixed["share_over_lo"] < 0.5 < mixed["share_over_hi"]


def test_each_day_counts_in_proportion_to_its_clean_hours():
    hours = [1.0] * 12 + [6.0] * 4  # twelve short loud days and four long quiet days
    frame = _days([0.5] * 12 + [0.0] * 4, hours=hours)
    row = _signal(frame)
    assert row["share_over_x"] == pytest.approx(12 / 36) and row["signal"] == "no consistent pattern"


def test_a_change_of_basal_rate_keeps_the_earlier_days_by_comparing_commanded_rates_with_the_current_one():
    # Before: profile 0.6 with 0.3 extra (commanded 0.78 U/h). After: profile 0.7 with none (commanded 0.7 U/h).
    old = _days([0.3] * 20, rate=0.6, start="2026-05-01")
    new = _days([0.0] * 12, rate=0.7, start="2026-05-21")
    row = _signal(pd.concat([old, new], ignore_index=True))
    assert row["profile_rate"] == 0.7 and row["days"] == 28 and row["signal"] == "no consistent pattern"
    # (16 old days x 5 h x 0.78 + 12 new days x 5 h x 0.7) / (28 x 5 h x 0.7) - 1
    assert row["pooled_excess"] == pytest.approx((16 * 5 * 0.78 + 12 * 5 * 0.7) / (28 * 5 * 0.7) - 1)
    # The same commanded rate that was only +30% over the old profile is +11% over the current one, below the 25% line.
    assert row["share_over_x"] == 0.0


def test_basal_signal_looks_only_at_the_most_recent_28_days_and_needs_enough_clean_time():
    assert _signal(_days([0.0] * 30 + [0.5] * 28))["signal"] == "basal likely too low"
    assert _signal(_days([0.4] * 8))["signal"].startswith("not enough clean time")
    assert _signal(_days([0.4] * 12, hours=1.0))["signal"].startswith("not enough clean time")  # 12 days but only 12 clean hours
    daytime = _signal(_days([0.4] * 14, hours=2.0, part="15:00-19:00"))  # 28 clean hours across 14 days is enough
    assert daytime["signal"] == "basal likely too low"


def test_loose_clean_time_rule_lets_a_partial_day_count_when_asked():
    rates = _rates(days=1)
    meals = _meals("2026-06-01 00:00")  # leaves 1 of 5 hours: under 70%
    strict = controller.daily_controller_pattern(rates, NO_MODES, _cgm(days=1), meals, [NIGHT])
    loose = controller.daily_controller_pattern(rates, NO_MODES, _cgm(days=1), meals, [NIGHT], min_clean_fraction=None, min_clean_hours=0.5)
    assert strict.empty and len(loose) == 1 and loose.loc[0, "clean_hours"] == pytest.approx(1.0, abs=0.1) and loose.loc[0, "profile_rate"] == 1.0


def test_a_segment_short_of_clean_time_widens_its_window_and_says_so():
    dates = pd.date_range("2026-04-01", periods=60, freq="4D")  # one clean day every 4 days: 7 fall in 28 days, 14 in 56
    frame = _days([0.4] * 60).assign(date=dates)
    row = _signal(frame)
    assert row["window_days"] == 56 and row["days"] >= 10 and row["signal"] == "basal likely too low"


def test_meal_windows_scale_with_the_meal_and_ignore_small_entries_and_automatic_corrections():
    carbs = pd.DataFrame({"timestamp": pd.to_datetime(["2026-06-01 08:00", "2026-06-01 10:00", "2026-06-01 12:00", "2026-06-01 14:00"]), "carb_grams": [10.0, 18.0, 35.0, 60.0]})
    bolus = pd.DataFrame({"timestamp": pd.to_datetime(["2026-06-01 09:00", "2026-06-01 11:00", "2026-06-01 13:00", "2026-06-01 15:00"]), "bolus_units": [0.3, 1.5, 3.0, 6.0]})
    windows = controller.meal_windows(bolus, carbs)
    assert windows["timestamp"].dt.hour.tolist() == [10, 11, 12, 13, 14, 15]  # the 10 g entry and the 0.3 U correction set no window
    assert windows["window_hours"].tolist() == [2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
    strict = controller.meal_windows(bolus, carbs, strict=True)
    assert len(strict) == 7 and set(strict["window_hours"]) == {4.0}  # every carb entry and the bolus of 0.5 U or more


def test_a_long_window_is_not_cut_short_by_a_later_short_one():
    meals = pd.DataFrame({"timestamp": pd.to_datetime(["2026-06-01 08:00", "2026-06-01 09:00"]), "window_hours": [4.0, 2.0]})
    times = pd.Series(pd.to_datetime(["2026-06-01 07:59", "2026-06-01 11:30", "2026-06-01 12:01", "2026-06-01 12:30"]))
    assert controller._inside_windows(times, meals).tolist() == [False, True, False, False]  # 08:00 + 4 h reaches 12:00 although the 09:00 + 2 h meal ends at 11:00


def test_a_day_with_no_clean_glucose_does_not_make_glucose_and_lows_unknown_for_the_window():
    frame = _days([0.4] * 20, below_70=5.0)
    frame.loc[0, ["mean_glucose", "below_70_pct", "below_54_pct"]] = np.nan
    row = _signal(frame)
    assert row["mean_glucose"] == pytest.approx(150.0) and row["below_70_pct"] == pytest.approx(5.0) and row["days_with_glucose"] == 19
    assert row["signal"].startswith("controller adds insulin but lows are over a limit")
    nothing = _signal(_days([0.4] * 20).assign(mean_glucose=np.nan, below_70_pct=np.nan, below_54_pct=np.nan))
    assert nothing["signal"].startswith("no clean glucose") and nothing["days_with_glucose"] == 0


def test_a_window_that_wraps_midnight_belongs_to_the_day_it_started():
    wrap = DayPart("21:00-04:00", 21 * 60, 4 * 60)
    stamps = pd.Series(pd.to_datetime(["2026-06-05 22:00", "2026-06-06 01:00", "2026-06-06 23:00"]))  # readings inside the bin
    assert wrap.analysis_date(stamps).tolist() == [pd.Timestamp("2026-06-05")] * 2 + [pd.Timestamp("2026-06-06")]
    assert DayPart("05:00-09:00", 300, 540).analysis_date(stamps).tolist() == list(stamps.dt.normalize())


def test_daily_units_and_hours_use_identical_known_pairs():
    rates = _rates(days=1).iloc[:4].copy()
    rates.loc[1, "profile_u_per_h"] = np.nan
    daily = controller.daily_controller_pattern(rates, NO_MODES, _cgm(), NO_MEALS, [NIGHT], min_clean_fraction=None)
    row = daily.iloc[0]
    assert row.clean_hours == pytest.approx(10 / 60)
    assert row.commanded_units == pytest.approx(1.5 * row.clean_hours)
    assert row.excess_units == pytest.approx(row.commanded_units - row.profile_units)


def test_controller_exclusion_counts_reconcile_with_daily_rows():
    flow = []
    daily = controller.daily_controller_pattern(_rates(), NO_MODES, _cgm(), NO_MEALS, [NIGHT],
                                                excluded_days=["2026-06-02"], flow=flow)
    assert flow[0]["days"] == 3
    assert flow[-1]["days"] == len(daily) == 2
    assert flow[-1]["hours"] == pytest.approx(daily.clean_hours.sum())
