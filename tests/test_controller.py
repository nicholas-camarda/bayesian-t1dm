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
NO_MEALS = np.array([], dtype="datetime64[ns]")


def test_extra_insulin_is_the_commanded_rate_over_the_profile_rate_as_a_share_of_the_profile():
    daily = controller.daily_controller_pattern(_rates(), NO_MODES, _cgm(), NO_MEALS, [NIGHT])
    assert len(daily) == 3
    assert daily["excess_fraction"].round(3).tolist() == [0.5, 0.5, 0.5]
    assert daily.loc[0, "excess_units"] == pytest.approx(0.5 * 5, rel=0.02)  # 0.5 U/h extra over 5 hours
    assert daily.loc[0, "mean_glucose"] == pytest.approx(130.0)


def test_time_after_a_meal_or_bolus_is_left_out_and_a_day_with_too_little_left_does_not_count():
    meals = np.array([np.datetime64("2026-06-01 00:00"), np.datetime64("2026-06-02 02:00")], dtype="datetime64[ns]")
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
