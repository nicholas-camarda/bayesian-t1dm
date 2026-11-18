from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bayesian_t1dm import episodes as ep
from bayesian_t1dm.outcomes import DayPart

START = pd.Timestamp("2026-06-01 00:00")


def _grid(glucose, *, carbs=None, bolus=None, iob=0.0, icr=7.0):
    index = pd.date_range(START, periods=len(glucose), freq="5min")
    frame = pd.DataFrame({"timestamp": index, "glucose": glucose, "iob_units": iob, "bolus_units": 0.0, "carb_grams": 0.0, "carb_ratio_g_per_unit": icr})
    for when, grams in (carbs or {}).items():
        frame.loc[frame["timestamp"].eq(pd.Timestamp(when)), "carb_grams"] = grams
    for when, units in (bolus or {}).items():
        frame.loc[frame["timestamp"].eq(pd.Timestamp(when)), "bolus_units"] = units
    return frame


def _series(parts):
    """Concatenate (value, readings) blocks into a glucose list."""
    return np.concatenate([np.full(n, v, dtype=float) for v, n in parts])


NO_SITES = pd.DataFrame(columns=["timestamp", "kind"])


def _classify(grid, sites=NO_SITES, workouts=None, delivery=None):
    cgm = grid[["timestamp", "glucose"]]
    found = ep.find_episodes(cgm, start=grid["timestamp"].min(), end=grid["timestamp"].max() + pd.Timedelta(minutes=5))
    return ep.classify_all(ep.describe_episodes(found, grid, sites, None, delivery, workouts, site_coverage_from=START))


def test_episodes_need_their_minimum_length():
    glucose = _series([(120, 50), (260, 17), (120, 20), (260, 18), (120, 20), (60, 3), (120, 20), (60, 2)])  # 85, 90 min highs; 15, 10 min lows
    found = ep.find_episodes(pd.DataFrame({"timestamp": pd.date_range(START, periods=len(glucose), freq="5min"), "glucose": glucose}), start=START, end=START + pd.Timedelta(days=1))
    kinds = found.groupby("kind").size().to_dict()
    assert kinds == {"high": 1, "low": 1}  # 90-minute high and 15-minute low count; 85 and 10 minutes do not


def test_a_meal_dosed_short_of_the_pumps_own_ratio_is_named_with_the_numbers():
    glucose = _series([(120, 24), (260, 30), (120, 20)])
    grid = _grid(glucose, carbs={"2026-06-01 01:30": 70.0}, bolus={"2026-06-01 01:30": 6.0}, icr=7.0)  # needs 10 U, got 6
    row = _classify(grid).iloc[0]
    assert row["pattern"] == "meal_underdosed" and "70 g logged, 6.0 U given" in row["evidence"] and "10.0 U" in row["evidence"]


def test_a_meal_dosed_as_calculated_that_still_goes_high_is_its_own_pattern():
    glucose = _series([(120, 24), (260, 30), (120, 20)])
    grid = _grid(glucose, carbs={"2026-06-01 01:30": 70.0}, bolus={"2026-06-01 01:30": 10.0}, icr=7.0)
    assert _classify(grid).iloc[0]["pattern"] == "meal_dosed_high"


def test_a_site_change_near_the_end_of_a_long_high_with_corrections_points_at_the_site():
    glucose = _series([(120, 24), (300, 48), (120, 20)])  # 4 hours high
    grid = _grid(glucose, bolus={"2026-06-01 03:00": 1.5})
    sites = pd.DataFrame({"timestamp": [pd.Timestamp("2026-06-01 05:30")], "kind": ["cannula filled"]})
    row = _classify(grid, sites).iloc[0]
    assert row["pattern"] == "site_or_insulin" and "despite 1.5 U of corrections" in row["evidence"]
    assert _classify(grid).iloc[0]["pattern"] == "persistent"  # without the site change it is a persistent high


def test_a_low_after_a_workout_and_a_meal_bolus_is_named_and_a_short_walk_does_not_count():
    glucose = _series([(120, 40), (60, 6), (120, 10)])
    grid = _grid(glucose, bolus={"2026-06-01 02:00": 4.0})
    long = pd.DataFrame({"start_timestamp": [START], "end_timestamp": [pd.Timestamp("2026-06-01 01:00")], "duration_seconds": [3600.0]})
    short = long.assign(duration_seconds=[300.0])
    assert _classify(grid, workouts=long).iloc[0]["pattern"] == "after_exercise"
    assert _classify(grid, workouts=short).iloc[0]["pattern"] != "after_exercise"


def test_repeats_need_enough_episodes_in_one_segment_and_a_majority_pattern():
    rows = [{"kind": "high", "start": pd.Timestamp(f"2026-06-{d:02d} 16:30"), "pattern": "meal_dosed_high"} for d in range(1, 6)] + [{"kind": "high", "start": pd.Timestamp("2026-06-09 16:30"), "pattern": "persistent"}]
    parts = [DayPart("09:00-15:00", 540, 900), DayPart("15:00-19:00", 900, 1140)]
    found = ep.repeats(pd.DataFrame(rows), parts)
    assert len(found) == 1 and found[0]["segment"] == "15:00-19:00" and (found[0]["n"], found[0]["of"]) == (5, 6) and "I:C" in found[0]["points_at"]
    assert ep.repeats(pd.DataFrame(rows[:3]), parts) == []


def test_exercise_effect_compares_lows_after_meals_with_and_without_a_recent_workout():
    days = 10
    index = pd.date_range(START, periods=288 * days, freq="5min")
    glucose = np.full(len(index), 120.0)
    grid = pd.DataFrame({"timestamp": index, "glucose": glucose, "bolus_units": 0.0, "carb_grams": 0.0})
    meals = [START + pd.Timedelta(days=d, hours=12) for d in range(days)]
    for when in meals:
        grid.loc[grid["timestamp"].eq(when), ["bolus_units", "carb_grams"]] = [3.0, 40.0]
    for when in meals[:5]:  # a long workout the morning of the first five meals, and a low after each
        low = grid["timestamp"].between(when + pd.Timedelta(hours=1), when + pd.Timedelta(hours=2))
        grid.loc[low, "glucose"] = 60.0
    workouts = pd.DataFrame({"start_timestamp": [m - pd.Timedelta(hours=5) for m in meals[:5]], "end_timestamp": [m - pd.Timedelta(hours=4) for m in meals[:5]], "duration_seconds": 3600.0})
    result = ep.exercise_effect(grid, workouts)
    assert result["after_workout_meals"] == 5 and result["after_workout_lows"] == 5 and result["other_meals"] == 5 and result["other_lows"] == 0
    assert result["after_workout_interval"][0] > result["other_interval"][1]


def test_wilson_interval_brackets_the_proportion():
    low, high = ep.wilson(14, 120)
    assert low < 14 / 120 < high and ep.wilson(0, 0)[0] != ep.wilson(0, 0)[0]
    assert ep.wilson(0, 30)[0] == 0.0 and pytest.approx(ep.wilson(30, 30)[1]) == 1.0
