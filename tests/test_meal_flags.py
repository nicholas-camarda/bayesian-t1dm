from __future__ import annotations

import numpy as np
import pandas as pd

from bayesian_t1dm import meal_flags


def _frame(glucose, *, carbs_at=(), bolus_at=()):
    index = pd.date_range("2026-03-01 06:00", periods=len(glucose), freq="5min")
    frame = pd.DataFrame({"timestamp": index, "glucose": glucose, "explicit_carb_grams": 0.0, "bolus_units": 0.0})
    for t in carbs_at:
        frame.loc[frame["timestamp"].eq(pd.Timestamp(t)), "explicit_carb_grams"] = 40.0
    for t in bolus_at:
        frame.loc[frame["timestamp"].eq(pd.Timestamp(t)), "bolus_units"] = 4.0
    return frame


def _food(*times, carbs=30.0):
    return pd.DataFrame({"timestamp": pd.to_datetime(list(times)), "carb_grams": carbs, "protein_grams": 5.0, "fat_grams": 5.0})


def test_food_is_flagged_only_when_no_pump_carbs_or_bolus_is_near():
    frame = _frame([100.0] * 120, carbs_at=["2026-03-01 07:00"])
    food = _food("2026-03-01 07:10", "2026-03-01 11:00")  # the first matches the pump meal; the second has nothing
    flagged = meal_flags.food_without_pump_carbs(frame, food)
    assert flagged["event_time"].tolist() == [pd.Timestamp("2026-03-01 11:00")]
    assert flagged.loc[0, "end"] - flagged.loc[0, "event_time"] == pd.Timedelta(minutes=meal_flags.AFTER_FLAG_MINUTES)
    assert meal_flags.food_without_pump_carbs(frame, _food("2026-03-01 11:00", carbs=2.0)).empty  # below the carb floor


def test_a_bolus_without_carbs_also_explains_nearby_food():
    frame = _frame([100.0] * 120, bolus_at=["2026-03-01 09:00"])
    assert meal_flags.food_without_pump_carbs(frame, _food("2026-03-01 09:20")).empty


def test_a_missing_bevel_entry_is_reported_not_treated_as_no_meal():
    frame = _frame([100.0] * 120, carbs_at=["2026-03-01 07:00", "2026-03-01 10:00"])
    missing = meal_flags.bevel_missing(frame, _food("2026-03-01 07:05"))
    assert missing["event_time"].tolist() == [pd.Timestamp("2026-03-01 10:00")]
    assert meal_flags.unlogged_intervals(frame, _food("2026-03-01 07:05")).empty  # no exclusion comes from it


def test_a_rise_with_no_logged_cause_is_flagged_once_and_a_logged_one_is_not():
    glucose = [100.0] * 60 + list(np.linspace(100, 190, 12)) + [190.0] * 48
    quiet = _frame(glucose)
    flagged = meal_flags.possible_unlogged_food(quiet, None)
    assert len(flagged) == 1 and flagged.loc[0, "kind"] == "possible_unlogged_food"
    logged = _frame(glucose, carbs_at=["2026-03-01 11:00"])  # 60 readings after 06:00 = 11:00
    assert meal_flags.possible_unlogged_food(logged, None).empty
    assert meal_flags.possible_unlogged_food(quiet, _food("2026-03-01 11:10")).empty


def test_overlap_reports_kinds_and_can_ignore_the_meal_itself():
    intervals = pd.DataFrame([meal_flags._interval("food_without_pump_carbs", pd.Timestamp("2026-03-01 09:00"))])
    assert meal_flags.overlaps(intervals, pd.Timestamp("2026-03-01 08:00"), pd.Timestamp("2026-03-01 11:00")) == ["food_without_pump_carbs"]
    assert meal_flags.overlaps(intervals, pd.Timestamp("2026-03-01 12:30"), pd.Timestamp("2026-03-01 14:00")) == []
    assert meal_flags.overlaps(intervals, pd.Timestamp("2026-03-01 08:00"), pd.Timestamp("2026-03-01 11:00"), ignore_event_within=(pd.Timestamp("2026-03-01 09:00"), 30)) == []
