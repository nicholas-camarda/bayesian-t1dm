"""Tests for the personal readout on synthetic data (no clinical data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bayesian_t1dm import readout as ro
from bayesian_t1dm import readout_model as rm
from bayesian_t1dm import readout_settings as rs
from bayesian_t1dm.readout_page import Readout, render, rings


def test_baseline_z_is_per_device_and_uses_only_earlier_days() -> None:
    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    rng = np.random.default_rng(1)
    values = pd.Series(60 + rng.normal(0, 2, 60), index=dates)
    devices = pd.Series(["Oura"] * 30 + ["Apple Watch"] * 30, index=dates)
    values.iloc[29] = 90.0  # a spike on the last ring day
    values.iloc[30:] += 15.0  # the watch reads 15 higher: a device offset, not a change in the person
    z = ro.baseline_z(values, devices)
    assert z.iloc[29] > 5
    assert z.iloc[30:40].isna().all(), "a new device needs its own baseline days before a z is given"
    assert z.iloc[45:].abs().max() < 4, "the watch offset must not look like a change"


def test_site_age_counts_days_from_the_last_cannula_fill() -> None:
    events = pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-01 20:00", "2025-01-04 08:00"]), "kind": ["cannula filled", "cannula filled"]})
    dates = pd.date_range("2025-01-01", periods=6, freq="D")
    age = ro.site_age_at_midnight(dates, events, pd.Timestamp("2025-01-01"))
    assert np.isnan(age.iloc[0])
    assert age.iloc[1] == pytest.approx(4 / 24)
    assert age.iloc[3] == pytest.approx(2 + 4 / 24)
    assert age.iloc[4] == pytest.approx(16 / 24)


def test_recovery_score_is_fifty_on_a_usual_day_and_rises_with_hrv() -> None:
    score = ro.recovery_score(pd.Series([0.0, -1.0, np.nan]), pd.Series([0.0, 1.0, 1.0]))
    assert score.iloc[0] == pytest.approx(50.0)
    assert score.iloc[1] > 90
    assert score.iloc[2] > 80  # HRV alone when resting HR is missing


def synthetic_days(n: int = 400, *, seed: int = 7, short_sleep_effect: float = -0.5) -> pd.DataFrame:
    """A day table whose in-range share drops on short-sleep days (logit effect ``short_sleep_effect``)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    days = pd.DataFrame(index=pd.DatetimeIndex(dates, name="date"))
    for column, share in zip(ro.DRIVER_COLUMNS, (0.15, 0.3, 0.35, 0.35, 0.1, 0.15)):
        days[column] = rng.random(n) < share
    days["weekend"] = days.index.dayofweek >= 5
    eta = 0.6 + short_sleep_effect * days["short_sleep"] + rng.normal(0, 0.3, n)
    days["readings"] = 288
    days["in_range_readings"] = rng.binomial(288, 1 / (1 + np.exp(-eta)))
    days["below_70_readings"] = rng.binomial(288, 0.02, n)
    days["counted"] = True
    days["excluded_day"] = False
    days["site_age_days"] = 1.0
    return days


def test_drivers_model_recovers_a_planted_short_sleep_effect() -> None:
    days, _ = ro.model_days(synthetic_days())
    fit = rm.fit_outcome(days, "time_in_range", draws=400, tune=400, chains=2)
    effects = fit.effects.set_index("driver")
    sleep = effects.loc["short_sleep"]
    assert sleep["upper_pp"] < 0, "a planted drop in time in range on short-sleep days must be found with its sign"
    true_pp = 100 * (1 / (1 + np.exp(-0.1)) - 1 / (1 + np.exp(-0.6)))
    assert sleep["lower_pp"] - 3 < true_pp < sleep["upper_pp"] + 3
    others = effects.drop(index="short_sleep")
    assert (others["estimate_pp"].abs() < 3).all(), "drivers with no planted effect should stay near zero"


def test_model_days_flow_reconciles_and_leaves_out_unmeasured_days() -> None:
    days = synthetic_days(50).astype({c: object for c in ro.DRIVER_COLUMNS})
    days.iloc[0, days.columns.get_loc("counted")] = False
    days.iloc[1, days.columns.get_loc("excluded_day")] = True
    days.iloc[2, days.columns.get_loc("short_sleep")] = np.nan
    kept, flow = ro.model_days(days)
    assert len(kept) == 47
    assert flow[-1]["days"] == len(kept)
    assert sum(step["left_out"] for step in flow) == len(days) - len(kept)


def test_drivers_with_too_few_days_are_not_estimated() -> None:
    days = synthetic_days(100)
    days["late_meal"] = False
    days.iloc[:5, days.columns.get_loc("late_meal")] = True
    estimable, counts = rm.estimable_drivers(days)
    assert "late_meal" not in estimable
    assert not counts.set_index("driver").loc["late_meal", "estimated"]


def test_suggestions_move_insulin_the_right_way_and_respect_low_limits() -> None:
    status, text = rs.suggestion_for("basal", 0.7, "needs more insulin", 0.43, lows_ok=True)
    assert status == "needs more insulin" and "0.84" in text and "20% more" in text
    status, text = rs.suggestion_for("I:C", 6.8, "needs more insulin", 0.08, lows_ok=True)
    assert "6.1" in text and "10% more" in text, "more insulin lowers the I:C (fewer grams per unit)"
    status, text = rs.suggestion_for("I:C", 6.0, "needs less insulin", -0.12, lows_ok=True)
    assert "6.6" in text and "less" in text
    status, text = rs.suggestion_for("basal", 0.7, "needs more insulin", 0.43, lows_ok=False)
    assert status == rs.HELD


def test_icr_status_compares_the_setting_with_the_effective_range() -> None:
    row = pd.Series({"had_breakfasts": True, "icr_now": 6.8, "evidence_icr": 5.4, "evidence_lower_90": 5.1, "evidence_upper_90": 5.7, "evidence_breakfasts": 40})
    assert rs.icr_status(row)[0] == "needs more insulin"
    row["icr_now"] = 5.5
    assert rs.icr_status(row)[0] == "fits"
    row["had_breakfasts"] = False
    assert rs.icr_status(row)[0] == "not enough data"


def test_basal_status_reads_the_controller_signal() -> None:
    base = {"share_over_x": 0.7, "pooled_excess": 0.4, "excess_lo": 0.1, "excess_hi": 0.7, "clean_hours": 50.0, "days": 13}
    assert rs.basal_status(pd.Series({**base, "signal": "basal likely too low"}))[0] == "needs more insulin"
    assert rs.basal_status(pd.Series({**base, "signal": "no consistent pattern"}))[0] == "fits"
    assert rs.basal_status(pd.Series({"share_over_x": np.nan, "signal": "not enough clean time"}))[0] == "not enough data"


def test_page_renders_every_section_at_phone_width() -> None:
    days = synthetic_days(40)
    days["time_in_range_pct"] = 100 * days["in_range_readings"] / days["readings"]
    days["time_below_70_pct"] = 100 * days["below_70_readings"] / days["readings"]
    days["mean_glucose_mgdl"] = 150.0
    for column, value in (("sleep_hours", 7.0), ("sleep_device", "Oura"), ("resting_hr", 60.0), ("resting_hr_z", 0.2), ("hrv_ms", 50.0), ("hrv_z", -0.3), ("recovery_score", 45.0),
                          ("insulin_units", 40.0), ("bolus_units", 20.0), ("basal_units", 20.0), ("carb_grams", 120.0), ("insulin_vs_usual_pct", 5.0), ("carbs_vs_usual_pct", -3.0)):
        days[column] = value
    effects = pd.DataFrame({"driver": list(ro.DRIVER_COLUMNS), "days_with": 30, "days_without": 10, "estimated": True, "estimate_pp": 1.0, "lower_pp": -1.0, "upper_pp": 3.0, "prob_above_zero": 0.8})
    settings = pd.DataFrame([{"part": "00:00-06:00", "setting": "basal", "now": 0.7, "status": "needs more insulin", "suggestion": "consider 0.7 to 0.77 U/h (10% more insulin)", "evidence": "x",
                              "tir_14d": 70.0, "tir_90d": 72.0, "below_70_90d": 1.0, "lows_within_limits": True}])
    as_of = days.index.max()
    cgm = pd.DataFrame({"timestamp": pd.date_range(as_of - pd.Timedelta(days=6), as_of + pd.Timedelta(hours=23, minutes=55), freq="5min")})
    cgm["glucose"] = 140 + 40 * np.sin(np.arange(len(cgm)) / 30)
    readout = Readout(as_of=as_of, days=days, usual_week=(60.0, 70.0, 78.0), effects={"time_in_range": effects, "time_below_70": effects}, usable={"time_in_range": True, "time_below_70": True},
                      model_days=40, settings=settings, changes=pd.DataFrame(columns=["changed"]), cgm=cgm,
                      sleep=pd.DataFrame({"sleep_start": [as_of - pd.Timedelta(hours=1)], "sleep_end": [as_of + pd.Timedelta(hours=7)]}),
                      workouts=pd.DataFrame({"start_timestamp": [as_of + pd.Timedelta(hours=17)], "end_timestamp": [as_of + pd.Timedelta(hours=18)]}),
                      carbs=pd.DataFrame({"timestamp": [as_of + pd.Timedelta(hours=8)], "carb_grams": [45.0]}), site_fills=pd.Series(pd.to_datetime([as_of + pd.Timedelta(hours=20)])),
                      run_id="test", built="now")
    html = render(readout)
    assert "width=device-width" in html and "<script" not in html
    for heading in ("This week", "What has gone with your glucose", "Settings check", "Recent setting changes", "The last seven days"):
        assert f"<h2>{heading}</h2>" in html
    assert "nan" not in html.lower().replace("nanos", "")
    assert "viewBox='0 0 340" in html, "charts are drawn at phone width"


def test_drivers_model_stops_with_a_clear_message_when_no_driver_is_estimable() -> None:
    days = synthetic_days(60)
    for column in ro.DRIVER_COLUMNS:
        days[column] = False
    with pytest.raises(ValueError, match="needs a longer record"):
        rm.fit_outcome(days, "time_in_range", draws=50, tune=50, chains=1)


def test_held_suggestion_has_its_own_status() -> None:
    status, text = rs.suggestion_for("basal", 0.7, "needs more insulin", 0.43, lows_ok=False)
    assert status == rs.HELD and status in rs.STATUSES
    assert "no added insulin" in text


def test_usual_comparison_is_missing_not_infinite_when_the_baseline_is_zero() -> None:
    dates = pd.date_range("2025-01-01", periods=21, freq="D")
    assert np.isnan(ro.versus_usual(pd.Series([0.0] * 20 + [90.0], index=dates)).iloc[-1])
    assert ro.versus_usual(pd.Series([40.0] * 20 + [44.0], index=dates)).iloc[-1] == pytest.approx(10.0)


def test_days_without_a_worn_wearable_have_unknown_workout_status() -> None:
    dates = pd.date_range("2025-01-01", periods=4, freq="D")
    workouts = pd.DataFrame({"start_timestamp": pd.to_datetime(["2025-01-01 07:00"]), "active_energy_value": [300.0]})
    measurements = pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-02 10:00", "2025-01-04 10:00"]), "metric": ["heart_rate", "heart_rate"]})
    energy = ro.workout_energy_by_day(workouts, measurements, dates)
    assert energy.iloc[0] == 300.0
    assert energy.iloc[1] == 0.0, "worn, no workout: a rest day"
    assert np.isnan(energy.iloc[2]), "nothing recording: unknown, not a rest day"


def test_time_in_range_ring_changes_color_below_the_target() -> None:
    from bayesian_t1dm.figures import READOUT_STYLE

    day = pd.Series({"time_in_range_pct": 38.0, "time_below_70_pct": 0.0, "insulin_vs_usual_pct": 2.0})
    assert READOUT_STYLE["below_target"].color in rings(day, None)
    assert READOUT_STYLE["range"].color in rings(pd.Series({**day.to_dict(), "time_in_range_pct": 80.0}), None)
