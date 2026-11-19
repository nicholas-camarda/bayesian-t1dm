from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bayesian_t1dm.analysis_data import AnalysisData  # noqa: F401  (the table reads the fields of AnalysisData)
from bayesian_t1dm.hour_day import GLUCOSE_COLUMNS, _dst_days, build_hour_day_table, hour_day_flow, rollup_segment_days
from bayesian_t1dm.outcomes import daily_metrics, day_parts_from_starts
from bayesian_t1dm.pump_profiles import PROFILE_EVENT_COLUMNS, SEGMENT_EVENT, PROFILE_EVENT, ProfileHistory

PUMP = "10001"
DAY1 = pd.Timestamp("2025-09-01")
EDIT = pd.Timestamp("2025-09-01 09:07")


def _event(timestamp: pd.Timestamp, sequence: int, **fields) -> dict:
    row = {name: np.nan for name in PROFILE_EVENT_COLUMNS}
    row.update(pump_serial=PUMP, timestamp=timestamp.to_pydatetime(), sequence=sequence, profile_id=0, profile_name="Main", **fields)
    return row


def _history() -> ProfileHistory:
    segment = dict(event=SEGMENT_EVENT, status=15, segment_index=0, start_minutes=0, basal_units_per_hour=0.8, isf_mgdl_per_unit=65, target_bg_mgdl=110)
    events = pd.DataFrame([
        _event(DAY1, 1, event=PROFILE_EVENT, status=0, source_profile_id=0),
        _event(DAY1, 2, modification=0, carb_ratio_g_per_unit=7.0, **segment),
        _event(DAY1, 3, event=PROFILE_EVENT, status=3, source_profile_id=0),
        _event(EDIT, 4, modification=3, carb_ratio_g_per_unit=10.0, **segment),
    ])
    return ProfileHistory(events, {PUMP: (dt.date(2025, 9, 1), dt.date(2025, 9, 2))}, pd.DataFrame(columns=["pump_serial", "sequence", "timestamp", "rate"]))


def _analysis() -> SimpleNamespace:
    slots = pd.date_range(DAY1, DAY1 + pd.Timedelta(days=2) - pd.Timedelta(minutes=5), freq="5min")
    glucose = np.full(len(slots), 100.0)
    glucose[(slots >= "2025-09-01 03:00") & (slots < "2025-09-01 03:20")] = 50.0      # 4 readings below 54
    glucose[(slots >= "2025-09-01 03:20") & (slots < "2025-09-01 03:30")] = 65.0      # 2 more below 70
    glucose[(slots >= "2025-09-02 10:00") & (slots < "2025-09-02 10:10")] = 200.0     # 2 above 180
    cgm = pd.DataFrame({"timestamp": slots, "glucose": glucose, "pump_serial": PUMP})
    duplicate = pd.DataFrame({"timestamp": [pd.Timestamp("2025-09-01 08:00:30")], "glucose": [300.0], "pump_serial": PUMP})
    cgm = pd.concat([cgm, duplicate], ignore_index=True)  # a second report of the 08:00 reading: dropped
    grid = pd.DataFrame({"timestamp": slots, "carb_grams": 0.0, "bolus_units": 0.0})
    grid.loc[grid.timestamp.eq("2025-09-01 08:05"), ["carb_grams", "bolus_units"]] = [30.0, 2.0]
    grid.loc[grid.timestamp.eq("2025-09-01 10:00"), "carb_grams"] = 20.0
    modes = pd.DataFrame({"pump_serial": [PUMP, PUMP], "timestamp": pd.to_datetime(["2025-09-01 00:00", "2025-09-01 06:00"]), "mode": [1, 0]})
    sleep = pd.DataFrame({"sleep_start": [pd.Timestamp("2025-09-01 00:00")], "sleep_end": [pd.Timestamp("2025-09-01 06:58")],
                          "covered_start_date": [DAY1], "covered_end_date": [DAY1 + pd.Timedelta(days=1)]})
    return SimpleNamespace(
        cgm=cgm, history=_history(), grid=grid,
        exercise=pd.DataFrame({"start": [pd.Timestamp("2025-09-01 14:15")], "end": [pd.Timestamp("2025-09-01 15:28")]}),
        mode_events=modes, event_span={PUMP: (dt.date(2025, 9, 1), dt.date(2025, 9, 2))},
        site_events=pd.DataFrame({"kind": ["cannula filled"], "timestamp": [pd.Timestamp("2025-08-31 12:00")]}),
        excluded_days=[pd.Timestamp("2025-09-02")],
        merged=SimpleNamespace(data=SimpleNamespace(sleep=sleep)),
    )


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return build_hour_day_table(_analysis()).set_index("hour_start")


def test_one_row_per_clock_hour_with_exact_counts(table):
    assert len(table) == 48
    assert table["n_valid"].eq(12).all()                      # the duplicate 08:00 report is dropped, not counted
    hour3 = table.loc["2025-09-01 03:00"]
    assert (hour3.n_below_54, hour3.n_below_70, hour3.n_in_range) == (4, 6, 6)
    hour10 = table.loc["2025-09-02 10:00"]
    assert (hour10.n_above_180, hour10.n_in_range) == (2, 10)
    assert table.loc["2025-09-01 08:00", "glucose_sum"] == 1200.0
    parts = table[["n_in_range", "n_below_70", "n_above_180"]].sum(axis=1)
    assert parts.eq(table["n_valid"]).all()
    assert (table["n_below_54"] <= table["n_below_70"]).all()


def test_an_edit_inside_an_hour_mixes_the_settings_of_that_hour(table):
    assert table.loc["2025-09-01 08:00", "icr"] == 7.0 and not table.loc["2025-09-01 08:00", "settings_mixed"]
    hour9 = table.loc["2025-09-01 09:00"]
    assert hour9.icr == pytest.approx((2 * 7.0 + 10 * 10.0) / 12)  # 09:00 and 09:05 hold the old value, the other ten the new
    assert hour9.settings_mixed and hour9.settings_known_slots == 12
    assert table.loc["2025-09-01 10:00", "icr"] == 10.0
    assert table.loc["2025-09-01 10:00", ["cf", "basal", "target"]].tolist() == pytest.approx([65.0, 0.8, 110.0])
    assert table["segment_start"].eq(0).all()


def test_changed_settings_are_marked_on_the_day_of_the_edit_only():
    analysis = _analysis()
    day0 = DAY1 - pd.Timedelta(days=1)
    segment = dict(event=SEGMENT_EVENT, status=15, segment_index=0, start_minutes=0, basal_units_per_hour=0.8, isf_mgdl_per_unit=65, target_bg_mgdl=110)
    events = pd.DataFrame([
        _event(day0, 1, event=PROFILE_EVENT, status=0, source_profile_id=0),
        _event(day0, 2, modification=0, carb_ratio_g_per_unit=7.0, **segment),
        _event(day0, 3, event=PROFILE_EVENT, status=3, source_profile_id=0),
        _event(EDIT, 4, modification=3, carb_ratio_g_per_unit=10.0, **segment),
    ])
    analysis.history = ProfileHistory(events, {PUMP: (day0.date(), dt.date(2025, 9, 2))}, pd.DataFrame(columns=["pump_serial", "sequence", "timestamp", "rate"]))
    slots = pd.date_range(day0, DAY1 + pd.Timedelta(days=2) - pd.Timedelta(minutes=5), freq="5min")
    analysis.cgm = pd.DataFrame({"timestamp": slots, "glucose": 100.0, "pump_serial": PUMP})
    table = build_hour_day_table(analysis).set_index("hour_start")

    assert table.loc["2025-09-01", "settings_changed"].tolist() == [False] * 9 + [True] * 15      # from the 09:05 moment on
    assert not table.loc["2025-08-31", "settings_changed"].any()
    # The hours that had already passed on the day of the edit first run the new value the next day; after that nothing changes.
    assert table.loc["2025-09-02", "settings_changed"].tolist() == [True] * 10 + [False] * 14   # through the 09:00-09:05 moments of the edited hour
    assert table.loc["2025-09-01 09:00", "d_icr"] == pytest.approx(3.0 * 10 / 12)   # ten of twelve moments went from 7 to 10
    assert table.loc["2025-09-01 12:00", "d_icr"] == pytest.approx(3.0) and table.loc["2025-09-01 12:00", ["d_cf", "d_basal", "d_target"]].eq(0).all()
    assert table["settings_changed"].sum() == 25


def test_settings_agree_with_the_profile_in_force_at_random_moments():
    analysis = _analysis()
    rng = np.random.default_rng(5)
    moments = [DAY1 + pd.Timedelta(minutes=int(m)) for m in rng.integers(0, 2 * 24 * 60, 50)]
    values = analysis.history.values_at(pd.Series([PUMP] * 50), pd.Series(moments), "carb_ratio_g_per_unit")
    for moment, value in zip(moments, values):
        segment = analysis.history.segment_in_force(PUMP, 0, moment.to_pydatetime())
        assert value == segment["carb_ratio_g_per_unit"]


def test_context_columns(table):
    assert table.loc["2025-09-01 10:00", "carbs_g"] == 20.0 and table.loc["2025-09-01 08:00", "bolus_u"] == 2.0
    # this hour and the three before it
    assert table.loc["2025-09-01 11:00", "carbs_prev3h_g"] == 50.0 and table.loc["2025-09-01 12:00", "carbs_prev3h_g"] == 20.0
    assert table.loc["2025-09-01 14:00", "carbs_prev3h_g"] == 0.0
    assert table.loc["2025-09-01 14:00", "exercise_share"] == pytest.approx(9 / 12) and table.loc["2025-09-01 15:00", "exercise_share"] == pytest.approx(6 / 12)
    assert table.loc["2025-09-01 05:00", "sleep_mode_share"] == 1.0 and table.loc["2025-09-01 06:00", "sleep_mode_share"] == 0.0
    assert table.loc["2025-09-01 06:00", "apple_asleep_share"] == 1.0 and table.loc["2025-09-01 07:00", "apple_asleep_share"] == 0.0
    assert table.loc["2025-09-01 12:00", "site_age_days"] == pytest.approx(1.0)
    assert table.loc["2025-09-02 10:00", "excluded_day"] and not table.loc["2025-09-01 10:00", "excluded_day"]


def test_flow_counts_hours_and_exclusions(table):
    flow = hour_day_flow(table.reset_index())
    assert (flow.hours, flow.hours_with_readings, flow.hours_with_all_settings, flow.hours_with_mixed_settings, flow.hours_on_excluded_days) == (48, 48, 48, 1, 24)


def test_dst_days_are_the_clock_change_days():
    flagged = _dst_days(pd.DatetimeIndex(["2025-03-08", "2025-03-09", "2025-03-10", "2025-11-02"]))
    assert flagged.tolist() == [False, True, False, True]


def test_spring_forward_hour_has_no_readings():
    analysis = _analysis()
    slots = pd.date_range("2025-03-09 00:00", "2025-03-09 23:55", freq="5min")
    slots = slots[(slots < "2025-03-09 02:00") | (slots >= "2025-03-09 03:00")]  # the 02:00 hour does not exist on the wall clock
    analysis.cgm = pd.DataFrame({"timestamp": slots, "glucose": 100.0, "pump_serial": PUMP})
    result = build_hour_day_table(analysis).set_index("hour_start")
    assert result.loc["2025-03-09 02:00", "n_valid"] == 0 and result.loc["2025-03-09 03:00", "n_valid"] == 12
    assert result.loc["2025-03-09 02:00", "dst_day"]


def test_segment_day_rollup_follows_the_part_and_reconciles_with_daily_metrics(table):
    parts = day_parts_from_starts([21 * 60, 5 * 60])
    rolled, split_hours = rollup_segment_days(table.reset_index(), parts)
    assert split_hours == 0
    night = next(part for part in parts if part.start_minutes == 21 * 60)
    # 01:00 on the 2nd belongs to the night that began on the 1st; the 2nd's own night has only its first three hours of data
    nights = rolled[rolled["part"].eq(night.name)].set_index("analysis_date")
    assert nights.loc[pd.Timestamp("2025-09-01"), "n_valid"] == 12 * 8   # 21:00-05:00, 8 hours
    assert nights.loc[pd.Timestamp("2025-09-02"), "n_valid"] == 12 * 3   # 21:00-24:00 only
    analysis = _analysis()
    for part in parts:
        daily = daily_metrics(analysis.cgm, part=part)
        theirs = rolled[rolled["part"].eq(part.name)].set_index("analysis_date")
        for date, row in daily.iterrows():
            assert theirs.loc[date, "n_valid"] == row["readings"]
            assert theirs.loc[date, "n_below_70"] == pytest.approx(row["time_below_70_pct"] / 100 * row["readings"])
    assert rolled["n_valid"].sum() == table["n_valid"].sum()
    assert set(GLUCOSE_COLUMNS) <= set(rolled.columns)


def test_rollup_counts_hours_that_straddle_a_boundary(table):
    parts = day_parts_from_starts([9 * 60 + 30, 21 * 60])
    rolled, split_hours = rollup_segment_days(table.reset_index(), parts)
    assert split_hours == 2   # the 09:00 hour on each day holds 30 minutes of each part
    assert rolled["n_valid"].sum() == table["n_valid"].sum()


def test_an_edit_to_a_profile_that_is_not_running_is_no_change_but_a_profile_switch_is():
    analysis = _analysis()
    day0 = DAY1
    segment = dict(event=SEGMENT_EVENT, status=15, segment_index=0, start_minutes=0, basal_units_per_hour=0.8, isf_mgdl_per_unit=65, target_bg_mgdl=110)

    def profile_event(timestamp, sequence, profile, **fields):
        return {**_event(timestamp, sequence, **fields), "profile_id": profile}

    events = pd.DataFrame([
        profile_event(day0, 1, 0, event=PROFILE_EVENT, status=0, source_profile_id=0),
        profile_event(day0, 2, 0, modification=0, carb_ratio_g_per_unit=7.0, **segment),
        profile_event(day0, 3, 0, event=PROFILE_EVENT, status=3, source_profile_id=0),
        profile_event(day0, 4, 1, event=PROFILE_EVENT, status=0, source_profile_id=1),
        profile_event(day0, 5, 1, modification=0, carb_ratio_g_per_unit=6.0, **segment),
        profile_event(day0 + pd.Timedelta(days=1, hours=12), 6, 1, modification=3, carb_ratio_g_per_unit=5.0, **segment),   # profile 1 is not running
        profile_event(day0 + pd.Timedelta(days=2, hours=8), 7, 1, event=PROFILE_EVENT, status=3, source_profile_id=1),      # then it is switched on
    ])
    analysis.history = ProfileHistory(events, {PUMP: (day0.date(), dt.date(2025, 9, 4))}, pd.DataFrame(columns=["pump_serial", "sequence", "timestamp", "rate"]))
    slots = pd.date_range(day0, day0 + pd.Timedelta(days=4) - pd.Timedelta(minutes=5), freq="5min")
    analysis.cgm = pd.DataFrame({"timestamp": slots, "glucose": 100.0, "pump_serial": PUMP})
    table = build_hour_day_table(analysis).set_index("hour_start")

    assert not table.loc["2025-09-02", "settings_changed"].any()           # the edit to the idle profile changes nothing in force
    assert table.loc["2025-09-03", "settings_changed"].tolist() == [False] * 8 + [True] * 16    # the switch at 08:00 puts I:C 5 in force
    assert table.loc["2025-09-03 09:00", "d_icr"] == pytest.approx(-2.0)
    assert table.loc["2025-09-04", "settings_changed"].tolist() == [True] * 8 + [False] * 16    # hours already passed first run it the next day
