from __future__ import annotations

import numpy as np
import pandas as pd

from bayesian_t1dm.changes import detect_setting_changes, evaluate_change
from bayesian_t1dm.outcomes import DayPart


def _event(seq, when, kind, **fields):
    base = {"pump_serial": "1", "timestamp": pd.Timestamp(when), "sequence": seq, "event": kind, "profile_id": 0, "status": 15, "source_profile_id": np.nan,
            "profile_name": None, "modification": 7, "segment_index": 0, "start_minutes": 0, "basal_units_per_hour": 0.7, "isf_mgdl_per_unit": 60,
            "target_bg_mgdl": 110, "carb_ratio_g_per_unit": 7.0}
    base.update(fields)
    return base


def test_only_a_changed_value_is_a_change_not_a_relogged_segment():
    events = pd.DataFrame(
        [
            _event(1, "2026-11-01 08:00", 69, status=0),
            _event(2, "2026-11-01 08:00", 68, segment_index=0, modification=0),
            _event(3, "2026-11-01 08:00", 68, segment_index=1, modification=0, start_minutes=300),
            _event(4, "2026-11-17 09:29", 68, segment_index=1, modification=4, start_minutes=300, carb_ratio_g_per_unit=6.5),
            _event(5, "2026-11-17 09:29", 68, segment_index=0, modification=7),  # re-logged unchanged
            _event(6, "2026-11-17 22:28", 68, segment_index=1, modification=5, start_minutes=300, carb_ratio_g_per_unit=6.5, isf_mgdl_per_unit=65),
        ]
    )
    changes = detect_setting_changes(events)
    assert changes[["setting", "old_value", "new_value", "segment_start_minutes"]].values.tolist() == [
        ["carb_ratio_g_per_unit", 7.0, 6.5, 300.0],
        ["isf_mgdl_per_unit", 60.0, 65.0, 300.0],
    ]
    assert changes["timestamp"].tolist() == [pd.Timestamp("2026-11-17 09:29"), pd.Timestamp("2026-11-17 22:28")]


def _cgm(start, days, tir_by_day):
    rows = []
    for day in range(days):
        index = pd.date_range(pd.Timestamp(start) + pd.Timedelta(days=day), periods=288, freq="5min")
        glucose = np.full(288, 120.0)
        in_hours = (index.hour >= 5) & (index.hour < 13)
        high = np.arange(288) % 100 < (100 - tir_by_day(day))
        glucose[in_hours & high] = 250.0
        rows.append(pd.DataFrame({"timestamp": index, "glucose": glucose}))
    return pd.concat(rows, ignore_index=True)


PART = DayPart("05:00-13:00", 300, 780)


def test_evaluation_waits_for_the_full_after_window():
    cgm = _cgm("2026-10-01", 40, lambda d: 60)
    result = evaluate_change(cgm, pd.Timestamp("2026-11-10 09:30"), PART, now=pd.Timestamp("2026-11-20"))
    assert result["status"] == "pending" and result["after_days_observed"] == 9


def test_evaluation_reports_a_rise_with_an_interval_and_the_low_limits():
    change = pd.Timestamp("2026-10-15 09:30")
    cgm = _cgm("2026-09-20", 60, lambda d: 60 if d < 25 else 80)  # day 25 = Oct 15
    result = evaluate_change(cgm, change, PART, now=pd.Timestamp("2026-11-20"))
    assert result["status"] == "evaluated" and result["days_before"] == 14 and result["days_after"] == 14
    assert 15 < result["tir_difference"] < 25 and result["tir_difference_lower_90"] > 0
    assert result["verdict"] == "time in range rose" and result["low_limits_after"] == "within limits"


def test_excluded_context_days_and_thin_coverage_are_left_out():
    change = pd.Timestamp("2026-10-15 09:30")
    cgm = _cgm("2026-09-20", 60, lambda d: 60)
    excluded = pd.date_range("2026-10-16", periods=3)
    result = evaluate_change(cgm, change, PART, now=pd.Timestamp("2026-11-20"), excluded_days=excluded)
    assert result["days_after"] == 11
    thin = cgm[cgm["timestamp"] < "2026-10-16"]  # nothing after the change
    assert evaluate_change(thin, change, PART, now=pd.Timestamp("2026-11-20"))["status"] == "insufficient"
