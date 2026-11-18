from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bayesian_t1dm import outcomes


def _day(date: str, values: list[float]) -> pd.DataFrame:
    times = pd.date_range(date, periods=len(values), freq="5min")
    return pd.DataFrame({"timestamp": times, "glucose": values})


def test_consensus_percentages_on_a_known_day():
    # 288 readings: 100 in range, 20 below 70 (8 of them below 54), 120 above 180 (48 of them above 250).
    values = [120.0] * 100 + [60.0] * 12 + [50.0] * 8 + [200.0] * 72 + [300.0] * 48 + [100.0] * 48
    assert len(values) == 288
    metrics = outcomes.daily_metrics(_day("2026-01-01", values)).iloc[0]
    assert metrics["time_in_range_pct"] == pytest.approx(100 * 148 / 288)
    assert metrics["time_below_70_pct"] == pytest.approx(100 * 20 / 288)
    assert metrics["time_below_54_pct"] == pytest.approx(100 * 8 / 288)
    assert metrics["time_above_180_pct"] == pytest.approx(100 * 120 / 288)
    assert metrics["time_above_250_pct"] == pytest.approx(100 * 48 / 288)
    mean = float(np.mean(values))
    assert metrics["mean_glucose_mgdl"] == pytest.approx(mean)
    assert metrics["gmi_pct"] == pytest.approx(3.31 + 0.02392 * mean)
    assert metrics["cv_pct"] == pytest.approx(100 * np.std(values) / mean)
    assert bool(metrics["counted"])


def test_a_day_under_seventy_percent_coverage_does_not_count():
    full = _day("2026-01-01", [120.0] * 288)
    sparse = _day("2026-01-02", [300.0] * 150)  # 52% of 288
    cgm = pd.concat([full, sparse], ignore_index=True)
    daily = outcomes.daily_metrics(cgm)
    assert daily["counted"].tolist() == [True, False]
    pooled = outcomes.pooled_metrics(cgm, start=pd.Timestamp("2026-01-01"), end=pd.Timestamp("2026-01-03"))
    assert pooled["days"] == 1 and pooled["time_in_range_pct"] == pytest.approx(100.0)


def test_duplicate_readings_in_one_slot_are_counted_once():
    cgm = pd.concat([_day("2026-01-01", [120.0] * 288), _day("2026-01-01", [400.0] * 288)], ignore_index=True)
    assert outcomes.daily_metrics(cgm).iloc[0]["readings"] == 288


def test_day_parts_follow_the_segment_starts_and_wrap_midnight():
    parts = outcomes.day_parts_from_starts([0, 6 * 60, 12 * 60, 22 * 60])
    assert [p.name for p in parts] == ["00:00-06:00", "06:00-12:00", "12:00-22:00", "22:00-00:00"]
    assert [p.hours for p in parts] == [6, 6, 10, 2]
    wrap = outcomes.day_parts_from_starts([6 * 60, 22 * 60])
    assert wrap[1].name == "22:00-06:00" and wrap[1].hours == 8
    assert wrap[1].contains(np.array([23 * 60, 3 * 60, 12 * 60])).tolist() == [True, True, False]


def test_day_part_metrics_use_only_that_part_and_its_own_coverage():
    values = [100.0] * 144 + [300.0] * 144  # midnight-noon in range, noon-midnight high
    cgm = _day("2026-01-01", values)
    morning = outcomes.DayPart("00:00-12:00", 0, 12 * 60)
    daily = outcomes.daily_metrics(cgm, part=morning).iloc[0]
    assert daily["time_in_range_pct"] == pytest.approx(100.0) and daily["coverage"] == pytest.approx(1.0)


def test_low_limits():
    assert outcomes.passes_low_limits({"time_below_70_pct": 3.9, "time_below_54_pct": 0.9})
    assert not outcomes.passes_low_limits({"time_below_70_pct": 4.0, "time_below_54_pct": 0.5})
    assert not outcomes.passes_low_limits({"time_below_70_pct": 1.0, "time_below_54_pct": float("nan")})


def test_baseline_windows_end_before_as_of():
    cgm = pd.concat([_day("2026-01-01", [100.0] * 288), _day("2026-01-02", [300.0] * 288)], ignore_index=True)
    table = outcomes.baseline_table(cgm, as_of=pd.Timestamp("2026-01-02"), windows_days=(14,))
    assert table.iloc[0]["days"] == 1 and table.iloc[0]["time_in_range_pct"] == pytest.approx(100.0)


def test_page_names_where_highs_and_lows_concentrate():
    from bayesian_t1dm.outcomes_report import outcomes_page

    parts = outcomes.day_parts_from_starts([0, 12 * 60])
    values = [100.0] * 144 + [300.0] * 144
    cgm = _day("2026-01-01", values)
    as_of = pd.Timestamp("2026-01-02")
    page = outcomes_page(
        outcomes.baseline_table(cgm, as_of=as_of, windows_days=(14,)),
        outcomes.day_part_table(cgm, parts, as_of=as_of, days=14),
        as_of=as_of,
        source="test pump",
    )
    assert "Highs concentrate most in 12:00-00:00 (100.0% above 180)" in page
    assert "within limits" in page and "test pump" in page
