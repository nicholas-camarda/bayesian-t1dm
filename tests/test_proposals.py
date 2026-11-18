from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bayesian_t1dm import proposals
from bayesian_t1dm.proposals import propose_breakfast_icr, proposed_step

AS_OF = pd.Timestamp("2026-06-01")
SEGMENT = 300.0


def _row(analysis="primary", *, programmed=7.0, estimate=6.0, lower=5.5, upper=6.5, n=20, current=True, start="2026-05-01", end="2026-05-30", segment=SEGMENT):
    return {
        "analysis": analysis, "pump_serial": "1", "profile_id": 0.0, "pump_segment_start_minutes": segment, "programmed_icr": programmed,
        "period_start": pd.Timestamp(start), "period_end": pd.Timestamp(end), "is_current_setting": current, "n_meals": n,
        "logged_meal_windows": n, "effective_icr": estimate, "effective_icr_lower_90": lower, "effective_icr_upper_90": upper,
    }


def _cgm(low_fraction: float = 0.0, days: int = 30) -> pd.DataFrame:
    """Flat 120 mg/dL with a share of readings at 60 in the 05:00-13:00 hours (segment 05:00-09:00 plus 4 hours)."""
    index = pd.date_range(AS_OF - pd.Timedelta(days=days), AS_OF, freq="5min", inclusive="left")
    glucose = np.full(len(index), 120.0)
    hour = index.hour
    in_part = (hour >= 5) & (hour < 13)
    rng = np.random.default_rng(0)
    glucose[in_part & (rng.random(len(index)) < low_fraction)] = 60.0
    return pd.DataFrame({"timestamp": index, "glucose": glucose})


ENDS = {SEGMENT: 540.0}


def test_step_is_the_smaller_of_ten_percent_and_the_distance_toward_the_estimate():
    assert proposed_step(7.5, 6.5) == 6.8  # 10% of 7.5 is 0.75, less than the distance 1.0
    assert proposed_step(7.0, 6.8) == 6.8  # the distance 0.2 is smaller than 0.7
    assert proposed_step(6.0, 8.0) == 6.6  # toward a larger estimate
    assert proposed_step(7.0, 7.0) == 7.0


def test_a_proposal_when_every_check_agrees_and_the_lows_are_within_limits():
    evidence = pd.DataFrame([_row(), _row("clean reference breakfasts only", estimate=6.1, n=6), _row("without correction boluses", estimate=6.2, n=12)])
    result = propose_breakfast_icr(evidence, _cgm(), ENDS, as_of=AS_OF)
    assert result.status == "proposed" and result.direction == "adds insulin"
    assert result.current_value == 7.0 and result.proposed_value == 6.3
    assert result.affected_hours == "05:00-13:00"
    assert all(value.startswith("agrees") for value in result.checks.values())
    assert "bootstrap" in result.evaluation


def test_held_when_the_current_setting_is_inside_the_interval():
    result = propose_breakfast_icr(pd.DataFrame([_row(programmed=6.2)]), _cgm(), ENDS, as_of=AS_OF)
    assert result.status == "held" and "inside the evidence interval" in result.reason


def test_held_with_too_few_breakfasts_at_the_current_setting():
    result = propose_breakfast_icr(pd.DataFrame([_row(n=1, estimate=float("nan"), lower=float("nan"), upper=float("nan"))]), _cgm(), ENDS, as_of=AS_OF)
    assert result.status == "held" and "only 1 breakfasts" in result.reason


def test_a_check_pointing_the_other_way_holds_the_proposal():
    evidence = pd.DataFrame([_row(), _row("clean reference breakfasts only", estimate=7.6, n=8)])
    result = propose_breakfast_icr(evidence, _cgm(), ENDS, as_of=AS_OF)
    assert result.status == "held" and "clean reference breakfasts only" in result.reason


def test_a_check_with_too_few_breakfasts_is_reported_but_does_not_block():
    evidence = pd.DataFrame([_row(), _row("clean reference breakfasts only", estimate=7.6, n=3)])
    result = propose_breakfast_icr(evidence, _cgm(), ENDS, as_of=AS_OF)
    assert result.status == "proposed" and result.checks["clean reference breakfasts only"].startswith("not assessable")


def test_lows_block_adding_insulin_but_not_removing_it():
    adds = propose_breakfast_icr(pd.DataFrame([_row()]), _cgm(low_fraction=0.10), ENDS, as_of=AS_OF)
    assert adds.status == "held" and "adding insulin is not allowed" in adds.reason
    removes = propose_breakfast_icr(pd.DataFrame([_row(programmed=6.0, estimate=7.0, lower=6.6, upper=7.4)]), _cgm(low_fraction=0.10), ENDS, as_of=AS_OF)
    assert removes.status == "proposed" and removes.direction == "removes insulin" and removes.proposed_value == 6.6


def test_the_segment_with_most_recent_breakfasts_is_the_one_proposed_for():
    evidence = pd.DataFrame([_row(n=5, segment=540.0), _row(n=30, segment=SEGMENT)])
    result = propose_breakfast_icr(evidence, _cgm(), {SEGMENT: 540.0, 540.0: 900.0}, as_of=AS_OF)
    assert result.segment_start_minutes == SEGMENT


def test_constants_match_the_written_plan():
    assert proposals.MAX_STEP_FRACTION == pytest.approx(0.10) and proposals.HOLD_DAYS == 14 and proposals.AFTER_MEAL_HOURS == 4


def test_a_change_already_entered_is_reported_not_proposed_again():
    evidence = pd.DataFrame([_row()])
    result = propose_breakfast_icr(evidence, _cgm(), ENDS, as_of=AS_OF, current_icr={SEGMENT: 6.5})
    assert result.status == "held" and "pump now holds 6.5" in result.reason and "supports the change" in result.reason
    assert result.current_value == 6.5
    same = propose_breakfast_icr(evidence, _cgm(), ENDS, as_of=AS_OF, current_icr={SEGMENT: 7.0})
    assert same.status == "proposed"


def test_segment_summary_keeps_segments_separate_and_leads_with_glucose():
    from bayesian_t1dm.outcomes import day_part_table, day_parts_from_starts
    from bayesian_t1dm.proposals import segment_markdown, segment_summary

    parts = day_parts_from_starts([0, 300, 540])
    cgm = _cgm(days=30)
    o90 = day_part_table(cgm, parts, as_of=AS_OF, days=30)
    o14 = day_part_table(cgm, parts, as_of=AS_OF, days=14)
    evidence = pd.DataFrame([_row(programmed=7.0, estimate=6.5, lower=6.0, upper=7.0, n=30)])
    table = segment_summary(
        parts, {0.0: {"icr": 7.0, "cf": 60.0, "basal": 0.7}, 300.0: {"icr": 6.5, "cf": 65.0, "basal": 0.71}, 540.0: {"icr": 6.5, "cf": 65.0, "basal": 0.71}},
        o90, o14, evidence, pump_serial="1", profile_id=0.0,
    )
    assert list(table["hours"]) == ["00:00-05:00", "05:00-09:00", "09:00-00:00"]
    five_to_nine = table[table["hours"].eq("05:00-09:00")].iloc[0]
    assert five_to_nine["evidence_breakfasts"] == 30 and five_to_nine["breakfasts_at_current_icr"] == 0
    other = table[table["hours"].eq("09:00-00:00")].iloc[0]
    assert other["evidence_breakfasts"] == 0 and not other["had_breakfasts"]
    page = segment_markdown(table)
    assert "05:00-09:00" in page and "30 breakfasts at 7.0 behaved like 6.5" in page and "no breakfasts in this segment" in page
