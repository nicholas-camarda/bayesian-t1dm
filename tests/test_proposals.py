from __future__ import annotations

import numpy as np
import pandas as pd


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



def test_segment_summary_keeps_segments_separate_and_leads_with_glucose():
    from bayesian_t1dm.outcomes import day_part_table, day_parts_from_starts
    from bayesian_t1dm.proposals import segment_summary

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
