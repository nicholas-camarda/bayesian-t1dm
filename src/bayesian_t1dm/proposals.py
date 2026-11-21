"""Breakfast I:C proposals: one small change the evidence supports, or the reason there is none.

A proposal is made for the current setting of the pump segment that covers most recent breakfasts. It needs:

1. at least ``MIN_LOGGED_MEALS_FOR_ICR`` breakfasts at the current setting;
2. the current I:C outside the primary 90% interval for the effective I:C;
3. every check that can be assessed to point the same way as the primary: the re-pool without the looser tiers, the
   clean breakfasts alone (assessable once 5 have accumulated), the breakfasts without a correction bolus, and the
   two logged-food (Bevel) sensitivities;
4. for a change that adds insulin (a smaller g/U), time below 70 under 4% and below 54 under 1% over the affected
   hours plus the 4 hours after, in the last 90 days. A change that removes insulin needs no such condition.

The step is the smaller of 10% of the current value and the distance to the estimate, rounded to 0.1 g/U, toward
the estimate. The proposal is a statement about one person's data; the person enters it on the pump, and its effect
is judged afterwards by the comparison written into the proposal (``EVALUATION``).
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from .latent_meal_trajectory import MIN_LOGGED_MEALS_FOR_ICR
from .outcomes import DayPart, LOW_LIMIT_PERCENT, VERY_LOW_LIMIT_PERCENT, passes_low_limits

MAX_STEP_FRACTION = 0.10
STEP_ROUNDING = 0.1
RECENT_DAYS = 90
AFTER_MEAL_HOURS = 4
HOLD_DAYS = 14
EVALUATION = (
    f"Compare the {HOLD_DAYS} days before the change with the {HOLD_DAYS} days after it, over the affected hours plus "
    f"{AFTER_MEAL_HOURS} hours: time in range 70-180 mg/dL with a day-level bootstrap 90% interval, time below 70 (limit "
    f"{LOW_LIMIT_PERCENT:.0f}%) and below 54 (limit {VERY_LOW_LIMIT_PERCENT:.0f}%), and the re-estimated effective I:C. "
    "Days listed in context_events.csv are left out. Before-versus-after in one person is associational."
)
PRIMARY = "primary"


def round_step(value: float) -> float:
    return round(round(value / STEP_ROUNDING) * STEP_ROUNDING, 1)


def segment_summary(
    parts: list[DayPart],
    settings: dict[float, dict[str, float]],
    outcomes_90: pd.DataFrame,
    outcomes_14: pd.DataFrame,
    evidence: pd.DataFrame,
    *,
    pump_serial: str,
    profile_id: float,
) -> pd.DataFrame:
    """One row per pump segment (never merged): its settings now, how glucose did over its hours, and what the
    breakfasts in it say about the I:C. Outcomes lead; the I:C evidence supports them.

    ``outcomes_90`` and ``outcomes_14`` are ``day_part_table`` results for the same parts.
    """
    primary = evidence[evidence["analysis"].eq(PRIMARY) & evidence["pump_serial"].eq(pump_serial) & evidence["profile_id"].eq(profile_id)] if evidence is not None and not evidence.empty else pd.DataFrame()
    rows = []
    for part in parts:
        start = float(part.start_minutes)
        now = settings.get(start, {})
        o90 = outcomes_90[outcomes_90["day_part"].eq(part.name)].iloc[0]
        o14 = outcomes_14[outcomes_14["day_part"].eq(part.name)].iloc[0]
        periods = primary[primary["pump_segment_start_minutes"].eq(start)].sort_values("period_end") if not primary.empty else primary
        at_now = periods[periods["programmed_icr"].map(round_step).eq(round_step(now.get("icr", np.nan)))] if not periods.empty else periods
        enough = periods[periods["n_meals"].ge(MIN_LOGGED_MEALS_FOR_ICR)] if not periods.empty else periods
        latest_enough = enough.iloc[-1] if not enough.empty else None
        collected = int(at_now["n_meals"].iloc[-1]) if not at_now.empty else 0
        if passes_low_limits(o90) is False:
            reading = "lows over a limit: no added insulin in these hours"
        elif o90["time_in_range_pct"] >= 70:
            reading = "meeting the time-in-range target"
        else:
            reading = "below the 70% time-in-range target with lows within limits"
        if o14["time_in_range_pct"] < o90["time_in_range_pct"] - 10:
            reading += f"; the last 14 days are {o90['time_in_range_pct'] - o14['time_in_range_pct']:.0f} points lower"
        rows.append({
            "hours": part.name, "icr_now": now.get("icr", np.nan), "cf_now": now.get("cf", np.nan), "basal_now": now.get("basal", np.nan),
            "tir_90d": o90["time_in_range_pct"], "below_70_90d": o90["time_below_70_pct"], "below_54_90d": o90["time_below_54_pct"], "above_180_90d": o90["time_above_180_pct"],
            "tir_14d": o14["time_in_range_pct"], "below_70_14d": o14["time_below_70_pct"],
            "reading": reading, "breakfasts_at_current_icr": collected,
            "evidence_icr": float(latest_enough["effective_icr"]) if latest_enough is not None else np.nan,
            "evidence_lower_90": float(latest_enough["effective_icr_lower_90"]) if latest_enough is not None else np.nan,
            "evidence_upper_90": float(latest_enough["effective_icr_upper_90"]) if latest_enough is not None else np.nan,
            "evidence_setting": float(latest_enough["programmed_icr"]) if latest_enough is not None else np.nan,
            "evidence_breakfasts": int(latest_enough["n_meals"]) if latest_enough is not None else 0,
            "had_breakfasts": bool(not periods.empty),
        })
    return pd.DataFrame(rows)


