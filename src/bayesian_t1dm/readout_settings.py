"""The readout's settings check: per pump segment, whether basal and I:C have been performing well.

Both pieces of evidence are reused as they are:

- basal: :func:`controller.basal_signal`, Control-IQ's delivery against the programmed basal over clean hours;
- I:C: the effective I:C from clean first meals (:func:`therapy_research.run_latent_meal_icr_research`, summarised by
  :func:`proposals.segment_summary`), compared with the ratio the segment holds now.

Each becomes one of four statuses: fits, needs more insulin, needs less insulin, not enough data. A suggestion is one
step of 10% or 20%, the step nearer the size of the gap. Nothing that adds insulin is suggested in a segment whose lows
are over a consensus limit over the last 90 days. Control-IQ adjusts basal throughout, so both are observational and
are for discussion with the endocrinologist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .changes import SETTING_FIELDS, detect_setting_changes, evaluate_change
from .controller import basal_signal, controller_inputs, programmed_units_by_slot
from .outcomes import day_part_table, passes_low_limits
from .proposals import segment_summary
from .therapy_research import (
    extract_latent_meal_therapy_settings_from_tandem_raw,
    load_excluded_periods,
    parse_therapy_segments,
    run_latent_meal_icr_research,
)

STATUSES = ("fits", "needs more insulin", "needs less insulin", "not enough data")
STEPS = (0.10, 0.20)
MEAL_SEGMENTS = ("morning", "afternoon", "evening")  # first meals of the day counted as I:C evidence (as the proposal does)
CHANGES_LOOKBACK_DAYS = 90


def step_for(gap: float) -> float:
    """The suggested step: 10% or 20%, whichever is nearer the absolute size of the gap."""
    return min(STEPS, key=lambda step: abs(step - abs(gap)))


def icr_evidence(analysis, raw_root) -> pd.DataFrame:
    """Effective I:C evidence from clean first meals, with the same settings the breakfast proposal uses."""
    merged = analysis.merged
    segments = parse_therapy_segments(None)
    result = run_latent_meal_icr_research(
        merged.preparation.dataset, segments=segments, meal_proxy_mode="strict", research_scope="full",
        therapy_settings=extract_latent_meal_therapy_settings_from_tandem_raw(raw_root, segments=segments),
        excluded_periods=load_excluded_periods(raw_root), nutrition=merged.data.nutrition, unlogged=analysis.unlogged,
        exercise=analysis.exercise, programmed_units=programmed_units_by_slot(analysis), meal_segments=MEAL_SEGMENTS,
    )
    return result.effective_icr_evidence


def basal_status(row: pd.Series | None) -> tuple[str, str, float]:
    """Status, evidence sentence and gap (the controller's pooled extra delivery) for one segment's basal."""
    if row is None or pd.isna(row.get("share_over_x")):
        return "not enough data", (row["signal"] if row is not None else "no controller records"), np.nan
    detail = (f"Control-IQ delivered {100 * row['pooled_excess']:+.0f}% against the programmed rate over {row['clean_hours']:.0f} clean hours "
              f"on {int(row['days'])} days (90% range {100 * row['excess_lo']:+.0f}% to {100 * row['excess_hi']:+.0f}%)")
    if row["signal"] == "basal likely too low":
        return "needs more insulin", detail, float(row["pooled_excess"])
    if row["signal"] == "basal likely too high":
        return "needs less insulin", detail, float(row["pooled_excess"])
    return "fits", detail, float(row["pooled_excess"])


def icr_status(row: pd.Series) -> tuple[str, str, float]:
    """Status, evidence sentence and gap for one segment's I:C: the setting against the 90% range of the effective I:C."""
    if not row["had_breakfasts"] or not np.isfinite(row["evidence_icr"]):
        return "not enough data", "no clean first meals in these hours", np.nan
    now, effective, lower, upper = row["icr_now"], row["evidence_icr"], row["evidence_lower_90"], row["evidence_upper_90"]
    detail = f"{int(row['evidence_breakfasts'])} clean meals behave like {effective:.1f} g/U (90% range {lower:.1f} to {upper:.1f}); the pump is set to {now:.1f}"
    gap = now / effective - 1.0  # positive: carbs need more insulin than the setting gives
    if upper < now:
        return "needs more insulin", detail, gap
    if lower > now:
        return "needs less insulin", detail, gap
    return "fits", detail, gap


def suggestion_for(setting: str, now: float, status: str, gap: float, *, lows_ok: bool) -> tuple[str, str]:
    """The status after the low-limit rule, and the suggested step (more insulin: basal up, I:C down)."""
    if status not in ("needs more insulin", "needs less insulin"):
        return status, ""
    more = status == "needs more insulin"
    if more and not lows_ok:
        return "fits", "held: lows in these hours are over a consensus limit, so no added insulin"
    step = step_for(gap)
    factor = (1 + step) if (setting == "basal") == more else (1 - step)
    unit, digits = ("U/h", 2) if setting == "basal" else ("g/U", 1)
    return status, f"consider {now:g} to {now * factor:.{digits}f} {unit} ({100 * step:.0f}% {'more' if more else 'less'} insulin)"


def settings_check(analysis, raw_root, *, as_of: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """One row per segment and setting with status, suggestion and evidence; plus the evidence tables behind them."""
    parts, _ = analysis.day_parts()
    settings = analysis.segment_settings()
    current_rates = {part.name: settings[float(part.start_minutes)]["basal"] for part in parts if float(part.start_minutes) in settings}
    controller_daily, _, _ = controller_inputs(analysis, parts)
    basal = basal_signal(controller_daily, current_rates) if not controller_daily.empty else pd.DataFrame(columns=["part", "signal"])
    evidence = icr_evidence(analysis, raw_root)
    outcomes_90 = day_part_table(analysis.cgm, parts, as_of=as_of, days=90)
    outcomes_14 = day_part_table(analysis.cgm, parts, as_of=as_of, days=14)
    segments = segment_summary(parts, settings, outcomes_90, outcomes_14, evidence, pump_serial=analysis.pump, profile_id=float(analysis.profile_id))

    rows = []
    for part, segment in zip(parts, segments.itertuples(index=False)):
        seg = pd.Series(segment._asdict())
        lows_ok = bool(passes_low_limits(outcomes_90[outcomes_90["day_part"].eq(part.name)].iloc[0]))
        basal_row = basal[basal["part"].eq(part.name)]
        for setting, now, (status, detail, gap) in (
            ("basal", seg["basal_now"], basal_status(basal_row.iloc[0] if not basal_row.empty else None)),
            ("I:C", seg["icr_now"], icr_status(seg)),
        ):
            status, suggestion = suggestion_for(setting, now, status, gap, lows_ok=lows_ok)
            rows.append({"part": part.name, "setting": setting, "now": now, "status": status, "suggestion": suggestion, "evidence": detail,
                         "tir_14d": seg["tir_14d"], "tir_90d": seg["tir_90d"], "below_70_90d": seg["below_70_90d"], "lows_within_limits": lows_ok})
    return pd.DataFrame(rows), {"basal_signal": basal, "segment_summary": segments, "icr_evidence": evidence}


def recent_changes(analysis, *, as_of: pd.Timestamp) -> pd.DataFrame:
    """Setting changes on the running profile in the last 90 days, each with its 14-day before/after check."""
    parts, _ = analysis.day_parts()
    changes = detect_setting_changes(analysis.merged.profile_events)
    changes = changes[changes["pump_serial"].eq(analysis.pump) & changes["profile_id"].eq(analysis.profile_id)
                      & (pd.to_datetime(changes["timestamp"]) >= as_of - pd.Timedelta(days=CHANGES_LOOKBACK_DAYS))]
    rows = []
    for change in changes.itertuples(index=False):
        part = next(p for p in parts if p.contains(np.array([int(change.segment_start_minutes)]))[0])
        result = evaluate_change(analysis.cgm, pd.Timestamp(change.timestamp), part, now=as_of, excluded_days=analysis.excluded_days)
        rows.append({"changed": pd.Timestamp(change.timestamp).date(), "setting": SETTING_FIELDS.get(change.setting, change.setting), "hours": part.name,
                     "old_value": change.old_value, "new_value": change.new_value,
                     **{key: result.get(key) for key in ("status", "reason", "tir_before", "tir_after", "tir_difference", "tir_difference_lower_90", "tir_difference_upper_90", "low_limits_after", "verdict")}})
    columns = ["changed", "setting", "hours", "old_value", "new_value", "status", "reason", "tir_before", "tir_after", "tir_difference", "tir_difference_lower_90", "tir_difference_upper_90", "low_limits_after", "verdict"]
    return pd.DataFrame(rows, columns=columns)
