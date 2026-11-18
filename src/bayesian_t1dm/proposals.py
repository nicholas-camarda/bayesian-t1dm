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

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .latent_meal_trajectory import MIN_LOGGED_MEALS_FOR_ICR
from .outcomes import DayPart, LOW_LIMIT_PERCENT, VERY_LOW_LIMIT_PERCENT, passes_low_limits, pooled_metrics

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


@dataclass(frozen=True)
class Proposal:
    setting: str
    status: str  # "proposed" or "held"
    reason: str
    pump_serial: str = ""
    profile_id: float = float("nan")
    segment_start_minutes: float = float("nan")
    affected_hours: str = ""
    current_value: float = float("nan")
    proposed_value: float = float("nan")
    direction: str = ""  # "adds insulin" or "removes insulin"
    estimate: float = float("nan")
    estimate_lower_90: float = float("nan")
    estimate_upper_90: float = float("nan")
    breakfasts: int = 0
    checks: dict[str, str] = field(default_factory=dict)
    safety: dict[str, float] = field(default_factory=dict)
    evaluation: str = EVALUATION

    def to_row(self) -> dict[str, Any]:
        row = {key: value for key, value in self.__dict__.items() if key not in {"checks", "safety"}}
        row["checks"] = "; ".join(f"{name}: {result}" for name, result in self.checks.items())
        row["safety"] = "; ".join(f"{name} {value:.2f}" for name, value in self.safety.items())
        return row


def round_step(value: float) -> float:
    return round(round(value / STEP_ROUNDING) * STEP_ROUNDING, 1)


def proposed_step(current: float, estimate: float) -> float:
    """The new I:C: the smaller of a 10% move and the distance to the estimate, toward the estimate."""
    distance = abs(estimate - current)
    move = min(MAX_STEP_FRACTION * current, distance)
    return round_step(current + np.sign(estimate - current) * move)


def affected_part(start_minutes: int, next_start_minutes: int) -> DayPart:
    """The pump segment's hours plus the 4 hours after, which its I:C still acts on."""
    end = (next_start_minutes + AFTER_MEAL_HOURS * 60) % (24 * 60)
    return DayPart(f"{start_minutes // 60:02d}:{start_minutes % 60:02d}-{end // 60:02d}:{end % 60:02d}", int(start_minutes), int(end))


def _held(reason: str, **fields: Any) -> Proposal:
    return Proposal(setting="I:C", status="held", reason=reason, **fields)


def propose_breakfast_icr(
    evidence: pd.DataFrame,
    cgm: pd.DataFrame,
    segment_ends: dict[float, float],
    *,
    as_of: pd.Timestamp,
    current_icr: dict[float, float] | None = None,
) -> Proposal:
    """The I:C proposal over all meals, or a held proposal that says why not.

    ``segment_ends`` maps a pump segment's start (minutes after midnight) to the start of the next, which bounds the
    affected hours. ``current_icr`` maps a segment's start to the I:C the pump holds now: when the person has
    already changed it since the last breakfast, evidence at the old setting is reported but nothing is proposed.
    """
    if evidence is None or evidence.empty:
        return _held("no meal evidence")
    frame = evidence.copy()
    for column in ("period_start", "period_end"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    primary = frame[frame["analysis"].eq(PRIMARY) & frame["is_current_setting"].astype(bool)]
    recent = frame[frame["analysis"].eq(PRIMARY) & (frame["period_end"] >= as_of - pd.Timedelta(days=RECENT_DAYS))]
    if primary.empty or recent.empty:
        return _held("no meal in the last 90 days")
    # Every pump segment of the pump and profile now running is assessed on its own; the best-supported proposal wins.
    now = primary.sort_values("period_end").iloc[-1]
    on_now = primary[primary["pump_serial"].eq(now["pump_serial"]) & primary["profile_id"].eq(now["profile_id"])]
    assessed = []
    for segment, group in on_now.groupby("pump_segment_start_minutes"):
        row = group.sort_values("period_end").iloc[-1]
        assessed.append((float(segment), row, _assess_segment(frame, row, float(segment), cgm, segment_ends, as_of, current_icr)))
    proposed = [(segment, row, result) for segment, row, result in assessed if result.status == "proposed"]
    if proposed:
        # Strongest first: how far the current setting sits outside the evidence interval, relative to the setting.
        def gap(item):
            _, row, result = item
            outside = max(row["effective_icr_lower_90"] - result.current_value, result.current_value - row["effective_icr_upper_90"])
            return (outside / result.current_value, int(row["n_meals"]))
        return max(proposed, key=gap)[2]
    recent_counts = recent[recent["pump_serial"].eq(now["pump_serial"]) & recent["profile_id"].eq(now["profile_id"])].groupby("pump_segment_start_minutes")["n_meals"].sum()
    busiest = float(recent_counts.idxmax()) if not recent_counts.empty else assessed[0][0]
    return next(result for segment, _, result in assessed if segment == busiest)


def _assess_segment(frame: pd.DataFrame, row: pd.Series, segment: float, cgm: pd.DataFrame, segment_ends: dict[float, float], as_of: pd.Timestamp, current_icr: dict[float, float] | None) -> Proposal:
    """The rule applied to one pump segment's current-setting period."""
    identity = dict(
        pump_serial=str(row["pump_serial"]), profile_id=float(row["profile_id"]), segment_start_minutes=segment,
        current_value=float(row["programmed_icr"]), breakfasts=int(row["n_meals"]),
        estimate=float(row["effective_icr"]), estimate_lower_90=float(row["effective_icr_lower_90"]), estimate_upper_90=float(row["effective_icr_upper_90"]),
    )
    held_now = (current_icr or {}).get(segment)
    if held_now is not None and round_step(held_now) != round_step(float(row["programmed_icr"])):
        identity["current_value"] = float(held_now)
        verdict = "supports" if np.isfinite(row["effective_icr"]) and abs(row["effective_icr"] - held_now) < abs(row["effective_icr"] - float(row["programmed_icr"])) else "does not clearly support"
        return _held(
            f"the pump now holds {held_now:.1f} g/U, changed after the last breakfast at {row['programmed_icr']:.1f} g/U; "
            f"the {int(row['n_meals'])} breakfasts at {row['programmed_icr']:.1f} behaved like {row['effective_icr']:.1f} g/U "
            f"(90% range {row['effective_icr_lower_90']:.1f}-{row['effective_icr_upper_90']:.1f}), which {verdict} the change. "
            f"Evidence at {held_now:.1f} is accumulating",
            **identity,
        )
    if int(row["n_meals"]) < MIN_LOGGED_MEALS_FOR_ICR or not np.isfinite(row["effective_icr"]):
        return _held(f"only {int(row['n_meals'])} breakfasts at the current setting ({row['programmed_icr']:.1f} g/U); {MIN_LOGGED_MEALS_FOR_ICR} are needed", **identity)
    current, estimate = float(row["programmed_icr"]), float(row["effective_icr"])
    if row["effective_icr_lower_90"] <= current <= row["effective_icr_upper_90"]:
        return _held("the current setting is inside the evidence interval", **identity)

    # Checks: the other analyses of the same period must agree on the direction when they can be assessed.
    same = frame[
        ~frame["analysis"].eq(PRIMARY)
        & frame["pump_serial"].eq(row["pump_serial"]) & frame["profile_id"].eq(row["profile_id"])
        & frame["pump_segment_start_minutes"].eq(segment) & frame["period_start"].eq(row["period_start"])
        & frame["programmed_icr"].eq(row["programmed_icr"])
    ]
    side = np.sign(estimate - current)
    checks: dict[str, str] = {}
    disagree: list[str] = []
    for check in same.itertuples(index=False):
        if int(check.n_meals) < MIN_LOGGED_MEALS_FOR_ICR or not np.isfinite(check.effective_icr):
            checks[str(check.analysis)] = f"not assessable ({int(check.logged_meal_windows)} breakfasts)"
        elif np.sign(check.effective_icr - current) == side:
            checks[str(check.analysis)] = f"agrees ({check.effective_icr:.1f} g/U, {int(check.n_meals)} breakfasts)"
        else:
            checks[str(check.analysis)] = f"disagrees ({check.effective_icr:.1f} g/U, {int(check.n_meals)} breakfasts)"
            disagree.append(str(check.analysis))
    direction = "adds insulin" if side < 0 else "removes insulin"
    next_start = segment_ends.get(segment)
    part = affected_part(int(segment), int(next_start)) if next_start is not None else None
    if disagree:
        return _held("a check points the other way: " + ", ".join(disagree), direction=direction, checks=checks, **identity)
    safety: dict[str, float] = {}
    if side < 0:
        if part is None:
            return _held("the hours this setting governs are unknown, so the low limits cannot be checked", direction=direction, checks=checks, **identity)
        metrics = pooled_metrics(cgm, start=as_of - pd.Timedelta(days=RECENT_DAYS), end=as_of, part=part)
        safety = {"time below 70 (%)": metrics["time_below_70_pct"], "time below 54 (%)": metrics["time_below_54_pct"]}
        if not passes_low_limits(metrics):
            return _held(
                f"adding insulin is not allowed: over {part.name} time below 70 is {metrics['time_below_70_pct']:.1f}% and below 54 is {metrics['time_below_54_pct']:.2f}%",
                direction=direction, checks=checks, safety=safety, affected_hours=part.name, **identity,
            )
    proposed = proposed_step(current, estimate)
    if proposed == round_step(current):
        return _held("the step rounds to no change", direction=direction, checks=checks, safety=safety, **identity)
    return Proposal(
        setting="I:C", status="proposed",
        reason=f"carbs behaved like {estimate:.1f} g/U (90% range {row['effective_icr_lower_90']:.1f}-{row['effective_icr_upper_90']:.1f}) against the programmed {current:.1f}",
        direction=direction, proposed_value=proposed, checks=checks, safety=safety,
        affected_hours=part.name if part else "", **identity,
    )


def segment_ends_from_parts(parts: list[DayPart]) -> dict[float, float]:
    """Map each pump segment start to the start of the next, from the day parts built on the pump's segments."""
    return {float(part.start_minutes): float(part.end_minutes) for part in parts}


def proposal_markdown(proposal: Proposal, *, as_of: pd.Timestamp) -> str:
    def hhmm(minutes: float) -> str:
        return "unknown" if not np.isfinite(minutes) else f"{int(minutes) // 60:02d}:{int(minutes) % 60:02d}"

    lines = [f"# Breakfast I:C proposal ({as_of.date()})", ""]
    if proposal.status == "proposed":
        lines += [
            f"**Proposed: change the pump segment starting {hhmm(proposal.segment_start_minutes)} from {proposal.current_value:.1f} to "
            f"{proposal.proposed_value:.1f} g/U ({proposal.direction}).** {proposal.reason}.",
            "",
            f"Affected hours (segment plus {AFTER_MEAL_HOURS} h): {proposal.affected_hours}. Evidence: {proposal.breakfasts} breakfasts at the current setting.",
            "",
            "Enter it on the pump yourself, then hold it for at least 14 days before the next change. Discuss it with your endocrinologist.",
        ]
    else:
        lines += [f"**No change proposed.** {proposal.reason[:1].upper()}{proposal.reason[1:]}.", ""]
        if np.isfinite(proposal.segment_start_minutes):
            lines.append(f"Segment checked: starting {hhmm(proposal.segment_start_minutes)}, current {proposal.current_value:.1f} g/U.")
    if proposal.checks:
        lines += ["", "## Checks", ""] + [f"- {name}: {result}" for name, result in proposal.checks.items()]
    if proposal.safety:
        lines += ["", "## Low limits over the affected hours, last 90 days", ""] + [f"- {name}: {value:.2f}" for name, value in proposal.safety.items()]
    lines += ["", "## How it will be judged", "", proposal.evaluation, ""]
    return "\n".join(lines)


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


def segment_markdown(table: pd.DataFrame) -> str:
    lines = [
        "## Glucose by pump segment (last 90 days; last 14 days in brackets)",
        "",
        "Each pump segment keeps its own I:C, correction factor and basal, so segments are not combined. Hours are the segment's own.",
        "",
        "| hours | I:C (g/U) | CF (mg/dL/U) | basal (U/h) | in range 70-180 | below 70 | below 54 | above 180 | reading | meals at this I:C | what the meals say about the I:C |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for row in table.itertuples(index=False):
        evidence = (
            f"{row.evidence_breakfasts} meals at {row.evidence_setting:.1f}{' (the current setting)' if abs(row.evidence_setting - row.icr_now) < 0.05 else ''} behaved like {row.evidence_icr:.1f} ({row.evidence_lower_90:.1f}-{row.evidence_upper_90:.1f})"
            if np.isfinite(row.evidence_icr) else ("no meals in this segment" if row.breakfasts_at_current_icr == 0 and row.evidence_breakfasts == 0 and not row.had_breakfasts else "fewer than 5 meals at any setting")
        )
        lines.append(
            f"| {row.hours} | {row.icr_now:.1f} | {row.cf_now:.0f} | {row.basal_now:.2f} | {row.tir_90d:.1f}% ({row.tir_14d:.1f}%) | {row.below_70_90d:.2f}% ({row.below_70_14d:.2f}%) | "
            f"{row.below_54_90d:.2f}% | {row.above_180_90d:.1f}% | {row.reading} | {min(row.breakfasts_at_current_icr, 99)}{'+' if row.breakfasts_at_current_icr >= MIN_LOGGED_MEALS_FOR_ICR else f' of {MIN_LOGGED_MEALS_FOR_ICR}'} | {evidence} |"
        )
    return "\n".join(lines) + "\n"
