"""The data behind the front page: everything the person looks at, gathered once from one run.

``build_dashboard_data`` returns plain JSON-ready values (no NaN); ``dashboard_page.render_dashboard`` turns them into
one self-contained HTML page. ``find_problem_areas`` is the short list the page opens with, in the order to look.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from . import episodes as ep
from .context_rules import EXERCISE_AFTER_HOURS
from .controller import basal_signal, controller_summary, delivery_intervals
from .outcomes import (
    LOW_LIMIT_PERCENT,
    TARGET_HIGH,
    TARGET_LOW,
    VERY_LOW_LIMIT_PERCENT,
    VERY_HIGH,
    DayPart,
    _prepare,
    baseline_table,
    daily_metrics,
    day_part_table,
    passes_low_limits,
)

AGP_DAYS = 90
SERIES_DAYS = 120
HIGH_EPISODE_MINUTES = 90
LOW_EPISODE_MINUTES = 15
TIME_IN_RANGE_TARGET = 70.0


def clean(value: Any) -> Any:
    """JSON-safe: NaN and infinity become None, numpy scalars become Python ones, timestamps become ISO text."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return None if pd.isna(value) else pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NaT or value is None:
        return None
    return value


def ambulatory_glucose_profile(cgm: pd.DataFrame, *, as_of: pd.Timestamp, days: int = AGP_DAYS) -> list[dict[str, float]]:
    """Glucose by time of day over the last ``days`` days: the 10th, 25th, 50th, 75th and 90th percentiles per 15 minutes."""
    frame = _prepare(cgm)
    frame = frame[(frame["timestamp"] >= as_of - pd.Timedelta(days=days)) & (frame["timestamp"] < as_of)]
    frame = frame.assign(bin=frame["minute"] // 15)
    rows = []
    for bin_, group in frame.groupby("bin"):
        p10, p25, p50, p75, p90 = np.percentile(group["glucose"], [10, 25, 50, 75, 90])
        rows.append({"minute": int(bin_) * 15, "p10": p10, "p25": p25, "p50": p50, "p75": p75, "p90": p90, "n": int(len(group))})
    return rows


def daily_series(cgm: pd.DataFrame, *, as_of: pd.Timestamp, days: int = SERIES_DAYS) -> list[dict[str, Any]]:
    """Daily time in range, below 70 and above 180 for counted days, with a 14-day rolling mean of time in range."""
    daily = daily_metrics(cgm)
    daily = daily[daily["counted"] & (daily.index < as_of) & (daily.index >= as_of - pd.Timedelta(days=days))].copy()
    daily["tir_14d"] = daily["time_in_range_pct"].rolling(14, min_periods=7).mean()
    return [
        {"date": str(index.date()), "tir": row["time_in_range_pct"], "below_70": row["time_below_70_pct"], "above_180": row["time_above_180_pct"],
         "mean": row["mean_glucose_mgdl"], "tir_14d": row["tir_14d"]}
        for index, row in daily.iterrows()
    ]


def episodes_by_hour(cgm: pd.DataFrame, *, as_of: pd.Timestamp, days: int = AGP_DAYS) -> dict[str, Any]:
    """Counts by starting hour of sustained highs (above 250 for 90 minutes or more) and lows (below 70 for 15 minutes or more)."""
    frame = _prepare(cgm)
    glucose = frame.set_index("slot")["glucose"].sort_index().asfreq("5min")
    glucose = glucose[(glucose.index >= as_of - pd.Timedelta(days=days)) & (glucose.index < as_of)]

    def runs(mask: pd.Series, minutes: int) -> list[pd.Timestamp]:
        mask = mask.fillna(False)
        group = (mask != mask.shift()).cumsum()
        return [g.index[0] for _, g in mask[mask].groupby(group[mask]) if len(g) * 5 >= minutes]

    highs, lows = runs(glucose > VERY_HIGH, HIGH_EPISODE_MINUTES), runs(glucose < TARGET_LOW, LOW_EPISODE_MINUTES)
    return {
        "days": days,
        "highs": [sum(1 for t in highs if t.hour == hour) for hour in range(24)],
        "lows": [sum(1 for t in lows if t.hour == hour) for hour in range(24)],
        "high_total": len(highs), "low_total": len(lows),
    }


def hhmm(minutes: float) -> str:
    return f"{int(minutes) // 60:02d}:{int(minutes) % 60:02d}"


def find_problem_areas(
    segments: pd.DataFrame,
    evidence_rows: list[dict[str, Any]],
    signals: pd.DataFrame,
    episodes: dict[str, Any],
    proposal: dict[str, Any],
) -> list[dict[str, str]]:
    """The short list to open with, safest first: lows over a limit, then what the evidence says to change, then what to watch.

    Each entry has a ``level`` (``act``, ``watch`` or ``ok``), a ``title``, a plain ``summary`` and the ``tab`` that shows it.
    """
    areas: list[dict[str, str]] = []
    for row in segments.itertuples(index=False):
        if row.below_70_90d >= LOW_LIMIT_PERCENT or row.below_54_90d >= VERY_LOW_LIMIT_PERCENT:
            areas.append({"level": "act", "tab": "glucose", "title": f"Lows over a limit, {row.hours}",
                          "summary": f"Below 70 for {row.below_70_90d:.1f}% and below 54 for {row.below_54_90d:.2f}% of the last 90 days (limits 4% and 1%). No added insulin in these hours."})
    for item in evidence_rows:
        if item["now"] is not None and abs(item["now"] - item["programmed"]) >= 0.05 and item["latest"] and item["n"] >= 5 and item["lower"] is not None:
            # The setting was changed after these meals: say whether the new value fits what they showed.
            fits = abs(item["effective"] - item["now"]) < abs(item["effective"] - item["programmed"])
            areas.append({"level": "ok" if fits else "watch", "tab": "meals", "title": f"I:C at {item['segment']} was changed to {item['now']:.1f}",
                          "summary": f"The {item['n']} meals at {item['programmed']:.1f} behaved like {item['effective']:.1f} g/U (90% confidence interval {item['lower']:.1f} to {item['upper']:.1f}), so the change " + ("fits them. Evidence at the new setting is still being collected." if fits else "does not clearly fit them.")})
            continue
        if item["current"] and item["n"] >= 5 and item["lower"] is not None and not (item["lower"] <= item["programmed"] <= item["upper"]):
            direction = "stronger (a smaller g/U)" if item["effective"] < item["programmed"] else "weaker (a larger g/U)"
            areas.append({"level": "act", "tab": "meals", "title": f"I:C at {item['segment']} looks too weak" if item["effective"] < item["programmed"] else f"I:C at {item['segment']} looks too strong",
                          "summary": f"{item['n']} meals behaved like {item['effective']:.1f} g/U (90% confidence interval {item['lower']:.1f} to {item['upper']:.1f}); the pump has {item['programmed']:.1f}. The evidence points to a {direction} ratio."})
    for row in signals.itertuples(index=False):
        if row.signal in ("basal likely too low", "basal likely too high"):
            strict = getattr(row, "strict_signal", None)
            summary = (f"Over {row.window_days} days ({row.days} days, {row.clean_hours:.0f} clean hours), the size-scaled rule "
                       f"estimated {row.pooled_excess:+.0%} commanded basal relative to the current programmed rate. "
                       f"Mean glucose was {row.mean_glucose:.0f} mg/dL.")
            if strict == row.signal:
                title = f"{row.signal.capitalize()}, {row.part}"
                summary += " The strict rule gave the same signal."
            else:
                title = f"Basal signal depends on the clean-time rule, {row.part}"
                strict_excess = getattr(row, "strict_excess", float("nan"))
                if pd.notna(strict_excess):
                    summary += f" The strict rule estimated {strict_excess:+.0%} and gave a different signal."
                else:
                    summary += " The strict rule had insufficient support."
            areas.append({"level": "watch", "tab": "basal", "title": title, "summary": summary})
    for row in segments.itertuples(index=False):
        if row.below_70_90d < LOW_LIMIT_PERCENT and row.tir_90d < TIME_IN_RANGE_TARGET:
            worse = row.tir_14d < row.tir_90d - 10
            areas.append({"level": "watch", "tab": "glucose", "title": f"{row.hours}: below the 70% target" + (", getting worse" if worse else ""),
                          "summary": f"In range {row.tir_90d:.0f}% over 90 days" + (f" and {row.tir_14d:.0f}% over the last 14." if worse else ".") + f" Above 180 for {row.above_180_90d:.0f}%."})
    highs = episodes["highs"]
    if episodes["high_total"] >= 6:
        pair = max(range(24), key=lambda h: highs[h] + highs[(h + 1) % 24])
        share = (highs[pair] + highs[(pair + 1) % 24]) / episodes["high_total"]
        if share >= 0.2:
            areas.append({"level": "watch", "tab": "glucose", "title": f"Sustained highs cluster {pair:02d}:00-{(pair + 2) % 24:02d}:00",
                          "summary": f"{highs[pair] + highs[(pair + 1) % 24]} of {episodes['high_total']} stretches above 250 mg/dL for 90 minutes or more in the last {episodes['days']} days started in these two hours."})
    if proposal.get("status") == "held" and proposal.get("reason", "").startswith("waiting"):
        areas.append({"level": "watch", "tab": "settings", "title": "A change is waiting", "summary": proposal["reason"][0].upper() + proposal["reason"][1:] + "."})
    if not any(a["level"] != "ok" for a in areas) and not areas:
        areas.append({"level": "ok", "tab": "glucose", "title": "Nothing needs attention", "summary": "Every segment meets the targets and no setting looks off."})
    order = {"act": 0, "watch": 1, "ok": 2}
    lows_first = sorted(areas, key=lambda a: (order[a["level"]], 0 if a["title"].startswith("Lows") else 1))
    return lows_first


def build_dashboard_data(
    *,
    cgm: pd.DataFrame,
    parts: list[DayPart],
    settings_now: dict[float, dict[str, float]],
    segments: pd.DataFrame,
    evidence: pd.DataFrame,
    proposal_row: dict[str, Any],
    records: pd.DataFrame,
    controller_daily: pd.DataFrame,
    as_of: pd.Timestamp,
    run: dict[str, Any],
    pump_serial: str,
    profile_id: float,
    episodes: dict[str, Any] | None = None,
    controller_daily_strict: pd.DataFrame | None = None,
    current_rates: dict[str, float] | None = None,
) -> dict[str, Any]:
    baselines = baseline_table(cgm, as_of=as_of)
    by_part_90 = day_part_table(cgm, parts, as_of=as_of, days=90)
    by_part_14 = day_part_table(cgm, parts, as_of=as_of, days=14)
    episodes_detail = episodes
    episodes = episodes_by_hour(cgm, as_of=as_of)

    evidence_rows: list[dict[str, Any]] = []
    if evidence is not None and not evidence.empty:
        # Per pump segment: the latest period with enough meals at the I:C the pump holds now is "current"; when no period
        # matches it, the setting was changed after the last meals and the latest period is reported as "earlier".
        primary = evidence[evidence["analysis"].eq("primary") & evidence["pump_serial"].eq(pump_serial) & evidence["profile_id"].eq(profile_id) & np.isfinite(evidence["effective_icr"])]
        rows = list(primary.itertuples(index=False))
        for segment in sorted({r.pump_segment_start_minutes for r in rows}):
            group = [r for r in rows if r.pump_segment_start_minutes == segment]
            now = settings_now.get(float(segment), {}).get("icr")
            matching = [r for r in group if now is not None and abs(r.programmed_icr - now) < 0.05 and r.n_meals >= 5]
            chosen = max(matching, key=lambda r: pd.Timestamp(r.period_end)) if matching else None
            latest = max(group, key=lambda r: pd.Timestamp(r.period_end))
            for row in group:
                evidence_rows.append(
                    {"segment": hhmm(segment), "programmed": float(row.programmed_icr), "effective": float(row.effective_icr), "lower": float(row.effective_icr_lower_90),
                     "upper": float(row.effective_icr_upper_90), "n": int(row.n_meals), "start": str(pd.Timestamp(row.period_start).date()), "end": str(pd.Timestamp(row.period_end).date()),
                     "current": chosen is row, "now": now, "latest": chosen is None and row is latest}
                )
    signals = basal_signal(controller_daily, current_rates) if not controller_daily.empty else pd.DataFrame(columns=["part", "profile_rate", "window_days", "days", "clean_hours", "share_over_x", "share_over_lo", "share_over_hi", "prob_rule_met", "pooled_excess", "excess_lo", "excess_hi", "mean_glucose", "below_70_pct", "below_54_pct", "signal"])
    if controller_daily_strict is not None and not controller_daily_strict.empty and not signals.empty:
        strict = basal_signal(controller_daily_strict, current_rates)
        signals = signals.merge(strict[["part", "days", "clean_hours", "pooled_excess", "signal"]].rename(columns={"days": "strict_days", "clean_hours": "strict_hours", "pooled_excess": "strict_excess", "signal": "strict_signal"}), on="part", how="left")
    summary = controller_summary(controller_daily) if not controller_daily.empty else pd.DataFrame()
    night = controller_daily[controller_daily["part"].eq(controller_daily["part"].iloc[0])] if not controller_daily.empty else pd.DataFrame()
    nightly = [{"date": str(pd.Timestamp(r.date).date()), "excess": r.excess_fraction, "glucose": r.mean_glucose} for r in night[night["date"] >= night["date"].max() - pd.Timedelta(days=180)].sort_values("date").itertuples(index=False)] if not night.empty else []

    problems = find_problem_areas(segments, evidence_rows, signals, episodes, proposal_row)
    overnight_flagged = any(a["title"].startswith("Basal likely too low") for a in problems)
    if episodes_detail:
        extra = episode_problem_areas(episodes_detail)
        if overnight_flagged:
            for area in extra:
                if area["title"].startswith("Overnight low"):
                    area["summary"] += " Basal is also flagged as too low in the same hours: the controller adds insulin on most nights, so check whether these lows followed nights it over-delivered before changing basal."
        problems = [a for a in problems if a["level"] != "ok"] + extra + [a for a in problems if a["level"] == "ok"]
    change_rows = []
    for r in records.sort_values("entered_on", na_position="last").itertuples(index=False):
        change_rows.append({"id": r.record_id, "origin": r.origin, "setting": r.setting, "segment": hhmm(r.segment_start_minutes), "old": r.old_value, "new": r.new_value,
                            "entered_on": r.entered_on, "status": r.status, "result": r.result, "concurrent": r.concurrent_changes})
    return clean(
        {
            "run": run,
            "as_of": str(as_of.date()),
            "targets": {"low": TARGET_LOW, "high": TARGET_HIGH, "tir": TIME_IN_RANGE_TARGET},
            "baseline": baselines.to_dict("records"),
            "agp": ambulatory_glucose_profile(cgm, as_of=as_of),
            "daily": daily_series(cgm, as_of=as_of),
            "segments": segments.assign(
                start=[p.start_minutes for p in parts], end=[p.end_minutes for p in parts],
                cf_now=[settings_now.get(float(p.start_minutes), {}).get("cf", np.nan) for p in parts],
            ).to_dict("records"),
            "parts_90": by_part_90.to_dict("records"),
            "parts_14": by_part_14.to_dict("records"),
            "episodes": episodes,
            "evidence": evidence_rows,
            "proposal": proposal_row,
            "changes": change_rows,
            "controller": {"nightly": nightly, "signals": signals.to_dict("records"), "summary": summary.to_dict("records")},
            "episode_detail": episodes_detail,
            "problems": problems,
            "low_limits_ok": bool(passes_low_limits(baselines.iloc[-1])),
        }
    )


def to_script_json(data: dict[str, Any]) -> str:
    """JSON safe to place inside a ``<script>`` element."""
    return json.dumps(data, separators=(",", ":")).replace("</", "<\\/").replace("<!--", "<\\!--")


TRACE_HOURS = 3
LATEST_EPISODES = 30


def episodes_payload(
    cgm: pd.DataFrame, grid: pd.DataFrame, parts: list[DayPart], site_events: pd.DataFrame, rates: pd.DataFrame, unlogged: pd.DataFrame, exercise: pd.DataFrame, workouts: pd.DataFrame, *, site_observed_from: pd.Timestamp, as_of: pd.Timestamp, days: int = SERIES_DAYS
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Episodes of the last ``days`` days with their patterns, the repeats, the exercise effect and what each trace looks like,
    and the classified episodes themselves."""
    found = ep.find_episodes(cgm, start=as_of - pd.Timedelta(days=days), end=as_of)
    delivery = delivery_intervals(rates) if rates is not None and not rates.empty else None
    classified = ep.classify_all(ep.describe_episodes(found, grid, site_events, unlogged, delivery, exercise, site_coverage_from=site_observed_from))
    series = grid.assign(timestamp=pd.to_datetime(grid["timestamp"])).set_index("timestamp").sort_index()
    latest = []
    for row in classified.sort_values("start", ascending=False).head(LATEST_EPISODES).itertuples(index=False):
        t0, t1 = row.start - pd.Timedelta(hours=TRACE_HOURS), row.end + pd.Timedelta(hours=TRACE_HOURS)
        window = series[(series.index >= t0) & (series.index <= t1)]
        offsets = ((window.index - t0).total_seconds() / 60).astype(int)
        bolus, carbs = window["bolus_units"].fillna(0.0), window["carb_grams"].fillna(0.0)
        sites = site_events[(site_events["timestamp"] >= t0) & (site_events["timestamp"] <= t1)]["timestamp"] if not site_events.empty else []
        part = next((p.name for p in parts if p.contains(np.array([row.start.hour * 60 + row.start.minute]))[0]), "")
        latest.append(
            {"kind": row.kind, "start": row.start, "end": row.end, "hours": row.hours, "extreme": row.extreme, "pattern": row.pattern, "label": {**ep.HIGH_PATTERNS, **ep.LOW_PATTERNS}[row.pattern],
             "evidence": row.evidence, "segment": part,
             "trace": {"t0": t0, "step": 5, "glucose": [None if not np.isfinite(g) else float(g) for g in window["glucose"]],
                       "boluses": [[int(o), float(u)] for o, u in zip(offsets, bolus) if u > 0], "carbs": [[int(o), float(g)] for o, g in zip(offsets, carbs) if g > 0],
                       "sites": [int((t - t0).total_seconds() / 60) for t in sites], "start_offset": TRACE_HOURS * 60, "end_offset": int((row.end - t0).total_seconds() / 60)}}
        )
    counts = (classified.groupby(["kind", "pattern"]).size().reset_index(name="n").to_dict("records")) if not classified.empty else []
    payload = {
        "days": days, "labels": {**ep.HIGH_PATTERNS, **ep.LOW_PATTERNS}, "counts": counts, "highs": int((classified["kind"] == "high").sum()) if not classified.empty else 0,
        "lows": int((classified["kind"] == "low").sum()) if not classified.empty else 0, "latest": latest,
        "repeats": ep.repeats(classified, parts), "exercise": exercise_over_logged_period(grid, exercise, workouts),
        "site": ep.site_change_link(classified, site_events), "playbook": {k: {"points_at": v[0], "advice": v[1]} for k, v in ep.PLAYBOOK.items()},
    }
    return payload, classified


def exercise_over_logged_period(grid: pd.DataFrame, exercise: pd.DataFrame, workouts: pd.DataFrame) -> dict[str, Any]:
    """The exercise effect over the whole period the workout log covers (first workout to last, plus the hours after): outside it a
    meal would count as 'not during exercise' only because no workout was recorded."""
    if workouts is None or workouts.empty:
        return {}
    start, end = pd.to_datetime(workouts["start_timestamp"]).min(), pd.to_datetime(workouts["end_timestamp"]).max() + pd.Timedelta(hours=EXERCISE_AFTER_HOURS)
    stamp = pd.to_datetime(grid["timestamp"])
    result = ep.exercise_effect(grid[(stamp >= start) & (stamp <= end)], exercise)
    return {**result, "from": str(start.date()), "to": str(end.date())} if result else {}


def episode_problem_areas(payload: dict[str, Any]) -> list[dict[str, str]]:
    areas = []
    for r in payload["repeats"]:
        areas.append({"level": "watch", "tab": "episodes", "title": f"{r['label']} repeats at {r['segment']}",
                      "summary": f"{r['n']} of {r['of']} {r['kind']} episodes in this segment over the last {payload['days']} days fit this pattern. It points at {r['points_at']}."})
    ex = payload.get("exercise") or {}
    # Only when the ranges are apart: after-workout lows clearly above the others.
    if ex.get("after_workout_meals", 0) >= 30 and ex.get("other_meals", 0) >= 30 and ex["after_workout_interval"][0] > ex["other_interval"][1]:
        areas.append({"level": "watch", "tab": "episodes", "title": "Lows follow meals during or after exercise more often",
                      "summary": f"During exercise or within {EXERCISE_AFTER_HOURS:.0f} hours after it, {ex['after_workout_share']:.0%} of meal boluses were followed by a low within 4 hours ({ex['after_workout_meals']} meals), against {ex['other_share']:.0%} otherwise ({ex['other_meals']} meals), {ex['from']} to {ex['to']}."})
    return areas
