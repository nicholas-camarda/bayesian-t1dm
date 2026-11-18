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

from .controller import basal_signal, controller_summary
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
                          "summary": f"The {item['n']} meals at {item['programmed']:.1f} behaved like {item['effective']:.1f} g/U (90% range {item['lower']:.1f} to {item['upper']:.1f}), so the change " + ("fits them. Evidence at the new setting is still being collected." if fits else "does not clearly fit them.")})
            continue
        if item["current"] and item["n"] >= 5 and item["lower"] is not None and not (item["lower"] <= item["programmed"] <= item["upper"]):
            direction = "stronger (a smaller g/U)" if item["effective"] < item["programmed"] else "weaker (a larger g/U)"
            areas.append({"level": "act", "tab": "meals", "title": f"I:C at {item['segment']} looks too weak" if item["effective"] < item["programmed"] else f"I:C at {item['segment']} looks too strong",
                          "summary": f"{item['n']} meals behaved like {item['effective']:.1f} g/U (90% range {item['lower']:.1f} to {item['upper']:.1f}); the pump has {item['programmed']:.1f}. The evidence points to a {direction} ratio."})
    for row in signals.itertuples(index=False):
        if row.signal in ("basal likely too low", "basal likely too high"):
            areas.append({"level": "act", "tab": "basal", "title": f"{row.signal.capitalize()}, {row.part}",
                          "summary": f"Over the last {row.days} clean days the controller ran above the profile by 25% or more on {row.share_over_x:.0%} of days (median {row.median_excess_fraction:+.0%}); mean glucose {row.mean_glucose:.0f} mg/dL."})
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
) -> dict[str, Any]:
    baselines = baseline_table(cgm, as_of=as_of)
    by_part_90 = day_part_table(cgm, parts, as_of=as_of, days=90)
    by_part_14 = day_part_table(cgm, parts, as_of=as_of, days=14)
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
    signals = basal_signal(controller_daily) if not controller_daily.empty else pd.DataFrame(columns=["part", "days", "share_over_x", "share_under_minus_x", "mean_glucose", "below_70_pct", "below_54_pct", "median_excess_fraction", "signal"])
    summary = controller_summary(controller_daily) if not controller_daily.empty else pd.DataFrame()
    night = controller_daily[controller_daily["part"].eq(controller_daily["part"].iloc[0])] if not controller_daily.empty else pd.DataFrame()
    nightly = [{"date": str(pd.Timestamp(r.date).date()), "excess": r.excess_fraction, "glucose": r.mean_glucose} for r in night.sort_values("date").itertuples(index=False)] if not night.empty else []

    problems = find_problem_areas(segments, evidence_rows, signals, episodes, proposal_row)
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
            "problems": problems,
            "low_limits_ok": bool(passes_low_limits(baselines.iloc[-1])),
        }
    )


def to_script_json(data: dict[str, Any]) -> str:
    """JSON safe to place inside a ``<script>`` element."""
    return json.dumps(data, separators=(",", ":")).replace("</", "<\\/").replace("<!--", "<\\!--")
