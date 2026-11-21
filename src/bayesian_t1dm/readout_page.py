"""The personal readout page: one self-contained HTML file sized for a phone.

Order: today's rings and why today; this week against the usual range; what moved glucose (the drivers model);
settings check by pump segment; recent setting changes; the last seven days, each as a glucose trace with sleep,
workouts, carbs and site changes marked. Charts are inline SVG so the page opens offline from the OneDrive app.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .figures import READOUT_STYLE
from .outcomes import TIME_IN_RANGE_TARGET_PERCENT
from .readout import DRIVER_COLUMNS, DRIVER_LABELS, DRIVERS, LATE_MEAL_MINUTE
from .readout_settings import HELD
from .readout_model import reading
from .report_page import CSS, _text, esc, fmt

TRACE_DAYS = 7
TRACE_LOW, TRACE_HIGH = 40.0, 350.0
CHART_WIDTH = 340  # drawn at phone width so text stays readable at its true size
STATUS_CLASS = {"fits": "good", "needs more insulin": "warn", "needs less insulin": "warn", HELD: "warn", "not enough data": "info"}
EXTRA_CSS = """
main { max-width: 640px; }
.rings { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 8px; text-align:center; }
.rings .l { color: var(--muted); font-size: 13px; } .rings .s { color: var(--muted); font-size: 12px; }
.why { list-style:none; padding:0; margin:0; } .why li { display:flex; justify-content:space-between; gap:10px; padding:7px 0; border-top:1px solid var(--line); font-size:15px; }
.why li:first-child { border-top: none; } .why .note { color: var(--muted); font-size: 13px; display:block; }
.pill { font-size: 12.5px; padding: 2px 9px; border-radius: 999px; white-space: nowrap; align-self:flex-start; }
.pill.good { background: var(--goodbg); color: var(--good); } .pill.warn { background: var(--warnbg); color: var(--warn); } .pill.info { background: var(--infobg); color: var(--info); }
.big { font-size: 28px; font-weight: 650; } .muted { color: var(--muted); font-size: 14px; }
.seg { padding: 10px 0; border-top: 1px solid var(--line); } .seg:first-of-type { border-top:none; }
.seg .row { display:flex; justify-content:space-between; gap: 10px; margin: 4px 0; font-size: 15px; }
.seg .row b { white-space: nowrap; } .seg .sug { font-size: 13.5px; margin: 2px 0 0; }
.changes li { display:block; } .seg .ev { color: var(--muted); font-size: 12.5px; margin: 2px 0 0; }
"""


@dataclass(frozen=True)
class Readout:
    as_of: pd.Timestamp
    days: pd.DataFrame
    usual_week: tuple[float, float, float]  # 10th percentile, median, 90th percentile of 7-day time in range over the 90 days before the week
    effects: dict[str, pd.DataFrame]
    usable: dict[str, bool]
    model_days: int
    settings: pd.DataFrame
    changes: pd.DataFrame
    cgm: pd.DataFrame
    sleep: pd.DataFrame
    workouts: pd.DataFrame
    carbs: pd.DataFrame
    site_fills: pd.Series
    run_id: str
    built: str


def _ring(value: float, fraction: float, label: str, sub: str, color: str) -> str:
    r, c = 34, 2 * math.pi * 34
    shown = 0.0 if not math.isfinite(fraction) else min(max(fraction, 0.0), 1.0)
    return (f"<div><svg viewBox='0 0 84 84' width='84' height='84' role='img' aria-label='{esc(label)} {esc(value)}'>"
            f"<circle cx='42' cy='42' r='{r}' fill='none' class='grid' stroke-width='8'/>"
            f"<circle cx='42' cy='42' r='{r}' fill='none' stroke='{color}' stroke-width='8' stroke-linecap='round' stroke-dasharray='{shown * c:.1f} {c:.1f}' transform='rotate(-90 42 42)'/>"
            f"<text x='42' y='48' text-anchor='middle' class='ink' style='font-size:18px;font-weight:650'>{esc(value)}</text></svg>"
            f"<div class='l'>{esc(label)}</div><div class='s'>{sub}</div></div>")


def _effect_note(effects: pd.DataFrame, driver: str) -> tuple[str, str]:
    """The pill class and a short note on how days like this have usually gone (time in range)."""
    row = effects.set_index("driver").loc[driver]
    text = reading(row)
    if text == "not enough data yet":
        return "info", text
    pill = "good" if row["lower_pp"] > 0 else "warn" if row["upper_pp"] < 0 else "info"
    return pill, f"days like this: {row['estimate_pp']:+.1f} points of time in range (90% range {row['lower_pp']:+.1f} to {row['upper_pp']:+.1f})"


def _hours(value: float) -> str:
    hours = int(value)
    return f"{hours} h {int(round((value - hours) * 60)):02d} m"


def why_today(day: pd.Series, yesterday: pd.Series | None, effects: pd.DataFrame) -> str:
    """The context of the day: sleep, recovery, exercise, late meal, site age, insulin and carbs against usual."""
    items = []

    def add(text: str, note: str, driver: str | None, active: bool) -> None:
        pill = ""
        if driver is not None and active:
            cls, effect = _effect_note(effects, driver)
            pill = f"<span class='pill {cls}'>{esc(DRIVER_LABELS[driver].lower())}</span>"
            note = f"{note}{'; ' if note else ''}{effect}"
        items.append(f"<li><span>{text}<span class='note'>{esc(note)}</span></span>{pill}</li>")

    if pd.notna(day.get("sleep_hours")):
        add(f"Slept {_hours(day['sleep_hours'])}", f"{day['sleep_device']}", "short_sleep", bool(day.get("short_sleep")))
    else:
        add("No sleep record for last night", "", None, False)
    if pd.notna(day.get("low_recovery")):
        values = []
        if yesterday is not None and pd.notna(yesterday.get("resting_hr")):
            values.append(f"resting heart rate {yesterday['resting_hr']:.0f}" + (f" ({yesterday['resting_hr_z']:+.1f} spread)" if pd.notna(yesterday.get("resting_hr_z")) else ""))
        if yesterday is not None and pd.notna(yesterday.get("hrv_ms")):
            values.append(f"HRV {yesterday['hrv_ms']:.0f} ms" + (f" ({yesterday['hrv_z']:+.1f} spread)" if pd.notna(yesterday.get("hrv_z")) else ""))
        add("Recovery yesterday", ", ".join(values) + " against your previous 28 days", "low_recovery", bool(day["low_recovery"]))
    if pd.notna(day.get("workout_1d")):
        if day["workout_1d"]:
            add("Workout yesterday", "", "workout_1d", True)
        elif day.get("workout_2d"):
            add("Workout two days ago", "", "workout_2d", True)
        else:
            add("No workout in the last two days", "", None, False)
    if day.get("late_meal"):
        add("Late meal last night", f"carbs entered at or after {LATE_MEAL_MINUTE // 60:02d}:{LATE_MEAL_MINUTE % 60:02d}", "late_meal", True)
    if pd.notna(day.get("site_age_days")):
        add(f"Infusion site day {int(day['site_age_days']) + 1}", "", "old_site", bool(day.get("old_site")))
    if pd.notna(day.get("insulin_vs_usual_pct")):
        add(f"Insulin {fmt(day['insulin_units'], 0)} U ({day['insulin_vs_usual_pct']:+.0f}% vs usual)",
            f"carbs {fmt(day['carb_grams'], 0)} g ({day['carbs_vs_usual_pct']:+.0f}% vs usual); usual is the median of the previous 28 days", None, False)
    return f"<ul class='why'>{''.join(items)}</ul>"


def rings(day: pd.Series, yesterday: pd.Series | None) -> str:
    """Time in range and insulin for the day; recovery for the day before, the same day the low-recovery driver reads."""
    tir = day["time_in_range_pct"]
    insulin = day.get("insulin_vs_usual_pct", np.nan)
    recovery = yesterday.get("recovery_score", np.nan) if yesterday is not None else np.nan
    tir_color = READOUT_STYLE["range"].color if tir >= TIME_IN_RANGE_TARGET_PERCENT else READOUT_STYLE["below_target"].color
    return "<div class='rings'>" + "".join([
        _ring(f"{tir:.0f}%", tir / 100, "Time in range", f"below 70: {day['time_below_70_pct']:.1f}%", tir_color),
        _ring("n/a" if pd.isna(insulin) else f"{insulin:+.0f}%", abs(insulin) / 50 if pd.notna(insulin) else 0.0, "Insulin vs usual", "against the previous 28 days", READOUT_STYLE["workout"].color),
        _ring("n/a" if pd.isna(recovery) else f"{recovery:.0f}", recovery / 100 if pd.notna(recovery) else 0.0, "Recovery yesterday", "50 is a usual day", READOUT_STYLE["sleep"].color),
    ]) + "</div>"


def week_bars(week: pd.DataFrame) -> str:
    """Daily time in range for the last seven days, with the 70% target as a dashed line."""
    width, height, left, top, bottom = CHART_WIDTH, 170, 36, 22, 146
    step = (width - left - 4) / max(len(week), 1)
    y = lambda value: bottom - value / 100 * (bottom - top)
    parts = []
    for tick in (0, 50, 100):
        parts.append(f"<line class='grid' x1='{left}' x2='{width - 4}' y1='{y(tick):.1f}' y2='{y(tick):.1f}'/>")
        parts.append(_text(left - 5, y(tick) + 4, f"{tick}%", anchor="end"))
    for i, (date, row) in enumerate(week.iterrows()):
        x = left + i * step + step * 0.15
        value = row["time_in_range_pct"]
        parts.append(f"<rect x='{x:.1f}' y='{y(value):.1f}' width='{step * 0.7:.1f}' height='{bottom - y(value):.1f}' rx='3' fill='{READOUT_STYLE['estimate'].color}'><title>{date:%a %d %b}: {value:.0f}%</title></rect>")
        parts.append(_text(x + step * 0.35, y(value) - 5, f"{value:.0f}", anchor="middle", ink=True, size=12))
        parts.append(_text(x + step * 0.35, bottom + 17, f"{date:%a}", anchor="middle"))
    parts.append(f"<line x1='{left}' x2='{width - 4}' y1='{y(70):.1f}' y2='{y(70):.1f}' stroke='{READOUT_STYLE['range'].color}' stroke-width='2' stroke-dasharray='7 4'/>")
    parts.append(_text(width - 4, top - 8, "dashed line: 70% target", anchor="end"))
    return f"<svg viewBox='0 0 {width} {height}' width='100%' role='img' aria-label='Daily time in range this week'>{''.join(parts)}</svg>"


def effects_chart(effects: pd.DataFrame, *, unit_label: str) -> str:
    """One row per driver: the average difference in percentage points with its 90% interval; zero is a solid line."""
    rows = effects.set_index("driver").reindex(list(DRIVER_COLUMNS))
    finite = rows[["lower_pp", "upper_pp"]].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    reach = max(1.0, float(np.abs(finite).max()) if finite.size else 1.0)
    reach = math.ceil(reach * 1.15 * 2) / 2
    width, row_h, top = CHART_WIDTH, 46, 6
    x0, x1 = 14, width - 14
    sx = lambda value: x0 + (value + reach) / (2 * reach) * (x1 - x0)
    height = top + row_h * len(rows) + 40
    color = READOUT_STYLE["estimate"].color
    parts = []
    for i, (driver, row) in enumerate(rows.iterrows()):
        label_y = top + row_h * i + 14
        cy = label_y + 17
        parts.append(f"<line x1='{sx(0):.1f}' x2='{sx(0):.1f}' y1='{cy - 10:.1f}' y2='{cy + 10:.1f}' stroke='var(--muted)' stroke-width='1.5'/>")
        parts.append(_text(x0, label_y, f"{DRIVER_LABELS[driver]} ({int(row['days_with'])} days)", ink=True, size=14))
        if not row["estimated"] or pd.isna(row["estimate_pp"]):
            parts.append(_text(sx(0) + 8, cy + 4, "not enough data yet", size=12))
            continue
        parts.append(f"<line x1='{sx(row['lower_pp']):.1f}' x2='{sx(row['upper_pp']):.1f}' y1='{cy:.1f}' y2='{cy:.1f}' stroke='{color}' stroke-width='3' stroke-linecap='round'/>")
        parts.append(f"<circle cx='{sx(row['estimate_pp']):.1f}' cy='{cy:.1f}' r='5.5' fill='{color}'><title>{esc(DRIVER_LABELS[driver])}: {row['estimate_pp']:+.2f} ({row['lower_pp']:+.2f} to {row['upper_pp']:+.2f})</title></circle>")
    axis_y = top + row_h * len(rows) + 6
    for tick in sorted({-reach, -reach / 2, 0.0, reach / 2, reach}):
        parts.append(_text(sx(tick), axis_y + 12, f"{tick:+g}" if tick else "0", anchor="middle"))
    parts.append(_text((x0 + x1) / 2, axis_y + 30, unit_label, anchor="middle", size=11))
    return f"<svg viewBox='0 0 {width} {height}' width='100%' role='img' aria-label='{esc(unit_label)}'>{''.join(parts)}</svg>"


def settings_card(settings: pd.DataFrame) -> str:
    blocks = []
    for part, group in settings.groupby("part", sort=False):
        first = group.iloc[0]
        lines = [f"<div class='seg'><div class='row'><b>{esc(part)}</b><span class='muted'>time in range {first['tir_14d']:.0f}% (14 d), {first['tir_90d']:.0f}% (90 d)</span></div>"]
        for row in group.itertuples(index=False):
            lines.append(f"<div class='row'><span>{esc(row.setting)} {row.now:g}</span><span class='pill {STATUS_CLASS[row.status]}'>{esc(row.status)}</span></div>")
            if row.suggestion:
                lines.append(f"<p class='sug'>{esc(row.suggestion)}</p>")
            lines.append(f"<p class='ev'>{esc(row.evidence)}</p>")
        blocks.append("".join(lines) + "</div>")
    return "".join(blocks)


def changes_list(changes: pd.DataFrame) -> str:
    if changes.empty:
        return "<p class='muted'>No setting changes on the running profile in the last 90 days.</p>"
    items = []
    for row in changes.sort_values("changed", ascending=False).itertuples(index=False):
        if row.status == "evaluated":
            result = f"time in range {row.tir_before:.0f}% to {row.tir_after:.0f}% ({row.tir_difference:+.0f} points, 90% range {row.tir_difference_lower_90:+.0f} to {row.tir_difference_upper_90:+.0f}); lows {row.low_limits_after}"
        else:
            result = f"{row.status}: {row.reason}"
        items.append(f"<li><b>{row.changed:%d %b}</b> {esc(row.setting)} {row.old_value:g} to {row.new_value:g}, {esc(row.hours)}<span class='note'>{esc(result)}</span></li>")
    return f"<ul class='why changes'>{''.join(items)}</ul>"


def day_trace(date: pd.Timestamp, readout: Readout) -> str:
    """One day's glucose with the target range, sleep and workouts as bands, carbs as triangles and site changes as dashed lines."""
    width, height, left, right, top, bottom = CHART_WIDTH, 196, 30, 4, 8, 160
    start, end = date, date + pd.Timedelta(days=1)
    sx = lambda t: left + (pd.Timestamp(t) - start) / pd.Timedelta(hours=24) * (width - left - right)
    sy = lambda g: bottom - (min(max(g, TRACE_LOW), TRACE_HIGH) - TRACE_LOW) / (TRACE_HIGH - TRACE_LOW) * (bottom - top)
    parts = []
    for name, frame, a, b in (("sleep", readout.sleep, "sleep_start", "sleep_end"), ("workout", readout.workouts, "start_timestamp", "end_timestamp")):
        for row in frame[(frame[a] < end) & (frame[b] > start)].itertuples(index=False):
            x_a, x_b = sx(max(getattr(row, a), start)), sx(min(getattr(row, b), end))
            parts.append(f"<rect x='{x_a:.1f}' y='{top}' width='{max(x_b - x_a, 1.5):.1f}' height='{bottom - top}' fill='{READOUT_STYLE[name].color}' opacity='0.22'/>")
    parts.append(f"<rect x='{left}' y='{sy(180):.1f}' width='{width - left - right}' height='{sy(70) - sy(180):.1f}' fill='{READOUT_STYLE['range'].color}' opacity='0.10'/>")
    for value in (70, 180, 250):
        parts.append(f"<line class='grid' x1='{left}' x2='{width - right}' y1='{sy(value):.1f}' y2='{sy(value):.1f}'/>")
        parts.append(_text(left - 5, sy(value) + 4, f"{value}", anchor="end", size=11))
    for fill in readout.site_fills[(readout.site_fills >= start) & (readout.site_fills < end)]:
        parts.append(f"<line x1='{sx(fill):.1f}' x2='{sx(fill):.1f}' y1='{top}' y2='{bottom}' stroke='{READOUT_STYLE['site'].color}' stroke-width='2' stroke-dasharray='6 4'/>")
        parts.append(_text(sx(fill) + 4, top + 12, "site", size=11))
    trace = readout.cgm[(readout.cgm["timestamp"] >= start) & (readout.cgm["timestamp"] < end)].sort_values("timestamp")
    segments, current, last = [], [], None
    for t, g in zip(trace["timestamp"], trace["glucose"]):
        if last is not None and (t - last) > pd.Timedelta(minutes=20):
            segments.append(current)
            current = []
        current.append(f"{sx(t):.1f},{sy(g):.1f}")
        last = t
    segments.append(current)
    for points in segments:
        if len(points) > 1:
            parts.append(f"<polyline fill='none' stroke='{READOUT_STYLE['glucose'].color}' class='ink' stroke-width='2' points='{' '.join(points)}' style='stroke:var(--ink)'/>")
    for row in readout.carbs[(readout.carbs["timestamp"] >= start) & (readout.carbs["timestamp"] < end)].itertuples(index=False):
        x = sx(row.timestamp)
        parts.append(f"<path d='M{x - 5:.1f},{bottom + 12} L{x + 5:.1f},{bottom + 12} L{x:.1f},{bottom + 3} Z' fill='{READOUT_STYLE['carbs'].color}'><title>{row.carb_grams:.0f} g</title></path>")
        parts.append(_text(x, bottom + 25, f"{row.carb_grams:.0f}", anchor="middle", size=11))
    for hour in (0, 6, 12, 18, 24):
        parts.append(_text(left + hour / 24 * (width - left - right), height - 2, f"{hour:02d}:00" if hour < 24 else "", anchor="middle", size=11))
    return f"<svg viewBox='0 0 {width} {height}' width='100%' role='img' aria-label='Glucose on {date:%d %b}'>{''.join(parts)}</svg>"


def day_notes(date: pd.Timestamp, readout: Readout) -> str:
    day = readout.days.loc[date]
    trace = readout.cgm[(readout.cgm["timestamp"] >= date) & (readout.cgm["timestamp"] < date + pd.Timedelta(days=1))]
    notes = [f"Time in range {day['time_in_range_pct']:.0f}%, below 70 {day['time_below_70_pct']:.1f}%, mean {day['mean_glucose_mgdl']:.0f} mg/dL"]
    if not trace.empty:
        peak = trace.loc[trace["glucose"].idxmax()]
        notes.append(f"Highest {peak['glucose']:.0f} mg/dL at {peak['timestamp']:%H:%M}")
    notes.append(f"Insulin {day['insulin_units']:.1f} U (bolus {day['bolus_units']:.1f}, basal {day['basal_units']:.1f}); carbs {day['carb_grams']:.0f} g")
    return f"<p class='muted'>{'<br>'.join(esc(note) for note in notes)}</p>"


def legend() -> str:
    keys = ("sleep", "workout", "range", "carbs", "site")
    return "<div class='legend'>" + "".join(f"<span><span class='sw' style='background:{READOUT_STYLE[k].color}'></span>{esc(READOUT_STYLE[k].label)}</span>" for k in keys) + "</div>"


def render(readout: Readout) -> str:
    days = readout.days
    today = days.loc[readout.as_of]
    previous = readout.as_of - pd.Timedelta(days=1)
    yesterday = days.loc[previous] if previous in days.index else None
    week = days.loc[readout.as_of - pd.Timedelta(days=TRACE_DAYS - 1): readout.as_of]
    week = week[week["counted"]]
    week_tir = 100 * week["in_range_readings"].sum() / week["readings"].sum()
    week_low = 100 * week["below_70_readings"].sum() / week["readings"].sum()
    lo, median, hi = readout.usual_week
    tir_effects = readout.effects["time_in_range"]
    week_counts = "; ".join(f"{DRIVER_LABELS[c].lower()}: {int(week[c].fillna(False).astype(bool).sum())} of {len(week)} days" for c in DRIVER_COLUMNS if week[c].fillna(False).astype(bool).any())
    definitions = "".join(f"<li><b>{esc(label)}</b>: {esc(text)}</li>" for _, label, text in DRIVERS)
    usable = all(readout.usable.values())
    body = [
        f"<p class='sub'>{readout.as_of:%A %d %B %Y} &middot; the latest full day of data</p>",
        f"<div class='card'>{rings(today, yesterday)}{why_today(today, yesterday, tir_effects)}</div>",
        "<h2>This week</h2>",
        f"<div class='card'><div class='big'>{week_tir:.0f}% in range</div><p class='muted'>{len(week)} full days ending {readout.as_of:%d %b}; below 70: {week_low:.1f}%. "
        f"Your usual week: {median:.0f}% (most weeks {lo:.0f}% to {hi:.0f}%, the middle 80% of 7-day stretches in the 90 days before).</p>"
        f"{week_bars(week)}" + (f"<p class='muted'>This week: {esc(week_counts)}.</p>" if week_counts else "") + "</div>",
        "<h2>What has gone with your glucose</h2>",
        "<div class='card'><p class='muted'>Each dot is how many percentage points of the day's time in range differ on days after this, compared with similar days of the same month "
        "without it; the line is the 90% range. A line crossing zero means no clear link. Days in brackets are how many modelled days had it.</p>"
        + effects_chart(tir_effects, unit_label="time in range, percentage points of the day")
        + "<p class='muted'>Time below 70 on the same days:</p>"
        + effects_chart(readout.effects["time_below_70"], unit_label="time below 70, percentage points of the day")
        + f"<p class='caption'>From {readout.model_days} days with every driver measured.{'' if usable else ' The model fit did not pass its sampler checks; treat these as unreliable.'}</p>"
        + f"<details><summary>What each driver means</summary><ul>{definitions}</ul></details></div>",
        "<h2>Settings check</h2>",
        "<div class='card'><p class='muted'>Basal: what Control-IQ delivered against the programmed rate in hours without meals, boluses or exercise. "
        "I:C: how clean first meals of the day behaved against the ratio set now. Suggestions are one 10% or 20% step, for discussion with your endocrinologist; nothing adds insulin where lows are over a limit.</p>"
        + settings_card(readout.settings) + "</div>",
        "<h2>Recent setting changes</h2>",
        f"<div class='card'>{changes_list(readout.changes)}</div>",
        "<h2>The last seven days</h2>",
        legend(),
    ]
    for i, date in enumerate(sorted(week.index, reverse=True)):
        body.append(f"<details{' open' if i == 0 else ''}><summary>{date:%a %d %b} &middot; {days.loc[date, 'time_in_range_pct']:.0f}% in range</summary>{day_trace(date, readout)}{day_notes(date, readout)}</details>")
    body.append(f"<p class='caption'>Run {esc(readout.run_id)}, built {esc(readout.built)}.</p>")
    return "\n".join([
        "<!doctype html>", "<html lang='en'>", "<head>", "<meta charset='utf-8' />", "<meta name='viewport' content='width=device-width, initial-scale=1' />",
        "<title>Glucose readout</title>", f"<style>{CSS}{EXTRA_CSS}</style>", "</head>", "<body><main>", "<h1>Your glucose readout</h1>", *body, "</main></body>", "</html>",
    ])
