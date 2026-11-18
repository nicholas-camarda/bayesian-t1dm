"""The outcome baseline page: where time in range stands, and which hours lows and highs concentrate in."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis_data import AnalysisData
from .outcomes import (
    LOW_LIMIT_PERCENT,
    VERY_LOW_LIMIT_PERCENT,
    baseline_table,
    day_part_table,
    daily_metrics,
    passes_low_limits,
)


def _fmt(value: float, digits: int = 1) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _status(row: pd.Series) -> str:
    return "within limits" if passes_low_limits(row) else "over a low limit"


def outcomes_page(baseline: pd.DataFrame, parts: pd.DataFrame, *, as_of: pd.Timestamp, source: str) -> str:
    def table(frame: pd.DataFrame, first: str, label_col: str) -> str:
        head = f"<tr><th>{first}</th><th>days counted</th><th>in range 70-180 (%)</th><th>below 70 (%)</th><th>below 54 (%)</th><th>above 180 (%)</th><th>above 250 (%)</th><th>mean (mg/dL)</th><th>CV (%)</th><th>GMI (%)</th><th>low limits</th></tr>"
        body = "".join(
            f"<tr><td>{html.escape(str(r[label_col]))}</td><td>{_fmt(r['days'], 0)} of {_fmt(r['days_in_window'], 0)}</td>"
            f"<td>{_fmt(r['time_in_range_pct'])}</td><td>{_fmt(r['time_below_70_pct'])}</td><td>{_fmt(r['time_below_54_pct'])}</td>"
            f"<td>{_fmt(r['time_above_180_pct'])}</td><td>{_fmt(r['time_above_250_pct'])}</td><td>{_fmt(r['mean_glucose_mgdl'], 0)}</td>"
            f"<td>{_fmt(r['cv_pct'])}</td><td>{_fmt(r['gmi_pct'], 2)}</td><td>{_status(r)}</td></tr>"
            for _, r in frame.iterrows()
        )
        return f"<div class=\"scroll\"><table>{head}{body}</table></div>"

    worst_high = parts.sort_values("time_above_180_pct", ascending=False).iloc[0]
    worst_low = parts.sort_values("time_below_70_pct", ascending=False).iloc[0]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Glucose outcomes</title>
<style>:root{{--ink:#1f2933;--muted:#616e7c;--line:#d9dee3;--bg:#fff}}@media (prefers-color-scheme:dark){{:root{{--ink:#e4e7eb;--muted:#9aa5b1;--line:#3e4c59;--bg:#1b1f23}}}}
body{{font:15px/1.5 system-ui,sans-serif;color:var(--ink);background:var(--bg);max-width:1100px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem;font-size:13px}}th,td{{border-bottom:1px solid var(--line);padding:.35rem .5rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{color:var(--muted);font-weight:600}}
.scroll{{overflow-x:auto}}.note{{color:var(--muted)}}</style></head><body>
<h1>Glucose outcomes</h1>
<p class="note">CGM readings up to {as_of.date()} (exclusive). A day counts when it holds at least 70% of its readings. Consensus targets: time in range above 70%, below 70 under {LOW_LIMIT_PERCENT:.0f}%, below 54 under {VERY_LOW_LIMIT_PERCENT:.0f}%. Descriptive only.</p>
<h2>Whole day</h2>
{table(baseline, "window", "window")}
<h2>By time of day, last 90 days</h2>
<p class="note">Bins are the pump's profile segments ({html.escape(source)}). Highs concentrate most in {html.escape(str(worst_high['day_part']))} ({_fmt(worst_high['time_above_180_pct'])}% above 180); lows concentrate most in {html.escape(str(worst_low['day_part']))} ({_fmt(worst_low['time_below_70_pct'])}% below 70). A setting change should target the hours where the limits are met and time in range is lowest.</p>
{table(parts, "time of day", "day_part")}
<p class="note">How to read it: a row over a low limit is not a candidate for more insulin in those hours, whatever its time in range. A change is judged on these same rows, before versus after.</p>
</body></html>
"""


def write_outcomes(analysis: AnalysisData, output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cgm = analysis.cgm
    latest = pd.Timestamp(cgm["timestamp"].max())
    as_of = latest.normalize()  # the last full day ends here; the partial day is left out
    parts, source = analysis.day_parts()
    baseline = baseline_table(cgm, as_of=as_of)
    by_part = day_part_table(cgm, parts, as_of=as_of)
    daily = daily_metrics(cgm)
    paths = {
        "page": output / "outcomes_report.html",
        "baseline": output / "outcomes_baseline.csv",
        "day_parts": output / "outcomes_day_parts.csv",
        "daily": output / "outcomes_daily.csv",
    }
    baseline.to_csv(paths["baseline"], index=False)
    by_part.to_csv(paths["day_parts"], index=False)
    daily.to_csv(paths["daily"])
    paths["page"].write_text(outcomes_page(baseline, by_part, as_of=as_of, source=source), encoding="utf-8")
    return paths
