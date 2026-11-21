"""Building blocks for the answer-first HTML review pages.

Every review page follows the same order: the question, a verdict card that answers it
in one sentence, a few headline numbers, "how much to trust this", "what to do next",
then fold-out details (figures, tables, files, glossary). Charts are inline SVG so a
page is one self-contained file that opens offline.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

# One color per role, used the same way on every page (JCO palette).
PRIMARY = "#0073C2"  # the estimate or the main series
SECONDARY = "#7AA6DC"  # weaker-evidence points of the same series
COMPARATOR = "#EFC000"  # a comparison series (for example the persistence baseline)
REFERENCE = "#868686"  # pump settings, targets, thresholds

STATUS_LABELS = {
    "good": "&#10003;",
    "warn": "!",
    "bad": "&#10007;",
    "info": "i",
}

# Plain-language versions of internal codes that appear in reports.
PLAIN_LABELS = {
    # data completeness
    "missing_kind": "a data type is missing for this window",
    "starts_late": "data starts later than requested",
    "ends_early": "data ends earlier than requested",
    "internal_gap": "gap inside the window",
    "duplicates": "duplicate rows",
    "overlap": "overlaps another window",
    # forecast recommendation policy
    "skipped_by_flag": "recommendations were turned off for this run",
    "walk_forward_missing": "no held-out validation was run",
    "walk_forward_insufficient_folds": "fewer than 2 validation periods",
    "walk_forward_small_test_fold": "a validation period had fewer than 200 test points",
    "coverage_out_of_range": "the 80% forecast range did not contain the real value 70 to 90% of the time",
    "mae_not_meaningfully_better_than_persistence": "the model is not at least 10% better than assuming glucose stays the same",
    "data_quality_missing": "data completeness was not assessed",
    "data_incomplete": "some Tandem data windows are incomplete",
    "normalized_plausibility_unknown": "plausibility of the cleaned data was not checked",
    "normalized_plausibility_failed": "the cleaned data failed plausibility checks",
    "sampler_chains_lt_2": "the model fit used fewer than 2 sampling chains",
    # overnight basal
    "no_research_frame": "no analysis data were available",
    "no_overnight_rows": "no overnight data were available",
    "limited_clean_overnight_windows": "only a few clean overnight stretches",
    "insufficient_clean_overnight_windows": "too few clean overnight stretches",
    "recent_meal_or_proxy": "a meal in the previous 2 hours",
    "recent_bolus_or_iob": "a bolus or insulin on board in the previous 2 hours",
    "recent_exercise": "recent exercise",
    "closed_loop_or_transition": "Control-IQ was adjusting insulin",
    "missing_cgm": "CGM reading missing",
    # first-meal windows
    "closed_loop_confounding_premeal": "Control-IQ was adjusting insulin before the meal",
    "high_premeal_iob": "a lot of insulin on board before the meal",
    "missing_cgm_first_90m": "CGM missing in the first 90 minutes after eating",
    "poor_cgm_coverage": "CGM coverage under 90% after the meal",
    "postmeal_additional_bolus_120m": "another bolus within 2 hours of the meal",
    "premeal_bg_out_of_range": "glucose before the meal was outside the target range",
    "recent_bolus_4h": "a bolus in the 4 hours before the meal",
    "steep_premeal_slope": "glucose was rising or falling fast before the meal",
    "exercise": "exercise (a workout of 10 minutes or more, or exercise mode) during or in the 4 hours before the meal",
    "context_event_excluded": "inside an excluded period (for example after a steroid injection)",
    "no_candidates": "no first meals were found",
    "accepted_without_exclusions": "accepted with no exclusions",
    # analysis tiers
    "clean_reference": "passed every check",
    "controller_context_allowed": "Control-IQ active before the meal",
    "baseline_drift_allowed": "started out of range or drifting",
    "workout_context_allowed": "recent exercise",
    "exploratory_cgm_quality_only": "several issues; CGM complete",
    # effective I:C
    "consistent_with_programmed": "consistent with the pump setting",
    "carbs_hit_harder_than_programmed": "carbs raise glucose more than the setting assumes",
    "carbs_hit_softer_than_programmed": "carbs raise glucose less than the setting assumes",
    "insufficient_logged_meals": "too few breakfasts with logged carbs",
    # data quality statuses and plausibility checks
    "good": "good",
    "degraded": "some problems",
    "broken": "incomplete",
    "failed": "failed one or more checks",
    "unknown": "not checked",
    "out_of_window_rows_rejected": "rows outside the requested dates (dropped)",
    "cgm_gap_ge_120m_rows": "a CGM gap of 2 hours or more",
    "invalid_cgm_rows": "a CGM value outside 40 to 400 mg/dL",
    "impossible_cgm_rows": "a CGM value of 0 or below, or above 1000 mg/dL",
    "negative_bolus_rows": "a negative bolus",
    "implausible_5min_bolus_rows": "more than 25 U of bolus in 5 minutes",
    "implausible_daily_bolus_days": "more than 150 U of bolus in a day (days)",
    "negative_iob_rows": "negative insulin on board",
    "implausible_iob_rows": "more than 50 U of insulin on board",
    "prepared_frame_empty": "no rows in the 5-minute dataset",
}


def plain(code: object) -> str:
    """Return the plain-language label for an internal code, or a readable fallback."""
    text = str(code or "").strip()
    if not text:
        return "none"
    return PLAIN_LABELS.get(text, text.replace("_", " "))


@dataclass(frozen=True)
class Verdict:
    """A one-sentence answer shared by the HTML page and the markdown summary."""

    status: str  # good | warn | bad | info
    label: str
    headline: str  # plain text
    explanation: str = ""


SAMPLER_RHAT_MAX = 1.01
SAMPLER_ESS_PER_CHAIN = 100  # Vehtari et al. (2021): bulk ESS of at least 100 per chain


def icr_verdict(segment: str, effective: float, lower: float, upper: float, programmed: float, status_code: str) -> Verdict:
    """Effective I:C from logged meals against the programmed ratio."""
    difference = (programmed / effective - 1) * 100
    direction = "harder" if difference > 0 else "softer"
    headline = f"Your {segment} meals behave like an I:C of {effective:.1f} g per unit (90% range {lower:.1f} to {upper:.1f}). Your pump is set to {programmed:.1f}."
    if status_code == "consistent_with_programmed":
        return Verdict(
            "good",
            "Consistent with your pump setting",
            headline,
            "The pump setting sits inside the range, so these meals give no evidence that it needs to change. "
            f"The point estimate has carbs hitting about {abs(difference):.0f}% {direction} than programmed, which is within the noise.",
        )
    return Verdict(
        "warn",
        plain(status_code).capitalize(),
        headline,
        f"The pump setting is outside the range: carbs hit about {abs(difference):.0f}% {direction} than programmed. "
        "This is research evidence to discuss with your clinician, not an instruction to change the pump.",
    )


def esc(value: object) -> str:
    return html.escape(str(value))


def fmt(value: object, digits: int = 1, *, suffix: str = "") -> str:
    """Format a number for display; missing values become an en dash."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "&ndash;"
    if not math.isfinite(number):
        return "&ndash;"
    if digits == 0:
        return f"{number:,.0f}{suffix}"
    return f"{number:,.{digits}f}{suffix}"


CSS = """
:root { --ink:#1f2933; --muted:#5f6b7a; --line:#e3e7ec; --bg:#f7f8fa; --card:#fff;
  --good:#20854E; --goodbg:#e8f4ee; --warn:#8F6A00; --warnbg:#fdf6e0; --bad:#A73030; --badbg:#fbeaea;
  --info:#003C67; --infobg:#e6eef6; --grid:#e3e7ec; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --ink:#e6e9ee; --muted:#b4bdc9; --line:#2c333d; --bg:#14181d; --card:#1b2027; --grid:#4a5361;
  --good:#6fcf97; --goodbg:#183326; --warn:#e9c46a; --warnbg:#332a12; --bad:#f08a8a; --badbg:#3a1c1c;
  --info:#9cc3e8; --infobg:#17283a; } }
:root[data-theme="dark"] {
  --ink:#e6e9ee; --muted:#b4bdc9; --line:#2c333d; --bg:#14181d; --card:#1b2027; --grid:#4a5361;
  --good:#6fcf97; --goodbg:#183326; --warn:#e9c46a; --warnbg:#332a12; --bad:#f08a8a; --badbg:#3a1c1c;
  --info:#9cc3e8; --infobg:#17283a; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.55 -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 860px; margin: 0 auto; padding: 28px 16px 64px; }
h1 { font-size: 26px; line-height: 1.25; margin: 0 0 4px; } h2 { font-size: 19px; margin: 32px 0 8px; }
.sub { color: var(--muted); margin: 0 0 20px; font-size: 14px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 18px 20px; margin: 14px 0; }
.verdict .headline { font-size: 19px; margin: 12px 0 6px; }
.verdict .explain { margin: 0; color: var(--muted); }
.caption { color: var(--muted); font-size: 13.5px; margin: 8px 0 0; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
.tile { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }
.tile .n { font-size: 26px; font-weight: 650; } .tile .l { color: var(--muted); font-size: 13.5px; }
.status { display:inline-flex; gap:6px; align-items:center; font-weight:600; font-size:14px; padding:3px 10px; border-radius: 999px; }
.status.good { background: var(--goodbg); color: var(--good); } .status.warn { background: var(--warnbg); color: var(--warn); }
.status.bad { background: var(--badbg); color: var(--bad); } .status.info { background: var(--infobg); color: var(--info); }
details { background: var(--card); border:1px solid var(--line); border-radius:10px; padding: 10px 16px; margin: 10px 0; }
summary { cursor: pointer; font-weight: 600; }
details[open] summary { margin-bottom: 8px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 600; } .num { text-align: right; font-variant-numeric: tabular-nums; }
.legend { display:flex; gap:16px; flex-wrap: wrap; font-size: 13px; color: var(--muted); margin-top: 6px; }
.sw { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align: middle; }
.sw.line { width: 18px; height: 0; border-top: 2px dashed; border-radius: 0; }
a { color: var(--info); }
svg text { fill: var(--muted); font-size: 12px; } svg text.ink { fill: var(--ink); } svg .grid { stroke: var(--grid); }
svg .halo { stroke: var(--card); } svg .hollow { fill: var(--card); }
td:first-child { white-space: nowrap; }
.sw.ring { background: transparent; border: 2px solid; width: 11px; height: 11px; }
"""


def page(*, title: str, question: str, subtitle: str, body: Sequence[str]) -> str:
    """Wrap the page body in a self-contained HTML document."""
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='utf-8' />",
            "<meta name='viewport' content='width=device-width, initial-scale=1' />",
            f"<title>{esc(title)}</title>",
            f"<style>{CSS}</style>",
            "</head>",
            "<body><main>",
            f"<h1>{esc(question)}</h1>",
            f"<p class='sub'>{subtitle}</p>",
            *[part for part in body if part],
            "</main></body>",
            "</html>",
        ]
    )


def verdict(status: str, label: str, headline: str, explanation: str = "", figure: str = "") -> str:
    """The answer card. ``headline`` and ``explanation`` are trusted HTML built by the caller."""
    icon = STATUS_LABELS.get(status, "i")
    return (
        f"<div class='card verdict'><span class='status {esc(status)}'>{icon} {esc(label)}</span>"
        f"<p class='headline'>{headline}</p>"
        + (f"<p class='explain'>{explanation}</p>" if explanation else "")
        + (f"<div style='margin-top:12px'>{figure}</div>" if figure else "")
        + "</div>"
    )


def details(summary: str, body: str, *, open_: bool = False) -> str:
    return f"<details{' open' if open_ else ''}><summary>{esc(summary)}</summary>{body}</details>"


def table(headers: Sequence[str], rows: Iterable[Sequence[object]], *, numeric: Sequence[int] = ()) -> str:
    """Simple table. Cells are trusted HTML; escape untrusted values with ``esc`` first."""
    head = "".join(f"<th class='num'>{esc(h)}</th>" if i in numeric else f"<th>{esc(h)}</th>" for i, h in enumerate(headers))
    body_rows = []
    for row in rows:
        cells = "".join(f"<td class='num'>{cell}</td>" if i in numeric else f"<td>{cell}</td>" for i, cell in enumerate(row))
        body_rows.append(f"<tr>{cells}</tr>")
    if not body_rows:
        return "<p class='caption'>No rows.</p>"
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"


def legend(items: Iterable[tuple[str, str, object]]) -> str:
    """Legend entries of (color, label, kind) where kind is False (filled dot), True (dashed line) or "ring" (hollow dot)."""
    spans = []
    for color, label, kind in items:
        if kind == "ring":
            spans.append(f"<span><span class='sw ring' style='border-color:{color}'></span>{esc(label)}</span>")
        elif kind:
            spans.append(f"<span><span class='sw line' style='border-color:{color}'></span>{esc(label)}</span>")
        else:
            spans.append(f"<span><span class='sw' style='background:{color}'></span>{esc(label)}</span>")
    return f"<div class='legend'>{''.join(spans)}</div>"


# ---------------------------------------------------------------------------
# SVG charts. Coordinates are in viewBox units; width scales to the container.
# ---------------------------------------------------------------------------


def _text(x: float, y: float, value: str, *, anchor: str = "start", ink: bool = False, size: int | None = None, weight: int | None = None, rotate: bool = False) -> str:
    attrs = f"x='{x:.1f}' y='{y:.1f}' text-anchor='{anchor}'"
    if ink:
        attrs += " class='ink'"
    style = []
    if size:
        style.append(f"font-size:{size}px")
    if weight:
        style.append(f"font-weight:{weight}")
    if style:
        attrs += f" style='{';'.join(style)}'"
    if rotate:
        attrs += f" transform='rotate(-90 {x:.1f} {y:.1f})'"
    return f"<text {attrs}>{esc(value)}</text>"


