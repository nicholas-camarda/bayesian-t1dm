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
    "recent_workout_6h": "exercise in the 6 hours before the meal",
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


FORECAST_REQUIRED_IMPROVEMENT = 0.10
SAMPLER_RHAT_MAX = 1.01
SAMPLER_ESS_PER_CHAIN = 100  # Vehtari et al. (2021): bulk ESS of at least 100 per chain


def sampler_healthy(diag: dict) -> bool:
    """A PyMC fit is usable when R-hat is under 1.01, there are no divergences and bulk ESS is at least 100 per chain."""
    try:
        rhat = float(diag.get("rhat_max"))
        ess = float(diag.get("ess_bulk_min"))
        chains = max(int(diag.get("chains") or 1), 1)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(rhat)
        and rhat < SAMPLER_RHAT_MAX
        and int(diag.get("divergences") or 0) == 0
        and math.isfinite(ess)
        and ess >= SAMPLER_ESS_PER_CHAIN * chains
    )


def forecast_fits(summary: dict) -> list[dict]:
    """Fitting diagnostics for every validation period plus the final fit, when present."""
    folds = (summary.get("walk_forward") or {}).get("folds") or []
    fits = [fold.get("fit_diagnostics") or {} for fold in folds]
    if summary.get("fit_diagnostics"):
        fits.append(summary["fit_diagnostics"])
    return [diag for diag in fits if diag]


def forecast_verdict(model_mae: object, baseline_mae: object, *, failed_fits: int = 0, total_fits: int = 0) -> Verdict:
    """Compare walk-forward model error with the persistence baseline (same 10% bar as the recommendation policy).

    A fit that failed its sampler checks overrides the comparison: its error numbers are not trustworthy either way.
    """
    try:
        model, baseline = float(model_mae), float(baseline_mae)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return Verdict("info", "Not assessed", "No validation results were produced.")
    if not (math.isfinite(model) and math.isfinite(baseline)) or baseline <= 0:
        return Verdict("info", "Not assessed", "No validation results were produced.")
    ratio = model / baseline
    numbers = f"{model:,.1f} mg/dL on average, against {baseline:,.1f} mg/dL for assuming glucose stays the same"
    if failed_fits:
        return Verdict(
            "bad",
            "Model fitting failed",
            f"The model's 30-minute forecasts are off by {numbers}.",
            f"The fitting procedure failed its checks in {failed_fits} of {total_fits} fits, so these numbers are not trustworthy and the model should not be used.",
        )
    if ratio <= 1 - FORECAST_REQUIRED_IMPROVEMENT:
        return Verdict("good", "Beats the simple baseline", f"The model's 30-minute forecasts are off by {numbers}.", f"That is {(1 - ratio) * 100:.0f}% better than the baseline on data the model never saw.")
    if ratio < 1:
        return Verdict("warn", "Barely better than the baseline", f"The model's 30-minute forecasts are off by {numbers}.", f"It needs to be at least {FORECAST_REQUIRED_IMPROVEMENT:.0%} better before its forecasts add anything.")
    return Verdict("bad", "Worse than the simple baseline", f"The model's 30-minute forecasts are off by {numbers}.", "Its forecasts should not be used for anything until this is fixed.")


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


def tiles(items: Iterable[tuple[str, str]]) -> str:
    cells = "".join(f"<div class='tile'><div class='n'>{value}</div><div class='l'>{label}</div></div>" for value, label in items)
    return f"<div class='tiles'>{cells}</div>"


def section(title: str, body: str) -> str:
    return f"<h2>{esc(title)}</h2>{body}"


def bullets(items: Iterable[str]) -> str:
    rendered = "".join(f"<li>{item}</li>" for item in items if item)
    return f"<ul>{rendered}</ul>" if rendered else ""


def details(summary: str, body: str, *, open_: bool = False) -> str:
    return f"<details{' open' if open_ else ''}><summary>{esc(summary)}</summary>{body}</details>"


def caption(text: str) -> str:
    return f"<p class='caption'>{text}</p>"


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


def glossary(pairs: Iterable[tuple[str, str]]) -> str:
    return details("Glossary", table(["term", "meaning"], [(f"<b>{esc(term)}</b>", meaning) for term, meaning in pairs]))


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


def file_links(entries: Iterable[tuple[str, str, bool]]) -> str:
    """List of (filename, description, exists) linking to companion files."""
    rows = []
    for name, description, exists in entries:
        link = f"<a href='{esc(name)}'>{esc(name)}</a>" if exists else f"{esc(name)} (not written)"
        rows.append((link, esc(description)))
    return table(["file", "what it holds"], rows)


# ---------------------------------------------------------------------------
# SVG charts. Coordinates are in viewBox units; width scales to the container.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Scale:
    lo: float
    hi: float
    a: float
    b: float

    def __call__(self, value: float) -> float:
        if self.hi == self.lo:
            return (self.a + self.b) / 2
        return self.a + (value - self.lo) / (self.hi - self.lo) * (self.b - self.a)


def _nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo]
    raw = (hi - lo) / max(count, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = min((m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw), default=raw)
    start = math.ceil(lo / step) * step
    ticks = []
    value = start
    while value <= hi + 1e-9:
        ticks.append(round(value, 10))
        value += step
    return ticks


def _tick_label(value: float) -> str:
    return f"{value:g}"


def _svg(width: int, height: int, label: str, parts: Iterable[str]) -> str:
    return (
        f"<svg viewBox='0 0 {width} {height}' width='100%' role='img' aria-label='{esc(label)}' "
        f"style='max-width:{width}px;display:block'>{''.join(parts)}</svg>"
    )


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


def interval_chart(
    *,
    estimate: float,
    lower: float,
    upper: float,
    reference: float | None,
    reference_label: str,
    axis_label: str,
    value_label: str,
) -> str:
    """One estimate with its 90% range on a horizontal axis, against a reference value."""
    values = [estimate, lower, upper] + ([reference] if reference is not None else [])
    span = max(values) - min(values)
    lo = math.floor(min(values) - max(span * 0.4, 0.5))
    hi = math.ceil(max(values) + max(span * 0.4, 0.5))
    width, height, x0, x1 = 560, 132, 30, 540
    sx = _Scale(lo, hi, x0, x1)
    band_y, axis_y = 46, 96
    parts: list[str] = [f"<line class='grid' x1='{x0}' x2='{x1}' y1='{axis_y}' y2='{axis_y}' />"]
    for tick in _nice_ticks(lo, hi, 5):
        parts.append(f"<line class='grid' x1='{sx(tick):.1f}' x2='{sx(tick):.1f}' y1='{axis_y}' y2='{axis_y + 5}' />")
        parts.append(_text(sx(tick), axis_y + 19, _tick_label(tick), anchor="middle"))
    parts.append(_text((x0 + x1) / 2, height - 2, axis_label, anchor="middle", size=11))
    # 90% range band and estimate; nothing is drawn across the band.
    parts.append(
        f"<rect x='{sx(lower):.1f}' y='{band_y - 8}' width='{max(sx(upper) - sx(lower), 2):.1f}' height='16' rx='5' fill='{PRIMARY}' opacity='0.35'>"
        f"<title>90% range {esc(value_label)}</title></rect>"
    )
    parts.append(f"<line x1='{sx(lower):.1f}' x2='{sx(upper):.1f}' y1='{band_y}' y2='{band_y}' stroke='{PRIMARY}' stroke-width='3' />")
    parts.append(f"<circle cx='{sx(estimate):.1f}' cy='{band_y}' r='8' fill='{PRIMARY}' class='halo' stroke-width='2.5'><title>{esc(value_label)}</title></circle>")
    parts.append(_text(sx(estimate), band_y - 16, value_label, anchor="middle", ink=True, weight=650, size=14))
    # Reference (pump setting): a dashed stub between the band and the axis, labelled beside it.
    if reference is not None:
        rx = sx(reference)
        parts.append(f"<line x1='{rx:.1f}' x2='{rx:.1f}' y1='{band_y + 14}' y2='{axis_y}' stroke='{REFERENCE}' stroke-width='2.5' stroke-dasharray='6 4' />")
        parts.append(f"<path d='M{rx - 6:.1f},{axis_y - 1} L{rx + 6:.1f},{axis_y - 1} L{rx:.1f},{axis_y - 9} Z' fill='{REFERENCE}' />")
        anchor = "end" if rx > (x0 + x1) / 2 else "start"
        parts.append(_text(rx + (-8 if anchor == "end" else 8), band_y + 34, reference_label, anchor=anchor, ink=True, size=13))
    return _svg(width, height, f"{value_label}; {reference_label}", parts)


@dataclass(frozen=True)
class ScatterPoint:
    x: float
    y: float
    lower: float | None
    upper: float | None
    strong: bool
    tooltip: str


def scatter_with_ranges(points: Sequence[ScatterPoint], *, x_label: str, y_label: str, identity_label: str) -> str:
    """Dots with vertical 90% ranges and a dashed y = x line. Same scale on both axes."""
    if not points:
        return "<p class='caption'>No points to plot.</p>"
    top = max(max(p.x for p in points), max(p.y for p in points))
    top = max(10.0, math.ceil(top * 1.1 / 10) * 10)
    size, margin = 420, 52
    sx = _Scale(0, top, margin, size - 16)
    sy = _Scale(0, top, size - margin, 16)
    parts: list[str] = []
    for tick in _nice_ticks(0, top, 4):
        parts.append(f"<line class='grid' x1='{sx(0):.1f}' x2='{sx(top):.1f}' y1='{sy(tick):.1f}' y2='{sy(tick):.1f}' />")
        parts.append(f"<line class='grid' x1='{sx(tick):.1f}' x2='{sx(tick):.1f}' y1='{sy(0):.1f}' y2='{sy(top):.1f}' />")
        parts.append(_text(sx(tick), size - margin + 16, _tick_label(tick), anchor="middle"))
        parts.append(_text(margin - 8, sy(tick) + 4, _tick_label(tick), anchor="end"))
    parts.append(f"<line x1='{sx(0):.1f}' y1='{sy(0):.1f}' x2='{sx(top):.1f}' y2='{sy(top):.1f}' stroke='{REFERENCE}' stroke-width='2' stroke-dasharray='8 5' />")
    parts.append(_text(sx(top * 0.72), sy(top * 0.6), identity_label, size=11))
    parts.append(_text((size + margin) / 2, size - 10, x_label, anchor="middle"))
    parts.append(_text(16, (size - margin) / 2, y_label, anchor="middle", rotate=True))
    for point in sorted(points, key=lambda p: p.strong):
        color = PRIMARY if point.strong else SECONDARY
        if point.lower is not None and point.upper is not None:
            parts.append(
                f"<line x1='{sx(point.x):.1f}' x2='{sx(point.x):.1f}' y1='{sy(min(point.upper, top)):.1f}' y2='{sy(max(point.lower, 0)):.1f}' "
                f"stroke='{color}' stroke-width='1.5' opacity='0.5' />"
            )
        # Strong evidence: filled dot. Weaker evidence: hollow ring, so the two differ by shape as well as color.
        marker = (
            f"r='5.5' fill='{color}' class='halo' stroke-width='2'"
            if point.strong
            else f"r='4.5' class='hollow' stroke='{color}' stroke-width='2'"
        )
        parts.append(f"<circle cx='{sx(point.x):.1f}' cy='{sy(min(point.y, top)):.1f}' {marker}><title>{esc(point.tooltip)}</title></circle>")
    return _svg(size, size, f"{y_label} against {x_label}", parts)


def bar_list(rows: Sequence[tuple[str, float]], *, color: str = PRIMARY, value_format: str = "{:,.0f}", max_value: float | None = None) -> str:
    """Horizontal bars with the label on the right, sorted as given."""
    if not rows:
        return "<p class='caption'>Nothing to show.</p>"
    width, row_h = 640, 30
    label_w = 300
    top = max_value if max_value is not None else max(value for _, value in rows) or 1.0
    height = row_h * len(rows) + 6
    parts: list[str] = []
    for i, (label, value) in enumerate(rows):
        bar = max((value / top) * (width - label_w - 60), 3) if value > 0 else 0
        y = i * row_h + 4
        if bar:
            parts.append(f"<rect x='0' y='{y}' width='{bar:.1f}' height='{row_h - 10}' rx='4' fill='{color}'><title>{esc(label)}: {esc(value_format.format(value))}</title></rect>")
        parts.append(_text(bar + 8, y + 15, value_format.format(value), ink=True, weight=600))
        parts.append(_text(bar + 58, y + 15, label))
    return _svg(width, height, "bar chart", parts)


def histogram(values: Sequence[float], *, bin_width: float, guide: float | None, guide_label: str, x_label: str) -> str:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return "<p class='caption'>No values to plot.</p>"
    top = max(max(clean), guide or 0) + bin_width
    edges = [i * bin_width for i in range(int(math.ceil(top / bin_width)) + 1)]
    counts = [sum(1 for v in clean if edges[i] <= v < edges[i + 1]) for i in range(len(edges) - 1)]
    width, height, margin = 480, 220, 40
    sx = _Scale(0, edges[-1], margin, width - 12)
    sy = _Scale(0, max(counts) or 1, height - margin, 26)
    parts: list[str] = []
    for tick in _nice_ticks(0, max(counts) or 1, 4):
        if float(tick).is_integer():
            parts.append(f"<line class='grid' x1='{margin}' x2='{width - 12}' y1='{sy(tick):.1f}' y2='{sy(tick):.1f}' />")
            parts.append(_text(margin - 6, sy(tick) + 4, _tick_label(tick), anchor="end"))
    for i, count in enumerate(counts):
        if count:
            parts.append(
                f"<rect x='{sx(edges[i]) + 1:.1f}' y='{sy(count):.1f}' width='{sx(edges[i + 1]) - sx(edges[i]) - 2:.1f}' height='{sy(0) - sy(count):.1f}' rx='3' fill='{PRIMARY}'>"
                f"<title>{count} between {edges[i]:g} and {edges[i + 1]:g}</title></rect>"
            )
    for edge in edges:
        parts.append(_text(sx(edge), height - margin + 16, _tick_label(edge), anchor="middle"))
    if guide is not None:
        parts.append(f"<line x1='{sx(guide):.1f}' x2='{sx(guide):.1f}' y1='14' y2='{sy(0):.1f}' stroke='{REFERENCE}' stroke-width='2' stroke-dasharray='8 5' />")
        parts.append(_text(sx(guide) + 6, 18, guide_label, ink=True, size=11))
    parts.append(_text((width + margin) / 2, height - 6, x_label, anchor="middle", size=11))
    return _svg(width, height, x_label, parts)


def paired_dots(rows: Sequence[tuple[str, float | None, float | None]], *, first_label: str, second_label: str, axis_label: str) -> str:
    """One row per group, two dots on a shared axis (for example model vs baseline error)."""
    values = [v for _, a, b in rows for v in (a, b) if v is not None and math.isfinite(v)]
    if not values:
        return "<p class='caption'>No values to plot.</p>"
    hi = max(values) * 1.15
    width, row_h, label_w = 560, 34, 90
    height = row_h * len(rows) + 46
    sx = _Scale(0, hi, label_w, width - 20)
    parts: list[str] = []
    for tick in _nice_ticks(0, hi, 5):
        parts.append(f"<line class='grid' x1='{sx(tick):.1f}' x2='{sx(tick):.1f}' y1='4' y2='{row_h * len(rows) + 4}' />")
        parts.append(_text(sx(tick), row_h * len(rows) + 20, _tick_label(tick), anchor="middle"))
    parts.append(_text((width + label_w) / 2, height - 6, axis_label, anchor="middle", size=11))
    for i, (label, first, second) in enumerate(rows):
        y = i * row_h + row_h / 2 + 4
        parts.append(_text(label_w - 10, y + 4, label, anchor="end", ink=True))
        if first is not None and second is not None and math.isfinite(first) and math.isfinite(second):
            parts.append(f"<line x1='{sx(first):.1f}' x2='{sx(second):.1f}' y1='{y:.1f}' y2='{y:.1f}' class='grid' stroke-width='3' />")
        for value, color, name in ((second, COMPARATOR, second_label), (first, PRIMARY, first_label)):
            if value is not None and math.isfinite(value):
                parts.append(f"<circle cx='{sx(value):.1f}' cy='{y:.1f}' r='7' fill='{color}' class='halo' stroke-width='2'><title>{esc(label)}: {esc(name)} {value:.1f}</title></circle>")
    return _svg(width, height, axis_label, parts)


def time_series(
    timestamps: Sequence[object],
    series: Sequence[tuple[str, Sequence[float | None], str, bool]],
    *,
    band: tuple[Sequence[float | None], Sequence[float | None], str] | None = None,
    y_label: str,
    reference_lines: Sequence[tuple[float, str]] = (),
) -> str:
    """Lines over time on one y axis. ``series`` entries are (name, values, color, dashed)."""
    import pandas as pd

    ts = pd.to_datetime(pd.Series(list(timestamps)), errors="coerce")
    if ts.dropna().empty:
        return "<p class='caption'>No time points to plot.</p>"
    all_values = [v for _, values, _, _ in series for v in values if v is not None and math.isfinite(v)]
    if band is not None:
        all_values += [v for v in list(band[0]) + list(band[1]) if v is not None and math.isfinite(v)]
    all_values += [value for value, _ in reference_lines]
    lo, hi = min(all_values), max(all_values)
    pad = max((hi - lo) * 0.08, 5)
    lo, hi = lo - pad, hi + pad
    width, height, left, bottom = 760, 260, 48, 36
    t0, t1 = ts.min(), ts.max()
    seconds = [(t - t0).total_seconds() if pd.notna(t) else None for t in ts]
    span = max((t1 - t0).total_seconds(), 1.0)
    sx = _Scale(0, span, left, width - 12)
    sy = _Scale(lo, hi, height - bottom, 12)
    parts: list[str] = []
    for tick in _nice_ticks(lo, hi, 5):
        parts.append(f"<line class='grid' x1='{left}' x2='{width - 12}' y1='{sy(tick):.1f}' y2='{sy(tick):.1f}' />")
        parts.append(_text(left - 6, sy(tick) + 4, _tick_label(tick), anchor="end"))
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        stamp = t0 + pd.Timedelta(seconds=span * fraction)
        anchor = "start" if fraction == 0 else "end" if fraction == 1 else "middle"
        parts.append(_text(sx(span * fraction), height - bottom + 16, stamp.strftime("%b %d %H:%M"), anchor=anchor, size=11))
    parts.append(_text(12, (height - bottom) / 2, y_label, anchor="middle", rotate=True, size=11))

    steps = sorted(b - a for a, b in zip(seconds, seconds[1:]) if a is not None and b is not None and b > a)
    max_step = 3 * steps[len(steps) // 2] if steps else float("inf")

    def path(values: Sequence[float | None]) -> str:
        # Break the line at missing values and at time gaps, so separate nights are not joined.
        segments: list[str] = []
        current: list[str] = []
        previous_x: float | None = None
        for x, v in zip(seconds, values):
            gap = previous_x is not None and x is not None and x - previous_x > max_step
            if x is None or v is None or not math.isfinite(v) or gap:
                if current:
                    segments.append("M" + " L".join(current))
                    current = []
                if x is None or v is None or not math.isfinite(v):
                    previous_x = x
                    continue
            current.append(f"{sx(x):.1f},{sy(v):.1f}")
            previous_x = x
        if current:
            segments.append("M" + " L".join(current))
        return " ".join(segments)

    if band is not None:
        lower, upper, color = band
        top_pts = [(x, u) for x, u in zip(seconds, upper) if x is not None and u is not None and math.isfinite(u)]
        bot_pts = [(x, l) for x, l in zip(seconds, lower) if x is not None and l is not None and math.isfinite(l)]
        if top_pts and bot_pts:
            poly = " ".join(f"{sx(x):.1f},{sy(v):.1f}" for x, v in top_pts) + " " + " ".join(f"{sx(x):.1f},{sy(v):.1f}" for x, v in reversed(bot_pts))
            parts.append(f"<polygon points='{poly}' fill='{color}' opacity='0.18' />")
    for value, label in reference_lines:
        parts.append(f"<line x1='{left}' x2='{width - 12}' y1='{sy(value):.1f}' y2='{sy(value):.1f}' stroke='{REFERENCE}' stroke-width='1.5' stroke-dasharray='8 5' />")
        parts.append(_text(width - 14, sy(value) - 4, label, anchor="end", size=11))
    for name, values, color, dashed in series:
        dash = " stroke-dasharray='8 5'" if dashed else ""
        parts.append(f"<path d='{path(values)}' fill='none' stroke='{color}' stroke-width='2'{dash}><title>{esc(name)}</title></path>")
    return _svg(width, height, y_label, parts)
