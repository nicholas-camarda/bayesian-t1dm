"""The front page as one self-contained HTML file: no script, no network, no library; it reads the same with and without a stylesheet.

The page answers three questions in order, each with its evidence under it: does the model forecast better than the last 28 days
(the prespecified win test), can the record show what a settings change would do, and is the record fit to be modelled. The earlier
goals (the breakfast I:C proposal, the glucose dashboard, the goal folders) follow, below the verdict. Group colors, labels, markers
and line types come from ``figures`` and are the same in every figure; every number is printed once, in a table or in the sentence
that states the verdict, and the prose around it cites the table instead of repeating it. Intervals are 90% intervals.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .figures import CHANGE_SET_STYLE, NEUTRAL_BAND, NEUTRAL_LINE, OUTCOME_LABEL, OUTCOME_SHORT, PREDICTOR_STYLE, STATUS_STYLE, VERDICT_STYLE, GroupStyle, StatusStyle, assert_ascii
from .front_page import Audit, Check, Forecast, FrontPage, Missing, Proposal, Replay, gather, plain_markdown
from .layout import GOAL_LINKS
from .paths import ProjectPaths

# Marker shapes as CSS clip paths (None is a circle, "" a square); the marker letters are those of ``figures.GroupStyle``.
MARKER_SHAPE = {
    "o": None,
    "s": "",
    "D": "polygon(50% 0, 100% 50%, 50% 100%, 0 50%)",
    "^": "polygon(50% 0, 100% 100%, 0 100%)",
    "v": "polygon(0 0, 100% 0, 50% 100%)",
    "P": "polygon(35% 0, 65% 0, 65% 35%, 100% 35%, 100% 65%, 65% 65%, 65% 100%, 35% 100%, 35% 65%, 0 65%, 0 35%, 35% 35%)",
    "X": "polygon(20% 0, 50% 30%, 80% 0, 100% 20%, 70% 50%, 100% 80%, 80% 100%, 50% 70%, 20% 100%, 0 80%, 30% 50%, 0 20%)",
    "*": "polygon(50% 0, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%)",
}
# How each predictor is named inside a sentence.
NOUN = {
    "last 28 days": "the last-28-day baseline",
    "last 365 days": "the 365-day mean",
    "all history": "the all-history mean",
    "hour-day model": "the hour-day model",
    "hour-day model (settings pinned)": "the hour-day model with its settings pinned",
}
SENTENCE = {"win": "forecasts better than", "tie": "cannot be told apart from", "loss": "forecasts worse than"}
CONTAINS_ZERO = {"win": "lies entirely below zero", "tie": "holds zero", "loss": "lies entirely above zero"}
CHECK_GROUPS = ("Settings reconstruction", "Clock", "Leakage", "Recent regime")
GROUP_QUESTION = {
    "Settings reconstruction": "Are the pump settings known for every hour?",
    "Clock": "Do glucose, insulin and settings share one clock?",
    "Leakage": "Is every model input known when the forecast is made?",
    "Recent regime": "Is the latest fortnight like the earlier record?",
}


def esc(value: object) -> str:
    return html.escape(str(value))


def signed(value: float) -> str:
    return f"{value:+.2f}"


def interval(mean: float, lower: float, upper: float) -> str:
    return f"{signed(mean)} ({signed(lower)} to {signed(upper)})"


def chip(style: StatusStyle) -> str:
    """A status or verdict as text, glyph and color: it reads the same without the color."""
    assert_ascii(style.label, style.glyph)
    outline = " outline" if style.outline else ""
    return f"<span class='chip{outline}' style='--c:{style.color}'><span aria-hidden='true'>{esc(style.glyph)}</span> {esc(style.label)}</span>"


def data_table(headers: Sequence[str], rows: Iterable[Sequence[str]], *, numeric: Sequence[int] = (), caption_id: str = "") -> str:
    """A table whose rows stack on a phone: each cell carries its column name. Cells are trusted HTML."""
    head = "".join(f"<th scope='col'{' class=num' if i in numeric else ''}>{esc(h)}</th>" for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(f"<td data-label='{esc(headers[i])}'{' class=num' if i in numeric else ''}>{cell}</td>" for i, cell in enumerate(row)) + "</tr>" for row in rows
    )
    return f"<div class='tablewrap'><table class='stack'{f' aria-describedby={caption_id}' if caption_id else ''}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


@dataclass(frozen=True)
class ForestRow:
    label: str
    style: GroupStyle
    estimate: float
    lower: float
    upper: float
    description: str
    emphasis: bool = False


def _ticks(lo: float, hi: float) -> list[float]:
    span = hi - lo
    raw = span / 7
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= raw)
    first = math.ceil(lo / step) * step
    return [round(first + step * i, 10) for i in range(int((hi - first) / step) + 1)]


def forest(rows: Sequence[ForestRow], *, axis_label: str, references: Sequence[tuple[float, str]], band: float | None = None, band_label: str = "", title: str) -> str:
    """Estimates with 90% intervals on one horizontal axis, as HTML: real text that wraps, scales and reads aloud.

    Each row is a labelled track. ``references`` are vertical lines (a no-difference line, a size-right line); ``band`` is a symmetric
    grey band about zero. A group keeps its marker, color and line type in every chart (``figures``).
    """
    low = min(min(row.lower for row in rows), -(band or 0), *(value for value, _ in references))
    high = max(max(row.upper for row in rows), band or 0, *(value for value, _ in references))
    pad = (high - low) * 0.06
    low, high = low - pad, high + pad
    ticks = _ticks(low, high)

    def at(value: float) -> float:
        return 100.0 * (value - low) / (high - low)

    guides = ""
    if band:
        guides += f"<i class='fband' style='left:{at(-band):.2f}%;width:{at(band) - at(-band):.2f}%' aria-hidden='true'></i>"
    guides += "".join(f"<i class='fref' style='left:{at(value):.2f}%' aria-hidden='true'></i>" for value, _ in references)
    body = []
    for row in rows:
        clip = MARKER_SHAPE[row.style.marker]
        shape = "" if clip is None else f"clip-path:{clip};border-radius:0;"
        if row.style.marker == "o":
            shape = "border-radius:50%;"
        dash = "dashed" if row.style.linestyle != "-" else "solid"
        body.append(
            f"<div class='frow{' strong' if row.emphasis else ''}' role='img' aria-label='{esc(row.description)}'>"
            f"<div class='flabel'>{esc(row.label)}</div>"
            f"<div class='ftrack'>{guides}<i class='fline' style='left:{at(row.lower):.2f}%;width:{max(at(row.upper) - at(row.lower), 0.4):.2f}%;border-top-style:{dash};color:{row.style.color}' aria-hidden='true'></i>"
            f"<i class='fmark' style='left:{at(row.estimate):.2f}%;background:{row.style.color};{shape}' aria-hidden='true'></i></div></div>"
        )
    axis = "".join(f"<span style='left:{at(tick):.2f}%'>{esc(f'{tick:g}')}</span>" for tick in ticks)
    legend = "".join(f"<span class='lg'><i class='lgline'></i>{esc(label)}</span>" for _, label in references)
    if band:
        legend += f"<span class='lg'><i class='lgband'></i>{esc(band_label)}</span>"
    return (f"<figure class='forest' aria-label='{esc(title)}'>{''.join(body)}<div class='frow faxis'><div class='flabel'></div><div class='ftrack'>{axis}</div></div>"
            f"<p class='axislabel'>{esc(axis_label)}</p><div class='legendline'>{legend}</div></figure>")


def missing_card(item: Missing, question: str) -> str:
    return (f"<div class='card missing'>{chip(VERDICT_STYLE['missing'])}<p class='headline'>{esc(item.what)} is not available.</p>"
            f"<p>{esc(question)} cannot be answered from this run: <code>{esc(item.path)}</code> was not found. Write it with <code>{esc(item.command)}</code>, then refresh.</p></div>")


def header(data: FrontPage) -> str:
    runs = []
    for label, item in (("forecast harness", data.forecast), ("replay as predictor", data.replay), ("data audit", data.audit)):
        runs.append(f"{label}: {'not available' if isinstance(item, Missing) else 'run ' + item.run_id}")
    span = "" if isinstance(data.forecast, Missing) else f"; record {esc(data.forecast.record_span)}"
    return (
        "<header class='top'><h1>Can this record say which pump settings to change?</h1>"
        "<p class='goal'>Goal: forecast the next 14 days of time in range (70-180 mg/dL) better than the last 28 days, then use that model to rank pump-setting changes "
        "(I:C, correction factor, basal) by their probable benefit.</p>"
        "<p class='goal'>Claims are predictive and associational, from one person's retrospective record; nothing here changes a pump setting.</p>"
        f"<p class='meta'>Evidence: {esc('; '.join(runs))}{span}.</p></header>"
    )


def _forecast_chart(forecast: Forecast) -> str:
    order = [name for name in PREDICTOR_STYLE if name != "last 28 days"]
    by_name = {row.candidate: row for row in forecast.chart}
    rows = []
    for name in order:
        if name not in by_name:
            continue
        row = by_name[name]
        style = PREDICTOR_STYLE[name]
        gate = row.role == "win test"
        label = style.label + (" (win test)" if gate else "")
        rows.append(ForestRow(label, style, row.mean, row.lower, row.upper,
                              f"{style.label}: {interval(row.mean, row.lower, row.upper)} points against the last 28 days", emphasis=gate))
    return forest(rows, axis_label="CRPS difference from the last 28 days, points of 14-day time in range (left of 0: better forecast)", references=[(0.0, "No difference from the last 28 days")],
                  band=forecast.half_width, band_label="Differences this small cannot be separated from zero with these origins", title="Paired CRPS differences against the last 28 days")


def verdict_block(forecast: Forecast | Missing) -> str:
    question = "Does the model forecast better than the last 28 days?"
    if isinstance(forecast, Missing):
        return f"<section id='verdict' aria-labelledby='h-verdict'><h2 id='h-verdict'>1. {esc(question)}</h2>{missing_card(forecast, 'The win test')}</section>"
    win = forecast.win
    verdict = win.verdict
    headline = {
        "win": "The hour-day model forecast 14-day time in range better than the last 28 days",
        "tie": "The hour-day model and the last 28 days cannot be told apart",
        "loss": "The hour-day model forecast 14-day time in range worse than the last 28 days",
    }[verdict]
    explanation = f"the whole interval {CONTAINS_ZERO[verdict]}"
    by_pair = {(row.candidate, row.reference): row for row in forecast.exploratory}
    mean365, model365 = by_pair[("last 365 days", "last 28 days")], by_pair[("hour-day model", "last 365 days")]
    reading = ""
    if mean365.verdict == "win" and model365.verdict == "tie":
        reading = "<li>The model's advantage over the last 28 days is matched by the 365-day mean, so this test does not show that the model adds anything beyond a long-run level.</li>"
    elif mean365.verdict == "win" and model365.verdict == "win":
        reading = "<li>The model also forecasts better than the 365-day mean, which itself forecasts better than the last 28 days.</li>"
    items = "".join(
        f"<li>{esc(NOUN[row.candidate][0].upper() + NOUN[row.candidate][1:])} {SENTENCE[row.verdict]} {esc(NOUN[row.reference])}.</li>" for row in forecast.exploratory
    )
    note = (f"<li>{forecast.windows_with_change} of {forecast.origins} scored windows contain a real settings change that the frozen settings profile misses, "
            f"so the win test says nothing about what a change of settings would do.</li>")
    rows = [(esc(NOUN[row.candidate].removeprefix("the ")), esc(NOUN[row.reference].removeprefix("the ")), signed(row.mean), f"{signed(row.lower)} to {signed(row.upper)}", esc(VERDICT_STYLE[row.verdict].label.lower()))
            for row in forecast.exploratory]
    return (
        f"<section id='verdict' aria-labelledby='h-verdict'><h2 id='h-verdict'>1. {esc(question)}</h2>"
        f"<div class='card verdict'><p class='kicker'>Prespecified win test {chip(VERDICT_STYLE[verdict])}</p>"
        f"<p class='headline'>{esc(headline)}: <strong>{signed(win.mean)} points</strong> of CRPS (90% interval {signed(win.lower)} to {signed(win.upper)}; {win.origins} weekly origins).</p>"
        f"<p>Win rule: the 90% block-bootstrap interval of the model's CRPS minus the last-28-day CRPS must lie entirely below zero. Here {esc(explanation)}. "
        "CRPS scores the whole forecast distribution against what happened; lower is better.</p></div>"
        f"{_forecast_chart(forecast)}"
        "<p class='howto'>How to read it: each mark is the mean paired CRPS difference from the last 28 days at the same origins and the line is its 90% interval; a line wholly left of the zero line is a better forecast.</p>"
        f"<p class='caption' id='cap-forecast'>{forecast.origins} weekly forecast origins over the last year, each scoring the next 14 days; all {forecast.origins} windows had at least 70% of their readings. "
        "Intervals: circular block bootstrap (blocks of 4 origins, 10,000 resamples), which keeps the overlap between neighbouring windows.</p>"
        f"<h3>What the win does and does not show</h3><p class='note'>Exploratory comparators, chosen after seeing the results; none gates the win test.</p><ul>{items}{reading}{note}</ul>"
        + data_table(["Predictor", "Against", "CRPS difference (points)", "90% interval", "Reading"], rows, numeric=(2, 3), caption_id="cap-forecast") + "</section>"
    )


def _slope_chart(replay: Replay) -> str:
    rows = []
    for outcome in OUTCOME_LABEL:
        for group in CHANGE_SET_STYLE:
            row = next((item for item in replay.slopes if item.outcome == outcome and item.group == group), None)
            if row is None:
                continue
            style = CHANGE_SET_STYLE[group]
            rows.append(ForestRow(f"{OUTCOME_LABEL[outcome]}, {style.label.lower()}", style, row.slope, row.lower, row.upper,
                                  f"{OUTCOME_LABEL[outcome]}, {style.label.lower()}: slope {interval(row.slope, row.lower, row.upper)}"))
    return forest(rows, axis_label="Slope of observed on predicted change (0 = predictions carry no information, 1 = predicted size is right)",
                  references=[(0.0, "0: no information"), (1.0, "1: predicted size is right")], title="Slope of observed change on predicted change")


def settings_block(replay: Replay | Missing) -> str:
    question = "Can the record show what a settings change would do?"
    if isinstance(replay, Missing):
        return (f"<section id='settings' aria-labelledby='h-settings'><h2 id='h-settings'>2. {esc(question)}</h2>{missing_card(replay, 'The replay-as-predictor test')}"
                "<p><strong>Ranked setting changes are not shown</strong>: they need this evidence first.</p></section>")
    primary = replay.primary
    information = next(row for row in replay.information if row.outcome == primary.outcome and row.group == primary.group)
    verdict_key = "shown" if replay.validated else "not shown"
    if replay.validated:
        headline = "The predicted effects of a settings change tracked what happened after past changes in the primary test."
    else:
        headline = "Ranked setting changes are not shown: the record does not validate the model's predicted effects of a settings change."
    information_reading = {
        "loss": "the old settings predicted the 14-day window better than the settings in force",
        "tie": "the record does not show that the settings in force predict the window better than the old settings",
        "win": "the settings in force predicted the window better than the old settings",
    }[information.verdict]
    power = ("The test has low power for time in range, so a slope interval that holds zero is uninformative: it neither shows that the predictions track nor that they do not."
             if replay.underpowered else "")
    against = [row for row in replay.slopes if row.reading == "below zero"]
    moved = ""
    if against:
        names = "; ".join(f"{OUTCOME_SHORT[row.outcome]}, {CHANGE_SET_STYLE[row.group].label.lower()}" for row in against)
        moved = f"<li>Slope interval below zero (observed changes moved against the predicted direction): {esc(names)}.</li>"
    slope_rows = [(esc(OUTCOME_LABEL[row.outcome]), esc(CHANGE_SET_STYLE[row.group].label), f"{row.changes} ({row.clusters})", interval(row.slope, row.lower, row.upper),
                   esc(row.reading), "not computed" if row.power_at_one is None else f"{100 * row.power_at_one:.0f}%") for row in replay.slopes]
    reading_text = {"win": "settings in force better", "tie": "cannot be told apart", "loss": "old settings better"}
    information_rows = [(esc(OUTCOME_LABEL[row.outcome]), esc(CHANGE_SET_STYLE[row.group].label), str(row.changes), f"{row.crps_new:.2f}", f"{row.crps_old:.2f}",
                         interval(row.difference, row.lower, row.upper), reading_text[row.verdict]) for row in replay.information]
    return (
        f"<section id='settings' aria-labelledby='h-settings'><h2 id='h-settings'>2. {esc(question)}</h2>"
        f"<div class='card verdict'><p class='kicker'>Replay as predictor {chip(VERDICT_STYLE[verdict_key])}</p><p class='headline'>{esc(headline)}</p>"
        f"<p>{replay.predicted} of the {replay.changes_in_record} past settings changes in the record could be predicted ({replay.isolated} isolated: no other change within 14 days). "
        f"Each prediction used only data from before its change. {esc(power)}</p></div>"
        f"<ul>{moved}<li>Primary settings-information test (time in range, isolated changes): {esc(information_reading)}.</li></ul>"
        f"{_slope_chart(replay)}"
        "<p class='howto'>How to read it: for each past change, the model predicted the change in a percentage-point outcome and we measured what happened relative to dates with no change. "
        "The slope is observed per predicted point; a line wholly right of zero would mean the predictions track what happened.</p>"
        f"<p class='caption' id='cap-replay'>{replay.predicted} predicted changes ({replay.isolated} isolated); clusters are groups of changes within 14 days of each other, resampled together. "
        "Intervals: cluster bootstrap, 10,000 resamples. Predicted change is the model with the new settings minus the settings that were in force; observed change is the replay's change minus the placebo dates' change.</p>"
        "<h3>Slopes</h3>"
        + data_table(["Outcome", "Changes included", "Changes (clusters)", "Slope (90% interval)", "Slope interval", "Chance of finding a true slope of 1"], slope_rows, numeric=(2, 3, 5), caption_id="cap-replay")
        + "<h3>Do the settings in force predict the window better than the old settings?</h3>"
        + "<p class='note'>CRPS in percentage points of the 14-day share; difference is new minus old, so a negative difference favours the settings in force.</p>"
        + data_table(["Outcome", "Changes included", "Changes", "CRPS, settings in force", "CRPS, old settings", "Difference (90% interval)", "Reading"], information_rows, numeric=(2, 3, 4, 5))
        + "</section>"
    )


def _status_row(check: Check) -> str:
    style = STATUS_STYLE[check.status]
    return (f"<li class='check'><div class='checkhead'>{chip(style)} <strong>{esc(check.check)}</strong></div>"
            f"<p class='result'>{esc(check.result)}</p><p class='reason'>{esc(check.reason)}</p></li>")


def quality_block(audit: Audit | Missing) -> str:
    question = "Is the record fit to be modelled?"
    if isinstance(audit, Missing):
        return f"<section id='quality' aria-labelledby='h-quality'><h2 id='h-quality'>3. {esc(question)}</h2>{missing_card(audit, 'The data-quality audit')}</section>"
    counts = "".join(f"<li>{chip(STATUS_STYLE[name])} <strong>{audit.counts[name]}</strong></li>" for name in STATUS_STYLE)
    groups = []
    for group in CHECK_GROUPS:
        rows = [check for check in audit.checks if check.group == group]
        groups.append(f"<div class='checkgroup'><h3>{esc(GROUP_QUESTION[group])}</h3><ul class='checks'>{''.join(_status_row(check) for check in rows)}</ul></div>")
    unassessed = ""
    if audit.not_assessed:
        listed = "; ".join(f"{check.check} ({check.group.replace('_', ' ').lower()})" for check in audit.not_assessed)
        unassessed = f"<p class='note'><strong>Not assessed</strong> (could not be run, so it says nothing either way): {esc(listed)}.</p>"
    return (
        f"<section id='quality' aria-labelledby='h-quality'><h2 id='h-quality'>3. {esc(question)}</h2>"
        "<p>Data-quality audit of the whole record. A concern is a reason to look before relying on a result, not a finding about the person's glucose. "
        "A check that was not assessed is never a pass.</p>"
        f"<ul class='counts' aria-label='Audit checks by status'>{counts}</ul>{unassessed}"
        f"{''.join(groups)}</section>"
    )


def answers(data: FrontPage) -> str:
    """The three answers at a glance, each a status and a sentence built from the same results as the sections below; each links to its section."""
    cards = []
    forecast = data.forecast
    if isinstance(forecast, Missing):
        cards.append(("#verdict", "1. Forecast", chip(VERDICT_STYLE["missing"]), "The win test result is not available."))
    else:
        by_pair = {(row.candidate, row.reference): row for row in forecast.exploratory}
        text = {"win": "The model beats the last 28 days by the prespecified rule.", "tie": "The model cannot be told apart from the last 28 days.", "loss": "The model forecasts worse than the last 28 days."}[forecast.win.verdict]
        if by_pair[("last 365 days", "last 28 days")].verdict == "win" and by_pair[("hour-day model", "last 365 days")].verdict == "tie":
            text += " A plain 365-day mean does as well."
        cards.append(("#verdict", "1. Forecast", chip(VERDICT_STYLE[forecast.win.verdict]), text))
    replay = data.replay
    if isinstance(replay, Missing):
        cards.append(("#settings", "2. Settings effects", chip(VERDICT_STYLE["missing"]), "The replay-as-predictor result is not available, so ranked setting changes are not shown."))
    elif replay.validated:
        cards.append(("#settings", "2. Settings effects", chip(VERDICT_STYLE["shown"]), "Predicted effects tracked what happened after past changes in the primary test."))
    else:
        cards.append(("#settings", "2. Settings effects", chip(VERDICT_STYLE["not shown"]), "Ranked setting changes are not shown: the record does not validate predicted effects." + (" The test has low power." if replay.underpowered else "")))
    audit = data.audit
    if isinstance(audit, Missing):
        cards.append(("#quality", "3. Data quality", chip(VERDICT_STYLE["missing"]), "The data-quality audit is not available."))
    else:
        concerns = [group for group in CHECK_GROUPS if any(check.group == group and check.status == "concern" for check in audit.checks)]
        overall = "concern" if audit.counts["concern"] else "not assessed" if audit.counts["not assessed"] else "no concern"
        text = ("Concerns in " + ", ".join(group.lower() for group in concerns) + ".") if concerns else "No concern in the checks named below."
        cards.append(("#quality", "3. Data quality", chip(STATUS_STYLE[overall]), text + (" Some checks were not assessed." if audit.not_assessed else "")))
    links = "".join(f"<a class='scard' href='{href}'><span class='st'>{esc(title)}</span>{status}<span class='sx'>{esc(text)}</span></a>" for href, title, status, text in cards)
    return f"<nav class='strip' aria-label='The three answers'>{links}</nav>"


def earlier_block(proposal: Proposal | Missing, run_dir: Path) -> str:
    if isinstance(proposal, Missing):
        body = missing_card(proposal, "The breakfast proposal")
    else:
        body = (f"<div class='card'><p class='kicker'>{esc(proposal.title)}</p><p>{plain_markdown(proposal.paragraph)}</p>"
                "<p class='note'>This comes from the meal-level effective I:C analysis. It is not a ranked change and does not use the forecast model or the evidence in section 2.</p></div>")
    link = "<p><a href='dashboard.html'>Glucose dashboard</a>: today's numbers, a typical day, episodes, meals, basal and the change log.</p>" if (run_dir / "dashboard.html").exists() else \
        "<p>The glucose dashboard (<code>dashboard.html</code>) was not written to this run.</p>"
    return f"<section id='earlier' aria-labelledby='h-earlier'><h2 id='h-earlier'>Earlier goals</h2>{body}{link}</section>"


def files_block(run_dir: Path) -> str:
    rows = []
    for folder, question, entries in GOAL_LINKS:
        links = []
        for path, description in entries:
            links.append(f"<a href='{esc(path)}'>{esc(description)}</a>" if (run_dir / path).exists() else f"{esc(description)} (not written)")
        rows.append((f"<strong>{esc(folder)}</strong>", esc(question), "<br>".join(links)))
    return ("<section id='files' aria-labelledby='h-files'><h2 id='h-files'>Goal folders and diagnostics workbooks</h2>"
            + data_table(["Folder", "Question it answers", "Files"], rows) + "</section>")


CSS = f"""
:root {{ color-scheme: light; --ink:#1b1f24; --muted:#4a525c; --line:#d5d9de; --bg:#f6f7f9; --card:#ffffff; --link:#003C67; --grid:#e6e9ed;
  --lab:16rem; --neutral-line:{NEUTRAL_LINE}; --band:{NEUTRAL_BAND}; --focus:#0073C2; }}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.55 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
a {{ color:var(--link); text-underline-offset:2px; }} a:focus-visible {{ outline:3px solid var(--focus); outline-offset:2px; }}
code {{ font-size:.9em; background:var(--grid); padding:1px 5px; border-radius:4px; overflow-wrap:anywhere; }}
h1 {{ font-size:28px; line-height:1.2; margin:0 0 10px; }} h2 {{ font-size:21px; margin:0 0 12px; }} h3 {{ font-size:17px; margin:22px 0 6px; }}
.goal {{ margin:0 0 6px; max-width:70ch; }} .meta {{ color:var(--muted); font-size:14px; margin:10px 0 0; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px 18px; margin:0 0 12px; }}
.card.verdict {{ border-left:6px solid var(--neutral-line); }} .card.missing {{ border-style:dashed; }}
.kicker {{ margin:0 0 6px; font-weight:650; color:var(--muted); font-size:14px; }}
.headline {{ font-size:19px; line-height:1.4; margin:0 0 8px; }} .note {{ color:var(--muted); font-size:14.5px; margin:6px 0; }}
.howto {{ margin:10px 0 4px; font-size:15px; }} .caption {{ color:var(--muted); font-size:14px; margin:4px 0 12px; max-width:80ch; }}
ul {{ padding-left:20px; margin:8px 0; }} li {{ margin:4px 0; }}
.chip {{ display:inline-flex; align-items:center; gap:5px; font-size:13px; font-weight:700; letter-spacing:.02em; text-transform:uppercase; color:var(--c); border:2px solid var(--c);
  background:color-mix(in srgb, var(--c) 9%, #fff); border-radius:999px; padding:1px 10px; vertical-align:middle; white-space:nowrap; }}
.chip.outline {{ border-style:dashed; background:transparent; }}
.tablewrap {{ overflow-x:auto; }} table {{ border-collapse:collapse; width:100%; font-size:14.5px; }}
th, td {{ text-align:left; padding:7px 8px; border-bottom:1px solid var(--line); vertical-align:top; }} th {{ color:var(--muted); font-weight:650; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.forest {{ margin:14px 0 4px; padding:0; }} .frow {{ display:grid; grid-template-columns:var(--lab) 1fr; align-items:center; gap:12px; min-height:34px; }}
.flabel {{ font-size:14.5px; }} .frow.strong .flabel {{ font-weight:700; }}
.ftrack {{ position:relative; height:34px; }} .faxis .ftrack {{ height:22px; }} .faxis .ftrack span {{ position:absolute; top:2px; transform:translateX(-50%); font-size:13px; color:var(--muted); font-variant-numeric:tabular-nums; }}
.fline {{ position:absolute; top:50%; height:0; border-top:4px solid currentColor; }} .fmark {{ position:absolute; top:50%; width:16px; height:16px; transform:translate(-50%,-50%); filter:drop-shadow(0 0 1px #3B3B3B) drop-shadow(0 0 .5px #3B3B3B); }}
.fref {{ position:absolute; top:0; bottom:0; width:0; border-left:2px dashed var(--neutral-line); }} .fband {{ position:absolute; top:0; bottom:0; background:var(--band); opacity:.7; }}
.axislabel {{ margin:2px 0 6px; padding-left:calc(var(--lab) + 12px); font-size:13.5px; color:var(--muted); }}
.legendline {{ display:flex; flex-wrap:wrap; gap:6px 18px; font-size:13.5px; color:var(--muted); padding-left:calc(var(--lab) + 12px); }}
.lg {{ display:inline-flex; align-items:center; gap:6px; }} .lgline {{ width:0; height:14px; border-left:2px dashed var(--neutral-line); display:inline-block; }} .lgband {{ width:18px; height:12px; background:var(--band); display:inline-block; }}
.counts {{ list-style:none; display:flex; flex-wrap:wrap; gap:8px 22px; padding:0; margin:10px 0; }} .counts li {{ margin:0; font-size:17px; }}
.checks {{ list-style:none; padding:0; margin:0; }} .check {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin:8px 0; }}
.checkhead {{ display:flex; flex-wrap:wrap; align-items:center; gap:4px 10px; }} .result {{ margin:6px 0 2px; }} .reason {{ margin:0; color:var(--muted); font-size:14.5px; }}
.strip {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; margin:18px 0 28px; }}
.scard {{ display:flex; flex-direction:column; align-items:flex-start; gap:8px; background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; text-decoration:none; color:inherit; }}
.scard:hover, .scard:focus-visible {{ border-color:var(--focus); outline:none; box-shadow:0 0 0 2px var(--focus); }}
.st {{ font-weight:700; font-size:17px; }} .sx {{ font-size:15px; color:var(--muted); }}
section {{ margin:0 0 32px; scroll-margin-top:12px; }}
@media (max-width:820px) {{ .strip {{ grid-template-columns:1fr; gap:8px; margin:14px 0 22px; }} .scard {{ padding:10px 14px; gap:4px; }} }}
@media (max-width:620px) {{
  h1 {{ font-size:23px; }} .headline {{ font-size:17.5px; }}
  .frow {{ grid-template-columns:1fr; gap:2px; padding-top:8px; }} .faxis {{ padding-top:0; }} .faxis .flabel {{ display:none; }} .axislabel, .legendline {{ padding-left:0; }}
  table.stack thead {{ position:absolute; left:-9999px; }} table.stack, table.stack tbody, table.stack tr, table.stack td {{ display:block; width:100%; }}
  table.stack tr {{ border:1px solid var(--line); border-radius:8px; margin:0 0 10px; padding:4px 0; background:var(--card); }}
  table.stack td {{ border:0; padding:3px 10px; text-align:left; }} table.stack td::before {{ content:attr(data-label) ": "; color:var(--muted); font-weight:650; }}
}}
"""


def document(body: str, *, title: str = "Can this record say which pump settings to change?") -> str:
    return (f"<!doctype html>\n<html lang='en'>\n<head>\n<meta charset='utf-8'>\n<meta name='viewport' content='width=device-width, initial-scale=1'>\n<title>{esc(title)}</title>\n"
            f"<style>{CSS}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n")


def blocks(data: FrontPage, run_dir: Path) -> dict[str, str]:
    """Every section of the page, built once; ``render_front_page`` puts them in order."""
    return {"header": header(data), "answers": answers(data), "verdict": verdict_block(data.forecast), "settings": settings_block(data.replay), "quality": quality_block(data.audit),
            "earlier": earlier_block(data.proposal, run_dir), "files": files_block(run_dir)}


def render_front_page(data: FrontPage, run_dir: Path) -> str:
    parts = blocks(data, run_dir)
    order = ("header", "answers", "verdict", "settings", "quality", "earlier", "files")
    return document("<main style='max-width:1000px;margin:0 auto;padding:24px 16px 64px'>" + "".join(parts[name] for name in order) + "</main>")


def write_front_page(paths: ProjectPaths, run_dir: Path) -> Path:
    """Read the research outputs and write ``index.html`` into ``run_dir`` (links point at files already copied there)."""
    target = run_dir / "index.html"
    target.write_text(render_front_page(gather(paths), run_dir), encoding="utf-8")
    return target
