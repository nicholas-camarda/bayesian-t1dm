"""Tables, figure and diagnostics workbook for the replay of past setting changes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import replay
from .diagnostics_workbook import read_workbook, write_workbook
from .figures import GROUP_STYLE, NEUTRAL_BAND, NEUTRAL_BAR, NEUTRAL_LINE, assert_ascii
from .hour_day import hour_day_flow
from .publish import current_commit, working_tree_is_clean

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_FILES = ("replay_figure.png", "replay_figure.svg")
PANELS = (("tir", "Time in range 70-180\n(percentage points)"), ("below70", "Time below 70\n(percentage points)"))


def _delta_rows(summary: pd.DataFrame, measure: str) -> pd.DataFrame:
    rows = summary[(summary["predictor"] == replay.PRIMARY) & (summary["measure"] == measure) & summary["group"].isin(["all changes", "isolated changes"])]
    return rows.set_index("group").reindex(["all changes", "isolated changes"]).dropna(subset=["delta"])


def write_figure(results: dict[str, pd.DataFrame], output: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    scores = results["scores"]
    primary = scores[scores["predictor"].eq(replay.PRIMARY)]
    changes = primary[primary["kind"].eq("change")].copy()
    changes["date"] = pd.to_datetime(changes["date"])
    placebos = primary[primary["kind"].eq("matched placebo")]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.6), gridspec_kw={"width_ratios": [3.6, 1.6], "hspace": 0.32, "wspace": 0.28})
    for row, (outcome, label) in enumerate(PANELS):
        axis, strip = axes[row]
        low, high = np.percentile(placebos[f"{outcome}_residual"], replay.INTERVAL_PERCENT)
        axis.axhspan(low, high, color=NEUTRAL_BAND, zorder=0)
        axis.axhline(0, color=NEUTRAL_LINE, linewidth=1.0, zorder=1)
        for _, change in changes.iterrows():
            axis.plot([change["date"]] * 2, [change[f"{outcome}_lower"] - change[f"{outcome}_predicted"], change[f"{outcome}_upper"] - change[f"{outcome}_predicted"]], color=NEUTRAL_BAR, linewidth=1.3, zorder=2)
        for direction, group in changes.groupby("direction"):
            style = GROUP_STYLE[direction]
            for isolated, part in group.groupby("isolated"):
                axis.scatter(part["date"], part[f"{outcome}_residual"], marker=style.marker, s=46 if isolated else 40, zorder=3, linewidths=1.6,
                             facecolors=style.color if isolated else "white", edgecolors=style.color)
        axis.set_ylabel(label, fontsize=10)
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axis.tick_params(labelsize=9)
        axis.spines[["top", "right"]].set_visible(False)
        estimates = _delta_rows(results["summary"], outcome)
        strip.axvline(0, color=NEUTRAL_LINE, linewidth=1.0)
        for position, (group, estimate) in enumerate(estimates.iterrows()):
            strip.errorbar(estimate["delta"], position, xerr=[[estimate["delta"] - estimate["delta_lower_90"]], [estimate["delta_upper_90"] - estimate["delta"]]], fmt="o", color=GROUP_STYLE["more insulin"].color, capsize=4, linewidth=2.2, markersize=7)
            strip.text(estimate["delta"], position + 0.12, f"{estimate['delta']:+.2f}\n({estimate['delta_lower_90']:+.2f} to {estimate['delta_upper_90']:+.2f})", fontsize=8, va="top", ha="center", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
        strip.set_yticks(range(len(estimates)))
        strip.set_yticklabels([f"{group.replace(' changes', '')}\n(n={int(est['n_changes'])})" for group, est in estimates.iterrows()], fontsize=9)
        strip.set_ylim(len(estimates) - 0.3, -0.5)
        strip.set_xlabel("Change minus placebo\n(mean, 90% interval)", fontsize=9)
        strip.tick_params(labelsize=9)
        strip.spines[["top", "right"]].set_visible(False)
        strip.margins(x=0.25)
    handles = [Line2D([0], [0], marker=GROUP_STYLE[name].marker, color="none", markerfacecolor=GROUP_STYLE[name].color, markeredgecolor=GROUP_STYLE[name].color, markersize=7, label=GROUP_STYLE[name].label)
               for name in ("more insulin", "less insulin", "mixed") if name in set(changes["direction"])]
    handles += [Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=NEUTRAL_LINE, markersize=7, label="Hollow: another change within 14 days"),
                Patch(color=NEUTRAL_BAND, label="Placebo dates, 5th to 95th percentile"), Line2D([0], [0], color=NEUTRAL_BAR, linewidth=1.5, label="90% range the baseline predicts")]
    figure.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.005))
    title = "Exploratory: outcomes over the 14 days after each settings change, against the last 28 days before it"
    note = "Each point: observed minus the last-28-day prediction over the affected hours. Grey band: where matched no-change dates fall.\nRight: mean of (change minus its matched placebo dates), with 90% interval."
    assert_ascii(title, note, *[label for _, label in PANELS])
    figure.suptitle(title, fontsize=12, x=0.01, ha="left", y=0.995)
    figure.text(0.01, 0.93, note, fontsize=8.5, ha="left", va="top", color=NEUTRAL_LINE)
    figure.subplots_adjust(top=0.87, bottom=0.17, left=0.08, right=0.97)
    paths = []
    for name in FIGURE_FILES:
        path = output / name
        figure.savefig(path, dpi=300)
        paths.append(path)
    plt.close(figure)
    return paths


def _checks(results: dict[str, pd.DataFrame], table: pd.DataFrame) -> pd.DataFrame:
    scores, summary = results["scores"], results["summary"]
    grid = replay.day_grid(table)
    flagged = np.flatnonzero(grid.changed.any(axis=1))
    position = {date.date(): index for index, date in enumerate(grid.dates)}
    primary = scores[scores["predictor"].eq(replay.PRIMARY)]
    placebo_days = np.array([position[date] for date in primary.loc[primary["kind"].str.contains("placebo"), "date"]])
    nearest = np.abs(placebo_days[:, None] - flagged[None, :]).min(axis=1).min()
    changes = primary[primary["kind"].eq("change")]
    matched = primary[primary["kind"].eq("matched placebo")]
    random = primary[primary["kind"].eq("random placebo")]
    gap_matched = float((matched["pre_tir"].to_numpy() - changes["pre_tir"].repeat(replay.PLACEBOS_PER_EVENT).to_numpy()).__abs__().mean())
    gap_random = float((random["pre_tir"].to_numpy() - changes["pre_tir"].repeat(replay.PLACEBOS_PER_EVENT).to_numpy()).__abs__().mean())
    coverage = results["calibration"].query("kind == 'matched placebo' and predictor == @replay.PRIMARY").iloc[0]
    isolated = int(changes["isolated"].sum())
    mixed = float(changes["direction"].eq("mixed").mean())
    rows = [
        ("Placebo dates sit clear of every change", f"closest placebo date is {nearest} days from a day holding a changed hour (limit {replay.CLEARANCE_DAYS + 1})", "no concern" if nearest > replay.CLEARANCE_DAYS else "concern"),
        ("Matched placebos resemble the changes before the change", f"mean gap in 28-day time in range: matched {gap_matched:.1f} points, random {gap_random:.1f} points", "no concern" if gap_matched < gap_random else "concern"),
        ("Baseline interval coverage on placebo dates (time in range)", f"{coverage['tir_coverage_90']:.2f} of matched placebo dates fall inside the 90% interval (nominal 0.90)", "no concern" if 0.80 <= coverage["tir_coverage_90"] <= 0.97 else "concern"),
        ("Baseline interval coverage on placebo dates (time below 70)", f"{coverage['below70_coverage_90']:.2f} (nominal 0.90)", "no concern" if 0.80 <= coverage["below70_coverage_90"] <= 0.97 else "concern"),
        ("Predictors read only days before the change", "structural: the predictor receives the 90 days before the change day; a test changes later days and checks that the prediction is identical", "no concern"),
        ("Isolated changes available", f"{isolated} of {len(changes)} scored changes have no other change within 14 days", "no concern" if isolated >= 15 else "concern"),
        ("Changes with mixed insulin direction", f"{mixed:.0%} of scored changes move insulin both ways and are left out of the direction-agreement rows", "no concern"),
        ("Cluster bootstrap", f"{int(summary.loc[summary['group'].eq('all changes'), 'n_clusters'].max())} clusters of changes, {replay.DRAWS} draws, seed {replay.SEED + 7}", "no concern"),
        ("Sampler convergence (R-hat, effective sample size)", "no sampler is used: predictors resample days and the interval is a bootstrap", "not applicable"),
        ("Settings known for every affected hour", "a change is detected only where both days' settings are known; hours the archive cannot show are not compared (see Denominators_Flow)", "not assessed"),
    ]
    return pd.DataFrame(rows, columns=["check", "result", "status"])


def _specification() -> pd.DataFrame:
    rows = [
        ("Question", "How far do outcomes over the 14 days after a settings change sit from what the days before predicted, compared with no-change dates?"),
        ("Claim type", "Predictive and associational, one person, exploratory; not an effect of the setting."),
        ("Population", "Every change in the settings in force from 2024-01 with its 14 following days in the record; days listed in context_events.csv are left out."),
        ("Unit of analysis", "A change event (consecutive days holding a changed hour are one event); the bootstrap resamples clusters of events within 14 days of each other."),
        ("Outcomes", "Over the affected hours (changed hours plus 4): pooled time in range 70-180 mg/dL, time below 70, time below 54 (percent of readings) and mean glucose (mg/dL)."),
        ("A day counts", f"when at least {replay.MIN_COVERAGE:.0%} of the expected readings in the affected hours are present; at least {replay.MIN_BEFORE_DAYS} of the 28 days before and {replay.MIN_AFTER_DAYS} of the 14 days after must count."),
        ("Predictors", "last 28 days resampled (primary); last 90 days with weights halving every 28 days (sensitivity). Each draws as many days as the observed window has and pools their readings."),
        ("Effect measure", "delta = mean over changes of (observed minus predicted) minus the mean over each change's 5 matched placebo dates of the same quantity. Units: percentage points (mg/dL for mean glucose). Positive time in range means the change's 14 days did better than a no-change date's."),
        ("Placebo dates", f"no change within {replay.CLEARANCE_DAYS} days either side, within {replay.SEASON_DAYS} days of the same time of year (any year), same data sufficiency; 5 matched on the pre-change time in range, 5 random as a check on the matching."),
        ("Direction agreement", "For changes whose settings move insulin one way: the share where mean glucose fell after more insulin (rose after less) and where time below 70 rose after more insulin (fell after less); estimated against the same share on matched placebo dates."),
        ("Interpretation", "A delta whose interval excludes 0 says the 14 days after changes differ from the pre-change level more than placebo dates do. It does not say the setting caused it: several settings and context changes can move together."),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def _warnings(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [
        ("Exploratory", "Intervals describe this one person's record; none of the estimates was prespecified as a test."),
        ("Overlapping windows", "Changes closer than 28 days share before or after days; the bootstrap resamples clusters, so the interval reflects it, but the number of independent changes is the cluster count, not the change count."),
        ("Baselines ignore the settings", "The predictors do not use the settings, so the residual contains every difference between the periods (season, illness, new device) except what placebo dates share."),
        ("Pump change", "On the day the active pump changes, settings are not compared across pumps; a real change on that day is not seen."),
        ("Early record", "Until a segment of the first pump's profile was edited or seen in the archive its values are unknown; 21:00-24:00 stays unknown for that pump, so changes there before 2025-09 are not detected."),
    ]
    return pd.DataFrame(rows, columns=["limitation", "detail"])


def write_replay_outputs(results: dict[str, pd.DataFrame], table: pd.DataFrame, output: Path, *, run_id: str, raw: str, seconds: float) -> None:
    output = Path(output)
    scores, summary, flow, issues, calibration = (results[name] for name in ("scores", "summary", "flow", "issues", "calibration"))
    scores.to_csv(output / "replay_scores.csv", index=False)
    summary.to_csv(output / "replay_summary.csv", index=False)
    calibration.to_csv(output / "replay_calibration.csv", index=False)
    figures = write_figure(results, output)
    hour_flow = hour_day_flow(table)
    flow_sheet = pd.concat([
        pd.DataFrame({"stage": [f"hour-day table: {name.replace('_', ' ')}" for name in hour_flow.__dict__], "n": list(hour_flow.__dict__.values())}),
        flow,
    ], ignore_index=True)
    dirty = not working_tree_is_clean(REPO_ROOT)
    provenance = pd.DataFrame([
        ("run id", run_id), ("commit", current_commit(REPO_ROOT)), ("working tree clean", "no" if dirty else "yes"), ("raw data root", raw),
        ("record span", f"{table['date'].min().date()} to {table['date'].max().date()}"), ("seed", replay.SEED), ("bootstrap draws", replay.DRAWS),
        ("command", "python -m bayesian_t1dm.replay"), ("wall time (seconds)", round(seconds, 1)),
        ("files", json.dumps(["replay_scores.csv", "replay_summary.csv", "replay_calibration.csv", *[path.name for path in figures]])),
    ], columns=["item", "value"])
    sheets = {
        "Specification_Coding": _specification(), "Denominators_Flow": flow_sheet, "Replay_Checks": _checks(results, table),
        "Record_Issues": issues if not issues.empty else pd.DataFrame(columns=["event_id", "change_date", "change_days", "step", "reason", "field", "value"]),
        "Warnings_Limitations": _warnings(results), "Artifacts_Provenance": provenance,
    }
    path = write_workbook(output / "diagnostics.xlsx", sheets)
    _reconcile(path, results)


def _reconcile(path: Path, results: dict[str, pd.DataFrame]) -> None:
    """Read the workbook back: its flow and record issues must agree with the scored changes in the tables."""
    workbook = read_workbook(path)
    flow = workbook["Denominators_Flow"].set_index("stage")["n"]
    changes = int(flow["changes (consecutive days merged)"])
    scored = int(flow["with five placebo dates (scored)"])
    primary = results["scores"].query("predictor == @replay.PRIMARY and kind == 'change'")
    if scored != len(primary):
        raise RuntimeError(f"Workbook flow says {scored} scored changes, the scores table holds {len(primary)}")
    if len(workbook["Record_Issues"]) != changes - scored:
        raise RuntimeError(f"Record_Issues holds {len(workbook['Record_Issues'])} rows, the flow excludes {changes - scored} changes")
