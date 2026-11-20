"""Tables, figure, summary and diagnostics workbook of the replay-as-predictor run (``replay_predictor``)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import outcomes_model as om
from . import replay
from . import replay_predictor as rp
from .diagnostics_workbook import read_workbook, write_workbook
from .figures import GROUP_STYLE, NEUTRAL_BAND, NEUTRAL_LINE, assert_ascii
from .publish import current_commit, working_tree_is_clean
from .sampler_health import sampler_gate

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_FILES = ("replay_predictor_figure.png", "replay_predictor_figure.svg")
OUTCOME_LABELS = {"tir": "time in range (70-180 mg/dL)", "below70": "time below 70 mg/dL", "below54": "time below 54 mg/dL"}
OUTCOME_SHORT = {"tir": "time in range", "below70": "time below 70", "below54": "time below 54"}
SETTING_BETAS = ("log I:C", "log correction factor", "log basal", "target")
PRIOR_DOMINATED_RATIO = rp.PRIOR_DOMINATED_RATIO


def _interval(estimate: float, lower: float, upper: float, digits: int = 2) -> str:
    return f"{estimate:+.{digits}f} (90% interval {lower:+.{digits}f} to {upper:+.{digits}f})"


def _short(estimate: float, lower: float, upper: float) -> str:
    return f"{estimate:+.2f} ({lower:+.2f} to {upper:+.2f})"


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def write_figure(joined: pd.DataFrame, information: pd.DataFrame, tracking: pd.DataFrame, scores: pd.DataFrame, output: Path) -> list[Path]:
    """Left: predicted against observed change in time in range, one point per change. Right: per change, CRPS under the settings in force minus CRPS under the old settings."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    frame = joined[joined["outcome"].eq(rp.PRIMARY_OUTCOME)].sort_values("change_date").reset_index(drop=True)
    spread = (float(frame["no_change_noise_lower_5th"].iloc[0]), float(frame["no_change_noise_upper_95th"].iloc[0]))
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 8.2), gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.42})
    lo = min(frame["predicted_lower_5th"].min(), frame["observed_change_points"].min()) - 1.0
    hi = max(frame["predicted_upper_95th"].max(), frame["observed_change_points"].max()) + 1.0
    limit = max(abs(lo), abs(hi))
    left.axhspan(spread[0], spread[1], color=NEUTRAL_BAND, zorder=0)
    left.plot([-limit, limit], [-limit, limit], color=NEUTRAL_LINE, linestyle="--", linewidth=1.4, zorder=1)
    left.axhline(0, color=NEUTRAL_LINE, linewidth=0.8, zorder=1)
    left.axvline(0, color=NEUTRAL_LINE, linewidth=0.8, zorder=1)
    for _, row in frame.iterrows():
        left.plot([row["predicted_lower_5th"], row["predicted_upper_95th"]], [row["observed_change_points"]] * 2, color=NEUTRAL_LINE, alpha=0.35, linewidth=1.2, zorder=2)
    for direction, group in frame.groupby("direction"):
        style = GROUP_STYLE[direction]
        for isolated, part in group.groupby("isolated"):
            left.scatter(part["predicted_change_points"], part["observed_change_points"], marker=style.marker, s=58 if isolated else 46, linewidths=1.7, zorder=3,
                         facecolors=style.color if isolated else "white", edgecolors=style.color)
    primary = tracking[(tracking["outcome"] == rp.PRIMARY_OUTCOME) & (tracking["group"] == "isolated changes")]
    isolated_rows = frame[frame["isolated"]]
    if len(primary) and len(isolated_rows) >= 3 and isolated_rows["predicted_change_points"].nunique() > 1:
        fitted = np.polyfit(isolated_rows["predicted_change_points"], isolated_rows["observed_change_points"], 1)
        xs = np.array([-limit, limit])
        left.plot(xs, np.polyval(fitted, xs), color=GROUP_STYLE["more insulin"].color, linewidth=2.2, zorder=2)
    left.set_xlim(-limit, limit)
    left.set_ylim(-limit, limit)
    left.set_aspect("equal", adjustable="box")
    left.set_xlabel("Model's predicted change in time in range\n(percentage points; bar = 90% credible interval)", fontsize=10)
    left.set_ylabel("Observed change minus placebo change\n(percentage points)", fontsize=10)
    left.tick_params(labelsize=9)
    left.spines[["top", "right"]].set_visible(False)
    left.set_title("Predicted against observed change", fontsize=11, loc="left")
    # right panel: one row per change
    positions = np.arange(len(frame))[::-1]
    right.axvline(0, color=NEUTRAL_LINE, linewidth=1.0)
    for position, (_, row) in zip(positions, frame.iterrows()):
        style = GROUP_STYLE[row["direction"]]
        right.scatter(row["crps_new_minus_old_points"], position, marker=style.marker, s=46, linewidths=1.6, zorder=3, facecolors=style.color if row["isolated"] else "white", edgecolors=style.color)
    right.set_yticks(positions)
    right.set_yticklabels([f"{row['change_date']}" for _, row in frame.iterrows()], fontsize=7.5)
    result = information[(information["outcome"] == rp.PRIMARY_OUTCOME)].set_index("group")
    base = -1.6
    for offset, group in enumerate(rp.GROUPS):
        if group not in result.index:
            continue
        est = result.loc[group]
        y = base - 1.6 * offset
        right.errorbar(est["mean_difference_points"], y, xerr=[[est["mean_difference_points"] - est["lower_5th"]], [est["upper_95th"] - est["mean_difference_points"]]], fmt="D", color=GROUP_STYLE["more insulin"].color if offset == 0 else NEUTRAL_LINE,
                       capsize=4, linewidth=2.2, markersize=7, zorder=4)
        right.text(right.get_xlim()[1], y, f"  {group.replace(' changes', '')} (n={int(est['changes'])}): {est['mean_difference_points']:+.2f}", fontsize=8, va="center", ha="left", clip_on=False)
    right.set_ylim(base - 3.2, len(frame) - 0.4)
    right.set_xlabel("CRPS under the settings in force minus CRPS under the old settings\n(percentage points; below 0 = the settings in force predict better)", fontsize=10)
    right.tick_params(labelsize=9)
    right.spines[["top", "right", "left"]].set_visible(False)
    right.tick_params(axis="y", length=0)
    right.set_title("Do the settings in force add information?", fontsize=11, loc="left")
    handles = [Line2D([0], [0], marker=GROUP_STYLE[name].marker, color="none", markerfacecolor=GROUP_STYLE[name].color, markeredgecolor=GROUP_STYLE[name].color, markersize=7, label=GROUP_STYLE[name].label)
               for name in ("more insulin", "less insulin", "mixed") if name in set(frame["direction"])]
    handles += [Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=NEUTRAL_LINE, markersize=7, label="Hollow: another change within 14 days"),
                Line2D([0], [0], color=NEUTRAL_LINE, linestyle="--", linewidth=1.4, label="Predicted = observed"),
                Line2D([0], [0], color=GROUP_STYLE["more insulin"].color, linewidth=2.2, label="Least-squares line, isolated changes"),
                Patch(color=NEUTRAL_BAND, label="Range of the observed change when nothing changed (5th to 95th percentile)")]
    figure.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    title = "Do the model's predicted effects of past settings changes match what happened? Time in range, 14 days after each change"
    note = ("Left: each point is one change; on the dashed line the model's predicted change equals the observed one. The grey band is the range the observed change takes when nothing changed, so points inside it cannot be told from chance.\n"
            "Right: each row is one change; a dot left of 0 means the settings in force predicted the window better than the same fit with the old settings. Diamonds: mean with 90% interval over clusters of changes.")
    assert_ascii(title, note)
    figure.suptitle(title, fontsize=12, x=0.01, ha="left", y=0.995)
    figure.text(0.01, 0.965, note, fontsize=8.5, ha="left", va="top", color=NEUTRAL_LINE)
    figure.subplots_adjust(top=0.88, bottom=0.2, left=0.07, right=0.84)
    paths = []
    for name in FIGURE_FILES:
        path = output / name
        figure.savefig(path, dpi=300)
        paths.append(path)
    plt.close(figure)
    return paths


# ---------------------------------------------------------------------------
# Workbook sheets
# ---------------------------------------------------------------------------

def _specification(outcomes: tuple[str, ...], draws: int, tune: int, chains: int, min_train_days: int) -> pd.DataFrame:
    rows = [
        ("Question", "Do the outcomes model's predicted effects of a setting change match what happened after the past changes, and does the model with the settings in force predict the post-change window better than the same fit with the old settings?"),
        ("Claim type", "Predictive and associational, one person's retrospective record; not the effect of a setting."),
        ("Population", f"Every change the replay scored (see Denominators_Flow) with at least {min_train_days} days with usable hours before it. The replay's own counting rules apply: a day counts when at least {replay.MIN_COVERAGE:.0%} of the expected readings in the affected hours are present and it is not a context-event day."),
        ("Unit of analysis", "A change event (consecutive days holding a changed hour are one event). Intervals resample clusters of changes within 14 days of each other, with their placebos."),
        ("Outcomes", "Share of valid readings in the outcome over the replay's affected hours (the changed hours and the 4 after) on the 14 days after the change, pooled over the counted days: " + ", ".join(OUTCOME_LABELS[name] for name in outcomes) + ". Units: percentage points."),
        ("Model", f"The hour-day outcomes model (beta-binomial, hour and day effects, 4 settings and 4 exposures), one fit per outcome refitted on the rows before the fit's cut: {chains} chains, {tune} tuning steps, {draws} draws per chain, the harness settings. Every fit must pass the sampler gate (R-hat below {om.R_HAT_LIMIT}, ESS above {om.ESS_LIMIT}, no divergences) or the run fails."),
        ("Fit cut", f"The first day of the change. A change within {rp.FIT_SHARE_DAYS} days after the first change of its group reuses that change's fit, so a fit never sees a day on or after the change it serves."),
        ("Predicted change", f"Expected share of readings in the outcome over the post-change window under the profile in force after the change (new, held for the 14 days) minus the expected share under the settings actually in force on each of the replay's baseline days (the qualifying days of the 28 before the change, weighted by their readings), in percentage points. This is the quantity the replay observes: the change's miss from the last-28-day predictor. Same posterior draw, day effects and resampled exposures in both; mean and 5th and 95th percentile over the {rp.DRAWS:,} posterior draws. Positive time in range: more readings in range under the new settings. For below 70 and below 54 a negative value means fewer lows. Sensitivity: the old arm as the profile in force the day before the change (Sensitivity sheet)."),
        ("Exposures", "Carbohydrate of the last 3 h, exercise share, sleep-mode share and site age: whole days resampled from the 28 days before the change, 14 per draw. Reading counts per hour are the realised ones on the counted days (the number of readings is not glucose information)."),
        ("Observed change", "The replay's delta per change: (observed minus last-28-day prediction over the affected hours) minus the mean of the same quantity over the change's 5 matched placebo dates. Units: percentage points. Positive time in range: the 14 days did better than no-change dates do."),
        ("Tracking statistics", "Least-squares slope and Pearson correlation of observed on predicted change; direction agreement (same sign); inside_credible_interval (the observed change lies in the 90% credible interval of the predicted change, which describes the expected change and not the sampling noise of an observed window, so under-covers by construction); inside_noise_range (the observed change lies within the predicted change plus the 5th to 95th percentile of misses on matched placebo dates, conservative because the observed change subtracts a mean of five placebos). A slope of 1 means the predicted size is right; 0 means the predictions carry no information about what happened."),
        ("Settings-information test", "CRPS (percentage points) of the 14-day share under the new settings and under the old settings (the profile in force the day before the change, held), same fit and exposures; difference new minus old, paired across changes. Below 0: the settings in force predict what happened better."),
        ("Groups", "Isolated changes (no other change within 14 days; primary) and all changes."),
        ("Intervals", f"90% percentile intervals of {rp.BOOTSTRAP_DRAWS:,} cluster-bootstrap resamples, seed {rp.SEED}."),
        ("Interpretation", "A slope interval above zero says changes the model predicts to help did better after the change than no-change dates; an interval holding zero says the record does not show that. It does not say that a setting caused the change in outcome: settings and context move together."),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def _flow(flow: pd.DataFrame, joined: pd.DataFrame, fits: dict[str, dict], targets: list[rp.EventTarget], outcomes: tuple[str, ...]) -> pd.DataFrame:
    rows = [(row.stage, int(row.changes), "changes") for row in flow.itertuples()]
    primary = joined[joined["outcome"].eq(rp.PRIMARY_OUTCOME)]
    rows += [("predicted (after the fits)", len(primary), "changes"), ("of which isolated", int(primary["isolated"].sum()), "changes"),
             ("clusters of changes, all", int(primary["cluster_id"].nunique()), "clusters"), ("clusters of changes, isolated", int(primary.loc[primary["isolated"], "cluster_id"].nunique()), "clusters")]
    for outcome in outcomes:
        rows.append((f"fits for {OUTCOME_SHORT[outcome]} (one per distinct cut)", len(fits[outcome]), "fits"))
    rows.append(("changes sharing an earlier change's fit", int(sum(target.cut != target.event_date for target in targets)), "changes"))
    return pd.DataFrame(rows, columns=["stage", "n", "unit"])


def _model_fits(fits: dict[str, dict], targets: list[rp.EventTarget]) -> pd.DataFrame:
    rows = []
    for outcome, by_cut in fits.items():
        for cut, posterior in sorted(by_cut.items()):
            served = [target.event_id for target in targets if target.cut == cut]
            diag = posterior.diagnostics
            ratio = posterior.beta[:, :4].std(axis=0) / rp.SETTING_PRIOR_SD
            row = {"outcome": outcome, "fit_cut": cut.date(), "training_rows": posterior.rows, "training_days": posterior.days, "changes_served": len(served), "event_ids": ", ".join(map(str, served)),
                   "r_hat_max": diag.rhat_max, "ess_bulk_min": diag.ess_bulk_min, "ess_tail_min": diag.ess_tail_min, "divergences": diag.divergences, "sampling_seconds": posterior.timing["sample_seconds"]}
            row.update({f"posterior_sd_over_prior_sd_{name}": float(value) for name, value in zip(SETTING_BETAS, ratio)})
            row["status"] = "no concern" if sampler_gate(diag) else "concern"
            rows.append(row)
    return pd.DataFrame(rows)


def _checks(joined: pd.DataFrame, tracking: pd.DataFrame, fits_table: pd.DataFrame, flow: pd.DataFrame, issues: pd.DataFrame, min_train_days: int) -> pd.DataFrame:
    rows = []
    rows.append(("Sampler gate on every fit", f"{len(fits_table)} fits; max R-hat {fits_table['r_hat_max'].max():.4f}, min bulk ESS {fits_table['ess_bulk_min'].min():.0f}, min tail ESS {fits_table['ess_tail_min'].min():.0f}, "
                 f"{int(fits_table['divergences'].sum())} divergences (a failed fit fails the run)", "no concern" if (fits_table["status"] == "no concern").all() else "concern"))
    rows.append(("No row on or after the change reaches its fit", "each change's fit is trained on rows dated before its cut, and the cut is on or before the change date (asserted for every change; a unit test plants future rows)", "no concern"))
    for outcome, frame in fits_table.groupby("outcome", sort=False):
        ratios = {name: float(frame[f"posterior_sd_over_prior_sd_{name}"].median()) for name in SETTING_BETAS}
        text = ", ".join(f"{name} {value:.2f}" for name, value in ratios.items())
        rows.append((f"Settings coefficients move from their prior ({OUTCOME_SHORT[outcome]})", f"median posterior SD over prior SD across fits: {text}; a ratio near 1 means the record barely informs that coefficient",
                     "concern" if max(ratios.values()) >= PRIOR_DOMINATED_RATIO else "no concern"))
    for outcome, frame in joined.groupby("outcome", sort=False):
        flagged = frame[frame["prior_dominated"]]
        rows.append((f"Predictions that rest on the prior alone ({OUTCOME_SHORT[outcome]})", f"{len(flagged)} of {len(frame)} changes move a setting whose coefficient the fit has not learnt (posterior SD over prior SD at least {PRIOR_DOMINATED_RATIO}); "
                     f"marked 'prior only' in Events" + (": " + ", ".join(map(str, flagged["change_date"])) if len(flagged) else ""), "concern" if len(flagged) else "no concern"))
    for group in rp.GROUPS:
        part = joined[joined["outcome"].eq(rp.PRIMARY_OUTCOME)]
        part = rp.group_rows(part, group)
        clusters = int(part["cluster_id"].nunique())
        rows.append((f"Clusters behind the interval ({group})", f"{len(part)} changes in {clusters} clusters; fewer than {rp.MIN_CLUSTERS} clusters make a cluster-bootstrap interval unreliable", "no concern" if clusters >= rp.MIN_CLUSTERS else "concern"))
    undefined = int(sum(tracking[f"{name}_undefined_resamples"].sum() for name in ("slope", "correlation", "direction_agreement")))
    rows.append(("Bootstrap resamples with an undefined statistic", f"{undefined} undefined resamples over every slope, correlation and agreement interval (a resample whose predicted changes do not vary has no slope)", "no concern" if undefined == 0 else "concern"))
    primary = joined[joined["outcome"].eq(rp.PRIMARY_OUTCOME)]
    covered = float(((primary["observed_window_pct"] >= primary["predictive_new_lower_5th"]) & (primary["observed_window_pct"] <= primary["predictive_new_upper_95th"])).mean())
    rows.append(("Predictive interval coverage under the settings in force (time in range)", f"{covered:.2f} of observed windows fall inside the model's 90% predictive interval (acceptable range {rp.COVERAGE_RANGE[0]:.2f} to {rp.COVERAGE_RANGE[1]:.2f})",
                 "no concern" if rp.COVERAGE_RANGE[0] <= covered <= rp.COVERAGE_RANGE[1] else "concern"))
    left_out = int(flow.set_index("stage")["changes"].iloc[0] - len(primary))
    rows.append(("Every change is accounted for", f"{len(primary)} predicted and {left_out} left out; Record_Issues holds {len(issues)} rows", "no concern" if left_out == len(issues) else "concern"))
    rows.append(("Early changes with little training data", f"changes with fewer than {min_train_days} usable training days are left out (listed in Record_Issues); the fits of the earliest included changes rest on the least data", "not assessed"))
    return pd.DataFrame(rows, columns=["check", "result", "status"])


def _warnings(note: str, draws: int, tune: int, chains: int) -> pd.DataFrame:
    rows = [
        ("Few independent changes", "Changes cluster in time (several within weeks of each other) and the isolated set is small, so the slope and its interval rest on a handful of clusters; the cluster bootstrap carries that into the interval and does not remove it. With about a dozen clusters a percentile cluster-bootstrap interval can be narrower than its nominal level."),
        ("Power", "The Power sheet simulates observed changes as a true slope times the real predicted changes plus noise with the real residual SD; a slope interval that holds zero is as consistent with predictions that carry information as with predictions that carry none when the power to find a true slope of 1 is low."),
        ("Same-hour model", "The model's settings act only within the same clock hour, so its predicted effect in the 4 hours after the changed hours is zero by construction, while the observed change over the affected hours includes those hours."),
        ("Later changes in the window", "The new settings are the profile in force after the change held for the 14 days; a later change inside the window is in the observed change and not in the prediction (changes with another change within 14 days are not isolated)."),
        ("Baseline days", "The replay's last-28-day predictor pools days that can be under earlier settings; the predicted change is measured against the settings actually in force on those days so that it is the same quantity as the observed change. The profile the day before the change is the sensitivity."),
        ("Observed change is noisy", "The observed change is one person's 14 days in a few hours of the day. Its sampling noise is about as large as the effects a setting change is expected to have, so a slope near zero can mean no tracking or too little signal to see it; the interval says which values the record rules out."),
        ("Placebo-adjusted comparator", "The observed change uses the last-28-day predictor and five matched no-change dates; it removes regression to the mean and season but not other things that changed with the settings (meals, activity, illness)."),
        ("Predictions use the settings that happened", "The new settings are the settings in force on each day of the window, which the person chose knowing the recent glucose. The model's coefficients are associations over the whole record, not the effect of changing a setting."),
        ("Shared fits", f"A change within {rp.FIT_SHARE_DAYS} days after the first change of its group uses that group's fit, which is up to {rp.FIT_SHARE_DAYS} days older than the change date; the old settings and exposures are still taken at the change's own date."),
        ("Settings held old", "The old settings are the profile at the last moment of the day before the change, held for the whole window; if an unrelated edit to another setting changed the same profile, it is part of the difference."),
        ("Sampler settings", f"{chains} chains, {tune} tuning steps, {draws} draws per chain at every cut; every fit passed the gate." + (f" Run note: {note}" if note else "")),
        ("Predictive claim", "A better prediction, or a predicted effect that matches, does not show that changing a setting would change the outcome."),
    ]
    return pd.DataFrame(rows, columns=["limitation", "detail"])


def _provenance(*, run_id: str, raw: str, table: pd.DataFrame, files: list[str], seconds: float, draws: int, tune: int, chains: int, command: str) -> pd.DataFrame:
    dirty = not working_tree_is_clean(REPO_ROOT)
    rows = [("run id", run_id), ("commit", current_commit(REPO_ROOT)), ("working tree clean", "no" if dirty else "yes"), ("raw data root", raw), ("record span", f"{table['date'].min().date()} to {table['date'].max().date()}"),
            ("seed", rp.SEED), ("bootstrap resamples", rp.BOOTSTRAP_DRAWS), ("predictive draws", rp.DRAWS), ("model sampler", f"{chains} chains, {tune} tuning, {draws} draws"), ("command", command),
            ("wall time (seconds)", round(seconds, 1)), ("files", json.dumps(files))]
    return pd.DataFrame(rows, columns=["item", "value"])


def _event_sheet(joined: pd.DataFrame) -> pd.DataFrame:
    columns = ["outcome", "event_id", "change_date", "change", "direction", "isolated", "cluster_id", "fit_cut", "training_days", "counted_days", "predicted_change_points", "predicted_lower_5th", "predicted_upper_95th", "predicted_change_day_before_points", "prior_sd_ratio_moved_settings", "prediction_basis",
               "observed_change_points", "inside_credible_interval", "inside_noise_range", "direction_agrees", "expected_new_pct", "expected_baseline_pct", "observed_window_pct",
               "predictive_new_lower_5th", "predictive_new_upper_95th", "crps_new_points", "crps_old_points", "crps_new_minus_old_points"]
    ordered = joined[columns].assign(_order=joined["outcome"].map({name: index for index, name in enumerate(rp.OUTCOMES)}))
    return ordered.sort_values(["_order", "change_date"]).drop(columns="_order")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _tracking_sentence(row: pd.Series) -> str:
    return (f"slope {_interval(row['slope'], row['slope_lower_5th'], row['slope_upper_95th'])}, correlation {_interval(row['correlation'], row['correlation_lower_5th'], row['correlation_upper_95th'])}, "
            f"direction agreement {row['direction_agreement']:.0%} (90% interval {row['direction_agreement_lower_5th']:.0%} to {row['direction_agreement_upper_95th']:.0%})")


def _label(outcome: str, group: str) -> str:
    return f"{OUTCOME_SHORT[outcome]}, {group}"


def verdict_lines(joined: pd.DataFrame, tracking: pd.DataFrame, information: pd.DataFrame, power: pd.DataFrame, sensitivity: pd.DataFrame) -> tuple[list[str], str]:
    """The headline bullets and the reading paragraph, every number and every group named from the tables (nothing about a result is written in advance)."""
    lines: list[str] = []
    estimated = tracking[tracking["slope"].notna()] if "slope" in tracking else tracking.iloc[0:0]
    missing = tracking[~tracking.index.isin(estimated.index)]
    above = estimated[estimated["slope_lower_5th"] > 0]
    below = estimated[estimated["slope_upper_95th"] < 0]
    holds = estimated[~estimated.index.isin(above.index) & ~estimated.index.isin(below.index)]
    lines.append(f"- **Slope intervals examined: {len(tracking)}** (each outcome and group). {len(holds)} hold zero, {len(below)} lie below zero, {len(above)} lie above zero, {len(missing)} could not be estimated.")
    for _, row in below.iterrows():
        lines.append(f"  - Below zero: {_label(row['outcome'], row['group'])} (slope {_interval(row['slope'], row['slope_lower_5th'], row['slope_upper_95th'])}): observed changes moved against the predicted direction.")
    for _, row in above.iterrows():
        lines.append(f"  - Above zero: {_label(row['outcome'], row['group'])} (slope {_interval(row['slope'], row['slope_lower_5th'], row['slope_upper_95th'])}).")
    for _, row in missing.iterrows():
        lines.append(f"  - Not estimated: {_label(row['outcome'], row['group'])}; {row['reading']}.")
    for group in rp.GROUPS:
        row = tracking[(tracking["outcome"] == rp.PRIMARY_OUTCOME) & (tracking["group"] == group)]
        if row.empty or pd.isna(row.iloc[0].get("slope")):
            continue
        row = row.iloc[0]
        lines.append(f"- **Time in range, {group} (n = {int(row['changes'])}, {int(row['clusters'])} clusters).** Mean predicted change {row['mean_predicted_points']:+.2f} points, mean observed change minus placebo {row['mean_observed_points']:+.2f}. "
                     f"{_tracking_sentence(row).capitalize()}. {row['reading'].capitalize()}.")
    worse, better = [], []
    for group in rp.GROUPS:
        row = information[(information["outcome"] == rp.PRIMARY_OUTCOME) & (information["group"] == group)]
        if row.empty or pd.isna(row.iloc[0].get("mean_difference_points")):
            lines.append(f"- **Settings information, time in range, {group}.** Not estimated.")
            continue
        row = row.iloc[0]
        (worse if row["lower_5th"] > 0 else better if row["upper_95th"] < 0 else []).append(group)
        lines.append(f"- **Settings information, time in range, {group}.** CRPS with the settings in force {row['mean_crps_new_points']:.2f} points, with the old settings {row['mean_crps_old_points']:.2f}; difference "
                     f"{_interval(row['mean_difference_points'], row['lower_5th'], row['upper_95th'])}. {row['reading'].capitalize()}.")
    low_power = []
    for _, row in power.iterrows():
        lines.append(f"- **Power, time in range, {row['group']}.** With the real predicted changes (SD {row['predicted_change_sd_points']:.2f} points) and the real residual noise (SD {row['noise_sd_points']:.2f} points), a true slope of 1 would give a slope interval above zero in "
                     f"{row['power_true_slope_1']:.0%} of simulations (slope 0.5: {row['power_true_slope_0.5']:.0%}; slope 2: {row['power_true_slope_2']:.0%}).")
        if row["power_true_slope_1"] < 0.8:
            low_power.append(row["group"])
    left_out = sensitivity[sensitivity["variant"].str.startswith("latest isolated")]
    if len(left_out):
        tir = joined[joined["outcome"].eq(rp.PRIMARY_OUTCOME)]
        latest = rp.latest_isolated(tir)
        row = left_out.iloc[0]
        full = tracking[(tracking["outcome"] == rp.PRIMARY_OUTCOME) & (tracking["group"] == "isolated changes")]
        with_it = f"{full.iloc[0]['slope']:+.2f}" if len(full) and pd.notna(full.iloc[0].get("slope")) else "not estimated"
        without = f"{_interval(row['slope'], row['slope_lower_5th'], row['slope_upper_95th'])}" if pd.notna(row.get("slope")) else "not estimated"
        lines.append(f"- **Latest isolated change ({latest['change_date']}; {int(latest['counted_days'])} counted days, predicted {latest['predicted_change_points']:+.2f}, observed {latest['observed_change_points']:+.2f} points).** "
                     f"Time-in-range isolated slope with it {with_it}, without it {without}.")
    if len(above):
        head = "Predicted changes line up with observed ones (slope interval above zero) for " + "; ".join(_label(r["outcome"], r["group"]) for _, r in above.iterrows())
    else:
        head = "The past changes do not validate the model's predicted settings effects: no slope interval lies above zero"
    parts = [head]
    if len(below):
        parts.append("for " + "; ".join(_label(r["outcome"], r["group"]) for _, r in below.iterrows()) + " the observed changes moved against the predicted direction")
    if worse:
        parts.append("for " + " and ".join(worse) + " the old settings predicted the 14-day window better than the settings in force")
    if better:
        parts.append("for " + " and ".join(better) + " the settings in force predicted the window better than the old settings")
    reading = "; ".join(parts) + "."
    if low_power:
        reading += (" The test has low power (a true slope of 1 would be found in fewer than 80% of simulations for " + " and ".join(low_power) + "), so a slope interval that holds zero is uninformative about whether the predictions carry real signal; "
                    "it is not evidence that they carry none.")
    return lines, reading


def write_summary(path: Path, *, run_id: str, joined: pd.DataFrame, tracking: pd.DataFrame, information: pd.DataFrame, sensitivity: pd.DataFrame, power: pd.DataFrame, flow: pd.DataFrame, fits_table: pd.DataFrame,
                  outcomes: tuple[str, ...], draws: int, tune: int, chains: int, min_train_days: int, note: str) -> None:
    primary = joined[joined["outcome"].eq(rp.PRIMARY_OUTCOME)].sort_values("change_date")
    stage = flow.set_index("stage")["changes"]
    bullets, reading = verdict_lines(joined, tracking, information, power, sensitivity)
    lines = ["# Replay as predictor: do the model's predicted effects of past settings changes match what happened?", "",
             f"Run `{run_id}`. {len(primary)} of {int(stage.iloc[0])} changes in the record could be predicted ({int(primary['isolated'].sum())} isolated, no other change within 14 days; {primary['cluster_id'].nunique()} clusters of changes). "
             f"Each change's model was fitted only to days before it; effects are for the replay's affected hours on the 14 days after the change, in percentage points.", "", "## Verdict", "", *bullets, "",
             f"**Reading.** {reading} These are associations in one person's record, not effects of a setting.", "", "## Do predicted changes track observed changes?", "",
             "Predicted change: the profile after the change against the settings actually in force on the replay's baseline days. Observed change: the replay's change minus placebo change. Slope: percentage points observed per percentage point predicted (1 = size right, 0 = no information).", "",
             "| Outcome | Group | Changes (clusters) | Slope (90% interval) | Correlation (90% interval) | Direction agrees (90% interval) | Observed inside credible interval | Inside predicted change plus no-change noise |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for _, row in tracking.iterrows():
        if pd.isna(row.get("slope")):
            lines.append(f"| {OUTCOME_SHORT[row['outcome']]} | {row['group']} | {int(row['changes'])} ({int(row['clusters'])}) | not estimated | | | | |")
            continue
        lines.append(f"| {OUTCOME_SHORT[row['outcome']]} | {row['group']} | {int(row['changes'])} ({int(row['clusters'])}) | {_short(row['slope'], row['slope_lower_5th'], row['slope_upper_95th'])} | "
                     f"{_short(row['correlation'], row['correlation_lower_5th'], row['correlation_upper_95th'])} | {row['direction_agreement']:.0%} ({row['direction_agreement_lower_5th']:.0%} to {row['direction_agreement_upper_95th']:.0%}) | "
                     f"{row['inside_credible_interval']:.0%} | {row['inside_noise_range']:.0%} |")
    lines += ["", "### Sensitivity", "", "| Variation | Outcome | Group | Changes (clusters) | Slope (90% interval) |", "| --- | --- | --- | ---: | ---: |"]
    for _, row in sensitivity.iterrows():
        slope_text = "not estimated" if pd.isna(row.get("slope")) else _short(row["slope"], row["slope_lower_5th"], row["slope_upper_95th"])
        lines.append(f"| {row['variant']} | {OUTCOME_SHORT[row['outcome']]} | {row['group']} | {int(row['changes'])} ({int(row['clusters'])}) | {slope_text} |")
    lines += ["", "## Do the settings in force predict the window better than the old settings?", "",
              "Old settings: the profile in force the day before the change, held for the 14 days. CRPS in percentage points of the 14-day share; difference = new minus old, so negative favours the settings in force.", "",
              "| Outcome | Group | Changes (clusters) | CRPS new | CRPS old | Difference (90% interval) | Changes where new is better |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for _, row in information.iterrows():
        if pd.isna(row.get("mean_difference_points")):
            lines.append(f"| {OUTCOME_SHORT[row['outcome']]} | {row['group']} | {int(row['changes'])} ({int(row['clusters'])}) | not estimated | | | |")
            continue
        lines.append(f"| {OUTCOME_SHORT[row['outcome']]} | {row['group']} | {int(row['changes'])} ({int(row['clusters'])}) | {row['mean_crps_new_points']:.2f} | {row['mean_crps_old_points']:.2f} | "
                     f"{_short(row['mean_difference_points'], row['lower_5th'], row['upper_95th'])} | {row['share_changes_new_better']:.0%} |")
    lines += ["", "## Each change (time in range)", "", "Settings are in the units the pump uses, as the mean over the changed clock hours where the setting differs. Predicted: model, new settings against the baseline days' settings. Observed: change minus placebo change. "
              "'Prior only': a setting the change moves has a coefficient the fit has not learnt.", "",
              "| Date | Change | Isolated | Predicted change, points (90% interval) | Observed change, points | Basis |", "| --- | --- | --- | ---: | ---: | --- |"]
    for row in primary.itertuples():
        lines.append(f"| {row.change_date} | {row.change} | {'yes' if row.isolated else 'no'} | {_short(row.predicted_change_points, row.predicted_lower_5th, row.predicted_upper_95th)} | {row.observed_change_points:+.2f} | {row.prediction_basis} |")
    lines += ["", "Details, the per-change rows for every outcome, the power simulation, the sampler record of every fit, the checks and the changes left out are in `diagnostics.xlsx`.", "", "## Run notes", "",
              f"{len(fits_table)} fits ({chains} chains, {tune} tuning steps, {draws} draws per chain); every fit passed the sampler gate (R-hat below {om.R_HAT_LIMIT}, ESS above {om.ESS_LIMIT}, no divergences). "
              f"Changes with fewer than {min_train_days} days of usable training data are left out."]
    if note:
        lines.append(note)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def write_outputs(*, folder: Path, run_id: str, raw: str, table: pd.DataFrame, flow: pd.DataFrame, issues: pd.DataFrame, joined: pd.DataFrame, targets: list[rp.EventTarget], posteriors: dict, replay_results: dict,
                  draws: int, tune: int, chains: int, min_train_days: int, seconds: float, command: str, note: str) -> None:
    folder = Path(folder)
    outcomes = tuple(name for name in rp.OUTCOMES if name in posteriors)
    tracking = rp.tracking_table(joined)
    information = rp.information_table(joined)
    sensitivity = rp.sensitivity_table(joined)
    power = rp.power_table(joined)
    fits_table = _model_fits(posteriors, targets)
    figures = write_figure(joined, information, tracking, replay_results["scores"], folder)
    joined.to_csv(folder / "replay_predictor_events.csv", index=False)
    files = ["replay_predictor_events.csv", "replay_predictor_summary.md", *[path.name for path in figures]]
    sheets = {
        "Specification_Coding": _specification(outcomes, draws, tune, chains, min_train_days),
        "Denominators_Flow": _flow(flow, joined, posteriors, targets, outcomes),
        "Events": _event_sheet(joined),
        "Tracking": tracking,
        "Sensitivity": sensitivity,
        "Settings_Information": information,
        "Power": power,
        "Model_Fits": fits_table,
        "Checks": _checks(joined, tracking, fits_table, flow, issues, min_train_days),
        "Record_Issues": issues,
        "Warnings_Limitations": _warnings(note, draws, tune, chains),
        "Artifacts_Provenance": _provenance(run_id=run_id, raw=raw, table=table, files=[*files, "diagnostics.xlsx"], seconds=seconds, draws=draws, tune=tune, chains=chains, command=command),
    }
    path = write_workbook(folder / "diagnostics.xlsx", sheets)
    write_summary(folder / "replay_predictor_summary.md", run_id=run_id, joined=joined, tracking=tracking, information=information, sensitivity=sensitivity, power=power, flow=flow, fits_table=fits_table,
                  outcomes=outcomes, draws=draws, tune=tune, chains=chains, min_train_days=min_train_days, note=note)
    reconcile(path, joined, tracking, information, flow, issues, fits_table, power)


def reconcile(path: Path, joined: pd.DataFrame, tracking: pd.DataFrame, information: pd.DataFrame, flow: pd.DataFrame, issues: pd.DataFrame, fits_table: pd.DataFrame, power: pd.DataFrame) -> None:
    """Read the workbook back and compare it with the tables it summarises."""
    workbook = read_workbook(path)
    stored_flow = workbook["Denominators_Flow"].set_index("stage")["n"]
    primary = joined[joined["outcome"].eq(rp.PRIMARY_OUTCOME)]
    if int(stored_flow["predicted (after the fits)"]) != len(primary) or int(stored_flow.iloc[0]) != int(flow["changes"].iloc[0]):
        raise RuntimeError("Workbook flow differs from the predicted changes")
    if len(workbook["Record_Issues"]) != int(flow["changes"].iloc[0]) - len(primary):
        raise RuntimeError("Record_Issues rows do not reconcile with the flow")
    if len(workbook["Events"]) != len(joined) or len(workbook["Model_Fits"]) != len(fits_table) or len(workbook["Power"]) != len(power):
        raise RuntimeError("Workbook rows differ from the result tables")
    stored = workbook["Tracking"]
    if not np.allclose(stored["slope"].to_numpy(dtype=float), tracking["slope"].to_numpy(dtype=float), atol=1e-3, equal_nan=True) or list(stored["reading"]) != list(tracking["reading"]):
        raise RuntimeError("Workbook tracking differs from the tracking table")
    if not np.allclose(workbook["Settings_Information"]["mean_difference_points"].to_numpy(dtype=float), information["mean_difference_points"].to_numpy(dtype=float), atol=1e-3, equal_nan=True):
        raise RuntimeError("Workbook settings information differs from the table")
    events = workbook["Events"].query("outcome == 'tir'")
    rederived = events["crps_new_points"] - events["crps_old_points"]
    if not np.allclose(rederived.to_numpy(dtype=float), events["crps_new_minus_old_points"].to_numpy(dtype=float), atol=1e-3):
        raise RuntimeError("Workbook CRPS differences are not new minus old")
    mean_check = events.merge(primary[["event_id", "predicted_change_points"]], on="event_id", suffixes=("", "_table"))
    if not np.allclose(mean_check["predicted_change_points"].to_numpy(dtype=float), mean_check["predicted_change_points_table"].to_numpy(), atol=1e-3):
        raise RuntimeError("Workbook predicted changes differ from the table")
