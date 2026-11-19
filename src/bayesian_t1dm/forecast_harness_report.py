"""Tables, figure, summary and diagnostics workbook of the forecast harness (``forecast_harness``)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import forecast_harness as fh
from . import outcomes_model as om
from . import win_test as wt
from .diagnostics_workbook import read_workbook, write_workbook
from .figures import NEUTRAL_LINE, PREDICTOR_STYLE, assert_ascii
from .publish import current_commit, working_tree_is_clean
from .sampler_health import sampler_gate

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTCOME_LABELS = {"tir": "time in range (70-180 mg/dL)", "below70": "time below 70 mg/dL", "below54": "time below 54 mg/dL"}
BLOCKS = (4, 2, 8)
PRIMARY_BLOCK = 4
MODEL = "hour-day model"
PLANNED = "hour-day model (planned settings)"
CANDIDATES = (MODEL, PLANNED, "ridge")
BASELINE_COMPARISONS = ("recency-weighted", "last 14 days")
POWER_PAIR = ("last 14 days", fh.REFERENCE)
PIT_BINS = 10
COVERAGE_RANGE = (0.80, 0.98)
FIGURE_FILES = ("forecast_harness_figure.png", "forecast_harness_figure.svg")


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------

def crps_by_origin(scores: pd.DataFrame) -> pd.DataFrame:
    """CRPS (percentage points) of every predictor by origin in time order; NaN where the predictor did not score that origin."""
    scored = scores[scores["reason"] == ""]
    return scored.pivot(index="origin", columns="predictor", values="crps_points").sort_index()


def paired_differences(table: pd.DataFrame, candidate: str, reference: str) -> pd.Series:
    """Candidate minus reference CRPS at every origin both scored, in time order (negative favours the candidate)."""
    if candidate not in table or reference not in table:
        return pd.Series(dtype=float)
    return (table[candidate] - table[reference]).dropna()


def comparison_status(candidate: str, result: wt.PairedInterval) -> tuple[str, str]:
    """Diagnostics status and reason of one comparison: a candidate is "no concern" only when it wins, a tie or loss is a "concern"; baseline comparisons are not candidates."""
    if candidate in BASELINE_COMPARISONS:
        return "not applicable", "baseline comparison reported for the reader; a baseline is not a candidate for the win"
    reading = {"win": "the interval is entirely below zero", "tie": "the interval holds zero, so no win is shown", "loss": "the interval is entirely above zero"}[result.verdict]
    return ("no concern" if result.verdict == "win" else "concern"), reading


def build_comparisons(table: pd.DataFrame) -> pd.DataFrame:
    """Paired CRPS differences against the last-28-day baseline (and the model against the other predictors) with their block-bootstrap intervals, for every block length."""
    pairs = [(candidate, fh.REFERENCE) for candidate in (*CANDIDATES, *BASELINE_COMPARISONS)] + [(MODEL, other) for other in ("last 14 days", "recency-weighted", "ridge")]
    rows = []
    for candidate, reference in pairs:
        difference = paired_differences(table, candidate, reference)
        if difference.empty:
            continue
        for block in BLOCKS:
            result = wt.paired_interval(difference.to_numpy(), block=block)
            status, reason = comparison_status(candidate, result) if reference == fh.REFERENCE else ("not applicable", "secondary pairwise comparison; the win test is against the last-28-day baseline")
            role = "win test" if (candidate, reference, block) == (MODEL, fh.REFERENCE, PRIMARY_BLOCK) else ("power check" if (candidate, reference) == POWER_PAIR and block == PRIMARY_BLOCK
                                                                                                                    else ("sensitivity (block length)" if block != PRIMARY_BLOCK else "secondary"))
            rows.append({"candidate": candidate, "reference": reference, "role": role, "block_length_origins": block, "origins": result.origins, "mean_difference_points": result.mean,
                         "lower_5th_points": result.lower, "upper_95th_points": result.upper, "verdict": result.verdict, "status": status, "reason": reason})
    return pd.DataFrame(rows)


def power_check(comparisons: pd.DataFrame) -> dict[str, float]:
    """The smallest difference the paired origins can resolve: half the width of the 90% interval of last-14 minus last-28."""
    row = comparisons[(comparisons["candidate"] == POWER_PAIR[0]) & (comparisons["reference"] == POWER_PAIR[1]) & (comparisons["block_length_origins"] == PRIMARY_BLOCK)].iloc[0]
    return {"half_width_points": float((row["upper_95th_points"] - row["lower_5th_points"]) / 2.0), "mean_difference_points": float(row["mean_difference_points"]), "origins": int(row["origins"])}


def predictor_summary(scores: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Per predictor: origins scored, mean CRPS, mean absolute error of the median, 90% interval coverage, the share of origins better than last 28 days, and the PIT histogram."""
    scored = scores[scores["reason"] == ""]
    rows = []
    for name in fh.PREDICTORS:
        part = scored[scored["predictor"] == name]
        if part.empty:
            continue
        better = float((paired_differences(table, name, fh.REFERENCE) < 0).mean()) if name != fh.REFERENCE else float("nan")
        coverage = float(part["inside_90"].mean())
        histogram = np.histogram(part["pit"], bins=np.linspace(0, 1, PIT_BINS + 1))[0]
        row = {"predictor": name, "origins_scored": len(part), "mean_crps_points": float(part["crps_points"].mean()), "mae_of_median_points": float((part["median_pct"] - part["observed_pct"]).abs().mean()),
               "coverage_90": coverage, "share_origins_better_than_last_28": better}
        row.update({f"pit_{k / PIT_BINS:.1f}_to_{(k + 1) / PIT_BINS:.1f}": int(histogram[k]) for k in range(PIT_BINS)})
        row["status"] = "no concern" if COVERAGE_RANGE[0] <= coverage <= COVERAGE_RANGE[1] else "concern"
        row["reason"] = f"90% interval held the observed share at {coverage:.0%} of origins; read as roughly calibrated between {COVERAGE_RANGE[0]:.0%} and {COVERAGE_RANGE[1]:.0%} (windows overlap, so this is approximate)"
        rows.append(row)
    return pd.DataFrame(rows)


def settings_change_table(origins: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Windows with and without a real settings change: how many, and the mean CRPS of the frozen-settings and planned-settings forecasts in each."""
    table = crps_by_origin(scores)
    rows = []
    for label, flag in (("settings changed inside the window", True), ("settings unchanged", False)):
        selected = origins[origins["scored"] & (origins["settings_changed"] == flag)]["origin"]
        row = {"windows": label, "origins": int(len(selected))}
        for name in (fh.REFERENCE, MODEL, PLANNED):
            row[f"mean_crps_{name}_points"] = float(table.reindex(selected)[name].mean()) if name in table and len(selected) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def write_figure(table: pd.DataFrame, comparisons: pd.DataFrame, outcome: str, output: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    shown = [name for name in (MODEL, PLANNED, "ridge", "recency-weighted") if name in table]
    figure, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.8), gridspec_kw={"width_ratios": [2.3, 1.0], "wspace": 0.42})
    reference_label = PREDICTOR_STYLE[fh.REFERENCE].label.lower()
    for name in shown:
        style = PREDICTOR_STYLE[name]
        difference = paired_differences(table, name, fh.REFERENCE)
        left.plot(difference.index, difference.to_numpy(), color=style.color, linestyle=style.linestyle, marker=style.marker, markersize=4.5, linewidth=1.8, label=style.label)
    primary = comparisons[(comparisons["candidate"] == MODEL) & (comparisons["reference"] == fh.REFERENCE) & (comparisons["block_length_origins"] == PRIMARY_BLOCK)]
    if not primary.empty:
        row = primary.iloc[0]
        left.axhspan(row["lower_5th_points"], row["upper_95th_points"], color=PREDICTOR_STYLE[MODEL].color, alpha=0.16, linewidth=0,
                     label=f"{PREDICTOR_STYLE[MODEL].label}: 90% interval of its mean ({row['mean_difference_points']:+.2f})")
    left.axhline(0, color=NEUTRAL_LINE, linewidth=1.2)
    left.set_ylabel(f"CRPS minus CRPS of {reference_label}\n(percentage points; below 0 = better)", fontsize=10)
    left.set_xlabel("First forecast day of the 14-day window (weekly origins)", fontsize=10)
    left.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    left.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=2)
    left.set_title("Each dot is one 14-day window", fontsize=11, loc="left")
    left.tick_params(labelsize=9)
    labels = []
    ticks = []
    for index, name in enumerate([n for n in (MODEL, PLANNED, "ridge", "recency-weighted", "last 14 days") if ((comparisons["candidate"] == n) & (comparisons["reference"] == fh.REFERENCE)).any()]):
        row = comparisons[(comparisons["candidate"] == name) & (comparisons["reference"] == fh.REFERENCE) & (comparisons["block_length_origins"] == PRIMARY_BLOCK)].iloc[0]
        style = PREDICTOR_STYLE[name]
        right.errorbar(row["mean_difference_points"], -index, xerr=[[row["mean_difference_points"] - row["lower_5th_points"]], [row["upper_95th_points"] - row["mean_difference_points"]]],
                       fmt=style.marker, color=style.color, markersize=7, linewidth=2.2, capsize=4)
        ticks.append(-index)
        labels.append(style.label.replace(", planned settings", ",\nplanned settings"))
    right.axvline(0, color=NEUTRAL_LINE, linewidth=1.2)
    right.set_yticks(ticks)
    right.set_yticklabels(labels, fontsize=9)
    right.set_xlabel("Mean CRPS difference, 90% interval\n(percentage points; below 0 = better)", fontsize=10)
    right.set_title("Win test needs the interval below 0", fontsize=11, loc="left")
    right.tick_params(labelsize=9)
    title = f"Forecasts of 14-day {OUTCOME_LABELS[outcome]}: score against the last-28-day baseline"
    note = ("CRPS scores a whole predictive distribution of the 14-day share (lower is better). Windows overlap by 7 days; intervals are circular block bootstrap, block of 4 origins,\n"
            "10,000 resamples, over the weekly origins that every compared predictor scored.")
    assert_ascii(title, note, *(style.label for style in PREDICTOR_STYLE.values()), *labels)
    figure.suptitle(title, fontsize=12.5, x=0.01, ha="left", y=0.995)
    figure.text(0.01, 0.935, note, fontsize=9, ha="left", va="top", color=NEUTRAL_LINE)
    figure.subplots_adjust(top=0.80, bottom=0.14, left=0.08, right=0.985)
    paths = []
    for name in FIGURE_FILES:
        path = output / name
        figure.savefig(path, dpi=300)
        paths.append(path)
    plt.close(figure)
    return paths


# ---------------------------------------------------------------------------
# Sheets
# ---------------------------------------------------------------------------

def _specification(outcome: str, draws: int, tune: int, chains: int, score_model: bool) -> pd.DataFrame:
    rows = [
        ("Question", f"Do forecasts of the next 14 days' {OUTCOME_LABELS[outcome]} improve on the last 28 days when they use the settings in force, scored at weekly origins?"),
        ("Claim type", "Predictive. Nothing here says what a settings change would do."),
        ("Unit of analysis", "One 14-day window starting at a weekly origin (the first forecast day); windows overlap by 7 days."),
        ("Outcome / estimand", f"Share of valid CGM readings in the outcome over the 14 days (readings-weighted, every hour included); {OUTCOME_LABELS[outcome]}."),
        ("Information at the origin", "Everything dated before 00:00 of the first forecast day, and the settings profile frozen at the last moment of the day before. Not known: carbohydrate, exercise, sleep mode, site age, context events inside the window."),
        ("Evaluation origins", f"{fh.EVALUATION_ORIGIN_COUNT} weekly origins, the last window ending on the last CGM day; a window with under {fh.MIN_WINDOW_COVERAGE:.0%} of its possible readings is not scored."),
        ("Development origins", f"Every second week from {fh.DEVELOPMENT_START.date()} to {fh.DEVELOPMENT_END.date()}; used for model choices only, never for the win test."),
        ("Predictors", "last 28 days (prespecified reference), last 14 days, recency-weighted (half-life 14 days over 90 days), ridge comparator, " + ("current hour-day model refitted at every origin, and its planned-settings variant." if score_model else "(hour-day model not scored in this run).")),
        ("Spread of baselines", "Each baseline's own past errors, logit(observed) - logit(point), at every earlier daily origin whose 14-day window ended before the origin, resampled to 4,000 draws."),
        ("Score", "CRPS of 4,000 predictive draws of the share, in percentage points (lower is better)."),
        ("Effect measure and direction", "Difference = CRPS of the candidate minus CRPS of the last-28-day baseline at the same origin, in percentage points. Negative means the candidate forecasts better."),
        ("Win rule", f"Circular block bootstrap of the mean difference over the scored origins in time order (block {PRIMARY_BLOCK} origins, {wt.BOOTSTRAP_RESAMPLES:,} resamples, seed {wt.BOOTSTRAP_SEED}); a win needs the 90% percentile interval entirely below 0. Blocks 2 and 8 are sensitivity."),
        ("Power check", "Half the width of the 90% interval of last-14-day minus last-28-day: the smallest mean difference the origins can separate from 0 (it uses no model)."),
        ("Model sampler", f"{chains} chains, {tune} tuning and {draws} draws each, target acceptance {om.TARGET_ACCEPT}; every refit must pass the sampler gate or the run fails." if score_model else "not applicable (model not scored)"),
        ("How to read it", "A mean difference of -0.50 with interval -0.90 to -0.10 means the candidate's forecast distribution is about half a percentage point closer to what happened, and the 52 windows are enough to say it beats the baseline. An interval of -0.50 to +0.30 is a tie: the data cannot tell them apart."),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def _flow(origin_tables: dict[str, pd.DataFrame], scores: dict[str, pd.DataFrame], score_model: bool) -> pd.DataFrame:
    rows = []
    for set_name, origins in origin_tables.items():
        scored = origins[origins["scored"]]
        rows.append((set_name, "weekly origins" if set_name == "evaluation" else "development origins", len(origins)))
        rows.append((set_name, f"windows with at least {fh.MIN_WINDOW_COVERAGE:.0%} of their possible readings", len(scored)))
        table = scores[set_name]
        for name in fh.PREDICTORS:
            if name.startswith("hour-day") and (not score_model or set_name != "evaluation"):
                continue
            rows.append((set_name, f"scored by {name}", int(((table["predictor"] == name) & (table["reason"] == "")).sum())))
    return pd.DataFrame(rows, columns=["set", "stage", "n"])


def _origins_sheet(origin_tables: dict[str, pd.DataFrame], scores: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for set_name, origins in origin_tables.items():
        wide = crps_by_origin(scores[set_name])
        observed = scores[set_name].drop_duplicates("origin").set_index("origin")["observed_pct"] if "observed_pct" in scores[set_name] else pd.Series(dtype=float)
        frame = origins.assign(observed_pct=origins["origin"].map(observed))
        for name in fh.PREDICTORS:
            if name in wide:
                frame[f"crps_{name}_points"] = frame["origin"].map(wide[name])
        frames.append(frame)
    sheet = pd.concat(frames, ignore_index=True)
    reasons = pd.concat([scores[name][scores[name]["reason"] != ""].assign(set=name) for name in scores], ignore_index=True)
    sheet["predictors_not_scored"] = [("; ".join(reasons[(reasons["set"] == row.set) & (reasons["origin"] == row.origin)]["reason"])) for row in sheet.itertuples()]
    sheet["origin"] = sheet["origin"].dt.date
    sheet["window_end"] = sheet["window_end"].dt.date
    return sheet.rename(columns={"coverage": "window_readings_share_of_possible", "scored": "window_scored", "reason": "window_not_scored_because", "frozen_profile_known": "frozen_profile_complete",
                                 "settings_changed": "settings_changed_inside_window", "hours_with_unknown_settings": "window_hours_with_unknown_settings", "observed_pct": "observed_share_pct"})


def _model_fits(posteriors: dict | None) -> pd.DataFrame:
    rows = []
    for origin, posterior in (posteriors or {}).items():
        d = posterior.diagnostics
        rows.append({"origin": origin.date(), "training_rows": posterior.rows, "training_days": posterior.days, "chains": d.chains, "tuning_steps": d.tune, "draws_per_chain": d.draws, "r_hat_max": round(d.rhat_max, 4),
                     "bulk_ess_min": round(d.ess_bulk_min), "tail_ess_min": round(d.ess_tail_min), "divergences": d.divergences, "transitions_at_max_tree_depth": d.max_tree_depth_hits,
                     "sampling_seconds": round(posterior.timing["sample_seconds"], 1), "status": "no concern" if sampler_gate(d) else "concern",
                     "reason": f"R-hat below {om.R_HAT_LIMIT}, bulk and tail ESS above {om.ESS_LIMIT}, no divergences"})
    return pd.DataFrame(rows)


def _checks(*, evaluation: pd.DataFrame, evaluation_scores: pd.DataFrame, comparisons: pd.DataFrame, summary: pd.DataFrame, power: dict, posteriors: dict | None, ridge: pd.DataFrame, score_model: bool) -> pd.DataFrame:
    scored = evaluation[evaluation["scored"]]
    changed = int(scored["settings_changed"].sum())
    unknown = int((scored["hours_with_unknown_settings"] > 0).sum())
    missing = evaluation_scores[evaluation_scores["reason"] != ""]
    rows = [
        ("Origin set", f"{len(evaluation)} weekly origins {evaluation['origin'].min().date()} to {evaluation['origin'].max().date()}; the last window ends {evaluation['window_end'].max().date()}, the last CGM day; windows overlap by 7 days, so the origins are not independent", "no concern"),
        ("Windows with enough readings", f"{len(scored)} of {len(evaluation)} windows hold at least {fh.MIN_WINDOW_COVERAGE:.0%} of their possible readings" + ("" if len(scored) == len(evaluation) else "; the rest are listed in Origins"),
         "no concern" if len(scored) == len(evaluation) else "concern"),
        ("Information at the origin", "At every origin the table, daily counts, doses and profiles handed to a predictor were checked to hold nothing dated on or after the first forecast day; the sentinel test is in the test suite", "no concern"),
        ("Frozen settings assumption", f"{changed} of {len(scored)} scored windows contain a real settings change (the frozen profile is wrong for part of them); the planned-settings variant feeds the settings actually in force; see Settings_Changes", "concern" if changed else "no concern"),
        ("Hours with unknown settings in the windows", f"{unknown} scored windows hold hours with readings whose settings the archive cannot show; these hours stay in the target; the planned-settings variant is not scored for them", "concern" if unknown else "no concern"),
        ("Predictors not scored at a scored window", "none" if missing.empty else f"{len(missing)} predictor-origin pairs: " + "; ".join(sorted(set(missing['reason']))), "no concern" if missing.empty else "concern"),
        ("Power", f"smallest difference the {power['origins']} origins separate from 0 about {power['half_width_points']:.2f} points (half-width of last-14 minus last-28; last-14 minus last-28 is {power['mean_difference_points']:+.2f}); it comes from two similar baselines, so a model's differences may vary more", "not applicable"),
        ("Spread of ridge predictions", "ridge residual pool is the last 120 out-of-fold residuals of the training windows; see Ridge_Fits and the coverage in Predictor_Scores", "not applicable" if ridge.empty else "no concern"),
        ("Bootstrap", f"circular block bootstrap, {wt.BOOTSTRAP_RESAMPLES:,} resamples, seed {wt.BOOTSTRAP_SEED}, blocks 4 (primary), 2 and 8 (sensitivity)", "no concern"),
    ]
    if score_model:
        rows.append(("Sampler gate on every refit", f"{len(posteriors or {})} fits, all passed (a failed fit fails the run)", "no concern" if posteriors else "not assessed"))
        primary = comparisons[(comparisons["role"] == "win test")]
        if not primary.empty:
            row = primary.iloc[0]
            rows.append(("Win test", wt.verdict_text(wt.PairedInterval(row["mean_difference_points"], row["lower_5th_points"], row["upper_95th_points"], int(row["origins"]), PRIMARY_BLOCK, row["verdict"])), row["status"]))
    else:
        rows.append(("Sampler gate on every refit", "the hour-day model was not scored in this run", "not assessed"))
        rows.append(("Win test", "the hour-day model was not scored in this run", "not assessed"))
    return pd.DataFrame(rows, columns=["check", "result", "status"])


def _warnings() -> pd.DataFrame:
    rows = [
        ("Overlapping windows", "Weekly windows of 14 days overlap by 7 days, so 52 origins hold about 13 independent blocks; the block bootstrap carries the dependence, but a window-by-window difference is not an independent trial."),
        ("Frozen settings", "The forecast uses the settings profile in force at the origin for all 14 days. Where the pump's settings changed inside a window the frozen forecast is wrong by construction; the planned-settings variant shows how much of the result depends on that."),
        ("Model counts", "The hour-day model simulates counts with the realised number of readings per hour. The number of readings is not glucose information, but it is not known at the origin."),
        ("Resampled exposures", "Carbohydrate, exercise, sleep mode and site age in the window are not known at the origin; each model draw uses 14 days resampled from the 28 days before it, so the model's use of them is an average pattern."),
        ("Baseline spread", "Each baseline's spread is learnt from its own errors at earlier daily origins, which overlap heavily; the pool holds hundreds of residuals but few independent ones."),
        ("Ridge", "The ridge comparator has about a dozen features and is fitted to overlapping windows; its spread uses the last 120 out-of-fold residuals, which can run narrower than its future errors."),
        ("Regime shift", "A recent change in level (the latest windows are below the earlier range) is missed by every predictor for several origins; the bootstrap includes those origins."),
        ("Predictive claim", "A better forecast does not show that changing a setting would change the outcome."),
    ]
    return pd.DataFrame(rows, columns=["limitation", "detail"])


def _provenance(*, run_id: str, raw: str, outcome: str, table: pd.DataFrame, files: list[str], seconds: float, draws: int, tune: int, chains: int, score_model: bool) -> pd.DataFrame:
    dirty = not working_tree_is_clean(REPO_ROOT)
    rows = [("run id", run_id), ("commit", current_commit(REPO_ROOT)), ("working tree clean", "no" if dirty else "yes"), ("raw data root", raw), ("record span", f"{table['date'].min().date()} to {table['date'].max().date()}"),
            ("outcome", outcome), ("seed", wt.BOOTSTRAP_SEED), ("bootstrap resamples", wt.BOOTSTRAP_RESAMPLES), ("predictive draws", fh.DRAWS),
            ("model sampler", f"{chains} chains, {tune} tuning, {draws} draws" if score_model else "not scored"),
            ("command", f"python -m bayesian_t1dm.forecast_harness --outcome {outcome}" + ("" if score_model else " --skip-model")), ("wall time (seconds)", round(seconds, 1)), ("files", json.dumps(files))]
    return pd.DataFrame(rows, columns=["item", "value"])


def _development(scores: pd.DataFrame, origins: pd.DataFrame) -> pd.DataFrame:
    table = crps_by_origin(scores)
    rows = []
    for name in (fh.REFERENCE, "last 14 days", "recency-weighted", "ridge"):
        if name not in table:
            continue
        difference = paired_differences(table, name, fh.REFERENCE)
        row = {"predictor": name, "origins": len(table[name].dropna()), "mean_crps_points": float(table[name].mean())}
        if name != fh.REFERENCE and len(difference) >= PRIMARY_BLOCK:
            result = wt.paired_interval(difference.to_numpy(), block=min(PRIMARY_BLOCK, len(difference)))
            row.update({"mean_difference_vs_last_28_points": result.mean, "lower_5th_points": result.lower, "upper_95th_points": result.upper})
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary(path: Path, *, run_id: str, outcome: str, evaluation: pd.DataFrame, summary: pd.DataFrame, comparisons: pd.DataFrame, power: dict, score_model: bool) -> None:
    primary = comparisons[comparisons["role"] == "win test"]
    scored = evaluation[evaluation["scored"]]
    lines = [f"# Forecast harness: {OUTCOME_LABELS[outcome]} over the next 14 days", "", f"Run `{run_id}`. Weekly origins {evaluation['origin'].min().date()} to {evaluation['origin'].max().date()}; {len(scored)} of {len(evaluation)} windows had enough readings to score.", "", "## Headline", ""]
    if score_model and not primary.empty:
        row = primary.iloc[0]
        base = summary.set_index("predictor")
        verdict = {"win": "a win: the whole interval is below zero", "tie": "a tie: the interval holds zero, so the model is not shown to forecast better", "loss": "a loss: the whole interval is above zero"}[row["verdict"]]
        lines.append(f"The current hour-day model's CRPS was {base.loc[MODEL, 'mean_crps_points']:.2f} percentage points against {base.loc[fh.REFERENCE, 'mean_crps_points']:.2f} for the last 28 days. "
                     f"Model minus last 28 days: {row['mean_difference_points']:+.2f} points (90% interval {row['lower_5th_points']:+.2f} to {row['upper_95th_points']:+.2f}), {verdict}.")
    else:
        lines.append("The hour-day model was not scored in this run; the table below is the baselines and the ridge comparator only.")
    lines += ["", f"Power check: the smallest difference these {power['origins']} origins separate from zero is about {power['half_width_points']:.2f} percentage points (last 14 days minus last 28 days: {power['mean_difference_points']:+.2f}).", "",
              "## Scores", "", "| Predictor | Windows | Mean CRPS (points) | Median error (points) | 90% interval held | Better than last 28 days |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples():
        better = "-" if row.predictor == fh.REFERENCE else f"{row.share_origins_better_than_last_28:.0%}"
        lines.append(f"| {PREDICTOR_STYLE[row.predictor].label} | {row.origins_scored} | {row.mean_crps_points:.2f} | {row.mae_of_median_points:.2f} | {row.coverage_90:.0%} | {better} |")
    lines += ["", "## Paired differences against last 28 days (block of 4 origins)", "", "| Predictor | Mean difference (points) | 90% interval | Verdict |", "| --- | ---: | ---: | --- |"]
    for row in comparisons[(comparisons["reference"] == fh.REFERENCE) & (comparisons["block_length_origins"] == PRIMARY_BLOCK)].itertuples():
        lines.append(f"| {PREDICTOR_STYLE[row.candidate].label} | {row.mean_difference_points:+.2f} | {row.lower_5th_points:+.2f} to {row.upper_95th_points:+.2f} | {row.verdict} |")
    lines += ["", "Negative differences favour the predictor. CRPS and differences are in percentage points of the 14-day share. Details, sensitivity to the block length, calibration and the sampler record are in `diagnostics.xlsx`.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(*, folder: Path, run_id: str, raw: str, outcome: str, table: pd.DataFrame, results: dict, posteriors: dict | None, draws: int, tune: int, chains: int, seconds: float, score_model: bool) -> None:
    folder = Path(folder)
    scores = {name: result[0] for name, result in results.items()}
    origin_tables = {name: result[1] for name, result in results.items()}
    ridge = pd.concat([result[2].assign(set=name) for name, result in results.items()], ignore_index=True)
    evaluation = origin_tables["evaluation"]
    wide = crps_by_origin(scores["evaluation"])
    comparisons = build_comparisons(wide)
    power = power_check(comparisons)
    summary = predictor_summary(scores["evaluation"], wide)
    figures = write_figure(wide, comparisons, outcome, folder)
    origins_sheet = _origins_sheet(origin_tables, scores)
    pd.concat([scores[name].assign(set=name) for name in scores], ignore_index=True).to_csv(folder / "forecast_harness_scores.csv", index=False)
    files = ["forecast_harness_scores.csv", "forecast_harness_summary.md", *[path.name for path in figures]]
    sheets = {
        "Specification_Coding": _specification(outcome, draws, tune, chains, score_model),
        "Denominators_Flow": _flow(origin_tables, scores, score_model),
        "Origins": origins_sheet,
        "Win_Test": comparisons,
        "Predictor_Scores": summary,
        "Settings_Changes": settings_change_table(evaluation, scores["evaluation"]),
    }
    if score_model:
        sheets["Model_Fits"] = _model_fits(posteriors)
    sheets["Ridge_Fits"] = ridge.assign(origin=ridge["origin"].dt.date)
    sheets["Development"] = _development(scores["development"], origin_tables["development"])
    sheets["Checks"] = _checks(evaluation=evaluation, evaluation_scores=scores["evaluation"], comparisons=comparisons, summary=summary, power=power, posteriors=posteriors, ridge=ridge, score_model=score_model)
    sheets["Warnings_Limitations"] = _warnings()
    sheets["Artifacts_Provenance"] = _provenance(run_id=run_id, raw=raw, outcome=outcome, table=table, files=[*files, "diagnostics.xlsx"], seconds=seconds, draws=draws, tune=tune, chains=chains, score_model=score_model)
    path = write_workbook(folder / "diagnostics.xlsx", sheets)
    write_summary(folder / "forecast_harness_summary.md", run_id=run_id, outcome=outcome, evaluation=evaluation, summary=summary, comparisons=comparisons, power=power, score_model=score_model)
    reconcile(path, evaluation, comparisons, summary, score_model, len(posteriors or {}))


def reconcile(path: Path, evaluation: pd.DataFrame, comparisons: pd.DataFrame, summary: pd.DataFrame, score_model: bool, fits: int) -> None:
    """Read the workbook back and compare it with the tables it summarises."""
    workbook = read_workbook(path)
    flow = workbook["Denominators_Flow"]
    flow = flow[flow["set"] == "evaluation"].set_index("stage")["n"]
    if int(flow.iloc[0]) != len(evaluation) or int(flow.iloc[1]) != int(evaluation["scored"].sum()):
        raise RuntimeError("Workbook flow differs from the evaluation origins")
    stored = workbook["Win_Test"]
    if not np.allclose(stored["mean_difference_points"].to_numpy(dtype=float), comparisons["mean_difference_points"].to_numpy(), atol=1e-3) or list(stored["verdict"]) != list(comparisons["verdict"]):
        raise RuntimeError("Workbook win test differs from the comparisons")
    scores = workbook["Predictor_Scores"].set_index("predictor")["mean_crps_points"]
    if not np.allclose(scores.reindex(summary["predictor"]).to_numpy(dtype=float), summary["mean_crps_points"].to_numpy(), atol=1e-3):
        raise RuntimeError("Workbook predictor scores differ from the summary")
    by_predictor = summary.set_index("predictor")
    for _, row in comparisons.iterrows():
        both = by_predictor.reindex([row["candidate"], row["reference"]])
        if (both["origins_scored"] == row["origins"]).all() and (both["mean_crps_points"].iloc[0] < both["mean_crps_points"].iloc[1]) != (row["mean_difference_points"] < 0):
            raise RuntimeError(f"The sign of {row['candidate']} minus {row['reference']} disagrees with the mean CRPS of the two predictors")
    if score_model and len(workbook["Model_Fits"]) != fits:
        raise RuntimeError("Workbook model fits differ from the fits run")
