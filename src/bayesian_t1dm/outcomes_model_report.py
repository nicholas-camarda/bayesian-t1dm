"""Tables, figure and diagnostics workbook for the hour-day outcomes model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from . import outcomes_model as om
from .diagnostics_workbook import read_workbook, write_workbook
from .figures import NEUTRAL_BAND, NEUTRAL_BAR, NEUTRAL_LINE, assert_ascii
from .publish import current_commit, working_tree_is_clean
from .sampler_health import FitDiagnostics, sampler_gate

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTCOME_LABELS = {"tir": "Share of readings 70-180 mg/dL", "below70": "Share of readings below 70 mg/dL", "below54": "Share of readings below 54 mg/dL"}
COVARIATE_LABELS = {
    "log_icr": "I:C (log)", "log_cf": "Correction factor (log)", "log_basal": "Basal rate (log)", "target_10": "Target (per 10 mg/dL)",
    "log_carbs_3h": "Carbohydrate, last 3 h (log)", "exercise_share": "Exercise share of the hour", "sleep_mode_share": "Sleep mode share of the hour", "site_age_capped": "Days since cannula fill (to 5)",
}
PREDICTOR_LABELS = {"hour-day model": "Hour-day\nmodel", "last 28 days": "Last 28\ndays"}
SETTING_COVARIATES = ("log_icr", "log_cf", "log_basal", "target_10")
FIGURE_FILES = ("outcomes_model_figure.png", "outcomes_model_figure.svg")


def write_figure(outcome: str, hourly: pd.DataFrame, effects: pd.DataFrame, origin_scores: pd.DataFrame | None, output: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = 3 if origin_scores is not None else 2
    figure, axes = plt.subplots(1, panels, figsize=(5.2 * panels, 5.6), gridspec_kw={"width_ratios": [1.5, 1.3, 1.0][:panels], "wspace": 0.55})
    clock, forest = axes[0], axes[1]
    scale = 100.0
    clock.fill_between(hourly["hour"], hourly["predicted_5th"] * scale, hourly["predicted_95th"] * scale, color=NEUTRAL_BAND, step="mid", label="Model: 5th to 95th percentile")
    clock.plot(hourly["hour"], hourly["predicted_median"] * scale, color=NEUTRAL_BAR, linewidth=2.2, drawstyle="steps-mid", label="Model: median")
    clock.scatter(hourly["hour"], hourly["observed"] * scale, s=34, color=NEUTRAL_LINE, zorder=3, label="Observed")
    clock.set_xticks(range(0, 24, 3))
    clock.set_xlabel("Clock hour (pump wall clock)", fontsize=10)
    clock.set_ylabel(OUTCOME_LABELS[outcome] + " (%)", fontsize=10)
    clock.set_title("24-hour check", fontsize=11, loc="left")
    clock.legend(fontsize=8.5, frameon=False, loc="best")
    ordered = effects.reset_index(drop=True)
    positions = np.arange(len(ordered))[::-1]
    for position, (_, row) in zip(positions, ordered.iterrows()):
        setting = row["covariate"] in SETTING_COVARIATES
        forest.errorbar(row["points_per_sd"], position, xerr=[[row["points_per_sd"] - row["points_5th"]], [row["points_95th"] - row["points_per_sd"]]], fmt="o" if setting else "s", color=NEUTRAL_LINE,
                        mfc=NEUTRAL_LINE if setting else "white", capsize=3, linewidth=2.0, markersize=6)
    forest.axvline(0, color=NEUTRAL_LINE, linewidth=1.0)
    forest.set_yticks(positions)
    forest.set_yticklabels([COVARIATE_LABELS[name] for name in ordered["covariate"]], fontsize=9)
    forest.set_xlabel("Change in share of readings (percentage\npoints) per +1 SD, 90% interval", fontsize=9.5)
    forest.set_title("Association of each covariate\n(filled: settings; hollow: context)", fontsize=11, loc="left")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)
    title = f"Hour-day model of {OUTCOME_LABELS[outcome].lower()}: fit to every usable hour, 4 chains"
    notes = ["Left: each dot is the observed share at that clock hour over the whole record; the band is what the fitted model predicts for those same hours.",
             "Middle: associations given clock hour and day; settings and context change together, so these are not effects of changing a setting."]
    labels = [title, *notes, *COVARIATE_LABELS.values(), *OUTCOME_LABELS.values()]
    if origin_scores is not None:
        summary = origin_summary(origin_scores)
        axis = axes[2]
        for index, predictor in enumerate(summary.index):
            axis.bar(index, summary.loc[predictor, "crps_points"], color=NEUTRAL_LINE if predictor == "hour-day model" else NEUTRAL_BAR, width=0.6)
            axis.text(index, summary.loc[predictor, "crps_points"], f"{summary.loc[predictor, 'crps_points']:.2f}\n(90% interval held\n{summary.loc[predictor, 'coverage']:.0%} of days)", ha="center", va="bottom", fontsize=8.5)
        axis.set_xticks(range(len(summary)))
        axis.set_xticklabels([PREDICTOR_LABELS[name] for name in summary.index], fontsize=9)
        axis.set_ylabel("Mean CRPS, daily share\n(percentage points, lower is better)", fontsize=9.5)
        axis.set_ylim(0, summary["crps_points"].max() * 1.45)
        axis.set_title("Next 14 days, 4 origins", fontsize=11, loc="left")
        notes.append("Right: days after each of four fortnightly origins, predicted from the days before; the model sees each day's settings, the baseline does not.")
        labels.append(notes[-1])
    assert_ascii(*labels)
    figure.suptitle(title, fontsize=12, x=0.01, ha="left", y=0.995)
    figure.text(0.01, 0.945, "\n".join(notes), fontsize=8.5, ha="left", va="top", color=NEUTRAL_LINE)
    figure.subplots_adjust(top=0.80, bottom=0.14, left=0.06, right=0.985)
    paths = []
    for name in FIGURE_FILES:
        path = output / name
        figure.savefig(path, dpi=300)
        paths.append(path)
    plt.close(figure)
    return paths


def origin_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean CRPS (percentage points) and 90%-interval coverage of each predictor over every scored held-out day."""
    grouped = scores.groupby("predictor")
    return pd.DataFrame({"crps_points": grouped["crps"].mean() * 100.0, "coverage": grouped["inside_90"].mean(), "days": grouped.size()})


def _checks(outcome: str, design: om.OutcomeDesign, diagnostics: FitDiagnostics, vif: pd.DataFrame, prior: dict[str, float], hourly: pd.DataFrame, scores: pd.DataFrame | None, skipped: pd.DataFrame | None) -> pd.DataFrame:
    above = int((hourly["observed"] > hourly["predicted_median"]).sum())
    gap = float((hourly["observed"] - hourly["predicted_median"]).mean() * 100.0)
    sign = binomtest(above, 24, 0.5)
    rows = [
        ("Sampler gate", f"R-hat max {diagnostics.rhat_max:.4f} (limit {om.R_HAT_LIMIT}), bulk ESS min {diagnostics.ess_bulk_min:.0f}, tail ESS min {diagnostics.ess_tail_min:.0f} (limit {om.ESS_LIMIT}), {diagnostics.divergences} divergences",
         "no concern" if sampler_gate(diagnostics) else "concern"),
        ("Tree depth", f"{diagnostics.max_tree_depth_hits} transitions reached depth {diagnostics.max_treedepth}; deepest {diagnostics.max_tree_depth_observed}", "no concern" if diagnostics.max_tree_depth_hits == 0 else "concern"),
        ("Collinearity of covariates", "largest variance inflation factor " + (f"{np.nanmax(vif[['vif_raw', 'vif_within_hour']].to_numpy()):.1f}" if np.isfinite(vif[["vif_raw", "vif_within_hour"]].to_numpy()).any() else "undefined") + " (raw or within clock hour; 5 or more is a concern)",
         "concern" if (vif["status"] == "concern").any() else ("not assessed" if (vif["status"] == "not assessed").any() else "no concern")),
        ("Prior predictive", f"overall share of readings in the outcome across 200 prior draws: {prior['overall_share_5th']:.2f} to {prior['overall_share_95th']:.2f}; the prior is centered on {prior['prior_mean']:.3f}",
         "no concern" if prior["overall_share_5th"] < prior["prior_mean"] < prior["overall_share_95th"] else "concern"),
        ("Posterior predictive by clock hour: coverage", f"{int(hourly['inside_90_interval'].sum())} of 24 clock hours have the observed share inside the model's 90% interval (about 22 expected)", "no concern" if hourly["inside_90_interval"].sum() >= 20 else "concern"),
        ("Posterior predictive by clock hour: direction", f"observed share above the model median at {above} of 24 clock hours (mean gap {gap:+.1f} percentage points; two-sided sign test p = {sign.pvalue:.3f}); a one-sided pattern means the model misses the overall level", "no concern" if sign.pvalue >= 0.05 else "concern"),
    ]
    if scores is None:
        rows.append(("Rolling-origin check", "not run", "not assessed"))
    else:
        summary = origin_summary(scores)
        better = summary.loc["hour-day model", "crps_points"] < summary.loc["last 28 days", "crps_points"]
        rows.append(("Rolling-origin check", f"mean CRPS {summary.loc['hour-day model', 'crps_points']:.2f} vs {summary.loc['last 28 days', 'crps_points']:.2f} points (model vs last-28-day baseline); 90% interval held {summary.loc['hour-day model', 'coverage']:.0%} vs {summary.loc['last 28 days', 'coverage']:.0%} of {int(summary.loc['hour-day model', 'days'])} days",
                     "no concern" if better else "concern"))
    if skipped is not None and not skipped.empty:
        rows.append(("Rolling origins not scored", "; ".join(f"{row.origin}: {row.reason}" for row in skipped.itertuples()), "not assessed"))
    rows.append(("Settings known for every row", "rows with any unknown setting moment are not modelled (see Denominators_Flow); unknown moments cluster in 2024 and in the 21:00-24:00 segment of the first pump", "not assessed"))
    return pd.DataFrame(rows, columns=["check", "result", "status"])


def _specification(outcome: str, draws: int, tune: int, chains: int) -> pd.DataFrame:
    rows = [
        ("Question", "How are the share of readings in range, below 70 and below 54 in each clock hour predicted by the settings in force and context?"),
        ("Claim type", "Predictive and associational, one person; coefficients are not effects of changing a setting."),
        ("Unit of analysis", "One clock hour of one day (up to 12 five-minute readings)."),
        ("Outcome", f"{OUTCOME_LABELS[outcome]}; count out of valid readings in the hour."),
        ("Likelihood", "Beta-binomial: counts over-dispersed relative to a binomial because readings within an hour are correlated."),
        ("Linear predictor", "logit(mu) = alpha + clock-hour effect (sums to zero) + day effect + covariates * beta; hour and day effects non-centered."),
        ("Covariates", ", ".join(om.COVARIATES) + "; standardized on the training rows (coefficient is the change in log odds per standard deviation; percentage points use the average of mu(1-mu))."),
        ("Left out", "Bolus, insulin on board and delivered basal (on the path from settings to glucose), Apple Health, any slow drift of the level, and any interaction between settings and clock hour."),
        ("Priors", "alpha Normal centered on the typical share (scale 1); beta Normal(0, 0.5); hour scale HalfNormal(1); day scale HalfNormal(0.5); log phi Normal(0, 1.5)."),
        ("Sampler", f"NUTS, {chains} chains, {tune} tuning and {draws} draws each, target acceptance {om.TARGET_ACCEPT}, maximum tree depth {om.MAX_TREEDEPTH}, seed {om.SEED}."),
        ("Interpretation", "A coefficient of -0.10 per SD means a one-SD higher covariate goes with about 10% lower odds of a reading in the outcome in the same clock hour and day structure, given the other covariates."),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def _warnings() -> pd.DataFrame:
    rows = [
        ("Association", "Settings and context change together and were chosen by the person and the controller in response to glucose; a coefficient is not the effect of changing that setting."),
        ("Settings that rarely changed", "A setting that changed a few times is estimated from those episodes; its interval is wide and its sign can follow whatever else changed then."),
        ("No slow drift term", "The person's level may drift over weeks or months; the model has only day-to-day effects. A random-walk level over weeks or 28-day blocks did not sample reliably (its scale is not identified beside the day effects), so a slow trend that coincides with a setting change can load on that setting."),
        ("Rows with unknown settings", "Hours with any unknown setting moment, early 2024 and the 21:00-24:00 segment of the first pump, are not modelled."),
        ("Rolling-origin window", "Four fortnights are scored; days in context-event fortnights and days with unknown settings are not, so some fortnights hold few days. The comparison with the last-28-day baseline is over the scored days only and carries no interval: a small difference in CRPS is within noise."),
        ("Context events", "Days listed in context_events.csv are left out, so the model says nothing about those days."),
    ]
    return pd.DataFrame(rows, columns=["limitation", "detail"])


def write_outputs(*, outcome: str, table: pd.DataFrame, design: om.OutcomeDesign, idata, diagnostics: FitDiagnostics, timing: dict[str, float], vif: pd.DataFrame, prior: dict[str, float],
                  origin_scores: pd.DataFrame | None, skipped_origins: pd.DataFrame | None, draws: int, tune: int, chains: int, output: Path, run_id: str, raw: str, seconds: float) -> None:
    output = Path(output)
    hourly = om.hourly_check(idata, design)
    effects = om.effects_table(idata, design)
    scales = om.scale_summary(idata)
    hourly.to_csv(output / "outcomes_model_hourly_check.csv", index=False)
    effects.to_csv(output / "outcomes_model_effects.csv", index=False)
    figures = write_figure(outcome, hourly, effects, origin_scores, output)
    if origin_scores is not None:
        origin_scores.to_csv(output / "outcomes_model_rolling_origin.csv", index=False)
    flow = om.design_flow(table).rename(columns={"rows": "n"})
    fit_sheet = pd.DataFrame([
        ("rows modelled", design.rows), ("days", len(design.days)), ("draws per chain", diagnostics.draws), ("tuning steps per chain", diagnostics.tune), ("chains", diagnostics.chains),
        ("R-hat max", round(diagnostics.rhat_max, 4)), ("bulk ESS min", round(diagnostics.ess_bulk_min)), ("tail ESS min", round(diagnostics.ess_tail_min)), ("divergences", diagnostics.divergences),
        ("transitions at maximum tree depth", diagnostics.max_tree_depth_hits), ("compile seconds", round(timing["compile_seconds"], 1)), ("sampling seconds", round(timing["sample_seconds"], 1)),
    ], columns=["item", "value"])
    dirty = not working_tree_is_clean(REPO_ROOT)
    files = ["outcomes_model_hourly_check.csv", "outcomes_model_effects.csv", *[path.name for path in figures]]
    if origin_scores is not None:
        files.append("outcomes_model_rolling_origin.csv")
    provenance = pd.DataFrame([
        ("run id", run_id), ("commit", current_commit(REPO_ROOT)), ("working tree clean", "no" if dirty else "yes"), ("raw data root", raw),
        ("record span", f"{table['date'].min().date()} to {table['date'].max().date()}"), ("seed", om.SEED), ("outcome", outcome),
        ("command", f"python -m bayesian_t1dm.outcomes_model --outcome {outcome}"), ("wall time (seconds)", round(seconds, 1)), ("files", json.dumps(files)),
    ], columns=["item", "value"])
    sheets = {
        "Specification_Coding": _specification(outcome, draws, tune, chains), "Denominators_Flow": flow, "Fit_Sampler": fit_sheet,
        "Checks": _checks(outcome, design, diagnostics, vif, prior, hourly, origin_scores, skipped_origins), "Collinearity": vif, "Effects": effects, "Scales": scales, "Hourly_Check": hourly,
    }
    if origin_scores is not None:
        sheets["Rolling_Origin"] = origin_summary(origin_scores).reset_index()
    sheets["Warnings_Limitations"] = _warnings()
    sheets["Artifacts_Provenance"] = provenance
    path = write_workbook(output / "diagnostics.xlsx", sheets)
    _reconcile(path, design, effects, hourly, origin_scores)


def _reconcile(path: Path, design: om.OutcomeDesign, effects: pd.DataFrame, hourly: pd.DataFrame, scores: pd.DataFrame | None) -> None:
    """Read the workbook back and compare it with the tables it summarises."""
    workbook = read_workbook(path)
    flow = workbook["Denominators_Flow"]["n"].iloc[-1]
    if int(flow) != design.rows:
        raise RuntimeError(f"Workbook flow ends at {flow} rows, the design holds {design.rows}")
    fit_sheet = workbook["Fit_Sampler"].set_index("item")["value"]
    if int(fit_sheet["rows modelled"]) != design.rows or int(fit_sheet["days"]) != len(design.days):
        raise RuntimeError("Workbook fit sheet differs from the design")
    stored = workbook["Effects"].set_index("covariate")["points_per_sd"]
    if not np.allclose(stored.to_numpy(dtype=float), effects.set_index("covariate")["points_per_sd"].reindex(stored.index).to_numpy(), atol=1e-3):
        raise RuntimeError("Workbook effects differ from the effects table")
    if not np.allclose(workbook["Hourly_Check"]["observed"].to_numpy(dtype=float), hourly["observed"].to_numpy(), atol=1e-6):
        raise RuntimeError("Workbook hourly check differs from the hourly table")
    if scores is not None:
        stored_summary = workbook["Rolling_Origin"].set_index("predictor")["crps_points"]
        expected = origin_summary(scores)["crps_points"]
        if not np.allclose(stored_summary.reindex(expected.index).to_numpy(dtype=float), expected.to_numpy(), atol=1e-3):
            raise RuntimeError("Workbook rolling-origin summary differs from the scores")
