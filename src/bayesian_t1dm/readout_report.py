"""Build the personal readout: day table, drivers model, settings check, page and diagnostics workbook.

Run with ``python -m bayesian_t1dm.readout_report``. Writes ``readout.html``, ``diagnostics.xlsx`` and the result tables
to ``runtime/output/readout/`` and copies the page to ``<Project Vault>/readout/readout.html`` so it opens from the
OneDrive app. The workbook carries a ``Record_Issues`` sheet (dates of left-out days) and stays out of Git.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import readout as ro
from . import readout_model as rm
from . import readout_settings as rs
from .diagnostics_workbook import read_workbook, write_workbook
from .publish import current_commit, working_tree_is_clean
from .readout_page import TRACE_DAYS, Readout, render
from .sampler_health import ESS_LIMIT, R_HAT_LIMIT

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
USUAL_WEEK_DAYS = 90
USUAL_WEEK_MIN_DAYS = 5  # a 7-day stretch counts toward the usual range with at least this many full days


def latest_full_day(days: pd.DataFrame) -> pd.Timestamp:
    counted = days.index[days["counted"]]
    if counted.empty:
        raise ValueError("No day in the record has 70% of its readings; the readout needs at least one full day.")
    return pd.Timestamp(counted.max())


def usual_week(days: pd.DataFrame, as_of: pd.Timestamp) -> tuple[float, float, float]:
    """10th, 50th and 90th percentiles of pooled 7-day time in range over the 90 days before this week."""
    week_start = as_of - pd.Timedelta(days=TRACE_DAYS - 1)
    full = days[days["counted"]]
    values = []
    for end in pd.date_range(week_start - pd.Timedelta(days=USUAL_WEEK_DAYS), week_start - pd.Timedelta(days=1), freq="D"):
        window = full.loc[end - pd.Timedelta(days=TRACE_DAYS - 1): end]
        if len(window) >= USUAL_WEEK_MIN_DAYS:
            values.append(100 * window["in_range_readings"].sum() / window["readings"].sum())
    if len(values) < 30:
        raise ValueError(f"Only {len(values)} 7-day stretches in the 90 days before this week have enough full days; the usual range needs 30.")
    return tuple(float(v) for v in np.percentile(values, [10, 50, 90]))


def _specification(fits: dict[str, rm.DriverFit]) -> pd.DataFrame:
    rows = []
    for outcome, (count_column, label) in rm.OUTCOMES.items():
        rows.append({"item": f"{label}: model", "value": "Beta-binomial regression of the day's readings in the outcome over the day's readings; logit link; partially pooled month effect; weekend term; one coefficient per driver, Normal(0, 0.5) prior on the logit scale."})
        rows.append({"item": f"{label}: estimand", "value": "Average marginal effect: over the modelled days, the mean difference in the predicted share of the day with the driver set to yes against no, in percentage points of the day."})
        rows.append({"item": f"{label}: contrast", "value": "Driver present minus driver absent, other drivers, weekend and month held at each day's values. Positive means more of the day in the outcome."})
        rows.append({"item": f"{label}: interpretation", "value": "A within-person association: days after the driver had this much more (or less) of the outcome than similar days of the same month without it. Not shown to be caused by it."})
        rows.append({"item": f"{label}: fitted days", "value": fits[outcome].days})
    for column, label, text in ro.DRIVERS:
        rows.append({"item": f"driver: {label}", "value": text})
    rows += [
        {"item": "personal baseline", "value": f"median and robust spread (median absolute deviation x 1.4826) of the previous {ro.BASELINE_DAYS} days on the same device, at least {ro.BASELINE_MIN_DAYS} days"},
        {"item": "recovery score", "value": "100 x normal percentile of (HRV z minus resting-HR z) / sqrt(2) on the day; one z alone when only one is measured; 50 is a usual day"},
        {"item": "settings check: basal", "value": "controller.basal_signal: Control-IQ's delivery against the current programmed rate over clean hours (no meal, bolus or exercise window), last 28 days widening to 56 or 90"},
        {"item": "settings check: I:C", "value": "effective I:C from clean first meals of the day (latent meal research, full scope, strict meal proxy), latest period with enough meals; needs more insulin when the 90% range lies below the current ratio, less when above"},
        {"item": "settings check: suggestion", "value": f"one step of {', '.join(f'{100 * s:.0f}%' for s in rs.STEPS)}, the step nearer the gap; no added insulin in a segment whose 90-day lows are over a consensus limit"},
    ]
    return pd.DataFrame(rows)


def _sampler_sheet(fits: dict[str, rm.DriverFit]) -> pd.DataFrame:
    rows = []
    for outcome, fit in fits.items():
        d = fit.diagnostics
        rows.append({"outcome": rm.OUTCOMES[outcome][1], "check": f"R-hat below {R_HAT_LIMIT}, bulk and tail ESS above {ESS_LIMIT}, no divergences",
                     "result": f"R-hat max {d.rhat_max:.4f}; ESS bulk min {d.ess_bulk_min:.0f}, tail min {d.ess_tail_min:.0f}; divergences {d.divergences}; tree-depth hits {d.max_tree_depth_hits}",
                     "status": "no concern" if fit.usable else "concern", "reason": "all limits met" if fit.usable else "a limit was not met: the effects are not reliable"})
    return pd.DataFrame(rows)


def _effects_sheet(fits: dict[str, rm.DriverFit]) -> pd.DataFrame:
    frames = []
    for outcome, fit in fits.items():
        frame = fit.effects.copy()
        frame.insert(0, "outcome", rm.OUTCOMES[outcome][1])
        frame["driver"] = frame["driver"].map(ro.DRIVER_LABELS)
        frame["reading (time in range only)"] = fit.effects.apply(rm.reading, axis=1) if outcome == "time_in_range" else ""
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    return out.rename(columns={"days_with": "modelled days with it", "days_without": "modelled days without it", "estimate_pp": "estimate (percentage points)",
                               "lower_pp": "90% lower (pp)", "upper_pp": "90% upper (pp)", "prob_above_zero": "probability above zero"})


def _record_issues(days: pd.DataFrame, kept: pd.DataFrame) -> pd.DataFrame:
    rows = []
    remaining = days
    for reason, mask in (
        ("fewer than 70% of the day's readings", ~days["counted"]),
        ("context-event day", days["excluded_day"]),
        ("before infusion-site changes are observed", days["site_age_days"].isna()),
        ("a driver was not measured", days[list(ro.DRIVER_COLUMNS)].isna().any(axis=1)),
    ):
        hit = remaining[mask.reindex(remaining.index).fillna(False).astype(bool)]
        for date, row in hit.iterrows():
            missing = [ro.DRIVER_LABELS[c] for c in ro.DRIVER_COLUMNS if pd.isna(row[c])] if reason == "a driver was not measured" else []
            rows.append({"date": date.date(), "step": "drivers model", "reason": reason, "field": ", ".join(missing) or "", "readings": int(row["readings"])})
        remaining = remaining.drop(hit.index)
    issues = pd.DataFrame(rows, columns=["date", "step", "reason", "field", "readings"])
    if len(days) - len(issues) != len(kept):
        raise ValueError(f"Record issues ({len(issues)}) do not reconcile with the modelled days ({len(kept)} of {len(days)}).")
    return issues


def _warnings(fits: dict[str, rm.DriverFit], kept: pd.DataFrame) -> pd.DataFrame:
    sources = kept["sleep_device"].value_counts().to_dict()
    return pd.DataFrame([
        {"topic": "design", "statement": "Associations within one person's record; a driver can travel with something unmeasured (for example illness behind both poor sleep and high glucose).", "status": "concern", "reason": "observational record"},
        {"topic": "device change", "statement": f"Sleep comes from more than one device over the record ({json.dumps(sources)}); baselines are per device and short sleep is an absolute threshold.", "status": "no concern" if len(sources) < 2 else "concern", "reason": "devices measure sleep differently"},
        {"topic": "context-event days", "statement": "Days the person listed as context events are left out of the model, as in the other analyses.", "status": "not applicable", "reason": "rule shared across the project"},
        {"topic": "settings check", "statement": "Control-IQ adjusts basal all the time; the basal and I:C evidence is observational and is for discussion with the endocrinologist.", "status": "concern", "reason": "closed-loop confounding"},
        {"topic": "sampler", "statement": "; ".join(f"{rm.OUTCOMES[o][1]}: {'passed' if f.usable else 'failed'}" for o, f in fits.items()), "status": "no concern" if all(f.usable for f in fits.values()) else "concern", "reason": "see Sampler_Health"},
    ])


def _provenance(run_id: str, raw: str, days: pd.DataFrame, files: list[str], seconds: float, command: str) -> pd.DataFrame:
    rows = [("run id", run_id), ("commit", current_commit(REPO_ROOT)), ("working tree clean", "yes" if working_tree_is_clean(REPO_ROOT) else "no"), ("raw data root", raw),
            ("record span", f"{days.index.min().date()} to {days.index.max().date()}"), ("seed", rm.SEED),
            ("sampler", f"{rm.CHAINS} chains, {rm.TUNE} tuning, {rm.DRAWS} draws, target acceptance {rm.TARGET_ACCEPT}"), ("command", command),
            ("wall time (seconds)", round(seconds, 1)), ("run record", f"readout_run_{run_id}.log"), ("files", json.dumps(files))]
    return pd.DataFrame(rows, columns=["item", "value"])


def reconcile(path: Path, fits: dict[str, rm.DriverFit], flow: pd.DataFrame, kept: pd.DataFrame) -> None:
    """Read the workbook back and check it against the result objects."""
    book = read_workbook(path)
    effects = book["Driver_Effects"]
    for outcome, fit in fits.items():
        sheet = effects[effects["outcome"].eq(rm.OUTCOMES[outcome][1])]
        estimated = fit.effects.dropna(subset=["estimate_pp"])
        if not np.allclose(sheet["estimate (percentage points)"].dropna().to_numpy(), estimated["estimate_pp"].to_numpy(), atol=1e-9):
            raise ValueError(f"Workbook effects for {outcome} do not match the fit.")
    if int(book["Denominators_Flow"]["days"].iloc[-1]) != len(kept):
        raise ValueError("Workbook flow does not end at the modelled days.")


def run(raw: str | None, output: str | None, *, draws: int, tune: int, chains: int, command: str) -> int:
    from .analysis_data import load_analysis_data
    from .paths import ProjectPaths

    paths = ProjectPaths.from_root(Path.cwd())
    raw = raw or str(paths.cloud_raw)
    folder = Path(output or paths.output_readout)
    folder.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(folder / f"readout_run_{run_id}.log")])
    started = time.time()
    LOGGER.info("readout started run_id=%s raw=%s", run_id, raw)
    analysis = load_analysis_data(raw)
    table = ro.daily_table(analysis)
    days = table.days
    as_of = latest_full_day(days)
    kept, flow = ro.model_days(days)
    flow = pd.DataFrame(flow)
    LOGGER.info("day flow\n%s", flow.to_string(index=False))
    fits = rm.fit_all(kept, draws=draws, tune=tune, chains=chains)
    for outcome, fit in fits.items():
        LOGGER.info("%s: %d days, sampler %s\n%s", outcome, fit.days, "passed" if fit.usable else "FAILED", fit.effects.to_string(index=False))
    settings, evidence = rs.settings_check(analysis, raw, as_of=as_of)
    changes = rs.recent_changes(analysis, as_of=as_of)
    merged = analysis.merged
    readout = Readout(
        as_of=as_of, days=days, usual_week=usual_week(days, as_of), effects={o: f.effects for o, f in fits.items()}, usable={o: f.usable for o, f in fits.items()},
        model_days=len(kept), settings=settings, changes=changes, cgm=analysis.cgm[["timestamp", "glucose"]],
        sleep=ro.sleep_by_night(merged.data.sleep).reset_index()[["sleep_start", "sleep_end"]],
        workouts=merged.data.workouts[["start_timestamp", "end_timestamp"]], carbs=merged.tandem_data.carbs[["timestamp", "carb_grams"]],
        site_fills=pd.to_datetime(analysis.site_events.loc[analysis.site_events["kind"].eq(ro.SITE_EVENT), "timestamp"]),
        run_id=run_id, built=f"{pd.Timestamp.now():%Y-%m-%d %H:%M}",
    )
    page = folder / "readout.html"
    page.write_text(render(readout), encoding="utf-8")
    effects_csv = folder / "readout_driver_effects.csv"
    _effects_sheet(fits).to_csv(effects_csv, index=False)
    settings.to_csv(folder / "readout_settings.csv", index=False)
    files = [page.name, effects_csv.name, "readout_settings.csv", "diagnostics.xlsx"]
    workbook = folder / "diagnostics.xlsx"
    write_workbook(workbook, {
        "Specification_Coding": _specification(fits),
        "Denominators_Flow": flow,
        "Driver_Counts": rm.estimable_drivers(kept)[1].assign(driver=lambda f: f["driver"].map(ro.DRIVER_LABELS)),
        "Driver_Effects": _effects_sheet(fits),
        "Sampler_Health": _sampler_sheet(fits),
        "Settings_Check": settings.rename(columns={"status": "reading"}),
        "Basal_Signal": evidence["basal_signal"],
        "ICR_By_Segment": evidence["segment_summary"],
        "Recent_Changes": changes.rename(columns={"status": "evaluation"}),
        "Record_Issues": _record_issues(days, kept),
        "Warnings_Limitations": _warnings(fits, kept),
        "Artifacts_Provenance": _provenance(run_id, raw, days, files, time.time() - started, command),
    })
    reconcile(workbook, fits, flow, kept)
    if not all(fit.usable for fit in fits.values()):
        raise RuntimeError("A drivers model failed its sampler checks; the page and workbook were written for inspection but not copied to the Vault.")
    paths.cloud_readout.mkdir(parents=True, exist_ok=True)
    shutil.copy2(page, paths.cloud_readout / "readout.html")
    LOGGER.info("readout written to %s and %s in %.1f s", page, paths.cloud_readout / "readout.html", time.time() - started)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", default=None, help="Raw data root (default: the Project Vault data/raw)")
    parser.add_argument("--output", default=None, help="Output folder (default: runtime/output/readout)")
    parser.add_argument("--draws", type=int, default=rm.DRAWS)
    parser.add_argument("--tune", type=int, default=rm.TUNE)
    parser.add_argument("--chains", type=int, default=rm.CHAINS)
    args = parser.parse_args(argv)
    command = f"python -m bayesian_t1dm.readout_report --draws {args.draws} --tune {args.tune} --chains {args.chains}"
    return run(args.raw, args.output, draws=args.draws, tune=args.tune, chains=args.chains, command=command)


if __name__ == "__main__":
    raise SystemExit(main())
