"""Answer-first HTML review pages.

Each page opens with the question it answers and a one-sentence verdict, then the
headline numbers, how much to trust the result and what to do next. Figures, full
tables, companion files and a glossary sit in fold-out sections below. Internal
status codes never appear on their own; they are translated with ``report_page.plain``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from . import report_page as rp
from .health_auto_export import ModelDataPreparationResult
from .therapy_research import LatentMealResearchResult, TherapyInfraValidationResult, TherapyResearchResult, summarize_overnight_basal_evidence

FIT_ERROR_GUIDE_MGDL = 10.0
FORECAST_INTERVAL_PERCENT = 80
FORECAST_COVERAGE_RANGE = (0.70, 0.90)
TRACE_CLIP_MGDL = (20.0, 450.0)


def _write(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _date_span(start: Any, end: Any) -> str:
    start_ts = pd.to_datetime(start, errors="coerce")
    end_ts = pd.to_datetime(end, errors="coerce")
    if pd.isna(start_ts) or pd.isna(end_ts):
        return "date range unknown"
    return f"{start_ts:%b %d, %Y} to {end_ts:%b %d, %Y}"


def _generated() -> str:
    return f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}"


def _files(root: Path, href_prefix: str, entries: list[tuple[str, str]]) -> str:
    return rp.file_links((f"{href_prefix}{name}", description, (root / name).exists()) for name, description in entries)


def _reason_list(raw: Any) -> list[str]:
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw]
    text = str(raw or "").strip()
    if not text or text in {"none", "nan"}:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except (ValueError, TypeError):
        pass
    return [part for part in text.replace(",", "|").split("|") if part]


# ---------------------------------------------------------------------------
# Data coverage
# ---------------------------------------------------------------------------


def write_coverage_review_html(summary: dict[str, Any], quality_rows: pd.DataFrame, path: str | Path) -> Path:
    data_quality = summary.get("data_quality") or {}
    status_code = str(data_quality.get("status") or "broken")
    windows = quality_rows["window_id"].nunique() if not quality_rows.empty else 0
    incomplete = int(data_quality.get("incomplete_window_count") or 0)
    reason_counts = data_quality.get("reason_counts") or {}

    status, label = {
        "good": ("good", "Complete"),
        "degraded": ("warn", "Usable, with gaps"),
    }.get(status_code, ("bad", "Incomplete"))
    headline = (
        f"<b>{incomplete} of {windows} download windows</b> have gaps or other problems."
        if windows
        else "<b>No download windows were found.</b>"
    )
    explanation = (
        "Analyses can still run on incomplete windows, but recommendations are held back until the data are complete."
        if incomplete
        else "Every download window covers the period it was asked for."
    )

    window_rows = []
    for window_id, group in quality_rows.groupby("window_id", dropna=False) if not quality_rows.empty else []:
        coverage = pd.to_numeric(group.get("coverage_fraction"), errors="coerce")
        reasons: list[str] = []
        for raw in group.get("completeness_reasons", pd.Series(dtype=str)).fillna("[]"):
            for reason in _reason_list(raw):
                if reason not in reasons:
                    reasons.append(reason)
        cgm = group.loc[group["kind"].astype(str).eq("cgm")] if "kind" in group else group.iloc[0:0]
        slot_coverage = pd.to_numeric(cgm["cgm_slot_coverage"], errors="coerce").dropna() if "cgm_slot_coverage" in cgm else pd.Series(dtype=float)
        long_gaps = pd.to_numeric(cgm["cgm_gaps_ge_2h"], errors="coerce").dropna() if "cgm_gaps_ge_2h" in cgm else pd.Series(dtype=float)
        window_rows.append(
            (
                rp.esc(window_id),
                rp.fmt(coverage.min() * 100 if not coverage.dropna().empty else float("nan"), 0, suffix="%"),
                rp.fmt(slot_coverage.iloc[0] * 100 if not slot_coverage.empty else float("nan"), 1, suffix="%"),
                rp.fmt(long_gaps.iloc[0] if not long_gaps.empty else float("nan"), 0),
                rp.esc("; ".join(rp.plain(r) for r in reasons) or "complete"),
            )
        )

    kind_rows = []
    if not quality_rows.empty and "kind" in quality_rows:
        counts = quality_rows.groupby("kind", dropna=False)["row_count"].sum().sort_values(ascending=False)
        kind_rows = [(rp.plain(kind), float(count)) for kind, count in counts.items()]

    body = [
        rp.verdict(status, label, headline, explanation),
        rp.tiles(
            [
                (rp.fmt(windows, 0), "download windows"),
                (rp.fmt(incomplete, 0), "with gaps or problems"),
                (rp.fmt(sum(v for _, v in kind_rows), 0), "rows downloaded"),
            ]
        ),
        rp.section(
            "What the problems are",
            rp.bar_list([(rp.plain(reason), float(count)) for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1])], color=rp.PRIMARY)
            + rp.caption("Number of download windows with each problem. One window can have several.")
            if reason_counts
            else "<p>No problems were recorded.</p>",
        ),
        rp.details(
            "Details: every window",
            rp.table(["window", "days covered", "CGM readings (% of 5-min slots)", "CGM gaps of 2 h or more", "download problems"], window_rows, numeric=[1, 2, 3])
            + rp.caption(
                "Days covered is the share of the requested dates the download spans, for the data type with the least. "
                "CGM readings is the share of 5-minute slots with a sensor reading. A window counts as incomplete only for download problems."
            ),
        ),
        rp.details("Details: rows by data type", rp.bar_list(kind_rows) if kind_rows else "<p>No rows.</p>"),
        rp.glossary(
            [
                ("Download window", "One request to Tandem Source for a fixed date range (about 30 days)."),
                ("Coverage", "Share of the requested period for which data came back."),
            ]
        ),
    ]
    return _write(path, rp.page(title="Tandem Data Coverage", question="Is the Tandem data complete?", subtitle=_generated(), body=body))


# ---------------------------------------------------------------------------
# Forecast validation
# ---------------------------------------------------------------------------

def _completeness_phrase(data_quality: dict[str, Any]) -> str:
    incomplete = int(data_quality.get("incomplete_window_count") or 0)
    status = str(data_quality.get("status") or "")
    if status == "good":
        return "all download windows complete"
    return f"{incomplete} download windows have gaps or other problems"


def _forecast_trace_figure(summary: dict[str, Any]) -> str:
    folds = (summary.get("walk_forward") or {}).get("folds", [])
    traced = [fold for fold in folds if fold.get("prediction_trace")]
    if not traced:
        return ""
    fold = traced[-1]
    trace = fold["prediction_trace"]
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(trace.get("timestamps", []), errors="coerce"),
            "actual": pd.to_numeric(pd.Series(trace.get("actual", [])), errors="coerce"),
            "predicted": pd.to_numeric(pd.Series(trace.get("predicted", [])), errors="coerce"),
            "lower": pd.to_numeric(pd.Series(trace.get("lower", [])), errors="coerce"),
            "upper": pd.to_numeric(pd.Series(trace.get("upper", [])), errors="coerce"),
        }
    ).dropna(subset=["timestamp"])
    if frame.empty:
        return ""
    end = frame["timestamp"].max()
    frame = frame.loc[frame["timestamp"] > end - pd.Timedelta(hours=48)].copy()
    # Keep the axis readable: CGM reports 40 to 400 mg/dL, so clip forecasts to a little beyond that.
    model_columns = ["predicted", "lower", "upper"]
    out_of_range = int(((frame[model_columns] < TRACE_CLIP_MGDL[0]) | (frame[model_columns] > TRACE_CLIP_MGDL[1])).any(axis=1).sum())
    frame[model_columns] = frame[model_columns].clip(*TRACE_CLIP_MGDL)
    figure = rp.time_series(
        frame["timestamp"].tolist(),
        [
            ("CGM", frame["actual"].tolist(), "#3B3B3B", False),
            ("model forecast", frame["predicted"].tolist(), rp.PRIMARY, False),
        ],
        band=(frame["lower"].tolist(), frame["upper"].tolist(), rp.PRIMARY),
        y_label="glucose (mg/dL)",
    )
    return (
        figure
        + rp.legend([("#3B3B3B", "CGM", False), (rp.PRIMARY, f"model forecast with {FORECAST_INTERVAL_PERCENT}% range", False)])
        + rp.caption(
            f"Last 48 hours of validation period {rp.esc(fold.get('fold'))}. Each point is the forecast made 30 minutes earlier."
            + (
                f" At {out_of_range} of {len(frame)} time points the forecast or its range went outside {TRACE_CLIP_MGDL[0]:.0f} to {TRACE_CLIP_MGDL[1]:.0f} mg/dL and is drawn at that edge."
                if out_of_range
                else ""
            )
        )
    )


def write_run_review_html(summary: dict[str, Any], path: str | Path) -> Path:
    walk_forward = summary.get("walk_forward") or {}
    aggregate = walk_forward.get("aggregate") or {}
    folds = walk_forward.get("folds") or []
    model_mae = aggregate.get("mae")
    baseline_mae = walk_forward.get("aggregate_persistence_mae")
    coverage = aggregate.get("coverage")
    policy = summary.get("recommendation_policy") or {}
    data_quality = summary.get("data_quality") or {}

    def finite(value: Any) -> bool:
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    fits = rp.forecast_fits(summary)
    failed_fits = sum(1 for diag in fits if not rp.sampler_healthy(diag))
    answer = rp.forecast_verdict(model_mae, baseline_mae, failed_fits=failed_fits, total_fits=len(fits)) if folds else rp.forecast_verdict(None, None)
    status, label = answer.status, answer.label
    headline, explanation = f"<b>{rp.esc(answer.headline)}</b>", rp.esc(answer.explanation)

    width = aggregate.get("interval_width")
    trust = [
        (
            f"<b>The model fitting failed its checks in {failed_fits} of {len(fits)} fits</b> (the sampler did not settle on an answer), so none of the model's numbers can be trusted."
            if failed_fits
            else f"Model fitting passed its checks in all {len(fits)} fits."
        )
        if fits
        else "",
        (
            f"The {FORECAST_INTERVAL_PERCENT}% forecast range contained the real value {rp.fmt(float(coverage) * 100, 0)}% of the time "
            f"(target {FORECAST_COVERAGE_RANGE[0]:.0%} to {FORECAST_COVERAGE_RANGE[1]:.0%}); on average the range was {rp.fmt(width, 0)} mg/dL wide."
        )
        if finite(coverage)
        else "",
        f"Tested on {len(folds)} later periods that the model was not trained on." if folds else "",
        (f"Data completeness: {_completeness_phrase(data_quality)}.") if data_quality else "",
    ]

    policy_status = str(policy.get("status") or "")
    reasons = [rp.plain(reason) for reason in policy.get("reasons") or []]
    if summary.get("recommendations"):
        next_steps = ["Settings suggestions were generated; see the run summary file."]
    elif policy_status == "skipped":
        next_steps = ["Settings suggestions were not requested for this run."]
    elif policy_status == "suppressed":
        next_steps = ["Settings suggestions were held back because: " + "; ".join(reasons) + "."]
    else:
        next_steps = ["No settings suggestions were produced."]

    fold_rows = [
        (
            f"period {fold.get('fold')}",
            float(fold["model_mae"]) if finite(fold.get("model_mae")) else None,
            float(fold["persistence_mae"]) if finite(fold.get("persistence_mae")) else None,
        )
        for fold in folds
    ]
    plottable = all(row[1] is None or row[1] < 10 * max((row[2] or 1), 1) for row in fold_rows)
    fold_table = rp.table(
        ["period", "test dates", "test points", "model error (mg/dL)", "baseline error (mg/dL)", f"in {FORECAST_INTERVAL_PERCENT}% range"],
        [
            (
                rp.esc(fold.get("fold")),
                rp.esc(_date_span(fold.get("test_start_timestamp"), fold.get("test_end_timestamp"))),
                rp.fmt(fold.get("n_test"), 0),
                rp.fmt(fold.get("model_mae")),
                rp.fmt(fold.get("persistence_mae")),
                rp.fmt(float(fold["model_coverage"]) * 100 if finite(fold.get("model_coverage")) else None, 0, suffix="%"),
            )
            for fold in folds
        ],
        numeric=[2, 3, 4, 5],
    )

    sampler_rows = []
    for name, diag in [(f"period {fold.get('fold')}", fold.get("fit_diagnostics") or {}) for fold in folds] + [("final fit", summary.get("fit_diagnostics") or {})]:
        if not diag:
            continue
        sampler_rows.append(
            (
                rp.esc(name),
                rp.esc(diag.get("chains")),
                rp.esc(diag.get("divergences")),
                rp.fmt(diag.get("rhat_max"), 3),
                rp.fmt(diag.get("ess_bulk_min"), 0),
                "OK" if rp.sampler_healthy(diag) else "failed",
            )
        )

    body = [
        rp.verdict(status, label, headline, explanation),
        rp.tiles(
            [
                (rp.fmt(model_mae, suffix=" mg/dL"), "model forecast error"),
                (rp.fmt(baseline_mae, suffix=" mg/dL"), "error if glucose stays the same"),
                (rp.fmt(float(coverage) * 100 if finite(coverage) else None, 0, suffix="%"), f"of readings inside the {FORECAST_INTERVAL_PERCENT}% range"),
            ]
        ),
        rp.section("How much should I trust this?", rp.bullets(trust)),
        rp.section("What to do next", rp.bullets(next_steps)),
        rp.details(
            "Details: error in each validation period",
            (
                rp.paired_dots(fold_rows, first_label="model", second_label="baseline", axis_label="average forecast error (mg/dL)")
                + rp.legend([(rp.PRIMARY, "model", False), (rp.COMPARATOR, "glucose stays the same", False)])
                if plottable
                else rp.caption("The model's error in at least one period is too large to plot on the same scale as the baseline; see the table.")
            )
            + fold_table,
            open_=True,
        ),
        rp.details("Details: forecast against CGM", _forecast_trace_figure(summary) or "<p>No forecast trace was saved.</p>"),
        rp.details(
            "Details: model fitting checks",
            rp.table(["fit", "chains", "divergences", "R-hat (max)", "effective samples (min)", "status"], sampler_rows, numeric=[1, 2, 3, 4])
            + rp.caption("OK means R-hat under 1.01, no divergences and at least 100 effective samples per chain. Anything else means the sampler did not settle on an answer and the fit is not reliable.")
            if sampler_rows
            else "<p>No fitting diagnostics were recorded.</p>",
        ),
        rp.glossary(
            [
                ("Forecast error", "Average distance in mg/dL between the 30-minute forecast and the CGM reading that followed (mean absolute error)."),
                ("Baseline", "Forecasting that glucose in 30 minutes equals glucose now. A useful model must beat it."),
                (f"{FORECAST_INTERVAL_PERCENT}% range", f"The band the model expects to contain the real value {FORECAST_INTERVAL_PERCENT}% of the time."),
                ("Validation period", "A later stretch of data held out from training, used to test the model as if forecasting the future."),
            ]
        ),
    ]
    return _write(path, rp.page(title="Glucose Forecast Check", question="Can the model forecast glucose 30 minutes ahead?", subtitle=_generated(), body=body))


# ---------------------------------------------------------------------------
# Therapy evidence (overnight basal)
# ---------------------------------------------------------------------------


def _overnight_figure(research_result: TherapyResearchResult) -> str:
    frame = research_result.research_frame
    if frame.empty or "therapy_segment" not in frame:
        return ""
    frame = frame.loc[frame["therapy_segment"].astype(str).eq("overnight")].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).tail(288)
    if frame.empty:
        return ""
    figure = rp.time_series(
        frame["timestamp"].tolist(),
        [("glucose", pd.to_numeric(frame["glucose"], errors="coerce").tolist(), rp.PRIMARY, False)],
        y_label="glucose (mg/dL)",
        reference_lines=[(180.0, "180"), (70.0, "70")],
    )
    nights = int(frame["timestamp"].dt.normalize().nunique())
    return figure + rp.caption(
        f"The last {len(frame) * 5 / 60:.0f} hours of overnight CGM in the analysis, spread over {nights} nights; daytime is left out, so the line breaks between nights. "
        "Dashed lines mark 70 and 180 mg/dL."
    )


def write_therapy_evidence_review_html(
    preparation: ModelDataPreparationResult,
    research_result: TherapyResearchResult,
    path: str | Path,
    *,
    validation_result: TherapyInfraValidationResult | None = None,
    artifact_root: str | Path | None = None,
    artifact_href_prefix: str = "",
) -> Path:
    path = Path(path)
    root = Path(artifact_root) if artifact_root is not None else path.parent
    overnight, exclusions = summarize_overnight_basal_evidence(research_result)
    recommendations = research_result.recommendations
    candidates = recommendations.loc[recommendations["status"].astype(str).eq("candidate")] if not recommendations.empty else recommendations
    clean_nights = int(overnight.get("clean_nights") or 0)
    clean_hours = int(overnight.get("clean_rows") or 0) * 5 / 60

    status_code = str(overnight.get("status"))
    if status_code == "identifiable":
        status, label = "good", "Enough clean overnight data"
        headline = f"<b>{clean_nights} nights</b> have enough quiet overnight time to judge basal ({rp.fmt(clean_hours, 0)} hours in all)."
    elif status_code == "weakly_identifiable":
        status, label = "warn", "Limited clean overnight data"
        headline = f"<b>Only {clean_nights} nights</b> have quiet overnight time ({rp.fmt(clean_hours, 0)} hours). Any conclusion about basal is weak."
    else:
        status, label = "bad", "Not enough clean overnight data"
        headline = f"<b>Overnight basal can't be judged yet:</b> {rp.esc(rp.plain(overnight.get('reason')))}."
    if not candidates.empty:
        lead = candidates.iloc[0]
        explanation = (
            f"The strongest signal: {rp.esc(lead['parameter'])} ({rp.esc(lead['segment'])}) "
            f"{rp.esc(rp.plain(lead['expected_direction']))} by about {rp.fmt(lead['proposed_change_percent'], 0)}%, "
            f"{rp.esc(rp.plain(lead['confidence']))} confidence. This is research evidence, not a setting change."
        )
    else:
        explanation = "No setting changes are suggested."

    gate_rows = []
    if not research_result.research_gate.empty:
        for row in research_result.research_gate.itertuples(index=False):
            gate_rows.append((rp.esc(rp.plain(row.parameter)), rp.esc(rp.plain(row.identifiability)), rp.esc(rp.plain(row.gate_status)), rp.esc(rp.plain(row.closed_loop_confounding_risk))))

    validation_line = ""
    if validation_result is not None and not validation_result.scenario_results.empty:
        passed = int(validation_result.scenario_results["passed"].sum())
        total = int(len(validation_result.scenario_results))
        validation_line = f"On simulated data with known answers, the method behaved correctly in {passed} of {total} scenarios."

    body = [
        rp.verdict(status, label, headline, explanation),
        rp.tiles(
            [
                (rp.fmt(clean_nights, 0), f"clean nights (of {rp.fmt(overnight.get('overnight_nights'), 0)})"),
                (rp.fmt(clean_hours, 0), "clean overnight hours"),
                (rp.fmt(overnight.get("mean_glucose"), 0, suffix=" mg/dL"), "average glucose in clean stretches"),
            ]
        ),
        rp.section(
            "How much should I trust this?",
            rp.bullets(
                [
                    "A night counts as clean when there was no meal, bolus, exercise or Control-IQ adjustment in the previous 2 hours and CGM was working.",
                    f"Clean data span {rp.fmt(overnight.get('stable_epochs'), 0)} periods with unchanged pump settings." if overnight.get("stable_epochs") else "",
                    validation_line,
                    f"Data: {rp.esc(_date_span(preparation.final_dataset_start, preparation.final_dataset_end))}; Apple Health {'included' if preparation.apple_available else 'not included'}.",
                ]
            ),
        ),
        rp.details(
            "Details: why overnight time was excluded",
            rp.bar_list(
                [(rp.plain(row.reason), float(row.count) * 5 / 60) for row in exclusions.itertuples(index=False) if row.count],
                value_format="{:,.0f} h",
            )
            + rp.caption("Hours of overnight data excluded for each reason. One stretch can have several reasons.")
            if not exclusions.empty and pd.to_numeric(exclusions["count"], errors="coerce").fillna(0).gt(0).any()
            else "<p>No overnight time was excluded.</p>",
        ),
        rp.details("Details: recent overnight glucose", _overnight_figure(research_result) or "<p>No overnight data.</p>"),
        rp.details("Details: what can be estimated for each setting", rp.table(["setting", "can it be estimated?", "status", "Control-IQ interference"], gate_rows)),
        rp.details(
            "Files behind this page",
            _files(
                root,
                artifact_href_prefix,
                [
                    ("model_data_preparation.md", "how the 5-minute dataset was built"),
                    ("therapy_research_gate.md", "which settings can be estimated"),
                    ("therapy_feature_audit.md", "inputs used by the models"),
                    ("therapy_model_comparison.md", "how the candidate models compare"),
                    ("therapy_segment_evidence.csv", "evidence for each time-of-day segment"),
                    ("therapy_recommendation_research.md", "research-only setting signals"),
                    ("tandem_source_report_card.md", "Tandem data quality"),
                    ("apple_source_report_card.md", "Apple Health data quality"),
                ]
                + (
                    [
                        ("therapy_infra_validation.md", "simulation checks of the method"),
                        ("therapy_synthetic_results.csv", "results of each simulation scenario"),
                    ]
                    if validation_result is not None
                    else []
                ),
            ),
        ),
        rp.glossary(
            [
                ("Basal", "Background insulin delivered continuously by the pump."),
                ("Clean overnight time", "Overnight periods with no meal, bolus, exercise or Control-IQ adjustment in the previous 2 hours and working CGM."),
                ("Stable settings period", "A stretch of time in which the programmed pump settings did not change."),
            ]
        ),
    ]
    return _write(
        path,
        rp.page(
            title="Overnight Basal Evidence",
            question="Is there enough data to judge my overnight basal?",
            subtitle=f"{rp.esc(_date_span(preparation.final_dataset_start, preparation.final_dataset_end))}. {_generated()}",
            body=body,
        ),
    )


# ---------------------------------------------------------------------------
# Latent meal / breakfast I:C
# ---------------------------------------------------------------------------


def _period_label(row: Any) -> str:
    start = pd.to_datetime(row["period_start"], errors="coerce")
    end = pd.to_datetime(row["period_end"], errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return "no breakfasts"
    return f"{start:%b %Y}" if (start.year, start.month) == (end.year, end.month) else f"{start:%b %Y} to {end:%b %Y}"


def _meal_points(result: LatentMealResearchResult) -> list[rp.ScatterPoint]:
    estimates = result.latent_carb_estimates
    windows = result.meal_windows
    if estimates is None or estimates.empty or windows.empty or "stated_carbs" not in windows:
        return []
    merged = estimates.merge(windows[["meal_id", "stated_carbs"]], on="meal_id", how="left")
    points = []
    for row in merged.itertuples(index=False):
        logged = pd.to_numeric(getattr(row, "stated_carbs", None), errors="coerce")
        estimate = pd.to_numeric(getattr(row, "latent_carb_median", None), errors="coerce")
        if pd.isna(logged) or pd.isna(estimate) or float(logged) <= 0:
            continue
        lower = pd.to_numeric(getattr(row, "latent_carb_lower_90", None), errors="coerce")
        upper = pd.to_numeric(getattr(row, "latent_carb_upper_90", None), errors="coerce")
        when = pd.to_datetime(getattr(row, "meal_start", None), errors="coerce")
        points.append(
            rp.ScatterPoint(
                x=float(logged),
                y=float(estimate),
                lower=None if pd.isna(lower) else float(lower),
                upper=None if pd.isna(upper) else float(upper),
                strong=str(getattr(row, "analysis_tier", "")) == "clean_reference",
                tooltip=f"{when:%Y-%m-%d} logged {float(logged):.0f} g, implied {float(estimate):.0f} g ({float(lower):.0f} to {float(upper):.0f}), bolus {float(row.bolus_units):.2f} U"
                if pd.notna(when)
                else f"logged {float(logged):.0f} g, implied {float(estimate):.0f} g",
            )
        )
    return points


def _latent_counts(result: LatentMealResearchResult) -> dict[str, int]:
    windows = result.meal_windows
    included = pd.to_numeric(windows.get("included"), errors="coerce").fillna(0).astype(bool) if not windows.empty else pd.Series(dtype=bool)
    estimates = result.latent_carb_estimates if result.latent_carb_estimates is not None else pd.DataFrame()
    tiers = estimates.get("analysis_tier", pd.Series(dtype=str)).astype(str) if not estimates.empty else pd.Series(dtype=str)
    logged = 0
    if not windows.empty and "stated_carbs" in windows:
        logged = int((included & pd.to_numeric(windows["stated_carbs"], errors="coerce").gt(0)).sum())
    return {
        "candidates": int(len(windows)),
        "accepted": int(included.sum()),
        "logged": logged,
        "clean": int(tiers.eq("clean_reference").sum()),
    }


def _latent_files(root: Path, prefix: str, scope: str) -> str:
    if scope == "foundation":
        entries = [
            ("latent_meal_research_gate.md", "what can be estimated"),
            ("meal_event_registry.csv", "every detected meal"),
            ("first_meal_clean_window_registry.csv", "every first meal and whether it was used"),
            ("first_meal_clean_window_audit.md", "summary of the breakfast selection"),
            ("first_meal_exclusion_summary.csv", "counts of exclusion reasons"),
        ]
    else:
        entries = [
            ("latent_carb_estimator_report.md", "estimator summary and effective I:C table"),
            ("latent_carb_estimates.csv", "per-breakfast carb estimates with 90% ranges"),
            ("effective_icr_evidence.csv", "effective I:C by time of day"),
            ("latent_meal_window_audit.csv", "every first meal and why it was used or excluded"),
            ("meal_window_audit.md", "summary of the breakfast selection"),
            ("meal_event_registry.csv", "every detected meal"),
            ("latent_meal_research_gate.md", "what can be estimated"),
            ("latent_meal_model_comparison.md", "comparison of the candidate estimators"),
        ]
    return _files(root, prefix, entries)


def _latent_glossary() -> str:
    return rp.glossary(
        [
            ("I:C ratio", "Grams of carbohydrate covered by one unit of insulin. A lower number means more insulin per gram."),
            ("Effective I:C", "The ratio that best explains how your glucose actually moved after these breakfasts, given what you logged and the insulin delivered."),
            ("90% range", "The true value lies in this range with 90% probability, given the model and data."),
            ("Glucose-implied carbs", "The carbohydrate amount that best explains the post-meal CGM curve, given the bolus and Control-IQ delivery."),
            ("Fit error", "Typical distance (root mean square) between the model's predicted glucose curve and the CGM after a meal."),
        ]
    )


def _write_latent_foundation(result: LatentMealResearchResult, path: Path, root: Path, prefix: str) -> Path:
    counts = _latent_counts(result)
    exclusions = result.first_meal_exclusion_summary
    status, label = ("good", "Breakfasts available") if counts["accepted"] else ("bad", "No usable breakfasts")
    body = [
        rp.verdict(
            status,
            label,
            f"<b>{counts['accepted']} of {counts['candidates']} first meals of the day</b> are clean enough to analyse.",
            "Run the full analysis (--research-scope full) to estimate carbs and the effective I:C from them.",
        ),
        rp.details(
            "Details: why breakfasts were excluded",
            rp.bar_list([(rp.plain(row.reason), float(row.count)) for row in exclusions.itertuples(index=False)]) if not exclusions.empty else "<p>No exclusions.</p>",
            open_=True,
        ),
        rp.details("Files behind this page", _latent_files(root, prefix, "foundation")),
        _latent_glossary(),
    ]
    return _write(path, rp.page(title="Breakfast Window Review", question="Which breakfasts are clean enough to analyse?", subtitle=_generated(), body=body))


def write_latent_meal_review_html(
    result: LatentMealResearchResult,
    path: str | Path,
    *,
    artifact_root: str | Path | None = None,
    artifact_href_prefix: str = "",
) -> Path:
    path = Path(path)
    root = Path(artifact_root) if artifact_root is not None else path.parent
    if str(getattr(result, "research_scope", "foundation")) == "foundation":
        return _write_latent_foundation(result, path, root, artifact_href_prefix)

    counts = _latent_counts(result)
    estimates = result.latent_carb_estimates if result.latent_carb_estimates is not None else pd.DataFrame()
    fit_rmse = pd.to_numeric(estimates.get("trajectory_fit_rmse_mgdl"), errors="coerce").dropna() if not estimates.empty else pd.Series(dtype=float)
    fallback_rows = int((estimates.get("estimator", pd.Series(dtype=str)).astype(str) != "trajectory").sum()) if not estimates.empty else 0
    median_fit = float(fit_rmse.median()) if not fit_rmse.empty else float("nan")
    windows = result.meal_windows
    logged_carbs = pd.to_numeric(windows.loc[pd.to_numeric(windows.get("included"), errors="coerce").fillna(0).astype(bool), "stated_carbs"], errors="coerce").dropna() if not windows.empty and "stated_carbs" in windows else pd.Series(dtype=float)
    logged_carbs = logged_carbs[logged_carbs > 0]

    evidence = result.effective_icr_evidence if result.effective_icr_evidence is not None else pd.DataFrame()
    estimated = evidence.loc[pd.to_numeric(evidence.get("effective_icr"), errors="coerce").notna()] if not evidence.empty else evidence
    current_rows = evidence.loc[evidence["is_current_setting"].astype(bool)] if not evidence.empty and "is_current_setting" in evidence else evidence.iloc[0:0]
    current = current_rows.iloc[0] if not current_rows.empty else None

    figure = ""
    if estimated.empty:
        status, label = "info", "Can't judge the I:C setting yet"
        headline = "<b>No accepted window has a logged carb count</b>, so the I:C setting itself can't be assessed."
        explanation = (
            "Without logged carbs, an under-counted meal and a too-weak ratio produce the same glucose curve. "
            "Log carbs on the pump at breakfast, or keep a <code>carbs.csv</code> (<code>timestamp,carb_grams</code>) in the raw data folder."
        )
        if counts["logged"]:
            headline = f"<b>Only {counts['logged']} usable breakfasts have logged carbs</b>; at least 5 are needed to estimate the I:C."
    elif current is not None and not math.isfinite(float(current["effective_icr"])):
        last = estimated.iloc[-1]
        status, label = "info", "Too early to judge your current setting"
        headline = (
            f"<b>Your current morning I:C of {float(current['programmed_icr']):.1f} g/U has {int(current['n_meals'])} usable breakfasts so far</b>; "
            "at least 5 are needed."
        )
        explanation = (
            f"Under your previous setting of {float(last['programmed_icr']):.1f} g/U ({rp.esc(_period_label(last))}), breakfasts behaved like "
            f"{float(last['effective_icr']):.1f} g/U (90% range {float(last['effective_icr_lower_90']):.1f} to {float(last['effective_icr_upper_90']):.1f}): "
            f"{rp.esc(rp.plain(last['status']))}."
        )
    else:
        lead = current if current is not None else estimated.iloc[-1]
        eff, lo, hi, prog = (float(lead[k]) for k in ("effective_icr", "effective_icr_lower_90", "effective_icr_upper_90", "programmed_icr"))
        answer = rp.icr_verdict(str(lead["segment"]), eff, lo, hi, prog, str(lead["status"]))
        status, label = answer.status, answer.label
        headline = f"<b>{rp.esc(answer.headline)}</b>"
        explanation = rp.esc(answer.explanation) + (f" Based on {int(lead['n_meals'])} breakfasts at this setting ({rp.esc(_period_label(lead))})." if "period_start" in lead else "")
        figure = rp.interval_chart(
            estimate=eff,
            lower=lo,
            upper=hi,
            reference=prog,
            reference_label=f"pump setting {prog:.1f}",
            axis_label="grams of carbohydrate covered by 1 unit of insulin",
            value_label=f"{eff:.1f} ({lo:.1f} to {hi:.1f})",
        )
    period_rows = [
        (
            rp.esc(_period_label(row)) + (" (current)" if bool(row["is_current_setting"]) else ""),
            rp.fmt(row["programmed_icr"]),
            rp.esc(row["settings_source"]),
            rp.fmt(row["n_meals"], 0),
            f"{rp.fmt(row['effective_icr'])} ({rp.fmt(row['effective_icr_lower_90'])} to {rp.fmt(row['effective_icr_upper_90'])})" if math.isfinite(float(row["effective_icr"])) else "&ndash;",
            rp.esc(rp.plain(row["status"])),
        )
        for _, row in evidence.iterrows()
    ] if not evidence.empty and "period_start" in evidence else []

    trust = [
        (
            f"<b>Model fit:</b> the fitted glucose curves are within {median_fit:.1f} mg/dL of CGM on a typical breakfast "
            + ("(good; under 10)." if median_fit <= FIT_ERROR_GUIDE_MGDL else "(poor; over 10, so real meals don't match the model well and the result is doubtful).")
        )
        if math.isfinite(median_fit)
        else "",
        f"{fallback_rows} breakfasts used the older, less accurate estimator." if fallback_rows else "",
        f"Only {counts['clean']} of {counts['accepted']} breakfasts passed every quality check. The rest started out of range, followed recent insulin or had Control-IQ adjusting beforehand, and count as weaker evidence."
        if counts["accepted"] and counts["clean"] < counts["accepted"]
        else "",
        f"Breakfasts were small (median {logged_carbs.median():.0f} g logged), so this says little about large meals." if not logged_carbs.empty else "",
        "It assumes your correction factor is right and your logged carbs are accurate. Consistent under-counting makes the ratio look stronger than it is." if not estimated.empty else "",
    ]
    next_steps = (
        [
            "Nothing to change on the pump from this result alone. Bring it to your endocrinologist as supporting evidence."
            if status == "good"
            else "Discuss this with your endocrinologist before changing the pump.",
            "To narrow the range, log weighed or packaged breakfasts for a few weeks.",
        ]
        if not estimated.empty
        else ["Log carbs on the pump for breakfasts, then rerun this analysis."]
    )

    points = _meal_points(result)
    strong = sum(p.strong for p in points)
    exclusions = result.first_meal_exclusion_summary
    per_meal_rows = []
    if not estimates.empty:
        merged = estimates.merge(windows[["meal_id", "stated_carbs"]], on="meal_id", how="left") if "stated_carbs" in windows else estimates
        for row in merged.sort_values("meal_start").itertuples(index=False):
            per_meal_rows.append(
                (
                    rp.esc(f"{pd.to_datetime(row.meal_start):%Y-%m-%d %H:%M}"),
                    rp.fmt(getattr(row, "stated_carbs", None), 0),
                    f"{rp.fmt(row.latent_carb_median, 0)} ({rp.fmt(row.latent_carb_lower_90, 0)} to {rp.fmt(row.latent_carb_upper_90, 0)})",
                    rp.fmt(row.bolus_units, 2),
                    rp.fmt(getattr(row, "trajectory_fit_rmse_mgdl", None)),
                    rp.esc(rp.plain(getattr(row, "analysis_tier", ""))),
                )
            )

    body = [
        rp.verdict(status, label, headline, explanation, figure),
        rp.tiles(
            [
                (rp.fmt(counts["accepted"], 0), f"breakfasts clean enough to use (of {counts['candidates']})"),
                (rp.fmt(counts["logged"], 0), "of those with carbs logged"),
                (rp.fmt(median_fit, suffix=" mg/dL"), "typical fit error (good is under 10)"),
            ]
        ),
        rp.section("How much should I trust this?", rp.bullets(trust)),
        rp.section("What to do next", rp.bullets(next_steps)),
        rp.details(
            "Details: each breakfast",
            (
                rp.scatter_with_ranges(points, x_label="carbs you logged on the pump (g)", y_label="carbs implied by your glucose (g)", identity_label="implied = logged")
                + rp.legend([(rp.PRIMARY, f"passed every check ({strong})", False), (rp.SECONDARY, f"passed looser checks ({len(points) - strong})", "ring")])
                + rp.caption(
                    "Each dot is one breakfast; vertical lines are 90% ranges. Horizontal position is what you logged; vertical position is the amount "
                    "the glucose curve implies given the bolus and Control-IQ delivery. Dots above the dashed line raised glucose more than the logged carbs predict."
                )
                if points
                else "<p>No breakfasts with logged carbs.</p>"
            )
            + rp.table(["time", "logged g", "implied g (90% range)", "bolus U", "fit error mg/dL", "quality"], per_meal_rows, numeric=[1, 2, 3, 4]),
        ),
        rp.details(
            "Details: I:C by settings period",
            rp.table(["settings period", "pump I:C (g/U)", "settings from", "breakfasts used", "effective I:C (90% range)", "result"], period_rows, numeric=[1, 3])
            + rp.caption(
                "Each breakfast is compared with the I:C and correction factor in force for it: the values the bolus calculator recorded, or the pump "
                "profile for hand-logged meals. A settings period is a run of breakfasts with the same programmed I:C."
            ),
        )
        if period_rows
        else "",
        rp.details(
            "Details: which breakfasts were used",
            rp.bar_list(
                [
                    ("first meals of the day", counts["candidates"]),
                    ("passed quality checks", counts["accepted"]),
                    ("with logged carbs", counts["logged"]),
                    ("passed every check", counts["clean"]),
                ],
                max_value=max(counts["candidates"], 1),
            )
            + "<h3 style='font-size:15px;margin:16px 0 4px'>Why breakfasts were excluded</h3>"
            + (
                rp.bar_list([(rp.plain(row.reason), float(row.count)) for row in exclusions.sort_values("count", ascending=False).itertuples(index=False)], color=rp.SECONDARY)
                if not exclusions.empty
                else "<p>No exclusions.</p>"
            )
            + rp.caption("Number of first meals with each problem. One meal can have several, so the counts add up to more than the number excluded."),
        ),
        rp.details("Files behind this page", _latent_files(root, artifact_href_prefix, "full")),
        _latent_glossary(),
    ]
    return _write(
        path,
        rp.page(
            title="Breakfast I:C Review",
            question="Is my morning I:C ratio right?",
            subtitle=f"{rp.esc(_date_span(result.prepared_dataset.frame['timestamp'].min() if 'timestamp' in result.prepared_dataset.frame else None, result.prepared_dataset.frame['timestamp'].max() if 'timestamp' in result.prepared_dataset.frame else None))}. {_generated()}",
            body=body,
        ),
    )


# ---------------------------------------------------------------------------
# Current status
# ---------------------------------------------------------------------------


def write_current_status_html(
    payload: dict[str, Any],
    path: str | Path,
    *,
    therapy_href: str,
    forecast_href: str,
) -> Path:
    state = str(payload.get("overall_state") or "blocked")
    status, label = {
        "recommendation_ready": ("good", "A setting change is supported"),
        "no_change_supported": ("info", "No change supported"),
    }.get(state, ("bad", "Blocked"))
    data_prep = payload.get("data_prep") or {}
    forecast = payload.get("forecast") or {}
    overnight = (payload.get("therapy") or {}).get("overnight") or {}
    blockers = [
        f"<b>{rp.esc(item.get('label') or rp.plain(item.get('code')))}</b>: {rp.esc(item.get('detail') or '')}"
        for item in payload.get("primary_blockers") or []
    ]
    body = [
        rp.verdict(status, label, f"<b>{rp.esc(payload.get('headline') or 'Status unavailable')}</b>", rp.esc(payload.get("summary") or "")),
        rp.tiles(
            [
                (rp.fmt(data_prep.get("final_row_count"), 0), "5-minute rows in the dataset"),
                (rp.fmt(overnight.get("clean_nights"), 0), "clean nights for basal"),
                (rp.fmt(forecast.get("model_mae"), suffix=" mg/dL"), "forecast error (model)"),
            ]
        ),
        rp.section("What is blocking a recommendation", rp.bullets(blockers) or "<p>Nothing is blocking.</p>"),
        rp.section("What to do next", rp.bullets(rp.esc(item) for item in payload.get("next_actions") or []) or "<p>No next actions recorded.</p>"),
        rp.section(
            "Detailed reports",
            rp.bullets(
                [
                    f"<a href='{rp.esc(therapy_href)}'>Overnight basal evidence</a>",
                    f"<a href='{rp.esc(forecast_href)}'>Glucose forecast check</a>",
                ]
            ),
        ),
        rp.details(
            "Details: data",
            rp.table(
                ["item", "value"],
                [
                    ("Apple Health included", rp.esc("yes" if data_prep.get("apple_available") else "no")),
                    ("Tandem and Apple Health overlap", rp.esc(_date_span(data_prep.get("overlap_start"), data_prep.get("overlap_end")))),
                    ("Forecast baseline error", rp.fmt(forecast.get("persistence_mae"), suffix=" mg/dL")),
                    ("Model fitting checks", rp.esc(rp.plain(forecast.get("sampler_health")))),
                ],
            ),
        ),
    ]
    return _write(
        path,
        rp.page(
            title="Project Status",
            question="Where do things stand?",
            subtitle=f"Run {rp.esc(payload.get('run_id') or 'unknown')}. {_generated()}",
            body=body,
        ),
    )
