from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html

from .health_auto_export import ModelDataPreparationResult
from .therapy_research import LatentMealResearchResult, TherapyInfraValidationResult, TherapyResearchResult, summarize_overnight_basal_evidence


def _figure_div(figure: go.Figure, *, include_plotlyjs: bool) -> str:
    return to_html(
        figure,
        include_plotlyjs="inline" if include_plotlyjs else False,
        full_html=False,
        config={"responsive": True, "displaylogo": False},
    )


def _page_shell(*, title: str, banner_html: str, sections: list[str]) -> str:
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='utf-8' />",
            f"<title>{html.escape(title)}</title>",
            "<meta name='viewport' content='width=device-width, initial-scale=1' />",
            "<style>",
            "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; color: #16202a; background: #f6f8fb; }",
            "h1, h2, h3 { color: #10233f; }",
            ".banner { padding: 16px 18px; border-radius: 12px; margin-bottom: 20px; font-weight: 600; }",
            ".banner.good { background: #e7f6ea; border: 1px solid #9bd0a9; }",
            ".banner.degraded { background: #fff3db; border: 1px solid #f0c36d; }",
            ".banner.broken { background: #fde8e8; border: 1px solid #ef9a9a; }",
            ".card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin: 12px 0 24px; }",
            ".card { background: white; border: 1px solid #d8e1ec; border-radius: 12px; padding: 14px 16px; box-shadow: 0 1px 2px rgba(16,35,63,0.05); }",
            ".muted { color: #4f6177; }",
            "table { width: 100%; border-collapse: collapse; background: white; }",
            "th, td { padding: 10px 12px; border: 1px solid #d8e1ec; text-align: left; }",
            "th { background: #eef3f9; }",
            ".section { margin: 24px 0 32px; }",
            ".empty-state { background: white; border: 1px dashed #b7c6d8; border-radius: 12px; padding: 18px; }",
            "code { background: #eef3f9; padding: 1px 4px; border-radius: 4px; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{html.escape(title)}</h1>",
            banner_html,
            *sections,
            "</body>",
            "</html>",
        ]
    )


def _data_quality_banner(summary: dict[str, Any], *, recommendation_policy: dict[str, Any] | None = None) -> str:
    data_quality = summary.get("data_quality") or {}
    status = str(data_quality.get("status") or "broken")
    recommendation_status = None if recommendation_policy is None else recommendation_policy.get("status")
    extra = ""
    if recommendation_status is not None:
        extra = f" Recommendation status: <code>{html.escape(str(recommendation_status))}</code>."
    return (
        f"<div class='banner {html.escape(status)}'>"
        f"Data quality is <code>{html.escape(status)}</code>. "
        f"Incomplete windows: <code>{html.escape(str(data_quality.get('incomplete_window_count', 0)))}</code>."
        f"{extra}"
        "</div>"
    )


def _window_timeline_figure(quality_rows: pd.DataFrame) -> go.Figure:
    timeline_rows: list[dict[str, Any]] = []
    grouped = quality_rows.groupby("window_id", dropna=False)
    for window_id, window_rows in grouped:
        first_row = window_rows.iloc[0]
        requested_start = pd.to_datetime(first_row.get("requested_start"), errors="coerce", utc=True)
        requested_end = pd.to_datetime(first_row.get("requested_end"), errors="coerce", utc=True)
        if pd.notna(requested_start) and pd.notna(requested_end):
            timeline_rows.append(
                {
                    "window_id": str(window_id),
                    "series": "requested",
                    "start": requested_start,
                    "end": requested_end + pd.Timedelta(days=1),
                }
            )
        observed_first = pd.to_datetime(window_rows["observed_first_timestamp"], errors="coerce", utc=True).dropna()
        observed_last = pd.to_datetime(window_rows["observed_last_timestamp"], errors="coerce", utc=True).dropna()
        if not observed_first.empty and not observed_last.empty:
            timeline_rows.append(
                {
                    "window_id": str(window_id),
                    "series": "observed",
                    "start": observed_first.min(),
                    "end": observed_last.max(),
                }
            )
    if not timeline_rows:
        return go.Figure()
    frame = pd.DataFrame(timeline_rows)
    fig = px.timeline(
        frame,
        x_start="start",
        x_end="end",
        y="window_id",
        color="series",
        color_discrete_map={"requested": "#9aa9bb", "observed": "#1f77b4"},
        title="Window Timeline",
    )
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), legend_title_text="")
    return fig


def _row_counts_figure(quality_rows: pd.DataFrame) -> go.Figure:
    if quality_rows.empty:
        return go.Figure()
    counts = (
        quality_rows.groupby("kind", dropna=False)["row_count"]
        .sum()
        .reset_index()
        .sort_values("kind")
    )
    fig = px.bar(counts, x="kind", y="row_count", title="Row Counts by Kind")
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), xaxis_title=None, yaxis_title="rows")
    return fig


def _reasons_table_html(quality_rows: pd.DataFrame) -> str:
    if quality_rows.empty:
        return "<div class='section'><h2>Completeness Reasons</h2><div class='empty-state'>No window-quality rows were available.</div></div>"
    rows: list[str] = []
    for window_id, window_rows in quality_rows.groupby("window_id", dropna=False):
        reasons: list[str] = []
        for raw in window_rows["completeness_reasons"].fillna("[]").astype(str):
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = []
            for reason in parsed:
                if reason not in reasons:
                    reasons.append(str(reason))
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(window_id))}</td>"
            f"<td>{html.escape(', '.join(reasons) if reasons else 'none')}</td>"
            "</tr>"
        )
    return "\n".join(
        [
            "<div class='section'>",
            "<h2>Completeness Reasons</h2>",
            "<table>",
            "<thead><tr><th>window</th><th>reasons</th></tr></thead>",
            "<tbody>",
            *rows,
            "</tbody></table>",
            "</div>",
        ]
    )


def _latest_window_card_html(quality_rows: pd.DataFrame) -> str:
    if quality_rows.empty:
        return "<div class='section'><h2>Latest Window</h2><div class='empty-state'>No latest window could be identified.</div></div>"
    ordered = quality_rows.copy()
    ordered["requested_end_dt"] = pd.to_datetime(ordered["requested_end"], errors="coerce")
    latest = ordered.sort_values(["requested_end_dt", "window_id"], na_position="last").iloc[-1]
    coverage_fraction = float(latest.get("coverage_fraction", 0.0) or 0.0)
    return "\n".join(
        [
            "<div class='section'>",
            "<h2>Latest Window</h2>",
            "<div class='card-grid'>",
            "<div class='card'>",
            f"<div><strong>window_id</strong>: <code>{html.escape(str(latest.get('window_id')))}</code></div>",
            f"<div><strong>kind</strong>: {html.escape(str(latest.get('kind')))}</div>",
            f"<div><strong>coverage_fraction</strong>: {html.escape(f'{coverage_fraction:.3f}')}</div>",
            f"<div><strong>observed span</strong>: {html.escape(str(latest.get('observed_first_timestamp')))} to {html.escape(str(latest.get('observed_last_timestamp')))}</div>",
            f"<div><strong>reasons</strong>: {html.escape(str(latest.get('completeness_reasons')))}</div>",
            "</div>",
            "</div>",
            "</div>",
        ]
    )


def write_coverage_review_html(summary: dict[str, Any], quality_rows: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    include_js = True
    figures = []
    for figure in [_window_timeline_figure(quality_rows), _row_counts_figure(quality_rows)]:
        figures.append(_figure_div(figure, include_plotlyjs=include_js))
        include_js = False

    sections = [
        "<div class='section'><h2>Window Timeline</h2>" + figures[0] + "</div>",
        "<div class='section'><h2>Rows by Kind</h2>" + figures[1] + "</div>",
        _reasons_table_html(quality_rows),
        _latest_window_card_html(quality_rows),
    ]
    html_text = _page_shell(
        title="Bayesian T1DM Coverage Review",
        banner_html=_data_quality_banner(summary),
        sections=sections,
    )
    path.write_text(html_text, encoding="utf-8")
    return path


def _run_trace_figures(summary: dict[str, Any]) -> list[str]:
    traces: list[str] = []
    include_js = True
    for fold in (summary.get("walk_forward") or {}).get("folds", []):
        prediction_trace = fold.get("prediction_trace")
        if not prediction_trace:
            continue
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(prediction_trace.get("timestamps", []), errors="coerce"),
                "actual": prediction_trace.get("actual", []),
                "predicted": prediction_trace.get("predicted", []),
                "lower": prediction_trace.get("lower", []),
                "upper": prediction_trace.get("upper", []),
            }
        )
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=frame["timestamp"], y=frame["actual"], name="actual", line=dict(color="#1f2937")))
        figure.add_trace(go.Scatter(x=frame["timestamp"], y=frame["predicted"], name="predicted", line=dict(color="#2563eb")))
        figure.add_trace(go.Scatter(x=frame["timestamp"], y=frame["upper"], name="upper", line=dict(color="#93c5fd"), opacity=0.6))
        figure.add_trace(go.Scatter(x=frame["timestamp"], y=frame["lower"], name="lower", line=dict(color="#93c5fd"), fill="tonexty", opacity=0.2))
        figure.update_layout(
            title=f"Fold {fold.get('fold')} Actual vs Predicted",
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_title=None,
            yaxis_title="glucose",
        )
        traces.append(_figure_div(figure, include_plotlyjs=include_js))
        include_js = False
    return traces


def _fold_metric_figure(summary: dict[str, Any], metric: str, *, title: str, baseline_label: str | None = None) -> go.Figure:
    folds = (summary.get("walk_forward") or {}).get("folds", [])
    frame = pd.DataFrame(
        {
            "fold": [fold.get("fold") for fold in folds],
            metric: [fold.get(metric) for fold in folds],
            "persistence_mae": [fold.get("persistence_mae") for fold in folds],
        }
    )
    figure = go.Figure()
    figure.add_trace(go.Bar(x=frame["fold"], y=frame[metric], name=metric))
    if baseline_label is not None:
        figure.add_trace(go.Bar(x=frame["fold"], y=frame["persistence_mae"], name=baseline_label))
    figure.update_layout(title=title, barmode="group", margin=dict(l=20, r=20, t=50, b=20), xaxis_title="fold")
    return figure


def _sampler_table(summary: dict[str, Any]) -> str:
    folds = (summary.get("walk_forward") or {}).get("folds", [])
    rows = [
        "<tr><th>fold</th><th>chains</th><th>divergences</th><th>rhat_max</th><th>ess_bulk_min</th><th>ess_tail_min</th></tr>"
    ]
    for fold in folds:
        diag = fold.get("fit_diagnostics") or {}
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(fold.get('fold')))}</td>"
            f"<td>{html.escape(str(diag.get('chains')))}</td>"
            f"<td>{html.escape(str(diag.get('divergences')))}</td>"
            f"<td>{html.escape(str(diag.get('rhat_max')))}</td>"
            f"<td>{html.escape(str(diag.get('ess_bulk_min')))}</td>"
            f"<td>{html.escape(str(diag.get('ess_tail_min')))}</td>"
            "</tr>"
        )
    return "\n".join(
        [
            "<div class='section'>",
            "<h2>Sampler Diagnostics</h2>",
            "<table>",
            "<thead>",
            rows[0],
            "</thead>",
            "<tbody>",
            *rows[1:],
            "</tbody>",
            "</table>",
            "</div>",
        ]
    )


def _final_fit_card(summary: dict[str, Any]) -> str:
    diagnostics = summary.get("fit_diagnostics") or {}
    if not diagnostics:
        return (
            "<div class='section'><h2>Final Fit Diagnostics</h2>"
            "<div class='empty-state'>No final recommendation fit was run.</div></div>"
        )
    fields = [
        ("chains", diagnostics.get("chains")),
        ("draws", diagnostics.get("draws")),
        ("tune", diagnostics.get("tune")),
        ("divergences", diagnostics.get("divergences")),
        ("max_tree_depth_hits", diagnostics.get("max_tree_depth_hits")),
        ("rhat_max", diagnostics.get("rhat_max")),
        ("ess_bulk_min", diagnostics.get("ess_bulk_min")),
        ("ess_tail_min", diagnostics.get("ess_tail_min")),
    ]
    items = [f"<div><strong>{html.escape(name)}</strong>: {html.escape(str(value))}</div>" for name, value in fields]
    return "\n".join(
        [
            "<div class='section'>",
            "<h2>Final Fit Diagnostics</h2>",
            "<div class='card-grid'>",
            "<div class='card'>",
            *items,
            "</div>",
            "</div>",
            "</div>",
        ]
    )


def _forecast_summary_html(summary: dict[str, Any]) -> str:
    walk_forward = summary.get("walk_forward") or {}
    aggregate = walk_forward.get("aggregate") or {}
    policy = summary.get("recommendation_policy") or {}
    data_quality = summary.get("data_quality") or {}
    items = [
        ("data_quality_status", data_quality.get("status") or "unknown"),
        ("incomplete_windows", data_quality.get("incomplete_window_count") if data_quality else "NA"),
        ("model_mae", "NA" if aggregate.get("mae") is None else f"{float(aggregate['mae']):.3f}"),
        ("persistence_mae", "NA" if walk_forward.get("aggregate_persistence_mae") is None else f"{float(walk_forward['aggregate_persistence_mae']):.3f}"),
        ("coverage", "NA" if aggregate.get("coverage") is None else f"{float(aggregate['coverage']):.3f}"),
        ("recommendation_policy", policy.get("status") or "unknown"),
    ]
    return "\n".join(
        [
            "<div class='section'>",
            "<h2>Decision Summary</h2>",
            "<div class='card-grid'>",
            "<div class='card'>",
            *[f"<div><strong>{html.escape(name)}</strong>: {html.escape(str(value))}</div>" for name, value in items],
            "</div>",
            "<div class='card'>",
            "<strong>Interpretation</strong>",
            "<div class='muted'>This page is the forecast-validation drill-down. The key questions are whether validation beats persistence, whether source quality is acceptable, and whether model diagnostics are trustworthy enough to support downstream decisions.</div>",
            "</div>",
            "</div>",
            "</div>",
        ]
    )


def write_run_review_html(summary: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    include_js = True
    figures: list[str] = []
    coverage_figure = _fold_metric_figure(summary, "model_coverage", title="Coverage by Fold")
    figures.append(_figure_div(coverage_figure, include_plotlyjs=include_js))
    include_js = False
    mae_figure = _fold_metric_figure(summary, "model_mae", title="Model vs Persistence MAE", baseline_label="persistence_mae")
    figures.append(_figure_div(mae_figure, include_plotlyjs=include_js))
    trace_figures = _run_trace_figures(summary)

    recommendation_policy = summary.get("recommendation_policy") or {}
    recommendations = summary.get("recommendations") or []
    reasons = recommendation_policy.get("reasons") or []
    reason_text = (
        "<div class='muted'>Reasons: " + html.escape(", ".join(str(reason) for reason in reasons)) + "</div>"
        if reasons
        else ""
    )
    if recommendations:
        recommendation_panel = (
            "<div class='card'><strong>Recommendations generated.</strong>"
            + reason_text
            + "</div>"
        )
    elif recommendation_policy.get("status") == "suppressed":
        recommendation_panel = (
            "<div class='empty-state'><strong>Recommendations were suppressed by policy.</strong>"
            + reason_text
            + "</div>"
        )
    elif recommendation_policy.get("status") == "skipped":
        recommendation_panel = (
            "<div class='empty-state'><strong>Recommendations were skipped by configuration.</strong>"
            + reason_text
            + "</div>"
        )
    else:
        recommendation_panel = "<div class='empty-state'><strong>No recommendations were generated.</strong></div>"

    sections = [
        _forecast_summary_html(summary),
        "<div class='section'><h2>Coverage by Fold</h2>" + figures[0] + "</div>",
        "<div class='section'><h2>Model vs Persistence</h2>" + figures[1] + "</div>",
        *[f"<div class='section'><h2>Forecast Trace</h2>{trace}</div>" for trace in trace_figures],
        _sampler_table(summary),
        _final_fit_card(summary),
        "<div class='section'><h2>Recommendation Policy</h2>" + recommendation_panel + "</div>",
    ]
    html_text = _page_shell(
        title="Bayesian T1DM Run Review",
        banner_html=_data_quality_banner(summary, recommendation_policy=recommendation_policy),
        sections=sections,
    )
    path.write_text(html_text, encoding="utf-8")
    return path


def _therapy_banner(preparation: ModelDataPreparationResult, overnight_summary: dict[str, Any]) -> str:
    status = {
        "identifiable": "good",
        "weakly_identifiable": "degraded",
        "blocked": "broken",
    }.get(str(overnight_summary.get("status")), "degraded")
    return (
        f"<div class='banner {html.escape(status)}'>"
        f"Prepared mode: <code>{html.escape(preparation.dataset.mode)}</code>. "
        f"Overnight basal status: <code>{html.escape(str(overnight_summary.get('status') or 'unknown'))}</code>. "
        f"Clean overnight rows: <code>{html.escape(str(overnight_summary.get('clean_rows', 0)))}</code> across "
        f"<code>{html.escape(str(overnight_summary.get('clean_nights', 0)))}</code> nights."
        "</div>"
    )


def _therapy_timeline_figure(preparation: ModelDataPreparationResult) -> go.Figure:
    rows: list[dict[str, Any]] = []
    for label, start, end in [
        ("tandem_before", preparation.tandem_span_before_start, preparation.tandem_span_before_end),
        ("tandem_after", preparation.tandem_span_after_start, preparation.tandem_span_after_end),
        ("apple", preparation.apple_span_start, preparation.apple_span_end),
        ("final_dataset", preparation.final_dataset_start, preparation.final_dataset_end),
    ]:
        if start is None or end is None:
            continue
        rows.append(
            {
                "series": label,
                "start": pd.Timestamp(start),
                "end": pd.Timestamp(end),
                "label": label.replace("_", " "),
            }
        )
    if not rows:
        return go.Figure()
    frame = pd.DataFrame(rows)
    fig = px.timeline(frame, x_start="start", x_end="end", y="label", color="series", title="Therapy Data Timeline")
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), legend_title_text="")
    return fig


def _overnight_trace_figure(research_result: TherapyResearchResult) -> go.Figure:
    frame = research_result.research_frame.loc[
        research_result.research_frame["therapy_segment"].astype(str).eq("overnight")
    ].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).tail(288)
    if frame.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["timestamp"], y=frame["glucose"], name="glucose", line=dict(color="#1f2937")))
    if "target_glucose" in frame.columns:
        fig.add_trace(go.Scatter(x=frame["timestamp"], y=frame["target_glucose"], name="target_glucose", line=dict(color="#2563eb", dash="dot")))
    if "basal_units_per_hour" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame["timestamp"],
                y=frame["basal_units_per_hour"],
                name="basal_units_per_hour",
                line=dict(color="#dc2626"),
                yaxis="y2",
            )
        )
    fig.update_layout(
        title="Recent Overnight Glucose and Basal",
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title=None,
        yaxis_title="glucose",
        yaxis2=dict(title="basal u/hr", overlaying="y", side="right"),
    )
    return fig


def _exclusion_figure(exclusions: pd.DataFrame) -> go.Figure:
    if exclusions.empty:
        return go.Figure()
    fig = px.bar(exclusions, x="reason", y="count", title="Overnight Exclusion Reasons")
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), xaxis_title=None)
    return fig


def _workflow_table_html() -> str:
    rows = [
        ("normalize-raw", "normalize_raw_summary.md", "src/bayesian_t1dm/acquisition.py::normalize_tconnectsync_archive", "Rebuild normalized Tandem windows from archived raw payloads."),
        ("prepare-model-data", "output/prepare/model_data_preparation.md + cache/prepared/prepared_model_data_5min.csv", "src/bayesian_t1dm/cli.py::_prepare_model_data and src/bayesian_t1dm/health_auto_export.py::build_prepared_model_dataset", "Create the Tandem-aligned model dataset and record source overlap."),
        ("research-therapy-settings", "therapy_research_gate.md + therapy_feature_audit.md", "src/bayesian_t1dm/therapy_research.py::run_therapy_research", "Build therapy contexts, gate identifiability, and compare models."),
        ("validate-therapy-infra", "therapy_infra_validation.md", "src/bayesian_t1dm/therapy_research.py::validate_therapy_infra", "Check truth-recovery and suppression behavior on synthetic scenarios."),
        ("review-therapy-evidence", "output/therapy/therapy_review.html", "src/bayesian_t1dm/review.py::write_therapy_evidence_review_html", "Explain what the system thinks is happening and why."),
    ]
    html_rows = [
        "<tr><th>step</th><th>artifact</th><th>code path</th><th>purpose</th></tr>",
        *[
            f"<tr><td><code>{html.escape(step)}</code></td><td>{html.escape(artifact)}</td><td><code>{html.escape(code)}</code></td><td>{html.escape(purpose)}</td></tr>"
            for step, artifact, code, purpose in rows
        ],
    ]
    return "\n".join(
        [
            "<div class='section'>",
            "<h2>Workflow Crosswalk</h2>",
            "<table>",
            "<thead>",
            html_rows[0],
            "</thead>",
            "<tbody>",
            *html_rows[1:],
            "</tbody></table>",
            "</div>",
        ]
    )


def _parameter_gate_html(research_result: TherapyResearchResult) -> str:
    gate = research_result.research_gate.copy()
    if gate.empty:
        return "<div class='section'><h2>Parameter Gate</h2><div class='empty-state'>No research gate rows were available.</div></div>"
    rows = [
        "<tr><th>parameter</th><th>identifiability</th><th>gate_status</th><th>source_quality_status</th><th>confounding</th></tr>"
    ]
    for row in gate.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.parameter))}</td>"
            f"<td><code>{html.escape(str(row.identifiability))}</code></td>"
            f"<td><code>{html.escape(str(row.gate_status))}</code></td>"
            f"<td><code>{html.escape(str(row.source_quality_status))}</code></td>"
            f"<td><code>{html.escape(str(row.closed_loop_confounding_risk))}</code></td>"
            "</tr>"
        )
    return "\n".join(
        [
            "<div class='section'>",
            "<h2>Parameter Gate</h2>",
            "<table>",
            "<thead>",
            rows[0],
            "</thead>",
            "<tbody>",
            *rows[1:],
            "</tbody></table>",
            "</div>",
        ]
    )


def _overnight_summary_html(overnight_summary: dict[str, Any]) -> str:
    items = [
        ("status", overnight_summary.get("status")),
        ("reason", overnight_summary.get("reason") or "none"),
        ("overnight_rows", overnight_summary.get("overnight_rows")),
        ("clean_rows", overnight_summary.get("clean_rows")),
        ("overnight_nights", overnight_summary.get("overnight_nights")),
        ("clean_nights", overnight_summary.get("clean_nights")),
        ("stable_epochs", overnight_summary.get("stable_epochs")),
        ("mean_glucose", "NA" if pd.isna(overnight_summary.get("mean_glucose")) else f"{float(overnight_summary.get('mean_glucose')):.2f}"),
        ("mean_target_delta", "NA" if pd.isna(overnight_summary.get("mean_target_delta")) else f"{float(overnight_summary.get('mean_target_delta')):.2f}"),
        ("expected_direction", overnight_summary.get("expected_direction") or "none"),
    ]
    return "\n".join(
        [
            "<div class='section'>",
            "<h2>Overnight Basal Proof</h2>",
            "<div class='card-grid'>",
            "<div class='card'>",
            *[f"<div><strong>{html.escape(name)}</strong>: {html.escape(str(value))}</div>" for name, value in items],
            "</div>",
            "<div class='card'>",
            "<strong>Interpretation</strong>",
            "<div class='muted'>This section answers the first proof question: do we have enough clean overnight / fasting rows, across enough nights and stable epochs, to treat overnight basal as identifiable rather than just modeled?</div>",
            "</div>",
            "</div>",
            "</div>",
        ]
    )


def _therapy_decision_summary_html(research_result: TherapyResearchResult, overnight_summary: dict[str, Any]) -> str:
    candidates = research_result.recommendations.loc[
        research_result.recommendations["status"].astype(str).eq("candidate")
    ].copy()
    suppressed = research_result.recommendations.loc[
        research_result.recommendations["status"].astype(str).isin(["suppressed", "exploratory"])
    ].copy()
    candidate_text = "No actionable therapy candidates were produced."
    if not candidates.empty:
        lead = candidates.iloc[0]
        candidate_text = (
            f"{lead['parameter']} / {lead['segment']}: "
            f"{lead['expected_direction']} {float(lead['proposed_change_percent']):.1f}% "
            f"({lead['confidence']} confidence)"
        )
    blocked_reason = overnight_summary.get("reason") or "none"
    blocked_rows = [
        ("overnight_status", overnight_summary.get("status") or "unknown"),
        ("blocked_reason", blocked_reason),
        ("candidate_count", int(len(candidates))),
        ("suppressed_count", int(len(suppressed))),
        ("lead_candidate", candidate_text),
    ]
    return "\n".join(
        [
            "<div class='section'>",
            "<h2>Decision Summary</h2>",
            "<div class='card-grid'>",
            "<div class='card'>",
            *[f"<div><strong>{html.escape(name)}</strong>: {html.escape(str(value))}</div>" for name, value in blocked_rows],
            "</div>",
            "<div class='card'>",
            "<strong>Interpretation</strong>",
            "<div class='muted'>This page should answer whether the current data support a therapy-setting change, support leaving settings alone, or are still blocked by identifiability or confounding.</div>",
            "</div>",
            "</div>",
            "</div>",
        ]
    )


def _artifact_index_html(
    artifact_root: Path,
    *,
    include_validation: bool,
    artifact_href_prefix: str = "",
) -> str:
    artifacts = [
        "model_data_preparation.md",
        "therapy_research_gate.md",
        "meal_proxy_audit.md",
        "therapy_feature_audit.md",
        "therapy_feature_registry.csv",
        "therapy_model_comparison.md",
        "therapy_segment_evidence.csv",
        "therapy_recommendation_research.md",
        "tandem_source_report_card.md",
        "apple_source_report_card.md",
        "source_numeric_summary.csv",
        "source_missingness_summary.csv",
    ]
    if include_validation:
        artifacts.extend(
            [
                "therapy_infra_validation.md",
                "therapy_synthetic_results.csv",
                "therapy_synthetic_recommendation_audit.md",
            ]
        )
    rows = []
    for artifact in artifacts:
        path = artifact_root / artifact
        link_target = f"{artifact_href_prefix}{artifact}"
        link_html = f"<a href=\"{html.escape(link_target)}\">open</a>" if path.exists() else ""
        rows.append(
            "<tr>"
            f"<td>{html.escape(artifact)}</td>"
            f"<td>{'yes' if path.exists() else 'no'}</td>"
            f"<td>{link_html}</td>"
            "</tr>"
        )
    return "\n".join(
        [
            "<div class='section'>",
            "<h2>Supporting Artifacts</h2>",
            "<table>",
            "<thead><tr><th>artifact</th><th>exists</th><th>link</th></tr></thead>",
            "<tbody>",
            *rows,
            "</tbody></table>",
            "</div>",
        ]
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
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact_root_path = Path(artifact_root) if artifact_root is not None else path.parent
    overnight_summary, exclusion_counts = summarize_overnight_basal_evidence(research_result)
    include_js = True
    figures = []
    for figure in [
        _therapy_timeline_figure(preparation),
        _overnight_trace_figure(research_result),
        _exclusion_figure(exclusion_counts),
    ]:
        figures.append(_figure_div(figure, include_plotlyjs=include_js))
        include_js = False
    validation_card = ""
    if validation_result is not None and not validation_result.scenario_results.empty:
        passed = int(validation_result.scenario_results["passed"].sum())
        total = int(len(validation_result.scenario_results))
        validation_card = (
            "<div class='section'><h2>Synthetic Validation</h2>"
            f"<div class='card'><strong>{passed}/{total}</strong> validation scenarios passed. "
            "This tests whether the therapy workflow recovers known truth and suppresses itself when it should."
            "</div></div>"
        )
    sections = [
        _therapy_decision_summary_html(research_result, overnight_summary),
        _parameter_gate_html(research_result),
        _overnight_summary_html(overnight_summary),
        validation_card,
        "<div class='section'><h2>Data Timeline</h2>" + figures[0] + "</div>",
        "<div class='section'><h2>Recent Overnight Evidence</h2>" + figures[1] + "</div>",
        "<div class='section'><h2>Overnight Exclusion Summary</h2>" + figures[2] + "</div>",
        _artifact_index_html(
            artifact_root_path,
            include_validation=validation_result is not None,
            artifact_href_prefix=artifact_href_prefix,
        ),
        _workflow_table_html(),
    ]
    html_text = _page_shell(
        title="Bayesian T1DM Therapy Evidence Review",
        banner_html=_therapy_banner(preparation, overnight_summary),
        sections=[section for section in sections if section],
    )
    path.write_text(html_text, encoding="utf-8")
    return path


def _latent_meal_banner(result: LatentMealResearchResult) -> str:
    if result.research_gate.empty:
        status = "broken"
        headline = "Latent meal foundation status unavailable."
    else:
        if str(getattr(result, "research_scope", "foundation")) == "foundation":
            gate_row = result.research_gate.iloc[0]
            accepted_windows = int(gate_row.get("accepted_windows") or 0)
            evaluated_candidates = int(gate_row.get("evaluated_candidates") or 0)
            explicit_available = bool(gate_row.get("explicit_carb_source_available"))
            status = "good" if accepted_windows > 0 else "degraded" if evaluated_candidates > 0 else "broken"
            headline = (
                f"Foundation scope: <code>{html.escape(str(gate_row.get('research_scope') or 'foundation'))}</code>. "
                f"Accepted first-meal windows: <code>{accepted_windows}</code> / <code>{evaluated_candidates}</code>. "
                f"Explicit carb source available: <code>{str(explicit_available).lower()}</code>."
            )
        else:
            explicit_available = bool(getattr(result.prepared_dataset, "explicit_carb_source_available", False))
            reportable_ic = int(result.posterior_meals["ic_posterior_mean"].notna().sum()) if not result.posterior_meals.empty else 0
            accepted_windows = int(pd.to_numeric(result.meal_windows.get("included"), errors="coerce").fillna(0.0).sum()) if not result.meal_windows.empty else 0
            selected_model_rows = result.model_comparison.loc[result.model_comparison["selected"].astype(bool)] if not result.model_comparison.empty else pd.DataFrame()
            selected_model = str(selected_model_rows.iloc[0]["model"]) if not selected_model_rows.empty else "unknown"
            status = "good" if reportable_ic > 0 else "degraded" if accepted_windows > 0 else "broken"
            headline = (
                f"Full latent-meal scope on <code>{accepted_windows}</code> accepted first-meal windows. "
                f"Selected model: <code>{html.escape(selected_model)}</code>. "
                f"Reportable morning I/C windows: <code>{reportable_ic}</code>. "
                f"Explicit carb source available: <code>{str(explicit_available).lower()}</code>."
            )
    return (
        f"<div class='banner {html.escape(status)}'>"
        f"{headline}"
        "</div>"
    )


def _latent_meal_gate_html(result: LatentMealResearchResult) -> str:
    if result.research_gate.empty:
        return "<div class='section'><h2>Research Gate</h2><div class='empty-state'>No latent meal gate was available.</div></div>"
    if str(getattr(result, "research_scope", "foundation")) == "foundation":
        row = result.research_gate.iloc[0]
        return "\n".join(
            [
                "<div class='section'>",
                "<h2>Research Gate</h2>",
                "<table>",
                "<thead><tr><th>research_scope</th><th>explicit_carb_source_available</th><th>latent_fit_status</th><th>evaluated_candidates</th><th>accepted_windows</th><th>source_quality</th><th>cohort_status</th></tr></thead>",
                "<tbody>",
                "<tr>"
                f"<td><code>{html.escape(str(row.research_scope))}</code></td>"
                f"<td><code>{html.escape(str(bool(row.explicit_carb_source_available)).lower())}</code></td>"
                f"<td><code>{html.escape(str(row.latent_fit_status))}</code></td>"
                f"<td>{int(row.evaluated_candidates)}</td>"
                f"<td>{int(row.accepted_windows)}</td>"
                f"<td><code>{html.escape(str(row.source_quality_status))}</code></td>"
                f"<td><code>{html.escape(str(row.cohort_status))}</code></td>"
                "</tr>",
                "</tbody></table>",
                "</div>",
            ]
        )
    rows = []
    for row in result.research_gate.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.parameter))}</td>"
            f"<td>{html.escape(str(row.estimand))}</td>"
            f"<td><code>{html.escape(str(row.identifiability))}</code></td>"
            f"<td><code>{html.escape(str(row.gate_status))}</code></td>"
            f"<td><code>{html.escape(str(row.source_quality_status))}</code></td>"
            f"<td><code>{html.escape(str(row.closed_loop_confounding_risk))}</code></td>"
            "</tr>"
        )
    return "\n".join(
        [
            "<div class='section'>",
            "<h2>Research Gate</h2>",
            "<table>",
            "<thead><tr><th>parameter</th><th>estimand</th><th>identifiability</th><th>gate_status</th><th>source_quality</th><th>confounding</th></tr></thead>",
            "<tbody>",
            *rows,
            "</tbody></table>",
            "</div>",
        ]
    )


def _latent_meal_semantics_html(result: LatentMealResearchResult) -> str:
    frame = result.research_frame.copy()
    if frame.empty:
        return "<div class='section'><h2>Meal Truth Semantics</h2><div class='empty-state'>No research rows were available.</div></div>"
    status_counts = frame.get("meal_truth_status", pd.Series(dtype=str)).astype(str).value_counts(dropna=False)
    rows = []
    for status, count in status_counts.items():
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(status))}</code></td>"
            f"<td>{int(count)}</td>"
            "</tr>"
        )
    explicit_available = bool(getattr(result.prepared_dataset, "explicit_carb_source_available", False))
    return "\n".join(
        [
            "<div class='section'>",
            "<h2>Meal Truth Semantics</h2>",
            f"<div class='muted'>Explicit carb source available: <code>{str(explicit_available).lower()}</code>. Legacy meal columns remain for compatibility and are not treated as observed truth when the explicit source is unavailable.</div>",
            "<table>",
            "<thead><tr><th>meal_truth_status</th><th>count</th></tr></thead>",
            "<tbody>",
            *rows,
            "</tbody></table>",
            "</div>",
        ]
    )


def _first_meal_clean_window_html(result: LatentMealResearchResult) -> str:
    if result.meal_windows.empty:
        return "<div class='section'><h2>First Meal Clean Windows</h2><div class='empty-state'>No morning first-meal candidates were evaluated.</div></div>"
    top = result.meal_windows.head(20)
    rows = []
    for row in top.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.candidate_id))}</td>"
            f"<td>{html.escape(str(row.timestamp))}</td>"
            f"<td>{html.escape(str(row.segment))}</td>"
            f"<td>{'yes' if bool(row.included) else 'no'}</td>"
            f"<td>{'NA' if pd.isna(row.premeal_glucose) else f'{float(row.premeal_glucose):.1f}'}</td>"
            f"<td>{'NA' if pd.isna(row.premeal_iob) else f'{float(row.premeal_iob):.2f}'}</td>"
            f"<td>{float(row.cgm_coverage_fraction):.3f}</td>"
            f"<td>{html.escape(str(row.exclusion_reasons or 'none'))}</td>"
            "</tr>"
        )
    return "\n".join(
        [
            "<div class='section'>",
            "<h2>First Meal Clean Windows</h2>",
            "<table>",
            "<thead><tr><th>candidate_id</th><th>timestamp</th><th>segment</th><th>included</th><th>premeal_glucose</th><th>premeal_iob</th><th>cgm_coverage</th><th>exclusions</th></tr></thead>",
            "<tbody>",
            *rows,
            "</tbody></table>",
            "</div>",
        ]
    )


def _first_meal_exclusion_html(result: LatentMealResearchResult) -> str:
    if result.first_meal_exclusion_summary.empty:
        return "<div class='section'><h2>Exclusion Summary</h2><div class='empty-state'>No exclusion summary was available.</div></div>"
    rows = []
    for row in result.first_meal_exclusion_summary.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.reason))}</td>"
            f"<td>{int(row.count)}</td>"
            "</tr>"
        )
    return "\n".join(
        [
            "<div class='section'>",
            "<h2>Exclusion Summary</h2>",
            "<table>",
            "<thead><tr><th>reason</th><th>count</th></tr></thead>",
            "<tbody>",
            *rows,
            "</tbody></table>",
            "</div>",
        ]
    )


def _latent_meal_model_comparison_html(result: LatentMealResearchResult) -> str:
    if result.model_comparison.empty:
        return "<div class='section'><h2>Model Comparison</h2><div class='empty-state'>No model comparison rows were available.</div></div>"
    rows = []
    for row in result.model_comparison.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.model))}</td>"
            f"<td>{int(row.meal_count)}</td>"
            f"<td>{'NA' if pd.isna(row.stated_carb_mae) else f'{float(row.stated_carb_mae):.3f}'}</td>"
            f"<td>{'NA' if pd.isna(row.peak_delta_mae) else f'{float(row.peak_delta_mae):.3f}'}</td>"
            f"<td>{'NA' if pd.isna(row.peak_delta_correlation) else f'{float(row.peak_delta_correlation):.3f}'}</td>"
            f"<td>{'yes' if bool(row.selected) else 'no'}</td>"
            "</tr>"
        )
    return "\n".join(
        [
            "<div class='section'>",
            "<h2>Model Comparison</h2>",
            "<table>",
            "<thead><tr><th>model</th><th>meal_count</th><th>stated_carb_mae</th><th>peak_delta_mae</th><th>peak_delta_correlation</th><th>selected</th></tr></thead>",
            "<tbody>",
            *rows,
            "</tbody></table>",
            "</div>",
        ]
    )


def _latent_meal_posterior_html(result: LatentMealResearchResult) -> str:
    if result.posterior_meals.empty:
        return "<div class='section'><h2>Posterior Meals</h2><div class='empty-state'>No posterior meal estimates were generated.</div></div>"
    top = result.posterior_meals.copy()
    top["abs_gap"] = (
        pd.to_numeric(top["latent_carbs_posterior_mean"], errors="coerce")
        - pd.to_numeric(top["stated_carbs"], errors="coerce")
    ).abs()
    top = top.sort_values(["carb_accuracy_score", "timestamp"], ascending=[False, True]).head(15)
    rows = []
    for row in top.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.meal_id))}</td>"
            f"<td>{html.escape(str(row.timestamp))}</td>"
            f"<td>{html.escape(str(row.segment))}</td>"
            f"<td>{html.escape(str(row.evidence_source))}</td>"
            f"<td>{'NA' if pd.isna(row.stated_carbs) else f'{float(row.stated_carbs):.1f}'}</td>"
            f"<td>{float(row.latent_carbs_posterior_mean):.1f}</td>"
            f"<td>{float(row.carb_accuracy_score):.3f}</td>"
            f"<td>{html.escape(str(row.identifiability_status))}</td>"
            f"<td>{html.escape(str(row.suppression_reason or 'none'))}</td>"
            "</tr>"
        )
    return "\n".join(
        [
            "<div class='section'>",
            "<h2>Posterior Meals</h2>",
            "<table>",
            "<thead><tr><th>meal_id</th><th>timestamp</th><th>segment</th><th>evidence</th><th>stated_carbs</th><th>latent_carbs</th><th>accuracy_score</th><th>identifiability</th><th>suppression</th></tr></thead>",
            "<tbody>",
            *rows,
            "</tbody></table>",
            "</div>",
        ]
    )


def _latent_meal_artifact_index_html(
    artifact_root: Path,
    *,
    research_scope: str = "foundation",
    artifact_href_prefix: str = "",
) -> str:
    if research_scope == "foundation":
        artifacts = [
            "latent_meal_research_gate.md",
            "meal_truth_semantics_report.md",
            "meal_event_registry.csv",
            "first_meal_clean_window_registry.csv",
            "first_meal_clean_window_audit.md",
            "first_meal_exclusion_summary.csv",
        ]
    else:
        artifacts = [
            "latent_meal_research_gate.md",
            "meal_event_registry.csv",
            "meal_window_audit.md",
            "latent_meal_fit_summary.md",
            "latent_meal_posterior_meals.csv",
            "latent_meal_confidence_report.md",
            "latent_meal_model_comparison.md",
        ]
    rows = []
    for artifact in artifacts:
        path = artifact_root / artifact
        link_target = f"{artifact_href_prefix}{artifact}"
        link_html = f"<a href=\"{html.escape(link_target)}\">open</a>" if path.exists() else ""
        rows.append(
            "<tr>"
            f"<td>{html.escape(artifact)}</td>"
            f"<td>{'yes' if path.exists() else 'no'}</td>"
            f"<td>{link_html}</td>"
            "</tr>"
        )
    return "\n".join(
        [
            "<div class='section'>",
            "<h2>Supporting Artifacts</h2>",
            "<table>",
            "<thead><tr><th>artifact</th><th>exists</th><th>link</th></tr></thead>",
            "<tbody>",
            *rows,
            "</tbody></table>",
            "</div>",
        ]
    )


def write_latent_meal_review_html(
    result: LatentMealResearchResult,
    path: str | Path,
    *,
    artifact_root: str | Path | None = None,
    artifact_href_prefix: str = "",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact_root_path = Path(artifact_root) if artifact_root is not None else path.parent
    if str(getattr(result, "research_scope", "foundation")) == "foundation":
        sections = [
            _latent_meal_gate_html(result),
            _latent_meal_semantics_html(result),
            _first_meal_clean_window_html(result),
            _first_meal_exclusion_html(result),
            _latent_meal_artifact_index_html(
                artifact_root_path,
                research_scope="foundation",
                artifact_href_prefix=artifact_href_prefix,
            ),
        ]
    else:
        sections = [
            _latent_meal_gate_html(result),
            _latent_meal_model_comparison_html(result),
            _latent_meal_posterior_html(result),
            _latent_meal_artifact_index_html(artifact_root_path, research_scope="full", artifact_href_prefix=artifact_href_prefix),
        ]
    html_text = _page_shell(
        title="Bayesian T1DM Latent Meal Review",
        banner_html=_latent_meal_banner(result),
        sections=sections,
    )
    path.write_text(html_text, encoding="utf-8")
    return path


def _current_status_banner(payload: dict[str, Any]) -> str:
    overall_state = str(payload.get("overall_state") or "blocked")
    banner_class = {
        "recommendation_ready": "good",
        "no_change_supported": "degraded",
        "blocked": "broken",
    }.get(overall_state, "degraded")
    return (
        f"<div class='banner {html.escape(banner_class)}'>"
        f"{html.escape(str(payload.get('headline') or 'Status unavailable'))} "
        f"Run id: <code>{html.escape(str(payload.get('run_id') or 'unknown'))}</code>."
        "</div>"
    )


def _current_status_summary_html(payload: dict[str, Any], *, therapy_href: str, forecast_href: str) -> str:
    data_prep = payload.get("data_prep") or {}
    blockers = payload.get("primary_blockers") or []
    blocker_lines = "".join(
        f"<li><strong>{html.escape(str(item.get('label') or item.get('code')))}</strong>: {html.escape(str(item.get('detail') or ''))}</li>"
        for item in blockers
    ) or "<li>No active blockers.</li>"
    next_actions = payload.get("next_actions") or []
    next_action_lines = "".join(f"<li>{html.escape(str(item))}</li>" for item in next_actions) or "<li>No next actions recorded.</li>"
    therapy = payload.get("therapy") or {}
    overnight = therapy.get("overnight") or {}
    forecast = payload.get("forecast") or {}
    return "\n".join(
        [
            "<div class='section'>",
            "<h2>Current Verdict</h2>",
            "<div class='card-grid'>",
            "<div class='card'>",
            f"<div><strong>overall_state</strong>: <code>{html.escape(str(payload.get('overall_state') or 'unknown'))}</code></div>",
            f"<div><strong>summary</strong>: {html.escape(str(payload.get('summary') or ''))}</div>",
            f"<div><strong>apple_available</strong>: {html.escape(str(data_prep.get('apple_available')))}</div>",
            f"<div><strong>overlap_window</strong>: {html.escape(str(data_prep.get('overlap_start')))} to {html.escape(str(data_prep.get('overlap_end')))}</div>",
            f"<div><strong>final_dataset_rows</strong>: {html.escape(str(data_prep.get('final_row_count')))}</div>",
            "</div>",
            "<div class='card'>",
            "<strong>Evidence Links</strong>",
            f"<div><a href=\"{html.escape(therapy_href)}\">Therapy evidence dashboard</a></div>",
            f"<div><a href=\"{html.escape(forecast_href)}\">Forecast validation dashboard</a></div>",
            "</div>",
            "</div>",
            "</div>",
            "<div class='section'>",
            "<h2>Primary Blockers</h2>",
            "<div class='card'><ul>",
            blocker_lines,
            "</ul></div>",
            "</div>",
            "<div class='section'>",
            "<h2>Next Actions</h2>",
            "<div class='card'><ol>",
            next_action_lines,
            "</ol></div>",
            "</div>",
            "<div class='section'>",
            "<h2>Therapy Summary</h2>",
            "<div class='card-grid'>",
            "<div class='card'>",
            f"<div><strong>overnight_status</strong>: {html.escape(str(overnight.get('status') or 'unknown'))}</div>",
            f"<div><strong>blocked_reason</strong>: {html.escape(str(overnight.get('reason') or 'none'))}</div>",
            f"<div><strong>clean_rows</strong>: {html.escape(str(overnight.get('clean_rows') or 0))}</div>",
            f"<div><strong>clean_nights</strong>: {html.escape(str(overnight.get('clean_nights') or 0))}</div>",
            "</div>",
            "<div class='card'>",
            f"<div><strong>forecast_quality</strong>: {html.escape(str(forecast.get('data_quality_status') or 'unknown'))}</div>",
            f"<div><strong>model_mae</strong>: {html.escape(str(forecast.get('model_mae') or 'NA'))}</div>",
            f"<div><strong>persistence_mae</strong>: {html.escape(str(forecast.get('persistence_mae') or 'NA'))}</div>",
            f"<div><strong>sampler_health</strong>: {html.escape(str(forecast.get('sampler_health') or 'unknown'))}</div>",
            "</div>",
            "</div>",
            "</div>",
        ]
    )


def write_current_status_html(
    payload: dict[str, Any],
    path: str | Path,
    *,
    therapy_href: str,
    forecast_href: str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html_text = _page_shell(
        title="Bayesian T1DM Current Status",
        banner_html=_current_status_banner(payload),
        sections=[_current_status_summary_html(payload, therapy_href=therapy_href, forecast_href=forecast_href)],
    )
    path.write_text(html_text, encoding="utf-8")
    return path
