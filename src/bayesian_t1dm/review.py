from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html


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
