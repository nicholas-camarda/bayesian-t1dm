from __future__ import annotations

import json
from dataclasses import asdict
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .evaluate import WalkForwardReport
from .ingest import TandemCoverage
from .model import FitDiagnostics
from .quality import DataQualitySummary
from .recommend import Recommendation, RecommendationPolicy


def build_run_summary(
    *,
    coverage: TandemCoverage,
    walk_forward: WalkForwardReport | None = None,
    recommendations: list[Recommendation] | None = None,
    fit_diagnostics: FitDiagnostics | None = None,
    data_quality: DataQualitySummary | None = None,
    recommendation_policy: RecommendationPolicy | None = None,
    review_artifacts: dict[str, str] | None = None,
    model_name: str = "BayesianGlucoseModel",
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "coverage": asdict(coverage),
        "walk_forward": asdict(walk_forward) if walk_forward is not None else None,
        "fit_diagnostics": asdict(fit_diagnostics) if fit_diagnostics is not None else None,
        "data_quality": asdict(data_quality) if data_quality is not None else None,
        "recommendation_policy": asdict(recommendation_policy) if recommendation_policy is not None else None,
        "review_artifacts": dict(review_artifacts or {}),
        "recommendations": [asdict(rec) for rec in (recommendations or [])],
    }


def _relative_link_target(markdown_path: Path, artifact_path: str | Path) -> str:
    try:
        return os.path.relpath(Path(artifact_path), start=markdown_path.parent)
    except Exception:
        return str(artifact_path)


def write_markdown_report(summary: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Bayesian T1DM Run Summary")
    lines.append("")
    lines.append("## Coverage")
    coverage = summary.get("coverage", {})
    for key, value in coverage.items():
        lines.append(f"- {key}: {value}")
    if summary.get("data_quality") is not None:
        lines.append("")
        lines.append("## Data Quality")
        data_quality = summary["data_quality"]
        lines.append(f"- status: {data_quality.get('status')}")
        lines.append(f"- incomplete_window_count: {data_quality.get('incomplete_window_count')}")
        lines.append(
            f"- evaluation_touches_incomplete_windows: {data_quality.get('evaluation_touches_incomplete_windows')}"
        )
        contributing = data_quality.get("contributing_window_ids") or []
        if contributing:
            lines.append(f"- contributing_window_ids: {', '.join(str(item) for item in contributing)}")
        reason_counts = data_quality.get("reason_counts") or {}
        if reason_counts:
            lines.append(
                "- reason_counts: "
                + ", ".join(f"{key}={value}" for key, value in reason_counts.items())
            )
    if summary.get("walk_forward"):
        lines.append("")
        lines.append("## Walk-Forward Evaluation")
        walk_forward = summary["walk_forward"]
        aggregate = walk_forward.get("aggregate") or {}
        for key in ["mae", "rmse", "coverage", "interval_width"]:
            if key in aggregate:
                lines.append(f"- aggregate_{key}: {aggregate[key]}")
        if "aggregate_persistence_mae" in walk_forward:
            lines.append(f"- aggregate_persistence_mae: {walk_forward['aggregate_persistence_mae']}")
        folds = walk_forward.get("folds") or []
        if folds:
            lines.append("")
            lines.append("| fold | n_train | n_test | model_mae | model_rmse | model_coverage | persistence_mae |")
            lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for fold in folds:
                lines.append(
                    "| {fold} | {n_train} | {n_test} | {model_mae:.4f} | {model_rmse:.4f} | {model_coverage:.3f} | {persistence_mae:.4f} |".format(
                        fold=int(fold.get("fold", 0)),
                        n_train=int(fold.get("n_train", 0)),
                        n_test=int(fold.get("n_test", 0)),
                        model_mae=float(fold.get("model_mae", float("nan"))),
                        model_rmse=float(fold.get("model_rmse", float("nan"))),
                        model_coverage=float(fold.get("model_coverage", float("nan"))),
                        persistence_mae=float(fold.get("persistence_mae", float("nan"))),
                    )
                )
            diagnostics_rows = [fold for fold in folds if fold.get("fit_diagnostics")]
            if diagnostics_rows:
                lines.append("")
                lines.append("### Walk-Forward Fit Diagnostics")
                lines.append("")
                lines.append("| fold | chains | draws | tune | divergences | max_treedepth_hits | rhat_max | ess_bulk_min | ess_tail_min |")
                lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
                for fold in diagnostics_rows:
                    diag = fold["fit_diagnostics"]
                    lines.append(
                        "| {fold} | {chains} | {draws} | {tune} | {divergences} | {max_treedepth_hits} | {rhat_max} | {ess_bulk_min} | {ess_tail_min} |".format(
                            fold=int(fold.get("fold", 0)),
                            chains=int(diag.get("chains", 0)),
                            draws=int(diag.get("draws", 0)),
                            tune=int(diag.get("tune", 0)),
                            divergences=int(diag.get("divergences", 0)),
                            max_treedepth_hits=int(diag.get("max_tree_depth_hits", 0)),
                            rhat_max="NA" if diag.get("rhat_max") is None else f"{float(diag['rhat_max']):.3f}",
                            ess_bulk_min="NA" if diag.get("ess_bulk_min") is None else f"{float(diag['ess_bulk_min']):.1f}",
                            ess_tail_min="NA" if diag.get("ess_tail_min") is None else f"{float(diag['ess_tail_min']):.1f}",
                        )
                    )
    if summary.get("fit_diagnostics") is not None:
        lines.append("")
        lines.append("## Final Fit Diagnostics")
        for key, value in summary["fit_diagnostics"].items():
            lines.append(f"- {key}: {value}")
    if summary.get("recommendation_policy") is not None:
        lines.append("")
        lines.append("## Recommendation Policy")
        policy = summary["recommendation_policy"]
        lines.append(f"- status: {policy.get('status')}")
        lines.append(f"- validation_passed: {policy.get('validation_passed')}")
        lines.append(f"- sampler_passed: {policy.get('sampler_passed')}")
        lines.append(f"- signal_passed: {policy.get('signal_passed')}")
        reasons = policy.get("reasons") or []
        if reasons:
            lines.append(f"- reasons: {', '.join(str(reason) for reason in reasons)}")
    review_artifacts = summary.get("review_artifacts") or {}
    if review_artifacts:
        lines.append("")
        lines.append("## Review Artifacts")
        for name, artifact_path in sorted(review_artifacts.items()):
            target = _relative_link_target(path, artifact_path)
            lines.append(f"- [{name}]({target})")
    lines.append("")
    lines.append("## Recommendations")
    recommendations = summary.get("recommendations", [])
    if not recommendations:
        policy_status = (summary.get("recommendation_policy") or {}).get("status")
        if policy_status == "skipped":
            lines.append("- Recommendations were skipped by configuration.")
        elif policy_status == "suppressed":
            lines.append("- Recommendations were suppressed by policy.")
        else:
            lines.append("- No recommendations were generated.")
    for rec in recommendations:
        flags = rec.get("flags") or []
        flag_suffix = f" [flags: {', '.join(flags)}]" if flags else ""
        lines.append(
            f"- {rec['setting']} {rec['direction']} by {rec['change_percent']:.1f}% "
            f"({rec.get('confidence', 'moderate')} confidence): {rec['rationale']}{flag_suffix}"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp,)):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return str(value)


def write_json_report(summary: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_jsonable(summary)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
