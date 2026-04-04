from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from .health_auto_export import ModelDataPreparationResult
from .paths import ProjectPaths
from .report import _to_jsonable
from .therapy_research import TherapyInfraValidationResult, TherapyResearchResult, summarize_overnight_basal_evidence


TOP_LEVEL_ENTRYPOINTS = {
    "current_status.html",
    "therapy_review.html",
    "forecast_review.html",
    "forecast",
    "therapy",
    "latent_meal",
    "fixture",
    "source",
    "prepare",
}


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def cleanup_legacy_top_level_output(paths: ProjectPaths, *, keep_names: set[str] | None = None) -> Path | None:
    keep = set(keep_names or set()) | TOP_LEVEL_ENTRYPOINTS
    extras = [item for item in sorted(paths.reports.iterdir(), key=lambda candidate: candidate.name) if item.name not in keep]
    if not extras:
        return None
    legacy_root = paths.legacy_output_archive
    if legacy_root.exists():
        shutil.rmtree(legacy_root)
    legacy_root.mkdir(parents=True, exist_ok=True)
    moved: list[dict[str, str]] = []
    for item in extras:
        destination = legacy_root / item.name
        shutil.move(str(item), str(destination))
        moved.append({"source": str(item), "destination": str(destination)})
    manifest_path = legacy_root / "legacy_cleanup_manifest.json"
    manifest_path.write_text(json.dumps({"moved": moved, "moved_at": _utc_timestamp()}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def reset_output_directory(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sampler_health(summary: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    folds = ((summary.get("walk_forward") or {}).get("folds") or [])
    blockers: list[dict[str, str]] = []
    if not folds:
        return "not_available", blockers
    failing = False
    for fold in folds:
        diagnostics = fold.get("fit_diagnostics") or {}
        if not diagnostics:
            continue
        rhat_max = diagnostics.get("rhat_max")
        ess_bulk_min = diagnostics.get("ess_bulk_min")
        divergences = diagnostics.get("divergences")
        if (rhat_max is not None and float(rhat_max) > 1.1) or (ess_bulk_min is not None and float(ess_bulk_min) < 100.0) or int(divergences or 0) > 0:
            failing = True
    if failing:
        blockers.append(
            {
                "code": "sampler_diagnostics_failure",
                "label": "Sampler diagnostics failure",
                "detail": "Walk-forward fit diagnostics did not meet basic trust thresholds.",
            }
        )
        return "failed", blockers
    return "good", blockers


def _forecast_blockers(summary: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    blockers: list[dict[str, str]] = []
    walk_forward = summary.get("walk_forward") or {}
    aggregate = walk_forward.get("aggregate") or {}
    model_mae = aggregate.get("mae")
    persistence_mae = walk_forward.get("aggregate_persistence_mae")
    coverage = aggregate.get("coverage")
    data_quality = summary.get("data_quality") or {}
    if str(data_quality.get("status") or "broken") != "good":
        blockers.append(
            {
                "code": "degraded_source_quality",
                "label": "Degraded source quality",
                "detail": f"Forecast validation saw source quality status {data_quality.get('status')}.",
            }
        )
    if model_mae is None or persistence_mae is None or (float(model_mae) >= float(persistence_mae)) or (coverage is not None and float(coverage) < 0.65):
        blockers.append(
            {
                "code": "weak_forecast_validation",
                "label": "Weak forecast validation",
                "detail": "The validation model is not clearly beating persistence or has weak interval coverage.",
            }
        )
    sampler_status, sampler_blockers = _sampler_health(summary)
    blockers.extend(sampler_blockers)
    return blockers, {
        "data_quality_status": data_quality.get("status"),
        "incomplete_window_count": data_quality.get("incomplete_window_count"),
        "model_mae": model_mae,
        "persistence_mae": persistence_mae,
        "coverage": coverage,
        "interval_width": aggregate.get("interval_width"),
        "recommendation_policy_status": (summary.get("recommendation_policy") or {}).get("status"),
        "sampler_health": sampler_status,
    }


def _therapy_blockers(research_result: TherapyResearchResult, overnight_summary: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any], list[dict[str, Any]]]:
    blockers: list[dict[str, str]] = []
    gate = research_result.research_gate.copy()
    recommendations = research_result.recommendations.copy()
    basal_candidates = recommendations.loc[
        recommendations["parameter"].astype(str).eq("basal") & recommendations["status"].astype(str).eq("candidate")
    ].copy()
    if str(overnight_summary.get("status")) == "blocked":
        blockers.append(
            {
                "code": "insufficient_clean_overnight_windows",
                "label": "Insufficient clean overnight windows",
                "detail": "There are not enough clean overnight rows to identify basal reliably.",
            }
        )
    if not gate.empty and gate["closed_loop_confounding_risk"].astype(str).eq("high").any():
        blockers.append(
            {
                "code": "high_closed_loop_confounding",
                "label": "High closed-loop confounding",
                "detail": "The current overnight evidence is heavily confounded by closed-loop behavior or proxy meal signal.",
            }
        )
    if not gate.empty and gate["source_quality_status"].astype(str).isin({"degraded", "failed"}).any():
        blockers.append(
            {
                "code": "degraded_source_quality",
                "label": "Degraded source quality",
                "detail": "Therapy research flagged source-quality issues in the prepared dataset.",
            }
        )
    if not gate.empty:
        icr_rows = gate.loc[gate["parameter"].astype(str).eq("I/C ratio")]
        if not icr_rows.empty:
            direct_meal_rows = int(icr_rows.iloc[0].get("direct_meal_rows", 0) or 0)
            proxy_meal_rows = int(icr_rows.iloc[0].get("proxy_meal_rows", 0) or 0)
            if direct_meal_rows == 0 and proxy_meal_rows < 48:
                blockers.append(
                    {
                        "code": "insufficient_carb_or_meal_signal",
                        "label": "Insufficient carb or meal signal",
                        "detail": "Meal evidence is too weak to cleanly evaluate meal-linked settings.",
                    }
                )
    recommendation_rows = [
        {
            "parameter": str(row.parameter),
            "segment": str(row.segment),
            "status": str(row.status),
            "proposed_change_percent": None if pd.isna(row.proposed_change_percent) else float(row.proposed_change_percent),
            "expected_direction": str(row.expected_direction),
            "mean_expected_gain": None if pd.isna(row.mean_expected_gain) else float(row.mean_expected_gain),
            "fold_better_fraction": None if pd.isna(row.fold_better_fraction) else float(row.fold_better_fraction),
            "confidence": str(row.confidence),
            "reasons_against": str(row.reasons_against or ""),
            "identifiability": str(getattr(row, "identifiability", "")),
        }
        for row in recommendations.itertuples(index=False)
    ]
    therapy_summary = {
        "overnight": {
            key: _to_jsonable(value)
            for key, value in overnight_summary.items()
        },
        "gate": [
            {
                "parameter": str(row.parameter),
                "identifiability": str(row.identifiability),
                "gate_status": str(row.gate_status),
                "source_quality_status": str(row.source_quality_status),
                "closed_loop_confounding_risk": str(row.closed_loop_confounding_risk),
                "apple_alignment_status": str(row.apple_alignment_status),
            }
            for row in gate.itertuples(index=False)
        ],
        "recommendations": recommendation_rows,
        "candidate_count": int(len(basal_candidates)),
    }
    return blockers, therapy_summary, recommendation_rows


def derive_current_status(
    *,
    preparation: ModelDataPreparationResult,
    research_result: TherapyResearchResult,
    forecast_summary: dict[str, Any],
    run_id: str,
    generated_at: str | None = None,
    artifact_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or _utc_timestamp()
    overnight_summary, _ = summarize_overnight_basal_evidence(research_result)
    therapy_blockers, therapy_summary, recommendation_rows = _therapy_blockers(research_result, overnight_summary)
    forecast_blockers, forecast_details = _forecast_blockers(forecast_summary)
    primary_blockers = therapy_blockers + [blocker for blocker in forecast_blockers if blocker["code"] not in {item["code"] for item in therapy_blockers}]

    candidates = [row for row in recommendation_rows if row["status"] == "candidate"]
    if candidates and not primary_blockers:
        overall_state = "recommendation_ready"
        headline = "A therapy-setting change is recommendation-ready."
        summary = "The current therapy and forecast evidence support at least one candidate change without an active blocker."
        next_actions = [
            "Review the candidate change details in the therapy dashboard.",
            "Sanity-check the forecast validation page before acting on any recommendation.",
        ]
    elif not primary_blockers:
        overall_state = "no_change_supported"
        headline = "No change is currently justified."
        summary = "The current evidence does not support a meaningful therapy-setting change right now."
        next_actions = [
            "Continue monitoring with the current settings unless new evidence changes the signal.",
            "Re-run status after materially new data arrive or after a deliberate settings change.",
        ]
    else:
        overall_state = "blocked"
        headline = "The system is blocked from making a trustworthy therapy decision."
        summary = "Important evidence or validation gates are still failing, so the current run is diagnostics-only."
        next_actions = []
        for blocker in primary_blockers:
            if blocker["code"] == "insufficient_clean_overnight_windows":
                next_actions.append("Collect more clean overnight windows with fewer recent meals, boluses, and exercise confounders.")
            elif blocker["code"] == "high_closed_loop_confounding":
                next_actions.append("Inspect overnight closed-loop and proxy meal activity before trusting basal conclusions.")
            elif blocker["code"] == "degraded_source_quality":
                next_actions.append("Repair incomplete or degraded source windows before interpreting recommendations.")
            elif blocker["code"] == "weak_forecast_validation":
                next_actions.append("Treat the current forecast model as untrusted until it clearly beats persistence on held-out folds.")
            elif blocker["code"] == "sampler_diagnostics_failure":
                next_actions.append("Fix sampler diagnostics before using model-based recommendations.")
            elif blocker["code"] == "insufficient_carb_or_meal_signal":
                next_actions.append("Capture stronger meal evidence before evaluating meal-linked settings.")
        if not next_actions:
            next_actions.append("Review the supporting artifacts and resolve the listed blockers before acting.")

    return {
        "overall_state": overall_state,
        "headline": headline,
        "summary": summary,
        "primary_blockers": primary_blockers,
        "next_actions": next_actions,
        "therapy": therapy_summary,
        "forecast": forecast_details,
        "data_prep": {
            "mode": preparation.dataset.mode,
            "apple_available": preparation.apple_available,
            "overlap_start": _to_jsonable(preparation.overlap_start),
            "overlap_end": _to_jsonable(preparation.overlap_end),
            "final_dataset_start": _to_jsonable(preparation.final_dataset_start),
            "final_dataset_end": _to_jsonable(preparation.final_dataset_end),
            "final_row_count": int(preparation.final_row_count),
            "backfill_status": preparation.backfill_status,
            "health_feature_count": int(len(preparation.dataset.health_feature_columns)),
        },
        "artifacts": dict(artifact_paths or {}),
        "run_id": run_id,
        "generated_at": generated_at,
    }


def write_status_json(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def publish_html_entrypoint(source: str | Path, destination: str | Path) -> Path:
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def finalize_status_logs(paths: ProjectPaths, log_run_dir: Path | None) -> Path | None:
    if log_run_dir is None or not log_run_dir.exists():
        return None
    return log_run_dir
