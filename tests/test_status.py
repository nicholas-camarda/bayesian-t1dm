from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bayesian_t1dm.features import FeatureConfig
from bayesian_t1dm.health_auto_export import AnalysisReadyHealthDataset, ModelDataPreparationResult
from bayesian_t1dm.paths import ProjectPaths
from bayesian_t1dm.status import cleanup_legacy_top_level_output, derive_current_status, finalize_status_logs, publish_html_entrypoint, reset_output_directory, write_status_json
from bayesian_t1dm.therapy_research import TherapyInfraValidationResult, TherapyResearchResult


def _dataset(*, apple: bool = True) -> AnalysisReadyHealthDataset:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01 00:00:00", periods=96, freq="5min"),
            "glucose": [110.0] * 96,
            "target_delta": [0.0] * 96,
            "heart_rate_avg_latest": [70.0] * 96,
        }
    )
    if not apple:
        frame = frame.drop(columns=["heart_rate_avg_latest"], errors="ignore")
    return AnalysisReadyHealthDataset(
        frame=frame,
        feature_columns=["glucose"],
        tandem_feature_columns=["glucose"],
        health_feature_columns=["heart_rate_avg_latest"] if apple else [],
        target_column="target_delta",
        horizon_minutes=30,
        config=FeatureConfig(horizon_minutes=30),
        mode="apple_enriched" if apple else "tandem_only",
        apple_available=apple,
    )


def _preparation(dataset: AnalysisReadyHealthDataset) -> ModelDataPreparationResult:
    start = pd.Timestamp("2025-01-01 00:00:00")
    end = pd.Timestamp("2025-01-05 00:00:00")
    return ModelDataPreparationResult(
        dataset=dataset,
        apple_available=dataset.apple_available,
        apple_span_start=start,
        apple_span_end=end,
        tandem_span_before_start=start,
        tandem_span_before_end=end,
        tandem_span_after_start=start,
        tandem_span_after_end=end,
        requested_tandem_start=start,
        requested_tandem_end=end,
        overlap_start=start,
        overlap_end=end,
        final_dataset_start=start,
        final_dataset_end=end,
        final_row_count=int(len(dataset.frame)),
        backfill_status="not_needed",
    )


def _research_result(
    *,
    overnight_clean_rows: int,
    overnight_nights: int,
    stable_epochs: int,
    confounding: str,
    direct_meal_rows: int,
    proxy_meal_rows: int,
    recommendation_status: str,
) -> TherapyResearchResult:
    rows = []
    for index in range(max(overnight_clean_rows, 1)):
        rows.append(
            {
                "timestamp": pd.Timestamp("2025-01-01 00:00:00") + pd.Timedelta(minutes=5 * index),
                "therapy_segment": "overnight",
                "basal_context": 1 if index < overnight_clean_rows else 0,
                "therapy_stable_epoch": index % max(stable_epochs, 1),
                "glucose": 110.0,
                "target_delta": 0.0,
                "recent_meal_120m": 0,
                "recent_bolus_120m": 0,
                "recent_exercise_context": 0,
                "closed_loop_confounding_flag": 1 if confounding == "high" else 0,
                "missing_cgm": 0,
            }
        )
    research_frame = pd.DataFrame(rows)
    if not research_frame.empty and overnight_nights > 1:
        day_offsets = [index % overnight_nights for index in range(len(research_frame))]
        research_frame["timestamp"] = [pd.Timestamp("2025-01-01 00:00:00") + pd.Timedelta(days=offset, minutes=5 * idx) for idx, offset in enumerate(day_offsets)]
    gate = pd.DataFrame(
        [
            {
                "parameter": "basal",
                "identifiability": "directly_observed" if overnight_clean_rows >= 96 else "not_identified",
                "gate_status": "research_enabled" if overnight_clean_rows >= 96 else "diagnostics_only",
                "source_quality_status": "good",
                "direct_meal_rows": direct_meal_rows,
                "proxy_meal_rows": proxy_meal_rows,
                "basal_context_rows": overnight_clean_rows,
                "correction_context_rows": 96,
                "closed_loop_confounding_risk": confounding,
                "apple_alignment_status": "credible",
            },
            {
                "parameter": "I/C ratio",
                "identifiability": "directly_observed" if direct_meal_rows >= 48 else "not_identified",
                "gate_status": "research_enabled" if direct_meal_rows >= 48 else "diagnostics_only",
                "source_quality_status": "good",
                "direct_meal_rows": direct_meal_rows,
                "proxy_meal_rows": proxy_meal_rows,
                "basal_context_rows": overnight_clean_rows,
                "correction_context_rows": 96,
                "closed_loop_confounding_risk": confounding,
                "apple_alignment_status": "credible",
            },
        ]
    )
    recommendations = pd.DataFrame(
        [
            {
                "parameter": "basal",
                "segment": "overnight",
                "status": recommendation_status,
                "proposed_change_percent": 5.0 if recommendation_status == "candidate" else None,
                "expected_direction": "increase" if recommendation_status == "candidate" else "hold",
                "mean_expected_gain": 6.0 if recommendation_status == "candidate" else 0.5,
                "fold_better_fraction": 0.9 if recommendation_status == "candidate" else 0.55,
                "confidence": "high" if recommendation_status == "candidate" else "low",
                "reasons_for": "",
                "reasons_against": "" if recommendation_status == "candidate" else "expected_gain_too_small",
                "identifiability": "directly_observed",
            }
        ]
    )
    dataset = _dataset()
    return TherapyResearchResult(
        prepared_dataset=dataset,
        research_frame=research_frame,
        research_gate=gate,
        feature_registry=pd.DataFrame(),
        meal_proxy_audit=pd.DataFrame(),
        model_comparison=pd.DataFrame(),
        segment_evidence=pd.DataFrame(),
        recommendations=recommendations,
        research_gate_markdown="# gate\n",
        feature_audit_markdown="# audit\n",
        meal_proxy_audit_markdown="# meal\n",
        model_comparison_markdown="# model\n",
        recommendation_markdown="# recs\n",
        tandem_source_report_markdown="# tandem\n",
        apple_source_report_markdown="# apple\n",
        source_numeric_summary=pd.DataFrame(),
        source_missingness_summary=pd.DataFrame(),
        segments=tuple(),
        include_models=("ridge",),
        meal_proxy_mode="strict",
        ic_policy="exploratory_only",
    )


def _forecast_summary(*, strong: bool = True, diagnostics_good: bool = True, quality_status: str = "good") -> dict[str, object]:
    return {
        "walk_forward": {
            "aggregate": {
                "mae": 10.0 if strong else 15.0,
                "rmse": 12.0,
                "coverage": 0.8 if strong else 0.2,
                "interval_width": 20.0,
            },
            "aggregate_persistence_mae": 14.0,
            "folds": [
                {
                    "fold": 1,
                    "fit_diagnostics": {
                        "chains": 2,
                        "divergences": 0 if diagnostics_good else 12,
                        "rhat_max": 1.01 if diagnostics_good else 1.8,
                        "ess_bulk_min": 200.0 if diagnostics_good else 5.0,
                    },
                }
            ],
        },
        "data_quality": {
            "status": quality_status,
            "incomplete_window_count": 0 if quality_status == "good" else 2,
        },
        "recommendation_policy": {
            "status": "skipped",
        },
    }


def test_derive_current_status_marks_candidate_as_recommendation_ready():
    dataset = _dataset()
    payload = derive_current_status(
        preparation=_preparation(dataset),
        research_result=_research_result(
            overnight_clean_rows=96,
            overnight_nights=4,
            stable_epochs=3,
            confounding="moderate",
            direct_meal_rows=60,
            proxy_meal_rows=0,
            recommendation_status="candidate",
        ),
        forecast_summary=_forecast_summary(strong=True, diagnostics_good=True),
        run_id="run123",
    )

    assert payload["overall_state"] == "recommendation_ready"
    assert payload["primary_blockers"] == []


def test_derive_current_status_marks_stable_case_as_no_change_supported():
    dataset = _dataset()
    payload = derive_current_status(
        preparation=_preparation(dataset),
        research_result=_research_result(
            overnight_clean_rows=96,
            overnight_nights=4,
            stable_epochs=3,
            confounding="moderate",
            direct_meal_rows=60,
            proxy_meal_rows=0,
            recommendation_status="suppressed",
        ),
        forecast_summary=_forecast_summary(strong=True, diagnostics_good=True),
        run_id="run123",
    )

    assert payload["overall_state"] == "no_change_supported"
    assert payload["primary_blockers"] == []


def test_derive_current_status_marks_overnight_or_confounding_case_as_blocked():
    dataset = _dataset()
    payload = derive_current_status(
        preparation=_preparation(dataset),
        research_result=_research_result(
            overnight_clean_rows=12,
            overnight_nights=1,
            stable_epochs=1,
            confounding="high",
            direct_meal_rows=0,
            proxy_meal_rows=10,
            recommendation_status="suppressed",
        ),
        forecast_summary=_forecast_summary(strong=False, diagnostics_good=False, quality_status="degraded"),
        run_id="run123",
    )

    codes = {item["code"] for item in payload["primary_blockers"]}
    assert payload["overall_state"] == "blocked"
    assert "insufficient_clean_overnight_windows" in codes
    assert "high_closed_loop_confounding" in codes


def test_output_cleanup_moves_top_level_clutter_into_runtime_archive(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    paths = ProjectPaths.from_root(repo_root, runtime_root=tmp_path / "runtime", cloud_root=tmp_path / "cloud").ensure()
    (paths.reports / "old.csv").write_text("old", encoding="utf-8")
    (paths.reports / "scratch").mkdir(parents=True, exist_ok=True)
    (paths.reports / "scratch" / "note.txt").write_text("note", encoding="utf-8")

    manifest_path = cleanup_legacy_top_level_output(paths)

    assert manifest_path is not None
    assert not (paths.reports / "old.csv").exists()
    moved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(moved["moved"]) == 2
    assert moved["migrated_logs"] == []
    assert manifest_path.parent == paths.legacy_output_archive

    log_dir = paths.logs / "status" / "run123"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "events.jsonl").write_text("{}\n", encoding="utf-8")
    assert finalize_status_logs(paths, log_dir) == log_dir

    source_html = paths.output_forecast / "forecast_review.html"
    source_html.write_text("<html>forecast</html>", encoding="utf-8")
    published = publish_html_entrypoint(source_html, paths.reports / "forecast_review.html")
    assert published.read_text(encoding="utf-8") == "<html>forecast</html>"


def test_output_cleanup_migrates_legacy_output_logs_into_runtime_logs(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    paths = ProjectPaths.from_root(repo_root, runtime_root=tmp_path / "runtime", cloud_root=tmp_path / "cloud").ensure()
    legacy_log_dir = paths.reports / "logs" / "status" / "run123"
    legacy_log_dir.mkdir(parents=True, exist_ok=True)
    (legacy_log_dir / "events.jsonl").write_text("{}\n", encoding="utf-8")
    (paths.reports / "logs" / "status" / "latest.json").write_text('{"run_dir":"old"}\n', encoding="utf-8")

    manifest_path = cleanup_legacy_top_level_output(paths)

    assert manifest_path is not None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(payload["migrated_logs"]) == 2
    assert not (paths.reports / "logs").exists()
    assert (paths.logs / "status" / "run123" / "events.jsonl").exists()
    assert (paths.logs / "status" / "latest.json").exists()


def test_reset_output_directory_replaces_stale_contents(tmp_path):
    directory = tmp_path / "output" / "forecast"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "stale.md").write_text("stale", encoding="utf-8")

    reset_output_directory(directory)

    assert directory.exists()
    assert list(directory.iterdir()) == []


def test_write_status_json_persists_expected_keys(tmp_path):
    path = tmp_path / "current_status.json"
    write_status_json({"overall_state": "blocked", "run_id": "abc123"}, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["overall_state"] == "blocked"
    assert payload["run_id"] == "abc123"
