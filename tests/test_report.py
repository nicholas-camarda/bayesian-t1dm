from __future__ import annotations

import json

import pandas as pd

from bayesian_t1dm.evaluate import CalibrationSummary, FoldResult, WalkForwardReport
from bayesian_t1dm.health_auto_export import ModelDataPreparationResult
from bayesian_t1dm.ingest import TandemCoverage
from bayesian_t1dm.model import FitDiagnostics
from bayesian_t1dm.quality import DataQualitySummary
from bayesian_t1dm.recommend import Recommendation, RecommendationPolicy
from bayesian_t1dm.report import build_run_summary, write_json_report, write_markdown_report
from bayesian_t1dm.review import write_coverage_review_html, write_run_review_html, write_therapy_evidence_review_html
from bayesian_t1dm.therapy_research import _synthetic_base_dataset, parse_model_list, parse_therapy_segments, run_therapy_research, validate_therapy_infra


def test_write_json_report_serializes_timestamps(tmp_path):
    coverage = TandemCoverage(
        source_files=1,
        manifest_rows=1,
        cgm_rows=10,
        bolus_rows=2,
        basal_rows=0,
        activity_rows=0,
        health_activity_rows=0,
        health_measurement_rows=0,
        sleep_rows=0,
        workout_rows=0,
        first_timestamp=pd.Timestamp("2024-01-01 00:00:00"),
        last_timestamp=pd.Timestamp("2024-01-01 01:00:00"),
        complete_windows=1,
        incomplete_windows=0,
        gap_count=0,
        overlap_count=0,
        duplicate_windows=0,
        out_of_order_windows=0,
        is_complete=True,
    )
    walk_forward = WalkForwardReport(
        folds=[
            FoldResult(
                fold=1,
                n_train=100,
                n_test=50,
                model_mae=10.0,
                model_rmse=12.0,
                model_coverage=0.8,
                persistence_mae=14.0,
                fit_diagnostics=FitDiagnostics(
                    draws=100,
                    tune=100,
                    chains=2,
                    target_accept=0.9,
                    max_treedepth=10,
                    wall_time_seconds=1.2,
                    divergences=0,
                    max_tree_depth_observed=7,
                    max_tree_depth_hits=0,
                    rhat_max=1.0,
                    ess_bulk_min=250.0,
                    ess_tail_min=260.0,
                ),
            )
        ],
        aggregate=CalibrationSummary(mae=10.0, rmse=12.0, coverage=0.8, interval_width=40.0),
        aggregate_persistence_mae=14.0,
    )
    recommendations = [
        Recommendation(
            setting="basal",
            direction="decrease",
            change_percent=10.0,
            expected_gain_mgdl=6.0,
            posterior_probability_better=0.75,
            rationale="test",
            scenario_name="basal_minus_10",
            confidence="moderate",
            flags=["low_signal"],
        )
    ]
    fit_diagnostics = FitDiagnostics(
        draws=1000,
        tune=1000,
        chains=2,
        target_accept=0.9,
        max_treedepth=10,
        wall_time_seconds=4.2,
        divergences=0,
        max_tree_depth_observed=8,
        max_tree_depth_hits=0,
        rhat_max=1.0,
        ess_bulk_min=300.0,
        ess_tail_min=280.0,
    )
    recommendation_policy = RecommendationPolicy(
        status="generated",
        reasons=["low_signal"],
        validation_passed=True,
        sampler_passed=True,
        signal_passed=True,
    )

    summary = build_run_summary(
        coverage=coverage,
        walk_forward=walk_forward,
        recommendations=recommendations,
        fit_diagnostics=fit_diagnostics,
        data_quality=DataQualitySummary(
            status="good",
            contributing_window_ids=["window-a"],
            incomplete_window_count=0,
            reason_counts={},
            evaluation_touches_incomplete_windows=False,
        ),
        recommendation_policy=recommendation_policy,
        review_artifacts={"run_review_html": str(tmp_path / "run_review.html")},
    )
    out_path = write_json_report(summary, tmp_path / "run_summary.json")

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(payload["coverage"]["first_timestamp"], str)
    assert payload["walk_forward"]["folds"][0]["fold"] == 1
    assert payload["fit_diagnostics"]["chains"] == 2
    assert payload["data_quality"]["status"] == "good"
    assert payload["recommendation_policy"]["status"] == "generated"
    assert payload["review_artifacts"]["run_review_html"].endswith("run_review.html")


def test_write_markdown_report_distinguishes_suppressed_recommendations(tmp_path):
    coverage = TandemCoverage(
        source_files=1,
        manifest_rows=1,
        cgm_rows=10,
        bolus_rows=0,
        basal_rows=0,
        activity_rows=0,
        health_activity_rows=0,
        health_measurement_rows=0,
        sleep_rows=0,
        workout_rows=0,
        first_timestamp=pd.Timestamp("2024-01-01 00:00:00"),
        last_timestamp=pd.Timestamp("2024-01-01 01:00:00"),
        complete_windows=1,
        incomplete_windows=0,
        gap_count=0,
        overlap_count=0,
        duplicate_windows=0,
        out_of_order_windows=0,
        is_complete=True,
    )
    summary = build_run_summary(
        coverage=coverage,
        recommendation_policy=RecommendationPolicy(
            status="suppressed",
            reasons=["coverage_out_of_range"],
            validation_passed=False,
            sampler_passed=True,
            signal_passed=True,
        ),
        recommendations=[],
    )

    out_path = write_markdown_report(summary, tmp_path / "run_summary.md")
    text = out_path.read_text(encoding="utf-8")

    assert "## Recommendation Policy" in text
    assert "status: suppressed" in text
    assert "Recommendations were suppressed by policy." in text


def test_review_html_writers_emit_expected_sections(tmp_path):
    coverage = TandemCoverage(
        source_files=1,
        manifest_rows=3,
        cgm_rows=100,
        bolus_rows=10,
        basal_rows=288,
        activity_rows=0,
        health_activity_rows=0,
        health_measurement_rows=0,
        sleep_rows=0,
        workout_rows=0,
        first_timestamp=pd.Timestamp("2024-01-01 00:00:00"),
        last_timestamp=pd.Timestamp("2024-01-02 00:00:00"),
        complete_windows=0,
        incomplete_windows=1,
        gap_count=0,
        overlap_count=0,
        duplicate_windows=0,
        out_of_order_windows=0,
        is_complete=False,
    )
    walk_forward = WalkForwardReport(
        folds=[
            FoldResult(
                fold=1,
                n_train=400,
                n_test=250,
                model_mae=10.0,
                model_rmse=12.0,
                model_coverage=0.8,
                persistence_mae=14.0,
                fit_diagnostics=FitDiagnostics(
                    draws=100,
                    tune=100,
                    chains=2,
                    target_accept=0.95,
                    max_treedepth=12,
                    wall_time_seconds=1.0,
                    divergences=0,
                    max_tree_depth_observed=8,
                    max_tree_depth_hits=0,
                    rhat_max=1.0,
                    ess_bulk_min=250.0,
                    ess_tail_min=260.0,
                ),
                prediction_trace={
                    "timestamps": ["2024-01-02T00:00:00", "2024-01-02T00:05:00"],
                    "actual": [110.0, 111.0],
                    "predicted": [109.0, 112.0],
                    "lower": [100.0, 101.0],
                    "upper": [118.0, 119.0],
                    "interval_hit": [True, True],
                },
            )
        ],
        aggregate=CalibrationSummary(mae=10.0, rmse=12.0, coverage=0.8, interval_width=20.0),
        aggregate_persistence_mae=14.0,
    )
    data_quality = DataQualitySummary(
        status="degraded",
        contributing_window_ids=["2024-01-01__2024-01-02"],
        incomplete_window_count=1,
        reason_counts={"ends_early": 1},
        evaluation_touches_incomplete_windows=True,
    )
    summary = build_run_summary(
        coverage=coverage,
        walk_forward=walk_forward,
        fit_diagnostics=FitDiagnostics(
            draws=1000,
            tune=1000,
            chains=2,
            target_accept=0.95,
            max_treedepth=12,
            wall_time_seconds=4.0,
            divergences=0,
            max_tree_depth_observed=8,
            max_tree_depth_hits=0,
            rhat_max=1.0,
            ess_bulk_min=300.0,
            ess_tail_min=280.0,
        ),
        data_quality=data_quality,
        recommendation_policy=RecommendationPolicy(
            status="suppressed",
            reasons=["data_incomplete"],
            validation_passed=False,
            sampler_passed=True,
            signal_passed=True,
        ),
    )
    quality_rows = pd.DataFrame(
        [
            {
                "window_id": "2024-01-01__2024-01-02",
                "kind": "cgm",
                "row_count": 100,
                "requested_start": "2024-01-01",
                "requested_end": "2024-01-02",
                "observed_first_timestamp": "2024-01-01T00:00:00-05:00",
                "observed_last_timestamp": "2024-01-01T18:00:00-05:00",
                "coverage_fraction": 0.75,
                "completeness_reasons": '["ends_early"]',
            }
        ]
    )

    coverage_path = write_coverage_review_html(summary, quality_rows, tmp_path / "coverage_review.html")
    run_path = write_run_review_html(summary, tmp_path / "run_review.html")

    coverage_text = coverage_path.read_text(encoding="utf-8")
    run_text = run_path.read_text(encoding="utf-8")
    assert "Window Timeline" in coverage_text
    assert "Completeness Reasons" in coverage_text
    assert "Latest Window" in coverage_text
    assert "Recommendation Policy" in run_text
    assert "Recommendations were suppressed by policy." in run_text
    assert "Final Fit Diagnostics" in run_text


def test_write_therapy_evidence_review_html_emits_expected_sections(tmp_path):
    dataset = _synthetic_base_dataset(apple=True, explicit_carbs=True)
    preparation = ModelDataPreparationResult(
        dataset=dataset,
        apple_available=True,
        apple_span_start=pd.Timestamp("2025-01-01 00:00:00"),
        apple_span_end=pd.Timestamp("2025-01-07 23:55:00"),
        tandem_span_before_start=pd.Timestamp("2025-01-01 00:00:00"),
        tandem_span_before_end=pd.Timestamp("2025-01-07 23:55:00"),
        tandem_span_after_start=pd.Timestamp("2025-01-01 00:00:00"),
        tandem_span_after_end=pd.Timestamp("2025-01-07 23:55:00"),
        requested_tandem_start=pd.Timestamp("2025-01-01 00:00:00"),
        requested_tandem_end=pd.Timestamp("2025-01-07 23:55:00"),
        overlap_start=pd.Timestamp("2025-01-01 00:00:00"),
        overlap_end=pd.Timestamp("2025-01-07 23:55:00"),
        final_dataset_start=pd.Timestamp("2025-01-01 00:00:00"),
        final_dataset_end=pd.Timestamp("2025-01-07 23:55:00"),
        final_row_count=len(dataset.frame),
    )
    research_result = run_therapy_research(
        dataset,
        segments=parse_therapy_segments(),
        include_models=parse_model_list("ridge,segmented_ridge,tree_boost"),
    )
    validation_result = validate_therapy_infra(include_models=parse_model_list("ridge,segmented_ridge,tree_boost"))

    out_path = write_therapy_evidence_review_html(preparation, research_result, tmp_path / "therapy_evidence_review.html", validation_result=validation_result)
    text = out_path.read_text(encoding="utf-8")

    assert "Therapy Data Timeline" in text
    assert "Workflow Crosswalk" in text
    assert "Overnight Basal Proof" in text
    assert "Supporting Artifacts" in text
