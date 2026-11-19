from __future__ import annotations

import json

import pandas as pd

from bayesian_t1dm.health_auto_export import ModelDataPreparationResult
from bayesian_t1dm.ingest import TandemCoverage
from bayesian_t1dm.quality import DataQualitySummary
from bayesian_t1dm.report import build_coverage_summary, write_coverage_markdown
from bayesian_t1dm.review import write_coverage_review_html, write_therapy_evidence_review_html
from bayesian_t1dm.therapy_research import _synthetic_base_dataset, parse_model_list, parse_therapy_segments, run_therapy_research, validate_therapy_infra


def test_write_coverage_markdown_lists_data_quality_and_review_links(tmp_path):
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
    summary = build_coverage_summary(
        coverage=coverage,
        data_quality=DataQualitySummary(
            status="degraded",
            contributing_window_ids=["window-a"],
            incomplete_window_count=1,
            reason_counts={"ends_early": 1},
            evaluation_touches_incomplete_windows=True,
        ),
        review_artifacts={"coverage_review_html": str(tmp_path / "coverage_review.html")},
    )

    text = write_coverage_markdown(summary, tmp_path / "coverage.md").read_text(encoding="utf-8")

    assert "### Data Quality" in text
    assert "- status: degraded" in text
    assert "reason_counts: ends_early=1" in text
    assert "[coverage_review_html](coverage_review.html)" in text


def test_coverage_review_html_emits_expected_sections(tmp_path, assert_answer_first_page):
    coverage = TandemCoverage(
        source_files=1,
        manifest_rows=3,
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
        complete_windows=0,
        incomplete_windows=1,
        gap_count=0,
        overlap_count=0,
        duplicate_windows=0,
        out_of_order_windows=0,
        is_complete=False,
    )
    summary = build_coverage_summary(
        coverage=coverage,
        data_quality=DataQualitySummary(
            status="degraded",
            contributing_window_ids=["2024-01-01__2024-01-02"],
            incomplete_window_count=1,
            reason_counts={"ends_early": 1},
            evaluation_touches_incomplete_windows=True,
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

    text = write_coverage_review_html(summary, quality_rows, tmp_path / "coverage_review.html").read_text(encoding="utf-8")

    assert "Is the Tandem data complete?" in text
    assert "1 of 1 download windows" in text
    assert "data ends earlier than requested" in text
    assert_answer_first_page(text)


def test_write_therapy_evidence_review_html_emits_expected_sections(tmp_path, assert_answer_first_page):
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

    assert "Is there enough data to judge my overnight basal?" in text
    assert "clean nights" in text
    assert "Files behind this page" in text
    assert "behaved correctly in" in text
    assert_answer_first_page(text)



def test_icr_verdict_reports_direction_and_setting():
    from bayesian_t1dm.report_page import icr_verdict

    consistent = icr_verdict("morning", 6.7, 6.1, 7.3, 7.0, "consistent_with_programmed")
    assert consistent.status == "good"
    assert "6.7 g per unit (90% range 6.1 to 7.3)" in consistent.headline and "set to 7.0" in consistent.headline
    assert "harder" in consistent.explanation
    harder = icr_verdict("morning", 5.5, 5.0, 6.0, 7.0, "carbs_hit_harder_than_programmed")
    assert harder.status == "warn" and harder.label == "Carbs raise glucose more than the setting assumes"


def test_window_completeness_depends_on_download_coverage_not_cgm_gaps():
    from bayesian_t1dm.quality import build_window_quality_row

    full = build_window_quality_row(
        window_id="w", kind="cgm", requested_start="2026-09-20", requested_end="2026-10-19",
        row_count=8000, observed_first_timestamp="2026-09-20T00:00:00", observed_last_timestamp="2026-10-19T23:55:00",
        has_internal_gap=True, has_duplicates=True, has_overlap=True,
    )
    assert full["is_complete_window"] and json.loads(full["completeness_reasons"]) == []
    short = build_window_quality_row(
        window_id="w", kind="cgm", requested_start="2026-09-20", requested_end="2026-10-19",
        row_count=4000, observed_first_timestamp="2026-09-20T00:00:00", observed_last_timestamp="2026-10-04T23:55:00",
    )
    assert json.loads(short["completeness_reasons"]) == ["ends_early"]
