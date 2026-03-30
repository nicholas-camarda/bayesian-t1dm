from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bayesian_t1dm.ingest import (
    build_export_manifest,
    load_tandem_exports,
    summarize_coverage,
    summarize_export_manifest,
    summarize_tandem_raw_dir,
    summarize_tandem_raw_source,
)


def test_load_tandem_exports_normalizes_fixture_data(tandem_fixture_dir):
    data = load_tandem_exports(tandem_fixture_dir)
    manifest = build_export_manifest(data)
    manifest_summary = summarize_export_manifest(manifest)

    assert len(data.source_files) == 4
    assert {"timestamp", "glucose", "source_file"}.issubset(data.cgm.columns)
    assert {"timestamp", "bolus_units", "source_file"}.issubset(data.bolus.columns)
    assert {"timestamp", "carb_grams", "source_file"}.issubset(data.carbs.columns)
    assert {"start_timestamp", "end_timestamp", "basal_units_per_hour"}.issubset(data.basal.columns)
    assert {"timestamp", "activity_value", "source_file"}.issubset(data.activity.columns)
    assert {"kind", "first_timestamp", "last_timestamp", "is_complete_window"}.issubset(manifest.columns)
    assert manifest_summary["is_complete"]

    coverage = summarize_coverage(data)
    assert coverage.manifest_rows == len(manifest)
    assert coverage.cgm_rows == 7
    assert coverage.bolus_rows == 2
    assert len(data.carbs) == 2
    assert coverage.basal_rows == 1
    assert coverage.activity_rows == 2
    assert str(coverage.first_timestamp) == "2023-06-01 00:00:00"
    assert str(coverage.last_timestamp) == "2023-06-02 00:00:00"
    assert coverage.is_complete


def test_load_tandem_exports_prefers_dense_egv_over_sparse_bg(tmp_path):
    workbook = tmp_path / "therapy_events_2023-02.xlsx"
    frame = pd.DataFrame(
        {
            "eventDateTime": pd.date_range("2023-02-04 09:00:00", periods=10, freq="5min"),
            "egv_estimatedGlucoseValue": [100 + index for index in range(10)],
            "bg": [210.0, pd.NA, pd.NA, pd.NA, pd.NA, 185.0, pd.NA, pd.NA, pd.NA, pd.NA],
        }
    )
    with pd.ExcelWriter(workbook) as writer:
        frame.to_excel(writer, index=False)

    data = load_tandem_exports(tmp_path)

    assert len(data.source_files) == 1
    assert data.cgm.shape[0] == 10
    assert {"egv_estimatedGlucoseValue", "bg", "glucose", "glucose_source"}.issubset(data.cgm.columns)
    assert data.cgm["glucose"].tolist() == [100 + index for index in range(10)]
    assert data.cgm["egv_estimatedGlucoseValue"].notna().all()
    assert int(data.cgm["bg"].notna().sum()) == 2
    assert data.cgm.loc[data.cgm["bg"].notna(), "glucose"].tolist() == [100, 105]


def test_summarize_tandem_raw_source_ignores_summary_only_value_table(tmp_path):
    summary_path = tmp_path / "tandem_daily_timeline.csv"
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-02-01", periods=7, freq="D"),
            "value": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    frame.to_csv(summary_path, index=False)

    report = summarize_tandem_raw_source(summary_path)

    assert report["cgm_rows"] == 0
    assert not report["has_dense_cgm_stream"]
    assert report["recognized_glucose_columns"] == ()


def test_summarize_tandem_raw_source_reports_dense_egv_stream():
    report = summarize_tandem_raw_source(Path("data/cgm_and_bolus/therapy_events_2023-02.xlsx"))

    assert report["cgm_rows"] == 1488
    assert report["has_dense_cgm_stream"]
    assert report["median_spacing_minutes"] == pytest.approx(5.0)
    assert "egv_estimatedGlucoseValue" in report["recognized_glucose_columns"]


def test_load_tandem_exports_raw_subtree_recovers_dense_cgm():
    raw_dir = Path("data/cgm_and_bolus")
    data = load_tandem_exports(raw_dir)
    summary = summarize_tandem_raw_dir(raw_dir)

    therapy = summary.loc[summary["source_file"] == "therapy_events_2023-02.xlsx"].iloc[0]

    assert data.cgm.shape[0] > 1000
    assert therapy["has_dense_cgm_stream"]
    assert therapy["cgm_rows"] == 1488
    assert therapy["median_spacing_minutes"] == pytest.approx(5.0)


def test_export_manifest_detects_gap_duplicate_and_out_of_order(tandem_fixture_dir):
    data = load_tandem_exports(tandem_fixture_dir)
    manifest = build_export_manifest(data).copy()
    cgm_rows = manifest[manifest["kind"] == "cgm"].index.tolist()
    assert cgm_rows
    first = cgm_rows[0]
    shifted = manifest.loc[first].copy()
    shifted["first_timestamp"] = manifest.loc[first, "last_timestamp"] + pd.Timedelta(days=2)
    shifted["last_timestamp"] = shifted["first_timestamp"] + pd.Timedelta(minutes=30)
    shifted["source_order"] = 0
    manifest.loc[first, "source_order"] = 1
    manifest.loc[len(manifest)] = shifted
    manifest.loc[len(manifest)] = manifest.loc[first]
    manifest.loc[len(manifest) - 1, "source_order"] = -1
    manifest.loc[first, "is_complete_window"] = False

    summary = summarize_export_manifest(manifest)

    assert summary["gap_count"] >= 1
    assert summary["duplicate_windows"] >= 1
    assert summary["out_of_order_windows"] >= 1
    assert summary["incomplete_windows"] >= 1
    assert not summary["is_complete"]
