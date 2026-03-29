from __future__ import annotations

import pandas as pd

from bayesian_t1dm.ingest import build_export_manifest, load_tandem_exports, summarize_coverage, summarize_export_manifest


def test_load_tandem_exports_normalizes_fixture_data(tandem_fixture_dir):
    data = load_tandem_exports(tandem_fixture_dir)
    manifest = build_export_manifest(data)
    manifest_summary = summarize_export_manifest(manifest)

    assert len(data.source_files) == 4
    assert {"timestamp", "glucose", "source_file"}.issubset(data.cgm.columns)
    assert {"timestamp", "bolus_units", "source_file"}.issubset(data.bolus.columns)
    assert {"start_timestamp", "end_timestamp", "basal_units_per_hour"}.issubset(data.basal.columns)
    assert {"timestamp", "activity_value", "source_file"}.issubset(data.activity.columns)
    assert {"kind", "first_timestamp", "last_timestamp", "is_complete_window"}.issubset(manifest.columns)
    assert manifest_summary["is_complete"]

    coverage = summarize_coverage(data)
    assert coverage.manifest_rows == len(manifest)
    assert coverage.cgm_rows == 7
    assert coverage.bolus_rows == 2
    assert coverage.basal_rows == 1
    assert coverage.activity_rows == 2
    assert str(coverage.first_timestamp) == "2023-06-01 00:00:00"
    assert str(coverage.last_timestamp) == "2023-06-02 00:00:00"
    assert coverage.is_complete


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
