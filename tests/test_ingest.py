from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bayesian_t1dm.ingest import (
    IngestedData,
    build_export_manifest,
    discover_source_files,
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


def _write_dense_therapy_events_workbook(tmp_path: Path) -> Path:
    workbook = tmp_path / "therapy_events_2023-02.xlsx"
    timestamps = pd.date_range("2023-02-04 00:00:00", periods=1488, freq="5min")
    bg = [pd.NA] * len(timestamps)
    # Sparse fingersticks sprinkled through the dense CGM stream.
    for idx, value in [(0, 210.0), (250, 185.0), (900, 160.0), (1200, 140.0)]:
        if 0 <= idx < len(bg):
            bg[idx] = value
    frame = pd.DataFrame(
        {
            "eventDateTime": timestamps,
            "egv_estimatedGlucoseValue": np.linspace(110, 140, num=len(timestamps)),
            "bg": bg,
        }
    )
    with pd.ExcelWriter(workbook) as writer:
        frame.to_excel(writer, index=False)
    return workbook


def test_summarize_tandem_raw_source_reports_dense_egv_stream(tmp_path):
    workbook = _write_dense_therapy_events_workbook(tmp_path)
    report = summarize_tandem_raw_source(workbook)

    assert report["cgm_rows"] == 1488
    assert report["has_dense_cgm_stream"]
    assert report["median_spacing_minutes"] == pytest.approx(5.0)
    assert "egv_estimatedGlucoseValue" in report["recognized_glucose_columns"]


def test_load_tandem_exports_raw_subtree_recovers_dense_cgm(tmp_path):
    workbook = _write_dense_therapy_events_workbook(tmp_path)
    data = load_tandem_exports(tmp_path)
    summary = summarize_tandem_raw_dir(tmp_path)

    therapy = summary.loc[summary["source_file"] == workbook.name].iloc[0]

    assert data.cgm.shape[0] > 1000
    assert therapy["has_dense_cgm_stream"]
    assert therapy["cgm_rows"] == 1488
    assert therapy["median_spacing_minutes"] == pytest.approx(5.0)


def test_parse_datetime_mixed_tz_and_naive_normalizes_to_local(monkeypatch):
    from bayesian_t1dm.ingest import to_local_wall_clock

    monkeypatch.setenv("TIMEZONE_NAME", "America/New_York")
    series = pd.Series(["2024-05-01T00:00:00-04:00", "2024-05-01 00:05:00"])
    parsed = to_local_wall_clock(series)

    assert str(parsed.dtype) == "datetime64[ns]"
    assert parsed.iloc[0].hour == 0
    assert parsed.iloc[1].hour == 0
    assert (parsed.iloc[1] - parsed.iloc[0]) == pd.Timedelta(minutes=5)


def test_parse_datetime_rejects_unreadable_timestamps():
    from bayesian_t1dm.ingest import to_local_wall_clock

    parsed = to_local_wall_clock(pd.Series(["2024-05-01 00:05:00", None, ""]))
    assert parsed.isna().tolist() == [False, True, True]
    with pytest.raises(ValueError, match="not ISO 8601"):
        to_local_wall_clock(pd.Series(["2024-05-01 00:05:00", "yesterday at noon"], name="timestamp"))


def test_load_tandem_exports_discovers_parquet(tmp_path):
    pytest.importorskip("pyarrow")
    workbook = _write_dense_therapy_events_workbook(tmp_path)
    cgm = pd.read_excel(workbook)
    parquet_path = tmp_path / "therapy_events_2023-02.parquet"
    cgm.to_parquet(parquet_path, index=False)

    data = load_tandem_exports(tmp_path)
    assert data.cgm.shape[0] >= 1488


def test_discover_source_files_excludes_archive_data_subtree(tmp_path):
    workbook = _write_dense_therapy_events_workbook(tmp_path)
    archive_dir = tmp_path / "archive data" / "apple-health-data" / "electrocardiograms"
    archive_dir.mkdir(parents=True, exist_ok=True)
    bad_csv = archive_dir / "ecg_2022-03-08.csv"
    bad_csv.write_text(
        "\n".join(
            [
                "Name,Nicholas Camarda",
                'Date of Birth,"Nov 13, 1993"',
                "Recorded Date,2022-03-08 17:16:56 -0500",
                "Classification,Sinus Rhythm",
                "Symptoms,Rapid, pounding, or fluttering heartbeat,Skipped heartbeat,Fatigue,Dizziness",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source_pool_dir = tmp_path / "apple_health_data"
    source_pool_dir.mkdir(parents=True, exist_ok=True)
    staged_csv = source_pool_dir / "staged.csv"
    staged_csv.write_text("timestamp,value\n2024-01-01,1\n", encoding="utf-8")

    discovered = discover_source_files(tmp_path)

    assert workbook in discovered
    assert bad_csv not in discovered
    assert staged_csv not in discovered


def test_load_tandem_exports_excludes_archive_data_subtree(tmp_path):
    workbook = _write_dense_therapy_events_workbook(tmp_path)
    archive_dir = tmp_path / "archive data" / "apple-health-data" / "electrocardiograms"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "ecg_2022-03-08.csv").write_text(
        "\n".join(
            [
                "Name,Nicholas Camarda",
                'Date of Birth,"Nov 13, 1993"',
                "Recorded Date,2022-03-08 17:16:56 -0500",
                "Classification,Sinus Rhythm",
                "Symptoms,Rapid, pounding, or fluttering heartbeat,Skipped heartbeat,Fatigue,Dizziness",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    data = load_tandem_exports(tmp_path)

    assert workbook.resolve() in data.source_files
    assert all("archive data" not in str(path) for path in data.source_files)
    assert data.cgm.shape[0] == 1488


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


def test_load_tandem_exports_dedupes_overlapping_normalized_windows(tmp_path):
    raw = tmp_path / "raw"
    for window_id in ["2024-06-01__2024-06-01", "2024-06-01__2024-06-02"]:
        normalized = raw / "tconnectsync" / window_id / "normalized"
        normalized.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "timestamp": ["2024-06-01T00:00:00"],
                "glucose": [100.0],
                "source_file": [f"tconnectsync::{window_id}::tconnectsync"],
                "source_label": ["tconnectsync"],
            }
        ).to_csv(normalized / "cgm.csv", index=False)
        pd.DataFrame(
            {
                "timestamp": ["2024-06-01T08:00:00"],
                "bolus_units": [2.0],
                "raw_bolus_units": [2.0],
                "source_event_id": ["bolus-a"],
                "source_file": [f"tconnectsync::{window_id}::tconnectsync"],
                "source_label": ["tconnectsync"],
            }
        ).to_csv(normalized / "bolus.csv", index=False)
        pd.DataFrame(
            {
                "start_timestamp": ["2024-06-01T00:00:00"],
                "end_timestamp": ["2024-06-02T00:00:00"],
                "basal_units_per_hour": [0.8],
                "source_file": [f"tconnectsync::{window_id}::tconnectsync"],
                "source_label": ["tconnectsync"],
            }
        ).to_csv(normalized / "basal.csv", index=False)
        pd.DataFrame(
            {
                "normalized_path": [
                    str(normalized / "cgm.csv"),
                    str(normalized / "bolus.csv"),
                    str(normalized / "basal.csv"),
                ]
            }
        ).to_csv(raw / "tconnectsync" / window_id / "window_manifest.csv", index=False)

    data = load_tandem_exports(raw)

    assert len(data.cgm) == 1
    assert len(data.bolus) == 1
    assert data.bolus["bolus_units"].sum() == pytest.approx(2.0)
    assert len(data.basal) == 1


def test_manifest_source_order_uses_tconnectsync_window_dates():
    ingested = IngestedData(
        cgm=pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-06-01", "2024-05-01"]),
                "glucose": [100.0, 101.0],
                "source_file": [
                    "tconnectsync::2024-06-01__2024-06-01::tconnectsync",
                    "tconnectsync::2024-05-01__2024-05-01::tconnectsync",
                ],
            }
        ),
        source_files=[
            Path("/tmp/raw/tconnectsync/2024-06-01__2024-06-01/normalized/cgm.csv"),
            Path("/tmp/raw/tconnectsync/2024-05-01__2024-05-01/normalized/cgm.csv"),
        ],
    )

    manifest = build_export_manifest(ingested)

    orders = dict(zip(manifest["source_file"], manifest["source_order"]))
    assert orders["tconnectsync::2024-05-01__2024-05-01::tconnectsync"] < orders["tconnectsync::2024-06-01__2024-06-01::tconnectsync"]


def _write_minimal_normalized_window(root):
    normalized = root / "tconnectsync" / "2026-11-01__2026-11-14" / "normalized"
    normalized.mkdir(parents=True)
    (normalized / "cgm.csv").write_text("timestamp,glucose\n2026-11-18 07:40:00,110\n2026-11-18 07:45:00,112\n", encoding="utf-8")
    (normalized / "bolus.csv").write_text("timestamp,bolus_units\n2026-11-18 07:45:00,4.5\n", encoding="utf-8")


def test_hand_kept_carb_log_is_ingested_from_any_directory(tmp_path):
    _write_minimal_normalized_window(tmp_path)
    log_dir = tmp_path / "meal_log"
    log_dir.mkdir()
    (log_dir / "carbs.csv").write_text("Timestamp, carb_grams\n2026-11-18 07:42:00,45\n2026-11-19 07:55:00,38\n", encoding="utf-8")
    (tmp_path / "carb_log.csv").write_text("timestamp,carb_grams\n2026-11-20 08:01:00,52\n", encoding="utf-8")

    data = load_tandem_exports(tmp_path)

    assert data.carbs["carb_grams"].tolist() == [45, 38, 52]
    assert data.carbs["timestamp"].tolist() == list(pd.to_datetime(["2026-11-18 07:42:00", "2026-11-19 07:55:00", "2026-11-20 08:01:00"]))
    assert len(data.bolus) == 1


def test_carb_log_without_required_columns_is_not_treated_as_canonical(tmp_path):
    _write_minimal_normalized_window(tmp_path)
    (tmp_path / "carbs.csv").write_text("when,grams\n2026-11-18 07:42:00,45\n", encoding="utf-8")

    data = load_tandem_exports(tmp_path)

    assert data.carbs.empty
