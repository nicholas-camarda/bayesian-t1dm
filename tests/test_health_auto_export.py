from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bayesian_t1dm.cli import main
from bayesian_t1dm.features import FeatureConfig, build_feature_frame
from bayesian_t1dm.health_auto_export import build_analysis_ready_health_dataset, build_health_context_frame
from bayesian_t1dm.ingest import IngestedData, load_tandem_exports


def _build_health_auto_export_payload(
    *,
    step_rows: list[dict[str, object]] | None = None,
    sleep_rows: list[dict[str, object]] | None = None,
    heart_rate_rows: list[dict[str, object]] | None = None,
    hrv_rows: list[dict[str, object]] | None = None,
    respiratory_rows: list[dict[str, object]] | None = None,
    resting_rows: list[dict[str, object]] | None = None,
    weight_rows: list[dict[str, object]] | None = None,
    workout_rows: list[dict[str, object]] | None = None,
    include_glucose: bool = True,
) -> dict[str, object]:
    metrics: list[dict[str, object]] = [
        {
            "name": "step_count",
            "units": "count",
            "data": step_rows
            or [
                {"date": "2025-05-24 00:00:00 -0400", "qty": 10, "source": "Oura"},
                {"date": "2025-05-24 00:05:00 -0400", "qty": 15, "source": "Oura"},
                {"date": "2025-05-24 00:10:00 -0400", "qty": 20, "source": "Oura"},
            ],
        },
        {
            "name": "sleep_analysis",
            "units": "hr",
            "data": sleep_rows
            or [
                {
                    "date": "2025-05-24 00:00:00 -0400",
                    "sleepStart": "2025-05-23 23:00:00 -0400",
                    "sleepEnd": "2025-05-24 07:00:00 -0400",
                    "inBedStart": "2025-05-23 22:45:00 -0400",
                    "inBedEnd": "2025-05-24 07:10:00 -0400",
                    "totalSleep": 7.5,
                    "inBed": 8.4167,
                    "core": 4.0,
                    "deep": 1.5,
                    "rem": 2.0,
                    "awake": 0.9167,
                    "source": "Oura",
                }
            ],
        },
        {
            "name": "heart_rate",
            "units": "count/min",
            "data": heart_rate_rows
            or [
                {"date": "2025-05-24 00:00:00 -0400", "Avg": 70, "Min": 68, "Max": 72, "source": "Oura"},
                {"date": "2025-05-24 00:05:00 -0400", "Avg": 74, "Min": 70, "Max": 78, "source": "Oura"},
            ],
        },
        {
            "name": "heart_rate_variability",
            "units": "ms",
            "data": hrv_rows
            or [
                {"date": "2025-05-24 00:00:00 -0400", "qty": 40, "source": "Apple Watch"},
                {"date": "2025-05-24 01:00:00 -0400", "qty": 50, "source": "Apple Watch"},
            ],
        },
        {
            "name": "respiratory_rate",
            "units": "count/min",
            "data": respiratory_rows
            or [
                {"date": "2025-05-24 02:00:00 -0400", "qty": 13.2, "source": "Oura"},
                {"date": "2025-05-24 03:00:00 -0400", "qty": 13.6, "source": "Oura"},
            ],
        },
        {
            "name": "resting_heart_rate",
            "units": "count/min",
            "data": resting_rows or [{"date": "2025-05-24 07:00:00 -0400", "qty": 62, "source": "Apple Watch"}],
        },
        {
            "name": "weight_body_mass",
            "units": "lb",
            "data": weight_rows or [{"date": "2025-05-24 08:00:00 -0400", "qty": 175.5, "source": "Apple Watch"}],
        },
        {
            "name": "active_energy",
            "units": "kcal",
            "data": [{"date": "2025-05-24 00:00:00 -0400", "qty": 5, "source": "Oura"}],
        },
    ]
    if include_glucose:
        metrics.append(
            {
                "name": "blood_glucose",
                "units": "mg/dL",
                "data": [{"date": "2025-05-24 00:02:00 -0400", "qty": 111, "source": "Dexcom G6"}],
            }
        )
    return {
        "data": {
            "metrics": metrics,
            "workouts": workout_rows
            or [
                {
                    "id": "workout-1",
                    "name": "Walk",
                    "start": "2025-05-24 06:00:00 -0400",
                    "end": "2025-05-24 06:20:00 -0400",
                    "duration": 1200,
                    "distance": {"qty": 1.2, "units": "mi"},
                    "activeEnergyBurned": {"qty": 110, "units": "kcal"},
                    "avgHeartRate": {"qty": 115, "units": "bpm"},
                    "maxHeartRate": {"qty": 132, "units": "bpm"},
                }
            ],
            "heartRateNotifications": [],
        }
    }


def _write_health_auto_export_bundle(bundle_dir: Path, *, json_name: str, payload: dict[str, object]) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    json_path = bundle_dir / json_name
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    (bundle_dir / "Walk-Route-20250524_060000.gpx").write_text("<gpx></gpx>\n", encoding="utf-8")
    return json_path


def _write_two_bundle_parent(parent_dir: Path) -> Path:
    old_bundle = parent_dir / "HealthAutoExport_20260331120000"
    new_bundle = parent_dir / "HealthAutoExport_20260331133020"
    _write_health_auto_export_bundle(
        old_bundle,
        json_name="HealthAutoExport-2024-05-24-2025-05-24.json",
        payload=_build_health_auto_export_payload(
            step_rows=[
                {"date": "2025-05-23 23:55:00 -0400", "qty": 8, "source": "Oura"},
                {"date": "2025-05-24 00:00:00 -0400", "qty": 10, "source": "Oura"},
                {"date": "2025-05-24 00:05:00 -0400", "qty": 15, "source": "Oura"},
            ],
            sleep_rows=[
                {
                    "date": "2025-05-24 00:00:00 -0400",
                    "sleepStart": "2025-05-23 23:00:00 -0400",
                    "sleepEnd": "2025-05-24 07:00:00 -0400",
                    "inBedStart": "2025-05-23 22:45:00 -0400",
                    "inBedEnd": "2025-05-24 07:10:00 -0400",
                    "totalSleep": 7.5,
                    "inBed": 8.4167,
                    "core": 4.0,
                    "deep": 1.5,
                    "rem": 2.0,
                    "awake": 0.9167,
                    "source": "Oura",
                }
            ],
            workout_rows=[
                {
                    "id": "workout-1",
                    "name": "Walk",
                    "start": "2025-05-24 06:00:00 -0400",
                    "end": "2025-05-24 06:20:00 -0400",
                    "duration": 1200,
                    "distance": {"qty": 1.2, "units": "mi"},
                    "activeEnergyBurned": {"qty": 110, "units": "kcal"},
                    "avgHeartRate": {"qty": 115, "units": "bpm"},
                    "maxHeartRate": {"qty": 132, "units": "bpm"},
                }
            ],
        ),
    )
    _write_health_auto_export_bundle(
        new_bundle,
        json_name="HealthAutoExport-2025-05-24-2026-03-31.json",
        payload=_build_health_auto_export_payload(
            step_rows=[
                {"date": "2025-05-24 00:00:00 -0400", "qty": 100, "source": "Oura"},
                {"date": "2025-05-24 00:05:00 -0400", "qty": 150, "source": "Oura"},
                {"date": "2025-05-24 00:10:00 -0400", "qty": 200, "source": "Oura"},
            ],
            sleep_rows=[
                {
                    "date": "2025-05-24 00:00:00 -0400",
                    "sleepStart": "2025-05-23 23:00:00 -0400",
                    "sleepEnd": "2025-05-24 07:00:00 -0400",
                    "inBedStart": "2025-05-23 22:40:00 -0400",
                    "inBedEnd": "2025-05-24 07:15:00 -0400",
                    "totalSleep": 8.2,
                    "inBed": 8.6,
                    "core": 4.4,
                    "deep": 1.7,
                    "rem": 2.1,
                    "awake": 0.4,
                    "source": "Oura",
                }
            ],
            heart_rate_rows=[
                {"date": "2025-05-24 00:00:00 -0400", "Avg": 80, "Min": 78, "Max": 84, "source": "Oura"},
                {"date": "2025-05-24 00:05:00 -0400", "Avg": 82, "Min": 79, "Max": 88, "source": "Oura"},
            ],
            hrv_rows=[{"date": "2025-05-24 00:00:00 -0400", "qty": 55, "source": "Apple Watch"}],
            workout_rows=[
                {
                    "id": "workout-1",
                    "name": "Walk",
                    "start": "2025-05-24 06:00:00 -0400",
                    "end": "2025-05-24 06:20:00 -0400",
                    "duration": 1500,
                    "distance": {"qty": 1.5, "units": "mi"},
                    "activeEnergyBurned": {"qty": 140, "units": "kcal"},
                    "avgHeartRate": {"qty": 121, "units": "bpm"},
                    "maxHeartRate": {"qty": 138, "units": "bpm"},
                }
            ],
        ),
    )
    return parent_dir


def _write_tandem_cgm_export(raw_dir: Path) -> Path:
    workbook = raw_dir / "therapy_events_2025-05.xlsx"
    frame = pd.DataFrame(
        {
            "eventDateTime": pd.date_range("2025-05-24 00:00:00", periods=288, freq="5min"),
            "egv_estimatedGlucoseValue": [110 + (index % 18) for index in range(288)],
        }
    )
    with pd.ExcelWriter(workbook) as writer:
        frame.to_excel(writer, index=False)
    return workbook


def _build_dataset(workspace_root: Path, raw_root: Path) -> Path:
    """Run build-dataset on a raw tree and return the merged dataset folder."""
    assert main(["--root", str(workspace_root), "build-dataset", "--raw", str(raw_root)]) == 0
    return raw_root.parent / "merged"


def test_health_exports_are_read_where_dropped_and_identical_copies_count_once(tmp_path):
    raw_root = tmp_path / "raw"
    exports = raw_root / "health_auto_export"
    _write_two_bundle_parent(exports)
    # The same export dropped a second time, loose in the folder, must not double-count.
    newest = next(exports.rglob("HealthAutoExport-2025-05-24-2026-03-31.json"))
    (exports / newest.name).write_bytes(newest.read_bytes())

    data = load_tandem_exports(raw_root, include_health_auto_export=True)
    activity = data.health_activity.loc[pd.to_datetime(data.health_activity["timestamp"]) == pd.Timestamp("2025-05-24 00:00:00")]
    assert len(activity) == 1 and float(activity["activity_value"].iloc[0]) == 100.0
    assert len(data.workouts.loc[data.workouts["workout_id"] == "workout-1"]) == 1

    moved = tmp_path / "moved" / "raw"
    moved.parent.mkdir()
    raw_root.rename(moved)
    assert len(load_tandem_exports(moved, include_health_auto_export=True).health_activity) == len(data.health_activity)


def test_load_tandem_exports_unifies_and_dedupes_health_auto_export(tmp_path, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    cloud_root = tmp_path / "cloud"
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(cloud_root))
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(runtime_root))

    _write_two_bundle_parent(cloud_root / "data" / "raw" / "health_auto_export")

    data = load_tandem_exports(cloud_root / "data" / "raw", include_health_auto_export=True)

    measurement = data.health_measurements.loc[
        (pd.to_datetime(data.health_measurements["timestamp"]) == pd.Timestamp("2025-05-24 00:00:00"))
        & (data.health_measurements["metric"] == "heart_rate")
        & (data.health_measurements["stat"] == "avg")
        & (data.health_measurements["source_device"] == "Oura")
    ]
    assert len(measurement) == 1
    assert float(measurement["value"].iloc[0]) == 80.0
    assert measurement["export_id"].iloc[0] == "HealthAutoExport-2025-05-24-2026-03-31"

    activity = data.health_activity.loc[pd.to_datetime(data.health_activity["timestamp"]) == pd.Timestamp("2025-05-24 00:00:00")]
    assert len(activity) == 1
    assert float(activity["activity_value"].iloc[0]) == 100.0

    glucose = data.cgm.loc[pd.to_datetime(data.cgm["timestamp"]) == pd.Timestamp("2025-05-24 00:00:00")]
    assert len(glucose) == 1
    assert float(glucose["glucose"].iloc[0]) == 111.0
    assert glucose["source_device"].iloc[0] == "Dexcom G6"

    sleep = data.sleep.loc[pd.to_datetime(data.sleep["date"]) == pd.Timestamp("2025-05-24")]
    assert len(sleep) == 1
    assert float(sleep["total_sleep_hours"].iloc[0]) == 8.2

    workout = data.workouts.loc[data.workouts["workout_id"] == "workout-1"]
    assert len(workout) == 1
    assert float(workout["active_energy_value"].iloc[0]) == 140.0

    assert any(pd.to_datetime(data.health_activity["timestamp"]) == pd.Timestamp("2025-05-23 23:55:00"))
    assert "blood_glucose" not in set(data.health_measurements["metric"])


def test_build_health_analysis_ready_command_writes_tandem_aligned_dataset(tmp_path, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    cloud_root = tmp_path / "cloud"
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(cloud_root))
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(runtime_root))

    raw_root = cloud_root / "data" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    _write_tandem_cgm_export(raw_root)
    _write_two_bundle_parent(raw_root / "health_auto_export")

    merged = _build_dataset(workspace_root, raw_root)
    analysis_ready = pd.read_parquet(merged / "prepared_model_data_5min.parquet")
    analysis_ready["timestamp"] = pd.to_datetime(analysis_ready["timestamp"]).astype(str)
    tandem_only = load_tandem_exports(raw_root, include_health_auto_export=False)
    expected = build_feature_frame(tandem_only, FeatureConfig(horizon_minutes=30)).frame
    pd.testing.assert_series_equal(
        pd.to_datetime(analysis_ready["timestamp"]),
        pd.to_datetime(expected["timestamp"]),
        check_names=False,
        check_dtype=False,
    )

    assert analysis_ready["timestamp"].min() == str(expected["timestamp"].min())
    assert analysis_ready["timestamp"].max() == str(expected["timestamp"].max())
    assert "health_activity_value" in analysis_ready.columns
    assert "prior_night_total_sleep_hours" in analysis_ready.columns
    assert "in_sleep" in analysis_ready.columns
    assert "last_workout_active_energy_value" in analysis_ready.columns
    assert "hrv_latest" in analysis_ready.columns
    assert all("blood_glucose" not in column for column in analysis_ready.columns)

    first_row = analysis_ready.loc[analysis_ready["timestamp"] == "2025-05-24 00:00:00"].iloc[0]
    assert float(first_row["health_activity_value"]) == 100.0
    assert float(first_row["prior_night_total_sleep_hours"]) == 8.2
    assert int(first_row["in_sleep"]) == 1
    assert float(first_row["hrv_latest"]) == 55.0


def test_analysis_ready_dataset_uses_apple_glucose_to_fill_tandem_cgm_gaps():
    tandem_data = IngestedData(
        cgm=pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2025-05-24 00:00:00"), pd.Timestamp("2025-05-24 00:10:00")],
                "glucose": [100.0, 120.0],
                "glucose_source": ["egv_estimatedGlucoseValue", "egv_estimatedGlucoseValue"],
            }
        ),
        bolus=pd.DataFrame({"timestamp": [pd.Timestamp("2025-05-24 00:00:00")], "bolus_units": [1.0]}),
    )
    health_data = IngestedData(
        cgm=pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2025-05-24 00:07:00")],
                "glucose": [111.0],
                "glucose_source": ["apple_health_blood_glucose"],
                "source_device": ["Dexcom G7"],
            }
        ),
        health_measurements=pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2025-05-24 00:00:00")],
                "metric": ["heart_rate_variability"],
                "stat": ["value"],
                "value": [40.0],
                "unit": ["ms"],
                "source_file": ["HealthAutoExport.json"],
                "source_device": ["Apple Watch"],
            }
        ),
    )

    dataset = build_analysis_ready_health_dataset(
        tandem_data=tandem_data,
        health_data=health_data,
        config=FeatureConfig(horizon_minutes=5),
    )

    row_0005 = dataset.frame.loc[dataset.frame["timestamp"].eq(pd.Timestamp("2025-05-24 00:05:00"))].iloc[0]
    assert int(row_0005["missing_cgm"]) == 0
    assert float(row_0005["glucose_observed"]) == 111.0


def test_build_health_analysis_ready_command_computes_plausible_workout_windows(tmp_path, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    cloud_root = tmp_path / "cloud"
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(cloud_root))
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(runtime_root))

    raw_root = cloud_root / "data" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    _write_tandem_cgm_export(raw_root)
    bundle_dir = raw_root / "health_auto_export" / "HealthAutoExport_20260331133020"
    _write_health_auto_export_bundle(
        bundle_dir,
        json_name="HealthAutoExport-2025-05-24-2025-05-24.json",
        payload=_build_health_auto_export_payload(
            workout_rows=[
                {
                    "id": "workout-1",
                    "name": "Walk",
                    "start": "2025-05-24 06:00:00 -0400",
                    "end": "2025-05-24 06:20:00 -0400",
                    "duration": 1200,
                    "distance": {"qty": 1.2, "units": "mi"},
                    "activeEnergyBurned": {"qty": 110, "units": "kcal"},
                    "avgHeartRate": {"qty": 115, "units": "bpm"},
                    "maxHeartRate": {"qty": 132, "units": "bpm"},
                },
                {
                    "id": "workout-2",
                    "name": "Run",
                    "start": "2025-05-24 18:00:00 -0400",
                    "end": "2025-05-24 18:20:00 -0400",
                    "duration": 1200,
                    "distance": {"qty": 2.4, "units": "mi"},
                    "activeEnergyBurned": {"qty": 220, "units": "kcal"},
                    "avgHeartRate": {"qty": 140, "units": "bpm"},
                    "maxHeartRate": {"qty": 160, "units": "bpm"},
                },
            ],
        ),
    )
    merged = _build_dataset(workspace_root, raw_root)
    analysis_ready = pd.read_parquet(merged / "prepared_model_data_5min.parquet")
    analysis_ready["timestamp"] = pd.to_datetime(analysis_ready["timestamp"]).astype(str)

    row_0000 = analysis_ready.loc[analysis_ready["timestamp"] == "2025-05-24 00:00:00"].iloc[0]
    row_0700 = analysis_ready.loc[analysis_ready["timestamp"] == "2025-05-24 07:00:00"].iloc[0]
    row_1300 = analysis_ready.loc[analysis_ready["timestamp"] == "2025-05-24 13:00:00"].iloc[0]
    row_1900 = analysis_ready.loc[analysis_ready["timestamp"] == "2025-05-24 19:00:00"].iloc[0]

    assert int(row_0000["recent_workout_6h"]) == 0
    assert float(row_0000["workout_count_24h"]) == 0.0
    assert int(row_0700["recent_workout_6h"]) == 1
    assert float(row_0700["workout_count_24h"]) == 1.0
    assert int(row_1300["recent_workout_6h"]) == 0
    assert float(row_1300["workout_count_24h"]) == 1.0
    assert int(row_1900["recent_workout_6h"]) == 1
    assert float(row_1900["workout_count_24h"]) == 2.0
    assert int(row_1900["workout_summary_plausible"]) == 1


def test_workout_recency_flags_use_elapsed_time_across_timestamp_gaps():
    base_frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 06:00:00",
                    "2025-01-01 11:59:00",
                    "2025-01-01 19:01:00",
                ]
            )
        }
    )
    workouts = pd.DataFrame(
        {
            "start_timestamp": [pd.Timestamp("2025-01-01 05:30:00")],
            "end_timestamp": [pd.Timestamp("2025-01-01 06:00:00")],
            "duration_seconds": [1800.0],
            "distance_value": [0.0],
            "active_energy_value": [120.0],
            "avg_heart_rate": [120.0],
            "max_heart_rate": [140.0],
        }
    )

    context, _ = build_health_context_frame(base_frame, IngestedData(workouts=workouts), freq="1min")

    row_1159 = context.loc[context["timestamp"].eq(pd.Timestamp("2025-01-01 11:59:00"))].iloc[0]
    row_1901 = context.loc[context["timestamp"].eq(pd.Timestamp("2025-01-01 19:01:00"))].iloc[0]
    assert float(row_1159["minutes_since_last_workout"]) == 359.0
    assert int(row_1159["recent_workout_6h"]) == 1
    assert float(row_1901["minutes_since_last_workout"]) == 781.0
    assert int(row_1901["recent_workout_12h"]) == 0


def test_screen_health_features_command_writes_scores_and_report(tmp_path, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    cloud_root = tmp_path / "cloud"
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(cloud_root))
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(runtime_root))

    raw_root = cloud_root / "data" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    _write_tandem_cgm_export(raw_root)
    _write_health_auto_export_bundle(
        raw_root / "health_auto_export" / "HealthAutoExport_20260331133020",
        json_name="HealthAutoExport-2025-05-24-2025-05-24.json",
        payload=_build_health_auto_export_payload(),
    )
    _build_dataset(workspace_root, raw_root)

    report = tmp_path / "health_feature_screening.md"
    scores = tmp_path / "health_feature_scores.csv"
    exit_code = main(
        [
            "--root",
            str(workspace_root),
            "screen-health-features",
            "--raw",
            str(raw_root),
            "--report",
            str(report),
            "--scores",
            str(scores),
        ]
    )

    assert exit_code == 0
    assert report.exists()
    assert scores.exists()

    score_frame = pd.read_csv(scores)
    assert not score_frame.empty
    assert "health_activity_value" in set(score_frame["feature"])
    assert "prior_night_total_sleep_hours" in set(score_frame["feature"])

    text = report.read_text(encoding="utf-8")
    assert "status: completed" in text
    assert "baseline_rmse_mean" in text
    assert "augmented_rmse_mean" in text


def test_screen_health_features_without_apple_skips_cleanly(tmp_path, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    cloud_root = tmp_path / "cloud"
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(cloud_root))
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(runtime_root))

    raw_root = cloud_root / "data" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    _write_tandem_cgm_export(raw_root)
    _build_dataset(workspace_root, raw_root)

    report = tmp_path / "health_feature_screening.md"
    scores = tmp_path / "health_feature_scores.csv"
    exit_code = main(
        [
            "--root",
            str(workspace_root),
            "screen-health-features",
            "--raw",
            str(raw_root),
            "--report",
            str(report),
            "--scores",
            str(scores),
        ]
    )

    assert exit_code == 0
    assert report.exists()
    assert scores.exists()
    score_frame = pd.read_csv(scores)
    assert score_frame.empty
    text = report.read_text(encoding="utf-8")
    assert "status: skipped" in text
    assert "Apple Health not available; screening skipped." in text


def test_build_dataset_with_apple_builds_enriched_dataset_and_reports(tmp_path, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    cloud_root = tmp_path / "cloud"
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(cloud_root))
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(runtime_root))

    raw_root = cloud_root / "data" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    _write_tandem_cgm_export(raw_root)
    _write_two_bundle_parent(raw_root / "health_auto_export")

    merged = _build_dataset(workspace_root, raw_root)
    prepared = pd.read_parquet(merged / "prepared_model_data_5min.parquet")
    assert "health_activity_value" in prepared.columns
    assert "prior_night_total_sleep_hours" in prepared.columns
    text = (runtime_root / "output" / "prepare" / "model_data_preparation.md").read_text(encoding="utf-8")
    assert "- mode: apple_enriched" in text
    assert "- apple_available: True" in text
    assert "- overlap_start: 2025-05-24 00:00:00" in text
    validation = (merged / "validation_report.md").read_text(encoding="utf-8")
    assert "Apple Health: 2 export files (2 distinct" in validation


def test_logged_food_entries_become_one_nutrition_row_per_entry(tmp_path):
    from bayesian_t1dm.health_auto_export import load_unified_health_auto_export_tables

    payload = _build_health_auto_export_payload()
    payload["data"]["metrics"] += [
        {"name": "carbohydrates", "units": "g", "data": [{"date": "2026-04-04 08:17:00 -0400", "qty": 49.2, "source": ""}]},
        {"name": "protein", "units": "g", "data": [{"date": "2026-04-04 08:17:00 -0400", "qty": 18.0, "source": ""}]},
        {"name": "total_fat", "units": "g", "data": [{"date": "2026-04-04 08:17:00 -0400", "qty": 22.5, "source": ""}]},
    ]
    _write_health_auto_export_bundle(tmp_path / "raw" / "health_auto_export" / "x", json_name="HealthAutoExport-2026-04-01-2026-04-30.json", payload=payload)

    nutrition = load_unified_health_auto_export_tables(tmp_path / "raw")["nutrition"]

    assert len(nutrition) == 1
    row = nutrition.iloc[0]
    assert pd.Timestamp(row["timestamp"]) == pd.Timestamp("2026-04-04 08:17:00")  # local wall-clock time
    assert (row["carb_grams"], row["protein_grams"], row["fat_grams"]) == (49.2, 18.0, 22.5)
