from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bayesian_t1dm.cli import main
from bayesian_t1dm.ingest import load_tandem_exports


def _write_health_auto_export_bundle(bundle_dir: Path) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "data": {
            "metrics": [
                {
                    "name": "step_count",
                    "units": "count",
                    "data": [
                        {"date": "2025-05-24 00:00:00 -0400", "qty": 10, "source": "Oura"},
                        {"date": "2025-05-24 00:05:00 -0400", "qty": 15, "source": "Oura"},
                        {"date": "2025-05-24 00:10:00 -0400", "qty": 20, "source": "Oura"},
                    ],
                },
                {
                    "name": "sleep_analysis",
                    "units": "hr",
                    "data": [
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
                    "data": [
                        {"date": "2025-05-24 00:00:00 -0400", "Avg": 70, "Min": 68, "Max": 72, "source": "Oura"},
                        {"date": "2025-05-24 00:05:00 -0400", "Avg": 74, "Min": 70, "Max": 78, "source": "Oura"},
                    ],
                },
                {
                    "name": "heart_rate_variability",
                    "units": "ms",
                    "data": [
                        {"date": "2025-05-24 00:00:00 -0400", "qty": 40, "source": "Apple Watch"},
                        {"date": "2025-05-24 01:00:00 -0400", "qty": 50, "source": "Apple Watch"},
                    ],
                },
                {
                    "name": "respiratory_rate",
                    "units": "count/min",
                    "data": [
                        {"date": "2025-05-24 02:00:00 -0400", "qty": 13.2, "source": "Oura"},
                        {"date": "2025-05-24 03:00:00 -0400", "qty": 13.6, "source": "Oura"},
                    ],
                },
                {
                    "name": "resting_heart_rate",
                    "units": "count/min",
                    "data": [
                        {"date": "2025-05-24 07:00:00 -0400", "qty": 62, "source": "Apple Watch"},
                    ],
                },
                {
                    "name": "weight_body_mass",
                    "units": "lb",
                    "data": [
                        {"date": "2025-05-24 08:00:00 -0400", "qty": 175.5, "source": "Apple Watch"},
                    ],
                },
                {
                    "name": "blood_glucose",
                    "units": "mg/dL",
                    "data": [
                        {"date": "2025-05-24 00:00:00 -0400", "qty": 111, "source": "Dexcom G6"},
                    ],
                },
                {
                    "name": "active_energy",
                    "units": "kcal",
                    "data": [
                        {"date": "2025-05-24 00:00:00 -0400", "qty": 5, "source": "Oura"},
                    ],
                },
            ],
            "workouts": [
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
                    "route": [{"latitude": 1.0, "longitude": 2.0}],
                    "heartRateData": [{"date": "2025-05-24 06:00:00 -0400", "Avg": 115, "Min": 110, "Max": 120}],
                    "stepCount": [{"date": "2025-05-24 06:00:00 -0400", "qty": 100, "units": "steps"}],
                    "activeEnergy": [{"date": "2025-05-24 06:00:00 -0400", "qty": 5, "units": "kcal"}],
                }
            ],
            "heartRateNotifications": [
                {
                    "heartNotification": "high_heart_rate_notification",
                    "start": "2025-05-24T10:00:00Z",
                    "end": "2025-05-24T10:03:00Z",
                    "threshhold": 120,
                    "heartRateData": [],
                    "hrvData": [],
                    "source": {"name": "Apple Watch"},
                }
            ],
        }
    }
    json_path = bundle_dir / "HealthAutoExport-2025-05-24-2025-05-24.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    (bundle_dir / "Walk-Route-20250524_060000.gpx").write_text("<gpx></gpx>\n", encoding="utf-8")
    return json_path


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


def test_import_health_auto_export_command_archives_and_normalizes_bundle(tmp_path, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    cloud_root = tmp_path / "cloud"
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(cloud_root))
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(runtime_root))

    bundle_dir = tmp_path / "HealthAutoExport_20260331133020"
    _write_health_auto_export_bundle(bundle_dir)

    exit_code = main(["--root", str(workspace_root), "import-health-auto-export", "--input", str(bundle_dir)])

    assert exit_code == 0
    archive_root = cloud_root / "data" / "raw" / "health_auto_export" / bundle_dir.name
    manifest = archive_root / "health_auto_export_manifest.csv"
    assert manifest.exists()
    assert (archive_root / "raw" / "HealthAutoExport-2025-05-24-2025-05-24.json").exists()
    assert (archive_root / "raw" / "Walk-Route-20250524_060000.gpx").exists()
    assert (archive_root / "normalized" / "activity.csv").exists()
    assert (archive_root / "normalized" / "health_measurements.csv").exists()
    assert (archive_root / "normalized" / "sleep.csv").exists()
    assert (archive_root / "normalized" / "workouts.csv").exists()

    data = load_tandem_exports(cloud_root / "data" / "raw", include_health_auto_export=True)
    assert len(data.health_activity) == 3
    assert len(data.health_measurements) > 0
    assert len(data.sleep) == 1
    assert len(data.workouts) == 1
    assert "blood_glucose" not in set(data.health_measurements["metric"])


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
    bundle_dir = tmp_path / "HealthAutoExport_20260331133020"
    _write_health_auto_export_bundle(bundle_dir)

    import_exit = main(["--root", str(workspace_root), "import-health-auto-export", "--input", str(bundle_dir)])
    assert import_exit == 0

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
            "--horizon",
            "30",
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
    assert "baseline_rmse_mean" in text
    assert "augmented_rmse_mean" in text
