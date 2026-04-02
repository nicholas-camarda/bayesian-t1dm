from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import pandas as pd

import bayesian_t1dm.cli as cli
from bayesian_t1dm.evaluate import CalibrationSummary, WalkForwardReport
from bayesian_t1dm.cli import main
from bayesian_t1dm.acquisition import ExportWindow, NormalizedWindowResult
from bayesian_t1dm.health_auto_export import ModelDataPreparationResult
from bayesian_t1dm.therapy_research import _synthetic_base_dataset


def _latest_log_dir(runtime_root: Path, command: str) -> Path:
    latest = json.loads((runtime_root / "output" / "logs" / command / "latest.json").read_text(encoding="utf-8"))
    return Path(latest["run_dir"])


def _write_tandem_cgm_export(raw_dir: Path, *, start: str = "2025-05-24 00:00:00", periods: int = 288) -> Path:
    workbook = raw_dir / "therapy_events_2025-05.xlsx"
    frame = pd.DataFrame(
        {
            "eventDateTime": pd.date_range(start, periods=periods, freq="5min"),
            "egv_estimatedGlucoseValue": [110 + (index % 18) for index in range(periods)],
        }
    )
    with pd.ExcelWriter(workbook) as writer:
        frame.to_excel(writer, index=False)
    return workbook


def test_validate_raw_command_writes_summary_report(tmp_path):
    workbook = tmp_path / "therapy_events_2023-02.xlsx"
    frame = pd.DataFrame(
        {
            "eventDateTime": pd.date_range("2023-02-04 09:00:00", periods=4, freq="5min"),
            "egv_estimatedGlucoseValue": [100, 101, 102, 103],
        }
    )
    with pd.ExcelWriter(workbook) as writer:
        frame.to_excel(writer, index=False)

    report = tmp_path / "validation.md"
    exit_code = main(["--root", str(tmp_path), "validate-raw", "--raw", str(tmp_path), "--report", str(report)])

    assert exit_code == 0
    text = report.read_text(encoding="utf-8")
    assert "dense_cgm_files: 1" in text
    assert "therapy_events_2023-02.xlsx" in text


class _FakeApiClient:
    last_instance = None

    def __init__(self, workspace, **kwargs) -> None:
        self.workspace = workspace
        self.kwargs = kwargs
        self.calls: list[tuple[str, str | None]] = []
        _FakeApiClient.last_instance = self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def login(self, credentials, step_log=None):
        self.calls.append(("login", credentials.email))

    def start_trace(self):
        self.calls.append(("trace_start", None))

    def stop_trace(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("trace", encoding="utf-8")
        return path

    def capture_screenshot(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("shot", encoding="utf-8")
        return path

    def capture_page_diagnostics(self, stem: str):
        return None

    def export_daily_timeline_window(self, window: ExportWindow, download_dir: Path, step_log=None):
        self.calls.append(("export", window.window_id))
        window_dir = self.workspace.cloud_raw / "tconnectsync" / window.window_id / "normalized"
        window_dir.mkdir(parents=True, exist_ok=True)
        is_complete_window = not window.window_id.endswith("2024-02-29")
        pd.DataFrame(
            {
                "timestamp": pd.date_range(window.start_date, periods=2, freq="5min"),
                "glucose": [100, 101],
                "source_file": [f"tconnectsync::{window.window_id}::tconnectsync"] * 2,
                "source_label": ["tconnectsync"] * 2,
            }
        ).to_csv(window_dir / "cgm.csv", index=False)
        pd.DataFrame(
            {
                "window_id": [window.window_id],
                "kind": ["cgm"],
                "normalized_path": [str(window_dir / "cgm.csv")],
            }
        ).to_csv(self.workspace.cloud_raw / "tconnectsync" / window.window_id / "window_manifest.csv", index=False)
        return NormalizedWindowResult(
            window_id=window.window_id,
            requested_start=window.start_date.isoformat(),
            requested_end=window.end_date.isoformat(),
            endpoint_family="tconnectsync",
            source_label=f"tconnectsync::{window.window_id}::tconnectsync",
            raw_artifacts=(),
            normalized_paths={"cgm": str(window_dir / "cgm.csv")},
            row_counts={"cgm": 2},
            observed_first_timestamp=pd.Timestamp(window.start_date).isoformat(),
            observed_last_timestamp=pd.Timestamp(window.end_date if is_complete_window else window.start_date).isoformat(),
            observed_window_days=1 if is_complete_window else 10,
            is_complete_window=is_complete_window,
            activity_present=False,
            manifest_path=self.workspace.cloud_raw / "tconnectsync" / window.window_id / "window_manifest.csv",
        )


def test_collect_command_uses_api_client_by_default(tmp_path, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    monkeypatch.setenv("TANDEM_SOURCE_EMAIL", "me@example.com")
    monkeypatch.setenv("TANDEM_SOURCE_PASSWORD", "secret")
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(tmp_path / "cloud"))
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setattr(cli, "TConnectSyncSourceClient", lambda paths, **kwargs: _FakeApiClient(paths, **kwargs))

    exit_code = main(
        [
            "--root",
            str(workspace_root),
            "collect",
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-01",
        ]
    )

    assert exit_code == 0
    client = _FakeApiClient.last_instance
    assert client is not None
    assert any(call[0] == "export" for call in client.calls)
    assert (client.workspace.cloud_raw / "tconnectsync").exists()
    assert (client.workspace.cloud_raw / "tconnectsync" / "2024-01-01__2024-01-01" / "window_manifest.csv").exists()
    assert (client.workspace.reports / "tandem_acquisition_summary.md").exists()


def test_collect_command_writes_redacted_run_logs(tmp_path, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    monkeypatch.setenv("TANDEM_SOURCE_EMAIL", "me@example.com")
    monkeypatch.setenv("TANDEM_SOURCE_PASSWORD", "secret")
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(tmp_path / "cloud"))
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.setattr(cli, "TConnectSyncSourceClient", lambda paths, **kwargs: _FakeApiClient(paths, **kwargs))

    exit_code = main(
        [
            "--root",
            str(workspace_root),
            "collect",
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-01",
        ]
    )

    assert exit_code == 0
    run_dir = _latest_log_dir(runtime_root, "collect")
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    event_names = {event["event"] for event in events}
    assert "login.start" in event_names
    assert "window.complete" in event_names
    login_start = next(event for event in events if event["event"] == "login.start")
    assert login_start["fields"]["email"] == "[REDACTED]"
    assert (run_dir / "run.log").exists()
    assert (run_dir / "run_meta.json").exists()


def test_backfill_command_continues_on_partial_window(tmp_path, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    monkeypatch.setenv("TANDEM_SOURCE_EMAIL", "me@example.com")
    monkeypatch.setenv("TANDEM_SOURCE_PASSWORD", "secret")
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(tmp_path / "cloud"))
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setattr(cli, "TConnectSyncSourceClient", lambda paths, **kwargs: _FakeApiClient(paths, **kwargs))

    exit_code = main(
        [
            "--root",
            str(workspace_root),
            "backfill",
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-02-29",
            "--window-days",
            "30",
        ]
    )

    assert exit_code == 0
    client = _FakeApiClient.last_instance
    assert client is not None
    export_calls = [call for call in client.calls if call[0] == "export"]
    assert len(export_calls) == 2
    assert (client.workspace.cloud_raw / "tandem_export_manifest.csv").exists()
    assert (client.workspace.reports / "tandem_acquisition_summary.md").exists()


def test_ingest_command_defaults_report_to_runtime_output(tmp_path, tandem_fixture_dir, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    cloud_root = tmp_path / "cloud"
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(cloud_root))
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(runtime_root))

    exit_code = main(
        [
            "--root",
            str(workspace_root),
            "ingest",
            "--raw",
            str(tandem_fixture_dir),
        ]
    )

    assert exit_code == 0
    assert (runtime_root / "output" / "coverage.md").exists()
    assert (runtime_root / "output" / "coverage_review.html").exists()
    assert (cloud_root / "data" / "raw" / "tandem_export_manifest.csv").exists()
    coverage_text = (runtime_root / "output" / "coverage.md").read_text(encoding="utf-8")
    assert "[coverage_review_html](coverage_review.html)" in coverage_text


def test_normalize_raw_command_rebuilds_archived_window(tmp_path, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    cloud_root = tmp_path / "cloud"
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(cloud_root))
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(runtime_root))

    raw_dir = cloud_root / "data" / "raw" / "tconnectsync" / "2024-05-01__2024-05-03" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for kind, payload in {
        "cgm": [{"eventDateTime": "2024-05-01T00:00:00-04:00", "egv_estimatedGlucoseValue": 110}],
        "bolus": [{"completionDateTime": "2024-05-01T08:00:00-04:00", "actualTotalBolusRequested": 2.5}],
        "basal": [{"startDateTime": "2024-05-01T00:00:00-04:00", "endDateTime": "2024-05-02 00:00:00", "basalRate": 0.7}],
        "activity": [],
    }.items():
        (raw_dir / f"{kind}.json").write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(["--root", str(workspace_root), "normalize-raw"])

    assert exit_code == 0
    assert (cloud_root / "data" / "raw" / "tconnectsync" / "2024-05-01__2024-05-03" / "window_manifest.csv").exists()
    assert (runtime_root / "output" / "normalize_raw_summary.md").exists()


class _FailIfFitModel:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def fit(self, feature_frame):
        raise AssertionError("final fit should be skipped when --skip-recommendations is set")


def test_run_skip_recommendations_writes_skipped_policy(tmp_path, tandem_fixture_dir, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(tmp_path / "cloud"))
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setattr(cli, "BayesianGlucoseModel", _FailIfFitModel)
    monkeypatch.setattr(
        cli,
        "run_walk_forward",
        lambda features, model, n_folds=4: WalkForwardReport(
            folds=[],
            aggregate=CalibrationSummary(mae=10.0, rmse=12.0, coverage=0.8, interval_width=30.0),
            aggregate_persistence_mae=14.0,
        ),
    )

    exit_code = main(
        [
            "--root",
            str(workspace_root),
            "run",
            "--raw",
            str(tandem_fixture_dir),
            "--skip-recommendations",
        ]
    )

    assert exit_code == 0
    report_json = tmp_path / "runtime" / "output" / "run_summary.json"
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["fit_diagnostics"] is None
    assert payload["data_quality"]["status"] in {"good", "degraded"}
    assert payload["recommendation_policy"]["status"] == "skipped"
    assert payload["recommendation_policy"]["reasons"] == ["skipped_by_flag"]
    assert (tmp_path / "runtime" / "output" / "run_review.html").exists()
    report_text = (tmp_path / "runtime" / "output" / "run_summary.md").read_text(encoding="utf-8")
    assert "[run_review_html](run_review.html)" in report_text
    run_dir = _latest_log_dir(tmp_path / "runtime", "run")
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["event"] == "command.stage.start" and event["stage"] == "run.walk_forward" for event in events)
    assert any(event["event"] == "command.complete" and event["status"] == "success" for event in events)


def test_prepare_model_data_without_apple_writes_tandem_only_dataset(tmp_path, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    cloud_root = tmp_path / "cloud"
    runtime_root = tmp_path / "runtime"
    raw_root = cloud_root / "data" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    _write_tandem_cgm_export(raw_root)

    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(cloud_root))
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(runtime_root))

    output = tmp_path / "prepared_model_data_5min.csv"
    report = tmp_path / "model_data_preparation.md"
    exit_code = main(
        [
            "--root",
            str(workspace_root),
            "prepare-model-data",
            "--raw",
            str(raw_root),
            "--output",
            str(output),
            "--report",
            str(report),
            "--skip-backfill",
        ]
    )

    assert exit_code == 0
    assert output.exists()
    assert report.exists()

    prepared = pd.read_csv(output)
    assert "health_activity_value" not in prepared.columns
    text = report.read_text(encoding="utf-8")
    assert "- mode: tandem_only" in text
    assert "- apple_available: False" in text
    assert "- health_features_included: False" in text
    assert "below the requested minimum" in text
    run_dir = _latest_log_dir(runtime_root, "prepare-model-data")
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["event"] == "prepare_model_data.apple_import.skipped" for event in events)
    assert any(event["event"] == "command.stage.start" and event["stage"] == "prepare_model_data.load_inputs" for event in events)
    assert any(event["event"] == "command.stage.start" and event["stage"] == "prepare_model_data.build_dataset" for event in events)
    assert any(event["event"] == "prepare_model_data.dataset_summary" for event in events)


def test_prepare_model_data_backfills_tandem_when_needed_without_apple(tmp_path, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    cloud_root = tmp_path / "cloud"
    runtime_root = tmp_path / "runtime"
    raw_root = cloud_root / "data" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    _write_tandem_cgm_export(raw_root)

    monkeypatch.setenv("TANDEM_SOURCE_EMAIL", "me@example.com")
    monkeypatch.setenv("TANDEM_SOURCE_PASSWORD", "secret")
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(cloud_root))
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.setattr(cli, "TConnectSyncSourceClient", lambda paths, **kwargs: _FakeApiClient(paths, **kwargs))

    report = tmp_path / "model_data_preparation.md"
    exit_code = main(
        [
            "--root",
            str(workspace_root),
            "prepare-model-data",
            "--raw",
            str(raw_root),
            "--report",
            str(report),
            "--history-days",
            "30",
        ]
    )

    assert exit_code == 0
    client = _FakeApiClient.last_instance
    assert client is not None
    assert any(call[0] == "export" for call in client.calls)
    text = report.read_text(encoding="utf-8")
    assert "- backfill_status: completed" in text
    assert "- requested_tandem_start: 2025-04-25 00:00:00" in text
    assert "- requested_tandem_end: 2025-05-24 00:00:00" in text


def test_validate_therapy_infra_command_writes_artifacts(tmp_path):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    report_dir = tmp_path / "reports"

    exit_code = main(
        [
            "--root",
            str(workspace_root),
            "validate-therapy-infra",
            "--report-dir",
            str(report_dir),
            "--include-models",
            "ridge,segmented_ridge,tree_boost",
        ]
    )

    assert exit_code == 0
    assert (report_dir / "therapy_infra_validation.md").exists()
    assert (report_dir / "therapy_synthetic_results.csv").exists()
    assert (report_dir / "therapy_synthetic_recommendation_audit.md").exists()


def test_review_therapy_evidence_command_writes_report_and_supporting_artifacts(tmp_path, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
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
    monkeypatch.setattr(cli, "_prepare_model_data", lambda args, paths, session=None: preparation)

    report = tmp_path / "therapy_evidence_review.html"
    exit_code = main(
        [
            "--root",
            str(workspace_root),
            "review-therapy-evidence",
            "--report",
            str(report),
            "--include-models",
            "ridge,segmented_ridge,tree_boost",
        ]
    )

    assert exit_code == 0
    assert report.exists()
    assert (tmp_path / "model_data_preparation.md").exists()
    assert (tmp_path / "therapy_research_gate.md").exists()
    assert (tmp_path / "therapy_infra_validation.md").exists()


def test_ingest_failure_is_logged(tmp_path, tandem_fixture_dir, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(tmp_path / "cloud"))
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.setattr(cli, "load_tandem_exports", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    try:
        main(
            [
                "--root",
                str(workspace_root),
                "ingest",
                "--raw",
                str(tandem_fixture_dir),
            ]
        )
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected ingest to fail")

    run_dir = _latest_log_dir(runtime_root, "ingest")
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["event"] == "command.error" and event["fields"]["error_type"] == "RuntimeError" for event in events)
    assert any(event["event"] == "command.complete" and event["status"] == "failed" for event in events)
