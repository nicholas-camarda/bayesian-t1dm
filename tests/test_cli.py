from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import bayesian_t1dm.cli as cli
from bayesian_t1dm.cli import main


def _fake_merged(preparation):
    """A merged dataset with the parts the research commands read: no food log, no workouts, no pump modes, an empty analysis grid."""
    grid = pd.DataFrame(columns=["timestamp", "glucose", "carb_grams", "bolus_units"])
    return SimpleNamespace(preparation=preparation, data_quality=DataQualitySummary(status="degraded", contributing_window_ids=[], incomplete_window_count=2, reason_counts={}, evaluation_touches_incomplete_windows=False), grid=grid, mode_events=pd.DataFrame(), data=SimpleNamespace(nutrition=pd.DataFrame(), workouts=pd.DataFrame()))
from bayesian_t1dm.acquisition import ExportWindow, NormalizedWindowResult
from bayesian_t1dm.quality import DataQualitySummary


def _latest_log_dir(runtime_root: Path, command: str) -> Path:
    latest = json.loads((runtime_root / "logs" / command / "latest.json").read_text(encoding="utf-8"))
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


def _built_raw_tree(tmp_path: Path, workspace_root: Path, fixture_dir: Path) -> Path:
    """Copy a fixture raw tree into the test cloud folder and build its merged dataset."""
    raw_root = tmp_path / "cloud" / "data" / "raw"
    shutil.copytree(fixture_dir, raw_root)
    assert main(["--root", str(workspace_root), "build-dataset", "--raw", str(raw_root)]) == 0
    return raw_root


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
    assert (client.workspace.output_source / "tandem_acquisition_summary.md").exists()


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
    assert (client.workspace.output_source / "tandem_acquisition_summary.md").exists()


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
    assert (runtime_root / "output" / "source" / "normalize_raw_summary.md").exists()


def test_build_dataset_without_apple_writes_tandem_only_dataset_and_skips_unchanged_inputs(tmp_path, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    cloud_root = tmp_path / "cloud"
    runtime_root = tmp_path / "runtime"
    raw_root = cloud_root / "data" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    _write_tandem_cgm_export(raw_root)
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(cloud_root))
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(runtime_root))

    assert main(["--root", str(workspace_root), "build-dataset", "--raw", str(raw_root)]) == 0

    merged = cloud_root / "data" / "merged"
    prepared = pd.read_parquet(merged / "prepared_model_data_5min.parquet")
    assert "health_activity_value" not in prepared.columns
    manifest = json.loads((merged / "dataset.json").read_text(encoding="utf-8"))
    assert manifest["dataset"]["mode"] == "tandem_only"
    assert manifest["dataset"]["apple_available"] is False
    built_at = manifest["built_at"]
    run_dir = _latest_log_dir(runtime_root, "build-dataset")
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["event"] == "command.stage.start" and event["stage"] == "dataset.build" for event in events)

    # Unchanged inputs: nothing is rebuilt.
    assert main(["--root", str(workspace_root), "build-dataset", "--raw", str(raw_root)]) == 0
    assert json.loads((merged / "dataset.json").read_text(encoding="utf-8"))["built_at"] == built_at

    # A new raw file makes analyses refuse the old dataset until it is rebuilt.
    (raw_root / "carbs.csv").write_text("timestamp,carb_grams\n2025-05-24 08:00:00,30\n", encoding="utf-8")
    from bayesian_t1dm.dataset import StaleDatasetError, load_merged_dataset

    try:
        load_merged_dataset(raw_root)
    except StaleDatasetError as exc:
        assert "1 added" in str(exc)
    else:
        raise AssertionError("expected a stale-dataset error after adding a raw file")

def test_a_failing_command_is_logged(tmp_path, tandem_fixture_dir, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(tmp_path / "cloud"))
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.setattr(cli, "build_merged_dataset", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    try:
        main(["--root", str(workspace_root), "build-dataset", "--raw", str(tandem_fixture_dir)])
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected build-dataset to fail")

    run_dir = _latest_log_dir(runtime_root, "build-dataset")
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["event"] == "command.error" and event["fields"]["error_type"] == "RuntimeError" for event in events)
    assert any(event["event"] == "command.complete" and event["status"] == "failed" for event in events)


def test_refresh_runs_collect_build_and_readout_in_order_and_stops_at_a_failure(tmp_path, monkeypatch):
    from bayesian_t1dm import readout_report

    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(tmp_path / "cloud"))
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(tmp_path / "runtime"))
    repo = tmp_path / "repo"
    repo.mkdir()
    calls: list[str] = []
    monkeypatch.setattr(cli, "_collect_or_backfill", lambda args, *rest: calls.append(args.command) or 0)
    monkeypatch.setattr(cli, "build_merged_dataset", lambda *args, **kwargs: calls.append("build-dataset"))
    monkeypatch.setattr(readout_report, "run", lambda *args, **kwargs: calls.append("readout") or 0)
    assert main(["--root", str(repo), "refresh"]) == 0
    assert calls == ["collect", "build-dataset", "readout"]

    calls.clear()
    assert main(["--root", str(repo), "refresh", "--no-collect"]) == 0
    assert calls == ["build-dataset", "readout"]

    calls.clear()
    monkeypatch.setattr(cli, "_collect_or_backfill", lambda args, *rest: calls.append(args.command) or 3)
    assert main(["--root", str(repo), "refresh"]) == 3
    assert calls == ["collect"], "a failed download stops the refresh before the dataset is rebuilt"
