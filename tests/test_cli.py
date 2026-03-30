from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

import bayesian_t1dm.cli as cli
from bayesian_t1dm.cli import main
from bayesian_t1dm.acquisition import ExportWindow, NormalizedWindowResult


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
            observed_last_timestamp=pd.Timestamp(window.start_date).isoformat(),
            observed_window_days=1,
            is_complete_window=True,
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
