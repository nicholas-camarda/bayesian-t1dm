from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from bayesian_t1dm.acquisition import AcquisitionRecord, ExportWindow, TandemCredentials, backfill_tandem_exports, collect_tandem_exports
from bayesian_t1dm.paths import ProjectPaths


class FakeTandemClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def login(self, credentials, step_log=None, *, allow_manual_login: bool = True) -> None:
        self.calls.append(("login", credentials.email))

    def export_daily_timeline_window(self, window, download_dir, step_log=None, *, allow_manual_export: bool = True):
        self.calls.append(("export", window.window_id))
        download_dir.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range(window.start_date, window.end_date, freq="D"),
                "glucose": [100 + index for index in range((window.end_date - window.start_date).days + 1)],
            }
        )
        path = download_dir / f"{window.window_id}.csv"
        frame.to_csv(path, index=False)
        return path

    def capture_screenshot(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("screenshot", encoding="utf-8")
        return path

    def start_trace(self) -> None:
        self.calls.append(("trace_start", None))

    def stop_trace(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("trace", encoding="utf-8")
        return path


def _workspace(tmp_path: Path) -> ProjectPaths:
    root = tmp_path / "bayesian-t1dm"
    root.mkdir()
    runtime_root = tmp_path / "runtime"
    cloud_root = tmp_path / "cloud"
    return ProjectPaths.from_root(root, runtime_root=runtime_root, cloud_root=cloud_root).ensure()


def test_collect_tandem_exports_writes_cloud_manifest_and_artifacts(tmp_path):
    workspace = _workspace(tmp_path)
    client = FakeTandemClient()
    credentials = TandemCredentials(email="me@example.com", password="secret")
    window = ExportWindow(date(2024, 1, 1), date(2024, 1, 30))

    records = collect_tandem_exports(
        client,
        [window],
        workspace,
        credentials,
        resume=False,
        manifest_path=workspace.cloud_raw / "tandem_export_manifest.csv",
        report_path=workspace.cloud_output / "tandem_acquisition_summary.md",
    )

    assert client.calls[:2] == [("login", "me@example.com"), ("trace_start", None)]
    assert len(records) == 1
    assert isinstance(records[0], AcquisitionRecord)
    assert records[0].status == "complete"
    assert records[0].is_complete_window
    assert (workspace.cloud_raw / "tandem_export_manifest.csv").exists()
    assert (workspace.cloud_output / "tandem_acquisition_summary.md").exists()
    assert Path(records[0].cloud_file).exists()
    assert Path(records[0].trace_path).exists()
    assert Path(records[0].screenshot_path).exists()


def test_backfill_skips_completed_windows(tmp_path):
    workspace = _workspace(tmp_path)
    client = FakeTandemClient()
    credentials = TandemCredentials(email="me@example.com", password="secret")

    records_first = backfill_tandem_exports(
        client,
        start_date="2024-01-01",
        end_date="2024-02-29",
        workspace=workspace,
        credentials=credentials,
        window_days=30,
        direction="forward",
        resume=True,
        manifest_path=workspace.cloud_raw / "tandem_export_manifest.csv",
        report_path=workspace.cloud_output / "tandem_acquisition_summary.md",
    )
    export_calls_after_first = [call for call in client.calls if call[0] == "export"]
    assert len(records_first) == 2
    assert len(export_calls_after_first) == 2

    records_second = backfill_tandem_exports(
        client,
        start_date="2024-01-01",
        end_date="2024-02-29",
        workspace=workspace,
        credentials=credentials,
        window_days=30,
        direction="forward",
        resume=True,
        manifest_path=workspace.cloud_raw / "tandem_export_manifest.csv",
        report_path=workspace.cloud_output / "tandem_acquisition_summary.md",
    )
    export_calls_after_second = [call for call in client.calls if call[0] == "export"]
    assert len(records_second) == 2
    assert len(export_calls_after_second) == 2
