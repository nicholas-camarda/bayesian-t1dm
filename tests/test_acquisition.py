from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import bayesian_t1dm.acquisition as acquisition
from bayesian_t1dm.acquisition import AcquisitionError, AcquisitionRecord, ExportWindow, StepLogger, TandemCredentials, backfill_tandem_exports, collect_tandem_exports, PlaywrightTandemSourceClient
from bayesian_t1dm.tandem_browser import LocatorSpec, TandemPageMap
from bayesian_t1dm.paths import ProjectPaths


class FakeTandemClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def login(self, credentials, step_log=None) -> None:
        self.calls.append(("login", credentials.email))

    def export_daily_timeline_window(self, window, download_dir, step_log=None):
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

    def capture_page_diagnostics(self, stem: str):
        return None


class FakePage:
    def __init__(self) -> None:
        self.waits: list[int] = []

    def goto(self, url: str, wait_until: str | None = None) -> None:
        return None

    def wait_for_timeout(self, ms: int) -> None:
        self.waits.append(ms)


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


def test_wait_for_login_controls_waits_until_inventory_ready(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    client = PlaywrightTandemSourceClient(workspace, page_map_path=workspace.cloud_archive / "tandem_page_map.json")
    client._page = FakePage()
    inventories = [
        [],
        [],
        [
            {"tag": "input", "type": "email", "autocomplete": "username", "visible": True},
            {"tag": "input", "type": "password", "autocomplete": "current-password", "visible": True},
            {"tag": "button", "type": "submit", "text": "Sign in", "visible": True},
        ],
    ]

    def fake_inventory(_page):
        return inventories.pop(0)

    monkeypatch.setattr(acquisition, "capture_control_inventory", fake_inventory)
    step_log = StepLogger(workspace.runtime_logs / "discover.log")

    result = client._wait_for_login_controls(step_log=step_log)

    assert len(client._page.waits) == 2
    assert result[-1]["type"] == "submit"
    assert "browser.discover.login_ready" in (workspace.runtime_logs / "discover.log").read_text(encoding="utf-8")


def test_wait_for_login_controls_times_out_with_hydration_error(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    client = PlaywrightTandemSourceClient(workspace, page_map_path=workspace.cloud_archive / "tandem_page_map.json", timeout_ms=1)
    client._page = FakePage()
    diagnostics: list[str] = []

    monkeypatch.setattr(acquisition, "capture_control_inventory", lambda _page: [])
    monkeypatch.setattr(client, "capture_page_diagnostics", lambda stem: diagnostics.append(stem))
    step_log = StepLogger(workspace.runtime_logs / "discover.log")

    with pytest.raises(AcquisitionError, match="hydration"):
        client._wait_for_login_controls(step_log=step_log)

    assert diagnostics == ["discover-login-timeout"]
    assert "browser.discover.login_wait_timeout" in (workspace.runtime_logs / "discover.log").read_text(encoding="utf-8")


def test_discover_login_controls_use_weak_attributes():
    snapshot = {"role": "document", "children": []}
    inventory = [
        {"tag": "input", "type": "email", "autocomplete": "username", "visible": True},
        {"tag": "input", "type": "password", "autocomplete": "current-password", "visible": True},
        {"tag": "button", "type": "submit", "text": "Next", "visible": True},
    ]

    email, password, submit = acquisition.discover_login_controls_from_controls(
        accessibility_snapshot=snapshot,
        control_inventory=inventory,
    )

    assert email.kind == "css"
    assert email.selector == 'input[type="email"]'
    assert password.kind == "css"
    assert password.selector == 'input[type="password"]'
    assert submit.kind == "css"
    assert submit.selector == 'button[type="submit"]'


def test_discover_accepts_partial_page_map_and_merges_bootstrap(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    page_map_path = workspace.cloud_archive / "tandem_page_map.json"
    partial = TandemPageMap(
        login_url="https://source.tandemdiabetes.com/",
        login_email=LocatorSpec(kind="css", selector='input[type="email"]'),
        login_password=LocatorSpec(kind="css", selector='input[type="password"]'),
        login_submit=LocatorSpec(kind="css", selector='input[type="submit"]'),
        source="bootstrap",
    )
    partial.save(page_map_path, validate=False)

    client = PlaywrightTandemSourceClient(workspace, page_map_path=page_map_path)
    client._page = FakePage()
    client.load_page_map(page_map_path, validate=False)

    dummy_diag = acquisition.PageDiagnostics(
        html_path=workspace.runtime_logs / "dummy.html",
        accessibility_path=workspace.runtime_logs / "dummy.a11y.json",
        inventory_path=workspace.runtime_logs / "dummy.controls.json",
        url_path=workspace.runtime_logs / "dummy.url.txt",
        frames_path=workspace.runtime_logs / "dummy.frames.json",
        screenshot_path=workspace.runtime_logs / "dummy.png",
    )
    for path in [dummy_diag.html_path, dummy_diag.accessibility_path, dummy_diag.inventory_path, dummy_diag.url_path, dummy_diag.frames_path, dummy_diag.screenshot_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")

    monkeypatch.setattr(client, "_wait_for_login_controls", lambda step_log=None: [{"tag": "input", "type": "email", "visible": True}])
    monkeypatch.setattr(client, "capture_page_diagnostics", lambda stem: dummy_diag)
    monkeypatch.setattr(acquisition, "capture_control_inventory", lambda page: [{"tag": "button", "type": "button", "text": "Export CSV", "visible": True}])
    monkeypatch.setattr(client, "_login_using_page_map", lambda credentials, page_map, step_log=None: None)
    monkeypatch.setattr(client, "_discover_daily_timeline_control", lambda page: (LocatorSpec(kind="css", selector="#daily"), "https://source.tandemdiabetes.com/daily"))
    monkeypatch.setattr(client, "_open_daily_timeline", lambda page_map: None)
    monkeypatch.setattr(
        client,
        "_discover_timeline_controls",
        lambda snapshot, inventory: (
            LocatorSpec(kind="css", selector="#start"),
            LocatorSpec(kind="css", selector="#end"),
            LocatorSpec(kind="css", selector="#export"),
        ),
    )

    result = client.discover_page_map(TandemCredentials(email="me@example.com", password="secret"), step_log=StepLogger(workspace.runtime_logs / "discover.log"))

    assert result.is_complete()
    assert result.login_email.selector == 'input[type="email"]'
    assert result.start_date.selector == "#start"
    assert page_map_path.exists()
    assert TandemPageMap.load(page_map_path).is_complete()
