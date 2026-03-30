from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import bayesian_t1dm.acquisition as acquisition
from bayesian_t1dm.acquisition import AcquisitionError, AcquisitionRecord, ExportArtifact, ExportWindow, NormalizedWindowResult, RawApiArtifact, StepLogger, TandemCredentials, TConnectSyncSourceClient, backfill_tandem_exports, collect_tandem_exports, PlaywrightTandemSourceClient, normalize_tconnectsync_payloads
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


class FakeTConnectSyncAdapter:
    def __init__(self, payloads: dict[str, object]) -> None:
        self.payloads = payloads
        self.calls: list[tuple[str, str | None]] = []

    def login(self, *args, **kwargs) -> None:
        self.calls.append(("login", kwargs.get("timezone") or kwargs.get("pump_serial")))

    def fetch_window_payloads(self, *, window=None, download_dir=None, timezone=None, pump_serial=None):
        self.calls.append(("fetch", getattr(window, "window_id", None)))
        return self.payloads


class FakePage:
    def __init__(self) -> None:
        self.waits: list[int] = []
        self.listeners: dict[str, list[object]] = {"response": [], "download": []}
        self.modal_opened = False
        self.mode = "download"
        self.launcher_clicks = 0
        self.confirm_clicks = 0

    def goto(self, url: str, wait_until: str | None = None) -> None:
        return None

    def wait_for_timeout(self, ms: int) -> None:
        self.waits.append(ms)

    def on(self, event: str, handler) -> None:
        self.listeners.setdefault(event, []).append(handler)

    def remove_listener(self, event: str, handler) -> None:
        if event in self.listeners and handler in self.listeners[event]:
            self.listeners[event].remove(handler)

    def emit(self, event: str, payload) -> None:
        for handler in list(self.listeners.get(event, [])):
            handler(payload)

    @property
    def main_frame(self):
        return FakeFrame(self)

    @property
    def frames(self):
        return [self.main_frame]


class FakeFrame:
    def __init__(self, page: FakePage) -> None:
        self.page = page
        self.name = ""
        self.url = "https://source.tandemdiabetes.com/reports/timeline"

    def get_by_role(self, role: str, name: str | None = None, exact: bool = True):
        return FakeLocator(self.page, role=role, name=name)

    def locator(self, selector: str):
        return FakeLocator(self.page, selector=selector)


class FakeLocator:
    def __init__(self, page: FakePage, role: str | None = None, name: str | None = None, selector: str | None = None) -> None:
        self.page = page
        self.role = role
        self.name = name
        self.selector = selector

    @property
    def first(self):
        return self

    def click(self) -> None:
        if self.name == "Export CSV":
            self.page.launcher_clicks += 1
            self.page.modal_opened = True
            return
        if self.name == "Export":
            self.page.confirm_clicks += 1
            if self.page.mode == "download":
                self.page.emit(
                    "response",
                    FakeResponse(
                        url="https://source.tandemdiabetes.com/api/reports/export",
                        content_type="text/csv; charset=utf-8",
                        body=b"timestamp,glucose\n2024-01-01,100\n",
                    ),
                )
                self.page.emit("download", FakeDownload("tandem-export.csv", b"timestamp,glucose\n2024-01-01,100\n"))
            elif self.page.mode == "response-csv":
                self.page.emit(
                    "response",
                    FakeResponse(
                        url="https://source.tandemdiabetes.com/api/reports/export",
                        content_type="text/csv; charset=utf-8",
                        body=b"timestamp,glucose\n2024-01-01,100\n",
                    ),
                )
            elif self.page.mode == "response-json":
                self.page.emit(
                    "response",
                    FakeResponse(
                        url="https://source.tandemdiabetes.com/api/reports/export",
                        content_type="application/json; charset=utf-8",
                        body=b'{"rows":[{"timestamp":"2024-01-01","glucose":100}]}',
                    ),
                )
            return
        if self.selector == "#export":
            self.page.confirm_clicks += 1
            return


class FakeDownload:
    def __init__(self, suggested_filename: str, body: bytes) -> None:
        self.suggested_filename = suggested_filename
        self.body = body

    def save_as(self, path: str) -> None:
        Path(path).write_bytes(self.body)


class FakeRequest:
    def __init__(self, method: str = "GET", resource_type: str = "xhr") -> None:
        self.method = method
        self.resource_type = resource_type


class FakeResponse:
    def __init__(self, *, url: str, content_type: str, body: bytes, method: str = "POST", status: int = 200, resource_type: str = "xhr") -> None:
        self.url = url
        self._body = body
        self.status = status
        self.headers = {"content-type": content_type, "content-length": str(len(body))}
        self.request = FakeRequest(method=method, resource_type=resource_type)

    def body(self) -> bytes:
        return self._body


def _workspace(tmp_path: Path) -> ProjectPaths:
    root = tmp_path / "bayesian-t1dm"
    root.mkdir()
    runtime_root = tmp_path / "runtime"
    cloud_root = tmp_path / "cloud"
    return ProjectPaths.from_root(root, runtime_root=runtime_root, cloud_root=cloud_root).ensure()


def _export_dialog_snapshot() -> dict[str, object]:
    return {
        "role": "document",
        "children": [
            {
                "role": "dialog",
                "name": "Export to CSV",
                "children": [
                    {"role": "button", "name": "Cancel"},
                    {"role": "button", "name": "Export"},
                ],
            }
        ],
    }


def _export_dialog_inventory(*, include_launcher: bool = True, include_confirm: bool = True) -> list[dict[str, object]]:
    inventory = []
    if include_launcher:
        inventory.append(
            {"tag": "button", "role": None, "id": "export-launcher", "name": None, "type": "button", "aria_label": None, "placeholder": None, "autocomplete": None, "text": "Export CSV", "title": None, "href": None, "data_testid": None, "visible": True},
        )
    inventory.extend([
        {"tag": "div", "role": "dialog", "id": "export-dialog", "name": None, "type": None, "aria_label": None, "placeholder": None, "autocomplete": None, "text": "Export to CSV Cancel", "title": None, "href": None, "data_testid": None, "visible": True},
        {"tag": "button", "role": None, "id": "cancel", "name": None, "type": "button", "aria_label": None, "placeholder": None, "autocomplete": None, "text": "Cancel", "title": None, "href": None, "data_testid": None, "visible": True},
    ])
    if include_confirm:
        inventory.append(
            {"tag": "button", "role": None, "id": "confirm", "name": None, "type": "button", "aria_label": None, "placeholder": None, "autocomplete": None, "text": "Export", "title": None, "href": None, "data_testid": None, "visible": True},
        )
    return inventory


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


def test_tconnectsync_client_archives_raw_payloads_and_normalizes_outputs(tmp_path):
    workspace = _workspace(tmp_path)
    window = ExportWindow(date(2024, 5, 1), date(2024, 5, 3))
    payloads = {
        "cgm": [
            {"eventDateTime": "2024-05-01T00:00:00", "egv_estimatedGlucoseValue": 110},
            {"eventDateTime": "2024-05-01T00:05:00", "egv_estimatedGlucoseValue": 111},
            {"eventDateTime": "2024-05-01T00:10:00", "egv_estimatedGlucoseValue": 112},
        ],
        "bolus": [
            {"completionDateTime": "2024-05-01T08:05:00", "actualTotalBolusRequested": 4.0, "carbSize": 45},
        ],
        "basal": [
            {"startDateTime": "2024-05-01T00:00:00", "endDateTime": "2024-05-02T00:00:00", "basalRate": 0.85},
        ],
        "activity": [
            {"startDateTime": "2024-05-01T12:00:00", "steps": 1000},
        ],
    }
    adapter = FakeTConnectSyncAdapter(payloads)
    client = TConnectSyncSourceClient(workspace, adapter=adapter, timezone="America/New_York", pump_serial="123456")

    result = client.export_daily_timeline_window(window, workspace.runtime_downloads)
    record = collect_tandem_exports(
        client,
        [window],
        workspace,
        TandemCredentials(email="me@example.com", password="secret", timezone="America/New_York", pump_serial="123456"),
        resume=False,
        manifest_path=workspace.cloud_raw / "tandem_export_manifest.csv",
        report_path=workspace.cloud_output / "tandem_acquisition_summary.md",
    )[0]

    assert isinstance(result, NormalizedWindowResult)
    assert isinstance(result.raw_artifacts[0], RawApiArtifact)
    assert result.is_complete_window
    assert result.row_counts["cgm"] == 3
    assert result.row_counts["bolus"] == 1
    assert result.row_counts["basal"] == 1
    assert result.row_counts["activity"] == 1
    assert result.manifest_path is not None and result.manifest_path.exists()
    assert (workspace.cloud_raw / "tconnectsync" / window.window_id / "raw" / "cgm.json").exists()
    assert (workspace.cloud_raw / "tconnectsync" / window.window_id / "normalized" / "cgm.csv").exists()
    assert record.source_type == "api"
    assert record.is_complete_window
    assert json.loads(record.raw_artifact_paths_json)
    assert json.loads(record.normalized_paths_json)["cgm"].endswith("cgm.csv")
    assert json.loads(record.completeness_json)["is_complete_window"]
    assert adapter.calls[0][0] == "fetch"
    assert adapter.calls[0][1] == window.window_id


def test_tconnectsync_payload_hash_is_deterministic(tmp_path):
    window = ExportWindow(date(2024, 6, 1), date(2024, 6, 1))
    payloads = {
        "cgm": [{"eventDateTime": "2024-06-01T00:00:00", "egv_estimatedGlucoseValue": 120}],
        "bolus": [{"completionDateTime": "2024-06-01T08:00:00", "actualTotalBolusRequested": 2.0}],
        "basal": [{"startDateTime": "2024-06-01T00:00:00", "endDateTime": "2024-06-02T00:00:00", "basalRate": 0.75}],
        "activity": [],
    }
    ingested_a, summary_a = normalize_tconnectsync_payloads(payloads, window=window, endpoint_family="tconnectsync", timezone="America/New_York")
    ingested_b, summary_b = normalize_tconnectsync_payloads(payloads, window=window, endpoint_family="tconnectsync", timezone="America/New_York")

    assert summary_a["payload_hash"] == summary_b["payload_hash"]
    assert ingested_a.manifest.equals(ingested_b.manifest)


def test_load_tandem_exports_reads_normalized_paths_from_manifest(tmp_path):
    raw_dir = tmp_path / "raw"
    normalized_dir = tmp_path / "normalized"
    raw_dir.mkdir()
    normalized_dir.mkdir()

    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="5min"),
            "glucose": [100, 101],
            "source_file": ["tconnectsync::window::tconnectsync"] * 2,
            "source_label": ["tconnectsync"] * 2,
        }
    ).to_csv(normalized_dir / "cgm.csv", index=False)
    pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01 08:00:00")],
            "bolus_units": [2.5],
            "source_file": ["tconnectsync::window::tconnectsync"],
            "source_label": ["tconnectsync"],
        }
    ).to_csv(normalized_dir / "bolus.csv", index=False)
    pd.DataFrame(
        {
            "start_timestamp": [pd.Timestamp("2024-01-01 00:00:00")],
            "end_timestamp": [pd.Timestamp("2024-01-02 00:00:00")],
            "basal_units_per_hour": [0.7],
            "source_file": ["tconnectsync::window::tconnectsync"],
            "source_label": ["tconnectsync"],
        }
    ).to_csv(normalized_dir / "basal.csv", index=False)
    pd.DataFrame(
        {
            "normalized_path": [
                str(normalized_dir / "cgm.csv"),
                str(normalized_dir / "bolus.csv"),
                str(normalized_dir / "basal.csv"),
            ]
        }
    ).to_csv(raw_dir / "window_manifest.csv", index=False)

    data = acquisition.load_tandem_exports(raw_dir)

    assert data.cgm.shape[0] == 2
    assert data.bolus.shape[0] == 1
    assert data.basal.shape[0] == 1


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
    monkeypatch.setattr(client, "_discover_report_timeline_tab", lambda page: (LocatorSpec(kind="css", selector="#timeline"), "https://source.tandemdiabetes.com/reports/timeline"))
    monkeypatch.setattr(client, "_open_daily_timeline", lambda page_map: None)
    monkeypatch.setattr(
        client,
        "_discover_timeline_controls",
        lambda snapshot, inventory: (
            LocatorSpec(kind="css", selector="#range"),
            LocatorSpec(kind="css", selector="#select"),
            LocatorSpec(kind="css", selector="#export"),
        ),
    )

    result = client.discover_page_map(TandemCredentials(email="me@example.com", password="secret"), step_log=StepLogger(workspace.runtime_logs / "discover.log"))

    assert result.is_complete()
    assert result.login_email.selector == 'input[type="email"]'
    assert result.start_date.selector == "#range"
    assert result.end_date.selector == "#select"
    assert page_map_path.exists()
    assert TandemPageMap.load(page_map_path).is_complete()


def test_playwright_export_clicks_launcher_then_confirm_and_downloads_csv(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    client = PlaywrightTandemSourceClient(workspace, page_map_path=workspace.cloud_archive / "tandem_page_map.json")
    client._page = FakePage()
    client._page.mode = "download"
    client.page_map = TandemPageMap(
        login_url="https://source.tandemdiabetes.com/",
        export_csv_launcher=LocatorSpec(kind="role", role="button", name="Export CSV"),
    )
    monkeypatch.setattr(client, "_open_daily_timeline", lambda page_map: None)
    monkeypatch.setattr(client, "_set_custom_date_range", lambda window, page_map: None)
    monkeypatch.setattr(client, "ensure_page_map", lambda credentials=None, step_log=None, bootstrap=False: client.page_map)
    monkeypatch.setattr(acquisition, "capture_accessibility_snapshot", lambda page: _export_dialog_snapshot())
    monkeypatch.setattr(acquisition, "capture_control_inventory", lambda page: _export_dialog_inventory(include_launcher=False, include_confirm=True))
    monkeypatch.setattr(client, "capture_page_diagnostics", lambda stem: None)

    window = ExportWindow(date(2024, 1, 1), date(2024, 1, 30))
    exported = client.export_daily_timeline_window(window, workspace.runtime_downloads)

    assert exported.exists()
    assert client._page.launcher_clicks == 1
    assert client._page.confirm_clicks == 1
    assert exported.read_text(encoding="utf-8").startswith("timestamp,glucose")
    assert client.page_map.export_csv_confirm is not None


def test_playwright_export_writes_raw_response_artifact_for_api_payload(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    client = PlaywrightTandemSourceClient(workspace, page_map_path=workspace.cloud_archive / "tandem_page_map.json")
    client._page = FakePage()
    client._page.mode = "response-json"
    client.page_map = TandemPageMap(
        login_url="https://source.tandemdiabetes.com/",
        export_csv_launcher=LocatorSpec(kind="role", role="button", name="Export CSV"),
    )
    monkeypatch.setattr(client, "_open_daily_timeline", lambda page_map: None)
    monkeypatch.setattr(client, "_set_custom_date_range", lambda window, page_map: None)
    monkeypatch.setattr(client, "ensure_page_map", lambda credentials=None, step_log=None, bootstrap=False: client.page_map)
    monkeypatch.setattr(acquisition, "capture_accessibility_snapshot", lambda page: _export_dialog_snapshot())
    monkeypatch.setattr(acquisition, "capture_control_inventory", lambda page: _export_dialog_inventory(include_launcher=False, include_confirm=True))
    monkeypatch.setattr(client, "capture_page_diagnostics", lambda stem: None)

    window = ExportWindow(date(2024, 2, 1), date(2024, 2, 29))
    with pytest.raises(AcquisitionError, match="non-CSV payload"):
        client.export_daily_timeline_window(window, workspace.runtime_downloads)

    body_path = workspace.runtime_downloads / f"{window.window_id}.export-response.json"
    meta_path = workspace.runtime_downloads / f"{window.window_id}.export-response.meta.json"
    assert body_path.exists()
    assert meta_path.exists()
    assert json.loads(meta_path.read_text(encoding="utf-8"))["content_type"].startswith("application/json")
    assert "rows" in body_path.read_text(encoding="utf-8")


def test_playwright_export_reports_incomplete_dialog_when_confirm_is_missing(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    client = PlaywrightTandemSourceClient(workspace, page_map_path=workspace.cloud_archive / "tandem_page_map.json")
    client._page = FakePage()
    client.page_map = TandemPageMap(
        login_url="https://source.tandemdiabetes.com/",
        export_csv_launcher=LocatorSpec(kind="role", role="button", name="Export CSV"),
    )
    monkeypatch.setattr(client, "_open_daily_timeline", lambda page_map: None)
    monkeypatch.setattr(client, "_set_custom_date_range", lambda window, page_map: None)
    monkeypatch.setattr(client, "ensure_page_map", lambda credentials=None, step_log=None, bootstrap=False: client.page_map)
    monkeypatch.setattr(acquisition, "capture_accessibility_snapshot", lambda page: {"role": "document", "children": [{"role": "dialog", "name": "Export to CSV", "children": [{"role": "button", "name": "Cancel"}]}]})
    monkeypatch.setattr(acquisition, "capture_control_inventory", lambda page: _export_dialog_inventory(include_launcher=False, include_confirm=False))
    monkeypatch.setattr(client, "capture_page_diagnostics", lambda stem: None)

    window = ExportWindow(date(2024, 3, 1), date(2024, 3, 29))
    with pytest.raises(AcquisitionError, match="Export dialog was incomplete"):
        client.export_daily_timeline_window(window, workspace.runtime_downloads)

    assert client._page.launcher_clicks == 1
    assert client._page.confirm_clicks == 0


def test_playwright_export_accepts_legacy_export_csv_page_map(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    client = PlaywrightTandemSourceClient(workspace, page_map_path=workspace.cloud_archive / "tandem_page_map.json")
    client._page = FakePage()
    client._page.mode = "download"
    legacy_map = TandemPageMap.from_dict(
        {
            "login_url": "https://source.tandemdiabetes.com/",
            "daily_timeline_url": "https://source.tandemdiabetes.com/reports/timeline",
            "login_email": {"kind": "css", "selector": "#email"},
            "login_password": {"kind": "css", "selector": "#password"},
            "login_submit": {"kind": "css", "selector": "#submit"},
            "daily_timeline_nav": {"kind": "role", "role": "link", "name": "Daily Timeline"},
            "start_date": {"kind": "role", "role": "combobox", "name": "Custom"},
            "end_date": {"kind": "role", "role": "button", "name": "Select"},
            "export_csv": {"kind": "role", "role": "button", "name": "Export CSV"},
        }
    )
    client.page_map = legacy_map
    monkeypatch.setattr(client, "_open_daily_timeline", lambda page_map: None)
    monkeypatch.setattr(client, "_set_custom_date_range", lambda window, page_map: None)
    monkeypatch.setattr(client, "ensure_page_map", lambda credentials=None, step_log=None, bootstrap=False: client.page_map)
    monkeypatch.setattr(acquisition, "capture_accessibility_snapshot", lambda page: _export_dialog_snapshot())
    monkeypatch.setattr(acquisition, "capture_control_inventory", lambda page: _export_dialog_inventory(include_launcher=False, include_confirm=True))
    monkeypatch.setattr(client, "capture_page_diagnostics", lambda stem: None)

    window = ExportWindow(date(2024, 4, 1), date(2024, 4, 30))
    exported = client.export_daily_timeline_window(window, workspace.runtime_downloads)

    assert exported.exists()
    assert client._page.launcher_clicks == 1
    assert client._page.confirm_clicks == 1
    assert client.page_map.export_csv_launcher is not None
    assert client.page_map.export_csv_confirm is not None
