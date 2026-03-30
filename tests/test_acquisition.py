from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

import bayesian_t1dm.acquisition as acquisition
from bayesian_t1dm.acquisition import (
    ExportWindow,
    NormalizedWindowResult,
    RawApiArtifact,
    TandemCredentials,
    TConnectSyncSourceClient,
    backfill_tandem_exports,
    collect_tandem_exports,
    normalize_tconnectsync_payloads,
)
from bayesian_t1dm.paths import ProjectPaths


class FakeTConnectSyncAdapter:
    def __init__(self, payloads: dict[str, object]) -> None:
        self.payloads = payloads
        self.calls: list[tuple[str, str | None]] = []

    def login(self, *args, **kwargs) -> None:
        self.calls.append(("login", kwargs.get("timezone") or kwargs.get("pump_serial")))

    def start_trace(self) -> None:
        self.calls.append(("trace_start", None))

    def stop_trace(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("trace", encoding="utf-8")
        return path

    def capture_screenshot(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("screenshot", encoding="utf-8")
        return path

    def capture_page_diagnostics(self, stem: str):
        return None

    def export_daily_timeline_window(self, window, download_dir, step_log=None):
        self.fetch_window_payloads(window=window, download_dir=download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)
        path = download_dir / f"{window.window_id}.csv"
        path.write_text("timestamp,glucose\n", encoding="utf-8")
        return NormalizedWindowResult(
            window_id=window.window_id,
            requested_start=window.start_date.isoformat(),
            requested_end=window.end_date.isoformat(),
            endpoint_family="tconnectsync",
            source_label="fake",
            is_complete_window=True,
            manifest_path=path,
            payload_hash="fake",
        )

    def fetch_window_payloads(self, *, window=None, download_dir=None, timezone=None, pump_serial=None):
        self.calls.append(("fetch", getattr(window, "window_id", None)))
        return self.payloads


def _workspace(tmp_path: Path) -> ProjectPaths:
    root = tmp_path / "bayesian-t1dm"
    root.mkdir()
    runtime_root = tmp_path / "runtime"
    cloud_root = tmp_path / "cloud"
    return ProjectPaths.from_root(root, runtime_root=runtime_root, cloud_root=cloud_root).ensure()


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


def test_tconnectsync_payloads_build_canonical_carb_table_from_explicit_meals(tmp_path):
    window = ExportWindow(date(2024, 6, 1), date(2024, 6, 1))
    payloads = {
        "cgm": [{"eventDateTime": "2024-06-01T00:00:00", "egv_estimatedGlucoseValue": 120}],
        "bolus": [{"completionDateTime": "2024-06-01T08:00:00", "actualTotalBolusRequested": 4.0, "carbSize": 45}],
        "meal": [{"eventDateTime": "2024-06-01T08:00:00", "carbSize": 45}],
        "basal": [],
        "activity": [],
    }

    ingested, summary = normalize_tconnectsync_payloads(payloads, window=window, endpoint_family="tconnectsync", timezone="America/New_York")

    assert len(ingested.carbs) == 1
    assert ingested.carbs.iloc[0]["carb_grams"] == 45.0
    assert ingested.carbs.iloc[0]["source_label"] == "meal"
    assert summary["row_counts"]["carbs"] == 1


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
    client = FakeTConnectSyncAdapter({"cgm": [], "bolus": [], "basal": [], "activity": []})
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
    export_calls_after_first = [call for call in client.calls if call[0] == "fetch"]
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
    export_calls_after_second = [call for call in client.calls if call[0] == "fetch"]
    assert len(records_second) == 2
    assert len(export_calls_after_second) == 2
