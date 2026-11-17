from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

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


class _FakeTandemSourceEvent:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def todict(self) -> dict[str, object]:
        return dict(self.payload)


class _FakeTandemSourceMultiDeviceAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, str, object]] = []
        self.metadata = [
            {
                "serialNumber": "90723647",
                "tconnectDeviceId": "723357",
                "minDateWithEvents": "2023-01-01T00:00:00",
                "maxDateWithEvents": "2025-08-27T11:54:42",
                "lastUpload": {"lastUploadedAt": "2025-08-27T16:04:01"},
            },
            {
                "serialNumber": "1453346",
                "tconnectDeviceId": "1032256",
                "minDateWithEvents": "2020-01-15T00:00:00",
                "maxDateWithEvents": "2026-04-22T14:22:38",
                "lastUpload": {"lastUploadedAt": "2026-04-22T18:23:18"},
            },
        ]
        self.events = {
            "723357": [
                _FakeTandemSourceEvent(
                    {
                        "id": 279,
                        "name": "LID_BASAL_DELIVERY",
                        "eventTimestamp": "2025-07-10T00:00:00-04:00",
                        "commandedRate": 700,
                    }
                ),
                _FakeTandemSourceEvent(
                    {
                        "id": 280,
                        "name": "LID_BOLUS_DELIVERY",
                        "eventTimestamp": "2025-07-10T08:00:00-04:00",
                        "deliveredTotal": 2500,
                    }
                ),
            ],
            "1032256": [],
        }

    def pump_event_metadata(self):
        return self.metadata

    def pump_events_raw(self, device_id, min_date=None, max_date=None, event_ids_filter=None):
        self.calls.append(("raw", str(device_id), str(min_date), str(max_date), event_ids_filter))
        return f"raw-{device_id}-{min_date}-{max_date}"

    def pump_events(self, device_id, min_date=None, max_date=None, fetch_all_event_types=False):
        self.calls.append(("events", str(device_id), str(min_date), str(max_date), fetch_all_event_types))
        return self.events.get(str(device_id), [])


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

    result = client.export_daily_timeline_window(window, workspace.runtime / "downloads")
    record = collect_tandem_exports(
        client,
        [window],
        workspace,
        TandemCredentials(email="me@example.com", password="secret", timezone="America/New_York", pump_serial="123456"),
        resume=False,
        strict=False,
        manifest_path=workspace.cloud_raw / "tandem_export_manifest.csv",
        report_path=workspace.reports / "tandem_acquisition_summary.md",
    )[0]

    assert isinstance(result, NormalizedWindowResult)
    assert isinstance(result.raw_artifacts[0], RawApiArtifact)
    assert not result.is_complete_window
    assert result.row_counts["cgm"] == 3
    assert result.row_counts["bolus"] == 1
    assert result.row_counts["basal"] == 1
    assert result.row_counts["activity"] == 1
    assert result.manifest_path is not None and result.manifest_path.exists()
    assert (workspace.cloud_raw / "tconnectsync" / window.window_id / "raw" / "cgm.json").exists()
    assert (workspace.cloud_raw / "tconnectsync" / window.window_id / "normalized" / "cgm.csv").exists()
    assert record.source_type == "api"
    assert not record.is_complete_window
    assert json.loads(record.raw_artifact_paths_json)
    assert json.loads(record.normalized_paths_json)["cgm"].endswith("cgm.csv")
    completeness = json.loads(record.completeness_json)
    assert completeness["is_complete_window"] is False
    assert "ends_early" in completeness["completeness_reasons"]
    cgm_text = (workspace.cloud_raw / "tconnectsync" / window.window_id / "normalized" / "cgm.csv").read_text(encoding="utf-8")
    assert "+00:00" not in cgm_text
    assert "-04:00" not in cgm_text
    assert adapter.calls[0][0] == "fetch"
    assert adapter.calls[0][1] == window.window_id


def test_tconnectsync_client_queries_all_window_overlapping_devices(tmp_path):
    workspace = _workspace(tmp_path)
    window = ExportWindow(date(2025, 7, 5), date(2025, 8, 3))
    adapter = _FakeTandemSourceMultiDeviceAdapter()
    client = TConnectSyncSourceClient(workspace, adapter=adapter, timezone="America/New_York")

    result = client.export_daily_timeline_window(window, workspace.runtime / "downloads")

    assert result.row_counts["bolus"] == 1
    assert result.row_counts["basal"] == 1
    queried_devices = {call[1] for call in adapter.calls if call[0] == "events"}
    assert queried_devices == {"723357", "1032256"}

    normalized_root = workspace.cloud_raw / "tconnectsync" / window.window_id / "normalized"
    bolus = pd.read_csv(normalized_root / "bolus.csv")
    basal = pd.read_csv(normalized_root / "basal.csv")
    assert bolus["pump_serial"].astype(str).tolist() == ["90723647"]
    assert bolus["tconnect_device_id"].astype(str).tolist() == ["723357"]
    assert basal["pump_serial"].astype(str).tolist() == ["90723647"]
    assert basal["tconnect_device_id"].astype(str).tolist() == ["723357"]

    raw_root = workspace.cloud_raw / "tconnectsync" / window.window_id / "raw"
    metadata = json.loads((raw_root / "pump_event_metadata.json").read_text(encoding="utf-8"))
    assert {str(pump["tconnectDeviceId"]) for pump in metadata["selected_pumps"]} == {"723357", "1032256"}
    assert (raw_root / "pump_events_raw_90723647_723357.txt").exists()
    assert (raw_root / "pump_events_raw_1453346_1032256.txt").exists()


def test_normalize_tconnectsync_archive_rebuilds_window_and_is_idempotent(tmp_path):
    workspace = _workspace(tmp_path)
    window_dir = workspace.cloud_raw / "tconnectsync" / "2024-05-01__2024-05-03" / "raw"
    window_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "cgm": [
            {"eventDateTime": "2024-05-01T00:00:00-04:00", "egv_estimatedGlucoseValue": 110},
            {"eventDateTime": "2024-05-01 00:05:00", "egv_estimatedGlucoseValue": 111},
        ],
        "bolus": [{"completionDateTime": "2024-05-01T08:05:00-04:00", "actualTotalBolusRequested": 4.0, "carbSize": 45}],
        "basal": [{"startDateTime": "2024-05-01T00:00:00-04:00", "endDateTime": "2024-05-02 00:00:00", "basalRate": 0.85}],
        "activity": [],
    }
    for kind, payload in payloads.items():
        (window_dir / f"{kind}.json").write_text(json.dumps(payload), encoding="utf-8")

    results = acquisition.normalize_tconnectsync_archive(
        workspace,
        raw_root=workspace.cloud_raw / "tconnectsync",
        timezone="America/New_York",
    )

    assert len(results) == 1
    manifest = pd.read_csv(workspace.cloud_raw / "tconnectsync" / "2024-05-01__2024-05-03" / "window_manifest.csv")
    assert {"requested_start", "requested_end", "observed_first_timestamp", "observed_last_timestamp", "observed_duration_days", "coverage_fraction", "completeness_reasons"} <= set(manifest.columns)
    normalized_basal = pd.read_csv(workspace.cloud_raw / "tconnectsync" / "2024-05-01__2024-05-03" / "normalized" / "basal.csv")
    assert normalized_basal["end_timestamp"].astype(str).str.contains(r"[+-]\d{2}:\d{2}").sum() == 0

    second = acquisition.normalize_tconnectsync_archive(
        workspace,
        raw_root=workspace.cloud_raw / "tconnectsync",
        timezone="America/New_York",
    )
    assert second == []

    report_text = (workspace.reports / "normalize_raw_summary.md").read_text(encoding="utf-8")
    assert "skipped (existing normalized artifacts present)" in report_text


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


def test_normalize_tconnectsync_rejects_out_of_window_rows_and_records_timezone():
    window = ExportWindow(date(2024, 6, 1), date(2024, 6, 1))
    payloads = {
        "cgm": [
            {"eventDateTime": "2024-05-31T23:55:00-04:00", "egv_estimatedGlucoseValue": 99},
            {"eventDateTime": "2024-06-01T00:00:00-04:00", "egv_estimatedGlucoseValue": 100},
            {"eventDateTime": "2024-06-02T00:00:00-04:00", "egv_estimatedGlucoseValue": 101},
        ],
        "bolus": [{"completionDateTime": "2024-06-01T08:00:00-04:00", "actualTotalBolusRequested": 2.0}],
        "basal": [{"startDateTime": "2024-06-01T00:00:00-04:00", "endDateTime": "2024-06-02T00:00:00-04:00", "basalRate": 0.75}],
        "activity": [{"startDateTime": "2024-06-02T00:05:00-04:00", "steps": 100}],
    }

    ingested, summary = normalize_tconnectsync_payloads(payloads, window=window, endpoint_family="tconnectsync", timezone="America/New_York")

    assert summary["timezone"] == "America/New_York"
    assert summary["rejected_counts"]["cgm"] == 2
    assert summary["rejected_counts"]["activity"] == 1
    assert ingested.cgm["timestamp"].tolist() == [pd.Timestamp("2024-06-01 00:00:00")]
    assert ingested.activity.empty


def test_normalize_tconnectsync_cumulative_bolus_progress_is_incremental():
    window = ExportWindow(date(2024, 6, 1), date(2024, 6, 1))
    payloads = {
        "cgm": [{"eventDateTime": "2024-06-01T00:00:00", "egv_estimatedGlucoseValue": 120}],
        "bolus": [
            {"completionDateTime": "2024-06-01T08:00:00", "actualTotalBolusRequested": 1.0, "source_event_name": "bolus-a"},
            {"completionDateTime": "2024-06-01T08:05:00", "actualTotalBolusRequested": 2.0, "source_event_name": "bolus-a"},
            {"completionDateTime": "2024-06-01T08:10:00", "actualTotalBolusRequested": 3.0, "source_event_name": "bolus-a"},
        ],
        "basal": [],
        "activity": [],
    }

    ingested, _ = normalize_tconnectsync_payloads(payloads, window=window, endpoint_family="tconnectsync", timezone="America/New_York")

    assert ingested.bolus["bolus_units"].sum() == pytest.approx(3.0)
    assert ingested.bolus["raw_bolus_units"].max() == pytest.approx(3.0)


def test_normalize_tconnectsync_dedupes_repeated_bolus_lifecycle_records():
    window = ExportWindow(date(2024, 3, 24), date(2024, 3, 24))
    payloads = {
        "cgm": [{"eventDateTime": "2024-03-24T00:00:00", "egv_estimatedGlucoseValue": 120}],
        "bolus": [
            {
                "completionDateTime": "2024-03-24T15:05:00",
                "actualTotalBolusRequested": 0.0,
                "source_event_name": "LID_BOLUS_DELIVERY",
                "source_seq_num": 520852,
                "bolus_id": 1621,
                "pump_serial": "90723647",
                "tconnect_device_id": "723357",
            },
            {
                "completionDateTime": "2024-03-24T15:10:00",
                "actualTotalBolusRequested": 11.61,
                "source_event_name": "LID_BOLUS_DELIVERY",
                "source_seq_num": 520868,
                "bolus_id": 1621,
                "pump_serial": "90723647",
                "tconnect_device_id": "723357",
            },
            {
                "completionDateTime": "2024-03-24T15:05:00",
                "actualTotalBolusRequested": 0.0,
                "source_event_name": "LID_BOLUS_DELIVERY",
                "source_seq_num": 520852,
                "bolus_id": 1621,
                "pump_serial": "90723647",
                "tconnect_device_id": "723357",
            },
            {
                "completionDateTime": "2024-03-24T15:10:00",
                "actualTotalBolusRequested": 11.61,
                "source_event_name": "LID_BOLUS_DELIVERY",
                "source_seq_num": 520868,
                "bolus_id": 1621,
                "pump_serial": "90723647",
                "tconnect_device_id": "723357",
            },
            {
                "completionDateTime": "2024-03-24T18:00:00",
                "actualTotalBolusRequested": 0.0,
                "source_event_name": "LID_BOLUS_DELIVERY",
                "source_seq_num": 520900,
                "bolus_id": 1622,
                "pump_serial": "90723647",
                "tconnect_device_id": "723357",
            },
            {
                "completionDateTime": "2024-03-24T18:05:00",
                "actualTotalBolusRequested": 3.0,
                "source_event_name": "LID_BOLUS_DELIVERY",
                "source_seq_num": 520901,
                "bolus_id": 1622,
                "pump_serial": "90723647",
                "tconnect_device_id": "723357",
            },
        ],
        "basal": [],
        "activity": [],
    }

    ingested, _ = normalize_tconnectsync_payloads(payloads, window=window, endpoint_family="tconnectsync", timezone="America/New_York")

    assert ingested.bolus["bolus_units"].sum() == pytest.approx(14.61)
    assert ingested.bolus["bolus_units"].max() == pytest.approx(11.61)
    assert len(ingested.bolus) == 2


def test_normalize_tconnectsync_preserves_dst_fallback_offsets_for_review():
    window = ExportWindow(date(2024, 11, 3), date(2024, 11, 3))
    payloads = {
        "cgm": [
            {"eventDateTime": "2024-11-03T01:30:00-04:00", "egv_estimatedGlucoseValue": 110},
            {"eventDateTime": "2024-11-03T01:30:00-05:00", "egv_estimatedGlucoseValue": 111},
        ],
        "bolus": [],
        "basal": [],
        "activity": [],
    }

    ingested, _ = normalize_tconnectsync_payloads(payloads, window=window, endpoint_family="tconnectsync", timezone="America/New_York")

    assert len(ingested.cgm) == 2
    assert ingested.manifest.loc[ingested.manifest["kind"].eq("cgm"), "has_duplicates"].iloc[0]


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
        report_path=workspace.reports / "tandem_acquisition_summary.md",
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
        report_path=workspace.reports / "tandem_acquisition_summary.md",
    )
    export_calls_after_second = [call for call in client.calls if call[0] == "fetch"]
    assert len(records_second) == 2
    assert len(export_calls_after_second) == 2


# ---------------------------------------------------------------------------
# Carbs entered in the pump's bolus calculator (LID_BOLUS_REQUESTED_MSG1)
# ---------------------------------------------------------------------------


def _encode_pump_event(event_id: int, seconds_since_tandem_epoch: int, seq_num: int, fields: dict[int, tuple[str, int | float]]) -> bytes:
    """Build one 26-byte Tandem history-log event (big endian), as Tandem Source returns them."""
    import struct

    raw = bytearray(26)
    struct.pack_into(">H", raw, 0, event_id & 0x0FFF)
    struct.pack_into(">I", raw, 2, seconds_since_tandem_epoch)
    struct.pack_into(">I", raw, 6, seq_num)
    for offset, (fmt, value) in fields.items():
        struct.pack_into(fmt, raw, offset, value)
    return bytes(raw)


def _bolus_request(seconds: int, seq: int, *, bolus_id: int, carbs: int) -> bytes:
    return _encode_pump_event(64, seconds, seq, {11: (">B", 1), 12: (">H", bolus_id), 14: (">H", 110), 16: (">H", carbs), 18: (">f", 0.0), 22: (">I", 10000)})


def _bolus_delivery(seconds: int, seq: int, *, bolus_id: int, milliunits: int) -> bytes:
    return _encode_pump_event(280, seconds, seq, {12: (">H", bolus_id), 14: (">H", milliunits), 22: (">H", milliunits)})


def test_event_payloads_keep_carbs_only_for_delivered_boluses():
    from bayesian_t1dm.acquisition import _tconnectsync_event_payloads

    events = [
        _FakeTandemSourceEvent({"id": 64, "name": "LID_BOLUS_REQUESTED_MSG1", "eventTimestamp": "2025-07-10T07:58:00-04:00", "bolusid": 11, "carbamount": 45}),
        _FakeTandemSourceEvent({"id": 280, "name": "LID_BOLUS_DELIVERY", "eventTimestamp": "2025-07-10T08:00:00-04:00", "bolusid": 11, "deliveredTotal": 4500}),
        # Requested with carbs but cancelled before anything was delivered: not a meal record.
        _FakeTandemSourceEvent({"id": 64, "name": "LID_BOLUS_REQUESTED_MSG1", "eventTimestamp": "2025-07-10T12:00:00-04:00", "bolusid": 12, "carbamount": 60}),
        _FakeTandemSourceEvent({"id": 280, "name": "LID_BOLUS_DELIVERY", "eventTimestamp": "2025-07-10T12:01:00-04:00", "bolusid": 12, "deliveredTotal": 0}),
        # Correction-only bolus: delivered, no carbs.
        _FakeTandemSourceEvent({"id": 64, "name": "LID_BOLUS_REQUESTED_MSG1", "eventTimestamp": "2025-07-10T15:00:00-04:00", "bolusid": 13, "carbamount": 0}),
        _FakeTandemSourceEvent({"id": 280, "name": "LID_BOLUS_DELIVERY", "eventTimestamp": "2025-07-10T15:01:00-04:00", "bolusid": 13, "deliveredTotal": 1000}),
        # tconnectsync 3.x spells the fields in camelCase.
        _FakeTandemSourceEvent({"id": 64, "name": "LID_BOLUS_REQUESTED_MSG1", "eventTimestamp": "2025-07-10T18:00:00-04:00", "bolusId": 14, "carbAmount": 70}),
        _FakeTandemSourceEvent({"id": 280, "name": "LID_BOLUS_DELIVERY", "eventTimestamp": "2025-07-10T18:02:00-04:00", "bolusId": 14, "deliveredTotal": 7000}),
    ]

    payloads = _tconnectsync_event_payloads(events, pump_serial="90723647", tconnect_device_id="723357")

    carbs = sorted(payloads["carbs"], key=lambda record: record["eventDateTime"])
    assert [record["carbs"] for record in carbs] == [45.0, 70.0]
    assert [str(record["bolus_id"]) for record in carbs] == ["11", "14"]
    assert carbs[0]["eventDateTime"] == "2025-07-10T07:58:00-04:00"
    assert carbs[0]["pump_serial"] == "90723647"
    assert len(payloads["bolus"]) == 4


def test_normalized_window_turns_pump_carbs_into_carb_truth(tmp_path):
    from bayesian_t1dm.acquisition import ExportWindow, _tconnectsync_event_payloads, normalize_tconnectsync_payloads

    events = [
        _FakeTandemSourceEvent({"id": 64, "name": "LID_BOLUS_REQUESTED_MSG1", "eventTimestamp": "2025-07-10T07:58:00", "bolusid": 21, "carbamount": 52}),
        _FakeTandemSourceEvent({"id": 280, "name": "LID_BOLUS_DELIVERY", "eventTimestamp": "2025-07-10T08:00:00", "bolusid": 21, "deliveredTotal": 5200}),
    ]
    payloads = _tconnectsync_event_payloads(events)
    window = ExportWindow(start_date=pd.Timestamp("2025-07-10").date(), end_date=pd.Timestamp("2025-07-11").date())

    ingested, _ = normalize_tconnectsync_payloads(payloads, window=window, endpoint_family="tconnectsync")

    assert ingested.carbs["carb_grams"].tolist() == [52.0]
    # Floored to the 5-minute analysis grid, like every other normalized event.
    assert ingested.carbs["timestamp"].iloc[0] == pd.Timestamp("2025-07-10 07:55:00")
    assert ingested.bolus["bolus_units"].tolist() == [5.2]


def test_archived_raw_pump_events_yield_carbs_on_normalize(tmp_path):
    import base64

    from bayesian_t1dm.acquisition import _decode_archived_pump_event_carbs, _load_archived_window_payloads

    pytest.importorskip("tconnectsync.eventparser.generic")
    raw_dir = tmp_path / "2025-07-10__2025-07-11" / "raw"
    raw_dir.mkdir(parents=True)
    # Seconds since 2008-01-01 for 2025-07-10 07:58 and 08:00 pump-local time.
    request_time = int((pd.Timestamp("2025-07-10 07:58:00") - pd.Timestamp("2008-01-01")).total_seconds())
    stream = b"".join(
        [
            _bolus_request(request_time, 100, bolus_id=31, carbs=48),
            _bolus_delivery(request_time + 120, 101, bolus_id=31, milliunits=4800),
            _bolus_request(request_time + 3600, 102, bolus_id=32, carbs=30),  # never delivered
        ]
    )
    (raw_dir / "pump_events_raw_90723647_723357.txt").write_text(base64.b64encode(stream).decode() + "\n", encoding="utf-8")
    (raw_dir / "pump_events_raw_1453346_1032256.txt").write_text("not base64 at all !!\n", encoding="utf-8")
    (raw_dir / "bolus.json").write_text("[]\n", encoding="utf-8")

    carbs = _decode_archived_pump_event_carbs(raw_dir)
    assert [record["carbs"] for record in carbs] == [48.0]
    assert carbs[0]["pump_serial"] == "90723647"
    assert carbs[0]["tconnect_device_id"] == "723357"
    assert str(carbs[0]["eventDateTime"]).startswith("2025-07-10T07:58:00")

    payloads = _load_archived_window_payloads(raw_dir.parent)
    assert [record["carbs"] for record in payloads["carbs"]] == [48.0]


def test_archived_carbs_are_not_decoded_when_window_already_has_a_carb_family(tmp_path):
    import base64

    from bayesian_t1dm.acquisition import _load_archived_window_payloads

    raw_dir = tmp_path / "w" / "raw"
    raw_dir.mkdir(parents=True)
    request_time = int((pd.Timestamp("2025-07-10 07:58:00") - pd.Timestamp("2008-01-01")).total_seconds())
    stream = _bolus_request(request_time, 1, bolus_id=1, carbs=40) + _bolus_delivery(request_time + 60, 2, bolus_id=1, milliunits=4000)
    (raw_dir / "pump_events_raw_1_2.txt").write_text(base64.b64encode(stream).decode(), encoding="utf-8")
    (raw_dir / "carbs.json").write_text('[{"eventDateTime": "2025-07-10T07:58:00", "carbs": 40}]\n', encoding="utf-8")

    payloads = _load_archived_window_payloads(raw_dir.parent)

    assert payloads["carbs"] == [{"eventDateTime": "2025-07-10T07:58:00", "carbs": 40}]
