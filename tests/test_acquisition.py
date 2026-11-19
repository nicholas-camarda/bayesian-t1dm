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
    TandemCredentials,
    TConnectSyncSourceClient,
    backfill_tandem_exports,
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


def _pump_log_event(code: int, when: str, seq: int, device: str, **properties) -> dict:
    """One event shaped like the Tandem Source pump-logs response (tconnectsync 3.x)."""
    return {
        "deviceAssignmentId": device,
        "eventCode": code,
        "sequenceGroup": 0,
        "sequenceNumber": seq,
        "pumpDateTime": when,
        "eventProperties": properties,
        "estimatedDateTime": when + "Z",
    }


def _tandem_seconds(when: str) -> int:
    return int((pd.Timestamp(when) - pd.Timestamp("2008-01-01")).total_seconds())


def _cgm_event(when: str, seq: int, device: str, glucose: int, measured: str | None = None) -> dict:
    """A G7 reading received at ``when`` and measured by the sensor at ``measured`` (default: same time)."""
    return _pump_log_event(
        399, when, seq, device,
        algorithmState=0, cgmDataType=0, currentGlucoseDisplayValue=glucose, egvInfoBitmask=0,
        egvTimeStamp=_tandem_seconds(measured or when), glucoseValueStatus=0, interval=5, rate=0, reservedD15=0, rssi=0,
    )


def _basal_event(when: str, seq: int, device: str, milliunits_per_hour: int) -> dict:
    return _pump_log_event(
        279, when, seq, device,
        algorithmRate=milliunits_per_hour, commandedRate=milliunits_per_hour, commandedRateSource=0,
        profileBasalRate=milliunits_per_hour, reservedA2=0, spareA3=0, tempRate=0,
    )


def _bolus_request_event(when: str, seq: int, device: str, bolus_id: int, carbs: int) -> dict:
    return _pump_log_event(64, when, seq, device, bg=120, bolusId=bolus_id, bolusType=2, carbAmount=carbs, carbRatio=7000, correctionBolusIncluded=0, iob=0)


def _bolus_delivery_event(when: str, seq: int, device: str, bolus_id: int, milliunits: int) -> dict:
    return _pump_log_event(
        280, when, seq, device,
        bolusDeliveryStatus=0, bolusId=bolus_id, bolusSource=0, bolusType=2, correction=0, deliveredTotal=milliunits,
        extendedDurationRequested=0, remoteId=0, requestedLater=0, requestedNow=milliunits,
    )


class _FakeTandemSourceV3Adapter:
    """Stands in for tconnectsync 3.x ``TandemSourceApi``: get_pumper() and get_pump_logs()."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, object]] = []
        self.pumps = [
            {
                "serialNumber": "90723647",
                "assignmentId": "uuid-old",
                "availableDataRange": {"start": "2023-01-01T00:00:00", "end": "2025-08-27T11:54:42"},
                "settings": {"details": {}},
            },
            {
                "serialNumber": "1453346",
                "assignmentId": "uuid-mobi",
                "availableDataRange": {"start": "2020-01-15T00:00:00", "end": "2026-11-17T14:07:39"},
                "settings": {"details": {}},
            },
            {
                "serialNumber": "849064",
                "assignmentId": "uuid-retired",
                "availableDataRange": {"start": "2021-01-02T00:00:00", "end": "2023-09-20T21:29:34"},
                "settings": {"details": {}},
            },
        ]
        self.events = {
            "uuid-old": [
                _basal_event("2025-07-10T00:00:00", 1, "uuid-old", 700),
                _bolus_request_event("2025-07-10T07:58:00", 2, "uuid-old", bolus_id=11, carbs=45),
                _bolus_delivery_event("2025-07-10T08:00:00", 3, "uuid-old", bolus_id=11, milliunits=2500),
                _cgm_event("2025-07-10T08:05:00", 4, "uuid-old", 140),
            ],
            "uuid-mobi": [],
        }

    def get_pumper(self):
        return {"pumps": self.pumps}

    def get_pump_logs(self, device_id, min_date=None, max_date=None, event_ids_filter=None):
        self.calls.append((str(device_id), str(min_date), str(max_date), event_ids_filter))
        # The fake returns the same events for every page, so the collector must dedupe across pages.
        return {"events": list(self.events.get(str(device_id), [])), "clockChanges": []}


def _workspace(tmp_path: Path) -> ProjectPaths:
    root = tmp_path / "bayesian-t1dm"
    root.mkdir()
    runtime_root = tmp_path / "runtime"
    cloud_root = tmp_path / "cloud"
    return ProjectPaths.from_root(root, runtime_root=runtime_root, cloud_root=cloud_root).ensure()


def test_tconnectsync_client_queries_overlapping_pumps_and_archives_v3_responses(tmp_path):
    workspace = _workspace(tmp_path)
    window = ExportWindow(date(2025, 7, 5), date(2025, 8, 3))
    adapter = _FakeTandemSourceV3Adapter()
    client = TConnectSyncSourceClient(workspace, adapter=adapter, timezone="America/New_York")

    result = client.export_daily_timeline_window(window, workspace.runtime / "downloads")

    # The retired 2023 pump does not overlap the window; the other two are both queried.
    assert {call[0] for call in adapter.calls} == {"uuid-old", "uuid-mobi"}
    # A 30-day window is paged in requests of at most 28 days, asking for every event type.
    old_calls = [call for call in adapter.calls if call[0] == "uuid-old" and call[3] is None]
    assert [(call[1], call[2]) for call in old_calls] == [("2025-07-05", "2025-08-01"), ("2025-08-02", "2025-08-03")]
    # Segment edits (event 68) are left out of Tandem Source's default list, so they are requested by name.
    edit_calls = [call for call in adapter.calls if call[3] == [68]]
    assert {call[0] for call in edit_calls} <= {"uuid-old", "uuid-mobi"} and edit_calls

    # Events repeated across the two pages are counted once.
    assert result.row_counts["cgm"] == 1
    assert result.row_counts["bolus"] == 1
    assert result.row_counts["basal"] == 1
    normalized_root = workspace.cloud_raw / "tconnectsync" / window.window_id / "normalized"
    assert pd.read_csv(normalized_root / "carbs.csv")["carb_grams"].tolist() == [45.0]
    assert pd.read_csv(normalized_root / "cgm.csv")["glucose"].tolist() == [140]
    bolus = pd.read_csv(normalized_root / "bolus.csv")
    assert bolus["pump_serial"].astype(str).tolist() == ["90723647"]
    assert bolus["tconnect_device_id"].astype(str).tolist() == ["uuid-old"]

    raw_root = workspace.cloud_raw / "tconnectsync" / window.window_id / "raw"
    pumper = json.loads((raw_root / "pumper.json").read_text(encoding="utf-8"))
    assert pumper["selected_pump_serials"] == ["1453346", "90723647"]
    logs = json.loads((raw_root / "pump_logs_90723647_uuid-old.json").read_text(encoding="utf-8"))
    assert len(logs["chunks"]) == 2 and logs["chunks"][0]["response"]["events"][0]["eventCode"] == 279
    assert (raw_root / "pump_logs_1453346_uuid-mobi.json").exists()


def test_tconnectsync_client_lists_a_requested_serial_first(tmp_path):
    workspace = _workspace(tmp_path)
    adapter = _FakeTandemSourceV3Adapter()
    window = ExportWindow(date(2025, 7, 5), date(2025, 8, 3))
    TConnectSyncSourceClient(workspace, adapter=adapter, pump_serial="90723647").export_daily_timeline_window(window, workspace.runtime / "downloads")
    pumper = json.loads((workspace.cloud_raw / "tconnectsync" / window.window_id / "raw" / "pumper.json").read_text(encoding="utf-8"))
    assert pumper["selected_pump_serials"] == ["90723647", "1453346"]


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

    # Both readings of the repeated 01:30 hour are kept; they share a local timestamp but are not
    # duplicate records (different glucose), so the window is not flagged for duplicates.
    assert len(ingested.cgm) == 2
    assert ingested.cgm["glucose"].tolist() == [110, 111]
    assert not ingested.manifest.loc[ingested.manifest["kind"].eq("cgm"), "has_duplicates"].iloc[0]


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

    from bayesian_t1dm.acquisition import _decode_archived_pump_events, _load_archived_window_payloads

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
    (raw_dir / "bolus.json").write_text("[]\n", encoding="utf-8")

    carbs = _decode_archived_pump_events(raw_dir)["carbs"]
    assert [record["carbs"] for record in carbs] == [48.0]
    assert carbs[0]["pump_serial"] == "90723647"
    assert carbs[0]["tconnect_device_id"] == "723357"
    assert str(carbs[0]["eventDateTime"]).startswith("2025-07-10T07:58:00")

    payloads = _load_archived_window_payloads(raw_dir.parent)
    assert [record["carbs"] for record in payloads["carbs"]] == [48.0]


def test_unreadable_archived_pump_events_are_an_error_not_a_silent_skip(tmp_path):
    from bayesian_t1dm.acquisition import AcquisitionError, _decode_archived_pump_events

    raw_dir = tmp_path / "w" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "pump_events_raw_1453346_1032256.txt").write_text("not base64 at all !!\n", encoding="utf-8")
    with pytest.raises(AcquisitionError, match="not valid base64"):
        _decode_archived_pump_events(raw_dir)


def test_raw_pump_events_replace_a_stored_carb_payload(tmp_path):
    """Carbs (and the settings in force) are always rebuilt from the raw events when a window has them."""
    import base64

    from bayesian_t1dm.acquisition import _load_archived_window_payloads

    raw_dir = tmp_path / "w" / "raw"
    raw_dir.mkdir(parents=True)
    request_time = int((pd.Timestamp("2025-07-10 07:58:00") - pd.Timestamp("2008-01-01")).total_seconds())
    stream = _bolus_request(request_time, 1, bolus_id=1, carbs=40) + _bolus_delivery(request_time + 60, 2, bolus_id=1, milliunits=4000)
    (raw_dir / "pump_events_raw_1_2.txt").write_text(base64.b64encode(stream).decode(), encoding="utf-8")
    (raw_dir / "carbs.json").write_text('[{"eventDateTime": "2025-07-10T07:58:00", "carbs": 999}]\n', encoding="utf-8")

    payloads = _load_archived_window_payloads(raw_dir.parent)

    assert [record["carbs"] for record in payloads["carbs"]] == [40.0]


def test_v3_pump_logs_archive_yields_carbs_with_settings(tmp_path):
    from bayesian_t1dm.acquisition import _load_archived_window_payloads

    raw_dir = tmp_path / "w" / "raw"
    raw_dir.mkdir(parents=True)
    request = _bolus_request_event("2025-07-10T07:58:00", 2, "uuid", bolus_id=11, carbs=45)
    request["eventProperties"]["carbRatio"] = 6500
    msg2 = _pump_log_event(65, "2025-07-10T07:58:00", 3, "uuid", bolusId=11, selectedIob=0, options=0, standardPercent=100, duration=0, isf=65, targetBg=110, userOverride=0, declinedCorrection=0)
    delivery = _bolus_delivery_event("2025-07-10T08:00:00", 4, "uuid", bolus_id=11, milliunits=6900)
    chunk = {"start": "2025-07-10", "end": "2025-07-10", "response": {"events": [request, msg2, delivery], "clockChanges": []}}
    (raw_dir / "pump_logs_1453346_uuid.json").write_text(json.dumps({"pump_serial": "1453346", "pump_device_id": "uuid", "chunks": [chunk, chunk]}), encoding="utf-8")
    (raw_dir / "bolus.json").write_text("[]\n", encoding="utf-8")

    carbs = _load_archived_window_payloads(raw_dir.parent)["carbs"]

    assert len(carbs) == 1  # repeated across chunks, counted once
    assert (carbs[0]["carbs"], carbs[0]["carb_ratio_g_per_unit"], carbs[0]["isf_mgdl_per_unit"]) == (45.0, 6.5, 65.0)
    assert carbs[0]["pump_serial"] == "1453346"

def test_carb_entries_carry_the_bolus_calculator_settings_in_force(tmp_path):
    """Each logged meal keeps the carb ratio (MSG1), correction factor and target (MSG2) the pump used."""
    from tconnectsync.eventparser.generic import Events

    from bayesian_t1dm.acquisition import _tconnectsync_event_payloads

    msg2 = _pump_log_event(
        65, "2025-07-10T07:58:00", 3, "uuid",
        bolusId=11, selectedIob=0, options=0, standardPercent=100, duration=0, isf=65, targetBg=110, userOverride=0, declinedCorrection=0,
    )
    events = list(
        Events(
            [
                _bolus_request_event("2025-07-10T07:58:00", 2, "uuid", bolus_id=11, carbs=45) | {"eventProperties": {**_bolus_request_event("x", 2, "uuid", 11, 45)["eventProperties"], "carbRatio": 6500}},
                msg2,
                _bolus_delivery_event("2025-07-10T08:00:00", 4, "uuid", bolus_id=11, milliunits=6900),
            ]
        )
    )
    carbs = _tconnectsync_event_payloads(events)["carbs"]
    assert len(carbs) == 1
    assert carbs[0]["carbs"] == 45.0
    assert carbs[0]["carb_ratio_g_per_unit"] == pytest.approx(6.5)
    assert carbs[0]["isf_mgdl_per_unit"] == pytest.approx(65.0)
    assert carbs[0]["target_bg_mgdl"] == pytest.approx(110.0)

    window = ExportWindow(date(2025, 7, 10), date(2025, 7, 11))
    ingested, _ = normalize_tconnectsync_payloads({"carbs": carbs}, window=window, endpoint_family="tconnectsync")
    row = ingested.carbs.iloc[0]
    assert (row["carb_ratio_g_per_unit"], row["isf_mgdl_per_unit"], row["target_bg_mgdl"]) == (6.5, 65.0, 110.0)


def test_cgm_uses_sensor_time_and_ignores_alerts(tmp_path):
    """Backfilled readings arrive together but belong at the time the sensor measured them."""
    from tconnectsync.eventparser.generic import Events

    from bayesian_t1dm.acquisition import _tconnectsync_event_payloads

    events = [
        _cgm_event("2026-10-04T08:03:30", 1, "uuid", 150),
        # Signal dropout: two readings measured at 08:08:30 and 08:13:30 arrive at 08:20:05.
        _cgm_event("2026-10-04T08:20:05", 2, "uuid", 155, measured="2026-10-04T08:08:30"),
        _cgm_event("2026-10-04T08:20:05", 3, "uuid", 161, measured="2026-10-04T08:13:30"),
        _cgm_event("2026-10-04T08:23:30", 4, "uuid", 166),
        # A CGM alert is not a reading.
        _pump_log_event(256 + 999, "2026-10-04T08:25:00", 5, "uuid"),
    ]
    cgm = _tconnectsync_event_payloads(list(Events(events)))["cgm"]
    times = [pd.Timestamp(row["eventDateTime"]).strftime("%H:%M:%S") for row in cgm]
    assert times == ["08:03:30", "08:08:30", "08:13:30", "08:23:30"]
    assert [row["egv_estimatedGlucoseValue"] for row in cgm] == [150, 155, 161, 166]

    window = ExportWindow(date(2026, 10, 4), date(2026, 10, 4))
    ingested, _ = normalize_tconnectsync_payloads({"cgm": cgm}, window=window, endpoint_family="tconnectsync")
    slots = pd.to_datetime(ingested.cgm["timestamp"]).dt.strftime("%H:%M").tolist()
    assert slots == ["08:00", "08:05", "08:10", "08:20"]  # one reading per slot; none piled into 08:20


def test_cgm_sensor_time_keeps_its_wall_clock_across_a_dst_change(monkeypatch):
    """A reading measured at 01:58 EST and received at 03:01 EDT (clocks jumped forward) is stored at 01:58."""
    from tconnectsync.eventparser.generic import Events

    from bayesian_t1dm.acquisition import _tconnectsync_event_payloads

    monkeypatch.setenv("TIMEZONE_NAME", "America/New_York")
    events = [_cgm_event("2026-03-08T03:01:00", 1, "uuid", 150, measured="2026-03-08T01:58:00")]
    cgm = _tconnectsync_event_payloads(list(Events(events)))["cgm"]
    window = ExportWindow(date(2026, 3, 8), date(2026, 3, 8))
    ingested, _ = normalize_tconnectsync_payloads({"cgm": cgm}, window=window, endpoint_family="tconnectsync")
    assert pd.to_datetime(ingested.cgm["timestamp"]).dt.strftime("%H:%M").tolist() == ["01:55"]


def test_mobi_carb_ratio_is_derived_from_the_food_portion_when_it_is_not_recorded():
    """The Mobi records carbRatio 0 for app boluses; carbs / food portion recovers it when nothing was folded in."""
    from tconnectsync.eventparser.generic import Events

    from bayesian_t1dm.acquisition import _tconnectsync_event_payloads

    def meal(bolus_id: int, minute: int, *, carbs: int, bg: int, food: float, correction: float, target: int = 0) -> list[dict]:
        when = f"2026-10-20T08:{minute:02d}:00"
        return [
            _pump_log_event(64, when, bolus_id * 10 + 1, "uuid", bg=bg, bolusId=bolus_id, bolusType=3, carbAmount=carbs, carbRatio=0, correctionBolusIncluded=int(correction > 0), iob=0),
            _pump_log_event(65, when, bolus_id * 10 + 2, "uuid", bolusId=bolus_id, declinedCorrection=0, duration=0, isf=0, options=4, selectedIob=1, spareB6=0, standardPercent=100, targetBg=target, userOverride=0),
            _pump_log_event(66, when, bolus_id * 10 + 3, "uuid", bolusId=bolus_id, correctionBolusSize=correction, foodBolusSize=food, queued=0, spareA3=0, totalBolusSize=food + correction),
            _bolus_delivery_event(when, bolus_id * 10 + 4, "uuid", bolus_id=bolus_id, milliunits=int((food + correction) * 1000)),
        ]

    events = [
        *meal(1, 0, carbs=35, bg=133, food=4.67, correction=0.38),  # positive correction: food portion is carbs / 7.5
        *meal(2, 10, carbs=30, bg=95, food=3.72, correction=0.0),  # below an unknown target: portion may be reduced
        *meal(3, 20, carbs=30, bg=140, food=4.0, correction=0.0, target=110),  # at or above a recorded target
    ]
    carbs = _tconnectsync_event_payloads(list(Events(events)))["carbs"]
    assert [row["carb_ratio_g_per_unit"] for row in carbs] == [7.5, None, 7.5]
    assert [row["correction_bolus_units"] for row in carbs] == [0.38, 0.0, 0.0]
    assert [row["calculator_glucose_mgdl"] for row in carbs] == [133, 95, 140]


def test_pump_times_are_stored_in_local_time_not_utc(monkeypatch):
    """An 08:00 Eastern bolus must be stored as 08:00, the same clock Apple Health uses."""
    monkeypatch.setenv("TIMEZONE_NAME", "America/New_York")
    window = ExportWindow(date(2026, 9, 20), date(2026, 9, 20))
    payloads = {"bolus": [{"completionDateTime": "2026-09-20T08:00:15-04:00", "actualTotalBolusRequested": 3.0}], "cgm": [], "basal": [], "activity": []}
    ingested, _ = normalize_tconnectsync_payloads(payloads, window=window, endpoint_family="tconnectsync")
    assert pd.Timestamp(ingested.bolus["timestamp"].iloc[0]) == pd.Timestamp("2026-09-20 08:00:00")


def test_missing_timezone_is_an_error_not_utc(monkeypatch):
    from bayesian_t1dm.paths import analysis_timezone

    monkeypatch.delenv("TIMEZONE_NAME", raising=False)
    with pytest.raises(ValueError, match="Set TIMEZONE_NAME"):
        analysis_timezone()
    monkeypatch.setenv("TIMEZONE_NAME", "Eastern")
    with pytest.raises(ValueError, match="not a valid IANA timezone"):
        analysis_timezone()


def test_renormalizing_an_archived_window_never_rewrites_its_raw_folder(tmp_path):
    workspace = _workspace(tmp_path)
    raw_dir = workspace.cloud_raw / "tconnectsync" / "2024-05-01__2024-05-03" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "cgm.json").write_text('[{"eventDateTime": "2024-05-01T00:00:00", "egv_estimatedGlucoseValue": 110}]', encoding="utf-8")
    (raw_dir / "bolus.json").write_text("[]", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in raw_dir.iterdir()}

    acquisition.normalize_tconnectsync_archive(workspace, raw_root=workspace.cloud_raw / "tconnectsync", force=True)

    assert {path.name: path.read_bytes() for path in raw_dir.iterdir()} == before
    assert (raw_dir.parent / "normalized" / "cgm.csv").exists()


def test_two_equal_carb_requests_in_one_slot_survive_repeated_export():
    window = ExportWindow(date(2024, 6, 1), date(2024, 6, 1))
    first = {"eventDateTime": "2024-06-01T08:00:10", "carbs": 15, "pump_serial": "1", "source_seq_num": 2}
    second = {**first, "eventDateTime": "2024-06-01T08:02:20", "source_seq_num": 3}
    data, _ = normalize_tconnectsync_payloads({"carbs": [first, second, first]}, window=window, endpoint_family="carbs")
    assert data.carbs.carb_grams.tolist() == [15, 15]
    assert data.carbs.event_time.nunique() == 2


@pytest.mark.parametrize(("day", "start", "end", "expected"),
                         [(date(2025, 3, 9), "01:58:00-05:00", "03:03:00-04:00", [2, 3]),
                          (date(2025, 11, 2), "01:58:00-04:00", "01:03:00-05:00", [3, 2])])
def test_basal_preserves_elapsed_exposure_at_clock_change(day, start, end, expected):
    window = ExportWindow(day, day)
    data, _ = normalize_tconnectsync_payloads(
        {"basal": [{"startDateTime": f"{day}T{start}", "endDateTime": f"{day}T{end}", "basalRate": 1}]},
        window=window, endpoint_family="basal", timezone="America/New_York",
    )
    elapsed = (data.basal.end_timestamp - data.basal.start_timestamp).dt.total_seconds() / 60
    assert elapsed.tolist() == expected
