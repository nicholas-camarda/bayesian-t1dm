from __future__ import annotations

import base64
import json
import struct

import pandas as pd

from bayesian_t1dm.context_rules import CGM_MAX_MGDL, CGM_MIN_MGDL, exercise_intervals, inside_intervals, valid_cgm
from bayesian_t1dm.device_events import decode_device_events

TANDEM_EPOCH = 1199145600


def _record(event_id: int, when: str, sequence: int, mode: int | None = None, profile_rate: int | None = None) -> bytes:
    seconds = int(pd.Timestamp(when).timestamp()) - TANDEM_EPOCH  # the pump's wall clock read as UTC, as tconnectsync does
    raw = bytearray(26)
    struct.pack_into(">H", raw, 0, event_id)
    struct.pack_into(">I", raw, 2, seconds)
    struct.pack_into(">I", raw, 6, sequence)
    if mode is not None:
        raw[13] = mode  # currentUserMode of event 229
    if profile_rate is not None:
        struct.pack_into(">H", raw, 14, profile_rate)  # profileBasalRate of event 279, in thousandths of a unit per hour
    return bytes(raw)


def _binary_window(root, name, pump, records):
    raw = root / name / "raw"
    raw.mkdir(parents=True)
    (raw / f"pump_events_raw_{pump}_9.txt").write_text(base64.b64encode(b"".join(records)).decode(), encoding="utf-8")


def test_modes_and_site_events_are_read_from_the_binary_archive_and_the_json_logs(tmp_path):
    root = tmp_path / "tconnectsync"
    _binary_window(root, "2024-01-01__2024-01-30", "111", [
        _record(229, "2024-01-10 21:30:00", 5, mode=1), _record(229, "2024-01-11 05:00:00", 6, mode=0), _record(33, "2024-01-12 08:00:00", 7),
        _record(61, "2024-01-12 08:05:00", 8), _record(229, "2020-01-15 06:00:00", 1, mode=0),  # a pump clock not yet set
        _record(279, "2024-01-12 23:55:00", 9, profile_rate=700), _record(279, "2024-01-12 23:50:00", 10, profile_rate=65535),  # a rate not set
    ])
    _binary_window(root, "2024-01-15__2024-02-14", "111", [_record(229, "2024-01-10 21:30:00", 5, mode=1)])  # overlapping window repeats an event
    json_dir = root / "2026-05-01__2026-05-30" / "raw"
    json_dir.mkdir(parents=True)
    events = [{"eventCode": 229, "sequenceNumber": 900, "pumpDateTime": "2026-05-02T22:00:00", "eventProperties": {"currentUserMode": 2}},
              {"eventCode": 63, "sequenceNumber": 901, "pumpDateTime": "2026-05-03T07:00:00", "eventProperties": {}},
              {"eventCode": 279, "sequenceNumber": 902, "pumpDateTime": "2026-05-03T07:00:00", "eventProperties": {"profileBasalRate": 850}},
              {"eventCode": 279, "sequenceNumber": 903, "pumpDateTime": "2026-05-03T07:05:00", "eventProperties": {"profileBasalRate": 65535}}]
    (json_dir / "pump_logs_1.json").write_text(json.dumps({"pump_serial": "111.0", "chunks": [{"response": {"events": events}}]}), encoding="utf-8")
    modes, sites, rates, span = decode_device_events(root)
    assert modes["timestamp"].tolist() == [pd.Timestamp("2024-01-10 21:30"), pd.Timestamp("2024-01-11 05:00"), pd.Timestamp("2026-05-02 22:00")]
    assert modes["mode"].tolist() == [1, 0, 2] and set(modes["pump_serial"]) == {"111"}
    assert sites["kind"].tolist() == ["cartridge filled", "cannula filled", "tubing filled"]
    assert rates["rate"].tolist() == [0.7, 0.85] and rates["timestamp"].tolist() == [pd.Timestamp("2024-01-12 23:55"), pd.Timestamp("2026-05-03 07:00")]
    assert span["111"][0] == pd.Timestamp("2024-01-01").date() and span["111"][1] == pd.Timestamp("2026-05-30").date()


def test_cgm_readings_outside_the_sensor_range_are_not_glucose():
    cgm = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=5, freq="5min"), "glucose": [0.0, 39.0, CGM_MIN_MGDL, 120.0, CGM_MAX_MGDL + 1]})
    assert valid_cgm(cgm)["glucose"].tolist() == [CGM_MIN_MGDL, 120.0]


def test_one_exercise_rule_covers_walks_workouts_and_exercise_mode():
    workouts = pd.DataFrame({"start_timestamp": pd.to_datetime(["2026-06-01 07:00", "2026-06-02 07:00"]), "end_timestamp": pd.to_datetime(["2026-06-01 07:30", "2026-06-02 07:05"]), "duration_seconds": [1800.0, 300.0]})
    modes = pd.DataFrame({"pump_serial": "1", "timestamp": pd.to_datetime(["2026-06-03 12:00", "2026-06-03 13:00"]), "mode": [2, 0]})
    rule = exercise_intervals(workouts, modes)
    times = pd.Series(pd.to_datetime(["2026-06-01 06:59", "2026-06-01 07:10", "2026-06-01 11:30", "2026-06-01 11:31", "2026-06-02 07:30", "2026-06-03 12:30", "2026-06-03 17:00", "2026-06-03 17:01"]))
    assert inside_intervals(times, rule).tolist() == [False, True, True, False, False, True, True, False]  # 10 min or more; mode until its end; then 4 hours
    assert not inside_intervals(times, exercise_intervals(None, None)).any()
