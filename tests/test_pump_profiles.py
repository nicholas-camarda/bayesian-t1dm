from __future__ import annotations

import base64
import datetime as dt
import struct
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bayesian_t1dm.pump_profiles import ProfileHistory, decode_profile_events

EPOCH = dt.datetime(2008, 1, 1)


def _record(event_id: int, when: dt.datetime, sequence: int, payload_le: bytes) -> bytes:
    """One archived event: big-endian header, then the pump's little-endian payload with each 4-byte word reversed."""
    assert len(payload_le) == 16
    header = struct.pack(">HII", event_id, int((when - EPOCH).total_seconds()), sequence)
    return header + b"".join(payload_le[i : i + 4][::-1] for i in range(0, 16, 4))


def _segment(when, sequence, *, profile, index, modification, start, icr, isf, target=110, basal=0.8, status=15):
    payload = struct.pack("<BBBBHHHHH", profile, status, index, modification, start, int(basal * 1000), isf, target, int(icr * 1000)) + b"\x00\x00"
    return _record(68, when, sequence, payload)


def _activate(when, sequence, *, profile, name):
    return _profile_action(when, sequence, profile=profile, status=3, name=name)


def _profile_action(when, sequence, *, profile, status, source=0, name=""):
    payload = struct.pack("<BBB5x", profile, status, source) + name.encode().ljust(8, b"\x00")
    return _record(69, when, sequence, payload)


def _archive(tmp_path, windows: dict[str, list[bytes]], serial="1453346"):
    root = tmp_path / "tconnectsync"
    for window, records in windows.items():
        raw = root / window / "raw"
        raw.mkdir(parents=True)
        (raw / f"pump_events_raw_{serial}_99.txt").write_text(base64.b64encode(b"".join(records)).decode())
    return root


def test_profile_history_replays_edits_copies_deletes_and_switches(tmp_path):
    day = dt.datetime(2025, 9, 1)
    records = [
        # Main (profile 0) is created here: midnight 7.0/65 and 09:00 6.5/60.
        _profile_action(day, 1, profile=0, status=0, name="Main"),
        _segment(day, 2, profile=0, index=0, modification=0, start=0, icr=7.0, isf=65),
        _segment(day, 3, profile=0, index=1, modification=0, start=540, icr=6.5, isf=60),
        _activate(day, 4, profile=0, name="Main"),
        # Two days later the 09:00 segment's start moves to 07:00 and its values change (modification 2).
        _segment(day + dt.timedelta(days=2), 5, profile=0, index=1, modification=2, start=420, icr=7.5, isf=55),
        # A copy of Main, trimmed by deleting its second segment (modification 1), then edited.
        _profile_action(day + dt.timedelta(days=3), 6, profile=3, status=1, source=0, name="Copy"),
        _segment(day + dt.timedelta(days=3), 7, profile=3, index=1, modification=1, start=420, icr=7.5, isf=55),
        _segment(day + dt.timedelta(days=3), 8, profile=3, index=0, modification=4, start=0, icr=9.0, isf=65),
        # Medium (profile 2) was created before the archive: only its 07:00 segment is logged.
        _segment(day + dt.timedelta(days=3), 9, profile=2, index=1, modification=4, start=420, icr=8.0, isf=60, target=120),
        _activate(day + dt.timedelta(days=4), 10, profile=2, name="Medium"),
    ]
    root = _archive(tmp_path, {"2025-08-25__2025-09-23": records})
    events, visible = decode_profile_events(root)
    assert visible == {"1453346": (dt.date(2025, 9, 1), dt.date(2025, 9, 23))}  # from the first profile event
    history = ProfileHistory(events, visible)

    def at(when):
        carbs = pd.DataFrame({"timestamp": [when], "pump_serial": ["1453346"]})
        return history.settings_for(carbs).iloc[0].tolist()

    assert at(dt.datetime(2025, 9, 1, 8, 0)) == [0, 1, 7.0, 65, 110, 0]  # before the edit: midnight segment
    assert at(dt.datetime(2025, 9, 3, 8, 0)) == [0, 1, 7.5, 55, 110, 420]  # the edited segment now starts at 07:00
    assert at(dt.datetime(2025, 9, 3, 6, 0)) == [0, 1, 7.0, 65, 110, 0]
    # The copy starts as Main was when copied, then diverges; Main is untouched.
    copy = history.segment_in_force("1453346", 3, dt.datetime(2025, 9, 4, 12, 0))
    assert (copy["carb_ratio_g_per_unit"], copy["start_minutes"]) == (9.0, 0)
    main = history.segment_in_force("1453346", 0, dt.datetime(2025, 9, 4, 12, 0))
    assert main["carb_ratio_g_per_unit"] == 7.5
    # Medium predates the archive: a slot after the last one seen might exist, so the table cannot answer.
    assert np.isnan(at(dt.datetime(2025, 9, 6, 6, 0))[2]) and np.isnan(at(dt.datetime(2025, 9, 6, 8, 0))[2])
    # After the archive's binary coverage ends, edits are invisible: the profile stays known, its values do not.
    late = at(dt.datetime(2025, 10, 10, 8, 0))
    assert late[:2] == [2, 0] and np.isnan(late[2])


def test_deleting_a_segment_of_a_fully_seen_profile_that_never_existed_fails(tmp_path):
    root = _archive(tmp_path, {"2025-08-25__2025-09-23": [
        _profile_action(dt.datetime(2025, 9, 1), 1, profile=0, status=0, name="Main"),
        _segment(dt.datetime(2025, 9, 1), 2, profile=0, index=3, modification=1, start=0, icr=7, isf=60),
    ]})
    events, visible = decode_profile_events(root)
    with pytest.raises(ValueError, match="never saw"):
        ProfileHistory(events, visible)


def test_profile_history_outranks_carrying_and_a_profile_switch_ends_the_carry():
    from bayesian_t1dm.therapy_research import _resolve_meal_settings

    def meal(profile, icr=None, cf=None, history=(None, None), visible=1.0):
        return SimpleNamespace(pump_serial="1453346", profile_id=profile, profile_edits_visible=visible, carb_ratio_g_per_unit=icr, isf_mgdl_per_unit=cf,
                               profile_carb_ratio_g_per_unit=history[0], profile_isf_mgdl_per_unit=history[1],
                               calculator_glucose_mgdl=None, calculator_iob_units=None, correction_bolus_units=None)

    rows = [
        meal(0, icr=7.0, history=(7.0, 65)),  # I:C from the calculator split, CF from the pump history
        meal(0),                              # nothing recorded: carried within Main
        meal(2, icr=8.0, history=(8.0, 60)),  # switched to Medium: Main's CF is not carried here
        meal(2, history=(8.0, 60)),           # history alone
        meal(2, visible=0.0),                 # edits no longer visible: Medium's 8.0 is not carried in; the 7.5 after is
        meal(2, icr=7.5, visible=0.0),        # keeps its own I:C; only the missing CF comes from the programmed profile
    ]
    resolved = _resolve_meal_settings(rows, "morning", settings=SimpleNamespace(icr_for_segment=lambda s: 6.5, correction_factor_for_segment=lambda s: 65.0))
    assert resolved == [(7.0, 65.0, "pump profile history"), (7.0, 65.0, "carried from nearby meals"),
                        (8.0, 60.0, "pump profile history"), (8.0, 60.0, "pump profile history"),
                        (7.5, 65.0, "pump profile"), (7.5, 65.0, "pump profile")]


def test_each_meal_is_judged_against_its_own_resolved_settings_not_the_snapshot():
    import pandas as pd

    from bayesian_t1dm.therapy_research import LatentMealTherapySettings, _attach_resolved_settings, _settings_series

    settings = LatentMealTherapySettings.default()
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 08:00", "2026-01-02 08:00"]),
            "segment": ["morning", "morning"],
            "pump_serial": ["1", "1"],
            "profile_id": [0, 0],
            "profile_edits_visible": [1.0, 1.0],
            "carb_ratio_g_per_unit": [7.0, 8.0],
            "isf_mgdl_per_unit": [65.0, 60.0],
            "profile_carb_ratio_g_per_unit": [7.0, 8.0],
            "profile_isf_mgdl_per_unit": [65.0, 60.0],
            "calculator_glucose_mgdl": [None, None],
            "calculator_iob_units": [None, None],
            "correction_bolus_units": [None, None],
        }
    )
    attached = _attach_resolved_settings(frame, settings)
    assert attached["resolved_icr"].tolist() == [7.0, 8.0]
    assert _settings_series(attached, settings, field="icr").tolist() == [7.0, 8.0]
    assert _settings_series(attached, settings, field="correction_factor").tolist() == [65.0, 60.0]


def test_a_carry_never_crosses_a_profile_switch_and_back():
    from bayesian_t1dm.therapy_research import _resolve_meal_settings

    def meal(profile, icr=None, cf=None):
        return SimpleNamespace(pump_serial="1", profile_id=profile, profile_edits_visible=0.0, carb_ratio_g_per_unit=icr, isf_mgdl_per_unit=cf,
                               profile_carb_ratio_g_per_unit=None, profile_isf_mgdl_per_unit=None,
                               calculator_glucose_mgdl=None, calculator_iob_units=None, correction_bolus_units=None)

    rows = [meal(0, icr=7.0, cf=65), meal(2, icr=8.0, cf=60), meal(0), meal(0)]  # Main, Medium, then Main again with nothing recorded
    resolved = _resolve_meal_settings(rows, "morning", settings=SimpleNamespace(icr_for_segment=lambda s: 6.5, correction_factor_for_segment=lambda s: 55.0))
    # The later Main meals do not borrow Main's 7.0 from before the switch: they fall back to the programmed profile.
    assert [(icr, cf) for icr, cf, _ in resolved] == [(7.0, 65.0), (8.0, 60.0), (6.5, 55.0), (6.5, 55.0)]


def test_segment_edits_from_pump_logs_json_extend_the_visible_history(tmp_path):
    import json

    root = tmp_path / "tconnectsync"
    raw = root / "2026-04-23__2026-05-22" / "raw"
    raw.mkdir(parents=True)

    def edit(seq, when, index, modification, start, icr, isf, basal=710):
        return {
            "eventCode": 68, "sequenceNumber": seq, "pumpDateTime": when,
            "eventProperties": {"idp": 0, "status": [0, 1, 2, 3], "segmentIndex": index, "modificationType": modification,
                                "startTime": start, "basalRate": basal, "isf": isf, "targetBg": 110, "carbRatio": int(icr * 1000)},
        }

    events = [
        {"eventCode": 69, "sequenceNumber": 1, "pumpDateTime": "2026-04-23T08:00:00",
         "eventProperties": {"idp": 0, "status": 0, "sourceIdp": 0, "name0": 77, "name1": 97, "name2": 105, "name3": 110, "name4": 0, "name5": 255, "name6": 255, "name7": 255}},
        edit(2, "2026-04-23T08:00:01", 0, 0, 0, 7.0, 60),
        edit(3, "2026-04-23T08:00:02", 1, 0, 300, 7.0, 65),
        {"eventCode": 69, "sequenceNumber": 4, "pumpDateTime": "2026-04-23T08:01:00",
         "eventProperties": {"idp": 0, "status": 3, "sourceIdp": 0, "name0": 77, "name1": 97, "name2": 105, "name3": 110, "name4": 0, "name5": 255, "name6": 255, "name7": 255}},
        edit(5, "2026-05-10T09:29:43", 1, 4, 300, 6.5, 65),  # carb ratio edit
    ]
    (raw / "pump_logs_1453346_dev.json").write_text(json.dumps({"pump_serial": "1453346", "chunks": [{"start": "2026-04-23", "end": "2026-05-22", "response": {"events": [e for e in events if e["eventCode"] == 69]}}]}))
    (raw / "pump_profile_edits_1453346_dev.json").write_text(json.dumps({"pump_serial": "1453346", "chunks": [{"start": "2026-04-23", "end": "2026-05-22", "response": {"events": [e for e in events if e["eventCode"] == 68]}}]}))
    frame, visible = decode_profile_events(root)
    assert visible == {"1453346": (dt.date(2026, 4, 23), dt.date(2026, 5, 22))}
    history = ProfileHistory(frame, visible)
    before = history.segment_in_force("1453346", 0, dt.datetime(2026, 5, 9, 7, 30))
    after = history.segment_in_force("1453346", 0, dt.datetime(2026, 5, 11, 7, 30))
    assert (before["carb_ratio_g_per_unit"], after["carb_ratio_g_per_unit"]) == (7.0, 6.5)


def test_asking_about_other_hours_uses_the_profile_as_it_stood_at_the_given_moment(tmp_path):
    day = dt.datetime(2025, 9, 1)
    records = [
        _profile_action(day, 1, profile=0, status=0, name="Main"),
        _segment(day, 2, profile=0, index=0, modification=0, start=0, icr=7.0, isf=65),
        _segment(day, 3, profile=0, index=1, modification=0, start=300, icr=7.0, isf=65),
        _activate(day, 4, profile=0, name="Main"),
        _segment(day.replace(hour=9, minute=30), 5, profile=0, index=1, modification=4, start=300, icr=6.5, isf=65),
    ]
    events, visible = decode_profile_events(_archive(tmp_path, {"2025-08-25__2025-09-23": records}))
    history = ProfileHistory(events, visible)
    at_noon = dt.datetime(2025, 9, 1, 12, 0)
    assert history.segment_in_force("1453346", 0, at_noon, minute_of_day=6 * 60)["carb_ratio_g_per_unit"] == 6.5
    # Asked at 07:00, before the 09:30 edit, the same hour of the day still shows the old value.
    assert history.segment_in_force("1453346", 0, dt.datetime(2025, 9, 1, 7, 0), minute_of_day=6 * 60)["carb_ratio_g_per_unit"] == 7.0


def test_a_meal_is_judged_by_the_profile_at_the_moment_recorded_not_its_grid_slot(tmp_path):
    day = dt.datetime(2025, 9, 1)
    records = [
        _profile_action(day, 1, profile=0, status=0, name="Main"),
        _segment(day, 2, profile=0, index=0, modification=0, start=0, icr=7.0, isf=65),
        _activate(day, 3, profile=0, name="Main"),
        _segment(day.replace(hour=7, minute=57), 4, profile=0, index=0, modification=4, start=0, icr=6.5, isf=65),
    ]
    events, visible = decode_profile_events(_archive(tmp_path, {"2025-08-25__2025-09-23": records}))
    history = ProfileHistory(events, visible)
    carbs = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-09-01 07:55", "2025-09-01 07:55"]),  # both on the 07:55 grid slot
            "event_time": pd.to_datetime(["2025-09-01 07:56:10", "2025-09-01 07:58:40"]),
            "pump_serial": ["1453346", "1453346"],
        }
    )
    assert history.settings_for(carbs)["profile_carb_ratio_g_per_unit"].tolist() == [7.0, 6.5]


def test_values_at_use_exact_edit_time_and_pump(tmp_path):
    day = dt.datetime(2025, 9, 1)
    edit = day.replace(hour=9, minute=7)
    events, visible = decode_profile_events(_archive(tmp_path, {"2025-09-01__2025-09-30": [
        _profile_action(day, 1, profile=0, status=0),
        _segment(day, 2, profile=0, index=0, modification=0, start=0, icr=7, isf=65, basal=0.8),
        _activate(day, 3, profile=0, name="Main"),
        _segment(edit, 4, profile=0, index=0, modification=4, start=0, icr=7, isf=65, basal=0.9),
    ]}, serial="10001"))
    history = ProfileHistory(events, visible)
    stamps = pd.Series([edit - dt.timedelta(seconds=1), edit, edit.replace(minute=10), edit, day - dt.timedelta(days=1)])
    rates = history.values_at(pd.Series(["10001"] * 3 + ["another", "10001"]), stamps, "basal_units_per_hour")
    np.testing.assert_allclose(rates, [0.8, 0.9, 0.9, np.nan, np.nan])
    # The same replay serves every segment value: the I:C and correction factor in force at the 09:07 edit are unchanged.
    ratios = history.values_at(pd.Series(["10001"] * 3), stamps.iloc[:3], "carb_ratio_g_per_unit")
    np.testing.assert_allclose(ratios, [7.0, 7.0, 7.0])
    np.testing.assert_allclose(history.values_at(pd.Series(["10001"] * 3), stamps.iloc[:3], "isf_mgdl_per_unit"), [65.0, 65.0, 65.0])
