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
    payload = struct.pack("<BBB5x", profile, 3, 0) + name.encode().ljust(8, b"\x00")
    return _record(69, when, sequence, payload)


def _archive(tmp_path, windows: dict[str, list[bytes]], serial="1453346"):
    root = tmp_path / "tconnectsync"
    for window, records in windows.items():
        raw = root / window / "raw"
        raw.mkdir(parents=True)
        (raw / f"pump_events_raw_{serial}_99.txt").write_text(base64.b64encode(b"".join(records)).decode())
    return root


def test_profile_history_replays_edits_inserts_and_switches(tmp_path):
    day = dt.datetime(2025, 9, 1)
    records = [
        # Main (profile 0): midnight 7.0/65, 09:00 6.5/60; then a 07:00 segment is inserted between them.
        _segment(day, 1, profile=0, index=0, modification=1, start=0, icr=7.0, isf=65),
        _segment(day, 2, profile=0, index=1, modification=1, start=540, icr=6.5, isf=60),
        _activate(day, 3, profile=0, name="Main"),
        _segment(day + dt.timedelta(days=2), 4, profile=0, index=1, modification=0, start=0, icr=0, isf=0, target=0, basal=0, status=0),
        _segment(day + dt.timedelta(days=2), 5, profile=0, index=1, modification=6, start=420, icr=7.5, isf=55),
        # Medium (profile 2) has only its 07:00 segment logged, so earlier minutes are unknown.
        _segment(day + dt.timedelta(days=3), 6, profile=2, index=1, modification=4, start=420, icr=8.0, isf=60, target=120),
        _activate(day + dt.timedelta(days=4), 7, profile=2, name="Medium"),
    ]
    root = _archive(tmp_path, {"2025-08-25__2025-09-23": records})
    events, visible = decode_profile_events(root)
    assert visible == {"1453346": (dt.date(2025, 9, 1), dt.date(2025, 9, 23))}  # from the first profile event
    assert events.loc[events["event"].eq(69), "profile_name"].tolist() == ["Main", "Medium"]
    history = ProfileHistory(events, visible)

    def at(when):
        carbs = pd.DataFrame({"timestamp": [when], "pump_serial": ["1453346"]})
        return history.settings_for(carbs).iloc[0].tolist()

    assert at(dt.datetime(2025, 9, 1, 8, 0)) == [0, 1, 7.0, 65, 110]  # before the insert: midnight segment
    assert at(dt.datetime(2025, 9, 3, 8, 0)) == [0, 1, 7.5, 55, 110]  # the inserted 07:00 segment
    assert at(dt.datetime(2025, 9, 3, 9, 30)) == [0, 1, 6.5, 60, 110]  # the 09:00 segment moved to index 2
    assert at(dt.datetime(2025, 9, 6, 8, 0)) == [2, 1, 8.0, 60, 120]  # after switching to Medium
    assert at(dt.datetime(2025, 9, 6, 6, 0))[:2] == [2, 1] and np.isnan(at(dt.datetime(2025, 9, 6, 6, 0))[2])  # unseen slot
    # After the archive's binary coverage ends, edits are invisible: the profile stays known, its values do not.
    late = at(dt.datetime(2025, 10, 10, 8, 0))
    assert late[:2] == [2, 0] and np.isnan(late[2])


def test_deleting_a_segment_the_history_never_saw_fails(tmp_path):
    root = _archive(tmp_path, {"2025-08-25__2025-09-23": [_segment(dt.datetime(2025, 9, 1), 1, profile=0, index=3, modification=2, start=0, icr=7, isf=60)]})
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
