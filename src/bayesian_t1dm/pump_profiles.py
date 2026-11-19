"""Pump profile (IDP) history from the Tandem event log.

A Tandem pump logs every edit to a personal profile and every profile switch:

- ``LID_IDP_TD_SEG`` (event 68): one time-of-day segment of one profile after a change, with its start time,
  basal rate, correction factor (ISF), target and carb ratio. ``modificationType`` says what changed. Replaying
  the archive shows 1 is a segment *deleted* (a profile copied from another is trimmed by deleting its segments
  one at a time, each record carrying the deleted segment's values) and every other value (0 created, 2 start
  time changed, 3-6 a value edited, 7 re-logged unchanged) leaves the segment at its index with these values;
  this reading agrees with the t:slim's own calculator records for 99.9% of carb entries (the earlier
  reading, 1 = insert and 2 = delete, agreed for 99.7% and left duplicate start times in 93 snapshots).
  ``status`` is a bitmask of the values that are set, 15 meaning all four.
- ``LID_IDP`` (event 69): an action on a whole profile; ``status`` 0 created, 1 copied (from ``sourceIdp``),
  2 deleted, 3 activated. ``LID_IDP_MSG2`` (57) carries the rest of the profile name.

tconnectsync does not decode event 68 or 69, so they are read here from the archived raw events. Tandem
Source serializes each 4-byte word of the pump's little-endian record in reverse byte order; the offsets
below are pumpX2's (``IdpTimeDependentSegmentHistoryLog``, ``IdpActionHistoryLog``) after that reversal.

The pump-logs JSON that tconnectsync 3.x downloads carries event 69 (decoded by the server) but not event 68,
which Tandem Source leaves out of its default event list. Event 68 is requested by name and archived as
``pump_profile_edits_*.json`` (decoded by the server), so edits are known from the binary archive (tconnectsync
2.x windows) and from those files; where neither covers the pump, edits are invisible and the history is not used.
"""

from __future__ import annotations

import base64
import bisect
import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EVENT_LEN = 26
SEGMENT_EVENT = 68
PROFILE_EVENT = 69
PROFILE_NAME_EVENT = 57
PROFILE_ACTIVATED = 3
SEGMENT_DELETE = 1
PROFILE_CREATED = 0
PROFILE_COPIED = 1
PROFILE_DELETED = 2
SEGMENT_COMPLETE_STATUS = 15
# A profile the archive never saw created shows its slots one edit at a time, so the last slot seen may or may not run
# to midnight. The pump's own log of the basal rate the profile calls for (event 279) settles it: if over the last hour
# of the day the logged rate equals that slot's rate, the slot is the last one and runs to midnight.
LAST_SLOT_EVIDENCE_FROM_MINUTE = 23 * 60
LAST_SLOT_MIN_RECORDS = 12  # one hour of the pump's five-minute records
LAST_SLOT_MIN_AGREEMENT = 0.95
LAST_SLOT_RATE_TOLERANCE = 0.0006  # the pump logs the rate in thousandths of a unit per hour
_TANDEM_EPOCH = dt.datetime(2008, 1, 1)

PROFILE_EVENT_COLUMNS = (
    "pump_serial", "timestamp", "sequence", "event", "profile_id", "status", "source_profile_id", "profile_name",
    "modification", "segment_index", "start_minutes", "basal_units_per_hour", "isf_mgdl_per_unit", "target_bg_mgdl",
    "carb_ratio_g_per_unit",
)
# Columns attached to each carb entry: the profile active at the meal and, when its edits are visible in the
# archive, that profile's segment in force at the meal's time of day.
# ``profile_edits_visible`` is 1 while the archive shows that pump's profile edits and 0 after, when a profile's
# values may have changed unseen. ``profile_segment_start_minutes`` is the start of the segment covering the meal's
# time of day: from the meal's own moment while edits are visible, and otherwise from the profile's segments as
# last seen (a later edit to a start time would not show).
PROFILE_SETTING_COLUMNS = (
    "profile_id", "profile_edits_visible", "profile_carb_ratio_g_per_unit", "profile_isf_mgdl_per_unit", "profile_target_bg_mgdl",
    "profile_segment_start_minutes",
)


def pump_key(value: Any) -> str:
    """The one string form of a pump serial: ``1453346.0``, ``1453346`` and ``" 1453346"`` all become ``1453346``; missing is ``""``."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _word_reversed(payload: np.ndarray) -> np.ndarray:
    """Undo Tandem Source's per-word byte reversal of the 16 payload bytes (rows of shape (n, 16))."""
    return payload.reshape(-1, 4, 4)[:, :, ::-1].reshape(-1, 16)


def _u16le(payload: np.ndarray, offset: int) -> np.ndarray:
    return payload[:, offset].astype(np.int64) | (payload[:, offset + 1].astype(np.int64) << 8)


def _name(chars: Any) -> str:
    raw = bytes(int(c) for c in chars)
    return raw.split(b"\x00")[0].replace(b"\xff", b"").decode("latin1")


def _binary_events(path: Path, pump_serial: str) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith('"'):
        text = json.loads(text)
    data = base64.b64decode(text)
    records = np.frombuffer(data[: len(data) - len(data) % EVENT_LEN], dtype=np.uint8).reshape(-1, EVENT_LEN)
    event_id = ((records[:, 0].astype(np.int64) << 8) | records[:, 1]) & 0x0FFF
    records = records[np.isin(event_id, (SEGMENT_EVENT, PROFILE_EVENT))]
    event_id = ((records[:, 0].astype(np.int64) << 8) | records[:, 1]) & 0x0FFF
    seconds = records[:, 2:6].astype(np.int64) @ np.array([1 << 24, 1 << 16, 1 << 8, 1])
    sequence = records[:, 6:10].astype(np.int64) @ np.array([1 << 24, 1 << 16, 1 << 8, 1])
    payload = _word_reversed(records[:, 10:26])  # now in pumpX2 order: payload[k] is record offset 10 + k
    rows = []
    for k in range(len(records)):
        p = payload[k]
        row: dict[str, Any] = {
            "pump_serial": pump_serial,
            "timestamp": _TANDEM_EPOCH + dt.timedelta(seconds=int(seconds[k])),
            "sequence": int(sequence[k]),
            "event": int(event_id[k]),
            "profile_id": int(p[0]),
            "status": int(p[1]),
        }
        if event_id[k] == SEGMENT_EVENT:
            row |= {
                "segment_index": int(p[2]),
                "modification": int(p[3]),
                "start_minutes": int(_u16le(payload[k : k + 1], 4)[0]),
                "basal_units_per_hour": int(_u16le(payload[k : k + 1], 6)[0]) / 1000.0,
                "isf_mgdl_per_unit": int(_u16le(payload[k : k + 1], 8)[0]),
                "target_bg_mgdl": int(_u16le(payload[k : k + 1], 10)[0]),
                # pumpX2 reads a single byte at offset 24; the archive holds the ratio in mg/U at offsets 22-23
                # (7000 = 7.0 g/U), which reproduces the t:slim's own calculator records.
                "carb_ratio_g_per_unit": int(_u16le(payload[k : k + 1], 12)[0]) / 1000.0,
            }
        else:
            row |= {"source_profile_id": int(p[2]), "profile_name": _name(p[8:16])}
        rows.append(row)
    return rows


def _edit_json_events(path: Path) -> list[dict[str, Any]]:
    """Segment edits (event 68) from ``pump_profile_edits_*.json``, decoded by the server.

    ``carbRatio`` is in mg/U like the binary record, ``basalRate`` in mU/h, and ``status`` lists the set value bits.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    pump_serial = str(payload["pump_serial"])
    rows = []
    for chunk in payload["chunks"]:
        for event in chunk["response"].get("events") or []:
            props = event["eventProperties"]
            rows.append(
                {
                    "pump_serial": pump_serial,
                    "timestamp": pd.Timestamp(event["pumpDateTime"]).to_pydatetime(),
                    "sequence": int(event["sequenceNumber"]),
                    "event": SEGMENT_EVENT,
                    "profile_id": int(props["idp"]),
                    "status": SEGMENT_COMPLETE_STATUS if sorted(props["status"]) == [0, 1, 2, 3] else sum(1 << int(bit) for bit in props["status"]),
                    "segment_index": int(props["segmentIndex"]),
                    "modification": int(props["modificationType"]),
                    "start_minutes": int(props["startTime"]),
                    "basal_units_per_hour": int(props["basalRate"]) / 1000.0,
                    "isf_mgdl_per_unit": int(props["isf"]),
                    "target_bg_mgdl": int(props["targetBg"]),
                    "carb_ratio_g_per_unit": int(props["carbRatio"]) / 1000.0,
                }
            )
    return rows


def _json_events(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pump_serial = str(payload["pump_serial"])
    rows = []
    for chunk in payload["chunks"]:
        for event in chunk["response"].get("events") or []:
            if event["eventCode"] not in (SEGMENT_EVENT, PROFILE_EVENT):
                continue
            if event["eventCode"] == SEGMENT_EVENT:
                # Not returned by the pump-logs endpoint so far; decoding it would need its server field names.
                raise ValueError(f"{path} contains event 68 records, which this decoder does not read from pump-logs JSON yet.")
            props = event["eventProperties"]
            rows.append(
                {
                    "pump_serial": pump_serial,
                    "timestamp": pd.Timestamp(event["pumpDateTime"]).to_pydatetime(),
                    "sequence": int(event["sequenceNumber"]),
                    "event": PROFILE_EVENT,
                    "profile_id": int(props["idp"]),
                    "status": int(props["status"]),
                    "source_profile_id": int(props["sourceIdp"]),
                    "profile_name": _name(props[f"name{i}"] for i in range(8)),
                }
            )
    return rows


def _window_dates(window_dir: Path) -> tuple[dt.date, dt.date]:
    start, end = window_dir.name.split("__", 1)
    return dt.date.fromisoformat(start), dt.date.fromisoformat(end)


def decode_profile_events(tandem_root: str | Path) -> tuple[pd.DataFrame, dict[str, tuple[dt.date, dt.date]]]:
    """All profile events in the archive, and for each pump the span over which its segment edits are visible.

    The visible span is the first unbroken run of windows holding that pump's segment-edit events (binary archive or ``pump_profile_edits`` JSON): before or after it,
    an edit may have happened that the archive does not show.
    """
    tandem_root = Path(tandem_root)
    rows: list[dict[str, Any]] = []
    binary_windows: dict[str, list[tuple[dt.date, dt.date]]] = {}
    window_dirs = sorted(path for path in tandem_root.iterdir() if path.is_dir() and "__" in path.name) if tandem_root.exists() else []
    for window_dir in window_dirs:
        raw = window_dir / "raw"
        for path in sorted(raw.glob("pump_events_raw_*.txt")):
            pump_serial = path.stem.removeprefix("pump_events_raw_").rpartition("_")[0]
            if pump_serial in {"", "unknown"}:
                raise ValueError(f"{path} does not name its pump serial; profile history needs it.")
            events = _binary_events(path, pump_serial)
            rows.extend(events)
            if path.stat().st_size:
                binary_windows.setdefault(pump_serial, []).append(_window_dates(window_dir))
        for path in sorted(raw.glob("pump_logs_*.json")):
            if not path.name.endswith(".meta.json"):
                rows.extend(_json_events(path))
        for path in sorted(raw.glob("pump_profile_edits_*.json")):
            rows.extend(_edit_json_events(path))
            binary_windows.setdefault(str(json.loads(path.read_text(encoding="utf-8"))["pump_serial"]), []).append(_window_dates(window_dir))
    frame = pd.DataFrame(rows, columns=list(PROFILE_EVENT_COLUMNS))
    frame = frame.drop_duplicates(subset=["pump_serial", "sequence"]).sort_values(["pump_serial", "sequence"]).reset_index(drop=True)
    visible: dict[str, tuple[dt.date, dt.date]] = {}
    first_event = frame.groupby("pump_serial")["timestamp"].min()
    for pump_serial, spans in binary_windows.items():
        if pump_serial not in first_event:
            continue  # this pump logged no profile events
        spans.sort()
        start, end = spans[0]
        for span_start, span_end in spans[1:]:
            if span_start > end + dt.timedelta(days=1):
                break
            end = max(end, span_end)
        visible[pump_serial] = (max(start, first_event[pump_serial].date()), end)
    return frame, visible


@dataclass
class ProfileHistory:
    """Replays profile events into the profile active, and its segment in force, at any time on each pump."""

    events: pd.DataFrame
    visible: dict[str, tuple[dt.date, dt.date]]
    profile_rates: pd.DataFrame | None = None  # pump_serial, timestamp, rate (U/h): the basal rate the active profile calls for, logged by the pump
    _tables: dict[tuple[str, int], tuple[list[dt.datetime], list[list[dict[str, float] | None]]]] = field(init=False, default_factory=dict)
    _active: dict[str, tuple[list[dt.datetime], list[int]]] = field(init=False, default_factory=dict)
    _whole: set[tuple[str, int]] = field(init=False, default_factory=set)  # profiles whose whole history the archive holds
    _complete: dict[tuple[str, int], list[bool]] = field(init=False, default_factory=dict)
    _closed: dict[tuple[str, int], list[bool]] = field(init=False, default_factory=dict)  # the table's last slot is known to run to midnight

    def __post_init__(self) -> None:
        actions = self.events[self.events["event"].isin((SEGMENT_EVENT, PROFILE_EVENT))]
        state: dict[tuple[str, int], list[dict[str, float] | None]] = {}
        for row in actions.itertuples(index=False):
            key = (row.pump_serial, int(row.profile_id))
            if int(row.event) == PROFILE_EVENT:
                status = int(row.status)
                if status == PROFILE_CREATED:
                    state[key] = []
                    self._whole.add(key)
                elif status == PROFILE_COPIED:
                    source = (row.pump_serial, int(row.source_profile_id))
                    state[key] = [None if seg is None else dict(seg) for seg in state.get(source, [])]
                    if source in self._whole:
                        self._whole.add(key)
                    else:
                        self._whole.discard(key)
                elif status == PROFILE_DELETED:
                    state.pop(key, None)
                    self._whole.discard(key)
                else:
                    continue
            else:
                table = state.setdefault(key, [])
                index = int(row.segment_index)
                values = {
                    "start_minutes": int(row.start_minutes),
                    "carb_ratio_g_per_unit": float(row.carb_ratio_g_per_unit),
                    "isf_mgdl_per_unit": float(row.isf_mgdl_per_unit),
                    "target_bg_mgdl": float(row.target_bg_mgdl),
                    "basal_units_per_hour": float(row.basal_units_per_hour),
                }
                complete = int(row.status) == SEGMENT_COMPLETE_STATUS
                if int(row.modification) == SEGMENT_DELETE:
                    if index < len(table):
                        table.pop(index)
                    elif key in self._whole:
                        raise ValueError(f"Profile history deletes segment {index} of profile {key} at {row.timestamp}, which it never saw.")
                    # otherwise the profile predates the archive: the slot was never seen and the table stays partly unknown
                else:
                    table.extend([None] * (index + 1 - len(table)))
                    table[index] = values if complete else None
            times, tables = self._tables.setdefault(key, ([], []))
            times.append(row.timestamp)
            tables.append(list(state.get(key, [])))
            wholes = self._complete.setdefault(key, [])
            wholes.append(key in self._whole)
        activations = self.events[self.events["event"].eq(PROFILE_EVENT) & self.events["status"].eq(PROFILE_ACTIVATED)]
        for pump_serial, group in activations.sort_values("timestamp").groupby("pump_serial"):
            self._active[pump_serial] = (list(group["timestamp"]), [int(v) for v in group["profile_id"]])
        self._closed = {key: list(flags) for key, flags in self._complete.items()}
        if self.profile_rates is not None and not self.profile_rates.empty:
            self._confirm_last_slots(self.profile_rates)

    def _confirm_last_slots(self, rates: pd.DataFrame) -> None:
        """Mark each stretch of a profile's history whose last slot the pump's own rate log shows running to midnight.

        A stretch is a run of snapshots in which the last slot keeps its start, rate and index: edits elsewhere in the
        profile do not end it, and a slot added or removed after it does. The pump logs the rate the active profile
        calls for about every five minutes. Over the last hour of the day, records at or after the last slot's start
        whose rate equals that slot's rate confirm it; the whole stretch then counts as closed. A stretch without
        enough records, or with records that disagree, stays unknown.
        """
        for pump, group in rates.groupby("pump_serial"):
            if pump not in self._active:
                continue
            stamps = pd.DatetimeIndex(group["timestamp"])
            times = stamps.to_numpy(dtype="datetime64[ns]")
            minutes = np.asarray(stamps.hour * 60 + stamps.minute)
            logged = group["rate"].to_numpy(dtype=float)
            activations, profiles = self._active[pump]
            position = np.searchsorted(np.asarray(activations, dtype="datetime64[ns]"), times, side="right") - 1
            active = np.where(position >= 0, np.asarray(profiles)[np.maximum(position, 0)], -1)
            late = (position >= 0) & (minutes >= LAST_SLOT_EVIDENCE_FROM_MINUTE) & np.isfinite(logged)
            for profile in np.unique(active[late]):
                key = (pump, int(profile))
                if key not in self._tables:
                    continue
                edits, tables = self._tables[key]
                last = [table[-1] if table and table[-1] is not None else None for table in tables]
                run = np.zeros(len(tables), dtype=int)
                for k in range(1, len(tables)):
                    same = (last[k] is not None and last[k - 1] is not None and len(tables[k]) == len(tables[k - 1])
                            and last[k]["start_minutes"] == last[k - 1]["start_minutes"]
                            and last[k]["basal_units_per_hour"] == last[k - 1]["basal_units_per_hour"])
                    run[k] = run[k - 1] + (0 if same else 1)
                chosen = np.flatnonzero(late & (active == profile))
                version = np.searchsorted(np.asarray(edits, dtype="datetime64[ns]"), times[chosen], side="right") - 1
                chosen, version = chosen[version >= 0], version[version >= 0]
                has_last = np.array([last[k] is not None for k in version], dtype=bool)
                chosen, version = chosen[has_last], version[has_last]
                start = np.array([last[k]["start_minutes"] for k in version])
                expected = np.array([last[k]["basal_units_per_hour"] for k in version])
                covered = minutes[chosen] >= start
                chosen, version, expected = chosen[covered], version[covered], expected[covered]
                agrees = np.abs(logged[chosen] - expected) <= LAST_SLOT_RATE_TOLERANCE
                for stretch in np.unique(run[version]):
                    inside = run[version] == stretch
                    if inside.sum() >= LAST_SLOT_MIN_RECORDS and agrees[inside].mean() >= LAST_SLOT_MIN_AGREEMENT and last[int(np.flatnonzero(run == stretch)[0])] is not None:
                        for k in np.flatnonzero(run == stretch):
                            self._closed[key][k] = True

    def active_profile(self, pump_serial: str, timestamp: dt.datetime) -> int | None:
        times, profiles = self._active.get(pump_serial, ([], []))
        k = bisect.bisect_right(times, timestamp) - 1
        return profiles[k] if k >= 0 else None

    def segment_in_force(self, pump_serial: str, profile_id: int, timestamp: dt.datetime, *, minute_of_day: int | None = None) -> dict[str, float] | None:
        """The profile's segment covering a time of day, as the profile stood at ``timestamp``, or None when the archive cannot show it.

        The time of day is ``timestamp``'s unless ``minute_of_day`` is given, which asks about another hour of the day
        under the profile as it stood at ``timestamp`` (the table now, for example, at each hour).
        """
        span = self.visible.get(pump_serial)
        if span is None or not span[0] <= timestamp.date() <= span[1]:
            return None
        times, tables = self._tables.get((pump_serial, profile_id), ([], []))
        k = bisect.bisect_right(times, timestamp) - 1
        if k < 0:
            return None
        table = tables[k]
        minute = timestamp.hour * 60 + timestamp.minute if minute_of_day is None else minute_of_day
        covering = None
        for index, segment in enumerate(table):
            if segment is not None and segment["start_minutes"] <= minute and (covering is None or segment["start_minutes"] > table[covering]["start_minutes"]):
                covering = index
        # A slot not yet seen could start before this minute, so the covering segment is unknown; and when the
        # archive never saw the profile created, a slot after the last one seen may exist too.
        if covering is None or (covering + 1 < len(table) and table[covering + 1] is None):
            return None
        if covering + 1 == len(table) and not self._closed[(pump_serial, profile_id)][k]:
            return None
        return table[covering]

    def values_at(self, pumps: pd.Series, timestamps: pd.Series, field: str) -> np.ndarray:
        """The value of ``field`` (a profile segment value such as ``basal_units_per_hour``, ``carb_ratio_g_per_unit``,
        ``isf_mgdl_per_unit``, ``target_bg_mgdl`` or ``start_minutes``) the profile in force holds at each time, for the
        pump named at the same position.

        Vectorized exact-time replay: an edit applies from its own timestamp, a time outside the span where the pump's
        edits are visible, before the first activation, or inside a slot the archive never saw is NaN.
        """
        stamps = pd.DatetimeIndex(timestamps)
        times = stamps.to_numpy(dtype="datetime64[ns]")
        minutes = np.asarray(stamps.hour * 60 + stamps.minute)
        serials = pumps.map(pump_key).to_numpy()
        rates = np.full(len(times), np.nan)
        for pump, (activations, profiles) in self._active.items():
            if pump not in self.visible:
                continue
            first, last = self.visible[pump]
            selected = np.flatnonzero((serials == pump) & (times >= np.datetime64(first)) &
                                      (times < np.datetime64(last + dt.timedelta(days=1))))
            activation = np.searchsorted(np.asarray(activations, dtype="datetime64[ns]"), times[selected], side="right") - 1
            selected = selected[activation >= 0]
            active = np.asarray(profiles)[activation[activation >= 0]]
            for profile in np.unique(active):
                key = (pump, profile)
                if key not in self._tables:
                    continue
                edits, snapshots = self._tables[key]
                idx = selected[active == profile]
                version = np.searchsorted(np.asarray(edits, dtype="datetime64[ns]"), times[idx], side="right") - 1
                for k in np.unique(version[version >= 0]):
                    positions = idx[version == k]
                    table = snapshots[k]
                    starts = sorted((seg["start_minutes"], i) for i, seg in enumerate(table) if seg is not None)
                    if not starts:
                        continue
                    bounds, slots = zip(*starts)
                    values = [table[i][field] if
                              (i + 1 < len(table) and table[i + 1] is not None or
                               i + 1 == len(table) and self._closed[key][k]) else np.nan for i in slots]
                    covering = np.searchsorted(bounds, minutes[positions], side="right") - 1
                    valid = covering >= 0
                    rates[positions[valid]] = np.asarray(values)[covering[valid]]
        return rates

    def settings_for(self, carbs: pd.DataFrame) -> pd.DataFrame:
        """``PROFILE_SETTING_COLUMNS`` for each carb entry (by ``pump_serial`` and ``timestamp``)."""
        out = pd.DataFrame(index=carbs.index, columns=list(PROFILE_SETTING_COLUMNS), dtype=float)
        if carbs.empty or "pump_serial" not in carbs.columns:
            return out  # carbs from hand-kept logs belong to no pump
        serials = carbs["pump_serial"].map(pump_key)
        # The profile is read at the moment the pump recorded (``event_time``), not at its 5-minute grid slot, so an
        # edit made inside the slot applies to the meals after it and not to those before.
        moments = pd.to_datetime(carbs["timestamp"])
        if "event_time" in carbs:
            moments = pd.to_datetime(carbs["event_time"]).fillna(moments)
        for index, pump_serial, timestamp in zip(carbs.index, serials, moments):
            if not pump_serial or pd.isna(timestamp):
                continue
            moment = timestamp.to_pydatetime()
            profile_id = self.active_profile(pump_serial, moment)
            if profile_id is None:
                continue
            out.at[index, "profile_id"] = profile_id
            span = self.visible.get(pump_serial)
            out.at[index, "profile_edits_visible"] = float(span is not None and span[0] <= moment.date() <= span[1])
            segment = self.segment_in_force(pump_serial, profile_id, moment)
            last_seen = self.visible[pump_serial][1] if pump_serial in self.visible else None
            if segment is None and last_seen is not None and moment.date() > last_seen:
                earlier = self.segment_in_force(pump_serial, profile_id, dt.datetime.combine(last_seen, moment.time()))
                if earlier is not None:
                    out.at[index, "profile_segment_start_minutes"] = earlier["start_minutes"]
            if segment is not None:
                out.at[index, "profile_segment_start_minutes"] = segment["start_minutes"]
                out.at[index, "profile_carb_ratio_g_per_unit"] = segment["carb_ratio_g_per_unit"]
                out.at[index, "profile_isf_mgdl_per_unit"] = segment["isf_mgdl_per_unit"]
                out.at[index, "profile_target_bg_mgdl"] = segment["target_bg_mgdl"]
        return out
